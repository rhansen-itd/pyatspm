"""
ATSPM Coordination / Split Diagram Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: cycles DataFrame, signal events DataFrame, optional detector DataFrame,
       metadata dict, and detector config series/dict.
Output: plotly.graph_objects.Figure (1 or 2 ring subplots).

Package Location: src/atspm/plotting/coordination.py

Ring Subplot Rule:
    A subplot for Ring 2 is only created when ``r2_phases`` in df_cycles
    contains real phase numbers (not the literal string "None" and not empty).

Dummy Legend Rule:
    One invisible scatter trace per unique phase colour is added at the end
    so each phase appears only once in the legend regardless of how many
    cycle bars it generates.  Bar traces themselves carry showlegend=False.

Gap Marker Rule:
    df_signal rows with Code == -1 (event_code == -1) are excluded before
    all duration and stacking calculations.

Arrival Offset Slider:
    Rendered only when arrival detector traces exist AND the number of
    cycles in the data is < MAX_CYCLES_FOR_SLIDER (2000 by default).
    Slider steps run from -30 s to +30 s in 1-second increments; active
    step initialised at 0 offset (index 30 in the step list).

Example SQL to build the three input DataFrames
------------------------------------------------
    -- df_cycles (from cycles table):
    SELECT
        datetime(cycle_start, 'unixepoch', 'localtime') AS Cycle_start,
        coord_plan                                        AS Coord_plan,
        r1_phases                                         AS r1_phases,
        r2_phases                                         AS r2_phases
    FROM cycles
    WHERE cycle_start BETWEEN :start_ts AND :end_ts
    ORDER BY cycle_start;

    -- df_signal (from events table – signal state codes):
    SELECT
        datetime(e.timestamp, 'unixepoch', 'localtime')   AS TS_start,
        e.event_code                                       AS Code,
        e.parameter                                        AS ID,
        datetime(c.cycle_start, 'unixepoch', 'localtime') AS Cycle_start
    FROM events e
    JOIN cycles c
      ON e.timestamp >= c.cycle_start
     AND (e.timestamp < (
             SELECT MIN(c2.cycle_start) FROM cycles c2
             WHERE c2.cycle_start > c.cycle_start
         ) OR (SELECT MIN(c2.cycle_start) FROM cycles c2 WHERE c2.cycle_start > c.cycle_start) IS NULL)
    WHERE e.event_code IN (-1, 1, 8, 9, 11, 12)
      AND e.timestamp BETWEEN :start_ts AND :end_ts
    ORDER BY e.timestamp;

    -- df_det (optional – detector on/off events):
    SELECT
        datetime(e.timestamp, 'unixepoch', 'localtime')   AS TS_start,
        e.event_code                                       AS Code,
        e.parameter                                        AS ID,
        datetime(c.cycle_start, 'unixepoch', 'localtime') AS Cycle_start
    FROM events e
    JOIN cycles c  -- same join as above
      ON ...
    WHERE e.event_code IN (81, 82)
      AND e.timestamp BETWEEN :start_ts AND :end_ts
    ORDER BY e.timestamp;
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1

# Maximum number of cycles before arrival offset slider is suppressed
MAX_CYCLES_FOR_SLIDER: int = 2000

# Phase colour mapping (matches legacy cl_P dict in plotting.py)
_PHASE_COLORS: Dict[str, str] = {
    '1':  'DarkOrange',    '5':  'LightSalmon',
    '3':  'Magenta',       '7':  'Purple',
    '2':  'Blue',          '6':  'Turquoise',
    '4':  'DarkGreen',     '8':  'Lime',
    '9':  'DarkOliveGreen','10': 'Olive',
    '11': 'DarkSlateGray', '12': 'Gray',
    '13': 'LightGray',     '14': 'Silver',
    '15': 'DimGray',       '16': 'Black',
}

# GYR override colours (applied when code is not green)
_GYR_COLORS: Dict[str, str] = {
    'Rc': 'Red',
    'Y':  'Yellow',
}

# Detector type rendering config
_DET_CONFIG: Dict[str, Dict[str, Any]] = {
    'Ar': {'color': 'DimGray',  'x_shift': -10, 'label': 'Arrival'},
    'Oc': {'color': 'Black',    'x_shift':   0, 'label': 'Occupancy'},
    'St': {'color': 'Crimson',  'x_shift':  10, 'label': 'Stop Bar'},
}

# ATSPM signal-state code → GYR label
_CODE_TO_GYR: Dict[int, str] = {
    1:  'G',
    8:  'Y',
    9:  'Rc',
    11: 'R',
    12: 'R',
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_coordination(
    df_cycles: pd.DataFrame,
    df_signal: pd.DataFrame,
    metadata: Dict[str, Any],
    df_det: Optional[pd.DataFrame] = None,
    det_config: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Build a stacked-bar Coordination / Split Diagram.

    One subplot per ring is generated; the Ring 2 subplot is omitted when
    ``r2_phases`` contains only ``"None"`` or empty strings.

    Args:
        df_cycles: Cycle-level DataFrame with columns::

            Cycle_start : datetime-like
            Coord_plan  : int/float, coordination plan number
            r1_phases   : str, comma-separated phase numbers for Ring 1
                          e.g. "2,6" or "None"
            r2_phases   : str, same format for Ring 2

        df_signal: Signal-event DataFrame with columns::

            TS_start    : datetime-like
            Code        : int (1=Green, 8=Yellow, 9=RedClearance, 11/12=Red)
            ID          : int, phase number
            Cycle_start : datetime-like
            Duration    : float, seconds in this state (optional; computed
                          if absent using within-phase shift logic)

        metadata: Dict with ``major_road_name``, ``minor_road_name``,
            ``intersection_name``.
        df_det: Optional detector events DataFrame with columns::

            TS_start    : datetime-like
            Code        : int (82=On, 81=Off)
            ID          : int, detector number
            Cycle_start : datetime-like
            t_cs        : float, time since cycle start (seconds)
            Duration    : float, actuation duration (seconds)

        det_config: Optional dict (or pandas Series) mapping detector-type
            keys (e.g. ``"P2 Arrival"``, ``"P6 Stop Bar"``) to
            comma-separated detector number strings (e.g. ``"14,15"``).
            Keys are expected to match the legacy ``int_cfg`` format:
            the first character is ``'P'``, the second is the phase number,
            and the rest describes detector type (``Arrival``, ``Stop Bar``,
            ``Occupancy``).

    Returns:
        ``plotly.graph_objects.Figure`` with 1 or 2 vertically stacked
        subplots.
    """
    _validate_signal_columns(df_signal)

    title = _build_title(metadata, suffix='Coordination / Split Diagram')

    # Exclude gap marker rows from signal data
    df_signal = df_signal[df_signal['Code'] != _GAP_CODE].copy()

    # Ensure Duration exists
    if 'Duration' not in df_signal.columns:
        df_signal = _compute_durations(df_signal)

    # Filter to GYR codes only (drop Red/idle to avoid double-counting)
    df_signal = df_signal[df_signal['Code'].isin([1, 8, 9])].copy()
    df_signal['ID'] = pd.to_numeric(df_signal['ID'], errors='coerce').astype('Int64')
    df_signal = df_signal.dropna(subset=['ID', 'Duration'])

    # Compute t_start: seconds from Cycle_start to TS_start for each event.
    # This is the floating-bar base offset – the vertical start position on
    # the 'seconds into cycle' y-axis.  Must be computed before ring bar
    # calls.  Handles both tz-aware Timestamps and float epoch columns.
    df_signal = _compute_t_start(df_signal)
    df_signal = df_signal.dropna(subset=['t_start'])

    # Parse ring membership from cycles DataFrame
    r1_phases_all, r2_phases_all = _collect_ring_phases(df_cycles)

    has_r2 = bool(r2_phases_all)

    n_rows = 2 if has_r2 else 1
    subplot_titles = [f'Ring 1']
    if has_r2:
        subplot_titles.append('Ring 2')

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    # Track which arrival traces are added (for the offset slider)
    arrival_traces: List[Dict[str, Any]] = []
    # Track dummy-legend phases already added (per ring, keyed by phase str)
    dummy_legend_added: set = set()

    # -----------------------------------------------------------------------
    # Ring 1 bars  (always row 1)
    # -----------------------------------------------------------------------
    _add_ring_bars(
        fig=fig,
        row=1,
        ring_phases=r1_phases_all,
        df_cycles=df_cycles,
        df_signal=df_signal,
        ring_col='r1_phases',
        dummy_legend_added=dummy_legend_added,
    )

    # -----------------------------------------------------------------------
    # Ring 2 bars  (row 2 only when real phases exist)
    # -----------------------------------------------------------------------
    if has_r2:
        _add_ring_bars(
            fig=fig,
            row=2,
            ring_phases=r2_phases_all,
            df_cycles=df_cycles,
            df_signal=df_signal,
            ring_col='r2_phases',
            dummy_legend_added=dummy_legend_added,
        )

    # -----------------------------------------------------------------------
    # Dummy legend traces (one per phase colour – appear only once)
    # -----------------------------------------------------------------------
    _add_dummy_legends(fig, dummy_legend_added)

    # -----------------------------------------------------------------------
    # Coord-plan markers (Cycle_start × y=-5 tick, coloured by plan)
    # -----------------------------------------------------------------------
    _add_coord_plan_markers(fig, df_cycles, row=1)

    # -----------------------------------------------------------------------
    # Detector scatter traces
    # -----------------------------------------------------------------------
    if df_det is not None and det_config is not None and not df_det.empty:
        arrival_traces = _add_detector_traces(
            fig=fig,
            df_det=df_det,
            det_config=det_config,
            ring1_phases=r1_phases_all,
            ring2_phases=r2_phases_all if has_r2 else set(),
            n_rows=n_rows,
        )

    # -----------------------------------------------------------------------
    # Arrival offset slider (only when arrivals exist and data is manageable)
    # -----------------------------------------------------------------------
    n_cycles = len(df_cycles)
    if arrival_traces and n_cycles < MAX_CYCLES_FOR_SLIDER:
        _add_arrival_slider(fig, arrival_traces)

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        barmode='overlay',  # floating bars use 'base'; 'stack' would add offsets
        showlegend=True,
        legend=dict(
            orientation='v',
            x=1.01,
            y=1.0,
            xanchor='left',
        ),
        hovermode='x unified',
        template='plotly_white',
        height=400 * n_rows + 80,
    )

    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text='Seconds', row=i, col=1)

    fig.update_xaxes(title_text='Cycle Start', row=n_rows, col=1)

    return fig


# ---------------------------------------------------------------------------
# Ring / phase resolution helpers
# ---------------------------------------------------------------------------

def _collect_ring_phases(df_cycles: pd.DataFrame) -> Tuple[set, set]:
    """
    Collect the full set of real phase IDs across all cycles for each ring.

    ``"None"`` strings and empty values are ignored.

    Args:
        df_cycles: Cycles DataFrame with ``r1_phases`` and ``r2_phases``
            columns.

    Returns:
        Tuple of (``r1_set``, ``r2_set``) – sets of integer phase IDs.
    """
    def _parse_col(col_name: str) -> set:
        if col_name not in df_cycles.columns:
            return set()
        phases: set = set()
        for cell in df_cycles[col_name].dropna().unique():
            for tok in str(cell).split(','):
                tok = tok.strip()
                if tok.isdigit():
                    phases.add(int(tok))
        return phases

    return _parse_col('r1_phases'), _parse_col('r2_phases')


def _phase_color(phase_id: int, gyr: str) -> str:
    """
    Return the display colour for a given phase and signal state.

    Green uses the phase-specific colour from ``_PHASE_COLORS``; yellow and
    red clearance use the shared ``_GYR_COLORS`` palette.

    Args:
        phase_id: Integer phase number.
        gyr: Signal state label – ``'G'``, ``'Y'``, or ``'Rc'``.

    Returns:
        CSS colour string.
    """
    if gyr == 'G':
        return _PHASE_COLORS.get(str(phase_id), 'Gray')
    return _GYR_COLORS.get(gyr, 'Gray')


# ---------------------------------------------------------------------------
# Bar-chart construction
# ---------------------------------------------------------------------------

def _modal_phase_sequence(
    df_cycles: pd.DataFrame,
    ring_col: str,
    ring_phases: set,
) -> dict:
    """
    Build a phase → legend-display-position mapping from ring-phase strings.

    Used only to determine the order in which dummy legend entries and Plotly
    trace layers are rendered (cosmetic).  Actual bar vertical positions are
    always determined by ``t_start``, not this mapping.

    Prefers non-reservice sequences (each phase appears once) for a stable
    display order.  Falls back to the raw modal string if all sequences
    contain reservice.

    Args:
        df_cycles: Cycles DataFrame containing the ring column.
        ring_col: Column name (``'r1_phases'`` or ``'r2_phases'``).
        ring_phases: Set of integer phase IDs in this ring.

    Returns:
        Dict mapping integer phase_id → integer display position.
        Phases absent from the modal sequence get position 999 (sorted last).
    """
    if ring_col not in df_cycles.columns:
        return {p: i for i, p in enumerate(sorted(ring_phases))}

    seq_series = (
        df_cycles[ring_col]
        .dropna()
        .pipe(lambda s: s[s != 'None'])
    )
    if seq_series.empty:
        return {p: i for i, p in enumerate(sorted(ring_phases))}

    # Prefer sequences without repeated phase numbers (no reservice)
    clean = seq_series[~seq_series.str.contains(r'(\d+),.*?\1', regex=True, na=False)]
    modal_seq = (clean if not clean.empty else seq_series).mode().iloc[0]

    order: dict = {}
    seen: set = set()
    for pos, tok in enumerate(modal_seq.split(',')):
        tok = tok.strip()
        if tok.isdigit():
            ph = int(tok)
            if ph not in seen:
                order[ph] = pos
                seen.add(ph)
    return order


def _add_ring_bars(
    fig: go.Figure,
    row: int,
    ring_phases: set,
    df_cycles: pd.DataFrame,
    df_signal: pd.DataFrame,
    ring_col: str,
    dummy_legend_added: set,
) -> None:
    """
    Add floating-bar traces for every individual signal interval in one ring.

    Each bar is positioned using Plotly's ``base`` parameter:

    * ``base`` = ``t_start`` (seconds from ``Cycle_start`` to interval start)
    * ``y``    = ``Duration`` (height of the bar in seconds)

    This means:

    * **Correct sequence**: bars sit at their actual temporal position within
      the cycle; ordering is determined by when each interval starts (t_start),
      not by phase number.
    * **Reservice support**: a phase served twice in one cycle (e.g. r1_phases
      stores ``"2,2,6"``) produces two separate bars at their respective
      ``t_start`` positions.  No aggregation is performed.

    One ``go.Bar`` trace per unique ``(phase_id, gyr, seq_pos)`` combination
    keeps the trace count bounded.  ``seq_pos`` distinguishes first service
    from reservice within the same cycle.

    ``barmode='overlay'`` must be set on the layout (done by
    ``plot_coordination``).

    Args:
        fig: Figure to mutate.
        row: Subplot row index (1-based).
        ring_phases: Set of integer phase IDs in this ring.
        df_cycles: Cycles DataFrame with ``Cycle_start`` and the ring-phase
            column (``r1_phases`` or ``r2_phases``).
        df_signal: Signal events DataFrame with columns
            ``[TS_start, Code, ID, Cycle_start, Duration, t_start]``.
            ``t_start`` must already be present (seconds from ``Cycle_start``).
        ring_col: Column in ``df_cycles`` giving the per-cycle phase-sequence
            string (``'r1_phases'`` or ``'r2_phases'``).
        dummy_legend_added: Mutable set; updated with phase IDs that have
            received a dummy legend trace (Green state, seq_pos==0 only).
    """
    if not ring_phases or df_cycles.empty or df_signal.empty:
        return

    sig = df_signal[df_signal['ID'].isin(ring_phases)].copy()
    if sig.empty:
        return

    sig['gyr'] = sig['Code'].map(_CODE_TO_GYR)
    sig = sig[sig['gyr'].isin(['G', 'Y', 'Rc'])].dropna(subset=['Duration', 't_start'])
    if sig.empty:
        return

    # -----------------------------------------------------------------------
    # Assign within-cycle sequence position to each interval.
    #
    # Reservice: a phase going green twice in one cycle (e.g. "2,2,6" in the
    # r1_phases string) means Code 1 for phase 2 appears twice in that cycle.
    # cumcount() within (Cycle_start, ID) on green rows gives position 0 for
    # the first service and 1 for the reservice.  Yellow / RedClearance rows
    # inherit the position of their preceding green onset via ffill.
    # -----------------------------------------------------------------------
    sig = sig.sort_values(['Cycle_start', 'ID', 'TS_start'])

    green_mask = sig['Code'] == 1
    sig['_green_count'] = (
        sig.loc[green_mask]
        .groupby(['Cycle_start', 'ID'])
        .cumcount()
    )
    sig['_seq_pos'] = (
        sig.groupby(['Cycle_start', 'ID'])['_green_count']
        .transform(lambda x: x.ffill())
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------------------------
    # Determine display order (cosmetic only).
    # -----------------------------------------------------------------------
    display_order = _modal_phase_sequence(df_cycles, ring_col, ring_phases)
    gyr_order = {'G': 0, 'Y': 1, 'Rc': 2}

    combos = (
        sig[['ID', 'gyr', '_seq_pos']]
        .drop_duplicates()
        .copy()
    )
    combos['_disp']    = combos['ID'].map(lambda p: display_order.get(p, 999))
    combos['_gyr_ord'] = combos['gyr'].map(gyr_order)
    combos = combos.sort_values(['_disp', '_seq_pos', '_gyr_ord'])

    for _, combo_row in combos.iterrows():
        phase_id = int(combo_row['ID'])
        gyr      = combo_row['gyr']
        seq_pos  = int(combo_row['_seq_pos'])
        color    = _phase_color(phase_id, gyr)

        mask = (
            (sig['ID']       == phase_id) &
            (sig['gyr']      == gyr)      &
            (sig['_seq_pos'] == seq_pos)
        )
        sub = sig[mask].dropna(subset=['Duration', 't_start'])
        if sub.empty:
            continue

        if gyr == 'G' and seq_pos == 0:
            dummy_legend_added.add(phase_id)

        svc_label = ' (reservice)' if seq_pos > 0 else ''

        fig.add_trace(
            go.Bar(
                x=sub['Cycle_start'],
                y=sub['Duration'],
                width=30000,  # 30-second default bar width (in ms on datetime x-axis)
                base=sub['t_start'],
                name=f'Ph {phase_id}',
                marker_color=color,
                showlegend=False,
                legendgroup=f'Ph {phase_id}',
                hovertemplate=(
                    f"Ph {phase_id} {gyr}{svc_label}<br>"
                    "Start: %{base:.1f} s<br>"
                    "Duration: %{y:.1f} s<br>"
                    "Cycle: %{x}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def _add_dummy_legends(fig: go.Figure, phase_ids: set) -> None:
    """
    Add one invisible scatter trace per phase so each phase colour appears
    exactly once in the legend.

    The traces use ``mode='markers'``, ``visible=True``, and
    ``showlegend=True`` but carry zero data points so they contribute
    nothing to the axes.

    Args:
        fig: Figure to mutate.
        phase_ids: Set of integer phase IDs that need legend entries.
    """
    for phase_id in sorted(phase_ids):
        color = _PHASE_COLORS.get(str(phase_id), 'Gray')
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(color=color, size=10, symbol='square'),
            name=f'Ph {phase_id}',
            legendgroup=f'Ph {phase_id}',
            showlegend=True,
        ))


def _add_coord_plan_markers(
    fig: go.Figure,
    df_cycles: pd.DataFrame,
    row: int = 1,
) -> None:
    """
    Add a tiny tick scatter at y=-5 per cycle, coloured by coordination plan.

    Provides a visual bottom strip indicating which coord plan is active,
    matching the legacy ``px.scatter`` on ``Coord_plan``.

    Args:
        fig: Figure to mutate.
        df_cycles: Cycles DataFrame with ``Cycle_start`` and ``Coord_plan``.
        row: Row in which to add the markers.
    """
    if 'Coord_plan' not in df_cycles.columns or df_cycles.empty:
        return

    df_plt = df_cycles[['Cycle_start', 'Coord_plan']].dropna().copy()

    fig.add_trace(
        go.Scatter(
            x=df_plt['Cycle_start'],
            y=[-5] * len(df_plt),
            mode='markers',
            marker=dict(
                symbol='line-ew',
                color=df_plt['Coord_plan'].astype(float),
                colorscale='Viridis',
                size=8,
                line=dict(width=2, color=df_plt['Coord_plan'].astype(float)),
                showscale=False,
            ),
            text=df_plt['Coord_plan'].astype(str),
            name='Coord Plan',
            showlegend=False,
            hovertemplate='Plan: %{text}<br>Cycle: %{x}<extra></extra>',
        ),
        row=row,
        col=1,
    )


# ---------------------------------------------------------------------------
# Detector helpers
# ---------------------------------------------------------------------------

def _add_detector_traces(
    fig: go.Figure,
    df_det: pd.DataFrame,
    det_config: Dict[str, Any],
    ring1_phases: set,
    ring2_phases: set,
    n_rows: int,
) -> List[Dict[str, Any]]:
    """
    Add detector scatter traces to the appropriate ring subplot.

    Each detector type (Arrival, Stop Bar, Occupancy) gets a distinct colour
    and an initial x-shift applied to its ``t_cs`` (time-in-cycle) y-values.
    Traces are added as ``visible='legendonly'`` so they do not clutter the
    default view.

    Args:
        fig: Figure to mutate.
        df_det: Detector events DataFrame (Code 82 = actuation start).
            Must have columns ``[TS_start, Code, ID, Cycle_start, t_cs, Duration]``.
        det_config: Mapping of detector-type keys to detector number strings.
            Expected key format (legacy): ``"P{phase} Arrival"``,
            ``"P{phase} Stop Bar"``, ``"P{phase} Occupancy"``.
        ring1_phases: Phase IDs in Ring 1.
        ring2_phases: Phase IDs in Ring 2.
        n_rows: Total subplot row count.

    Returns:
        List of arrival-trace descriptor dicts
        ``{"trace_index": int, "orig_y": np.ndarray}``
        for use by the offset slider builder.
    """
    arrival_traces: List[Dict[str, Any]] = []

    # Only actuation-start events are plotted
    df_on = df_det[df_det['Code'] == 82].copy()
    if df_on.empty:
        return arrival_traces

    # Normalise det_config to a plain dict
    if hasattr(det_config, 'items'):
        cfg_items = list(det_config.items())
    else:
        cfg_items = list(det_config)

    for key, det_str in cfg_items:
        if not det_str or pd.isna(det_str):
            continue
        det_str = str(det_str).strip()
        if not det_str:
            continue

        # Parse phase number and detector type from key
        # e.g. "P2 Arrival" → phase=2, dtype="Ar"
        phase_num, dtype_key = _parse_det_key(key)
        if phase_num is None or dtype_key not in _DET_CONFIG:
            continue

        dcfg = _DET_CONFIG[dtype_key]
        x_shift = dcfg['x_shift']
        color = dcfg['color']

        # Determine subplot row from phase ring membership
        if phase_num in ring2_phases and n_rows == 2:
            row = 2
        else:
            row = 1  # default to ring 1 if ambiguous

        for det_id_str in det_str.split(','):
            det_id_str = det_id_str.strip()
            if not det_id_str.isdigit():
                continue
            det_id = int(det_id_str)

            df_d = df_on[df_on['ID'] == det_id].copy()
            if df_d.empty:
                continue

            # Apply x_shift to t_cs (time-in-cycle y-axis value)
            if 't_cs' in df_d.columns:
                orig_y = df_d['t_cs'].values.copy()
                shifted_y = orig_y + x_shift
            else:
                orig_y = np.zeros(len(df_d))
                shifted_y = orig_y + x_shift

            trace_label = f'{dtype_key[:2]} (Det {det_id})'
            trace = go.Scatter(
                x=df_d['Cycle_start'],
                y=shifted_y,
                mode='markers',
                marker=dict(color=color, size=5, opacity=0.75),
                name=trace_label,
                legendgroup=trace_label,
                showlegend=True,
                visible='legendonly',
                hovertemplate=(
                    f"Det {det_id} ({dcfg['label']})<br>"
                    "t_cs: %{y:.1f} s<br>"
                    "Cycle: %{x}<extra></extra>"
                ),
            )
            fig.add_trace(trace, row=row, col=1)

            if dtype_key == 'Ar':
                arrival_traces.append({
                    'trace_index': len(fig.data) - 1,
                    'orig_y': orig_y,
                })

    return arrival_traces


def _parse_det_key(key: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse a legacy detector config key into (phase_number, type_key).

    Expected format: ``"P{phase} {TypeLabel}"`` e.g. ``"P2 Arrival"``.

    Args:
        key: Detector config key string.

    Returns:
        Tuple of (phase_int, dtype_key) where dtype_key is one of
        ``'Ar'``, ``'St'``, ``'Oc'``, or ``(None, None)`` on parse failure.
    """
    key = str(key).strip()
    parts = key.split()
    if len(parts) < 2:
        return None, None

    phase_part = parts[0]
    if not phase_part.startswith('P') or not phase_part[1:].isdigit():
        return None, None
    phase_num = int(phase_part[1:])

    type_label = ' '.join(parts[1:]).lower()
    if 'arrival' in type_label:
        return phase_num, 'Ar'
    if 'stop' in type_label:
        return phase_num, 'St'
    if 'occupancy' in type_label or 'occ' in type_label:
        return phase_num, 'Oc'

    return phase_num, None


# ---------------------------------------------------------------------------
# Arrival offset slider
# ---------------------------------------------------------------------------

def _add_arrival_slider(
    fig: go.Figure,
    arrival_traces: List[Dict[str, Any]],
) -> None:
    """
    Attach a Plotly slider widget that shifts arrival trace y-values.

    Steps run from -30 s to +30 s in 1-second increments.  The slider
    active index is 30, corresponding to an offset of 0.

    Args:
        fig: Figure to mutate.
        arrival_traces: List of dicts with ``trace_index`` (int) and
            ``orig_y`` (numpy array) for each arrival trace.
    """
    steps: List[Dict[str, Any]] = []

    for offset in range(-30, 31, 1):
        updated_y = [tr['orig_y'] + offset for tr in arrival_traces]
        trace_indices = [tr['trace_index'] for tr in arrival_traces]
        step = {
            'label': str(offset),
            'method': 'restyle',
            'args': [
                {'y': updated_y},
                trace_indices,
            ],
        }
        steps.append(step)

    fig.update_layout(
        sliders=[{
            'active': 30,  # index 30 → offset 0
            'steps': steps,
            'currentvalue': {'prefix': 'Arrival Offset (sec): '},
            'pad': {'t': 50},
        }]
    )


# ---------------------------------------------------------------------------
# Signal duration computation (fallback)
# ---------------------------------------------------------------------------

def _compute_t_start(df_signal: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ``t_start``: seconds from ``Cycle_start`` to ``TS_start`` for each row.

    The floating-bar base offset for ``_add_ring_bars``.  Handles both
    tz-aware Timestamp columns and float (UTC epoch) columns.  Values
    clipped to 0 to absorb sub-second float precision drift.

    Args:
        df_signal: DataFrame with ``TS_start`` and ``Cycle_start`` columns.

    Returns:
        Copy of ``df_signal`` with ``t_start`` (float, seconds) column added.
    """
    df = df_signal.copy()
    ts = df['TS_start']
    cs = df['Cycle_start']

    if pd.api.types.is_float_dtype(ts) and pd.api.types.is_float_dtype(cs):
        df['t_start'] = (ts - cs).clip(lower=0.0)
    else:
        if pd.api.types.is_float_dtype(ts):
            ts = pd.to_datetime(ts, unit='s', utc=True)
        if pd.api.types.is_float_dtype(cs):
            cs = pd.to_datetime(cs, unit='s', utc=True)
        df['t_start'] = (ts - cs).dt.total_seconds().clip(lower=0.0)

    return df


def _compute_durations(df_signal: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ``Duration`` for each signal-state row using a within-phase shift.

    If the DataFrame already contains a ``Duration`` column this function is
    a no-op.  The shift is performed per ``ID`` across the full dataset;
    duration for the last event of each phase group is NaN and those rows are
    later dropped.

    Gap markers (Code == -1) must be excluded *before* calling this function
    to prevent spurious durations spanning a discontinuity.

    Args:
        df_signal: Signal events DataFrame with
            ``[TS_start, Code, ID, Cycle_start]``.

    Returns:
        df_signal copy with a ``Duration`` column added.
    """
    df = df_signal.sort_values(['ID', 'TS_start']).copy()
    df['_ts_end'] = df.groupby('ID')['TS_start'].shift(-1)
    df['Duration'] = (df['_ts_end'] - df['TS_start']).dt.total_seconds()
    df = df.drop(columns=['_ts_end'])
    return df


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _build_title(metadata: Dict[str, Any], suffix: str = '') -> str:
    """
    Construct a plot title from metadata.

    Prefers ``{major_road_name} @ {minor_road_name}``; falls back to
    ``intersection_name``.

    Args:
        metadata: Dict with road/intersection name keys.
        suffix: String appended after the location name.

    Returns:
        Formatted title string.
    """
    major = str(metadata.get('major_road_name', '') or '').strip()
    minor = str(metadata.get('minor_road_name', '') or '').strip()
    intx  = str(metadata.get('intersection_name', 'Intersection') or '').strip()

    location = f'{major} @ {minor}' if (major and minor) else intx
    return f'{location} – {suffix}' if suffix else location


def _validate_signal_columns(df: pd.DataFrame) -> None:
    """
    Raise ValueError if required signal DataFrame columns are absent.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: Listing missing columns.
    """
    required = ['TS_start', 'Code', 'ID', 'Cycle_start']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"df_signal is missing required columns: {missing}"
        )
