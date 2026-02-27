"""
ATSPM Coordination / Split Diagram Plot (Functional Core).

Pure function – no SQL, no file I/O, no side effects.
Input:  cycles DataFrame, signal events DataFrame, optional detector DataFrame,
        metadata dict, and detector config series/dict.
Output: plotly.graph_objects.Figure (1 or 2 ring subplots).

Package Location: src/atspm/plotting/coordination.py

Ring Subplot Rule:
    A subplot for Ring 2 is only created when ``r2_phases`` in df_cycles
    contains real phase numbers (not the literal string ``"None"`` and not
    empty).

Legend Groups:
    Phase dummy-legend entries are listed under bold ``"Ring 1 Phases"`` /
    ``"Ring 2 Phases"`` visual headers, sorted by phase number within each
    ring.  Detector entries appear under ``"Ring 1 Detectors"`` /
    ``"Ring 2 Detectors"`` headers in the same style, sorted by phase then
    detector type (Arrival → Stop Bar → Occupancy).

Hover Behaviour:
    ``hovermode='closest'`` is used so that a tooltip fires per bar element,
    giving the user toolbar toggle control (Compare / Closest).  Only the
    **Green** bar for each phase carries a hover tooltip; that tooltip
    includes the full G / Y / Rc breakdown (start and duration) for the
    cycle × phase, so no information is lost despite Y and Rc bars having
    ``hoverinfo='skip'``.  Detector scatter traces carry individual element
    hover only and are not part of the phase tooltip.

Gap Marker Rule:
    df_signal rows with Code == -1 (event_code == -1) are excluded before
    all duration and stacking calculations.

Bar Rendering:
    Bars are rendered semi-transparent (opacity ``_BAR_OPACITY``) so that
    overlapping detector scatter points remain clearly visible against the
    bar fill.  Text labels on bars are intentionally omitted.  Green bars
    use each phase's primary colour; Yellow bars use ``'Gold'``; Red-
    clearance bars use ``'Red'``.

Detector X-Shift:
    Each detector type is offset slightly along the **x-axis** (time axis)
    relative to its ``Cycle_start`` to visually align detectors near the
    edges of their corresponding signal bars rather than clustering all
    detector types at the same cycle timestamp.  This is a cosmetic offset
    only and does not alter the ``t_cs`` (time-in-cycle) y-value.  Arrival
    detectors shift left (−10 s), Occupancy detectors are centred (0 s),
    and Stop Bar detectors shift right (+10 s).

Arrival Offset Sliders:
    One compact slider per ring, independent of each other.  Rendered only
    when arrival detector traces exist AND ``len(df_cycles) <
    MAX_CYCLES_FOR_SLIDER`` (2 000).  Steps run from −30 s to +30 s in
    1-second increments; labels are shown every 5 s.  Sliders are positioned
    below their respective ring subplot.

Example SQL to build the three input DataFrames
-----------------------------------------------
    .. code-block:: sql

        -- df_cycles
        SELECT
            datetime(cycle_start, 'unixepoch', 'localtime') AS Cycle_start,
            coord_plan                                        AS Coord_plan,
            r1_phases                                         AS r1_phases,
            r2_phases                                         AS r2_phases
        FROM cycles
        WHERE cycle_start BETWEEN :start_ts AND :end_ts
        ORDER BY cycle_start;

        -- df_signal
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
             ) OR (SELECT MIN(c2.cycle_start) FROM cycles c2
                   WHERE c2.cycle_start > c.cycle_start) IS NULL)
        WHERE e.event_code IN (-1, 1, 8, 9, 11, 12)
          AND e.timestamp BETWEEN :start_ts AND :end_ts
        ORDER BY e.timestamp;

        -- df_det  (t_cs pre-computed in SQL for efficiency)
        SELECT
            datetime(e.timestamp, 'unixepoch', 'localtime')   AS TS_start,
            e.event_code                                       AS Code,
            e.parameter                                        AS ID,
            datetime(c.cycle_start, 'unixepoch', 'localtime') AS Cycle_start,
            (e.timestamp - c.cycle_start)                      AS t_cs
        FROM events e
        JOIN cycles c
          ON e.timestamp >= c.cycle_start
         AND (e.timestamp < (
                 SELECT MIN(c2.cycle_start) FROM cycles c2
                 WHERE c2.cycle_start > c.cycle_start
             ) OR (SELECT MIN(c2.cycle_start) FROM cycles c2
                   WHERE c2.cycle_start > c.cycle_start) IS NULL)
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

# Suppress the arrival slider when the dataset is too large to animate smoothly.
MAX_CYCLES_FOR_SLIDER: int = 2_000

# Bar fill opacity – semi-transparent so detector scatter points show through.
_BAR_OPACITY: float = 0.55

# Phase colour mapping (matches legacy cl_P dict in plotting.py).
_PHASE_COLORS: Dict[str, str] = {
    '1':  'DarkOrange',     '5':  'LightSalmon',
    '3':  'Magenta',        '7':  'Purple',
    '2':  'Blue',           '6':  'Turquoise',
    '4':  'DarkGreen',      '8':  'Lime',
    '9':  'DarkOliveGreen', '10': 'Olive',
    '11': 'DarkSlateGray',  '12': 'Gray',
    '13': 'LightGray',      '14': 'Silver',
    '15': 'DimGray',        '16': 'Black',
}

# Colour overrides for Yellow and Red-Clearance intervals.
_GYR_COLORS: Dict[str, str] = {
    'Rc': 'Red',
    'Y':  'Gold',
}

# Detector rendering config.
# ``x_shift_s`` is a cosmetic *time-axis* offset in seconds that spreads
# detector types horizontally so they align near the edges of their signal
# bars rather than stacking at the exact cycle-start x-coordinate.
# This offset does NOT affect the t_cs (y-axis) value.
_DET_CONFIG: Dict[str, Dict[str, Any]] = {
    'Ar': {'color': 'DimGray', 'symbol': 'circle',  'label': 'Arrival',   'x_shift_s': -10},
    'Oc': {'color': 'Black',   'symbol': 'diamond', 'label': 'Occupancy', 'x_shift_s':   0},
    'St': {'color': 'Crimson', 'symbol': 'square',  'label': 'Stop Bar',  'x_shift_s':  10},
}

# Mapping from raw ATSPM event code to GYR label.
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
    individual_detectors: bool = False,
) -> go.Figure:
    """Build a stacked-bar Coordination / Split Diagram.

    One subplot per ring is generated; the Ring 2 subplot is omitted when
    ``r2_phases`` contains only ``"None"`` or empty strings.

    Args:
        df_cycles: Cycle-level DataFrame with columns:

            * ``Cycle_start`` – datetime-like cycle boundary timestamp.
            * ``Coord_plan``  – int/float coordination plan number.
            * ``r1_phases``   – comma-separated phase numbers for Ring 1
              (e.g. ``"2,6"`` or ``"None"``).
            * ``r2_phases``   – same format for Ring 2.

        df_signal: Signal-event DataFrame with columns:

            * ``TS_start``    – datetime-like event timestamp.
            * ``Code``        – int (1=Green, 8=Yellow, 9=RedClearance,
              11/12=Red, -1=Gap).
            * ``ID``          – int phase number.
            * ``Cycle_start`` – datetime-like cycle boundary for this event.

        metadata: Dict containing at minimum ``major_road_name``,
            ``minor_road_name``, and ``intersection_name``.
        df_det: Optional detector events DataFrame with columns:

            * ``TS_start``    – datetime-like event timestamp.
            * ``Code``        – int (82=actuation on, 81=actuation off).
            * ``ID``          – int detector number.
            * ``Cycle_start`` – datetime-like cycle boundary.
            * ``t_cs``        – float seconds from ``Cycle_start`` to
              ``TS_start`` (pre-computed in SQL; computed here if absent).

        det_config: Optional dict or pandas Series mapping legacy config
            keys to comma-separated detector number strings.  Expected key
            format: ``"P{phase} {TypeLabel}"`` e.g. ``"P2 Arrival"``,
            ``"P6 Stop Bar"``.
        individual_detectors: When ``True``, each detector ID gets its own
            legend-toggleable trace instead of being grouped by phase × type.
            Useful for lane-by-lane analysis.  Default ``False`` (grouped by
            config key, i.e. one trace per ``"P{phase} {Type}"`` entry).

    Returns:
        A ``plotly.graph_objects.Figure`` with 1 or 2 vertically stacked
        ring subplots.
    """
    _validate_signal_columns(df_signal)

    title = _build_title(metadata, suffix='Coordination / Split Diagram')

    # Exclude gap-marker rows before any duration calculation.
    df_signal = df_signal[df_signal['Code'] != _GAP_CODE].copy()

    if 'Duration' not in df_signal.columns:
        df_signal = _compute_durations(df_signal)

    # Retain only Green / Yellow / Red-Clearance codes.
    df_signal = df_signal[df_signal['Code'].isin([1, 8, 9])].copy()
    df_signal['ID'] = pd.to_numeric(df_signal['ID'], errors='coerce').astype('Int64')
    df_signal = df_signal.dropna(subset=['ID', 'Duration'])
    # Cap Duration to guard against inter-cycle bleed from pre-computed columns.
    df_signal['Duration'] = df_signal['Duration'].clip(upper=300.0)
    df_signal['gyr'] = df_signal['Code'].map(_CODE_TO_GYR)

    df_signal = _compute_t_start(df_signal)
    df_signal = df_signal.dropna(subset=['t_start'])

    r1_phases_all, r2_phases_all = _collect_ring_phases(df_cycles)
    has_r2 = bool(r2_phases_all)

    n_rows = 2 if has_r2 else 1
    subplot_titles = ['Ring 1', 'Ring 2'] if has_r2 else ['Ring 1']

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
    )

    dummy_legend_added: set = set()

    # Ring 1 bars ─────────────────────────────────────────────────────────────
    _add_ring_bars(
        fig=fig,
        row=1,
        ring_phases=r1_phases_all,
        ring_label='Ring 1',
        df_cycles=df_cycles,
        df_signal=df_signal,
        ring_col='r1_phases',
        dummy_legend_added=dummy_legend_added,
    )

    # Ring 2 bars ─────────────────────────────────────────────────────────────
    if has_r2:
        _add_ring_bars(
            fig=fig,
            row=2,
            ring_phases=r2_phases_all,
            ring_label='Ring 2',
            df_cycles=df_cycles,
            df_signal=df_signal,
            ring_col='r2_phases',
            dummy_legend_added=dummy_legend_added,
        )

    # Dummy legend entries (phase colours) ────────────────────────────────────
    _add_dummy_legends(fig, r1_phases_all, r2_phases_all)

    # Coord-plan tick markers ──────────────────────────────────────────────────
    _add_coord_plan_markers(fig, df_cycles, row=1)

    # Detector scatter traces ──────────────────────────────────────────────────
    r1_arrival_traces: List[Dict[str, Any]] = []
    r2_arrival_traces: List[Dict[str, Any]] = []

    if df_det is not None and det_config is not None and not df_det.empty:
        r1_arrival_traces, r2_arrival_traces = _add_detector_traces(
            fig=fig,
            df_det=df_det,
            det_config=det_config,
            ring1_phases=r1_phases_all,
            ring2_phases=r2_phases_all if has_r2 else set(),
            n_rows=n_rows,
            individual_detectors=individual_detectors,
        )

    # Per-ring arrival offset sliders ──────────────────────────────────────────
    n_cycles = len(df_cycles)
    sliders: List[Dict[str, Any]] = []
    slider_len = 0.15

    if r1_arrival_traces and n_cycles < MAX_CYCLES_FOR_SLIDER:
        slider_y = -0.05 if not has_r2 else 0.55
        sliders.append(_build_arrival_slider(
            arrival_traces=r1_arrival_traces,
            label='R1 Arrival Offset (s)',
            x=0.0,
            x_anchor='left',
            slider_len=slider_len,
            y=slider_y,
        ))

    if r2_arrival_traces and n_cycles < MAX_CYCLES_FOR_SLIDER and has_r2:
        sliders.append(_build_arrival_slider(
            arrival_traces=r2_arrival_traces,
            label='R2 Arrival Offset (s)',
            x=0.0,
            x_anchor='left',
            slider_len=slider_len,
            y=0.0,
        ))

    # Initial zoom: 06:00-19:00 window with y scaled to daytime bar heights.
    x_range, y_range = _compute_initial_zoom(df_cycles, df_signal)

    # Layout ───────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        barmode='overlay',
        showlegend=True,
        legend=dict(
            orientation='v',
            x=1.01,
            y=1.0,
            xanchor='left',
            tracegroupgap=5,
        ),
        hovermode='closest',
        template='plotly_white',
        height=420 * n_rows + 100,
    )

    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        for _row in range(1, n_rows + 1):
            fig.update_yaxes(range=y_range, row=_row, col=1)

    if sliders:
        fig.update_layout(sliders=sliders)

    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text='Seconds into Cycle', row=i, col=1)

    _date_str = ''
    if not df_cycles.empty and 'Cycle_start' in df_cycles.columns:
        _cs = df_cycles['Cycle_start'].iloc[-1]
        _date_str = _cs.strftime('%m/%d/%Y') if hasattr(_cs, 'strftime') else str(_cs)[:10]
    
    fig.update_xaxes(title_text=_date_str, row=n_rows, col=1)
    fig.update_xaxes(tickformat='%H:%M')

    return fig


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Initial zoom helpers
# ---------------------------------------------------------------------------

def _compute_initial_zoom(
    df_cycles: pd.DataFrame,
    df_signal: pd.DataFrame,
    zoom_hour_start: int = 6,
    zoom_hour_end: int = 19,
    y_cushion_pct: float = 0.08,
) -> Tuple[Optional[List], Optional[List]]:
    """Compute a sensible initial x- and y-axis zoom range for daytime hours.

    Overnight cycles commonly have very long cycle lengths (e.g. 300-600 s)
    that cause the default auto-zoom to display an unhelpfully tall y-range.
    This function restricts the initial viewport to a configurable daytime
    window on the x-axis and derives the y-axis ceiling from the tallest bar
    top (``t_start + Duration``) within that window.

    The figure retains full pan/zoom capability; this only sets the initial
    viewport when the HTML is first rendered.

    Args:
        df_cycles: Cycles DataFrame with a ``Cycle_start`` column.
        df_signal: Signal events DataFrame with ``t_start`` and ``Duration``
            columns (both already computed before this is called).
        zoom_hour_start: Local hour (0-23) for the left edge of the initial
            x-axis window.  Defaults to 6 (06:00).
        zoom_hour_end: Local hour (0-23) for the right edge of the initial
            x-axis window.  Defaults to 19 (19:00).
        y_cushion_pct: Fractional headroom added above the tallest bar top
            within the zoom window.  Defaults to 0.08 (8 %).

    Returns:
        Tuple ``(x_range, y_range)`` where each element is either a two-item
        list suitable for Plotly's ``range`` axis property, or ``None`` if
        the data are insufficient to compute that range.
    """
    if df_cycles.empty or 'Cycle_start' not in df_cycles.columns:
        return None, None

    cs = pd.to_datetime(df_cycles['Cycle_start'], errors='coerce').dropna()
    if cs.empty:
        return None, None

    hour = cs.dt.hour
    daytime_mask = (hour >= zoom_hour_start) & (hour < zoom_hour_end)
    cs_day = cs[daytime_mask]

    x_range: Optional[List] = None
    if not cs_day.empty:
        x_range = [cs_day.min().isoformat(), cs_day.max().isoformat()]

    # Y ceiling from bar tops (t_start + Duration) in the daytime window.
    if 't_start' not in df_signal.columns or 'Duration' not in df_signal.columns:
        return x_range, None

    sig_cs = pd.to_datetime(df_signal['Cycle_start'], errors='coerce')
    sig_hour = sig_cs.dt.hour
    day_sig = df_signal[
        (sig_hour >= zoom_hour_start) & (sig_hour < zoom_hour_end)
    ]

    if day_sig.empty:
        return x_range, None

    bar_tops = (
        pd.to_numeric(day_sig['t_start'],   errors='coerce') +
        pd.to_numeric(day_sig['Duration'], errors='coerce')
    ).dropna()

    if bar_tops.empty:
        return x_range, None

    y_max = float(bar_tops.max())
    y_top = y_max * (1.0 + y_cushion_pct)
    # -8 provides room for coord-plan tick markers rendered at y = -5.
    y_range: Optional[List] = [-8.0, y_top]

    return x_range, y_range


# Ring / phase resolution helpers
# ---------------------------------------------------------------------------

def _collect_ring_phases(df_cycles: pd.DataFrame) -> Tuple[set, set]:
    """Collect the full set of real phase IDs for each ring across all cycles.

    ``"None"`` strings and empty/non-numeric tokens are ignored.

    Args:
        df_cycles: Cycles DataFrame with ``r1_phases`` and ``r2_phases``
            columns.

    Returns:
        Tuple of ``(r1_set, r2_set)`` – sets of integer phase IDs.
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
    """Return the display colour for a given phase and signal state.

    Green intervals use the phase-specific colour from ``_PHASE_COLORS``;
    Yellow and Red-Clearance use ``_GYR_COLORS``.

    Args:
        phase_id: Integer phase number.
        gyr: Signal-state label – ``'G'``, ``'Y'``, or ``'Rc'``.

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
    """Build a phase → display-position mapping from the modal ring-phase string.

    Used only to determine the cosmetic order of legend entries and Plotly
    trace layers.  Actual bar vertical positions are always determined by
    ``t_start``, never by this mapping.

    Prefers non-reservice sequences (each phase appearing once) for a stable
    ordering.  Falls back to the raw modal string when all sequences contain
    reservice.

    Args:
        df_cycles: Cycles DataFrame containing the ring column.
        ring_col: Column name (``'r1_phases'`` or ``'r2_phases'``).
        ring_phases: Set of integer phase IDs present in this ring.

    Returns:
        Dict mapping integer ``phase_id`` → integer display position.
        Phases absent from the modal sequence receive position 999 (last).
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
    ring_label: str,
    df_cycles: pd.DataFrame,
    df_signal: pd.DataFrame,
    ring_col: str,
    dummy_legend_added: set,
) -> None:
    """Add floating-bar traces for every signal interval in one ring.

    Each bar is positioned using Plotly's ``base`` parameter:

    * ``base`` = ``t_start``  – seconds from ``Cycle_start`` to interval start.
    * ``y``    = ``Duration`` – height of the bar in seconds.

    Bars are semi-transparent (``_BAR_OPACITY``) to keep detector scatter
    points visible.  Text labels on bars are intentionally omitted.

    Only the **Green** bar for each phase × cycle carries a hover tooltip;
    that tooltip shows the full G / Y / Rc breakdown (start and duration).
    Yellow and Red-Clearance bars have ``hoverinfo='skip'`` to avoid
    redundant or cluttered tooltips.

    Short-green fallback: when a Green bar's computed ``Duration`` is ≤ 5 s
    (indicating a spurious intermediate Code 1 re-assertion logged by some
    controllers immediately before the yellow transition) the duration is
    replaced by ``yellow_t_start − green_t_start`` for that cycle × phase,
    giving a visually accurate bar height.  If no Yellow row exists for that
    cycle the original short duration is kept.

    Args:
        fig: Figure to mutate in place.
        row: Subplot row index (1-based).
        ring_phases: Set of integer phase IDs assigned to this ring.
        ring_label: Human-readable ring label (e.g. ``'Ring 1'``), used to
            track which phases have received dummy legend entries.
        df_cycles: Cycles DataFrame with ``Cycle_start`` and the ring-phase
            column.
        df_signal: Signal events DataFrame.  Must already contain columns
            ``t_start``, ``Duration``, and ``gyr``.
        ring_col: Column in ``df_cycles`` giving the per-cycle phase-sequence
            string (``'r1_phases'`` or ``'r2_phases'``).
        dummy_legend_added: Mutable set; updated with ``(phase_id, ring_label)``
            tuples as phases are processed.
    """
    if not ring_phases:
        return

    sig = df_signal[df_signal['ID'].isin(ring_phases)].copy()
    if sig.empty:
        return

    sig = sig.sort_values(['Cycle_start', 'ID', 'TS_start'])

    # Assign a reservice sequence index to each green interval per cycle × phase.
    green_mask = sig['Code'] == 1
    sig.loc[green_mask, '_green_count'] = (
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

    display_order = _modal_phase_sequence(df_cycles, ring_col, ring_phases)
    gyr_order = {'G': 0, 'Y': 1, 'Rc': 2}

    combos = sig[['ID', 'gyr', '_seq_pos']].drop_duplicates().copy()
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
        sub = sig[mask].dropna(subset=['Duration', 't_start']).copy()
        if sub.empty:
            continue

        svc_label = ' (reservice)' if seq_pos > 0 else ''

        if gyr == 'G':
            # Short-green fallback: when Duration ≤ 5 s the computed endpoint
            # is likely a spurious intermediate Code 1 re-assertion (some
            # controllers briefly re-log green before transitioning to yellow).
            # For these cycles substitute (yellow_t_start − green_t_start)
            # so the bar height matches the visually meaningful green interval.
            # Short-green fallback: carry the flag as a column so it
            # survives the merge (a boolean Series index is invalidated
            # after pd.merge resets the index, causing silent misalignment).
            sub['_short'] = sub['Duration'] <= 5.0
            if sub['_short'].any():
                yel = sig[
                    (sig['ID']       == phase_id) &
                    (sig['gyr']      == 'Y')       &
                    (sig['_seq_pos'] == seq_pos)
                ][['Cycle_start', 't_start']].rename(
                    columns={'t_start': '_yel_t_start'}
                )
                if not yel.empty:
                    sub = sub.merge(yel, on='Cycle_start', how='left')
                    fix = sub['_short'] & sub['_yel_t_start'].notna()
                    if fix.any():
                        sub.loc[fix, 'Duration'] = (
                            sub.loc[fix, '_yel_t_start'] - sub.loc[fix, 't_start']
                        ).clip(lower=0.0)
                    sub = sub.drop(columns=['_yel_t_start'])
            sub = sub.drop(columns=['_short'])
            # Build per-cycle GYR lookup for hover construction.
            phase_sig = sig[
                (sig['ID']       == phase_id) &
                (sig['_seq_pos'] == seq_pos)
            ]
            gyr_lookup: Dict[Any, Dict[str, Tuple[float, float]]] = {}
            for cs_v, grp in phase_sig.groupby('Cycle_start'):
                gyr_lookup[cs_v] = {
                    r['gyr']: (r['t_start'], r['Duration'])
                    for _, r in grp.iterrows()
                }

            def _fmt_gyr(cycle_val: Any, gyr_key: str) -> str:
                """Format one GYR row for the Green-bar hover tooltip."""
                entry = gyr_lookup.get(cycle_val, {}).get(gyr_key)
                if entry is None:
                    return '  –'
                ts_val, dur_val = entry
                dur_str = f'{dur_val:.1f} s' if not pd.isna(dur_val) else '–'
                return f'  Start: {ts_val:.1f} s   Dur: {dur_str}'

            hover_texts = []
            for cs in sub['Cycle_start']:
                ct = cs.strftime('%H:%M:%S') if hasattr(cs, 'strftime') else str(cs)
                hover_texts.append(
                    f'<b>{ct}</b><br>'
                    f'<b>Phase {phase_id}{svc_label}</b><br>'
                    f'G{_fmt_gyr(cs, "G")}<br>'
                    f'Y{_fmt_gyr(cs, "Y")}<br>'
                    f'Rc{_fmt_gyr(cs, "Rc")}'
                )

            dummy_legend_added.add((phase_id, ring_label))

            fig.add_trace(
                go.Bar(
                    x=sub['Cycle_start'],
                    y=sub['Duration'],
                    width=30_000,
                    base=sub['t_start'],
                    name=f'Ph {phase_id}',
                    marker=dict(color=color, opacity=_BAR_OPACITY),
                    showlegend=False,
                    legendgroup=f'Ph {phase_id}',
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts,
                    textposition='none',  # suppress text rendering on bar face
                ),
                row=row,
                col=1,
            )
        else:
            # Y / Rc bars: semi-transparent fill, no hover tooltip.
            fig.add_trace(
                go.Bar(
                    x=sub['Cycle_start'],
                    y=sub['Duration'],
                    width=30_000,
                    base=sub['t_start'],
                    name=f'Ph {phase_id} {gyr}',
                    marker=dict(color=color, opacity=_BAR_OPACITY),
                    showlegend=False,
                    legendgroup=f'Ph {phase_id}',
                    hoverinfo='skip',
                ),
                row=row,
                col=1,
            )


def _add_dummy_legends(
    fig: go.Figure,
    r1_phases: set,
    r2_phases: set,
) -> None:
    """Add one invisible scatter trace per phase for the legend panel.

    Phases are listed under bold ``"Ring N Phases"`` visual headers
    implemented as zero-data scatter traces with transparent markers.
    Entries are sorted by phase number within each ring group.

    Args:
        fig: Figure to mutate in place.
        r1_phases: Set of integer phase IDs present in Ring 1.
        r2_phases: Set of integer phase IDs present in Ring 2.
    """
    for ring_label, ring_phases in [
        ('Ring 1 Phases', r1_phases),
        ('Ring 2 Phases', r2_phases),
    ]:
        if not ring_phases:
            continue

        # Bold section header – invisible, non-interactive.
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=1),
            name=f'<b>{ring_label}</b>',
            showlegend=True,
            hoverinfo='skip',
        ))

        for phase_id in sorted(ring_phases):
            color = _PHASE_COLORS.get(str(phase_id), 'Gray')
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
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
    """Add a small horizontal-tick scatter at y=−5 per cycle, coloured by plan.

    Provides a visual strip at the bottom of Ring 1 indicating which
    coordination plan is active at each cycle boundary.

    Args:
        fig: Figure to mutate in place.
        df_cycles: Cycles DataFrame with ``Cycle_start`` and ``Coord_plan``
            columns.
        row: Subplot row in which to render the markers (default ``1``).
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
            hovertemplate='Plan: %{text}<br>Cycle: %{x|%H:%M:%S}<extra></extra>',
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
    individual_detectors: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Add detector scatter traces to the appropriate ring subplot.

    Two grouping modes are supported, controlled by ``individual_detectors``:

    **Grouped** (default, ``individual_detectors=False``):
        Each config key ``"P{phase} {Type}"`` produces one ``go.Scatter``
        trace containing all detector IDs for that key, concatenated with
        ``None`` separators.  One legend toggle controls the whole
        phase × type group.  One arrival trace per ``(phase, Ar)`` key is
        registered with the slider.

    **Individual** (``individual_detectors=True``):
        Each detector ID gets its own ``go.Scatter`` trace for lane-by-lane
        toggle-ability.  One arrival trace per detector ID is registered
        with the slider.

    In both modes, per-point ``customdata`` carries
    ``[det_id, phase_num, type_label]`` so hover identifies the detector.
    A cosmetic x-axis shift is applied per detector type (Arrival: −10 s,
    Occupancy: 0 s, Stop Bar: +10 s) to spread dots near bar edges.
    Legend headers (``"Ring N Detectors"``) are invisible sentinel traces
    added once per ring.

    Args:
        fig: Figure to mutate in place.
        df_det: Detector events DataFrame with columns
            ``[TS_start, Code, ID, Cycle_start]``.  ``t_cs`` is computed
            here if absent.
        det_config: Mapping of legacy config keys (``"P{phase} {Type}"``)
            to comma-separated detector number strings.
        ring1_phases: Set of integer phase IDs in Ring 1.
        ring2_phases: Set of integer phase IDs in Ring 2.
        n_rows: Total subplot row count.
        individual_detectors: When ``True``, emit one trace per detector ID
            rather than one per config key.  Default ``False``.

    Returns:
        Tuple ``(r1_arrival_traces, r2_arrival_traces)``, each a list of
        dicts ``{"trace_index": int, "orig_y": np.ndarray}`` — one entry
        per arrival trace registered for slider control.
    """
    r1_arrival_traces: List[Dict[str, Any]] = []
    r2_arrival_traces: List[Dict[str, Any]] = []

    df_on = df_det[df_det['Code'] == 82].copy()
    if df_on.empty:
        return r1_arrival_traces, r2_arrival_traces

    if 't_cs' not in df_on.columns:
        df_on = _compute_t_cs(df_on)
    df_on['t_cs'] = pd.to_numeric(df_on['t_cs'], errors='coerce')
    df_on = df_on.dropna(subset=['t_cs'])
    df_on['t_cs'] = df_on['t_cs'].clip(lower=0.0)

    if hasattr(det_config, 'items'):
        cfg_items = list(det_config.items())
    else:
        cfg_items = list(det_config)

    def _sort_key(item: Tuple[str, Any]) -> Tuple[int, int]:
        ph, dt = _parse_det_key(item[0])
        type_order = {'Ar': 0, 'St': 1, 'Oc': 2}
        return (ph or 999, type_order.get(dt or '', 9))

    cfg_items_sorted = sorted(cfg_items, key=_sort_key)

    r1_header_added = False
    r2_header_added = False

    for key, det_str in cfg_items_sorted:
        if not det_str or pd.isna(det_str):
            continue
        det_str = str(det_str).strip()
        if not det_str:
            continue

        phase_num, dtype_key = _parse_det_key(key)
        if phase_num is None or dtype_key not in _DET_CONFIG:
            continue

        dcfg       = _DET_CONFIG[dtype_key]
        color      = dcfg['color']
        symbol     = dcfg['symbol']
        type_label = dcfg['label']
        x_shift_s  = dcfg['x_shift_s']

        in_r2 = phase_num in ring2_phases and n_rows == 2
        row   = 2 if in_r2 else 1

        # Legend header sentinel on first encounter for this ring.
        if in_r2 and not r2_header_added:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color='rgba(0,0,0,0)', size=1),
                name='<b>Ring 2 Detectors</b>',
                showlegend=True, hoverinfo='skip',
            ), row=row, col=1)
            r2_header_added = True
        elif not in_r2 and not r1_header_added:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color='rgba(0,0,0,0)', size=1),
                name='<b>Ring 1 Detectors</b>',
                showlegend=True, hoverinfo='skip',
            ), row=row, col=1)
            r1_header_added = True

        # Build per-detector data segments, then emit as either one grouped
        # trace (default) or individual traces (individual_detectors=True).
        det_segments: List[Tuple[int, Any, Any]] = []  # (det_id, x_shifted, y_vals)

        for det_id_str in det_str.split(','):
            det_id_str = det_id_str.strip()
            if not det_id_str.isdigit():
                continue
            det_id = int(det_id_str)

            df_d = df_on[df_on['ID'] == det_id].copy()
            if df_d.empty:
                continue

            y_vals    = df_d['t_cs'].values
            x_shifted = _apply_x_shift(df_d['Cycle_start'], x_shift_s)
            det_segments.append((det_id, x_shifted, y_vals))

        if not det_segments:
            continue

        def _emit_trace(
            tr_x: List[Any],
            tr_y: List[Any],
            tr_cd: List[Any],
            tr_name: str,
            lg: str,
        ) -> None:
            """Emit one Scatter trace and register it with the slider if Ar."""
            fig.add_trace(go.Scatter(
                x=tr_x,
                y=tr_y,
                mode='markers',
                marker=dict(color=color, size=5, opacity=0.85, symbol=symbol),
                name=tr_name,
                legendgroup=lg,
                showlegend=True,
                visible='legendonly',
                customdata=tr_cd,
                hovertemplate=(
                    '<b>%{x|%H:%M:%S}</b><br>'
                    'Det %{customdata[0]} – Ph %{customdata[1]} %{customdata[2]}<br>'
                    't_cs: %{y:.1f} s'
                    '<extra></extra>'
                ),
            ), row=row, col=1)
            if dtype_key == 'Ar':
                y_arr = np.array(
                    [v if v is not None else np.nan for v in tr_y], dtype=float
                )
                entry = {'trace_index': len(fig.data) - 1, 'orig_y': y_arr}
                (r2_arrival_traces if in_r2 else r1_arrival_traces).append(entry)

        if individual_detectors:
            # One trace per detector ID — maximum toggle granularity.
            for det_id, x_shifted, y_vals in det_segments:
                _emit_trace(
                    tr_x=x_shifted.tolist(),
                    tr_y=y_vals.tolist(),
                    tr_cd=[[det_id, phase_num, type_label]] * len(y_vals),
                    tr_name=f'Ph {phase_num} {dtype_key} ({det_id})',
                    lg=f'det_{phase_num}_{dtype_key}_{det_id}',
                )
        else:
            # One trace per config key (phase × type) — grouped mode.
            grp_x:  List[Any] = []
            grp_y:  List[Any] = []
            grp_cd: List[Any] = []
            for det_id, x_shifted, y_vals in det_segments:
                if grp_x:
                    grp_x.append(None); grp_y.append(None)
                    grp_cd.append([None, None, None])
                grp_x.extend(x_shifted.tolist())
                grp_y.extend(y_vals.tolist())
                grp_cd.extend([[det_id, phase_num, type_label]] * len(y_vals))
            _emit_trace(
                tr_x=grp_x,
                tr_y=grp_y,
                tr_cd=grp_cd,
                tr_name=f'Ph {phase_num} {dtype_key}',
                lg=f'det_{phase_num}_{dtype_key}',
            )

    return r1_arrival_traces, r2_arrival_traces


def _apply_x_shift(
    cycle_start_series: pd.Series,
    shift_seconds: int,
) -> pd.Series:
    """Apply a cosmetic x-axis offset in seconds to a Cycle_start series.

    Handles both ``datetime64`` and float (Unix epoch seconds) dtypes so that
    the shift is applied consistently regardless of how ``Cycle_start`` is
    stored.

    Args:
        cycle_start_series: Series of cycle-start timestamps.
        shift_seconds: Integer offset in seconds (may be negative).

    Returns:
        New Series with the offset applied, preserving the original dtype
        where possible.
    """
    if shift_seconds == 0:
        return cycle_start_series

    if pd.api.types.is_float_dtype(cycle_start_series):
        return cycle_start_series + shift_seconds

    return pd.to_datetime(cycle_start_series) + pd.Timedelta(seconds=shift_seconds)


def _compute_t_cs(df_det: pd.DataFrame) -> pd.DataFrame:
    """Compute ``t_cs``: seconds from ``Cycle_start`` to ``TS_start``.

    Fallback used when ``t_cs`` is not pre-computed by the SQL query.  Values
    are clipped to 0 to absorb sub-second floating-point drift at cycle
    boundaries.

    Args:
        df_det: Detector DataFrame with ``TS_start`` and ``Cycle_start``
            columns.

    Returns:
        Copy of ``df_det`` with a ``t_cs`` (float, seconds ≥ 0) column added.
    """
    df = df_det.copy()
    ts = df['TS_start']
    cs = df['Cycle_start']

    if pd.api.types.is_float_dtype(ts) and pd.api.types.is_float_dtype(cs):
        df['t_cs'] = (ts - cs).clip(lower=0.0)
    else:
        if pd.api.types.is_float_dtype(ts):
            ts = pd.to_datetime(ts, unit='s', utc=True)
        if pd.api.types.is_float_dtype(cs):
            cs = pd.to_datetime(cs, unit='s', utc=True)
        df['t_cs'] = (
            pd.to_datetime(ts) - pd.to_datetime(cs)
        ).dt.total_seconds().clip(lower=0.0)

    return df


def _parse_det_key(key: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse a legacy detector config key into ``(phase_number, type_key)``.

    Expected format: ``"P{phase} {TypeLabel}"`` e.g. ``"P2 Arrival"``.

    Args:
        key: Detector config key string.

    Returns:
        Tuple of ``(phase_int, dtype_key)`` where ``dtype_key`` is one of
        ``'Ar'``, ``'St'``, ``'Oc'``.  Returns ``(None, None)`` on parse
        failure.
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

def _build_arrival_slider(
    arrival_traces: List[Dict[str, Any]],
    label: str,
    x: float,
    x_anchor: str,
    slider_len: float,
    y: float,
) -> Dict[str, Any]:
    """Build a compact Plotly slider dict that shifts arrival trace y-values.

    Steps run from −30 s to +30 s in 1-second increments.  The active index
    is 30, corresponding to an offset of 0.  Every step carries a numeric
    label so the current-value display always shows the exact offset; visual
    tick marks are hidden via ``ticklen=0`` to keep the slider compact.

    Since all arrival detectors for a ring are consolidated into a single
    trace (see ``_add_detector_traces``), each slider step restyles exactly
    one trace, keeping browser-side restyle calls fast even for large
    datasets.  ``NaN`` sentinels used as gap separators in the consolidated
    array are preserved because ``NaN + scalar == NaN`` in numpy.

    Args:
        arrival_traces: Single-element list containing a dict with keys
            ``trace_index`` (int) and ``orig_y`` (numpy float array, NaN
            for separator positions) for the consolidated arrival trace.
        label: Prefix shown next to the current value display.
        x: Horizontal position of the slider in paper coordinates (0–1).
        x_anchor: Anchor point for ``x`` (``'left'``, ``'center'``,
            ``'right'``).
        slider_len: Fractional width of the slider track (0–1).
        y: Vertical position of the slider in paper coordinates.

    Returns:
        Slider dict suitable for ``fig.update_layout(sliders=[...])``.
    """
    steps: List[Dict[str, Any]] = []

    for offset in range(-30, 31, 1):
        updated_y     = [tr['orig_y'] + offset for tr in arrival_traces]
        trace_indices = [tr['trace_index'] for tr in arrival_traces]
        step_label    = offset
        steps.append({
            'label':  step_label,
            'method': 'restyle',
            'args':   [{'y': updated_y}, trace_indices],
        })

    return {
        'active': 30,
        'steps': steps,
        'currentvalue': {
            'prefix':   f'{label}: ',
            'font':     {'size': 11},
            'xanchor':  'left',
        },
        'pad':      {'t': 20, 'b': 5},
        'len':      slider_len,
        'x':        x,
        'xanchor':  x_anchor,
        'y':        y,
        'yanchor':  'top',
        'tickwidth': 0,
        'ticklen':   0,  # hide tick marks; label shows in currentvalue display
        'font':     {'size': 9, 'color': 'White'},
        'currentvalue': {
            'prefix': f'{label}: ',
            'font': {'size': 11, 'color': 'Black'},
            'xanchor': 'left',
        }
    }


# ---------------------------------------------------------------------------
# Signal duration / t_start computation
# ---------------------------------------------------------------------------

def _compute_t_start(df_signal: pd.DataFrame) -> pd.DataFrame:
    """Compute ``t_start``: seconds from ``Cycle_start`` to ``TS_start``.

    This is the floating-bar base offset used by ``_add_ring_bars``.  Handles
    both tz-aware Timestamp columns and float (Unix epoch) columns.  Values
    are clipped to 0 to absorb sub-second floating-point drift.

    Args:
        df_signal: DataFrame with ``TS_start`` and ``Cycle_start`` columns.

    Returns:
        Copy of ``df_signal`` with ``t_start`` (float, seconds ≥ 0) column
        added.
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
        df['t_start'] = (
            pd.to_datetime(ts) - pd.to_datetime(cs)
        ).dt.total_seconds().clip(lower=0.0)

    return df


def _compute_durations(df_signal: pd.DataFrame) -> pd.DataFrame:
    """Compute ``Duration`` for each signal-state row via a within-phase shift.

    ``Duration`` = next ``TS_start`` − current ``TS_start``, computed per
    phase ID.  The last event for each phase group receives ``NaN`` and will
    be dropped downstream.  Gap markers (Code == −1) must be excluded
    *before* calling this function to prevent spurious durations spanning a
    discontinuity.

    Args:
        df_signal: Signal events DataFrame with at minimum
            ``[TS_start, Code, ID, Cycle_start]``.

    Returns:
        Copy of ``df_signal`` with a ``Duration`` (float, seconds) column
        added.
    """
    df = df_signal.sort_values(['ID', 'Cycle_start', 'TS_start']).copy()

    ts = df['TS_start']
    # Group by (ID, Cycle_start) so the shift never crosses a cycle boundary.
    # Without Cycle_start in the groupby key, the last Green event of phase N
    # in cycle K receives the first event of phase N in cycle K+1 as its
    # _ts_end, producing a bar that spans the entire inter-cycle gap.
    df['_ts_end'] = df.groupby(['ID', 'Cycle_start'])['TS_start'].shift(-1)

    if pd.api.types.is_float_dtype(ts):
        df['Duration'] = (df['_ts_end'] - df['TS_start']).clip(lower=0.0)
    else:
        df['Duration'] = (
            pd.to_datetime(df['_ts_end']) - pd.to_datetime(df['TS_start'])
        ).dt.total_seconds().clip(lower=0.0)

    return df.drop(columns=['_ts_end'])


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _build_title(metadata: Dict[str, Any], suffix: str = '') -> str:
    """Construct a plot title from intersection metadata.

    Prefers the ``"{major_road_name} @ {minor_road_name}"`` form; falls back
    to ``intersection_name`` when one or both road names are absent.

    Args:
        metadata: Dict with road/intersection name keys.
        suffix: String appended after the location component, separated by
            ``' – '``.

    Returns:
        Formatted title string.
    """
    major = str(metadata.get('major_road_name', '') or '').strip()
    minor = str(metadata.get('minor_road_name', '') or '').strip()
    intx  = str(metadata.get('intersection_name', 'Intersection') or '').strip()

    location = f'{major} @ {minor}' if (major and minor) else intx
    return f'{location} – {suffix}' if suffix else location


def _validate_signal_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if required signal DataFrame columns are absent.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: Listing all missing column names.
    """
    required = {'TS_start', 'Code', 'ID', 'Cycle_start'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f'df_signal is missing required columns: {sorted(missing)}'
        )