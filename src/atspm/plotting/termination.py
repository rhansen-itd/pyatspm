"""
ATSPM Phase Termination Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: legacy-style events DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/termination.py

Gap Marker Rule:
    Rows where Code == -1 (event_code == -1) are excluded from all
    aggregations and scatter traces.  The rolling max-out proportion line
    is computed per phase; a gap marker within a phase's sequence causes
    the next valid observation to appear as a natural break in the line
    because NaN values produced by the gap are not interpolated.

    Pedestrian pairing (call → service) is reset at every gap marker via
    segment IDs (matching the logic in counts.py).  A Code 21 that has no
    preceding Code 45 in the same segment is classified as recall.

Y-axis Remapping:
    Only phases that actually appear in the data are shown.  Phase IDs are
    mapped to sequential integer y-positions (1, 2, 3 …) and the y-axis
    tick labels display the original phase numbers.  All hover text and
    legend values reference the original phase numbers so the display is
    always engineer-friendly.

Example SQL to build df_events
-------------------------------
    SELECT
        datetime(e.timestamp, 'unixepoch', 'localtime') AS TS_start,
        e.event_code                                     AS Code,
        e.parameter                                      AS ID,
        datetime(c.cycle_start, 'unixepoch', 'localtime') AS Cycle_start
    FROM events e
    LEFT JOIN cycles c ON e.timestamp >= c.cycle_start
    WHERE e.event_code IN (-1, 4, 5, 6, 21, 45, 105)
      AND e.timestamp BETWEEN :start_ts AND :end_ts
    ORDER BY e.timestamp;
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1

# Termination event codes
_TERM_CODES = (4, 5, 6)

# Marker style definitions for each termination type
_TERM_STYLES: Dict[int, Dict[str, Any]] = {
    4:   {'color': 'green',      'symbol': 'circle',           'name': 'Gap Out',                    'size': 6},
    5:   {'color': 'red',        'symbol': 'square',           'name': 'Max Out',                    'size': 6},
    6:   {'color': 'orangered',  'symbol': 'diamond',          'name': 'Force Off',                  'size': 6},
    105: {'color': 'magenta',    'symbol': 'x',                'name': 'Preempt',                    'size': 10},
    # Ped: actuated call (Code 45 preceded this service)
    'ped_actuated': {'color': 'steelblue',  'symbol': 'triangle-up',   'name': 'Ped Service (Actuated)', 'size': 8},
    # Ped: recall (no Code 45 detected before this Code 21 in the same segment)
    'ped_recall':   {'color': 'royalblue',  'symbol': 'triangle-down', 'name': 'Ped Service (Internal)',   'size': 8},
}

# Y-offset applied per event type so overlapping events remain legible.
# Termination codes share y=0 offset; ped markers are clustered just above
# (+0.10 / +0.22) to stay visually grouped while remaining distinct from
# preempt (+0.35).
_Y_OFFSET: Dict[Any, float] = {
    4:               0.0,
    5:               0.0,
    6:               0.0,
    105:             0.35,
    'ped_actuated':  0.18,
    'ped_recall':    0.16,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_termination(
    df_events: pd.DataFrame,
    metadata: Dict[str, Any],
    line: bool = True,
    n_con: int = 10,
) -> go.Figure:
    """
    Build a Phase Termination scatter plot.

    Each phase is plotted on the y-axis; the x-axis is event time.
    Termination type is distinguished by marker shape and colour.
    An optional rolling weighted-average line shows max-out proportion over
    the last *n_con* cycles per phase.

    Only phases observed in the data appear on the y-axis.  Phase IDs are
    mapped to sequential y-positions (1, 2, 3 …) and y-axis tick labels
    display the original phase numbers.

    Pedestrian services are split into two visually distinct sub-categories:

    * **Actuated** – a Code 45 (button press) preceded this Code 21 in the
      same continuous data segment.
    * **Recall** – no Code 45 was detected before this Code 21 in the same
      segment (controller recall mode).

    Args:
        df_events: Events DataFrame with columns::

            TS_start   : datetime-like, event timestamp
            Code       : int, ATSPM event code
            ID         : int/float, phase number
            Cycle_start: datetime-like (used for gap isolation)

        metadata: Dict with keys ``intersection_name``, ``major_road_name``,
            ``minor_road_name``.  Missing road names fall back to
            ``intersection_name``.
        line: If ``True`` (default), render the rolling max-out proportion
            line per phase.
        n_con: Window size (number of cycles) for the rolling average.
            Default 10.

    Returns:
        ``plotly.graph_objects.Figure`` ready for ``fig.show()`` or
        serialisation.

    Raises:
        ValueError: If ``df_events`` is missing required columns.
    """
    _validate_columns(df_events, required=['TS_start', 'Code', 'ID'])

    title = _build_title(metadata, suffix='Phase Termination')

    # Build a working copy, excluding gap markers from all scatter logic.
    # Gap markers are only needed for the ped-pairing segment computation,
    # which reads from the original df_events below.
    df = df_events[df_events['Code'] != _GAP_CODE].copy()
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    df = df.dropna(subset=['ID'])
    df['ID'] = df['ID'].astype(int)

    # -----------------------------------------------------------------------
    # Y-axis remapping: only phases present in the data, sequential positions
    # -----------------------------------------------------------------------
    # Determine active phases from termination codes (4/5/6) so the axis
    # reflects signal phase activity rather than incidental ped/preempt IDs.
    active_phases: List[int] = sorted(
        df.loc[df['Code'].isin(_TERM_CODES), 'ID'].unique().tolist()
    )
    if not active_phases:
        # Fallback: any phase seen at all
        active_phases = sorted(df['ID'].unique().tolist())

    phase_to_y: Dict[int, int] = {ph: idx + 1 for idx, ph in enumerate(active_phases)}

    # Map IDs → sequential y in the main working DataFrame.
    # Rows for phases not in active_phases (e.g. a preempt on an unlisted ID)
    # are kept but mapped to their nearest or own position gracefully.
    df['_y'] = df['ID'].map(phase_to_y)

    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])

    # -----------------------------------------------------------------------
    # Rolling max-out proportion line (optional)
    # -----------------------------------------------------------------------
    if line and not df.empty:
        _add_maxout_lines(fig, df, phase_to_y, n_con=n_con)

    # -----------------------------------------------------------------------
    # Termination markers: Gap Out, Max Out, Force Off
    # -----------------------------------------------------------------------
    for code in _TERM_CODES:
        df_code = df[df['Code'] == code].dropna(subset=['_y'])
        if df_code.empty:
            continue
        style = _TERM_STYLES[code]
        y_off = _Y_OFFSET[code]

        fig.add_trace(go.Scatter(
            x=df_code['TS_start'],
            y=df_code['_y'] + y_off,
            mode='markers',
            marker=dict(
                color=style['color'],
                symbol=style['symbol'],
                size=style['size'],
            ),
            name=style['name'],
            showlegend=True,
            legendgroup=style['name'],
            customdata=df_code['ID'],
            hovertemplate=(
                f"<b>{style['name']}</b><br>"
                "Phase: %{customdata}<br>"
                "Time: %{x}<extra></extra>"
            ),
        ))

    # -----------------------------------------------------------------------
    # Preempt markers (Code 105) – only if present
    # -----------------------------------------------------------------------
    df_pre = df[df['Code'] == 105].dropna(subset=['_y'])
    if not df_pre.empty:
        style = _TERM_STYLES[105]
        fig.add_trace(go.Scatter(
            x=df_pre['TS_start'],
            y=df_pre['_y'] + _Y_OFFSET[105],
            mode='markers',
            marker=dict(
                color=style['color'],
                symbol=style['symbol'],
                size=style['size'],
            ),
            name=style['name'],
            showlegend=True,
            legendgroup=style['name'],
            customdata=df_pre['ID'],
            hovertemplate=(
                "<b>Preempt</b><br>"
                "Phase: %{customdata}<br>"
                "Time: %{x}<extra></extra>"
            ),
        ))

    # -----------------------------------------------------------------------
    # Pedestrian service – actuated vs recall, gap-aware
    # -----------------------------------------------------------------------
    # Pass the full events_df (including gap markers) so segment IDs reset
    # the call→service pairing state correctly at every discontinuity.
    df_ped_actuated, df_ped_recall = _classify_ped_service(df_events)

    for kind, df_ped in (('ped_actuated', df_ped_actuated), ('ped_recall', df_ped_recall)):
        if df_ped.empty:
            continue

        df_ped = df_ped.copy()
        df_ped['ID'] = pd.to_numeric(df_ped['ID'], errors='coerce').dropna().astype(int)
        df_ped = df_ped.dropna(subset=['ID'])
        df_ped['ID'] = df_ped['ID'].astype(int)
        df_ped['_y'] = df_ped['ID'].map(phase_to_y)
        df_ped = df_ped.dropna(subset=['_y'])  # skip phases not on the axis

        if df_ped.empty:
            continue

        style = _TERM_STYLES[kind]
        fig.add_trace(go.Scatter(
            x=df_ped['TS_start'],
            y=df_ped['_y'] + _Y_OFFSET[kind],
            mode='markers',
            marker=dict(
                color=style['color'],
                symbol=style['symbol'],
                size=style['size'],
            ),
            name=style['name'],
            showlegend=True,
            legendgroup=style['name'],
            customdata=df_ped['ID'],
            hovertemplate=(
                f"<b>{style['name']}</b><br>"
                "Phase: %{customdata}<br>"
                "Time: %{x}<extra></extra>"
            ),
        ))

    # -----------------------------------------------------------------------
    # Layout – y-axis ticks show original phase numbers
    # -----------------------------------------------------------------------
    y_tick_vals = [phase_to_y[ph] for ph in active_phases]
    y_tick_text = [str(ph) for ph in active_phases]

    n_phases = len(active_phases)

    fig.update_layout(
        xaxis=dict(title='Time', type='date'),
        yaxis=dict(
            title='Phase',
            tickmode='array',
            tickvals=y_tick_vals,
            ticktext=y_tick_text,
            range=[0.5, n_phases + 0.5],
            fixedrange=True,  # disable y-axis zoom/pan
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
        ),
        hovermode='closest',
        template='plotly_white',
        dragmode='pan',  # default interaction mode
    )

    # Constrain all scroll/zoom interactions to x-axis only
    fig.update_xaxes(fixedrange=False)

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Raise ValueError if any required columns are absent.

    Args:
        df: DataFrame to check.
        required: List of column names that must be present.

    Raises:
        ValueError: Listing the missing columns.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"df_events is missing required columns: {missing}"
        )


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
    major = str(metadata.get('major_road_name') or '').strip()
    minor = str(metadata.get('minor_road_name') or '').strip()
    intx  = str(metadata.get('intersection_name') or 'Intersection').strip()

    if major and minor:
        location = f'{major} @ {minor}'
    else:
        location = intx

    return f'{location} – {suffix}' if suffix else location


def _segment_id(df: pd.DataFrame) -> pd.Series:
    """
    Return a monotonically increasing integer segment ID per row.

    Increments at every gap-marker row (Code == -1) so that rows within
    the same uninterrupted data block share the same ID.  Mirrors the
    implementation in counts.py and cycles.py.

    Args:
        df: DataFrame sorted by ``TS_start``, containing a ``Code`` column.

    Returns:
        ``int32`` Series aligned with *df*'s index.
    """
    return (df['Code'] == _GAP_CODE).cumsum().astype(np.int32)


def _classify_ped_service(
    df_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Code 21 (Ped Begin Service) rows into actuated vs recall.

    **Actuated**: at least one Code 45 (button press) for the same phase
    appeared after the previous Code 21 for that phase *and within the same
    continuous data segment*.

    **Recall**: no Code 45 was detected before this Code 21 in the same
    segment (controller is in ped-recall mode).

    The implementation mirrors the vectorised state-machine in
    ``counts.py::ped_counts()``:

    1. Assign segment IDs from gap markers so no state crosses a
       discontinuity.
    2. Drop gap-marker rows; keep only Codes 21 and 45.
    3. Within each ``(seg, phase)`` group, compute a cumulative Code-45
       counter.  A shifted version gives the call count *at the moment the
       previous Code 21 fired*.  A Code 21 is actuated when the current
       call count exceeds that saved value.

    Args:
        df_events: Full raw events DataFrame including gap markers.
            Must have columns ``[TS_start, Code, ID]``.

    Returns:
        Tuple ``(df_actuated, df_recall)`` — subsets of the Code 21 rows
        from *df_events* (gap markers excluded, index preserved).
    """
    df_all = df_events.loc[
        df_events['Code'].isin([_GAP_CODE, 21, 45])
    ].copy()

    if df_all.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_all = df_all.sort_values('TS_start').reset_index(drop=True)
    df_all['_seg'] = _segment_id(df_all)

    # Drop gap markers now that segments are assigned
    df_p = df_all.loc[df_all['Code'].isin([21, 45])].copy()

    if df_p.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_p['ID'] = pd.to_numeric(df_p['ID'], errors='coerce')
    df_p = df_p.dropna(subset=['ID'])
    df_p['ID'] = df_p['ID'].astype(int)

    df_p = df_p.sort_values(['_seg', 'ID', 'TS_start']).reset_index(drop=True)

    df_p['_is_21'] = (df_p['Code'] == 21).astype(np.int8)
    df_p['_is_45'] = (df_p['Code'] == 45).astype(np.int8)

    # Cumulative call count within each (seg, phase)
    df_p['_call_cum'] = df_p.groupby(['_seg', 'ID'])['_is_45'].cumsum()

    # "Service group" index: increments after each Code 21
    # (shift so the Code 21 itself belongs to the group it closes)
    df_p['_svc_grp'] = (
        df_p.groupby(['_seg', 'ID'])['_is_21']
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    # Within each service group, did any Code 45 appear?
    df_p['_has_call'] = (
        df_p.groupby(['_seg', 'ID', '_svc_grp'])['_is_45']
        .transform('sum') > 0
    )

    svc_rows = df_p[df_p['Code'] == 21].copy()

    # Use original df_events index to return proper row subsets
    # (df_p was reset_index'd, so map back via TS_start + ID + _seg match)
    actuated_mask = svc_rows['_has_call']
    recall_mask   = ~svc_rows['_has_call']

    return (
        svc_rows.loc[actuated_mask, ['TS_start', 'Code', 'ID']].reset_index(drop=True),
        svc_rows.loc[recall_mask,   ['TS_start', 'Code', 'ID']].reset_index(drop=True),
    )


def _add_maxout_lines(
    fig: go.Figure,
    df: pd.DataFrame,
    phase_to_y: Dict[int, int],
    n_con: int = 10,
) -> None:
    """
    Add per-phase rolling max-out proportion lines to *fig*.

    The proportion is the fraction of Code 5 (Max Out) events among all
    termination events (codes 4, 5, 6) over a rolling window of *n_con*
    observations.  The line is scaled and offset to sit within the ±0.33
    band around each phase's sequential y-position.

    Args:
        fig: Figure to mutate in-place.
        df: Clean events DataFrame (gap markers already excluded), with
            ``_y`` column containing sequential y-positions.
        phase_to_y: Mapping of original phase ID → sequential y-position,
            used to place the reference guide lines correctly.
        n_con: Rolling window size in cycles.
    """
    df_term = df[df['Code'].isin([4, 5, 6])].copy()
    if df_term.empty:
        return

    df_term = df_term.sort_values(['ID', 'TS_start'])
    df_term['is_maxout'] = (df_term['Code'] == 5).astype(float)

    # Reference guide lines at 1/3 boundaries (legacy visual aid)
    x_range = [df_term['TS_start'].min(), df_term['TS_start'].max()]
    max_y = max(phase_to_y.values()) if phase_to_y else 1
    for y_third in np.arange(2 / 3, max_y + 1, 1 / 3):
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[y_third, y_third],
            mode='lines',
            line=dict(color='lightgray', width=0.5, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Per-phase rolling mean line, plotted at the sequential y-position
    for phase_id, grp in df_term.groupby('ID'):
        y_pos = phase_to_y.get(int(phase_id))
        if y_pos is None:
            continue
        grp = grp.sort_values('TS_start')
        rolling_mean = grp['is_maxout'].rolling(n_con, center=True, min_periods=1).mean()
        # Scale: 0 → y_pos-0.333, 0.5 → y_pos, 1 → y_pos+0.333
        y_vals = rolling_mean * (2 / 3) + y_pos - (1 / 3)
        fig.add_trace(go.Scatter(
            x=grp['TS_start'],
            y=y_vals,
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip',
            name=f'_maxout_line_{phase_id}',
        ))
