"""
ATSPM Phase Termination Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: flat events DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/termination.py

Gap Marker Rule:
    Rows where event_code == -1 are excluded from all
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
        datetime(e.timestamp, 'unixepoch', 'localtime')   AS timestamp,
        e.event_code                                      AS event_code,
        e.parameter                                       AS parameter,
        datetime(c.cycle_start, 'unixepoch', 'localtime') AS cycle_start
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

            timestamp  : datetime-like, event timestamp
            event_code : int, ATSPM event code
            parameter  : int/float, phase number
            cycle_start: datetime-like (used for gap isolation)

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
    # Changed: Require native column names
    _validate_columns(df_events, required=['timestamp', 'event_code', 'parameter'])

    title = _build_title(metadata, suffix='Phase Termination')

    # Build a working copy, excluding gap markers from all scatter logic.
    # Gap markers are only needed for the ped-pairing segment computation,
    # which reads from the original df_events below.
    # Changed: Native event_code and parameter column usage
    df = df_events[df_events['event_code'] != _GAP_CODE].copy()
    df['parameter'] = pd.to_numeric(df['parameter'], errors='coerce')
    df = df.dropna(subset=['parameter'])
    df['parameter'] = df['parameter'].astype(int)

    # -----------------------------------------------------------------------
    # Y-axis remapping: only phases present in the data, sequential positions
    # -----------------------------------------------------------------------
    active_phases: List[int] = sorted(
        df.loc[df['event_code'].isin(_TERM_CODES), 'parameter'].unique().tolist()
    )
    if not active_phases:
        # Fallback: any phase seen at all
        active_phases = sorted(df['parameter'].unique().tolist())

    phase_to_y: Dict[int, int] = {ph: idx + 1 for idx, ph in enumerate(active_phases)}

    df['_y'] = df['parameter'].map(phase_to_y)

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
        df_code = df[df['event_code'] == code].dropna(subset=['_y'])
        if df_code.empty:
            continue
        style = _TERM_STYLES[code]
        y_off = _Y_OFFSET[code]

        fig.add_trace(go.Scatter(
            x=df_code['timestamp'],
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
            customdata=df_code['parameter'],
            hovertemplate=(
                f"<b>{style['name']}</b><br>"
                "Phase: %{customdata}<br>"
                "Time: %{x}<extra></extra>"
            ),
        ))

    # -----------------------------------------------------------------------
    # Preempt markers (Code 105) – only if present
    # -----------------------------------------------------------------------
    df_pre = df[df['event_code'] == 105].dropna(subset=['_y'])
    if not df_pre.empty:
        style = _TERM_STYLES[105]
        fig.add_trace(go.Scatter(
            x=df_pre['timestamp'],
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
            customdata=df_pre['parameter'],
            hovertemplate=(
                "<b>Preempt</b><br>"
                "Phase: %{customdata}<br>"
                "Time: %{x}<extra></extra>"
            ),
        ))

    # -----------------------------------------------------------------------
    # Pedestrian service – actuated vs recall, gap-aware
    # -----------------------------------------------------------------------
    df_ped_actuated, df_ped_recall = _classify_ped_service(df_events)

    for kind, df_ped in (('ped_actuated', df_ped_actuated), ('ped_recall', df_ped_recall)):
        if df_ped.empty:
            continue

        df_ped = df_ped.copy()
        df_ped['parameter'] = pd.to_numeric(df_ped['parameter'], errors='coerce').dropna().astype(int)
        df_ped = df_ped.dropna(subset=['parameter'])
        df_ped['parameter'] = df_ped['parameter'].astype(int)
        df_ped['_y'] = df_ped['parameter'].map(phase_to_y)
        df_ped = df_ped.dropna(subset=['_y'])  

        if df_ped.empty:
            continue

        style = _TERM_STYLES[kind]
        fig.add_trace(go.Scatter(
            x=df_ped['timestamp'],
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
            customdata=df_ped['parameter'],
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
            fixedrange=True,  
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
        dragmode='pan',  
    )

    fig.update_xaxes(fixedrange=False)

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"df_events is missing required columns: {missing}"
        )


def _build_title(metadata: Dict[str, Any], suffix: str = '') -> str:
    """Construct a plot title from metadata."""
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
    """
    return (df['event_code'] == _GAP_CODE).cumsum().astype(np.int32)


def _classify_ped_service(
    df_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Code 21 (Ped Begin Service) rows into actuated vs recall.
    """
    df_all = df_events.loc[
        df_events['event_code'].isin([_GAP_CODE, 21, 45])
    ].copy()

    if df_all.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    df_all['_seg'] = _segment_id(df_all)

    df_p = df_all.loc[df_all['event_code'].isin([21, 45])].copy()

    if df_p.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_p['parameter'] = pd.to_numeric(df_p['parameter'], errors='coerce')
    df_p = df_p.dropna(subset=['parameter'])
    df_p['parameter'] = df_p['parameter'].astype(int)

    df_p = df_p.sort_values(['_seg', 'parameter', 'timestamp']).reset_index(drop=True)

    df_p['_is_21'] = (df_p['event_code'] == 21).astype(np.int8)
    df_p['_is_45'] = (df_p['event_code'] == 45).astype(np.int8)

    df_p['_call_cum'] = df_p.groupby(['_seg', 'parameter'])['_is_45'].cumsum()

    df_p['_svc_grp'] = (
        df_p.groupby(['_seg', 'parameter'])['_is_21']
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    df_p['_has_call'] = (
        df_p.groupby(['_seg', 'parameter', '_svc_grp'])['_is_45']
        .transform('sum') > 0
    )

    svc_rows = df_p[df_p['event_code'] == 21].copy()

    actuated_mask = svc_rows['_has_call']
    recall_mask   = ~svc_rows['_has_call']

    return (
        svc_rows.loc[actuated_mask, ['timestamp', 'event_code', 'parameter']].reset_index(drop=True),
        svc_rows.loc[recall_mask,   ['timestamp', 'event_code', 'parameter']].reset_index(drop=True),
    )


def _add_maxout_lines(
    fig: go.Figure,
    df: pd.DataFrame,
    phase_to_y: Dict[int, int],
    n_con: int = 10,
) -> None:
    """Add per-phase rolling max-out proportion lines to *fig*."""
    df_term = df[df['event_code'].isin([4, 5, 6])].copy()
    if df_term.empty:
        return

    df_term = df_term.sort_values(['parameter', 'timestamp'])
    df_term['is_maxout'] = (df_term['event_code'] == 5).astype(float)

    x_range = [df_term['timestamp'].min(), df_term['timestamp'].max()]
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

    for phase_id, grp in df_term.groupby('parameter'):
        y_pos = phase_to_y.get(int(phase_id))
        if y_pos is None:
            continue
        grp = grp.sort_values('timestamp')
        rolling_mean = grp['is_maxout'].rolling(n_con, center=True, min_periods=1).mean()
        y_vals = rolling_mean * (2 / 3) + y_pos - (1 / 3)
        fig.add_trace(go.Scatter(
            x=grp['timestamp'],
            y=y_vals,
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip',
            name=f'_maxout_line_{phase_id}',
        ))