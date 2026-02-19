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

Example SQL to build df_events
-------------------------------
    -- Legacy-style columns expected by plot_termination():
    --   TS_start (datetime), Code (int), ID (int), Cycle_start (datetime)
    --
    SELECT
        datetime(e.timestamp, 'unixepoch', 'localtime') AS TS_start,
        e.event_code                                     AS Code,
        e.parameter                                      AS ID,
        datetime(c.cycle_start, 'unixepoch', 'localtime') AS Cycle_start
    FROM events e
    LEFT JOIN (
        SELECT cycle_start, MIN(timestamp) OVER (
            ORDER BY cycle_start
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cycle_start
        FROM cycles
    ) c ON e.timestamp >= c.cycle_start
    WHERE e.event_code IN (-1, 4, 5, 6, 21, 45, 105)
      AND e.timestamp BETWEEN :start_ts AND :end_ts
    ORDER BY e.timestamp;
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1

# Marker style definitions for each termination type
_TERM_STYLES: Dict[int, Dict[str, Any]] = {
    4:   {'color': 'green',      'symbol': 'circle',       'name': 'Gap Out',          'size': 6},
    5:   {'color': 'red',        'symbol': 'square',       'name': 'Max Out',           'size': 6},
    6:   {'color': 'orangered',  'symbol': 'diamond',      'name': 'Force Off',         'size': 6},
    105: {'color': 'magenta',    'symbol': 'x',            'name': 'Preempt',           'size': 10},
    21:  {'color': 'blue',       'symbol': 'triangle-up',  'name': 'Pedestrian Service','size': 8},
}

# Y-offset applied per marker type so overlapping events remain legible
_Y_OFFSET: Dict[int, float] = {
    4:   0.0,
    5:   0.0,
    6:   0.0,
    105: 0.35,
    21:  0.15,
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

    Args:
        df_events: Events DataFrame with columns::

            TS_start   : datetime-like, event timestamp
            Code       : int, ATSPM event code
            ID         : int/float, phase number
            Cycle_start: datetime-like (optional but used for gap isolation)

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

    # Exclude gap markers from all plotting logic
    df = df_events[df_events['Code'] != _GAP_CODE].copy()
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    df = df.dropna(subset=['ID'])
    df['ID'] = df['ID'].astype(int)

    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])

    # -----------------------------------------------------------------------
    # Rolling max-out proportion line (optional)
    # -----------------------------------------------------------------------
    if line and not df.empty:
        _add_maxout_lines(fig, df, n_con=n_con)

    # -----------------------------------------------------------------------
    # Termination markers: Gap Out, Max Out, Force Off
    # -----------------------------------------------------------------------
    legend_added: set = set()
    for code in (4, 5, 6):
        df_code = df[df['Code'] == code]
        if df_code.empty:
            continue
        style = _TERM_STYLES[code]
        y_off = _Y_OFFSET[code]
        show_legend = code not in legend_added
        legend_added.add(code)

        fig.add_trace(go.Scatter(
            x=df_code['TS_start'],
            y=df_code['ID'] + y_off,
            mode='markers',
            marker=dict(
                color=style['color'],
                symbol=style['symbol'],
                size=style['size'],
            ),
            name=style['name'],
            showlegend=show_legend,
            legendgroup=style['name'],
            hovertemplate=(
                f"<b>{style['name']}</b><br>"
                "Phase: %{y}<br>"
                "Time: %{x}<extra></extra>"
            ),
        ))

    # -----------------------------------------------------------------------
    # Preempt markers (Code 105) – only if present
    # -----------------------------------------------------------------------
    df_pre = df[df['Code'] == 105]
    if not df_pre.empty:
        style = _TERM_STYLES[105]
        fig.add_trace(go.Scatter(
            x=df_pre['TS_start'],
            y=df_pre['ID'] + _Y_OFFSET[105],
            mode='markers',
            marker=dict(
                color=style['color'],
                symbol=style['symbol'],
                size=style['size'],
            ),
            name=style['name'],
            showlegend=True,
            legendgroup=style['name'],
            hovertemplate=(
                "<b>Preempt</b><br>"
                "Phase: %{y:.0f}<br>"
                "Time: %{x}<extra></extra>"
            ),
        ))

    # -----------------------------------------------------------------------
    # Pedestrian service (Code 21 preceded by Code 45 in same phase)
    # Replicates the legacy pairing logic from plot_term() – vectorized.
    # -----------------------------------------------------------------------
    df_ped = _extract_legitimate_ped_service(df)
    if not df_ped.empty:
        style = _TERM_STYLES[21]
        fig.add_trace(go.Scatter(
            x=df_ped['TS_start'],
            y=df_ped['ID'] + _Y_OFFSET[21],
            mode='markers',
            marker=dict(
                color=style['color'],
                symbol=style['symbol'],
                size=style['size'],
            ),
            name=style['name'],
            showlegend=True,
            legendgroup=style['name'],
            hovertemplate=(
                "<b>Pedestrian Service</b><br>"
                "Phase: %{y:.1f}<br>"
                "Time: %{x}<extra></extra>"
            ),
        ))

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    fig.update_layout(
        xaxis=dict(title='Time', type='date'),
        yaxis=dict(
            title='Phase',
            dtick=1,
            tickmode='linear',
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
    )

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


def _add_maxout_lines(
    fig: go.Figure,
    df: pd.DataFrame,
    n_con: int = 10,
) -> None:
    """
    Add per-phase rolling max-out proportion lines to *fig*.

    The proportion is the fraction of Code 5 (Max Out) events among all
    termination events (codes 4, 5, 6) over a rolling window of *n_con*
    observations.  The line is scaled and offset to sit within the ±0.5
    band around each phase's integer y-value (matching legacy behaviour).

    Gap marker rows must already be removed from *df* before calling.

    The computation is fully vectorised:
    1. Filter to codes 4/5/6 only.
    2. Create binary column ``is_maxout`` (1 if code == 5, else 0).
    3. Per phase: rolling mean → scale to ±0.33 band → add phase offset.

    Args:
        fig: Figure to mutate in-place.
        df: Clean events DataFrame (gap markers already excluded).
        n_con: Rolling window size in cycles.
    """
    df_term = df[df['Code'].isin([4, 5, 6])].copy()
    if df_term.empty:
        return

    df_term = df_term.sort_values(['ID', 'TS_start'])
    df_term['is_maxout'] = (df_term['Code'] == 5).astype(float)

    # Reference horizontal lines at 1/3 boundaries (legacy guide lines)
    x_range = [df_term['TS_start'].min(), df_term['TS_start'].max()]
    max_phase = int(df_term['ID'].max())
    for y_third in np.arange(2 / 3, (max_phase + 1), 1 / 3):
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[y_third, y_third],
            mode='lines',
            line=dict(color='lightgray', width=0.5, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Per-phase rolling mean line
    for phase_id, grp in df_term.groupby('ID'):
        grp = grp.sort_values('TS_start')
        rolling_mean = grp['is_maxout'].rolling(n_con, center=True, min_periods=1).mean()
        # Scale: 0→phase-0.333, 0.5→phase, 1→phase+0.333 (matches legacy formula)
        y_vals = rolling_mean * (2 / 3) + phase_id - (1 / 3)
        fig.add_trace(go.Scatter(
            x=grp['TS_start'],
            y=y_vals,
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False,
            hoverinfo='skip',
            name=f'_maxout_line_{phase_id}',
        ))


def _extract_legitimate_ped_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return Code 21 (Ped Begin Service) rows that were legitimately preceded
    by a Code 45 (Ped Call) on the same phase, with no intervening Code 21.

    This replicates the legacy pairing logic from ``plot_term()``, rewritten
    as a vectorized operation using a cumulative sum state machine:

    1. Filter to codes 21 and 45.
    2. Sort by ``[ID, TS_start]``.
    3. Within each phase group, track whether a Code 45 has been seen since
       the last Code 21 using ``cumsum`` on Code 45 rows.
    4. A Code 21 is legitimate if the cumulative Code-45 count advanced at
       least once since the prior Code 21.

    Gap markers do not appear in *df* (they are stripped by the caller).

    Args:
        df: Clean events DataFrame with at minimum columns
            [TS_start, Code, ID].

    Returns:
        Subset of *df* containing only legitimate Code 21 rows.
    """
    df_ped = df[df['Code'].isin([21, 45])].copy()
    if df_ped.empty:
        return pd.DataFrame()

    df_ped = df_ped.sort_values(['ID', 'TS_start']).reset_index(drop=True)

    legit_indices: list[int] = []

    for _phase_id, grp in df_ped.groupby('ID', sort=True):
        # Cumulative count of Code 45 events seen so far in this phase
        grp = grp.reset_index()  # keep original index as 'index' column
        grp['_call_cumsum'] = (grp['Code'] == 45).cumsum()

        # For each Code 21 row: it is legitimate if at least one Code 45
        # appeared between the *previous* Code 21 and this one.
        # We track the last seen call_cumsum at a Code 21 boundary.
        last_call_at_21 = -1
        for _, row in grp.iterrows():
            if row['Code'] == 45:
                continue  # advance is already captured in cumsum
            if row['Code'] == 21:
                if row['_call_cumsum'] > last_call_at_21:
                    legit_indices.append(int(row['index']))
                last_call_at_21 = row['_call_cumsum']

    if not legit_indices:
        return pd.DataFrame()

    return df.loc[legit_indices]
