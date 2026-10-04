"""
ATSPM Left Turn Gap Analysis Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: bins DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/left_turn_gap.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .termination import _build_title
from ..analysis.left_turn_gap import (
    DEFAULT_EDGES,
    DEFAULT_TREND_S,
    bin_columns,
    bin_labels,
)

_LEFT_ORDER = ("NBL", "SBL", "EBL", "WBL")

_DEFAULT_COLORS = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
)


def plot_left_turn_gap(
    bins_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    edges: Sequence[float] = DEFAULT_EDGES,
    trend_s: float = DEFAULT_TREND_S,
) -> go.Figure:
    """Build the Left Turn Gap Analysis chart.

    UDOT draws one subplot row per left turn: the gap counts per time bin as
    stacked columns, one colour per gap bin, and '% of green time where gaps >=
    {trend_s:g}s' as a dashed line on a 0-100 secondary axis.

    Args:
        bins_df: Binned summary DataFrame from :func:`summarize_left_turn_gaps`.
        metadata: Optional metadata dict with intersection details (road names, etc.).
        edges: Gap bin edges in seconds.
        trend_s: Turnable gap threshold in seconds for the secondary trend line.

    Returns:
        Plotly Figure with one subplot row per left turn.
    """
    title = _build_title(metadata or {}, suffix="Left Turn Gap Analysis")

    if bins_df is None or bins_df.empty or "left" not in bins_df.columns:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    lefts = [l for l in _LEFT_ORDER if l in bins_df["left"].values]
    if not lefts:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    n = len(lefts)
    sub_titles = []
    for l in lefts:
        ph_vals = bins_df.loc[bins_df["left"] == l, "opposing_phase"].dropna()
        ph_str = f"Ph{int(ph_vals.iloc[0])}" if not ph_vals.empty else "unknown"
        sub_titles.append(f"{l} across opposing {ph_str}")

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=sub_titles,
        specs=[[{"secondary_y": True}]] * n,
    )

    cols = bin_columns(edges)
    labels = bin_labels(edges)
    colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(len(cols))]
    line_name = f"% green with gaps ≥ {trend_s:g}s"

    for idx, l in enumerate(lefts):
        r = idx + 1
        sub = bins_df.loc[bins_df["left"] == l]

        # Stacked bar per gap bin
        for col, label, color in zip(cols, labels, colors):
            fig.add_trace(
                go.Bar(
                    x=sub["time"],
                    y=sub[col],
                    name=label,
                    legendgroup=label,
                    showlegend=(idx == 0),
                    marker=dict(color=color),
                    hovertemplate=f"<b>{label}</b><br>Time: %{{x}}<br>Gaps: %{{y}}<extra></extra>",
                ),
                row=r,
                col=1,
                secondary_y=False,
            )

        # Trend line on secondary axis
        fig.add_trace(
            go.Scatter(
                x=sub["time"],
                y=sub["pct_turnable"],
                mode="lines",
                line_shape="hv",
                connectgaps=False,
                name=line_name,
                legendgroup=line_name,
                showlegend=(idx == 0),
                line=dict(dash="dash", color="#333333"),
                hovertemplate=f"<b>{line_name}</b><br>Time: %{{x}}<br>%: %{{y:.1f}}%<extra></extra>",
            ),
            row=r,
            col=1,
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Gaps", row=r, col=1, secondary_y=False)
        fig.update_yaxes(
            title_text=line_name,
            range=[0, 100],
            row=r,
            col=1,
            secondary_y=True,
        )

    fig.update_xaxes(title_text="Time", row=n, col=1)
    fig.update_layout(
        title=title,
        barmode="stack",
        template="plotly_white",
    )

    return fig
