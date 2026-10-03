"""
ATSPM Split Failure Scatter Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: cycle DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/split_failures.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .termination import _build_title


def plot_split_failures(
    cycle_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    threshold: float = 0.79,
) -> go.Figure:
    """Build a Purdue Split Failure GOR vs ROR5 scatter plot.

    Args:
        cycle_df: Per-cycle split failure DataFrame containing columns:
            ``phase``, ``green_ts``, ``gor``, ``ror5``, ``n_lanes``,
            ``n_lanes_failed``, ``fail``, and optionally ``aggregate``.
        metadata: Optional metadata dictionary with intersection details.
        threshold: Occupancy ratio threshold guide line. Default 0.79.

    Returns:
        Plotly Figure with one subplot per phase.
    """
    metadata = metadata or {}

    aggregate = "union"
    if cycle_df is not None and not cycle_df.empty and "aggregate" in cycle_df.columns:
        aggregate = str(cycle_df["aggregate"].iloc[0])

    meta = dict(metadata) if metadata else {}
    if meta.get("major_road_name") and not meta.get("minor_road_name"):
        intx = meta.get("intersection_name")
        if intx and intx != meta["major_road_name"]:
            meta["intersection_name"] = f"{intx} ({meta['major_road_name']})"
        else:
            meta["intersection_name"] = str(meta["major_road_name"])

    title = _build_title(meta, suffix=f"Split Failures — GOR vs ROR5 ({aggregate})")

    if cycle_df is None or cycle_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    phases = sorted(cycle_df["phase"].dropna().unique())
    if not phases:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    n_phases = len(phases)
    cols = min(n_phases, 4)
    rows = (n_phases + cols - 1) // cols
    subplot_titles = [f"Phase {ph}" for ph in phases]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
    )

    for idx, ph in enumerate(phases):
        r = (idx // cols) + 1
        c = (idx % cols) + 1

        # Threshold guide trace with [a, b, None] segment pattern (no layout shapes)
        thr_trace = go.Scatter(
            x=[threshold, threshold, None, 0.0, 1.0],
            y=[0.0, 1.0, None, threshold, threshold],
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name="Threshold",
            legendgroup="Threshold",
            showlegend=(idx == 0),
            hoverinfo="skip",
        )
        fig.add_trace(thr_trace, row=r, col=c)

        df_ph = cycle_df.loc[cycle_df["phase"] == ph]

        # Pass and fail traces
        for is_fail, trace_suffix, color in [
            (False, "pass", "#2ca02c"),
            (True, "fail", "#d62728"),
        ]:
            sub = df_ph.loc[df_ph["fail"] == is_fail]
            if sub.empty:
                continue

            if pd.api.types.is_datetime64_any_dtype(sub["green_ts"]):
                time_str = sub["green_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = pd.to_datetime(sub["green_ts"], unit="s").dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            hover = [
                f"Time: {t}<br>GOR: {g:.3f}<br>ROR5: {r5:.3f}<br>Lanes: {nl}<br>Failed Lanes: {nlf}"
                for t, g, r5, nl, nlf in zip(
                    time_str,
                    sub["gor"],
                    sub["ror5"],
                    sub["n_lanes"],
                    sub["n_lanes_failed"],
                )
            ]

            fig.add_trace(
                go.Scatter(
                    x=sub["gor"],
                    y=sub["ror5"],
                    mode="markers",
                    name=f"Ph{ph} {trace_suffix}",
                    marker=dict(color=color, size=6),
                    hoverlabel=dict(bgcolor=color),
                    hoverinfo="text",
                    hovertext=hover,
                ),
                row=r,
                col=c,
            )

    fig.update_xaxes(range=[0, 1], title_text="Green Occupancy Ratio (GOR)")
    fig.update_yaxes(range=[0, 1], title_text="Red Occupancy Ratio (ROR5)")
    fig.update_layout(title=title, template="plotly_white")
    return fig
