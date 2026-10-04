"""
ATSPM Approach Volume Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: bins DataFrame + days DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/approach_volume.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .termination import _build_title
from ..analysis.approach_volume import COMBINED, PAIRS


def plot_approach_volume(
    bins_df: pd.DataFrame,
    days_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Build the Approach Volume chart.

    UDOT draws one subplot row per direction pair: each direction's hourly
    volume and the combined volume as step lines, and each direction's
    D-factor dashed on a 0–1 secondary axis.

    Args:
        bins_df: Binned volume DataFrame from :func:`approach_volume`.
        days_df: Summary days DataFrame from :func:`approach_volume`.
        metadata: Optional metadata dict with intersection details (road names, etc.).

    Returns:
        Plotly Figure with one subplot row per direction pair.
    """
    title = _build_title(metadata or {}, suffix="Approach Volume")

    if bins_df is None or bins_df.empty or "pair" not in bins_df.columns:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    # Pairs present in bins_df in PAIRS order (NB/SB, then EB/WB)
    pairs = [f"{p[0]}/{p[1]}" for p in PAIRS if f"{p[0]}/{p[1]}" in bins_df["pair"].values]
    if not pairs:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    n = len(pairs)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=pairs,
        specs=[[{"secondary_y": True}]] * n,
    )

    pair_to_row = {pair_str: idx + 1 for idx, pair_str in enumerate(pairs)}

    # Configure axes for each row
    for r in range(1, n + 1):
        fig.update_yaxes(title_text="Volume (veh/h)", row=r, col=1, secondary_y=False)
        fig.update_yaxes(
            title_text="Directional split",
            range=[0, 1],
            row=r,
            col=1,
            secondary_y=True,
        )
    fig.update_xaxes(title_text="Time", row=n, col=1)

    # Traces per pair
    for primary, opposing in PAIRS:
        pair_str = f"{primary}/{opposing}"
        if pair_str not in pair_to_row:
            continue
        r = pair_to_row[pair_str]

        # Configured directions of this pair
        configured_dirs = [
            d for d in (primary, opposing)
            if ((bins_df["pair"] == pair_str) & (bins_df["direction"] == d)).any()
        ]

        # 1. Volume traces for configured directions
        for d in configured_dirs:
            color = "red" if d == primary else "blue"
            sub = bins_df.loc[(bins_df["pair"] == pair_str) & (bins_df["direction"] == d)]
            trace_name = f"{d} Volume"
            fig.add_trace(
                go.Scatter(
                    x=sub["time"],
                    y=sub["vph"],
                    mode="lines",
                    line_shape="hv",
                    connectgaps=False,
                    name=trace_name,
                    line=dict(color=color),
                    hovertemplate=f"<b>{trace_name}</b><br>Time: %{{x}}<br>Volume: %{{y}} veh/h<extra></extra>",
                ),
                row=r,
                col=1,
                secondary_y=False,
            )

        # 2. Combined volume trace
        sub_comb = bins_df.loc[
            (bins_df["pair"] == pair_str) & (bins_df["direction"] == COMBINED)
        ]
        if not sub_comb.empty:
            trace_name = f"{pair_str} Combined"
            fig.add_trace(
                go.Scatter(
                    x=sub_comb["time"],
                    y=sub_comb["vph"],
                    mode="lines",
                    line_shape="hv",
                    connectgaps=False,
                    name=trace_name,
                    line=dict(color="green"),
                    hovertemplate=f"<b>{trace_name}</b><br>Time: %{{x}}<br>Volume: %{{y}} veh/h<extra></extra>",
                ),
                row=r,
                col=1,
                secondary_y=False,
            )

        # 3. D-factor traces: only when BOTH directions of the pair have rows
        if primary in configured_dirs and opposing in configured_dirs:
            for d in (primary, opposing):
                color = "red" if d == primary else "blue"
                sub_d = bins_df.loc[
                    (bins_df["pair"] == pair_str) & (bins_df["direction"] == d)
                ]
                trace_name = f"{d} D-Factor"
                fig.add_trace(
                    go.Scatter(
                        x=sub_d["time"],
                        y=sub_d["d_split"],
                        mode="lines",
                        line_shape="hv",
                        connectgaps=False,
                        name=trace_name,
                        line=dict(color=color, dash="dash"),
                        hovertemplate=f"<b>{trace_name}</b><br>Time: %{{x}}<br>D-Factor: %{{y:.3f}}<extra></extra>",
                    ),
                    row=r,
                    col=1,
                    secondary_y=True,
                )

    # Shading for peak hours: from days_df direction == "combined"
    if days_df is not None and not days_df.empty:
        comb_days = days_df.loc[
            (days_df["direction"] == COMBINED) & days_df["peak_start"].notna()
        ]
        for pair_val, peak_start, k_val in zip(
            comb_days["pair"],
            comb_days["peak_start"],
            comb_days["k_factor"],
        ):
            if pair_val not in pair_to_row:
                continue
            r = pair_to_row[pair_val]
            x0 = peak_start
            x1 = peak_start + pd.Timedelta(hours=1)
            hhmm = peak_start.strftime("%H:%M")
            if pd.notna(k_val):
                ann_text = f"Peak {hhmm} · K {float(k_val):.3f}"
            else:
                ann_text = f"Peak {hhmm}"

            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(128, 128, 128, 0.2)",
                layer="below",
                line_width=0,
                annotation_text=ann_text,
                annotation_position="top left",
                row=r,
                col=1,
            )

    fig.update_layout(title=title, template="plotly_white")
    return fig
