"""
ATSPM Green Time Utilization Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: bins DataFrame + splits DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/green_time_utilization.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .termination import _build_title


def plot_green_time(
    bins_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    max_green_s: Optional[float] = 120.0,
) -> go.Figure:
    """Build the UDOT Green Time Utilization heat map chart.

    Args:
        bins_df: Binned actuations DataFrame from :func:`summarize_gtu_bins`.
        splits_df: Summary splits DataFrame from :func:`summarize_gtu_splits`.
        metadata: Optional metadata dict with intersection details (road names, etc.).
        max_green_s: Green duration threshold in seconds to cap plot heatmap bins.
            Default 120.0. When None, all bins are kept.

    Returns:
        Plotly Figure with one subplot row per phase.
    """
    title = _build_title(metadata or {}, suffix="Green Time Utilization")

    if bins_df is None or bins_df.empty or "phase" not in bins_df.columns:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    phases = sorted(bins_df["phase"].dropna().unique().tolist())
    n = len(phases)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Ph{ph}" for ph in phases],
    )

    seen_legendgroups: Set[str] = set()

    for idx, ph in enumerate(phases):
        r = idx + 1
        df_ph = bins_df.loc[bins_df["phase"] == ph]

        if max_green_s is not None:
            df_ph = df_ph.loc[df_ph["bin_start_s"] < max_green_s]

        # 1. Heat map: Ph{N} Utilization
        if not df_ph.empty:
            piv = df_ph.pivot(index="bin_start_s", columns="time", values="act_per_cycle")
            piv = piv.sort_index().sort_index(axis=1)
            piv_reached = df_ph.pivot(
                index="bin_start_s", columns="time", values="n_reached"
            ).reindex(index=piv.index, columns=piv.columns)
            piv_flow = df_ph.pivot(
                index="bin_start_s", columns="time", values="flow_vph"
            ).reindex(index=piv.index, columns=piv.columns)
            custom = np.dstack([piv_reached.to_numpy(), piv_flow.to_numpy()])

            fig.add_trace(
                go.Heatmap(
                    x=piv.columns,
                    y=piv.index,
                    z=piv.to_numpy(dtype=float),
                    coloraxis="coloraxis",
                    name=f"Ph{ph} Utilization",
                    customdata=custom,
                    hovertemplate=(
                        f"<b>Ph{ph} Utilization</b><br>"
                        "Time: %{x}<br>"
                        "Bin start: %{y} s<br>"
                        "Actuations / cycle: %{z:.2f}<br>"
                        "Cycles reaching bin: %{customdata[0]}<br>"
                        "Flow: %{customdata[1]:.1f} veh/h<extra></extra>"
                    ),
                ),
                row=r,
                col=1,
            )

        # 2. Step lines from splits_df
        if splits_df is not None and not splits_df.empty and "phase" in splits_df.columns:
            sp_ph = splits_df.loc[splits_df["phase"] == ph].sort_values("time")

            # Average Green
            sp_avg = sp_ph.dropna(subset=["avg_green_s"])
            if not sp_avg.empty:
                show_leg = "Average Green" not in seen_legendgroups
                seen_legendgroups.add("Average Green")
                fig.add_trace(
                    go.Scatter(
                        x=sp_avg["time"],
                        y=sp_avg["avg_green_s"],
                        mode="lines",
                        line_shape="hv",
                        name=f"Ph{ph} Average Green",
                        line=dict(color="#1f77b4", width=2),
                        legendgroup="Average Green",
                        showlegend=show_leg,
                        hovertemplate=(
                            f"<b>Ph{ph} Average Green</b><br>"
                            "Time: %{x}<br>"
                            "Average Green: %{y:.1f} s<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=1,
                )

            # Programmed Green
            sp_prog = sp_ph.dropna(subset=["programmed_green"])
            if not sp_prog.empty:
                show_leg = "Programmed Green" not in seen_legendgroups
                seen_legendgroups.add("Programmed Green")
                fig.add_trace(
                    go.Scatter(
                        x=sp_prog["time"],
                        y=sp_prog["programmed_green"],
                        mode="lines",
                        line_shape="hv",
                        name=f"Ph{ph} Programmed Green",
                        line=dict(color="#d62728", width=2, dash="dash"),
                        legendgroup="Programmed Green",
                        showlegend=show_leg,
                        hovertemplate=(
                            f"<b>Ph{ph} Programmed Green</b><br>"
                            "Time: %{x}<br>"
                            "Programmed Green: %{y:.1f} s<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=1,
                )

        fig.update_yaxes(title_text="Seconds into green", row=r, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        coloraxis=dict(
            colorscale="Viridis",
            cmin=0,
            colorbar=dict(title="Actuations / cycle"),
        ),
    )
    return fig
