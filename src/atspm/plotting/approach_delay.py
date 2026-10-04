"""
ATSPM Approach Delay Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: binned DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/approach_delay.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .termination import _build_title


def plot_approach_delay(
    binned_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Build a UDOT Approach Delay chart.

    Args:
        binned_df: Binned approach delay DataFrame with columns:
            ``phase``, ``time``, ``coord_plan``, ``arrivals``,
            ``total_delay_s``, ``delay_per_veh``, ``delay_vh_per_hr``,
            and optionally ``n_cycles``.
        metadata: Optional metadata dictionary with intersection details.

    Returns:
        Plotly Figure with one subplot row per phase.
    """
    metadata = metadata or {}
    title = _build_title(metadata, suffix="Approach Delay")

    if binned_df is None or binned_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    phases = sorted(binned_df["phase"].dropna().unique().astype(int))
    if not phases:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    # 1. Recombine plans first
    # A bin that spans a plan change has one row per plan.
    # Group by (phase, time) before plotting:
    # - delay/veh = Σ total_delay_s / Σ arrivals, NaN when there are no arrivals
    # - veh-h/h = Σ delay_vh_per_hr
    if "n_cycles" in binned_df.columns:
        sorted_df = binned_df.sort_values(
            by=["phase", "time", "n_cycles"], ascending=[True, True, False]
        )
    else:
        sorted_df = binned_df.sort_values(by=["phase", "time"])
    plan_map = sorted_df.drop_duplicates(subset=["phase", "time"])[
        ["phase", "time", "coord_plan"]
    ]

    grouped = binned_df.groupby(["phase", "time"], as_index=False).agg(
        total_delay_s=("total_delay_s", "sum"),
        arrivals=("arrivals", "sum"),
        delay_vh_per_hr=("delay_vh_per_hr", "sum"),
    )
    arr = grouped["arrivals"].to_numpy(dtype=float)
    tot_delay = grouped["total_delay_s"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        grouped["delay_per_veh"] = np.where(arr > 0, tot_delay / arr, np.nan)
    grouped = grouped.merge(plan_map, on=["phase", "time"], how="left")
    grouped = grouped.sort_values(["phase", "time"]).reset_index(drop=True)

    n = len(phases)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}]] * n,
    )

    color_delay = "#1f77b4"
    color_rate = "#ff7f0e"

    for idx, ph in enumerate(phases):
        r = idx + 1
        sub = grouped.loc[grouped["phase"] == ph]

        fig.add_trace(
            go.Scatter(
                x=sub["time"],
                y=sub["delay_per_veh"],
                mode="lines+markers",
                name=f"Ph{ph} delay/veh",
                line=dict(color=color_delay),
                marker=dict(color=color_delay, size=6),
                hoverlabel=dict(bgcolor=color_delay),
            ),
            row=r,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=sub["time"],
                y=sub["delay_vh_per_hr"],
                mode="lines+markers",
                name=f"Ph{ph} veh-h/h",
                line=dict(color=color_rate),
                marker=dict(color=color_rate, size=6),
                hoverlabel=dict(bgcolor=color_rate),
            ),
            row=r,
            col=1,
            secondary_y=True,
        )

        fig.update_yaxes(
            title_text="Delay per vehicle (s)", row=r, col=1, secondary_y=False
        )
        fig.update_yaxes(
            title_text="Delay (veh-h/h)", row=r, col=1, secondary_y=True
        )

    # 2. Plan bands
    # Take runs of consecutive times with the same plan from the lowest-numbered phase.
    unique_times = (
        pd.Series(pd.to_datetime(binned_df["time"].unique()))
        .sort_values()
        .reset_index(drop=True)
    )
    if len(unique_times) > 1:
        diffs = unique_times.diff().dropna()
        pos_diffs = diffs[diffs > pd.Timedelta(0)]
        bin_width = pos_diffs.min() if not pos_diffs.empty else pd.Timedelta(minutes=15)
    else:
        bin_width = pd.Timedelta(minutes=15)

    lowest_phase = phases[0]
    df_lowest = grouped.loc[grouped["phase"] == lowest_phase].sort_values("time").reset_index(drop=True)

    if not df_lowest.empty and "coord_plan" in df_lowest.columns:
        dt_col = pd.to_datetime(df_lowest["time"])
        plan_col = df_lowest["coord_plan"]

        plan_changed = plan_col != plan_col.shift()
        time_gap = dt_col.diff() > bin_width
        run_id = (plan_changed | time_gap).cumsum()

        fill_colors = ["rgba(0, 0, 0, 0.02)", "rgba(0, 0, 0, 0.07)"]

        for i, (_, run_df) in enumerate(df_lowest.groupby(run_id, sort=False)):
            plan = run_df["coord_plan"].iloc[0]
            t_start = run_df["time"].iloc[0]
            dt_last = pd.to_datetime(run_df["time"].iloc[-1])
            t_end = dt_last + bin_width
            if getattr(t_start, "tzinfo", None) is not None:
                t_end = t_end.tz_convert(t_start.tz)

            fill = fill_colors[i % len(fill_colors)]
            ann_text = f"Plan {int(plan)}" if pd.notna(plan) else ""

            fig.add_vrect(
                x0=t_start,
                x1=t_end,
                fillcolor=fill,
                layer="below",
                line_width=0,
                annotation_text=ann_text,
                annotation_position="top left",
            )

    fig.update_layout(title=title, template="plotly_white")
    return fig
