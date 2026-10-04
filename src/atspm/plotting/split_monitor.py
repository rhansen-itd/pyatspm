"""
ATSPM Split Monitor Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: cycle DataFrame + timeline DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/split_monitor.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .termination import _TERM_STYLES, _build_title

_UNKNOWN_COLOR = "grey"

_TERM_CONFIG = {
    "gap_out": {
        "name": "Gap Out",
        "color": _TERM_STYLES[4]["color"],
        "symbol": _TERM_STYLES[4]["symbol"],
    },
    "max_out": {
        "name": "Max Out",
        "color": _TERM_STYLES[5]["color"],
        "symbol": _TERM_STYLES[5]["symbol"],
    },
    "force_off": {
        "name": "Force Off",
        "color": _TERM_STYLES[6]["color"],
        "symbol": _TERM_STYLES[6]["symbol"],
    },
    "unknown": {
        "name": "Unknown",
        "color": _UNKNOWN_COLOR,
        "symbol": "circle",
    },
}


def plot_split_monitor(
    cycle_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Build the UDOT split-monitor chart.

    Args:
        cycle_df: Output of :func:`split_monitor`, one row per phase service.
        timeline_df: Output of :func:`plan_timeline`, programmed state timeline.
        metadata: Optional metadata dict with intersection details (road names, etc.).

    Returns:
        Plotly Figure with one subplot row per phase.
    """
    title = _build_title(metadata or {}, suffix="Split Monitor")

    if cycle_df is None or cycle_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    phases = sorted(cycle_df["phase"].unique().tolist())
    n = len(phases)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
    )

    seen_legendgroups: Set[str] = set()

    for idx, ph in enumerate(phases):
        r = idx + 1
        df_ph = cycle_df.loc[cycle_df["phase"] == ph]

        # 1. Service markers
        for term_key, term_cfg in _TERM_CONFIG.items():
            sub_term = df_ph.loc[df_ph["termination"] == term_key]
            if sub_term.empty:
                continue

            trace_name = f"Ph{ph} {term_cfg['name']}"
            show_leg = term_key not in seen_legendgroups
            seen_legendgroups.add(term_key)

            custom = np.column_stack([
                sub_term["green_ts"].astype(str),
                sub_term["split_dur"],
                sub_term["programmed_split"],
                sub_term["plan"],
            ])

            fig.add_trace(
                go.Scatter(
                    x=sub_term["green_ts"],
                    y=sub_term["split_dur"],
                    mode="markers",
                    name=trace_name,
                    marker=dict(
                        color=term_cfg["color"],
                        symbol=term_cfg["symbol"],
                        size=6,
                    ),
                    legendgroup=term_key,
                    showlegend=show_leg,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{trace_name}</b><br>"
                        "Green: %{customdata[0]}<br>"
                        "Split: %{y:.1f} s<br>"
                        "Programmed: %{customdata[2]}<br>"
                        "Plan: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=r,
                col=1,
            )

        # 2. Ped walk markers
        if "ped_walk" in df_ph.columns:
            sub_walk = df_ph.loc[df_ph["ped_walk"].astype(bool)]
            if not sub_walk.empty:
                trace_name = f"Ph{ph} Ped Walk"
                show_leg = "ped_walk" not in seen_legendgroups
                seen_legendgroups.add("ped_walk")

                fig.add_trace(
                    go.Scatter(
                        x=sub_walk["green_ts"],
                        y=sub_walk["split_dur"],
                        mode="markers",
                        name=trace_name,
                        marker=dict(
                            symbol="circle-open",
                            size=8,
                            color="blue",
                            line=dict(width=1.5, color="blue"),
                        ),
                        legendgroup="ped_walk",
                        showlegend=show_leg,
                        hovertemplate=(
                            f"<b>{trace_name}</b><br>"
                            "Green: %{x}<br>"
                            "Split: %{y:.1f} s<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=1,
                )

        # 3. Programmed split line
        split_col = f"split_{ph}"
        if (
            timeline_df is not None
            and not timeline_df.empty
            and "cycle" in timeline_df.columns
            and split_col in timeline_df.columns
        ):
            valid_mask = (
                timeline_df["cycle"].notna()
                & (timeline_df["cycle"] > 0)
                & timeline_df[split_col].notna()
                & (timeline_df[split_col] > 0)
            )
            valid_tl = timeline_df.loc[valid_mask]
            if not valid_tl.empty:
                m = len(valid_tl)
                starts = valid_tl["start"].to_numpy()
                ends = valid_tl["end"].to_numpy()
                splits = valid_tl[split_col].to_numpy()

                x_arr = np.empty(3 * m, dtype=object)
                x_arr[0::3] = starts
                x_arr[1::3] = ends
                x_arr[2::3] = None

                y_arr = np.empty(3 * m, dtype=object)
                y_arr[0::3] = splits
                y_arr[1::3] = splits
                y_arr[2::3] = None

                trace_name = f"Ph{ph} Programmed"
                show_leg = "programmed" not in seen_legendgroups
                seen_legendgroups.add("programmed")

                fig.add_trace(
                    go.Scatter(
                        x=list(x_arr),
                        y=list(y_arr),
                        mode="lines",
                        name=trace_name,
                        line_shape="hv",
                        line=dict(shape="hv", color="black", width=1.5),
                        legendgroup="programmed",
                        showlegend=show_leg,
                        hovertemplate=(
                            f"<b>{trace_name}</b><br>"
                            "Time: %{x}<br>"
                            "Programmed Split: %{y} s<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=1,
                )

        fig.update_yaxes(title_text="Split (s)", row=r, col=1)

    # 4. Plan bands
    if (
        timeline_df is not None
        and not timeline_df.empty
        and "plan" in timeline_df.columns
        and not cycle_df.empty
    ):
        x_min = cycle_df["green_ts"].min()
        x_max = cycle_df["green_ts"].max()

        plan_series = timeline_df["plan"]
        plan_changed = (plan_series != plan_series.shift()) | (
            plan_series.isna() != plan_series.shift().isna()
        )
        run_ids = plan_changed.cumsum()

        fill_colors = ["rgba(0, 0, 0, 0.02)", "rgba(0, 0, 0, 0.07)"]
        band_idx = 0

        for _, run_df in timeline_df.groupby(run_ids, sort=False):
            first_row = run_df.iloc[0]
            plan_val = first_row["plan"]
            if pd.isna(plan_val):
                continue

            run_start = first_row["start"]
            run_end = run_df.iloc[-1]["end"]

            # Skip bands entirely outside the x-range
            if run_end <= x_min or run_start >= x_max:
                continue

            band_start = max(run_start, x_min)
            band_end = min(run_end, x_max)
            if band_start >= band_end:
                continue

            is_free = (
                "cycle" in first_row
                and pd.notna(first_row["cycle"])
                and first_row["cycle"] == 0
            )
            ann_text = f"Plan {int(plan_val)}" + (" (free)" if is_free else "")

            fill = fill_colors[band_idx % len(fill_colors)]
            fig.add_vrect(
                x0=band_start,
                x1=band_end,
                fillcolor=fill,
                layer="below",
                line_width=0,
                annotation_text=ann_text,
                annotation_position="top left",
            )
            band_idx += 1

    fig.update_layout(title=title, template="plotly_white")
    return fig
