"""
ATSPM Yellow and Red Actuations Plot (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: cycle DataFrame + actuations DataFrame + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/yellow_red_actuations.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .termination import _build_title
from ..analysis.yellow_red_actuations import DEFAULT_SEVERE_SEC

# Actuation category definitions: order, name, color, and filter key
_CATEGORIES = [
    ("Yellow", "#D48806"),        # amber
    ("Red Clearance", "#F57C00"), # orange
    ("Red", "#D32F2F"),           # red
    ("Severe", "#8B0000"),        # dark red
]


def plot_yellow_red(
    cycle_df: pd.DataFrame,
    act_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    severe_sec: float = DEFAULT_SEVERE_SEC,
) -> go.Figure:
    """Build the UDOT Yellow and Red Actuations chart.

    Args:
        cycle_df: Output cycle DataFrame from :func:`yellow_red_actuations`.
        act_df: Output actuation DataFrame from :func:`yellow_red_actuations`.
        metadata: Optional metadata dict with intersection details (road names, etc.).
        severe_sec: Severe violation threshold in seconds after red start. Default 4.0.

    Returns:
        Plotly Figure with one subplot row per phase.
    """
    title = _build_title(metadata or {}, suffix="Yellow and Red Actuations")

    if cycle_df is None or cycle_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    phases = sorted(cycle_df["phase"].dropna().unique().tolist())
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

        # 1. Actuation markers (Green actuations are NOT drawn)
        if act_df is not None and not act_df.empty and "phase" in act_df.columns:
            act_ph = act_df.loc[act_df["phase"] == ph]
            if not act_ph.empty:
                is_severe = act_ph["severe"].astype(bool)
                state = act_ph["state"]

                cat_subsets = {
                    "Yellow": act_ph.loc[(state == "yellow") & (~is_severe)],
                    "Red Clearance": act_ph.loc[(state == "red_clear") & (~is_severe)],
                    "Red": act_ph.loc[(state == "red") & (~is_severe)],
                    "Severe": act_ph.loc[is_severe],
                }

                for cat_name, cat_color in _CATEGORIES:
                    sub = cat_subsets[cat_name]
                    if sub.empty:
                        continue

                    trace_name = f"Ph{ph} {cat_name}"
                    show_leg = cat_name not in seen_legendgroups
                    seen_legendgroups.add(cat_name)

                    custom = np.column_stack([
                        sub["detector"].astype(str),
                        sub["state"].astype(str),
                        sub["t_red"].astype(float).round(2).astype(str),
                    ])

                    fig.add_trace(
                        go.Scatter(
                            x=sub["timestamp"],
                            y=sub["t_yellow"],
                            mode="markers",
                            name=trace_name,
                            marker=dict(
                                color=cat_color,
                                size=6,
                            ),
                            legendgroup=cat_name,
                            showlegend=show_leg,
                            customdata=custom,
                            hovertemplate=(
                                f"<b>{trace_name}</b><br>"
                                "Time: %{x}<br>"
                                "Detector: %{customdata[0]}<br>"
                                "State: %{customdata[1]}<br>"
                                "Time into red: %{customdata[2]} s<extra></extra>"
                            ),
                        ),
                        row=r,
                        col=1,
                    )

        # 2. Reference lines from uncensored cycles
        uncensored = df_ph.loc[~df_ph["censored"].astype(bool)]
        if not uncensored.empty:
            m = len(uncensored)
            starts = uncensored["yellow_ts"].to_numpy()
            ends = np.empty(m, dtype=object)
            if m > 1:
                ends[:-1] = uncensored["yellow_ts"].to_numpy()[1:]
            ends[-1] = uncensored["red_end_ts"].iloc[-1]

            x_arr = np.empty(3 * m, dtype=object)
            x_arr[0::3] = starts
            x_arr[1::3] = ends
            x_arr[2::3] = None

            ref_specs = [
                (
                    f"Ph{ph} Red Clearance Begin",
                    "Red Clearance Begin",
                    uncensored["yellow_dur"].to_numpy(dtype=float),
                    dict(color="#FFA000", width=1.5, shape="hv"),
                ),
                (
                    f"Ph{ph} Red Begin",
                    "Red Begin",
                    (uncensored["yellow_dur"] + uncensored["red_clear_dur"]).to_numpy(dtype=float),
                    dict(color="#D32F2F", width=1.5, shape="hv"),
                ),
                (
                    f"Ph{ph} Severe Threshold",
                    "Severe Threshold",
                    (uncensored["yellow_dur"] + severe_sec).to_numpy(dtype=float),
                    dict(color="#8B0000", width=1.5, dash="dash", shape="hv"),
                ),
            ]

            for trace_name, leg_group, y_vals, line_dict in ref_specs:
                y_arr = np.empty(3 * m, dtype=object)
                y_arr[0::3] = y_vals
                y_arr[1::3] = y_vals
                y_arr[2::3] = None

                show_leg = leg_group not in seen_legendgroups
                seen_legendgroups.add(leg_group)

                fig.add_trace(
                    go.Scatter(
                        x=list(x_arr),
                        y=list(y_arr),
                        mode="lines",
                        line_shape="hv",
                        name=trace_name,
                        line=line_dict,
                        legendgroup=leg_group,
                        showlegend=show_leg,
                        hovertemplate=(
                            f"<b>{trace_name}</b><br>"
                            "Time: %{x}<br>"
                            "Duration: %{y:.1f} s<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=1,
                )

        fig.update_yaxes(title_text="Time since start of yellow (s)", row=r, col=1)

    fig.update_layout(title=title, template="plotly_white")
    return fig
