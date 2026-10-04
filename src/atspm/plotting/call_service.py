"""
ATSPM Pedestrian Delay and Wait Time Plots (Functional Core)

Pure functions only.  No SQL, no file I/O, no side effects.
Input: Delay/Wait DataFrames + binned summary DataFrames + metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/call_service.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .termination import _build_title
from ..analysis.call_service import DEFAULT_MAX_WAIT_S

# Termination categories for wait time: key, label, base marker symbol, color
_WAIT_TERMINATIONS = [
    ("gap_out", "Gap Out", "circle", "green"),
    ("max_out", "Max Out", "square", "red"),
    ("force_off", "Force Off", "diamond", "orangered"),
    ("unknown", "Unknown", "triangle-up", "gray"),
]


def plot_ped_delay(
    delay_df: pd.DataFrame,
    binned_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Build the Pedestrian Delay chart.

    Args:
        delay_df: Output delay DataFrame from :func:`ped_delay`.
        binned_df: Output binned summary DataFrame from :func:`summarize_ped_delay`.
        metadata: Optional metadata dict with intersection details (road names, etc.).

    Returns:
        Plotly Figure with one subplot row per phase.
    """
    title = _build_title(metadata or {}, suffix="Pedestrian Delay")

    if delay_df is None or delay_df.empty or "phase" not in delay_df.columns:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    phases = sorted(delay_df["phase"].dropna().unique().tolist())
    n = len(phases)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
    )

    seen_legendgroups: Set[str] = set()

    for idx, ph in enumerate(phases):
        r = idx + 1
        df_ph = delay_df.loc[delay_df["phase"] == ph]

        # 1. Delay markers (kind == "waited")
        sub_delay = df_ph.loc[df_ph["kind"] == "waited"]
        if not sub_delay.empty:
            trace_name = f"Ph{ph} Delay"
            cat_name = "Delay"
            show_leg = cat_name not in seen_legendgroups
            seen_legendgroups.add(cat_name)

            custom = np.column_stack([
                sub_delay["call_ts"].astype(str),
                sub_delay["n_presses"].astype(str),
            ])
            fig.add_trace(
                go.Scatter(
                    x=sub_delay["walk_ts"],
                    y=sub_delay["delay_s"],
                    mode="markers",
                    name=trace_name,
                    marker=dict(color="#1976D2", size=6, symbol="circle"),
                    legendgroup=cat_name,
                    showlegend=show_leg,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{trace_name}</b><br>"
                        "Walk: %{x}<br>"
                        "Delay: %{y:.1f} s<br>"
                        "Call: %{customdata[0]}<br>"
                        "Presses: %{customdata[1]}<extra></extra>"
                    ),
                ),
                row=r,
                col=1,
            )

        # 2. In Walk markers (kind == "in_walk", y = 0)
        sub_iw = df_ph.loc[df_ph["kind"] == "in_walk"]
        if not sub_iw.empty:
            trace_name = f"Ph{ph} In Walk"
            cat_name = "In Walk"
            show_leg = cat_name not in seen_legendgroups
            seen_legendgroups.add(cat_name)

            custom = sub_iw["n_presses"].astype(str)
            fig.add_trace(
                go.Scatter(
                    x=sub_iw["walk_ts"],
                    y=np.zeros(len(sub_iw)),
                    mode="markers",
                    name=trace_name,
                    marker=dict(color="#388E3C", size=6, symbol="triangle-up"),
                    legendgroup=cat_name,
                    showlegend=show_leg,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{trace_name}</b><br>"
                        "Walk: %{x}<br>"
                        "Presses: %{customdata}<extra></extra>"
                    ),
                ),
                row=r,
                col=1,
            )

        # 3. Uncalled Walk markers (kind == "uncalled", y = 0, hollow symbol)
        sub_unc = df_ph.loc[df_ph["kind"] == "uncalled"]
        if not sub_unc.empty:
            trace_name = f"Ph{ph} Uncalled Walk"
            cat_name = "Uncalled Walk"
            show_leg = cat_name not in seen_legendgroups
            seen_legendgroups.add(cat_name)

            fig.add_trace(
                go.Scatter(
                    x=sub_unc["walk_ts"],
                    y=np.zeros(len(sub_unc)),
                    mode="markers",
                    name=trace_name,
                    marker=dict(color="#7B1FA2", size=6, symbol="circle-open"),
                    legendgroup=cat_name,
                    showlegend=show_leg,
                    hovertemplate=(
                        f"<b>{trace_name}</b><br>"
                        "Walk: %{x}<extra></extra>"
                    ),
                ),
                row=r,
                col=1,
            )

        # 4. Average line from binned_df
        if binned_df is not None and not binned_df.empty and "phase" in binned_df.columns:
            b_ph = binned_df.loc[
                (binned_df["phase"] == ph) & binned_df["avg_delay_s"].notna()
            ]
            if not b_ph.empty:
                trace_name = f"Ph{ph} Average"
                cat_name = "Average"
                show_leg = cat_name not in seen_legendgroups
                seen_legendgroups.add(cat_name)

                fig.add_trace(
                    go.Scatter(
                        x=b_ph["time"],
                        y=b_ph["avg_delay_s"],
                        mode="lines",
                        line_shape="hv",
                        name=trace_name,
                        line=dict(color="black", width=2),
                        legendgroup=cat_name,
                        showlegend=show_leg,
                        hovertemplate=(
                            f"<b>{trace_name}</b><br>"
                            "Time: %{x}<br>"
                            "Avg Delay: %{y:.1f} s<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=1,
                )

        fig.update_yaxes(title_text="Pedestrian delay (s)", row=r, col=1)

    fig.update_layout(title=title, template="plotly_white")
    return fig


def plot_wait_time(
    wait_df: pd.DataFrame,
    binned_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    max_wait: Optional[float] = DEFAULT_MAX_WAIT_S,
) -> go.Figure:
    """Build the Wait Time chart.

    Args:
        wait_df: Output wait DataFrame from :func:`wait_time`.
        binned_df: Output binned summary DataFrame from :func:`summarize_wait_time`.
        metadata: Optional metadata dict with intersection details (road names, etc.).
        max_wait: Maximum wait time in seconds to plot. Rows with wait_s > max_wait
            are excluded from plotting. Default 360.0.

    Returns:
        Plotly Figure with one subplot row per phase.
    """
    title = _build_title(metadata or {}, suffix="Wait Time")

    if wait_df is None or wait_df.empty or "phase" not in wait_df.columns:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    phases = sorted(wait_df["phase"].dropna().unique().tolist())
    n = len(phases)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
    )

    seen_legendgroups: Set[str] = set()

    # Filter to called, not censored, and wait_s <= max_wait (if specified)
    mask = wait_df["called"].astype(bool) & (~wait_df["censored"].astype(bool))
    if max_wait is not None:
        mask = mask & (wait_df["wait_s"] <= max_wait)
    filtered_df = wait_df.loc[mask]

    for idx, ph in enumerate(phases):
        r = idx + 1
        df_ph = filtered_df.loc[filtered_df["phase"] == ph]

        # 1. Termination traces
        for term, term_label, base_sym, color in _WAIT_TERMINATIONS:
            sub = df_ph.loc[df_ph["termination"] == term]
            if sub.empty:
                continue

            trace_name = f"Ph{ph} {term_label}"
            show_leg = term_label not in seen_legendgroups
            seen_legendgroups.add(term_label)

            symbols = np.where(
                sub["held"].to_numpy(dtype=bool),
                f"{base_sym}-open",
                base_sym,
            )
            held_label = np.where(sub["held"].to_numpy(dtype=bool), " (held)", "")
            custom = np.column_stack([
                sub["red_ts"].astype(str),
                sub["call_ts"].astype(str),
                held_label,
            ])

            fig.add_trace(
                go.Scatter(
                    x=sub["green_ts"],
                    y=sub["wait_s"],
                    mode="markers",
                    name=trace_name,
                    marker=dict(symbol=symbols, color=color, size=6),
                    legendgroup=term_label,
                    showlegend=show_leg,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{trace_name}</b>"
                        "%{customdata[2]}<br>"
                        "Green: %{x}<br>"
                        "Wait: %{y:.1f} s<br>"
                        "Red start: %{customdata[0]}<br>"
                        "Call: %{customdata[1]}<extra></extra>"
                    ),
                ),
                row=r,
                col=1,
            )

        # 2. Average line from binned_df
        if binned_df is not None and not binned_df.empty and "phase" in binned_df.columns:
            b_ph = binned_df.loc[
                (binned_df["phase"] == ph) & binned_df["avg_wait_s"].notna()
            ]
            if not b_ph.empty:
                trace_name = f"Ph{ph} Average"
                cat_name = "Average"
                show_leg = cat_name not in seen_legendgroups
                seen_legendgroups.add(cat_name)

                fig.add_trace(
                    go.Scatter(
                        x=b_ph["time"],
                        y=b_ph["avg_wait_s"],
                        mode="lines",
                        line_shape="hv",
                        name=trace_name,
                        line=dict(color="black", width=2),
                        legendgroup=cat_name,
                        showlegend=show_leg,
                        hovertemplate=(
                            f"<b>{trace_name}</b><br>"
                            "Time: %{x}<br>"
                            "Avg Wait: %{y:.1f} s<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=1,
                )

        fig.update_yaxes(title_text="Wait time (s)", row=r, col=1)

    fig.update_layout(title=title, template="plotly_white")
    return fig
