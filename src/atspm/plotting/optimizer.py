"""
Throughput Cycle-Length Optimizer Plots (Functional Core)

Pure functions for visualising cycle-length optimization results, produced by
:func:`atspm.analysis.optimizer.optimize`.

Figures:
* Throughput vs Cycle Length: objective curve over candidate C values, C* marker,
  and flat band shading.
* Split Allocation at C*: per-ring stacked horizontal split bars grouped by barrier
  group at C*.
* Marginal Discharge Rates: instantaneous flow-rate curves per optimized phase with
  end-of-split markers.

No side effects — no file I/O, no ``write_html()``. The caller (imperative shell)
is responsible for saving the returned Figure.

Package Location: src/atspm/plotting/optimizer.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .flow import _build_title

# Palette for phases / bases
_COLOR_SAT_THROUGHPUT = "#1f77b4"
_COLOR_TOTAL_THROUGHPUT = "#666666"
_COLOR_C_STAR = "#d62728"
_COLOR_FLAT_BAND = "rgba(44, 160, 44, 0.18)"
_COLOR_INFEASIBLE = "rgba(214, 39, 40, 0.12)"

_BASIS_COLORS = {
    "optimized": "#2ca02c",
    "sufficiency": "#1f77b4",
    "minimum": "#ff7f0e",
}

_PHASE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_throughput_curve(
    scan_df: pd.DataFrame,
    optimum: Dict[str, Any],
    metadata: Dict[str, Any],
) -> go.Figure:
    """Plot saturated and total throughput across candidate cycle lengths.

    Args:
        scan_df: Scan results DataFrame with columns ``C``, ``feasible``,
            ``throughput_sat_vph``, ``throughput_total_vph``.
        optimum: Optimum scalars dict from the optimizer.
        metadata: Intersection metadata dict for title formatting.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Determine y-range for filled areas
    y_min, y_max = 0.0, 1000.0
    if scan_df is not None and not scan_df.empty:
        feas_mask = scan_df["feasible"].fillna(False).astype(bool)
        feas = scan_df.loc[feas_mask]
        vals = []
        if not feas.empty:
            for col in ("throughput_sat_vph", "throughput_total_vph"):
                if col in feas.columns:
                    v = feas[col].to_numpy(dtype=float)
                    fin = v[np.isfinite(v)]
                    if len(fin) > 0:
                        vals.append(float(fin.max()))
        if vals:
            y_max = max(vals) * 1.10
            y_max = max(y_max, 10.0)

    # 1. Filled trace: Flat band
    flat_lo = optimum.get("flat_c_lo") if optimum else None
    flat_hi = optimum.get("flat_c_hi") if optimum else None
    if (
        flat_lo is not None
        and flat_hi is not None
        and pd.notna(flat_lo)
        and pd.notna(flat_hi)
        and np.isfinite(flat_lo)
        and np.isfinite(flat_hi)
        and flat_lo <= flat_hi
    ):
        fb_x = [flat_lo, flat_hi, flat_hi, flat_lo]
        fb_y = [y_min, y_min, y_max, y_max]
    else:
        fb_x, fb_y = [], []

    fig.add_trace(
        go.Scatter(
            x=fb_x,
            y=fb_y,
            fill="toself",
            fillcolor=_COLOR_FLAT_BAND,
            line=dict(width=0),
            name="Flat band",
            hoverinfo="skip",
        )
    )

    # 2. Filled trace: one rectangle per contiguous run of infeasible C values
    if scan_df is not None and not scan_df.empty:
        scan_sorted = scan_df.sort_values("C")
        c_all = scan_sorted["C"].to_numpy(dtype=float)
        infeasible = ~scan_sorted["feasible"].fillna(False).astype(bool).to_numpy()
        if infeasible.any():
            half = float(np.median(np.diff(c_all))) / 2.0 if len(c_all) > 1 else 0.5
            run_id = np.cumsum(np.r_[True, infeasible[1:] != infeasible[:-1]])
            runs = (
                pd.DataFrame({"C": c_all, "run": run_id})[infeasible]
                .groupby("run")["C"].agg(["min", "max"])
            )
            lo = runs["min"].to_numpy() - half
            hi = runs["max"].to_numpy() + half
            n = len(runs)
            inf_x = np.column_stack(
                [lo, hi, hi, lo, lo, np.full(n, np.nan)]
            ).ravel()
            inf_y = np.tile([y_min, y_min, y_max, y_max, y_min, np.nan], n)
            fig.add_trace(
                go.Scatter(
                    x=inf_x,
                    y=inf_y,
                    fill="toself",
                    fillcolor=_COLOR_INFEASIBLE,
                    line=dict(width=0),
                    name="Infeasible",
                    hoverinfo="skip",
                )
            )

    # 3. Line trace: Saturated throughput (feasible rows only)
    if scan_df is not None and not scan_df.empty:
        feas_mask = scan_df["feasible"].fillna(False).astype(bool)
        feas = scan_df.loc[feas_mask]
        sat_x = feas["C"].tolist()
        sat_y = feas["throughput_sat_vph"].tolist()
    else:
        sat_x, sat_y = [], []

    fig.add_trace(
        go.Scatter(
            x=sat_x,
            y=sat_y,
            mode="lines",
            line=dict(color=_COLOR_SAT_THROUGHPUT, width=2.5),
            name="Saturated throughput",
            hovertemplate="C: %{x:.1f} s<br>Saturated: %{y:.1f} vph<extra></extra>",
        )
    )

    # 4. Dotted line trace: Total throughput
    if scan_df is not None and not scan_df.empty:
        feas_mask = scan_df["feasible"].fillna(False).astype(bool)
        feas = scan_df.loc[feas_mask]
        tot_x = feas["C"].tolist()
        tot_y = feas["throughput_total_vph"].tolist()
    else:
        tot_x, tot_y = [], []

    fig.add_trace(
        go.Scatter(
            x=tot_x,
            y=tot_y,
            mode="lines",
            line=dict(color=_COLOR_TOTAL_THROUGHPUT, width=2, dash="dot"),
            name="Total throughput",
            hovertemplate="C: %{x:.1f} s<br>Total: %{y:.1f} vph<extra></extra>",
        )
    )

    # 5. Marker trace: C*
    c_star = optimum.get("c_star") if optimum else None
    thr_star = optimum.get("throughput_sat_vph") if optimum else None
    if (
        c_star is not None
        and pd.notna(c_star)
        and np.isfinite(c_star)
        and thr_star is not None
        and pd.notna(thr_star)
        and np.isfinite(thr_star)
    ):
        fig.add_trace(
            go.Scatter(
                x=[float(c_star)],
                y=[float(thr_star)],
                mode="markers",
                marker=dict(size=11, symbol="star", color=_COLOR_C_STAR),
                name="C*",
                hovertemplate="C*: %{x:.1f} s<br>Throughput: %{y:.1f} vph<extra></extra>",
            )
        )

    # Title suffix
    state = optimum.get("state") if optimum else None
    suffix = "Throughput vs Cycle Length"
    if state and state != "interior":
        suffix = f"Throughput vs Cycle Length ({state})"

    fig.update_layout(
        title=_build_title(metadata, suffix),
        xaxis=dict(title="Cycle Length C (s)"),
        yaxis=dict(title="Throughput (vph)", range=[y_min, y_max]),
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def plot_allocation(
    splits_df: pd.DataFrame,
    optimum: Dict[str, Any],
    metadata: Dict[str, Any],
) -> go.Figure:
    """Plot split allocation per ring at C*.

    Args:
        splits_df: Splits DataFrame at C* with columns ``phase``, ``ring``,
            ``barrier_group``, ``allocation_basis``, ``s_star``, etc.
        optimum: Optimum scalars dict from the optimizer.
        metadata: Intersection metadata dict for title formatting.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    c_star = optimum.get("c_star") if optimum else None

    if splits_df is not None and not splits_df.empty and "ring" in splits_df.columns:
        # Barrier group first, then splits_df row order (structure order)
        # within a (group, ring). Excluded phases and NaN s_star draw nothing.
        df = splits_df.reset_index(drop=True)
        s_star = pd.to_numeric(df["s_star"], errors="coerce")
        keep = (
            df["ring"].notna()
            & np.isfinite(s_star)
            & (df["allocation_basis"] != "excluded")
        )
        df = df.loc[keep].assign(
            s_star=s_star[keep].astype(float), _order=np.flatnonzero(keep)
        )
        sort_cols = ["ring", "barrier_group", "_order"] if "barrier_group" in df.columns else ["ring", "_order"]
        df = df.sort_values(sort_cols, kind="stable")
        df["_end"] = df.groupby("ring")["s_star"].cumsum()
        df["_start"] = df["_end"] - df["s_star"]

        def _fmt_sec(col: str) -> pd.Series:
            v = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            return v.map(lambda x: f"{x:.1f}s" if np.isfinite(x) else "N/A")

        flag_cols = [
            c for c in (
                "at_boundary", "surplus", "sufficiency_at_boundary",
                "curve_missing", "sufficiency_unverified",
            ) if c in df.columns
        ]
        flags = pd.Series("", index=df.index)
        for c in flag_cols:
            on = df[c].fillna(False).astype(bool)
            flags = flags.where(~on, flags + np.where(flags == "", "", ", ") + c)
        flags = flags.replace("", "none")

        df["_hover"] = (
            "Ph" + df["phase"].astype(int).astype(str)
            + " (" + df["allocation_basis"].astype(str) + ")<br>"
            + "s*: " + df["s_star"].map("{:.1f}s".format) + "<br>"
            + "s_min: " + _fmt_sec("s_min") + "<br>"
            + "s_domain: " + _fmt_sec("s_domain") + "<br>"
            + "flags: " + flags
        )
        df["_ring_label"] = "Ring " + df["ring"].astype(int).astype(str)

        for basis, g in df.groupby("allocation_basis", sort=False):
            n = len(g)
            x = np.column_stack(
                [g["_start"].to_numpy(), g["_end"].to_numpy(), np.full(n, None)]
            ).ravel()
            y = np.column_stack(
                [g["_ring_label"].to_numpy(), g["_ring_label"].to_numpy(), np.full(n, None)]
            ).ravel()
            text = np.column_stack(
                [g["_hover"].to_numpy(), g["_hover"].to_numpy(), np.full(n, None)]
            ).ravel()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(
                        width=28,
                        color=_BASIS_COLORS.get(basis, "#7f7f7f"),
                    ),
                    name=basis,
                    hovertext=text,
                    hoverinfo="text",
                )
            )

    x_max = (
        max(float(c_star) * 1.05, 10.0)
        if c_star is not None and pd.notna(c_star) and np.isfinite(c_star)
        else 100.0
    )

    fig.update_layout(
        title=_build_title(metadata, "Split Allocation at C*"),
        xaxis=dict(title="Time in Cycle (s)", range=[0, x_max]),
        yaxis=dict(title="", autorange="reversed"),
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def plot_marginal_rates(
    curves: Dict[int, pd.DataFrame],
    splits_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> go.Figure:
    """Plot marginal (instantaneous) discharge rates with end-of-split markers.

    Args:
        curves: Dict mapping phase number to discharge profile DataFrame
            (with index ``t`` and column ``inst``).
        splits_df: Splits DataFrame at C*.
        metadata: Intersection metadata dict for title formatting.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    opt_phases = []
    if (
        splits_df is not None
        and not splits_df.empty
        and "allocation_basis" in splits_df.columns
    ):
        opt_rows = splits_df.loc[splits_df["allocation_basis"] == "optimized"]
        opt_phases = sorted(opt_rows["phase"].dropna().astype(int).tolist())

    color_idx = 0
    end_x, end_y, end_text = [], [], []

    for p in opt_phases:
        row = splits_df.loc[splits_df["phase"] == p]
        s_star = row["s_star"].iloc[0] if not row.empty else np.nan
        rate = row["end_inst_rate_vph"].iloc[0] if not row.empty else np.nan

        color = _PHASE_PALETTE[color_idx % len(_PHASE_PALETTE)]
        color_idx += 1

        if p in curves and curves[p] is not None and not curves[p].empty:
            curve = curves[p]
            if "inst" in curve.columns:
                t_vals = curve.index.to_numpy(dtype=float)
                inst_vals = curve["inst"].to_numpy(dtype=float)
                fig.add_trace(
                    go.Scatter(
                        x=t_vals,
                        y=inst_vals,
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=f"Ph{p}",
                        hovertemplate=f"Ph{p}<br>t: %{{x:.1f}} s<br>inst: %{{y:.1f}} vph<extra></extra>",
                    )
                )

        if (
            pd.notna(s_star)
            and np.isfinite(s_star)
            and pd.notna(rate)
            and np.isfinite(rate)
        ):
            end_x.append(float(s_star))
            end_y.append(float(rate))
            end_text.append(f"Ph{p} End of split<br>s*: {s_star:.1f}s<br>rate: {rate:.1f} vph")

    fig.add_trace(
        go.Scatter(
            x=end_x,
            y=end_y,
            mode="markers",
            marker=dict(size=10, symbol="circle", color=_COLOR_C_STAR),
            name="End of split",
            hovertext=end_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=_build_title(metadata, "Marginal Discharge Rates"),
        xaxis=dict(title="Green Time t (s)"),
        yaxis=dict(title="Instantaneous Rate (vph)"),
        template="plotly_white",
        hovermode="closest",
    )
    return fig
