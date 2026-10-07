"""ATSPM Clock Drift Plotting (Functional Core)

Pure function – no SQL, no file I/O, no side effects.
Input: drift and clock-sets DataFrames + metadata dict + timezone.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/clock_marks.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .termination import _build_title


def _epoch_to_dt(ts_series: pd.Series, tz: Optional[str] = None) -> pd.Series:
    """Convert epoch float series to datetime, tz-aware when timezone is given."""
    dt = pd.to_datetime(ts_series, unit="s", utc=True)
    if tz:
        dt = dt.dt.tz_convert(tz)
    return dt


def plot_clock_drift(
    drift_df: pd.DataFrame,
    sets_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    timezone: Optional[str] = None,
    model_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Build an interactive Plotly figure displaying controller clock drift and clock sets.

    Args:
        drift_df: Decoded drift samples with columns ``[ts, off, drift, drift_lo,
            drift_hi, saturated, role, status]``.
        sets_df: Decoded clock sets with columns ``[bracket_on, bracket_off, width,
            shift, status]``.
        metadata: Optional dictionary of intersection metadata.
        timezone: Local timezone string for timestamp conversion (e.g. 'US/Mountain').
        model_df: Optional drift model segments (``analysis.true_time``),
            drawn as one line per segment; the gaps between are dead zones.

    Returns:
        Plotly Figure object.
    """
    title = _build_title(metadata or {}, suffix="Controller Clock Drift")
    fig = go.Figure()

    # Determine plot drift y-range for vertical set spans
    valid_drift = (
        drift_df.loc[drift_df["drift"].notna()]
        if not drift_df.empty and "drift" in drift_df.columns
        else pd.DataFrame()
    )
    sat_rows = (
        drift_df.loc[drift_df["drift"].isna() & drift_df["saturated"]]
        if not drift_df.empty and "saturated" in drift_df.columns and "drift" in drift_df.columns
        else pd.DataFrame()
    )

    y_vals = []
    if not valid_drift.empty:
        y_vals.extend(valid_drift["drift"].tolist())
        if "drift_lo" in valid_drift.columns:
            finite_lo = valid_drift.loc[np.isfinite(valid_drift["drift_lo"]), "drift_lo"].tolist()
            y_vals.extend(finite_lo)
        if "drift_hi" in valid_drift.columns:
            finite_hi = valid_drift.loc[np.isfinite(valid_drift["drift_hi"]), "drift_hi"].tolist()
            y_vals.extend(finite_hi)

    if not sat_rows.empty:
        if "drift_lo" in sat_rows.columns:
            finite_sat_lo = sat_rows.loc[np.isfinite(sat_rows["drift_lo"]), "drift_lo"].tolist()
            y_vals.extend(finite_sat_lo)
        if "drift_hi" in sat_rows.columns:
            finite_sat_hi = sat_rows.loc[np.isfinite(sat_rows["drift_hi"]), "drift_hi"].tolist()
            y_vals.extend(finite_sat_hi)

    if y_vals:
        y_min = float(min(y_vals))
        y_max = float(max(y_vals))
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
    else:
        y_min = -1.0
        y_max = 1.0

    # Horizontal zero reference line trace
    all_times = []
    if not drift_df.empty:
        if "ts" in drift_df.columns:
            all_times.extend(drift_df["ts"].dropna().tolist())
        if "off" in drift_df.columns:
            all_times.extend(drift_df["off"].dropna().tolist())
    if not sets_df.empty:
        if "bracket_on" in sets_df.columns:
            all_times.extend(sets_df["bracket_on"].dropna().tolist())
        if "bracket_off" in sets_df.columns:
            all_times.extend(sets_df["bracket_off"].dropna().tolist())

    if all_times:
        t_min = min(all_times)
        t_max = max(all_times)
        pad = max((t_max - t_min) * 0.02, 1.0)
        x_zero = _epoch_to_dt(pd.Series([t_min - pad, t_max + pad]), timezone)
        fig.add_trace(
            go.Scatter(
                x=x_zero,
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(128, 128, 128, 0.4)", dash="dash", width=1),
                name="Zero",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Drift trace
    if not valid_drift.empty:
        drift_x = _epoch_to_dt(valid_drift["ts"].fillna(valid_drift["off"]), timezone)
        drift_y = valid_drift["drift"].to_numpy(dtype=float)

        arr_hi = np.where(
            np.isfinite(valid_drift["drift_hi"]),
            valid_drift["drift_hi"].to_numpy(dtype=float) - drift_y,
            0.0,
        )
        arr_lo = np.where(
            np.isfinite(valid_drift["drift_lo"]),
            drift_y - valid_drift["drift_lo"].to_numpy(dtype=float),
            0.0,
        )
        arr_hi = np.maximum(arr_hi, 0.0)
        arr_lo = np.maximum(arr_lo, 0.0)

        roles = valid_drift["role"].to_numpy() if "role" in valid_drift.columns else [""] * len(valid_drift)
        peds = valid_drift["ped"].to_numpy() if "ped" in valid_drift.columns else [""] * len(valid_drift)
        statuses = valid_drift["status"].to_numpy() if "status" in valid_drift.columns else [""] * len(valid_drift)

        hover_texts = [
            f"Drift: {d:+.3f} s<br>Bounds: [{lo:.3f}, {hi:.3f}]<br>Ped: {p}<br>Role: {r}<br>Status: {s}"
            for d, lo, hi, p, r, s in zip(
                drift_y,
                valid_drift["drift_lo"],
                valid_drift["drift_hi"],
                peds,
                roles,
                statuses,
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=drift_x,
                y=drift_y,
                mode="markers",
                name="Drift",
                marker=dict(color="#1f77b4", size=8),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=arr_hi,
                    arrayminus=arr_lo,
                    color="rgba(31, 119, 180, 0.6)",
                    thickness=1.5,
                    width=4,
                ),
                hovertext=hover_texts,
                hoverinfo="text+x",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                name="Drift",
                marker=dict(color="#1f77b4", size=8),
            )
        )

    # Saturated rows whose drift is NaN
    if not sat_rows.empty:
        sat_x = _epoch_to_dt(sat_rows["ts"].fillna(sat_rows["off"]), timezone)
        finite_bound = np.where(
            np.isfinite(sat_rows["drift_lo"]),
            sat_rows["drift_lo"].to_numpy(dtype=float),
            sat_rows["drift_hi"].to_numpy(dtype=float),
        )
        symbols = ["triangle-up" if b >= 0 else "triangle-down" for b in finite_bound]
        sat_hover = [
            f"Saturated (bound only)<br>Bound: {b:+.2f} s<br>Ped: {p}<br>Role: {r}"
            for b, p, r in zip(
                finite_bound,
                sat_rows["ped"] if "ped" in sat_rows.columns else [""] * len(sat_rows),
                sat_rows["role"] if "role" in sat_rows.columns else [""] * len(sat_rows),
            )
        ]
        fig.add_trace(
            go.Scatter(
                x=sat_x,
                y=finite_bound,
                mode="markers",
                name="Saturated (bound only)",
                marker=dict(symbol=symbols, size=10, color="#ff7f0e"),
                hovertext=sat_hover,
                hoverinfo="text+x",
            )
        )

    # Drift model: one [start, end, None] line per segment
    if model_df is not None and not model_df.empty:
        seg_t = model_df[["seg_start", "seg_end"]].to_numpy(dtype=float)
        seg_d = (
            model_df["intercept"].to_numpy(dtype=float)[:, None]
            + model_df["slope"].to_numpy(dtype=float)[:, None]
            * (seg_t - model_df["t_ref"].to_numpy(dtype=float)[:, None])
        )
        n_seg = len(model_df)
        x_model = np.empty(n_seg * 3, dtype=object)
        x_model[0::3] = _epoch_to_dt(pd.Series(seg_t[:, 0]), timezone).to_numpy()
        x_model[1::3] = _epoch_to_dt(pd.Series(seg_t[:, 1]), timezone).to_numpy()
        x_model[2::3] = None
        y_model = np.empty(n_seg * 3, dtype=object)
        y_model[0::3] = seg_d[:, 0]
        y_model[1::3] = seg_d[:, 1]
        y_model[2::3] = None
        model_hover = [
            f"Model: {r * 1e6:+.1f} ppm, {n} samples, MAD {m:.3f} s"
            for r, n, m in zip(model_df["slope"], model_df["n_samples"], model_df["resid_mad"])
        ]
        hover_model = np.empty(n_seg * 3, dtype=object)
        hover_model[0::3] = model_hover
        hover_model[1::3] = model_hover
        hover_model[2::3] = None
        fig.add_trace(
            go.Scatter(
                x=x_model,
                y=y_model,
                mode="lines",
                name="Drift model",
                line=dict(color="#2ca02c", width=2),
                hovertext=hover_model,
                hoverinfo="text+x",
            )
        )

    # Clock set trace using [x, x, None] segment pattern
    if not sets_df.empty:
        set_times = sets_df["bracket_on"].fillna(sets_df["bracket_off"])
        set_dts = _epoch_to_dt(set_times, timezone)

        shifts = sets_df["shift"].to_numpy()
        statuses = sets_df["status"].to_numpy() if "status" in sets_df.columns else np.array(["ok"] * len(sets_df))
        widths = sets_df["width"].to_numpy() if "width" in sets_df.columns else np.zeros(len(sets_df))

        hover_labels = []
        for sh, st, w in zip(shifts, statuses, widths):
            if pd.notna(sh):
                hover_labels.append(f"Shift: {sh:+.0f} s (width {w:.1f} s, status {st})")
            else:
                hover_labels.append(f"Status: {st} (width {w:.1f} s)")

        n_sets = len(sets_df)
        x_segments = np.empty(n_sets * 3, dtype=object)
        x_segments[0::3] = set_dts
        x_segments[1::3] = set_dts
        x_segments[2::3] = None

        y_segments = np.empty(n_sets * 3, dtype=object)
        y_segments[0::3] = y_min
        y_segments[1::3] = y_max
        y_segments[2::3] = None

        text_segments = np.empty(n_sets * 3, dtype=object)
        text_segments[0::3] = hover_labels
        text_segments[1::3] = hover_labels
        text_segments[2::3] = None

        fig.add_trace(
            go.Scatter(
                x=x_segments,
                y=y_segments,
                mode="lines",
                name="Clock set",
                line=dict(color="#d62728", width=1.5, dash="dash"),
                hovertext=text_segments,
                hoverinfo="text+x",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Clock set",
                line=dict(color="#d62728", width=1.5, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Controller − true (s)",
        hovermode="closest",
        template="plotly_white",
    )

    return fig
