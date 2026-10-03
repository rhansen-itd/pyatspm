"""
ATSPM Detector Health Plot (Functional Core)

Pure functions only. No SQL, no file I/O, no side effects.
Input: activity profile DataFrame, findings DataFrame, and optional metadata dict.
Output: plotly.graph_objects.Figure.

Package Location: src/atspm/plotting/detector_health.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .termination import _build_title


def plot_detector_health(
    profile: Optional[pd.DataFrame],
    findings: Optional[pd.DataFrame],
    metadata: Optional[Dict[str, Any]] = None,
    window: str = "day",
) -> go.Figure:
    """Generate detector-health heatmap with overlaid findings.

    Args:
        profile: Activity profile DataFrame containing columns
            ``['date', 'window', 'detector', 'n_act']``.
        findings: Findings DataFrame conforming to ``FINDINGS_SCHEMA``.
        metadata: Optional metadata dictionary with intersection road names.
        window: Window name to plot (e.g. ``'day'``, ``'am'``, ``'pm'``).

    Returns:
        Plotly Figure object displaying a detector x date heatmap normalized
        to each detector's own median, with flagged finding cells overlaid.
    """
    title_text = _build_title(
        metadata or {}, suffix="Detector Health — count vs the detector's median"
    )

    if profile is None or profile.empty:
        fig = go.Figure()
        fig.update_layout(title=dict(text=title_text))
        return fig

    # Filter to selected window
    df_win = profile.loc[profile["window"] == window].copy()
    if df_win.empty:
        fig = go.Figure()
        fig.update_layout(title=dict(text=title_text))
        return fig

    df_win["date"] = df_win["date"].astype(str)
    detectors = sorted(df_win["detector"].unique())
    dates = sorted(df_win["date"].unique())

    # Build vectorized pivot matrix
    pivot = df_win.pivot(index="detector", columns="date", values="n_act")
    pivot = pivot.reindex(index=detectors, columns=dates)

    # Normalize each detector's row to its own median across the shown days.
    # Median 0 -> leave row un-normalized / NaN (never divide by zero).
    medians = pivot.median(axis=1)
    valid_medians = medians.where(medians > 0)
    z = pivot.divide(valid_medians, axis=0).to_numpy(dtype=float)

    hm = go.Heatmap(
        x=dates,
        y=detectors,
        z=z,
        colorscale="Blues",
        colorbar=dict(title="Act / Median"),
        hovertemplate=(
            "Detector: %{y}<br>Date: %{x}<br>Norm Count: %{z:.2f}<extra></extra>"
        ),
    )

    fig = go.Figure(data=[hm])

    # Overlay findings markers if present
    if findings is not None and not findings.empty:
        f = findings.copy()
        if "window" in f.columns and window not in (None, "all"):
            f_win = f[f["window"] == window]
            if not f_win.empty:
                f = f_win
        f["date"] = f["date"].astype(str)
        f_det = f[f["detector"].isin(detectors)]
        if not f_det.empty:
            hover = [
                f"Det {det} ({d})<br>{r} [{s}]<br>{m}"
                for det, d, r, s, m in zip(
                    f_det["detector"],
                    f_det["date"],
                    f_det["rule"],
                    f_det["severity"],
                    f_det["message"].fillna(""),
                )
            ]
            fig.add_trace(
                go.Scatter(
                    x=f_det["date"],
                    y=f_det["detector"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=12,
                        color="red",
                        line=dict(width=2, color="crimson"),
                    ),
                    text=hover,
                    hoverinfo="text",
                    name="Findings",
                )
            )

    fig.update_layout(
        title=dict(text=title_text),
        xaxis=dict(title="Date", type="category"),
        yaxis=dict(title="Detector", type="category"),
    )

    return fig
