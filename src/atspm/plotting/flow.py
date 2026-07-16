"""
Split Flow Rate Profiles Plot (Functional Core)

Pure function for visualising effective cumulative flow-rate profiles and
their instantaneous counterparts, produced by
``atspm.analysis.flow.rate_profiles``.

The figure is a two-row subplot with a shared time axis:

* Row 1 — effective cumulative flow rate: per-cycle markers plus
  per-detector and overall mean lines.  The peak of a mean curve marks the
  throughput-optimal split length for the approach.
* Row 2 — instantaneous flow rate (3600 / headway): mean lines only,
  smoothed with a centred rolling mean.

No side effects — no file I/O, no ``write_html()``.  The caller (imperative
shell) is responsible for saving the returned Figure.

Package Location: src/atspm/plotting/flow.py
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

# Per-detector colour palette, cycled in sorted-detector order.
_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

# Overall mean trace
_MEAN_COLOR = "#666666"
_MEAN_WIDTH = 3

_DET_MEAN_WIDTH = 2
_MARKER_OPACITY = 0.5
_MARKER_SIZE = 5

_ROW_HEIGHTS = [0.62, 0.38]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_title(metadata: Dict[str, Any], suffix: str = '') -> str:
    """Construct a plot title from metadata."""
    major = str(metadata.get('major_road_name') or '').strip()
    minor = str(metadata.get('minor_road_name') or '').strip()
    intx  = str(metadata.get('intersection_name') or 'Intersection').strip()

    if major and minor:
        location = f'{major} @ {minor}'
    else:
        location = intx

    return f'{location} – {suffix}' if suffix else location


def _detector_colors(columns: pd.Index) -> Dict[str, str]:
    """Map each detector label (first token of a column name) to a colour.

    Args:
        columns: Wide-profile column labels (``"{det} {green_ts}"``,
            ``"{det} Mean"``, ``"Mean"``).

    Returns:
        ``{det_str: hex_color}`` in sorted-detector order; the overall
        ``"Mean"`` column maps to the dedicated grey.
    """
    dets = sorted(
        {c.split(" ")[0] for c in columns if c != "Mean"},
        key=lambda d: int(d) if d.isdigit() else 0,
    )
    colors = {d: _PALETTE[i % len(_PALETTE)] for i, d in enumerate(dets)}
    colors["Mean"] = _MEAN_COLOR
    return colors


def _hover_text(label: str, values: pd.Series) -> list:
    """Build per-point hover strings for a trace."""
    return [f"{label}: {v:.0f} vphpl" for v in values]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_flow_profiles(
    rate_df: pd.DataFrame,
    inst_df: pd.DataFrame,
    metadata: Dict[str, Any],
    phase: int,
    rolling: int = 5,
) -> go.Figure:
    """Render cumulative and instantaneous flow-rate profiles for one phase.

    Args:
        rate_df: Wide effective-cumulative-rate profile from
            ``rate_profiles`` — indexed by ``t`` with per-cycle
            ``"{det} {green_ts}"`` columns, per-detector ``"{det} Mean"``
            columns and an overall ``"Mean"`` column.
        inst_df: Wide instantaneous-rate profile (same layout).
        metadata: Intersection metadata dict (``major_road_name``,
            ``minor_road_name``, ``intersection_name``) used for the title.
        phase: Signal phase number (title only).
        rolling: Centred rolling-mean window (grid rows) applied to the
            instantaneous mean traces.  ``1`` disables smoothing.
            Default ``5``.

    Returns:
        Plotly ``Figure`` with two rows sharing the time axis.  An empty
        figure (annotated title only) when *rate_df* is empty.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=_ROW_HEIGHTS,
    )

    if rate_df is not None and not rate_df.empty:
        colors = _detector_colors(rate_df.columns)

        # --- Row 1: cumulative effective rate --------------------------------
        for c in rate_df.columns:
            srs = rate_df[c].dropna().sort_index()
            if srs.empty:
                continue
            det = c.split(" ")[0]

            if c == "Mean":
                trace = go.Scatter(
                    x=srs.index, y=srs.values, mode="lines",
                    line=dict(color=_MEAN_COLOR, width=_MEAN_WIDTH),
                    hoverinfo="x+y+text", text=_hover_text(c, srs),
                )
            elif c.endswith("Mean"):
                trace = go.Scatter(
                    x=srs.index, y=srs.values, mode="lines",
                    line=dict(color=colors[det], width=_DET_MEAN_WIDTH),
                    hoverinfo="x+y+text", text=_hover_text(c, srs),
                )
            else:
                trace = go.Scatter(
                    x=srs.index, y=srs.values, mode="markers",
                    marker=dict(color=colors[det],
                                opacity=_MARKER_OPACITY, size=_MARKER_SIZE),
                    hoverinfo="x+y+text", text=_hover_text(c, srs),
                )
            fig.add_trace(trace, row=1, col=1)

        # --- Row 2: instantaneous rate (mean traces only) ---------------------
        if inst_df is not None and not inst_df.empty:
            mean_cols = [c for c in inst_df.columns if c.endswith("Mean")]
            for c in mean_cols:
                srs = (
                    inst_df[c].sort_index()
                    .rolling(rolling, center=True).mean()
                    .dropna()
                )
                if srs.empty:
                    continue
                det = c.split(" ")[0]
                color = _MEAN_COLOR if c == "Mean" else colors.get(det, _MEAN_COLOR)
                width = _MEAN_WIDTH if c == "Mean" else _DET_MEAN_WIDTH
                fig.add_trace(
                    go.Scatter(
                        x=srs.index, y=srs.values, mode="lines",
                        line=dict(color=color, width=width),
                        hoverinfo="x+y+text", text=_hover_text(c, srs),
                    ),
                    row=2, col=1,
                )

    fig.update_layout(
        title=_build_title(metadata, suffix=f"Phase {phase} Flow Rate"),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Cumulative flow rate (vphpl)", row=1, col=1)
    fig.update_yaxes(title_text="Instantaneous flow rate (vphpl)", row=2, col=1)
    fig.update_xaxes(title_text="Time in split (s)", row=2, col=1)
    fig.update_xaxes(fixedrange=False)

    return fig
