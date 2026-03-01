"""
Detector Comparison Plot (Functional Core)

Pure function for visualising co-located detector actuations side-by-side,
with optional anomaly overlays derived from
``atspm.analysis.detectors.analyze_discrepancies``.

Package Location: src/atspm/plotting/detectors.py
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..analysis.detectors import _reconstruct_intervals

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# Y-axis layout: each pair occupies a 3-unit band.
# det_a sits at band_base + 2, det_b at band_base + 1 (matching legacy).
_BAND_HEIGHT  = 3
_DET_A_OFFSET = 2
_DET_B_OFFSET = 1

# Detector trace colours
_COLOR_A    = "#1f77b4"   # steel blue   -- Detector A
_COLOR_B    = "#ff7f0e"   # burnt orange -- Detector B
_LINE_WIDTH = 6

# ---------------------------------------------------------------------------
# Anomaly styling
# ---------------------------------------------------------------------------

# Isolated pulse: small diamond, colour-coordinated with the source detector.
# Size is intentionally small (~0.3 of the previous 13px value).
_PULSE_MARKER_SIZE = 4

# Extended disagreement: directional fill colours.
#   Det A stuck ON  -> faint blue  (matched to _COLOR_A hue)
#   Det B stuck ON  -> faint orange (matched to _COLOR_B hue)
_DISAGREE_FILL_A = "rgba(31,  119, 180, 0.13)"
_DISAGREE_FILL_B = "rgba(255, 127,  14, 0.13)"
_DISAGREE_LINE_A = "rgba(31,  119, 180, 0.50)"
_DISAGREE_LINE_B = "rgba(255, 127,  14, 0.50)"

# Y-axis: padding beyond the outermost detector rows
_Y_PAD = 0.6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _y_positions(pair_idx: int) -> tuple[float, float]:
    """Return ``(y_a, y_b)`` for the given pair index (0-based).

    Args:
        pair_idx: Zero-based position of the pair in the display list.

    Returns:
        Tuple ``(y_a, y_b)`` -- Y-axis values for det_a and det_b traces.
    """
    base = pair_idx * _BAND_HEIGHT
    return float(base + _DET_A_OFFSET), float(base + _DET_B_OFFSET)


def _epochs_to_local(
    epoch_arr: np.ndarray,
    tz: str,
) -> pd.DatetimeIndex:
    """Convert a float epoch array to timezone-aware pandas Timestamps.

    Args:
        epoch_arr: 1-D float array of UTC epoch seconds.
        tz: pytz-compatible timezone string (e.g. ``'US/Mountain'``).

    Returns:
        DatetimeIndex localised to *tz*.
    """
    return pd.to_datetime(epoch_arr, unit="s", utc=True).tz_convert(tz)


def _build_scatter_coords(
    intervals_df: pd.DataFrame,
    y_value: float,
    tz: str,
) -> tuple[list, list, list[str]]:
    """Vectorise intervals -> flat Scatter ``(x, y, hover)`` using the None-gap pattern.

    Each actuation is represented as ``[on_ts, off_ts, None]`` so Plotly
    draws exactly one line segment per actuation with no connecting line
    between successive actuations.  X values are converted from UTC epoch
    floats to timezone-aware Timestamps so Plotly renders local time.

    Args:
        intervals_df: Output of ``_reconstruct_intervals`` -- columns
            ``['on_ts', 'off_ts', 'duration_sec']``.
        y_value: Fixed Y position for all segments in this trace.
        tz: pytz-compatible timezone string for x-axis conversion.

    Returns:
        Tuple ``(x_coords, y_coords, hover_texts)`` ready for ``go.Scatter``.
        Returns three empty lists when ``intervals_df`` is empty.
    """
    if intervals_df.empty:
        return [], [], []

    on_arr  = intervals_df["on_ts"].values
    off_arr = intervals_df["off_ts"].values
    dur_arr = intervals_df["duration_sec"].values
    n = len(intervals_df)

    on_local  = _epochs_to_local(on_arr,  tz)
    off_local = _epochs_to_local(off_arr, tz)

    # 3 slots per interval: on_ts, off_ts, None (segment break)
    x = np.empty(n * 3, dtype=object)
    x[0::3] = on_local
    x[1::3] = off_local
    x[2::3] = None

    y = np.empty(n * 3, dtype=object)
    y[0::3] = y_value
    y[1::3] = y_value
    y[2::3] = None

    hover: list[str] = []
    for on_t, off_t, dur in zip(on_local, off_local, dur_arr):
        txt = (
            f"ON:  {on_t.strftime('%H:%M:%S.%f')[:-3]}<br>"
            f"OFF: {off_t.strftime('%H:%M:%S.%f')[:-3]}<br>"
            f"Dur: {dur:.3f} s"
        )
        hover += [txt, txt, ""]

    return x.tolist(), y.tolist(), hover


def _format_title(metadata: dict) -> str:
    """Build a dynamic intersection title from metadata keys.

    Format: ``"{major_route} ({major_road_name}) & {minor_route} ({minor_road_name})"``
    Components that are ``None`` / empty are omitted gracefully.

    Args:
        metadata: Intersection metadata dict.

    Returns:
        Formatted title string, prefixed with ``"Detector Comparison -- "``.
    """
    def _segment(route: Optional[str], name: Optional[str]) -> str:
        r, n = (route or "").strip(), (name or "").strip()
        if r and n:
            return f"{r} ({n})"
        return r or n

    major = _segment(
        metadata.get("major_road_route"),
        metadata.get("major_road_name"),
    )
    minor = _segment(
        metadata.get("minor_road_route"),
        metadata.get("minor_road_name"),
    )

    if major and minor:
        location = f"{major} & {minor}"
    elif major:
        location = major
    else:
        location = metadata.get("intersection_name", "Unknown Intersection")

    return f"Detector Comparison -- {location}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_detector_comparison(
    events_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    detector_pairs: List[Dict[str, int]],
    metadata: Optional[dict] = None,
) -> go.Figure:
    """Build an interactive Plotly figure comparing co-located detector pairs.

    For each ``{"phase": int, "det_a": int, "det_b": int}`` pair, actuation
    intervals are reconstructed via
    :func:`~atspm.analysis.detectors._reconstruct_intervals` and rendered as
    horizontal line segments on a shared local-time axis.  Pairs are stacked
    vertically with a 3-unit Y-band gap, matching the legacy layout.

    **Legend** -- contains only anomaly-type entries (detector lines are
    self-labelled by their Y-axis tick text and are excluded from the legend):

    * Two dummy square markers for the disagreement colour key (blue / orange),
      conditionally added only when those anomaly types are present.
    * One real diamond marker trace for isolated pulses, with ``showlegend=True``
      so the legend toggle directly controls the visible markers.

    **Anomaly overlays**:

    * ``extended_disagreement`` -- translucent directional rectangle:

      * Faint blue when Det A was stuck ON.
      * Faint orange when Det B was stuck ON.
      * An invisible-but-hoverable Scatter point at the midpoint; its marker
        colour is set to match the shape border so the Plotly hover box
        border colour matches the rectangle.

    * ``isolated_pulse`` -- small diamond markers, colour-matched to the
      source detector's trace colour (blue for Det A, orange for Det B),
      plotted on the source detector's Y lane.  All pulses are split into two
      ``go.Scatter`` traces (one per detector colour) so the legend entry
      is toggleable and actually hides/shows the real markers.

    **Title** -- anchored above the plot area via ``yref="paper"`` and a
    generous top margin, so it never overlaps the chart.

    **Y-axis zoom** -- ``range`` is hard-capped to ``[y_min, y_max]`` so
    users can zoom in but cannot pan the Y-axis outside the data bounds.

    Args:
        events_df: Raw ATSPM events with columns
            ``['timestamp', 'event_code', 'parameter']``.
            Codes: 82 = ON, 81 = OFF, -1 = gap marker.
        anomalies_df: DataFrame returned by
            :func:`~atspm.analysis.detectors.analyze_discrepancies`.
            Pass an empty DataFrame to suppress all overlays.
        detector_pairs: List of ``{"phase": int, "det_a": int, "det_b": int}``
            dicts ordered as they should appear top-to-bottom on the plot.
        metadata: Intersection metadata dict.  Keys used: ``major_road_route``,
            ``major_road_name``, ``minor_road_route``, ``minor_road_name``,
            ``intersection_name``, ``timezone``.  Defaults to ``{}`` when
            ``None``.

    Returns:
        Plotly ``go.Figure``.  Caller is responsible for all I/O.
    """
    if metadata is None:
        metadata = {}

    tz: str = metadata.get("timezone") or "UTC"

    fig = go.Figure()

    if events_df.empty or not detector_pairs:
        fig.update_layout(
            title=_format_title(metadata),
            xaxis_title="Time",
            yaxis_title="Detector",
            template="plotly_white",
        )
        return fig

    events_sorted = events_df.sort_values("timestamp").reset_index(drop=True)

    y_tick_vals:  list[float] = []
    y_tick_texts: list[str]   = []
    shapes:       list[dict]  = []

    # ------------------------------------------------------------------
    # Detector traces -- showlegend=False: Y-axis tick labels are the
    # self-evident key; per-detector legend entries add no value and clutter
    # the legend with toggleable items that aren't anomaly indicators.
    # ------------------------------------------------------------------
    for idx, pair in enumerate(detector_pairs):
        phase    = pair["phase"]
        det_a    = pair["det_a"]
        det_b    = pair["det_b"]
        y_a, y_b = _y_positions(idx)

        ivs_a = _reconstruct_intervals(events_sorted, det_a)
        ivs_b = _reconstruct_intervals(events_sorted, det_b)

        x_a, ya_coords, hover_a = _build_scatter_coords(ivs_a, y_a, tz)
        fig.add_trace(go.Scatter(
            x=x_a,
            y=ya_coords,
            mode="lines",
            line=dict(color=_COLOR_A, width=_LINE_WIDTH),
            name=f"Ph{phase} Det {det_a}",
            showlegend=False,          # labelled by Y-axis tick text
            hovertext=hover_a,
            hoverinfo="text",
        ))

        x_b, yb_coords, hover_b = _build_scatter_coords(ivs_b, y_b, tz)
        fig.add_trace(go.Scatter(
            x=x_b,
            y=yb_coords,
            mode="lines",
            line=dict(color=_COLOR_B, width=_LINE_WIDTH),
            name=f"Ph{phase} Det {det_b}",
            showlegend=False,          # labelled by Y-axis tick text
            hovertext=hover_b,
            hoverinfo="text",
        ))

        y_tick_vals.extend([y_a, y_b])
        y_tick_texts.extend([
            f"Ph{phase} Det {det_a}",
            f"Ph{phase} Det {det_b}",
        ])

    # ------------------------------------------------------------------
    # Anomaly overlays
    # ------------------------------------------------------------------
    has_disagree_a = False
    has_disagree_b = False
    has_pulse      = False

    if not anomalies_df.empty:
        has_disagree_a, has_disagree_b, has_pulse = _add_anomaly_overlays(
            fig=fig,
            anomalies_df=anomalies_df,
            detector_pairs=detector_pairs,
            shapes=shapes,
            tz=tz,
        )

    # ------------------------------------------------------------------
    # Dummy legend entries for disagreement colour key.
    # Shapes never appear in the Plotly legend; invisible Scatter markers
    # (x=[None]) with showlegend=True serve as stand-ins for the colour swatch.
    # These are intentionally NOT toggleable for shapes (shapes have no
    # trace binding), so we keep them purely informational.
    # ------------------------------------------------------------------
    if has_disagree_a:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                size=12,
                color=_DISAGREE_FILL_A,
                line=dict(color=_DISAGREE_LINE_A, width=2),
            ),
            name="Det A ON (disagreement)",
            showlegend=True,
            hoverinfo="skip",
        ))

    if has_disagree_b:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                size=12,
                color=_DISAGREE_FILL_B,
                line=dict(color=_DISAGREE_LINE_B, width=2),
            ),
            name="Det B ON (disagreement)",
            showlegend=True,
            hoverinfo="skip",
        ))

    # ------------------------------------------------------------------
    # Y-axis hard range
    # ------------------------------------------------------------------
    n_pairs = len(detector_pairs)
    y_min = _DET_B_OFFSET - _Y_PAD
    y_max = (n_pairs - 1) * _BAND_HEIGHT + _DET_A_OFFSET + _Y_PAD

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=_format_title(metadata),
            font=dict(size=16),
            x=0.5,
            xanchor="center",
            # yref="paper" + y=1.0 anchors to the very top of the paper
            # coordinate system.  The top margin (t=90) reserves enough
            # whitespace above the plot area so the title text sits entirely
            # outside it and never overlaps chart content.
            yref="paper",
            y=1.0,
            yanchor="bottom",
        ),
        xaxis=dict(
            title="Time",              # timezone omitted -- assumed by viewer
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
            zeroline=False,
            type="date",
        ),
        yaxis=dict(
            title="Detector",
            tickvals=y_tick_vals,
            ticktext=y_tick_texts,
            showgrid=False,
            zeroline=False,
            range=[y_min, y_max],
            fixedrange=False,
        ),
        shapes=shapes,
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(150,150,150,0.4)",
            borderwidth=1,
            font=dict(size=12),
        ),
        # t=90 gives the title (above paper y=1.0) room to breathe.
        # b=160 keeps the below-chart legend from being clipped.
        margin=dict(l=140, r=40, t=90, b=160),
    )

    return fig


# ---------------------------------------------------------------------------
# Anomaly overlay helper (module-private)
# ---------------------------------------------------------------------------

def _add_anomaly_overlays(
    fig: go.Figure,
    anomalies_df: pd.DataFrame,
    detector_pairs: List[Dict[str, int]],
    shapes: list,
    tz: str,
) -> tuple[bool, bool, bool]:
    """Mutate *fig* and *shapes* in-place to add anomaly visualisations.

    Processes ``anomalies_df`` in one pass:

    * ``extended_disagreement`` -- appends a directional rect shape and an
      invisible hover Scatter point whose marker colour matches the shape
      border, so Plotly's hover-box colour matches the rectangle.
    * ``isolated_pulse`` -- accumulates midpoints into two colour-keyed
      lists (one for Det A source, one for Det B source).  Each list becomes
      its own ``go.Scatter`` trace with ``showlegend=True``, making the legend
      toggle directly control the visible markers.

    Args:
        fig: The ``go.Figure`` being built (mutated in place).
        anomalies_df: Anomaly DataFrame from ``analyze_discrepancies``.
            Must include an ``on_det_id`` column.
        detector_pairs: Ordered pair list -- used to derive correct Y positions.
        shapes: Mutable list extended with rectangle shape dicts.
        tz: pytz-compatible timezone string for x-axis timestamp conversion.

    Returns:
        Tuple ``(has_disagree_a, has_disagree_b, has_pulse)`` -- flags
        so the caller can add dummy legend entries for disagreement colours.
        ``has_pulse`` is ``True`` whenever any isolated-pulse trace was added.
    """
    det_y: dict[int, float] = {}
    pair_y_bounds: dict[tuple[int, int], tuple[float, float]] = {}

    for idx, pair in enumerate(detector_pairs):
        y_a, y_b = _y_positions(idx)
        det_y[pair["det_a"]] = y_a
        det_y[pair["det_b"]] = y_b
        pair_y_bounds[(pair["det_a"], pair["det_b"])] = (y_b - 0.4, y_a + 0.4)

    # Isolated pulse accumulators -- split by source detector colour so each
    # trace's legend toggle actually controls its visible markers.
    pulse_a_x:     list = []   # pulses where Det A was the source
    pulse_a_y:     list[float] = []
    pulse_a_hover: list[str]   = []

    pulse_b_x:     list = []   # pulses where Det B was the source
    pulse_b_y:     list[float] = []
    pulse_b_hover: list[str]   = []

    has_disagree_a = False
    has_disagree_b = False
    has_pulse      = False

    for _, row in anomalies_df.iterrows():
        atype   = str(row["anomaly_type"])
        t_start = float(row["start_timestamp"])
        t_end   = float(row["end_timestamp"])
        desc    = str(row.get("description", ""))
        det_a   = int(row["det_a_id"])
        det_b   = int(row["det_b_id"])
        on_det  = int(row.get("on_det_id", det_a))

        y_a = det_y.get(det_a)
        y_b = det_y.get(det_b)
        if y_a is None or y_b is None:
            continue

        y_lo, y_hi = pair_y_bounds.get(
            (det_a, det_b),
            (min(y_a, y_b) - 0.4, max(y_a, y_b) + 0.4),
        )

        ts_start_local = _epochs_to_local(np.array([t_start]), tz)[0]
        ts_end_local   = _epochs_to_local(np.array([t_end]),   tz)[0]

        if atype == "extended_disagreement":
            if on_det == det_a:
                fill, border = _DISAGREE_FILL_A, _DISAGREE_LINE_A
                has_disagree_a = True
            else:
                fill, border = _DISAGREE_FILL_B, _DISAGREE_LINE_B
                has_disagree_b = True

            shapes.append(dict(
                type="rect",
                xref="x",
                yref="y",
                x0=ts_start_local,
                x1=ts_end_local,
                y0=y_lo,
                y1=y_hi,
                fillcolor=fill,
                line=dict(color=border, width=1.5, dash="dot"),
                layer="below",
            ))

            # Invisible hover point.  Setting marker.color to the shape's
            # border colour causes Plotly to draw the hover-box border in that
            # same colour, so blue rectangles get blue hovers and orange
            # rectangles get orange hovers.
            mid_epoch = (t_start + t_end) / 2.0
            mid_local = _epochs_to_local(np.array([mid_epoch]), tz)[0]
            fig.add_trace(go.Scatter(
                x=[mid_local],
                y=[(y_a + y_b) / 2.0],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=8,
                    opacity=0,
                    color=border,      # drives hover-box border colour
                ),
                hovertext=[f"Extended Disagreement<br>{desc}"],
                hoverinfo="text",
                showlegend=False,
                name="",
            ))

        elif atype == "isolated_pulse":
            has_pulse = True
            mid_epoch = (t_start + t_end) / 2.0
            mid_local = _epochs_to_local(np.array([mid_epoch]), tz)[0]
            src_y     = det_y.get(on_det, (y_a + y_b) / 2.0)

            if on_det == det_a:
                pulse_a_x.append(mid_local)
                pulse_a_y.append(src_y)
                pulse_a_hover.append(f"Isolated Pulse<br>{desc}")
            else:
                pulse_b_x.append(mid_local)
                pulse_b_y.append(src_y)
                pulse_b_hover.append(f"Isolated Pulse<br>{desc}")

    # ------------------------------------------------------------------
    # Add isolated-pulse traces.  Each trace has showlegend=True so the
    # legend entry is directly bound to the real markers -- clicking it
    # in the legend actually hides/shows the diamonds on the plot.
    # Two separate traces (one per detector colour) let the user toggle
    # Det-A pulses and Det-B pulses independently.
    # ------------------------------------------------------------------
    _pulse_border_a = "rgba(15, 75, 130, 0.90)"    # darker blue border for Det A diamonds
    _pulse_border_b = "rgba(180, 80, 0, 0.90)"      # darker orange border for Det B diamonds

    if pulse_a_x:
        fig.add_trace(go.Scatter(
            x=pulse_a_x,
            y=pulse_a_y,
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=_PULSE_MARKER_SIZE,
                color=_COLOR_A,
                line=dict(color=_pulse_border_a, width=1),
            ),
            name="Isolated Pulse (Det A)",
            hovertext=pulse_a_hover,
            hoverinfo="text",
            showlegend=True,
        ))

    if pulse_b_x:
        fig.add_trace(go.Scatter(
            x=pulse_b_x,
            y=pulse_b_y,
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=_PULSE_MARKER_SIZE,
                color=_COLOR_B,
                line=dict(color=_pulse_border_b, width=1),
            ),
            name="Isolated Pulse (Det B)",
            hovertext=pulse_b_hover,
            hoverinfo="text",
            showlegend=True,
        ))

    return has_disagree_a, has_disagree_b, has_pulse