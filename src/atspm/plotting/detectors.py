"""
Detector Comparison Plot (Functional Core)

Pure function for visualising co-located detector actuations side-by-side,
with optional anomaly overlays derived from
``atspm.analysis.detectors.analyze_discrepancies``.

Package Location: src/atspm/plotting/detectors.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..analysis.detectors import _reconstruct_intervals, _unique_pairs
from ..utils.timezone import DEFAULT_TIMEZONE

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# Y-axis layout: each pair occupies a 3-unit band, first pair at the top.
# det_a sits at band_base + 2, det_b at band_base + 1.
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
_PULSE_MARKER_SIZE = 7

# Extended disagreement: directional fill colours.
#   Det A stuck ON  -> faint blue  (matched to _COLOR_A hue)
#   Det B stuck ON  -> faint orange (matched to _COLOR_B hue)
_DISAGREE_FILL_A = "rgba(31,  119, 180, 0.13)"
_DISAGREE_FILL_B = "rgba(255, 127,  14, 0.13)"
_DISAGREE_LINE_A = "rgba(31,  119, 180, 0.50)"
_DISAGREE_LINE_B = "rgba(255, 127,  14, 0.50)"

# Isolated-pulse diamond borders (darker shade of each detector colour)
_PULSE_BORDER_A = "rgba(15, 75, 130, 0.90)"
_PULSE_BORDER_B = "rgba(180, 80, 0, 0.90)"

# Disagreement rectangles extend this far past the pair's two lanes
_RECT_PAD = 0.4

# Y-axis: padding beyond the outermost detector rows
_Y_PAD = 0.6

# Hard-reset (gap marker) lines
_GAP_LINE_COLOR = "rgba(120, 120, 120, 0.70)"

# Figure height: fixed margins plus a per-pair allowance, with a floor.
_MARGIN_TOP    = 90
_MARGIN_BOTTOM = 160
_PX_PER_PAIR   = 64
_MIN_PLOT_PX   = 320
_LEGEND_GAP_PX = 40      # legend top sits this far below the plot area

# Per-pair summary text, right of the plot area
_SUMMARY_FONT_COLOR = "rgba(90, 90, 90, 1)"

# Columns identifying a pair; an anomaly row is matched to its lane on these.
_PAIR_KEY = ["phase", "det_a_id", "det_b_id"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pair_lanes(detector_pairs: List[Dict[str, int]]) -> pd.DataFrame:
    """Assign Y-axis lanes to each pair, keyed by the pair's full identity.

    Lanes are keyed by ``(phase, det_a, det_b)`` rather than by detector ID,
    so a detector that appears in several pairs gets its own lane in each
    and every anomaly lands in the band of the pair that produced it.

    Args:
        detector_pairs: De-duplicated ``{"phase", "det_a", "det_b"}`` dicts,
            ordered top-to-bottom.

    Returns:
        DataFrame with columns ``['phase', 'det_a_id', 'det_b_id', 'y_a',
        'y_b']``, one row per pair in input order.
    """
    n = len(detector_pairs)
    base = (n - 1 - np.arange(n)) * _BAND_HEIGHT
    return pd.DataFrame({
        "phase":    [p["phase"] for p in detector_pairs],
        "det_a_id": [p["det_a"] for p in detector_pairs],
        "det_b_id": [p["det_b"] for p in detector_pairs],
        "y_a":      (base + _DET_A_OFFSET).astype(float),
        "y_b":      (base + _DET_B_OFFSET).astype(float),
    })


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


def _epochs_to_axis(epoch_arr: np.ndarray, tz: str) -> np.ndarray:
    """Convert UTC epoch seconds to local wall-clock milliseconds for a date axis.

    Plotly reads a number on a date axis as milliseconds since 1970 and
    shows it without any timezone shift, so shifting each value by its local
    UTC offset (DST-aware) makes the axis read in local time.  Numeric
    arrays are base64-encoded by Plotly, which keeps full-day figures far
    smaller than per-point date strings.  NaN entries stay NaN (line breaks).

    Args:
        epoch_arr: 1-D float array of UTC epoch seconds (may contain NaN).
        tz: pytz-compatible timezone string (e.g. ``'US/Mountain'``).

    Returns:
        Float ndarray of local wall-clock epoch milliseconds.
    """
    local = _epochs_to_local(np.asarray(epoch_arr, dtype=float), tz).tz_localize(None)
    ms = local.to_numpy(dtype="datetime64[ms]").astype("int64").astype(float)
    ms[np.isnat(local.to_numpy())] = np.nan
    return ms


def _broken_lines(points: List[np.ndarray]) -> np.ndarray:
    """Interleave per-item point arrays and append a NaN break after each item.

    Args:
        points: ``k`` equal-length float arrays; item ``i``'s vertices are
            ``points[0][i], ..., points[k-1][i]``.

    Returns:
        Flat float array of length ``n * (k + 1)`` for a None-gap style trace.
    """
    n, k = len(points[0]), len(points)
    out = np.full(n * (k + 1), np.nan)
    for j, arr in enumerate(points):
        out[j::k + 1] = arr
    return out


def _build_scatter_coords(
    intervals_df: pd.DataFrame,
    y_value: float,
    label: str,
    tz: str,
) -> tuple[list, list, list[str]]:
    """Vectorise intervals -> flat Scatter ``(x, y, hover)`` using the None-gap pattern.

    Each actuation is represented as ``[on_ts, mid_ts, off_ts, None]`` so
    Plotly draws exactly one line segment per actuation with no connecting
    line between successive actuations.  The midpoint vertex gives long
    actuations a hover target in their middle, not only at their ends.
    X values are converted to local wall-clock milliseconds so Plotly
    renders local time.

    Args:
        intervals_df: Output of ``_reconstruct_intervals`` -- columns
            ``['on_ts', 'off_ts', 'duration_sec']``.
        y_value: Fixed Y position for all segments in this lane.
        label: Lane label shown on the first hover line (e.g. ``"Ph2 Det 42"``).
        tz: pytz-compatible timezone string for x-axis conversion.

    Returns:
        Tuple ``(x, y, hover)`` arrays ready for ``go.Scatter``; ``x`` is in
        local wall-clock ms (see :func:`_epochs_to_axis`).  Empty arrays when
        ``intervals_df`` is empty.
    """
    if intervals_df.empty:
        return np.empty(0), np.empty(0), np.empty(0, dtype=object)

    n = len(intervals_df)
    on_ts  = intervals_df["on_ts"].to_numpy(float)
    off_ts = intervals_df["off_ts"].to_numpy(float)
    on_local  = _epochs_to_local(on_ts,  tz)
    off_local = _epochs_to_local(off_ts, tz)

    # 4 slots per interval: on_ts, midpoint, off_ts, NaN (segment break)
    x = _broken_lines([
        _epochs_to_axis(on_ts, tz),
        _epochs_to_axis((on_ts + off_ts) / 2.0, tz),
        _epochs_to_axis(off_ts, tz),
    ])
    y = np.where(np.isnan(x), np.nan, y_value)

    txt = (
        f"{label}<br>ON:  "
        + pd.Series(on_local.strftime("%H:%M:%S.%f")).str[:-3]
        + "<br>OFF: "
        + pd.Series(off_local.strftime("%H:%M:%S.%f")).str[:-3]
        + "<br>Dur: "
        + pd.Series(intervals_df["duration_sec"].to_numpy(float)).map("{:.3f} s".format)
    ).to_numpy(dtype=object)
    hover = np.empty(n * 4, dtype=object)
    hover[0::4] = txt
    hover[1::4] = txt
    hover[2::4] = txt
    hover[3::4] = ""

    return x, y, hover


def _pair_summaries(
    anomalies_df: pd.DataFrame,
    lanes: pd.DataFrame,
    window: Tuple[float, float],
) -> pd.DataFrame:
    """Per-pair share of the window in disagreement and isolated-pulse count.

    Args:
        anomalies_df: Anomaly DataFrame from ``analyze_discrepancies``.
        lanes: Output of :func:`_pair_lanes`.
        window: ``(start_epoch, end_epoch)`` the percentage is taken over.

    Returns:
        ``lanes`` with added columns ``disagree_pct`` (float) and
        ``n_pulses`` (int); zeros for pairs without anomalies.
    """
    w_start, w_end = window
    out = lanes.copy()
    out["disagree_pct"] = 0.0
    out["n_pulses"] = 0
    if anomalies_df.empty or w_end <= w_start:
        return out

    an = anomalies_df.astype({c: int for c in _PAIR_KEY})
    in_window = (
        np.minimum(an["end_timestamp"].to_numpy(float), w_end)
        - np.maximum(an["start_timestamp"].to_numpy(float), w_start)
    ).clip(min=0.0)
    ext = (an["anomaly_type"] == "extended_disagreement").to_numpy()
    agg = pd.DataFrame({
        "phase":    an["phase"],
        "det_a_id": an["det_a_id"],
        "det_b_id": an["det_b_id"],
        "disagree_sec": np.where(ext, in_window, 0.0),
        "pulse": (an["anomaly_type"] == "isolated_pulse").to_numpy().astype(int),
    }).groupby(_PAIR_KEY, as_index=False).sum()

    out = out.drop(columns=["disagree_pct", "n_pulses"]).merge(agg, on=_PAIR_KEY, how="left")
    out["disagree_pct"] = out["disagree_sec"].fillna(0.0) / (w_end - w_start) * 100.0
    out["n_pulses"] = out["pulse"].fillna(0).astype(int)
    return out.drop(columns=["disagree_sec", "pulse"])


def _gap_marker_trace(
    events_df: pd.DataFrame,
    window: Tuple[float, float],
    y_range: Tuple[float, float],
    tz: str,
) -> Optional[go.Scatter]:
    """Vertical dashed lines at each hard reset (``event_code == -1``) in the window.

    Args:
        events_df: Raw events (may extend past the window).
        window: ``(start_epoch, end_epoch)``; markers outside are skipped.
        y_range: ``(y_min, y_max)`` the lines span.
        tz: pytz-compatible timezone string for x-axis conversion.

    Returns:
        A single ``go.Scatter`` using the None-gap pattern, or ``None`` when
        the window holds no gap markers.
    """
    ts = events_df.loc[events_df["event_code"] == -1, "timestamp"].to_numpy(float)
    ts = np.unique(ts[(ts >= window[0]) & (ts < window[1])])
    if not ts.size:
        return None

    local = _epochs_to_local(ts, tz)
    n = ts.size
    x_ms = _epochs_to_axis(ts, tz)
    x = _broken_lines([x_ms, x_ms])
    y = _broken_lines([np.full(n, y_range[0]), np.full(n, y_range[1])])
    txt = ("Hard reset<br>" + pd.Series(local.strftime("%H:%M:%S.%f")).str[:-3]).to_numpy(dtype=object)
    hover = np.empty(n * 3, dtype=object)
    hover[0::3] = txt
    hover[1::3] = txt
    hover[2::3] = ""

    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color=_GAP_LINE_COLOR, width=1.5, dash="dash"),
        name="Hard reset",
        hovertext=hover.tolist(),
        hoverinfo="text",
        showlegend=True,
    )


def _location_title(metadata: dict) -> str:
    """Intersection location from metadata keys.

    Format: ``"{major_route} ({major_road_name}) & {minor_route} ({minor_road_name})"``
    Components that are ``None`` / empty are omitted gracefully; with no road
    information at all, ``intersection_name`` is used.

    Args:
        metadata: Intersection metadata dict.

    Returns:
        Location string.
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
        return f"{major} & {minor}"
    if major:
        return major
    return metadata.get("intersection_name", "Unknown Intersection")


def _format_title(metadata: dict) -> str:
    """Build a dynamic intersection title from metadata keys.

    Args:
        metadata: Intersection metadata dict.

    Returns:
        :func:`_location_title`, prefixed with ``"Detector Comparison -- "``.
    """
    return f"Detector Comparison -- {_location_title(metadata)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_detector_comparison(
    events_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    detector_pairs: List[Dict[str, int]],
    metadata: Optional[dict] = None,
    window: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """Build an interactive Plotly figure comparing co-located detector pairs.

    For each ``{"phase": int, "det_a": int, "det_b": int}`` pair, actuation
    intervals are reconstructed via
    :func:`~atspm.analysis.detectors._reconstruct_intervals` and rendered as
    horizontal line segments on a shared local-time axis.  Each pair owns a
    3-unit Y band (Det A lane above Det B lane), stacked top-to-bottom in
    list order; figure height grows with the number of pairs.

    Lanes are keyed by the whole pair ``(phase, det_a, det_b)``, never by a
    detector ID alone, so a detector that belongs to several pairs is drawn
    once per pair and each anomaly is placed in the band of the pair that
    produced it.  Exact duplicate pairs are drawn once.

    All Det A lanes share one trace and all Det B lanes share another; the
    hover text names the lane and is available along the whole actuation.
    A detector with no actuations in the window is labelled ``(no data)``.
    To the right of each band, the pair's share of the window spent in
    extended disagreement and its isolated-pulse count are shown.

    **Legend** -- one toggleable entry per overlay:

    * ``Det A ON (disagreement)`` / ``Det B ON (disagreement)`` -- the
      directional rectangles together with their hover points.
    * ``Isolated Pulse (Det A)`` / ``Isolated Pulse (Det B)`` -- diamond
      markers on the source detector's lane.
    * ``Hard reset`` -- dashed vertical line at each gap marker
      (``event_code == -1``) in the window.

    **Anomaly overlays**:

    * ``extended_disagreement`` -- translucent rectangle spanning the pair's
      two lanes, faint blue when Det A was ON, faint orange when Det B was
      ON.  Rectangles are drawn as one filled trace per side (not layout
      shapes, which slow rendering at full-day scale), beneath the lanes,
      with an invisible hover point at each midpoint whose marker colour
      matches the border (so the hover box border matches).
    * ``isolated_pulse`` -- diamond, colour-matched to the source detector,
      on the source detector's lane.

    Args:
        events_df: Raw ATSPM events with columns
            ``['timestamp', 'event_code', 'parameter']``.
            Codes: 82 = ON, 81 = OFF, -1 = gap marker.  May extend past
            ``window`` so actuations crossing its edges are drawn whole.
        anomalies_df: DataFrame returned by
            :func:`~atspm.analysis.detectors.analyze_discrepancies`.
            Pass an empty DataFrame to suppress all overlays.  Rows whose
            ``(phase, det_a_id, det_b_id)`` match no pair are ignored.
        detector_pairs: List of ``{"phase": int, "det_a": int, "det_b": int}``
            dicts ordered as they should appear top-to-bottom on the plot.
        metadata: Intersection metadata dict.  Keys used: ``major_road_route``,
            ``major_road_name``, ``minor_road_route``, ``minor_road_name``,
            ``intersection_name``, ``timezone``.  Defaults to ``{}`` when
            ``None``.
        window: Optional ``(start_epoch, end_epoch)`` display window.  Sets
            the x-axis range and the base of the disagreement percentage.
            Defaults to the span of ``events_df``.

    Returns:
        Plotly ``go.Figure``.  Caller is responsible for all I/O.
    """
    if metadata is None:
        metadata = {}

    tz: str = metadata.get("timezone") or DEFAULT_TIMEZONE

    fig = go.Figure()
    pairs = _unique_pairs(detector_pairs)

    if events_df.empty or not pairs:
        fig.update_layout(
            title=_format_title(metadata),
            xaxis_title="Time",
            yaxis_title="Detector",
            template="plotly_white",
        )
        return fig

    events_sorted = events_df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if window is None:
        ts = events_sorted["timestamp"].to_numpy(float)
        window = (float(ts.min()), float(ts.max()))
    w_start, w_end = window

    lanes = _pair_lanes(pairs)
    y_min = _DET_B_OFFSET - _Y_PAD
    y_max = (len(pairs) - 1) * _BAND_HEIGHT + _DET_A_OFFSET + _Y_PAD

    # Reconstruct each detector once, however many pairs share it.
    det_ids = pd.unique(lanes[["det_a_id", "det_b_id"]].to_numpy().ravel())
    intervals = {int(d): _reconstruct_intervals(events_sorted, int(d)) for d in det_ids}

    # ------------------------------------------------------------------
    # Anomaly traces: rectangles go beneath the lanes, markers above.
    # ------------------------------------------------------------------
    below, above = ([], [])
    if not anomalies_df.empty:
        below, above = _anomaly_traces(anomalies_df, lanes, tz)
    for trace in below:
        fig.add_trace(trace)

    # ------------------------------------------------------------------
    # Detector lanes -- one trace per side (A / B) covering every pair.
    # ------------------------------------------------------------------
    y_tick_vals:  list[float] = []
    y_tick_texts: list[str]   = []
    side_parts: Dict[str, list] = {"a": [], "b": []}

    for lane in lanes.itertuples(index=False):
        for side, det, y in (("a", lane.det_a_id, lane.y_a), ("b", lane.det_b_id, lane.y_b)):
            label = f"Ph{lane.phase} Det {det}"
            ivs = intervals[int(det)]
            side_parts[side].append(_build_scatter_coords(ivs, y, label, tz))
            has_data = bool(
                ((ivs["off_ts"] > w_start) & (ivs["on_ts"] < w_end)).any()
            ) if not ivs.empty else False
            y_tick_vals.append(y)
            y_tick_texts.append(label + ("" if has_data else " (no data)"))

    for side, color in (("a", _COLOR_A), ("b", _COLOR_B)):
        xs, ys, hs = zip(*side_parts[side])
        fig.add_trace(go.Scatter(
            x=np.concatenate(xs),
            y=np.concatenate(ys),
            mode="lines",
            line=dict(color=color, width=_LINE_WIDTH),
            name=f"Det {side.upper()}",
            showlegend=False,          # labelled by Y-axis tick text
            hovertext=np.concatenate(hs).tolist(),
            hoverinfo="text",
        ))

    gap_trace = _gap_marker_trace(events_sorted, window, (y_min, y_max), tz)
    if gap_trace is not None:
        fig.add_trace(gap_trace)

    for trace in above:
        fig.add_trace(trace)

    # ------------------------------------------------------------------
    # Per-pair summary, right of each band
    # ------------------------------------------------------------------
    summaries = _pair_summaries(anomalies_df, lanes, window)
    annotations = [
        dict(
            xref="paper",
            x=1.0,
            xanchor="left",
            xshift=8,
            yref="y",
            y=(row.y_a + row.y_b) / 2.0,
            text=(
                f"{row.disagree_pct:.1f}% disagree<br>"
                f"{row.n_pulses} pulse{'' if row.n_pulses == 1 else 's'}"
            ),
            showarrow=False,
            align="left",
            font=dict(size=10, color=_SUMMARY_FONT_COLOR),
        )
        for row in summaries.itertuples(index=False)
    ]

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    plot_px = max(_MIN_PLOT_PX, len(pairs) * _PX_PER_PAIR)
    height = _MARGIN_TOP + _MARGIN_BOTTOM + plot_px
    fig.update_layout(
        title=dict(
            text=_format_title(metadata),
            font=dict(size=16),
            x=0.5,
            xanchor="center",
            # yref="paper" + y=1.0 anchors to the very top of the paper
            # coordinate system.  The top margin reserves enough whitespace
            # above the plot area so the title text sits entirely outside it
            # and never overlaps chart content.
            yref="paper",
            y=1.0,
            yanchor="bottom",
        ),
        height=height,
        xaxis=dict(
            title="Time",              # timezone omitted -- assumed by viewer
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
            zeroline=False,
            type="date",
            range=_epochs_to_axis(np.array([w_start, w_end]), tz).tolist(),
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
        annotations=annotations,
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-_LEGEND_GAP_PX / plot_px,   # paper units are plot-area fractions
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(150,150,150,0.4)",
            borderwidth=1,
            font=dict(size=12),
        ),
        # The bottom margin keeps the below-chart legend from being clipped;
        # the right margin holds the per-pair summary text.
        margin=dict(l=140, r=110, t=_MARGIN_TOP, b=_MARGIN_BOTTOM),
    )

    return fig


# ---------------------------------------------------------------------------
# Anomaly overlay helper (module-private)
# ---------------------------------------------------------------------------

def _anomaly_traces(
    anomalies_df: pd.DataFrame,
    lanes: pd.DataFrame,
    tz: str,
) -> tuple[list[go.Scatter], list[go.Scatter]]:
    """Build the anomaly overlay traces.

    Each anomaly is joined to its lane on ``(phase, det_a_id, det_b_id)``,
    so its position comes from the pair that produced it.  Per side
    (Det A / Det B) this yields one filled rectangle trace, one invisible
    hover-point trace sharing its legend group, and one diamond trace for
    isolated pulses.

    Args:
        anomalies_df: Anomaly DataFrame from ``analyze_discrepancies``.
        lanes: Output of :func:`_pair_lanes`.
        tz: pytz-compatible timezone string for x-axis timestamp conversion.

    Returns:
        ``(below, above)`` -- rectangle traces to draw beneath the detector
        lanes, and hover/pulse traces to draw on top of them.
    """
    below: list[go.Scatter] = []
    above: list[go.Scatter] = []

    an = anomalies_df.astype({c: int for c in _PAIR_KEY + ["on_det_id"]})
    an = an.merge(lanes, on=_PAIR_KEY, how="inner")
    if an.empty:
        return below, above

    t_start = an["start_timestamp"].to_numpy(float)
    t_end   = an["end_timestamp"].to_numpy(float)
    x0      = _epochs_to_axis(t_start, tz)
    x1      = _epochs_to_axis(t_end, tz)
    x_mid   = _epochs_to_axis((t_start + t_end) / 2.0, tz)
    y_a     = an["y_a"].to_numpy()
    y_b     = an["y_b"].to_numpy()
    desc    = an["description"].astype(str).to_numpy(dtype=object)
    atype   = an["anomaly_type"].to_numpy()
    from_a  = an["on_det_id"].to_numpy() == an["det_a_id"].to_numpy()

    is_ext   = atype == "extended_disagreement"
    is_pulse = atype == "isolated_pulse"

    sides = (
        ("A", from_a,  _DISAGREE_FILL_A, _DISAGREE_LINE_A, _COLOR_A, _PULSE_BORDER_A, y_a),
        ("B", ~from_a, _DISAGREE_FILL_B, _DISAGREE_LINE_B, _COLOR_B, _PULSE_BORDER_B, y_b),
    )

    for side, on_side, fill, border, color, pulse_border, src_y in sides:
        # ---- Extended disagreement: filled rectangles + hover points ---
        idx = np.flatnonzero(is_ext & on_side)
        if idx.size:
            group = f"disagree_{side}"
            lo = y_b[idx] - _RECT_PAD
            hi = y_a[idx] + _RECT_PAD

            # Closed outline per rectangle: 5 corners + NaN break.
            rx = _broken_lines([x0[idx], x1[idx], x1[idx], x0[idx], x0[idx]])
            ry = _broken_lines([lo, lo, hi, hi, lo])

            below.append(go.Scatter(
                x=rx,
                y=ry,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line=dict(color=border, width=1.5, dash="dot"),
                name=f"Det {side} ON (disagreement)",
                legendgroup=group,
                showlegend=True,
                hoverinfo="skip",
            ))
            # Invisible hover points.  marker.color = the rectangle border
            # colour, so Plotly draws the hover-box border in that colour.
            above.append(go.Scatter(
                x=x_mid[idx],
                y=(y_a[idx] + y_b[idx]) / 2.0,
                mode="markers",
                marker=dict(symbol="circle", size=8, opacity=0, color=border),
                hovertext=["Extended Disagreement<br>" + d for d in desc[idx]],
                hoverinfo="text",
                legendgroup=group,
                showlegend=False,
                name="",
            ))

        # ---- Isolated pulses: diamonds on the source detector's lane ----
        idx = np.flatnonzero(is_pulse & on_side)
        if idx.size:
            above.append(go.Scatter(
                x=x_mid[idx],
                y=src_y[idx],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=_PULSE_MARKER_SIZE,
                    color=color,
                    line=dict(color=pulse_border, width=1),
                ),
                name=f"Isolated Pulse (Det {side})",
                hovertext=["Isolated Pulse<br>" + d for d in desc[idx]],
                hoverinfo="text",
                showlegend=True,
            ))

    return below, above
