"""
Timing and Actuation Plot (Functional Core)

Pure function rendering the row layout and intervals from
:mod:`atspm.analysis.timing_actuation` as one Plotly figure: per-phase
green / yellow / red bars, phase calls, ped service, detector on-intervals
grouped under their phase by role, preempt rows on top, hard resets as
dashed lines and detector-health findings overlaid.

Package Location: src/atspm/plotting/timing_actuation.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .detectors import _broken_lines, _epochs_to_axis, _epochs_to_local, _location_title

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

# (row kind, style key) -> (legend name, colour, line width px).  Detector
# rows are keyed by role; every other row by interval state.  Detector role
# colours follow the coordination plot's arrival / occupancy / stop-bar.
_STYLES: Dict[Tuple[str, str], Tuple[str, str, int]] = {
    ("preempt_call", "call"): ("Preempt call", "#c51b8a", 6),
    ("preempt", "entry"):     ("Preempt entry", "#fa9fb5", 12),
    ("preempt", "track"):     ("Preempt track clearance", "#f768a1", 12),
    ("preempt", "dwell"):     ("Preempt dwell", "#7a0177", 12),
    ("phase", "G"):           ("Green", "#2ca02c", 14),
    ("phase", "Y"):           ("Yellow", "#ffbf00", 14),
    ("phase", "RC"):          ("Red clearance", "#d62728", 14),
    ("phase", "R"):           ("Red", "rgba(214, 39, 40, 0.30)", 14),
    ("call", "call"):         ("Phase call", "#1f77b4", 6),
    ("ped", "walk"):          ("Walk", "#17becf", 10),
    ("ped", "fdw"):           ("Ped clearance", "#ff7f0e", 10),
    ("detector", "arrival"):      ("Arrival det", "DimGray", 8),
    ("detector", "occupancy"):    ("Occupancy det", "Black", 8),
    ("detector", "stop_bar"):     ("Stop-bar det", "Crimson", 8),
    ("detector", "pairs"):        ("Pair det", "#9467bd", 8),
    ("detector", "tm"):           ("TM det", "#8c564b", 8),
    ("detector", "watchdog"):     ("Watchdog det", "#e377c2", 8),
    ("detector", "unconfigured"): ("Unconfigured det", "#7f7f7f", 8),
}

_STATE_NAMES = {
    "G": "Green", "Y": "Yellow", "RC": "Red clearance", "R": "Red",
    "call": "Call", "walk": "Walk", "fdw": "Ped clearance",
    "entry": "Entry", "track": "Track clearance", "dwell": "Dwell", "on": "On",
}

_MARK_STYLES = {
    "ped":     dict(symbol="triangle-up", size=9, color="#ff7f0e"),
    "preempt": dict(symbol="line-ns-open", size=14, color="#7a0177"),
}

_GAP_LINE_COLOR = "rgba(120, 120, 120, 0.70)"
_FINDING_COLORS = {"high": "#d62728", "low": "#ff7f0e", "info": "#7f7f7f"}
_BLOCK_SHADE = "rgba(0, 0, 0, 0.04)"

# Intersection-level findings (detector = -1) sit in a band above row 0.
_TOP_BAND_Y = -0.9

_MARGIN_TOP = 110
_MARGIN_BOTTOM = 60
_PX_PER_ROW = 22
_MIN_PLOT_PX = 260

_TIME_FMT = "%H:%M:%S.%f"


def _local_str(epochs: np.ndarray, tz: str) -> pd.Series:
    """Local ``HH:MM:SS.d`` strings for UTC epoch seconds."""
    return pd.Series(_epochs_to_local(np.asarray(epochs, dtype=float), tz).strftime(_TIME_FMT)).str[:-5]


def _style_key(df: pd.DataFrame) -> pd.Series:
    """Style key per joined interval row: role for detectors, else state."""
    return df["state"].where(df["kind"] != "detector", df["role"])


def _segment_trace(seg: pd.DataFrame, tz: str, style: Tuple[str, str, int]) -> go.Scatter:
    """One None-gap line trace for all intervals sharing a style.

    Each interval is drawn ``[start, mid, end, None]`` at its row's y; the
    midpoint gives long intervals a hover target in their middle.
    """
    name, color, width = style
    start = seg["start_ts"].to_numpy(float)
    end = seg["end_ts"].to_numpy(float)
    x = _broken_lines([
        _epochs_to_axis(start, tz),
        _epochs_to_axis((start + end) / 2.0, tz),
        _epochs_to_axis(end, tz),
    ])
    yv = seg["row"].to_numpy(float)
    y = _broken_lines([yv, yv, yv])

    note = (
        np.where(seg["open_start"].to_numpy(), "<br>start not logged", "")
        + np.where(seg["open_end"].to_numpy(), "<br>end not logged", "")
    )
    txt = (
        seg["label"].to_numpy(dtype=object)
        + "<br>" + seg["state"].map(_STATE_NAMES).fillna(seg["state"]).to_numpy(dtype=object)
        + "<br>" + _local_str(start, tz).to_numpy(dtype=object)
        + " → " + _local_str(end, tz).to_numpy(dtype=object)
        + "<br>" + pd.Series(end - start).map("{:.1f} s".format).to_numpy(dtype=object)
        + note
    )
    n = len(seg)
    hover = np.empty(n * 4, dtype=object)
    hover[0::4] = txt
    hover[1::4] = txt
    hover[2::4] = txt
    hover[3::4] = ""

    return go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=width),
        name=name, legendgroup=name,
        hovertext=hover.tolist(), hoverinfo="text",
        connectgaps=False,
    )


def _mark_traces(rows: pd.DataFrame, marks: pd.DataFrame, tz: str) -> list:
    """Ped-call and preempt-exit markers on their rows."""
    traces = []
    for kind, label, name in (("ped", "ped call", "Ped call"), ("preempt", "exit", "Preempt exit")):
        m = marks.loc[(marks["kind"] == kind) & (marks["label"] == label)]
        if m.empty:
            continue
        m = m.merge(rows.loc[rows["kind"] == kind, ["param", "row", "label"]].rename(
            columns={"label": "row_label"}), on="param", how="inner")
        if m.empty:
            continue
        ts = m["ts"].to_numpy(float)
        st = _MARK_STYLES[kind]
        traces.append(go.Scatter(
            x=_epochs_to_axis(ts, tz), y=m["row"].to_numpy(float), mode="markers",
            marker=dict(symbol=st["symbol"], size=st["size"], color=st["color"],
                        line=dict(width=2, color=st["color"])),
            name=name,
            hovertext=(m["row_label"] + "<br>" + name + "<br>" + _local_str(ts, tz)).tolist(),
            hoverinfo="text",
        ))
    return traces


def _gap_trace(marks: pd.DataFrame, y_range: Tuple[float, float], tz: str) -> Optional[go.Scatter]:
    """Vertical dashed lines at each hard reset in the window."""
    ts = marks.loc[marks["kind"] == "gap", "ts"].to_numpy(float)
    if not ts.size:
        return None
    n = ts.size
    x_ms = _epochs_to_axis(ts, tz)
    txt = ("Hard reset<br>" + _local_str(ts, tz)).to_numpy(dtype=object)
    hover = np.empty(n * 3, dtype=object)
    hover[0::3] = txt
    hover[1::3] = txt
    hover[2::3] = ""
    return go.Scatter(
        x=_broken_lines([x_ms, x_ms]),
        y=_broken_lines([np.full(n, y_range[0]), np.full(n, y_range[1])]),
        mode="lines",
        line=dict(color=_GAP_LINE_COLOR, width=1.5, dash="dash"),
        name="Hard reset", hovertext=hover.tolist(), hoverinfo="text",
    )


def _finding_trace(
    rows: pd.DataFrame,
    findings: pd.DataFrame,
    window: Tuple[float, float],
    tz: str,
) -> Optional[go.Scatter]:
    """Detector-health findings: ``x`` at ``ts``, or open diamond at the left edge.

    A finding lands on every row drawing its detector; an intersection-level
    one (``detector = -1``) in the band above the top row.  Findings with a
    ``ts`` outside the window, or naming a detector with no row, are skipped.
    """
    if findings is None or findings.empty:
        return None
    f = findings.copy()
    f["ts"] = f["ts"].astype(float)
    has_ts = f["ts"].notna()
    f = f.loc[~has_ts | ((f["ts"] >= window[0]) & (f["ts"] < window[1]))]
    if f.empty:
        return None

    det_rows = rows.loc[rows["kind"] == "detector", ["param", "row"]].rename(columns={"param": "detector"})
    on_rows = f.loc[f["detector"] != -1].merge(det_rows, on="detector", how="inner")
    top = f.loc[f["detector"] == -1].assign(row=_TOP_BAND_Y)
    f = pd.concat([on_rows, top], ignore_index=True)
    if f.empty:
        return None

    has_ts = f["ts"].notna().to_numpy()
    x_ep = np.where(has_ts, f["ts"].to_numpy(float), float(window[0]))
    when = np.where(has_ts, _local_str(np.nan_to_num(x_ep, nan=window[0]), tz).to_numpy(dtype=object),
                    "day-level")
    sev = f["severity"].astype(str)
    txt = (
        "<b>" + f["rule"].astype(str) + "</b> (" + sev + ")<br>"
        + "det " + f["detector"].astype(str) + " · " + pd.Series(when, index=f.index)
        + "<br>" + f["message"].astype(str)
    )
    return go.Scatter(
        x=_epochs_to_axis(x_ep, tz), y=f["row"].to_numpy(float), mode="markers",
        marker=dict(
            symbol=np.where(has_ts, "x", "diamond-open").tolist(),
            size=12, color=sev.map(_FINDING_COLORS).fillna("#7f7f7f").tolist(),
            line=dict(width=1.5, color=sev.map(_FINDING_COLORS).fillna("#7f7f7f").tolist()),
        ),
        name="Finding", hovertext=txt.tolist(), hoverinfo="text",
    )


def _block_shapes(rows: pd.DataFrame) -> list:
    """Shade every other block so phase groups read apart."""
    spans = rows.groupby("block", sort=False)["row"].agg(["min", "max"]).reset_index()
    return [
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y",
             y0=lo - 0.5, y1=hi + 0.5, fillcolor=_BLOCK_SHADE,
             line=dict(width=0), layer="below")
        for i, (lo, hi) in enumerate(zip(spans["min"], spans["max"])) if i % 2 == 0
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_timing_actuation(
    rows: pd.DataFrame,
    intervals: pd.DataFrame,
    marks: Optional[pd.DataFrame],
    window: Sequence[float],
    tz: str,
    metadata: Optional[Dict[str, Any]] = None,
    findings: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Timing-and-actuation figure: one row per phase state, call, ped or detector.

    Args:
        rows: Row layout from
            :func:`~atspm.analysis.timing_actuation.timing_actuation_rows`.
        intervals: ``intervals`` from
            :func:`~atspm.analysis.timing_actuation.timing_actuation_intervals`.
        marks: ``marks`` from the same call (ped calls, preempt exits, hard
            resets), or None.
        window: ``(start, end)`` UTC epoch seconds; the x range.
        tz: IANA zone; the x axis reads local wall-clock time.
        metadata: Intersection metadata for the title.
        findings: Optional detector-health findings to overlay
            (``FINDINGS_SCHEMA``).  Those with a ``ts`` in the window are
            drawn as ``x`` at ``ts``; day-level ones (``ts`` NaN) as an open
            diamond at the left edge of their detector's rows.

    Returns:
        Plotly Figure.  Rows top to bottom in ``rows.row`` order, y tick
        labels from ``rows.label``.  One line trace per style (phase state,
        call, ped state, preempt state, detector role), so the legend
        toggles a whole class.  Empty rows still get a labelled tick.
    """
    metadata = metadata or {}
    w0, w1 = float(window[0]), float(window[1])
    start_s, end_s = _epochs_to_local(np.array([w0, w1]), tz).strftime("%Y-%m-%d %H:%M")
    title = (
        f"{_location_title(metadata)} — Timing & Actuation"
        f"<br><sup>{start_s} → {end_s} ({tz})</sup>"
    )

    fig = go.Figure()
    if rows is None or rows.empty:
        fig.update_layout(title=dict(text=title), template="plotly_white")
        return fig

    marks = marks if marks is not None else pd.DataFrame(columns=["kind", "param", "label", "ts"])

    joined = intervals.merge(rows[["kind", "param", "row", "role", "label"]], on=["kind", "param"], how="inner")
    if not joined.empty:
        joined["_style"] = _style_key(joined)
        for key, style in _STYLES.items():
            seg = joined.loc[(joined["kind"] == key[0]) & (joined["_style"] == key[1])]
            if not seg.empty:
                fig.add_trace(_segment_trace(seg.sort_values(["row", "start_ts"], kind="mergesort"), tz, style))

    for tr in _mark_traces(rows, marks, tz):
        fig.add_trace(tr)

    n_rows = len(rows)
    has_top = findings is not None and not findings.empty and (findings["detector"] == -1).any()
    y_top = (_TOP_BAND_Y - 0.6) if has_top else -0.6
    y_bot = n_rows - 0.4

    gap = _gap_trace(marks, (y_top, y_bot), tz)
    if gap is not None:
        fig.add_trace(gap)
    ft = _finding_trace(rows, findings, (w0, w1), tz) if findings is not None else None
    if ft is not None:
        fig.add_trace(ft)

    plot_px = max(_MIN_PLOT_PX, _PX_PER_ROW * (n_rows + (1 if has_top else 0)))
    x_range = _epochs_to_axis(np.array([w0, w1]), tz)
    fig.update_layout(
        title=dict(text=title),
        template="plotly_white",
        height=_MARGIN_TOP + _MARGIN_BOTTOM + plot_px,
        margin=dict(t=_MARGIN_TOP, b=_MARGIN_BOTTOM, l=110, r=30),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left"),
        shapes=_block_shapes(rows),
        xaxis=dict(type="date", range=list(x_range), showgrid=True),
        yaxis=dict(
            tickmode="array",
            tickvals=rows["row"].tolist(),
            ticktext=rows["label"].tolist(),
            range=[y_bot, y_top],
            showgrid=False, zeroline=False, fixedrange=True,
        ),
    )
    return fig
