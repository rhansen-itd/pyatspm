"""
True-Time Axis (Functional Core)

Pure functions only. No I/O, no SQL, no side effects.

Maps controller label time onto true (head-unit host) time using the drift
series decoded from ``eos_set_time``'s clock marks (``analysis.clock_marks``):

    true = label - d(label)

where ``d`` is the controller's drift (controller minus true), interpolated
between drift samples within a segment and stepping by ``shift`` at each
clock set.  ``events`` keeps
controller labels; this map is applied on read (``docs/ROADMAP.md``, S2).

Segments and dead zones:
    The label axis is cut at every clock break.  Between two breaks lies one
    segment, over which ``d`` is linear between consecutive drift samples
    (the hourly checks) and held at the first / last sample beyond them.
    Holding beats carrying a trend on: an end slope from pulse-only samples
    (~0.15 s each) amplifies their noise, and a set's anchor and residual
    sit seconds apart.  The cost is up to ``_MAX_EXTRAPOLATION`` of
    unfollowed wander past a segment's outermost sample.
    Field controllers keep time off the 60 Hz line, whose frequency
    wanders: 701 drifted 0.5-1 s per hour in both directions within a day,
    so one line per segment missed by ~0.35 s where interpolating through
    the samples tracks it.  A sample is dropped only as a spike, more than
    ``_SPIKE_FLOOR`` from both neighbours in the same direction.  A
    break owns a *dead zone* of labels that are not mapped (NaN):

    - **Decoded set** (``shift`` known): labels that the step makes
      ambiguous or that the edits may have written mid-correction.  For a
      backward set that is the replayed band, where two real moments share
      labels; file order is gone by the time rows reach SQLite, so they are
      flagged rather than split.  When the correction crosses a minute (or
      hour), the edits go through intermediate labels anywhere in the
      pre-step minute (hour), so the zone grows to cover it.  The segment
      after a decoded set opens at the zone's end, anchored by
      ``drift_pre + shift``.
    - **Undecoded set**: the bracket padded by ``_UNDECODED_PAD``; the next
      segment opens only at its first drift sample.
    - **Unmarked backward-step fence** (``parameter = -2`` with no set to
      explain it) and **comms gap** (``parameter = -1``): the segment before
      ends at the marker and the next one opens only at its first drift
      sample (CLAUDE.md section 5: never interpolate across a gap marker).

    A segment is also cut ``_MAX_EXTRAPOLATION`` beyond its outermost
    sample, so a site whose pulses stop is not extrapolated indefinitely.

Gap Marker Rule:
    ``apply_true_time`` drops unmapped rows and inserts an ``event_code =
    -1`` marker wherever mapped rows on either side of it belong to
    different segments or have dropped rows between them, so no consumer
    pairs across a dead zone.

Known limit: a correction that crosses a day boundary is not widened past
the hour; ``eos_set_time`` sets at 09:17 UTC, far from local midnight.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .decoders import CLOCK_STEP_FENCE_PARAM, COMMS_GAP_PARAM


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1

# An interior sample further than this from the line through its
# neighbours, on the same side of both, is dropped as a spike.  Line-
# frequency wander moved 701's drift 0.8 s off that line in the field, so
# this only catches a misdecoded pulse (e.g. a flipped sign).
_SPIKE_FLOOR: float = 2.0

# A segment is mapped at most this far beyond its outermost sample (two
# missed hourly checks).
_MAX_EXTRAPOLATION: float = 7200.0

# Padding around an undecoded bracket, whose shift is unknown.
_UNDECODED_PAD: float = 60.0

# A backward set's fence sits at the first post-step label, which is up to
# |shift| before the bracket ON; look this far back for it.
_FENCE_LOOKBACK: float = 600.0

# Fences sit 0.05 s below the event they fence.
_FENCE_TOL: float = 0.1

# Field edits replace whole seconds, then minutes, then hours.
_EDIT_UNITS: Tuple[float, ...] = (60.0, 3600.0)

_USABLE_DRIFT = ("ok", "send_log")

# Break kinds, least to most severe (a merged zone takes the worst name).
_KIND_SEVERITY = {"set": 0, "set_undecoded": 1, "fence": 2, "gap": 3}

MODEL_COLUMNS = [
    "segment", "seg_start", "seg_end", "t_ref", "intercept", "slope",
    "n_samples", "span", "resid_mad", "opened_by", "closed_by",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def drift_model(
    drift_df: pd.DataFrame,
    sets_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    window: Tuple[float, float],
) -> pd.DataFrame:
    """Model the controller's drift, piecewise linear between drift samples.

    Args:
        drift_df: Drift samples from ``decode_clock_marks`` (label time).
            Rows with status ``'ok'`` or ``'send_log'`` are used; the send
            log's edge-timed ``drift_host`` is preferred where present.
        sets_df: Clock sets from ``decode_clock_marks``.
        gaps_df: Gap-marker rows ``[timestamp, parameter]`` (``event_code =
            -1``) over the same window.
        window: ``(lo, hi)`` label-time range the inputs were fetched over.
            The caller must fetch back to the previous break so the first
            segment's samples are all present.

    Returns:
        DataFrame with ``MODEL_COLUMNS``, one row per piece, sorted and
        non-overlapping.  ``d(t) = intercept + slope * (t - t_ref)`` on
        ``[seg_start, seg_end)``.  Pieces of one segment share ``segment``
        and are contiguous; a piece runs between two drift samples, or holds
        the end sample's drift (``slope = 0``) out to the segment's bounds.
        ``n_samples``, ``span`` and ``resid_mad`` describe the whole
        segment; ``resid_mad`` is the median miss of each interior sample
        against the line through its neighbours (NaN under three samples),
        i.e. how well the sampling rate follows the clock.  ``opened_by`` /
        ``closed_by`` name the bounding break on a segment's first / last
        piece: ``'window'``, ``'set'``, ``'set_undecoded'``, ``'fence'``,
        ``'gap'`` or ``'extrapolation'``; ``'knot'`` between pieces.
    """
    lo_w, hi_w = float(window[0]), float(window[1])
    breaks = _build_breaks(sets_df, gaps_df)
    t, d, role = _usable_samples(drift_df)

    n_b = len(breaks)
    b_lo = breaks["lo"].to_numpy()
    b_hi = breaks["hi"].to_numpy()

    # Segment k lies between break k-1 and break k (k = 0..n_b).
    seg = np.searchsorted(b_hi, t, side="right")
    nxt = np.minimum(seg, max(n_b - 1, 0))
    inside = (seg < n_b) & (t >= b_lo[nxt]) if n_b else np.zeros(t.shape, bool)
    seg = np.where(inside & (role == "residual"), seg + 1, seg)
    keep = ~inside | np.isin(role, ("pre_set", "residual"))
    t, d, seg = t[keep], d[keep], seg[keep]

    # A decoded set anchors the segment after it at drift_pre + shift.
    anchored = ~breaks["strict"].to_numpy() & np.isfinite(breaks["anchor_d"].to_numpy())
    t = np.concatenate([t, b_hi[anchored]])
    d = np.concatenate([d, breaks["anchor_d"].to_numpy()[anchored]])
    seg = np.concatenate([seg, np.flatnonzero(anchored) + 1])

    kinds = breaks["kind"].to_numpy()
    strict = breaks["strict"].to_numpy()
    rows = []
    for k in np.unique(seg):
        sel = seg == k
        ts, ds = t[sel], d[sel]
        if k == 0:
            start, opened = lo_w, "window"
        elif strict[k - 1]:
            start, opened = max(b_hi[k - 1], ts.min()), kinds[k - 1]
        else:
            start, opened = b_hi[k - 1], kinds[k - 1]
        end, closed = (b_lo[k], kinds[k]) if k < n_b else (hi_w, "window")

        lo_x, hi_x = ts.min() - _MAX_EXTRAPOLATION, ts.max() + _MAX_EXTRAPOLATION
        if lo_x > start:
            start, opened = lo_x, "extrapolation"
        if hi_x < end:
            end, closed = hi_x, "extrapolation"
        start, end = max(start, lo_w), min(end, hi_w)
        if end <= start:
            continue

        kt, kd = _knots(ts, ds)
        edges = np.concatenate([[start], kt[(kt > start) & (kt < end)], [end]])
        d_e = np.interp(edges, kt, kd)
        n_p = edges.size - 1
        opened_p = np.full(n_p, "knot", dtype=object)
        closed_p = np.full(n_p, "knot", dtype=object)
        opened_p[0], closed_p[-1] = opened, closed
        rows.append(pd.DataFrame({
            "segment": len(rows), "seg_start": edges[:-1], "seg_end": edges[1:],
            "t_ref": edges[:-1], "intercept": d_e[:-1],
            "slope": np.diff(d_e) / np.diff(edges),
            "n_samples": int(sel.sum()), "span": float(ts.max() - ts.min()),
            "resid_mad": _interp_mad(kt, kd),
            "opened_by": opened_p, "closed_by": closed_p,
        }))
    if not rows:
        return pd.DataFrame(columns=MODEL_COLUMNS)
    return pd.concat(rows, ignore_index=True)[MODEL_COLUMNS]


def to_true_time(ts: np.ndarray, model: pd.DataFrame) -> np.ndarray:
    """Map controller labels to true time; NaN where no segment covers.

    Args:
        ts: Label times (UTC epoch floats).
        model: Output of ``drift_model``.

    Returns:
        Array of true times, NaN in dead zones and outside every segment.
    """
    ts = np.asarray(ts, dtype=float)
    return ts - _drift_at(ts, model)[0]


def apply_true_time(
    df: pd.DataFrame,
    model: pd.DataFrame,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Move an events frame onto the true-time axis.

    ``timestamp`` decides each row's fate: rows in no segment are dropped,
    and a gap marker (``event_code = -1``, ``parameter = COMMS_GAP_PARAM``)
    is inserted wherever consecutive kept rows change segment or straddle
    dropped rows.  Any *extra_columns* holding label times (e.g.
    ``cycle_start``) are mapped by their own label, NaN in a dead zone.
    Marker rows take the remaining columns from the kept row before them.

    Args:
        df: Events with ``[timestamp, event_code, parameter]`` (label time,
            UTC epoch floats), plus any other columns.
        model: Output of ``drift_model``.
        extra_columns: Further label-time columns to map.

    Returns:
        The kept rows plus markers in label order, which is true-time order
        within each segment and, the dead zones removed, across them.  New
        index.
    """
    if df.empty:
        return df.reset_index(drop=True)

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    d, seg = _drift_at(df["timestamp"].to_numpy(dtype=float), model)
    kept = seg >= 0

    out = df.copy()
    out["timestamp"] = out["timestamp"].to_numpy(dtype=float) - d
    for col in extra_columns:
        out[col] = to_true_time(out[col].to_numpy(dtype=float), model)

    # A kept row needs a marker before it when the kept row before it is in
    # another segment or dropped rows lie between them.
    pos = np.flatnonzero(kept)
    new_break = np.zeros(pos.size, dtype=bool)
    new_break[1:] = (seg[pos[1:]] != seg[pos[:-1]]) | (np.diff(pos) > 1)
    after = pos[new_break]
    before = pos[np.flatnonzero(new_break) - 1]

    out = out.loc[kept]
    if after.size == 0:
        return out.reset_index(drop=True)

    t_prev = out.loc[before, "timestamp"].to_numpy()
    t_next = out.loc[after, "timestamp"].to_numpy()
    markers = out.loc[before].copy()
    markers["timestamp"] = t_prev + 0.5 * (t_next - t_prev)
    markers["event_code"] = _GAP_CODE
    markers["parameter"] = COMMS_GAP_PARAM

    # Label order, each marker just ahead of the row it fences.
    key = np.concatenate([2 * after.astype(float) - 1, 2 * pos.astype(float)])
    combined = pd.concat([markers, out], ignore_index=True)
    combined = combined.iloc[np.argsort(key, kind="stable")]
    combined = combined.astype({"event_code": df["event_code"].dtype,
                                "parameter": df["parameter"].dtype})
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _usable_samples(drift_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(t, d, role)`` for drift samples with a known drift."""
    if drift_df.empty:
        return np.empty(0), np.empty(0), np.empty(0, dtype=object)
    host = drift_df["drift_host"].to_numpy(dtype=float)
    d = np.where(np.isfinite(host), host, drift_df["drift"].to_numpy(dtype=float))
    t = drift_df["ts"].to_numpy(dtype=float)
    ok = drift_df["status"].isin(_USABLE_DRIFT).to_numpy() & np.isfinite(d) & np.isfinite(t)
    return t[ok], d[ok], drift_df["role"].to_numpy(dtype=object)[ok]


def _build_breaks(sets_df: pd.DataFrame, gaps_df: pd.DataFrame) -> pd.DataFrame:
    """Merge every clock break into sorted, non-overlapping dead zones.

    Returns a frame ``[lo, hi, kind, strict, anchor_d]``.  ``strict`` means
    the segment after it opens only at its first drift sample.
    """
    cols = ["lo", "hi", "kind", "strict", "anchor_d"]
    gap_ts = gaps_df["timestamp"].to_numpy(dtype=float) if not gaps_df.empty else np.empty(0)
    gap_param = gaps_df["parameter"].to_numpy() if not gaps_df.empty else np.empty(0)
    is_fence = gap_param == CLOCK_STEP_FENCE_PARAM
    fences = np.sort(gap_ts[is_fence])
    explained = np.zeros(fences.size, dtype=bool)

    parts = []
    if not sets_df.empty:
        on = sets_df["bracket_on"].to_numpy(dtype=float)
        off = sets_df["bracket_off"].to_numpy(dtype=float)
        shift = sets_df["shift"].to_numpy(dtype=float)
        decoded = np.isfinite(shift)

        # Decoded: the step lies in [step_lo, step_hi] (pre-step labels) and
        # post-step labels in the same range moved by shift.
        s_lo = sets_df["step_lo"].to_numpy(dtype=float)
        s_hi = sets_df["step_hi"].to_numpy(dtype=float)
        lo = np.fmin(np.fmin(s_lo, s_hi), np.fmin(s_lo + shift, s_hi + shift))
        hi = np.fmax(np.fmax(s_lo, s_hi), np.fmax(s_lo + shift, s_hi + shift))
        # A correction crossing a minute (hour) edits seconds first, leaving
        # labels anywhere in the pre-step minute (hour) until the next edit.
        for unit in _EDIT_UNITS:
            crosses = np.floor(lo / unit) != np.floor(hi / unit)
            lo = np.where(crosses, np.fmin(lo, np.floor(s_lo / unit) * unit), lo)
            hi = np.where(crosses, np.fmax(hi, (np.floor(s_hi / unit) + 1) * unit), hi)

        # Undecoded: the bracket, reaching back to its own fence if any.
        first = np.fmin(on, off)
        idx = np.searchsorted(fences, first - _FENCE_LOOKBACK, side="left")
        cand = fences[np.minimum(idx, max(fences.size - 1, 0))] if fences.size else np.full(first.shape, np.nan)
        has_fence = (idx < fences.size) & (cand <= np.fmax(on, off))
        depth = np.where(has_fence, np.fmax(first - cand, 0.0), 0.0)
        u_lo = np.where(has_fence, np.fmin(first, cand), first) - _UNDECODED_PAD
        u_hi = np.fmax(on, off) + depth + _UNDECODED_PAD

        lo = np.where(decoded, lo, u_lo)
        hi = np.where(decoded, hi, u_hi)
        valid = np.isfinite(lo) & np.isfinite(hi)
        anchor = sets_df["drift_pre"].to_numpy(dtype=float) + shift

        parts.append(pd.DataFrame({
            "lo": lo[valid], "hi": hi[valid],
            "kind": np.where(decoded, "set", "set_undecoded")[valid],
            "strict": ~decoded[valid],
            "anchor_d": np.where(decoded, anchor, np.nan)[valid],
        }))
        if fences.size:
            zl = np.where(decoded, lo - _FENCE_TOL, lo)[valid]
            zh = hi[valid]
            explained = ((fences[:, None] >= zl[None, :])
                         & (fences[:, None] <= zh[None, :])).any(axis=1)

    unexplained = fences[~explained]
    comms = gap_ts[~is_fence]
    parts.append(pd.DataFrame({
        "lo": np.concatenate([unexplained, comms]),
        "hi": np.concatenate([unexplained, comms]),
        "kind": ["fence"] * unexplained.size + ["gap"] * comms.size,
        "strict": True,
        "anchor_d": np.nan,
    }))
    breaks = pd.concat(parts, ignore_index=True).sort_values("lo", kind="stable")
    if breaks.empty:
        return pd.DataFrame(columns=cols).astype({"lo": float, "hi": float, "strict": bool, "anchor_d": float})

    # Merge overlapping zones.  A merged zone is strict if any part is, is
    # named after its most severe part, and keeps its last decoded anchor.
    lo = breaks["lo"].to_numpy()
    hi = np.maximum.accumulate(breaks["hi"].to_numpy())
    group = np.concatenate([[0], np.cumsum(lo[1:] > hi[:-1])])
    severity = breaks["kind"].map(_KIND_SEVERITY)
    merged = breaks.assign(_g=group, hi=hi, _sev=severity).groupby("_g", sort=True).agg(
        lo=("lo", "min"), hi=("hi", "max"), _sev=("_sev", "max"),
        strict=("strict", "any"), anchor_d=("anchor_d", "last"),
    )
    merged["kind"] = np.asarray(list(_KIND_SEVERITY))[merged["_sev"].to_numpy()]
    merged.loc[merged["strict"], "anchor_d"] = np.nan
    return merged.reset_index(drop=True)[cols]


def _knots(t: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted, unique interpolation knots with spikes removed.

    Samples sharing a label are averaged.  An interior sample more than
    ``_SPIKE_FLOOR`` off the line through its neighbours, and above (or
    below) both of them, is dropped.
    """
    kt, inv = np.unique(t, return_inverse=True)
    kd = np.bincount(inv, weights=d) / np.bincount(inv)
    if kt.size < 3:
        return kt, kd
    left, right = kd[1:-1] - kd[:-2], kd[1:-1] - kd[2:]
    miss = kd[1:-1] - _neighbour_line(kt, kd)
    spike = (np.abs(miss) > _SPIKE_FLOOR) & (np.sign(left) == np.sign(right))
    keep = np.concatenate([[True], ~spike, [True]])
    return kt[keep], kd[keep]


def _neighbour_line(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Each interior knot's drift as predicted by the line through its neighbours."""
    w = (t[1:-1] - t[:-2]) / (t[2:] - t[:-2])
    return d[:-2] + w * (d[2:] - d[:-2])


def _interp_mad(t: np.ndarray, d: np.ndarray) -> float:
    """Median leave-one-out miss of interior knots; NaN under three knots."""
    if t.size < 3:
        return float("nan")
    return float(np.median(np.abs(d[1:-1] - _neighbour_line(t, d))))


def _drift_at(ts: np.ndarray, model: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the model at *ts*: ``(drift, segment id)``; NaN / -1 if uncovered."""
    if model.empty or ts.size == 0:
        return np.full(ts.shape, np.nan), np.full(ts.shape, -1, dtype=np.int64)
    start = model["seg_start"].to_numpy(dtype=float)
    end = model["seg_end"].to_numpy(dtype=float)
    k = np.searchsorted(start, ts, side="right") - 1
    kc = np.maximum(k, 0)
    covered = (k >= 0) & (ts < end[kc]) & np.isfinite(ts)
    d = (model["intercept"].to_numpy(dtype=float)[kc]
         + model["slope"].to_numpy(dtype=float)[kc]
         * (ts - model["t_ref"].to_numpy(dtype=float)[kc]))
    seg = model["segment"].to_numpy(dtype=np.int64)[kc]
    return np.where(covered, d, np.nan), np.where(covered, seg, -1)
