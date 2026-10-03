"""
Purdue Split Failure Analysis (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.
Input / output is DataFrames and plain Python scalars.

Algorithm overview
------------------
For each phase split window, presence-detector occupancy is measured over two
windows:

* **GOR** (green occupancy ratio): ``[green_ts, yellow_ts)``, Code 1 → Code 8.
  ``include_yellow=True`` extends it to the end of yellow (Code 9, else
  Code 10), which is the SPMs notebook definition.
* **ROR5** (red occupancy ratio): the first ``ror_seconds`` (default 5) of red,
  ``[yellow_end_ts, yellow_end_ts + 5)``, clipped at the phase's next Code 1.

A cycle fails when **both** ratios are strictly greater than ``threshold``
(UDOT default 0.79).

Each configured presence detector is one lane.  Occupancy is measured per lane
and then aggregated three ways, all always reported:

* ``union`` — the window is occupied while *any* lane's detector is on.  This
  is what a single multi-lane detector measures, and UDOT's method.  For
  independent lanes it rises with lane count as ``1 − Π(1 − o_i)``.
* ``mean`` — the average of the per-lane occupied seconds (equivalently, of
  the per-lane ratios, since every lane shares the window).
* ``any`` — the cycle fails when any lane fails on its own ratios, so the
  threshold keeps its single-detector meaning at every lane count.  It reports
  the *worst* lane's GOR/ROR5: the lane with the highest ``min(GOR, ROR5)``
  (ties: higher ``GOR + ROR5``, then the lowest channel), which is the lane
  that fails whenever any does.

``aggregate`` chooses which one drives ``gor`` / ``ror5`` / ``fail``.  The
per-lane frame and ``n_lanes_failed`` (lanes failing on their own) are
returned too.

Detector state
--------------
Lane on-intervals come from ``_reconstruct_intervals`` (repeated ONs extend an
interval, a stray OFF is ignored, a gap marker closes an open interval).  Two
states it cannot see are recovered *within the same data segment only*:

* the first event of a lane in a segment is an OFF (Code 81): the lane was on
  from the segment start until that OFF;
* the lane is still on at the end of the data: on until the last event.

A lane that logs no 81/82 at all in a window's segment has unknown state and
is excluded from that window (NaN), never counted as empty.  A window where
no lane has a known state is dropped.

Gap Marker Rule
---------------
Split windows come from ``_build_split_windows`` (``_build_phase_intervals``),
so green through clearance never spans a gap.  The red-5 window can run past
Code 11, so any cycle with a gap marker in ``[green_ts, ror_end]`` is dropped,
as is a cycle whose red-5 window runs past the end of the data.  Lane state
is never carried across a gap marker.

Event code reference
--------------------
    1   – Phase Begin Green
    8   – Phase Begin Yellow Clearance  (end of the GOR window)
    9   – Phase End Yellow Clearance    (start of the ROR window)
    10  – Phase Begin Red Clearance     (start of the ROR window w/o Code 9)
    11  – Phase End Red Clearance
    12  – Phase Inactive
    81  – Detector Off
    82  – Detector On
    -1  – Data gap marker (hard reset)

Package Location: src/atspm/analysis/split_failures.py
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .aog import _ts_to_float
from .detectors import _reconstruct_intervals
from .flow import _build_split_windows

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1
_CODE_GREEN = 1
_CODE_DET_OFF = 81
_CODE_DET_ON = 82

AGGREGATES = ("union", "mean", "any")

_CYCLE_SCHEMA = [
    "phase",
    "green_ts",
    "cycle_start",
    "coord_plan",
    "g_dur",
    "r_dur",
    "n_lanes",
    "g_occ_union",
    "r_occ_union",
    "g_occ_mean",
    "r_occ_mean",
    "gor_union",
    "ror5_union",
    "gor_mean",
    "ror5_mean",
    "g_occ_any",
    "r_occ_any",
    "gor_any",
    "ror5_any",
    "worst_det",
    "n_lanes_failed",
    "g_occ",
    "r_occ",
    "gor",
    "ror5",
    "fail",
]

_LANE_SCHEMA = ["phase", "green_ts", "det", "g_occ", "r_occ", "gor", "ror5", "fail"]

_BIN_SCHEMA = [
    "time",
    "phase",
    "coord_plan",
    "n_cycles",
    "n_fail",
    "sf_pct",
    "gor",
    "ror5",
    "g_dur",
    "g_occ",
    "r_dur",
    "r_occ",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _epoch_seconds(series: pd.Series) -> np.ndarray:
    """Vectorised UTC epoch seconds for epoch-float or tz-aware timestamps."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return (
            (series - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
        ).to_numpy(dtype=np.float64)
    return series.to_numpy(dtype=np.float64)


def _occupied(
    on: np.ndarray,
    off: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
) -> np.ndarray:
    """On-time inside each window ``[t0, t1)`` for disjoint sorted intervals.

    Uses the cumulative on-time ``F(t)``, so the result is ``F(t1) − F(t0)``
    with two ``searchsorted`` calls and no per-window loop.  Clipping at
    either window edge (and capping at the window length) is implicit.

    Args:
        on: Interval starts, sorted ascending, non-overlapping.
        off: Interval ends (same length as *on*).
        t0: Window starts.
        t1: Window ends.

    Returns:
        Float array of occupied seconds, one per window.
    """
    if len(on) == 0:
        return np.zeros(len(t0), dtype=np.float64)

    dur = off - on
    cum = np.concatenate(([0.0], np.cumsum(dur)))

    def _f(t: np.ndarray) -> np.ndarray:
        j = np.searchsorted(on, t, side="right") - 1
        jc = np.clip(j, 0, None)
        part = np.clip(t - on[jc], 0.0, dur[jc])
        return np.where(j >= 0, cum[jc] + part, 0.0)

    return _f(t1) - _f(t0)


def _merge_intervals(on: np.ndarray, off: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Union of possibly overlapping intervals, as disjoint sorted intervals."""
    if len(on) == 0:
        return on, off
    order = np.argsort(on, kind="stable")
    on, off = on[order], off[order]
    run_max = np.maximum.accumulate(off)
    new = np.empty(len(on), dtype=bool)
    new[0] = True
    new[1:] = on[1:] > run_max[:-1]
    starts = np.flatnonzero(new)
    return on[starts], np.maximum.reduceat(off, starts)


def _lane_intervals(
    ev: pd.DataFrame,
    det: int,
    gap_ts: np.ndarray,
    data_start: float,
    data_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """On-intervals of one lane, plus the segments in which it logged events.

    Wraps ``_reconstruct_intervals`` and recovers the two states it drops,
    within a segment only: a leading OFF (on since the segment start) and an
    interval still open at the end of the data.

    Args:
        ev: Events with float ``timestamp``, sorted; detector rows for the
            presence lanes plus every gap marker.
        det: Detector channel (``parameter``).
        gap_ts: Sorted gap-marker timestamps.
        data_start: First timestamp of the data (start of segment 0).
        data_end: Last timestamp of the data.

    Returns:
        ``(on, off, active_segments)``: sorted disjoint intervals and the
        unique segment ids with at least one 81/82 for this lane.
    """
    iv = _reconstruct_intervals(ev, det)
    on = iv["on_ts"].to_numpy(dtype=np.float64)
    off = iv["off_ts"].to_numpy(dtype=np.float64)

    rows = ev.loc[
        (ev["parameter"] == det)
        & ev["event_code"].isin((_CODE_DET_OFF, _CODE_DET_ON))
    ]
    if rows.empty:
        return on, off, np.empty(0, dtype=np.int64)

    ts = rows["timestamp"].to_numpy(dtype=np.float64)
    code = rows["event_code"].to_numpy()
    seg = np.searchsorted(gap_ts, ts, side="right")

    extra_on: List[np.ndarray] = []
    extra_off: List[np.ndarray] = []

    # Leading OFF in a segment: on from the segment start until that OFF.
    first = np.concatenate(([True], seg[1:] != seg[:-1]))
    lead = first & (code == _CODE_DET_OFF)
    if lead.any():
        seg_starts = np.concatenate(([data_start], gap_ts))
        extra_on.append(seg_starts[seg[lead]])
        extra_off.append(ts[lead])

    # Still on at the end of the data (last segment only; a gap marker
    # already closes an open interval in every earlier segment).
    last_seg = len(gap_ts)
    if seg[-1] == last_seg and code[-1] == _CODE_DET_ON:
        in_last = seg == last_seg
        offs = np.flatnonzero(in_last & (code == _CODE_DET_OFF))
        after = offs[-1] + 1 if len(offs) else np.flatnonzero(in_last)[0]
        extra_on.append(ts[after:after + 1])
        extra_off.append(np.array([data_end]))

    if extra_on:
        on = np.concatenate([on] + extra_on)
        off = np.concatenate([off] + extra_off)
        order = np.argsort(on, kind="stable")
        on, off = on[order], off[order]

    return on, off, np.unique(seg)


def _empty_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return empty (cycle_df, lane_df) with the documented schemas."""
    return pd.DataFrame(columns=_CYCLE_SCHEMA), pd.DataFrame(columns=_LANE_SCHEMA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_failures(
    events_df: pd.DataFrame,
    phase: int,
    detector_ids: List[int],
    threshold: float = 0.79,
    aggregate: str = "union",
    ror_seconds: float = 5.0,
    include_yellow: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cycle Purdue split failures for one phase over its presence lanes.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            Timestamps may be UTC epoch floats **or** tz-aware Timestamps.
            Must carry phase codes 1/8/9/10/11/12, detector codes 81/82 and
            gap markers (``event_code == -1``).
        phase: Signal phase number.
        detector_ids: Presence detector channels of *phase* at the stop
            line, one per lane (``Det_P{N}_Occupancy``).
        threshold: A cycle fails when GOR > threshold **and**
            ROR5 > threshold.  Default ``0.79`` (UDOT).
        aggregate: ``"union"``, ``"mean"`` or ``"any"``; selects which lane
            aggregate fills ``g_occ``, ``r_occ``, ``gor``, ``ror5`` and
            ``fail``.  All three are always reported in their own columns.
        ror_seconds: Length of the red window measured from the end of
            yellow.  Default ``5.0``.
        include_yellow: Measure GOR over green + yellow (Code 1 → end of
            yellow, the SPMs definition) instead of green only.

    Returns:
        ``(cycle_df, lane_df)``.

        ``cycle_df`` has one row per kept split window, sorted by
        ``green_ts``::

            phase           int
            green_ts        float | Timestamp  (Code 1; matches input dtype)
            cycle_start     float | Timestamp
            coord_plan      float
            g_dur           float  – GOR window seconds
            r_dur           float  – ROR window seconds (≤ ror_seconds)
            n_lanes         int    – lanes with a known state
            g_occ_union     float  – seconds any lane was on in GOR window
            r_occ_union     float
            g_occ_mean      float  – mean per-lane on-seconds
            r_occ_mean      float
            gor_union       float  – g_occ_union / g_dur
            ror5_union      float
            gor_mean        float
            ror5_mean       float
            g_occ_any       float  – the worst lane's on-seconds
            r_occ_any       float
            gor_any         float  – the worst lane's GOR
            ror5_any        float
            worst_det       int    – the worst lane (see module docstring)
            n_lanes_failed  int    – lanes failing on their own ratios
            g_occ, r_occ,
            gor, ror5       float  – the chosen aggregate's values
            fail            bool   – the chosen aggregate's verdict

        ``lane_df`` has one row per (window, lane with a known state)::

            phase, green_ts, det, g_occ, r_occ, gor, ror5, fail

    Raises:
        ValueError: If *aggregate* is not in :data:`AGGREGATES`.
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")

    dets = sorted({int(d) for d in detector_ids})
    if events_df.empty or not dets:
        return _empty_results()

    windows = _build_split_windows(events_df, phase)
    if windows.empty:
        return _empty_results()

    # --- Window bounds (float epoch) --------------------------------------
    g0 = _ts_to_float(windows["green_ts"])
    yel = _ts_to_float(windows["yellow_ts"])
    r0 = _ts_to_float(windows["yellow_end_ts"])
    g1 = r0 if include_yellow else yel

    all_ts = _epoch_seconds(events_df["timestamp"])
    data_start, data_end = float(all_ts.min()), float(all_ts.max())
    code_all = events_df["event_code"].to_numpy()
    param_all = events_df["parameter"].to_numpy()

    gap_ts = np.sort(all_ts[code_all == _GAP_CODE])
    greens = np.sort(all_ts[(code_all == _CODE_GREEN) & (param_all == phase)])

    nxt_idx = np.searchsorted(greens, g0, side="right")
    next_green = np.where(
        nxt_idx < len(greens), greens[np.clip(nxt_idx, 0, len(greens) - 1)], np.inf
    )
    r1 = np.minimum(r0 + ror_seconds, next_green)

    n_gaps = (
        np.searchsorted(gap_ts, r1, side="right")
        - np.searchsorted(gap_ts, g0, side="left")
    )
    keep = (r0 > yel) & (r1 > r0) & (r1 <= data_end) & (n_gaps == 0)
    if not keep.any():
        return _empty_results()

    windows = windows.loc[keep].reset_index(drop=True)
    g0, g1, r0, r1 = g0[keep], g1[keep], r0[keep], r1[keep]
    win_seg = np.searchsorted(gap_ts, g0, side="right")
    g_dur = g1 - g0
    r_dur = r1 - r0

    # --- Per-lane occupancy -------------------------------------------------
    ev_mask = (
        np.isin(param_all, dets) & np.isin(code_all, (_CODE_DET_OFF, _CODE_DET_ON))
    ) | (code_all == _GAP_CODE)
    ev = pd.DataFrame({
        "timestamp": all_ts[ev_mask],
        "event_code": code_all[ev_mask],
        "parameter": param_all[ev_mask],
    }).sort_values("timestamp", kind="stable").reset_index(drop=True)

    n_win = len(windows)
    lane_g = np.full((len(dets), n_win), np.nan)
    lane_r = np.full((len(dets), n_win), np.nan)
    all_on: List[np.ndarray] = []
    all_off: List[np.ndarray] = []

    for k, det in enumerate(dets):
        on, off, active = _lane_intervals(ev, det, gap_ts, data_start, data_end)
        known = np.isin(win_seg, active)
        lane_g[k] = np.where(known, _occupied(on, off, g0, g1), np.nan)
        lane_r[k] = np.where(known, _occupied(on, off, r0, r1), np.nan)
        all_on.append(on)
        all_off.append(off)

    known_lanes = ~np.isnan(lane_g)
    n_lanes = known_lanes.sum(axis=0)
    has_lane = n_lanes > 0
    if not has_lane.any():
        return _empty_results()

    # --- Aggregates ---------------------------------------------------------
    # Union: lanes with unknown state contribute no intervals in that segment
    # (gap markers close intervals; leading ones start at the segment start).
    u_on, u_off = _merge_intervals(np.concatenate(all_on), np.concatenate(all_off))
    g_union = _occupied(u_on, u_off, g0, g1)
    r_union = _occupied(u_on, u_off, r0, r1)

    denom = np.where(has_lane, n_lanes, 1)
    g_mean = np.where(has_lane, np.nansum(lane_g, axis=0) / denom, np.nan)
    r_mean = np.where(has_lane, np.nansum(lane_r, axis=0) / denom, np.nan)

    lane_gor = lane_g / g_dur
    lane_ror = lane_r / r_dur
    lane_fail = (lane_gor > threshold) & (lane_ror > threshold)

    # Worst lane per window: highest min(GOR, ROR5), then GOR + ROR5, then
    # the lowest channel.  np.lexsort's last key is primary; the last index
    # of each row is the maximum.
    margin = np.where(known_lanes, np.fmin(lane_gor, lane_ror), -np.inf)
    total = np.where(known_lanes, lane_gor + lane_ror, -np.inf)
    chan = np.broadcast_to(-np.asarray(dets, dtype=float)[:, None], margin.shape)
    order = np.lexsort((chan.T, np.nan_to_num(total.T, nan=-np.inf),
                        np.nan_to_num(margin.T, nan=-np.inf)))
    worst = order[:, -1]
    cols = np.arange(n_win)

    cyc = windows[["phase", "green_ts", "cycle_start", "coord_plan"]].copy()
    cyc["phase"] = int(phase)
    cyc["g_dur"] = g_dur
    cyc["r_dur"] = r_dur
    cyc["n_lanes"] = n_lanes.astype(int)
    cyc["g_occ_union"] = g_union
    cyc["r_occ_union"] = r_union
    cyc["g_occ_mean"] = g_mean
    cyc["r_occ_mean"] = r_mean
    cyc["gor_union"] = g_union / g_dur
    cyc["ror5_union"] = r_union / r_dur
    cyc["gor_mean"] = g_mean / g_dur
    cyc["ror5_mean"] = r_mean / r_dur
    cyc["g_occ_any"] = lane_g[worst, cols]
    cyc["r_occ_any"] = lane_r[worst, cols]
    cyc["gor_any"] = lane_gor[worst, cols]
    cyc["ror5_any"] = lane_ror[worst, cols]
    cyc["worst_det"] = np.asarray(dets, dtype=int)[worst]
    cyc["n_lanes_failed"] = lane_fail.sum(axis=0).astype(int)

    sfx = aggregate
    cyc["g_occ"] = cyc[f"g_occ_{sfx}"]
    cyc["r_occ"] = cyc[f"r_occ_{sfx}"]
    cyc["gor"] = cyc[f"gor_{sfx}"]
    cyc["ror5"] = cyc[f"ror5_{sfx}"]
    cyc["fail"] = (cyc["gor"] > threshold) & (cyc["ror5"] > threshold)

    cyc = cyc.loc[has_lane].reset_index(drop=True)

    # --- Lane frame -------------------------------------------------------------
    k_idx, w_idx = np.nonzero(known_lanes)
    lane_df = pd.DataFrame({
        "phase": int(phase),
        "green_ts": windows["green_ts"].to_numpy()[w_idx],
        "det": np.asarray(dets, dtype=int)[k_idx],
        "g_occ": lane_g[k_idx, w_idx],
        "r_occ": lane_r[k_idx, w_idx],
        "gor": lane_gor[k_idx, w_idx],
        "ror5": lane_ror[k_idx, w_idx],
        "fail": lane_fail[k_idx, w_idx],
    })
    lane_df = lane_df.sort_values(["green_ts", "det"]).reset_index(drop=True)

    return cyc[_CYCLE_SCHEMA], lane_df[_LANE_SCHEMA]


def bin_split_failures(
    cycle_df: pd.DataFrame,
    bin_len: int = 60,
) -> pd.DataFrame:
    """Aggregate per-cycle split failures into fixed-width time bins.

    GOR and ROR5 are **time-weighted**: occupied seconds and window seconds
    are summed within the bin, then divided (``Σ g_occ / Σ g_dur``), not
    averaged per cycle.  ``sf_pct = n_fail / n_cycles`` (a fraction, 0–1).
    Grouped per bin × phase × coord plan, using the chosen aggregate columns
    (``g_occ``, ``r_occ``, ``fail``) of :func:`split_failures`.

    Args:
        cycle_df: First element returned by :func:`split_failures`; may
            hold several phases.
        bin_len: Bin width in minutes.  Default ``60``.

    Returns:
        DataFrame with columns::

            time       Timestamp  – bin start (floor of green_ts)
            phase      int
            coord_plan float
            n_cycles   int
            n_fail     int
            sf_pct     float      – n_fail / n_cycles
            gor        float      – Σ g_occ / Σ g_dur
            ror5       float      – Σ r_occ / Σ r_dur
            g_dur, g_occ, r_dur, r_occ  float (summed seconds)

        Sorted by ``["phase", "time", "coord_plan"]``.  Empty (same schema)
        when *cycle_df* is empty.
    """
    if cycle_df is None or cycle_df.empty:
        return pd.DataFrame(columns=_BIN_SCHEMA)

    df = cycle_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["green_ts"]):
        df["_bin"] = df["green_ts"].dt.floor(f"{bin_len}min")
    else:
        df["_bin"] = (
            pd.to_datetime(df["green_ts"], unit="s", utc=True)
            .dt.floor(f"{bin_len}min")
        )
    df["_fail"] = df["fail"].astype(int)

    agg = (
        df.groupby(["_bin", "phase", "coord_plan"], sort=False)
        .agg(
            n_cycles=("_fail", "size"),
            n_fail=("_fail", "sum"),
            g_dur=("g_dur", "sum"),
            g_occ=("g_occ", "sum"),
            r_dur=("r_dur", "sum"),
            r_occ=("r_occ", "sum"),
        )
        .reset_index()
        .rename(columns={"_bin": "time"})
    )
    agg["sf_pct"] = agg["n_fail"] / agg["n_cycles"]
    agg["gor"] = agg["g_occ"] / agg["g_dur"]
    agg["ror5"] = agg["r_occ"] / agg["r_dur"]
    for col in ("n_cycles", "n_fail"):
        agg[col] = agg[col].astype(int)

    agg = agg.sort_values(["phase", "time", "coord_plan"]).reset_index(drop=True)
    return agg[_BIN_SCHEMA]
