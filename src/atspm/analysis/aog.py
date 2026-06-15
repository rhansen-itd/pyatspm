"""
Arrival on Green (AOG) Analysis (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.
Input / output is DataFrames and plain Python scalars.

Algorithm overview
------------------
Green windows are derived from phase state-change events using
``_build_phase_intervals`` (from ``atspm.analysis.phases``), which already
enforces the Gap Marker Rule: any phase interval that straddles a data
discontinuity (``event_code == -1``) is silently dropped.  The green window
for each cycle is the half-open interval ``[green_ts, yellow_ts)`` — Code 1
onset to Code 8 onset.

Detector actuation timestamps (Code 82) are optionally shifted forward by
``arrival_offset_sec`` seconds (a copy is made; the input is never mutated)
to account for travel time from an advance detector to the stop bar.
Shifted timestamps are then matched against the per-cycle green windows using
``np.searchsorted`` — the same O(n log m) vectorised containment pattern used
in ``atspm.analysis.detectors``.

Gap Marker Rule
---------------
``_build_phase_intervals`` guarantees that every green window it emits lies
entirely within a single contiguous data segment.  The searchsorted containment
check further ensures that a detector event can only match a green window whose
``green_ts`` ≤ shifted_event_ts < ``yellow_ts`` — so events on one side of a
gap can never be matched to a green window on the other side.

Event code reference
--------------------
    1   – Phase Begin Green
    8   – Phase Begin Yellow Clearance  (exclusive end of green window)
    9   – Phase End Yellow Clearance
    10  – Phase Begin Red Clearance
    11  – Phase End Red Clearance
    12  – Phase Inactive
    82  – Detector On  (arrival actuation)
    -1  – Data gap marker (hard reset)

Package Location: src/atspm/analysis/aog.py
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .phases import _build_phase_intervals, _segment_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1
_CODE_GREEN = 1
_CODE_YELLOW = 8
_CODE_DET_ON = 82

# Phase codes needed to drive _build_phase_intervals
_PHASE_CODES = frozenset({1, 8, 9, 10, 11, 12})

# Schema for an empty per-cycle result
_CYCLE_SCHEMA = [
    "phase",
    "coord_plan",
    "green_dur",
    "cycle_len",
    "green_pct",
    "total_arrivals",
    "aog",
    "aog_pct",
]

# Schema for an empty binned result
_BIN_SCHEMA = [
    "time",
    "phase",
    "coord_plan",
    "green_dur",
    "green_pct",
    "total_arrivals",
    "aog",
    "aog_pct",
    "n_cycles",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_green_windows(
    events_df: pd.DataFrame,
    phase: int,
) -> pd.DataFrame:
    """Extract per-cycle green windows for a single phase.

    Delegates to ``_build_phase_intervals`` from ``atspm.analysis.phases``,
    which owns the state-machine logic and gap-marker enforcement.  Only the
    columns required for AOG matching are retained.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            ``timestamp`` values may be UTC epoch floats **or** tz-aware
            ``Timestamps`` — both are handled by the dtype guard already
            present in ``_build_phase_intervals``.
        phase: Signal phase number (``parameter`` value to filter on).

    Returns:
        DataFrame with columns::

            phase         int
            cycle_start   float | Timestamp  (matches input dtype)
            coord_plan    float
            green_ts      float | Timestamp  (Code 1 onset)
            yellow_ts     float | Timestamp  (Code 8 onset — exclusive bound)
            green_dur     float              (seconds)

        One row per valid, gap-isolated green interval.  Returns an empty
        DataFrame with the correct schema when no valid intervals exist.
    """
    _EMPTY = pd.DataFrame(
        columns=["phase", "cycle_start", "coord_plan",
                 "green_ts", "yellow_ts", "green_dur"]
    )

    if events_df.empty:
        return _EMPTY

    # Filter to this phase's state codes + gap markers
    mask = (
        (events_df["event_code"].isin(_PHASE_CODES) & (events_df["parameter"] == phase))
        | (events_df["event_code"] == _GAP_CODE)
    )
    ph_df = (
        events_df.loc[mask]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if ph_df.empty:
        return _EMPTY

    # Assign segment IDs (increments at every gap marker row)
    ph_df["_seg"] = _segment_id(ph_df)

    # Strip gap marker rows before passing to the state machine
    ph_df = ph_df.loc[ph_df["event_code"] != _GAP_CODE].copy()

    if ph_df.empty:
        return _EMPTY

    # _build_phase_intervals expects [timestamp, event_code, parameter,
    # cycle_start, _seg].  coord_plan is extra and passes through harmlessly.
    intervals = _build_phase_intervals(ph_df, include_no_clearance=False)

    if intervals.empty:
        return _EMPTY

    # Restrict to the requested phase (the builder groups by parameter, so
    # this is typically a no-op, but it is an explicit safety guard).
    intervals = intervals.loc[intervals["phase"] == phase].copy()

    if intervals.empty:
        return _EMPTY

    # Attach coord_plan: take the coord_plan of the Code-1 event that opened
    # each green interval, matched by (cycle_start, phase).
    green_events = (
        events_df.loc[
            (events_df["event_code"] == _CODE_GREEN)
            & (events_df["parameter"] == phase)
        ]
        [["cycle_start", "coord_plan"]]
        .drop_duplicates(subset="cycle_start")
    )

    intervals = intervals.merge(green_events, on="cycle_start", how="left")
    intervals["coord_plan"] = (
        pd.to_numeric(intervals["coord_plan"], errors="coerce").fillna(0.0)
    )

    return intervals[
        ["phase", "cycle_start", "coord_plan", "green_ts", "yellow_ts", "green_dur"]
    ].reset_index(drop=True)


def _shift_detector_timestamps(
    events_df: pd.DataFrame,
    detector_ids: List[int],
    arrival_offset_sec: float,
) -> pd.DataFrame:
    """Return a copy of Code-82 rows with timestamps shifted by *arrival_offset_sec*.

    The original DataFrame is never mutated.  When ``arrival_offset_sec`` is
    zero the function still returns a filtered copy (no shift applied).

    Args:
        events_df: Flat events DataFrame.
        detector_ids: ``parameter`` values identifying the advance detectors.
        arrival_offset_sec: Seconds to add to each detector timestamp.
            Positive values move arrivals forward in time (accounting for
            travel time from an advance detector to the stop bar).

    Returns:
        DataFrame of Code-82 rows with shifted ``timestamp`` values, sorted
        by ``timestamp``.  Returns an empty DataFrame when no matching rows
        exist.
    """
    mask = (
        (events_df["event_code"] == _CODE_DET_ON)
        & (events_df["parameter"].isin(detector_ids))
    )
    det = events_df.loc[mask].copy()

    if det.empty:
        return det

    if arrival_offset_sec != 0.0:
        sample_ts = det["timestamp"].iloc[0]
        if hasattr(sample_ts, "timestamp"):
            # tz-aware Timestamp path — use pd.Timedelta
            det["timestamp"] = (
                det["timestamp"] + pd.Timedelta(seconds=arrival_offset_sec)
            )
        else:
            # epoch float path — plain addition
            det["timestamp"] = det["timestamp"] + arrival_offset_sec

    return det.sort_values("timestamp").reset_index(drop=True)


def _ts_to_float(series: pd.Series) -> np.ndarray:
    """Convert a Series of epoch floats or tz-aware Timestamps to a float64 array.

    Args:
        series: Timestamp Series (either dtype float or DatetimeTZDtype).

    Returns:
        numpy float64 array of UTC epoch seconds.
    """
    sample = series.iloc[0]
    if hasattr(sample, "timestamp"):
        return np.array([t.timestamp() for t in series], dtype=np.float64)
    return series.values.astype(np.float64)


def _cycle_lengths(cycle_starts: np.ndarray) -> np.ndarray:
    """Compute cycle length in seconds from an array of sorted cycle-start floats.

    The last entry is NaN (no successor cycle known).

    Args:
        cycle_starts: Sorted float array of UTC epoch cycle-start values.

    Returns:
        Float array of the same length; last element is NaN.
    """
    lengths = np.empty(len(cycle_starts), dtype=np.float64)
    lengths[:-1] = np.diff(cycle_starts)
    lengths[-1] = np.nan
    return lengths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def arrival_on_green(
    events_df: pd.DataFrame,
    phase: int,
    detector_ids: List[int],
    arrival_offset_sec: float = 0.0,
) -> pd.DataFrame:
    """Compute per-cycle Arrival on Green for one phase and its advance detectors.

    Algorithm
    ---------
    1. Build ``[green_ts, yellow_ts)`` windows per cycle via
       ``_build_green_windows`` (which wraps ``_build_phase_intervals``).
    2. Extract and optionally shift Code-82 actuation timestamps.
    3. Use two ``np.searchsorted`` calls for O(n log m) vectorised interval
       containment — identical in spirit to the pattern in
       ``atspm.analysis.detectors``.
    4. Group by ``cycle_start`` to aggregate per-cycle counts and durations.

    Gap Marker Rule
    ~~~~~~~~~~~~~~~
    ``_build_green_windows`` inherits full gap isolation from
    ``_build_phase_intervals``.  The ``[green_ts, yellow_ts)`` bound on
    ``searchsorted`` ensures that a shifted detector event can never be
    attributed to a green window from an adjacent, gap-separated segment.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            Timestamps may be UTC epoch floats **or** tz-aware Timestamps.
            Gap markers (``event_code == -1``) must be present.
        phase: Signal phase number.
        detector_ids: ``parameter`` values of the advance detectors
            associated with *phase*.  Typically sourced from the
            ``Det_P{phase}_Arrival`` config key.
        arrival_offset_sec: Seconds to add to each detector timestamp before
            evaluating green-window containment.  Positive values shift
            arrivals forward in time to account for travel time from an
            advance detector to the stop bar.  Default ``0.0``.

    Returns:
        DataFrame indexed by ``cycle_start`` (preserving input dtype) with
        columns::

            phase           int     – echoed phase number
            coord_plan      float   – coordination plan active at cycle start
            green_dur       float   – seconds of green in this cycle
            cycle_len       float   – cycle length in seconds (NaN for last)
            green_pct       float   – green_dur / cycle_len  (NaN when cycle_len is NaN)
            total_arrivals  int     – Code-82 actuations in the full cycle window
            aog             int     – Code-82 actuations in [green_ts, yellow_ts)
            aog_pct         float   – aog / total_arrivals  (0.0 when total == 0)

        The index name is ``"cycle_start"``.
        Returns an empty DataFrame (same column schema) when no green windows
        or no detector events are found, or when ``detector_ids`` is empty.
    """
    _empty = pd.DataFrame(columns=_CYCLE_SCHEMA)
    _empty.index.name = "cycle_start"

    if events_df.empty or not detector_ids:
        return _empty

    # --- 1. Green windows -------------------------------------------------------
    windows = _build_green_windows(events_df, phase)

    if windows.empty:
        return _empty

    # --- 2. Detector events (shifted copy) -------------------------------------
    det = _shift_detector_timestamps(events_df, detector_ids, arrival_offset_sec)

    # Convert timestamps to float for searchsorted regardless of input dtype
    green_starts_f = _ts_to_float(windows["green_ts"])
    yellow_starts_f = _ts_to_float(windows["yellow_ts"])
    cycle_starts_f = _ts_to_float(windows["cycle_start"])

    # Cycle end = next cycle's start (use yellow_ts as a safe upper bound for
    # the final cycle).  We only need this to count *total* arrivals per cycle.
    # For the last cycle we use yellow_ts + a generous buffer (3600 s) so that
    # late-arriving detections are still captured.
    cycle_ends_f = np.empty(len(cycle_starts_f), dtype=np.float64)
    cycle_ends_f[:-1] = cycle_starts_f[1:]
    cycle_ends_f[-1] = yellow_starts_f[-1] + 3600.0

    # --- 3. Vectorised interval containment (searchsorted) ---------------------
    if det.empty:
        # No actuations: build zero-count result from windows alone
        result = windows[["cycle_start", "coord_plan", "green_dur"]].copy()
        result["phase"] = phase
        result["total_arrivals"] = 0
        result["aog"] = 0
    else:
        det_ts_f = _ts_to_float(det["timestamp"])

        # For each actuation find which cycle it belongs to
        # (backward-fill: last cycle_start <= det_ts)
        cycle_idx = np.searchsorted(cycle_starts_f, det_ts_f, side="right") - 1

        # For each actuation test whether it falls in [green_ts, yellow_ts)
        # using the cycle index resolved above.
        valid_cycle = (cycle_idx >= 0) & (cycle_idx < len(windows))

        # Default both flags to False
        on_green = np.zeros(len(det_ts_f), dtype=bool)
        in_cycle = np.zeros(len(det_ts_f), dtype=bool)

        if valid_cycle.any():
            vc_idx = cycle_idx[valid_cycle]
            vc_ts = det_ts_f[valid_cycle]

            # in_cycle: det_ts < cycle_end  (already know det_ts >= cycle_start
            # by construction of cycle_idx)
            in_cycle[valid_cycle] = vc_ts < cycle_ends_f[vc_idx]

            # on_green: green_ts <= det_ts < yellow_ts
            on_green[valid_cycle] = (
                (vc_ts >= green_starts_f[vc_idx])
                & (vc_ts < yellow_starts_f[vc_idx])
            )

        # Aggregate per cycle
        det_frame = pd.DataFrame({
            "cycle_idx":  cycle_idx,
            "in_cycle":   in_cycle,
            "on_green":   on_green,
        })

        agg = (
            det_frame.loc[det_frame["in_cycle"]]
            .groupby("cycle_idx")
            .agg(
                total_arrivals=("in_cycle", "sum"),
                aog=("on_green", "sum"),
            )
        )

        # Re-index against all cycles so cycles with zero counts are present
        agg = agg.reindex(range(len(windows)), fill_value=0).reset_index(drop=True)

        result = windows[["cycle_start", "coord_plan", "green_dur"]].copy()
        result = result.reset_index(drop=True)
        result["phase"] = phase
        result["total_arrivals"] = agg["total_arrivals"].values
        result["aog"] = agg["aog"].values

    # --- 4. Derived columns -----------------------------------------------------
    cs_f = _ts_to_float(result["cycle_start"])
    cl = _cycle_lengths(cs_f)
    result["cycle_len"] = cl

    result["green_pct"] = np.where(
        np.isnan(result["cycle_len"].values),
        np.nan,
        result["green_dur"].values / result["cycle_len"].values,
    )

    total = result["total_arrivals"].values.astype(float)
    result["aog_pct"] = np.where(
        total == 0,
        0.0,
        result["aog"].values / total,
    )

    # --- 5. Final shape ---------------------------------------------------------
    result = result.set_index("cycle_start")
    result.index.name = "cycle_start"
    result = result[_CYCLE_SCHEMA]

    return result.round({"green_dur": 2, "cycle_len": 2,
                         "green_pct": 4, "aog_pct": 4})


def bin_arrival_on_green(
    cycle_df: pd.DataFrame,
    bin_len: int = 60,
) -> pd.DataFrame:
    """Aggregate per-cycle AOG records into fixed-width time bins.

    Raw counts (``aog``, ``total_arrivals``, ``green_dur``) are summed within
    each bin before computing bin-level rates.  This avoids the
    averaging-of-averages error: ``aog_pct`` is derived as
    ``sum(aog) / sum(total_arrivals)``, not ``mean(aog_pct)``.

    ``green_pct`` is computed as the mean of per-cycle ``green_dur / cycle_len``
    ratios, which correctly weights each cycle's contribution regardless of
    varying cycle lengths.

    Args:
        cycle_df: Output of ``arrival_on_green`` (indexed by ``cycle_start``
            with columns as documented there).  May contain multiple phases
            if the caller concatenates results for a phase list.
        bin_len: Bin width in **minutes**.  Default ``60``.

    Returns:
        DataFrame with columns::

            time            Timestamp  – bin start (floor of cycle_start to bin_len)
            phase           int
            coord_plan      float
            green_dur       float      – total green seconds in bin
            green_pct       float      – mean per-cycle green fraction (0.0–1.0)
            total_arrivals  int        – sum of all actuations in bin
            aog             int        – sum of on-green actuations in bin
            aog_pct         float      – aog / total_arrivals  (0.0 when total == 0)
            n_cycles        int        – number of cycles contributing to bin

        Sorted by ``["phase", "time"]``.  Bins with ``n_cycles == 0`` are
        excluded.  Returns an empty DataFrame (same column schema) when
        *cycle_df* is empty.
    """
    _empty = pd.DataFrame(columns=_BIN_SCHEMA)

    if cycle_df is None or cycle_df.empty:
        return _empty

    df = cycle_df.copy().reset_index()   # cycle_start becomes a column

    # Resolve timestamp dtype for flooring
    sample_cs = df["cycle_start"].iloc[0]
    if hasattr(sample_cs, "floor"):
        # tz-aware Timestamp
        df["_bin"] = df["cycle_start"].dt.floor(f"{bin_len}min")
    else:
        # UTC epoch float — convert to tz-naive UTC Timestamp for flooring
        df["_bin"] = (
            pd.to_datetime(df["cycle_start"], unit="s", utc=True)
            .dt.floor(f"{bin_len}min")
        )

    # Per-cycle green fraction for averaging (NaN rows are excluded by skipna)
    df["_green_frac"] = df["green_dur"] / df["cycle_len"].replace(0, np.nan)

    group_keys = ["_bin", "phase", "coord_plan"]

    agg = df.groupby(group_keys, sort=False).agg(
        green_dur=("green_dur", "sum"),
        green_pct=("_green_frac", "mean"),       # mean of per-cycle ratios
        total_arrivals=("total_arrivals", "sum"),
        aog=("aog", "sum"),
        n_cycles=("aog", "count"),
    ).reset_index()

    # Recompute aog_pct from summed totals (not from mean of cycle-level rates)
    total = agg["total_arrivals"].values.astype(float)
    agg["aog_pct"] = np.where(total == 0, 0.0, agg["aog"].values / total)

    agg = agg.rename(columns={"_bin": "time"})
    agg = agg.loc[agg["n_cycles"] > 0].copy()
    agg = agg.sort_values(["phase", "time"]).reset_index(drop=True)

    # Cast integer columns
    for col in ("total_arrivals", "aog", "n_cycles"):
        agg[col] = agg[col].astype(int)

    return agg[_BIN_SCHEMA].round(
        {"green_dur": 2, "green_pct": 4, "aog_pct": 4}
    )
