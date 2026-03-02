"""
ATSPM Phase Split Analysis (Functional Core)

Pure functions only. No I/O, no SQL, no side effects.
Input/output is DataFrames and plain dicts.

Calculates per-cycle or binned signal phase timing components:
    - Green duration      (Code 1 onset → Code 8 onset)
    - Clearance duration  (Code 8 onset → Code 11 onset if red clearance served,
                           otherwise Code 8 onset → Code 9 onset; Yellow + Red Clearance)
    - Total split         (Green + Clearance)
    - Cycle length        (cycle_start to next cycle_start, seconds)

Event code reference (Purdue / Indiana Hi-Res Logger Enumerations):
    1  – Phase Begin Green
    7  – Phase Green Termination  (not used here)
    8  – Phase Begin Yellow Clearance
    9  – Phase End Yellow Clearance
    10 – Phase Begin Red Clearance  (only present when RC is served)
    11 – Phase End Red Clearance    (only present when RC is served)
    12 – Phase Inactive

Clearance logic:
    Yellow-only phases  (no red clearance programmed):
        Code 8 → Code 9     clear_end = Code 9 timestamp
    Phases with red clearance:
        Code 8 → Code 10 → Code 11     clear_end = Code 11 timestamp
    Code 12 (Phase Inactive) is treated as a hard terminator if it arrives
    before the expected clearance endpoint — covers dummy/overlap-driven phases
    that have no yellow or a very short yellow not separately logged.

Dummy-phase handling:
    Some intersections use phases that drive overlaps and carry no independent
    yellow timing (the yellow is logged on the overlap or a following phase).
    Such phases produce Code 1 → Code 12 with nothing in between.  By default
    these are silently dropped (no meaningful split to report).  Pass
    ``include_no_clearance=True`` to ``phase_splits()`` to instead emit a
    green-only interval with ``YR = 0`` and ``Split = Green``.

Output is a wide table with columns:
    Ph{N} G, Ph{N} YR, Ph{N} Split, ..., Cycle Length

Bin-mode reporting modes (``report_mode`` argument):
    ``"seconds"``     – mean seconds per phase component per cycle in the bin
    ``"total"``       – total seconds accumulated across all cycles in the bin
    ``"proportion"``  – mean fraction of the bin duration (0.0–1.0)

Package Location: src/atspm/analysis/phases.py

Gap Marker Rule:
    Rows with event_code == -1 mark data discontinuities.  Phase state transitions
    depend on sequential continuity within a phase; _segment_id() increments
    a counter at every gap marker row.  All groupby operations that build
    intervals include the segment as an additional key.  An interval whose
    green onset and clearance endpoint span a segment boundary is silently
    dropped — its endpoints are definitionally unknown.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1

_CODE_GREEN       = 1    # Phase Begin Green
_CODE_YELLOW      = 8    # Phase Begin Yellow Clearance
_CODE_END_YELLOW  = 9    # Phase End Yellow Clearance
_CODE_BEGIN_RC    = 10   # Phase Begin Red Clearance (only if RC served)
_CODE_END_RC      = 11   # Phase End Red Clearance   (only if RC served)
_CODE_INACTIVE    = 12   # Phase Inactive (hard terminator)

_PHASE_CODES = frozenset({
    _CODE_GREEN, _CODE_YELLOW, _CODE_END_YELLOW,
    _CODE_BEGIN_RC, _CODE_END_RC, _CODE_INACTIVE,
})

ReportMode = Literal["seconds", "total", "proportion"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def phase_splits(
    events_df: pd.DataFrame,
    bin_len: Union[int, str] = "cycle",
    report_mode: ReportMode = "seconds",
    phases: Optional[List[int]] = None,
    include_no_clearance: bool = False,
) -> pd.DataFrame:
    """
    Compute per-cycle or binned phase green, clearance, and split timing.

    Args:
        events_df: flat DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            ``timestamp`` must be tz-aware ``datetime64``. Gap markers
            (Code -1) must be present for discontinuity enforcement.
        bin_len: ``"cycle"`` for one row per cycle, or a positive integer
            number of minutes for time-bin aggregation.  Default ``"cycle"``.
        report_mode: Controls how binned values are expressed.  Ignored when
            ``bin_len="cycle"``.

            ``"seconds"``    – mean seconds per phase component across cycles
                               whose green onset falls within the bin.
            ``"total"``      – total seconds summed across those cycles.
            ``"proportion"`` – mean seconds divided by the bin duration,
                               yielding a 0.0–1.0 utilisation fraction.

        phases: Explicit ordered list of phase IDs to include as columns.
            When ``None``, all phases observed in *events_df* are used,
            sorted ascending.
        include_no_clearance: When ``True``, phases that go directly from
            Code 1 (green) to Code 12 (inactive) with no yellow logged are
            included as green-only intervals (``YR = 0``,
            ``Split = Green``).

    Returns:
        DataFrame indexed by ``Time`` (bin-start ``Timestamp`` or
        ``cycle_start`` value, matching the dtype of the input index).
    """
    if events_df.empty:
        return pd.DataFrame()

    mask = events_df["event_code"].isin(_PHASE_CODES | {_GAP_CODE})
    ph_df = events_df.loc[mask].copy().sort_values("timestamp").reset_index(drop=True)

    if ph_df.empty:
        return pd.DataFrame()

    ph_df["_seg"] = _segment_id(ph_df)
    ph_df = ph_df.loc[ph_df["event_code"] != _GAP_CODE].copy()

    if ph_df.empty:
        return pd.DataFrame()

    intervals = _build_phase_intervals(ph_df, include_no_clearance)

    if intervals.empty:
        return pd.DataFrame()

    if phases is not None:
        intervals = intervals.loc[intervals["phase"].isin(phases)].copy()
        all_phases = phases
    else:
        all_phases = sorted(intervals["phase"].unique().tolist())

    if intervals.empty or not all_phases:
        return pd.DataFrame()

    if bin_len == "cycle":
        return _pivot_by_cycle(intervals, all_phases)
    else:
        return _pivot_by_bin(intervals, int(bin_len), report_mode, all_phases)


# ---------------------------------------------------------------------------
# Interval construction
# ---------------------------------------------------------------------------


def _segment_id(df: pd.DataFrame) -> pd.Series:
    return (df["event_code"] == _GAP_CODE).cumsum().astype(np.int32)


def _build_phase_intervals(
    ph_df: pd.DataFrame,
    include_no_clearance: bool = False,
) -> pd.DataFrame:
    """
    Convert sequential phase state-change events into tidy intervals.

    Args:
        ph_df: Phase-code events (Codes 1, 8, 9, 10, 11, 12) with columns
            ``[timestamp, event_code, parameter, cycle_start, _seg]``.
            Gap markers already removed.  Sorted by ``timestamp``.
    """
    _IDLE     = 0
    _GREEN    = 1
    _YELLOW   = 2
    _POST_YEL = 3   
    _RED_CLR  = 4   

    records: List[Dict[str, Any]] = []

    _SORT_PRIORITY = {
        _CODE_GREEN:      0,
        _CODE_YELLOW:     1,
        _CODE_END_YELLOW: 2,
        _CODE_BEGIN_RC:   3,
        _CODE_END_RC:     4,   
        _CODE_INACTIVE:   5,
    }

    for (seg, phase), grp in ph_df.groupby(["_seg", "parameter"], sort=True):
        grp = grp.copy()
        grp["_sort_key"] = grp["event_code"].map(_SORT_PRIORITY).fillna(99)
        grp = grp.sort_values(["timestamp", "_sort_key"])

        state       = _IDLE
        green_ts    = None
        yellow_ts   = None   
        clear_end   = None   
        cycle_start = None

        def _emit(clr_end_ts):
            records.append(dict(
                phase=int(phase),
                seg=int(seg),
                green_ts=green_ts,
                yellow_ts=yellow_ts,
                clear_end_ts=clr_end_ts,
                cycle_start=cycle_start,
            ))

        for row in grp.itertuples(index=False):
            code = row.event_code
            ts   = row.timestamp
            cs   = row.cycle_start

            if code == _CODE_GREEN:
                green_ts    = ts
                yellow_ts   = None
                clear_end   = None
                cycle_start = cs
                state       = _GREEN

            elif code == _CODE_YELLOW:
                if state == _GREEN:
                    yellow_ts = ts
                    clear_end = ts   
                    state     = _YELLOW

            elif code == _CODE_END_YELLOW:
                if state == _YELLOW:
                    clear_end = ts   
                    state     = _POST_YEL

            elif code == _CODE_BEGIN_RC:
                if state in (_YELLOW, _POST_YEL):
                    clear_end = ts
                    state = _RED_CLR

            elif code == _CODE_END_RC:
                if state in (_RED_CLR, _YELLOW, _POST_YEL):
                    _emit(ts)
                    state    = _IDLE
                    green_ts = None

            elif code == _CODE_INACTIVE:
                if state in (_YELLOW, _POST_YEL, _RED_CLR):
                    _emit(clear_end)
                elif state == _GREEN and include_no_clearance:
                    yellow_ts = ts   
                    _emit(ts)        
                state    = _IDLE
                green_ts = None

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["green_dur"] = _ts_diff_seconds(df["yellow_ts"],    df["green_ts"])
    df["clear_dur"] = _ts_diff_seconds(df["clear_end_ts"], df["yellow_ts"])
    df["split_dur"] = df["green_dur"] + df["clear_dur"]

    valid = df["green_dur"] > 0.0
    return df.loc[valid].reset_index(drop=True)


def _ts_diff_seconds(later: pd.Series, earlier: pd.Series) -> pd.Series:
    sample = later.iloc[0]
    if isinstance(sample, (int, float, np.floating, np.integer)):
        return (later - earlier).astype(float)
    return (later - earlier).dt.total_seconds()


# ---------------------------------------------------------------------------
# Per-cycle pivot
# ---------------------------------------------------------------------------


def _pivot_by_cycle(
    intervals: pd.DataFrame,
    all_phases: List[int],
) -> pd.DataFrame:
    """
    Build a wide table with one row per cycle.
    """
    agg = (
        intervals.groupby(["cycle_start", "phase"])[
            ["green_dur", "clear_dur", "split_dur"]
        ]
        .sum()
        .reset_index()
    )

    wide_g   = agg.pivot(index="cycle_start", columns="phase", values="green_dur")
    wide_yr  = agg.pivot(index="cycle_start", columns="phase", values="clear_dur")
    wide_spl = agg.pivot(index="cycle_start", columns="phase", values="split_dur")

    col_frames: List[pd.DataFrame] = []
    for ph in all_phases:
        col_frames.append(
            pd.DataFrame(
                {
                    f"Ph{ph} G":     wide_g.get(ph),
                    f"Ph{ph} YR":    wide_yr.get(ph),
                    f"Ph{ph} Split": wide_spl.get(ph),
                },
                index=wide_g.index.union(wide_yr.index).union(wide_spl.index),
            )
        )

    result = pd.concat(col_frames, axis=1) if col_frames else pd.DataFrame()
    result.index.name = "Time"
    result.sort_index(inplace=True)

    result["Cycle Length"] = _cycle_length_series(result.index)

    return result.round(2)


# ---------------------------------------------------------------------------
# Binned pivot
# ---------------------------------------------------------------------------


def _pivot_by_bin(
    intervals: pd.DataFrame,
    bin_len: int,
    report_mode: ReportMode,
    all_phases: List[int],
) -> pd.DataFrame:
    sample_ts = intervals["green_ts"].iloc[0]
    if isinstance(sample_ts, (int, float, np.floating, np.integer)):
        intervals = intervals.copy()
        intervals["_bin"] = pd.to_datetime(
            intervals["green_ts"], unit="s", utc=True
        ).dt.floor(f"{bin_len}min")
    else:
        intervals = intervals.copy()
        intervals["_bin"] = intervals["green_ts"].dt.floor(f"{bin_len}min")

    grp = intervals.groupby(["_bin", "phase"])

    if report_mode == "total":
        agg_g   = grp["green_dur"].sum()
        agg_yr  = grp["clear_dur"].sum()
        agg_spl = grp["split_dur"].sum()
    elif report_mode == "proportion":
        mean_cycle = _mean_cycle_length_by_bin(intervals, bin_len)
        mean_g_wide   = grp["green_dur"].mean().unstack("phase")
        mean_yr_wide  = grp["clear_dur"].mean().unstack("phase")
        mean_spl_wide = grp["split_dur"].mean().unstack("phase")
        agg_g   = mean_g_wide.div(mean_cycle, axis=0).stack("phase")
        agg_yr  = mean_yr_wide.div(mean_cycle, axis=0).stack("phase")
        agg_spl = mean_spl_wide.div(mean_cycle, axis=0).stack("phase")
    else:  
        agg_g   = grp["green_dur"].mean()
        agg_yr  = grp["clear_dur"].mean()
        agg_spl = grp["split_dur"].mean()

    wide_g   = agg_g.unstack("phase")
    wide_yr  = agg_yr.unstack("phase")
    wide_spl = agg_spl.unstack("phase")

    col_frames: List[pd.DataFrame] = []
    for ph in all_phases:
        col_frames.append(
            pd.DataFrame(
                {
                    f"Ph{ph} G":     wide_g.get(ph),
                    f"Ph{ph} YR":    wide_yr.get(ph),
                    f"Ph{ph} Split": wide_spl.get(ph),
                },
                index=wide_g.index.union(wide_yr.index).union(wide_spl.index),
            )
        )

    result = pd.concat(col_frames, axis=1) if col_frames else pd.DataFrame()
    result.index.name = "Time"
    result.sort_index(inplace=True)

    result["Cycle Length"] = _mean_cycle_length_by_bin(intervals, bin_len)

    return result.round(4 if report_mode == "proportion" else 2)


# ---------------------------------------------------------------------------
# Cycle length helpers
# ---------------------------------------------------------------------------


def _cycle_length_series(cycle_index: pd.Index) -> pd.Series:
    idx = cycle_index
    if len(idx) < 2:
        return pd.Series(np.nan, index=idx)

    sample = idx[0]
    if isinstance(sample, (int, float, np.floating, np.integer)):
        values = np.array(idx, dtype=float)
        diffs = np.diff(values, append=np.nan)
    else:
        diffs_td = pd.Series(idx).diff().shift(-1)
        diffs = diffs_td.dt.total_seconds().values

    return pd.Series(diffs, index=idx)


def _mean_cycle_length_by_bin(
    intervals: pd.DataFrame,
    bin_len: int,
) -> pd.Series:
    sample_cs = intervals["cycle_start"].iloc[0]
    if isinstance(sample_cs, (int, float, np.floating, np.integer)):
        cs_bins = pd.to_datetime(
            intervals["cycle_start"], unit="s", utc=True
        ).dt.floor(f"{bin_len}min")
    else:
        cs_bins = intervals["cycle_start"].dt.floor(f"{bin_len}min")

    unique_cs = (
        intervals.assign(_cs_bin=cs_bins)[["_cs_bin", "cycle_start"]]
        .drop_duplicates()
        .copy()
    )
    unique_cs = unique_cs.sort_values("cycle_start")

    sample_cs2 = unique_cs["cycle_start"].iloc[0]
    if isinstance(sample_cs2, (int, float, np.floating, np.integer)):
        unique_cs["_cs_float"] = unique_cs["cycle_start"].astype(float)
        unique_cs["_len"] = unique_cs.groupby("_cs_bin")["_cs_float"].diff().shift(-1)
    else:
        unique_cs["_len"] = (
            unique_cs.groupby("_cs_bin")["cycle_start"]
            .diff()
            .shift(-1)
            .dt.total_seconds()
        )

    mean_len = unique_cs.groupby("_cs_bin")["_len"].mean()
    mean_len.index.name = "Time"
    mean_len.name = "Cycle Length"
    return mean_len