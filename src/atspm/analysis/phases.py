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
    Rows with Code == -1 mark data discontinuities.  Phase state transitions
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
        events_df: Legacy-format DataFrame with columns
            ``[TS_start, Code, ID, Cycle_start, Coord_plan]``.
            ``TS_start`` must be tz-aware ``datetime64`` (as produced by
            ``get_legacy_dataframe`` with a timezone argument).  Gap markers
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
            ``Split = Green``).  This covers dummy phases that drive
            overlaps and carry no independent yellow timing.  When
            ``False`` (default), such phases are silently omitted.

    Returns:
        DataFrame indexed by ``Time`` (bin-start ``Timestamp`` or
        ``Cycle_start`` value, matching the dtype of the input index).
        Columns::

            Ph1 G, Ph1 YR, Ph1 Split, Ph2 G, ..., Cycle Length

        ``Cycle Length`` in cycle mode is the elapsed seconds from this
        cycle's start to the next (``NaN`` for the last cycle).  In bin
        mode it is the mean cycle length across cycles whose start falls
        in the bin.

        Values are ``float``, rounded to two decimal places.
        Returns an empty ``DataFrame`` when no usable phase events are found.
    """
    if events_df.empty:
        return pd.DataFrame()

    # Pull phase-relevant codes + gap markers for segment assignment
    mask = events_df["Code"].isin(_PHASE_CODES | {_GAP_CODE})
    ph_df = events_df.loc[mask].copy().sort_values("TS_start").reset_index(drop=True)

    if ph_df.empty:
        return pd.DataFrame()

    ph_df["_seg"] = _segment_id(ph_df)
    # Drop gap rows after segment assignment — they served their purpose
    ph_df = ph_df.loc[ph_df["Code"] != _GAP_CODE].copy()

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
    """
    Return a monotonically increasing integer segment ID per row.

    Increments at every gap-marker row (Code == -1).  All rows within the
    same uninterrupted data block share the same ID.

    Args:
        df: DataFrame sorted by ``TS_start``, containing a ``Code`` column.

    Returns:
        ``int32`` Series aligned with *df*'s index.
    """
    return (df["Code"] == _GAP_CODE).cumsum().astype(np.int32)


def _build_phase_intervals(
    ph_df: pd.DataFrame,
    include_no_clearance: bool = False,
) -> pd.DataFrame:
    """
    Convert sequential phase state-change events into tidy intervals.

    State machine per (segment, phase) — five states:

        IDLE → (Code 1) → GREEN → (Code 8) → YELLOW → (Code 9) → POST_YELLOW
            POST_YELLOW → (Code 10) → RED_CLR → (Code 11) → emit, IDLE
            POST_YELLOW → (Code 12) → emit using Code-9 ts as clear_end, IDLE
            YELLOW      → (Code 12) → emit using Code-8 ts as clear_end, IDLE
            RED_CLR     → (Code 11) → emit using Code-11 ts, IDLE
            GREEN       → (Code 12) → green-only if include_no_clearance, IDLE

    Code 12 (Phase Inactive) acts as a hard terminator in all post-green
    states, covering firmware variants that omit Code 9 before Code 12, and
    dummy/overlap-driven phases that carry no independent yellow timing.

    Args:
        ph_df: Phase-code events (Codes 1, 8, 9, 10, 11, 12) with columns
            ``[TS_start, Code, ID, Cycle_start, _seg]``.
            Gap markers already removed.  Sorted by ``TS_start``.
        include_no_clearance: When True, emit green-only intervals for phases
            that go Code 1 → Code 12 with no yellow logged (dummy phases).

    Returns:
        DataFrame with columns::

            phase, seg, green_ts, yellow_ts, clear_end_ts,
            Cycle_start, green_dur, clear_dur, split_dur

        Durations are in seconds (float).  Rows with non-positive green
        duration are dropped.
    """
    _IDLE     = 0
    _GREEN    = 1
    _YELLOW   = 2
    _POST_YEL = 3   # Code 9 seen; waiting to confirm whether Code 10 follows
    _RED_CLR  = 4   # Code 10 seen; waiting for Code 11

    records: List[Dict[str, Any]] = []

    # Priority order within the same timestamp: higher-priority codes must be
    # processed before lower-priority ones so that simultaneous events (which
    # are common — e.g. Code 9+10 at the same decisecond, Code 11+12 at the
    # same decisecond) advance the state machine in the correct sequence.
    # Lower sort_key = processed first.
    _SORT_PRIORITY = {
        _CODE_GREEN:      0,
        _CODE_YELLOW:     1,
        _CODE_END_YELLOW: 2,
        _CODE_BEGIN_RC:   3,
        _CODE_END_RC:     4,   # must precede Code 12 at the same timestamp
        _CODE_INACTIVE:   5,
    }

    for (seg, phase), grp in ph_df.groupby(["_seg", "ID"], sort=True):
        grp = grp.copy()
        grp["_sort_key"] = grp["Code"].map(_SORT_PRIORITY).fillna(99)
        grp = grp.sort_values(["TS_start", "_sort_key"])

        state       = _IDLE
        green_ts    = None
        yellow_ts   = None   # timestamp of Code 8 (begin yellow)
        clear_end   = None   # best known clearance endpoint so far
        cycle_start = None

        def _emit(clr_end_ts):
            records.append(dict(
                phase=int(phase),
                seg=int(seg),
                green_ts=green_ts,
                yellow_ts=yellow_ts,
                clear_end_ts=clr_end_ts,
                Cycle_start=cycle_start,
            ))

        for row in grp.itertuples(index=False):
            code = row.Code
            ts   = row.TS_start
            cs   = row.Cycle_start

            if code == _CODE_GREEN:
                # Begin a fresh interval; discard any open partial one
                green_ts    = ts
                yellow_ts   = None
                clear_end   = None
                cycle_start = cs
                state       = _GREEN

            elif code == _CODE_YELLOW:
                if state == _GREEN:
                    yellow_ts = ts
                    clear_end = ts   # provisional end if no further codes arrive
                    state     = _YELLOW

            elif code == _CODE_END_YELLOW:
                if state == _YELLOW:
                    clear_end = ts   # yellow-only: end of yellow IS the clear end
                    state     = _POST_YEL

            elif code == _CODE_BEGIN_RC:
                if state in (_YELLOW, _POST_YEL):
                    # RC is being served; clear_end updated by Code 11.
                    # Store ts now so Code-12 fallback has RC-start at minimum.
                    clear_end = ts
                    state = _RED_CLR

            elif code == _CODE_END_RC:
                if state in (_RED_CLR, _YELLOW, _POST_YEL):
                    # Code 11: definitive end of clearance
                    _emit(ts)
                    state    = _IDLE
                    green_ts = None

            elif code == _CODE_INACTIVE:
                if state in (_YELLOW, _POST_YEL, _RED_CLR):
                    # Hard terminator: use best known clear_end
                    _emit(clear_end)
                elif state == _GREEN and include_no_clearance:
                    # Dummy/overlap phase: no clearance, green-only
                    yellow_ts = ts   # treat inactive time as notional split end
                    _emit(ts)        # clear_dur will be 0
                # state == IDLE or GREEN without flag: silently discard
                state    = _IDLE
                green_ts = None

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["green_dur"] = _ts_diff_seconds(df["yellow_ts"],    df["green_ts"])
    df["clear_dur"] = _ts_diff_seconds(df["clear_end_ts"], df["yellow_ts"])
    df["split_dur"] = df["green_dur"] + df["clear_dur"]

    # Drop rows with non-positive green (timestamp collision / bad data)
    valid = df["green_dur"] > 0.0
    return df.loc[valid].reset_index(drop=True)


def _ts_diff_seconds(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """
    Compute element-wise (later − earlier) in seconds.

    Handles both tz-aware ``Timestamp`` series (from ``get_legacy_dataframe``
    with a timezone) and raw UTC-epoch float series.

    Args:
        later:   Series of timestamps (end of interval).
        earlier: Series of timestamps (start of interval).

    Returns:
        Float Series of elapsed seconds.
    """
    sample = later.iloc[0]
    if isinstance(sample, (int, float, np.floating, np.integer)):
        return (later - earlier).astype(float)
    # Timestamp / datetime64 path
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

    Each interval is attributed to the ``Cycle_start`` recorded when its
    green onset occurred (captured by the reader's merge_asof join).  If a
    phase appears more than once within a cycle (dual-ring actuations), its
    durations are summed.

    ``Cycle Length`` is derived from the diff of successive ``Cycle_start``
    values.  The final cycle receives ``NaN``.

    Args:
        intervals: Per-phase intervals with ``Cycle_start`` column.
        all_phases: Ordered list of phase IDs to emit as columns.

    Returns:
        Wide DataFrame indexed by ``Cycle_start`` with name ``"Time"``.
    """
    agg = (
        intervals.groupby(["Cycle_start", "phase"])[
            ["green_dur", "clear_dur", "split_dur"]
        ]
        .sum()
        .reset_index()
    )

    # Pivot phases into columns
    wide_g   = agg.pivot(index="Cycle_start", columns="phase", values="green_dur")
    wide_yr  = agg.pivot(index="Cycle_start", columns="phase", values="clear_dur")
    wide_spl = agg.pivot(index="Cycle_start", columns="phase", values="split_dur")

    # Build final column set in display order
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

    # Cycle Length: elapsed seconds between successive cycle starts
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
    """
    Aggregate intervals into fixed-width time bins, producing a wide table.

    Each interval is bucketed by its ``green_ts`` (the moment the phase
    turned green), so the count of cycles contributing to each bin is the
    number of green onsets that fell within it.  The ``Cycle_start`` column
    is used to calculate per-bin mean cycle length independently.

    Args:
        intervals:   Per-phase intervals DataFrame.
        bin_len:     Bin width in minutes.
        report_mode: ``"seconds"`` | ``"total"`` | ``"proportion"``.
        all_phases:  Ordered list of phase IDs.

    Returns:
        Wide DataFrame indexed by bin-start ``Timestamp`` with name ``"Time"``.
    """

    # Assign each interval to a time bin via its green_ts
    sample_ts = intervals["green_ts"].iloc[0]
    if isinstance(sample_ts, (int, float, np.floating, np.integer)):
        # UTC-epoch floats: convert to tz-naive UTC Timestamps for resampling
        intervals = intervals.copy()
        intervals["_bin"] = pd.to_datetime(
            intervals["green_ts"], unit="s", utc=True
        ).dt.floor(f"{bin_len}min")
    else:
        intervals = intervals.copy()
        intervals["_bin"] = intervals["green_ts"].dt.floor(f"{bin_len}min")

    # Per-phase, per-bin aggregation
    grp = intervals.groupby(["_bin", "phase"])

    if report_mode == "total":
        agg_g   = grp["green_dur"].sum()
        agg_yr  = grp["clear_dur"].sum()
        agg_spl = grp["split_dur"].sum()
    elif report_mode == "proportion":
        # Proportion = mean phase seconds / mean cycle length.
        # Dividing by bin duration would be wrong — the bin contains red time
        # for all phases and values would sum nowhere near 1.0.
        # Use mean cycle length (same series as the Cycle Length output column)
        # so that splits for complementary phases sum correctly to ~1.0.
        mean_cycle = _mean_cycle_length_by_bin(intervals, bin_len)
        # mean_g/yr/spl are MultiIndex (bin, phase); unstack to (bin × phase),
        # then divide each column by mean_cycle (bin-indexed) via .div(axis=0).
        mean_g_wide   = grp["green_dur"].mean().unstack("phase")
        mean_yr_wide  = grp["clear_dur"].mean().unstack("phase")
        mean_spl_wide = grp["split_dur"].mean().unstack("phase")
        agg_g   = mean_g_wide.div(mean_cycle, axis=0).stack("phase")
        agg_yr  = mean_yr_wide.div(mean_cycle, axis=0).stack("phase")
        agg_spl = mean_spl_wide.div(mean_cycle, axis=0).stack("phase")
    else:  # "seconds" (default)
        agg_g   = grp["green_dur"].mean()
        agg_yr  = grp["clear_dur"].mean()
        agg_spl = grp["split_dur"].mean()

    wide_g   = agg_g.unstack("phase")
    wide_yr  = agg_yr.unstack("phase")
    wide_spl = agg_spl.unstack("phase")

    # Build final column set in display order
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

    # Mean cycle length per bin (keyed by Cycle_start, not green_ts, to avoid
    # double-counting when multiple phases share a cycle)
    result["Cycle Length"] = _mean_cycle_length_by_bin(intervals, bin_len)

    return result.round(4 if report_mode == "proportion" else 2)


# ---------------------------------------------------------------------------
# Cycle length helpers
# ---------------------------------------------------------------------------


def _cycle_length_series(cycle_index: pd.Index) -> pd.Series:
    """
    Compute per-cycle elapsed time (seconds) from successive index values.

    Works for both tz-aware Timestamp and float (UTC epoch) indices.

    Args:
        cycle_index: Sorted index of cycle-start values.

    Returns:
        Float Series aligned with *cycle_index*; last entry is ``NaN``.
    """
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
    """
    Compute the mean cycle length for cycles starting within each bin.

    Uses the unique ``Cycle_start`` values present in *intervals* (after
    binning by ``green_ts``) rather than all interval rows, to avoid
    inflating the mean when multiple phases actuate in the same cycle.

    Args:
        intervals: Intervals DataFrame with ``_bin`` and ``Cycle_start`` columns.
        bin_len:   Bin width in minutes (used to assign ``Cycle_start`` to bins).

    Returns:
        Series indexed by bin-start Timestamp with name ``"Cycle Length"``.
    """
    sample_cs = intervals["Cycle_start"].iloc[0]
    if isinstance(sample_cs, (int, float, np.floating, np.integer)):
        cs_bins = pd.to_datetime(
            intervals["Cycle_start"], unit="s", utc=True
        ).dt.floor(f"{bin_len}min")
    else:
        cs_bins = intervals["Cycle_start"].dt.floor(f"{bin_len}min")

    # Unique cycle starts per bin to avoid double-counting phases
    unique_cs = (
        intervals.assign(_cs_bin=cs_bins)[["_cs_bin", "Cycle_start"]]
        .drop_duplicates()
        .copy()
    )
    unique_cs = unique_cs.sort_values("Cycle_start")

    # Diff of sorted cycle_start within each bin = individual cycle lengths
    sample_cs2 = unique_cs["Cycle_start"].iloc[0]
    if isinstance(sample_cs2, (int, float, np.floating, np.integer)):
        unique_cs["_cs_float"] = unique_cs["Cycle_start"].astype(float)
        unique_cs["_len"] = unique_cs.groupby("_cs_bin")["_cs_float"].diff().shift(-1)
    else:
        unique_cs["_len"] = (
            unique_cs.groupby("_cs_bin")["Cycle_start"]
            .diff()
            .shift(-1)
            .dt.total_seconds()
        )

    mean_len = unique_cs.groupby("_cs_bin")["_len"].mean()
    mean_len.index.name = "Time"
    mean_len.name = "Cycle Length"
    return mean_len