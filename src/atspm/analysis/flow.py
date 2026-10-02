"""
Split Flow Rate Analysis (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.
Input / output is DataFrames and plain Python scalars.

Algorithm overview
------------------
For each phase split window (Code 1 green onset through Code 11 end of red
clearance), stop-bar detector departures (Code 81) are counted per detector.
Each departure yields:

* ``t``        – seconds elapsed since green onset,
* ``n``        – cumulative vehicles discharged so far,
* ``headway``  – seconds to the next departure on the same detector.

A split/detector combination is kept only when the slack between the last
departure and the end of the split window (``lost``) is at most ``max_lost``
seconds — i.e. the phase was still discharging near the end of its split and
therefore operating at or near capacity.

``rate_profiles`` then selects comparable cycles (split within a tolerance
of the modal split, then the busiest ``pct`` percent by total volume) and
computes the *effective cumulative flow rate*

    ``rate(n) = 3600 * n / (t + overhead)``

which answers: *"if the split had been terminated right after vehicle n
discharged, what flow rate would this approach have achieved?"*  The
``overhead`` term charges every hypothetical split length the fixed cost of
terminating a split, penalising short splits (where the overhead is a larger
proportion) while long splits are penalised naturally by declining discharge
rates late in green.  The point where the curve peaks identifies the
throughput-optimal split length.  Start-up lost time needs no special
treatment — it is already embedded in ``t`` (measured from green onset).

The ``normalize`` parameter selects how the overhead is estimated; see
:func:`rate_profiles`.

``discharge_profiles`` uses the same cycle selection but returns the raw
approach cumulative curve ``N(t)``, unnormalised, for the throughput
optimizer.  ``saturation_state`` classifies phases as saturated from the
share of max-out / force-off cycles in an unfiltered ``flow_rate``
result.  It is advisory: the optimizer takes saturated phases as declared.

Gap Marker Rule
---------------
Split windows are built via ``_build_phase_intervals`` (from
``atspm.analysis.phases``), which guarantees every window lies entirely
within a single contiguous data segment — no window contains a gap marker,
so per-window headways and elapsed times never span a hard reset
(``event_code == -1``).

Event code reference
--------------------
    1   – Phase Begin Green
    8   – Phase Begin Yellow Clearance
    9   – Phase End Yellow Clearance
    10  – Phase Begin Red Clearance
    4   – Phase Gap Out    ┐ termination, logged with
    5   – Phase Max Out    │ the yellow onset (Code 8)
    6   – Phase Force Off  ┘
    11  – Phase End Red Clearance   (exclusive end of split window)
    12  – Phase Inactive
    81  – Detector Off  (vehicle departing the stop-bar detector)
    -1  – Data gap marker (hard reset)

Package Location: src/atspm/analysis/flow.py
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .phases import _build_phase_intervals, _segment_id
from .aog import _ts_to_float

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1
_CODE_GREEN = 1
_CODE_DET_OFF = 81

# Phase termination codes, logged at the same timestamp as yellow onset.
# When a window holds more than one, the highest code wins (force-off over
# max-out over gap-out).
_TERMINATION_NAMES = {4: "gap_out", 5: "max_out", 6: "force_off"}
_SATURATED_TERMINATIONS = frozenset({"max_out", "force_off"})

# Phase codes needed to drive _build_phase_intervals
_PHASE_CODES = frozenset({1, 8, 9, 10, 11, 12})

# Schema for the per-cycle summary (one row per split window x detector)
_CYCLE_SCHEMA = [
    "det",
    "phase",
    "green_ts",
    "cycle_start",
    "coord_plan",
    "split",
    "green_dur",
    "clear_dur",
    "lost",
    "q",
    "termination",
]

# Schema for the per-vehicle detail (one row per detector departure)
_VEHICLE_SCHEMA = ["det", "green_ts", "t", "n", "headway"]

# Schema for the selected-cycle summary returned by rate_profiles
_SELECTED_SCHEMA = _CYCLE_SCHEMA + ["t_max", "max_rate", "n_at_max"]

# Valid overhead-normalisation modes (see rate_profiles docstring)
NORMALIZE_MODES = ("end_shift", "pooled", "clearance", "fixed", "none")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_split_windows(
    events_df: pd.DataFrame,
    phase: int,
) -> pd.DataFrame:
    """Extract per-cycle split windows for a single phase.

    Mirrors ``atspm.analysis.aog._build_green_windows`` but retains the
    clearance columns (``clear_end_ts``, ``clear_dur``, ``split_dur``)
    because the flow-rate split window runs from green onset through the
    end of red clearance, not just to yellow onset.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            ``timestamp`` values may be UTC epoch floats **or** tz-aware
            ``Timestamps``.
        phase: Signal phase number (``parameter`` value to filter on).

    Returns:
        DataFrame sorted by ``green_ts`` with columns::

            phase         int
            cycle_start   float | Timestamp  (matches input dtype)
            coord_plan    float
            green_ts      float | Timestamp  (Code 1 onset)
            clear_end_ts  float | Timestamp  (Code 11 — exclusive bound)
            green_dur     float              (seconds)
            clear_dur     float              (seconds)
            split_dur     float              (seconds, green + clearance)

        One row per valid, gap-isolated split window.  Returns an empty
        DataFrame with the correct schema when no valid windows exist.
    """
    _EMPTY = pd.DataFrame(
        columns=["phase", "cycle_start", "coord_plan", "green_ts",
                 "clear_end_ts", "green_dur", "clear_dur", "split_dur"]
    )

    if events_df.empty:
        return _EMPTY

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

    ph_df["_seg"] = _segment_id(ph_df)
    ph_df = ph_df.loc[ph_df["event_code"] != _GAP_CODE].copy()

    if ph_df.empty:
        return _EMPTY

    intervals = _build_phase_intervals(ph_df, include_no_clearance=False)

    if intervals.empty:
        return _EMPTY

    intervals = intervals.loc[intervals["phase"] == phase].copy()

    if intervals.empty:
        return _EMPTY

    # Attach coord_plan from the Code-1 event that opened each green,
    # matched by cycle_start (same approach as _build_green_windows).
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

    return (
        intervals[
            ["phase", "cycle_start", "coord_plan", "green_ts",
             "clear_end_ts", "green_dur", "clear_dur", "split_dur"]
        ]
        .sort_values("green_ts")
        .reset_index(drop=True)
    )


def _window_terminations(
    events_df: pd.DataFrame,
    phase: int,
    green_f: np.ndarray,
    clear_f: np.ndarray,
) -> np.ndarray:
    """Name how each split window's green ended.

    Codes 4/5/6 for *phase* are assigned to windows with the same
    ``searchsorted`` containment as departures.  They share the yellow
    onset's timestamp, so they fall inside ``[green_ts, clear_end_ts)``.

    Args:
        events_df: Flat events DataFrame (``timestamp``, ``event_code``,
            ``parameter``).
        phase: Signal phase number.
        green_f: Window starts as epoch floats, sorted ascending.
        clear_f: Window exclusive ends as epoch floats.

    Returns:
        Object array, one entry per window: ``'gap_out'``, ``'max_out'``,
        ``'force_off'``, or NaN when no termination code was logged (e.g.
        codes 4–6 not loaded).  The highest code wins when several land in
        one window.
    """
    out = np.full(len(green_f), np.nan, dtype=object)
    term = events_df.loc[
        events_df["event_code"].isin(list(_TERMINATION_NAMES))
        & (events_df["parameter"] == phase),
        ["timestamp", "event_code"],
    ]
    if term.empty or len(green_f) == 0:
        return out

    ts = _ts_to_float(term["timestamp"])
    win = np.searchsorted(green_f, ts, side="right") - 1
    ok = (win >= 0) & (ts < clear_f[np.clip(win, 0, None)])
    if not ok.any():
        return out

    best = (
        pd.Series(term["event_code"].to_numpy()[ok], index=win[ok])
        .groupby(level=0).max()
    )
    out[best.index.to_numpy()] = best.map(_TERMINATION_NAMES).to_numpy()
    return out


def _empty_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return empty (cycle_df, vehicle_df) with the documented schemas."""
    return (
        pd.DataFrame(columns=_CYCLE_SCHEMA),
        pd.DataFrame(columns=_VEHICLE_SCHEMA),
    )


def _resolve_overhead(
    veh: pd.DataFrame,
    selected: pd.DataFrame,
    normalize: str,
    fixed_lost: Optional[float],
) -> pd.Series:
    """Compute the per-vehicle split-termination overhead in seconds.

    Args:
        veh: Per-vehicle rows already merged with the selected-cycle
            columns ``lost`` and ``clear_dur``.
        selected: Selected-cycle summary (used for pooled statistics).
        normalize: One of :data:`NORMALIZE_MODES`.
        fixed_lost: Constant overhead in seconds; required when
            ``normalize == 'fixed'``.

    Returns:
        Float Series aligned with *veh*.

    Raises:
        ValueError: On an unknown mode, or ``'fixed'`` without *fixed_lost*.
    """
    if normalize == "end_shift":
        return veh["lost"].astype(float)
    if normalize == "pooled":
        pooled = selected.groupby("det")["lost"].median()
        return veh["det"].map(pooled).astype(float)
    if normalize == "clearance":
        return veh["clear_dur"].astype(float)
    if normalize == "fixed":
        if fixed_lost is None:
            raise ValueError("normalize='fixed' requires fixed_lost (seconds).")
        return pd.Series(float(fixed_lost), index=veh.index)
    if normalize == "none":
        return pd.Series(0.0, index=veh.index)
    raise ValueError(
        f"Unknown normalize mode {normalize!r}; expected one of {NORMALIZE_MODES}."
    )


def _select_cycles(
    cycle_df: pd.DataFrame,
    pct: float,
    split_tolerance: float,
    stratify: bool,
) -> pd.DataFrame:
    """Select comparable, busy cycles from a per-cycle summary.

    Shared by :func:`rate_profiles` and :func:`discharge_profiles`.  A
    cycle is one split window, keyed by ``green_ts``; its volume is ``q``
    summed across detectors.

    * Default mode keeps cycles whose ``split`` lies strictly within
      ``(1 ± split_tolerance) ×`` the modal split, then the busiest *pct*
      percent of those.
    * Stratified mode skips the modal filter.  It groups cycles into
      strata by ``(coord_plan, round(split))`` and keeps the busiest *pct*
      percent **within** each stratum, then pools the survivors.  Splits
      shorter than the modal one stay represented, so a curve's domain
      isn't set by the longest-split plan alone.

    Args:
        cycle_df: Per-cycle summary from :func:`flow_rate`.
        pct: Percentage of the busiest cycles to keep (``1.0`` = top 1%).
        split_tolerance: Fractional tolerance around the modal split
            (default mode only).
        stratify: Use stratified selection.

    Returns:
        The selected rows of *cycle_df* (same columns, original order).
        Empty when nothing survives.
    """
    if cycle_df.empty:
        return cycle_df.iloc[0:0]

    q_level = (100.0 - pct) / 100.0

    if not stratify:
        modal = cycle_df["split"].mode().iloc[0]
        lo, hi = (1.0 - split_tolerance) * modal, (1.0 + split_tolerance) * modal
        sel = cycle_df.loc[(cycle_df["split"] > lo) & (cycle_df["split"] < hi)]
        if sel.empty:
            return sel
        q_by_cycle = sel.groupby("green_ts")["q"].sum()
        threshold = q_by_cycle.quantile(q_level)
        keep = q_by_cycle.index[q_by_cycle >= threshold]
        return sel.loc[sel["green_ts"].isin(keep)]

    per_cycle = (
        cycle_df.groupby("green_ts", as_index=False)
        .agg(coord_plan=("coord_plan", "first"),
             split=("split", "first"),
             q=("q", "sum"))
    )
    per_cycle["_stratum_split"] = per_cycle["split"].round()
    strata = per_cycle.groupby(["coord_plan", "_stratum_split"])["q"]
    threshold = strata.transform(lambda q: q.quantile(q_level))
    keep = per_cycle.loc[per_cycle["q"] >= threshold, "green_ts"]
    return cycle_df.loc[cycle_df["green_ts"].isin(keep)]


def _wide_profile(
    veh: pd.DataFrame,
    value_col: str,
    min_cycles: int,
) -> pd.DataFrame:
    """Pivot per-vehicle values into a wide t-indexed profile with mean columns.

    Per-detector ``"{det} Mean"`` columns interpolate each cycle column only
    *inside* its observed range (no extension past a cycle's last vehicle)
    and require at least *min_cycles* contributing cycles per grid row.
    The overall ``"Mean"`` column reproduces the established behaviour of
    carrying each cycle's final value forward (plain ``interpolate()``) and
    averaging across all cycle columns.

    Args:
        veh: Per-vehicle rows with ``_t`` (grid-rounded time), ``_label``
            (``"{det} {green_ts}"`` cycle column label), ``det`` and
            *value_col* columns.
        value_col: Name of the value column to pivot (``rate`` or ``inst``).
        min_cycles: Minimum simultaneous cycle observations required for a
            per-detector mean value at a given grid row.

    Returns:
        Wide DataFrame indexed by ``t`` — one column per cycle, one
        ``"{det} Mean"`` column per detector, and a final overall ``"Mean"``
        column.  Empty DataFrame when *veh* is empty.
    """
    if veh.empty:
        return pd.DataFrame()

    # pivot_table drops all-NaN columns (e.g. ``inst`` for a cycle with a
    # single vehicle has no headway); restore them so every detector keeps
    # its block and its ``"{det} Mean"`` column.
    wide = (
        veh.pivot_table(index="_t", columns="_label",
                        values=value_col, aggfunc="last")
        .reindex(columns=sorted(veh["_label"].unique()))
        .sort_index()
    )

    frames: List[pd.DataFrame] = []
    for det in sorted(veh["det"].unique()):
        cols = [c for c in wide.columns if c.split(" ")[0] == str(det)]
        if not cols:
            continue
        block = wide[cols].copy()
        inner = block.interpolate(limit_area="inside")
        block[f"{det} Mean"] = inner.dropna(thresh=min_cycles).mean(axis=1)
        frames.append(block)

    out = pd.concat(frames, axis=1)
    cycle_cols = [c for c in out.columns if not c.endswith("Mean")]
    out["Mean"] = out[cycle_cols].interpolate().mean(axis=1)
    out.index.name = "t"
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flow_rate(
    events_df: pd.DataFrame,
    phase: int,
    detector_ids: List[int],
    max_lost: Optional[float] = 10.0,
    plans: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Measure per-split stop-bar discharge for one phase and its detectors.

    Raw measurement only — no rate normalisation is applied here; pass the
    results to :func:`rate_profiles` to compute effective cumulative flow
    rates under a chosen overhead model.

    Algorithm
    ---------
    1. Build split windows ``[green_ts, clear_end_ts)`` per cycle via
       ``_build_split_windows`` (which wraps ``_build_phase_intervals``).
    2. Assign Code-81 departures to windows with two ``np.searchsorted``
       calls — the same O(n log m) vectorised containment pattern used in
       ``atspm.analysis.aog``.
    3. Per (window, detector) group: cumulative count, elapsed time since
       green onset, and headway to the next departure.
    4. Drop groups whose end slack (``clear_end_ts`` minus last departure)
       exceeds *max_lost* — the phase was not discharging at capacity.
       Skipped when *max_lost* is ``None``: every group is kept with its
       ``lost`` value, so a saturation classifier can count the cycles
       that would fail the filter.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            Timestamps may be UTC epoch floats **or** tz-aware Timestamps.
            Gap markers (``event_code == -1``) must be present.
        phase: Signal phase number.
        detector_ids: ``parameter`` values of the stop-bar detectors for
            *phase*.  Typically sourced from the ``Det_P{phase}_Stopbar``
            config key.
        max_lost: Maximum seconds of slack between the last departure and
            the end of the split window.  Default ``10.0``.  ``None``
            disables the filter.
        plans: Optional coordination-plan filter; when provided, only split
            windows whose ``coord_plan`` is in this list are analysed.

    Returns:
        Tuple ``(cycle_df, vehicle_df)``:

        * ``cycle_df`` — one row per (split window, detector)::

              det          int     – detector ID
              phase        int     – echoed phase number
              green_ts     float | Timestamp – green onset (cycle key)
              cycle_start  float | Timestamp
              coord_plan   float
              split        float   – split window length in seconds
              green_dur    float   – green portion in seconds
              clear_dur    float   – yellow + red clearance in seconds
              lost         float   – end slack in seconds (≤ max_lost
                                     unless max_lost is None)
              q            int     – vehicles discharged in the window
              termination  str     – 'gap_out' / 'max_out' / 'force_off',
                                     NaN when codes 4–6 are absent

          With ``max_lost=None`` every (window, detector) pair has a row:
          a lane with no departure in a window gets ``q = 0`` and ``lost``
          equal to the whole window.

        * ``vehicle_df`` — one row per departure::

              det          int
              green_ts     float | Timestamp – key into cycle_df
              t            float   – seconds since green onset
              n            int     – cumulative vehicle count
              headway      float   – seconds to next departure (NaN for last)

        Both empty (correct schemas) when no qualifying windows or
        departures exist, or when *detector_ids* is empty.  Codes 4/5/6 in
        *events_df* fill ``termination``; without them it is all NaN.
    """
    if events_df.empty or not detector_ids:
        return _empty_results()

    windows = _build_split_windows(events_df, phase)

    if plans is not None and not windows.empty:
        windows = (
            windows.loc[windows["coord_plan"].isin(plans)]
            .reset_index(drop=True)
        )

    if windows.empty:
        return _empty_results()

    det_mask = (
        (events_df["event_code"] == _CODE_DET_OFF)
        & (events_df["parameter"].isin(detector_ids))
    )
    det = (
        events_df.loc[det_mask, ["timestamp", "parameter"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if det.empty:
        return _empty_results()

    green_f = _ts_to_float(windows["green_ts"])
    clear_f = _ts_to_float(windows["clear_end_ts"])
    det_f = _ts_to_float(det["timestamp"])

    # Window containment: last green onset <= departure < that window's end.
    win_idx = np.searchsorted(green_f, det_f, side="right") - 1
    ok = win_idx >= 0
    safe_idx = np.clip(win_idx, 0, None)
    ok &= det_f < clear_f[safe_idx]

    if not ok.any():
        return _empty_results()

    veh = pd.DataFrame({
        "win": win_idx[ok],
        "det": det["parameter"].values[ok].astype(int),
        "ts_f": det_f[ok],
    }).sort_values(["win", "det", "ts_f"]).reset_index(drop=True)

    grp = veh.groupby(["win", "det"], sort=False)
    veh["n"] = grp.cumcount() + 1
    veh["t"] = veh["ts_f"] - green_f[veh["win"].values]
    veh["headway"] = grp["ts_f"].shift(-1) - veh["ts_f"]
    veh["_lost"] = clear_f[veh["win"].values] - grp["ts_f"].transform("max")

    if max_lost is not None:
        veh = veh.loc[veh["_lost"] <= max_lost].reset_index(drop=True)

    if veh.empty:
        return _empty_results()

    summary = (
        veh.groupby(["win", "det"], as_index=False)
        .agg(q=("n", "max"), lost=("_lost", "first"))
    )
    if max_lost is None:
        # Unfiltered output covers every (window, detector), so a lane with
        # no departure in a window is visible to a classifier; its slack is
        # the whole window.
        full = pd.MultiIndex.from_product(
            [np.arange(len(windows)), sorted(set(int(d) for d in detector_ids))],
            names=["win", "det"],
        ).to_frame(index=False)
        summary = full.merge(summary, on=["win", "det"], how="left")
        empty_lane = summary["q"].isna()
        win_len = clear_f - green_f
        summary.loc[empty_lane, "lost"] = win_len[summary.loc[empty_lane, "win"].to_numpy()]
        summary["q"] = summary["q"].fillna(0).astype(int)

    summary["termination"] = _window_terminations(
        events_df, phase, green_f, clear_f
    )[summary["win"].to_numpy()]

    win_cols = (
        windows[["green_ts", "cycle_start", "coord_plan",
                 "split_dur", "green_dur", "clear_dur"]]
        .rename(columns={"split_dur": "split"})
    )
    summary = summary.join(win_cols, on="win")
    summary["phase"] = int(phase)

    cycle_df = (
        summary[_CYCLE_SCHEMA]
        .sort_values(["green_ts", "det"])
        .reset_index(drop=True)
        .round({"split": 2, "green_dur": 2, "clear_dur": 2, "lost": 2})
    )

    veh = veh.join(windows["green_ts"], on="win")
    vehicle_df = veh[_VEHICLE_SCHEMA].reset_index(drop=True)

    return cycle_df, vehicle_df


def rate_profiles(
    cycle_df: pd.DataFrame,
    vehicle_df: pd.DataFrame,
    pct: float = 1.0,
    split_tolerance: float = 0.10,
    normalize: str = "end_shift",
    fixed_lost: Optional[float] = None,
    grid_step: float = 0.5,
    min_cycles: int = 5,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select comparable cycles and build effective flow-rate profiles.

    Cycle selection
    ---------------
    1. Keep cycles whose ``split`` lies strictly within
       ``(1 ± split_tolerance) ×`` the modal split, so like cycles are
       compared on the same time base.
    2. Of those, keep the busiest *pct* percent by total vehicles across
       all detectors — split optimisation only matters under demand.

    With *stratify*, step 1 is skipped and step 2 runs within each
    ``(coord_plan, round(split))`` stratum (see ``_select_cycles``).

    Normalisation (overhead) modes
    ------------------------------
    The effective cumulative rate at each departure is
    ``3600·n / (t + overhead)``, where the overhead charges every
    hypothetical split length the fixed cost of terminating a split:

    * ``'end_shift'`` (default) — each cycle's own measured end slack
      (``lost``); reproduces the established notebook behaviour.
    * ``'pooled'``    — the per-detector **median** end slack across the
      selected cycles; same intent, less cycle-to-cycle noise.
    * ``'clearance'`` — each cycle's actual yellow + red clearance
      duration; a deterministic "cost of ending a split here".
    * ``'fixed'``     — a constant *fixed_lost* seconds.
    * ``'none'``      — raw ``3600·n / t``.

    The instantaneous rate (``3600 / headway``) is never normalised.

    Args:
        cycle_df: Per-cycle summary from :func:`flow_rate`.
        vehicle_df: Per-vehicle detail from :func:`flow_rate`.
        pct: Percentage of the busiest modal-split cycles to keep
            (``1.0`` = top 1%).  Default ``1.0``.
        split_tolerance: Fractional tolerance around the modal split.
            Default ``0.10``.
        normalize: Overhead mode; one of :data:`NORMALIZE_MODES`.
        fixed_lost: Constant overhead in seconds; required when
            ``normalize == 'fixed'``.
        grid_step: Time-grid resolution in seconds for the wide profiles.
            Default ``0.5``.
        min_cycles: Minimum simultaneous cycles required for a per-detector
            mean value at a grid row.  Default ``5``.
        stratify: Select the busiest cycles within each
            ``(coord_plan, round(split))`` stratum instead of around the
            modal split.  Default ``False``.

    Returns:
        Tuple ``(selected_df, rate_df, inst_df)``:

        * ``selected_df`` — the selected rows of *cycle_df* plus, under the
          chosen normalisation::

              t_max      float – elapsed seconds at the peak effective rate
              max_rate   float – peak effective cumulative rate (vphpl)
              n_at_max   int   – cumulative vehicles at the peak

        * ``rate_df`` — wide effective-cumulative-rate profile indexed by
          ``t`` (``grid_step`` grid): one ``"{det} {green_ts}"`` column per
          selected cycle, ``"{det} Mean"`` per detector, overall ``"Mean"``.
        * ``inst_df`` — same layout for the instantaneous rate.

        All empty (correct schemas) when nothing survives selection.

    Raises:
        ValueError: On an unknown *normalize* mode, or ``'fixed'`` without
            *fixed_lost*.
    """
    if normalize not in NORMALIZE_MODES:
        raise ValueError(
            f"Unknown normalize mode {normalize!r}; expected one of {NORMALIZE_MODES}."
        )

    _empty_sel = pd.DataFrame(columns=_SELECTED_SCHEMA)

    if cycle_df.empty or vehicle_df.empty:
        return _empty_sel, pd.DataFrame(), pd.DataFrame()

    # --- 1–2. Comparable, busiest cycles ------------------------------------
    selected = _select_cycles(cycle_df, pct, split_tolerance, stratify).copy()

    if selected.empty:
        return _empty_sel, pd.DataFrame(), pd.DataFrame()

    # --- 3. Normalised rates per vehicle -------------------------------------
    veh = vehicle_df.merge(
        selected[["det", "green_ts", "lost", "clear_dur"]],
        on=["det", "green_ts"],
        how="inner",
    )

    if veh.empty:
        return _empty_sel, pd.DataFrame(), pd.DataFrame()

    overhead = _resolve_overhead(veh, selected, normalize, fixed_lost)
    denom = veh["t"] + overhead
    veh["rate"] = np.where(denom > 0.0, 3600.0 * veh["n"] / denom, np.nan)
    veh["inst"] = (3600.0 / veh["headway"]).replace([np.inf, -np.inf], np.nan)

    # --- 4. Peak of the effective-rate curve per cycle ------------------------
    peaks_idx = (
        veh.dropna(subset=["rate"])
        .groupby(["det", "green_ts"])["rate"]
        .idxmax()
    )
    peaks = (
        veh.loc[peaks_idx, ["det", "green_ts", "t", "rate", "n"]]
        .rename(columns={"t": "t_max", "rate": "max_rate", "n": "n_at_max"})
    )
    selected = (
        selected.merge(peaks, on=["det", "green_ts"], how="left")
        [_SELECTED_SCHEMA]
        .round({"t_max": 2, "max_rate": 1})
        .reset_index(drop=True)
    )

    # --- 5. Wide t-indexed profiles -------------------------------------------
    veh["_t"] = (veh["t"] / grid_step).round() * grid_step
    veh["_label"] = veh["det"].astype(str) + " " + veh["green_ts"].astype(str)

    rate_df = _wide_profile(veh, "rate", min_cycles)
    inst_df = _wide_profile(veh, "inst", min_cycles)

    return selected, rate_df, inst_df


def discharge_profiles(
    cycle_df: pd.DataFrame,
    vehicle_df: pd.DataFrame,
    pct: float = 1.0,
    split_tolerance: float = 0.10,
    stratify: bool = False,
    grid_step: float = 0.5,
    min_cycles: int = 5,
    rolling: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the approach cumulative discharge curve ``N(t)`` for one phase.

    This is the curve the throughput optimizer consumes
    (``docs/design_optimizer_solver.md`` D0): approach-total vehicles
    served by elapsed split time ``t``.  No overhead normalisation is
    applied.  In the optimizer each split already includes its clearance
    and the splits sum to the cycle length, so ``end_shift`` would charge
    the termination cost twice.

    Algorithm
    ---------
    1. Select cycles with the same rules as :func:`rate_profiles`.
    2. Pivot each selected cycle's cumulative count ``n`` onto the
       ``grid_step`` grid and take the per-detector mean, which needs at
       least *min_cycles* cycles at a grid row.
    3. The data domain ``t_dom`` is the earliest of the detectors' last
       supported grid rows.  The profile is truncated there, so its last
       index marks the boundary of what the data can say.
    4. On the full grid ``[0, t_dom]``: interior gaps are interpolated
       linearly, rows before a detector's first supported value are 0.
       The approach total is the **sum** of the per-detector means (not
       their average, which is a per-lane quantity), made monotone with a
       running maximum.
    5. ``inst`` is the sum of the per-detector mean instantaneous rates
       (``3600 / headway``), smoothed with a centred rolling mean.  It is
       a diagnostic only.

    Known approximation: for a hypothetical split ``s`` shorter than the
    measured one, ``N(s)`` counts continued-green discharge at ``s``
    rather than the few vehicles that would cross during a clearance
    ending at ``s``.  The bias is at most about one vehicle and is the
    same for every candidate split.

    Gap Marker Rule: the inputs come from :func:`flow_rate`, whose split
    windows never span a gap marker; this function adds no event logic.

    Args:
        cycle_df: Per-cycle summary from :func:`flow_rate`.
        vehicle_df: Per-vehicle detail from :func:`flow_rate`.
        pct: Percentage of the busiest cycles to keep (``1.0`` = top 1%).
        split_tolerance: Fractional tolerance around the modal split.
        stratify: Select within ``(coord_plan, round(split))`` strata
            instead of around the modal split.
        grid_step: Time-grid resolution in seconds.  Default ``0.5``.
        min_cycles: Minimum simultaneous cycles for a per-detector mean
            at a grid row.  Default ``5``.
        rolling: Centred rolling-mean window (grid rows) for ``inst``.
            Default ``5``.

    Returns:
        Tuple ``(selected_df, profile_df)``:

        * ``selected_df`` — the selected rows of *cycle_df*.
        * ``profile_df`` — indexed by ``t`` (uniform grid from 0.0 to
          ``t_dom``) with float columns::

              n     – approach cumulative vehicles served by t (monotone)
              inst  – approach instantaneous rate, vph (smoothed; NaN where
                      any detector lacks support or the window is short)

        ``profile_df`` is empty (same columns) when nothing survives
        selection, or when any detector never reaches *min_cycles*
        support, because the approach total would then omit a lane.
    """
    empty_profile = pd.DataFrame(
        columns=["n", "inst"], index=pd.Index([], name="t"), dtype=float
    )

    if cycle_df.empty or vehicle_df.empty:
        return pd.DataFrame(columns=_CYCLE_SCHEMA), empty_profile

    selected = (
        _select_cycles(cycle_df, pct, split_tolerance, stratify)
        .reset_index(drop=True)
    )
    if selected.empty:
        return selected, empty_profile

    veh = vehicle_df.merge(
        selected[["det", "green_ts"]], on=["det", "green_ts"], how="inner"
    )
    if veh.empty:
        return selected, empty_profile

    veh["inst"] = (3600.0 / veh["headway"]).replace([np.inf, -np.inf], np.nan)
    veh["_t"] = (veh["t"] / grid_step).round() * grid_step
    veh["_label"] = veh["det"].astype(str) + " " + veh["green_ts"].astype(str)

    mean_cols = [f"{d} Mean" for d in sorted(veh["det"].unique())]
    n_means = _wide_profile(veh, "n", min_cycles)[mean_cols]
    inst_means = _wide_profile(veh, "inst", min_cycles)[mean_cols]

    last_valid = n_means.apply(pd.Series.last_valid_index)
    if last_valid.isna().any():
        return selected, empty_profile

    n_steps = int(round(float(last_valid.min()) / grid_step))
    grid = pd.Index(np.arange(n_steps + 1) * grid_step, name="t")

    def _on_grid(means: pd.DataFrame) -> pd.DataFrame:
        # Interpolate on the union so a grid row between two sparse rows,
        # including the t_dom row itself, takes its value from both sides.
        full = means.reindex(means.index.union(grid))
        return full.interpolate(method="index", limit_area="inside").reindex(grid)

    n_total = _on_grid(n_means).fillna(0.0).sum(axis=1)
    inst_total = (
        _on_grid(inst_means)
        .sum(axis=1, min_count=len(mean_cols))
        .rolling(rolling, center=True)
        .mean()
    )

    profile_df = pd.DataFrame(
        {
            "n": np.maximum.accumulate(n_total.to_numpy(dtype=float)),
            "inst": inst_total.to_numpy(dtype=float),
        },
        index=grid,
    )
    return selected, profile_df



_SATURATION_SCHEMA = [
    "phase",
    "n_cycles",
    "n_obs",
    "capped_rate",
    "pass_rate",
    "min_lane_pass_rate",
    "saturated",
]


def _saturation_pass(
    cycle_df: pd.DataFrame,
    max_lost: float,
    all_lanes: bool,
) -> pd.Series:
    """Boolean pass flag per *cycle_df* row.

    A row passes when its window ended by max-out or force-off and its
    lane had ``lost <= max_lost``.  With *all_lanes*, every lane of the
    window must pass for any row of it to pass.
    """
    capped = cycle_df["termination"].isin(_SATURATED_TERMINATIONS)
    lane_ok = cycle_df["lost"].astype(float) <= max_lost
    if not all_lanes:
        return capped & lane_ok
    every_lane = lane_ok.groupby(
        [cycle_df["phase"], cycle_df["green_ts"]]
    ).transform("all")
    return capped & every_lane


def saturation_state(
    cycle_df: pd.DataFrame,
    max_lost: float = 10.0,
    threshold: float = 0.8,
    all_lanes: bool = True,
) -> pd.DataFrame:
    """Advisory saturation check per phase from its share of capped cycles.

    **Advisory only.**  The optimizer takes the saturated phases as an
    engineer's declaration, not from this function: end slack can't
    reliably tell saturation apart (a coordinated phase always forces off,
    and a busy unsaturated through movement often has a departure near
    yellow).  This reports the evidence beside that declaration.  Curves
    are built by percentile selection (:func:`discharge_profiles`), never
    from these verdicts.

    A window qualifies when it maxed out or was forced off, and
    (``all_lanes=True``, the default) every lane had ``lost <= max_lost``.
    Gap-outs never qualify, because ``lost`` includes clearance and a
    gap-out scores under ``max_lost`` by construction.  ``pass_rate`` is the share of windows that
    qualify; with ``all_lanes=False`` it is the share of (window, lane)
    rows.  A phase is saturated when ``pass_rate`` reaches *threshold*.

    The input must be **unfiltered** — ``flow_rate(..., max_lost=None)`` —
    so that failing lanes and empty lanes have rows.  For a per-period
    verdict (e.g. per coordination plan), filter *cycle_df* first.
    ``threshold=0.8`` is provisional until real distributions are seen.

    Gap Marker Rule: the input comes from :func:`flow_rate`, whose split
    windows never span a gap marker; this function adds no event logic.

    Args:
        cycle_df: Unfiltered per-cycle summary from :func:`flow_rate`.
        max_lost: Per-lane end-slack limit in seconds.  Default ``10.0``.
        threshold: Minimum ``pass_rate`` for a saturated verdict.
            Default ``0.8``.
        all_lanes: Require every lane of a window to pass.  Default
            ``True``.

    Returns:
        One row per phase, sorted by phase::

            phase               int
            n_cycles            int    – split windows observed
            n_obs               int    – (window, detector) rows
            capped_rate         float  – share of windows ended by
                                         max-out or force-off
            pass_rate           float  – share of windows (or rows, when
                                         all_lanes=False) that qualify
            min_lane_pass_rate  float  – lowest per-lane share of rows
                                         qualifying on their own
            saturated           bool   – pass_rate >= threshold

        Empty (same columns) when *cycle_df* is empty.
    """
    if cycle_df.empty:
        return pd.DataFrame(columns=_SATURATION_SCHEMA)

    obs = cycle_df[["phase", "det", "green_ts"]].copy()
    obs["_capped"] = cycle_df["termination"].isin(_SATURATED_TERMINATIONS)
    obs["_lane"] = _saturation_pass(cycle_df, max_lost, all_lanes=False)
    obs["_pass"] = _saturation_pass(cycle_df, max_lost, all_lanes)

    windows = obs.groupby(["phase", "green_ts"]).agg(
        _capped=("_capped", "first"), _pass=("_pass", "all")
    )
    by_window = windows.groupby(level="phase").agg(
        n_cycles=("_pass", "size"),
        capped_rate=("_capped", "mean"),
        _window_rate=("_pass", "mean"),
    )
    by_phase = obs.groupby("phase").agg(
        n_obs=("_pass", "size"), _row_rate=("_pass", "mean")
    )
    by_lane = (
        obs.groupby(["phase", "det"])["_lane"].mean()
        .groupby(level="phase").min()
        .rename("min_lane_pass_rate")
    )

    out = by_window.join(by_phase).join(by_lane).reset_index()
    out["pass_rate"] = out["_window_rate"] if all_lanes else out["_row_rate"]
    out["phase"] = out["phase"].astype(int)
    out["saturated"] = out["pass_rate"] >= threshold
    return out[_SATURATION_SCHEMA].sort_values("phase").reset_index(drop=True)
