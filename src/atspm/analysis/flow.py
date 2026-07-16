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

    wide = (
        veh.pivot_table(index="_t", columns="_label",
                        values=value_col, aggfunc="last")
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
    max_lost: float = 10.0,
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
            the end of the split window.  Default ``10.0``.
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
              lost         float   – end slack in seconds (≤ max_lost)
              q            int     – vehicles discharged in the window

        * ``vehicle_df`` — one row per departure::

              det          int
              green_ts     float | Timestamp – key into cycle_df
              t            float   – seconds since green onset
              n            int     – cumulative vehicle count
              headway      float   – seconds to next departure (NaN for last)

        Both empty (correct schemas) when no qualifying windows or
        departures exist, or when *detector_ids* is empty.
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

    veh = veh.loc[veh["_lost"] <= max_lost].reset_index(drop=True)

    if veh.empty:
        return _empty_results()

    summary = (
        veh.groupby(["win", "det"], as_index=False)
        .agg(q=("n", "max"), lost=("_lost", "first"))
    )
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select comparable cycles and build effective flow-rate profiles.

    Cycle selection
    ---------------
    1. Keep cycles whose ``split`` lies strictly within
       ``(1 ± split_tolerance) ×`` the modal split, so like cycles are
       compared on the same time base.
    2. Of those, keep the busiest *pct* percent by total vehicles across
       all detectors — split optimisation only matters under demand.

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

    # --- 1. Comparable-split filter -----------------------------------------
    modal = cycle_df["split"].mode().iloc[0]
    lo, hi = (1.0 - split_tolerance) * modal, (1.0 + split_tolerance) * modal
    sel = cycle_df.loc[(cycle_df["split"] > lo) & (cycle_df["split"] < hi)]

    if sel.empty:
        return _empty_sel, pd.DataFrame(), pd.DataFrame()

    # --- 2. Busiest-percentile filter ----------------------------------------
    q_by_cycle = sel.groupby("green_ts")["q"].sum()
    threshold = q_by_cycle.quantile((100.0 - pct) / 100.0)
    keep = q_by_cycle.index[q_by_cycle >= threshold]
    selected = sel.loc[sel["green_ts"].isin(keep)].copy()

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
