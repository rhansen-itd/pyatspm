"""
ATSPM Count Calculations (Functional Core)

Pure functions only. No I/O, no SQL, no side effects.
Input/output is DataFrames and plain dicts.

Provides vehicle (detector) counts aggregated to movements and pedestrian
actuation counts, both on a time-bin or per-cycle basis.

Package Location: src/atspm/analysis/counts.py

Gap Marker Rule:
    Rows with Code == -1 (event_code = -1, parameter = -1) mark data
    discontinuities in the events table.  All stateful logic — phase
    signal-state forward-fill, pedestrian call-to-service pairing, detector
    on/off carry-forward — must treat a gap marker as a hard reset for ALL
    phases and detectors.  No state derived before a gap marker may influence
    any calculation after it.  This is enforced here via ``_segment_id()``,
    which assigns a monotonically increasing segment number that increments
    at every gap marker.  All groupby operations that depend on continuity
    include the segment as an additional grouping key.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gap-marker sentinel
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1  # event_code value used for discontinuity markers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vehicle_counts(
    events_df: pd.DataFrame,
    movements: Dict[str, List[int]],
    exclusions: Optional[List[Dict[str, Any]]] = None,
    bin_len: Union[int, str] = 60,
    hourly: bool = False,
    include_detectors: bool = False,
) -> pd.DataFrame:
    """
    Aggregate detector on-events (Code 82) into movement and/or detector counts.

    Exclusions remove detector actuations that occur while a given phase is in
    a specified signal state (e.g., drop detector 33 when phase 2 is Red).
    Phase status is derived directly from Code 1/8/9/11/12 events in
    *events_df* — no pre-joined comb_gyr_det DataFrame is required.

    Gap markers (Code -1) reset phase-state forward-fill so that signal state
    from before a data discontinuity cannot contaminate exclusion decisions
    made after it.

    Args:
        events_df: Legacy-format DataFrame with columns
            ``[TS_start, Code, ID, Cycle_start, Coord_plan]``.
            ``TS_start`` and ``Cycle_start`` must be timezone-aware
            ``datetime64`` or UTC-epoch floats (both are handled).
        movements: Mapping of movement label to list of detector IDs.
            Example: ``{"EBL": [5, 6], "EBT": [7, 8], "NBL": [1]}``.
        exclusions: Optional list of exclusion dicts, each with keys
            ``detector`` (int), ``phase`` (int), ``status`` (str, one of
            ``"Green" | "Yellow" | "Red"``).  Any actuation of *detector*
            while *phase* is in *status* is dropped before aggregation.
        bin_len: Aggregation interval in minutes, or ``"cycle"`` for
            per-cycle aggregation.  Default ``60``.
        hourly: When ``True`` and *bin_len* is numeric, scale counts to an
            equivalent hourly flow rate (counts * 60 / bin_len).
        include_detectors: When ``True``, raw per-detector counts are
            included as additional columns alongside movement totals.

    Returns:
        DataFrame indexed by ``Time`` (bin start or ``Cycle_start``).
        Columns are movement labels followed by ``TEV`` (total entering
        vehicles), then optionally individual detector IDs if
        *include_detectors* is ``True``.
        Movement columns contain ``float`` to accommodate hourly scaling;
        detector columns are ``int`` when *hourly* is ``False``.
    """
    # ---- isolate detector-on events (gap markers are Code -1, never 82) ----
    df_det = events_df.loc[events_df["Code"] == 82].copy()

    if df_det.empty:
        return pd.DataFrame()

    # ---- apply exclusions (gap-aware via _apply_exclusions) -----------------
    if exclusions:
        df_det = _apply_exclusions(df_det, events_df, exclusions)

    # ---- aggregate ----------------------------------------------------------
    if bin_len == "cycle":
        det_agg = _aggregate_by_cycle(df_det)
        hourly_factor = 1.0
    else:
        det_agg = _aggregate_by_bin(df_det, int(bin_len))
        hourly_factor = (60.0 / int(bin_len)) if hourly else 1.0

    if det_agg.empty:
        return pd.DataFrame()

    det_agg.index.name = "Time"

    # ---- build movement columns (vectorised) --------------------------------
    mvmt_df = _sum_movements(det_agg, movements, hourly_factor)

    # ---- optionally append raw detector columns -----------------------------
    if include_detectors:
        det_cols = det_agg.reindex(columns=sorted(det_agg.columns))
        if hourly_factor != 1.0:
            det_cols = (det_cols * hourly_factor).round(1)
        result = pd.concat([mvmt_df, det_cols], axis=1)
    else:
        result = mvmt_df

    return result


def ped_counts(
    events_df: pd.DataFrame,
    bin_len: Union[int, str] = 60,
    hourly: bool = False,
) -> pd.DataFrame:
    """
    Count pedestrian phases served that were preceded by a pedestrian call.

    A pedestrian *service* (Code 21) is counted only when at least one
    pedestrian *call* (Code 45) for the same phase was registered after the
    previous service for that phase — and within the same continuous data
    segment (i.e. no gap marker between the call and the service).
    Multiple button presses between services count as a single actuation.
    Recall (Code 21 without a prior Code 45 in the same segment) is excluded.

    Args:
        events_df: Legacy-format DataFrame with columns
            ``[TS_start, Code, ID, Cycle_start]``.  Gap markers (Code -1)
            must be present in this DataFrame to take effect.
        bin_len: Aggregation interval in minutes, or ``"cycle"`` for
            per-cycle aggregation.  Default ``60``.
        hourly: When ``True`` and *bin_len* is numeric, scale counts to
            hourly rate.

    Returns:
        DataFrame indexed by ``Time``.  Columns are ``"Ped {phase_id}"``
        for each phase with activity, plus ``"Ped Total"``.
        Returns an empty DataFrame when no relevant events are present.
    """
    # Include gap markers so _segment_id can see them, then filter to
    # ped-relevant codes for actual processing.
    df_all = events_df.loc[events_df["Code"].isin([_GAP_CODE, 21, 45])].copy()
    if df_all.empty:
        return pd.DataFrame()

    df_all = df_all.sort_values("TS_start").reset_index(drop=True)

    # Assign a segment ID that increments at each gap marker.
    # Gap marker rows themselves are dropped after segment assignment.
    df_all["_seg"] = _segment_id(df_all)
    df_p = df_all.loc[df_all["Code"].isin([21, 45])].copy()

    if df_p.empty:
        return pd.DataFrame()

    df_p = df_p.sort_values(["_seg", "ID", "TS_start"]).reset_index(drop=True)

    # ---- vectorised legitimate-service detection ----------------------------
    # Grouping by (_seg, ID) ensures no state crosses a gap marker.
    #
    # Within each group:
    #   1. Assign a "service group" index that increments at each Code 21
    #      (shifted so the 21 itself belongs to the group it closes).
    #   2. A Code 21 is legitimate when its group contained at least one
    #      Code 45.

    df_p["_is_21"] = (df_p["Code"] == 21).astype(np.int8)
    df_p["_is_45"] = (df_p["Code"] == 45).astype(np.int8)

    df_p["_svc_grp"] = (
        df_p.groupby(["_seg", "ID"])["_is_21"]
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    df_p["_has_call"] = (
        df_p.groupby(["_seg", "ID", "_svc_grp"])["_is_45"]
        .transform("sum") > 0
    )

    legit_mask = (df_p["Code"] == 21) & df_p["_has_call"]
    df_legit = df_p.loc[legit_mask].copy()

    if df_legit.empty:
        return pd.DataFrame()

    # ---- aggregate ----------------------------------------------------------
    hourly_factor = 1.0
    if bin_len == "cycle":
        df_counts = (
            df_legit.groupby(["Cycle_start", "ID"])
            .size()
            .unstack(fill_value=0)
        )
    else:
        hourly_factor = (60.0 / int(bin_len)) if hourly else 1.0
        df_counts = (
            df_legit.set_index("TS_start")
            .groupby("ID")
            .resample(f"{int(bin_len)}min")
            .size()
            .unstack(level="ID")
            .fillna(0)
            .astype(int)
        )

    if hourly_factor != 1.0:
        df_counts = df_counts * hourly_factor

    df_counts["Ped Total"] = df_counts.sum(axis=1)
    df_counts.index.name = "Time"
    df_counts.rename(
        columns={c: f"Ped {c}" for c in df_counts.columns if c != "Ped Total"},
        inplace=True,
    )

    return df_counts


# ---------------------------------------------------------------------------
# Config parsing helpers (called by the Imperative Shell)
# ---------------------------------------------------------------------------

def parse_movements_from_config(config: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Extract movement-to-detector mapping from a flat config dict.

    Config keys of the form ``TM_{label}`` (e.g., ``TM_EBL``) whose values
    are comma-separated detector ID strings are parsed into a dict of
    ``{label: [det_id, ...]}``.

    Args:
        config: Flat config dict as returned by
            ``DatabaseManager.get_config_at_date()``.

    Returns:
        Dict mapping movement labels to lists of integer detector IDs.
        Empty dict if no ``TM_*`` keys are found.
    """
    movements: Dict[str, List[int]] = {}
    for key, value in config.items():
        if not key.startswith("TM_") or key == "TM_Exclusions":
            continue
        if not value or (isinstance(value, float) and pd.isna(value)):
            continue
        label = key[3:]  # strip "TM_"
        det_ids = [
            int(v.strip())
            for v in str(value).split(",")
            if v.strip().isdigit()
        ]
        if det_ids:
            movements[label] = det_ids
    return movements


def parse_exclusions_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract exclusion rules from a flat config dict.

    Reads the ``TM_Exclusions`` JSON field produced by
    ``DatabaseManager._parse_exclusions()``.

    Args:
        config: Flat config dict.

    Returns:
        List of exclusion dicts with keys ``detector`` (int),
        ``phase`` (int), ``status`` (str).
    """
    raw = config.get("TM_Exclusions")
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Phase signal-state codes used for exclusion filtering
_PHASE_GREEN_CODES  = frozenset({1})         # Green start
_PHASE_YELLOW_CODES = frozenset({8})         # Yellow start
_PHASE_RED_CODES    = frozenset({9, 11, 12}) # Red clearance, Red, Red+overlap
_ALL_PHASE_CODES    = _PHASE_GREEN_CODES | _PHASE_YELLOW_CODES | _PHASE_RED_CODES

_STATUS_CODES: Dict[str, frozenset] = {
    "Green":  _PHASE_GREEN_CODES,
    "Yellow": _PHASE_YELLOW_CODES,
    "Red":    _PHASE_RED_CODES,
}

_CODE_TO_STATE: Dict[int, str] = {
    c: label
    for label, codes in _STATUS_CODES.items()
    for c in codes
}


def _segment_id(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of integer segment IDs, one per row of *df*.

    The segment ID starts at 0 and increments by 1 at every row where
    ``Code == _GAP_CODE`` (-1).  Rows within the same continuous block of
    data share the same segment ID.  Gap-marker rows themselves belong to
    the segment that closes at the gap; callers drop them after this call.

    Args:
        df: DataFrame sorted by ``TS_start``, containing a ``Code`` column.

    Returns:
        Integer Series aligned with *df*'s index.
    """
    return (df["Code"] == _GAP_CODE).cumsum().astype(np.int32)


def _apply_exclusions(
    df_det: pd.DataFrame,
    events_df: pd.DataFrame,
    exclusions: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Remove detector actuations that occur while a specified phase is in a
    given signal state.

    Phase state is forward-filled from Code 1/8/9/11/12 events.  A gap
    marker (Code -1) resets the forward-fill for all phases, so that signal
    state from before a data discontinuity cannot affect exclusion decisions
    made after it.

    For each exclusion rule ``{detector, phase, status}``:
    - Extract phase state-change events and gap markers for *phase*.
    - Insert ``None`` sentinels at gap-marker timestamps to break the fill.
    - Use ``merge_asof`` backward-fill to assign state to each actuation.
    - Drop actuations where state matches *status*.

    NaN state (actuation after a gap but before the next known phase-state
    event) is treated as unknown and the actuation is kept.

    Args:
        df_det:     Detector-on events (Code 82) to filter.
        events_df:  Full raw event DataFrame (needed for phase state and gaps).
        exclusions: List of exclusion rule dicts.

    Returns:
        Filtered copy of *df_det*.
    """
    if df_det.empty or not exclusions:
        return df_det

    needed_phases = {int(exc["phase"]) for exc in exclusions}

    # Build a gap-reset state DataFrame for each needed phase.
    # Gap-marker rows are inserted with _state = None so that merge_asof
    # backward-fill returns NaN for any actuation after a gap (until the
    # next real phase-state event appears).
    phase_state_df: Dict[int, pd.DataFrame] = {}
    for ph in needed_phases:
        ph_events = events_df.loc[
            events_df["Code"].isin(_ALL_PHASE_CODES) & (events_df["ID"] == ph),
            ["TS_start", "Code"],
        ].copy()
        ph_events["_state"] = ph_events["Code"].map(_CODE_TO_STATE)

        gap_events = events_df.loc[
            events_df["Code"] == _GAP_CODE,
            ["TS_start"],
        ].copy()
        gap_events["_state"] = None  # explicit reset sentinel

        combined = (
            pd.concat(
                [ph_events[["TS_start", "_state"]], gap_events],
                ignore_index=True,
            )
            .sort_values("TS_start")
            .drop_duplicates("TS_start", keep="last")
            .reset_index(drop=True)
        )

        if not combined.empty:
            phase_state_df[ph] = combined

    if not phase_state_df:
        return df_det

    df_det = df_det.copy()
    drop_idx: set = set()

    for exc in exclusions:
        det_id = int(exc["detector"])
        ph     = int(exc["phase"])
        status = str(exc["status"])

        if ph not in phase_state_df:
            continue

        state_df = phase_state_df[ph]

        # Carry original df_det index through the merge
        mask = df_det["ID"] == det_id
        det_subset = (
            df_det.loc[mask, ["TS_start"]]
            .sort_values("TS_start")
            .reset_index()
            .rename(columns={"index": "_orig_idx"})
        )

        if det_subset.empty:
            continue

        # merge_asof backward-fill: each actuation gets the most recent
        # state entry at or before its timestamp.  Gap-sentinel rows
        # (state=None) cause NaN to be returned for actuations that follow
        # a gap but precede the next real phase-state event.
        merged = pd.merge_asof(
            det_subset,
            state_df,
            on="TS_start",
            direction="backward",
        )

        to_drop = merged.loc[merged["_state"] == status, "_orig_idx"].tolist()
        drop_idx.update(to_drop)

    return df_det.drop(index=list(drop_idx))


def _aggregate_by_cycle(df_det: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot detector counts per cycle.

    Args:
        df_det: Filtered Code-82 events with ``Cycle_start`` and ``ID`` columns.

    Returns:
        DataFrame with ``Cycle_start`` as index and detector IDs as columns.
    """
    result = (
        df_det.groupby(["Cycle_start", "ID"])["Code"]
        .count()
        .unstack(fill_value=0)
    )
    return result


def _aggregate_by_bin(df_det: pd.DataFrame, bin_len: int) -> pd.DataFrame:
    """
    Resample detector counts into fixed-width time bins.

    Args:
        df_det:  Filtered Code-82 events.
        bin_len: Bin width in minutes.

    Returns:
        DataFrame with time-bin start as index and detector IDs as columns.
    """
    result = (
        df_det.set_index("TS_start")
        .groupby("ID")["Code"]
        .resample(f"{bin_len}min")
        .count()
        .unstack(level="ID")
        .fillna(0)
        .astype(int)
    )
    return result


def _sum_movements(
    det_agg: pd.DataFrame,
    movements: Dict[str, List[int]],
    hourly_factor: float,
) -> pd.DataFrame:
    """
    Sum detector columns into movement totals, vectorised via matrix multiply.

    Args:
        det_agg:       Aggregated detector counts (rows=time/cycle, cols=det IDs).
        movements:     Mapping of movement label to list of detector IDs.
        hourly_factor: Scalar applied to all counts.

    Returns:
        DataFrame with movement columns plus ``TEV``, same index as *det_agg*.
    """
    if not movements:
        return pd.DataFrame(index=det_agg.index)

    available_dets = det_agg.columns.tolist()
    labels = list(movements.keys())

    mapping = pd.DataFrame(0, index=available_dets, columns=labels, dtype=float)
    for label, det_ids in movements.items():
        for det_id in det_ids:
            if det_id in mapping.index:
                mapping.loc[det_id, label] = 1.0

    mvmt_df = det_agg.reindex(columns=available_dets).fillna(0) @ mapping
    mvmt_df = mvmt_df * hourly_factor
    mvmt_df["TEV"] = mvmt_df[labels].sum(axis=1)

    return mvmt_df