"""
Video Overlay — Point-in-Time Status Lookup (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.
Input/output is DataFrames, numpy arrays, and plain scalars.

Purpose
-------
Given a stream of arbitrary frame timestamps (from a recorded video), answer
"what was phase N / overlap N / detector N doing at timestamp T" for each
frame.  This is the data layer behind ``atspm.video``'s OpenCV overlay
rendering — it does not touch frames, shapes, or files.

Phase status reuses ``_build_phase_intervals`` from ``atspm.analysis.phases``
as-is; it already emits gap-isolated ``[green_ts, yellow_ts, clear_end_ts]``
intervals per cycle, which is exactly what a G/Y/R lookup needs.

Overlap status has no existing handler anywhere in this codebase — overlaps
are logged under dedicated event codes (61-66), not the phase codes with an
offset parameter.  ``_build_overlap_intervals`` below is a small, separate
state machine for those codes, deliberately emitting the same
``[green_ts, yellow_ts, clear_end_ts, dark_ts]`` shape so the same lookup
function (``_status_at_timestamps``) serves both.

Detector status is not reimplemented here at all — it directly reuses
``_reconstruct_intervals`` / ``_build_state_array`` from
``atspm.analysis.detectors``, which already do this for Code 81/82.

Event code reference (Indiana/Purdue Hi-Res Logger Enumerations):
    Phase Events:
        1   - Phase Begin Green
        8   - Phase Begin Yellow Clearance
        9   - Phase End Yellow Clearance
        10  - Phase Begin Red Clearance   (only if RC served)
        11  - Phase End Red Clearance     (only if RC served)
        12  - Phase Inactive
    Overlap Events (parameter = overlap # as number, A=1, B=2, ... P=16):
        61  - Overlap Begin Green
        63  - Overlap Begin Yellow
        64  - Overlap Begin Red Clearance (only if RC served)
        65  - Overlap Off (inactive, red indication still shown)
        66  - Overlap Dark (no active output -- renders as 'na', not red)
    Detector Events:
        81  - Detector Off
        82  - Detector On
    -1      - Data gap marker (hard reset)

Gap Marker Rule
---------------
Both interval builders already isolate intervals to a single contiguous
data segment (via ``_segment_id``), so no single G/Y interval can itself
straddle a gap.  The remaining risk is the *inferred* steady-red period
between one interval's ``clear_end_ts`` and the next interval's
``green_ts`` -- ``_status_at_timestamps`` only labels that gap ``'R'`` when
both intervals share the same segment id; otherwise (or past the last known
interval) it returns ``'na'`` rather than guessing through a discontinuity.

Package Location: src/atspm/analysis/video.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .aog import _ts_to_float
from .detectors import _build_state_array, _reconstruct_intervals
from .phases import _CODE_END_YELLOW, _CODE_YELLOW, _build_phase_intervals, _segment_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1

_PHASE_CODES = frozenset({1, 8, 9, 10, 11, 12})

_CODE_OL_GREEN    = 61
_CODE_OL_YELLOW   = 63
_CODE_OL_BEGIN_RC = 64
_CODE_OL_OFF      = 65
_CODE_OL_DARK     = 66
_OVERLAP_CODES = frozenset(
    {_CODE_OL_GREEN, _CODE_OL_YELLOW, _CODE_OL_BEGIN_RC, _CODE_OL_OFF, _CODE_OL_DARK}
)

_IDLE, _GREEN, _YELLOW, _RED_CLR = range(4)

_EMPTY_OVERLAP_INTERVALS = pd.DataFrame(
    columns=["overlap", "seg", "green_ts", "yellow_ts", "clear_end_ts", "dark_ts"]
)

# ---------------------------------------------------------------------------
# Overlap interval construction
# ---------------------------------------------------------------------------


def _build_overlap_intervals(events_df: pd.DataFrame, overlap_num: int) -> pd.DataFrame:
    """Convert sequential overlap state-change events (codes 61/63/64/65/66)
    into tidy G/Y/R/dark intervals for a single overlap number.

    Mirrors the shape of ``atspm.analysis.phases._build_phase_intervals``
    (``green_ts``, ``yellow_ts``, ``clear_end_ts``) plus an overlap-specific
    ``dark_ts`` boundary, so both can be consumed by the same
    ``_status_at_timestamps`` lookup.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter]``.  Gap markers
            (``event_code == -1``) must be present for discontinuity
            enforcement.
        overlap_num: Overlap number (``A=1, B=2, ... P=16``) -- the
            ``parameter`` value to filter on.

    Returns:
        DataFrame with columns
        ``[overlap, seg, green_ts, yellow_ts, clear_end_ts, dark_ts]``.
        ``dark_ts`` is ``None`` for intervals that ended via Code 65
        (Off) rather than Code 66 (Dark).  Returns an empty DataFrame
        with the correct schema when no valid intervals exist.
    """
    if events_df.empty:
        return _EMPTY_OVERLAP_INTERVALS

    mask = (
        (events_df["event_code"].isin(_OVERLAP_CODES) & (events_df["parameter"] == overlap_num))
        | (events_df["event_code"] == _GAP_CODE)
    )
    ol_df = events_df.loc[mask].copy().sort_values("timestamp").reset_index(drop=True)

    if ol_df.empty:
        return _EMPTY_OVERLAP_INTERVALS

    ol_df["_seg"] = _segment_id(ol_df)
    ol_df = ol_df.loc[ol_df["event_code"] != _GAP_CODE].copy()

    if ol_df.empty:
        return _EMPTY_OVERLAP_INTERVALS

    records: list[dict] = []

    for seg, grp in ol_df.groupby("_seg", sort=True):
        grp = grp.sort_values("timestamp")

        state = _IDLE
        green_ts = yellow_ts = clear_end = None

        def _emit(dark_ts=None):
            records.append(dict(
                overlap=overlap_num,
                seg=int(seg),
                green_ts=green_ts,
                yellow_ts=yellow_ts,
                clear_end_ts=clear_end,
                dark_ts=dark_ts,
            ))

        for row in grp.itertuples(index=False):
            code = row.event_code
            ts = row.timestamp

            if code == _CODE_OL_GREEN:
                green_ts, yellow_ts, clear_end = ts, None, None
                state = _GREEN

            elif code == _CODE_OL_YELLOW:
                if state == _GREEN:
                    yellow_ts = ts
                    clear_end = None
                    state = _YELLOW

            elif code == _CODE_OL_BEGIN_RC:
                if state == _YELLOW:
                    clear_end = ts
                    state = _RED_CLR

            elif code == _CODE_OL_OFF:
                # No Begin-RC was logged: yellow ends right here instead of
                # at a Code-64 boundary.
                if state == _YELLOW:
                    clear_end = ts
                if state in (_YELLOW, _RED_CLR):
                    _emit(None)
                state = _IDLE
                green_ts = None

            elif code == _CODE_OL_DARK:
                if state == _YELLOW:
                    clear_end = ts
                if state in (_YELLOW, _RED_CLR):
                    _emit(ts)
                state = _IDLE
                green_ts = None

    if not records:
        return _EMPTY_OVERLAP_INTERVALS

    df = pd.DataFrame(records)
    valid = df["yellow_ts"].notna()
    return df.loc[valid].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared point-in-time lookup
# ---------------------------------------------------------------------------


def _nullable_ts_to_float(series: pd.Series) -> np.ndarray:
    """Like ``aog._ts_to_float`` but tolerant of an all-``None``/``NaT`` column.

    ``aog._ts_to_float`` inspects ``series.iloc[0]`` to pick a dtype path,
    which assumes a non-null first element.  ``dark_ts`` is frequently
    entirely absent (phase intervals never set it; most overlap intervals
    end via Code 65, not Code 66), so a dedicated NaN-tolerant variant is
    used here instead of relaxing that helper's precondition for its other
    callers.

    Args:
        series: Column that may contain epoch floats, tz-aware
            ``Timestamp`` objects, ``None``, or ``NaT``.

    Returns:
        float64 ndarray of epoch seconds, with ``NaN`` for missing values.
    """
    if len(series) == 0:
        return np.array([], dtype=np.float64)

    sample = next((v for v in series if pd.notna(v)), None)
    if sample is None:
        return np.full(len(series), np.nan, dtype=np.float64)

    if hasattr(sample, "timestamp"):
        return np.array(
            [v.timestamp() if pd.notna(v) else np.nan for v in series],
            dtype=np.float64,
        )
    return pd.to_numeric(series, errors="coerce").astype(np.float64).to_numpy()


def _status_at_timestamps(intervals_df: pd.DataFrame, query_ts: np.ndarray) -> np.ndarray:
    """Vectorised G/Y/R/na lookup shared by phase and overlap intervals.

    Uses the same ``np.searchsorted`` interval-containment pattern as
    ``atspm.analysis.aog`` (cycle bucketing) and
    ``atspm.analysis.detectors`` (``_build_state_array``).

    Args:
        intervals_df: DataFrame with columns ``[green_ts, yellow_ts,
            clear_end_ts, seg]`` and optionally ``dark_ts``.  Either
            ``atspm.analysis.phases._build_phase_intervals``'s output
            (phase) or ``_build_overlap_intervals``'s output (overlap).
        query_ts: 1-D float64 array of epoch-second frame timestamps, any
            order.

    Returns:
        Object ndarray of the same length as ``query_ts`` with values
        ``'G'``, ``'Y'``, ``'R'``, or ``'na'``.  ``'na'`` covers: before
        the first known interval, after the last known interval, any
        period whose inferred steady-red gap crosses a segment boundary
        (Gap Marker Rule), and -- for overlaps -- any period at or after
        a Code-66 Dark event.
    """
    n = len(query_ts)
    result = np.full(n, "na", dtype=object)
    if intervals_df.empty or n == 0:
        return result

    df = intervals_df.sort_values("green_ts").reset_index(drop=True)

    green_f  = _ts_to_float(df["green_ts"])
    yellow_f = _ts_to_float(df["yellow_ts"])
    clear_f  = _ts_to_float(df["clear_end_ts"])
    dark_f   = _nullable_ts_to_float(df["dark_ts"]) if "dark_ts" in df.columns \
        else np.full(len(df), np.nan, dtype=np.float64)
    seg_arr  = df["seg"].to_numpy()

    idx = np.searchsorted(green_f, query_ts, side="right") - 1
    valid = idx >= 0
    if not valid.any():
        return result

    vi = idx[valid]
    vq = query_ts[valid]
    last = len(df) - 1
    next_idx = np.minimum(vi + 1, last)
    has_next = vi < last
    same_seg = has_next & (seg_arr[vi] == seg_arr[next_idx])

    is_green  = vq < yellow_f[vi]
    is_yellow = (~is_green) & (vq < clear_f[vi])
    is_dark   = (~is_green) & (~is_yellow) & (vq >= dark_f[vi])  # False when dark_f is NaN
    is_red    = (~is_green) & (~is_yellow) & (~is_dark) & same_seg & (vq < green_f[next_idx])

    labels = np.full(len(vq), "na", dtype=object)
    labels[is_green]  = "G"
    labels[is_yellow] = "Y"
    labels[is_red]    = "R"

    result[valid] = labels
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def phase_status_at_timestamps(
    events_df: pd.DataFrame,
    phase: int,
    query_ts: np.ndarray,
) -> np.ndarray:
    """Per-frame G/Y/R/na status for a single signal phase.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start]`` (the
            "Events-with-cycles exporter" shape -- ``cycle_start`` is
            required by ``_build_phase_intervals``).  Gap markers
            (``event_code == -1``) must be present.
        phase: Signal phase number.
        query_ts: 1-D float64 array of epoch-second frame timestamps.

    Returns:
        Object ndarray of ``'G'``/``'Y'``/``'R'``/``'na'``, same length and
        order as ``query_ts``.
    """
    if events_df.empty or len(query_ts) == 0:
        return np.full(len(query_ts), "na", dtype=object)

    mask = events_df["event_code"].isin(_PHASE_CODES | {_GAP_CODE})
    ph_df = events_df.loc[mask].copy().sort_values("timestamp").reset_index(drop=True)
    if ph_df.empty:
        return np.full(len(query_ts), "na", dtype=object)

    ph_df["_seg"] = _segment_id(ph_df)
    ph_df = ph_df.loc[ph_df["event_code"] != _GAP_CODE].copy()
    if ph_df.empty:
        return np.full(len(query_ts), "na", dtype=object)

    intervals = _build_phase_intervals(ph_df, include_no_clearance=False)
    intervals = intervals.loc[intervals["phase"] == phase] if not intervals.empty else intervals

    return _status_at_timestamps(intervals, query_ts)


def overlap_status_at_timestamps(
    events_df: pd.DataFrame,
    overlap_num: int,
    query_ts: np.ndarray,
) -> np.ndarray:
    """Per-frame G/Y/R/na status for a single overlap.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter]``.  Gap markers
            (``event_code == -1``) must be present.
        overlap_num: Overlap number (``A=1, B=2, ... P=16``).
        query_ts: 1-D float64 array of epoch-second frame timestamps.

    Returns:
        Object ndarray of ``'G'``/``'Y'``/``'R'``/``'na'``, same length and
        order as ``query_ts``.
    """
    if events_df.empty or len(query_ts) == 0:
        return np.full(len(query_ts), "na", dtype=object)

    intervals = _build_overlap_intervals(events_df, overlap_num)
    return _status_at_timestamps(intervals, query_ts)


def detector_status_at_timestamps(
    events_df: pd.DataFrame,
    det_id: int,
    query_ts: np.ndarray,
) -> np.ndarray:
    """Per-frame On/Off boolean status for a single detector.

    Directly reuses ``atspm.analysis.detectors._reconstruct_intervals`` and
    ``_build_state_array`` -- no new interval logic.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter]``.  Gap markers
            (``event_code == -1``) must be present.
        det_id: Detector channel number (``parameter`` value).
        query_ts: 1-D float64 array of epoch-second frame timestamps,
            **sorted ascending** (required by ``_build_state_array``).

    Returns:
        Boolean ndarray, same length and order as ``query_ts``. ``True``
        means the detector is On at that timestamp.
    """
    if events_df.empty or len(query_ts) == 0:
        return np.zeros(len(query_ts), dtype=bool)

    intervals = _reconstruct_intervals(events_df, det_id)
    return _build_state_array(intervals, query_ts)


_TRANSITION_CODES = {
    "green_to_yellow": _CODE_YELLOW,
    "yellow_to_red": _CODE_END_YELLOW,
}


def first_phase_transition_after(
    events_df: pd.DataFrame,
    phase: int,
    after_ts: float,
    transition: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    """Earliest visual color change for *phase* at or after *after_ts*.

    ``"yellow_to_red"`` resolves to Code 9 (Phase End Yellow Clearance)
    rather than Code 10/11 (Begin/End Red Clearance) -- Code 9 fires at the
    exact instant yellow ends, which is also the exact instant steady red
    begins whether or not red clearance is served for this phase, so it is
    the correct single code for the visual edge regardless of timing plan.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter]``.
        phase: Signal phase number.
        after_ts: Epoch-second timestamp; only events at or after this
            instant are considered.
        transition: Restrict to ``"green_to_yellow"`` or ``"yellow_to_red"``.
            When ``None``, both are considered and whichever occurs first
            wins -- useful for auto-selecting which edge to use for
            alignment without the caller having to know in advance.

    Returns:
        ``(transition, timestamp)`` for the earliest match, or ``None`` if
        *events_df* has no matching event for this phase at or after
        *after_ts*.

    Raises:
        ValueError: If *transition* is given but isn't a recognised edge.
    """
    if transition is not None and transition not in _TRANSITION_CODES:
        raise ValueError(
            f"Unrecognised transition {transition!r}: expected one of "
            f"{sorted(_TRANSITION_CODES)}."
        )
    codes = [_TRANSITION_CODES[transition]] if transition else list(_TRANSITION_CODES.values())
    code_to_transition = {v: k for k, v in _TRANSITION_CODES.items()}

    matches = events_df.loc[
        events_df["event_code"].isin(codes)
        & (events_df["parameter"] == phase)
        & (events_df["timestamp"] >= after_ts)
    ]
    if matches.empty:
        return None
    row = matches.loc[matches["timestamp"].idxmin()]
    return code_to_transition[row["event_code"]], float(row["timestamp"])
