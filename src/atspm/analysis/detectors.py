"""
Co-Located Detector Discrepancy Analysis (Functional Core)

Pure functions for post-hoc identification of disagreements between two
detectors covering the same physical zone (e.g., Radar vs. Video).

Package Location: src/atspm/analysis/detectors.py
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reconstruct_intervals(events_df: pd.DataFrame, det_id: int) -> pd.DataFrame:
    """Build a DataFrame of ON intervals for a single detector.

    Each row represents a continuous ON period derived from Code-82 (ON) /
    Code-81 (OFF) events.  Gap markers (event_code == -1) act as hard resets
    and force any open interval to close immediately.

    Args:
        events_df: Raw events with columns ['timestamp', 'event_code',
            'parameter']. Must already be sorted by timestamp ascending.
        det_id: The detector parameter value to filter on.

    Returns:
        DataFrame with columns ['on_ts', 'off_ts', 'duration_sec'],
        sorted by on_ts. Returns an empty DataFrame if no ON events exist.
    """
    mask = (events_df["parameter"] == det_id) | (events_df["event_code"] == -1)
    det = events_df.loc[mask].copy().sort_values("timestamp").reset_index(drop=True)

    intervals: list[dict] = []
    open_ts: float | None = None

    for _, row in det.iterrows():
        code = int(row["event_code"])

        if code == -1:                          # Hard reset — close any open interval
            if open_ts is not None:
                intervals.append({"on_ts": open_ts, "off_ts": row["timestamp"]})
                open_ts = None
            continue

        if int(row["parameter"]) != det_id:
            continue

        if code == 82:                          # ON
            if open_ts is None:
                open_ts = float(row["timestamp"])

        elif code == 81:                        # OFF
            if open_ts is not None:
                intervals.append({"on_ts": open_ts, "off_ts": float(row["timestamp"])})
                open_ts = None

    if not intervals:
        return pd.DataFrame(columns=["on_ts", "off_ts", "duration_sec"])

    df = pd.DataFrame(intervals)
    df["duration_sec"] = df["off_ts"] - df["on_ts"]
    return df.sort_values("on_ts").reset_index(drop=True)


def _build_state_array(intervals_df: pd.DataFrame, query_ts: np.ndarray) -> np.ndarray:
    """Map detector ON/OFF state onto a sorted array of query timestamps.

    Uses vectorised interval containment: a query timestamp is ON if it
    falls within any [on_ts, off_ts) interval.

    Args:
        intervals_df: Output of _reconstruct_intervals.
        query_ts: Sorted 1-D float array of epoch seconds to evaluate.

    Returns:
        Boolean ndarray of the same length as query_ts.
    """
    state = np.zeros(len(query_ts), dtype=bool)
    if intervals_df.empty:
        return state

    on_arr = intervals_df["on_ts"].values
    off_arr = intervals_df["off_ts"].values

    for on_t, off_t in zip(on_arr, off_arr):
        lo = np.searchsorted(query_ts, on_t, side="left")
        hi = np.searchsorted(query_ts, off_t, side="left")
        state[lo:hi] = True

    return state


def _analyze_pair(
    events_df: pd.DataFrame,
    phase: int,
    det_a_id: int,
    det_b_id: int,
    lag_threshold_sec: float,
) -> list[dict]:
    """Core discrepancy logic for a single detector pair.

    Applies three classification rules:

    * **Rule 1 – Extended Disagreement**: One detector is ON while the other
      is OFF, and that mismatch persists continuously for more than
      ``lag_threshold_sec`` seconds.
    * **Rule 2 – Isolated Pulse**: One detector fires ON→OFF (duration
      ``< lag_threshold_sec``) while the other was completely OFF throughout
      the expanded window ``[pulse_on − threshold, pulse_off + threshold]``.
    * **Rule 3 – Chatter Exception** (implicit): Rapid ON→OFF pulses on one
      detector are *not* flagged when the partner was active during the
      expanded window (indicating real vehicle presence, not a false pulse).

    Gap markers (``event_code == -1``) act as hard resets: no anomaly is ever
    reported across a gap boundary.

    Args:
        events_df: Raw ATSPM events, pre-filtered to the query window and
            sorted by timestamp.
        phase: Signal phase number associated with this pair (echoed in output).
        det_a_id: parameter value identifying Detector A.
        det_b_id: parameter value identifying Detector B.
        lag_threshold_sec: Minimum disagreement duration (seconds) to raise a
            Rule-1 anomaly; also the half-window for the Rule-2 pulse check.

    Returns:
        List of anomaly dicts (may be empty).  Each dict includes an
        ``on_det_id`` key for ``extended_disagreement`` rows, indicating which
        detector was stuck ON during the window.
    """
    intervals_a = _reconstruct_intervals(events_df, det_a_id)
    intervals_b = _reconstruct_intervals(events_df, det_b_id)

    if intervals_a.empty and intervals_b.empty:
        return []

    gap_ts = events_df.loc[events_df["event_code"] == -1, "timestamp"].values

    ts_a = (
        np.concatenate([intervals_a["on_ts"].values, intervals_a["off_ts"].values])
        if not intervals_a.empty else np.array([], dtype=float)
    )
    ts_b = (
        np.concatenate([intervals_b["on_ts"].values, intervals_b["off_ts"].values])
        if not intervals_b.empty else np.array([], dtype=float)
    )

    change_points = np.unique(np.concatenate([ts_a, ts_b, gap_ts]))
    change_points.sort()

    if len(change_points) == 0:
        return []

    next_cp = np.empty_like(change_points)
    next_cp[:-1] = change_points[1:]
    next_cp[-1] = change_points[-1] + 1.0
    mid_ts = (change_points + next_cp) / 2.0

    state_a = _build_state_array(intervals_a, mid_ts)
    state_b = _build_state_array(intervals_b, mid_ts)

    mismatch = state_a != state_b
    transitions = np.diff(mismatch.astype(np.int8), prepend=mismatch[0].astype(np.int8)) != 0
    segment_id = np.cumsum(transitions)

    anomalies: list[dict] = []

    # ------------------------------------------------------------------
    # Rule 1 – Extended Disagreement
    # ------------------------------------------------------------------
    for seg in np.unique(segment_id[mismatch]):
        indices = np.where((segment_id == seg) & mismatch)[0]

        seg_start = change_points[indices[0]]
        end_idx = indices[-1] + 1
        seg_end = (
            change_points[end_idx] if end_idx < len(change_points)
            else change_points[-1] + 1.0
        )
        duration = seg_end - seg_start

        if gap_ts.size and np.any((gap_ts >= seg_start) & (gap_ts <= seg_end)):
            continue
        if duration <= lag_threshold_sec:
            continue

        # on_det_id: the detector that was ON during the disagreement window.
        # This is captured in the anomaly record so the plotting layer can
        # render directional colour coding without re-parsing description text.
        on_det_id  = det_a_id if state_a[indices[0]] else det_b_id
        off_det_id = det_b_id if on_det_id == det_a_id else det_a_id

        anomalies.append({
            "phase":           phase,
            "anomaly_type":    "extended_disagreement",
            "start_timestamp": float(seg_start),
            "end_timestamp":   float(seg_end),
            "duration_sec":    float(duration),
            "det_a_id":        det_a_id,
            "det_b_id":        det_b_id,
            "on_det_id":       on_det_id,   # NEW: which detector was ON
            "description": (
                f"Ph{phase}: Det {on_det_id} ON / Det {off_det_id} OFF for "
                f"{duration:.2f}s (threshold={lag_threshold_sec}s)"
            ),
        })

    # ------------------------------------------------------------------
    # Rule 2 – Isolated Pulse
    #    Evaluate each detector's short pulses independently.
    # ------------------------------------------------------------------
    for src_ivs, src_id, oth_ivs in (
        (intervals_a, det_a_id, intervals_b),
        (intervals_b, det_b_id, intervals_a),
    ):
        if src_ivs.empty:
            continue

        short_pulses = src_ivs[src_ivs["duration_sec"] < lag_threshold_sec]
        if short_pulses.empty:
            continue

        p_on   = short_pulses["on_ts"].values
        p_off  = short_pulses["off_ts"].values
        w_start = p_on  - lag_threshold_sec
        w_end   = p_off + lag_threshold_sec
        other_id = det_b_id if src_id == det_a_id else det_a_id

        for i in range(len(short_pulses)):
            ws, we = w_start[i], w_end[i]

            if gap_ts.size and np.any((gap_ts >= ws) & (gap_ts <= we)):
                continue

            # Chatter exception: partner active in window → not an isolated pulse
            if not oth_ivs.empty:
                overlaps = (oth_ivs["on_ts"].values < we) & (oth_ivs["off_ts"].values > ws)
                if overlaps.any():
                    continue

            dur = float(p_off[i] - p_on[i])
            anomalies.append({
                "phase":           phase,
                "anomaly_type":    "isolated_pulse",
                "start_timestamp": float(p_on[i]),
                "end_timestamp":   float(p_off[i]),
                "duration_sec":    dur,
                "det_a_id":        det_a_id,
                "det_b_id":        det_b_id,
                "on_det_id":       src_id,   # the detector that fired the pulse
                "description": (
                    f"Ph{phase}: Det {src_id} isolated pulse ON/OFF for {dur:.3f}s; "
                    f"Det {other_id} was completely OFF in ±{lag_threshold_sec}s window"
                ),
            })

    return anomalies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_discrepancies(
    events_df: pd.DataFrame,
    detector_pairs: List[Dict[str, int]],
    lag_threshold_sec: float = 2.0,
) -> pd.DataFrame:
    """Identify disagreements across one or more co-located detector pairs.

    Applies three classification rules to historical ATSPM event data for
    every ``(det_a, det_b)`` pair supplied:

    * **Rule 1 – Extended Disagreement**: One detector is ON while the other
      is OFF, and that exact mismatch persists continuously for more than
      ``lag_threshold_sec`` seconds.
    * **Rule 2 – Isolated Pulse**: One detector fires ON→OFF (duration
      ``< lag_threshold_sec``) while the partner was completely OFF throughout
      the expanded window ``[pulse_on − threshold, pulse_off + threshold]``.
    * **Rule 3 – Chatter Exception** (implicit): When the partner detector
      was active during the expanded window, the short pulse is *not* flagged
      as anomalous (real vehicle presence, not a false detection).

    Gap markers (``event_code == -1``) are treated as hard resets: any open
    interval is closed at the reset, and no anomaly is ever reported across a
    gap boundary.

    Args:
        events_df: Raw ATSPM events with columns
            ['timestamp', 'event_code', 'parameter'].
            Expected codes: 82 = detector ON, 81 = detector OFF,
            -1 = gap marker.
        detector_pairs: List of ``{"phase": int, "det_a": int, "det_b": int}``
            dicts.  Typically sourced from
            ``config["detector_pairs"]`` as returned by
            ``DatabaseManager.get_config_at_date``.
        lag_threshold_sec: Minimum disagreement duration (seconds) required
            to raise a Rule-1 anomaly. Also used as the half-window size for
            the Rule-2 pulse check. Defaults to 2.0.

    Returns:
        DataFrame of identified anomalies with columns:

            phase            – Signal phase the pair is associated with.
            anomaly_type     – ``'extended_disagreement'`` or
                               ``'isolated_pulse'``.
            start_timestamp  – Epoch float marking anomaly start.
            end_timestamp    – Epoch float marking anomaly end.
            duration_sec     – ``end_timestamp − start_timestamp``.
            det_a_id         – Detector A parameter value (echoed).
            det_b_id         – Detector B parameter value (echoed).
            on_det_id        – For ``extended_disagreement``: the detector
                               that was ON during the window.  For
                               ``isolated_pulse``: the detector that fired.
            description      – Human-readable summary string.

        Returns an empty DataFrame (same schema) when no anomalies are found
        or ``detector_pairs`` is empty.
    """
    _SCHEMA = [
        "phase", "anomaly_type", "start_timestamp", "end_timestamp",
        "duration_sec", "det_a_id", "det_b_id", "on_det_id", "description",
    ]
    _EMPTY = pd.DataFrame(columns=_SCHEMA)

    if events_df.empty or not detector_pairs:
        return _EMPTY

    events_df = events_df.sort_values("timestamp").reset_index(drop=True)

    all_anomalies: list[dict] = []
    for pair in detector_pairs:
        all_anomalies.extend(
            _analyze_pair(
                events_df=events_df,
                phase=pair["phase"],
                det_a_id=pair["det_a"],
                det_b_id=pair["det_b"],
                lag_threshold_sec=lag_threshold_sec,
            )
        )

    if not all_anomalies:
        return _EMPTY

    return (
        pd.DataFrame(all_anomalies, columns=_SCHEMA)
        .sort_values(["phase", "start_timestamp"])
        .reset_index(drop=True)
    )
