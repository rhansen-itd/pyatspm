"""Video/data sync from signal lamps (Functional Core).

Analyzes measured signal lamp intensities across a recorded video, converts
them to debounced boolean on/off series, correlates against database signal phase
and overlap state intervals across candidate start times, and refines the optimal
start time and clock slip via sub-frame edge matching.

Pure functions and dataclasses only. No file I/O, no database queries, no OpenCV.

Package Location: src/atspm/analysis/video_sync.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .video import overlap_status_at_timestamps, phase_status_at_timestamps

INDICATIONS = ("green", "yellow", "red")
DEBOUNCE_SECONDS = 0.25

_TARGET_LETTERS = {
    "green": "G",
    "yellow": "Y",
    "red": "R",
}


@dataclass(frozen=True)
class LampSeries:
    """Per-frame measured BGR intensity values for a single signal lamp ROI.

    Attributes:
        kind: Target type, either 'phase' or 'overlap'.
        number: Phase or overlap number (e.g. 2 for phase 2, 2 for OLB).
        indication: One of 'green', 'yellow', or 'red'.
        bgr: 2-D float array of shape (n_frames, 3) containing mean B, G, R.
    """

    kind: str
    number: int
    indication: str
    bgr: np.ndarray


@dataclass(frozen=True)
class SyncResult:
    """Outcome of video/data temporal synchronization.

    Attributes:
        accepted: True if all acceptance criteria are satisfied, False otherwise.
        reason: None if accepted; short human-readable refusal reason if refused.
        start_epoch: Synchronized epoch timestamp for frame time 0.0.
        mid_start_epoch: Start timestamp that makes the clip-midpoint frame exact.
        slip_s_per_10min: Extrapolated clock drift in seconds per 10 minutes.
        score: Peak combined Matthews correlation score.
        runner_up_score: Score of the highest local maximum > 2.0s from the peak.
        agreement: Fraction of valid frame-lamp pairs matching the database at the peak.
        n_frames_compared: Valid frame-lamp pairs divided by number of lamps.
        n_edges_used: Number of inlier sub-frame transition edges used in line fitting.
        gap_clamped: True if data gap markers forced clamping to a single segment.
    """

    accepted: bool
    reason: Optional[str]
    start_epoch: float
    mid_start_epoch: float
    slip_s_per_10min: float
    score: float
    runner_up_score: float
    agreement: float
    n_frames_compared: int
    n_edges_used: int
    gap_clamped: bool


def lamp_contrast(bgr: np.ndarray, indication: str) -> np.ndarray:
    """Compute indication-specific color contrast from BGR values.

    Args:
        bgr: Array of shape (..., 3) with blue, green, and red values.
        indication: One of 'green', 'yellow', or 'red'.

    Returns:
        1-D or N-D array of contrast values.

    Raises:
        ValueError: If indication is not recognised.
    """
    arr = np.asarray(bgr, dtype=float)
    if indication == "green":
        return arr[..., 1] - arr[..., 2]
    if indication == "red":
        return arr[..., 2] - arr[..., 1]
    if indication == "yellow":
        return (arr[..., 2] + arr[..., 1]) / 2.0 - arr[..., 0]
    raise ValueError(
        f"Unrecognised indication {indication!r}: expected one of {INDICATIONS}."
    )


def otsu_threshold(values: np.ndarray) -> float:
    """Find the threshold separating two modes by maximizing between-class variance.

    Args:
        values: Numerical array of contrast samples.

    Returns:
        Right edge of the last bin in the lower class, or nan if constant or empty.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")

    min_val, max_val = float(np.min(v)), float(np.max(v))
    if min_val == max_val:
        return float("nan")

    counts, bin_edges = np.histogram(v, bins=256)
    p = counts.astype(float)
    total = p.sum()
    if total == 0:
        return float("nan")

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(p)
    weight2 = total - weight1
    cum_mean = np.cumsum(p * bin_centers)
    total_mean = cum_mean[-1]

    w1 = weight1[:-1]
    w2 = weight2[:-1]
    valid = (w1 > 0) & (w2 > 0)
    if not np.any(valid):
        return float("nan")

    mean1 = np.zeros_like(w1)
    mean2 = np.zeros_like(w2)
    mean1[valid] = cum_mean[:-1][valid] / w1[valid]
    mean2[valid] = (total_mean - cum_mean[:-1][valid]) / w2[valid]

    variance = np.zeros_like(w1)
    variance[valid] = w1[valid] * w2[valid] * ((mean1[valid] - mean2[valid]) ** 2)

    best_split = int(np.argmax(variance))
    return float(bin_edges[best_split + 1])


def debounce_frames(fps: float) -> int:
    """Number of consecutive frames needed to count as a persistent state.

    Args:
        fps: Effective frame rate in frames per second.

    Returns:
        Minimum run length in whole frames (at least 1).
    """
    return max(1, math.ceil(DEBOUNCE_SECONDS * fps))


def lamp_on_series(
    bgr: np.ndarray,
    indication: str,
    fps: float,
) -> Tuple[np.ndarray, float]:
    """Convert raw lamp BGR values into a debounced boolean on/off series.

    Args:
        bgr: Array of shape (n_frames, 3) with BGR measurements.
        indication: One of 'green', 'yellow', or 'red'.
        fps: Effective frame rate in frames per second.

    Returns:
        Tuple of (boolean on-series array, Otsu contrast threshold).
    """
    contrast = lamp_contrast(bgr, indication)
    threshold = otsu_threshold(contrast)
    if np.isnan(threshold):
        return np.zeros(len(bgr), dtype=bool), threshold

    on = (contrast > threshold).astype(bool)
    min_len = debounce_frames(fps)

    # Debounce interior runs shorter than min_len (up to 3 passes)
    for _ in range(3):
        if len(on) <= 2:
            break
        changes = np.flatnonzero(on[1:] != on[:-1]) + 1
        if len(changes) == 0:
            break
        starts = np.r_[0, changes]
        ends = np.r_[changes, len(on)]
        lengths = ends - starts

        interior_mask = np.zeros(len(starts), dtype=bool)
        interior_mask[1:-1] = lengths[1:-1] < min_len
        if not np.any(interior_mask):
            break
        for idx in np.flatnonzero(interior_mask):
            on[starts[idx]:ends[idx]] = ~on[starts[idx]:ends[idx]]

    return on, threshold


def align_clip(
    frame_times_s: np.ndarray,
    lamps: List[LampSeries],
    events_df: pd.DataFrame,
    start_guess_epoch: float,
    *,
    search_s: float = 30.0,
    fps: Optional[float] = None,
    min_score: float = 0.6,
    min_margin: float = 0.2,
    min_edges: int = 6,
    min_coverage: float = 0.5,
) -> SyncResult:
    """Find the optimal clip start time matching lamp measurements to controller events.

    Args:
        frame_times_s: Strictly increasing frame times in seconds starting at 0.0.
        lamps: Sequence of measured LampSeries instances for the clip.
        events_df: Flat controller events DataFrame.
        start_guess_epoch: Approximate video start epoch in seconds.
        search_s: Search half-window around start_guess_epoch in seconds.
        fps: Optional frame rate; derived from frame_times_s spacing if omitted.
        min_score: Minimum peak Matthews correlation score for acceptance.
        min_margin: Minimum margin between peak score and runner-up score.
        min_edges: Minimum sub-frame transition edges required.
        min_coverage: Minimum fraction of clip frames compared to database.

    Returns:
        SyncResult containing alignment parameters and acceptance status.
    """
    frame_times = np.asarray(frame_times_s, dtype=float)
    n_frames = len(frame_times)
    default_guess = float(start_guess_epoch)

    if n_frames == 0 or len(lamps) == 0:
        return SyncResult(
            accepted=False,
            reason="Insufficient data: empty frames or lamp series.",
            start_epoch=default_guess,
            mid_start_epoch=default_guess,
            slip_s_per_10min=0.0,
            score=0.0,
            runner_up_score=0.0,
            agreement=0.0,
            n_frames_compared=0,
            n_edges_used=0,
            gap_clamped=False,
        )

    if fps is None or fps <= 0:
        diffs = np.diff(frame_times)
        med_diff = float(np.median(diffs)) if len(diffs) > 0 else 0.1
        effective_fps = 1.0 / med_diff if med_diff > 0 else 10.0
    else:
        effective_fps = float(fps)

    # A.5 Gap-marker clamp
    w0 = default_guess - search_s
    clip_span = frame_times[-1] - frame_times[0]
    w1 = default_guess + clip_span + search_s

    has_gap_info = "event_code" in events_df.columns and "timestamp" in events_df.columns
    gap_rows = events_df.loc[events_df["event_code"] == -1] if has_gap_info else pd.DataFrame()
    markers_in_w = (
        gap_rows.loc[
            (gap_rows["timestamp"] > w0) & (gap_rows["timestamp"] < w1), "timestamp"
        ].sort_values().to_numpy()
        if not gap_rows.empty
        else np.array([])
    )

    if len(markers_in_w) == 0:
        gap_clamped = False
        clamped_events = events_df
    else:
        gap_clamped = True
        split_points = np.r_[w0, markers_in_w, w1]
        piece_lengths = split_points[1:] - split_points[:-1]
        longest_idx = int(np.argmax(piece_lengths))
        piece_start = split_points[longest_idx]
        piece_end = split_points[longest_idx + 1]

        all_markers = gap_rows["timestamp"].sort_values().to_numpy()
        markers_before = all_markers[all_markers <= piece_start]
        markers_after = all_markers[all_markers >= piece_end]

        row_mask = pd.Series(True, index=events_df.index)
        if len(markers_before) > 0:
            row_mask &= (events_df["timestamp"] >= markers_before[-1])
        if len(markers_after) > 0:
            row_mask &= (events_df["timestamp"] <= markers_after[0])

        clamped_events = events_df.loc[row_mask].copy()

    # A.6 Coarse search candidate starts
    step = 0.5 / effective_fps
    candidates = default_guess + np.arange(-search_s, search_s + step / 2.0, step)
    num_candidates = len(candidates)

    query_grid = candidates[:, None] + frame_times[None, :]
    query_flat = query_grid.ravel()

    lamp_on_list = []
    scores_per_lamp = []
    status_grid_list = []

    for lamp in lamps:
        on_series, _ = lamp_on_series(lamp.bgr, lamp.indication, effective_fps)
        lamp_on_list.append(on_series)
        tgt = _TARGET_LETTERS.get(lamp.indication, "G")

        if clamped_events.empty:
            status_grid = np.full((num_candidates, n_frames), "na", dtype=object)
        else:
            if lamp.kind == "phase":
                st = phase_status_at_timestamps(clamped_events, lamp.number, query_flat)
            elif lamp.kind == "overlap":
                st = overlap_status_at_timestamps(clamped_events, lamp.number, query_flat)
            else:
                st = np.full(len(query_flat), "na", dtype=object)
            status_grid = st.reshape(num_candidates, n_frames)

        status_grid_list.append(status_grid)

        # Matthews correlation (phi coefficient) across all candidates
        is_valid = (status_grid != "na")
        db_on = (status_grid == tgt)
        m_counts = is_valid.sum(axis=1)
        x_grid = np.broadcast_to(on_series, (num_candidates, n_frames)) & is_valid
        y_grid = db_on
        tp = (x_grid & y_grid).sum(axis=1)
        n_x = x_grid.sum(axis=1)
        n_y = y_grid.sum(axis=1)

        numerator = m_counts * tp - n_x * n_y
        denom_sq = n_x * (m_counts - n_x) * n_y * (m_counts - n_y)
        valid_denom = (denom_sq > 0) & (m_counts > 0)

        phi = np.zeros(num_candidates, dtype=float)
        phi[valid_denom] = numerator[valid_denom] / np.sqrt(
            denom_sq[valid_denom].astype(float)
        )
        scores_per_lamp.append(phi)

    combined_score = (
        np.mean(scores_per_lamp, axis=0)
        if scores_per_lamp
        else np.zeros(num_candidates, dtype=float)
    )
    peak_idx = int(np.argmax(combined_score))
    peak_score = float(combined_score[peak_idx])
    s0 = float(candidates[peak_idx])

    # Runner-up: highest local maximum lying strictly > 2.0s from coarse peak
    window_radius = int(math.floor(2.0 / step + 1e-9))
    local_maxima = []
    for i in range(num_candidates):
        if abs(candidates[i] - candidates[peak_idx]) > 2.0:
            left_bound = max(0, i - window_radius)
            right_bound = min(num_candidates, i + window_radius + 1)
            if combined_score[i] >= np.max(combined_score[left_bound:right_bound]):
                local_maxima.append(combined_score[i])
    runner_up_score = float(max(local_maxima)) if local_maxima else 0.0

    # Agreement and valid frames compared at peak candidate
    total_valid = 0
    total_matching = 0
    for l_idx, lamp in enumerate(lamps):
        st_peak = status_grid_list[l_idx][peak_idx]
        val_peak = (st_peak != "na")
        tgt = _TARGET_LETTERS.get(lamp.indication, "G")
        db_peak = (st_peak == tgt)
        match_peak = (lamp_on_list[l_idx] == db_peak) & val_peak
        total_valid += int(np.sum(val_peak))
        total_matching += int(np.sum(match_peak))

    agreement = float(total_matching / total_valid) if total_valid > 0 else 0.0
    n_frames_compared = int(total_valid // len(lamps)) if len(lamps) > 0 else 0

    # A.7 Sub-frame refinement and slip
    g_grid = np.round(np.arange(-1.0, 1.0 + 1e-5, 0.01), 2)
    residuals_pool: List[float] = []
    t_edges_pool: List[float] = []

    for l_idx, lamp in enumerate(lamps):
        on_s = lamp_on_list[l_idx]
        edge_indices = np.flatnonzero(on_s[1:] != on_s[:-1])
        if len(edge_indices) == 0 or clamped_events.empty:
            continue
        t_edges = (frame_times[edge_indices] + frame_times[edge_indices + 1]) / 2.0
        polarities = on_s[edge_indices + 1]

        sample_ts = s0 + t_edges[:, None] + g_grid[None, :]
        tgt = _TARGET_LETTERS.get(lamp.indication, "G")
        if lamp.kind == "phase":
            st_edges = phase_status_at_timestamps(
                clamped_events, lamp.number, sample_ts.ravel()
            ).reshape(len(t_edges), len(g_grid))
        elif lamp.kind == "overlap":
            st_edges = overlap_status_at_timestamps(
                clamped_events, lamp.number, sample_ts.ravel()
            ).reshape(len(t_edges), len(g_grid))
        else:
            continue

        val_edges = (st_edges != "na")
        db_on_edges = (st_edges == tgt)

        both_val = val_edges[:, :-1] & val_edges[:, 1:]
        changed = (db_on_edges[:, :-1] != db_on_edges[:, 1:])
        same_polarity = (db_on_edges[:, 1:] == polarities[:, None])
        matches = both_val & changed & same_polarity

        midpoints = (g_grid[:-1] + g_grid[1:]) / 2.0
        for i in range(len(t_edges)):
            m_indices = np.flatnonzero(matches[i])
            if len(m_indices) > 0:
                best_m = m_indices[np.argmin(np.abs(midpoints[m_indices]))]
                residuals_pool.append(float(midpoints[best_m]))
                t_edges_pool.append(float(t_edges[i]))

    residuals_arr = np.array(residuals_pool, dtype=float)
    t_edges_arr = np.array(t_edges_pool, dtype=float)

    if len(residuals_arr) > 0:
        med_r = np.median(residuals_arr)
        inlier_mask = np.abs(residuals_arr - med_r) < 0.3
        r_inliers = residuals_arr[inlier_mask]
        t_inliers = t_edges_arr[inlier_mask]
    else:
        r_inliers = np.array([], dtype=float)
        t_inliers = np.array([], dtype=float)

    n_edges_used = len(r_inliers)

    if n_edges_used >= 2:
        var_t = float(np.var(t_inliers))
        if var_t > 1e-12:
            cov_tr = float(
                np.mean((t_inliers - np.mean(t_inliers)) * (r_inliers - np.mean(r_inliers)))
            )
            b = cov_tr / var_t
            a = float(np.mean(r_inliers) - b * np.mean(t_inliers))
        else:
            b = 0.0
            a = float(np.mean(r_inliers))
    elif n_edges_used == 1:
        b = 0.0
        a = float(r_inliers[0])
    else:
        b = 0.0
        a = 0.0

    start_epoch = s0 + a
    slip_s_per_10min = 600.0 * b
    mid_start_epoch = start_epoch + b * (
        frame_times[0] + (frame_times[-1] - frame_times[0]) / 2.0
    )

    # A.8 Acceptance evaluation
    coverage = (n_frames_compared / n_frames) if n_frames > 0 else 0.0
    accepted = True
    reason = None

    if coverage < min_coverage:
        accepted = False
        reason = f"Coverage {coverage:.1%} below minimum {min_coverage:.1%}."
    elif peak_score < min_score:
        accepted = False
        reason = f"Score {peak_score:.3f} below minimum {min_score:.3f}."
    elif peak_score - runner_up_score < min_margin:
        accepted = False
        reason = (
            f"Margin {peak_score - runner_up_score:.3f} below minimum {min_margin:.3f}."
        )
    elif n_edges_used < min_edges:
        accepted = False
        reason = f"Edges used {n_edges_used} below minimum {min_edges}."

    return SyncResult(
        accepted=accepted,
        reason=reason,
        start_epoch=start_epoch,
        mid_start_epoch=mid_start_epoch,
        slip_s_per_10min=slip_s_per_10min,
        score=peak_score,
        runner_up_score=runner_up_score,
        agreement=agreement,
        n_frames_compared=n_frames_compared,
        n_edges_used=n_edges_used,
        gap_clamped=gap_clamped,
    )
