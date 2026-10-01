"""Video/data sync from signal lamps (Imperative Shell).

Measures signal lamp pixel intensities from recorded video files, coordinates
database event retrieval, and invokes the functional alignment core.

Package Location: src/atspm/video/sync.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

from ..analysis.video_sync import LampSeries, SyncResult, align_clip
from ..data.reader import _resolve_timezone, get_events_with_cycles_df
from ..data.video import ShapeConfig, resolve_stopbar_target
from ..utils.timezone import localize_naive
from .processor import (
    _GAP_CODE,
    _OVERLAP_CODES,
    _PHASE_CODES,
    _open_capture,
    _pts_usable,
)

LAMP_DISC_RADIUS = 3


@dataclass
class LampMeasurement:
    """Per-frame measured intensity data extracted from a video clip.

    Attributes:
        frame_times_s: Relative frame timestamps starting at 0.0, strictly increasing.
        lamps: Sequence of LampSeries instances matching shape_config.lamp_shapes().
        fps: Effective frames per second.
        timing_source: 'pts' if presentation timestamps were used, otherwise 'fps'.
    """

    frame_times_s: np.ndarray
    lamps: List[LampSeries]
    fps: float
    timing_source: str


def measure_lamps(
    video_path: Union[str, Path],
    shape_config: ShapeConfig,
) -> LampMeasurement:
    """Measure mean BGR values inside each configured lamp ROI across all frames.

    Args:
        video_path: Path to the input video file.
        shape_config: ShapeConfig containing one or more lamp shapes.

    Returns:
        LampMeasurement with frame timestamps and measured LampSeries.

    Raises:
        ValueError: If shape_config has no lamp shapes, or resolution mismatches.
    """
    lamp_shapes = shape_config.lamp_shapes()
    if not lamp_shapes:
        raise ValueError("Shape configuration contains no lamp shapes.")

    video_path = Path(video_path)
    cap = _open_capture(video_path)

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        shape_config.validate_resolution(width, height)

        # Build boolean mask for each lamp shape
        masks: List[np.ndarray] = []
        for shape in lamp_shapes:
            mask = np.zeros((height, width), dtype=np.uint8)
            pts = shape["points"]
            if len(pts) == 1:
                cv2.circle(mask, tuple(pts[0]), LAMP_DISC_RADIUS, 255, -1)
            else:
                poly = np.array(pts, dtype=np.int32)
                cv2.fillPoly(mask, [poly], 255)
            masks.append(mask > 0)

        # Decode frames and measure mean BGR
        lamp_bgr_lists: List[List[np.ndarray]] = [[] for _ in lamp_shapes]
        pos_msec: List[float] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pos_msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            for l_idx, m_bool in enumerate(masks):
                mean_bgr = frame[m_bool].mean(axis=0)
                lamp_bgr_lists[l_idx].append(mean_bgr)
    finally:
        cap.release()

    n_frames = len(pos_msec)
    if _pts_usable(pos_msec):
        times = np.asarray(pos_msec, dtype=float) / 1000.0
        frame_times_s = times - times[0]
        timing_source = "pts"
        diffs = np.diff(frame_times_s)
        fps = float(1.0 / np.median(diffs)) if len(diffs) > 0 and np.median(diffs) > 0 else nominal_fps
    else:
        frame_times_s = np.arange(n_frames, dtype=float) / nominal_fps
        timing_source = "fps"
        fps = nominal_fps

    lamp_series_list: List[LampSeries] = []
    for l_idx, shape in enumerate(lamp_shapes):
        kind, number = resolve_stopbar_target(shape["phase"])
        bgr_arr = (
            np.asarray(lamp_bgr_lists[l_idx], dtype=np.float32)
            if lamp_bgr_lists[l_idx]
            else np.empty((0, 3), dtype=np.float32)
        )
        lamp_series_list.append(
            LampSeries(
                kind=kind,
                number=number,
                indication=shape["indication"],
                bgr=bgr_arr,
            )
        )

    return LampMeasurement(
        frame_times_s=frame_times_s,
        lamps=lamp_series_list,
        fps=fps,
        timing_source=timing_source,
    )


def sync_video(
    db_path: Union[str, Path],
    shape_config: ShapeConfig,
    video_path: Union[str, Path],
    start_guess: datetime,
    search_s: float = 30.0,
) -> SyncResult:
    """Synchronize video to database controller phase and overlap events.

    Args:
        db_path: Path to the intersection SQLite database.
        shape_config: Config containing calibrated lamp shapes.
        video_path: Path to the video clip file.
        start_guess: Estimated timestamp of video start.
        search_s: Half-window search duration in seconds.

    Returns:
        SyncResult computed by the functional alignment core.
    """
    db_path = Path(db_path)
    localized_guess = localize_naive(start_guess, _resolve_timezone(db_path))

    measurement = measure_lamps(video_path, shape_config)

    clip_duration = (
        float(measurement.frame_times_s[-1] - measurement.frame_times_s[0])
        if len(measurement.frame_times_s) > 0
        else 0.0
    )

    fetch_start = localized_guess - timedelta(seconds=search_s + 600.0)
    fetch_end = localized_guess + timedelta(seconds=clip_duration + search_s + 600.0)

    event_codes = [_GAP_CODE] + list(_PHASE_CODES) + list(_OVERLAP_CODES)
    events_df = get_events_with_cycles_df(
        db_path,
        fetch_start,
        fetch_end,
        event_codes=event_codes,
    )

    start_guess_epoch = localized_guess.timestamp()
    return align_clip(
        measurement.frame_times_s,
        measurement.lamps,
        events_df,
        start_guess_epoch,
        search_s=search_s,
        fps=measurement.fps,
    )
