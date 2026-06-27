"""
Video Overlay Processor (Imperative Shell)

Orchestrates rendering a recorded intersection video with loop/stopbar
shapes recolored by phase, overlap, and detector status pulled from the
SQLite ``events``/``cycles`` tables.  All DB access and OpenCV file I/O
lives here; the actual status math is delegated to
``atspm.analysis.video`` and the actual pixel drawing to
``atspm.video.overlay``.

Uses the *correct* at-or-before status semantics throughout (a frame shows
whatever was true at-or-before its timestamp).  The legacy
``spmfunctions.video_processing.load_and_process_data``'s
``pd.merge_asof(..., direction='forward')`` bug -- which looks up the
*next* event instead of the most recent one -- has no equivalent here;
``atspm.analysis.video``'s ``np.searchsorted`` lookups are at-or-before by
construction.

Package Location: src/atspm/video/processor.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np

from ..analysis.video import (
    detector_status_at_timestamps,
    overlap_status_at_timestamps,
    phase_status_at_timestamps,
)
from ..data.reader import get_events_with_cycles_df
from ..data.video import ShapeConfig, resolve_stopbar_target
from .overlay import draw_shape_overlay

# Event codes needed to drive every lookup in analysis.video.
_GAP_CODE = -1
_PHASE_CODES = [1, 8, 9, 10, 11, 12]
_OVERLAP_CODES = [61, 63, 64, 65, 66]
_DETECTOR_CODES = [81, 82]

# Frames are buffered in chunks so status lookups are vectorised
# (one call per chunk, not per frame) without holding the whole video in
# memory at once.
_DEFAULT_CHUNK_FRAMES = 150


@dataclass
class VideoOverlayResult:
    """Summary of a completed overlay render."""
    output_path: Path
    frame_count: int
    fps: float


def render_overlay(
    db_path: Union[str, Path],
    shape_config: ShapeConfig,
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    start_dt: datetime,
    lookback_minutes: float = 10.0,
    lookahead_minutes: float = 10.0,
    chunk_frames: int = _DEFAULT_CHUNK_FRAMES,
) -> VideoOverlayResult:
    """Render *video_path* with live status overlays into *output_path*.

    Events are always fetched as raw UTC-epoch floats (``timezone`` is never
    passed to ``get_events_with_cycles_df``) -- ``analysis.detectors
    ._reconstruct_intervals``, reused here for detector status, only
    accepts epoch floats, not tz-aware ``Timestamp``s. ``start_dt`` may
    still be tz-aware; only its UTC instant matters.

    Args:
        db_path: Path to the intersection's SQLite database.
        shape_config: Loaded shape config; validated against the video's
            actual resolution before processing starts.
        video_path: Input video file.
        output_path: Destination video file (created/overwritten).
        start_dt: Real-world timestamp of the video's first frame.
        lookback_minutes: How far before ``start_dt`` to fetch events, so
            a cycle already in progress when recording started is still
            resolved correctly instead of showing ``'na'``.
        lookahead_minutes: How far past the video's end to fetch events,
            so the final cycle in the window has a confirmed "next green"
            and its trailing red period isn't reported as ``'na'`` (see
            ``atspm.analysis.video._status_at_timestamps``).
        chunk_frames: Number of frames buffered per vectorised status
            lookup batch.

    Returns:
        ``VideoOverlayResult`` with the output path, frame count, and FPS.

    Raises:
        ValueError: If the video cannot be opened, or its resolution
            doesn't match ``shape_config``.
    """
    db_path = Path(db_path)
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    shape_config.validate_resolution(width, height)

    duration_sec = (raw_count / fps) if raw_count > 0 else None
    end_dt = start_dt + timedelta(seconds=duration_sec) if duration_sec else start_dt + timedelta(hours=2)

    fetch_start = start_dt - timedelta(minutes=lookback_minutes)
    fetch_end   = end_dt + timedelta(minutes=lookahead_minutes)

    relevant_phases    = shape_config.relevant_phases()
    relevant_overlaps  = shape_config.relevant_overlaps()
    relevant_detectors = shape_config.relevant_detectors()

    event_codes = [_GAP_CODE]
    if relevant_phases:
        event_codes += _PHASE_CODES
    if relevant_overlaps:
        event_codes += _OVERLAP_CODES
    if relevant_detectors:
        event_codes += _DETECTOR_CODES

    events_df = get_events_with_cycles_df(
        db_path, fetch_start, fetch_end, event_codes=event_codes,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")

    start_epoch = start_dt.timestamp()
    frame_idx = 0

    try:
        while True:
            frames = []
            for _ in range(chunk_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if not frames:
                break

            chunk_ts = start_epoch + (np.arange(len(frames)) + frame_idx) / fps
            status_lookup = _build_status_lookup(
                events_df, relevant_phases, relevant_overlaps, relevant_detectors, chunk_ts,
            )

            for offset, frame in enumerate(frames):
                _apply_shapes(frame, shape_config, status_lookup, offset)
                ts_label = datetime.fromtimestamp(chunk_ts[offset]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
                cv2.putText(frame, ts_label, (10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                out.write(frame)

            frame_idx += len(frames)
    finally:
        cap.release()
        out.release()

    return VideoOverlayResult(output_path=output_path, frame_count=frame_idx, fps=fps)


def extract_labeled_clip(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    center_offset_sec: float,
    window_sec: float = 3.0,
) -> VideoOverlayResult:
    """Crop a short clip around *center_offset_sec* with elapsed-time labels.

    Every frame gets its elapsed video-time burned in (``t=44.367s``), so a
    user scrubbing the clip can read off the exact instant a visual event
    (e.g. a signal head changing color) occurs, for time-alignment purposes
    (see ``atspm.analysis.video.nearest_phase_transition``).

    ``cv2``'s frame-seek can land on the nearest keyframe rather than the
    exact requested frame on some codecs/containers, so the actual landed
    position is re-read via ``CAP_PROP_POS_FRAMES`` immediately after
    seeking and used as the basis for every label -- the label always
    matches the frame actually written, even if the seek itself was
    imprecise.

    Args:
        video_path: Input video file.
        output_path: Destination clip file (created/overwritten).
        center_offset_sec: Elapsed video-time (seconds from the start of
            *video_path*) to center the clip on.
        window_sec: Half-width of the clip, in seconds.

    Returns:
        ``VideoOverlayResult`` with the output path, frame count, and FPS.

    Raises:
        ValueError: If the video or output writer cannot be opened.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    requested_start_frame = max(0, int((center_offset_sec - window_sec) * fps))
    end_sec = center_offset_sec + window_sec

    cap.set(cv2.CAP_PROP_POS_FRAMES, requested_start_frame)
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")

    written = 0
    try:
        while (frame_idx / fps) <= end_sec:
            ret, frame = cap.read()
            if not ret:
                break
            elapsed = frame_idx / fps
            cv2.putText(frame, f"t={elapsed:.3f}s", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
            written += 1
            frame_idx += 1
    finally:
        cap.release()
        out.release()

    return VideoOverlayResult(output_path=output_path, frame_count=written, fps=fps)


def _build_status_lookup(
    events_df,
    relevant_phases,
    relevant_overlaps,
    relevant_detectors,
    query_ts: np.ndarray,
) -> Dict[Tuple[str, int], np.ndarray]:
    """Vectorised per-chunk status arrays, keyed by ``(kind, number)``."""
    lookup: Dict[Tuple[str, int], np.ndarray] = {}
    for phase in relevant_phases:
        lookup[("phase", phase)] = phase_status_at_timestamps(events_df, phase, query_ts)
    for overlap in relevant_overlaps:
        lookup[("overlap", overlap)] = overlap_status_at_timestamps(events_df, overlap, query_ts)
    for det in relevant_detectors:
        lookup[("det", det)] = detector_status_at_timestamps(events_df, det, query_ts)
    return lookup


def _apply_shapes(
    frame: np.ndarray,
    shape_config: ShapeConfig,
    status_lookup: Dict[Tuple[str, int], np.ndarray],
    offset: int,
) -> None:
    """Draw every shape in *shape_config* onto *frame* at chunk index *offset*."""
    for shape in shape_config.shapes:
        if shape["type"] == "loop" and shape["input"] is not None:
            status = status_lookup.get(("det", shape["input"]))
            if status is not None:
                draw_shape_overlay(frame, shape, status[offset])
        elif shape["type"] == "stopbar" and shape["phase"] is not None:
            kind, num = resolve_stopbar_target(shape["phase"])
            status = status_lookup.get((kind, num))
            if status is not None:
                draw_shape_overlay(frame, shape, status[offset])
