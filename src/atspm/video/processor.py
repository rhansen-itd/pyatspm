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

Input containers
----------------
Both recorder backends are supported: the frame-decode recorder's constant
-rate ``.mp4`` and the remux recorder's stream-copied ``.ts`` (MPEG-TS).
They demand different frame->wall-clock math, and the difference is not
cosmetic:

- The ``.mp4`` recorder derives one exact FPS from the measured wall-clock
  span of the clip, so ``frame_index / fps`` *is* elapsed real time.
- The ``.ts`` recorder copies encoded packets straight through, so real
  timing rides on each packet's presentation timestamp.  Dropped frames
  and stream stalls leave the nominal FPS intact while elapsed time runs
  ahead of ``frame_index / fps`` -- a 3.3 s stall in a 20 s clip puts the
  last frame 3.3 s off, and every overlay after the stall with it.

So frames are timed by presentation timestamp (``CAP_PROP_POS_MSEC``)
whenever the container reports one, falling back to the nominal
``frame_index / fps`` clock otherwise.  On a constant-rate ``.mp4`` the
two agree, so this is the strictly better clock in both cases.

Seeking is likewise container-sensitive: an MPEG-TS capture carries no
index, so a seek snaps to the enclosing keyframe and a seek back toward
zero cannot recover frames already consumed.  Nothing here seeks before
the main read loop, and the one function that does seek
(:func:`extract_labeled_clip`) labels from where it actually landed.

Package Location: src/atspm/video/processor.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import cv2
import numpy as np

from ..analysis.video import (
    detector_status_at_timestamps,
    overlap_status_at_timestamps,
    phase_status_at_timestamps,
)
from ..data.reader import _resolve_timezone, get_events_with_cycles_df
from ..utils.timezone import localize_naive
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

# Input containers this module is known to read.  Used only to phrase the
# error when a file won't open -- OpenCV's FFmpeg backend reads more than
# this, so nothing is rejected on extension alone.
_KNOWN_INPUT_SUFFIXES = (".mp4", ".ts", ".mkv", ".mov", ".m4v", ".avi")

# Output containers cv2.VideoWriter can be driven to produce here, mapped
# to the FOURCC each one needs.  Rendering an overlay re-encodes, and the
# MPEG-TS muxer accepts none of OpenCV's FOURCC tags -- so .ts is an input
# format only, and asking for it as output is rejected outright rather
# than silently writing a file with a mismatched codec tag.
_OUTPUT_FOURCC = {".mp4": "mp4v", ".m4v": "mp4v", ".mov": "mp4v", ".avi": "MJPG"}

# How far behind a wanted position to aim a seek, before decoding forward to
# it.  A seek into an indexless container resolves to a keyframe, and near the
# head of an MPEG-TS clip it can resolve to the keyframe *after* the requested
# position rather than before -- the muxer's initial offset (~1.4 s for
# FFmpeg's mpegts muxer) puts early requests behind the first packet's real
# timestamp.  Undershooting deliberately and skipping forward lands on the
# wanted frame instead of somewhere past it; a target at or before zero skips
# the seek entirely and reads from the first frame.  Sized at ~2x a typical
# GOP, matching the recorder's own keyframe margin.
_SEEK_MARGIN_SEC = 4.0


@dataclass
class VideoOverlayResult:
    """Summary of a completed overlay render.

    Args:
        output_path: The written video file.
        frame_count: Frames written.
        fps: The input's reported frame rate, carried through to the output.
        timing_source: ``'pts'`` if frames were timed by presentation
            timestamp, ``'fps'`` if the container reported none and the
            nominal ``frame_index / fps`` clock was used instead.  See the
            module docstring for why the distinction matters on ``.ts``.
    """
    output_path: Path
    frame_count: int
    fps: float
    timing_source: str = "pts"


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
    still be tz-aware; only its UTC instant matters. A naive ``start_dt``
    is localized to the intersection's own zone up front, so the fetch
    window and the per-frame epochs derived from it cannot drift apart.

    Frames are timed by presentation timestamp when *video_path*'s
    container reports one, which is what keeps a stream-copied ``.ts``
    clip aligned -- see the module docstring.  ``start_dt`` anchors the
    clip's *first* frame either way, so the two clocks share an origin and
    a caller's ``--start`` calibration carries over between formats
    unchanged.

    Args:
        db_path: Path to the intersection's SQLite database.
        shape_config: Loaded shape config; validated against the video's
            actual resolution before processing starts.
        video_path: Input video file (``.mp4`` from the frame-decode
            recorder or ``.ts`` from the remux recorder; anything else
            OpenCV's FFmpeg backend can open also works).
        output_path: Destination video file (created/overwritten).  The
            extension selects the writer's codec and must be one of
            ``.mp4``/``.m4v``/``.mov``/``.avi`` -- the overlay is
            re-encoded, and ``.ts`` cannot be written by OpenCV.
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
        ``VideoOverlayResult`` with the output path, frame count, FPS, and
        which frame clock was used.

    Raises:
        ValueError: If the video cannot be opened, its resolution doesn't
            match ``shape_config``, or ``output_path``'s extension names a
            container that cannot be written.
    """
    db_path = Path(db_path)
    video_path = Path(video_path)
    output_path = Path(output_path)

    fourcc_tag = _output_fourcc(output_path)
    start_dt = localize_naive(start_dt, _resolve_timezone(db_path))

    cap = _open_capture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    shape_config.validate_resolution(width, height)

    # Only ever an estimate: an indexless container (MPEG-TS) has no true
    # frame count, so CAP_PROP_FRAME_COUNT is itself derived from duration
    # x fps and comes back slightly long or short.  It only sizes the event
    # fetch window, which lookahead_minutes pads generously either way.
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

    fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")

    start_epoch = start_dt.timestamp()
    # Burn the label in the zone the caller framed start_dt in, not the host's.
    label_tz = start_dt.tzinfo
    frame_idx = 0
    # Decided once, on the first chunk, and held for the whole render -- the
    # clock must not change mid-video.  Deciding here rather than by probing
    # up front is deliberate: an MPEG-TS capture cannot be rewound to frame
    # zero once frames have been consumed (a seek back snaps forward to the
    # enclosing keyframe), and this loop already buffers a chunk to sample.
    use_pts: Union[bool, None] = None

    try:
        while True:
            frames = []
            frame_msec = []
            for _ in range(chunk_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                # Read straight after the decode: POS_MSEC then reports the
                # frame just returned, not the one queued next.
                frame_msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            if not frames:
                break

            if use_pts is None:
                use_pts = _pts_usable(frame_msec)

            if use_pts:
                chunk_ts = start_epoch + np.asarray(frame_msec, dtype=float) / 1000.0
            else:
                chunk_ts = start_epoch + (np.arange(len(frames)) + frame_idx) / fps
            status_lookup = _build_status_lookup(
                events_df, relevant_phases, relevant_overlaps, relevant_detectors, chunk_ts,
            )

            for offset, frame in enumerate(frames):
                _apply_shapes(frame, shape_config, status_lookup, offset)
                ts_label = datetime.fromtimestamp(
                    chunk_ts[offset], label_tz
                ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
                cv2.putText(frame, ts_label, (10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                out.write(frame)

            frame_idx += len(frames)
    finally:
        cap.release()
        out.release()

    return VideoOverlayResult(
        output_path=output_path,
        frame_count=frame_idx,
        fps=fps,
        timing_source="pts" if use_pts else "fps",
    )


def extract_labeled_clip(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    expected_offset_sec: float,
    window_sec: float = 3.0,
) -> VideoOverlayResult:
    """Crop a short clip around *expected_offset_sec* with a normalized countdown label.

    Every frame gets a signed countdown to *expected_offset_sec* burned in
    (``+0.300s`` / ``-0.300s``), not the raw elapsed video-time -- positive
    while the clip hasn't yet reached the frame where the transition
    *should* occur (assuming the caller's ``--start`` guess is exactly
    right), negative after. A user reading the label off the frame where
    the transition *actually*, visually happens gets a value that can be
    added directly to that ``--start`` guess to correct it: if the
    transition happens later than expected the label is negative there
    (subtract), if earlier it's positive (add) -- see
    ``atspm.analysis.video.first_phase_transition_after`` for how
    *expected_offset_sec* is derived.

    A seek lands on the nearest keyframe rather than the exact requested
    position on plenty of codec/container combinations -- always, on an
    indexless MPEG-TS clip -- so nothing is inferred from where the seek
    was *asked* to go.  Each frame is labeled from its own position, read
    back after its decode, so the label matches the frame actually written
    however imprecise the seek was.  Frames the seek overshot backwards
    onto are skipped rather than written, keeping the clip to the window
    that was asked for.

    Args:
        video_path: Input video file (``.mp4``, ``.ts``, or anything else
            OpenCV's FFmpeg backend can open).
        output_path: Destination clip file (created/overwritten).  Its
            extension selects the writer's codec -- see
            :func:`render_overlay`.
        expected_offset_sec: Elapsed video-time (seconds from the start of
            *video_path*) at which the transition is expected, and the
            point the clip is centered on and labels are normalized to.
        window_sec: Half-width of the clip, in seconds.

    Returns:
        ``VideoOverlayResult`` with the output path, frame count, FPS, and
        which frame clock was used.

    Raises:
        ValueError: If the video or output writer cannot be opened, or if
            the requested window contains no frames (a recording gap, or a
            window past the end of the clip).
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    fourcc_tag = _output_fourcc(output_path)

    # Probed on its own capture, which is then discarded: sampling the head
    # of an MPEG-TS clip consumes frames a seek cannot get back, so the
    # capture that does the real work below must be untouched.
    use_pts = _probe_pts_timing(video_path)

    cap = _open_capture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_sec = max(0.0, expected_offset_sec - window_sec)
    end_sec = expected_offset_sec + window_sec
    # Half a frame of slack, so a frame landing exactly on the window edge
    # isn't dropped by float noise in the position readback.
    skip_before_sec = start_sec - 0.5 / fps

    frame_idx = 0
    if use_pts:
        # Seek by timestamp, not frame index: an indexless container has no
        # frame numbers to seek by, and the timestamp is what the labels are
        # derived from anyway.  Aim short of the window and decode forward
        # into it -- see _SEEK_MARGIN_SEC.
        seek_sec = start_sec - _SEEK_MARGIN_SEC
        if seek_sec > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000.0)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_sec * fps)))
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")

    written = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if use_pts:
                elapsed = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                elapsed = frame_idx / fps
                frame_idx += 1

            if elapsed > end_sec:
                break
            if elapsed < skip_before_sec:
                continue

            label = expected_offset_sec - elapsed
            cv2.putText(frame, f"{label:+.3f}s", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
            written += 1
    finally:
        cap.release()
        out.release()

    if written == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"No frames in the requested window "
            f"({start_sec:.3f}s-{end_sec:.3f}s) of {video_path.name}. "
            f"The clip may end before then, or the recording may have a gap there."
        )

    return VideoOverlayResult(
        output_path=output_path,
        frame_count=written,
        fps=fps,
        timing_source="pts" if use_pts else "fps",
    )


def _open_capture(video_path: Path) -> cv2.VideoCapture:
    """Open *video_path* for reading, or raise with the formats we read.

    Args:
        video_path: The video file to open.

    Returns:
        An opened ``cv2.VideoCapture``.

    Raises:
        ValueError: If OpenCV cannot open the file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(
            f"Cannot open video: {video_path}\n"
            f"Recognized formats: {', '.join(_KNOWN_INPUT_SUFFIXES)} "
            f"(the file may be truncated, or its codec unsupported by this "
            f"OpenCV build)."
        )
    return cap


def _output_fourcc(output_path: Path) -> str:
    """Return the writer FOURCC for *output_path*'s container.

    Args:
        output_path: The destination video file.

    Returns:
        The FOURCC tag to hand ``cv2.VideoWriter_fourcc``.

    Raises:
        ValueError: If the extension names a container this module cannot
            write -- notably ``.ts``, which is read-only here (see
            ``_OUTPUT_FOURCC``).
    """
    suffix = output_path.suffix.lower()
    try:
        return _OUTPUT_FOURCC[suffix]
    except KeyError:
        raise ValueError(
            f"Cannot write {suffix or 'an extensionless file'} as output: "
            f"{output_path}\n"
            f"Rendered video must be one of: {', '.join(_OUTPUT_FOURCC)}. "
            f"(.ts is an input format only -- OpenCV cannot mux to MPEG-TS.)"
        ) from None


def _pts_usable(pos_msec: Sequence[float]) -> bool:
    """Whether a run of ``CAP_PROP_POS_MSEC`` readings is a usable clock.

    Some backend/container pairs report nothing -- a constant 0.0, or a
    NaN -- in which case the caller has to fall back to the nominal
    ``frame_index / fps`` clock.  A real timestamp run advances and never
    goes backwards.

    Args:
        pos_msec: Position readings, in order, one per decoded frame.

    Returns:
        ``True`` if the readings can time frames; ``False`` to fall back.
    """
    if len(pos_msec) < 2:
        return False
    if not all(math.isfinite(m) for m in pos_msec):
        return False
    if pos_msec[0] < 0.0 or pos_msec[-1] <= 0.0:
        return False
    return all(b >= a for a, b in zip(pos_msec, pos_msec[1:]))


def _probe_pts_timing(video_path: Path) -> bool:
    """Whether *video_path* reports presentation timestamps, via a throwaway read.

    Opens its own capture and reads the first two frames.  The capture is
    released rather than rewound: on an indexless container those frames
    cannot be seeked back to, so a caller that needs the head of the file
    intact must open a fresh capture afterwards.

    Args:
        video_path: The video file to probe.

    Returns:
        ``True`` if frames can be timed by presentation timestamp.

    Raises:
        ValueError: If OpenCV cannot open the file.
    """
    cap = _open_capture(video_path)
    samples = []
    try:
        for _ in range(2):
            ret, _frame = cap.read()
            if not ret:
                break
            samples.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()
    return _pts_usable(samples)


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
