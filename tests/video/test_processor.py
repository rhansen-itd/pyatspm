"""Frame-timing tests for the video processor across .mp4 and .ts input.

Target: src/atspm/video/processor.py — render_overlay, extract_labeled_clip,
and the clock/container helpers behind them.

The fixture that matters is a stream-copied-style MPEG-TS with a stall: 20 s
at a nominal 15 fps, with source frames 100-149 removed but every surviving
frame keeping its original presentation timestamp.  That is what the remux
recorder produces when the camera stream drops out -- the container still
reports fps=15 and FRAME_COUNT=300, but only 250 frames exist, and every
frame after the hole is 50/15 = 3.33 s later than ``frame_index / fps``
says.  A constant-rate .mp4 of the same 300 frames is the control, where
the two clocks agree.

Clips are synthesized with the ffmpeg CLI; the module is skipped without it.
"""

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from atspm.data.video import ShapeConfig
from atspm.video import processor
from atspm.video.processor import (
    _output_fourcc,
    _probe_pts_timing,
    _pts_usable,
    extract_labeled_clip,
    render_overlay,
)
from tests.conftest import seed_events

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg CLI needed to synthesize clips"
)

FPS = 15
N_SOURCE = 300                       # 20 s at 15 fps
HOLE = (100, 149)                    # source frames dropped from the .ts
HOLE_LEN = HOLE[1] - HOLE[0] + 1     # 50 frames = 3.33 s
WIDTH, HEIGHT = 320, 240

# Phase 2 around the clip's first frame (T0): green from before the clip,
# yellow 10-14 s, red until the next green at 40 s.  The next interval is
# logged in full, since the red gap is only labelled 'R' once the following
# green is confirmed in the same segment (see analysis.video).
T0 = 1_780_000_000.0
PHASE = 2
PHASE_EVENTS = [
    (T0 - 5.0, 1, PHASE),    # green on
    (T0 + 10.0, 8, PHASE),   # yellow on
    (T0 + 14.0, 9, PHASE),   # yellow end
    (T0 + 15.0, 12, PHASE),  # phase inactive
    (T0 + 40.0, 1, PHASE),   # next green
    (T0 + 50.0, 8, PHASE),
    (T0 + 54.0, 9, PHASE),
    (T0 + 55.0, 12, PHASE),
]
STOPBAR_Y = 120

BGR = {"G": (0, 255, 0), "Y": (0, 255, 255), "R": (0, 0, 255), "na": (128, 128, 128)}


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict:
    """The stalled .ts and its constant-rate .mp4 control."""
    d = tmp_path_factory.mktemp("clips")
    src = ["-f", "lavfi", "-i",
           f"color=c=black:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={N_SOURCE // FPS}"]
    enc = ["-c:v", "libx264", "-g", str(FPS), "-pix_fmt", "yuv420p"]
    ts = d / "stall.ts"
    _ffmpeg(*src, "-vf", f"select='not(between(n\\,{HOLE[0]}\\,{HOLE[1]}))'",
            "-fps_mode", "passthrough", *enc, "-f", "mpegts", str(ts))
    mp4 = d / "steady.mp4"
    _ffmpeg(*src, *enc, str(mp4))
    return {"ts": ts, "mp4": mp4}


def _source_time(out_idx: int) -> float:
    """Real elapsed time of the stalled .ts's *out_idx*-th decoded frame."""
    n = out_idx if out_idx < HOLE[0] else out_idx + HOLE_LEN
    return n / FPS


def _read_frames(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _stopbar_status(frame: np.ndarray) -> str:
    """Nearest overlay colour to the stopbar's centre pixels (survives re-encode)."""
    px = frame[STOPBAR_Y - 1:STOPBAR_Y + 2, 60:260].reshape(-1, 3).mean(axis=0)
    return min(BGR, key=lambda k: np.linalg.norm(px - np.array(BGR[k])))


@pytest.fixture
def phase_db(empty_db: Path) -> Path:
    seed_events(empty_db, PHASE_EVENTS)
    return empty_db


@pytest.fixture
def stopbar_config() -> ShapeConfig:
    return ShapeConfig(
        shapes=[{"type": "stopbar", "points": [(40, STOPBAR_Y), (280, STOPBAR_Y)],
                 "color": (0, 0, 0), "input": None, "phase": str(PHASE), "name": "p2"}],
        video_width=WIDTH, video_height=HEIGHT,
    )


START_DT = datetime.fromtimestamp(T0, timezone.utc)


# ---------------------------------------------------------------------------
# Clock and container helpers
# ---------------------------------------------------------------------------

class TestPtsUsable:
    @pytest.mark.parametrize("readings", [
        [],
        [66.7],                          # too few to judge
        [0.0, 0.0, 0.0],                 # backend reports nothing
        [0.0, float("nan"), 133.3],
        [-5.0, 60.0],                    # negative start
        [0.0, 66.7, 60.0],               # goes backwards
    ])
    def test_rejects(self, readings):
        assert _pts_usable(readings) is False

    @pytest.mark.parametrize("readings", [
        [0.0, 66.7],
        [0.0, 66.7, 66.7, 3400.0],       # repeats and a stall-sized jump are fine
    ])
    def test_accepts(self, readings):
        assert _pts_usable(readings) is True


class TestOutputFourcc:
    @pytest.mark.parametrize("name", ["out.ts", "out.TS", "out", "out.mkv"])
    def test_unwritable_container_rejected(self, name):
        with pytest.raises(ValueError, match="Cannot write"):
            _output_fourcc(Path(name))

    def test_suffix_case_insensitive(self):
        assert _output_fourcc(Path("OUT.MP4")) == "mp4v"
        assert _output_fourcc(Path("out.avi")) == "MJPG"

    def test_ts_output_rejected_before_input_opened(self, tmp_path):
        # A missing input would raise "Cannot open video"; the output check
        # must come first so no work is done toward an unwritable file.
        with pytest.raises(ValueError, match="Cannot write"):
            extract_labeled_clip(tmp_path / "missing.ts", tmp_path / "clip.ts", 5.0)
        assert not (tmp_path / "clip.ts").exists()


def test_probe_reports_pts_on_both_containers(clips):
    assert _probe_pts_timing(clips["ts"]) is True
    assert _probe_pts_timing(clips["mp4"]) is True


# ---------------------------------------------------------------------------
# extract_labeled_clip
# ---------------------------------------------------------------------------

def _expected_count(times, offset, window):
    lo = max(0.0, offset - window) - 0.5 / FPS
    return sum(lo <= t <= offset + window for t in times)


TS_TIMES = [_source_time(i) for i in range(N_SOURCE - HOLE_LEN)]
MP4_TIMES = [n / FPS for n in range(N_SOURCE)]


class TestExtractLabeledClip:
    @pytest.mark.parametrize("offset", [12.0, 17.5])
    def test_ts_window_after_stall_uses_real_time(self, clips, tmp_path, offset):
        out = tmp_path / "clip.mp4"
        result = extract_labeled_clip(clips["ts"], out, offset, window_sec=1.0)
        assert result.timing_source == "pts"
        assert result.frame_count == _expected_count(TS_TIMES, offset, 1.0)
        assert len(_read_frames(out)) == result.frame_count

    def test_ts_window_at_head_skips_seek(self, clips, tmp_path):
        result = extract_labeled_clip(clips["ts"], tmp_path / "clip.mp4", 0.5, window_sec=1.0)
        assert result.frame_count == _expected_count(TS_TIMES, 0.5, 1.0)

    def test_ts_window_straddling_stall(self, clips, tmp_path):
        # 5.6-9.6 s: only source frames 84-99 (5.6-6.6 s) exist; the hole
        # runs to 10.0 s.
        result = extract_labeled_clip(clips["ts"], tmp_path / "clip.mp4", 7.6, window_sec=2.0)
        assert result.frame_count == _expected_count(TS_TIMES, 7.6, 2.0) == 16

    def test_ts_window_inside_stall_raises_and_leaves_no_file(self, clips, tmp_path):
        out = tmp_path / "clip.mp4"
        with pytest.raises(ValueError, match="No frames in the requested window"):
            extract_labeled_clip(clips["ts"], out, 8.3, window_sec=1.0)
        assert not out.exists()

    def test_window_past_end_raises(self, clips, tmp_path):
        with pytest.raises(ValueError, match="No frames"):
            extract_labeled_clip(clips["mp4"], tmp_path / "clip.mp4", 60.0, window_sec=1.0)

    def test_mp4_control(self, clips, tmp_path):
        result = extract_labeled_clip(clips["mp4"], tmp_path / "clip.mp4", 12.0, window_sec=1.0)
        assert result.timing_source == "pts"
        assert result.frame_count == _expected_count(MP4_TIMES, 12.0, 1.0)


# ---------------------------------------------------------------------------
# render_overlay
# ---------------------------------------------------------------------------

# Output frames probed, with the status each clock implies:
#   50  -> 3.33 s either way (before the stall): G
#   140 -> real 12.67 s (Y);  nominal 9.33 s (G)
#   200 -> real 16.67 s (R);  nominal 13.33 s (Y)
PROBES = [50, 140, 200]


def _phase_at(t: float) -> str:
    if t < 10.0:
        return "G"
    if t < 14.0:
        return "Y"
    return "R"


class TestRenderOverlay:
    def test_ts_stall_timed_by_pts(self, clips, tmp_path, phase_db, stopbar_config):
        out = tmp_path / "overlay.mp4"
        result = render_overlay(phase_db, stopbar_config, clips["ts"], out, START_DT)
        assert result.timing_source == "pts"
        assert result.frame_count == N_SOURCE - HOLE_LEN
        frames = _read_frames(out)
        assert len(frames) == result.frame_count
        got = [_stopbar_status(frames[i]) for i in PROBES]
        assert got == [_phase_at(_source_time(i)) for i in PROBES] == ["G", "Y", "R"]

    def test_fps_fallback_is_what_pts_prevents(
        self, clips, tmp_path, phase_db, stopbar_config, monkeypatch,
    ):
        # Forcing the nominal clock on the same clip reproduces the drift the
        # PTS clock exists to remove: every frame after the stall lags 3.33 s.
        monkeypatch.setattr(processor, "_pts_usable", lambda _: False)
        out = tmp_path / "overlay.mp4"
        result = render_overlay(phase_db, stopbar_config, clips["ts"], out, START_DT)
        assert result.timing_source == "fps"
        frames = _read_frames(out)
        assert [_stopbar_status(frames[i]) for i in PROBES] == ["G", "G", "Y"]

    def test_mp4_clocks_agree(self, clips, tmp_path, phase_db, stopbar_config):
        out = tmp_path / "overlay.mp4"
        result = render_overlay(phase_db, stopbar_config, clips["mp4"], out, START_DT)
        assert result.timing_source == "pts"
        assert result.frame_count == N_SOURCE
        frames = _read_frames(out)
        assert [_stopbar_status(frames[i]) for i in PROBES] == [
            _phase_at(i / FPS) for i in PROBES
        ]

    def test_ts_output_rejected(self, clips, tmp_path, phase_db, stopbar_config):
        with pytest.raises(ValueError, match="Cannot write"):
            render_overlay(phase_db, stopbar_config, clips["ts"], tmp_path / "o.ts", START_DT)
