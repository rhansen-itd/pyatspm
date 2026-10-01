"""End-to-end golden tests for lamp measurement and video sync (Imperative Shell).

Target: src/atspm/video/sync.py — measure_lamps, sync_video, LampMeasurement,
LAMP_DISC_RADIUS; and the ``video-sync`` parser in src/atspm/cli.py.

FROZEN: written by Opus.  An implementation may not edit this file; if a
test looks wrong, stop and ask.

The clip is synthetic: a 96x96, 10 fps, 180 s .mp4 with a green lamp and a
red lamp drawn from a known phase-2 plan.  The clip's first frame is
T0 + 3.3 s; the caller's guess is T0, so the right answer is known exactly.
Green times vary cycle to cycle (as actuated greens do) so that no
whole-cycle shift scores as well as the truth.  DB edges sit 0.05 s off the
frame grid so each one falls cleanly between two frames.
"""

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from atspm import cli
from atspm.analysis.video_sync import LampSeries, SyncResult
from atspm.data.video import ShapeConfig
from atspm.video.sync import LAMP_DISC_RADIUS, LampMeasurement, measure_lamps, sync_video
from tests.conftest import seed_events

T0 = 1_780_000_000.0
TRUE_LEAD = 3.3                      # first frame is at T0 + TRUE_LEAD
FPS = 10
DURATION = 180.0
SIZE = 96
GREEN_XY = (30, 40)
RED_XY = (60, 40)
PHASE = 2
GREENS = [14.0, 9.0, 21.0, 11.0, 17.0, 8.0, 24.0, 12.0, 15.0, 10.0]
YELLOW, ALL_RED, RED_REST = 3.0, 1.0, 12.0

GREEN_ON = (60, 210, 60)
RED_ON = (60, 60, 220)
LAMP_OFF = (55, 55, 60)
BACKGROUND = (40, 40, 40)


def _plan():
    """Phase-2 events from T0 - 60 s to well past the clip, plus G/R lookups."""
    events, greens = [], []
    t = T0 - 60.0 + 0.35
    i = 0
    while t < T0 + DURATION + 120.0:
        g = GREENS[i % len(GREENS)]
        events += [
            (t, 1, PHASE),
            (t + g, 8, PHASE),
            (t + g + YELLOW, 9, PHASE),
            (t + g + YELLOW, 10, PHASE),
            (t + g + YELLOW + ALL_RED, 11, PHASE),
            (t + g + YELLOW + ALL_RED + 0.1, 12, PHASE),
        ]
        greens.append((t, t + g, t + g + YELLOW))
        t += g + YELLOW + ALL_RED + RED_REST
        i += 1
    return events, greens


EVENTS, GREEN_SPANS = _plan()


def _state(epoch: float) -> str:
    for g0, y0, r0 in GREEN_SPANS:
        if g0 <= epoch < y0:
            return "G"
        if y0 <= epoch < r0:
            return "Y"
    return "R"


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("sync") / "lamps.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    assert out.isOpened()
    for k in range(int(DURATION * FPS)):
        s = _state(T0 + TRUE_LEAD + k / FPS)
        frame = np.full((SIZE, SIZE, 3), BACKGROUND, dtype=np.uint8)
        cv2.circle(frame, GREEN_XY, 5, GREEN_ON if s == "G" else LAMP_OFF, -1)
        cv2.circle(frame, RED_XY, 5, RED_ON if s == "R" else LAMP_OFF, -1)
        out.write(frame)
    out.release()
    return path


@pytest.fixture
def db(empty_db: Path) -> Path:
    seed_events(empty_db, EVENTS)
    return empty_db


def _config(phase=str(PHASE), indications=("green", "red")) -> ShapeConfig:
    xy = {"green": GREEN_XY, "red": RED_XY}
    shapes = [
        {"type": "loop", "points": [(1, 1), (5, 1), (5, 5)], "color": (255, 0, 0),
         "input": 38, "phase": None, "name": None, "indication": None},
    ]
    shapes += [
        {"type": "lamp", "points": [xy[ind]], "color": (0, 255, 0), "input": None,
         "phase": phase, "name": f"p{phase} {ind}", "indication": ind}
        for ind in indications
    ]
    return ShapeConfig(shapes=shapes, video_width=SIZE, video_height=SIZE)


GUESS = datetime.fromtimestamp(T0, timezone.utc)


class TestMeasureLamps:

    def test_shape_of_measurement(self, clip):
        m = measure_lamps(clip, _config())
        assert isinstance(m, LampMeasurement)
        n = len(m.frame_times_s)
        assert n == pytest.approx(DURATION * FPS, abs=2)
        assert m.frame_times_s[0] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(m.frame_times_s) > 0)
        assert m.fps == pytest.approx(FPS, rel=0.01)
        assert m.timing_source in ("pts", "fps")
        # One LampSeries per lamp shape, in lamp_shapes() order; loops ignored.
        assert [l.indication for l in m.lamps] == ["green", "red"]
        assert all(isinstance(l, LampSeries) for l in m.lamps)
        assert all((l.kind, l.number) == ("phase", PHASE) for l in m.lamps)
        assert all(l.bgr.shape == (n, 3) for l in m.lamps)

    def test_disc_means_follow_the_drawn_lamp(self, clip):
        assert LAMP_DISC_RADIUS == 3
        m = measure_lamps(clip, _config())
        epochs = T0 + TRUE_LEAD + m.frame_times_s
        truth_g = np.array([_state(e) == "G" for e in epochs])
        g = m.lamps[0].bgr
        contrast = g[:, 1] - g[:, 2]
        # Interior frames only: skip the frame either side of an edge.
        stable = np.r_[False, truth_g[1:] == truth_g[:-1]] & np.r_[truth_g[:-1] == truth_g[1:], False]
        assert contrast[stable & truth_g].min() > 80
        assert np.abs(contrast[stable & ~truth_g]).max() < 30

    def test_overlap_lamp_resolves_target(self, clip):
        m = measure_lamps(clip, _config(phase="OLB", indications=("green",)))
        assert (m.lamps[0].kind, m.lamps[0].number) == ("overlap", 2)

    def test_no_lamp_shapes_raises(self, clip):
        cfg = ShapeConfig(shapes=[], video_width=SIZE, video_height=SIZE)
        with pytest.raises(ValueError, match="lamp"):
            measure_lamps(clip, cfg)

    def test_resolution_mismatch_raises(self, clip):
        cfg = _config()
        cfg.video_width = 720
        with pytest.raises(ValueError, match="resolution"):
            measure_lamps(clip, cfg)


class TestSyncVideo:

    def test_finds_true_start_from_a_late_guess(self, db, clip):
        res = sync_video(db, _config(), clip, GUESS)
        assert isinstance(res, SyncResult)
        assert res.accepted, res.reason
        assert res.start_epoch == pytest.approx(T0 + TRUE_LEAD, abs=0.1)
        assert res.mid_start_epoch == pytest.approx(T0 + TRUE_LEAD, abs=0.1)
        assert abs(res.slip_s_per_10min) < 0.1

    def test_green_lamp_alone_suffices(self, db, clip):
        res = sync_video(db, _config(indications=("green",)), clip, GUESS)
        assert res.accepted, res.reason
        assert res.start_epoch == pytest.approx(T0 + TRUE_LEAD, abs=0.1)

    def test_wrong_phase_is_refused(self, db, clip):
        res = sync_video(db, _config(phase="4"), clip, GUESS)
        assert not res.accepted

    def test_search_window_is_passed_through(self, db, clip):
        # Truth is 3.3 s away; a 2 s window cannot reach it.
        res = sync_video(db, _config(), clip, GUESS, search_s=2.0)
        assert not res.accepted

    def test_gap_marker_inside_the_clip_is_honoured(self, empty_db, clip):
        seed_events(empty_db, EVENTS, gap_at=[T0 + TRUE_LEAD + 40.0])
        res = sync_video(empty_db, _config(), clip, GUESS)
        assert res.gap_clamped is True
        assert res.accepted, res.reason
        assert res.start_epoch == pytest.approx(T0 + TRUE_LEAD, abs=0.1)


class TestCliParser:

    def test_video_sync_subcommand(self):
        parser = cli._build_parser()
        args = parser.parse_args([
            "video-sync", "--targetid", "201", "--camera", "fisheye",
            "--video", "clip.ts", "--start-guess", "2026-10-01 12:25:00",
        ])
        assert args.camera == "fisheye"
        assert args.video == "clip.ts"
        assert args.start_guess == "2026-10-01 12:25:00"
        assert args.search == pytest.approx(30.0)

    def test_video_sync_search_flag(self):
        parser = cli._build_parser()
        args = parser.parse_args([
            "video-sync", "--target", "201_SH-55", "--camera", "fisheye",
            "--video", "clip.ts", "--start-guess", "2026-10-01T12:25:00-06:00",
            "--search", "45",
        ])
        assert args.search == pytest.approx(45.0)

    def test_video_sync_has_no_all_flag(self):
        parser = cli._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "video-sync", "--all", "--camera", "fisheye",
                "--video", "clip.ts", "--start-guess", "2026-10-01 12:25:00",
            ])
