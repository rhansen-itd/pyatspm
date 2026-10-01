"""Golden tests for video/data sync from signal lamps (Functional Core).

Target: src/atspm/analysis/video_sync.py — align_clip, lamp_on_series,
lamp_contrast, otsu_threshold, debounce_frames, LampSeries, SyncResult.

FROZEN: written by Opus from the 2026-10-01 hand-built prototype. An
implementation may not edit this file; if a test looks wrong, stop and ask.

Fixture (tests/analysis/fixtures/video_sync_201/), cut from the gitignored
``intersections/201_.../201_20261001_1209/`` collection so these tests stand
alone:

``lamps.npz``
    For each of the three 10-minute fisheye clips (720x720, ~10 fps, .ts):
    ``clipN_pts_ms`` -- ``CAP_PROP_POS_MSEC`` read straight after each decode
    (frame intervals jitter 50-150 ms around 100 ms), and
    ``clipN_{green,yellow,red}_bgr`` -- per-frame mean B,G,R over a radius-3
    disc (29 px) centred on the NB (phase 2) head's green (643,417), yellow
    (647,417) and red (652,418) lamps.  That is exactly what the shell's
    lamp measurement hands the core.
``events.csv``
    ``get_events_with_cycles_df`` output for 12:00-13:15 MDT, phase codes
    1/8/9/10/11/12 for every phase plus gap markers (one real marker at
    12:05:06, from a 2.6 s backward clock step).

Golden first-frame times come from matching every green-lamp edge to the
DB by hand; they agreed to within one frame per clip (ROADMAP, *Automatic
video/data sync from signal lamps*).  The yellow lamp is deliberately not
held to the golden accuracy: in the prototype its alignment came out
0.2-1.0 s early on every clip, cause unknown.

Tolerances: the floor is about half a frame (50 ms) plus the DB's 0.1 s
quantisation, and the golden values are themselves median-of-edges
estimates, so ``mid_start_epoch`` (the clip-midpoint value, which is what a
median of edges estimates) is held to 0.15 s.  ``start_epoch`` (the first
frame) also absorbs the slip extrapolated over half a clip, so 0.25 s.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.video_sync import (
    LampSeries,
    SyncResult,
    align_clip,
    debounce_frames,
    lamp_contrast,
    lamp_on_series,
    otsu_threshold,
)

FIXTURE = Path(__file__).parent / "fixtures" / "video_sync_201"
MT = ZoneInfo("US/Mountain")

# (start_guess from the collector's manifest, golden first-frame time)
CLIPS = {
    1: (datetime(2026, 10, 1, 12, 25, 0, 0, MT), datetime(2026, 10, 1, 12, 24, 57, 550000, MT)),
    2: (datetime(2026, 10, 1, 12, 39, 59, 990000, MT), datetime(2026, 10, 1, 12, 39, 55, 952000, MT)),
    3: (datetime(2026, 10, 1, 12, 54, 59, 995000, MT), datetime(2026, 10, 1, 12, 54, 54, 950000, MT)),
}
LAMP_PHASE = 2
FPS = 10.0
MID_TOL = 0.15
START_TOL = 0.25


@pytest.fixture(scope="module")
def lamps_npz():
    with np.load(FIXTURE / "lamps.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def events():
    return pd.read_csv(FIXTURE / "events.csv")


def _clip(lamps_npz, n, indications=("green",), phase=LAMP_PHASE):
    t = lamps_npz[f"clip{n}_pts_ms"] / 1000.0
    lamps = [
        LampSeries(kind="phase", number=phase, indication=ind,
                   bgr=lamps_npz[f"clip{n}_{ind}_bgr"])
        for ind in indications
    ]
    return t, lamps


def _guess(n, shift=0.0):
    return CLIPS[n][0].timestamp() + shift


def _gold(n):
    return CLIPS[n][1].timestamp()


# ---------------------------------------------------------------------------
# Golden alignment on the real 201 clips
# ---------------------------------------------------------------------------


class TestGoldenClips:

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_green_lamp_hits_golden_start(self, lamps_npz, events, n):
        t, lamps = _clip(lamps_npz, n)
        res = align_clip(t, lamps, events, _guess(n))
        assert isinstance(res, SyncResult)
        assert res.accepted, res.reason
        assert res.reason is None
        assert abs(res.mid_start_epoch - _gold(n)) <= MID_TOL
        assert abs(res.start_epoch - _gold(n)) <= START_TOL

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_green_and_red_lamps_together(self, lamps_npz, events, n):
        t, lamps = _clip(lamps_npz, n, ("green", "red"))
        res = align_clip(t, lamps, events, _guess(n))
        assert res.accepted, res.reason
        assert abs(res.mid_start_epoch - _gold(n)) <= MID_TOL

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_slip_is_reported(self, lamps_npz, events, n):
        # Prototype: video vs controller slipped 0.1-0.3 s per 10 min, with
        # the implied first-frame time drifting earlier as the clip runs.
        t, lamps = _clip(lamps_npz, n)
        res = align_clip(t, lamps, events, _guess(n))
        assert -0.45 <= res.slip_s_per_10min <= -0.05
        # mid_start is start advanced by the slip over half the clip.
        half = (t[-1] - t[0]) / 2.0
        expected_mid = res.start_epoch + res.slip_s_per_10min / 600.0 * (t[0] + half)
        assert res.mid_start_epoch == pytest.approx(expected_mid, abs=1e-6)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_confidence_fields(self, lamps_npz, events, n):
        t, lamps = _clip(lamps_npz, n)
        res = align_clip(t, lamps, events, _guess(n))
        # Prototype values: score 0.875-0.998, runner-up (a wrong-cycle
        # peak ~44-64 s away) 0.03-0.35.
        assert res.score >= 0.8
        assert res.score - res.runner_up_score >= 0.4
        assert 0.9 <= res.agreement <= 1.0
        assert res.n_edges_used >= 10
        assert res.n_frames_compared >= 0.95 * len(t)
        assert res.gap_clamped is False  # the 12:05:06 marker is outside the window

    @pytest.mark.parametrize("shift", [-12.0, -5.0, 5.0, 12.0])
    def test_insensitive_to_guess_error_inside_search(self, lamps_npz, events, shift):
        # The collector's trigger time was 2.5-5 s late; any guess inside
        # the search window must land on the same answer.
        t, lamps = _clip(lamps_npz, 2)
        res = align_clip(t, lamps, events, _guess(2, shift))
        assert res.accepted, res.reason
        assert abs(res.mid_start_epoch - _gold(2)) <= MID_TOL

    def test_frame_times_are_used_not_index_over_fps(self, lamps_npz, events):
        # A camera stall: 50 frames (5 s) vanish mid-clip but every surviving
        # frame keeps its PTS, as the remux recorder produces.  Frames after
        # the hole are 5 s later than index/fps says, so only a PTS-timed
        # aligner still lands on the golden start.
        t, lamps = _clip(lamps_npz, 2)
        keep = np.ones(len(t), dtype=bool)
        keep[3000:3050] = False
        lamps = [LampSeries(l.kind, l.number, l.indication, l.bgr[keep]) for l in lamps]
        res = align_clip(t[keep], lamps, events, _guess(2))
        assert res.accepted, res.reason
        assert abs(res.mid_start_epoch - _gold(2)) <= MID_TOL
        # And the answer moves with the frame clock.
        res_shifted = align_clip(t[keep] + 1.0, lamps, events, _guess(2))
        assert res_shifted.mid_start_epoch == pytest.approx(res.mid_start_epoch - 1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Refusal: a wrong sync is worse than none
# ---------------------------------------------------------------------------


class TestRefusal:

    @pytest.mark.parametrize("wrong_phase", [4, 8])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_wrong_phase_is_refused(self, lamps_npz, events, n, wrong_phase):
        t, lamps = _clip(lamps_npz, n, phase=wrong_phase)
        res = align_clip(t, lamps, events, _guess(n))
        assert not res.accepted
        assert res.reason

    def test_true_offset_outside_search_window_is_refused(self, lamps_npz, events):
        # Guess one whole cycle late with a window too small to reach the
        # truth: the best in-window peak is a wrong-cycle lock.
        t, lamps = _clip(lamps_npz, 1)
        res = align_clip(t, lamps, events, _guess(1, 44.0), search_s=20.0)
        assert not res.accepted

    def test_constant_lamp_is_refused_without_raising(self, lamps_npz, events):
        t, _ = _clip(lamps_npz, 2)
        flat = LampSeries("phase", LAMP_PHASE, "green", np.full((len(t), 3), 80.0, dtype=np.float32))
        res = align_clip(t, [flat], events, _guess(2))
        assert not res.accepted

    def test_noise_lamp_is_refused(self, lamps_npz, events):
        t, _ = _clip(lamps_npz, 2)
        rng = np.random.default_rng(0)
        noise = LampSeries("phase", LAMP_PHASE, "green",
                           rng.uniform(0, 255, (len(t), 3)).astype(np.float32))
        res = align_clip(t, [noise], events, _guess(2))
        assert not res.accepted

    def test_no_events_is_refused(self, lamps_npz, events):
        t, lamps = _clip(lamps_npz, 2)
        res = align_clip(t, lamps, events.iloc[0:0], _guess(2))
        assert not res.accepted

    def test_thresholds_are_parameters(self, lamps_npz, events):
        t, lamps = _clip(lamps_npz, 2)
        assert not align_clip(t, lamps, events, _guess(2), min_score=1.01).accepted
        assert not align_clip(t, lamps, events, _guess(2), min_margin=1.5).accepted
        assert not align_clip(t, lamps, events, _guess(2), min_edges=10_000).accepted


# ---------------------------------------------------------------------------
# Hazards: occlusion, PWM dropouts, gap markers
# ---------------------------------------------------------------------------


class TestHazards:

    def test_long_occlusion_is_absorbed(self, lamps_npz, events):
        # A 30 s truck parked in front of the head: lamp reads "off".
        t, lamps = _clip(lamps_npz, 2)
        bgr = lamps[0].bgr.copy()
        block = (t > 200) & (t < 230)
        bgr[block] = [60.0, 55.0, 65.0]   # dark housing, G-R < 0
        res = align_clip(t, [LampSeries("phase", LAMP_PHASE, "green", bgr)], events, _guess(2))
        assert res.accepted, res.reason
        assert abs(res.mid_start_epoch - _gold(2)) <= MID_TOL

    def test_single_frame_pwm_dropouts_are_debounced(self, lamps_npz, events):
        # Every 7th lit frame reads dark (LED PWM aliasing against the shutter).
        t, lamps = _clip(lamps_npz, 2)
        bgr = lamps[0].bgr.copy()
        lit = np.flatnonzero(lamp_contrast(bgr, "green") > 5.0)
        bgr[lit[::7]] = [60.0, 55.0, 65.0]
        res = align_clip(t, [LampSeries("phase", LAMP_PHASE, "green", bgr)], events, _guess(2))
        assert res.accepted, res.reason
        assert abs(res.mid_start_epoch - _gold(2)) <= MID_TOL
        # The edge count must not balloon from the dropouts.
        clean = align_clip(t, lamps, events, _guess(2))
        assert res.n_edges_used <= clean.n_edges_used + 2

    def test_gap_marker_mid_clip_clamps_to_one_segment(self, lamps_npz, events):
        # A hard reset 150 s into clip 2: align on the larger side only.
        t, lamps = _clip(lamps_npz, 2)
        marker_ts = _gold(2) + 150.0
        gap = pd.DataFrame({"timestamp": [marker_ts], "event_code": [-1],
                            "parameter": [-1], "cycle_start": [np.nan]})
        ev = pd.concat([events, gap], ignore_index=True).sort_values("timestamp", kind="stable")
        res = align_clip(t, lamps, ev, _guess(2))
        assert res.gap_clamped is True
        assert res.accepted, res.reason
        assert abs(res.mid_start_epoch - _gold(2)) <= MID_TOL
        # Frames before the marker are not compared.
        assert res.n_frames_compared <= int(0.8 * len(t))

    def test_never_aligns_across_a_clock_step(self, lamps_npz, events):
        # Shift every event after a mid-clip marker by +2.6 s (a backward
        # clock set, logged as a gap).  An aligner that spans the marker
        # would average the two clocks; the clamped one keeps the bigger
        # side and lands on that side's answer.
        t, lamps = _clip(lamps_npz, 2)
        marker_ts = _gold(2) + 150.0
        ev = events.copy()
        after = ev["timestamp"] > marker_ts
        ev.loc[after, "timestamp"] += 2.6
        ev.loc[after & ev["cycle_start"].notna(), "cycle_start"] += 2.6
        gap = pd.DataFrame({"timestamp": [marker_ts], "event_code": [-1],
                            "parameter": [-1], "cycle_start": [np.nan]})
        ev = pd.concat([ev, gap], ignore_index=True).sort_values("timestamp", kind="stable")
        res = align_clip(t, lamps, ev, _guess(2))
        assert res.gap_clamped is True
        assert res.accepted, res.reason
        assert abs(res.mid_start_epoch - (_gold(2) + 2.6)) <= MID_TOL

    def test_small_segment_alone_is_refused(self, lamps_npz, events):
        # Two markers 200 s apart around the middle: no single segment
        # covers even half the clip (min_coverage defaults to 0.5).
        t, lamps = _clip(lamps_npz, 2)
        marks = [_gold(2) + 200.0, _gold(2) + 400.0]
        gap = pd.DataFrame({"timestamp": marks, "event_code": [-1, -1],
                            "parameter": [-1, -1], "cycle_start": [np.nan, np.nan]})
        ev = pd.concat([events, gap], ignore_index=True).sort_values("timestamp", kind="stable")
        res = align_clip(t, lamps, ev, _guess(2))
        assert res.gap_clamped is True
        assert not res.accepted


# ---------------------------------------------------------------------------
# Helper contracts
# ---------------------------------------------------------------------------


class TestHelpers:

    def test_contrast_per_indication(self):
        bgr = np.array([[10.0, 50.0, 20.0]])        # B, G, R
        assert lamp_contrast(bgr, "green")[0] == pytest.approx(30.0)   # G - R
        assert lamp_contrast(bgr, "red")[0] == pytest.approx(-30.0)    # R - G
        assert lamp_contrast(bgr, "yellow")[0] == pytest.approx(25.0)  # (R+G)/2 - B

    def test_contrast_rejects_unknown_indication(self):
        with pytest.raises(ValueError):
            lamp_contrast(np.zeros((1, 3)), "blue")

    def test_otsu_splits_two_modes(self):
        rng = np.random.default_rng(1)
        x = np.r_[rng.normal(-5, 1.5, 4000), rng.normal(15, 2.0, 2000)]
        th = otsu_threshold(x)
        assert 0.0 < th < 10.0

    def test_otsu_follows_the_clip_not_a_constant(self):
        # Same shape, different exposure: threshold must move with it.
        rng = np.random.default_rng(2)
        lo, hi = rng.normal(20, 2, 4000), rng.normal(60, 3, 2000)
        th = otsu_threshold(np.r_[lo, hi])
        assert 20.0 < th < 60.0
        assert (lo < th).mean() > 0.99 and (hi > th).mean() > 0.99

    def test_otsu_constant_input_is_nan(self):
        assert np.isnan(otsu_threshold(np.full(100, 3.0)))

    @pytest.mark.parametrize("fps,expected", [(10.0, 3), (15.0, 4), (30.0, 8), (2.0, 1)])
    def test_debounce_sized_from_fps(self, fps, expected):
        # 0.25 s, rounded up to whole frames, at least one.
        assert debounce_frames(fps) == expected

    def test_lamp_on_series_fills_short_dropouts_keeps_real_states(self):
        on_runs = [(False, 20), (True, 30), (False, 1), (True, 30), (False, 40), (True, 2), (False, 20)]
        truth = np.concatenate([np.full(n, v) for v, n in on_runs])
        bgr = np.where(truth[:, None], [40.0, 120.0, 40.0], [40.0, 40.0, 60.0])
        on, th = lamp_on_series(bgr, "green", 10.0)
        assert on.dtype == bool and len(on) == len(truth)
        assert -20.0 < th < 80.0
        # One-frame dark blip inside a green is filled.
        assert on[20:81].all()
        # Two-frame flash inside a long off is removed.
        assert not on[81:].any()
        assert not on[:20].any()

    def test_real_green_lamp_is_bimodal(self, lamps_npz):
        # ~-5 off, ~+15 on in the prototype; the per-clip threshold sits between.
        for n in (1, 2, 3):
            c = lamp_contrast(lamps_npz[f"clip{n}_green_bgr"], "green")
            th = otsu_threshold(c)
            assert 0.0 < th < 10.0
            on, _ = lamp_on_series(lamps_npz[f"clip{n}_green_bgr"], "green", FPS)
            assert 0.5 < on.mean() < 0.85
