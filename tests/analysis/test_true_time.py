# Tests for the true-time axis (functional core).
#
# Golden half: a simulated controller clock (known offset, rate and daily
# sets, logging eos_set_time's marker protocol with the bench-measured width
# bias) is the oracle, so every corrected timestamp can be checked against
# the true instant it was logged at.  The bench files then confirm the model
# agrees with the head unit's own host-timed measurements.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.clock_marks import MarkerPeds, decode_clock_marks, send_log_pulses
from atspm.analysis.decoders import CLOCK_STEP_FENCE_PARAM, COMMS_GAP_PARAM
from atspm.analysis.true_time import (
    MODEL_COLUMNS,
    apply_true_time,
    drift_model,
    to_true_time,
)
from atspm.data.ingestion import IngestionEngine
from atspm.data.manager import DatabaseManager

BENCH = Path(__file__).resolve().parents[1] / "fixtures" / "clock_marks_bench_2026_09_30"
PEDS = MarkerPeds(behind=15, ahead=16, set=14)

DAY = 86400.0
T0 = 1_790_726_400.0  # 2026-09-30 00:00 UTC
SHARED_CODE, SHARED_PARAM = 1, 2  # a stimulus both simulated controllers log


# ---------------------------------------------------------------------------
# Simulated controller
# ---------------------------------------------------------------------------

class Controller:
    """A controller clock: label = true + d(true), logged at 0.1 s.

    ``d`` starts at *d0*, runs at *ppm*, and steps by each applied shift.
    It logs eos_set_time's pulses: hourly checks at :37, a daily set at
    09:17 UTC, with the ON registering 0.0-0.3 s late (bench-measured).
    *wander* adds a line-frequency style excursion ``wander(true - T0)``.
    """

    def __init__(self, d0, ppm, seed, days=2, set_min_drift=0.75, wander=None):
        self.rng = np.random.default_rng(seed)
        self.d0, self.rate = d0, ppm * 1e-6
        self.wander = wander
        self.host = []  # (true send time, host-timed drift), as eos-time.jsonl
        self.steps = []  # (true time, shift)
        self.rows = []   # (true time, code, param, pulse_on_delay)
        self.true_shift = []
        self.days = days
        self.set_min_drift = set_min_drift

    def d(self, t):
        t = np.asarray(t, dtype=float)
        out = self.d0 + self.rate * (t - T0)
        if self.wander is not None:
            out = out + self.wander(t - T0)
        for ts, s in self.steps:
            out = out + np.where(t >= ts, s, 0.0)
        return out

    def _pulse(self, t, width, ped):
        delay = self.rng.uniform(0.0, min(0.3, width))
        self.rows += [(t + delay, 90, ped), (t + delay, 45, ped), (t + width, 89, ped)]
        return t + width

    def _drift_pulse(self, t):
        m = float(self.d(t)) + self.rng.normal(0, 0.02)
        self.host.append((t, m))
        width = float(np.clip(round(abs(m), 1), 0.1, 30.0))
        return self._pulse(t, width, PEDS.ahead if m > 0 else PEDS.behind), m

    def _set(self, t, edit_lead=2.0):
        off, m = self._drift_pulse(t)
        if abs(m) < self.set_min_drift:
            return
        shift = -round(m)
        period = 10.0
        while period + shift < 5:
            period += 10.0
        on = off + 0.05
        self.steps.append((on + edit_lead, float(shift)))
        self.true_shift.append(shift)
        b_off = self._pulse(on, period, PEDS.set)
        self._drift_pulse(b_off + 0.05)

    def run(self, extra_steps=(), shared=True, background=True):
        for t, s in extra_steps:  # unmarked steps (keypad, power event)
            self.steps.append((t, float(s)))
        for day in range(self.days):
            base = T0 + day * DAY
            for hour in range(24):
                if hour == 9:
                    self._set(base + 9 * 3600 + 17 * 60)
                self._drift_pulse(base + hour * 3600 + 37 * 60)
        span = self.days * DAY
        if background:
            t = np.cumsum(self.rng.uniform(0.3, 1.5, int(span / 0.9) + 10))
            t = T0 + t[t < span]
            self.rows += [(x, 82, 3) for x in t]
        if shared:
            self.rows += [(x, SHARED_CODE, SHARED_PARAM) for x in shared_times(self.days)]
        return self

    def host_log(self):
        """The head unit's own drift measurements, keyed by ON label time."""
        t = np.array([h[0] for h in self.host])
        return pd.DataFrame({"label": t + self.d(t), "drift": [h[1] for h in self.host]})

    def log(self):
        """Events as the DB holds them: label time, fenced, sorted, unique."""
        rows = sorted(self.rows, key=lambda r: r[0])
        true = np.array([r[0] for r in rows])
        label = np.floor((true + self.d(true)) * 10 + 1e-6) / 10
        df = pd.DataFrame({
            "timestamp": label,
            "event_code": [r[1] for r in rows],
            "parameter": [r[2] for r in rows],
            "true": true,
        })
        drops = np.flatnonzero(np.diff(label) < 0) + 1
        fences = pd.DataFrame({
            "timestamp": label[drops] - 0.05, "event_code": -1,
            "parameter": CLOCK_STEP_FENCE_PARAM, "true": np.nan,
        })
        df = pd.concat([df, fences], ignore_index=True)
        df = df.drop_duplicates(["timestamp", "event_code", "parameter"])
        return df.sort_values(["timestamp", "event_code", "parameter"]).reset_index(drop=True)


def shared_times(days):
    return T0 + 3.3 + 10.0 * np.arange(int(days * DAY / 10) - 1)


def correct(events, send_log=None, host=None):
    """The read path: decode, fit, map. Returns (mapped events, model, sets).

    *host* (``Controller.host_log()``) fills ``drift_host`` the way a
    matched eos-time.jsonl does.
    """
    lab = events[["timestamp", "event_code", "parameter"]]
    drift, sets = decode_clock_marks(lab, PEDS, send_log)
    if host is not None:
        lbl = host["label"].to_numpy()
        ts = drift["ts"].to_numpy(dtype=float)
        i = np.clip(np.searchsorted(lbl, ts), 1, len(lbl) - 1)
        i = np.where(np.abs(lbl[i - 1] - ts) < np.abs(lbl[i] - ts), i - 1, i)
        near = np.abs(lbl[i] - ts) < 1.0
        drift["drift_host"] = np.where(near, host["drift"].to_numpy()[i], np.nan)
    gaps = lab.loc[lab["event_code"] == -1, ["timestamp", "parameter"]]
    window = (lab["timestamp"].min(), lab["timestamp"].max() + 1)
    model = drift_model(drift, sets, gaps, window)
    return apply_true_time(events, model), model, sets


def real_rows(out):
    return out.loc[out["event_code"] != -1]


def bounds(model, col="opened_by"):
    """Per segment: its first piece's opened_by (or last piece's closed_by)."""
    g = model.groupby("segment", sort=True)[col]
    return (g.first() if col == "opened_by" else g.last()).tolist()


def segment_rates(model):
    """Per segment spanning over 6 h, the ppm of a line through its knots."""
    return np.array([
        np.polyfit(g["t_ref"], g["intercept"], 1)[0] * 1e6
        for _, g in model.groupby("segment") if g["span"].iloc[0] > 6 * 3600
    ])


# ---------------------------------------------------------------------------
# Golden: simulated controllers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fast():
    # 40 ppm fast (~3.5 s/day), so every daily set steps backward.
    c = Controller(d0=1.2, ppm=40, seed=1).run()
    out, model, sets = correct(c.log())
    return c, out, model, sets


class TestKnownClock:

    def test_sets_decode_and_are_backward(self, fast):
        c, _, _, sets = fast
        assert sets["shift"].tolist() == c.true_shift
        assert all(s < 0 for s in c.true_shift)

    def test_every_mapped_event_lands_on_its_true_instant(self, fast):
        # Pulse widths only: interpolation carries their ~0.15 s noise.
        _, out, _, _ = fast
        err = (real_rows(out)["timestamp"] - real_rows(out)["true"]).abs()
        assert err.max() < 0.4
        assert err.median() < 0.12

    def test_host_timed_samples_tighten_it(self, fast):
        c = fast[0]
        out, _, _ = correct(c.log(), host=c.host_log())
        err = (real_rows(out)["timestamp"] - real_rows(out)["true"]).abs()
        assert err.max() < 0.2
        assert err.median() < 0.06

    def test_correction_is_needed(self, fast):
        c, out, _, _ = fast
        raw = c.log()
        assert (raw["timestamp"] - raw["true"]).abs().max() > 2.5

    def test_knots_follow_the_ppm(self, fast):
        _, _, model, _ = fast
        rates = segment_rates(model)
        assert len(rates) >= 2
        assert np.allclose(rates, 40, atol=6)

    def test_almost_everything_maps(self, fast):
        c, out, _, _ = fast
        lost = 1 - len(real_rows(out)) / (c.log()["event_code"] != -1).sum()
        assert lost < 0.002  # the dead zones are seconds per day

    def test_output_is_true_time_ordered(self, fast):
        _, out, _, _ = fast
        assert out["timestamp"].is_monotonic_increasing

    def test_model_columns(self, fast):
        _, _, model, _ = fast
        assert list(model.columns) == MODEL_COLUMNS
        assert (model["seg_start"].to_numpy()[1:] >= model["seg_end"].to_numpy()[:-1]).all()
        # Pieces of a segment are contiguous and the drift is continuous.
        same = model["segment"].to_numpy()[1:] == model["segment"].to_numpy()[:-1]
        end_d = model["intercept"] + model["slope"] * (model["seg_end"] - model["t_ref"])
        assert np.allclose(model["seg_start"].to_numpy()[1:][same], model["seg_end"].to_numpy()[:-1][same])
        assert np.allclose(model["intercept"].to_numpy()[1:][same], end_d.to_numpy()[:-1][same])


class TestLineFrequencyClock:
    """A clock on the 60 Hz line wanders by up to a few hundred ppm either
    way (701, 2026-10-06/07).  Interpolating the hourly host-timed samples
    follows it; one line per segment could not."""

    @staticmethod
    def wander(x):
        return 1.0 * np.sin(2 * np.pi * x / (8 * 3600)) + 0.25 * np.sin(2 * np.pi * x / (4 * 3600) + 1)

    def test_interpolation_tracks_a_wandering_clock(self):
        c = Controller(d0=0.0, ppm=0, seed=7, wander=self.wander).run(shared=False)
        log = c.log()
        out, model, _ = correct(log, host=c.host_log())
        # Between samples; before the first (00:37) and after the last
        # (47:37) the drift is held and the wander is not followed.
        kept = real_rows(out)
        kept = kept.loc[kept["true"].between(T0 + 3600, T0 + 47.5 * 3600)]
        err = (kept["timestamp"] - kept["true"]).abs()
        assert err.max() < 0.3
        assert err.median() < 0.08
        assert np.nanmax(model["resid_mad"]) < 0.25

        # The best single line per segment misses by well over a second.
        ev = log.dropna(subset=["true"])
        lab = ev["timestamp"].to_numpy()
        drift = lab - ev["true"].to_numpy()
        k = np.searchsorted(model["seg_start"].to_numpy(), lab, side="right") - 1
        seg = np.where(k >= 0, model["segment"].to_numpy()[np.maximum(k, 0)], -1)
        misses = []
        for s_id in np.unique(seg[seg >= 0]):
            m = seg == s_id
            if np.ptp(lab[m]) > 6 * 3600:
                fit = np.polyval(np.polyfit(lab[m], drift[m], 1), lab[m])
                misses.append(np.abs(drift[m] - fit).max())
        assert max(misses) > 1.0


class TestTwoControllersAgree:
    """S2's done criterion, on a shared stimulus: after correction both
    controllers time it within the drift-sample noise."""

    @pytest.mark.parametrize("a, b", [
        ((1.2, 40, 11), (-0.9, -30, 12)),   # one fast, one slow
        ((0.4, 5, 13), (-2.5, 60, 14)),     # starts far apart
    ])
    @pytest.mark.parametrize("host, tol_max, tol_med", [
        (False, 0.5, 0.15),  # pulse widths only: interpolation carries their noise
        (True, 0.2, 0.06),   # with the head unit's host-timed measurements
    ])
    def test_shared_events_agree(self, a, b, host, tol_max, tol_med):
        out = {}
        raw = {}
        for name, (d0, ppm, seed) in zip("ab", (a, b)):
            c = Controller(d0=d0, ppm=ppm, seed=seed).run()
            log = c.log()
            mapped, _, _ = correct(log, host=c.host_log() if host else None)
            pick = lambda f: f.loc[(f["event_code"] == SHARED_CODE) & (f["parameter"] == SHARED_PARAM)]
            slot = lambda f: np.round((f["true"] - T0 - 3.3) / 10).astype(int)
            out[name] = pick(mapped).set_index(slot(pick(mapped)))["timestamp"]
            raw[name] = pick(log).set_index(slot(pick(log)))["timestamp"]
        both = out["a"].index.intersection(out["b"].index)
        assert len(both) > 0.99 * len(shared_times(2))
        diff = (out["a"][both] - out["b"][both]).abs()
        assert diff.max() < tol_max
        assert diff.median() < tol_med
        uncorrected = (raw["a"][both] - raw["b"][both]).abs()
        assert uncorrected.max() > 3.0


class TestBackwardSetBand:

    def test_band_is_flagged_never_mixed(self):
        # Dense traffic through a -3 s set: the replayed band's events are
        # dropped, not assigned to either side, and a marker fences the hole.
        c = Controller(d0=3.2, ppm=0, seed=3, days=1).run()
        log = c.log()
        out, _, sets = correct(log)
        (shift,) = sets["shift"].tolist()
        assert shift == -3
        assert (real_rows(out)["timestamp"] - real_rows(out)["true"]).abs().max() < 0.3

        step_true = c.steps[0][0]
        near = (log["true"] > step_true - 10) & (log["true"] < step_true + 10)
        replayed = near & log["timestamp"].between(
            log.loc[near & (log["true"] >= step_true), "timestamp"].min(),
            log.loc[near & (log["true"] < step_true), "timestamp"].max(),
        )
        assert replayed.sum() > 3
        assert not out["true"].isin(log.loc[replayed, "true"]).any()

        markers = out.loc[out["event_code"] == -1]
        hole = markers.loc[(markers["timestamp"] > step_true - 30)
                           & (markers["timestamp"] < step_true + 30)]
        assert hole["parameter"].tolist() == [COMMS_GAP_PARAM]

    def test_forward_set_still_gets_a_marker(self):
        c = Controller(d0=-2.3, ppm=0, seed=4, days=1).run()
        out, _, sets = correct(c.log())
        assert sets["shift"].tolist() == [2]
        step_true = c.steps[0][0]
        markers = out.loc[out["event_code"] == -1, "timestamp"]
        assert ((markers > step_true - 30) & (markers < step_true + 30)).sum() == 1


class TestUnmarkedBreaks:

    def test_unmarked_fence_stops_the_model_until_the_next_sample(self):
        # A keypad set back 5 s at 14:05; the next drift check is 14:37.
        t_key = T0 + 14 * 3600 + 5 * 60
        c = Controller(d0=0.3, ppm=0, seed=5, days=1).run(extra_steps=[(t_key, -5.0)])
        out, model, _ = correct(c.log())
        assert "fence" in model["closed_by"].tolist()
        kept = real_rows(out)
        assert (kept["timestamp"] - kept["true"]).abs().max() < 0.3
        dark = kept["true"].between(t_key, T0 + 14 * 3600 + 37 * 60 - 1)
        assert not dark.any()
        assert kept["true"].between(t_key - 600, t_key - 1).sum() > 100
        assert kept["true"].between(T0 + 14 * 3600 + 38 * 60, T0 + 15 * 3600).sum() > 100

    def test_comms_gap_restarts_at_the_next_sample(self):
        c = Controller(d0=0.5, ppm=20, seed=6, days=1).run()
        log = c.log()
        t_gap = T0 + 16 * 3600 + 10 * 60
        lost = log["true"].between(t_gap, t_gap + 120)
        log = log.loc[~lost]
        gap_label = log.loc[log["true"] < t_gap, "timestamp"].max() + 0.1
        log = pd.concat([log, pd.DataFrame([{
            "timestamp": gap_label, "event_code": -1, "parameter": COMMS_GAP_PARAM, "true": np.nan,
        }])]).sort_values("timestamp").reset_index(drop=True)
        out, model, _ = correct(log)
        assert "gap" in model["closed_by"].tolist()
        kept = real_rows(out)
        assert not kept["true"].between(t_gap, T0 + 16 * 3600 + 37 * 60 - 1).any()
        assert (kept["timestamp"] - kept["true"]).abs().max() < 0.3


# ---------------------------------------------------------------------------
# Unit: zones, fit limits, mapping
# ---------------------------------------------------------------------------

def _drift(ts, d, role="check", status="ok"):
    n = len(ts)
    return pd.DataFrame({
        "ts": ts, "off": np.asarray(ts) + 0.1, "ped": PEDS.ahead, "width": 0.1,
        "drift": d, "drift_lo": np.nan, "drift_hi": np.nan, "saturated": False,
        "role": [role] * n if isinstance(role, str) else role,
        "drift_host": np.nan, "status": [status] * n if isinstance(status, str) else status,
    })


def _sets(rows):
    cols = ["bracket_on", "bracket_off", "width", "drift_pre", "shift_est",
            "period", "shift", "step_lo", "step_hi", "shift_host", "status"]
    return pd.DataFrame(rows, columns=cols)


NO_GAPS = pd.DataFrame(columns=["timestamp", "parameter"])


class TestZones:

    def test_set_crossing_a_minute_darkens_the_whole_minute(self):
        # +3 s with the step somewhere in :50.9-:59.5 lands post-step labels
        # in :53.9-:02.5 of the next minute; the seconds edit can leave
        # labels anywhere in the :17 minute first.
        minute = T0 + 9 * 3600 + 17 * 60
        on = minute + 49.5
        sets = _sets([(on, on + 13, 13, -2.9, 2.9, 10, 3, on + 1.4, on + 10, np.nan, "ok")])
        drift = _drift([on - 3, on + 13.2, on + 1800], [-2.9, 0.1, 0.1],
                       role=["pre_set", "residual", "check"])
        model = drift_model(drift, sets, NO_GAPS, (T0, T0 + DAY))
        dark = np.array([minute + 0.5, minute + 59.5, minute + 62.4])
        assert np.isnan(to_true_time(dark, model)).all()
        assert np.isfinite(to_true_time(np.array([minute - 0.5, minute + 62.6]), model)).all()

    def test_set_inside_one_minute_darkens_only_its_window(self):
        on = T0 + 9 * 3600 + 17 * 60 + 10.0
        sets = _sets([(on, on + 8, 8, 2.1, -2.1, 10, -2, on + 1.4, on + 10, np.nan, "ok")])
        drift = _drift([on - 2.3, on + 8.2], [2.1, 0.1], role=["pre_set", "residual"])
        model = drift_model(drift, sets, NO_GAPS, (T0, T0 + DAY))
        assert model["seg_end"].iloc[0] == pytest.approx(on + 1.4 - 2)
        assert model["seg_start"].iloc[1] == pytest.approx(on + 10)

    def test_decoded_set_anchors_the_next_segment(self):
        # No residual pulse and no check for an hour: the anchor
        # drift_pre + shift still maps the post-set labels.
        on = T0 + 9 * 3600 + 17 * 60 + 10.0
        sets = _sets([(on, on + 8, 8, 2.1, -2.1, 10, -2, on + 1.4, on + 10, np.nan, "ok")])
        drift = _drift([on - 2.3], [2.1], role=["pre_set"])
        model = drift_model(drift, sets, NO_GAPS, (T0, T0 + DAY))
        assert bounds(model)[1:] == ["set"]
        assert to_true_time(np.array([on + 20]), model)[0] == pytest.approx(on + 20 - 0.1)

    def test_undecoded_set_is_strict(self):
        on = T0 + 9 * 3600 + 17 * 60 + 10.0
        sets = _sets([(on, on + 8, 8, np.nan, np.nan, np.nan, np.nan, on + 1.4, np.nan, np.nan, "no_pre_pulse")])
        drift = _drift([on - 3600, on + 1800], [2.1, 0.1])
        model = drift_model(drift, sets, NO_GAPS, (T0, T0 + DAY))
        assert bounds(model)[1:] == ["set_undecoded"]
        seg = model.groupby("segment").agg(lo=("seg_start", "min"), hi=("seg_end", "max"))
        assert seg["lo"].iloc[1] == pytest.approx(on + 1800)
        assert seg["hi"].iloc[0] == pytest.approx(on - 60)

    def test_fence_explained_by_a_set_does_not_break_strictly(self):
        on = T0 + 9 * 3600 + 17 * 60 + 10.0
        sets = _sets([(on, on + 8, 8, 2.1, -2.1, 10, -2, on + 1.4, on + 10, np.nan, "ok")])
        drift = _drift([on - 2.3], [2.1], role=["pre_set"])
        gaps = pd.DataFrame({"timestamp": [on + 2.0 - 2 - 0.05], "parameter": [CLOCK_STEP_FENCE_PARAM]})
        model = drift_model(drift, sets, gaps, (T0, T0 + DAY))
        assert bounds(model)[1:] == ["set"]

    def test_saturated_and_unpaired_samples_are_not_used(self):
        drift = _drift([T0 + 100, T0 + 200, T0 + 300], [np.nan, 5.0, 0.2],
                       status=["saturated", "unpaired_on", "ok"])
        model = drift_model(drift, _sets([]), NO_GAPS, (T0, T0 + DAY))
        assert model["n_samples"].unique().tolist() == [1]
        assert to_true_time(np.array([T0 + 300]), model)[0] == pytest.approx(T0 + 300 - 0.2)

    def test_send_log_measurement_is_preferred(self):
        drift = _drift([T0 + 100], [0.25]).assign(drift_host=0.31)
        model = drift_model(drift, _sets([]), NO_GAPS, (T0, T0 + DAY))
        assert model["intercept"].iloc[0] == pytest.approx(0.31)


class TestFitLimits:

    def test_interpolates_between_samples_and_holds_beyond(self):
        drift = _drift([T0 + 3600, T0 + 7200, T0 + 10800], [0.1, 0.5, 0.2])
        model = drift_model(drift, _sets([]), NO_GAPS, (T0, T0 + DAY))
        t = T0 + np.array([1800, 3600, 5400, 9000, 12600])
        assert to_true_time(t, model) == pytest.approx(t - np.array([0.1, 0.1, 0.3, 0.35, 0.2]))
        assert model["resid_mad"].iloc[0] == pytest.approx(0.35)

    def test_drops_a_spike_keeps_a_swing(self):
        ts = T0 + 3600 * np.arange(6)
        d = np.array([0.0, -1.1, -2.7, -2.2, 5.0, -1.7])  # -2.7: real; 5.0: misdecoded
        model = drift_model(_drift(ts, d), _sets([]), NO_GAPS, (T0, T0 + DAY))
        got = ts - to_true_time(ts, model)
        assert got[:4] == pytest.approx(d[:4])
        assert got[4] == pytest.approx(-1.95)

    def test_samples_sharing_a_label_are_averaged(self):
        drift = _drift([T0 + 100, T0 + 100], [0.2, 0.4])
        model = drift_model(drift, _sets([]), NO_GAPS, (T0, T0 + DAY))
        assert to_true_time(np.array([T0 + 100]), model)[0] == pytest.approx(T0 + 99.7)

    def test_extrapolation_is_bounded(self):
        drift = _drift([T0 + 10 * 3600], [0.4])
        model = drift_model(drift, _sets([]), NO_GAPS, (T0, T0 + DAY))
        assert model["seg_start"].min() == pytest.approx(T0 + 8 * 3600)
        assert model["seg_end"].max() == pytest.approx(T0 + 12 * 3600)
        assert bounds(model) == ["extrapolation"]
        assert bounds(model, "closed_by") == ["extrapolation"]

    def test_no_samples_no_model(self):
        model = drift_model(_drift([], []), _sets([]), NO_GAPS, (T0, T0 + DAY))
        assert model.empty
        assert np.isnan(to_true_time(np.array([T0 + 5]), model)).all()


class TestApply:

    def test_extra_columns_are_mapped_by_their_own_label(self):
        drift = _drift([T0 + 100], [0.5])
        model = drift_model(drift, _sets([]), NO_GAPS, (T0, T0 + 3600))
        df = pd.DataFrame({"timestamp": [T0 + 200.0], "event_code": [82], "parameter": [3],
                           "cycle_start": [T0 + 150.0], "coord_plan": [1.0]})
        out = apply_true_time(df, model, extra_columns=("cycle_start",))
        assert out["timestamp"].iloc[0] == pytest.approx(T0 + 199.5)
        assert out["cycle_start"].iloc[0] == pytest.approx(T0 + 149.5)

    def test_marker_takes_the_previous_rows_other_columns(self):
        on = T0 + 9 * 3600 + 17 * 60 + 10.0
        sets = _sets([(on, on + 8, 8, 2.1, -2.1, 10, -2, on + 1.4, on + 10, np.nan, "ok")])
        drift = _drift([on - 2.3, on + 8.2], [2.1, 0.1], role=["pre_set", "residual"])
        model = drift_model(drift, sets, NO_GAPS, (T0, T0 + DAY))
        df = pd.DataFrame({
            "timestamp": [on - 5.0, on + 3.0, on + 20.0], "event_code": [82, 82, 82],
            "parameter": [3, 3, 3], "coord_plan": [4.0, 4.0, 5.0],
        })
        out = apply_true_time(df, model)
        assert out["event_code"].tolist() == [82, -1, 82]
        assert out["coord_plan"].tolist() == [4.0, 4.0, 5.0]
        assert out["timestamp"].is_monotonic_increasing

    def test_empty_frame(self):
        empty = pd.DataFrame(columns=["timestamp", "event_code", "parameter"])
        assert apply_true_time(empty, pd.DataFrame(columns=MODEL_COLUMNS)).empty


# ---------------------------------------------------------------------------
# Bench (golden): the model against the head unit's host-timed record
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bench_events(tmp_path_factory) -> pd.DataFrame:
    root = tmp_path_factory.mktemp("bench_tt")
    raw = root / "raw_data"
    raw.mkdir()
    for f in sorted(BENCH.glob("*.datZ")):
        (raw / f.name).write_bytes(f.read_bytes())
    db = root / "bench.db"
    with DatabaseManager(db) as m:
        m.init_db()
    IngestionEngine(db, raw, timezone="US/Mountain").run()
    with DatabaseManager(db) as m:
        return pd.read_sql("SELECT timestamp, event_code, parameter FROM events", m.conn)


def _drift_on(row, t):
    return row["intercept"] + row["slope"] * (t - row["t_ref"])


class TestBench:

    def test_set_anchor_predicts_the_host_measured_residual(self, bench_events):
        # drift_pre + shift, carried past each set with the residual pulse
        # withheld, against the residual the head unit timed on its own
        # clock.  Each run is decoded on its own slice: the bench sets run
        # 20-60 s apart, so their zones would merge.
        with open(BENCH / "bench.jsonl") as fh:
            runs = [json.loads(line) for line in fh if '"mode": "set"' in line]
        assert len(runs) == 4
        log = send_log_pulses(runs)
        for run in runs:
            pre, _, res = run["pulses"]
            lo = pre["on_epoch"] + pre["drift_s"] - 1
            hi = res["off_epoch"] + run["after"]["drift"] + 1
            ev = bench_events.loc[bench_events["timestamp"].between(lo, hi)]
            drift, sets = decode_clock_marks(ev, PEDS, log)
            assert sets["shift"].tolist() == [run["shift_s"]]
            drift = drift.loc[drift["role"] != "residual"]
            model = drift_model(drift, sets, NO_GAPS, (lo, lo + 600))
            post = model.loc[model["opened_by"] == "set"].iloc[0]
            res_label = res["on_epoch"] + run["after"]["drift"]
            assert _drift_on(post, res_label) == pytest.approx(run["after"]["drift"], abs=0.2)
