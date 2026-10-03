"""Golden tests for the detector-health rule core (S-D2)."""

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import pytz

from atspm.analysis.detector_activity import UDOT_AM_WINDOW, detector_activity_profile
from atspm.analysis.detector_health import (
    BURST_SCHEMA,
    DEFAULT_THRESHOLDS,
    FINDINGS_SCHEMA,
    chatter,
    configured_silent,
    controller_fault_findings,
    detector_health_findings,
    failsafe_findings,
    low_detector_hits,
    onset_bursts,
    stuck_on,
    unconfigured_detector,
)
from atspm.analysis.detector_roles import parse_detector_roles

TZ = "US/Mountain"
_MT = pytz.timezone(TZ)
D = dt.date(2026, 6, 10)
DAY = 86400.0


def loc(s: str) -> float:
    return _MT.localize(pd.Timestamp(s).to_pydatetime(), is_dst=None).timestamp()


def ev(rows):
    return pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])


def prof(rows, window="day", bin_s=DAY):
    """Profile rows from dicts of overrides; defaults are a healthy full day."""
    base = dict(date=D, window=window, bin_start=0.0, bin_end=bin_s, bin_s=bin_s,
                observed_s=bin_s, configured=True, n_act=500, n_censored=0,
                on_time_s=3000.0, occupancy=3000.0 / bin_s, max_on_s=30.0, p95_on_s=5.0,
                n_short=10, min_off_gap_s=1.0, open_on_s=np.nan,
                on_at_start=False, on_at_end=False)
    return pd.DataFrame([{**base, **r} for r in rows])


# 2/6 stop-bar loops, P2 presence zones, a TM movement and a watchdog zone.
ROLES = parse_detector_roles({
    "Det_P2_Stop_Bar": "1,2,3",
    "Det_P2_Occupancy": "10,11",
    "Det_P6_Arrival": "20",
    "TM_NBT": "1,2,3",
    "WD_Sensor1": "56",
})


def one(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    assert len(sel) == 1, sel
    return sel.iloc[0]


class TestConfiguredSilent:
    def test_flags_silent_configured_only(self):
        p = prof([
            {"detector": 1, "n_act": 0, "on_time_s": 0.0},
            {"detector": 2},
            {"detector": 56, "n_act": 0, "on_time_s": 0.0},          # watchdog: meant to be silent
            {"detector": 40, "configured": False, "n_act": 0, "on_time_s": 0.0},
        ])
        f = configured_silent(p, ROLES)
        assert list(f.columns) == FINDINGS_SCHEMA
        r = one(f, detector=1)
        assert (r.rule, r.severity, r.role, r.phase) == ("ConfiguredSilent", "high", "stop_bar+tm", 2)
        assert len(f) == 1

    def test_stuck_on_is_not_silent(self):
        p = prof([{"detector": 1, "n_act": 0, "on_time_s": DAY, "occupancy": 1.0}])
        assert configured_silent(p, ROLES).empty

    def test_partial_day_not_judged(self):
        p = prof([{"detector": 1, "n_act": 0, "on_time_s": 0.0, "observed_s": 0.85 * DAY}])
        assert configured_silent(p, ROLES).empty
        p = prof([{"detector": 1, "n_act": 0, "on_time_s": 0.0, "observed_s": 0.9 * DAY}])
        assert len(configured_silent(p, ROLES)) == 1

    def test_only_day_window(self):
        p = prof([{"detector": 1, "n_act": 0, "on_time_s": 0.0}], window="am", bin_s=4 * 3600.0)
        assert configured_silent(p, ROLES).empty


class TestLowHitsAndUnconfigured:
    def test_low_hits(self):
        p = prof([{"detector": 1, "n_act": 19}, {"detector": 2, "n_act": 20},
                  {"detector": 3, "n_act": 0, "on_time_s": 0.0}])
        f = low_detector_hits(p, ROLES)
        r = one(f, detector=1)
        assert (r.severity, r.value, r.threshold) == ("low", 19.0, 20.0)
        assert len(f) == 1  # 20 is not low; 0 is ConfiguredSilent's

    def test_unconfigured(self):
        p = prof([{"detector": 40, "configured": False, "n_act": 20},
                  {"detector": 41, "configured": False, "n_act": 19},
                  {"detector": 1, "n_act": 900}])
        f = unconfigured_detector(p, ROLES)
        r = one(f, detector=40)
        assert (r.rule, r.severity, r.role) == ("UnconfiguredDetector", "info", "unconfigured")
        assert len(f) == 1

    def test_unconfigured_needs_no_full_day(self):
        p = prof([{"detector": 40, "configured": False, "n_act": 300, "observed_s": 0.3 * DAY}])
        assert len(unconfigured_detector(p, ROLES)) == 1


class TestStuckOn:
    def test_role_limits(self):
        # stop_bar+tm: 15 min; occupancy: 30 min; unconfigured: default 30 min.
        p = prof([
            {"detector": 1, "max_on_s": 16 * 60.0},
            {"detector": 10, "max_on_s": 16 * 60.0},
            {"detector": 11, "max_on_s": 31 * 60.0},
            {"detector": 40, "configured": False, "max_on_s": 29 * 60.0},
        ])
        f = stuck_on(p, ROLES)
        assert sorted(f["detector"]) == [1, 11]
        assert one(f, detector=1).threshold == 900.0
        assert one(f, detector=11).threshold == 1800.0

    def test_censored_lower_bound_counts(self):
        p = prof([{"detector": 1, "max_on_s": np.nan, "open_on_s": 2 * 3600.0, "n_censored": 1}])
        r = one(stuck_on(p, ROLES), detector=1)
        assert r.value == 7200.0 and "lower bound" in r.message

    def test_held_window_after_onset_day(self):
        # Day 2 of a long stuck-on: no onset, held all day; and the AM window.
        p = pd.concat([
            prof([{"detector": 2, "n_act": 0, "on_time_s": DAY, "occupancy": 1.0,
                   "max_on_s": np.nan}]),
            prof([{"detector": 2, "n_act": 0, "on_time_s": 14400.0, "occupancy": 1.0,
                   "max_on_s": np.nan}], window="am", bin_s=14400.0),
        ])
        f = stuck_on(p, ROLES)
        assert sorted(f["window"]) == ["am", "day"]
        assert (f["value"] == 1.0).all()

    def test_held_dropped_when_duration_fires(self):
        p = pd.concat([
            prof([{"detector": 2, "max_on_s": 5 * 3600.0}]),
            prof([{"detector": 2, "occupancy": 1.0}], window="am", bin_s=14400.0),
        ])
        f = stuck_on(p, ROLES)
        assert len(f) == 1 and one(f, detector=2).window == "day"

    def test_held_needs_judgeable_bin(self):
        p = prof([{"detector": 2, "occupancy": 1.0, "observed_s": 3600.0}], window="am",
                 bin_s=14400.0)
        assert stuck_on(p, ROLES).empty

    def test_watchdog_excluded(self):
        p = prof([{"detector": 56, "max_on_s": DAY / 2}])
        assert stuck_on(p, ROLES).empty


class TestChatter:
    def test_peer_relative_not_absolute(self):
        # 1,2,3 share stop_bar:P2 and tm:NBT.  All at 60 % short: no finding.
        p = prof([{"detector": d, "n_act": 1000, "n_short": 600} for d in (1, 2, 3)])
        assert chatter(p, ROLES).empty

    def test_flags_outlier(self):
        p = prof([
            {"detector": 1, "n_act": 1000, "n_short": 400, "min_off_gap_s": 0.1},
            {"detector": 2, "n_act": 1000, "n_short": 100},
            {"detector": 3, "n_act": 1000, "n_short": 120},
        ])
        f = chatter(p, ROLES)
        r = one(f, detector=1)
        assert r.severity == "low"  # off-gap is reported, never raises it
        assert "min off-gap 0.1 s" in r.message
        assert r.value == pytest.approx(0.4)
        assert r.threshold == pytest.approx(3 * 0.11)
        assert len(f) == 1

    def test_floor_and_min_short(self):
        # Peers at 0 %: the floor (2 %) sets the bar at 6 %, and 50 short pulses minimum.
        p = prof([
            {"detector": 1, "n_act": 600, "n_short": 49},
            {"detector": 2, "n_act": 1000, "n_short": 0},
        ])
        assert chatter(p, ROLES).empty
        p.loc[p["detector"] == 1, "n_short"] = 50
        assert len(chatter(p, ROLES)) == 1

    def test_censored_excluded_from_denominator_and_peers_need_volume(self):
        p = prof([
            {"detector": 10, "n_act": 300, "n_censored": 200, "n_short": 60},  # 100 measured
            {"detector": 11, "n_act": 49, "n_short": 0},  # too few to be a peer
        ])
        assert chatter(p, ROLES).empty

    def test_most_lenient_group(self):
        # 1 is in stop_bar:P2 with 2 (5 %) and tm:NBT with 2 only... add 3 at 30 % to both.
        p = prof([
            {"detector": 1, "n_act": 1000, "n_short": 500},
            {"detector": 2, "n_act": 1000, "n_short": 50},
            {"detector": 3, "n_act": 1000, "n_short": 300},
        ])
        # peer median for 1 = median(5 %, 30 %) = 17.5 %; limit 52.5 % > 50 %.
        assert chatter(p, ROLES).empty


def burst_events(t0, channels, hold_s, code_on=82):
    rows = [(t0, code_on, c) for c in channels]
    rows += [(t0 + hold_s, 81, c) for c in channels]
    return rows


def background(start, end, det=99, step=60.0):
    ts = np.arange(loc(start), loc(end), step)
    return [(t, 82, det) for t in ts] + [(t + 0.3, 81, det) for t in ts]


class TestOnsetBursts:
    def test_basic_burst_and_release(self):
        t0 = loc("2026-06-10 01:55:12.2")
        rows = background("2026-06-10 01:00", "2026-06-10 03:00")
        rows += burst_events(t0, range(1, 17), 70.0)
        b = onset_bursts(ev(rows), min_channels=6)
        assert list(b.columns) == BURST_SCHEMA
        r = one(b, n_channels=16)
        assert r.ts == pytest.approx(t0)
        assert r.channels == tuple(range(1, 17))
        assert r.released_s == pytest.approx(70.0)
        assert (r.unit, r.n_censored, r.reasserted) == ("all", 0, False)

    def test_reassertion(self):
        # Channels on before, logged off then on again 0.4 s later.
        t0 = loc("2026-06-10 13:00:01.2")
        chans = range(30, 38)
        rows = [(t0 - 20.0 - c, 82, c) for c in chans]
        rows += [(t0 - 0.4, 81, c) for c in chans]
        rows += burst_events(t0, chans, 25.0)
        r = one(onset_bursts(ev(rows), 6), n_channels=8)
        assert r.reasserted and r.n_reasserted == 8

    def test_reassert_not_across_gap(self):
        t0 = loc("2026-06-10 13:00:01.2")
        chans = range(30, 38)
        rows = [(t0 - 0.4, 81, c) for c in chans] + [(t0 - 0.2, -1, 0)]
        rows += burst_events(t0, chans, 25.0)
        assert not one(onset_bursts(ev(rows), 6), n_channels=8).reasserted

    def test_release_censored_by_gap(self):
        t0 = loc("2026-06-10 23:59:07.5")
        rows = burst_events(t0, range(1, 17), 100.0)
        rows += [(t0 + 50.0, -1, 0)]
        r = one(onset_bursts(ev(rows), 6), n_channels=16)
        assert np.isnan(r.released_s) and r.n_censored == 16

    def test_release_median_with_some_censored(self):
        t0 = loc("2026-06-10 02:00")
        rows = burst_events(t0, range(1, 7), 30.0)          # 6 released at 30 s
        rows += [(t0, 82, c) for c in range(7, 10)]          # 3 never released
        r = one(onset_bursts(ev(rows), 6), n_channels=9)
        assert r.released_s == pytest.approx(30.0) and r.n_censored == 3

    def test_units_split_and_min_channels(self):
        t0 = loc("2026-06-10 02:00")
        rows = burst_events(t0, range(1, 9), 30.0) + burst_events(t0, range(17, 21), 30.0)
        units = {"biu2": range(1, 17), "biu3": range(17, 33)}
        b = onset_bursts(ev(rows), 6, units=units)
        assert list(b["unit"]) == ["biu2"]
        b = onset_bursts(ev(rows), 4, units=units)
        assert list(b["unit"]) == ["biu2", "biu3"]
        b = onset_bursts(ev(rows), 12)
        assert one(b, unit="all").n_channels == 12

    def test_decisecond_grouping(self):
        t0 = loc("2026-06-10 02:00")
        rows = [(t0 + 1e-7 * c, 82, c) for c in range(1, 7)] + [(t0 + 0.1, 82, 7)]
        assert one(onset_bursts(ev(rows), 6), n_channels=6).channels == tuple(range(1, 7))

    def test_empty(self):
        assert list(onset_bursts(ev([]), 6).columns) == BURST_SCHEMA
        assert onset_bursts(ev([(1.0, 82, 1)]), 6).empty


class TestFailsafe:
    REBOOT = {"nightly": ("23:58", "24:00"), "early": ("01:54", "01:57")}

    def _labelled_201(self, release=70.0):
        """The 201 2026-03-19 signature: 1–16 at 12.2 s, 17–32 at 15.0 s, max-outs before."""
        t1 = loc("2026-06-10 01:55:12.2")
        rows = background("2026-06-10 01:00", "2026-06-10 03:00")
        rows += burst_events(t1, range(1, 17), release)
        rows += burst_events(t1 + 2.8, range(17, 33), release / 2)
        rows += [(t1 - 9.0, 5, 2), (t1 - 9.0, 5, 6), (t1 + 40.0, 5, 4)]
        return ev(rows), t1

    def test_episode_merge_and_severity_high_without_window(self):
        e, t1 = self._labelled_201()
        f = failsafe_findings(e, TZ, ROLES)
        r = one(f, rule="Failsafe")
        assert (r.severity, r.value, r.detector, r.ts) == ("high", 32.0, -1, pytest.approx(t1))
        assert "2 bursts" in r.message and "phases maxed out: 2, 4, 6" in r.message

    def test_info_inside_reboot_window(self):
        e, _ = self._labelled_201()
        r = one(failsafe_findings(e, TZ, ROLES, reboot_windows=self.REBOOT), rule="Failsafe")
        assert r.severity == "info" and "reboot window 'early'" in r.message

    def test_long_release_in_window_is_high(self):
        e, _ = self._labelled_201(release=500.0)
        r = one(failsafe_findings(e, TZ, ROLES, reboot_windows=self.REBOOT), rule="Failsafe")
        assert r.severity == "high"

    def test_unobserved_release_in_window_is_low(self):
        t0 = loc("2026-06-10 23:59:07.5")
        rows = background("2026-06-10 23:00", "2026-06-10 23:59") + burst_events(t0, range(1, 17), 80.0)
        rows += [(t0 + 30.0, -1, 0)]
        r = one(failsafe_findings(ev(rows), TZ, ROLES, reboot_windows=self.REBOOT), rule="Failsafe")
        assert r.severity == "low" and "release not observed" in r.message

    def test_reasserted_and_small_bursts_excluded(self):
        t0 = loc("2026-06-10 13:00:01.2")
        chans = range(1, 33)
        rows = [(t0 - 0.4, 81, c) for c in chans] + burst_events(t0, chans, 25.0)
        rows += burst_events(loc("2026-06-10 15:00"), range(1, 11), 6.3)  # 10 < 12
        assert failsafe_findings(ev(rows), TZ, ROLES).empty

    def test_window_is_local_time_across_dst(self):
        # Winter (MST) burst at 23:59 local is inside 23:58–24:00.
        t0 = _MT.localize(pd.Timestamp("2026-01-10 23:59:06").to_pydatetime()).timestamp()
        rows = burst_events(t0, range(1, 17), 60.0)
        r = one(failsafe_findings(ev(rows), TZ, ROLES, reboot_windows=self.REBOOT), rule="Failsafe")
        assert r.severity == "info" and r.date == dt.date(2026, 1, 10)

    def test_watchdog_corroborated_vs_lone(self):
        e, t1 = self._labelled_201()
        extra = ev([(t1 + 5.0, 82, 56), (t1 + 9.0, 81, 56),
                    (loc("2026-06-10 02:40"), 82, 56), (loc("2026-06-10 02:41"), 81, 56)])
        f = failsafe_findings(pd.concat([e, extra]), TZ, ROLES)
        assert "watchdog zones called: 56" in one(f, rule="Failsafe").message
        w = one(f, rule="WatchdogCall")
        assert (w.detector, w.severity, w.value) == (56, "low", 1.0)

    def test_units_fallback_and_partition(self):
        e, _ = self._labelled_201()
        units = {"biu2": range(1, 17), "biu3": range(17, 33)}
        f = failsafe_findings(e, TZ, ROLES, units=units)
        assert sorted(f["role"]) == ["unit:biu2", "unit:biu3"]
        assert sorted(f["value"]) == [16.0, 16.0]


class TestControllerFaults:
    def test_pass_through(self):
        t = loc("2026-06-10 08:00")
        e = ev([(t, 86, 5), (t + 60, 86, 5), (t + 120, 83, 5), (t, 91, 2), (t, 92, 2)])
        f = controller_fault_findings(e, TZ)
        assert len(f) == 3
        r = one(f, detector=5, severity="high")
        assert r.value == 2.0 and "open loop" in r.message and r.ts == t
        assert one(f, detector=5, severity="info").message.startswith("code 83")
        assert one(f, detector=2).role == "ped"


class TestEndToEnd:
    def test_profile_to_findings(self):
        """Real profile → rules: silent 3, stuck 10 across midnight, unconfigured 40."""
        rows = []
        for d in ("2026-06-10", "2026-06-11"):
            for det in (1, 2, 11, 20):
                ts = np.arange(loc(f"{d} 00:00"), loc(f"{d} 23:59"), 120.0)
                rows += [(t, 82, det) for t in ts] + [(t + 2.0, 81, det) for t in ts]
            ts = np.arange(loc(f"{d} 06:00"), loc(f"{d} 07:00"), 60.0)
            rows += [(t, 82, 40) for t in ts] + [(t + 1.0, 81, 40) for t in ts]
        rows += [(loc("2026-06-10 22:00"), 82, 10), (loc("2026-06-11 09:00"), 81, 10)]
        p = detector_activity_profile(
            ev(rows), TZ, ROLES, windows={"day": ("00:00", "24:00"), "am": UDOT_AM_WINDOW},
            start_date=D, end_date=D + dt.timedelta(days=1))
        f = detector_health_findings(p, ROLES, ev(rows), TZ)
        got = set(zip(f["date"], f["window"], f["detector"], f["rule"]))
        d10, d11 = D, D + dt.timedelta(days=1)
        assert got == {
            (d10, "day", 3, "ConfiguredSilent"), (d11, "day", 3, "ConfiguredSilent"),
            (d10, "day", 10, "StuckOn"),        # 11 h on, measured at the onset day
                                                # (its 1 actuation isn't LowDetectorHits)
            (d11, "am", 10, "StuckOn"),         # held through the next AM window
            (d10, "day", 40, "UnconfiguredDetector"), (d11, "day", 40, "UnconfiguredDetector"),
        }
        assert list(f.columns) == FINDINGS_SCHEMA

    def test_thresholds_override(self):
        p = prof([{"detector": 1, "n_act": 30}])
        assert low_detector_hits(p, ROLES).empty
        th = replace(DEFAULT_THRESHOLDS, low_hits_min=50)
        assert len(low_detector_hits(p, ROLES, th)) == 1

    def test_empty_everything(self):
        f = detector_health_findings(detector_activity_profile(ev([]), TZ), None, None, TZ)
        assert list(f.columns) == FINDINGS_SCHEMA and f.empty


def test_burst_release_censored_by_unmarked_silence():
    """201 2026-06-24 23:59:07.5: the reboot burst's offs come 98 days later."""
    t0 = loc("2026-06-24 23:59:07.5")
    rows = burst_events(t0, range(1, 17), 98 * DAY)
    r = one(onset_bursts(ev(rows), 6), n_channels=16)
    assert np.isnan(r.released_s) and r.n_censored == 16
