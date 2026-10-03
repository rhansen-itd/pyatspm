"""Golden tests for the per-detector activity profile (S-D1)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import pytz

from atspm.analysis.detector_activity import (
    ACTIVITY_SCHEMA,
    UDOT_AM_WINDOW,
    detector_activity_profile,
)

TZ = "US/Mountain"
_MT = pytz.timezone(TZ)
H = 3600.0


def loc(s: str) -> float:
    """Local wall-clock 'YYYY-MM-DD HH:MM[:SS.f]' → UTC epoch seconds."""
    naive = pd.Timestamp(s).to_pydatetime()
    return _MT.localize(naive, is_dst=None).timestamp()


def ev(rows):
    return pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])


def heartbeat(start: str, end: str, det: int = 99, step_s: float = 600.0):
    """A pulsing detector logging every *step_s* so the bins count as logged."""
    t0, t1 = loc(start), loc(end)
    ts = np.arange(t0, t1 + 1e-6, step_s)
    rows = [(t, 82, det) for t in ts] + [(t + 0.2, 81, det) for t in ts]
    return rows


def row(df, det, date, window="day"):
    sel = df[(df["detector"] == det) & (df["date"] == date) & (df["window"] == window)]
    assert len(sel) == 1
    return sel.iloc[0]


D10, D11, D12 = dt.date(2026, 6, 10), dt.date(2026, 6, 11), dt.date(2026, 6, 12)


class TestStuckOn:
    @pytest.fixture
    def prof(self):
        rows = heartbeat("2026-06-10 00:00", "2026-06-13 00:30")
        rows += [(loc("2026-06-10 02:00"), 82, 5), (loc("2026-06-11 06:00"), 81, 5)]
        # Stuck on into the end of the data: never logs an off.
        rows += [(loc("2026-06-12 20:00"), 82, 7)]
        return detector_activity_profile(
            ev(rows), TZ, windows={"day": ("00:00", "24:00"), "am": UDOT_AM_WINDOW},
            start_date=D10, end_date=D12)

    def test_schema(self, prof):
        assert list(prof.columns) == ACTIVITY_SCHEMA
        assert len(prof) == 3 * 2 * 3        # dates × windows × detectors

    def test_start_day_gets_interval_and_clipped_on_time(self, prof):
        r = row(prof, 5, D10)
        assert r.n_act == 1 and r.n_censored == 0
        assert r.max_on_s == pytest.approx(28 * H)
        assert r.p95_on_s == pytest.approx(28 * H)
        assert r.on_time_s == pytest.approx(22 * H)
        assert not r.on_at_start and r.on_at_end
        assert r.occupancy == pytest.approx(22 / 24)

    def test_next_day_on_time_without_actuation(self, prof):
        r = row(prof, 5, D11)
        assert r.n_act == 0 and np.isnan(r.max_on_s)
        assert r.on_time_s == pytest.approx(6 * H)
        assert r.on_at_start and not r.on_at_end

    def test_on_across_whole_am_window(self, prof):
        r = row(prof, 5, D11, "am")
        assert r.bin_s == pytest.approx(4 * H)
        assert r.n_act == 0 and r.occupancy == pytest.approx(1.0)
        assert r.on_at_start and r.on_at_end
        # The interval's own AM bin is D10 (02:00 lies in 01:00–05:00).
        assert row(prof, 5, D10, "am").n_act == 1

    def test_open_at_data_end_is_censored_not_measured(self, prof):
        r = row(prof, 7, D12)
        assert r.n_act == 1 and r.n_censored == 1
        assert np.isnan(r.max_on_s) and np.isnan(r.p95_on_s) and r.n_short == 0
        assert r.on_time_s == pytest.approx(4 * H)
        assert r.open_on_s == pytest.approx(4.5 * H + 0.2)   # observed until the last event
        assert r.on_at_end


class TestChatter:
    def test_short_pulses_and_off_gap(self):
        t0 = loc("2026-06-10 08:00")
        on = t0 + np.arange(100) * 1.0
        long_on = t0 + 200 + np.arange(10) * 10.0
        rows = [(t, 82, 3) for t in on] + [(t + 0.1, 81, 3) for t in on]
        rows += [(t, 82, 3) for t in long_on] + [(t + 2.0, 81, 3) for t in long_on]
        rows += heartbeat("2026-06-10 00:00", "2026-06-11 00:00")
        prof = detector_activity_profile(ev(rows), TZ, start_date=D10, end_date=D10)
        r = row(prof, 3, D10)
        assert r.n_act == 110 and r.n_short == 100
        assert r.min_off_gap_s == pytest.approx(0.9)
        assert r.max_on_s == pytest.approx(2.0)
        durs = np.r_[np.full(100, 0.1), np.full(10, 2.0)]
        assert r.p95_on_s == pytest.approx(np.quantile(durs, 0.95), abs=1e-4)
        assert r.on_time_s == pytest.approx(100 * 0.1 + 10 * 2.0, abs=1e-3)

    def test_threshold_is_inclusive_and_configurable(self):
        t0 = loc("2026-06-10 08:00")
        rows = [(t0, 82, 3), (t0 + 0.1, 81, 3), (t0 + 5, 82, 3), (t0 + 5.3, 81, 3)]
        rows += heartbeat("2026-06-10 00:00", "2026-06-11 00:00")
        prof = detector_activity_profile(ev(rows), TZ, start_date=D10, end_date=D10)
        assert row(prof, 3, D10).n_short == 1
        prof = detector_activity_profile(ev(rows), TZ, start_date=D10, end_date=D10,
                                         short_pulse_s=0.3)
        assert row(prof, 3, D10).n_short == 2


class TestSilence:
    def test_configured_silent_detector_gets_zero_row(self):
        roles = pd.DataFrame({"detector": [9, 99], "phase": [2, 2], "role": ["occupancy"] * 2})
        rows = heartbeat("2026-06-10 00:00", "2026-06-11 00:00")
        rows += [(loc("2026-06-10 09:00"), 82, 4), (loc("2026-06-10 09:00:01"), 81, 4)]
        prof = detector_activity_profile(ev(rows), TZ, roles=roles,
                                         start_date=D10, end_date=D11)
        r = row(prof, 9, D10)
        assert r.configured and r.n_act == 0 and r.on_time_s == 0
        assert r.occupancy == 0 and r.on_at_end == False   # noqa: E712 (logged, off)
        assert np.isnan(r.max_on_s) and np.isnan(r.min_off_gap_s)
        assert not row(prof, 4, D10).configured
        assert row(prof, 99, D10).configured

    def test_unlogged_day_has_no_occupancy(self):
        roles = pd.DataFrame({"detector": [9], "phase": [2], "role": ["occupancy"]})
        rows = heartbeat("2026-06-10 00:00", "2026-06-10 23:50")
        prof = detector_activity_profile(ev(rows), TZ, roles=roles,
                                         start_date=D10, end_date=D11)
        r = row(prof, 9, D11)
        assert r.observed_s == 0 and np.isnan(r.occupancy)
        assert pd.isna(r.on_at_start) and pd.isna(r.on_at_end)

    def test_no_events_no_dates_is_empty(self):
        out = detector_activity_profile(ev([]), TZ)
        assert out.empty and list(out.columns) == ACTIVITY_SCHEMA


class TestCensoredAcrossGap:
    @pytest.fixture
    def prof(self):
        t = loc("2026-06-10 10:00")
        rows = heartbeat("2026-06-10 00:00", "2026-06-10 09:50")
        rows += [
            (t - 50.0, 82, 6), (t - 40.0, 81, 6),   # measured; off-gap to next on spans the gap
            (t, 82, 5),
            (t + 100.0, -1, -1),                    # gap marker: 5 is still on
            (t + 400.0, 82, 99), (t + 400.2, 81, 99),
            (t + 500.0, 81, 5),                     # off after the reset: ignored
            (t + 600.0, 82, 6), (t + 601.0, 81, 6),
        ]
        rows += heartbeat("2026-06-10 10:10", "2026-06-11 00:00")
        return detector_activity_profile(ev(rows), TZ, start_date=D10, end_date=D10)

    def test_interval_is_censored(self, prof):
        r = row(prof, 5, D10)
        assert r.n_act == 1 and r.n_censored == 1
        assert np.isnan(r.max_on_s) and np.isnan(r.p95_on_s) and r.n_short == 0
        assert r.on_time_s == pytest.approx(100.0)
        assert r.open_on_s == pytest.approx(100.0)

    def test_unlogged_span_leaves_denominator(self, prof):
        r = row(prof, 5, D10)
        assert r.observed_s == pytest.approx(24 * H - 300.0)

    def test_no_off_gap_across_marker(self, prof):
        r = row(prof, 6, D10)
        assert r.n_act == 2 and r.max_on_s == pytest.approx(10.0)
        assert np.isnan(r.min_off_gap_s)

    def test_clock_step_marker_censors_too(self):
        t = loc("2026-06-10 10:00")
        rows = heartbeat("2026-06-10 00:00", "2026-06-11 00:00")
        rows += [(t, 82, 5), (t + 30.0 - 0.001, -1, -1), (t + 30.0, 82, 99),
                 (t + 30.2, 81, 99), (t + 60.0, 81, 5)]
        r = row(detector_activity_profile(ev(rows), TZ, start_date=D10, end_date=D10), 5, D10)
        assert r.n_censored == 1 and np.isnan(r.max_on_s)
        assert r.on_time_s == pytest.approx(30.0, abs=0.01)


class TestDST:
    def test_spring_forward_day_and_am_window(self):
        d = dt.date(2026, 3, 8)
        rows = heartbeat("2026-03-08 00:00", "2026-03-09 00:00")
        rows += [(loc("2026-03-08 00:30"), 82, 5), (loc("2026-03-08 03:30"), 81, 5)]
        prof = detector_activity_profile(
            ev(rows), TZ, windows={"day": ("00:00", "24:00"), "am": UDOT_AM_WINDOW},
            start_date=d, end_date=d)
        day, am = row(prof, 5, d), row(prof, 5, d, "am")
        assert day.bin_s == pytest.approx(23 * H)
        assert day.on_time_s == pytest.approx(2 * H)          # 00:30 MST → 03:30 MDT
        assert day.max_on_s == pytest.approx(2 * H)
        assert am.bin_s == pytest.approx(3 * H)               # 01:00 MST → 05:00 MDT
        assert am.bin_start == pytest.approx(loc("2026-03-08 01:00"))
        assert am.on_time_s == pytest.approx(1.5 * H)
        assert am.n_act == 0                                  # the on was at 00:30

    def test_fall_back_day_is_25_hours(self):
        d = dt.date(2026, 11, 1)
        rows = heartbeat("2026-10-31 23:50", "2026-11-02 00:10")
        prof = detector_activity_profile(ev(rows), TZ, start_date=d, end_date=d)
        r = row(prof, 99, d)
        assert r.bin_s == pytest.approx(25 * H)
        assert r.observed_s == pytest.approx(25 * H)

    def test_local_date_defaults(self):
        # 23:30 local on 06-10 is 05:30 UTC on 06-11: the bin is still 06-10.
        t = loc("2026-06-10 23:30")
        prof = detector_activity_profile(ev([(t, 82, 1), (t + 1, 81, 1)]), TZ)
        assert list(prof["date"]) == [D10]


class TestInputs:
    def test_tz_aware_timestamps_match_epoch(self):
        rows = heartbeat("2026-06-10 00:00", "2026-06-11 00:00")
        df = ev(rows)
        aware = df.assign(timestamp=pd.to_datetime(df["timestamp"], unit="s", utc=True))
        a = detector_activity_profile(df, TZ, start_date=D10, end_date=D10)
        b = detector_activity_profile(aware, TZ, start_date=D10, end_date=D10)
        pd.testing.assert_frame_equal(a, b)

    @pytest.mark.parametrize("win", [("05:00", "01:00"), ("00:00", "25:00")])
    def test_bad_window_raises(self, win):
        with pytest.raises(ValueError):
            detector_activity_profile(ev(heartbeat("2026-06-10 00:00", "2026-06-10 01:00")),
                                      TZ, windows={"w": win})
