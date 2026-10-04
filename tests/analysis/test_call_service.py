# Golden tests for call-to-service pairing: pedestrian delay and wait time
# (UDOT S-M5, analysis/call_service.py).
#
# Semantics are UDOT v5's (PedPhaseService, WaitTimeService,
# CycleService.GetWaitTimeCyclesAsync) except where the module docstring
# names a departure: multi-press windows are kept, and a call already on at
# red start ("held") starts the wait at the red.

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.call_service import (
    PED_DELAY_SCHEMA,
    PED_SUMMARY_SCHEMA,
    WAIT_SUMMARY_SCHEMA,
    WAIT_TIME_SCHEMA,
    first_call_in_windows,
    ped_delay,
    ped_service_calls,
    summarize_ped_delay,
    summarize_wait_time,
    wait_time,
)
from atspm.analysis.counts import ped_counts
from atspm.plotting.termination import _classify_ped_service

T0 = pd.Timestamp("2025-06-02 07:00", tz="US/Mountain")


def _events(rows, epoch=False, plan=1.0):
    df = pd.DataFrame(rows, columns=["s", "event_code", "parameter"])
    if epoch:
        df["timestamp"] = T0.timestamp() + df.pop("s").astype(float)
    else:
        df["timestamp"] = T0 + pd.to_timedelta(df.pop("s"), unit="s")
    df["cycle_start"] = df["timestamp"]
    df["coord_plan"] = plan
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _s(ts) -> float:
    return (ts - T0).total_seconds()


# ---------------------------------------------------------------------------
# Shared pairing
# ---------------------------------------------------------------------------


class TestFirstCallInWindows:

    def test_first_call_and_inclusive_ends(self):
        calls = np.array([5.0, 10.0, 12.0, 30.0])
        idx, n, _, _ = first_call_in_windows(
            np.array([10.0, 13.0, 0.0]), np.array([30.0, 20.0, 4.0]), calls)
        assert idx.tolist() == [1, -1, -1]       # call at the open counts
        assert n.tolist() == [3, 0, 0]           # ... and at the service

    def test_drop_restarts_the_window(self):
        calls = np.array([1.0, 6.0])
        drops = np.array([3.0])
        idx, n, nd, _ = first_call_in_windows(np.array([0.0]), np.array([10.0]), calls, drops)
        assert idx.tolist() == [1] and n.tolist() == [2] and nd.tolist() == [1]

    def test_drop_with_no_later_call_has_no_wait(self):
        idx, _, _, _ = first_call_in_windows(
            np.array([0.0]), np.array([10.0]), np.array([1.0]), np.array([3.0]))
        assert idx.tolist() == [-1]

    def test_drop_at_the_service_does_not_restart(self):
        idx, _, nd, _ = first_call_in_windows(
            np.array([0.0]), np.array([10.0]), np.array([1.0]), np.array([10.0]))
        assert idx.tolist() == [0] and nd.tolist() == [0]

    def test_held_starts_at_open_unless_a_drop_restarts(self):
        calls = np.array([6.0, 26.0])
        drops = np.array([24.0])
        idx, _, _, from_open = first_call_in_windows(
            np.array([0.0, 20.0]), np.array([10.0, 30.0]), calls, drops,
            held=np.array([True, True]))
        assert from_open.tolist() == [True, False]
        assert idx.tolist() == [-1, 1]


# ---------------------------------------------------------------------------
# Pedestrian delay
# ---------------------------------------------------------------------------

# Phase 2 (Code 90 presses, Code 45 alongside the first):
#   walk @0 (no clearance before it: censored), clear @7,
#   presses @20, @22 → walk @50 (delay 30, 2 presses),
#   press @53 during the walk → in_walk, clear @57,
#   press @57 at the clearance → waits for walk @100 (delay 43).
PED = [
    (0, 21, 2), (7, 22, 2),
    (20, 45, 2), (20, 90, 2), (21, 89, 2), (22, 90, 2), (23, 89, 2),
    (50, 21, 2), (53, 45, 2), (53, 90, 2), (57, 22, 2),
    (57, 90, 2), (57, 45, 2), (100, 21, 2), (107, 22, 2),
]


class TestPedDelay:

    def test_schema_and_kinds(self):
        d, why = ped_delay(_events(PED))
        assert why is None
        assert list(d.columns) == PED_DELAY_SCHEMA
        assert d["kind"].tolist() == ["censored", "waited", "in_walk", "waited"]
        assert d["delay_s"].iloc[1:].tolist() == [30.0, 0.0, 43.0]
        assert np.isnan(d["delay_s"].iloc[0]) and pd.isna(d["call_ts"].iloc[0])
        assert d["n_presses"].tolist() == [0, 2, 1, 1]
        assert (d["source"] == 90).all()

    def test_in_walk_row_points_at_its_walk(self):
        d, _ = ped_delay(_events(PED))
        iw = d.loc[d["kind"] == "in_walk"].iloc[0]
        assert _s(iw["walk_ts"]) == 50 and _s(iw["call_ts"]) == 53

    def test_falls_back_to_45_without_90(self):
        ev = _events([r for r in PED if r[1] != 90])
        d, _ = ped_delay(ev)
        assert (d["source"] == 45).all()
        assert d["delay_s"].iloc[1:].tolist() == [30.0, 0.0, 43.0]
        assert d["n_presses"].tolist() == [0, 1, 1, 1]

    def test_press_in_the_walks_decisecond_is_zero_delay(self):
        d, _ = ped_delay(_events([(0, 22, 4), (30, 90, 4), (30, 21, 4), (37, 22, 4)]))
        assert d["kind"].tolist() == ["waited"] and d["delay_s"].tolist() == [0.0]

    def test_uncalled_walk(self):
        d, _ = ped_delay(_events([(0, 22, 4), (30, 21, 4), (37, 22, 4), (40, 90, 6)]))
        assert d["kind"].tolist() == ["uncalled"]
        assert np.isnan(d["delay_s"].iloc[0])

    @pytest.mark.parametrize("gap_param", [-1, -2])
    def test_gap_marker_never_pairs_across(self, gap_param):
        ev = _events([(0, 22, 4), (10, 90, 4), (15, -1, gap_param), (40, 21, 4), (47, 22, 4),
                      (60, 90, 4), (90, 21, 4)])
        d, _ = ped_delay(ev)
        assert d["kind"].tolist() == ["censored", "waited"]
        assert d["delay_s"].iloc[1] == 30.0

    def test_gap_inside_walk_drops_the_in_walk_press(self):
        ev = _events([(0, 22, 4), (10, 90, 4), (40, 21, 4), (42, -1, -1), (43, 90, 4), (47, 22, 4)])
        d, _ = ped_delay(ev)
        assert d["kind"].tolist() == ["waited"]

    def test_missing_clearance_opens_after_previous_walk(self):
        # UDOT's case 4 (21 → 90 → 21): the press after the first walk
        # waits for the second.
        ev = _events([(0, 22, 4), (5, 90, 4), (10, 21, 4), (40, 90, 4), (70, 21, 4)])
        d, _ = ped_delay(ev)
        assert d["kind"].tolist() == ["waited", "waited"]
        assert d["delay_s"].tolist() == [5.0, 30.0]

    def test_ped_detector_map(self):
        ev = _events([(0, 22, 2), (10, 90, 11), (40, 21, 2)])
        d, _ = ped_delay(ev, ped_detectors={11: 2})
        assert d["delay_s"].tolist() == [30.0]
        d, _ = ped_delay(ev)
        assert d["kind"].tolist() == ["uncalled"]   # detector 11 ≠ phase 2

    def test_phases_filter_and_plan(self):
        ev = _events(PED + [(0, 22, 4), (10, 90, 4), (40, 21, 4)], plan=5.0)
        d, _ = ped_delay(ev, phases=[4])
        assert set(d["phase"]) == {4} and (d["coord_plan"] == 5.0).all()

    def test_epoch_input(self):
        d, _ = ped_delay(_events(PED, epoch=True))
        assert d["delay_s"].iloc[1:].tolist() == [30.0, 0.0, 43.0]
        assert d["walk_ts"].dtype == float

    @pytest.mark.parametrize("rows,needle", [
        ([(0, 1, 2), (5, 43, 2)], "Code 21"),
        ([(0, 21, 2), (10, 90, 2)], "Code 22"),
        ([(0, 21, 2), (7, 22, 2)], "Code 45 or 90"),
    ])
    def test_reason_when_not_computable(self, rows, needle):
        d, why = ped_delay(_events(rows))
        assert d.empty and list(d.columns) == PED_DELAY_SCHEMA
        assert needle in why

    def test_empty_events(self):
        d, why = ped_delay(_events([]))
        assert d.empty and why


class TestSummarizePedDelay:

    def test_counts_and_stats(self):
        d, _ = ped_delay(_events(PED))
        s = summarize_ped_delay(d, bin_len=None)
        assert list(s.columns) == PED_SUMMARY_SCHEMA and len(s) == 1
        r = s.iloc[0]
        assert (r["n_walks"], r["n_called"], r["n_uncalled"], r["n_censored"],
                r["n_in_walk"], r["n_delays"], r["presses"]) == (2, 2, 0, 1, 1, 3, 4)
        assert r["avg_delay_s"] == pytest.approx(73 / 3, abs=0.01)
        assert (r["min_delay_s"], r["max_delay_s"], r["total_delay_s"]) == (0.0, 43.0, 73.0)

    def test_binned(self):
        d, _ = ped_delay(_events(PED))
        s = summarize_ped_delay(d, bin_len=1)
        assert s["n_delays"].sum() == 3 and len(s) == 2

    def test_empty(self):
        assert list(summarize_ped_delay(pd.DataFrame()).columns) == PED_SUMMARY_SCHEMA


# ---------------------------------------------------------------------------
# Wait time
# ---------------------------------------------------------------------------


def _phase4(cycles=2, red_clear=True, term=4):
    """Phase 4, 100 s cycles: green 0, yellow 20, end yellow 24, red at 26."""
    rows = []
    for c in range(cycles):
        b = 100.0 * c
        rows += [(b, 1, 4), (b + 20, term, 4), (b + 20, 8, 4), (b + 24, 9, 4)]
        rows += [(b + 24, 10, 4), (b + 26, 11, 4)] if red_clear else [(b + 24, 12, 4)]
    return rows


def _w(rows, **kw):
    w = wait_time(_events(rows), **kw)
    return w.loc[w["phase"] == 4].reset_index(drop=True)


class TestWaitTime:

    def test_first_call_to_green(self):
        w = _w(_phase4() + [(40, 43, 4), (45, 44, 4), (70, 43, 4)])
        assert list(w.columns) == WAIT_TIME_SCHEMA
        r = w.iloc[0]
        assert _s(r["red_ts"]) == 26 and _s(r["green_ts"]) == 100 and _s(r["call_ts"]) == 40
        assert r["wait_s"] == 60.0 and r["called"] and not r["held"]
        assert (r["n_calls"], r["n_drops"], r["termination"]) == (2, 1, "gap_out")
        assert not r["censored"]

    def test_dropping_restarts_after_last_drop(self):
        w = _w(_phase4() + [(40, 43, 4), (45, 44, 4), (70, 43, 4)], dropping=[4])
        assert w["wait_s"].iloc[0] == 30.0 and _s(w["call_ts"].iloc[0]) == 70

    def test_dropping_with_drop_last_has_no_wait(self):
        rows = _phase4() + [(40, 43, 4), (45, 44, 4)]
        assert not _w(rows, dropping=[4])["called"].iloc[0]
        assert _w(rows)["wait_s"].iloc[0] == 60.0

    def test_held_call_starts_at_red(self):
        w = _w(_phase4() + [(15, 43, 4)])
        r = w.iloc[0]
        assert r["held"] and r["called"] and r["wait_s"] == 74.0
        assert _s(r["call_ts"]) == 26 and r["n_calls"] == 0

    def test_held_call_dropped_in_red_restarts_with_dropping(self):
        rows = _phase4() + [(15, 43, 4), (50, 44, 4), (80, 43, 4)]
        assert _w(rows, dropping=[4])["wait_s"].iloc[0] == 20.0
        assert _w(rows)["wait_s"].iloc[0] == 74.0

    def test_call_dropped_before_red_is_not_held(self):
        w = _w(_phase4() + [(10, 43, 4), (15, 44, 4)])
        assert not w["held"].iloc[0] and not w["called"].iloc[0]
        assert np.isnan(w["wait_s"].iloc[0]) and pd.isna(w["call_ts"].iloc[0])

    def test_call_at_red_start_is_registered_not_held(self):
        w = _w(_phase4() + [(26, 43, 4)])
        assert not w["held"].iloc[0] and w["wait_s"].iloc[0] == 74.0

    def test_call_at_green_is_zero_wait(self):
        assert _w(_phase4() + [(100, 43, 4)])["wait_s"].iloc[0] == 0.0

    def test_gap_in_window_censors(self):
        w = _w(_phase4() + [(40, 43, 4), (60, -1, -1)])
        r = w.iloc[0]
        assert r["censored"] and not r["called"] and np.isnan(r["wait_s"])
        assert r["n_calls"] == 0

    def test_gap_between_held_state_and_red_is_not_held(self):
        # The gap also ends the first green's interval; the second red has
        # a call state from before the gap only.
        rows = _phase4(3) + [(15, 43, 4), (110, -1, -1)]
        w = _w(rows)
        r = w.loc[w["red_ts"].map(_s) == 226].iloc[0]
        assert not r["held"]

    def test_last_red_has_no_green(self):
        w = _w(_phase4() + [(150, 43, 4)])
        assert len(w) == 2
        last = w.iloc[1]
        assert last["censored"] and pd.isna(last["green_ts"])

    @pytest.mark.parametrize("code,label", [(5, "max_out"), (6, "force_off")])
    def test_termination(self, code, label):
        assert _w(_phase4(term=code) + [(40, 43, 4)])["termination"].iloc[0] == label

    def test_termination_unknown(self):
        rows = [r for r in _phase4() if r[1] != 4] + [(40, 43, 4)]
        assert _w(rows)["termination"].iloc[0] == "unknown"

    def test_no_red_clearance_red_starts_at_end_yellow(self):
        # 201 P3: 1 → 8 → 9 → 12, no Codes 10/11.
        w = _w(_phase4(red_clear=False) + [(40, 43, 4)])
        assert _s(w["red_ts"].iloc[0]) == 24 and w["wait_s"].iloc[0] == 60.0

    def test_immediate_reservice(self):
        # Free mode: Code 11 and the next Code 1 in one decisecond.
        rows = [(0, 1, 4), (20, 8, 4), (24, 9, 4), (24, 10, 4), (26, 11, 4), (26, 43, 4),
                (26, 1, 4), (40, 8, 4), (44, 9, 4), (44, 10, 4), (46, 11, 4)]
        r = _w(rows).iloc[0]
        assert _s(r["green_ts"]) == 26 and r["wait_s"] == 0.0

    def test_plan_at_green_and_epoch_input(self):
        w = wait_time(_events(_phase4() + [(40, 43, 4)], epoch=True, plan=7.0))
        assert w["wait_s"].iloc[0] == 60.0 and w["coord_plan"].iloc[0] == 7.0
        assert w["green_ts"].dtype == float

    def test_phases_filter(self):
        rows = _phase4() + [(r[0], r[1], 8) for r in _phase4()]
        assert set(wait_time(_events(rows), phases=[8])["phase"]) == {8}

    def test_empty(self):
        assert list(wait_time(_events([])).columns) == WAIT_TIME_SCHEMA


class TestSummarizeWaitTime:

    def _frame(self):
        # Three cycles: wait 60 (gap out), held 74 (gap out), and one over
        # the cap via a 400 s red.
        rows = _phase4(2) + [(40, 43, 4), (115, 43, 4)]
        rows += [(205, 44, 4), (200, 1, 4), (220, 5, 4), (220, 8, 4), (224, 9, 4), (224, 10, 4), (226, 11, 4),
                 (230, 43, 4), (700, 1, 4), (720, 8, 4), (724, 9, 4), (724, 10, 4), (726, 11, 4)]
        return wait_time(_events(rows))

    def test_plan_row(self):
        s = summarize_wait_time(self._frame(), bin_len=None)
        assert list(s.columns) == WAIT_SUMMARY_SCHEMA and len(s) == 1
        r = s.iloc[0]
        assert (r["n_windows"], r["n_censored"], r["n_called"], r["n_held"], r["n_over_max"]) == \
            (3, 1, 2, 1, 1)
        assert r["avg_wait_s"] == pytest.approx(67.0)
        assert r["max_wait_s"] == 74.0
        assert r["avg_wait_udot_s"] == pytest.approx(60.0)
        assert (r["n_gap_out"], r["n_max_out"], r["n_force_off"], r["n_unknown"]) == (2, 0, 0, 0)
        assert r["avg_wait_gap_out_s"] == pytest.approx(67.0)
        assert np.isnan(r["avg_wait_max_out_s"])

    def test_no_cap(self):
        r = summarize_wait_time(self._frame(), bin_len=None, max_wait=None).iloc[0]
        assert r["n_over_max"] == 0 and r["n_called"] == 3 and r["n_max_out"] == 1
        assert r["max_wait_s"] == 470.0

    def test_binned_by_green(self):
        s = summarize_wait_time(self._frame(), bin_len=5)
        assert s["n_windows"].sum() == 3 and s["n_censored"].sum() == 1
        assert (s["time"].diff().dropna() >= pd.Timedelta(minutes=5)).all()

    def test_empty(self):
        assert list(summarize_wait_time(pd.DataFrame()).columns) == WAIT_SUMMARY_SCHEMA


# ---------------------------------------------------------------------------
# Ped counts and the termination plot share the pairing
# ---------------------------------------------------------------------------


def test_ped_counts_and_termination_use_ped_service_calls():
    rows = [(0, 45, 2), (10, 21, 2), (100, 21, 2), (150, 45, 2), (155, -1, -1), (160, 21, 2),
            (200, 45, 3), (200, 21, 3)]
    ev = _events(rows)
    svc = ped_service_calls(ev)
    assert svc["called"].tolist() == [True, False, False, True]
    assert int(ped_counts(ev, bin_len=60)["Ped Total"].sum()) == 2
    act, rec = _classify_ped_service(ev)
    assert (len(act), len(rec)) == (2, 2)


# ---------------------------------------------------------------------------
# Corpus: 315 (DB skipped when absent)
# ---------------------------------------------------------------------------

_DB = Path(__file__).resolve().parents[2] / "intersections" / \
    "315_US-20-26_Franklin_Rd_and_KCID_Rd" / "315_data.db"


def _load(start, end, codes):
    if not _DB.exists():
        pytest.skip("corpus DB not present")
    from atspm.data.reader import get_events_with_cycles_df
    return get_events_with_cycles_df(_DB, start, end, event_codes=codes, timezone="US/Mountain")


@pytest.fixture(scope="module")
def ped315():
    return _load(datetime(2025, 12, 15), datetime(2025, 12, 20), [-1, 21, 22, 45, 89, 90])


@pytest.fixture(scope="module")
def wait315():
    return _load(datetime(2025, 12, 15), datetime(2025, 12, 16),
                 [-1, 1, 4, 5, 6, 8, 9, 10, 11, 12, 43, 44])


def test_corpus_315_ped_delay(ped315):
    d, why = ped_delay(ped315)
    assert why is None
    kinds = d.groupby(["phase", "kind"]).size().to_dict()
    assert kinds == {(2, "censored"): 1, (2, "waited"): 9, (4, "censored"): 1, (4, "in_walk"): 1,
                     (4, "waited"): 16, (6, "censored"): 1, (6, "waited"): 9,
                     (8, "censored"): 1, (8, "in_walk"): 2, (8, "waited"): 3}
    assert (d["source"] == 90).all()
    s = summarize_ped_delay(d, bin_len=None).groupby("phase")["total_delay_s"].sum()
    assert s.round(1).to_dict() == pytest.approx({2: 261.5, 4: 608.0, 6: 266.9, 8: 127.2}, abs=0.1)


def test_corpus_315_code45_matches_code90(ped315):
    # 45 logs in the decisecond of the first press: identical delays.
    d90, _ = ped_delay(ped315)
    d45, _ = ped_delay(ped315.loc[ped315["event_code"] != 90])
    assert (d45["source"] == 45).all()
    np.testing.assert_array_equal(d90["kind"], d45["kind"])
    np.testing.assert_allclose(d90["delay_s"], d45["delay_s"])


def test_corpus_315_wait_time(wait315):
    w = wait_time(wait315, phases=[2, 4, 6, 8])
    g = w.groupby("phase")
    assert g.size().to_dict() == {2: 918, 4: 844, 6: 952, 8: 847}
    assert g["censored"].sum().to_dict() == {2: 0, 4: 1, 6: 0, 8: 1}
    assert g["called"].sum().to_dict() == {2: 910, 4: 670, 6: 940, 8: 603}
    assert g["held"].sum().to_dict() == {2: 156, 4: 79, 6: 194, 8: 90}
    assert g["wait_s"].mean().round(2).to_dict() == {2: 14.79, 4: 40.7, 6: 18.92, 8: 37.59}


def test_corpus_315_dropping_shortens_coordinated_waits(wait315):
    # P2/P6 have stop-bar presence zones (Det_P{N}_Occupancy): calls bounce
    # in the red, and the dropping algorithm restarts at the last drop.
    w = wait_time(wait315, phases=[2, 6], dropping=[2, 6])
    g = w.groupby("phase")
    assert g["called"].sum().to_dict() == {2: 900, 6: 939}
    assert g["wait_s"].mean().round(2).to_dict() == {2: 11.81, 6: 16.27}
