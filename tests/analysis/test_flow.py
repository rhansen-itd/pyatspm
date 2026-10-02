"""Tests for the split flow-rate extensions (Functional Core).

Target: src/atspm/analysis/flow.py.

Contract summary
----------------
flow_rate(max_lost=None): the end-slack filter is off; every (window,
    detector) group is kept with its measured ``lost``.
_select_cycles: default mode = modal split ± tolerance, then the busiest
    pct percent of cycles by summed q.  Stratified mode = busiest pct
    percent within each (coord_plan, round(split)) stratum, pooled.
saturation_state: per phase, pass_rate = share of (window, detector) rows
    with lost <= max_lost; saturated = pass_rate >= threshold.  Diagnostics:
    window_pass_rate (all lanes in a window pass) and min_lane_pass_rate.
discharge_profiles: approach cumulative curve N(t) = SUM of per-detector
    mean cumulative counts on a uniform grid from 0 to t_dom, where t_dom
    is the earliest per-detector last supported row.  Leading rows are 0,
    the curve is monotone, and a detector that never reaches min_cycles
    support empties the profile rather than undercounting the approach.
"""

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.flow import (
    _select_cycles,
    discharge_profiles,
    flow_rate,
    rate_profiles,
    saturation_state,
)

_PHASE = 2
_DET_A = 11
_DET_B = 12
_T0 = 1_000_000.0


def _events(cycles) -> pd.DataFrame:
    """Build a flat events frame for phase 2.

    Args:
        cycles: list of ``(green_s, plan, {det: [t, ...]})``; each cycle is
            green, then 4 s yellow and 2 s red clearance, so the split is
            ``green_s + 6``.  Departure times are seconds after green onset.
    """
    rows = []
    start = _T0
    for green_s, plan, departures in cycles:
        phase_rows = [
            (start, 1),
            (start + green_s, 8),
            (start + green_s + 4, 9),
            (start + green_s + 4, 10),
            (start + green_s + 6, 11),
        ]
        for ts, code in phase_rows:
            rows.append((ts, code, _PHASE, start, plan))
        for det, times in departures.items():
            for t in times:
                rows.append((start + t, 81, det, start, plan))
        start += green_s + 6 + 30.0
    return (
        pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter",
                                    "cycle_start", "coord_plan"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _cycle_rows(specs) -> pd.DataFrame:
    """Per-cycle summary rows: ``(green_ts, plan, split, q)``, one detector."""
    return pd.DataFrame(
        [
            dict(det=_DET_A, phase=_PHASE, green_ts=g, cycle_start=g,
                 coord_plan=float(p), split=s, green_dur=s - 6.0,
                 clear_dur=6.0, lost=1.0, q=q)
            for g, p, s, q in specs
        ]
    )


# A 2 s headway on A from t=2 to 40, a 4 s headway on B from t=4 to 36.
_STEADY = {_DET_A: list(np.arange(2.0, 40.1, 2.0)),
           _DET_B: list(np.arange(4.0, 36.1, 4.0))}


class TestFlowRateMaxLost:

    def _two_cycles(self):
        # Cycle 1 discharges to t=40 of a 46 s split (lost 6).
        # Cycle 2 stops at t=10 (lost 36): not at capacity.
        return _events([
            (40.0, 1, {_DET_A: [5.0, 20.0, 40.0]}),
            (40.0, 1, {_DET_A: [5.0, 10.0]}),
        ])

    def test_default_drops_unsaturated_window(self):
        cycle_df, vehicle_df = flow_rate(self._two_cycles(), _PHASE, [_DET_A])
        assert len(cycle_df) == 1
        assert cycle_df["lost"].iloc[0] == pytest.approx(6.0)
        assert len(vehicle_df) == 3

    def test_none_keeps_every_window_with_its_lost(self):
        cycle_df, vehicle_df = flow_rate(
            self._two_cycles(), _PHASE, [_DET_A], max_lost=None
        )
        assert cycle_df["lost"].tolist() == pytest.approx([6.0, 36.0])
        assert cycle_df["q"].tolist() == [3, 2]
        assert len(vehicle_df) == 5


class TestSelectCycles:

    def test_default_modal_filter_then_busiest(self):
        # Modal split 46; 60 is outside ±10%.  pct=50 keeps the busier half.
        df = _cycle_rows([
            (1.0, 1, 46.0, 10), (2.0, 1, 46.0, 20),
            (3.0, 1, 46.0, 30), (4.0, 1, 46.0, 40),
            (5.0, 2, 60.0, 99),
        ])
        sel = _select_cycles(df, pct=50.0, split_tolerance=0.10, stratify=False)
        assert sorted(sel["green_ts"]) == [3.0, 4.0]

    def test_stratified_keeps_busiest_within_each_stratum(self):
        df = _cycle_rows([
            (1.0, 1, 46.0, 10), (2.0, 1, 46.2, 20),   # stratum (1, 46)
            (3.0, 2, 30.0, 5), (4.0, 2, 29.9, 7),     # stratum (2, 30)
        ])
        sel = _select_cycles(df, pct=50.0, split_tolerance=0.10, stratify=True)
        # The plan-2 cycles survive even though their split isn't modal
        # and their volumes are below every plan-1 cycle.
        assert sorted(sel["green_ts"]) == [2.0, 4.0]

    def test_stratified_sums_q_across_detectors(self):
        df = _cycle_rows([(1.0, 1, 46.0, 10), (2.0, 1, 46.0, 12)])
        extra = _cycle_rows([(1.0, 1, 46.0, 5)]).assign(det=_DET_B)
        sel = _select_cycles(pd.concat([df, extra]), pct=50.0,
                             split_tolerance=0.10, stratify=True)
        # Cycle 1 totals 15 across both detectors, so it beats cycle 2's 12.
        assert set(sel["green_ts"]) == {1.0}
        assert len(sel) == 2

    def test_empty_input(self):
        df = _cycle_rows([]).reindex(columns=_cycle_rows(
            [(1.0, 1, 46.0, 1)]).columns)
        assert _select_cycles(df, 1.0, 0.10, True).empty


class TestRateProfilesStratify:

    def test_stratify_reaches_the_short_split_plan(self):
        events = _events(
            [(40.0, 1, _STEADY)] * 6
            + [(20.0, 2, {_DET_A: list(np.arange(2.0, 20.1, 2.0))})] * 2
        )
        cycle_df, vehicle_df = flow_rate(events, _PHASE, [_DET_A, _DET_B],
                                         max_lost=None)
        modal, _, _ = rate_profiles(cycle_df, vehicle_df, pct=100.0)
        strat, _, _ = rate_profiles(cycle_df, vehicle_df, pct=100.0,
                                    stratify=True)
        assert set(modal["coord_plan"]) == {1.0}
        assert set(strat["coord_plan"]) == {1.0, 2.0}


class TestDischargeProfiles:

    def _profile(self, cycles, **kw):
        cycle_df, vehicle_df = flow_rate(
            _events(cycles), _PHASE, [_DET_A, _DET_B], max_lost=None
        )
        kw.setdefault("pct", 100.0)
        return discharge_profiles(cycle_df, vehicle_df, **kw)

    def test_sum_of_detector_means_on_full_grid(self):
        selected, prof = self._profile([(40.0, 1, _STEADY)] * 6)
        assert len(selected) == 12           # 6 cycles x 2 detectors
        assert list(prof.columns) == ["n", "inst"]
        assert prof.index.name == "t"
        assert prof.index[0] == 0.0
        assert np.allclose(np.diff(prof.index), 0.5)
        # Domain: B's last departure (36 s) ends before A's (40 s).
        assert prof.index[-1] == 36.0
        n = prof["n"]
        assert n.loc[0.0] == 0.0
        assert n.loc[2.0] == pytest.approx(1.0)       # A only
        assert n.loc[3.0] == pytest.approx(1.5)       # A interpolated, B 0
        assert n.loc[20.0] == pytest.approx(15.0)     # 10 + 5, a sum
        assert n.loc[21.0] == pytest.approx(15.75)    # 10.5 + 5.25
        assert n.loc[36.0] == pytest.approx(27.0)     # 18 + 9
        assert (np.diff(n.to_numpy()) >= 0).all()

    def test_inst_is_summed_approach_rate(self):
        _, prof = self._profile([(40.0, 1, _STEADY)] * 6)
        # A at 2 s headways (1800 vph) plus B at 4 s (900 vph).
        assert prof["inst"].loc[20.0] == pytest.approx(2700.0)

    def test_domain_is_last_row_with_min_cycles_support(self):
        # A runs to 40 s in 5 cycles but only to 30 s in the sixth; with
        # min_cycles=6, A's mean stops at 30 s and so does the domain.
        short_a = {_DET_A: list(np.arange(2.0, 30.1, 2.0)),
                   _DET_B: _STEADY[_DET_B]}
        _, prof = self._profile([(40.0, 1, _STEADY)] * 5
                                + [(40.0, 1, short_a)], min_cycles=6)
        assert prof.index[-1] == 30.0

    def test_unsupported_detector_empties_profile(self):
        only_a = {_DET_A: _STEADY[_DET_A]}
        cycles = [(40.0, 1, _STEADY)] * 3 + [(40.0, 1, only_a)] * 3
        selected, prof = self._profile(cycles, min_cycles=5)
        assert not selected.empty
        assert prof.empty
        assert list(prof.columns) == ["n", "inst"]

    def test_running_max_removes_dips(self):
        # The slow cycles' first departure (n=1 at 10 s) joins a mean
        # that the fast cycles hold above 3, so the raw mean dips there.
        fast = {_DET_A: [1.0, 2.0, 3.0, 30.0], _DET_B: [1.0, 30.0]}
        slow = {_DET_A: [10.0, 20.0, 29.0, 30.0], _DET_B: [15.0, 30.0]}
        _, prof = self._profile([(30.0, 1, fast)] * 3 + [(30.0, 1, slow)] * 3,
                                min_cycles=3)
        assert (np.diff(prof["n"].to_numpy()) >= 0).all()

    def test_empty_inputs(self):
        cycle_df, vehicle_df = flow_rate(pd.DataFrame(
            columns=["timestamp", "event_code", "parameter",
                     "cycle_start", "coord_plan"]), _PHASE, [_DET_A])
        selected, prof = discharge_profiles(cycle_df, vehicle_df)
        assert selected.empty and prof.empty
        assert list(prof.columns) == ["n", "inst"]


def _obs(rows) -> pd.DataFrame:
    """cycle_df rows from ``(phase, det, green_ts, lost)``."""
    return pd.DataFrame(
        [dict(det=d, phase=p, green_ts=g, cycle_start=g, coord_plan=1.0,
              split=46.0, green_dur=40.0, clear_dur=6.0, lost=l, q=10)
         for p, d, g, l in rows]
    )


class TestSaturationState:

    def test_pooled_rate_and_diagnostics(self):
        # Phase 2: lane A passes 4/4; lane B passes 2/4.
        # Windows 1-2 all pass, 3-4 fail on B.
        rows = [(2, _DET_A, g, 3.0) for g in (1.0, 2.0, 3.0, 4.0)]
        rows += [(2, _DET_B, 1.0, 3.0), (2, _DET_B, 2.0, 10.0),
                 (2, _DET_B, 3.0, 25.0), (2, _DET_B, 4.0, 10.01)]
        out = saturation_state(_obs(rows)).iloc[0]
        assert out["phase"] == 2
        assert out["n_cycles"] == 4
        assert out["n_obs"] == 8
        assert out["pass_rate"] == pytest.approx(0.75)   # 10.0 is a pass
        assert out["window_pass_rate"] == pytest.approx(0.5)
        assert out["min_lane_pass_rate"] == pytest.approx(0.5)
        assert not out["saturated"]                       # 0.75 < 0.8

    def test_threshold_and_max_lost_are_parameters(self):
        rows = [(2, _DET_A, g, l) for g, l in
                ((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 30.0))]
        assert saturation_state(_obs(rows))["saturated"].iloc[0]          # 0.8
        assert not saturation_state(_obs(rows), max_lost=5.0)["saturated"].iloc[0]
        assert not saturation_state(_obs(rows), threshold=0.9)["saturated"].iloc[0]

    def test_one_row_per_phase_sorted(self):
        rows = [(6, _DET_A, 1.0, 30.0), (2, _DET_B, 1.0, 1.0)]
        out = saturation_state(_obs(rows))
        assert out["phase"].tolist() == [2, 6]
        assert out["saturated"].tolist() == [True, False]

    def test_from_unfiltered_flow_rate(self):
        events = _events([(40.0, 1, {_DET_A: [5.0, 20.0, 40.0]}),
                          (40.0, 1, {_DET_A: [5.0, 10.0]})])
        cycle_df, _ = flow_rate(events, _PHASE, [_DET_A], max_lost=None)
        out = saturation_state(cycle_df).iloc[0]
        assert out["n_cycles"] == 2
        assert out["pass_rate"] == pytest.approx(0.5)

    def test_empty(self):
        out = saturation_state(pd.DataFrame(columns=_obs(
            [(2, _DET_A, 1.0, 1.0)]).columns))
        assert out.empty
        assert "saturated" in out.columns
