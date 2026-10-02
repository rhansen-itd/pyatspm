"""Tests for the optimizer's TOD-plan validation (Functional Core).

Target: src/atspm/analysis/optimizer_validation.py.  Definitions are in
its module docstring (design_optimizer_solver.md D8, amended 2026-10-02).

Synthetic plans: phases 2 and 6 under test, two lanes each.  Every lane
departs at a constant headway from t = h until a stop time, identically
in every cycle, so a curve is the step-interpolated departure count and
every expected number is closed-form arithmetic in this file.
"""

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.optimizer_validation import (
    STATUS_CURVE_MISSING,
    STATUS_NOT_COVERED,
    STATUS_TESTED,
    STATUS_TOO_FEW_CYCLES,
    valid_cycles,
    validate_plans,
)

_DETS = {2: [1, 2], 6: [3, 4]}
_T0 = 1_750_000_000.0


def _departures(stop, h=2.0):
    """Departure times h, 2h, ... up to and including *stop*.

    Inclusive so that a window ending on a departure counts it, which is
    what the interpolated curve says at that time; an "exact" plan then
    has zero model error.
    """
    return np.arange(h, stop + 1e-9, h)


def _build(plans, n_cycles=40, start=_T0):
    """Build (flow, cycles_df) for consecutive plan periods.

    Args:
        plans: list of ``(plan, C, {phase: (split, stop, h)})``.  A lane
            departs every *h* seconds before *stop*; the window is
            *split* long (clearance is ``split - stop`` when stop is the
            green end, zero when stop == split).
        n_cycles: cycles per plan.  One trailing cycle is appended so the
            last real cycle has a length.

    Returns:
        ``(flow, cycles_df)`` with ``flow[p] = (cycle_df, vehicle_df)``.
    """
    cyc_rows, c_rows, v_rows = [], [], []
    t = start
    for plan, c_len, spec in plans:
        for _ in range(n_cycles):
            cyc_rows.append((t, float(plan)))
            for p, (split, stop, h) in spec.items():
                dep = _departures(stop, h)
                for d in _DETS[p]:
                    lost = split - (dep[-1] if len(dep) else 0.0)
                    c_rows.append(dict(
                        det=d, phase=p, green_ts=t, cycle_start=t,
                        coord_plan=float(plan), split=float(split),
                        green_dur=float(stop), clear_dur=float(split - stop),
                        lost=float(lost), q=len(dep), termination="force_off",
                    ))
                    hw = np.append(np.diff(dep), np.nan)
                    for i, (tt, hh) in enumerate(zip(dep, hw)):
                        v_rows.append(dict(det=d, green_ts=t, t=float(tt),
                                           n=i + 1, headway=float(hh)))
            t += c_len
    cyc_rows.append((t, cyc_rows[-1][1]))  # trailing cycle: closes the last
    cycles_df = pd.DataFrame(cyc_rows, columns=["cycle_start", "coord_plan"])
    cdf, vdf = pd.DataFrame(c_rows), pd.DataFrame(v_rows)
    flow = {
        p: (cdf.loc[cdf["phase"] == p].reset_index(drop=True),
            vdf.loc[vdf["det"].isin(_DETS[p])].reset_index(drop=True))
        for p in _DETS
    }
    return flow, cycles_df


def _n(s, stop, h=2.0):
    """Model N(s) per lane: interpolated departure count, held past the end."""
    dep = _departures(stop, h)
    return float(np.interp(s, dep, np.arange(1, len(dep) + 1)))


def _q(stop, h=2.0):
    """Observed vehicles per lane per cycle."""
    return len(_departures(stop, h))


def _run(plans, **kw):
    flow, cycles_df = _build(plans, n_cycles=kw.pop("n_cycles", 40))
    kw.setdefault("pct", 100.0)
    return validate_plans(flow, cycles_df, **kw)


# Plan A: C=120, splits 60.  Plan B: C=90, splits 46 (on a departure).
# "Exact" lanes discharge through the whole window (stop == split), which
# is the model's own assumption, so its prediction is exact.
_A_EXACT = (1, 120.0, {2: (60.0, 60.0, 2.0), 6: (60.0, 60.0, 2.0)})
_B_EXACT = (2, 90.0, {2: (46.0, 46.0, 2.0), 6: (46.0, 46.0, 2.0)})


# ---------------------------------------------------------------------------
# valid_cycles
# ---------------------------------------------------------------------------


class TestValidCycles:

    def test_length_is_gap_to_next_start_and_last_is_dropped(self):
        df = pd.DataFrame({"cycle_start": [0.0, 100.0, 220.0],
                           "coord_plan": [1.0, 1.0, 1.0]})
        out = valid_cycles(df)
        assert out["cycle_start"].tolist() == [0.0, 100.0]
        assert out["cycle_len"].tolist() == [100.0, 120.0]

    def test_unsorted_input_is_sorted(self):
        df = pd.DataFrame({"cycle_start": [100.0, 0.0, 200.0],
                           "coord_plan": [1.0, 1.0, 1.0]})
        assert valid_cycles(df)["cycle_len"].tolist() == [100.0, 100.0]

    def test_transition_cycle_is_dropped(self):
        df = pd.DataFrame({"cycle_start": [0.0, 100.0, 200.0, 290.0],
                           "coord_plan": [1.0, 1.0, 2.0, 2.0]})
        out = valid_cycles(df)
        assert out["cycle_start"].tolist() == [0.0, 200.0]

    def test_gap_marker_inside_a_cycle_drops_it(self):
        df = pd.DataFrame({"cycle_start": [0.0, 100.0, 200.0, 300.0],
                           "coord_plan": [1.0] * 4})
        # A gap at the start of cycle 100 drops it; one at 299.9 drops 200.
        out = valid_cycles(df, gap_ts=[100.0, 299.9])
        assert out["cycle_start"].tolist() == [0.0]

    def test_gap_at_next_start_belongs_to_the_next_cycle(self):
        df = pd.DataFrame({"cycle_start": [0.0, 100.0, 200.0],
                           "coord_plan": [1.0] * 3})
        out = valid_cycles(df, gap_ts=[200.0])
        assert out["cycle_start"].tolist() == [0.0, 100.0]

    def test_non_numeric_cycle_start_raises(self):
        df = pd.DataFrame({
            "cycle_start": pd.to_datetime([0, 100], unit="s", utc=True),
            "coord_plan": [1.0, 1.0],
        })
        with pytest.raises(ValueError, match="epoch"):
            valid_cycles(df)


# ---------------------------------------------------------------------------
# Observed throughput and operating points
# ---------------------------------------------------------------------------


class TestPlans:

    def test_observed_vph_is_served_over_cycle_time(self):
        res = _run([_A_EXACT, _B_EXACT])
        plans = res["plans"].set_index("coord_plan")
        # 2 phases x 2 lanes x q per lane, per cycle.
        assert plans.loc[1.0, "observed_vph"] == pytest.approx(
            3600 * 4 * _q(60.0) / 120.0)
        assert plans.loc[2.0, "observed_vph"] == pytest.approx(
            3600 * 4 * _q(46.0) / 90.0)
        # Plan 1's last cycle is a transition cycle and has no length.
        assert plans.loc[1.0, "n_cycles"] == 39
        assert plans.loc[2.0, "n_cycles"] == 40
        assert plans.loc[1.0, "c_median"] == 120.0
        assert plans.loc[2.0, "split_p2"] == 46.0
        assert plans.loc[2.0, "split_p6"] == 46.0

    def test_insample_prediction_is_the_plans_own_curve(self):
        res = _run([_A_EXACT])
        row = res["plans"].iloc[0]
        assert row["insample_vph"] == pytest.approx(
            3600 * 4 * _n(60.0, 60.0) / 120.0)
        assert row["insample_pct_error"] == pytest.approx(0.0, abs=1e-9)
        assert bool(row["curve_ok"]) and bool(row["eligible"])

    def test_cycle_missing_a_phase_window_is_excluded_everywhere(self):
        flow, cycles_df = _build([_A_EXACT])
        cdf, vdf = flow[6]
        drop = cycles_df["cycle_start"].iloc[3]
        flow[6] = (cdf.loc[cdf["cycle_start"] != drop],
                   vdf.loc[vdf["green_ts"] != drop])
        res = validate_plans(flow, cycles_df, pct=100.0)
        row = res["plans"].iloc[0]
        assert row["n_cycles"] == 39
        # Numerator and denominator both lose the cycle: rate unchanged.
        assert row["observed_vph"] == pytest.approx(
            3600 * 4 * _q(60.0) / 120.0)

    def test_cycle_with_two_windows_for_a_phase_is_excluded(self):
        flow, cycles_df = _build([_A_EXACT])
        cdf, vdf = flow[2]
        # Re-attribute cycle 4's Ph2 window to cycle 3: cycle 3 now has two.
        c3, c4 = cycles_df["cycle_start"].iloc[3], cycles_df["cycle_start"].iloc[4]
        cdf = cdf.assign(cycle_start=cdf["cycle_start"].replace(c4, c3))
        flow[2] = (cdf, vdf)
        res = validate_plans(flow, cycles_df, pct=100.0)
        row = res["plans"].iloc[0]
        # Cycle 3 is dropped; cycle 4 has no Ph2 window and is dropped too.
        assert row["n_cycles"] == 38
        assert row["observed_vph"] == pytest.approx(
            3600 * 4 * _q(60.0) / 120.0)
        assert any("more than one window" in w
                   for w in res["verdict"]["warnings"])

    def test_cycle_spanning_a_gap_is_excluded(self):
        flow, cycles_df = _build([_A_EXACT])
        gap = cycles_df["cycle_start"].iloc[5] + 100.0  # after the windows
        res = validate_plans(flow, cycles_df, gap_ts=[gap], pct=100.0)
        assert res["plans"].iloc[0]["n_cycles"] == 39

    def test_median_operating_point(self):
        flow, cycles_df = _build([_A_EXACT], n_cycles=41)
        # Stretch every other cycle by 10 s: median length stays 120 only
        # if fewer than half are stretched (20 of 41).
        starts = cycles_df["cycle_start"].to_numpy().copy()
        shift = np.cumsum([0.0] + [10.0 if i % 2 else 0.0
                                   for i in range(len(starts) - 1)])
        new = starts + shift
        remap = dict(zip(starts, new))
        cycles_df = cycles_df.assign(cycle_start=new)
        flow = {p: (c.assign(cycle_start=c["cycle_start"].map(remap),
                             green_ts=c["green_ts"].map(remap)),
                    v.assign(green_ts=v["green_ts"].map(remap)))
                for p, (c, v) in flow.items()}
        res = validate_plans(flow, cycles_df, pct=100.0)
        row = res["plans"].iloc[0]
        assert row["c_median"] == 120.0
        assert row["observed_vph"] == pytest.approx(
            3600 * 4 * _q(60.0) * 41 / (41 * 120.0 + 20 * 10.0))

    def test_advisory_pass_rate_per_plan(self):
        res = _run([_A_EXACT, _B_EXACT])
        plans = res["plans"]
        # Every window forces off with lost = h = 2 s <= 10: all pass.
        assert (plans["pass_rate_p2"] == 1.0).all()
        assert (plans["pass_rate_p6"] == 1.0).all()


# ---------------------------------------------------------------------------
# Pairs and verdict
# ---------------------------------------------------------------------------


class TestPairs:

    def test_exact_model_passes(self):
        res = _run([_A_EXACT, _B_EXACT])
        pair = res["pairs"].iloc[0]
        assert pair["status"] == STATUS_TESTED
        assert (pair["anchor_plan"], pair["target_plan"]) == (1.0, 2.0)
        exp_b = 3600 * 4 * _n(46.0, 60.0) / 90.0
        assert pair["predicted_target_vph"] == pytest.approx(exp_b)
        assert pair["observed_target_vph"] == pytest.approx(
            3600 * 4 * _q(46.0) / 90.0)
        assert pair["change_error_pp"] == pytest.approx(0.0, abs=1e-6)
        v = res["verdict"]
        assert v["verdict"] == "PASS"
        assert v["ranking_pass"] and v["magnitude_pass"]
        assert (v["n_pairs"], v["n_tested"], v["n_sign_tested"]) == (1, 1, 1)
        assert v["phases"] == [2, 6]

    def test_anchor_is_the_covering_plan_whatever_the_order(self):
        res = _run([_B_EXACT, _A_EXACT])  # shorter plan first in time
        pair = res["pairs"].iloc[0]
        assert (pair["anchor_plan"], pair["target_plan"]) == (1.0, 2.0)

    def test_clearance_shortfall_is_caught(self):
        # Real lanes stop at the end of green; the window adds 6 s of
        # clearance.  The anchor curve counts full-rate discharge through
        # the target's clearance, so it over-predicts the shorter plan.
        a = (1, 120.0, {2: (60.0, 54.0, 2.0), 6: (60.0, 54.0, 2.0)})
        b = (2, 90.0, {2: (45.0, 39.0, 2.0), 6: (45.0, 39.0, 2.0)})
        res = _run([a, b])
        pair = res["pairs"].iloc[0]
        obs_a = 3600 * 4 * _q(54.0) / 120.0
        obs_b = 3600 * 4 * _q(39.0) / 90.0
        pred_a = 3600 * 4 * _n(60.0, 54.0) / 120.0
        pred_b = 3600 * 4 * _n(45.0, 54.0) / 90.0
        assert pair["predicted_anchor_vph"] == pytest.approx(pred_a)
        assert pair["predicted_target_vph"] == pytest.approx(pred_b)
        exp_err = 100 * (pred_b - pred_a) / pred_a - 100 * (obs_b - obs_a) / obs_a
        assert pair["change_error_pp"] == pytest.approx(exp_err)
        assert exp_err > 10.0  # 3.5 extra vehicles per lane per cycle
        # The bias flips the ranking: the shorter plan serves less, but the
        # model says it serves more.
        assert pair["observed_change_pct"] < -2.0
        assert pair["predicted_change_pct"] > 0.0
        v = res["verdict"]
        assert v["verdict"] == "FAIL"
        assert not v["magnitude_pass"] and not v["ranking_pass"]

    def test_wrong_sign_fails_ranking(self):
        # Plan B discharges faster (h = 1.6 s, e.g. better platoon arrival),
        # so it out-serves A although the model, built on A, says it loses.
        a = (1, 100.0, {2: (50.0, 50.0, 2.0), 6: (50.0, 50.0, 2.0)})
        b = (2, 100.0, {2: (46.0, 46.0, 1.6), 6: (46.0, 46.0, 1.6)})
        res = _run([a, b], change_tol_pp=1000.0)
        pair = res["pairs"].iloc[0]
        assert pair["observed_change_pct"] > 2.0
        assert pair["predicted_change_pct"] < 0.0
        assert bool(pair["sign_tested"]) and not bool(pair["sign_agree"])
        v = res["verdict"]
        assert v["verdict"] == "FAIL"
        assert not v["ranking_pass"] and v["magnitude_pass"]

    def test_mixed_splits_are_not_covered(self):
        a = (1, 100.0, {2: (50.0, 50.0, 2.0), 6: (40.0, 40.0, 2.0)})
        b = (2, 100.0, {2: (40.0, 40.0, 2.0), 6: (50.0, 50.0, 2.0)})
        res = _run([a, b])
        pair = res["pairs"].iloc[0]
        assert pair["status"] == STATUS_NOT_COVERED
        assert np.isnan(pair["predicted_target_vph"])
        assert not bool(pair["sign_tested"])
        assert res["verdict"]["verdict"] == "INCONCLUSIVE"
        assert any("cover" in w for w in res["verdict"]["warnings"])

    def test_split_cover_tolerance(self):
        a = (1, 100.0, {2: (50.0, 50.0, 2.0), 6: (40.0, 40.0, 2.0)})
        b = (2, 100.0, {2: (40.0, 40.0, 2.0), 6: (41.0, 41.0, 2.0)})
        assert _run([a, b])["pairs"].iloc[0]["status"] == STATUS_TESTED
        assert (_run([a, b], split_cover_tol=0.5)["pairs"].iloc[0]["status"]
                == STATUS_NOT_COVERED)

    def test_mutual_cover_anchors_the_longer_total(self):
        a = (1, 100.0, {2: (50.0, 50.0, 2.0), 6: (40.0, 40.0, 2.0)})
        b = (2, 100.0, {2: (50.5, 50.5, 2.0), 6: (40.5, 40.5, 2.0)})
        pair = _run([a, b])["pairs"].iloc[0]
        assert (pair["anchor_plan"], pair["target_plan"]) == (2.0, 1.0)

    def test_dead_band_skips_the_sign_but_not_the_magnitude(self):
        a = (1, 100.0, {2: (50.0, 50.0, 2.0), 6: (50.0, 50.0, 2.0)})
        b = (2, 99.0, {2: (50.0, 50.0, 2.0), 6: (50.0, 50.0, 2.0)})
        res = _run([a, b])
        pair = res["pairs"].iloc[0]
        assert abs(pair["observed_change_pct"]) < 2.0
        assert pair["status"] == STATUS_TESTED and not bool(pair["sign_tested"])
        assert pd.isna(pair["sign_agree"])
        v = res["verdict"]
        assert v["n_tested"] == 1 and v["n_sign_tested"] == 0
        assert v["verdict"] == "INCONCLUSIVE"

    def test_too_few_cycles(self):
        res = _run([_A_EXACT, _B_EXACT], min_plan_cycles=41)
        assert res["pairs"].iloc[0]["status"] == STATUS_TOO_FEW_CYCLES
        assert not res["plans"]["eligible"].any()
        assert res["verdict"]["verdict"] == "INCONCLUSIVE"

    def test_missing_anchor_curve(self):
        # min_cycles above the plan's cycle count: no curve anywhere.
        res = _run([_A_EXACT, _B_EXACT], min_cycles=41)
        assert res["pairs"].iloc[0]["status"] == STATUS_CURVE_MISSING
        assert not res["plans"]["curve_ok"].any()
        assert res["plans"]["insample_vph"].isna().all()
        assert res["verdict"]["verdict"] == "INCONCLUSIVE"
        assert any("no curve" in w for w in res["verdict"]["warnings"])

    def test_three_nested_plans_test_every_pair(self):
        c = (3, 75.0, {2: (36.0, 36.0, 2.0), 6: (36.0, 36.0, 2.0)})
        res = _run([_A_EXACT, _B_EXACT, c])
        pairs = res["pairs"]
        got = set(zip(pairs["anchor_plan"], pairs["target_plan"]))
        assert got == {(1.0, 2.0), (1.0, 3.0), (2.0, 3.0)}
        assert (pairs["status"] == STATUS_TESTED).all()
        assert res["verdict"]["verdict"] == "PASS"

    def test_mean_abs_error_against_tolerance(self):
        a = (1, 120.0, {2: (60.0, 54.0, 2.0), 6: (60.0, 54.0, 2.0)})
        b = (2, 90.0, {2: (45.0, 39.0, 2.0), 6: (45.0, 39.0, 2.0)})
        res = _run([a, b])
        err = abs(res["pairs"].iloc[0]["change_error_pp"])
        assert res["verdict"]["mean_abs_change_error_pp"] == pytest.approx(err)
        assert not res["verdict"]["magnitude_pass"]
        loose = _run([a, b], change_tol_pp=err + 0.01)["verdict"]
        assert loose["magnitude_pass"]
        assert loose["verdict"] == "FAIL"  # still fails on ranking


class TestContract:

    def test_empty_flow_raises(self):
        _, cycles_df = _build([_A_EXACT])
        with pytest.raises(ValueError, match="flow"):
            validate_plans({}, cycles_df)

    def test_datetime_cycle_start_in_flow_raises(self):
        flow, cycles_df = _build([_A_EXACT])
        cdf, vdf = flow[2]
        flow[2] = (cdf.assign(cycle_start=pd.to_datetime(
            cdf["cycle_start"], unit="s", utc=True)), vdf)
        with pytest.raises(ValueError, match="epoch"):
            validate_plans(flow, cycles_df)

    def test_pair_schema(self):
        res = _run([_A_EXACT, _B_EXACT])
        assert list(res["pairs"].columns) == [
            "anchor_plan", "target_plan", "status", "c_anchor", "c_target",
            "observed_anchor_vph", "observed_target_vph",
            "predicted_anchor_vph", "predicted_target_vph",
            "observed_change_pct", "predicted_change_pct", "change_error_pp",
            "sign_tested", "sign_agree",
        ]
        assert list(res["plans"].columns) == [
            "coord_plan", "n_cycles", "c_median", "observed_vph",
            "insample_vph", "insample_pct_error", "eligible", "curve_ok",
            "split_p2", "split_p6", "pass_rate_p2", "pass_rate_p6",
        ]

    def test_single_plan_has_no_pairs(self):
        res = _run([_A_EXACT])
        assert res["pairs"].empty
        assert res["verdict"]["verdict"] == "INCONCLUSIVE"
