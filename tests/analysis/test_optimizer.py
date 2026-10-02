"""Tests for the throughput cycle-length optimizer (Functional Core).

Target: src/atspm/analysis/optimizer.py.  The battery follows
docs/design_optimizer_solver.md D8; each class names its D8 item.

Contract summary
----------------
optimize: saturated phases with curves are allocated by exact max-plus DP;
    other included phases are fixed at max(s_min, sufficiency(C)); both
    rings of a barrier group share the group budget B_g and the budgets
    sum to C.  Infeasible C is skipped, never clamped.  State is
    interior / boundary / infeasible, with a re-measurement directive on
    boundary.  Minimums round up to the grid.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.optimizer import (
    _flat_band,
    _maxplus_convolve,
    _maxplus_fold,
    _backtrack,
    _sufficiency_split,
    optimize,
)


def _curve(n_fn, t_dom, grid=0.5) -> pd.DataFrame:
    """Curve per the D0 contract; ``inst`` is the numeric slope in vph."""
    t = np.arange(int(round(t_dom / grid)) + 1) * grid
    n = np.asarray(n_fn(t), dtype=float)
    inst = np.gradient(n, grid) * 3600.0 if len(t) > 1 else np.zeros(1)
    return pd.DataFrame({"n": n, "inst": inst}, index=pd.Index(t, name="t"))


def _structure(rows) -> pd.DataFrame:
    """``rows`` of (phase, ring, barrier_group); all observed."""
    return pd.DataFrame(
        [dict(phase=p, ring=r, barrier_group=float(g), position=1.0,
              in_config=True, observed_share=1.0, source="config")
         for p, r, g in rows]
    )


def _split(result, phase) -> pd.Series:
    sp = result["splits"]
    return sp.loc[sp["phase"] == phase].iloc[0]


def _linear(rate, startup=0.0):
    return lambda t: rate * np.maximum(0.0, t - startup)


def _expo(a, tau):
    return lambda t: a * (1.0 - np.exp(-t / tau))


# ---------------------------------------------------------------------------
# D8.11 — max-plus primitives
# ---------------------------------------------------------------------------


class TestMaxplus:

    def test_convolve_matches_brute_force(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            a = rng.normal(size=rng.integers(1, 9))
            b = rng.normal(size=rng.integers(1, 9))
            a[rng.random(len(a)) < 0.2] = -np.inf
            c, arg = _maxplus_convolve(a, b)
            for k in range(len(c)):
                cands = [(a[j] + b[k - j], j) for j in range(len(a))
                         if 0 <= k - j < len(b)]
                best = max(v for v, _ in cands)
                assert c[k] == best
                first = min(j for v, j in cands if v == best)
                assert arg[k] == first          # first-hit tie-break

    def test_fold_backtrack_recovers_budgets(self):
        rng = np.random.default_rng(3)
        arrays = [np.cumsum(rng.random(10)) for _ in range(3)]
        vals, args = _maxplus_fold(arrays, 10)
        for k in range(10):
            parts = _backtrack(args, k)
            assert sum(parts) == k
            assert sum(a[p] for a, p in zip(arrays, parts)) == pytest.approx(vals[k])


# ---------------------------------------------------------------------------
# D8.8 — sufficiency
# ---------------------------------------------------------------------------


class TestSufficiencySplit:

    def test_ceils_to_grid(self):
        n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert _sufficiency_split(n, 2.5) == (3, False)
        assert _sufficiency_split(n, 2.0) == (2, False)
        assert _sufficiency_split(n, 0.0) == (0, False)

    def test_beyond_domain_clamps_and_flags(self):
        n = np.array([0.0, 1.0, 2.0])
        assert _sufficiency_split(n, 2.1) == (2, True)

    def test_clamp_flag_reaches_splits(self):
        # Phase 1 (unsat) serves at most 5 veh; demand 720 vph at C=60
        # needs 12 per cycle.
        res = optimize(
            curves={1: _curve(_linear(0.5), 10.0), 2: _curve(_linear(0.5), 100.0)},
            structure_df=_structure([(1, 1, 0), (2, 1, 0)]),
            saturated={2: True}, demand_vph={1: 720.0},
            min_splits={1: 5.0, 2: 5.0}, c_min=60.0, c_max=60.0,
        )
        s1 = _split(res, 1)
        assert s1["sufficiency_at_boundary"]
        assert s1["s_star"] == 10.0
        # Served volume is capped at what the curve can deliver.
        assert res["optimum"]["throughput_total_vph"] == pytest.approx(
            res["optimum"]["throughput_sat_vph"] + 3600.0 * 5.0 / 60.0)


# ---------------------------------------------------------------------------
# D8.1 — concave curves: KKT / equalization
# ---------------------------------------------------------------------------


class TestConcave:

    def test_matches_kkt_and_equalizes_rates(self):
        a1, t1, a2, t2, C = 30.0, 20.0, 20.0, 10.0, 80.0
        res = optimize(
            curves={2: _curve(_expo(a1, t1), C), 4: _curve(_expo(a2, t2), C)},
            structure_df=_structure([(2, 1, 0), (4, 1, 0)]),
            saturated={2: True, 4: True}, demand_vph={},
            min_splits={2: 0.0, 4: 0.0}, c_min=C, c_max=C,
        )
        # KKT: a1/t1·e^(−s1/t1) = a2/t2·e^(−(C−s1)/t2)
        s1 = (np.log(a1 / t1) - np.log(a2 / t2) + C / t2) / (1 / t1 + 1 / t2)
        assert abs(_split(res, 2)["s_star"] - s1) <= 0.5
        assert _split(res, 2)["s_star"] + _split(res, 4)["s_star"] == C
        r2 = _split(res, 2)["end_inst_rate_vph"]
        r4 = _split(res, 4)["end_inst_rate_vph"]
        slope = 3600.0 * a1 / t1 ** 2 * 0.5      # rate change per grid step
        assert abs(r2 - r4) <= 2 * slope
        assert res["optimum"]["state"] == "interior"


# ---------------------------------------------------------------------------
# D8.2 / D8.9 — bumpy curves vs brute force; barrier coupling
# ---------------------------------------------------------------------------


def _bumpy(rng, n_pts):
    n = np.concatenate([[0.0], np.cumsum(rng.random(n_pts - 1) ** 3)])
    return pd.DataFrame({"n": n, "inst": np.gradient(n) * 3600.0},
                        index=pd.Index(np.arange(n_pts, dtype=float), name="t"))


class TestBruteForce:
    # Group 0: ring 1 phases 1, 2; ring 2 phase 5.
    # Group 1: ring 1 phases 3, 4; ring 2 phase 7.
    _ROWS = [(1, 1, 0), (2, 1, 0), (5, 2, 0), (3, 1, 1), (4, 1, 1), (7, 2, 1)]

    def _run(self, seed, C=20.0):
        rng = np.random.default_rng(seed)
        curves = {p: _bumpy(rng, 21) for p, _, _ in self._ROWS}
        mins = {p: 2.0 for p, _, _ in self._ROWS}
        res = optimize(curves, _structure(self._ROWS),
                       saturated={p: True for p in curves}, demand_vph={},
                       min_splits=mins, c_min=C, c_max=C)
        return curves, res

    @staticmethod
    def _brute(curves, C):
        N = {p: c["n"].to_numpy() for p, c in curves.items()}
        best = -np.inf
        for b0 in range(4, int(C) - 4 + 1):
            b1 = int(C) - b0
            g0 = max(N[1][s] + N[2][b0 - s] for s in range(2, b0 - 1)) + N[5][b0]
            g1 = max(N[3][s] + N[4][b1 - s] for s in range(2, b1 - 1)) + N[7][b1]
            best = max(best, g0 + g1)
        return best

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_dp_equals_brute_force(self, seed):
        curves, res = self._run(seed)
        sp = res["splits"]
        total = sp.loc[sp["allocation_basis"] == "optimized", "n_served"].sum()
        assert total == pytest.approx(self._brute(curves, 20.0))
        assert res["optimum"]["throughput_sat_vph"] == pytest.approx(
            3600.0 * total / 20.0)

    def test_rings_share_the_group_budget(self):
        _, res = self._run(11)
        scan = res["scan"].iloc[0]
        sp = res["splits"]
        assert scan["b_g0"] + scan["b_g1"] == 20.0
        for g in (0, 1):
            for r in (1, 2):
                ring = sp.loc[(sp["barrier_group"] == g) & (sp["ring"] == r)]
                assert ring["s_star"].sum() == scan[f"b_g{g}"]

    def test_shifting_budget_between_groups(self):
        # Group 1's phases discharge 4x faster, so it takes all spare time.
        curves = {1: _curve(_linear(0.25), 100.0), 5: _curve(_linear(0.25), 100.0),
                  3: _curve(_linear(1.0), 100.0), 7: _curve(_linear(1.0), 100.0)}
        res = optimize(curves, _structure([(1, 1, 0), (5, 2, 0), (3, 1, 1), (7, 2, 1)]),
                       saturated={p: True for p in curves}, demand_vph={},
                       min_splits={p: 10.0 for p in curves},
                       c_min=80.0, c_max=80.0)
        scan = res["scan"].iloc[0]
        assert scan["b_g0"] == 10.0 and scan["b_g1"] == 70.0


# ---------------------------------------------------------------------------
# D8.3 — single saturated phase
# ---------------------------------------------------------------------------


class TestSingleSaturated:

    def test_gets_the_rest_of_the_ring(self):
        # Phase 1 unsat: 0.5 veh/s, 360 vph → 0.1·C veh → s = 0.2·C.
        C = 90.0
        res = optimize(
            curves={1: _curve(_linear(0.5), 100.0), 2: _curve(_expo(40.0, 30.0), 200.0)},
            structure_df=_structure([(1, 1, 0), (2, 1, 0)]),
            saturated={2: True}, demand_vph={1: 360.0},
            min_splits={1: 5.0, 2: 5.0}, c_min=C, c_max=C,
        )
        assert _split(res, 1)["s_star"] == 18.0
        assert _split(res, 1)["allocation_basis"] == "sufficiency"
        assert _split(res, 2)["s_star"] == 72.0
        expected = 3600.0 * 40.0 * (1 - np.exp(-72.0 / 30.0)) / C
        assert res["optimum"]["throughput_sat_vph"] == pytest.approx(expected)
        assert res["optimum"]["throughput_total_vph"] == pytest.approx(expected + 360.0)


# ---------------------------------------------------------------------------
# D8.4 / D8.5 — minimums and infeasibility
# ---------------------------------------------------------------------------


class TestMinimums:

    def test_minimums_dominate(self):
        curves = {1: _curve(_linear(0.5), 200.0), 2: _curve(_linear(0.5), 200.0)}
        res = optimize(curves, _structure([(1, 1, 0), (2, 1, 0)]),
                       saturated={1: True, 2: True}, demand_vph={},
                       min_splits={1: 40.0, 2: 40.0}, c_min=60.0, c_max=100.0)
        scan = res["scan"].set_index("C")
        assert not scan.loc[79.0, "feasible"]
        assert scan.loc[80.0, "feasible"]
        assert res["optimum"]["c_feasible_min"] == 80.0
        assert scan.loc[60.0:79.0, "throughput_sat_vph"].isna().all()

    def test_minimum_rounds_up_to_grid(self):
        res = optimize({1: _curve(_linear(0.5), 100.0)}, _structure([(1, 1, 0)]),
                       saturated={1: True}, demand_vph={},
                       min_splits={1: 40.2}, c_min=40.0, c_max=41.0, c_step=0.5)
        scan = res["scan"].set_index("C")
        assert not scan.loc[40.0, "feasible"]
        assert scan.loc[40.5, "feasible"]

    def test_infeasible_everywhere(self):
        curves = {1: _curve(_linear(0.5), 100.0), 2: _curve(_linear(0.5), 100.0)}
        res = optimize(curves, _structure([(1, 1, 0), (2, 1, 0)]),
                       saturated={1: True, 2: True}, demand_vph={},
                       min_splits={1: 150.0, 2: 150.0})
        assert res["optimum"]["state"] == "infeasible"
        assert np.isnan(res["optimum"]["c_feasible_min"])
        assert np.isnan(res["optimum"]["c_star"])
        assert res["scan"]["throughput_sat_vph"].isna().all()
        assert res["splits"]["s_star"].isna().all()
        assert res["optimum"]["group_min_at_c_max"] == {0: 300.0}
        assert res["directive"] is None


# ---------------------------------------------------------------------------
# D8.6 / D8.7 — data boundary and C-edge
# ---------------------------------------------------------------------------


class TestBoundary:

    def _ring_full(self, n_fn):
        # C = 60 fills both 30 s domains exactly.
        curves = {1: _curve(n_fn, 30.0), 2: _curve(n_fn, 30.0)}
        return optimize(curves, _structure([(1, 1, 0), (2, 1, 0)]),
                        saturated={1: True, 2: True}, demand_vph={},
                        min_splits={1: 5.0, 2: 5.0}, c_min=60.0, c_max=60.0)

    def test_rising_edge_is_boundary_with_directive(self):
        res = self._ring_full(_linear(0.5))
        assert res["optimum"]["state"] == "boundary"
        d = res["directive"]
        assert d["action"] == "extend_and_remeasure"
        assert [p["phase"] for p in d["phases"]] == [1, 2]
        assert all(p["suggested_split"] == 36.0 for p in d["phases"])
        assert d["c_edge"] is None
        assert _split(res, 1)["at_boundary"] and not _split(res, 1)["surplus"]

    def test_flat_tail_is_surplus_and_interior(self):
        res = self._ring_full(lambda t: 0.5 * np.minimum(t, 20.0))
        assert res["optimum"]["state"] == "interior"
        assert res["directive"] is None
        # D1a: phase 1 stops where its curve flattens; the surplus lands on
        # phase 2, past its domain, flagged as surplus rather than boundary.
        s1, s2 = _split(res, 1), _split(res, 2)
        assert s1["s_star"] == 20.0 and not s1["surplus"]
        assert s2["s_star"] == 40.0
        assert s2["surplus"] and not s2["at_boundary"]

    def test_high_c_edge(self):
        # Startup loss makes throughput rise with C all the way to c_max.
        curves = {1: _curve(_linear(0.5, 5.0), 300.0), 2: _curve(_linear(0.5, 5.0), 300.0)}
        res = optimize(curves, _structure([(1, 1, 0), (2, 1, 0)]),
                       saturated={1: True, 2: True}, demand_vph={},
                       min_splits={1: 5.0, 2: 5.0}, c_min=60.0, c_max=120.0)
        assert res["optimum"]["c_star"] == 120.0
        assert res["optimum"]["state"] == "boundary"
        assert res["directive"]["c_edge"] == {"edge": "high", "c_star": 120.0,
                                              "scan_limit": 120.0}

    def test_low_c_edge(self):
        curves = {1: _curve(_expo(30.0, 20.0), 300.0), 2: _curve(_expo(30.0, 20.0), 300.0)}
        res = optimize(curves, _structure([(1, 1, 0), (2, 1, 0)]),
                       saturated={1: True, 2: True}, demand_vph={},
                       min_splits={1: 5.0, 2: 5.0}, c_min=60.0, c_max=120.0)
        assert res["optimum"]["c_star"] == 60.0
        assert res["directive"]["c_edge"]["edge"] == "low"

    def test_interior_optimum(self):
        # Startup loss plus decaying discharge gives a finite optimum.
        n_fn = lambda t: 40.0 * (1 - np.exp(-np.maximum(0.0, t - 4.0) / 25.0))
        curves = {1: _curve(n_fn, 300.0), 2: _curve(n_fn, 300.0)}
        res = optimize(curves, _structure([(1, 1, 0), (2, 1, 0)]),
                       saturated={1: True, 2: True}, demand_vph={},
                       min_splits={1: 5.0, 2: 5.0}, c_min=20.0, c_max=200.0)
        assert res["optimum"]["state"] == "interior"
        assert 20.0 < res["optimum"]["c_star"] < 200.0
        assert res["directive"] is None

    def test_feasibility_limited_optimum_is_a_warning(self):
        curves = {1: _curve(_expo(30.0, 20.0), 300.0), 2: _curve(_expo(30.0, 20.0), 300.0)}
        res = optimize(curves, _structure([(1, 1, 0), (2, 1, 0)]),
                       saturated={1: True, 2: True}, demand_vph={},
                       min_splits={1: 40.0, 2: 40.0}, c_min=60.0, c_max=120.0)
        assert res["optimum"]["c_star"] == 80.0
        assert res["optimum"]["state"] == "interior"
        assert any(w.startswith("c_star_at_feasibility_limit") for w in res["warnings"])


# ---------------------------------------------------------------------------
# D8.10 — determinism and tie-break
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_flat_curves_push_surplus_to_higher_phase(self):
        flat = lambda t: np.zeros_like(t)
        args = dict(
            curves={1: _curve(flat, 50.0), 2: _curve(flat, 50.0)},
            structure_df=_structure([(1, 1, 0), (2, 1, 0)]),
            saturated={1: True, 2: True}, demand_vph={},
            min_splits={1: 10.0, 2: 10.0}, c_min=60.0, c_max=60.0,
        )
        a, b = optimize(**args), optimize(**args)
        pd.testing.assert_frame_equal(a["splits"], b["splits"])
        pd.testing.assert_frame_equal(a["scan"], b["scan"])
        assert _split(a, 1)["s_star"] == 10.0
        assert _split(a, 2)["s_star"] == 50.0

    def test_ring_without_optimised_phase_gives_surplus_to_top_phase(self):
        res = optimize(
            curves={2: _curve(_linear(0.5), 100.0)},
            structure_df=_structure([(2, 1, 0), (5, 2, 0), (6, 2, 0)]),
            saturated={2: True}, demand_vph={},
            min_splits={2: 5.0, 5: 10.0, 6: 10.0}, c_min=60.0, c_max=60.0,
        )
        assert _split(res, 5)["s_star"] == 10.0
        assert _split(res, 6)["s_star"] == 50.0 and _split(res, 6)["surplus"]
        assert _split(res, 5)["allocation_basis"] == "minimum"


# ---------------------------------------------------------------------------
# Phase handling (D2)
# ---------------------------------------------------------------------------


class TestPhaseHandling:

    def test_excluded_and_missing_curve(self):
        st = _structure([(1, 1, 0), (2, 1, 0), (3, 1, 0)])
        st.loc[st["phase"] == 3, "observed_share"] = 0.0
        unplaced = _structure([(9, 2, 0)]).assign(barrier_group=np.nan)
        res = optimize(
            curves={2: _curve(_linear(0.5), 100.0)},
            structure_df=pd.concat([st, unplaced], ignore_index=True),
            saturated={1: True, 2: True}, demand_vph={},
            min_splits={1: 12.0, 2: 5.0}, c_min=60.0, c_max=60.0,
        )
        assert _split(res, 3)["allocation_basis"] == "excluded"
        assert _split(res, 9)["allocation_basis"] == "excluded"
        s1 = _split(res, 1)
        assert s1["curve_missing"] and s1["s_star"] == 12.0
        assert s1["allocation_basis"] == "minimum"
        assert _split(res, 2)["s_star"] == 48.0
        assert any(w.startswith("curve_missing") for w in res["warnings"])
        assert any(w.startswith("excluded_unconfigured") for w in res["warnings"])

    def test_sufficiency_unverified_without_curve(self):
        res = optimize(
            curves={2: _curve(_linear(0.5), 100.0)},
            structure_df=_structure([(1, 1, 0), (2, 1, 0)]),
            saturated={2: True}, demand_vph={1: 300.0},
            min_splits={1: 8.0, 2: 5.0}, c_min=60.0, c_max=60.0,
        )
        s1 = _split(res, 1)
        assert s1["sufficiency_unverified"] and s1["s_star"] == 8.0

    def test_saturated_vc_from_supplied_demand(self):
        res = optimize(
            curves={2: _curve(_linear(0.5), 100.0)},
            structure_df=_structure([(2, 1, 0)]),
            saturated={2: True}, demand_vph={2: 2000.0},
            min_splits={2: 5.0}, c_min=60.0, c_max=60.0,
        )
        s2 = _split(res, 2)
        assert s2["capacity_vph"] == pytest.approx(1800.0)
        assert s2["vc"] == pytest.approx(2000.0 / 1800.0)
        assert s2["queue_growth_vph"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# D8.12 — contract validation
# ---------------------------------------------------------------------------


class TestContract:

    _ST = _structure([(1, 1, 0), (2, 1, 0)])

    def _call(self, curves, **kw):
        base = dict(saturated={1: True, 2: True}, demand_vph={},
                    min_splits={1: 5.0, 2: 5.0})
        base.update(kw)
        return optimize(curves, self._ST, **base)

    def test_mismatched_grids(self):
        with pytest.raises(ValueError, match="grid"):
            self._call({1: _curve(_linear(0.5), 50.0, grid=0.5),
                        2: _curve(_linear(0.5), 50.0, grid=1.0)})

    def test_non_monotone_n(self):
        bad = _curve(_linear(0.5), 50.0)
        bad.iloc[10, 0] = 100.0
        with pytest.raises(ValueError, match="monotone"):
            self._call({1: bad})

    def test_c_step_not_grid_multiple(self):
        with pytest.raises(ValueError, match="c_step"):
            self._call({1: _curve(_linear(0.5), 50.0)}, c_step=0.3)

    def test_empty_structure(self):
        with pytest.raises(ValueError, match="empty"):
            optimize({}, pd.DataFrame(), {}, {}, {})

    def test_missing_min_split(self):
        with pytest.raises(ValueError, match="min_splits"):
            self._call({1: _curve(_linear(0.5), 50.0)}, min_splits={1: 5.0})


# ---------------------------------------------------------------------------
# D8.13 — flatness
# ---------------------------------------------------------------------------


class TestFlatBand:

    @staticmethod
    def _scan(thr, feasible=None, boundary=None):
        n = len(thr)
        return pd.DataFrame({
            "C": 60.0 + np.arange(n),
            "feasible": feasible if feasible is not None else [True] * n,
            "throughput_sat_vph": thr,
            "any_phase_boundary": boundary if boundary is not None else [False] * n,
        })

    def test_exact_band(self):
        scan = self._scan([90, 95, 99.5, 100, 100, 99.2, 98, 90])
        assert _flat_band(scan, 63.0, 1.0) == (62.0, 65.0, False)

    def test_censored_at_feasibility_edge(self):
        scan = self._scan([np.nan, 99.5, 100, 99.8, 90],
                          feasible=[False, True, True, True, True])
        assert _flat_band(scan, 62.0, 1.0) == (61.0, 63.0, True)

    def test_censored_at_scan_end_and_data_boundary(self):
        scan = self._scan([99.5, 100, 99.9, 99.9],
                          boundary=[False, False, False, True])
        assert _flat_band(scan, 61.0, 1.0) == (60.0, 62.0, True)

    def test_reported_in_optimum(self):
        n_fn = lambda t: 40.0 * (1 - np.exp(-np.maximum(0.0, t - 4.0) / 25.0))
        res = optimize({1: _curve(n_fn, 300.0), 2: _curve(n_fn, 300.0)},
                       _structure([(1, 1, 0), (2, 1, 0)]),
                       saturated={1: True, 2: True}, demand_vph={},
                       min_splits={1: 5.0, 2: 5.0}, c_min=20.0, c_max=200.0)
        o = res["optimum"]
        assert o["flat_c_lo"] <= o["c_star"] <= o["flat_c_hi"]
        assert not o["flat_range_censored"]
        scan = res["scan"].set_index("C")
        peak = o["throughput_sat_vph"]
        band = scan.loc[o["flat_c_lo"]:o["flat_c_hi"], "throughput_sat_vph"]
        assert (band >= 0.99 * peak - 1e-9).all()
        assert scan.loc[o["flat_c_lo"] - 1, "throughput_sat_vph"] < 0.99 * peak
        assert scan.loc[o["flat_c_hi"] + 1, "throughput_sat_vph"] < 0.99 * peak
