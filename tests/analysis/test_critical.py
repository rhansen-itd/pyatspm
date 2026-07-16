"""Tests for critical movement analysis (Functional Core).

Target: src/atspm/analysis/critical.py.

Contract summary
----------------
ring_barrier_structure: RB_R1/RB_R2 config ('1,2|3,4') → long (ring, phase)
    structure with barrier_group / position; NEMA-standard fallback when the
    config keys are absent; observed_share cross-checked from the cycles
    table's r1_phases/r2_phases strings, with observed-but-unconfigured
    phases appended unplaced (barrier_group NaN, in_config False).
movement_phase_map: TM_* movements assigned to the phase whose stop-bar
    detector set (Det_P{N}_Stopbar / Det_P{N}_Stop_Bar) shares the most
    detectors; no overlap or a tied maximal overlap → phase NA.
phase_demand: hourly-rate movement bins summed to per-phase series first,
    then mean/peak; n_detectors = distinct detectors (lane proxy).
critical_movement_analysis: per barrier group the ring with the larger
    demand sum is the critical path (tie → lower ring); per concurrent slot
    (barrier_group, position) the higher-demand phase is slot_critical.
"""

import numpy as np
import pandas as pd

from atspm.analysis.critical import (
    critical_movement_analysis,
    movement_phase_map,
    phase_demand,
    ring_barrier_structure,
)

# Standard NEMA dual-ring config used across tests
_RB_CONFIG = {"RB_R1": "1,2|3,4", "RB_R2": "5,6|7,8"}

# Config where stop-bar sets identify each movement's phase unambiguously
_MAP_CONFIG = {
    "TM_EBL": "25",
    "TM_EBT": "26,27,28",
    "TM_WBT": "18,19,20",
    "TM_NBR": "24",
    "Det_P5_Stopbar": "25",
    "Det_P2_Stop_Bar": "26,27,28",   # alternate key spelling
    "Det_P6_Stopbar": "18,19,20",
}


def _cycles(r1_strings, r2_strings) -> pd.DataFrame:
    return pd.DataFrame({
        "cycle_start": [float(i) for i in range(len(r1_strings))],
        "r1_phases": r1_strings,
        "r2_phases": r2_strings,
    })


def _structure_row(df: pd.DataFrame, ring: int, phase: int) -> pd.Series:
    match = df.loc[(df["ring"] == ring) & (df["phase"] == phase)]
    assert len(match) == 1
    return match.iloc[0]


class TestRingBarrierStructure:

    def test_config_groups_and_positions(self):
        out = ring_barrier_structure(_RB_CONFIG)

        assert set(out["phase"]) == set(range(1, 9))
        assert (out["source"] == "config").all()
        assert out["in_config"].all()

        row = _structure_row(out, 1, 3)
        assert row["barrier_group"] == 1.0
        assert row["position"] == 1.0

        row = _structure_row(out, 2, 6)
        assert row["barrier_group"] == 0.0
        assert row["position"] == 2.0

    def test_default_fallback_when_config_absent(self):
        out = ring_barrier_structure({})
        assert (out["source"] == "default").all()
        assert set(out.loc[out["ring"] == 1, "phase"]) == {1, 2, 3, 4}
        assert set(out.loc[out["ring"] == 2, "phase"]) == {5, 6, 7, 8}

    def test_observed_share_counts_presence_once(self):
        # Phase 2 in every cycle (re-served twice in one), phase 1 in half
        cycles = _cycles(
            ["1,2", "2,2", "2", "1,2"],
            ["6", "5,6", "6", "6"],
        )
        out = ring_barrier_structure(_RB_CONFIG, cycles)

        assert _structure_row(out, 1, 2)["observed_share"] == 1.0
        assert _structure_row(out, 1, 1)["observed_share"] == 0.5
        assert _structure_row(out, 2, 5)["observed_share"] == 0.25
        assert _structure_row(out, 1, 3)["observed_share"] == 0.0

    def test_observed_unconfigured_phase_appended_unplaced(self):
        cycles = _cycles(["2,9"], ["6"])
        out = ring_barrier_structure(_RB_CONFIG, cycles)

        row = _structure_row(out, 1, 9)
        assert not row["in_config"]
        assert np.isnan(row["barrier_group"])
        assert row["observed_share"] == 1.0

    def test_no_cycles_leaves_share_nan(self):
        out = ring_barrier_structure(_RB_CONFIG)
        assert out["observed_share"].isna().all()


class TestMovementPhaseMap:

    def test_maps_by_detector_overlap_both_key_spellings(self):
        out = movement_phase_map(_MAP_CONFIG).set_index("movement")

        assert out.loc["EBL", "phase"] == 5
        assert out.loc["EBT", "phase"] == 2      # Det_P2_Stop_Bar spelling
        assert out.loc["WBT", "phase"] == 6
        assert out.loc["EBT", "n_matched"] == 3

    def test_no_overlap_is_unmapped(self):
        out = movement_phase_map(_MAP_CONFIG).set_index("movement")
        assert pd.isna(out.loc["NBR", "phase"])
        assert out.loc["NBR", "n_matched"] == 0

    def test_ambiguous_tie_is_unmapped(self):
        config = {
            "TM_EBT": "10,11",
            "Det_P2_Stopbar": "10",
            "Det_P6_Stopbar": "11",
        }
        out = movement_phase_map(config).set_index("movement")
        assert pd.isna(out.loc["EBT", "phase"])

    def test_partial_overlap_prefers_larger(self):
        config = {
            "TM_EBT": "10,11,12",
            "Det_P2_Stopbar": "10,11",
            "Det_P6_Stopbar": "12",
        }
        out = movement_phase_map(config).set_index("movement")
        assert out.loc["EBT", "phase"] == 2
        assert out.loc["EBT", "n_matched"] == 2


class TestPhaseDemand:

    def _counts(self) -> pd.DataFrame:
        # Two bins of hourly rates; quality columns must be ignored
        return pd.DataFrame({
            "EBL": [100.0, 200.0],
            "EBT": [400.0, 600.0],
            "WBT": [500.0, 300.0],
            "TEV": [1000.0, 1100.0],
            "coverage": [1.0, 1.0],
            "data_quality": ["ok", "ok"],
        })

    def test_per_phase_mean_peak_and_lanes(self):
        mmap = movement_phase_map(_MAP_CONFIG)
        out = phase_demand(self._counts(), mmap).set_index("phase")

        assert out.loc[5, "demand_vph"] == 150.0
        assert out.loc[5, "peak_vph"] == 200.0
        assert out.loc[2, "n_detectors"] == 3
        assert out.loc[2, "demand_per_lane"] == round(500.0 / 3, 1)
        assert out.loc[6, "peak_vph"] == 500.0

    def test_phase_peak_from_summed_series(self):
        # Two movements on one phase peaking in different bins: the phase
        # peak is the max of the summed series, not the sum of maxes.
        config = {
            "TM_EBT": "10",
            "TM_EBR": "11",
            "Det_P2_Stopbar": "10,11",
        }
        counts = pd.DataFrame({
            "EBT": [600.0, 100.0],
            "EBR": [100.0, 500.0],
        })
        out = phase_demand(counts, movement_phase_map(config))
        assert out.loc[0, "peak_vph"] == 700.0  # not 1100

    def test_empty_inputs_yield_empty_schema(self):
        mmap = movement_phase_map(_MAP_CONFIG)
        out = phase_demand(pd.DataFrame(), mmap)
        assert out.empty
        assert "demand_per_lane" in out.columns


class TestCriticalMovementAnalysis:

    def _demand(self, per_phase: dict) -> pd.DataFrame:
        return pd.DataFrame({
            "phase": list(per_phase),
            "movements": ["M"] * len(per_phase),
            "n_detectors": [1] * len(per_phase),
            "demand_vph": list(per_phase.values()),
            "peak_vph": list(per_phase.values()),
            "demand_per_lane": list(per_phase.values()),
            "peak_per_lane": list(per_phase.values()),
        })

    def test_critical_ring_per_barrier_group(self):
        structure = ring_barrier_structure(_RB_CONFIG)
        # Group 0: R1 = 100+400 = 500 < R2 = 200+500 = 700 → R2 critical
        # Group 1: R1 = 300+300 = 600 > R2 = 100+200 = 300 → R1 critical
        demand = self._demand({
            1: 100, 2: 400, 3: 300, 4: 300,
            5: 200, 6: 500, 7: 100, 8: 200,
        })
        phase_df, group_df = critical_movement_analysis(
            structure, demand, basis="total"
        )

        crit = group_df.loc[group_df["is_critical_path"]].set_index(
            "barrier_group"
        )
        assert crit.loc[0.0, "ring"] == 2
        assert crit.loc[0.0, "demand_sum"] == 700.0
        assert crit.loc[1.0, "ring"] == 1
        assert crit.loc[1.0, "demand_sum"] == 600.0

        by_phase = phase_df.set_index("phase")
        assert by_phase.loc[[5, 6, 3, 4], "on_critical_path"].all()
        assert not by_phase.loc[[1, 2, 7, 8], "on_critical_path"].any()

    def test_slot_critical_phase(self):
        structure = ring_barrier_structure(_RB_CONFIG)
        demand = self._demand({
            1: 100, 2: 400, 3: 300, 4: 300,
            5: 200, 6: 500, 7: 100, 8: 200,
        })
        phase_df, _ = critical_movement_analysis(
            structure, demand, basis="total"
        )
        by_phase = phase_df.set_index("phase")

        # Slots pair by position across rings: (1,5), (2,6), (3,7), (4,8)
        assert not by_phase.loc[1, "slot_critical"]
        assert by_phase.loc[5, "slot_critical"]
        assert by_phase.loc[6, "slot_critical"]
        assert by_phase.loc[3, "slot_critical"]
        assert by_phase.loc[4, "slot_critical"]

    def test_ring_tie_resolves_to_lower_ring(self):
        structure = ring_barrier_structure(_RB_CONFIG)
        demand = self._demand({
            1: 300, 2: 300, 5: 300, 6: 300,
            3: 100, 4: 100, 7: 100, 8: 100,
        })
        _, group_df = critical_movement_analysis(
            structure, demand, basis="total"
        )
        crit = group_df.loc[group_df["is_critical_path"]].set_index(
            "barrier_group"
        )
        assert crit.loc[0.0, "ring"] == 1

    def test_phase_without_demand_flagged(self):
        structure = ring_barrier_structure(_RB_CONFIG)
        demand = self._demand({2: 400, 6: 500})
        phase_df, _ = critical_movement_analysis(
            structure, demand, basis="total"
        )
        by_phase = phase_df.set_index("phase")

        assert not by_phase.loc[1, "has_demand"]
        assert by_phase.loc[1, "demand_vph"] == 0.0
        assert by_phase.loc[2, "has_demand"]

    def test_unplaced_phase_excluded_from_groups(self):
        cycles = _cycles(["2,9"], ["6"])
        structure = ring_barrier_structure(_RB_CONFIG, cycles)
        demand = self._demand({2: 400, 6: 500, 9: 999})
        phase_df, group_df = critical_movement_analysis(
            structure, demand, basis="total"
        )

        assert 9 in set(phase_df["phase"])
        assert not phase_df.set_index("phase").loc[9, "on_critical_path"]
        # 999 vph on phase 9 must not leak into any ring sum
        assert group_df["demand_sum"].max() == 500.0

    def test_empty_structure_yields_empty_schemas(self):
        phase_df, group_df = critical_movement_analysis(
            pd.DataFrame(), self._demand({2: 400}), basis="total"
        )
        assert phase_df.empty and group_df.empty
        assert "on_critical_path" in phase_df.columns
        assert "is_critical_path" in group_df.columns
