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
    then mean/peak; n_lanes = distinct physical lanes from Lanes_{dir}_Layout,
    else summed Lanes_{movement} counts, else n_detectors (lane proxy).
critical_movement_analysis: per barrier group the ring with the larger
    demand sum is the critical path (tie → lower ring); per concurrent slot
    (barrier_group, position) the higher-demand phase is slot_critical.
"""

import numpy as np
import pandas as pd

import pytest

from atspm.analysis.critical import (
    LaneConfig,
    critical_movement_analysis,
    movement_phase_map,
    parse_lane_config,
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


# EB approach: L|T|TR — the TR lane is shared by EBT and EBR.
# P2 serves EBT+EBR (3 detectors), P5 serves EBL.  WB has per-movement
# rows only; NB has no lane config at all.
_LANE_MAP_CONFIG = {
    "TM_EBL": "1",
    "TM_EBT": "2,3",
    "TM_EBR": "4",
    "TM_WBT": "5,6,7",
    "TM_WBR": "8",
    "TM_NBT": "9",
    "Det_P5_Stopbar": "1",
    "Det_P2_Stopbar": "2,3,4",
    "Det_P6_Stopbar": "5,6,7,8",
    "Det_P8_Stopbar": "9",
    "Lanes_EBL": "1",
    "Lanes_EBT": "2",
    "Lanes_EBR": "1",
    "Lanes_EB_Layout": "L|T|TR",
    "Lanes_WBT": "2",
    "Lanes_WBR": "1",
}


def _lane_counts() -> pd.DataFrame:
    return pd.DataFrame({
        "EBL": [100.0], "EBT": [500.0], "EBR": [100.0],
        "WBT": [600.0], "WBR": [300.0], "NBT": [400.0],
    })


class TestParseLaneConfig:

    def test_layout_and_counts(self):
        lanes = parse_lane_config(_LANE_MAP_CONFIG)
        assert lanes.layouts == {
            "EB": (frozenset("L"), frozenset("T"), frozenset("TR")),
        }
        assert lanes.counts == {
            "EBL": 1, "EBT": 2, "EBR": 1, "WBT": 2, "WBR": 1,
        }

    def test_blank_and_nan_values_skipped(self):
        lanes = parse_lane_config({
            "Lanes_EBT": "", "Lanes_WBT": float("nan"),
            "Lanes_EB_Layout": None, "TM_EBT": "1",
        })
        assert lanes == LaneConfig({}, {})

    def test_numeric_count_from_db_accepted(self):
        # A config column read back from SQLite may arrive as 2.0
        assert parse_lane_config({"Lanes_EBT": 2.0}).counts == {"EBT": 2}

    @pytest.mark.parametrize("layout", ["L|X", "L||T", "L|TT"])
    def test_bad_layout_raises(self, layout):
        with pytest.raises(ValueError, match="Lanes_EB_Layout"):
            parse_lane_config({"Lanes_EB_Layout": layout})

    @pytest.mark.parametrize("count", ["two", "-1", "1.5"])
    def test_bad_count_raises(self, count):
        with pytest.raises(ValueError, match="Lanes_EBT"):
            parse_lane_config({"Lanes_EBT": count})


class TestPhaseDemandLanes:

    def _out(self, config=_LANE_MAP_CONFIG) -> pd.DataFrame:
        return phase_demand(
            _lane_counts(),
            movement_phase_map(config),
            parse_lane_config(config),
        ).set_index("phase")

    def test_shared_lane_counted_once_from_layout(self):
        # EBT+EBR over L|T|TR: lanes T and TR → 2, not 2+1 = 3 by sum
        out = self._out()
        assert out.loc[2, "n_lanes"] == 2
        assert out.loc[2, "lane_source"] == "layout"
        assert out.loc[2, "n_detectors"] == 3
        assert out.loc[2, "demand_per_lane"] == 300.0
        assert out.loc[2, "peak_per_lane"] == 300.0

    def test_left_lane_from_layout(self):
        out = self._out()
        assert out.loc[5, "n_lanes"] == 1
        assert out.loc[5, "lane_source"] == "layout"

    def test_movement_rows_without_layout_are_summed(self):
        out = self._out()
        assert out.loc[6, "n_lanes"] == 3        # WBT 2 + WBR 1
        assert out.loc[6, "lane_source"] == "movement"
        assert out.loc[6, "demand_per_lane"] == 300.0

    def test_uncovered_phase_falls_back_to_detectors(self):
        out = self._out()
        assert out.loc[8, "lane_source"] == "detectors"
        assert out.loc[8, "n_lanes"] == out.loc[8, "n_detectors"] == 1

    def test_layout_without_counts_rows(self):
        config = {k: v for k, v in _LANE_MAP_CONFIG.items()
                  if not k.startswith("Lanes_EB") or k.endswith("Layout")}
        out = self._out(config)
        assert out.loc[2, "n_lanes"] == 2
        assert out.loc[2, "lane_source"] == "layout"

    def test_layout_serving_none_of_the_turns_falls_back(self):
        config = {**_LANE_MAP_CONFIG, "Lanes_EB_Layout": "L|L"}
        out = self._out(config)
        assert out.loc[2, "lane_source"] == "detectors"
        assert out.loc[2, "n_lanes"] == 3

    def test_no_lane_config_matches_detector_proxy(self):
        config = {k: v for k, v in _LANE_MAP_CONFIG.items()
                  if not k.startswith("Lanes_")}
        with_none = phase_demand(_lane_counts(), movement_phase_map(config))
        with_empty = phase_demand(
            _lane_counts(), movement_phase_map(config),
            parse_lane_config(config),
        )
        pd.testing.assert_frame_equal(with_none, with_empty)
        assert (with_none["n_lanes"] == with_none["n_detectors"]).all()
        assert (with_none["lane_source"] == "detectors").all()

    def test_lanes_carried_into_phase_output(self):
        demand = phase_demand(
            _lane_counts(), movement_phase_map(_LANE_MAP_CONFIG),
            parse_lane_config(_LANE_MAP_CONFIG),
        )
        phase_df, _ = critical_movement_analysis(
            ring_barrier_structure(_RB_CONFIG), demand,
        )
        row = phase_df.set_index("phase").loc[2]
        assert row["n_lanes"] == 2
        assert row["lane_source"] == "layout"
        # Unserved structure phase: zero lanes, empty source
        assert phase_df.set_index("phase").loc[1, "n_lanes"] == 0
        assert phase_df.set_index("phase").loc[1, "lane_source"] == ""


class TestCriticalMovementAnalysis:

    def _demand(self, per_phase: dict) -> pd.DataFrame:
        return pd.DataFrame({
            "phase": list(per_phase),
            "movements": ["M"] * len(per_phase),
            "n_detectors": [1] * len(per_phase),
            "n_lanes": [1] * len(per_phase),
            "lane_source": ["detectors"] * len(per_phase),
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
