# Tests for calculate_cycles across the controller families ACHD runs.
#
# ACHD is moving from Trafficware to Econolite, and the two report Code 31
# (barrier) differently (measured on ACHD 392 / 271 / 269, Aug 2025):
#
#   * Econolite:   one pulse per barrier crossing at the next group's green
#                  start; parameter = barrier number (2 = into the lead group,
#                  1 = into the second group).
#   * Trafficware: one pulse per *ring* crossing, all at the same instant, at
#                  yellow onset of the terminating phases; parameter = ring.
#                  Where ring 2 has no phase in a group (ACHD 269), that
#                  crossing carries a single 31:1.
#
# Whatever the family, a cycle must start at the first green of the RB lead
# group, so a site that changed controllers keeps one cycle definition.

import pandas as pd
import pytest

from atspm.analysis.cycles import calculate_cycles

BASE = 1_700_000_000.0
C = 100.0           # cycle length
B_GREEN = 46.0      # second-group green start within a cycle
LEAD_YELLOW = 40.0  # lead-group yellow onset
B_YELLOW = 94.0     # second-group yellow onset (end-of-cycle barrier)
N_CYCLES = 3

RB = {"RB_R1": "1,2|3,4", "RB_R2": "5,6|7,8"}
EXPECTED_STARTS = [BASE + k * C for k in range(N_CYCLES)]


def _greens(lead_r2=True):
    """Code 1 rows for a lead-in second group, then N_CYCLES full cycles."""
    rows = [(BASE - C + B_GREEN, 1, 4), (BASE - C + B_GREEN, 1, 8)]
    for k in range(N_CYCLES):
        t = BASE + k * C
        rows += [(t, 1, 2)] + ([(t, 1, 6)] if lead_r2 else [])
        rows += [(t + B_GREEN, 1, 4), (t + B_GREEN, 1, 8)]
    return rows


def _econolite(k_range):
    """Barrier-numbered pulses at green start: 31:2 into lead, 31:1 into B."""
    rows = []
    for k in k_range:
        t = BASE + k * C
        rows += [(t, 31, 2), (t + B_GREEN, 31, 1)]
    return rows


def _trafficware(k_range, lead_r2=True):
    """Ring-tagged pulses at yellow onset; one ring only where R2 has no lead phase."""
    rows = []
    for k in k_range:
        t = BASE + k * C
        rows += [(t + LEAD_YELLOW, 31, 1)]
        if lead_r2:
            rows += [(t + LEAD_YELLOW, 31, 2)]
        rows += [(t + B_YELLOW, 31, 1), (t + B_YELLOW, 31, 2)]
    return rows


def _frame(rows):
    return (
        pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )


def _starts(cycles):
    return cycles["cycle_start"].tolist()


class TestBarrierStyles:
    """Every barrier family yields lead-group green starts."""

    def test_econolite_uses_barrier_pulses(self):
        # Lead-in 31:1 at the second group's green before the first cycle.
        rows = _greens() + [(BASE - C + B_GREEN, 31, 1)] + _econolite(range(N_CYCLES))
        cycles = calculate_cycles(_frame(rows), RB)

        assert set(cycles["detection_method"]) == {"barrier_pulse"}
        assert _starts(cycles) == EXPECTED_STARTS

    def test_trafficware_ring_pairs_use_ring_barrier_greens(self):
        rows = _greens() + _trafficware(range(N_CYCLES))
        cycles = calculate_cycles(_frame(rows), RB)

        assert set(cycles["detection_method"]) == {"ring_barrier_config"}
        # Green starts, never the yellow-onset pulse times.
        assert _starts(cycles) == EXPECTED_STARTS

    def test_trafficware_single_ring_crossing(self):
        """ACHD 269 shape: lone 31:1 mid-cycle, 31:1+31:2 at end of cycle."""
        rows = _greens(lead_r2=False) + _trafficware(range(N_CYCLES), lead_r2=False)
        cycles = calculate_cycles(_frame(rows), RB)

        assert set(cycles["detection_method"]) == {"ring_barrier_config"}
        assert _starts(cycles) == EXPECTED_STARTS

    def test_controller_change_mid_window(self):
        """Trafficware then Econolite in one window keeps the same starts."""
        rows = _greens() + _trafficware(range(1)) + _econolite(range(1, N_CYCLES))
        cycles = calculate_cycles(_frame(rows), RB)

        assert set(cycles["detection_method"]) == {"ring_barrier_config"}
        assert _starts(cycles) == EXPECTED_STARTS

    @pytest.mark.parametrize("family", ["econolite", "trafficware"])
    def test_families_agree(self, family):
        lead_in = [(BASE - C + B_GREEN, 31, 1)] if family == "econolite" else []
        pulses = (
            _econolite(range(N_CYCLES)) if family == "econolite"
            else _trafficware(range(N_CYCLES))
        )
        cycles = calculate_cycles(_frame(_greens() + lead_in + pulses), RB)

        assert _starts(cycles) == EXPECTED_STARTS
