"""Adversarial edge-case tests for cycle validation (Functional Core).

Target: src/atspm/analysis/cycles.py — validate_cycles.
Pure function: takes a cycles DataFrame, returns (is_valid, warnings).
No mocking, no DB.

Gap-rule note (CLAUDE.md §5): validate_cycles receives only cycle_start
values — the cycles DataFrame carries no gap information, so the function
currently diffs every consecutive pair of cycle starts, including pairs
that straddle an event_code == -1 hard reset in the underlying events.
That pairing across a discontinuity is exposed (not papered over) by the
xfail test at the bottom of this module.
"""

import pandas as pd
import pytest

from atspm.analysis.cycles import validate_cycles

T0 = 1_609_459_200.0  # 2021-01-01 00:00:00 UTC


def _cycles(starts) -> pd.DataFrame:
    return pd.DataFrame({'cycle_start': list(starts)})


class TestValidateCyclesDegenerate:

    def test_zero_cycles_is_valid_with_no_warnings(self):
        is_valid, warnings = validate_cycles(_cycles([]))
        assert is_valid is True
        assert warnings == []

    def test_single_cycle_is_valid_with_no_warnings(self):
        # One cycle yields no inter-cycle diffs, so nothing can warn.
        is_valid, warnings = validate_cycles(_cycles([T0]))
        assert is_valid is True
        assert warnings == []

    def test_nan_cycle_start_does_not_crash(self):
        is_valid, warnings = validate_cycles(_cycles([T0, float('nan'), T0 + 90.0]))
        assert isinstance(is_valid, bool)
        assert isinstance(warnings, list)


class TestValidateCyclesHappyAndBoundary:

    def test_uniform_90s_cycles_are_valid(self):
        is_valid, warnings = validate_cycles(_cycles([T0 + 90.0 * i for i in range(10)]))
        assert is_valid is True
        assert warnings == []

    def test_gap_exactly_at_min_length_is_not_flagged(self):
        # Comparison is strict (<), so a diff of exactly min_cycle_length passes.
        is_valid, warnings = validate_cycles(
            _cycles([T0, T0 + 10.0]), min_cycle_length=10.0
        )
        assert is_valid is True
        assert warnings == []

    def test_gap_exactly_at_max_length_is_not_flagged(self):
        is_valid, warnings = validate_cycles(
            _cycles([T0, T0 + 300.0]), max_cycle_length=300.0
        )
        assert is_valid is True
        assert warnings == []

    def test_too_short_cycle_is_flagged(self):
        is_valid, warnings = validate_cycles(_cycles([T0, T0 + 5.0, T0 + 95.0]))
        assert is_valid is False
        assert len(warnings) == 1
        assert "shorter than 10.0s" in warnings[0]
        assert "1 cycles" in warnings[0]

    def test_too_long_cycle_is_flagged(self):
        is_valid, warnings = validate_cycles(_cycles([T0, T0 + 90.0, T0 + 90.0 + 400.0]))
        assert is_valid is False
        assert len(warnings) == 1
        assert "longer than 300.0s" in warnings[0]

    def test_custom_thresholds_are_respected(self):
        # 90 s cycles: fine by default, too long when max is tightened to 60.
        starts = _cycles([T0, T0 + 90.0, T0 + 180.0])
        assert validate_cycles(starts)[0] is True
        is_valid, warnings = validate_cycles(starts, max_cycle_length=60.0)
        assert is_valid is False
        assert "longer than 60.0s" in warnings[0]


class TestValidateCyclesOrderingAndDuplicates:

    def test_out_of_order_cycles_are_flagged_but_lengths_use_sorted_order(self):
        # Sorted, the diffs are all 90 s — only the ordering warning fires,
        # proving lengths are computed on sorted starts, not raw row order.
        is_valid, warnings = validate_cycles(_cycles([T0 + 90.0, T0, T0 + 180.0]))
        assert is_valid is False
        assert warnings == ["Cycles are not sorted by cycle_start"]

    def test_duplicate_cycle_start_flags_duplicate_and_zero_length(self):
        # A duplicated start produces a 0 s diff, so the same defect fires
        # both the duplicate warning and the too-short warning.
        is_valid, warnings = validate_cycles(_cycles([T0, T0, T0 + 90.0]))
        assert is_valid is False
        assert len(warnings) == 2
        assert "1 duplicate cycle start times" in warnings[0]
        assert "shorter than" in warnings[1]

    def test_triplicate_start_counts_two_duplicates(self):
        _, warnings = validate_cycles(_cycles([T0, T0, T0, T0 + 90.0]))
        assert any("2 duplicate cycle start times" in w for w in warnings)


class TestValidateCyclesGapRule:

    @pytest.mark.xfail(
        reason=(
            "CLAUDE.md §5 gap rule: duration/pairing logic must stop at an "
            "event_code == -1 hard reset. validate_cycles has no gap "
            "awareness — its signature only receives cycle_start values — so "
            "it diffs the last pre-gap cycle against the first post-gap "
            "cycle and misreports the data outage as an over-long cycle. "
            "Fixing this requires passing gap information (e.g. gap "
            "timestamps or the events frame) into validate_cycles."
        ),
        strict=True,
    )
    def test_cycles_straddling_a_hard_reset_are_not_paired_across_it(self):
        # Healthy 90 s cycles, then an event_code == -1 hard reset at
        # T0 + 200 (an hour of missing data), then healthy cycles resume.
        pre_gap = [T0, T0 + 90.0, T0 + 180.0]
        post_gap = [T0 + 3800.0, T0 + 3890.0, T0 + 3980.0]
        is_valid, warnings = validate_cycles(_cycles(pre_gap + post_gap))

        # The 3620 s pre/post-gap diff is missing data, not a cycle length.
        # Correct behavior: no cycle-length warning derived from that pair.
        assert is_valid is True
        assert warnings == []
