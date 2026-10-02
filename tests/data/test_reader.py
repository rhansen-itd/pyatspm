"""Fixture-connectivity smoke test for data quality reporting.

Target: src/atspm/data/reader.py — check_data_quality.
Smoke test that the empty_db fixture (tests/conftest.py) connects and counts
round-trip, plus the scoring edge cases: markers on the window edges, the
zero-event and gaps-exceed-events floors, and clock-step fences kept apart
from comms gaps.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from atspm.analysis.decoders import CLOCK_STEP_FENCE_PARAM
from atspm.data.reader import check_data_quality

from ..conftest import seed_events

START = datetime(2026, 1, 5, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
START_EPOCH = START.timestamp()


class TestCheckDataQualitySmoke:

    def test_fixture_connects_and_counts_seeded_rows(self, empty_db: Path):
        seed_events(
            empty_db,
            events=[(START_EPOCH + 60.0, 82, 1), (START_EPOCH + 120.0, 82, 1)],
            gap_at=[START_EPOCH + 90.0],
        )

        result = check_data_quality(empty_db, START, END)

        assert result["event_count"] == 2
        assert result["gap_count"] == 1
        assert result["cycle_count"] == 0
        assert result["has_cycles"] is False



class TestCheckDataQualityEdges:
    """Window edges, scoring floors and the marker-kind split (CLAUDE.md §5)."""

    def test_gap_on_window_start_counts_and_on_window_end_does_not(self, empty_db: Path):
        seed_events(empty_db, events=[(START_EPOCH + 60.0, 82, 1)],
                    gap_at=[START_EPOCH, END.timestamp()])
        result = check_data_quality(empty_db, START, END)
        assert result["gap_count"] == 1

    def test_no_gaps_is_complete_even_with_no_events(self, empty_db: Path):
        result = check_data_quality(empty_db, START, END)
        assert result["event_count"] == 0
        assert result["completeness_pct"] == 100.0

    def test_gaps_with_no_events_score_zero_not_divide_by_zero(self, empty_db: Path):
        seed_events(empty_db, events=[], gap_at=[START_EPOCH + 10.0])
        assert check_data_quality(empty_db, START, END)["completeness_pct"] == 0.0

    def test_more_gaps_than_events_floors_at_zero(self, empty_db: Path):
        seed_events(empty_db, events=[(START_EPOCH + 60.0, 82, 1)],
                    gap_at=[START_EPOCH + 10.0, START_EPOCH + 20.0, START_EPOCH + 30.0])
        assert check_data_quality(empty_db, START, END)["completeness_pct"] == 0.0

    def test_score_is_gaps_per_event(self, empty_db: Path):
        events = [(START_EPOCH + 60.0 + i, 82, 1) for i in range(4)]
        seed_events(empty_db, events=events, gap_at=[START_EPOCH + 10.0])
        assert check_data_quality(empty_db, START, END)["completeness_pct"] == 75.0

    def test_clock_step_fences_are_reported_apart_and_cost_nothing(self, empty_db: Path):
        seed_events(empty_db, events=[(START_EPOCH + 60.0, 82, 1),
                                      (START_EPOCH + 70.0, -1, CLOCK_STEP_FENCE_PARAM)])
        result = check_data_quality(empty_db, START, END)
        assert result["gap_count"] == 0
        assert result["clock_step_count"] == 1
        assert result["event_count"] == 1
        assert result["completeness_pct"] == 100.0
