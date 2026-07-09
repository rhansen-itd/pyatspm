"""Fixture-connectivity smoke test for data quality reporting.

Target: src/atspm/data/reader.py — check_data_quality.
Happy-path only: confirms the empty_db fixture (tests/conftest.py) connects,
the events/cycles tables exist, and counts round-trip through seeded rows
(including a seeded gap marker). Edge-case assertions around quality-gap
scoring are left for a Fable pass per ROADMAP Session E Phase 2.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

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

    # TODO(fable): completeness_pct scoring around gap markers — e.g. a gap
    # exactly at the window boundary, gap_count > event_count (division
    # floor at 0.0 via max(0.0, ...)), and the zero-event-count guard
    # (reader.py:388-391, max(1, event_count) denominator).
