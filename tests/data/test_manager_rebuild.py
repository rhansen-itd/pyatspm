# Tests for DatabaseManager.clear_ingested_data (imperative shell).
#
# This is the destructive half of `atspm process --rebuild`. The contract that
# matters: ingested and derived rows go, configuration and metadata stay, and
# a missing cycles table (created lazily by CycleProcessor) is not an error.

import sqlite3
from pathlib import Path

import pytest

from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor

from ..conftest import seed_events

BASE = 1_700_000_000.0


def _seed_all(db_path: Path) -> None:
    """Populate every table clear_ingested_data touches, plus config/metadata."""
    seed_events(
        db_path,
        [(BASE, 1, 2), (BASE + 10.0, 8, 2), (BASE + 20.0, 1, 6)],
        gap_at=[BASE + 5.0],
    )
    with DatabaseManager(db_path) as m:
        m.set_metadata(intersection_id="201", intersection_name="Scratch")
        m._insert_config_row({"start_date": "2000-01-01T00:00:00", "end_date": None})
        cur = m.conn.cursor()
        cur.executemany(
            "INSERT INTO cycles (cycle_start, coord_plan, detection_method, "
            "r1_phases, r2_phases) VALUES (?, 0, '', '2', '6')",
            [(BASE,), (BASE + 90.0,)],
        )
        cur.execute(
            "INSERT INTO ingestion_log (span_start, span_end, processed_at, "
            "row_count) VALUES (?, ?, '2026-01-01T00:00:00', 4)",
            (BASE, BASE + 900.0),
        )
        m.conn.commit()


def _counts(db_path: Path) -> dict:
    with DatabaseManager(db_path) as m:
        cur = m.conn.cursor()
        return {
            t: cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("events", "cycles", "ingestion_log", "config", "metadata")
        }


class TestClearIngestedData:

    def test_clears_events_cycles_and_log(self, empty_db: Path):
        _seed_all(empty_db)
        with DatabaseManager(empty_db) as m:
            m.clear_ingested_data()

        after = _counts(empty_db)
        assert after["events"] == 0
        assert after["cycles"] == 0
        assert after["ingestion_log"] == 0

    def test_preserves_config_and_metadata(self, empty_db: Path):
        _seed_all(empty_db)
        with DatabaseManager(empty_db) as m:
            m.clear_ingested_data()
            meta = m.get_metadata()

        after = _counts(empty_db)
        assert after["config"] == 1
        assert after["metadata"] == 1
        assert meta["intersection_id"] == "201"

    def test_returns_rows_deleted_per_table(self, empty_db: Path):
        _seed_all(empty_db)
        with DatabaseManager(empty_db) as m:
            deleted = m.clear_ingested_data()

        # 3 real events + 1 gap marker
        assert deleted == {"events": 4, "cycles": 2, "ingestion_log": 1}

    def test_gap_markers_are_cleared_with_the_events(self, empty_db: Path):
        # Markers are rows in `events`; a rebuild must not leave stale ones
        # behind to fence off data that is about to be re-derived.
        _seed_all(empty_db)
        with DatabaseManager(empty_db) as m:
            m.clear_ingested_data()
            cur = m.conn.cursor()
            remaining = cur.execute(
                "SELECT COUNT(*) FROM events WHERE event_code = -1"
            ).fetchone()[0]

        assert remaining == 0

    def test_is_idempotent(self, empty_db: Path):
        _seed_all(empty_db)
        with DatabaseManager(empty_db) as m:
            m.clear_ingested_data()
            second = m.clear_ingested_data()

        assert second == {"events": 0, "cycles": 0, "ingestion_log": 0}

    def test_missing_cycles_table_is_not_an_error(self, db_path: Path):
        # cycles is created lazily by CycleProcessor, so a DB that has only
        # ever been through init_db() has no such table.
        with DatabaseManager(db_path) as m:
            m.init_db()
            deleted = m.clear_ingested_data()

        assert deleted["cycles"] == 0

    def test_reingestion_after_clear_does_not_duplicate(self, empty_db: Path):
        # The whole point of the rebuild path: INSERT OR IGNORE against the
        # UNIQUE(timestamp, event_code, parameter) index would otherwise add
        # shifted rows *alongside* the originals.
        rows = [(BASE, 1, 2), (BASE + 10.0, 8, 2)]
        seed_events(empty_db, rows)
        with DatabaseManager(empty_db) as m:
            m.clear_ingested_data()
            m.insert_events(rows)
            m.conn.commit()

        assert _counts(empty_db)["events"] == 2

    def test_raises_without_an_active_connection(self, empty_db: Path):
        manager = DatabaseManager(empty_db)
        with pytest.raises(RuntimeError, match="No active connection"):
            manager.clear_ingested_data()
