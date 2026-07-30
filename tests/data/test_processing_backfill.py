# Tests for CycleProcessor.backfill_ring_phases (imperative shell).
#
# Focus: the event fetch window. The ring-string maths itself is covered by
# tests/analysis/test_assign_ring_phases.py — here we only care that the
# right greens reach it and that the right rows get written back.

from pathlib import Path
from typing import List, Tuple

import pytest

from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor

from ..conftest import seed_events


BASE = 1_700_000_000.0  # arbitrary epoch anchor; all offsets are relative


def seed_config(db_path: Path) -> None:
    """Insert one open-ended config row so ``get_config_at_date`` matches.

    No ``RB_*`` columns, so ``assign_ring_phases`` uses its default ring
    membership (R1 = 1-4/9-12, R2 = 5-8/13-16).
    """
    with DatabaseManager(db_path) as m:
        m._insert_config_row({"start_date": "2000-01-01T00:00:00", "end_date": None})
        m.conn.commit()


def seed_cycles(db_path: Path, rows: List[Tuple[float, str, str]]) -> None:
    """Insert ``(cycle_start, r1_phases, r2_phases)`` rows into ``cycles``."""
    with DatabaseManager(db_path) as m:
        m.conn.cursor().executemany(
            "INSERT INTO cycles (cycle_start, coord_plan, detection_method, "
            "r1_phases, r2_phases) VALUES (?, 0, '', ?, ?)",
            rows,
        )
        m.conn.commit()


def read_cycles(db_path: Path) -> dict:
    """Return ``{cycle_start: (r1_phases, r2_phases)}``."""
    with DatabaseManager(db_path) as m:
        cur = m.conn.cursor()
        cur.execute("SELECT cycle_start, r1_phases, r2_phases FROM cycles")
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


class TestBackfillFetchWindow:
    """The fetch window must reach past the last pending cycle_start."""

    def test_last_pending_cycle_is_populated(self, empty_db: Path) -> None:
        """Greens beyond max(cycle_start) + 1s still reach the final row.

        Regression: the window used to stop at ``end_epoch + 1``, so the
        last pending cycle's own greens fell outside it and that row kept
        the 'None' default forever.
        """
        seed_config(empty_db)
        seed_cycles(
            empty_db,
            [(BASE, "None", "None"), (BASE + 100, "None", "None")],
        )
        seed_events(
            empty_db,
            [
                (BASE + 5, 1, 2),
                (BASE + 40, 1, 6),
                # Well past end_epoch + 1 — the old window missed these.
                (BASE + 105, 1, 4),
                (BASE + 140, 1, 8),
            ],
        )

        updated = CycleProcessor(empty_db).backfill_ring_phases()

        assert updated == 2
        cycles = read_cycles(empty_db)
        assert cycles[BASE] == ("2", "6")
        assert cycles[BASE + 100] == ("4", "8")

    def test_window_stops_at_the_next_committed_cycle(self, empty_db: Path) -> None:
        """A populated row after the pending block bounds the fetch window.

        Greens belonging to that later cycle must not be swept backwards
        into the last pending row.
        """
        seed_config(empty_db)
        seed_cycles(
            empty_db,
            [
                (BASE, "None", "None"),
                (BASE + 100, "None", "None"),
                (BASE + 200, "2", "6"),
            ],
        )
        seed_events(
            empty_db,
            [
                (BASE + 5, 1, 2),
                (BASE + 105, 1, 4),
                (BASE + 205, 1, 3),   # belongs to the populated cycle
                (BASE + 240, 1, 7),
            ],
        )

        updated = CycleProcessor(empty_db).backfill_ring_phases()

        assert updated == 2
        cycles = read_cycles(empty_db)
        assert cycles[BASE + 100] == ("4", "None")
        assert cycles[BASE + 200] == ("2", "6")  # untouched

    def test_populated_row_inside_the_span_bounds_its_neighbour(
        self, empty_db: Path
    ) -> None:
        """A populated row between two pending rows still anchors the join.

        Without it in ``cycles_df`` the greens it owns would be attributed
        backwards to the preceding pending row.
        """
        seed_config(empty_db)
        seed_cycles(
            empty_db,
            [
                (BASE, "None", "None"),
                (BASE + 100, "3", "7"),
                (BASE + 200, "None", "None"),
            ],
        )
        seed_events(
            empty_db,
            [
                (BASE + 5, 1, 2),
                (BASE + 105, 1, 3),   # owned by the populated middle cycle
                (BASE + 205, 1, 4),
            ],
        )

        updated = CycleProcessor(empty_db).backfill_ring_phases()

        assert updated == 2
        cycles = read_cycles(empty_db)
        assert cycles[BASE] == ("2", "None")     # not "2,3"
        assert cycles[BASE + 100] == ("3", "7")  # untouched
        assert cycles[BASE + 200] == ("4", "None")

    def test_gap_marker_still_bounds_the_extended_window(
        self, empty_db: Path
    ) -> None:
        """A hard reset inside the widened window still blocks pairing.

        CLAUDE.md §5: sequential pairing must stop at ``event_code = -1``.
        The green after the marker has no cycle_start in its own segment,
        so it attaches to nothing.
        """
        seed_config(empty_db)
        seed_cycles(empty_db, [(BASE, "None", "None")])
        seed_events(
            empty_db,
            [(BASE + 5, 1, 2), (BASE + 80, 1, 6)],
            gap_at=[BASE + 50],
        )

        updated = CycleProcessor(empty_db).backfill_ring_phases()

        assert updated == 1
        assert read_cycles(empty_db)[BASE] == ("2", "None")


class TestBackfillNoOps:
    """Guard paths that must not write anything."""

    def test_nothing_pending_returns_zero(self, empty_db: Path) -> None:
        seed_config(empty_db)
        seed_cycles(empty_db, [(BASE, "2", "6")])
        seed_events(empty_db, [(BASE + 5, 1, 2)])

        assert CycleProcessor(empty_db).backfill_ring_phases() == 0

    def test_no_events_returns_zero(self, empty_db: Path) -> None:
        seed_config(empty_db)
        seed_cycles(empty_db, [(BASE, "None", "None")])

        assert CycleProcessor(empty_db).backfill_ring_phases() == 0
        assert read_cycles(empty_db)[BASE] == ("None", "None")

    def test_missing_config_returns_zero(self, empty_db: Path) -> None:
        seed_cycles(empty_db, [(BASE, "None", "None")])
        seed_events(empty_db, [(BASE + 5, 1, 2)])

        assert CycleProcessor(empty_db).backfill_ring_phases() == 0
