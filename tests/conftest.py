# Fixtures
#
# Shell-side plumbing only (CLAUDE.md §4): building/tearing down a throwaway
# SQLite DB and seeding raw rows into it. No assertions live here.

from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path to a throwaway per-test SQLite DB (not yet initialised)."""
    return tmp_path / "test_intersection.db"


@pytest.fixture
def empty_db(db_path: Path) -> Path:
    """A throwaway SQLite DB with the full real schema, WAL mode enabled.

    Builds the schema through the same code the app uses rather than
    hand-written DDL: ``DatabaseManager.init_db()`` creates events/config/
    metadata/ingestion_log, and ``CycleProcessor._init_cycles_table()``
    creates cycles (lazily created by the app itself — see
    processing.py:_init_cycles_table). Nothing is torn down explicitly since
    tmp_path is per-test and discarded by pytest.

    Returns:
        Path to the initialised DB file.
    """
    with DatabaseManager(db_path) as manager:
        manager.init_db()
    CycleProcessor(db_path)._init_cycles_table()
    return db_path


def seed_events(
    db_path: Path,
    events: List[Tuple[float, int, int]],
    gap_at: Optional[List[float]] = None,
) -> None:
    """Insert raw event rows into an already-initialised DB.

    Args:
        db_path: Path to a DB already built via ``empty_db``.
        events:  List of ``(timestamp, event_code, parameter)`` tuples.
        gap_at:  Optional list of timestamps at which to insert an
                 ``event_code = -1`` hard-reset gap marker (CLAUDE.md §5 —
                 sequential/duration logic must stop at these).
    """
    rows = list(events)
    rows.extend((ts, -1, 0) for ts in gap_at or [])
    with DatabaseManager(db_path) as manager:
        manager.insert_events(rows)
