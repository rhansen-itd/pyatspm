"""Fixture-connectivity smoke test for the Imperative Shell DB layer.

Target: src/atspm/data/manager.py — DatabaseManager.get_metadata.
Happy-path only: confirms the empty_db fixture (tests/conftest.py) connects
and the metadata table exists / round-trips a row. Edge cases (missing
metadata table -> {"timezone": "US/Mountain"} fallback) are left for a
Fable pass per ROADMAP Session E Phase 2.
"""

from pathlib import Path

from atspm.data.manager import DatabaseManager


class TestGetMetadataSmoke:

    def test_fixture_connects_and_metadata_round_trips(self, empty_db: Path):
        with DatabaseManager(empty_db) as manager:
            manager.set_metadata(
                intersection_id="2068",
                intersection_name="Main St & Oak Ave",
            )
            meta = manager.get_metadata()

        assert meta["intersection_id"] == "2068"
        assert meta["intersection_name"] == "Main St & Oak Ave"
        assert meta["timezone"] == "US/Mountain"

    # TODO(fable): missing metadata table -> get_metadata() falls back to
    # {"timezone": "US/Mountain"} (manager.py:788-789, sqlite3.OperationalError
    # branch). Requires a DB initialised without ever calling init_db(), or a
    # DROP TABLE metadata before querying.

    # TODO(fable): metadata table exists but has zero rows (lock_id=1 never
    # inserted) -> same {"timezone": "US/Mountain"} fallback via the
    # `if not row` branch (manager.py:791-792). Distinct code path from the
    # missing-table case above; assert both are pinned separately.
