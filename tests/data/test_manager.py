"""Fixture-connectivity smoke test for the Imperative Shell DB layer.

Target: src/atspm/data/manager.py — DatabaseManager.get_metadata.
Confirms the empty_db fixture (tests/conftest.py) round-trips a metadata row,
and pins both fallbacks to {"timezone": "US/Mountain"}: metadata table
missing (the SELECT raises) and present but empty (no lock_id = 1 row).
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

    def test_missing_metadata_table_falls_back_to_default_timezone(self, db_path: Path):
        # A DB that never went through init_db(): the SELECT itself raises.
        with DatabaseManager(db_path) as manager:
            manager.conn.execute("CREATE TABLE events (timestamp REAL)")
            assert manager.get_metadata() == {"timezone": "US/Mountain"}

    def test_empty_metadata_table_falls_back_to_default_timezone(self, empty_db: Path):
        # Table present, lock_id = 1 row never written: a distinct branch.
        with DatabaseManager(empty_db) as manager:
            manager.conn.execute("DELETE FROM metadata")
            assert manager.get_metadata() == {"timezone": "US/Mountain"}
