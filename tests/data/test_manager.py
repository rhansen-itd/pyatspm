"""Fixture-connectivity smoke test for the Imperative Shell DB layer.

Target: src/atspm/data/manager.py — DatabaseManager.get_metadata.
Confirms the empty_db fixture (tests/conftest.py) round-trips a metadata row,
and pins both fallbacks to {"timezone": "US/Mountain"}: metadata table
missing (the SELECT raises) and present but empty (no lock_id = 1 row).
"""

from datetime import datetime
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


class TestImportConfigLanes:

    def test_lanes_rows_become_config_columns(self, empty_db: Path, tmp_path: Path):
        csv = tmp_path / "int_cfg.csv"
        csv.write_text(
            ",,1/1/2020,6/1/2026\n"
            "TM:,EBT,63,63\n"
            "Lanes:,EBL,,1\n"
            "Lanes:,EBT,,2\n"
            "Lanes:,EB Layout,,L|T|TR\n"
        )
        with DatabaseManager(empty_db) as m:
            m.import_config(csv)
            old = m.get_config_at_date(datetime(2021, 1, 1))
            new = m.get_config_at_date(datetime(2026, 7, 1))

        assert new["Lanes_EBL"] == "1"
        assert new["Lanes_EBT"] == "2"
        assert new["Lanes_EB_Layout"] == "L|T|TR"
        # Only the latest column is filled; the earlier period has none
        assert not old.get("Lanes_EB_Layout")
