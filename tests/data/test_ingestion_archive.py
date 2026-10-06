"""
Tests for IngestionEngine scanning and ingesting monthly .datZ zip archives.
"""

import sqlite3
import struct
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import pytest
import pytz

from atspm.data.ingestion import DatzSource, IngestionEngine
from atspm.data.manager import DatabaseManager, init_db
from atspm.data.raw_archive import pack_monthly_archive

TZ = pytz.timezone("US/Mountain")


def _datz_bytes(clock: str, offsets_deciseconds=(0, 100)) -> bytes:
    """Generate real compressed .datZ bytes with a valid controller preamble."""
    preamble = (
        b"Version #:,3\n"
        b"Controller Data Log Beginning:," + clock.encode() + b"\n"
        b"Phases in use:,1,2,3,4,5,6,7,8\n"
    )
    payload = b"".join(struct.pack(">BBH", 1, 2, o) for o in offsets_deciseconds)
    return zlib.compress(preamble + payload)


def _write_datz(
    raw_dir: Path,
    filename: str,
    clock: str,
    offsets_deciseconds=(0, 100),
) -> Path:
    """Write one compressed .datZ file to disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / filename
    path.write_bytes(_datz_bytes(clock, offsets_deciseconds))
    return path


def _create_zip_archive(
    archive_path: Path,
    members: dict[str, bytes],
) -> Path:
    """Create a raw_YYYY_MM.zip archive with flat members."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return archive_path


@pytest.fixture
def empty_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Test Scanning Candidates
# ---------------------------------------------------------------------------

class TestScanningWithArchives:

    def test_scan_discovers_both_loose_and_zipped(self, empty_db, raw_dir):
        # 1 loose file
        _write_datz(
            raw_dir,
            "ECON_10.0.0.1_2026_06_01_0100.datZ",
            "6/1/2026,01:00:00.0",
        )
        # 1 zip archive with 2 files
        _create_zip_archive(
            raw_dir / "raw_2026_05.zip",
            {
                "ECON_10.0.0.1_2026_05_01_0100.datZ": _datz_bytes("5/1/2026,01:00:00.0"),
                "ECON_10.0.0.1_2026_05_01_0200.datZ": _datz_bytes("5/1/2026,02:00:00.0"),
            },
        )

        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        candidates = engine._scan_all_candidates()

        assert len(candidates) == 3
        # Chronological order
        assert candidates[0].name == "ECON_10.0.0.1_2026_05_01_0100.datZ"
        assert candidates[0].archive_path == raw_dir / "raw_2026_05.zip"

        assert candidates[1].name == "ECON_10.0.0.1_2026_05_01_0200.datZ"
        assert candidates[1].archive_path == raw_dir / "raw_2026_05.zip"

        assert candidates[2].name == "ECON_10.0.0.1_2026_06_01_0100.datZ"
        assert candidates[2].archive_path is None

    def test_deduplication_loose_takes_precedence(self, empty_db, raw_dir):
        # Same file exists loose and in zip archive
        fn = "ECON_10.0.0.1_2026_05_01_0100.datZ"
        _write_datz(raw_dir, fn, "5/1/2026,01:00:00.0")
        _create_zip_archive(
            raw_dir / "raw_2026_05.zip",
            {fn: _datz_bytes("5/1/2026,01:00:00.0")},
        )

        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        candidates = engine._scan_all_candidates()

        assert len(candidates) == 1
        assert candidates[0].name == fn
        assert candidates[0].archive_path is None  # loose takes precedence


# ---------------------------------------------------------------------------
# Test Ingestion from Archives
# ---------------------------------------------------------------------------

class TestArchiveIngestion:

    def test_ingest_pure_zip_archive(self, empty_db, raw_dir):
        _create_zip_archive(
            raw_dir / "raw_2026_05.zip",
            {
                "ECON_10.0.0.1_2026_05_01_0100.datZ": _datz_bytes(
                    "5/1/2026,01:00:00.0", offsets_deciseconds=(0, 50, 100)
                ),
                "ECON_10.0.0.1_2026_05_01_0115.datZ": _datz_bytes(
                    "5/1/2026,01:15:00.0", offsets_deciseconds=(0, 20)
                ),
            },
        )

        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        stats = engine.get_ingestion_stats()
        assert stats["files_processed"] == 2
        assert stats["total_events"] == 5
        assert stats["gap_markers"] == 0

        # Query events table directly
        with DatabaseManager(empty_db) as mgr:
            cur = mgr.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM events")
            assert cur.fetchone()[0] == 5

            cur.execute("SELECT COUNT(*) FROM ingestion_log")
            assert cur.fetchone()[0] == 1

    def test_mixed_loose_and_zipped_continuous_span(self, empty_db, raw_dir):
        # Consecutive intervals: 01:00 in zip, 01:15 loose
        _create_zip_archive(
            raw_dir / "raw_2026_05.zip",
            {
                "ECON_10.0.0.1_2026_05_01_0100.datZ": _datz_bytes(
                    "5/1/2026,01:00:00.0", offsets_deciseconds=(0, 50)
                ),
            },
        )
        _write_datz(
            raw_dir,
            "ECON_10.0.0.1_2026_05_01_0115.datZ",
            "5/1/2026,01:15:00.0",
            offsets_deciseconds=(0, 100),
        )

        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        stats = engine.get_ingestion_stats()
        assert stats["files_processed"] == 2
        assert stats["total_events"] == 4
        assert stats["gap_markers"] == 0

    def test_rebuild_mode_re_ingests_all_from_zip(self, empty_db, raw_dir):
        # Pack files into zip
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_05_01_0100.datZ", "5/1/2026,01:00:00.0"
        )
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_05_01_0115.datZ", "5/1/2026,01:15:00.0"
        )
        pack_res = pack_monthly_archive(raw_dir, year=2026, month=5, remove_loose=True)
        assert pack_res.ok and pack_res.files_removed == 2
        assert not list(raw_dir.glob("*.datZ"))

        # First ingestion
        engine1 = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine1.run()
        assert engine1.get_ingestion_stats()["total_events"] == 4
        assert engine1.get_ingestion_stats()["gap_markers"] == 0

        # Simulate rebuild: clear database tables
        with DatabaseManager(empty_db) as mgr:
            deleted = mgr.clear_ingested_data()
            assert deleted["events"] == 4

        # Second ingestion (rebuild)
        engine2 = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine2.run()

        stats2 = engine2.get_ingestion_stats()
        assert stats2["files_processed"] == 2
        assert stats2["total_events"] == 4

    def test_gap_fill_detects_gap_across_archives(self, empty_db, raw_dir):
        # Zip 1: May 1 01:00
        _create_zip_archive(
            raw_dir / "raw_2026_05.zip",
            {
                "ECON_10.0.0.1_2026_05_01_0100.datZ": _datz_bytes(
                    "5/1/2026,01:00:00.0", offsets_deciseconds=(0, 100)
                ),
            },
        )
        # Loose: May 1 05:00 (4-hour gap opens a gap marker)
        _write_datz(
            raw_dir,
            "ECON_10.0.0.1_2026_05_01_0500.datZ",
            "5/1/2026,05:00:00.0",
            offsets_deciseconds=(0, 100),
        )

        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        stats = engine.get_ingestion_stats()
        assert stats["files_processed"] == 2
        assert stats["gap_markers"] == 1

    def test_multiple_zip_archives_chronological(self, empty_db, raw_dir):
        _create_zip_archive(
            raw_dir / "raw_2026_04.zip",
            {
                "ECON_10.0.0.1_2026_04_30_2345.datZ": _datz_bytes(
                    "4/30/2026,23:45:00.0", offsets_deciseconds=(0, 100)
                ),
            },
        )
        _create_zip_archive(
            raw_dir / "raw_2026_05.zip",
            {
                "ECON_10.0.0.1_2026_05_01_0000.datZ": _datz_bytes(
                    "5/1/2026,00:00:00.0", offsets_deciseconds=(0, 100)
                ),
            },
        )

        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        stats = engine.get_ingestion_stats()
        assert stats["files_processed"] == 2
        assert stats["total_events"] == 4
        assert stats["gap_markers"] == 0

    def test_corrupt_zip_archive_ignored_gracefully(self, empty_db, raw_dir):
        # One valid loose file
        _write_datz(
            raw_dir,
            "ECON_10.0.0.1_2026_05_01_0100.datZ",
            "5/1/2026,01:00:00.0",
        )
        # One corrupt zip
        (raw_dir / "raw_2026_04.zip").write_bytes(b"not-a-valid-zip")

        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        stats = engine.get_ingestion_stats()
        assert stats["files_processed"] == 1
        assert stats["total_events"] == 2

