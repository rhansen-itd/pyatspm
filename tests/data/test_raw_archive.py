"""
Unit tests for monthly .datZ zip packaging and verification (atspm.data.raw_archive).
"""

import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from atspm.data.raw_archive import (
    PackResult,
    list_monthly_candidates,
    pack_intersection_raw,
    pack_monthly_archive,
    parse_datz_month,
    verify_archive,
)


def _make_dummy_datz(path: Path, content: bytes = b"datz-test-payload") -> Path:
    """Helper to create a dummy .datZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compressed data like real .datZ
    path.write_bytes(zlib.compress(content))
    return path


# ---------------------------------------------------------------------------
# parse_datz_month
# ---------------------------------------------------------------------------

class TestParseDatzMonth:

    def test_standard_econ_filename(self):
        fn = "ECON_10.0.0.1_2026_06_20_0400.datZ"
        assert parse_datz_month(fn) == (2026, 6)

    def test_january_and_december(self):
        assert parse_datz_month("DET_2025_01_01_0000.datZ") == (2025, 1)
        assert parse_datz_month("DET_2025_12_31_2345.datZ") == (2025, 12)

    def test_accepts_path_object(self):
        p = Path("/some/dir/ECON_10.0.0.1_2026_08_15_1200.datZ")
        assert parse_datz_month(p) == (2026, 8)

    def test_invalid_filename_returns_none(self):
        assert parse_datz_month("not_a_datz_file.txt") is None
        assert parse_datz_month("ECON_10.0.0.1.datZ") is None
        assert parse_datz_month("raw_2026_06.zip") is None

    def test_invalid_month_returns_none(self):
        assert parse_datz_month("ECON_10.0.0.1_2026_00_20_0400.datZ") is None
        assert parse_datz_month("ECON_10.0.0.1_2026_13_20_0400.datZ") is None


# ---------------------------------------------------------------------------
# list_monthly_candidates
# ---------------------------------------------------------------------------

class TestListMonthlyCandidates:

    def test_groups_by_month_and_sorts(self, tmp_path):
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_05_10_0200.datZ")
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_05_10_0100.datZ")
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_01_0000.datZ")
        (tmp_path / "other.txt").write_text("ignore me")

        candidates = list_monthly_candidates(tmp_path, include_current=True)
        assert set(candidates.keys()) == {(2026, 5), (2026, 6)}
        assert len(candidates[(2026, 5)]) == 2
        # Verify sorted
        assert candidates[(2026, 5)][0].name == "ECON_10.0.0.1_2026_05_10_0100.datZ"
        assert candidates[(2026, 5)][1].name == "ECON_10.0.0.1_2026_05_10_0200.datZ"

    def test_excludes_current_month_by_default(self, tmp_path):
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_05_10_0100.datZ")
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_01_0000.datZ")

        # Set current month to (2026, 6)
        candidates = list_monthly_candidates(
            tmp_path,
            include_current=False,
            current_month=(2026, 6),
        )
        assert (2026, 6) not in candidates
        assert (2026, 5) in candidates

    def test_includes_current_month_when_requested(self, tmp_path):
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_05_10_0100.datZ")
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_01_0000.datZ")

        candidates = list_monthly_candidates(
            tmp_path,
            include_current=True,
            current_month=(2026, 6),
        )
        assert (2026, 6) in candidates
        assert (2026, 5) in candidates

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        assert list_monthly_candidates(tmp_path / "nonexistent") == {}


# ---------------------------------------------------------------------------
# verify_archive
# ---------------------------------------------------------------------------

class TestVerifyArchive:

    def test_verify_valid_archive(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        content = b"sample-binary-payload"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("file1.datZ", content)

        ok, detail = verify_archive(zip_path, expected_files={"file1.datZ": len(content)})
        assert ok is True
        assert detail == "ok"

    def test_verify_missing_archive(self, tmp_path):
        ok, detail = verify_archive(tmp_path / "missing.zip")
        assert ok is False
        assert "not found" in detail

    def test_verify_corrupted_crc(self, tmp_path):
        zip_path = tmp_path / "corrupt.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("file1.datZ", b"original-content")

        # Corrupt bytes in the file
        raw = zip_path.read_bytes()
        # Mutate payload byte
        corrupted = raw.replace(b"original-content", b"corruptd-content")
        zip_path.write_bytes(corrupted)

        ok, detail = verify_archive(zip_path)
        assert ok is False
        assert "CRC" in detail or "Bad zip" in detail or "Corrupted" in detail

    def test_verify_size_mismatch(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("file1.datZ", b"hello")

        ok, detail = verify_archive(zip_path, expected_files={"file1.datZ": 999})
        assert ok is False
        assert "Size mismatch" in detail

    def test_verify_missing_member(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("file1.datZ", b"hello")

        ok, detail = verify_archive(zip_path, expected_files={"file2.datZ": 5})
        assert ok is False
        assert "Missing member" in detail


# ---------------------------------------------------------------------------
# pack_monthly_archive
# ---------------------------------------------------------------------------

class TestPackMonthlyArchive:

    def test_pack_and_remove_loose_files(self, tmp_path):
        f1 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0100.datZ", b"file-1")
        f2 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0200.datZ", b"file-2")
        f1_size = f1.stat().st_size
        f2_size = f2.stat().st_size

        res = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=True)

        assert res.ok is True
        assert res.files_packed == 2
        assert res.bytes_packed == f1_size + f2_size
        assert res.files_removed == 2
        assert res.archive_path == tmp_path / "raw_2026_06.zip"
        assert res.archive_path.exists()

        # Loose files must be deleted
        assert not f1.exists()
        assert not f2.exists()

        # Inspect zip
        with zipfile.ZipFile(res.archive_path, "r") as zf:
            infolist = zf.infolist()
            assert len(infolist) == 2
            # Flat names
            assert {info.filename for info in infolist} == {
                "ECON_10.0.0.1_2026_06_15_0100.datZ",
                "ECON_10.0.0.1_2026_06_15_0200.datZ",
            }
            # ZIP_STORED (0)
            assert all(info.compress_type == zipfile.ZIP_STORED for info in infolist)
            assert zf.read("ECON_10.0.0.1_2026_06_15_0100.datZ") == zlib.compress(b"file-1")

    def test_pack_keep_loose(self, tmp_path):
        f1 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0100.datZ", b"file-1")

        res = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=False)

        assert res.ok is True
        assert res.files_packed == 1
        assert res.files_removed == 0
        assert f1.exists()
        assert res.archive_path.exists()

    def test_dry_run_makes_no_changes(self, tmp_path):
        f1 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0100.datZ", b"file-1")
        f1_size = f1.stat().st_size

        res = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=True, dry_run=True)

        assert res.ok is True
        assert res.files_packed == 1
        assert res.bytes_packed == f1_size
        assert res.files_removed == 1
        assert not (tmp_path / "raw_2026_06.zip").exists()
        assert f1.exists()

    def test_idempotent_append_and_pruning(self, tmp_path):
        # Initial pack with keep_loose=True
        f1 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0100.datZ", b"file-1")
        res1 = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=False)
        assert res1.ok is True and res1.files_packed == 1

        # Add second file loose, while f1 is still loose
        f2 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0200.datZ", b"file-2")

        # Second pack with remove_loose=True
        res2 = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=True)
        assert res2.ok is True
        assert res2.files_packed == 1  # Only f2 was newly packed
        assert res2.files_removed == 2  # Both f1 and f2 removed

        # Check archive contains exactly 2 members
        with zipfile.ZipFile(tmp_path / "raw_2026_06.zip", "r") as zf:
            assert len(zf.namelist()) == 2

    def test_no_files_to_pack(self, tmp_path):
        res = pack_monthly_archive(tmp_path, year=2026, month=6)
        assert res.ok is True
        assert res.files_packed == 0
        assert res.files_removed == 0
        assert "no files" in res.detail

    def test_corrupt_existing_archive_aborts_without_deleting_loose(self, tmp_path):
        archive = tmp_path / "raw_2026_06.zip"
        archive.write_bytes(b"corrupt-non-zip-data")

        f1 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0100.datZ", b"file-1")

        res = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=True)
        assert res.ok is False
        assert "Existing archive" in res.detail
        # Loose file must NOT be deleted
        assert f1.exists()

    def test_size_mismatch_aborts_without_deleting_loose(self, tmp_path):
        archive = tmp_path / "raw_2026_06.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("ECON_10.0.0.1_2026_06_15_0100.datZ", b"short")

        # Loose file has different size
        f1 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0100.datZ", b"much-longer-content")

        res = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=True)
        assert res.ok is False
        assert "differs from the loose file" in res.detail
        assert f1.exists()

    def test_same_size_different_content_aborts_without_deleting_loose(self, tmp_path):
        name = "ECON_10.0.0.1_2026_06_15_0100.datZ"
        archive = tmp_path / "raw_2026_06.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(name, b"AAAA")
        f1 = tmp_path / name
        f1.write_bytes(b"BBBB")

        res = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=True)
        assert res.ok is False
        assert f1.read_bytes() == b"BBBB"

    def test_interrupted_append_leaves_existing_archive_intact(self, tmp_path, monkeypatch):
        f1 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0100.datZ", b"file-1")
        assert pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=True).ok
        assert not f1.exists()
        archive = tmp_path / "raw_2026_06.zip"
        before = archive.read_bytes()

        f2 = _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_16_0100.datZ", b"file-2")

        # Members get written but the central directory never does, as on a
        # crash or power loss mid-pack.
        def _crash(self, *a, **k):
            raise OSError("power loss")

        monkeypatch.setattr(zipfile.ZipFile, "_write_end_record", _crash)
        res = pack_monthly_archive(tmp_path, year=2026, month=6, remove_loose=True)

        assert res.ok is False
        assert archive.read_bytes() == before  # month already pruned stays readable
        assert f2.exists()
        assert list(tmp_path.glob("*.packtmp")) == []

    def test_append_to_existing_archive_keeps_prior_members(self, tmp_path):
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_15_0100.datZ", b"file-1")
        assert pack_monthly_archive(tmp_path, year=2026, month=6).ok
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_06_16_0100.datZ", b"file-2")
        res = pack_monthly_archive(tmp_path, year=2026, month=6)

        assert res.ok and res.files_packed == 1 and res.files_removed == 1
        with zipfile.ZipFile(tmp_path / "raw_2026_06.zip") as zf:
            assert sorted(zf.namelist()) == [
                "ECON_10.0.0.1_2026_06_15_0100.datZ",
                "ECON_10.0.0.1_2026_06_16_0100.datZ",
            ]
        assert list(tmp_path.glob("*.packtmp")) == []


# ---------------------------------------------------------------------------
# pack_intersection_raw
# ---------------------------------------------------------------------------

class TestPackIntersectionRaw:

    def test_packs_multiple_months(self, tmp_path):
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_04_15_0100.datZ")
        _make_dummy_datz(tmp_path / "ECON_10.0.0.1_2026_05_15_0100.datZ")

        results = pack_intersection_raw(tmp_path, include_current=True, remove_loose=True)
        assert len(results) == 2
        assert all(r.ok for r in results)
        assert (tmp_path / "raw_2026_04.zip").exists()
        assert (tmp_path / "raw_2026_05.zip").exists()
        assert list(tmp_path.glob("*.datZ")) == []
