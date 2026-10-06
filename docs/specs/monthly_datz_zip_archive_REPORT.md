# Completion Report: Monthly .datZ Zip Archive and Sync Workflow

**Date:** 2026-10-06  
**Status:** Completed  
**Spec Reference:** `docs/specs/monthly_datz_zip_archive.md`

---

## 1. Summary of Changes

To resolve filesystem metadata and transfer latency bottlenecks when syncing hundreds of thousands of loose `.datZ` controller export files over removable storage, we implemented monthly uncompressed container archives (`raw_YYYY_MM.zip`).

### Key Modules Implemented & Modified

1. **`src/atspm/data/raw_archive.py` (New):**
   - Implemented `parse_datz_month` to extract `(year, month)` from `.datZ` controller export filenames.
   - Implemented `list_monthly_candidates` to group loose files in `raw_data/` by month while protecting the active/current calendar month by default.
   - Implemented `pack_monthly_archive` using uncompressed `zipfile.ZIP_STORED` and `allowZip64=True` with single-pass batch append mode. Verifies archive integrity and member file sizes prior to pruning loose source files.
   - Implemented `verify_archive` checking CRC checksums via `testzip()` and expected member sizes.
   - Implemented `pack_intersection_raw` orchestrator across eligible months with `--dry-run` support.

2. **`src/atspm/data/ingestion.py` (Updated):**
   - Defined `DatzSource(name, timestamp, archive_path)` abstraction.
   - Updated file scanning (`_scan_all_candidates`, `_scan_files_append`, and `_scan_files_gap_fill`) to scan both loose `.datZ` files and member files in `raw_*.zip` archives without extracting to disk.
   - Deduplicated candidate files with loose files taking precedence over archives.
   - Streamed member bytes directly from open `ZipFile` handles during batch ingestion (`_process_batches` and `_process_file`), reusing handles across consecutive members of the same archive.
   - Preserved full compatibility with Fast Append, Gap Fill, and Rebuild (`--rebuild`) workflows.

3. **`src/atspm/data/__init__.py` (Updated):**
   - Exported `DatzSource`, `PackResult`, `parse_datz_month`, `list_monthly_candidates`, `verify_archive`, `pack_monthly_archive`, and `pack_intersection_raw`.

4. **`src/atspm/cli.py` (Updated):**
   - Added `atspm pack-raw` subcommand supporting `--target`, `--targetid`, `--all`, `--include-current`, `--keep-loose`, `--dry-run`, and `--verbose`.
   - Added optional `--pack` flag to `atspm sync push` to automatically package closed historical months before copying to the archive drive.

5. **`tests/data/test_raw_archive.py` & `tests/data/test_ingestion_archive.py` (New):**
   - 22 unit tests for filename parsing, monthly candidate listing, archive packing, verification, corruption detection, idempotency, and pruning.
   - 8 unit tests for transparent ingestion and rebuilding from zip archives, mixed loose/zipped spans, gap detection, deduplication, and corruption handling.
   - CLI unit tests added in `tests/test_cli_sync.py` for `pack-raw` and `sync push --pack`.

---

## 2. Test Verification

Executed full test suite:
```bash
PYTHONPATH=src ./.venv/bin/pytest tests/data/test_raw_archive.py tests/data/test_ingestion_archive.py tests/data/test_ingestion*.py tests/data/test_sync.py tests/test_cli_sync.py -q
```
**Results:** `92 passed, 28 warnings in 21.52s` (100% passing).
