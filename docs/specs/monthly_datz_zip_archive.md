# Spec: Monthly .datZ Zip Archive and Sync Workflow

Written for autonomous execution in a dedicated session.
Source: Sync latency bottlenecks when transferring hundreds of thousands of loose `.datZ` files over USB / 9p removable media to an external SSD.

## Background & Problem Statement

Each intersection's `raw_data/` folder accumulates tens of thousands of individual `.datZ` files (e.g. `ECON_10.0.0.1_2026_06_20_0400.datZ`).
Transferring 100,000+ tiny files over USB to an external SSD or removable filesystem causes severe I/O bottlenecks due to per-file filesystem metadata round-trips, directory entry allocation, and bus latency.

**The Solution:**
1. Group `.datZ` files by calendar month (`YYYY_MM`) into single uncompressed ZIP container files: `raw_YYYY_MM.zip` (e.g. `raw_2026_06.zip`).
2. Store them directly in `raw_data/`.
3. Because `.datZ` files are already gzip-compressed by the controller, the zip container must use **store-only mode (`zipfile.ZIP_STORED`)** and **`allowZip64=True`**. This eliminates redundant compression CPU overhead while allowing multi-gigabyte archives containing >65,535 files.
4. Update `IngestionEngine` ([`src/atspm/data/ingestion.py`](file:///home/hansrkid/pyatspm/src/atspm/data/ingestion.py)) so that `atspm process` and `atspm process --rebuild` transparently read `.datZ` files from both loose files and `raw_*.zip` archives without needing to unpack them to disk.
5. Provide a CLI command `atspm pack-raw` to archive closed months and verify them before pruning loose files.
6. Seamlessly leverage the existing `atspm sync` engine: `copy_verify_dir` will transfer `raw_YYYY_MM.zip` as single atomic files (seconds per month instead of hours).

---

## 0. Rules for the Implementation Run

### Files you must NOT modify
- `src/atspm/analysis/**` (Functional Core: math, vectorization, Plotly logic)
- `tests/analysis/**`
- Any existing test files not directly related to raw data ingestion or sync

### Files you may create or edit
- **create** `src/atspm/data/raw_archive.py` (archiving, packing, and validation logic)
- **create** `tests/data/test_raw_archive.py` (unit tests for archiving and zip extraction)
- **edit** `src/atspm/data/ingestion.py` (support scanning and streaming bytes from `.zip` members)
- **edit** `src/atspm/data/__init__.py` (export raw archive helpers if needed)
- **edit** `src/atspm/cli.py` (add `pack-raw` subcommand; add optional `--pack` on `sync push`)
- **create** `tests/data/test_ingestion_archive.py` (tests verifying `IngestionEngine` with zipped files)
- **create** `docs/specs/monthly_datz_zip_archive_REPORT.md` (brief completion report)
- **append** to `docs/PENDING_DOC_CHANGES.md` (terse bullets for new CLI command and public exports)

### Acceptance Check
```bash
PYTHONPATH=src pytest tests/data/test_raw_archive.py tests/data/test_ingestion_archive.py tests/data/test_ingestion*.py tests/data/test_sync.py tests/test_cli_sync.py -q
```
All tests must pass cleanly.

---

## 1. Storage & Archive Specification

### Filename Conventions
- Loose files: `*_<YYYY>_<MM>_<DD>_<HHMM>.datZ` (standard controller export format).
- Monthly zip archive: `raw_<YYYY>_<MM>.zip` (e.g. `raw_2026_01.zip`, `raw_2026_10.zip`).
- Location: `intersections/<target>/raw_data/raw_<YYYY>_<MM>.zip`.

### Zip Format Requirements
- **Compression**: `zipfile.ZIP_STORED` (do not use `ZIP_DEFLATED`; `.datZ` files are already gzip compressed).
- **ZIP64**: Always enable `allowZip64=True` to support archives with >65,535 files and >4 GB size.
- **Member paths**: Flat basename inside the zip (e.g. `ECON_10.0.0.1_2026_06_20_0400.datZ`, not nested folders).
- **Batching**: Always batch additions inside a single `with zipfile.ZipFile(..., mode="a")` context to avoid rewriting the central directory repeatedly.

---

## 2. Core Archiving Module (`src/atspm/data/raw_archive.py`)

Create `src/atspm/data/raw_archive.py` containing:

```python
@dataclass
class PackResult:
    year: int
    month: int
    archive_path: Path
    files_packed: int
    bytes_packed: int
    files_removed: int
    ok: bool
    detail: str = ""
```

### Key Functions

1. `parse_datz_month(filename: str) -> Optional[Tuple[int, int]]`
   - Uses `re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{4})", filename)` to extract `(int(year), int(month))`.
   - Returns `None` if filename does not match.

2. `list_monthly_candidates(raw_dir: Path, include_current: bool = False, current_month: Optional[Tuple[int, int]] = None) -> Dict[Tuple[int, int], List[Path]]`
   - Scans loose `*.datZ` in `raw_dir`.
   - Groups by `(year, month)`.
   - If `include_current is False`, excludes files belonging to the current calendar month (determined from system clock or `current_month` override).

3. `pack_monthly_archive(raw_dir: Path, year: int, month: int, remove_loose: bool = True, log: Callable[[str], None] = _noop) -> PackResult`
   - Target archive: `raw_dir / f"raw_{year}_{month:02d}.zip"`.
   - Inspects existing archive members if the zip exists (avoids re-adding identical files).
   - Writes candidate `.datZ` files to the archive using `ZIP_STORED` and `allowZip64=True`.
   - Verifies the archive: checks that every expected file is present in the zip with matching byte size.
   - If verified and `remove_loose is True`: unlinks the loose source files from `raw_dir`.
   - If verification fails: leaves loose source files untouched and returns `ok=False`.

4. `pack_intersection_raw(raw_dir: Path, include_current: bool = False, remove_loose: bool = True, log: Callable[[str], None] = _noop) -> List[PackResult]`
   - High-level orchestrator for an intersection. Runs `pack_monthly_archive` across all eligible months.

---

## 3. Ingestion Engine Integration (`src/atspm/data/ingestion.py`)

Update `IngestionEngine` so it can scan and stream bytes from both loose `.datZ` and `raw_*.zip` files:

### Abstraction: `DatzSource`
Instead of passing raw `Path` objects through the scanning pipeline, represent candidate items as:
```python
@dataclass(frozen=True)
class DatzSource:
    name: str                 # filename, e.g. "ECON_10.0.0.1_2026_06_20_0400.datZ"
    timestamp: float          # UTC epoch derived from filename
    archive_path: Optional[Path] = None  # None if loose file; Path to zip if in archive
```

### Byte Reading:
- If `source.archive_path is None`: reads `(raw_data_dir / source.name).read_bytes()`.
- If `source.archive_path is not None`: reads bytes directly from the zip without extracting to disk:
  ```python
  with zipfile.ZipFile(source.archive_path, "r") as zf:
      data = zf.read(source.name)
  ```
  *(Tip: in batch processing `_process_file_batch`, keep the `ZipFile` open for files sharing the same archive to prevent re-opening).*

### Scanning (`_scan_files_append` and `_scan_files_gap_fill`):
- Find all loose files: `self.raw_data_dir.glob("*.datZ")`.
- Find all zip archives: `self.raw_data_dir.glob("raw_*.zip")`.
- For each zip archive, inspect `zf.infolist()` to yield member names and timestamps (no decompression needed!).
- Deduplicate: if a file is present both loose and in a zip, loose takes precedence.
- Sort all candidate `DatzSource` items chronologically by `timestamp`.
- Apply `min_timestamp` or `_is_covered` filters exactly as before.

---

## 4. CLI Subcommand (`atspm pack-raw`)

Add `pack-raw` to `src/atspm/cli.py`:

```bash
atspm pack-raw (--target FOLDER | --targetid ID | --all) [--include-current] [--keep-loose] [--dry-run] [--verbose]
```

- `--target` / `--targetid` / `--all`: Standard target selection group.
- `--include-current`: By default, the active/current calendar month is kept as loose files for live incoming `atspm retrieve` pulls. This flag packs the active month as well.
- `--keep-loose`: Do not delete loose files after verifying they are packed into the zip.
- `--dry-run`: Reports what would be packed/deleted without making disk changes.
- `--verbose`: Prints per-file progress.

---

## 5. Sync Workflow Interaction

Because `raw_YYYY_MM.zip` files reside inside `intersections/<target>/raw_data/`:
1. `atspm sync status`:
   - Inspects `raw_data/` and reports `raw_YYYY_MM.zip` alongside any remaining loose files.
2. `atspm sync push`:
   - Copies `raw_YYYY_MM.zip` as single files using atomic `.synctmp` copy + verify.
   - Skips files that already exist in the archive with matching size / SHA-256.
   - Result: 100,000 files previously taking hours to sync will now sync as 12–24 zip files in seconds.
3. `atspm sync pull`:
   - If `--components raw` or `--components all` is requested, pulls the `.zip` files directly.
