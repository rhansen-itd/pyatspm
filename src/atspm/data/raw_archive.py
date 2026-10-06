"""
Monthly .datZ Zip Archive Management (Imperative Shell)

Groups loose .datZ controller export files by calendar month into uncompressed
(ZIP_STORED, allowZip64=True) container archives named ``raw_YYYY_MM.zip``.
Verifies archive integrity before optionally pruning loose source files.

Package Location: src/atspm/data/raw_archive.py
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union


def _noop(_msg: str) -> None:
    """Default no-op logger."""
    pass


@dataclass
class PackResult:
    """Outcome of packing a single calendar month's .datZ files."""

    year: int
    month: int
    archive_path: Path
    files_packed: int
    bytes_packed: int
    files_removed: int
    ok: bool
    detail: str = ""


# Standard controller export pattern: *_YYYY_MM_DD_HHMM.datZ
_DATZ_DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{4})")


def parse_datz_month(filename: Union[str, Path]) -> Optional[Tuple[int, int]]:
    """Extract (year, month) from a .datZ filename.

    Args:
        filename: Filename string or Path (e.g. 'ECON_10.0.0.1_2026_06_20_0400.datZ').

    Returns:
        (year, month) tuple, or None if the pattern does not match or date is invalid.
    """
    name = filename.name if isinstance(filename, Path) else str(filename)
    match = _DATZ_DATE_PATTERN.search(name)
    if not match:
        return None
    try:
        year = int(match.group(1))
        month = int(match.group(2))
        if 1 <= month <= 12 and year > 0:
            return (year, month)
    except (ValueError, TypeError):
        pass
    return None


def list_monthly_candidates(
    raw_dir: Path,
    include_current: bool = False,
    current_month: Optional[Tuple[int, int]] = None,
) -> Dict[Tuple[int, int], List[Path]]:
    """Group loose .datZ files in raw_dir by calendar month.

    Args:
        raw_dir: Directory containing .datZ files.
        include_current: If True, include candidates from the active/current month.
                         If False (default), exclude them.
        current_month: Optional (year, month) tuple override. Defaults to current system date.

    Returns:
        Dict mapping (year, month) to sorted list of loose .datZ Path objects.
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        return {}

    if current_month is not None:
        curr_ym = current_month
    else:
        now = datetime.now()
        curr_ym = (now.year, now.month)

    candidates: Dict[Tuple[int, int], List[Path]] = {}
    for fp in raw_path.glob("*.datZ"):
        ym = parse_datz_month(fp.name)
        if ym is None:
            continue
        if not include_current and ym == curr_ym:
            continue
        candidates.setdefault(ym, []).append(fp)

    return {
        ym: sorted(paths, key=lambda p: p.name)
        for ym, paths in sorted(candidates.items())
    }


def verify_archive(
    archive_path: Path,
    expected_files: Optional[Dict[str, int]] = None,
) -> Tuple[bool, str]:
    """Verify zip archive integrity and member sizes.

    Args:
        archive_path: Path to the .zip file.
        expected_files: Optional mapping of member name -> expected byte size.

    Returns:
        (True, "ok") or (False, error_detail).
    """
    path = Path(archive_path)
    if not path.is_file():
        return False, f"Archive not found: {path}"

    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                return False, f"Corrupted zip member (CRC mismatch): {bad_member}"

            if expected_files:
                member_sizes = {info.filename: info.file_size for info in zf.infolist()}
                for name, exp_size in expected_files.items():
                    if name not in member_sizes:
                        return False, f"Missing member in archive: {name}"
                    actual_size = member_sizes[name]
                    if actual_size != exp_size:
                        return (
                            False,
                            f"Size mismatch for {name}: expected {exp_size}, got {actual_size}",
                        )
    except zipfile.BadZipFile as exc:
        return False, f"Bad zip file: {exc}"
    except Exception as exc:
        return False, f"Archive read error: {exc}"

    return True, "ok"


def pack_monthly_archive(
    raw_dir: Path,
    year: int,
    month: int,
    remove_loose: bool = True,
    dry_run: bool = False,
    log: Callable[[str], None] = _noop,
) -> PackResult:
    """Pack loose .datZ files for (year, month) into raw_YYYY_MM.zip.

    Args:
        raw_dir: Directory containing .datZ files.
        year: Calendar year.
        month: Calendar month (1-12).
        remove_loose: Delete loose source files after successful verification.
        dry_run: Report what would be packed and removed without modifying disk.
        log: Progress logging callback.

    Returns:
        PackResult summary.
    """
    raw_path = Path(raw_dir)
    archive_path = raw_path / f"raw_{year}_{month:02d}.zip"

    candidate_files = sorted(
        [
            p
            for p in raw_path.glob("*.datZ")
            if parse_datz_month(p.name) == (year, month)
        ],
        key=lambda p: p.name,
    )

    if not candidate_files:
        log(f"No loose files to pack for {year}_{month:02d}")
        return PackResult(
            year=year,
            month=month,
            archive_path=archive_path,
            files_packed=0,
            bytes_packed=0,
            files_removed=0,
            ok=True,
            detail="no files to pack",
        )

    # Inspect existing archive if present
    existing_members: Dict[str, int] = {}
    if archive_path.exists():
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    return PackResult(
                        year=year,
                        month=month,
                        archive_path=archive_path,
                        files_packed=0,
                        bytes_packed=0,
                        files_removed=0,
                        ok=False,
                        detail=f"Existing archive is corrupt (bad member: {bad_member})",
                    )
                existing_members = {
                    info.filename: info.file_size for info in zf.infolist()
                }
        except Exception as exc:
            return PackResult(
                year=year,
                month=month,
                archive_path=archive_path,
                files_packed=0,
                bytes_packed=0,
                files_removed=0,
                ok=False,
                detail=f"Existing archive cannot be opened: {exc}",
            )

    files_to_write: List[Tuple[Path, int]] = []
    files_already_packed: List[Path] = []
    for cf in candidate_files:
        try:
            cf_size = cf.stat().st_size
        except OSError as exc:
            return PackResult(
                year=year,
                month=month,
                archive_path=archive_path,
                files_packed=0,
                bytes_packed=0,
                files_removed=0,
                ok=False,
                detail=f"Cannot stat loose file {cf.name}: {exc}",
            )

        if cf.name in existing_members:
            if existing_members[cf.name] == cf_size:
                files_already_packed.append(cf)
            else:
                return PackResult(
                    year=year,
                    month=month,
                    archive_path=archive_path,
                    files_packed=0,
                    bytes_packed=0,
                    files_removed=0,
                    ok=False,
                    detail=(
                        f"Size mismatch for existing archive member {cf.name}: "
                        f"archive has {existing_members[cf.name]}, loose file has {cf_size}"
                    ),
                )
        else:
            files_to_write.append((cf, cf_size))

    bytes_to_write = sum(sz for _, sz in files_to_write)

    if dry_run:
        log(
            f"[dry-run] Would pack {len(files_to_write)} files ({bytes_to_write} bytes) "
            f"into {archive_path.name}"
        )
        return PackResult(
            year=year,
            month=month,
            archive_path=archive_path,
            files_packed=len(files_to_write),
            bytes_packed=bytes_to_write,
            files_removed=len(candidate_files) if remove_loose else 0,
            ok=True,
            detail="dry run",
        )

    # Batch append to zip archive using ZIP_STORED and allowZip64=True
    if files_to_write:
        raw_path.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(
                archive_path,
                mode="a",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as zf:
                for cf, _ in files_to_write:
                    zf.write(cf, arcname=cf.name)
                    log(f"Packed {cf.name} -> {archive_path.name}")
        except Exception as exc:
            return PackResult(
                year=year,
                month=month,
                archive_path=archive_path,
                files_packed=0,
                bytes_packed=0,
                files_removed=0,
                ok=False,
                detail=f"Failed writing to archive: {exc}",
            )

    # Verify integrity of archive and all candidate files
    expected_files = {cf.name: cf.stat().st_size for cf in candidate_files}
    verified, v_detail = verify_archive(archive_path, expected_files=expected_files)
    if not verified:
        return PackResult(
            year=year,
            month=month,
            archive_path=archive_path,
            files_packed=len(files_to_write),
            bytes_packed=bytes_to_write,
            files_removed=0,
            ok=False,
            detail=f"Verification failed: {v_detail}",
        )

    # Prune loose files if requested
    files_removed = 0
    if remove_loose:
        for cf in candidate_files:
            try:
                cf.unlink()
                files_removed += 1
                log(f"Removed loose file {cf.name}")
            except OSError as exc:
                log(f"Warning: could not remove loose file {cf.name}: {exc}")

    return PackResult(
        year=year,
        month=month,
        archive_path=archive_path,
        files_packed=len(files_to_write),
        bytes_packed=bytes_to_write,
        files_removed=files_removed,
        ok=True,
        detail="ok",
    )


def pack_intersection_raw(
    raw_dir: Path,
    include_current: bool = False,
    remove_loose: bool = True,
    dry_run: bool = False,
    log: Callable[[str], None] = _noop,
) -> List[PackResult]:
    """High-level orchestrator for an intersection.

    Packs all eligible monthly candidates into raw_YYYY_MM.zip files.

    Args:
        raw_dir: Intersection raw_data directory.
        include_current: If True, pack current month as well.
        remove_loose: If True, unlink loose files after verification.
        dry_run: If True, simulate without modifying disk.
        log: Progress logging callback.

    Returns:
        List of PackResult objects for each processed month.
    """
    candidates = list_monthly_candidates(
        raw_dir=raw_dir,
        include_current=include_current,
    )
    results: List[PackResult] = []
    for (year, month), _ in sorted(candidates.items()):
        res = pack_monthly_archive(
            raw_dir=raw_dir,
            year=year,
            month=month,
            remove_loose=remove_loose,
            dry_run=dry_run,
            log=log,
        )
        results.append(res)
    return results
