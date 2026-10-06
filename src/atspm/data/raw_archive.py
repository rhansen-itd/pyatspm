"""
Monthly .datZ Zip Archive Management (Imperative Shell)

Groups loose .datZ controller export files by calendar month into uncompressed
(ZIP_STORED, allowZip64=True) container archives named ``raw_YYYY_MM.zip``.
Verifies archive integrity before optionally pruning loose source files.

Package Location: src/atspm/data/raw_archive.py
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
import zlib
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


_CHUNK = 1 << 20  # 1 MiB streaming reads for checksums


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


def _crc32_file(path: Path) -> int:
    """Return the CRC-32 of a file, streamed in chunks (zip member checksum)."""
    crc = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def verify_archive(
    archive_path: Path,
    expected_files: Optional[Dict[str, int]] = None,
    expected_crcs: Optional[Dict[str, int]] = None,
) -> Tuple[bool, str]:
    """Verify zip archive integrity, member sizes and (optionally) checksums.

    Args:
        archive_path: Path to the .zip file.
        expected_files: Optional mapping of member name -> expected byte size.
        expected_crcs: Optional mapping of member name -> expected CRC-32 of the
            source bytes.  Lets the caller prove the archived bytes are the
            loose file's bytes, not merely a same-sized file of the same name.

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

            infos = {info.filename: info for info in zf.infolist()}
            for name, exp_size in (expected_files or {}).items():
                if name not in infos:
                    return False, f"Missing member in archive: {name}"
                actual_size = infos[name].file_size
                if actual_size != exp_size:
                    return (
                        False,
                        f"Size mismatch for {name}: expected {exp_size}, got {actual_size}",
                    )
            for name, exp_crc in (expected_crcs or {}).items():
                if name not in infos:
                    return False, f"Missing member in archive: {name}"
                if infos[name].CRC != exp_crc:
                    return False, f"Checksum mismatch for {name}"
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
    candidate_files: Optional[List[Path]] = None,
) -> PackResult:
    """Pack loose .datZ files for (year, month) into raw_YYYY_MM.zip.

    The archive is never modified in place: new members are appended to a
    temporary copy (``raw_YYYY_MM.zip.packtmp``) that replaces the archive
    atomically only after it verifies.  An interrupted run therefore can never
    corrupt an existing archive whose loose files were already pruned.  Loose
    files are deleted only once every one of them is proven present in the
    archive by name, size and CRC-32.

    Args:
        raw_dir: Directory containing .datZ files.
        year: Calendar year.
        month: Calendar month (1-12).
        remove_loose: Delete loose source files after successful verification.
        dry_run: Report what would be packed and removed without modifying disk.
        log: Progress logging callback.
        candidate_files: Loose files for this month, if the caller already
            scanned ``raw_dir``; otherwise the directory is scanned here.

    Returns:
        PackResult summary.
    """
    raw_path = Path(raw_dir)
    archive_path = raw_path / f"raw_{year}_{month:02d}.zip"

    def _fail(detail: str, packed: int = 0, n_bytes: int = 0) -> PackResult:
        return PackResult(
            year=year,
            month=month,
            archive_path=archive_path,
            files_packed=packed,
            bytes_packed=n_bytes,
            files_removed=0,
            ok=False,
            detail=detail,
        )

    if candidate_files is None:
        candidate_files = [
            p for p in raw_path.glob("*.datZ") if parse_datz_month(p.name) == (year, month)
        ]
    candidate_files = sorted(candidate_files, key=lambda p: p.name)

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
    existing_members: Dict[str, zipfile.ZipInfo] = {}
    if archive_path.exists():
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    return _fail(f"Existing archive is corrupt (bad member: {bad_member})")
                existing_members = {info.filename: info for info in zf.infolist()}
        except Exception as exc:
            return _fail(f"Existing archive cannot be opened: {exc}")

    expected_sizes: Dict[str, int] = {}
    expected_crcs: Dict[str, int] = {}
    files_to_write: List[Tuple[Path, int]] = []
    for cf in candidate_files:
        try:
            cf_size = cf.stat().st_size
            cf_crc = _crc32_file(cf)
        except OSError as exc:
            return _fail(f"Cannot read loose file {cf.name}: {exc}")
        expected_sizes[cf.name] = cf_size
        expected_crcs[cf.name] = cf_crc

        member = existing_members.get(cf.name)
        if member is None:
            files_to_write.append((cf, cf_size))
        elif member.file_size != cf_size or member.CRC != cf_crc:
            return _fail(
                f"Existing archive member {cf.name} differs from the loose file "
                f"(archive {member.file_size} B, loose {cf_size} B)"
            )

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

    if files_to_write:
        # Build in a temporary copy, verify, then atomically replace.
        tmp_path = archive_path.with_name(archive_path.name + ".packtmp")
        try:
            tmp_path.unlink(missing_ok=True)
            if archive_path.exists():
                shutil.copyfile(archive_path, tmp_path)
            with zipfile.ZipFile(
                tmp_path,
                mode="a",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as zf:
                for cf, _ in files_to_write:
                    zf.write(cf, arcname=cf.name)
                    log(f"Packed {cf.name} -> {archive_path.name}")

            verified, v_detail = verify_archive(
                tmp_path, expected_files=expected_sizes, expected_crcs=expected_crcs
            )
            if not verified:
                tmp_path.unlink(missing_ok=True)
                return _fail(
                    f"Verification failed: {v_detail}", len(files_to_write), bytes_to_write
                )
            os.replace(tmp_path, archive_path)
        except Exception as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return _fail(f"Failed writing to archive: {exc}")
    else:
        verified, v_detail = verify_archive(
            archive_path, expected_files=expected_sizes, expected_crcs=expected_crcs
        )
        if not verified:
            return _fail(f"Verification failed: {v_detail}")

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
    for (year, month), files in candidates.items():
        res = pack_monthly_archive(
            raw_dir=raw_dir,
            year=year,
            month=month,
            remove_loose=remove_loose,
            dry_run=dry_run,
            log=log,
            candidate_files=files,
        )
        results.append(res)
    return results
