"""Project-agnostic data-sync engine (Imperative Shell).

This module moves bulky data units between a *local* working drive and an
*archive* drive (e.g. an external SSD), verifying every copy before it is
trusted and — only on explicit request — deleting the local source to free
space.  It exists because SQLite/DuckDB databases cannot be queried reliably
over the 9p ChromeOS removable-media share: heavy work must happen on local
disk, so data is checked out to local, worked on, then pushed back.

Design notes
------------
* **Portable on purpose.**  Nothing here imports from the rest of ``atspm``;
  it depends only on the standard library and an optional ``log`` callback.
  A sibling project (e.g. Inrix) can reuse this engine unchanged and supply
  its own adapter that enumerates :class:`SyncItem` objects — the only
  project-specific part.
* **Verify before trust, verify before delete.**  Copies land in a temporary
  sidecar file and are renamed into place only after the size (and, by
  default, the SHA-256 checksum) matches.  A local source is never deleted on
  ``release`` unless its archive copy verified by checksum.
* **Additive, never clobbering the archive.**  A directory sync copies files
  that are missing or differ; it never deletes files already in the
  destination.  The one deletion this module performs is of the *local source*
  during a ``release`` push, and only after verification.

This is shell code (CLAUDE.md §4): it performs file I/O and manages paths.  It
holds no analysis logic and no DB connections.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# A no-op logger keeps the engine silent unless the caller wires one in.
Logger = Callable[[str], None]


def _noop(_msg: str) -> None:  # pragma: no cover - trivial
    pass


_CHUNK = 1 << 20  # 1 MiB streaming reads keep multi-GB hashes memory-flat.


# ===========================================================================
# Data model
# ===========================================================================

@dataclass(frozen=True)
class SyncItem:
    """One logical unit of data to move between local and archive.

    Attributes:
        name:    Human-readable unit name, unique within a sync set
                 (e.g. ``'db'`` or ``'config:metadata.json'``).
        group:   Coarse category used by ``--components`` filtering
                 (e.g. ``'db'``, ``'raw'``, ``'outputs'``, ``'other'``).
        local:   Absolute path on the local working drive.
        archive: Absolute path on the archive drive (same basename semantics).
        kind:    ``'file'`` or ``'dir'``.
        companions: For a file unit, sibling suffixes that travel with it
                 (e.g. ``('-wal', '-shm')`` for a SQLite DB).  Each existing
                 ``<path><suffix>`` is copied, verified and (on release)
                 deleted alongside the primary file.
    """

    name: str
    group: str
    local: Path
    archive: Path
    kind: str
    companions: Tuple[str, ...] = ()


@dataclass
class SyncResult:
    """Outcome of syncing one :class:`SyncItem`.

    Attributes:
        item:    The item this result describes.
        action:  One of ``'copied'``, ``'copied+released'``, ``'in-sync'``,
                 ``'missing-source'``, ``'verify-failed'``, ``'dry-run'``.
        ok:      ``True`` when the operation succeeded (or was a no-op because
                 source and destination already matched).
        n_bytes: Bytes copied (0 when nothing was transferred).
        detail:  Human-readable note for the CLI summary / error surfacing.
    """

    item: SyncItem
    action: str
    ok: bool
    n_bytes: int = 0
    detail: str = ""


@dataclass
class ItemState:
    """A snapshot of where an item currently lives, for ``status``.

    Sizes are byte totals; ``state`` is a compact verdict computed from sizes
    only (fast) unless the caller requested checksum comparison.
    """

    item: SyncItem
    local_exists: bool
    archive_exists: bool
    local_bytes: int
    archive_bytes: int
    state: str  # local-only | archive-only | in-sync | differ | absent
    extra: Dict[str, int] = field(default_factory=dict)


# ===========================================================================
# Hashing / sizing helpers
# ===========================================================================

def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a single file, streamed in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest string.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _iter_regular_files(root: Path):
    """Yield ``(relative_path, absolute_path)`` for every regular file under
    ``root`` (recursively), skipping symlinks and special files.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            ap = Path(dirpath) / fn
            if ap.is_symlink() or not ap.is_file():
                continue
            yield ap.relative_to(root), ap


def dir_total_bytes(root: Path) -> int:
    """Return the total size in bytes of all regular files under ``root``."""
    return sum(_file_size(ap) for _rel, ap in _iter_regular_files(root))


def _companion_paths(primary: Path, companions: Tuple[str, ...]) -> List[Path]:
    """Return the *existing* companion sidecar paths for a primary file path."""
    out = []
    for suf in companions:
        p = Path(str(primary) + suf)
        if p.is_file():
            out.append(p)
    return out


# ===========================================================================
# Copy + verify primitives
# ===========================================================================

def copy_verify_file(
    src: Path,
    dst: Path,
    *,
    checksum: bool = True,
) -> Tuple[bool, int, str]:
    """Copy one file to ``dst`` and verify it before committing.

    The copy is written to a temporary sidecar in the destination directory
    and atomically renamed into place only after verification passes, so an
    interrupted copy never leaves a corrupt file at ``dst``.

    Args:
        src:      Source file (must exist and be a regular file).
        dst:      Destination path.
        checksum: When ``True`` verify a SHA-256 match; otherwise compare
                  byte size only (faster, weaker).

    Returns:
        ``(ok, n_bytes, detail)``.
    """
    if not src.is_file():
        return False, 0, f"source is not a regular file: {src}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".synctmp")
    try:
        if tmp.exists():
            tmp.unlink()
        shutil.copyfile(src, tmp)
        shutil.copystat(src, tmp, follow_symlinks=True)

        src_size = _file_size(src)
        tmp_size = _file_size(tmp)
        if src_size != tmp_size:
            tmp.unlink(missing_ok=True)
            return False, 0, f"size mismatch ({src_size} vs {tmp_size})"

        if checksum:
            if sha256_file(src) != sha256_file(tmp):
                tmp.unlink(missing_ok=True)
                return False, 0, "checksum mismatch"

        os.replace(tmp, dst)  # atomic within a filesystem
        return True, src_size, "ok"
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False, 0, f"copy error: {exc}"


def copy_verify_dir(
    src: Path,
    dst: Path,
    *,
    checksum: bool = True,
    log: Logger = _noop,
) -> Tuple[bool, int, int, str]:
    """Recursively copy ``src`` into ``dst`` (additive), verifying each file.

    Files already present at the destination with a matching size (and, when
    ``checksum`` is set, a matching SHA-256) are skipped.  Destination files
    with no source counterpart are left untouched — the archive is never
    pruned by a sync.

    Args:
        src:      Source directory.
        dst:      Destination directory (created if absent).
        checksum: Verify SHA-256 per file when ``True``.
        log:      Optional progress callback.

    Returns:
        ``(ok, n_files_copied, n_bytes_copied, detail)``.  ``ok`` is ``False``
        if any file failed to copy/verify.
    """
    if not src.is_dir():
        return False, 0, 0, f"source is not a directory: {src}"

    dst.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_bytes = 0
    failures: List[str] = []
    skipped_special = 0

    # Detect non-regular files so release-deletion can be refused if any exist.
    for dirpath, _dirnames, filenames in os.walk(src):
        for fn in filenames:
            ap = Path(dirpath) / fn
            if ap.is_symlink() or not ap.is_file():
                skipped_special += 1

    for rel, abs_src in _iter_regular_files(src):
        abs_dst = dst / rel
        if abs_dst.is_file() and _file_size(abs_dst) == _file_size(abs_src):
            if not checksum or sha256_file(abs_dst) == sha256_file(abs_src):
                continue  # already present and identical
        ok, nb, detail = copy_verify_file(abs_src, abs_dst, checksum=checksum)
        if ok:
            n_files += 1
            n_bytes += nb
            log(f"      + {rel} ({nb:,} B)")
        else:
            failures.append(f"{rel}: {detail}")

    if failures:
        return False, n_files, n_bytes, "; ".join(failures[:5])
    note = "ok"
    if skipped_special:
        note = f"ok ({skipped_special} non-regular file(s) skipped)"
    return True, n_files, n_bytes, note


# ===========================================================================
# State inspection (for `status`)
# ===========================================================================

def _dir_in_sync(a: Path, b: Path, *, checksum: bool) -> bool:
    """Return True if every file under ``a`` exists under ``b`` with matching
    size (and checksum when requested).  Extra files under ``b`` are ignored.
    """
    for rel, ap in _iter_regular_files(a):
        bp = b / rel
        if not bp.is_file() or _file_size(bp) != _file_size(ap):
            return False
        if checksum and sha256_file(bp) != sha256_file(ap):
            return False
    return True


def item_state(item: SyncItem, *, checksum: bool = False) -> ItemState:
    """Compute where ``item`` currently lives and whether the copies agree.

    By default this uses sizes only (fast); pass ``checksum=True`` for a
    byte-exact verdict at the cost of hashing both sides.

    Args:
        item:     The item to inspect.
        checksum: Compare SHA-256 as well as size.

    Returns:
        An :class:`ItemState`.
    """
    if item.kind == "file":
        local_exists = item.local.is_file()
        archive_exists = item.archive.is_file()
        local_bytes = _file_size(item.local) if local_exists else 0
        archive_bytes = _file_size(item.archive) if archive_exists else 0
        if local_exists and archive_exists:
            same = local_bytes == archive_bytes and (
                not checksum or sha256_file(item.local) == sha256_file(item.archive)
            )
            state = "in-sync" if same else "differ"
        elif local_exists:
            state = "local-only"
        elif archive_exists:
            state = "archive-only"
        else:
            state = "absent"
    else:  # dir
        local_exists = item.local.is_dir()
        archive_exists = item.archive.is_dir()
        local_bytes = dir_total_bytes(item.local) if local_exists else 0
        archive_bytes = dir_total_bytes(item.archive) if archive_exists else 0
        if local_exists and archive_exists:
            # Symmetric check so a drifted copy on either side reads as differ.
            same = _dir_in_sync(item.local, item.archive, checksum=checksum) and \
                _dir_in_sync(item.archive, item.local, checksum=checksum)
            state = "in-sync" if same else "differ"
        elif local_exists:
            state = "local-only"
        elif archive_exists:
            state = "archive-only"
        else:
            state = "absent"

    return ItemState(
        item=item,
        local_exists=local_exists,
        archive_exists=archive_exists,
        local_bytes=local_bytes,
        archive_bytes=archive_bytes,
        state=state,
    )


# ===========================================================================
# The one public operation
# ===========================================================================

def sync_item(
    item: SyncItem,
    *,
    direction: str,
    release: bool = False,
    checksum: bool = True,
    dry_run: bool = False,
    log: Logger = _noop,
) -> SyncResult:
    """Copy one item in ``direction`` and (optionally) free the local source.

    Args:
        item:      The unit to sync.
        direction: ``'push'`` (local → archive) or ``'pull'`` (archive →
                   local).
        release:   Only honoured for ``push``.  When ``True`` the local source
                   is deleted after its archive copy verifies — this is how
                   local space is reclaimed.  Ignored for ``pull``.
        checksum:  Verify SHA-256 (not just size).  **Required** for a
                   ``release`` push; the caller must not pass ``checksum=False``
                   with ``release=True`` (a local file is never deleted on an
                   unverified copy — this function enforces it).
        dry_run:   Report what would happen without copying or deleting.
        log:       Optional progress callback.

    Returns:
        A :class:`SyncResult`.
    """
    if direction not in ("push", "pull"):
        raise ValueError(f"direction must be 'push' or 'pull', got {direction!r}")

    src = item.local if direction == "push" else item.archive
    dst = item.archive if direction == "push" else item.local
    do_release = release and direction == "push"

    # Safety invariant: never delete a local source behind an unverified copy.
    if do_release and not checksum:
        return SyncResult(
            item, "verify-failed", ok=False,
            detail="refusing to release without checksum verification",
        )

    exists = src.is_file() if item.kind == "file" else src.is_dir()
    if not exists:
        return SyncResult(item, "missing-source", ok=False,
                           detail=f"source absent: {src}")

    # Already identical?  A push/pull of matching data is a no-op (but a
    # release still needs to delete the now-redundant local copy).
    state = item_state(item, checksum=False)
    already = state.state == "in-sync"

    if dry_run:
        what = "release" if do_release else direction
        return SyncResult(item, "dry-run", ok=True,
                          detail=f"would {what} {item.name}"
                                 + (" (already in sync)" if already else ""))

    n_bytes = 0
    if not already:
        if item.kind == "file":
            ok, n_bytes, detail = copy_verify_file(src, dst, checksum=checksum)
            if ok:
                for comp in _companion_paths(src, item.companions):
                    comp_dst = Path(str(dst) + comp.name[len(src.name):])
                    c_ok, c_nb, c_detail = copy_verify_file(
                        comp, comp_dst, checksum=checksum)
                    if not c_ok:
                        ok, detail = False, f"companion {comp.name}: {c_detail}"
                        break
                    n_bytes += c_nb
        else:
            ok, _nf, n_bytes, detail = copy_verify_dir(
                src, dst, checksum=checksum, log=log)
        if not ok:
            return SyncResult(item, "verify-failed", ok=False,
                              n_bytes=n_bytes, detail=detail)

    # Copy verified (or data already matched).  Reclaim local space if asked.
    if do_release:
        try:
            _delete_local(item)
        except OSError as exc:
            return SyncResult(item, "verify-failed", ok=False, n_bytes=n_bytes,
                              detail=f"copied ok but local delete failed: {exc}")
        return SyncResult(item, "copied+released", ok=True, n_bytes=n_bytes,
                          detail="verified, local freed"
                                 if not already else "already archived, local freed")

    action = "in-sync" if already else "copied"
    return SyncResult(item, action, ok=True, n_bytes=n_bytes,
                      detail="already in sync" if already else "verified")


def _delete_local(item: SyncItem) -> None:
    """Delete an item's local copy (file + companions, or directory tree)."""
    if item.kind == "file":
        item.local.unlink(missing_ok=True)
        for comp in _companion_paths(item.local, item.companions):
            comp.unlink(missing_ok=True)
    else:
        shutil.rmtree(item.local, ignore_errors=False)


def human_bytes(n: int) -> str:
    """Format a byte count as a short human-readable string (e.g. ``'5.8 GB'``)."""
    step = 1024.0
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < step or unit == "TB":
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= step
    return f"{val:.1f} TB"  # pragma: no cover
