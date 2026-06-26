"""
ATSPM Device Retrieval Engine (Imperative Shell)

Pulls raw ``.datZ`` files from intersection-side devices (traffic
controllers, secondary detection systems such as EVO radar units) onto
local disk via SCP, ahead of ``run_ingestion``. Per-device pull state
(``last_retrieved``) is tracked in that intersection's ``devices.json`` and
rewritten after each run.

Pull order
==========
A ``secondary`` device (long-term storage, e.g. EVO radar) is always pulled
before the ``controller``, regardless of list order in ``devices.json``.
Controllers retain only a rolling FIFO window (~30 hours); pulling the
controller first advances its bookmark to "now" before the secondary device
has had a chance to report files that fell out of that window.

Host resolution
===============
Each device entry may set an explicit ``host``. Only ``controller`` devices
may omit it -- there's exactly one controller per intersection, so it falls
back unambiguously to ``metadata.json``'s ``controller_ip``. Every other
role (``secondary`` and beyond) must set ``host`` explicitly, since a
secondary device isn't necessarily the detection system and there's no
other metadata.json field that could stand in for it.

Package Location: src/atspm/data/retrieval.py
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import paramiko
from scp import SCPClient

# Pull order by role: secondary (long-term storage) before controller
# (short FIFO window) -- see module docstring.
_ROLE_ORDER = {"secondary": 0, "controller": 1}


def _parse_generic_filename(filename: str) -> Optional[datetime]:
    """Parse a trailing ``..._YYYY_MM_DD_HHMM.datZ`` timestamp out of a filename.

    Matches both Econolite (``ECON_10.70.10.51_2025_11_25_2136.datZ``) and EVO
    export naming, which share this trailing date/time layout.

    Args:
        filename: Remote filename, with or without a leading path.

    Returns:
        Parsed (naive) datetime, or ``None`` if the name doesn't match.
    """
    try:
        base = Path(filename).name
        name_part = re.sub(r"\.datZ?$", "", base, flags=re.IGNORECASE)
        date_str = "_".join(name_part.split("_")[-4:])
        return datetime.strptime(date_str, "%Y_%m_%d_%H%M")
    except (ValueError, IndexError):
        return None


# device_type -> filename parser. Add an entry here for a new vendor whose
# files don't follow the generic trailing-timestamp layout; no engine
# changes required.
_FILENAME_PARSERS = {}


def _get_filename_parser(device_type: Optional[str]):
    """Look up the filename-timestamp parser for a device type, defaulting to the generic one."""
    return _FILENAME_PARSERS.get(device_type, _parse_generic_filename)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse a stored ``last_retrieved`` ISO string, tolerating ``None``/empty/invalid."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class RetrievalEngine:
    """Pulls new ``.datZ`` files for every device configured at one intersection."""

    def __init__(
        self,
        target_dir: Path,
        meta: Dict[str, Any],
        devices: List[Dict[str, Any]],
    ):
        """Initialize the engine.

        Args:
            target_dir: Absolute path to the intersection folder.
            meta:       Parsed ``metadata.json`` dict; supplies
                        ``controller_ip`` / ``detection_ip`` by role.
            devices:    Parsed ``devices.json`` list. Mutated in place —
                        each device's ``last_retrieved`` is advanced as
                        files are pulled.
        """
        self.target_dir = target_dir
        self.raw_dir = target_dir / "raw_data"
        self.meta = meta
        self.devices = devices

    def run(self) -> List[Dict[str, Any]]:
        """Pull every configured device, secondary devices first.

        Returns:
            List of per-device result dicts: ``role``, ``host``,
            ``downloaded`` (file count), ``error`` (``None`` on success).
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            self.devices, key=lambda d: _ROLE_ORDER.get(d.get("role"), 99)
        )
        return [self._pull_device(device) for device in ordered]

    def _resolve_host(self, device: Dict[str, Any]) -> str:
        """Resolve a device's IP: an explicit ``host`` wins; ``controller``
        devices without one fall back to ``metadata.json``'s
        ``controller_ip``.

        There's exactly one controller per intersection, so that mapping is
        unambiguous. Every other role must set ``host`` explicitly — a
        secondary/long-term-storage device isn't necessarily the detection
        system, so there's no metadata.json field that could stand in for it.

        Args:
            device: The device entry (may or may not have a ``host`` key).

        Returns:
            The resolved IP/hostname.

        Raises:
            ValueError: If no explicit ``host`` is set and the device isn't
                a ``controller`` with ``controller_ip`` set in metadata.json.
        """
        host = device.get("host")
        if host:
            return host

        role = device.get("role")
        if role != "controller":
            raise ValueError(
                f"Device with role '{role}' has no 'host' set in devices.json "
                f"(only 'controller' devices can fall back to metadata.json)."
            )
        host = self.meta.get("controller_ip")
        if not host:
            raise ValueError(
                "Device has no 'host', and metadata.json has no 'controller_ip' "
                "set to fall back on."
            )
        return host

    def _pull_device(self, device: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to one device and download any files newer than its bookmark."""
        role = device.get("role")
        host = None
        try:
            host = self._resolve_host(device)
            port = device.get("port", 22)
            user = device["user"]
            password = device["password"]
            remote_folder = device["remote_folder"]
        except (ValueError, KeyError) as exc:
            print(f"  Skipping device (role={role}): {exc}")
            return {"role": role, "host": host, "downloaded": 0, "error": str(exc)}

        last_retrieved = _parse_iso(device.get("last_retrieved"))
        parser = _get_filename_parser(device.get("device_type"))

        print(f"\n[{datetime.now()}] Retrieving from {role}@{host} ({remote_folder})...")

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(host, port=port, username=user, password=password, timeout=15)

            stdin, stdout, stderr = ssh.exec_command(f"ls -p {remote_folder} | grep -v /")
            output = stdout.read().decode().strip()
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                err_text = stderr.read().decode().strip()
                raise RuntimeError(f"Remote 'ls {remote_folder}' failed: {err_text or 'unknown error'}")

            files = output.splitlines() if output else []
            to_fetch = sorted(
                (
                    (ts, fname)
                    for fname in files
                    for ts in (parser(fname),)
                    if ts and (last_retrieved is None or ts > last_retrieved)
                ),
                key=lambda pair: pair[0],
            )

            if not to_fetch:
                print(f"  No new files (last_retrieved={last_retrieved}).")
                return {"role": role, "host": host, "downloaded": 0, "error": None}

            # Another device for this intersection (e.g. an EVO mirroring the
            # controller via its own cron job) may have already landed some
            # of these filenames in raw_data/. Skip re-downloading those --
            # the bookmark still advances past them since we have them.
            print(f"  {len(to_fetch)} candidate file(s) since {last_retrieved}.")
            downloaded = 0
            with SCPClient(ssh.get_transport()) as scp:
                for ts, fname in to_fetch:
                    if not (self.raw_dir / fname).exists():
                        scp.get(f"{remote_folder}/{fname}", str(self.raw_dir / fname))
                        downloaded += 1
                    if last_retrieved is None or ts > last_retrieved:
                        last_retrieved = ts

            device["last_retrieved"] = last_retrieved.isoformat()
            print(
                f"  Downloaded {downloaded} file(s) ({len(to_fetch) - downloaded} already on disk); "
                f"bookmark advanced to {last_retrieved}."
            )
            return {"role": role, "host": host, "downloaded": downloaded, "error": None}

        except paramiko.AuthenticationException:
            msg = f"Authentication failed for {user}@{host}"
            print(f"  Error: {msg}")
            return {"role": role, "host": host, "downloaded": 0, "error": msg}
        except (paramiko.SSHException, OSError, RuntimeError) as exc:
            msg = f"Connection error: {exc}"
            print(f"  Error: {msg}")
            return {"role": role, "host": host, "downloaded": 0, "error": msg}
        finally:
            ssh.close()


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def _load_devices(devices_path: Path) -> List[Dict[str, Any]]:
    with devices_path.open() as fh:
        return json.load(fh)


def _save_devices(devices_path: Path, devices: List[Dict[str, Any]]) -> None:
    """Write devices.json atomically (temp file + replace)."""
    tmp_path = devices_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as fh:
        json.dump(devices, fh, indent=4)
    tmp_path.replace(devices_path)


def run_retrieval(
    target_dir: Path,
    meta: Dict[str, Any],
    devices_path: Path,
) -> List[Dict[str, Any]]:
    """Pull new ``.datZ`` files for every device configured for an intersection.

    Devices with role ``'secondary'`` are always pulled before role
    ``'controller'``, regardless of their order in ``devices.json``, so a
    long-term-storage device isn't starved by the controller's short FIFO
    window advancing first. ``devices.json`` is rewritten with updated
    ``last_retrieved`` bookmarks once the run completes.

    Args:
        target_dir:   Absolute path to the intersection folder.
        meta:         Parsed ``metadata.json`` dict (supplies
                      ``controller_ip`` / ``detection_ip``).
        devices_path: Path to that intersection's ``devices.json``.

    Returns:
        List of per-device result dicts (``role``, ``host``, ``downloaded``,
        ``error``).
    """
    devices = _load_devices(devices_path)
    if not devices:
        print("No devices configured (devices.json is empty).")
        return []

    results = RetrievalEngine(target_dir, meta, devices).run()
    _save_devices(devices_path, devices)

    total = sum(r["downloaded"] for r in results)
    failed = [r for r in results if r["error"]]
    print(f"\nRetrieval complete: {total} file(s) downloaded across {len(devices)} device(s).")
    if failed:
        print(
            f"  {len(failed)} device(s) failed: "
            + ", ".join(f"{r['role']}@{r['host']}" for r in failed)
        )
    return results
