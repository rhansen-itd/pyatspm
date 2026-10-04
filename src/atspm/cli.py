"""
ATSPM Unified Command-Line Interface

Exposes subcommands from intersection configuration setup through reporting and visualization:

    atspm setup              --targetid <id>             Create a new intersection environment
    atspm retrieve           --targetid <id> [...]       Pull new .datZ files from configured devices via SCP
    atspm process            --targetid <id> [...]       Ingest data and compute cycles
    atspm report             --targetid <id> [...]       Generate ATSPM performance reports
    atspm counts             --targetid <id> [...]       Generate vehicle and pedestrian counts
    atspm splits             --targetid <id> [...]       Generate phase split and timing records
    atspm split-monitor      --targetid <id> [...]       Generate split monitor tables and plots
    atspm aog                --targetid <id> [...]       Generate Arrival on Green (AOG) tables
    atspm approach-delay     --targetid <id> [...]       Generate approach delay and Arrival on Red tables and plots
    atspm yellow-red         --targetid <id> [...]       Generate yellow and red actuation tables and plots
    atspm ped-delay          --targetid <id> [...]       Generate pedestrian delay tables and plots
    atspm wait-time          --targetid <id> [...]       Generate vehicle wait time tables and plots
    atspm detector-health    --targetid <id> [...]       Evaluate detector health rules and generate heatmap
    atspm split-failures     --targetid <id> [...]       Generate Purdue split-failure tables and plots
    atspm infer-detectors    --targetid <id> [...]       Propose detector configuration for review
    atspm discrepancies      --targetid <id> [...]       Analyze detector discrepancies
    atspm plot-detectors     --targetid <id> [...]       Generate interactive detector comparison plots
    atspm plot-coordination  --targetid <id> [...]       Generate interactive coordination diagram plots
    atspm plot-termination   --targetid <id> [...]       Generate interactive phase termination plots
    atspm video-calibrate-shapes --targetid <id> [...]   Interactively draw/edit camera shape config
    atspm video-overlay      --targetid <id> [...]       Render a video with live status overlays
    atspm video-locate-phase-change --targetid <id> [...] Find a phase's exact transition time for --start alignment
    atspm video-sync         --targetid <id> [...]       Find corrected --start from signal lamps
    atspm optimize           --targetid <id> [...]       Optimize cycle length and splits for saturated throughput
    atspm clock-drift        --targetid <id> [...]       Decode clock marks and plot controller clock drift
    atspm preempt            --targetid <id> [...]       Analyze preemption episodes and write CSV tables

The package must be installed (``pip install -e .``) for the ``atspm`` entry
point to be available.  All logic uses clean absolute imports from the
``atspm`` package — no ``sys.path`` manipulation.

Package Location: src/atspm/cli.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# The only eager package import: a bare string constant used by a dozen
# handlers and by the `setup` parser's default. Everything heavier (engines,
# plotting, pytz) stays lazy inside the handler that needs it.
from .utils.timezone import DEFAULT_TIMEZONE

# ---------------------------------------------------------------------------
# The intersections directory is always a sibling of the working-directory
# root.  We derive it at call-time (inside helpers, not at module load) so
# the module can be imported safely from any location.
# ---------------------------------------------------------------------------
_INTERSECTIONS_DIRNAME = "intersections"


# ===========================================================================
# Shared path helpers
# ===========================================================================

def _find_project_root() -> Path:
    """Walk upward from ``cwd`` until a directory containing
    ``intersections/`` is found.

    Returns:
        Absolute Path to the project root.

    Raises:
        SystemExit: When no suitable root is found.
    """
    current = Path.cwd().resolve()
    while True:
        if (current / _INTERSECTIONS_DIRNAME).is_dir():
            return current
        parent = current.parent
        if parent == current:
            _die(
                f"Could not locate the project root.\n"
                f"Make sure you are running 'atspm' from inside a directory "
                f"that contains an '{_INTERSECTIONS_DIRNAME}/' folder."
            )
        current = parent


def _get_intersections_dir() -> Path:
    """Return the absolute path to the ``intersections/`` directory.

    Returns:
        Path object for the intersections directory.
    """
    return _find_project_root() / _INTERSECTIONS_DIRNAME


def _resolve_target_name(target: Optional[str], targetid: Optional[str]) -> str:
    """Resolve the full intersection folder name from either a target or targetid.
    
    Args:
        target: The exact folder name (e.g., '2068_US-95_and_SH-8').
        targetid: The intersection ID prefix (e.g., '2068').
        
    Returns:
        The exact folder name string.
        
    Raises:
        SystemExit: If no matching folder or multiple matching folders are found.
    """
    if target:
        return target
        
    if not targetid:
        _die("Either --target or --targetid must be provided.")
        
    intersections_dir = _get_intersections_dir()
    if not intersections_dir.exists():
        _die(f"Intersections directory not found: {intersections_dir}")
        
    matches = []
    for p in intersections_dir.iterdir():
        if p.is_dir() and p.name.split('_')[0] == str(targetid):
            matches.append(p.name)
            
    if not matches:
        _die(f"No intersection folder found for ID '{targetid}'.")
    if len(matches) > 1:
        _die(f"Multiple folders found for ID '{targetid}': {', '.join(matches)}")
        
    return matches[0]


def _get_target_dir(target_name: str, must_exist: bool = True) -> Path:
    """Resolve the ``intersections/<target_name>`` directory.

    Args:
        target_name: The folder name of the intersection (e.g.
                     ``'2068_US-95_and_SH-8'``).
        must_exist:  When ``True`` exit with an error if the directory does
                     not yet exist on disk.

    Returns:
        Absolute Path to the intersection directory.

    Raises:
        SystemExit: If ``must_exist`` is ``True`` and the directory is absent.
    """
    target_dir = _get_intersections_dir() / target_name
    if must_exist and not target_dir.exists():
        _die(
            f"Target directory not found: {target_dir}\n"
            f"Tip: run 'atspm setup --target {target_name}' first."
        )
    return target_dir


def _load_metadata(target_dir: Path) -> dict:
    """Read and return the ``metadata.json`` for an intersection.

    Args:
        target_dir: Absolute path to the intersection directory.

    Returns:
        Parsed metadata dict.

    Raises:
        SystemExit: If ``metadata.json`` is missing or unparseable.
    """
    meta_path = target_dir / "metadata.json"
    if not meta_path.exists():
        _die(
            f"metadata.json not found in {target_dir}.\n"
            f"Tip: run 'atspm setup --target {target_dir.name}' to create it."
        )
    try:
        with meta_path.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        _die(f"Failed to parse metadata.json: {exc}")


def _load_devices_path(target_dir: Path) -> Path:
    """Resolve the ``devices.json`` path for an intersection, erroring if absent.

    Args:
        target_dir: Absolute path to the intersection directory.

    Returns:
        Path to ``devices.json``.

    Raises:
        SystemExit: If ``devices.json`` is missing.
    """
    devices_path = target_dir / "devices.json"
    if not devices_path.exists():
        _die(
            f"devices.json not found in {target_dir}.\n"
            f"Tip: run 'atspm setup --target {target_dir.name}' to create it, "
            f"then add controller/secondary device entries."
        )
    return devices_path


def _resolve_db_path(target_dir: Path, meta: dict) -> Path:
    """Derive the SQLite database path from directory and metadata.

    Prefers ``meta['db_filename']`` when present; falls back to
    auto-discovery (first ``*.db`` file) and finally to
    ``<intersection_id>_data.db``.

    Args:
        target_dir: Absolute path to the intersection directory.
        meta:       Parsed metadata dict.

    Returns:
        Absolute Path to the ``*.db`` file (may not exist yet).
    """
    if meta.get("db_filename"):
        return target_dir / meta["db_filename"]
    candidates = list(target_dir.glob("*.db"))
    if candidates:
        return candidates[0]
    int_id = meta.get("intersection_id", target_dir.name.split("_")[0])
    return target_dir / f"{int_id}_data.db"


def _die(message: str) -> None:
    """Print an error message and exit with status 1.

    Args:
        message: Human-readable error text.
    """
    print(f"\n❌  Error: {message}", file=sys.stderr)
    sys.exit(1)


def _sanitize_name(text: str) -> str:
    """Sanitize a free-text string for use as part of a folder name.

    Args:
        text: Arbitrary intersection name.

    Returns:
        Filesystem-safe string.
    """
    text = text.replace("&", "and").replace(" ", "_")
    return re.sub(r"[^\w\-.]", "", text)


# ===========================================================================
# Subcommand handlers
# ===========================================================================

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def handle_setup(args: argparse.Namespace) -> None:
    """Create the standard intersection directory structure.

    Generates:
    - ``intersections/<target>/``
    - ``intersections/<target>/raw_data/``
    - ``intersections/<target>/outputs/``
    - ``intersections/<target>/metadata.json``  (template)
    - ``intersections/<target>/int_cfg.csv``    (empty placeholder)
    - ``intersections/<target>/devices.json``   (empty placeholder)

    Args:
        args: Parsed CLI arguments.  Required field: ``args.target``.
    """
    target = args.target
    intersections_dir = _get_intersections_dir()
    target_dir = intersections_dir / target

    db_filename   = f"{target.split('_')[0]}_data.db"
    raw_dir       = target_dir / "raw_data"
    outputs_dir   = target_dir / "outputs"
    metadata_path = target_dir / "metadata.json"
    config_path   = target_dir / "int_cfg.csv"
    devices_path  = target_dir / "devices.json"

    print(f"\n📂  Setting up intersection environment: {target}")
    print(f"    Location: {target_dir}")

    # Directories ----------------------------------------------------------------
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    print("    ✅  Directories created")

    # metadata.json --------------------------------------------------------------
    if metadata_path.exists():
        print(
            "    ⏭️   metadata.json already exists — "
            "skipping (delete it to regenerate)"
        )
    else:
        # Try to parse id and name out of the target folder string so the
        # template is pre-filled when the caller used the recommended naming
        # convention (<id>_<SanitizedName>).
        parts = target.split("_", 1)
        derived_id   = parts[0] if parts else target
        derived_name = parts[1].replace("_", " ") if len(parts) > 1 else target

        metadata = {
            # --- System Identifiers (Required) ---
            "intersection_id":   derived_id,
            "intersection_name": derived_name,
            "timezone":          args.timezone,
            "folder_name":       target,
            "db_filename":       db_filename,

            # --- Operational (fill in or leave null) ---
            "controller_ip":     None,   # e.g. "10.71.10.50"
            "detection_type":    None,   # e.g. "Radar" | "Loops" | "Video"
            "detection_ip":      None,   # e.g. "10.71.10.51"
            "agency_id":         None,   # e.g. "ITD-D2"

            # --- Geographic (fill in or leave null) ---
            "major_road_route":  None,   # e.g. "US-95"
            "major_road_name":   None,   # e.g. "Main St"
            "minor_road_route":  None,   # e.g. "SH-8"
            "minor_road_name":   None,   # e.g. "Troy Hwy"

            # --- Coordinates (fill in or leave null) ---
            "latitude":          None,   # e.g. 46.732
            "longitude":         None,   # e.g. -117.001
        }
        with metadata_path.open("w") as fh:
            json.dump(metadata, fh, indent=4)
        print(f"    ✅  metadata.json created (timezone: {args.timezone})")

    # int_cfg.csv ----------------------------------------------------------------
    if config_path.exists():
        print("    ⏭️   int_cfg.csv already exists — skipping")
    else:
        config_path.write_text("Category,Parameter,Value\n")
        print("    ✅  int_cfg.csv placeholder created")

    # devices.json -----------------------------------------------------------------
    if devices_path.exists():
        print("    ⏭️   devices.json already exists — skipping")
    else:
        with devices_path.open("w") as fh:
            json.dump([], fh, indent=4)
        print("    ✅  devices.json placeholder created")

    # Summary --------------------------------------------------------------------
    rel = lambda p: p.relative_to(_find_project_root())  # noqa: E731
    print("\n✅  Setup complete.")
    print(f"    1. Edit metadata:       {rel(metadata_path)}")
    print(f"    2. Configure devices:   {rel(devices_path)}")
    print(f"    3. Add .datZ files to:  {rel(raw_dir)}")
    print(
        f"    4. Ingest data:         "
        f"atspm process --target \"{target}\""
    )


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

def _retrieve_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to retrieve new .datZ files for a single intersection."""
    from atspm.data import run_retrieval

    target_dir   = _get_target_dir(target_name)
    meta         = _load_metadata(target_dir)
    devices_path = _load_devices_path(target_dir)

    print(f"\n📡  Retrieving files for: {target_name}")
    run_retrieval(target_dir, meta, devices_path)


def handle_retrieve(args: argparse.Namespace) -> None:
    """Pull new .datZ files from each intersection's configured devices.

    Secondary devices (long-term storage, e.g. EVO radar) are always pulled
    before the controller (short FIFO retention window) — see
    ``atspm.data.retrieval`` for why pull order matters.

    Args:
        args: Parsed CLI arguments.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch retrieving for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _retrieve_single_intersection(target_name, args)
        except SystemExit:
            # Catch _die() to prevent a single failure from crashing the batch loop
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error retrieving {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

def _confirm_rebuild(targets: List[str], assume_yes: bool) -> None:
    """Gate a destructive rebuild behind an explicit confirmation.

    Asked once for the whole run rather than per intersection, so ``--all``
    does not turn into a prompt storm.

    Args:
        targets:    Intersection folder names about to be rebuilt.
        assume_yes: Skip the prompt (``--yes``).

    Raises:
        SystemExit: If the user declines, or if stdin is not interactive and
            ``--yes`` was not given.
    """
    if assume_yes:
        return

    print(
        f"\n⚠️   --rebuild deletes all events, cycles and ingestion_log rows "
        f"for {len(targets)} intersection(s):"
    )
    for name in targets:
        print(f"      • {name}")
    print(
        "    They are re-ingested from raw_data/, so nothing is lost that "
        "raw_data/ can still supply.\n"
        "    Config and metadata are left untouched."
    )
    sys.stdout.flush()  # keep the warning ahead of anything _die writes to stderr

    if not sys.stdin.isatty():
        _die(
            "Refusing to rebuild without confirmation on a non-interactive "
            "stdin. Re-run with --yes to proceed."
        )

    if input("    Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        _die("Rebuild cancelled.")


def _process_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to process a single intersection."""
    # Resolve fill_gaps: --fill-gaps activates Path B.
    fill_gaps: bool = args.fill_gaps
    rebuild:   bool = getattr(args, "rebuild", False)

    # Lazy imports keep module load fast and decouple from missing deps.
    from atspm.data import init_db, import_config, run_ingestion
    from atspm.data.manager import DatabaseManager

    target_dir = _get_target_dir(target_name)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)
    config_csv = target_dir / "int_cfg.csv"
    raw_dir    = target_dir / "raw_data"

    # Determine effective timezone: CLI override > metadata > default
    timezone: Optional[str] = (
        args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE
    )

    int_name = meta.get("intersection_name", target_name)
    int_id   = meta.get("intersection_id",   target_name.split("_")[0])
    if rebuild:
        mode_tag = "Rebuild"
    elif fill_gaps:
        mode_tag = "Gap Fill"
    else:
        mode_tag = "Fast Append"

    print(
        f"\n🚦  Processing {int_name} "
        f"(ID: {int_id})  [{mode_tag}]"
    )
    print(f"    DB:  {db_path.name}")
    print(f"    TZ:  {timezone}")

    # 1. Initialise DB -----------------------------------------------------------
    print("\n  🔧  Initialising database…")
    try:
        init_db(db_path)
    except Exception as exc:
        _die(f"init_db failed: {exc}")

    # 1b. Clear ingested + derived rows (Path C) ---------------------------------
    if rebuild:
        print("  🧹  Clearing events, cycles and ingestion_log…")
        try:
            with DatabaseManager(db_path) as mgr:
                deleted = mgr.clear_ingested_data()
        except Exception as exc:
            _die(f"Rebuild clear failed: {exc}")
        print(
            f"      Deleted {deleted['events']:,} events, "
            f"{deleted['cycles']:,} cycles, "
            f"{deleted['ingestion_log']} log spans."
        )

    # 2. Sync metadata -----------------------------------------------------------
    print("  📋  Syncing metadata to DB…")
    try:
        with DatabaseManager(db_path) as mgr:
            mgr.set_metadata(
                intersection_id=meta.get("intersection_id"),
                intersection_name=meta.get("intersection_name"),
                timezone=timezone,
                controller_ip=meta.get("controller_ip"),
                detection_type=meta.get("detection_type"),
                detection_ip=meta.get("detection_ip"),
                major_road_route=meta.get("major_road_route"),
                major_road_name=meta.get("major_road_name"),
                minor_road_route=meta.get("minor_road_route"),
                minor_road_name=meta.get("minor_road_name"),
                latitude=meta.get("latitude"),
                longitude=meta.get("longitude"),
                agency_id=meta.get("agency_id"),
            )
    except Exception as exc:
        _die(f"set_metadata failed: {exc}")

    # 3. Import configuration CSV ------------------------------------------------
    if config_csv.exists():
        print("  ⚙️   Importing intersection config…")
        try:
            import_config(config_csv, db_path)
        except Exception as exc:
            # Config import is non-fatal: missing/malformed CSV is common for
            # brand-new intersections.
            print(f"  ⚠️   Config import warning (non-fatal): {exc}")
    else:
        print(f"  ⚠️   int_cfg.csv not found — skipping config import")

    # 4. Ingest + (optionally) cycle processing ----------------------------------
    run_cycles = not args.no_cycles
    cycle_tag  = "enabled" if run_cycles else "skipped (--no-cycles)"
    print(
        f"\n  🚀  Running ingestion "
        f"(batch_size={args.batch_size}, cycles={cycle_tag})…"
    )
    try:
        run_ingestion(
            db_path=db_path,
            data_dir=raw_dir,
            timezone=timezone,
            fill_gaps=fill_gaps,
            batch_size=args.batch_size,
            run_cycles=run_cycles,
        )
    except Exception as exc:
        traceback.print_exc()
        _die(f"Ingestion failed: {exc}")

    print("\n✅  Processing complete.")


def handle_process(args: argparse.Namespace) -> None:
    """Ingest raw ``.datZ`` data and compute signal cycles.

    Loads ``metadata.json``, syncs it to the SQLite ``metadata`` table,
    imports the configuration CSV, then delegates to ``run_ingestion``
    (which drives both ingestion and optional cycle processing in one pass).

    Path A (Fast Append, default): only files newer than the last ingested
    span are scanned; cycles are updated from the last known cycle boundary.

    Path B (Gap Fill, ``--fill-gaps``): the full file list is scanned for
    uncovered holes; gap markers made obsolete by new data are scrubbed;
    cycles are surgically repaired between gap-bounded anchors.

    Path C (Rebuild, ``--rebuild``): ``events``, ``cycles`` and
    ``ingestion_log`` are deleted first, then every ``.datZ`` file is
    re-ingested from scratch.  Destructive, so it prompts for confirmation
    unless ``--yes`` is given.

    Args:
        args: Parsed CLI arguments.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch processing {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    if getattr(args, "rebuild", False):
        _confirm_rebuild(targets, assume_yes=getattr(args, "yes", False))

    for target_name in targets:
        try:
            _process_single_intersection(target_name, args)
        except SystemExit:
            # Catch _die() to prevent a single failure from crashing the batch loop
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error processing {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------

def _counts_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate counts for a single intersection."""
    from atspm.data.counts import CountEngine

    target_dir = _get_target_dir(target_name)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)
    
    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚗  Generating counts for {int_name}")
    print(f"    DB:     {db_path.name}")
    print(f"    Window: {args.start} → {args.end}")
    print(f"    Bins:   {args.bin_len}{' (hourly)' if args.hourly else ''}")

    # handle bin_len type (int or string "cycle")
    bin_len = args.bin_len
    if bin_len.isdigit():
        bin_len = int(bin_len)

    engine = CountEngine(db_path=db_path, timezone=timezone)
    
    try:
        if args.type == "vehicle":
            engine.vehicle_counts(
                start=args.start, end=args.end, bin_len=bin_len, hourly=args.hourly,
                include_detectors=args.include_detectors, exclude_missing=args.exclude_missing,
                output_dir=output_dir,
            )
        elif args.type == "ped":
            engine.ped_counts(
                start=args.start, end=args.end, bin_len=bin_len, hourly=args.hourly,
                exclude_missing=args.exclude_missing, output_dir=output_dir,
            )
        else:
            engine.combined_counts(
                start=args.start, end=args.end, bin_len=bin_len, hourly=args.hourly,
                include_detectors=args.include_detectors, exclude_missing=args.exclude_missing,
                output_dir=output_dir,
            )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Count generation failed: {exc}")


def handle_counts(args: argparse.Namespace) -> None:
    """Generate vehicle and pedestrian counts to CSV.
    
    Args:
        args: Parsed CLI arguments.
    """
    intersections_dir = _get_intersections_dir()
    
    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating counts for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _counts_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error generating counts for {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def _splits_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate phase splits for a single intersection."""
    from atspm.data.phases import PhaseEngine

    target_dir = _get_target_dir(target_name)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)
    
    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n⏱️   Generating phase splits for {int_name}")
    print(f"    DB:     {db_path.name}")
    print(f"    Window: {args.start} → {args.end}")
    print(f"    Bins:   {args.bin_len}")

    bin_len = args.bin_len
    if bin_len.isdigit():
        bin_len = int(bin_len)

    engine = PhaseEngine(db_path=db_path, timezone=timezone)
    
    try:
        engine.phase_splits(
            start=args.start, end=args.end, bin_len=bin_len, 
            report_mode=args.report_mode, phases=args.phases,
            include_no_clearance=args.include_no_clearance, 
            exclude_missing=args.exclude_missing, output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Phase splits generation failed: {exc}")


def handle_splits(args: argparse.Namespace) -> None:
    """Generate binned or per-cycle phase split and timing records to CSV.
    
    Args:
        args: Parsed CLI arguments.
    """
    intersections_dir = _get_intersections_dir()
    
    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating phase splits for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _splits_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error generating splits for {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# aog
# ---------------------------------------------------------------------------

def _aog_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate Arrival on Green for a single intersection.

    Resolves the database path and timezone from ``metadata.json``, then
    delegates entirely to :class:`atspm.data.aog.AogEngine`.  All I/O (event
    queries, CSV writing) is handled inside the engine; this function is
    responsible only for path resolution, argument forwarding, and error
    surfacing.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``aog`` subcommand.
    """
    from atspm.data.aog import AogEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🟢  Generating Arrival on Green for {int_name}")
    print(f"    DB:     {db_path.name}")
    print(f"    Window: {args.start} → {args.end}")
    print(f"    Bins:   {args.bin_len}")
    if args.offset:
        print(f"    Offset: {args.offset}s")
    if args.phases:
        print(f"    Phases: {args.phases}")

    # Coerce bin_len to int when it is a digit string; leave "cycle" as-is.
    bin_len = args.bin_len
    if isinstance(bin_len, str) and bin_len.isdigit():
        bin_len = int(bin_len)

    engine = AogEngine(db_path=db_path, timezone=timezone)

    try:
        engine.arrival_on_green(
            start=args.start,
            end=args.end,
            phases=args.phases,
            arrival_offset_sec=args.offset,
            bin_len=bin_len,
            exclude_missing=args.exclude_missing,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"AOG generation failed: {exc}")


def handle_aog(args: argparse.Namespace) -> None:
    """Generate Arrival on Green tables to CSV for one or more intersections.

    Reads advance detector mappings from the active configuration
    (``Det_P{N}_Arrival`` keys) and writes per-cycle and/or binned AOG
    tables to ``intersections/<target>/outputs/``.

    Args:
        args: Parsed CLI arguments from the ``aog`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating AOG for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _aog_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error generating AOG for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# approach-delay
# ---------------------------------------------------------------------------

def _approach_delay_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate approach delay for a single intersection.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``approach-delay`` subcommand.
    """
    import pandas as pd
    from atspm.data.approach_delay import ApproachDelayEngine
    from atspm.data.critical import CriticalMovementEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚦  Generating Approach Delay for {int_name}")
    print(f"    DB:        {db_path.name}")
    print(f"    Window:    {args.start} → {args.end}")
    print(f"    Offset:    {args.offset}s")
    print(f"    Bins:      {args.bin_len}")
    if args.phases:
        print(f"    Phases:    {args.phases}")

    engine = ApproachDelayEngine(db_path=db_path, timezone=timezone)

    try:
        engine.approach_delay(
            start=args.start,
            end=args.end,
            phases=args.phases,
            travel_time_sec=args.offset,
            bin_len=args.bin_len,
            exclude_missing=args.exclude_missing,
            make_plot=not args.no_plot,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Approach delay generation failed: {exc}")

    # After writing, print a short per-phase summary from the cycle CSV
    start_dt, end_dt = CriticalMovementEngine._parse_range(args.start, args.end)
    stamp = engine._format_stamp(start_dt, end_dt)
    cycle_file = output_dir / f"AD_Cycle_{stamp}.csv"
    if cycle_file.exists():
        cyc = pd.read_csv(cycle_file)
        if not cyc.empty and "phase" in cyc.columns:
            for ph in sorted(cyc["phase"].unique()):
                sub = cyc.loc[cyc["phase"] == ph]
                n_total = len(sub)
                n_cens = int(sub["censored"].sum())
                n_ok = n_total - n_cens
                ok_sub = sub.loc[~sub["censored"]]

                tot_arr = ok_sub["arrivals"].sum() if "arrivals" in ok_sub.columns else 0
                tot_g = ok_sub["arrivals_green"].sum() if "arrivals_green" in ok_sub.columns else 0
                tot_y = ok_sub["arrivals_yellow"].sum() if "arrivals_yellow" in ok_sub.columns else 0
                tot_r = ok_sub["arrivals_red"].sum() if "arrivals_red" in ok_sub.columns else 0
                tot_delay = ok_sub["total_delay_s"].sum() if "total_delay_s" in ok_sub.columns else 0.0

                src = str(sub["travel_source"].iloc[0]) if "travel_source" in sub.columns else "offset"
                if tot_arr > 0:
                    aog_pct = (tot_g / tot_arr) * 100.0
                    aoy_pct = (tot_y / tot_arr) * 100.0
                    aor_pct = (tot_r / tot_arr) * 100.0
                    delay_per_veh = tot_delay / tot_arr
                    print(
                        f"    Ph{ph}: {n_ok} cycles ({n_cens} censored), "
                        f"AoG {aog_pct:.1f}% / AoY {aoy_pct:.1f}% / AoR {aor_pct:.1f}%, "
                        f"delay {delay_per_veh:.1f} s/veh ({src})"
                    )
                else:
                    print(
                        f"    Ph{ph}: {n_ok} cycles ({n_cens} censored), "
                        f"AoG NaN% / AoY NaN% / AoR NaN%, "
                        f"delay NaN s/veh ({src})"
                    )


def handle_approach_delay(args: argparse.Namespace) -> None:
    """Generate approach delay and Arrival on Red tables and plots for one or more intersections.

    Args:
        args: Parsed CLI arguments from the ``approach-delay`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating approach delay for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _approach_delay_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error generating approach delay for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# yellow-red
# ---------------------------------------------------------------------------

def _yellow_red_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate yellow and red actuations for a single intersection.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``yellow-red`` subcommand.
    """
    import pandas as pd
    from atspm.data.critical import CriticalMovementEngine
    from atspm.data.yellow_red_actuations import YellowRedEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚦  Generating Yellow and Red Actuations for {int_name}")
    print(f"    DB:        {db_path.name}")
    print(f"    Window:    {args.start} → {args.end}")
    print(f"    Role:      {args.role}")
    print(f"    Severe:    {args.severe_sec}s")
    print(f"    Bins:      {args.bin_len}min")
    if args.phases:
        print(f"    Phases:    {args.phases}")

    engine = YellowRedEngine(db_path=db_path, timezone=timezone)

    try:
        engine.yellow_red(
            start=args.start,
            end=args.end,
            phases=args.phases,
            role=args.role,
            severe_sec=args.severe_sec,
            bin_len=args.bin_len,
            use_exclusions=not args.no_exclusions,
            make_plot=not args.no_plot,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Yellow and red actuations generation failed: {exc}")

    # After writing, print a short summary from the plans CSV
    start_dt, end_dt = CriticalMovementEngine._parse_range(args.start, args.end)
    stamp = engine._format_stamp(start_dt, end_dt)
    plans_file = output_dir / f"YRA_Plans_{stamp}.csv"
    if plans_file.exists():
        plans_df = pd.read_csv(plans_file)
        if not plans_df.empty:
            for _, row in plans_df.iterrows():
                ph = int(row["phase"])
                plan = int(row["coord_plan"]) if pd.notna(row["coord_plan"]) else "?"
                n_cyc = int(row["n_cycles"])
                viol = int(row["violations"])
                sev = int(row["severe"])
                vpc = float(row["violations_per_cycle"]) if pd.notna(row["violations_per_cycle"]) else 0.0
                pct = float(row["pct_violations"]) * 100.0 if pd.notna(row["pct_violations"]) else 0.0
                print(
                    f"    Ph{ph} Plan {plan}: {n_cyc} cycles, "
                    f"{viol} violations ({sev} severe), "
                    f"{vpc:.2f} viol/cyc, {pct:.1f}% violations"
                )


def handle_yellow_red(args: argparse.Namespace) -> None:
    """Generate yellow and red actuation tables and plots for one or more intersections.

    Args:
        args: Parsed CLI arguments from the ``yellow-red`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating yellow and red actuations for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _yellow_red_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error generating yellow and red actuations for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# ped-delay
# ---------------------------------------------------------------------------

def _ped_delay_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate pedestrian delay for a single intersection.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``ped-delay`` subcommand.
    """
    import pandas as pd
    from atspm.data.call_service import CallServiceEngine
    from atspm.data.critical import CriticalMovementEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚶  Generating Pedestrian Delay for {int_name}")
    print(f"    DB:        {db_path.name}")
    print(f"    Window:    {args.start} → {args.end}")
    print(f"    Bins:      {args.bin_len}min")
    if args.phases:
        print(f"    Phases:    {args.phases}")

    engine = CallServiceEngine(db_path=db_path, timezone=timezone)

    try:
        engine.ped_delay(
            start=args.start,
            end=args.end,
            phases=args.phases,
            bin_len=args.bin_len,
            make_plot=not args.no_plot,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Pedestrian delay generation failed: {exc}")

    # After writing, print a short summary from the plans CSV
    start_dt, end_dt = CriticalMovementEngine._parse_range(args.start, args.end)
    stamp = engine._format_stamp(start_dt, end_dt)
    plans_file = output_dir / f"PedDelay_Plans_{stamp}.csv"
    if plans_file.exists():
        plans_df = pd.read_csv(plans_file)
        if not plans_df.empty:
            for _, row in plans_df.iterrows():
                ph = int(row["phase"])
                plan = int(row["coord_plan"]) if pd.notna(row["coord_plan"]) else "?"
                n_walks = int(row["n_walks"])
                n_called = int(row["n_called"])
                avg_d = f"{row['avg_delay_s']:.1f}s" if pd.notna(row.get("avg_delay_s")) else "N/A"
                max_d = f"{row['max_delay_s']:.1f}s" if pd.notna(row.get("max_delay_s")) else "N/A"
                print(
                    f"    Ph{ph} Plan {plan}: {n_walks} walks, {n_called} called, "
                    f"avg delay {avg_d}, max delay {max_d}"
                )


def handle_ped_delay(args: argparse.Namespace) -> None:
    """Generate pedestrian delay tables and plots for one or more intersections.

    Args:
        args: Parsed CLI arguments from the ``ped-delay`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating pedestrian delay for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _ped_delay_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error generating pedestrian delay for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# wait-time
# ---------------------------------------------------------------------------

def _wait_time_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate vehicle wait time for a single intersection.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``wait-time`` subcommand.
    """
    import pandas as pd
    from atspm.data.call_service import CallServiceEngine
    from atspm.data.critical import CriticalMovementEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    max_wait = None if args.max_wait == 0 else args.max_wait

    print(f"\n⏱️  Generating Wait Time for {int_name}")
    print(f"    DB:        {db_path.name}")
    print(f"    Window:    {args.start} → {args.end}")
    print(f"    Dropping:  {args.dropping}")
    print(f"    Max Wait:  {args.max_wait}s")
    print(f"    Bins:      {args.bin_len}min")
    if args.phases:
        print(f"    Phases:    {args.phases}")

    engine = CallServiceEngine(db_path=db_path, timezone=timezone)

    try:
        engine.wait_time(
            start=args.start,
            end=args.end,
            phases=args.phases,
            dropping=args.dropping,
            max_wait=max_wait,
            bin_len=args.bin_len,
            make_plot=not args.no_plot,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Wait time generation failed: {exc}")

    # After writing, print a short summary from the plans CSV
    start_dt, end_dt = CriticalMovementEngine._parse_range(args.start, args.end)
    stamp = engine._format_stamp(start_dt, end_dt)
    plans_file = output_dir / f"WaitTime_Plans_{stamp}.csv"
    if plans_file.exists():
        plans_df = pd.read_csv(plans_file)
        if not plans_df.empty:
            for _, row in plans_df.iterrows():
                ph = int(row["phase"])
                plan = int(row["coord_plan"]) if pd.notna(row["coord_plan"]) else "?"
                n_win = int(row["n_windows"])
                n_called = int(row["n_called"])
                n_held = int(row["n_held"])
                avg_w = f"{row['avg_wait_s']:.1f}s" if pd.notna(row.get("avg_wait_s")) else "N/A"
                avg_udot = f"{row['avg_wait_udot_s']:.1f}s" if pd.notna(row.get("avg_wait_udot_s")) else "N/A"
                print(
                    f"    Ph{ph} Plan {plan}: {n_win} windows, {n_called} called, {n_held} held, "
                    f"avg wait {avg_w}, avg wait (UDOT) {avg_udot}"
                )


def handle_wait_time(args: argparse.Namespace) -> None:
    """Generate wait time tables and plots for one or more intersections.

    Args:
        args: Parsed CLI arguments from the ``wait-time`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating wait time for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _wait_time_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error generating wait time for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# split-monitor
# ---------------------------------------------------------------------------

def _split_monitor_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate split monitor for a single intersection.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``split-monitor`` subcommand.
    """
    import numpy as np
    import pandas as pd
    from atspm.data.split_monitor import SplitMonitorEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚦  Generating Split Monitor for {int_name}")
    print(f"    DB:        {db_path.name}")
    print(f"    Window:    {args.start} → {args.end}")
    if args.phases:
        print(f"    Phases:    {args.phases}")

    engine = SplitMonitorEngine(db_path=db_path, timezone=timezone)

    try:
        engine.split_monitor(
            start=args.start,
            end=args.end,
            phases=args.phases,
            percentiles=tuple(args.percentiles),
            make_plot=not args.no_plot,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Split monitor generation failed: {exc}")

    # After writing, print a short summary from the stats CSV
    start_dt, end_dt = SplitMonitorEngine._parse_range(args.start, args.end)
    stamp = engine._format_stamp(start_dt, end_dt)
    stats_file = output_dir / f"SM_Stats_{stamp}.csv"
    if stats_file.exists():
        st = pd.read_csv(stats_file)
        if not st.empty and "phase" in st.columns:
            pa, pb = tuple(args.percentiles)
            na, nb = f"split_p{pa:g}", f"split_p{pb:g}"
            for _, row in st.iterrows():
                ph = int(row["phase"])
                plan = row["plan"]
                plan_str = f"Plan {int(plan)}" if pd.notna(plan) else "Plan NA"
                n_cyc = int(row["n_cycles"])
                prog = row["programmed_split"]
                prog_str = f"{prog:.1f}s" if pd.notna(prog) else "NaN"
                val_a = row[na] if na in row else np.nan
                val_b = row[nb] if nb in row else np.nan
                gap_pct = row["gap_out_pct"] * 100.0 if "gap_out_pct" in row else 0.0
                max_pct = row["max_out_pct"] * 100.0 if "max_out_pct" in row else 0.0
                fo_pct = row["force_off_pct"] * 100.0 if "force_off_pct" in row else 0.0
                print(
                    f"    Ph{ph} {plan_str}: {n_cyc} services, prog {prog_str}, "
                    f"{na} {val_a:.1f}s / {nb} {val_b:.1f}s, "
                    f"gap-out {gap_pct:.1f}% / max-out {max_pct:.1f}% / force-off {fo_pct:.1f}%"
                )


def handle_split_monitor(args: argparse.Namespace) -> None:
    """Generate split monitor tables and plots for one or more intersections.

    Args:
        args: Parsed CLI arguments from the ``split-monitor`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating split monitor for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _split_monitor_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error generating split monitor for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# detector-health
# ---------------------------------------------------------------------------

def _detector_health_single_intersection(target_name: str, args: argparse.Namespace) -> int:
    """Core logic to evaluate detector health for a single intersection.

    Resolves the database path and timezone from ``metadata.json``, then
    delegates entirely to :class:`atspm.data.detector_health.DetectorHealthEngine`.
    All I/O (event queries, CSV and HTML heatmap writing) is handled inside the engine;
    this function is responsible only for path resolution, argument forwarding,
    and error surfacing.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``detector-health`` subcommand.

    Returns:
        Exit code (0 = clean, 1 = low, 2 = high).
    """
    from atspm.data.detector_health import DetectorHealthEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    start = args.start
    end = args.end or args.start

    print(f"\n🔍  Evaluating Detector Health for {int_name}")
    print(f"    DB:           {db_path.name}")
    print(f"    Window:       {start} → {end}")
    print(f"    Filter win:   {args.window}")
    print(f"    Min severity: {args.min_severity}")

    engine = DetectorHealthEngine(db_path=db_path, timezone=timezone)

    try:
        result = engine.detector_health(
            start=start,
            end=end,
            window=args.window,
            min_severity=args.min_severity,
            output_dir=output_dir,
        )
        return int(result["exit_code"])
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Detector health evaluation failed: {exc}")


def handle_detector_health(args: argparse.Namespace) -> None:
    """Evaluate detector health for one or more intersections.

    Runs deterministic detector-health rules, records findings to
    ``detector_findings``, and writes reported CSV and HTML heatmap to
    ``intersections/<target>/outputs/``.

    Args:
        args: Parsed CLI arguments from the ``detector-health`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch evaluating detector health for {len(targets)} intersections...")
        for target_name in targets:
            try:
                _detector_health_single_intersection(target_name, args)
            except SystemExit:
                print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
            except Exception as exc:
                print(
                    f"\n❌ Unexpected error evaluating detector health for {target_name}: {exc}",
                    file=sys.stderr,
                )
                if getattr(args, "verbose", False):
                    traceback.print_exc()
    else:
        target_name = _resolve_target_name(args.target, args.targetid)
        exit_code = _detector_health_single_intersection(target_name, args)
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# split-failures
# ---------------------------------------------------------------------------

def _split_failures_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate split failures for a single intersection.

    Resolves the database path and timezone from ``metadata.json``, then
    delegates entirely to :class:`atspm.data.split_failures.SplitFailureEngine`.
    All I/O (event queries, CSV/HTML writing) is handled inside the engine;
    this function is responsible only for path resolution, argument forwarding,
    and error surfacing.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``split-failures`` subcommand.
    """
    from atspm.data.split_failures import SplitFailureEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚦  Generating Split Failures for {int_name}")
    print(f"    DB:        {db_path.name}")
    print(f"    Window:    {args.start} → {args.end}")
    print(f"    Aggregate: {args.aggregate}")
    print(f"    Threshold: {args.threshold}")
    print(f"    ROR Sec:   {args.ror_seconds}s")
    print(f"    Bins:      {args.bin_len}")
    if args.phases:
        print(f"    Phases:    {args.phases}")

    engine = SplitFailureEngine(db_path=db_path, timezone=timezone)

    try:
        engine.split_failures(
            start=args.start,
            end=args.end,
            phases=args.phases,
            aggregate=args.aggregate,
            threshold=args.threshold,
            ror_seconds=args.ror_seconds,
            include_yellow=args.include_yellow,
            bin_len=args.bin_len,
            exclude_missing=args.exclude_missing,
            make_plot=not args.no_plot,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Split failure generation failed: {exc}")


def handle_split_failures(args: argparse.Namespace) -> None:
    """Generate Purdue split failures for one or more intersections.

    Reads stop-bar detector mappings from the active configuration
    (``Det_P{N}_Stop_Bar`` / ``Det_P{N}_Stopbar`` keys) and writes per-cycle,
    lane, and binned split-failure tables and scatter plots to
    ``intersections/<target>/outputs/``.

    Args:
        args: Parsed CLI arguments from the ``split-failures`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating split failures for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _split_failures_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error generating split failures for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# infer-detectors
# ---------------------------------------------------------------------------

def _infer_detectors_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to infer detector configuration for a single intersection.

    Resolves the database path and timezone from ``metadata.json``, then
    delegates entirely to :class:`atspm.data.detector_inference.DetectorInferenceEngine`.
    All I/O (event queries, CSV writing) is handled inside the engine; this function
    is responsible only for path resolution, argument forwarding, and error surfacing.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``infer-detectors`` subcommand.
    """
    from atspm.data.detector_inference import DetectorInferenceEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = getattr(args, "timezone", None) or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🔍  Inferring detector configuration for {int_name}")
    print(f"    DB:             {db_path.name}")
    print(f"    Window:         {args.start} → {args.end}")
    print(f"    Min actuations: {args.min_actuations}")
    if args.all_phases:
        print("    Candidates:     all phases (ignoring RB_* ring config)")

    engine = DetectorInferenceEngine(db_path=db_path, timezone=timezone)

    try:
        engine.infer(
            start=args.start,
            end=args.end,
            use_ring_config=not args.all_phases,
            min_actuations=args.min_actuations,
            output_dir=output_dir,
        )
    except Exception as exc:
        if getattr(args, "verbose", False):
            traceback.print_exc()
        _die(f"Detector inference failed: {exc}")


def handle_infer_detectors(args: argparse.Namespace) -> None:
    """Propose a detector configuration for review and never edit int_cfg.csv.

    Args:
        args: Parsed CLI arguments from the ``infer-detectors`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch inferring detector configurations for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _infer_detectors_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error inferring detector configuration for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# flow
# ---------------------------------------------------------------------------

def _flow_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate split flow-rate outputs for one intersection.

    Resolves the database path and timezone from ``metadata.json``, then
    delegates entirely to :class:`atspm.data.flow.FlowRateEngine`.  All I/O
    (event queries, CSV/HTML writing) is handled inside the engine; this
    function is responsible only for path resolution, argument forwarding,
    and error surfacing.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``flow`` subcommand.
    """
    from atspm.data.flow import FlowRateEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚗  Generating split flow rate for {int_name}")
    print(f"    DB:     {db_path.name}")
    print(f"    Window: {args.start} → {args.end}")
    print(f"    Normalize: {args.normalize}")
    if args.stratify:
        print("    Selection: stratified by (plan, split)")
    if args.phases:
        print(f"    Phases: {args.phases}")
    if args.plans:
        print(f"    Plans:  {args.plans}")

    engine = FlowRateEngine(db_path=db_path, timezone=timezone)

    try:
        engine.flow(
            start=args.start,
            end=args.end,
            phases=args.phases,
            plans=args.plans,
            pct=args.pct,
            max_lost=args.max_lost,
            split_tolerance=args.split_tolerance,
            normalize=args.normalize,
            fixed_lost=args.fixed_lost,
            stratify=args.stratify,
            rolling=args.rolling,
            make_plot=not args.no_plot,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Flow-rate generation failed: {exc}")


def handle_flow(args: argparse.Namespace) -> None:
    """Generate split flow-rate tables and plots for one or more intersections.

    Reads stop-bar detector mappings from the active configuration
    (``P{N} Stop Bar`` rows: ``Det_P{N}_Stop_Bar``; ``Det_P{N}_Stopbar``
    also accepted) and writes per-cycle CSVs, wide rate
    profiles, and interactive HTML plots to
    ``intersections/<target>/outputs/``.

    Args:
        args: Parsed CLI arguments from the ``flow`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating flow rate for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _flow_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error generating flow rate for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# critical
# ---------------------------------------------------------------------------

def _critical_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to run critical movement analysis for one intersection.

    Resolves the database path and timezone from ``metadata.json``, then
    delegates entirely to
    :class:`atspm.data.critical.CriticalMovementEngine`.  All I/O (count
    queries, cycle queries, CSV writing) is handled inside the engine;
    this function is responsible only for path resolution, argument
    forwarding, and error surfacing.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``critical`` subcommand.
    """
    from atspm.data.critical import CriticalMovementEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚦  Critical movement analysis for {int_name}")
    print(f"    DB:     {db_path.name}")
    print(f"    Window: {args.start} → {args.end}")
    print(f"    Basis:  {args.basis}")

    engine = CriticalMovementEngine(db_path=db_path, timezone=timezone)

    try:
        engine.critical(
            start=args.start,
            end=args.end,
            bin_len=args.bin_len,
            basis=args.basis,
            exclude_missing=not args.include_missing,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Critical movement analysis failed: {exc}")


def handle_critical(args: argparse.Namespace) -> None:
    """Run critical movement analysis for one or more intersections.

    Derives the ring/barrier structure from ``RB_*`` config and observed
    cycle sequences, maps movement counts (``TM_*``) to phases via
    stop-bar detector overlap, and writes per-phase and per-barrier-group
    criticality tables to ``intersections/<target>/outputs/``.

    Args:
        args: Parsed CLI arguments from the ``critical`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(
            f"\n🌍 Batch critical movement analysis for "
            f"{len(targets)} intersections..."
        )
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _critical_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error in critical movement analysis for "
                f"{target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

def _optimize_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to run cycle length and split optimization for one intersection.

    Args:
        target_name: Exact intersection folder name.
        args: Parsed CLI arguments from the ``optimize`` subcommand.
    """
    from atspm.data.optimizer import OptimizerEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    if getattr(args, "validate", False):
        print(f"\n🚦  Throughput validation for {int_name}")
    else:
        print(f"\n🚦  Throughput optimization for {int_name}")
    print(f"    DB:        {db_path.name}")
    print(f"    Window:    {args.start} → {args.end}")
    print(f"    Saturated: {args.saturated}")

    engine = OptimizerEngine(db_path=db_path, timezone=timezone)

    try:
        if getattr(args, "validate", False):
            engine.validate(
                start=args.start,
                end=args.end,
                saturated=args.saturated,
                plans=args.plans,
                pct=args.pct,
                split_tolerance=args.split_tolerance,
                max_lost=args.max_lost,
                sat_threshold=args.sat_threshold,
                min_plan_cycles=args.min_plan_cycles,
                split_cover_tol=args.split_cover_tol,
                rank_deadband_pct=args.rank_deadband_pct,
                change_tol_pp=args.change_tol_pp,
                output_dir=output_dir,
            )
        else:
            engine.optimize(
                start=args.start,
                end=args.end,
                saturated=args.saturated,
                plans=args.plans,
                pct=args.pct,
                split_tolerance=args.split_tolerance,
                stratify=args.stratify,
                max_lost=args.max_lost,
                sat_threshold=args.sat_threshold,
                demand_stat=args.demand_stat,
                default_min_split=args.default_min_split,
                c_min=args.c_min,
                c_max=args.c_max,
                c_step=args.c_step,
                flat_tol_pct=args.flat_tol_pct,
                boundary_rate_tol=100.0,
                bin_len=args.bin_len,
                exclude_missing=not args.include_missing,
                make_plot=not args.no_plot,
                output_dir=output_dir,
            )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Optimization failed: {exc}")


def handle_optimize(args: argparse.Namespace) -> None:
    """Run throughput cycle-length and split optimization for one or more intersections.

    Args:
        args: Parsed CLI arguments from the ``optimize`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(
            f"\n🌍 Batch throughput optimization for "
            f"{len(targets)} intersections..."
        )
        print(
            f"   Note: saturated phases {args.saturated} apply to all intersections."
        )
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _optimize_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error in optimization for "
                f"{target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# clock-drift
# ---------------------------------------------------------------------------

def _clock_drift_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to decode clock marks and plot drift for one intersection.

    Resolves the database path and timezone from ``metadata.json``, then
    delegates to :class:`atspm.data.clock_marks.ClockMarkEngine`. All I/O
    (SQL queries, CSV writing, HTML plot rendering) is handled inside the engine.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``clock-drift`` subcommand.
    """
    from atspm.data.clock_marks import ClockMarkEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    # Determine send_log_path
    send_log_path = args.send_log
    if send_log_path is None:
        candidate = target_dir / "eos-time.jsonl"
        if candidate.exists():
            send_log_path = candidate

    print(f"\n⏰  Clock drift analysis for {int_name}")
    print(f"    DB:       {db_path.name}")
    print(f"    Window:   {args.start} → {args.end}")
    if send_log_path:
        print(f"    Send log: {send_log_path}")

    engine = ClockMarkEngine(db_path=db_path, timezone=timezone)

    try:
        engine.decode(
            start=args.start,
            end=args.end,
            send_log_path=send_log_path,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Clock drift analysis failed: {exc}")


def handle_clock_drift(args: argparse.Namespace) -> None:
    """Decode clock marks and plot drift for one or more intersections.

    Args:
        args: Parsed CLI arguments from the ``clock-drift`` subcommand.
    """
    if getattr(args, "all", False) and getattr(args, "send_log", None):
        _die("--send-log cannot be combined with --all (one send log belongs to one controller).")

    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(
            f"\n🌍 Batch clock drift analysis for "
            f"{len(targets)} intersections..."
        )
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _clock_drift_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error in clock drift analysis for "
                f"{target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# preempt
# ---------------------------------------------------------------------------

def _preempt_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to analyze preemption episodes for one intersection.

    Resolves the database path and timezone from ``metadata.json``, then
    delegates to :class:`atspm.data.preempt.PreemptEngine`. All I/O
    (SQL queries, CSV writing) is handled inside the engine.

    Args:
        target_name: Exact intersection folder name
            (e.g., ``'2068_US-95_and_SH-8'``).
        args: Parsed CLI arguments from the ``preempt`` subcommand.
    """
    from atspm.data.preempt import PreemptEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n🚨  Preemption analysis for {int_name}")
    print(f"    DB:     {db_path.name}")
    print(f"    Window: {args.start} → {args.end}")

    engine = PreemptEngine(db_path=db_path, timezone=timezone)

    try:
        res = engine.preempt(
            start=args.start,
            end=args.end,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Preemption analysis failed: {exc}")

    episodes = res.get("episodes")
    if episodes is None or episodes.empty:
        print("  No preemption requests found.")
        return

    import pandas as pd
    for p, group in episodes.groupby("preempt", sort=True):
        reqs = len(group)
        served = int(group["served"].sum())
        unserved = int((~group["served"] & ~group["censored"]).sum())
        censored = int(group["censored"].sum())
        max_pres = int(group["max_presence"].sum())
        dwell = group.loc[group["served"] & ~group["censored"], "dwell_s"].dropna()
        mean_dwell = f"{dwell.mean():.1f}s" if not dwell.empty else "N/A"
        max_dwell = f"{dwell.max():.1f}s" if not dwell.empty else "N/A"
        print(
            f"  Preempt {p}: {reqs} requests, {served} served, {unserved} unserved, "
            f"{censored} censored, {max_pres} max-presence hits, "
            f"dwell: {mean_dwell} mean / {max_dwell} max"
        )


def handle_preempt(args: argparse.Namespace) -> None:
    """Analyze preemption episodes for one or more intersections.

    Args:
        args: Parsed CLI arguments from the ``preempt`` subcommand.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch preemption analysis for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _preempt_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(
                f"\n❌ Unexpected error in preemption analysis for {target_name}: {exc}",
                file=sys.stderr,
            )
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _report_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to generate reports for a single intersection."""
    from atspm.data.processing import CycleProcessor
    from atspm.reports.generators import PlotGenerator

    target_dir = _get_target_dir(target_name)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)
    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    print(f"\n📊  Generating reports for {int_name}")
    print(f"    DB:     {db_path.name}")
    print(f"    Output: {output_dir}")
    print(f"    Dates:  {', '.join(args.dates)}")

    # Cycle processor – used for validation and on-demand gap fills.
    processor = CycleProcessor(db_path)

    # Optional ring-phase backfill (safe to run repeatedly; fast when done).
    if args.backfill:
        print("\n  🔄  Running ring-phase backfill…")
        updated = processor.backfill_ring_phases()
        print(f"      Backfilled {updated} rows.")

    # Per-date validation --------------------------------------------------------
    for date_str in args.dates:
        print(f"\n  📅  Validating cycles for {date_str}…")
        # get_cycle_summary_for_date lives on the old-style processor; for
        # forward-compatibility we call run() on a narrow span when the date
        # has no coverage at all.
        stats = _get_cycle_summary(processor, date_str)

        if stats is None:
            print(
                f"      ⚠️   No cycles found for {date_str}. "
                "Attempting on-demand reprocess…"
            )
            _reprocess_date(processor, date_str, meta.get("timezone"))
            stats = _get_cycle_summary(processor, date_str)

        if stats:
            print(
                f"      ✅  {stats.get('cycle_count', '?')} cycles "
                f"({stats.get('detection_method', '?')})"
            )
        else:
            print(
                f"      ⚠️   Still no cycles for {date_str} — "
                "report may be empty."
            )

    # Plot generation ------------------------------------------------------------
    print("\n  🖼️   Generating plots…")
    gen = PlotGenerator(db_path, output_dir)
    errors: list[str] = []
    for date_str in args.dates:
        try:
            gen.generate_for_date(date_str)
            print(f"      ✅  {date_str} → {output_dir / date_str}")
        except Exception as exc:
            errors.append(date_str)
            print(f"      ❌  {date_str} failed: {exc}")
            if args.verbose:
                traceback.print_exc()

    # Summary --------------------------------------------------------------------
    succeeded = len(args.dates) - len(errors)
    print(
        f"\n✅  Done.  {succeeded}/{len(args.dates)} dates generated "
        f"successfully."
    )
    if errors:
        print(f"    Failed dates: {', '.join(errors)}")
        _die(f"Report generation completed with errors for {target_name}.")


def handle_report(args: argparse.Namespace) -> None:
    """Generate ATSPM performance reports for one or more dates.

    Validates cycle data for each requested date (running on-demand
    reprocessing when a date is absent), then invokes ``PlotGenerator``
    to produce the full suite of Plotly reports.

    Args:
        args: Parsed CLI arguments.
    """
    intersections_dir = _get_intersections_dir()
    
    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating reports for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _report_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error reporting {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# discrepancies
# ---------------------------------------------------------------------------

def _discrepancies_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to analyze discrepancies for a single intersection."""
    from atspm.data.detectors import get_detector_discrepancies

    target_dir = _get_target_dir(target_name)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)
    
    # Optional outputs dir
    output_dir = None
    if args.output:
        output_dir = target_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    timezone = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    try:
        start_dt = datetime.fromisoformat(args.start)
        end_dt   = datetime.fromisoformat(args.end)
    except ValueError as exc:
        _die(f"Invalid date format for --start or --end: {exc}. Use ISO format (e.g., 2024-06-01T06:00:00).")

    print(f"\n🔍  Analyzing detector discrepancies for {int_name}")
    print(f"    DB:     {db_path.name}")
    print(f"    Window: {start_dt.isoformat()} → {end_dt.isoformat()}")
    print(f"    Lag:    {args.lag}s")
    print(f"    TZ:     {timezone}")

    try:
        result = get_detector_discrepancies(
            db_path=db_path,
            start=start_dt,
            end=end_dt,
            lag_threshold_sec=args.lag,
            timezone=timezone,
            output_dir=output_dir,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Discrepancy analysis failed: {exc}")

    if result.empty:
        print("\n✅  No anomalies detected.")
        return # Changed from sys.exit to allow batch looping

    print(f"\n⚠️  Found {len(result)} anomaly(ies):\n")
    print(result.to_string(index=False))

    if output_dir:
        print(f"\nReport saved to {output_dir}/")


def handle_discrepancies(args: argparse.Namespace) -> None:
    """Analyze co-located detector discrepancies for a time window.

    Reads detector pair mappings from the active configuration
    (Det_Ph<X>_Pairs columns) and reports extended disagreements and
    unconfirmed pulses to stdout (and optionally a CSV file).

    Args:
        args: Parsed CLI arguments.
    """
    intersections_dir = _get_intersections_dir()
    
    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch analyzing discrepancies for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _discrepancies_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error analyzing {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()

# ---------------------------------------------------------------------------
# plot-coordination
# ---------------------------------------------------------------------------

def _plot_coordination_single_intersection(
    target_name: str,
    args: argparse.Namespace,
) -> None:
    """Core logic to generate a coordination plot for a specific window."""
    import pytz
    from atspm.reports.generators import PlotGenerator

    target_dir = _get_target_dir(target_name)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)

    if not db_path.exists():
        _die(f"Database not found: {db_path}\nRun 'atspm process --target {target_name}' first.")

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    tz_str = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE
    tz = pytz.timezone(tz_str)

    try:
        start_naive = datetime.fromisoformat(args.start)
        end_naive   = datetime.fromisoformat(args.end)
        start_dt = tz.localize(start_naive)
        end_dt   = tz.localize(end_naive)
    except ValueError as exc:
        _die(f"Invalid datetime format: {exc}. Use ISO-8601 (e.g. 2024-06-01T06:00:00).")

    print(f"\n📈  Generating coordination plot for {int_name}")
    print(f"    Window: {start_dt.isoformat()} → {end_dt.isoformat()}")

    # Output to a date folder based on the start time
    date_str = start_dt.strftime("%Y-%m-%d")
    date_dir = output_dir / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    try:
        gen = PlotGenerator(db_path, output_dir)
        gen._generate_coordination(
            date_str=date_str,
            start_dt=start_dt,
            end_dt=end_dt,
            metadata=gen._get_metadata(),
            date_dir=date_dir,
            tz_str=tz_str,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Plot generation failed: {exc}")


def handle_plot_coordination(args: argparse.Namespace) -> None:
    """Generate interactive coordination plots."""
    intersections_dir = _get_intersections_dir()
    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating coordination plots for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _plot_coordination_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error processing {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()

# ---------------------------------------------------------------------------
# plot-termination
# ---------------------------------------------------------------------------

def _plot_termination_single_intersection(
    target_name: str,
    args: argparse.Namespace,
) -> None:
    """Core logic to generate a termination plot for a specific window."""
    import pytz
    from atspm.reports.generators import PlotGenerator

    target_dir = _get_target_dir(target_name)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)

    if not db_path.exists():
        _die(f"Database not found: {db_path}\nRun 'atspm process --target {target_name}' first.")

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    tz_str = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE
    tz = pytz.timezone(tz_str)

    try:
        start_naive = datetime.fromisoformat(args.start)
        end_naive   = datetime.fromisoformat(args.end)
        start_dt = tz.localize(start_naive)
        end_dt   = tz.localize(end_naive)
    except ValueError as exc:
        _die(f"Invalid datetime format: {exc}. Use ISO-8601 (e.g. 2024-06-01T06:00:00).")

    print(f"\n📈  Generating termination plot for {int_name}")
    print(f"    Window: {start_dt.isoformat()} → {end_dt.isoformat()}")

    date_str = start_dt.strftime("%Y-%m-%d")
    date_dir = output_dir / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    try:
        gen = PlotGenerator(db_path, output_dir)
        gen._generate_termination(
            date_str=date_str,
            start_dt=start_dt,
            end_dt=end_dt,
            metadata=gen._get_metadata(),
            date_dir=date_dir,
            tz_str=tz_str,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Plot generation failed: {exc}")


def handle_plot_termination(args: argparse.Namespace) -> None:
    """Generate interactive termination plots."""
    intersections_dir = _get_intersections_dir()
    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating termination plots for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _plot_termination_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error processing {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()

# ---------------------------------------------------------------------------
# plot-detectors
# ---------------------------------------------------------------------------

def _plot_detectors_single_intersection(
    target_name: str,
    args: argparse.Namespace,
) -> None:
    """Core logic to generate a detector comparison plot for one intersection.

    Args:
        target_name: Exact intersection folder name.
        args: Parsed CLI arguments from the ``plot-detectors`` subcommand.
    """
    import pytz
    from atspm.reports.generators import PlotGenerator

    target_dir = _get_target_dir(target_name)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    tz_str = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE
    tz = pytz.timezone(tz_str)

    try:
        # Localise naive inputs immediately
        start_naive = datetime.fromisoformat(args.start)
        end_naive   = datetime.fromisoformat(args.end)
        start_dt = tz.localize(start_naive)
        end_dt   = tz.localize(end_naive)
    except ValueError as exc:
        _die(f"Invalid datetime format: {exc}. Use ISO-8601 (e.g. 2024-06-01T06:00:00).")

    print(f"\n📈  Generating detector comparison plot for {int_name}")
    print(f"    Window: {start_dt.isoformat()} → {end_dt.isoformat()}")
    if args.phases:
        print(f"    Phases: {args.phases}")

    try:
        gen = PlotGenerator(db_path, output_dir)
        gen._generate_detector_comparison(
            start_dt=start_dt,
            end_dt=end_dt,
            phases=args.phases,
            lag_threshold_sec=args.lag,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Plot generation failed: {exc}")


def handle_plot_detectors(args: argparse.Namespace) -> None:
    """Generate interactive detector comparison plots.

    Args:
        args: Parsed CLI arguments.
    """
    intersections_dir = _get_intersections_dir()
    
    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating detector plots for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _plot_detectors_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error processing {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# plot-timing-actuation
# ---------------------------------------------------------------------------

def _plot_timing_actuation_single_intersection(
    target_name: str,
    args: argparse.Namespace,
) -> None:
    """Core logic to generate a timing and actuation plot for one intersection.

    Args:
        target_name: Exact intersection folder name.
        args: Parsed CLI arguments from the ``plot-timing-actuation`` subcommand.
    """
    from atspm.data.timing_actuation import TimingActuationEngine

    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {target_name}' first."
        )

    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    tz_str = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE

    print(f"\n📈  Generating timing and actuation plot for {int_name}")
    print(f"    Window: {args.start} → {args.end}")
    if args.phases:
        print(f"    Phases: {args.phases}")
    if args.detectors:
        print(f"    Detectors: {args.detectors}")

    try:
        engine = TimingActuationEngine(db_path, timezone=tz_str)
        engine.plot(
            start=args.start,
            end=args.end,
            phases=args.phases,
            detectors=args.detectors,
            output_dir=output_dir,
        )
    except ValueError as exc:
        _die(str(exc))
    except Exception as exc:
        if getattr(args, "verbose", False):
            traceback.print_exc()
        _die(f"Plot generation failed: {exc}")


def handle_plot_timing_actuation(args: argparse.Namespace) -> None:
    """Generate interactive timing and actuation plots.

    Args:
        args: Parsed CLI arguments.
    """
    intersections_dir = _get_intersections_dir()

    if getattr(args, "all", False):
        targets = [p.name for p in intersections_dir.iterdir() if p.is_dir()]
        if not targets:
            _die(f"No intersection directories found in {intersections_dir}")
        print(f"\n🌍 Batch generating timing and actuation plots for {len(targets)} intersections...")
    else:
        targets = [_resolve_target_name(args.target, args.targetid)]

    for target_name in targets:
        try:
            _plot_timing_actuation_single_intersection(target_name, args)
        except SystemExit:
            print(f"\n⏭️ Skipping {target_name} due to errors.", file=sys.stderr)
        except Exception as exc:
            print(f"\n❌ Unexpected error processing {target_name}: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                traceback.print_exc()


# ---------------------------------------------------------------------------
# video-calibrate-shapes / video-overlay
#
# Both are single-target only -- no --all.  video-calibrate-shapes is an
# interactive, one-time-per-camera session; video-overlay's --video always
# names one specific file, which inherently belongs to one camera/
# intersection.  See docs/ROADMAP.md's Video Overlay planning entry.
# ---------------------------------------------------------------------------

def _video_shape_path(target_dir: Path, camera: str) -> Path:
    """Resolve the per-camera shape config path for an intersection.

    Args:
        target_dir: Absolute path to the intersection directory.
        camera: Camera name (used verbatim as a filename stem).

    Returns:
        ``intersections/<folder>/video/<camera>_shapes.csv``
    """
    return target_dir / "video" / f"{camera}_shapes.csv"


def _resolve_video_path(target_dir: Path, video_arg: str) -> Path:
    """Resolve a ``--video`` argument against the intersection's video directory.

    A bare filename (or relative path) is resolved against
    ``intersections/<folder>/video/`` -- the same directory the shape-config
    CSVs live in -- rather than the process's working directory. An absolute
    path is used as-is, as an escape hatch for videos stored elsewhere.

    Args:
        target_dir: Absolute path to the intersection directory.
        video_arg: The raw ``--video`` CLI argument.

    Returns:
        The resolved video file path.
    """
    video_path = Path(video_arg)
    return video_path if video_path.is_absolute() else target_dir / "video" / video_path


def handle_video_calibrate_shapes(args: argparse.Namespace) -> None:
    """Open the interactive shape-calibration tool for one camera.

    Args:
        args: Parsed CLI arguments from the ``video-calibrate-shapes``
            subcommand.
    """
    from atspm.video import calibrate_shapes
    from atspm.data.video import ShapeConfig

    target_name = _resolve_target_name(args.target, args.targetid)
    target_dir = _get_target_dir(target_name)
    shape_path = _video_shape_path(target_dir, args.camera)
    shape_path.parent.mkdir(parents=True, exist_ok=True)
    video_path = _resolve_video_path(target_dir, args.video)
    if not video_path.exists():
        _die(f"Video not found: {video_path}")

    existing = ShapeConfig.load(shape_path) if shape_path.exists() else None
    if existing:
        print(f"\n🎯  Editing existing shape config: {shape_path} ({len(existing.shapes)} shapes)")
    else:
        print(f"\n🎯  Creating new shape config: {shape_path}")
    print(f"    Video: {video_path}")

    try:
        config = calibrate_shapes(video_path, existing, save_path=shape_path)
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Calibration failed: {exc}")

    print(f"\n🏁  Calibration session ended ({len(config.shapes)} shapes in memory).")


def handle_video_overlay(args: argparse.Namespace) -> None:
    """Render a video with live phase/overlap/detector status overlays.

    Args:
        args: Parsed CLI arguments from the ``video-overlay`` subcommand.
    """
    import pytz
    from atspm.video import render_overlay
    from atspm.data.video import ShapeConfig

    target_name = _resolve_target_name(args.target, args.targetid)
    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    if not db_path.exists():
        _die(f"Database not found: {db_path}\nRun 'atspm process --target {target_name}' first.")

    shape_path = _video_shape_path(target_dir, args.camera)
    if not shape_path.exists():
        _die(
            f"Shape config not found: {shape_path}\n"
            f"Run 'atspm video-calibrate-shapes --target {target_name} "
            f"--camera {args.camera} --video <video>' first."
        )
    shape_config = ShapeConfig.load(shape_path)

    video_path = _resolve_video_path(target_dir, args.video)
    if not video_path.exists():
        _die(f"Video not found: {video_path}")

    tz_str = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE
    tz = pytz.timezone(tz_str)
    try:
        start_dt = tz.localize(datetime.fromisoformat(args.start))
    except ValueError as exc:
        _die(f"Invalid datetime format for --start: {exc}. Use ISO-8601 (e.g. 2024-06-01T06:00:00).")

    if args.output:
        output_path = Path(args.output)
    else:
        # Output to a date folder based on the start time, consistent with plotting outputs.
        # The filename includes --start's time-of-day (to a tenth of a second) so
        # re-renders after correcting --start don't silently overwrite each other.
        date_dir = target_dir / "outputs" / start_dt.strftime("%Y-%m-%d")
        time_str = start_dt.strftime("%H%M%S") + f".{start_dt.microsecond // 100000}"
        output_path = date_dir / f"{args.camera}_overlay_{time_str}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    print(f"\n🎬  Rendering video overlay for {int_name}")
    print(f"    Video:  {video_path}")
    print(f"    Start:  {start_dt.isoformat()}")
    print(f"    Output: {output_path}")

    try:
        result = render_overlay(
            db_path,
            shape_config,
            video_path,
            output_path,
            start_dt,
            lookback_minutes=args.lookback,
            lookahead_minutes=args.lookback,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Video overlay rendering failed: {exc}")

    print(f"\n✅  Wrote {result.frame_count} frames @ {result.fps:.2f} fps to {result.output_path}")
    if result.timing_source == "fps":
        print(
            "    ⚠️  This container reports no frame timestamps, so frames were timed at a "
            "constant\n        rate. Any dropped frames or stream stalls will skew overlays "
            "after the fact."
        )


_PHASE_LOCATE_SEARCH_HORIZON_SEC = 240.0  # generous upper bound on cycle length


def handle_video_locate_phase_change(args: argparse.Namespace) -> None:
    """Auto-select and locate a phase's exact color-change timestamp for alignment.

    By default, finds the first green->yellow or yellow->red change for
    ``--phase`` at or after ``--start`` + ``--min-offset`` (whichever edge
    comes first); pass ``--transition`` to pin it to one edge. See
    ``atspm.analysis.video.first_phase_transition_after`` for the event-code
    rationale and ``atspm.video.extract_labeled_clip`` for the confirmation
    clip's normalized countdown label. Call once without
    ``--observed-delta`` to get a labeled clip to watch; call again with
    ``--observed-delta <value read off the clip>`` to get the corrected
    ``--start`` for ``video-overlay`` -- no DB lookup math required, just
    add the signed value read off the screen to the original ``--start``.

    Args:
        args: Parsed CLI arguments from the ``video-locate-phase-change``
            subcommand.
    """
    import pytz
    from atspm.analysis.video import _TRANSITION_CODES, first_phase_transition_after
    from atspm.data.reader import get_events_with_cycles_df
    from atspm.utils.timezone import resolve_pytz
    from atspm.video import extract_labeled_clip

    target_name = _resolve_target_name(args.target, args.targetid)
    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    if not db_path.exists():
        _die(f"Database not found: {db_path}\nRun 'atspm process --target {target_name}' first.")

    video_path = _resolve_video_path(target_dir, args.video)
    if not video_path.exists():
        _die(f"Video not found: {video_path}")

    tz_str = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE
    tz = resolve_pytz(tz_str)
    try:
        start_dt = tz.localize(datetime.fromisoformat(args.start))
    except ValueError as exc:
        _die(f"Invalid datetime format for --start: {exc}. Use ISO-8601 (e.g. 2024-06-01T06:00:00).")

    start_epoch = start_dt.timestamp()
    after_ts = start_epoch + args.min_offset
    fetch_start = datetime.fromtimestamp(after_ts, tz=pytz.UTC)
    fetch_end = datetime.fromtimestamp(after_ts + _PHASE_LOCATE_SEARCH_HORIZON_SEC, tz=pytz.UTC)
    event_codes = (
        [_TRANSITION_CODES[args.transition]] if args.transition
        else list(_TRANSITION_CODES.values())
    )
    events_df = get_events_with_cycles_df(db_path, fetch_start, fetch_end, event_codes=event_codes)

    found = first_phase_transition_after(events_df, args.phase, after_ts, transition=args.transition)
    if found is None:
        _die(
            f"No phase {args.phase} transition found within "
            f"{_PHASE_LOCATE_SEARCH_HORIZON_SEC:.0f}s of --start + --min-offset.\n"
            f"Check --phase, or that this video's phase is actually active that early."
        )
    transition, actual_ts = found
    expected_offset = actual_ts - start_epoch

    actual_dt = datetime.fromtimestamp(actual_ts, tz=pytz.UTC).astimezone(tz)
    int_name = meta.get("intersection_name", target_name)
    print(f"\n🔎  Locating phase {args.phase} {transition} for {int_name}")
    print(f"    Nearest DB transition: {actual_dt.isoformat()}")
    print(f"    Expected at {expected_offset:.3f}s into the video (per your --start guess)")

    if args.observed_delta is not None:
        corrected_dt = start_dt + timedelta(seconds=args.observed_delta)
        print(f"\n✅  Corrected --start: {corrected_dt.isoformat()}")
        return

    date_dir = target_dir / "outputs" / start_dt.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    clip_path = date_dir / f"{args.camera}_locate_phase{args.phase}_{transition}.mp4"

    try:
        result = extract_labeled_clip(video_path, clip_path, expected_offset, window_sec=args.window)
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Clip extraction failed: {exc}")

    print(f"    Clip: {result.output_path} ({result.frame_count} frames @ {result.fps:.2f} fps)")
    print(
        "\nWatch the clip -- its on-screen counter reads +0.000s at the frame the "
        "transition should occur if your --start guess is exact, counting down to "
        "0 and going negative after. Read off the value at the instant the change "
        "actually, visually happens, then rerun this command with --observed-delta "
        "<that value, with its sign> to get the corrected --start "
        "(corrected = original --start + that value)."
    )


def handle_video_sync(args: argparse.Namespace) -> None:
    """Find corrected --start timestamp from signal lamp measurements.

    Args:
        args: Parsed CLI arguments from the ``video-sync`` subcommand.
    """
    from atspm.data.video import ShapeConfig, resolve_stopbar_target
    from atspm.utils.timezone import resolve_pytz
    from atspm.video.sync import sync_video

    target_name = _resolve_target_name(args.target, args.targetid)
    target_dir = _get_target_dir(target_name)
    meta = _load_metadata(target_dir)
    db_path = _resolve_db_path(target_dir, meta)

    if not db_path.exists():
        _die(f"Database not found: {db_path}\nRun 'atspm process --target {target_name}' first.")

    shape_path = _video_shape_path(target_dir, args.camera)
    if not shape_path.exists():
        _die(
            f"Shape config not found: {shape_path}\n"
            f"Run 'atspm video-calibrate-shapes --target {target_name} "
            f"--camera {args.camera} --video <video>' first."
        )
    shape_config = ShapeConfig.load(shape_path)

    video_path = _resolve_video_path(target_dir, args.video)
    if not video_path.exists():
        _die(f"Video not found: {video_path}")

    lamp_shapes = shape_config.lamp_shapes()
    if not lamp_shapes:
        _die(
            f"No lamp shapes found in {shape_path}.\n"
            f"Add at least one lamp shape (type=lamp, indication=green, a point on the lit lamp; "
            f"see 'atspm video-calibrate-shapes')."
        )

    tz_str = args.timezone or meta.get("timezone") or DEFAULT_TIMEZONE
    tz = resolve_pytz(tz_str)
    try:
        dt_parsed = datetime.fromisoformat(args.start_guess)
        if dt_parsed.tzinfo is None:
            start_guess_dt = tz.localize(dt_parsed)
        else:
            start_guess_dt = dt_parsed.astimezone(tz)
    except ValueError as exc:
        _die(f"Invalid datetime format for --start-guess: {exc}. Use ISO-8601 (e.g. 2026-10-01T12:25:00).")

    try:
        result = sync_video(
            db_path,
            shape_config,
            video_path,
            start_guess_dt,
            search_s=args.search,
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Video sync failed: {exc}")

    if result.accepted:
        corrected_dt = datetime.fromtimestamp(result.start_epoch, tz=tz)
        corrected_iso = corrected_dt.isoformat(timespec="milliseconds")
        mid_dt = datetime.fromtimestamp(result.mid_start_epoch, tz=tz)
        mid_iso = mid_dt.isoformat(timespec="milliseconds")
        delta_s = result.start_epoch - start_guess_dt.timestamp()

        print(f"\n✅  Synchronized video start for {args.camera} ({target_name})")
        print(f"    Corrected start:  {corrected_iso}")
        print(f"    Delta from guess: {delta_s:+.3f}s")
        print(f"    Mid-clip start:   {mid_iso}")
        print(f"    Slip:             {result.slip_s_per_10min:+.3f} s / 10 min")
        print(f"    Score:            {result.score:.3f} (runner-up: {result.runner_up_score:.3f})")
        print(f"    Agreement:        {result.agreement:.1%}")
        print(f"    Edges used:       {result.n_edges_used}")
        print(f"    Gap clamped:      {result.gap_clamped}")
        print(f"\nReady to render overlay:")
        print(f"    atspm video-overlay --target {target_name} --camera {args.camera} --video {args.video} --start {corrected_iso}\n")
    else:
        best_dt = datetime.fromtimestamp(result.start_epoch, tz=tz)
        best_iso = best_dt.isoformat(timespec="milliseconds")

        phase_arg = ""
        for lamp in lamp_shapes:
            kind, num = resolve_stopbar_target(lamp["phase"])
            if kind == "phase":
                phase_arg = f"--phase {num} "
                break

        print(f"\n❌  Video synchronization refused: {result.reason}")
        print(f"    Diagnostics:")
        print(f"      Score:            {result.score:.3f} (runner-up: {result.runner_up_score:.3f})")
        print(f"      Agreement:        {result.agreement:.1%}")
        print(f"      Edges used:       {result.n_edges_used}")
        print(f"      Frames compared:  {result.n_frames_compared}")
        print(f"      Gap clamped:      {result.gap_clamped}")
        print(f"\nManual fallback alignment:")
        print(f"    atspm video-locate-phase-change --target {target_name} --camera {args.camera} --video {args.video} {phase_arg}--start {best_iso}\n")
        sys.exit(2)


# ---------------------------------------------------------------------------
# Report helper shims (bridge between new anchor-based processor and the
# date-level stats / reprocess API that test_reporting.py relied on)
# ---------------------------------------------------------------------------

def _get_cycle_summary(processor, date_str: str) -> Optional[dict]:
    """Return cycle summary for a local calendar date, or None.

    Queries the cycles table directly rather than relying on a specific
    public method, making this robust against API changes in CycleProcessor.

    Args:
        processor: Initialised CycleProcessor.
        date_str:  Local date in ``YYYY-MM-DD`` format.

    Returns:
        Dict with ``cycle_count``, ``detection_method``, and
        ``coord_plan_range``; or ``None`` if no cycles exist.
    """
    from datetime import datetime, timedelta, time
    from atspm.data.manager import DatabaseManager

    try:
        local_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None

    tz = processor.tz
    start_epoch = tz.localize(datetime.combine(local_date, time.min)).timestamp()
    end_epoch = tz.localize(datetime.combine(local_date + timedelta(days=1), time.min)).timestamp()

    with DatabaseManager(processor.db_path) as m:
        cur = m.conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*), MIN(coord_plan), MAX(coord_plan), detection_method
            FROM cycles
            WHERE cycle_start >= ? AND cycle_start < ?
            GROUP BY detection_method
            ORDER BY COUNT(*) DESC
            LIMIT 1
            """,
            (start_epoch, end_epoch),
        )
        row = cur.fetchone()

    if not row or row[0] == 0:
        return None
    return {
        "date":             date_str,
        "cycle_count":      row[0],
        "coord_plan_range": (row[1], row[2]),
        "detection_method": row[3],
    }


def _reprocess_date(processor, date_str: str, timezone: Optional[str] = None) -> None:
    """Trigger on-demand cycle reprocessing for a single local date.

    Converts the local date to UTC epoch bounds and calls
    ``processor.process_span`` in Gap-Fill mode so it is safe to call even
    when the cycles table already has partial data for the date.

    Args:
        processor: Initialised CycleProcessor.
        date_str:  Local date in ``YYYY-MM-DD`` format.
        timezone:  IANA timezone string; falls back to processor's timezone.
    """
    from datetime import datetime, timedelta, time
    import pytz

    tz_name = timezone or str(processor.tz)
    tz      = pytz.timezone(tz_name)
    try:
        local_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return

    t_start = tz.localize(datetime.combine(local_date, time.min)).timestamp()
    t_end   = tz.localize(datetime.combine(local_date + timedelta(days=1), time.min)).timestamp()

    # Use gap-fill (Path B) so the repair is surgically bounded.
    processor.process_span(t_start, t_end, fill_gaps=True)


# ===========================================================================
# Argument parser construction
# ===========================================================================

def _add_setup_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``setup`` subcommand parser."""
    p_setup = subs.add_parser(
        "setup",
        help="Create a new intersection folder, metadata template, and config stub.",
        description=(
            "Scaffold a new intersection directory under intersections/<target>.\n\n"
            "The <target> name should follow the convention:\n"
            "  <numeric_id>_<RoadA>_and_<RoadB>   e.g. 2068_US-95_and_SH-8"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_setup.add_argument(
        "--target",
        required=True,
        metavar="FOLDER",
        help="Intersection folder name, e.g. '2068_US-95_and_SH-8'."
    )
    p_setup.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        metavar="TZ",
        help=f"IANA timezone for the new metadata.json (default: {DEFAULT_TIMEZONE})."
    )
    p_setup.set_defaults(func=handle_setup)


def _add_retrieve_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``retrieve`` subcommand parser."""
    p_retr = subs.add_parser(
        "retrieve",
        help="Pull new .datZ files from an intersection's configured devices via SCP.",
        description=(
            "Pull new .datZ files from each device listed in devices.json.\n\n"
            "Secondary devices (long-term storage, e.g. EVO radar) are always\n"
            "pulled before the controller (short FIFO retention window), so\n"
            "the controller's bookmark never advances ahead of data the\n"
            "secondary device hasn't reported yet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_retr = p_retr.add_mutually_exclusive_group(required=True)
    group_retr.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_retr.add_argument("--targetid", metavar="ID", help="Intersection ID (prefix of folder name).")
    group_retr.add_argument("--all", action="store_true", help="Retrieve for all intersections in the directory.")
    p_retr.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tracebacks for unexpected per-intersection errors during --all."
    )
    p_retr.set_defaults(func=handle_retrieve)


def _add_process_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``process`` subcommand parser."""
    p_proc = subs.add_parser(
        "process",
        help="Ingest .datZ files and compute signal cycles.",
        description=(
            "Ingest raw .datZ data for an intersection and compute cycles.\n\n"
            "PATH A – Fast Append (default):\n"
            "  Only files newer than the last ingested span are scanned.\n"
            "  Cycles are recalculated forward from the last known anchor.\n\n"
            "PATH B – Gap Fill (--fill-gaps):\n"
            "  All files are scanned; historical gaps are filled.\n"
            "  Obsolete gap markers are scrubbed; cycles are surgically repaired.\n\n"
            "PATH C – Rebuild (--rebuild):\n"
            "  events, cycles and ingestion_log are deleted, then every .datZ\n"
            "  file is re-ingested from scratch. Config and metadata survive.\n"
            "  Use when stored timestamps need to be re-derived on the current\n"
            "  decoder basis. Destructive — prompts unless --yes is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_proc = p_proc.add_mutually_exclusive_group(required=True)
    group_proc.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_proc.add_argument("--targetid", metavar="ID", help="Intersection ID (prefix of folder name).")
    group_proc.add_argument("--all", action="store_true", help="Process all intersections in the directory.")
    mode_proc = p_proc.add_mutually_exclusive_group()
    mode_proc.add_argument(
        "--fill-gaps",
        action="store_true",
        default=False,
        help=(
            "Enable Gap Fill mode (Path B): scan for and ingest historical gaps; "
            "scrub obsolete gap markers; surgically repair affected cycles."
        ),
    )
    mode_proc.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help=(
            "Enable Rebuild mode (Path C): DELETE events, cycles and "
            "ingestion_log, then re-ingest every .datZ file from raw_data/. "
            "Config and metadata are preserved."
        ),
    )
    p_proc.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Skip the --rebuild confirmation prompt (for scripted runs).",
    )
    p_proc.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Number of .datZ files per transaction commit (default: 50)."
    )
    p_proc.add_argument(
        "--no-cycles",
        action="store_true",
        default=False,
        help="Skip cycle processing; only ingest raw events."
    )
    p_proc.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help=(
            "Override the timezone from metadata.json "
            "(e.g. 'US/Pacific').  Useful for one-off corrections."
        ),
    )
    p_proc.set_defaults(func=handle_process)


def _add_report_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``report`` subcommand parser."""
    p_rep = subs.add_parser(
        "report",
        help="Generate ATSPM performance reports for one or more dates.",
        description=(
            "Validate cycle data and generate the full suite of ATSPM Plotly\n"
            "reports for the specified intersection and dates.\n\n"
            "Output files are written to:\n"
            "  intersections/<target>/outputs/<date>/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_rep = p_rep.add_mutually_exclusive_group(required=True)
    group_rep.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_rep.add_argument("--targetid", metavar="ID", help="Intersection ID (prefix of folder name).")
    group_rep.add_argument("--all", action="store_true", help="Generate reports for all intersections in the directory.")
    p_rep.add_argument(
        "--dates",
        required=True,
        nargs="+",
        metavar="YYYY-MM-DD",
        help=(
            "One or more local calendar dates to report on, "
            "e.g. --dates 2026-02-19 2026-02-20"
        ),
    )
    p_rep.add_argument(
        "--backfill",
        action="store_true",
        default=False,
        help=(
            "Run backfill_ring_phases() before generating reports. "
            "Safe to use repeatedly; fast when already complete."
        ),
    )
    p_rep.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any per-date generation errors."
    )
    p_rep.set_defaults(func=handle_report)


def _add_counts_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``counts`` subcommand parser."""
    p_counts = subs.add_parser(
        "counts",
        help="Generate vehicle and pedestrian counts.",
        description=(
            "Generates binned or per-cycle volume counts to CSV.\n\n"
            "Outputs are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_counts = p_counts.add_mutually_exclusive_group(required=True)
    group_counts.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_counts.add_argument("--targetid", metavar="ID", help="Intersection ID (prefix of folder name).")
    group_counts.add_argument("--all", action="store_true", help="Generate counts for all intersections in the directory.")
    
    p_counts.add_argument("--start", required=True, metavar="YYYY-MM-DD", help="Query window start (local time).")
    p_counts.add_argument("--end", required=True, metavar="YYYY-MM-DD", help="Query window end (local time).")
    p_counts.add_argument("--bin-len", default="60", metavar="N", help="Aggregation interval in minutes, or 'cycle' (default: 60).")
    p_counts.add_argument("--type", choices=["vehicle", "ped", "combined"], default="combined", help="Type of counts to run (default: combined).")
    p_counts.add_argument("--hourly", action="store_true", help="Scale numeric bins to hourly flow rate.")
    p_counts.add_argument("--include-detectors", action="store_true", help="Include raw per-detector count columns.")
    p_counts.add_argument("--exclude-missing", action="store_true", help="Drop partial and missing bins from the output.")
    p_counts.add_argument("--timezone", default=None, metavar="TZ", help="Override the timezone from metadata.json.")
    p_counts.add_argument("--verbose", action="store_true", help="Print full tracebacks for any errors.")
    p_counts.set_defaults(func=handle_counts)


def _add_splits_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``splits`` subcommand parser."""
    p_splits = subs.add_parser(
        "splits",
        help="Generate phase split and timing records.",
        description=(
            "Generates binned or per-cycle phase timing splits to CSV.\n\n"
            "Outputs are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_splits = p_splits.add_mutually_exclusive_group(required=True)
    group_splits.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_splits.add_argument("--targetid", metavar="ID", help="Intersection ID (prefix of folder name).")
    group_splits.add_argument("--all", action="store_true", help="Generate splits for all intersections in the directory.")
    
    p_splits.add_argument("--start", required=True, metavar="YYYY-MM-DD", help="Query window start (local time).")
    p_splits.add_argument("--end", required=True, metavar="YYYY-MM-DD", help="Query window end (local time).")
    p_splits.add_argument("--bin-len", default="cycle", metavar="N", help="Aggregation interval in minutes, or 'cycle' (default: cycle).")
    p_splits.add_argument("--report-mode", choices=["seconds", "total", "proportion"], default="seconds", help="How binned values are expressed (default: seconds).")
    p_splits.add_argument("--phases", nargs="+", type=int, metavar="N", default=None, help="Filter to specific phase IDs.")
    p_splits.add_argument("--include-no-clearance", action="store_true", help="Include phases with no yellow logged as green-only intervals.")
    p_splits.add_argument("--exclude-missing", action="store_true", help="Drop partial and missing bins from the output.")
    p_splits.add_argument("--timezone", default=None, metavar="TZ", help="Override the timezone from metadata.json.")
    p_splits.add_argument("--verbose", action="store_true", help="Print full tracebacks for any errors.")
    p_splits.set_defaults(func=handle_splits)


def _add_aog_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``aog`` subcommand parser."""
    p_aog = subs.add_parser(
        "aog",
        help="Generate Arrival on Green (AOG) tables.",
        description=(
            "Compute per-cycle or time-binned Arrival on Green for one or\n"
            "more signal phases.  Advance detector IDs are read from the\n"
            "active configuration (Det_P{N}_Arrival keys in int_cfg.csv).\n\n"
            "Outputs are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_aog = p_aog.add_mutually_exclusive_group(required=True)
    group_aog.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_aog.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_aog.add_argument(
        "--all",
        action="store_true",
        help="Generate AOG for all intersections in the directory.",
    )
    p_aog.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DD",
        help="Query window start date (local time, inclusive).",
    )
    p_aog.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM-DD",
        help="Query window end date (local time, inclusive).",
    )
    p_aog.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Signal phase numbers to analyse, e.g. --phases 2 6. "
            "Omit to analyse all phases with a configured Det_P{N}_Arrival key."
        ),
    )
    p_aog.add_argument(
        "--offset",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "Arrival offset in seconds: added to each detector timestamp "
            "before evaluating green-window containment. "
            "Use this to account for travel time from an advance detector "
            "to the stop bar (default: 0.0)."
        ),
    )
    p_aog.add_argument(
        "--bin-len",
        default="60",
        metavar="N",
        help=(
            "Aggregation interval in minutes, or 'cycle' for one row per "
            "detected cycle (default: 60)."
        ),
    )
    p_aog.add_argument(
        "--exclude-missing",
        action="store_true",
        help=(
            "Drop bins labeled 'partial' or 'missing' from binned output. "
            "Full-day-missing days are always dropped regardless of this flag. "
            "Ignored in cycle mode."
        ),
    )
    p_aog.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_aog.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_aog.set_defaults(func=handle_aog)


def _add_approach_delay_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``approach-delay`` subcommand parser."""
    p_ad = subs.add_parser(
        "approach-delay",
        help="Generate approach delay and Arrival on Red tables and plots.",
        description=(
            "Calculate per-cycle and binned approach delay and arrival shares (AoG/AoY/AoR)\n"
            "for advance detector arrivals. Detector IDs and travel times are read from\n"
            "active configuration ('P{N} Arrival' and 'P{N} Arrival Travel' keys).\n\n"
            "Outputs (CSV + interactive HTML) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_ad = p_ad.add_mutually_exclusive_group(required=True)
    group_ad.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_ad.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_ad.add_argument(
        "--all",
        action="store_true",
        help="Generate approach delay for all intersections in the directory.",
    )
    p_ad.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_ad.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_ad.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Signal phase numbers to analyse, e.g. --phases 2 6. Omit to analyse all configured phases.",
    )
    p_ad.add_argument(
        "--offset",
        type=float,
        default=0.0,
        metavar="SEC",
        help="travel time from the advance detector to the stop line, used for phases without a Det_P{N}_Arrival_Travel key",
    )
    p_ad.add_argument(
        "--bin-len",
        default="15",
        metavar="N",
        help=(
            "Aggregation interval in minutes, or 'cycle' for one row per "
            "detected cycle (default: 15)."
        ),
    )
    p_ad.add_argument(
        "--exclude-missing",
        action="store_true",
        default=False,
        help=(
            "Drop bins labeled 'partial' or 'missing' from binned output. "
            "Full-day-missing days are always dropped regardless of this flag. "
            "Ignored in cycle mode."
        ),
    )
    p_ad.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        default=False,
        help="Disable interactive HTML plot generation.",
    )
    p_ad.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_ad.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_ad.set_defaults(func=handle_approach_delay)


def _add_yellow_red_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``yellow-red`` subcommand parser."""
    p_yr = subs.add_parser(
        "yellow-red",
        help="Generate yellow and red actuation tables and plots.",
        description=(
            "Calculate per-cycle, binned, and per-plan yellow and red actuation counts\n"
            "and violations (UDOT S-M4) for stop-bar or occupancy detectors.\n\n"
            "Outputs (CSV + interactive HTML) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_yr = p_yr.add_mutually_exclusive_group(required=True)
    group_yr.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_yr.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_yr.add_argument(
        "--all",
        action="store_true",
        help="Generate yellow and red actuations for all intersections in the directory.",
    )
    p_yr.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_yr.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_yr.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Signal phase numbers to analyse, e.g. --phases 2 6. Omit to analyse all configured phases.",
    )
    p_yr.add_argument(
        "--role",
        choices=["stop_bar", "occupancy"],
        default="stop_bar",
        help="Detector role to classify ('stop_bar' or 'occupancy', default: 'stop_bar').",
    )
    p_yr.add_argument(
        "--severe-sec",
        type=float,
        default=4.0,
        metavar="S",
        help="Severe violation threshold in seconds after red start (default: 4.0).",
    )
    p_yr.add_argument(
        "--bin-len",
        type=int,
        default=15,
        metavar="M",
        help="Aggregation interval in minutes (default: 15).",
    )
    p_yr.add_argument(
        "--no-exclusions",
        action="store_true",
        default=False,
        help="Ignore TM_Exclusions configured in int_cfg.csv.",
    )
    p_yr.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        default=False,
        help="Disable interactive HTML plot generation.",
    )
    p_yr.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_yr.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_yr.set_defaults(func=handle_yellow_red)


def _add_ped_delay_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``ped-delay`` subcommand parser."""
    p_pd = subs.add_parser(
        "ped-delay",
        help="Generate pedestrian delay tables and plots.",
        description=(
            "Calculate per-walk, binned, and per-plan pedestrian delay\n"
            "(UDOT S-M5).\n\n"
            "Outputs (CSV + interactive HTML) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p_pd.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Generate pedestrian delay for all intersections in the directory.",
    )
    p_pd.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_pd.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_pd.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Signal phase numbers to analyse, e.g. --phases 2 4. Omit to analyse all configured phases.",
    )
    p_pd.add_argument(
        "--bin-len",
        type=int,
        default=60,
        metavar="MINUTES",
        help="Summary aggregation interval in minutes (default: 60).",
    )
    p_pd.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        default=False,
        help="Disable interactive HTML plot generation.",
    )
    p_pd.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_pd.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_pd.set_defaults(func=handle_ped_delay)


def _add_wait_time_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``wait-time`` subcommand parser."""
    p_wt = subs.add_parser(
        "wait-time",
        help="Generate vehicle wait time tables and plots.",
        description=(
            "Calculate per-window, binned, and per-plan vehicle wait time\n"
            "(UDOT S-M5).\n\n"
            "Outputs (CSV + interactive HTML) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p_wt.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Generate wait time for all intersections in the directory.",
    )
    p_wt.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_wt.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_wt.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Signal phase numbers to analyse, e.g. --phases 2 4. Omit to analyse all configured phases.",
    )
    p_wt.add_argument(
        "--dropping",
        choices=["auto", "on", "off"],
        default="auto",
        help="Phases using UDOT's dropping algorithm ('auto', 'on', or 'off', default: 'auto').",
    )
    p_wt.add_argument(
        "--max-wait",
        type=float,
        default=360.0,
        metavar="SEC",
        help="Maximum wait time in seconds for summary averages (default: 360.0, 0 means no cap).",
    )
    p_wt.add_argument(
        "--bin-len",
        type=int,
        default=15,
        metavar="MINUTES",
        help="Summary aggregation interval in minutes (default: 15).",
    )
    p_wt.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        default=False,
        help="Disable interactive HTML plot generation.",
    )
    p_wt.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_wt.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_wt.set_defaults(func=handle_wait_time)


def _add_split_monitor_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``split-monitor`` subcommand parser."""
    p_sm = subs.add_parser(
        "split-monitor",
        help="Generate split monitor tables and plots.",
        description=(
            "Generate UDOT split monitor tables and plots (cycle services, per-plan\n"
            "statistics, and timeline) for configured phases.\n\n"
            "Outputs (CSV + interactive HTML) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_sm = p_sm.add_mutually_exclusive_group(required=True)
    group_sm.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_sm.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_sm.add_argument(
        "--all",
        action="store_true",
        help="Generate split monitor for all intersections in the directory.",
    )
    p_sm.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_sm.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_sm.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Signal phase numbers to analyse, e.g. --phases 2 6. Omit to analyse all phases.",
    )
    p_sm.add_argument(
        "--percentiles",
        nargs=2,
        type=float,
        default=[50.0, 85.0],
        metavar=("A", "B"),
        help="Two split percentiles to report in stats (default: 50.0 85.0).",
    )
    p_sm.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        default=False,
        help="Disable interactive HTML plot generation.",
    )
    p_sm.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_sm.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_sm.set_defaults(func=handle_split_monitor)


def _add_detector_health_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``detector-health`` subcommand parser."""
    p_dh = subs.add_parser(
        "detector-health",
        help="Evaluate detector health rules and generate heatmap and findings.",
        description=(
            "Run deterministic detector-health rules over raw events and activity profiles.\n"
            "Records findings into the detector_findings table and exports CSV and HTML\n"
            "heatmaps to intersections/<target>/outputs/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_dh = p_dh.add_mutually_exclusive_group(required=True)
    group_dh.add_argument(
        "--target",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_dh.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_dh.add_argument(
        "--all",
        action="store_true",
        help="Run detector health for all intersections in the directory.",
    )
    p_dh.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DD",
        help="Query window start date (local time, inclusive).",
    )
    p_dh.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Query window end date (local time, inclusive; defaults to --start).",
    )
    p_dh.add_argument(
        "--window",
        choices=["am", "pm", "day"],
        default="day",
        help="Window to filter and report (am, pm, day; default: day).",
    )
    p_dh.add_argument(
        "--min-severity",
        choices=["info", "low", "high"],
        default="low",
        help="Minimum severity threshold to report (info, low, high; default: low).",
    )
    p_dh.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_dh.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_dh.set_defaults(func=handle_detector_health)


def _add_split_failures_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``split-failures`` subcommand parser."""
    p_sf = subs.add_parser(
        "split-failures",
        help="Generate Purdue split-failure tables and scatter plots.",
        description=(
            "Calculate Purdue split failures (GOR vs ROR5) per phase split window.\n"
            "Presence detector IDs (zones at the stop line, one per lane) are\n"
            "read from the active configuration ('P{N} Occupancy' rows in\n"
            "int_cfg.csv → Det_P{N}_Occupancy). Stop Bar channels are not used.\n\n"
            "Outputs (CSV + interactive HTML) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_sf = p_sf.add_mutually_exclusive_group(required=True)
    group_sf.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_sf.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_sf.add_argument(
        "--all",
        action="store_true",
        help="Generate split failures for all intersections in the directory.",
    )
    p_sf.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_sf.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_sf.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Signal phase numbers to analyse, e.g. --phases 2 6. Omit to analyse all configured phases.",
    )
    p_sf.add_argument(
        "--aggregate",
        choices=["union", "mean", "any"],
        default="union",
        help=(
            "Lane aggregation method (default: union). "
            "union = occupied when any lane is on (UDOT, like one multi-lane detector); "
            "mean = average of per-lane GOR/ROR5; "
            "any = fails when any lane fails on its own, reporting the worst lane's GOR/ROR5."
        ),
    )
    p_sf.add_argument(
        "--threshold",
        type=float,
        default=0.79,
        metavar="FRAC",
        help="Occupancy threshold above which a cycle fails (default: 0.79).",
    )
    p_sf.add_argument(
        "--ror-seconds",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Red occupancy window length in seconds from yellow end (default: 5.0).",
    )
    p_sf.add_argument(
        "--include-yellow",
        action="store_true",
        help="Measure GOR over green + yellow (SPMs definition) instead of green only.",
    )
    p_sf.add_argument(
        "--bin-len",
        default="60",
        metavar="N",
        help=(
            "Aggregation interval in minutes, or 'cycle' for one row per "
            "detected cycle (default: 60)."
        ),
    )
    p_sf.add_argument(
        "--exclude-missing",
        action="store_true",
        help=(
            "Drop bins labeled 'partial' or 'missing' from binned output. "
            "Full-day-missing days are always dropped regardless of this flag. "
            "Ignored in cycle mode."
        ),
    )
    p_sf.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        default=False,
        help="Disable interactive HTML scatter plot generation.",
    )
    p_sf.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_sf.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_sf.set_defaults(func=handle_split_failures)


def _add_infer_detectors_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``infer-detectors`` subcommand parser."""
    p_inf = subs.add_parser(
        "infer-detectors",
        help="Propose detector configuration for review (never edits int_cfg.csv).",
        description=(
            "Propose a detector configuration for review from actuation behaviour.\n"
            "Never edits int_cfg.csv or the config table.\n\n"
            "Outputs are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_inf = p_inf.add_mutually_exclusive_group(required=True)
    group_inf.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_inf.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_inf.add_argument(
        "--all",
        action="store_true",
        help="Infer detector configuration for all intersections in the directory.",
    )
    p_inf.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DD",
        help="Query window start date (local time, inclusive).",
    )
    p_inf.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM-DD",
        help="Query window end date (local time, inclusive).",
    )
    p_inf.add_argument(
        "--min-actuations",
        type=int,
        default=50,
        metavar="N",
        help="Minimum uncensored on-intervals to classify a detector (default: 50).",
    )
    p_inf.add_argument(
        "--all-phases",
        action="store_true",
        default=False,
        help="Do not limit candidates to the RB_* ring phases.",
    )
    p_inf.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_inf.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_inf.set_defaults(func=handle_infer_detectors)


def _add_flow_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``flow`` subcommand parser."""
    p_flow = subs.add_parser(
        "flow",
        help="Generate split flow-rate tables and plots.",
        description=(
            "Compute effective cumulative flow-rate profiles from stop-bar\n"
            "detector departures (Code 81) within each phase split window.\n"
            "Stop-bar detector IDs are read from the active configuration\n"
            "(P{N} Stop Bar rows: Det_P{N}_Stop_Bar; Det_P{N}_Stopbar also accepted).\n\n"
            "Only near-capacity cycles qualify (end slack <= --max-lost);\n"
            "cycles are then restricted to the modal split and the busiest\n"
            "--pct percent.  The peak of the mean profile identifies the\n"
            "throughput-optimal split length.\n\n"
            "Outputs (CSV + interactive HTML) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_flow = p_flow.add_mutually_exclusive_group(required=True)
    group_flow.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_flow.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_flow.add_argument(
        "--all",
        action="store_true",
        help="Generate flow rate for all intersections in the directory.",
    )
    p_flow.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DD",
        help="Query window start date (local time, inclusive).",
    )
    p_flow.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM-DD",
        help="Query window end date (local time, inclusive).",
    )
    p_flow.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Signal phase numbers to analyse, e.g. --phases 2 6. "
            "Omit to analyse all phases with configured P{N} Stop Bar rows "
            "(Det_P{N}_Stop_Bar; Det_P{N}_Stopbar also accepted)."
        ),
    )
    p_flow.add_argument(
        "--plans",
        nargs="+",
        type=int,
        metavar="P",
        default=None,
        help=(
            "Coordination plan numbers to include, e.g. --plans 1 2. "
            "Omit to analyse all plans."
        ),
    )
    p_flow.add_argument(
        "--pct",
        type=float,
        default=1.0,
        metavar="PCT",
        help=(
            "Keep the busiest PCT percent of modal-split cycles by total "
            "vehicles (default: 1.0)."
        ),
    )
    p_flow.add_argument(
        "--max-lost",
        type=float,
        default=10.0,
        metavar="SEC",
        help=(
            "Maximum seconds between the last detector departure and the "
            "end of the split window for a cycle to qualify as "
            "near-capacity (default: 10.0)."
        ),
    )
    p_flow.add_argument(
        "--split-tolerance",
        type=float,
        default=0.10,
        metavar="FRAC",
        help=(
            "Fractional tolerance around the modal split length "
            "(default: 0.10 = ±10%%)."
        ),
    )
    p_flow.add_argument(
        "--stratify",
        action="store_true",
        help=(
            "Keep the busiest --pct percent within each (plan, split) "
            "stratum and pool them, instead of filtering around the modal "
            "split.  Keeps shorter-split plans in the profile."
        ),
    )
    p_flow.add_argument(
        "--normalize",
        choices=["end_shift", "pooled", "clearance", "fixed", "none"],
        default="end_shift",
        help=(
            "Split-termination overhead added to elapsed time when computing "
            "the effective cumulative rate: 'end_shift' = each cycle's "
            "measured end slack (default), 'pooled' = per-detector median "
            "slack, 'clearance' = actual yellow+red clearance duration, "
            "'fixed' = constant --fixed-lost seconds, 'none' = raw rate."
        ),
    )
    p_flow.add_argument(
        "--fixed-lost",
        type=float,
        default=None,
        metavar="SEC",
        help="Constant overhead in seconds (required with --normalize fixed).",
    )
    p_flow.add_argument(
        "--rolling",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Centred rolling-mean window (grid rows) applied to the "
            "instantaneous-rate traces in the plot (default: 5; 1 disables)."
        ),
    )
    p_flow.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip HTML plot generation; write CSV tables only.",
    )
    p_flow.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_flow.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_flow.set_defaults(func=handle_flow)


def _add_critical_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``critical`` subcommand parser."""
    p_crit = subs.add_parser(
        "critical",
        help="Run critical movement analysis.",
        description=(
            "Identify the critical phases and the critical path per barrier\n"
            "group for a chosen period.  The ring/barrier structure comes\n"
            "from RB_R1/RB_R2 config (NEMA-standard fallback), cross-checked\n"
            "against observed cycle sequences; movement counts (TM_* keys)\n"
            "are mapped to phases by stop-bar detector overlap\n"
            "(P{N} Stop Bar rows: Det_P{N}_Stop_Bar; Det_P{N}_Stopbar also accepted).\n\n"
            "Demand (vph, or vphpl with --basis per_lane) is the\n"
            "required-time proxy: per barrier group, the ring with the\n"
            "larger demand sum is the critical path.\n\n"
            "Outputs (CSV) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_crit = p_crit.add_mutually_exclusive_group(required=True)
    group_crit.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_crit.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_crit.add_argument(
        "--all",
        action="store_true",
        help="Run the analysis for all intersections in the directory.",
    )
    p_crit.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_crit.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_crit.add_argument(
        "--bin-len",
        type=int,
        default=15,
        metavar="N",
        help="Demand-aggregation bin width in minutes (default: 15).",
    )
    p_crit.add_argument(
        "--basis",
        choices=["per_lane", "total"],
        default="per_lane",
        help=(
            "Demand basis for criticality: 'per_lane' = vph per detector "
            "(lane-count proxy; default), 'total' = raw vph."
        ),
    )
    p_crit.add_argument(
        "--include-missing",
        action="store_true",
        help=(
            "Keep bins with partial/missing data when averaging demand "
            "(by default only quality-'ok' bins are used, since zero-filled "
            "missing bins bias mean demand downward)."
        ),
    )
    p_crit.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_crit.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_crit.set_defaults(func=handle_critical)


def _add_optimize_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``optimize`` subcommand parser."""
    p_opt = subs.add_parser(
        "optimize",
        help="Optimize cycle length and splits for saturated throughput.",
        description=(
            "Picks the cycle length C and splits that maximize saturated\n"
            "throughput Σ 3600·N_p(s_p) / C over the saturated phases, using\n"
            "measured cumulative discharge curves. Saturated phases are the\n"
            "engineer's declaration (--saturated); the end-slack classifier is\n"
            "printed as an advisory only.\n"
            "--validate tests the throughput model against the existing TOD plans\n"
            "before its recommendations are trusted.\n\n"
            "Outputs (CSV and interactive HTML plots) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_opt = p_opt.add_mutually_exclusive_group(required=True)
    group_opt.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_opt.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_opt.add_argument(
        "--all",
        action="store_true",
        help="Run the optimization for all intersections in the directory.",
    )
    p_opt.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help="Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'.",
    )
    p_opt.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help="Period end (local time): 'YYYY-MM-DD' (inclusive) or 'YYYY-MM-DD HH:MM' (exclusive).",
    )
    p_opt.add_argument(
        "--saturated",
        required=True,
        type=int,
        nargs="+",
        metavar="N",
        help="Declared saturated phase numbers (required).",
    )
    p_opt.add_argument(
        "--plans",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Optional coordination plan IDs to filter cycles.",
    )
    p_opt.add_argument(
        "--pct",
        type=float,
        default=1.0,
        metavar="PCT",
        help="Percentage of the busiest modal-split cycles to keep (default: 1.0 = top 1%%; 100 = all).",
    )
    p_opt.add_argument(
        "--split-tolerance",
        type=float,
        default=0.10,
        metavar="TOL",
        help="Split duration tolerance around target percentile (default: 0.10).",
    )
    p_opt.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify discharge profiles by coordination plan.",
    )
    p_opt.add_argument(
        "--max-lost",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Per-lane end-slack limit in seconds for advisory saturation (default: 10.0).",
    )
    p_opt.add_argument(
        "--sat-threshold",
        type=float,
        default=0.8,
        metavar="FRAC",
        help="Threshold pass rate for advisory saturation (default: 0.8).",
    )
    p_opt.add_argument(
        "--demand-stat",
        choices=["mean", "peak"],
        default="mean",
        help="Statistic used for unsaturated phase demand (default: 'mean').",
    )
    p_opt.add_argument(
        "--default-min-split",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Fallback minimum split in seconds (default: 10.0).",
    )
    p_opt.add_argument(
        "--c-min",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Shortest cycle scanned in seconds (default: 60.0).",
    )
    p_opt.add_argument(
        "--c-max",
        type=float,
        default=220.0,
        metavar="SEC",
        help="Longest cycle scanned in seconds (default: 220.0).",
    )
    p_opt.add_argument(
        "--c-step",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Scan step in seconds (default: 1.0).",
    )
    p_opt.add_argument(
        "--flat-tol-pct",
        type=float,
        default=1.0,
        metavar="PCT",
        help="Flat-band tolerance percent of peak throughput (default: 1.0).",
    )
    p_opt.add_argument(
        "--bin-len",
        type=int,
        default=15,
        metavar="MIN",
        help="Demand aggregation bin width in minutes (default: 15).",
    )
    p_opt.add_argument(
        "--include-missing",
        action="store_true",
        help="Include bins with partial/missing count data when averaging demand.",
    )
    p_opt.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating HTML plot files.",
    )
    p_opt.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_opt.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Run model validation instead of the optimizer.",
    )
    p_opt.add_argument(
        "--min-plan-cycles",
        type=int,
        default=30,
        metavar="N",
        help="Minimum complete cycles for a plan to be tested (applies to --validate only, default: 30).",
    )
    p_opt.add_argument(
        "--split-cover-tol",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Split cover tolerance in seconds (applies to --validate only, default: 1.0).",
    )
    p_opt.add_argument(
        "--rank-deadband-pct",
        type=float,
        default=2.0,
        metavar="PCT",
        help="Ranking deadband percent (applies to --validate only, default: 2.0).",
    )
    p_opt.add_argument(
        "--change-tol-pp",
        type=float,
        default=3.0,
        metavar="PP",
        help="Magnitude tolerance in percentage points (applies to --validate only, default: 3.0).",
    )
    p_opt.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_opt.set_defaults(func=handle_optimize)


def _add_clock_drift_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``clock-drift`` subcommand parser."""
    p_clk = subs.add_parser(
        "clock-drift",
        help="Decode clock marks and plot controller clock drift.",
        description=(
            "Decode pedestrian-call clock marks from the controller log,\n"
            "measure controller clock drift against the host clock,\n"
            "and identify clock correction sets.\n\n"
            "Outputs (CSV and HTML plot) are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_clk = p_clk.add_mutually_exclusive_group(required=True)
    group_clk.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group_clk.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_clk.add_argument(
        "--all",
        action="store_true",
        help="Run the analysis for all intersections in the directory.",
    )
    p_clk.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_clk.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_clk.add_argument(
        "--send-log",
        dest="send_log",
        default=None,
        metavar="PATH",
        help="The head unit's eos-time.jsonl send log file.",
    )
    p_clk.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_clk.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_clk.set_defaults(func=handle_clock_drift)


def _add_preempt_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``preempt`` subcommand parser."""
    p_preempt = subs.add_parser(
        "preempt",
        help="Analyze preemption episodes and write episode and summary tables.",
        description=(
            "Analyze preemption episodes from controller events and write\n"
            "episode and summary CSV tables.\n\n"
            "Outputs are saved to:\n"
            "  intersections/<target>/outputs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p_preempt.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name (e.g. '2068_US-95_and_SH-8').",
    )
    group.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run the analysis for all intersections in the directory.",
    )
    p_preempt.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help=(
            "Period start (local time): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' "
            "for sub-day peak periods."
        ),
    )
    p_preempt.add_argument(
        "--end",
        required=True,
        metavar="DATETIME",
        help=(
            "Period end (local time): 'YYYY-MM-DD' (inclusive whole day) or "
            "'YYYY-MM-DD HH:MM' (exclusive)."
        ),
    )
    p_preempt.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json (e.g. 'US/Pacific').",
    )
    p_preempt.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_preempt.set_defaults(func=handle_preempt)


def _add_plot_coordination_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``plot-coordination`` subcommand parser."""
    p_coord = subs.add_parser(
        "plot-coordination",
        help="Generate a coordination / split diagram for a specific time window.",
        description="Generates an interactive coordination plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_coord = p_coord.add_mutually_exclusive_group(required=True)
    group_coord.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_coord.add_argument("--targetid", metavar="ID", help="Intersection ID prefix.")
    group_coord.add_argument("--all", action="store_true", help="Generate plots for all intersections in the directory.")
    p_coord.add_argument("--start", required=True, metavar="ISO8601", help="Window start (local time, ISO-8601).")
    p_coord.add_argument("--end", required=True, metavar="ISO8601", help="Window end, exclusive (local time, ISO-8601).")
    p_coord.add_argument("--timezone", default=None, metavar="TZ", help="Override the timezone from metadata.json.")
    p_coord.add_argument("--verbose", action="store_true", default=False, help="Print full tracebacks for any errors.")
    p_coord.set_defaults(func=handle_plot_coordination)


def _add_plot_termination_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``plot-termination`` subcommand parser."""
    p_term = subs.add_parser(
        "plot-termination",
        help="Generate a phase termination plot for a specific time window.",
        description="Generates an interactive phase termination plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_term = p_term.add_mutually_exclusive_group(required=True)
    group_term.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_term.add_argument("--targetid", metavar="ID", help="Intersection ID prefix.")
    group_term.add_argument("--all", action="store_true", help="Generate plots for all intersections in the directory.")
    p_term.add_argument("--start", required=True, metavar="ISO8601", help="Window start (local time, ISO-8601).")
    p_term.add_argument("--end", required=True, metavar="ISO8601", help="Window end, exclusive (local time, ISO-8601).")
    p_term.add_argument("--timezone", default=None, metavar="TZ", help="Override the timezone from metadata.json.")
    p_term.add_argument("--verbose", action="store_true", default=False, help="Print full tracebacks for any errors.")
    p_term.set_defaults(func=handle_plot_termination)


def _add_discrepancies_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``discrepancies`` subcommand parser."""
    p_disc = subs.add_parser(
        "discrepancies",
        help="Analyze co-located detector discrepancies for a time window.",
        description=(
            "Identify disagreements across co-located detector pairs for a specific\n"
            "time window. Reads detector mappings directly from the configuration database.\n\n"
            "To save the output, use the --output flag."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_disc = p_disc.add_mutually_exclusive_group(required=True)
    group_disc.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_disc.add_argument("--targetid", metavar="ID", help="Intersection ID (prefix of folder name).")
    group_disc.add_argument("--all", action="store_true", help="Analyze discrepancies for all intersections in the directory.")
    p_disc.add_argument(
        "--start",
        required=True,
        metavar="ISO8601",
        help="Query window start (local time, ISO-8601). E.g. '2024-06-01T06:00:00'."
    )
    p_disc.add_argument(
        "--end",
        required=True,
        metavar="ISO8601",
        help="Query window end, exclusive (local time, ISO-8601)."
    )
    p_disc.add_argument(
        "--lag",
        type=float,
        default=2.0,
        help="Minimum disagreement duration in seconds (default: 2.0)."
    )
    p_disc.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json."
    )
    p_disc.add_argument(
        "--output",
        action="store_true",
        default=False,
        help="Write results to a CSV file in the intersection's outputs directory."
    )
    p_disc.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors."
    )
    p_disc.set_defaults(func=handle_discrepancies)


def _add_plot_detectors_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``plot-detectors`` subcommand parser."""
    p_det = subs.add_parser(
        "plot-detectors",
        help="Generate interactive detector comparison plots.",
        description=(
            "Visualise co-located detector actuations side-by-side. Highlights\n"
            "identified discrepancies (unconfirmed pulses, extended disagreements).\n"
            "Reads pairs directly from configuration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_det = p_det.add_mutually_exclusive_group(required=True)
    group_det.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name.",
    )
    group_det.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_det.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all intersections in the directory.",
    )
    p_det.add_argument(
        "--start",
        required=True,
        metavar="ISO8601",
        help=(
            "Window start (local time, ISO-8601). "
            "E.g. '2024-06-01T06:00:00'."
        ),
    )
    p_det.add_argument(
        "--end",
        required=True,
        metavar="ISO8601",
        help="Window end, exclusive (local time, ISO-8601).",
    )
    p_det.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Filter to specific signal phases, e.g. --phases 2 6. "
            "Omit to include all configured pairs."
        ),
    )
    p_det.add_argument(
        "--lag",
        type=float,
        default=2.0,
        metavar="SEC",
        help=(
            "Minimum disagreement duration (seconds) for extended-disagreement "
            "classification (default: 2.0)."
        ),
    )
    p_det.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json.",
    )
    p_det.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_det.set_defaults(func=handle_plot_detectors)


def _add_plot_timing_actuation_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``plot-timing-actuation`` subcommand parser."""
    p_ta = subs.add_parser(
        "plot-timing-actuation",
        help="Generate interactive timing and actuation plots (window capped at 4 h, or 24 h with --phases or --detectors).",
        description=(
            "Visualise per-phase timing intervals (green, yellow, red), calls, "
            "pedestrian service, and detector actuations grouped by role.\n"
            "Window is capped at 4 h, or at 24 h with --phases or --detectors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_ta = p_ta.add_mutually_exclusive_group(required=True)
    group_ta.add_argument(
        "--target",
        metavar="FOLDER",
        help="Exact intersection folder name.",
    )
    group_ta.add_argument(
        "--targetid",
        metavar="ID",
        help="Intersection ID prefix (e.g. '2068').",
    )
    group_ta.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all intersections in the directory.",
    )
    p_ta.add_argument(
        "--start",
        required=True,
        metavar="ISO8601",
        help=(
            "Window start (local time, ISO-8601). "
            "E.g. '2024-06-01T06:00:00'."
        ),
    )
    p_ta.add_argument(
        "--end",
        required=True,
        metavar="ISO8601",
        help="Window end, exclusive (local time, ISO-8601).",
    )
    p_ta.add_argument(
        "--phases",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Filter to specific signal phases, e.g. --phases 2 6.",
    )
    p_ta.add_argument(
        "--detectors",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Filter to specific detector channels, e.g. --detectors 21 53.",
    )
    p_ta.add_argument(
        "--timezone",
        default=None,
        metavar="TZ",
        help="Override the timezone from metadata.json.",
    )
    p_ta.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full tracebacks for any errors.",
    )
    p_ta.set_defaults(func=handle_plot_timing_actuation)


def _add_video_calibrate_shapes_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``video-calibrate-shapes`` subcommand parser.

    Single-target only, no ``--all`` -- this is an interactive calibration
    session, not a batch operation.
    """
    p_vidcal = subs.add_parser(
        "video-calibrate-shapes",
        help="Interactively draw/edit loop and stopbar shapes for one camera.",
        description=(
            "Opens an interactive OpenCV window over a video's first frame to draw,\n"
            "edit, and save loop/stopbar shapes for video-overlay. One\n"
            "camera at a time -- this is a calibration session, not a batch operation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_vidcal = p_vidcal.add_mutually_exclusive_group(required=True)
    group_vidcal.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_vidcal.add_argument("--targetid", metavar="ID", help="Intersection ID prefix (e.g. '2068').")
    p_vidcal.add_argument("--camera", required=True, metavar="NAME", help="Camera name (used as the shape-config filename stem).")
    p_vidcal.add_argument("--video", required=True, metavar="PATH", help="Video file to calibrate against (first frame is used); .mp4 or .ts. A relative path is resolved against <target>/video/; an absolute path is used as-is.")
    p_vidcal.add_argument("--verbose", action="store_true", default=False, help="Print full tracebacks for any errors.")
    p_vidcal.set_defaults(func=handle_video_calibrate_shapes)


def _add_video_overlay_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``video-overlay`` subcommand parser.

    Single-target only, no ``--all`` -- one video = one camera.
    """
    p_vidov = subs.add_parser(
        "video-overlay",
        help="Render a video with live phase/overlap/detector status overlays.",
        description=(
            "Recolors loop and stopbar shapes (drawn via video-calibrate-shapes) by\n"
            "their actual phase, overlap, and detector status pulled from the\n"
            "events/cycles database, and writes a new output video."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_vidov = p_vidov.add_mutually_exclusive_group(required=True)
    group_vidov.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_vidov.add_argument("--targetid", metavar="ID", help="Intersection ID prefix (e.g. '2068').")
    p_vidov.add_argument("--camera", required=True, metavar="NAME", help="Camera name (matches the shape-config filename stem).")
    p_vidov.add_argument("--video", required=True, metavar="PATH", help="Input video file to overlay; .mp4 (frame-decode recorder) or .ts (remux recorder). A relative path is resolved against <target>/video/; an absolute path is used as-is.")
    p_vidov.add_argument("--start", required=True, metavar="ISO8601", help="Real-world timestamp of the video's first frame (local time, ISO-8601).")
    p_vidov.add_argument("--output", default=None, metavar="PATH", help="Output video path; .mp4/.m4v/.mov/.avi (the overlay is re-encoded, so .ts is input-only). Defaults to <target>/outputs/<start-date>/<camera>_overlay_<start-time>.mp4.")
    p_vidov.add_argument("--lookback", type=float, default=10.0, metavar="MIN", help="Minutes of event data to fetch before/after the video window, for correct status at the clip's edges (default: 10.0).")
    p_vidov.add_argument("--timezone", default=None, metavar="TZ", help="Override the timezone from metadata.json.")
    p_vidov.add_argument("--verbose", action="store_true", default=False, help="Print full tracebacks for any errors.")
    p_vidov.set_defaults(func=handle_video_overlay)


def _add_video_locate_phase_change_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``video-locate-phase-change`` subcommand parser.

    Single-target only, no ``--all`` -- one video = one camera, same
    exception as the other two video commands.
    """
    p_vidloc = subs.add_parser(
        "video-locate-phase-change",
        help="Auto-locate a phase's exact color-change time to correct a --start guess.",
        description=(
            "Finds the first green->yellow or yellow->red change for --phase at\n"
            "or after --start + --min-offset (whichever edge comes first; pin it\n"
            "with --transition), and reports its exact database timestamp. Call\n"
            "once without --observed-delta to get a short confirmation clip (the\n"
            "on-screen counter reads +0.000s at the frame the transition should\n"
            "occur if --start is exact, counting down through 0 and negative\n"
            "after). Watch it, read off the signed value at the instant the change\n"
            "actually happens, then call again with --observed-delta <that value>\n"
            "to get the corrected --start (= original --start + that value) --\n"
            "no further database lookup needed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_vidloc = p_vidloc.add_mutually_exclusive_group(required=True)
    group_vidloc.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_vidloc.add_argument("--targetid", metavar="ID", help="Intersection ID prefix (e.g. '2068').")
    p_vidloc.add_argument("--camera", required=True, metavar="NAME", help="Camera name (used to name the output clip).")
    p_vidloc.add_argument("--video", required=True, metavar="PATH", help="Input video file; .mp4 or .ts. A relative path is resolved against <target>/video/; an absolute path is used as-is.")
    p_vidloc.add_argument("--phase", required=True, type=int, metavar="N", help="Signal phase number visible in the camera view.")
    p_vidloc.add_argument("--transition", default=None, choices=["green_to_yellow", "yellow_to_red"], help="Pin the search to one edge. Default: auto-pick whichever of green->yellow/yellow->red occurs first.")
    p_vidloc.add_argument("--start", required=True, metavar="ISO8601", help="Rough guess for the real-world timestamp of the video's first frame (local time, ISO-8601).")
    p_vidloc.add_argument("--min-offset", type=float, default=5.0, metavar="SEC", help="Only consider transitions at least this many seconds into the video, per --start (default: 5.0).")
    p_vidloc.add_argument("--window", type=float, default=3.0, metavar="SEC", help="Half-width of the confirmation clip, in seconds (default: 3.0).")
    p_vidloc.add_argument("--observed-delta", type=float, default=None, metavar="SEC", help="Signed value read off the clip's counter at the instant the change actually happens. When given, prints the corrected --start instead of rendering a clip.")
    p_vidloc.add_argument("--timezone", default=None, metavar="TZ", help="Override the timezone from metadata.json.")
    p_vidloc.add_argument("--verbose", action="store_true", default=False, help="Print full tracebacks for any errors.")
    p_vidloc.set_defaults(func=handle_video_locate_phase_change)


def _add_video_sync_parser(subs: argparse._SubParsersAction) -> None:
    """Attach the ``video-sync`` subcommand parser.

    Single-target only, no ``--all`` -- one video = one camera, same
    exception as the other video commands.
    """
    p_vidsync = subs.add_parser(
        "video-sync",
        help="Find corrected --start timestamp by synchronizing video signal lamps to controller events.",
        description=(
            "Measures signal lamps in the recorded clip and aligns them against the\n"
            "database's controller phase and overlap states to determine an accurate\n"
            "video --start timestamp and camera clock slip."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_vidsync = p_vidsync.add_mutually_exclusive_group(required=True)
    group_vidsync.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_vidsync.add_argument("--targetid", metavar="ID", help="Intersection ID prefix (e.g. '2068').")
    p_vidsync.add_argument("--camera", required=True, metavar="NAME", help="Camera name matching <camera>_shapes.csv.")
    p_vidsync.add_argument("--video", required=True, metavar="PATH", help="Input video file; .mp4 or .ts. A relative path is resolved against <target>/video/; an absolute path is used as-is.")
    p_vidsync.add_argument("--start-guess", required=True, metavar="ISO8601", help="Estimated timestamp of the video's first frame (local time, ISO-8601).")
    p_vidsync.add_argument("--search", type=float, default=30.0, metavar="SECONDS", help="Half-window search duration in seconds (default: 30.0).")
    p_vidsync.add_argument("--timezone", default=None, metavar="TZ", help="Override the timezone from metadata.json.")
    p_vidsync.add_argument("--verbose", action="store_true", default=False, help="Print full tracebacks for any errors.")
    p_vidsync.set_defaults(func=handle_video_sync)


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser.

    Returns:
        Configured ``ArgumentParser`` with ``setup``, ``process``,
        ``report``, ``counts``, ``splits``, and ``discrepancies``
        subcommands attached.
    """
    parser = argparse.ArgumentParser(
        prog="atspm",
        description=(
            "ATSPM – Automated Traffic Signal Performance Measures\n"
            "Unified CLI for intersection setup, data ingestion, reporting, counts, splits, and discrepancy analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command", metavar="<command>")
    subs.required = True

    _add_setup_parser(subs)
    _add_retrieve_parser(subs)
    _add_process_parser(subs)
    _add_report_parser(subs)
    _add_counts_parser(subs)
    _add_splits_parser(subs)
    _add_split_monitor_parser(subs)
    _add_aog_parser(subs)
    _add_approach_delay_parser(subs)
    _add_yellow_red_parser(subs)
    _add_ped_delay_parser(subs)
    _add_wait_time_parser(subs)
    _add_detector_health_parser(subs)
    _add_split_failures_parser(subs)
    _add_infer_detectors_parser(subs)
    _add_flow_parser(subs)
    _add_critical_parser(subs)
    _add_optimize_parser(subs)
    _add_clock_drift_parser(subs)
    _add_preempt_parser(subs)
    _add_plot_coordination_parser(subs)
    _add_plot_termination_parser(subs)
    _add_discrepancies_parser(subs)
    _add_plot_detectors_parser(subs)
    _add_plot_timing_actuation_parser(subs)
    _add_video_calibrate_shapes_parser(subs)
    _add_video_overlay_parser(subs)
    _add_video_locate_phase_change_parser(subs)
    _add_video_sync_parser(subs)

    return parser


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate handler.

    This function is registered as the ``atspm`` console script entry point
    in ``pyproject.toml``.
    """
    parser = _build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()