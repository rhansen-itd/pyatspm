"""
ATSPM Unified Command-Line Interface

Exposes subcommands from intersection configuration setup through reporting and visualization:

    atspm setup              --targetid <id>             Create a new intersection environment
    atspm process            --targetid <id> [...]       Ingest data and compute cycles
    atspm report             --targetid <id> [...]       Generate ATSPM performance reports
    atspm counts             --targetid <id> [...]       Generate vehicle and pedestrian counts
    atspm splits             --targetid <id> [...]       Generate phase split and timing records
    atspm aog                --targetid <id> [...]       Generate Arrival on Green (AOG) tables
    atspm discrepancies      --targetid <id> [...]       Analyze detector discrepancies
    atspm plot-detectors     --targetid <id> [...]       Generate interactive detector comparison plots
    atspm plot-coordination  --targetid <id> [...]       Generate interactive coordination diagram plots
    atspm plot-termination   --targetid <id> [...]       Generate interactive phase termination plots
    atspm video-calibrate-shapes --targetid <id> [...]   Interactively draw/edit camera shape config
    atspm video-overlay      --targetid <id> [...]       Render a video with live status overlays

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
from datetime import datetime
from pathlib import Path
from typing import Optional

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

    # Summary --------------------------------------------------------------------
    rel = lambda p: p.relative_to(_find_project_root())  # noqa: E731
    print("\n✅  Setup complete.")
    print(f"    1. Edit metadata:       {rel(metadata_path)}")
    print(f"    2. Add .datZ files to:  {rel(raw_dir)}")
    print(
        f"    3. Ingest data:         "
        f"atspm process --target \"{target}\""
    )


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

def _process_single_intersection(target_name: str, args: argparse.Namespace) -> None:
    """Core logic to process a single intersection."""
    # Resolve fill_gaps: --fill-gaps activates Path B.
    fill_gaps: bool = args.fill_gaps

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
        args.timezone or meta.get("timezone") or "US/Mountain"
    )

    int_name = meta.get("intersection_name", target_name)
    int_id   = meta.get("intersection_id",   target_name.split("_")[0])
    mode_tag = "Gap Fill" if fill_gaps else "Fast Append"

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
    timezone = args.timezone or meta.get("timezone") or "US/Mountain"

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
    timezone = args.timezone or meta.get("timezone") or "US/Mountain"

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
    timezone = args.timezone or meta.get("timezone") or "US/Mountain"

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
    timezone = args.timezone or meta.get("timezone") or "US/Mountain"

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
    tz_str = args.timezone or meta.get("timezone") or "US/Mountain"
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
    tz_str = args.timezone or meta.get("timezone") or "US/Mountain"
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
    tz_str = args.timezone or meta.get("timezone") or "US/Mountain"
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

    existing = ShapeConfig.load(shape_path) if shape_path.exists() else None
    if existing:
        print(f"\n🎯  Editing existing shape config: {shape_path} ({len(existing.shapes)} shapes)")
    else:
        print(f"\n🎯  Creating new shape config: {shape_path}")

    try:
        config = calibrate_shapes(args.video, existing)
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _die(f"Calibration failed: {exc}")

    config.save(shape_path)
    print(f"\n✅  Saved {len(config.shapes)} shapes to {shape_path}")


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

    tz_str = args.timezone or meta.get("timezone") or "US/Mountain"
    tz = pytz.timezone(tz_str)
    try:
        start_dt = tz.localize(datetime.fromisoformat(args.start))
    except ValueError as exc:
        _die(f"Invalid datetime format for --start: {exc}. Use ISO-8601 (e.g. 2024-06-01T06:00:00).")

    output_path = Path(args.output) if args.output else target_dir / "outputs" / f"{args.camera}_overlay.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", target_name)
    print(f"\n🎬  Rendering video overlay for {int_name}")
    print(f"    Video:  {args.video}")
    print(f"    Start:  {start_dt.isoformat()}")
    print(f"    Output: {output_path}")

    try:
        result = render_overlay(
            db_path,
            shape_config,
            args.video,
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

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
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
        default="US/Mountain",
        metavar="TZ",
        help="IANA timezone for the new metadata.json (default: US/Mountain)."
    )
    p_setup.set_defaults(func=handle_setup)

    # ------------------------------------------------------------------
    # process
    # ------------------------------------------------------------------
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
            "  Obsolete gap markers are scrubbed; cycles are surgically repaired."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_proc = p_proc.add_mutually_exclusive_group(required=True)
    group_proc.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_proc.add_argument("--targetid", metavar="ID", help="Intersection ID (prefix of folder name).")
    group_proc.add_argument("--all", action="store_true", help="Process all intersections in the directory.")
    p_proc.add_argument(
        "--fill-gaps",
        action="store_true",
        default=False,
        help=(
            "Enable Gap Fill mode (Path B): scan for and ingest historical gaps; "
            "scrub obsolete gap markers; surgically repair affected cycles."
        ),
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

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # counts
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # splits
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # aog
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # plot-coordination
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # plot-termination
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # discrepancies
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # plot-detectors
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # video-calibrate-shapes (single-target only, no --all -- interactive)
    # ------------------------------------------------------------------
    p_vidcal = subs.add_parser(
        "video-calibrate-shapes",
        help="Interactively draw/edit loop and stopbar shapes for one camera.",
        description=(
            "Opens an interactive OpenCV window over a video's first frame to draw,\n"
            "edit, and save loop/stopbar/approach shapes for video-overlay. One\n"
            "camera at a time -- this is a calibration session, not a batch operation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_vidcal = p_vidcal.add_mutually_exclusive_group(required=True)
    group_vidcal.add_argument("--target", metavar="FOLDER", help="Exact intersection folder name.")
    group_vidcal.add_argument("--targetid", metavar="ID", help="Intersection ID prefix (e.g. '2068').")
    p_vidcal.add_argument("--camera", required=True, metavar="NAME", help="Camera name (used as the shape-config filename stem).")
    p_vidcal.add_argument("--video", required=True, metavar="PATH", help="Video file to calibrate against (first frame is used).")
    p_vidcal.add_argument("--verbose", action="store_true", default=False, help="Print full tracebacks for any errors.")
    p_vidcal.set_defaults(func=handle_video_calibrate_shapes)

    # ------------------------------------------------------------------
    # video-overlay (single-target only, no --all -- one video = one camera)
    # ------------------------------------------------------------------
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
    p_vidov.add_argument("--video", required=True, metavar="PATH", help="Input video file to overlay.")
    p_vidov.add_argument("--start", required=True, metavar="ISO8601", help="Real-world timestamp of the video's first frame (local time, ISO-8601).")
    p_vidov.add_argument("--output", default=None, metavar="PATH", help="Output video path. Defaults to <target>/outputs/<camera>_overlay.mp4.")
    p_vidov.add_argument("--lookback", type=float, default=10.0, metavar="MIN", help="Minutes of event data to fetch before/after the video window, for correct status at the clip's edges (default: 10.0).")
    p_vidov.add_argument("--timezone", default=None, metavar="TZ", help="Override the timezone from metadata.json.")
    p_vidov.add_argument("--verbose", action="store_true", default=False, help="Print full tracebacks for any errors.")
    p_vidov.set_defaults(func=handle_video_overlay)

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