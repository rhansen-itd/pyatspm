"""
ATSPM Unified Command-Line Interface

Exposes three subcommands that replace the legacy ``scripts/`` folder:

    atspm setup   --target <folder_name>       Create a new intersection environment
    atspm process --target <folder_name> [...]  Ingest data and compute cycles
    atspm report  --target <folder_name> [...]  Generate ATSPM performance reports

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
    # Resolve fill_gaps: --fill-gaps OR --full both activate Path B.
    fill_gaps: bool = args.fill_gaps or args.full

    # Lazy imports keep module load fast and decouple from missing deps.
    from atspm.data import init_db, import_config, run_ingestion
    from atspm.data.manager import DatabaseManager

    target_dir = _get_target_dir(args.target)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)
    config_csv = target_dir / "int_cfg.csv"
    raw_dir    = target_dir / "raw_data"

    # Determine effective timezone: CLI override > metadata > default
    timezone: Optional[str] = (
        args.timezone or meta.get("timezone") or "US/Mountain"
    )

    int_name = meta.get("intersection_name", args.target)
    int_id   = meta.get("intersection_id",   args.target.split("_")[0])
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


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def handle_report(args: argparse.Namespace) -> None:
    """Generate ATSPM performance reports for one or more dates.

    Validates cycle data for each requested date (running on-demand
    reprocessing when a date is absent), then invokes ``PlotGenerator``
    to produce the full suite of Plotly reports.

    Args:
        args: Parsed CLI arguments.
    """
    from atspm.data.processing import CycleProcessor
    from atspm.reports.generators import PlotGenerator

    target_dir = _get_target_dir(args.target)
    meta       = _load_metadata(target_dir)
    db_path    = _resolve_db_path(target_dir, meta)
    output_dir = target_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_name = meta.get("intersection_name", args.target)

    if not db_path.exists():
        _die(
            f"Database not found: {db_path}\n"
            f"Run 'atspm process --target {args.target}' first."
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
        sys.exit(1)


# ---------------------------------------------------------------------------
# Report helper shims (bridge between new anchor-based processor and the
# date-level stats / reprocess API that test_reporting.py relied on)
# ---------------------------------------------------------------------------

def _get_cycle_summary(processor: "CycleProcessor", date_str: str) -> Optional[dict]:
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
    start_epoch = tz.localize(
        datetime.combine(local_date, time.min)
    ).timestamp()
    end_epoch = tz.localize(
        datetime.combine(local_date + timedelta(days=1), time.min)
    ).timestamp()

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


def _reprocess_date(
    processor: "CycleProcessor",
    date_str: str,
    timezone: Optional[str] = None,
) -> None:
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
    t_end   = tz.localize(
        datetime.combine(local_date + timedelta(days=1), time.min)
    ).timestamp()

    # Use gap-fill (Path B) so the repair is surgically bounded.
    processor.process_span(t_start, t_end, fill_gaps=True)


# ===========================================================================
# Argument parser construction
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser.

    Returns:
        Configured ``ArgumentParser`` with ``setup``, ``process``, and
        ``report`` subcommands attached.
    """
    parser = argparse.ArgumentParser(
        prog="atspm",
        description=(
            "ATSPM – Automated Traffic Signal Performance Measures\n"
            "Unified CLI for intersection setup, data ingestion, and reporting."
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
        help="Intersection folder name, e.g. '2068_US-95_and_SH-8'.",
    )
    p_setup.add_argument(
        "--timezone",
        default="US/Mountain",
        metavar="TZ",
        help="IANA timezone for the new metadata.json (default: US/Mountain).",
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
            "PATH B – Gap Fill (--fill-gaps or --full):\n"
            "  All files are scanned; historical gaps are filled.\n"
            "  Obsolete gap markers are scrubbed; cycles are surgically repaired."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_proc.add_argument(
        "--target",
        required=True,
        metavar="FOLDER",
        help="Intersection folder name.",
    )
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
        "--full",
        action="store_true",
        default=False,
        help="Legacy alias for --fill-gaps.",
    )
    p_proc.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Number of .datZ files per transaction commit (default: 50).",
    )
    p_proc.add_argument(
        "--no-cycles",
        action="store_true",
        default=False,
        help="Skip cycle processing; only ingest raw events.",
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
    p_rep.add_argument(
        "--target",
        required=True,
        metavar="FOLDER",
        help="Intersection folder name.",
    )
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
        help="Print full tracebacks for any per-date generation errors.",
    )
    p_rep.set_defaults(func=handle_report)

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
