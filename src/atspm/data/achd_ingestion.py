"""
ACHD Event-CSV Ingestion Engine (Imperative Shell)

Ingests ACHD high-resolution event exports (``{id}_Events_{YYYYMMDD}T0000.csv``)
into a normalised pyATSPM database, the CSV analogue of the ``.datZ`` path in
:mod:`atspm.data.ingestion`.  This shell owns file reading, the database
connection, metadata, gap insertion, and span bookkeeping; all parsing is
delegated to the functional core :mod:`atspm.analysis.achd`.

Scope (first pass)
==================
Populates ``events`` (with comms-gap markers), ``ingestion_log`` spans, and
``metadata`` (id / name / timezone / agency).  Cycle and config derivation are
deliberately out of scope — ACHD intersections ship no ``int_cfg.csv`` — and
are left to a separate pass.

File discovery and ordering
===========================
Only ``{id}_Events_*.csv`` files are considered for a given intersection.
Files whose requested export window is fully contained in another's are
skipped (overlapping re-exports); residual row overlap is absorbed by the
``events`` UNIQUE/IGNORE constraint and the core's de-duplication.

Gap handling
============
An ACHD file covers a contiguous date range, so a data gap shows up as a jump
between the last real event of one file and the first of the next.  When that
jump exceeds :data:`_GAP_THRESHOLD_S` a hard-reset marker (``event_code=-1``,
``parameter=COMMS_GAP_PARAM``) is written at the start of the later segment and
a new ``ingestion_log`` span is opened, so every duration / sequential-pairing
consumer stops at the break (CLAUDE.md §5).  The threshold sits well above the
few-minutes quiet an intersection can have between coordination events, and far
below a genuinely missing export (hours or days).

Package Location: src/atspm/data/achd_ingestion.py
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from ..analysis import achd
from ..analysis.decoders import COMMS_GAP_PARAM
from ..utils.timezone import DEFAULT_TIMEZONE
from .manager import DatabaseManager

# A jump between the last event of one file and the first of the next larger
# than this (seconds) is treated as a data gap, not a quiet period.  Ten
# minutes clears normal inter-event silence (coordination events recur each
# cycle) while staying far below any real missing-export hole.
_GAP_THRESHOLD_S = 600.0

#: Agency stamped on every ingested ACHD intersection's metadata.
ACHD_AGENCY_ID = "ACHD"

_FILENAME_DATE_RE = re.compile(r"_Events_(\d{8})T\d{4}", re.IGNORECASE)


class AchdIngestionEngine:
    """Ingests one intersection's ACHD event CSVs into a pyATSPM database.

    Responsibilities:
        - Discover and order ``{id}_Events_*.csv`` files for the intersection.
        - Parse each file's header and body via the functional core.
        - Detect cross-file data gaps and fence them with comms-gap markers.
        - Insert events (INSERT OR IGNORE) and record ``ingestion_log`` spans
          per contiguous segment.
        - Seed ``metadata`` from the first file's header.
    """

    def __init__(
        self,
        db_path: Path,
        raw_data_dir: Path,
        intersection_id: str,
        timezone: Optional[str] = None,
    ):
        """Initialise the engine.

        Args:
            db_path:         Path to the destination SQLite database.
            raw_data_dir:    Directory containing ``{id}_Events_*.csv`` files.
            intersection_id: Numeric ACHD signal id (e.g. ``'271'``); selects
                             which files in *raw_data_dir* belong to this DB.
            timezone:        Intersection wall-clock IANA zone.  ``None`` falls
                             back to the DB metadata, then
                             :data:`DEFAULT_TIMEZONE`.
        """
        self.db_path = Path(db_path)
        self.raw_data_dir = Path(raw_data_dir)
        self.intersection_id = str(intersection_id)
        self.timezone = self._resolve_timezone(timezone)

        self._files_processed = 0
        self._total_events = 0
        self._gap_markers = 0
        self._header_signal_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_timezone(self, timezone: Optional[str]) -> str:
        if timezone is not None:
            return timezone
        try:
            with DatabaseManager(self.db_path) as m:
                tz = m.get_metadata().get("timezone")
                if tz:
                    return tz
        except Exception:
            pass
        return DEFAULT_TIMEZONE

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Ingest every (non-contained) ACHD file for the intersection.

        Reads files in chronological order, inserting each file's events in
        its own transaction so memory stays bounded for multi-million-row
        exports.  Spans are opened per contiguous segment and closed when a
        gap is detected or the file list ends.
        """
        files = self._scan_files()
        if not files:
            print(
                f"No '{self.intersection_id}_Events_*.csv' files found in "
                f"{self.raw_data_dir}"
            )
            return

        print(f"Found {len(files)} ACHD file(s) for intersection "
              f"{self.intersection_id}")

        # Open span state: (span_start, span_end, row_count).
        active: Optional[List[float]] = None
        prev_last_ts: Optional[float] = None
        spans: List[Tuple[float, float, int]] = []

        for path, header in files:
            events = self._load_file_events(path)
            if events is None or events.empty:
                print(f"  Skipping {path.name}: no usable events")
                continue

            events = events.sort_values(
                ["timestamp", "event_code", "parameter"]
            ).reset_index(drop=True)
            first_ts = float(events["timestamp"].iloc[0])
            last_ts = float(events["timestamp"].iloc[-1])

            if self._header_signal_name is None and header.signal_name:
                self._header_signal_name = header.signal_name

            gap_opened = (
                prev_last_ts is not None
                and first_ts - prev_last_ts > _GAP_THRESHOLD_S
            )
            if gap_opened:
                events = self._prepend_gap_marker(events, first_ts)

            row_count = len(events)
            self._insert_events(events)
            self._files_processed += 1
            self._total_events += row_count

            if active is None or gap_opened:
                if active is not None:
                    spans.append((active[0], active[1], int(active[2])))
                active = [first_ts, last_ts, row_count]
            else:
                active[1] = max(active[1], last_ts)
                active[2] += row_count

            prev_last_ts = last_ts
            print(f"  Ingested {path.name}: {row_count:,} rows")

        if active is not None:
            spans.append((active[0], active[1], int(active[2])))

        self._write_spans(spans)
        self._write_metadata()

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the run and the resulting database.

        Returns:
            Dict with ``files_processed``, ``total_events``, ``gap_markers``
            (database-wide ``event_code=-1`` count), ``span_count`` and
            ``date_range`` (local ISO ``start``/``end`` or ``None``).
        """
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute(
                "SELECT COUNT(*), MIN(span_start), MAX(span_end) FROM ingestion_log"
            )
            span_count, min_ts, max_ts = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM events WHERE event_code = -1")
            gap_count = cur.fetchone()[0]

        tz = pytz.timezone(self.timezone)
        return {
            "files_processed": self._files_processed,
            "total_events": self._total_events,
            "gap_markers": gap_count,
            "span_count": span_count or 0,
            "date_range": {
                "start": (
                    datetime.fromtimestamp(min_ts, tz).isoformat()
                    if min_ts else None
                ),
                "end": (
                    datetime.fromtimestamp(max_ts, tz).isoformat()
                    if max_ts else None
                ),
            },
        }

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _scan_files(self) -> List[Tuple[Path, achd.AchdHeader]]:
        """Return this intersection's files ordered chronologically.

        Reads each file's header, drops any whose requested window is fully
        contained in another's, and sorts the survivors by window start
        (falling back to the filename date when a header has no start).

        Returns:
            List of ``(path, header)`` tuples in ingestion order.
        """
        candidates: List[Tuple[Path, achd.AchdHeader, Optional[datetime]]] = []
        for path in self.raw_data_dir.glob(f"{self.intersection_id}_Events_*.csv"):
            header = self._read_header(path)
            sort_key = header.start or self._filename_date(path.name)
            candidates.append((path, header, sort_key))

        kept = self._drop_contained(candidates)
        kept.sort(
            key=lambda c: (c[2] is None, c[2] or datetime.min, c[0].name)
        )
        return [(path, header) for path, header, _ in kept]

    @staticmethod
    def _drop_contained(
        candidates: List[Tuple[Path, achd.AchdHeader, Optional[datetime]]],
    ) -> List[Tuple[Path, achd.AchdHeader, Optional[datetime]]]:
        """Remove files whose header window is fully inside another's.

        Files lacking a parseable ``start``/``end`` window are always kept —
        containment cannot be judged for them, and the DB constraint absorbs
        any resulting row overlap.
        """
        windowed = [c for c in candidates if c[1].start and c[1].end]
        others = [c for c in candidates if not (c[1].start and c[1].end)]
        keep = []
        for i, (path_i, h_i, _) in enumerate(windowed):
            contained = False
            for j, (_, h_j, _) in enumerate(windowed):
                if i == j:
                    continue
                # Strictly contained, or equal-and-earlier-index wins the tie.
                if h_j.start <= h_i.start and h_j.end >= h_i.end:
                    if (h_j.start, h_j.end) != (h_i.start, h_i.end) or j < i:
                        contained = True
                        break
            if not contained:
                keep.append(windowed[i])
        return keep + others

    def _read_header(self, path: Path) -> achd.AchdHeader:
        """Read and parse the metadata header of one file (I/O)."""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = [fh.readline() for _ in range(achd.ACHD_HEADER_ROWS)]
        except OSError as exc:
            print(f"Warning: cannot read header of {path.name}: {exc}")
            return achd.AchdHeader(None, None, None, None, None)
        return achd.parse_achd_header("".join(lines))

    @staticmethod
    def _filename_date(filename: str) -> Optional[datetime]:
        """Extract the ``YYYYMMDD`` export date from an ACHD filename."""
        m = _FILENAME_DATE_RE.search(filename)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Per-file parsing
    # ------------------------------------------------------------------

    def _load_file_events(self, path: Path) -> Optional[pd.DataFrame]:
        """Read one file's body and transform it via the functional core.

        Returns:
            Standard ``[timestamp, event_code, parameter]`` frame, or ``None``
            on read / decode error (the file is skipped, not fatal).
        """
        try:
            raw = pd.read_csv(
                path,
                skiprows=achd.ACHD_HEADER_ROWS,
                dtype=str,
                keep_default_na=False,
            )
        except (OSError, pd.errors.ParserError) as exc:
            print(f"Error reading {path.name}: {exc}")
            return None
        try:
            return achd.achd_events_from_frame(raw, self.timezone)
        except achd.AchdDecodingError as exc:
            print(f"Error decoding {path.name}: {exc}")
            return None

    def _prepend_gap_marker(
        self, events: pd.DataFrame, gap_ts: float
    ) -> pd.DataFrame:
        """Insert a comms-gap marker at *gap_ts* ahead of a discontinuity."""
        self._gap_markers += 1
        marker = pd.DataFrame(
            {"timestamp": [gap_ts], "event_code": [-1], "parameter": [COMMS_GAP_PARAM]}
        )
        return pd.concat([marker, events], ignore_index=True)

    # ------------------------------------------------------------------
    # Database writes
    # ------------------------------------------------------------------

    def _insert_events(self, events: pd.DataFrame) -> None:
        """Insert one frame of events in its own transaction (INSERT OR IGNORE)."""
        tuples = list(events.itertuples(index=False, name=None))
        if not tuples:
            return
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            try:
                cur.executemany(
                    "INSERT OR IGNORE INTO events "
                    "(timestamp, event_code, parameter) VALUES (?, ?, ?)",
                    tuples,
                )
                m.conn.commit()
            except sqlite3.Error as exc:
                m.conn.rollback()
                raise RuntimeError(f"Error inserting ACHD events: {exc}")

    def _write_spans(self, spans: List[Tuple[float, float, int]]) -> None:
        """Replace the ingestion_log with the contiguous spans from this run."""
        if not spans:
            return
        now_iso = datetime.utcnow().isoformat()
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            try:
                for span_start, span_end, row_count in spans:
                    cur.execute(
                        """
                        INSERT INTO ingestion_log
                            (span_start, span_end, processed_at, row_count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(span_start) DO UPDATE SET
                            span_end     = excluded.span_end,
                            processed_at = excluded.processed_at,
                            row_count    = excluded.row_count
                        """,
                        (span_start, span_end, now_iso, row_count),
                    )
                m.conn.commit()
            except sqlite3.Error as exc:
                m.conn.rollback()
                raise RuntimeError(f"Error writing ingestion spans: {exc}")

    def _write_metadata(self) -> None:
        """Seed the metadata row from the header, without clobbering fields.

        Only writes when the table has no intersection_id yet, so re-running an
        append does not overwrite any metadata a later pass may have enriched.
        """
        with DatabaseManager(self.db_path) as m:
            existing = m.get_metadata()
            if existing.get("intersection_id"):
                return
            m.set_metadata(
                intersection_id=self.intersection_id,
                intersection_name=self._header_signal_name or self.intersection_id,
                timezone=self.timezone,
                agency_id=ACHD_AGENCY_ID,
            )


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def run_achd_ingestion(
    db_path: Path,
    raw_data_dir: Path,
    intersection_id: str,
    timezone: Optional[str] = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """Ingest ACHD event CSVs for one intersection into a pyATSPM database.

    Initialises the schema if needed, optionally clears prior ingested rows
    (``rebuild``), runs the engine, and prints a summary.

    Args:
        db_path:         Destination SQLite database path.
        raw_data_dir:    Directory holding ``{id}_Events_*.csv`` files.
        intersection_id: ACHD signal id selecting the files to ingest.
        timezone:        Intersection wall-clock zone (``None`` → metadata →
                         :data:`DEFAULT_TIMEZONE`).
        rebuild:         When ``True`` start from an empty database, deleting
                         any existing DB file first.

    Returns:
        The engine's ``get_ingestion_stats()`` dict.
    """
    from .manager import init_db

    db_path = Path(db_path)
    if rebuild:
        # Delete the file outright rather than DELETE-in-place: a full year of
        # a busy arterial is ~100M+ rows, and an in-place clear leaves the
        # multi-GB file at full size (SQLite frees pages but does not shrink),
        # so re-inserting on top can exhaust the disk.  A fresh file is also a
        # true rebuild — config and metadata are re-established by the caller
        # (CLI re-imports int_cfg) and the engine (re-seeds metadata).
        for suffix in ("", "-wal", "-shm"):
            p = db_path.with_name(db_path.name + suffix)
            if p.exists():
                p.unlink()
        print("Rebuild: removed existing database file.")

    init_db(db_path)
    engine = AchdIngestionEngine(db_path, raw_data_dir, intersection_id, timezone)
    engine.run()

    stats = engine.get_ingestion_stats()
    print("\nACHD Ingestion Complete!")
    print(f"  Files processed : {stats['files_processed']}")
    print(f"  Total events    : {stats['total_events']:,}")
    print(f"  Gap markers     : {stats['gap_markers']}")
    print(f"  Log spans       : {stats['span_count']}")
    if stats["date_range"]["start"]:
        print(
            f"  Date range      : {stats['date_range']['start']} "
            f"→ {stats['date_range']['end']}"
        )
    return stats
