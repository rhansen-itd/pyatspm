"""
ATSPM Data Ingestion Engine (Imperative Shell)

Handles incremental ingestion of ``.datZ`` files into the SQLite database.
Manages file I/O, state tracking, timezone conversions, gap detection, and
span merging for the Gap-Fill path.

Two operating modes
====================

Fast Append (``fill_gaps=False``, default)
------------------------------------------
Scans only for files *newer* than the absolute maximum ``span_end`` in
``ingestion_log``.  No span-coverage analysis is performed.  This is
maximally efficient for the common daily-append workflow.

Gap Fill (``fill_gaps=True``)
------------------------------
Scans **all** ``.datZ`` files in the directory, skipping only those that are
already fully covered by an existing ingestion span.  Files that fall into
holes between spans are processed, and their spans are subsequently **merged**
with any adjacent spans they close.  Gap markers that the filled data makes
obsolete are removed by the ``CycleProcessor`` before cycle recalculation.

Span merging rule
-----------------
After committing a batch that fills a gap, any ``ingestion_log`` spans that
are now adjacent or overlapping are coalesced into a single row
(``span_start = MIN``, ``span_end = MAX``, ``row_count = SUM``).

Package Location: src/atspm/data/ingestion.py
"""

import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from ..analysis import decoders
from .manager import DatabaseManager


class IngestionEngine:
    """Manages incremental ingestion of ``.datZ`` files into the SQLite database.

    Responsibilities:
        - File scanning (append-only or gap-aware, depending on ``fill_gaps``).
        - Filename-timestamp parsing (local → UTC).
        - Gap detection and gap-marker insertion into ``events``.
        - Span-based state tracking in ``ingestion_log``.
        - Span merging after gap-fill commits.
        - Driving the ``CycleProcessor`` via the post-commit hook.
    """

    def __init__(
        self,
        db_path: Path,
        raw_data_dir: Path,
        timezone: str = None,
        cycle_processor=None,
    ):
        """Initialise the engine.

        Args:
            db_path:         Path to the SQLite database.
            raw_data_dir:    Directory containing ``.datZ`` files.
            timezone:        Controller local timezone (e.g. ``'US/Mountain'``).
                             ``None`` reads from the metadata table.
            cycle_processor: Optional ``CycleProcessor`` instance.  When
                             provided, ``process_span`` is called for each
                             committed time range so cycles stay current
                             without a separate pass.
        """
        self.db_path = Path(db_path)
        self.raw_data_dir = Path(raw_data_dir)
        self.tz = pytz.timezone(self._resolve_timezone(timezone))
        self._cycle_processor = cycle_processor

        self._files_processed: int = 0
        self._total_events: int = 0
        self._gap_markers: int = 0

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_timezone(self, timezone: Optional[str]) -> str:
        if timezone is not None:
            return timezone
        try:
            with DatabaseManager(self.db_path) as m:
                meta = m.get_metadata()
                if meta and meta.get("timezone"):
                    print(
                        f"IngestionEngine: using timezone from metadata: "
                        f"{meta['timezone']}"
                    )
                    return meta["timezone"]
        except Exception:
            pass
        print("IngestionEngine: no timezone found, defaulting to US/Mountain")
        return "US/Mountain"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, fill_gaps: bool = False, batch_size: int = 50) -> None:
        """Execute the ingestion process.

        Args:
            fill_gaps:  When ``False`` (default) only process files newer than
                        the most recent span end (Fast Append).  When ``True``
                        scan all files and fill any gaps (Gap Fill).
            batch_size: Number of files to buffer before committing a
                        transaction.
        """
        self._fill_gaps = fill_gaps  # stored for use in _commit_batch

        if fill_gaps:
            print("Ingestion mode: Gap Fill (scanning all files)")
            file_list = self._scan_files_gap_fill()
        else:
            last_ts = self._get_last_ingested_timestamp()
            if last_ts:
                print(
                    f"Ingestion mode: Fast Append – resuming from "
                    f"{datetime.fromtimestamp(last_ts, self.tz)}"
                )
            else:
                print("Ingestion mode: Fast Append – starting fresh")
            file_list = self._scan_files_append(last_ts)

        if not file_list:
            print("No new files to process")
            return

        print(f"Found {len(file_list)} files to process")
        self._process_batches(file_list, batch_size, fill_gaps=fill_gaps)

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Return summary statistics from the current run and the full log.

        Returns:
            Dict with keys ``files_processed``, ``total_events``,
            ``gap_markers``, ``span_count``, ``date_range``.
        """
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute(
                "SELECT COUNT(*), MIN(span_start), MAX(span_end) FROM ingestion_log"
            )
            span_count, min_ts, max_ts = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM events WHERE event_code = -1")
            gap_count = cur.fetchone()[0]

        return {
            "files_processed": self._files_processed,
            "total_events": self._total_events,
            "gap_markers": gap_count,
            "span_count": span_count or 0,
            "date_range": {
                "start": (
                    datetime.fromtimestamp(min_ts, self.tz).isoformat()
                    if min_ts else None
                ),
                "end": (
                    datetime.fromtimestamp(max_ts, self.tz).isoformat()
                    if max_ts else None
                ),
            },
        }

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_last_ingested_timestamp(self) -> Optional[float]:
        """Return the ``span_end`` of the most recently ingested span.

        Returns:
            UTC epoch float, or ``None`` if the log is empty.
        """
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute("SELECT MAX(span_end) FROM ingestion_log")
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else None

    def _get_current_span(self) -> Optional[Tuple[float, float, int]]:
        """Return ``(span_start, span_end, row_count)`` for the latest span."""
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute("""
                SELECT span_start, span_end, row_count
                FROM ingestion_log
                ORDER BY span_start DESC LIMIT 1
            """)
            row = cur.fetchone()
            return tuple(row) if row else None

    def _get_all_spans(self) -> List[Tuple[float, float]]:
        """Return all ``(span_start, span_end)`` pairs, sorted ascending.

        Returns:
            List of ``(span_start, span_end)`` UTC epoch float tuples.
        """
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute(
                "SELECT span_start, span_end FROM ingestion_log "
                "ORDER BY span_start"
            )
            return [(r[0], r[1]) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # File scanning
    # ------------------------------------------------------------------

    def _scan_files_append(
        self,
        min_timestamp: Optional[float],
    ) -> List[Tuple[Path, float]]:
        """Scan for files strictly newer than ``min_timestamp``.

        Args:
            min_timestamp: Exclude files at or before this UTC epoch.
                           ``None`` includes everything.

        Returns:
            Sorted list of ``(filepath, utc_epoch)`` tuples.
        """
        result = []
        for fp in self.raw_data_dir.glob("*.datZ"):
            ts = self._parse_filename_timestamp(fp.name)
            if ts is None:
                print(f"Warning: cannot parse timestamp from {fp.name}")
                continue
            if min_timestamp is not None and ts <= min_timestamp:
                continue
            result.append((fp, ts))
        result.sort(key=lambda x: x[1])
        return result

    def _scan_files_gap_fill(self) -> List[Tuple[Path, float]]:
        """Scan all files, skipping those already covered by existing spans.

        A file is considered covered when its timestamp falls strictly inside
        an existing ``[span_start, span_end]`` interval (inclusive).  Files at
        span edges are included to allow the engine to decide whether they are
        genuine extensions or exact duplicates.

        Returns:
            Sorted list of ``(filepath, utc_epoch)`` tuples for uncovered files.
        """
        existing_spans = self._get_all_spans()

        def _is_covered(ts: float) -> bool:
            for s_start, s_end in existing_spans:
                if s_start <= ts <= s_end:
                    return True
            return False

        result = []
        for fp in self.raw_data_dir.glob("*.datZ"):
            ts = self._parse_filename_timestamp(fp.name)
            if ts is None:
                print(f"Warning: cannot parse timestamp from {fp.name}")
                continue
            if _is_covered(ts):
                continue
            result.append((fp, ts))
        result.sort(key=lambda x: x[1])
        return result

    def _parse_filename_timestamp(self, filename: str) -> Optional[float]:
        """Extract the local datetime from a filename and convert to UTC epoch.

        Expected format: ``*_YYYY_MM_DD_HHMM.datZ``

        Args:
            filename: ``.datZ`` filename.

        Returns:
            UTC epoch float, or ``None`` on parse failure.
        """
        match = re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{4})", filename)
        if not match:
            return None
        try:
            year, month, day, hhmm = match.groups()
            naive = datetime(
                int(year), int(month), int(day),
                int(hhmm[:2]), int(hhmm[2:])
            )
            return self.tz.localize(naive).timestamp()
        except (
            ValueError,
            pytz.exceptions.AmbiguousTimeError,
            pytz.exceptions.NonExistentTimeError,
        ) as exc:
            print(f"Warning: error parsing {filename}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batches(
        self,
        file_list: List[Tuple[Path, float]],
        batch_size: int,
        fill_gaps: bool,
    ) -> None:
        """Process ``file_list`` in batches, committing every ``batch_size`` files.

        Maintains running span state so consecutive files collapse into one
        ``ingestion_log`` row.  Collects ``(T_start, T_end)`` pairs for each
        committed span and passes them to the cycle-processing hook.

        Args:
            file_list:  Sorted ``(filepath, utc_epoch)`` tuples.
            batch_size: Maximum files per transaction.
            fill_gaps:  Forwarded to the cycle-processing hook.
        """
        total = len(file_list)
        existing_span = self._get_current_span()
        # active_span: (span_start, span_end, row_count, last_event_ts)
        active_span: Optional[Tuple[float, float, int, float]] = (
            (
                existing_span[0], existing_span[1],
                existing_span[2], existing_span[1]
            )
            if existing_span is not None else None
        )

        events_buffer: List[pd.DataFrame] = []
        span_updates:  List[Tuple[float, float, int]] = []
        # Ranges committed in this batch; fed to the cycle hook after commit.
        batch_cycle_ranges: List[Tuple[float, float]] = []

        for i, (file_path, utc_start) in enumerate(file_list):
            next_file_utc      = file_list[i + 1][1] if i + 1 < total else None
            prev_last_event_ts = active_span[3] if active_span is not None else None

            result = self._process_file(
                file_path, utc_start, next_file_utc,
                active_span, prev_last_event_ts,
            )
            if result is None:
                continue

            events_df, row_count, gap_opened, new_span = result
            self._files_processed += 1
            self._total_events    += row_count

            if not events_df.empty:
                events_buffer.append(events_df)

            if gap_opened and active_span is not None:
                span_updates.append(active_span[:3])
                batch_cycle_ranges.append((active_span[0], active_span[1]))
                active_span = new_span
            elif active_span is None:
                active_span = new_span
            else:
                active_span = (
                    active_span[0],
                    utc_start,
                    active_span[2] + row_count,
                    new_span[3],
                )

            should_commit = (
                len(span_updates) + 1 >= batch_size or i == total - 1
            )
            if should_commit and (events_buffer or span_updates or
                                  active_span is not None):
                all_spans = span_updates[:]
                if active_span is not None:
                    all_spans.append(active_span[:3])

                self._commit_batch(events_buffer, all_spans, fill_gaps=fill_gaps)
                print(f"  Committed {i + 1}/{total} files...")

                if active_span is not None:
                    batch_cycle_ranges.append((active_span[0], active_span[1]))

                self._trigger_cycle_processing(batch_cycle_ranges, fill_gaps)

                events_buffer       = []
                span_updates        = []
                batch_cycle_ranges  = []
                # active_span stays open — it may still grow

    # ------------------------------------------------------------------
    # Cycle processing hook
    # ------------------------------------------------------------------

    def _trigger_cycle_processing(
        self,
        cycle_ranges: List[Tuple[float, float]],
        fill_gaps: bool,
    ) -> None:
        """Call ``CycleProcessor.process_span`` for each committed range.

        Merges overlapping / touching ranges before dispatching to avoid
        redundant double-processing.

        Args:
            cycle_ranges: ``(T_start, T_end)`` pairs written in the last commit.
            fill_gaps:    Forwarded to ``process_span``.
        """
        if not self._cycle_processor or not cycle_ranges:
            return

        merged = sorted(cycle_ranges, key=lambda r: r[0])
        coalesced: List[Tuple[float, float]] = [merged[0]]
        for start, end in merged[1:]:
            ps, pe = coalesced[-1]
            if start <= pe + 1:
                coalesced[-1] = (ps, max(pe, end))
            else:
                coalesced.append((start, end))

        for t_start, t_end in coalesced:
            try:
                self._cycle_processor.process_span(t_start, t_end,
                                                   fill_gaps=fill_gaps)
            except Exception as exc:
                print(
                    f"IngestionEngine WARNING: cycle processing failed for "
                    f"range [{t_start}, {t_end}]: {exc}"
                )

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------

    def _process_file(
        self,
        file_path: Path,
        utc_start: float,
        next_file_utc: Optional[float],
        active_span: Optional[Tuple[float, float, int, float]],
        prev_last_event_ts: Optional[float],
    ) -> Optional[Tuple[pd.DataFrame, int, bool, Tuple[float, float, int, float]]]:
        """Decode one ``.datZ`` file and determine whether a gap precedes it.

        Args:
            file_path:          Path to the ``.datZ`` file.
            utc_start:          UTC epoch of this file.
            next_file_utc:      UTC epoch of the next file (for duration
                                inference), or ``None``.
            active_span:        Current open span
                                ``(start, end, row_count, last_event_ts)``
                                or ``None``.
            prev_last_event_ts: UTC epoch of the last real event in the
                                preceding file; used to position gap markers.

        Returns:
            ``(events_df, row_count, gap_opened, new_span_tuple)`` or ``None``
            on decode error.
        """
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            print(f"Error reading {file_path.name}: {exc}")
            return None

        try:
            df = decoders.parse_datz_bytes(raw_bytes, utc_start)
        except decoders.DatZDecodingError as exc:
            print(f"Error decoding {file_path.name}: {exc}")
            return None

        row_count     = len(df)
        last_event_ts = df["timestamp"].max() if not df.empty else utc_start

        gap_opened = False
        if active_span is not None:
            prev_end  = active_span[1]
            duration  = self._calculate_duration(utc_start, last_event_ts,
                                                 next_file_utc)
            expected  = prev_end + duration
            if utc_start - expected > 5.0:
                marker_ts = (
                    prev_last_event_ts + 0.1
                    if prev_last_event_ts is not None
                    else prev_end + 0.1
                )
                df = decoders.insert_gap_marker(df, marker_ts)
                self._gap_markers += 1
                gap_opened = True

        new_span: Tuple[float, float, int, float] = (
            utc_start, utc_start, row_count, last_event_ts
        )
        return df, row_count, gap_opened, new_span

    # ------------------------------------------------------------------
    # Duration inference
    # ------------------------------------------------------------------

    def _calculate_duration(
        self,
        start_ts: float,
        last_event_ts: float,
        next_file_start_ts: Optional[float],
    ) -> float:
        """Infer file duration (seconds) to determine the expected file end.

        Three signals favour Grid Mode (15-min boundary):
            1. Data span > 61 s
            2. Gap to next file > 90 s
            3. Start minute aligns with 0/15/30/45

        Falls back to 60 s (1-Minute Mode) otherwise.

        Args:
            start_ts:           UTC epoch of file start.
            last_event_ts:      UTC epoch of the last event in the file.
            next_file_start_ts: UTC epoch of the next file, or ``None``.

        Returns:
            Duration in seconds.
        """
        if last_event_ts - start_ts > 61.0:
            return self._grid_duration(start_ts)
        if next_file_start_ts is not None and next_file_start_ts - start_ts > 90.0:
            return self._grid_duration(start_ts)
        local_dt = datetime.fromtimestamp(start_ts, self.tz)
        if local_dt.minute in (0, 15, 30, 45):
            return self._grid_duration(start_ts)
        return 60.0

    def _grid_duration(self, start_ts: float) -> float:
        """Compute seconds from ``start_ts`` to the next 15-minute boundary.

        Args:
            start_ts: UTC epoch.

        Returns:
            Seconds to the next quarter-hour mark.
        """
        local_dt = datetime.fromtimestamp(start_ts, self.tz)
        minute   = local_dt.minute
        nxt      = next((q for q in (0, 15, 30, 45) if minute < q), None)
        if nxt is None:
            next_dt = (
                local_dt.replace(minute=0, second=0, microsecond=0)
                + timedelta(hours=1)
            )
        else:
            next_dt = local_dt.replace(minute=nxt, second=0, microsecond=0)
        return (next_dt - local_dt).total_seconds()

    # ------------------------------------------------------------------
    # DB commit
    # ------------------------------------------------------------------

    def _commit_batch(
        self,
        events_buffer: List[pd.DataFrame],
        span_updates: List[Tuple[float, float, int]],
        fill_gaps: bool = False,
    ) -> None:
        """Commit events and span log rows in a single transaction.

        When ``fill_gaps=True``, after upserting spans, calls
        :meth:`_merge_adjacent_spans` to coalesce any spans that the newly
        written data has made contiguous.

        Args:
            events_buffer: List of event DataFrames to insert.
            span_updates:  List of ``(span_start, span_end, row_count)`` tuples.
            fill_gaps:     When ``True`` trigger span-merge logic post-commit.
        """
        event_tuples = []
        if events_buffer:
            all_events   = pd.concat(events_buffer, ignore_index=True)
            event_tuples = list(all_events.itertuples(index=False, name=None))

        now_iso = datetime.utcnow().isoformat()

        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            try:
                if event_tuples:
                    cur.executemany(
                        "INSERT OR IGNORE INTO events "
                        "(timestamp, event_code, parameter) VALUES (?, ?, ?)",
                        event_tuples,
                    )
                for span_start, span_end, row_count in span_updates:
                    cur.execute("""
                        INSERT INTO ingestion_log
                            (span_start, span_end, processed_at, row_count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(span_start) DO UPDATE SET
                            span_end     = excluded.span_end,
                            processed_at = excluded.processed_at,
                            row_count    = row_count + excluded.row_count
                    """, (span_start, span_end, now_iso, row_count))
                m.conn.commit()
            except sqlite3.Error as exc:
                m.conn.rollback()
                print(f"Error committing batch: {exc}")
                raise

        if fill_gaps:
            self._merge_adjacent_spans()

    # ------------------------------------------------------------------
    # Span merging (Gap Fill path only)
    # ------------------------------------------------------------------

    def _merge_adjacent_spans(self) -> None:
        """Coalesce any overlapping or touching spans in ``ingestion_log``.

        After a gap-fill commit, two previously separate spans may now be
        contiguous (the gap between them is gone).  This method sweeps the
        full span list and collapses any that touch or overlap into one row
        (keeping ``span_start = MIN``, ``span_end = MAX``,
        ``row_count = SUM``).

        The merge is idempotent: re-running it on an already-clean log is a
        no-op.
        """
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute(
                "SELECT span_start, span_end, row_count "
                "FROM ingestion_log ORDER BY span_start"
            )
            rows = cur.fetchall()

        if len(rows) < 2:
            return

        # Sweep and coalesce.
        merged: List[Tuple[float, float, int]] = [rows[0]]
        for span_start, span_end, row_count in rows[1:]:
            ps, pe, pr = merged[-1]
            # Two spans are adjacent/overlapping if the next starts at or
            # before the current end (no events-table gap between them).
            if span_start <= pe:
                merged[-1] = (ps, max(pe, span_end), pr + row_count)
            else:
                merged.append((span_start, span_end, row_count))

        if len(merged) == len(rows):
            return  # nothing to do

        now_iso = datetime.utcnow().isoformat()
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            try:
                cur.execute("DELETE FROM ingestion_log")
                cur.executemany(
                    "INSERT INTO ingestion_log "
                    "(span_start, span_end, processed_at, row_count) "
                    "VALUES (?, ?, ?, ?)",
                    [(s, e, now_iso, r) for s, e, r in merged],
                )
                m.conn.commit()
            except sqlite3.Error as exc:
                m.conn.rollback()
                raise RuntimeError(f"Error merging spans: {exc}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def run_ingestion(
    db_path: Path,
    data_dir: Path,
    timezone: str = None,
    fill_gaps: bool = False,
    batch_size: int = 50,
    run_cycles: bool = True,
) -> None:
    """Run ingestion of ``.datZ`` files into the database.

    When ``run_cycles=True`` (the default) a ``CycleProcessor`` is created and
    wired into the engine so cycles are updated incrementally as each batch is
    committed.  The same ``fill_gaps`` flag is forwarded to the processor.

    Args:
        db_path:    Path to the SQLite database.
        data_dir:   Directory containing ``.datZ`` files.
        timezone:   Controller timezone.  ``None`` reads from metadata.
        fill_gaps:  When ``True`` scan for and ingest historical gaps; use
                    Path B (Gap Fill) for cycle processing.
        batch_size: Files per transaction commit.
        run_cycles: Wire a ``CycleProcessor`` for live cycle updates.
    """
    cycle_processor = None
    if run_cycles:
        from .processing import CycleProcessor  # deferred to avoid circular import
        cycle_processor = CycleProcessor(db_path, timezone=timezone)

    engine = IngestionEngine(
        db_path, data_dir, timezone,
        cycle_processor=cycle_processor,
    )
    engine.run(fill_gaps=fill_gaps, batch_size=batch_size)

    stats = engine.get_ingestion_stats()
    print("\nIngestion Complete!")
    print(f"  Files processed : {stats['files_processed']}")
    print(f"  Total events    : {stats['total_events']:,}")
    print(f"  Gap markers     : {stats['gap_markers']}")
    print(f"  Log spans       : {stats['span_count']}")
    if stats["date_range"]["start"]:
        print(
            f"  Date range      : {stats['date_range']['start']} "
            f"→ {stats['date_range']['end']}"
        )
