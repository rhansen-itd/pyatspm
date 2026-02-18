"""
ATSPM Data Ingestion Engine (Imperative Shell)

Handles ingestion of .datZ files into the SQLite database.
Manages file I/O, state tracking, timezone conversions, and gap detection.

ingestion_log schema (span-based):
    span_start  REAL PRIMARY KEY  -- UTC epoch of first file in span
    span_end    REAL NOT NULL     -- UTC epoch of last file in span
    processed_at TEXT NOT NULL    -- ISO timestamp of last write to this span
    row_count   INTEGER NOT NULL  -- total events ingested for the span

A "span" is a maximal run of consecutive files with no gap between them.
When a gap is detected, the current span is closed and a new one begins.
The gap marker event (-1, -1) in the events table remains the authoritative
record of discontinuity; the log is purely for ingestion-state bookkeeping.

Package Location: src/atspm/data/ingestion.py
"""

import re
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd
import pytz

from ..analysis import decoders
from .manager import DatabaseManager


class IngestionEngine:
    """
    Manages incremental ingestion of .datZ files into the SQLite database.

    Responsibilities:
    - File scanning and filename-timestamp parsing (local â†’ UTC)
    - Gap detection and gap-marker insertion into events
    - Continuous-span tracking in ingestion_log
    - Batch processing with transaction safety
    """

    def __init__(
        self,
        db_path: Path,
        raw_data_dir: Path,
        timezone: str = None
    ):
        """
        Args:
            db_path: Path to the SQLite database.
            raw_data_dir: Directory containing .datZ files.
            timezone: Controller local timezone (e.g. 'US/Mountain').
                      If None, reads from the metadata table.
        """
        self.db_path = Path(db_path)
        self.raw_data_dir = Path(raw_data_dir)
        self.tz = pytz.timezone(self._resolve_timezone(timezone))

        # Running counters for summary output
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
                if meta and meta.get('timezone'):
                    print(f"IngestionEngine: using timezone from metadata: {meta['timezone']}")
                    return meta['timezone']
        except Exception:
            pass
        print("IngestionEngine: no timezone found, defaulting to US/Mountain")
        return 'US/Mountain'

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, incremental: bool = True, batch_size: int = 50) -> None:
        """
        Execute the ingestion process.

        Args:
            incremental: If True, skip files already covered by ingestion_log spans.
            batch_size: Number of files to buffer before committing a transaction.
        """
        last_ts = self._get_last_ingested_timestamp() if incremental else None

        if last_ts:
            print(f"Resuming from: {datetime.fromtimestamp(last_ts, self.tz)}")
        else:
            print("Starting fresh ingestion")

        file_list = self._scan_files(last_ts)

        if not file_list:
            print("No new files to process")
            return

        print(f"Found {len(file_list)} files to process")
        self._process_batches(file_list, batch_size)

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics from the current run and the full log.

        Returns:
            Dict with keys: files_processed, total_events, gap_markers,
            date_range {start, end}, span_count.
        """
        with DatabaseManager(self.db_path) as m:
            cursor = m.conn.cursor()
            cursor.execute(
                "SELECT COUNT(*), MIN(span_start), MAX(span_end) FROM ingestion_log"
            )
            span_count, min_ts, max_ts = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) FROM events WHERE event_code = -1")
            gap_count = cursor.fetchone()[0]

        return {
            'files_processed': self._files_processed,
            'total_events': self._total_events,
            'gap_markers': gap_count,
            'span_count': span_count or 0,
            'date_range': {
                'start': (
                    datetime.fromtimestamp(min_ts, self.tz).isoformat()
                    if min_ts else None
                ),
                'end': (
                    datetime.fromtimestamp(max_ts, self.tz).isoformat()
                    if max_ts else None
                ),
            }
        }

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_last_ingested_timestamp(self) -> Optional[float]:
        """
        Return the span_end of the most recently ingested span.

        Returns:
            UTC epoch float or None if log is empty.
        """
        with DatabaseManager(self.db_path) as m:
            cursor = m.conn.cursor()
            cursor.execute("SELECT MAX(span_end) FROM ingestion_log")
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else None

    def _get_current_span(self) -> Optional[Tuple[float, float, int]]:
        """
        Return (span_start, span_end, row_count) for the most recent span,
        or None if no spans exist.
        """
        with DatabaseManager(self.db_path) as m:
            cursor = m.conn.cursor()
            cursor.execute("""
                SELECT span_start, span_end, row_count
                FROM ingestion_log
                ORDER BY span_start DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            return tuple(row) if row else None

    # ------------------------------------------------------------------
    # File scanning
    # ------------------------------------------------------------------

    def _scan_files(
        self,
        min_timestamp: Optional[float]
    ) -> List[Tuple[Path, float]]:
        """
        Scan raw_data_dir for .datZ files newer than min_timestamp.

        Args:
            min_timestamp: Exclude files at or before this UTC epoch.

        Returns:
            List of (filepath, utc_epoch) sorted by timestamp.
        """
        file_list = []
        for fp in self.raw_data_dir.glob('*.datZ'):
            ts = self._parse_filename_timestamp(fp.name)
            if ts is None:
                print(f"Warning: cannot parse timestamp from {fp.name}")
                continue
            if min_timestamp is not None and ts <= min_timestamp:
                continue
            file_list.append((fp, ts))

        file_list.sort(key=lambda x: x[1])
        return file_list

    def _parse_filename_timestamp(self, filename: str) -> Optional[float]:
        """
        Extract the local datetime from the filename and convert to UTC epoch.

        Expected format: *_YYYY_MM_DD_HHMM.datZ

        Args:
            filename: .datZ filename.

        Returns:
            UTC epoch float, or None on parse failure.
        """
        match = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{4})', filename)
        if not match:
            return None
        try:
            year, month, day, hhmm = match.groups()
            naive_dt = datetime(int(year), int(month), int(day),
                                int(hhmm[:2]), int(hhmm[2:]))
            aware_dt = self.tz.localize(naive_dt)
            return aware_dt.timestamp()
        except (ValueError,
                pytz.exceptions.AmbiguousTimeError,
                pytz.exceptions.NonExistentTimeError) as e:
            print(f"Warning: error parsing {filename}: {e}")
            return None

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batches(
        self,
        file_list: List[Tuple[Path, float]],
        batch_size: int
    ) -> None:
        """
        Process file_list in batches, committing every batch_size files.

        Maintains a running span state so that consecutive files collapse
        into a single ingestion_log row and gaps open a new span.

        Args:
            file_list: Sorted list of (filepath, utc_epoch) tuples.
            batch_size: Max files per transaction.
        """
        total = len(file_list)

        # Seed the active span from the last committed span in the log.
        # This lets us correctly determine whether the first new file is
        # continuous with what was previously ingested.
        existing_span = self._get_current_span()
        # active_span: (span_start, span_end, accumulated_row_count, last_event_ts)
        # last_event_ts is the UTC epoch of the final real event in the span,
        # used to place gap markers at last_event_ts + 0.1 rather than at a
        # computed boundary.  For spans seeded from the DB we have no event-level
        # detail, so we fall back to span_end as an approximation.
        # None means no open span yet this run.
        active_span: Optional[Tuple[float, float, int, float]] = (
            (existing_span[0], existing_span[1], existing_span[2], existing_span[1])
            if existing_span is not None else None
        )

        events_buffer: List[pd.DataFrame] = []
        span_updates: List[Tuple[float, float, int]] = []  # (start, end, rows)

        for i, (file_path, utc_start) in enumerate(file_list):
            next_file_utc = file_list[i + 1][1] if i + 1 < total else None

            # Pass the previous span's last_event_ts so _process_file can place
            # the gap marker at last_event_ts + 0.1 rather than a computed boundary.
            prev_last_event_ts = active_span[3] if active_span is not None else None

            result = self._process_file(file_path, utc_start, next_file_utc,
                                        active_span, prev_last_event_ts)
            if result is None:
                continue

            events_df, row_count, gap_opened, new_span = result

            self._files_processed += 1
            self._total_events += row_count

            if not events_df.empty:
                events_buffer.append(events_df)

            if gap_opened and active_span is not None:
                # Close the current span before this file's gap marker.
                # Only the first three fields are stored in the DB.
                span_updates.append(active_span[:3])
                active_span = new_span
            elif active_span is None:
                active_span = new_span
            else:
                # Extend active span: update end, accumulate rows, keep last_event_ts
                # from new_span (index 3) so it always reflects the most recent file.
                active_span = (
                    active_span[0],
                    utc_start,
                    active_span[2] + row_count,
                    new_span[3],
                )

            should_commit = (len(span_updates) + 1 >= batch_size or
                             i == total - 1)

            if should_commit and (events_buffer or span_updates or
                                  active_span is not None):
                # Include the currently-open span in this commit.
                # _commit_batch expects 3-tuples; strip last_event_ts (index 3).
                all_spans = span_updates[:]
                if active_span is not None:
                    all_spans.append(active_span[:3])

                self._commit_batch(events_buffer, all_spans)
                print(f"  Committed {i + 1}/{total} files...")

                events_buffer = []
                span_updates = []
                # active_span persists â€” it may still grow

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
        """
        Decode a single .datZ file and decide whether a gap precedes it.

        Args:
            file_path:          Path to the .datZ file.
            utc_start:          UTC epoch of this file.
            next_file_utc:      UTC epoch of the next file (for duration inference).
            active_span:        Current open span (start, end, row_count,
                                last_event_ts) or None.
            prev_last_event_ts: UTC epoch of the last real event in the
                                preceding file (or span_end for DB-seeded spans).
                                Used to position the gap marker at
                                prev_last_event_ts + 0.1 s.

        Returns:
            (events_df, row_count, gap_opened, new_span_tuple) or None on error.
            new_span_tuple is a 4-tuple (span_start, span_end, row_count,
            last_event_ts).
            gap_opened is True when the file follows a gap that closes
            active_span.
        """
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as e:
            print(f"Error reading {file_path.name}: {e}")
            return None

        try:
            df = decoders.parse_datz_bytes(raw_bytes, utc_start)
        except decoders.DatZDecodingError as e:
            print(f"Error decoding {file_path.name}: {e}")
            return None

        row_count = len(df)
        last_event_ts = df['timestamp'].max() if not df.empty else utc_start

        # Gap detection: does this file start a new span?
        gap_opened = False
        if active_span is not None:
            prev_end = active_span[1]
            duration = self._calculate_duration(utc_start, last_event_ts,
                                                next_file_utc)
            expected_next = prev_end + duration

            gap_seconds = utc_start - expected_next
            if gap_seconds > 5.0:
                # Place the gap marker 0.1 s after the last real event of the
                # preceding file.  This correctly handles the case where a file
                # ends early (e.g. power loss at minute 3 of a 15-min bin) and
                # the controller restarts mid-bin: the marker lands immediately
                # after meaningful data rather than at the hypothetical bin end,
                # which would silently absorb valid events from the recovery file.
                gap_marker_ts = (
                    prev_last_event_ts + 0.1
                    if prev_last_event_ts is not None
                    else prev_end + 0.1
                )
                df = decoders.insert_gap_marker(df, gap_marker_ts)
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
        next_file_start_ts: Optional[float]
    ) -> float:
        """
        Infer file duration (seconds) to determine the expected end of a file.

        Three signals favour Grid Mode (15-min boundary):
        1. Data span > 61 s
        2. Gap to next file > 90 s
        3. Start minute aligns with 0/15/30/45

        Falls back to 60 s (1-Minute Mode) if none apply.

        Args:
            start_ts: UTC epoch of file start.
            last_event_ts: UTC epoch of last event in file.
            next_file_start_ts: UTC epoch of next file, or None.

        Returns:
            Duration in seconds.
        """
        if last_event_ts - start_ts > 61.0:
            return self._grid_duration(start_ts)

        if next_file_start_ts is not None:
            if next_file_start_ts - start_ts > 90.0:
                return self._grid_duration(start_ts)

        local_dt = datetime.fromtimestamp(start_ts, self.tz)
        if local_dt.minute in (0, 15, 30, 45):
            return self._grid_duration(start_ts)

        return 60.0

    def _grid_duration(self, start_ts: float) -> float:
        """
        Compute seconds from start_ts to the next 15-minute boundary in local time.

        Args:
            start_ts: UTC epoch.

        Returns:
            Seconds to the next quarter-hour mark.
        """
        local_dt = datetime.fromtimestamp(start_ts, self.tz)
        current_minute = local_dt.minute

        next_quarter = next(
            (qh for qh in (0, 15, 30, 45) if current_minute < qh),
            None
        )

        if next_quarter is None:
            next_dt = (local_dt.replace(minute=0, second=0, microsecond=0)
                       + timedelta(hours=1))
        else:
            next_dt = local_dt.replace(minute=next_quarter, second=0,
                                       microsecond=0)

        return (next_dt - local_dt).total_seconds()

    # ------------------------------------------------------------------
    # DB commit
    # ------------------------------------------------------------------

    def _commit_batch(
        self,
        events_buffer: List[pd.DataFrame],
        span_updates: List[Tuple[float, float, int]]
    ) -> None:
        """
        Commit events and span log rows in a single transaction.

        For each span in span_updates, upsert into ingestion_log using
        INSERT OR REPLACE so that an existing open span gets its span_end
        and row_count updated in place rather than creating a duplicate.

        Args:
            events_buffer: List of event DataFrames to insert.
            span_updates:  List of (span_start, span_end, row_count) tuples.
        """
        event_tuples = []
        if events_buffer:
            all_events = pd.concat(events_buffer, ignore_index=True)
            event_tuples = list(all_events.itertuples(index=False, name=None))

        now_iso = datetime.utcnow().isoformat()

        with DatabaseManager(self.db_path) as m:
            cursor = m.conn.cursor()
            try:
                if event_tuples:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO events "
                        "(timestamp, event_code, parameter) VALUES (?, ?, ?)",
                        event_tuples
                    )

                for span_start, span_end, row_count in span_updates:
                    cursor.execute("""
                        INSERT INTO ingestion_log
                            (span_start, span_end, processed_at, row_count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(span_start) DO UPDATE SET
                            span_end     = excluded.span_end,
                            processed_at = excluded.processed_at,
                            row_count    = row_count + excluded.row_count
                    """, (span_start, span_end, now_iso, row_count))

                m.conn.commit()
            except sqlite3.Error as e:
                m.conn.rollback()
                print(f"Error committing batch: {e}")
                raise


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def run_ingestion(
    db_path: Path,
    data_dir: Path,
    timezone: str = None,
    incremental: bool = True,
    batch_size: int = 50
) -> None:
    """
    Run ingestion of .datZ files into the database.

    Args:
        db_path:     Path to SQLite database.
        data_dir:    Directory containing .datZ files.
        timezone:    Controller timezone. If None, reads from metadata table.
        incremental: Only process files newer than the last ingested span_end.
        batch_size:  Files per transaction commit.
    """
    engine = IngestionEngine(db_path, data_dir, timezone)
    engine.run(incremental=incremental, batch_size=batch_size)

    stats = engine.get_ingestion_stats()
    print("\nIngestion Complete!")
    print(f"  Files processed : {stats['files_processed']}")
    print(f"  Total events    : {stats['total_events']:,}")
    print(f"  Gap markers     : {stats['gap_markers']}")
    print(f"  Log spans       : {stats['span_count']}")
    if stats['date_range']['start']:
        print(
            f"  Date range      : {stats['date_range']['start']} "
            f"â†’ {stats['date_range']['end']}"
        )