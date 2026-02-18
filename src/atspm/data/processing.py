"""
ATSPM Cycle Processing Engine (Imperative Shell)

This module orchestrates cycle detection on the SQLite database.
It manages I/O, transactions, and calls the Functional Core for logic.

Package Location: src/atspm/data/processing.py

Timezone Notes:
    All timestamps in the DB are UTC epochs. Date-level operations (identifying
    which calendar day to process, slicing events for a day, saving/deleting
    cycles) must convert UTC epochs to local time before applying date math.
    SQLite's built-in DATE(ts, 'unixepoch') always returns the UTC date, which
    is wrong for intersections in non-UTC timezones. We do date math in Python
    using pytz-aware datetimes, then pass UTC epoch bounds to SQL.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, date, time
from typing import List, Optional, Tuple

import pandas as pd
import pytz

from .manager import DatabaseManager
from ..analysis.cycles import calculate_cycles, CycleDetectionError


class CycleProcessor:
    """
    Manages cycle detection processing on the database.

    Responsibilities:
    - Create and manage cycles table
    - Identify unprocessed LOCAL dates (not UTC dates)
    - Fetch appropriate config for each date
    - Call Functional Core for cycle detection
    - Persist results with transaction safety
    """

    def __init__(self, db_path: Path, timezone: str = None):
        """
        Initialize the cycle processor.

        Args:
            db_path: Path to SQLite database
            timezone: Local timezone string (e.g., 'US/Mountain'). If None,
                      reads from the metadata table, falling back to 'US/Mountain'.
        """
        self.db_path = Path(db_path)
        self.tz = pytz.timezone(self._resolve_timezone(timezone))
        self._init_cycles_table()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_timezone(self, timezone: Optional[str]) -> str:
        """
        Resolve timezone from argument, metadata table, or default.

        Args:
            timezone: Caller-supplied timezone string or None.

        Returns:
            Resolved timezone string.
        """
        if timezone is not None:
            return timezone

        try:
            with DatabaseManager(self.db_path) as manager:
                meta = manager.get_metadata()
                if meta and meta.get('timezone'):
                    tz_str = meta['timezone']
                    print(f"CycleProcessor: using timezone from metadata: {tz_str}")
                    return tz_str
        except Exception:
            pass

        print("CycleProcessor: no timezone found, defaulting to US/Mountain")
        return 'US/Mountain'

    def _init_cycles_table(self) -> None:
        """
        Create cycles table and index if they do not already exist.
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_start REAL PRIMARY KEY,
                    coord_plan INTEGER NOT NULL,
                    detection_method TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cycles_start
                ON cycles (cycle_start)
            """)
            manager.conn.commit()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, reprocess: bool = False) -> None:
        """
        Process all unprocessed local dates in the events table.

        Args:
            reprocess: If True, delete and reprocess all dates.
        """
        dates_to_process = self._get_dates_to_process(reprocess)

        if not dates_to_process:
            print("No dates to process")
            return

        print(f"Processing {len(dates_to_process)} dates...")
        successful = 0
        failed = 0

        for date_str in dates_to_process:
            try:
                self._process_day(date_str)
                successful += 1
            except Exception as e:
                failed += 1
                print(f"ERROR processing {date_str}: {e}")

        print(f"\nCycle Processing Complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

    # ------------------------------------------------------------------
    # Date identification (local-time aware)
    # ------------------------------------------------------------------

    def _get_dates_to_process(self, reprocess: bool) -> List[str]:
        """
        Identify which LOCAL calendar dates need cycle processing.

        SQLite's DATE(ts, 'unixepoch') returns UTC dates, which is wrong for
        non-UTC intersections. Instead we pull the raw epoch bounds from
        ingestion_log (cheap — thousands of rows) and compute local dates in
        Python.

        Args:
            reprocess: If True, return all dates that have events.

        Returns:
            Sorted list of date strings in 'YYYY-MM-DD' (local) format.
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()

            # Use ingestion_log per the Performance Rule: it's tiny compared
            # to events (a handful of span rows vs. millions of event rows).
            # Query the earliest span_start and latest span_end.
            cursor.execute(
                "SELECT MIN(span_start), MAX(span_end) FROM ingestion_log"
            )
            row = cursor.fetchone()

        if not row or row[0] is None:
            return []

        min_epoch, max_epoch = row

        # Build the full set of local dates covered by the ingestion log.
        min_local = datetime.fromtimestamp(min_epoch, self.tz).date()
        max_local = datetime.fromtimestamp(max_epoch, self.tz).date()

        all_local_dates = set()
        current = min_local
        while current <= max_local:
            all_local_dates.add(current)
            current += timedelta(days=1)

        if reprocess:
            candidate_dates = all_local_dates
        else:
            # Exclude dates that already have cycles.
            processed_dates = self._get_processed_local_dates()
            candidate_dates = all_local_dates - processed_dates

        return sorted(d.strftime('%Y-%m-%d') for d in candidate_dates)

    def _get_processed_local_dates(self) -> set:
        """
        Return the set of local dates for which cycles already exist.

        Returns:
            Set of datetime.date objects (local timezone).
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            cursor.execute("SELECT DISTINCT cycle_start FROM cycles")
            rows = cursor.fetchall()

        return {
            datetime.fromtimestamp(row[0], self.tz).date()
            for row in rows
        }

    # ------------------------------------------------------------------
    # Per-day processing
    # ------------------------------------------------------------------

    def _process_day(self, date_str: str) -> None:
        """
        Process a single local calendar day's worth of events.

        Args:
            date_str: Local date in 'YYYY-MM-DD' format.
        """
        try:
            local_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}': {e}")

        # Epoch bounds for this local calendar day
        start_epoch, end_epoch = self._local_day_to_utc_epoch_range(local_date)

        # Fetch config active on this date
        config = self._get_config_for_date(
            datetime.combine(local_date, time.min)
        )
        if config is None:
            raise ValueError(f"No configuration found for {date_str}")

        # Fetch events for the UTC epoch window that covers this local day
        events_df = self._get_events_for_epoch_range(start_epoch, end_epoch)

        if events_df.empty:
            print(f"Processed {date_str}: No events found")
            return

        # Functional Core
        try:
            cycles_df = calculate_cycles(events_df, config)
        except CycleDetectionError as e:
            raise CycleDetectionError(
                f"Cycle detection failed for {date_str}: {e}"
            )

        if cycles_df.empty:
            print(f"Processed {date_str}: No cycles detected")
            return

        # Persist
        self._save_cycles(cycles_df, start_epoch, end_epoch)

        detection_method = cycles_df['detection_method'].iloc[0]
        print(
            f"Processed {date_str}: Found {len(cycles_df)} cycles "
            f"using {detection_method}"
        )

    def _local_day_to_utc_epoch_range(
        self, local_date: date
    ) -> Tuple[float, float]:
        """
        Convert a local calendar date to UTC epoch [start, end) bounds.

        Args:
            local_date: A datetime.date in the intersection's local timezone.

        Returns:
            (start_epoch, end_epoch) as UTC floats — midnight-to-midnight local.
        """
        local_midnight_start = self.tz.localize(
            datetime.combine(local_date, time.min)
        )
        local_midnight_end = local_midnight_start + timedelta(days=1)

        return local_midnight_start.timestamp(), local_midnight_end.timestamp()

    # ------------------------------------------------------------------
    # DB fetch / save helpers
    # ------------------------------------------------------------------

    def _get_config_for_date(self, date: datetime) -> Optional[dict]:
        """
        Get the configuration active on a specific date.

        Args:
            date: Naive or aware datetime; only the date portion is used.

        Returns:
            Configuration dict or None if not found.
        """
        with DatabaseManager(self.db_path) as manager:
            return manager.get_config_at_date(date)

    def _get_events_for_epoch_range(
        self, start_epoch: float, end_epoch: float
    ) -> pd.DataFrame:
        """
        Fetch all events (including gap markers) within a UTC epoch range.

        Args:
            start_epoch: Start of range (UTC epoch, inclusive).
            end_epoch:   End of range (UTC epoch, exclusive).

        Returns:
            DataFrame with columns [timestamp, event_code, parameter].
        """
        with DatabaseManager(self.db_path) as manager:
            return manager.query_events(
                start_time=start_epoch,
                end_time=end_epoch
            )

    def _save_cycles(
        self,
        cycles_df: pd.DataFrame,
        start_epoch: float,
        end_epoch: float
    ) -> None:
        """
        Persist cycles for a local day, deleting any prior records first.

        Idempotency: existing cycles whose cycle_start falls within
        [start_epoch, end_epoch) are deleted before the new ones are inserted.

        Args:
            cycles_df:   Cycles DataFrame with columns [cycle_start, coord_plan,
                         detection_method].
            start_epoch: UTC epoch for start of local day (inclusive).
            end_epoch:   UTC epoch for end of local day (exclusive).
        """
        cycle_records = [
            (
                row['cycle_start'],
                int(row['coord_plan']),
                row['detection_method']
            )
            for _, row in cycles_df.iterrows()
        ]

        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            try:
                cursor.execute(
                    "DELETE FROM cycles WHERE cycle_start >= ? AND cycle_start < ?",
                    (start_epoch, end_epoch)
                )
                cursor.executemany(
                    "INSERT INTO cycles (cycle_start, coord_plan, detection_method) "
                    "VALUES (?, ?, ?)",
                    cycle_records
                )
                manager.conn.commit()
            except sqlite3.Error as e:
                manager.conn.rollback()
                raise RuntimeError(f"Database error saving cycles: {e}")

    # ------------------------------------------------------------------
    # Statistics and validation (timezone-aware)
    # ------------------------------------------------------------------

    def get_processing_stats(self) -> dict:
        """
        Get statistics about cycle processing.

        Returns:
            Dictionary with keys: total_cycles, days_processed,
            unprocessed_days, detection_methods, coord_plans, date_range.
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM cycles")
            total_cycles = cursor.fetchone()[0]

            cursor.execute(
                "SELECT detection_method, COUNT(*) FROM cycles "
                "GROUP BY detection_method"
            )
            method_counts = dict(cursor.fetchall())

            cursor.execute(
                "SELECT coord_plan, COUNT(*) FROM cycles "
                "GROUP BY coord_plan"
            )
            coord_counts = dict(cursor.fetchall())

            cursor.execute(
                "SELECT MIN(cycle_start), MAX(cycle_start) FROM cycles"
            )
            min_ts, max_ts = cursor.fetchone()

            # Derive local-date counts from epoch values rather than
            # SQLite's UTC-biased DATE() function.
            cursor.execute("SELECT DISTINCT cycle_start FROM cycles")
            cycle_epochs = [r[0] for r in cursor.fetchall()]

        days_processed = len({
            datetime.fromtimestamp(ts, self.tz).date()
            for ts in cycle_epochs
        })

        all_dates = self._get_dates_to_process(reprocess=True)
        processed_dates = self._get_processed_local_dates()
        unprocessed_days = len(set(all_dates) - {
            d.strftime('%Y-%m-%d') for d in processed_dates
        })

        return {
            'total_cycles': total_cycles,
            'days_processed': days_processed,
            'unprocessed_days': unprocessed_days,
            'detection_methods': method_counts,
            'coord_plans': coord_counts,
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

    def get_cycle_summary_for_date(self, date_str: str) -> Optional[dict]:
        """
        Get cycle summary for a specific local calendar date.

        Args:
            date_str: Local date in 'YYYY-MM-DD' format.

        Returns:
            Dictionary with cycle statistics or None if no data.
        """
        try:
            local_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return None

        start_epoch, end_epoch = self._local_day_to_utc_epoch_range(local_date)

        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*)           AS cycle_count,
                    MIN(coord_plan)    AS min_coord,
                    MAX(coord_plan)    AS max_coord,
                    detection_method
                FROM cycles
                WHERE cycle_start >= ? AND cycle_start < ?
                GROUP BY detection_method
                """,
                (start_epoch, end_epoch)
            )
            results = cursor.fetchall()

        if not results:
            return None

        return {
            'date': date_str,
            'cycle_count': results[0][0],
            'coord_plan_range': (results[0][1], results[0][2]),
            'detection_method': results[0][3]
        }

    def reprocess_date(self, date_str: str) -> bool:
        """
        Reprocess a specific local calendar date.

        Args:
            date_str: Local date in 'YYYY-MM-DD' format.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self._process_day(date_str)
            return True
        except Exception as e:
            print(f"Error reprocessing {date_str}: {e}")
            return False

    def validate_cycles_table(self) -> Tuple[bool, List[str]]:
        """
        Validate the cycles table for data quality issues.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues = []

        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()

            cursor.execute(
                "SELECT cycle_start, COUNT(*) FROM cycles "
                "GROUP BY cycle_start HAVING COUNT(*) > 1"
            )
            duplicates = cursor.fetchall()
            if duplicates:
                issues.append(
                    f"Found {len(duplicates)} duplicate cycle_start timestamps"
                )

            cursor.execute("""
                SELECT
                    MIN(next_start - cycle_start) AS min_length,
                    MAX(next_start - cycle_start) AS max_length
                FROM (
                    SELECT
                        cycle_start,
                        LEAD(cycle_start) OVER (ORDER BY cycle_start) AS next_start
                    FROM cycles
                )
                WHERE next_start IS NOT NULL
            """)
            result = cursor.fetchone()
            if result and result[0] is not None:
                min_length, max_length = result
                if min_length < 10.0:
                    issues.append(
                        f"Found cycle length < 10 s (minimum: {min_length:.1f} s)"
                    )
                if max_length > 300.0:
                    issues.append(
                        f"Found cycle length > 300 s (maximum: {max_length:.1f} s)"
                    )

            cursor.execute(
                "SELECT COUNT(*) FROM cycles "
                "WHERE coord_plan IS NULL OR detection_method IS NULL"
            )
            null_count = cursor.fetchone()[0]
            if null_count > 0:
                issues.append(f"Found {null_count} rows with NULL values")

        return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def run_cycle_processing(
    db_path: Path,
    reprocess: bool = False,
    timezone: str = None
) -> None:
    """
    Convenience function to run cycle processing.

    Args:
        db_path:   Path to SQLite database.
        reprocess: If True, reprocess all dates.
        timezone:  Local timezone string. If None, reads from metadata table.
    """
    processor = CycleProcessor(db_path, timezone=timezone)
    processor.run(reprocess=reprocess)

    stats = processor.get_processing_stats()
    print("\nCycle Processing Statistics:")
    print(f"  Total cycles:      {stats['total_cycles']:,}")
    print(f"  Days processed:    {stats['days_processed']}")
    print(f"  Unprocessed days:  {stats['unprocessed_days']}")
    print(f"  Detection methods: {stats['detection_methods']}")
    print(f"  Coordination plans:{stats['coord_plans']}")
    if stats['date_range']['start']:
        print(
            f"  Date range:        "
            f"{stats['date_range']['start']} to {stats['date_range']['end']}"
        )