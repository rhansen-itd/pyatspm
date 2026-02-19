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
from ..analysis.cycles import (
    calculate_cycles,
    assign_ring_phases,
    CycleDetectionError,
)

# A date is considered stale (needs reprocessing) when the latest ingested
# data for that local date extends more than this many seconds past the
# latest cycle_start on that date.  One full max-cycle-length (300 s) gives
# enough headroom to avoid re-triggering on a stray late event, while still
# catching a genuine new half-day of data.
_STALE_THRESHOLD_SECONDS: int = 300


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

        If the table already exists but is missing the ring-phase columns
        (r1_phases, r2_phases) from an older schema, they are added via
        ALTER TABLE so existing rows are preserved.  Callers can then run
        backfill_ring_phases() to populate those columns without reingesting.
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()

            # Create table with full schema (IF NOT EXISTS is a no-op on
            # existing DBs so we handle migration separately below)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_start      REAL PRIMARY KEY,
                    coord_plan       REAL NOT NULL DEFAULT 0,
                    detection_method TEXT NOT NULL DEFAULT '',
                    r1_phases        TEXT NOT NULL DEFAULT 'None',
                    r2_phases        TEXT NOT NULL DEFAULT 'None'
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cycles_start
                ON cycles (cycle_start)
            """)

            # Migration guard: add ring columns to any pre-existing table
            # that was created with the old 3-column schema.
            cursor.execute("PRAGMA table_info(cycles)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            for col in ('r1_phases', 'r2_phases'):
                if col not in existing_cols:
                    cursor.execute(
                        f"ALTER TABLE cycles "
                        f"ADD COLUMN {col} TEXT NOT NULL DEFAULT 'None'"
                    )
                    print(
                        f"CycleProcessor: migrated cycles table – added '{col}'. "
                        "Run backfill_ring_phases() to populate existing rows."
                    )

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

    def backfill_ring_phases(self) -> int:
        """
        Populate r1_phases / r2_phases for cycles rows that still carry the
        default 'None' value from a pre-migration schema.

        Does NOT reingest events or rerun cycle detection.  Reads existing
        cycle_start values, fetches the matching Code 1 events and config,
        then writes the computed ring strings back in place.

        Returns:
            Number of cycle rows updated.

        Example::

            processor = CycleProcessor(Path("2068_data.db"))
            updated = processor.backfill_ring_phases()
            print(f"Backfilled {updated} cycle rows")
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()

            # Only target rows that still have the default placeholder
            cursor.execute("""
                SELECT DISTINCT cycle_start FROM cycles
                WHERE r1_phases = 'None' AND r2_phases = 'None'
                ORDER BY cycle_start
            """)
            rows = cursor.fetchall()

        if not rows:
            print("backfill_ring_phases: nothing to do – all rows already populated")
            return 0

        epochs = [r[0] for r in rows]
        start_epoch = min(epochs)
        end_epoch   = max(epochs)

        # Fetch only Code 1 (green start) events covering the full range.
        # This is the minimal event set needed by assign_ring_phases.
        with DatabaseManager(self.db_path) as manager:
            events_df = manager.query_events(
                start_time=start_epoch,
                end_time=end_epoch + 1,   # +1 s to make end inclusive
                event_codes=[1],
            )
            config = manager.get_config_at_date(
                datetime.fromtimestamp(start_epoch, self.tz)
            )

        if events_df.empty or config is None:
            print("backfill_ring_phases: no events or config found – aborting")
            return 0

        # Build a minimal cycles_df from the DB values
        cycles_df = pd.DataFrame({'cycle_start': epochs})

        updated = assign_ring_phases(cycles_df, events_df, config)

        # Write back only the two ring columns
        records = list(
            updated[['r1_phases', 'r2_phases', 'cycle_start']]
            .itertuples(index=False, name=None)
        )

        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            try:
                cursor.executemany(
                    "UPDATE cycles SET r1_phases = ?, r2_phases = ? "
                    "WHERE cycle_start = ?",
                    records
                )
                manager.conn.commit()
            except sqlite3.Error as e:
                manager.conn.rollback()
                raise RuntimeError(f"Database error during backfill: {e}")

        print(f"backfill_ring_phases: updated {len(records)} rows")
        return len(records)

    # ------------------------------------------------------------------
    # Date identification (local-time aware)
    # ------------------------------------------------------------------

    def _get_dates_to_process(self, reprocess: bool) -> List[str]:
        """
        Identify which LOCAL calendar dates need cycle processing.

        Three categories of dates are returned:

        1. Dates with no cycles at all (never processed).
        2. Dates where ingestion has added data since cycles were last run —
           i.e. the ingestion span_end for that local date extends more than
           ``_STALE_THRESHOLD_SECONDS`` past the last cycle_start on that
           date.  This handles the common case where a partial day was
           processed, then new files were ingested for the rest of that day.
        3. If ``reprocess=True``, all dates that have any ingested data.

        Uses ingestion_log (tiny) and cycles (small) — never scans events.

        Args:
            reprocess: If True, return every date that has ingested data.

        Returns:
            Sorted list of date strings in ``'YYYY-MM-DD'`` (local) format.
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()

            # Full ingestion epoch range from ingestion_log (Performance Rule)
            cursor.execute(
                "SELECT MIN(span_start), MAX(span_end) FROM ingestion_log"
            )
            row = cursor.fetchone()

        if not row or row[0] is None:
            return []

        min_epoch, max_epoch = row
        min_local = datetime.fromtimestamp(min_epoch, self.tz).date()
        max_local = datetime.fromtimestamp(max_epoch, self.tz).date()

        all_local_dates = set()
        current = min_local
        while current <= max_local:
            all_local_dates.add(current)
            current += timedelta(days=1)

        if reprocess:
            return sorted(d.strftime('%Y-%m-%d') for d in all_local_dates)

        stale_or_missing = self._get_stale_or_missing_dates(all_local_dates)
        return sorted(d.strftime('%Y-%m-%d') for d in stale_or_missing)

    def _get_stale_or_missing_dates(self, all_local_dates: set) -> set:
        """
        Return dates that are either unprocessed or stale.

        A date is **stale** when the ingestion log shows data extending more
        than ``_STALE_THRESHOLD_SECONDS`` beyond the latest cycle_start on
        that local date.  This catches the partial-day scenario: cycles were
        run after ingesting half a day, then the rest of the day was ingested
        later.

        Both tables (ingestion_log and cycles) are fetched in full — both are
        small and this avoids per-date round-trips.

        Args:
            all_local_dates: Complete set of local dates covered by ingestion.

        Returns:
            Subset of ``all_local_dates`` that need (re)processing.
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()

            # Latest ingested epoch per local date, derived from span_end values.
            # span_end is the end of each ingestion span; MAX over spans whose
            # span_end falls within or just after a local date gives us the
            # furthest point data has been ingested for that date.
            cursor.execute("SELECT span_start, span_end FROM ingestion_log")
            spans = cursor.fetchall()

            # Latest cycle_start per local date
            cursor.execute("SELECT cycle_start FROM cycles ORDER BY cycle_start")
            cycle_rows = cursor.fetchall()

        # Build: local_date → latest ingested epoch
        ingested_end: dict = {}
        for span_start, span_end in spans:
            # The span may cross a local midnight; attribute its end to the
            # local date of span_end (the latest data point in the span).
            local_date = datetime.fromtimestamp(span_end, self.tz).date()
            if local_date not in ingested_end or span_end > ingested_end[local_date]:
                ingested_end[local_date] = span_end
            # Also attribute span_start's date in case a span starts and ends
            # on different local dates — we want every covered date represented.
            local_start_date = datetime.fromtimestamp(span_start, self.tz).date()
            if local_start_date not in ingested_end:
                ingested_end[local_start_date] = span_end

        # Build: local_date → latest cycle_start epoch
        latest_cycle: dict = {}
        for (cs,) in cycle_rows:
            local_date = datetime.fromtimestamp(cs, self.tz).date()
            if local_date not in latest_cycle or cs > latest_cycle[local_date]:
                latest_cycle[local_date] = cs

        stale_or_missing = set()
        for d in all_local_dates:
            if d not in latest_cycle:
                # Never processed
                stale_or_missing.add(d)
            else:
                data_end = ingested_end.get(d, 0.0)
                cycle_end = latest_cycle[d]
                if data_end - cycle_end > _STALE_THRESHOLD_SECONDS:
                    stale_or_missing.add(d)

        return stale_or_missing

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

        start_epoch, end_epoch = self._local_day_to_utc_epoch_range(local_date)

        config = self._get_config_for_date(
            datetime.combine(local_date, time.min)
        )
        if config is None:
            raise ValueError(f"No configuration found for {date_str}")

        events_df = self._get_events_for_epoch_range(start_epoch, end_epoch)

        if events_df.empty:
            print(f"Processed {date_str}: No events found")
            return

        # Functional Core – calculate_cycles() now returns r1_phases and
        # r2_phases as part of its standard output via assign_ring_phases()
        try:
            cycles_df = calculate_cycles(events_df, config)
        except CycleDetectionError as e:
            raise CycleDetectionError(
                f"Cycle detection failed for {date_str}: {e}"
            )

        if cycles_df.empty:
            print(f"Processed {date_str}: No cycles detected")
            return

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
            (start_epoch, end_epoch) as UTC floats – midnight-to-midnight local.
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
                         detection_method, r1_phases, r2_phases].
            start_epoch: UTC epoch for start of local day (inclusive).
            end_epoch:   UTC epoch for end of local day (exclusive).
        """
        # Vectorized extraction – no row-iteration
        records = list(
            cycles_df[
                ['cycle_start', 'coord_plan', 'detection_method',
                 'r1_phases', 'r2_phases']
            ]
            .fillna({'r1_phases': 'None', 'r2_phases': 'None',
                     'detection_method': '', 'coord_plan': 0})
            .itertuples(index=False, name=None)
        )

        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            try:
                cursor.execute(
                    "DELETE FROM cycles "
                    "WHERE cycle_start >= ? AND cycle_start < ?",
                    (start_epoch, end_epoch)
                )
                cursor.executemany(
                    """
                    INSERT INTO cycles
                        (cycle_start, coord_plan, detection_method,
                         r1_phases, r2_phases)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    records
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

            cursor.execute("SELECT DISTINCT cycle_start FROM cycles")
            cycle_epochs = [r[0] for r in cursor.fetchall()]

        days_processed = len({
            datetime.fromtimestamp(ts, self.tz).date()
            for ts in cycle_epochs
        })

        all_dates_set = {
            datetime.strptime(d, '%Y-%m-%d').date()
            for d in self._get_dates_to_process(reprocess=True)
        }
        # Use stale-aware check so partially-processed days count as unprocessed
        stale_or_missing = self._get_stale_or_missing_dates(all_dates_set)
        unprocessed_days = len(stale_or_missing)

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
# Convenience entry-points
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