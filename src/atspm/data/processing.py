"""
ATSPM Cycle Processing Engine (Imperative Shell)

This module orchestrates cycle detection on the SQLite database.
It manages I/O, transactions, and calls the Functional Core for logic.

Package Location: src/atspm/data/processing.py
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from .manager import DatabaseManager
from ..analysis.cycles import calculate_cycles, CycleDetectionError


class CycleProcessor:
    """
    Manages cycle detection processing on the database.
    
    Responsibilities:
    - Create and manage cycles table
    - Identify unprocessed dates
    - Fetch appropriate config for each date
    - Call Functional Core for cycle detection
    - Persist results with transaction safety
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the cycle processor.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        
        # Initialize cycles table
        self._init_cycles_table()
    
    def _init_cycles_table(self) -> None:
        """
        Create cycles table if it doesn't exist.
        
        Schema:
        - cycle_start: REAL (UTC epoch) PRIMARY KEY
        - coord_plan: INTEGER
        - detection_method: TEXT
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
            
            # Index for faster queries by date range
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cycles_start
                ON cycles (cycle_start)
            """)
            
            manager.conn.commit()
    
    def run(self, reprocess: bool = False) -> None:
        """
        Process all unprocessed dates in the events table.
        
        Args:
            reprocess: If True, reprocess all dates (delete existing cycles)
        """
        # Step 1: Identify dates to process
        dates_to_process = self._get_dates_to_process(reprocess)
        
        if not dates_to_process:
            print("No dates to process")
            return
        
        print(f"Processing {len(dates_to_process)} dates...")
        
        # Step 2: Process each date
        successful = 0
        failed = 0
        
        for date_str in dates_to_process:
            try:
                self._process_day(date_str)
                successful += 1
            except Exception as e:
                failed += 1
                print(f"ERROR processing {date_str}: {e}")
                continue
        
        # Summary
        print(f"\nCycle Processing Complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
    
    def _get_dates_to_process(self, reprocess: bool) -> List[str]:
        """
        Identify which dates need cycle processing.
        
        Args:
            reprocess: If True, return all dates with events
        
        Returns:
            List of date strings in 'YYYY-MM-DD' format, sorted
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            
            if reprocess:
                # Get all dates with events
                cursor.execute("""
                    SELECT DISTINCT DATE(timestamp, 'unixepoch') as event_date
                    FROM events
                    WHERE event_code != -1
                    ORDER BY event_date
                """)
            else:
                # Get dates with events but no cycles
                cursor.execute("""
                    SELECT DISTINCT DATE(e.timestamp, 'unixepoch') as event_date
                    FROM events e
                    WHERE e.event_code != -1
                    AND NOT EXISTS (
                        SELECT 1 FROM cycles c
                        WHERE DATE(c.cycle_start, 'unixepoch') = DATE(e.timestamp, 'unixepoch')
                    )
                    ORDER BY event_date
                """)
            
            results = cursor.fetchall()
            
            return [row[0] for row in results]
    
    def _process_day(self, date_str: str) -> None:
        """
        Process a single day's worth of events.
        
        Args:
            date_str: Date in 'YYYY-MM-DD' format
        """
        # Parse date
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}': {e}")
        
        # Step 1: Fetch config for this date
        config = self._get_config_for_date(date)
        
        if config is None:
            raise ValueError(f"No configuration found for {date_str}")
        
        # Step 2: Fetch events for this 24-hour window
        events_df = self._get_events_for_day(date)
        
        if events_df.empty:
            print(f"Processed {date_str}: No events found")
            return
        
        # Step 3: Calculate cycles (Functional Core)
        try:
            cycles_df = calculate_cycles(events_df, config)
        except CycleDetectionError as e:
            raise CycleDetectionError(
                f"Cycle detection failed for {date_str}: {e}"
            )
        
        if cycles_df.empty:
            print(f"Processed {date_str}: No cycles detected")
            return
        
        # Step 4: Persist to database
        self._save_cycles(cycles_df, date)
        
        # Step 5: Log results
        detection_method = cycles_df['detection_method'].iloc[0]
        cycle_count = len(cycles_df)
        
        print(f"Processed {date_str}: Found {cycle_count} cycles using {detection_method}")
    
    def _get_config_for_date(self, date: datetime) -> Optional[dict]:
        """
        Get the configuration active on a specific date.
        
        Args:
            date: Date to query
        
        Returns:
            Configuration dict or None if not found
        """
        with DatabaseManager(self.db_path) as manager:
            config = manager.get_config_at_date(date)
            
            return config
    
    def _get_events_for_day(self, date: datetime) -> pd.DataFrame:
        """
        Fetch all events for a 24-hour period.
        
        Args:
            date: Date (time will be set to 00:00:00)
        
        Returns:
            DataFrame with events for that day
        """
        # Calculate UTC epoch range for the day
        day_start = datetime.combine(date.date(), datetime.min.time())
        day_end = day_start + timedelta(days=1)
        
        start_epoch = day_start.timestamp()
        end_epoch = day_end.timestamp()
        
        with DatabaseManager(self.db_path) as manager:
            # Query events for this time range
            df = manager.query_events(
                start_time=start_epoch,
                end_time=end_epoch
            )
            
            return df
    
    def _save_cycles(self, cycles_df: pd.DataFrame, date: datetime) -> None:
        """
        Save cycles to database with transaction safety.
        
        Ensures idempotency by deleting existing cycles for the day first.
        
        Args:
            cycles_df: Cycles DataFrame to save
            date: Date being processed (for cleanup)
        """
        # Calculate day boundaries
        day_start = datetime.combine(date.date(), datetime.min.time())
        day_end = day_start + timedelta(days=1)
        
        start_epoch = day_start.timestamp()
        end_epoch = day_end.timestamp()
        
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            
            try:
                # Step 1: Delete existing cycles for this day (idempotency)
                cursor.execute("""
                    DELETE FROM cycles
                    WHERE cycle_start >= ? AND cycle_start < ?
                """, (start_epoch, end_epoch))
                
                # Step 2: Insert new cycles
                cycle_records = [
                    (
                        row['cycle_start'],
                        int(row['coord_plan']),
                        row['detection_method']
                    )
                    for _, row in cycles_df.iterrows()
                ]
                
                cursor.executemany("""
                    INSERT INTO cycles (cycle_start, coord_plan, detection_method)
                    VALUES (?, ?, ?)
                """, cycle_records)
                
                # Commit transaction
                manager.conn.commit()
                
            except sqlite3.Error as e:
                # Rollback on error
                manager.conn.rollback()
                raise RuntimeError(f"Database error saving cycles: {e}")
    
    def get_processing_stats(self) -> dict:
        """
        Get statistics about cycle processing.
        
        Returns:
            Dictionary with processing statistics
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            
            # Total cycles
            cursor.execute("SELECT COUNT(*) FROM cycles")
            total_cycles = cursor.fetchone()[0]
            
            # Detection method breakdown
            cursor.execute("""
                SELECT detection_method, COUNT(*) 
                FROM cycles 
                GROUP BY detection_method
            """)
            method_counts = dict(cursor.fetchall())
            
            # Coordination plan breakdown
            cursor.execute("""
                SELECT coord_plan, COUNT(*) 
                FROM cycles 
                GROUP BY coord_plan
            """)
            coord_counts = dict(cursor.fetchall())
            
            # Date range
            cursor.execute("""
                SELECT 
                    MIN(cycle_start),
                    MAX(cycle_start)
                FROM cycles
            """)
            min_ts, max_ts = cursor.fetchone()
            
            # Days processed
            cursor.execute("""
                SELECT COUNT(DISTINCT DATE(cycle_start, 'unixepoch'))
                FROM cycles
            """)
            days_processed = cursor.fetchone()[0]
            
            # Unprocessed days (have events but no cycles)
            cursor.execute("""
                SELECT COUNT(DISTINCT DATE(timestamp, 'unixepoch'))
                FROM events
                WHERE event_code != -1
                AND NOT EXISTS (
                    SELECT 1 FROM cycles c
                    WHERE DATE(c.cycle_start, 'unixepoch') = DATE(events.timestamp, 'unixepoch')
                )
            """)
            unprocessed_days = cursor.fetchone()[0]
            
            return {
                'total_cycles': total_cycles,
                'days_processed': days_processed,
                'unprocessed_days': unprocessed_days,
                'detection_methods': method_counts,
                'coord_plans': coord_counts,
                'date_range': {
                    'start': datetime.fromtimestamp(min_ts).isoformat() if min_ts else None,
                    'end': datetime.fromtimestamp(max_ts).isoformat() if max_ts else None,
                }
            }
    
    def reprocess_date(self, date_str: str) -> bool:
        """
        Reprocess a specific date (useful for fixing errors).
        
        Args:
            date_str: Date in 'YYYY-MM-DD' format
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._process_day(date_str)
            return True
        except Exception as e:
            print(f"Error reprocessing {date_str}: {e}")
            return False
    
    def get_cycle_summary_for_date(self, date_str: str) -> Optional[dict]:
        """
        Get cycle summary for a specific date.
        
        Args:
            date_str: Date in 'YYYY-MM-DD' format
        
        Returns:
            Dictionary with cycle statistics for that date
        """
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None
        
        day_start = datetime.combine(date.date(), datetime.min.time())
        day_end = day_start + timedelta(days=1)
        
        start_epoch = day_start.timestamp()
        end_epoch = day_end.timestamp()
        
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            
            # Get cycles for this day
            cursor.execute("""
                SELECT 
                    COUNT(*) as cycle_count,
                    MIN(coord_plan) as min_coord,
                    MAX(coord_plan) as max_coord,
                    detection_method
                FROM cycles
                WHERE cycle_start >= ? AND cycle_start < ?
                GROUP BY detection_method
            """, (start_epoch, end_epoch))
            
            results = cursor.fetchall()
            
            if not results:
                return None
            
            return {
                'date': date_str,
                'cycle_count': results[0][0],
                'coord_plan_range': (results[0][1], results[0][2]),
                'detection_method': results[0][3]
            }
    
    def validate_cycles_table(self) -> Tuple[bool, List[str]]:
        """
        Validate the cycles table for data quality issues.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            
            # Check for duplicate cycle starts
            cursor.execute("""
                SELECT cycle_start, COUNT(*)
                FROM cycles
                GROUP BY cycle_start
                HAVING COUNT(*) > 1
            """)
            duplicates = cursor.fetchall()
            
            if duplicates:
                issues.append(
                    f"Found {len(duplicates)} duplicate cycle_start timestamps"
                )
            
            # Check for unreasonable cycle lengths
            cursor.execute("""
                SELECT 
                    MIN(next_start - cycle_start) as min_length,
                    MAX(next_start - cycle_start) as max_length
                FROM (
                    SELECT 
                        cycle_start,
                        LEAD(cycle_start) OVER (ORDER BY cycle_start) as next_start
                    FROM cycles
                )
                WHERE next_start IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result:
                min_length, max_length = result
                
                if min_length and min_length < 10.0:
                    issues.append(
                        f"Found cycle length < 10s (minimum: {min_length:.1f}s)"
                    )
                
                if max_length and max_length > 300.0:
                    issues.append(
                        f"Found cycle length > 300s (maximum: {max_length:.1f}s)"
                    )
            
            # Check for NULL values
            cursor.execute("""
                SELECT COUNT(*) 
                FROM cycles 
                WHERE coord_plan IS NULL OR detection_method IS NULL
            """)
            null_count = cursor.fetchone()[0]
            
            if null_count > 0:
                issues.append(f"Found {null_count} rows with NULL values")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues


def run_cycle_processing(
    db_path: Path,
    reprocess: bool = False
) -> None:
    """
    Convenience function to run cycle processing.
    
    Args:
        db_path: Path to SQLite database
        reprocess: If True, reprocess all dates
    """
    processor = CycleProcessor(db_path)
    processor.run(reprocess=reprocess)
    
    # Print statistics
    stats = processor.get_processing_stats()
    print("\nCycle Processing Statistics:")
    print(f"  Total cycles: {stats['total_cycles']:,}")
    print(f"  Days processed: {stats['days_processed']}")
    print(f"  Unprocessed days: {stats['unprocessed_days']}")
    print(f"  Detection methods: {stats['detection_methods']}")
    print(f"  Coordination plans: {stats['coord_plans']}")
    if stats['date_range']['start']:
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")