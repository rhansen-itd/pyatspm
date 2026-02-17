"""
ATSPM Data Ingestion Engine (Imperative Shell)

This module handles ingestion of .datZ files into the SQLite database.
It manages file I/O, state tracking, timezone conversions, and gap detection.

Package Location: src/atspm/data/ingestion.py
"""

import re
import math
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
    Manages incremental ingestion of .datZ files into SQLite database.
    
    Responsibilities:
    - File scanning and timestamp parsing
    - Timezone conversion (local → UTC)
    - State tracking via ingestion_log table
    - Gap detection and marker insertion
    - Batch processing with transaction safety
    """
    
    def __init__(
        self, 
        db_path: Path, 
        raw_data_dir: Path, 
        timezone: str = None
    ):
        """
        Initialize the ingestion engine.
        
        Args:
            db_path: Path to SQLite database
            raw_data_dir: Directory containing .datZ files
            timezone: Local timezone for controller (e.g., 'US/Mountain')
                     If None, will attempt to read from metadata table
        """
        self.db_path = Path(db_path)
        self.raw_data_dir = Path(raw_data_dir)
        
        # Try to get timezone from metadata if not provided
        if timezone is None:
            timezone = self._get_timezone_from_metadata()
            if timezone is None:
                timezone = 'US/Mountain'
                print(f"Warning: No timezone specified, using default: {timezone}")
        
        self.tz = pytz.timezone(timezone)
    
    def _get_timezone_from_metadata(self) -> Optional[str]:
        """
        Try to read timezone from metadata table.
        
        Returns:
            Timezone string or None if not found
        """
        try:
            with DatabaseManager(self.db_path) as manager:
                metadata = manager.get_metadata()
                if metadata and metadata.get('timezone'):
                    print(f"Using timezone from metadata: {metadata['timezone']}")
                    return metadata['timezone']
        except Exception:
            pass
        return None
    
    def run(self, incremental: bool = True, batch_size: int = 50) -> None:
        """
        Execute the ingestion process.
        
        Args:
            incremental: If True, only process files newer than last ingestion
            batch_size: Number of files to process before committing transaction
        """
        # Step 1: Get last processed timestamp
        last_timestamp = self._get_last_timestamp() if incremental else None
        
        if last_timestamp:
            print(f"Resuming from timestamp: {datetime.fromtimestamp(last_timestamp, self.tz)}")
        else:
            print("Starting fresh ingestion (no prior state)")
        
        # Step 2: Scan and parse filenames
        file_list = self._scan_files(last_timestamp)
        
        if not file_list:
            print("No new files to process")
            return
        
        print(f"Found {len(file_list)} files to process")
        
        # Step 3: Process in batches
        self._process_batches(file_list, batch_size)
    
    def _get_last_timestamp(self) -> Optional[float]:
        """
        Get the maximum file timestamp from ingestion log.
        
        Returns:
            UTC epoch float or None if table is empty
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            cursor.execute("SELECT MAX(file_timestamp) FROM ingestion_log")
            result = cursor.fetchone()
            
            return result[0] if result[0] is not None else None
    
    def _scan_files(
        self, 
        min_timestamp: Optional[float]
    ) -> List[Tuple[Path, float]]:
        """
        Scan directory for .datZ files and extract timestamps.
        
        Args:
            min_timestamp: Minimum UTC timestamp to include (or None for all)
        
        Returns:
            List of (filepath, utc_timestamp) tuples, sorted by timestamp
        """
        file_list = []
        
        for file_path in self.raw_data_dir.glob('*.datZ'):
            # Parse filename timestamp
            utc_ts = self._parse_filename_timestamp(file_path.name)
            
            if utc_ts is None:
                print(f"Warning: Could not parse timestamp from {file_path.name}")
                continue
            
            # Filter by minimum timestamp
            if min_timestamp is not None and utc_ts <= min_timestamp:
                continue
            
            file_list.append((file_path, utc_ts))
        
        # Sort by timestamp
        file_list.sort(key=lambda x: x[1])
        
        return file_list
    
    def _parse_filename_timestamp(self, filename: str) -> Optional[float]:
        """
        Extract timestamp from filename and convert to UTC epoch.
        
        Expected format: *_YYYY_MM_DD_HHMM.datZ
        Example: ECON_10.70.10.51_2025_01_15_1430.datZ
        
        Args:
            filename: Name of .datZ file
        
        Returns:
            UTC epoch float or None if parsing fails
        """
        # Extract YYYY_MM_DD_HHMM pattern
        pattern = r'(\d{4})_(\d{2})_(\d{2})_(\d{4})'
        match = re.search(pattern, filename)
        
        if not match:
            return None
        
        try:
            year, month, day, hhmm = match.groups()
            hour = int(hhmm[:2])
            minute = int(hhmm[2:])
            
            # Create naive datetime (controller local time)
            naive_dt = datetime(
                int(year), int(month), int(day), 
                hour, minute
            )
            
            # Localize to controller timezone
            aware_dt = self.tz.localize(naive_dt)
            
            # Convert to UTC epoch
            utc_ts = aware_dt.timestamp()
            
            return utc_ts
            
        except (ValueError, pytz.exceptions.AmbiguousTimeError, 
                pytz.exceptions.NonExistentTimeError) as e:
            print(f"Warning: Error parsing {filename}: {e}")
            return None
    
    def _process_batches(
        self, 
        file_list: List[Tuple[Path, float]], 
        batch_size: int
    ) -> None:
        """
        Process files in batches with transaction safety.
        
        Args:
            file_list: List of (filepath, utc_timestamp) tuples
            batch_size: Files per transaction
        """
        total_files = len(file_list)
        events_buffer = []
        logs_buffer = []
        
        for i, (file_path, utc_start) in enumerate(file_list):
            # Determine next file timestamp for gap detection
            next_file_utc = file_list[i + 1][1] if i + 1 < len(file_list) else None
            
            # Process file
            result = self._process_file(file_path, utc_start, next_file_utc)
            
            if result is None:
                # File failed to process - skip
                continue
            
            events_df, log_entry = result
            
            # Add to buffers
            if not events_df.empty:
                events_buffer.append(events_df)
            logs_buffer.append(log_entry)
            
            # Commit when batch is full or at end of list
            should_commit = (
                len(logs_buffer) >= batch_size or 
                i == len(file_list) - 1
            )
            
            if should_commit and logs_buffer:
                self._commit_batch(events_buffer, logs_buffer)
                
                # Progress report
                processed_count = i + 1
                print(f"Processed {processed_count}/{total_files} files...")
                
                # Clear buffers
                events_buffer = []
                logs_buffer = []
    
    def _process_file(
        self,
        file_path: Path,
        utc_start: float,
        next_file_utc: Optional[float]
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Process a single .datZ file.
        
        Args:
            file_path: Path to .datZ file
            utc_start: File start timestamp (UTC epoch)
            next_file_utc: Next file's start timestamp (or None)
        
        Returns:
            Tuple of (events_df, log_entry_dict) or None on error
        """
        try:
            # Read file bytes (Imperative I/O)
            raw_bytes = file_path.read_bytes()
            
        except (IOError, OSError) as e:
            print(f"Error reading {file_path.name}: {e}")
            return None
        
        # Decode using Functional Core
        try:
            df = decoders.parse_datz_bytes(raw_bytes, utc_start)
        except decoders.DatZDecodingError as e:
            print(f"Error decoding {file_path.name}: {e}")
            return None
        
        # Calculate statistics
        if not df.empty:
            last_event_ts = df['timestamp'].max()
            row_count = len(df)
        else:
            last_event_ts = utc_start
            row_count = 0
        
        # Gap detection
        df = self._handle_gaps(
            df, 
            utc_start, 
            last_event_ts, 
            next_file_utc
        )
        
        # Create log entry
        log_entry = {
            'filename': file_path.name,
            'file_timestamp': utc_start,
            'processed_at': datetime.utcnow().isoformat(),
            'row_count': row_count
        }
        
        return df, log_entry
    
    def _handle_gaps(
        self,
        df: pd.DataFrame,
        utc_start: float,
        last_event_ts: float,
        next_file_utc: Optional[float]
    ) -> pd.DataFrame:
        """
        Detect gaps and insert markers if needed.
        
        Args:
            df: Events DataFrame
            utc_start: File start timestamp
            last_event_ts: Last event timestamp in file
            next_file_utc: Next file's start timestamp
        
        Returns:
            DataFrame with gap marker inserted (if applicable)
        """
        # Calculate expected file duration
        duration = self._calculate_duration(
            utc_start, 
            last_event_ts, 
            next_file_utc
        )
        
        expected_end = utc_start + duration
        
        # Check for gap
        if next_file_utc is not None:
            gap_seconds = next_file_utc - expected_end
            
            if gap_seconds > 5.0:  # 5-second tolerance
                # Insert gap marker at expected end
                df = decoders.insert_gap_marker(df, expected_end)
        
        return df
    
    def _calculate_duration(
        self,
        start_ts: float,
        last_event_ts: float,
        next_file_start_ts: Optional[float]
    ) -> float:
        """
        Determine file duration for gap detection.
        
        Uses multiple signals to detect if file is in "Grid Mode" (15-min)
        or "1-Minute Mode".
        
        Args:
            start_ts: File start timestamp (UTC epoch)
            last_event_ts: Last event timestamp (UTC epoch)
            next_file_start_ts: Next file timestamp (UTC epoch, or None)
        
        Returns:
            Duration in seconds (float)
        """
        # Evidence 1: Data span > 61 seconds → Grid Mode
        data_span = last_event_ts - start_ts
        if data_span > 61.0:
            return self._grid_duration(start_ts)
        
        # Evidence 2: Next file > 90 seconds away → Grid Mode
        if next_file_start_ts is not None:
            sequence_span = next_file_start_ts - start_ts
            if sequence_span > 90.0:
                return self._grid_duration(start_ts)
        
        # Evidence 3: Start time aligns with 15-min grid → Grid Mode
        local_dt = datetime.fromtimestamp(start_ts, self.tz)
        if local_dt.minute in [0, 15, 30, 45]:
            return self._grid_duration(start_ts)
        
        # Fallback: 1-Minute Mode
        return 60.0
    
    def _grid_duration(self, start_ts: float) -> float:
        """
        Calculate seconds until next 15-minute boundary.
        
        Args:
            start_ts: Start timestamp (UTC epoch)
        
        Returns:
            Seconds to next quarter-hour mark
        """
        # Convert to local time
        local_dt = datetime.fromtimestamp(start_ts, self.tz)
        
        # Calculate next 15-minute boundary
        current_minute = local_dt.minute
        current_second = local_dt.second
        
        # Find next quarter-hour mark
        quarter_hours = [0, 15, 30, 45]
        next_quarter = None
        
        for qh in quarter_hours:
            if current_minute < qh:
                next_quarter = qh
                break
        
        if next_quarter is None:
            # Roll over to next hour
            next_dt = local_dt.replace(minute=0, second=0, microsecond=0)
            next_dt += timedelta(hours=1)
        else:
            # Same hour, next quarter
            next_dt = local_dt.replace(minute=next_quarter, second=0, microsecond=0)
        
        # Calculate duration
        duration = (next_dt - local_dt).total_seconds()
        
        return duration
    
    def _commit_batch(
        self,
        events_buffer: List[pd.DataFrame],
        logs_buffer: List[Dict[str, Any]]
    ) -> None:
        """
        Commit buffered events and logs in a single transaction.
        
        Args:
            events_buffer: List of event DataFrames
            logs_buffer: List of log entry dictionaries
        """
        # Combine all events
        if events_buffer:
            all_events = pd.concat(events_buffer, ignore_index=True)
            # Convert to list of tuples for insertion
            event_tuples = [
                (row['timestamp'], row['event_code'], row['parameter'])
                for _, row in all_events.iterrows()
            ]
        else:
            event_tuples = []
        
        # Use DatabaseManager connection directly for transaction
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            
            try:
                # Start transaction (implicit with first statement)
                
                # Insert events
                if event_tuples:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO events (timestamp, event_code, parameter) VALUES (?, ?, ?)",
                        event_tuples
                    )
                
                # Insert logs
                for log in logs_buffer:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO ingestion_log 
                        (filename, file_timestamp, processed_at, row_count)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            log['filename'],
                            log['file_timestamp'],
                            log['processed_at'],
                            log['row_count']
                        )
                    )
                
                # Commit transaction
                manager.conn.commit()
                
            except sqlite3.Error as e:
                # Rollback on error
                manager.conn.rollback()
                print(f"Error committing batch: {e}")
                raise
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ingested data.
        
        Returns:
            Dictionary with ingestion statistics
        """
        with DatabaseManager(self.db_path) as manager:
            cursor = manager.conn.cursor()
            
            # Files processed
            cursor.execute("SELECT COUNT(*) FROM ingestion_log")
            file_count = cursor.fetchone()[0]
            
            # Total events
            cursor.execute("SELECT COUNT(*) FROM events")
            event_count = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(file_timestamp), MAX(file_timestamp) FROM ingestion_log")
            min_ts, max_ts = cursor.fetchone()
            
            # Gap count
            cursor.execute("SELECT COUNT(*) FROM events WHERE event_code = -1")
            gap_count = cursor.fetchone()[0]
            
            return {
                'files_processed': file_count,
                'total_events': event_count,
                'gap_markers': gap_count,
                'date_range': {
                    'start': datetime.fromtimestamp(min_ts, self.tz).isoformat() if min_ts else None,
                    'end': datetime.fromtimestamp(max_ts, self.tz).isoformat() if max_ts else None,
                }
            }
    
    def reprocess_file(self, filename: str) -> bool:
        """
        Reprocess a single file (useful for fixing corrupted ingestion).
        
        Args:
            filename: Name of file to reprocess
        
        Returns:
            True if successful, False otherwise
        """
        file_path = self.raw_data_dir / filename
        
        if not file_path.exists():
            print(f"File not found: {filename}")
            return False
        
        # Parse timestamp
        utc_start = self._parse_filename_timestamp(filename)
        if utc_start is None:
            print(f"Could not parse timestamp from {filename}")
            return False
        
        # Process file
        result = self._process_file(file_path, utc_start, None)
        
        if result is None:
            return False
        
        events_df, log_entry = result
        
        # Commit
        self._commit_batch(
            [events_df] if not events_df.empty else [],
            [log_entry]
        )
        
        print(f"Reprocessed {filename}: {log_entry['row_count']} events")
        return True


def run_ingestion(
    db_path: Path,
    data_dir: Path,
    timezone: str = None,
    incremental: bool = True,
    batch_size: int = 50
) -> None:
    """
    Convenience function to run ingestion process.
    
    Args:
        db_path: Path to SQLite database
        data_dir: Directory containing .datZ files
        timezone: Local timezone (e.g., 'US/Mountain'). If None, reads from metadata table.
        incremental: Only process new files
        batch_size: Files per transaction
    """
    engine = IngestionEngine(db_path, data_dir, timezone)
    engine.run(incremental=incremental, batch_size=batch_size)
    
    # Print statistics
    stats = engine.get_ingestion_stats()
    print("\nIngestion Complete!")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Gap markers: {stats['gap_markers']}")
    if stats['date_range']['start']:
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")