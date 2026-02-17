"""
Database Manager for ATSPM System (Imperative Shell)

This module handles all database operations including initialization,
configuration import, and query execution. It follows the "Imperative Shell"
pattern - managing state, I/O, and resources while delegating logic to the
Functional Core.

Package Location: src/atspm/data/manager.py
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class DatabaseManager:
    """
    Manages SQLite database operations for ATSPM system.
    
    Responsibilities:
    - Database initialization with proper schema
    - Configuration import and transformation
    - Transaction management
    - Query execution
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize DatabaseManager with path to SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
    
    def __enter__(self):
        """Context manager entry - open connection"""
        self.conn = sqlite3.connect(self.db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection"""
        if self.conn:
            self.conn.close()
    
    def init_db(self) -> None:
        """
        Initialize database with schema and indices.
        
        Creates:
        1. events table: Raw traffic event data
        2. config table: Hybrid schema for intersection configuration
        3. metadata table: Static intersection attributes (IPs, location, etc.)
        4. ingestion_log table: Tracks processed files
        
        Enables WAL mode for concurrent access.
        """
        if not self.conn:
            raise RuntimeError("Database connection not established. Use 'with' statement.")
        
        cursor = self.conn.cursor()
        
        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # 1. Create events table (Raw Data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                timestamp REAL NOT NULL,
                event_code INTEGER NOT NULL,
                parameter INTEGER NOT NULL,
                UNIQUE(timestamp, event_code, parameter) ON CONFLICT IGNORE
            )
        """)
        
        # Index on timestamp for fast range queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp 
            ON events (timestamp)
        """)
        
        # Compound index for event code filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_code_param 
            ON events (event_code, parameter)
        """)
        
        # Covering index for timestamp + code (avoids table lookups for range+filter queries)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_ts_code
            ON events (timestamp, event_code)
        """)
        
        # 2. Create config table (Hybrid Schema)
        # Note: Additional columns will be added dynamically during import
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_date TEXT NOT NULL,
                end_date TEXT,
                UNIQUE(start_date) ON CONFLICT REPLACE
            )
        """)
        
        # 3. Create metadata table (Static/Operational Attributes)
        # lock_id=1 enforces a single-row pattern
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                lock_id INTEGER PRIMARY KEY CHECK (lock_id = 1),
                intersection_id TEXT,
                intersection_name TEXT,
                controller_ip TEXT,
                detection_type TEXT,
                detection_ip TEXT,
                major_road_route TEXT,
                major_road_name TEXT,
                minor_road_route TEXT,
                minor_road_name TEXT,
                latitude REAL,
                longitude REAL,
                timezone TEXT NOT NULL DEFAULT 'US/Mountain',
                agency_id TEXT
            )
        """)
        
        # 4. Create ingestion_log table (Control Table)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                filename TEXT PRIMARY KEY,
                file_timestamp REAL NOT NULL,
                processed_at TEXT NOT NULL,
                row_count INTEGER NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ingestion_timestamp
            ON ingestion_log (file_timestamp)
        """)
        
        self.conn.commit()
        print(f"Database initialized at {self.db_path}")
    
    def get_config_columns(self) -> List[str]:
        """
        Get list of existing columns in config table.
        
        Returns:
            List of column names
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(config)")
        return [row[1] for row in cursor.fetchall()]
    
    def add_config_column(self, column_name: str, column_type: str = "TEXT") -> None:
        """
        Add a new column to config table if it doesn't exist.
        
        Args:
            column_name: Name of column to add
            column_type: SQL type (default: TEXT)
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        existing_cols = self.get_config_columns()
        
        if column_name not in existing_cols:
            cursor = self.conn.cursor()
            # Use parameterized column name safely
            safe_name = column_name.replace('"', '""')
            cursor.execute(f'ALTER TABLE config ADD COLUMN "{safe_name}" {column_type}')
            self.conn.commit()
    
    def import_config(self, csv_path: Path) -> None:
        """
        Import intersection configuration from legacy CSV format.
        
        Transforms int_cfg.csv into hybrid schema:
        - Standard rows (TM, RB, Det, WD) become wide columns
        - Exclusion rows (Exc) become JSON in TM_Exclusions column
        
        Args:
            csv_path: Path to int_cfg.csv file
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Config file not found: {csv_path}")
        
        # Read CSV with multi-index
        df_cfg = pd.read_csv(csv_path, index_col=[0, 1])
        
        # Drop empty columns
        df_cfg = df_cfg.dropna(how='all', axis=1)
        
        # Convert column names to datetime
        df_cfg.columns = pd.to_datetime(df_cfg.columns, errors='coerce')
        df_cfg = df_cfg.loc[:, df_cfg.columns.notna()]
        
        if df_cfg.empty or len(df_cfg.columns) == 0:
            raise ValueError("No valid date columns found in config CSV")
        
        # Sort columns by date
        df_cfg = df_cfg.sort_index(axis=1)
        
        # Process each date column
        date_columns = sorted(df_cfg.columns)
        
        for i, start_date in enumerate(date_columns):
            # Calculate end_date
            if i < len(date_columns) - 1:
                end_date = date_columns[i + 1]
            else:
                end_date = None
            
            # Extract configuration for this date
            config_data = self._transform_config_column(df_cfg[start_date])
            
            # Prepare row for insertion
            row_data = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat() if end_date else None,
            }
            
            # Add dynamic columns for all config (including TM_Exclusions)
            row_data.update(config_data)
            
            # Ensure columns exist in table
            for col_name in row_data.keys():
                if col_name not in ['start_date', 'end_date']:
                    self.add_config_column(col_name)
            
            # Insert row
            self._insert_config_row(row_data)
        
        self.conn.commit()
        print(f"Imported {len(date_columns)} configuration periods from {csv_path}")
    
    def _transform_config_column(self, config_series: pd.Series) -> Dict[str, Any]:
        """
        Transform a single date column from int_cfg.csv into dict format.
        
        Args:
            config_series: Pandas Series with multi-index (category, parameter)
        
        Returns:
            Dictionary with transformed configuration
        """
        result = {}
        
        # Get unique categories
        categories = config_series.index.get_level_values(0).unique()
        
        # Parse exclusions first (needed for TM group)
        if 'Exc:' in categories:
            cat_data = config_series.loc['Exc:']
            exclusions = self._parse_exclusions(cat_data)
        else:
            exclusions = []
        
        for category in categories:
            cat_data = config_series.loc[category]
            
            if category == 'Exc:':
                # Already handled above
                continue
            elif category == 'TM:':
                # Movement configuration
                for movement, value in cat_data.items():
                    if pd.notna(value) and str(value).strip():
                        col_name = f"TM_{movement}"
                        result[col_name] = str(value).strip()
                
                # Add exclusions to TM group as JSON
                result['TM_Exclusions'] = json.dumps(exclusions)
                
            elif category in ['Det:', 'Plt:']:
                # Detector configuration (handles both Det: and legacy Plt: formats)
                for param, value in cat_data.items():
                    if pd.notna(value) and str(value).strip():
                        col_name = f"Det_{param.replace(' ', '_')}"
                        result[col_name] = str(value).strip()
            elif category == 'RB:':
                # Ring-Barrier configuration
                for param, value in cat_data.items():
                    if pd.notna(value) and str(value).strip():
                        col_name = f"RB_{param}"
                        result[col_name] = str(value).strip()
            elif category == 'WD:':
                # Watchdog configuration
                for param, value in cat_data.items():
                    if pd.notna(value) and str(value).strip():
                        col_name = f"WD_{param.replace(' ', '_')}"
                        result[col_name] = str(value).strip()
        
        return result
    
    def _parse_exclusions(self, exc_series: pd.Series) -> List[Dict[str, Any]]:
        """
        Parse exclusion rows into list of dictionaries.
        
        Expected format in CSV:
        - Exc:Detector -> "33,34,35"
        - Exc:Phase -> "2,2,4"
        - Exc:Status -> "Red,Yellow,Green"
        
        Returns:
            List of dicts: [{"detector": 33, "phase": 2, "status": "Red"}, ...]
        """
        exclusions = []
        
        def clean_val(val) -> str:
            """Safely convert pandas value to string, treating NaN as empty."""
            if pd.isna(val):
                return ''
            return str(val).strip()
        
        # Get the three exclusion rows safely
        detector_str = clean_val(exc_series.get('Detector'))
        phase_str = clean_val(exc_series.get('Phase'))
        status_str = clean_val(exc_series.get('Status'))
        
        # Skip if all are empty
        if not any([detector_str, phase_str, status_str]):
            return exclusions
        
        # Parse comma-separated values
        try:
            detectors = [int(d.strip()) for d in detector_str.split(',') if d.strip()]
            phases = [int(p.strip()) for p in phase_str.split(',') if p.strip()]
            statuses = [s.strip() for s in status_str.split(',') if s.strip()]
            
            # All lists should have same length
            if not (len(detectors) == len(phases) == len(statuses)):
                if len(detectors) > 0:
                    print(f"Warning: Exclusion row lengths don't match. Det:{len(detectors)}, Ph:{len(phases)}, Status:{len(statuses)}")
                return exclusions
            
            # Create exclusion dicts
            for det, ph, status in zip(detectors, phases, statuses):
                exclusions.append({
                    'detector': det,
                    'phase': ph,
                    'status': status
                })
        
        except (ValueError, AttributeError) as e:
            print(f"Warning: Error parsing exclusions: {e}")
        
        return exclusions
    
    def _insert_config_row(self, row_data: Dict[str, Any]) -> None:
        """
        Insert a configuration row into the database.
        
        Args:
            row_data: Dictionary of column names to values
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        # Build INSERT statement
        columns = list(row_data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join([f'"{col}"' for col in columns])
        
        sql = f"""
            INSERT OR REPLACE INTO config ({column_names})
            VALUES ({placeholders})
        """
        
        values = [row_data[col] for col in columns]
        
        cursor = self.conn.cursor()
        cursor.execute(sql, values)
    
    def get_config_at_date(self, query_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Retrieve configuration active at a specific date.
        
        Args:
            query_date: Date to query
        
        Returns:
            Dictionary of configuration values or None if not found
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        query_str = query_date.isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM config
            WHERE start_date <= ?
            AND (end_date IS NULL OR end_date > ?)
            ORDER BY start_date DESC
            LIMIT 1
        """, (query_str, query_str))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Build result dict
        result = dict(zip(columns, row))
        
        # Parse TM_Exclusions JSON
        if result.get('TM_Exclusions'):
            result['exclusions'] = json.loads(result['TM_Exclusions'])
        else:
            result['exclusions'] = []
        
        return result
    
    def insert_events(self, events: List[Tuple[float, int, int]]) -> int:
        """
        Bulk insert events into database.
        
        Args:
            events: List of (timestamp, event_code, parameter) tuples
        
        Returns:
            Number of rows inserted (may be less than input due to duplicates)
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        cursor = self.conn.cursor()
        
        # Get count before insert
        cursor.execute("SELECT COUNT(*) FROM events")
        count_before = cursor.fetchone()[0]
        
        # Bulk insert
        cursor.executemany(
            "INSERT OR IGNORE INTO events (timestamp, event_code, parameter) VALUES (?, ?, ?)",
            events
        )
        
        # Get count after insert
        cursor.execute("SELECT COUNT(*) FROM events")
        count_after = cursor.fetchone()[0]
        
        self.conn.commit()
        
        rows_inserted = count_after - count_before
        return rows_inserted
    
    def query_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_codes: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Query events from database.
        
        Args:
            start_time: Minimum timestamp (inclusive)
            end_time: Maximum timestamp (exclusive)
            event_codes: List of event codes to filter
        
        Returns:
            DataFrame with columns: timestamp, event_code, parameter
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        # Build query
        where_clauses = []
        params = []
        
        if start_time is not None:
            where_clauses.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time is not None:
            where_clauses.append("timestamp < ?")
            params.append(end_time)
        
        if event_codes:
            placeholders = ', '.join(['?' for _ in event_codes])
            where_clauses.append(f"event_code IN ({placeholders})")
            params.extend(event_codes)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        sql = f"""
            SELECT timestamp, event_code, parameter
            FROM events
            WHERE {where_sql}
            ORDER BY timestamp, event_code, parameter
        """
        
        # Execute query and return DataFrame
        df = pd.read_sql_query(sql, self.conn, params=params)
        return df
    
    def get_event_date_range(self) -> Optional[Tuple[float, float]]:
        """
        Get the min and max timestamps in the events table.
        
        Returns:
            Tuple of (min_timestamp, max_timestamp) or None if table is empty
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events")
        result = cursor.fetchone()
        
        if result[0] is None:
            return None
        
        return result
    
    def vacuum(self) -> None:
        """
        Optimize database by running VACUUM command.
        Should be run after large deletions or periodic maintenance.
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        self.conn.execute("VACUUM")
        print("Database vacuumed successfully")
    
    def set_metadata(
        self,
        intersection_id: Optional[str] = None,
        intersection_name: Optional[str] = None,
        timezone: str = 'US/Mountain',
        controller_ip: Optional[str] = None,
        detection_type: Optional[str] = None,
        detection_ip: Optional[str] = None,
        major_road_route: Optional[str] = None,
        major_road_name: Optional[str] = None,
        minor_road_route: Optional[str] = None,
        minor_road_name: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        agency_id: Optional[str] = None
    ) -> None:
        """
        Set or update intersection metadata.
        
        Args:
            intersection_id: Primary reference ID
            intersection_name: Human-readable name
            timezone: Controller timezone (default: 'US/Mountain')
            controller_ip: Controller IP for data retrieval
            detection_type: e.g., 'Radar', 'Loops'
            detection_ip: Detection system IP
            major_road_route: e.g., 'US-95'
            major_road_name: e.g., 'Main St'
            minor_road_route: e.g., 'SR-44'
            minor_road_name: e.g., 'Oak Ave'
            latitude: Decimal degrees
            longitude: Decimal degrees
            agency_id: Managing agency identifier
        
        Example:
            >>> manager.set_metadata(
            ...     intersection_id='INT_1001',
            ...     controller_ip='10.70.10.51',
            ...     detection_type='Radar',
            ...     major_road_route='US-95',
            ...     major_road_name='Main St',
            ...     minor_road_name='Oak Ave',
            ...     timezone='US/Mountain',
            ...     latitude=43.6150,
            ...     longitude=-116.2023
            ... )
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        cursor = self.conn.cursor()
        
        # Upsert into single-row table (lock_id=1 enforced by CHECK constraint)
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (
                lock_id,
                intersection_id, intersection_name, timezone,
                controller_ip, detection_type, detection_ip,
                major_road_route, major_road_name,
                minor_road_route, minor_road_name,
                latitude, longitude, agency_id
            ) VALUES (
                1, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?
            )
        """, (
            intersection_id, intersection_name, timezone,
            controller_ip, detection_type, detection_ip,
            major_road_route, major_road_name,
            minor_road_route, minor_road_name,
            latitude, longitude, agency_id
        ))
        
        self.conn.commit()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get intersection metadata.
        
        Returns:
            Dictionary with metadata fields. Returns default dict with timezone
            if metadata has not been set or table does not exist (legacy DB).
        
        Example:
            >>> metadata = manager.get_metadata()
            >>> print(f"Intersection: {metadata['intersection_id']}")
            >>> print(f"Location: {metadata['major_road_name']} & {metadata['minor_road_name']}")
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM metadata WHERE lock_id = 1")
        except sqlite3.OperationalError:
            # Table doesn't exist (legacy DB)
            return {'timezone': 'US/Mountain'}
        
        row = cursor.fetchone()
        if not row:
            return {'timezone': 'US/Mountain'}
        
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))


def init_db(db_path: Path) -> None:
    """
    Convenience function to initialize a database.
    
    Args:
        db_path: Path to SQLite database file
    """
    with DatabaseManager(db_path) as manager:
        manager.init_db()


def import_config(csv_path: Path, db_path: Path) -> None:
    """
    Convenience function to import configuration.
    
    Args:
        csv_path: Path to int_cfg.csv
        db_path: Path to SQLite database file
    """
    with DatabaseManager(db_path) as manager:
        manager.import_config(csv_path)