"""
ATSPM Data Reader - Legacy Adapter (Imperative Shell)

This module queries the normalized SQLite schema and reconstructs
the "flat DataFrame" format used by legacy plotting/analysis scripts.

Package Location: src/atspm/data/reader.py
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd

from .manager import DatabaseManager


def get_legacy_dataframe(
    db_path: Path,
    start: datetime,
    end: datetime,
    event_codes: Optional[List[int]] = None,
    timezone: Optional[str] = None
) -> pd.DataFrame:
    """
    Query events and cycles, returning a legacy-format DataFrame.
    
    This adapter reconstructs the flat DataFrame format used by legacy
    plotting scripts by joining normalized tables.
    
    Args:
        db_path: Path to SQLite database
        start: Start datetime (inclusive)
        end: End datetime (exclusive)
        event_codes: Optional list of event codes to filter
    
    Returns:
        DataFrame with columns: ['TS_start', 'Code', 'ID', 'Cycle_start', 'Coord_plan']
        All timestamps are float (UTC epoch)
    
    Example:
        >>> df = get_legacy_dataframe(
        ...     db_path=Path("intersection.db"),
        ...     start=datetime(2025, 1, 1),
        ...     end=datetime(2025, 1, 2)
        ... )
        >>> df.head()
           TS_start  Code  ID  Cycle_start  Coord_plan
        0  1.735e9     1   2    1.735e9           1.0
        1  1.735e9    82  33    1.735e9           1.0
    """
    # Convert datetime to UTC epoch
    start_epoch = start.timestamp()
    end_epoch = end.timestamp()
    
    # Step 1: Query events
    events_df = _query_events(db_path, start_epoch, end_epoch, event_codes)
    
    if events_df.empty:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=['TS_start', 'Code', 'ID', 'Cycle_start', 'Coord_plan'])
    
    # Step 2: Query cycles (with buffer before start to catch active cycle)
    buffer_seconds = 3600  # 1 hour buffer
    cycles_df = _query_cycles(db_path, start_epoch - buffer_seconds, end_epoch)
    
    # Step 3: Merge events with cycles
    result_df = _merge_events_with_cycles(events_df, cycles_df)
    
    # Step 4: Format columns for legacy compatibility
    result_df = _format_legacy_columns(result_df)

    if timezone:
        result_df['TS_start'] = pd.to_datetime(result_df['TS_start'], unit='s', utc=True).dt.tz_convert(timezone)
        result_df['Cycle_start'] = pd.to_datetime(result_df['Cycle_start'], unit='s', utc=True).dt.tz_convert(timezone)

    return result_df


def _query_events(
    db_path: Path,
    start_epoch: float,
    end_epoch: float,
    event_codes: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Query events from database.
    
    Args:
        db_path: Path to database
        start_epoch: Start timestamp (UTC epoch)
        end_epoch: End timestamp (UTC epoch)
        event_codes: Optional event code filter
    
    Returns:
        DataFrame with columns: ['timestamp', 'event_code', 'parameter']
    """
    with DatabaseManager(db_path) as manager:
        # Use manager's query_events method
        df = manager.query_events(
            start_time=start_epoch,
            end_time=end_epoch,
            event_codes=event_codes
        )
        
        return df


def _query_cycles(
    db_path: Path,
    start_epoch: float,
    end_epoch: float
) -> pd.DataFrame:
    """
    Query cycles from database.
    
    Args:
        db_path: Path to database
        start_epoch: Start timestamp (UTC epoch, with buffer)
        end_epoch: End timestamp (UTC epoch)
    
    Returns:
        DataFrame with columns: ['cycle_start', 'coord_plan']
    """
    with DatabaseManager(db_path) as manager:
        cursor = manager.conn.cursor()
        
        # Query cycles in range
        cursor.execute("""
            SELECT cycle_start, coord_plan
            FROM cycles
            WHERE cycle_start >= ? AND cycle_start < ?
            ORDER BY cycle_start
        """, (start_epoch, end_epoch))
        
        results = cursor.fetchall()
        
        if not results:
            return pd.DataFrame(columns=['cycle_start', 'coord_plan'])
        
        df = pd.DataFrame(results, columns=['cycle_start', 'coord_plan'])
        
        # Ensure proper types
        df['cycle_start'] = df['cycle_start'].astype(float)
        df['coord_plan'] = df['coord_plan'].astype(float)
        
        return df


def _merge_events_with_cycles(
    events_df: pd.DataFrame,
    cycles_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge events with their corresponding cycle information.
    
    Uses merge_asof to assign each event to the most recent cycle
    that started before it.
    
    Args:
        events_df: Events DataFrame
        cycles_df: Cycles DataFrame
    
    Returns:
        Merged DataFrame with cycle information attached
    """
    if cycles_df.empty:
        # No cycles - fill with default values
        events_df['cycle_start'] = events_df['timestamp'].min()
        events_df['coord_plan'] = 0.0
        return events_df
    
    # Ensure both are sorted
    events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    cycles_df = cycles_df.sort_values('cycle_start').reset_index(drop=True)
    
    # Merge using merge_asof (backward direction)
    # Each event gets the cycle that started most recently before it
    merged_df = pd.merge_asof(
        events_df,
        cycles_df,
        left_on='timestamp',
        right_on='cycle_start',
        direction='backward'
    )
    
    # Handle events before first cycle (shouldn't happen, but be safe)
    if merged_df['cycle_start'].isna().any():
        first_cycle = cycles_df['cycle_start'].iloc[0]
        first_coord = cycles_df['coord_plan'].iloc[0]
        
        merged_df['cycle_start'].fillna(first_cycle, inplace=True)
        merged_df['coord_plan'].fillna(first_coord, inplace=True)
    
    return merged_df


def _format_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to match legacy format.
    
    Args:
        df: DataFrame with normalized column names
    
    Returns:
        DataFrame with legacy column names
    """
    # Rename columns
    df = df.rename(columns={
        'timestamp': 'TS_start',
        'event_code': 'Code',
        'parameter': 'ID',
        'cycle_start': 'Cycle_start',
        'coord_plan': 'Coord_plan'
    })
    
    # Ensure types match legacy expectations
    df['TS_start'] = df['TS_start'].astype(float)
    df['Code'] = df['Code'].astype(int)
    df['ID'] = df['ID'].astype(int)
    df['Cycle_start'] = df['Cycle_start'].astype(float)
    df['Coord_plan'] = df['Coord_plan'].astype(float)
    
    # Select only legacy columns (in correct order)
    df = df[['TS_start', 'Code', 'ID', 'Cycle_start', 'Coord_plan']]
    
    # Sort by timestamp
    df = df.sort_values('TS_start').reset_index(drop=True)
    
    return df


def get_config_df(
    db_path: Path,
    date: datetime
) -> pd.Series:
    """
    Get intersection configuration for a specific date.
    
    Returns a flat Series/Dict format for easy access.
    Legacy code can be adapted to use config['TM_EBL'] instead
    of config['TM']['EBL'].
    
    Args:
        db_path: Path to SQLite database
        date: Date to query configuration
    
    Returns:
        Series with configuration values (flat format)
    
    Example:
        >>> config = get_config_df(Path("intersection.db"), datetime(2025, 1, 1))
        >>> config['TM_EBL']
        '5,6,7'
        >>> config['RB_R1']
        '1,2|3,4'
        >>> import json
        >>> exclusions = json.loads(config['TM_Exclusions'])
    """
    with DatabaseManager(db_path) as manager:
        config_dict = manager.get_config_at_date(date)
        
        if config_dict is None:
            return pd.Series(dtype=object)
        
        # Convert to Series for easier access
        config_series = pd.Series(config_dict)
        
        # Remove internal columns
        internal_cols = ['id', 'start_date', 'end_date', 'exclusions']
        config_series = config_series.drop(internal_cols, errors='ignore')
        
        return config_series


def get_config_dict(
    db_path: Path,
    date: datetime
) -> Dict[str, Any]:
    """
    Get intersection configuration as a dictionary.
    
    Alternative to get_config_df for users who prefer dict access.
    
    Args:
        db_path: Path to SQLite database
        date: Date to query configuration
    
    Returns:
        Dictionary with configuration values
    
    Example:
        >>> config = get_config_dict(Path("intersection.db"), datetime(2025, 1, 1))
        >>> config['TM_EBL']
        '5,6,7'
        >>> config['exclusions']  # Already parsed as list
        [{'detector': 33, 'phase': 2, 'status': 'Red'}]
    """
    with DatabaseManager(db_path) as manager:
        config_dict = manager.get_config_at_date(date)
        
        if config_dict is None:
            return {}
        
        # Remove internal ID column
        config_dict.pop('id', None)
        
        return config_dict


def get_date_range(db_path: Path) -> Optional[Dict[str, datetime]]:
    """
    Get the date range of available data.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        Dict with 'start' and 'end' datetime or None if no data
    
    Example:
        >>> date_range = get_date_range(Path("intersection.db"))
        >>> print(f"Data from {date_range['start']} to {date_range['end']}")
    """
    with DatabaseManager(db_path) as manager:
        result = manager.get_event_date_range()
        
        if result is None:
            return None
        
        min_ts, max_ts = result
        
        return {
            'start': datetime.fromtimestamp(min_ts),
            'end': datetime.fromtimestamp(max_ts)
        }


def get_available_dates(db_path: Path) -> List[str]:
    """
    Get list of all dates with processed cycles.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        List of date strings in 'YYYY-MM-DD' format
    
    Example:
        >>> dates = get_available_dates(Path("intersection.db"))
        >>> print(f"Found {len(dates)} days of data")
        >>> for date in dates[:5]:
        ...     print(date)
    """
    with DatabaseManager(db_path) as manager:
        cursor = manager.conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT DATE(cycle_start, 'unixepoch') as cycle_date
            FROM cycles
            ORDER BY cycle_date
        """)
        
        results = cursor.fetchall()
        
        return [row[0] for row in results]


def check_data_quality(
    db_path: Path,
    start: datetime,
    end: datetime
) -> Dict[str, Any]:
    """
    Check data quality for a date range.
    
    Args:
        db_path: Path to SQLite database
        start: Start datetime
        end: End datetime
    
    Returns:
        Dictionary with quality metrics
    
    Example:
        >>> quality = check_data_quality(
        ...     Path("intersection.db"),
        ...     datetime(2025, 1, 1),
        ...     datetime(2025, 1, 2)
        ... )
        >>> print(f"Gap count: {quality['gap_count']}")
        >>> print(f"Event count: {quality['event_count']}")
    """
    start_epoch = start.timestamp()
    end_epoch = end.timestamp()
    
    with DatabaseManager(db_path) as manager:
        cursor = manager.conn.cursor()
        
        # Count total events
        cursor.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE timestamp >= ? AND timestamp < ?
            AND event_code != -1
        """, (start_epoch, end_epoch))
        event_count = cursor.fetchone()[0]
        
        # Count gap markers
        cursor.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE timestamp >= ? AND timestamp < ?
            AND event_code = -1
        """, (start_epoch, end_epoch))
        gap_count = cursor.fetchone()[0]
        
        # Count cycles
        cursor.execute("""
            SELECT COUNT(*)
            FROM cycles
            WHERE cycle_start >= ? AND cycle_start < ?
        """, (start_epoch, end_epoch))
        cycle_count = cursor.fetchone()[0]
        
        # Check if cycles exist for this period
        has_cycles = cycle_count > 0
        
        # Estimate data completeness (no gaps = 100%)
        completeness = 100.0 if gap_count == 0 else max(0, 100.0 - (gap_count * 100.0 / max(1, event_count)))
        
        return {
            'event_count': event_count,
            'gap_count': gap_count,
            'cycle_count': cycle_count,
            'has_cycles': has_cycles,
            'completeness_pct': round(completeness, 2),
            'start': start.isoformat(),
            'end': end.isoformat()
        }


def preview_data(
    db_path: Path,
    date: datetime,
    max_rows: int = 10
) -> pd.DataFrame:
    """
    Preview data for a specific date.
    
    Useful for quick inspection of data quality.
    
    Args:
        db_path: Path to SQLite database
        date: Date to preview
        max_rows: Maximum rows to return
    
    Returns:
        DataFrame with sample data
    
    Example:
        >>> preview = preview_data(Path("intersection.db"), datetime(2025, 1, 1))
        >>> print(preview)
    """
    # Get data for entire day
    start = datetime.combine(date.date(), datetime.min.time())
    end = start + timedelta(days=1)
    
    df = get_legacy_dataframe(db_path, start, end)
    
    # Return first N rows
    return df.head(max_rows)


def convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp columns from float to datetime.
    
    Some legacy scripts may expect datetime objects.
    Use this helper to convert after reading.
    
    Args:
        df: DataFrame with TS_start and Cycle_start as floats
    
    Returns:
        DataFrame with datetime columns
    
    Example:
        >>> df = get_legacy_dataframe(db_path, start, end)
        >>> df = convert_to_datetime(df)
        >>> # Now TS_start is datetime, not float
    """
    df = df.copy()
    
    if 'TS_start' in df.columns:
        df['TS_start'] = pd.to_datetime(df['TS_start'], unit='s')
    
    if 'Cycle_start' in df.columns:
        df['Cycle_start'] = pd.to_datetime(df['Cycle_start'], unit='s')
    
    return df


def get_legacy_dataframe_by_date(
    db_path: Path,
    date_str: str,
    event_codes: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Convenience function to get data for a full day.
    
    Args:
        db_path: Path to SQLite database
        date_str: Date in 'YYYY-MM-DD' format
        event_codes: Optional event code filter
    
    Returns:
        DataFrame for that entire day
    
    Example:
        >>> df = get_legacy_dataframe_by_date(
        ...     Path("intersection.db"),
        ...     "2025-01-01"
        ... )
    """
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Use 'YYYY-MM-DD'")
    
    start = datetime.combine(date.date(), datetime.min.time())
    end = start + timedelta(days=1)
    
    return get_legacy_dataframe(db_path, start, end, event_codes)