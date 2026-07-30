"""
ATSPM Data Reader (Imperative Shell)

This module queries the normalized SQLite schema and reconstructs
the DataFrame formats required by plotting and analysis functions.

Package Location: src/atspm/data/reader.py

Two output styles are supported:

1. Flat format (get_events_with_cycles_df / get_events_with_cycles_df_by_date):
   Columns: timestamp, event_code, parameter, cycle_start, coord_plan.
   Used by plot_termination and other analytical scripts.

2. Coordination format (get_coordination_data):
   Returns (df_cycles, df_signal, df_det) as separate DataFrames, each
   retaining normalized column names. Used by plot_coordination, which
   needs the ring-phase strings on df_cycles and separate signal/detector
   DataFrames rather than a merged flat file.

Timezone note:
   All timestamps stored in the DB are UTC epoch floats.  Functions that
   return human-readable date lists (get_available_dates) convert to local
   dates in Python using pytz, matching the CycleProcessor approach and
   avoiding SQLite's UTC-biased DATE() built-in.

   Window bounds are interpreted in the *intersection's* timezone, never the
   host machine's.  A naive ``start``/``end`` is localized with the explicit
   *timezone* argument when one is given, otherwise with the zone recorded in
   the database's own ``metadata`` table (final fallback UTC).  Aware bounds
   are used as-is.  Conversion goes through ``utils.timezone.to_epoch`` so the
   SQL window and the quality bin grid in ``utils.quality`` agree; calling
   ``datetime.timestamp()`` on a naive bound would resolve it through the host
   clock and shift every row returned.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, date, time as dt_time
from typing import Optional, List, Dict, Any, Tuple, Union

import pandas as pd
import pytz

from .manager import DatabaseManager, db_timezone
from ..utils.timezone import to_epoch

# ---------------------------------------------------------------------------
# Signal codes used for coordination plots
# ---------------------------------------------------------------------------
_SIGNAL_CODES: List[int] = [1, 8, 9, 11, 12]
_DETECTOR_CODES: List[int] = [81, 82]
_TERMINATION_CODES: List[int] = [4, 5, 6, 21, 45, 105]

# Cycle look-back buffer: fetch cycles up to this many seconds before the
# requested window start so that the cycle active at window-open is included.
_CYCLE_BUFFER_SECONDS: int = 3600


def _bounds_to_epoch(
    db_path: Path,
    start: datetime,
    end: datetime,
    timezone: Optional[str] = None,
) -> Tuple[float, float]:
    """Convert a window's bounds to UTC epochs in the intersection's zone.

    Args:
        db_path: Path to the intersection database, consulted for its
            ``metadata`` timezone only when a naive bound needs localizing
            and no explicit *timezone* was supplied.
        start: Window start (inclusive).
        end: Window end (exclusive).
        timezone: Optional pytz timezone string used to interpret naive
            bounds.  Aware bounds keep their own offset either way.

    Returns:
        ``(start_epoch, end_epoch)`` as UTC epoch floats.
    """
    if start.tzinfo is not None and end.tzinfo is not None:
        # Both carry their own offset; no metadata read needed.
        return start.timestamp(), end.timestamp()

    tz_str = timezone or _resolve_timezone(db_path)
    return to_epoch(start, tz_str), to_epoch(end, tz_str)


# ---------------------------------------------------------------------------
# Public API – flat format
# ---------------------------------------------------------------------------

# Changed: Renamed from get_legacy_dataframe to get_events_with_cycles_df
def get_events_with_cycles_df(
    db_path: Path,
    start: datetime,
    end: datetime,
    event_codes: Optional[List[int]] = None,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query events and cycles, returning a joined flat DataFrame.

    Reconstructs a flat format joining raw events with active cycle data
    (timestamp, event_code, parameter, cycle_start, coord_plan).

    Args:
        db_path: Path to SQLite database.
        start: Window start (inclusive).  Naive datetimes are read as
            intersection-local wall clock — see the module Timezone note.
        end: Window end (exclusive), same interpretation as *start*.
        event_codes: Optional list of ATSPM event codes to filter.
            If ``None``, all codes are returned.
        timezone: Optional pytz timezone string (e.g. ``'US/Mountain'``).
            Interprets naive *start*/*end*, and converts the returned
            timestamp and cycle_start from UTC epoch floats to tz-aware
            Timestamps.  Naive bounds fall back to the database's own
            ``metadata`` timezone when this is ``None``.

    Returns:
        DataFrame with columns [timestamp, event_code, parameter, cycle_start, coord_plan].
        Timestamps are UTC epoch floats unless *timezone* is supplied.
        Returns an empty DataFrame with the correct schema if no events found.
    """
    start_epoch, end_epoch = _bounds_to_epoch(db_path, start, end, timezone)

    events_df = _query_events(db_path, start_epoch, end_epoch, event_codes)

    if events_df.empty:
        # Changed: Updated to use native DB column names
        return pd.DataFrame(
            columns=['timestamp', 'event_code', 'parameter', 'cycle_start', 'coord_plan']
        )

    # Fetch cycles with a buffer before start to capture any cycle that
    # started before the window but is still active at window-open.
    cycles_df = _query_cycles(
        db_path,
        start_epoch - _CYCLE_BUFFER_SECONDS,
        end_epoch,
    )

    result_df = _merge_events_with_cycles(events_df, cycles_df)
    
    # Changed: Removed _format_legacy_columns() entirely. Native columns pass through.
    
    if timezone:
        # Changed: Using 'timestamp' and 'cycle_start' instead of legacy names
        result_df['timestamp'] = (
            pd.to_datetime(result_df['timestamp'], unit='s', utc=True)
            .dt.tz_convert(timezone)
        )
        result_df['cycle_start'] = (
            pd.to_datetime(result_df['cycle_start'], unit='s', utc=True)
            .dt.tz_convert(timezone)
        )

    return result_df


# Changed: Renamed from get_legacy_dataframe_by_date to get_events_with_cycles_df_by_date
def get_events_with_cycles_df_by_date(
    db_path: Path,
    date_str: str,
    event_codes: Optional[List[int]] = None,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: get a full local calendar day as a flat DataFrame.

    Args:
        db_path: Path to SQLite database.
        date_str: Local date in ``'YYYY-MM-DD'`` format.
        event_codes: Optional event code filter.
        timezone: pytz timezone string used both to interpret *date_str* as
            a local date and to convert returned timestamps.  If ``None``,
            the timezone is read from the metadata table; falls back to UTC.

    Returns:
        Flat-format DataFrame for the full calendar day.

    Raises:
        ValueError: If *date_str* is not in ``'YYYY-MM-DD'`` format.
    """
    try:
        local_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Use 'YYYY-MM-DD'.")

    tz_str = timezone or _resolve_timezone(db_path)
    start_epoch, end_epoch = _local_day_to_epoch_range(local_date, tz_str)

    start_dt = datetime.fromtimestamp(start_epoch, tz=pytz.utc)
    end_dt = datetime.fromtimestamp(end_epoch, tz=pytz.utc)

    # Changed: Calls the newly renamed function
    return get_events_with_cycles_df(
        db_path, start_dt, end_dt, event_codes, timezone=tz_str
    )


# ---------------------------------------------------------------------------
# Public API – coordination format
# ---------------------------------------------------------------------------

def get_coordination_data(
    db_path: Path,
    start: datetime,
    end: datetime,
    timezone: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch the three DataFrames required by ``plot_coordination``.

    Args:
        db_path: Path to SQLite database.
        start: Window start (inclusive).  Naive datetimes are read as
            intersection-local wall clock — see the module Timezone note.
        end: Window end (exclusive), same interpretation as *start*.
        timezone: Optional pytz timezone string.  Interprets naive
            *start*/*end*, and converts all ``cycle_start`` and ``timestamp``
            columns from UTC epoch floats to tz-aware Timestamps.  Naive
            bounds fall back to the database's own ``metadata`` timezone
            when this is ``None``.

    Returns:
        Tuple of three DataFrames:

        **df_cycles** – one row per cycle with columns::

            cycle_start  : float (UTC epoch) or Timestamp if tz supplied
            coord_plan   : float
            r1_phases    : str  e.g. "2,6"
            r2_phases    : str  e.g. "4,8"

        **df_signal** – signal-state events (codes 1, 8, 9, 11, 12) with
        columns::

            timestamp    : float or Timestamp
            event_code   : int
            parameter    : int   (phase number)
            cycle_start  : float or Timestamp  (cycle this event belongs to)
            Duration     : float  (seconds in this state; NaN for last
                          event per phase – caller drops these)

        **df_det** – detector events (codes 81, 82) with columns::

            timestamp    : float or Timestamp
            event_code   : int
            parameter    : int   (detector number)
            cycle_start  : float or Timestamp
            t_cs         : float  (seconds from cycle_start to timestamp)
            Duration     : float  (actuation duration; NaN for last per det)
    """
    start_epoch, end_epoch = _bounds_to_epoch(db_path, start, end, timezone)

    # --- df_cycles ---
    df_cycles = _query_cycles(
        db_path,
        start_epoch - _CYCLE_BUFFER_SECONDS,
        end_epoch,
    )

    # --- df_signal ---
    sig_events = _query_events(db_path, start_epoch, end_epoch, _SIGNAL_CODES)
    if not sig_events.empty and not df_cycles.empty:
        df_signal = _build_signal_df(sig_events, df_cycles)
    else:
        # Changed: Native columns
        df_signal = pd.DataFrame(
            columns=['timestamp', 'event_code', 'parameter', 'cycle_start', 'Duration']
        )

    # --- df_det ---
    det_events = _query_events(db_path, start_epoch, end_epoch, _DETECTOR_CODES)
    if not det_events.empty and not df_cycles.empty:
        df_det = _build_detector_df(det_events, df_cycles)
    else:
        # Changed: Native columns
        df_det = pd.DataFrame(
            columns=['timestamp', 'event_code', 'parameter', 'cycle_start', 't_cs', 'Duration']
        )

    # --- timezone conversion ---
    if timezone:
        df_cycles, df_signal, df_det = _convert_coordination_tz(
            df_cycles, df_signal, df_det, timezone
        )

    # Changed: Removed the dictionary rename that forced cycle_start -> Cycle_start
    return df_cycles, df_signal, df_det


# ---------------------------------------------------------------------------
# Public API – configuration helpers
# ---------------------------------------------------------------------------

def get_config_df(db_path: Path, date: datetime) -> pd.Series:
    """
    Get intersection configuration for a specific date as a flat Series.
    """
    with DatabaseManager(db_path) as manager:
        config_dict = manager.get_config_at_date(date)

    if config_dict is None:
        return pd.Series(dtype=object)

    config_series = pd.Series(config_dict)
    config_series = config_series.drop(
        ['id', 'start_date', 'end_date', 'exclusions'], errors='ignore'
    )
    return config_series


def get_config_dict(db_path: Path, date: datetime) -> Dict[str, Any]:
    """
    Get intersection configuration for a specific date as a plain dict.
    """
    with DatabaseManager(db_path) as manager:
        config_dict = manager.get_config_at_date(date)

    if config_dict is None:
        return {}

    config_dict.pop('id', None)
    return config_dict


def get_det_config(db_path: Path, date: datetime) -> Dict[str, str]:
    """
    Extract detector configuration keys in the expected ``"P{phase} {Type}"`` format.
    """
    config = get_config_dict(db_path, date)
    result: Dict[str, str] = {}

    for key, val in config.items():
        if not key.startswith('Det_') or not val:
            continue
        suffix = key[4:]
        config_key = suffix.replace('_', ' ')
        result[config_key] = str(val).strip()

    return result


def get_date_range(
    db_path: Path,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, datetime]]:
    """
    Get the min/max timestamp range of ingested events, in local time.

    Args:
        db_path: Path to SQLite database.
        timezone: Optional pytz timezone string.  Defaults to the database's
            own ``metadata`` timezone.

    Returns:
        ``{'start': ..., 'end': ...}`` as tz-aware local datetimes, or
        ``None`` when nothing has been ingested.
    """
    tz = pytz.timezone(timezone or _resolve_timezone(db_path))

    with DatabaseManager(db_path) as manager:
        result = manager.get_event_date_range()

    if result is None:
        return None

    min_ts, max_ts = result
    return {
        'start': datetime.fromtimestamp(min_ts, tz),
        'end':   datetime.fromtimestamp(max_ts, tz),
    }


def get_available_dates(
    db_path: Path,
    timezone: Optional[str] = None,
) -> List[str]:
    """
    Return the list of local calendar dates that have processed cycles.
    """
    tz_str = timezone or _resolve_timezone(db_path)
    tz = pytz.timezone(tz_str)

    with DatabaseManager(db_path) as manager:
        cursor = manager.conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT cycle_start FROM cycles")
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            return []

    if not rows:
        return []

    local_dates: set = {
        datetime.fromtimestamp(row[0], tz).date()
        for row in rows
    }
    return sorted(d.strftime('%Y-%m-%d') for d in local_dates)


# ---------------------------------------------------------------------------
# Public API – data quality / preview
# ---------------------------------------------------------------------------

def check_data_quality(
    db_path: Path,
    start: datetime,
    end: datetime,
    timezone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check data quality metrics for a date range.

    Args:
        db_path: Path to SQLite database.
        start: Window start (inclusive).  Naive datetimes are read as
            intersection-local wall clock — see the module Timezone note.
        end: Window end (exclusive), same interpretation as *start*.
        timezone: Optional pytz timezone string used to interpret naive
            bounds.  Falls back to the database's own ``metadata`` timezone.

    Returns:
        Dict of event/gap/cycle counts and a completeness percentage.
    """
    start_epoch, end_epoch = _bounds_to_epoch(db_path, start, end, timezone)

    with DatabaseManager(db_path) as manager:
        cursor = manager.conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE timestamp >= ? AND timestamp < ? AND event_code != -1",
            (start_epoch, end_epoch),
        )
        event_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE timestamp >= ? AND timestamp < ? AND event_code = -1",
            (start_epoch, end_epoch),
        )
        gap_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM cycles "
            "WHERE cycle_start >= ? AND cycle_start < ?",
            (start_epoch, end_epoch),
        )
        cycle_count = cursor.fetchone()[0]

    completeness = (
        100.0 if gap_count == 0
        else max(0.0, 100.0 - gap_count * 100.0 / max(1, event_count))
    )

    return {
        'event_count':      event_count,
        'gap_count':        gap_count,
        'cycle_count':      cycle_count,
        'has_cycles':       cycle_count > 0,
        'completeness_pct': round(completeness, 2),
        'start':            start.isoformat(),
        'end':              end.isoformat(),
    }


def preview_data(
    db_path: Path,
    date: datetime,
    max_rows: int = 10,
) -> pd.DataFrame:
    """
    Return the first *max_rows* flat-format rows for a given date.
    """
    day_start = datetime.combine(date.date(), dt_time.min)
    day_end = day_start + timedelta(days=1)
    # Changed: Call new function name
    return get_events_with_cycles_df(db_path, day_start, day_end).head(max_rows)


def convert_to_datetime(
    df: pd.DataFrame, 
    columns: Union[List[str], Tuple[str, ...]] = ('timestamp', 'cycle_start'),
    tz: str = 'UTC'
) -> pd.DataFrame:
    """
    Convert float timestamp columns to pandas Timestamps with timezone support.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], unit='s', utc=True).dt.tz_convert(tz)
    return df

# ---------------------------------------------------------------------------
# Private helpers – querying
# ---------------------------------------------------------------------------

def _query_events(
    db_path: Path,
    start_epoch: float,
    end_epoch: float,
    event_codes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Query events table for a UTC epoch range, optionally filtering by code.
    """
    with DatabaseManager(db_path) as manager:
        return manager.query_events(
            start_time=start_epoch,
            end_time=end_epoch,
            event_codes=event_codes,
        )


def _query_cycles(
    db_path: Path,
    start_epoch: float,
    end_epoch: float,
) -> pd.DataFrame:
    """
    Query cycles table for a UTC epoch range.
    """
    sql = """
        SELECT
            cycle_start,
            coord_plan,
            detection_method,
            COALESCE(r1_phases, 'None') AS r1_phases,
            COALESCE(r2_phases, 'None') AS r2_phases
        FROM cycles
        WHERE cycle_start >= ? AND cycle_start < ?
        ORDER BY cycle_start
    """
    empty = pd.DataFrame(
        columns=['cycle_start', 'coord_plan', 'detection_method',
                 'r1_phases', 'r2_phases']
    )

    with DatabaseManager(db_path) as manager:
        try:
            df = pd.read_sql_query(
                sql, manager.conn, params=(start_epoch, end_epoch)
            )
        except Exception:
            return empty

    if df.empty:
        return empty

    df['cycle_start'] = df['cycle_start'].astype(float)
    df['coord_plan']  = pd.to_numeric(df['coord_plan'], errors='coerce').fillna(0.0)
    df['r1_phases']   = df['r1_phases'].fillna('None').astype(str)
    df['r2_phases']   = df['r2_phases'].fillna('None').astype(str)

    return df


# ---------------------------------------------------------------------------
# Private helpers – merging / building coordination DataFrames
# ---------------------------------------------------------------------------

def _merge_events_with_cycles(
    events_df: pd.DataFrame,
    cycles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each event row to its containing cycle via ``merge_asof``.
    """
    if cycles_df.empty:
        events_df = events_df.copy()
        events_df['cycle_start'] = events_df['timestamp'].min()
        events_df['coord_plan']  = 0.0
        return events_df

    events_sorted = events_df.sort_values('timestamp').reset_index(drop=True)
    cycles_sorted = (
        cycles_df[['cycle_start', 'coord_plan']]
        .sort_values('cycle_start')
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        events_sorted,
        cycles_sorted,
        left_on='timestamp',
        right_on='cycle_start',
        direction='backward',
    )

    if merged['cycle_start'].isna().any():
        merged['cycle_start'] = merged['cycle_start'].fillna(
            cycles_sorted['cycle_start'].iloc[0]
        )
        merged['coord_plan'] = merged['coord_plan'].fillna(
            cycles_sorted['coord_plan'].iloc[0]
        )

    return merged


def _build_signal_df(
    sig_events: pd.DataFrame,
    cycles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the ``df_signal`` DataFrame for ``plot_coordination``.
    """
    df = sig_events.copy().sort_values('timestamp')

    cycles_sorted = (
        cycles_df[['cycle_start', 'coord_plan']]
        .sort_values('cycle_start')
        .reset_index(drop=True)
    )
    df = pd.merge_asof(
        df,
        cycles_sorted,
        left_on='timestamp',
        right_on='cycle_start',
        direction='backward',
    )

    df = df.sort_values(['parameter', 'timestamp'])
    df['Duration'] = (
        df.groupby('parameter')['timestamp'].shift(-1) - df['timestamp']
    )

    # Changed: Removed the block renaming columns to legacy names.
    cols = ['timestamp', 'event_code', 'parameter', 'cycle_start', 'Duration']
    return df[cols].reset_index(drop=True)


def _build_detector_df(
    det_events: pd.DataFrame,
    cycles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the ``df_det`` DataFrame for ``plot_coordination``.
    """
    df = det_events.copy().sort_values(['parameter', 'timestamp'])

    df['Duration'] = (
        df.groupby('parameter')['timestamp'].shift(-1) - df['timestamp']
    )

    cycles_sorted = (
        cycles_df[['cycle_start']]
        .sort_values('cycle_start')
        .reset_index(drop=True)
    )
    df = df.sort_values('timestamp')
    df = pd.merge_asof(
        df,
        cycles_sorted,
        left_on='timestamp',
        right_on='cycle_start',
        direction='backward',
    )

    df['t_cs'] = df['timestamp'] - df['cycle_start']

    # Changed: Removed the block renaming columns to legacy names.
    cols = ['timestamp', 'event_code', 'parameter', 'cycle_start', 't_cs', 'Duration']
    return df[cols].reset_index(drop=True)


def _convert_coordination_tz(
    df_cycles: pd.DataFrame,
    df_signal: pd.DataFrame,
    df_det: pd.DataFrame,
    timezone: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert UTC epoch float timestamps to tz-aware Timestamps.
    """
    def _to_ts(series: pd.Series) -> pd.Series:
        if pd.api.types.is_float_dtype(series):
            return pd.to_datetime(series, unit='s', utc=True).dt.tz_convert(timezone)
        return series

    df_cycles = df_cycles.copy()
    df_cycles['cycle_start'] = _to_ts(df_cycles['cycle_start'])

    df_signal = df_signal.copy()
    df_det = df_det.copy()

    # Changed: Iterate over native column names
    for col in ('timestamp', 'cycle_start'):
        if col in df_signal.columns:
            df_signal[col] = _to_ts(df_signal[col])
        if col in df_det.columns:
            df_det[col] = _to_ts(df_det[col])

    return df_cycles, df_signal, df_det

# ---------------------------------------------------------------------------
# Private helpers – timezone resolution
# ---------------------------------------------------------------------------

def _resolve_timezone(db_path: Path) -> str:
    """Read the intersection timezone from the metadata table."""
    return db_timezone(db_path)


def _local_day_to_epoch_range(
    local_date: date,
    tz_str: str,
) -> Tuple[float, float]:
    """Convert a local calendar date to UTC epoch ``[start, end)`` bounds."""
    tz = pytz.timezone(tz_str)
    local_midnight = tz.localize(datetime.combine(local_date, dt_time.min))
    return (
        local_midnight.timestamp(),
        (local_midnight + timedelta(days=1)).timestamp(),
    )