"""
ATSPM Data Reader - Legacy Adapter (Imperative Shell)

This module queries the normalized SQLite schema and reconstructs
the DataFrame formats required by plotting and analysis functions.

Package Location: src/atspm/data/reader.py

Two output styles are supported:

1. Legacy flat format (get_legacy_dataframe / get_legacy_dataframe_by_date):
   Columns: TS_start, Code, ID, Cycle_start, Coord_plan.
   Used by plot_termination and any code ported from the legacy system.

2. Coordination format (get_coordination_data):
   Returns (df_cycles, df_signal, df_det) as separate DataFrames, each
   retaining normalized column names.  Used by plot_coordination, which
   needs the ring-phase strings on df_cycles and separate signal/detector
   DataFrames rather than a merged flat file.

Timezone note:
   All timestamps stored in the DB are UTC epoch floats.  Functions that
   return human-readable date lists (get_available_dates) convert to local
   dates in Python using pytz, matching the CycleProcessor approach and
   avoiding SQLite's UTC-biased DATE() built-in.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, date, time as dt_time
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import pytz

from .manager import DatabaseManager

# ---------------------------------------------------------------------------
# Signal codes used for coordination plots
# ---------------------------------------------------------------------------
_SIGNAL_CODES: List[int] = [1, 8, 9, 11, 12]
_DETECTOR_CODES: List[int] = [81, 82]
_TERMINATION_CODES: List[int] = [4, 5, 6, 21, 45, 105]

# Cycle look-back buffer: fetch cycles up to this many seconds before the
# requested window start so that the cycle active at window-open is included.
_CYCLE_BUFFER_SECONDS: int = 3600


# ---------------------------------------------------------------------------
# Public API – legacy flat format
# ---------------------------------------------------------------------------

def get_legacy_dataframe(
    db_path: Path,
    start: datetime,
    end: datetime,
    event_codes: Optional[List[int]] = None,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query events and cycles, returning a legacy-format flat DataFrame.

    Reconstructs the flat format used by legacy plotting scripts
    (TS_start, Code, ID, Cycle_start, Coord_plan).

    Args:
        db_path: Path to SQLite database.
        start: Window start (inclusive).  Naive datetimes are treated as
            UTC; use tz-aware datetimes for non-UTC intersections.
        end: Window end (exclusive).
        event_codes: Optional list of ATSPM event codes to filter.
            If ``None``, all codes are returned.
        timezone: Optional pytz timezone string (e.g. ``'US/Mountain'``).
            When provided, TS_start and Cycle_start are converted from UTC
            epoch floats to tz-aware Timestamps before returning.

    Returns:
        DataFrame with columns [TS_start, Code, ID, Cycle_start, Coord_plan].
        Timestamps are UTC epoch floats unless *timezone* is supplied.
        Returns an empty DataFrame with the correct schema if no events found.
    """
    start_epoch = start.timestamp()
    end_epoch = end.timestamp()

    events_df = _query_events(db_path, start_epoch, end_epoch, event_codes)

    if events_df.empty:
        return pd.DataFrame(
            columns=['TS_start', 'Code', 'ID', 'Cycle_start', 'Coord_plan']
        )

    # Fetch cycles with a buffer before start to capture any cycle that
    # started before the window but is still active at window-open.
    cycles_df = _query_cycles(
        db_path,
        start_epoch - _CYCLE_BUFFER_SECONDS,
        end_epoch,
    )

    result_df = _merge_events_with_cycles(events_df, cycles_df)
    result_df = _format_legacy_columns(result_df)

    if timezone:
        result_df['TS_start'] = (
            pd.to_datetime(result_df['TS_start'], unit='s', utc=True)
            .dt.tz_convert(timezone)
        )
        result_df['Cycle_start'] = (
            pd.to_datetime(result_df['Cycle_start'], unit='s', utc=True)
            .dt.tz_convert(timezone)
        )

    return result_df


def get_legacy_dataframe_by_date(
    db_path: Path,
    date_str: str,
    event_codes: Optional[List[int]] = None,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: get a full local calendar day as a legacy DataFrame.

    Args:
        db_path: Path to SQLite database.
        date_str: Local date in ``'YYYY-MM-DD'`` format.
        event_codes: Optional event code filter.
        timezone: pytz timezone string used both to interpret *date_str* as
            a local date and to convert returned timestamps.  If ``None``,
            the timezone is read from the metadata table; falls back to UTC.

    Returns:
        Legacy-format DataFrame for the full calendar day.

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

    return get_legacy_dataframe(
        db_path, start_dt, end_dt, event_codes, timezone=tz_str
    )


# ---------------------------------------------------------------------------
# Public API – coordination format (new)
# ---------------------------------------------------------------------------

def get_coordination_data(
    db_path: Path,
    start: datetime,
    end: datetime,
    timezone: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch the three DataFrames required by ``plot_coordination``.

    Unlike ``get_legacy_dataframe``, this function does **not** merge
    everything into a flat file.  Each DataFrame is returned with its own
    natural schema so that the plotting function can access ring-phase
    strings and keep signal/detector events separate.

    Args:
        db_path: Path to SQLite database.
        start: Window start (inclusive).
        end: Window end (exclusive).
        timezone: Optional pytz timezone string.  When provided, all
            ``Cycle_start`` and ``TS_start`` columns are converted from UTC
            epoch floats to tz-aware Timestamps.

    Returns:
        Tuple of three DataFrames:

        **df_cycles** – one row per cycle with columns::

            Cycle_start  : float (UTC epoch) or Timestamp if tz supplied
            Coord_plan   : float
            r1_phases    : str  e.g. "2,6"
            r2_phases    : str  e.g. "4,8"

        **df_signal** – signal-state events (codes 1, 8, 9, 11, 12) with
        columns::

            TS_start     : float or Timestamp
            Code         : int
            ID           : int   (phase number)
            Cycle_start  : float or Timestamp  (cycle this event belongs to)
            Duration     : float  (seconds in this state; NaN for last
                          event per phase – caller drops these)

        **df_det** – detector events (codes 81, 82) with columns::

            TS_start     : float or Timestamp
            Code         : int
            ID           : int   (detector number)
            Cycle_start  : float or Timestamp
            t_cs         : float  (seconds from Cycle_start to TS_start)
            Duration     : float  (actuation duration; NaN for last per det)

        Any of the three may be an empty DataFrame with the correct schema
        if no data exists for that category in the requested window.
    """
    start_epoch = start.timestamp()
    end_epoch = end.timestamp()

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
        df_signal = pd.DataFrame(
            columns=['TS_start', 'Code', 'ID', 'Cycle_start', 'Duration']
        )

    # --- df_det ---
    det_events = _query_events(db_path, start_epoch, end_epoch, _DETECTOR_CODES)
    if not det_events.empty and not df_cycles.empty:
        df_det = _build_detector_df(det_events, df_cycles)
    else:
        df_det = pd.DataFrame(
            columns=['TS_start', 'Code', 'ID', 'Cycle_start', 't_cs', 'Duration']
        )

    # --- timezone conversion ---
    if timezone:
        df_cycles, df_signal, df_det = _convert_coordination_tz(
            df_cycles, df_signal, df_det, timezone
        )

    # Rename cycles columns to legacy-compatible names expected by
    # plot_coordination (Cycle_start, Coord_plan, r1_phases, r2_phases)
    df_cycles = df_cycles.rename(columns={
        'cycle_start': 'Cycle_start',
        'coord_plan':  'Coord_plan',
    })

    return df_cycles, df_signal, df_det


# ---------------------------------------------------------------------------
# Public API – configuration helpers (unchanged)
# ---------------------------------------------------------------------------

def get_config_df(db_path: Path, date: datetime) -> pd.Series:
    """
    Get intersection configuration for a specific date as a flat Series.

    Args:
        db_path: Path to SQLite database.
        date: Date to query.

    Returns:
        Series with configuration values.  Internal columns (id,
        start_date, end_date, exclusions) are removed.  Returns an empty
        Series if no config exists.

    Example::

        config = get_config_df(Path("intersection.db"), datetime(2025, 1, 1))
        config['RB_R1']         # '1,2|3,4'
        config['Det_P2_Arrival'] # '14,15'
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

    Args:
        db_path: Path to SQLite database.
        date: Date to query.

    Returns:
        Dict with configuration values.  The ``exclusions`` key is already
        parsed to a list of dicts.  Returns ``{}`` if no config exists.

    Example::

        config = get_config_dict(Path("intersection.db"), datetime(2025, 1, 1))
        config['TM_EBL']      # '5,6,7'
        config['exclusions']  # [{'detector': 33, 'phase': 2, 'status': 'Red'}]
    """
    with DatabaseManager(db_path) as manager:
        config_dict = manager.get_config_at_date(date)

    if config_dict is None:
        return {}

    config_dict.pop('id', None)
    return config_dict


def get_det_config(db_path: Path, date: datetime) -> Dict[str, str]:
    """
    Extract detector configuration keys in the legacy ``"P{phase} {Type}"``
    format expected by ``plot_coordination``.

    Reads ``Det_*`` columns from the config table and transforms them back
    to the legacy key format so the plotting function can parse phase and
    detector-type from the key name.

    Args:
        db_path: Path to SQLite database.
        date: Date to query config for.

    Returns:
        Dict mapping ``"P{phase} Arrival"`` / ``"P{phase} Stop Bar"`` /
        ``"P{phase} Occupancy"`` to comma-separated detector number strings.
        Empty dict if no detector config found.

    Example::

        det_cfg = get_det_config(Path("2068_data.db"), datetime(2025, 1, 1))
        # {'P2 Arrival': '14,15', 'P6 Stop Bar': '22', ...}
    """
    config = get_config_dict(db_path, date)
    result: Dict[str, str] = {}

    for key, val in config.items():
        if not key.startswith('Det_') or not val:
            continue
        # Stored as Det_P2_Arrival → "P2 Arrival"
        # Also handle Det_P2_Stop_Bar → "P2 Stop Bar"
        suffix = key[4:]                          # strip "Det_"
        legacy_key = suffix.replace('_', ' ')    # "P2 Arrival", "P2 Stop Bar"
        result[legacy_key] = str(val).strip()

    return result


def get_date_range(db_path: Path) -> Optional[Dict[str, datetime]]:
    """
    Get the min/max timestamp range of ingested events.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Dict ``{'start': datetime, 'end': datetime}`` or ``None`` if the
        events table is empty.
    """
    with DatabaseManager(db_path) as manager:
        result = manager.get_event_date_range()

    if result is None:
        return None

    min_ts, max_ts = result
    return {
        'start': datetime.fromtimestamp(min_ts),
        'end':   datetime.fromtimestamp(max_ts),
    }


def get_available_dates(
    db_path: Path,
    timezone: Optional[str] = None,
) -> List[str]:
    """
    Return the list of local calendar dates that have processed cycles.

    Fixes the previous UTC-biased ``DATE(cycle_start, 'unixepoch')``
    approach.  Epoch values are fetched raw and converted to local dates in
    Python, matching the ``CycleProcessor`` pattern.

    Args:
        db_path: Path to SQLite database.
        timezone: pytz timezone string.  If ``None``, read from metadata;
            fall back to ``'UTC'``.

    Returns:
        Sorted list of ``'YYYY-MM-DD'`` strings in local time.  Empty list
        if no cycles exist.
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
# Public API – data quality / preview (unchanged logic, fixed imports)
# ---------------------------------------------------------------------------

def check_data_quality(
    db_path: Path,
    start: datetime,
    end: datetime,
) -> Dict[str, Any]:
    """
    Check data quality metrics for a date range.

    Args:
        db_path: Path to SQLite database.
        start: Start of range.
        end: End of range.

    Returns:
        Dict with keys: event_count, gap_count, cycle_count, has_cycles,
        completeness_pct, start, end.
    """
    start_epoch = start.timestamp()
    end_epoch = end.timestamp()

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
    Return the first *max_rows* legacy-format rows for a given date.

    Args:
        db_path: Path to SQLite database.
        date: Date to preview.
        max_rows: Maximum rows to return.

    Returns:
        Legacy-format DataFrame (up to *max_rows* rows).
    """
    day_start = datetime.combine(date.date(), dt_time.min)
    day_end = day_start + timedelta(days=1)
    return get_legacy_dataframe(db_path, day_start, day_end).head(max_rows)


def convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float timestamp columns to pandas Timestamps (UTC).

    Args:
        df: DataFrame with TS_start and/or Cycle_start as UTC epoch floats.

    Returns:
        Copy of *df* with those columns converted to ``datetime64[ns, UTC]``.
    """
    df = df.copy()
    for col in ('TS_start', 'Cycle_start'):
        if col in df.columns and pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], unit='s', utc=True)
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

    Args:
        db_path: Path to database.
        start_epoch: Inclusive lower bound (UTC epoch).
        end_epoch: Exclusive upper bound (UTC epoch).
        event_codes: If provided, only these codes are returned.

    Returns:
        DataFrame with columns [timestamp, event_code, parameter].
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

    Selects all five schema columns including the ring-phase strings added
    in the Task 0 schema update.

    Args:
        db_path: Path to database.
        start_epoch: Inclusive lower bound.
        end_epoch: Exclusive upper bound.

    Returns:
        DataFrame with columns
        [cycle_start, coord_plan, detection_method, r1_phases, r2_phases].
        ``r1_phases`` and ``r2_phases`` default to ``'None'`` when the column
        does not exist on an older DB (graceful degradation via COALESCE).
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
            # Table may not yet exist on a fresh or pre-migration DB
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

    Only ``cycle_start`` and ``coord_plan`` are joined (not ring-phase
    strings) because this function feeds ``_format_legacy_columns`` which
    produces the flat legacy schema.

    Args:
        events_df: Events with column [timestamp].
        cycles_df: Cycles with columns [cycle_start, coord_plan, …].

    Returns:
        events_df with [cycle_start, coord_plan] columns attached.
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

    Assigns each signal event to its cycle (``merge_asof``), then computes
    ``Duration`` as the within-phase time until the next event.

    Gap markers (event_code == -1) are absent from *sig_events* because the
    caller already filters to codes 1/8/9/11/12.

    Args:
        sig_events: Events with columns [timestamp, event_code, parameter].
        cycles_df: Cycles with [cycle_start, coord_plan].

    Returns:
        DataFrame with columns
        [TS_start, Code, ID, Cycle_start, Duration].
    """
    df = sig_events.copy().sort_values('timestamp')

    # Assign cycle
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

    # Duration: time to next event for the same phase (parameter)
    df = df.sort_values(['parameter', 'timestamp'])
    df['Duration'] = (
        df.groupby('parameter')['timestamp'].shift(-1) - df['timestamp']
    )

    df = df.rename(columns={
        'timestamp':   'TS_start',
        'event_code':  'Code',
        'parameter':   'ID',
        'cycle_start': 'Cycle_start',
    })

    cols = ['TS_start', 'Code', 'ID', 'Cycle_start', 'Duration']
    return df[cols].reset_index(drop=True)


def _build_detector_df(
    det_events: pd.DataFrame,
    cycles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the ``df_det`` DataFrame for ``plot_coordination``.

    Assigns detector events to their cycle, computes ``t_cs`` (time from
    cycle start to event) and ``Duration`` (actuation duration from Code 82
    on to the following Code 81 off, per detector).

    Args:
        det_events: Events with columns [timestamp, event_code, parameter].
            Codes 81 (off) and 82 (on).
        cycles_df: Cycles with [cycle_start, coord_plan].

    Returns:
        DataFrame with columns
        [TS_start, Code, ID, Cycle_start, t_cs, Duration].
    """
    df = det_events.copy().sort_values(['parameter', 'timestamp'])

    # Duration = time to next event per detector (on→off pairing)
    df['Duration'] = (
        df.groupby('parameter')['timestamp'].shift(-1) - df['timestamp']
    )

    # Assign cycle
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

    # t_cs: seconds from cycle start to event timestamp
    df['t_cs'] = df['timestamp'] - df['cycle_start']

    df = df.rename(columns={
        'timestamp':   'TS_start',
        'event_code':  'Code',
        'parameter':   'ID',
        'cycle_start': 'Cycle_start',
    })

    cols = ['TS_start', 'Code', 'ID', 'Cycle_start', 't_cs', 'Duration']
    return df[cols].reset_index(drop=True)


def _convert_coordination_tz(
    df_cycles: pd.DataFrame,
    df_signal: pd.DataFrame,
    df_det: pd.DataFrame,
    timezone: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert UTC epoch float timestamps to tz-aware Timestamps in all three
    coordination DataFrames.

    Args:
        df_cycles: Cycles DataFrame.
        df_signal: Signal events DataFrame.
        df_det: Detector events DataFrame.
        timezone: pytz timezone string.

    Returns:
        Tuple of the three DataFrames with converted timestamps.
    """
    def _to_ts(series: pd.Series) -> pd.Series:
        if pd.api.types.is_float_dtype(series):
            return pd.to_datetime(series, unit='s', utc=True).dt.tz_convert(timezone)
        return series

    df_cycles = df_cycles.copy()
    df_cycles['cycle_start'] = _to_ts(df_cycles['cycle_start'])

    for df in (df_signal, df_det):
        df = df.copy()  # noqa – local alias
    # Re-assign properly (the loop above doesn't mutate the outer refs)
    df_signal = df_signal.copy()
    df_det = df_det.copy()

    for col in ('TS_start', 'Cycle_start'):
        if col in df_signal.columns:
            df_signal[col] = _to_ts(df_signal[col])
        if col in df_det.columns:
            df_det[col] = _to_ts(df_det[col])

    return df_cycles, df_signal, df_det


# ---------------------------------------------------------------------------
# Private helpers – legacy column formatting
# ---------------------------------------------------------------------------

def _format_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename normalized columns to the legacy flat-format names and enforce
    expected dtypes.  Ring-phase columns are intentionally excluded here;
    use ``get_coordination_data`` when those are needed.

    Args:
        df: Merged DataFrame with normalized column names.

    Returns:
        DataFrame with columns [TS_start, Code, ID, Cycle_start, Coord_plan],
        sorted by TS_start.
    """
    df = df.rename(columns={
        'timestamp':   'TS_start',
        'event_code':  'Code',
        'parameter':   'ID',
        'cycle_start': 'Cycle_start',
        'coord_plan':  'Coord_plan',
    })

    df['TS_start']    = df['TS_start'].astype(float)
    df['Code']        = df['Code'].astype(int)
    df['ID']          = df['ID'].astype(int)
    df['Cycle_start'] = df['Cycle_start'].astype(float)
    df['Coord_plan']  = df['Coord_plan'].astype(float)

    return (
        df[['TS_start', 'Code', 'ID', 'Cycle_start', 'Coord_plan']]
        .sort_values('TS_start')
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Private helpers – timezone resolution
# ---------------------------------------------------------------------------

def _resolve_timezone(db_path: Path) -> str:
    """
    Read the intersection timezone from the metadata table.

    Args:
        db_path: Path to SQLite database.

    Returns:
        pytz timezone string; ``'UTC'`` if metadata is absent or unpopulated.
    """
    try:
        with DatabaseManager(db_path) as manager:
            meta = manager.get_metadata()
            if meta and meta.get('timezone'):
                return meta['timezone']
    except Exception:
        pass
    return 'UTC'


def _local_day_to_epoch_range(
    local_date: date,
    tz_str: str,
) -> Tuple[float, float]:
    """
    Convert a local calendar date to UTC epoch ``[start, end)`` bounds.

    Args:
        local_date: The local calendar date.
        tz_str: pytz timezone string.

    Returns:
        ``(start_epoch, end_epoch)`` as UTC floats covering midnight-to-midnight
        in the given timezone.
    """
    tz = pytz.timezone(tz_str)
    local_midnight = tz.localize(datetime.combine(local_date, dt_time.min))
    return (
        local_midnight.timestamp(),
        (local_midnight + timedelta(days=1)).timestamp(),
    )
