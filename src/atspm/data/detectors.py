"""
Co-Located Detector Discrepancy Data Fetcher (Imperative Shell)

Handles database I/O for the post-hoc detector health analysis.  Delegates
all heavy pandas logic to the Functional Core in
``src/atspm/analysis/detectors.py``.

Package Location: src/atspm/data/detectors.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .manager import DatabaseManager
from ..analysis.detectors import analyze_colocated_discrepancies


def get_detector_discrepancies(
    db_path: Path,
    start: datetime,
    end: datetime,
    det_a_id: int,
    det_b_id: int,
    lag_threshold_sec: float = 2.0,
) -> pd.DataFrame:
    """Fetch events and return co-located detector discrepancy anomalies.

    This is the single public entry-point for the detector health feature.
    It:

    1. Opens the intersection's SQLite database via ``DatabaseManager``.
    2. Queries the ``events`` table for the specified window, filtering on
       ``event_code IN (81, 82, -1)`` and
       ``parameter IN (det_a_id, det_b_id)`` (gap markers use
       ``parameter = -1`` by convention and are always included).
    3. Passes the resulting DataFrame to
       :func:`~atspm.analysis.detectors.analyze_colocated_discrepancies`.
    4. Returns the anomalies DataFrame unchanged.

    Args:
        db_path: Path to the intersection's SQLite database file
            (e.g., ``Path("2068_data.db")``).
        start: Query window start (naive local datetime).  Converted to a
            UTC epoch float via ``datetime.timestamp()``.
        end: Query window end (naive local datetime, exclusive).
        det_a_id: ``parameter`` value identifying Detector A
            (e.g., a radar channel number).
        det_b_id: ``parameter`` value identifying Detector B
            (e.g., a video channel number).
        lag_threshold_sec: Passed through to
            :func:`~atspm.analysis.detectors.analyze_colocated_discrepancies`.
            Defaults to ``2.0``.

    Returns:
        DataFrame of identified anomalies — see
        :func:`~atspm.analysis.detectors.analyze_colocated_discrepancies`
        for the full column schema.  Returns an empty DataFrame (same schema)
        when no anomalies are found or the events table contains no matching
        rows.

    Raises:
        RuntimeError: If the database connection cannot be established or the
            query fails.
        FileNotFoundError: If ``db_path`` does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    start_epoch = start.timestamp()
    end_epoch = end.timestamp()

    # Gap markers are stored with parameter = -1; include them so that
    # _reconstruct_intervals can honour hard resets within the window.
    sql = """
        SELECT timestamp, event_code, parameter
        FROM events
        WHERE timestamp >= :start
          AND timestamp <  :end
          AND (
                (event_code IN (81, 82) AND parameter IN (:det_a, :det_b))
                OR event_code = -1
              )
        ORDER BY timestamp, event_code, parameter
    """
    params = {
        "start": start_epoch,
        "end": end_epoch,
        "det_a": det_a_id,
        "det_b": det_b_id,
    }

    with DatabaseManager(db_path) as manager:
        events_df = pd.read_sql_query(sql, manager.conn, params=params)

    return analyze_colocated_discrepancies(
        events_df=events_df,
        det_a_id=det_a_id,
        det_b_id=det_b_id,
        lag_threshold_sec=lag_threshold_sec,
    )