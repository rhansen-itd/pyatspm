"""ATSPM True-Time Model Loader (Imperative Shell)

Fetches the clock-mark rows and gap markers a drift model needs, and hands
them to the Functional Core (``atspm.analysis.clock_marks`` to decode,
``atspm.analysis.true_time`` to fit).  ``events`` keeps controller labels;
callers map them onto true time on read with
``atspm.analysis.true_time.apply_true_time``.

Package Location: src/atspm/data/true_time.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import pytz

from .manager import DatabaseManager
from ..analysis.clock_marks import (
    MARKER_CODES,
    MARKER_ON_CODES,
    decode_clock_marks,
    marker_phases_from_config,
    send_log_pulses,
)
from ..analysis.true_time import drift_model

# How far past the requested range to look for the bounding breaks.  A
# segment normally spans one day (daily set to daily set); past this the
# window is simply cut, which only trims samples from a long fit.
_BREAK_SEARCH: float = 7 * 86400.0

# Margin beyond the bounding break so its own pulses are fetched whole.
_FETCH_MARGIN: float = 300.0

SEND_LOG_NAME = "eos-time.jsonl"

_CODES_SQL = ", ".join(str(c) for c in MARKER_CODES)
_ON_CODES_SQL = ", ".join(str(c) for c in MARKER_ON_CODES)


def load_drift_model(
    db_path: Path,
    start_epoch: float,
    end_epoch: float,
    send_log_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build the drift model covering a label-time range.

    The fetch reaches back to the last clock break before *start_epoch*
    (a set bracket or a gap marker) and forward to the first one after
    *end_epoch*, so the segments touching the range are fitted on all their
    samples (``docs/ROADMAP.md``, S2).

    Args:
        db_path: Path to the intersection SQLite database.
        start_epoch: Range start, UTC epoch (label time).
        end_epoch: Range end, UTC epoch (label time).
        send_log_path: The head unit's ``eos-time.jsonl``.  Defaults to
            ``eos-time.jsonl`` beside the database when that file exists.

    Returns:
        The model frame from ``atspm.analysis.true_time.drift_model``.

    Raises:
        ValueError: When the intersection has no (or invalid) ``Clk_*``
            config, so it writes no clock marks to correct from.
    """
    db_path = Path(db_path)
    with DatabaseManager(db_path) as mgr:
        config = mgr.get_config_at_date(
            datetime.fromtimestamp(start_epoch, tz=pytz.utc)
        ) or {}
        phases = marker_phases_from_config(config)
        if phases is None:
            raise ValueError(
                f"{db_path.name}: no Clk_* config, so no clock marks to "
                f"build a true-time axis from"
            )

        lo, hi = _break_bounds(mgr, phases.set, start_epoch, end_epoch)
        events_df = pd.read_sql_query(
            "SELECT timestamp, event_code, parameter FROM events "
            "WHERE timestamp >= ? AND timestamp < ? "
            f"AND event_code IN (-1, {_CODES_SQL}) "
            "ORDER BY timestamp, event_code, parameter",
            mgr.conn,
            params=(lo, hi),
        )

    events_df = events_df.astype(
        {"timestamp": float, "event_code": int, "parameter": int}
    )

    if send_log_path is None and (db_path.parent / SEND_LOG_NAME).exists():
        send_log_path = db_path.parent / SEND_LOG_NAME
    send_log = _read_send_log(send_log_path) if send_log_path else None

    drift_df, sets_df = decode_clock_marks(events_df, phases, send_log)
    gaps_df = events_df.loc[events_df["event_code"] == -1, ["timestamp", "parameter"]]
    return drift_model(drift_df, sets_df, gaps_df, (lo, hi))


def _break_bounds(
    mgr: DatabaseManager,
    set_phase: int,
    start_epoch: float,
    end_epoch: float,
) -> Tuple[float, float]:
    """Label range from the break before *start_epoch* to the one after *end_epoch*."""
    is_break = f"(event_code = -1 OR (event_code IN ({_ON_CODES_SQL}) AND parameter = ?))"
    before = mgr.conn.execute(
        f"SELECT MAX(timestamp) FROM events WHERE timestamp >= ? "
        f"AND timestamp < ? AND {is_break}",
        (start_epoch - _BREAK_SEARCH, start_epoch, set_phase),
    ).fetchone()[0]
    after = mgr.conn.execute(
        f"SELECT MIN(timestamp) FROM events WHERE timestamp >= ? "
        f"AND timestamp < ? AND {is_break}",
        (end_epoch, end_epoch + _BREAK_SEARCH, set_phase),
    ).fetchone()[0]
    lo = before - _FETCH_MARGIN if before is not None else start_epoch - _BREAK_SEARCH
    hi = after + _FETCH_MARGIN if after is not None else end_epoch + _BREAK_SEARCH
    return float(lo), float(hi)


def _read_send_log(path: Union[str, Path]) -> pd.DataFrame:
    """Parse ``eos-time.jsonl`` into the core's per-pulse frame."""
    with open(path, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return send_log_pulses(records)
