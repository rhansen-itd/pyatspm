"""ATSPM Clock-Mark Engine (Imperative Shell)

Orchestrates clock-mark decoding by querying the SQLite database for marker
pedestrian pulses and gap markers, resolving configuration, and delegating all
decoding calculations to the Functional Core (``atspm.analysis.clock_marks``).

Package Location: src/atspm/data/clock_marks.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from .manager import DatabaseManager, db_timezone
from ..analysis.clock_marks import (
    decode_clock_marks,
    marker_peds_from_config,
    send_log_pulses,
)
from ..plotting.clock_marks import plot_clock_drift
from .true_time import load_drift_model
from ..utils.timezone import to_epoch

# Window around requested range to catch pulses & brackets spanning window edges
_FETCH_MARGIN: float = 300.0

# Accepted --start/--end string formats (date-only end extends to end-of-day)
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
)
_DATE_FORMAT = "%Y-%m-%d"


class ClockMarkEngine:
    """Queries the database and produces clock drift and clock set tables.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = ClockMarkEngine(Path("2068_data.db"))

        # In-memory results
        results = engine.decode("2026-09-30", "2026-09-30")

        # Whole-day analysis written to disk
        engine.decode(
            "2026-09-30", "2026-09-30",
            send_log_path=Path("eos-time.jsonl"),
            output_dir=Path("./outputs"),
        )
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize the ClockMarkEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g., ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def decode(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        send_log_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        true_time: bool = False,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Decode clock marks for a chosen period.

        Args:
            start: Period start — ``'YYYY-MM-DD'``, ``'YYYY-MM-DD HH:MM[:SS]'``,
                or a naive local ``datetime``.
            end: Period end (same formats). A date-only *end* is extended
                to end-of-day; a datetime *end* is exclusive as given.
            send_log_path: Optional path to the head unit's ``eos-time.jsonl``.
            output_dir: When provided, write CSVs and HTML plot to this
                directory and return ``None``. When ``None``, return the
                result dict.
            true_time: Also fit the drift model the true-time axis maps
                through (``atspm.data.true_time``): returned as ``"model"``,
                written as ``Clock_Model_*.csv`` and drawn on the plot.

        Returns:
            ``dict`` with keys ``"drift"`` and ``"sets"`` (and ``"model"``
            with *true_time*) — or ``None`` when
            *output_dir* is set, and an empty dict when configuration is missing
            or invalid.
        """
        start_dt, end_dt = self._parse_range(start, end)
        start_epoch = to_epoch(start_dt, self.timezone)
        end_epoch = to_epoch(end_dt, self.timezone)

        config = self._get_config(start_dt)
        try:
            peds = marker_peds_from_config(config)
        except ValueError as exc:
            print(f"  ⚠️  ClockMark: {exc} (Clk_* config invalid)")
            return None if output_dir is not None else {}

        if peds is None:
            print("  ⚠️  ClockMark: no Clk_* config — no clock marks to decode")
            return None if output_dir is not None else {}

        fetch_start = start_epoch - _FETCH_MARGIN
        fetch_end = end_epoch + _FETCH_MARGIN

        with DatabaseManager(self.db_path) as mgr:
            sql = (
                "SELECT timestamp, event_code, parameter FROM events "
                "WHERE timestamp >= ? AND timestamp < ? "
                "AND event_code IN (-1, 89, 90) "
                "ORDER BY timestamp, event_code, parameter"
            )
            events_df = pd.read_sql_query(sql, mgr.conn, params=(fetch_start, fetch_end))

        if not events_df.empty:
            events_df = events_df.astype({
                "timestamp": float,
                "event_code": int,
                "parameter": int,
            })
        else:
            events_df = pd.DataFrame(columns=["timestamp", "event_code", "parameter"])

        send_log_df = None
        if send_log_path is not None:
            records = []
            with open(send_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            send_log_df = send_log_pulses(records)

        drift_df, sets_df = decode_clock_marks(events_df, peds, send_log_df)

        drift_time = drift_df["ts"].fillna(drift_df["off"])
        drift_mask = (drift_time >= start_epoch) & (drift_time < end_epoch)
        drift_df = drift_df.loc[drift_mask].reset_index(drop=True)

        sets_time = sets_df["bracket_on"].fillna(sets_df["bracket_off"])
        sets_mask = (sets_time >= start_epoch) & (sets_time < end_epoch)
        sets_df = sets_df.loc[sets_mask].reset_index(drop=True)

        n_drift = len(drift_df)
        n_drift_flagged = int((drift_df["status"] != "ok").sum()) if n_drift else 0
        n_sets = len(sets_df)
        n_sets_flagged = int((sets_df["status"] != "ok").sum()) if n_sets else 0
        print(
            f"  Clock marks: {n_drift} drift samples ({n_drift_flagged} flagged), "
            f"{n_sets} clock sets ({n_sets_flagged} flagged)."
        )

        model_df = None
        if true_time:
            model_df = load_drift_model(
                self.db_path, start_epoch, end_epoch, send_log_path
            )
            model_df = model_df.loc[
                (model_df["seg_end"] > start_epoch) & (model_df["seg_start"] < end_epoch)
            ].reset_index(drop=True)
            print(f"  Drift model: {model_df['segment'].nunique()} segment(s), {len(model_df)} piece(s).")

        if output_dir is not None:
            self._write_outputs(drift_df, sets_df, output_dir, start_dt, end_dt, model_df)
            return None

        result = {"drift": drift_df, "sets": sets_df}
        if model_df is not None:
            result["model"] = model_df
        return result

    def _read_timezone(self) -> str:
        """Read the intersection timezone from the database."""
        return db_timezone(self.db_path)

    def _get_config(self, date: datetime) -> Dict[str, Any]:
        """Retrieve the active configuration dict for a given date.

        Args:
            date: Reference datetime (naive local) used to select the
                correct temporal config row.

        Returns:
            Config dict (may be empty if no config has been imported).
        """
        with DatabaseManager(self.db_path) as m:
            config = m.get_config_at_date(date)
        return config or {}

    @staticmethod
    def _parse_range(
        start: Union[str, datetime],
        end: Union[str, datetime],
    ) -> tuple[datetime, datetime]:
        """Coerce date or datetime strings to naive local datetimes.

        A date-only *end* is extended by one day (whole-day convention);
        a datetime *end* is used as-is (exclusive), enabling sub-day peak
        periods.

        Args:
            start: Start date/datetime string or naive datetime.
            end: End date/datetime string or naive datetime.

        Returns:
            Tuple of ``(start_dt, end_dt)`` as naive datetimes.

        Raises:
            ValueError: When a string matches no accepted format.
        """
        def parse(value: Union[str, datetime], is_end: bool) -> datetime:
            if isinstance(value, datetime):
                return value
            for fmt in _DATETIME_FORMATS:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            parsed = datetime.strptime(value, _DATE_FORMAT)
            return parsed + timedelta(days=1) if is_end else parsed

        return parse(start, False), parse(end, True)

    def _write_outputs(
        self,
        drift_df: pd.DataFrame,
        sets_df: pd.DataFrame,
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
        model_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Write result DataFrames to CSV and interactive plot to HTML.

        Args:
            drift_df: Decoded drift DataFrame with epoch timestamps.
            sets_df: Decoded clock sets DataFrame with epoch timestamps.
            output_dir: Destination directory (created if absent).
            start_dt: Parsed period start (naive local datetime).
            end_dt: Parsed period end (exclusive).
            model_df: Optional drift model segments, written and drawn when
                given.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def stamp(dt: datetime) -> str:
            if (dt.hour, dt.minute) == (0, 0):
                return f"{dt:%Y_%m_%d}"
            return f"{dt:%Y_%m_%d_%H%M}"

        end_label = (
            end_dt - timedelta(days=1)
            if (end_dt.hour, end_dt.minute) == (0, 0)
            else end_dt
        )
        date_str = f"{stamp(start_dt)}-{stamp(end_label)}"

        drift_csv = drift_df.copy()
        for col in ("ts", "off"):
            if col in drift_csv.columns:
                drift_csv[col] = (
                    pd.to_datetime(drift_csv[col], unit="s", utc=True)
                    .dt.tz_convert(self.timezone)
                    .dt.tz_localize(None)
                )

        sets_csv = sets_df.copy()
        for col in ("bracket_on", "bracket_off", "step_lo", "step_hi"):
            if col in sets_csv.columns:
                sets_csv[col] = (
                    pd.to_datetime(sets_csv[col], unit="s", utc=True)
                    .dt.tz_convert(self.timezone)
                    .dt.tz_localize(None)
                )

        drift_csv_name = f"Clock_Drift_{date_str}.csv"
        drift_csv.to_csv(output_dir / drift_csv_name, index=False)
        print(f"Wrote {drift_csv_name}")

        sets_csv_name = f"Clock_Sets_{date_str}.csv"
        sets_csv.to_csv(output_dir / sets_csv_name, index=False)
        print(f"Wrote {sets_csv_name}")

        if model_df is not None:
            model_csv = model_df.copy()
            for col in ("seg_start", "seg_end", "t_ref"):
                model_csv[col] = (
                    pd.to_datetime(model_csv[col], unit="s", utc=True)
                    .dt.tz_convert(self.timezone)
                    .dt.tz_localize(None)
                )
            model_csv["rate_ppm"] = model_csv["slope"] * 1e6
            model_csv_name = f"Clock_Model_{date_str}.csv"
            model_csv.to_csv(output_dir / model_csv_name, index=False)
            print(f"Wrote {model_csv_name}")

        with DatabaseManager(self.db_path) as mgr:
            metadata = mgr.get_metadata()

        fig = plot_clock_drift(
            drift_df, sets_df, metadata=metadata, timezone=self.timezone,
            model_df=model_df,
        )
        html_name = f"Clock_Drift_{date_str}.html"
        fig.write_html(output_dir / html_name)
        print(f"Wrote {html_name}")


def get_clock_marks(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    send_log_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
    true_time: bool = False,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`ClockMarkEngine`.decode.

    Args:
        db_path: Path to the intersection SQLite database.
        start: Period start (local date or datetime).
        end: Period end (local date or datetime).
        send_log_path: Path to head unit's eos-time.jsonl log.
        output_dir: Write CSVs and HTML plot and return None when provided.
        timezone: Override intersection timezone.
        true_time: Also fit and return the drift model.

    Returns:
        Result dict or None if output_dir is set.
    """
    return ClockMarkEngine(db_path, timezone).decode(
        start=start,
        end=end,
        send_log_path=send_log_path,
        output_dir=output_dir,
        true_time=true_time,
    )
