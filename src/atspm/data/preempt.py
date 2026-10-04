"""ATSPM Preemption Engine (Imperative Shell)

Orchestrates preemption analysis by querying the SQLite database for
preemption-related events and gap markers, and delegating episode
reconstruction and summarization to the Functional Core
(``atspm.analysis.preempt``).

Package Location: src/atspm/data/preempt.py
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from .manager import DatabaseManager, db_timezone
from ..analysis.preempt import (
    PREEMPT_CODES,
    preempt_episodes,
    preempt_summary,
)
from ..utils.timezone import resolve_pytz, to_epoch

# Window around requested range to catch requests spanning window boundaries
FETCH_MARGIN_S: float = 1800.0

# Accepted --start/--end string formats (date-only end extends to end-of-day)
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
)
_DATE_FORMAT = "%Y-%m-%d"


class PreemptEngine:
    """Queries the database and produces preemption episode and summary tables.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = PreemptEngine(Path("315_data.db"))
        results = engine.preempt("2026-01-10", "2026-01-12")
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize the PreemptEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g., ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def _read_timezone(self) -> str:
        """Read the intersection timezone from the database."""
        return db_timezone(self.db_path)

    @staticmethod
    def _parse_range(
        start: Union[str, datetime, date],
        end: Union[str, datetime, date],
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
        def parse(value: Union[str, datetime, date], is_end: bool) -> datetime:
            if isinstance(value, datetime):
                return value
            if isinstance(value, date):
                dt = datetime(value.year, value.month, value.day)
                return dt + timedelta(days=1) if is_end else dt
            for fmt in _DATETIME_FORMATS:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            parsed = datetime.strptime(value, _DATE_FORMAT)
            return parsed + timedelta(days=1) if is_end else parsed

        return parse(start, False), parse(end, True)

    def preempt(
        self,
        start: Union[str, datetime, date],
        end: Union[str, datetime, date],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, object]:
        """Analyze preemption episodes for a chosen period.

        Args:
            start: Period start — ``'YYYY-MM-DD'``, ``'YYYY-MM-DD HH:MM[:SS]'``,
                or a naive local ``datetime``.
            end: Period end (same formats). A date-only *end* is extended
                to end-of-day; a datetime *end* is exclusive as given.
            output_dir: Optional directory to write CSV tables to.

        Returns:
            Dict with keys ``"episodes"``, ``"summary"``, and ``"html"``.
        """
        start_dt, end_dt = self._parse_range(start, end)
        w0 = to_epoch(start_dt, self.timezone)
        w1 = to_epoch(end_dt, self.timezone)

        fetch_start = w0 - FETCH_MARGIN_S
        fetch_end = w1 + FETCH_MARGIN_S

        with DatabaseManager(self.db_path) as mgr:
            events_df = mgr.query_events(
                start_time=fetch_start,
                end_time=fetch_end,
                event_codes=list(PREEMPT_CODES),
            )

        episodes = preempt_episodes(events_df)
        if not episodes.empty:
            episodes = episodes.loc[
                (episodes["call_on"] >= w0) & (episodes["call_on"] < w1)
            ].reset_index(drop=True)

        summary = preempt_summary(episodes, self.timezone)

        if output_dir is not None:
            self._write_outputs(episodes, summary, output_dir, start_dt, end_dt)

        return {"episodes": episodes, "summary": summary, "html": None}

    def _write_outputs(
        self,
        episodes: pd.DataFrame,
        summary: pd.DataFrame,
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        """Write preemption CSV tables to disk.

        Args:
            episodes: DataFrame of preemption episodes.
            summary: DataFrame of preemption daily summaries.
            output_dir: Destination directory.
            start_dt: Naive local start datetime.
            end_dt: Naive local end datetime.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        d0 = start_dt.date()
        d1 = (end_dt - timedelta(microseconds=1)).date()
        stamp = f"{d0:%Y_%m_%d}-{d1:%Y_%m_%d}"

        episodes_csv = episodes.copy()
        if episodes_csv.empty:
            episodes_csv["call_on_local"] = pd.Series(dtype="object")
            episodes_csv["timing_plot"] = pd.Series(dtype="object")
        else:
            tz = resolve_pytz(self.timezone)
            dt_call_on = pd.to_datetime(
                episodes_csv["call_on"], unit="s", utc=True
            ).dt.tz_convert(tz)
            episodes_csv["call_on_local"] = (
                dt_call_on.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-5]
            )

            with DatabaseManager(self.db_path) as mgr:
                meta = mgr.get_metadata()
            int_id = meta.get("intersection_id")
            if int_id and not pd.isna(int_id) and str(int_id).strip():
                target_flag = f"--targetid {int_id}"
            else:
                target_flag = f"--target {self.db_path.parent.name}"

            s_dt = pd.to_datetime(
                episodes_csv["call_on"] - 120.0, unit="s", utc=True
            ).dt.tz_convert(tz)
            s_str = s_dt.dt.strftime("%Y-%m-%dT%H:%M:%S")

            end_raw = (
                episodes_csv["exit_ts"]
                .fillna(episodes_csv["call_off"])
                .fillna(episodes_csv["call_on"])
            )
            e_dt = pd.to_datetime(
                end_raw + 120.0, unit="s", utc=True
            ).dt.tz_convert(tz)
            e_str = e_dt.dt.strftime("%Y-%m-%dT%H:%M:%S")

            episodes_csv["timing_plot"] = (
                f"atspm plot-timing-actuation {target_flag} --start "
                + s_str
                + " --end "
                + e_str
            )

        episodes_csv_name = f"Preempt_Episodes_{stamp}.csv"
        episodes_csv.to_csv(output_dir / episodes_csv_name, index=False)
        print(f"Wrote {episodes_csv_name}")

        summary_csv_name = f"Preempt_Summary_{stamp}.csv"
        summary.to_csv(output_dir / summary_csv_name, index=False)
        print(f"Wrote {summary_csv_name}")


def get_preempt(
    db_path: Path,
    start: Union[str, datetime, date],
    end: Union[str, datetime, date],
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Dict[str, object]:
    """Convenience wrapper around :class:`PreemptEngine`.preempt.

    Args:
        db_path: Path to the intersection SQLite database.
        start: Period start (local date or datetime).
        end: Period end (local date or datetime).
        output_dir: Write CSVs to this directory when provided.
        timezone: Override intersection timezone.

    Returns:
        Dict with keys ``"episodes"``, ``"summary"``, and ``"html"``.
    """
    return PreemptEngine(db_path, timezone=timezone).preempt(
        start=start,
        end=end,
        output_dir=output_dir,
    )
