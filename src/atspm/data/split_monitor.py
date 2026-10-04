"""
ATSPM Split Monitor Engine (Imperative Shell)

Orchestrates UDOT Split Monitor analysis by querying the SQLite database,
reconstructing programmed plan timelines, calculating phase services and
terminations, and delegating core calculations to the Functional Core
(``atspm.analysis.split_monitor``).

Package Location: src/atspm/data/split_monitor.py

Phase Codes & Plan Codes
------------------------
Phase state codes and termination codes:
    Gap marker        : -1
    Phase state codes : 1, 8, 9, 10, 11, 12
    Termination codes : 4, 5, 6
    Ped walk code     : 21

Plan codes:
    Gap marker        : -1
    Plan state codes  : 131–149

Look-back & Fetch Margins
-------------------------
Plan codes (131–149) are logged by controllers as a change log following
a full dump at local midnight. The engine uses a 26-hour look-back for plan
events to ensure the preceding midnight dump is captured.
Phase events use an 1800-second (30-minute) fetch margin to allow clearance
intervals of in-window green intervals to complete.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .manager import DatabaseManager, db_timezone
from .reader import get_events_with_cycles_df
from ..analysis.detector_inference import _to_epoch
from ..analysis.split_monitor import (
    plan_timeline,
    split_monitor,
    split_monitor_stats,
)
from ..plotting.split_monitor import plot_split_monitor
from ..utils.timezone import to_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHASE_SM_CODES: List[int] = [-1, 1, 4, 5, 6, 8, 9, 10, 11, 12, 21]
_PLAN_SM_CODES: List[int] = [-1] + list(range(131, 150))
_ALL_SM_CODES: List[int] = sorted(set(_PHASE_SM_CODES) | set(_PLAN_SM_CODES))
FETCH_MARGIN_S: float = 1800.0
PLAN_LOOKBACK_S: float = 26 * 3600.0

_DATETIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M")
_DATE_FORMAT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SplitMonitorEngine:
    """Queries the database and produces Split Monitor tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = SplitMonitorEngine(Path("sm.db"))
        results = engine.split_monitor("2025-06-02 07:00", "2025-06-02 09:00")
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize SplitMonitorEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def split_monitor(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[Sequence[int]] = None,
        percentiles: Tuple[float, float] = (50.0, 85.0),
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run split monitor analysis and optionally write results to disk.

        Args:
            start: Period start — ``'YYYY-MM-DD'``, ``'YYYY-MM-DD HH:MM'``,
                or a naive local ``datetime``.
            end: Period end. Date-only extends to end-of-day; datetime is exclusive.
            phases: Optional sequence of phase numbers to analyse. When ``None``,
                all discovered phases are analysed.
            percentiles: Exactly two split percentiles to report in stats.
                Default (50, 85).
            make_plot: When ``True`` and ``output_dir`` is provided, generate
                an interactive HTML plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"cycle"``, ``"stats"``, and ``"timeline"`` DataFrames,
            or ``None`` when ``output_dir`` is set.
        """
        start_dt, end_dt = self._parse_range(start, end)
        margin = timedelta(seconds=FETCH_MARGIN_S)
        lookback = timedelta(seconds=PLAN_LOOKBACK_S)

        # 1. Fetch twice, then combine
        phase_events = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt - margin,
            end=end_dt + margin,
            event_codes=_PHASE_SM_CODES,
            timezone=self.timezone,
        )
        plan_events = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt - lookback,
            end=end_dt + margin,
            event_codes=_PLAN_SM_CODES,
            timezone=self.timezone,
        )

        if phase_events.empty and plan_events.empty:
            events = pd.DataFrame()
        elif phase_events.empty:
            events = plan_events.copy()
        elif plan_events.empty:
            events = phase_events.copy()
        else:
            events = pd.concat([phase_events, plan_events], ignore_index=True)
            events = events.drop_duplicates(subset=["timestamp", "event_code", "parameter"])
            events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)

        if events.empty:
            print("  ⚠️  SplitMonitor: no events found for the requested window.")
            return {} if output_dir is None else None

        # 2. Core
        timeline = plan_timeline(events)
        cycle = split_monitor(events, phases=phases, timeline=timeline)

        # 3. Trim to the window
        w0 = to_epoch(start_dt, self.timezone)
        w1 = to_epoch(end_dt, self.timezone)

        if not cycle.empty:
            g_epoch = _to_epoch(cycle["green_ts"])
            cycle = cycle.loc[(g_epoch >= w0) & (g_epoch < w1)].reset_index(drop=True)

        if not timeline.empty:
            tl_start_epoch = _to_epoch(timeline["start"])
            tl_end_epoch = _to_epoch(timeline["end"])
            tl_mask = (tl_end_epoch > w0) & (tl_start_epoch < w1)
            timeline = timeline.loc[tl_mask].reset_index(drop=True)

        if cycle.empty:
            print("  ⚠️  SplitMonitor: no phase services found in the requested window.")
            return {} if output_dir is None else None

        if phases is not None:
            found_phases = set(cycle["phase"].unique())
            for ph in phases:
                if ph not in found_phases:
                    print(f"  ⚠️  SplitMonitor Ph{ph}: no services found in the requested window.")

        # 4. Stats
        stats = split_monitor_stats(cycle, percentiles=percentiles)

        # 5. Outputs
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._format_stamp(start_dt, end_dt)

            cycle_file = output_dir / f"SM_Cycle_{stamp}.csv"
            cycle.to_csv(cycle_file, index=False)
            print(f"Wrote {cycle_file.name}")

            stats_file = output_dir / f"SM_Stats_{stamp}.csv"
            stats.to_csv(stats_file, index=False)
            print(f"Wrote {stats_file.name}")

            timeline_file = output_dir / f"SM_Plans_{stamp}.csv"
            timeline.to_csv(timeline_file, index=False)
            print(f"Wrote {timeline_file.name}")

            if make_plot:
                with DatabaseManager(self.db_path) as m:
                    metadata = m.get_metadata() or {}
                fig = plot_split_monitor(cycle, timeline, metadata=metadata)
                plot_file = output_dir / f"SM_Splits_{stamp}.html"
                fig.write_html(str(plot_file))
                print(f"Wrote {plot_file.name}")

            return None

        return {
            "cycle": cycle,
            "stats": stats,
            "timeline": timeline,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_timezone(self) -> str:
        """Read the intersection timezone from the database."""
        return db_timezone(self.db_path)

    @staticmethod
    def _parse_range(
        start: Union[str, datetime],
        end: Union[str, datetime],
    ) -> Tuple[datetime, datetime]:
        """Coerce date or datetime strings to naive local datetimes.

        A date-only *end* is extended by one day (whole-day convention);
        a datetime *end* is used as-is (exclusive), enabling sub-day peak
        periods.

        Args:
            start: Start date/datetime string or naive datetime.
            end: End date/datetime string or naive datetime.

        Returns:
            Tuple of ``(start_dt, end_dt)`` as naive datetimes.
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

    @staticmethod
    def _format_stamp(start_dt: datetime, end_dt: datetime) -> str:
        """Format start-end timestamp window matching CriticalMovementEngine convention."""
        def stamp(dt: datetime) -> str:
            if (dt.hour, dt.minute) == (0, 0):
                return f"{dt:%Y_%m_%d}"
            return f"{dt:%Y_%m_%d_%H%M}"

        end_label = (
            end_dt - timedelta(days=1)
            if (end_dt.hour, end_dt.minute) == (0, 0)
            else end_dt
        )
        return f"{stamp(start_dt)}-{stamp(end_label)}"


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_split_monitor(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[Sequence[int]] = None,
    percentiles: Tuple[float, float] = (50.0, 85.0),
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`SplitMonitorEngine`.split_monitor."""
    return SplitMonitorEngine(db_path, timezone).split_monitor(
        start=start,
        end=end,
        phases=phases,
        percentiles=percentiles,
        make_plot=make_plot,
        output_dir=output_dir,
    )
