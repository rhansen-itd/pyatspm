"""
ATSPM Call-to-Service Engine: Pedestrian Delay and Wait Time (Imperative Shell)

Orchestrates pedestrian delay and vehicle wait time analysis (UDOT S-M5) by
querying the SQLite database, resolving detector configuration for stop-bar
presence zones, enforcing gap markers, and delegating core mathematical logic
to the Functional Core (``atspm.analysis.call_service``).

Package Location: src/atspm/data/call_service.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .critical import CriticalMovementEngine
from .manager import DatabaseManager, db_timezone
from .reader import get_events_with_cycles_df
from ..analysis.call_service import (
    DEFAULT_MAX_WAIT_S,
    PED_DELAY_SCHEMA,
    PED_SUMMARY_SCHEMA,
    WAIT_SUMMARY_SCHEMA,
    WAIT_TIME_SCHEMA,
    ped_delay,
    summarize_ped_delay,
    summarize_wait_time,
    wait_time,
)
from ..analysis.detector_inference import _to_epoch
from ..analysis.detector_roles import detector_sets, parse_detector_roles
from ..plotting.call_service import plot_ped_delay, plot_wait_time
from ..utils.timezone import to_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PED_CODES: List[int] = [-1, 21, 22, 45, 90]
_WAIT_CODES: List[int] = [-1, 1, 4, 5, 6, 8, 9, 10, 11, 12, 43, 44]
FETCH_MARGIN_S: float = 1800.0
DROPPING: Tuple[str, ...] = ("auto", "on", "off")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CallServiceEngine:
    """Queries the database and produces Pedestrian Delay and Wait Time tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = CallServiceEngine(Path("2068_data.db"))

        # Pedestrian delay tables
        ped_res = engine.ped_delay("2025-06-01", "2025-06-07")

        # Vehicle wait time tables
        wait_res = engine.wait_time("2025-06-01", "2025-06-07")
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize CallServiceEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def ped_delay(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[List[int]] = None,
        bin_len: int = 60,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run pedestrian delay analysis and optionally write results to disk.

        Args:
            start: Inclusive start date or datetime (``'YYYY-MM-DD'`` or
                ``'YYYY-MM-DD HH:MM'``).
            end: Period end. Date-only extends to end-of-day; datetime is
                exclusive.
            phases: Phase numbers to analyse. When ``None``, all phases
                with pedestrian walk events are analysed.
            bin_len: Aggregation interval in minutes. Default 60.
            make_plot: When True and ``output_dir`` is provided, generate an
                interactive HTML plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"delays"``, ``"binned"``, and ``"plans"``
            DataFrames, or ``None`` when ``output_dir`` is set.
        """
        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)

        # 1. Fetch once with margin
        margin = timedelta(seconds=FETCH_MARGIN_S)
        events_df = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt - margin,
            end=end_dt + margin,
            event_codes=_PED_CODES,
            timezone=self.timezone,
        )
        # 2. Core
        delays, reason = ped_delay(events_df, phases=phases)
        if reason is not None:
            print(f"Ped delay not computable: {reason}")
            return {} if output_dir is None else None

        if phases is not None:
            present_phases = (
                set(delays["phase"].dropna().unique())
                if not delays.empty
                else set()
            )
            for ph in phases:
                if ph not in present_phases:
                    print(f"  ⚠️  PedDelay: phase Ph{ph} has no records")

        if delays.empty:
            print("  ⚠️  PedDelay: no pedestrian delay records found.")
            return {} if output_dir is None else None

        # 3. Trim to window
        w0 = to_epoch(start_dt, self.timezone)
        w1 = to_epoch(end_dt, self.timezone)
        w_epoch = _to_epoch(delays["walk_ts"])
        in_window = (w_epoch >= w0) & (w_epoch < w1)
        delays = delays.loc[in_window].reset_index(drop=True)

        if delays.empty:
            print("  ⚠️  PedDelay: no pedestrian delay records in the requested window.")
            return {} if output_dir is None else None

        delays = delays[PED_DELAY_SCHEMA]

        # 4. Summaries
        binned = summarize_ped_delay(delays, bin_len=bin_len)[PED_SUMMARY_SCHEMA]
        plans = summarize_ped_delay(delays, bin_len=None)[PED_SUMMARY_SCHEMA]

        # 5. Outputs
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._format_stamp(start_dt, end_dt)

            walks_file = output_dir / f"PedDelay_Walks_{stamp}.csv"
            delays.to_csv(walks_file, index=False)
            print(f"Wrote {walks_file.name}")

            binned_file = output_dir / f"PedDelay_{bin_len}min_{stamp}.csv"
            binned.to_csv(binned_file, index=False)
            print(f"Wrote {binned_file.name}")

            plans_file = output_dir / f"PedDelay_Plans_{stamp}.csv"
            plans.to_csv(plans_file, index=False)
            print(f"Wrote {plans_file.name}")

            if make_plot:
                with DatabaseManager(self.db_path) as m:
                    metadata = m.get_metadata() or {}
                fig = plot_ped_delay(delays, binned, metadata=metadata)
                plot_file = output_dir / f"PedDelay_Chart_{stamp}.html"
                fig.write_html(str(plot_file))
                print(f"Wrote {plot_file.name}")

            return None

        return {
            "delays": delays,
            "binned": binned,
            "plans": plans,
        }

    def wait_time(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[List[int]] = None,
        dropping: str = "auto",
        max_wait: Optional[float] = DEFAULT_MAX_WAIT_S,
        bin_len: int = 15,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run wait time analysis and optionally write results to disk.

        Args:
            start: Inclusive start date or datetime (``'YYYY-MM-DD'`` or
                ``'YYYY-MM-DD HH:MM'``).
            end: Period end. Date-only extends to end-of-day; datetime is
                exclusive.
            phases: Phase numbers to analyse. When ``None``, all configured
                phases are analysed.
            dropping: Dropping algorithm selection (``'auto'``, ``'on'``, or
                ``'off'``). Default ``'auto'``.
            max_wait: Maximum wait time in seconds for summary aggregations.
                Default 360.0.
            bin_len: Aggregation interval in minutes. Default 15.
            make_plot: When True and ``output_dir`` is provided, generate an
                interactive HTML plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"windows"``, ``"binned"``, and ``"plans"``
            DataFrames, or ``None`` when ``output_dir`` is set.

        Raises:
            ValueError: If ``dropping`` is not in ``DROPPING``.
        """
        # 1. Dropping validation (before touching the DB)
        if dropping not in DROPPING:
            raise ValueError(f"Unknown dropping {dropping!r}; expected one of {DROPPING}")

        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)
        config = self._get_config(start_dt)

        if dropping == "auto":
            roles = parse_detector_roles(config)
            occupancy_sets = detector_sets(roles, "occupancy")
            drop_phases: Optional[List[int]] = sorted(occupancy_sets.keys())
        elif dropping == "on":
            drop_phases = list(range(1, 17))
        else:  # "off"
            drop_phases = None

        # 2. Fetch events once
        margin = timedelta(seconds=FETCH_MARGIN_S)
        events_df = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt - margin,
            end=end_dt + margin,
            event_codes=_WAIT_CODES,
            timezone=self.timezone,
        )
        if events_df.empty:
            print("  ⚠️  WaitTime: no events found for the requested window.")
            return {} if output_dir is None else None

        # 3. Core
        windows = wait_time(events_df, phases=phases, dropping=drop_phases)

        if not windows.empty and drop_phases:
            drop_set = set(drop_phases)
            for ph in sorted(windows["phase"].dropna().unique()):
                if int(ph) in drop_set:
                    print(f"Ph{int(ph)}: dropping algorithm (presence detection)")

        if phases is not None:
            present_phases = (
                set(windows["phase"].dropna().unique())
                if not windows.empty
                else set()
            )
            for ph in phases:
                if ph not in present_phases:
                    print(f"  ⚠️  WaitTime: phase Ph{ph} has no records")

        if windows.empty:
            print("  ⚠️  WaitTime: no wait time windows found.")
            return {} if output_dir is None else None

        # 4. Trim to window
        w0 = to_epoch(start_dt, self.timezone)
        w1 = to_epoch(end_dt, self.timezone)
        time_series = windows["green_ts"].combine_first(windows["red_ts"])
        t_epoch = _to_epoch(time_series)
        in_window = (t_epoch >= w0) & (t_epoch < w1)
        windows = windows.loc[in_window].reset_index(drop=True)

        if windows.empty:
            print("  ⚠️  WaitTime: no wait time windows in the requested window.")
            return {} if output_dir is None else None

        windows = windows[WAIT_TIME_SCHEMA]

        # 5. Summaries
        binned = summarize_wait_time(windows, bin_len=bin_len, max_wait=max_wait)[WAIT_SUMMARY_SCHEMA]
        plans = summarize_wait_time(windows, bin_len=None, max_wait=max_wait)[WAIT_SUMMARY_SCHEMA]

        # 6. Outputs
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._format_stamp(start_dt, end_dt)

            windows_file = output_dir / f"WaitTime_Windows_{stamp}.csv"
            windows.to_csv(windows_file, index=False)
            print(f"Wrote {windows_file.name}")

            binned_file = output_dir / f"WaitTime_{bin_len}min_{stamp}.csv"
            binned.to_csv(binned_file, index=False)
            print(f"Wrote {binned_file.name}")

            plans_file = output_dir / f"WaitTime_Plans_{stamp}.csv"
            plans.to_csv(plans_file, index=False)
            print(f"Wrote {plans_file.name}")

            if make_plot:
                with DatabaseManager(self.db_path) as m:
                    metadata = m.get_metadata() or {}
                fig = plot_wait_time(windows, binned, metadata=metadata, max_wait=max_wait)
                plot_file = output_dir / f"WaitTime_Chart_{stamp}.html"
                fig.write_html(str(plot_file))
                print(f"Wrote {plot_file.name}")

            return None

        return {
            "windows": windows,
            "binned": binned,
            "plans": plans,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_timezone(self) -> str:
        """Read the intersection timezone from the database."""
        return db_timezone(self.db_path)

    def _get_config(self, date: datetime) -> dict:
        """Retrieve the active configuration dict for a given date."""
        with DatabaseManager(self.db_path) as m:
            config = m.get_config_at_date(date)
        return config or {}

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
# Convenience entry-points
# ---------------------------------------------------------------------------


def get_ped_delay(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[List[int]] = None,
    bin_len: int = 60,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`CallServiceEngine`.ped_delay."""
    return CallServiceEngine(db_path, timezone).ped_delay(
        start=start,
        end=end,
        phases=phases,
        bin_len=bin_len,
        make_plot=make_plot,
        output_dir=output_dir,
    )


def get_wait_time(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[List[int]] = None,
    dropping: str = "auto",
    max_wait: Optional[float] = DEFAULT_MAX_WAIT_S,
    bin_len: int = 15,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`CallServiceEngine`.wait_time."""
    return CallServiceEngine(db_path, timezone).wait_time(
        start=start,
        end=end,
        phases=phases,
        dropping=dropping,
        max_wait=max_wait,
        bin_len=bin_len,
        make_plot=make_plot,
        output_dir=output_dir,
    )
