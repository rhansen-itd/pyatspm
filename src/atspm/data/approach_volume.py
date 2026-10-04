"""
ATSPM Approach Volume Engine (Imperative Shell)

Orchestrates approach volume analysis (UDOT S-M8) by querying the SQLite
database for counts (Code 82) and movement configurations (TM_*), validating
detectors and pairs, and delegating volume calculations, peak hour detection,
K-factor and D-factor computations to the Functional Core
(``atspm.analysis.approach_volume``).

Package Location: src/atspm/data/approach_volume.py

Configuration and Detector Grouping
-----------------------------------
Movement detectors are read from the active ``config`` row and parsed via
``atspm.analysis.counts.parse_movements_from_config``. Detectors are grouped
into approach directions (``NB``, ``SB``, ``EB``, ``WB``) by their movement label
prefix via ``atspm.analysis.approach_volume.direction_detectors``.
Opposing direction pairs (``NB/SB`` and ``EB/WB``) are analysed.

Gap Marker Rule
---------------
Discontinuities are flagged by gap markers (``event_code == -1``) and ingestion
coverage. Bins containing gap markers or partial data are marked incomplete
by ``CountEngine`` (``data_quality != "ok"``). The Functional Core requires
complete bins for rate (vph), directional split (d_split), and rolling peak
hour windows. Days with incomplete bins do not report K-factors.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .critical import CriticalMovementEngine
from .counts import CountEngine
from .manager import DatabaseManager, db_timezone
from ..analysis.approach_volume import (
    approach_volume,
    direction_detectors,
    unparsed_movements,
    BIN_SCHEMA,
    DAY_SCHEMA,
    PAIRS,
    DEFAULT_BIN_LEN,
)
from ..analysis.counts import parse_movements_from_config
from ..plotting.approach_volume import plot_approach_volume


class ApproachVolumeEngine:
    """Queries the database and produces approach volume tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = ApproachVolumeEngine(Path("2068_data.db"))
        results = engine.approach_volume("2025-06-01", "2025-06-07")
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize ApproachVolumeEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def approach_volume(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        bin_len: int = DEFAULT_BIN_LEN,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run approach volume analysis and optionally write results to disk.

        Args:
            start: Inclusive start date or datetime (``'YYYY-MM-DD'`` or
                ``'YYYY-MM-DD HH:MM'``).
            end: Period end. Date-only extends to end-of-day; datetime is
                exclusive.
            bin_len: Time aggregation interval in minutes (must divide 60).
                Default 15.
            make_plot: When True and ``output_dir`` is provided, generate an
                interactive HTML plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"bins"`` and ``"days"`` DataFrames,
            or ``None`` when ``output_dir`` is set.

        Raises:
            ValueError: If ``bin_len`` is not a positive divisor of 60.
        """
        # 1. Validate before touching the DB
        bin_len = int(bin_len)
        if bin_len <= 0 or 60 % bin_len != 0:
            raise ValueError(f"bin_len must divide 60, got {bin_len}")

        # 2. Parse the window
        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)

        # 3. Config at start_dt
        config = self._get_config(start_dt)
        movements = parse_movements_from_config(config)
        dir_dets = direction_detectors(movements)

        if not dir_dets:
            print(
                "  ⚠️  ApproachVolume: no TM_{NB|SB|EB|WB}* movement config found. "
                "Check TM_* rows in int_cfg.csv."
            )
            return {} if output_dir is None else None

        for label in unparsed_movements(movements):
            print(f"  ⚠️  ApproachVolume: unparsed movement TM_{label} has no direction prefix; ignored.")

        # Check for detectors configured in multiple directions (in PAIRS order)
        all_dirs = [d for pair in PAIRS for d in pair]
        all_dets = sorted({d for dets in dir_dets.values() for d in dets})
        for d in all_dets:
            in_dirs = [d_name for d_name in all_dirs if d_name in dir_dets and d in dir_dets[d_name]]
            if len(in_dirs) > 1:
                labels = sorted(f"TM_{mv}" for mv, dets in movements.items() if d in dets)
                mv_str = ", ".join(labels)
                print(
                    f"  ⚠️  ApproachVolume: detector {d} is configured in both "
                    f"{in_dirs[0]} and {in_dirs[1]} ({mv_str}); it counts in both."
                )

        # 4. Counts
        counts = CountEngine(self.db_path, self.timezone).vehicle_counts(
            start_dt, end_dt, bin_len=bin_len, include_detectors=True
        )
        if counts is None or counts.empty:
            print("  ⚠️  ApproachVolume: no count data found for the requested window.")
            return {} if output_dir is None else None

        # 5. Silent detectors: in PAIRS order
        for primary, opposing in PAIRS:
            for d_name in (primary, opposing):
                if d_name in dir_dets:
                    silent = [
                        det for det in dir_dets[d_name]
                        if det not in counts.columns or counts[det].sum() == 0
                    ]
                    if silent:
                        silent_str = ", ".join(str(d) for d in sorted(silent))
                        print(
                            f"  ⚠️  ApproachVolume: {d_name} detector(s) {silent_str} "
                            f"logged no actuation in the window."
                        )

        # 6. Core
        bins, days = approach_volume(counts, movements, bin_len=bin_len)
        if bins.empty and days.empty:
            return {} if output_dir is None else None

        # 7. K-factor note
        if not days.empty and not days["complete_day"].any():
            print("  ℹ️  ApproachVolume: no complete day in the window; K-factor not reported.")

        # 8. Outputs
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._format_stamp(start_dt, end_dt)

            bins_file = output_dir / f"AV_Bins_{bin_len}min_{stamp}.csv"
            bins.to_csv(bins_file, index=False)
            print(f"Wrote {bins_file.name}")

            days_file = output_dir / f"AV_Days_{stamp}.csv"
            days.to_csv(days_file, index=False)
            print(f"Wrote {days_file.name}")

            if make_plot:
                with DatabaseManager(self.db_path) as m:
                    metadata = m.get_metadata() or {}
                fig = plot_approach_volume(bins, days, metadata=metadata)
                chart_file = output_dir / f"AV_Chart_{stamp}.html"
                fig.write_html(str(chart_file))
                print(f"Wrote {chart_file.name}")

            return None

        return {"bins": bins, "days": days}

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
        s_start = stamp(start_dt)
        s_end = stamp(end_label)
        if s_start == s_end:
            return s_start
        return f"{s_start}-{s_end}"


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_approach_volume(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    bin_len: int = DEFAULT_BIN_LEN,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`ApproachVolumeEngine`.approach_volume."""
    return ApproachVolumeEngine(db_path, timezone).approach_volume(
        start=start,
        end=end,
        bin_len=bin_len,
        make_plot=make_plot,
        output_dir=output_dir,
    )
