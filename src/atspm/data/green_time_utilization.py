"""
ATSPM Green Time Utilization Engine (Imperative Shell)

Orchestrates green time utilization analysis (UDOT S-M7) by querying the
SQLite database, resolving per-phase stop-bar or occupancy detector configuration,
checking for phase overlaps and exclusions, and delegating all calculations to
the Functional Core (``atspm.analysis.green_time_utilization``).

Package Location: src/atspm/data/green_time_utilization.py

Configuration and Detector Roles
--------------------------------
The default detector role is ``stop_bar``. ``Det_P{N}_Stop_Bar`` channels are
short count loops just *past* the stop line (detection type 4, lane-by-lane count).
``Det_P{N}_Occupancy`` presence zones sit *at* the stop line and register a queued
vehicle once upon arrival on red, so they miss the queue discharge at the start of
green. ``occupancy`` is offered for sites whose only stop-line detection is presence.

Detector channels are read from the active ``config`` row and parsed via
``atspm.analysis.detector_roles.detector_sets``. Overlap mappings are parsed
from ``Det_P{N}_Overlap`` via ``atspm.analysis.detector_roles.phase_overlaps``.
Exclusions are parsed from ``TM_Exclusions`` via
``atspm.analysis.counts.parse_exclusions_from_config``.

Gap Marker Rule
---------------
Gap markers (``event_code == -1``) are always included in the event-code filter
sent to the database (``_GTU_CODES``), ensuring the Functional Core can
enforce state resets and cycle censoring at data discontinuities.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .critical import CriticalMovementEngine
from .manager import DatabaseManager, db_timezone
from .reader import get_events_with_cycles_df
from ..analysis.counts import parse_exclusions_from_config
from ..analysis.detector_roles import (
    detector_sets,
    parse_detector_roles,
    phase_overlaps,
)
from ..analysis.detector_inference import _to_epoch
from ..analysis.green_time_utilization import (
    ACTUATION_SCHEMA,
    BIN_SCHEMA,
    CYCLE_SCHEMA,
    DEFAULT_BIN_S,
    SPLIT_SCHEMA,
    green_time_utilization,
    summarize_gtu_bins,
    summarize_gtu_splits,
)
from ..analysis.split_monitor import plan_timeline
from ..plotting.green_time_utilization import plot_green_time
from ..utils.timezone import to_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GTU_CODES: List[int] = [-1, 1, 8, 9, 10, 11, 12, 61, 63, 64, 65, 82]
_PLAN_CODES: List[int] = [-1] + list(range(131, 150))
FETCH_MARGIN_S: float = 1800.0
PLAN_LOOKBACK_S: float = 26 * 3600.0
ROLES: Tuple[str, ...] = ("stop_bar", "occupancy")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class GreenTimeEngine:
    """Queries the database and produces Green Time Utilization tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = GreenTimeEngine(Path("2068_data.db"))

        # In-memory per-cycle, binned, and per-plan tables
        results = engine.green_time("2025-06-01", "2025-06-07", phases=[2, 6])
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize GreenTimeEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def green_time(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[List[int]] = None,
        role: str = "stop_bar",
        bin_s: float = DEFAULT_BIN_S,
        bin_len: int = 15,
        use_overlap: bool = False,
        use_exclusions: bool = True,
        max_green_s: Optional[float] = 120.0,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run green time utilization analysis and optionally write results to disk.

        Args:
            start: Inclusive start date or datetime (``'YYYY-MM-DD'`` or
                ``'YYYY-MM-DD HH:MM'``).
            end: Period end. Date-only extends to end-of-day; datetime is
                exclusive.
            phases: Phase numbers to analyse. When ``None``, all configured
                phases with detectors in the specified role are analysed.
            role: Detection role to classify (``'stop_bar'`` or ``'occupancy'``).
                Default ``'stop_bar'``.
            bin_s: Second-of-green bin width in seconds. Default 2.0.
            bin_len: Time aggregation interval in minutes. Default 15.
            use_overlap: When True, measure phase's configured overlap instead.
            use_exclusions: Apply ``TM_Exclusions`` from config when True.
            max_green_s: Green duration threshold in seconds to cap plot heatmap bins.
                Default 120.0.
            make_plot: When True and ``output_dir`` is provided, generate an
                interactive HTML plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"cycle"``, ``"actuations"``, ``"bins"``,
            ``"splits"``, ``"plan_bins"``, and ``"plan_splits"`` DataFrames,
            or ``None`` when ``output_dir`` is set.

        Raises:
            ValueError: If ``role`` is not in ``ROLES`` or ``bin_s <= 0``.
        """
        # 1. Validate before touching the DB
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
        if bin_s <= 0:
            raise ValueError(f"bin_s must be positive, got {bin_s}")

        # 2. Config at start_dt
        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)
        config = self._get_config(start_dt)

        all_sets = detector_sets(parse_detector_roles(config), role)
        overlaps = phase_overlaps(config) if use_overlap else {}
        exclusions = parse_exclusions_from_config(config) if use_exclusions else None

        role_label = "Stop_Bar" if role == "stop_bar" else "Occupancy"
        if phases is not None:
            phase_dets = {p: all_sets[p] for p in phases if p in all_sets}
            for ph in phases:
                if ph not in all_sets:
                    print(
                        f"  ⚠️  GreenTime: no Det_P{ph}_{role_label} config found — "
                        f"phase Ph{ph} skipped."
                    )
        else:
            phase_dets = dict(all_sets)

        if not phase_dets:
            _phases_req = phases if phases is not None else "all"
            print(
                f"  ⚠️  GreenTime: no {role} detector config found for "
                f"phases={_phases_req}. Check Det_P{{N}}_{role_label} rows in int_cfg.csv."
            )
            return {} if output_dir is None else None

        # 3. Fetch twice and combine
        margin = timedelta(seconds=FETCH_MARGIN_S)
        lookback = timedelta(seconds=PLAN_LOOKBACK_S)

        phase_events = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt - margin,
            end=end_dt + margin,
            event_codes=_GTU_CODES,
            timezone=self.timezone,
        )
        plan_events = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt - lookback,
            end=end_dt + margin,
            event_codes=_PLAN_CODES,
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
            print("  ⚠️  GreenTime: no events found for the requested window.")
            return {} if output_dir is None else None

        # 4. Core per phase
        timeline = plan_timeline(events)
        w0 = to_epoch(start_dt, self.timezone)
        w1 = to_epoch(end_dt, self.timezone)

        cycle_frames: List[pd.DataFrame] = []
        act_frames: List[pd.DataFrame] = []

        for ph in sorted(phase_dets.keys()):
            dets = sorted(phase_dets[ph])
            overlap_val = overlaps.get(ph) if use_overlap else None
            if overlap_val is not None:
                overlap_letter = chr(ord("A") + overlap_val - 1)
                print(f"Ph{ph}: green of overlap {overlap_letter}")

            cy, ac = green_time_utilization(
                events_df=events,
                phase=ph,
                detector_ids=dets,
                bin_s=bin_s,
                overlap=overlap_val,
                exclusions=exclusions,
                timeline=timeline,
            )
            if cy.empty:
                print(
                    f"  ⚠️  GreenTime Ph{ph}: no cycles found in the requested window — "
                    f"skipping."
                )
                continue

            g_epoch = _to_epoch(cy["green_ts"])
            in_window = (g_epoch >= w0) & (g_epoch < w1)
            cy = cy.loc[in_window].reset_index(drop=True)

            if cy.empty:
                print(
                    f"  ⚠️  GreenTime Ph{ph}: no cycles found in the requested window — "
                    f"skipping."
                )
                continue

            if not ac.empty:
                ac_g_epoch = _to_epoch(ac["green_ts"])
                ac_in_window = (ac_g_epoch >= w0) & (ac_g_epoch < w1)
                ac = ac.loc[ac_in_window].reset_index(drop=True)

            cycle_frames.append(cy)
            if not ac.empty:
                act_frames.append(ac)

        if not cycle_frames:
            return {} if output_dir is None else None

        cycle_df = pd.concat(cycle_frames, ignore_index=True)[CYCLE_SCHEMA]
        if act_frames:
            act_df = pd.concat(act_frames, ignore_index=True)[ACTUATION_SCHEMA]
        else:
            act_df = pd.DataFrame(columns=ACTUATION_SCHEMA)

        # 5. Summaries
        bins_df = summarize_gtu_bins(cycle_df, act_df, bin_s=bin_s, bin_len=bin_len)
        splits_df = summarize_gtu_splits(cycle_df, bin_len=bin_len)
        plan_bins_df = summarize_gtu_bins(cycle_df, act_df, bin_s=bin_s, bin_len=None)
        plan_splits_df = summarize_gtu_splits(cycle_df, bin_len=None)

        # 6. Outputs
        if output_dir is not None:
            self._write_outputs(
                cycle_df=cycle_df,
                act_df=act_df,
                bins_df=bins_df,
                splits_df=splits_df,
                plan_bins_df=plan_bins_df,
                plan_splits_df=plan_splits_df,
                output_dir=output_dir,
                start_dt=start_dt,
                end_dt=end_dt,
                bin_len=bin_len,
                max_green_s=max_green_s,
                make_plot=make_plot,
            )
            return None

        return {
            "cycle": cycle_df,
            "actuations": act_df,
            "bins": bins_df,
            "splits": splits_df,
            "plan_bins": plan_bins_df,
            "plan_splits": plan_splits_df,
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

    def _write_outputs(
        self,
        cycle_df: pd.DataFrame,
        act_df: pd.DataFrame,
        bins_df: pd.DataFrame,
        splits_df: pd.DataFrame,
        plan_bins_df: pd.DataFrame,
        plan_splits_df: pd.DataFrame,
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
        bin_len: int,
        max_green_s: Optional[float],
        make_plot: bool,
    ) -> None:
        """Write green time utilization DataFrames and plot to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._format_stamp(start_dt, end_dt)

        cycle_file = output_dir / f"GTU_Cycle_{stamp}.csv"
        cycle_df.to_csv(cycle_file, index=False)
        print(f"Wrote {cycle_file.name}")

        act_file = output_dir / f"GTU_Actuations_{stamp}.csv"
        act_df.to_csv(act_file, index=False)
        print(f"Wrote {act_file.name}")

        bins_file = output_dir / f"GTU_Bins_{bin_len}min_{stamp}.csv"
        bins_df.to_csv(bins_file, index=False)
        print(f"Wrote {bins_file.name}")

        splits_file = output_dir / f"GTU_Splits_{bin_len}min_{stamp}.csv"
        splits_df.to_csv(splits_file, index=False)
        print(f"Wrote {splits_file.name}")

        plan_bins_file = output_dir / f"GTU_PlanBins_{stamp}.csv"
        plan_bins_df.to_csv(plan_bins_file, index=False)
        print(f"Wrote {plan_bins_file.name}")

        plan_splits_file = output_dir / f"GTU_PlanSplits_{stamp}.csv"
        plan_splits_df.to_csv(plan_splits_file, index=False)
        print(f"Wrote {plan_splits_file.name}")

        if make_plot:
            with DatabaseManager(self.db_path) as m:
                metadata = m.get_metadata() or {}
            fig = plot_green_time(bins_df, splits_df, metadata=metadata, max_green_s=max_green_s)
            plot_file = output_dir / f"GTU_Chart_{stamp}.html"
            fig.write_html(str(plot_file))
            print(f"Wrote {plot_file.name}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_green_time(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[List[int]] = None,
    role: str = "stop_bar",
    bin_s: float = DEFAULT_BIN_S,
    bin_len: int = 15,
    use_overlap: bool = False,
    use_exclusions: bool = True,
    max_green_s: Optional[float] = 120.0,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`GreenTimeEngine`.green_time."""
    return GreenTimeEngine(db_path, timezone).green_time(
        start=start,
        end=end,
        phases=phases,
        role=role,
        bin_s=bin_s,
        bin_len=bin_len,
        use_overlap=use_overlap,
        use_exclusions=use_exclusions,
        max_green_s=max_green_s,
        make_plot=make_plot,
        output_dir=output_dir,
    )
