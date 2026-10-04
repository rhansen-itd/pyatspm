"""
ATSPM Yellow and Red Actuations Engine (Imperative Shell)

Orchestrates yellow and red actuation analysis (UDOT S-M4) by querying the
SQLite database, resolving per-phase stop-bar or occupancy detector configuration,
checking for phase overlaps and exclusions, and delegating all calculations to
the Functional Core (``atspm.analysis.yellow_red_actuations``).

Package Location: src/atspm/data/yellow_red_actuations.py

Configuration and Detector Roles
--------------------------------
The default detector role is ``stop_bar``.  ``Det_P{N}_Stop_Bar`` channels are
short count loops just *past* the stop line, so an actuation in red means a
vehicle crossed the line.  ``Det_P{N}_Occupancy`` zones sit *at* the line and
fire for every vehicle that stops on red (315, 2025-12-15, P6: 1.3 red
actuations per cycle on the zones vs 0.02 on the loops).  ``occupancy`` is
offered for sites whose only stop-line detection is presence.

Detector channels are read from the active ``config`` row and parsed via
``atspm.analysis.detector_roles.detector_sets``.  Overlap mappings are parsed
from ``Det_P{N}_Overlap`` via ``atspm.analysis.detector_roles.phase_overlaps``.
Exclusions are parsed from ``TM_Exclusions`` via
``atspm.analysis.counts.parse_exclusions_from_config``.

Gap Marker Rule
---------------
Gap markers (``event_code == -1``) are always included in the event-code filter
sent to the database (``_ALL_YRA_CODES``), ensuring the Functional Core can
enforce state resets and cycle censoring at data discontinuities.
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
from ..analysis.counts import parse_exclusions_from_config
from ..analysis.detector_roles import (
    detector_sets,
    parse_detector_roles,
    phase_overlaps,
)
from ..analysis.yellow_red_actuations import (
    ACTUATION_SCHEMA,
    CYCLE_SCHEMA,
    DEFAULT_SEVERE_SEC,
    summarize_yellow_red,
    yellow_red_actuations,
)
from ..plotting.yellow_red_actuations import plot_yellow_red
from ..utils.timezone import to_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_YRA_CODES: List[int] = [-1, 1, 8, 9, 10, 11, 12, 61, 63, 64, 65, 82]
FETCH_MARGIN_S: float = 1800.0
ROLES: Tuple[str, ...] = ("stop_bar", "occupancy")


def _to_epoch(ts: pd.Series) -> np.ndarray:
    """Convert a timestamp Series (tz-aware datetime or epoch float) to UTC epoch float."""
    if pd.api.types.is_datetime64_any_dtype(ts):
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC")
        return (
            (ts.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC"))
            .dt.total_seconds()
            .to_numpy()
        )
    return ts.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class YellowRedEngine:
    """Queries the database and produces Yellow and Red Actuation tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = YellowRedEngine(Path("2068_data.db"))

        # In-memory per-cycle and summary tables
        results = engine.yellow_red("2025-06-01", "2025-06-07", phases=[2, 6])
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize YellowRedEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def yellow_red(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[List[int]] = None,
        role: str = "stop_bar",
        severe_sec: float = DEFAULT_SEVERE_SEC,
        bin_len: int = 15,
        use_exclusions: bool = True,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run yellow/red actuation analysis and optionally write results to disk.

        Args:
            start: Inclusive start date or datetime (``'YYYY-MM-DD'`` or
                ``'YYYY-MM-DD HH:MM'``).
            end: Period end. Date-only extends to end-of-day; datetime is
                exclusive.
            phases: Phase numbers to analyse. When ``None``, all configured
                phases with detectors in the specified role are analysed.
            role: Detection role to classify (``'stop_bar'`` or ``'occupancy'``).
                Default ``'stop_bar'``.
            severe_sec: A violation is severe when it occurs more than this many
                seconds after red start. Default 4.0.
            bin_len: Aggregation interval in minutes. Default 15.
            use_exclusions: Apply ``TM_Exclusions`` from config when True.
            make_plot: When True and ``output_dir`` is provided, generate an
                interactive HTML plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"cycle"``, ``"actuations"``, ``"binned"``, and
            ``"plans"`` DataFrames, or ``None`` when ``output_dir`` is set.

        Raises:
            ValueError: If ``role`` is not in ``ROLES``.
        """
        # 1. Role validation (before touching the DB)
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")

        # 2. Config at start_dt
        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)
        config = self._get_config(start_dt)

        all_sets = detector_sets(parse_detector_roles(config), role)
        overlaps = phase_overlaps(config)
        exclusions = parse_exclusions_from_config(config) if use_exclusions else None

        role_label = "Stop_Bar" if role == "stop_bar" else "Occupancy"
        if phases is not None:
            phase_dets = {p: all_sets[p] for p in phases if p in all_sets}
            for ph in phases:
                if ph not in all_sets:
                    print(
                        f"  ⚠️  YellowRed: no Det_P{ph}_{role_label} config found — "
                        f"phase Ph{ph} skipped."
                    )
        else:
            phase_dets = dict(all_sets)

        if not phase_dets:
            _phases_req = phases if phases is not None else "all"
            print(
                f"  ⚠️  YellowRed: no {role} detector config found for "
                f"phases={_phases_req}. Check Det_P{{N}}_{role_label} rows in int_cfg.csv."
            )
            return {} if output_dir is None else None

        # 3. Fetch events once
        margin = timedelta(seconds=FETCH_MARGIN_S)
        events_df = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt - margin,
            end=end_dt + margin,
            event_codes=_ALL_YRA_CODES,
            timezone=self.timezone,
        )
        if events_df.empty:
            print("  ⚠️  YellowRed: no events found for the requested window.")
            return {} if output_dir is None else None

        # 4. Core per phase
        w0 = to_epoch(start_dt, self.timezone)
        w1 = to_epoch(end_dt, self.timezone)

        cycle_frames: List[pd.DataFrame] = []
        act_frames: List[pd.DataFrame] = []

        for ph in sorted(phase_dets.keys()):
            dets = sorted(phase_dets[ph])
            overlap_val = overlaps.get(ph)
            if overlap_val is not None:
                overlap_letter = chr(ord("A") + overlap_val - 1)
                print(f"Ph{ph}: classified against overlap {overlap_letter}")

            cy, ac = yellow_red_actuations(
                events_df=events_df,
                phase=ph,
                detector_ids=dets,
                severe_sec=severe_sec,
                overlap=overlap_val,
                exclusions=exclusions,
            )
            if cy.empty:
                print(
                    f"  ⚠️  YellowRed Ph{ph}: no cycles found in the requested window — "
                    f"skipping."
                )
                continue

            g_epoch = _to_epoch(cy["green_ts"])
            in_window = (g_epoch >= w0) & (g_epoch < w1)
            cy = cy.loc[in_window].reset_index(drop=True)

            if cy.empty:
                print(
                    f"  ⚠️  YellowRed Ph{ph}: no cycles found in the requested window — "
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
        binned_df = summarize_yellow_red(cycle_df, bin_len=bin_len)
        plans_df = summarize_yellow_red(cycle_df, bin_len=None)

        # 6. Outputs
        if output_dir is not None:
            self._write_outputs(
                cycle_df=cycle_df,
                act_df=act_df,
                binned_df=binned_df,
                plans_df=plans_df,
                output_dir=output_dir,
                start_dt=start_dt,
                end_dt=end_dt,
                bin_len=bin_len,
                severe_sec=severe_sec,
                make_plot=make_plot,
            )
            return None

        return {
            "cycle": cycle_df,
            "actuations": act_df,
            "binned": binned_df,
            "plans": plans_df,
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
        binned_df: pd.DataFrame,
        plans_df: pd.DataFrame,
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
        bin_len: int,
        severe_sec: float,
        make_plot: bool,
    ) -> None:
        """Write yellow/red actuation DataFrames and plot to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._format_stamp(start_dt, end_dt)

        cycle_file = output_dir / f"YRA_Cycle_{stamp}.csv"
        cycle_df.to_csv(cycle_file, index=False)
        print(f"Wrote {cycle_file.name}")

        act_file = output_dir / f"YRA_Actuations_{stamp}.csv"
        act_df.to_csv(act_file, index=False)
        print(f"Wrote {act_file.name}")

        binned_file = output_dir / f"YRA_{bin_len}min_{stamp}.csv"
        binned_df.to_csv(binned_file, index=False)
        print(f"Wrote {binned_file.name}")

        plans_file = output_dir / f"YRA_Plans_{stamp}.csv"
        plans_df.to_csv(plans_file, index=False)
        print(f"Wrote {plans_file.name}")

        if make_plot:
            with DatabaseManager(self.db_path) as m:
                metadata = m.get_metadata() or {}
            fig = plot_yellow_red(cycle_df, act_df, metadata=metadata, severe_sec=severe_sec)
            plot_file = output_dir / f"YRA_Chart_{stamp}.html"
            fig.write_html(str(plot_file))
            print(f"Wrote {plot_file.name}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_yellow_red(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[List[int]] = None,
    role: str = "stop_bar",
    severe_sec: float = DEFAULT_SEVERE_SEC,
    bin_len: int = 15,
    use_exclusions: bool = True,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`YellowRedEngine`.yellow_red."""
    return YellowRedEngine(db_path, timezone).yellow_red(
        start=start,
        end=end,
        phases=phases,
        role=role,
        severe_sec=severe_sec,
        bin_len=bin_len,
        use_exclusions=use_exclusions,
        make_plot=make_plot,
        output_dir=output_dir,
    )
