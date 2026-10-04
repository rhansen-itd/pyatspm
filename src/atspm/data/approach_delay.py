"""
ATSPM Approach Delay Engine (Imperative Shell)

Orchestrates approach delay and Arrival on Red (AoR) analysis by querying the
SQLite database, resolving per-phase advance detector and travel time configuration,
and delegating all calculations to the Functional Core
(``atspm.analysis.approach_delay``).

Package Location: src/atspm/data/approach_delay.py

Configuration
-------------
Advance arrival detector IDs are read from the active ``config`` row via
``DatabaseManager.get_config_at_date`` and parsed by
``atspm.analysis.detector_roles.detector_sets``.  The key is::

    Det_P{phase}_Arrival   →   "7,8"  (comma-separated detector IDs)

Per-phase travel time to the stop line is read from::

    Det_P{phase}_Arrival_Travel  →  "5.0"  (seconds, advance detector to stop line)

If a phase has no ``Det_P{N}_Arrival_Travel`` key, the scalar ``travel_time_sec``
parameter is used and a warning is printed.

Gap Marker Rule
---------------
Gap markers (``event_code == -1``) are always included in the event-code
filter sent to the database, ensuring the Functional Core can enforce state
resets at data discontinuities.  The required codes are::

    Gap marker        : -1
    Phase state codes : 1, 8, 9, 10, 11, 12
    Detector ON       : 82

Data Quality
------------
Time-binned results receive the same ``coverage`` / ``data_quality``
annotation used by ``PhaseEngine``, ``AogEngine``, and ``SplitFailureEngine``.
Coverage is computed from ``ingestion_log`` spans; bins containing gap markers
are downgraded to at most ``"partial"``.  Full-day-missing days are always
dropped.  ``exclude_missing=True`` additionally removes ``"partial"`` and
``"missing"`` bins.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .critical import CriticalMovementEngine
from .manager import DatabaseManager, db_timezone
from .reader import get_events_with_cycles_df
from ..analysis.approach_delay import (
    CYCLE_SCHEMA,
    approach_delay as _approach_delay_core,
    bin_approach_delay as _bin_approach_delay_core,
)
from ..analysis.detector_roles import (
    arrival_travel_times,
    detector_sets,
    parse_detector_roles,
)
from ..plotting.approach_delay import plot_approach_delay
from ..utils.quality import compute_bin_quality
from ..utils.timezone import to_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_AD_CODES: List[int] = [-1, 1, 8, 9, 10, 11, 12, 82]  # sorted; gap marker always included
FETCH_MARGIN_S: float = 1800.0


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


class ApproachDelayEngine:
    """Queries the database and produces Approach Delay tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = ApproachDelayEngine(Path("2068_data.db"))

        # Per-cycle and binned tables
        results = engine.approach_delay(
            "2025-06-01", "2025-06-07",
            phases=[2, 6],
            travel_time_sec=5.0,
        )
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize ApproachDelayEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def approach_delay(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[List[int]] = None,
        travel_time_sec: float = 0.0,
        bin_len: Union[int, str] = 15,
        exclude_missing: bool = False,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run approach delay analysis and optionally write results to disk.

        Args:
            start: Inclusive start date or datetime (``'YYYY-MM-DD'`` or
                ``'YYYY-MM-DD HH:MM'``).
            end: Period end. Date-only extends to end-of-day; datetime is
                exclusive.
            phases: Phase numbers to analyse. When ``None``, all configured
                phases with arrival (``Det_P{N}_Arrival``) detectors are analysed.
            travel_time_sec: Default travel time in seconds from advance detector
                to stop line, used when ``Det_P{N}_Arrival_Travel`` is not
                configured for a phase.
            bin_len: Bin length in minutes, or ``"cycle"``. Default 15.
            exclude_missing: Drop partial/missing bins from binned output.
            make_plot: When ``True`` and ``output_dir`` is provided, generate
                an interactive HTML plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"cycle"`` and optionally ``"binned"`` DataFrames,
            or ``None`` when ``output_dir`` is set.
        """
        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)
        config = self._get_config(start_dt)

        arrival_sets = detector_sets(parse_detector_roles(config), "arrival")

        if phases is not None:
            phase_dets = {p: arrival_sets[p] for p in phases if p in arrival_sets}
            for ph in phases:
                if ph not in arrival_sets:
                    print(
                        f"  ⚠️  ApproachDelay: no Det_P{ph}_Arrival config found — "
                        f"phase {ph} skipped."
                    )
        else:
            phase_dets = dict(arrival_sets)

        if not phase_dets:
            _phases_req = phases if phases is not None else "all"
            print(
                f"  ⚠️  ApproachDelay: no arrival detector config found for "
                f"phases={_phases_req}. Check Det_P{{N}}_Arrival (P{{N}} Arrival) rows in int_cfg.csv."
            )
            return {} if output_dir is None else None

        # Travel time per phase
        tt = arrival_travel_times(config)

        # Margin fetch
        margin = timedelta(seconds=FETCH_MARGIN_S)
        events_df = self._load_events(start_dt - margin, end_dt + margin)
        if events_df.empty:
            print("  ⚠️  ApproachDelay: no events found for the requested window.")
            return {} if output_dir is None else None

        w0 = to_epoch(start_dt, self.timezone)
        w1 = to_epoch(end_dt, self.timezone)

        cycle_frames: List[pd.DataFrame] = []

        for ph in sorted(phase_dets.keys()):
            dets = sorted(phase_dets[ph])
            if ph in tt:
                ph_tt = tt[ph]
                source = "config"
                travel_val = float(np.mean(list(ph_tt.values())))
            else:
                ph_tt = float(travel_time_sec)
                source = "offset"
                travel_val = float(travel_time_sec)
                print(
                    f"  ⚠️  ApproachDelay Ph{ph}: no Det_P{ph}_Arrival_Travel config found — "
                    f"using offset {travel_time_sec}s."
                )

            ph_cyc = _approach_delay_core(
                events_df=events_df,
                phase=ph,
                detector_ids=dets,
                travel_time_sec=ph_tt,
            )
            if ph_cyc.empty:
                print(
                    f"  ⚠️  ApproachDelay Ph{ph}: no cycles found in the requested window — "
                    f"skipping."
                )
                continue

            g_epoch = _to_epoch(ph_cyc["green_ts"])
            in_window = (g_epoch >= w0) & (g_epoch < w1)
            ph_cyc = ph_cyc.loc[in_window].reset_index(drop=True)

            if ph_cyc.empty:
                print(
                    f"  ⚠️  ApproachDelay Ph{ph}: no cycles found in the requested window — "
                    f"skipping."
                )
                continue

            ph_cyc["travel_time_s"] = travel_val
            ph_cyc["travel_source"] = source
            cycle_frames.append(ph_cyc)

        if not cycle_frames:
            return {} if output_dir is None else None

        cycle_df = pd.concat(cycle_frames, ignore_index=True)
        cycle_df = cycle_df[CYCLE_SCHEMA + ["travel_time_s", "travel_source"]]

        binned_df: Optional[pd.DataFrame] = None
        if bin_len != "cycle":
            bin_int = int(bin_len)
            binned_df = _bin_approach_delay_core(cycle_df[CYCLE_SCHEMA], bin_len=bin_int)
            if not binned_df.empty:
                binned_df = self._add_quality(
                    binned_df, events_df, start_dt, end_dt, bin_int, exclude_missing
                )

        if output_dir is not None:
            self._write_outputs(
                cycle_df=cycle_df,
                binned_df=binned_df,
                output_dir=output_dir,
                start_dt=start_dt,
                end_dt=end_dt,
                bin_len=bin_len,
                make_plot=make_plot,
            )
            return None

        results: Dict[str, pd.DataFrame] = {"cycle": cycle_df}
        if bin_len != "cycle" and binned_df is not None:
            results["binned"] = binned_df

        return results

    # ------------------------------------------------------------------
    # Data quality (mirrors SplitFailureEngine / AogEngine)
    # ------------------------------------------------------------------

    def _add_quality(
        self,
        binned_df: pd.DataFrame,
        events_df: pd.DataFrame,
        start: datetime,
        end: datetime,
        bin_len: int,
        exclude_missing: bool,
    ) -> pd.DataFrame:
        """Annotate binned approach delay DataFrame with coverage and quality labels."""
        if binned_df.empty:
            return binned_df

        with DatabaseManager(self.db_path) as m:
            spans_df = m.get_ingestion_spans()
        quality = compute_bin_quality(
            events_df, spans_df, start, end, bin_len, self.timezone
        )

        out = binned_df.copy()
        out = out.merge(
            quality.reset_index().rename(columns={"index": "time"}),
            on="time",
            how="left",
        )
        out["coverage"] = out["coverage"].fillna(0.0)
        out["data_quality"] = out["data_quality"].fillna("missing")

        out = self._drop_missing_days(out)

        if exclude_missing:
            out = out.loc[out["data_quality"] == "ok"].copy()

        return out.reset_index(drop=True)

    def _drop_missing_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows belonging to local calendar days where every bin is missing."""
        if df.empty or "data_quality" not in df.columns or "time" not in df.columns:
            return df

        local_dates = df["time"].dt.normalize()
        day_has_data = (
            df.assign(_local_date=local_dates)
            .groupby("_local_date")["data_quality"]
            .apply(lambda s: (s != "missing").any())
        )
        good_days = day_has_data.index[day_has_data]
        return df.loc[local_dates.isin(good_days)].drop(
            columns=["_local_date"], errors="ignore"
        )

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

    def _load_events(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch all approach-delay relevant events from the database."""
        return get_events_with_cycles_df(
            db_path=self.db_path,
            start=start,
            end=end,
            event_codes=_ALL_AD_CODES,
            timezone=self.timezone,
        )

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
        binned_df: Optional[pd.DataFrame],
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
        bin_len: Union[int, str],
        make_plot: bool,
    ) -> None:
        """Write approach delay DataFrames and plot to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._format_stamp(start_dt, end_dt)

        cycle_file = output_dir / f"AD_Cycle_{stamp}.csv"
        cycle_df.to_csv(cycle_file, index=False)
        print(f"Wrote {cycle_file.name}")

        if bin_len != "cycle" and binned_df is not None:
            bin_str = f"{int(bin_len)}min"
            binned_file = output_dir / f"AD_{bin_str}_{stamp}.csv"
            binned_df.to_csv(binned_file, index=False)
            print(f"Wrote {binned_file.name}")

        if make_plot:
            if bin_len == "cycle":
                plot_df = _bin_approach_delay_core(cycle_df[CYCLE_SCHEMA], bin_len=15)
            else:
                plot_df = binned_df
            with DatabaseManager(self.db_path) as m:
                metadata = m.get_metadata() or {}
            fig = plot_approach_delay(plot_df, metadata=metadata)
            plot_file = output_dir / f"AD_Delay_{stamp}.html"
            fig.write_html(str(plot_file))
            print(f"Wrote {plot_file.name}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_approach_delay(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[List[int]] = None,
    travel_time_sec: float = 0.0,
    bin_len: Union[int, str] = 15,
    exclude_missing: bool = False,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`ApproachDelayEngine`.approach_delay."""
    return ApproachDelayEngine(db_path, timezone).approach_delay(
        start=start,
        end=end,
        phases=phases,
        travel_time_sec=travel_time_sec,
        bin_len=bin_len,
        exclude_missing=exclude_missing,
        make_plot=make_plot,
        output_dir=output_dir,
    )
