"""
ATSPM Purdue Split Failure Engine (Imperative Shell)

Orchestrates Purdue split failure analysis by querying the SQLite database,
resolving per-phase stop-bar detector configuration, and delegating all
calculations to the Functional Core (``atspm.analysis.split_failures``).

Package Location: src/atspm/data/split_failures.py

Configuration
-------------
Stop-bar detector IDs are read from the active ``config`` row via
``DatabaseManager.get_config_at_date`` and parsed by
``atspm.analysis.critical._parse_stopbar_sets``.  The expected key formats are::

    Det_P{phase}_Stop_Bar   →   "1,2,3"  (comma-separated detector IDs)
    Det_P{phase}_Stopbar    →   "1,2,3"

If no key is found for a requested phase, that phase is skipped with a
printed warning.

Gap Marker Rule
---------------
Gap markers (``event_code == -1``) are always included in the event-code
filter sent to the database, ensuring the Functional Core can enforce state
resets at data discontinuities.  The required codes are::

    Phase state codes : 1, 8, 9, 10, 11, 12
    Detector OFF / ON : 81, 82
    Gap marker        : -1

Data Quality
------------
Time-binned results receive the same ``coverage`` / ``data_quality``
annotation used by ``PhaseEngine`` and ``AogEngine``.  Coverage is
computed from ``ingestion_log`` spans; bins containing gap markers
are downgraded to at most ``"partial"``.  Full-day-missing days are always
dropped.  ``exclude_missing=True`` additionally removes ``"partial"`` and
``"missing"`` bins.

Quality annotation is skipped in ``bin_len="cycle"`` mode because per-cycle
records are inherently gap-bounded.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .critical import CriticalMovementEngine
from .manager import DatabaseManager, db_timezone
from .reader import get_events_with_cycles_df
from ..analysis.critical import _parse_stopbar_sets
from ..analysis.split_failures import (
    AGGREGATES,
    bin_split_failures as _bin_split_failures_core,
    split_failures as _split_failures_core,
)
from ..plotting.split_failures import plot_split_failures
from ..utils.quality import compute_bin_quality

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Event codes required for split failure analysis:
# Gap marker (-1), Phase states (1, 8, 9, 10, 11, 12), Detector states (81, 82)
_ALL_SF_CODES: List[int] = [-1, 1, 8, 9, 10, 11, 12, 81, 82]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SplitFailureEngine:
    """Queries the database and produces Purdue Split Failure tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = SplitFailureEngine(Path("2068_data.db"))

        # Per-cycle and binned tables
        results = engine.split_failures(
            "2025-06-01", "2025-06-07",
            phases=[2, 6],
            aggregate="union",
        )
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize SplitFailureEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def split_failures(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[List[int]] = None,
        aggregate: str = "union",
        threshold: float = 0.79,
        ror_seconds: float = 5.0,
        include_yellow: bool = False,
        bin_len: Union[int, str] = 60,
        exclude_missing: bool = False,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run split failure analysis and optionally write results to disk.

        Args:
            start: Inclusive start date or datetime (``'YYYY-MM-DD'`` or
                ``'YYYY-MM-DD HH:MM'``).
            end: Period end. Date-only extends to end-of-day; datetime is
                exclusive.
            phases: Phase numbers to analyse. When ``None``, all configured
                phases with stop-bar detectors are analysed.
            aggregate: ``"union"`` (default) or ``"mean"`` lane aggregation.
            threshold: Occupancy ratio threshold (default 0.79).
            ror_seconds: Length of red occupancy window (default 5.0).
            include_yellow: Extend GOR window across yellow clearance.
            bin_len: Bin length in minutes, or ``"cycle"``. Default 60.
            exclude_missing: Drop partial/missing bins from binned output.
            make_plot: When ``True`` and ``output_dir`` is provided, generate
                an interactive HTML scatter plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"cycle"``, ``"lane"``, and optionally
            ``"binned"`` DataFrames, or ``None`` when ``output_dir`` is set.

        Raises:
            ValueError: If ``aggregate`` is not ``"union"`` or ``"mean"``.
        """
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")

        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)
        config = self._get_config(start_dt)

        stopbar_sets = _parse_stopbar_sets(config)

        if phases is not None:
            phase_dets = {p: stopbar_sets[p] for p in phases if p in stopbar_sets}
            for ph in phases:
                if ph not in stopbar_sets:
                    print(
                        f"  ⚠️  SplitFailures: no Det_P{ph}_Stop_Bar config found — "
                        f"phase {ph} skipped."
                    )
        else:
            phase_dets = dict(stopbar_sets)

        if not phase_dets:
            _phases_req = phases if phases is not None else "all"
            print(
                f"  ⚠️  SplitFailures: no Stop_Bar detector config found for "
                f"phases={_phases_req}. Check Det_P{{N}}_Stop_Bar keys in int_cfg.csv."
            )
            return {} if output_dir is None else None

        events_df = self._load_events(start_dt, end_dt)
        if events_df.empty:
            print("  ⚠️  SplitFailures: no events found for the requested window.")
            return {} if output_dir is None else None

        cycle_frames: List[pd.DataFrame] = []
        lane_frames: List[pd.DataFrame] = []

        for ph in sorted(phase_dets.keys()):
            dets = sorted(phase_dets[ph])
            ph_cyc, ph_lane = _split_failures_core(
                events_df=events_df,
                phase=ph,
                detector_ids=dets,
                threshold=threshold,
                aggregate=aggregate,
                ror_seconds=ror_seconds,
                include_yellow=include_yellow,
            )
            if ph_cyc.empty:
                print(f"  ⚠️  SplitFailures Ph{ph}: no split windows found — skipping.")
                continue
            cycle_frames.append(ph_cyc)
            if not ph_lane.empty:
                lane_frames.append(ph_lane)

        if not cycle_frames:
            return {} if output_dir is None else None

        cycle_df = pd.concat(cycle_frames, ignore_index=True)
        lane_df = (
            pd.concat(lane_frames, ignore_index=True)
            if lane_frames
            else pd.DataFrame(
                columns=["phase", "green_ts", "det", "g_occ", "r_occ", "gor", "ror5", "fail"]
            )
        )
        cycle_df["aggregate"] = aggregate

        binned_df: Optional[pd.DataFrame] = None
        if bin_len != "cycle":
            bin_int = int(bin_len)
            binned_df = _bin_split_failures_core(cycle_df, bin_len=bin_int)
            if not binned_df.empty:
                binned_df = self._add_quality(
                    binned_df, events_df, start_dt, end_dt, bin_int, exclude_missing
                )

        # Print summary per phase
        for ph in sorted(cycle_df["phase"].unique()):
            sub = cycle_df.loc[cycle_df["phase"] == ph]
            n_cyc = len(sub)
            n_fail = int(sub["fail"].sum())
            sf_pct = (n_fail / n_cyc * 100.0) if n_cyc > 0 else 0.0
            n_lanes = int(sub["n_lanes"].iloc[0]) if not sub.empty else 0
            print(f"    Ph{ph}: {n_cyc} cycles, {n_fail} fails ({sf_pct:.1f}% SF), {n_lanes} lanes")

        if output_dir is not None:
            self._write_outputs(
                cycle_df=cycle_df,
                lane_df=lane_df,
                binned_df=binned_df,
                output_dir=output_dir,
                start_dt=start_dt,
                end_dt=end_dt,
                aggregate=aggregate,
                bin_len=bin_len,
                threshold=threshold,
                make_plot=make_plot,
            )
            return None

        results: Dict[str, pd.DataFrame] = {
            "cycle": cycle_df,
            "lane": lane_df,
        }
        if bin_len != "cycle" and binned_df is not None:
            results["binned"] = binned_df

        return results

    # ------------------------------------------------------------------
    # Data quality (mirrors AogEngine / PhaseEngine)
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
        """Annotate binned split failures DataFrame with coverage and quality labels."""
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
        """Fetch all split-failure relevant events from the database."""
        return get_events_with_cycles_df(
            db_path=self.db_path,
            start=start,
            end=end,
            event_codes=_ALL_SF_CODES,
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
        lane_df: pd.DataFrame,
        binned_df: Optional[pd.DataFrame],
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
        aggregate: str,
        bin_len: Union[int, str],
        threshold: float,
        make_plot: bool,
    ) -> None:
        """Write split failure DataFrames and plot to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._format_stamp(start_dt, end_dt)

        cycle_file = output_dir / f"SF_Cycle_{stamp}_{aggregate}.csv"
        cycle_df.to_csv(cycle_file, index=False)
        print(f"Wrote {cycle_file.name}")

        lane_file = output_dir / f"SF_Lane_{stamp}.csv"
        lane_df.to_csv(lane_file, index=False)
        print(f"Wrote {lane_file.name}")

        if bin_len != "cycle" and binned_df is not None:
            bin_str = f"{int(bin_len)}min"
            binned_file = output_dir / f"SF_{bin_str}_{stamp}_{aggregate}.csv"
            binned_df.to_csv(binned_file, index=False)
            print(f"Wrote {binned_file.name}")

        if make_plot:
            with DatabaseManager(self.db_path) as m:
                metadata = m.get_metadata() or {}
            fig = plot_split_failures(cycle_df, metadata=metadata, threshold=threshold)
            plot_file = output_dir / f"SF_Scatter_{stamp}_{aggregate}.html"
            fig.write_html(str(plot_file))
            print(f"Wrote {plot_file.name}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_split_failures(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[List[int]] = None,
    aggregate: str = "union",
    threshold: float = 0.79,
    ror_seconds: float = 5.0,
    include_yellow: bool = False,
    bin_len: Union[int, str] = 60,
    exclude_missing: bool = False,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`SplitFailureEngine`.split_failures."""
    return SplitFailureEngine(db_path, timezone).split_failures(
        start=start,
        end=end,
        phases=phases,
        aggregate=aggregate,
        threshold=threshold,
        ror_seconds=ror_seconds,
        include_yellow=include_yellow,
        bin_len=bin_len,
        exclude_missing=exclude_missing,
        make_plot=make_plot,
        output_dir=output_dir,
    )
