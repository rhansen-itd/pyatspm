"""
ATSPM Left Turn Gap Analysis Engine (Imperative Shell)

Orchestrates left-turn gap analysis (UDOT S-M9) by querying the SQLite database,
resolving per-left opposing through movements and detectors from config, and
delegating calculations to the Functional Core (``atspm.analysis.left_turn_gap``).

Package Location: src/atspm/data/left_turn_gap.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .critical import CriticalMovementEngine
from .manager import DatabaseManager, db_timezone
from .reader import get_events_with_cycles_df
from ..analysis.counts import parse_exclusions_from_config
from ..analysis.left_turn_gap import (
    DEFAULT_BIN_LEN,
    DEFAULT_EDGES,
    DEFAULT_TREND_S,
    GAP_SCHEMA,
    PAIR_SCHEMA,
    check_edges,
    cycle_schema,
    left_turn_gaps,
    left_turn_pairs,
    summarize_left_turn_gaps,
    through_phases,
)
from ..plotting.left_turn_gap import plot_left_turn_gap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LTG_CODES: Tuple[int, ...] = (-1, 1, 8, 9, 10, 11, 12, 81)
FETCH_MARGIN_S: float = 3600.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LeftTurnGapEngine:
    """Queries the database and produces Left Turn Gap tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = LeftTurnGapEngine(Path("2068_data.db"))
        results = engine.left_turn_gap("2025-06-01", "2025-06-07")
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize LeftTurnGapEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def left_turn_gap(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        lefts: Optional[List[str]] = None,
        bin_len: int = DEFAULT_BIN_LEN,
        edges: Sequence[float] = DEFAULT_EDGES,
        trend_s: float = DEFAULT_TREND_S,
        critical_s: Optional[float] = None,
        use_exclusions: bool = True,
        write_gaps: bool = False,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run left-turn gap analysis and optionally write results to disk.

        Args:
            start: Inclusive start date or datetime (``'YYYY-MM-DD'`` or
                ``'YYYY-MM-DD HH:MM'``).
            end: Period end. Date-only extends to end-of-day; datetime is exclusive.
            lefts: Optional subset of left-turn labels to analyse (e.g. ``['EBL', 'WBL']``).
            bin_len: Time aggregation interval in minutes (must divide 60). Default 15.
            edges: Gap bin edges in seconds. Default (1.0, 3.3, 3.7, 7.4, inf).
            trend_s: Gaps of at least this many seconds count as turnable. Default 7.4.
            critical_s: Optional critical gap threshold override in seconds.
            use_exclusions: Apply ``TM_Exclusions`` from config when True.
            write_gaps: When True, also write individual gap CSV.
            make_plot: When True and ``output_dir`` is provided, generate an
                interactive HTML plot.
            output_dir: Destination directory for CSV and HTML files. When
                provided, files are written and ``None`` is returned.

        Returns:
            Dict containing ``"pairs"``, ``"cycles"``, ``"gaps"``, ``"bins"``,
            and ``"summary"`` DataFrames, or ``None`` when ``output_dir`` is set.

        Raises:
            ValueError: If ``bin_len`` is not a positive divisor of 60 or
                ``edges`` fails validation.
        """
        # 1. Validate before touching the DB
        if not (isinstance(bin_len, int) and bin_len > 0 and 60 % bin_len == 0):
            raise ValueError(f"bin_len must be a positive divisor of 60, got {bin_len!r}")
        check_edges(edges)

        # 2. Parse the window
        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)

        # 3. Config at start_dt
        config = self._get_config(start_dt)
        pairs = left_turn_pairs(config)
        exclusions = parse_exclusions_from_config(config) if use_exclusions else None

        if pairs.empty:
            print(
                "  ⚠️  LeftTurnGap: no TM_{NB|SB|EB|WB}L left-turn movement config found. "
                "Check TM_* rows in int_cfg.csv."
            )
            return {} if output_dir is None else None

        if lefts is not None:
            configured_lefts = set(pairs["left"].dropna().tolist())
            for l in lefts:
                if l not in configured_lefts:
                    print(f"  ⚠️  LeftTurnGap: {l}: no TM_{l} key; skipped.")
            selected_pairs = pairs.loc[pairs["left"].isin(lefts)]
        else:
            selected_pairs = pairs

        thr = through_phases(config).set_index("direction")
        runnable = []
        for pair in selected_pairs.itertuples():
            if pd.isna(pair.opposing_phase):
                cand_s = thr.at[pair.opposing, "candidates"] if pair.opposing in thr.index else ""
                cands_fmt = (
                    f", candidates {'/'.join('P' + c for c in cand_s.split(','))}"
                    if cand_s
                    else ""
                )
                print(
                    f"  ⚠️  LeftTurnGap: {pair.left}: no through phase for {pair.opposing} "
                    f"({pair.source}{cands_fmt}); add Det:,P{{N}} Direction,{pair.opposing} to int_cfg.csv."
                )
            elif not pair.detectors:
                print(f"  ⚠️  LeftTurnGap: {pair.left}: no TM_{pair.opposing}T/R detectors; skipped.")
            else:
                runnable.append(pair)

        for pair in runnable:
            if pair.shared:
                det_str = ", ".join(str(d) for d in pair.shared)
                print(
                    f"  ⚠️  LeftTurnGap: {pair.left}: opposing detector(s) {det_str} "
                    f"also configured under another direction (TM_*); they count as opposing traffic."
                )
            crit = critical_s if critical_s is not None else pair.critical_s
            dets_str = ",".join(str(d) for d in pair.detectors)
            print(
                f"  ℹ️  LeftTurnGap: {pair.left} opposed by {pair.opposing} through, "
                f"Ph{int(pair.opposing_phase)} ({pair.source}), detectors {dets_str}, "
                f"critical {crit:g} s"
            )

        if not runnable:
            return {} if output_dir is None else None

        # 4. Events: one fetch with margin
        margin = timedelta(seconds=FETCH_MARGIN_S)
        events = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt - margin,
            end=end_dt + margin,
            event_codes=list(_LTG_CODES),
            timezone=self.timezone,
        )
        if events.empty:
            print("  ⚠️  LeftTurnGap: no events found for the requested window.")
            return {} if output_dir is None else None

        # 5. Silent detectors
        start_ts = pd.Timestamp(start_dt, tz=self.timezone)
        end_ts = pd.Timestamp(end_dt, tz=self.timezone)
        ev_window = events[(events["timestamp"] >= start_ts) & (events["timestamp"] < end_ts)]
        active_dets = set(
            ev_window.loc[ev_window["event_code"] == 81, "parameter"].dropna().astype(int)
        )

        for pair in runnable:
            silent = sorted([d for d in pair.detectors if d not in active_dets])
            if silent:
                det_str = ", ".join(str(d) for d in silent)
                print(
                    f"  ⚠️  LeftTurnGap: {pair.opposing} detector(s) {det_str} "
                    f"logged no actuation in the window."
                )

        # 6. Core per runnable pair
        cycle_frames: List[pd.DataFrame] = []
        gap_frames: List[pd.DataFrame] = []

        for pair in runnable:
            crit = critical_s if critical_s is not None else pair.critical_s
            cy, gp = left_turn_gaps(
                events_df=events,
                opposing_phase=int(pair.opposing_phase),
                detector_ids=pair.detectors,
                left=pair.left,
                edges=edges,
                trend_s=trend_s,
                critical_s=crit,
                exclusions=exclusions,
            )
            cy_keep = (cy["green_ts"] >= start_ts) & (cy["green_ts"] < end_ts)
            cy_filt = cy.loc[cy_keep].reset_index(drop=True)

            gp_keep = (gp["green_ts"] >= start_ts) & (gp["green_ts"] < end_ts)
            gp_filt = gp.loc[gp_keep].reset_index(drop=True)

            if not cy_filt.empty:
                cycle_frames.append(cy_filt)
            if not gp_filt.empty:
                gap_frames.append(gp_filt)

        cycles = (
            pd.concat(cycle_frames, ignore_index=True)
            if cycle_frames
            else pd.DataFrame(columns=cycle_schema(edges))
        )
        gaps = (
            pd.concat(gap_frames, ignore_index=True)
            if gap_frames
            else pd.DataFrame(columns=GAP_SCHEMA)
        )

        # 7. Summaries
        if cycles.empty:
            return {} if output_dir is None else None

        bins = summarize_left_turn_gaps(cycles, bin_len, edges)
        summary = summarize_left_turn_gaps(cycles, None, edges)

        # 8. Outputs
        if output_dir is not None:
            self._write_outputs(
                pairs=pairs,
                cycles=cycles,
                bins=bins,
                summary=summary,
                gaps=gaps,
                output_dir=output_dir,
                start_dt=start_dt,
                end_dt=end_dt,
                bin_len=bin_len,
                edges=edges,
                trend_s=trend_s,
                write_gaps=write_gaps,
                make_plot=make_plot,
            )
            return None

        return {
            "pairs": pairs,
            "cycles": cycles,
            "gaps": gaps,
            "bins": bins,
            "summary": summary,
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
        pairs: pd.DataFrame,
        cycles: pd.DataFrame,
        bins: pd.DataFrame,
        summary: pd.DataFrame,
        gaps: pd.DataFrame,
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
        bin_len: int,
        edges: Sequence[float],
        trend_s: float,
        write_gaps: bool,
        make_plot: bool,
    ) -> None:
        """Write left turn gap DataFrames and plot to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._format_stamp(start_dt, end_dt)

        pairs_file = output_dir / f"LTG_Pairs_{stamp}.csv"
        pairs_export = pairs.copy()
        pairs_export["detectors"] = pairs_export["detectors"].apply(
            lambda lst: ",".join(str(x) for x in lst)
            if isinstance(lst, (list, tuple, set))
            else ("" if pd.isna(lst) else str(lst))
        )
        pairs_export["shared"] = pairs_export["shared"].apply(
            lambda lst: ",".join(str(x) for x in lst)
            if isinstance(lst, (list, tuple, set))
            else ("" if pd.isna(lst) else str(lst))
        )
        pairs_export.to_csv(pairs_file, index=False)
        print(f"Wrote {pairs_file.name}")

        cycles_file = output_dir / f"LTG_Cycles_{stamp}.csv"
        cycles.to_csv(cycles_file, index=False)
        print(f"Wrote {cycles_file.name}")

        bins_file = output_dir / f"LTG_Bins_{bin_len}min_{stamp}.csv"
        bins.to_csv(bins_file, index=False)
        print(f"Wrote {bins_file.name}")

        summary_file = output_dir / f"LTG_Summary_{stamp}.csv"
        summary.to_csv(summary_file, index=False)
        print(f"Wrote {summary_file.name}")

        if write_gaps:
            gaps_file = output_dir / f"LTG_Gaps_{stamp}.csv"
            gaps.to_csv(gaps_file, index=False)
            print(f"Wrote {gaps_file.name}")

        if make_plot:
            with DatabaseManager(self.db_path) as m:
                metadata = m.get_metadata() or {}
            fig = plot_left_turn_gap(bins, metadata=metadata, edges=edges, trend_s=trend_s)
            plot_file = output_dir / f"LTG_Chart_{stamp}.html"
            fig.write_html(str(plot_file))
            print(f"Wrote {plot_file.name}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_left_turn_gap(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    lefts: Optional[List[str]] = None,
    bin_len: int = DEFAULT_BIN_LEN,
    edges: Sequence[float] = DEFAULT_EDGES,
    trend_s: float = DEFAULT_TREND_S,
    critical_s: Optional[float] = None,
    use_exclusions: bool = True,
    write_gaps: bool = False,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`LeftTurnGapEngine`.left_turn_gap."""
    return LeftTurnGapEngine(db_path, timezone).left_turn_gap(
        start=start,
        end=end,
        lefts=lefts,
        bin_len=bin_len,
        edges=edges,
        trend_s=trend_s,
        critical_s=critical_s,
        use_exclusions=use_exclusions,
        write_gaps=write_gaps,
        make_plot=make_plot,
        output_dir=output_dir,
    )
