"""
ATSPM Throughput Optimizer Engine (Imperative Shell)

Orchestrates cycle length and split optimization for saturated throughput by
querying the SQLite database, resolving configuration, measuring flow-rate
discharge profiles, extracting movement demand, and delegating the optimization
solve and plotting to the Functional Core.

Package Location: src/atspm/data/optimizer.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .counts import CountEngine
from .flow import _ALL_FLOW_CODES
from .manager import DatabaseManager, db_timezone
from .reader import _query_cycles, get_events_with_cycles_df
from ..analysis.critical import (
    _parse_stopbar_sets,
    movement_phase_map,
    phase_demand,
    ring_barrier_structure,
)
from ..analysis.flow import discharge_profiles, flow_rate, saturation_state
from ..analysis.optimizer import optimize
from ..plotting.optimizer import (
    plot_allocation,
    plot_marginal_rates,
    plot_throughput_curve,
)
from ..utils.timezone import to_epoch

# Accepted --start/--end string formats (date-only end extends to end-of-day)
_DATETIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M")
_DATE_FORMAT = "%Y-%m-%d"


def _to_json_compatible(obj: Any) -> Any:
    """Recursively convert numpy types and dict keys to standard Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj


class OptimizerEngine:
    """Queries the database and performs cycle length / split throughput optimization.

    All date/time arguments are interpreted in the intersection's local timezone
    (read from the ``metadata`` table).
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """
        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g., ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def optimize(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        saturated: List[int],
        plans: Optional[List[int]] = None,
        pct: float = 1.0,
        split_tolerance: float = 0.10,
        stratify: bool = False,
        max_lost: float = 10.0,
        sat_threshold: float = 0.8,
        demand_stat: str = "mean",
        default_min_split: float = 10.0,
        c_min: float = 60.0,
        c_max: float = 220.0,
        c_step: float = 1.0,
        flat_tol_pct: float = 1.0,
        boundary_rate_tol: float = 100.0,
        bin_len: int = 15,
        exclude_missing: bool = True,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, object]]:
        """Run throughput cycle-length and split optimization.

        Args:
            start: Period start (string or datetime).
            end: Period end (string or datetime).
            saturated: Declared saturated phase numbers.
            plans: Optional list of coordination plan IDs to filter cycles.
            pct: Percentage of the busiest modal-split cycles to keep
                (``1.0`` = top 1%, ``100`` = all).
            split_tolerance: Tolerance around target percentile green duration.
            stratify: Whether to stratify percentile selection by plan.
            max_lost: Maximum lost time for saturation qualification.
            sat_threshold: Threshold share of qualifying cycles for advisory.
            demand_stat: Column to extract from phase demand (``'mean'`` or ``'peak'``).
            default_min_split: Fallback minimum split in seconds.
            c_min: Shortest cycle scanned in seconds.
            c_max: Longest cycle scanned in seconds.
            c_step: Cycle scan step in seconds.
            flat_tol_pct: Flat-band tolerance percent of peak throughput.
            boundary_rate_tol: Tail rate tolerance for boundary classification.
            bin_len: Minutes per demand aggregation bin.
            exclude_missing: Exclude missing bins in CountEngine.
            make_plot: Whether to generate Plotly figures.
            output_dir: If set, write CSV and HTML outputs to this directory and return None.

        Returns:
            Dictionary containing optimization results, DataFrames, figures,
            or None if output_dir is specified, or empty dict if execution fails.

        Raises:
            ValueError: If saturated is empty or demand_stat is invalid.
        """
        # 1. Validate arguments
        if not saturated:
            raise ValueError("saturated phase list cannot be empty.")
        if demand_stat not in ("mean", "peak"):
            raise ValueError(
                f"demand_stat must be 'mean' or 'peak', got {demand_stat!r}."
            )

        # 2. Parse range and get config at start
        start_dt, end_dt = self._parse_range(start, end)
        config = self._get_config(start_dt)
        if not config:
            print("  ⚠️  Optimizer: no configuration found — import int_cfg.csv first.")
            return None if output_dir is not None else {}

        # 3. Structure
        start_epoch = to_epoch(start_dt, self.timezone)
        end_epoch = to_epoch(end_dt, self.timezone)
        cycles_df = _query_cycles(self.db_path, start_epoch, end_epoch)
        if cycles_df.empty:
            print("  ⚠️  Optimizer: no cycles found in the requested window.")
            return None if output_dir is not None else {}

        structure_df = ring_barrier_structure(config, cycles_df)

        # 4. Declaration check
        st_phases = (
            set(structure_df["phase"].astype(int))
            if not structure_df.empty
            else set()
        )
        for p in saturated:
            if p not in st_phases:
                print(f"  ⚠️  Declared Ph{p} is not in structure, ignoring.")
        saturated_map = {int(p): True for p in saturated}

        # 5. Events
        events = get_events_with_cycles_df(
            self.db_path,
            start_dt,
            end_dt,
            event_codes=_ALL_FLOW_CODES,
            timezone=self.timezone,
        )

        # 6. Detectors
        all_stopbar_sets = _parse_stopbar_sets(config)
        stopbar_dets = {
            p: sorted(all_stopbar_sets[p])
            for p in sorted(st_phases)
            if p in all_stopbar_sets and all_stopbar_sets[p]
        }

        # 7. Per phase with detectors
        cycle_dfs = []
        curves: Dict[int, pd.DataFrame] = {}
        for ph, dets in stopbar_dets.items():
            cycle_df, vehicle_df = flow_rate(
                events, ph, dets, max_lost=None, plans=plans
            )
            if not cycle_df.empty:
                cycle_dfs.append(cycle_df)
            _, profile = discharge_profiles(
                cycle_df,
                vehicle_df,
                pct=pct,
                split_tolerance=split_tolerance,
                stratify=stratify,
            )
            if not profile.empty:
                curves[ph] = profile

        for p in saturated:
            if p not in curves:
                print(
                    f"  ⚠️  No discharge curve for Ph{p} — raise --pct or widen the window."
                )

        # 8. Advisory
        if cycle_dfs:
            combined_cycles = pd.concat(cycle_dfs, ignore_index=True)
            advisory_df = saturation_state(
                combined_cycles, max_lost=max_lost, threshold=sat_threshold
            )
        else:
            advisory_df = pd.DataFrame(
                columns=[
                    "phase",
                    "n_cycles",
                    "n_obs",
                    "capped_rate",
                    "pass_rate",
                    "min_lane_pass_rate",
                    "saturated",
                ]
            )

        adv_map = dict(zip(advisory_df["phase"].astype(int), advisory_df["saturated"]))
        for p in saturated:
            if p in adv_map and not adv_map[p]:
                print(
                    f"  ℹ️  Advisory: Ph{p} is declared saturated but advisory-unsaturated."
                )
        for p, is_sat in adv_map.items():
            if is_sat and p not in saturated:
                print(
                    f"  ℹ️  Advisory: Ph{p} is advisory-saturated but not declared saturated."
                )

        # 9. Demand
        counts_df = CountEngine(self.db_path, self.timezone).vehicle_counts(
            start_dt,
            end_dt,
            bin_len=bin_len,
            hourly=True,
            exclude_missing=exclude_missing,
        )
        if counts_df is None or counts_df.empty:
            print("  ⚠️  Optimizer: no movement counts found for the requested window.")
            demand_df = pd.DataFrame()
            demand_vph: Dict[int, float] = {}
        else:
            mv_map = movement_phase_map(config)
            demand_df = phase_demand(counts_df, mv_map)
            dem_col = "peak_vph" if demand_stat == "peak" else "demand_vph"
            if demand_df.empty or dem_col not in demand_df.columns:
                demand_vph = {}
            else:
                valid_dem = demand_df.loc[
                    demand_df["phase"].notna()
                    & demand_df[dem_col].notna()
                    & np.isfinite(demand_df[dem_col])
                ]
                demand_vph = dict(
                    zip(
                        valid_dem["phase"].astype(int),
                        valid_dem[dem_col].astype(float),
                    )
                )

        # 10. Minimum splits
        min_splits: Dict[int, float] = {}
        bg_phases = (
            structure_df.loc[structure_df["barrier_group"].notna(), "phase"]
            .astype(int)
            .tolist()
        )
        for p in sorted(bg_phases):
            key = f"Min_P{p}_Split"
            val = config.get(key)
            parsed = None
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                try:
                    parsed = float(str(val).strip())
                except (ValueError, TypeError):
                    parsed = None
            if parsed is None:
                print(
                    f"  ⚠️  Optimizer: {key} is missing or invalid — using default {default_min_split:.1f}s."
                )
                min_splits[p] = float(default_min_split)
            else:
                min_splits[p] = float(parsed)

        # 11. Solve
        try:
            solver_res = optimize(
                curves=curves,
                structure_df=structure_df,
                saturated=saturated_map,
                demand_vph=demand_vph,
                min_splits=min_splits,
                c_min=c_min,
                c_max=c_max,
                c_step=c_step,
                flat_tol_pct=flat_tol_pct,
                boundary_rate_tol=boundary_rate_tol,
            )
        except ValueError as exc:
            print(f"  ❌  Optimizer solve failed: {exc}")
            return None if output_dir is not None else {}

        # 12. Summary print
        opt = solver_res["optimum"]
        state = opt.get("state")
        c_star = opt.get("c_star")
        flat_lo = opt.get("flat_c_lo")
        flat_hi = opt.get("flat_c_hi")
        warnings = solver_res.get("warnings", [])
        directive = solver_res.get("directive")

        print(f"\nOptimizer Summary:")
        print(f"  State:    {state}")
        if pd.notna(c_star) and np.isfinite(c_star):
            print(f"  C*:       {c_star:.1f} s")
        else:
            print(f"  C*:       NaN")
        if (
            flat_lo is not None
            and flat_hi is not None
            and pd.notna(flat_lo)
            and pd.notna(flat_hi)
            and np.isfinite(flat_lo)
            and np.isfinite(flat_hi)
        ):
            print(f"  Flat band: [{flat_lo:.1f}, {flat_hi:.1f}] s")
        if directive:
            print(f"  Directive: {directive}")
        for w in warnings:
            print(f"  ⚠️  {w}")

        # 13. Result dict
        figures: Dict[str, Any] = {}
        if make_plot:
            with DatabaseManager(self.db_path) as m:
                metadata = m.get_metadata() or {}
            figures["curve"] = plot_throughput_curve(
                solver_res["scan"], opt, metadata
            )
            figures["allocation"] = plot_allocation(
                solver_res["splits"], opt, metadata
            )
            figures["marginal"] = plot_marginal_rates(
                curves, solver_res["splits"], metadata
            )

        results: Dict[str, object] = {
            "scan": solver_res["scan"],
            "splits": solver_res["splits"],
            "optimum": solver_res["optimum"],
            "directive": solver_res["directive"],
            "warnings": solver_res["warnings"],
            "curves": curves,
            "saturation": advisory_df,
            "demand": demand_df,
            "min_splits": min_splits,
            "figures": figures,
        }

        # 14. Output dir
        if output_dir is not None:
            self._write_outputs(
                results,
                output_dir,
                start_dt,
                end_dt,
                make_plot=make_plot,
            )
            return None

        return results

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
    def _parse_range(
        start: Union[str, datetime],
        end: Union[str, datetime],
    ) -> tuple[datetime, datetime]:
        """Coerce date or datetime strings to naive local datetimes.

        A date-only *end* is extended by one day (whole-day convention);
        a datetime *end* is used as-is (exclusive), enabling sub-day peak
        periods.
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
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
        make_plot: bool = True,
    ) -> None:
        """Write optimization CSV and HTML outputs to disk.

        Sub-day windows include the time component (``HHMM``) in the
        window stamps so peak-period runs on the same day don't collide.
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

        # 1. Optimize_Scan_{stamp}.csv
        scan_df = results.get("scan")
        if scan_df is not None and not scan_df.empty:
            scan_file = f"Optimize_Scan_{date_str}.csv"
            scan_df.to_csv(output_dir / scan_file, index=False)
            print(f"Wrote {scan_file}")

        # 2. Optimize_Splits_{stamp}.csv
        splits_df = results.get("splits")
        if splits_df is not None and not splits_df.empty:
            splits_file = f"Optimize_Splits_{date_str}.csv"
            splits_df.to_csv(output_dir / splits_file, index=False)
            print(f"Wrote {splits_file}")

        # 3. Optimize_Saturation_{stamp}.csv
        sat_df = results.get("saturation")
        if sat_df is not None:
            sat_file = f"Optimize_Saturation_{date_str}.csv"
            sat_df.to_csv(output_dir / sat_file, index=False)
            print(f"Wrote {sat_file}")

        # 4. Optimize_Summary_{stamp}.csv
        optimum = results.get("optimum") or {}
        summary_row = {}
        for k, v in optimum.items():
            if isinstance(v, dict):
                summary_row[k] = json.dumps(_to_json_compatible(v))
            else:
                summary_row[k] = _to_json_compatible(v)
        directive = results.get("directive")
        summary_row["directive"] = (
            json.dumps(_to_json_compatible(directive))
            if directive is not None
            else ""
        )
        summary_file = f"Optimize_Summary_{date_str}.csv"
        pd.DataFrame([summary_row]).to_csv(output_dir / summary_file, index=False)
        print(f"Wrote {summary_file}")

        # 5. HTML plots
        if make_plot:
            figs = results.get("figures", {})
            plot_files = [
                ("Optimize_Curve", "curve"),
                ("Optimize_Allocation", "allocation"),
                ("Optimize_Marginal", "marginal"),
            ]
            for prefix, key in plot_files:
                fig = figs.get(key)
                if fig is not None:
                    html_file = f"{prefix}_{date_str}.html"
                    fig.write_html(str(output_dir / html_file))
                    print(f"Wrote {html_file}")


def get_optimization(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    saturated: List[int],
    plans: Optional[List[int]] = None,
    pct: float = 1.0,
    split_tolerance: float = 0.10,
    stratify: bool = False,
    max_lost: float = 10.0,
    sat_threshold: float = 0.8,
    demand_stat: str = "mean",
    default_min_split: float = 10.0,
    c_min: float = 60.0,
    c_max: float = 220.0,
    c_step: float = 1.0,
    flat_tol_pct: float = 1.0,
    boundary_rate_tol: float = 100.0,
    bin_len: int = 15,
    exclude_missing: bool = True,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """Convenience wrapper around :class:`OptimizerEngine`.optimize."""
    return OptimizerEngine(db_path, timezone).optimize(
        start=start,
        end=end,
        saturated=saturated,
        plans=plans,
        pct=pct,
        split_tolerance=split_tolerance,
        stratify=stratify,
        max_lost=max_lost,
        sat_threshold=sat_threshold,
        demand_stat=demand_stat,
        default_min_split=default_min_split,
        c_min=c_min,
        c_max=c_max,
        c_step=c_step,
        flat_tol_pct=flat_tol_pct,
        boundary_rate_tol=boundary_rate_tol,
        bin_len=bin_len,
        exclude_missing=exclude_missing,
        make_plot=make_plot,
        output_dir=output_dir,
    )
