"""
ATSPM Critical Movement Engine (Imperative Shell)

Orchestrates critical movement analysis by querying the SQLite database,
resolving configuration, and delegating all calculations to the Functional
Core (``atspm.analysis.critical``).

Package Location: src/atspm/data/critical.py

Configuration
-------------
Three config families feed the analysis:

    RB_R1 / RB_R2              → ring/barrier structure ('1,2|3,4')
    TM_{movement}              → movement detector lists (demand)
    Det_P{N}_Stopbar or
    Det_P{N}_Stop_Bar          → per-phase stop-bar detectors, used to map
                                 movements to phases by detector overlap

Missing RB_* config falls back to the NEMA-standard dual-ring structure
with a printed warning; movements whose detectors overlap no phase's
stop-bar set (or overlap ambiguously) are reported as unmapped.

Data quality
------------
Demand is the mean of hourly-scaled bin rates, so bins with missing data
would bias it toward zero.  Only bins labeled ``"ok"`` by the counts
quality pass are used by default (``exclude_missing=True``); pass
``exclude_missing=False`` to keep partial/missing bins.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from .counts import CountEngine
from .manager import DatabaseManager
from .reader import _query_cycles
from ..analysis.critical import (
    critical_movement_analysis as _critical_core,
    movement_phase_map as _movement_map_core,
    phase_demand as _phase_demand_core,
    ring_barrier_structure as _structure_core,
)

# Accepted --start/--end string formats (date-only end extends to end-of-day)
_DATETIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M")
_DATE_FORMAT = "%Y-%m-%d"


class CriticalMovementEngine:
    """Queries the database and produces critical movement analysis tables.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = CriticalMovementEngine(Path("2068_data.db"))

        # AM peak, in-memory results
        results = engine.critical("2025-08-17 07:00", "2025-08-17 09:00")

        # Whole-day analysis written to disk
        engine.critical(
            "2025-08-17", "2025-08-17",
            output_dir=Path("./outputs"),
        )
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """
        Args:
            db_path:  Path to the intersection SQLite database.
            timezone: Local timezone string (e.g., ``'US/Mountain'``).
                      Defaults to the value stored in the ``metadata`` table,
                      with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def critical(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        bin_len: int = 15,
        basis: str = "per_lane",
        exclude_missing: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run critical movement analysis for a chosen period.

        Args:
            start: Period start — ``'YYYY-MM-DD'``, ``'YYYY-MM-DD HH:MM'``,
                or a naive local ``datetime``.
            end: Period end (same formats).  A date-only *end* is extended
                to end-of-day, matching the ``counts`` convention; a
                datetime *end* is exclusive as given, so peak periods
                (e.g. 07:00–09:00) can be analysed directly.
            bin_len: Demand-aggregation bin width in minutes.  Default 15.
            basis: ``'per_lane'`` (default) or ``'total'`` — see
                ``atspm.analysis.critical.critical_movement_analysis``.
            exclude_missing: Keep only bins whose counts data quality is
                ``"ok"``.  Default ``True`` — zero-filled missing bins
                would bias mean demand downward.
            output_dir: When provided, write CSVs to this directory and
                return ``None``.  When ``None``, return the result dict.

        Returns:
            ``dict`` with keys ``"phases"``, ``"groups"``, ``"movements"``
            (movement→phase map), ``"demand"``, and ``"structure"`` — or
            ``None`` when *output_dir* is set, and an empty dict when
            nothing could be computed.
        """
        start_dt, end_dt = self._parse_range(start, end)
        config = self._get_config(start_dt)

        if not config:
            print("  ⚠️  Critical: no configuration found — "
                  "import int_cfg.csv first.")
            return {} if output_dir is None else None

        movement_map = _movement_map_core(config)
        self._warn_unmapped(movement_map)

        if movement_map.empty:
            print("  ⚠️  Critical: no TM_* movement config found.")
            return {} if output_dir is None else None

        counts_df = CountEngine(self.db_path, self.timezone).vehicle_counts(
            start_dt, end_dt,
            bin_len=bin_len,
            hourly=True,
            exclude_missing=exclude_missing,
        )
        if counts_df is None or counts_df.empty:
            print("  ⚠️  Critical: no movement counts found for the "
                  "requested window.")
            return {} if output_dir is None else None

        cycles_df = _query_cycles(
            self.db_path, start_dt.timestamp(), end_dt.timestamp()
        )
        if cycles_df.empty:
            print("  ⚠️  Critical: no cycles found in the window — "
                  "observed_share will be 0 for all phases.")

        structure_df = _structure_core(config, cycles_df)
        self._warn_structure(structure_df)

        demand_df = _phase_demand_core(counts_df, movement_map)
        if demand_df.empty:
            print("  ⚠️  Critical: no mapped movement produced demand — "
                  "check TM_* and Det_P{N}_Stopbar config.")
            return {} if output_dir is None else None

        phase_df, group_df = _critical_core(structure_df, demand_df, basis)

        self._print_summary(group_df, basis)

        results: Dict[str, pd.DataFrame] = {
            "phases": phase_df,
            "groups": group_df,
            "movements": movement_map,
            "demand": demand_df,
            "structure": structure_df,
        }

        if output_dir is not None:
            self._write_outputs(results, output_dir, start_dt, end_dt)
            return None

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_timezone(self) -> str:
        """Read timezone from metadata table, falling back to US/Mountain."""
        try:
            with DatabaseManager(self.db_path) as m:
                meta = m.get_metadata()
                return meta.get("timezone") or "US/Mountain"
        except Exception:
            return "US/Mountain"

    def _get_config(self, date: datetime) -> dict:
        """Retrieve the active configuration dict for a given date.

        Args:
            date: Reference datetime (naive local) used to select the
                  correct temporal config row.

        Returns:
            Config dict (may be empty if no config has been imported).
        """
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

        Args:
            start: Start date/datetime string or naive datetime.
            end:   End date/datetime string or naive datetime.

        Returns:
            Tuple of ``(start_dt, end_dt)`` as naive datetimes.

        Raises:
            ValueError: When a string matches no accepted format.
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
    def _warn_unmapped(movement_map: pd.DataFrame) -> None:
        """Print a warning listing movements without a phase assignment."""
        if movement_map.empty:
            return
        unmapped = movement_map.loc[
            movement_map["phase"].isna(), "movement"
        ].tolist()
        if unmapped:
            print(
                f"  ⚠️  Critical: movements not mapped to a phase "
                f"(no or ambiguous stop-bar detector overlap): "
                f"{', '.join(unmapped)}.  Their demand is excluded."
            )

    @staticmethod
    def _warn_structure(structure_df: pd.DataFrame) -> None:
        """Print warnings for fallback structure and config mismatches."""
        if structure_df.empty:
            return

        if (structure_df["source"] == "default").any():
            print(
                "  ⚠️  Critical: no RB_R1/RB_R2 config found — using the "
                "NEMA-standard dual-ring structure (R1='1,2|3,4', "
                "R2='5,6|7,8')."
            )

        unplaced = structure_df.loc[
            structure_df["barrier_group"].isna(), "phase"
        ].tolist()
        if unplaced:
            print(
                f"  ⚠️  Critical: phases observed in cycles but absent "
                f"from the configured structure: {unplaced}.  They are "
                f"excluded from the critical path."
            )

        never_seen = structure_df.loc[
            structure_df["in_config"]
            & (structure_df["observed_share"] == 0.0),
            "phase",
        ].tolist()
        if never_seen:
            print(
                f"  ℹ️  Critical: configured phases never observed in the "
                f"window: {never_seen}."
            )

    @staticmethod
    def _print_summary(group_df: pd.DataFrame, basis: str) -> None:
        """Print the critical path per barrier group and the critical sum."""
        if group_df.empty:
            return

        unit = "vphpl" if basis == "per_lane" else "vph"
        critical = group_df.loc[group_df["is_critical_path"]]

        for row in critical.itertuples(index=False):
            print(
                f"    Barrier group {int(row.barrier_group)}: critical "
                f"path = Ring {int(row.ring)} (phases {row.phases}, "
                f"{row.demand_sum:.1f} {unit})"
            )
        total = critical["demand_sum"].sum()
        print(f"    Critical sum: {total:.1f} {unit}")

    def _write_outputs(
        self,
        results: Dict[str, pd.DataFrame],
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        """Write result DataFrames to CSV.

        Filename patterns::

            Critical_Phases_{start}-{end}.csv
            Critical_Groups_{start}-{end}.csv
            Critical_Movements_{start}-{end}.csv

        Sub-day windows include the time component (``HHMM``) in the
        window stamps so peak-period runs on the same day don't collide.

        Args:
            results:    Dict returned by the internal calculation step.
            output_dir: Destination directory (created if absent).
            start_dt:   Parsed period start (naive local datetime).
            end_dt:     Parsed period end (exclusive).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def stamp(dt: datetime) -> str:
            if (dt.hour, dt.minute) == (0, 0):
                return f"{dt:%Y_%m_%d}"
            return f"{dt:%Y_%m_%d_%H%M}"

        # Whole-day convention: label with the inclusive last day
        end_label = (
            end_dt - timedelta(days=1)
            if (end_dt.hour, end_dt.minute) == (0, 0)
            else end_dt
        )
        date_str = f"{stamp(start_dt)}-{stamp(end_label)}"

        for kind, key in (("Phases", "phases"), ("Groups", "groups"),
                          ("Movements", "movements")):
            df = results.get(key)
            if df is None or df.empty:
                continue
            filename = f"Critical_{kind}_{date_str}.csv"
            df.to_csv(output_dir / filename, index=False)
            print(f"Wrote {filename}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_critical_movements(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    bin_len: int = 15,
    basis: str = "per_lane",
    exclude_missing: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`CriticalMovementEngine`.critical.

    Args:
        db_path:         Path to the intersection SQLite database.
        start:           Period start (local date or datetime).
        end:             Period end (local date or datetime).
        bin_len:         Demand bin width in minutes.  Default 15.
        basis:           ``'per_lane'`` or ``'total'``.
        exclude_missing: Keep only quality-``"ok"`` bins.  Default ``True``.
        output_dir:      Write CSVs and return ``None`` when provided.
        timezone:        Override intersection timezone.

    Returns:
        Result dict or ``None`` if *output_dir* is set.
    """
    return CriticalMovementEngine(db_path, timezone).critical(
        start=start,
        end=end,
        bin_len=bin_len,
        basis=basis,
        exclude_missing=exclude_missing,
        output_dir=output_dir,
    )
