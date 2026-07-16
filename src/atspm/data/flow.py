"""
ATSPM Split Flow Rate Engine (Imperative Shell)

Orchestrates split flow-rate analysis by querying the SQLite database,
resolving per-phase stop-bar detector configuration, and delegating all
calculations to the Functional Core (``atspm.analysis.flow``) and all
figure construction to ``atspm.plotting.flow``.

Package Location: src/atspm/data/flow.py

Configuration
-------------
Stop-bar detector IDs are read from the active ``config`` row via
``DatabaseManager.get_config_at_date``.  The expected key format is::

    Det_P{phase}_Stopbar   →   "33,34"  (comma-separated detector IDs)

If no key is found for a requested phase, that phase is skipped with a
printed warning.

Gap Marker Rule
---------------
Gap markers (``event_code == -1``) are always included in the event-code
filter sent to the database, ensuring the Functional Core can enforce state
resets at data discontinuities.  The required codes are::

    Phase state codes : 1, 8, 9, 10, 11, 12
    Detector OFF      : 81
    Gap marker        : -1
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .manager import DatabaseManager
from .reader import get_events_with_cycles_df
from ..analysis.flow import (
    flow_rate as _flow_rate_core,
    rate_profiles as _rate_profiles_core,
)
from ..plotting.flow import plot_flow_profiles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase state codes required by _build_phase_intervals
_PHASE_CODES: List[int] = [1, 8, 9, 10, 11, 12]
# Detector departure code (stop-bar detector off)
_DET_OFF_CODE: int = 81
# Gap marker — always included so the core sees discontinuities
_GAP_CODE: int = -1

_ALL_FLOW_CODES: List[int] = sorted(set(_PHASE_CODES) | {_DET_OFF_CODE, _GAP_CODE})


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class FlowRateEngine:
    """Queries the database and produces split flow-rate tables and plots.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).  The underlying
    ``get_events_with_cycles_df`` call converts to UTC epochs for the SQL
    query and returns timezone-aware timestamps.

    Example::

        engine = FlowRateEngine(Path("2068_data.db"))

        # In-memory results for phase 6, coordination plan 1
        results = engine.flow(
            "2025-08-17", "2025-08-17",
            phases=[6],
            plans=[1],
        )

        # CSVs + interactive HTML plots written to disk
        engine.flow(
            "2025-08-17", "2025-08-17",
            phases=[4, 8],
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

    def flow(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[List[int]] = None,
        plans: Optional[List[int]] = None,
        pct: float = 1.0,
        max_lost: float = 10.0,
        split_tolerance: float = 0.10,
        normalize: str = "end_shift",
        fixed_lost: Optional[float] = None,
        rolling: int = 5,
        make_plot: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, object]]:
        """Run split flow-rate analysis and optionally write results to disk.

        Args:
            start: Inclusive start date (``'YYYY-MM-DD'`` string or naive
                   local ``datetime``).
            end:   Inclusive end date (same format).  The query window
                   extends to end-of-day, matching the ``counts`` /
                   ``splits`` convention.
            phases: Phase numbers to analyse.  When ``None``, all phases
                    with a configured ``Det_P{N}_Stopbar`` key in the active
                    config row are analysed.
            plans: Optional coordination-plan filter; only split windows on
                   these plans are analysed.
            pct: Percentage of the busiest modal-split cycles to keep
                 (``1.0`` = top 1%).  Default ``1.0``.
            max_lost: Maximum end-slack in seconds for a cycle to qualify
                as near-capacity.  Default ``10.0``.
            split_tolerance: Fractional tolerance around the modal split.
                Default ``0.10``.
            normalize: Split-termination overhead mode — one of
                ``'end_shift'`` (default), ``'pooled'``, ``'clearance'``,
                ``'fixed'``, ``'none'``.  See
                ``atspm.analysis.flow.rate_profiles``.
            fixed_lost: Constant overhead in seconds; required when
                ``normalize == 'fixed'``.
            rolling: Centred rolling-mean window for the instantaneous
                traces in the plot.  Default ``5``.
            make_plot: Build a Plotly figure per phase.  Default ``True``.
            output_dir: When provided, write CSV/HTML files to this
                directory and return ``None``.  When ``None``, return the
                result dict without writing.

        Returns:
            ``dict`` with keys:

            * ``"cycle"``    → per-cycle raw summary (all phases).
            * ``"selected"`` → normalised selected-cycle summary
              (all phases, with ``t_max`` / ``max_rate`` / ``n_at_max``).
            * ``"profiles"`` → ``{phase: {"rate": df, "inst": df}}``.
            * ``"figures"``  → ``{phase: go.Figure}``
              (only when *make_plot* is ``True``).

            Returns ``None`` when *output_dir* is set, and an empty dict
            when nothing could be computed.
        """
        start_dt, end_dt = self._parse_range(start, end)
        config = self._get_config(start_dt)

        phase_det_map = self._resolve_detector_map(config, phases)

        if not phase_det_map:
            _phases_requested = phases or "all"
            print(
                f"  ⚠️  Flow: no stop-bar detector config found for "
                f"phases={_phases_requested}.  "
                f"Check Det_P{{N}}_Stopbar keys in int_cfg.csv."
            )
            return {} if output_dir is None else None

        events_df = self._load_events(start_dt, end_dt)

        if events_df.empty:
            print("  ⚠️  Flow: no events found for the requested window.")
            return {} if output_dir is None else None

        metadata = self._get_metadata()

        cycle_frames: List[pd.DataFrame] = []
        selected_frames: List[pd.DataFrame] = []
        profiles: Dict[int, Dict[str, pd.DataFrame]] = {}
        figures: Dict[int, object] = {}

        for ph, det_ids in sorted(phase_det_map.items()):
            cycle_df, vehicle_df = _flow_rate_core(
                events_df=events_df,
                phase=ph,
                detector_ids=det_ids,
                max_lost=max_lost,
                plans=plans,
            )
            if cycle_df.empty:
                print(
                    f"  ⚠️  Flow Ph{ph}: no qualifying split windows or "
                    f"detector departures found — skipping."
                )
                continue

            selected_df, rate_df, inst_df = _rate_profiles_core(
                cycle_df=cycle_df,
                vehicle_df=vehicle_df,
                pct=pct,
                split_tolerance=split_tolerance,
                normalize=normalize,
                fixed_lost=fixed_lost,
            )
            if selected_df.empty:
                print(
                    f"  ⚠️  Flow Ph{ph}: no cycles survived selection "
                    f"(split_tolerance={split_tolerance}, pct={pct}) — skipping."
                )
                continue

            cycle_frames.append(cycle_df)
            selected_frames.append(selected_df)
            profiles[ph] = {"rate": rate_df, "inst": inst_df}

            if make_plot:
                figures[ph] = plot_flow_profiles(
                    rate_df=rate_df,
                    inst_df=inst_df,
                    metadata=metadata,
                    phase=ph,
                    rolling=rolling,
                )

        if not cycle_frames:
            return {} if output_dir is None else None

        results: Dict[str, object] = {
            "cycle": pd.concat(cycle_frames, ignore_index=True),
            "selected": pd.concat(selected_frames, ignore_index=True),
            "profiles": profiles,
        }
        if make_plot:
            results["figures"] = figures

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

    def _get_metadata(self) -> dict:
        """Read the metadata dict used for plot titles (may be empty)."""
        try:
            with DatabaseManager(self.db_path) as m:
                return m.get_metadata() or {}
        except Exception:
            return {}

    def _parse_range(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
    ) -> tuple[datetime, datetime]:
        """Coerce ``'YYYY-MM-DD'`` strings to naive midnight datetimes.

        The *end* date is extended by one day so the query window covers the
        full last calendar day, matching the ``counts`` / ``splits`` convention.

        Args:
            start: Start date string or naive datetime.
            end:   End date string or naive datetime.

        Returns:
            Tuple of ``(start_dt, end_dt)`` as naive datetimes.
        """
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        return start, end

    def _load_events(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch all flow-relevant events from the database.

        Always includes gap markers so the Functional Core can enforce
        state resets at data discontinuities.

        Args:
            start: Naive local start datetime.
            end:   Naive local end datetime.

        Returns:
            DataFrame with tz-aware ``timestamp`` / ``cycle_start`` columns.
        """
        return get_events_with_cycles_df(
            db_path=self.db_path,
            start=start,
            end=end,
            event_codes=_ALL_FLOW_CODES,
            timezone=self.timezone,
        )

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

    def _resolve_detector_map(
        self,
        config: dict,
        phases: Optional[List[int]],
    ) -> Dict[int, List[int]]:
        """Build a mapping of phase → list[detector_id] from the config dict.

        Reads ``Det_P{N}_Stopbar`` keys.  Phases with an empty or absent key
        are silently skipped.  If *phases* is provided, only those phases are
        resolved.

        Args:
            config: Active config dict from ``DatabaseManager.get_config_at_date``.
            phases: Explicit phase list, or ``None`` to discover from config.

        Returns:
            ``{phase_int: [det_id, ...]}`` for all phases with valid detector
            config.  Empty dict if nothing is configured.
        """
        result: Dict[int, List[int]] = {}

        stopbar_keys = {
            k: v for k, v in config.items()
            if k.startswith("Det_P") and k.endswith("_Stopbar") and v
        }

        for key, raw_val in stopbar_keys.items():
            # Key format: Det_P{N}_Stopbar  →  extract N
            try:
                ph_str = key[5:key.index("_Stopbar")]
                ph = int(ph_str)
            except (ValueError, IndexError):
                continue

            if phases is not None and ph not in phases:
                continue

            try:
                det_ids = [int(x.strip()) for x in str(raw_val).split(",") if x.strip()]
            except ValueError:
                continue

            if det_ids:
                result[ph] = det_ids

        if phases is not None:
            for ph in phases:
                if ph not in result:
                    print(
                        f"  ⚠️  Flow: no Det_P{ph}_Stopbar config found — "
                        f"phase {ph} skipped."
                    )

        return result

    def _write_outputs(
        self,
        results: Dict[str, object],
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        """Write result DataFrames to CSV and figures to HTML.

        Filename patterns::

            Flow_Cycle_{start}-{end}.csv
            Flow_Selected_{start}-{end}.csv
            Flow_Rate_Ph{N}_{start}-{end}.csv
            Flow_Inst_Ph{N}_{start}-{end}.csv
            Flow_Ph{N}_{start}-{end}.html

        Args:
            results:    Dict returned by the internal calculation step.
            output_dir: Destination directory (created if absent).
            start_dt:   Parsed query start (naive local datetime).
            end_dt:     Parsed query end (exclusive; one day past the
                        inclusive end date).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        last_day = end_dt - timedelta(days=1)
        date_str = f"{start_dt:%Y_%m_%d}-{last_day:%Y_%m_%d}"

        cycle_df = results.get("cycle")
        if cycle_df is not None and not cycle_df.empty:
            filename = f"Flow_Cycle_{date_str}.csv"
            cycle_df.to_csv(output_dir / filename, index=False)
            print(f"Wrote {filename}")

        selected_df = results.get("selected")
        if selected_df is not None and not selected_df.empty:
            filename = f"Flow_Selected_{date_str}.csv"
            selected_df.to_csv(output_dir / filename, index=False)
            print(f"Wrote {filename}")

        for ph, prof in results.get("profiles", {}).items():
            for kind, df in (("Rate", prof["rate"]), ("Inst", prof["inst"])):
                if df is None or df.empty:
                    continue
                filename = f"Flow_{kind}_Ph{ph}_{date_str}.csv"
                df.to_csv(output_dir / filename)
                print(f"Wrote {filename}")

        for ph, fig in results.get("figures", {}).items():
            filename = f"Flow_Ph{ph}_{date_str}.html"
            fig.write_html(output_dir / filename)
            print(f"Wrote {filename}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_flow_rate(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[List[int]] = None,
    plans: Optional[List[int]] = None,
    pct: float = 1.0,
    max_lost: float = 10.0,
    split_tolerance: float = 0.10,
    normalize: str = "end_shift",
    fixed_lost: Optional[float] = None,
    rolling: int = 5,
    make_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """Convenience wrapper around :class:`FlowRateEngine`.flow.

    Args:
        db_path:         Path to the intersection SQLite database.
        start:           Inclusive start date/datetime (local time).
        end:             Inclusive end date/datetime (local time).
        phases:          Phase IDs to analyse; ``None`` for all configured.
        plans:           Optional coordination-plan filter.
        pct:             Busiest percentage of modal-split cycles to keep.
        max_lost:        Maximum end-slack seconds to qualify a cycle.
        split_tolerance: Fractional tolerance around the modal split.
        normalize:       Overhead mode (see ``rate_profiles``).
        fixed_lost:      Constant overhead seconds for ``'fixed'`` mode.
        rolling:         Rolling window for instantaneous plot traces.
        make_plot:       Build Plotly figures.  Default ``True``.
        output_dir:      Write CSV/HTML files and return ``None`` when provided.
        timezone:        Override intersection timezone.

    Returns:
        Result dict or ``None`` if *output_dir* is set.
    """
    return FlowRateEngine(db_path, timezone).flow(
        start=start,
        end=end,
        phases=phases,
        plans=plans,
        pct=pct,
        max_lost=max_lost,
        split_tolerance=split_tolerance,
        normalize=normalize,
        fixed_lost=fixed_lost,
        rolling=rolling,
        make_plot=make_plot,
        output_dir=output_dir,
    )
