"""
ATSPM Arrival on Green Engine (Imperative Shell)

Orchestrates Arrival on Green (AOG) analysis by querying the SQLite
database, resolving per-phase detector configuration, and delegating all
calculations to the Functional Core (``atspm.analysis.aog``).

Package Location: src/atspm/data/aog.py

Configuration
-------------
Advance detector IDs are read from the active ``config`` row via
``DatabaseManager.get_config_at_date``.  The expected key format is::

    Det_P{phase}_Arrival   →   "33,34"  (comma-separated detector IDs)

If no key is found for a requested phase, that phase is skipped with a
printed warning.

Gap Marker Rule
---------------
Gap markers (``event_code == -1``) are always included in the event-code
filter sent to the database, ensuring the Functional Core can enforce state
resets at data discontinuities.  The required codes are::

    Phase state codes : 1, 8, 9, 10, 11, 12
    Detector ON       : 82
    Gap marker        : -1

Data Quality
------------
Time-binned results receive the same ``coverage`` / ``data_quality``
annotation used by ``PhaseEngine`` and ``CountEngine``.  Coverage is
computed cheaply from ``ingestion_log`` spans; bins containing gap markers
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

import numpy as np
import pandas as pd
import pytz

from .manager import DatabaseManager
from .reader import get_events_with_cycles_df
from ..analysis.aog import (
    arrival_on_green as _aog_core,
    bin_arrival_on_green as _bin_aog_core,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase state codes required by _build_phase_intervals
_PHASE_CODES: List[int] = [1, 8, 9, 10, 11, 12]
# Detector actuation code
_DET_ON_CODE: int = 82
# Gap marker — always included so the core sees discontinuities
_GAP_CODE: int = -1

_ALL_AOG_CODES: List[int] = sorted(set(_PHASE_CODES) | {_DET_ON_CODE, _GAP_CODE})


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AogEngine:
    """Queries the database and produces Arrival on Green tables.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).  The underlying
    ``get_events_with_cycles_df`` call converts to UTC epochs for the SQL
    query and returns timezone-aware timestamps.

    Example::

        engine = AogEngine(Path("2068_data.db"))

        # Per-cycle table for phases 2 and 6, 3-second advance-detector offset
        df = engine.arrival_on_green(
            "2025-06-01", "2025-06-07",
            phases=[2, 6],
            arrival_offset_sec=3.0,
        )

        # 15-minute bins saved directly to CSV
        engine.arrival_on_green(
            "2025-06-01", "2025-06-07",
            bin_len=15,
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

    def arrival_on_green(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[List[int]] = None,
        arrival_offset_sec: float = 0.0,
        bin_len: Union[int, str] = 60,
        exclude_missing: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Run AOG analysis and optionally write results to CSV files.

        Args:
            start: Inclusive start date (``'YYYY-MM-DD'`` string or naive
                   local ``datetime``).
            end:   Inclusive end date (same format).  The query window
                   extends to end-of-day, matching the ``counts`` /
                   ``splits`` convention.
            phases: Phase numbers to analyse.  When ``None``, all phases
                    with a configured ``Det_P{N}_Arrival`` key in the active
                    config row are analysed.
            arrival_offset_sec: Seconds added to every detector timestamp
                before evaluating green-window containment.  Positive values
                shift arrivals forward in time to account for travel time
                from an advance detector to the stop bar (formerly ``tt``).
                Default ``0.0``.
            bin_len: ``"cycle"`` for one row per detected cycle, or an
                     integer number of minutes for time-bin aggregation.
                     Default ``60``.
            exclude_missing: When ``True``, drop ``"partial"`` and
                ``"missing"`` bins from binned output in addition to the
                always-dropped full-day-missing days.  Ignored in cycle mode.
            output_dir: When provided, write CSV files to this directory and
                        return ``None``.  When ``None``, return the result
                        dict without writing.

        Returns:
            ``dict`` with keys:

            * ``"cycle"`` → per-cycle DataFrame (always present, may be
              empty if no green intervals were found).
            * ``"binned"`` → binned DataFrame (only present when
              ``bin_len != "cycle"``; may be empty).

            Returns ``None`` when *output_dir* is set.

        Raises:
            ValueError: If no phases with configured arrival detectors are
                found and *phases* is ``None``.
        """
        start_dt, end_dt = self._parse_range(start, end)
        config = self._get_config(start_dt)

        # Resolve phase → detector mapping from config
        phase_det_map = self._resolve_detector_map(config, phases)

        if not phase_det_map:
            _phases_requested = phases or "all"
            print(
                f"  ⚠️  AOG: no arrival-detector config found for "
                f"phases={_phases_requested}.  "
                f"Check Det_P{{N}}_Arrival keys in int_cfg.csv."
            )
            return {} if output_dir is None else None

        # Load all required event codes in one query
        events_df = self._load_events(start_dt, end_dt)

        if events_df.empty:
            print("  ⚠️  AOG: no events found for the requested window.")
            return {} if output_dir is None else None

        # --- Per-cycle calculation across all configured phases ---------------
        cycle_frames: List[pd.DataFrame] = []
        for ph, det_ids in sorted(phase_det_map.items()):
            ph_df = _aog_core(
                events_df=events_df,
                phase=ph,
                detector_ids=det_ids,
                arrival_offset_sec=arrival_offset_sec,
            )
            if not ph_df.empty:
                cycle_frames.append(ph_df)
            else:
                print(
                    f"  ⚠️  AOG Ph{ph}: no green intervals or detector "
                    f"events found — skipping."
                )

        if not cycle_frames:
            return {} if output_dir is None else None

        cycle_df = pd.concat(cycle_frames, axis=0)
        # Preserve original index name after concat
        cycle_df.index.name = "cycle_start"

        results: Dict[str, pd.DataFrame] = {"cycle": cycle_df}

        # --- Binned aggregation -----------------------------------------------
        if bin_len != "cycle":
            binned_df = _bin_aog_core(cycle_df, bin_len=int(bin_len))

            if not binned_df.empty and bin_len != "cycle":
                binned_df = self._add_quality(
                    binned_df, events_df, start_dt, end_dt,
                    int(bin_len), exclude_missing,
                )

            results["binned"] = binned_df

        # --- Output -------------------------------------------------------------
        if output_dir is not None:
            self._write_csvs(results, output_dir, bin_len)
            return None

        return results

    # ------------------------------------------------------------------
    # Data quality  (mirrors PhaseEngine._add_quality exactly)
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
        """Annotate a binned AOG DataFrame with coverage and quality labels.

        Joins the quality DataFrame on the ``time`` column (which holds
        bin-start Timestamps, matching the full grid index produced by
        ``_compute_bin_quality``).  Full-day-missing days are always dropped;
        ``"partial"`` and ``"missing"`` rows are additionally removed when
        *exclude_missing* is ``True``.

        Args:
            binned_df:       Output of ``bin_arrival_on_green``, with a
                             ``time`` column of bin-start Timestamps.
            events_df:       Raw events (used for gap marker timestamps only).
            start:           Query start (naive local datetime).
            end:             Query end (naive local datetime).
            bin_len:         Bin width in minutes.
            exclude_missing: Drop partial and missing rows when ``True``.

        Returns:
            Annotated and filtered DataFrame.
        """
        if binned_df.empty:
            return binned_df

        quality = self._compute_bin_quality(events_df, start, end, bin_len)

        # quality is indexed by tz-aware bin-start Timestamp; binned_df has
        # a ``time`` column of the same type.
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

    def _compute_bin_quality(
        self,
        events_df: pd.DataFrame,
        start: datetime,
        end: datetime,
        bin_len: int,
    ) -> pd.DataFrame:
        """Compute coverage fraction and quality label for each bin.

        Mirrors ``PhaseEngine._compute_bin_quality`` exactly.  Coverage is
        derived from ``ingestion_log`` spans (cheap); bins containing gap
        markers are downgraded to at most ``"partial"``.

        Args:
            events_df: Raw events (used for gap marker timestamp extraction).
            start:     Query start (naive local datetime).
            end:       Query end (naive local datetime).
            bin_len:   Bin width in minutes.

        Returns:
            DataFrame indexed by tz-aware bin-start Timestamps with columns
            ``["coverage", "data_quality"]``.
        """
        bin_td = timedelta(minutes=bin_len)

        tz = pytz.timezone(self.timezone)
        grid_start = tz.localize(start)
        grid_end = tz.localize(end)
        full_grid = pd.date_range(
            start=grid_start,
            end=grid_end - bin_td,
            freq=f"{bin_len}min",
            tz=self.timezone,
        )

        bin_starts_utc = np.array([t.timestamp() for t in full_grid])
        bin_ends_utc = bin_starts_utc + bin_len * 60.0

        # ------------------------------------------------------------------
        # 1. Coverage from ingestion_log spans
        # ------------------------------------------------------------------
        with DatabaseManager(self.db_path) as m:
            spans_df = m.get_ingestion_spans()

        query_start_epoch = grid_start.timestamp()
        query_end_epoch = grid_end.timestamp()
        spans_df = spans_df.loc[
            (spans_df["span_end"] > query_start_epoch)
            & (spans_df["span_start"] < query_end_epoch)
        ].copy()

        coverage = np.zeros(len(full_grid), dtype=float)

        if not spans_df.empty:
            span_starts = spans_df["span_start"].values
            span_ends = spans_df["span_end"].values
            for i, (b_s, b_e) in enumerate(zip(bin_starts_utc, bin_ends_utc)):
                overlaps = np.maximum(
                    0.0,
                    np.minimum(span_ends, b_e) - np.maximum(span_starts, b_s),
                )
                coverage[i] = overlaps.sum() / (bin_len * 60.0)

        coverage = np.clip(coverage, 0.0, 1.0)

        # ------------------------------------------------------------------
        # 2. Downgrade bins containing a gap marker
        # ------------------------------------------------------------------
        gap_ts = events_df.loc[
            events_df["event_code"] == _GAP_CODE, "timestamp"
        ]
        if not gap_ts.empty:
            sample = gap_ts.iloc[0]
            if hasattr(sample, "timestamp"):
                gap_epochs = np.array([t.timestamp() for t in gap_ts])
            else:
                gap_epochs = gap_ts.values.astype(float)

            for i, (b_s, b_e) in enumerate(zip(bin_starts_utc, bin_ends_utc)):
                if np.any((gap_epochs >= b_s) & (gap_epochs < b_e)):
                    coverage[i] = min(coverage[i], 0.9999)

        # ------------------------------------------------------------------
        # 3. Quality labels
        # ------------------------------------------------------------------
        quality_labels = np.where(
            coverage == 1.0, "ok",
            np.where(coverage == 0.0, "missing", "partial"),
        )

        return pd.DataFrame(
            {"coverage": coverage, "data_quality": quality_labels},
            index=full_grid,
        )

    def _drop_missing_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows belonging to local calendar days where every bin is missing.

        Args:
            df: Binned AOG DataFrame with ``data_quality`` and ``time``
                columns.  ``time`` must contain tz-aware Timestamps.

        Returns:
            Filtered DataFrame.
        """
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
        """Read timezone from metadata table, falling back to US/Mountain."""
        try:
            with DatabaseManager(self.db_path) as m:
                meta = m.get_metadata()
                return meta.get("timezone") or "US/Mountain"
        except Exception:
            return "US/Mountain"

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
        """Fetch all AOG-relevant events from the database.

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
            event_codes=_ALL_AOG_CODES,
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

        Reads ``Det_P{N}_Arrival`` keys.  Phases with an empty or absent key
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

        # Collect all phases that have an Arrival config key
        arrival_keys = {
            k: v for k, v in config.items()
            if k.startswith("Det_P") and k.endswith("_Arrival") and v
        }

        for key, raw_val in arrival_keys.items():
            # Key format: Det_P{N}_Arrival  →  extract N
            try:
                ph_str = key[5:key.index("_Arrival")]  # between "Det_P" and "_Arrival"
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

        # Warn about explicitly requested phases that have no config
        if phases is not None:
            for ph in phases:
                if ph not in result:
                    print(
                        f"  ⚠️  AOG: no Det_P{ph}_Arrival config found — "
                        f"phase {ph} skipped."
                    )

        return result

    def _write_csvs(
        self,
        results: Dict[str, pd.DataFrame],
        output_dir: Union[str, Path],
        bin_len: Union[int, str],
    ) -> None:
        """Write result DataFrames to CSV files.

        Filename patterns::

            AOG_Cycle_{start}-{end}.csv              (cycle-level)
            AOG_{N}min_{start}-{end}.csv             (binned, integer bin_len)

        Args:
            results:    Dict returned by the internal calculation step.
            output_dir: Destination directory (created if absent).
            bin_len:    Bin length used (for filename).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cycle_df = results.get("cycle")
        binned_df = results.get("binned")

        def _date_range_str(df: pd.DataFrame, time_col: Optional[str] = None) -> str:
            """Extract start–end date string from a DataFrame."""
            if df is None or df.empty:
                return "unknown-unknown"
            if time_col and time_col in df.columns:
                ts = df[time_col]
            else:
                ts = df.index.to_series()
            # Handle both tz-aware Timestamps and float epochs
            sample = ts.iloc[0]
            if hasattr(sample, "strftime"):
                start_s = ts.min().strftime("%Y_%m_%d")
                end_s = ts.max().strftime("%Y_%m_%d")
            else:
                start_s = pd.Timestamp(sample, unit="s").strftime("%Y_%m_%d")
                end_s = pd.Timestamp(ts.max(), unit="s").strftime("%Y_%m_%d")
            return f"{start_s}-{end_s}"

        if cycle_df is not None and not cycle_df.empty:
            date_str = _date_range_str(cycle_df)
            filename = f"AOG_Cycle_{date_str}.csv"
            cycle_df.to_csv(output_dir / filename)
            print(f"Wrote {filename}")

        if binned_df is not None and not binned_df.empty:
            date_str = _date_range_str(binned_df, time_col="time")
            bin_str = "Cycle" if bin_len == "cycle" else f"{bin_len}min"
            filename = f"AOG_{bin_str}_{date_str}.csv"
            binned_df.to_csv(output_dir / filename, index=False)
            print(f"Wrote {filename}")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_arrival_on_green(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[List[int]] = None,
    arrival_offset_sec: float = 0.0,
    bin_len: Union[int, str] = 60,
    exclude_missing: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`AogEngine`.arrival_on_green.

    Args:
        db_path:             Path to the intersection SQLite database.
        start:               Inclusive start date/datetime (local time).
        end:                 Inclusive end date/datetime (local time).
        phases:              Phase IDs to analyse; ``None`` for all configured.
        arrival_offset_sec:  Travel-time offset in seconds.  Default ``0.0``.
        bin_len:             ``"cycle"`` or integer minutes.  Default ``60``.
        exclude_missing:     Drop partial/missing bins when ``True``.
        output_dir:          Write CSVs and return ``None`` when provided.
        timezone:            Override intersection timezone.

    Returns:
        Result dict or ``None`` if *output_dir* is set.
    """
    return AogEngine(db_path, timezone).arrival_on_green(
        start=start,
        end=end,
        phases=phases,
        arrival_offset_sec=arrival_offset_sec,
        bin_len=bin_len,
        exclude_missing=exclude_missing,
        output_dir=output_dir,
    )
