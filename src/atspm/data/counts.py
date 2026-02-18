"""
ATSPM Count Engine (Imperative Shell)

Orchestrates vehicle and pedestrian count generation by querying the
SQLite database, resolving configuration, and delegating to the
Functional Core (analysis/counts.py).

Package Location: src/atspm/data/counts.py

Gap Marker Rule:
    Gap markers (event_code = -1) must be present in every DataFrame
    passed to the Functional Core so that _segment_id() can enforce
    state resets at discontinuities.  This module ensures that by always
    including -1 in the event-code filter sent to the database:
    - vehicle_counts: loads 82 + phase-state codes + -1 (or 82 + -1 when
      no exclusions are configured).
    - ped_counts:     loads 21 + 45 + -1.
    - combined_counts: loads all codes (None), which includes -1.

Data Quality:
    Every time-binned result includes two extra columns:

    ``coverage`` (float, 0.0 – 1.0)
        Fraction of the bin duration that falls within a known ingested
        span.  Computed from ``ingestion_log`` (cheap — thousands of rows)
        without touching the events table.  A bin that straddles a span
        boundary or that contains a gap marker is further downgraded to at
        most ``"partial"`` regardless of the interval coverage fraction.

    ``data_quality`` (str)
        ``"ok"``      — coverage == 1.0 and no gap marker in bin
        ``"partial"`` — some data present but bin is incomplete
        ``"missing"`` — no ingested data covers this bin at all

    Full-day-missing days (every bin in a local calendar day is
    ``"missing"``) are always dropped from the output.

    The ``exclude_missing`` parameter (default ``False``) additionally
    removes ``"partial"`` and ``"missing"`` rows, leaving only bins
    where the data are complete.

    Quality labeling is not applied in ``bin_len="cycle"`` mode because
    cycles are bounded by detected barrier pulses and are inherently
    gap-aware.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .manager import DatabaseManager
from .reader import get_legacy_dataframe
from ..analysis.counts import (
    parse_exclusions_from_config,
    parse_movements_from_config,
    ped_counts as _ped_counts_core,
    vehicle_counts as _vehicle_counts_core,
)

# Event codes needed for phase-state exclusion filtering
_PHASE_STATE_CODES: List[int] = [1, 8, 9, 11, 12]
# Gap marker code — must always be included so the core can reset state
_GAP_CODE: int = -1


class CountEngine:
    """
    Queries the database and produces vehicle and pedestrian count tables.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).  The underlying
    ``get_legacy_dataframe`` call converts to UTC epochs for the SQL query
    and returns timezone-aware timestamps.

    Example::

        engine = CountEngine(Path("2068_data.db"))

        # 15-min bins with quality labels; drop days with no data at all
        df = engine.vehicle_counts("2025-06-01", "2025-06-07", bin_len=15)

        # Same but also drop partial bins
        df = engine.vehicle_counts(
            "2025-06-01", "2025-06-07", bin_len=15, exclude_missing=True
        )

        # Combined vehicle + ped in one call
        df = engine.combined_counts("2025-06-01", "2025-06-07")
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None):
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
    # Public methods
    # ------------------------------------------------------------------

    def vehicle_counts(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        bin_len: Union[int, str] = 60,
        hourly: bool = False,
        include_detectors: bool = False,
        exclude_missing: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Produce movement (and optionally detector) counts for a date range.

        The required event codes are determined automatically from the active
        configuration.  Gap markers (-1) are always included so data
        discontinuities are respected by the Functional Core.

        Args:
            start: Inclusive start date/datetime (``'YYYY-MM-DD'`` string or
                   ``datetime`` object, interpreted as local time).
            end:   Exclusive end date/datetime.
            bin_len: Aggregation interval in minutes, or ``"cycle"`` for
                     per-cycle aggregation (quality columns not added in
                     cycle mode).
            hourly: Scale numeric bins to hourly flow rate.
            include_detectors: Append raw per-detector count columns.
            exclude_missing: When ``True``, drop ``"partial"`` and
                ``"missing"`` bins from the output in addition to the
                always-dropped full-day-missing days.  Ignored in cycle mode.
            output_dir: When provided, write the result to a CSV file in this
                        directory and return ``None``.

        Returns:
            DataFrame indexed by ``Time`` with movement, TEV, ``coverage``,
            and ``data_quality`` columns (time-bin mode), or ``None`` if
            *output_dir* is set.
        """
        start_dt, end_dt = self._parse_range(start, end)
        config = self._get_config(start_dt)
        movements = parse_movements_from_config(config)
        exclusions = parse_exclusions_from_config(config)

        if exclusions:
            codes = sorted({82, _GAP_CODE} | set(_PHASE_STATE_CODES))
        else:
            codes = [82, _GAP_CODE]

        events_df = self._load_events(start_dt, end_dt, codes)

        if events_df.empty:
            return pd.DataFrame()

        result = _vehicle_counts_core(
            events_df=events_df,
            movements=movements,
            exclusions=exclusions,
            bin_len=bin_len,
            hourly=hourly,
            include_detectors=include_detectors,
        )

        if bin_len != "cycle":
            result = self._add_quality(
                result, events_df, start_dt, end_dt,
                int(bin_len), exclude_missing,
            )

        if output_dir is not None:
            self._write_csv(result, output_dir, "Counts", bin_len, hourly)
            return None

        return result

    def ped_counts(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        bin_len: Union[int, str] = 60,
        hourly: bool = False,
        exclude_missing: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Produce pedestrian actuation counts for a date range.

        Only pedestrian *services* (Code 21) preceded by at least one
        pedestrian *call* (Code 45) within the same continuous data segment
        are counted.  Gap markers (-1) are always loaded.

        Args:
            start: Inclusive start date/datetime (local time).
            end:   Exclusive end date/datetime.
            bin_len: Aggregation interval in minutes, or ``"cycle"``.
            hourly: Scale to hourly rate.
            exclude_missing: Drop ``"partial"`` and ``"missing"`` bins.
                Ignored in cycle mode.
            output_dir: Write CSV and return ``None`` when provided.

        Returns:
            DataFrame indexed by ``Time`` with ``Ped {phase}`` columns,
            ``Ped Total``, ``coverage``, and ``data_quality``, or ``None``
            if *output_dir* is set.
        """
        start_dt, end_dt = self._parse_range(start, end)
        events_df = self._load_events(start_dt, end_dt, [21, 45, _GAP_CODE])

        if events_df.empty:
            return pd.DataFrame()

        result = _ped_counts_core(
            events_df=events_df,
            bin_len=bin_len,
            hourly=hourly,
        )

        if bin_len != "cycle":
            result = self._add_quality(
                result, events_df, start_dt, end_dt,
                int(bin_len), exclude_missing,
            )

        if output_dir is not None:
            self._write_csv(result, output_dir, "PedCounts", bin_len, hourly)
            return None

        return result

    def combined_counts(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        bin_len: Union[int, str] = 60,
        hourly: bool = False,
        include_detectors: bool = False,
        exclude_missing: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Produce vehicle movement counts and pedestrian counts in a single call.

        Loads all event codes in one query (naturally includes gap markers)
        and runs both aggregations, then merges on the ``Time`` index.
        Quality columns are computed once and applied to the merged result.

        Args:
            start: Inclusive start (local time).
            end:   Exclusive end (local time).
            bin_len: Aggregation interval in minutes, or ``"cycle"``.
            hourly: Scale to hourly rate.
            include_detectors: Include raw detector columns in vehicle section.
            exclude_missing: Drop ``"partial"`` and ``"missing"`` bins.
                Ignored in cycle mode.
            output_dir: Write CSV and return ``None`` when provided.

        Returns:
            Merged DataFrame with vehicle, pedestrian, ``coverage``, and
            ``data_quality`` columns, or ``None`` if *output_dir* is set.
        """
        start_dt, end_dt = self._parse_range(start, end)
        config = self._get_config(start_dt)
        movements = parse_movements_from_config(config)
        exclusions = parse_exclusions_from_config(config)

        events_df = self._load_events(start_dt, end_dt, event_codes=None)

        if events_df.empty:
            return pd.DataFrame()

        veh = _vehicle_counts_core(
            events_df=events_df,
            movements=movements,
            exclusions=exclusions,
            bin_len=bin_len,
            hourly=hourly,
            include_detectors=include_detectors,
        )

        ped = _ped_counts_core(
            events_df=events_df,
            bin_len=bin_len,
            hourly=hourly,
        )

        if veh.empty and ped.empty:
            return pd.DataFrame()

        if ped.empty:
            result = veh
        elif veh.empty:
            result = ped
        else:
            result = pd.merge(
                veh, ped, how="outer", left_index=True, right_index=True
            )

        if bin_len != "cycle":
            result = self._add_quality(
                result, events_df, start_dt, end_dt,
                int(bin_len), exclude_missing,
            )

        if output_dir is not None:
            self._write_csv(result, output_dir, "Counts", bin_len, hourly)
            return None

        return result

    # ------------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------------

    def _add_quality(
        self,
        counts_df: pd.DataFrame,
        events_df: pd.DataFrame,
        start: datetime,
        end: datetime,
        bin_len: int,
        exclude_missing: bool,
    ) -> pd.DataFrame:
        """
        Annotate a binned counts DataFrame with coverage and quality labels,
        drop full-day-missing days, and optionally drop partial/missing bins.

        Args:
            counts_df:       Aggregated counts, indexed by bin-start timestamp.
            events_df:       Raw events DataFrame (used for gap marker
                             timestamps only — no full scan).
            start:           Query start (naive local datetime).
            end:             Query end (naive local datetime).
            bin_len:         Bin width in minutes.
            exclude_missing: Drop partial and missing rows when True.

        Returns:
            counts_df with ``coverage`` and ``data_quality`` columns added,
            full-day-missing days removed, and optionally partial/missing
            rows removed.
        """
        if counts_df.empty:
            return counts_df

        quality = self._compute_bin_quality(
            counts_df.index, events_df, start, end, bin_len
        )

        result = counts_df.copy()
        result["coverage"] = quality["coverage"]
        result["data_quality"] = quality["data_quality"]

        # Fill count columns with 0 for missing/partial bins that the core
        # produced no rows for (the index was built from the full bin grid).
        count_cols = [c for c in result.columns
                      if c not in ("coverage", "data_quality")]
        result[count_cols] = result[count_cols].fillna(0)

        # Drop full-day-missing days (every bin in a local date is "missing")
        result = self._drop_missing_days(result)

        if exclude_missing:
            result = result.loc[result["data_quality"] == "ok"]

        return result

    def _compute_bin_quality(
        self,
        bin_index: pd.DatetimeIndex,
        events_df: pd.DataFrame,
        start: datetime,
        end: datetime,
        bin_len: int,
    ) -> pd.DataFrame:
        """
        Compute coverage fraction and quality label for each bin.

        Coverage is computed from ``ingestion_log`` spans (cheap).  Any bin
        that also contains a gap marker event is downgraded to at most
        ``"partial"`` regardless of span coverage.

        A full bin grid is built from *start* to *end* so that bins with
        zero counts (true zeros) are present alongside missing-data bins.

        Args:
            bin_index: Index from the aggregated counts DataFrame.
            events_df: Raw events (used only to locate gap marker timestamps).
            start:     Query start (naive local datetime).
            end:       Query end (naive local datetime).
            bin_len:   Bin width in minutes.

        Returns:
            DataFrame indexed by bin start with columns
            ``["coverage", "data_quality"]``.
        """
        bin_td = timedelta(minutes=bin_len)

        # Build complete bin grid so every possible bin in the range is
        # represented, even those with no events at all.
        import pytz
        tz = pytz.timezone(self.timezone)
        grid_start = tz.localize(start)
        grid_end   = tz.localize(end)
        full_grid = pd.date_range(
            start=grid_start, end=grid_end - bin_td,
            freq=f"{bin_len}min", tz=self.timezone,
        )

        bin_starts_utc = full_grid.map(lambda t: t.timestamp())
        bin_ends_utc   = bin_starts_utc + bin_len * 60.0

        # ------------------------------------------------------------------
        # 1. Coverage from ingestion_log spans
        # ------------------------------------------------------------------
        with DatabaseManager(self.db_path) as m:
            spans_df = m.get_ingestion_spans()

        # Filter to spans that overlap the query range at all
        query_start_epoch = grid_start.timestamp()
        query_end_epoch   = grid_end.timestamp()
        spans_df = spans_df.loc[
            (spans_df["span_end"] > query_start_epoch) &
            (spans_df["span_start"] < query_end_epoch)
        ].copy()

        coverage = np.zeros(len(full_grid), dtype=float)

        if not spans_df.empty:
            span_starts = spans_df["span_start"].values
            span_ends   = spans_df["span_end"].values

            for i, (b_start, b_end) in enumerate(
                zip(bin_starts_utc, bin_ends_utc)
            ):
                # Covered seconds = sum of overlap between bin and each span
                overlaps = np.maximum(
                    0.0,
                    np.minimum(span_ends, b_end) - np.maximum(span_starts, b_start),
                )
                coverage[i] = overlaps.sum() / (bin_len * 60.0)

        coverage = np.clip(coverage, 0.0, 1.0)

        # ------------------------------------------------------------------
        # 2. Downgrade bins that contain a gap marker
        # ------------------------------------------------------------------
        gap_ts = events_df.loc[
            events_df["Code"] == _GAP_CODE, "TS_start"
        ]
        # Convert to UTC epoch floats for comparison
        if not gap_ts.empty:
            if hasattr(gap_ts.iloc[0], "timestamp"):
                gap_epochs = np.array([t.timestamp() for t in gap_ts])
            else:
                gap_epochs = gap_ts.values.astype(float)

            for i, (b_start, b_end) in enumerate(
                zip(bin_starts_utc, bin_ends_utc)
            ):
                if np.any((gap_epochs >= b_start) & (gap_epochs < b_end)):
                    # Has a gap marker — cap coverage so it never labels "ok"
                    coverage[i] = min(coverage[i], 0.9999)

        # ------------------------------------------------------------------
        # 3. Assign quality labels
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
        """
        Remove all rows belonging to a local calendar day where every bin
        is labeled ``"missing"``.

        Args:
            df: Counts DataFrame with ``data_quality`` column, indexed by
                tz-aware bin-start timestamps.

        Returns:
            Filtered DataFrame.
        """
        if df.empty or "data_quality" not in df.columns:
            return df

        local_dates = df.index.normalize()
        # A day is "all missing" when none of its bins are ok or partial
        day_has_data = (
            df.groupby(local_dates)["data_quality"]
            .apply(lambda s: (s != "missing").any())
        )
        good_days = day_has_data.index[day_has_data]
        keep_mask = local_dates.isin(good_days)

        return df.loc[keep_mask]

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
        """Coerce string dates to datetime objects (midnight, naive local)."""
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        return start, end

    def _load_events(
        self,
        start: datetime,
        end: datetime,
        event_codes: Optional[List[int]],
    ) -> pd.DataFrame:
        """
        Fetch events from the database via the legacy reader.

        Args:
            start:       Naive local start datetime.
            end:         Naive local end datetime.
            event_codes: Code filter list, or ``None`` to load all codes.

        Returns:
            Legacy-format DataFrame with tz-aware ``TS_start`` / ``Cycle_start``.
        """
        return get_legacy_dataframe(
            db_path=self.db_path,
            start=start,
            end=end,
            event_codes=event_codes,
            timezone=self.timezone,
        )

    def _get_config(self, date: datetime) -> Dict[str, Any]:
        """Retrieve the active configuration for a given date."""
        with DatabaseManager(self.db_path) as m:
            config = m.get_config_at_date(date)
        return config or {}

    @staticmethod
    def _write_csv(
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        prefix: str,
        bin_len: Union[int, str],
        hourly: bool,
    ) -> None:
        """Write *df* to a timestamped CSV file in *output_dir*."""
        if df is None or df.empty:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_str = df.index.min().strftime("%Y_%m_%d")
        end_str   = df.index.max().strftime("%Y_%m_%d")
        bin_str   = "Cycle" if bin_len == "cycle" else f"{bin_len}min"
        hrly_str  = "hrly_" if hourly else ""
        filename  = f"{prefix}_{bin_str}_{hrly_str}{start_str}-{end_str}.csv"

        df.to_csv(output_dir / filename)
        print(f"Wrote {filename}")


# ---------------------------------------------------------------------------
# Convenience entry-points
# ---------------------------------------------------------------------------

def get_vehicle_counts(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    bin_len: Union[int, str] = 60,
    hourly: bool = False,
    include_detectors: bool = False,
    exclude_missing: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Convenience wrapper around :class:`CountEngine`.vehicle_counts."""
    return CountEngine(db_path, timezone).vehicle_counts(
        start=start, end=end, bin_len=bin_len, hourly=hourly,
        include_detectors=include_detectors, exclude_missing=exclude_missing,
        output_dir=output_dir,
    )


def get_ped_counts(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    bin_len: Union[int, str] = 60,
    hourly: bool = False,
    exclude_missing: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Convenience wrapper around :class:`CountEngine`.ped_counts."""
    return CountEngine(db_path, timezone).ped_counts(
        start=start, end=end, bin_len=bin_len, hourly=hourly,
        exclude_missing=exclude_missing, output_dir=output_dir,
    )


def get_combined_counts(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    bin_len: Union[int, str] = 60,
    hourly: bool = False,
    include_detectors: bool = False,
    exclude_missing: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Convenience wrapper around :class:`CountEngine`.combined_counts."""
    return CountEngine(db_path, timezone).combined_counts(
        start=start, end=end, bin_len=bin_len, hourly=hourly,
        include_detectors=include_detectors, exclude_missing=exclude_missing,
        output_dir=output_dir,
    )