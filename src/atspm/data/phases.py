"""
ATSPM Phase Split Engine (Imperative Shell)

Orchestrates phase timing analysis by querying the SQLite database and
delegating all calculations to the Functional Core (analysis/phases.py).

Package Location: src/atspm/data/phases.py

Data Quality:
    Time-binned results receive the same ``coverage`` / ``data_quality``
    annotation used by the counts engine (computed from ``ingestion_log``
    spans, downgraded for bins containing gap markers).  Full-day-missing
    days are always dropped.  The ``exclude_missing`` flag additionally
    removes ``"partial"`` and ``"missing"`` bins.

    Quality annotation is skipped in ``bin_len="cycle"`` mode because
    cycle boundaries are defined by detected barrier pulses and are
    inherently gap-aware.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from .manager import DatabaseManager
from .reader import get_legacy_dataframe
from ..analysis.phases import phase_splits as _phase_splits_core, ReportMode

# Phase state event codes — must include gap marker so the core sees discontinuities
_PHASE_CODES: List[int] = [1, 8, 9, 11, 12]
_GAP_CODE: int = -1


class PhaseEngine:
    """
    Queries the database and produces phase timing tables.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = PhaseEngine(Path("2068_data.db"))

        # Per-cycle table
        df = engine.phase_splits("2025-06-01", "2025-06-07")

        # 15-min bins, average seconds per cycle
        df = engine.phase_splits(
            "2025-06-01", "2025-06-07", bin_len=15, report_mode="seconds"
        )

        # 15-min bins, proportion of bin duration
        df = engine.phase_splits(
            "2025-06-01", "2025-06-07", bin_len=15, report_mode="proportion"
        )

        # Save to CSV
        engine.phase_splits(
            "2025-06-01", "2025-06-07", bin_len=15, output_dir=Path("./out")
        )
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
    # Public method
    # ------------------------------------------------------------------

    def phase_splits(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        bin_len: Union[int, str] = "cycle",
        report_mode: ReportMode = "seconds",
        phases: Optional[List[int]] = None,
        include_no_clearance: bool = False,
        exclude_missing: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Produce a phase timing table for a date range.

        Args:
            start: Inclusive start date/datetime (``'YYYY-MM-DD'`` string or
                   ``datetime`` object, interpreted as local time).
            end:   Exclusive end date/datetime.
            bin_len: ``"cycle"`` for one row per detected cycle, or an integer
                     number of minutes for time-bin aggregation.
                     Default ``"cycle"``.
            report_mode: How binned values are expressed.  Ignored when
                ``bin_len="cycle"``.

                ``"seconds"``    – mean seconds per phase component per cycle
                                   in the bin (e.g., ``12.4`` s green).
                ``"total"``      – total seconds summed across all cycles in
                                   the bin (e.g., ``63`` s green in 15 min).
                ``"proportion"`` – mean seconds divided by bin duration,
                                   a 0.0–1.0 utilisation fraction
                                   (e.g., ``0.07`` ≈ 7 % green).

            phases: Restrict output to these phase IDs (sorted ascending).
                    ``None`` includes all phases with observed activity.
            include_no_clearance: When ``True``, phases that go Code 1 →
                Code 12 with no yellow logged are included as green-only
                intervals (``YR = 0``, ``Split = Green``).  Use this for
                dummy phases that drive overlaps and carry no independent
                yellow timing.  Default ``False``.
            exclude_missing: When ``True``, drop ``"partial"`` and
                ``"missing"`` bins from the output in addition to the
                always-dropped full-day-missing days.  Ignored in cycle mode.
            output_dir: When provided, write the result to a timestamped CSV
                        in this directory and return ``None``.

        Returns:
            DataFrame indexed by ``Time`` with phase timing columns and
            ``Cycle Length``.  Time-bin mode additionally includes
            ``coverage`` and ``data_quality`` columns.  Returns ``None``
            if *output_dir* is set, or an empty DataFrame when no phase
            events are found.
        """
        start_dt, end_dt = self._parse_range(start, end)
        events_df = self._load_events(start_dt, end_dt)

        if events_df.empty:
            return pd.DataFrame()

        result = _phase_splits_core(
            events_df=events_df,
            bin_len=bin_len,
            report_mode=report_mode,
            phases=phases,
            include_no_clearance=include_no_clearance,
        )

        if result.empty:
            return pd.DataFrame()

        if bin_len != "cycle":
            result = self._add_quality(
                result, events_df, start_dt, end_dt,
                int(bin_len), exclude_missing,
            )

        if output_dir is not None:
            self._write_csv(result, output_dir, bin_len, report_mode)
            return None

        return result

    # ------------------------------------------------------------------
    # Data quality (mirrors CountEngine._add_quality / _compute_bin_quality)
    # ------------------------------------------------------------------

    def _add_quality(
        self,
        result: pd.DataFrame,
        events_df: pd.DataFrame,
        start: datetime,
        end: datetime,
        bin_len: int,
        exclude_missing: bool,
    ) -> pd.DataFrame:
        """
        Annotate a binned result with ``coverage`` and ``data_quality``
        columns, drop full-day-missing days, and optionally drop
        partial/missing rows.

        Args:
            result:          Aggregated phase timing DataFrame.
            events_df:       Raw events (used for gap marker timestamps only).
            start:           Query start (naive local datetime).
            end:             Query end (naive local datetime).
            bin_len:         Bin width in minutes.
            exclude_missing: Drop partial and missing rows when True.

        Returns:
            Annotated and filtered DataFrame.
        """
        if result.empty:
            return result

        quality = self._compute_bin_quality(
            result.index, events_df, start, end, bin_len
        )

        out = result.copy()
        out["coverage"]     = quality["coverage"]
        out["data_quality"] = quality["data_quality"]

        out = self._drop_missing_days(out)

        if exclude_missing:
            out = out.loc[out["data_quality"] == "ok"]

        return out

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

        Coverage is derived from ``ingestion_log`` spans (cheap — O(spans),
        not O(events)).  Bins containing a gap marker are capped at
        ``"partial"`` regardless of span coverage.

        Args:
            bin_index: Index from the aggregated result DataFrame.
            events_df: Raw events (used only to locate gap marker timestamps).
            start:     Query start (naive local datetime).
            end:       Query end (naive local datetime).
            bin_len:   Bin width in minutes.

        Returns:
            DataFrame indexed by bin-start with columns
            ``["coverage", "data_quality"]``.
        """
        import pytz

        bin_td = timedelta(minutes=bin_len)
        tz = pytz.timezone(self.timezone)

        grid_start = tz.localize(start)
        grid_end   = tz.localize(end)
        full_grid  = pd.date_range(
            start=grid_start,
            end=grid_end - bin_td,
            freq=f"{bin_len}min",
            tz=self.timezone,
        )

        bin_starts_utc = np.array([t.timestamp() for t in full_grid])
        bin_ends_utc   = bin_starts_utc + bin_len * 60.0

        # ------------------------------------------------------------------
        # 1. Coverage from ingestion_log spans
        # ------------------------------------------------------------------
        with DatabaseManager(self.db_path) as m:
            spans_df = m.get_ingestion_spans()

        query_start_epoch = grid_start.timestamp()
        query_end_epoch   = grid_end.timestamp()
        spans_df = spans_df.loc[
            (spans_df["span_end"]   > query_start_epoch) &
            (spans_df["span_start"] < query_end_epoch)
        ].copy()

        coverage = np.zeros(len(full_grid), dtype=float)

        if not spans_df.empty:
            span_starts = spans_df["span_start"].values
            span_ends   = spans_df["span_end"].values
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
        gap_ts = events_df.loc[events_df["Code"] == _GAP_CODE, "TS_start"]
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
        """
        Remove rows belonging to local calendar days where every bin is
        labeled ``"missing"``.

        Args:
            df: Result DataFrame with ``data_quality`` column, indexed by
                tz-aware bin-start timestamps.

        Returns:
            Filtered DataFrame.
        """
        if df.empty or "data_quality" not in df.columns:
            return df

        local_dates = df.index.normalize()
        day_has_data = (
            df.groupby(local_dates)["data_quality"]
            .apply(lambda s: (s != "missing").any())
        )
        good_days = day_has_data.index[day_has_data]
        return df.loc[local_dates.isin(good_days)]

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
        """Coerce string dates to naive midnight datetimes."""
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
        """
        Fetch phase state-change events and gap markers from the database.

        Args:
            start: Naive local start datetime.
            end:   Naive local end datetime.

        Returns:
            Legacy-format DataFrame with tz-aware ``TS_start`` / ``Cycle_start``.
        """
        codes = sorted(set(_PHASE_CODES) | {_GAP_CODE})
        return get_legacy_dataframe(
            db_path=self.db_path,
            start=start,
            end=end,
            event_codes=codes,
            timezone=self.timezone,
        )

    def _write_csv(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        bin_len: Union[int, str],
        report_mode: ReportMode,
    ) -> None:
        """
        Write *df* to a timestamped CSV file.

        Filename pattern::

            PhaseSplits_{bin}_{mode}_{start}-{end}.csv

        Args:
            df:          Result DataFrame to write.
            output_dir:  Destination directory (created if absent).
            bin_len:     Bin length used (for filename).
            report_mode: Report mode used (for filename).
        """
        if df is None or df.empty:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_str = df.index.min().strftime("%Y_%m_%d")
        end_str   = df.index.max().strftime("%Y_%m_%d")
        bin_str   = "Cycle" if bin_len == "cycle" else f"{bin_len}min"
        mode_str  = "" if bin_len == "cycle" else f"_{report_mode}"
        filename  = f"PhaseSplits_{bin_str}{mode_str}_{start_str}-{end_str}.csv"

        df.to_csv(output_dir / filename)
        print(f"Wrote {filename}")


# ---------------------------------------------------------------------------
# Convenience entry-points
# ---------------------------------------------------------------------------


def get_phase_splits(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    bin_len: Union[int, str] = "cycle",
    report_mode: ReportMode = "seconds",
    phases: Optional[List[int]] = None,
    include_no_clearance: bool = False,
    exclude_missing: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Convenience wrapper around :class:`PhaseEngine`.phase_splits.

    Args:
        db_path:              Path to the intersection SQLite database.
        start:                Inclusive start date/datetime (local time).
        end:                  Exclusive end date/datetime (local time).
        bin_len:              ``"cycle"`` or integer minutes.  Default ``"cycle"``.
        report_mode:          ``"seconds"`` | ``"total"`` | ``"proportion"``.
                              Ignored when ``bin_len="cycle"``.
        phases:               Phase IDs to include; ``None`` for all.
        include_no_clearance: Include dummy/overlap-driven phases with no
                              yellow timing as green-only intervals.
        exclude_missing:      Drop partial/missing bins when ``True``.
        output_dir:           Write CSV and return ``None`` when provided.
        timezone:             Override intersection timezone.

    Returns:
        Phase timing DataFrame or ``None`` if *output_dir* is set.
    """
    return PhaseEngine(db_path, timezone).phase_splits(
        start=start,
        end=end,
        bin_len=bin_len,
        report_mode=report_mode,
        phases=phases,
        include_no_clearance=include_no_clearance,
        exclude_missing=exclude_missing,
        output_dir=output_dir,
    )