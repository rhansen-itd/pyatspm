"""Shared bin-quality computation for binned engine results (Functional Core).

Pure function: DataFrames and scalars in, DataFrame out — no SQL, no file
I/O.  Callers in the Imperative Shell (``CountEngine``, ``PhaseEngine``,
``AogEngine``) fetch ingestion spans and raw events themselves and pass
them in, mirroring the ``utils.timezone.resolve_pytz`` precedent.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .timezone import resolve_pytz

# Gap marker event code — a hard reset; a bin containing one can never be "ok".
_GAP_CODE: int = -1


def compute_bin_quality(
    events_df: pd.DataFrame,
    spans_df: pd.DataFrame,
    start: datetime,
    end: datetime,
    bin_len: int,
    timezone: str,
) -> pd.DataFrame:
    """Compute coverage fraction and quality label for each bin.

    Coverage is derived from ``ingestion_log`` spans (cheap — O(spans),
    not O(events)).  Bins containing a gap marker (``event_code == -1``)
    are capped at ``"partial"`` regardless of span coverage; a bin with
    zero coverage stays ``"missing"`` even if it holds a marker.

    A full bin grid is built from *start* to *end* so that bins with no
    events at all (true zeros) are present alongside missing-data bins.

    Args:
        events_df: Raw events (used only to locate gap marker timestamps).
            ``timestamp`` may hold UTC-epoch floats or Timestamps.
        spans_df:  Ingestion spans with ``span_start`` / ``span_end``
            UTC-epoch columns (``DatabaseManager.get_ingestion_spans``).
        start:     Query start (naive local datetime).
        end:       Query end (naive local datetime).
        bin_len:   Bin width in minutes.
        timezone:  IANA timezone name used to localize the bin grid.

    Returns:
        DataFrame indexed by tz-aware bin-start Timestamps with columns
        ``["coverage", "data_quality"]``.
    """
    bin_td = timedelta(minutes=bin_len)
    tz = resolve_pytz(timezone)

    grid_start = tz.localize(start)
    grid_end = tz.localize(end)
    full_grid = pd.date_range(
        start=grid_start,
        end=grid_end - bin_td,
        freq=f"{bin_len}min",
        tz=tz,
    )

    bin_starts_utc = np.array([t.timestamp() for t in full_grid])
    bin_ends_utc = bin_starts_utc + bin_len * 60.0

    # ------------------------------------------------------------------
    # 1. Coverage from ingestion_log spans
    # ------------------------------------------------------------------
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
    gap_ts = events_df.loc[events_df["event_code"] == _GAP_CODE, "timestamp"]
    if not gap_ts.empty:
        sample = gap_ts.iloc[0]
        if hasattr(sample, "timestamp"):
            gap_epochs = np.array([t.timestamp() for t in gap_ts])
        else:
            gap_epochs = gap_ts.values.astype(float)

        # Bins are [start, end): a marker exactly on a bin edge downgrades
        # the bin it starts, never the bin it ends.
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
