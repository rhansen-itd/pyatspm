"""Gap-marker correctness tests for shared bin-quality computation.

Target: src/atspm/utils/quality.py — compute_bin_quality.
Pure function: events + ingestion spans in, coverage/quality DataFrame out.
Bins are [start, end): the CLAUDE.md §5 gap-marker rule (``event_code == -1``
downgrades a bin to at most "partial") is pinned here, including the
edge-exact case where drift between the former per-engine copies would hide.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

from atspm.utils.quality import compute_bin_quality

TZ = "UTC"
START = datetime(2026, 1, 5)
END = START + timedelta(hours=1)
BIN_LEN = 15  # minutes → four bins: 00:00, 00:15, 00:30, 00:45

START_EPOCH = pytz.utc.localize(START).timestamp()
END_EPOCH = pytz.utc.localize(END).timestamp()


def _events(gap_offsets_min=(), as_timestamps=False):
    """Build an events DataFrame with detector noise plus optional gap markers.

    Args:
        gap_offsets_min: Offsets (minutes after START) for event_code -1 rows.
        as_timestamps:   Emit tz-aware Timestamps instead of epoch floats.

    Returns:
        DataFrame with ``event_code`` and ``timestamp`` columns.
    """
    rows = [(82, START_EPOCH + 60.0 * m) for m in (1, 20, 40)]
    rows += [(-1, START_EPOCH + 60.0 * m) for m in gap_offsets_min]
    df = pd.DataFrame(rows, columns=["event_code", "timestamp"])
    if as_timestamps:
        df["timestamp"] = df["timestamp"].map(
            lambda e: pd.Timestamp(e, unit="s", tz="UTC")
        )
    return df


def _full_span():
    """Single ingestion span covering the whole query window."""
    return pd.DataFrame(
        {"span_start": [START_EPOCH], "span_end": [END_EPOCH]}
    )


class TestCoverageBaseline:

    def test_full_span_no_gap_labels_every_bin_ok(self):
        out = compute_bin_quality(
            _events(), _full_span(), START, END, BIN_LEN, TZ
        )
        assert len(out) == 4
        assert (out["data_quality"] == "ok").all()
        assert (out["coverage"] == 1.0).all()

    def test_no_spans_labels_every_bin_missing(self):
        empty_spans = pd.DataFrame({"span_start": [], "span_end": []})
        out = compute_bin_quality(
            _events(), empty_spans, START, END, BIN_LEN, TZ
        )
        assert (out["data_quality"] == "missing").all()
        assert (out["coverage"] == 0.0).all()


class TestGapMarkerDowngrade:

    def test_marker_mid_bin_downgrades_only_that_bin_to_partial(self):
        # Marker at 00:22:30 — strictly inside bin 1 (00:15–00:30).
        out = compute_bin_quality(
            _events(gap_offsets_min=[22.5]), _full_span(),
            START, END, BIN_LEN, TZ,
        )
        assert list(out["data_quality"]) == ["ok", "partial", "ok", "ok"]
        assert out["coverage"].iloc[1] < 1.0  # capped despite full span

    def test_marker_exactly_on_bin_edge_downgrades_bin_it_starts(self):
        # Marker at exactly 00:15:00. Bins are [start, end): the marker
        # belongs to bin 1 (which it starts), never bin 0 (which it ends).
        out = compute_bin_quality(
            _events(gap_offsets_min=[15]), _full_span(),
            START, END, BIN_LEN, TZ,
        )
        assert out["data_quality"].iloc[0] == "ok"
        assert out["data_quality"].iloc[1] == "partial"
        assert list(out["data_quality"].iloc[2:]) == ["ok", "ok"]

    def test_marker_in_zero_coverage_bin_leaves_it_missing(self):
        # Span covers only the first three bins; marker falls in the last.
        spans = pd.DataFrame(
            {"span_start": [START_EPOCH],
             "span_end": [START_EPOCH + 45 * 60.0]}
        )
        out = compute_bin_quality(
            _events(gap_offsets_min=[50]), spans, START, END, BIN_LEN, TZ
        )
        assert out["data_quality"].iloc[3] == "missing"
        assert out["coverage"].iloc[3] == 0.0

    def test_timestamp_typed_gap_markers_are_handled(self):
        # Same mid-bin case, but timestamps arrive as tz-aware Timestamps
        # (exercises the Timestamp→epoch conversion branch).
        out = compute_bin_quality(
            _events(gap_offsets_min=[22.5], as_timestamps=True),
            _full_span(), START, END, BIN_LEN, TZ,
        )
        assert list(out["data_quality"]) == ["ok", "partial", "ok", "ok"]


class TestPurity:

    def test_inputs_are_not_mutated(self):
        events = _events(gap_offsets_min=[22.5])
        spans = _full_span()
        events_before = events.copy(deep=True)
        spans_before = spans.copy(deep=True)

        compute_bin_quality(events, spans, START, END, BIN_LEN, TZ)

        pd.testing.assert_frame_equal(events, events_before)
        pd.testing.assert_frame_equal(spans, spans_before)


class TestAwareBoundsAccepted:
    """The engines pass their public start/end straight through, so a caller
    handing in aware datetimes must not hit pytz's naive-only ``localize``."""

    def test_aware_bounds_produce_the_same_grid_as_naive(self):
        naive = compute_bin_quality(
            _events(), _full_span(), START, END, BIN_LEN, TZ,
        )
        aware = compute_bin_quality(
            _events(),
            _full_span(),
            pytz.utc.localize(START),
            pytz.utc.localize(END),
            BIN_LEN,
            TZ,
        )
        pd.testing.assert_frame_equal(naive, aware)

    def test_aware_bounds_in_another_zone_describe_the_same_instants(self):
        mountain = pytz.timezone("US/Mountain")
        naive = compute_bin_quality(
            _events(), _full_span(), START, END, BIN_LEN, TZ,
        )
        # 2026-01-05 00:00 UTC expressed on a Mountain wall clock.
        shifted = compute_bin_quality(
            _events(),
            _full_span(),
            pytz.utc.localize(START).astimezone(mountain),
            pytz.utc.localize(END).astimezone(mountain),
            BIN_LEN,
            TZ,
        )
        assert list(shifted["coverage"]) == list(naive["coverage"])
        assert list(shifted["data_quality"]) == list(naive["data_quality"])
