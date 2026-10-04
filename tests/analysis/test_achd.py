# Tests for the ACHD event-CSV parser (functional core).
#
# Pure-function contract only: header extraction, the raw-frame → events
# transform, timezone→UTC conversion, dtype coercion, de-duplication, and the
# error surface. No I/O, no database — the shell side is covered by
# tests/data/test_achd_ingestion.py.

from datetime import datetime

import pandas as pd
import pytest
import pytz

from atspm.analysis.achd import (
    AchdDecodingError,
    achd_events_from_frame,
    parse_achd_header,
)

TZ = "US/Mountain"

_HEADER = (
    "Signal,271 - Eagle Rd Ustick,\n"
    'Start time,"Sunday, 01 September 2024 00:00:00",\n'
    'End time,"Sunday, 15 September 2024 23:59:59",\n'
    'Total Events,"1,423,589",\n'
)


def _raw(rows):
    """Build a raw ACHD body frame with the padded ACHD column names."""
    return pd.DataFrame(
        rows,
        columns=["Event Time", " Event Code", " Event Description", " Event Parameter"],
    )


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def test_parse_header_splits_id_and_name():
    h = parse_achd_header(_HEADER)
    assert h.signal_id == "271"
    assert h.signal_name == "Eagle Rd Ustick"
    assert h.start == datetime(2024, 9, 1, 0, 0, 0)
    assert h.end == datetime(2024, 9, 15, 23, 59, 59)
    assert h.total_events == 1_423_589


def test_parse_header_tolerates_missing_fields():
    h = parse_achd_header("Signal,271 - Eagle Rd Ustick,\n")
    assert h.signal_id == "271"
    assert h.start is None and h.end is None and h.total_events is None


def test_parse_header_name_without_id_prefix():
    h = parse_achd_header("Signal,Some Road Only,\n")
    assert h.signal_id is None
    assert h.signal_name == "Some Road Only"


# ---------------------------------------------------------------------------
# Event transform
# ---------------------------------------------------------------------------

def test_timestamp_is_utc_epoch_from_local_wall_clock():
    # September → Mountain Daylight (UTC-6). Compare against an independent
    # pytz localisation rather than a hard-coded epoch.
    raw = _raw([["09/12/24 00:00:00.000", "81", "Veh Det Off", "9"]])
    out = achd_events_from_frame(raw, TZ)
    expected = pytz.timezone(TZ).localize(
        datetime(2024, 9, 12, 0, 0, 0)
    ).timestamp()
    assert out.loc[0, "timestamp"] == pytest.approx(expected)
    assert out.loc[0, "event_code"] == 81
    assert out.loc[0, "parameter"] == 9


def test_subsecond_precision_preserved():
    raw = _raw([["09/12/24 00:00:00.400", "131", "Coord Pattern", "254"]])
    out = achd_events_from_frame(raw, TZ)
    base = pytz.timezone(TZ).localize(datetime(2024, 9, 12, 0, 0, 0)).timestamp()
    assert out.loc[0, "timestamp"] == pytest.approx(base + 0.4, abs=1e-6)


def test_output_schema_and_dtypes():
    raw = _raw([["09/12/24 00:00:00.000", "82", "Veh Det On", "10"]])
    out = achd_events_from_frame(raw, TZ)
    assert list(out.columns) == ["timestamp", "event_code", "parameter"]
    assert str(out["timestamp"].dtype) == "float64"
    assert str(out["event_code"].dtype) == "int64"
    assert str(out["parameter"].dtype) == "int64"


def test_duplicate_triples_dropped():
    raw = _raw([
        ["09/12/24 00:00:00.000", "82", "Veh Det On", "10"],
        ["09/12/24 00:00:00.000", "82", "Veh Det On", "10"],  # exact dup
        ["09/12/24 00:00:00.000", "82", "Veh Det On", "35"],  # distinct param
    ])
    out = achd_events_from_frame(raw, TZ)
    assert len(out) == 2


def test_non_numeric_code_or_param_rows_dropped():
    raw = _raw([
        ["09/12/24 00:00:00.000", "81", "Veh Det Off", "9"],
        ["09/12/24 00:00:00.000", "", "Blank Code", "9"],
        ["09/12/24 00:00:00.000", "81", "Blank Param", ""],
    ])
    out = achd_events_from_frame(raw, TZ)
    assert len(out) == 1
    assert out.loc[0, "event_code"] == 81 and out.loc[0, "parameter"] == 9


def test_empty_input_returns_typed_empty_frame():
    out = achd_events_from_frame(_raw([]), TZ)
    assert out.empty
    assert list(out.columns) == ["timestamp", "event_code", "parameter"]


def test_missing_required_column_raises():
    bad = pd.DataFrame({"Event Time": ["09/12/24 00:00:00.000"], "Code": ["81"]})
    with pytest.raises(AchdDecodingError):
        achd_events_from_frame(bad, TZ)


def test_fall_back_dst_hour_does_not_crash_or_drop_rows():
    # 2024-11-03 01:00–02:00 local occurs twice in US/Mountain (MDT→MST).
    # Because real ACHD exports are sorted by wall clock, the two passes
    # interleave and 'infer' would raise ("N dst switches when there should
    # only be 1") — the bug the full-271 smoke test surfaced. The scalar
    # standard-time policy must localise the whole file without crashing and
    # without dropping any row, keeping UTC monotonic non-decreasing.
    rows = [
        ["11/03/24 00:59:59.000", "1", "x", "1"],  # pre-transition (MDT)
        ["11/03/24 01:00:30.000", "1", "x", "1"],  # ambiguous hour
        ["11/03/24 01:30:00.000", "1", "x", "1"],
        ["11/03/24 01:30:00.000", "1", "x", "2"],  # repeated wall time
        ["11/03/24 01:59:59.000", "1", "x", "3"],
        ["11/03/24 02:00:00.000", "1", "x", "1"],  # post-transition (MST)
    ]
    out = achd_events_from_frame(_raw(rows), TZ)
    assert len(out) == 6
    assert out["timestamp"].is_monotonic_increasing


def test_spring_forward_gap_hour_shifted_not_dropped():
    # 2025-03-09 02:00–03:00 local does not exist in US/Mountain. A stray
    # timestamp in that hole is shifted forward rather than raising.
    # A real spring-forward day has no 02:xx events (the clock jumps
    # 02:00→03:00), so the only invariant that matters is: a stray timestamp
    # in the hole localises (shifted forward) rather than raising, and no row
    # is dropped.
    rows = [
        ["03/09/25 01:59:59.000", "1", "x", "1"],
        ["03/09/25 02:30:00.000", "1", "x", "1"],  # non-existent → shifted fwd
        ["03/09/25 03:30:00.000", "1", "x", "1"],
    ]
    out = achd_events_from_frame(_raw(rows), TZ)
    assert len(out) == 3
