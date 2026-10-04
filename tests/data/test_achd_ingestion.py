# Tests for the ACHD ingestion engine (imperative shell).
#
# The parser's arithmetic is covered by tests/analysis/test_achd.py. Here we
# assert the shell contract: events land in the normalised schema, metadata is
# seeded from the header, ingestion_log spans track real event extent, a data
# gap between files is fenced with a comms-gap marker and splits the span, and
# a fully-contained re-export is skipped.

from datetime import datetime, timedelta
from pathlib import Path

import pytz

from atspm.analysis.decoders import COMMS_GAP_PARAM
from atspm.data.achd_ingestion import AchdIngestionEngine, run_achd_ingestion
from atspm.data.manager import DatabaseManager

TZ = pytz.timezone("US/Mountain")


def _epoch(y, mo, d, h, mi, s=0, us=0):
    return TZ.localize(datetime(y, mo, d, h, mi, s, us)).timestamp()


def _write_achd_csv(raw_dir: Path, filename: str, start: datetime,
                    end: datetime, rows, signal="271 - Eagle Rd Ustick"):
    """Write a minimal ACHD event CSV with the four-line metadata header.

    Args:
        rows: iterable of ``(time_str, code, param)`` — the description column
            is filled in automatically.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%A, %d %B %Y %H:%M:%S"
    lines = [
        f"Signal,{signal},",
        f'Start time,"{start.strftime(fmt)}",',
        f'End time,"{end.strftime(fmt)}",',
        f'Total Events,"{len(rows)}",',
        "Event Time, Event Code, Event Description, Event Parameter,",
    ]
    for t, code, param in rows:
        lines.append(f"{t},{code},Event,{param},")
    path = raw_dir / filename
    path.write_text("\n".join(lines) + "\n")
    return path


def _query(db_path, sql):
    with DatabaseManager(db_path) as m:
        return m.conn.cursor().execute(sql).fetchall()


def test_single_file_ingests_events_metadata_and_span(tmp_path):
    raw = tmp_path / "raw"
    _write_achd_csv(
        raw, "271_Events_20240912T0000.csv",
        datetime(2024, 9, 12, 0, 0, 0), datetime(2024, 9, 12, 23, 59, 59),
        rows=[
            ("09/12/24 00:00:00.000", 81, 9),
            ("09/12/24 00:00:01.000", 82, 10),
            ("09/12/24 00:00:02.500", 131, 254),
        ],
    )
    db_path = tmp_path / "271_data.db"
    run_achd_ingestion(db_path, raw, "271")

    events = _query(db_path, "SELECT timestamp, event_code, parameter FROM events ORDER BY timestamp")
    assert len(events) == 3
    assert events[0] == (_epoch(2024, 9, 12, 0, 0, 0), 81, 9)

    meta = _query(db_path, "SELECT intersection_id, intersection_name, timezone, agency_id FROM metadata")
    assert meta == [("271", "Eagle Rd Ustick", "US/Mountain", "ACHD")]

    spans = _query(db_path, "SELECT span_start, span_end, row_count FROM ingestion_log")
    assert len(spans) == 1
    assert spans[0][0] == _epoch(2024, 9, 12, 0, 0, 0)
    assert spans[0][1] == _epoch(2024, 9, 12, 0, 0, 2, 500000)
    assert spans[0][2] == 3


def test_gap_between_files_fenced_and_split(tmp_path):
    raw = tmp_path / "raw"
    # Day 1, then a jump of several days before the next export → a real gap.
    _write_achd_csv(
        raw, "271_Events_20240912T0000.csv",
        datetime(2024, 9, 12, 0, 0, 0), datetime(2024, 9, 12, 23, 59, 59),
        rows=[("09/12/24 00:00:00.000", 81, 9), ("09/12/24 23:59:59.000", 82, 10)],
    )
    _write_achd_csv(
        raw, "271_Events_20240920T0000.csv",
        datetime(2024, 9, 20, 0, 0, 0), datetime(2024, 9, 20, 23, 59, 59),
        rows=[("09/20/24 00:00:00.000", 81, 9), ("09/20/24 00:00:01.000", 82, 10)],
    )
    db_path = tmp_path / "271_data.db"
    run_achd_ingestion(db_path, raw, "271")

    gaps = _query(
        db_path,
        f"SELECT timestamp FROM events WHERE event_code = -1 AND parameter = {COMMS_GAP_PARAM}",
    )
    assert len(gaps) == 1
    # Marker sits at the first event of the later segment.
    assert gaps[0][0] == _epoch(2024, 9, 20, 0, 0, 0)

    # Two contiguous segments → two spans.
    spans = _query(db_path, "SELECT span_start, span_end FROM ingestion_log ORDER BY span_start")
    assert len(spans) == 2
    assert spans[0][1] == _epoch(2024, 9, 12, 23, 59, 59)
    assert spans[1][0] == _epoch(2024, 9, 20, 0, 0, 0)


def test_no_gap_marker_within_threshold(tmp_path):
    raw = tmp_path / "raw"
    # Back-to-back daily exports hand off within seconds → one continuous span.
    _write_achd_csv(
        raw, "271_Events_20240912T0000.csv",
        datetime(2024, 9, 12, 0, 0, 0), datetime(2024, 9, 12, 23, 59, 59),
        rows=[("09/12/24 23:59:59.000", 82, 10)],
    )
    _write_achd_csv(
        raw, "271_Events_20240913T0000.csv",
        datetime(2024, 9, 13, 0, 0, 0), datetime(2024, 9, 13, 23, 59, 59),
        rows=[("09/13/24 00:00:00.000", 81, 9)],
    )
    db_path = tmp_path / "271_data.db"
    run_achd_ingestion(db_path, raw, "271")

    gaps = _query(db_path, "SELECT COUNT(*) FROM events WHERE event_code = -1")
    assert gaps[0][0] == 0
    spans = _query(db_path, "SELECT COUNT(*) FROM ingestion_log")
    assert spans[0][0] == 1


def test_contained_file_is_skipped(tmp_path):
    raw = tmp_path / "raw"
    # A wide export covering the whole window…
    _write_achd_csv(
        raw, "271_Events_20240912T0000.csv",
        datetime(2024, 9, 12, 0, 0, 0), datetime(2024, 9, 14, 23, 59, 59),
        rows=[("09/12/24 00:00:00.000", 81, 9), ("09/13/24 00:00:00.000", 82, 10)],
    )
    # …and a narrow re-export fully inside it (different rows, to prove it is
    # the file — not just duplicate rows — that is skipped).
    _write_achd_csv(
        raw, "271_Events_20240913T0000.csv",
        datetime(2024, 9, 13, 0, 0, 0), datetime(2024, 9, 13, 23, 59, 59),
        rows=[("09/13/24 12:00:00.000", 99, 1)],
    )
    db_path = tmp_path / "271_data.db"
    stats = run_achd_ingestion(db_path, raw, "271")

    assert stats["files_processed"] == 1
    codes = {c for (c,) in _query(db_path, "SELECT DISTINCT event_code FROM events")}
    assert 99 not in codes  # the contained file's event never made it in


def test_only_matching_intersection_files_ingested(tmp_path):
    raw = tmp_path / "raw"
    _write_achd_csv(
        raw, "271_Events_20240912T0000.csv",
        datetime(2024, 9, 12, 0, 0, 0), datetime(2024, 9, 12, 23, 59, 59),
        rows=[("09/12/24 00:00:00.000", 81, 9)], signal="271 - Eagle Rd Ustick",
    )
    _write_achd_csv(
        raw, "272_Events_20240912T0000.csv",
        datetime(2024, 9, 12, 0, 0, 0), datetime(2024, 9, 12, 23, 59, 59),
        rows=[("09/12/24 00:00:00.000", 81, 9)], signal="272 - Other Rd",
    )
    db_path = tmp_path / "271_data.db"
    stats = run_achd_ingestion(db_path, raw, "271")
    assert stats["files_processed"] == 1
    assert stats["total_events"] == 1
