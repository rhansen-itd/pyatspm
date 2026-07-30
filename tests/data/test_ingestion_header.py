# Tests for the .datZ header sub-minute offset at the ingestion boundary
# (imperative shell).
#
# The decoder's arithmetic is covered by tests/analysis/test_decoders.py.
# Here we care about the shell contract: stored event timestamps carry the
# header offset, the ingestion_log span stays anchored to the filename clock
# boundary (never shifted), and a header whose date/HH:MM disagrees with the
# filename is surfaced rather than silently absorbed.

import struct
import zlib
from datetime import datetime
from pathlib import Path

import pytest
import pytz

from atspm.data.ingestion import IngestionEngine
from atspm.data.manager import DatabaseManager

TZ = pytz.timezone("US/Mountain")


def _boundary_epoch(year: int, month: int, day: int, hour: int, minute: int) -> float:
    """UTC epoch of a local clock boundary, matching _parse_filename_timestamp."""
    return TZ.localize(datetime(year, month, day, hour, minute)).timestamp()


def _write_datz(
    raw_dir: Path,
    filename: str,
    clock: str,
    offsets_deciseconds=(0, 100),
) -> Path:
    """Write one compressed .datZ file with a real controller preamble.

    Args:
        raw_dir:            Directory to write into.
        filename:           ``*_YYYY_MM_DD_HHMM.datZ`` name.
        clock:              ``<M/D/YYYY>,<HH:MM:SS.s>`` for the
                            ``Controller Data Log Beginning`` line.
        offsets_deciseconds: Binary time offsets to emit, one event each.
    """
    preamble = (
        b"Version #:,3\n"
        b"Controller Data Log Beginning:," + clock.encode() + b"\n"
        b"Phases in use:,1,2,3,4,5,6,7,8\n"
    )
    payload = b"".join(struct.pack(">BBH", 1, 2, o) for o in offsets_deciseconds)
    path = raw_dir / filename
    path.write_bytes(zlib.compress(preamble + payload))
    return path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw_data"
    d.mkdir()
    return d


def _stored_timestamps(db_path: Path):
    with DatabaseManager(db_path) as m:
        cur = m.conn.cursor()
        cur.execute(
            "SELECT timestamp FROM events WHERE event_code != -1 ORDER BY timestamp"
        )
        return [r[0] for r in cur.fetchall()]


class TestHeaderOffsetAtIngestion:

    def test_stored_events_carry_the_header_offset(self, empty_db, raw_dir):
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.7"
        )
        IngestionEngine(empty_db, raw_dir, timezone="US/Mountain").run()

        base = _boundary_epoch(2026, 6, 20, 4, 0)
        assert _stored_timestamps(empty_db) == pytest.approx([base + 0.7, base + 10.7])

    def test_span_stays_anchored_to_the_filename_boundary(self, empty_db, raw_dir):
        # utc_start is the grid anchor for ingestion_log, duration inference
        # and cycle ranges — only the event base moves.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.7"
        )
        IngestionEngine(empty_db, raw_dir, timezone="US/Mountain").run()

        with DatabaseManager(empty_db) as m:
            cur = m.conn.cursor()
            cur.execute("SELECT span_start, span_end FROM ingestion_log")
            span_start, span_end = cur.fetchone()

        base = _boundary_epoch(2026, 6, 20, 4, 0)
        assert span_start == base
        assert span_end == base

    def test_cross_file_spacing_reflects_each_files_own_offset(self, empty_db, raw_dir):
        # The bug this fixes: two files with different header offsets used to
        # collapse to the same base, corrupting any duration spanning the edge.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.1",
            offsets_deciseconds=(8990,),
        )
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0415.datZ", "6/20/2026,04:15:00.9",
            offsets_deciseconds=(0,),
        )
        IngestionEngine(empty_db, raw_dir, timezone="US/Mountain").run()

        first, second = _stored_timestamps(empty_db)
        # 900.0 boundary gap − 899.0 in-file span + (0.9 − 0.1) header delta
        assert second - first == pytest.approx(1.8)

    def test_file_without_header_falls_back_to_the_boundary(self, empty_db, raw_dir):
        path = raw_dir / "ECON_10.0.0.1_2026_06_20_0400.datZ"
        payload = struct.pack(">BBH", 1, 2, 0)
        path.write_bytes(zlib.compress(b"Phases in use:,1,2\n" + payload))

        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        assert _stored_timestamps(empty_db) == [_boundary_epoch(2026, 6, 20, 4, 0)]
        assert engine.get_ingestion_stats()["header_mismatches"] == 0


class TestHeaderAlignmentWarning:

    def test_matching_header_reports_no_mismatch(self, empty_db, raw_dir):
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.7"
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        assert engine.get_ingestion_stats()["header_mismatches"] == 0

    def test_off_grid_header_is_counted_and_warned(self, empty_db, raw_dir, capsys):
        # Header claims 04:07 but the filename says 04:00 — the file does not
        # start on the boundary its name advertises.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:07:00.3"
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        assert engine.get_ingestion_stats()["header_mismatches"] == 1
        assert "does not match" in capsys.readouterr().out

    def test_mismatched_date_is_counted(self, empty_db, raw_dir):
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/19/2026,04:00:00.3"
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        assert engine.get_ingestion_stats()["header_mismatches"] == 1
