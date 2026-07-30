# Tests for backward controller-clock-step detection at the ingestion
# boundary (imperative shell).
#
# A backward clock set makes recorded time decrease part-way through a file:
# the binary offsets replay a band, so events from two different real moments
# carry the same labels and land out of sequence once sorted. The shell
# contract asserted here: the step is detected in file order (before any
# sort), a hard-reset marker (event_code = -1) is inserted just below the
# first post-step event so duration/pairing logic stops at the break
# (CLAUDE.md §5), the count is surfaced separately from comms-gap markers,
# and normal chronological files are never flagged.

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
        raw_dir:             Directory to write into.
        filename:            ``*_YYYY_MM_DD_HHMM.datZ`` name.
        clock:               ``<M/D/YYYY>,<HH:MM:SS.s>`` for the
                             ``Controller Data Log Beginning`` line.
        offsets_deciseconds: Binary time offsets to emit, one event each, in
                             payload order. The format takes any uint16, so a
                             decreasing sequence reproduces a backward set.
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


def _markers(db_path: Path):
    with DatabaseManager(db_path) as m:
        cur = m.conn.cursor()
        cur.execute(
            "SELECT timestamp FROM events WHERE event_code = -1 ORDER BY timestamp"
        )
        return [r[0] for r in cur.fetchall()]


def _events(db_path: Path):
    with DatabaseManager(db_path) as m:
        cur = m.conn.cursor()
        cur.execute(
            "SELECT timestamp FROM events WHERE event_code != -1 ORDER BY timestamp"
        )
        return [r[0] for r in cur.fetchall()]


class TestBackwardClockStepWithinAFile:

    def test_step_inserts_a_marker_below_the_first_post_step_event(
        self, empty_db, raw_dir
    ):
        # Mirrors the hardware case: a -5 s set on 10.70.10.51 produced
        # offsets 310 -> 262, replaying a 4.8 s band.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(0, 200, 310, 262, 300),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        base = _boundary_epoch(2026, 6, 20, 4, 0)
        # Just below the first post-step event (26.2 s), never on a real one.
        assert _markers(empty_db) == pytest.approx([base + 26.15])
        assert engine.get_ingestion_stats()["clock_steps"] == 1

    def test_marker_sorts_between_the_pre_step_band_and_the_replay(
        self, empty_db, raw_dir
    ):
        # Sorted order is what every consumer sees; the marker must land after
        # every event that is unambiguously pre-step and before the replay.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(0, 200, 310, 262, 300),
        )
        IngestionEngine(empty_db, raw_dir, timezone="US/Mountain").run()

        base = _boundary_epoch(2026, 6, 20, 4, 0)
        marker = _markers(empty_db)[0]
        before = [t for t in _events(empty_db) if t < marker]
        after = [t for t in _events(empty_db) if t > marker]

        assert before == pytest.approx([base + 0.0, base + 20.0])
        assert after == pytest.approx([base + 26.2, base + 30.0, base + 31.0])

    def test_multiple_steps_each_get_their_own_marker(self, empty_db, raw_dir):
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(100, 300, 200, 400, 250),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        base = _boundary_epoch(2026, 6, 20, 4, 0)
        assert _markers(empty_db) == pytest.approx([base + 19.95, base + 24.95])
        assert engine.get_ingestion_stats()["clock_steps"] == 2

    def test_step_is_warned_about(self, empty_db, raw_dir, capsys):
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(0, 310, 262),
        )
        IngestionEngine(empty_db, raw_dir, timezone="US/Mountain").run()

        out = capsys.readouterr().out
        assert "backward clock step" in out
        assert "4.8 s" in out

    def test_events_are_still_ingested(self, empty_db, raw_dir):
        # Fencing must not drop data — only mark the discontinuity. The one
        # loss is to UNIQUE(timestamp, event_code, parameter), not to us.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(0, 200, 310, 262, 300),
        )
        IngestionEngine(empty_db, raw_dir, timezone="US/Mountain").run()

        assert len(_events(empty_db)) == 5


class TestBackwardClockStepAcrossAFileBoundary:

    def test_break_across_a_file_edge_is_detected(self, empty_db, raw_dir):
        # The preceding file closes after this one opens, so the two files'
        # events interleave out of sequence. Within a file the offsets never
        # reach the next boundary, so this signature means the previous file
        # was itself anomalous — an off-grid header or a set that landed on the
        # edge — and the same pairing damage follows either way.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(0, 700),
        )
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0401.datZ", "6/20/2026,04:01:00.0",
            offsets_deciseconds=(0, 100),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        base = _boundary_epoch(2026, 6, 20, 4, 0)
        # First file runs to 04:01:10; the second opens ten seconds earlier.
        assert engine.get_ingestion_stats()["clock_steps"] == 1
        assert _markers(empty_db) == pytest.approx([base + 59.95])

    def test_contiguous_files_are_not_flagged(self, empty_db, raw_dir):
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(0, 500),
        )
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0401.datZ", "6/20/2026,04:01:00.0",
            offsets_deciseconds=(0, 500),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        assert engine.get_ingestion_stats()["clock_steps"] == 0
        assert _markers(empty_db) == []

    def test_gap_fill_backfill_is_not_mistaken_for_a_step(self, empty_db, raw_dir):
        # In Gap Fill the previously *processed* file can sit later on the
        # clock than the one being filled. That is ordinary backfill, not a
        # clock set, and must not raise a marker.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(0, 500),
        )
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0500.datZ", "6/20/2026,05:00:00.0",
            offsets_deciseconds=(0, 500),
        )
        IngestionEngine(empty_db, raw_dir, timezone="US/Mountain").run()

        # Now a file lands in the hole between the two ingested spans.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0430.datZ", "6/20/2026,04:30:00.0",
            offsets_deciseconds=(0, 500),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run(fill_gaps=True)

        assert engine.get_ingestion_stats()["clock_steps"] == 0


class TestNormalOperationIsUnaffected:

    def test_chronological_file_raises_nothing(self, empty_db, raw_dir):
        # Zero backward steps were found in all 14,124 corpus files of normal
        # operation — the detector must stay silent on ordinary data.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.7",
            offsets_deciseconds=(0, 100, 100, 250, 8990),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        assert engine.get_ingestion_stats()["clock_steps"] == 0
        assert _markers(empty_db) == []

    def test_single_event_file_raises_nothing(self, empty_db, raw_dir):
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(300,),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        assert engine.get_ingestion_stats()["clock_steps"] == 0

    def test_empty_file_raises_nothing(self, empty_db, raw_dir):
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        assert engine.get_ingestion_stats()["clock_steps"] == 0

    def test_comms_gap_marker_is_not_counted_as_a_clock_step(
        self, empty_db, raw_dir
    ):
        # A missing hour between two files opens an ordinary gap marker; it
        # must land in gap_markers only, leaving clock_steps at zero.
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0400.datZ", "6/20/2026,04:00:00.0",
            offsets_deciseconds=(0, 500),
        )
        _write_datz(
            raw_dir, "ECON_10.0.0.1_2026_06_20_0500.datZ", "6/20/2026,05:00:00.0",
            offsets_deciseconds=(0, 500),
        )
        engine = IngestionEngine(empty_db, raw_dir, timezone="US/Mountain")
        engine.run()

        stats = engine.get_ingestion_stats()
        assert stats["clock_steps"] == 0
        assert stats["gap_markers"] == 1
