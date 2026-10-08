# Tests for cycle derivation on ACHD databases (imperative shell).
#
# The detection maths is covered by tests/analysis/test_cycles_barrier_styles.py.
# Here: CycleProcessor keeps the final cycle when the event that closes it is
# the last event in the DB, and 'atspm ingest-achd' imports int_cfg.csv and
# derives cycles (or skips them with --no-cycles).

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytz

from atspm import cli
from atspm.data import import_config
from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor

from ..conftest import seed_events

TZ = pytz.timezone("US/Mountain")
BASE = TZ.localize(datetime(2025, 8, 1, 8, 0)).timestamp()

INT_CFG = (
    ",,1/1/2020\n"
    'RB:,R1,"1,2|3,4"\n'
    'RB:,R2,"5,6|7,8"\n'
    "RB:,B,\n"
)


def _econolite_events(n_cycles, cycle=100.0, b_green=46.0):
    """Econolite-style greens and barrier pulses, ending on the closing 31:1."""
    rows = [(BASE - cycle + b_green, 1, 4), (BASE - cycle + b_green, 1, 8),
            (BASE - cycle + b_green, 31, 1)]
    for k in range(n_cycles):
        t = BASE + k * cycle
        rows += [(t, 1, 2), (t, 1, 6), (t, 31, 2), (t + 10, 82, 5)]
        rows += [(t + b_green, 1, 4), (t + b_green, 1, 8), (t + b_green, 31, 1)]
    return rows


def _cycle_starts(db_path):
    with DatabaseManager(db_path) as m:
        cur = m.conn.cursor()
        cur.execute("SELECT cycle_start FROM cycles ORDER BY cycle_start")
        return [r[0] for r in cur.fetchall()]


def test_final_cycle_closed_by_last_event_is_kept(empty_db: Path, tmp_path: Path):
    """The 31:1 that confirms the last wrap is the DB's final event.

    Regression: the open tail ended *at* the last timestamp and segments are
    half-open, so that pulse was dropped and the final cycle went missing.
    """
    cfg = tmp_path / "int_cfg.csv"
    cfg.write_text(INT_CFG)
    import_config(cfg, empty_db)
    rows = _econolite_events(3)
    seed_events(empty_db, rows)

    CycleProcessor(empty_db, "US/Mountain").process_span(rows[0][0], rows[-1][0])

    assert _cycle_starts(empty_db) == [BASE, BASE + 100.0, BASE + 200.0]


def _write_achd_csv(raw_dir: Path, rows):
    raw_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%A, %d %B %Y %H:%M:%S"
    day = datetime(2025, 8, 1)
    lines = [
        "Signal,271 - Eagle Rd Ustick,",
        f'Start time,"{day.strftime(fmt)}",',
        f'End time,"{(day + timedelta(days=15) - timedelta(seconds=1)).strftime(fmt)}",',
        f'Total Events,"{len(rows)}",',
        "Event Time, Event Code, Event Description, Event Parameter,",
    ]
    for ts, code, param in sorted(rows):
        local = datetime.fromtimestamp(ts, TZ)
        lines.append(f"{local.strftime('%m/%d/%y %H:%M:%S.%f')[:-3]},{code},Event,{param},")
    (raw_dir / "271_Events_20250801T0000.csv").write_text("\n".join(lines) + "\n")


def _run_ingest(monkeypatch, tmp_path, *extra):
    raw = tmp_path / "raw"
    _write_achd_csv(raw, _econolite_events(3))
    inter = tmp_path / "intersections"
    (inter / "achd" / "271").mkdir(parents=True)
    (inter / "achd" / "271" / "int_cfg.csv").write_text(INT_CFG)
    monkeypatch.setattr(cli, "_get_intersections_dir", lambda: inter)
    monkeypatch.setattr(
        sys, "argv",
        ["atspm", "ingest-achd", "--targetid", "271", "--source", str(raw), *extra],
    )
    cli.main()
    return inter / "achd" / "271" / "271_data.db"


def test_ingest_achd_derives_cycles(monkeypatch, tmp_path):
    db = _run_ingest(monkeypatch, tmp_path)

    assert _cycle_starts(db) == [BASE, BASE + 100.0, BASE + 200.0]


def test_ingest_achd_no_cycles_flag(monkeypatch, tmp_path):
    db = _run_ingest(monkeypatch, tmp_path, "--no-cycles")

    with DatabaseManager(db) as m:
        cur = m.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE name = 'cycles'")
        has_table = cur.fetchone() is not None
    assert not has_table or _cycle_starts(db) == []
