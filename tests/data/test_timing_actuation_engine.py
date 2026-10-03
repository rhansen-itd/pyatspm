"""Acceptance gate for the S-D5 TimingActuationEngine, CLI and finding links.

Opus-written; the shell is built by the delegated run.  The functional core
(``analysis/timing_actuation.py``) and the figure
(``plotting/timing_actuation.py``) already exist and have their own goldens.
These assert the observable shell contract:

* the engine fetches with a margin, so a detector on since before the window
  fills it;
* rows come from the config in effect, filters pass through, the window cap
  holds (4 h, or 24 h when --phases / --detectors narrows the rows);
* stored findings are overlaid with WD_Ignore applied;
* the HTML is written under the expected name;
* ``atspm plot-timing-actuation`` parses like the other plot commands;
* detector-health's reported findings carry a ``timing_plot`` command.
"""

import shlex
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

from atspm import cli
from atspm.analysis.detector_health import FINDINGS_SCHEMA
from atspm.data.manager import DatabaseManager
# Built by the delegated S-D5 run; these imports fail until it exists.
from atspm.data.timing_actuation import TimingActuationEngine
from atspm.data.detector_health import DetectorHealthEngine

TZ = "US/Mountain"
_MT = pytz.timezone(TZ)
DATE = "2026-01-10"


def _loc(s: str) -> float:
    return _MT.localize(pd.Timestamp(s).to_pydatetime(), is_dst=None).timestamp()


def _cycles(start: str, end: str):
    """Phase 2 / 4 alternating 60 s cycles with calls, plus detector 21 pulses on P2 green."""
    rows = []
    t, t_end = _loc(start), _loc(end)
    while t < t_end:
        rows += [(t, 1, 2), (t + 25, 8, 2), (t + 29, 9, 2), (t + 29, 10, 2), (t + 30, 11, 2),
                 (t + 30, 1, 4), (t + 55, 8, 4), (t + 59, 9, 4), (t + 59, 10, 4), (t + 60, 11, 4),
                 (t + 2, 43, 4), (t + 31, 44, 4),
                 (t + 5, 82, 21), (t + 7, 81, 21), (t + 12, 82, 21), (t + 13, 81, 21)]
        t += 60
    return rows


@pytest.fixture
def seeded_db(empty_db: Path) -> Path:
    from ..conftest import seed_events

    # The whole local day, so detector-health judges it (min_observed_share).
    rows = _cycles(f"{DATE} 00:00", "2026-01-11 00:00")
    rows += [(_loc(f"{DATE} 07:50"), 82, 30)]        # stuck on from 07:50, never off
    seed_events(empty_db, rows)
    with DatabaseManager(empty_db) as m:
        m.set_metadata(intersection_id="201", intersection_name="Scratch",
                       timezone=TZ, major_road_route="US-1", major_road_name="Main",
                       minor_road_route="SR-2", minor_road_name="Cross")
        cols = {
            "RB_R1": "2|4",
            "Det_P2_Occupancy": "21",
            "Det_P2_Stop_Bar": "53",          # configured, silent
            "Det_P4_Occupancy": "30",
            "WD_Ignore": "21:StuckOn",
        }
        for c in cols:
            m.add_config_column(c)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00", "end_date": None, **cols})
        m.conn.commit()
    return empty_db


def _seed_findings(db: Path, rows):
    df = pd.DataFrame(rows, columns=FINDINGS_SCHEMA).astype(
        {"ts": "float64", "detector": "int64", "phase": "Int64"})
    with DatabaseManager(db) as m:
        m.replace_findings(df, DATE, DATE)


class TestEngine:
    def test_rows_from_config_and_html(self, seeded_db: Path, tmp_path: Path):
        out = tmp_path / "outputs"
        eng = TimingActuationEngine(seeded_db, timezone=TZ)
        res = eng.plot(f"{DATE} 08:00", f"{DATE} 08:30", output_dir=out)
        rows = res["rows"]
        assert rows.loc[rows["kind"] == "phase", "param"].tolist() == [2, 4]
        labels = rows["label"].tolist()
        assert {"Occ 21", "Stop 53", "Occ 30"} <= set(labels)      # silent 53 still a row
        html = res["html"]
        assert html is not None and html.exists()
        assert html.parent == out
        assert html.name == "TimingActuation_2026_01_10_0800-2026_01_10_0830.html"
        assert "Timing & Actuation" in res["figure"].layout.title.text

    def test_no_output_dir_writes_nothing(self, seeded_db: Path, tmp_path: Path, monkeypatch):
        cwd = tmp_path / "cwd"          # tmp_path itself holds the DB
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        res = TimingActuationEngine(seeded_db, timezone=TZ).plot(f"{DATE} 08:00", f"{DATE} 08:30")
        assert res["html"] is None
        assert list(cwd.iterdir()) == []

    def test_stuck_on_since_before_window_fills_it(self, seeded_db: Path):
        res = TimingActuationEngine(seeded_db, timezone=TZ).plot(f"{DATE} 08:00", f"{DATE} 08:30")
        iv = res["intervals"]
        d30 = iv.loc[(iv["kind"] == "detector") & (iv["param"] == 30)]
        assert len(d30) == 1
        np.testing.assert_allclose(d30[["start_ts", "end_ts"]].iloc[0],
                                   [_loc(f"{DATE} 08:00"), _loc(f"{DATE} 08:30")])

    def test_filters_pass_through_and_suffix_name(self, seeded_db: Path, tmp_path: Path):
        eng = TimingActuationEngine(seeded_db, timezone=TZ)
        res = eng.plot(f"{DATE} 08:00", f"{DATE} 08:30", phases=[4], output_dir=tmp_path)
        assert set(res["rows"]["block"]) == {"P4"}
        assert res["html"].name == "TimingActuation_2026_01_10_0800-2026_01_10_0830_P4.html"
        res = eng.plot(f"{DATE} 08:00", f"{DATE} 08:30", detectors=[53], output_dir=tmp_path)
        assert res["rows"].loc[res["rows"]["kind"] == "detector", "param"].tolist() == [53]
        assert res["html"].name == "TimingActuation_2026_01_10_0800-2026_01_10_0830_D53.html"

    def test_window_cap(self, seeded_db: Path):
        eng = TimingActuationEngine(seeded_db, timezone=TZ)
        with pytest.raises(ValueError, match="4 h"):
            eng.plot(f"{DATE} 06:00", f"{DATE} 10:30")
        eng.plot(f"{DATE} 06:00", f"{DATE} 10:00")                  # exactly 4 h is fine
        eng.plot(f"{DATE} 06:00", f"{DATE} 16:00", phases=[2])      # narrowed: up to 24 h
        with pytest.raises(ValueError, match="24 h"):
            eng.plot(f"{DATE} 06:00", "2026-01-11 07:00", detectors=[21])
        with pytest.raises(ValueError):
            eng.plot(f"{DATE} 09:00", f"{DATE} 08:00")

    def test_findings_overlaid_with_ignore(self, seeded_db: Path):
        day = datetime.fromisoformat(DATE).date()
        _seed_findings(seeded_db, [
            [day, "day", _loc(f"{DATE} 08:10"), 30, 4, "occupancy", "StuckOn", "high", 1, 1, "stuck 30"],
            [day, "day", _loc(f"{DATE} 08:12"), 21, 2, "occupancy", "StuckOn", "high", 1, 1, "ignored"],
            [day, "day", _loc(f"{DATE} 08:14"), 21, 2, "occupancy", "Chatter", "info", 1, 1, "info"],
        ])
        res = TimingActuationEngine(seeded_db, timezone=TZ).plot(f"{DATE} 08:00", f"{DATE} 08:30")
        tr = [t for t in res["figure"].data if t.name == "Finding"]
        assert len(tr) == 1
        text = " ".join(tr[0].hovertext)
        assert "stuck 30" in text
        assert "ignored" not in text        # WD_Ignore 21:StuckOn
        assert "info" not in text           # info findings aren't overlaid


class TestCli:
    def _parse(self, *argv):
        return cli._build_parser().parse_args(["plot-timing-actuation", *argv])

    def test_parses(self):
        a = self._parse("--targetid", "201", "--start", "2026-01-10T08:00:00",
                        "--end", "2026-01-10T08:30:00", "--phases", "2", "6",
                        "--detectors", "53")
        assert a.func is cli.handle_plot_timing_actuation
        assert a.targetid == "201" and a.phases == [2, 6] and a.detectors == [53]

    def test_all_and_exclusive_group(self):
        assert self._parse("--all", "--start", "2026-01-10T08:00",
                           "--end", "2026-01-10T08:30").all is True
        with pytest.raises(SystemExit):
            self._parse("--start", "2026-01-10T08:00", "--end", "2026-01-10T08:30")
        with pytest.raises(SystemExit):
            self._parse("--target", "x", "--targetid", "201",
                        "--start", "2026-01-10T08:00", "--end", "2026-01-10T08:30")

    def test_start_end_required(self):
        with pytest.raises(SystemExit):
            self._parse("--targetid", "201", "--start", "2026-01-10T08:00")


class TestFindingLinks:
    def test_reported_findings_carry_timing_plot(self, seeded_db: Path, tmp_path: Path):
        res = DetectorHealthEngine(seeded_db, timezone=TZ).detector_health(
            DATE, DATE, window="day", min_severity="info", output_dir=tmp_path)
        rep = res["reported"]
        assert "timing_plot" in rep.columns
        silent = rep.loc[(rep["detector"] == 53) & (rep["rule"] == "ConfiguredSilent")]
        assert len(silent) == 1
        cmd = shlex.split(silent["timing_plot"].iloc[0])
        assert cmd[:2] == ["atspm", "plot-timing-actuation"]
        assert cmd[cmd.index("--targetid") + 1] == "201"
        start = datetime.fromisoformat(cmd[cmd.index("--start") + 1])
        end = datetime.fromisoformat(cmd[cmd.index("--end") + 1])
        assert (end - start).total_seconds() == 3600       # busiest hour, day-level finding
        assert start.minute == 0 and start.second == 0
        assert ("--phases" in cmd and cmd[cmd.index("--phases") + 1] == "2") or (
            "--detectors" in cmd and cmd[cmd.index("--detectors") + 1] == "53")
        # the CSV carries the column too
        csv = next(p for p in tmp_path.iterdir() if p.suffix == ".csv")
        assert "timing_plot" in pd.read_csv(csv).columns

    def test_ts_findings_centre_on_ts(self, seeded_db: Path):
        res = DetectorHealthEngine(seeded_db, timezone=TZ).detector_health(
            DATE, DATE, window="day", min_severity="info")
        rep = res["reported"]
        with_ts = rep.loc[rep["ts"].notna() & (rep["timing_plot"] != "")]
        for ts, link in zip(with_ts["ts"], with_ts["timing_plot"]):
            cmd = shlex.split(link)
            start = _MT.localize(datetime.fromisoformat(cmd[cmd.index("--start") + 1])).timestamp()
            end = _MT.localize(datetime.fromisoformat(cmd[cmd.index("--end") + 1])).timestamp()
            assert abs((start + end) / 2 - ts) <= 1.0 and abs(end - start - 1200) <= 1.0

    def test_findings_table_schema_unchanged(self, seeded_db: Path):
        DetectorHealthEngine(seeded_db, timezone=TZ).detector_health(DATE, DATE, window="day")
        with DatabaseManager(seeded_db) as m:
            cols = [r[1] for r in m.conn.execute("PRAGMA table_info(detector_findings)")]
        assert "timing_plot" not in cols
