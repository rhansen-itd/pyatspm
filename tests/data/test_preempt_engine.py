"""Acceptance gate for the S-M6 PreemptEngine and `atspm preempt` CLI.

Opus-written; the shell is built by the delegated run.  The core
(``analysis/preempt.py``) has its own goldens.  These assert the shell
contract: date-range fetch with a margin, CSV outputs with local-time and
timing-plot columns, and CLI parsing.
"""

import shlex
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import pytz

from atspm import cli
from atspm.data.manager import DatabaseManager
# Built by the delegated S-M6 run; this import fails until it exists.
from atspm.data.preempt import PreemptEngine

TZ = "US/Mountain"
_MT = pytz.timezone(TZ)


def _loc(s: str) -> float:
    return _MT.localize(pd.Timestamp(s).to_pydatetime(), is_dst=None).timestamp()


@pytest.fixture
def seeded_db(empty_db: Path) -> Path:
    from ..conftest import seed_events

    t = _loc("2026-01-10 08:00:00.5")
    u = _loc("2026-01-10 23:59:50")          # crosses midnight: needs the margin
    rows = [
        (t, 102, 5), (t, 105, 5), (t, 116, 5), (t + 6.2, 107, 5), (t + 36.3, 104, 5),
        (t + 36.3, 116, 5), (t + 41.4, 111, 5),
        (t + 600, 102, 4), (t + 604, 104, 4),                    # unserved
        (u, 102, 4), (u, 105, 4), (u + 5, 107, 4), (u + 30, 104, 4), (u + 35, 111, 4),
        (_loc("2026-01-12 09:00"), 102, 3), (_loc("2026-01-12 09:00"), 105, 3),
        (_loc("2026-01-12 09:00:05"), 107, 3), (_loc("2026-01-12 09:00:30"), 104, 3),
        (_loc("2026-01-12 09:00:35"), 111, 3),
    ]
    seed_events(empty_db, rows)
    with DatabaseManager(empty_db) as m:
        m.set_metadata(intersection_id="315", intersection_name="Scratch", timezone=TZ,
                       major_road_route="US-20", major_road_name="Chinden",
                       minor_road_route=None, minor_road_name="KCID Rd")
    return empty_db


class TestEngine:
    def test_episodes_for_range(self, seeded_db: Path):
        res = PreemptEngine(seeded_db, timezone=TZ).preempt("2026-01-10", "2026-01-10")
        ep = res["episodes"]
        # requests whose Call On falls on the local date range only
        assert ep["preempt"].tolist() == [5, 4, 4]
        assert ep["served"].tolist() == [True, False, True]
        # the 23:59:50 request completes after midnight: the margin keeps it whole
        last = ep.iloc[2]
        assert not last["censored"] and last["dwell_s"] == pytest.approx(30)
        s = res["summary"]
        assert set(zip(s["preempt"], s["requests"])) == {(5, 1), (4, 2)}

    def test_multi_day_and_empty(self, seeded_db: Path):
        eng = PreemptEngine(seeded_db, timezone=TZ)
        assert len(eng.preempt("2026-01-10", "2026-01-12")["episodes"]) == 4
        res = eng.preempt("2026-01-11", "2026-01-11")
        assert res["episodes"].empty and res["summary"].empty

    def test_outputs(self, seeded_db: Path, tmp_path: Path):
        out = tmp_path / "outputs"
        res = PreemptEngine(seeded_db, timezone=TZ).preempt(
            "2026-01-10", "2026-01-12", output_dir=out)
        names = sorted(p.name for p in out.iterdir())
        assert names == ["Preempt_Episodes_2026_01_10-2026_01_12.csv",
                         "Preempt_Summary_2026_01_10-2026_01_12.csv"]
        ep = pd.read_csv(out / names[0])
        # local wall-clock companion of call_on, ISO seconds
        assert ep["call_on_local"].iloc[0] == "2026-01-10T08:00:00.5"
        cmd = shlex.split(ep["timing_plot"].iloc[0])
        assert cmd[:2] == ["atspm", "plot-timing-actuation"]
        assert cmd[cmd.index("--targetid") + 1] == "315"
        start = datetime.fromisoformat(cmd[cmd.index("--start") + 1])
        end = datetime.fromisoformat(cmd[cmd.index("--end") + 1])
        assert start == datetime(2026, 1, 10, 7, 58)          # call_on - 2 min, floored to s
        assert end == datetime(2026, 1, 10, 8, 2, 41)         # exit + 2 min (41.9 floored)
        # an unserved request plots call_on .. call_off ± 2 min
        cmd = shlex.split(ep["timing_plot"].iloc[1])
        end = datetime.fromisoformat(cmd[cmd.index("--end") + 1])
        assert end == datetime(2026, 1, 10, 8, 12, 4)
        assert res["html"] is None or Path(res["html"]).exists()

    def test_no_output_dir_writes_nothing(self, seeded_db: Path, tmp_path: Path, monkeypatch):
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        PreemptEngine(seeded_db, timezone=TZ).preempt("2026-01-10", "2026-01-10")
        assert list(cwd.iterdir()) == []


class TestCli:
    def _parse(self, *argv):
        return cli._build_parser().parse_args(["preempt", *argv])

    def test_parses(self):
        a = self._parse("--targetid", "315", "--start", "2026-01-10", "--end", "2026-01-12")
        assert a.func is cli.handle_preempt and a.targetid == "315"

    def test_all_and_exclusive_group(self):
        assert self._parse("--all", "--start", "2026-01-10", "--end", "2026-01-10").all is True
        with pytest.raises(SystemExit):
            self._parse("--start", "2026-01-10", "--end", "2026-01-10")
        with pytest.raises(SystemExit):
            self._parse("--target", "x", "--targetid", "315", "--start", "2026-01-10",
                        "--end", "2026-01-10")
