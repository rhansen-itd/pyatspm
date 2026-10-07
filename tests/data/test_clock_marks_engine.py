# Acceptance tests for the clock-mark shell, plot and CLI (S1).
#
# Opus-written; the implementation must make these pass without editing
# them. The decoding itself is pinned in tests/analysis/test_clock_marks.py.

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.data.clock_marks import ClockMarkEngine, get_clock_marks
from atspm.data.ingestion import IngestionEngine
from atspm.data.manager import DatabaseManager
from atspm.plotting.clock_marks import plot_clock_drift

BENCH = Path(__file__).resolve().parents[1] / "fixtures" / "clock_marks_bench_2026_09_30"
TZ = "US/Mountain"


def _bench_db(root: Path, clk: bool = True) -> Path:
    raw = root / "raw_data"
    raw.mkdir()
    for f in sorted(BENCH.glob("*.datZ")):
        (raw / f.name).write_bytes(f.read_bytes())
    db = root / "bench.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Bench", timezone=TZ)
        cfg = {"Clk_Behind": "15", "Clk_Ahead": "16", "Clk_Set": "14"} if clk else {}
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00", "end_date": None, **cfg})
        m.conn.commit()
    IngestionEngine(db, raw, timezone=TZ).run()
    return db


@pytest.fixture
def bench_db(tmp_path) -> Path:
    return _bench_db(tmp_path)


class TestEngine:

    def test_whole_day_decodes_all_sets_with_send_log(self, bench_db):
        res = ClockMarkEngine(bench_db).decode(
            "2026-09-30", "2026-09-30", send_log_path=BENCH / "bench.jsonl"
        )
        assert set(res) == {"drift", "sets"}
        assert res["sets"]["shift"].tolist() == [-2, 3, 40, -40]
        assert len(res["drift"]) == 9

    def test_without_send_log_saturated_sets_stay_flagged(self, bench_db):
        res = ClockMarkEngine(bench_db).decode("2026-09-30", "2026-09-30")
        assert res["sets"]["status"].tolist() == ["ok", "ok", "pre_saturated", "pre_saturated"]

    def test_window_keeps_pulses_whose_on_falls_inside(self, bench_db):
        # Local 13:04-13:05 holds the first two sets' brackets (ON 13:04:27,
        # 13:04:50); the pre-set pulse of the first sits at 13:04:25, so the
        # fetch margin must not be what decides membership.
        res = ClockMarkEngine(bench_db).decode("2026-09-30 13:04", "2026-09-30 13:05")
        assert res["sets"]["shift"].tolist() == [-2, 3]
        assert res["drift"]["role"].tolist() == ["check", "pre_set", "residual", "pre_set"]

    def test_window_start_does_not_orphan_a_bracket(self, bench_db):
        # Window opens between the first set's pre pulse (13:04:25.7) and its
        # bracket (13:04:27.5): the bracket still decodes from the margin.
        res = ClockMarkEngine(bench_db).decode("2026-09-30 13:04:27", "2026-09-30 13:05")
        assert res["sets"]["shift"].tolist() == [-2, 3]

    def test_times_stay_utc_epoch_floats(self, bench_db):
        res = ClockMarkEngine(bench_db).decode("2026-09-30", "2026-09-30")
        assert pd.api.types.is_float_dtype(res["sets"]["bracket_on"])
        assert pd.api.types.is_float_dtype(res["drift"]["ts"])

    def test_no_clk_config_returns_empty_dict(self, tmp_path, capsys):
        db = _bench_db(tmp_path, clk=False)
        assert ClockMarkEngine(db).decode("2026-09-30", "2026-09-30") == {}
        assert "Clk" in capsys.readouterr().out

    def test_output_dir_writes_csvs_and_html(self, bench_db, tmp_path):
        out = tmp_path / "outputs"
        assert get_clock_marks(
            bench_db, "2026-09-30", "2026-09-30",
            send_log_path=BENCH / "bench.jsonl", output_dir=out,
        ) is None
        names = sorted(p.name for p in out.iterdir())
        assert names == [
            "Clock_Drift_2026_09_30-2026_09_30.csv",
            "Clock_Drift_2026_09_30-2026_09_30.html",
            "Clock_Sets_2026_09_30-2026_09_30.csv",
        ]
        sets = pd.read_csv(out / "Clock_Sets_2026_09_30-2026_09_30.csv")
        assert sets["shift"].tolist() == [-2, 3, 40, -40]
        # CSV times are intersection-local wall clock.
        assert str(sets["bracket_on"].iloc[0]).startswith("2026-09-30 13:04:27")


class TestPlot:

    def test_pure_figure_from_core_frames(self, bench_db):
        res = ClockMarkEngine(bench_db).decode(
            "2026-09-30", "2026-09-30", send_log_path=BENCH / "bench.jsonl"
        )
        fig = plot_clock_drift(
            res["drift"], res["sets"],
            metadata={"intersection_name": "Bench"}, timezone=TZ,
        )
        assert isinstance(fig, go.Figure)
        drift_traces = [t for t in fig.data if t.name == "Drift"]
        assert len(drift_traces) == 1
        assert len(drift_traces[0].y) == res["drift"]["drift"].notna().sum()
        assert "Bench" in fig.layout.title.text
        assert not fig.layout.shapes  # sets drawn as traces, not layout shapes

    def test_empty_frames_still_make_a_figure(self):
        from atspm.analysis.clock_marks import decode_clock_marks, MarkerPhases
        empty = pd.DataFrame(columns=["timestamp", "event_code", "parameter"])
        drift, sets = decode_clock_marks(empty, MarkerPhases(15, 16, 14))
        fig = plot_clock_drift(drift, sets, metadata={}, timezone=TZ)
        assert isinstance(fig, go.Figure)


class TestCli:

    def _parse(self, *argv):
        return cli._build_parser().parse_args(["clock-drift", *argv])

    def test_parses_single_target_with_send_log(self):
        args = self._parse("--targetid", "201", "--start", "2026-09-30",
                           "--end", "2026-09-30", "--send-log", "x.jsonl")
        assert args.func is cli.handle_clock_drift
        assert args.targetid == "201" and args.send_log == "x.jsonl"

    def test_all_is_accepted(self):
        args = self._parse("--all", "--start", "2026-09-30", "--end", "2026-09-30")
        assert args.all is True

    def test_target_group_is_required_and_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse("--start", "2026-09-30", "--end", "2026-09-30")
        with pytest.raises(SystemExit):
            self._parse("--all", "--targetid", "201",
                        "--start", "2026-09-30", "--end", "2026-09-30")
