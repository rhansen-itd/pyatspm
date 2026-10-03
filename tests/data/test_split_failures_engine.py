# Acceptance tests for the split-failure shell engine, plot and CLI
# (UDOT S-M1; spec docs/specs/split_failures_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The measure itself is pinned in tests/analysis/test_split_failures.py.

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.split_failures import bin_split_failures, split_failures
from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor
from atspm.data.reader import get_events_with_cycles_df
from atspm.data.split_failures import (
    _ALL_SF_CODES,
    SplitFailureEngine,
    get_split_failures,
)
from atspm.plotting.split_failures import plot_split_failures
from atspm.utils.timezone import to_epoch

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 09:00"
STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection: one ring 2|4, 100 s cycles for two hours.
# P2 (lanes 1, 2, 3): every 3rd cycle lanes 1 and 2 are held on from before
# green until after red-5 (a split failure under union and under mean of 1/2
# only when lane 3 is excluded); lane 3 logs only short pulses.  P4 (lane 5)
# gets a few short pulses.  Lane 9 is configured for P8 but P8 never runs.
# ---------------------------------------------------------------------------

_CYCLE = 100.0
_PLAN = {2: (0.0, 54.0), 4: (60.0, 34.0)}


def _build_db(root: Path, key: str = "Stop_Bar") -> Path:
    t0 = to_epoch(datetime.strptime(START, "%Y-%m-%d %H:%M"), TZ)
    n_cycles = int(2 * 3600 / _CYCLE)
    rng = np.random.default_rng(3)
    events = []
    for c in range(n_cycles):
        base = t0 + c * _CYCLE
        for ph, (off, g) in _PLAN.items():
            gs = base + off
            events += [(gs, 1, ph), (gs + g, 8, ph), (gs + g + 4, 9, ph),
                       (gs + g + 4, 10, ph), (gs + g + 6, 11, ph),
                       (gs + g + 6, 12, ph)]
        if c % 3 == 0:
            for d in (1, 2):
                events += [(base - 3.0, 82, d), (base + 64.0, 81, d)]
        else:
            for d in (1, 2):
                for t in rng.uniform(2, 50, size=3):
                    events += [(base + t, 82, d), (base + t + 0.5, 81, d)]
        for t in (10.0, 30.0):
            events += [(base + t, 82, 3), (base + t + 0.4, 81, 3)]
        events += [(base + 70.0, 82, 5), (base + 71.5, 81, 5)]
    db = root / "sf.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        cfg = {"RB_R1": "2|4",
               f"Det_P2_{key}": "1,2,3", f"Det_P4_{key}": "5", f"Det_P8_{key}": "9"}
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00",
                              "end_date": None, **cfg})
        m.insert_events(sorted(events))
        m.conn.execute(
            "INSERT INTO ingestion_log (span_start, span_end, processed_at, "
            "row_count) VALUES (?, ?, '2025-06-02T00:00:00', ?)",
            (t0, t0 + n_cycles * _CYCLE, len(events)),
        )
        m.conn.commit()
    CycleProcessor(db, TZ).run()
    return db


@pytest.fixture
def db(tmp_path) -> Path:
    return _build_db(tmp_path)


def _core(db: Path, phase: int, dets, **kw):
    start = datetime.strptime(START, "%Y-%m-%d %H:%M")
    end = datetime.strptime(END, "%Y-%m-%d %H:%M")
    events = get_events_with_cycles_df(db, start, end, event_codes=_ALL_SF_CODES, timezone=TZ)
    return split_failures(events, phase, dets, **kw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_codes_include_gap_marker_and_detector_codes(self):
        assert set(_ALL_SF_CODES) == {-1, 1, 8, 9, 10, 11, 12, 81, 82}

    @pytest.mark.parametrize("aggregate", ["union", "mean"])
    def test_reproduces_the_core(self, db, aggregate):
        res = SplitFailureEngine(db).split_failures(START, END, aggregate=aggregate)
        cyc = res["cycle"]
        exp, exp_lanes = _core(db, 2, [1, 2, 3], aggregate=aggregate)
        got = cyc.loc[cyc["phase"] == 2].reset_index(drop=True)
        assert len(got) == len(exp)
        np.testing.assert_allclose(got["gor"].to_numpy(float), exp["gor"].to_numpy(float))
        np.testing.assert_allclose(got["ror5"].to_numpy(float), exp["ror5"].to_numpy(float))
        assert got["fail"].tolist() == exp["fail"].tolist()
        lanes = res["lane"]
        assert len(lanes.loc[lanes["phase"] == 2]) == len(exp_lanes)

    def test_result_keys_and_phases(self, db):
        res = SplitFailureEngine(db).split_failures(START, END)
        assert set(res) == {"cycle", "lane", "binned"}
        assert set(res["cycle"]["phase"]) == {2, 4}
        assert "aggregate" in res["cycle"].columns
        assert set(res["cycle"]["aggregate"]) == {"union"}

    def test_union_fails_more_than_mean_here(self, db):
        u = SplitFailureEngine(db).split_failures(START, END, aggregate="union")["cycle"]
        m = SplitFailureEngine(db).split_failures(START, END, aggregate="mean")["cycle"]
        u2, m2 = u.loc[u["phase"] == 2], m.loc[m["phase"] == 2]
        # lanes 1+2 held on, lane 3 empty: union 1.0, mean 2/3 -> 0 fails at 0.79
        assert int(u2["fail"].sum()) == 24
        assert int(m2["fail"].sum()) == 0
        mean_lo, _ = _core(db, 2, [1, 2, 3], aggregate="mean", threshold=0.6)
        assert int(mean_lo["fail"].sum()) == 24

    def test_threshold_and_options_pass_through(self, db):
        res = SplitFailureEngine(db).split_failures(
            START, END, phases=[2], threshold=0.5, ror_seconds=3.0,
            include_yellow=True, aggregate="mean")
        exp, _ = _core(db, 2, [1, 2, 3], threshold=0.5, ror_seconds=3.0,
                       include_yellow=True, aggregate="mean")
        got = res["cycle"].reset_index(drop=True)
        assert set(got["phase"]) == {2}
        np.testing.assert_allclose(got["r_dur"].to_numpy(float), exp["r_dur"].to_numpy(float))
        assert got["fail"].tolist() == exp["fail"].tolist()

    def test_binned_matches_core_binning(self, db):
        res = SplitFailureEngine(db).split_failures(START, END, phases=[2], bin_len=60)
        exp, _ = _core(db, 2, [1, 2, 3])
        exp_b = bin_split_failures(exp, bin_len=60)
        b = res["binned"]
        assert {"coverage", "data_quality"} <= set(b.columns)
        np.testing.assert_allclose(b["sf_pct"].to_numpy(float), exp_b["sf_pct"].to_numpy(float))
        np.testing.assert_allclose(b["gor"].to_numpy(float), exp_b["gor"].to_numpy(float))

    def test_cycle_mode_has_no_binned(self, db):
        res = SplitFailureEngine(db).split_failures(START, END, bin_len="cycle")
        assert set(res) == {"cycle", "lane"}

    def test_stopbar_spelling_also_accepted(self, tmp_path):
        db = _build_db(tmp_path, key="Stopbar")
        res = SplitFailureEngine(db).split_failures(START, END)
        assert set(res["cycle"]["phase"]) == {2, 4}

    def test_configured_phase_with_no_cycles_is_warned(self, db, capsys):
        SplitFailureEngine(db).split_failures(START, END)
        assert "Ph8" in capsys.readouterr().out

    def test_requested_phase_without_config_is_warned(self, db, capsys):
        res = SplitFailureEngine(db).split_failures(START, END, phases=[2, 6])
        assert "Det_P6_Stop_Bar" in capsys.readouterr().out
        assert set(res["cycle"]["phase"]) == {2}

    def test_no_stopbar_config_returns_empty(self, tmp_path, capsys):
        db = _build_db(tmp_path, key="Arrival")
        assert SplitFailureEngine(db).split_failures(START, END) == {}
        assert "Stop_Bar" in capsys.readouterr().out

    def test_bad_aggregate_raises(self, db):
        with pytest.raises(ValueError):
            SplitFailureEngine(db).split_failures(START, END, aggregate="max")

    def test_date_only_end_is_whole_day(self, db):
        res = SplitFailureEngine(db).split_failures(DAY, DAY, phases=[2])
        assert len(res["cycle"]) == 72

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert SplitFailureEngine(db).split_failures(
            START, END, aggregate="mean", output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {
            f"SF_Cycle_{STAMP}_mean.csv",
            f"SF_Lane_{STAMP}.csv",
            f"SF_60min_{STAMP}_mean.csv",
            f"SF_Scatter_{STAMP}_mean.html",
        }
        cyc = pd.read_csv(out / f"SF_Cycle_{STAMP}_mean.csv")
        assert {"gor_union", "gor_mean", "n_lanes", "fail"} <= set(cyc.columns)

    def test_no_plot_writes_no_html(self, db, tmp_path):
        out = tmp_path / "out"
        SplitFailureEngine(db).split_failures(START, END, make_plot=False, output_dir=out)
        assert not list(out.glob("*.html"))

    def test_convenience_wrapper(self, db):
        res = get_split_failures(db, START, END, phases=[4], timezone=TZ)
        assert set(res["cycle"]["phase"]) == {4}


# ---------------------------------------------------------------------------
# Plot (functional core: pure)
# ---------------------------------------------------------------------------


class TestPlot:

    @pytest.fixture
    def cycle_df(self, db):
        return SplitFailureEngine(db).split_failures(START, END)["cycle"]

    def test_returns_figure_with_pass_and_fail_traces(self, cycle_df):
        fig = plot_split_failures(cycle_df, metadata={"intersection_name": "X"})
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "Ph2 fail" in names and "Ph2 pass" in names and "Ph4 pass" in names
        fail = next(t for t in fig.data if t.name == "Ph2 fail")
        assert len(fail.x) == 24
        assert min(fail.x) > 0.79 and min(fail.y) > 0.79

    def test_points_are_gor_vs_ror5(self, cycle_df):
        fig = plot_split_failures(cycle_df)
        ph2 = cycle_df.loc[cycle_df["phase"] == 2]
        xs = np.concatenate([np.asarray(t.x, float) for t in fig.data
                             if t.name in ("Ph2 pass", "Ph2 fail")])
        assert sorted(xs) == pytest.approx(sorted(ph2["gor"].astype(float)))

    def test_threshold_lines_are_a_trace_not_shapes(self, cycle_df):
        fig = plot_split_failures(cycle_df, threshold=0.6)
        thr = [t for t in fig.data if t.name == "Threshold"]
        assert thr
        assert 0.6 in [v for v in thr[0].x if v is not None]
        assert not fig.layout.shapes

    def test_title_uses_metadata_and_aggregate(self, cycle_df):
        fig = plot_split_failures(
            cycle_df.assign(aggregate="mean"),
            metadata={"intersection_name": "X", "major_road_name": "Main St",
                      "minor_road_name": "Side St"})
        assert "Main St" in fig.layout.title.text
        assert "mean" in fig.layout.title.text

    def test_empty_frame_gives_figure(self):
        assert isinstance(plot_split_failures(pd.DataFrame()), go.Figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    @staticmethod
    def _parse(*argv):
        return cli._build_parser().parse_args(["split-failures", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "900", "--start", START, "--end", END)
        assert args.func is cli.handle_split_failures
        assert args.aggregate == "union"
        assert args.threshold == 0.79
        assert args.ror_seconds == 5.0
        assert args.include_yellow is False
        assert args.bin_len == "60"
        assert args.no_plot is False
        assert args.phases is None

    def test_aggregate_choices(self):
        assert self._parse("--all", "--start", DAY, "--end", DAY,
                           "--aggregate", "mean").aggregate == "mean"
        with pytest.raises(SystemExit):
            self._parse("--all", "--start", DAY, "--end", DAY, "--aggregate", "max")

    def test_target_group_is_required_and_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse("--start", DAY, "--end", DAY)
        with pytest.raises(SystemExit):
            self._parse("--all", "--targetid", "900", "--start", DAY, "--end", DAY)

    def test_end_to_end(self, tmp_path, monkeypatch):
        folder = tmp_path / "intersections" / "900_Synthetic"
        folder.mkdir(parents=True)
        _build_db(folder)
        (folder / "metadata.json").write_text(json.dumps({
            "intersection_name": "Synthetic", "intersection_id": "900",
            "db_filename": "sf.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--aggregate", "mean", "--threshold", "0.6", "--no-plot")
        args.func(args)
        cyc = pd.read_csv(folder / "outputs" / f"SF_Cycle_{STAMP}_mean.csv")
        assert int(cyc.loc[cyc["phase"] == 2, "fail"].sum()) == 24
        assert not list((folder / "outputs").glob("*.html"))
