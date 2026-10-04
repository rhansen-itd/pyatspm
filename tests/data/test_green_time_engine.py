# Acceptance tests for the green time utilization shell engine, plot and CLI
# (UDOT S-M7; spec docs/specs/green_time_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The measure itself is pinned in tests/analysis/test_green_time_utilization.py.

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.green_time_utilization import (
    ACTUATION_SCHEMA,
    BIN_SCHEMA,
    CYCLE_SCHEMA,
    SPLIT_SCHEMA,
    green_time_utilization,
    summarize_gtu_bins,
    summarize_gtu_splits,
)
from atspm.analysis.split_monitor import plan_timeline
from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor
from atspm.utils.timezone import to_epoch

# Built by the delegated S-M7 run; these imports fail until they exist.
from atspm.data.green_time_utilization import (
    _GTU_CODES,
    _PLAN_CODES,
    PLAN_LOOKBACK_S,
    GreenTimeEngine,
    get_green_time,
)
from atspm.plotting.green_time_utilization import plot_green_time

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 09:00"
STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection: one ring 2|4, 100 s cycles from 06:50 to 09:10.
#   P2  green 0, yellow 54, end yellow 58 (9, 10), end red clr 60 (11):
#       green 54 s, clearance 6 s.
#       Stop-bar loop 26 @ +3, +3.5, +5, +20  → bins 1, 1, 2, 10 (2 s bins)
#       Presence zone 50 @ +1                 → bin 0 (Det_P2_Occupancy)
#   P4  green 60, yellow 94, end yellow 98, end red clr 100: green 34 s.
#       Stop-bar loop 31 @ +62, +70, +94 (yellow instant, not counted)
#                                              → bins 1, 5
#   Plans: a dump at 06:00 (plan 1, cycle 100, split 2 = 60, split 4 = 40),
#   then at 07:59:30 only plan 2 and split 2 = 50 (a change log).
#   Programmed green: P2 54 (plan 1) / 44 (plan 2); P4 34 throughout.
#   In 07:00–09:00: P2 greens 07:00:00 … 08:58:20 (72; 36 per plan),
#   P4 greens 07:01:00 … 08:59:20 (72; 36 per plan by their cycle).
# Variants: a comms gap at 08:30:20 (inside P2's 08:30:00 green), and a
# config with no Det_P{N}_Stop_Bar keys (as at 701).
# ---------------------------------------------------------------------------

_CYCLE = 100.0
_N = 84                                  # 06:50 → 09:10


def _local(hhmm: str) -> float:
    return to_epoch(datetime.strptime(f"{DAY} {hhmm}", "%Y-%m-%d %H:%M"), TZ)


def _plan_dump(t, plan, cycle, splits):
    rows = [(t, 131, plan), (t, 132, cycle), (t, 133, 0)]
    rows += [(t, 133 + p, splits.get(p, 0)) for p in range(1, 17)]
    return rows


def _build_db(root: Path, gap: bool = False, stop_bar: bool = True) -> Path:
    t0 = _local("06:50")
    events = _plan_dump(_local("06:00"), 1, 100, {2: 60, 4: 40})
    events += [(_local("07:59") + 30, 131, 2), (_local("07:59") + 30, 135, 50)]
    for c in range(_N):
        b = t0 + c * _CYCLE
        events += [(b, 1, 2), (b + 54, 8, 2), (b + 58, 9, 2), (b + 58, 10, 2), (b + 60, 11, 2)]
        events += [(b + 60, 1, 4), (b + 94, 8, 4), (b + 98, 9, 4), (b + 98, 10, 4),
                   (b + 100, 11, 4)]
        for t, d in [(3, 26), (3.5, 26), (5, 26), (20, 26), (1, 50), (62, 31), (70, 31), (94, 31)]:
            events += [(b + t, 82, d), (b + t + 0.3, 81, d)]
    if gap:
        events.append((_local("08:30") + 20.0, -1, -1))
    db = root / "gtu.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        cfg = {"RB_R1": "2|4", "Det_P2_Occupancy": "50"}
        if stop_bar:
            cfg.update({"Det_P2_Stop_Bar": "26", "Det_P4_Stop_Bar": "31"})
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00",
                              "end_date": None, **cfg})
        m.insert_events(sorted(events))
        m.conn.execute(
            "INSERT INTO ingestion_log (span_start, span_end, processed_at, "
            "row_count) VALUES (?, ?, '2025-06-02T00:00:00', ?)",
            (_local("06:00"), t0 + _N * _CYCLE, len(events)),
        )
        m.conn.commit()
    CycleProcessor(db, TZ).run()
    return db


@pytest.fixture
def db(tmp_path) -> Path:
    return _build_db(tmp_path)


def _ph(df, phase):
    return df.loc[df["phase"] == phase].reset_index(drop=True)


def _epoch(s):
    return np.array([t.timestamp() for t in s])


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_codes(self):
        assert set(_GTU_CODES) == {-1, 1, 8, 9, 10, 11, 12, 61, 63, 64, 65, 82}
        assert set(_PLAN_CODES) == {-1, *range(131, 150)}
        assert PLAN_LOOKBACK_S == 26 * 3600.0

    def test_result_keys_and_columns(self, db):
        res = GreenTimeEngine(db).green_time(START, END)
        assert set(res) == {"cycle", "actuations", "bins", "splits", "plan_bins", "plan_splits"}
        assert list(res["cycle"].columns) == CYCLE_SCHEMA
        assert list(res["actuations"].columns) == ACTUATION_SCHEMA
        assert list(res["bins"].columns) == BIN_SCHEMA
        assert list(res["plan_bins"].columns) == BIN_SCHEMA
        assert list(res["splits"].columns) == SPLIT_SCHEMA
        assert list(res["plan_splits"].columns) == SPLIT_SCHEMA
        assert set(res["cycle"]["phase"]) == {2, 4}

    def test_window_by_green(self, db):
        res = GreenTimeEngine(db).green_time(START, END)
        cy, ac = res["cycle"], res["actuations"]
        for ph in (2, 4):
            p = _ph(cy, ph)
            assert len(p) == 72 and not p["censored"].any()
            g = _epoch(p["green_ts"])
            assert (g >= _local("07:00")).all() and (g < _local("09:00")).all()
            ga = _epoch(_ph(ac, ph)["green_ts"])
            assert (ga >= _local("07:00")).all() and (ga < _local("09:00")).all()

    def test_stop_bar_role_and_bins(self, db):
        res = GreenTimeEngine(db).green_time(START, END)
        cy, ac = res["cycle"], res["actuations"]
        p2, p4 = _ph(cy, 2), _ph(cy, 4)
        assert (p2["green_dur"] == 54.0).all() and (p2["clearance_dur"] == 6.0).all()
        assert (p4["green_dur"] == 34.0).all()
        assert (p2["actuations"] == 4).all() and (p4["actuations"] == 2).all()
        assert set(_ph(ac, 2)["detector"]) == {26}
        first = _ph(ac, 2).loc[lambda d: d["green_ts"] == d["green_ts"].min(), "green_bin"]
        assert first.tolist() == [1, 1, 2, 10]
        first4 = _ph(ac, 4).loc[lambda d: d["green_ts"] == d["green_ts"].min(), "green_bin"]
        assert first4.tolist() == [1, 5]

    def test_occupancy_role(self, db):
        res = GreenTimeEngine(db).green_time(START, END, role="occupancy")
        cy, ac = res["cycle"], res["actuations"]
        assert set(cy["phase"]) == {2}
        assert (cy["actuations"] == 1).all()
        assert set(ac["detector"]) == {50} and (ac["green_bin"] == 0).all()

    def test_unknown_role_and_bad_bin_raise(self, db):
        with pytest.raises(ValueError):
            GreenTimeEngine(db).green_time(START, END, role="pairs")
        with pytest.raises(ValueError):
            GreenTimeEngine(db).green_time(START, END, bin_s=0)

    def test_programmed_split_from_plan_lookback(self, db):
        # The 06:00 dump is before the fetch margin: only the 26 h plan
        # lookback finds it.
        cy = GreenTimeEngine(db).green_time(START, END)["cycle"]
        p2, p4 = _ph(cy, 2), _ph(cy, 4)
        before = _epoch(p2["green_ts"]) < _local("07:59") + 30
        assert (p2.loc[before, "programmed_split"] == 60).all()
        assert (p2.loc[before, "programmed_green"] == 54.0).all()
        assert (p2.loc[~before, "programmed_split"] == 50).all()
        assert (p2.loc[~before, "programmed_green"] == 44.0).all()
        assert before.sum() == 36
        # Split 4 carried across the change log entry that did not touch it.
        assert (p4["programmed_split"] == 40).all() and (p4["programmed_green"] == 34.0).all()

    def test_plan_summaries(self, db):
        res = GreenTimeEngine(db).green_time(START, END)
        ps = res["plan_splits"]
        assert len(ps) == 4
        p2 = _ph(ps, 2).set_index("coord_plan")
        assert p2.loc[1.0, "n_cycles"] == 36 and p2.loc[2.0, "n_cycles"] == 36
        assert p2.loc[1.0, "avg_green_s"] == 54.0
        assert p2.loc[1.0, "programmed_green"] == 54.0 and p2.loc[2.0, "programmed_green"] == 44.0
        assert p2.loc[1.0, "act_per_cycle"] == 4.0
        pb = _ph(res["plan_bins"], 4)
        pb1 = pb.loc[pb["coord_plan"] == 1.0].set_index("green_bin")
        assert pb1.index.tolist() == list(range(17))          # 34 s green, 2 s bins
        assert pb1.loc[1, "act_per_cycle"] == 1.0 and pb1.loc[5, "act_per_cycle"] == 1.0
        assert pb1.loc[0, "actuations"] == 0
        assert (pb1["n_reached"] == 36).all() and (pb1["exposure_s"] == 72.0).all()

    def test_matches_the_core_on_the_same_events(self, db):
        from atspm.data.reader import get_events_with_cycles_df
        res = GreenTimeEngine(db).green_time(START, END, phases=[2])
        s = datetime.strptime(START, "%Y-%m-%d %H:%M")
        e = datetime.strptime(END, "%Y-%m-%d %H:%M")
        ev = get_events_with_cycles_df(db, s - timedelta(hours=26), e + timedelta(hours=1),
                                       event_codes=sorted(set(_GTU_CODES) | set(_PLAN_CODES)),
                                       timezone=TZ)
        core, acts = green_time_utilization(ev, 2, [26], timeline=plan_timeline(ev))
        keep = (core["green_ts"] >= pd.Timestamp(START, tz=TZ)) & \
            (core["green_ts"] < pd.Timestamp(END, tz=TZ))
        core = core.loc[keep].reset_index(drop=True)
        got = res["cycle"].reset_index(drop=True)
        assert len(got) == len(core)
        np.testing.assert_array_equal(got["actuations"].to_numpy(float),
                                      core["actuations"].to_numpy(float))
        np.testing.assert_array_equal(got["programmed_green"].to_numpy(float),
                                      core["programmed_green"].to_numpy(float))

    def test_bins_and_splits_time_binned(self, db):
        res = GreenTimeEngine(db).green_time(START, END, bin_len=30)
        sp = _ph(res["splits"], 2)
        assert len(sp) == 4 and sp["n_cycles"].sum() == 72
        assert (sp["time"].diff().dropna() == pd.Timedelta(minutes=30)).all()
        b = _ph(res["bins"], 2)
        assert b["time"].nunique() == 4 and b["green_bin"].max() == 26

    def test_bin_width(self, db):
        ac = GreenTimeEngine(db).green_time(START, END, phases=[2], bin_s=5.0)["actuations"]
        first = ac.loc[ac["green_ts"] == ac["green_ts"].min(), "green_bin"]
        assert first.tolist() == [0, 0, 1, 4]

    def test_gap_censors_the_green(self, tmp_path):
        db = _build_db(tmp_path, gap=True)
        res = GreenTimeEngine(db).green_time(START, END)
        p2 = _ph(res["cycle"], 2)
        cut = p2["green_ts"] == pd.Timestamp(f"{DAY} 08:30:00", tz=TZ)
        assert cut.sum() == 1 and p2.loc[cut, "censored"].all()
        assert p2.loc[cut, "actuations"].isna().all()
        assert (p2["censored"].sum(), len(p2)) == (1, 72)
        assert not (_ph(res["actuations"], 2)["green_ts"]
                    == pd.Timestamp(f"{DAY} 08:30:00", tz=TZ)).any()
        p2s = _ph(res["plan_splits"], 2).set_index("coord_plan")
        assert p2s.loc[2.0, "n_cycles"] == 35 and p2s.loc[2.0, "n_censored"] == 1

    def test_phases_filter_and_warning(self, db, capsys):
        res = GreenTimeEngine(db).green_time(START, END, phases=[4, 6])
        assert "Ph6" in capsys.readouterr().out
        assert set(res["cycle"]["phase"]) == {4}

    def test_no_role_config_returns_empty(self, tmp_path, capsys):
        # As at 701: Pairs and Arrival only, no Det_P{N}_Stop_Bar.
        db = _build_db(tmp_path, stop_bar=False)
        assert GreenTimeEngine(db).green_time(START, END) == {}
        assert "Stop_Bar" in capsys.readouterr().out
        out = tmp_path / "out"
        assert GreenTimeEngine(db).green_time(START, END, output_dir=out) is None
        assert not out.exists() or not list(out.iterdir())

    def test_no_greens_in_window_returns_empty(self, db):
        assert GreenTimeEngine(db).green_time(f"{DAY} 12:00", f"{DAY} 13:00") == {}

    def test_date_only_end_is_whole_day(self, db):
        cy = GreenTimeEngine(db).green_time(DAY, DAY, phases=[2])["cycle"]
        assert len(cy) == _N and not cy["censored"].any()

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert GreenTimeEngine(db).green_time(START, END, output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {
            f"GTU_Cycle_{STAMP}.csv",
            f"GTU_Actuations_{STAMP}.csv",
            f"GTU_Bins_15min_{STAMP}.csv",
            f"GTU_Splits_15min_{STAMP}.csv",
            f"GTU_PlanBins_{STAMP}.csv",
            f"GTU_PlanSplits_{STAMP}.csv",
            f"GTU_Chart_{STAMP}.html",
        }
        assert list(pd.read_csv(out / f"GTU_Cycle_{STAMP}.csv").columns) == CYCLE_SCHEMA
        assert list(pd.read_csv(out / f"GTU_PlanBins_{STAMP}.csv").columns) == BIN_SCHEMA

    def test_no_plot_writes_no_html(self, db, tmp_path):
        out = tmp_path / "out"
        GreenTimeEngine(db).green_time(START, END, make_plot=False, output_dir=out)
        assert not list(out.glob("*.html"))

    def test_convenience_wrapper(self, db):
        res = get_green_time(db, START, END, phases=[4], timezone=TZ)
        assert len(res["cycle"]) == 72


# ---------------------------------------------------------------------------
# Plot (functional core: pure)
# ---------------------------------------------------------------------------

def _frames():
    t0 = to_epoch(datetime.strptime(f"{DAY} 07:00", "%Y-%m-%d %H:%M"), TZ)
    rows = []
    for k in range(6):
        b = t0 + 100.0 * k
        rows += [(b, 1, 2), (b + 20, 8, 2), (b + 24, 9, 2), (b + 24, 10, 2), (b + 26, 11, 2)]
        rows += [(b + 1, 82, 26), (b + 3, 82, 26)]
        rows += [(b + 30, 1, 4), (b + 40, 8, 4), (b + 44, 9, 4), (b + 44, 10, 4),
                 (b + 46, 11, 4), (b + 31, 82, 31)]
    rows += _plan_dump(t0 - 10, 1, 100, {2: 40, 4: 20})
    ev = pd.DataFrame(rows, columns=["t", "event_code", "parameter"])
    ev["timestamp"] = pd.to_datetime(ev["t"], unit="s", utc=True).dt.tz_convert(TZ)
    ev["cycle_start"] = ev["timestamp"]
    ev["coord_plan"] = 1.0
    ev = ev.sort_values("timestamp", kind="stable").drop(columns="t").reset_index(drop=True)
    tl = plan_timeline(ev)
    parts = [green_time_utilization(ev, ph, [d], timeline=tl) for ph, d in ((2, 26), (4, 31))]
    cy = pd.concat([p[0] for p in parts], ignore_index=True)
    ac = pd.concat([p[1] for p in parts], ignore_index=True)
    return summarize_gtu_bins(cy, ac, bin_len=5), summarize_gtu_splits(cy, bin_len=5)


class TestPlot:

    def test_traces_per_phase(self):
        bins, splits = _frames()
        fig = plot_green_time(bins, splits, metadata={"intersection_name": "X"})
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        for ph in (2, 4):
            assert f"Ph{ph} Utilization" in names
            assert f"Ph{ph} Average Green" in names
            assert f"Ph{ph} Programmed Green" in names
        hm = next(t for t in fig.data if t.name == "Ph2 Utilization")
        assert isinstance(hm, go.Heatmap)
        z = np.asarray(hm.z, dtype=float)
        # 10 bins of a 20 s green; actuations at 1 s and 3 s → bins 0 and 1.
        assert z.shape[0] == 10
        assert np.nanmax(z[0]) == 1.0 and np.nanmax(z[1]) == 1.0 and np.nanmax(z[2]) == 0.0
        avg = next(t for t in fig.data if t.name == "Ph2 Average Green")
        assert set(np.asarray(avg.y, dtype=float)) == {20.0}
        prog = next(t for t in fig.data if t.name == "Ph2 Programmed Green")
        assert set(np.asarray(prog.y, dtype=float)) == {34.0}
        prog4 = next(t for t in fig.data if t.name == "Ph4 Programmed Green")
        assert set(np.asarray(prog4.y, dtype=float)) == {14.0}

    def test_max_green_caps_the_heatmap(self):
        bins, splits = _frames()
        fig = plot_green_time(bins, splits, max_green_s=6.0)
        hm = next(t for t in fig.data if t.name == "Ph2 Utilization")
        assert np.asarray(hm.z, dtype=float).shape[0] == 3        # bins 0, 2, 4 s

    def test_plan_change_inside_a_time_bin_pools_the_plans(self):
        # Summaries group by plan, so one 5-min bin holds two groups.  The
        # heat map pools them: actuations over every cycle of the time bin.
        bins, splits = _frames()
        b = bins.copy()
        first = b["time"] == b["time"].min()
        extra = b.loc[first & (b["phase"] == 2)].copy()
        extra["coord_plan"] = 2.0
        extra = extra.loc[extra["green_bin"] < 2]          # shorter greens
        extra["n_cycles"] = 2
        extra["actuations"] = 0
        fig = plot_green_time(pd.concat([b, extra], ignore_index=True), splits)
        hm = next(t for t in fig.data if t.name == "Ph2 Utilization")
        z = np.asarray(hm.z, dtype=float)
        n1 = int(b.loc[first & (b["phase"] == 2), "n_cycles"].iloc[0])
        assert z[0, 0] == pytest.approx(n1 / (n1 + 2))
        assert z[5, 0] == 0.0

    def test_title_uses_metadata(self):
        bins, splits = _frames()
        meta = {"intersection_name": "X", "major_road_name": "Main St",
                "minor_road_name": "Side St"}
        fig = plot_green_time(bins, splits, metadata=meta)
        assert "Main St" in fig.layout.title.text
        assert "Green Time Utilization" in fig.layout.title.text

    def test_empty_frames_give_figure(self):
        assert isinstance(plot_green_time(pd.DataFrame(), pd.DataFrame()), go.Figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    @staticmethod
    def _parse(*argv):
        return cli._build_parser().parse_args(["green-time", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "900", "--start", START, "--end", END)
        assert args.func is cli.handle_green_time
        assert args.role == "stop_bar" and args.bin_s == 2.0 and args.bin_len == 15
        assert args.overlap is False and args.no_exclusions is False
        assert args.max_green == 120.0
        assert args.no_plot is False and args.phases is None

    def test_role_choices(self):
        assert self._parse("--targetid", "900", "--start", DAY, "--end", DAY,
                           "--role", "occupancy").role == "occupancy"
        with pytest.raises(SystemExit):
            self._parse("--targetid", "900", "--start", DAY, "--end", DAY, "--role", "pairs")

    def test_target_group_is_required_and_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse("--start", DAY, "--end", DAY)
        with pytest.raises(SystemExit):
            self._parse("--all", "--targetid", "900", "--start", DAY, "--end", DAY)

    @staticmethod
    def _site(tmp_path, monkeypatch, **kw):
        folder = tmp_path / "intersections" / "900_Synthetic"
        folder.mkdir(parents=True)
        _build_db(folder, **kw)
        (folder / "metadata.json").write_text(json.dumps({
            "intersection_name": "Synthetic", "intersection_id": "900",
            "db_filename": "gtu.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        return folder

    def test_end_to_end(self, tmp_path, monkeypatch, capsys):
        folder = self._site(tmp_path, monkeypatch)
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--phases", "2", "--no-plot")
        args.func(args)
        ps = pd.read_csv(folder / "outputs" / f"GTU_PlanSplits_{STAMP}.csv")
        assert set(ps["phase"]) == {2} and ps["n_cycles"].sum() == 72
        assert sorted(ps["programmed_green"].tolist()) == [44.0, 54.0]
        assert not list((folder / "outputs").glob("*.html"))
        assert "Ph2" in capsys.readouterr().out

    def test_no_role_config_does_not_raise(self, tmp_path, monkeypatch, capsys):
        folder = self._site(tmp_path, monkeypatch, stop_bar=False)
        args = self._parse("--targetid", "900", "--start", START, "--end", END)
        args.func(args)
        assert "Stop_Bar" in capsys.readouterr().out
        out = folder / "outputs"
        assert not out.exists() or not list(out.glob("GTU_*"))
