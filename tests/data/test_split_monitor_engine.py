# Acceptance tests for the split-monitor shell engine, plot and CLI
# (UDOT S-M2; spec docs/specs/split_monitor_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The measure itself is pinned in tests/analysis/test_split_monitor.py.

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.split_monitor import (
    CYCLE_SCHEMA,
    STATS_SCHEMA,
    TIMELINE_SCHEMA,
    plan_timeline,
    split_monitor,
)
from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor
from atspm.utils.timezone import to_epoch

# Built by the delegated S-M2 run; these imports fail until they exist.
from atspm.data.split_monitor import (
    _ALL_SM_CODES,
    PLAN_LOOKBACK_S,
    SplitMonitorEngine,
    get_split_monitor,
)
from atspm.plotting.split_monitor import plot_split_monitor

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 09:00"
STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection: one ring 2|4, 100 s cycles from 06:50 to 09:10.
#   P2  green 0, yellow 54, end yellow 58, red clearance to 60 → split 60.
#       Force-off (6) at its yellow.
#   P4  green 60, yellow 94, end yellow 98, red clearance to 100 → split 40.
#       Gap-out (4) at its yellow; walk (21) at its green.
# Plan codes: the full midnight dump at 00:00 (plan 1, cycle 100, offset 0,
# P2 60, P4 40, all others 0), then at 08:00 plan 2 logs only what changes
# (131 = 2, P2 50, P4 50).  The 07:00 window sees plan 1 only through the
# 26 h plan-code lookback.  Optionally a comms-gap marker at 08:30.
# ---------------------------------------------------------------------------

_CYCLE = 100.0
_PLAN = {2: (0.0, 54.0, 6), 4: (60.0, 34.0, 4)}
_N = 84                                  # 06:50 → 09:10


def _local(hhmm: str) -> float:
    return to_epoch(datetime.strptime(f"{DAY} {hhmm}", "%Y-%m-%d %H:%M"), TZ)


def _build_db(root: Path, gap_at: str = None) -> Path:
    t0 = _local("06:50")
    events = []
    midnight = _local("00:00")
    dump = {131: 1, 132: 100, 133: 0}
    dump.update({133 + p: 0 for p in range(1, 17)})
    dump.update({135: 60, 137: 40})
    events += [(midnight, c, v) for c, v in dump.items()]
    t8 = _local("08:00")
    events += [(t8, 131, 2), (t8, 135, 50), (t8, 137, 50)]
    for c in range(_N):
        base = t0 + c * _CYCLE
        for ph, (off, g, term) in _PLAN.items():
            gs = base + off
            events += [(gs, 1, ph), (gs + g, 8, ph), (gs + g, term, ph),
                       (gs + g + 4, 9, ph), (gs + g + 4, 10, ph),
                       (gs + g + 6, 11, ph), (gs + g + 6, 12, ph)]
        events.append((base + 60.0, 21, 4))
    if gap_at:
        events.append((_local(gap_at) + 0.05, -1, -1))
    db = root / "sm.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        cfg = {"RB_R1": "2|4"}
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00",
                              "end_date": None, **cfg})
        m.insert_events(sorted(events))
        m.conn.execute(
            "INSERT INTO ingestion_log (span_start, span_end, processed_at, "
            "row_count) VALUES (?, ?, '2025-06-02T00:00:00', ?)",
            (midnight, t0 + _N * _CYCLE, len(events)),
        )
        m.conn.commit()
    CycleProcessor(db, TZ).run()
    return db


@pytest.fixture
def db(tmp_path) -> Path:
    return _build_db(tmp_path)


def _ph(df, phase):
    return df.loc[df["phase"] == phase].reset_index(drop=True)


def _na_list(s):
    return [None if pd.isna(v) else v for v in s]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_codes_and_lookback(self):
        assert set(_ALL_SM_CODES) == {-1, 1, 4, 5, 6, 8, 9, 10, 11, 12, 21} | set(range(131, 150))
        assert PLAN_LOOKBACK_S >= 25 * 3600

    def test_result_keys_and_columns(self, db):
        res = SplitMonitorEngine(db).split_monitor(START, END)
        assert set(res) == {"cycle", "stats", "timeline"}
        assert list(res["cycle"].columns) == CYCLE_SCHEMA
        assert list(res["stats"].columns) == STATS_SCHEMA
        assert list(res["timeline"].columns) == TIMELINE_SCHEMA
        assert set(res["cycle"]["phase"]) == {2, 4}

    def test_window_by_green(self, db):
        cyc = SplitMonitorEngine(db).split_monitor(START, END)["cycle"]
        w0, w1 = _local("07:00"), _local("09:00")
        for ph in (2, 4):
            p = _ph(cyc, ph)
            assert len(p) == 72
            g = np.array([t.timestamp() for t in p["green_ts"]])
            assert (g >= w0).all() and (g < w1).all()

    def test_splits_terminations_and_walk(self, db):
        cyc = SplitMonitorEngine(db).split_monitor(START, END)["cycle"]
        p2, p4 = _ph(cyc, 2), _ph(cyc, 4)
        np.testing.assert_allclose(p2["split_dur"].to_numpy(float), 60.0)
        np.testing.assert_allclose(p4["split_dur"].to_numpy(float), 40.0)
        assert set(p2["termination"]) == {"force_off"}
        assert set(p4["termination"]) == {"gap_out"}
        assert p4["ped_walk"].all() and not p2["ped_walk"].any()

    def test_lookback_finds_the_midnight_dump(self, db):
        cyc = SplitMonitorEngine(db).split_monitor(START, END)["cycle"]
        p2 = _ph(cyc, 2)
        before = p2["green_ts"] < pd.Timestamp(f"{DAY} 08:00", tz=TZ)
        assert _na_list(p2.loc[before, "plan"]) == [1] * 36
        assert _na_list(p2.loc[~before, "plan"]) == [2] * 36
        np.testing.assert_allclose(p2.loc[before, "programmed_split"].to_numpy(float), 60.0)
        np.testing.assert_allclose(p2.loc[~before, "programmed_split"].to_numpy(float), 50.0)
        np.testing.assert_allclose(p2.loc[~before, "split_minus_programmed"].to_numpy(float), 10.0)
        # Plan 2 logged no cycle: it is inherited from the dump.
        assert _na_list(p2["programmed_cycle"]) == [100] * 72

    def test_matches_the_core_on_the_same_events(self, db):
        from atspm.data.reader import get_events_with_cycles_df
        from datetime import timedelta
        res = SplitMonitorEngine(db).split_monitor(START, END)
        s = datetime.strptime(START, "%Y-%m-%d %H:%M")
        ev = get_events_with_cycles_df(db, s - timedelta(seconds=PLAN_LOOKBACK_S),
                                       datetime.strptime(END, "%Y-%m-%d %H:%M") + timedelta(hours=1),
                                       event_codes=_ALL_SM_CODES, timezone=TZ)
        core = split_monitor(ev)
        core = core.loc[(core["green_ts"] >= pd.Timestamp(START, tz=TZ))
                        & (core["green_ts"] < pd.Timestamp(END, tz=TZ))].reset_index(drop=True)
        got = res["cycle"].reset_index(drop=True)
        assert len(got) == len(core)
        np.testing.assert_allclose(got["split_dur"].to_numpy(float), core["split_dur"].to_numpy(float))
        assert got["termination"].tolist() == core["termination"].tolist()
        assert _na_list(got["plan"]) == _na_list(core["plan"])

    def test_stats_per_plan(self, db):
        st = SplitMonitorEngine(db).split_monitor(START, END)["stats"]
        p2 = _ph(st, 2)
        assert _na_list(p2["plan"]) == [1, 2]
        assert p2["programmed_split"].tolist() == [60.0, 50.0]
        assert p2["n_cycles"].tolist() == [36, 36]
        assert p2["force_off_pct"].tolist() == [1.0, 1.0]
        assert _ph(st, 4)["ped_walk_pct"].tolist() == [1.0, 1.0]

    def test_percentiles_rename_stats_columns(self, db):
        st = SplitMonitorEngine(db).split_monitor(START, END, percentiles=(50, 95))["stats"]
        assert "split_p95" in st.columns and "split_p85" not in st.columns

    def test_timeline_is_trimmed_to_the_window(self, db):
        tl = SplitMonitorEngine(db).split_monitor(START, END)["timeline"]
        assert _na_list(tl["plan"]) == [1, 2]
        assert tl["start"].iloc[0] == pd.Timestamp(f"{DAY} 00:00", tz=TZ)
        assert tl["start"].iloc[1] == pd.Timestamp(f"{DAY} 08:00", tz=TZ)

    def test_gap_resets_the_plan(self, tmp_path):
        db = _build_db(tmp_path, gap_at="08:30")
        res = SplitMonitorEngine(db).split_monitor(START, END, phases=[2])
        p2 = res["cycle"]
        after = p2["green_ts"] >= pd.Timestamp(f"{DAY} 08:30", tz=TZ)
        assert after.sum() > 0
        assert p2.loc[after, "plan"].isna().all()
        assert p2.loc[after, "programmed_split"].isna().all()
        assert res["stats"]["plan"].isna().any()

    def test_phases_filter(self, db):
        res = SplitMonitorEngine(db).split_monitor(START, END, phases=[4])
        assert set(res["cycle"]["phase"]) == {4}
        assert set(res["stats"]["phase"]) == {4}

    def test_requested_phase_without_services_is_warned(self, db, capsys):
        res = SplitMonitorEngine(db).split_monitor(START, END, phases=[2, 6])
        assert "Ph6" in capsys.readouterr().out
        assert set(res["cycle"]["phase"]) == {2}

    def test_no_services_returns_empty(self, db, capsys):
        res = SplitMonitorEngine(db).split_monitor(f"{DAY} 12:00", f"{DAY} 13:00")
        assert res == {}
        assert "no phase services" in capsys.readouterr().out.lower()

    def test_date_only_end_is_whole_day(self, db):
        cyc = SplitMonitorEngine(db).split_monitor(DAY, DAY, phases=[2])["cycle"]
        assert len(cyc) == _N

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert SplitMonitorEngine(db).split_monitor(START, END, output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {
            f"SM_Cycle_{STAMP}.csv",
            f"SM_Stats_{STAMP}.csv",
            f"SM_Plans_{STAMP}.csv",
            f"SM_Splits_{STAMP}.html",
        }
        cyc = pd.read_csv(out / f"SM_Cycle_{STAMP}.csv")
        assert list(cyc.columns) == CYCLE_SCHEMA

    def test_no_plot_writes_no_html(self, db, tmp_path):
        out = tmp_path / "out"
        SplitMonitorEngine(db).split_monitor(START, END, make_plot=False, output_dir=out)
        assert not list(out.glob("*.html"))

    def test_convenience_wrapper(self, db):
        res = get_split_monitor(db, START, END, phases=[4], timezone=TZ)
        assert set(res["cycle"]["phase"]) == {4}


# ---------------------------------------------------------------------------
# Plot (functional core: pure)
# ---------------------------------------------------------------------------


def _synthetic_frames():
    """Two phases; plan 1 then plan 2 at +300 s, then a gap at +500 s."""
    t0 = to_epoch(datetime.strptime(f"{DAY} 07:00", "%Y-%m-%d %H:%M"), TZ)
    rows = [(0, 131, 1), (0, 132, 100), (0, 133, 0)]
    rows += [(0, 133 + p, 0) for p in range(1, 17)]
    rows += [(0, 135, 60), (0, 137, 40)]
    rows += [(300, 131, 2), (300, 135, 50), (300, 137, 50)]
    rows += [(500, -1, -1)]
    for k, g in enumerate((10, 110, 210, 310, 410, 510, 610)):
        rows += [(g, 1, 2), (g + 54, 8, 2), (g + 54, 6, 2), (g + 58, 9, 2),
                 (g + 58, 10, 2), (g + 60, 11, 2)]
        rows += [(g + 60, 1, 4), (g + 60 + 30, 8, 4), (g + 60 + 30, 4 if k % 2 else 5, 4),
                 (g + 90 + 4, 9, 4), (g + 94, 10, 4), (g + 96, 11, 4)]
        if k == 0:
            rows.append((g + 60, 21, 4))
    ev = pd.DataFrame(rows, columns=["t", "event_code", "parameter"])
    ev["timestamp"] = pd.to_datetime(t0 + ev["t"], unit="s", utc=True).dt.tz_convert(TZ)
    ev["cycle_start"] = pd.NaT
    ev = ev.sort_values("timestamp", kind="stable").drop(columns="t").reset_index(drop=True)
    tl = plan_timeline(ev)
    return split_monitor(ev, timeline=tl), tl


class TestPlot:

    def test_traces_per_phase_and_termination(self):
        cyc, tl = _synthetic_frames()
        fig = plot_split_monitor(cyc, tl, metadata={"intersection_name": "X"})
        assert isinstance(fig, go.Figure)
        names = {t.name for t in fig.data}
        assert {"Ph2 Force Off", "Ph4 Gap Out", "Ph4 Max Out", "Ph4 Ped Walk",
                "Ph2 Programmed", "Ph4 Programmed"} <= names
        assert "Ph2 Gap Out" not in names                   # no empty traces
        fo = next(t for t in fig.data if t.name == "Ph2 Force Off")
        assert list(fo.y) == pytest.approx([60.0] * 7)
        walk = next(t for t in fig.data if t.name == "Ph4 Ped Walk")
        assert len(walk.y) == 1

    def test_programmed_split_is_a_segment_line(self):
        cyc, tl = _synthetic_frames()
        fig = plot_split_monitor(cyc, tl)
        prog = next(t for t in fig.data if t.name == "Ph2 Programmed")
        # [start, end, None] per known timeline row; the gap row is skipped.
        assert list(prog.y) == [60, 60, None, 50, 50, None]
        assert prog.x[2] is None and prog.x[5] is None

    def test_plan_bands_are_labelled(self):
        cyc, tl = _synthetic_frames()
        fig = plot_split_monitor(cyc, tl)
        texts = " ".join(a.text for a in fig.layout.annotations if a.text)
        assert "Plan 1" in texts and "Plan 2" in texts
        assert len(fig.layout.shapes) >= 2

    def test_title_uses_metadata(self):
        cyc, tl = _synthetic_frames()
        fig = plot_split_monitor(cyc, tl, metadata={
            "intersection_name": "X", "major_road_name": "Main St",
            "minor_road_name": "Side St"})
        assert "Main St" in fig.layout.title.text
        assert "Split Monitor" in fig.layout.title.text

    def test_empty_frame_gives_figure(self):
        assert isinstance(plot_split_monitor(pd.DataFrame(), pd.DataFrame()), go.Figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    @staticmethod
    def _parse(*argv):
        return cli._build_parser().parse_args(["split-monitor", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "900", "--start", START, "--end", END)
        assert args.func is cli.handle_split_monitor
        assert args.percentiles == [50.0, 85.0]
        assert args.no_plot is False
        assert args.phases is None

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
            "db_filename": "sm.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--phases", "4", "--no-plot")
        args.func(args)
        st = pd.read_csv(folder / "outputs" / f"SM_Stats_{STAMP}.csv")
        assert set(st["phase"]) == {4}
        assert st["programmed_split"].tolist() == [40.0, 50.0]
        assert not list((folder / "outputs").glob("*.html"))
