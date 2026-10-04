# Acceptance tests for the ped delay / wait time shell engine, plots and CLI
# (UDOT S-M5; spec docs/specs/call_service_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The measures themselves are pinned in tests/analysis/test_call_service.py.

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.call_service import (
    PED_DELAY_SCHEMA,
    PED_SUMMARY_SCHEMA,
    WAIT_SUMMARY_SCHEMA,
    WAIT_TIME_SCHEMA,
    ped_delay,
    summarize_ped_delay,
    summarize_wait_time,
    wait_time,
)
from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor
from atspm.utils.timezone import to_epoch

# Built by the delegated S-M5 run; these imports fail until they exist.
from atspm.data.call_service import (
    _PED_CODES,
    _WAIT_CODES,
    CallServiceEngine,
    get_ped_delay,
    get_wait_time,
)
from atspm.plotting.call_service import plot_ped_delay, plot_wait_time

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 09:00"
STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection: one ring 2|4, 100 s cycles from 06:50 to 09:10.
#   P2  green 0, force off (6) + yellow 54, end yellow 58 (9, 10), red 60 (11).
#       Phase call 43 @ +70 (its red), dropped 44 @ +105 (next green):
#       wait 30 s, termination force_off.
#   P4  green 60, gap out (4) + yellow 94, end yellow 98, red 100.
#       Phase call 43 @ +20, dropped 44 @ +25, again 43 @ +45 (its red,
#       which started at +0), dropped 44 @ +65 (green):
#       wait 40 s from the first call, 15 s with the dropping algorithm.
#       Det_P4_Occupancy is configured, so dropping="auto" uses it on P4.
#       Ped: Begin Walk 21 @ +60, clearance 22 @ +67, press 90 + call 45
#       @ +30 → ped delay 30 s.
# Variants: a comms gap at 08:31:15 (inside P2's red and P4's green), and a
# controller that logs no ped events.
# ---------------------------------------------------------------------------

_CYCLE = 100.0
_N = 84                                  # 06:50 → 09:10


def _local(hhmm: str) -> float:
    return to_epoch(datetime.strptime(f"{DAY} {hhmm}", "%Y-%m-%d %H:%M"), TZ)


def _build_db(root: Path, gap: bool = False, ped: bool = True) -> Path:
    t0 = _local("06:50")
    events = []
    for c in range(_N):
        b = t0 + c * _CYCLE
        events += [(b, 1, 2), (b + 54, 6, 2), (b + 54, 8, 2), (b + 58, 9, 2), (b + 58, 10, 2),
                   (b + 60, 11, 2), (b + 60, 12, 2)]
        events += [(b + 60, 1, 4), (b + 94, 4, 4), (b + 94, 8, 4), (b + 98, 9, 4),
                   (b + 98, 10, 4), (b + 100, 11, 4), (b + 100, 12, 4)]
        events += [(b + 70, 43, 2), (b + 105, 44, 2)]
        events += [(b + 20, 43, 4), (b + 25, 44, 4), (b + 45, 43, 4), (b + 65, 44, 4)]
        if ped:
            events += [(b + 30, 45, 4), (b + 30, 90, 4), (b + 30.4, 89, 4),
                       (b + 60, 21, 4), (b + 67, 22, 4), (b + 80, 23, 4)]
    if gap:
        events.append((_local("08:30") + 75.0, -1, -1))
    db = root / "cs.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        cfg = {"RB_R1": "2|4", "Det_P4_Occupancy": "50", "Det_P2_Stop_Bar": "26"}
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00",
                              "end_date": None, **cfg})
        m.insert_events(sorted(events))
        m.conn.execute(
            "INSERT INTO ingestion_log (span_start, span_end, processed_at, "
            "row_count) VALUES (?, ?, '2025-06-02T00:00:00', ?)",
            (t0, t0 + _N * _CYCLE, len(events)),
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
# Engine: ped delay
# ---------------------------------------------------------------------------


class TestPedDelayEngine:

    def test_codes(self):
        assert set(_PED_CODES) == {-1, 21, 22, 45, 90}

    def test_result_keys_and_columns(self, db):
        res = CallServiceEngine(db).ped_delay(START, END)
        assert set(res) == {"delays", "binned", "plans"}
        assert list(res["delays"].columns) == PED_DELAY_SCHEMA
        assert list(res["binned"].columns) == PED_SUMMARY_SCHEMA
        assert list(res["plans"].columns) == PED_SUMMARY_SCHEMA

    def test_window_by_walk(self, db):
        d = CallServiceEngine(db).ped_delay(START, END)["delays"]
        assert len(d) == 72 and set(d["phase"]) == {4}
        w = _epoch(d["walk_ts"])
        assert (w >= _local("07:00")).all() and (w < _local("09:00")).all()
        assert (d["kind"] == "waited").all()
        assert (d["delay_s"] == 30.0).all() and (d["source"] == 90).all()

    def test_matches_the_core_on_the_same_events(self, db):
        from atspm.data.reader import get_events_with_cycles_df
        res = CallServiceEngine(db).ped_delay(START, END)
        s = datetime.strptime(START, "%Y-%m-%d %H:%M")
        e = datetime.strptime(END, "%Y-%m-%d %H:%M")
        ev = get_events_with_cycles_df(db, s - timedelta(hours=1), e + timedelta(hours=1),
                                       event_codes=_PED_CODES, timezone=TZ)
        core, why = ped_delay(ev)
        assert why is None
        core = core.loc[(core["walk_ts"] >= pd.Timestamp(START, tz=TZ))
                        & (core["walk_ts"] < pd.Timestamp(END, tz=TZ))].reset_index(drop=True)
        got = res["delays"].reset_index(drop=True)
        assert len(got) == len(core)
        np.testing.assert_allclose(got["delay_s"].to_numpy(float), core["delay_s"].to_numpy(float))

    def test_binned_and_plans(self, db):
        res = CallServiceEngine(db).ped_delay(START, END, bin_len=30)
        b = res["binned"]
        assert len(b) == 4 and b["n_called"].sum() == 72
        assert (b["time"].diff().dropna() == pd.Timedelta(minutes=30)).all()
        pl = res["plans"]
        assert len(pl) == 1
        r = pl.iloc[0]
        assert (r["n_walks"], r["n_called"], r["n_delays"], r["presses"]) == (72, 72, 72, 72)
        assert r["avg_delay_s"] == pytest.approx(30.0)
        assert r["total_delay_s"] == pytest.approx(2160.0)

    def test_gap_censors_the_walk(self, tmp_path):
        db = _build_db(tmp_path, gap=True)
        d = CallServiceEngine(db).ped_delay(START, END)["delays"]
        assert len(d) == 72
        cens = d.loc[d["kind"] == "censored"]
        assert len(cens) == 1
        assert cens["walk_ts"].iloc[0] == pd.Timestamp(f"{DAY} 08:32:40", tz=TZ)
        assert cens["delay_s"].isna().all()

    def test_not_computable_returns_empty_with_reason(self, tmp_path, capsys):
        db = _build_db(tmp_path, ped=False)
        assert CallServiceEngine(db).ped_delay(START, END) == {}
        assert "Code 21" in capsys.readouterr().out
        out = tmp_path / "out"
        assert CallServiceEngine(db).ped_delay(START, END, output_dir=out) is None
        assert not out.exists() or not list(out.iterdir())

    def test_phases_filter_and_warning(self, db, capsys):
        res = CallServiceEngine(db).ped_delay(START, END, phases=[4, 6])
        assert "Ph6" in capsys.readouterr().out
        assert set(res["delays"]["phase"]) == {4}

    def test_no_walks_in_window_returns_empty(self, db):
        assert CallServiceEngine(db).ped_delay(f"{DAY} 12:00", f"{DAY} 13:00") == {}

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert CallServiceEngine(db).ped_delay(START, END, output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {
            f"PedDelay_Walks_{STAMP}.csv",
            f"PedDelay_60min_{STAMP}.csv",
            f"PedDelay_Plans_{STAMP}.csv",
            f"PedDelay_Chart_{STAMP}.html",
        }
        assert list(pd.read_csv(out / f"PedDelay_Walks_{STAMP}.csv").columns) == PED_DELAY_SCHEMA

    def test_no_plot_writes_no_html(self, db, tmp_path):
        out = tmp_path / "out"
        CallServiceEngine(db).ped_delay(START, END, make_plot=False, output_dir=out)
        assert not list(out.glob("*.html"))

    def test_convenience_wrapper(self, db):
        res = get_ped_delay(db, START, END, phases=[4], timezone=TZ)
        assert len(res["delays"]) == 72


# ---------------------------------------------------------------------------
# Engine: wait time
# ---------------------------------------------------------------------------


class TestWaitTimeEngine:

    def test_codes(self):
        assert set(_WAIT_CODES) == {-1, 1, 4, 5, 6, 8, 9, 10, 11, 12, 43, 44}

    def test_result_keys_and_columns(self, db):
        res = CallServiceEngine(db).wait_time(START, END)
        assert set(res) == {"windows", "binned", "plans"}
        assert list(res["windows"].columns) == WAIT_TIME_SCHEMA
        assert list(res["binned"].columns) == WAIT_SUMMARY_SCHEMA
        assert list(res["plans"].columns) == WAIT_SUMMARY_SCHEMA
        assert set(res["windows"]["phase"]) == {2, 4}

    def test_window_by_green(self, db):
        w = CallServiceEngine(db).wait_time(START, END)["windows"]
        for ph in (2, 4):
            p = _ph(w, ph)
            assert len(p) == 72
            g = _epoch(p["green_ts"])
            assert (g >= _local("07:00")).all() and (g < _local("09:00")).all()
            assert not p["censored"].any() and p["called"].all() and not p["held"].any()

    def test_waits_and_terminations(self, db):
        w = CallServiceEngine(db).wait_time(START, END, dropping="off")["windows"]
        p2, p4 = _ph(w, 2), _ph(w, 4)
        assert (p2["wait_s"] == 30.0).all() and (p2["termination"] == "force_off").all()
        assert (p4["wait_s"] == 40.0).all() and (p4["termination"] == "gap_out").all()
        assert (p4["n_calls"] == 2).all() and (p4["n_drops"] == 1).all()

    def test_dropping_auto_uses_occupancy_phases(self, db, capsys):
        w = CallServiceEngine(db).wait_time(START, END)["windows"]
        assert "Ph4" in capsys.readouterr().out          # the dropping info line
        assert (_ph(w, 4)["wait_s"] == 15.0).all()
        assert (_ph(w, 2)["wait_s"] == 30.0).all()

    def test_dropping_on_and_off(self, db):
        eng = CallServiceEngine(db)
        assert (_ph(eng.wait_time(START, END, dropping="on")["windows"], 4)["wait_s"] == 15.0).all()
        assert (_ph(eng.wait_time(START, END, dropping="off")["windows"], 4)["wait_s"] == 40.0).all()

    def test_unknown_dropping_raises(self, db):
        with pytest.raises(ValueError):
            CallServiceEngine(db).wait_time(START, END, dropping="sometimes")

    def test_matches_the_core_on_the_same_events(self, db):
        from atspm.data.reader import get_events_with_cycles_df
        res = CallServiceEngine(db).wait_time(START, END, phases=[4])
        s = datetime.strptime(START, "%Y-%m-%d %H:%M")
        e = datetime.strptime(END, "%Y-%m-%d %H:%M")
        ev = get_events_with_cycles_df(db, s - timedelta(hours=1), e + timedelta(hours=1),
                                       event_codes=_WAIT_CODES, timezone=TZ)
        core = wait_time(ev, phases=[4], dropping=[4])
        core = core.loc[(core["green_ts"] >= pd.Timestamp(START, tz=TZ))
                        & (core["green_ts"] < pd.Timestamp(END, tz=TZ))].reset_index(drop=True)
        got = res["windows"].reset_index(drop=True)
        assert len(got) == len(core)
        np.testing.assert_allclose(got["wait_s"].to_numpy(float), core["wait_s"].to_numpy(float))

    def test_max_wait(self, db):
        eng = CallServiceEngine(db)
        pl = eng.wait_time(START, END, dropping="off", max_wait=35.0)["plans"]
        p4 = _ph(pl, 4).iloc[0]
        assert p4["n_over_max"] == 72 and p4["n_called"] == 0
        pl = eng.wait_time(START, END, dropping="off", max_wait=None)["plans"]
        assert _ph(pl, 4).iloc[0]["n_called"] == 72

    def test_gap_censors_the_window(self, tmp_path):
        db = _build_db(tmp_path, gap=True)
        w = CallServiceEngine(db).wait_time(START, END)["windows"]
        p2 = _ph(w, 2)
        cut = p2["green_ts"] == pd.Timestamp(f"{DAY} 08:31:40", tz=TZ)
        assert cut.sum() == 1 and p2.loc[cut, "censored"].all()
        assert p2.loc[cut, "wait_s"].isna().all()
        assert p2["called"].sum() == 71
        # The gap splits P4's green at 08:31:00, so the red after it is lost.
        assert len(_ph(w, 4)) == 71

    def test_binned_and_plans(self, db):
        res = CallServiceEngine(db).wait_time(START, END, bin_len=30)
        b = _ph(res["binned"], 2)
        assert len(b) == 4 and b["n_called"].sum() == 72
        assert (b["time"].diff().dropna() == pd.Timedelta(minutes=30)).all()
        pl = res["plans"]
        assert len(pl) == 2
        p2 = _ph(pl, 2).iloc[0]
        assert (p2["n_windows"], p2["n_called"], p2["n_force_off"]) == (72, 72, 72)
        assert p2["avg_wait_s"] == pytest.approx(30.0)
        assert p2["avg_wait_force_off_s"] == pytest.approx(30.0)

    def test_phases_filter_and_warning(self, db, capsys):
        res = CallServiceEngine(db).wait_time(START, END, phases=[2, 6])
        assert "Ph6" in capsys.readouterr().out
        assert set(res["windows"]["phase"]) == {2}

    def test_no_windows_returns_empty(self, db):
        assert CallServiceEngine(db).wait_time(f"{DAY} 12:00", f"{DAY} 13:00") == {}

    def test_date_only_end_is_whole_day(self, db):
        w = CallServiceEngine(db).wait_time(DAY, DAY, phases=[2])["windows"]
        # Greens 06:51:40 → 09:08:20 follow a red in the data; the last red
        # (09:09:20) has no green after it: present but censored.
        assert len(w) == _N and w["censored"].sum() == 1

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert CallServiceEngine(db).wait_time(START, END, output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {
            f"WaitTime_Windows_{STAMP}.csv",
            f"WaitTime_15min_{STAMP}.csv",
            f"WaitTime_Plans_{STAMP}.csv",
            f"WaitTime_Chart_{STAMP}.html",
        }
        assert list(pd.read_csv(out / f"WaitTime_Windows_{STAMP}.csv").columns) == WAIT_TIME_SCHEMA

    def test_convenience_wrapper(self, db):
        res = get_wait_time(db, START, END, phases=[4], dropping="off", timezone=TZ)
        assert (res["windows"]["wait_s"] == 40.0).all()


# ---------------------------------------------------------------------------
# Plots (functional core: pure)
# ---------------------------------------------------------------------------

def _ev(rows):
    t0 = to_epoch(datetime.strptime(f"{DAY} 07:00", "%Y-%m-%d %H:%M"), TZ)
    ev = pd.DataFrame(rows, columns=["t", "event_code", "parameter"])
    ev["timestamp"] = pd.to_datetime(t0 + ev["t"], unit="s", utc=True).dt.tz_convert(TZ)
    ev["cycle_start"] = ev["timestamp"]
    ev["coord_plan"] = 1.0
    return ev.sort_values("timestamp", kind="stable").drop(columns="t").reset_index(drop=True)


def _ped_frames():
    rows = [(0, 22, 4)]
    for k in range(4):
        b = 100.0 * k
        rows += [(b + 30, 90, 4), (b + 60, 21, 4), (b + 63, 90, 4), (b + 67, 22, 4)]
    rows += [(400, 21, 4), (407, 22, 4)]              # an uncalled walk
    d, _ = ped_delay(_ev(rows))
    return d, summarize_ped_delay(d, bin_len=5)


def _wait_frames():
    rows = []
    for k in range(5):
        b = 100.0 * k
        term = 4 if k % 2 == 0 else 5
        rows += [(b + 60, 1, 4), (b + 94, term, 4), (b + 94, 8, 4), (b + 98, 9, 4),
                 (b + 98, 10, 4), (b + 100, 11, 4), (b + 120, 43, 4), (b + 165, 44, 4)]
    w = wait_time(_ev(rows))
    return w, summarize_wait_time(w, bin_len=5)


class TestPlots:

    def test_ped_delay_traces(self):
        d, b = _ped_frames()
        fig = plot_ped_delay(d, b, metadata={"intersection_name": "X"})
        assert isinstance(fig, go.Figure)
        names = {t.name for t in fig.data}
        assert {"Ph4 Delay", "Ph4 In Walk", "Ph4 Uncalled Walk", "Ph4 Average"} <= names
        delay = next(t for t in fig.data if t.name == "Ph4 Delay")
        assert list(delay.y) == pytest.approx([30.0] * 4)
        iw = next(t for t in fig.data if t.name == "Ph4 In Walk")
        assert len(iw.x) == 4 and set(iw.y) == {0.0}
        unc = next(t for t in fig.data if t.name == "Ph4 Uncalled Walk")
        assert len(unc.x) == 1

    def test_wait_time_traces(self):
        w, b = _wait_frames()
        fig = plot_wait_time(w, b, metadata={"intersection_name": "X"})
        names = {t.name for t in fig.data}
        assert {"Ph4 Gap Out", "Ph4 Max Out", "Ph4 Average"} <= names
        assert "Ph4 Force Off" not in names and "Ph4 Unknown" not in names
        go_ = next(t for t in fig.data if t.name == "Ph4 Gap Out")
        mo = next(t for t in fig.data if t.name == "Ph4 Max Out")
        # 4 timed waits of 40 s: greens after reds following gap-out and
        # max-out greens alternately; the last red has no green.
        assert len(go_.y) + len(mo.y) == 4
        assert set(go_.y) | set(mo.y) == {40.0}

    def test_wait_time_hides_waits_over_max(self):
        w, b = _wait_frames()
        fig = plot_wait_time(w, b, max_wait=30.0)
        assert not any(t.name in ("Ph4 Gap Out", "Ph4 Max Out") for t in fig.data)

    def test_titles_use_metadata(self):
        d, b = _ped_frames()
        meta = {"intersection_name": "X", "major_road_name": "Main St", "minor_road_name": "Side St"}
        f1 = plot_ped_delay(d, b, metadata=meta)
        assert "Main St" in f1.layout.title.text and "Pedestrian Delay" in f1.layout.title.text
        w, wb = _wait_frames()
        f2 = plot_wait_time(w, wb, metadata=meta)
        assert "Main St" in f2.layout.title.text and "Wait Time" in f2.layout.title.text

    def test_empty_frames_give_figure(self):
        assert isinstance(plot_ped_delay(pd.DataFrame(), pd.DataFrame()), go.Figure)
        assert isinstance(plot_wait_time(pd.DataFrame(), pd.DataFrame()), go.Figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    @staticmethod
    def _parse(cmd, *argv):
        return cli._build_parser().parse_args([cmd, *argv])

    def test_ped_delay_parses_with_defaults(self):
        args = self._parse("ped-delay", "--targetid", "900", "--start", START, "--end", END)
        assert args.func is cli.handle_ped_delay
        assert args.bin_len == 60 and args.no_plot is False and args.phases is None

    def test_wait_time_parses_with_defaults(self):
        args = self._parse("wait-time", "--targetid", "900", "--start", START, "--end", END)
        assert args.func is cli.handle_wait_time
        assert args.dropping == "auto"
        assert args.max_wait == 360.0
        assert args.bin_len == 15 and args.no_plot is False and args.phases is None

    def test_dropping_choices(self):
        assert self._parse("wait-time", "--targetid", "900", "--start", DAY, "--end", DAY,
                           "--dropping", "off").dropping == "off"
        with pytest.raises(SystemExit):
            self._parse("wait-time", "--targetid", "900", "--start", DAY, "--end", DAY,
                        "--dropping", "sometimes")

    @pytest.mark.parametrize("cmd", ["ped-delay", "wait-time"])
    def test_target_group_is_required_and_exclusive(self, cmd):
        with pytest.raises(SystemExit):
            self._parse(cmd, "--start", DAY, "--end", DAY)
        with pytest.raises(SystemExit):
            self._parse(cmd, "--all", "--targetid", "900", "--start", DAY, "--end", DAY)

    @staticmethod
    def _site(tmp_path, monkeypatch, **kw):
        folder = tmp_path / "intersections" / "900_Synthetic"
        folder.mkdir(parents=True)
        _build_db(folder, **kw)
        (folder / "metadata.json").write_text(json.dumps({
            "intersection_name": "Synthetic", "intersection_id": "900",
            "db_filename": "cs.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        return folder

    def test_ped_delay_end_to_end(self, tmp_path, monkeypatch):
        folder = self._site(tmp_path, monkeypatch)
        args = self._parse("ped-delay", "--targetid", "900", "--start", START, "--end", END,
                           "--no-plot")
        args.func(args)
        pl = pd.read_csv(folder / "outputs" / f"PedDelay_Plans_{STAMP}.csv")
        assert pl["n_called"].sum() == 72 and pl["avg_delay_s"].iloc[0] == pytest.approx(30.0)
        assert not list((folder / "outputs").glob("*.html"))

    def test_ped_delay_not_computable_does_not_raise(self, tmp_path, monkeypatch, capsys):
        folder = self._site(tmp_path, monkeypatch, ped=False)
        args = self._parse("ped-delay", "--targetid", "900", "--start", START, "--end", END)
        args.func(args)
        assert "Code 21" in capsys.readouterr().out
        assert not list((folder / "outputs").glob("PedDelay_*")) if (folder / "outputs").exists() else True

    def test_wait_time_end_to_end(self, tmp_path, monkeypatch):
        folder = self._site(tmp_path, monkeypatch)
        args = self._parse("wait-time", "--targetid", "900", "--start", START, "--end", END,
                           "--phases", "4", "--dropping", "off", "--max-wait", "0", "--no-plot")
        args.func(args)
        pl = pd.read_csv(folder / "outputs" / f"WaitTime_Plans_{STAMP}.csv")
        assert set(pl["phase"]) == {4}
        assert pl["n_called"].sum() == 72 and pl["avg_wait_s"].iloc[0] == pytest.approx(40.0)
        assert not list((folder / "outputs").glob("*.html"))
