# Acceptance tests for the yellow/red actuations shell engine, plot and CLI
# (UDOT S-M4; spec docs/specs/yellow_red_actuations_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The measure itself is pinned in tests/analysis/test_yellow_red_actuations.py.

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.yellow_red_actuations import (
    ACTUATION_SCHEMA,
    CYCLE_SCHEMA,
    SUMMARY_SCHEMA,
    yellow_red_actuations,
)
from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor
from atspm.utils.timezone import to_epoch

# Built by the delegated S-M4 run; these imports fail until they exist.
from atspm.data.yellow_red_actuations import (
    _ALL_YRA_CODES,
    YellowRedEngine,
    get_yellow_red,
)
from atspm.plotting.yellow_red_actuations import plot_yellow_red

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 09:00"
STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection: one ring 2|4, 100 s cycles from 06:50 to 09:10.
#   P2  green 0, yellow 54, end yellow 58 (9, 10), end red clearance 60 (11).
#       Stop-bar loops 26, 27; presence zone 50.  Per cycle:
#         26 @ +10  green
#         26 @ +56  yellow     (t_yellow 2)
#         27 @ +59  red_clear  (t_red 1)
#         27 @ +70  red        (t_red 12 → severe at the default 4 s)
#         50 @ +80  red        (presence zone only)
#   P4  green 60, yellow 94, end yellow 98, end red clearance 100.
#       Det_P4_Overlap = "B": overlap 2 is green 60 → 97, yellow 97 → 99,
#       red clearance 99 → 100.  Loop 31 @ +95 is yellow for the phase but
#       green for the overlap.
# Variants: a comms gap at 08:30 + 75 s (inside P2's red), and
# TM_Exclusions dropping loop 27 while P2 is Red.
# ---------------------------------------------------------------------------

_CYCLE = 100.0
_N = 84                                  # 06:50 → 09:10
_P2_ACTS = [(10.0, 26), (56.0, 26), (59.0, 27), (70.0, 27), (80.0, 50)]


def _local(hhmm: str) -> float:
    return to_epoch(datetime.strptime(f"{DAY} {hhmm}", "%Y-%m-%d %H:%M"), TZ)


def _build_db(root: Path, gap: bool = False, exclusions: bool = False) -> Path:
    t0 = _local("06:50")
    events = []
    for c in range(_N):
        b = t0 + c * _CYCLE
        events += [(b, 1, 2), (b + 54, 8, 2), (b + 58, 9, 2), (b + 58, 10, 2),
                   (b + 60, 11, 2), (b + 60, 12, 2)]
        events += [(b + 60, 1, 4), (b + 94, 8, 4), (b + 98, 9, 4), (b + 98, 10, 4),
                   (b + 100, 11, 4), (b + 100, 12, 4)]
        events += [(b + 60, 61, 2), (b + 97, 63, 2), (b + 99, 64, 2), (b + 100, 65, 2)]
        for off, det in _P2_ACTS:
            events += [(b + off, 82, det), (b + off + 0.3, 81, det)]
        events += [(b + 95, 82, 31), (b + 95.3, 81, 31)]
    if gap:
        events.append((_local("08:30") + 75.0, -1, -1))
    db = root / "yra.db"
    exc = [{"detector": 27, "phase": 2, "status": "Red"}] if exclusions else []
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        cfg = {
            "RB_R1": "2|4",
            "Det_P2_Stop_Bar": "26,27",
            "Det_P2_Occupancy": "50",
            "Det_P4_Stop_Bar": "31",
            "Det_P4_Overlap": "B",
            "TM_Exclusions": json.dumps(exc),
        }
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
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_codes(self):
        assert set(_ALL_YRA_CODES) == {-1, 1, 8, 9, 10, 11, 12, 61, 63, 64, 65, 82}

    def test_result_keys_and_columns(self, db):
        res = YellowRedEngine(db).yellow_red(START, END)
        assert set(res) == {"cycle", "actuations", "binned", "plans"}
        assert list(res["cycle"].columns) == CYCLE_SCHEMA
        assert list(res["actuations"].columns) == ACTUATION_SCHEMA
        assert list(res["binned"].columns) == SUMMARY_SCHEMA
        assert list(res["plans"].columns) == SUMMARY_SCHEMA
        assert set(res["cycle"]["phase"]) == {2, 4}

    def test_window_by_green(self, db):
        res = YellowRedEngine(db).yellow_red(START, END)
        w0, w1 = _local("07:00"), _local("09:00")
        p2 = _ph(res["cycle"], 2)
        assert len(p2) == 72
        g = _epoch(p2["green_ts"])
        assert (g >= w0).all() and (g < w1).all()
        assert not p2["censored"].any()
        # Actuations belong to the kept cycles only.
        assert set(_epoch(res["actuations"]["green_ts"])) <= set(_epoch(res["cycle"]["green_ts"]))

    def test_stop_bar_counts(self, db):
        res = YellowRedEngine(db).yellow_red(START, END, phases=[2])
        p2 = res["cycle"]
        assert p2["volume"].sum() == 4 * 72
        assert p2["yellow_act"].sum() == 72
        assert p2["red_clear_act"].sum() == 72
        assert p2["red_act"].sum() == 72
        assert p2["violations"].sum() == 144
        assert p2["severe"].sum() == 72
        assert set(res["actuations"]["detector"]) == {26, 27}

    def test_matches_the_core_on_the_same_events(self, db):
        from atspm.data.reader import get_events_with_cycles_df
        res = YellowRedEngine(db).yellow_red(START, END, phases=[2])
        s = datetime.strptime(START, "%Y-%m-%d %H:%M")
        e = datetime.strptime(END, "%Y-%m-%d %H:%M")
        ev = get_events_with_cycles_df(db, s - timedelta(hours=1), e + timedelta(hours=1),
                                       event_codes=_ALL_YRA_CODES, timezone=TZ)
        core, _ = yellow_red_actuations(ev, 2, [26, 27])
        core = core.loc[(core["green_ts"] >= pd.Timestamp(START, tz=TZ))
                        & (core["green_ts"] < pd.Timestamp(END, tz=TZ))].reset_index(drop=True)
        got = res["cycle"].reset_index(drop=True)
        assert len(got) == len(core)
        for col in ("volume", "violations", "severe", "red_dur"):
            np.testing.assert_allclose(got[col].to_numpy(float), core[col].to_numpy(float))

    def test_occupancy_role(self, db):
        res = YellowRedEngine(db).yellow_red(START, END, phases=[2], role="occupancy")
        p2 = res["cycle"]
        assert p2["volume"].sum() == 72 and p2["red_act"].sum() == 72
        assert set(res["actuations"]["detector"]) == {50}

    def test_unknown_role_raises(self, db):
        with pytest.raises(ValueError):
            YellowRedEngine(db).yellow_red(START, END, role="arrival")

    def test_severe_sec(self, db):
        res = YellowRedEngine(db).yellow_red(START, END, phases=[2], severe_sec=15.0)
        assert res["cycle"]["severe"].sum() == 0
        assert not res["actuations"]["severe"].any()

    def test_overlap_from_config(self, db):
        p4 = YellowRedEngine(db).yellow_red(START, END, phases=[4])["cycle"]
        assert (p4["overlap"] == 2).all()
        # Loop 31 @ +95 is green for overlap B (the phase is already yellow).
        assert p4["green_act"].sum() == len(p4) and p4["yellow_act"].sum() == 0

    def test_exclusions_from_config(self, tmp_path):
        db = _build_db(tmp_path, exclusions=True)
        p2 = YellowRedEngine(db).yellow_red(START, END, phases=[2])["cycle"]
        assert p2["violations"].sum() == 0 and p2["yellow_act"].sum() == 72
        p2_all = YellowRedEngine(db).yellow_red(START, END, phases=[2], use_exclusions=False)["cycle"]
        assert p2_all["violations"].sum() == 144

    def test_gap_censors_the_cycle(self, tmp_path):
        db = _build_db(tmp_path, gap=True)
        res = YellowRedEngine(db).yellow_red(START, END, phases=[2])
        p2 = res["cycle"]
        cut = p2["green_ts"] == pd.Timestamp(f"{DAY} 08:30", tz=TZ)
        assert cut.sum() == 1 and p2.loc[cut, "censored"].all()
        assert p2.loc[cut, "violations"].isna().all()
        assert p2.loc[~cut, "violations"].sum() == 142
        assert (res["actuations"]["green_ts"] != pd.Timestamp(f"{DAY} 08:30", tz=TZ)).all()
        assert res["plans"]["n_censored"].sum() == 1

    def test_binned_and_plans(self, db):
        res = YellowRedEngine(db).yellow_red(START, END, phases=[2], bin_len=30)
        b = res["binned"]
        assert len(b) == 4 and b["n_cycles"].sum() == 72
        assert (b["time"].diff().dropna() == pd.Timedelta(minutes=30)).all()
        pl = res["plans"]
        assert len(pl) == 1
        r = pl.iloc[0]
        assert (r["n_cycles"], r["violations"], r["severe"]) == (72, 144, 72)
        assert r["pct_violations"] == pytest.approx(0.5)
        assert r["violations_per_cycle"] == pytest.approx(2.0)
        assert r["pct_violations_udot"] == pytest.approx(2 / 3, abs=1e-4)

    def test_phases_filter_and_warning(self, db, capsys):
        res = YellowRedEngine(db).yellow_red(START, END, phases=[2, 6])
        assert "Ph6" in capsys.readouterr().out
        assert set(res["cycle"]["phase"]) == {2}

    def test_no_detectors_for_role_returns_empty(self, db):
        res = YellowRedEngine(db).yellow_red(START, END, phases=[6])
        assert res == {}

    def test_no_cycles_returns_empty(self, db):
        res = YellowRedEngine(db).yellow_red(f"{DAY} 12:00", f"{DAY} 13:00")
        assert res == {}

    def test_date_only_end_is_whole_day(self, db):
        cyc = YellowRedEngine(db).yellow_red(DAY, DAY, phases=[2])["cycle"]
        # The last green (09:08:20) has no next green: present but censored.
        assert len(cyc) == _N and cyc["censored"].sum() == 1

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert YellowRedEngine(db).yellow_red(START, END, output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {
            f"YRA_Cycle_{STAMP}.csv",
            f"YRA_Actuations_{STAMP}.csv",
            f"YRA_15min_{STAMP}.csv",
            f"YRA_Plans_{STAMP}.csv",
            f"YRA_Chart_{STAMP}.html",
        }
        assert list(pd.read_csv(out / f"YRA_Cycle_{STAMP}.csv").columns) == CYCLE_SCHEMA
        assert list(pd.read_csv(out / f"YRA_Actuations_{STAMP}.csv").columns) == ACTUATION_SCHEMA

    def test_no_plot_writes_no_html(self, db, tmp_path):
        out = tmp_path / "out"
        YellowRedEngine(db).yellow_red(START, END, make_plot=False, output_dir=out)
        assert not list(out.glob("*.html"))

    def test_convenience_wrapper(self, db):
        res = get_yellow_red(db, START, END, phases=[2], role="occupancy", timezone=TZ)
        assert set(res["actuations"]["detector"]) == {50}


# ---------------------------------------------------------------------------
# Plot (functional core: pure)
# ---------------------------------------------------------------------------


def _frames(severe_sec=4.0):
    t0 = to_epoch(datetime.strptime(f"{DAY} 07:00", "%Y-%m-%d %H:%M"), TZ)
    rows = []
    for k in range(5):
        b = k * _CYCLE
        rows += [(b, 1, 2), (b + 54, 8, 2), (b + 58, 9, 2), (b + 58, 10, 2), (b + 60, 11, 2)]
        rows += [(b + off, 82, det) for off, det in _P2_ACTS]
    ev = pd.DataFrame(rows, columns=["t", "event_code", "parameter"])
    ev["timestamp"] = pd.to_datetime(t0 + ev["t"], unit="s", utc=True).dt.tz_convert(TZ)
    ev["cycle_start"] = ev["timestamp"]
    ev["coord_plan"] = 1.0
    ev = ev.sort_values("timestamp", kind="stable").drop(columns="t").reset_index(drop=True)
    return yellow_red_actuations(ev, 2, [26, 27], severe_sec=severe_sec)


class TestPlot:

    def test_traces_per_category(self):
        cy, ac = _frames()
        fig = plot_yellow_red(cy, ac, metadata={"intersection_name": "X"})
        assert isinstance(fig, go.Figure)
        names = {t.name for t in fig.data}
        assert {"Ph2 Yellow", "Ph2 Red Clearance", "Ph2 Severe",
                "Ph2 Red Clearance Begin", "Ph2 Red Begin", "Ph2 Severe Threshold"} <= names
        assert "Ph2 Red" not in names            # every red actuation here is severe
        assert not any("Green" in (t.name or "") for t in fig.data)
        # 4 uncensored cycles (the 5th has no next green).
        yel = next(t for t in fig.data if t.name == "Ph2 Yellow")
        assert list(yel.y) == pytest.approx([2.0] * 4)
        sev = next(t for t in fig.data if t.name == "Ph2 Severe")
        assert list(sev.y) == pytest.approx([16.0] * 4)    # t_yellow of +70

    def test_reference_lines(self):
        cy, ac = _frames()
        fig = plot_yellow_red(cy, ac, severe_sec=4.0)
        rc = next(t for t in fig.data if t.name == "Ph2 Red Clearance Begin")
        rb = next(t for t in fig.data if t.name == "Ph2 Red Begin")
        st = next(t for t in fig.data if t.name == "Ph2 Severe Threshold")
        assert set(v for v in rc.y if v is not None) == {4.0}
        assert set(v for v in rb.y if v is not None) == {6.0}
        assert set(v for v in st.y if v is not None) == {8.0}

    def test_severe_sec_moves_the_threshold(self):
        cy, ac = _frames(severe_sec=15.0)
        fig = plot_yellow_red(cy, ac, severe_sec=15.0)
        names = {t.name for t in fig.data}
        assert "Ph2 Red" in names and "Ph2 Severe" not in names
        st = next(t for t in fig.data if t.name == "Ph2 Severe Threshold")
        assert set(v for v in st.y if v is not None) == {19.0}

    def test_title_uses_metadata(self):
        cy, ac = _frames()
        fig = plot_yellow_red(cy, ac, metadata={
            "intersection_name": "X", "major_road_name": "Main St",
            "minor_road_name": "Side St"})
        assert "Main St" in fig.layout.title.text
        assert "Yellow and Red Actuations" in fig.layout.title.text

    def test_empty_frames_give_figure(self):
        assert isinstance(plot_yellow_red(pd.DataFrame(), pd.DataFrame()), go.Figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    @staticmethod
    def _parse(*argv):
        return cli._build_parser().parse_args(["yellow-red", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "900", "--start", START, "--end", END)
        assert args.func is cli.handle_yellow_red
        assert args.role == "stop_bar"
        assert args.severe_sec == 4.0
        assert args.bin_len == 15
        assert args.no_exclusions is False
        assert args.no_plot is False
        assert args.phases is None

    def test_role_choices(self):
        assert self._parse("--targetid", "900", "--start", DAY, "--end", DAY,
                           "--role", "occupancy").role == "occupancy"
        with pytest.raises(SystemExit):
            self._parse("--targetid", "900", "--start", DAY, "--end", DAY, "--role", "arrival")

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
            "db_filename": "yra.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--phases", "2", "--severe-sec", "15", "--no-plot")
        args.func(args)
        pl = pd.read_csv(folder / "outputs" / f"YRA_Plans_{STAMP}.csv")
        assert set(pl["phase"]) == {2}
        assert pl["violations"].sum() == 144 and pl["severe"].sum() == 0
        assert not list((folder / "outputs").glob("*.html"))
