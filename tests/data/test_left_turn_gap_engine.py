# Acceptance tests for the left-turn gap shell engine, plot and CLI
# (UDOT S-M9; spec docs/specs/left_turn_gap_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The measure itself is pinned in tests/analysis/test_left_turn_gap.py.

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.counts import parse_exclusions_from_config
from atspm.analysis.left_turn_gap import (
    DEFAULT_EDGES,
    GAP_SCHEMA,
    PAIR_SCHEMA,
    bin_columns,
    cycle_schema,
    left_turn_gaps,
    left_turn_pairs,
    summary_schema,
)
from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor
from atspm.data.reader import get_events_with_cycles_df
from atspm.utils.timezone import to_epoch

# Built by the delegated S-M9 run; these imports fail until they exist.
from atspm.data.left_turn_gap import (
    FETCH_MARGIN_S,
    _LTG_CODES,
    LeftTurnGapEngine,
    get_left_turn_gap,
)
from atspm.plotting.left_turn_gap import plot_left_turn_gap

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 09:00"
STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection.  A 120 s cycle from 06:50 for 75 cycles (→ 09:20):
#   Ph6 (WB through)  green +0, yellow +40, red clr +44, end red clr +46
#   Ph4 (SB through)  green +60, yellow +80, red clr +84, end red clr +86
# WB offs (Code 81, each after an 82 0.3 s earlier): det 18 at +5, +7, +15
# and det 21 (WBR) at +30.  EBL's window is [0, 44): gaps 5, 2, 8, 15, 14
#   → n_short 0 · bin_1 1 · bin_2 0 · bin_3 1 · bin_4 3, turnable 37 s
#   (84.091 %), 1 opposing lane → critical 4.1: sum 42 s over 4 gaps.
# Config: TM_EBL 25, TM_WBT 18, TM_WBR 21 and Det_P6_Stop_Bar 18 derive
# WB = Ph6.  TM_NBL 22 / TM_SBT 31 has no Det_P key: SB is unresolved
# unless the explicit variant adds Det_P4_Direction = SB (then NBL's window
# is Ph4's [60, 84): one 24 s gap per green, and det 31 never logs).
# 07:00–09:00 holds 60 Ph6 greens (07:00, 07:02, … 08:58), 8 or 7 per
# 15-min bin.
# Variants: a gap marker at 08:00:10 (censors the 08:00 green); TM_EBL
# also listing det 21 (shared with WBR); no TM_*L keys at all.
# ---------------------------------------------------------------------------

_CYCLE = 120.0
_N = 75
_GREENS_IN_WINDOW = 60


def _local(hhmm: str) -> float:
    return to_epoch(datetime.strptime(f"{DAY} {hhmm}", "%Y-%m-%d %H:%M"), TZ)


def _build_db(root: Path, gap: bool = False, explicit: bool = False,
              shared: bool = False, lefts: bool = True) -> Path:
    t0 = _local("06:50")
    events = []
    for c in range(_N):
        b = t0 + c * _CYCLE
        events += [(b, 1, 6), (b + 40, 8, 6), (b + 44, 9, 6), (b + 44, 10, 6), (b + 46, 11, 6)]
        events += [(b + 60, 1, 4), (b + 80, 8, 4), (b + 84, 9, 4), (b + 84, 10, 4),
                   (b + 86, 11, 4)]
        for t, d in [(5, 18), (7, 18), (15, 18), (30, 21)]:
            events += [(b + t - 0.3, 82, d), (b + t, 81, d)]
    if gap:
        events.append((_local("08:00") + 10.0, -1, -1))
    db = root / "ltg.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        cfg = {"RB_R1": "6|4", "TM_WBT": "18", "TM_WBR": "21", "TM_SBT": "31",
               "TM_Exclusions": "[]", "Det_P6_Stop_Bar": "18"}
        if lefts:
            cfg.update({"TM_EBL": "25,21" if shared else "25", "TM_NBL": "22"})
        if explicit:
            cfg["Det_P4_Direction"] = "SB"
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00",
                              "end_date": None, **cfg})
        m.insert_events(sorted(events))
        m.conn.execute(
            "INSERT INTO ingestion_log (span_start, span_end, processed_at, "
            "row_count) VALUES (?, ?, '2025-06-03T00:00:00', ?)",
            (t0, t0 + _N * _CYCLE, len(events)),
        )
        m.conn.commit()
    CycleProcessor(db, TZ).run()
    return db


@pytest.fixture
def db(tmp_path) -> Path:
    return _build_db(tmp_path)


def _left(df, left):
    return df.loc[df["left"] == left].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_constants(self):
        assert set(_LTG_CODES) == {-1, 1, 8, 9, 10, 11, 12, 81}
        assert FETCH_MARGIN_S == 3600.0

    def test_result_keys_and_columns(self, db):
        res = LeftTurnGapEngine(db).left_turn_gap(START, END)
        assert set(res) == {"pairs", "cycles", "gaps", "bins", "summary"}
        assert list(res["pairs"].columns) == PAIR_SCHEMA
        assert list(res["cycles"].columns) == cycle_schema()
        assert list(res["gaps"].columns) == GAP_SCHEMA
        assert list(res["bins"].columns) == summary_schema()
        assert list(res["summary"].columns) == summary_schema()

    def test_pairs_are_the_core_pairs(self, db):
        res = LeftTurnGapEngine(db).left_turn_gap(START, END)
        with DatabaseManager(db) as m:
            cfg = m.get_config_at_date(datetime(2025, 6, 2, 7))
        pd.testing.assert_frame_equal(res["pairs"].reset_index(drop=True),
                                      left_turn_pairs(cfg))

    def test_ebl_values(self, db):
        res = LeftTurnGapEngine(db).left_turn_gap(START, END)
        cy = _left(res["cycles"], "EBL")
        assert len(cy) == _GREENS_IN_WINDOW and not cy["censored"].any()
        assert (cy["opposing_phase"] == 6).all()
        assert cy["green_ts"].min() == pd.Timestamp(START, tz=TZ)
        assert cy["green_ts"].max() == pd.Timestamp(f"{DAY} 08:58", tz=TZ)
        assert (cy["window_s"] == 44.0).all()
        assert (cy["n_actuations"] == 4).all() and (cy["n_gaps"] == 5).all()
        assert list(cy.loc[0, ["n_short", *bin_columns()]]) == [0, 1, 0, 1, 3]
        assert cy.loc[0, "turnable_s"] == pytest.approx(37.0)
        assert cy.loc[0, "pct_turnable"] == pytest.approx(100 * 37 / 44, abs=1e-3)
        assert cy.loc[0, "sum_ge_critical"] == pytest.approx(42.0)
        assert cy.loc[0, "n_ge_critical"] == 4
        assert len(_left(res["gaps"], "EBL")) == 5 * _GREENS_IN_WINDOW

    def test_matches_the_core(self, db):
        res = LeftTurnGapEngine(db).left_turn_gap(START, END)
        with DatabaseManager(db) as m:
            cfg = m.get_config_at_date(datetime(2025, 6, 2, 7))
        ev = get_events_with_cycles_df(
            db_path=db, start=datetime(2025, 6, 2, 6), end=datetime(2025, 6, 2, 10),
            event_codes=list(_LTG_CODES), timezone=TZ)
        cy, gp = left_turn_gaps(ev, 6, [18, 21], left="EBL", critical_s=4.1,
                                exclusions=parse_exclusions_from_config(cfg))
        keep = (cy["green_ts"] >= pd.Timestamp(START, tz=TZ)) & \
               (cy["green_ts"] < pd.Timestamp(END, tz=TZ))
        pd.testing.assert_frame_equal(_left(res["cycles"], "EBL"),
                                      cy.loc[keep].reset_index(drop=True))

    def test_bins_and_summary(self, db):
        res = LeftTurnGapEngine(db).left_turn_gap(START, END)
        b = _left(res["bins"], "EBL")
        assert len(b) == 8
        assert b["time"].min() == pd.Timestamp(START, tz=TZ)
        assert list(b["n_cycles"]) == [8, 7] * 4
        assert (b["bin_4"] == 3 * b["n_cycles"]).all()
        s = _left(res["summary"], "EBL")
        assert len(s) == 1 and s.loc[0, "n_cycles"] == _GREENS_IN_WINDOW
        assert s.loc[0, "pct_turnable"] == pytest.approx(100 * 37 / 44, abs=1e-3)

    def test_unresolved_left_is_skipped_with_a_hint(self, db, capsys):
        res = LeftTurnGapEngine(db).left_turn_gap(START, END)
        out = capsys.readouterr().out
        assert "NBL" in out and "Det:,P{N} Direction,SB" in out
        assert set(res["cycles"]["left"]) == {"EBL"}
        assert set(res["pairs"]["left"]) == {"NBL", "EBL"}

    def test_explicit_direction_key(self, tmp_path, capsys):
        res = LeftTurnGapEngine(_build_db(tmp_path, explicit=True)).left_turn_gap(START, END)
        cy = _left(res["cycles"], "NBL")
        assert len(cy) == _GREENS_IN_WINDOW and (cy["opposing_phase"] == 4).all()
        assert (cy["window_s"] == 24.0).all() and (cy["bin_4"] == 1).all()
        assert (cy["pct_turnable"] == 100.0).all()
        assert "SB detector(s) 31 logged no actuation" in capsys.readouterr().out

    def test_lefts_filter(self, tmp_path, capsys):
        eng = LeftTurnGapEngine(_build_db(tmp_path, explicit=True))
        res = eng.left_turn_gap(START, END, lefts=["NBL"])
        assert set(res["cycles"]["left"]) == {"NBL"}
        res = eng.left_turn_gap(START, END, lefts=["SBL", "EBL"])
        assert set(res["cycles"]["left"]) == {"EBL"}
        assert "SBL" in capsys.readouterr().out

    def test_gap_marker_censors_its_green(self, tmp_path):
        res = LeftTurnGapEngine(_build_db(tmp_path, gap=True)).left_turn_gap(START, END)
        cy = _left(res["cycles"], "EBL")
        cen = cy.loc[cy["censored"]]
        assert list(cen["green_ts"]) == [pd.Timestamp(f"{DAY} 08:00", tz=TZ)]
        assert cen["n_gaps"].isna().all()
        assert _left(res["summary"], "EBL").loc[0, "n_censored"] == 1
        assert len(_left(res["gaps"], "EBL")) == 5 * (_GREENS_IN_WINDOW - 1)

    def test_shared_detector_warning(self, tmp_path, capsys):
        LeftTurnGapEngine(_build_db(tmp_path, shared=True)).left_turn_gap(START, END)
        assert "EBL: opposing detector(s) 21 also configured under another direction" \
            in capsys.readouterr().out

    def test_edges_trend_and_critical_override(self, db):
        edges = (1.0, 6.0, 10.0, math.inf)
        res = LeftTurnGapEngine(db).left_turn_gap(START, END, edges=edges, trend_s=15.0,
                                                  critical_s=10.0)
        cy = _left(res["cycles"], "EBL")
        assert list(cy.columns) == cycle_schema(edges)
        # gaps 5, 2, 8, 15, 14
        assert list(cy.loc[0, ["n_short", *bin_columns(edges)]]) == [0, 2, 1, 2]
        assert cy.loc[0, "turnable_s"] == pytest.approx(15.0)
        assert cy.loc[0, "sum_ge_critical"] == pytest.approx(29.0)
        assert list(res["bins"].columns) == summary_schema(edges)

    @pytest.mark.parametrize("kw", [{"bin_len": 7}, {"bin_len": 0},
                                    {"edges": (3.0, 1.0)}, {"edges": (1.0,)}])
    def test_bad_arguments_raise_before_the_db(self, tmp_path, kw):
        with pytest.raises(ValueError):
            LeftTurnGapEngine(tmp_path / "absent.db", timezone=TZ).left_turn_gap(
                START, END, **kw)

    def test_no_left_config_returns_empty(self, tmp_path, capsys):
        eng = LeftTurnGapEngine(_build_db(tmp_path, lefts=False))
        assert eng.left_turn_gap(START, END) == {}
        assert "TM_" in capsys.readouterr().out
        out = tmp_path / "out"
        assert eng.left_turn_gap(START, END, output_dir=out) is None
        assert not out.exists() or not list(out.iterdir())

    def test_no_events_returns_empty(self, db):
        assert LeftTurnGapEngine(db).left_turn_gap("2025-07-01", "2025-07-01") == {}

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert LeftTurnGapEngine(db).left_turn_gap(START, END, output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {f"LTG_Pairs_{STAMP}.csv", f"LTG_Cycles_{STAMP}.csv",
                         f"LTG_Bins_15min_{STAMP}.csv", f"LTG_Summary_{STAMP}.csv",
                         f"LTG_Chart_{STAMP}.html"}
        pairs = pd.read_csv(out / f"LTG_Pairs_{STAMP}.csv", dtype={"detectors": str})
        assert list(pairs.columns) == PAIR_SCHEMA
        assert pairs.set_index("left").at["EBL", "detectors"] == "18,21"
        cy = pd.read_csv(out / f"LTG_Cycles_{STAMP}.csv")
        assert list(cy.columns) == cycle_schema() and len(cy) == _GREENS_IN_WINDOW
        s = pd.read_csv(out / f"LTG_Summary_{STAMP}.csv")
        assert list(s.columns) == summary_schema()

    def test_gaps_csv_and_no_plot(self, db, tmp_path):
        out = tmp_path / "out"
        LeftTurnGapEngine(db).left_turn_gap(START, END, write_gaps=True, make_plot=False,
                                            output_dir=out)
        names = {p.name for p in out.iterdir()}
        assert f"LTG_Gaps_{STAMP}.csv" in names
        assert not any(n.endswith(".html") for n in names)
        g = pd.read_csv(out / f"LTG_Gaps_{STAMP}.csv")
        assert list(g.columns) == GAP_SCHEMA and len(g) == 5 * _GREENS_IN_WINDOW

    def test_convenience_wrapper(self, db):
        res = get_left_turn_gap(db, START, END, timezone=TZ)
        assert _left(res["summary"], "EBL").loc[0, "n_cycles"] == _GREENS_IN_WINDOW


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


class TestPlot:

    def test_stacked_bins_and_trend_line(self, db):
        res = LeftTurnGapEngine(db).left_turn_gap(START, END)
        fig = plot_left_turn_gap(res["bins"], {"intersection_name": "X"})
        assert isinstance(fig, go.Figure)
        bars = [t for t in fig.data if isinstance(t, go.Bar)]
        assert [t.name for t in bars] == ["1-3.3s", "3.3-3.7s", "3.7-7.4s", "7.4s+"]
        assert fig.layout.barmode == "stack"
        b4 = next(t for t in bars if t.name == "7.4s+")
        assert list(b4.y) == [24, 21] * 4
        line = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(line) == 1 and line[0].name == "% green with gaps ≥ 7.4s"
        assert line[0].line.dash == "dash" and line[0].yaxis != b4.yaxis
        assert line[0].y[0] == pytest.approx(100 * 37 / 44, abs=1e-3)

    def test_one_row_per_left_and_one_legend_entry_per_bin(self, tmp_path):
        res = LeftTurnGapEngine(_build_db(tmp_path, explicit=True)).left_turn_gap(START, END)
        fig = plot_left_turn_gap(res["bins"])
        bars = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bars) == 8
        assert sum(1 for t in bars if t.showlegend is not False) == 4
        titles = [a.text for a in fig.layout.annotations]
        assert any("NBL" in t for t in titles) and any("EBL" in t for t in titles)
        # NBL above EBL (NB, SB, EB, WB order).
        assert [t for t in titles if "NBL" in t or "EBL" in t][0].startswith("NBL")

    def test_custom_edges_and_trend(self, db):
        edges = (1.0, 6.0, math.inf)
        res = LeftTurnGapEngine(db).left_turn_gap(START, END, edges=edges, trend_s=10.0)
        fig = plot_left_turn_gap(res["bins"], edges=edges, trend_s=10.0)
        assert [t.name for t in fig.data if isinstance(t, go.Bar)] == ["1-6s", "6s+"]
        assert any(t.name == "% green with gaps ≥ 10s" for t in fig.data)

    def test_censored_only_bin_breaks_the_line(self, tmp_path):
        res = LeftTurnGapEngine(_build_db(tmp_path, gap=True)).left_turn_gap(
            f"{DAY} 08:00", f"{DAY} 08:02", bin_len=1)
        fig = plot_left_turn_gap(res["bins"])
        line = next(t for t in fig.data if isinstance(t, go.Scatter))
        assert any(v is None or np.isnan(v) for v in line.y)
        assert not line.connectgaps

    def test_title_uses_metadata(self, db):
        res = LeftTurnGapEngine(db).left_turn_gap(START, END)
        fig = plot_left_turn_gap(res["bins"], {"major_road_name": "Main St",
                                               "minor_road_name": "Side St"})
        assert "Main St" in fig.layout.title.text
        assert "Left Turn Gap" in fig.layout.title.text

    def test_empty_frame_gives_figure(self):
        fig = plot_left_turn_gap(pd.DataFrame(columns=summary_schema()))
        assert isinstance(fig, go.Figure) and len(fig.data) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    @staticmethod
    def _parse(*argv):
        return cli._build_parser().parse_args(["left-turn-gap", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "900", "--start", DAY, "--end", DAY)
        assert args.func is cli.handle_left_turn_gap
        assert args.bin_len == 15 and args.no_plot is False and args.gaps is False
        assert args.left is None and args.edges is None
        assert args.trend == 7.4 and args.critical is None
        assert args.no_exclusions is False

    def test_parses_options(self):
        args = self._parse("--targetid", "900", "--start", DAY, "--end", DAY,
                           "--left", "EBL", "WBL", "--edges", "1,2.5,5",
                           "--trend", "5", "--critical", "4.5", "--gaps")
        assert args.left == ["EBL", "WBL"]
        assert args.edges == (1.0, 2.5, 5.0, math.inf)
        assert args.trend == 5.0 and args.critical == 4.5 and args.gaps is True

    def test_bad_arguments_exit(self):
        with pytest.raises(SystemExit):
            self._parse("--start", DAY, "--end", DAY)
        with pytest.raises(SystemExit):
            self._parse("--all", "--targetid", "900", "--start", DAY, "--end", DAY)
        with pytest.raises(SystemExit):
            self._parse("--targetid", "900", "--start", DAY, "--end", DAY, "--left", "EBT")
        with pytest.raises(SystemExit):
            self._parse("--targetid", "900", "--start", DAY, "--end", DAY, "--edges", "1,x")

    @staticmethod
    def _site(tmp_path, monkeypatch, **kw):
        folder = tmp_path / "intersections" / "900_Synthetic"
        folder.mkdir(parents=True)
        _build_db(folder, **kw)
        (folder / "metadata.json").write_text(json.dumps({
            "intersection_name": "Synthetic", "intersection_id": "900",
            "db_filename": "ltg.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        return folder

    def test_end_to_end(self, tmp_path, monkeypatch, capsys):
        folder = self._site(tmp_path, monkeypatch)
        args = self._parse("--targetid", "900", "--start", START, "--end", END, "--no-plot")
        args.func(args)
        s = pd.read_csv(folder / "outputs" / f"LTG_Summary_{STAMP}.csv")
        assert list(s["left"]) == ["EBL"]
        assert not list((folder / "outputs").glob("*.html"))
        out = capsys.readouterr().out
        assert "EBL" in out and "Ph6" in out and "84.1" in out

    def test_no_left_config_does_not_raise(self, tmp_path, monkeypatch, capsys):
        folder = self._site(tmp_path, monkeypatch, lefts=False)
        args = self._parse("--targetid", "900", "--start", DAY, "--end", DAY)
        args.func(args)
        assert "TM_" in capsys.readouterr().out
        out = folder / "outputs"
        assert not out.exists() or not list(out.glob("LTG_*"))
