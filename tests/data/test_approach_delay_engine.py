# Acceptance tests for the approach-delay shell engine, plot and CLI
# (UDOT S-M3; spec docs/specs/approach_delay_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The measure itself is pinned in tests/analysis/test_approach_delay.py.

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.approach_delay import CYCLE_SCHEMA, bin_approach_delay
from atspm.data.manager import DatabaseManager
from atspm.data.processing import CycleProcessor
from atspm.utils.timezone import to_epoch

# Built by the delegated S-M3 run; these imports fail until they exist.
from atspm.data.approach_delay import (
    _ALL_AD_CODES,
    ApproachDelayEngine,
    get_approach_delay,
)
from atspm.plotting.approach_delay import plot_approach_delay

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 09:00"
STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection: one ring 2|4, 100 s cycles from 06:50 to 09:10, so
# the 07:00–09:00 window needs the fetch margin to see the red before its
# first green.  Per cycle c (seconds after c):
#   P2  green 0, yellow 54, end yellow 58, red clearance to 60.
#       Detector 7 (P2 Arrival) on at 20, 56, 80.
#   P4  green 60, yellow 94, end yellow 98, red clearance to 100.
#       Detector 8 (P4 Arrival) on at 70, 99.
# With Det_P2_Arrival_Travel = 5 every in-window P2 cycle has 3 arrivals:
# red at −39 and −15 (delay 39 + 15 = 54) and green at +25.
# P4 has no travel key; with --offset 0 each cycle has a red arrival at −1
# (delay 61) and a green one at +70.  With --offset 2 the delay is 59.
# Detector 9 is configured for P8, which never runs.
# ---------------------------------------------------------------------------

_CYCLE = 100.0
_PLAN = {2: (0.0, 54.0), 4: (60.0, 34.0)}
_N = 84                                  # 06:50 → 09:10


def _build_db(root: Path, travel_key: bool = True, arrival_key: str = "Arrival") -> Path:
    t0 = to_epoch(datetime.strptime(f"{DAY} 06:50", "%Y-%m-%d %H:%M"), TZ)
    events = []
    for c in range(_N):
        base = t0 + c * _CYCLE
        for ph, (off, g) in _PLAN.items():
            gs = base + off
            events += [(gs, 1, ph), (gs + g, 8, ph), (gs + g + 4, 9, ph),
                       (gs + g + 4, 10, ph), (gs + g + 6, 11, ph),
                       (gs + g + 6, 12, ph)]
        for t in (20.0, 56.0, 80.0):
            events += [(base + t, 82, 7), (base + t + 0.4, 81, 7)]
        for t in (70.0, 99.0):
            events += [(base + t, 82, 8), (base + t + 0.4, 81, 8)]
    db = root / "ad.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        cfg = {"RB_R1": "2|4", f"Det_P2_{arrival_key}": "7",
               f"Det_P4_{arrival_key}": "8", f"Det_P8_{arrival_key}": "9"}
        if travel_key:
            cfg["Det_P2_Arrival_Travel"] = "5"
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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_codes_include_gap_marker_and_detector_on(self):
        assert set(_ALL_AD_CODES) == {-1, 1, 8, 9, 10, 11, 12, 82}

    def test_result_keys_phases_and_columns(self, db):
        res = ApproachDelayEngine(db).approach_delay(START, END)
        assert set(res) == {"cycle", "binned"}
        cyc = res["cycle"]
        assert set(cyc["phase"]) == {2, 4}
        assert list(cyc.columns) == CYCLE_SCHEMA + ["travel_time_s", "travel_source"]

    def test_window_by_serving_green_and_margin_uncensors_the_first(self, db):
        cyc = ApproachDelayEngine(db).approach_delay(START, END)["cycle"]
        p2 = _ph(cyc, 2)
        assert len(p2) == 72
        assert not p2["censored"].any()
        w0 = to_epoch(datetime.strptime(START, "%Y-%m-%d %H:%M"), TZ)
        first = p2["green_ts"].iloc[0]
        assert first.timestamp() == pytest.approx(w0)
        # Its red arrivals were logged before 07:00 and still count.
        assert int(p2["arrivals_red"].iloc[0]) == 2
        assert len(_ph(cyc, 4)) == 72

    def test_config_travel_time_wins_over_offset(self, db):
        cyc = ApproachDelayEngine(db).approach_delay(START, END, travel_time_sec=2.0)["cycle"]
        p2, p4 = _ph(cyc, 2), _ph(cyc, 4)
        assert (p2["arrivals"].astype(int) == 3).all()
        assert (p2["arrivals_red"].astype(int) == 2).all()
        np.testing.assert_allclose(p2["total_delay_s"].to_numpy(float), 54.0)
        np.testing.assert_allclose(p2["delay_per_veh"].to_numpy(float), 18.0)
        assert set(p2["travel_source"]) == {"config"}
        np.testing.assert_allclose(p2["travel_time_s"].to_numpy(float), 5.0)
        np.testing.assert_allclose(p4["total_delay_s"].to_numpy(float), 59.0)
        assert set(p4["travel_source"]) == {"offset"}
        np.testing.assert_allclose(p4["travel_time_s"].to_numpy(float), 2.0)

    def test_offset_default_is_zero(self, tmp_path):
        db = _build_db(tmp_path, travel_key=False)
        cyc = ApproachDelayEngine(db).approach_delay(START, END)["cycle"]
        p2 = _ph(cyc, 2)
        assert (p2["arrivals_green"].astype(int) == 1).all()
        assert (p2["arrivals_yellow"].astype(int) == 1).all()
        np.testing.assert_allclose(p2["total_delay_s"].to_numpy(float), 20.0)
        np.testing.assert_allclose(_ph(cyc, 4)["total_delay_s"].to_numpy(float), 61.0)

    def test_missing_travel_key_is_warned(self, db, capsys):
        ApproachDelayEngine(db).approach_delay(START, END)
        out = capsys.readouterr().out
        assert "Det_P4_Arrival_Travel" in out
        assert "Det_P2_Arrival_Travel" not in out

    def test_bad_travel_key_raises(self, tmp_path):
        db = _build_db(tmp_path, travel_key=False)
        with DatabaseManager(db) as m:
            m.add_config_column("Det_P2_Arrival_Travel")
            m.conn.execute("UPDATE config SET Det_P2_Arrival_Travel = '5,6'")
            m.conn.commit()
        with pytest.raises(ValueError):
            ApproachDelayEngine(db).approach_delay(START, END)

    def test_binned_matches_core_binning(self, db):
        res = ApproachDelayEngine(db).approach_delay(START, END, bin_len=15)
        exp = bin_approach_delay(res["cycle"][CYCLE_SCHEMA], bin_len=15)
        b = res["binned"]
        assert {"coverage", "data_quality"} <= set(b.columns)
        assert len(b) == len(exp) == 16
        np.testing.assert_allclose(b["delay_per_veh"].to_numpy(float),
                                   exp["delay_per_veh"].to_numpy(float))
        np.testing.assert_allclose(b["delay_vh_per_hr"].to_numpy(float),
                                   exp["delay_vh_per_hr"].to_numpy(float))
        assert b["n_cycles"].tolist() == exp["n_cycles"].tolist()

    def test_phases_filter(self, db):
        res = ApproachDelayEngine(db).approach_delay(START, END, phases=[4])
        assert set(res["cycle"]["phase"]) == {4}

    def test_cycle_mode_has_no_binned(self, db):
        res = ApproachDelayEngine(db).approach_delay(START, END, bin_len="cycle")
        assert set(res) == {"cycle"}

    def test_configured_phase_with_no_cycles_is_warned(self, db, capsys):
        ApproachDelayEngine(db).approach_delay(START, END)
        assert "Ph8" in capsys.readouterr().out

    def test_requested_phase_without_config_is_warned(self, db, capsys):
        res = ApproachDelayEngine(db).approach_delay(START, END, phases=[2, 6])
        assert "Det_P6_Arrival" in capsys.readouterr().out
        assert set(res["cycle"]["phase"]) == {2}

    def test_no_arrival_config_returns_empty(self, tmp_path, capsys):
        db = _build_db(tmp_path, travel_key=False, arrival_key="Occupancy")
        assert ApproachDelayEngine(db).approach_delay(START, END) == {}
        assert "Arrival" in capsys.readouterr().out

    def test_date_only_end_is_whole_day(self, db):
        res = ApproachDelayEngine(db).approach_delay(DAY, DAY, phases=[2])
        # Every green of the day except the first in the data (no red before it).
        cyc = res["cycle"]
        assert len(cyc) == _N
        assert int(cyc["censored"].sum()) == 1

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert ApproachDelayEngine(db).approach_delay(START, END, output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {
            f"AD_Cycle_{STAMP}.csv",
            f"AD_15min_{STAMP}.csv",
            f"AD_Delay_{STAMP}.html",
        }
        cyc = pd.read_csv(out / f"AD_Cycle_{STAMP}.csv")
        assert {"aor_pct", "delay_per_veh", "censored", "travel_source"} <= set(cyc.columns)

    def test_cycle_mode_still_plots(self, db, tmp_path):
        out = tmp_path / "out"
        ApproachDelayEngine(db).approach_delay(START, END, bin_len="cycle", output_dir=out)
        names = {p.name for p in out.iterdir()}
        assert names == {f"AD_Cycle_{STAMP}.csv", f"AD_Delay_{STAMP}.html"}

    def test_no_plot_writes_no_html(self, db, tmp_path):
        out = tmp_path / "out"
        ApproachDelayEngine(db).approach_delay(START, END, make_plot=False, output_dir=out)
        assert not list(out.glob("*.html"))

    def test_convenience_wrapper(self, db):
        res = get_approach_delay(db, START, END, phases=[4], travel_time_sec=2.0, timezone=TZ)
        np.testing.assert_allclose(res["cycle"]["total_delay_s"].to_numpy(float), 59.0)


# ---------------------------------------------------------------------------
# Plot (functional core: pure)
# ---------------------------------------------------------------------------


def _binned(rows):
    """rows: (minute offset, phase, plan, arrivals, total_delay_s)."""
    t0 = pd.Timestamp(f"{DAY} 07:00", tz=TZ)
    df = pd.DataFrame(rows, columns=["m", "phase", "coord_plan", "arrivals", "total_delay_s"])
    df["time"] = t0 + pd.to_timedelta(df["m"], unit="min")
    df["delay_per_veh"] = df["total_delay_s"] / df["arrivals"]
    df["total_delay_vh"] = df["total_delay_s"] / 3600
    df["delay_vh_per_hr"] = df["total_delay_vh"] * 4
    for c in ("arrivals_green", "arrivals_yellow", "arrivals_red", "n_cycles", "n_censored"):
        df[c] = 0
    return df.drop(columns="m")


class TestPlot:

    def test_traces_per_phase_on_two_axes(self):
        b = _binned([(0, 2, 1, 10, 100.0), (15, 2, 1, 10, 200.0), (0, 6, 1, 5, 50.0)])
        fig = plot_approach_delay(b, metadata={"intersection_name": "X"})
        assert isinstance(fig, go.Figure)
        names = {t.name for t in fig.data}
        assert {"Ph2 delay/veh", "Ph2 veh-h/h", "Ph6 delay/veh", "Ph6 veh-h/h"} <= names
        per_veh = next(t for t in fig.data if t.name == "Ph2 delay/veh")
        rate = next(t for t in fig.data if t.name == "Ph2 veh-h/h")
        assert list(per_veh.y) == pytest.approx([10.0, 20.0])
        assert per_veh.yaxis != rate.yaxis                     # secondary y

    def test_a_bin_split_by_plan_is_recombined(self):
        # Same phase and bin under two plans: delay/veh = Σdelay / Σarrivals,
        # veh-h/h = Σ of the rates.
        b = _binned([(0, 2, 1, 10, 100.0), (0, 2, 2, 30, 500.0)])
        fig = plot_approach_delay(b)
        per_veh = next(t for t in fig.data if t.name == "Ph2 delay/veh")
        rate = next(t for t in fig.data if t.name == "Ph2 veh-h/h")
        assert list(per_veh.y) == pytest.approx([15.0])
        assert list(rate.y) == pytest.approx([600.0 / 3600 * 4])

    def test_plan_bands_are_labelled(self):
        b = _binned([(m, 2, 1 if m < 60 else 2, 10, 100.0) for m in range(0, 120, 15)])
        fig = plot_approach_delay(b)
        texts = " ".join(a.text for a in fig.layout.annotations if a.text)
        assert "Plan 1" in texts and "Plan 2" in texts
        assert len(fig.layout.shapes) >= 2

    def test_title_uses_metadata(self):
        b = _binned([(0, 2, 1, 10, 100.0)])
        fig = plot_approach_delay(b, metadata={
            "intersection_name": "X", "major_road_name": "Main St",
            "minor_road_name": "Side St"})
        assert "Main St" in fig.layout.title.text
        assert "Approach Delay" in fig.layout.title.text

    def test_empty_frame_gives_figure(self):
        assert isinstance(plot_approach_delay(pd.DataFrame()), go.Figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    @staticmethod
    def _parse(*argv):
        return cli._build_parser().parse_args(["approach-delay", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "900", "--start", START, "--end", END)
        assert args.func is cli.handle_approach_delay
        assert args.offset == 0.0
        assert args.bin_len == "15"
        assert args.no_plot is False
        assert args.exclude_missing is False
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
            "db_filename": "ad.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--offset", "2", "--phases", "4", "--no-plot")
        args.func(args)
        cyc = pd.read_csv(folder / "outputs" / f"AD_Cycle_{STAMP}.csv")
        assert set(cyc["phase"]) == {4}
        np.testing.assert_allclose(cyc["total_delay_s"].to_numpy(float), 59.0)
        assert not list((folder / "outputs").glob("*.html"))
