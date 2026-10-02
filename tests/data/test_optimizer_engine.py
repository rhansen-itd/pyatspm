# Acceptance tests for the optimizer shell engine, plots and CLI
# (ROADMAP optimizer steps 4-5; spec docs/specs/optimizer_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The solver itself is pinned in tests/analysis/test_optimizer.py.

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.flow import discharge_profiles, flow_rate
from atspm.analysis.optimizer import optimize
from atspm.data.critical import CriticalMovementEngine
from atspm.data.flow import _ALL_FLOW_CODES
from atspm.data.manager import DatabaseManager
from atspm.data.optimizer import OptimizerEngine, get_optimization
from atspm.data.processing import CycleProcessor
from atspm.data.reader import get_events_with_cycles_df
from atspm.plotting.optimizer import (
    plot_allocation,
    plot_marginal_rates,
    plot_throughput_curve,
)
from atspm.utils.timezone import to_epoch

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 09:00"
STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection: two rings, two barrier groups (2|4 over 6|8),
# 100 s cycles for two hours.  Phases 2/6 discharge a queue until the
# force-off; 4/8 serve a few vehicles.  Every green ends by force-off.
# ---------------------------------------------------------------------------

_CYCLE = 100.0
_PLAN = {2: (0.0, 54.0), 6: (0.0, 54.0), 4: (60.0, 34.0), 8: (60.0, 34.0)}
_DETS = {2: [1, 2], 6: [3, 4], 4: [5], 8: [6]}
_QUEUED = {2, 6}


def _departures(green, queued, rng, decay):
    if queued:
        t, h, out = 2.0, 2.0, []
        while t < green - 0.2:
            out.append(t)
            t += h
            h *= decay
        return out
    return list(np.cumsum(rng.uniform(2.0, 6.0, size=4)))


def _build_db(root: Path, decay: float = 1.015, min_split: bool = True) -> Path:
    """Write the synthetic intersection DB.

    Args:
        decay: Headway growth per queued vehicle.  1.015 keeps phases 2/6
            discharging to the force-off (a boundary result); 1.06 makes
            the rate decay so the optimum is interior (with c_min=30).
        min_split: Write Min_P2/P6/P4_Split (P8 is always left out).
    """
    rng = np.random.default_rng(42)
    t0 = to_epoch(datetime.strptime(START, "%Y-%m-%d %H:%M"), TZ)
    n_cycles = int(2 * 3600 / _CYCLE)
    events = []
    for c in range(n_cycles):
        base = t0 + c * _CYCLE
        for ph, (off, g) in _PLAN.items():
            gs = base + off
            events += [(gs, 1, ph), (gs + g, 6, ph), (gs + g, 8, ph),
                       (gs + g + 4, 9, ph), (gs + g + 4, 10, ph),
                       (gs + g + 6, 11, ph), (gs + g + 6, 12, ph)]
            for d in _DETS[ph]:
                for t in _departures(g, ph in _QUEUED, rng, decay):
                    events += [(gs + t - 0.3, 82, d), (gs + t, 81, d)]
    db = root / "opt.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ)
        cfg = {"RB_R1": "2|4", "RB_R2": "6|8",
               "Det_P2_Stop_Bar": "1,2", "Det_P6_Stop_Bar": "3,4",
               "Det_P4_Stop_Bar": "5", "Det_P8_Stop_Bar": "6",
               "TM_EBT": "1,2", "TM_WBT": "3,4", "TM_NBT": "5", "TM_SBT": "6"}
        if min_split:
            cfg.update({"Min_P2_Split": "20", "Min_P6_Split": "20",
                        "Min_P4_Split": "15"})
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
def boundary_db(tmp_path) -> Path:
    return _build_db(tmp_path)


@pytest.fixture
def interior_db(tmp_path) -> Path:
    return _build_db(tmp_path, decay=1.06)


def _expected(db: Path, **kw):
    """The result the engine must reproduce, assembled from the pure core."""
    start = datetime.strptime(START, "%Y-%m-%d %H:%M")
    end = datetime.strptime(END, "%Y-%m-%d %H:%M")
    events = get_events_with_cycles_df(db, start, end, event_codes=_ALL_FLOW_CODES)
    curves = {}
    for ph, dets in {2: [1, 2], 4: [5], 6: [3, 4], 8: [6]}.items():
        cycle_df, vehicle_df = flow_rate(events, ph, dets, max_lost=None)
        _, prof = discharge_profiles(cycle_df, vehicle_df, pct=100.0)
        if not prof.empty:
            curves[ph] = prof
    crit = CriticalMovementEngine(db).critical(START, END)
    demand = dict(zip(crit["demand"]["phase"], crit["demand"]["demand_vph"]))
    return optimize(
        curves, crit["structure"], {2: True, 6: True}, demand,
        {2: 20.0, 4: 15.0, 6: 20.0, 8: 10.0}, **kw,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_reproduces_the_core_on_its_own_inputs(self, interior_db):
        res = OptimizerEngine(interior_db).optimize(
            START, END, saturated=[2, 6], pct=100.0, c_min=30.0, make_plot=False,
        )
        exp = _expected(interior_db, c_min=30.0)
        assert res["optimum"]["state"] == "interior"
        assert res["optimum"]["c_star"] == exp["optimum"]["c_star"] == 50.0
        pd.testing.assert_frame_equal(res["splits"], exp["splits"])
        pd.testing.assert_frame_equal(res["scan"], exp["scan"])
        assert res["directive"] is None

    def test_boundary_site_returns_the_directive(self, boundary_db):
        res = OptimizerEngine(boundary_db).optimize(
            START, END, saturated=[2, 6], pct=100.0, make_plot=False,
        )
        exp = _expected(boundary_db)
        assert res["optimum"]["state"] == "boundary"
        assert res["directive"] == exp["directive"]
        assert [p["phase"] for p in res["directive"]["phases"]] == [2, 6]

    def test_result_keys(self, interior_db):
        res = OptimizerEngine(interior_db).optimize(
            START, END, saturated=[2, 6], pct=100.0, c_min=30.0, make_plot=True,
        )
        assert {"scan", "splits", "optimum", "directive", "warnings", "curves",
                "saturation", "demand", "min_splits", "figures"} <= set(res)
        assert sorted(res["curves"]) == [2, 4, 6, 8]
        assert set(res["figures"]) == {"curve", "allocation", "marginal"}
        assert all(isinstance(f, go.Figure) for f in res["figures"].values())

    def test_min_splits_from_config_with_default_and_warning(self, interior_db, capsys):
        res = OptimizerEngine(interior_db).optimize(
            START, END, saturated=[2, 6], pct=100.0, c_min=30.0,
            default_min_split=10.0, make_plot=False,
        )
        assert res["min_splits"] == {2: 20.0, 4: 15.0, 6: 20.0, 8: 10.0}
        assert "Min_P8_Split" in capsys.readouterr().out

    def test_default_min_split_everywhere_without_config(self, tmp_path):
        db = _build_db(tmp_path, decay=1.06, min_split=False)
        res = OptimizerEngine(db).optimize(
            START, END, saturated=[2, 6], pct=100.0, c_min=30.0,
            default_min_split=12.0, make_plot=False,
        )
        assert res["min_splits"] == {2: 12.0, 4: 12.0, 6: 12.0, 8: 12.0}

    def test_advisory_is_reported_and_disagreement_printed(self, interior_db, capsys):
        res = OptimizerEngine(interior_db).optimize(
            START, END, saturated=[2, 4], pct=100.0, c_min=30.0, make_plot=False,
        )
        adv = res["saturation"].set_index("phase")["saturated"]
        assert adv.to_dict() == {2: True, 4: False, 6: True, 8: False}
        out = capsys.readouterr().out.lower()
        # Declared-but-not-advised (4) and advised-but-not-declared (6).
        lines = [ln for ln in out.splitlines() if "advisory" in ln]
        assert any("ph4" in ln for ln in lines)
        assert any("ph6" in ln for ln in lines)
        # The declaration, not the advisory, decides.
        sp = res["splits"].set_index("phase")
        assert bool(sp.loc[4, "saturated"]) and not bool(sp.loc[6, "saturated"])

    def test_missing_curve_is_warned_with_a_hint(self, interior_db, capsys):
        # No window is on plan 99, so no phase gets a curve.
        res = OptimizerEngine(interior_db).optimize(
            START, END, saturated=[2, 6], plans=[99], pct=100.0, c_min=30.0,
            make_plot=False,
        )
        assert any(w.startswith("curve_missing") for w in res["warnings"])
        out = capsys.readouterr().out
        assert "--pct" in out and "Ph2" in out

    def test_undeclared_structure_phase_is_warned_and_ignored(self, interior_db, capsys):
        res = OptimizerEngine(interior_db).optimize(
            START, END, saturated=[2, 6, 9], pct=100.0, c_min=30.0, make_plot=False,
        )
        assert "ph9" in capsys.readouterr().out.lower()
        assert res["optimum"]["c_star"] == 50.0

    def test_empty_declaration_raises(self, interior_db):
        with pytest.raises(ValueError, match="saturated"):
            OptimizerEngine(interior_db).optimize(START, END, saturated=[])

    def test_bad_demand_stat_raises(self, interior_db):
        with pytest.raises(ValueError, match="demand_stat"):
            OptimizerEngine(interior_db).optimize(
                START, END, saturated=[2, 6], demand_stat="median")

    def test_window_without_cycles_returns_empty(self, interior_db, capsys):
        res = OptimizerEngine(interior_db).optimize(
            "2025-06-03 07:00", "2025-06-03 09:00", saturated=[2, 6])
        assert res == {}
        assert capsys.readouterr().out.strip()

    def test_output_dir_writes_files(self, interior_db, tmp_path):
        out = tmp_path / "outputs"
        assert get_optimization(
            interior_db, START, END, saturated=[2, 6], pct=100.0, c_min=30.0,
            output_dir=out,
        ) is None
        names = sorted(p.name for p in out.iterdir())
        assert names == sorted([
            f"Optimize_Allocation_{STAMP}.html",
            f"Optimize_Curve_{STAMP}.html",
            f"Optimize_Marginal_{STAMP}.html",
            f"Optimize_Saturation_{STAMP}.csv",
            f"Optimize_Scan_{STAMP}.csv",
            f"Optimize_Splits_{STAMP}.csv",
            f"Optimize_Summary_{STAMP}.csv",
        ])
        summary = pd.read_csv(out / f"Optimize_Summary_{STAMP}.csv")
        assert len(summary) == 1
        assert summary["c_star"].iloc[0] == 50.0
        assert summary["state"].iloc[0] == "interior"
        assert json.loads(summary["binding_ring"].iloc[0]) == {"0": 1, "1": 1}

    def test_no_plot_writes_no_html(self, interior_db, tmp_path):
        out = tmp_path / "outputs"
        get_optimization(interior_db, START, END, saturated=[2, 6], pct=100.0,
                         c_min=30.0, make_plot=False, output_dir=out)
        assert not list(out.glob("*.html"))


# ---------------------------------------------------------------------------
# Plots (pure)
# ---------------------------------------------------------------------------


def _segments(trace):
    """(y, length) per [start, end, None] segment of a trace."""
    xs, ys = list(trace.x), list(trace.y)
    return [(ys[i], xs[i + 1] - xs[i]) for i in range(0, len(xs) - 2, 3)]


class TestPlots:

    @pytest.fixture
    def result(self, interior_db):
        return OptimizerEngine(interior_db).optimize(
            START, END, saturated=[2, 6], pct=100.0, c_min=30.0, make_plot=False,
        )

    def test_throughput_curve(self, result):
        fig = plot_throughput_curve(result["scan"], result["optimum"],
                                    {"intersection_name": "Synthetic"})
        names = [t.name for t in fig.data]
        thr = next(t for t in fig.data if t.name == "Saturated throughput")
        assert len(thr.x) == int(result["scan"]["feasible"].sum())
        star = next(t for t in fig.data if t.name == "C*")
        assert list(star.x) == [50.0]
        assert "Flat band" in names
        assert "Synthetic" in fig.layout.title.text
        assert not fig.layout.shapes

    def test_allocation_rings_fill_the_cycle(self, result):
        fig = plot_allocation(result["splits"], result["optimum"],
                              {"intersection_name": "Synthetic"})
        per_ring = {}
        for t in fig.data:
            if t.name in ("optimized", "sufficiency", "minimum"):
                for y, length in _segments(t):
                    per_ring[y] = per_ring.get(y, 0.0) + length
        assert len(per_ring) == 2
        assert all(v == pytest.approx(50.0) for v in per_ring.values())
        assert not fig.layout.shapes

    def test_marginal_rates(self, result):
        fig = plot_marginal_rates(result["curves"], result["splits"],
                                  {"intersection_name": "Synthetic"})
        names = [t.name for t in fig.data]
        assert "Ph2" in names and "Ph6" in names
        assert "Ph4" not in names               # only optimized phases
        end = next(t for t in fig.data if t.name == "End of split")
        assert len(end.x) == 2
        assert not fig.layout.shapes

    def test_infeasible_result_still_plots(self, result):
        scan = result["scan"].assign(feasible=False, throughput_sat_vph=np.nan)
        optimum = dict(result["optimum"], state="infeasible", c_star=np.nan)
        splits = result["splits"].assign(s_star=np.nan)
        meta = {"intersection_name": "Synthetic"}
        assert isinstance(plot_throughput_curve(scan, optimum, meta), go.Figure)
        assert isinstance(plot_allocation(splits, optimum, meta), go.Figure)
        assert isinstance(plot_marginal_rates({}, splits, meta), go.Figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    def _parse(self, *argv):
        return cli._build_parser().parse_args(["optimize", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "201", "--start", START, "--end", END,
                           "--saturated", "2", "6")
        assert args.func is cli.handle_optimize
        assert args.saturated == [2, 6]
        assert args.pct == 1.0 and args.stratify is False
        assert args.demand_stat == "mean" and args.default_min_split == 10.0
        assert (args.c_min, args.c_max, args.c_step) == (60.0, 220.0, 1.0)

    def test_saturated_is_required(self):
        with pytest.raises(SystemExit):
            self._parse("--targetid", "201", "--start", START, "--end", END)

    def test_target_group_is_required_and_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse("--start", START, "--end", END, "--saturated", "2")
        with pytest.raises(SystemExit):
            self._parse("--all", "--targetid", "201", "--start", START,
                        "--end", END, "--saturated", "2")

    def test_end_to_end(self, tmp_path, monkeypatch):
        folder = tmp_path / "intersections" / "900_Synthetic"
        folder.mkdir(parents=True)
        _build_db(folder, decay=1.06)
        (folder / "metadata.json").write_text(json.dumps({
            "intersection_name": "Synthetic", "intersection_id": "900",
            "db_filename": "opt.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--saturated", "2", "6", "--pct", "100",
                           "--c-min", "30", "--no-plot")
        args.func(args)
        summary = pd.read_csv(folder / "outputs" / f"Optimize_Summary_{STAMP}.csv")
        assert summary["c_star"].iloc[0] == 50.0
