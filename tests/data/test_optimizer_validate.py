# Acceptance tests for `OptimizerEngine.validate` and `atspm optimize
# --validate` (ROADMAP optimizer step 6; spec docs/specs/optimizer_validate.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The validation math is pinned in
# tests/analysis/test_optimizer_validation.py.

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm import cli
from atspm.analysis.flow import flow_rate
from atspm.analysis.optimizer_validation import validate_plans
from atspm.data import optimizer as shell
from atspm.data.flow import _ALL_FLOW_CODES
from atspm.data.manager import DatabaseManager
from atspm.data.optimizer import OptimizerEngine, get_validation
from atspm.data.processing import CycleProcessor
from atspm.data.reader import _query_cycles, get_events_with_cycles_df
from atspm.utils.timezone import to_epoch

TZ = "US/Mountain"
DAY = "2025-06-02"
START, END = f"{DAY} 07:00", f"{DAY} 10:00"
STAMP = "2025_06_02_0700-2025_06_02_1000"

# ---------------------------------------------------------------------------
# Synthetic intersection: rings 2|4 over 6|8.  Plan 1 runs a 120 s cycle
# 07:00-08:30, plan 2 a 90 s cycle 08:30-10:00 (Code 131 at each change).
# Phases 2/6 discharge a queue at a 2 s headway until the end of green;
# 4/8 serve a few vehicles.  Every green ends by force-off.
# ---------------------------------------------------------------------------

_PLANS = [
    # (plan, cycle, minutes, {phase: (offset, green)})
    (1, 120.0, 90, {2: (0.0, 54.0), 6: (0.0, 54.0), 4: (60.0, 54.0), 8: (60.0, 54.0)}),
    (2, 90.0, 90, {2: (0.0, 39.0), 6: (0.0, 39.0), 4: (45.0, 39.0), 8: (45.0, 39.0)}),
]
_DETS = {2: [1, 2], 6: [3, 4], 4: [5], 8: [6]}
_QUEUED = {2, 6}


def _departures(green, queued, rng):
    if queued:
        return list(np.arange(2.0, green - 0.2, 2.0))
    return list(np.cumsum(rng.uniform(2.0, 6.0, size=4)))


def _build_db(root: Path, stopbar_p6: bool = True, gap_at: float = None) -> Path:
    """Write the two-plan synthetic DB.

    Args:
        stopbar_p6: Configure Ph6's stop-bar detectors.
        gap_at: Seconds after 07:00 to insert a gap marker, or None.
    """
    rng = np.random.default_rng(7)
    t = to_epoch(datetime.strptime(START, "%Y-%m-%d %H:%M"), TZ)
    t_first = t
    events = []
    for plan, c_len, minutes, spec in _PLANS:
        events.append((t, 131, plan))
        for _ in range(int(minutes * 60 / c_len)):
            for ph, (off, g) in spec.items():
                gs = t + off
                events += [(gs, 1, ph), (gs + g, 6, ph), (gs + g, 8, ph),
                           (gs + g + 4, 9, ph), (gs + g + 4, 10, ph),
                           (gs + g + 6, 11, ph), (gs + g + 6, 12, ph)]
                for d in _DETS[ph]:
                    for dt in _departures(g, ph in _QUEUED, rng):
                        events += [(gs + dt - 0.3, 82, d), (gs + dt, 81, d)]
            t += c_len
    if gap_at is not None:
        events.append((t_first + gap_at, -1, 0))
    db = root / "val.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ)
        cfg = {"RB_R1": "2|4", "RB_R2": "6|8",
               "Det_P2_Stop_Bar": "1,2", "Det_P4_Stop_Bar": "5",
               "Det_P8_Stop_Bar": "6",
               "TM_EBT": "1,2", "TM_WBT": "3,4", "TM_NBT": "5", "TM_SBT": "6"}
        if stopbar_p6:
            cfg["Det_P6_Stop_Bar"] = "3,4"
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00",
                              "end_date": None, **cfg})
        m.insert_events(sorted(events))
        m.conn.execute(
            "INSERT INTO ingestion_log (span_start, span_end, processed_at, "
            "row_count) VALUES (?, ?, '2025-06-02T00:00:00', ?)",
            (t_first, t, len(events)),
        )
        m.conn.commit()
    CycleProcessor(db, TZ).run()
    return db


@pytest.fixture
def db(tmp_path) -> Path:
    return _build_db(tmp_path)


def _expected(db: Path, phases=(2, 6), plans=None, **kw):
    """The result the engine must reproduce, assembled from the pure core."""
    start = datetime.strptime(START, "%Y-%m-%d %H:%M")
    end = datetime.strptime(END, "%Y-%m-%d %H:%M")
    events = get_events_with_cycles_df(db, start, end, event_codes=_ALL_FLOW_CODES)
    flow = {p: flow_rate(events, p, _DETS[p], max_lost=None, plans=plans)
            for p in phases}
    cycles = _query_cycles(db, to_epoch(start, TZ), to_epoch(end, TZ))
    if plans is not None:
        cycles = cycles.loc[cycles["coord_plan"].isin(plans)]
    gaps = events.loc[events["event_code"] == -1, "timestamp"].to_numpy()
    return validate_plans(flow, cycles, gap_ts=gaps, **kw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_reproduces_the_core_on_its_own_inputs(self, db):
        got = OptimizerEngine(db).validate(START, END, saturated=[2, 6], pct=100.0)
        exp = _expected(db, pct=100.0)
        pd.testing.assert_frame_equal(got["plans"], exp["plans"])
        pd.testing.assert_frame_equal(got["pairs"], exp["pairs"])
        assert got["verdict"] == exp["verdict"]
        # Sanity on the fixture itself: two plans, one tested pair.
        assert sorted(got["plans"]["coord_plan"]) == [1.0, 2.0]
        assert got["pairs"].iloc[0]["status"] == "tested"

    def test_core_receives_epoch_floats_even_with_a_timezone(self, db, monkeypatch):
        seen = {}

        def spy(flow, cycles_df, **kw):
            seen["flow"], seen["cycles"], seen["kw"] = flow, cycles_df, kw
            return validate_plans(flow, cycles_df, **kw)

        monkeypatch.setattr(shell, "validate_plans", spy)
        OptimizerEngine(db, timezone=TZ).validate(
            START, END, saturated=[2, 6], pct=100.0)
        assert pd.api.types.is_float_dtype(seen["cycles"]["cycle_start"])
        for cdf, _ in seen["flow"].values():
            assert pd.api.types.is_float_dtype(cdf["cycle_start"])
            assert pd.api.types.is_float_dtype(cdf["green_ts"])

    def test_parameters_pass_through(self, db, monkeypatch):
        seen = {}

        def spy(flow, cycles_df, **kw):
            seen.update(kw)
            return validate_plans(flow, cycles_df, **kw)

        monkeypatch.setattr(shell, "validate_plans", spy)
        OptimizerEngine(db).validate(
            START, END, saturated=[2, 6], pct=50.0, split_tolerance=0.2,
            max_lost=8.0, sat_threshold=0.7, min_plan_cycles=10,
            split_cover_tol=2.0, rank_deadband_pct=1.5, change_tol_pp=4.0)
        for k, v in dict(pct=50.0, split_tolerance=0.2, max_lost=8.0,
                         sat_threshold=0.7, min_plan_cycles=10,
                         split_cover_tol=2.0, rank_deadband_pct=1.5,
                         change_tol_pp=4.0).items():
            assert seen[k] == v, k

    def test_gap_markers_reach_the_core(self, tmp_path, monkeypatch):
        db = _build_db(tmp_path, gap_at=1000.0)
        seen = {}

        def spy(flow, cycles_df, **kw):
            seen["gap_ts"] = np.asarray(kw["gap_ts"], dtype=float)
            return validate_plans(flow, cycles_df, **kw)

        monkeypatch.setattr(shell, "validate_plans", spy)
        OptimizerEngine(db).validate(START, END, saturated=[2, 6], pct=100.0)
        t_gap = to_epoch(datetime.strptime(START, "%Y-%m-%d %H:%M"), TZ) + 1000.0
        assert list(seen["gap_ts"]) == [pytest.approx(t_gap)]

    def test_plans_filter(self, db):
        got = OptimizerEngine(db).validate(
            START, END, saturated=[2, 6], plans=[1], pct=100.0)
        exp = _expected(db, plans=[1], pct=100.0)
        pd.testing.assert_frame_equal(got["plans"], exp["plans"])
        assert got["plans"]["coord_plan"].tolist() == [1.0]
        assert got["pairs"].empty

    def test_phase_without_stopbar_is_warned_and_dropped(self, tmp_path, capsys):
        db = _build_db(tmp_path, stopbar_p6=False)
        got = OptimizerEngine(db).validate(START, END, saturated=[2, 6], pct=100.0)
        assert "Ph6" in capsys.readouterr().out
        assert got["verdict"]["phases"] == [2]
        exp = _expected(db, phases=(2,), pct=100.0)
        pd.testing.assert_frame_equal(got["pairs"], exp["pairs"])

    def test_no_phase_left_returns_empty(self, tmp_path, capsys):
        db = _build_db(tmp_path, stopbar_p6=False)
        assert OptimizerEngine(db).validate(START, END, saturated=[6]) == {}
        assert "Ph6" in capsys.readouterr().out

    def test_empty_declaration_raises(self, db):
        with pytest.raises(ValueError, match="saturated"):
            OptimizerEngine(db).validate(START, END, saturated=[])

    def test_window_without_cycles_returns_empty(self, db):
        assert OptimizerEngine(db).validate(
            "2025-06-03 07:00", "2025-06-03 08:00", saturated=[2, 6]) == {}

    def test_verdict_is_printed(self, db, capsys):
        got = OptimizerEngine(db).validate(START, END, saturated=[2, 6], pct=100.0)
        out = capsys.readouterr().out
        assert f"Validation: {got['verdict']['verdict']}" in out

    def test_output_dir_writes_three_csvs(self, db, tmp_path):
        out = tmp_path / "out"
        assert OptimizerEngine(db).validate(
            START, END, saturated=[2, 6], pct=100.0, output_dir=out) is None
        exp = _expected(db, pct=100.0)
        plans = pd.read_csv(out / f"Optimize_Validation_Plans_{STAMP}.csv")
        pairs = pd.read_csv(out / f"Optimize_Validation_Pairs_{STAMP}.csv")
        summary = pd.read_csv(out / f"Optimize_Validation_Summary_{STAMP}.csv")
        assert list(plans.columns) == list(exp["plans"].columns)
        assert list(pairs.columns) == list(exp["pairs"].columns)
        assert len(summary) == 1
        assert summary["verdict"].iloc[0] == exp["verdict"]["verdict"]
        assert json.loads(summary["phases"].iloc[0]) == [2, 6]
        assert json.loads(summary["warnings"].iloc[0]) == exp["verdict"]["warnings"]

    def test_wrapper(self, db):
        got = get_validation(db, START, END, saturated=[2, 6], pct=100.0)
        assert got["verdict"] == _expected(db, pct=100.0)["verdict"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    def _parse(self, *argv):
        return cli._build_parser().parse_args(["optimize", *argv])

    def test_validate_flags_and_defaults(self):
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--saturated", "2", "6", "--validate")
        assert args.validate is True
        assert args.min_plan_cycles == 30
        assert args.split_cover_tol == 1.0
        assert args.rank_deadband_pct == 2.0
        assert args.change_tol_pp == 3.0

    def test_validate_defaults_off(self):
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--saturated", "2", "6")
        assert args.validate is False

    def test_end_to_end(self, tmp_path, monkeypatch):
        folder = tmp_path / "intersections" / "900_Synthetic"
        folder.mkdir(parents=True)
        _build_db(folder)
        (folder / "metadata.json").write_text(json.dumps({
            "intersection_name": "Synthetic", "intersection_id": "900",
            "db_filename": "val.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        args = self._parse("--targetid", "900", "--start", START, "--end", END,
                           "--saturated", "2", "6", "--pct", "100",
                           "--validate", "--change-tol-pp", "5")
        args.func(args)
        out = folder / "outputs"
        summary = pd.read_csv(out / f"Optimize_Validation_Summary_{STAMP}.csv")
        exp = _expected(folder / "val.db", pct=100.0, change_tol_pp=5.0)
        assert summary["verdict"].iloc[0] == exp["verdict"]["verdict"]
        assert summary["change_tol_pp"].iloc[0] == 5.0
        # --validate runs instead of the optimizer, not as well.
        assert not list(out.glob("Optimize_Scan_*.csv"))
