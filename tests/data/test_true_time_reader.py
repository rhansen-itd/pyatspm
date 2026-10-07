# Shell tests for the true-time axis (S2): the reader's ``true_time`` flag,
# the model loader's fetch bounds, and ``clock-drift --true-time``.
#
# The DB is seeded with a simulated controller (tests/analysis/
# test_true_time.py) in label time, so every row's true instant is known.

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

from atspm import cli
from atspm.data.clock_marks import ClockMarkEngine
from atspm.data.manager import DatabaseManager
from atspm.data.reader import get_events_with_cycles_df
from atspm.analysis.true_time import to_true_time
from atspm.data.true_time import load_drift_model
from tests.analysis.test_true_time import SHARED_CODE, SHARED_PARAM, T0, Controller, shared_times

TZ = "US/Mountain"
UTC = pytz.utc


def _seed(root: Path, controller: Controller, clk: bool = True) -> tuple[Path, pd.DataFrame]:
    log = controller.log()
    db = root / "sim_data.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Sim", timezone=TZ)
        cfg = {"Clk_Behind": "15", "Clk_Ahead": "16", "Clk_Set": "14"} if clk else {}
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00", "end_date": None, **cfg})
        m.conn.executemany(
            "INSERT INTO events (timestamp, event_code, parameter) VALUES (?, ?, ?)",
            log[["timestamp", "event_code", "parameter"]].itertuples(index=False, name=None),
        )
        # One cycle every 120 s of label time, so cycle_start is mapped too.
        m.conn.execute(
            "CREATE TABLE IF NOT EXISTS cycles (cycle_start REAL PRIMARY KEY, "
            "coord_plan REAL NOT NULL DEFAULT 0, detection_method TEXT NOT NULL DEFAULT '', "
            "r1_phases TEXT NOT NULL DEFAULT 'None', r2_phases TEXT NOT NULL DEFAULT 'None')"
        )
        starts = np.arange(T0, T0 + 2 * 86400, 120.0)
        m.conn.executemany(
            "INSERT INTO cycles (cycle_start, coord_plan, detection_method) VALUES (?, 1, 'test')",
            [(float(s),) for s in starts],
        )
        m.conn.commit()
    return db, log


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    c = Controller(d0=1.2, ppm=40, seed=21).run()
    db, log = _seed(tmp_path_factory.mktemp("tt"), c)
    return c, db, log


def _utc(t):
    return datetime.fromtimestamp(t, tz=UTC)


class TestReader:

    def test_true_time_rows_land_on_their_true_instants(self, sim):
        # The shared stimulus is logged at known true instants.
        _, db, _ = sim
        start, end = T0 + 8 * 3600, T0 + 11 * 3600   # spans the 09:17 set
        out = get_events_with_cycles_df(db, _utc(start), _utc(end), true_time=True)
        got = out.loc[(out["event_code"] == SHARED_CODE) & (out["parameter"] == SHARED_PARAM), "timestamp"]
        truth = shared_times(2)
        truth = truth[(truth >= start) & (truth < end)]
        assert len(got) >= len(truth) - 2      # at most the dead zone's
        nearest = truth[np.clip(np.searchsorted(truth, got - 5), 0, len(truth) - 1)]
        assert np.abs(got.to_numpy() - nearest).max() < 0.3

    def test_window_is_applied_in_true_time(self, sim):
        _, db, _ = sim
        start, end = T0 + 3600, T0 + 7200
        out = get_events_with_cycles_df(db, _utc(start), _utc(end), true_time=True)
        assert out["timestamp"].min() >= start
        assert out["timestamp"].max() < end
        lab = get_events_with_cycles_df(db, _utc(start), _utc(end))
        # d ~ +1.3 s here: true = label - d, so the true window reads later labels.
        assert lab["timestamp"].min() < out["timestamp"].min() + 1.0

    def test_cycle_start_is_mapped_with_its_own_label(self, sim):
        _, db, _ = sim
        start, end = T0 + 3600, T0 + 7200
        out = get_events_with_cycles_df(db, _utc(start), _utc(end), true_time=True)
        lab = get_events_with_cycles_df(db, _utc(start), _utc(end))
        shift = lab["cycle_start"].min() - out.loc[out["cycle_start"].notna(), "cycle_start"].min()
        assert 0 < abs(shift) < 120
        offs = (out["cycle_start"].dropna().unique() - T0) % 120
        assert np.ptp(offs) < 0.3  # one drift for all cycles in a smooth hour

    def test_set_inserts_a_gap_marker(self, sim):
        c, db, _ = sim
        step = c.steps[0][0]
        out = get_events_with_cycles_df(db, _utc(step - 600), _utc(step + 600), true_time=True)
        markers = out.loc[out["event_code"] == -1, "timestamp"]
        assert ((markers > step - 60) & (markers < step + 60)).sum() == 1

    def test_event_code_filter_still_gets_markers(self, sim):
        c, db, _ = sim
        step = c.steps[0][0]
        out = get_events_with_cycles_df(
            db, _utc(step - 600), _utc(step + 600), event_codes=[82], true_time=True
        )
        assert set(out["event_code"]) == {82, -1}

    def test_default_is_label_time(self, sim):
        _, db, log = sim
        start, end = T0 + 3600, T0 + 3700
        out = get_events_with_cycles_df(db, _utc(start), _utc(end))
        assert out["timestamp"].isin(log["timestamp"]).all()

    def test_no_clk_config_raises(self, tmp_path):
        db, _ = _seed(tmp_path, Controller(d0=0.2, ppm=0, seed=2, days=1).run(), clk=False)
        with pytest.raises(ValueError, match="Clk_"):
            get_events_with_cycles_df(db, _utc(T0), _utc(T0 + 3600), true_time=True)


class TestLoader:

    def test_phase_hold_marks_build_the_same_model(self, sim, tmp_path):
        # Upstream's marks from 2026-10-07: the loader's SQL must fetch
        # 41/42 and find the SET bracket's hold ON as a break.
        c, db, _ = sim
        hold = Controller(d0=1.2, ppm=40, seed=21, mark="hold").run()
        hold_db, _ = _seed(tmp_path, hold)
        start = T0 + 86400 + 20 * 3600
        model = load_drift_model(hold_db, start, start + 3600)
        assert (model["opened_by"] == "set").sum() == 1
        ped = load_drift_model(db, start, start + 3600)
        assert model["segment"].nunique() == ped["segment"].nunique()
        t = np.linspace(start, start + 3600, 50)
        assert np.abs(to_true_time(t, model) - to_true_time(t, ped)).max() < 0.2

    def test_fetch_reaches_back_to_the_previous_set(self, sim):
        c, db, _ = sim
        # A window late on day 2: its segment opened at day 2's 09:17 set,
        # whose samples must all be in the fit.
        start = T0 + 86400 + 20 * 3600
        model = load_drift_model(db, start, start + 3600)
        piece = model.loc[(model["seg_start"] <= start) & (model["seg_end"] > start)].iloc[0]
        seg = model.loc[model["segment"] == piece["segment"]]
        assert seg["opened_by"].iloc[0] == "set"
        assert piece["n_samples"] >= 12
        rate = np.polyfit(seg["t_ref"], seg["intercept"], 1)[0]
        assert rate * 1e6 == pytest.approx(40, abs=8)

    def test_send_log_beside_the_db_is_picked_up(self, sim, tmp_path):
        # An empty send log file is read without error.
        _, db, _ = sim
        (db.parent / "eos-time.jsonl").write_text("")
        try:
            assert not load_drift_model(db, T0 + 3600, T0 + 7200).empty
        finally:
            (db.parent / "eos-time.jsonl").unlink()


class TestClockDriftCli:

    def test_engine_returns_model(self, sim):
        _, db, _ = sim
        res = ClockMarkEngine(db, timezone="UTC").decode(
            "2026-09-30", "2026-09-30", true_time=True
        )
        assert set(res) == {"drift", "sets", "model"}
        assert (res["model"]["opened_by"] == "set").sum() == 1

    def test_cli_writes_model_csv(self, sim, tmp_path, monkeypatch):
        _, db, _ = sim
        target = tmp_path / "intersections" / "999_Sim"
        target.mkdir(parents=True)
        (target / db.name).write_bytes(db.read_bytes())
        (target / "metadata.json").write_text(
            '{"intersection_id": "999", "intersection_name": "Sim", "timezone": "UTC"}'
        )
        monkeypatch.setattr(cli, "_get_intersections_dir", lambda: tmp_path / "intersections")
        args = argparse.Namespace(
            target="999_Sim", targetid=None, all=False, start="2026-09-30",
            end="2026-09-30", send_log=None, timezone="UTC", verbose=True,
            true_time=True,
        )
        cli.handle_clock_drift(args)
        out = target / "outputs"
        assert (out / "Clock_Model_2026_09_30-2026_09_30.csv").exists()
        model = pd.read_csv(out / "Clock_Model_2026_09_30-2026_09_30.csv")
        assert "rate_ppm" in model.columns
        html = (out / "Clock_Drift_2026_09_30-2026_09_30.html").read_text()
        assert "Drift model" in html

    def test_parser_has_the_flag(self):
        args = cli._build_parser().parse_args(
            ["clock-drift", "--targetid", "1", "--start", "2026-09-30",
             "--end", "2026-09-30", "--true-time"]
        )
        assert args.true_time is True
