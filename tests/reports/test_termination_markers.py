# The termination plot must not see clock-mark ped calls (imperative shell).
#
# Marks are ped calls on unused ped phases (S1, docs/ROADMAP.md). The
# generator drops them explicitly when Clk_* config names marker peds.

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from atspm.data.manager import DatabaseManager
from atspm.reports import generators
from atspm.reports.generators import PlotGenerator

from ..conftest import seed_events

T0 = datetime(2026, 9, 30, 19, 0, tzinfo=timezone.utc).timestamp()


def _seed_config(db_path: Path, clk: dict) -> None:
    row = {"start_date": "2000-01-01T00:00:00", "end_date": None, **clk}
    with DatabaseManager(db_path) as m:
        for col in clk:
            m.add_config_column(col)
        m._insert_config_row(row)
        m.conn.commit()


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_plot(df_events, metadata):
        seen["df"] = df_events
        return type("Fig", (), {"write_html": lambda self, path: None})()

    monkeypatch.setattr(generators, "plot_termination", fake_plot)
    return seen


def _run(db_path: Path, tmp_path: Path) -> None:
    start = datetime(2026, 9, 30, 18, 0, tzinfo=timezone.utc)
    end = datetime(2026, 9, 30, 20, 0, tzinfo=timezone.utc)
    PlotGenerator(db_path, tmp_path)._generate_termination(
        "2026-09-30", start, end, {}, tmp_path, "US/Mountain"
    )


EVENTS = [(T0, 45, 14), (T0 + 1, 21, 14), (T0, 45, 2), (T0 + 1, 21, 2), (T0 + 2, 4, 2)]


def test_marker_ped_calls_are_dropped(empty_db, tmp_path, captured):
    _seed_config(empty_db, {"Clk_Behind": "15", "Clk_Ahead": "16", "Clk_Set": "14"})
    seed_events(empty_db, EVENTS)
    _run(empty_db, tmp_path)
    assert sorted(captured["df"]["parameter"].unique()) == [2]


def test_without_clk_config_nothing_is_dropped(empty_db, tmp_path, captured):
    _seed_config(empty_db, {})
    seed_events(empty_db, EVENTS)
    _run(empty_db, tmp_path)
    assert sorted(captured["df"]["parameter"].unique()) == [2, 14]
