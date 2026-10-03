"""Goldens for the S-D5 timing-and-actuation figure (plotting/timing_actuation.py)."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz

from atspm.analysis.detector_health import FINDINGS_SCHEMA
from atspm.analysis.detector_roles import parse_detector_roles
from atspm.analysis.timing_actuation import (
    timing_actuation_intervals,
    timing_actuation_rows,
)
from atspm.plotting.timing_actuation import plot_timing_actuation

TZ = "US/Mountain"
T0 = pytz.timezone(TZ).localize(datetime(2026, 3, 19, 1, 45)).timestamp()
W = (T0, T0 + 600)
META = {"major_road_route": "SH-55", "major_road_name": "Main St",
        "minor_road_route": None, "minor_road_name": "Banks-Lowman Rd"}


def _ev(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    df["timestamp"] = df["timestamp"].astype(float) + T0
    return df.sort_values(["timestamp", "event_code", "parameter"]).reset_index(drop=True)


def _build(rows, cfg=None, findings=None):
    out = timing_actuation_intervals(_ev(rows), W)
    roles = parse_detector_roles(cfg or {})
    lay = timing_actuation_rows(roles, out["intervals"], out["marks"])
    fig = plot_timing_actuation(lay, out["intervals"], out["marks"], W, TZ, META, findings)
    return fig, lay


def _local_ms(epoch):
    local = pd.Timestamp(epoch, unit="s", tz="UTC").tz_convert(TZ).tz_localize(None)
    return local.value / 1e6


def _trace(fig, name):
    tr = [t for t in fig.data if t.name == name]
    assert len(tr) == 1, [t.name for t in fig.data]
    return tr[0]


CYCLE = [(10, 1, 2), (40, 8, 2), (44, 9, 2), (44, 10, 2), (46, 11, 2), (300, 1, 2),
         (5, 43, 2), (12, 44, 2),
         (20, 82, 21), (22, 81, 21), (30, 82, 21), (33, 81, 21)]


def test_returns_figure_with_title():
    fig, _ = _build(CYCLE, {"Det_P2_Occupancy": "21"})
    assert isinstance(fig, go.Figure)
    title = fig.layout.title.text
    assert title.startswith("SH-55 (Main St) & Banks-Lowman Rd — Timing & Actuation")
    assert "2026-03-19 01:45" in title and "01:55" in title


def test_title_falls_back_to_intersection_name():
    out = timing_actuation_intervals(_ev(CYCLE), W)
    lay = timing_actuation_rows(None, out["intervals"], out["marks"])
    fig = plot_timing_actuation(lay, out["intervals"], out["marks"], W, TZ, {"intersection_name": "X & Y"})
    assert fig.layout.title.text.startswith("X & Y — Timing & Actuation")


def test_empty_rows_valid_figure():
    lay = timing_actuation_rows(None, timing_actuation_intervals(_ev([]), W)["intervals"])
    fig = plot_timing_actuation(lay, timing_actuation_intervals(_ev([]), W)["intervals"], None, W, TZ, META)
    assert len(fig.data) == 0
    assert "Timing & Actuation" in fig.layout.title.text


def test_one_trace_per_style_and_none_gap_pattern():
    fig, lay = _build(CYCLE, {"Det_P2_Occupancy": "21"})
    names = [t.name for t in fig.data]
    for n in ("Green", "Yellow", "Red clearance", "Red", "Phase call", "Occupancy det"):
        assert names.count(n) == 1, n
    det = _trace(fig, "Occupancy det")
    x = np.asarray(det.x, dtype=float)
    assert len(x) == 2 * 4 and np.isnan(x[3::4]).all()
    row = lay.loc[(lay["kind"] == "detector") & (lay["param"] == 21), "row"].item()
    y = np.asarray(det.y, dtype=float)
    assert set(y[~np.isnan(y)]) == {row}
    # x is local wall-clock ms: start, mid, end of the first actuation
    np.testing.assert_allclose(x[:3], [_local_ms(T0 + 20), _local_ms(T0 + 21), _local_ms(T0 + 22)])


def test_trace_count_independent_of_interval_count():
    rows = [(1 + 2 * k, 82, 21) for k in range(250)] + [(2 + 2 * k, 81, 21) for k in range(250)]
    fig, _ = _build(rows, {"Det_P2_Occupancy": "21"})
    assert [t.name for t in fig.data] == ["Occupancy det"]
    assert len(fig.data[0].x) == 250 * 4


def test_y_axis_labels_and_reversed_range():
    fig, lay = _build(CYCLE, {"Det_P2_Occupancy": "21"})
    assert list(fig.layout.yaxis.tickvals) == lay["row"].tolist()
    assert list(fig.layout.yaxis.ticktext) == lay["label"].tolist()
    lo, hi = fig.layout.yaxis.range
    assert lo > hi, "row 0 must be at the top"
    assert lo >= lay["row"].max() and hi < 0


def test_x_range_is_window():
    fig, _ = _build(CYCLE)
    np.testing.assert_allclose(list(fig.layout.xaxis.range), [_local_ms(W[0]), _local_ms(W[1])])


def test_hover_text_has_state_times_and_open_end():
    fig, _ = _build([(100, 82, 21)], {"Det_P2_Occupancy": "21"})
    det = _trace(fig, "Occupancy det")
    h = det.hovertext[0]
    assert "Occ 21" in h and "On" in h and "01:46:40.0" in h and "end not logged" in h
    assert det.hovertext[3] == ""


def test_silent_configured_detector_has_row_but_no_trace():
    fig, lay = _build(CYCLE, {"Det_P2_Occupancy": "21", "Det_P2_Arrival": "60"})
    assert "Arr 60" in list(fig.layout.yaxis.ticktext)
    assert "Arrival det" not in [t.name for t in fig.data]


def test_marks_and_gap_lines():
    fig, lay = _build(CYCLE + [(50, 45, 2), (60, 21, 2), (67, 22, 2), (80, 23, 2), (200, -1, 0)])
    ped = _trace(fig, "Ped call")
    assert list(ped.y) == [lay.loc[lay["kind"] == "ped", "row"].item()]
    gap = _trace(fig, "Hard reset")
    x = np.asarray(gap.x, dtype=float)
    np.testing.assert_allclose(x[:2], [_local_ms(T0 + 200)] * 2)
    assert np.isnan(x[2])
    y = np.asarray(gap.y, dtype=float)[:2]
    assert min(y) < 0 and max(y) > lay["row"].max()


def _findings(rows):
    return pd.DataFrame(rows, columns=FINDINGS_SCHEMA).astype(
        {"ts": "float64", "detector": "int64", "phase": "Int64"})


def test_findings_overlay():
    d = date(2026, 3, 19)
    f = _findings([
        [d, "day", T0 + 30, 21, 2, "occupancy", "StuckOn", "high", 900, 600, "stuck"],
        [d, "day", np.nan, 21, 2, "occupancy", "LowHits", "low", 1, 5, "low hits"],
        [d, "day", T0 + 312, -1, pd.NA, "unit:all", "Failsafe", "high", 30, 16, "burst"],
        [d, "day", T0 + 5000, 21, 2, "occupancy", "Chatter", "low", 1, 1, "outside"],
        [d, "day", T0 + 40, 99, pd.NA, "", "StuckOn", "high", 1, 1, "no row"],
    ])
    fig, lay = _build(CYCLE, {"Det_P2_Occupancy": "21"}, f)
    tr = _trace(fig, "Finding")
    row21 = lay.loc[lay["param"] == 21, "row"].item()
    pts = sorted(zip(np.asarray(tr.x, float), np.asarray(tr.y, float), tr.marker.symbol))
    assert len(pts) == 3
    np.testing.assert_allclose([p[0] for p in pts], sorted([_local_ms(T0 + 30), _local_ms(W[0]), _local_ms(T0 + 312)]))
    by_x = {round(p[0]): p for p in pts}
    assert by_x[round(_local_ms(T0 + 30))][1:] == (row21, "x")
    assert by_x[round(_local_ms(W[0]))][1:] == (row21, "diamond-open")
    top = by_x[round(_local_ms(T0 + 312))]
    assert top[1] < 0 and top[2] == "x"
    assert any("Failsafe" in h for h in tr.hovertext)
    lo, hi = fig.layout.yaxis.range
    assert hi < top[1], "top band must be inside the y range"


def test_pure_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _build(CYCLE, {"Det_P2_Occupancy": "21"})
    assert list(tmp_path.iterdir()) == []
