"""Goldens for the S-D5 timing-and-actuation core (analysis/timing_actuation.py)."""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

from atspm.analysis.detector_health import FINDINGS_SCHEMA
from atspm.analysis.detector_roles import parse_detector_roles
from atspm.analysis.detectors import _reconstruct_intervals
from atspm.analysis.phases import _build_phase_intervals, _segment_id
from atspm.analysis.timing_actuation import (
    INTERVAL_SCHEMA,
    MARK_SCHEMA,
    ROW_SCHEMA,
    TIMING_CODES,
    finding_plot_windows,
    ring_phase_order,
    timing_actuation_intervals,
    timing_actuation_rows,
)

T0 = 1_700_000_000.0


def _ev(rows):
    """rows: (offset_s, code, param) -> events frame sorted like query_events."""
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    df["timestamp"] = df["timestamp"].astype(float) + T0
    return df.sort_values(["timestamp", "event_code", "parameter"]).reset_index(drop=True)


def _iv(events, w=(0, 1000), d=None):
    window = (T0 + w[0], T0 + w[1])
    data = None if d is None else (T0 + d[0], T0 + d[1])
    return timing_actuation_intervals(events, window, data)


def _rel(df):
    """Intervals as (kind, param, state, start, end, open_start, open_end) offsets."""
    return [
        (r.kind, r.param, r.state, round(r.start_ts - T0, 3), round(r.end_ts - T0, 3),
         bool(r.open_start), bool(r.open_end))
        for r in df.itertuples()
    ]


# ---------------------------------------------------------------------------
# Intervals: detectors
# ---------------------------------------------------------------------------

def test_schema_and_empty():
    out = timing_actuation_intervals(pd.DataFrame(columns=["timestamp", "event_code", "parameter"]),
                                     (T0, T0 + 10))
    assert list(out["intervals"].columns) == INTERVAL_SCHEMA
    assert list(out["marks"].columns) == MARK_SCHEMA
    assert out["intervals"].empty and out["marks"].empty


def test_timing_codes_cover_every_kind():
    for c in (-1, 1, 8, 9, 10, 11, 12, 21, 22, 23, 43, 44, 45, 81, 82, 102, 104, 105, 106, 107, 111):
        assert c in TIMING_CODES


def test_detector_closed_repeated_on_and_edges():
    ev = _ev([
        (5, 81, 7),            # leading OFF: was on since data start
        (10, 82, 7), (12, 82, 7), (15, 81, 7),   # repeated ON extends
        (900, 82, 7),          # still on at data end
    ])
    iv = _iv(ev)["intervals"]
    assert _rel(iv) == [
        ("detector", 7, "on", 0.0, 5.0, True, False),
        ("detector", 7, "on", 10.0, 15.0, False, False),
        ("detector", 7, "on", 900.0, 1000.0, False, True),
    ]


def test_stuck_on_whole_window_is_drawn():
    """A detector that turned on before the window and never off fills it."""
    ev = _ev([(-300, 82, 4)])
    iv = _iv(ev, w=(0, 600), d=(-900, 600))["intervals"]
    assert _rel(iv) == [("detector", 4, "on", 0.0, 600.0, False, True)]


def test_gap_marker_cuts_and_blocks_inference():
    ev = _ev([
        (10, 82, 3),
        (20, -1, 0),           # hard reset: interval ends, end not logged
        (30, 81, 3),           # OFF after a gap: prior state unknown -> nothing
        (40, 82, 3), (50, 81, 3),
    ])
    iv = _iv(ev)["intervals"]
    assert _rel(iv) == [
        ("detector", 3, "on", 10.0, 20.0, False, True),
        ("detector", 3, "on", 40.0, 50.0, False, False),
    ]
    marks = _iv(ev)["marks"]
    assert marks["kind"].tolist() == ["gap"]
    assert marks["ts"].iloc[0] == T0 + 20


def test_gap_at_same_instant_precedes_event():
    ev = _ev([(10, -1, 0), (10, 82, 3), (20, 81, 3)])
    assert _rel(_iv(ev)["intervals"]) == [("detector", 3, "on", 10.0, 20.0, False, False)]


def test_off_then_on_same_instant():
    ev = _ev([(10, 82, 3), (20, 81, 3), (20, 82, 3), (30, 81, 3)])
    assert _rel(_iv(ev)["intervals"]) == [("detector", 3, "on", 10.0, 30.0, False, False)]


def test_window_clipping():
    ev = _ev([(10, 82, 3), (50, 81, 3), (60, 82, 3), (70, 81, 3)])
    iv = _iv(ev, w=(20, 65), d=(0, 1000))["intervals"]
    assert _rel(iv) == [
        ("detector", 3, "on", 20.0, 50.0, False, False),
        ("detector", 3, "on", 60.0, 65.0, False, False),
    ]


# ---------------------------------------------------------------------------
# Intervals: phases, calls, peds, preempts
# ---------------------------------------------------------------------------

def test_phase_full_cycle_with_red_clearance():
    ev = _ev([(10, 1, 2), (40, 8, 2), (44, 9, 2), (44, 10, 2), (46, 11, 2), (46, 12, 2), (100, 1, 2),
              (130, 8, 2)])
    iv = _iv(ev, w=(0, 120), d=(0, 200))["intervals"]
    assert _rel(iv) == [
        ("phase", 2, "R", 0.0, 10.0, True, False),     # leading: first code 1 => was red
        ("phase", 2, "G", 10.0, 40.0, False, False),
        ("phase", 2, "Y", 40.0, 44.0, False, False),
        ("phase", 2, "RC", 44.0, 46.0, False, False),
        ("phase", 2, "R", 46.0, 100.0, False, False),  # 11 and 12 merge
        ("phase", 2, "G", 100.0, 120.0, False, False),
    ]


def test_phase_without_red_clearance_and_leading_green():
    ev = _ev([(5, 8, 4), (9, 9, 4), (9, 12, 4)])
    iv = _iv(ev, w=(0, 20))["intervals"]
    assert _rel(iv) == [
        ("phase", 4, "G", 0.0, 5.0, True, False),
        ("phase", 4, "Y", 5.0, 9.0, False, False),
        ("phase", 4, "R", 9.0, 20.0, False, True),
    ]


def test_phase_fya_code12_mid_green_continues():
    """315: Code 12 mid-green then Code 8 is a delayed FYA start, not an end."""
    ev = _ev([(10, 1, 2), (20, 12, 2), (40, 8, 2), (44, 9, 2)])
    iv = _iv(ev, w=(0, 50))["intervals"]
    assert ("phase", 2, "G", 10.0, 40.0, False, False) in _rel(iv)


def test_phase_repeated_begin_green_leaves_lost_span_blank():
    """315 2025-12-15 08:44: P4 green, a 55 s unlogged hole, P4 green again.

    The first green's end was never logged; the second is the real one.
    """
    ev = _ev([(0, 1, 4), (88, 1, 4), (110, 8, 4), (114, 9, 4)])
    rel = _rel(_iv(ev, w=(-10, 120), d=(-10, 120))["intervals"])
    assert ("phase", 4, "G", 88.0, 110.0, False, False) in rel
    assert not any(r[2] == "G" and r[3] < 88 for r in rel)


def test_phase_dummy_green_ends_at_code12():
    ev = _ev([(10, 1, 9), (20, 12, 9), (50, 1, 9), (60, 12, 9)])
    iv = _iv(ev, w=(0, 70))["intervals"]
    greens = [r for r in _rel(iv) if r[2] == "G"]
    assert greens == [("phase", 9, "G", 10.0, 20.0, False, False),
                      ("phase", 9, "G", 50.0, 60.0, False, False)]


def test_call_ped_preempt_and_marks():
    ev = _ev([
        (5, 43, 2), (25, 44, 2),
        (8, 45, 4), (10, 21, 4), (17, 22, 4), (30, 23, 4),
        (50, 102, 1), (51, 105, 1), (60, 107, 1), (80, 104, 1), (85, 111, 1),
    ])
    out = _iv(ev, w=(0, 100))
    rel = _rel(out["intervals"])
    assert ("call", 2, "call", 5.0, 25.0, False, False) in rel
    assert ("ped", 4, "walk", 10.0, 17.0, False, False) in rel
    assert ("ped", 4, "fdw", 17.0, 30.0, False, False) in rel
    assert ("preempt_call", 1, "call", 50.0, 80.0, False, False) in rel
    assert ("preempt", 1, "entry", 51.0, 60.0, False, False) in rel
    assert ("preempt", 1, "dwell", 60.0, 85.0, False, False) in rel
    assert not any(r[0] == "preempt" and r[2] not in ("entry", "track", "dwell") for r in rel)
    m = out["marks"]
    assert [(k, p, lab, round(t - T0)) for k, p, lab, t in m.itertuples(index=False)] == [
        ("ped", 4, "ped call", 8), ("preempt", 1, "exit", 85)]


def test_output_sorted_by_kind_then_param():
    ev = _ev([(5, 82, 9), (6, 81, 9), (5, 1, 6), (7, 43, 6), (8, 44, 6)])
    kinds = _iv(ev, w=(0, 10))["intervals"]["kind"].tolist()
    order = ["phase", "call", "ped", "preempt_call", "preempt", "detector"]
    assert kinds == sorted(kinds, key=order.index)


# ---------------------------------------------------------------------------
# Parity with the existing helpers (synthetic, then corpus)
# ---------------------------------------------------------------------------

def _synthetic_stream(seed=0, n_cycles=40):
    rng = np.random.default_rng(seed)
    rows, t = [], 0.0
    for _ in range(n_cycles):
        for ph in (2, 4):
            g = rng.uniform(5, 30)
            rows += [(t, 1, ph), (t + g, 8, ph), (t + g + 4, 9, ph)]
            if rng.random() < 0.7:
                rows += [(t + g + 4, 10, ph), (t + g + 5.5, 11, ph)]
            t += g + 6
        if rng.random() < 0.1:
            rows.append((t - 1, -1, 0))
    for det in (11, 12, 13):
        on = np.sort(rng.uniform(0, t, 120))
        dur = rng.uniform(0.1, 5, 120)
        rows += [(a, 82, det) for a in on] + [(a + d, 81, det) for a, d in zip(on, dur)]
    return _ev(rows), t


def _assert_parity(ev, w0, w1):
    iv = timing_actuation_intervals(ev, (w0, w1))["intervals"]

    pe = ev.loc[ev["event_code"].isin([1, 8, 9, 10, 11, 12, -1])].copy()
    pe["cycle_start"] = 0.0
    pe["_seg"] = _segment_id(pe)
    pe = pe.loc[pe["event_code"] != -1]
    bp = _build_phase_intervals(pe)
    for state, a, b in (("G", "green_ts", "yellow_ts"), ("Y", "yellow_ts", "yellow_end_ts")):
        got = iv.loc[(iv["kind"] == "phase") & (iv["state"] == state)]
        m = bp.merge(got, left_on=["phase", a], right_on=["param", "start_ts"], how="left")
        assert m["end_ts"].notna().all(), state
        np.testing.assert_allclose(m["end_ts"], m[b])

    for det in sorted(ev.loc[ev["event_code"] == 82, "parameter"].unique()):
        ref = _reconstruct_intervals(ev, int(det))
        got = iv.loc[(iv["kind"] == "detector") & (iv["param"] == det)
                     & ~iv["open_start"] & ~iv["open_end"]]
        m = ref.merge(got, left_on="on_ts", right_on="start_ts", how="left")
        assert m["end_ts"].notna().all(), det
        np.testing.assert_allclose(m["end_ts"], m["off_ts"])


def test_parity_synthetic():
    ev, t_end = _synthetic_stream()
    _assert_parity(ev, T0 - 1, T0 + t_end + 100)


_REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("folder,db,start", [
    ("201_SH-55_and_Banks-Lowman_Rd", "201_data.db", "2026-06-21T06:00"),
    ("315_US-20-26_Franklin_Rd_and_KCID_Rd", "315_data.db", "2025-12-15T06:00"),
])
def test_parity_corpus(folder, db, start):
    path = _REPO / "intersections" / folder / db
    if not path.exists():
        pytest.skip("corpus DB not present")
    import sqlite3
    t0 = pytz.timezone("US/Mountain").localize(datetime.fromisoformat(start)).timestamp()
    codes = ",".join(str(c) for c in TIMING_CODES)
    with sqlite3.connect(path) as con:
        ev = pd.read_sql(
            f"SELECT timestamp, event_code, parameter FROM events WHERE timestamp >= ? "
            f"AND timestamp < ? AND event_code IN ({codes}) ORDER BY timestamp, event_code, parameter",
            con, params=(t0, t0 + 4 * 3600))
    assert len(ev) > 1000
    _assert_parity(ev, t0, t0 + 4 * 3600)


# ---------------------------------------------------------------------------
# Ring order
# ---------------------------------------------------------------------------

def test_ring_phase_order():
    assert ring_phase_order({"RB_R1": "1,2|3,4", "RB_R2": "5,6|7,8"}) == [1, 2, 5, 6, 3, 4, 7, 8]
    assert ring_phase_order({"RB_R1": "2|4"}) == [2, 4]
    assert ring_phase_order({"RB_R1": "2,6|4", "RB_R2": "6|8"}) == [2, 6, 4, 8]
    assert ring_phase_order({}) == []


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

CFG = {
    "RB_R1": "1,2|3,4", "RB_R2": "5,6|7,8",
    "Det_P2_Arrival": "20", "Det_P2_Occupancy": "21,22", "Det_P2_Stop_Bar": "21,23",
    "Det_P4_Occupancy": "40",
    "Det_P6_Arrival": "60",          # silent, still gets a row
    "TM_NBT": "23", "TM_SBR": "45",  # 23 shown under P2, 45 goes to Other
    "WD_Sensor1": "56",
}


def _intervals(rows):
    df = pd.DataFrame(rows, columns=["kind", "param", "state"])
    df["start_ts"], df["end_ts"] = T0, T0 + 1
    df["open_start"] = df["open_end"] = False
    return df[INTERVAL_SCHEMA]


def _rows_list(rows):
    return [(r.block, r.kind, r.param, r.label) for r in rows.itertuples()]


def test_rows_layout():
    roles = parse_detector_roles(CFG)
    iv = _intervals([
        ("phase", 2, "G"), ("phase", 4, "G"), ("phase", 6, "G"), ("phase", 8, "G"),
        ("ped", 4, "walk"), ("preempt", 1, "dwell"),
        ("detector", 21, "on"), ("detector", 99, "on"), ("detector", 3, "on"),
    ])
    rows = timing_actuation_rows(roles, iv, phase_order=ring_phase_order(CFG))
    assert list(rows.columns) == ROW_SCHEMA
    assert rows["row"].tolist() == list(range(len(rows)))
    assert _rows_list(rows) == [
        ("Preempt", "preempt_call", 1, "Preempt 1 call"),
        ("Preempt", "preempt", 1, "Preempt 1"),
        ("P2", "phase", 2, "Phase 2"),
        ("P2", "call", 2, "Call 2"),
        ("P2", "detector", 20, "Arr 20"),
        ("P2", "detector", 21, "Occ 21"),     # occupancy before stop_bar; shown once
        ("P2", "detector", 22, "Occ 22"),
        ("P2", "detector", 23, "Stop 23"),
        ("P6", "phase", 6, "Phase 6"),
        ("P6", "call", 6, "Call 6"),
        ("P6", "detector", 60, "Arr 60"),
        ("P4", "phase", 4, "Phase 4"),
        ("P4", "call", 4, "Call 4"),
        ("P4", "ped", 4, "Ped 4"),
        ("P4", "detector", 40, "Occ 40"),
        ("P8", "phase", 8, "Phase 8"),
        ("P8", "call", 8, "Call 8"),
        ("Other", "detector", 45, "TM SBR 45"),
        ("Other", "detector", 56, "WD 56"),
        ("Unconfigured", "detector", 3, "Det 3"),
        ("Unconfigured", "detector", 99, "Det 99"),
    ]
    assert rows.loc[rows["param"] == 21, "role"].tolist() == ["occupancy"]


def test_rows_phase_not_in_ring_follows_numerically():
    roles = parse_detector_roles({"RB_R1": "2|4", "Det_P15_Occupancy": "70"})
    iv = _intervals([("phase", 2, "G"), ("phase", 4, "G"), ("phase", 16, "G")])
    rows = timing_actuation_rows(roles, iv, phase_order=[2, 4])
    assert rows.loc[rows["kind"] == "phase", "param"].tolist() == [2, 4, 15, 16]


def test_rows_no_phase_order_numeric():
    iv = _intervals([("phase", 6, "G"), ("phase", 2, "G")])
    rows = timing_actuation_rows(None, iv)
    assert rows.loc[rows["kind"] == "phase", "param"].tolist() == [2, 6]


def test_rows_ped_row_from_mark_only():
    iv = _intervals([("phase", 2, "G")])
    marks = pd.DataFrame({"kind": ["ped"], "param": [2], "label": ["ped call"], "ts": [T0]})
    rows = timing_actuation_rows(None, iv, marks)
    assert ("P2", "ped", 2, "Ped 2") in _rows_list(rows)


def test_rows_phase_filter():
    roles = parse_detector_roles(CFG)
    iv = _intervals([("phase", 2, "G"), ("preempt", 1, "dwell"), ("detector", 99, "on")])
    rows = timing_actuation_rows(roles, iv, phase_order=ring_phase_order(CFG), phases=[2])
    assert set(rows["block"]) == {"Preempt", "P2"}


def test_rows_detector_filter():
    roles = parse_detector_roles(CFG)
    iv = _intervals([("phase", 2, "G"), ("phase", 6, "G"), ("detector", 99, "on")])
    rows = timing_actuation_rows(roles, iv, phase_order=ring_phase_order(CFG), detectors=[60, 99, 77])
    assert _rows_list(rows) == [
        ("P6", "phase", 6, "Phase 6"),
        ("P6", "call", 6, "Call 6"),
        ("P6", "detector", 60, "Arr 60"),
        ("Unconfigured", "detector", 77, "Det 77"),   # requested, no data: still a row
        ("Unconfigured", "detector", 99, "Det 99"),
    ]


def test_rows_phase_and_detector_filter_intersect():
    roles = parse_detector_roles(CFG)
    iv = _intervals([("phase", 2, "G")])
    rows = timing_actuation_rows(roles, iv, phase_order=ring_phase_order(CFG),
                                 phases=[2], detectors=[60])
    # P2 holds no detector 60 and P6 is filtered out; 60 is configured, so it
    # isn't moved to Unconfigured either.
    assert rows.empty


def test_rows_detector_under_two_phases():
    roles = parse_detector_roles({"Det_P2_Occupancy": "21", "Det_P5_Occupancy": "21"})
    rows = timing_actuation_rows(roles, _intervals([]))
    assert rows.loc[rows["param"] == 21, "block"].tolist() == ["P2", "P5"]


def test_rows_empty():
    rows = timing_actuation_rows(None, _intervals([]))
    assert rows.empty and list(rows.columns) == ROW_SCHEMA


# ---------------------------------------------------------------------------
# Finding links
# ---------------------------------------------------------------------------

TZ = "US/Mountain"
DAY = date(2026, 3, 19)
MIDNIGHT = pytz.timezone(TZ).localize(datetime(2026, 3, 19)).timestamp()


def _f(rows):
    df = pd.DataFrame(rows, columns=FINDINGS_SCHEMA)
    return df.astype({"ts": "float64", "detector": "int64", "phase": "Int64"})


def _onsets(spec):
    """spec: {detector: {local_hour: n}} -> Code 82 events."""
    rows = []
    for det, by_h in spec.items():
        for h, n in by_h.items():
            for k in range(n):
                rows.append((MIDNIGHT + h * 3600 + 60 + k, 82, det))
    return pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])


def test_finding_windows():
    ts = MIDNIGHT + 1.92 * 3600
    f = _f([
        [DAY, "day", ts, -1, pd.NA, "unit:all", "Failsafe", "high", 30, 16, "m"],
        [DAY, "day", np.nan, 11, 2, "occupancy", "LowHits", "low", 1, 5, "m"],
        [DAY, "day", np.nan, 60, pd.NA, "", "ConfiguredSilent", "high", 0, 0, "m"],
        [DAY, "day", np.nan, -1, pd.NA, "", "RecordCount", "low", 0.5, 0.9, "m"],
        [date(2026, 3, 20), "day", np.nan, 11, pd.NA, "", "LowHits", "low", 0, 0, "m"],
    ])
    ev = _onsets({11: {9: 5, 15: 5}, 12: {17: 20, 8: 3}})
    out = finding_plot_windows(f, ev, TZ)
    assert out.index.equals(f.index)
    np.testing.assert_allclose(out["plot_start"].iloc[0], ts - 600)
    np.testing.assert_allclose(out["plot_end"].iloc[0], ts + 600)
    # own detector's busiest hour, tie -> earlier (09:00 local)
    np.testing.assert_allclose(out["plot_start"].iloc[1], MIDNIGHT + 9 * 3600)
    np.testing.assert_allclose(out["plot_end"].iloc[1], MIDNIGHT + 10 * 3600)
    # silent detector and intersection-level: all-detector busiest hour (17:00)
    np.testing.assert_allclose(out["plot_start"].iloc[2], MIDNIGHT + 17 * 3600)
    np.testing.assert_allclose(out["plot_start"].iloc[3], MIDNIGHT + 17 * 3600)
    # no onsets that date
    assert np.isnan(out["plot_start"].iloc[4])
    # row filter: phase when known, else detector (not -1)
    assert out["plot_phase"].isna().tolist() == [True, False, True, True, True]
    assert out["plot_phase"].iloc[1] == 2
    assert out["plot_detector"].isna().tolist() == [True, True, False, True, False]
    assert out["plot_detector"].iloc[2] == 60


def test_finding_windows_empty():
    out = finding_plot_windows(_f([]), _onsets({}), TZ)
    assert out.empty
    assert {"plot_start", "plot_end", "plot_phase", "plot_detector"} <= set(out.columns)
