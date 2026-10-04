"""Golden tests for Green Time Utilization (Functional Core).

Target: src/atspm/analysis/green_time_utilization.py (UDOT roadmap S-M7).  Opus-written.

Contract summary (UDOT ``GreenTimeUtilizationService.cs``)
---------------------------------------------------------
Rows are green events of one signal (phase, or overlap N).  An actuation
(Code 82) at t in [green, yellow) lies in bin floor((t − green) / bin_s).
Cells: actuations / n_cycles (UDOT, every cycle of the group), plus
n_reached, exposure_s and flow_vph over the green actually served.
Censored (flagged, counts NA, not binned): a gap marker in [green, end of
red clearance], or a green with no yellow.  Programmed green = split in force − the
cycle's own yellow + red clearance; NA when free (cycle 0), split 0, or in
overlap mode.

Timeline (seconds after each green, green length *d*)::

    0 green (1) · d yellow (8) · d+4 end yellow (9) + red clr (10) · d+6 end red clr (11)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.green_time_utilization import (
    ACTUATION_SCHEMA,
    BIN_SCHEMA,
    CYCLE_SCHEMA,
    SPLIT_SCHEMA,
    green_time_utilization,
    summarize_gtu_bins,
    summarize_gtu_splits,
)
from atspm.analysis.split_monitor import plan_timeline

_PH = 2
_A, _B, _C = 26, 27, 99
# Not a multiple of 0.5: epoch subtraction leaves ~1e-10 noise that a bare
# floor() would push across a bin edge.
_T0 = 1_000_000.3


def _green(g, d, ph=_PH, codes=(1, 8, 9, 10, 11)):
    c1, c8, c9, c10, c11 = codes
    return [(g, c1, ph), (g + d, c8, ph), (g + d + 4, c9, ph), (g + d + 4, c10, ph),
            (g + d + 6, c11, ph)]


def _events(greens, acts=(), gaps=(), extra=(), plan=1.0):
    """*greens* are ``(start, green_len)``; *acts* ``t`` (det A) or ``(t, det)``;
    *gaps* ``t`` or ``(t, param)``; *extra* raw ``(t, code, param)``."""
    rows = [(_T0 + t, c, p) for g, d in greens for t, c, p in _green(g, d)]
    for a in acts:
        t, d = (a, _A) if np.isscalar(a) else a
        rows += [(_T0 + t, 82, d), (_T0 + t + 0.05, 81, d)]
    for gp in gaps:
        t, p = (gp, -1) if np.isscalar(gp) else gp
        rows.append((_T0 + t, -1, p))
    rows += [(_T0 + t, c, p) for t, c, p in extra]
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cs = np.asarray([_T0 + g for g, _ in greens]) if greens else np.asarray([_T0])
    idx = np.searchsorted(cs, df["timestamp"].to_numpy(), side="right") - 1
    df["cycle_start"] = cs[np.clip(idx, 0, None)]
    df["coord_plan"] = plan
    return df


def _row(cy, green):
    return cy.loc[cy["green_ts"] == _T0 + green].iloc[0]


# Two greens: 10 s (bins 0–4) and 5 s (bins 0–2, bin 2 half served).
_GREENS = [(0, 10), (60, 5)]
_ACTS = [0.0, 1.99, 2.0, 9.9, 10.0, 12.0, 64.5]


# ---------------------------------------------------------------------------
# Cycles and binning
# ---------------------------------------------------------------------------

def test_schema_and_one_row_per_green():
    cy, ac = green_time_utilization(_events(_GREENS, _ACTS), _PH, [_A])
    assert list(cy.columns) == CYCLE_SCHEMA
    assert list(ac.columns) == ACTUATION_SCHEMA
    assert cy["green_ts"].tolist() == [_T0, _T0 + 60]
    assert cy["green_dur"].tolist() == [10.0, 5.0]
    assert cy["clearance_dur"].tolist() == [6.0, 6.0]
    assert not cy["censored"].any()
    assert cy["overlap"].isna().all() and (cy["phase"] == _PH).all()


def test_bins_from_green_start_half_open_at_yellow():
    cy, ac = green_time_utilization(_events(_GREENS, _ACTS), _PH, [_A])
    # 10.0 is the yellow instant and 12.0 is in yellow: neither counts.
    assert cy["actuations"].tolist() == [4, 1]
    assert ac["t_green"].tolist() == [0.0, 1.99, 2.0, 9.9, 4.5]
    assert ac["green_bin"].tolist() == [0, 0, 1, 4, 2]
    assert ac["green_ts"].tolist() == [_T0] * 4 + [_T0 + 60]


def test_bin_width_parameter():
    _, ac = green_time_utilization(_events(_GREENS, _ACTS), _PH, [_A], bin_s=5.0)
    assert ac["green_bin"].tolist() == [0, 0, 0, 1, 0]
    with pytest.raises(ValueError):
        green_time_utilization(_events(_GREENS, _ACTS), _PH, [_A], bin_s=0)


def test_actuation_before_green_is_not_counted():
    # 59.9 is red before the second green.
    cy, ac = green_time_utilization(_events(_GREENS, [59.9, 60.0]), _PH, [_A])
    assert cy["actuations"].tolist() == [0, 1]
    assert ac["t_green"].tolist() == [0.0]


def test_multiple_detectors_and_unlisted_ignored():
    ev = _events(_GREENS, [1.0, (1.5, _B), (2.5, _C)])
    cy, ac = green_time_utilization(ev, _PH, [_A, _B])
    assert cy["actuations"].tolist() == [2, 0]
    assert ac["detector"].tolist() == [_A, _B]


def test_coord_plan_from_green_event():
    ev = _events(_GREENS, [1.0])
    ev.loc[ev["timestamp"] >= _T0 + 60, "coord_plan"] = 3.0
    cy, _ = green_time_utilization(ev, _PH, [_A])
    assert cy["coord_plan"].tolist() == [1.0, 3.0]


# ---------------------------------------------------------------------------
# Gap rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param", [-1, -2])
def test_gap_in_green_censors(param):
    ev = _events([(0, 10), (60, 10), (120, 10)], [1.0, 61.0, 121.0], gaps=[(65, param)])
    cy, ac = green_time_utilization(ev, _PH, [_A])
    r = _row(cy, 60)
    assert r["censored"] and pd.isna(r["actuations"]) and np.isnan(r["green_dur"])
    assert cy["censored"].tolist() == [False, True, False]
    assert ac["green_ts"].tolist() == [_T0, _T0 + 120]


def test_gap_in_clearance_censors_gap_in_red_does_not():
    # The shared interval builder drops an interval with a gap marker
    # anywhere in [green, end of red clearance]: clearance_dur would be
    # wrong.  A gap in red (40) touches nothing.
    ev = _events([(0, 10), (60, 10)], [1.0, 61.0], gaps=[13])
    cy, _ = green_time_utilization(ev, _PH, [_A])
    assert cy["censored"].tolist() == [True, False]
    ev = _events([(0, 10), (60, 10)], [1.0, 61.0], gaps=[40])
    cy, _ = green_time_utilization(ev, _PH, [_A])
    assert not cy["censored"].any()
    assert cy["actuations"].tolist() == [1, 1]


def test_green_without_yellow_is_censored():
    ev = _events([(0, 10)], [1.0], extra=[(60, 1, _PH)])
    ev.loc[ev["timestamp"] == _T0 + 60, "cycle_start"] = _T0 + 60
    ev = pd.concat([ev, pd.DataFrame({"timestamp": [_T0 + 61], "event_code": [82],
                                      "parameter": [_A], "cycle_start": [_T0 + 60],
                                      "coord_plan": [1.0]})]).sort_values("timestamp")
    cy, ac = green_time_utilization(ev.reset_index(drop=True), _PH, [_A])
    assert cy["censored"].tolist() == [False, True]
    r = _row(cy, 60)
    assert pd.isna(r["yellow_ts"]) and pd.isna(r["actuations"])
    assert len(ac) == 1


# ---------------------------------------------------------------------------
# Programmed split
# ---------------------------------------------------------------------------

def _plan_rows(t, plan, cycle, split2):
    rows = [(t, 131, plan), (t, 132, cycle), (t, 133, 0)]
    rows += [(t, 133 + p, split2 if p == _PH else 20) for p in range(1, 17)]
    return rows


def test_programmed_green_is_split_less_own_clearance():
    ev = _events([(0, 10), (60, 10), (120, 10)], extra=_plan_rows(-5, 1, 100, 30)
                 + _plan_rows(100, 2, 100, 25))
    cy, _ = green_time_utilization(ev, _PH, [_A], timeline=plan_timeline(ev))
    assert cy["programmed_split"].tolist() == [30, 30, 25]
    assert cy["programmed_green"].tolist() == [24.0, 24.0, 19.0]


def test_programmed_split_na_when_free_or_not_in_plan_or_no_timeline():
    ev = _events([(0, 10), (60, 10)], extra=_plan_rows(-5, 0, 0, 0) + _plan_rows(50, 1, 100, 0))
    cy, _ = green_time_utilization(ev, _PH, [_A], timeline=plan_timeline(ev))
    assert cy["programmed_split"].isna().all() and cy["programmed_green"].isna().all()
    cy2, _ = green_time_utilization(_events(_GREENS), _PH, [_A])
    assert cy2["programmed_split"].isna().all()
    assert str(cy2["programmed_split"].dtype) == "Int64"


def test_programmed_green_floored_at_zero():
    ev = _events([(0, 10)], extra=_plan_rows(-5, 1, 100, 4))
    cy, _ = green_time_utilization(ev, _PH, [_A], timeline=plan_timeline(ev))
    assert cy["programmed_green"].tolist() == [0.0]


# ---------------------------------------------------------------------------
# Overlap mode
# ---------------------------------------------------------------------------

def test_overlap_mode_uses_overlap_greens_and_no_programmed_split():
    # Phase 1 green 0–5; overlap A green 0–30 (protected + FYA permissive).
    rows = _green(0, 5, ph=1) + _green(0, 30, ph=1, codes=(61, 63, 64, 64, 65))
    rows = [r for r in rows if not (r[1] == 64 and r[0] == 34)]
    extra = rows + _plan_rows(-5, 1, 100, 30)
    ev = _events([], [(2.0, _A), (20.0, _A)], extra=extra)
    tl = plan_timeline(ev)
    cy_ph, ac_ph = green_time_utilization(ev, 1, [_A], timeline=tl)
    cy_ol, ac_ol = green_time_utilization(ev, 1, [_A], overlap=1, timeline=tl)
    assert cy_ph["green_dur"].tolist() == [5.0] and cy_ph["actuations"].tolist() == [1]
    assert cy_ol["green_dur"].tolist() == [30.0] and cy_ol["actuations"].tolist() == [2]
    assert (cy_ol["overlap"] == 1).all() and (cy_ol["phase"] == 1).all()
    assert ac_ol["green_bin"].tolist() == [1, 10]
    assert cy_ph["programmed_split"].notna().all()
    assert cy_ol["programmed_split"].isna().all()


# ---------------------------------------------------------------------------
# Exclusions, dtypes, empty input
# ---------------------------------------------------------------------------

def test_exclusions_drop_actuations_in_state():
    ev = _events(_GREENS, [1.0, 3.0, (4.0, _B)])
    exc = [{"detector": _A, "phase": _PH, "status": "Green"}]
    cy, ac = green_time_utilization(ev, _PH, [_A, _B], exclusions=exc)
    assert cy["actuations"].tolist() == [1, 0]
    assert ac["detector"].tolist() == [_B]


def test_tz_aware_timestamps_match_epoch():
    ev = _events(_GREENS, _ACTS, extra=_plan_rows(-5, 1, 100, 30))
    cy_e, ac_e = green_time_utilization(ev, _PH, [_A], timeline=plan_timeline(ev))
    tz = ev.copy()
    for col in ("timestamp", "cycle_start"):
        tz[col] = pd.to_datetime(tz[col], unit="s", utc=True).dt.tz_convert("US/Mountain")
    cy_t, ac_t = green_time_utilization(tz, _PH, [_A], timeline=plan_timeline(tz))
    for col in ("green_dur", "clearance_dur", "actuations", "programmed_green"):
        np.testing.assert_array_equal(cy_t[col].astype(float).to_numpy(),
                                      cy_e[col].astype(float).to_numpy())
    assert ac_t["green_bin"].tolist() == ac_e["green_bin"].tolist()
    assert str(cy_t["green_ts"].dt.tz) == "US/Mountain"
    assert str(ac_t["timestamp"].dt.tz) == "US/Mountain"
    b_e = summarize_gtu_bins(cy_e, ac_e)
    b_t = summarize_gtu_bins(cy_t, ac_t)
    assert b_t["actuations"].tolist() == b_e["actuations"].tolist()


def test_dtypes():
    cy, ac = green_time_utilization(_events(_GREENS, _ACTS), _PH, [_A])
    assert str(cy["actuations"].dtype) == "Int64"
    assert str(cy["overlap"].dtype) == "Int64"
    assert cy["censored"].dtype == bool
    assert ac["green_bin"].dtype == np.int64 and ac["detector"].dtype == np.int64


def test_empty_inputs():
    empty = pd.DataFrame(columns=["timestamp", "event_code", "parameter", "cycle_start",
                                  "coord_plan"])
    cy, ac = green_time_utilization(empty, _PH, [_A])
    assert cy.empty and list(cy.columns) == CYCLE_SCHEMA
    assert ac.empty and list(ac.columns) == ACTUATION_SCHEMA
    cy, ac = green_time_utilization(_events(_GREENS, _ACTS), 7, [_A])
    assert cy.empty and ac.empty
    cy, ac = green_time_utilization(_events(_GREENS, _ACTS), _PH, [])
    assert cy["actuations"].tolist() == [0, 0] and ac.empty
    assert list(summarize_gtu_bins(cy.iloc[:0], ac).columns) == BIN_SCHEMA
    assert list(summarize_gtu_splits(cy.iloc[:0]).columns) == SPLIT_SCHEMA


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def test_bin_summary_denominators():
    cy, ac = green_time_utilization(_events(_GREENS, _ACTS), _PH, [_A])
    b = summarize_gtu_bins(cy, ac, bin_s=2.0, bin_len=15)
    assert list(b.columns) == BIN_SCHEMA
    assert b["green_bin"].tolist() == [0, 1, 2, 3, 4]
    assert b["bin_start_s"].tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert (b["n_cycles"] == 2).all()
    assert b["n_reached"].tolist() == [2, 2, 2, 1, 1]
    assert b["exposure_s"].tolist() == [4.0, 4.0, 3.0, 2.0, 2.0]
    assert b["actuations"].tolist() == [2, 1, 1, 0, 1]
    assert b["act_per_cycle"].tolist() == [1.0, 0.5, 0.5, 0.0, 0.5]
    assert b["act_per_reached"].tolist() == [1.0, 0.5, 0.5, 0.0, 1.0]
    assert b["flow_vph"].tolist() == [1800.0, 900.0, 1200.0, 0.0, 1800.0]
    assert str(b["time"].dt.tz) == "UTC"


def test_bin_summary_green_exactly_on_bin_edge():
    # A 10 s green fills bins 0–4 and reaches no bin 5.
    cy, ac = green_time_utilization(_events([(0, 10)], [9.99]), _PH, [_A])
    b = summarize_gtu_bins(cy, ac)
    assert b["green_bin"].max() == 4
    assert b["exposure_s"].tolist() == [2.0] * 5


def test_bin_summary_excludes_censored_cycles():
    ev = _events([(0, 10), (60, 10), (120, 4)], [1.0, 61.0, 121.0], gaps=[65])
    cy, ac = green_time_utilization(ev, _PH, [_A])
    b = summarize_gtu_bins(cy, ac)
    assert (b["n_cycles"] == 2).all()
    assert b["n_reached"].tolist() == [2, 2, 1, 1, 1]
    assert b.loc[0, "actuations"] == 2


def test_bin_summary_time_bins_and_per_plan():
    # Greens at 0 and 1000 s fall in different 15-min bins; plan changes at 1000.
    ev = _events([(0, 4), (1000, 4)], [1.0, 1001.0, 1003.0])
    ev.loc[ev["timestamp"] >= _T0 + 1000, "coord_plan"] = 2.0
    cy, ac = green_time_utilization(ev, _PH, [_A])
    b = summarize_gtu_bins(cy, ac, bin_len=15)
    assert b["time"].nunique() == 2 and len(b) == 4
    p = summarize_gtu_bins(cy, ac, bin_len=None)
    assert p["coord_plan"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert p["actuations"].tolist() == [1, 0, 1, 1]
    assert p.loc[p["coord_plan"] == 2.0, "time"].iloc[0] == pd.Timestamp(
        _T0 + 1000, unit="s", tz="UTC")


def test_split_summary():
    ev = _events([(0, 10), (60, 6), (120, 4)], [1.0, 2.0, 61.0], gaps=[125],
                 extra=_plan_rows(-5, 1, 100, 30))
    cy, _ = green_time_utilization(ev, _PH, [_A], timeline=plan_timeline(ev))
    s = summarize_gtu_splits(cy, bin_len=15)
    assert list(s.columns) == SPLIT_SCHEMA and len(s) == 1
    r = s.iloc[0]
    assert (r["n_cycles"], r["n_censored"], r["actuations"]) == (2, 1, 3)
    assert r["avg_green_s"] == 8.0 and r["avg_clearance_s"] == 6.0
    assert r["programmed_split"] == 30.0 and r["programmed_green"] == 24.0
    assert r["act_per_cycle"] == 1.5


def test_split_summary_needs_most_greens_programmed():
    # Plan label 0 on all three greens; only the last sees a running plan
    # (a label lagging the plan change).  1 of 3 → no programmed split.
    ev = _events([(0, 10), (60, 10), (120, 10)], plan=0.0,
                 extra=_plan_rows(-5, 0, 0, 0) + _plan_rows(110, 1, 100, 30))
    cy, _ = green_time_utilization(ev, _PH, [_A], timeline=plan_timeline(ev))
    assert cy["programmed_split"].notna().tolist() == [False, False, True]
    s = summarize_gtu_splits(cy, bin_len=None)
    assert np.isnan(s.loc[0, "programmed_split"]) and np.isnan(s.loc[0, "programmed_green"])
    # 2 of 3 is enough.
    ev = _events([(0, 10), (60, 10), (120, 10)], plan=1.0,
                 extra=_plan_rows(-5, 0, 0, 0) + _plan_rows(50, 1, 100, 30))
    cy, _ = green_time_utilization(ev, _PH, [_A], timeline=plan_timeline(ev))
    s = summarize_gtu_splits(cy, bin_len=None)
    assert s.loc[0, "programmed_green"] == 24.0


def test_split_summary_keeps_censored_only_group():
    ev = _events([(0, 10), (1000, 10)], [1.0], gaps=[1005])
    cy, _ = green_time_utilization(ev, _PH, [_A])
    s = summarize_gtu_splits(cy, bin_len=15)
    assert s["n_cycles"].tolist() == [1, 0]
    assert s["n_censored"].tolist() == [0, 1]
    assert np.isnan(s.loc[1, "avg_green_s"]) and np.isnan(s.loc[1, "act_per_cycle"])
    assert s.loc[1, "actuations"] == 0


# ---------------------------------------------------------------------------
# Corpus: 315, Monday 2025-12-15
# ---------------------------------------------------------------------------

_DB = Path(__file__).resolve().parents[2] / "intersections" / \
    "315_US-20-26_Franklin_Rd_and_KCID_Rd" / "315_data.db"
_CODES = [-1, 1, 8, 9, 10, 11, 12, 61, 63, 64, 65, 82] + list(range(131, 150))


@pytest.fixture(scope="module")
def ev315():
    if not _DB.exists():
        pytest.skip("corpus DB not present")
    from atspm.data.reader import get_events_with_cycles_df
    # From the previous midnight's plan dump.
    ev = get_events_with_cycles_df(_DB, datetime(2025, 12, 14), datetime(2025, 12, 16),
                                   event_codes=_CODES, timezone="US/Mountain")
    tl = plan_timeline(ev)
    day = ev.loc[ev["timestamp"] >= pd.Timestamp("2025-12-15", tz="US/Mountain")]
    return day.reset_index(drop=True), tl


def _plan1(ev_tl, phase, dets, **kw):
    ev, tl = ev_tl
    cy, ac = green_time_utilization(ev, phase, dets, timeline=tl, **kw)
    return cy, summarize_gtu_bins(cy, ac, bin_len=None).query("coord_plan == 1.0"), \
        summarize_gtu_splits(cy, bin_len=None).query("coord_plan == 1.0").iloc[0]


def test_corpus_315_stop_bar_loops_show_queue_discharge(ev315):
    # P6 count loops just past the line: nothing in the first 2 s, the
    # queue's discharge peak at 4–6 s, then decay.
    cy, b, s = _plan1(ev315, 6, [18, 19, 20])
    assert int(cy["censored"].sum()) <= 1
    apc = b.set_index("green_bin")["act_per_cycle"]
    assert apc[0] == 0 and apc.idxmax() == 2 and apc[2] > 1.5
    assert (s["n_cycles"], s["programmed_split"], s["programmed_green"]) == (234, 40.0, 33.9)
    # Coordinated phase: early gap-outs of the minors give it more than
    # its programmed green.
    assert s["avg_green_s"] > s["programmed_green"]


def test_corpus_315_presence_zones_miss_the_discharge(ev315):
    # The stop-line presence zones (Det_P6_Occupancy) are already occupied
    # at green by the queue, so their profile has no discharge peak.
    _, b_sb, _ = _plan1(ev315, 6, [18, 19, 20])
    _, b_oc, _ = _plan1(ev315, 6, [34, 35, 36])
    sb = b_sb.set_index("green_bin")["act_per_cycle"]
    oc = b_oc.set_index("green_bin")["act_per_cycle"]
    assert oc[2] < 0.5 * sb[2]
    assert oc[:4].max() < 0.5


def test_corpus_315_protected_left_and_its_overlap(ev315):
    # P1's protected arrow is short and uses its programmed green; overlap A
    # spans the permissive flashing yellow too, so it carries no programmed
    # split and its green is several times longer.
    cy_ph, b_ph, s_ph = _plan1(ev315, 1, [17])
    cy_ol, _, s_ol = _plan1(ev315, 1, [17], overlap=1)
    assert s_ph["avg_green_s"] < 10 and s_ph["programmed_green"] == 11.3
    assert s_ol["avg_green_s"] > 4 * s_ph["avg_green_s"]
    assert np.isnan(s_ol["programmed_split"])
    # Unused tail: no actuation past 10 s of the protected arrow.
    assert b_ph.loc[b_ph["green_bin"] >= 5, "actuations"].sum() == 0
