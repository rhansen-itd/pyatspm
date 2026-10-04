"""Golden tests for Yellow and Red Actuations (Functional Core).

Target: src/atspm/analysis/yellow_red_actuations.py (UDOT roadmap S-M4).  Opus-written.

Contract summary (UDOT ``YellowRedActivationsCycle.cs``)
-------------------------------------------------------
Rows are green-to-green cycles of one signal (phase, or overlap N):
    green      [green, yellow)
    yellow     [yellow, end of yellow]            (Code 9, else 10)
    red_clear  (end of yellow, end of red clearance]   (Code 11)
    red        (end of red clearance, next green)
violation = t > end of yellow; severe = t − end of yellow > severe_sec.
Censored (flagged, counts NA, actuations not reported): a gap marker in
[green, next green), or no next green.  Exclusions use the counts helper.

Timeline used throughout (seconds after each green)::

    0 green (1) · 20 yellow (8) · 24 end yellow (9) + red clearance (10)
    · 26 end red clearance (11) · 60 next green
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.detector_roles import phase_overlaps
from atspm.analysis.yellow_red_actuations import (
    ACTUATION_SCHEMA,
    CYCLE_SCHEMA,
    SUMMARY_SCHEMA,
    summarize_yellow_red,
    yellow_red_actuations,
)

_PH = 2
_A, _B, _C = 26, 27, 99
_T0 = 1_000_000.0


def _green(g, ph=_PH, rc=True):
    if rc:
        return [(g, 1, ph), (g + 20, 8, ph), (g + 24, 9, ph), (g + 24, 10, ph), (g + 26, 11, ph)]
    return [(g, 1, ph), (g + 20, 8, ph), (g + 24, 9, ph), (g + 24, 12, ph)]


def _events(greens, acts=(), gaps=(), extra=(), plan=1.0, rc=True):
    """Flat events; *acts* are ``t`` (detector A) or ``(t, det)``; *gaps* ``t`` or ``(t, param)``."""
    rows = [(_T0 + t, c, p) for g in greens for t, c, p in _green(g, rc=rc)]
    for a in acts:
        t, d = (a, _A) if np.isscalar(a) else a
        rows += [(_T0 + t, 82, d), (_T0 + t + 0.05, 81, d)]
    for gp in gaps:
        t, p = (gp, -1) if np.isscalar(gp) else gp
        rows.append((_T0 + t, -1, p))
    rows += [(_T0 + t, c, p) for t, c, p in extra]
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cs = np.asarray([_T0 + g for g in greens]) if greens else np.asarray([_T0])
    idx = np.searchsorted(cs, df["timestamp"].to_numpy(), side="right") - 1
    df["cycle_start"] = cs[np.clip(idx, 0, None)]
    df["coord_plan"] = plan
    return df


def _row(cy, green):
    return cy.loc[cy["green_ts"] == _T0 + green].iloc[0]


# ---------------------------------------------------------------------------
# Cycles and classification
# ---------------------------------------------------------------------------

def test_schema_and_one_row_per_green():
    cy, ac = yellow_red_actuations(_events([0, 60, 120], [10]), _PH, [_A])
    assert list(cy.columns) == CYCLE_SCHEMA
    assert list(ac.columns) == ACTUATION_SCHEMA
    assert cy["green_ts"].tolist() == [_T0, _T0 + 60, _T0 + 120]
    # The last green has no next green: its red end is unknown.
    assert cy["censored"].tolist() == [False, False, True]
    r = _row(cy, 0)
    assert r["red_end_ts"] == _T0 + 60
    assert (r["green_dur"], r["yellow_dur"], r["red_clear_dur"], r["red_dur"]) == (20, 4, 2, 34)
    assert pd.isna(_row(cy, 120)["red_end_ts"])
    assert pd.isna(_row(cy, 120)["volume"])
    assert pd.isna(cy["overlap"]).all()


def test_udot_state_boundaries():
    acts = [10, 20, 24, 24.1, 26, 26.1, 59.9, 60]
    cy, ac = yellow_red_actuations(_events([0, 60, 120], acts), _PH, [_A])
    r = _row(cy, 0)
    assert (r["volume"], r["green_act"], r["yellow_act"], r["red_clear_act"], r["red_act"]) == (7, 1, 2, 2, 2)
    assert r["violations"] == 4
    first = ac.loc[ac["green_ts"] == _T0]
    assert first["state"].tolist() == ["green", "yellow", "yellow", "red_clear", "red_clear", "red", "red"]
    assert first["violation"].tolist() == [False, False, False, True, True, True, True]
    # The actuation at the next green belongs to the next cycle, as green.
    nxt = ac.loc[ac["green_ts"] == _T0 + 60]
    assert nxt["state"].tolist() == ["green"]
    assert _row(cy, 60)["green_act"] == 1


def test_times_relative_to_yellow_and_red():
    cy, ac = yellow_red_actuations(_events([0, 60, 120], [21, 23, 25, 30]), _PH, [_A])
    assert ac["t_yellow"].tolist() == [1.0, 3.0, 5.0, 10.0]
    assert ac["t_red"].tolist() == [-3.0, -1.0, 1.0, 6.0]
    r = _row(cy, 0)
    assert r["yellow_time_s"] == pytest.approx(4.0)       # 1 + 3, time into yellow
    assert r["violation_time_s"] == pytest.approx(7.0)    # 1 + 6, time into red


def test_severe_is_strictly_more_than_n_seconds_into_red():
    acts = [25, 28, 28.1, 50]                              # t_red 1, 4, 4.1, 26
    cy, ac = yellow_red_actuations(_events([0, 60, 120], acts), _PH, [_A])
    assert ac["severe"].tolist() == [False, False, True, True]
    assert _row(cy, 0)["severe"] == 2
    cy1, _ = yellow_red_actuations(_events([0, 60, 120], acts), _PH, [_A], severe_sec=0.5)
    assert _row(cy1, 0)["severe"] == 4
    cy2, _ = yellow_red_actuations(_events([0, 60, 120], acts), _PH, [_A], severe_sec=30)
    assert _row(cy2, 0)["severe"] == 0


def test_no_red_clearance_served():
    # 201 style: 8 → 9 → 12, no 10/11; red starts at the end of yellow.
    cy, ac = yellow_red_actuations(_events([0, 60, 120], [24, 24.1], rc=False), _PH, [_A])
    r = _row(cy, 0)
    assert r["red_clear_ts"] == r["red_ts"] == _T0 + 24
    assert r["red_clear_dur"] == 0
    assert ac["state"].tolist() == ["yellow", "red"]


def test_multiple_detectors_and_unlisted_ignored():
    acts = [(10, _A), (25, _B), (30, _C), (40, _B)]
    cy, ac = yellow_red_actuations(_events([0, 60, 120], acts), _PH, [_A, _B])
    r = _row(cy, 0)
    assert (r["volume"], r["violations"]) == (3, 2)
    assert ac["detector"].tolist() == [_A, _B, _B]
    assert ac["detector"].dtype == np.int64


def test_coord_plan_from_green_and_phase_column():
    cy, _ = yellow_red_actuations(_events([0, 60, 120], [], plan=7.0), _PH, [_A])
    assert (cy["coord_plan"] == 7.0).all()
    assert (cy["phase"] == _PH).all()


# ---------------------------------------------------------------------------
# Gap markers and lost greens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param", [-1, -2])
def test_gap_in_red_censors_cycle(param):
    ev = _events([0, 60, 120, 180], [25, 85, 145], gaps=[(70, param)])
    cy, ac = yellow_red_actuations(ev, _PH, [_A])
    # The gap at 70 lies in the 60 cycle's green: that interval is dropped by
    # the builder; the 0 cycle ends at the raw green at 60 and stays valid.
    assert _row(cy, 0)["censored"] == False  # noqa: E712
    assert _T0 + 60 not in cy["green_ts"].tolist()
    ev2 = _events([0, 60, 120, 180], [25, 85, 145], gaps=[(40, param)])
    cy2, ac2 = yellow_red_actuations(ev2, _PH, [_A])
    r = _row(cy2, 0)
    assert r["censored"] and pd.isna(r["violations"]) and np.isnan(r["violation_time_s"])
    assert _T0 + 25 not in ac2["timestamp"].tolist()
    assert ac2["green_ts"].tolist() == [_T0 + 60, _T0 + 120]


def test_lost_green_end_closes_the_red():
    # A Code 1 at 40 with no end: the green at 0's red ends there.
    ev = _events([0, 60, 120], [30, 50], extra=[(40, 1, _PH)])
    cy, ac = yellow_red_actuations(ev, _PH, [_A])
    r = _row(cy, 0)
    assert r["red_end_ts"] == _T0 + 40 and r["red_dur"] == 14
    assert (r["red_act"], r["volume"]) == (1, 1)
    assert ac["timestamp"].tolist() == [_T0 + 30]


# ---------------------------------------------------------------------------
# Overlap mode and exclusions
# ---------------------------------------------------------------------------

def _overlap(g, ol=1):
    return [(g, 61, ol), (g + 20, 63, ol), (g + 24, 64, ol), (g + 26, 65, ol)]


def test_overlap_mode_classifies_against_the_overlap():
    # The phase itself turns red at 10 (protected portion ends); the overlap
    # (FYA) stays green to 20.  An actuation at 15 is a violation of the
    # phase but green for the overlap.
    phase_ev = [(t, c, 1) for g in (0, 60, 120) for t, c in
                [(g, 1), (g + 6, 8), (g + 9, 9), (g + 9, 10), (g + 10, 11)]]
    ol_ev = [e for g in (0, 60, 120) for e in _overlap(g)]
    ev = _events([], [15, 25, 30], extra=phase_ev + ol_ev)
    cy_ph, _ = yellow_red_actuations(ev, 1, [_A])
    cy_ol, ac_ol = yellow_red_actuations(ev, 1, [_A], overlap=1)
    assert _row(cy_ph, 0)["violations"] == 3
    r = _row(cy_ol, 0)
    assert (r["green_act"], r["red_clear_act"], r["red_act"], r["violations"]) == (1, 1, 1, 2)
    assert r["red_clear_ts"] == _T0 + 24 and r["red_ts"] == _T0 + 26
    assert (cy_ol["overlap"] == 1).all() and (cy_ol["phase"] == 1).all()
    assert ac_ol["state"].tolist() == ["green", "red_clear", "red"]


def test_exclusions_drop_actuations_in_state():
    ev = _events([0, 60, 120], [10, 22, 25, 40, (41, _B)])
    exc = [{"detector": _A, "phase": _PH, "status": "Red"}]
    cy, ac = yellow_red_actuations(ev, _PH, [_A, _B], exclusions=exc)
    r = _row(cy, 0)
    # A's red-clearance and red actuations go; its green/yellow and B stay.
    assert (r["volume"], r["green_act"], r["yellow_act"], r["violations"]) == (3, 1, 1, 1)
    assert ac["detector"].tolist() == [_A, _A, _B]


# ---------------------------------------------------------------------------
# Dtypes and empty input
# ---------------------------------------------------------------------------

def test_tz_aware_timestamps_match_epoch():
    ev = _events([0, 60, 120], [10, 22, 25, 50])
    cy_e, ac_e = yellow_red_actuations(ev, _PH, [_A])
    tz = ev.copy()
    for col in ("timestamp", "cycle_start"):
        tz[col] = pd.to_datetime(tz[col], unit="s", utc=True).dt.tz_convert("US/Mountain")
    cy_t, ac_t = yellow_red_actuations(tz, _PH, [_A])
    for col in ("volume", "violations", "severe", "violation_time_s", "red_dur"):
        np.testing.assert_array_equal(cy_t[col].astype(float).to_numpy(),
                                      cy_e[col].astype(float).to_numpy())
    assert str(cy_t["green_ts"].dt.tz) == "US/Mountain"
    assert str(cy_t["red_end_ts"].dt.tz) == "US/Mountain"
    assert str(ac_t["timestamp"].dt.tz) == "US/Mountain"
    assert ac_t["t_red"].tolist() == ac_e["t_red"].tolist()


def test_count_dtypes():
    cy, _ = yellow_red_actuations(_events([0, 60, 120], [25]), _PH, [_A])
    for col in ("volume", "green_act", "yellow_act", "red_clear_act", "red_act", "violations", "severe"):
        assert str(cy[col].dtype) == "Int64"
    assert cy["censored"].dtype == bool


def test_empty_inputs():
    cy, ac = yellow_red_actuations(_events([], []), _PH, [_A])
    assert cy.empty and list(cy.columns) == CYCLE_SCHEMA and list(ac.columns) == ACTUATION_SCHEMA
    cy, ac = yellow_red_actuations(_events([0, 60, 120], [25]), _PH, [])
    assert len(cy) == 3 and ac.empty
    assert _row(cy, 0)["volume"] == 0
    cy, ac = yellow_red_actuations(_events([0, 60, 120], [25]), 4, [_A])   # phase never served
    assert cy.empty and ac.empty


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_bins_and_ratios():
    # Cycles at 0 and 60 (same 15-min bin): 0 → green 1, yellow 1, red 2 (one
    # severe, t_red 1 and 10); 60 → green 1, red_clear 1 (t_red 1).
    acts = [10, 22, 25, 34, 70, 85]
    cy, _ = yellow_red_actuations(_events([0, 60, 120], acts), _PH, [_A])
    s = summarize_yellow_red(cy, bin_len=15)
    assert list(s.columns) == SUMMARY_SCHEMA
    assert len(s) == 1
    r = s.iloc[0]
    assert (r["n_cycles"], r["n_censored"], r["volume"], r["violations"], r["severe"]) == (2, 1, 6, 3, 1)
    assert r["pct_violations"] == pytest.approx(0.5)
    assert r["pct_severe"] == pytest.approx(1 / 6, abs=1e-4)
    assert r["pct_violations_udot"] == pytest.approx(3 / 4)
    assert r["violations_per_cycle"] == pytest.approx(1.5)
    assert r["avg_violation_time_s"] == pytest.approx((1 + 10 + 1) / 3)
    assert r["avg_yellow_time_s"] == pytest.approx(2.0)
    assert r["time"] == pd.Timestamp(_T0, unit="s", tz="UTC").floor("15min")


def test_summary_per_plan_and_censored_only_group():
    # Plan 1 greens at 0 and 60 (the 60 cycle's red runs to the plan-2 green
    # at 1000); plan 2's only green has no next green, so it is censored.
    ev = pd.concat([
        _events([0, 60], [25], plan=1.0),
        _events([1000], [], plan=2.0),
    ]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    cy, _ = yellow_red_actuations(ev, _PH, [_A])
    s = summarize_yellow_red(cy, bin_len=None)
    assert s["coord_plan"].tolist() == [1.0, 2.0]
    assert s["n_cycles"].tolist() == [2, 0] and s["n_censored"].tolist() == [0, 1]
    assert s["violations"].tolist() == [1, 0]
    assert np.isnan(s["pct_violations"].iloc[1]) and np.isnan(s["violations_per_cycle"].iloc[1])
    assert s["time"].tolist() == [pd.Timestamp(_T0, unit="s", tz="UTC"),
                                  pd.Timestamp(_T0 + 1000, unit="s", tz="UTC")]
    assert summarize_yellow_red(pd.DataFrame(columns=CYCLE_SCHEMA)).columns.tolist() == SUMMARY_SCHEMA


# ---------------------------------------------------------------------------
# Config: Det_P{N}_Overlap
# ---------------------------------------------------------------------------

def test_phase_overlaps_parsing():
    cfg = {"Det_P1_Overlap": "1", "Det_P3_Overlap": "b", "Det_P5_Overlap": " C ",
           "Det_P7_Overlap": "", "Det_P8_Overlap": float("nan"), "Det_P2_Stop_Bar": "26"}
    assert phase_overlaps(cfg) == {1: 1, 3: 2, 5: 3}
    for bad in ("0", "17", "Q", "1.5"):
        with pytest.raises(ValueError):
            phase_overlaps({"Det_P1_Overlap": bad})


# ---------------------------------------------------------------------------
# Corpus: 315, Monday 2025-12-15
# ---------------------------------------------------------------------------

_DB = Path(__file__).resolve().parents[2] / "intersections" / \
    "315_US-20-26_Franklin_Rd_and_KCID_Rd" / "315_data.db"
_CODES = [-1, 1, 8, 9, 10, 11, 12, 61, 63, 64, 65, 81, 82]


@pytest.fixture(scope="module")
def ev315():
    if not _DB.exists():
        pytest.skip("corpus DB not present")
    from atspm.data.reader import get_events_with_cycles_df
    return get_events_with_cycles_df(_DB, datetime(2025, 12, 15), datetime(2025, 12, 16),
                                     event_codes=_CODES, timezone="US/Mountain")


def test_corpus_315_stop_bar_loops(ev315):
    # P6 count loops just past the stop bar (Det_P6_Stop_Bar).
    cy, ac = yellow_red_actuations(ev315, 6, [18, 19, 20])
    ok = cy.loc[~cy["censored"]]
    assert len(ok) == 952
    assert ok[["yellow_act", "red_clear_act", "red_act", "severe"]].sum().tolist() == [247, 7, 14, 6]
    # Red running clusters at the start of red.
    assert ac.loc[ac["violation"], "t_red"].median() < 4


def test_corpus_315_presence_zones_count_stopping_vehicles(ev315):
    # The stop-line presence zones (Det_P6_Occupancy) see every vehicle that
    # stops on red: ~1.3 red actuations per cycle against ~0.02 on the loops.
    cy, _ = yellow_red_actuations(ev315, 6, [34, 35, 36])
    ok = cy.loc[~cy["censored"]]
    assert ok["violations"].sum() / len(ok) > 1.0
    cy_sb, _ = yellow_red_actuations(ev315, 6, [18, 19, 20])
    ok_sb = cy_sb.loc[~cy_sb["censored"]]
    assert ok_sb["violations"].sum() / len(ok_sb) < 0.05


def test_corpus_315_fya_left_needs_its_overlap(ev315):
    # P1's left loop (17) is served permissively under overlap A's flashing
    # yellow arrow while phase 1 is red.
    cy_ph, _ = yellow_red_actuations(ev315, 1, [17])
    cy_ol, _ = yellow_red_actuations(ev315, 1, [17], overlap=1)
    assert cy_ph["red_act"].sum() > 400
    assert cy_ol["red_act"].sum() < 30
    assert cy_ol["volume"].sum() == cy_ph["volume"].sum()
