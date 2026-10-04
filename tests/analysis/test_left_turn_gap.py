"""Golden tests for Left Turn Gap Analysis (Functional Core).

Target: src/atspm/analysis/left_turn_gap.py (UDOT roadmap S-M9).  Opus-written.

Contract summary (UDOT ``LeftTurnGapAnalysisService.cs``)
--------------------------------------------------------
Rows are green events of the *opposing through* phase.  The window is
green (Code 1) → red clearance (Code 9/10), yellow included.  Gaps are the
consecutive differences of [green, off-events (Code 81) of the union of
opposing detectors strictly inside the window, red clearance].  Bin i is
(edges[i-1], edges[i]]; a gap ≤ edges[0] counts in n_short.  turnable_s
sums gaps ≥ trend_s; sum_ge_critical sums gaps ≥ critical_s.  Censored
(flagged, counts NA, no gap rows): a gap marker in [green, end of red
clearance], or a green with no yellow.

Timeline (seconds after each green, green length *d*)::

    0 green (1) · d yellow (8) · d+4 end yellow (9) + red clr (10) · d+6 end red clr (11)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.detector_roles import phase_directions
from atspm.analysis.left_turn_gap import (
    DEFAULT_EDGES,
    GAP_SCHEMA,
    PAIR_SCHEMA,
    THROUGH_SCHEMA,
    bin_columns,
    bin_labels,
    critical_gap,
    cycle_schema,
    left_turn_gaps,
    left_turn_pairs,
    summarize_left_turn_gaps,
    summary_schema,
    through_phases,
)

_PH = 6
_A, _B, _C = 18, 19, 99
# Realistic epoch magnitude: decisecond differences carry ~1e-7 noise, so
# 3.3 s is not exactly 3.3 without rounding.
_T0 = 1_765_800_000.3
_TZ = "US/Mountain"


def _green(g, d, ph=_PH):
    return [(g, 1, ph), (g + d, 8, ph), (g + d + 4, 9, ph), (g + d + 4, 10, ph),
            (g + d + 6, 11, ph)]


def _events(greens, offs=(), gaps=(), extra=(), plan=1.0):
    """*greens* ``(start, green_len)``; *offs* ``t`` (det A) or ``(t, det)``,
    each logged as an on 0.3 s before the off; *gaps* ``t``; *extra* raw
    ``(t, code, param)``."""
    rows = [(_T0 + t, c, p) for g, d in greens for t, c, p in _green(g, d)]
    for a in offs:
        t, d = (a, _A) if np.isscalar(a) else a
        rows += [(_T0 + t - 0.3, 82, d), (_T0 + t, 81, d)]
    rows += [(_T0 + t, -1, 0) for t in gaps]
    rows += [(_T0 + t, c, p) for t, c, p in extra]
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cs = np.asarray([_T0 + g for g, _ in greens]) if greens else np.asarray([_T0])
    idx = np.searchsorted(cs, df["timestamp"].to_numpy(), side="right") - 1
    df["cycle_start"] = cs[np.clip(idx, 0, None)]
    df["coord_plan"] = plan
    return df


def _run(df, **kw):
    kw.setdefault("left", "EBL")
    return left_turn_gaps(df, _PH, [_A, _B], **kw)


def _row(cy, green):
    return cy.loc[cy["green_ts"] == _T0 + green].iloc[0]


# One 26 s green → window [0, 30).  Offs give the gaps
# 1.0 (≤1: short) · 1.5 (b1) · 3.3 (b1, upper edge) · 0.0 (short, two lanes
# at once) · 7.4 (b3, upper edge) · 3.7 (b2, upper edge) · 13.1 (b4).
_OFFS = [1.0, 2.5, 5.8, (5.8, _B), 13.2, 16.9]


# ---------------------------------------------------------------------------
# Gaps and bins
# ---------------------------------------------------------------------------


def test_golden_window_gaps_and_bins():
    cy, gp = _run(_events([(0, 26)], _OFFS))
    r = _row(cy, 0)
    assert not r["censored"]
    assert r["window_s"] == pytest.approx(30.0)
    assert r["red_clear_ts"] == pytest.approx(_T0 + 30)
    assert r["n_actuations"] == 6 and r["n_gaps"] == 7
    assert (r["n_short"], r["bin_1"], r["bin_2"], r["bin_3"], r["bin_4"]) == (2, 2, 1, 1, 1)
    assert list(gp["gap_s"]) == pytest.approx([1.0, 1.5, 3.3, 0.0, 7.4, 3.7, 13.1])
    assert list(gp["gap_bin"]) == [0, 1, 1, 0, 3, 2, 4]
    # Gaps tile the window exactly.
    assert gp["gap_s"].sum() == pytest.approx(30.0)
    assert gp["start_ts"].iloc[0] == pytest.approx(_T0)
    assert gp["end_ts"].iloc[-1] == pytest.approx(_T0 + 30)
    assert (gp["green_ts"] == _T0).all()


def test_golden_turnable_and_critical_sums():
    cy, _ = _run(_events([(0, 26)], _OFFS))
    r = _row(cy, 0)
    # ≥ 7.4: 7.4 + 13.1 (inclusive, unlike the bins' lower edge).
    assert r["turnable_s"] == pytest.approx(20.5)
    assert r["pct_turnable"] == pytest.approx(100 * 20.5 / 30, abs=1e-3)
    assert r["sum_ge_critical"] == pytest.approx(20.5) and r["n_ge_critical"] == 2
    cy, _ = _run(_events([(0, 26)], _OFFS), critical_s=3.7)
    r = _row(cy, 0)
    assert r["sum_ge_critical"] == pytest.approx(24.2) and r["n_ge_critical"] == 3
    cy, _ = _run(_events([(0, 26)], _OFFS), trend_s=13.2)
    assert _row(cy, 0)["turnable_s"] == 0.0


def test_offs_at_the_window_ends_and_outside_are_ignored():
    # Off exactly at green, exactly at red clearance, during red, before
    # the green; an on-event alone; a detector not in the set.
    extra = [(0.0, 81, _A), (30.0, 81, _A), (33.0, 81, _A), (-5.0, 81, _B),
             (10.0, 82, _A), (10.0, 81, _C)]
    cy, gp = _run(_events([(0, 26)], extra=extra))
    r = _row(cy, 0)
    assert r["n_actuations"] == 0 and r["n_gaps"] == 1 and r["bin_4"] == 1
    assert list(gp["gap_s"]) == pytest.approx([30.0])
    assert r["pct_turnable"] == pytest.approx(100.0)


def test_no_actuation_green_is_one_gap():
    cy, gp = _run(_events([(0, 3)]))
    r = _row(cy, 0)
    assert r["window_s"] == pytest.approx(7.0) and r["n_gaps"] == 1
    assert r["bin_1"] == 0 and r["bin_3"] == 1
    assert len(gp) == 1 and gp["gap_bin"].iloc[0] == 3


def test_each_off_belongs_to_its_own_green():
    # Greens [0, 30) and [60, 74); the off at 45 is in red.
    cy, gp = _run(_events([(0, 26), (60, 10)], [5.0, 45.0, 62.0, 70.0]))
    assert list(cy["n_actuations"]) == [1, 2]
    assert list(gp.loc[gp["green_ts"] == _T0 + 60, "gap_s"]) == pytest.approx([2.0, 8.0, 4.0])
    assert len(gp) == 5


def test_custom_edges_and_gap_above_finite_last_edge():
    edges = (0.5, 2.0, 5.0)
    cy, gp = _run(_events([(0, 26)], _OFFS), edges=edges)
    r = _row(cy, 0)
    assert list(cy.columns) == cycle_schema(edges)
    # 1.0 1.5 (b1) · 3.3 3.7 (b2) · 0.0 short · 7.4 13.1 above 5.0
    assert (r["n_short"], r["bin_1"], r["bin_2"]) == (1, 2, 2)
    assert r["n_gaps"] == 7
    assert sorted(gp.loc[gp["gap_bin"] == 3, "gap_s"]) == pytest.approx([7.4, 13.1])


@pytest.mark.parametrize("edges", [(1.0,), (3.0, 1.0), (-1.0, 2.0), (1.0, np.inf, 9.0), (1.0, 1.0, 2.0)])
def test_bad_edges_raise(edges):
    with pytest.raises(ValueError):
        _run(_events([(0, 26)]), edges=edges)


def test_bin_labels_and_columns():
    assert bin_labels() == ["1-3.3s", "3.3-3.7s", "3.7-7.4s", "7.4s+"]
    assert bin_columns() == ["bin_1", "bin_2", "bin_3", "bin_4"]
    assert bin_labels((0.5, 2.0, 5.0)) == ["0.5-2s", "2-5s"]


# ---------------------------------------------------------------------------
# Gap marker rule
# ---------------------------------------------------------------------------


def test_gap_marker_in_window_censors_the_green():
    cy, gp = _run(_events([(0, 26), (60, 10)], _OFFS + [65.0], gaps=[12.0]))
    r = _row(cy, 0)
    assert r["censored"]
    for c in ["n_actuations", "n_gaps", "n_short", *bin_columns(), "n_ge_critical"]:
        assert pd.isna(r[c]), c
    for c in ["window_s", "turnable_s", "pct_turnable", "sum_ge_critical"]:
        assert np.isnan(r[c]), c
    assert (gp["green_ts"] == _T0 + 60).all() and len(gp) == 2
    assert not _row(cy, 60)["censored"]


def test_gap_marker_in_red_clearance_censors_the_green():
    # Marker at 31 s: after the window end (30) but before end of red
    # clearance (32); the shared interval builder drops the interval.
    cy, gp = _run(_events([(0, 26)], _OFFS, gaps=[31.0]))
    assert _row(cy, 0)["censored"] and gp.empty


def test_gap_marker_in_red_does_not_censor():
    cy, _ = _run(_events([(0, 26), (60, 10)], _OFFS, gaps=[45.0]))
    assert not _row(cy, 0)["censored"]


def test_unpaired_green_is_censored():
    df = _events([(0, 26)], _OFFS, extra=[(80.0, 1, _PH), (85.0, 82, _A), (86.0, 81, _A)])
    cy, gp = _run(df)
    r = _row(cy, 80)
    assert r["censored"] and pd.isna(r["n_gaps"]) and pd.isna(r["red_clear_ts"])
    assert (gp["green_ts"] == _T0).all()


# ---------------------------------------------------------------------------
# Exclusions, timestamps, schema
# ---------------------------------------------------------------------------


def test_exclusions_drop_offs_first():
    # Exclude det B while phase 6 is green: the two-lane 0.0 gap goes away.
    exc = [{"detector": _B, "phase": _PH, "status": "Green"}]
    cy, gp = _run(_events([(0, 26)], _OFFS), exclusions=exc)
    r = _row(cy, 0)
    assert r["n_actuations"] == 5 and r["n_short"] == 1


def test_tz_aware_timestamps_match_epoch():
    df = _events([(0, 26), (60, 10)], _OFFS + [65.0])
    cy_e, gp_e = _run(df)
    dft = df.copy()
    for c in ("timestamp", "cycle_start"):
        dft[c] = pd.to_datetime(dft[c], unit="s", utc=True).dt.tz_convert(_TZ)
    cy_t, gp_t = _run(dft)
    num = ["window_s", "n_gaps", "n_short", *bin_columns(), "turnable_s", "pct_turnable"]
    pd.testing.assert_frame_equal(cy_e[num], cy_t[num])
    assert list(gp_t["gap_s"]) == list(gp_e["gap_s"])
    assert str(cy_t["green_ts"].dt.tz) == _TZ and str(gp_t["start_ts"].dt.tz) == _TZ


def test_dtypes_and_labels():
    cy, gp = _run(_events([(0, 26)], _OFFS))
    assert list(cy.columns) == cycle_schema() and list(gp.columns) == GAP_SCHEMA
    for c in ["n_actuations", "n_gaps", "n_short", *bin_columns(), "n_ge_critical"]:
        assert str(cy[c].dtype) == "Int64", c
    assert cy["censored"].dtype == bool
    assert (cy["left"] == "EBL").all() and (cy["opposing_phase"] == _PH).all()
    assert gp["gap_bin"].dtype == np.int64 and gp["gap_s"].dtype == float


def test_no_events_or_no_green():
    cy, gp = _run(pd.DataFrame(columns=["timestamp", "event_code", "parameter",
                                        "cycle_start", "coord_plan"]))
    assert cy.empty and gp.empty and list(cy.columns) == cycle_schema()
    cy, gp = left_turn_gaps(_events([(0, 26)], _OFFS), 2, [_A])
    assert cy.empty and gp.empty


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_bins_by_green_time():
    # T0 is 0.3 s past a quarter hour: greens at 0 and 60 s share a 15-min
    # bin, 1000 s is in the next and 2000 s (censored) in the third.
    df = _events([(0, 26), (60, 10), (1000, 6), (2000, 6)],
                 _OFFS + [62.0, 70.0], gaps=[2002.0])
    cy, _ = _run(df)
    s = summarize_left_turn_gaps(cy, 15)
    assert list(s.columns) == summary_schema()
    assert list(s["n_cycles"]) == [2, 1, 0] and list(s["n_censored"]) == [0, 0, 1]
    first = s.iloc[0]
    assert first["green_s"] == pytest.approx(44.0)
    assert first["n_gaps"] == 10
    # UDOT's line is the mean of per-green percents; the time-weighted one
    # divides summed turnable time by summed green.
    c0, c1 = _row(cy, 0), _row(cy, 60)
    assert first["pct_turnable"] == pytest.approx((c0["pct_turnable"] + c1["pct_turnable"]) / 2, abs=1e-3)
    assert first["pct_turnable_time"] == pytest.approx(
        100 * (c0["turnable_s"] + c1["turnable_s"]) / 44.0, abs=1e-3)
    assert np.isnan(s.iloc[2]["pct_turnable"]) and np.isnan(s.iloc[2]["pct_turnable_time"])
    assert s.iloc[2]["green_s"] == 0.0


def test_summary_whole_window():
    cy, _ = _run(_events([(0, 26), (60, 10), (1000, 6)], _OFFS + [62.0, 70.0]))
    s = summarize_left_turn_gaps(cy, None)
    assert len(s) == 1 and s["n_cycles"].iloc[0] == 3
    assert s["time"].iloc[0] == pd.to_datetime(_T0, unit="s", utc=True)
    assert s["bin_4"].iloc[0] == int(cy["bin_4"].sum())
    assert summarize_left_turn_gaps(cy.iloc[:0], 15).empty


# ---------------------------------------------------------------------------
# Config: through phases and left-turn pairs
# ---------------------------------------------------------------------------

_CFG315 = {
    "TM_EBL": "25", "TM_EBT": "26,27,28", "TM_EBR": "29",
    "TM_WBL": "17", "TM_WBT": "18,19,20", "TM_WBR": "21",
    "TM_NBL": "22", "TM_NBT": "23", "TM_NBR": "24",
    "TM_SBL": "30", "TM_SBT": "31", "TM_SBR": "32",
    "TM_Exclusions": "[]",
    "Det_P1_Stop_Bar": "17", "Det_P2_Stop_Bar": "26,27,28", "Det_P3_Stop_Bar": "22",
    "Det_P4_Stop_Bar": "31", "Det_P5_Stop_Bar": "25", "Det_P6_Stop_Bar": "18,19,20",
    "Det_P7_Stop_Bar": "30", "Det_P8_Stop_Bar": "23",
    "Det_P2_Occupancy": "50,51,52", "Det_P6_Occupancy": "34,35,36",
}


def _thr(cfg):
    return through_phases(cfg).set_index("direction")


def test_through_phases_derived_from_tm_through_channels():
    t = _thr(_CFG315)
    assert list(through_phases(_CFG315).columns) == THROUGH_SCHEMA
    assert dict(t["phase"]) == {"NB": 8, "SB": 4, "EB": 2, "WB": 6}
    assert (t["source"] == "derived").all()


def test_explicit_direction_key_wins():
    cfg = {**_CFG315, "Det_P4_Direction": "wb"}
    t = _thr(cfg)
    assert t.at["WB", "phase"] == 4 and t.at["WB", "source"] == "config"
    assert t.at["WB", "candidates"] == "6"


def test_ambiguous_and_missing_derivations():
    # 201-like: the WB through channel sits in two phases' keys.
    cfg = {"TM_WBT": "41", "Det_P3_Pairs": "[[41,30]]", "Det_P4_Stop_Bar": "41",
           "TM_EBT": "63", "TM_NBT": "60"}
    t = _thr(cfg)
    assert pd.isna(t.at["WB", "phase"]) and t.at["WB", "source"] == "ambiguous"
    assert t.at["WB", "candidates"] == "3,4"
    assert pd.isna(t.at["EB", "phase"]) and t.at["EB", "source"] == "none"
    assert "SB" not in t.index
    # Right-turn channels are not evidence.
    assert _thr({"TM_WBT": "63", "TM_WBR": "64", "Det_P6_Pairs": "[[52,64]]"}).at["WB", "source"] == "none"


def test_direction_key_errors():
    with pytest.raises(ValueError):
        through_phases({**_CFG315, "Det_P2_Direction": "EB", "Det_P6_Direction": "EB"})
    with pytest.raises(ValueError):
        phase_directions({"Det_P2_Direction": "North"})
    assert phase_directions({"Det_P2_Direction": " eb ", "Det_P4_Direction": ""}) == {2: "EB"}


def test_left_turn_pairs_315():
    p = left_turn_pairs(_CFG315).set_index("left")
    assert list(left_turn_pairs(_CFG315).columns) == PAIR_SCHEMA
    assert list(p.index) == ["NBL", "SBL", "EBL", "WBL"]
    assert p.at["EBL", "opposing"] == "WB" and p.at["EBL", "opposing_phase"] == 6
    assert p.at["EBL", "detectors"] == [18, 19, 20, 21]
    assert p.at["EBL", "n_lanes"] == 3 and p.at["EBL", "critical_s"] == 5.3
    assert p.at["NBL", "detectors"] == [31, 32] and p.at["NBL", "critical_s"] == 4.1
    assert p.at["NBL", "opposing_phase"] == 4
    assert all(s == [] for s in p["shared"])


def test_left_turn_pairs_unresolved_and_shared_detector():
    # 701-like: channel 54 is in both TM_EBL and TM_WBR; no Det_P keys.
    cfg = {"TM_EBL": "54,55", "TM_EBT": "56", "TM_WBL": "62", "TM_WBT": "63",
           "TM_WBR": "54", "TM_NBL": "37"}
    p = left_turn_pairs(cfg).set_index("left")
    assert pd.isna(p.at["EBL", "opposing_phase"]) and p.at["EBL", "source"] == "none"
    assert p.at["EBL", "detectors"] == [54, 63] and p.at["EBL", "shared"] == [54]
    assert p.at["WBL", "shared"] == []
    # NBL: no SB movement at all.
    assert p.at["NBL", "detectors"] == [] and p.at["NBL", "n_lanes"] == 0
    assert p.at["NBL", "source"] == "none"
    with_key = left_turn_pairs({**cfg, "Det_P6_Direction": "WB"}).set_index("left")
    assert with_key.at["EBL", "opposing_phase"] == 6 and with_key.at["EBL", "source"] == "config"


def test_critical_gap_rule():
    assert [critical_gap(n) for n in (0, 1, 2, 3, 4)] == [4.1, 4.1, 4.1, 5.3, 5.3]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2] / "intersections"
_DB315 = _ROOT / "315_US-20-26_Franklin_Rd_and_KCID_Rd" / "315_data.db"


def test_corpus_315_monday():
    # 2025-12-15: all four lefts resolve by derivation.  The EB/WB arterial
    # (3 lanes, critical 5.3 s) logs ~6 offs per opposing green; the minor
    # NB/SB approaches ~1.  One censored green per left is the day's end.
    if not _DB315.exists():
        pytest.skip("corpus DB not present")
    from atspm.analysis.counts import parse_exclusions_from_config
    from atspm.data.manager import DatabaseManager
    from atspm.data.reader import get_events_with_cycles_df
    day = datetime(2025, 12, 15)
    with DatabaseManager(_DB315) as m:
        cfg = m.get_config_at_date(day)
    ev = get_events_with_cycles_df(db_path=_DB315, start=day, end=datetime(2025, 12, 16),
                                   event_codes=[-1, 1, 8, 9, 10, 11, 12, 81, 82],
                                   timezone=_TZ)
    pairs = left_turn_pairs(cfg)
    assert list(pairs["opposing_phase"]) == [4, 8, 6, 2]
    frames = []
    for r in pairs.itertuples():
        cy, gp = left_turn_gaps(ev, int(r.opposing_phase), r.detectors, left=r.left,
                                critical_s=r.critical_s,
                                exclusions=parse_exclusions_from_config(cfg))
        ok = cy.loc[~cy["censored"]]
        # Gaps tile every uncensored window.
        tiled = gp.groupby("green_ts")["gap_s"].sum()
        assert np.allclose(tiled.to_numpy(), ok.set_index("green_ts").loc[tiled.index, "window_s"],
                           atol=0.01)
        assert (ok["n_gaps"] == ok["n_short"] + ok[bin_columns()].sum(axis=1)).all()
        frames.append(cy)
    s = summarize_left_turn_gaps(pd.concat(frames), None).set_index("left")
    assert dict(s["n_cycles"]) == {"EBL": 952, "NBL": 844, "SBL": 847, "WBL": 918}
    assert dict(s["n_censored"]) == {"EBL": 1, "NBL": 1, "SBL": 1, "WBL": 1}
    assert dict(s["n_gaps"]) == {"EBL": 6737, "NBL": 1741, "SBL": 1541, "WBL": 6861}
    assert list(s.loc["EBL", bin_columns()]) == [1910, 156, 1287, 1700]
    assert 70 < s.at["EBL", "pct_turnable"] < 78 and 70 < s.at["WBL", "pct_turnable"] < 78
