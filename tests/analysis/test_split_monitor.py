"""Golden tests for the Split Monitor and programmed-plan timeline (Functional Core).

Target: src/atspm/analysis/split_monitor.py (UDOT roadmap S-M2).  Opus-written.

Contract summary
----------------
Codes 131 plan, 132 cycle, 133 offset, 134–149 split of phase 1–16 form a
change log: a full dump at local midnight, deltas in between.  The timeline
is a state register: each code holds its last value; same-timestamp events
apply together; identical consecutive states merge; 0 is a real value.
A comms-gap marker (-1, parameter -1) resets every value to NA; a clock-step
fence (-1, parameter -2) does not.  The last row ends at the data end.

The split monitor gives one row per phase service: split = green + yellow +
red clearance; termination 4/5/6 (highest wins) else 'unknown'; ped_walk =
a Code 21 of the phase in [green, clear end); programmed split at the green,
NaN when the cycle is 0/NA or the phase's split is 0/NA.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.split_monitor import (
    CYCLE_SCHEMA,
    STATS_SCHEMA,
    TIMELINE_SCHEMA,
    plan_timeline,
    programmed_at,
    split_monitor,
    split_monitor_stats,
)

T0 = 1_765_800_000.0   # an arbitrary UTC epoch


def _ev(rows):
    """rows: (seconds after T0, code, parameter) → flat events frame."""
    df = pd.DataFrame(rows, columns=["t", "event_code", "parameter"])
    df["timestamp"] = T0 + df["t"].astype(float)
    df["cycle_start"] = np.nan
    return df[["timestamp", "event_code", "parameter", "cycle_start"]] \
        .sort_values("timestamp", kind="stable").reset_index(drop=True)


def _dump(t, plan, cycle, offset, splits):
    """A full 131–149 dump at t; splits: {phase: seconds}, others 0."""
    rows = [(t, 131, plan), (t, 132, cycle), (t, 133, offset)]
    rows += [(t, 133 + p, splits.get(p, 0)) for p in range(1, 17)]
    return rows


PLAN1 = {1: 19, 2: 46, 4: 25, 5: 25, 6: 40, 8: 25}
PLAN11 = {1: 17, 2: 41, 3: 14, 4: 18, 5: 22, 6: 36, 7: 14, 8: 18}


def _na_list(s):
    return [None if pd.isna(v) else v for v in s]


def _row(tl, i):
    return {k: (None if pd.isna(v) else int(v)) for k, v in tl.iloc[i].drop(["start", "end"]).items()}


# ---------------------------------------------------------------------------
# plan_timeline
# ---------------------------------------------------------------------------

def test_schema_and_empty():
    assert list(plan_timeline(pd.DataFrame(columns=["timestamp", "event_code", "parameter"])).columns) == TIMELINE_SCHEMA
    no_plan = _ev([(0, 1, 2), (5, 8, 2)])
    out = plan_timeline(no_plan)
    assert out.empty and list(out.columns) == TIMELINE_SCHEMA


def test_delta_logging_inherits_unlogged_values():
    # Plan 11 logs its splits only; it shares cycle and offset with plan 1.
    rows = _dump(0, 1, 90, 60, PLAN1)
    rows += [(100, 131, 11)] + [(100, 133 + p, s) for p, s in PLAN11.items()]
    rows += [(200, 1, 2)]
    tl = plan_timeline(_ev(rows))
    assert len(tl) == 2
    r = _row(tl, 1)
    assert (r["plan"], r["cycle"], r["offset"]) == (11, 90, 60)
    assert r["split_3"] == 14 and r["split_7"] == 14 and r["split_2"] == 41
    assert r["split_9"] == 0
    assert tl["start"].tolist() == [T0, T0 + 100]
    assert tl["end"].tolist() == [T0 + 100, T0 + 200]


def test_same_timestamp_is_one_update_and_last_value_wins():
    rows = _dump(0, 1, 90, 60, PLAN1) + [(0, 132, 100)]   # 132 logged twice
    tl = plan_timeline(_ev(rows))
    assert len(tl) == 1
    assert _row(tl, 0)["cycle"] == 100


def test_identical_relog_merges():
    rows = _dump(0, 1, 90, 60, PLAN1) + _dump(50, 1, 90, 60, PLAN1) + [(80, 133 + 2, 46)]
    tl = plan_timeline(_ev(rows))
    assert len(tl) == 1
    assert tl["start"].iloc[0] == T0


def test_parameters_lagging_the_plan_code_give_a_transient_row():
    # Free, then plan 1 selected at 100; its parameters land at 109.9.
    rows = _dump(0, 100, 0, 0, {})
    rows += [(100, 131, 1)] + [(109.9, 132, 90), (109.9, 133, 60)]
    rows += [(109.9, 133 + p, s) for p, s in PLAN1.items()]
    tl = plan_timeline(_ev(rows))
    assert [(r["plan"], r["cycle"]) for r in (_row(tl, i) for i in range(len(tl)))] == \
        [(100, 0), (1, 0), (1, 90)]
    assert tl["start"].iloc[2] == pytest.approx(T0 + 109.9)


def test_zero_split_and_zero_offset_are_values_not_missing():
    rows = _dump(0, 2, 60, 0, {2: 10, 6: 10})
    tl = plan_timeline(_ev(rows))
    r = _row(tl, 0)
    assert r["offset"] == 0 and r["split_2"] == 10 and r["split_3"] == 0


def test_preemption_zeroing_is_its_own_row():
    rows = _dump(0, 1, 90, 60, PLAN1)
    rows += [(100, c, 0) for c in (132, 133, 134, 135, 137, 138, 139, 141)]
    rows += [(140, 132, 90), (140, 133, 60)] + [(140, 133 + p, s) for p, s in PLAN1.items()]
    tl = plan_timeline(_ev(rows))
    assert [(r["plan"], r["cycle"]) for r in (_row(tl, i) for i in range(3))] == \
        [(1, 90), (1, 0), (1, 90)]


def test_comms_gap_resets_every_value():
    rows = _dump(0, 1, 90, 60, PLAN1) + [(100, -1, -1), (150, 131, 11), (300, 1, 2)]
    tl = plan_timeline(_ev(rows))
    assert len(tl) == 3
    assert all(v is None for v in _row(tl, 1).values())            # the gap row
    assert tl["start"].iloc[1] == T0 + 100
    r = _row(tl, 2)
    assert r["plan"] == 11 and r["cycle"] is None and r["split_2"] is None
    assert tl["end"].iloc[0] == T0 + 100


def test_clock_step_fence_does_not_reset():
    rows = _dump(0, 1, 90, 60, PLAN1) + [(100, -1, -2), (300, 1, 2)]
    tl = plan_timeline(_ev(rows))
    assert len(tl) == 1
    assert _row(tl, 0)["cycle"] == 90
    assert tl["end"].iloc[0] == T0 + 300


def test_gap_marker_sharing_a_timestamp_applies_before_the_plan_code():
    rows = _dump(0, 1, 90, 60, PLAN1) + [(100, 131, 11), (100, -1, -1)]
    tl = plan_timeline(_ev(rows))
    r = _row(tl, len(tl) - 1)
    assert r["plan"] == 11 and r["cycle"] is None


def test_last_row_ends_at_data_end_from_any_code():
    tl = plan_timeline(_ev(_dump(0, 1, 90, 60, PLAN1) + [(500, 82, 3)]))
    assert tl["end"].iloc[-1] == T0 + 500


def test_tz_aware_input_keeps_dtype_and_values():
    ev = _ev(_dump(0, 1, 90, 60, PLAN1) + [(100, 131, 11), (200, 1, 2)])
    ev_tz = ev.copy()
    ev_tz["timestamp"] = pd.to_datetime(ev["timestamp"], unit="s", utc=True).dt.tz_convert("US/Mountain")
    a, b = plan_timeline(ev), plan_timeline(ev_tz)
    assert str(b["start"].dtype).startswith("datetime64") and b["start"].dt.tz is not None
    pd.testing.assert_frame_equal(a.drop(columns=["start", "end"]), b.drop(columns=["start", "end"]))
    assert (b["start"] == ev_tz["timestamp"].iloc[[0, 19]].reset_index(drop=True)).all()


def test_programmed_at_boundaries():
    tl = plan_timeline(_ev(_dump(10, 1, 90, 60, PLAN1) + [(100, 131, 11), (200, 1, 2)]))
    ts = pd.Series([T0 + 5, T0 + 10, T0 + 99.9, T0 + 100, T0 + 200, T0 + 200.1])
    got = programmed_at(tl, ts, phase=[2, 2, 2, 2, 2, 2])
    assert _na_list(got["plan"]) == [None, 1, 1, 11, 11, None]
    assert _na_list(got["split"]) == [None, 46, 46, 46, 46, None]
    assert programmed_at(tl, ts.iloc[:0]).empty


# ---------------------------------------------------------------------------
# split_monitor
# ---------------------------------------------------------------------------

def _service(g, phase, green, yellow=4.0, red=2.0, term=None, walk_at=None):
    """Phase service starting at g: 1 → 8 (+green) → 9/10 (+yellow) → 11 (+red)."""
    rows = [(g, 1, phase), (g + green, 8, phase), (g + green + yellow, 9, phase)]
    if red:
        rows += [(g + green + yellow, 10, phase), (g + green + yellow + red, 11, phase)]
    else:
        rows.append((g + green + yellow, 12, phase))
    if term is not None:
        rows.append((g + green, term, phase))
    if walk_at is not None:
        rows.append((g + walk_at, 21, phase))
    return rows


def test_split_is_green_plus_clearance_with_termination():
    rows = _dump(0, 1, 90, 60, PLAN1)
    rows += _service(10, 2, 30, term=4)
    rows += _service(100, 2, 40, term=5)
    rows += _service(190, 2, 44, term=6)
    rows += _service(280, 2, 20)                              # nothing logged
    rows += _service(370, 2, 20, term=4) + [(390, 6, 2)]      # 4 and 6: 6 wins
    out = split_monitor(_ev(rows))
    assert list(out.columns) == CYCLE_SCHEMA
    assert out["split_dur"].tolist() == [36.0, 46.0, 50.0, 26.0, 26.0]
    assert out["green_dur"].tolist() == [30.0, 40.0, 44.0, 20.0, 20.0]
    assert out["clear_dur"].tolist() == [6.0] * 5
    assert out["termination"].tolist() == ["gap_out", "max_out", "force_off", "unknown", "force_off"]


def test_yellow_only_phase_split():
    out = split_monitor(_ev(_dump(0, 1, 90, 60, PLAN1) + _service(10, 4, 15, red=0, term=4)))
    assert out["split_dur"].tolist() == [19.0]


def test_ped_walk_flag():
    rows = _dump(0, 1, 90, 60, PLAN1)
    rows += _service(10, 4, 20, walk_at=0.0)            # walk at green onset
    rows += _service(100, 4, 20, walk_at=25.9)          # inside red clearance
    rows += _service(200, 4, 20, walk_at=26.0)          # at clear end: outside
    rows += _service(300, 4, 20) + [(305, 21, 8)]       # another phase's walk
    out = split_monitor(_ev(rows))
    assert out["ped_walk"].tolist() == [True, True, False, False]


def test_programmed_split_lookup_and_nan_rules():
    rows = _dump(0, 1, 90, 60, PLAN1)
    rows += _service(10, 2, 40, term=6)                      # plan 1, P2 = 46
    rows += _service(10, 3, 10, term=4)                      # plan 1, P3 = 0 → NaN
    rows += [(100, 131, 11)] + [(100, 133 + p, s) for p, s in PLAN11.items()]
    rows += _service(110, 3, 10, term=4)                     # plan 11, P3 = 14
    rows += [(200, 132, 0)]                                  # preempt zeroing (cycle only)
    rows += _service(210, 2, 30, term=4)                     # cycle 0 → NaN
    rows += [(300, -1, -1)] + _service(310, 2, 30, term=4)   # after a gap → unknown
    out = split_monitor(_ev(rows)).sort_values("green_ts").reset_index(drop=True)
    assert out["phase"].tolist() == [2, 3, 3, 2, 2]
    assert _na_list(out["plan"]) == [1, 1, 11, 11, None]
    assert _na_list(out["programmed_cycle"]) == [90, 90, 90, 0, None]
    np.testing.assert_array_equal(out["programmed_split"].to_numpy(), [46.0, np.nan, 14.0, np.nan, np.nan])
    np.testing.assert_array_equal(out["split_minus_programmed"].to_numpy(), [0.0, np.nan, 2.0, np.nan, np.nan])


def test_service_spanning_a_gap_is_dropped_but_fence_also_drops_it():
    rows = _dump(0, 1, 90, 60, PLAN1)
    rows += _service(10, 2, 30) + [(20, -1, -1)]
    rows += _service(100, 2, 30) + [(110, -1, -2)]
    rows += _service(200, 2, 30)
    out = split_monitor(_ev(rows))
    assert out["green_ts"].tolist() == [T0 + 200]
    assert _na_list(out["plan"]) == [None]          # the -1 reset the register


def test_phase_filter_and_explicit_timeline():
    rows = _dump(0, 1, 90, 60, PLAN1) + _service(10, 2, 30) + _service(10, 6, 30)
    ev = _ev(rows)
    a = split_monitor(ev, phases=[6])
    assert a["phase"].tolist() == [6]
    b = split_monitor(ev.loc[ev["event_code"] < 100], phases=[6], timeline=plan_timeline(ev))
    pd.testing.assert_frame_equal(a, b)


def test_empty_inputs():
    assert list(split_monitor(pd.DataFrame(columns=["timestamp", "event_code", "parameter"])).columns) == CYCLE_SCHEMA
    out = split_monitor(_ev(_dump(0, 1, 90, 60, PLAN1)))
    assert out.empty and list(out.columns) == CYCLE_SCHEMA


# ---------------------------------------------------------------------------
# split_monitor_stats
# ---------------------------------------------------------------------------

def test_stats_per_plan_version():
    rows = _dump(0, 1, 90, 60, PLAN1)
    splits = [40, 42, 44, 46, 50]
    for i, (s, term) in enumerate(zip(splits, [4, 4, 6, 6, 5])):
        rows += _service(10 + 90 * i, 2, s - 6, term=term, walk_at=1 if i == 0 else None)
    rows += [(500, 135, 50)]                                   # P2 split edited
    rows += _service(510, 2, 44, term=6)
    rows += [(600, -1, -1)] + _service(610, 2, 30)             # unknown plan
    cy = split_monitor(_ev(rows))
    st = split_monitor_stats(cy)
    assert list(st.columns) == STATS_SCHEMA
    assert st["programmed_split"].tolist()[:2] == [46.0, 50.0]
    assert pd.isna(st["plan"].iloc[2]) and np.isnan(st["programmed_split"].iloc[2])
    assert st["n_cycles"].tolist() == [5, 1, 1]
    r = st.iloc[0]
    assert (r.gap_out_pct, r.max_out_pct, r.force_off_pct, r.unknown_pct) == (0.4, 0.2, 0.4, 0.0)
    assert r.ped_walk_pct == 0.2
    assert r.split_mean == pytest.approx(np.mean(splits))
    assert r.split_p50 == pytest.approx(np.percentile(splits, 50))
    assert r.split_p85 == pytest.approx(np.percentile(splits, 85))
    assert r.first_green == T0 + 10 and r.last_green == T0 + 370
    assert st["unknown_pct"].iloc[2] == 1.0
    assert str(st["plan"].dtype) == "Int64"


def test_stats_custom_percentiles_and_empty():
    rows = _dump(0, 1, 90, 60, PLAN1) + _service(10, 2, 30) + _service(100, 2, 40)
    st = split_monitor_stats(split_monitor(_ev(rows)), percentiles=(50, 95))
    assert "split_p95" in st.columns and "split_p85" not in st.columns
    assert st["split_p95"].iloc[0] == pytest.approx(np.percentile([36, 46], 95))
    assert list(split_monitor_stats(split_monitor(_ev([]))).columns) == STATS_SCHEMA


# ---------------------------------------------------------------------------
# Corpus: 315, Monday 2025-12-15 (plans 100 → 1 → 11 → 1 → 100)
# ---------------------------------------------------------------------------

_DB = Path(__file__).resolve().parents[2] / "intersections" / \
    "315_US-20-26_Franklin_Rd_and_KCID_Rd" / "315_data.db"
_CODES = [-1, 1, 4, 5, 6, 8, 9, 10, 11, 12, 21] + list(range(131, 150))


@pytest.fixture(scope="module")
def ev315():
    if not _DB.exists():
        pytest.skip("corpus DB not present")
    from atspm.data.reader import get_events_with_cycles_df
    return get_events_with_cycles_df(_DB, datetime(2025, 12, 14), datetime(2025, 12, 16),
                                     event_codes=_CODES, timezone="US/Mountain")


def test_corpus_315_timeline(ev315):
    tl = plan_timeline(ev315)
    tz = "US/Mountain"
    at = programmed_at(tl, pd.Series(pd.to_datetime(
        ["2025-12-15 03:00:00", "2025-12-15 08:00:00", "2025-12-15 12:00:00", "2025-12-15 14:45:30",
         "2025-12-15 22:00:00"]).tz_localize(tz)), phase=[2, 3, 3, 2, 2])
    assert at["plan"].tolist() == [100, 11, 1, 1, 100]
    assert at["cycle"].tolist() == [0, 90, 90, 0, 0]          # 14:45:30 is a preempt
    # The midnight dump omits 133 at 315, so the offset is unknown until logged.
    assert _na_list(at["offset"]) == [None, 60, 60, 0, 0]
    assert at["split"].tolist() == [0, 14, 0, 0, 0]
    # Plan 11 logs only its splits; cycle/offset come from plan 1.
    p11 = tl.loc[tl["plan"] == 11].iloc[0]
    assert [int(p11[f"split_{p}"]) for p in range(1, 9)] == [17, 41, 14, 18, 22, 36, 14, 18]


def test_corpus_315_split_monitor(ev315):
    cy = split_monitor(ev315)
    day = cy.loc[cy["green_ts"] >= pd.Timestamp("2025-12-15", tz="US/Mountain")]
    p2 = day.loc[(day["phase"] == 2) & (day["plan"] == 1) & day["programmed_split"].notna()]
    assert len(p2) > 300
    assert (p2["programmed_split"] == 46).all()
    assert (p2["termination"] == "force_off").mean() > 0.99
    # The coordinated phase absorbs unused time, so it runs at or over its split.
    assert p2["split_dur"].median() >= 46
    # Phases 3 and 7 have split 0 in plan 1: no programmed split there.
    assert day.loc[day["phase"].isin([3, 7]) & (day["plan"] == 1), "programmed_split"].isna().all()
    # Under free (cycle 0) nothing has a programmed split.
    assert day.loc[day["programmed_cycle"] == 0, "programmed_split"].isna().all()
    st = split_monitor_stats(day)
    row = st.loc[(st["phase"] == 6) & (st["plan"] == 11)].iloc[0]
    assert row["programmed_split"] == 36 and row["n_cycles"] >= 35
