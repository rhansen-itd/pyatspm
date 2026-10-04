"""Golden tests for Arrivals on Red and Approach Delay (Functional Core).

Target: src/atspm/analysis/approach_delay.py (UDOT roadmap S-M3).  Opus-written.

Contract summary
----------------
Rows are red-to-red cycles named by their serving green:
    red    = [previous end of yellow, green)   delay = green − t
    green  = [green, yellow)                    delay 0
    yellow = [yellow, this end of yellow)       delay 0
End of yellow = Code 9, else Code 10; red clearance counts as red.
Censored (flagged, counts NA): first green of a segment, a gap marker since
the previous green, or a Code 1 of the phase inside (red_start, green).
Arrivals whose travel-time shift crosses a gap marker are dropped.
delay_per_veh = total delay / all arrivals.  Binned: sum, then divide.

Timeline used throughout (seconds after each green)::

    0 green (1) · 20 yellow (8) · 24 end yellow (9) + red clearance (10)
    · 26 end red clearance (11) · 60 next green

so a cycle served by the green at G spans [G − 36, G + 24).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.aog import arrival_on_green
from atspm.analysis.approach_delay import (
    BIN_SCHEMA,
    CYCLE_SCHEMA,
    approach_delay,
    bin_approach_delay,
)
from atspm.analysis.detector_roles import arrival_travel_times, parse_detector_roles

_PH = 2
_A, _B = 33, 34
_T0 = 1_000_000.0


def _green(g):
    return [(g, 1), (g + 20, 8), (g + 24, 9), (g + 24, 10), (g + 26, 11)]


def _events(greens, arrivals=(), gaps=(), extra=(), plan=1.0):
    """Flat events; *arrivals* are ``t`` (detector A) or ``(t, det)``."""
    rows = [(_T0 + t, c, _PH) for g in greens for t, c in _green(g)]
    for a in arrivals:
        t, d = (a, _A) if np.isscalar(a) else a
        rows += [(_T0 + t, 82, d), (_T0 + t + 0.5, 81, d)]
    rows += [(_T0 + t, -1, -1) for t in gaps]
    rows += [(_T0 + t, c, _PH) for t, c in extra]
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cs = np.asarray([_T0 + g for g in greens])
    idx = np.searchsorted(cs, df["timestamp"].to_numpy(), side="right") - 1
    df["cycle_start"] = cs[np.clip(idx, 0, None)]
    df["coord_plan"] = plan
    return df


def _row(out, green):
    return out.loc[out["green_ts"] == _T0 + green].iloc[0]


# ---------------------------------------------------------------------------
# Classification and delay
# ---------------------------------------------------------------------------

def test_schema_and_one_row_per_green():
    out = approach_delay(_events([0, 60, 120], [30]), _PH, [_A])
    assert list(out.columns) == CYCLE_SCHEMA
    assert out["green_ts"].tolist() == [_T0, _T0 + 60, _T0 + 120]
    assert out["censored"].tolist() == [True, False, False]   # first green has no red start


def test_red_green_yellow_and_delay():
    # Cycle of green 60 spans [24, 84): red clearance 25, red 30 and 50,
    # green 65, yellow 81.  Cycle of green 120: red arrival at 100.
    out = approach_delay(_events([0, 60, 120], [25, 30, 50, 65, 81, 100]), _PH, [_A])
    r = _row(out, 60)
    assert (r.arrivals, r.arrivals_green, r.arrivals_yellow, r.arrivals_red) == (5, 1, 1, 3)
    assert r.total_delay_s == pytest.approx(35 + 30 + 10)
    assert r.delay_per_veh == pytest.approx(75 / 5)          # over all arrivals
    assert (r.aog_pct, r.aoy_pct, r.aor_pct) == pytest.approx((0.2, 0.2, 0.6))
    assert (r.cycle_len, r.red_dur, r.green_dur, r.yellow_dur) == (60.0, 36.0, 20.0, 4.0)
    r = _row(out, 120)
    assert (r.arrivals, r.arrivals_red, r.total_delay_s) == (1, 1, 20.0)


def test_half_open_boundaries():
    # Exactly at green → green; exactly at yellow → yellow; exactly at end of
    # yellow (84) → red of the next cycle, delay 120 − 84.
    out = approach_delay(_events([0, 60, 120], [60, 80, 84]), _PH, [_A])
    r = _row(out, 60)
    assert (r.arrivals_green, r.arrivals_yellow, r.arrivals_red) == (1, 1, 0)
    r = _row(out, 120)
    assert (r.arrivals_red, r.total_delay_s) == (1, 36.0)


def test_end_of_yellow_falls_back_to_code_10():
    # No Code 9 logged: red starts at Code 10.
    ev = _events([0, 60, 120], [83])
    ev = ev.loc[ev["event_code"] != 9].reset_index(drop=True)
    r = _row(approach_delay(ev, _PH, [_A]), 60)
    assert r.arrivals_yellow == 1 and r.next_red_start == _T0 + 84


def test_censored_cycle_counts_are_na():
    out = approach_delay(_events([0, 60], [10]), _PH, [_A])
    r = _row(out, 0)
    assert r.censored and pd.isna(r.arrivals) and np.isnan(r.total_delay_s)
    assert np.isnan(r.aor_pct) and np.isnan(r.delay_per_veh)


def test_no_arrivals_gives_zero_counts_and_nan_rates():
    out = approach_delay(_events([0, 60, 120], [(1000, _B)]), _PH, [_A])
    r = _row(out, 60)
    assert r.arrivals == 0 and r.total_delay_s == 0.0
    assert np.isnan(r.aor_pct) and np.isnan(r.delay_per_veh)


def test_empty_inputs_return_schema():
    assert list(approach_delay(_events([0, 60]), _PH, []).columns) == CYCLE_SCHEMA
    assert approach_delay(_events([0, 60]).iloc[0:0], _PH, [_A]).empty
    assert list(bin_approach_delay(pd.DataFrame()).columns) == BIN_SCHEMA


# ---------------------------------------------------------------------------
# Gap-marker rule
# ---------------------------------------------------------------------------

def test_gap_between_greens_censors_the_cycle_and_its_arrivals():
    out = approach_delay(_events([0, 60, 120], [30, 50, 100], gaps=[40]), _PH, [_A])
    assert _row(out, 60).censored                       # red started before the gap
    assert pd.isna(_row(out, 60).arrivals)
    r = _row(out, 120)
    assert not r.censored and (r.arrivals_red, r.total_delay_s) == (1, 20.0)


def test_lost_green_end_censors_the_next_cycle():
    # A Code 1 at 60 with no clearance after it: that green was served but its
    # end was lost.  The red from 24 did not run unbroken to the green at 120.
    out = approach_delay(_events([0, 120, 180], [30, 100, 150], extra=[(60, 1)]),
                         _PH, [_A])
    assert out["green_ts"].tolist() == [_T0, _T0 + 120, _T0 + 180]
    assert _row(out, 120).censored
    r = _row(out, 180)
    assert not r.censored and (r.arrivals_red, r.total_delay_s) == (1, 30.0)


def test_shift_across_a_gap_drops_the_arrival():
    # Gap at 50; green 60 is first in its segment.  A detector event at 48
    # with 40 s travel lands at 88, inside the valid cycle of green 120, but
    # it was logged before the gap: dropped.  One at 52 (→ 92) is kept.
    ev = _events([60, 120], [48, 52], gaps=[50])
    out = approach_delay(ev, _PH, [_A], travel_time_sec=40.0)
    r = _row(out, 120)
    assert (r.arrivals, r.arrivals_red, r.total_delay_s) == (1, 1, 28.0)


# ---------------------------------------------------------------------------
# Travel time
# ---------------------------------------------------------------------------

def test_scalar_travel_time_moves_red_arrival_onto_green():
    ev = _events([0, 60, 120], [57])
    assert _row(approach_delay(ev, _PH, [_A]), 60).arrivals_red == 1
    r = _row(approach_delay(ev, _PH, [_A], travel_time_sec=5.0), 60)
    assert (r.arrivals_green, r.arrivals_red, r.total_delay_s) == (1, 0, 0.0)


def test_per_detector_travel_times():
    ev = _events([0, 60, 120], [(40, _A), (40, _B)])
    out = approach_delay(ev, _PH, [_A, _B], travel_time_sec={_A: 5.0, _B: 25.0})
    r = _row(out, 60)
    # A reaches the stop line at 45 (red, delay 15); B at 65 (green).
    assert (r.arrivals_red, r.arrivals_green, r.total_delay_s) == (1, 1, 15.0)


def test_travel_mapping_missing_a_detector_raises():
    with pytest.raises(ValueError, match="34"):
        approach_delay(_events([0, 60]), _PH, [_A, _B], travel_time_sec={_A: 5.0})


def test_aog_accepts_a_per_detector_offset_mapping():
    ev = _events([0, 60, 120], [(57, _A), (57, _B)])
    out = arrival_on_green(ev, _PH, [_A, _B], arrival_offset_sec={_A: 5.0})
    assert int(out["aog"].sum()) == 1


# ---------------------------------------------------------------------------
# Timestamp inputs and the AoG cross-check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit", ["s", "ms", "ns"])
def test_tz_aware_timestamps_any_resolution_match_floats(unit):
    ev = _events([0, 60, 120, 180], [25, 30, 65, 81, 100, 130, 170], gaps=[140])
    tz = ev.copy()
    for col in ("timestamp", "cycle_start"):
        tz[col] = (pd.to_datetime(tz[col], unit="s", utc=True)
                   .astype(f"datetime64[{unit}, UTC]").dt.tz_convert("US/Mountain"))
    a = approach_delay(ev, _PH, [_A], travel_time_sec=2.0)
    b = approach_delay(tz, _PH, [_A], travel_time_sec=2.0)
    cols = [c for c in CYCLE_SCHEMA if c not in
            ("cycle_start", "red_start", "green_ts", "yellow_ts", "next_red_start")]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_green_arrivals_match_aog_on_uncensored_cycles():
    rng = np.random.default_rng(7)
    greens = list(range(0, 1200, 60)) + [1230]          # one phase served twice
    arr = np.round(rng.uniform(0, 1300, 400), 1)
    ev = _events(greens, arr, gaps=[600.5])
    ad = approach_delay(ev, _PH, [_A], travel_time_sec=3.0)
    aog = arrival_on_green(ev, _PH, [_A], arrival_offset_sec=3.0)
    ok = ~ad["censored"]
    m = ad.loc[ok, ["green_ts", "arrivals_green"]].merge(
        aog.reset_index()[["cycle_start", "aog"]].assign(
            green_ts=_build_greens(ev)), on="green_ts")
    assert len(m) == ok.sum()
    assert (m["arrivals_green"].astype(int) == m["aog"].astype(int)).all()


def _build_greens(ev):
    from atspm.analysis.aog import _build_green_windows
    return _build_green_windows(ev, _PH).sort_values("green_ts")["green_ts"].to_numpy()


def test_cycles_tile_time_so_each_arrival_counts_once():
    rng = np.random.default_rng(11)
    arr = np.round(rng.uniform(25, 1164, 300), 1)       # inside cycles 2..last
    ad = approach_delay(_events(range(0, 1200, 60), arr), _PH, [_A])
    assert int(ad["arrivals"].sum()) == 300


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def test_bin_sums_then_divides():
    ev = _events([0, 60, 120, 180], [25, 30, 65, 100, 130, 170])
    cyc = approach_delay(ev, _PH, [_A])
    b = bin_approach_delay(cyc, bin_len=15)
    assert list(b.columns) == BIN_SCHEMA
    assert len(b) == 1
    r = b.iloc[0]
    assert (r.n_cycles, r.n_censored) == (3, 1)
    ok = cyc.loc[~cyc["censored"]]
    assert r.arrivals == int(ok["arrivals"].sum())
    assert r.total_delay_s == pytest.approx(ok["total_delay_s"].sum())
    assert r.delay_per_veh == pytest.approx(r.total_delay_s / r.arrivals, abs=0.01)
    assert r.aor_pct == pytest.approx(ok["arrivals_red"].sum() / r.arrivals, abs=1e-4)
    assert r.total_delay_vh == pytest.approx(r.total_delay_s / 3600, abs=1e-4)
    assert r.delay_vh_per_hr == pytest.approx(r.total_delay_vh * 4, abs=1e-4)


def test_bin_keeps_a_censored_only_bin():
    # Green at 0 is censored and alone in its 15-min bin; the rest fall later.
    ev = _events([0, 960, 1020], [990])
    b = bin_approach_delay(approach_delay(ev, _PH, [_A]), bin_len=15)
    first = b.iloc[0]
    assert (first.n_cycles, first.n_censored, first.arrivals) == (0, 1, 0)
    assert np.isnan(first.delay_per_veh)


def test_bin_splits_by_plan():
    a = approach_delay(_events([0, 60, 120], [30, 100], plan=1.0), _PH, [_A])
    b = approach_delay(_events([0, 60, 120], [30], plan=2.0), _PH, [_A])
    out = bin_approach_delay(pd.concat([a, b]), bin_len=60)
    assert sorted(out["coord_plan"].tolist()) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Config: Det_P{N}_Arrival_Travel
# ---------------------------------------------------------------------------

def test_travel_key_scalar_applies_to_every_arrival_detector():
    cfg = {"Det_P2_Arrival": "54,55,56", "Det_P2_Arrival_Travel": "5.4"}
    assert arrival_travel_times(cfg) == {2: {54: 5.4, 55: 5.4, 56: 5.4}}


def test_travel_key_list_aligns_with_arrival_order():
    cfg = {"Det_P6_Arrival": "40,38", "Det_P6_Arrival_Travel": "5.0, 6.0"}
    assert arrival_travel_times(cfg) == {6: {40: 5.0, 38: 6.0}}


@pytest.mark.parametrize("cfg", [
    {"Det_P2_Arrival": "54,55,56", "Det_P2_Arrival_Travel": "5,6"},
    {"Det_P2_Arrival_Travel": "5"},
    {"Det_P2_Arrival": "54", "Det_P2_Arrival_Travel": "fast"},
    {"Det_P2_Arrival": "54", "Det_P2_Arrival_Travel": "-1"},
])
def test_travel_key_errors(cfg):
    with pytest.raises(ValueError):
        arrival_travel_times(cfg)


def test_travel_key_blank_or_absent_and_not_a_role():
    cfg = {"Det_P2_Arrival": "54", "Det_P2_Arrival_Travel": "",
           "Det_P6_Arrival": "38", "Det_P6_Arrival_Travel": "5"}
    assert arrival_travel_times(cfg) == {6: {38: 5.0}}
    roles = parse_detector_roles(cfg)
    assert set(roles["key"]) == {"Det_P2_Arrival", "Det_P6_Arrival"}


# ---------------------------------------------------------------------------
# Corpus: an independent per-arrival recount on 315 and 201
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2] / "intersections"
_CORPUS = [
    ("315_US-20-26_Franklin_Rd_and_KCID_Rd/315_data.db", 2, [54, 55, 56], 5.4,
     datetime(2025, 12, 15), datetime(2025, 12, 16)),
    ("315_US-20-26_Franklin_Rd_and_KCID_Rd/315_data.db", 6, [38, 39, 40], 5.4,
     datetime(2025, 12, 15), datetime(2025, 12, 16)),
    ("201_SH-55_and_Banks-Lowman_Rd/201_data.db", 2, [49], 6.5,
     datetime(2026, 3, 19), datetime(2026, 3, 20)),
]


@pytest.mark.parametrize("db,phase,dets,tt,start,end", _CORPUS)
def test_corpus_matches_per_arrival_recount(db, phase, dets, tt, start, end):
    path = _ROOT / db
    if not path.exists():
        pytest.skip("corpus DB not present")
    from atspm.analysis.detector_inference import _to_epoch
    from atspm.data.reader import get_events_with_cycles_df

    ev = get_events_with_cycles_df(path, start, end,
                                   event_codes=[-1, 1, 8, 9, 10, 11, 12, 82],
                                   timezone="US/Mountain")
    out = approach_delay(ev, phase, dets, travel_time_sec=tt)
    ok = out.loc[~out["censored"]]
    assert len(ok) > 100
    assert out["censored"].mean() < 0.02

    # Recount: for each arrival, the next Code 8 decides its cycle; red if
    # before that green.  Only cycles the core kept are compared.
    t = _to_epoch(ev["timestamp"])
    code, par = ev["event_code"].to_numpy(), ev["parameter"].to_numpy()
    arr = np.sort(t[(code == 82) & np.isin(par, dets)] + tt)
    G = _to_epoch(ok["green_ts"])
    lo, hi = _to_epoch(ok["red_start"]), _to_epoch(ok["next_red_start"])
    i0, i1 = np.searchsorted(arr, lo, "left"), np.searchsorted(arr, hi, "left")
    i2 = np.searchsorted(arr, G, "left")
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    n_red = i2 - i0
    red_delay = n_red * G - (csum[i2] - csum[i0])
    # Arrivals shifted across a gap are dropped by the core; allow that slack.
    assert np.abs((i1 - i0) - ok["arrivals"].astype(int).to_numpy()).sum() <= 5
    assert np.abs(n_red - ok["arrivals_red"].astype(int).to_numpy()).sum() <= 5
    assert ok["total_delay_s"].sum() == pytest.approx(red_delay.sum(), rel=1e-3)
    # Red delay can never exceed the red it waited through.
    per_red = ok["total_delay_s"] / ok["arrivals_red"].astype(float).replace(0, np.nan)
    assert (per_red.dropna() <= ok.loc[per_red.notna(), "red_dur"] + 0.01).all()


@pytest.mark.parametrize("db,phase,dets,tt,start,end", _CORPUS)
def test_corpus_green_arrivals_match_aog(db, phase, dets, tt, start, end):
    path = _ROOT / db
    if not path.exists():
        pytest.skip("corpus DB not present")
    from atspm.data.reader import get_events_with_cycles_df

    ev = get_events_with_cycles_df(path, start, end,
                                   event_codes=[-1, 1, 8, 9, 10, 11, 12, 82],
                                   timezone="US/Mountain")
    ad = approach_delay(ev, phase, dets, travel_time_sec=tt)
    aog = arrival_on_green(ev, phase, dets, arrival_offset_sec=tt)
    assert len(ad) == len(aog)                       # same greens, same order
    ok = ~ad["censored"].to_numpy()
    assert (ad["arrivals_green"].to_numpy()[ok].astype(int)
            == aog["aog"].to_numpy()[ok].astype(int)).all()
