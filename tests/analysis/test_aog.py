# Golden tests for arrival_on_green when a phase is served more than once in
# one detected cycle (UDOT roadmap S-M1 PCD parity finding, 2026-10-02).
#
# Opus-written.  Arrivals are tested against the green that contains them,
# never only against the last green sharing their cycle_start.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.aog import arrival_on_green, bin_arrival_on_green

_PH = 2
_DET = 33
_T0 = 1_000_000.0


def _green(t, g=10.0):
    """Phase-2 green at *t* lasting *g* s, then 3 s yellow and 2 s red clearance."""
    return [(t, 1), (t + g, 8), (t + g + 3, 9), (t + g + 3, 10), (t + g + 5, 11)]


def _events(cycle_starts, greens, arrivals, gaps=()):
    """Flat events with cycle_start assigned as the reader does (last start ≤ t)."""
    rows = [(_T0 + t, c, _PH) for g in greens for t, c in _green(g)]
    rows += [(_T0 + t, 82, _DET) for t in arrivals]
    rows += [(_T0 + t, 81, _DET) for t in arrivals]
    rows += [(_T0 + t, -1, -1) for t in gaps]
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cs = np.asarray([_T0 + c for c in cycle_starts])
    idx = np.searchsorted(cs, df["timestamp"].to_numpy(), side="right") - 1
    df["cycle_start"] = cs[np.clip(idx, 0, None)]
    df["coord_plan"] = 0.0
    return df


def test_two_greens_in_one_cycle_both_count():
    # One detected cycle [0, 100); P2 green twice: [0,10) and [50,60).  Next cycle at 100.
    ev = _events([0, 100], greens=[0, 50, 100], arrivals=[2, 5, 20, 52, 55, 70, 102])
    out = arrival_on_green(ev, _PH, [_DET])
    assert len(out) == 3                                    # one row per green
    assert out["aog"].tolist() == [2, 2, 1]
    assert out["total_arrivals"].tolist() == [2, 4, 1]     # red at 20 and 70 → second green (split at first yellow)
    assert int(out["aog"].sum()) == 5
    assert int(out["total_arrivals"].sum()) == 7


def test_shared_cycle_rows_report_the_cycle_length():
    ev = _events([0, 100, 200], greens=[0, 50, 100, 200], arrivals=[1, 101, 201])
    out = arrival_on_green(ev, _PH, [_DET])
    assert out["cycle_len"].tolist()[:3] == [100.0, 100.0, 100.0]
    assert np.isfinite(out["green_pct"].iloc[:3]).all()


def test_single_green_per_cycle_is_unchanged():
    ev = _events([0, 100, 200], greens=[0, 100, 200],
                 arrivals=[2, 5, 20, 99, 101, 150, 205])
    out = arrival_on_green(ev, _PH, [_DET])
    assert out["aog"].tolist() == [2, 1, 1]
    assert out["total_arrivals"].tolist() == [4, 2, 1]
    assert out["cycle_len"].tolist()[:2] == [100.0, 100.0]


def test_green_running_past_the_next_cycle_start():
    # Detected cycles at 0 and 5; the green [0,10) runs past 5.  Arrival at 7 is on green.
    ev = _events([0, 5], greens=[0, 50], arrivals=[7, 30, 52])
    out = arrival_on_green(ev, _PH, [_DET])
    assert int(out["aog"].sum()) == 2
    assert int(out["total_arrivals"].sum()) == 3


def test_arrival_offset_moves_arrival_into_the_first_green():
    ev = _events([0, 100], greens=[0, 50, 100], arrivals=[45, 101])
    plain = arrival_on_green(ev, _PH, [_DET])
    shifted = arrival_on_green(ev, _PH, [_DET], arrival_offset_sec=6.0)
    assert int(plain["aog"].sum()) == 1
    assert int(shifted["aog"].sum()) == 2


def test_bin_over_shared_cycle_rows():
    ev = _events([0, 100], greens=[0, 50, 100], arrivals=[2, 5, 20, 52, 55, 70, 102])
    b = bin_arrival_on_green(arrival_on_green(ev, _PH, [_DET]), bin_len=60)
    assert int(b["aog"].sum()) == 5
    assert int(b["total_arrivals"].sum()) == 7
    assert np.isfinite(b["green_pct"]).all()


# ---------------------------------------------------------------------------
# Real data: day AoG matches a UDOT-style red-to-red recount on 315
# ---------------------------------------------------------------------------

_DB = Path(__file__).resolve().parents[2] / \
    "intersections/315_US-20-26_Franklin_Rd_and_KCID_Rd/315_data.db"


@pytest.mark.parametrize("phase,dets", [(2, [54, 55, 56]), (6, [38, 39, 40])])
def test_315_green_arrivals_match_red_to_red_recount(phase, dets):
    if not _DB.exists():
        pytest.skip("corpus DB not present")
    from datetime import datetime

    from atspm.data.reader import get_events_with_cycles_df

    ev = get_events_with_cycles_df(_DB, datetime(2025, 12, 15), datetime(2025, 12, 16),
                                   event_codes=[-1, 1, 8, 9, 10, 11, 12, 82],
                                   timezone="US/Mountain")
    out = arrival_on_green(ev, phase, dets)

    # Independent recount: every arrival inside any [Code 1, Code 8) of the phase.
    t = ev["timestamp"]
    t = np.array([x.timestamp() for x in t]) if hasattr(t.iloc[0], "timestamp") else t.to_numpy(float)
    code, par = ev["event_code"].to_numpy(), ev["parameter"].to_numpy()
    g = t[(code == 1) & (par == phase)]
    y_all = t[(code == 8) & (par == phase)]
    j = np.searchsorted(y_all, g, side="left")              # first Code 8 after each green
    nxt = np.append(g[1:], np.inf)
    keep = (j < len(y_all)) & (y_all[np.clip(j, 0, len(y_all) - 1)] < nxt)
    g, y = g[keep], y_all[j[keep]]
    arr = t[(code == 82) & np.isin(par, dets)]
    k = np.searchsorted(g, arr, side="right") - 1
    on = (k >= 0) & (arr < y[np.clip(k, 0, None)])
    expected = int(on.sum())
    # Greens dropped by the gap-marker rule may lose a few arrivals; nothing else may.
    assert expected - 30 <= int(out["aog"].sum()) <= expected
