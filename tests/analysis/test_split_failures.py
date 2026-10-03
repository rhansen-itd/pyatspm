"""Golden tests for Purdue split failures (Functional Core).

Target: src/atspm/analysis/split_failures.py.

Contract summary
----------------
GOR window  = [Code 1, Code 8)          (include_yellow: [Code 1, end of yellow))
ROR5 window = [end of yellow, +5 s)      clipped at the phase's next Code 1
fail        = GOR > threshold and ROR5 > threshold (strict)
union       = on while any lane is on;  mean = mean per-lane on-seconds
Lanes with no 81/82 in the window's segment are unknown (excluded, not 0).
A gap marker in [green, ror_end] drops the cycle; lane state never crosses one.
Binned GOR/ROR5 are time-weighted; sf_pct = n_fail / n_cycles.

Timeline used throughout (seconds after each cycle's green onset)::

    0 green (1) · 20 yellow (8) · 24 end yellow (9) + red clearance (10)
    · 26 end red clearance (11) · 60 next green

so GOR = [0, 20) (20 s) and ROR5 = [24, 29) (5 s).
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.split_failures import bin_split_failures, split_failures

_PH = 2
_A, _B, _C = 11, 12, 13
_T0 = 1_000_000.0
_CYC = 60.0


def _events(n_cycles=3, det=None, gaps=(), plan=1, plans=None, short_red_at=None):
    """Flat events frame for phase 2.

    Args:
        n_cycles: Number of cycles (green every 60 s from _T0).
        det: ``{channel: [(t, code), ...]}`` with *t* in seconds after _T0.
        gaps: Gap-marker times (seconds after _T0).
        plan: Coord plan for every cycle unless *plans* is given.
        plans: Per-cycle coord plans.
        short_red_at: ``(cycle_index, next_green_offset)`` — move the next
            cycle's green to this offset after the given cycle's green.
    """
    rows = []
    starts = [_T0 + i * _CYC for i in range(n_cycles)]
    if short_red_at is not None:
        i, off = short_red_at
        starts[i + 1] = starts[i] + off
    for i, s in enumerate(starts):
        p = plans[i] if plans else plan
        for dt, code in ((0, 1), (20, 8), (24, 9), (24, 10), (26, 11)):
            rows.append((s + dt, code, _PH, s, p))
    for ch, evs in (det or {}).items():
        for t, code in evs:
            rows.append((_T0 + t, code, ch, np.nan, np.nan))
    for t in gaps:
        rows.append((_T0 + t, -1, -1, np.nan, np.nan))
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter",
                                     "cycle_start", "coord_plan"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    # cycle_start / coord_plan as the reader supplies them: forward-filled
    df[["cycle_start", "coord_plan"]] = df[["cycle_start", "coord_plan"]].ffill().bfill()
    return df


def _pulse(on, off):
    return [(on, 82), (off, 81)]


def _row(cyc, k=0):
    return cyc.iloc[k]


# ---------------------------------------------------------------------------
# Single lane: windows and closed-form ratios
# ---------------------------------------------------------------------------


def test_single_lane_closed_form():
    ev = _events(2, det={_A: _pulse(5, 15) + _pulse(23, 27)})
    cyc, lanes = split_failures(ev, _PH, [_A])
    r = _row(cyc)
    assert r["g_dur"] == pytest.approx(20.0)
    assert r["r_dur"] == pytest.approx(5.0)
    assert r["g_occ"] == pytest.approx(10.0)
    assert r["r_occ"] == pytest.approx(3.0)       # [24, 27)
    assert r["gor"] == pytest.approx(0.5)
    assert r["ror5"] == pytest.approx(0.6)
    assert not r["fail"]
    # one lane: union == mean == lane
    assert r["gor_union"] == pytest.approx(r["gor_mean"])
    assert lanes.iloc[0]["gor"] == pytest.approx(0.5)
    assert int(r["n_lanes"]) == 1


def test_include_yellow_extends_gor_window_to_end_of_yellow():
    ev = _events(2, det={_A: _pulse(18, 23) + _pulse(40, 41)})
    g, _ = split_failures(ev, _PH, [_A])
    gy, _ = split_failures(ev, _PH, [_A], include_yellow=True)
    assert g.iloc[0]["g_dur"] == pytest.approx(20.0)
    assert g.iloc[0]["g_occ"] == pytest.approx(2.0)
    assert gy.iloc[0]["g_dur"] == pytest.approx(24.0)
    assert gy.iloc[0]["g_occ"] == pytest.approx(5.0)


def test_ror_window_clipped_at_next_green():
    # next green 27 s after this one: ROR5 = [24, 27)
    ev = _events(3, det={_A: _pulse(24, 26.5) + _pulse(100, 101)},
                 short_red_at=(0, 27.0))
    cyc, _ = split_failures(ev, _PH, [_A])
    r = _row(cyc)
    assert r["r_dur"] == pytest.approx(3.0)
    assert r["r_occ"] == pytest.approx(2.5)


def test_failure_threshold_is_strict():
    # integer seconds keep the ratios exact: GOR = 16/20 = 0.8, ROR = 4/5 = 0.8
    ev = _events(2, det={_A: _pulse(4, 20) + _pulse(24, 28) + _pulse(40, 41)})
    at, _ = split_failures(ev, _PH, [_A], threshold=0.8)
    assert _row(at)["gor"] == 0.8 and _row(at)["ror5"] == 0.8
    assert not _row(at)["fail"]
    below, _ = split_failures(ev, _PH, [_A], threshold=0.79)
    assert _row(below)["fail"]


# ---------------------------------------------------------------------------
# The five SPMs detector edge cases (spms_notebook_ideas.md §2.1)
# ---------------------------------------------------------------------------


def test_edge1_consecutive_duplicates_collapse():
    ev = _events(2, det={_A: [(5, 82), (7, 82), (15, 81), (16, 81), (40, 82), (41, 81)]})
    cyc, _ = split_failures(ev, _PH, [_A])
    assert _row(cyc)["g_occ"] == pytest.approx(10.0)


def test_edge2_on_through_window_with_no_transitions():
    # on before green, off after the red-5 window: whole windows occupied
    ev = _events(2, det={_A: [(-1, 82), (40, 81)]})
    cyc, _ = split_failures(ev, _PH, [_A])
    r = _row(cyc)
    assert r["gor"] == pytest.approx(1.0)
    assert r["ror5"] == pytest.approx(1.0)
    assert r["fail"]


def test_edge3_first_event_in_window_is_off():
    # on at -1 (before green) and off at 8: occupied [0, 8)
    ev = _events(2, det={_A: [(-1, 82), (8, 81), (40, 82), (41, 81)]})
    cyc, _ = split_failures(ev, _PH, [_A])
    assert _row(cyc)["g_occ"] == pytest.approx(8.0)


def test_edge3_leading_off_in_segment_means_on_since_segment_start():
    # the data begins with an OFF: the lane was on from the data start
    ev = _events(2, det={_A: [(8, 81), (40, 82), (41, 81)]})
    cyc, _ = split_failures(ev, _PH, [_A])
    assert _row(cyc)["g_occ"] == pytest.approx(8.0)


def test_edge4_actuation_running_past_window_end_is_clipped():
    ev = _events(2, det={_A: _pulse(18, 40)})
    cyc, _ = split_failures(ev, _PH, [_A])
    r = _row(cyc)
    assert r["g_occ"] == pytest.approx(2.0)
    assert r["r_occ"] == pytest.approx(5.0)


def test_edge5_occupancy_capped_at_window_length():
    ev = _events(2, det={_A: _pulse(-30, 100)})
    cyc, _ = split_failures(ev, _PH, [_A])
    r = _row(cyc)
    assert r["g_occ"] == pytest.approx(r["g_dur"])
    assert r["r_occ"] == pytest.approx(r["r_dur"])


def test_still_on_at_end_of_data_counts():
    # on at 70 and never off: cycle 1's windows are occupied to the end
    ev = _events(3, det={_A: _pulse(5, 6) + [(70, 82)]})
    cyc, _ = split_failures(ev, _PH, [_A])
    c1 = cyc.loc[cyc["green_ts"] == _T0 + 60].iloc[0]
    assert c1["g_occ"] == pytest.approx(10.0)    # [70, 80)
    assert c1["r_occ"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Gap markers
# ---------------------------------------------------------------------------


def test_gap_inside_green_drops_cycle():
    ev = _events(3, det={_A: _pulse(5, 6) + _pulse(65, 66) + _pulse(125, 126)}, gaps=[70])
    cyc, lanes = split_failures(ev, _PH, [_A])
    assert _T0 + 60 not in set(cyc["green_ts"])
    assert _T0 + 60 not in set(lanes["green_ts"])


def test_gap_inside_red5_after_code11_drops_cycle():
    # Code 11 at 26; a gap at 27 lies in ROR5 = [24, 29)
    ev = _events(3, det={_A: _pulse(5, 6) + _pulse(65, 66) + _pulse(125, 126)}, gaps=[27])
    cyc, _ = split_failures(ev, _PH, [_A])
    assert _T0 not in set(cyc["green_ts"])
    assert _T0 + 60 in set(cyc["green_ts"])


def test_gap_after_red5_keeps_cycle():
    ev = _events(3, det={_A: _pulse(5, 6) + _pulse(65, 66) + _pulse(125, 126)}, gaps=[40])
    cyc, _ = split_failures(ev, _PH, [_A])
    assert _T0 in set(cyc["green_ts"])


def test_lane_state_never_crosses_a_gap():
    # on at 30 before a gap at 40, no event until an ON at 130: the post-gap
    # cycle at 60 must not inherit "on"; the lane logs nothing in that
    # segment before 130, but it does log in the segment, so it is known.
    ev = _events(4, det={_A: [(5, 82), (6, 81), (30, 82), (130, 82), (131, 81)]}, gaps=[40])
    cyc, _ = split_failures(ev, _PH, [_A])
    c1 = cyc.loc[cyc["green_ts"] == _T0 + 60].iloc[0]
    assert c1["g_occ"] == pytest.approx(0.0)


def test_leading_off_after_gap_is_on_since_the_gap():
    # gap at 40; first lane event after it is an OFF at 68 -> on [40, 68)
    ev = _events(3, det={_A: _pulse(5, 6) + [(68, 81)] + _pulse(125, 126)}, gaps=[40])
    cyc, _ = split_failures(ev, _PH, [_A])
    c1 = cyc.loc[cyc["green_ts"] == _T0 + 60].iloc[0]
    assert c1["g_occ"] == pytest.approx(8.0)


def test_lane_silent_in_segment_is_unknown_not_empty():
    # lane B logs only before the gap at 40: unknown in cycles after it
    ev = _events(3, det={_A: _pulse(5, 15) + _pulse(65, 75) + _pulse(125, 135),
                         _B: _pulse(5, 6)}, gaps=[40])
    cyc, lanes = split_failures(ev, _PH, [_A, _B], aggregate="mean")
    c0 = cyc.loc[cyc["green_ts"] == _T0].iloc[0]
    c1 = cyc.loc[cyc["green_ts"] == _T0 + 60].iloc[0]
    assert int(c0["n_lanes"]) == 2
    assert c0["gor_mean"] == pytest.approx((10 + 1) / 2 / 20)
    assert int(c1["n_lanes"]) == 1
    assert c1["gor_mean"] == pytest.approx(0.5)          # B excluded, not 0
    assert set(lanes.loc[lanes["green_ts"] == _T0 + 60, "det"]) == {_A}


def test_window_with_no_known_lane_is_dropped():
    ev = _events(3, det={_A: _pulse(5, 6)}, gaps=[40])
    cyc, _ = split_failures(ev, _PH, [_A])
    assert list(cyc["green_ts"]) == [_T0]


def test_red5_past_end_of_data_drops_cycle():
    ev = _events(2, det={_A: _pulse(5, 6)})
    ev = ev.loc[ev["timestamp"] < _T0 + 60 + 27].reset_index(drop=True)
    cyc, _ = split_failures(ev, _PH, [_A])
    assert list(cyc["green_ts"]) == [_T0]


# ---------------------------------------------------------------------------
# Lane aggregation: union vs mean, closed form
# ---------------------------------------------------------------------------


def test_union_vs_mean_disjoint_lanes():
    # A [0,10) B [5,15) C [10,20): union covers green, mean = 10 s
    det = {_A: _pulse(0, 10), _B: _pulse(5, 15), _C: _pulse(10, 20)}
    for ch in det:
        det[ch] = det[ch] + _pulse(40, 41)
    ev = _events(2, det=det)
    cyc, lanes = split_failures(ev, _PH, [_A, _B, _C])
    r = _row(cyc)
    assert r["gor_union"] == pytest.approx(1.0)
    assert r["gor_mean"] == pytest.approx(0.5)
    assert sorted(lanes.loc[lanes["green_ts"] == _T0, "gor"]) == pytest.approx([0.5] * 3)


@pytest.mark.parametrize("n, union", [(1, 0.5), (2, 0.75), (3, 0.875)])
def test_union_rises_with_lane_count_for_independent_lanes(n, union):
    # each lane is on half the green, in mutually "independent" patterns
    # (halves, quarters, eighths): union = 1 - (1 - 0.5)**n, mean = 0.5
    patterns = {
        _A: [(0, 10)],
        _B: [(0, 5), (10, 15)],
        _C: [(0, 2.5), (5, 7.5), (10, 12.5), (15, 17.5)],
    }
    chans = [_A, _B, _C][:n]
    det = {ch: sum((_pulse(a, b) for a, b in patterns[ch]), []) + _pulse(40, 41) for ch in chans}
    ev = _events(2, det=det)
    cyc, _ = split_failures(ev, _PH, chans)
    r = _row(cyc)
    assert r["gor_union"] == pytest.approx(union)
    assert r["gor_union"] == pytest.approx(1 - 0.5 ** n)
    assert r["gor_mean"] == pytest.approx(0.5)


def test_one_jammed_lane_beside_empty_lanes():
    # A on through green and red-5; B and C empty in those windows
    det = {_A: _pulse(-1, 40), _B: _pulse(40, 41), _C: _pulse(40, 41)}
    ev = _events(2, det=det)
    union, _ = split_failures(ev, _PH, [_A, _B, _C], aggregate="union")
    mean, _ = split_failures(ev, _PH, [_A, _B, _C], aggregate="mean")
    assert _row(union)["fail"]
    assert not _row(mean)["fail"]
    assert _row(mean)["gor"] == pytest.approx(1 / 3)
    assert int(_row(mean)["n_lanes_failed"]) == 1      # "any lane" would fail it


def test_aggregate_selects_chosen_columns():
    det = {_A: _pulse(0, 10) + _pulse(40, 41), _B: _pulse(5, 15) + _pulse(40, 41)}
    ev = _events(2, det=det)
    u, _ = split_failures(ev, _PH, [_A, _B], aggregate="union")
    m, _ = split_failures(ev, _PH, [_A, _B], aggregate="mean")
    assert _row(u)["g_occ"] == pytest.approx(15.0)
    assert _row(m)["g_occ"] == pytest.approx(10.0)
    # both aggregates are reported either way
    assert _row(u)["gor_mean"] == pytest.approx(_row(m)["gor_mean"])


def test_invalid_aggregate_raises():
    with pytest.raises(ValueError):
        split_failures(_events(2), _PH, [_A], aggregate="max")


def test_empty_inputs_return_schema():
    cyc, lanes = split_failures(_events(2), _PH, [])
    assert cyc.empty and "fail" in cyc.columns
    assert lanes.empty and "det" in lanes.columns
    cyc, _ = split_failures(_events(2, det={_A: _pulse(5, 6)}), 7, [_A])
    assert cyc.empty


def test_tz_aware_timestamps_match_float():
    ev = _events(3, det={_A: _pulse(5, 15) + _pulse(23, 27) + _pulse(70, 90)})
    f_cyc, _ = split_failures(ev, _PH, [_A])
    tz = ev.copy()
    for col in ("timestamp", "cycle_start"):
        tz[col] = pd.to_datetime(tz[col], unit="s", utc=True).dt.tz_convert("US/Mountain")
    t_cyc, _ = split_failures(tz, _PH, [_A])
    cols = ["g_occ", "r_occ", "gor", "ror5"]
    np.testing.assert_allclose(t_cyc[cols].to_numpy(float), f_cyc[cols].to_numpy(float))
    assert list(t_cyc["fail"]) == list(f_cyc["fail"])


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def test_binned_is_time_weighted_per_plan():
    # cycle 0 (plan 1): GOR 1.0, fails.  Cycle 1 (plan 1): GOR 0.  Cycle 2
    # (plan 2): GOR 0.5.  Plan 1's bin: GOR = 20/40 and sf_pct = 1/2.
    det = {_A: _pulse(-1, 40) + _pulse(125, 135) + _pulse(170, 171)}
    ev = _events(3, det=det, plans=[1, 1, 2])
    cyc, _ = split_failures(ev, _PH, [_A])
    b = bin_split_failures(cyc, bin_len=60)
    p1 = b.loc[b["coord_plan"] == 1].iloc[0]
    p2 = b.loc[b["coord_plan"] == 2].iloc[0]
    assert int(p1["n_cycles"]) == 2 and int(p1["n_fail"]) == 1
    assert p1["sf_pct"] == pytest.approx(0.5)
    assert p1["gor"] == pytest.approx(20 / 40)
    assert p1["ror5"] == pytest.approx(5 / 10)
    assert p2["gor"] == pytest.approx(0.5)


def test_binned_time_weighting_differs_from_mean_of_ratios():
    # cycle 0 has a 20 s green fully occupied; cycle 1 a 40 s green, empty.
    rows = []
    for s, g in ((_T0, 20), (_T0 + 60, 40)):
        for dt, code in ((0, 1), (g, 8), (g + 4, 9), (g + 4, 10), (g + 6, 11)):
            rows.append((s + dt, code, _PH, s, 1))
    rows.append((_T0 + 120, 1, _PH, _T0 + 120, 1))
    for t, c in [(-1, 82), (30, 81), (115, 82), (116, 81)]:
        rows.append((_T0 + t, c, _A, _T0, 1))
    ev = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter",
                                     "cycle_start", "coord_plan"]).sort_values("timestamp")
    cyc, _ = split_failures(ev, _PH, [_A])
    b = bin_split_failures(cyc, bin_len=60)
    assert b["gor"].sum() == pytest.approx(20 / 60)


def test_bin_empty():
    assert bin_split_failures(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Oracle: SPMs split_failures, one lane, its window definitions
# ---------------------------------------------------------------------------

_SPMS = Path.home() / "SPMs" / "Notebooks" / "spmfunctions"


def _load_spms_moes():
    pkg = types.ModuleType("_spms_oracle")
    pkg.__path__ = [str(_SPMS)]
    sys.modules["_spms_oracle"] = pkg
    for name in ("misc_tools", "moes"):
        spec = importlib.util.spec_from_file_location(
            f"_spms_oracle.{name}", _SPMS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        src = (_SPMS / f"{name}.py").read_text()
        # pandas 3 refuses an int sentinel in a datetime column; use epoch 0
        src = src.replace("'Period'] = 0", "'Period'] = pd.Timestamp(0)")
        src = src.replace("(dfc.Period != 0)", "(dfc.Period != pd.Timestamp(0))")
        exec(compile(src, spec.origin, "exec"), mod.__dict__)
    return sys.modules["_spms_oracle.moes"]


def _append_shim(self, other, ignore_index=False):
    """``DataFrame.append`` (removed in pandas 2.0) via ``pd.concat``."""
    if isinstance(other, pd.Series):
        other = other.to_frame().T
    return pd.concat([self, other], ignore_index=ignore_index)


@pytest.mark.skipif(not (_SPMS / "moes.py").exists(), reason="SPMs notebooks not present")
@pytest.mark.parametrize("thresh", [0.8, 0.5])
def test_single_lane_matches_spms_oracle(monkeypatch, thresh):
    """One lane, random actuations: our GOR/ROR5 with include_yellow and the
    SPMs threshold equal SPMs' split_failures on the same events.

    Aligned on purpose: Code 9 and Code 10 coincide (so end of yellow = SPMs'
    Code 10), every red lasts > 5 s (no clipping), no gap markers, detector
    events are offset 0.05 s from phase events (no tie-ordering
    differences), and the lane is off at the end of the data.
    """
    monkeypatch.setattr(pd.DataFrame, "append", _append_shim, raising=False)
    moes = _load_spms_moes()

    rng = np.random.default_rng(7)
    n = 40
    rows, t = [], _T0
    for i in range(n):
        g = float(rng.integers(8, 40))
        for dt, code in ((0, 1), (g, 8), (g + 4, 9), (g + 4, 10), (g + 6, 11)):
            rows.append((t + dt, code, _PH, t, 1 + i % 2))
        t += g + 6 + float(rng.integers(10, 40))
    end = t
    # alternating on/off times on a 0.1 s grid, offset by 0.05
    k = int(rng.integers(150, 250)) * 2
    times = np.sort(rng.choice(np.arange(_T0 + 1, end - 30, 0.1), size=k, replace=False)) + 0.05
    for j, ts in enumerate(times):
        rows.append((float(ts), 82 if j % 2 == 0 else 81, _A, np.nan, np.nan))
    ev = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter",
                                     "cycle_start", "coord_plan"])
    ev = ev.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ev[["cycle_start", "coord_plan"]] = ev[["cycle_start", "coord_plan"]].ffill()

    cyc, _ = split_failures(ev, _PH, [_A], threshold=thresh, include_yellow=True)

    raw = pd.DataFrame({
        "TS_start": pd.to_datetime(ev["timestamp"], unit="s").dt.round("ms"),
        "Code": ev["event_code"],
        "ID": ev["parameter"],
        "Coord_plan": ev["coord_plan"],
    })
    ref = moes.split_failures(raw, _PH, _A, t_red=5, thresh=thresh)

    # SPMs keys rows by Code 10; ours by green.  End of yellow = Code 10.
    ours = cyc.assign(
        _key=(pd.to_datetime(cyc["green_ts"] + cyc["g_dur"], unit="s").dt.round("ms"))
    ).set_index("_key")
    ref = ref.copy()
    ref.index = pd.DatetimeIndex(ref.index).round("ms")
    common = ours.index.intersection(ref.index)
    assert len(common) >= n - 2           # the last cycles may lack a red-5 window

    o, r = ours.loc[common], ref.loc[common]
    np.testing.assert_allclose(o["g_dur"], r["G"].astype(float), atol=1e-3)
    np.testing.assert_allclose(o["g_occ"], r["G_occ"].astype(float), atol=1e-3)
    np.testing.assert_allclose(o["gor"], r["GOR"].astype(float), atol=1e-4)
    np.testing.assert_allclose(o["r_occ"], r["R_occ"].astype(float), atol=1e-3)
    np.testing.assert_allclose(o["ror5"], r["ROR"].astype(float), atol=1e-4)
    assert list(o["fail"].astype(int)) == list(r["Split Fail"].astype(int))
    if thresh == 0.5:
        assert 0 < int(o["fail"].sum()) < len(o)     # the flags are exercised
