# Golden tests for detector configuration inference (UDOT S-D6).
#
# Opus-written; implementations must make these pass without editing them.
# Synthetic intersection: P2 and P4 alternate on a 60 s cycle.  One P2 lane
# has an advance zone (1), a stop-line presence zone (2) and a count loop
# downstream of the stop bar (3); vehicles queue on red and discharge 2 s
# apart from green.  P4 has a presence zone (4).  Channel 9 pulses at random.
# With calls=True the controller logs Code 43 in the same tenth as an on-event
# outside the called phase's green (once per red): 1 and 2 call P2, 4 calls
# P4, and count loop 3 calls dummy phase 9 (never green).

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.detector_inference import (
    DIFF_SCHEMA,
    INFERRED_SCHEMA,
    diff_detector_roles,
    infer_detector_roles,
)
from atspm.analysis.detector_roles import parse_detector_roles

_T0 = 1_700_000_000.0          # an hour boundary
_CYC = 60.0
_G2 = (0.0, 27.0)              # P2 green [0, 27), yellow to 30
_G4 = (32.0, 55.0)             # P4 green [32, 55), yellow to 58
_LIGHT_HOURS = {0, 1, 2, 3, 4, 5}


def _union(iv):
    """Merge overlapping [on, off) intervals."""
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _serve(arrivals, green):
    """Stop-line presence holds and departure times for vehicles reaching the stop line."""
    holds, departs = [], []
    last = -np.inf
    for s in np.sort(arrivals):
        c0 = _T0 + np.floor((s - _T0) / _CYC) * _CYC
        g0, g1 = c0 + green[0], c0 + green[1]
        if s < g0:
            nxt = g0
        elif s < g1:
            nxt = s
        else:
            nxt = g0 + _CYC
        d = max(nxt, last + 2.0)
        if d >= nxt + (green[1] - green[0]):     # queue overflow: next green
            d = nxt + _CYC
        last = d
        holds.append((s, max(s + 0.6, d + 0.5)))
        departs.append(d)
    return holds, np.asarray(departs)


def _calls(rows, chan, assign, greens):
    """Code-43 rows: same tenth as an on-event, once per red of the called phase."""
    out = []
    for det, phases in assign.items():
        ons = np.sort([a for a, _ in chan[det]])
        for ph in phases:
            g = greens.get(ph)
            if g is None:                       # never green (dummy phase): always calls
                called_cycle = None
                out += [(np.round(a, 1), 43, ph) for a in ons[::7]]
                continue
            rel = (ons - _T0) % _CYC
            cyc = np.floor((ons - _T0) / _CYC)
            red = (rel < g[0]) | (rel >= g[1])
            first = pd.Series(ons[red]).groupby(cyc[red] + (rel[red] >= g[1])).first()
            out += [(np.round(a, 1), 43, ph) for a in first.to_numpy()]
    return out


def _synthetic(days=2, p6_offset=None, gap_at=None, seed=7, calls=False, multi=False):
    rng = np.random.default_rng(seed)
    hours = days * 24
    rows = []
    n_cyc = int(hours * 3600 / _CYC)
    for k in range(n_cyc):
        c0 = _T0 + k * _CYC
        rows += [(c0 + _G2[0], 1, 2), (c0 + _G2[1], 8, 2), (c0 + _G2[1] + 3, 9, 2),
                 (c0 + _G4[0], 1, 4), (c0 + _G4[1], 8, 4), (c0 + _G4[1] + 3, 9, 4)]
        if p6_offset is not None:
            rows += [(c0 + _G2[0] + p6_offset, 1, 6), (c0 + _G2[1], 8, 6),
                     (c0 + _G2[1] + 3, 9, 6)]

    def poisson(rate_light, rate_busy):
        out = []
        for h in range(hours):
            rate = rate_light if (h % 24) in _LIGHT_HOURS else rate_busy
            n = rng.poisson(rate)
            out.append(_T0 + h * 3600 + np.sort(rng.uniform(0, 3600, n)))
        return np.concatenate(out)

    adv = poisson(30, 240)                                   # P2 lane, advance at 1
    holds2, dep2 = _serve(adv + 4.0, _G2)
    chan = {
        1: [(a, a + 0.3) for a in adv],
        2: _union(holds2),
        3: [(d + 1.0, d + 1.15) for d in dep2],
    }
    holds4, _ = _serve(poisson(20, 120), _G4)
    chan[4] = _union(holds4)
    noise = poisson(60, 60)
    chan[9] = [(a, a + 0.2) for a in noise]
    chan = {d: [(np.round(a, 1), np.round(b, 1)) for a, b in ivs] for d, ivs in chan.items()}
    for det, ivs in chan.items():
        for a, b in ivs:
            rows += [(a, 82, det), (b, 81, det)]
    if calls:
        assign = {1: [2], 2: [2], 3: [9], 4: [4, 2] if multi else [4]}
        rows += _calls(rows, chan, assign, {2: _G2, 4: _G4})
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    if gap_at is not None:
        g = _T0 + gap_at
        df = df[(df["timestamp"] < g) | (df["timestamp"] >= g + 600)]
        df = pd.concat([df, pd.DataFrame([(g, -1, -1)], columns=df.columns)])
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def base():
    return infer_detector_roles(_synthetic()).set_index("detector")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_and_dtypes():
    out = infer_detector_roles(_synthetic(days=1))
    assert list(out.columns) == INFERRED_SCHEMA
    assert out["phase"].dtype == "Int64" and out["lane_group"].dtype == "Int64"
    assert out["detector"].is_monotonic_increasing


def test_empty_input():
    out = infer_detector_roles(pd.DataFrame(columns=["timestamp", "event_code", "parameter"]))
    assert out.empty and list(out.columns) == INFERRED_SCHEMA


def test_datetime_timestamps_match_epoch():
    ev = _synthetic(days=1)
    ev_dt = ev.assign(timestamp=pd.to_datetime(ev["timestamp"], unit="s", utc=True))
    a = infer_detector_roles(ev)
    b = infer_detector_roles(ev_dt)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Roles, phases, lanes
# ---------------------------------------------------------------------------

def test_presence_zone(base):
    r = base.loc[2]
    assert (r["role"], r["phase"], r["confidence"]) == ("occupancy", 2, "high")


def test_count_loop(base):
    r = base.loc[3]
    assert (r["role"], r["phase"], r["confidence"]) == ("stop_bar", 2, "high")


def test_advance_takes_phase_from_its_lane(base):
    # No calls: the phase comes only from the lane chain, so medium.
    r = base.loc[1]
    assert (r["role"], r["phase"], r["confidence"]) == ("arrival", 2, "medium")
    assert "2(4.0s)" in r["downstream"].replace("2(3.9s)", "2(4.0s)").replace("2(4.1s)", "2(4.0s)")


def test_lane_group(base):
    assert base.loc[1, "lane_group"] == base.loc[2, "lane_group"] == base.loc[3, "lane_group"]
    assert pd.isna(base.loc[9, "lane_group"])
    assert not base["wide"].any()


def test_other_phase_presence(base):
    r = base.loc[4]
    assert (r["role"], r["phase"]) == ("occupancy", 4)


def test_random_channel_is_unknown(base):
    r = base.loc[9]
    assert r["role"] == "unknown" and pd.isna(r["phase"])


def test_min_actuations_drops_quiet_channels():
    out = infer_detector_roles(_synthetic(days=1), min_actuations=100_000)
    assert out.empty


# ---------------------------------------------------------------------------
# Concurrency, candidate phases, gap markers
# ---------------------------------------------------------------------------

def test_coincident_phase_is_a_tie():
    out = infer_detector_roles(_synthetic(days=2, p6_offset=0.0)).set_index("detector")
    for det in (2, 3):
        assert pd.isna(out.loc[det, "phase"])
        assert out.loc[det, "candidates"] == "P2|P6"
        assert out.loc[det, "confidence"] == "low"


def test_offset_concurrent_phase_resolves():
    # P6 starts 6 s after P2: the onset jump separates them.
    out = infer_detector_roles(_synthetic(days=2, p6_offset=6.0)).set_index("detector")
    assert out.loc[2, "phase"] == 2
    assert out.loc[3, "phase"] == 2


def test_phases_argument_restricts_candidates():
    out = infer_detector_roles(_synthetic(days=2, p6_offset=0.0), phases=[2, 4]).set_index("detector")
    assert out.loc[2, "phase"] == 2
    assert out.loc[2, "candidates"] == ""


def test_gap_marker_censors_and_does_not_change_the_answer(base):
    out = infer_detector_roles(_synthetic(gap_at=30 * 3600 + 17)).set_index("detector")
    for det in (1, 2, 3, 4, 9):
        assert out.loc[det, "role"] == base.loc[det, "role"]
        assert out.loc[det, "phase"] is base.loc[det, "phase"] or out.loc[det, "phase"] == base.loc[det, "phase"]
    assert out.loc[2, "n_act"] < base.loc[2, "n_act"]


def test_interval_closed_by_gap_marker_is_excluded():
    ev = _synthetic(days=1)
    ev = ev[ev["parameter"] != 9]
    # channel 9: 60 normal pulses, then one ON that a gap marker closes
    t = _T0 + 3600 * np.arange(1, 61) / 3
    extra = [(x, 82, 9) for x in t] + [(x + 0.2, 81, 9) for x in t]
    extra += [(_T0 + 80_000.0, 82, 9), (_T0 + 80_100.0, -1, -1)]
    ev = pd.concat([ev, pd.DataFrame(extra, columns=ev.columns)]).sort_values("timestamp")
    out = infer_detector_roles(ev).set_index("detector")
    assert out.loc[9, "n_act"] == 60
    assert out.loc[9, "med_on"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Diff against configuration
# ---------------------------------------------------------------------------

def _proposed(rows):
    cols = INFERRED_SCHEMA
    base_row = {c: None for c in cols}
    data = []
    for r in rows:
        d = dict(base_row)
        d.update({"lane_group": None, "wide": False, "n_act": 100, "med_on": 1.0,
                  "frac_long": 0.1, "onset_ratio": 1.0, "reg_margin": 0.0,
                  "upstream": "", "downstream": "", "candidates": ""})
        d.update(r)
        data.append(d)
    df = pd.DataFrame(data, columns=cols)
    return df.astype({"detector": "int64", "phase": "Int64", "lane_group": "Int64",
                      "onset_phase": "Int64", "reg_phase": "Int64", "n_act": "int64"})


def test_diff_statuses():
    proposed = _proposed([
        {"detector": 10, "role": "occupancy", "phase": 2, "confidence": "high"},     # match
        {"detector": 11, "role": "occupancy", "phase": 3, "confidence": "high"},     # conflict
        {"detector": 12, "role": "stop_bar", "phase": None, "confidence": "low",
         "candidates": "P4|P8"},                                                     # consistent
        {"detector": 13, "role": "occupancy", "phase": 6, "confidence": "medium"},   # new
        {"detector": 14, "role": "unknown", "phase": None, "confidence": "low"},     # unclassified
        {"detector": 15, "role": "stop_bar", "phase": None, "confidence": "low",
         "candidates": "P4|P8"},                                                     # conflict
    ])
    cfg = parse_detector_roles({
        "Det_P2_Occupancy": "10", "Det_P4_Stop_Bar": "11",
        "Det_P8_Stop_Bar": "12", "Det_P3_Stop_Bar": "15",
        "Det_P6_Arrival": "20", "Det_P7_Arrival": "21",
        "TM_EBT": "13", "Det_P2_Pairs": "[[13,14]]",
    })
    out = diff_detector_roles(proposed, cfg, active_counts={21: 5}).set_index("detector")
    assert list(out.reset_index().columns) == DIFF_SCHEMA
    assert out["status"].to_dict() == {
        10: "match", 11: "conflict", 12: "consistent", 13: "new",
        14: "unclassified", 15: "conflict", 20: "silent", 21: "low_volume",
    }
    assert out.loc[11, "configured"] == "stop_bar P4"
    assert (out.loc[11, "proposed_role"], out.loc[11, "proposed_phase"]) == ("occupancy", 3)
    assert out.loc[21, "n_act"] == 5


def test_diff_detector_with_two_configured_roles():
    proposed = _proposed([{"detector": 41, "role": "occupancy", "phase": 3, "confidence": "high"}])
    cfg = parse_detector_roles({"Det_P4_Stop_Bar": "41", "Det_P3_Occupancy": "41"})
    out = diff_detector_roles(proposed, cfg).set_index("detector")
    assert out.loc[41, "status"] == "match"
    assert out.loc[41, "configured"] == "occupancy P3,stop_bar P4"


# ---------------------------------------------------------------------------
# Corpus goldens (skip without the DBs)
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]


def _corpus(folder, db, start, days):
    path = _REPO / "intersections" / folder / db
    if not path.exists():
        pytest.skip("corpus DB not present")
    import sqlite3

    import pytz

    t0 = pytz.timezone("US/Mountain").localize(datetime.fromisoformat(start)).timestamp()
    with sqlite3.connect(path) as con:
        ev = pd.read_sql(
            "SELECT timestamp, event_code, parameter FROM events WHERE timestamp >= ? "
            "AND timestamp < ? AND event_code IN (-1, 1, 8, 9, 43, 81, 82) ORDER BY timestamp",
            con, params=(t0, t0 + days * 86400))
    return ev, path


def test_315_confirmed_mapping():
    ev, _ = _corpus("315_US-20-26_Franklin_Rd_and_KCID_Rd", "315_data.db", "2025-12-15", 3)
    out = infer_detector_roles(ev).set_index("detector")
    expect = {
        **{d: ("occupancy", 2) for d in (50, 51, 52)},
        **{d: ("occupancy", 6) for d in (34, 35, 36)},
        **{d: ("stop_bar", 2) for d in (26, 27, 28)},
        **{d: ("stop_bar", 6) for d in (18, 19, 20)},
        **{d: ("arrival", 2) for d in (54, 55, 56, 53)},
        **{d: ("arrival", 6) for d in (38, 39, 40, 37)},
    }
    for det, (role, ph) in expect.items():
        assert (out.loc[det, "role"], out.loc[det, "phase"]) == (role, ph), det
        assert out.loc[det, "confidence"] == "high", det
    # lanes: advance → presence → count loop
    for lane in ((54, 50, 26), (55, 51, 27), (56, 52, 28),
                 (38, 34, 18), (39, 35, 19), (40, 36, 20)):
        assert out.loc[list(lane), "lane_group"].nunique() == 1, lane
    assert out.loc[37, "wide"] and out.loc[53, "wide"]
    # Minor approaches: P4/P8 onsets coincide, so only the controller's calls place them.
    for det, ph in {42: 4, 43: 3, 44: 8, 58: 8, 59: 7, 60: 4}.items():
        assert (out.loc[det, "phase"], out.loc[det, "confidence"]) == (ph, "high"), det
    # Count loops call dummy phase 9 at 315: only stray same-tenth calls of
    # other detectors match them, never decisively.
    assert (out.loc[[18, 26], "call_share"].fillna(0) < 0.8).all()


def test_201_diff():
    ev, path = _corpus("201_SH-55_and_Banks-Lowman_Rd", "201_data.db", "2026-06-21", 3)
    from atspm.data.manager import DatabaseManager

    with DatabaseManager(path) as m:
        cfg = m.get_config_at_date(datetime(2026, 6, 23))
    counts = ev[ev["event_code"] == 82].groupby("parameter").size().to_dict()
    from atspm.analysis.cycles import _parse_ring_groups

    ring = sorted({p for k in ("RB_R1", "RB_R2") for g in _parse_ring_groups(cfg.get(k)) for p in g})
    out = diff_detector_roles(infer_detector_roles(ev, phases=ring), parse_detector_roles(cfg), counts)
    out = out.set_index("detector")
    assert (out.loc[41, "status"], out.loc[41, "proposed_role"], out.loc[41, "proposed_phase"]) == \
        ("conflict", "occupancy", 3)
    for det in (60, 63, 64):
        assert out.loc[det, "status"] == "silent"
    assert out.loc[38, "status"] == "match"
    assert out.loc[33, "status"] == "match"           # its calls say P6; timing alone tied P2|P6


# ---------------------------------------------------------------------------
# Phase calls (Code 43 in the same tenth as the on-event)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def called():
    return infer_detector_roles(_synthetic(days=2, calls=True)).set_index("detector")


def test_calls_give_advance_its_own_phase(called):
    r = called.loc[1]
    assert (r["role"], r["phase"], r["confidence"]) == ("arrival", 2, "high")
    assert r["call_phase"] == 2 and r["call_share"] >= 0.99


def test_calls_break_a_coincident_tie():
    out = infer_detector_roles(_synthetic(days=2, p6_offset=0.0, calls=True)).set_index("detector")
    assert (out.loc[2, "phase"], out.loc[2, "confidence"], out.loc[2, "candidates"]) == (2, "high", "")
    assert out.loc[1, "phase"] == 2


def test_calls_to_a_dummy_phase_are_ignored(called):
    r = called.loc[3]
    # its own calls go to P9 (not a candidate); only stray coincidences remain
    assert r["n_calls"] < 0.2 * r["n_act"]
    assert (r["role"], r["phase"]) == ("stop_bar", 2)


def test_detector_calling_two_phases_is_reported_as_candidates():
    out = infer_detector_roles(_synthetic(days=2, calls=True, multi=True)).set_index("detector")
    r = out.loc[4]
    assert pd.isna(r["phase"]) and r["candidates"] == "P2|P4" and r["confidence"] == "medium"


def test_calls_override_timing():
    ev = _synthetic(days=2, calls=True)
    # Relabel P4 zone 4's calls as P2: the controller says P2, timing says P4.
    is4 = (ev["event_code"] == 43) & (ev["parameter"] == 4)
    ev.loc[is4, "parameter"] = 2
    out = infer_detector_roles(ev).set_index("detector")
    assert (out.loc[4, "phase"], out.loc[4, "confidence"]) == (2, "high")
    assert out.loc[4, "onset_phase"] == 4


def test_too_few_calls_leave_timing_in_charge():
    out = infer_detector_roles(_synthetic(days=2, calls=True), call_min_calls=10**9)
    out = out.set_index("detector")
    assert out.loc[1, "confidence"] == "medium"        # chain only
    assert out.loc[2, "phase"] == 2
