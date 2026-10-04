"""Goldens for the S-M6 preemption core (analysis/preempt.py)."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

from atspm.analysis.preempt import (
    EPISODE_SCHEMA,
    PREEMPT_CODES,
    SUMMARY_SCHEMA,
    preempt_episodes,
    preempt_summary,
)

TZ = "US/Mountain"
T0 = pytz.timezone(TZ).localize(pd.Timestamp("2026-01-10 08:00").to_pydatetime()).timestamp()


def _ev(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])
    df["timestamp"] = df["timestamp"].astype(float) + T0
    return df.sort_values(["timestamp", "event_code"]).reset_index(drop=True)


def _one(rows):
    out = preempt_episodes(_ev(rows))
    assert len(out) == 1, out
    return out.iloc[0]


def test_empty_and_schema():
    out = preempt_episodes(_ev([]))
    assert out.empty and list(out.columns) == EPISODE_SCHEMA
    assert preempt_episodes(_ev([(0, 105, 1), (5, 111, 1)])).empty   # no Call On
    for c in (-1, 102, 103, 104, 105, 106, 107, 110, 111, 116):
        assert c in PREEMPT_CODES


def test_full_rail_sequence():
    r = _one([(0, 102, 1), (3, 105, 1), (3, 116, 1), (5, 103, 1), (8, 106, 1),
              (20, 107, 1), (50, 104, 1), (55, 111, 1)])
    assert r.preempt == 1 and r.served and not r.censored and not r.max_presence
    assert (r.entry_delay_s, r.track_clear_s, r.dwell_s, r.service_s, r.call_s) == (3, 12, 35, 52, 50)
    assert r.gate_down_ts == T0 + 5 and r.n_force_off == 1 and r.n_reapplied == 0


def test_no_track_clearance_like_315():
    r = _one([(0, 102, 5), (0, 105, 5), (0, 116, 5), (6.2, 107, 5), (36.3, 104, 5),
              (36.3, 116, 5), (41.4, 111, 5)])
    assert r.entry_delay_s == 0 and np.isnan(r.track_clear_s)
    assert round(r.dwell_s, 1) == 35.2 and round(r.service_s, 1) == 41.4 and r.n_force_off == 2


def test_reapplication_during_service_is_same_request():
    """315 2025-12-14 04:07: call drops, comes back during dwell, re-logs 107."""
    r = _one([(0, 102, 5), (0, 105, 5), (0, 116, 5), (6.2, 107, 5), (10.3, 104, 5),
              (11.2, 116, 5), (14.0, 102, 5), (18.2, 116, 5), (27.8, 107, 5),
              (36.3, 104, 5), (36.3, 116, 5), (41.4, 111, 5)])
    assert r.n_reapplied == 1 and r.n_force_off == 4
    assert round(r.dwell_s, 1) == 35.2          # from the first dwell
    assert round(r.call_s, 1) == 36.3           # to the last Call Off


def test_unserved_then_served():
    out = preempt_episodes(_ev([
        (0, 102, 2), (4, 104, 2),                         # dropped during delay
        (30, 102, 2), (40, 105, 2), (45, 107, 2), (70, 104, 2), (75, 111, 2),
    ]))
    assert out["served"].tolist() == [False, True]
    assert out["call_on"].tolist() == [T0, T0 + 30]
    assert out["call_s"].tolist() == [4, 40]
    assert out["entry_delay_s"].iloc[1] == 10 and np.isnan(out["entry_delay_s"].iloc[0])
    assert not out["censored"].any()


def test_max_presence():
    r = _one([(0, 102, 1), (0, 105, 1), (5, 107, 1), (125, 110, 1), (130, 111, 1), (200, 104, 1)])
    assert r.max_presence and r.served and r.dwell_s == 125
    # the late Call Off falls after Exit: it belongs to no request
    assert np.isnan(r.call_off) or r.call_off < T0 + 130


def test_gap_marker_censors_and_splits():
    out = preempt_episodes(_ev([
        (0, 102, 1), (0, 105, 1), (5, 107, 1),
        (20, -1, 0),                                       # hard reset mid-dwell
        (30, 104, 1), (35, 111, 1),                        # tail: no Call On in segment
        (100, 102, 1), (100, 105, 1), (106, 107, 1), (130, 104, 1), (135, 111, 1),
    ]))
    assert len(out) == 2
    a, b = out.iloc[0], out.iloc[1]
    assert a.censored and a.served and np.isnan(a.exit_ts) and np.isnan(a.dwell_s)
    assert not b.censored and b.dwell_s == 29


def test_data_end_censors_unserved_request():
    r = _one([(0, 102, 1)])
    assert r.censored and not r.served


def test_events_before_first_call_dropped():
    r = _one([(0, 107, 1), (3, 104, 1), (5, 111, 1), (50, 102, 1), (50, 105, 1),
              (55, 107, 1), (80, 104, 1), (85, 111, 1)])
    assert r.call_on == T0 + 50 and r.dwell_s == 30


def test_interleaved_preempts():
    out = preempt_episodes(_ev([
        (0, 102, 1), (0, 105, 1), (10, 102, 2), (12, 107, 1), (30, 104, 1),
        (35, 111, 1), (36, 105, 2), (40, 107, 2), (60, 104, 2), (65, 111, 2),
    ]))
    assert out["preempt"].tolist() == [1, 2]
    assert out["entry_delay_s"].tolist() == [0, 26]
    assert out["service_s"].tolist() == [35, 29]


def test_datetime_timestamps():
    ev = _ev([(0, 102, 1), (0, 105, 1), (5, 107, 1), (30, 104, 1), (35, 111, 1)])
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
    r = preempt_episodes(ev).iloc[0]
    assert r.call_on == pytest.approx(T0) and r.dwell_s == pytest.approx(30)


def test_summary():
    ev = _ev([
        (0, 102, 1), (4, 104, 1),
        (30, 102, 1), (32, 105, 1), (32, 116, 1), (40, 107, 1), (70, 104, 1), (75, 111, 1),
        (100, 102, 2), (100, 105, 2), (110, 107, 2), (120, -1, 0),
        (86400, 102, 1), (86400, 105, 1), (86410, 107, 1), (86430, 104, 1), (86440, 111, 1),
    ])
    s = preempt_summary(preempt_episodes(ev), TZ)
    assert list(s.columns) == SUMMARY_SCHEMA
    rows = {(r.date, r.preempt): r for r in s.itertuples()}
    d1, d2 = date(2026, 1, 10), date(2026, 1, 11)
    r = rows[(d1, 1)]
    assert (r.requests, r.served, r.unserved, r.censored, r.force_offs) == (2, 1, 1, 0, 1)
    assert r.entry_delay_mean_s == 2 and r.dwell_mean_s == 35 and r.service_mean_s == 43
    r = rows[(d1, 2)]
    assert (r.requests, r.served, r.censored) == (1, 1, 1) and np.isnan(r.dwell_mean_s)
    assert rows[(d2, 1)].dwell_max_s == 30


def test_summary_empty():
    s = preempt_summary(preempt_episodes(_ev([])), TZ)
    assert s.empty and list(s.columns) == SUMMARY_SCHEMA


_REPO = Path(__file__).resolve().parents[2]


def test_corpus_315():
    path = _REPO / "intersections" / "315_US-20-26_Franklin_Rd_and_KCID_Rd" / "315_data.db"
    if not path.exists():
        pytest.skip("corpus DB not present")
    import sqlite3
    with sqlite3.connect(path) as con:
        ev = pd.read_sql(
            f"SELECT timestamp, event_code, parameter FROM events WHERE event_code IN "
            f"({','.join(map(str, PREEMPT_CODES))}) ORDER BY timestamp, event_code", con)
    out = preempt_episodes(ev)
    # 24 Call Ons = 23 requests + 1 re-application (2025-12-14 04:07, preempt 5)
    assert len(out) == 23 and out["n_reapplied"].sum() == 1
    assert out["served"].all() and not out["censored"].any()
    assert sorted(out["preempt"].unique()) == [3, 4, 5]
    assert out["n_force_off"].sum() == 48
    assert (out["entry_delay_s"] == 0).all()
    assert out["track_clear_s"].isna().all()
    assert out["dwell_s"].between(10, 40).all()
