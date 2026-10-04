"""
Preemption Detail (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

Pairs the preemption codes of the Indiana enumerations into one row per
preempt request:

    102 Call Input On      103 Gate Down        104 Call Input Off
    105 Entry Started      106 Begin Track Clearance
    107 Begin Dwell        110 Max Presence Exceeded
    111 Begin Exit         116 Preemption Force Off

Episodes
--------
Per preempt number, an episode runs from a Call On to its Begin Exit (111),
inclusive.  Inside a served episode a further Call On is a *re-application*
(the input dropped and came back during service, which also re-logs 107),
not a new request: 315 2025-12-14 04:07 shows one.  A request whose input
drops before Entry Started is *unserved* and ends at that Call Off.

Exit has no end code, so no exit duration is reported, only ``exit_ts``.

Gap markers
-----------
``event_code = -1`` splits the stream; no episode spans one.  An episode
whose terminal event (111, or the 104 of an unserved request) isn't seen
before a gap marker or the end of the data is ``censored``; its durations
use only the events that were logged.

Package Location: src/atspm/analysis/preempt.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.timezone import resolve_pytz

_GAP_CODE = -1
_CALL_ON, _GATE_DOWN, _CALL_OFF = 102, 103, 104
_ENTRY, _TRACK, _DWELL = 105, 106, 107
_MAX_PRESENCE, _EXIT, _FORCE_OFF = 110, 111, 116

#: Every event code the preemption core reads (gap markers included).
PREEMPT_CODES: tuple = (
    _GAP_CODE, _CALL_ON, _GATE_DOWN, _CALL_OFF, _ENTRY, _TRACK, _DWELL,
    _MAX_PRESENCE, _EXIT, _FORCE_OFF,
)

EPISODE_SCHEMA = [
    "preempt", "call_on", "call_off", "served", "entry_ts", "gate_down_ts",
    "track_ts", "dwell_ts", "exit_ts", "max_presence", "n_reapplied",
    "n_force_off", "entry_delay_s", "track_clear_s", "dwell_s", "service_s",
    "call_s", "censored",
]
_EPISODE_DTYPES = {
    "preempt": "int64", "call_on": "float64", "call_off": "float64",
    "served": "bool", "entry_ts": "float64", "gate_down_ts": "float64",
    "track_ts": "float64", "dwell_ts": "float64", "exit_ts": "float64",
    "max_presence": "bool", "n_reapplied": "int64", "n_force_off": "int64",
    "entry_delay_s": "float64", "track_clear_s": "float64",
    "dwell_s": "float64", "service_s": "float64", "call_s": "float64",
    "censored": "bool",
}

SUMMARY_SCHEMA = [
    "date", "preempt", "requests", "served", "unserved", "censored",
    "max_presence", "force_offs", "entry_delay_mean_s", "entry_delay_max_s",
    "dwell_mean_s", "dwell_max_s", "service_mean_s", "service_max_s",
]


def _epoch(ts: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(ts):
        t = pd.to_datetime(ts, utc=True)
        return (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy(float)
    return ts.to_numpy(dtype=float)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=EPISODE_SCHEMA).astype(_EPISODE_DTYPES)


def preempt_episodes(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per preempt request, from raw events.

    Args:
        events_df: Events with ``timestamp`` (UTC epoch seconds or
            datetimes), ``event_code`` and ``parameter``.  Pass
            :data:`PREEMPT_CODES`, gap markers included.

    Returns:
        DataFrame (:data:`EPISODE_SCHEMA`), sorted by ``call_on`` then
        preempt.  Timestamps are UTC epoch seconds; NaN when not logged::

            preempt        int64   – preempt number
            call_on        float64 – the request's Call On (102)
            call_off       float64 – last Call Off (104) in the episode
            served         bool    – Entry Started (105) was logged
            entry_ts       float64 – 105
            gate_down_ts   float64 – first 103
            track_ts       float64 – first 106
            dwell_ts       float64 – first 107
            exit_ts        float64 – 111
            max_presence   bool    – 110 was logged
            n_reapplied    int64   – Call Ons after entry (same episode)
            n_force_off    int64   – 116 count
            entry_delay_s  float64 – entry_ts − call_on
            track_clear_s  float64 – dwell_ts − track_ts
            dwell_s        float64 – exit_ts − dwell_ts
            service_s      float64 – exit_ts − entry_ts
            call_s         float64 – call_off − call_on
            censored       bool    – terminal event not logged (gap
                                     marker or data end came first)

        Events before a segment's first Call On (a request already running
        when the data or a gap starts) belong to no row.
    """
    if events_df is None or events_df.empty:
        return _empty()

    ev = pd.DataFrame({
        "t": _epoch(events_df["timestamp"]),
        "c": events_df["event_code"].to_numpy(dtype=np.int64),
        "p": events_df["parameter"].to_numpy(dtype=np.int64),
    }).sort_values(["t", "c"], kind="mergesort")
    ev["seg"] = (ev["c"] == _GAP_CODE).cumsum()
    ev = ev.loc[ev["c"].isin(PREEMPT_CODES[1:])]
    if not (ev["c"] == _CALL_ON).any():
        return _empty()
    # Same-instant order: Call On first, Exit last.
    rank = {_CALL_ON: 0, _ENTRY: 1, _GATE_DOWN: 2, _TRACK: 3, _DWELL: 4,
            _FORCE_OFF: 5, _CALL_OFF: 6, _MAX_PRESENCE: 7, _EXIT: 8}
    ev["r"] = ev["c"].map(rank)
    ev = ev.sort_values(["p", "seg", "t", "r"], kind="mergesort").reset_index(drop=True)
    key = [ev["p"], ev["seg"]]

    # 1. Chunks end at each Exit (111): the chunk id rises after every 111.
    after_exit = (ev["c"] == _EXIT).astype(int).groupby(key).shift(1, fill_value=0)
    ev["chunk"] = after_exit.groupby(key).cumsum()
    ck = [ev["p"], ev["seg"], ev["chunk"]]

    # 2. Within a chunk, the first Entry splits served from unserved calls.
    is_entry = ev["c"] == _ENTRY
    entered = is_entry.astype(int).groupby(ck).cumsum() > 0
    is_on = ev["c"] == _CALL_ON
    # Call Ons before entry each open a request; the last of them is the
    # one the entry serves.  Call Ons after entry are re-applications.
    pre_on = is_on & ~entered
    ev["req"] = pre_on.astype(int).groupby(ck).cumsum()
    n_pre = pre_on.groupby(ck).transform("sum")
    has_entry = is_entry.groupby(ck).transform("any")
    # Post-entry rows join the serving (last pre-entry) request.
    ev.loc[entered, "req"] = n_pre[entered]
    ev["served_req"] = has_entry & (ev["req"] == n_pre)
    # Rows before a chunk's first Call On belong to no request.
    ev = ev.loc[ev["req"] > 0]
    if ev.empty:
        return _empty()

    g = ev.groupby(["p", "seg", "chunk", "req"], sort=False)

    def first(code):
        return ev["t"].where(ev["c"] == code).groupby([ev["p"], ev["seg"], ev["chunk"], ev["req"]], sort=False).min()

    def last(code):
        return ev["t"].where(ev["c"] == code).groupby([ev["p"], ev["seg"], ev["chunk"], ev["req"]], sort=False).max()

    def count(code):
        return (ev["c"] == code).groupby([ev["p"], ev["seg"], ev["chunk"], ev["req"]], sort=False).sum()

    out = pd.DataFrame({
        "preempt": g["p"].first(),
        "call_on": first(_CALL_ON),
        "call_off": last(_CALL_OFF),
        "served": g["served_req"].first(),
        "entry_ts": first(_ENTRY),
        "gate_down_ts": first(_GATE_DOWN),
        "track_ts": first(_TRACK),
        "dwell_ts": first(_DWELL),
        "exit_ts": first(_EXIT),
        "max_presence": count(_MAX_PRESENCE) > 0,
        "n_reapplied": count(_CALL_ON) - 1,
        "n_force_off": count(_FORCE_OFF),
    })
    out["entry_delay_s"] = out["entry_ts"] - out["call_on"]
    out["track_clear_s"] = out["dwell_ts"] - out["track_ts"]
    out["dwell_s"] = out["exit_ts"] - out["dwell_ts"]
    out["service_s"] = out["exit_ts"] - out["entry_ts"]
    out["call_s"] = out["call_off"] - out["call_on"]
    out["censored"] = np.where(out["served"], out["exit_ts"].isna(), out["call_off"].isna())

    out = out.sort_values(["call_on", "preempt"], kind="mergesort").reset_index(drop=True)
    return out[EPISODE_SCHEMA].astype(_EPISODE_DTYPES)


def preempt_summary(episodes: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Per local day and preempt: request vs service counts and durations.

    Args:
        episodes: Output of :func:`preempt_episodes`.
        tz: IANA zone; days are local, keyed by ``call_on``.

    Returns:
        DataFrame (:data:`SUMMARY_SCHEMA`), sorted by date, preempt::

            date                object  – local datetime.date
            preempt             int64
            requests            int64   – rows (re-applications excluded)
            served              int64
            unserved            int64   – not served and not censored
            censored            int64
            max_presence        int64   – requests that hit max presence
            force_offs          int64   – 116 events
            entry_delay_mean_s  float64 – over served requests
            entry_delay_max_s   float64
            dwell_mean_s        float64 – over uncensored served requests
            dwell_max_s         float64
            service_mean_s      float64 – entry → exit, same set
            service_max_s       float64
    """
    if episodes is None or episodes.empty:
        return pd.DataFrame(columns=SUMMARY_SCHEMA)
    e = episodes.copy()
    e["date"] = pd.to_datetime(e["call_on"], unit="s", utc=True).dt.tz_convert(resolve_pytz(tz)).dt.date
    done = e["served"] & ~e["censored"]
    e["_unserved"] = ~e["served"] & ~e["censored"]
    e["_entry"] = e["entry_delay_s"].where(e["served"])
    e["_dwell"] = e["dwell_s"].where(done)
    e["_service"] = e["service_s"].where(done)
    g = e.groupby(["date", "preempt"], sort=True)
    out = pd.DataFrame({
        "requests": g.size(),
        "served": g["served"].sum(),
        "unserved": g["_unserved"].sum(),
        "censored": g["censored"].sum(),
        "max_presence": g["max_presence"].sum(),
        "force_offs": g["n_force_off"].sum(),
        "entry_delay_mean_s": g["_entry"].mean(),
        "entry_delay_max_s": g["_entry"].max(),
        "dwell_mean_s": g["_dwell"].mean(),
        "dwell_max_s": g["_dwell"].max(),
        "service_mean_s": g["_service"].mean(),
        "service_max_s": g["_service"].max(),
    }).reset_index()
    int_cols = ["requests", "served", "unserved", "censored", "max_presence", "force_offs"]
    out[int_cols] = out[int_cols].astype("int64")
    return out[SUMMARY_SCHEMA]
