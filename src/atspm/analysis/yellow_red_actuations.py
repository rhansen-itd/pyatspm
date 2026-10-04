"""
Yellow and Red Actuations (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

UDOT's Yellow and Red Actuations measure (``YellowRedActivationsCycle.cs``,
OpenSourceTransportation/Atspm v5), on green-to-green cycles of one signal::

    green_ts ── green ──▶ yellow_ts ─ yellow ─▶ red_clear_ts ─ red clr ─▶ red_ts ── red ──▶ red_end_ts
    (Code 1)              (Code 8)              (Code 9, else 10)        (Code 11)          (next Code 1)

An actuation (Code 82) of a configured detector is classed by the time *t*
it occurs, with UDOT's boundaries:

* green      — ``[green_ts, yellow_ts)``
* yellow     — ``[yellow_ts, red_clear_ts]``  (UDOT "yellow occurrence")
* red_clear  — ``(red_clear_ts, red_ts]``
* red        — ``(red_ts, red_end_ts)``

A *violation* is a red-clearance or red actuation (``t > red_clear_ts``, as
UDOT's ``d.TimeStamp > RedClearanceEvent``).  It is *severe* when it comes
more than ``severe_sec`` after the start of red, ``t − red_clear_ts >
severe_sec`` (UDOT's ``SevereRedLightViolations``): late entries are the
dangerous ones.  When red clearance is not served (201 logs 9 → 12),
``red_ts == red_clear_ts`` and the red-clearance state is empty.

Signal: a phase (Codes 1/8/9/10/11/12) or, with ``overlap=N``, overlap *N*
(61 green → 63 yellow → 64 red clearance → 65 red, mapped to 1/8/10/11 and
run through the same interval builder).  A protected left that runs a
flashing-yellow-arrow overlap must use the overlap: its phase is "red" for
the whole permissive interval, so every permissive left would count as a
violation (315 P1: 501 phase-red actuations vs 20 overlap-red on 2025-12-15).

Exclusions (the ``TM_Exclusions`` list, ``{detector, phase, status}``) are
applied first with the counts module's gap-aware helper, so a detector
excluded in a state is skipped in that state here too.

Gap Marker Rule (censoring)
---------------------------
Every ``event_code == -1`` row is a gap marker (comms gap ``-1`` and
clock-step fence ``-2`` alike).  A cycle is *censored* — reported, flagged,
counts NA, its actuations not reported — when a gap marker lies in
``[green_ts, red_end_ts)``, or no later green of the signal exists
(``red_end_ts`` unknown).  ``red_end_ts`` is the first green event of the
signal after ``red_ts`` in the raw events, so a green whose end was lost
still closes the red before it.

Package Location: src/atspm/analysis/yellow_red_actuations.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .counts import _apply_exclusions
from .detector_inference import _to_epoch
from .phases import _PHASE_CODES, _build_phase_intervals, _segment_id

_GAP_CODE: int = -1
_CODE_DET_ON: int = 82
_CODE_GREEN: int = 1
_OVERLAP_TO_PHASE_CODE = {61: 1, 63: 8, 64: 10, 65: 11}

DEFAULT_SEVERE_SEC: float = 4.0

STATES = ("green", "yellow", "red_clear", "red")

CYCLE_SCHEMA = [
    "phase",
    "overlap",
    "coord_plan",
    "cycle_start",
    "green_ts",
    "yellow_ts",
    "red_clear_ts",
    "red_ts",
    "red_end_ts",
    "green_dur",
    "yellow_dur",
    "red_clear_dur",
    "red_dur",
    "censored",
    "volume",
    "green_act",
    "yellow_act",
    "red_clear_act",
    "red_act",
    "violations",
    "severe",
    "violation_time_s",
    "yellow_time_s",
]

ACTUATION_SCHEMA = [
    "phase",
    "detector",
    "timestamp",
    "green_ts",
    "state",
    "t_yellow",
    "t_red",
    "violation",
    "severe",
]

SUMMARY_SCHEMA = [
    "time",
    "phase",
    "coord_plan",
    "n_cycles",
    "n_censored",
    "volume",
    "yellow_act",
    "red_clear_act",
    "red_act",
    "violations",
    "severe",
    "pct_violations",
    "pct_severe",
    "pct_violations_udot",
    "violations_per_cycle",
    "avg_violation_time_s",
    "avg_yellow_time_s",
]

_COUNT_COLS = ["volume", "green_act", "yellow_act", "red_clear_act", "red_act",
               "violations", "severe"]


def _signal_intervals(events_df: pd.DataFrame, phase: int, overlap: Optional[int]) -> pd.DataFrame:
    """Green → red intervals of the phase (or overlap), plus the green code's rows."""
    if overlap is None:
        sig = events_df.loc[events_df["event_code"].isin(_PHASE_CODES)
                            & (events_df["parameter"] == phase)]
    else:
        sig = events_df.loc[events_df["event_code"].isin(list(_OVERLAP_TO_PHASE_CODE))
                            & (events_df["parameter"] == overlap)].copy()
        sig["event_code"] = sig["event_code"].map(_OVERLAP_TO_PHASE_CODE).astype(np.int64)
    gaps = events_df.loc[events_df["event_code"] == _GAP_CODE]
    ph_df = pd.concat([sig, gaps]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    ph_df["_seg"] = _segment_id(ph_df)
    ph_df = ph_df.loc[ph_df["event_code"] != _GAP_CODE]
    greens = sig.loc[sig["event_code"] == _CODE_GREEN, ["timestamp", "coord_plan"]]
    if ph_df.empty:
        return pd.DataFrame(), greens
    iv = _build_phase_intervals(ph_df, include_no_clearance=False)
    return iv, greens


def yellow_red_actuations(
    events_df: pd.DataFrame,
    phase: int,
    detector_ids: List[int],
    severe_sec: float = DEFAULT_SEVERE_SEC,
    overlap: Optional[int] = None,
    exclusions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cycle yellow/red actuation counts and the classified actuations.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            Timestamps may be UTC epoch floats or tz-aware Timestamps.  Gap
            markers (``event_code == -1``) must be present.
        phase: Phase the detectors belong to (reported in every row).
        detector_ids: Detectors whose on-events (Code 82) are classified.
        severe_sec: A violation is severe when it occurs more than this
            many seconds after the start of red (end of yellow).  Default 4.
        overlap: Classify against overlap *N* (Codes 61/63/64/65) instead
            of the phase's own indications.
        exclusions: ``TM_Exclusions`` entries ``{detector, phase, status}``;
            matching actuations are dropped before classification.

    Returns:
        ``(cycles, actuations)``.

        *cycles*, one row per green of the signal, sorted by ``green_ts``,
        columns :data:`CYCLE_SCHEMA`::

            phase            int
            overlap          Int64   – NA in phase mode
            coord_plan       float   – plan at the green event
            cycle_start      input dtype – detected (barrier) cycle of the green
            green_ts, yellow_ts, red_clear_ts, red_ts   input dtype
            red_end_ts       input dtype – next green of the signal (NaT/NaN
                                       when none)
            green_dur, yellow_dur, red_clear_dur, red_dur   float s
            censored         bool
            volume           Int64 – all actuations in [green_ts, red_end_ts)
            green_act, yellow_act, red_clear_act, red_act   Int64
            violations       Int64 – red_clear_act + red_act
            severe           Int64
            violation_time_s float – sum of t_red over violations
            yellow_time_s    float – sum of t_yellow over yellow actuations
            (counts NA and times NaN when censored)

        *actuations*, one row per actuation of an uncensored cycle, sorted
        by ``timestamp``, columns :data:`ACTUATION_SCHEMA`::

            phase      int
            detector   int
            timestamp  input dtype
            green_ts   input dtype – the cycle it belongs to
            state      str   – one of :data:`STATES`
            t_yellow   float s – timestamp − yellow_ts (negative in green)
            t_red      float s – timestamp − red_clear_ts (time into red)
            violation  bool
            severe     bool

        Both frames are empty with their schema when the signal has no
        intervals; *actuations* is empty when *detector_ids* is.
    """
    empty = (pd.DataFrame(columns=CYCLE_SCHEMA), pd.DataFrame(columns=ACTUATION_SCHEMA))
    if events_df.empty:
        return empty

    iv, greens = _signal_intervals(events_df, phase, overlap)
    if iv.empty:
        return empty
    iv = iv.sort_values("green_ts", kind="stable").reset_index(drop=True)
    n = len(iv)

    G = _to_epoch(iv["green_ts"])
    Y = _to_epoch(iv["yellow_ts"])
    YE = _to_epoch(iv["yellow_end_ts"])
    CE = _to_epoch(iv["clear_end_ts"])

    # Red ends at the first green of the signal after red starts.
    g_raw = _to_epoch(greens["timestamp"])
    order = np.argsort(g_raw, kind="stable")
    g_sorted = g_raw[order]
    j = np.searchsorted(g_sorted, CE, side="right")
    has_next = j < len(g_sorted)
    RE = np.full(n, np.nan)
    RE[has_next] = g_sorted[j[has_next]]
    gaps = np.sort(_to_epoch(events_df.loc[events_df["event_code"] == _GAP_CODE, "timestamp"]))
    clear = np.searchsorted(gaps, G, side="left") == np.searchsorted(
        gaps, np.where(has_next, RE, G), side="left")
    valid = has_next & clear

    # Red end in the input dtype: the green event's own timestamp.
    green_ts_in = greens["timestamp"].iloc[order].reset_index(drop=True)
    red_end = green_ts_in.reindex(np.where(has_next, j, -1)).reset_index(drop=True)
    plan_by_green = pd.Series(
        pd.to_numeric(greens["coord_plan"], errors="coerce").to_numpy()[order], index=g_sorted
    )
    plan_by_green = plan_by_green[~plan_by_green.index.duplicated()]

    cycles = pd.DataFrame({
        "phase": int(phase),
        "overlap": pd.array([overlap] * n, dtype="Int64"),
        "coord_plan": plan_by_green.reindex(G).fillna(0.0).to_numpy(dtype=float),
        "cycle_start": iv["cycle_start"],
        "green_ts": iv["green_ts"],
        "yellow_ts": iv["yellow_ts"],
        "red_clear_ts": iv["yellow_end_ts"],
        "red_ts": iv["clear_end_ts"],
        "red_end_ts": red_end,
        "green_dur": Y - G,
        "yellow_dur": YE - Y,
        "red_clear_dur": CE - YE,
        "red_dur": RE - CE,
        "censored": ~valid,
    })

    counts = np.zeros((n, 4), dtype=np.int64)
    severe_n = np.zeros(n, dtype=np.int64)
    vtime = np.zeros(n)
    ytime = np.zeros(n)
    acts = pd.DataFrame(columns=ACTUATION_SCHEMA)

    det = events_df.loc[(events_df["event_code"] == _CODE_DET_ON)
                        & events_df["parameter"].isin(list(detector_ids))]
    if exclusions and not det.empty:
        det = _apply_exclusions(det, events_df, exclusions)
    if not det.empty:
        det = det.sort_values("timestamp", kind="stable")
        A = _to_epoch(det["timestamp"])
        k = np.searchsorted(G, A, side="right") - 1
        kk = np.clip(k, 0, n - 1)
        take = (k >= 0) & valid[kk] & (A < np.where(valid, RE, -np.inf)[kk])
        A, kk = A[take], kk[take]
        state = np.select(
            [A < Y[kk], A <= YE[kk], A <= CE[kk]], [0, 1, 2], default=3
        )
        t_red = A - YE[kk]
        t_yel = A - Y[kk]
        viol = state >= 2
        sev = viol & (t_red > severe_sec)
        for s in range(4):
            counts[:, s] = np.bincount(kk[state == s], minlength=n)
        severe_n = np.bincount(kk[sev], minlength=n)
        vtime = np.bincount(kk[viol], weights=t_red[viol], minlength=n)
        ytime = np.bincount(kk[state == 1], weights=t_yel[state == 1], minlength=n)

        acts = pd.DataFrame({
            "phase": int(phase),
            "detector": det["parameter"].to_numpy()[take].astype(np.int64),
            "timestamp": det["timestamp"].iloc[np.flatnonzero(take)].reset_index(drop=True),
            "green_ts": iv["green_ts"].iloc[kk].reset_index(drop=True),
            "state": np.asarray(STATES, dtype=object)[state],
            "t_yellow": np.round(t_yel, 2),
            "t_red": np.round(t_red, 2),
            "violation": viol,
            "severe": sev,
        })[ACTUATION_SCHEMA]

    cycles["volume"] = counts.sum(axis=1)
    cycles["green_act"] = counts[:, 0]
    cycles["yellow_act"] = counts[:, 1]
    cycles["red_clear_act"] = counts[:, 2]
    cycles["red_act"] = counts[:, 3]
    cycles["violations"] = counts[:, 2] + counts[:, 3]
    cycles["severe"] = severe_n
    cycles["violation_time_s"] = vtime
    cycles["yellow_time_s"] = ytime
    for col in _COUNT_COLS:
        cycles[col] = cycles[col].astype("Int64")
        cycles.loc[~valid, col] = pd.NA
    cycles.loc[~valid, ["violation_time_s", "yellow_time_s"]] = np.nan

    cycles = cycles[CYCLE_SCHEMA].round({
        "green_dur": 2, "yellow_dur": 2, "red_clear_dur": 2, "red_dur": 2,
        "violation_time_s": 2, "yellow_time_s": 2,
    })
    return cycles, acts


def summarize_yellow_red(cycle_df: pd.DataFrame, bin_len: Optional[int] = 15) -> pd.DataFrame:
    """Aggregate per-cycle yellow/red actuations into time bins or per plan.

    Counts and times are summed over uncensored cycles before any ratio is
    taken.  A cycle is placed by its ``green_ts``.

    Args:
        cycle_df: First element of :func:`yellow_red_actuations`, possibly
            concatenated over phases.
        bin_len: Bin width in minutes; ``None`` gives one row per
            (phase, coord_plan) over the whole input, with ``time`` the
            first green of that group.

    Returns:
        Columns :data:`SUMMARY_SCHEMA`, sorted by ``phase, time``::

            time                  Timestamp – bin start (UTC when input was epoch)
            phase                 int
            coord_plan            float
            n_cycles, n_censored  int
            volume, yellow_act, red_clear_act, red_act, violations, severe   int
            pct_violations        float – violations / volume
            pct_severe            float – severe / volume
            pct_violations_udot   float – violations / (yellow + red_clear + red
                                         actuations), UDOT's denominator
            violations_per_cycle  float – violations / n_cycles
            avg_violation_time_s  float – mean time into red of a violation
            avg_yellow_time_s     float – mean time into yellow of a yellow
                                         actuation
            (ratios NaN when their denominator is 0)

        A group holding only censored cycles is kept with zero counts and
        NaN ratios, so lost coverage stays visible.
    """
    if cycle_df is None or cycle_df.empty:
        return pd.DataFrame(columns=SUMMARY_SCHEMA)

    df = cycle_df.copy()
    g = df["green_ts"]
    t = g if pd.api.types.is_datetime64_any_dtype(g) else pd.to_datetime(
        g.astype(float), unit="s", utc=True)
    df["time"] = t.dt.floor(f"{bin_len}min") if bin_len is not None else t

    ok = ~df["censored"].astype(bool)
    df["_ok"] = ok.astype(int)
    df["_cens"] = (~ok).astype(int)
    for col in _COUNT_COLS:
        df[col] = df[col].fillna(0).astype(np.int64)
    for col in ("violation_time_s", "yellow_time_s"):
        df[col] = df[col].fillna(0.0)

    keys = ["time", "phase", "coord_plan"] if bin_len is not None else ["phase", "coord_plan"]
    agg = df.groupby(keys, sort=False).agg(
        **({} if bin_len is not None else {"time": ("time", "min")}),
        n_cycles=("_ok", "sum"),
        n_censored=("_cens", "sum"),
        volume=("volume", "sum"),
        yellow_act=("yellow_act", "sum"),
        red_clear_act=("red_clear_act", "sum"),
        red_act=("red_act", "sum"),
        violations=("violations", "sum"),
        severe=("severe", "sum"),
        violation_time_s=("violation_time_s", "sum"),
        yellow_time_s=("yellow_time_s", "sum"),
    ).reset_index()

    def _ratio(num, den):
        num = agg[num].astype(float).to_numpy()
        den = np.asarray(den, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    window = agg["yellow_act"] + agg["red_clear_act"] + agg["red_act"]
    agg["pct_violations"] = _ratio("violations", agg["volume"])
    agg["pct_severe"] = _ratio("severe", agg["volume"])
    agg["pct_violations_udot"] = _ratio("violations", window)
    agg["violations_per_cycle"] = _ratio("violations", agg["n_cycles"])
    agg["avg_violation_time_s"] = _ratio("violation_time_s", agg["violations"])
    agg["avg_yellow_time_s"] = _ratio("yellow_time_s", agg["yellow_act"])
    for col in ["n_cycles", "n_censored", "volume", "yellow_act", "red_clear_act",
                "red_act", "violations", "severe"]:
        agg[col] = agg[col].astype(int)
    agg["phase"] = agg["phase"].astype(int)

    agg = agg.sort_values(["phase", "time"], kind="stable").reset_index(drop=True)
    return agg[SUMMARY_SCHEMA].round({
        "pct_violations": 4, "pct_severe": 4, "pct_violations_udot": 4,
        "violations_per_cycle": 4, "avg_violation_time_s": 2, "avg_yellow_time_s": 2,
    })
