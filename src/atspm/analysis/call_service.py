"""
Call-to-Service Pairing: Pedestrian Delay and Wait Time (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

Both measures time the first call since a phase was last served to the
service that answers it.  One helper, :func:`first_call_in_windows`, does
that pairing on gap-free windows ``[open, service]``; each measure only
says where its windows open and close.  ``ped_counts`` shares it through
:func:`ped_service_calls`.

Pedestrian Delay (UDOT ``PedPhaseService``, OpenSourceTransportation/Atspm v5)
-----------------------------------------------------------------------------
Per phase, a window opens at Begin Ped Clearance (Code 22) and closes at the
next Begin Walk (Code 21)::

    ── walk ──▶ clearance (22) ── don't walk ──▶ walk (21)
                │ window: first press here ……… delay = walk − press

* **Call code.**  Ped detector on (Code 90, UDOT's choice) when the phase
  logs any, else Ped Call Registered (Code 45).  Code 90's parameter is the
  ped detector; it is taken as the phase (UDOT's default when an approach
  names no ped detectors) unless ``ped_detectors`` maps it.  On 315 and
  701 the 90 parameters are 2/4/6/8 and 2/3/4/6, the phases themselves,
  and Code 45 logs in the same decisecond as the first 90.
* **Repeat presses** in one window count once (the first starts the delay);
  ``n_presses`` keeps the raw count.
* **A press during the walk** (after Code 21, before its Code 22) is served
  by that walk: a ``kind="in_walk"`` row with delay 0 (UDOT's case 2).  The
  controller agrees: at 315 a press 3 s before clearance logs a fresh 45
  that is not carried to the next walk.
* **A walk with no press** in its window is ``kind="uncalled"`` (ped recall,
  or a call placed before the data starts); it has no delay.
* **A press logged in the same decisecond as the walk** belongs to that walk
  (0.1 s resolution can't order them).
* Departure from UDOT: UDOT matches the literal sequence 22 → 90 → 21 after
  collapsing only *adjacent* 90s, so a window with two presses (90, 89, 90)
  matches nothing and is dropped.  Here every called window is kept.

Ped delay needs Codes 21/22 and 45 or 90, which only 315 and 701 log.
:func:`ped_delay` returns an empty frame and a *reason* instead of raising
when they are missing.

Wait Time (UDOT ``WaitTimeService``, ``CycleService.GetWaitTimeCyclesAsync``)
------------------------------------------------------------------------------
Per phase, a window opens when the phase's red starts (end of red
clearance, Code 11; Code 9 when red clearance is not served, as at 201 P3)
and closes at its next green (Code 1).  The wait is ``green − first Phase
Call Registered (Code 43) in the window``, placed at the green.

* **Dropping algorithm** (UDOT, for approaches with stop-bar presence):
  with ``dropping=True`` the wait starts at the first Code 43 after the last
  Phase Call Dropped (Code 44) in the window; a window whose last event is
  a drop has no wait.  A vehicle that left (turn on red) no longer waits.
* **Held calls** (departure from UDOT).  When the phase's call is already on
  as red starts (the last 43/44 before it, in the same segment, is a 43),
  the wait starts at the red: ``held=True``.  UDOT ignores the state at red
  start, so it times these from a later re-registration or drops them; they
  are the long waits (a vehicle arriving in yellow waits the whole red).
  6–21 % of red windows on 315 and 701.  ``summarize_wait_time`` reports
  the UDOT-comparable average without them.
* **Termination** of the phase's own green before the red: the last
  Code 4/5/6 in ``[green, red start]`` → ``gap_out`` / ``max_out`` /
  ``force_off``, else ``unknown`` (UDOT's split of the chart).
* UDOT drops waits over 360 s; the core keeps them and the summary
  counts them in ``n_over_max`` and leaves them out of every average.

Gap Marker Rule (censoring)
---------------------------
Every ``event_code == -1`` row (any parameter) is a gap marker.  No call is
ever paired with a service across one.  A window with a gap marker in
``[open, service]`` is censored — reported, flagged, no delay — and so is a
window whose opening is unknown (a walk with no Code 22 or 21 before it in
the segment, or a red with no green after it).

Package Location: src/atspm/analysis/call_service.py
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .detector_inference import _to_epoch
from .phases import _PHASE_CODES, _build_phase_intervals, _segment_id

_GAP_CODE: int = -1
_CODE_GREEN: int = 1
_CODE_WALK: int = 21
_CODE_PED_CLEAR: int = 22
_CODE_PHASE_CALL: int = 43
_CODE_PHASE_DROP: int = 44
_CODE_PED_CALL: int = 45
_CODE_PED_DET_ON: int = 90
_TERMINATION = {4: "gap_out", 5: "max_out", 6: "force_off"}

DEFAULT_MAX_WAIT_S: float = 360.0

PED_KINDS = ("waited", "in_walk", "uncalled", "censored")
TERMINATIONS = ("gap_out", "max_out", "force_off", "unknown")

PED_DELAY_SCHEMA = [
    "phase",
    "coord_plan",
    "walk_ts",
    "call_ts",
    "delay_s",
    "kind",
    "source",
    "n_presses",
]

WAIT_TIME_SCHEMA = [
    "phase",
    "coord_plan",
    "red_ts",
    "green_ts",
    "call_ts",
    "wait_s",
    "called",
    "held",
    "n_calls",
    "n_drops",
    "termination",
    "censored",
]

PED_SUMMARY_SCHEMA = [
    "time",
    "phase",
    "coord_plan",
    "n_walks",
    "n_called",
    "n_uncalled",
    "n_censored",
    "n_in_walk",
    "n_delays",
    "presses",
    "avg_delay_s",
    "min_delay_s",
    "max_delay_s",
    "total_delay_s",
]

WAIT_SUMMARY_SCHEMA = [
    "time",
    "phase",
    "coord_plan",
    "n_windows",
    "n_censored",
    "n_called",
    "n_held",
    "n_over_max",
    "avg_wait_s",
    "max_wait_s",
    "avg_wait_udot_s",
    "n_gap_out",
    "n_max_out",
    "n_force_off",
    "n_unknown",
    "avg_wait_gap_out_s",
    "avg_wait_max_out_s",
    "avg_wait_force_off_s",
]


# ---------------------------------------------------------------------------
# Shared pairing
# ---------------------------------------------------------------------------


def first_call_in_windows(
    opens: np.ndarray,
    services: np.ndarray,
    calls: np.ndarray,
    drops: Optional[np.ndarray] = None,
    held: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First call in each window ``[open, service]`` (both ends inclusive).

    The caller guarantees each window is gap-free; this is pure array work.
    A call at the service's own timestamp belongs to the window (a call and
    the service it brought up can share a decisecond).

    Args:
        opens: Window starts, epoch seconds.
        services: Window ends (the service), epoch seconds, ``>= opens``.
        calls: Call times, epoch seconds, sorted ascending.
        drops: Call-dropped times, sorted ascending.  When given, a window
            with a drop in ``[open, service)`` restarts at its last drop:
            only calls at or after it count.
        held: Per window, the call was already on at ``open``.  Where true
            and no drop restarts the window, the wait starts at ``open``.

    Returns:
        ``(first_idx, n_calls, n_drops, from_open)``: index into *calls* of
        the first counted call (``-1`` when none or when the window starts
        at ``open``), calls and drops in the window, and whether the wait
        starts at ``open`` (a held call).
    """
    opens = np.asarray(opens, dtype=float)
    services = np.asarray(services, dtype=float)
    calls = np.asarray(calls, dtype=float)
    lo = np.searchsorted(calls, opens, side="left")
    hi = np.searchsorted(calls, services, side="right")
    n_calls = (hi - lo).astype(np.int64)

    start = lo
    n_drops = np.zeros(len(opens), dtype=np.int64)
    restarted = np.zeros(len(opens), dtype=bool)
    if drops is not None and len(drops):
        drops = np.asarray(drops, dtype=float)
        dlo = np.searchsorted(drops, opens, side="left")
        dhi = np.searchsorted(drops, services, side="left")
        n_drops = (dhi - dlo).astype(np.int64)
        restarted = n_drops > 0
        last_drop = drops[np.clip(dhi - 1, 0, len(drops) - 1)]
        start = np.where(restarted, np.searchsorted(calls, last_drop, side="left"), lo)

    from_open = np.zeros(len(opens), dtype=bool) if held is None else (
        np.asarray(held, dtype=bool) & ~restarted)
    first_idx = np.where((start < hi) & ~from_open, start, -1)
    return first_idx, n_calls, n_drops, from_open


def _at(arr: np.ndarray, idx: np.ndarray, fill) -> np.ndarray:
    """``arr[idx]`` where ``0 <= idx < len(arr)``, else *fill*."""
    idx = np.asarray(idx)
    ok = (idx >= 0) & (idx < len(arr))
    if not len(arr):
        return np.full(idx.shape, fill)
    return np.where(ok, arr[np.clip(idx, 0, len(arr) - 1)], fill)


def _gaps(events_df: pd.DataFrame) -> np.ndarray:
    return np.sort(_to_epoch(events_df.loc[events_df["event_code"] == _GAP_CODE, "timestamp"]))


def _clear_of_gaps(starts: np.ndarray, ends: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """True where no gap marker lies in ``[start, end]``."""
    return np.searchsorted(gaps, starts, side="left") == np.searchsorted(gaps, ends, side="right")


def _sorted_rows(events_df: pd.DataFrame, code: int, param: Optional[int] = None) -> pd.DataFrame:
    m = events_df["event_code"] == code
    if param is not None:
        m &= events_df["parameter"] == param
    return events_df.loc[m].sort_values("timestamp", kind="stable").reset_index(drop=True)


def _take(series: pd.Series, idx: np.ndarray) -> pd.Series:
    """``series[idx]`` in the series' dtype, NaT/NaN where ``idx == -1``."""
    return series.reset_index(drop=True).reindex(idx).reset_index(drop=True)


def _plan_at(rows: pd.DataFrame) -> np.ndarray:
    if "coord_plan" not in rows.columns:
        return np.zeros(len(rows))
    return pd.to_numeric(rows["coord_plan"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Ped service pairing (shared with ped_counts and the termination plot)
# ---------------------------------------------------------------------------


def ped_service_calls(events_df: pd.DataFrame) -> pd.DataFrame:
    """Each Begin Walk (Code 21) and whether a Ped Call (Code 45) preceded it.

    A walk is *called* when a Code 45 for its phase was registered after the
    previous walk of that phase (a call in the previous walk's own
    decisecond belongs to that walk) and after the last gap marker before
    it.  Multiple presses between walks count once.

    Args:
        events_df: Events with ``[timestamp, event_code, parameter]`` and
            any other columns, which are carried through.  Gap markers must
            be present.

    Returns:
        The Code 21 rows (all input columns), sorted by phase then time,
        plus a bool ``called`` column.  Empty when there are no walks.
    """
    walks_all = events_df.loc[events_df["event_code"] == _CODE_WALK]
    if walks_all.empty:
        return walks_all.assign(called=pd.Series(dtype=bool))
    gaps = _gaps(events_df)
    out = []
    for ph in sorted(pd.unique(walks_all["parameter"])):
        w = _sorted_rows(walks_all, _CODE_WALK, ph)
        W = _to_epoch(w["timestamp"])
        C = _to_epoch(_sorted_rows(events_df, _CODE_PED_CALL, ph)["timestamp"])
        prev = np.concatenate([[-np.inf], np.nextafter(W[:-1], np.inf)])
        last_gap = _at(gaps, np.searchsorted(gaps, W, side="right") - 1, -np.inf)
        opens = np.maximum(prev, last_gap)
        _, n_calls, _, _ = first_call_in_windows(opens, W, C)
        out.append(w.assign(called=n_calls > 0))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Pedestrian delay
# ---------------------------------------------------------------------------


def _empty_ped() -> pd.DataFrame:
    return pd.DataFrame(columns=PED_DELAY_SCHEMA)


def ped_delay(
    events_df: pd.DataFrame,
    phases: Optional[Iterable[int]] = None,
    ped_detectors: Optional[Dict[int, int]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Pedestrian delay: first press since the last walk's clearance → walk.

    Args:
        events_df: Flat events with ``[timestamp, event_code, parameter]``
            and optionally ``coord_plan``.  Timestamps may be UTC epoch
            floats or tz-aware Timestamps.  Gap markers must be present.
        phases: Phases to report; default every phase with a Code 21.
        ped_detectors: ``{ped detector (Code 90 parameter): phase}``.
            Unmapped detectors are taken as their own phase number.

    Returns:
        ``(delays, reason)``.  *reason* is ``None`` when the measure could
        be computed, else a sentence saying which codes are missing (and
        *delays* is empty with its schema).

        *delays*, sorted by phase then walk, columns
        :data:`PED_DELAY_SCHEMA`::

            phase       int
            coord_plan  float   – plan at the walk
            walk_ts     input dtype – Begin Walk (Code 21)
            call_ts     input dtype – first press of the window (NaT/NaN
                                      for uncalled and censored walks)
            delay_s     float s – walk_ts − call_ts; 0 for in_walk
            kind        str – waited | in_walk | uncalled | censored
            source      int – call code used for the phase (90 or 45)
            n_presses   int – calls of that code in the window

        Each walk gives one row (``waited``, ``uncalled`` or ``censored``)
        and, when pressed during its walk interval, one ``in_walk`` row.
    """
    if events_df.empty or not (events_df["event_code"] == _CODE_WALK).any():
        return _empty_ped(), "no Begin Walk events (Code 21): the controller does not log pedestrian phases"
    if not (events_df["event_code"] == _CODE_PED_CLEAR).any():
        return _empty_ped(), "no Begin Ped Clearance events (Code 22): the walk interval cannot be bounded"
    is_det = events_df["event_code"] == _CODE_PED_DET_ON
    if not (is_det | (events_df["event_code"] == _CODE_PED_CALL)).any():
        return _empty_ped(), "no pedestrian call events (Code 45 or 90): delay cannot be timed"

    det = events_df.loc[is_det]
    if ped_detectors and not det.empty:
        det = det.assign(parameter=det["parameter"].replace(
            {int(k): int(v) for k, v in ped_detectors.items()}).astype(np.int64))
    calls_by_code = {
        _CODE_PED_DET_ON: det,
        _CODE_PED_CALL: events_df.loc[events_df["event_code"] == _CODE_PED_CALL],
    }
    gaps = _gaps(events_df)
    walk_phases = sorted(pd.unique(events_df.loc[events_df["event_code"] == _CODE_WALK, "parameter"]))
    if phases is not None:
        wanted = {int(p) for p in phases}
        walk_phases = [p for p in walk_phases if int(p) in wanted]

    out = []
    for ph in walk_phases:
        w = _sorted_rows(events_df, _CODE_WALK, ph)
        W = _to_epoch(w["timestamp"])
        CL = np.sort(_to_epoch(events_df.loc[(events_df["event_code"] == _CODE_PED_CLEAR)
                                             & (events_df["parameter"] == ph), "timestamp"]))
        source = _CODE_PED_DET_ON if (calls_by_code[_CODE_PED_DET_ON]["parameter"] == ph).any() \
            else _CODE_PED_CALL
        c = _sorted_rows(calls_by_code[source], source, ph)
        C = _to_epoch(c["timestamp"])
        n = len(W)

        # Window opens at the last clearance before the walk, or just after
        # the previous walk when its clearance was not logged.
        last_clear = _at(CL, np.searchsorted(CL, W, side="left") - 1, -np.inf)
        prev_walk = np.concatenate([[-np.inf], np.nextafter(W[:-1], np.inf)])
        opens = np.maximum(last_clear, prev_walk)
        censored = ~np.isfinite(opens)
        censored[~censored] = ~_clear_of_gaps(opens[~censored], W[~censored], gaps)
        safe_open = np.where(censored, W, opens)
        idx, n_calls, _, _ = first_call_in_windows(safe_open, W, C)
        idx = np.where(censored, -1, idx)
        called = idx >= 0
        kind = np.where(censored, "censored", np.where(called, "waited", "uncalled"))

        walk_rows = pd.DataFrame({
            "phase": int(ph),
            "coord_plan": _plan_at(w),
            "walk_ts": w["timestamp"].reset_index(drop=True),
            "call_ts": _take(c["timestamp"], idx),
            "delay_s": W - _at(C, idx, np.nan),
            "kind": kind,
            "source": int(source),
            "n_presses": np.where(censored, 0, n_calls).astype(np.int64),
        })

        # Presses during the walk interval: after the walk's decisecond,
        # before its clearance.  A walk whose clearance was not logged
        # before the next walk has no known end and gets no in-walk row.
        ends = _at(CL, np.searchsorted(CL, W, side="right"), np.inf)
        next_walk = np.concatenate([W[1:], [np.inf]])
        bounded = ends < next_walk
        bounded[bounded] = _clear_of_gaps(W[bounded], ends[bounded], gaps)
        iw_open = np.nextafter(W, np.inf)
        iw_end = np.where(bounded, np.nextafter(ends, -np.inf), iw_open)
        iw_idx, iw_n, _, _ = first_call_in_windows(iw_open, iw_end, C)
        has_iw = bounded & (iw_idx >= 0)
        sel = np.flatnonzero(has_iw)
        frames = [walk_rows]
        if len(sel):
            frames.append(pd.DataFrame({
                "phase": int(ph),
                "coord_plan": walk_rows["coord_plan"].to_numpy()[sel],
                "walk_ts": walk_rows["walk_ts"].iloc[sel].reset_index(drop=True),
                "call_ts": _take(c["timestamp"], iw_idx[sel]),
                "delay_s": 0.0,
                "kind": "in_walk",
                "source": int(source),
                "n_presses": iw_n[sel].astype(np.int64),
            }))
        ph_df = pd.concat(frames, ignore_index=True)
        ph_df["_k"] = np.concatenate([np.arange(n) * 2, sel * 2 + 1])
        out.append(ph_df.sort_values("_k", kind="stable").drop(columns="_k"))

    if not out:
        return _empty_ped(), None
    res = pd.concat(out, ignore_index=True)[PED_DELAY_SCHEMA]
    res["delay_s"] = res["delay_s"].astype(float).round(2)
    return res, None


# ---------------------------------------------------------------------------
# Wait time
# ---------------------------------------------------------------------------


def _phase_intervals(events_df: pd.DataFrame, phase: int) -> pd.DataFrame:
    sig = events_df.loc[events_df["event_code"].isin(_PHASE_CODES) & (events_df["parameter"] == phase)]
    gaps = events_df.loc[events_df["event_code"] == _GAP_CODE]
    ph_df = pd.concat([sig, gaps]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    ph_df["_seg"] = _segment_id(ph_df)
    ph_df = ph_df.loc[ph_df["event_code"] != _GAP_CODE]
    if ph_df.empty:
        return pd.DataFrame()
    return _build_phase_intervals(ph_df, include_no_clearance=False)


def wait_time(
    events_df: pd.DataFrame,
    phases: Optional[Iterable[int]] = None,
    dropping: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Vehicle wait time: first phase call in the phase's red → its green.

    Args:
        events_df: Flat events with ``[timestamp, event_code, parameter]``
            and optionally ``coord_plan``.  Needs Codes 1, 4/5/6, 8–12 and
            43/44.  Timestamps may be UTC epoch floats or tz-aware
            Timestamps.  Gap markers must be present.
        phases: Phases to report; default every phase with a green.
        dropping: Phases that use UDOT's dropping algorithm (stop-bar
            presence detection): the wait restarts at each dropped call.

    Returns:
        One row per red of each phase (from the end of its clearance),
        sorted by phase then red, columns :data:`WAIT_TIME_SCHEMA`::

            phase        int
            coord_plan   float   – plan at the green (at the previous green
                                   when there is none after the red)
            red_ts       input dtype – red start (Code 11, else 9)
            green_ts     input dtype – next green (Code 1); NaT/NaN when none
            call_ts      input dtype – start of the wait: the counted Code 43,
                                       or red_ts when held; NaT/NaN when uncalled
            wait_s       float s – green_ts − call_ts (NaN when uncalled or
                                   censored)
            called       bool – a wait was timed
            held         bool – the call was on at red start
            n_calls      int  – Code 43s in [red_ts, green_ts]
            n_drops      int  – Code 44s in [red_ts, green_ts)
            termination  str  – how the green before this red ended
                                (gap_out | max_out | force_off | unknown)
            censored     bool – gap marker in the window, or no next green

        Empty with its schema when no phase has a red.
    """
    if events_df.empty:
        return pd.DataFrame(columns=WAIT_TIME_SCHEMA)
    gaps = _gaps(events_df)
    drop_set = {int(p) for p in (dropping or [])}
    green_phases = sorted(pd.unique(events_df.loc[events_df["event_code"] == _CODE_GREEN, "parameter"]))
    if phases is not None:
        wanted = {int(p) for p in phases}
        green_phases = [p for p in green_phases if int(p) in wanted]

    out = []
    for ph in green_phases:
        iv = _phase_intervals(events_df, ph)
        if iv.empty:
            continue
        iv = iv.sort_values("clear_end_ts", kind="stable").reset_index(drop=True)
        G0 = _to_epoch(iv["green_ts"])
        R = _to_epoch(iv["clear_end_ts"])

        greens = _sorted_rows(events_df, _CODE_GREEN, ph)
        GR = _to_epoch(greens["timestamp"])
        j = np.searchsorted(GR, R, side="left")
        has_next = j < len(GR)
        jj = np.where(has_next, j, -1)
        plans = _plan_at(greens)
        G1 = _at(GR, jj, np.nan)
        censored = ~has_next
        censored[has_next] = ~_clear_of_gaps(R[has_next], G1[has_next], gaps)

        calls = _sorted_rows(events_df, _CODE_PHASE_CALL, ph)
        C = _to_epoch(calls["timestamp"])
        D = _to_epoch(_sorted_rows(events_df, _CODE_PHASE_DROP, ph)["timestamp"])

        # Call state at red start: the last 43/44 before it, same segment.
        st = events_df.loc[events_df["event_code"].isin([_CODE_PHASE_CALL, _CODE_PHASE_DROP])
                           & (events_df["parameter"] == ph)]
        ST = _to_epoch(st["timestamp"])
        o = np.argsort(ST, kind="stable")
        ST, SC = ST[o], st["event_code"].to_numpy()[o]
        k = np.searchsorted(ST, R, side="left") - 1
        held = _at(SC, k, 0) == _CODE_PHASE_CALL
        held &= _clear_of_gaps(_at(ST, k, np.inf), R, gaps)

        svc = np.where(censored, R, G1)
        idx, n_calls, _, from_open = first_call_in_windows(
            R, svc, C, D if int(ph) in drop_set else None, held)
        n_drops = np.searchsorted(D, svc, side="left") - np.searchsorted(D, R, side="left")
        called = ~censored & ((idx >= 0) | from_open)
        start = np.where(from_open, R, _at(C, idx, np.nan))
        wait = np.where(called, G1 - start, np.nan)

        # Termination of the green that ended at this red.
        term = events_df.loc[events_df["event_code"].isin(list(_TERMINATION))
                             & (events_df["parameter"] == ph)]
        TT = _to_epoch(term["timestamp"])
        to = np.argsort(TT, kind="stable")
        TT, TC = TT[to], term["event_code"].to_numpy()[to]
        t = np.searchsorted(TT, R, side="right") - 1
        code = np.where(_at(TT, t, -np.inf) >= G0, _at(TC, t, 0), 0)
        termination = pd.Series(code).map(_TERMINATION).fillna("unknown").to_numpy()

        red_rows = iv["clear_end_ts"].reset_index(drop=True)
        call_ts = _take(calls["timestamp"], np.where(called & ~from_open, idx, -1))
        call_ts = call_ts.where(~from_open | ~called, red_rows)
        out.append(pd.DataFrame({
            "phase": int(ph),
            "coord_plan": np.where(has_next, _at(plans, jj, 0.0),
                                   _at(plans, np.searchsorted(GR, G0, side="left"), 0.0)),
            "red_ts": red_rows,
            "green_ts": _take(greens["timestamp"], jj),
            "call_ts": call_ts,
            "wait_s": np.round(wait, 2),
            "called": called,
            "held": held & ~censored,
            "n_calls": np.where(censored, 0, n_calls).astype(np.int64),
            "n_drops": np.where(censored, 0, n_drops).astype(np.int64),
            "termination": termination,
            "censored": censored,
        }))

    if not out:
        return pd.DataFrame(columns=WAIT_TIME_SCHEMA)
    return pd.concat(out, ignore_index=True)[WAIT_TIME_SCHEMA]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _bin_time(ts: pd.Series, bin_len: Optional[int]) -> pd.Series:
    t = ts if pd.api.types.is_datetime64_any_dtype(ts) else pd.to_datetime(
        ts.astype(float), unit="s", utc=True)
    return t.dt.floor(f"{bin_len}min") if bin_len is not None else t


def _ratio(num: pd.Series, den: pd.Series) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _group(df: pd.DataFrame, bin_len: Optional[int], **aggs) -> pd.DataFrame:
    keys = ["time", "phase", "coord_plan"] if bin_len is not None else ["phase", "coord_plan"]
    if bin_len is None:
        aggs = {"time": ("time", "min"), **aggs}
    return df.groupby(keys, sort=False).agg(**aggs).reset_index()


def summarize_ped_delay(ped_df: pd.DataFrame, bin_len: Optional[int] = 60) -> pd.DataFrame:
    """Aggregate :func:`ped_delay` rows into time bins or per plan.

    A row is placed by its ``walk_ts``.  Delay statistics cover ``waited``
    and ``in_walk`` rows (UDOT's "cycles with ped requests").

    Args:
        ped_df: First element of :func:`ped_delay`.
        bin_len: Bin width in minutes; ``None`` gives one row per
            (phase, coord_plan), ``time`` its first walk.

    Returns:
        Columns :data:`PED_SUMMARY_SCHEMA`, sorted by ``phase, time``::

            n_walks       int – uncensored walks (waited + uncalled)
            n_called      int – waited walks
            n_uncalled    int
            n_censored    int
            n_in_walk     int – presses served by the walk under way
            n_delays      int – n_called + n_in_walk
            presses       int – n_presses summed over all rows
            avg_delay_s, min_delay_s, max_delay_s, total_delay_s   float
                          (NaN when n_delays is 0; total 0)
    """
    if ped_df is None or ped_df.empty:
        return pd.DataFrame(columns=PED_SUMMARY_SCHEMA)
    df = ped_df.copy()
    df["time"] = _bin_time(df["walk_ts"], bin_len)
    for k in PED_KINDS:
        df[f"_{k}"] = (df["kind"] == k).astype(np.int64)
    df["_d"] = df["delay_s"].where(df["kind"].isin(["waited", "in_walk"]))
    agg = _group(
        df, bin_len,
        n_called=("_waited", "sum"),
        n_uncalled=("_uncalled", "sum"),
        n_censored=("_censored", "sum"),
        n_in_walk=("_in_walk", "sum"),
        presses=("n_presses", "sum"),
        avg_delay_s=("_d", "mean"),
        min_delay_s=("_d", "min"),
        max_delay_s=("_d", "max"),
        total_delay_s=("_d", "sum"),
    )
    agg["n_walks"] = agg["n_called"] + agg["n_uncalled"]
    agg["n_delays"] = agg["n_called"] + agg["n_in_walk"]
    for col in ("n_walks", "n_called", "n_uncalled", "n_censored", "n_in_walk", "n_delays", "presses"):
        agg[col] = agg[col].astype(int)
    agg["phase"] = agg["phase"].astype(int)
    agg = agg.sort_values(["phase", "time"], kind="stable").reset_index(drop=True)
    return agg[PED_SUMMARY_SCHEMA].round({
        "avg_delay_s": 2, "min_delay_s": 2, "max_delay_s": 2, "total_delay_s": 2,
    })


def summarize_wait_time(
    wait_df: pd.DataFrame,
    bin_len: Optional[int] = 15,
    max_wait: Optional[float] = DEFAULT_MAX_WAIT_S,
) -> pd.DataFrame:
    """Aggregate :func:`wait_time` rows into time bins or per plan.

    A row is placed by its ``green_ts`` (``red_ts`` when censored without a
    green).  Waits over *max_wait* are counted in ``n_over_max`` and left
    out of every average, as UDOT drops them.

    Args:
        wait_df: Output of :func:`wait_time`.
        bin_len: Bin width in minutes; ``None`` gives one row per
            (phase, coord_plan), ``time`` its first green.
        max_wait: Cap in seconds; ``None`` keeps every wait.

    Returns:
        Columns :data:`WAIT_SUMMARY_SCHEMA`, sorted by ``phase, time``::

            n_windows        int – uncensored reds
            n_censored       int
            n_called         int – timed waits within the cap
            n_held           int – of those, held at red start
            n_over_max       int
            avg_wait_s       float – mean over n_called
            max_wait_s       float
            avg_wait_udot_s  float – mean excluding held waits (UDOT-comparable)
            n_gap_out, n_max_out, n_force_off, n_unknown   int – n_called by
                             the termination of the green before the red
            avg_wait_gap_out_s, avg_wait_max_out_s, avg_wait_force_off_s  float
            (averages NaN when their count is 0)
    """
    if wait_df is None or wait_df.empty:
        return pd.DataFrame(columns=WAIT_SUMMARY_SCHEMA)
    df = wait_df.copy()
    when = df["green_ts"].where(df["green_ts"].notna(), df["red_ts"])
    df["time"] = _bin_time(when, bin_len)
    cens = df["censored"].astype(bool)
    timed = df["called"].astype(bool) & ~cens
    over = timed & (df["wait_s"] > max_wait) if max_wait is not None else timed & False
    ok = timed & ~over
    held = ok & df["held"].astype(bool)
    df["_win"] = (~cens).astype(np.int64)
    df["_cens"] = cens.astype(np.int64)
    df["_ok"] = ok.astype(np.int64)
    df["_held"] = held.astype(np.int64)
    df["_over"] = over.astype(np.int64)
    df["_w"] = df["wait_s"].where(ok)
    df["_wu"] = df["wait_s"].where(ok & ~held)
    aggs = dict(
        n_windows=("_win", "sum"),
        n_censored=("_cens", "sum"),
        n_called=("_ok", "sum"),
        n_held=("_held", "sum"),
        n_over_max=("_over", "sum"),
        avg_wait_s=("_w", "mean"),
        max_wait_s=("_w", "max"),
        avg_wait_udot_s=("_wu", "mean"),
    )
    for t in TERMINATIONS:
        m = ok & (df["termination"] == t)
        df[f"_n_{t}"] = m.astype(np.int64)
        df[f"_w_{t}"] = df["wait_s"].where(m)
        aggs[f"n_{t}"] = (f"_n_{t}", "sum")
        if t != "unknown":
            aggs[f"avg_wait_{t}_s"] = (f"_w_{t}", "mean")
    agg = _group(df, bin_len, **aggs)
    for col in ("n_windows", "n_censored", "n_called", "n_held", "n_over_max",
                "n_gap_out", "n_max_out", "n_force_off", "n_unknown"):
        agg[col] = agg[col].astype(int)
    agg["phase"] = agg["phase"].astype(int)
    agg = agg.sort_values(["phase", "time"], kind="stable").reset_index(drop=True)
    return agg[WAIT_SUMMARY_SCHEMA].round({
        "avg_wait_s": 2, "max_wait_s": 2, "avg_wait_udot_s": 2,
        "avg_wait_gap_out_s": 2, "avg_wait_max_out_s": 2, "avg_wait_force_off_s": 2,
    })
