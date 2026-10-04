"""
Timing and Actuation (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

UDOT's visual detector check: per-phase rows of green / yellow / red, the
phase call, pedestrian service, and the phase's detector on-intervals
grouped by role, plus preempt rows on top.  This module turns raw events
into a tidy interval table and a row layout; the figure is built by
:func:`atspm.plotting.timing_actuation.plot_timing_actuation`.

Why not ``_build_phase_intervals`` / ``_reconstruct_intervals`` directly
-----------------------------------------------------------------------
Both drop what they can't see whole: a green or detector actuation that
started before the fetched data, or is still running at its end.  A
timing plot must draw exactly those (a detector stuck on for the whole
window is the point of the plot).  :func:`timing_actuation_intervals`
therefore draws each state from one event to the next, and fills the
leading state from the first event's code (an 81 means the detector was
on), but only when no gap marker lies before it.  For intervals both
helpers do emit, the boundaries are identical (tested golden).

Gap markers
-----------
``event_code = -1`` ends every open state, and nothing is inferred after
one: an interval that a gap cuts short is flagged ``open_end``, and the
state between a gap and the next event is left blank.

Package Location: src/atspm/analysis/timing_actuation.py
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.timezone import resolve_pytz
from .cycles import _parse_ring_groups
from .detector_roles import ROLE_SCHEMA

_GAP_CODE = -1

# ---------------------------------------------------------------------------
# Event codes per row kind
# ---------------------------------------------------------------------------
# Each kind maps code -> state entered at that event (None = no bar), code ->
# state the row was in *before* that event (for the leading interval), and a
# same-timestamp sort priority (lower first).

_KINDS: Dict[str, Dict[str, Mapping[int, Any]]] = {
    # 1 Begin Green, 8 Begin Yellow, 9 End Yellow, 10 Begin Red Clearance,
    # 11 End Red Clearance, 12 Phase Inactive.  At one timestamp the end of
    # a cycle sorts before the start of the next: 201 logs 11 and the next
    # 1 in the same decisecond (zero-length greens and yellows don't occur).
    "phase": {
        "enter": {1: "G", 8: "Y", 9: "R", 10: "RC", 11: "R", 12: "R"},
        "before": {1: "R", 8: "G", 9: "Y", 10: "Y", 11: "RC"},
        "order": {8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 1: 5},
    },
    # 43 Phase Call Registered, 44 Phase Call Dropped.
    "call": {
        "enter": {43: "call", 44: None},
        "before": {44: "call"},
        "order": {44: 0, 43: 1},
    },
    # 21 Begin Walk, 22 Begin Ped Clearance, 23 Begin Solid Don't Walk.
    "ped": {
        "enter": {21: "walk", 22: "fdw", 23: None},
        "before": {22: "walk", 23: "fdw"},
        "order": {21: 0, 22: 1, 23: 2},
    },
    # 102 Preempt Call Input On, 104 Preempt Call Input Off.
    "preempt_call": {
        "enter": {102: "call", 104: None},
        "before": {104: "call"},
        "order": {104: 0, 102: 1},
    },
    # 105 Entry Started, 106 Begin Track Clearance, 107 Begin Dwell,
    # 111 Begin Exit Interval.  The exit interval has no end code, so 111
    # closes the bar and is drawn as a mark instead.
    "preempt": {
        "enter": {105: "entry", 106: "track", 107: "dwell", 111: None},
        "before": {},
        "order": {105: 0, 106: 1, 107: 2, 111: 3},
    },
    # 81 Detector Off, 82 Detector On.  Off sorts first at one timestamp,
    # as in _reconstruct_intervals (stable sort on the ORDER BY code feed).
    "detector": {
        "enter": {82: "on", 81: None},
        "before": {81: "on"},
        "order": {81: 0, 82: 1},
    },
}

_CODE_PED_CALL = 45
_CODE_PREEMPT_EXIT = 111

#: Every event code the timing plot reads (gap markers included).
TIMING_CODES: tuple = tuple(sorted(
    {_GAP_CODE, _CODE_PED_CALL}
    | {c for spec in _KINDS.values() for c in spec["enter"]}
))

INTERVAL_SCHEMA = [
    "kind", "param", "state", "start_ts", "end_ts", "open_start", "open_end",
]
_INTERVAL_DTYPES = {
    "kind": "str", "param": "int64", "state": "str", "start_ts": "float64",
    "end_ts": "float64", "open_start": "bool", "open_end": "bool",
}

MARK_SCHEMA = ["kind", "param", "label", "ts"]
_MARK_DTYPES = {"kind": "str", "param": "int64", "label": "str", "ts": "float64"}

ROW_SCHEMA = ["row", "block", "kind", "param", "phase", "role", "label"]
_ROW_DTYPES = {
    "row": "int64", "block": "str", "kind": "str", "param": "int64",
    "phase": "Int64", "role": "str", "label": "str",
}

# Detector roles drawn under their phase, in display order.
_PHASE_ROLE_ORDER = ("arrival", "occupancy", "stop_bar", "pairs")
_ROLE_ABBR = {
    "arrival": "Arr", "occupancy": "Occ", "stop_bar": "Stop", "pairs": "Pair",
    "tm": "TM", "watchdog": "WD", "unconfigured": "Det",
}

BLOCK_PREEMPT = "Preempt"
BLOCK_OTHER = "Other"
BLOCK_UNCONFIGURED = "Unconfigured"

FINDING_LINK_HALF_WIDTH_S = 600.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty(schema: List[str], dtypes: Dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(columns=schema).astype(dtypes)


def _epoch(ts: pd.Series) -> np.ndarray:
    """UTC epoch seconds from float or datetime timestamps."""
    if pd.api.types.is_datetime64_any_dtype(ts):
        t = pd.to_datetime(ts, utc=True)
        return (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy(float)
    return ts.to_numpy(dtype=float)


def _kind_intervals(
    kind: str,
    t: np.ndarray,
    code: np.ndarray,
    par: np.ndarray,
    gaps: np.ndarray,
    data_start: float,
    data_end: float,
) -> pd.DataFrame:
    """State intervals for one row kind, all parameters at once.

    Each event opens the state it enters and runs to the next event of the
    same parameter, the next gap marker, or ``data_end``, whichever is
    first.  The leading state (``data_start`` → first event) is filled from
    the first event's code when no gap marker precedes it.
    """
    spec = _KINDS[kind]
    enter, before, order = spec["enter"], spec["before"], spec["order"]

    mask = np.isin(code, list(enter))
    if kind == "phase":
        mask &= par > 0
    if not mask.any():
        return _empty(INTERVAL_SCHEMA, _INTERVAL_DTYPES)

    ev = pd.DataFrame({"t": t[mask], "code": code[mask], "par": par[mask]})
    ev["_o"] = ev["code"].map(order)
    ev = ev.sort_values(["par", "t", "_o"], kind="mergesort").reset_index(drop=True)

    if kind == "phase":
        # Code 12 mid-green followed by Code 8 is a delayed FYA permissive
        # start, not a termination: the green continues (as in
        # _build_phase_intervals).  Drop each 12 whose next non-12 event of
        # the same phase is an 8.
        non12 = ev["code"].where(ev["code"] != 12)
        next_non12 = non12.groupby(ev["par"]).bfill()
        ev = ev.loc[~((ev["code"] == 12) & (next_non12 == 8))].reset_index(drop=True)

    p = ev["par"].to_numpy()
    ts = ev["t"].to_numpy(float)
    same_next = np.append(p[1:] == p[:-1], False)
    next_t = np.where(same_next, np.append(ts[1:], np.nan), np.inf)

    # First gap marker strictly after each event; a gap at the same instant
    # sorts before the event (ORDER BY timestamp, event_code), so it doesn't
    # cut the state the event opens.
    next_gap = np.append(gaps, np.inf)[np.searchsorted(gaps, ts, side="right")]

    state = ev["code"].map(enter).to_numpy(dtype=object)

    if kind == "phase":
        # A Begin Green while already green means the green's end was never
        # logged (a silent hole, or a clock update: 315 2025-12-15 08:44,
        # 313 2026-08-05 15:36).  The second green is the real one; the span
        # before it is unknown, so it stays blank rather than merging into a
        # green that never ran.  _build_phase_intervals drops it the same way.
        c = ev["code"].to_numpy()
        lost_end = (c == 1) & same_next & (np.append(c[1:], -999) == 1)
        state[lost_end] = None

    end = np.minimum(np.minimum(next_t, next_gap), data_end)
    # End not logged: cut by a gap marker, or the last event of the param.
    open_end = (next_gap < next_t) | ~same_next

    body = pd.DataFrame({
        "param": p, "state": state, "start_ts": ts, "end_ts": end,
        "open_start": False, "open_end": open_end,
    })

    # Leading interval: first event of each parameter, no gap before it.
    first = np.append(True, p[1:] != p[:-1])
    lead_state = ev["code"].map(before).to_numpy(dtype=object)
    gaps_before = np.searchsorted(gaps, ts, side="right") > 0
    lead_ok = first & ~gaps_before & pd.notna(lead_state) & (ts > data_start)
    lead = pd.DataFrame({
        "param": p[lead_ok], "state": lead_state[lead_ok],
        "start_ts": data_start, "end_ts": ts[lead_ok],
        "open_start": True, "open_end": False,
    })

    out = pd.concat([lead, body], ignore_index=True)
    out = out.loc[pd.notna(out["state"]).to_numpy() & (out["end_ts"] > out["start_ts"]).to_numpy()]
    out = out.sort_values(["param", "start_ts"], kind="mergesort").reset_index(drop=True)
    if out.empty:
        return _empty(INTERVAL_SCHEMA, _INTERVAL_DTYPES)

    # Merge runs of one state that touch end-to-start (repeated ONs, 11→12).
    prev_same = (
        (out["param"].to_numpy()[1:] == out["param"].to_numpy()[:-1])
        & (out["state"].to_numpy()[1:] == out["state"].to_numpy()[:-1])
        & (out["start_ts"].to_numpy()[1:] == out["end_ts"].to_numpy()[:-1])
    )
    run = np.cumsum(np.append(True, ~prev_same))
    out = out.groupby(run, sort=False).agg(
        param=("param", "first"), state=("state", "first"),
        start_ts=("start_ts", "first"), end_ts=("end_ts", "last"),
        open_start=("open_start", "first"), open_end=("open_end", "last"),
    ).reset_index(drop=True)

    out.insert(0, "kind", kind)
    return out[INTERVAL_SCHEMA].astype(_INTERVAL_DTYPES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def timing_actuation_intervals(
    events_df: pd.DataFrame,
    window: Sequence[float],
    data_range: Optional[Sequence[float]] = None,
) -> Dict[str, pd.DataFrame]:
    """Tidy state intervals and point marks for a timing-and-actuation plot.

    Args:
        events_df: Events with ``timestamp`` (UTC epoch seconds, or
            datetimes), ``event_code`` and ``parameter``.  Pass every
            :data:`TIMING_CODES` event over ``data_range``, gap markers
            included.  Fetch with a margin around ``window`` so states that
            began before it (a long green, a held presence call) are seen.
        window: ``(start, end)`` UTC epoch seconds to clip the output to.
        data_range: ``(start, end)`` UTC epoch seconds the events were
            fetched over.  A state still open at ``data_range[1]`` runs to
            it (``open_end``); one in effect at ``data_range[0]`` is filled
            from the first event's code (``open_start``).  Defaults to
            ``window``.

    Returns:
        ``{"intervals": DataFrame, "marks": DataFrame}``.

        ``intervals`` (:data:`INTERVAL_SCHEMA`), clipped to ``window``,
        sorted by kind, param, start_ts::

            kind        str     – phase | call | ped | preempt_call |
                                  preempt | detector
            param       int64   – phase, preempt or detector number
            state       str     – phase: G, Y, RC, R; call: call;
                                  ped: walk, fdw; preempt_call: call;
                                  preempt: entry, track, dwell;
                                  detector: on
            start_ts    float64 – UTC epoch seconds
            end_ts      float64
            open_start  bool    – start inferred, not logged
            open_end    bool    – end not logged (gap marker or data end)

        ``marks`` (:data:`MARK_SCHEMA`), inside ``window``, sorted by ts::

            kind   str     – ped (label 'ped call', Code 45), preempt
                             (label 'exit', Code 111), gap (param -1,
                             label 'hard reset')
            param  int64
            label  str
            ts     float64
    """
    w0, w1 = float(window[0]), float(window[1])
    d0, d1 = (float(data_range[0]), float(data_range[1])) if data_range is not None else (w0, w1)

    if events_df is None or events_df.empty:
        return {"intervals": _empty(INTERVAL_SCHEMA, _INTERVAL_DTYPES),
                "marks": _empty(MARK_SCHEMA, _MARK_DTYPES)}

    t = _epoch(events_df["timestamp"])
    code = events_df["event_code"].to_numpy(dtype=np.int64)
    par = events_df["parameter"].to_numpy(dtype=np.int64)
    in_data = (t >= d0) & (t < d1)
    t, code, par = t[in_data], code[in_data], par[in_data]
    gaps = np.unique(t[code == _GAP_CODE])

    parts = [_kind_intervals(k, t, code, par, gaps, d0, d1) for k in _KINDS]
    iv = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(
        not p.empty for p in parts) else _empty(INTERVAL_SCHEMA, _INTERVAL_DTYPES)

    if not iv.empty:
        iv = iv.loc[(iv["end_ts"] > w0) & (iv["start_ts"] < w1)].copy()
        iv["start_ts"] = iv["start_ts"].clip(lower=w0)
        iv["end_ts"] = iv["end_ts"].clip(upper=w1)
        kind_rank = {k: i for i, k in enumerate(_KINDS)}
        iv = (iv.assign(_k=iv["kind"].map(kind_rank))
              .sort_values(["_k", "param", "start_ts"], kind="mergesort")
              .drop(columns="_k").reset_index(drop=True))
        iv = iv[INTERVAL_SCHEMA].astype(_INTERVAL_DTYPES)

    in_win = (t >= w0) & (t < w1)
    mk = []
    sel = in_win & (code == _CODE_PED_CALL)
    mk.append(pd.DataFrame({"kind": "ped", "param": par[sel], "label": "ped call", "ts": t[sel]}))
    sel = in_win & (code == _CODE_PREEMPT_EXIT)
    mk.append(pd.DataFrame({"kind": "preempt", "param": par[sel], "label": "exit", "ts": t[sel]}))
    g = gaps[(gaps >= w0) & (gaps < w1)]
    mk.append(pd.DataFrame({"kind": "gap", "param": -1, "label": "hard reset", "ts": g}))
    marks = pd.concat([m for m in mk if not m.empty], ignore_index=True) if any(
        not m.empty for m in mk) else _empty(MARK_SCHEMA, _MARK_DTYPES)
    marks = marks.sort_values(["ts", "kind", "param"], kind="mergesort").reset_index(drop=True)

    return {"intervals": iv, "marks": marks[MARK_SCHEMA].astype(_MARK_DTYPES)}


def ring_phase_order(config: Mapping[str, Any]) -> List[int]:
    """Phase display order from the ``RB_R1`` / ``RB_R2`` ring config.

    Barrier group by barrier group, ring 1's phases then ring 2's, each in
    listed order: ``RB_R1='1,2|3,4'``, ``RB_R2='5,6|7,8'`` gives
    ``[1, 2, 5, 6, 3, 4, 7, 8]``.  A phase listed twice keeps its first
    position.

    Args:
        config: Active config dict.

    Returns:
        Ordered phase list; empty when neither ring key parses.
    """
    r1 = _parse_ring_groups(config.get("RB_R1"))
    r2 = _parse_ring_groups(config.get("RB_R2"))
    out: List[int] = []
    for i in range(max(len(r1), len(r2))):
        for grp in (r1[i] if i < len(r1) else [], r2[i] if i < len(r2) else []):
            out.extend(ph for ph in grp if ph not in out)
    return out


def timing_actuation_rows(
    roles: Optional[pd.DataFrame],
    intervals: pd.DataFrame,
    marks: Optional[pd.DataFrame] = None,
    phase_order: Optional[Sequence[int]] = None,
    phases: Optional[Iterable[int]] = None,
    detectors: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Row layout, top to bottom, for a timing-and-actuation plot.

    Blocks, in order:

    1. ``Preempt`` — for each preempt with events in the window, a
       ``preempt_call`` row then a ``preempt`` row.
    2. One block per phase (``"P{n}"``): a ``phase`` row, a ``call`` row, a
       ``ped`` row when the phase has ped intervals or marks, then its
       detectors by role (arrival, occupancy, stop_bar, pairs; by detector
       within a role).  A detector shows once per phase, under its first
       role in that order; it can show under several phases.
    3. ``Other`` — ``tm`` and ``watchdog`` detectors not already shown in a
       phase block.
    4. ``Unconfigured`` — channels with detector intervals that no config
       key names.

    Configured detectors get a row even when silent.

    Filters:

    - ``phases`` keeps only those phase blocks and drops ``Other`` and
      ``Unconfigured``.  Preempt rows always stay.
    - ``detectors`` keeps only those detector rows, and only the phase
      blocks that hold one of them (intersected with ``phases`` when both
      are given).  A requested detector that no config key names goes in
      ``Unconfigured`` even when silent, so a finding link for it still
      shows a row; a configured one shows only under its own blocks.

    Args:
        roles: Role table from
            :func:`~atspm.analysis.detector_roles.parse_detector_roles`, or
            None.
        intervals: ``intervals`` from :func:`timing_actuation_intervals`.
        marks: ``marks`` from :func:`timing_actuation_intervals`.
        phase_order: Phase block order (see :func:`ring_phase_order`).
            A phase gets a block only when it has phase / call / ped
            intervals or phase-keyed detectors; those ``phase_order``
            doesn't list follow it in numeric order.  None or empty:
            numeric order.
        phases: Optional phase filter.
        detectors: Optional detector filter.

    Returns:
        DataFrame (:data:`ROW_SCHEMA`)::

            row    int64  – 0 = top
            block  str    – 'Preempt', 'P{n}', 'Other', 'Unconfigured'
            kind   str    – interval kind the row draws
            param  int64  – the interval ``param`` it draws
            phase  Int64  – NA outside phase blocks
            role   str    – detector role; '' for non-detector rows
            label  str    – y tick label
    """
    roles = roles if roles is not None and not roles.empty else pd.DataFrame(columns=ROLE_SCHEMA)
    marks = marks if marks is not None else _empty(MARK_SCHEMA, _MARK_DTYPES)
    phase_filter = None if phases is None else {int(x) for x in phases}
    det_filter = None if detectors is None else {int(x) for x in detectors}

    rows: List[Dict[str, Any]] = []

    def add(block, kind, param, phase, role, label):
        rows.append({"block": block, "kind": kind, "param": int(param),
                     "phase": phase, "role": role, "label": label})

    # 1. Preempts
    pre = sorted(set(intervals.loc[intervals["kind"].isin(["preempt", "preempt_call"]), "param"].astype(int))
                 | set(marks.loc[marks["kind"] == "preempt", "param"].astype(int)))
    for n in pre:
        add(BLOCK_PREEMPT, "preempt_call", n, pd.NA, "", f"Preempt {n} call")
        add(BLOCK_PREEMPT, "preempt", n, pd.NA, "", f"Preempt {n}")

    # 2. Phase blocks
    role_ph = roles.loc[roles["role"].isin(_PHASE_ROLE_ORDER)].dropna(subset=["phase"])
    seen = set(intervals.loc[intervals["kind"].isin(["phase", "call", "ped"]), "param"].astype(int))
    seen |= set(role_ph["phase"].astype(int))
    order = [int(x) for x in (phase_order or []) if int(x) in seen]
    order += sorted(seen - set(order))
    if phase_filter is not None:
        order = [x for x in order if x in phase_filter]

    ped_phases = set(intervals.loc[intervals["kind"] == "ped", "param"].astype(int)) | set(
        marks.loc[marks["kind"] == "ped", "param"].astype(int))

    shown: set = set()
    for ph in order:
        dets = []
        sub = role_ph.loc[role_ph["phase"].astype(int) == ph]
        placed = set()
        for role in _PHASE_ROLE_ORDER:
            for d in sorted(set(sub.loc[sub["role"] == role, "detector"].astype(int))):
                if d in placed or (det_filter is not None and d not in det_filter):
                    continue
                placed.add(d)
                dets.append((d, role))
        if det_filter is not None and not dets:
            continue
        block = f"P{ph}"
        add(block, "phase", ph, ph, "", f"Phase {ph}")
        add(block, "call", ph, ph, "", f"Call {ph}")
        if ph in ped_phases:
            add(block, "ped", ph, ph, "", f"Ped {ph}")
        for d, role in dets:
            add(block, "detector", d, ph, role, f"{_ROLE_ABBR[role]} {d}")
        shown |= placed

    # 3. Other: tm / watchdog not shown in a phase block
    if phase_filter is None or det_filter is not None:
        other = roles.loc[roles["role"].isin(["tm", "watchdog"])]
        for role in ("tm", "watchdog"):
            sub = other.loc[other["role"] == role].sort_values("detector", kind="mergesort")
            for d, mv in zip(sub["detector"].astype(int), sub["movement"]):
                if d in shown or (det_filter is not None and d not in det_filter):
                    continue
                shown.add(d)
                lab = f"TM {mv} {d}" if role == "tm" and isinstance(mv, str) and mv else f"{_ROLE_ABBR[role]} {d}"
                add(BLOCK_OTHER, "detector", d, pd.NA, role, lab)

    # 4. Unconfigured: active but unnamed, plus requested detectors with no row
    if phase_filter is None or det_filter is not None:
        configured = set(roles["detector"].astype(int)) if not roles.empty else set()
        active = set(intervals.loc[intervals["kind"] == "detector", "param"].astype(int))
        extra = (active - configured)
        if det_filter is not None:
            extra = (extra & det_filter) | (det_filter - configured)
        for d in sorted(extra - shown):
            add(BLOCK_UNCONFIGURED, "detector", d, pd.NA, "unconfigured", f"Det {d}")

    if not rows:
        return _empty(ROW_SCHEMA, _ROW_DTYPES)
    out = pd.DataFrame(rows)
    out.insert(0, "row", np.arange(len(out)))
    return out[ROW_SCHEMA].astype(_ROW_DTYPES)


def finding_plot_windows(
    findings: pd.DataFrame,
    events_df: pd.DataFrame,
    tz: str,
    half_width_s: float = FINDING_LINK_HALF_WIDTH_S,
) -> pd.DataFrame:
    """Timing-plot window and row filter for each detector-health finding.

    A finding with a ``ts`` gets ``ts ± half_width_s``.  One without (a
    day-level rule such as ConfiguredSilent) gets the local clock hour of
    that date with the most detector onsets (Code 82) on its own detector,
    or, when that detector has none or the finding is intersection-level
    (``detector = -1``), on all detectors.  Ties go to the earlier hour.
    No onsets that date: no window (NaN).

    The row filter is the finding's phase when it has one, else its
    detector, else none.

    Args:
        findings: Findings frame (``FINDINGS_SCHEMA`` of
            :mod:`atspm.analysis.detector_health`).
        events_df: Events over the findings' dates (``timestamp``,
            ``event_code``, ``parameter``); only Code 82 is read.
        tz: IANA zone of the intersection; ``findings.date`` is local.
        half_width_s: Half-width of a ``ts``-centred window.

    Returns:
        ``findings`` (same index) with added columns::

            plot_start     float64 – UTC epoch seconds, NaN if none
            plot_end       float64
            plot_phase     Int64   – --phases filter, NA if none
            plot_detector  Int64   – --detectors filter (only when
                                     plot_phase is NA), NA if none
    """
    out = findings.copy()
    n = len(out)
    start = np.full(n, np.nan)
    end = np.full(n, np.nan)

    ts = out["ts"].to_numpy(dtype=float) if n else np.empty(0)
    has_ts = ~np.isnan(ts)
    start[has_ts] = ts[has_ts] - half_width_s
    end[has_ts] = ts[has_ts] + half_width_s

    need = ~has_ts
    if need.any() and events_df is not None and not events_df.empty:
        ev = events_df.loc[events_df["event_code"].to_numpy() == 82]
        if not ev.empty:
            t = _epoch(ev["timestamp"])
            local = pd.to_datetime(t, unit="s", utc=True).tz_convert(resolve_pytz(tz))
            hour0 = t - (local.minute.to_numpy() * 60 + local.second.to_numpy()
                         + local.microsecond.to_numpy() / 1e6)
            hb = pd.DataFrame({
                "date": local.date, "hour0": hour0,
                "detector": ev["parameter"].to_numpy(dtype=np.int64),
            })
            per_det = hb.groupby(["date", "detector", "hour0"]).size().rename("n").reset_index()
            per_det = per_det.sort_values(["date", "detector", "n", "hour0"],
                                          ascending=[True, True, False, True], kind="mergesort")
            best_det = per_det.drop_duplicates(["date", "detector"]).set_index(["date", "detector"])["hour0"]
            per_all = hb.groupby(["date", "hour0"]).size().rename("n").reset_index()
            per_all = per_all.sort_values(["date", "n", "hour0"],
                                          ascending=[True, False, True], kind="mergesort")
            best_all = per_all.drop_duplicates("date").set_index("date")["hour0"]

            dates = out["date"].to_numpy()
            dets = out["detector"].to_numpy(dtype=np.int64)
            key = pd.MultiIndex.from_arrays([dates, dets])
            h = best_det.reindex(key).to_numpy(dtype=float)
            h_all = best_all.reindex(dates).to_numpy(dtype=float)
            h = np.where(np.isnan(h) | (dets == -1), h_all, h)
            start[need] = h[need]
            end[need] = h[need] + 3600.0

    out["plot_start"] = start
    out["plot_end"] = end
    phase = out["phase"].astype("Int64") if n else pd.Series([], dtype="Int64")
    det = out["detector"].astype("Int64") if n else pd.Series([], dtype="Int64")
    out["plot_phase"] = phase
    out["plot_detector"] = det.where(phase.isna() & (det != -1)).astype("Int64")
    return out
