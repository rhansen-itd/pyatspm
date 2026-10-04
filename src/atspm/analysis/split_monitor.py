"""
Split Monitor and Programmed Splits (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

Programmed plan timeline (Codes 131–149)
----------------------------------------
Indiana enumerations: 131 coord pattern (plan), 132 cycle length, 133
offset, 134–149 split of phases 1–16 (phase *p* is code ``133 + p``).
Measured on the corpus (2026-10-04), the controller logs these as a
**change log**, not as one complete row per plan selection:

* a full dump of all nineteen codes at local midnight (DST-aware);
* between dumps, only the values that change.  Plan 11 at 315 logs its
  splits but not its cycle or offset, which it shares with plan 1;
* the plan code and its parameters can be up to ~30 s apart (131 at
  13:30:00.0, the parameters at 13:30:09.9), while the transition runs;
* preemption logs cycle and splits as 0 at entry and restores them at
  exit, with no 131.  Cycle 0 means "not running a split", free or
  preempted.

So the timeline is a state register: each code holds its last logged value.
Events sharing a timestamp apply as one update, and consecutive identical
states merge into one row.  A value of 0 is a real value: a 10 s split or a
0 s offset is kept (the SPMs notebook dropped them).

Gap Marker Rule
---------------
A comms-gap marker (``event_code = -1``, ``parameter = -1``) resets every
value to unknown: changes made during the gap were never logged, so nothing
is carried across it.  Values return as they are next logged, at the latest
with the next midnight dump.  A backward-clock-step fence
(``event_code = -1``, ``parameter = -2``) does **not** reset the register.
The controller kept running and logging, so its parameters are unchanged.
Resetting there would blank most of every day that ``eos_set_time`` sets
the clock.

Split Monitor
-------------
One row per phase service.  The split is green + yellow + red clearance,
as built by ``_build_phase_intervals`` (which drops any interval that
spans a gap marker, ``-2`` fences included, since a clock step inside a
split corrupts its duration).  Each service carries how its green ended
(Code 4/5/6, highest wins, else ``unknown``), whether the phase's walk
began during it (Code 21), and the programmed split in force at its green.
The programmed split is NaN when the cycle is 0 or unknown, or the phase's
split is 0 (not in the plan).

Package Location: src/atspm/analysis/split_monitor.py
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .detector_inference import _to_epoch
from .flow import _window_terminations
from .phases import _PHASE_CODES, _build_phase_intervals, _segment_id

_GAP_CODE: int = -1
_FENCE_PARAM: int = -2          # backward-clock-step fence (ingestion)
_CODE_PLAN: int = 131
_CODE_LAST_SPLIT: int = 149
_CODE_BEGIN_WALK: int = 21
_TERM_CODES = (4, 5, 6)

N_PHASES: int = 16
PLAN_COLUMNS: List[str] = (
    ["plan", "cycle", "offset"] + [f"split_{p}" for p in range(1, N_PHASES + 1)]
)

TIMELINE_SCHEMA: List[str] = ["start", "end"] + PLAN_COLUMNS

CYCLE_SCHEMA: List[str] = [
    "phase",
    "cycle_start",
    "green_ts",
    "yellow_ts",
    "clear_end_ts",
    "green_dur",
    "clear_dur",
    "split_dur",
    "termination",
    "ped_walk",
    "plan",
    "programmed_cycle",
    "programmed_split",
    "split_minus_programmed",
]

STATS_SCHEMA: List[str] = [
    "phase",
    "plan",
    "programmed_split",
    "n_cycles",
    "gap_out_pct",
    "max_out_pct",
    "force_off_pct",
    "unknown_pct",
    "ped_walk_pct",
    "split_mean",
    "split_p50",
    "split_p85",
    "first_green",
    "last_green",
]

TERMINATIONS = ("gap_out", "max_out", "force_off", "unknown")


# ---------------------------------------------------------------------------
# Plan timeline
# ---------------------------------------------------------------------------


def plan_timeline(events_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the programmed plan/cycle/offset/splits in force over time.

    Args:
        events_df: Flat events DataFrame with ``timestamp``, ``event_code``
            and ``parameter``.  Timestamps may be UTC epoch floats or
            tz-aware Timestamps.  Should hold Codes 131–149 and the gap
            markers; other codes are ignored.  To know the state at the start
            of a window, include the plan codes from the previous local
            midnight's dump (26 h before the window is enough).

    Returns:
        One row per run of identical state, sorted by ``start``, columns
        :data:`TIMELINE_SCHEMA`::

            start   input dtype – first event of the state
            end     input dtype – next row's start; for the last row, the
                                  latest timestamp in *events_df*
            plan, cycle, offset, split_1 … split_16
                    Int64       – NA where not logged since the segment began

        A comms-gap marker starts an all-NA row.  Empty frame with this
        schema when *events_df* holds no plan codes.
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=TIMELINE_SCHEMA)

    code = events_df["event_code"].to_numpy()
    par = events_df["parameter"].to_numpy()
    is_plan = (code >= _CODE_PLAN) & (code <= _CODE_LAST_SPLIT)
    is_gap = (code == _GAP_CODE) & (par != _FENCE_PARAM)
    sel = events_df.loc[is_plan | is_gap, ["timestamp", "event_code", "parameter"]]
    if not is_plan.any():
        return pd.DataFrame(columns=TIMELINE_SCHEMA)

    t = _to_epoch(sel["timestamp"])
    c = sel["event_code"].to_numpy()
    # Time order; a gap marker sorts ahead of a plan code at the same time.
    order = np.lexsort((c != _GAP_CODE, t))
    sel = sel.iloc[order]
    t, c = t[order], c[order]
    v = sel["parameter"].to_numpy()
    gap = c == _GAP_CODE

    # One batch per distinct timestamp of plan codes; each gap marker is a
    # batch of its own.
    new_batch = np.ones(len(t), dtype=bool)
    new_batch[1:] = (t[1:] != t[:-1]) | gap[1:] | gap[:-1]
    batch = np.cumsum(new_batch) - 1
    n_batch = int(batch[-1]) + 1
    first_row = np.flatnonzero(new_batch)
    batch_gap = gap[first_row]

    wide = np.full((n_batch, len(PLAN_COLUMNS)), np.nan)
    pc = ~gap
    # Later rows overwrite earlier ones, so a code logged twice in one
    # batch keeps its last value.
    wide[batch[pc], c[pc] - _CODE_PLAN] = v[pc]

    state = pd.DataFrame(wide, columns=PLAN_COLUMNS)
    seg = np.cumsum(batch_gap)
    state = state.groupby(seg).ffill()

    a = state.to_numpy()
    prev = np.vstack([np.full((1, a.shape[1]), np.inf), a[:-1]])
    same = ((a == prev) | (np.isnan(a) & np.isnan(prev))).all(axis=1)
    keep = ~same
    keep[0] = True

    kept_rows = first_row[keep]
    out = state.loc[keep].reset_index(drop=True).astype("Int64")
    starts = sel["timestamp"].iloc[kept_rows].reset_index(drop=True)
    data_end = events_df["timestamp"].max()
    ends = pd.concat(
        [starts.iloc[1:], pd.Series([data_end], dtype=starts.dtype)],
        ignore_index=True,
    )
    out.insert(0, "end", ends)
    out.insert(0, "start", starts)
    return out[TIMELINE_SCHEMA]


def programmed_at(
    timeline: pd.DataFrame, ts: pd.Series, phase: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Look up the timeline row in force at each timestamp.

    Args:
        timeline: Output of :func:`plan_timeline`.
        ts: Timestamps (epoch floats or tz-aware), any order.
        phase: Optional phase per timestamp; when given, the result gains a
            ``split`` column with that phase's split.

    Returns:
        Frame aligned with *ts* (same length, positional) with ``plan``,
        ``cycle``, ``offset`` (Int64) and optionally ``split`` (Int64).
        NA where *ts* falls before the first row or at/after the last
        row's ``end``.
    """
    n = len(ts)
    cols = ["plan", "cycle", "offset"]
    if timeline is None or timeline.empty or n == 0:
        out = pd.DataFrame({k: pd.array([pd.NA] * n, dtype="Int64") for k in cols})
        if phase is not None:
            out["split"] = pd.array([pd.NA] * n, dtype="Int64")
        return out

    s = _to_epoch(timeline["start"])
    e = _to_epoch(timeline["end"])
    x = _to_epoch(pd.Series(ts).reset_index(drop=True))
    k = np.searchsorted(s, x, side="right") - 1
    kk = np.clip(k, 0, len(s) - 1)
    # The last row's end is the data end; a timestamp equal to it is still
    # inside, every other row is half-open.
    last = kk == len(s) - 1
    ok = (k >= 0) & ((x < e[kk]) | (last & (x <= e[kk])))

    out = {}
    for col in cols:
        vals = timeline[col].astype("Float64").to_numpy(dtype=float, na_value=np.nan)[kk]
        out[col] = pd.array(np.where(ok, vals, np.nan), dtype="Float64").astype("Int64")
    res = pd.DataFrame(out)
    if phase is not None:
        ph = np.asarray(phase, dtype=int)
        split_mat = timeline[[f"split_{p}" for p in range(1, N_PHASES + 1)]] \
            .astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        in_range = (ph >= 1) & (ph <= N_PHASES)
        vals = np.full(n, np.nan)
        vals[in_range] = split_mat[kk[in_range], ph[in_range] - 1]
        res["split"] = pd.array(np.where(ok, vals, np.nan), dtype="Float64").astype("Int64")
    return res


# ---------------------------------------------------------------------------
# Split monitor
# ---------------------------------------------------------------------------


def _walk_in_window(events_df: pd.DataFrame, phase: int,
                    g: np.ndarray, end: np.ndarray) -> np.ndarray:
    """True where a Code 21 for *phase* falls in ``[g, end)``."""
    is_walk = (events_df["event_code"] == _CODE_BEGIN_WALK) & (events_df["parameter"] == phase)
    w = np.sort(_to_epoch(events_df.loc[is_walk, "timestamp"]))
    if len(w) == 0:
        return np.zeros(len(g), dtype=bool)
    return np.searchsorted(w, end, side="left") > np.searchsorted(w, g, side="left")


def split_monitor(
    events_df: pd.DataFrame,
    phases: Optional[Iterable[int]] = None,
    timeline: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-service split, termination and programmed split for each phase.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start]``.  Needs
            Codes 1, 8–12 (phase states), 4/5/6 (terminations), 21 (begin
            walk) and the gap markers.  Timestamps may be UTC epoch floats
            or tz-aware Timestamps.
        phases: Phases to report; all observed phases when ``None``.
        timeline: Output of :func:`plan_timeline`.  When ``None`` it is
            built from *events_df* (which then needs Codes 131–149).

    Returns:
        One row per phase service, sorted by ``phase, green_ts``, columns
        :data:`CYCLE_SCHEMA`::

            phase                 int
            cycle_start           input dtype
            green_ts, yellow_ts, clear_end_ts   input dtype
            green_dur, clear_dur, split_dur     float s
            termination           str   – gap_out / max_out / force_off / unknown
            ped_walk              bool  – Code 21 of the phase in [green, clear end)
            plan                  Int64 – plan in force at green_ts (NA unknown)
            programmed_cycle      Int64
            programmed_split      float s – NaN when cycle is 0/NA or the
                                            phase's split is 0/NA
            split_minus_programmed float s

        Empty frame with this schema when no phase service is found.
    """
    empty = pd.DataFrame(columns=CYCLE_SCHEMA)
    if events_df is None or events_df.empty:
        return empty

    mask = events_df["event_code"].isin(_PHASE_CODES | {_GAP_CODE})
    ph_df = events_df.loc[mask].sort_values("timestamp", kind="stable").reset_index(drop=True)
    if "cycle_start" not in ph_df.columns:
        ph_df["cycle_start"] = np.nan
    ph_df["_seg"] = _segment_id(ph_df)
    ph_df = ph_df.loc[ph_df["event_code"] != _GAP_CODE]
    if ph_df.empty:
        return empty

    iv = _build_phase_intervals(ph_df)
    if iv.empty:
        return empty
    if phases is not None:
        iv = iv.loc[iv["phase"].isin(list(phases))]
        if iv.empty:
            return empty
    iv = iv.sort_values(["phase", "green_ts"], kind="stable").reset_index(drop=True)

    term = np.empty(len(iv), dtype=object)
    walk = np.zeros(len(iv), dtype=bool)
    for phase, idx in iv.groupby("phase").indices.items():
        g = _to_epoch(iv["green_ts"].iloc[idx])
        ce = _to_epoch(iv["clear_end_ts"].iloc[idx])
        term[idx] = _window_terminations(events_df, int(phase), g, ce)
        walk[idx] = _walk_in_window(events_df, int(phase), g, ce)
    term = pd.Series(term).fillna("unknown").to_numpy()

    if timeline is None:
        timeline = plan_timeline(events_df)
    prog = programmed_at(timeline, iv["green_ts"], phase=iv["phase"].to_numpy())

    cyc = prog["cycle"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    spl = prog["split"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(invalid="ignore"):
        prog_split = np.where((cyc > 0) & (spl > 0), spl, np.nan)

    out = pd.DataFrame({
        "phase": iv["phase"].astype(int).to_numpy(),
        "cycle_start": iv["cycle_start"],
        "green_ts": iv["green_ts"],
        "yellow_ts": iv["yellow_ts"],
        "clear_end_ts": iv["clear_end_ts"],
        "green_dur": iv["green_dur"].astype(float),
        "clear_dur": iv["clear_dur"].astype(float),
        "split_dur": iv["split_dur"].astype(float),
        "termination": term,
        "ped_walk": walk,
        "plan": prog["plan"].to_numpy(),
        "programmed_cycle": prog["cycle"].to_numpy(),
        "programmed_split": prog_split,
    })
    out["split_minus_programmed"] = out["split_dur"] - out["programmed_split"]
    return out[CYCLE_SCHEMA].round({
        "green_dur": 2, "clear_dur": 2, "split_dur": 2, "split_minus_programmed": 2,
    })


def split_monitor_stats(
    cycle_df: pd.DataFrame, percentiles: Sequence[float] = (50, 85),
) -> pd.DataFrame:
    """UDOT per-plan split statistics for each phase.

    Grouped by ``phase``, ``plan`` and ``programmed_split``, so a plan whose
    split was edited mid-window reports each version separately, and
    services under an unknown plan form their own (NA) group.

    Args:
        cycle_df: Output of :func:`split_monitor`.
        percentiles: Exactly two split percentiles to report, reported as
            ``split_p50`` and ``split_p85`` under the defaults.  Linear
            interpolation (NumPy's default).

    Returns:
        Columns :data:`STATS_SCHEMA` (percentile column names follow
        *percentiles*), sorted by ``phase, first_green``::

            n_cycles                 int
            gap_out_pct … unknown_pct  float – share of services, 0–1
            ped_walk_pct             float
            split_mean, split_pXX    float s
            first_green, last_green  green_ts of the group's first/last service
    """
    pa, pb = (float(p) for p in percentiles)
    na, nb = f"split_p{pa:g}", f"split_p{pb:g}"
    schema = [na if c == "split_p50" else nb if c == "split_p85" else c for c in STATS_SCHEMA]
    if cycle_df is None or cycle_df.empty:
        return pd.DataFrame(columns=schema)

    df = cycle_df.copy()
    for name in TERMINATIONS:
        df[f"_{name}"] = (df["termination"] == name).astype(float)
    df["_walk"] = df["ped_walk"].astype(float)
    df["_ps"] = df["programmed_split"].astype(float)

    keys = ["phase", "plan", "_ps"]
    agg = df.groupby(keys, dropna=False, sort=False).agg(
        n_cycles=("split_dur", "size"),
        gap_out_pct=("_gap_out", "mean"),
        max_out_pct=("_max_out", "mean"),
        force_off_pct=("_force_off", "mean"),
        unknown_pct=("_unknown", "mean"),
        ped_walk_pct=("_walk", "mean"),
        split_mean=("split_dur", "mean"),
        _pa=("split_dur", lambda s: np.percentile(s, pa)),
        _pb=("split_dur", lambda s: np.percentile(s, pb)),
        first_green=("green_ts", "min"),
        last_green=("green_ts", "max"),
    ).reset_index()
    agg = agg.rename(columns={"_ps": "programmed_split", "_pa": na, "_pb": nb})
    agg["n_cycles"] = agg["n_cycles"].astype(int)
    agg["plan"] = agg["plan"].astype("Int64")
    agg = agg.sort_values(["phase", "first_green"], kind="stable").reset_index(drop=True)
    return agg[schema].round({
        "gap_out_pct": 4, "max_out_pct": 4, "force_off_pct": 4, "unknown_pct": 4,
        "ped_walk_pct": 4, "split_mean": 2, na: 2, nb: 2,
    })
