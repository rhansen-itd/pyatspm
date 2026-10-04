"""
Green Time Utilization (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

UDOT's Green Time Utilization measure (``GreenTimeUtilizationService.cs``,
OpenSourceTransportation/Atspm v5) is a heat map: x is a time bin (minutes),
y is seconds into green in bins of ``bin_s`` (UDOT ``YAxisBinSize``), and
each cell is the mean number of detector actuations per cycle in that
second-of-green bin.  Two step lines overlay it: the average green
duration ("Average Split" in UDOT, which is green only) and the programmed
green (UDOT's "Programmed Split" = programmed split − yellow − red
clearance).  Late bins with few actuations show green left unused at the
end of the split.

Pinned from the v5 source
-------------------------
* Detectors: on-events (Code 82) of the approach's detectors with
  detection type 4, *Lane-by-lane Count*.  Here that is the
  ``Det_P{N}_Stop_Bar`` count loops just past the line; the presence zones
  (``Det_P{N}_Occupancy``) register a queued vehicle once, when it arrives
  on red, so they miss the queue discharge at the start of green.
* Green: each Code 1 of the phase paired with its next Code 8.  An
  actuation at *t* lies in bin ``floor((t − green_ts) / bin_s)``; bin 0
  starts at the green event.  UDOT counts ``[green_ts, yellow_ts]``; here
  the window is half-open, ``[green_ts, yellow_ts)``, so an actuation at
  the yellow event is a yellow actuation as in yellow/red actuations.
* Denominator: every cycle of the time bin, whatever its green length
  (``act_per_cycle``).  A bin past a short green is diluted by the cycles
  that never reached it, so this module adds ``n_reached`` (cycles whose
  green covers the bin start), ``exposure_s`` (green seconds served in the
  bin over all cycles) and ``flow_vph`` (actuations per hour of green
  served in the bin), which separate "no demand" from "no green".
* Programmed split: UDOT reads the plan's split event once per plan and
  subtracts one measured yellow + red clearance (the first of the
  window).  Here each cycle carries the split in force at its green
  (``split_monitor.programmed_at``) less its own clearance
  (``clear_end_ts − yellow_ts``).  UDOT's phase → event-code map skips
  136 (phase 3 → 137); the plan timeline uses Indiana's ``133 + phase``.
* Time binning: UDOT bins greens starting strictly after the bin start;
  here a green at the bin start belongs to that bin (``floor``).  UDOT
  emits bins up to the last non-zero one; here every bin any cycle of the
  group reached is emitted, so unused green shows as zeros.

Signal: a phase, or with ``overlap=N`` overlap *N* (Codes 61/63/64/65),
built by the yellow/red actuations interval helper.  For a protected left
with a flashing-yellow-arrow overlap, the overlap's green spans the
protected arrow and the permissive flashing yellow, so overlap mode
measures the whole left-turn service; phase mode measures the protected
arrow only.  Measured on 315 (2025-12-15, plan 1): overlap A's green
averages 58.8 s against phase 1's 7.8 s, and is mostly permissive service
alongside phase 2, so the phase's programmed split is not an overlay for
it.  Phase mode is the default; overlap mode reports no programmed split.

Gap Marker Rule (censoring)
---------------------------
Every ``event_code == -1`` row is a gap marker.  A green is *censored* —
reported, flagged, counts NA, its actuations not binned — when its green
event has no interval: no yellow to pair with, or a gap marker anywhere in
``[green_ts, clear_end_ts]`` (the shared interval builder drops intervals
across a gap marker, since ``clearance_dur`` would be wrong).  A gap marker
in ``[green_ts, yellow_ts]`` is also checked directly.  Censored greens count in ``n_censored`` and in no
denominator.

Package Location: src/atspm/analysis/green_time_utilization.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .call_service import _bin_time, _ratio
from .counts import _apply_exclusions
from .detector_inference import _to_epoch
from .split_monitor import programmed_at
from .yellow_red_actuations import _signal_intervals

_GAP_CODE: int = -1
_CODE_DET_ON: int = 82
_EPS: float = 1e-6          # epoch subtraction noise before flooring

DEFAULT_BIN_S: float = 2.0

CYCLE_SCHEMA = [
    "phase",
    "overlap",
    "coord_plan",
    "cycle_start",
    "green_ts",
    "yellow_ts",
    "green_dur",
    "clearance_dur",
    "censored",
    "actuations",
    "programmed_split",
    "programmed_green",
]

ACTUATION_SCHEMA = [
    "phase",
    "detector",
    "timestamp",
    "green_ts",
    "t_green",
    "green_bin",
]

BIN_SCHEMA = [
    "time",
    "phase",
    "coord_plan",
    "green_bin",
    "bin_start_s",
    "n_cycles",
    "n_reached",
    "exposure_s",
    "actuations",
    "act_per_cycle",
    "act_per_reached",
    "flow_vph",
]

SPLIT_SCHEMA = [
    "time",
    "phase",
    "coord_plan",
    "n_cycles",
    "n_censored",
    "avg_green_s",
    "avg_clearance_s",
    "programmed_split",
    "programmed_green",
    "actuations",
    "act_per_cycle",
]


def green_time_utilization(
    events_df: pd.DataFrame,
    phase: int,
    detector_ids: List[int],
    bin_s: float = DEFAULT_BIN_S,
    overlap: Optional[int] = None,
    exclusions: Optional[List[Dict[str, Any]]] = None,
    timeline: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-green actuation counts and the binned actuations of one signal.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            Timestamps may be UTC epoch floats or tz-aware Timestamps.  Gap
            markers (``event_code == -1``) must be present.
        phase: Phase the detectors belong to (reported in every row, and
            the phase whose programmed split is looked up).
        detector_ids: Detectors whose on-events (Code 82) are counted.
        bin_s: Width of a second-of-green bin, seconds.  Default 2.
        overlap: Measure overlap *N*'s greens (Codes 61/63/64/65) instead
            of the phase's.
        exclusions: ``TM_Exclusions`` entries ``{detector, phase, status}``;
            matching actuations are dropped first.
        timeline: Output of ``split_monitor.plan_timeline``; when given,
            each green carries the programmed split in force at it.
            Ignored in overlap mode: an overlap's green is not bounded by
            the phase's split.

    Returns:
        ``(cycles, actuations)``.

        *cycles*, one row per green event of the signal, sorted by
        ``green_ts``, columns :data:`CYCLE_SCHEMA`::

            phase             int
            overlap           Int64   – NA in phase mode
            coord_plan        float   – plan at the green event
            cycle_start       input dtype – detected cycle of the green
                                        (NaN/NaT when censored unpaired)
            green_ts          input dtype
            yellow_ts         input dtype (NaN/NaT when unpaired)
            green_dur         float s – yellow_ts − green_ts
            clearance_dur     float s – yellow + red clearance served
            censored          bool
            actuations        Int64 – actuations in [green_ts, yellow_ts)
            programmed_split  Int64 – split in force at the green; NA
                                      without *timeline*, in overlap mode,
                                      or when the cycle or split is 0/unknown
            programmed_green  float s – programmed_split − clearance_dur,
                                      floored at 0
            (counts NA and durations NaN when censored)

        *actuations*, one row per actuation in an uncensored green, sorted
        by ``timestamp``, columns :data:`ACTUATION_SCHEMA`::

            phase      int
            detector   int
            timestamp  input dtype
            green_ts   input dtype – the green it lies in
            t_green    float s – timestamp − green_ts
            green_bin  int – floor(t_green / bin_s)

        Both frames are empty with their schema when the signal has no
        green events.
    """
    if bin_s <= 0:
        raise ValueError(f"bin_s must be positive, got {bin_s}")
    empty = (pd.DataFrame(columns=CYCLE_SCHEMA), pd.DataFrame(columns=ACTUATION_SCHEMA))
    if events_df.empty:
        return empty

    iv, greens = _signal_intervals(events_df, phase, overlap)
    if greens.empty:
        return empty

    # One row per distinct green event; paired ones take their interval.
    greens = greens.assign(_g=_to_epoch(greens["timestamp"]))
    greens = greens.sort_values("_g", kind="stable").drop_duplicates("_g").reset_index(drop=True)
    gr = greens["_g"].to_numpy()
    n = len(greens)

    pos = np.full(n, -1)
    if not iv.empty:
        iv = iv.sort_values("green_ts", kind="stable").reset_index(drop=True)
        g_iv = _to_epoch(iv["green_ts"])
        k = np.searchsorted(g_iv, gr, side="left")
        kk = np.clip(k, 0, len(g_iv) - 1)
        hit = (k < len(g_iv)) & (g_iv[kk] == gr)
        pos[hit] = kk[hit]
    paired = pos >= 0
    take_iv = np.where(paired, pos, 0)

    def _from_iv(col: str) -> pd.Series:
        if iv.empty:
            return greens["timestamp"].where(np.zeros(n, dtype=bool))
        s = iv[col].iloc[take_iv].reset_index(drop=True)
        return s.where(paired)

    G = gr
    Y = np.where(paired, _to_epoch(_from_iv("yellow_ts")) if not iv.empty else np.nan, np.nan)
    CE = np.where(paired, _to_epoch(_from_iv("clear_end_ts")) if not iv.empty else np.nan, np.nan)

    gaps = np.sort(_to_epoch(events_df.loc[events_df["event_code"] == _GAP_CODE, "timestamp"]))
    clear = np.searchsorted(gaps, G, side="left") == np.searchsorted(
        gaps, np.where(paired, Y, G), side="right")
    valid = paired & clear

    cycles = pd.DataFrame({
        "phase": int(phase),
        "overlap": pd.array([overlap] * n, dtype="Int64"),
        "coord_plan": pd.to_numeric(greens["coord_plan"], errors="coerce").fillna(0.0)
                        .to_numpy(dtype=float),
        "cycle_start": _from_iv("cycle_start"),
        "green_ts": greens["timestamp"],
        "yellow_ts": _from_iv("yellow_ts"),
        "green_dur": np.where(valid, Y - G, np.nan),
        "clearance_dur": np.where(valid, CE - Y, np.nan),
        "censored": ~valid,
    })

    counts = np.zeros(n, dtype=np.int64)
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
        take = (k >= 0) & valid[kk] & (A < np.where(valid, Y, -np.inf)[kk])
        A, kk = A[take], kk[take]
        t_green = np.round(A - G[kk], 6)
        counts = np.bincount(kk, minlength=n)
        acts = pd.DataFrame({
            "phase": int(phase),
            "detector": det["parameter"].to_numpy()[take].astype(np.int64),
            "timestamp": det["timestamp"].iloc[np.flatnonzero(take)].reset_index(drop=True),
            "green_ts": greens["timestamp"].iloc[kk].reset_index(drop=True),
            "t_green": np.round(t_green, 2),
            "green_bin": np.floor(t_green / bin_s + _EPS).astype(np.int64),
        })[ACTUATION_SCHEMA]

    cycles["actuations"] = pd.array(counts, dtype="Int64")
    cycles.loc[~valid, "actuations"] = pd.NA

    split = pd.array([pd.NA] * n, dtype="Int64")
    if timeline is not None and not timeline.empty and overlap is None:
        prog = programmed_at(timeline, greens["timestamp"], phase=[int(phase)] * n)
        # Free/preempted (cycle 0) and phases not in the plan (split 0)
        # have no programmed split, as in the split monitor.
        running = (prog["cycle"].fillna(0) > 0) & (prog["split"].fillna(0) > 0)
        split = prog["split"].where(running.to_numpy(), pd.NA).array
    cycles["programmed_split"] = pd.array(split, dtype="Int64")
    sp = cycles["programmed_split"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    cycles["programmed_green"] = np.maximum(sp - cycles["clearance_dur"].to_numpy(dtype=float), 0.0)

    cycles = cycles[CYCLE_SCHEMA].round({
        "green_dur": 2, "clearance_dur": 2, "programmed_green": 2,
    })
    return cycles, acts


def _keys(bin_len: Optional[int]) -> List[str]:
    return ["time", "phase", "coord_plan"] if bin_len is not None else ["phase", "coord_plan"]


def _placed(cycle_df: pd.DataFrame, bin_len: Optional[int]) -> pd.DataFrame:
    df = cycle_df.copy()
    df["time"] = _bin_time(df["green_ts"], bin_len)
    df["_ok"] = (~df["censored"].astype(bool)).astype(int)
    return df


def summarize_gtu_bins(
    cycle_df: pd.DataFrame,
    act_df: pd.DataFrame,
    bin_s: float = DEFAULT_BIN_S,
    bin_len: Optional[int] = 15,
) -> pd.DataFrame:
    """Actuations per second-of-green bin, per time bin (or per plan).

    Args:
        cycle_df: First element of :func:`green_time_utilization`, possibly
            concatenated over phases.
        act_df: Second element, concatenated the same way.
        bin_s: Second-of-green bin width; must match the one the
            actuations were binned with.
        bin_len: Time bin in minutes; ``None`` gives one group per
            (phase, coord_plan), with ``time`` its first green.

    Returns:
        One row per (group, green_bin) for every bin that some uncensored
        cycle of the group reached (``0 … ceil(max green_dur / bin_s) − 1``),
        sorted by ``phase, time, green_bin``, columns :data:`BIN_SCHEMA`::

            time            Timestamp – bin start (UTC when input was epoch)
            phase           int
            coord_plan      float
            green_bin       int
            bin_start_s     float – green_bin × bin_s
            n_cycles        int – uncensored cycles in the group (UDOT's
                                  denominator)
            n_reached       int – cycles with green_dur > bin_start_s
            exposure_s      float – Σ min(max(green_dur − bin_start_s, 0), bin_s)
            actuations      int
            act_per_cycle   float – actuations / n_cycles (UDOT's cell)
            act_per_reached float – actuations / n_reached
            flow_vph        float – actuations / exposure_s × 3600

        Groups holding only censored cycles have no rows.
    """
    if cycle_df is None or cycle_df.empty:
        return pd.DataFrame(columns=BIN_SCHEMA)
    df = _placed(cycle_df, bin_len)
    df = df.loc[df["_ok"] == 1]
    if df.empty:
        return pd.DataFrame(columns=BIN_SCHEMA)
    keys = _keys(bin_len)
    df = df.reset_index(drop=True)
    if bin_len is None:
        first = df.groupby(keys, sort=False)["time"].transform("min")
        df["time"] = first
    df["_gid"] = df.groupby(keys, sort=False).ngroup()
    n_cyc = df.groupby("_gid").size()

    g = df["green_dur"].to_numpy(dtype=float)
    nb = np.ceil(np.round(g / bin_s, 6)).astype(np.int64)
    rows = np.repeat(np.arange(len(df)), nb)
    b = np.arange(nb.sum()) - np.repeat(np.cumsum(nb) - nb, nb)
    exp = np.clip(g[rows] - b * bin_s, 0.0, bin_s)
    cell = pd.DataFrame({"_gid": df["_gid"].to_numpy()[rows], "green_bin": b,
                         "exposure_s": exp, "_one": 1})
    out = cell.groupby(["_gid", "green_bin"], sort=True).agg(
        n_reached=("_one", "sum"), exposure_s=("exposure_s", "sum")).reset_index()

    act_n = pd.Series(dtype=np.int64)
    if act_df is not None and not act_df.empty:
        gid_by_green = pd.Series(df["_gid"].to_numpy(),
                                 index=pd.MultiIndex.from_arrays(
                                     [df["phase"].to_numpy(), _to_epoch(df["green_ts"])]))
        a = act_df
        a_gid = gid_by_green.reindex(pd.MultiIndex.from_arrays(
            [a["phase"].to_numpy(), _to_epoch(a["green_ts"])])).to_numpy()
        ok = ~np.isnan(a_gid.astype(float))
        act_n = pd.DataFrame({"_gid": a_gid[ok].astype(np.int64),
                              "green_bin": a["green_bin"].to_numpy()[ok].astype(np.int64)}) \
            .groupby(["_gid", "green_bin"]).size()
    out["actuations"] = act_n.reindex(
        pd.MultiIndex.from_frame(out[["_gid", "green_bin"]])).fillna(0).to_numpy(dtype=np.int64)

    grp = df.drop_duplicates("_gid").set_index("_gid")[keys + ([] if bin_len is not None else ["time"])]
    out = out.join(grp, on="_gid")
    out["n_cycles"] = n_cyc.reindex(out["_gid"]).to_numpy(dtype=np.int64)
    out["bin_start_s"] = out["green_bin"] * float(bin_s)
    out["act_per_cycle"] = _ratio(out["actuations"], out["n_cycles"])
    out["act_per_reached"] = _ratio(out["actuations"], out["n_reached"])
    out["flow_vph"] = _ratio(out["actuations"] * 3600.0, out["exposure_s"])
    out["phase"] = out["phase"].astype(int)
    out["n_reached"] = out["n_reached"].astype(int)
    out = out.sort_values(["phase", "time", "green_bin"], kind="stable").reset_index(drop=True)
    return out[BIN_SCHEMA].round({
        "exposure_s": 2, "act_per_cycle": 4, "act_per_reached": 4, "flow_vph": 1,
    })


def summarize_gtu_splits(cycle_df: pd.DataFrame, bin_len: Optional[int] = 15) -> pd.DataFrame:
    """Average green and programmed green per time bin (or per plan).

    Args:
        cycle_df: First element of :func:`green_time_utilization`, possibly
            concatenated over phases.
        bin_len: Time bin in minutes; ``None`` gives one row per
            (phase, coord_plan), with ``time`` its first green.

    Returns:
        Columns :data:`SPLIT_SCHEMA`, sorted by ``phase, time``::

            time              Timestamp
            phase             int
            coord_plan        float
            n_cycles          int   – uncensored greens
            n_censored        int
            avg_green_s       float – mean green_dur (UDOT "Average Split")
            avg_clearance_s   float – mean yellow + red clearance
            programmed_split  float – median split in force (NaN if unknown)
            programmed_green  float – median programmed_green (UDOT
                                      "Programmed Split")
            actuations        int
            act_per_cycle     float – actuations / n_cycles

        A group holding only censored greens is kept with zero counts and
        NaN averages, so lost coverage stays visible.
    """
    if cycle_df is None or cycle_df.empty:
        return pd.DataFrame(columns=SPLIT_SCHEMA)
    df = _placed(cycle_df, bin_len)
    df["_cens"] = 1 - df["_ok"]
    df["_act"] = df["actuations"].fillna(0).astype(np.int64)
    ok = df["_ok"] == 1
    for col in ("green_dur", "clearance_dur", "programmed_green"):
        df[col] = df[col].astype(float).where(ok)
    df["_split"] = df["programmed_split"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    df["_split"] = df["_split"].where(ok)
    keys = _keys(bin_len)
    aggs = {} if bin_len is not None else {"time": ("time", "min")}
    agg = df.groupby(keys, sort=False).agg(
        **aggs,
        n_cycles=("_ok", "sum"),
        n_censored=("_cens", "sum"),
        avg_green_s=("green_dur", "mean"),
        avg_clearance_s=("clearance_dur", "mean"),
        programmed_split=("_split", "median"),
        programmed_green=("programmed_green", "median"),
        actuations=("_act", "sum"),
    ).reset_index()
    agg["act_per_cycle"] = _ratio(agg["actuations"], agg["n_cycles"])
    for col in ("n_cycles", "n_censored", "actuations", "phase"):
        agg[col] = agg[col].astype(int)
    agg = agg.sort_values(["phase", "time"], kind="stable").reset_index(drop=True)
    return agg[SPLIT_SCHEMA].round({
        "avg_green_s": 2, "avg_clearance_s": 2, "programmed_split": 1,
        "programmed_green": 2, "act_per_cycle": 4,
    })
