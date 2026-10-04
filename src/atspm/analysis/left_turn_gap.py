"""
Left Turn Gap Analysis (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

UDOT's Left Turn Gap Analysis (``LeftTurnGapAnalysisService.cs`` and
``LeftTurnGapAnalysisReportService.cs``, OpenSourceTransportation/Atspm
v5) measures the gaps in the opposing through traffic that a permissive
left turn could use.  It is a property of the opposing flow, so it is
computed whether or not the left is run permissive: the S-M10 report uses
it to screen protected-only lefts for permissive operation.

Pinned from the v5 source
-------------------------
* Pairing: the left from the approach of through phase 2 crosses the
  phase-6 through, 4 crosses 8, and the reverse (hardcoded).  Here the
  pairing comes from the ``TM_*`` labels: a ``{D}L`` movement is opposed
  by the opposite direction's through traffic, and the window is the green
  of the phase serving it (:func:`through_phases`).  UDOT emits a pair
  even without a left-turn movement; here a pair needs a ``TM_{D}L`` key.
* Detectors: the opposing approach's *Lane-by-lane Count* detectors
  (type 4, else Stop Bar Presence) with movement T, R, TR or TL.  Here
  that is ``TM_{O}T`` ∪ ``TM_{O}R`` (a shared lane is listed under both
  labels).  UDOT's critical gap counts the T/TR/TL lane detectors only:
  4.1 s for up to 2 lanes, else 5.3 s (:func:`critical_gap`).
* Window: each Code 1 (green) of the opposing through phase to its next
  Code 10 (red clearance), so yellow is included.  Here the end is the
  interval builder's ``yellow_end_ts`` (Code 9, else Code 10).
* Gaps: detector *off* events (Code 81) of the union of the detectors
  strictly inside the window, plus the window's two ends; every
  difference of consecutive times is a gap.  The first gap runs from the
  green to the first off and the last from the last off to the red, and a
  green with no actuation is one gap the length of the window.  Off-to-off
  is a headway, rear to rear.
* Bins: a gap is in bin *i* when ``edges[i-1] < gap <= edges[i]``
  (UDOT's ``gap > min && gap <= max``).  Defaults (``MeasureOption``
  preset 31): 1–3.3, 3.3–3.7, 3.7–7.4 and > 7.4 s.  A gap of at most
  ``edges[0]`` is in no UDOT bin; here it counts in ``n_short``.
* Percent turnable: per green, the sum of gaps ≥ ``trend_s`` (7.4 s) over
  the window length; UDOT's chart plots the mean over the greens of a time
  bin.  The time bin is that of the green event, 15 min by default.
* Sums of gaps ≥ a threshold: UDOT's ``SumDurationGap1/2`` are read by
  the gap report as capacity (sum / 4.1 or / 5.3).  Here each green
  carries ``sum_ge_critical`` for one ``critical_s``.

Departures: gaps are rounded to 1 ms before binning, so a 3.3 s gap
between decisecond timestamps is not 3.2999999 (UDOT compares raw
doubles).  UDOT's ``TimeOfDay`` subtraction breaks across midnight; epoch
time does not.

Gap Marker Rule (censoring)
---------------------------
Every ``event_code == -1`` row is a gap marker.  A green is *censored* —
reported, flagged, all counts NA, no gaps emitted — when it has no
interval (no yellow, or the shared interval builder dropped it because a
gap marker lies between green and the end of red clearance) or when a
gap marker lies in ``[green_ts, red_clear_ts]``.  A gap never spans a
marker, and a censored green adds no green time to any denominator.

Package Location: src/atspm/analysis/left_turn_gap.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .call_service import _bin_time, _ratio
from .counts import _apply_exclusions, parse_movements_from_config
from .detector_inference import _to_epoch
from .detector_roles import DIRECTIONS, detector_sets, parse_detector_roles, phase_directions
from .yellow_red_actuations import _signal_intervals

_GAP_CODE: int = -1
_CODE_DET_OFF: int = 81

DEFAULT_EDGES: Tuple[float, ...] = (1.0, 3.3, 3.7, 7.4, np.inf)
DEFAULT_TREND_S: float = 7.4
DEFAULT_BIN_LEN: int = 15

OPPOSITE: Dict[str, str] = {"NB": "SB", "SB": "NB", "EB": "WB", "WB": "EB"}

THROUGH_SCHEMA = ["direction", "phase", "source", "candidates"]

PAIR_SCHEMA = [
    "left",
    "opposing",
    "opposing_phase",
    "source",
    "detectors",
    "n_lanes",
    "critical_s",
    "shared",
]

GAP_SCHEMA = [
    "left",
    "opposing_phase",
    "green_ts",
    "start_ts",
    "end_ts",
    "gap_s",
    "gap_bin",
]

_PHASE_ROLES = ("stop_bar", "occupancy", "arrival", "pairs")


def bin_columns(edges: Sequence[float] = DEFAULT_EDGES) -> List[str]:
    """Count column names for *edges*: ``bin_1`` … ``bin_K``."""
    return [f"bin_{i}" for i in range(1, len(edges))]


def bin_labels(edges: Sequence[float] = DEFAULT_EDGES) -> List[str]:
    """Legend labels for *edges*, as UDOT writes them (``"1-3.3s"``, ``"7.4s+"``)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"{lo:g}s+" if np.isinf(hi) else f"{lo:g}-{hi:g}s")
    return out


def cycle_schema(edges: Sequence[float] = DEFAULT_EDGES) -> List[str]:
    """Columns of :func:`left_turn_gaps`' *cycles* frame for *edges*."""
    return [
        "left",
        "opposing_phase",
        "coord_plan",
        "cycle_start",
        "green_ts",
        "red_clear_ts",
        "window_s",
        "censored",
        "n_actuations",
        "n_gaps",
        "n_short",
        *bin_columns(edges),
        "turnable_s",
        "pct_turnable",
        "sum_ge_critical",
        "n_ge_critical",
    ]


def summary_schema(edges: Sequence[float] = DEFAULT_EDGES) -> List[str]:
    """Columns of :func:`summarize_left_turn_gaps` for *edges*."""
    return [
        "time",
        "left",
        "opposing_phase",
        "n_cycles",
        "n_censored",
        "green_s",
        "n_gaps",
        "n_short",
        *bin_columns(edges),
        "turnable_s",
        "pct_turnable",
        "pct_turnable_time",
        "sum_ge_critical",
        "n_ge_critical",
    ]


def critical_gap(n_lanes: int) -> float:
    """UDOT's critical gap for *n_lanes* opposing through lanes (4.1 or 5.3 s)."""
    return 4.1 if n_lanes <= 2 else 5.3


def check_edges(edges: Sequence[float]) -> np.ndarray:
    """*edges* as a float array.

    Raises:
        ValueError: Fewer than 2 values, not strictly increasing, negative,
            NaN, or ``inf`` anywhere but last.
    """
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or len(e) < 2 or np.isnan(e).any() or e[0] < 0 \
            or not (np.diff(e) > 0).all() or np.isinf(e[:-1]).any():
        raise ValueError(
            f"edges must be ≥ 2 strictly increasing non-negative values, "
            f"only the last may be inf; got {tuple(edges)!r}"
        )
    return e


# ---------------------------------------------------------------------------
# Config: which phase serves each direction, which movement opposes each left
# ---------------------------------------------------------------------------


def through_phases(config: Dict[str, Any]) -> pd.DataFrame:
    """The through phase of each ``TM_*`` direction.

    An explicit ``Det_P{N}_Direction`` key wins.  Otherwise the phase is
    derived: the phases whose ``Det_P{N}_{Stop_Bar,Occupancy,Arrival,Pairs}``
    detectors include a ``TM_{D}T`` detector are the candidates, and a single
    candidate is taken.  Right-turn and left-turn channels are not used:
    a right-turn loop can sit on an overlap's detection.

    Args:
        config: Active config dict, e.g. from
            ``DatabaseManager.get_config_at_date``.

    Returns:
        One row per direction with a ``TM_{D}T`` key or an explicit phase,
        in :data:`DIRECTIONS` order, columns :data:`THROUGH_SCHEMA`::

            direction   str
            phase       Int64 – NA when unresolved
            source      str   – 'config', 'derived', 'ambiguous' (several
                                candidates) or 'none' (no candidate)
            candidates  str   – derived candidate phases, "2,6"; "" when none

    Raises:
        ValueError: Two ``Det_P{N}_Direction`` keys name the same direction,
            or a value is not a direction (from
            ``detector_roles.phase_directions``).
    """
    explicit = phase_directions(config)
    by_dir: Dict[str, int] = {}
    for ph, d in sorted(explicit.items()):
        if d in by_dir:
            raise ValueError(
                f"Det_P{by_dir[d]}_Direction and Det_P{ph}_Direction both name {d}"
            )
        by_dir[d] = ph

    movements = parse_movements_from_config(config)
    roles = parse_detector_roles(config)
    phase_dets: Dict[int, frozenset] = {}
    for role in _PHASE_ROLES:
        for ph, dets in detector_sets(roles, role).items():
            phase_dets[ph] = phase_dets.get(ph, frozenset()) | dets

    rows = []
    for d in DIRECTIONS:
        thru = set(movements.get(f"{d}T", []))
        cands = sorted(ph for ph, dets in phase_dets.items() if dets & thru)
        cand_s = ",".join(str(c) for c in cands)
        if d in by_dir:
            rows.append((d, by_dir[d], "config", cand_s))
        elif not thru:
            continue
        elif len(cands) == 1:
            rows.append((d, cands[0], "derived", cand_s))
        else:
            rows.append((d, None, "ambiguous" if cands else "none", cand_s))
    out = pd.DataFrame(rows, columns=THROUGH_SCHEMA)
    out["phase"] = pd.array(out["phase"], dtype="Int64")
    return out.astype({"direction": str, "source": str, "candidates": str})


def left_turn_pairs(config: Dict[str, Any]) -> pd.DataFrame:
    """Each ``TM_{D}L`` left turn with its opposing through movement.

    Args:
        config: Active config dict.

    Returns:
        One row per left-turn label ``NBL``/``SBL``/``EBL``/``WBL`` present,
        in :data:`DIRECTIONS` order, columns :data:`PAIR_SCHEMA`::

            left            str   – the left-turn movement label, "EBL"
            opposing        str   – the opposing direction, "WB"
            opposing_phase  Int64 – its through phase (NA when unresolved
                                    or the direction has no through)
            source          str   – :func:`through_phases` source, or
                                    'none' when the opposing direction has
                                    no ``TM_{O}T`` key or explicit phase
            detectors       object – sorted ``TM_{O}T`` ∪ ``TM_{O}R``
                                    detector IDs (list; empty when none)
            n_lanes         int   – number of ``TM_{O}T`` detectors
            critical_s      float – :func:`critical_gap` of ``n_lanes``
            shared          object – those *detectors* also listed under a
                                    ``TM_*`` key of another direction
                                    (list; a likely config typo, e.g. 701
                                    channel 54 in both EBL and WBR)
    """
    movements = parse_movements_from_config(config)
    thr = through_phases(config).set_index("direction")
    rows = []
    for d in DIRECTIONS:
        if f"{d}L" not in movements:
            continue
        o = OPPOSITE[d]
        thru = sorted(set(movements.get(f"{o}T", [])))
        dets = sorted(set(thru) | set(movements.get(f"{o}R", [])))
        if o in thr.index:
            phase, source = thr.at[o, "phase"], thr.at[o, "source"]
        else:
            phase, source = pd.NA, "none"
        others = {int(x) for lab, ids in movements.items()
                  if not lab.startswith(o) for x in ids}
        shared = sorted(set(dets) & others)
        rows.append((f"{d}L", o, phase, source, dets, len(thru),
                     critical_gap(len(thru)), shared))
    out = pd.DataFrame(rows, columns=PAIR_SCHEMA)
    out["opposing_phase"] = pd.array(out["opposing_phase"], dtype="Int64")
    out["n_lanes"] = out["n_lanes"].astype(np.int64)
    out["critical_s"] = out["critical_s"].astype(float)
    return out


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


def left_turn_gaps(
    events_df: pd.DataFrame,
    opposing_phase: int,
    detector_ids: List[int],
    left: str = "",
    edges: Sequence[float] = DEFAULT_EDGES,
    trend_s: float = DEFAULT_TREND_S,
    critical_s: float = 4.1,
    exclusions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-green gap counts in the opposing through traffic, and the gaps.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            Timestamps may be UTC epoch floats or tz-aware Timestamps.  Gap
            markers (``event_code == -1``) must be present.
        opposing_phase: Through phase of the opposing movement; its greens
            are the windows.
        detector_ids: Opposing detectors; the union of their off-events
            (Code 81) delimits the gaps.
        left: Left-turn label carried in every row (e.g. ``"EBL"``).
        edges: Gap bin edges, seconds; bin *i* is ``(edges[i-1],
            edges[i]]``.  The last edge may be ``inf``.
        trend_s: Gaps of at least this many seconds count as turnable time.
        critical_s: Gaps of at least this many seconds count in
            ``sum_ge_critical`` / ``n_ge_critical``.
        exclusions: ``TM_Exclusions`` entries ``{detector, phase, status}``;
            matching off-events are dropped first.

    Returns:
        ``(cycles, gaps)``.

        *cycles*, one row per distinct green event of the phase, sorted by
        ``green_ts``, columns ``cycle_schema(edges)``::

            left            str
            opposing_phase  int
            coord_plan      float – plan at the green event
            cycle_start     input dtype – detected cycle of the green
                                          (NaN/NaT when unpaired)
            green_ts        input dtype
            red_clear_ts    input dtype – window end, Code 9/10 (NaN/NaT
                                          when unpaired)
            window_s        float s – red_clear_ts − green_ts
            censored        bool
            n_actuations    Int64 – off-events in the window
            n_gaps          Int64 – n_actuations + 1
            n_short         Int64 – gaps ≤ edges[0]
            bin_1 … bin_K   Int64 – gaps in each bin
            turnable_s      float s – sum of gaps ≥ trend_s
            pct_turnable    float – 100 · turnable_s / window_s
            sum_ge_critical float s – sum of gaps ≥ critical_s
            n_ge_critical   Int64 – gaps ≥ critical_s
            (counts NA and sums/window NaN when censored)

        *gaps*, one row per gap of an uncensored green, sorted by
        ``start_ts``, columns :data:`GAP_SCHEMA`::

            left, opposing_phase
            green_ts        input dtype – the green it lies in
            start_ts        input dtype – green or the off-event opening it
            end_ts          input dtype – the off-event or red closing it
            gap_s           float s, rounded to 1 ms
            gap_bin         int – 0 when ≤ edges[0], 1 … K in a bin, K + 1
                                  when above a finite last edge

        Both frames are empty with their schema when the phase has no
        green events.

    Raises:
        ValueError: *edges* are not strictly increasing and non-negative.
    """
    e = check_edges(edges)
    cschema = cycle_schema(edges)
    empty = (pd.DataFrame(columns=cschema), pd.DataFrame(columns=GAP_SCHEMA))
    if events_df.empty:
        return empty

    iv, greens = _signal_intervals(events_df, opposing_phase, None)
    if greens.empty:
        return empty

    greens = greens.assign(_g=_to_epoch(greens["timestamp"]))
    greens = greens.sort_values("_g", kind="stable").drop_duplicates("_g").reset_index(drop=True)
    G = greens["_g"].to_numpy()
    n = len(greens)

    pos = np.full(n, -1)
    if not iv.empty:
        iv = iv.sort_values("green_ts", kind="stable").reset_index(drop=True)
        g_iv = _to_epoch(iv["green_ts"])
        k = np.searchsorted(g_iv, G, side="left")
        kk = np.clip(k, 0, len(g_iv) - 1)
        hit = (k < len(g_iv)) & (g_iv[kk] == G)
        pos[hit] = kk[hit]
    paired = pos >= 0
    take_iv = np.where(paired, pos, 0)

    def _from_iv(col: str) -> pd.Series:
        if iv.empty:
            return greens["timestamp"].where(np.zeros(n, dtype=bool))
        return iv[col].iloc[take_iv].reset_index(drop=True).where(paired)

    red_clear = _from_iv("yellow_end_ts")
    R = np.where(paired, _to_epoch(red_clear) if not iv.empty else np.nan, np.nan)

    markers = np.sort(_to_epoch(events_df.loc[events_df["event_code"] == _GAP_CODE, "timestamp"]))
    clear = np.searchsorted(markers, G, side="left") == np.searchsorted(
        markers, np.where(paired, R, G), side="right")
    valid = paired & clear
    R_v = np.where(valid, R, -np.inf)

    # Off-events strictly inside an uncensored window.
    det = events_df.loc[(events_df["event_code"] == _CODE_DET_OFF)
                        & events_df["parameter"].isin(list(detector_ids))]
    if exclusions and not det.empty:
        det = _apply_exclusions(det, events_df, exclusions)
    det = det.sort_values("timestamp", kind="stable")
    A = _to_epoch(det["timestamp"])
    k = np.searchsorted(G, A, side="left") - 1          # last green strictly before
    kk = np.clip(k, 0, max(n - 1, 0))
    take = (k >= 0) & valid[kk] & (A < R_v[kk])
    A, kk = A[take], kk[take]
    det_ts = det["timestamp"].iloc[np.flatnonzero(take)].reset_index(drop=True)

    # Window boundaries: each valid green opens, each red closes; the gaps
    # are the consecutive differences within a window.
    vi = np.flatnonzero(valid)
    n_act = np.bincount(kk, minlength=n)
    t_all = np.concatenate([G[vi], A, R[vi]])
    w_all = np.concatenate([vi, kk, vi])
    o_all = np.concatenate([np.zeros(len(vi)), np.ones(len(A)), np.full(len(vi), 2.0)])
    order = np.lexsort((o_all, t_all, w_all))
    t_s, w_s = t_all[order], w_all[order]
    # Input-dtype timestamps in the same order, for the gap rows.
    ts_all = pd.concat([greens["timestamp"].iloc[vi], det_ts,
                        red_clear.iloc[vi]], ignore_index=True).iloc[order].reset_index(drop=True)

    same = w_s[1:] == w_s[:-1]
    gap_s = np.round(t_s[1:] - t_s[:-1], 3)[same]
    gap_w = w_s[1:][same]
    gap_bin = np.where(gap_s <= e[0], 0, np.searchsorted(e, gap_s, side="left"))
    gap_bin = np.minimum(gap_bin, len(e))                # > last finite edge
    in_range = gap_bin < len(e)                          # ≤ e[-1] (always if inf)

    K = len(e) - 1
    counts = np.zeros((n, K + 1), dtype=np.int64)
    np.add.at(counts, (gap_w[in_range], gap_bin[in_range]), 1)
    turn = np.bincount(gap_w, weights=np.where(gap_s >= trend_s, gap_s, 0.0), minlength=n)
    crit = gap_s >= critical_s
    sum_c = np.bincount(gap_w, weights=np.where(crit, gap_s, 0.0), minlength=n)
    n_c = np.bincount(gap_w[crit], minlength=n)

    window = np.where(valid, R - G, np.nan)
    cycles = pd.DataFrame({
        "left": str(left),
        "opposing_phase": int(opposing_phase),
        "coord_plan": pd.to_numeric(greens["coord_plan"], errors="coerce").fillna(0.0)
                        .to_numpy(dtype=float),
        "cycle_start": _from_iv("cycle_start"),
        "green_ts": greens["timestamp"],
        "red_clear_ts": red_clear,
        "window_s": np.round(window, 3),
        "censored": ~valid,
    })

    def _int(x: np.ndarray) -> pd.arrays.IntegerArray:
        a = pd.array(x.astype(np.int64), dtype="Int64")
        a[~valid] = pd.NA
        return a

    cycles["n_actuations"] = _int(n_act)
    cycles["n_gaps"] = _int(n_act + 1)
    cycles["n_short"] = _int(counts[:, 0])
    for i, col in enumerate(bin_columns(edges), start=1):
        cycles[col] = _int(counts[:, i])
    cycles["turnable_s"] = np.where(valid, np.round(turn, 3), np.nan)
    cycles["pct_turnable"] = np.round(100.0 * _ratio(cycles["turnable_s"], window), 3)
    cycles["sum_ge_critical"] = np.where(valid, np.round(sum_c, 3), np.nan)
    cycles["n_ge_critical"] = _int(n_c)

    starts = ts_all.iloc[:-1][same].reset_index(drop=True)
    ends = ts_all.iloc[1:][same].reset_index(drop=True)
    gaps = pd.DataFrame({
        "left": str(left),
        "opposing_phase": int(opposing_phase),
        "green_ts": greens["timestamp"].iloc[gap_w].reset_index(drop=True),
        "start_ts": starts,
        "end_ts": ends,
        "gap_s": gap_s,
        "gap_bin": gap_bin.astype(np.int64),
    })[GAP_SCHEMA]
    return cycles[cschema], gaps


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_left_turn_gaps(
    cycles: pd.DataFrame,
    bin_len: Optional[int] = DEFAULT_BIN_LEN,
    edges: Sequence[float] = DEFAULT_EDGES,
) -> pd.DataFrame:
    """Gap counts per time bin (by green event), as UDOT charts them.

    Args:
        cycles: *cycles* of :func:`left_turn_gaps` (several lefts may be
            concatenated).
        bin_len: Minutes per time bin; ``None`` gives one row per
            (left, opposing_phase) with ``time`` = its first green.
        edges: The edges *cycles* was built with.

    Returns:
        One row per (time, left, opposing_phase) with at least one green,
        sorted, columns ``summary_schema(edges)``::

            time              datetime – bin start (UTC when epoch input)
            left, opposing_phase
            n_cycles          int – uncensored greens
            n_censored        int
            green_s           float s – summed windows of uncensored greens
            n_gaps, n_short, bin_1 … bin_K, n_ge_critical   int – sums
            turnable_s, sum_ge_critical                    float s – sums
            pct_turnable      float – mean of the greens' pct_turnable
                                      (UDOT's trend line); NaN when no
                                      uncensored green
            pct_turnable_time float – 100 · turnable_s / green_s
    """
    schema = summary_schema(edges)
    if cycles.empty:
        return pd.DataFrame(columns=schema)
    counts = ["n_gaps", "n_short", *bin_columns(edges), "n_ge_critical"]
    df = cycles.assign(
        time=_bin_time(cycles["green_ts"], bin_len),
        _ok=~cycles["censored"].astype(bool),
        **{c: cycles[c].astype("Float64").fillna(0).astype(np.int64) for c in counts},
    )
    keys = ["time", "left", "opposing_phase"] if bin_len is not None else ["left", "opposing_phase"]
    aggs = {
        "n_cycles": ("_ok", "sum"),
        "n_censored": ("censored", "sum"),
        "green_s": ("window_s", "sum"),
        **{c: (c, "sum") for c in counts},
        "turnable_s": ("turnable_s", "sum"),
        "pct_turnable": ("pct_turnable", "mean"),
        "sum_ge_critical": ("sum_ge_critical", "sum"),
    }
    if bin_len is None:
        aggs = {"time": ("time", "min"), **aggs}
    out = df.groupby(keys, sort=True).agg(**aggs).reset_index()
    out["n_cycles"] = out["n_cycles"].astype(np.int64)
    out["n_censored"] = out["n_censored"].astype(np.int64)
    out["pct_turnable_time"] = np.round(100.0 * _ratio(out["turnable_s"], out["green_s"]), 3)
    out["pct_turnable"] = np.round(out["pct_turnable"].astype(float), 3)
    for c in ("green_s", "turnable_s", "sum_ge_critical"):
        out[c] = np.round(out[c].astype(float), 3)
    return out.sort_values(["left", "opposing_phase", "time"], kind="stable") \
              .reset_index(drop=True)[schema]
