"""
Detector Configuration Inference (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

Proposes the detector role table (:mod:`atspm.analysis.detector_roles`) from
actuation behaviour alone, for the owner to review — nothing here writes
config.  Design and calibration: ``docs/design_detector_config_inference.md``
(UDOT roadmap S-D6).

Evidence, per active channel
----------------------------
1. **Mode** (on-durations): a *presence* zone holds vehicles through red
   (share of on-intervals > 5 s ≥ ``presence_long_share``, or median on-time
   ≥ ``presence_median_s``); anything else is a *pulse* channel.
2. **Onset jump** (phase): around each green onset (Code 1) of phase *p*,
   events in ``[g, g + W)`` over events in ``[g − W, g)`` (each + 1).  The
   events are presence *releases* (off-times of holds ≥ ``long_on_s``) or
   pulse *on*-times.  A channel of *p* jumps at every onset of *p*; a channel
   of a concurrent phase already green does not, so concurrency resolves
   wherever the two phases' onsets ever differ.
3. **Release regression** (presence only): ridge least squares of per-second
   release counts on every phase's ``[g, g + release_window_s)`` indicator;
   the coefficients are the release rate each phase's onset adds.  It
   separates short left-turn phases that the onset jump blurs.
4. **Lane chains** (light-traffic hours only): for channels *a*, *b*, the lag
   from each *a* on to the next *b* on within ``chain_max_lag_s``.  A sharp
   mode (share of *a*'s actuations in the densest 1-s lag window ≥
   ``chain_min_share``, and that window ≥ ``chain_min_peak`` of the lags
   within ±3 s of it) links *a* → *b* (same lane, *b* downstream).  Vehicles held
   on red spread the lag out, which is why the peak — not the spread of all
   lags — is tested.  A channel entered from ≥ 2 unlinked lanes is
   ``wide`` (spans lanes) and is kept out of lane groups.

5. **Phase calls** (Code 43): a call registered in the *same tenth of a
   second* as the channel's on-event is the controller's own detector →
   phase assignment.  Calls register only when the phase isn't green or
   already called, so they are sparse but unambiguous; they also separate
   phases whose onsets always coincide (4/8).  Only calls to candidate
   phases count (a dummy phase a count loop is assigned to, P9 at 315, is
   ignored).  Per phase, the share of the channel's on-events that called
   it: one phase ≥ ``call_min_share`` and no other ≥ ``call_multi_share``,
   over at least ``call_min_calls`` such on-events, wins over timing
   (confidence high).  Two or more phases ≥ ``call_multi_share`` means the
   detector is assigned to several phases; they are reported as candidates.

Roles: presence → ``occupancy``; pulse with a decisive onset jump →
``stop_bar`` (count loop at/after the stop line); pulse without one that
leads a chain → ``arrival`` (advance); otherwise ``unknown``.  A channel
with no phase of its own (calls or timing) takes its lane chain's phase at
medium confidence.

Gap Marker Rule
---------------
On-intervals closed by a gap marker are censored and excluded (never
measured).  Onsets whose ``[g − W, g + W)`` holds a gap marker are skipped,
and no chain lag spans a gap marker.

Package Location: src/atspm/analysis/detector_inference.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .detectors import _reconstruct_intervals

_GAP_CODE = -1
_CODE_GREEN = 1
_CODE_CALL = 43
_CODE_DET_OFF = 81
_CODE_DET_ON = 82

INFERRED_SCHEMA = [
    "detector", "role", "phase", "lane_group", "wide", "confidence", "candidates",
    "n_act", "med_on", "frac_long", "onset_phase", "onset_ratio",
    "reg_phase", "reg_margin", "n_calls", "call_phase", "call_share",
    "upstream", "downstream",
]
DIFF_SCHEMA = [
    "detector", "status", "configured", "proposed_role", "proposed_phase",
    "confidence", "candidates", "n_act",
]
DIFF_ROLES = ("arrival", "stop_bar", "occupancy")

_INF_DTYPES = {
    "detector": "int64", "role": "str", "phase": "Int64", "lane_group": "Int64",
    "wide": "bool", "confidence": "str", "candidates": "str", "n_act": "int64",
    "med_on": "float64", "frac_long": "float64", "onset_phase": "Int64",
    "onset_ratio": "float64", "reg_phase": "Int64", "reg_margin": "float64",
    "n_calls": "int64", "call_phase": "Int64", "call_share": "float64",
    "upstream": "str", "downstream": "str",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_epoch(ts: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(ts):
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC")
        return (ts.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    return ts.to_numpy(dtype=float)


def _windows_clear_of_gaps(starts: np.ndarray, ends: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """True where no gap marker lies in ``[start, end)``."""
    return np.searchsorted(gaps, starts, side="left") == np.searchsorted(gaps, ends, side="left")


def _count_in(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> int:
    """Events of sorted *x* inside ``[starts, ends)``, summed over windows."""
    return int((np.searchsorted(x, ends) - np.searchsorted(x, starts)).sum())


def _onset_scores(x: np.ndarray, onsets: Dict[int, np.ndarray], w: float) -> Dict[int, float]:
    return {
        ph: (_count_in(x, g, g + w) + 1.0) / (_count_in(x, g - w, g) + 1.0)
        for ph, g in onsets.items()
    }


def _release_design(onsets: Dict[int, np.ndarray], t0: float, n_bins: int, win: float) -> np.ndarray:
    """Per-second coverage of each phase's ``[g, g + win)`` (column per phase)."""
    cols = []
    for g in onsets.values():
        acc = np.zeros(n_bins + 1)
        np.add.at(acc, np.clip((g - t0).astype(int), 0, n_bins), 1.0)
        np.add.at(acc, np.clip((g + win - t0).astype(int), 0, n_bins), -1.0)
        cols.append(np.clip(np.cumsum(acc)[:n_bins], 0.0, 1.0))
    return np.column_stack([np.ones(n_bins)] + cols)


def _ridge(x_design: np.ndarray, gram_inv: np.ndarray, events: np.ndarray, t0: float) -> np.ndarray:
    n_bins = x_design.shape[0]
    y = np.bincount(np.clip((events - t0).astype(int), 0, n_bins - 1), minlength=n_bins)
    return (gram_inv @ (x_design.T @ y.astype(float)))[1:]


def _best_two(scores: Dict[int, float]) -> Tuple[Optional[int], Optional[int]]:
    order = sorted(scores, key=lambda k: (-scores[k], k))
    return (order[0] if order else None), (order[1] if len(order) > 1 else None)


def _light_hours_mask(on_times: Dict[int, np.ndarray]) -> Optional[Tuple[float, np.ndarray]]:
    allon = np.concatenate(list(on_times.values())) if on_times else np.array([])
    if allon.size == 0:
        return None
    t0 = allon.min()
    counts = np.bincount(((allon - t0) // 3600).astype(int))
    busy = counts[counts > 0]
    light = (counts > 0) & (counts <= np.percentile(busy, 25))
    return t0, light


def _chain_links(on_times: Dict[int, np.ndarray], gaps: np.ndarray, max_lag: float,
                 min_share: float, min_peak: float, min_events: int) -> pd.DataFrame:
    """Same-lane links a → b found in light-traffic hours."""
    cols = ["up", "down", "lag", "share"]
    lh = _light_hours_mask(on_times)
    if lh is None:
        return pd.DataFrame(columns=cols)
    t0, light = lh
    light_on = {}
    for d, x in on_times.items():
        h = ((x - t0) // 3600).astype(int)
        light_on[d] = x[light[np.clip(h, 0, len(light) - 1)]]
    rows = []
    for a, xa in light_on.items():
        if len(xa) < min_events:
            continue
        for b, xb in light_on.items():
            if b == a or len(xb) < min_events:
                continue
            k = np.searchsorted(xb, xa, side="right")
            ok = k < len(xb)
            ta, tb = xa[ok], xb[k[ok]]
            lag = tb - ta
            keep = (lag <= max_lag) & _windows_clear_of_gaps(ta, tb, gaps)
            lag = lag[keep]
            if len(lag) < min_events:
                continue
            # Densest 1-s window, centred on an observed lag (no bin edges).
            lag = np.sort(lag)
            in_1s = np.searchsorted(lag, lag + 0.5, side="right") - np.searchsorted(lag, lag - 0.5)
            k = int(in_1s.argmax())
            c = lag[k]
            in_6s = np.searchsorted(lag, c + 3.0, side="right") - np.searchsorted(lag, c - 3.0)
            share = in_1s[k] / len(xa)
            peak = in_1s[k] / in_6s
            if share >= min_share and peak >= min_peak:
                near = lag[(lag >= c - 0.5) & (lag <= c + 0.5)]
                rows.append((a, b, float(np.median(near)), float(share)))
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _coincident_phases(onsets: Dict[int, np.ndarray], tol_s: float, share: float) -> Dict[int, List[int]]:
    """Phases whose green onsets coincide (within *tol_s*) at least *share* of the time.

    No timing evidence can tell such phases apart (e.g. 2 and 6 always starting together).
    """
    out: Dict[int, List[int]] = {p: [] for p in onsets}
    for a, ga in onsets.items():
        for b, gb in onsets.items():
            if a == b or len(ga) == 0 or len(gb) == 0:
                continue
            k = np.searchsorted(gb, ga)
            near = np.minimum(np.abs(gb[np.clip(k, 0, len(gb) - 1)] - ga),
                              np.abs(ga - gb[np.clip(k - 1, 0, len(gb) - 1)]))
            if (near <= tol_s).mean() >= share:
                out[a].append(b)
    return out


def infer_detector_roles(
    events_df: pd.DataFrame,
    phases: Optional[List[int]] = None,
    min_actuations: int = 50,
    onset_window_s: float = 8.0,
    release_window_s: float = 6.0,
    long_on_s: float = 2.0,
    presence_long_share: float = 0.05,
    presence_median_s: float = 0.8,
    onset_min_ratio: float = 1.5,
    onset_high_ratio: float = 3.0,
    count_min_jump: float = 3.0,
    call_min_calls: int = 20,
    call_min_share: float = 0.8,
    call_multi_share: float = 0.5,
    reg_min_margin: float = 0.5,
    chain_max_lag_s: float = 20.0,
    chain_min_share: float = 0.3,
    chain_min_peak: float = 0.5,
) -> pd.DataFrame:
    """Propose a detector role table from actuation behaviour.

    Args:
        events_df: Events with ``timestamp`` (UTC epoch float or datetime),
            ``event_code``, ``parameter``; must include Code 1 (phase green),
            81/82 (detector off/on) and gap markers (-1).  Several days give
            the best result; chains use only the lightest-quartile hours.
        phases: Candidate phases (e.g. those in the ``RB_*`` ring config).
            Default: every phase with a Code 1 in *events_df*.
        min_actuations: Channels with fewer uncensored on-intervals are not
            classified.
        onset_window_s: ``W`` of the onset jump.
        release_window_s: Window after green onset used by the regression.
        long_on_s: A presence hold this long or longer counts as a release.
        presence_long_share: Mode cut: share of on-intervals > 5 s.
        presence_median_s: Mode cut: median on-time.
        onset_min_ratio: Best / runner-up onset jump for a decisive phase.
        onset_high_ratio: Same, for high confidence.
        count_min_jump: A pulse channel whose best onset jump reaches this is
            a count loop (``stop_bar``) even when concurrent phases tie.
        call_min_calls: On-events with a same-tenth call to a candidate
            phase needed to use the call cue.
        call_min_share: Share of those on-events one phase needs to win.
        call_multi_share: Share at which a second phase marks the detector
            as calling several phases.
        reg_min_margin: ``(best − second) / best`` regression coefficient for
            a decisive phase.
        chain_max_lag_s: Longest a → b lag searched.
        chain_min_share: Share of *a*'s actuations in the densest 1-s lag window.
        chain_min_peak: That window's share of the lags within ±3 s of it.

    Returns:
        DataFrame, one row per channel with at least *min_actuations*
        intervals, sorted by detector::

            detector     int64
            role         str    – occupancy | stop_bar | arrival | unknown
            phase        Int64  – NA when the evidence ties or is absent
            lane_group   Int64  – channels chained into one lane; NA if none
            wide         bool   – entered from several lanes (spans lanes)
            confidence   str    – high | medium | low
            candidates   str    – "P4|P8" when *phase* is NA because the
                                  evidence ties, or the phases' onsets
                                  coincide (no timing cue can separate
                                  them); else empty
            n_act        int64  – uncensored on-intervals
            med_on       float  – median on-time (s)
            frac_long    float  – share of on-intervals > 5 s
            onset_phase  Int64  – best phase by onset jump
            onset_ratio  float  – its jump over the runner-up's
            reg_phase    Int64  – best phase by release regression (presence)
            reg_margin   float
            n_calls      int64  – on-events with a same-tenth Code-43 call
                                  to a candidate phase
            call_phase   Int64  – the phase called most often
            call_share   float  – share of those on-events calling it
            upstream     str    – linked upstream channels "54(4.0s)", comma-joined
            downstream   str    – linked downstream channels
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=INFERRED_SCHEMA).astype(_INF_DTYPES)

    ev = pd.DataFrame({
        "timestamp": _to_epoch(events_df["timestamp"]),
        "event_code": events_df["event_code"].to_numpy(),
        "parameter": events_df["parameter"].to_numpy(),
    }).sort_values("timestamp", kind="stable").reset_index(drop=True)
    t = ev["timestamp"].to_numpy()
    code = ev["event_code"].to_numpy()
    par = ev["parameter"].to_numpy()
    gaps = t[code == _GAP_CODE]

    seen = sorted(int(p) for p in np.unique(par[code == _CODE_GREEN]))
    phases = [p for p in seen if phases is None or p in set(phases)]
    w = onset_window_s
    onsets: Dict[int, np.ndarray] = {}
    for ph in phases:
        g = t[(code == _CODE_GREEN) & (par == ph)]
        onsets[ph] = g[_windows_clear_of_gaps(g - w, g + w, gaps)]

    coincident = _coincident_phases(onsets, tol_s=1.0, share=0.9)
    # Regress on groups of coincident phases: their columns are collinear.
    groups: List[List[int]] = []
    for ph in phases:
        if not any(ph in g for g in groups):
            groups.append(sorted({ph, *coincident[ph]}))
    group_onsets = {i: np.sort(np.concatenate([onsets[p] for p in g])) for i, g in enumerate(groups)}

    t0 = float(np.floor(t[0]))
    n_bins = int(np.ceil(t[-1] - t0)) + 1
    design = gram_inv = None
    if phases:
        design = _release_design(group_onsets, t0, n_bins, release_window_s)
        ridge = np.eye(design.shape[1])
        ridge[0, 0] = 0.0
        gram_inv = np.linalg.inv(design.T @ design + ridge)

    rows: List[dict] = []
    on_times: Dict[int, np.ndarray] = {}
    for det in np.unique(par[np.isin(code, (_CODE_DET_ON, _CODE_DET_OFF))]):
        iv = _reconstruct_intervals(ev, int(det))
        if iv.empty:
            continue
        on = iv["on_ts"].to_numpy(float)
        off = iv["off_ts"].to_numpy(float)
        keep = ~np.isin(off, gaps)                       # censored at a gap marker
        on, off = on[keep], off[keep]
        if len(on) < min_actuations:
            continue
        dur = off - on
        frac_long = float((dur > 5.0).mean())
        med = float(np.median(dur))
        presence = frac_long >= presence_long_share or med >= presence_median_s
        on_times[int(det)] = np.sort(on)

        x = np.sort(off[dur >= long_on_s]) if presence else np.sort(on)
        onset = _onset_scores(x, onsets, w) if phases else {}
        o1, _ = _best_two(onset)
        # Runner-up = best phase not coincident with the leader (a coincident
        # partner always scores the same and says nothing).
        rivals = {q: v for q, v in onset.items() if q != o1 and q not in coincident.get(o1, [])}
        o2, _ = _best_two(rivals)
        ratio = onset[o1] / onset[o2] if o2 is not None else (np.inf if o1 is not None else np.nan)

        reg_p, reg_m = None, np.nan
        if presence and phases and len(x) >= min_actuations // 2:
            coef = dict(zip(range(len(groups)), _ridge(design, gram_inv, x, t0)))
            r1, r2 = _best_two(coef)
            if r1 is not None and coef[r1] > 0:
                reg_p = groups[r1][0]           # coincident partners are tied below
                reg_m = (coef[r1] - (coef[r2] if r2 is not None else 0.0)) / coef[r1]

        rows.append({
            "detector": int(det), "presence": presence, "n_act": len(on),
            "med_on": med, "frac_long": frac_long,
            "onset_phase": o1, "onset_ratio": float(ratio), "onset_second": o2,
            "onset_top": onset.get(o1, np.nan) if o1 is not None else np.nan,
            "reg_phase": reg_p, "reg_margin": float(reg_m),
        })

    if not rows:
        return pd.DataFrame(columns=INFERRED_SCHEMA).astype(_INF_DTYPES)
    df = pd.DataFrame(rows).set_index("detector")

    # --- Phase calls in the same tenth as the on-event -----------------------
    on_k = pd.DataFrame({
        "detector": par[code == _CODE_DET_ON],
        "k": np.rint(t[code == _CODE_DET_ON] * 10.0).astype(np.int64),
    })
    call_k = pd.DataFrame({
        "call": par[code == _CODE_CALL],
        "k": np.rint(t[code == _CODE_CALL] * 10.0).astype(np.int64),
    }).drop_duplicates()
    call_k = call_k[call_k["call"].isin(phases)]
    matched = on_k.drop_duplicates().merge(call_k, on="k")
    n_events = matched.groupby("detector")["k"].nunique()
    per = matched.groupby(["detector", "call"]).size().rename("n").reset_index()
    per["share"] = per["n"] / per["detector"].map(n_events)
    top = per.sort_values(["detector", "share", "call"], ascending=[True, False, True]) \
             .drop_duplicates("detector").set_index("detector")
    multi = (per[per["share"] >= call_multi_share].groupby("detector")["call"]
             .agg(lambda c: sorted(int(v) for v in c)))
    df["n_calls"] = n_events.reindex(df.index).fillna(0).astype(int)
    df["call_phase"] = top["call"].reindex(df.index)
    df["call_share"] = top["share"].reindex(df.index)

    # --- Lane chains ----------------------------------------------------------
    links = _chain_links(on_times, gaps, chain_max_lag_s, chain_min_share,
                         chain_min_peak, min_events=20)
    # A channel entered from >= 2 sources that are not linked to each other spans lanes.
    linked_pairs = set(zip(links["up"], links["down"])) if not links.empty else set()
    wide = set()
    if not links.empty:
        for b, grp in links.groupby("down"):
            ups = list(grp["up"])
            unlinked = [(u, v) for i, u in enumerate(ups) for v in ups[i + 1:]
                        if (u, v) not in linked_pairs and (v, u) not in linked_pairs]
            if unlinked:
                wide.add(int(b))
    lane_links = links[~links["up"].isin(wide) & ~links["down"].isin(wide)] if not links.empty else links
    df["wide"] = df.index.isin(wide)

    # Lane groups: connected components of non-wide links.
    parent = {d: d for d in df.index}

    def find(d):
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    for a, b in zip(lane_links["up"], lane_links["down"]):
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    roots = pd.Series({d: find(d) for d in df.index})
    sizes = roots.map(roots.value_counts())
    group_ids = {r: i + 1 for i, r in enumerate(sorted(roots[sizes > 1].unique()))}
    df["lane_group"] = pd.array([group_ids.get(roots[d]) for d in df.index], dtype="Int64")

    def fmt(sub: pd.DataFrame, col: str) -> pd.Series:
        if sub.empty:
            return pd.Series(dtype=str)
        lbl = sub[col].astype(int).astype(str) + "(" + sub["lag"].round(1).astype(str) + "s)"
        key = "down" if col == "up" else "up"
        return lbl.groupby(sub[key]).agg(",".join)

    df["upstream"] = fmt(links, "up").reindex(df.index).fillna("")
    df["downstream"] = fmt(links, "down").reindex(df.index).fillna("")

    # --- Roles, phases, confidence -------------------------------------------------
    decisive_onset = df["onset_ratio"] >= onset_min_ratio
    jumps = df["onset_top"] >= count_min_jump
    if links.empty:
        has_down = after_presence = np.zeros(len(df), bool)
    else:
        has_down = df.index.isin(links["up"])
        presence_ups = links.loc[links["up"].map(df["presence"]).fillna(False).astype(bool), "down"]
        after_presence = df.index.isin(presence_ups)
    df["role"] = np.select(
        [df["presence"], jumps | after_presence, has_down],
        ["occupancy", "stop_bar", "arrival"], default="unknown",
    )
    # A lane-spanning channel fed only by advance lanes is an advance zone too.
    if not links.empty:
        ups_roles = links.assign(r=links["up"].map(df["role"])).groupby("down")["r"].agg(set)
        wide_adv = [d for d in wide if df.at[d, "role"] == "unknown"
                    and ups_roles.get(d) == {"arrival"}]
        df.loc[wide_adv, "role"] = "arrival"

    phase = pd.Series(pd.NA, index=df.index, dtype="Int64")
    conf = pd.Series("low", index=df.index, dtype=object)
    cand = pd.Series("", index=df.index, dtype=object)

    occ = df["role"] == "occupancy"
    on_ok = decisive_onset
    reg_ok = df["reg_margin"] >= reg_min_margin
    group_of = {p: g[0] for g in groups for p in g}
    same = (df["onset_phase"].map(group_of) == df["reg_phase"]).fillna(False).astype(bool)
    agree = occ & on_ok & reg_ok & same
    only_on = occ & on_ok & ~reg_ok
    only_reg = occ & reg_ok & ~on_ok
    disagree = occ & on_ok & reg_ok & ~same
    phase[agree | only_on] = df.loc[agree | only_on, "onset_phase"]
    phase[only_reg] = df.loc[only_reg, "reg_phase"]
    conf[agree] = "high"
    conf[only_on | only_reg] = "medium"
    group_members = {g[0]: g for g in groups}
    for det in df.index[disagree]:
        tied = {*group_members.get(group_of.get(int(df.at[det, "onset_phase"])), []),
                *group_members.get(int(df.at[det, "reg_phase"]), [])}
        cand[det] = "|".join(f"P{q}" for q in sorted(tied))

    sb = (df["role"] == "stop_bar") & decisive_onset
    phase[sb] = df.loc[sb, "onset_phase"]
    conf[sb] = np.where(df.loc[sb, "onset_ratio"] >= onset_high_ratio, "high", "medium")

    # Phases with coincident onsets are indistinguishable: report the tie.
    for det in df.index[phase.notna()]:
        partners = coincident.get(int(phase[det]), [])
        if partners:
            cand[det] = "|".join(f"P{q}" for q in sorted([int(phase[det])] + partners))
            phase[det] = pd.NA
            conf[det] = "low"

    # Ties: both onset leaders listed when no phase was assigned.
    tie = (phase.isna() & (cand == "") & df["onset_phase"].notna()
           & df["onset_second"].notna() & df["role"].isin(["occupancy", "stop_bar"]))
    cand[tie] = ("P" + df.loc[tie, "onset_phase"].astype(int).astype(str)
                 + "|P" + df.loc[tie, "onset_second"].astype(int).astype(str))

    # The controller's calls override timing where they are decisive.
    enough = df["n_calls"] >= call_min_calls
    n_multi = pd.Series({d: len(v) for d, v in multi.items()}).reindex(df.index).fillna(0)
    by_call = enough & (df["call_share"] >= call_min_share) & (n_multi <= 1)
    phase[by_call] = df.loc[by_call, "call_phase"].astype(int)
    conf[by_call] = "high"
    cand[by_call] = ""
    several = enough & (n_multi >= 2)
    for det in df.index[several]:
        phase[det] = pd.NA
        conf[det] = "medium"
        cand[det] = "|".join(f"P{q}" for q in multi[det])

    # Phase-less channels take the phase of their lane chain, repeated so it
    # propagates along a chain (presence → advance lane → lane-spanning zone).
    if not links.empty:
        for _ in range(3):
            via = pd.concat([
                links.assign(det=links["up"], ph=links["down"].map(phase)),
                links.assign(det=links["down"], ph=links["up"].map(phase)),
            ])[["det", "ph"]].dropna(subset=["ph"])
            votes = via.groupby("det")["ph"].agg(lambda s: sorted(set(int(v) for v in s)))
            changed = False
            for det, phs in votes.items():             # one entry per channel
                if det not in df.index or not pd.isna(phase[det]):
                    continue
                if len(phs) == 1:
                    phase[det] = phs[0]
                    conf[det] = "medium"          # indirect: the lane's, not its own
                    cand[det] = ""
                    changed = True
                else:
                    cand[det] = "|".join(f"P{p}" for p in phs)
            # A tie travels along the chain too.
            tied = pd.concat([
                links.assign(det=links["up"], c=links["down"].map(cand)),
                links.assign(det=links["down"], c=links["up"].map(cand)),
            ])[["det", "c"]]
            tied = tied[tied["c"].fillna("").str.len() > 0]
            for det, c in zip(tied["det"], tied["c"]):
                if det in df.index and pd.isna(phase[det]) and not cand[det]:
                    cand[det] = c
                    changed = True
            if not changed:
                break

    df["phase"] = phase
    df["confidence"] = conf
    df["candidates"] = cand
    out = df.reset_index()
    out["onset_phase"] = out["onset_phase"].astype("Int64")
    out["reg_phase"] = out["reg_phase"].astype("Int64")
    out["call_phase"] = out["call_phase"].astype("Int64")
    return out[INFERRED_SCHEMA].astype(_INF_DTYPES).sort_values("detector").reset_index(drop=True)


def diff_detector_roles(
    proposed: pd.DataFrame,
    configured_roles: pd.DataFrame,
    active_counts: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """Compare an inferred role table with the configured one.

    Args:
        proposed: Output of :func:`infer_detector_roles`.
        configured_roles: Output of ``parse_detector_roles(config)``; only the
            ``arrival`` / ``stop_bar`` / ``occupancy`` rows are compared.
        active_counts: Optional ``{detector: n_on_intervals}`` for every
            channel seen, so a configured channel below the inference's
            ``min_actuations`` is told apart from a silent one.

    Returns:
        DataFrame, one row per detector in either table, sorted by detector::

            detector        int64
            status          str    – match | consistent | conflict | new |
                                     silent | low_volume | unclassified
            configured      str    – "occupancy P2" items, comma-joined
            proposed_role   str
            proposed_phase  Int64
            confidence      str
            candidates      str
            n_act           Int64

        ``consistent``: the role matches and the configured phase is one of
        the tied *candidates*.  ``unclassified``: active, unconfigured, role
        ``unknown``.
    """
    cfg = configured_roles.loc[configured_roles["role"].isin(DIFF_ROLES)]
    cfg_items = (
        (cfg["role"] + " P" + cfg["phase"].astype(int).astype(str))
        .groupby(cfg["detector"]).agg(lambda s: ",".join(sorted(set(s))))
    )
    prop = proposed.set_index("detector")
    dets = sorted(set(cfg_items.index) | set(prop.index))
    counts = active_counts or {}

    rows = []
    for det in dets:                                   # one row per channel
        configured = cfg_items.get(det, "")
        if det in prop.index:
            p = prop.loc[det]
            role, ph = p["role"], p["phase"]
            n_act = int(p["n_act"])
            if not configured:
                status = "unclassified" if role == "unknown" else "new"
            else:
                pairs = {tuple(s.split(" P")) for s in configured.split(",")}
                cfg_roles = {r for r, _ in pairs}
                if not pd.isna(ph) and (role, str(int(ph))) in pairs:
                    status = "match"
                elif pd.isna(ph) and role in cfg_roles and any(
                        f"P{q}" in str(p["candidates"]).split("|") for r, q in pairs if r == role):
                    status = "consistent"
                else:
                    status = "conflict"
            rows.append({"detector": det, "status": status, "configured": configured,
                         "proposed_role": role, "proposed_phase": ph,
                         "confidence": p["confidence"], "candidates": p["candidates"],
                         "n_act": n_act})
        else:
            n = counts.get(det, 0)
            rows.append({"detector": det, "status": "low_volume" if n > 0 else "silent",
                         "configured": configured, "proposed_role": None,
                         "proposed_phase": pd.NA, "confidence": None,
                         "candidates": None, "n_act": n})
    out = pd.DataFrame(rows, columns=DIFF_SCHEMA)
    return out.astype({"detector": "int64", "proposed_phase": "Int64", "n_act": "Int64"})
