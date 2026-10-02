"""
Throughput Optimizer Validation (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.
Input / output is DataFrames, dicts and plain Python scalars.

Existing time-of-day plans are natural experiments: each runs the same
intersection at a different cycle length and splits.  Before the
optimizer's recommendations are trusted, its model must predict the
observed throughput differences between plans
(``docs/design_optimizer_solver.md`` D8, as amended 2026-10-02).

The model under test
--------------------
Exactly what :func:`atspm.analysis.optimizer.optimize` maximises: the
throughput of an operating point ``(C, {s_p})`` is
``Σ_p 3600·N_p(s_p) / C`` over the saturated phases, with ``N_p`` the
cumulative discharge curve from
:func:`atspm.analysis.flow.discharge_profiles`, held at ``N_p(t_dom)``
beyond its domain.  No correction is applied here, so whatever bias the
optimizer carries shows up in the comparison.

Observed throughput per plan
----------------------------
Over the plan's *complete* cycles: a cycle with a known length (the next
cycle starts in the same plan, with no gap marker in between) and exactly
one split window for every phase under test.  A cycle with two windows
for a phase (a reservice, or windows before the first detected cycle
start attributed to it) is dropped: the model has one split per phase.
``observed_vph = 3600 · Σ_cycles Σ_p q_p / Σ_cycles length``.  The plan's
operating point is the median cycle length and the median per-cycle split
of each phase over the same cycles.

Pairwise prediction
-------------------
For each pair of plans, the *anchor* is the plan whose splits cover the
other's (every phase's median split at least the target's, within
``split_cover_tol``), so the target's operating point lies inside the
anchor's curves and nothing is extrapolated.  One set of curves, built
from the anchor's cycles only, predicts both plans.  The anchor's own
prediction is in-sample; the target's is out-of-sample.  The compared
quantity is the relative change from anchor to target, predicted vs
observed, so a level bias common to both points (from busiest-cycle
selection, for instance) cancels.  A pair where neither plan covers the
other (one longer on Ph2, the other on Ph6) is reported but not tested.

Pass criteria
-------------
* **Ranking:** for every tested pair whose observed change is at least
  ``rank_deadband_pct``, the predicted change has the same sign.
* **Magnitude:** the mean absolute error of the predicted change, in
  percentage points, is at most ``change_tol_pp`` over all tested pairs.

Verdict is ``PASS`` (both hold, at least one sign-tested pair),
``FAIL`` (either fails) or ``INCONCLUSIVE`` (no tested pair, or no pair
outside the dead band).  Both tolerances are provisional until real
distributions are seen.

Gap Marker Rule
---------------
A cycle's length is the gap to the next cycle start, so a cycle with a
gap marker (``event_code == -1``) in ``[cycle_start, next_start)`` has no
length and is dropped; durations never span a gap.  Split windows and
departures come from :func:`atspm.analysis.flow.flow_rate`, whose windows
never span a gap marker.

Package Location: src/atspm/analysis/optimizer_validation.py
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .flow import discharge_profiles, saturation_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_TESTED = "tested"
STATUS_NOT_COVERED = "not_covered"
STATUS_CURVE_MISSING = "curve_missing"
STATUS_TOO_FEW_CYCLES = "too_few_cycles"

_PAIR_SCHEMA = [
    "anchor_plan",
    "target_plan",
    "status",
    "c_anchor",
    "c_target",
    "observed_anchor_vph",
    "observed_target_vph",
    "predicted_anchor_vph",
    "predicted_target_vph",
    "observed_change_pct",
    "predicted_change_pct",
    "change_error_pp",
    "sign_tested",
    "sign_agree",
]

_VALID_CYCLE_SCHEMA = ["cycle_start", "coord_plan", "cycle_len"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def valid_cycles(
    cycles_df: pd.DataFrame,
    gap_ts: Sequence[float] = (),
) -> pd.DataFrame:
    """Cycles with a usable length.

    A cycle's length is the time to the next cycle start.  The last cycle
    has none.  A cycle is dropped when the next cycle runs a different
    plan (a transition cycle) or when a gap marker falls in
    ``[cycle_start, next_start)``.

    Args:
        cycles_df: ``cycles`` rows with ``cycle_start`` (UTC epoch float)
            and ``coord_plan``.
        gap_ts: Timestamps (UTC epoch floats) of the gap markers in the
            window.

    Returns:
        ``[cycle_start, coord_plan, cycle_len]``, sorted by
        ``cycle_start``.

    Raises:
        ValueError: If ``cycle_start`` is not numeric.
    """
    if cycles_df.empty:
        return pd.DataFrame(columns=_VALID_CYCLE_SCHEMA)
    if not pd.api.types.is_numeric_dtype(cycles_df["cycle_start"]):
        raise ValueError("cycles_df.cycle_start must be UTC epoch floats.")

    df = (
        cycles_df[["cycle_start", "coord_plan"]]
        .sort_values("cycle_start")
        .reset_index(drop=True)
    )
    start = df["cycle_start"].to_numpy(dtype=float)
    nxt = df["cycle_start"].shift(-1).to_numpy(dtype=float)
    nxt_plan = df["coord_plan"].shift(-1)

    gaps = np.sort(np.asarray(gap_ts, dtype=float))
    n_gaps = (
        np.searchsorted(gaps, np.nan_to_num(nxt, nan=np.inf), side="left")
        - np.searchsorted(gaps, start, side="left")
    )
    keep = (
        ~np.isnan(nxt)
        & (nxt_plan == df["coord_plan"]).to_numpy()
        & (n_gaps == 0)
    )
    out = df.loc[keep].copy()
    out["cycle_len"] = (nxt - start)[keep]
    return out.reset_index(drop=True)


def _per_cycle(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Served vehicles, split and window count per cycle for one phase.

    Sums ``q`` over detectors and windows, and ``split`` over windows
    (one value per window), keyed by ``cycle_start``.
    """
    windows = cycle_df.groupby(["cycle_start", "green_ts"]).agg(
        q=("q", "sum"), split=("split", "first")
    )
    return windows.groupby(level="cycle_start").agg(
        q=("q", "sum"), split=("split", "sum"), n_windows=("q", "size")
    )


def _curve_value(curve: pd.DataFrame, s: float) -> float:
    """``N(s)`` by linear interpolation, held at the ends of the domain."""
    return float(
        np.interp(s, curve.index.to_numpy(dtype=float),
                  curve["n"].to_numpy(dtype=float))
    )


def _predict_vph(
    curves: Dict[int, pd.DataFrame],
    c: float,
    splits: Dict[int, float],
) -> float:
    """Model throughput ``Σ_p 3600·N_p(s_p) / C`` at one operating point."""
    total = sum(_curve_value(curves[p], s) for p, s in splits.items())
    return 3600.0 * total / c


def _covers(anchor: Dict[int, float], target: Dict[int, float],
            tol: float) -> bool:
    """True when every anchor split is at least the target's, within *tol*."""
    return all(target[p] <= anchor[p] + tol for p in target)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_plans(
    flow: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
    cycles_df: pd.DataFrame,
    gap_ts: Sequence[float] = (),
    pct: float = 1.0,
    split_tolerance: float = 0.10,
    grid_step: float = 0.5,
    min_cycles: int = 5,
    max_lost: float = 10.0,
    sat_threshold: float = 0.8,
    min_plan_cycles: int = 30,
    split_cover_tol: float = 1.0,
    rank_deadband_pct: float = 2.0,
    change_tol_pp: float = 3.0,
) -> Dict[str, object]:
    """Test the optimizer's throughput model against TOD plans.

    See the module docstring for the definitions.  Every phase in *flow*
    is under test; pass the declared saturated phases that have stop-bar
    detectors.

    Args:
        flow: ``{phase: (cycle_df, vehicle_df)}`` from
            ``flow_rate(..., max_lost=None)``, with ``cycle_start`` as UTC
            epoch floats matching *cycles_df*.
        cycles_df: ``cycles`` rows (``cycle_start``, ``coord_plan``).
        gap_ts: Gap-marker timestamps (UTC epoch floats).
        pct: Busiest-cycle percentage for each anchor's curves (same
            meaning as in ``discharge_profiles``).
        split_tolerance: Modal-split tolerance for curve selection.
        grid_step: Curve grid step in seconds.
        min_cycles: Minimum cycles per curve grid row.
        max_lost: End-slack limit for the advisory pass rate.
        sat_threshold: Advisory threshold (``saturation_state``).
        min_plan_cycles: Minimum complete cycles for a plan to be tested.
        split_cover_tol: Seconds a target split may exceed the anchor's
            and still count as covered.
        rank_deadband_pct: Observed changes smaller than this (percent)
            are not sign-tested.
        change_tol_pp: Magnitude tolerance, mean absolute error of the
            predicted change in percentage points.

    Returns:
        Dict with:

        * ``plans`` — one row per plan with complete cycles::

              coord_plan, n_cycles, c_median, observed_vph,
              insample_vph, insample_pct_error, eligible, curve_ok,
              split_p{N}, pass_rate_p{N}   (per phase under test)

          ``insample_vph`` is the plan's own curves at its own operating
          point, so ``insample_pct_error`` is the model's level bias.
          ``pass_rate_p{N}`` is the advisory ``saturation_state`` pass rate
          on the plan's windows (NaN without windows).

        * ``pairs`` — one row per unordered plan pair, columns
          ``_PAIR_SCHEMA``.  ``status`` is ``tested``, ``too_few_cycles``,
          ``not_covered`` or ``curve_missing`` (first that applies, in that
          order after the cycle check).  Predictions are filled when the
          anchor covers the target and has every curve.  Changes are
          percent of the anchor; ``change_error_pp`` is predicted minus
          observed.  ``sign_agree`` is NA unless ``sign_tested``.

        * ``verdict`` — dict: ``verdict`` (``PASS`` / ``FAIL`` /
          ``INCONCLUSIVE``), ``ranking_pass``, ``magnitude_pass``,
          ``n_pairs``, ``n_tested``, ``n_sign_tested``,
          ``mean_abs_change_error_pp``, ``change_tol_pp``,
          ``rank_deadband_pct``, ``phases``, ``warnings``.

    Raises:
        ValueError: If *flow* is empty or a ``cycle_start`` column is not
            numeric.
    """
    if not flow:
        raise ValueError("flow must hold at least one phase.")
    phases = sorted(int(p) for p in flow)
    for p in phases:
        cdf = flow[p][0]
        if not cdf.empty and not pd.api.types.is_numeric_dtype(cdf["cycle_start"]):
            raise ValueError(
                f"flow[{p}] cycle_start must be UTC epoch floats."
            )

    warnings: List[str] = []

    # ---- Complete cycles and observed throughput ----------------------
    complete = valid_cycles(cycles_df, gap_ts).set_index("cycle_start")
    for p in phases:
        pc = _per_cycle(flow[p][0]).add_suffix(f"_p{p}")
        complete = complete.join(pc, how="inner")
    single = (complete[[f"n_windows_p{p}" for p in phases]] == 1).all(axis=1)
    if (~single).any():
        warnings.append(
            f"{int((~single).sum())} cycles with more than one window for a "
            f"phase dropped"
        )
    complete = complete.loc[single]
    q_cols = [f"q_p{p}" for p in phases]
    complete["q_total"] = complete[q_cols].sum(axis=1)

    agg = {
        "n_cycles": ("cycle_len", "size"),
        "c_median": ("cycle_len", "median"),
        "sum_len": ("cycle_len", "sum"),
        "sum_q": ("q_total", "sum"),
    }
    agg.update({f"split_p{p}": (f"split_p{p}", "median") for p in phases})
    plans_df = complete.groupby("coord_plan").agg(**agg).reset_index()
    plans_df["observed_vph"] = 3600.0 * plans_df["sum_q"] / plans_df["sum_len"]
    plans_df["eligible"] = plans_df["n_cycles"] >= min_plan_cycles

    # ---- Per-plan curves, advisory and in-sample prediction -----------
    curves: Dict[float, Dict[int, pd.DataFrame]] = {}
    pass_rates: Dict[str, List[float]] = {f"pass_rate_p{p}": [] for p in phases}
    for plan in plans_df["coord_plan"]:
        curves[plan] = {}
        for p in phases:
            cdf, vdf = flow[p]
            sel = cdf.loc[cdf["coord_plan"] == plan]
            _, prof = discharge_profiles(
                sel, vdf, pct=pct, split_tolerance=split_tolerance,
                stratify=False, grid_step=grid_step, min_cycles=min_cycles,
            )
            if not prof.empty:
                curves[plan][p] = prof
            adv = saturation_state(sel, max_lost=max_lost,
                                   threshold=sat_threshold)
            rate = adv.loc[adv["phase"] == p, "pass_rate"]
            pass_rates[f"pass_rate_p{p}"].append(
                float(rate.iloc[0]) if len(rate) else np.nan
            )
    for col, vals in pass_rates.items():
        plans_df[col] = vals

    splits = {
        row.coord_plan: {p: float(getattr(row, f"split_p{p}")) for p in phases}
        for row in plans_df.itertuples(index=False)
    }
    plans_df["curve_ok"] = [len(curves[pl]) == len(phases)
                            for pl in plans_df["coord_plan"]]
    plans_df["insample_vph"] = [
        _predict_vph(curves[pl], c, splits[pl]) if ok else np.nan
        for pl, c, ok in zip(plans_df["coord_plan"], plans_df["c_median"],
                             plans_df["curve_ok"])
    ]
    plans_df["insample_pct_error"] = (
        100.0 * (plans_df["insample_vph"] - plans_df["observed_vph"])
        / plans_df["observed_vph"]
    )
    plans_df = plans_df[
        ["coord_plan", "n_cycles", "c_median", "observed_vph",
         "insample_vph", "insample_pct_error", "eligible", "curve_ok"]
        + [f"split_p{p}" for p in phases]
        + [f"pass_rate_p{p}" for p in phases]
    ]

    for row in plans_df.itertuples(index=False):
        if not row.eligible:
            warnings.append(
                f"plan {row.coord_plan:g}: {row.n_cycles} complete cycles "
                f"(< {min_plan_cycles}), not tested"
            )
        if not row.curve_ok:
            missing = [p for p in phases if p not in curves[row.coord_plan]]
            warnings.append(
                f"plan {row.coord_plan:g}: no curve for phases {missing} "
                f"(raise pct or widen the window)"
            )

    # ---- Pairs ----------------------------------------------------------
    info = plans_df.set_index("coord_plan")
    rows = []
    for a, b in itertools.combinations(sorted(plans_df["coord_plan"]), 2):
        a_covers = _covers(splits[a], splits[b], split_cover_tol)
        b_covers = _covers(splits[b], splits[a], split_cover_tol)
        if a_covers and b_covers:
            # Mutual cover: the longer total split anchors; ties keep a.
            if sum(splits[b].values()) > sum(splits[a].values()):
                a, b = b, a
        elif b_covers:
            a, b = b, a
        covered = a_covers or b_covers
        anchor, target = info.loc[a], info.loc[b]

        if not (anchor["eligible"] and target["eligible"]):
            status = STATUS_TOO_FEW_CYCLES
        elif not covered:
            status = STATUS_NOT_COVERED
        elif not anchor["curve_ok"]:
            status = STATUS_CURVE_MISSING
        else:
            status = STATUS_TESTED

        pred_a = pred_b = np.nan
        if covered and anchor["curve_ok"]:
            pred_a = _predict_vph(curves[a], anchor["c_median"], splits[a])
            pred_b = _predict_vph(curves[a], target["c_median"], splits[b])

        obs_a, obs_b = anchor["observed_vph"], target["observed_vph"]
        obs_chg = 100.0 * (obs_b - obs_a) / obs_a
        pred_chg = 100.0 * (pred_b - pred_a) / pred_a
        rows.append({
            "anchor_plan": a,
            "target_plan": b,
            "status": status,
            "c_anchor": anchor["c_median"],
            "c_target": target["c_median"],
            "observed_anchor_vph": obs_a,
            "observed_target_vph": obs_b,
            "predicted_anchor_vph": pred_a,
            "predicted_target_vph": pred_b,
            "observed_change_pct": obs_chg,
            "predicted_change_pct": pred_chg,
            "change_error_pp": pred_chg - obs_chg,
        })

    pairs_df = pd.DataFrame(rows, columns=_PAIR_SCHEMA[:-2])
    tested = pairs_df["status"] == STATUS_TESTED
    pairs_df["sign_tested"] = (
        tested & (pairs_df["observed_change_pct"].abs() >= rank_deadband_pct)
    ).astype(bool)
    agree = (
        np.sign(pairs_df["predicted_change_pct"])
        == np.sign(pairs_df["observed_change_pct"])
    )
    pairs_df["sign_agree"] = pd.array(
        [bool(g) if t else pd.NA
         for g, t in zip(agree, pairs_df["sign_tested"])],
        dtype="boolean",
    )
    pairs_df = pairs_df[_PAIR_SCHEMA]

    for row in pairs_df.loc[pairs_df["status"] == STATUS_NOT_COVERED].itertuples():
        warnings.append(
            f"plans {row.anchor_plan:g}/{row.target_plan:g}: neither plan's "
            f"splits cover the other's, not tested"
        )

    # ---- Verdict --------------------------------------------------------
    n_tested = int(tested.sum())
    n_sign = int(pairs_df["sign_tested"].sum())
    mean_err = (
        float(pairs_df.loc[tested, "change_error_pp"].abs().mean())
        if n_tested else np.nan
    )
    magnitude_pass = bool(n_tested and mean_err <= change_tol_pp)
    any_disagree = bool((pairs_df["sign_agree"] == False).any())  # noqa: E712
    ranking_pass = bool(n_sign and not any_disagree)

    if n_tested == 0:
        verdict = "INCONCLUSIVE"
    elif not magnitude_pass or any_disagree:
        verdict = "FAIL"
    elif n_sign == 0:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"

    return {
        "plans": plans_df,
        "pairs": pairs_df,
        "verdict": {
            "verdict": verdict,
            "ranking_pass": ranking_pass,
            "magnitude_pass": magnitude_pass,
            "n_pairs": int(len(pairs_df)),
            "n_tested": n_tested,
            "n_sign_tested": n_sign,
            "mean_abs_change_error_pp": mean_err,
            "change_tol_pp": float(change_tol_pp),
            "rank_deadband_pct": float(rank_deadband_pct),
            "phases": phases,
            "warnings": warnings,
        },
    }
