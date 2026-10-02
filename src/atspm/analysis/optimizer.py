"""
Throughput Cycle-Length Optimizer (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.
Input / output is DataFrames, dicts and plain Python scalars.

Picks the cycle length ``C`` and splits that maximise saturated
throughput ``Σ 3600·N_p(s_p) / C`` over the saturated phases, using the
measured cumulative discharge curves ``N_p(t)`` from
:func:`atspm.analysis.flow.discharge_profiles`.  The design is fixed in
``docs/design_throughput_optimizer.md`` (objective, regime) and
``docs/design_optimizer_solver.md`` (D0–D6, the solver); section tags
below refer to the latter.

Algorithm overview
------------------
Everything runs on integer grid indices (``k`` ↔ ``k·Δ`` seconds, ``Δ``
the curves' grid step), so splits are exact grid multiples.

* Saturated phases with a curve are the variables.  Each gets a value
  function ``v_p[k]``: ``-inf`` below its minimum split, ``N_p(kΔ)``
  inside its measured domain, frozen at ``N_p(t_dom)`` beyond it (D1).
* Every other included phase gets a fixed split per candidate ``C``:
  an unsaturated phase gets ``max(s_min, s_suff(C))``, the smallest grid
  time whose cumulative service covers ``q·C/3600`` vehicles (D2).  A
  saturated phase without a curve is pinned at its minimum.
* Per (barrier group, ring) the saturated value functions are combined
  once by exact max-plus convolution (D1, D3).  Per ``C`` the fixed time
  enters as an index shift, both rings of a group share the group budget
  ``B_g``, and the group budgets sum to ``C``.  An infeasible ``C`` is
  skipped, never clamped (D2.5).
* The best ``C`` is the scan's argmax of saturated throughput.  The
  result is classified ``interior`` / ``boundary`` / ``infeasible`` with
  a re-measurement directive for the boundary case (D4), and the flat
  band around the optimum is reported (D5).

v1 assumption: every included phase is served every cycle (no skipped
phases, no phase-on-demand modelling).

Gap Marker Rule
---------------
This module consumes grid arrays only.  All event-sequence logic, and so
all ``event_code == -1`` handling, lives upstream in ``flow_rate`` /
``_build_phase_intervals`` / ``vehicle_counts``, which never pair across a
gap marker.  There is no gap logic here by construction.

Package Location: src/atspm/analysis/optimizer.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grid step used only when no curve has two rows to infer it from
# (matches discharge_profiles' default).
_DEFAULT_GRID_STEP = 0.5

# Tolerance for float grid / monotonicity checks.
_EPS = 1e-9

# Seconds at the end of a curve's domain over which the tail rate is
# judged rising or flat, and within which a split on a rising curve counts
# as at the data boundary (D4, widened from the design's Δ/2).
_TAIL_SECONDS = 5.0

# Number of feasible scan points that must rise strictly toward a scan
# limit for a C-edge boundary (D4).
_EDGE_POINTS = 3

_SPLITS_SCHEMA = [
    "phase",
    "ring",
    "barrier_group",
    "saturated",
    "allocation_basis",
    "s_star",
    "s_min",
    "s_domain",
    "n_served",
    "capacity_vph",
    "demand_vph",
    "vc",
    "queue_growth_vph",
    "end_inst_rate_vph",
    "at_boundary",
    "surplus",
    "sufficiency_at_boundary",
    "curve_missing",
    "sufficiency_unverified",
]


# ---------------------------------------------------------------------------
# Max-plus primitives
# ---------------------------------------------------------------------------


def _maxplus_convolve(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Max-plus convolution with an argmax memo.

    ``c[k] = max_j a[j] + b[k - j]`` and ``arg[k]`` is the maximising
    ``j``.  Ties resolve to the smallest ``j`` (``np.argmax`` first hit),
    i.e. the least budget to the first operand (D1a).

    Args:
        a: First operand (float, may hold ``-inf``).
        b: Second operand.

    Returns:
        ``(c, arg)``, both of length ``len(a) + len(b) - 1``.
    """
    na, nb = len(a), len(b)
    n_out = na + nb - 1
    c = np.full(n_out, -np.inf)
    arg = np.zeros(n_out, dtype=int)
    for k in range(n_out):
        j = np.arange(max(0, k - nb + 1), min(k, na - 1) + 1)
        s = a[j] + b[k - j]
        i = int(np.argmax(s))
        c[k] = s[i]
        arg[k] = j[i]
    return c, arg


def _maxplus_fold(
    arrays: Sequence[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Fold value functions by max-plus convolution, truncated to *n*.

    Args:
        arrays: Value functions in fold order (ascending phase number).
            Must be non-empty.
        n: Output length (budgets ``0 .. n-1``).

    Returns:
        ``(values, args)``: ``values[k]`` is the best total for budget
        ``k``; ``args[i]`` is the argmax memo of fold step ``i`` (the
        budget kept by the operands before array ``i + 1``).
    """
    acc = np.asarray(arrays[0], dtype=float)[:n]
    args: List[np.ndarray] = []
    for nxt in arrays[1:]:
        acc, arg = _maxplus_convolve(acc, np.asarray(nxt, dtype=float))
        acc, arg = acc[:n], arg[:n]
        args.append(arg)
    return acc, args


def _backtrack(args: List[np.ndarray], k: int) -> List[int]:
    """Recover each operand's budget from a :func:`_maxplus_fold` memo.

    Args:
        args: Memo list from :func:`_maxplus_fold`.
        k: Total budget chosen.

    Returns:
        Budget per operand, in fold order; sums to *k*.
    """
    parts: List[int] = []
    for arg in reversed(args):
        j = int(arg[k])
        parts.append(k - j)
        k = j
    parts.append(k)
    return parts[::-1]


# ---------------------------------------------------------------------------
# Per-phase helpers
# ---------------------------------------------------------------------------


def _phase_value_function(
    n_array: np.ndarray,
    s_min_k: int,
    n_grid: int,
) -> np.ndarray:
    """Build a saturated phase's value function ``v_p`` (D1).

    Args:
        n_array: Cumulative curve ``N_p`` on grid indices ``0 .. t_dom``.
        s_min_k: Minimum split in grid steps.
        n_grid: Output length.

    Returns:
        Float array: ``-inf`` below *s_min_k*, ``N_p`` within the domain,
        frozen at ``N_p(t_dom)`` beyond it.
    """
    v = np.full(n_grid, float(n_array[-1]))
    m = min(len(n_array), n_grid)
    v[:m] = n_array[:m]
    v[:min(s_min_k, n_grid)] = -np.inf
    return v


def _sufficiency_split(
    n_array: np.ndarray,
    required_n: float,
) -> Tuple[int, bool]:
    """Smallest grid index whose cumulative service covers *required_n*.

    Args:
        n_array: Cumulative curve on grid indices ``0 .. t_dom``.
        required_n: Vehicles that must be served per cycle.

    Returns:
        ``(k, at_boundary)``.  When *required_n* exceeds ``N(t_dom)`` the
        split is clamped to the domain edge and *at_boundary* is True —
        never extrapolated.
    """
    if required_n > n_array[-1] + _EPS:
        return len(n_array) - 1, True
    k = int(np.searchsorted(n_array, required_n - _EPS, side="left"))
    return k, False


def _tail_rate(curve: pd.DataFrame, grid_step: float) -> float:
    """Mean approach rate (vph) over the last few seconds of a curve.

    Uses the smoothed ``inst`` column where it has values; ``inst`` is NaN
    at the ends of a centred rolling mean, so when the whole tail is NaN
    the slope of ``n`` over the same span is used instead.
    """
    t_dom = float(curve.index[-1])
    tail = curve.loc[curve.index >= t_dom - _TAIL_SECONDS + _EPS]
    inst = tail["inst"].to_numpy(dtype=float)
    if np.isfinite(inst).any():
        return float(np.nanmean(inst))
    span = float(tail.index[-1] - tail.index[0])
    if span <= 0.0:
        return 0.0
    n = tail["n"].to_numpy(dtype=float)
    return 3600.0 * (n[-1] - n[0]) / span


def _to_steps(seconds: float, grid_step: float) -> int:
    """Seconds → grid steps, rounding up (a minimum is never shortened)."""
    return int(math.ceil(seconds / grid_step - _EPS))


def _validate_curves(
    curves: Dict[int, pd.DataFrame],
) -> float:
    """Check the D0 curve contract and return the shared grid step.

    Raises:
        ValueError: Non-uniform or mismatched grids, an index not starting
            at 0, missing columns, or non-monotone ``n``.
    """
    steps: List[float] = []
    for ph, df in curves.items():
        if not {"n", "inst"}.issubset(df.columns):
            raise ValueError(f"curve for phase {ph} needs columns 'n' and 'inst'.")
        if df.empty:
            raise ValueError(f"curve for phase {ph} is empty.")
        t = df.index.to_numpy(dtype=float)
        if abs(t[0]) > _EPS:
            raise ValueError(f"curve for phase {ph} must start at t = 0.")
        n = df["n"].to_numpy(dtype=float)
        if not np.isfinite(n).all() or (np.diff(n) < -_EPS).any():
            raise ValueError(f"curve for phase {ph}: 'n' is not monotone.")
        if len(t) > 1:
            d = np.diff(t)
            if not np.allclose(d, d[0], atol=_EPS, rtol=0.0):
                raise ValueError(f"curve for phase {ph} has a non-uniform grid.")
            steps.append(float(d[0]))
    if not steps:
        return _DEFAULT_GRID_STEP
    if not np.allclose(steps, steps[0], atol=_EPS, rtol=0.0):
        raise ValueError(f"curves use different grid steps: {sorted(set(steps))}.")
    return steps[0]


def _is_grid_multiple(value: float, grid_step: float) -> bool:
    ratio = value / grid_step
    return abs(ratio - round(ratio)) < 1e-6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize(
    curves: Dict[int, pd.DataFrame],
    structure_df: pd.DataFrame,
    saturated: Dict[int, bool],
    demand_vph: Dict[int, float],
    min_splits: Dict[int, float],
    c_min: float = 60.0,
    c_max: float = 220.0,
    c_step: float = 1.0,
    flat_tol_pct: float = 1.0,
    boundary_rate_tol: float = 100.0,
) -> Dict[str, object]:
    """Find the throughput-maximising cycle length and splits.

    Phases
    ------
    * Included: every structure phase with a barrier group and a non-zero
      ``observed_share``.  A configured phase never served
      (``observed_share == 0``) and an observed phase outside the
      configured structure (``barrier_group`` NaN) are listed as
      ``excluded`` and get no time.
    * Saturated with a curve: optimised.  Saturated without a curve:
      pinned at its minimum, contributes nothing, flagged
      ``curve_missing``.
    * Unsaturated (or absent from *saturated*): fixed at
      ``max(s_min, s_suff(C))``.  Without a curve or without demand it is
      pinned at its minimum, flagged ``sufficiency_unverified`` when the
      other of the two is present.
    * Minimum splits are rounded **up** to the grid.

    For saturated phases true demand is unobservable, so ``vc`` and
    ``queue_growth_vph`` are NaN unless a demand is supplied; a supplied
    one comes from stop-bar counts, which measure *served* volume, so a
    v/c near 1.0 there is a floor, not demand.

    Ties resolve deterministically (D1a): the least budget to the
    lower-numbered phase and to the lower barrier group, so surplus lands
    on the higher-numbered one.  A ring-group with no optimised phase
    gives any surplus to its highest-numbered phase, flagged ``surplus``.

    A ``C*`` held at the lowest feasible cycle by minimums or sufficiency,
    rather than by the scan limit, is a constraint, not a data edge; it is
    reported as a warning, not as a ``boundary`` state.

    Args:
        curves: Per-phase curves from ``discharge_profiles``: indexed by
            ``t`` on one uniform grid from 0, columns ``n`` (monotone) and
            ``inst``.
        structure_df: :func:`atspm.analysis.critical.ring_barrier_structure`
            output, verbatim.
        saturated: Engineer-declared saturation per phase; missing → False.
        demand_vph: Total vph per phase (needed for unsaturated phases).
        min_splits: Full-split minimums in seconds, including clearance,
            for every included phase.
        c_min: Shortest cycle scanned, seconds.  Default ``60.0``.
        c_max: Longest cycle scanned, seconds.  Default ``220.0``.
        c_step: Scan step; a multiple of the grid step.  Default ``1.0``.
        flat_tol_pct: Flat-band tolerance, percent of peak.  Default
            ``1.0``.
        boundary_rate_tol: Tail rate (vph) above which a curve is still
            rising at its domain edge.  Default ``100.0``.

    Returns:
        Dict with keys ``scan`` (DataFrame, one row per candidate C),
        ``optimum`` (scalars), ``splits`` (DataFrame, one row per
        structure phase at C*), ``directive`` (None or dict) and
        ``warnings`` (list of str) — schemas per D6.  ``optimum`` also
        carries ``binding_ring`` (``{group: ring}`` at C*) and
        ``group_min_at_c_max`` (``{group: seconds}``, the per-group
        minimum budget at ``c_max``, which explains an infeasible result).

    Raises:
        ValueError: Empty structure or no included phase; a curve
            violating the contract; ``c_min``/``c_max``/``c_step`` not
            grid multiples or out of order; an included phase without a
            minimum split.
    """
    if structure_df is None or structure_df.empty:
        raise ValueError("structure_df is empty.")

    grid = _validate_curves(curves)
    for name, val in (("c_step", c_step), ("c_min", c_min), ("c_max", c_max)):
        if not _is_grid_multiple(val, grid):
            raise ValueError(f"{name}={val} is not a multiple of the grid step {grid}.")
    if c_step <= 0 or c_min <= 0 or c_max < c_min:
        raise ValueError("need 0 < c_min <= c_max and c_step > 0.")

    warnings: List[str] = []
    n_grid = int(round(c_max / grid)) + 1

    # --- Phase sets -------------------------------------------------------
    st = structure_df.copy()
    st["phase"] = st["phase"].astype(int)
    unplaced = st["barrier_group"].isna()
    unserved = st["observed_share"].fillna(1.0) <= 0.0
    st["_included"] = ~unplaced & ~unserved
    if unplaced.any():
        warnings.append(
            f"excluded_unconfigured: {sorted(st.loc[unplaced, 'phase'].tolist())}"
        )
    inc = st.loc[st["_included"]].copy()
    if inc.empty:
        raise ValueError("structure_df has no served, configured phase.")
    inc["barrier_group"] = inc["barrier_group"].astype(int)
    inc["ring"] = inc["ring"].astype(int)

    phases = sorted(inc["phase"].tolist())
    missing_min = [p for p in phases if p not in min_splits]
    if missing_min:
        raise ValueError(f"min_splits has no entry for phases {missing_min}.")
    stray = sorted(set(curves) - set(st["phase"]))
    if stray:
        warnings.append(f"curves_not_in_structure: {stray}")

    sat = {p: bool(saturated.get(p, False)) for p in phases}
    curve_n = {
        p: curves[p]["n"].to_numpy(dtype=float) for p in phases if p in curves
    }
    dom_k = {p: len(a) - 1 for p, a in curve_n.items()}
    smin_k = {p: _to_steps(float(min_splits[p]), grid) for p in phases}
    dem = {
        p: float(demand_vph[p])
        for p in phases
        if p in demand_vph and demand_vph[p] is not None
        and np.isfinite(demand_vph[p])
    }

    optimised = [p for p in phases if sat[p] and p in curve_n]
    curve_missing = [p for p in phases if sat[p] and p not in curve_n]
    if curve_missing:
        warnings.append(f"curve_missing: {curve_missing}")
    if not optimised:
        warnings.append("no_optimised_phases: saturated throughput is 0 at every C")

    groups = sorted(inc["barrier_group"].unique().tolist())
    rings = sorted(inc["ring"].unique().tolist())
    members = {
        (g, r): sorted(inc.loc[(inc["barrier_group"] == g)
                               & (inc["ring"] == r), "phase"].tolist())
        for g in groups for r in rings
    }

    # --- Level 0: fold each ring-group's optimised phases once (D3) --------
    fold: Dict[Tuple[int, int], Tuple[np.ndarray, List[np.ndarray], List[int]]] = {}
    for key, ph_list in members.items():
        opt = [p for p in ph_list if p in optimised]
        if opt:
            vals, args = _maxplus_fold(
                [_phase_value_function(curve_n[p], smin_k[p], n_grid) for p in opt],
                n_grid,
            )
        else:
            vals, args = np.zeros(n_grid), []
        fold[key] = (vals, args, opt)

    # --- Fixed splits for one C (D2 steps 1-3) ------------------------------
    def _fixed(c_k: int) -> Tuple[Dict[int, int], Dict[int, bool]]:
        s_k: Dict[int, int] = {}
        clamp: Dict[int, bool] = {}
        c_sec = c_k * grid
        for p in phases:
            if p in optimised:
                continue
            clamp[p] = False
            if sat[p] or p not in curve_n or p not in dem:
                s_k[p] = smin_k[p]
                continue
            k, at_edge = _sufficiency_split(curve_n[p], dem[p] * c_sec / 3600.0)
            s_k[p] = max(smin_k[p], k)
            clamp[p] = at_edge
        return s_k, clamp

    def _group_value(c_k: int, s_k: Dict[int, int]):
        """G_g(B) per group and each ring-group's fixed time F."""
        f_time: Dict[Tuple[int, int], int] = {}
        g_vals: Dict[int, np.ndarray] = {}
        for g in groups:
            total = np.zeros(n_grid)
            for r in rings:
                f = sum(s_k[p] for p in members[(g, r)] if p not in optimised)
                f_time[(g, r)] = f
                ring_val = np.full(n_grid, -np.inf)
                if f < n_grid:
                    ring_val[f:] = fold[(g, r)][0][:n_grid - f]
                total = total + ring_val
            g_vals[g] = total
        return g_vals, f_time

    def _solve(c_k: int):
        """Best saturated vehicles/cycle at C and the group budgets."""
        s_k, clamp = _fixed(c_k)
        g_vals, f_time = _group_value(c_k, s_k)
        if len(groups) == 1:
            best = g_vals[groups[0]][c_k]
            budgets = [c_k]
        else:
            acc, args = _maxplus_fold([g_vals[g] for g in groups[:-1]], n_grid)
            last = g_vals[groups[-1]]
            b = np.arange(c_k + 1)
            s = acc[b] + last[c_k - b]
            i = int(np.argmax(s))
            best = s[i]
            budgets = _backtrack(args, i) + [c_k - i]
        return best, dict(zip(groups, budgets)), s_k, clamp, f_time

    def _allocate(budgets, s_k, f_time):
        """Per-phase split (grid steps) and surplus flags at one C."""
        alloc = dict(s_k)
        surplus_fixed: Dict[int, bool] = {}
        for (g, r), ph_list in members.items():
            if not ph_list:
                continue
            spare = budgets[g] - f_time[(g, r)]
            vals, args, opt = fold[(g, r)]
            if opt:
                for p, k in zip(opt, _backtrack(args, spare)):
                    alloc[p] = k
            elif spare > 0:
                top = ph_list[-1]
                alloc[top] += spare
                surplus_fixed[top] = True
        return alloc, surplus_fixed

    tail_rate = {p: _tail_rate(curves[p], grid) for p in optimised}

    tail_k = _to_steps(_TAIL_SECONDS, grid)

    def _edge_flags(alloc):
        # at_boundary looks at the whole tail window, not just the last grid
        # row.  Each cycle's curve stops at its last departure, so the mean
        # curve gains almost nothing in its final step and the DP stops just
        # short of t_dom; a strict edge test would call that interior.
        at_b, surp = {}, {}
        for p in optimised:
            rising = tail_rate[p] > boundary_rate_tol
            at_b[p] = rising and alloc[p] >= dom_k[p] - tail_k
            surp[p] = (not rising) and alloc[p] >= dom_k[p]
        return at_b, surp

    # --- Scan over C --------------------------------------------------------
    c_step_k = int(round(c_step / grid))
    c_ks = list(range(int(round(c_min / grid)), n_grid, c_step_k))
    rows = []
    solved = {}
    for c_k in c_ks:
        best, budgets, s_k, clamp, f_time = _solve(c_k)
        c_sec = c_k * grid
        row = {"C": c_sec, "feasible": bool(np.isfinite(best))}
        if row["feasible"]:
            alloc, surplus_fixed = _allocate(budgets, s_k, f_time)
            at_b, _ = _edge_flags(alloc)
            served_unsat = 0.0
            for p in phases:
                if p in optimised or sat[p] or p not in dem:
                    continue
                served = dem[p]
                if clamp.get(p) and p in curve_n:
                    served = min(served, 3600.0 * curve_n[p][-1] / c_sec)
                served_unsat += served
            row["throughput_sat_vph"] = 3600.0 * best / c_sec
            row["throughput_total_vph"] = row["throughput_sat_vph"] + served_unsat
            for g in groups:
                row[f"b_g{g}"] = budgets[g] * grid
            row["any_phase_boundary"] = any(at_b.values())
            solved[c_k] = (budgets, s_k, clamp, f_time, alloc, surplus_fixed)
        else:
            row["throughput_sat_vph"] = np.nan
            row["throughput_total_vph"] = np.nan
            for g in groups:
                row[f"b_g{g}"] = np.nan
            row["any_phase_boundary"] = False
        rows.append(row)
    scan_df = pd.DataFrame(rows)

    # --- Minimum budget per group at c_max (explains infeasibility) ---------
    s_k_max, _ = _fixed(c_ks[-1])
    group_min = {}
    for g in groups:
        need = []
        for r in rings:
            ph_list = members[(g, r)]
            need.append(sum(
                smin_k[p] if p in optimised else s_k_max[p] for p in ph_list
            ))
        group_min[g] = max(need) * grid

    # --- Optimum and state (D4) ---------------------------------------------
    feas = scan_df.loc[scan_df["feasible"]]
    optimum: Dict[str, object] = {
        "c_star": np.nan, "throughput_sat_vph": np.nan,
        "throughput_total_vph": np.nan, "state": "infeasible",
        "flat_c_lo": np.nan, "flat_c_hi": np.nan,
        "flat_tol_pct": float(flat_tol_pct), "flat_range_censored": False,
        "c_feasible_min": np.nan, "binding_ring": {},
        "group_min_at_c_max": group_min,
    }
    directive = None

    if feas.empty:
        warnings.append(
            f"infeasible: minimum group budgets at c_max {group_min} "
            f"exceed c_max={c_max}"
        )
        splits_df = _splits_frame(
            st, phases, sat, curve_n, dom_k, smin_k, dem, grid, curves,
            optimised, c_sec=None, alloc=None, clamp={}, at_b={}, surp={},
            surplus_fixed={},
        )
        return {"scan": scan_df, "optimum": optimum, "splits": splits_df,
                "directive": None, "warnings": warnings}

    thr = feas["throughput_sat_vph"].to_numpy()
    i_star = int(np.argmax(thr))
    c_star = float(feas["C"].iloc[i_star])
    c_star_k = int(round(c_star / grid))
    budgets, s_k, clamp, f_time, alloc, surplus_fixed = solved[c_star_k]
    at_b, surp = _edge_flags(alloc)

    optimum["c_star"] = c_star
    optimum["throughput_sat_vph"] = float(thr[i_star])
    optimum["throughput_total_vph"] = float(feas["throughput_total_vph"].iloc[i_star])
    optimum["c_feasible_min"] = float(feas["C"].iloc[0])
    optimum["binding_ring"] = {
        g: max(rings, key=lambda r: (
            sum(smin_k[p] if p in optimised else s_k[p] for p in members[(g, r)]),
            -r,
        ))
        for g in groups
    }

    # A C-edge needs at least two feasible points to judge the approach.
    c_edge = None
    n_feas = len(feas)
    first_c, last_c = c_ks[0] * grid, c_ks[-1] * grid
    if n_feas >= 2 and c_star == last_c and i_star == n_feas - 1:
        tail = thr[-min(_EDGE_POINTS, n_feas):]
        if (np.diff(tail) > 0).all():
            c_edge = {"edge": "high", "c_star": c_star, "scan_limit": last_c}
    if n_feas >= 2 and c_star == first_c and i_star == 0:
        head = thr[:min(_EDGE_POINTS, n_feas)]
        if (np.diff(head) < 0).all():
            c_edge = {"edge": "low", "c_star": c_star, "scan_limit": first_c}
    if i_star == 0 and c_star > c_ks[0] * grid:
        warnings.append(
            f"c_star_at_feasibility_limit: C*={c_star} is the shortest feasible cycle"
        )

    boundary_phases = [p for p in optimised if at_b[p]]
    if boundary_phases or c_edge is not None:
        optimum["state"] = "boundary"
        directive = {
            "action": "extend_and_remeasure",
            "phases": [
                {"phase": p, "s_star": alloc[p] * grid,
                 "s_domain": dom_k[p] * grid,
                 "suggested_split": round(1.2 * dom_k[p] * grid, 1)}
                for p in boundary_phases
            ],
            "c_edge": c_edge,
        }
    else:
        optimum["state"] = "interior"

    # --- Flat band (D5) -------------------------------------------------------
    lo, hi, censored = _flat_band(scan_df, c_star, flat_tol_pct)
    optimum["flat_c_lo"], optimum["flat_c_hi"] = lo, hi
    optimum["flat_range_censored"] = censored

    splits_df = _splits_frame(
        st, phases, sat, curve_n, dom_k, smin_k, dem, grid, curves,
        optimised, c_sec=c_star, alloc=alloc, clamp=clamp, at_b=at_b,
        surp=surp, surplus_fixed=surplus_fixed,
    )
    return {"scan": scan_df, "optimum": optimum, "splits": splits_df,
            "directive": directive, "warnings": warnings}


def _flat_band(
    scan_df: pd.DataFrame,
    c_star: float,
    flat_tol_pct: float,
) -> Tuple[float, float, bool]:
    """Contiguous feasible C-interval around C* within tolerance (D5).

    The band grows outward from C* while the next scan row is feasible
    and within ``flat_tol_pct`` of the peak.  It stops early at a row
    whose allocation sits on a data boundary, unless C* itself does.
    ``censored`` is True when a side stopped for any reason other than
    leaving the tolerance: the end of the scan, an infeasible row, or a
    data-boundary row.

    Returns:
        ``(flat_c_lo, flat_c_hi, censored)``.
    """
    i0 = int(scan_df.index[scan_df["C"] == c_star][0])
    peak = float(scan_df.at[i0, "throughput_sat_vph"])
    floor = peak * (1.0 - flat_tol_pct / 100.0) - _EPS
    clip_boundary = not bool(scan_df.at[i0, "any_phase_boundary"])
    censored = False

    def _walk(step: int) -> int:
        nonlocal censored
        i = i0
        while True:
            j = i + step
            if j < 0 or j >= len(scan_df):
                censored = True
                return i
            row = scan_df.iloc[j]
            if not row["feasible"]:
                censored = True
                return i
            if row["throughput_sat_vph"] < floor:
                return i
            if clip_boundary and row["any_phase_boundary"]:
                censored = True
                return i
            i = j

    lo_i, hi_i = _walk(-1), _walk(+1)
    return float(scan_df.at[lo_i, "C"]), float(scan_df.at[hi_i, "C"]), censored


def _splits_frame(
    st: pd.DataFrame,
    phases: List[int],
    sat: Dict[int, bool],
    curve_n: Dict[int, np.ndarray],
    dom_k: Dict[int, int],
    smin_k: Dict[int, int],
    dem: Dict[int, float],
    grid: float,
    curves: Dict[int, pd.DataFrame],
    optimised: List[int],
    c_sec: Optional[float],
    alloc: Optional[Dict[int, int]],
    clamp: Dict[int, bool],
    at_b: Dict[int, bool],
    surp: Dict[int, bool],
    surplus_fixed: Dict[int, bool],
) -> pd.DataFrame:
    """Assemble ``splits_df`` (D6): one row per structure phase."""
    rows = []
    for rec in st.sort_values("phase").itertuples(index=False):
        p = int(rec.phase)
        included = p in phases
        row = {
            "phase": p,
            "ring": int(rec.ring),
            "barrier_group": float(rec.barrier_group),
            "saturated": bool(sat.get(p, False)),
            "s_min": smin_k[p] * grid if included else np.nan,
            "s_domain": dom_k[p] * grid if p in dom_k else np.nan,
            "demand_vph": dem.get(p, np.nan),
            "curve_missing": included and sat.get(p, False) and p not in curve_n,
            "sufficiency_unverified": (
                included and not sat.get(p, False)
                and ((p in dem) != (p in curve_n))
            ),
            "sufficiency_at_boundary": bool(clamp.get(p, False)),
            "at_boundary": bool(at_b.get(p, False)),
            "surplus": bool(surp.get(p, False) or surplus_fixed.get(p, False)),
        }
        if not included:
            row["allocation_basis"] = "excluded"
        elif p in optimised:
            row["allocation_basis"] = "optimized"
        elif (not sat.get(p, False) and p in dem and p in curve_n
              and alloc is not None and alloc[p] > smin_k[p]):
            row["allocation_basis"] = "sufficiency"
        else:
            row["allocation_basis"] = "minimum"

        if included and alloc is not None:
            k = alloc[p]
            row["s_star"] = k * grid
            if p in curve_n:
                kk = min(k, dom_k[p])
                n_served = float(curve_n[p][kk])
                inst = curves[p]["inst"].to_numpy(dtype=float)[kk]
                row["n_served"] = n_served
                row["capacity_vph"] = 3600.0 * n_served / c_sec
                row["end_inst_rate_vph"] = inst
            else:
                row["n_served"] = row["capacity_vph"] = np.nan
                row["end_inst_rate_vph"] = np.nan
        else:
            row["s_star"] = row["n_served"] = row["capacity_vph"] = np.nan
            row["end_inst_rate_vph"] = np.nan

        cap = row["capacity_vph"]
        d = row["demand_vph"]
        if np.isfinite(d) and np.isfinite(cap) and cap > 0:
            row["vc"] = d / cap
            row["queue_growth_vph"] = max(0.0, d - cap)
        else:
            row["vc"] = row["queue_growth_vph"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=_SPLITS_SCHEMA)
