# Design Addendum: Throughput Optimizer Solver (§6.4)

**Status:** implementation-ready spec (planning session 2026-07-19).
**Parent:** `docs/design_throughput_optimizer.md` — §3 objective and §5 solver
strategy are fixed and inherited; this addendum pins every remaining design
decision to pseudocode level so implementation requires no further design
work.

All column names, signatures, and schemas below reference the code as it
exists on branch `feat/flow-critical-analysis`:

- `src/atspm/analysis/flow.py` — `flow_rate()` → `(cycle_df, vehicle_df)`
  with `cycle_df` columns `[det, phase, green_ts, cycle_start, coord_plan,
  split, green_dur, clear_dur, lost, q]` and `vehicle_df` columns
  `[det, green_ts, t, n, headway]`; `rate_profiles()` →
  `(selected_df, rate_df, inst_df)` where the wide profiles are indexed by
  `t` (grid `grid_step=0.5` s) with per-cycle `"{det} {green_ts}"` columns,
  per-detector `"{det} Mean"` columns, and an overall `"Mean"` column
  (built by `_wide_profile`).
- `src/atspm/analysis/critical.py` — `ring_barrier_structure()` →
  columns `[phase, ring, barrier_group, position, in_config,
  observed_share, source]`; `phase_demand()` → `[phase, movements,
  n_detectors, demand_vph, peak_vph, demand_per_lane, peak_per_lane]`;
  `movement_phase_map()` → `[movement, phase, detectors, n_detectors,
  n_matched]`.
- Shell/CLI patterns mirrored: `FlowRateEngine` / `get_flow_rate`
  (`src/atspm/data/flow.py`), `CriticalMovementEngine` /
  `get_critical_movements` (`src/atspm/data/critical.py`),
  `_add_flow_parser` / `handle_flow` / `_flow_single_intersection`
  (`src/atspm/cli.py`).

---

## D0. Curve representation — what the optimizer actually consumes

**DECISION.** The optimizer consumes **raw cumulative discharge curves**
`N_p(t)` (approach-total vehicles served by elapsed split time `t`), *not*
the effective-rate profiles from `rate_profiles`. A new §6.2 core function
`discharge_profiles` (in `analysis/flow.py`) builds them by pivoting
`vehicle_df["n"]` through the existing `_wide_profile` helper (which
accepts any `value_col`), then:

1. Per-detector mean cumulative curves are the `"{det} Mean"` columns.
2. The **approach total** is the *sum* across per-detector means (the
   existing `"Mean"` column is a per-lane average — wrong quantity for
   throughput; do not use it here).
3. Reindex to the full grid `[0, t_dom]` at `grid_step`; leading NaNs
   (before each detector's first departure) → `0.0`; interior NaNs →
   linear interpolation; then enforce monotonicity with
   `np.maximum.accumulate` (averaging interpolated cycles can produce
   tiny non-monotone artifacts).
4. **Data domain** `t_dom_p` = the *minimum* over configured detectors of
   the last grid `t` where that detector's mean has ≥ `min_cycles`
   support (last valid index of each `"{det} Mean"`). The profile is
   truncated at `t_dom_p`; the last index *is* the boundary marker.
5. Approach instantaneous rate `R_p(t)` = sum across detectors of the
   `inst` per-detector means (from the same `_wide_profile(veh, "inst",
   min_cycles)` pivot), smoothed with a centred rolling mean
   (`rolling=5`, same convention as `plotting/flow.py`). Diagnostic only
   — see D1.

**No overhead normalization is applied** (`normalize` modes do not enter
the optimizer path). Justification: the `end_shift` denominator answers
the *standalone* question "effective rate if the split ended here" by
charging the termination cost to a single approach. In the joint
optimizer, each phase's split `s_p` *includes* its clearance
(`split = green_dur + clear_dur`, exactly as `flow_rate` measures it) and
the splits sum to `C` — the termination cost is therefore charged
explicitly and structurally. Applying `end_shift` on top would
double-charge it. This is consistent with the recorded normalization
intent (overhead in the denominator for the flow report; never shift the
time axis; the optimizer needs neither).

Known approximation (accept, document in the docstring): for a
hypothetical split `s` shorter than measured, `N_p(s)` counts
continued-green discharge at elapsed `s` rather than the 0–2 stragglers
that would cross during the hypothetical clearance ending at `s`. The
bias is ≤ ~1 vehicle, applies identically to every candidate, and does
not move the argmax materially.

Cycle **selection** (modal-split ±`split_tolerance`, busiest-`pct`,
optional stratified mode) is refactored into a shared private helper
`_select_cycles(cycle_df, pct, split_tolerance, stratify)` used by both
`rate_profiles` and `discharge_profiles` (DRY). Stratified mode
(§6.2): group cycles by `(coord_plan, round(split))`, keep the top-`pct`
by summed `q` **within** each stratum, pool survivors. Default
`stratify=False`.

**Per-phase curve contract** (the seam): a `pd.DataFrame` indexed by `t`
(uniform `grid_step` grid from 0.0 to `t_dom_p`) with float columns:

| column | meaning |
|---|---|
| `n` | approach cumulative vehicles served by split time `t` (monotone) |
| `inst` | approach instantaneous rate, vph (smoothed; diagnostic) |

`optimize()` takes `curves: Dict[int, pd.DataFrame]` in this contract and
validates it (uniform grid, monotone `n`) with `ValueError` on violation.

---

## D1. Inner allocation algorithm — exact discrete max-plus DP, not iterative water-filling

**DECISION.** The inner split-allocation solve is implemented as an
**exact discrete dynamic program (max-plus convolution) on the cumulative
curves `N_p`**, not as an iterative water-filling loop on the
instantaneous-rate arrays. Marginal-rate equalization is the *optimality
diagnostic* reported in the output (end-of-split `inst` values per
saturated phase), not the iteration mechanism.

Justification: iterative water-filling is only exact when marginal gains
are non-increasing, and the empirical `inst` curves are lightly smoothed
and locally non-monotone — an equalization loop can stall on local
plateaus, needs isotonic pre-processing, and has convergence/termination
edge cases. On a discrete grid the problem is a classic separable
resource allocation; with the small NEMA structure (≤ 2 saturated phases
per ring per barrier group, ≤ 2 barrier groups, grid ≤ `c_max/grid_step ≈
440` points) exact enumeration via max-plus convolution is globally
optimal for *any* curve shape, deterministic, and cheap. Where the curves
happen to be concave, the DP optimum satisfies end-of-green rate
equalization automatically — the parent §3 marginal condition is
recovered, not abandoned.

Definitions (all arrays on the shared grid, index `k` ↔ time `k·Δ`,
`Δ = grid_step = 0.5` s):

- Per-phase **value function** for saturated phase `p`:

  ```
  v_p[k] = -inf                       if k·Δ < s_min_p         (minimum split, D2)
         = N_p(k·Δ)                   if s_min_p ≤ k·Δ ≤ t_dom_p
         = N_p(t_dom_p)               if k·Δ > t_dom_p          (freeze — never extrapolate, D4)
  ```

- **Max-plus convolution** with argmax memo (the only algorithmic
  primitive; O(n²) once per ring-group, n ≈ 440):

  ```
  def _maxplus_convolve(a, b):
      # c[k] = max_{j} a[j] + b[k - j];  arg[k] = argmax j
      n = len(a); c = full(2n-1, -inf); arg = zeros(2n-1, int)
      for k in range(2n-1):                     # outer loop is fine: ≤ ~880 iterations
          j = arange(max(0, k-n+1), min(k, n-1) + 1)
          s = a[j] + b[k - j]                   # vectorized inner slice
          i = argmax(s)                         # first-hit ⇒ deterministic tie-break (D1a)
          c[k] = s[i]; arg[k] = j[i]
      return c, arg
  ```

- **Tie-breaking (D1a):** `np.argmax` first-hit on the enumeration order,
  which is ascending split for the *first* operand. Operands are folded
  in ascending phase-number order, so ties (including the all-flat
  beyond-domain case where a ring has surplus time) resolve to the
  smallest split for the lower-numbered phase, pushing surplus to the
  higher-numbered phase. This is arbitrary but deterministic and
  documented; surplus dumped onto a flat (beyond-domain) curve is flagged
  `surplus`, not `boundary` (D4).

- **No convergence machinery exists** — the DP is finite and exact by
  construction. There is no step size, no tolerance, no iteration cap.

The allocation grid resolution is `grid_step` (0.5 s), inherited from the
flow profiles; reported splits are exact grid multiples. No sub-grid
refinement — 0.5 s is below field-deployable split precision.

---

## D2. Minimums, sufficiency, and infeasible-C handling

**DECISION — order of operations at each candidate `C`:**

1. **Minimum splits.** `s_min_p` per phase, from config key
   `Min_P{N}_Split` (new; seconds, *full split* including clearance —
   one key covers ped walk + ped clearance + agency minimum, computed by
   the agency). Shell fallback for missing keys:
   `default_min_split = 10.0` s (≈ 5 s min green + 5 s clearance),
   overridable via `--default-min-split`. The core takes the resolved
   `min_splits: Dict[int, float]` and never reads config.
2. **Phase classification** (inputs, not computed here): `saturated:
   Dict[int, bool]` from the §6.3 classifier (D7); `demand_vph:
   Dict[int, float]` for unsaturated phases from
   `phase_demand()["demand_vph"]` (per-phase *total* vph — sufficiency is
   a volume constraint; the per-lane basis is for criticality ranking
   only). `--demand-stat {mean,peak}` selects `demand_vph` vs `peak_vph`,
   default `mean`.
3. **Unsaturated phases are fixed, not variables:**
   `s_j = max(s_min_j, s_suff_j(C))` where
   `s_suff_j(C) = Δ · searchsorted(N_j_array, q_j·C/3600, side="left")`
   — the smallest grid time whose cumulative service covers the cycle's
   demand (ceil to grid). If `q_j·C/3600 > N_j(t_dom_j)` the phase is
   clamped to `t_dom_j` and flagged `sufficiency_at_boundary` (an
   "unsaturated" phase whose requirement exceeds its measured envelope is
   effectively saturated — surfaced, never extrapolated).
4. **Saturated phases** are the DP variables with floor `s_min_p`
   (encoded as `-inf` in `v_p`, D1).
5. **Feasibility check.** Per (barrier group `g`, ring `r`):
   `F_{g,r}(C) = Σ s_j (fixed unsaturated)`;
   `B_min_{g}(C) = max_r [ F_{g,r}(C) + Σ_{p sat in (g,r)} s_min_p ]`.
   If `Σ_g B_min_g(C) > C`, the candidate **C is infeasible: skipped,
   not clamped** — the scan row gets `feasible=False`,
   `throughput = NaN`. Clamping would silently violate ped/agency
   minimums, which is never acceptable output. Because `s_suff_j`
   depends on `C`, the minimum feasible `C` is a fixed point — the grid
   scan resolves it for free: the smallest grid `C` with
   `feasible=True` is reported as `c_feasible_min`. If *no* candidate is
   feasible, the overall result state is `"infeasible"` (D4) and the
   output carries the per-group deficits at `c_max` so the user can see
   which minimums/sufficiency dominate.

Phases handled specially:
- `observed_share == 0.0` (configured, never served) → excluded from
  allocation entirely (they consume no time today; v1 optimizes the
  observed phase set). Listed in `splits_df` with
  `allocation_basis="excluded"`.
- `barrier_group` NaN (observed-but-unconfigured, per
  `ring_barrier_structure`) → excluded, warning in shell (same policy as
  `critical.py`, which drops them from `group_df`).
- Saturated phase **without a curve** (no qualifying cycles) → pinned at
  `s_min_p`, contributes 0 to the objective, flagged
  `curve_missing=True`; the overall result gains
  `warnings: ["curve_missing: [phases]"]`. Erroring out would make the
  tool unusable on partially instrumented sites; pinning at minimum is
  conservative and visibly flagged.
- Unsaturated phase without curve *and* without demand → pinned at
  `s_min_p` (basis `"minimum"`). Without curve but *with* demand →
  pinned at `s_min_p`, flagged `sufficiency_unverified=True`.

v1 assumption (document in docstring): every included phase is served
every cycle (no skipped phases / phase-on-demand modeling).

---

## D3. Ring/barrier coupling — two-level max-plus decomposition

**DECISION.** The barrier constraint is modeled exactly: within barrier
group `g`, both rings receive the **same** group time `B_g`
(phases dwell until the barrier), and `Σ_g B_g = C`. The inner solve is a
two-level decomposition, each level exact:

```
# ---- Level 0: precompute ONCE (independent of C) --------------------
for each (g, r):
    V_sat[g,r], ARG_sat[g,r] = maxplus_fold([v_p for p in saturated
                                             phases of (g,r), ascending
                                             phase order])
    # k-th entry = best total vehicles from that ring-group's saturated
    # phases given k·Δ seconds among them.  A ring-group with one
    # saturated phase folds to v_p itself; with none, to the scalar 0.

# ---- Per candidate C -------------------------------------------------
compute s_fix (D2 steps 1-3)                       # searchsorted, cheap
for each (g, r):
    F = Σ s_fix over unsaturated phases in (g,r)   # C-dependent
    Ring[g,r](B) = V_sat[g,r][round((B - F)/Δ)]    # pure index shift
                   (-inf where B < F + Σ s_min of its sat phases)
for each g:
    G[g](B) = Ring[g,1](B) + Ring[g,2](B)          # elementwise;
                                                   # barrier sync = shared B
# combine barrier groups: choose {B_g} with Σ B_g = C
if 1 group:   T_sat(C) = G[0](C)
if 2 groups:  T_sat(C) = max_b  G[0](b) + G[1](C - b)   # O(n) slice-max
else:         fold sequentially with _maxplus_convolve  # rare; still exact

# ---- Backtrack at chosen C ------------------------------------------
b_g*   from the group-level argmax
per (g,r): sat budget = b_g* - F_{g,r}; walk ARG_sat[g,r] back to per-phase s_p*
```

Justification: this is the exact formalization of "critical path per
barrier group = the ring whose required time is larger" — in the DP the
distinction dissolves, because both rings *fill* `B_g` and the maximizer
counts every vehicle served in both rings (parent §3's objective sums
saturated phases; saturated phases on the nominally non-critical ring
genuinely serve extra vehicles with their surplus green, and counting
them is correct total throughput). The reconciliation question ("how do
the two rings' barrier boundaries reconcile?") has a one-line answer:
they share the single variable `B_g`. The `slot_critical` /
`on_critical_path` flags from `critical_movement_analysis` are **not**
inputs to the solve — the binding ring emerges from the data; they remain
useful for reporting/cross-checks in the shell. The C-independence of
`V_sat` (unsaturated fixed time enters as an index *shift*) makes the
whole 60–220 s scan effectively free after the one-time O(n²) folds.

Diagnostics reported per group at `C*`: `binding_ring` = the ring with
the larger `F + Σ s_min` (which ring constrains `B_min_g`), and per-phase
end-of-split `inst` rates — approximately equalized across saturated
phases when curves are concave (parent §3 marginal condition).

---

## D4. Per-phase data boundaries, result classification, and the directive

**DECISION — freeze, flag, classify:**

- **Mid-fill behavior:** `v_p` is frozen at `N_p(t_dom_p)` beyond the
  measured domain (D1). Zero marginal gain means the DP never *prefers*
  time beyond a phase's data; it lands there only when a ring has
  surplus after every phase's data (or minimum) is exhausted.
- **Per-phase boundary flag** at the chosen allocation:

  ```
  hit_edge_p    = s*_p ≥ t_dom_p - Δ/2
  rising_p      = mean of curve `inst` over the last 5.0 s of the domain
                  > boundary_rate_tol            (default 100.0 vph)
  at_boundary_p = hit_edge_p and rising_p        # data plausibly rising past edge
  surplus_p     = hit_edge_p and not rising_p    # flat tail; edge is benign
  ```

  **Amended 2026-10-02 (implementation finding):** `at_boundary_p` uses
  `s*_p ≥ t_dom_p − 5.0 s` (the same tail window as `rising_p`), not
  `− Δ/2`. Each cycle's curve stops at its last departure, so the mean
  curve gains almost nothing in its final grid step and the DP stops
  just short of `t_dom`. In the saturated regime the domain is always
  about the current green, so the strict test called the textbook
  boundary case `interior`. `surplus_p` keeps the strict `− Δ/2` edge.

- **Overall result state** (exactly one):

  ```
  "infeasible"  no feasible C in [c_min, c_max]                    (D2.5)
  "boundary"    at C*: any at_boundary_p, OR C* is the first/last
                feasible grid C and T_sat is strictly monotone toward
                that edge over the last 3 feasible grid points
  "interior"    otherwise
  ```

- **Directive payload** (present iff state == `"boundary"`; data, not
  prose — the shell prints it, plots annotate it):

  ```
  directive = {
      "action": "extend_and_remeasure",
      "phases": [ {"phase": p, "s_star": s*_p, "s_domain": t_dom_p,
                   "suggested_split": round(1.2 * t_dom_p, 1)}   # +20%
                  for p where at_boundary_p ],
      "c_edge": None | {"edge": "low"|"high", "c_star": C*,
                        "scan_limit": c_min|c_max},
  }
  ```

  The +20% suggested re-measurement split is a starting point for the
  iterative field procedure (adjust → collect → re-run), not a
  recommendation to run at that split. No throughput number is presented
  as "the optimum" when state is `"boundary"`: `optimum["throughput_*"]`
  fields are still populated (best *within data*) but the state field
  gates how the shell and plots caption them.

---

## D5. Sensitivity / flatness metric

**DECISION.** Primary scalar: **`flat_range` — the contiguous feasible
`C`-interval containing `C*` over which predicted saturated throughput
stays within `flat_tol_pct` (default 1.0 %) of the peak**, reported as
`(flat_c_lo, flat_c_hi)` plus the full `scan_df` curve so users and plots
can judge shape directly. Justification: curvature at `C*` is a
second-difference of an empirical, lightly smoothed curve — noise-
dominated and unitless to practitioners. A tolerance-band width is robust
to local wiggle, directly actionable ("any C in [128, 156] is within 1 %
of optimum — pick the corridor-friendly value"), and degrades gracefully:
a wide band honestly reports that the objective doesn't discriminate.
The band is clipped at feasibility edges and at data-boundary C values;
if clipped, `flat_range_censored=True`.

---

## D6. Data contract — the seam Opus implements against

Pure core, `src/atspm/analysis/optimizer.py`:

```python
def optimize(
    curves: Dict[int, pd.DataFrame],   # per-phase, t-indexed, cols ["n","inst"] (D0)
    structure_df: pd.DataFrame,        # ring_barrier_structure() output, verbatim
    saturated: Dict[int, bool],        # §6.3 classifier verdict per phase
    demand_vph: Dict[int, float],      # total vph per phase (unsaturated at least)
    min_splits: Dict[int, float],      # resolved full-split minimums, seconds
    c_min: float = 60.0,
    c_max: float = 220.0,
    c_step: float = 1.0,               # must be an integer multiple of grid_step
    flat_tol_pct: float = 1.0,
    boundary_rate_tol: float = 100.0,
) -> Dict[str, object]:
```

`grid_step` is inferred from the curve index (validated uniform and
identical across phases). Raises `ValueError` on: non-uniform/mismatched
grids, non-monotone `n` after tolerance, `c_step` not a multiple of the
grid, empty `structure_df`.

**Return dict** (plain dict of DataFrames/scalars — house style, no new
dataclasses):

```python
{
  "scan": scan_df,        # one row per candidate C:
                          #   C float, feasible bool,
                          #   throughput_sat_vph float,   # Σ_sat 3600·N_p(s_p)/C  (the objective)
                          #   throughput_total_vph float, # + Σ_unsat q_j (served demand)
                          #   b_g{idx} float per barrier group (budget, s)
                          #   any_phase_boundary bool
  "optimum": {            # scalars
      "c_star": float, "throughput_sat_vph": float,
      "throughput_total_vph": float,
      "state": "interior" | "boundary" | "infeasible",
      "flat_c_lo": float, "flat_c_hi": float,
      "flat_tol_pct": float, "flat_range_censored": bool,
      "c_feasible_min": float | nan,
  },
  "splits": splits_df,    # one row per structure phase at C*:
                          #   phase int, ring int, barrier_group float,
                          #   saturated bool,
                          #   allocation_basis str  ("optimized" | "sufficiency"
                          #                          | "minimum" | "excluded"),
                          #   s_star float, s_min float, s_domain float,
                          #   n_served float          (veh/cycle = N_p(s_star)),
                          #   capacity_vph float      (3600·n_served/C*),
                          #   demand_vph float        (NaN when unobservable),
                          #   vc float                (demand_vph/capacity_vph; NaN when
                          #                            demand unknown — saturated phases),
                          #   queue_growth_vph float  (max(0, demand−capacity); NaN
                          #                            when demand unknown),
                          #   end_inst_rate_vph float (inst at s_star; equalization
                          #                            diagnostic),
                          #   at_boundary bool, surplus bool,
                          #   sufficiency_at_boundary bool,
                          #   curve_missing bool, sufficiency_unverified bool
  "directive": None | dict,   # D4 payload
  "warnings": List[str],
}
```

Notes recorded so nobody re-derives them: for saturated phases true
demand is unobservable (parent §4) so `vc`/`queue_growth_vph` are NaN
unless mapped stop-bar counts exist — and those measure *served* volume,
so a reported v/c ≈ 1.0 there is a floor, not demand. The docstring must
say this.

**Gap markers:** the optimizer core consumes grid arrays only; all
event-sequence logic (and therefore all `event_code == -1` handling)
lives upstream in `flow_rate` / `_build_phase_intervals` /
`vehicle_counts`, which are already gap-aware. `optimizer.py` contains no
gap logic by construction — state this in the module docstring's "Gap
Marker Rule" section (house convention).

---

## D7. Module decomposition, upstream deltas, and sequencing

### Upstream deltas that must land first

1. **§6.2a — `discharge_profiles`** (`analysis/flow.py`, pure):

   ```python
   def discharge_profiles(cycle_df, vehicle_df, pct=1.0,
                          split_tolerance=0.10, stratify=False,
                          grid_step=0.5, min_cycles=5, rolling=5,
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
       # returns (selected_df, profile_df); profile_df per D0
       # (t-indexed, columns "n", "inst", truncated at t_dom)
   ```

   plus the shared `_select_cycles` selection helper and the
   `stratify` mode (also exposed through `rate_profiles` and the
   existing `flow` CLI as `--stratify`).
2. **§6.2b — `flow_rate(max_lost: Optional[float])`:** `max_lost=None`
   disables the end-slack filter (the `lost` column is retained). Needed
   because `flow_rate` currently *drops* non-qualifying (window,
   detector) groups, so the classifier below could not see the
   denominator. Backwards compatible (default stays `10.0`).
3. **§6.3 — saturation classifier** (`analysis/flow.py`, pure, thin):

   ```python
   def saturation_state(cycle_df, max_lost=10.0, threshold=0.8,
                        ) -> pd.DataFrame:
       # input: unfiltered cycle_df (flow_rate(..., max_lost=None))
       # per (phase, det → aggregated to phase): n_cycles,
       # pass_rate = share of cycles with lost ≤ max_lost,
       # saturated = pass_rate ≥ threshold
   ```

   `threshold=0.8` is provisional pending real distributions (parent
   §6.3); it is a plain parameter end to end.

   **Amended 2026-10-01 (owner decision; supersedes the classifier's role
   above and in the shell orchestration below).** `saturated` comes from
   the engineer as `--saturated N ...`, not from `saturation_state`. End
   slack can't reliably separate the regimes: `lost` includes clearance,
   so gap-outs pass, and coordinated phases always force off. As built,
   `saturation_state` gates on max-out/force-off (`flow_rate`'s new
   `termination` column), requires all lanes by default, and is printed
   as an **advisory** beside the declaration. Curves stay on percentile
   selection (`discharge_profiles`); no per-cycle saturation filter feeds
   them. The intended eventual advisory is split failures (GOR/ROR5).

§6.1 (`critical.py`) is already implemented and is consumed verbatim.

### New modules

- **`src/atspm/analysis/optimizer.py`** (Functional Core; pure; Google
  docstrings; no I/O):
  `_phase_value_function(curve_df, s_min, n_grid)`,
  `_maxplus_convolve(a, b)`, `_maxplus_fold(arrays)` (with argmax memos
  and backtracking), `_sufficiency_split(n_array, required_n,
  grid_step)`, `_inner_allocation(...)` (single-C solve, D2/D3),
  `optimize(...)` (D6). Export `optimize` from
  `src/atspm/analysis/__init__.py` alongside the existing exports.
- **`src/atspm/plotting/optimizer.py`** (Functional Core; figure-
  returning only; reuse `_build_title` via
  `from .flow import _build_title` — do not reimplement):
  - `plot_throughput_curve(scan_df, optimum, metadata)` — throughput vs
    C; flat-band shading; `c_star` marker; infeasible region hatched;
    boundary-state annotation.
  - `plot_allocation(splits_df, optimum, metadata)` — per-ring stacked
    horizontal split bars grouped by barrier group at `C*`; minimums and
    `t_dom` overlaid; boundary phases highlighted. Vectorized trace
    construction (`[start, end, None]` segment pattern); no dummy-trace
    legend hacks.
  - `plot_marginal_rates(curves, splits_df, metadata)` — `inst` curves
    per saturated phase with end-of-split markers (the equalization
    view).
- **`src/atspm/data/optimizer.py`** (Imperative Shell), mirroring
  `FlowRateEngine`/`CriticalMovementEngine` exactly:

  ```python
  class OptimizerEngine:
      def __init__(self, db_path: Path, timezone: Optional[str] = None): ...
      def optimize(self, start, end, plans=None,
                   pct=1.0, max_lost=10.0, split_tolerance=0.10,
                   stratify=False, sat_threshold=0.8,
                   demand_stat="mean", default_min_split=10.0,
                   c_min=60.0, c_max=220.0, c_step=1.0,
                   flat_tol_pct=1.0, boundary_rate_tol=100.0,
                   bin_len=15, exclude_missing=True,
                   make_plot=True, output_dir=None,
                   ) -> Optional[Dict[str, object]]: ...
      def validate(self, start, end, plans=None, ...) -> Optional[pd.DataFrame]: ...

  def get_optimization(db_path, start, end, ..., timezone=None): ...
  ```

  Orchestration inside `optimize`: `_parse_range` follows
  `CriticalMovementEngine._parse_range` (datetime-capable end, exclusive
  — peak periods matter here; **not** the date-only `FlowRateEngine`
  variant); one `get_events_with_cycles_df` load with the existing
  `_ALL_FLOW_CODES`; per configured phase (`Det_P{N}_Stopbar` via the
  `_resolve_detector_map` pattern): `flow_rate(..., max_lost=None)` →
  `saturation_state` + `discharge_profiles`; demand via
  `CountEngine.vehicle_counts(hourly=True)` → `movement_phase_map` →
  `phase_demand`; structure via `_query_cycles` + `ring_barrier_structure`;
  `min_splits` from config `Min_P{N}_Split` keys with
  `default_min_split` fallback (warn per missing key); then the pure
  `optimize()`; then plots; `_write_outputs` writes
  `Optimize_Scan_{...}.csv`, `Optimize_Splits_{...}.csv`,
  `Optimize_Summary_{...}.csv` (optimum dict as one row),
  `Optimize_{Curve,Allocation,Marginal}_{...}.html`, using the
  sub-day-aware stamp convention from `CriticalMovementEngine._write_outputs`.
  Console summary prints `state`, `c_star`, flat range, and the directive
  when present.
- **CLI** (`src/atspm/cli.py`): `optimize` subcommand —
  `_add_optimize_parser`, `handle_optimize`,
  `_optimize_single_intersection`, exact mutual-exclusion group
  `--target/--targetid/--all` with batch loop and per-target
  try/except, matching `handle_flow`. Arguments: `--start`, `--end`
  (datetime formats per `critical`), `--plans`, `--pct`, `--max-lost`,
  `--split-tolerance`, `--stratify`, `--sat-threshold`,
  `--demand-stat {mean,peak}`, `--default-min-split`, `--c-min`,
  `--c-max`, `--c-step`, `--flat-tol-pct`, `--no-plot`, `--output`,
  `--validate` (D8). Also add `--stratify` to the existing `flow`
  parser (§6.2a parity).

### Sequencing (each step independently testable)

1. §6.2a/b flow extensions (+ unit tests, + `flow --stratify` CLI flag).
2. §6.3 `saturation_state` (+ tests).
3. `analysis/optimizer.py` core (+ the D8 unit-test battery).
4. `plotting/optimizer.py`.
5. `data/optimizer.py` shell + `optimize` CLI subcommand.
6. `--validate` mode (D8) against real data.

---

## D8. Validation and test strategy

### Natural-experiment validation (parent §7) — `OptimizerEngine.validate` / `atspm optimize --validate`

**DECISION.** Concrete procedure, delivered as an engine method + CLI
flag producing `Optimize_Validation_{...}.csv`:

1. **Data:** a user-chosen window with sustained oversaturation spanning
   ≥ 2 TOD plans (e.g., weekday 15:00–19:00 crossing a plan change).
   Saturation is confirmed per (phase, plan) via `saturation_state`
   pass-rates on plan-filtered cycles; plan/phase combinations that fail
   are excluded (the regime assumption doesn't hold there).
2. **Observed throughput per plan:** served vehicles from the flow
   measurement itself — `cycle_df` (from `flow_rate(...,
   max_lost=None)`, filtered to each `coord_plan`) gives `q` per
   (cycle, det); observed vph = `Σ q / duration_hours` per plan, summed
   over curve-instrumented phases.
3. **Predicted throughput per plan:** each plan's *actual* operating
   point — `C̄` = median cycle length (diff of `cycles.cycle_start`
   within plan) and `s̄_p` = median `split` per (phase, plan) from
   `cycle_df` — evaluated on curves built **leave-one-plan-out**
   (stratified selection over the *other* plans' cycles), predicted vph
   `= Σ_p 3600·N_p^{-plan}(s̄_p)/C̄`. Leave-one-plan-out prevents the
   evaluated plan's own cycles from trivially reproducing themselves.
4. **Pass criteria** (both required before recommendations are trusted):
   - **Ranking:** the sign of the predicted between-plan throughput
     difference matches the observed sign for every saturated plan pair.
   - **Magnitude:** mean absolute percentage error of predicted vs
     observed per-plan throughput ≤ 10 % (provisional tolerance —
     tighten/loosen after the first real runs; it is a parameter of
     `validate`, `mape_tol=10.0`).
   The CSV carries one row per plan: `coord_plan, n_cycles, c_median,
   observed_vph, predicted_vph, pct_error`, plus a printed PASS/FAIL.

### Solver unit tests (`tests/analysis/test_optimizer.py`, unittest style per `tests/analysis/test_critical.py`)

Synthetic curves with analytically known optima; every test asserts
against closed-form or brute-force answers:

1. **Concave analytic curves** (e.g., `N_p(t) = a_p·(1 − exp(−t/τ_p))`
   discretized): DP optimum matches the KKT/water-filling solution;
   end-of-split `inst` rates equalized within one grid step.
2. **Non-monotone `inst`** (bumpy marginal gains): DP result equals
   exhaustive brute-force enumeration over all grid allocations (small
   grid, 2 phases × 2 groups) — the case iterative water-filling would
   get wrong.
3. **Single saturated phase:** receives the entire ring budget minus
   fixed/minimum time; throughput formula exact.
4. **Minimums-dominated:** minimums + sufficiency leave zero slack at
   some C — allocation pins to `s_min` everywhere feasible; below that,
   `feasible=False`.
5. **Infeasible everywhere:** minimums sum > `c_max` for all C → state
   `"infeasible"`, all-NaN throughput, `c_feasible_min` NaN.
6. **All-boundary:** strictly rising truncated curves → state
   `"boundary"`, directive lists every phase with
   `suggested_split = 1.2·t_dom`; flat/surplus tail instead → state
   `"interior"` with `surplus=True` (rising-vs-flat discrimination).
7. **C-edge boundary:** optimum at `c_min`/`c_max` with monotone
   approach → `"boundary"` with `c_edge` payload.
8. **Sufficiency inversion:** `_sufficiency_split` ceils to grid;
   requirement beyond domain → clamp + `sufficiency_at_boundary`.
9. **Barrier coupling:** hand-built 2-group structure where shifting
   budget between groups changes the optimum; verify `b_g*` and that
   both rings of a group share `B_g` exactly.
10. **Determinism/tie-break:** flat curves → surplus lands per D1a,
    byte-identical across runs.
11. **`_maxplus_convolve`** against brute force on random arrays
    (property test, fixed seed).
12. **Contract validation:** mismatched grids, non-monotone `n`,
    `c_step` not a grid multiple → `ValueError`.
13. **Flatness:** synthetic flat-topped scan → `flat_c_lo/hi` exact;
    censoring at feasibility edge sets `flat_range_censored`.

---

## Assumptions to confirm at implementation time

1. **`Min_P{N}_Split` config key** (name, and that values are full-split
   seconds including clearance) — new `int_cfg.csv` column; needs the
   user's sign-off before the hybrid-schema import picks it up.
2. **Saturation threshold default (0.8)** and **`boundary_rate_tol`
   (100 vph)** — provisional until real distributions are inspected
   (parent §6.3 explicitly defers this); both are plain parameters.
3. **Validation MAPE tolerance (10 %)** — placeholder until the first
   real `--validate` runs; ranking criterion is the hard gate.
4. **`demand_stat` default `mean`** — mean-of-bins matches the
   throughput accounting period; switch the default to `peak` if field
   review shows unsaturated phases failing to clear on peak bins.
