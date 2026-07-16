# Design Brief: Throughput Cycle-Length Optimizer

**Status:** scoped, not implemented (strategy discussion 2026-07-15).
**Audience:** implementation session starting fresh — this document is the
distilled spec; it assumes familiarity with the codebase but not with the
scoping discussion that produced it.

---

## 1. Problem statement

For intersections with **sustained oversaturation**, find the cycle length
`C` and split allocation that maximize vehicle throughput, using the
empirically measured discharge behavior from `src/atspm/analysis/flow.py`
rather than HCM constant-saturation-flow assumptions.

Primary use case: **critical intersections that control `C` for a
coordinated corridor**, where the operating hypothesis is that existing
cycles (up to 210 s) are *longer* than throughput-optimal. The measured
effective-rate curves decay after ~40 s of green (vehicles defect from
through lanes into turn bays, weave, and spread out at speed), so — unlike
the textbook model where capacity rises monotonically with `C` — max
throughput has a **finite interior optimum**. The measured decay tail is
what makes this objective well-posed.

## 2. Regime and validity assumptions

- **Oversaturation is sustained** during the analysis periods: arrivals at
  the stop bar are always from the back of a standing queue, never from
  upstream progression. Consequences:
  - The cumulative discharge curve `n_p(t)` is a property of the approach
    (geometry, lane configuration, driver behavior), **not** of the current
    timing plan — so it is valid to evaluate it under a hypothetical `C`.
  - Queue state at green onset transfers: a residual queue exists
    regardless of red duration, so early-green discharge is queue-limited
    either way. (Near saturation this would break, but those phases fall
    into the demand-sufficiency branch, not the curve branch — the model is
    internally consistent.)
- **Curve domain:** a saturated cycle with split `S` measures `n_p(t)` over
  the whole interval `[0, S]`. Under the too-long-cycle hypothesis, the
  modal-split data from long-cycle plans therefore contains every
  shorter candidate split. No extrapolation is ever performed; see the
  boundary-reporting requirement in §5.
- **Turn-bay storage** at the target intersections is long relative to the
  expected splits (> 300 ft vs < 30 s protected splits ≈ a dozen vehicles),
  so spillback/starvation are **reported as outputs, not enforced as
  constraints** in v1.

## 3. Objective (decided — do not re-litigate)

**Pure max throughput**, not reserve capacity and not delay:

```
maximize   Σ_p  3600 · n_p(s_p) / C      over saturated critical phases p
subject to critical-path splits summing to C (per ring/barrier structure),
           s_p ≥ minimum splits (ped walk + clearance, agency minimums),
           unsaturated phases receive exactly-sufficient green:
               n_j(s_j) = q_j · C / 3600   (plus minimums)
```

Rationale, recorded because it was the pivotal scoping decision:

- Reserve capacity = max throughput *holding the served mix pinned to the
  demand mix* (maximize λ s.t. `n_p(s_p) ≥ λ·q_p·C/3600` ∀p). Pure max
  throughput drops the proportionality constraint and lets green flow to
  wherever a marginal second buys the most vehicles.
- Under oversaturation these genuinely differ: pure max throughput will
  drive low-discharge-rate movements (e.g., protected lefts) toward their
  minimum splits. This is **accepted** for the target intersections
  (adequate turn-bay storage; the user wants the heavy flow favored).
- Marginal condition: at the optimum, the **instantaneous rates at
  end-of-green are equalized** across saturated critical phases (subject to
  minimums/sufficiency). The inst-rate profile from the flow module is the
  objective's gradient.
- Optional future knob (v2, not v1): serve each movement at least
  `λ_min ×` its demand share — `λ_min = 0` is pure throughput,
  `λ_min = λ*` recovers reserve capacity.

## 4. Inputs and where they come from

| Input | Source | Status |
|---|---|---|
| Cumulative discharge curves `n_p(t)` + inst rates | `analysis/flow.py` profiles (mean columns) | exists; needs pooling extension (§6.2) |
| Per-phase lost time / clearance | `flow` cycle output (`lost`, `clear_dur`) | exists |
| Saturation state per phase per period | end-slack (`max_lost`) pass-rate | new, thin (§6.3) |
| Demand `q_j` for **unsaturated** phases only | stop-bar counts (`CountEngine`) — accurate where queues clear | exists |
| Ring/barrier concurrency structure | `cycles.r1_phases` / `r2_phases` + `RB_*` config | new analysis (§6.1) |
| Minimum splits (ped walk + clearance) | new config keys (proposed `Min_P{N}_Split` or similar) | new |

Key simplification (decided): saturated phases need only saturation
**detection**, not demand estimation — true demand is unobservable once
queues form, and the objective doesn't need it. Demand estimation is
confined to unsaturated phases, where stop-bar counts are trustworthy.

## 5. Solver shape (decided)

**1-D grid search over `C`** (e.g., 60–220 s), with an inner
split-allocation solve per candidate:

1. Allocate minimums to all phases; exactly-sufficient green to
   unsaturated phases (`n_j(s_j) = q_j·C/3600`).
2. Distribute the remaining critical-path time across saturated critical
   phases by **marginal-rate equalization** (water-filling on the
   instantaneous-rate curves).
3. Score total throughput; record per-phase predicted queues as outputs.

No fixed-point iteration is needed — per-cycle normalization makes the
grid scan exact, and it is transparent and trivially cheap against curves
that are just arrays.

**Boundary reporting is a hard requirement.** The optimizer must
distinguish and report:

- *Interior max* — optimum within measured data for `C` **and for every
  phase's split** (per-phase check: `s*_p ≤` max observed saturated split
  for phase `p`; reallocation can push a minor phase past its own data
  even when `C` shrinks).
- *Boundary max* — throughput still rising at the data edge. Output is a
  directive, not a number: **"optimum exceeds observed range; lengthen and
  re-measure."** This makes the tool an iterative field procedure
  (adjust → collect → re-run) rather than an extrapolator.

## 6. Module decomposition and sequencing

Standard architecture rules apply throughout (functional core /
imperative shell, gap-marker rule, vectorization, CLI parity, engine
pattern mirroring `AogEngine`/`FlowRateEngine`).

### 6.1 Critical movement analysis (build first — standalone value)

`analysis/critical.py` (core) + shell + CLI subcommand.

- Derive ring/barrier concurrency structure from observed sequences
  (`cycles.r1_phases`, `r2_phases`) and `RB_*` config columns.
- Map movement counts to phases (`TM_*` config), produce demand per phase
  for a chosen period.
- Identify the critical phase per concurrent group; critical path per
  barrier group = the ring whose required time is larger.
- Independently useful as a standard ATSPM measure; independently testable.

### 6.2 Flow-module extensions

- **Pooling/stratification:** the modal-split ±10% filter exists to make
  top-Q% selection rank *saturation intensity* rather than split duration
  (at fixed split length, more vehicles = higher rate). Keep it as the
  default. Add a stratified mode for wider curve domains: select top-Q%
  **within** each plan/split-length stratum, then pool survivors into one
  curve. Needed for the boundary case (§5) and for sites without long
  cycles.
- **Curve preparation:** v1 uses the empirical mean profiles directly
  (lightly smoothed; monotone cumulative by construction), derivative from
  the smoothed inst-rate mean, with an explicit no-data boundary marker
  per phase. Defer parametric fits until validation demands them.
- Normalization intent (do not regress): the `end_shift` overhead answers
  "effective rate if the split ended here" — overhead stays in the
  denominator; never shift the time axis (HCM startup-lost-time shifts
  were explicitly rejected as answering a different question).

### 6.3 Saturation classifier (thin)

Per phase per period: fraction of cycles passing the end-slack test
(`lost ≤ max_lost`) above a threshold ⇒ saturated. Threshold chosen
empirically from real distributions once available.

### 6.4 Optimizer

`analysis/optimizer.py` (pure core): inputs = curves, lost times, demands
(unsaturated), structure (from 6.1), constraints; outputs = optimal `C`,
splits, predicted throughput, per-phase v/c and queue estimates,
sensitivity (flatness of the optimum matters as much as its location),
and the interior/boundary result state. Shell engine + `optimize` CLI
subcommand (`--target/--targetid/--all`, `--plans`, date window — reuse
existing period-selection patterns; no peak auto-detection in v1).

Plotting core: throughput-vs-C curve, allocation diagram, marginal-rate
view. Plotly, figure-returning, shell does `write_html`.

## 7. Validation plan

Existing TOD plans are natural experiments: under oversaturation, served
volume differences between plans are attributable to `C` and allocation
alone (arrival patterns are back-of-queue, not progression). The model
must predict observed throughput differences between existing plans
before its recommendations are trusted.

## 8. Out of scope (v1)

- Permissive movements, overlaps, phase resequencing (dual-ring with
  protected lefts and observed sequences only).
- Corridor progression/offset design; re-deriving splits at non-critical
  intersections inheriting the new `C` (conventional methods suffice —
  they are under-saturated).
- Spillback/starvation as *constraints* (reported as outputs only).
- Peak-period auto-detection.
- Modeling downstream absorption of increased throughput (the predicted
  throughput delta is reported so an engineer can sanity-check it).
