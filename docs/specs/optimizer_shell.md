# Spec: optimizer shell engine, plots and `atspm optimize` CLI (ROADMAP optimizer steps 4–5)

The pure solver already exists and is tested: `src/atspm/analysis/optimizer.py`.
Read its module docstring and the `optimize()` docstring first. Then read
`docs/design_optimizer_solver.md` sections D6 and D7, **including the two
"Amended" notes** (D7 §6.3 and D4). Where this spec and the design doc
differ, this spec wins. Your job is the imperative shell around the solver,
three pure Plotly figures, and a CLI subcommand.

## Acceptance

`.venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_optimizer_engine.py`. Read that file before you start: it
builds a synthetic intersection and states every behaviour below as a test.

## Files you must NOT modify

- `tests/**` (every test file)
- `src/atspm/analysis/**` (the functional core: `optimizer.py`, `flow.py`, `critical.py`, …)
- `src/atspm/data/` modules other than the new `optimizer.py` and `__init__.py`
- `src/atspm/plotting/` modules other than the new `optimizer.py` and `__init__.py`
- `src/atspm/reports/**`
- `docs/ROADMAP.md`, `README.md`, `docs/design_*.md`, and every other doc not named below

## Files you may create or edit

- **create** `src/atspm/data/optimizer.py`
- **create** `src/atspm/plotting/optimizer.py`
- **edit** `src/atspm/cli.py`: the new subcommand only, plus its line in the module docstring's command list. Don't touch other commands.
- **edit** `src/atspm/data/__init__.py`, `src/atspm/plotting/__init__.py` (exports only)
- **append** to `docs/PENDING_DOC_CHANGES.md` (one bullet per touched file, format in `CLAUDE.md`)
- **create** `docs/specs/optimizer_shell_REPORT.md` (your report)

## 1. `src/atspm/data/optimizer.py`

Mirror `src/atspm/data/critical.py` (`CriticalMovementEngine`) for structure,
Google-style docstrings, timezone resolution (`db_timezone`), config lookup
(`DatabaseManager.get_config_at_date`), `_parse_range` (copy its behaviour:
datetime end is exclusive, a date-only end extends to end of day) and
`_write_outputs` (same sub-day-aware stamp rule).

```python
class OptimizerEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None
    def optimize(
        self, start, end, saturated: List[int], plans=None,
        pct=1.0, split_tolerance=0.10, stratify=False,
        max_lost=10.0, sat_threshold=0.8,
        demand_stat="mean", default_min_split=10.0,
        c_min=60.0, c_max=220.0, c_step=1.0,
        flat_tol_pct=1.0, boundary_rate_tol=100.0,
        bin_len=15, exclude_missing=True,
        make_plot=True, output_dir=None,
    ) -> Optional[Dict[str, object]]

def get_optimization(db_path, start, end, saturated, ..., timezone=None)  # same kwargs, convenience wrapper
```

Steps inside `optimize`, in order:

1. **Validate arguments.** An empty `saturated` raises `ValueError` mentioning
   `saturated`. A `demand_stat` other than `"mean"`/`"peak"` raises `ValueError`
   mentioning `demand_stat`.
2. **Parse the range** and get the config at `start`. If there is no config,
   print a warning and return `{}` (or `None` when `output_dir` is set). Every
   later "return empty" follows the same rule.
3. **Structure.** Call `_query_cycles(db_path, start_epoch, end_epoch)` (from
   `atspm.data.reader`) and then `ring_barrier_structure(config, cycles_df)`.
   If `cycles_df` is empty, print a warning and return empty. The solver
   raises without served phases.
4. **Declaration check.** For each declared phase that isn't in the
   structure, print a line containing `Ph{N}` and say it is ignored. Build
   `saturated_map = {p: True for p in declared}` and pass it to the solver
   as-is; phases outside the structure are harmless there.
5. **Events.** Load once with
   `get_events_with_cycles_df(db_path, start_dt, end_dt, event_codes=_ALL_FLOW_CODES, timezone=self.timezone)`,
   importing `_ALL_FLOW_CODES` from `atspm.data.flow`. Don't redefine it.
6. **Detectors.** Use `_parse_stopbar_sets(config)` from `atspm.analysis.critical`
   (it accepts both `Det_P{N}_Stopbar` and `Det_P{N}_Stop_Bar`). Keep only
   structure phases, as `{phase: sorted(dets)}`.
7. **Per phase with detectors:**
   `cycle_df, vehicle_df = flow_rate(events, ph, dets, max_lost=None, plans=plans)`,
   then
   `_, profile = discharge_profiles(cycle_df, vehicle_df, pct=pct, split_tolerance=split_tolerance, stratify=stratify)`
   with every other argument left at its default. A non-empty `profile` goes
   into `curves[ph]`. Also collect every phase's `cycle_df` for step 8.
   - For each **declared** phase with no curve, print a line containing
     `Ph{N}` and the hint text `--pct` (e.g. "no discharge curve for Ph2 —
     raise --pct or widen the window").
8. **Advisory.** Run `saturation_state(pd.concat(cycle_dfs), max_lost=max_lost, threshold=sat_threshold)`.
   Print one line containing the word `advisory` and `Ph{N}` for each phase
   that is declared but advisory-unsaturated, and for each phase that is
   advisory-saturated but not declared. It is information only and never
   changes `saturated_map`.
9. **Demand.** Follow `CriticalMovementEngine.critical`'s steps, but don't call
   it (its summary printing is noise here):
   `CountEngine(db_path, tz).vehicle_counts(start_dt, end_dt, bin_len=bin_len, hourly=True, exclude_missing=exclude_missing)`
   → `movement_phase_map(config)` → `phase_demand(counts_df, movement_map)`.
   `demand_vph = {phase: value}` from the `demand_vph` column (or `peak_vph`
   when `demand_stat == "peak"`), finite values only. If counts are empty,
   print a warning and use `{}` (the solver then pins unsaturated phases at
   their minimums).
10. **Minimum splits.** For every structure phase with a non-NaN
    `barrier_group`: read config key `Min_P{N}_Split` as a float, in seconds,
    the full split including clearance. If it is missing, empty or
    unparseable, use `default_min_split` and print one warning line naming
    the key (e.g. `Min_P8_Split`).
11. **Solve.** Call `optimize(curves, structure_df, saturated_map, demand_vph, min_splits, c_min=…, c_max=…, c_step=…, flat_tol_pct=…, boundary_rate_tol=…)`.
    On `ValueError`, print it and return empty.
12. **Summary print.** Print `state`, `c_star`, the flat range, each warning,
    and the directive when present, in the style of the other engines.
13. **Result dict:** the solver's five keys (`scan`, `splits`, `optimum`,
    `directive`, `warnings`) plus:
    - `curves`: `{phase: profile_df}`;
    - `saturation`: the advisory DataFrame;
    - `demand`: the `phase_demand` DataFrame;
    - `min_splits`: `{phase: float}`;
    - `figures`: `{"curve", "allocation", "marginal"}` → `go.Figure`, only when `make_plot` (otherwise an empty dict).
14. **`output_dir`** set: write the files and return `None`. `{stamp}` follows `critical.py`.
    - `Optimize_Scan_{stamp}.csv`, `Optimize_Splits_{stamp}.csv`,
      `Optimize_Saturation_{stamp}.csv`, all with `index=False`.
    - `Optimize_Summary_{stamp}.csv`: one row from the `optimum` dict. Its
      dict-valued fields (`binding_ring`, `group_min_at_c_max`) are written
      as JSON strings (`json.dumps`, with plain-int/float keys and values,
      not numpy types). Add a `directive` column: the directive as a JSON
      string, or empty.
    - When `make_plot`, write `Optimize_Curve_{stamp}.html`,
      `Optimize_Allocation_{stamp}.html` and `Optimize_Marginal_{stamp}.html`
      with `fig.write_html`. That is the only place `write_html` may appear.
    - Metadata for plot titles comes from `DatabaseManager(...).get_metadata()`.

No `iterrows()` and no row loops over events.

## 2. `src/atspm/plotting/optimizer.py` (functional core: pure, no I/O)

Reuse `_build_title` with `from .flow import _build_title`; don't copy it.
Don't use `fig.add_shape`, `add_vline`, `add_hline`, `add_vrect` or any
other layout shape: build everything as traces. No dummy legend traces.
Hover colours match their traces. Every function must return a valid figure
for an infeasible result (all-NaN throughput, NaN `c_star`, NaN `s_star`)
and for empty inputs.

- `plot_throughput_curve(scan_df, optimum, metadata) -> go.Figure`
  - A line trace named exactly `"Saturated throughput"`: x = `C`, y =
    `throughput_sat_vph`, feasible rows only.
  - A dotted line trace for `throughput_total_vph`, named
    `"Total throughput"`.
  - A marker trace named exactly `"C*"` at `(c_star, throughput_sat_vph)`
    when `c_star` is finite.
  - A filled trace named exactly `"Flat band"` spanning
    `[flat_c_lo, flat_c_hi]` over the plot's y-range (`fill="toself"`).
  - Infeasible C values as a filled trace named `"Infeasible"`, when any exist.
  - Title suffix: `"Throughput vs Cycle Length"`. Append the state, e.g.
    `" (boundary)"`, when it isn't `interior`.
- `plot_allocation(splits_df, optimum, metadata) -> go.Figure`
  - A horizontal bar per ring, drawn with the vectorized
    `[start, end, None]` segment pattern: y = `"Ring {r}"`, x from 0 to `c_star`.
  - Order phases by barrier group, then structure order within the ring.
    Use `splits_df` row order within a (group, ring); it is sorted by phase.
    Each phase's segment is `s_star` long, so each ring sums to `c_star`.
  - **One trace per `allocation_basis` value present**, named exactly by that
    value (`"optimized"`, `"sufficiency"`, `"minimum"`). Excluded phases and
    NaN `s_star` draw nothing.
  - Hover shows phase, `s_star`, `s_min`, `s_domain`, and the flags.
  - Extra marker traces (for example `s_min` ticks) are allowed under other names.
  - Title suffix: `"Split Allocation at C*"`.
- `plot_marginal_rates(curves, splits_df, metadata) -> go.Figure`
  - For each **optimized** phase with a curve, a line trace named
    `"Ph{N}"`: x = `t`, y = `inst`.
  - One marker trace named exactly `"End of split"`, with one point per
    optimized phase at `(s_star, end_inst_rate_vph)`.
  - Title suffix: `"Marginal Discharge Rates"`.

## 3. CLI: `atspm optimize`

Mirror the `flow` and `critical` subcommands: a `_optimize_single_intersection`
/ `handle_optimize` / `_add_optimize_parser` triple, with the batch loop and
per-target try/except copied from `handle_critical`. Register the parser in
`_build_parser()` right after the `critical` parser.

- A required, mutually exclusive group: `--target` / `--targetid` / `--all`.
- `--start`, `--end`: required, with formats as in `critical`.
- `--saturated N [N ...]`: required, `type=int`, `nargs="+"`. With `--all`,
  the same list applies to every intersection; print a one-line note saying so.
- `--plans` (`nargs="+"`, int), `--pct` (1.0), `--split-tolerance` (0.10),
  `--stratify` (store_true), `--max-lost` (10.0), `--sat-threshold` (0.8),
  `--demand-stat {mean,peak}` (mean), `--default-min-split` (10.0),
  `--c-min` (60.0), `--c-max` (220.0), `--c-step` (1.0),
  `--flat-tol-pct` (1.0), `--bin-len` (15), `--include-missing` (store_true;
  passed as `exclude_missing=not args.include_missing`, as in `critical`),
  `--no-plot`, `--timezone`, `--verbose`.
- Outputs go to `intersections/<target>/outputs/` via `output_dir`.
- Help text: the description says saturated phases are the engineer's
  declaration, and the end-slack classifier is printed as an advisory only.

Out of scope here: `--validate` (roadmap step 6) and `--output`.

## Stop and ask (write it in the report and stop) if

- any existing test fails for a reason you'd have to change a forbidden file to fix;
- the solver's output columns differ from its docstring or from D6;
- an acceptance test looks wrong to you. Explain why in the report rather than working around it;
- you'd need a new dependency.

## Report

Write `docs/specs/optimizer_shell_REPORT.md`: the files you touched, the
test result line, anything you were unsure about, and anything you deviated
from in this spec. Keep it under 40 lines. Commit everything on your branch
with a descriptive message.
