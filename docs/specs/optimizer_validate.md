# Spec: `OptimizerEngine.validate` and `atspm optimize --validate` (ROADMAP optimizer step 6)

The validation math already exists and is tested:
`src/atspm/analysis/optimizer_validation.py` (`validate_plans`, `valid_cycles`).
Read its module docstring and the `validate_plans` docstring first. Then read
`docs/design_optimizer_solver.md` D8, **including its "Amended 2026-10-02"
note**. Where this spec and the design doc differ, this spec wins. Your job
is the imperative shell that loads the data, calls `validate_plans`, prints
and writes the result, and a CLI flag.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the
new `tests/data/test_optimizer_validate.py`. Read that file before you start:
it builds a two-plan synthetic intersection and states every behaviour below
as a test.

## Files you must NOT modify

- `tests/**` (every test file)
- `src/atspm/analysis/**` (the functional core, including `optimizer_validation.py`)
- `src/atspm/data/` modules other than `optimizer.py` and `__init__.py`
- `src/atspm/plotting/**`, `src/atspm/reports/**`
- `docs/ROADMAP.md`, `README.md`, `docs/design_*.md`, and every other doc not named below

## Files you may create or edit

- **edit** `src/atspm/data/optimizer.py`: add `validate` and `get_validation`. Don't change `optimize`'s behaviour.
- **edit** `src/atspm/cli.py`: the `optimize` subcommand only (its parser, `_optimize_single_intersection`, and its module-docstring line if needed).
- **edit** `src/atspm/data/__init__.py` (export `get_validation` beside the optimizer's existing exports, if it has any)
- **append** to `docs/PENDING_DOC_CHANGES.md` (one bullet per touched file, format in `CLAUDE.md`)
- **create** `docs/specs/optimizer_validate_REPORT.md` (your report)

## 1. `OptimizerEngine.validate` in `src/atspm/data/optimizer.py`

```python
def validate(
    self, start, end, saturated: List[int], plans=None,
    pct=1.0, split_tolerance=0.10, max_lost=10.0, sat_threshold=0.8,
    min_plan_cycles=30, split_cover_tol=1.0,
    rank_deadband_pct=2.0, change_tol_pp=3.0,
    output_dir=None,
) -> Optional[Dict[str, object]]

def get_validation(db_path, start, end, saturated, ..., timezone=None)  # same kwargs, convenience wrapper
```

Import `validate_plans` into the module namespace with
`from ..analysis.optimizer_validation import validate_plans` and call it by
that name: the tests monkeypatch `atspm.data.optimizer.validate_plans`.

Steps, in order:

1. **Arguments.** An empty `saturated` raises `ValueError` mentioning `saturated`.
2. **Range and config.** `self._parse_range(start, end)`, then the config at
   `start`. With no config, print a warning and return `{}` (or `None` when
   `output_dir` is set). Every later "return empty" follows the same rule.
3. **Cycles.** `_query_cycles(db_path, start_epoch, end_epoch)` with
   `to_epoch(..., self.timezone)` bounds, exactly as `optimize` does. If it is
   empty, print a warning and return empty. If `plans` is given, keep only
   rows whose `coord_plan` is in `plans`.
4. **Events, as UTC epoch floats.** The core joins on `cycle_start` and needs
   epoch floats, so do **not** pass `timezone=` to the reader. Convert the
   bounds to aware UTC datetimes yourself:
   `datetime.fromtimestamp(start_epoch, tz=timezone.utc)` (and the same for
   end). Then call
   `get_events_with_cycles_df(db_path, start_utc, end_utc, event_codes=_ALL_FLOW_CODES)`.
   Aware bounds keep their own offset, and with no `timezone` the frame comes
   back as epoch floats. This keeps a `--timezone` override honoured.
   `gap_ts` is `events.loc[events["event_code"] == -1, "timestamp"].to_numpy()`.
5. **Phases under test.** Use `_parse_stopbar_sets(config)`. For each declared
   phase without stop-bar detectors, print a line containing `Ph{N}` saying it
   has no stop-bar detectors and is not validated. If no declared phase is
   left, return empty. Unlike `optimize`, don't filter by ring/barrier
   structure; validation needs only the detectors.
6. **Flow.** For each phase left, in ascending order:
   `flow[p] = flow_rate(events, p, sorted(dets), max_lost=None, plans=plans)`.
7. **Core.** `res = validate_plans(flow, cycles_df, gap_ts=gap_ts, pct=…, split_tolerance=…, max_lost=…, sat_threshold=…, min_plan_cycles=…, split_cover_tol=…, rank_deadband_pct=…, change_tol_pp=…)`.
   Pass every argument by keyword. Leave `grid_step` and `min_cycles` at
   their defaults.
8. **Console summary.** In the style of `optimize`'s summary:
   - one line per plan: plan, `n_cycles`, `c_median`, `observed_vph`,
     `insample_pct_error`;
   - one line per pair: anchor → target, status, and for tested pairs the
     observed and predicted change (%) and the error (pp);
   - the line `Validation: {verdict}` (exactly that text, e.g.
     `Validation: FAIL`), followed by `mean_abs_change_error_pp` against
     `change_tol_pp`;
   - each warning on its own line.
9. **Return** `res` unchanged (keys `plans`, `pairs`, `verdict`), or write and
   return `None` when `output_dir` is set.
10. **`output_dir`** set: write with `index=False`, stamped as in `_write_outputs`
    (factor the stamp rule into a helper both use rather than copying it):
    - `Optimize_Validation_Plans_{stamp}.csv` from `res["plans"]`;
    - `Optimize_Validation_Pairs_{stamp}.csv` from `res["pairs"]`;
    - `Optimize_Validation_Summary_{stamp}.csv`: one row from `res["verdict"]`,
      with `phases` and `warnings` written as JSON strings (`json.dumps`, plain
      Python types; reuse `_to_json_compatible`).

No `iterrows()` and no row loops over events. Printing one line per plan or
per pair from the small result frames is fine.

## 2. CLI: `atspm optimize --validate`

In `_add_optimize_parser`, add:

- `--validate` (store_true): run the validation **instead of** the optimizer.
- `--min-plan-cycles` (int, default 30), `--split-cover-tol` (float, 1.0),
  `--rank-deadband-pct` (float, 2.0), `--change-tol-pp` (float, 3.0), each
  with a one-line help that says it applies to `--validate` only.

In `_optimize_single_intersection`, when `args.validate` is set, call
`engine.validate(...)` with `start`, `end`, `saturated`, `plans`, `pct`,
`split_tolerance`, `max_lost`, `sat_threshold`, the four new values and
`output_dir`, and **don't** call `engine.optimize`. Change the banner's first
line to say validation. Error handling stays as it is. `--saturated` stays
required. Add a sentence to the subcommand's description: `--validate` tests
the throughput model against the existing TOD plans before its
recommendations are trusted.

## Stop and ask (write it in the report and stop) if

- any existing test fails for a reason you'd have to change a forbidden file to fix;
- `validate_plans` behaves differently from its docstring;
- an acceptance test looks wrong to you. Explain why in the report rather than working around it;
- you'd need a new dependency.

## Report

Write `docs/specs/optimizer_validate_REPORT.md`: the files you touched, the
test result line, anything you were unsure about, and anything you deviated
from in this spec. Keep it under 40 lines. Commit everything on your branch
with a descriptive message.
