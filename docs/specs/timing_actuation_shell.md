# Spec: timing-and-actuation shell, `atspm plot-timing-actuation`, and finding links (UDOT S-D5)

The functional core and the figure already exist and are tested. Opus wrote
them; **do not reimplement or edit any of it**:

- `src/atspm/analysis/timing_actuation.py`: `TIMING_CODES`,
  `timing_actuation_intervals(events_df, window, data_range)`,
  `timing_actuation_rows(roles, intervals, marks, phase_order, phases, detectors)`,
  `ring_phase_order(config)`, `finding_plot_windows(findings, events_df, tz)`.
  Read the module docstring and every public docstring first.
- `src/atspm/plotting/timing_actuation.py`: `plot_timing_actuation(rows,
  intervals, marks, window, tz, metadata, findings)`.
- Both are already exported from `atspm.analysis` / `atspm.plotting`.

Your job is the imperative shell (`TimingActuationEngine`), the
`atspm plot-timing-actuation` CLI subcommand, and a `timing_plot` link column
on detector-health's reported findings.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_timing_actuation_engine.py`. Read it first; it is the
contract. Everything else was green before it was added
(`tests/analysis/test_timing_actuation.py` and
`tests/plotting/test_timing_actuation_plot.py` already pass).

## Files you must NOT modify

- `tests/**` (every test file and fixture)
- `src/atspm/analysis/**`, `src/atspm/plotting/**` (functional core, figures, and
  the package `__init__.py` files there)
- `src/atspm/data/manager.py`, `ingestion.py`, `processing.py`, `reader.py`, and
  every `src/atspm/data/*.py` engine other than the two named below
- `src/atspm/utils/**`
- `docs/UDOT_MOE_ROADMAP.md`, `docs/ROADMAP.md`, `README.md`, and `docs/*.md`
  other than the two named below

## Files you may create or edit

- **create** `src/atspm/data/timing_actuation.py`
- **edit** `src/atspm/data/detector_health.py`: the `timing_plot` column only (§3)
- **edit** `src/atspm/cli.py`: the new subcommand only; don't touch other commands
- **edit** `src/atspm/data/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`: one bullet per touched file, in the format
  given in CLAUDE.md
- **create** `docs/specs/timing_actuation_shell_REPORT.md` (your report)

## 1. `src/atspm/data/timing_actuation.py`

Mirror `src/atspm/data/clock_marks.py` (`ClockMarkEngine`, `get_clock_marks`)
for structure, Google-style docstrings, timezone (`db_timezone` /
`_read_timezone`), `_get_config` and datetime parsing (`_DATETIME_FORMATS`).

```python
class TimingActuationEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def plot(self, start, end, phases=None, detectors=None,
             output_dir=None) -> Dict[str, object]: ...

def get_timing_actuation(db_path, start, end, phases=None, detectors=None,
                         output_dir=None, timezone=None) -> Dict[str, object]: ...
```

`plot()` does the following:

1. Parse `start` and `end` (a string or a naive datetime, in local time) with the
   clock-marks formats. Convert both with `utils.timezone.to_epoch`.
2. **Window cap.** Raise `ValueError` when:
   - `end <= start`;
   - `end - start` is over 4 h with no filter (the message contains `"4 h"`);
   - it is over 24 h even when `phases` or `detectors` is given (the message
     contains `"24 h"`).

   Exactly 4 h is allowed. Module constants: `MAX_WINDOW_S = 4 * 3600` and
   `MAX_NARROWED_WINDOW_S = 24 * 3600`.
3. Fetch `TIMING_CODES` events over `[w0 - FETCH_MARGIN_S, w1 + FETCH_MARGIN_S)`
   with `DatabaseManager.query_events`, where `FETCH_MARGIN_S = 900.0`. Pass that
   fetched range as `data_range` and `(w0, w1)` as `window` to
   `timing_actuation_intervals`. The margin is what lets a detector that was on
   since before the window fill it.
4. Config = `get_config_at_date(start)` (`{}` if none). Then
   `roles = parse_detector_roles(config)` and
   `phase_order = ring_phase_order(config)`.
5. `rows = timing_actuation_rows(roles, intervals, marks, phase_order, phases, detectors)`.
6. **Findings overlay.** Call `DatabaseManager.get_findings(d0, d1)` over the
   local dates the window touches. Apply `apply_ignore(findings,
   wd_ignore(config))`, then `filter_min_severity(..., "low")`, so info
   findings aren't overlaid. If the `detector_findings` table doesn't exist or
   anything raises, overlay nothing (`findings=None`) and carry on.
7. Metadata from `get_metadata()`. Then `fig = plot_timing_actuation(rows,
   intervals, marks, (w0, w1), tz, metadata, findings)`.
8. If `output_dir` is given, create it and `fig.write_html(...)` to
   `TimingActuation_{start:%Y_%m_%d_%H%M}-{end:%Y_%m_%d_%H%M}{suffix}.html`, where
   `start` and `end` are local. `suffix` is `"_P" + "_".join(phases)` when phases
   are given, then `"_D" + "_".join(detectors)` when detectors are given (for
   example `_P4`, `_D53`, `_P2_6_D53`). Print `Wrote <name>`, like the other
   engines do.
9. Return `{"figure", "rows", "intervals", "marks", "findings", "html"}`. `html`
   is the written `Path`, or `None`.

## 2. CLI: `atspm plot-timing-actuation`

Copy `plot-coordination` / `plot-detectors` (`_add_plot_coordination_parser`,
`_plot_detectors_single_intersection`, `handle_plot_detectors`):

- a required mutually exclusive `--target` / `--targetid` / `--all`;
- `--start` and `--end` (required, ISO-8601 local);
- `--phases N [N ...]` and `--detectors N [N ...]` (`type=int`, `nargs="+"`,
  default `None`);
- `--timezone` and `--verbose`.

The handler is named `handle_plot_timing_actuation`. Its output goes to
`<intersection>/outputs/`. A `ValueError` from the window cap becomes `_die(...)`
with its message. With `--all`, one failing intersection must not stop the
others (the same `SystemExit` / `Exception` handling as `handle_plot_detectors`).
Register the parser in `_build_parser` next to the other plot commands. In the
help text, say that the window is capped at 4 h, or at 24 h with `--phases` or
`--detectors`.

## 3. `timing_plot` link on detector-health findings

In `DetectorHealthEngine.detector_health` (`src/atspm/data/detector_health.py`),
after `reported` is built (step 6) and before the outputs are written:

- `links = finding_plot_windows(reported, events_df, self.timezone)`. Use the
  events the engine already fetched; they include Code 82.
- Build a `timing_plot` string column on `reported`:
  `atspm plot-timing-actuation --targetid {intersection_id} --start {local} --end {local}`,
  followed by ` --phases {plot_phase}` when `plot_phase` isn't NA, or
  otherwise ` --detectors {plot_detector}` when that isn't NA. The local times
  use the format `%Y-%m-%dT%H:%M:%S`, converted from epoch with the
  intersection timezone. `intersection_id` comes from `get_metadata()`; when
  it's missing, use `--target {db_path.parent.name}`. Use `""` when
  `plot_start` is NaN. Vectorize it: no `iterrows`.
- The column belongs only to `reported` and to the findings CSV. **Don't** add
  it to the `detector_findings` table: `replace_findings` already ran on
  `all_findings` and must keep doing so unchanged. The heatmap call must
  ignore the extra column.

## 4. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `TimingActuationEngine` and
  `get_timing_actuation`.
- Append these bullets to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new plot-timing-actuation subcommand`
  - `[src/atspm/data/__init__.py] export TimingActuationEngine, get_timing_actuation`
  - `[src/atspm/data/detector_health.py] reported findings / CSV gain timing_plot command column`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output columns differ from what its docstring says;
- the engine golden requires behaviour this spec doesn't describe;
- you'd need a new dependency.

## Report

Write `docs/specs/timing_actuation_shell_REPORT.md`. Include the files you
touched, the test result line, anything you were unsure about, and anywhere
you deviated from this spec. Keep it under 40 lines. Commit everything on your
branch with a descriptive message.
