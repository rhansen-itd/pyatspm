# Spec: approach-delay shell, plot and `atspm approach-delay` CLI (UDOT S-M3)

The functional core already exists and is tested:

- `src/atspm/analysis/approach_delay.py`: `approach_delay`, `bin_approach_delay`,
  `CYCLE_SCHEMA`, `BIN_SCHEMA`.
- `src/atspm/analysis/detector_roles.py`: `arrival_travel_times(config)`,
  `parse_detector_roles`, `detector_sets`.

Read the module docstring of `approach_delay.py` and the docstrings of these
functions first. **Don't reimplement or edit any of them.**

Your job:

- the imperative shell (`ApproachDelayEngine`);
- the plot, a pure function in `plotting/`;
- the `atspm approach-delay` CLI subcommand.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_approach_delay_engine.py`. Read it first; it is the contract.
Everything else was green before it was added.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**` and `src/atspm/utils/**`
- every `src/atspm/plotting/*.py` other than the new file
- `src/atspm/data/manager.py`, `src/atspm/data/reader.py`, and every
  `src/atspm/data/*.py` other than the new file
- `docs/*.md` other than the two named below, and `README.md`

## Files you may create or edit

- **create** `src/atspm/data/approach_delay.py`
- **create** `src/atspm/plotting/approach_delay.py`
- **edit** `src/atspm/cli.py`: the new subcommand only
- **edit** `src/atspm/data/__init__.py` and `src/atspm/plotting/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/approach_delay_shell_REPORT.md`

## 1. `src/atspm/data/approach_delay.py`

Mirror `src/atspm/data/split_failures.py` (`SplitFailureEngine`,
`get_split_failures`) for structure, Google-style docstrings, timezone,
`_parse_range`, `_format_stamp`, `_add_quality` / `_drop_missing_days`, the
phase-discovery warnings and the output writer. That file itself mirrors
`data/aog.py`.

```python
_ALL_AD_CODES = [-1, 1, 8, 9, 10, 11, 12, 82]   # sorted; gap marker always included
FETCH_MARGIN_S = 1800.0

class ApproachDelayEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def approach_delay(self, start, end, phases=None, travel_time_sec: float = 0.0,
                       bin_len: Union[int, str] = 15, exclude_missing: bool = False,
                       make_plot: bool = True, output_dir=None
                       ) -> Optional[Dict[str, pd.DataFrame]]: ...

def get_approach_delay(db_path, start, end, phases=None, travel_time_sec=0.0,
                       bin_len=15, exclude_missing=False, make_plot=True,
                       output_dir=None, timezone=None): ...
```

`approach_delay()` does the following:

1. **Range and config.**
   - Parse the range as `SplitFailureEngine` does (a date-only end extends to
     the end of that day).
   - Read the config at the start date.
   - Build the phase→detectors map from
     `detector_sets(parse_detector_roles(config), "arrival")`, filtered by
     `phases`.
   - Warnings, worded as `SplitFailureEngine` words them but for the key
     `Det_P{N}_Arrival`:
     - a requested phase with no key: print a warning naming
       `Det_P{N}_Arrival`;
     - no arrival config at all: print a warning containing `Arrival` and
       return `{}` (or `None` when `output_dir` is set).
2. **Travel time per phase.**
   - Call `tt = arrival_travel_times(config)`. Let a `ValueError` from it
     propagate.
   - If a phase is in `tt`, pass `tt[phase]` (a `{detector: seconds}`
     mapping) to the core, with source `"config"`.
   - Otherwise pass the scalar `travel_time_sec`, with source `"offset"`,
     and print one warning per such phase naming `Det_P{N}_Arrival_Travel`
     and the offset used.
3. **Fetch with a margin.**
   - `get_events_with_cycles_df(db, start_dt - margin, end_dt + margin, event_codes=_ALL_AD_CODES, timezone=tz)`,
     where `margin = timedelta(seconds=FETCH_MARGIN_S)`.
   - The margin lets the first green in the window see the red before it,
     and the last one its yellow.
4. **Core, then trim.**
   - Per phase, call
     `approach_delay(events, phase, dets, travel_time_sec=...)`.
   - Keep the rows whose serving green falls in the window:
     `w0 <= green epoch < w1`, with `w0, w1` from `utils.timezone.to_epoch`
     on the parsed range. Compare in epoch seconds; don't compare tz-aware
     timestamps to naive datetimes.
   - Append two columns:
     - `travel_time_s`: the scalar, or the mean of the mapping's values;
     - `travel_source`: `"config"` or `"offset"`.
   - A configured phase that yields no rows prints a warning containing
     `Ph{N}`, as `SplitFailureEngine` does, and is skipped.
   - Concatenate the phases. The cycle frame's columns are
     `CYCLE_SCHEMA + ["travel_time_s", "travel_source"]`.
5. **Binned.**
   - Skip this step when `bin_len == "cycle"`; the result then has no
     `"binned"` key.
   - Otherwise, `bin_approach_delay(cycle_df[CYCLE_SCHEMA], bin_len=int(bin_len))`.
   - Then add `coverage` / `data_quality` with the same `_add_quality`
     logic as `SplitFailureEngine`, over the *unmargined* range `[start_dt, end_dt)`.
6. **Outputs**, only when `output_dir` is given, with
   `stamp = _format_stamp(start_dt, end_dt)`:
   - `AD_Cycle_{stamp}.csv`, the cycle frame (`index=False`);
   - `AD_{N}min_{stamp}.csv`, the binned frame (not in cycle mode);
   - `AD_Delay_{stamp}.html`, the plot, when `make_plot`. The plot is drawn
     from the binned frame. In cycle mode it is drawn from
     `bin_approach_delay(cycle_df[CYCLE_SCHEMA], 15)`, and that frame is not
     written. Metadata comes from `DatabaseManager.get_metadata()`.
     `write_html` lives here, never in the plot module.

   Print `Wrote <name>` for each file. Return `None` when `output_dir` is
   set; otherwise return `{"cycle": ..., "binned": ...}`.

## 2. `src/atspm/plotting/approach_delay.py`

```python
def plot_approach_delay(binned_df: pd.DataFrame,
                        metadata: Optional[Dict[str, Any]] = None) -> go.Figure: ...
```

This is a pure function, the UDOT approach-delay chart: no I/O and no
`write_html`. Look at `plotting/split_failures.py` for style.

- **Title:** `_build_title(metadata, suffix="Approach Delay")` from
  `plotting/termination.py`.
- **Layout:** one row per phase, in ascending order. Use
  `make_subplots(rows=n, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]] * n)`.
- **Recombine plans first.** A bin that spans a plan change has one row per
  plan. Group by `(phase, time)` before plotting:
  - `delay/veh = Σ total_delay_s / Σ arrivals`, NaN when there are no
    arrivals;
  - `veh-h/h = Σ delay_vh_per_hr`.
  - Do this vectorized: no `iterrows`.
- **Traces per phase**, both lines with markers, x = `time`:
  - `"Ph{N} delay/veh"`: delay per vehicle in seconds, on the primary y.
  - `"Ph{N} veh-h/h"`: total delay in vehicle-hours per hour, on the
    secondary y.
  - Axis titles: "Delay per vehicle (s)" and "Delay (veh-h/h)". Keep each
    trace's hover in the trace's own colour.
- **Plan bands:**
  - Take the plan of each `(phase, time)` from the row with the most
    `n_cycles`, falling back to the first row.
  - Take runs of consecutive times with the same plan from the
    lowest-numbered phase.
  - Shade each run with `fig.add_vrect` (alternating light fills,
    `layer="below"`, across all rows) and give each run an annotation
    reading `Plan {int(plan)}` at the top.
  - Each vrect spans `[run's first time, run's last time + bin width]`, where
    the bin width is the smallest positive time step in the frame (15 min if
    there is only one time).
- **Empty input:** an empty `binned_df` returns a titled, empty `go.Figure`.

## 3. CLI: `atspm approach-delay`

Copy `split-failures`' parser and handler pattern (`_split_failures_single_intersection`
/ `handle_split_failures`), with these arguments:

| Argument | Notes |
|---|---|
| `--target` / `--targetid` / `--all` | required, mutually exclusive |
| `--start`, `--end` | required, local dates or datetimes |
| `--phases N [N ...]` | `type=int`, default None |
| `--offset SEC` | `type=float`, default `0.0`; help: "travel time from the advance detector to the stop line, used for phases without a Det_P{N}_Arrival_Travel key" |
| `--bin-len` | default `"15"`; minutes or `cycle` |
| `--exclude-missing` | flag |
| `--no-plot` | flag |
| `--timezone` | |
| `--verbose` | |

- The handler is named `handle_approach_delay`, and the subcommand is
  registered in `_build_parser`.
- Output goes to `<intersection>/outputs/`.
- After writing, print a short per-phase summary from the cycle CSV:
  uncensored cycles, censored cycles, AoG/AoY/AoR percentages, delay per
  vehicle (s), and the travel source.
- With `--all`, one failing intersection must not stop the others.
- Add the command to the usage block at the top of `cli.py`, beside `aog`.

## 4. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `ApproachDelayEngine` and
  `get_approach_delay`.
- `src/atspm/plotting/__init__.py`: export `plot_approach_delay`.
- Append to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new approach-delay subcommand`
  - `[src/atspm/data/__init__.py] export ApproachDelayEngine, get_approach_delay`
  - `[src/atspm/plotting/__init__.py] export plot_approach_delay`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output differs from what its docstring says;
- a golden test looks wrong to you (explain why rather than working around it);
- you'd need a new dependency.

## Report

Write `docs/specs/approach_delay_shell_REPORT.md`. Include the files you
touched, the test result line, anything you were unsure about, and anywhere you
deviated from this spec. Keep it under 30 lines. Commit everything on your
branch.
