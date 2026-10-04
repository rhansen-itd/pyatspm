# Spec: yellow/red actuations shell, plot and `atspm yellow-red` CLI (UDOT S-M4)

The functional core already exists and is tested:

- `src/atspm/analysis/yellow_red_actuations.py`: `yellow_red_actuations`,
  `summarize_yellow_red`, `CYCLE_SCHEMA`, `ACTUATION_SCHEMA`,
  `SUMMARY_SCHEMA`, `STATES`, `DEFAULT_SEVERE_SEC`.
- `src/atspm/analysis/detector_roles.py`: `phase_overlaps` (the
  `Det_P{N}_Overlap` key), beside `parse_detector_roles` / `detector_sets`.
- `src/atspm/analysis/counts.py`: `parse_exclusions_from_config` (the
  `TM_Exclusions` JSON).

Read their docstrings first. **Don't reimplement or edit any of it.**
`yellow_red_actuations` returns a *tuple* `(cycles, actuations)`.

Your job:

- the imperative shell (`YellowRedEngine`);
- the plot, a pure function in `plotting/`;
- the `atspm yellow-red` CLI subcommand.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_yellow_red_engine.py`. Read it first; it is the contract.
Everything else was green before it was added.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**` and `src/atspm/utils/**`
- every `src/atspm/plotting/*.py` other than the new file
- `src/atspm/data/manager.py`, `src/atspm/data/reader.py`, and every
  `src/atspm/data/*.py` other than the new file
- `docs/*.md` other than the two named below, and `README.md`

## Files you may create or edit

- **create** `src/atspm/data/yellow_red_actuations.py`
- **create** `src/atspm/plotting/yellow_red_actuations.py`
- **edit** `src/atspm/cli.py`: the new subcommand only
- **edit** `src/atspm/data/__init__.py` and `src/atspm/plotting/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/yellow_red_actuations_shell_REPORT.md`

## 1. `src/atspm/data/yellow_red_actuations.py`

Mirror `src/atspm/data/approach_delay.py` (`ApproachDelayEngine`,
`get_approach_delay`) for structure, Google-style docstrings, timezone
handling, `_get_config`, `_format_stamp` and the output writer, and use
`CriticalMovementEngine._parse_range` (a date-only end extends to the end of
that day). There is **no** data-quality annotation for this measure: skip
`_add_quality` / `exclude_missing`.

```python
_ALL_YRA_CODES = [-1, 1, 8, 9, 10, 11, 12, 61, 63, 64, 65, 82]
FETCH_MARGIN_S = 1800.0
ROLES = ("stop_bar", "occupancy")

class YellowRedEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def yellow_red(self, start, end, phases=None, role: str = "stop_bar",
                   severe_sec: float = DEFAULT_SEVERE_SEC, bin_len: int = 15,
                   use_exclusions: bool = True, make_plot: bool = True,
                   output_dir=None) -> Optional[Dict[str, pd.DataFrame]]: ...

def get_yellow_red(db_path, start, end, phases=None, role="stop_bar",
                   severe_sec=DEFAULT_SEVERE_SEC, bin_len=15, use_exclusions=True,
                   make_plot=True, output_dir=None, timezone=None): ...
```

`yellow_red()` does the following:

1. **Role.** `role` not in `ROLES` → `raise ValueError` (before touching the DB).
   - Why `stop_bar` is the default (put this in the module docstring):
     `Det_P{N}_Stop_Bar` channels are short count loops just *past* the stop
     line, so an actuation in red means a vehicle crossed the line.
     `Det_P{N}_Occupancy` zones sit *at* the line and fire for every vehicle
     that stops on red (315, 2025-12-15, P6: 1.3 red actuations per cycle on
     the zones vs 0.02 on the loops). `occupancy` is offered for sites whose
     only stop-line detection is presence.
2. **Config** at `start_dt`:
   - `sets = detector_sets(parse_detector_roles(config), role)`;
   - `overlaps = phase_overlaps(config)`;
   - `exclusions = parse_exclusions_from_config(config) if use_exclusions else None`.
   - Phase selection exactly as approach delay: with `phases` given, keep
     those in `sets` and print a warning containing `Ph{N}` for each one
     missing; with no detectors at all, print a warning naming the
     `Det_P{N}_{Stop_Bar|Occupancy}` key and return `{}` (`None` when
     `output_dir` is set).
3. **Fetch** once:
   `get_events_with_cycles_df(db, start_dt - margin, end_dt + margin, event_codes=_ALL_YRA_CODES, timezone=tz)`,
   `margin = timedelta(seconds=FETCH_MARGIN_S)`. The margin lets the last
   cycle in the window see its next green. Empty → warning, return `{}`
   (`None` with `output_dir`).
4. **Core**, per phase in ascending order:
   `cy, ac = yellow_red_actuations(events, ph, sorted(dets), severe_sec=severe_sec, overlap=overlaps.get(ph), exclusions=exclusions)`.
   - Trim **both** frames to cycles whose green is in the window:
     `w0 <= green epoch < w1` (`utils.timezone.to_epoch` on the parsed range;
     compare epoch seconds, never tz-aware vs naive). For `ac`, use its
     `green_ts` column, so every kept actuation belongs to a kept cycle.
   - A phase with no cycles left: warning containing `Ph{N}`, skip it.
   - When the phase has an overlap, print one info line
     `Ph{N}: classified against overlap {X}` (X as a letter, 1 → A).
   - No phase has cycles: return `{}` (`None` with `output_dir`).
5. **Summaries** on the trimmed, concatenated cycle frame:
   - `binned = summarize_yellow_red(cycle, bin_len=bin_len)`;
   - `plans = summarize_yellow_red(cycle, bin_len=None)`.
6. **Outputs**, only when `output_dir` is given, with
   `stamp = _format_stamp(start_dt, end_dt)`:
   - `YRA_Cycle_{stamp}.csv`: the cycle frame (`index=False`);
   - `YRA_Actuations_{stamp}.csv`: the actuation frame;
   - `YRA_{bin_len}min_{stamp}.csv`: the binned summary;
   - `YRA_Plans_{stamp}.csv`: the per-plan summary;
   - `YRA_Chart_{stamp}.html`: the plot, when `make_plot`
     (`plot_yellow_red(cycle, actuations, metadata, severe_sec=severe_sec)`;
     metadata from `DatabaseManager.get_metadata()`; `write_html` lives
     here, never in the plot module).

   Print `Wrote <name>` for each file. Return `None` when `output_dir` is
   set; otherwise return
   `{"cycle": ..., "actuations": ..., "binned": ..., "plans": ...}` with the
   core schemas' columns exactly (no extra columns).

**Dtype care:** the count columns are nullable `Int64` (NA on censored
cycles). Don't cast them to `int`, don't `fillna(0)` them in the returned
frames, and don't `iterrows` anything.

## 2. `src/atspm/plotting/yellow_red_actuations.py`

```python
def plot_yellow_red(cycle_df: pd.DataFrame, act_df: pd.DataFrame,
                    metadata: Optional[Dict[str, Any]] = None,
                    severe_sec: float = DEFAULT_SEVERE_SEC) -> go.Figure: ...
```

A pure function, UDOT's Yellow and Red Actuations chart: no I/O, no
`write_html`. Look at `plotting/approach_delay.py` and
`plotting/split_monitor.py` for style.

- **Title:** `_build_title(metadata, suffix="Yellow and Red Actuations")`
  from `plotting/termination.py`.
- **Layout:** one row per phase in `cycle_df`, ascending, with
  `make_subplots(rows=n, cols=1, shared_xaxes=True)`; y-axis title
  "Time since start of yellow (s)" on each row. x is time.
- **Actuation markers**, x = `timestamp`, y = `t_yellow`. Green actuations
  (`state == "green"`) are **not** drawn. The rest are split into four
  categories, one scatter trace per (phase, category) that has rows:
  - `"Ph{N} Yellow"`: `state == "yellow"`;
  - `"Ph{N} Red Clearance"`: `state == "red_clear"` and not `severe`;
  - `"Ph{N} Red"`: `state == "red"` and not `severe`;
  - `"Ph{N} Severe"`: `severe` (any state).
  Fixed colours per category (amber, orange, red, dark red), `legendgroup`
  = the category, legend shown only on its first trace. Hover: time,
  detector, state, `t_red` ("s into red").
- **Reference lines**, one trace per phase each, from the uncensored cycles
  only, x = `yellow_ts`, built with the vectorized `[x, x_next, None]`
  segment pattern (each `None` is the Python `None`, not NaN; no row
  iteration). Each segment spans one cycle, from its `yellow_ts` to the next
  uncensored cycle's `yellow_ts` (the last one to its own `red_end_ts`):
  - `"Ph{N} Red Clearance Begin"`: y = `yellow_dur`;
  - `"Ph{N} Red Begin"`: y = `yellow_dur + red_clear_dur`;
  - `"Ph{N} Severe Threshold"`: y = `yellow_dur + severe_sec`, dashed.
  Use `line_shape="hv"`.
- **Empty input:** empty `cycle_df` returns a titled, empty `go.Figure`.

## 3. CLI: `atspm yellow-red`

Copy `approach-delay`'s parser and handler pattern
(`_approach_delay_single_intersection` / `handle_approach_delay`), with:

| Argument | Notes |
|---|---|
| `--target` / `--targetid` / `--all` | required, mutually exclusive |
| `--start`, `--end` | required, local dates or datetimes |
| `--phases N [N ...]` | `type=int`, default None |
| `--role {stop_bar,occupancy}` | default `stop_bar`; which detector role is classified |
| `--severe-sec S` | `type=float`, default 4.0; severe = more than S s after red start |
| `--bin-len M` | `type=int`, default 15 |
| `--no-exclusions` | flag; ignore `TM_Exclusions` |
| `--no-plot` | flag |
| `--timezone` | |
| `--verbose` | |

- The handler is named `handle_yellow_red`; register the subcommand in
  `_build_parser`.
- Output goes to `<intersection>/outputs/`.
- After writing, print a short summary from the plans CSV: one line per
  (phase, plan) with cycles, violations, severe, violations per cycle, and
  `pct_violations` as a percentage.
- With `--all`, one failing intersection must not stop the others.
- Add the command to the usage block at the top of `cli.py`, beside
  `approach-delay`.

## 4. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `YellowRedEngine` and `get_yellow_red`.
- `src/atspm/plotting/__init__.py`: export `plot_yellow_red`.
- Append to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new yellow-red subcommand`
  - `[src/atspm/data/__init__.py] export YellowRedEngine, get_yellow_red`
  - `[src/atspm/plotting/__init__.py] export plot_yellow_red`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output differs from what its docstring says;
- a golden test looks wrong to you (explain why rather than working around it);
- you'd need a new dependency.

## Report

Write `docs/specs/yellow_red_actuations_shell_REPORT.md`. Include the files you
touched, the test result line, anything you were unsure about, and anywhere you
deviated from this spec. Keep it under 30 lines. Commit everything on your
branch.
