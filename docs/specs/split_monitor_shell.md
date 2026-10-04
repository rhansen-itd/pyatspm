# Spec: split-monitor shell, plot and `atspm split-monitor` CLI (UDOT S-M2)

The functional core already exists and is tested:

- `src/atspm/analysis/split_monitor.py`: `plan_timeline`, `programmed_at`,
  `split_monitor`, `split_monitor_stats`, `TIMELINE_SCHEMA`, `CYCLE_SCHEMA`,
  `STATS_SCHEMA`, `TERMINATIONS`.

Read its module docstring and function docstrings first. **Don't reimplement
or edit any of it.** In particular, the plan codes 131–149 are a *change log*
(a full dump at local midnight, deltas in between), which is why the shell
needs a 26 h look-back for them.

Your job:

- the imperative shell (`SplitMonitorEngine`);
- the plot, a pure function in `plotting/`;
- the `atspm split-monitor` CLI subcommand.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_split_monitor_engine.py`. Read it first; it is the contract.
Everything else was green before it was added.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**` and `src/atspm/utils/**`
- every `src/atspm/plotting/*.py` other than the new file
- `src/atspm/data/manager.py`, `src/atspm/data/reader.py`, and every
  `src/atspm/data/*.py` other than the new file
- `docs/*.md` other than the two named below, and `README.md`

## Files you may create or edit

- **create** `src/atspm/data/split_monitor.py`
- **create** `src/atspm/plotting/split_monitor.py`
- **edit** `src/atspm/cli.py`: the new subcommand only
- **edit** `src/atspm/data/__init__.py` and `src/atspm/plotting/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/split_monitor_shell_REPORT.md`

## 1. `src/atspm/data/split_monitor.py`

Mirror `src/atspm/data/approach_delay.py` (`ApproachDelayEngine`,
`get_approach_delay`) for structure, Google-style docstrings, timezone
handling, `_parse_range` (a date-only end extends to the end of that day),
`_format_stamp` and the output writer. There is no config to read: phases
are discovered from the data.

```python
_PHASE_SM_CODES = [-1, 1, 4, 5, 6, 8, 9, 10, 11, 12, 21]
_PLAN_SM_CODES = [-1] + list(range(131, 150))
_ALL_SM_CODES = sorted(set(_PHASE_SM_CODES) | set(_PLAN_SM_CODES))
FETCH_MARGIN_S = 1800.0
PLAN_LOOKBACK_S = 26 * 3600.0

class SplitMonitorEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def split_monitor(self, start, end, phases=None,
                      percentiles: Tuple[float, float] = (50, 85),
                      make_plot: bool = True, output_dir=None
                      ) -> Optional[Dict[str, pd.DataFrame]]: ...

def get_split_monitor(db_path, start, end, phases=None, percentiles=(50, 85),
                      make_plot=True, output_dir=None, timezone=None): ...
```

`split_monitor()` does the following:

1. **Fetch twice, then combine.**
   - Phase events: `get_events_with_cycles_df(db, start_dt - margin, end_dt + margin, event_codes=_PHASE_SM_CODES, timezone=tz)`,
     where `margin = timedelta(seconds=FETCH_MARGIN_S)`. The margin lets the
     last green in the window see its clearance end.
   - Plan events: the same call from `start_dt - timedelta(seconds=PLAN_LOOKBACK_S)`
     to `end_dt + margin` with `event_codes=_PLAN_SM_CODES`. The look-back
     reaches the previous local midnight's full dump.
   - Concatenate, `drop_duplicates(subset=["timestamp", "event_code", "parameter"])`
     (the gap markers come back in both), and sort by `timestamp` with
     `kind="stable"`.
   - Empty combined frame: print a warning and return `{}` (`None` when
     `output_dir` is set).
2. **Core.**
   - `timeline = plan_timeline(events)`.
   - `cycle = split_monitor(events, phases=phases, timeline=timeline)`.
3. **Trim to the window.**
   - Keep services whose green falls in the window: `w0 <= green epoch < w1`,
     with `w0, w1` from `utils.timezone.to_epoch` on the parsed range.
     Compare in epoch seconds; don't compare tz-aware timestamps to naive
     datetimes.
   - Keep the timeline rows that overlap the window: `end > w0` and
     `start < w1`, again in epoch seconds.
   - No services left: print a warning containing `no phase services` and
     return `{}` (`None` when `output_dir` is set).
   - For each phase in `phases` (when given) with no services, print a
     warning containing `Ph{N}`.
4. **Stats.** `stats = split_monitor_stats(cycle, percentiles=percentiles)`,
   on the trimmed cycle frame.
5. **Outputs**, only when `output_dir` is given, with
   `stamp = _format_stamp(start_dt, end_dt)`:
   - `SM_Cycle_{stamp}.csv`: the cycle frame (`index=False`);
   - `SM_Stats_{stamp}.csv`: the stats;
   - `SM_Plans_{stamp}.csv`: the trimmed timeline;
   - `SM_Splits_{stamp}.html`: the plot, when `make_plot`. Metadata comes
     from `DatabaseManager.get_metadata()`. `write_html` lives here, never in
     the plot module.

   Print `Wrote <name>` for each file. Return `None` when `output_dir` is
   set; otherwise return `{"cycle": ..., "stats": ..., "timeline": ...}`.

## 2. `src/atspm/plotting/split_monitor.py`

```python
def plot_split_monitor(cycle_df: pd.DataFrame, timeline_df: pd.DataFrame,
                       metadata: Optional[Dict[str, Any]] = None) -> go.Figure: ...
```

This is a pure function, the UDOT split-monitor chart: no I/O and no
`write_html`. Look at `plotting/approach_delay.py` for style.

- **Title:** `_build_title(metadata, suffix="Split Monitor")` from
  `plotting/termination.py`.
- **Layout:** one row per phase in `cycle_df`, in ascending order, with
  `make_subplots(rows=n, cols=1, shared_xaxes=True)` and a y-axis title
  "Split (s)" on each row.
- **Service markers**, x = `green_ts`, y = `split_dur`, one scatter trace
  per (phase, termination) that has rows, named `"Ph{N} Gap Out"`,
  `"Ph{N} Max Out"`, `"Ph{N} Force Off"` or `"Ph{N} Unknown"`.
  - Take the gap/max/force-off colours from `_TERM_STYLES` in
    `plotting/termination.py`; use grey for unknown.
  - Use `legendgroup` = the termination, and show the legend only on that
    termination's first trace.
  - Hover: green time, split, programmed split, and plan.
- **Ped walk:** one trace per phase that has `ped_walk` rows, named
  `"Ph{N} Ped Walk"`. Use open markers drawn over the same points, in one
  colour, with `legendgroup="ped_walk"`.
- **Programmed split:** one line trace per phase, named `"Ph{N} Programmed"`,
  built with the vectorized `[start, end, None]` segment pattern from
  `timeline_df`.
  - Include a timeline row only where `cycle > 0` and `split_{N} > 0`
    (both known); skip the other rows entirely.
  - So y is `[s, s, None, ...]` and x is `[start, end, None, ...]`. Each
    `None` is the Python `None`, not NaN.
  - Draw it as a step line (`line_shape="hv"`). No `iterrows`.
- **Plan bands:** merge consecutive timeline rows that have the same known
  `plan` into runs.
  - Shade each run with `fig.add_vrect` (alternating light fills,
    `layer="below"`, across all rows).
  - Annotate each run at the top as `Plan {plan}`, adding ` (free)` when
    the run's first row has `cycle == 0`.
  - Rows with an NA plan get no band.
  - Clip each band to the x-range of `cycle_df["green_ts"]`, and skip any
    band entirely outside it.
- **Empty input:** an empty `cycle_df` returns a titled, empty `go.Figure`.

## 3. CLI: `atspm split-monitor`

Copy `approach-delay`'s parser and handler pattern
(`_approach_delay_single_intersection` / `handle_approach_delay`), with these
arguments:

| Argument | Notes |
|---|---|
| `--target` / `--targetid` / `--all` | required, mutually exclusive |
| `--start`, `--end` | required, local dates or datetimes |
| `--phases N [N ...]` | `type=int`, default None |
| `--percentiles A B` | `nargs=2, type=float`, default `[50.0, 85.0]`; the two split percentiles in the stats |
| `--no-plot` | flag |
| `--timezone` | |
| `--verbose` | |

- The handler is named `handle_split_monitor`, and the subcommand is
  registered in `_build_parser`.
- Output goes to `<intersection>/outputs/`.
- After writing, print a short summary from the stats CSV: one line per
  (phase, plan) giving the services, the programmed split, the two
  percentiles, and the gap-out / max-out / force-off percentages.
- With `--all`, one failing intersection must not stop the others.
- Add the command to the usage block at the top of `cli.py`, beside `splits`.

## 4. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `SplitMonitorEngine` and `get_split_monitor`.
- `src/atspm/plotting/__init__.py`: export `plot_split_monitor`.
- Append to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new split-monitor subcommand`
  - `[src/atspm/data/__init__.py] export SplitMonitorEngine, get_split_monitor`
  - `[src/atspm/plotting/__init__.py] export plot_split_monitor`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output differs from what its docstring says;
- a golden test looks wrong to you (explain why rather than working around it);
- you'd need a new dependency.

## Report

Write `docs/specs/split_monitor_shell_REPORT.md`. Include the files you
touched, the test result line, anything you were unsure about, and anywhere you
deviated from this spec. Keep it under 30 lines. Commit everything on your
branch.
