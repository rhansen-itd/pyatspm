# Spec: ped delay / wait time shell, plots and `atspm ped-delay` / `atspm wait-time` CLI (UDOT S-M5)

The functional core already exists and is tested:

- `src/atspm/analysis/call_service.py`: `ped_delay`, `wait_time`,
  `summarize_ped_delay`, `summarize_wait_time`, `PED_DELAY_SCHEMA`,
  `WAIT_TIME_SCHEMA`, `PED_SUMMARY_SCHEMA`, `WAIT_SUMMARY_SCHEMA`,
  `PED_KINDS`, `TERMINATIONS`, `DEFAULT_MAX_WAIT_S`.
- `src/atspm/analysis/detector_roles.py`: `parse_detector_roles` /
  `detector_sets` (the `occupancy` role is the stop-bar presence zones).

Read their docstrings first, the module docstring of `call_service.py` in
particular. **Don't reimplement or edit any of it.** `ped_delay` returns a
*tuple* `(delays, reason)`; `wait_time` returns one DataFrame.

Your job:

- the imperative shell (`CallServiceEngine`, two measures);
- two plots, pure functions in one new `plotting/` module;
- two CLI subcommands, `atspm ped-delay` and `atspm wait-time`.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_call_service_engine.py`. Read it first; it is the contract.
Everything else was green before it was added.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**` and `src/atspm/utils/**`
- every `src/atspm/plotting/*.py` other than the new file
- `src/atspm/data/manager.py`, `src/atspm/data/reader.py`, and every
  `src/atspm/data/*.py` other than the new file
- `docs/*.md` other than the two named below, and `README.md`

## Files you may create or edit

- **create** `src/atspm/data/call_service.py`
- **create** `src/atspm/plotting/call_service.py`
- **edit** `src/atspm/cli.py`: the two new subcommands only
- **edit** `src/atspm/data/__init__.py` and `src/atspm/plotting/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/call_service_shell_REPORT.md`

## 1. `src/atspm/data/call_service.py`

Mirror `src/atspm/data/yellow_red_actuations.py` (`YellowRedEngine`,
`get_yellow_red`) for structure, Google-style docstrings, timezone
handling, `_get_config`, `_format_stamp`, `_parse_range` use (a date-only
end extends to the end of that day), the epoch-second window trim and the
output writer. There is **no** data-quality annotation for these measures.

```python
_PED_CODES = [-1, 21, 22, 45, 90]
_WAIT_CODES = [-1, 1, 4, 5, 6, 8, 9, 10, 11, 12, 43, 44]
FETCH_MARGIN_S = 1800.0
DROPPING = ("auto", "on", "off")

class CallServiceEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def ped_delay(self, start, end, phases=None, bin_len: int = 60,
                  make_plot: bool = True, output_dir=None) -> Optional[Dict[str, pd.DataFrame]]: ...
    def wait_time(self, start, end, phases=None, dropping: str = "auto",
                  max_wait: Optional[float] = DEFAULT_MAX_WAIT_S, bin_len: int = 15,
                  make_plot: bool = True, output_dir=None) -> Optional[Dict[str, pd.DataFrame]]: ...

def get_ped_delay(db_path, start, end, phases=None, bin_len=60, make_plot=True,
                  output_dir=None, timezone=None): ...
def get_wait_time(db_path, start, end, phases=None, dropping="auto",
                  max_wait=DEFAULT_MAX_WAIT_S, bin_len=15, make_plot=True,
                  output_dir=None, timezone=None): ...
```

"Return empty" below means: return `{}`, or `None` when `output_dir` is set,
and write no files.

### 1a. `ped_delay()`

1. **Fetch** once:
   `get_events_with_cycles_df(db, start_dt - margin, end_dt + margin, event_codes=_PED_CODES, timezone=tz)`,
   `margin = timedelta(seconds=FETCH_MARGIN_S)`. The margin lets the first
   walk in the window see the clearance that opens its window. Empty →
   warning, return empty.
2. **Core:** `delays, reason = ped_delay(events, phases=phases)`.
   - `reason` not `None` → print
     `Ped delay not computable: {reason}` and return empty. This is the
     normal outcome at sites that don't log ped events (only 315 and 701
     do), so it is **not** an error: don't raise.
   - With `phases` given, print a warning containing `Ph{N}` for each one
     with no rows.
3. **Trim** to rows whose `walk_ts` is in the window: `w0 <= walk epoch < w1`
   (`utils.timezone.to_epoch` on the parsed range; compare epoch seconds,
   never tz-aware vs naive). Nothing left → warning, return empty.
4. **Summaries** on the trimmed frame:
   `binned = summarize_ped_delay(delays, bin_len=bin_len)`,
   `plans = summarize_ped_delay(delays, bin_len=None)`.
5. **Outputs**, only when `output_dir` is given, with
   `stamp = _format_stamp(start_dt, end_dt)`:
   - `PedDelay_Walks_{stamp}.csv`: the delays frame (`index=False`);
   - `PedDelay_{bin_len}min_{stamp}.csv`: the binned summary;
   - `PedDelay_Plans_{stamp}.csv`: the per-plan summary;
   - `PedDelay_Chart_{stamp}.html`: the plot, when `make_plot`
     (`plot_ped_delay(delays, binned, metadata)`; metadata from
     `DatabaseManager.get_metadata()`; `write_html` lives here, never in the
     plot module).

   Print `Wrote <name>` for each file. Return `None` when `output_dir` is
   set; otherwise return `{"delays": ..., "binned": ..., "plans": ...}`
   with the core schemas' columns exactly (no extra columns).

### 1b. `wait_time()`

1. **Dropping.** `dropping` not in `DROPPING` → `raise ValueError` (before
   touching the DB). It picks the phases that get UDOT's dropping algorithm
   (the wait restarts at the first call after the last dropped call):
   - `"auto"`: phases with stop-bar presence zones in config,
     `sorted(detector_sets(parse_detector_roles(config), "occupancy"))`,
     with `config` from `_get_config(start_dt)`. This is UDOT's rule (it
     uses the algorithm when the approach has Stop Bar Presence detection).
   - `"on"`: every phase (`list(range(1, 17))`).
   - `"off"`: none (`None`).

   Print one info line `Ph{N}: dropping algorithm (presence detection)`
   for each phase that has rows and is in the dropping set (Ph{N} with
   its number, ascending).
2. **Fetch** once, as for ped delay, with `_WAIT_CODES`. Empty → warning,
   return empty.
3. **Core:** `windows = wait_time(events, phases=phases, dropping=drop_phases)`.
   With `phases` given, print a warning containing `Ph{N}` for each one
   with no rows.
4. **Trim** to rows whose *time* is in the window, where *time* is
   `green_ts`, or `red_ts` when `green_ts` is missing (a censored last red):
   `w0 <= time epoch < w1`. Nothing left → warning, return empty.
5. **Summaries** on the trimmed frame:
   `binned = summarize_wait_time(windows, bin_len=bin_len, max_wait=max_wait)`,
   `plans = summarize_wait_time(windows, bin_len=None, max_wait=max_wait)`.
6. **Outputs**, as for ped delay, with prefix `WaitTime_`:
   `WaitTime_Windows_{stamp}.csv`, `WaitTime_{bin_len}min_{stamp}.csv`,
   `WaitTime_Plans_{stamp}.csv`, and `WaitTime_Chart_{stamp}.html` from
   `plot_wait_time(windows, binned, metadata, max_wait=max_wait)`.
   Return `{"windows": ..., "binned": ..., "plans": ...}`.

**Dtype care:** `call_ts`, `green_ts` and `wait_s` / `delay_s` hold
NaT/NaN on uncalled and censored rows. Don't `fillna` or cast them in the
returned frames, don't drop those rows (they carry `n_censored` and
`n_uncalled`), and don't `iterrows` anything.

## 2. `src/atspm/plotting/call_service.py`

```python
def plot_ped_delay(delay_df: pd.DataFrame, binned_df: pd.DataFrame,
                   metadata: Optional[Dict[str, Any]] = None) -> go.Figure: ...
def plot_wait_time(wait_df: pd.DataFrame, binned_df: pd.DataFrame,
                   metadata: Optional[Dict[str, Any]] = None,
                   max_wait: Optional[float] = DEFAULT_MAX_WAIT_S) -> go.Figure: ...
```

Pure functions (no I/O, no `write_html`). Look at
`plotting/yellow_red_actuations.py` for style.

Both:

- **Title:** `_build_title(metadata, suffix=...)` from
  `plotting/termination.py`, with suffix `"Pedestrian Delay"` or
  `"Wait Time"`.
- **Layout:** one row per phase present in the row frame, ascending, with
  `make_subplots(rows=n, cols=1, shared_xaxes=True)`. x is time.
- One trace per (phase, category) that has rows; fixed colour per category;
  `legendgroup` = the category; legend shown only on its first trace.
- **Average line** `"Ph{N} Average"` from `binned_df` (x = `time`, y =
  `avg_delay_s` / `avg_wait_s`, rows where it is not NaN), `line_shape="hv"`.
- **Empty input:** an empty row frame returns a titled, empty `go.Figure`.

`plot_ped_delay`, y-axis "Pedestrian delay (s)", markers at x = `walk_ts`:

- `"Ph{N} Delay"`: `kind == "waited"`, y = `delay_s`; hover shows the
  call time and `n_presses`;
- `"Ph{N} In Walk"`: `kind == "in_walk"`, y = 0;
- `"Ph{N} Uncalled Walk"`: `kind == "uncalled"`, y = 0, a distinct
  hollow symbol.

Censored rows are not drawn.

`plot_wait_time`, y-axis "Wait time (s)", markers at x = `green_ts`,
y = `wait_s`, for rows that are `called`, not `censored`, and (when
`max_wait` is not `None`) `wait_s <= max_wait`. One trace per termination
of the green before the red:

- `"Ph{N} Gap Out"`, `"Ph{N} Max Out"`, `"Ph{N} Force Off"`,
  `"Ph{N} Unknown"` (`termination` `gap_out` / `max_out` / `force_off` /
  `unknown`).

Held waits (`held`) use a hollow variant of the same symbol in their trace
(build `marker.symbol` as a vectorized array, not per-row). Hover shows the
red start, the call time and "held" when true.

## 3. CLI: `atspm ped-delay` and `atspm wait-time`

Copy `yellow-red`'s parser and handler pattern
(`_yellow_red_single_intersection` / `handle_yellow_red`).

`ped-delay`:

| Argument | Notes |
|---|---|
| `--target` / `--targetid` / `--all` | required, mutually exclusive |
| `--start`, `--end` | required, local dates or datetimes |
| `--phases N [N ...]` | `type=int`, default None |
| `--bin-len M` | `type=int`, default 60 |
| `--no-plot` | flag |
| `--timezone` | |
| `--verbose` | |

`wait-time`: the same, but `--bin-len` defaults to 15, plus

| Argument | Notes |
|---|---|
| `--dropping {auto,on,off}` | default `auto`; phases using UDOT's dropping algorithm |
| `--max-wait S` | `type=float`, default 360.0; `0` means no cap (pass `None` to the engine) |

- Handlers are named `handle_ped_delay` and `handle_wait_time`; register
  both subcommands in `_build_parser`.
- Output goes to `<intersection>/outputs/`.
- After writing, print a short summary from the plans CSV, one line per
  (phase, plan):
  - ped delay: walks, called, average and maximum delay;
  - wait time: windows, called, held, average wait and the UDOT-comparable
    average (`avg_wait_udot_s`).
- When the engine returns `None` without writing a plans CSV (ped delay not
  computable, or nothing in the window), print nothing more and don't
  raise.
- With `--all`, one failing intersection must not stop the others.
- Add both commands to the usage block at the top of `cli.py`, beside
  `yellow-red`.

## 4. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `CallServiceEngine`, `get_ped_delay`
  and `get_wait_time`.
- `src/atspm/plotting/__init__.py`: export `plot_ped_delay` and
  `plot_wait_time`.
- Append to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new ped-delay and wait-time subcommands`
  - `[src/atspm/data/__init__.py] export CallServiceEngine, get_ped_delay, get_wait_time`
  - `[src/atspm/plotting/__init__.py] export plot_ped_delay, plot_wait_time`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output differs from what its docstring says;
- a golden test looks wrong to you (explain why rather than working around it);
- you'd need a new dependency.

## Report

Write `docs/specs/call_service_shell_REPORT.md`. Include the files you
touched, the test result line, anything you were unsure about, and anywhere you
deviated from this spec. Keep it under 30 lines. Commit everything on your
branch.
