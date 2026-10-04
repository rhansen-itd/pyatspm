# Spec: green time utilization shell, plot and `atspm green-time` CLI (UDOT S-M7)

The functional core already exists and is tested:

- `src/atspm/analysis/green_time_utilization.py`: `green_time_utilization`,
  `summarize_gtu_bins`, `summarize_gtu_splits`, `CYCLE_SCHEMA`,
  `ACTUATION_SCHEMA`, `BIN_SCHEMA`, `SPLIT_SCHEMA`, `DEFAULT_BIN_S`.
- `src/atspm/analysis/split_monitor.py`: `plan_timeline` (the programmed
  splits, Codes 131–149).
- `src/atspm/analysis/detector_roles.py`: `parse_detector_roles` /
  `detector_sets` / `phase_overlaps`.

Read their docstrings first, the module docstring of
`green_time_utilization.py` in particular. **Don't reimplement or edit any
of it.** `green_time_utilization` returns a *tuple* `(cycles, actuations)`.

Your job:

- the imperative shell (`GreenTimeEngine`);
- one plot, a pure function in one new `plotting/` module;
- one CLI subcommand, `atspm green-time`.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_green_time_engine.py`. Read it first; it is the contract.
Everything else was green before it was added.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**` and `src/atspm/utils/**`
- every `src/atspm/plotting/*.py` other than the new file
- `src/atspm/data/manager.py`, `src/atspm/data/reader.py`, and every
  `src/atspm/data/*.py` other than the new file
- `docs/*.md` other than the two named below, and `README.md`

## Files you may create or edit

- **create** `src/atspm/data/green_time_utilization.py`
- **create** `src/atspm/plotting/green_time_utilization.py`
- **edit** `src/atspm/cli.py`: the new subcommand only
- **edit** `src/atspm/data/__init__.py` and `src/atspm/plotting/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/green_time_shell_REPORT.md`

## 1. `src/atspm/data/green_time_utilization.py`

Mirror `src/atspm/data/yellow_red_actuations.py` (`YellowRedEngine`,
`get_yellow_red`) for structure, Google-style docstrings, timezone
handling, `_get_config`, `_format_stamp`, the role validation and
`Det_P{N}_{Role}` warnings, `CriticalMovementEngine._parse_range` use (a
date-only end extends to the end of that day), the epoch-second window
trim and the output writer. Mirror `src/atspm/data/split_monitor.py` for
the two-fetch combine (phase events with a margin, plan events with a 26 h
lookback, concatenated, de-duplicated on
`(timestamp, event_code, parameter)`, sorted).

```python
_GTU_CODES = [-1, 1, 8, 9, 10, 11, 12, 61, 63, 64, 65, 82]
_PLAN_CODES = [-1] + list(range(131, 150))
FETCH_MARGIN_S = 1800.0
PLAN_LOOKBACK_S = 26 * 3600.0
ROLES = ("stop_bar", "occupancy")

class GreenTimeEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def green_time(self, start, end, phases=None, role: str = "stop_bar",
                   bin_s: float = DEFAULT_BIN_S, bin_len: int = 15,
                   use_overlap: bool = False, use_exclusions: bool = True,
                   max_green_s: Optional[float] = 120.0,
                   make_plot: bool = True, output_dir=None) -> Optional[Dict[str, pd.DataFrame]]: ...

def get_green_time(db_path, start, end, phases=None, role="stop_bar",
                   bin_s=DEFAULT_BIN_S, bin_len=15, use_overlap=False,
                   use_exclusions=True, max_green_s=120.0, make_plot=True,
                   output_dir=None, timezone=None): ...
```

"Return empty" below means: return `{}`, or `None` when `output_dir` is set,
and write no files.

### Steps of `green_time()`

1. **Validate** before touching the DB: `role` not in `ROLES` →
   `ValueError`; `bin_s <= 0` → `ValueError`.
2. **Config** at `start_dt` (`_get_config`):
   `sets = detector_sets(parse_detector_roles(config), role)`;
   `exclusions = parse_exclusions_from_config(config)` when
   `use_exclusions`, else `None`.
   - The role label for messages is `Stop_Bar` / `Occupancy`. With
     `phases` given, warn (containing `Ph{N}` and `Det_P{N}_{label}`) for
     each phase with no detectors in the role, and skip it.
   - No phase left → print a warning containing
     `Det_P{N}_{label}` (the literal braces, e.g.
     `no stop_bar detector config ... Check Det_P{N}_Stop_Bar rows in int_cfg.csv.`),
     and return empty. This is the normal outcome at a site without that
     role (701 has only Pairs and Arrival detectors), **not** an error.
   - **Overlaps:** only when `use_overlap`, take
     `overlaps = phase_overlaps(config)`; a phase with an entry runs in
     overlap mode, and prints `Ph{N}: green of overlap {letter}`. Without
     `use_overlap` every phase runs in phase mode, even if
     `Det_P{N}_Overlap` is configured: the programmed split belongs to the
     phase, and an FYA overlap's green is mostly permissive service (see
     the core's module docstring).
3. **Fetch** twice and combine, as split monitor does:
   `get_events_with_cycles_df(db, start_dt - margin, end_dt + margin, event_codes=_GTU_CODES, timezone=tz)`
   and the same with `start_dt - lookback` and `_PLAN_CODES`. Both empty →
   warning, return empty.
4. **Core**, once for the plans then per phase (ascending):
   `timeline = plan_timeline(events)`;
   `cy, ac = green_time_utilization(events, ph, sorted(dets), bin_s=bin_s, overlap=..., exclusions=exclusions, timeline=timeline)`.
5. **Trim** each phase's frames to greens in the window: rows of `cy` and
   of `ac` whose `green_ts` satisfies `w0 <= epoch < w1`
   (`utils.timezone.to_epoch` on the parsed range; compare epoch seconds,
   never tz-aware vs naive). A phase with nothing left prints a warning
   with `Ph{N}` and is skipped. No phase left → return empty.
6. **Concatenate** (`[CYCLE_SCHEMA]`, `[ACTUATION_SCHEMA]`; an empty
   actuation frame keeps its columns) and **summarise**:
   - `bins = summarize_gtu_bins(cycle, acts, bin_s=bin_s, bin_len=bin_len)`
   - `splits = summarize_gtu_splits(cycle, bin_len=bin_len)`
   - `plan_bins = summarize_gtu_bins(cycle, acts, bin_s=bin_s, bin_len=None)`
   - `plan_splits = summarize_gtu_splits(cycle, bin_len=None)`
7. **Outputs**, only when `output_dir` is given, with
   `stamp = _format_stamp(start_dt, end_dt)`, `index=False`, and a
   `Wrote <name>` line each:
   - `GTU_Cycle_{stamp}.csv`, `GTU_Actuations_{stamp}.csv`
   - `GTU_Bins_{bin_len}min_{stamp}.csv`, `GTU_Splits_{bin_len}min_{stamp}.csv`
   - `GTU_PlanBins_{stamp}.csv`, `GTU_PlanSplits_{stamp}.csv`
   - `GTU_Chart_{stamp}.html` when `make_plot`:
     `plot_green_time(bins, splits, metadata, max_green_s=max_green_s)`,
     metadata from `DatabaseManager.get_metadata()`. `write_html` lives
     here, never in the plot module.

   Return `None` when `output_dir` is set; otherwise
   `{"cycle", "actuations", "bins", "splits", "plan_bins", "plan_splits"}`
   with the core schemas' columns exactly (no extra columns).

**Dtype care:** `actuations` and `programmed_split` are `Int64` with NA on
censored or free-mode rows; `yellow_ts` is NaT on unpaired greens. Don't
`fillna` or cast them in the returned frames, don't drop censored rows
(they carry `n_censored`), and don't `iterrows` anything.

## 2. `src/atspm/plotting/green_time_utilization.py`

```python
def plot_green_time(bins_df: pd.DataFrame, splits_df: pd.DataFrame,
                    metadata: Optional[Dict[str, Any]] = None,
                    max_green_s: Optional[float] = 120.0) -> go.Figure: ...
```

Pure function (no I/O, no `write_html`). Look at
`plotting/yellow_red_actuations.py` and `plotting/call_service.py` for
style. UDOT's chart is a heat map with two step lines; build the same.

- **Title:** `_build_title(metadata, suffix="Green Time Utilization")` from
  `plotting/termination.py`.
- **Layout:** one row per phase present in `bins_df`, ascending,
  `make_subplots(rows=n, cols=1, shared_xaxes=True)`; x is time, y is
  "Seconds into green". Subplot titles `Ph{N}`.
- **Heat map** `"Ph{N} Utilization"`, one `go.Heatmap` per phase:
  pivot (vectorized: `pivot_table` / `pivot`, never a loop over rows) of
  `act_per_cycle` with **rows = `bin_start_s`** ascending and **columns =
  `time`** ascending; `y` = the `bin_start_s` values, `x` = the times,
  `z` the pivot (missing cells NaN). When `max_green_s` is not `None`,
  keep only bins with `bin_start_s < max_green_s` (free-mode rest-in-green
  produces greens of 1000 s and more; the CSVs keep everything). All
  phases share one colour axis (`coloraxis="coloraxis"`, a sequential
  scale starting at 0, colour bar titled "Actuations / cycle").
  `customdata` carries the matching pivots of `n_reached` and `flow_vph`
  so the hover shows time, bin start, actuations per cycle, cycles
  reaching the bin and flow (veh/h of green).
- **Lines** from `splits_df` for the phase, `line_shape="hv"`,
  x = `time`, rows where the value is not NaN:
  `"Ph{N} Average Green"` (y = `avg_green_s`, solid) and
  `"Ph{N} Programmed Green"` (y = `programmed_green`, dashed). A trace
  with no rows is omitted. `legendgroup` = "Average Green" /
  "Programmed Green"; legend shown only on the first phase's trace.
- **Empty input:** an empty `bins_df` returns a titled, empty `go.Figure`.

## 3. CLI: `atspm green-time`

Copy `yellow-red`'s parser and handler pattern
(`_yellow_red_single_intersection` / `handle_yellow_red`).

| Argument | Notes |
|---|---|
| `--target` / `--targetid` / `--all` | required, mutually exclusive |
| `--start`, `--end` | required, local dates or datetimes |
| `--phases N [N ...]` | `type=int`, default None |
| `--role {stop_bar,occupancy}` | default `stop_bar` (UDOT's lane-by-lane count loops) |
| `--bin-s S` | `type=float`, default 2.0; seconds-into-green bin |
| `--bin-len M` | `type=int`, default 15; time bin, minutes |
| `--overlap` | flag; measure each phase's `Det_P{N}_Overlap` overlap instead |
| `--no-exclusions` | flag |
| `--max-green S` | `type=float`, default 120.0; plot only; `0` means no cap (pass `None`) |
| `--no-plot` | flag |
| `--timezone` | |
| `--verbose` | |

- Handler `handle_green_time`, registered in `_build_parser`; `args.func`,
  `args.role`, `args.bin_s`, `args.bin_len`, `args.overlap`,
  `args.no_exclusions`, `args.max_green`, `args.no_plot`, `args.phases`.
- Output goes to `<intersection>/outputs/`.
- After writing, print a short summary from `GTU_PlanSplits_*.csv`, one
  line per (phase, plan), starting `Ph{N}`: cycles, average green,
  programmed green (or `-` when NaN), actuations per cycle.
- When the engine wrote no plan-splits CSV (no role config, nothing in the
  window), print nothing more and don't raise.
- With `--all`, one failing intersection must not stop the others.
- Add the command to the usage block at the top of `cli.py`, beside
  `yellow-red`.

## 4. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `GreenTimeEngine` and `get_green_time`.
- `src/atspm/plotting/__init__.py`: export `plot_green_time`.
- Append to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new green-time subcommand`
  - `[src/atspm/data/__init__.py] export GreenTimeEngine, get_green_time`
  - `[src/atspm/plotting/__init__.py] export plot_green_time`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output differs from what its docstring says;
- a golden test looks wrong to you (explain why rather than working around it);
- you'd need a new dependency.

## Report

Write `docs/specs/green_time_shell_REPORT.md`. Include the files you
touched, the test result line, anything you were unsure about, and anywhere you
deviated from this spec. Keep it under 30 lines. Commit everything on your
branch.
