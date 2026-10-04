# Spec: approach volume shell, plot and `atspm approach-volume` CLI (UDOT S-M8)

The functional core already exists and is tested:

- `src/atspm/analysis/approach_volume.py`: `approach_volume`,
  `direction_detectors`, `unparsed_movements`, `BIN_SCHEMA`, `DAY_SCHEMA`,
  `PAIRS`, `COMBINED`, `DEFAULT_BIN_LEN`.
- `src/atspm/data/counts.py`: `CountEngine.vehicle_counts`, which bins the
  Code 82 counts, applies `TM_Exclusions` and labels each bin's
  `data_quality` (gap markers, ingestion coverage).
- `src/atspm/analysis/counts.py`: `parse_movements_from_config`.

Read their docstrings first, the module docstring of `approach_volume.py`
in particular. **Don't reimplement or edit any of it.** `approach_volume`
returns a *tuple* `(bins, days)`. The shell does not count events itself:
it hands `CountEngine.vehicle_counts(..., include_detectors=True)` output
straight to the core.

Your job:

- the imperative shell (`ApproachVolumeEngine`);
- one plot, a pure function in one new `plotting/` module;
- one CLI subcommand, `atspm approach-volume`.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_approach_volume_engine.py`. Read it first; it is the contract.
Everything else was green before it was added.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**` and `src/atspm/utils/**`
- every `src/atspm/plotting/*.py` other than the new file
- `src/atspm/data/manager.py`, `src/atspm/data/reader.py`,
  `src/atspm/data/counts.py`, and every `src/atspm/data/*.py` other than
  the new file
- `docs/*.md` other than the two named below, and `README.md`

## Files you may create or edit

- **create** `src/atspm/data/approach_volume.py`
- **create** `src/atspm/plotting/approach_volume.py`
- **edit** `src/atspm/cli.py`: the new subcommand only
- **edit** `src/atspm/data/__init__.py` and `src/atspm/plotting/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/approach_volume_shell_REPORT.md`

## 1. `src/atspm/data/approach_volume.py`

Mirror `src/atspm/data/green_time_utilization.py` (`GreenTimeEngine`,
`get_green_time`) for structure, Google-style docstrings, timezone
handling, `_read_timezone`, `_get_config`, `_format_stamp`,
`CriticalMovementEngine._parse_range` use (a date-only end extends to the
end of that day; a datetime end is exclusive) and the output writer.

```python
class ApproachVolumeEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def approach_volume(self, start, end, bin_len: int = DEFAULT_BIN_LEN,
                        make_plot: bool = True,
                        output_dir=None) -> Optional[Dict[str, pd.DataFrame]]: ...

def get_approach_volume(db_path, start, end, bin_len=DEFAULT_BIN_LEN,
                        make_plot=True, output_dir=None, timezone=None): ...
```

"Return empty" below means: return `{}`, or `None` when `output_dir` is set,
and write no files.

### Steps of `approach_volume()`

1. **Validate** before touching the DB: `bin_len` must be a positive
   divisor of 60, else `ValueError` (the engine must raise even when the
   DB file does not exist; pass `timezone` in the test so `__init__` needn't
   read it).
2. **Parse** the window: `start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)`.
   Pass these naive datetimes to `CountEngine`; its own `_parse_range`
   accepts date-only strings only, so never pass it the raw strings.
3. **Config** at `start_dt` (`_get_config`):
   `movements = parse_movements_from_config(config)`,
   `dir_dets = direction_detectors(movements)`.
   - `dir_dets` empty → print a warning containing `TM_` (e.g.
     `no TM_{NB|SB|EB|WB}* movement config found. Check TM_* rows in int_cfg.csv.`)
     and return empty.
   - For each label in `unparsed_movements(movements)` print a warning
     containing `TM_{label}` (it has no direction prefix and is ignored).
   - For each detector that appears in more than one direction, print
     `  ⚠️  ApproachVolume: detector {d} is configured in both {A} and {B} (TM_...); it counts in both.`
     The test matches the substring
     `detector 4 is configured in both EB and WB`; directions in `PAIRS`
     order (NB, SB, EB, WB). (701's `TM_EBL` and `TM_WBR` share
     channel 54 today.)
4. **Counts:**
   `counts = CountEngine(self.db_path, self.timezone).vehicle_counts(start_dt, end_dt, bin_len=bin_len, include_detectors=True)`.
   Empty → print a warning, return empty. Don't pass `exclude_missing`
   (the core needs the partial bins' labels) and don't `fillna`/cast it.
5. **Silent detectors:** for each direction (in `PAIRS` order), the
   configured detectors with no column in `counts` or a column summing to
   zero. When any, print
   `  ⚠️  ApproachVolume: {dir} detector(s) {a, b} logged no actuation in the window.`
   (detector IDs ascending, comma-separated; the test matches
   `WB detector(s) 5 logged no actuation`). This is informational; the
   direction still runs. At 201 the NB/EB TM channels log only on
   2026-03-18..21, and at 313 the EB/WB ones never do.
6. **Core:** `bins, days = approach_volume(counts, movements, bin_len=bin_len)`.
   Both empty → return empty.
7. **K-factor note:** when no row of `days` has `complete_day`, print
   `  ℹ️  ApproachVolume: no complete day in the window; K-factor not reported.`
   (the test matches `K-factor`).
8. **Outputs**, only when `output_dir` is given, with
   `stamp = _format_stamp(start_dt, end_dt)`, `index=False`, and a
   `Wrote <name>` line each:
   - `AV_Bins_{bin_len}min_{stamp}.csv`, `AV_Days_{stamp}.csv`
   - `AV_Chart_{stamp}.html` when `make_plot`:
     `plot_approach_volume(bins, days, metadata)`, metadata from
     `DatabaseManager.get_metadata()`. `write_html` lives here, never in
     the plot module.

   Return `None` when `output_dir` is set; otherwise `{"bins", "days"}`
   with the core schemas' columns exactly (no extra columns), exactly as
   the core returned them.

**Dtype care:** `days` has nullable `Int64` columns and NaT/NaN on days
with no complete hour; `bins.vph` and `bins.d_split` are NaN on
incomplete bins. Don't `fillna`, cast or drop them, and don't `iterrows`
anything.

## 2. `src/atspm/plotting/approach_volume.py`

```python
def plot_approach_volume(bins_df: pd.DataFrame, days_df: pd.DataFrame,
                         metadata: Optional[Dict[str, Any]] = None) -> go.Figure: ...
```

Pure function (no I/O, no `write_html`). Look at
`plotting/green_time_utilization.py` and `plotting/call_service.py` for
style. UDOT draws one chart per direction pair: each direction's hourly
volume and the combined volume as step lines, and each direction's
D-factor (share of the combined volume) dashed on a 0–1 secondary axis.
Build the same, one subplot row per pair.

- **Title:** `_build_title(metadata or {}, suffix="Approach Volume")` from
  `plotting/termination.py`.
- **Layout:** one row per pair present in `bins_df`, in `PAIRS` order
  (`NB/SB`, then `EB/WB`), `make_subplots(rows=n, cols=1,
  shared_xaxes=True, specs=[[{"secondary_y": True}]] * n)`, subplot
  titles = the pair strings. Primary y "Volume (veh/h)", secondary y
  "Directional split", range `[0, 1]`. x is time.
- **Traces,** for each pair, from the rows of `bins_df` with that `pair`
  and `direction` (select with boolean masks; never pivot or merge: there
  is exactly one row per (time, pair, direction)):
  - `"{dir} Volume"` for each configured direction of the pair, y = `vph`;
  - `"{pair} Combined"`, y = `vph` of the `combined` rows;
  - `"{dir} D-Factor"`, y = `d_split`, `line.dash="dash"`, on the
    secondary axis; only when **both** directions of the pair have rows
    (a one-sided pair has no D-factor trace).
  All are `go.Scatter(mode="lines", line_shape="hv", connectgaps=False)`,
  with x = `time` and y the column values unchanged (NaN on an incomplete
  bin breaks the line). Each trace name appears once in the figure.
  Colours: first direction red, second blue, combined green (UDOT), the
  D-factor traces in their direction's colour. Hover shows time, the
  value and the trace name.
- **Peak hour shading:** for each row of `days_df` with
  `direction == "combined"` and a non-NaT `peak_start`, one
  `fig.add_vrect(x0=peak_start, x1=peak_start + 1 h, row=<pair's row>, col=1)`,
  translucent, with a short annotation (`"Peak {HH:MM} · K {k:.3f}"`, or
  without the K part when `k_factor` is NaN). Iterate with `zip` over the
  columns, not `iterrows`. Nothing else may add to `layout.shapes`.
- **Empty input:** an empty `bins_df` returns a titled, empty `go.Figure`.

## 3. CLI: `atspm approach-volume`

Copy `green-time`'s parser and handler pattern
(`_green_time_single_intersection` / `handle_green_time`).

| Argument | Notes |
|---|---|
| `--target` / `--targetid` / `--all` | required, mutually exclusive |
| `--start`, `--end` | required, local dates or datetimes |
| `--bin-len M` | `type=int`, default 15; time bin, minutes (must divide 60) |
| `--no-plot` | flag |
| `--timezone` | |
| `--verbose` | |

- Handler `handle_approach_volume`, registered in `_build_parser`;
  `args.func`, `args.bin_len`, `args.no_plot`.
- Output goes to `<intersection>/outputs/`.
- After writing, print a short summary from `AV_Days_*.csv`, one line per
  (date, pair) from its `combined` row, e.g.
  `2025-06-02 NB/SB  peak 16:00 116 veh  PHF 0.95  K 0.154  D NB 0.93 / SB 0.86`.
  The peak time is `HH:MM` local (parse `peak_start` with
  `pd.to_datetime(..., utc=True)` and convert to the CLI's timezone; the
  CSV holds offset-aware strings), `-` for any NA value. The D values come
  from the pair's direction rows on that date, omitted for a one-sided
  pair.
- When the engine wrote no days CSV (no TM config, nothing in the
  window), print nothing more and don't raise.
- With `--all`, one failing intersection must not stop the others.
- Add the command to the usage block at the top of `cli.py`, beside
  `green-time`.

## 4. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `ApproachVolumeEngine` and `get_approach_volume`.
- `src/atspm/plotting/__init__.py`: export `plot_approach_volume`.
- Append to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new approach-volume subcommand`
  - `[src/atspm/data/__init__.py] export ApproachVolumeEngine, get_approach_volume`
  - `[src/atspm/plotting/__init__.py] export plot_approach_volume`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output differs from what its docstring says;
- a golden test looks wrong to you (explain why rather than working around it);
- you'd need a new dependency.

## Report

Write `docs/specs/approach_volume_shell_REPORT.md`. Include the files you
touched, the test result line, anything you were unsure about, and anywhere you
deviated from this spec. Keep it under 30 lines. Commit everything on your
branch.
