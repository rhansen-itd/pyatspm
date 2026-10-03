# Spec: split-failure shell engine, scatter plot and `atspm split-failures` CLI (UDOT S-M1)

The pure core already exists and is tested: `src/atspm/analysis/split_failures.py`
(read its module docstring and the `split_failures` / `bin_split_failures`
docstrings first). Your job is the imperative shell around it, a pure Plotly
figure, and a CLI subcommand.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_split_failures_engine.py`.

## Files you must NOT modify

- `tests/**` (every test file, fixtures included)
- `src/atspm/analysis/**` (the functional core)
- `src/atspm/data/ingestion.py`, `processing.py`, `manager.py`, `reader.py`, `aog.py`, `flow.py`, `critical.py`
- `src/atspm/reports/**`
- `docs/ROADMAP.md`, `docs/UDOT_MOE_ROADMAP.md`, `README.md`, `docs/*.md` other than the two named below

## Files you may create or edit

- **create** `src/atspm/data/split_failures.py`
- **create** `src/atspm/plotting/split_failures.py`
- **edit** `src/atspm/cli.py` (new subcommand only; don't touch other commands)
- **edit** `src/atspm/data/__init__.py`, `src/atspm/plotting/__init__.py` (exports only)
- **append** to `docs/PENDING_DOC_CHANGES.md` (one bullet per touched file, format in CLAUDE.md)
- **create** `docs/specs/split_failures_shell_REPORT.md` (your report)

## 1. `src/atspm/data/split_failures.py`

Mirror `src/atspm/data/aog.py` (`AogEngine`) for structure, docstrings (Google
style), timezone (`db_timezone`), config lookup, `_add_quality` /
`_drop_missing_days` (copy the AOG behaviour; the binned frame's time column is
`time`). Use `CriticalMovementEngine._parse_range` semantics from
`data/critical.py` for start/end (date or `"%Y-%m-%d %H:%M"`; date-only end =
whole day; a datetime end is exclusive) and its `_write_outputs` window
stamp rule (`2025_06_02_0700-2025_06_02_0900`, or `2025_06_02-2025_06_02` for a
whole day). Import those helpers or the `_DATETIME_FORMATS` constants rather
than re-typing them where practical.

```python
_ALL_SF_CODES: List[int]   # exactly [-1, 1, 8, 9, 10, 11, 12, 81, 82] (sorted)

class SplitFailureEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None
    def split_failures(self, start, end, phases=None, aggregate="union",
                       threshold=0.79, ror_seconds=5.0, include_yellow=False,
                       bin_len=60, exclude_missing=False, make_plot=True,
                       output_dir=None) -> Optional[Dict[str, pd.DataFrame]]

def get_split_failures(db_path, start, end, phases=None, aggregate="union",
                       threshold=0.79, ror_seconds=5.0, include_yellow=False,
                       bin_len=60, exclude_missing=False, make_plot=True,
                       output_dir=None, timezone=None)   # convenience wrapper
```

Behaviour of `split_failures`:

1. Validate `aggregate` first: not `"union"`/`"mean"` → `ValueError`.
2. Config at `start` (`DatabaseManager.get_config_at_date`). Stop-bar lanes per
   phase come from `atspm.analysis.critical._parse_stopbar_sets(config)`
   (accepts both `Det_P{N}_Stop_Bar` and `Det_P{N}_Stopbar`). **Import it; do
   not write a new parser.** Filter to `phases` when given; for each requested
   phase with no key print a warning containing `Det_P{N}_Stop_Bar`. If nothing
   is left, print a warning containing `Stop_Bar` and return `{}` (or `None`
   when `output_dir` is set).
3. Load events once with `get_events_with_cycles_df(db_path, start_dt, end_dt,
   event_codes=_ALL_SF_CODES, timezone=self.timezone)`. Gap markers MUST be in
   the code list. Empty → warning, return `{}`/`None`.
4. For each phase (sorted) call the core `split_failures(events, phase,
   sorted(dets), threshold=..., aggregate=..., ror_seconds=...,
   include_yellow=...)`. A phase whose cycle frame is empty prints a warning
   containing `Ph{N}` and is skipped. Concatenate the cycle frames and the lane
   frames (`ignore_index=True`). Add a column `aggregate` (the string) to the
   cycle frame. Nothing left → return `{}`/`None`.
5. Unless `bin_len == "cycle"`: `binned = bin_split_failures(cycle, int(bin_len))`,
   then the AOG-style quality annotation (`coverage`, `data_quality`) and
   `exclude_missing` handling.
6. Print a short summary per phase (cycles, fails, SF %, lanes), in the style
   of the other engines.
7. Return `{"cycle", "lane", "binned"}` (no `"binned"` in cycle mode), unless
   `output_dir` is set: then write and return `None`:
   - `SF_Cycle_{stamp}_{aggregate}.csv` (cycle frame, `index=False`)
   - `SF_Lane_{stamp}.csv` (lane frame; it does not depend on the aggregate)
   - `SF_{bin_len}min_{stamp}_{aggregate}.csv` (binned; skipped in cycle mode)
   - `SF_Scatter_{stamp}_{aggregate}.html` from `plot_split_failures(cycle,
     metadata, threshold=threshold)` via `fig.write_html` (the only place
     `write_html` may appear), only when `make_plot` is true.
     Metadata: `DatabaseManager(...).get_metadata()`.

No `iterrows()` or row loops over events.

## 2. `src/atspm/plotting/split_failures.py` (functional core: pure, no I/O)

```python
def plot_split_failures(cycle_df, metadata=None, threshold=0.79) -> go.Figure
```

- One scatter subplot per phase (`make_subplots`, up to 4 columns, wrapping),
  x = `gor`, y = `ror5`, both axes 0–1, titled `Phase {N}`.
- Per phase two marker traces named exactly `"Ph{N} pass"` and `"Ph{N} fail"`
  (split on `fail`); omit a trace when it would be empty. Hover shows the
  local green time, `gor`, `ror5`, `n_lanes`, `n_lanes_failed`; hover colour
  matches the trace.
- Threshold guides: **one** trace per subplot named `"Threshold"` (legend shown
  once) drawing the vertical line x = threshold and the horizontal line
  y = threshold with the vectorized `[a, b, None]` segment pattern.
  **No `fig.add_shape`, `add_vline` or `add_hline`** (no layout shapes).
- Title: reuse `_build_title` from `atspm.plotting.termination` (import it,
  don't copy it) with suffix `f"Split Failures — GOR vs ROR5 ({aggregate})"`,
  where `aggregate` is the frame's `aggregate` column value (default `"union"`
  when absent).
- An empty frame returns a valid empty figure.

## 3. CLI: `atspm split-failures`

Mirror the `aog` subcommand (`_aog_single_intersection`, `handle_aog`,
`_add_aog_parser`) and register `_add_split_failures_parser(subs)` in
`_build_parser()` right after `_add_aog_parser(subs)`. Add the line to the
module docstring's command list.

- Mutually exclusive required group `--target` / `--targetid` / `--all`.
- `--start`, `--end` (required; accept `YYYY-MM-DD` or `YYYY-MM-DD HH:MM`, help
  text like `critical`'s).
- `--phases N [N ...]` (int, default `None`).
- `--aggregate {union,mean}` (default `union`). Help text: union = occupied when
  any lane is on (UDOT, like one multi-lane detector); mean = average of
  per-lane GOR/ROR5.
- `--threshold FRAC` (float, default `0.79`), `--ror-seconds SEC` (float,
  default `5.0`), `--include-yellow` (store_true; GOR over green + yellow, the
  SPMs definition).
- `--bin-len` (string, default `"60"`; `"cycle"` or minutes, as in `aog`),
  `--exclude-missing`, `--no-plot` (dest `no_plot`), `--timezone`, `--verbose`.
- Handler `handle_split_failures(args)`; outputs to `intersections/<target>/outputs/`;
  pass `make_plot=not args.no_plot`.

## Stop and ask (write it in the report and stop) if

- any existing test fails for a reason you'd have to change a forbidden file to fix;
- the core's output columns differ from what its docstring says;
- you'd need a new dependency.

## Report

Write `docs/specs/split_failures_shell_REPORT.md`: files touched, test result line,
anything you were unsure about, and anything you deviated from in this spec. Keep it
under 40 lines. Commit everything on your branch with a descriptive message.
