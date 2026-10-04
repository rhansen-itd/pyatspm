# Spec: left-turn gap shell, plot and `atspm left-turn-gap` CLI (UDOT S-M9)

The functional core already exists and is tested:

- `src/atspm/analysis/left_turn_gap.py`: `left_turn_pairs`, `through_phases`,
  `left_turn_gaps`, `summarize_left_turn_gaps`, `check_edges`, `bin_columns`,
  `bin_labels`, `cycle_schema`, `summary_schema`, `critical_gap`,
  `GAP_SCHEMA`, `PAIR_SCHEMA`, `DEFAULT_EDGES`, `DEFAULT_TREND_S`,
  `DEFAULT_BIN_LEN`.
- `src/atspm/analysis/counts.py`: `parse_exclusions_from_config`.
- `src/atspm/data/reader.py`: `get_events_with_cycles_df`.

Read their docstrings first, the module docstring of `left_turn_gap.py`
in particular. **Don't reimplement or edit any of it.** `left_turn_gaps`
returns a *tuple* `(cycles, gaps)`, one call per left turn. The column
lists depend on `edges` (`bin_1 … bin_K`), so always build schemas with
`cycle_schema(edges)` / `summary_schema(edges)`, never a hard-coded list.

Your job:

- the imperative shell (`LeftTurnGapEngine`);
- one plot, a pure function in one new `plotting/` module;
- one CLI subcommand, `atspm left-turn-gap`.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_left_turn_gap_engine.py`. Read it first; it is the contract.
Everything else was green before it was added.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**` and `src/atspm/utils/**`
- every `src/atspm/plotting/*.py` other than the new file
- every `src/atspm/data/*.py` other than the new file
- `docs/*.md` other than the two named below, and `README.md`

## Files you may create or edit

- **create** `src/atspm/data/left_turn_gap.py`
- **create** `src/atspm/plotting/left_turn_gap.py`
- **edit** `src/atspm/cli.py`: the new subcommand only
- **edit** `src/atspm/data/__init__.py` and `src/atspm/plotting/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/left_turn_gap_shell_REPORT.md`

## 1. `src/atspm/data/left_turn_gap.py`

Mirror `src/atspm/data/green_time_utilization.py` (`GreenTimeEngine`,
`get_green_time`) for structure, Google-style docstrings, timezone
handling, `_read_timezone`, `_get_config`, `_format_stamp`,
`CriticalMovementEngine._parse_range` use (a date-only end extends to the
end of that day; a datetime end is exclusive) and the output writer.

```python
_LTG_CODES = (-1, 1, 8, 9, 10, 11, 12, 81)   # gap markers, phase states, detector off
FETCH_MARGIN_S: float = 3600.0               # a resting through green can last 20 min

class LeftTurnGapEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def left_turn_gap(self, start, end, lefts: Optional[List[str]] = None,
                      bin_len: int = DEFAULT_BIN_LEN,
                      edges: Sequence[float] = DEFAULT_EDGES,
                      trend_s: float = DEFAULT_TREND_S,
                      critical_s: Optional[float] = None,
                      use_exclusions: bool = True,
                      write_gaps: bool = False,
                      make_plot: bool = True,
                      output_dir=None) -> Optional[Dict[str, pd.DataFrame]]: ...

def get_left_turn_gap(db_path, start, end, lefts=None, bin_len=DEFAULT_BIN_LEN,
                      edges=DEFAULT_EDGES, trend_s=DEFAULT_TREND_S, critical_s=None,
                      use_exclusions=True, write_gaps=False, make_plot=True,
                      output_dir=None, timezone=None): ...
```

"Return empty" below means: return `{}`, or `None` when `output_dir` is set,
and write no files. Every warning line starts `  ⚠️  LeftTurnGap: `.

### Steps of `left_turn_gap()`

1. **Validate** before touching the DB: `bin_len` must be a positive
   divisor of 60, and `check_edges(edges)` must not raise; otherwise
   `ValueError`. The engine must raise even when the DB file does not
   exist (the test passes `timezone`, so `__init__` needn't read it).
2. **Parse** the window: `start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)`.
3. **Config** at `start_dt` (`_get_config`): `pairs = left_turn_pairs(config)`;
   `exclusions = parse_exclusions_from_config(config) if use_exclusions else None`.
   - `pairs` empty → warning containing `TM_` (e.g.
     `no TM_{NB|SB|EB|WB}L left-turn movement config found. Check TM_* rows in int_cfg.csv.`)
     and return empty.
   - When `lefts` is given, a label not in `pairs["left"]` gets a warning
     naming it (`SBL: no TM_SBL key; skipped.`); the run continues with
     the others.
   - The **runnable** pairs are those (in `pairs` order, after the `lefts`
     filter) with a non-NA `opposing_phase` and non-empty `detectors`.
     For each other pair, a warning. When `opposing_phase` is NA:
     `{left}: no through phase for {opposing} ({source}{, candidates P3/P4 when any}); add Det:,P{N} Direction,{opposing} to int_cfg.csv.`
     The test matches `NBL` and the literal `Det:,P{N} Direction,SB`.
     When `detectors` is empty: `{left}: no TM_{opposing}T/R detectors; skipped.`
   - For each runnable pair with a non-empty `shared`, print
     `  ⚠️  LeftTurnGap: {left}: opposing detector(s) {a, b} also configured under another direction (TM_*); they count as opposing traffic.`
     (701's `TM_WBR = 54` is also in `TM_EBL`.)
   - For each runnable pair, an info line naming the phase and source, e.g.
     `  ℹ️  LeftTurnGap: EBL opposed by WB through, Ph6 (derived), detectors 18,19,20,21, critical 5.3 s`.
   - No runnable pair → return empty (after the warnings).
4. **Events:** one fetch,
   `get_events_with_cycles_df(db_path=self.db_path, start=start_dt - margin, end=end_dt + margin, event_codes=list(_LTG_CODES), timezone=self.timezone)`
   with `margin = timedelta(seconds=FETCH_MARGIN_S)`, so a green that
   starts inside the window is seen to its end. Empty → warning, return
   empty.
5. **Silent detectors:** for each runnable pair, the opposing detectors
   with no Code 81 between `start_dt` and `end_dt` (local). When any, print
   `  ⚠️  LeftTurnGap: {opposing} detector(s) {a, b} logged no actuation in the window.`
   (IDs ascending; the test matches `SB detector(s) 31 logged no actuation`).
   Informational only; the pair still runs (its greens are one gap each).
   At 201 the EB TM channel 63 logs only on 2026-03-18..21.
6. **Core** per runnable pair:
   `cy, gp = left_turn_gaps(events, int(opposing_phase), detectors, left=left, edges=edges, trend_s=trend_s, critical_s=critical_s if critical_s is not None else pair.critical_s, exclusions=exclusions)`.
   Keep only the rows whose `green_ts` is in `[start_dt, end_dt)` (local,
   tz-aware comparison; `gaps` by its `green_ts` too). Concatenate across
   pairs with `ignore_index=True`; when no pair produced a row, use empty
   frames with `cycle_schema(edges)` / `GAP_SCHEMA`.
   Don't `fillna`, cast or re-round anything: the result must equal the
   core's output row for row (`test_matches_the_core`).
7. **Summaries:** `bins = summarize_left_turn_gaps(cycles, bin_len, edges)`,
   `summary = summarize_left_turn_gaps(cycles, None, edges)`.
   `cycles` empty → return empty.
8. **Outputs**, only when `output_dir` is given, with
   `stamp = _format_stamp(start_dt, end_dt)`, `index=False`, and a
   `Wrote <name>` line each:
   - `LTG_Pairs_{stamp}.csv`: `pairs` (all of them, not only the runnable
     ones) with the `detectors` and `shared` lists written as
     comma-joined strings (`"18,21"`, `""` when empty);
   - `LTG_Cycles_{stamp}.csv`, `LTG_Bins_{bin_len}min_{stamp}.csv`,
     `LTG_Summary_{stamp}.csv`;
   - `LTG_Gaps_{stamp}.csv` only when `write_gaps` (a month at 315 is
     ~500k rows);
   - `LTG_Chart_{stamp}.html` when `make_plot`:
     `plot_left_turn_gap(bins, metadata, edges=edges, trend_s=trend_s)`,
     metadata from `DatabaseManager.get_metadata()`. `write_html` lives
     here, never in the plot module.

   Return `None` when `output_dir` is set; otherwise
   `{"pairs", "cycles", "gaps", "bins", "summary"}`, `pairs` exactly as
   `left_turn_pairs` returned it (lists intact) and the others with the
   core's columns exactly.

**Dtype care:** count columns are nullable `Int64` (NA on censored
greens); `window_s`, `turnable_s`, `pct_turnable`, `sum_ge_critical` are
NaN there. Don't `iterrows` anything; iterate `pairs` with `itertuples`
(at most four rows).

## 2. `src/atspm/plotting/left_turn_gap.py`

```python
def plot_left_turn_gap(bins_df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None,
                       edges: Sequence[float] = DEFAULT_EDGES,
                       trend_s: float = DEFAULT_TREND_S) -> go.Figure: ...
```

Pure function (no I/O, no `write_html`). Look at
`plotting/green_time_utilization.py` and `plotting/approach_volume.py`
for style. UDOT draws one chart per left turn: the gap counts per time
bin as stacked columns, one colour per gap bin, and "% of green time where
gaps ≥ 7.4 s" as a dashed line on a 0–100 secondary axis.

- **Title:** `_build_title(metadata or {}, suffix="Left Turn Gap Analysis")`
  from `plotting/termination.py`.
- **Layout:** one row per `left` present in `bins_df`, in the order
  NBL, SBL, EBL, WBL; `make_subplots(rows=n, cols=1, shared_xaxes=True,
  specs=[[{"secondary_y": True}]] * n)`. Subplot title
  `"{left} across opposing Ph{opposing_phase}"`. Primary y "Gaps",
  secondary y `"% green with gaps ≥ {trend_s:g}s"`, range `[0, 100]`.
  `barmode="stack"`.
- **Traces,** per left, from its rows of `bins_df` (boolean masks; never
  pivot or merge: one row per (time, left)):
  - one `go.Bar` per bin column of `bin_columns(edges)`, named by the
    matching `bin_labels(edges)` entry (`"1-3.3s"`, …, `"7.4s+"`), x =
    `time`, y = the column. Same colour for a bin on every row,
    `legendgroup` = the label, `showlegend` only on the first row.
    `n_short` is not drawn (UDOT doesn't).
  - one `go.Scatter(mode="lines", line_shape="hv", connectgaps=False)`
    named `"% green with gaps ≥ {trend_s:g}s"`, `line.dash="dash"`, y =
    `pct_turnable` unchanged (NaN on a censored-only bin breaks the line),
    on the secondary axis, `showlegend` only on the first row.
  Hover shows time, the value and the trace name.
- **Empty input:** an empty `bins_df` returns a titled, empty `go.Figure`.

## 3. CLI: `atspm left-turn-gap`

Copy `green-time`'s parser and handler pattern
(`_green_time_single_intersection` / `handle_green_time`).

| Argument | Notes |
|---|---|
| `--target` / `--targetid` / `--all` | required, mutually exclusive |
| `--start`, `--end` | required, local dates or datetimes |
| `--left L [L ...]` | `nargs="+"`, `choices=["NBL", "SBL", "EBL", "WBL"]`, default `None` (all) |
| `--bin-len M` | `type=int`, default 15; time bin, minutes (must divide 60) |
| `--edges LIST` | comma-separated seconds, e.g. `1,3.3,3.7,7.4`; parsed by an argparse `type` function into a tuple of floats with `math.inf` appended (the last bin is open, as UDOT's); a non-number → `argparse.ArgumentTypeError`. Default `None` → `DEFAULT_EDGES` |
| `--trend S` | `type=float`, default `DEFAULT_TREND_S` (7.4) |
| `--critical S` | `type=float`, default `None` (UDOT's lane rule per left) |
| `--gaps` | flag; also write `LTG_Gaps_*.csv` |
| `--no-exclusions` | flag, as `green-time` |
| `--no-plot` | flag |
| `--timezone`, `--verbose` | |

- Handler `handle_left_turn_gap`, registered in `_build_parser`;
  `args.func`, `args.left`, `args.bin_len`, `args.edges`, `args.trend`,
  `args.critical`, `args.gaps`, `args.no_exclusions`, `args.no_plot`.
- Output goes to `<intersection>/outputs/`.
- After writing, print one line per row of `LTG_Summary_*.csv`, e.g.
  `EBL vs Ph6  952 greens (1 censored)  gaps 1-3.3s 1910 · 3.3-3.7s 156 · 3.7-7.4s 1287 · 7.4s+ 1700  turnable 73.8%  ≥crit 52444 s`
  (`pct_turnable` with one decimal, `-` when NA; the bin labels from
  `bin_labels(edges)`).
- When the engine wrote no summary CSV (no left config, nothing in the
  window), print nothing more and don't raise.
- With `--all`, one failing intersection must not stop the others.
- Add the command to the usage block at the top of `cli.py`, beside
  `green-time`.

**Heads-up:** another, unrelated `sync` subcommand may land in `cli.py`
on `main` in parallel. Keep your `cli.py` edits contiguous (one handler
block, one parser function, one `_build_parser` line, one usage line) so
a merge is easy. Don't touch anything sync-related.

## 4. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `LeftTurnGapEngine` and `get_left_turn_gap`.
- `src/atspm/plotting/__init__.py`: export `plot_left_turn_gap`.
- Append to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new left-turn-gap subcommand`
  - `[src/atspm/data/__init__.py] export LeftTurnGapEngine, get_left_turn_gap`
  - `[src/atspm/plotting/__init__.py] export plot_left_turn_gap`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output differs from what its docstring says;
- a golden test looks wrong to you (explain why rather than working around it);
- you'd need a new dependency.

## Report

Write `docs/specs/left_turn_gap_shell_REPORT.md`. Include the files you
touched, the test result line, anything you were unsure about, and anywhere you
deviated from this spec. Keep it under 30 lines. Commit everything on your
branch.
