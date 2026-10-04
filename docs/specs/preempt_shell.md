# Spec: preemption-detail shell and `atspm preempt` CLI (UDOT S-M6)

The functional core already exists and is tested:
`src/atspm/analysis/preempt.py` (`PREEMPT_CODES`, `preempt_episodes`,
`preempt_summary`), exported from `atspm.analysis`. Read its module docstring
and the two function docstrings first. **Don't reimplement or edit it.**

Your job is the imperative shell (`PreemptEngine`) and the `atspm preempt`
CLI subcommand.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_preempt_engine.py`. Read it first; it is the contract.
Everything else was green before it was added.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**`, `src/atspm/plotting/**`, `src/atspm/utils/**`
- `src/atspm/data/manager.py`, and every `src/atspm/data/*.py` other than the new file
- `docs/*.md` other than the two named below, and `README.md`

## Files you may create or edit

- **create** `src/atspm/data/preempt.py`
- **edit** `src/atspm/cli.py`: the new subcommand only
- **edit** `src/atspm/data/__init__.py`: exports only
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/preempt_shell_REPORT.md`

## 1. `src/atspm/data/preempt.py`

Mirror `src/atspm/data/clock_marks.py` (`ClockMarkEngine`, `get_clock_marks`):
structure, Google-style docstrings, timezone (`db_timezone`), and `_parse_range`
(a date-only end extends to the end of that day).

```python
class PreemptEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None: ...
    def preempt(self, start, end, output_dir=None) -> Dict[str, object]: ...

def get_preempt(db_path, start, end, output_dir=None, timezone=None) -> Dict[str, object]: ...
```

`preempt()` does the following:

1. Parse the range as local times and convert it to epoch with
   `utils.timezone.to_epoch`.
2. Fetch the `PREEMPT_CODES` events over
   `[w0 - FETCH_MARGIN_S, w1 + FETCH_MARGIN_S)` with
   `DatabaseManager.query_events`, where `FETCH_MARGIN_S = 1800.0`.
3. `episodes = preempt_episodes(events)`. Keep only the rows with
   `w0 <= call_on < w1`. The margin exists so that a request near an edge is
   seen whole, but a request belongs to the range its Call On falls in.
   Then `summary = preempt_summary(episodes, tz)`.
4. If `output_dir` is given, write two CSVs named with
   `stamp = f"{d0:%Y_%m_%d}-{d1:%Y_%m_%d}"`, where `d1` is the last local
   date included:
   - `Preempt_Episodes_{stamp}.csv`: the episodes plus two columns.
     - `call_on_local`: local wall clock, ISO with deciseconds, e.g.
       `2026-01-10T08:00:00.5`. That's `strftime("%Y-%m-%dT%H:%M:%S.%f")`
       with the last 5 characters cut.
     - `timing_plot`: `atspm plot-timing-actuation --targetid {intersection_id} --start {s} --end {e}`.
       Here `s = call_on − 120 s` and `e = (exit_ts, else call_off, else
       call_on) + 120 s`, both as local `%Y-%m-%dT%H:%M:%S` (truncating, not
       rounding). Use `--target {db_path.parent.name}` when the metadata has
       no `intersection_id`. Build it vectorized: no `iterrows`.
   - `Preempt_Summary_{stamp}.csv`: the summary.

   Print `Wrote <name>` for each, as the other engines do. Write nothing when
   `output_dir` is None.
5. Return `{"episodes": episodes, "summary": summary, "html": None}`. There's no
   plot in this item; the `timing_plot` command is the visual.

## 2. CLI: `atspm preempt`

Copy `clock-drift`'s parser and handler pattern:

- a required mutually exclusive `--target` / `--targetid` / `--all`;
- `--start` and `--end` (local dates or datetimes, required);
- `--timezone` and `--verbose`.

The handler is named `handle_preempt`. Its output goes to
`<intersection>/outputs/`. After writing, print a short per-preempt summary:
requests, served, unserved, censored, max-presence hits, and mean/max dwell.
With `--all`, one failing intersection must not stop the others. Register it
in `_build_parser`.

## 3. Exports and doc bullets

- `src/atspm/data/__init__.py`: export `PreemptEngine` and `get_preempt`.
- Append these bullets to `docs/PENDING_DOC_CHANGES.md`:
  - `[src/atspm/cli.py] new preempt subcommand`
  - `[src/atspm/data/__init__.py] export PreemptEngine, get_preempt`

## Stop and ask (write it in the report and stop) if

- an existing test fails for a reason you could only fix by changing a
  forbidden file;
- a core function's output differs from what its docstring says;
- a golden test looks wrong to you (explain why rather than working around it);
- you'd need a new dependency.

## Report

Write `docs/specs/preempt_shell_REPORT.md`. Include the files you touched, the
test result line, anything you were unsure about, and anywhere you deviated
from this spec. Keep it under 30 lines. Commit everything on your branch.
