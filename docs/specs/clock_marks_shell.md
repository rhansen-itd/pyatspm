# Spec: clock-mark shell engine, plot and `atspm clock-drift` CLI (ROADMAP S1)

The pure decoder already exists and is tested: `src/atspm/analysis/clock_marks.py`
(read its module docstring and `decode_clock_marks` docstring first). Your job is
the imperative shell around it, a pure Plotly figure, and a CLI subcommand.

## Acceptance

`python -m pytest -q` passes in full, including the new
`tests/data/test_clock_marks_engine.py`. Use the repo venv: `.venv/bin/python`.

## Files you must NOT modify

- `tests/**` (every test file, fixtures included)
- `src/atspm/analysis/**` (the functional core, including `clock_marks.py`)
- `src/atspm/data/ingestion.py`, `src/atspm/data/processing.py`, `src/atspm/data/manager.py`, `src/atspm/data/reader.py`
- `src/atspm/reports/**`
- `docs/ROADMAP.md`, `README.md`, `docs/*.md` other than the two named below

## Files you may create or edit

- **create** `src/atspm/data/clock_marks.py`
- **create** `src/atspm/plotting/clock_marks.py`
- **edit** `src/atspm/cli.py` (new subcommand only; don't touch other commands)
- **edit** `src/atspm/data/__init__.py`, `src/atspm/plotting/__init__.py` (exports only)
- **append** to `docs/PENDING_DOC_CHANGES.md` (one bullet per touched file, format in CLAUDE.md)
- **create** `docs/specs/clock_marks_shell_REPORT.md` (your report)

## 1. `src/atspm/data/clock_marks.py`

Mirror `src/atspm/data/critical.py` (`CriticalMovementEngine`) for structure,
docstrings (Google style), timezone resolution (`db_timezone`), config lookup
(`DatabaseManager.get_config_at_date`) and output stamping (`_write_outputs`).

```python
class ClockMarkEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None
    def decode(self, start, end, send_log_path=None, output_dir=None
               ) -> Optional[Dict[str, pd.DataFrame]]

def get_clock_marks(db_path, start, end, send_log_path=None,
                    output_dir=None, timezone=None)   # convenience wrapper
```

Behaviour of `decode`:

1. Parse `start`/`end` exactly like `CriticalMovementEngine._parse_range`, but also
   accept `"%Y-%m-%d %H:%M:%S"` (and the `T` variant). A date-only end extends to
   end of day. Times are intersection-local; convert with
   `atspm.utils.timezone.to_epoch`.
2. Config: `marker_peds_from_config(config_at_start)`. If it returns `None`,
   print a warning containing the text `Clk` (e.g. "no Clk_* config — no clock
   marks to decode") and return `{}` (or `None` when `output_dir` is set). If it
   raises `ValueError`, print it and do the same.
3. Fetch events with SQL from `events` for `event_code IN (-1, 89, 90)` over
   `[start_epoch - 300, end_epoch + 300)` (a module constant `_FETCH_MARGIN = 300.0`;
   a run's pulses and bracket span at most ~2 minutes). Columns
   `timestamp, event_code, parameter`, as epoch floats. Gap markers MUST be
   included (all `event_code = -1` rows, any parameter).
4. Send log: if `send_log_path` is given, read it as JSONL (one JSON object per
   line, skip blank lines) and pass `send_log_pulses(records)` to the core.
5. Call `decode_clock_marks(events, peds, send_log)` on the whole margin fetch,
   **then** keep only rows whose time lies in `[start_epoch, end_epoch)`: drift
   rows by `ts` (or `off` where `ts` is NaN), set rows by `bracket_on` (or
   `bracket_off` where that is NaN). Filtering after decoding is what lets a
   bracket find a pre-set pulse just outside the window. Reset indexes.
6. Print a short summary (counts of drift samples and sets, and how many are
   not `status == "ok"`), in the style of the other engines.
7. Return `{"drift": drift_df, "sets": sets_df}` with times still as UTC epoch
   floats, unless `output_dir` is set: then write and return `None`:
   - `Clock_Drift_{stamp}.csv`, `Clock_Sets_{stamp}.csv`: the same frames with
     every time column (`ts`, `off`, `bracket_on`, `bracket_off`, `step_lo`,
     `step_hi`) converted to intersection-local, timezone-naive wall-clock
     datetimes, written so the CSV shows e.g. `2026-09-30 13:04:27.500`.
     Use `index=False`.
   - `Clock_Drift_{stamp}.html`: `plot_clock_drift(...)` written with
     `fig.write_html` (the only place `write_html` may appear).
   - `{stamp}` uses the same rule as `critical.py`'s `_write_outputs`
     (`2026_09_30-2026_09_30` for a whole day).
   - Metadata for the plot title: `DatabaseManager(...).get_metadata()`.

No `iterrows()` or row loops over events.

## 2. `src/atspm/plotting/clock_marks.py` (functional core: pure, no I/O)

```python
def plot_clock_drift(drift_df, sets_df, metadata=None, timezone=None) -> go.Figure
```

- x axis: time, converted from epoch to `timezone` (tz-aware) when given.
- One scatter trace named exactly `"Drift"` holding the rows with non-NaN
  `drift` (markers), with asymmetric error bars from `drift_lo`/`drift_hi`
  (finite ones only). y axis title "Controller − true (s)".
- Saturated rows whose `drift` is NaN: one separate trace named
  `"Saturated (bound only)"`, plotted at their finite bound, with a distinct
  marker symbol (e.g. `triangle-up` for positive, `triangle-down` for negative).
- Sets: one trace named `"Clock set"` built with the vectorized
  `[x, x, None]` segment pattern (vertical line per set at `bracket_on`, y
  spanning the plot's drift range, or ±1 when there's no drift), hover text showing
  `shift` (or the `status` when shift is NaN). **No `fig.add_shape` / layout
  shapes, and no `add_vline`** (those create layout shapes).
- A horizontal zero reference may be a trace too.
- Title: reuse `_build_title` from `atspm.plotting.termination` (import it,
  don't copy it) with suffix `"Controller Clock Drift"`.
- Empty frames must still return a valid (empty) figure.
- Hover colours match their traces.

## 3. CLI: `atspm clock-drift`

Mirror the `critical` subcommand exactly (`_critical_single_intersection`,
`handle_critical`, `_add_critical_parser`) and register `_add_clock_drift_parser`
in `_build_parser()` after `_add_critical_parser(subs)`. Add the line to the
module docstring's command list.

- Mutually exclusive required group `--target` / `--targetid` / `--all`.
- `--start`, `--end` (required; same help text style as `critical`).
- `--send-log PATH` (dest `send_log`, default `None`): the head unit's
  `eos-time.jsonl`. If omitted, use `<target_dir>/eos-time.jsonl` when that
  file exists. Combining `--send-log` with `--all` is an error (`_die`), since
  one send log belongs to one controller.
- `--timezone`, `--verbose`, as in `critical`.
- Handler `handle_clock_drift(args)`; outputs go to `intersections/<target>/outputs/`.

## Stop and ask (write it in the report and stop) if

- any existing test fails for a reason you'd have to change a forbidden file to fix;
- the core's output columns differ from what its docstring says;
- you'd need a new dependency.

## Report

Write `docs/specs/clock_marks_shell_REPORT.md`: files touched, test result line,
anything you were unsure about, and anything you deviated from in this spec. Keep it
under 40 lines. Commit everything on your branch with a descriptive message.
