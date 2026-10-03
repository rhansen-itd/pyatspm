# Spec: detector-health shell engine, heatmap and `atspm detector-health` CLI (UDOT S-D4)

The rule core already exists and is tested: `src/atspm/analysis/detector_health.py`
and `src/atspm/analysis/detector_activity.py` (read their module docstrings and
the `detector_health_findings`, `detector_activity_profile` docstrings first).
Opus has already written everything in the functional core and the DB layer that
this shell needs — **do not reimplement any of it**:

- `DatabaseManager.replace_findings(findings, date_start, date_end, computed_at=None)`
  — idempotent delete-then-replace of `detector_findings` for a date range, in
  one transaction. NA sentinels (`phase`/`ts` → -1, `role` → '') are applied
  inside it.
- `DatabaseManager.get_findings(date_start, date_end, window=None)` — reads the
  range back with sentinels restored to the analysis schema.
- `detector_findings` table (in `init_db`) and its clearing in
  `clear_ingested_data()` are already done.
- Core helpers in `atspm.analysis.detector_health`, all pure, all exported from
  `atspm.analysis`: `wd_reboot_windows`, `wd_profile_windows`, `wd_units`,
  `wd_ignore`, `apply_ignore`, `wd_thresholds`, `filter_min_severity`,
  `severity_exit_code`.

Your job is the imperative shell (`DetectorHealthEngine`), a pure Plotly figure
(`plot_detector_health`), the `atspm detector-health` CLI subcommand, and a
findings summary hook in `atspm report`.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including the new
`tests/data/test_detector_health_engine.py` and
`tests/plotting/test_detector_health_plot.py`. Read both first — they are the
contract. The whole suite was green before these two files were added.

## Files you must NOT modify

- `tests/**` (every test file and fixture)
- `src/atspm/analysis/**` (the functional core, including `detector_health.py`,
  `detector_activity.py`, `detector_roles.py`)
- `src/atspm/data/manager.py`, `ingestion.py`, `processing.py`, `reader.py`,
  `aog.py`, `flow.py`, `critical.py`, `split_failures.py`
- `src/atspm/utils/**`
- `docs/UDOT_MOE_ROADMAP.md`, `docs/ROADMAP.md`, `README.md`, `docs/*.md` other
  than the two named below

## Files you may create or edit

- **create** `src/atspm/data/detector_health.py`
- **create** `src/atspm/plotting/detector_health.py`
- **edit** `src/atspm/cli.py` (new subcommand + the report hook only; don't touch
  other commands)
- **edit** `src/atspm/reports/generators.py` (add a findings summary only)
- **edit** `src/atspm/data/__init__.py`, `src/atspm/plotting/__init__.py` (exports only)
- **append** to `docs/PENDING_DOC_CHANGES.md` (one bullet per touched file, format in CLAUDE.md)
- **create** `docs/specs/detector_health_shell_REPORT.md` (your report)

## 1. `src/atspm/data/detector_health.py`

Mirror `src/atspm/data/aog.py` (`AogEngine`) for structure, docstrings (Google
style), timezone (`db_timezone` / `_read_timezone`), `_parse_range`, config
lookup, output writing and `--all` error surfacing. Reuse the AOG
`_add_quality` / ingestion-span approach only where RecordCount needs it (below).

```python
# codes the profile + event rules need: detector on/off, gap marker, max-out,
# and the controller-fault codes the core reads (CONTROLLER_FAULT_CODES).
_ALL_HEALTH_CODES: List[int]   # sorted set of {-1, 5, 81, 82} | CONTROLLER_FAULT_CODES keys
_EDGE_MARGIN_S: float = 3600.0 # event fetch margin so burst releases / on-durations near a day edge are measured

class DetectorHealthEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None
    def detector_health(self, start, end, window="day", min_severity="low",
                        thresholds=None, output_dir=None) -> Dict[str, object]

def get_detector_health(db_path, start, end, window="day", min_severity="low",
                        thresholds=None, output_dir=None, timezone=None)  # convenience wrapper
```

Behaviour of `detector_health`:

1. `start_dt, end_dt = self._parse_range(start, end)` (AOG semantics: `end`
   extended to end-of-day). Derive the inclusive local date strings
   `date_start`, `date_end` (`YYYY-MM-DD`) for persistence.
2. Fetch events **once** for `[start_dt - _EDGE_MARGIN_S, end_dt + _EDGE_MARGIN_S]`
   with `get_events_with_cycles_df(..., event_codes=_ALL_HEALTH_CODES,
   timezone=self.timezone)`. Gap markers (-1) MUST be in the code list. Convert
   the `timestamp` column to a plain UTC-epoch `float` Series for the core (the
   core accepts tz-aware or epoch; epoch is simplest). Empty → warn, persist an
   empty frame for the range (clears it), return the empty result (below).
3. `configs = DatabaseManager(...).get_configs_for_range(lo, hi)` where `lo`/`hi`
   are aware datetimes at the range edges (copy the `sweep_detector_health.py`
   `load()` idiom: `datetime.fromtimestamp(t, zone)`). For **each** config
   period (mirrors the sweep's `profile_site`):
   - `roles = parse_detector_roles({k: v for k, v in cfg.items() if not k.startswith("_")})`
   - `windows = wd_profile_windows(cfg)` (day + am + pm-if-configured — profile
     ALL of them regardless of `--window`),
   - `th = thresholds or wd_thresholds(cfg)`,
   - `units, _types = wd_units(cfg)`, `reboot = wd_reboot_windows(cfg)`,
   - slice events to `[cfg["_epoch_start"], cfg["_epoch_end"])`; skip empty,
   - `profile = detector_activity_profile(sub_events, self.timezone, roles, windows)`
     (restricted to the requested dates via its `start_date`/`end_date` args —
     the config slice may extend past the range because of the edge margin),
   - `findings = detector_health_findings(profile, roles, sub_events,
     self.timezone, th, reboot_windows=reboot, units=units)`.
   Concatenate the per-period findings and the per-period profiles
   (`ignore_index=True`). Keep only findings whose `date` is within
   `[date_start, date_end]` (drop any edge-margin bleakage).
4. **RecordCount (best-effort shell rule).** Using
   `DatabaseManager.get_ingestion_spans()` and the AOG daily-coverage idea,
   append one `RecordCount` finding (`detector=-1`, `phase`=NA, `rule=RecordCount`,
   `severity=low`, `value`=coverage, `threshold`=`th.min_observed_share`) per
   `(date, window)` whose logged coverage is below `th.min_observed_share`. This
   is the only rule the shell adds; it is not covered by a strict golden, so keep
   it simple and documented. If `ingestion_log` is empty, skip it.
5. Persist **everything** (all severities, ignored included):
   `DatabaseManager(...).replace_findings(all_findings, date_start, date_end)`.
   One call for the whole range (it is idempotent and transactional).
6. Build the reported view (does NOT change the table):
   - `ignore = wd_ignore(cfg_latest)` (use the most recent config period's
     ignore list), `reported = apply_ignore(all_findings, ignore)`,
   - filter to the selected window unless `window in (None, "all")`:
     `reported = reported[reported["window"] == window]`,
   - `reported = filter_min_severity(reported, min_severity)`,
   - `exit_code = severity_exit_code(reported, min_severity)`.
7. Return `{"findings": all_findings, "reported": reported, "exit_code": exit_code}`
   **always** (so the CLI can set the process exit code even when writing files).
   When `output_dir` is set, also write:
   - `DetectorHealth_Findings_{stamp}.csv` — `reported`, `index=False`
     (stamp = `date_start`–`date_end`, AOG style),
   - `DetectorHealth_Heatmap_{stamp}.html` from
     `plot_detector_health(profile_all[profile_all["window"]==plot_window],
     reported, metadata, window=plot_window)` via `fig.write_html` (the ONLY
     place `write_html` may appear), where `plot_window` is `window` or `"day"`.
     Metadata: `DatabaseManager(...).get_metadata()`.
8. Print a short per-run summary in the style of the other engines (counts per
   rule and per severity, how many were ignored).

No `iterrows()` or row loops over events.

## 2. `src/atspm/plotting/detector_health.py` (functional core: pure, no I/O)

```python
def plot_detector_health(profile, findings, metadata=None, window="day") -> go.Figure
```

- A single `go.Heatmap`: `y` = detectors (sorted), `x` = local dates (sorted),
  `z[detector, day]` = that `(detector, day)`'s `n_act` **normalized to the
  detector's own median `n_act`** across the shown days (median 0 → leave the
  row un-normalized / NaN; never divide by zero). The profile is already one row
  per `(date, window, detector)`; select the given `window`.
- Findings overlaid: one `go.Scatter` marker layer over the flagged
  `(detector, date)` cells (hover shows rule + severity + message). Omit it when
  `findings` is empty. Do not use `fig.add_shape`.
- Title: reuse `_build_title` from `atspm.plotting.termination` (import it, don't
  copy) with a suffix like `"Detector Health — count vs the detector's median"`.
  Missing road names must not crash (the title helper already handles them).
- Vectorized construction only — build `z` with a pivot, not row loops.
- An empty profile returns a valid empty figure.

## 3. CLI: `atspm detector-health`

Mirror the `aog` subcommand (`_aog_single_intersection`, `handle_aog`, its
parser) and register the parser in `_build_parser()` near the other analysis
commands. Add the line to the module docstring's command list.

- Mutually exclusive required group `--target` / `--targetid` / `--all`.
- `--start`, `--end` (required; accept `YYYY-MM-DD`, help like `aog`'s). A single
  date → `--end` defaults to `--start`.
- `--window {am,pm,day}` (default `day`).
- `--min-severity {info,low,high}` (default `low`).
- `--timezone`, `--verbose`.
- Handler writes outputs to `intersections/<target>/outputs/` and, for a
  single target (not `--all`), calls `sys.exit(result["exit_code"])` after the
  run. Under `--all`, do not exit mid-batch; print a per-target summary and
  continue (the batch loop pattern already in `handle_aog`).

## 4. Report hook (`atspm report`)

In `src/atspm/reports/generators.py`, add a small method on `PlotGenerator`
(e.g. `_generate_detector_health_summary`) that, for the report's date,
reads `DatabaseManager.get_findings(date, date)`, applies `apply_ignore` with
the active config's `wd_ignore`, and writes a compact findings summary
(CSV or an HTML table — your choice, consistent with the other `_generate_*`
outputs) into the report output dir. It must not recompute findings (read the
table only) and must be a no-op with a clear message when the table is empty.
Wire it into `generate_for_date` alongside the other `_generate_*` calls.

## Stop and ask (write it in the report and stop) if

- any existing test fails for a reason you'd have to change a forbidden file to fix;
- a core function's output columns differ from what its docstring says;
- the engine golden requires behaviour the spec doesn't describe;
- you'd need a new dependency.

## Report

Write `docs/specs/detector_health_shell_REPORT.md`: files touched, the test
result line, anything you were unsure about, and anything you deviated from in
this spec. Keep it under 40 lines. Commit everything on your branch with a
descriptive message.
