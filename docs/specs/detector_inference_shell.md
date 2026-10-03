# Spec: detector-inference shell engine and `atspm infer-detectors` CLI (UDOT S-D6)

The pure core already exists and is tested: `src/atspm/analysis/detector_inference.py`
(`infer_detector_roles`, `diff_detector_roles`; read the module docstring and both
function docstrings first). Your job is the imperative shell around it and a CLI
subcommand. Design background: `docs/design_detector_config_inference.md`.

**The engine proposes; it never writes config.** No `int_cfg.csv` writes, and no
`config`-table writes.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including
`tests/data/test_detector_inference_engine.py`.

## Files you must NOT modify

- `tests/**`
- `src/atspm/analysis/**`
- `src/atspm/plotting/**`
- `src/atspm/data/manager.py`, `ingestion.py`, `processing.py`, `reader.py`, and every
  other existing `data/` module (import from them; don't edit them)
- `docs/ROADMAP.md`, `docs/UDOT_MOE_ROADMAP.md`, `README.md`, and `docs/*.md` other than the two below

## Files you may create or edit

- **create** `src/atspm/data/detector_inference.py`
- **edit** `src/atspm/cli.py` (new subcommand only; don't touch other commands)
- **edit** `src/atspm/data/__init__.py` (exports only)
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/detector_inference_shell_REPORT.md`

## 1. `src/atspm/data/detector_inference.py`

Mirror `src/atspm/data/aog.py` (`AogEngine`) for structure, Google-style docstrings,
timezone resolution (`self.timezone = timezone or <metadata timezone>`, like AogEngine)
and config lookup.

```python
_INFERENCE_CODES: List[int]   # exactly [-1, 1, 8, 9, 81, 82] (sorted)

class DetectorInferenceEngine:
    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None
    def infer(self, start, end, use_ring_config: bool = True,
              min_actuations: int = 50,
              output_dir: Optional[Union[str, Path]] = None
              ) -> Optional[Dict[str, pd.DataFrame]]

def get_detector_inference(db_path, start, end, **kwargs)   # thin wrapper, like aog's get_* helpers
```

`infer`:
1. Range: use `CriticalMovementEngine._parse_range` from `data/critical.py`. A
   date-only `end` covers the whole day.
2. Events: raw rows `timestamp, event_code, parameter` with `event_code IN
   _INFERENCE_CODES` and `start ≤ timestamp < end` (UTC epoch, converted with the
   project's timezone helpers as AogEngine does), ordered by timestamp. Plain SQL via
   `DatabaseManager`/`sqlite3` or an existing reader function, whichever AogEngine's
   pattern suggests. The cycles table isn't needed.
3. Config: `DatabaseManager.get_config_at_date(start_dt)`.
4. Candidate phases: when `use_ring_config` and the config has `RB_R1` and/or
   `RB_R2`, pass `phases` = the sorted union of every phase in them. Parse with
   `atspm.analysis.cycles._parse_ring_groups`. Otherwise pass `phases=None`.
5. `proposed = infer_detector_roles(events, phases=..., min_actuations=...)`.
6. `active_counts` = the number of Code-82 rows per `parameter` in the window.
   `diff = diff_detector_roles(proposed, parse_detector_roles(config), active_counts)`.
7. With `output_dir=None`, return `{"proposed": proposed, "diff": diff}`.
   Otherwise write `Detector_Inference_{stamp}.csv` (proposed) and
   `Detector_Inference_Diff_{stamp}.csv` (diff), where `stamp` follows
   `CriticalMovementEngine._write_outputs`'s rule (`2023_11_14-2023_11_16` for whole
   days), print one line per written file (`Wrote …`), and return `None`.
8. Always print a short summary: the count per status. Then list every diff row whose
   status is `conflict`, `new` or `silent` as
   `  P{phase or candidates} {detector}: {status} — configured {configured or '-'}; proposed {role} {phase|candidates} ({confidence})`.
   The exact format is free, but it must contain the status words.

An empty window must still return the diff (every configured channel `silent`). No
exceptions.

## 2. CLI: `atspm infer-detectors`

Follow `split-failures` / `aog` in `cli.py` exactly: the mutually exclusive required
`--target` / `--targetid` / `--all` group, `--start` / `--end` (dates), a
`handle_infer_detectors(args)` handler that loops targets like the other handlers, and
outputs to the intersection's outputs folder the way `aog` does. Options:
- `--min-actuations N` (int, default 50)
- `--all-phases` (store_true): don't limit candidates to the `RB_*` ring phases.

The help text says it proposes a detector configuration for review and never edits
`int_cfg.csv`.

## Stop and ask (write it in the report and stop) if

- an existing test fails and the fix would need an edit to a file you must not modify;
- the core's output can't be reproduced through the engine (e.g. the events query
  changes dtypes in a way `assert_frame_equal` rejects): describe it, don't edit tests.

## Report

`docs/specs/detector_inference_shell_REPORT.md`: files touched (one line each), the
final `pytest -q` line, anything you stopped on. Under 30 lines.

`docs/PENDING_DOC_CHANGES.md`: append exactly
- `- [src/atspm/cli.py] new infer-detectors subcommand`
- `- [src/atspm/data/__init__.py] export DetectorInferenceEngine, get_detector_inference`
