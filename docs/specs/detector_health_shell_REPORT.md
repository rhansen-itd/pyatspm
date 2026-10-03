# S-D4 Detector-Health Shell Report

## Files Touched
- `src/atspm/data/detector_health.py` (created `DetectorHealthEngine`, `get_detector_health`)
- `src/atspm/plotting/detector_health.py` (created `plot_detector_health`)
- `src/atspm/data/__init__.py` (exported `DetectorHealthEngine`, `get_detector_health`)
- `src/atspm/plotting/__init__.py` (exported `plot_detector_health`)
- `src/atspm/cli.py` (added `detector-health` subcommand, handler, and docstring entry)
- `src/atspm/reports/generators.py` (wired `_generate_detector_health_summary` hook)
- `docs/PENDING_DOC_CHANGES.md` (logged doc changes per CLAUDE.md)
- `docs/specs/detector_health_shell_REPORT.md` (this report)

## Test Results
`896 passed, 30 warnings in 175.14s (0:02:55)` (`PYTHONPATH=src .venv/bin/python -m pytest -q`)
Including contract tests:
- `tests/data/test_detector_health_engine.py`: 6 passed
- `tests/plotting/test_detector_health_plot.py`: 7 passed
- `tests/data/test_detector_findings_table.py`: 10 passed
- `tests/analysis/test_detector_health_config.py`: 28 passed

## Notes and Deviations
- None. No deviations from the spec; no forbidden files touched.
- `RecordCount` shell rule implemented using `DatabaseManager.get_ingestion_spans()` and `th.min_observed_share` per `(date, window)`.
- Re-runs over date ranges are idempotent and transactional via `DatabaseManager.replace_findings`.
