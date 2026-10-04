# Approach Volume Shell (UDOT S-M8) Report

## Files Touched
- `src/atspm/data/approach_volume.py` (created)
- `src/atspm/plotting/approach_volume.py` (created)
- `src/atspm/cli.py` (added `approach-volume` subcommand)
- `src/atspm/data/__init__.py` (exported `ApproachVolumeEngine`, `get_approach_volume`)
- `src/atspm/plotting/__init__.py` (exported `plot_approach_volume`)
- `docs/PENDING_DOC_CHANGES.md` (appended doc change bullets)
- `docs/specs/approach_volume_shell_REPORT.md` (this report)

## Test Results
- `PYTHONPATH=src .venv/bin/python -m pytest tests/data/test_approach_volume_engine.py`:
  `24 passed in 3.99s`
- Full test suite `PYTHONPATH=src .venv/bin/python -m pytest -q`:
  `1302 passed, 30 skipped, 30 warnings in 170.67s`

## Ambiguities and Deviations
- Nothing was ambiguous; the spec and acceptance contract test suite were clear.
- No deviations from the specification.
