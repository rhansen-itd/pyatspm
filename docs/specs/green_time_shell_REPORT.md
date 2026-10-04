# Green Time Utilization Shell Report

## Touched Files
- `src/atspm/data/green_time_utilization.py` (created): `GreenTimeEngine`, `get_green_time`
- `src/atspm/plotting/green_time_utilization.py` (created): `plot_green_time`
- `src/atspm/cli.py`: `green-time` subcommand, parser, and handlers
- `src/atspm/data/__init__.py`: exported `GreenTimeEngine` and `get_green_time`
- `src/atspm/plotting/__init__.py`: exported `plot_green_time`
- `docs/PENDING_DOC_CHANGES.md`: recorded doc change bullets

## Test Results
- `tests/data/test_green_time_engine.py`: 28 passed in 6.91s
- Full test suite: 1253 passed, 28 skipped, 31 warnings in 168.98s

## Observations & Deviations
- No deviations from the specification.
- No ambiguities encountered; all contracts and golden tests passed as specified.
