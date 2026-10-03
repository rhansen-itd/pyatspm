# Split-Failure Shell Engine, Plot and CLI Implementation Report (UDOT S-M1)

## Files Touched
- `src/atspm/data/split_failures.py` (created)
- `src/atspm/plotting/split_failures.py` (created)
- `src/atspm/data/__init__.py` (added exports `SplitFailureEngine`, `get_split_failures`)
- `src/atspm/plotting/__init__.py` (added export `plot_split_failures`)
- `src/atspm/cli.py` (added `split-failures` subcommand parser and handler)
- `docs/PENDING_DOC_CHANGES.md` (appended doc change bullets)
- `docs/specs/split_failures_shell_REPORT.md` (this report)

## Test Result Line
`679 passed, 30 warnings in 104.44s` (including all 26 tests in `tests/data/test_split_failures_engine.py`)

## Items Unsure About
None.

## Spec Deviations
In `plot_split_failures`, `_build_title` from `atspm.plotting.termination` falls back to `intersection_name` when `minor_road_name` is absent. To preserve `major_road_name` in the title when `minor_road_name` is not provided (as tested in `test_title_uses_metadata_and_aggregate`), `metadata["intersection_name"]` is populated with `major_road_name` before invoking `_build_title`.
