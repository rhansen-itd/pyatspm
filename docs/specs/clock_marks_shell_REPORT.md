# Clock-Marks Shell, Plot, and CLI Report

## Files Touched
- `src/atspm/data/clock_marks.py`: Created `ClockMarkEngine` and `get_clock_marks` wrapper.
- `src/atspm/plotting/clock_marks.py`: Created pure `plot_clock_drift` Plotly function.
- `src/atspm/data/__init__.py`: Exported `ClockMarkEngine` and `get_clock_marks`.
- `src/atspm/plotting/__init__.py`: Exported `plot_clock_drift`.
- `src/atspm/cli.py`: Added `atspm clock-drift` subcommand, parser, and handler.
- `docs/PENDING_DOC_CHANGES.md`: Appended doc-relevant changes.
- `docs/specs/clock_marks_shell_REPORT.md`: Created this report.

## Test Result
- `.venv/bin/python -m pytest -q`: 469 passed, 32 warnings in 68.37s.
- `tests/data/test_clock_marks_engine.py`: 12 passed.

## Uncertainties
- None.

## Deviations
- None. Spec executed exactly as written.
