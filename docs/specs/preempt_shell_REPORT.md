# S-M6 Preemption Shell Implementation Report

## Touched Files
- `src/atspm/data/preempt.py` (created): `PreemptEngine` and `get_preempt` convenience wrapper with `FETCH_MARGIN_S = 1800.0`, timezone handling, range parsing, vectorized CSV formatting (`call_on_local`, `timing_plot`), and output generation.
- `src/atspm/cli.py` (edited): Added `atspm preempt` subcommand, `_preempt_single_intersection`, `handle_preempt`, and registered `_add_preempt_parser`.
- `src/atspm/data/__init__.py` (edited): Exported `PreemptEngine` and `get_preempt`.
- `docs/PENDING_DOC_CHANGES.md` (edited): Appended bullets for `cli.py` and `data/__init__.py`.
- `docs/specs/preempt_shell_REPORT.md` (created): This report.

## Test Results
`972 passed, 30 warnings in 167.83s (0:02:47)` via `PYTHONPATH=src .venv/bin/python -m pytest -q` (including all 6 test cases in `tests/data/test_preempt_engine.py`).

## Notes & Deviations
- Unsure about: Nothing; spec and contracts were clear and unambiguous.
- Deviations: None; executed exactly as specified.
