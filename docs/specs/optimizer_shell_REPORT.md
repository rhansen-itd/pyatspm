# Optimizer Shell Implementation Report

## Files Touched
- `src/atspm/data/optimizer.py` (created): `OptimizerEngine` and `get_optimization`
- `src/atspm/plotting/optimizer.py` (created): `plot_throughput_curve`, `plot_allocation`, `plot_marginal_rates`
- `src/atspm/data/__init__.py`: exported `OptimizerEngine`, `get_optimization`
- `src/atspm/plotting/__init__.py`: exported `plot_allocation`, `plot_marginal_rates`, `plot_throughput_curve`
- `src/atspm/cli.py`: added `optimize` subcommand and updated module docstring
- `docs/PENDING_DOC_CHANGES.md`: appended doc-relevant change bullets

## Test Results
- `tests/data/test_optimizer_engine.py`: 21 passed in 10.61s
- Full test suite (`pytest -q`): 508 passed, 21 warnings in 98.05s

## Ambiguities and Deviations
- None. The specification was followed exactly. All acceptance tests passed on the first run without deviations.
