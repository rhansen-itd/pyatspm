# Left Turn Gap Shell Report (UDOT S-M9)

## Touched Files
- `src/atspm/data/left_turn_gap.py`: created `LeftTurnGapEngine` and `get_left_turn_gap`.
- `src/atspm/plotting/left_turn_gap.py`: created `plot_left_turn_gap`.
- `src/atspm/data/__init__.py`: exported `LeftTurnGapEngine` and `get_left_turn_gap`.
- `src/atspm/plotting/__init__.py`: exported `plot_left_turn_gap`.
- `src/atspm/cli.py`: added `left-turn-gap` subcommand parser and handler.
- `docs/PENDING_DOC_CHANGES.md`: appended doc bullets for CLI and exports.
- `docs/specs/left_turn_gap_shell_REPORT.md`: this report.

## Test Results
- `tests/data/test_left_turn_gap_engine.py`: `32 passed, 1 warning in 7.79s`
- Full test suite: `1364 passed, 31 skipped, 31 warnings in 201.10s`

## Ambiguities and Formatting Decisions
- Summary line in CLI: printed `({n_censored} censored)` when `n_censored > 0` (matching the `(1 censored)` example in the spec) and omitted when 0.
- Opposing phase candidates in unresolved phase warning: formatted as `, candidates P{...}` when candidates exist in `through_phases`.

## Deviations
- None.
