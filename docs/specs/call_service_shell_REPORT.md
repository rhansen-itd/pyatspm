# Call Service Shell Implementation Report

## Files Touched
- Created: `src/atspm/data/call_service.py` (`CallServiceEngine`, `get_ped_delay`, `get_wait_time`)
- Created: `src/atspm/plotting/call_service.py` (`plot_ped_delay`, `plot_wait_time`)
- Modified: `src/atspm/cli.py` (added `ped-delay` and `wait-time` subcommands and usage block)
- Modified: `src/atspm/data/__init__.py` (exported engine and convenience functions)
- Modified: `src/atspm/plotting/__init__.py` (exported plot functions)
- Modified: `docs/PENDING_DOC_CHANGES.md` (appended three doc change bullets)
- Created: `docs/specs/call_service_shell_REPORT.md`

## Test Result
- `1201 passed, 25 skipped, 30 warnings in 158.97s`
- `tests/data/test_call_service_engine.py`: 41 passed in 7.63s

## Ambiguities and Uncertainties
- None. The specification and tests in `tests/data/test_call_service_engine.py` provided an unambiguous contract.

## Deviations from Spec
- None. Implementation follows the spec verbatim.
