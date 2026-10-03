# Timing Actuation Shell Report (S-D5)

## Files Touched
- `src/atspm/data/timing_actuation.py` (created): TimingActuationEngine and get_timing_actuation
- `src/atspm/data/__init__.py`: exported TimingActuationEngine and get_timing_actuation
- `src/atspm/cli.py`: plot-timing-actuation CLI subcommand and parser
- `src/atspm/data/detector_health.py`: timing_plot command column on reported findings
- `docs/PENDING_DOC_CHANGES.md`: logged touched components
- `docs/specs/timing_actuation_shell_REPORT.md`: this report

## Test Result
- `tests/data/test_timing_actuation_engine.py`: 10 passed, 2 failed in 3.31s

## Stop and Ask / Issues Encountered
Two tests in `tests/data/test_timing_actuation_engine.py` fail due to constraints outside the permitted scope:
1. `TestEngine.test_no_output_dir_writes_nothing`:
   Asserts `list(tmp_path.iterdir()) == []`. The `seeded_db` fixture creates `test_intersection.db` inside `tmp_path`, so `tmp_path` is never empty. Fixing this requires modifying `tests/` (forbidden file).
2. `TestFindingLinks.test_reported_findings_carry_timing_plot`:
   Asserts detector 53 produces a `ConfiguredSilent` finding. `seeded_db` seeds only 14 h (06:00–20:00), so `share` (58.3%) falls below `HealthThresholds.min_observed_share` (90%), causing `_judgeable` to reject the day. Furthermore, `detector_health.py` lines 145-148 divide integer-second `datetime64[s]` timestamps by `1e9`, putting timestamps in 1969. Fixing requires editing `tests/` and/or `detector_health.py` outside §3.

## Deviations from Spec
None. Execution stopped per spec instructions ("an existing test fails for a reason you could only fix by changing a forbidden file").
