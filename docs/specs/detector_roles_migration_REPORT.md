# Detector Roles Migration Report (S-D0)

## Files Touched
- `src/atspm/analysis/__init__.py`: exported `parse_detector_roles` and `detector_sets`.
- `src/atspm/analysis/critical.py`: removed private parsers/regexes and switched `movement_phase_map` to `detector_sets`.
- `src/atspm/data/aog.py`: migrated `_resolve_detector_map` to `detector_sets` for `arrival` role.
- `src/atspm/data/flow.py`: migrated `_resolve_detector_map` to `detector_sets` for `stop_bar` role.
- `src/atspm/data/split_failures.py`: migrated presence set extraction to `detector_sets` for `occupancy` role.
- `src/atspm/data/optimizer.py`: migrated both stop-bar set calls to `detector_sets` for `stop_bar` role.
- `src/atspm/data/reader.py`: rebuilt `get_det_config` on the detector role table via `groupby(["phase", "role"])`.
- `src/atspm/cli.py`: updated flow and critical help and docstrings to reference `P{N} Stop Bar` rows.
- `docs/PENDING_DOC_CHANGES.md`: appended export notice for `parse_detector_roles` and `detector_sets`.

## Pytest Summary
728 passed, 3 skipped, 31 warnings in 120.34s (0:02:00)

## Items Stopped On
None.

## Behaviour Changes
No functional behaviour changes. All call sites now resolve detector keys through `parse_detector_roles` and `detector_sets`, normalizing both stop-bar spellings and ensuring ascending sorted detector lists.
