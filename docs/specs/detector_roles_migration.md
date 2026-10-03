# Spec: migrate detector-key parsing onto the role table (UDOT S-D0)

The pure core already exists and is tested: `src/atspm/analysis/detector_roles.py`
(`parse_detector_roles(config) -> DataFrame[detector, phase, role, movement, partner, key]`
and `detector_sets(roles, role) -> {phase: frozenset}`). Read its module
docstring first. Your job: make every shell/core call site that parses
`Det_P{N}_*` keys use it, and delete the per-module parsers.

## Acceptance

`PYTHONPATH=src .venv/bin/python -m pytest -q` passes in full, including
`tests/analysis/test_detector_roles.py` and `tests/data/test_detector_role_callsites.py`.
Two tests in the latter skip in a worktree (they need the untracked corpus DBs);
that is expected.

## Files you must NOT modify

- `tests/**` (every test file, fixtures included)
- `src/atspm/analysis/detector_roles.py`
- `src/atspm/plotting/**` (the coordination plot keeps its `"P{N} {Type}"` input format)
- `src/atspm/data/manager.py` (its `_parse_detector_pairs` stays)
- `src/atspm/analysis/split_failures.py`, `analysis/flow.py`, `analysis/aog.py`,
  `analysis/counts.py`, `analysis/detectors.py`, and every other `analysis/` module not listed below
- `docs/ROADMAP.md`, `docs/UDOT_MOE_ROADMAP.md`, `README.md`, `docs/*.md` other than the two named below

## Files you may edit

- `src/atspm/analysis/critical.py`
- `src/atspm/analysis/__init__.py` (exports only)
- `src/atspm/data/aog.py`, `data/flow.py`, `data/split_failures.py`, `data/optimizer.py`,
  `data/critical.py`, `data/reader.py`
- `src/atspm/cli.py` (help/description strings only, see step 6)
- **append** to `docs/PENDING_DOC_CHANGES.md`
- **create** `docs/specs/detector_roles_migration_REPORT.md`

## Steps

1. **`analysis/__init__.py`**: export `parse_detector_roles` and `detector_sets`
   from `.detector_roles` (add a `- detector_roles: ...` line to the module
   docstring list, and add both names to `__all__` if the file has one).

2. **`analysis/critical.py`**: delete `_STOPBAR_KEY_RE`, `_OCCUPANCY_KEY_RE`,
   `_parse_stopbar_sets`, `_parse_occupancy_sets`, `_parse_detector_sets` (and the
   `re` import if unused). In `movement_phase_map`, replace
   `_parse_stopbar_sets(config)` with
   `detector_sets(parse_detector_roles(config), "stop_bar")`. Behaviour must not
   change (`tests/analysis/test_critical.py` pins it). Leave
   `parse_movements_from_config` use as it is.

3. **Shell resolvers.** Keep each method's name, signature, return type, warning
   text and phase filtering; only the parsing changes. Return lists **sorted
   ascending** (the tests expect it).
   - `data/aog.py` `AogEngine._resolve_detector_map`: role `"arrival"`. Replace the
     whole hand-rolled `startswith/endswith/key[5:...]` loop.
   - `data/flow.py` `FlowRateEngine._resolve_detector_map`: role `"stop_bar"`.
   - `data/split_failures.py`: `presence_sets = detector_sets(parse_detector_roles(config), "occupancy")`.
     Update the module docstring line that names `_parse_occupancy_sets`.
   - `data/optimizer.py`: both `_parse_stopbar_sets(config)` calls →
     `detector_sets(parse_detector_roles(config), "stop_bar")`; fix the import.
   - `data/critical.py`: docstring only, if it names the removed helpers.
   Import from `..analysis.detector_roles`.

4. **`data/reader.py` `get_det_config`**: rebuild it on the role table. Output stays
   `{"P{phase} {Label}": "<ids>"}` for the coordination plot, with
   - `Label` = `Arrival` / `Stop Bar` / `Occupancy` for roles `arrival` / `stop_bar` /
     `occupancy`; no other roles (no `Pairs`, `TM`, `WD`);
   - `<ids>` = the phase's detector IDs, sorted ascending, comma-joined, no spaces;
   - both stop-bar spellings merged into one `"P{N} Stop Bar"` entry.
   No row iteration over the role table is needed: a `groupby(["phase","role"])` works.
   Update its docstring (Google style).

5. Search `src/` for any other parsing of `Det_P` keys or of the removed helper
   names and migrate it (`grep -rn "Det_P\|_parse_stopbar_sets\|_parse_occupancy_sets" src/`).
   `tests/data/test_detector_role_callsites.py::test_no_detector_key_parsing_outside_role_module`
   is the check. **Exception:** `data/manager.py` (excluded above).

6. **`cli.py` help strings** that say the flow/critical/optimizer commands read
   `Det_P{N}_Stopbar keys`: change to say they read the `P{N} Stop Bar` rows
   (`Det_P{N}_Stop_Bar`; `Det_P{N}_Stopbar` also accepted). No argument changes.

7. Terminology: no "legacy"/"orphan" in anything you write.

## Stop and ask (write it in the report and stop) if

- an existing test outside the two new files fails after your change and the fix
  would need a behaviour change rather than a parsing change;
- a call site needs a role the table doesn't have;
- you find detector-key parsing in `plotting/` that the tests flag (they shouldn't).

## Report

Write `docs/specs/detector_roles_migration_REPORT.md`: files touched (one line
each), the final `pytest -q` summary line, anything you stopped on, and any
behaviour you think changed. Under 40 lines.

`docs/PENDING_DOC_CHANGES.md`: append `- [src/atspm/analysis/__init__.py] export parse_detector_roles, detector_sets`
and nothing else (the other changes are internal).
