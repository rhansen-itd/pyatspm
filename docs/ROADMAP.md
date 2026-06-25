# Roadmap

Working backlog, triaged from Jules's accumulated review-only suggestions (a few months' worth, collected while suggestions ran unattended). Each suggestion was checked against the current code before being included here — many had gone stale. Items are grouped into self-contained sessions; each has enough file/line detail to be picked up cold in a future conversation. Pick one session at a time.

## Session A — Security hardening in config import (`data/manager.py`)

Two real SQL-identifier-handling gaps, same root cause, same file — fix together.

- `add_config_column()` (`manager.py:190`): `column_type` is f-string-interpolated into `ALTER TABLE config ADD COLUMN "{safe}" {column_type}` with no validation. Currently unreachable in practice — the only caller (`import_config`, line 242) always uses the default `"TEXT"` — but the method is public API, so any external script calling it with an untrusted `column_type` has an injection point. Fix: allow-list against SQLite's affinity types (`TEXT`, `INTEGER`, `REAL`, `BLOB`, `NUMERIC`) and raise on anything else.
- `_insert_config_row()` (`manager.py:324`): column names are f-string-quoted (`f'"{c}"'`) **without** the `.replace('"', '""')` escaping that `add_config_column` already does for the same kind of identifier. Column names here come from `int_cfg.csv` row labels (`_transform_config_column`), so a malformed/malicious CSV with a `"` in a Category/Parameter cell could break out of the identifier into the `INSERT OR REPLACE` statement. Fix: apply the same escaping pattern used in `add_config_column`.
- While in this area: `get_gap_prev` / `get_gap_next` (`manager.py:474` and `497`) are two ~10-line methods with identical SQL shape (`MAX(timestamp) WHERE ... <= ?` vs `MIN(timestamp) WHERE ... >= ?`). Worth collapsing into one private helper taking direction as a parameter — small, low-risk, same file as the rest of this session.

## Session B — Deduplicate `_compute_bin_quality` (data layer)

Confirmed triplicated, near-verbatim, across three Engine classes:
- `CountEngine._compute_bin_quality` — `data/counts.py:374`
- `PhaseEngine._compute_bin_quality` — `data/phases.py:212`
- `AogEngine._compute_bin_quality` — `data/aog.py:294` (its own docstring literally says "Mirrors `PhaseEngine._compute_bin_quality` exactly")

All three build a full bin grid from `ingestion_log` spans, compute coverage, and downgrade to `"partial"` when a gap marker falls in the bin. Extract to a shared helper (e.g. `utils/quality.py:compute_bin_quality(...)`) matching the existing precedent of `utils/timezone.resolve_pytz` — a cross-engine pure function, not tied to any one Engine's state. Needs a careful read of all three copies first in case one has drifted from the others in a way that matters (e.g. different bin-edge handling) rather than assuming they're truly identical.

## Session C — Safe `iterrows()` → `itertuples()` swaps (mechanical, low priority)

Five confirmed, still-accurate locations. Each is a small, pre-filtered DataFrame already deliberately using row-iteration for sequential state-machine logic (see `docs/architecture.md` "Vectorization" note) — `itertuples` doesn't change that, it's a free, behavior-preserving speedup (attribute access instead of dict-like), not a redesign. Zero risk, batch all five in one pass:
- `data/processing.py:233` — span loop in `CycleProcessor`
- `plotting/coordination.py:567` — combo bar rendering
- `plotting/coordination.py:612` — GYR hover lookup dict
- `plotting/detectors.py:487` — anomaly overlay rendering
- `analysis/detectors.py:44` — `_reconstruct_intervals` interval state machine

Also bundle in: unused `Literal` import in `data/phases.py:25` (the one genuine unused-import finding — see "Rejected" below for the other eight).

## Session D — Long-function refactors (judgment-heavy, do opportunistically)

Not urgent; tackle individually, not as a batch, since each needs domain understanding:

- `_build_parser()` in `cli.py:1262-1742` (~480 lines) — still fully accurate. Good candidate for one `_add_<subcommand>_parser(subparsers)` function per subcommand, mirroring the existing per-subcommand `handle_*` convention.
- `assign_ring_phases()` in `analysis/cycles.py:120-258` (~138 lines) — genuinely long, nested ring-membership/merge_asof logic. This is Functional Core cycle-detection code that was just touched by the coord_plan bug fix (see recent commits) — give it its own session rather than stacking another change on top immediately.
- `get_phase_splits()`'s 10-keyword-argument signature (`data/phases.py:423`) — real, but **every** Engine's `get_X` convenience wrapper in this codebase has a similarly long explicit kwarg list by convention (`get_vehicle_counts`, `get_arrival_on_green`, etc.). Fixing just this one in isolation would make it inconsistent with its siblings. If this is worth doing, decide on a project-wide convention (e.g. a shared options dataclass) first, then apply it everywhere at once — don't one-off it.

## Session E — Establish test infrastructure, then add tests

There is currently no test suite at all (no `tests/` directory, no `pytest` dependency). The five "missing tests" suggestions are really one foundational gap, not five independent fixes:

1. Add `pytest` as a dev dependency and a `tests/` directory layout.
2. Phase 1 — pure Functional Core, no mocking needed (start here): `analysis/decoders.py` (`parse_datz_bytes` et al.), `analysis/cycles.py:validate_cycles`, `utils/timezone.py:resolve_pytz`.
3. Phase 2 — I/O-bound, needs a throwaway SQLite fixture via `DatabaseManager`: `data/manager.py:get_metadata` fallback path (missing `metadata` table → `{"timezone": "US/Mountain"}`), `data/reader.py:check_data_quality`.

## Rejected / no action

Logged here so these don't get re-suggested or re-investigated later.

- **8 of 9 "Unused Import" suggestions were false positives** — checked every one against current usage; only `Literal` in `data/phases.py` (folded into Session C) was real. The rest (`argparse`/`sys` in `cli.py`, `List` in `reports/generators.py`, `Optional` in `utils/timezone.py`/`data/reader.py`, `Any`/`Dict` in `utils/logging.py`, `Dict` in `plotting/termination.py`, `pd` in `data/ingestion.py`/`data/detectors.py`) are all genuinely used. Likely stale from before recent edits.
- **`reports/__init__.py:22` "__all__ unused"** — references a class `ReportGenerator` that doesn't exist; the actual export is `PlotGenerator`. Stale/incorrect suggestion.
- **"Code Duplication in counts endpoints"** (`get_vehicle_counts`/`get_ped_counts`/`get_combined_counts`, `data/counts.py:595+`) — these are intentional thin wrappers matching the established Engine + `get_X` convenience-function pattern used everywhere else in `data/` (`AogEngine`/`get_arrival_on_green`, `PhaseEngine`/`get_phase_splits`, etc.). Collapsing them into one generic dispatcher would break consistency with their siblings for no real benefit — not recommended.
- **"Many Arguments" and "Overly Long" on `plot_coordination`** — both reference an old 11-argument signature (`split_failures_df`, `aog_df`, `phase_splits_df`, `hide_uncoordinated`, etc.) that no longer exists. The function has already been refactored to 6 arguments and decomposed into ~15 helper functions (`_add_ring_bars`, `_add_coord_plan_markers`, `_add_detector_traces`, etc.). Stale — already addressed.
- **Notebook "Read stdout once Fix" / "List Files Fix"** (`notebooks/_Datz_SCP.ipynb`) — Jules's own rationale notes these already appear implemented. Not part of the package itself (personal SCP automation script). No action.
