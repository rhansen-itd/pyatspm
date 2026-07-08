# Roadmap

Working backlog, triaged from Jules's accumulated review-only suggestions (a few months' worth, collected while suggestions ran unattended). Each suggestion was checked against the current code before being included here — many had gone stale. Items are grouped into self-contained sessions; each has enough file/line detail to be picked up cold in a future conversation. Pick one session at a time.

## Execution plan (model-lane assignments)

The remaining work is split across two execution models, chosen by the rule **spend the scarce Fable window only where a weaker model produces plausibly-but-subtly-wrong output** — correctness-critical work that benefits from adversarial edge-case reasoning, not work that's merely large.

- **Fable** — math, deep Functional Core refactoring, security hardening, complex/edge-case test generation. Every Fable session below carries an explicit `event_code == -1` gap-marker audit mandate (CLAUDE.md §5).
- **Sonnet/Opus** — structural refactoring, boilerplate, I/O, CLI, basic fixtures.

**Jules note:** Session C *looked* like Jules's job (mechanical `iterrows` swaps) but its locations are in `plotting/` and `analysis/` — zones AGENTS.md explicitly forbids Jules from touching. It went to Sonnet instead. Jules has no assigned work in this plan; the only in-bounds item (the unused `Literal` import) was folded into Session C rather than split into its own PR.

### Sequence & status

| # | Session | Model | Status |
|---|---|---|---|
| Sonnet #1 | E — pytest infra + `tests/` layout | Sonnet/Opus | ✅ done |
| Sonnet #2 | C — `iterrows` → `itertuples` swaps (+ unused `Literal`) | Sonnet/Opus | ✅ done (commit `0f7f3a8`) |
| Sonnet #3 | D — `_build_parser()` decomposition | Sonnet/Opus | ✅ done |
| Fable #1 | A — SQL-identifier security hardening + injection audit | Fable | ✅ done |
| Fable #2 | E Phase 1 — Functional Core edge-case tests | Fable | ✅ done (1 gap-rule bug found → Session F) |
| Fable #3 | B — `_compute_bin_quality` dedup + drift audit | Fable | ✅ done |
| Sonnet #4 | E Phase 2 — SQLite fixture scaffolding | Sonnet/Opus | ⬜ pending |
| Fable #4 | D — `assign_ring_phases()` refactor | Fable | ✅ done (1 gap-rule bug found → Session F) |
| Fable #5 | Video Overlay — audit + edge-case tests | Fable | ⬜ pending (feature already shipped; see below) |
| Fable #6 | F — fix the two deferred gap-rule bugs | Fable | ⬜ pending (see Session F) |

Hard dependency: Fable #2 needs Sonnet #1 (done). Everything else is parallelizable.

**Test suite:** 86 passed, 2 xfailed. The two `xfail(strict=True)` markers are the deferred gap-rule bugs tracked in Session F — they are not failures, they are pinned known defects that flip to failures the moment the buggy behavior changes.

Highest-value Fable targets in the remaining window, ranked: **Fable #2** (edge-case test suite is foundational and pure adversarial-reasoning work), **Fable #5** (audits brand-new, untested, math-heavy video code that ships with a self-documented residual bug), **Fable #1** (the *audit* for other injection points, not the two known patches), **Fable #4** (Functional Core cycle logic just touched by a bug fix, high regression risk), then **Fable #3** (mostly DRY, but the drift check is correctness-sensitive).

## Session A — Security hardening in config import (`data/manager.py`) — Fable #1

Two real SQL-identifier-handling gaps, same root cause, same file — fix together. Fable also runs a full-file injection audit, not just the two known patches.

- `add_config_column()` (`manager.py:190`): `column_type` is f-string-interpolated into `ALTER TABLE config ADD COLUMN "{safe}" {column_type}` with no validation. Currently unreachable in practice — the only caller (`import_config`, line 242) always uses the default `"TEXT"` — but the method is public API, so any external script calling it with an untrusted `column_type` has an injection point. Fix: allow-list against SQLite's affinity types (`TEXT`, `INTEGER`, `REAL`, `BLOB`, `NUMERIC`) and raise on anything else.
- `_insert_config_row()` (`manager.py:324`): column names are f-string-quoted (`f'"{c}"'`) **without** the `.replace('"', '""')` escaping that `add_config_column` already does for the same kind of identifier. Column names here come from `int_cfg.csv` row labels (`_transform_config_column`), so a malformed/malicious CSV with a `"` in a Category/Parameter cell could break out of the identifier into the `INSERT OR REPLACE` statement. Fix: apply the same escaping pattern used in `add_config_column`.
- While in this area: `get_gap_prev` / `get_gap_next` (`manager.py:474` and `497`) are two ~10-line methods with identical SQL shape (`MAX(timestamp) WHERE ... <= ?` vs `MIN(timestamp) WHERE ... >= ?`). Worth collapsing into one private helper taking direction as a parameter — small, low-risk, same file as the rest of this session.
- **Audit mandate (the reason this is Fable, not Sonnet):** sweep the entire file for every other value reaching SQL via f-string/`%`/`.format`/concatenation rather than a `?` placeholder; distinguish identifiers (quote+escape) from values (parameterize). **Gap-marker check:** confirm the merged `get_gap_prev/next` helper preserves the exact `<=`/`>=` boundary semantics and never bridges across an `event_code == -1` marker in a way the originals didn't.

## Session B — Deduplicate `_compute_bin_quality` (data layer) — Fable #3

Confirmed triplicated, near-verbatim, across three Engine classes:
- `CountEngine._compute_bin_quality` — `data/counts.py:374`
- `PhaseEngine._compute_bin_quality` — `data/phases.py:212`
- `AogEngine._compute_bin_quality` — `data/aog.py:294` (its own docstring literally says "Mirrors `PhaseEngine._compute_bin_quality` exactly")

All three build a full bin grid from `ingestion_log` spans, compute coverage, and downgrade to `"partial"` when a gap marker falls in the bin. Extract to a shared helper (e.g. `utils/quality.py:compute_bin_quality(...)`) matching the existing precedent of `utils/timezone.resolve_pytz` — a cross-engine pure function, not tied to any one Engine's state.

**Drift audit first (why this is Fable):** produce a line-level diff of all three copies before writing anything — they are *assumed* identical but may have drifted on bin-edge handling, coverage computation, or the gap-marker downgrade. If they've materially drifted, stop and report rather than silently flattening. **Gap-marker correctness (CLAUDE.md §5):** preserve the `event_code == -1` → `"partial"` bin downgrade exactly, and add a focused test for a marker landing exactly on a bin edge (where drift would hide).

## Session C — Safe `iterrows()` → `itertuples()` swaps — Sonnet #2 ✅ DONE

Completed in commit `0f7f3a8`. All five deliberate row-iteration loops swapped (`data/processing.py:233`, `plotting/coordination.py:567` & `:612`, `plotting/detectors.py:487`, `analysis/detectors.py:44`) — behavior-preserving, sequential state-machine logic left intact per the `architecture.md` "Vectorization" note. The one genuine unused-import finding (`Literal` in `data/phases.py:25`) removed in the same pass.

## Session D — Long-function refactors (judgment-heavy, do opportunistically)

Not a batch; each needs domain understanding.

- **`_build_parser()` (`cli.py`) — Sonnet #3 ✅ DONE.** Decomposed into 14 `_add_<subcommand>_parser(subs)` functions mirroring the existing `handle_*` convention; generated argparse namespace unchanged (same subcommands, flags, defaults, and the `--target`/`--targetid`/`--all` mutually-exclusive group).
- **`assign_ring_phases()` (`analysis/cycles.py`) — Fable #4 ✅ DONE.** Decomposed into four pure helpers (`_extract_green_events`, `_assign_greens_to_cycles`, `_ring_phase_strings`, `_merge_ring_phase_strings`); public signature and output unchanged (verified against a 301-case randomized golden master plus 27 regression tests in `tests/analysis/test_assign_ring_phases.py`, written before the refactor). **Gap-marker audit finding:** marker rows themselves can never become phases (code-1 filter), but the backward `merge_asof` has no tolerance/segment awareness, so a green landing *after* a `-1` hard reset yet *before* the first post-gap `cycle_start` is attributed to the last pre-gap cycle — a CLAUDE.md §5 violation. Current behavior is pinned by a characterization test; the correct behavior is a strict-xfail test. **Fix candidate (behavior change, future session):** make green→cycle assignment segment-aware via `_segment_id`.
- **`get_phase_splits()`'s 10-keyword-argument signature (`data/phases.py:423`) — deferred, needs a project-wide decision.** Real, but **every** Engine's `get_X` convenience wrapper has a similarly long explicit kwarg list by convention (`get_vehicle_counts`, `get_arrival_on_green`, etc.). Fixing just this one in isolation would make it inconsistent with its siblings. If worth doing, decide on a project-wide convention (e.g. a shared options dataclass) first, then apply everywhere at once — don't one-off it.

## Session E — Establish test infrastructure, then add tests

1. **Infra — Sonnet #1 ✅ DONE.** `pytest` added as a dev dependency (`pyproject.toml` `[tool.pytest.ini_options]` + `requirements.txt`) and a `tests/` package layout created mirroring `src/atspm/` (`tests/analysis/`, `tests/data/`, `tests/utils/`, `conftest.py`).
2. **Phase 1 — pure Functional Core, no mocking — Fable #2 ⬜.** `analysis/decoders.py` (`parse_datz_bytes` et al.), `analysis/cycles.py:validate_cycles`, `utils/timezone.py:resolve_pytz`. Adversarial edge cases are the point: truncated/empty `.datz` buffers, a decode stream *containing* an `event_code == -1` marker (assert it survives intact), `validate_cycles` on cycles straddling a `-1` hard reset (assert it does **not** interpolate duration/pairing across it — the highest-value assertion), out-of-order/duplicate `cycle_start`, and `resolve_pytz` on garbage zones / DST boundaries (assert the *actual* fallback in the code, don't invent one). Report any latent bugs the edge cases expose as separate fix candidates.
3. **Phase 2 — I/O-bound, throwaway SQLite fixture via `DatabaseManager` — Sonnet #4 ⬜.** Fixture scaffolding (tmp_path DB, WAL, real schema code — not hand-written DDL, plus a seed helper that can insert a `-1` gap marker), then happy-path smoke tests for `data/manager.py:get_metadata` and `data/reader.py:check_data_quality`. Leave the real edge-case assertions (missing `metadata` table → `{"timezone": "US/Mountain"}` fallback, quality gaps) as `# TODO(fable):` stubs for a Fable edge-case pass.

## Session F — Fix the two deferred gap-rule bugs (from Fable #2 & #4) — Fable #6

Two genuine CLAUDE.md §5 violations surfaced by the edge-case test work, each pinned as `@pytest.mark.xfail(strict=True)` with full rationale in the test file — so they fail loudly the moment the behavior changes rather than rotting silently. Neither was fixed in the session that found it because each needs a signature/behavior change out of that session's scope. Same root cause (sequential/duration logic must stop at an `event_code == -1` hard reset), so fix together.

- **`validate_cycles` misreports a data outage as an over-long cycle** — `analysis/cycles.py`; xfail at `tests/analysis/test_cycles.py:115`. The function receives only `cycle_start` values, so it diffs the last pre-gap cycle against the first post-gap cycle straight across the gap. Fix: pass gap information into `validate_cycles` (gap timestamps, or the events frame) and suppress any cycle-length check whose interval spans a `-1` marker. The signature change ripples to the imperative-shell callers — update them.
- **`assign_ring_phases` attributes a post-gap green to the pre-gap cycle** — `analysis/cycles.py`; xfail at `tests/analysis/test_assign_ring_phases.py:240`. Its backward `merge_asof` has no tolerance and no segment awareness, so a green landing after a `-1` marker but before the first post-gap `cycle_start` binds backward across the reset. Fix: derive a `_segment_id` (incremented at each `-1` marker) and constrain assignment to within a segment.
- **Acceptance signal:** flip both xfails to passing — remove the `strict=True` markers once fixed, and update the companion `test_characterization_*` guards to the corrected behavior. Re-read Fable #4's characterization tests first so the fix preserves the mid-cycle-marker case, which is *correct* today and must stay that way.

## Video Overlay — SHIPPED (was a planning entry; now built)

The Video Overlay feature described in earlier revisions of this roadmap **has been implemented** across commits `4653fc8` → `c37c25a` → `dd38a38`. The former "open design questions" are resolved in code:

- **Module layout:** `src/atspm/video/` (`overlay.py`, `processor.py`, `calibrate.py`), plus `analysis/video.py` (Functional Core status lookups) and `data/video.py` (`ShapeConfig` CSV I/O).
- **Overlap support (Q1):** overlap status is derived from the event stream in `analysis/video.py`.
- **Shape config location (Q2):** `intersections/<folder>/video/<camera>_shapes.csv` (`cli.py:_video_shape_path`).
- **Resolution mismatch (Q3):** resolved as **reject** — `ShapeConfig.validate_resolution()` (`data/video.py:173`) raises `ValueError`, called from `processor.py:116`.
- **Functional Core purity (Q4):** resolved as a **documented in-place-mutation exception** in `video/overlay.py`'s module docstring, citing the `_add_anomaly_overlays` precedent.
- **CLI packaging (Q5):** three single-target subcommands — `video-calibrate-shapes`, `video-overlay`, `video-locate-phase-change` (`cli.py:1259+`), documented as no-`--all` because calibration is interactive and overlay targets one camera's video.
- **Gap markers:** `_GAP_CODE = -1` filtered in the status builders; the `np.searchsorted` interval-containment pattern (`analysis/video.py:279`) is used as intended.

### Follow-up — Video Overlay audit + edge-case tests — Fable #5 ⬜

The feature is built but **untested**, math-heavy, and gap-marker-sensitive — a top Fable target. Audit and lock behavior down with tests; do not rewrite wholesale.

- **Gap-marker audit (priority):** trace every status-lookup path (`_status_at_timestamps` and the overlap/phase/detector builders) and prove no status forward-fills across a `-1` marker — a frame after a gap but before the next real event must read "unknown/reset", never the stale pre-gap color.
- **Known residual bug:** `analysis/video.py:52` documents a real risk — an *inferred* steady-red period whose gap crosses a segment boundary can be mislabeled `'R'`. Reproduce with a concrete failing test, characterize exactly when it fires (check the `side='right'-1` searchsorted at line 279 against markers landing exactly on `green_ts`), and propose the fix.
- **Edge-case matrix:** searchsorted query before first / after last / exactly on a boundary / empty interval frame; `ShapeConfig` malformed CSV, save→load round-trip fidelity, `validate_resolution` off-by-one; overlap event→status mapping against real controller codes (flag any unvalidated parameter-ID-range assumption).

### Future / deferred

- **Turning-movement counting via computer vision** (`EnhancedIOUTracker`, background-subtraction or YOLOv8 vehicle detection, approach-line crossing logic — from the legacy `video_processing.py`). Deliberately split out of Video Overlay: model selection/tuning and tracking-accuracy validation is a substantially larger, different kind of problem. Revisit now that Video Overlay is built and if there's appetite for it.

## Rejected / no action

Logged here so these don't get re-suggested or re-investigated later.

- **8 of 9 "Unused Import" suggestions were false positives** — checked every one against current usage; only `Literal` in `data/phases.py` (folded into Session C, now done) was real. The rest (`argparse`/`sys` in `cli.py`, `List` in `reports/generators.py`, `Optional` in `utils/timezone.py`/`data/reader.py`, `Any`/`Dict` in `utils/logging.py`, `Dict` in `plotting/termination.py`, `pd` in `data/ingestion.py`/`data/detectors.py`) are all genuinely used. Likely stale from before recent edits.
- **`reports/__init__.py:22` "__all__ unused"** — references a class `ReportGenerator` that doesn't exist; the actual export is `PlotGenerator`. Stale/incorrect suggestion.
- **"Code Duplication in counts endpoints"** (`get_vehicle_counts`/`get_ped_counts`/`get_combined_counts`, `data/counts.py:595+`) — these are intentional thin wrappers matching the established Engine + `get_X` convenience-function pattern used everywhere else in `data/` (`AogEngine`/`get_arrival_on_green`, `PhaseEngine`/`get_phase_splits`, etc.). Collapsing them into one generic dispatcher would break consistency with their siblings for no real benefit — not recommended.
- **"Many Arguments" and "Overly Long" on `plot_coordination`** — both reference an old 11-argument signature (`split_failures_df`, `aog_df`, `phase_splits_df`, `hide_uncoordinated`, etc.) that no longer exists. The function has already been refactored to 6 arguments and decomposed into ~15 helper functions (`_add_ring_bars`, `_add_coord_plan_markers`, `_add_detector_traces`, etc.). Stale — already addressed.
- **Notebook "Read stdout once Fix" / "List Files Fix"** (`notebooks/_Datz_SCP.ipynb`) — Jules's own rationale notes these already appear implemented. Not part of the package itself (personal SCP automation script). No action.
