# Roadmap

Planned and deferred work only. The original backlog — a few months of Jules's accumulated review-only suggestions, triaged against the live code — was split into self-contained sessions (A–F, plus the Video Overlay build and audit) and worked through. That history lives in git; the decisions those sessions settled are reflected in [architecture.md](architecture.md) and the test suite. What remains below is the open fix candidates, one deferred decision, a future feature, and a standing "don't re-investigate" list.

## Open fix candidates

Small and actionable; each carries enough file/line detail to be picked up cold.

- **`backfill_ring_phases` can never backfill the last pending cycle.** It fetches events only up to `max(cycle_start) + 1s`, so the final pending cycle's greens fall outside the window and that row is never populated. Fix: extend the fetch window past the last pending cycle (e.g. to the next cycle start, or the end of events). Small, Sonnet-sized. (Surfaced while verifying the gap-rule fixes.)
- **Video phase lookup shows `'Y'` through red clearance.** `_build_phase_intervals` emits `clear_end_ts` = Code 11 (End RC) — correct for `phase_splits`' combined yellow+red-clearance reporting, but wrong for a per-frame visual lookup, so a phase serving red clearance displays yellow the whole time. The overlap builder already handles this correctly (Code 64 ends its `'Y'`). Pinned as a strict xfail at `tests/analysis/test_video_status.py::TestPhaseRedClearance`. Fix (behavior change): expose the end-of-yellow boundary separately — e.g. a `yellow_end_ts` column on phase intervals — and have the video lookup use it, leaving `phase_splits` untouched.
- **`resolve_stopbar_target` has no phase-number range check.** It accepts any integer as a phase number. Low-risk input-validation tightening.
- **Fable edge-case pass on the SQLite fixture smoke tests.** `tests/data/test_manager.py` and `tests/data/test_reader.py` carry happy-path smoke tests plus `# TODO(fable):` stubs for the real edge cases: `get_metadata` falling back to `{"timezone": "US/Mountain"}` when the `metadata` table is *missing* vs. *present-but-empty* (two distinct code paths in `manager.py`), and `check_data_quality` completeness scoring around gap markers (marker on the window boundary, `gap_count > event_count` flooring at 0.0, the zero-event-count denominator guard). Adversarial edge-case reasoning → a Fable lane, carrying the standing `event_code == -1` gap-marker audit mandate (CLAUDE.md §5).

## Deferred — needs a project-wide decision first

- **`get_phase_splits()`'s 10-keyword-argument signature (`data/phases.py:423`).** Real, but *every* Engine's `get_X` convenience wrapper has a similarly long explicit kwarg list by convention (`get_vehicle_counts`, `get_arrival_on_green`, …). Fixing this one in isolation would make it inconsistent with its siblings. If worth doing, decide on a project-wide convention (e.g. a shared options dataclass) first, then apply everywhere at once — don't one-off it.

## Future features

- **Turning-movement counting via computer vision** (`EnhancedIOUTracker`, background-subtraction or YOLOv8 vehicle detection, approach-line crossing logic — carried over from the earlier `video_processing.py`). Deliberately split out of the Video Overlay work: model selection/tuning and tracking-accuracy validation is a substantially larger, different kind of problem. Revisit if there's appetite for it.

## Rejected / no action

Logged here so these don't get re-suggested or re-investigated later.

- **8 of 9 "Unused Import" suggestions were false positives** — checked every one against current usage; only `Literal` in `data/phases.py` was real (since removed). The rest (`argparse`/`sys` in `cli.py`, `List` in `reports/generators.py`, `Optional` in `utils/timezone.py`/`data/reader.py`, `Any`/`Dict` in `utils/logging.py`, `Dict` in `plotting/termination.py`, `pd` in `data/ingestion.py`/`data/detectors.py`) are all genuinely used. Likely stale from before recent edits.
- **`reports/__init__.py:22` "__all__ unused"** — references a class `ReportGenerator` that doesn't exist; the actual export is `PlotGenerator`. Stale/incorrect suggestion.
- **"Code Duplication in counts endpoints"** (`get_vehicle_counts`/`get_ped_counts`/`get_combined_counts`, `data/counts.py:595+`) — these are intentional thin wrappers matching the established Engine + `get_X` convenience-function pattern used everywhere else in `data/` (`AogEngine`/`get_arrival_on_green`, `PhaseEngine`/`get_phase_splits`, etc.). Collapsing them into one generic dispatcher would break consistency with their siblings for no real benefit — not recommended.
- **"Many Arguments" and "Overly Long" on `plot_coordination`** — both reference an old 11-argument signature that no longer exists. The function has already been refactored to 6 arguments and decomposed into ~15 helper functions (`_add_ring_bars`, `_add_coord_plan_markers`, `_add_detector_traces`, etc.). Stale — already addressed.
- **Notebook "Read stdout once Fix" / "List Files Fix"** (`notebooks/_Datz_SCP.ipynb`) — these already appear implemented, and are not part of the package itself (personal SCP automation). No action.
