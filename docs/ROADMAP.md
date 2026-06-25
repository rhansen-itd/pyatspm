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

## New Feature — Video Overlay (planning session, not a direct-implementation session)

**Goal:** Take a video feed of an intersection and overlay live detector and phase status visually (loop/stopbar shapes recolored by current state), reconstructing the visualization built in the legacy `spmfunctions/video_processing.py` (`VideoProcessor.overlay_shapes`, `process_video`) against the new SQLite/events architecture instead of the old flat-pickle format.

**Scope note:** The legacy file also contains a turning-movement counter (`EnhancedIOUTracker`, `count_turning_movements`, `crosses_approach_line`, YOLOv8/background-subtraction vehicle tracking). That's a separate, much larger problem (object detection/tracking model selection and tuning) and is deliberately **out of scope** here — logged as its own future item below, not bundled in.

**How to use this entry as a prompt:** start the future session with a planning/strategizing conversation before writing any code — work through the open questions below together, settle the module boundaries, then implement. Don't jump straight to code from this prompt.

### Old → new architecture mapping

The legacy code operates on the pre-rewrite flat DataFrame format (`TS_start`, `Code`, `ID` — the direct ancestor of the new `events` table's `timestamp`/`event_code`/`parameter`) and three `spmfunctions.misc_tools` helpers that forward-fill state into continuous status columns. That module isn't available anywhere in this repo or elsewhere on disk — don't guess at its internals; re-derive the needed behavior from scratch against the new schema.

| Legacy (`misc_tools`, internals unavailable) | Produces | New-architecture equivalent |
|---|---|---|
| `detector_status(df, relevant_detectors)` | `Det {id} Status` (On/Off), continuous | Codes 81/82 in `events` — need a new continuous-status lookup, not yet built |
| `phase_status(df, relevant_phases)` | `Ph {n} Status` (G/Y/R/Rc), continuous | Same codes (1,8,9,10,11,12) already used by `analysis/phases.py:_build_phase_intervals` — that gives discrete intervals, not a continuous forward-filled status |
| `overlap_status(df, relevant_overlaps)` | `OL{A..P} Status`, continuous | **No equivalent exists yet** — the new schema/config has no overlap concept at all (see open question 1 below) |
| `comb_gyr_det(df)` | preprocessing step, exact internals unknown | n/a — re-derive from scratch against `get_events_with_cycles_df` rather than guessing at this function |

None of the three status functions need to be ported as-is. What overlay needs is a **point-in-time status lookup**: "what was phase 4 / detector 38 / overlap C doing at timestamp T," for an arbitrary stream of frame timestamps. The closest existing pattern in the new codebase is `analysis/aog.py`'s `np.searchsorted`-based interval-containment check (green window vs. detector timestamp) — reuse that vectorized pattern against `_build_phase_intervals`-style intervals, rather than porting the old forward-fill approach.

**Known legacy bug not to port:** `video_processing.py`'s `load_and_process_data()` builds per-frame status via `pd.merge_asof(expanded_df, df, on='TS_start', direction='forward')` — `direction='forward'` finds the *next* event at-or-after each frame timestamp, which is backwards for a "current status as of now" lookup. The more recently-written `process_video()` method does it correctly elsewhere (`df[df['TS_start'] <= timestamp].iloc[-1]` — most recent status at-or-before). Use the `process_video` semantics, not `load_and_process_data`'s.

### Open design questions for the planning session

1. **Overlap support.** Stopbars in the shape CSV can be tagged with a phase number *or* an overlap code (`"OLA"`–`"OLP"`, mapped to 1–26). The current schema/config (`config` table, `int_cfg.csv` categories `TM:`/`RB:`/`Det:`/`WD:`) has no overlap concept. Does overlap status need first-class data-layer support (a new config category, e.g. `OL:`), or can overlaps be treated as ordinary phase-numbered events from the controller's point of view (i.e. the controller already logs overlap state changes under the same event codes 1/8/9/10/11/12 with a parameter in a distinct ID range)? Needs checking against real controller log data before deciding.
2. **Where does shape config live?** The shape CSV (columns: `type, points, color, input, phase, direction, video_width, video_height`) is per-camera (tied to a specific `video_width`/`video_height`), not per-signal-timing-period like `int_cfg.csv`. Likely needs its own file per intersection folder (e.g. `intersections/<folder>/video/<camera_name>_shapes.csv`), separate from `int_cfg.csv`. Confirm folder convention in the planning session.
3. **Resolution mismatch.** The CSV records the resolution the shapes were drawn at. The legacy code has no rescaling logic if the input video's actual resolution differs — decide whether the new version should validate/reject mismatches or rescale points proportionally.
4. **Functional Core purity for frame drawing.** OpenCV draws (`cv2.polylines`, `cv2.fillPoly`) mutate the frame array in place rather than returning a new one — copying every frame for purity would be wasteful for video. Decide whether to accept in-place mutation as a pragmatic, documented exception to the Functional Core rule for this module specifically, the way the `iterrows()` exceptions are already documented in `architecture.md`.
5. **Shape calibration tool packaging.** The legacy `draw_shapes_interface()` is an interactive Tkinter+OpenCV point-and-click tool (draw/edit/drag/undo loop/stopbar shapes on a frame, save to CSV) — a one-time-per-camera calibration step, not a batch operation. Decide whether it becomes an `atspm video calibrate-shapes` CLI command that opens an interactive window, or stays a standalone script outside the CLI's `--target`/`--targetid`/`--all` convention (interactive GUI tools don't fit that pattern naturally).

### Reference files for the future prompt

- **Attach `video_processing.py`.** Specifically valuable for: the shape CSV load/save round-trip (`save_shapes_to_csv`/`load_shapes_from_csv`), the `overlay_shapes()` recoloring logic, and the interactive `draw_shapes_interface()` calibration tool (mouse-driven point editing, undo, mode-cycling — a non-trivial UI state machine that would lose fidelity if re-derived from a prose description alone). Ignore everything related to `EnhancedIOUTracker`, `count_turning_movements`, and `crosses_approach_line` — that's the deferred turning-movement-counting feature below, not this one.
- **Attach `vid_cfg720.csv`** (or any real shape-config export) as a concrete example of the format — already described in the table above, but a real example removes ambiguity about edge cases (e.g. the three `OLD` stopbar rows in that file sharing one overlap tag across multiple line segments).
- **Skip `Video_Overlay.txt`** — it's an earlier, strictly simpler draft of the same `VideoProcessor` class (no edit mode, no overlap/direction support) that's fully superseded by `video_processing.py`. Attaching it adds no information.

### Future / deferred

- **Turning-movement counting via computer vision** (`EnhancedIOUTracker`, background-subtraction or YOLOv8 vehicle detection, approach-line crossing logic — same legacy `video_processing.py`). Deliberately split out of Video Overlay above — model selection/tuning and tracking-accuracy validation is a substantially larger, different kind of problem. Revisit once Video Overlay is built and there's appetite for it.

## Rejected / no action

Logged here so these don't get re-suggested or re-investigated later.

- **8 of 9 "Unused Import" suggestions were false positives** — checked every one against current usage; only `Literal` in `data/phases.py` (folded into Session C) was real. The rest (`argparse`/`sys` in `cli.py`, `List` in `reports/generators.py`, `Optional` in `utils/timezone.py`/`data/reader.py`, `Any`/`Dict` in `utils/logging.py`, `Dict` in `plotting/termination.py`, `pd` in `data/ingestion.py`/`data/detectors.py`) are all genuinely used. Likely stale from before recent edits.
- **`reports/__init__.py:22` "__all__ unused"** — references a class `ReportGenerator` that doesn't exist; the actual export is `PlotGenerator`. Stale/incorrect suggestion.
- **"Code Duplication in counts endpoints"** (`get_vehicle_counts`/`get_ped_counts`/`get_combined_counts`, `data/counts.py:595+`) — these are intentional thin wrappers matching the established Engine + `get_X` convenience-function pattern used everywhere else in `data/` (`AogEngine`/`get_arrival_on_green`, `PhaseEngine`/`get_phase_splits`, etc.). Collapsing them into one generic dispatcher would break consistency with their siblings for no real benefit — not recommended.
- **"Many Arguments" and "Overly Long" on `plot_coordination`** — both reference an old 11-argument signature (`split_failures_df`, `aog_df`, `phase_splits_df`, `hide_uncoordinated`, etc.) that no longer exists. The function has already been refactored to 6 arguments and decomposed into ~15 helper functions (`_add_ring_bars`, `_add_coord_plan_markers`, `_add_detector_traces`, etc.). Stale — already addressed.
- **Notebook "Read stdout once Fix" / "List Files Fix"** (`notebooks/_Datz_SCP.ipynb`) — Jules's own rationale notes these already appear implemented. Not part of the package itself (personal SCP automation script). No action.
