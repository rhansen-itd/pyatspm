Last doc sync: 5fe391a8434e629edf0cb6b1bf8548c4bf8fbc45

<!--
Queue, not a log — cleared on every doc sync. One bullet per doc-relevant
change, terse: `- [file/path.py] what changed, one phrase`.
Only log changes to: SQLite schema, CLI subcommands/flags, public
__init__.py exports, or the Functional Core/Imperative Shell boundary.
See CLAUDE.md "Documentation Workflow" for the rules.
-->

- [src/atspm/cli.py] Added `video-calibrate-shapes` and `video-overlay` subcommands (single-target only, no `--all`).
- [src/atspm/video/__init__.py] New top-level package (peer to analysis/data/plotting/reports): exports `render_overlay`, `VideoOverlayResult`, `calibrate_shapes`, `draw_shape_overlay`, `draw_loop_overlay`, `draw_stopbar_overlay`.
- [src/atspm/data/__init__.py] New exports: `ShapeConfig`, `resolve_stopbar_target`, `OVERLAP_LETTER_MAP`.
- [src/atspm/data/video.py] New per-camera shape-config convention: `intersections/<folder>/video/<camera>_shapes.csv`.
- [src/atspm/video/overlay.py] New documented Functional Core mutation exception (in-place `cv2` frame drawing), parallel to the existing `.iterrows()` exceptions and `plotting/detectors.py`'s in-place `go.Figure` mutation.
- [src/atspm/analysis/video.py] New event-code handling: overlap codes 61/63/64/65/66 (previously unhandled anywhere in src/).
- [pyproject.toml] Added `opencv-python` dependency.
- [src/atspm/cli.py] Added `retrieve` subcommand (`--target`/`--targetid`/`--all`) that pulls `.datZ` files from devices via SCP, replacing the old `_Datz_SCP.ipynb` workflow; `setup` now also scaffolds `devices.json`.
- [src/atspm/data/__init__.py] New exports: `RetrievalEngine`, `run_retrieval`.
- [src/atspm/data/retrieval.py] New module; new per-intersection config file convention: `intersections/<folder>/devices.json` (list of device entries: role/device_type/port/user/password/remote_folder/last_retrieved/host). Host resolves from an explicit `host` field if present; only `role: controller` may omit it and fall back to `metadata.json`'s `controller_ip` (one controller per intersection, unambiguous). Every other role must set `host` explicitly — no fallback to `detection_ip`.
- [pyproject.toml] Added `paramiko`, `scp` dependencies.
- [src/atspm/data/video.py] Shape CSV reworked: `direction`/per-row `video_width`/`video_height` columns removed, `name` column added; resolution now lives in a one-row metadata header instead of repeating per shape.
- [src/atspm/video/calibrate.py] `calibrate_shapes()` gained a `save_path` param; it now owns saving (`'w'` to save, `'q'` prompts to save) instead of the caller saving afterward.
- [src/atspm/cli.py] `video-calibrate-shapes`/`video-overlay --video` now resolves relative paths against `<target>/video/` instead of the working directory; `video-overlay`'s default `--output` now nests under `<start-date>/` and includes `--start`'s time-of-day in the filename.
- [src/atspm/cli.py] Added `video-locate-phase-change` subcommand (single-target only, no `--all`): auto-selects the first green-to-yellow/yellow-to-red change for a phase at least `--min-offset` seconds into the video, renders a confirmation clip with a normalized (signed, zero-at-expected-frame) countdown label, or computes a corrected `--start` (= original + `--observed-delta`) given `--observed-delta`.
- [src/atspm/analysis/video.py] New function `first_phase_transition_after` (earliest-event-at-or-after lookup backing `video-locate-phase-change`).
- [src/atspm/video/__init__.py] New export: `extract_labeled_clip`.
