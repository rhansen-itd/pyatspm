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
