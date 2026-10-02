Last doc sync: ba3cac10fbd1db2b2dd186ea4d9cd916bee9e8b8

<!--
Queue, not a log — cleared on every doc sync. One bullet per doc-relevant
change, terse: `- [file/path.py] what changed, one phrase`.
Only log changes to: SQLite schema, CLI subcommands/flags, public
__init__.py exports, or the Functional Core/Imperative Shell boundary.
See CLAUDE.md "Documentation Workflow" for the rules.
-->

- [src/atspm/video/processor.py] accepts .ts input alongside .mp4; VideoOverlayResult gains timing_source
- [src/atspm/cli.py] video-overlay/-calibrate-shapes/-locate-phase-change --video document .ts input; --output restricted to writable containers
- [src/atspm/cli.py] new video-sync subcommand
- [src/atspm/data/video.py] lamp shape type and indication CSV column
- [src/atspm/video/__init__.py] export draw_lamp_overlay, LampMeasurement, measure_lamps, sync_video


- [src/atspm/analysis/detectors.py] analyze_discrepancies gains window=; flipping disagreements split; one-side-silent pairs yield none
- [src/atspm/plotting/detectors.py] plot_detector_comparison gains window=; hard-reset lines, per-pair summary, no layout shapes
- [src/atspm/data/detectors.py] DetectorEngine uses every config overlapping the window; fetches ±900 s edge margin
- [src/atspm/analysis/__init__.py] export clock-mark decoder (MarkerPeds, marker_peds_from_config, drop_marker_events, send_log_pulses, pair_marker_pulses, decode_clock_marks)
- [src/atspm/data/manager.py] int_cfg.csv Clk: category → Clk_Behind/Clk_Ahead/Clk_Set config columns
- [src/atspm/data/ingestion.py] backward-clock-step fences stored with parameter = -2 (comms gaps stay -1)
- [src/atspm/data/reader.py] check_data_quality: gap_count excludes clock-step fences; new clock_step_count
