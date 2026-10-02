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
- [src/atspm/analysis/__init__.py] export discharge_profiles
- [src/atspm/cli.py] flow --stratify; flow now reads Det_P{N}_Stop_Bar as well as _Stopbar
- [src/atspm/analysis/__init__.py] export saturation_state (advisory)
- [src/atspm/analysis/flow.py] flow_rate cycle_df gains termination column (Flow_Cycle CSV too)
- [src/atspm/analysis/__init__.py] export optimize (new analysis/optimizer.py)
- [src/atspm/data/optimizer.py] new OptimizerEngine and get_optimization
- [src/atspm/plotting/optimizer.py] new plot_throughput_curve, plot_allocation, plot_marginal_rates
- [src/atspm/data/__init__.py] export OptimizerEngine, get_optimization
- [src/atspm/plotting/__init__.py] export plot_allocation, plot_marginal_rates, plot_throughput_curve
- [src/atspm/cli.py] new optimize subcommand


