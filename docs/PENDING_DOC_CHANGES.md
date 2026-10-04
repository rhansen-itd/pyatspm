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
- [src/atspm/analysis/__init__.py] export validate_plans (new module analysis/optimizer_validation.py)
- [src/atspm/data/optimizer.py] new OptimizerEngine.validate and get_validation
- [src/atspm/data/__init__.py] export get_validation
- [src/atspm/cli.py] optimize subcommand gains --validate and validation flags
- [src/atspm/analysis/__init__.py] export clock-mark decoder (MarkerPeds, marker_peds_from_config, drop_marker_events, send_log_pulses, pair_marker_pulses, decode_clock_marks)
- [src/atspm/data/manager.py] int_cfg.csv Clk: category → Clk_Behind/Clk_Ahead/Clk_Set config columns
- [src/atspm/data/ingestion.py] backward-clock-step fences stored with parameter = -2 (comms gaps stay -1)
- [src/atspm/data/reader.py] check_data_quality: gap_count excludes clock-step fences; new clock_step_count
- [src/atspm/data/__init__.py] export ClockMarkEngine, get_clock_marks
- [src/atspm/plotting/__init__.py] export plot_clock_drift
- [src/atspm/cli.py] new clock-drift subcommand
- [src/atspm/data/split_failures.py] new SplitFailureEngine and get_split_failures
- [src/atspm/plotting/split_failures.py] new plot_split_failures
- [src/atspm/data/__init__.py] export SplitFailureEngine, get_split_failures
- [src/atspm/plotting/__init__.py] export plot_split_failures
- [src/atspm/cli.py] new split-failures subcommand
- [src/atspm/analysis/__init__.py] export split_failures, bin_split_failures (new analysis/split_failures.py)
- [src/atspm/data/split_failures.py] split-failures reads Det_P{N}_Occupancy (presence), not Stop_Bar
- [src/atspm/analysis/__init__.py] export parse_detector_roles, detector_sets
- [src/atspm/cli.py] split-failures --aggregate gains any (worst-lane)
- [src/atspm/cli.py] new infer-detectors subcommand
- [src/atspm/data/__init__.py] export DetectorInferenceEngine, get_detector_inference
- [src/atspm/analysis/__init__.py] export detector_activity_profile (new analysis/detector_activity.py)
- [src/atspm/analysis/__init__.py] export HealthThresholds, detector_health_findings, onset_bursts (new analysis/detector_health.py)
- [src/atspm/analysis/detector_activity.py] detector_activity_profile gains max_silence_s (unmarked silences > 1 h read as gaps)
- [src/atspm/data/manager.py] new detector_findings table (S-D4); clear_ingested_data clears it; new replace_findings/get_findings
- [src/atspm/analysis/__init__.py] export wd_reboot_windows, wd_profile_windows, wd_units, wd_ignore, apply_ignore, wd_thresholds, filter_min_severity, severity_exit_code
- [src/atspm/data/__init__.py] export DetectorHealthEngine, get_detector_health
- [src/atspm/plotting/__init__.py] export plot_detector_health
- [src/atspm/cli.py] new detector-health subcommand
- [src/atspm/analysis/__init__.py] export TIMING_CODES, timing_actuation_intervals, timing_actuation_rows, ring_phase_order, finding_plot_windows (new analysis/timing_actuation.py)
- [src/atspm/plotting/__init__.py] export plot_timing_actuation
- [src/atspm/cli.py] new plot-timing-actuation subcommand
- [src/atspm/data/__init__.py] export TimingActuationEngine, get_timing_actuation
- [src/atspm/data/detector_health.py] reported findings / CSV gain timing_plot command column
- [src/atspm/analysis/__init__.py] export PREEMPT_CODES, preempt_episodes, preempt_summary (new analysis/preempt.py)
- [src/atspm/cli.py] new preempt subcommand
- [src/atspm/data/__init__.py] export PreemptEngine, get_preempt
- [src/atspm/analysis/__init__.py] export approach_delay, bin_approach_delay (new analysis/approach_delay.py), arrival_travel_times
- [int_cfg.csv / config] new optional Det: P{N} Arrival Travel key (seconds) → Det_P{N}_Arrival_Travel
