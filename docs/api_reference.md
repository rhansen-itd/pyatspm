# API Reference

Public exports per package, as declared in each `__init__.py`. Most users should reach this through the CLI (see [cli_reference.md](cli_reference.md)); this is for scripting against the package directly.

## `atspm.data` — Imperative Shell

| Function / Class | Description |
|---|---|
| `DatabaseManager(db_path)` | Context manager for direct DB access (raw `sqlite3`); `get_metadata()`, `get_timezone()`, `set_metadata()`, `import_config()`, span/anchor queries, `clear_ingested_data()` (drops all `events`/`cycles`/`ingestion_log` rows, leaving `config`/`metadata` intact — backs `atspm process --rebuild`) |
| `init_db(db_path)` | Create a new intersection DB with the full schema (WAL mode, all tables/indexes) |
| `import_config(csv_path, db_path)` | Import `int_cfg.csv` into the `config` table |
| `db_timezone(db_path, timezone=None)` | The zone to use for an intersection: *timezone* if given, else the DB's `metadata.timezone`, else `DEFAULT_TIMEZONE`. Every engine's timezone resolution goes through here |
| `RetrievalEngine(target_dir, meta, devices)` | Pulls new `.datZ` files for every device in a parsed `devices.json`, secondary devices before controller; `run()` returns per-device result dicts |
| `run_retrieval(target_dir, meta, devices_path)` | Module-level convenience wrapper around `RetrievalEngine` — loads/saves `devices.json` for the caller |
| `IngestionEngine` | Orchestrates `.datZ` file scanning, parsing, gap detection, backward-clock-step fencing, and triggers cycle processing; `get_ingestion_stats()` reports `files_processed`, `total_events`, `gap_markers`, `header_mismatches`, `clock_steps`, `span_count`, `date_range` |
| `run_ingestion(db_path, data_dir, timezone, incremental, batch_size)` | Ingest `.datZ` files into `events` |
| `AchdIngestionEngine(db_path, raw_data_dir, intersection_id, timezone=None)` | Ingests ACHD high-resolution event CSV exports (`{id}_Events_*.csv`) into a normalized DB (events + comms-gap fencing, `ingestion_log` spans, `metadata`) |
| `run_achd_ingestion(db_path, raw_data_dir, intersection_id, timezone=None, rebuild=False)` | Module-level convenience wrapper around `AchdIngestionEngine`; returns an ingestion-stats dict. Backs `atspm ingest-achd` |
| `CycleProcessor` | Orchestrates cycle detection re-entry (fast-append vs. gap-fill paths) |
| `run_cycle_processing(db_path, reprocess)` | Detect and store `cycles` |
| `get_events_with_cycles_df(db_path, start, end, event_codes)` | Main reader — flat events+cycles DataFrame for a window |
| `get_events_with_cycles_df_by_date(db_path, date_str)` | Convenience — full local day |
| `get_coordination_data(...)` | Reader for `plot_coordination` inputs |
| `get_config_df(db_path, date)` | Active config row as `pd.Series` |
| `get_config_dict(db_path, date)` | Active config row as `dict` |
| `get_det_config(...)` | Resolved detector pair/arrival config for a date |
| `get_date_range(db_path, timezone=None)` | Min/max event timestamps in the DB, as tz-aware local datetimes |
| `get_available_dates(db_path)` | All local dates with computed cycles |
| `check_data_quality(db_path, start, end, timezone=None)` | Event/gap/cycle counts for a window. `gap_count` counts comms-gap markers only; backward-clock-step fences are reported separately as `clock_step_count` |
| `convert_to_datetime(...)` | Timestamp/timezone conversion helper |
| `CountEngine` | Counts orchestration; `vehicle_counts()`, `ped_counts()`, `combined_counts()` |
| `get_vehicle_counts(...)`, `get_ped_counts(...)`, `get_combined_counts(...)` | Module-level convenience wrappers around `CountEngine` |
| `PhaseEngine` | Phase-splits orchestration; `phase_splits(...)` |
| `get_phase_splits(...)` | Module-level convenience wrapper around `PhaseEngine` |
| `AogEngine` | Arrival-on-Green orchestration; `arrival_on_green(...)` |
| `get_arrival_on_green(...)` | Module-level convenience wrapper around `AogEngine` |
| `FlowRateEngine` | Flow-rate orchestration; `flow(...)` — resolves `Det_P<N>_Stopbar` detectors, calls the Core, optionally writes CSV + HTML |
| `get_flow_rate(db_path, start, end, phases=None, plans=None, pct=1.0, max_lost=10.0, split_tolerance=0.1, normalize="end_shift", fixed_lost=None, rolling=5, make_plot=True, output_dir=None, timezone=None)` | Module-level convenience wrapper around `FlowRateEngine` |
| `CriticalMovementEngine` | Critical-movement orchestration; `critical(...)` — pulls counts via `CountEngine`, resolves ring/barrier structure, calls the Core |
| `get_critical_movements(db_path, start, end, bin_len=15, basis="per_lane", exclude_missing=True, output_dir=None, timezone=None)` | Module-level convenience wrapper around `CriticalMovementEngine` |
| `DetectorEngine` | Detector-discrepancy orchestration; `get_discrepancies()`, `get_plot_data()` |
| `get_detector_discrepancies(...)` | Module-level convenience wrapper around `DetectorEngine` |
| `SplitFailureEngine` / `get_split_failures(db_path, start, end, phases=None, aggregate="union", threshold=0.79, ror_seconds=5.0, include_yellow=False, bin_len=60, exclude_missing=False, make_plot=True, output_dir=None, timezone=None)` | Purdue split failures (GOR vs ROR5) from `Det_P<N>_Occupancy` presence detectors; backs `atspm split-failures` |
| `ApproachDelayEngine` / `get_approach_delay(db_path, start, end, phases=None, travel_time_sec=0.0, bin_len=15, exclude_missing=False, make_plot=True, output_dir=None, timezone=None)` | Approach delay and arrival shares from advance detectors; backs `atspm approach-delay` |
| `ApproachVolumeEngine` / `get_approach_volume(db_path, start, end, bin_len=15, make_plot=True, output_dir=None, timezone=None)` | Approach volume with peak hour / PHF / K / D; backs `atspm approach-volume` |
| `YellowRedEngine` / `get_yellow_red(db_path, start, end, phases=None, role="stop_bar", severe_sec=4.0, bin_len=15, use_exclusions=True, make_plot=True, output_dir=None, timezone=None)` | Yellow/red actuation counts and violations; backs `atspm yellow-red` |
| `GreenTimeEngine` / `get_green_time(db_path, start, end, phases=None, role="stop_bar", bin_s=2.0, bin_len=15, use_overlap=False, use_exclusions=True, max_green_s=120.0, make_plot=True, output_dir=None, timezone=None)` | Green time utilization; backs `atspm green-time` |
| `LeftTurnGapEngine` / `get_left_turn_gap(db_path, start, end, lefts=None, bin_len=15, edges=(1.0,3.3,3.7,7.4,inf), trend_s=7.4, critical_s=None, use_exclusions=True, write_gaps=False, make_plot=True, output_dir=None, timezone=None)` | Permissive-left gap counts in opposing through traffic; backs `atspm left-turn-gap` |
| `CallServiceEngine` / `get_ped_delay(...)`, `get_wait_time(...)` | Pedestrian delay and vehicle wait time from call→service pairing; back `atspm ped-delay` / `atspm wait-time` |
| `SplitMonitorEngine` / `get_split_monitor(db_path, start, end, phases=None, percentiles=(50.0,85.0), make_plot=True, output_dir=None, timezone=None)` | Split-monitor services, per-plan stats, and timeline; backs `atspm split-monitor` |
| `OptimizerEngine` / `get_optimization(db_path, start, end, saturated, plans=None, pct=1.0, split_tolerance=0.1, stratify=False, max_lost=10.0, sat_threshold=0.8, demand_stat="mean", default_min_split=10.0, c_min=60.0, c_max=220.0, c_step=1.0, flat_tol_pct=1.0, boundary_rate_tol=100.0, bin_len=15, exclude_missing=True, make_plot=True, output_dir=None, timezone=None)` | Throughput-maximizing cycle length and split optimizer; backs `atspm optimize` |
| `get_validation(db_path, start, end, saturated, plans=None, ..., min_plan_cycles=30, split_cover_tol=1.0, rank_deadband_pct=2.0, change_tol_pp=3.0, output_dir=None, timezone=None)` | `OptimizerEngine.validate` — tests the throughput model against existing TOD plans; backs `atspm optimize --validate` |
| `ClockMarkEngine` / `get_clock_marks(db_path, start, end, send_log_path=None, output_dir=None, timezone=None)` | Decodes pedestrian-call clock marks, measures controller clock drift, finds correction sets; backs `atspm clock-drift` |
| `PreemptEngine` / `get_preempt(db_path, start, end, output_dir=None, timezone=None)` | Preemption episode and summary tables; backs `atspm preempt` |
| `DetectorInferenceEngine` / `get_detector_inference(db_path, start, end, use_ring_config=True, min_actuations=50, output_dir=None, timezone=None)` | Proposes a detector configuration from actuation behaviour (never edits config); backs `atspm infer-detectors` |
| `DetectorHealthEngine` / `get_detector_health(db_path, start, end, window="day", min_severity="low", thresholds=None, output_dir=None, timezone=None)` | Deterministic detector-health rules; writes the `detector_findings` table and CSV/HTML; backs `atspm detector-health`. Reported findings carry a `timing_plot` column with the `plot-timing-actuation` command to inspect each finding |
| `TimingActuationEngine` / `get_timing_actuation(db_path, start, end, phases=None, detectors=None, output_dir=None, timezone=None)` | Per-phase timing intervals, calls, ped service, and detector actuations; backs `atspm plot-timing-actuation` |
| `ShapeConfig` | Per-camera loop/stopbar shape config; `load(path)`/`save(path)` round-trip a `<camera>_shapes.csv`, `validate_resolution(w, h)`, `relevant_phases()`/`relevant_overlaps()`/`relevant_detectors()` |
| `resolve_stopbar_target(phase_field)` | Resolves a stopbar shape's `phase` field to a `(kind, number)` lookup target — `kind` is `"phase"` or `"overlap"`. Raises `ValueError` on a non-numeric, non-overlap field, and separately on a phase number outside `MIN_PHASE_NUMBER`-`MAX_PHASE_NUMBER` |
| `OVERLAP_LETTER_MAP` | `dict` mapping overlap letters `"OLA"`-`"OLP"` to numbers `1`-`16` |
| `MIN_PHASE_NUMBER` / `MAX_PHASE_NUMBER` | `1` / `16` — the valid signal phase range, matching `OVERLAP_LETTER_MAP`'s span from the same Hi-Res Enumerations spec |
| `SyncItem` / `SyncResult` / `ItemState` | Project-agnostic dataclasses for the local↔archive data sync (`atspm sync`): a syncable component, the outcome of copying it, and its two-sided state |
| `sync_item(item, *, direction, release=False, checksum=True, dry_run=False, log=...)` | Copy one `SyncItem` in a direction (`pull`/`push`), verifying the copy; returns a `SyncResult` |
| `item_state(item, *, checksum=False)` | Inspect a `SyncItem`'s local/archive presence, sizes, and agreement; returns an `ItemState` |
| `copy_verify_file(src, dst, *, checksum=True)` / `copy_verify_dir(src, dst, *, checksum=True, log=...)` | Copy a file/directory and verify by size (and SHA-256 unless skipped) |
| `human_bytes(n)` | Format a byte count as a human-readable string |

## `atspm.analysis` — Functional Core

Pure functions: DataFrames/dicts in, DataFrames/dicts/figures out. No I/O.

| Function / Class | Module | Description |
|---|---|---|
| `parse_datz_bytes(raw_bytes, file_timestamp)` | `decoders` | Decode one `.datZ` payload → `DataFrame[timestamp, event_code, parameter]`. Binary offsets are measured from the header instant, so the header's sub-minute delta is added to `file_timestamp` to form the event base; files with no header line fall back to `file_timestamp` unchanged |
| `parse_datz_batch(file_data)` | `decoders` | Decode and merge multiple files |
| `parse_datz_header(raw_bytes)` | `decoders` | Read the `Controller Data Log Beginning` instant → dict of `year`/`month`/`day`/`hour`/`minute` plus `second_offset` (seconds past the HH:MM boundary), or `None` when no valid header line is present. Scans the text preamble only |
| `validate_datz_file(raw_bytes)` | `decoders` | Quick validity check |
| `estimate_event_count(raw_bytes)` | `decoders` | Pre-parse row-count estimate |
| `insert_gap_marker(df, gap_timestamp)` | `decoders` | Insert an `event_code = -1` discontinuity row |
| `detect_corruption(raw_bytes)` | `decoders` | Heuristic corruption check |
| `DatZDecodingError` | `decoders` | Raised on unparseable `.datZ` input |
| `parse_achd_header(header_text)` | `achd` | Parse an ACHD export's text header → `AchdHeader(signal_id, signal_name, start, end, total_events)` |
| `achd_events_from_frame(raw, timezone)` | `achd` | Convert a raw ACHD event CSV frame → standard `DataFrame[timestamp, event_code, parameter]` in UTC epoch seconds |
| `AchdHeader` | `achd` | Dataclass of parsed ACHD header fields |
| `AchdDecodingError` | `achd` | Raised on unparseable ACHD CSV input |
| `calculate_cycles(events_df, config)` | `cycles` | Cycle-start detection (Code-31 barrier pulses, or ring-barrier fallback) |
| `assign_ring_phases(cycles_df, events_df, config)` | `cycles` | Adds `r1_phases`/`r2_phases` to a cycles DataFrame |
| `assign_events_to_cycles(events_df, cycles_df)` | `cycles` | `merge_asof` join of events onto their owning cycle |
| `validate_cycles(cycles_df, min_cycle_length=10.0, max_cycle_length=300.0, gap_timestamps=None)` | `cycles` | Sanity checks (duplicate/short/long cycles); pass `gap_timestamps` (a Series of `event_code = -1` marker times) to exclude intervals straddling a hard reset from the length checks |
| `get_cycle_stats(cycles_df)` | `cycles` | Summary statistics dict |
| `CycleDetectionError` | `cycles` | Raised when cycle detection cannot proceed |
| `vehicle_counts(events_df, movements, exclusions=None, bin_len=60, hourly=False, include_detectors=False)` | `counts` | Per-movement vehicle volume table |
| `ped_counts(events_df, bin_len=60, hourly=False)` | `counts` | Per-phase pedestrian service table (Code 21 paired with a preceding Code 45) |
| `parse_movements_from_config(config)` | `counts` | Parses `TM_*` config keys into a movement→detector-ID map |
| `parse_exclusions_from_config(config)` | `counts` | Parses `TM_Exclusions` JSON |
| `analyze_discrepancies(events_df, detector_pairs, lag_threshold_sec=2.0, window=None)` | `detectors` | Classifies co-located detector disagreements as `extended_disagreement` or `isolated_pulse`. Flipping disagreements are split and one-side-silent pairs yield none; `window=(start, end)` clips to an epoch range |
| `phase_splits(events_df, bin_len="cycle", report_mode="seconds", phases=None, include_no_clearance=False)` | `phases` | Per-cycle/binned green-yellow-red-clearance timing table |
| `arrival_on_green(events_df, phase, detector_ids, arrival_offset_sec=0.0)` | `aog` | Per-cycle Arrival on Green for one phase |
| `bin_arrival_on_green(cycle_df, bin_len=60)` | `aog` | Aggregates per-cycle AOG into fixed time bins |
| `flow_rate(events_df, phase, detector_ids, max_lost=10.0, plans=None)` | `flow` | Per-cycle and per-vehicle stop-bar departure tables for one phase; returns `(cycle_df, vehicle_df)`. `cycle_df` carries a `termination` column |
| `rate_profiles(cycle_df, vehicle_df, pct=1.0, split_tolerance=0.1, normalize="end_shift", fixed_lost=None, grid_step=0.5, min_cycles=5, stratify=False)` | `flow` | Collapses qualifying cycles onto a common elapsed-time grid; returns `(rate_df, inst_df, summary_df)`. `stratify=True` pools the busiest `pct` within each `(plan, split)` stratum |
| `discharge_profiles(cycle_df, vehicle_df, pct=1.0, split_tolerance=0.1, stratify=False, grid_step=0.5, min_cycles=5, rolling=5)` | `flow` | Per-phase cumulative discharge curves consumed by `optimize`; returns `(curve_df, summary_df)` |
| `saturation_state(cycle_df, max_lost=10.0, threshold=0.8, all_lanes=True)` | `flow` | Advisory per-phase saturated/unsaturated classification from end slack |
| `ring_barrier_structure(config, cycles_df=None)` | `critical` | Ring/barrier phase groups from `RB_R1`/`RB_R2` (NEMA fallback), cross-checked against observed cycle sequences |
| `movement_phase_map(config)` | `critical` | Maps `TM_*` movements to phases by stop-bar detector overlap |
| `phase_demand(counts_df, movement_map)` | `critical` | Aggregates movement counts into per-phase demand |
| `critical_movement_analysis(structure_df, demand_df, basis="per_lane")` | `critical` | Critical phase per ring and critical path per barrier group; returns `(phase_df, group_df)` |
| `split_failures(events_df, phase, detector_ids, threshold=0.79, aggregate="union", ror_seconds=5.0, include_yellow=False)` | `split_failures` | Per-cycle Purdue GOR/ROR5 split-failure table for one phase; returns `(cycle_df, detail_df)` |
| `bin_split_failures(cycle_df, bin_len=60)` | `split_failures` | Aggregates per-cycle split failures into fixed time bins |
| `optimize(curves, structure_df, saturated, demand_vph, min_splits, c_min=60.0, c_max=220.0, c_step=1.0, flat_tol_pct=1.0, boundary_rate_tol=100.0)` | `optimizer` | Scans cycle lengths to maximize saturated throughput; returns the optimum, scan, and split allocation |
| `validate_plans(flow, cycles_df, gap_ts=(), pct=1.0, split_tolerance=0.1, grid_step=0.5, min_cycles=5, max_lost=10.0, sat_threshold=0.8, min_plan_cycles=30, split_cover_tol=1.0, rank_deadband_pct=2.0, change_tol_pp=3.0)` | `optimizer_validation` | Tests the throughput model against the existing TOD plans |
| `parse_detector_roles(config)` | `detector_roles` | Parses all `Det_P<N>_*` config keys into a tidy roles DataFrame |
| `detector_sets(roles, role)` | `detector_roles` | `{phase: frozenset(detector_ids)}` for one role (`arrival`, `stop_bar`, `occupancy`, …) |
| `arrival_travel_times(config)` | `detector_roles` | `{phase: {detector: travel_s}}` from `Det_P<N>_Arrival_Travel` |
| `phase_overlaps(config)` | `detector_roles` | `{phase: overlap_number}` from `Det_P<N>_Overlap` |
| `phase_directions(config)` | `detector_roles` | `{phase: direction}` from `Det_P<N>_Direction` |
| `through_phases(config)` / `direction_detectors(movements)` | `detector_roles` / `approach_volume` | Through-movement phases, and movement→detector grouping by direction |
| `approach_delay(events_df, phase, detector_ids, travel_time_sec=0.0)` | `approach_delay` | Per-cycle approach delay and AoG/AoY/AoR arrival shares for one phase |
| `bin_approach_delay(cycle_df, bin_len=15)` | `approach_delay` | Aggregates per-cycle approach delay into fixed time bins |
| `approach_volume(counts, movements, bin_len=15)` | `approach_volume` | Approach volume bins and per-day peak-hour / PHF / K / D; returns `(bins_df, days_df)` |
| `yellow_red_actuations(events_df, phase, detector_ids, severe_sec=4.0, overlap=None, exclusions=None)` | `yellow_red_actuations` | Per-cycle yellow/red actuation and violation counts; returns `(cycle_df, act_df)` |
| `summarize_yellow_red(cycle_df, bin_len=15)` | `yellow_red_actuations` | Bins the per-cycle yellow/red table |
| `green_time_utilization(events_df, phase, detector_ids, bin_s=2.0, overlap=None, exclusions=None, timeline=None)` | `green_time_utilization` | Per-cycle and second-of-green occupancy; returns `(cycle_df, act_df)` |
| `summarize_gtu_bins(cycle_df, act_df, bin_s=2.0, bin_len=15)` / `summarize_gtu_splits(cycle_df, bin_len=15)` | `green_time_utilization` | Time-bin and per-plan green-time summaries |
| `left_turn_gaps(events_df, opposing_phase, detector_ids, left="", edges=(1.0,3.3,3.7,7.4,inf), trend_s=7.4, critical_s=4.1, exclusions=None)` | `left_turn_gap` | Per-green opposing-traffic gap counts for a permissive left; returns `(cycles_df, gaps_df)` |
| `left_turn_pairs(config)` | `left_turn_gap` | Pairs each left-turn movement with its opposing through phase from config |
| `summarize_left_turn_gaps(cycles, bin_len=15, edges=(1.0,3.3,3.7,7.4,inf))` | `left_turn_gap` | Bins the per-green gap counts |
| `ped_delay(events_df, phases=None, ped_detectors=None)` | `call_service` | Per-walk pedestrian delay (call→service); returns `(ped_df, note)` |
| `summarize_ped_delay(ped_df, bin_len=60)` | `call_service` | Bins pedestrian delay |
| `wait_time(events_df, phases=None, dropping=None)` | `call_service` | Per-window vehicle wait time (call→service), optional UDOT dropping algorithm |
| `summarize_wait_time(wait_df, bin_len=15, max_wait=360.0)` | `call_service` | Bins vehicle wait time with an optional cap |
| `plan_timeline(events_df)` | `split_monitor` | Coordination-plan timeline (Code 131/132) |
| `programmed_at(timeline, ts, phase=None)` | `split_monitor` | Programmed split in effect at each timestamp |
| `split_monitor(events_df, phases=None, timeline=None)` | `split_monitor` | Per-cycle served-split table vs. programmed splits |
| `split_monitor_stats(cycle_df, percentiles=(50, 85))` | `split_monitor` | Per-plan split percentile statistics |
| `preempt_episodes(events_df)` / `preempt_summary(episodes, tz)` / `PREEMPT_CODES` | `preempt` | Preemption episodes, their summary, and the preempt event-code set |
| `TIMING_CODES` | `timing_actuation` | The event-code set the timing/actuation view draws |
| `timing_actuation_intervals(events_df, window, data_range=None)` | `timing_actuation` | Per-phase green/yellow/red intervals, calls, ped service, and detector actuations for a window |
| `timing_actuation_rows(roles, intervals, marks=None, phase_order=None, phases=None, detectors=None)` | `timing_actuation` | Flattens intervals into plot-ready rows grouped by role |
| `ring_phase_order(config)` / `finding_plot_windows(findings, events_df, tz, half_width_s=600.0)` | `timing_actuation` | Ring phase ordering, and the per-finding plot windows a health report links to |
| `MarkerPeds` / `marker_peds_from_config(config)` | `clock_marks` | The `Clk_Behind`/`Clk_Ahead`/`Clk_Set` ped phases, parsed from config |
| `drop_marker_events(events_df, peds)` | `clock_marks` | Remove clock-mark ped actuations from an events frame |
| `send_log_pulses(records)` | `clock_marks` | Parse a head-unit `eos-time.jsonl` send log into a pulse frame |
| `pair_marker_pulses(events_df, peds)` / `decode_clock_marks(events_df, peds, send_log=None)` | `clock_marks` | Pair encoded mark pulses and decode controller clock drift and correction sets |
| `detector_activity_profile(events_df, tz, roles=None, windows=None, start_date=None, end_date=None, short_pulse_s=0.1, max_silence_s=3600.0)` | `detector_activity` | Per-detector per-window activity profile; unmarked silences longer than `max_silence_s` read as gaps |
| `HealthThresholds` | `detector_health` | Dataclass of detector-health rule thresholds |
| `detector_health_findings(profile, roles=None, events_df=None, tz=None, thresholds=HealthThresholds(), reboot_windows=None, units=None)` | `detector_health` | Deterministic detector-health findings from an activity profile |
| `onset_bursts(events_df, min_channels=6, units=None, reassert_s=1.0)` | `detector_health` | Simultaneous multi-channel onset bursts (reboot signature) |
| `apply_ignore(findings, ignore)` / `filter_min_severity(findings, min_severity="low")` / `severity_exit_code(findings, min_severity="low")` | `detector_health` | Suppress ignored `(detector, rule)` pairs, filter by severity, and map severity to a process exit code |
| `wd_ignore(config)`, `wd_profile_windows(config)`, `wd_reboot_windows(config)`, `wd_thresholds(config, base=HealthThresholds())`, `wd_units(config)` | `detector_health` | Read the `WD_*` config family into ignore lists, profile/reboot windows, thresholds, and unit groupings |
| `phase_status_at_timestamps(events_df, phase, query_ts)` | `video` | Per-frame `'G'`/`'Y'`/`'R'`/`'na'` status for one signal phase |
| `overlap_status_at_timestamps(events_df, overlap_num, query_ts)` | `video` | Per-frame `'G'`/`'Y'`/`'R'`/`'na'` status for one overlap (Codes 61/63/64/65/66) |
| `detector_status_at_timestamps(events_df, det_id, query_ts)` | `video` | Per-frame On/Off boolean status for one detector; reuses `analysis.detectors._reconstruct_intervals` |
| `first_phase_transition_after(events_df, phase, after_ts, transition=None)` | `video` | Earliest green→yellow/yellow→red color change for a phase at or after a timestamp |

## `atspm.plotting` — Functional Core

Pure functions: DataFrames/metadata in, `plotly.graph_objects.Figure` out. No file I/O — callers are responsible for `.write_html()`.

| Function | Module | Description |
|---|---|---|
| `plot_termination(df_events, metadata, line=True, n_con=10)` | `termination` | Phase termination scatter (gap out / max out / force off / preempt / ped service), with an optional rolling max-out-proportion line |
| `plot_coordination(df_cycles, df_signal, metadata, df_det=None, det_config=None, individual_detectors=False)` | `coordination` | Stacked green/yellow/red-clearance bar diagram per ring, with optional detector-activation overlay |
| `plot_detector_comparison(events_df, anomalies_df, detector_pairs, metadata=None, window=None)` | `detectors` | Side-by-side detector actuation timelines with discrepancy overlays, hard-reset lines, and a per-pair summary; `window=(start, end)` clips to an epoch range |
| `plot_flow_profiles(rate_df, inst_df, metadata, phase, rolling=5)` | `flow` | Mean effective cumulative rate against elapsed split time, with the throughput-optimal peak marked and instantaneous-rate traces |
| `plot_allocation(splits_df, optimum, metadata)` | `optimizer` | Recommended split allocation at the optimal cycle length |
| `plot_marginal_rates(curves, splits_df, metadata)` | `optimizer` | Per-phase marginal discharge rates behind the allocation |
| `plot_throughput_curve(scan_df, optimum, metadata)` | `optimizer` | Total throughput against scanned cycle length, with the optimum marked |
| `plot_clock_drift(drift_df, sets_df, metadata=None, timezone=None)` | `clock_marks` | Controller clock drift over time with correction sets annotated |
| `plot_split_failures(cycle_df, metadata=None, threshold=0.79)` | `split_failures` | GOR-vs-ROR5 scatter with the failure threshold |
| `plot_approach_delay(binned_df, metadata=None)` | `approach_delay` | Binned approach delay and arrival-share bars |
| `plot_approach_volume(bins_df, days_df, metadata=None)` | `approach_volume` | Approach volume over time with peak-hour markers |
| `plot_yellow_red(cycle_df, act_df, metadata=None, severe_sec=4.0)` | `yellow_red_actuations` | Yellow/red actuation and violation counts |
| `plot_green_time(bins_df, splits_df, metadata=None, max_green_s=120.0)` | `green_time_utilization` | Green-time utilization heatmap and per-plan splits |
| `plot_left_turn_gap(bins_df, metadata=None, edges=(1.0,3.3,3.7,7.4,inf), trend_s=7.4)` | `left_turn_gap` | Binned opposing-traffic gap counts with turnable-gap trend |
| `plot_ped_delay(delay_df, binned_df, metadata=None)` | `call_service` | Pedestrian delay per walk and binned |
| `plot_wait_time(wait_df, binned_df, metadata=None, max_wait=360.0)` | `call_service` | Vehicle wait time per window and binned |
| `plot_split_monitor(cycle_df, timeline_df, metadata=None)` | `split_monitor` | Served splits against programmed splits over time |
| `plot_detector_health(profile, findings, metadata=None, window="day")` | `detector_health` | Detector activity heatmap with findings overlaid |
| `plot_timing_actuation(rows, intervals, marks, window, tz, metadata=None, findings=None)` | `timing_actuation` | Per-phase timing intervals, calls, ped service, and detector actuations grouped by role |

## `atspm.utils` — shared helpers

Not re-exported from `atspm/utils/__init__.py`; import from the module directly. Documented here because the timezone contract binds every caller, including external ones.

| Function / Constant | Module | Description |
|---|---|---|
| `DEFAULT_TIMEZONE` | `timezone` | `"US/Mountain"`. The single fallback for "what zone is this intersection?" — the `metadata.timezone` column default, the `atspm setup` default, and what every resolver returns when a database records no zone |
| `resolve_pytz(tz_string)` | `timezone` | IANA name → `pytz` timezone. Falls back to **UTC** with a logged warning when the name is missing or unparseable — a different question from `DEFAULT_TIMEZONE`, deliberately answered differently so a bad name degrades to an unambiguous zone |
| `localize_naive(dt, tz_string)` | `timezone` | Attaches *tz_string* to a naive datetime; aware values pass through unchanged |
| `to_epoch(dt, tz_string)` | `timezone` | Datetime → UTC epoch float. The single conversion point for query bounds; see the Timezone contract below |
| `compute_bin_quality(events_df, spans_df, start, end, bin_len, timezone)` | `quality` | `coverage`/`data_quality` per bin from `ingestion_log` spans, with gap-marker downgrades. Pure — callers fetch the spans and events |

## Timezone contract

All timestamps are stored as UTC epoch floats. Around that:

- **Ingest** converts a `.datZ` filename's local wall clock to UTC through the intersection's zone.
- **Query bounds** — a **naive** `start`/`end` means *intersection local wall clock*; an **aware** one keeps its own offset. Neither ever consults the host machine's clock. Naive bounds resolve their zone as: explicit `timezone=` argument → the database's `metadata.timezone` → `DEFAULT_TIMEZONE`.
- **Returned timestamps** are tz-aware in the intersection's zone when a `timezone` is supplied, and raw UTC epoch floats otherwise.

Passing `datetime.timestamp()` output, or a naive `pandas.Timestamp`, bypasses this — a naive `datetime.timestamp()` reads as host-local and a naive `Timestamp.timestamp()` reads as UTC. Hand engines `datetime` objects or `YYYY-MM-DD` strings and let them do the conversion.

## `atspm.reports` — Imperative Shell

| Function / Class | Description |
|---|---|
| `PlotGenerator(db_path, output_dir)` | `generate_for_date(date_str)` and `generate_date_range(start_date, end_date)` — fetches data via `atspm.data.reader`, builds figures via `atspm.plotting`, writes HTML to `{output_dir}/{YYYY-MM-DD}/` |
| `generate_reports(db_path, output_dir, date_str)` | Convenience wrapper around `PlotGenerator` |

## `atspm.video` — Imperative Shell (one documented exception)

A peer of `data`/`analysis`/`plotting`/`reports`, not a submodule of `plotting` — see [architecture.md](architecture.md).

| Function / Class | Module | Description |
|---|---|---|
| `calibrate_shapes(video_path, shape_config=None, save_path=None)` | `calibrate` | Interactive Tkinter+OpenCV session to draw/edit loop/stopbar shapes; owns saving when `save_path` is given |
| `draw_shape_overlay(frame, shape, status)` | `overlay` | Dispatches to `draw_loop_overlay`/`draw_stopbar_overlay` by `shape["type"]`; mutates `frame` in place |
| `draw_loop_overlay(frame, shape, is_on)` | `overlay` | In-place loop-detector outline recolor |
| `draw_stopbar_overlay(frame, shape, status)` | `overlay` | In-place stopbar outline recolor by `'G'`/`'Y'`/`'R'`/`'na'` |
| `draw_lamp_overlay(frame, shape, status)` | `overlay` | In-place signal-lamp disc recolor, used by the video-sync measurement overlay |
| `measure_lamps(video_path, shape_config)` | `sync` | Measures mean BGR inside each configured lamp ROI across all frames; returns a `LampMeasurement` |
| `LampMeasurement` | `sync` | Dataclass: `frame_times_s`, `lamps`, `fps`, `timing_source` (`'pts'` or `'fps'`) |
| `sync_video(db_path, shape_config, video_path, start_guess, search_s=30.0)` | `sync` | Aligns measured lamps against controller phase/overlap states to recover the true first-frame timestamp and camera clock slip; backs `atspm video-sync` |
| `render_overlay(db_path, shape_config, video_path, output_path, start_dt, lookback_minutes=10.0, lookahead_minutes=10.0, chunk_frames=150)` | `processor` | Renders a full video with live phase/overlap/detector overlays; returns `VideoOverlayResult` |
| `extract_labeled_clip(video_path, output_path, expected_offset_sec, window_sec=3.0)` | `processor` | Crops a short clip around an expected transition time with a signed countdown label burned in; returns `VideoOverlayResult` |
| `VideoOverlayResult` | `processor` | Dataclass: `output_path`, `frame_count`, `fps`, `timing_source` (`'pts'` or `'fps'`) |
