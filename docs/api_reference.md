# API Reference

Public exports per package, as declared in each `__init__.py`. Most users should reach this through the CLI (see [cli_reference.md](cli_reference.md)); this is for scripting against the package directly.

## `atspm.data` — Imperative Shell

| Function / Class | Description |
|---|---|
| `DatabaseManager(db_path)` | Context manager for direct DB access (raw `sqlite3`); `get_metadata()`, `get_timezone()`, `set_metadata()`, `import_config()`, span/anchor queries |
| `init_db(db_path)` | Create a new intersection DB with the full schema (WAL mode, all tables/indexes) |
| `import_config(csv_path, db_path)` | Import `int_cfg.csv` into the `config` table |
| `db_timezone(db_path, timezone=None)` | The zone to use for an intersection: *timezone* if given, else the DB's `metadata.timezone`, else `DEFAULT_TIMEZONE`. Every engine's timezone resolution goes through here |
| `RetrievalEngine(target_dir, meta, devices)` | Pulls new `.datZ` files for every device in a parsed `devices.json`, secondary devices before controller; `run()` returns per-device result dicts |
| `run_retrieval(target_dir, meta, devices_path)` | Module-level convenience wrapper around `RetrievalEngine` — loads/saves `devices.json` for the caller |
| `IngestionEngine` | Orchestrates `.datZ` file scanning, parsing, gap detection, and triggers cycle processing |
| `run_ingestion(db_path, data_dir, timezone, incremental, batch_size)` | Ingest `.datZ` files into `events` |
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
| `check_data_quality(db_path, start, end, timezone=None)` | Event/gap/cycle counts for a window |
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
| `ShapeConfig` | Per-camera loop/stopbar shape config; `load(path)`/`save(path)` round-trip a `<camera>_shapes.csv`, `validate_resolution(w, h)`, `relevant_phases()`/`relevant_overlaps()`/`relevant_detectors()` |
| `resolve_stopbar_target(phase_field)` | Resolves a stopbar shape's `phase` field to a `(kind, number)` lookup target — `kind` is `"phase"` or `"overlap"` |
| `OVERLAP_LETTER_MAP` | `dict` mapping overlap letters `"OLA"`-`"OLP"` to numbers `1`-`16` |

## `atspm.analysis` — Functional Core

Pure functions: DataFrames/dicts in, DataFrames/dicts/figures out. No I/O.

| Function / Class | Module | Description |
|---|---|---|
| `parse_datz_bytes(raw_bytes, file_timestamp)` | `decoders` | Decode one `.datZ` payload → `DataFrame[timestamp, event_code, parameter]` |
| `parse_datz_batch(file_data)` | `decoders` | Decode and merge multiple files |
| `validate_datz_file(raw_bytes)` | `decoders` | Quick validity check |
| `estimate_event_count(raw_bytes)` | `decoders` | Pre-parse row-count estimate |
| `insert_gap_marker(df, gap_timestamp)` | `decoders` | Insert an `event_code = -1` discontinuity row |
| `detect_corruption(raw_bytes)` | `decoders` | Heuristic corruption check |
| `DatZDecodingError` | `decoders` | Raised on unparseable `.datZ` input |
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
| `analyze_discrepancies(events_df, detector_pairs, lag_threshold_sec=2.0)` | `detectors` | Classifies co-located detector disagreements as `extended_disagreement` or `isolated_pulse` |
| `phase_splits(events_df, bin_len="cycle", report_mode="seconds", phases=None, include_no_clearance=False)` | `phases` | Per-cycle/binned green-yellow-red-clearance timing table |
| `arrival_on_green(events_df, phase, detector_ids, arrival_offset_sec=0.0)` | `aog` | Per-cycle Arrival on Green for one phase |
| `bin_arrival_on_green(cycle_df, bin_len=60)` | `aog` | Aggregates per-cycle AOG into fixed time bins |
| `flow_rate(events_df, phase, detector_ids, max_lost=10.0, plans=None)` | `flow` | Per-cycle and per-vehicle stop-bar departure tables for one phase; returns `(cycle_df, vehicle_df)` |
| `rate_profiles(cycle_df, vehicle_df, pct=1.0, split_tolerance=0.1, normalize="end_shift", fixed_lost=None, grid_step=0.5, min_cycles=5)` | `flow` | Collapses qualifying cycles onto a common elapsed-time grid; returns `(rate_df, inst_df, summary_df)` |
| `ring_barrier_structure(config, cycles_df=None)` | `critical` | Ring/barrier phase groups from `RB_R1`/`RB_R2` (NEMA fallback), cross-checked against observed cycle sequences |
| `movement_phase_map(config)` | `critical` | Maps `TM_*` movements to phases by stop-bar detector overlap |
| `phase_demand(counts_df, movement_map)` | `critical` | Aggregates movement counts into per-phase demand |
| `critical_movement_analysis(structure_df, demand_df, basis="per_lane")` | `critical` | Critical phase per ring and critical path per barrier group; returns `(phase_df, group_df)` |
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
| `plot_detector_comparison(events_df, anomalies_df, detector_pairs, metadata=None)` | `detectors` | Side-by-side detector actuation timelines with discrepancy overlays |
| `plot_flow_profiles(rate_df, inst_df, metadata, phase, rolling=5)` | `flow` | Mean effective cumulative rate against elapsed split time, with the throughput-optimal peak marked and instantaneous-rate traces |

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
| `render_overlay(db_path, shape_config, video_path, output_path, start_dt, lookback_minutes=10.0, lookahead_minutes=10.0, chunk_frames=150)` | `processor` | Renders a full video with live phase/overlap/detector overlays; returns `VideoOverlayResult` |
| `extract_labeled_clip(video_path, output_path, expected_offset_sec, window_sec=3.0)` | `processor` | Crops a short clip around an expected transition time with a signed countdown label burned in; returns `VideoOverlayResult` |
| `VideoOverlayResult` | `processor` | Dataclass: `output_path`, `frame_count`, `fps` |
