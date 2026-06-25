# API Reference

Public exports per package, as declared in each `__init__.py`. Most users should reach this through the CLI (see [cli_reference.md](cli_reference.md)); this is for scripting against the package directly.

## `atspm.data` — Imperative Shell

| Function / Class | Description |
|---|---|
| `DatabaseManager(db_path)` | Context manager for direct DB access (raw `sqlite3`); `get_metadata()`, `set_metadata()`, `import_config()`, span/anchor queries |
| `init_db(db_path)` | Create a new intersection DB with the full schema (WAL mode, all tables/indexes) |
| `import_config(csv_path, db_path)` | Import `int_cfg.csv` into the `config` table |
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
| `get_date_range(db_path)` | Min/max event timestamps in the DB |
| `get_available_dates(db_path)` | All local dates with computed cycles |
| `check_data_quality(db_path, start, end)` | Event/gap/cycle counts for a window |
| `convert_to_datetime(...)` | Timestamp/timezone conversion helper |
| `CountEngine` | Counts orchestration; `vehicle_counts()`, `ped_counts()`, `combined_counts()` |
| `get_vehicle_counts(...)`, `get_ped_counts(...)`, `get_combined_counts(...)` | Module-level convenience wrappers around `CountEngine` |
| `PhaseEngine` | Phase-splits orchestration; `phase_splits(...)` |
| `get_phase_splits(...)` | Module-level convenience wrapper around `PhaseEngine` |
| `AogEngine` | Arrival-on-Green orchestration; `arrival_on_green(...)` |
| `get_arrival_on_green(...)` | Module-level convenience wrapper around `AogEngine` |
| `DetectorEngine` | Detector-discrepancy orchestration; `get_discrepancies()`, `get_plot_data()` |
| `get_detector_discrepancies(...)` | Module-level convenience wrapper around `DetectorEngine` |

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
| `validate_cycles(cycles_df, min_cycle_length=10.0, max_cycle_length=300.0)` | `cycles` | Sanity checks (duplicate/short/long cycles) |
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

## `atspm.plotting` — Functional Core

Pure functions: DataFrames/metadata in, `plotly.graph_objects.Figure` out. No file I/O — callers are responsible for `.write_html()`.

| Function | Module | Description |
|---|---|---|
| `plot_termination(df_events, metadata, line=True, n_con=10)` | `termination` | Phase termination scatter (gap out / max out / force off / preempt / ped service), with an optional rolling max-out-proportion line |
| `plot_coordination(df_cycles, df_signal, metadata, df_det=None, det_config=None, individual_detectors=False)` | `coordination` | Stacked green/yellow/red-clearance bar diagram per ring, with optional detector-activation overlay |
| `plot_detector_comparison(events_df, anomalies_df, detector_pairs, metadata=None)` | `detectors` | Side-by-side detector actuation timelines with discrepancy overlays |

## `atspm.reports` — Imperative Shell

| Function / Class | Description |
|---|---|
| `PlotGenerator(db_path, output_dir)` | `generate_for_date(date_str)` and `generate_date_range(start_date, end_date)` — fetches data via `atspm.data.reader`, builds figures via `atspm.plotting`, writes HTML to `{output_dir}/{YYYY-MM-DD}/` |
| `generate_reports(db_path, output_dir, date_str)` | Convenience wrapper around `PlotGenerator` |
