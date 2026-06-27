# Architecture

## Functional Core, Imperative Shell

pyATSPM splits every feature into two layers:

- **Functional Core** (`src/atspm/analysis/`, `src/atspm/plotting/`) — pure functions only. Input/output is DataFrames, dicts, or Plotly `go.Figure` objects. No SQL connections, no file I/O, no `.write_html()`. Fully testable in isolation.
- **Imperative Shell** (`src/atspm/data/`, `src/atspm/reports/`) — owns all state: DB connections, file paths, transactions. Fetches data via SQL, calls the Core to do the math/rendering, then persists or writes the result.
- **`src/atspm/video/`** — a peer package to all four above, not a submodule of `plotting/` (OpenCV video frames are a different output shape and library than Plotly). Mostly Imperative Shell (`calibrate.py`, `processor.py` own all DB/OpenCV I/O), with one documented Functional Core exception: `overlay.py`'s in-place `cv2` frame drawing — see the Vectorization bullet below.

```
.datZ files
    │
    ▼
IngestionEngine            ← Imperative Shell (src/atspm/data/ingestion.py)
    │  calls
    ▼
parse_datz_bytes()         ← Functional Core (src/atspm/analysis/decoders.py)
    │  returns DataFrame[timestamp, event_code, parameter]
    ▼
events table (SQLite)
    │
    ▼
CycleProcessor              ← Imperative Shell (src/atspm/data/processing.py)
    │  calls
    ▼
calculate_cycles()           ← Functional Core (src/atspm/analysis/cycles.py)
    │  returns DataFrame[cycle_start, coord_plan, detection_method, r1_phases, r2_phases]
    ▼
cycles table (SQLite)
    │
    ▼
get_events_with_cycles_df() ← Imperative Shell (src/atspm/data/reader.py)
    │  SQL JOIN events ↔ cycles
    ▼
Flat events+cycles DataFrame ── fed into analysis/* and plotting/* functions
```

`*Engine` classes (`AogEngine`, `CountEngine`, `PhaseEngine`, `DetectorEngine`) sit at the Shell/Core boundary: each wraps a `db_path` (+ optional `timezone`), reads events via `reader.py`, and delegates the actual computation to a Core function (`arrival_on_green`, `vehicle_counts`/`ped_counts`, `phase_splits`, `analyze_discrepancies`). They share the same shape — constructor takes `db_path`, public methods return DataFrames or write CSV via an optional `output_dir` argument, and binned results carry `coverage`/`data_quality` columns sourced from `ingestion_log`.

`reports/generators.py` (`PlotGenerator`) is the one place allowed to call `.write_html()` — it resolves a local date to UTC bounds, pulls DataFrames via `reader.py`, calls into `plotting/*`, and writes the resulting figures to `{output_dir}/{YYYY-MM-DD}/*.html`.

## Design Principles

- **One SQLite file per intersection** — e.g. `2068_data.db`. See [database_schema.md](database_schema.md).
- **WAL mode** — `PRAGMA journal_mode=WAL` is always enabled for concurrent read/write.
- **No ORMs** — raw `sqlite3` for ingestion (speed), `pandas.read_sql_query` for analysis (convenience).
- **UTC epoch floats** — all timestamps are stored as `REAL` (UTC epoch seconds); conversion to local time happens only at the CLI/display boundary.
- **Gap markers** — `event_code = -1` marks a data discontinuity (controller reset, missing file, etc.). Any logic computing a duration or pairing sequential events must stop at a gap marker rather than bridge across it. See [database_schema.md](database_schema.md#gap-markers).
- **Vectorization** — no row-iteration in hot paths; pandas vectorization or SQL aggregation is the default. A few `.iterrows()` calls remain in `analysis/detectors.py` and `plotting/*` where the input is a small, pre-filtered DataFrame and the logic is an inherently sequential state machine (e.g. tracking open/close detector intervals) — these are deliberate exceptions, not oversights. `video/overlay.py`'s shape-drawing functions mutate a `cv2` frame array in place (no return value) for the same reason `plotting/detectors.py` mutates a `go.Figure` in place — redrawing per-frame/per-trace would be wasteful copy overhead, not a meaningful purity gain.

## Terminology

- "pulse" / "isolated_pulse" — unconfirmed/brief detector actuation. Not "orphan".
- "Events-with-cycles exporter" — the adapter that joins `events` to `cycles` and returns a flat DataFrame (`reader.get_events_with_cycles_df`).
- "legacy" is avoided in new code; where it still appears (e.g. `analysis/cycles.py` describing the pre-refactor ring-membership default), it refers to historical behavior being preserved on purpose, not to dead code that should be removed.

## CLI as the user-facing layer

`src/atspm/cli.py` is an `argparse` CLI (entry point `atspm`) sitting on top of the Shell. Every subcommand other than `setup` accepts a mutually exclusive target group — `--target` (exact folder name), `--targetid` (numeric intersection ID prefix), or `--all` (batch over every folder in `intersections/`, skipping failures). The three `video-*` subcommands are single-target only (`--target`/`--targetid`, no `--all`) — one video file corresponds to one camera, so there's no meaningful batch form. See [cli_reference.md](cli_reference.md) for the full subcommand list.
