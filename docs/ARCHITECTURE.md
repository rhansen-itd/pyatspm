# pyATSPM Architecture

pyATSPM is built on a modern, queryable, per-intersection database backend, designed following the **Functional Core, Imperative Shell** pattern.

## Database Strategy

* **One SQLite file per intersection:** e.g., `2068_data.db`.
* **WAL mode:** Always enabled for concurrent read/write operations.
* **No ORMs:** Raw `sqlite3` for ingestion (speed), `pandas.read_sql` for analysis (convenience).

## Implemented Schema

### `events`
* `timestamp`: REAL NOT NULL (UTC epoch float)
* `event_code`: INTEGER NOT NULL (e.g., 1=Green, 82=Det On, -1=Gap marker)
* `parameter`: INTEGER NOT NULL (Phase or Detector ID)
* *Indices:* `idx_events_timestamp`, `idx_events_code_param`, `idx_events_ts_code`

### `cycles`
* `cycle_start`: REAL PRIMARY KEY
* `coord_plan`: REAL NOT NULL DEFAULT 0
* `detection_method`: TEXT NOT NULL DEFAULT ''
* `r1_phases`: TEXT NOT NULL DEFAULT 'None'
* `r2_phases`: TEXT NOT NULL DEFAULT 'None'
* *Indices:* `idx_cycles_start`

### `config`
* `id`: INTEGER PRIMARY KEY AUTOINCREMENT
* `start_date`: TEXT NOT NULL
* `end_date`: TEXT
* *(Dynamic columns added based on configuration CSV)*

### `metadata`
* `lock_id`: INTEGER PRIMARY KEY CHECK (lock_id = 1)
* `intersection_id`: TEXT
* `intersection_name`: TEXT
* `controller_ip`: TEXT
* `detection_type`: TEXT
* `detection_ip`: TEXT
* `major_road_route`: TEXT
* `major_road_name`: TEXT
* `minor_road_route`: TEXT
* `minor_road_name`: TEXT
* `latitude`: REAL
* `longitude`: REAL
* `timezone`: TEXT NOT NULL DEFAULT 'US/Mountain'
* `agency_id`: TEXT

### `ingestion_log`
* `span_start`: REAL PRIMARY KEY
* `span_end`: REAL NOT NULL
* `processed_at`: TEXT NOT NULL
* `row_count`: INTEGER NOT NULL
* *Indices:* `idx_ingestion_span_end`

---

## Code Structure (Functional Core / Imperative Shell)

The project separates stateful, effectful operations from pure analytical logic.

### Imperative Shell: `src/atspm/data/`
Manages all state, DB connections, and file I/O.
* **`manager.py`**: Initializes DB schemas and imports configurations.
* **`ingestion.py`**: Reads `.datZ` files and ingests them into the DB.
* **`processing.py`**: Orchestrates cycle detection using the DB.
* **`reader.py`**: Adapters to retrieve data as DataFrames.

### Functional Core: `src/atspm/analysis/`
Pure functions that perform analytical operations (DataFrames in, DataFrames out).
* **`decoders.py`**: Decodes raw binary `.datZ` into DataFrame events.
* **`cycles.py`**: The raw logic determining cycle state boundaries.

### Outputs: `src/atspm/reports/` and `src/atspm/plotting/`
Take data inputs (often DataFrames via the shell or directly) and generate interactive visualizations.

---

## Traffic Engineering Philosophy (Gap Markers & Discrepancies)

*(Placeholder for future traffic engineering philosophical discussion regarding gap markers and discrepancy metrics.)*
