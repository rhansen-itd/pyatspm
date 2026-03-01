# pyATSPM

A high-performance Python package for Automated Traffic Signal Performance Measures (ATSPM), built on a normalized SQLite architecture with a Functional Core / Imperative Shell design pattern.

Replaces a legacy flat-file system (CSV/Pickle + complex pandas logic) with a modern, queryable, per-intersection database backend — while maintaining backward compatibility with existing analysis and plotting code.

## Documentation

For deep technical dives, refer to the documents below:

- **[Installation Guide](docs/INSTALL.md)**: Instructions on environment setup and package installation.
- **[CLI Reference](docs/CLI_REFERENCE.md)**: Complete list of subcommands, arguments, and flags for the `atspm` CLI.
- **[Architecture](docs/ARCHITECTURE.md)**: Detailed breakdown of the Database Strategy, Implemented Schema, and Functional Core/Imperative Shell pattern.

## Quickstart (CLI)

The `atspm` CLI handles intersection setup, ingestion, and reporting.

1. **Setup a new intersection environment**:
```bash
atspm setup --target 2068_US-95_and_SH-8 --timezone "US/Mountain"
```
*(This creates `intersections/2068_US-95_and_SH-8/` with `metadata.json` and a config placeholder.)*

2. **Process and ingest raw `.datZ` files**:
```bash
atspm process --target 2068_US-95_and_SH-8
```
*(This reads `.datZ` files, ingests data into SQLite, and orchestrates cycle detection.)*

3. **Generate Reports**:
```bash
atspm report --target 2068_US-95_and_SH-8 --dates 2026-02-19
```

---

## Project Structure

```
pyatspm/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── docs/
│   ├── architecture.md          # Functional Core / Imperative Shell explained
│   ├── configuration_guide.md   # How to edit int_cfg.csv
│   └── database_schema.md       # events, cycles, config, metadata tables
│
├── src/
│   └── atspm/
│       ├── data/                # Imperative Shell (I/O, DB, state)
│       │   ├── manager.py       # DB init, config import, metadata
│       │   ├── ingestion.py     # DatZ ingestion with timezone handling
│       │   ├── processing.py    # Cycle detection orchestration
│       │   └── reader.py        # Legacy adapter (flat DataFrame output)
│       └── analysis/            # Functional Core (pure transformations)
│           ├── decoders.py      # Binary DatZ parsing
│           └── cycles.py        # Cycle detection logic
│
├── scripts/
│   ├── setup_intersection.py    # Create folder structure + metadata.json
│   └── run_ingestion.py         # Ingest data for a target intersection
│
└── intersections/               # Data (gitignored)
    └── 2068_US-95_and_SH-8/
        ├── metadata.json
        ├── int_cfg.csv
        ├── 2068_data.db
        ├── raw_data/            # .datZ files go here
        └── outputs/
```

---

## Architecture

### Functional Core / Imperative Shell

**Functional Core** (`src/atspm/analysis/`) — pure functions, no I/O, no side effects. Takes DataFrames in, returns DataFrames out. Fully testable in isolation.

**Imperative Shell** (`src/atspm/data/`) — manages all state: DB connections, file I/O, transactions. Calls the Functional Core for logic, persists results.

```
DatZ files
    │
    ▼
IngestionEngine          ← Imperative Shell
    │  calls
    ▼
parse_datz_bytes()       ← Functional Core
    │  returns DataFrame
    ▼
DatabaseManager          ← Imperative Shell
    │  writes to
    ▼
events table (SQLite)
    │
    ▼
CycleProcessor           ← Imperative Shell
    │  calls
    ▼
calculate_cycles()       ← Functional Core
    │  returns DataFrame
    ▼
cycles table (SQLite)
    │
    ▼
get_legacy_dataframe()   ← Reader / Legacy Adapter
    │  SQL JOIN + merge_asof
    ▼
Flat DataFrame           ← legacy plotting/analysis code works unchanged
```

### Design Principles

- **One SQLite file per intersection** — e.g., `2068_data.db`
- **WAL mode** — always enabled for concurrent read/write
- **No ORMs** — raw `sqlite3` for ingestion (speed), `pandas.read_sql` for analysis (convenience)
- **UTC epoch floats** — all timestamps stored as `REAL` (8-byte float), converted from local time at ingestion boundary
- **Vectorization** — no row-iteration; pandas vectorization or SQL aggregations throughout

---

## Database Schema

### `events` — Raw Data

```sql
CREATE TABLE events (
    timestamp   REAL    NOT NULL,   -- UTC epoch float
    event_code  INTEGER NOT NULL,   -- e.g., 1=Green, 82=Det On, -1=Gap marker
    parameter   INTEGER NOT NULL,   -- Phase or Detector ID
    UNIQUE(timestamp, event_code, parameter) ON CONFLICT IGNORE
)
```

Indices: `idx_events_timestamp`, `idx_events_code_param`, `idx_events_ts_code` (covering)

### `cycles` — Derived Data (computed once, reusable)

```sql
CREATE TABLE cycles (
    cycle_start      REAL    PRIMARY KEY,
    coord_plan       INTEGER,
    detection_method TEXT    -- 'barrier_pulse' | 'ring_barrier_config' | 'single_cycle_fallback'
)
```

### `config` — Hybrid Schema (temporal, wide columns)

```sql
CREATE TABLE config (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    start_date TEXT    NOT NULL,
    end_date   TEXT,
    -- Dynamic columns added at import time, e.g.:
    TM_EBL         TEXT,
    TM_SBThru      TEXT,
    TM_Exclusions  TEXT,   -- JSON: [{"detector":33,"phase":2,"status":"Red"}, ...]
    RB_R1          TEXT,
    RB_R2          TEXT,
    Det_P2_Arrival TEXT,
    -- ...one column per CSV row, prefixed by category (TM_, RB_, Det_, WD_)
    UNIQUE(start_date) ON CONFLICT REPLACE
)
```

### `metadata` — Static Intersection Attributes

```sql
CREATE TABLE metadata (
    lock_id           INTEGER PRIMARY KEY CHECK (lock_id = 1),  -- enforces single row
    intersection_id   TEXT,
    intersection_name TEXT,
    controller_ip     TEXT,
    detection_type    TEXT,    -- e.g., 'Radar', 'Loops', 'Video'
    detection_ip      TEXT,
    major_road_route  TEXT,    -- e.g., 'US-95'
    major_road_name   TEXT,    -- e.g., 'Main St'
    minor_road_route  TEXT,
    minor_road_name   TEXT,
    latitude          REAL,
    longitude         REAL,
    timezone          TEXT NOT NULL DEFAULT 'US/Mountain',
    agency_id         TEXT
)
```

### `ingestion_log` — State Tracking

```sql
CREATE TABLE ingestion_log (
    filename        TEXT PRIMARY KEY,
    file_timestamp  REAL NOT NULL,   -- UTC epoch of file (parsed from filename)
    processed_at    TEXT NOT NULL,   -- ISO timestamp of ingestion run
    row_count       INTEGER NOT NULL
)
```

---

## Common Event Codes

| Code | Meaning |
|------|---------|
| 1 | Green Start |
| 4 | Gap Out |
| 5 | Max Out |
| 6 | Force Off |
| 8 | Yellow Start |
| 9 | Red Clearance Start |
| 11 | Red Start |
| 31 | Barrier Event (cycle detection) |
| 81 | Detector Off |
| 82 | Detector On |
| 131/132 | Coordination Plan |
| -1 | Data gap marker (inserted by ingestion) |

---

## Workflow

### 1. Set Up a New Intersection

```bash
python scripts/setup_intersection.py --id 2068 --name "US-95 & SH-8" --tz "US/Mountain"
```

Creates:
```
intersections/2068_US-95_and_SH-8/
├── metadata.json    ← fill in IPs, coordinates, detection type
├── int_cfg.csv      ← placeholder; replace with real config
├── raw_data/        ← drop .datZ files here
└── outputs/
```

Then edit `metadata.json` to fill in optional fields before running ingestion.

### 2. Run Ingestion

```bash
python scripts/run_ingestion.py --target "2068_US-95_and_SH-8"

# Full reprocess (re-reads all files, recomputes all cycles):
python scripts/run_ingestion.py --target "2068_US-95_and_SH-8" --full
```

This script:
1. Initializes the DB (`init_db`)
2. Syncs `metadata.json` → `metadata` table (`set_metadata`)
3. Imports `int_cfg.csv` → `config` table (`import_config`)
4. Ingests `.datZ` files → `events` table (`run_ingestion`)
5. Detects cycles → `cycles` table (`run_cycle_processing`)

### 3. Read Data (Legacy Format)

```python
from pathlib import Path
from datetime import datetime
from src.atspm.data import get_legacy_dataframe, get_config_df

db_path = Path("intersections/2068_US-95_and_SH-8/2068_data.db")

df = get_legacy_dataframe(
    db_path,
    start=datetime(2025, 1, 1),
    end=datetime(2025, 1, 2)
)
# Returns: DataFrame['TS_start', 'Code', 'ID', 'Cycle_start', 'Coord_plan']
# All timestamps are UTC epoch floats — identical to legacy pickle format

config = get_config_df(db_path, datetime(2025, 1, 1))
# Returns: pd.Series — access as config['TM_EBL'], config['RB_R1'], etc.
```

### 4. Direct DB Access

```python
from src.atspm.data import DatabaseManager

with DatabaseManager(db_path) as manager:
    # Read metadata
    meta = manager.get_metadata()

    # Set/update metadata
    manager.set_metadata(
        intersection_id='2068',
        major_road_route='US-95',
        latitude=46.732,
        longitude=-117.001
    )

    # Raw event query
    df = manager.query_events(
        start_time=datetime(2025, 1, 1).timestamp(),
        end_time=datetime(2025, 1, 2).timestamp(),
        event_codes=[1, 8, 82]
    )
```

---

## `metadata.json` Reference

Created by `setup_intersection.py`. Edit before running ingestion.

```json
{
    "intersection_id": "2068",
    "intersection_name": "US-95 & SH-8",
    "timezone": "US/Mountain",
    "folder_name": "2068_US-95_and_SH-8",
    "db_filename": "2068_data.db",

    "controller_ip": "10.71.10.50",
    "detection_type": "Radar",
    "detection_ip": "10.71.10.51",
    "agency_id": "ITD-D2",

    "major_road_route": "US-95",
    "major_road_name": "Main St",
    "minor_road_route": "SH-8",
    "minor_road_name": "Troy Hwy",

    "latitude": 46.732,
    "longitude": -117.001
}
```

---

## `int_cfg.csv` Structure

Multi-index CSV with `(Category, Parameter)` as the first two columns and date columns as headers. Each date column represents a configuration period; the system automatically calculates `end_date` as the start of the next period.

| Category | Parameter | 2024-01-01 | 2024-06-15 |
|----------|-----------|------------|------------|
| TM: | EBL | 5,6,7 | 5,6,7 |
| TM: | SBThru | 10,11,12 | 10,11 |
| RB: | R1 | 1,2\|3,4 | 1,2\|3,4 |
| RB: | R2 | 5,6\|7,8 | 5,6\|7,8 |
| Det: | P2 Arrival | 33,34 | 33,34 |
| Exc: | Detector | 33,34 | |
| Exc: | Phase | 2,2 | |
| Exc: | Status | Red,Yellow | |

**Category prefixes in DB:** `TM_`, `RB_`, `Det_` (legacy `Plt:` normalized to `Det_`), `WD_`. Exclusion rows (`Exc:`) are bundled into `TM_Exclusions` as a JSON array.

---

## API Reference

### `src.atspm.data`

| Function / Class | Description |
|-----------------|-------------|
| `init_db(db_path)` | Initialize DB with full schema |
| `import_config(csv_path, db_path)` | Import `int_cfg.csv` |
| `run_ingestion(db_path, data_dir, timezone, incremental, batch_size)` | Ingest `.datZ` files |
| `run_cycle_processing(db_path, reprocess)` | Detect and store cycles |
| `get_legacy_dataframe(db_path, start, end, event_codes)` | Main reader — legacy format |
| `get_legacy_dataframe_by_date(db_path, date_str)` | Convenience — full day |
| `get_config_df(db_path, date)` | Config as `pd.Series` |
| `get_config_dict(db_path, date)` | Config as `dict` |
| `get_date_range(db_path)` | Min/max timestamps in DB |
| `get_available_dates(db_path)` | All dates with cycles |
| `check_data_quality(db_path, start, end)` | Event/gap/cycle counts |
| `DatabaseManager(db_path)` | Context manager for direct DB access |

### `src.atspm.analysis`

| Function | Description |
|----------|-------------|
| `parse_datz_bytes(raw_bytes, file_timestamp)` | Decode binary `.datZ` → DataFrame |
| `parse_datz_batch(file_data)` | Multi-file batch decoding |
| `validate_datz_file(raw_bytes)` | Quick validity check |
| `calculate_cycles(events_df, config)` | Cycle detection — barrier pulse or ring-barrier fallback |
| `validate_cycles(cycles_df)` | Check cycle lengths, duplicates |
| `get_cycle_stats(cycles_df)` | Summary statistics |
| `assign_events_to_cycles(events_df, cycles_df)` | `merge_asof` join |

---

## Performance Notes

- **Never aggregate on `events`** for date-level summaries — use `ingestion_log` or `cycles` instead (millions of rows vs. thousands)
- `get_processing_stats()` and `_get_dates_to_process()` both use `ingestion_log` for this reason
- The covering index `idx_events_ts_code (timestamp, event_code)` satisfies most range+filter queries without a full table scan

---

## Dependencies

```
pandas
pytz
sqlite3    # stdlib
zlib       # stdlib
struct     # stdlib
```
