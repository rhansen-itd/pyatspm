# Database Schema

Each intersection has its own SQLite file (e.g. `2068_data.db`) under `intersections/<folder>/`. `PRAGMA journal_mode=WAL` is always enabled. All timestamps are stored as `REAL` UTC epoch seconds — the zone lives in exactly one place, `metadata.timezone`, and is applied when converting to and from local time (see the Timezone contract in [api_reference.md](api_reference.md#timezone-contract)).

## `events` — raw data

```sql
CREATE TABLE events (
    timestamp   REAL    NOT NULL,
    event_code  INTEGER NOT NULL,
    parameter   INTEGER NOT NULL,
    UNIQUE(timestamp, event_code, parameter) ON CONFLICT IGNORE
)
```

Indexes: `idx_events_timestamp`, `idx_events_code_param`, `idx_events_ts_code` (covering index on `(timestamp, event_code)` — satisfies most range+filter queries without a full scan).

### Gap markers

`event_code = -1` is inserted by the ingestion pipeline whenever a discontinuity is detected (missing/corrupt file, controller reset). Any logic that computes a duration or pairs sequential events (phase splits, AOG, counts, detector intervals) must stop at a gap marker and never interpolate or bridge across it.

### Common event codes

| Code | Meaning |
|------|---------|
| -1 | Gap marker (data discontinuity, inserted by ingestion) |
| 1 | Phase Begin Green |
| 4 | Phase Gap Out |
| 5 | Phase Max Out |
| 6 | Phase Force Off |
| 8 | Phase Begin Yellow Clearance |
| 9 | Phase End Yellow Clearance |
| 10 | Phase Begin Red Clearance (only present if red clearance is served) |
| 11 | Phase End Red Clearance (only present if red clearance is served) |
| 12 | Phase Inactive (hard terminator) |
| 21 | Pedestrian Service Begin |
| 31 | Barrier pulse (primary cycle/coordination boundary marker) |
| 45 | Pedestrian Call Registered |
| 81 | Detector Off |
| 82 | Detector On |
| 61 | Overlap Begin Green |
| 63 | Overlap Begin Yellow |
| 64 | Overlap Begin Red Clearance (only present if RC served) |
| 65 | Overlap Off (inactive, red indication still shown) |
| 66 | Overlap Dark (no active output) |
| 105 | Preemption |
| 131 | Coordination plan change — `parameter` is the plan ID |
| 132 | Cycle length (seconds) — `parameter` is a duration, not a plan ID; not a `coord_plan` source |

## `cycles` — derived data

```sql
CREATE TABLE cycles (
    cycle_start      REAL    PRIMARY KEY,
    coord_plan       REAL    NOT NULL DEFAULT 0,
    detection_method TEXT    NOT NULL DEFAULT '',
    r1_phases        TEXT    NOT NULL DEFAULT 'None',
    r2_phases        TEXT    NOT NULL DEFAULT 'None'
)
```

Index: `idx_cycles_start`.

`detection_method` records how the cycle boundary was found:
- Code 31 barrier pulses, when present and unambiguous (no two Code-31 events sharing a timestamp).
- A ring-barrier-config fallback otherwise.

`r1_phases` / `r2_phases` are comma-separated phase lists per ring (e.g. `"2,6"`), or the literal string `"None"`.

Computed once by `CycleProcessor` (`src/atspm/data/processing.py`), which supports two re-entry paths:
- **Path A (fast append):** anchor at `MAX(cycle_start) <= T_start`; delete and recompute cycles after the anchor.
- **Path B (gap fill):** dual anchors (`CS_prev`, `CS_next`) around a gap; delete cycles between them and scrub gap markers in that range before recomputing.

## `config` — hybrid/temporal schema

```sql
CREATE TABLE config (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    start_date TEXT    NOT NULL,
    end_date   TEXT,
    -- dynamic columns added at import time, one per int_cfg.csv row category:
    -- TM_*, RB_*, Det_*, WD_*, TM_Exclusions
    UNIQUE(start_date) ON CONFLICT REPLACE
)
```

See [configuration.md](configuration.md) for how `int_cfg.csv` populates this table and what the dynamic column families mean.

## `metadata` — static intersection attributes

```sql
CREATE TABLE metadata (
    lock_id           INTEGER PRIMARY KEY CHECK (lock_id = 1),  -- enforces a single row
    intersection_id   TEXT,
    intersection_name TEXT,
    controller_ip     TEXT,
    detection_type    TEXT,
    detection_ip      TEXT,
    major_road_route  TEXT,
    major_road_name   TEXT,
    minor_road_route  TEXT,
    minor_road_name   TEXT,
    latitude          REAL,
    longitude         REAL,
    timezone          TEXT NOT NULL DEFAULT 'US/Mountain',
    agency_id         TEXT
)
```

`timezone` is the intersection's IANA zone and the only zone the package trusts. Its column default is interpolated from `utils.timezone.DEFAULT_TIMEZONE`, so the schema default and the Python-side fallback cannot drift apart. A database whose row is missing or whose `timezone` is blank resolves to that same default everywhere — `data.manager.db_timezone()` is the single entry point.

## `ingestion_log` — state tracking

```sql
CREATE TABLE ingestion_log (
    span_start   REAL PRIMARY KEY,
    span_end     REAL NOT NULL,
    processed_at TEXT NOT NULL,
    row_count    INTEGER NOT NULL
)
```

Index: `idx_ingestion_span_end`.

Tracks contiguous spans of ingested data rather than individual filenames. Adjacent/overlapping spans are merged (`MIN(start)`, `MAX(end)`, summed `row_count`) during gap-fill ingestion. Use this table — not `events` — for date-level coverage/summary queries; `events` can hold millions of rows per intersection.

## No ORM

All writes go through raw `sqlite3` (`executemany`, `INSERT OR IGNORE` / `ON CONFLICT REPLACE`) inside the `DatabaseManager` context manager, which guarantees commit/rollback atomicity. All bulk reads go through `pandas.read_sql_query`. No SQLAlchemy or other ORM is used anywhere in the data layer.
