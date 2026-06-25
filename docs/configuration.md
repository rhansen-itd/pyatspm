# Configuration

Each intersection folder (`intersections/<id>_<Name>/`) holds two human-edited config files: `metadata.json` (static attributes) and `int_cfg.csv` (temporal, per-period configuration). `atspm setup --target <folder>` scaffolds both as placeholders.

## `metadata.json`

Created by `atspm setup`, derived from the target folder name (`<id>_<Name>` → `intersection_id`/`intersection_name`), then synced into the `metadata` table by `atspm process` / `atspm report` via `set_metadata`.

```json
{
    "intersection_id":   "2068",
    "intersection_name": "US-95 and SH-8",
    "timezone":          "US/Mountain",
    "folder_name":       "2068_US-95_and_SH-8",
    "db_filename":       "2068_data.db",

    "controller_ip":     null,
    "detection_type":    null,
    "detection_ip":      null,
    "agency_id":         null,

    "major_road_route":  null,
    "major_road_name":   null,
    "minor_road_route":  null,
    "minor_road_name":   null,

    "latitude":          null,
    "longitude":         null
}
```

`intersection_id`, `intersection_name`, `timezone`, `folder_name`, and `db_filename` are filled in automatically. Everything else is left `null` and should be filled in by hand before running `atspm process`/`atspm report` if you want it reflected in plot titles and metadata-driven labeling (see [architecture.md](architecture.md) and the title format rule below).

Plot titles built from this metadata follow the format `"{major_route} ({major_road_name}) & {minor_route} ({minor_road_name})"`, omitting parentheses/spaces around a road name that's missing, and omitting parentheses around `minor_road_name` if `minor_route` is missing.

## `int_cfg.csv`

A multi-index CSV: the first two columns are `(Category, Parameter)`, and every column after that is a date (the start of a configuration period). `atspm setup` writes an empty placeholder (`Category,Parameter,Value`) for you to replace.

Loaded with `pd.read_csv(csv_path, index_col=[0, 1])`. Each date column becomes one row in the `config` table; `end_date` is computed automatically as the next date column (or `NULL` for the most recent period).

| Category | Parameter | 2024-01-01 | 2024-06-15 |
|----------|-----------|------------|------------|
| TM: | EBL | 5,6,7 | 5,6,7 |
| TM: | SBThru | 10,11,12 | 10,11 |
| RB: | R1 | 1,2\|3,4 | 1,2\|3,4 |
| RB: | R2 | 5,6\|7,8 | 5,6\|7,8 |
| Det: | P2_Arrival | 33,34 | 33,34 |
| Det: | P2_Pairs | [[33,40]] | [[33,40]] |
| Exc: | Detector | 33,34 | |
| Exc: | Phase | 2,2 | |
| Exc: | Status | Red,Yellow | |

### Category → column mapping (`manager.py: _transform_config_column`)

| CSV category | DB column prefix | Notes |
|---|---|---|
| `TM:` | `TM_<movement>` | e.g. `TM_EBL` — detector IDs assigned to a named movement, used by `vehicle_counts` |
| `RB:` | `RB_<param>` | e.g. `RB_R1` — ring-barrier phase membership |
| `Det:` / `Plt:` | `Det_<param>` | both prefixes map to the same `Det_` column family |
| `WD:` | `WD_<param>` | watchdog/timing parameters |
| `Exc:` | — | rows are collected and JSON-encoded into a single `TM_Exclusions` column instead of one column per row |

### `Det_` column patterns

- **`Det_P<N>_Arrival`** — comma-separated advance-detector IDs for phase `N` (e.g. `"33,34"`). Read by `AogEngine`/`arrival_on_green` to find which detectors count as "arrivals" for that phase.
- **`Det_P<N>_Pairs`** — JSON list of `[det_a, det_b]` pairs for phase `N` (e.g. `"[[33,40]]"`), matched by the regex `^Det_P(\d+)_Pairs$` and parsed into `[{"phase": N, "det_a": ..., "det_b": ...}, ...]`. Read by `DetectorEngine`/`analyze_discrepancies` to find co-located detector pairs to compare.

### `TM_Exclusions`

`Exc:` rows are parsed into a JSON array stored in the `TM_Exclusions` column:

```json
[{"detector": 33, "phase": 2, "status": "Red"}, {"detector": 34, "phase": 2, "status": "Yellow"}]
```

Consumed by `parse_exclusions_from_config` / `vehicle_counts` to drop detector actuations that occur while the listed phase is in the listed state (e.g. a detector that double-counts during a particular signal state).
