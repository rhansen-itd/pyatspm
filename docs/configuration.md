# Configuration

Each intersection folder (`intersections/<id>_<Name>/`) holds human-edited config files: `metadata.json` (static attributes), `int_cfg.csv` (temporal, per-period configuration), and `devices.json` (retrieval device list). `atspm setup --target <folder>` scaffolds all three as placeholders. A fourth, per-camera config (`video/<camera>_shapes.csv`) is scaffolded separately by `atspm video-calibrate-shapes`.

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
- **`Det_P<N>_Stopbar`** — comma-separated stop-bar detector IDs for phase `N` (e.g. `"1,2"`). Read by `FlowRateEngine`/`flow_rate` to measure departures within the split window, and by `CriticalMovementEngine` to map `TM_*` movements onto phases by detector overlap. A movement whose detectors overlap no phase's stop-bar set, or more than one, is reported as unmapped and excluded from the demand total rather than guessed at.

### `TM_Exclusions`

`Exc:` rows are parsed into a JSON array stored in the `TM_Exclusions` column:

```json
[{"detector": 33, "phase": 2, "status": "Red"}, {"detector": 34, "phase": 2, "status": "Yellow"}]
```

Consumed by `parse_exclusions_from_config` / `vehicle_counts` to drop detector actuations that occur while the listed phase is in the listed state (e.g. a detector that double-counts during a particular signal state).

## `devices.json`

Created by `atspm setup` as an empty `[]` placeholder; consumed by `atspm retrieve` (`RetrievalEngine`/`run_retrieval`). A JSON list of device entries, one per device that `.datZ` files should be pulled from via SCP:

```json
[
    {
        "role":          "controller",
        "device_type":   null,
        "host":          null,
        "port":          22,
        "user":          "user",
        "password":      "password",
        "remote_folder": "/path/to/datz",
        "last_retrieved": null
    }
]
```

- **`role`** — `"secondary"` devices (long-term storage, e.g. an EVO radar unit) are always pulled before `"controller"` devices, regardless of list order, so the controller's short FIFO retention window can't advance past data a secondary device hasn't reported yet.
- **`host`** — explicit IP/hostname. Only a `"controller"` entry may omit it, in which case it falls back to `metadata.json`'s `controller_ip` (there is exactly one controller per intersection, so the fallback is unambiguous). Every other role must set `host` explicitly — there is no analogous `detection_ip`-style fallback for secondary devices.
- **`device_type`** — selects the filename→timestamp parser used to find files newer than `last_retrieved`; falls back to a generic parser when `null` or unrecognized.
- **`last_retrieved`** — ISO-8601 bookmark, advanced in place after each successful run; `atspm retrieve` rewrites `devices.json` with the updated value.

## `video/<camera>_shapes.csv`

Per-camera loop/stopbar shape definitions, scaffolded and edited interactively by `atspm video-calibrate-shapes` (`ShapeConfig.load`/`.save`) and consumed by `atspm video-overlay` (`render_overlay`). Lives at `intersections/<folder>/video/<camera>_shapes.csv` — a sibling of `raw_data/`/`outputs/`, but tied to a specific camera and its recorded resolution rather than date-versioned like `int_cfg.csv`.

A 2-section CSV: a one-row metadata header (`video_width`/`video_height`) followed by the per-shape table, so resolution is recorded once per file instead of repeated on every shape row:

```
video_width,video_height
1920,1080
type,points,color,input,phase,name
loop,100,100;200,100;200,200;100,200,"0,255,0",3,,South Loop 3
stopbar,50,300;250,300,"0,0,255",,2,Southbound Stop Bar
```

| Column | Meaning |
|---|---|
| `type` | `"loop"` or `"stopbar"` |
| `points` | `;`-separated `x,y` pixel coordinates |
| `color` | `R,G,B` outline color |
| `input` | Detector channel number (loop shapes only) |
| `phase` | Phase number `1`-`16`, or an overlap letter `"OLA"`-`"OLP"` (stopbar shapes only) — resolved by `resolve_stopbar_target`, which rejects out-of-range numbers |
| `name` | Free-text label |

Overlap letters map to numbers `1`-`16` (`A=1, B=2, ...`) via `OVERLAP_LETTER_MAP`, matching the Indiana/Purdue Hi-Res Logger Enumerations spec's overlap-number convention for event codes 61-66. `render_overlay` rejects a shape config whose recorded `video_width`/`video_height` doesn't match the actual video's resolution — shapes are calibrated pixel-exact and are not rescaled.
