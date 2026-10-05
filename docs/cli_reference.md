# CLI Reference

Entry point: `atspm` (installed via `pip install -e .`, see `pyproject.toml`). Implemented with `argparse` in `src/atspm/cli.py`.

## Target selection

Every subcommand except `setup` requires exactly one of:

| Flag | Meaning |
|---|---|
| `--target FOLDER` | Exact intersection folder name under `intersections/`, e.g. `2068_US-95_and_SH-8` |
| `--targetid ID` | Numeric intersection ID prefix, e.g. `2068`. Resolved by matching the part of the folder name before the first `_`. Fails if zero or more than one folder matches. |
| `--all` | Run the command for every intersection folder under `intersections/`. Failures on one intersection are logged and skipped; the batch continues. |

Most subcommands also accept `--timezone TZ` (overrides the IANA timezone in `metadata.json`) and `--verbose` (print full tracebacks instead of short error messages).

## Time arguments

Every `--start`/`--end` is **intersection local time** — the zone from `--timezone`, else `metadata.json`, else the intersection database's `metadata` table, else `US/Mountain`. The machine you run the command on never affects which rows are returned: reading a `US/Mountain` database from a UTC or Pacific host gives identical output.

## `atspm setup`

Scaffold a new intersection folder: `metadata.json` template, an empty `int_cfg.csv` placeholder, and an empty `devices.json` placeholder.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target FOLDER` | yes | — | Folder name to create, e.g. `2068_US-95_and_SH-8` |
| `--timezone TZ` | no | `US/Mountain` | IANA timezone written into `metadata.json` |

## `atspm retrieve`

Pull new `.datZ` files from an intersection's configured devices via SCP (see `devices.json` in [configuration.md](configuration.md)). Secondary devices (long-term storage) are always pulled before the controller, so the controller's bookmark never advances ahead of data a secondary device hasn't reported yet.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection (see above) |
| `--verbose` | no | off | Full tracebacks for unexpected per-intersection errors during `--all` |

## `atspm process`

Ingest `.datZ` files into `events` and compute `cycles`. Three mutually exclusive modes:

- **Fast Append** (default) — only files newer than the last ingested span are read; cycles are recalculated forward from the last known anchor.
- **Gap Fill** (`--fill-gaps`) — all files are scanned and historical gaps filled; obsolete gap markers are scrubbed and affected cycles surgically repaired.
- **Rebuild** (`--rebuild`) — `events`, `cycles` and `ingestion_log` are deleted, then every `.datZ` file is re-ingested from scratch. `config` and `metadata` survive (they come from `int_cfg.csv` and `metadata.json`). Use when stored timestamps need re-deriving on the current decoder basis. Destructive, so it confirms once per run and refuses outright on non-interactive stdin without `--yes`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection (see above) |
| `--fill-gaps` | no | off | Gap-fill mode (mutually exclusive with `--rebuild`) |
| `--rebuild` | no | off | Rebuild mode (mutually exclusive with `--fill-gaps`): delete `events`/`cycles`/`ingestion_log`, then re-ingest everything from `raw_data/` |
| `--yes` | no | off | Skip the `--rebuild` confirmation prompt (for scripted runs) |
| `--batch-size N` | no | `50` | `.datZ` files committed per transaction |
| `--no-cycles` | no | off | Ingest raw events only; skip cycle detection |
| `--timezone TZ` | no | metadata.json value | Override timezone |

## `atspm ingest-achd`

Ingest ACHD high-resolution event exports (`{id}_Events_*.csv`) into normalized pyATSPM databases, one per intersection under `intersections/achd/<id>/<id>_data.db`. Populates `events` (with comms-gap markers), `ingestion_log` spans, and `metadata` (id/name/timezone/agency). ACHD ids are namespaced under `achd/` so they never collide with the ITD intersections; cycle and config derivation are a separate pass (ACHD ships no `int_cfg.csv`). When an `intersections/achd/<id>/int_cfg.csv` is present it is imported into the DB `config` table (non-fatal), mirroring `process`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--targetid ID` / `--all` | yes | — | ACHD intersection id (e.g. `271`), or every id found under `--source` |
| `--source DIR` | yes | — | Directory of raw `{id}_Events_*.csv` exports |
| `--timezone TZ` | no | `US/Mountain` | Intersection wall-clock IANA zone |
| `--rebuild` | no | off | Clear existing `events`/`cycles`/`ingestion_log` before ingesting |
| `--verbose` | no | off | Full tracebacks on per-intersection errors during `--all` |

## `atspm report`

Generate the full set of Plotly HTML reports for one or more local dates (reprocessing cycles on demand if a date has none yet).

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--dates YYYY-MM-DD [...]` | yes | — | One or more local calendar dates |
| `--backfill` | no | off | Run `backfill_ring_phases()` before generating reports |
| `--verbose` | no | off | Full tracebacks on per-date errors |

## `atspm counts`

Vehicle/pedestrian counts to CSV.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start YYYY-MM-DD` | yes | — | Window start (local) |
| `--end YYYY-MM-DD` | yes | — | Window end (local) |
| `--bin-len N` | no | `60` | Minutes per bin, or `cycle` |
| `--type {vehicle,ped,combined}` | no | `combined` | Count type |
| `--hourly` | no | off | Scale numeric bins to an hourly flow rate |
| `--include-detectors` | no | off | Include raw per-detector count columns |
| `--exclude-missing` | no | off | Drop partial/missing bins from output |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm splits`

Phase timing splits to CSV.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start YYYY-MM-DD` | yes | — | Window start (local) |
| `--end YYYY-MM-DD` | yes | — | Window end (local) |
| `--bin-len N` | no | `cycle` | Minutes per bin, or `cycle` |
| `--report-mode {seconds,total,proportion}` | no | `seconds` | How values are expressed |
| `--phases N [N ...]` | no | all configured phases | Filter to specific phase IDs |
| `--include-no-clearance` | no | off | Treat phases with no served yellow as green-only |
| `--exclude-missing` | no | off | Drop partial/missing bins from output |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm aog`

Arrival on Green, per-cycle or binned.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start YYYY-MM-DD` | yes | — | Window start (local, inclusive) |
| `--end YYYY-MM-DD` | yes | — | Window end (local, inclusive) |
| `--phases N [N ...]` | no | all configured phases | Phases to analyze |
| `--offset SEC` | no | `0.0` | Arrival offset (travel-time correction), seconds |
| `--bin-len N` | no | `60` | Minutes per bin, or `cycle` |
| `--exclude-missing` | no | off | Drop `partial`/`missing` bins (full-day-missing is always dropped) |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm flow`

Effective cumulative flow-rate profiles from stop-bar detector departures (Code 81) within each phase split window. Stop-bar detector IDs come from the `Det_P<N>_Stop_Bar` config keys (`Det_P<N>_Stopbar` also accepted — see [configuration.md](configuration.md)). Only near-capacity cycles qualify (end slack ≤ `--max-lost`); those are then restricted to the modal split length and the busiest `--pct` percent. The peak of the mean profile identifies the throughput-optimal split length. Writes CSV plus an interactive HTML plot to the intersection's `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start YYYY-MM-DD` | yes | — | Window start (local, inclusive) |
| `--end YYYY-MM-DD` | yes | — | Window end (local, inclusive) |
| `--phases N [N ...]` | no | all phases with a `Det_P<N>_Stop_Bar` key | Phases to analyze |
| `--plans P [P ...]` | no | all plans | Coordination plan numbers to include |
| `--pct PCT` | no | `1.0` | Keep the busiest PCT percent of modal-split cycles by total vehicles |
| `--max-lost SEC` | no | `10.0` | Max seconds between the last departure and the end of the split window for a cycle to count as near-capacity |
| `--split-tolerance FRAC` | no | `0.10` | Fractional tolerance around the modal split length (±10%) |
| `--stratify` | no | off | Keep the busiest `--pct` percent within each `(plan, split)` stratum and pool them, instead of filtering around the modal split. Keeps shorter-split plans in the profile |
| `--normalize {end_shift,pooled,clearance,fixed,none}` | no | `end_shift` | Split-termination overhead added to elapsed time: `end_shift` = each cycle's measured end slack, `pooled` = per-detector median slack, `clearance` = actual yellow+red clearance, `fixed` = constant `--fixed-lost`, `none` = raw rate |
| `--fixed-lost SEC` | no | — | Constant overhead; required with `--normalize fixed` |
| `--rolling N` | no | `5` | Centred rolling-mean window (grid rows) for the instantaneous-rate plot traces; `1` disables |
| `--no-plot` | no | off | Write CSV tables only, skip HTML |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm critical`

Critical phases and the critical path per barrier group for a chosen period. Ring/barrier structure comes from `RB_R1`/`RB_R2` config (with a NEMA-standard fallback), cross-checked against observed cycle sequences; movement counts (`TM_*`) are mapped to phases by stop-bar detector overlap (`Det_P<N>_Stopbar`). Demand — vph, or vphpl with `--basis per_lane` — is the required-time proxy: per barrier group, the ring with the larger demand sum is the critical path. Writes CSV to the intersection's `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD`, or `YYYY-MM-DD HH:MM` for a sub-day peak period |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` (inclusive whole day) or `YYYY-MM-DD HH:MM` (exclusive) |
| `--bin-len N` | no | `15` | Demand-aggregation bin width in minutes |
| `--basis {per_lane,total}` | no | `per_lane` | Demand basis: `per_lane` = vph per detector (lane-count proxy), `total` = raw vph |
| `--include-missing` | no | off | Keep partial/missing bins when averaging demand. By default only quality-`ok` bins are used, since zero-filled missing bins bias mean demand downward |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm split-failures`

Purdue split failures (Green Occupancy Ratio vs. 5-second Red Occupancy Ratio) per phase split window. Presence detector IDs (one zone per lane) come from the `Det_P<N>_Occupancy` config keys — Stop Bar count channels are not used. Writes CSV plus an interactive HTML scatter plot to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` (inclusive) or `YYYY-MM-DD HH:MM` (exclusive) |
| `--phases N [N ...]` | no | all configured phases | Phases to analyze |
| `--aggregate {union,mean,any}` | no | `union` | Lane aggregation: `union` = occupied when any lane is on (UDOT); `mean` = average of per-lane GOR/ROR5; `any` = fails when any lane fails alone, reporting the worst lane |
| `--threshold FRAC` | no | `0.79` | Occupancy ratio above which a cycle fails |
| `--ror-seconds SEC` | no | `5.0` | Red occupancy window length, from yellow end |
| `--include-yellow` | no | off | Measure GOR over green + yellow (SPMs definition) instead of green only |
| `--bin-len N` | no | `60` | Minutes per bin, or `cycle` |
| `--exclude-missing` | no | off | Drop partial/missing bins (ignored in cycle mode) |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm approach-delay`

Per-cycle and binned approach delay and arrival shares (AoG/AoY/AoR) for advance-detector arrivals (UDOT S-M3). Detector IDs and travel times come from active config (`Det_P<N>_Arrival` and optional `Det_P<N>_Arrival_Travel`). Writes CSV + interactive HTML to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--phases N [N ...]` | no | all configured phases | Phases to analyze |
| `--offset SEC` | no | `0.0` | Stop-line travel time for phases lacking a `Det_P<N>_Arrival_Travel` key |
| `--bin-len N` | no | `15` | Minutes per bin, or `cycle` |
| `--exclude-missing` | no | off | Drop partial/missing bins (ignored in cycle mode) |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm approach-volume`

Approach volume with peak hour, PHF, K-factor, and D-factor (UDOT S-M8). Writes CSV + interactive HTML to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--bin-len M` | no | `15` | Minutes per bin (must divide 60) |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm yellow-red`

Per-cycle, binned, and per-plan yellow and red actuation counts and violations (UDOT S-M4) for stop-bar or occupancy detectors. Writes CSV + interactive HTML to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--phases N [N ...]` | no | all configured phases | Phases to analyze |
| `--role {stop_bar,occupancy}` | no | `stop_bar` | Detector role to classify |
| `--severe-sec S` | no | `4.0` | Severe-violation threshold, seconds after red start |
| `--bin-len M` | no | `15` | Minutes per bin |
| `--no-exclusions` | no | off | Ignore `TM_Exclusions` from `int_cfg.csv` |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm green-time`

Per-cycle, binned, and per-plan green time utilization (UDOT S-M7) for stop-bar or occupancy detectors. Writes CSV + interactive HTML to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--phases N [N ...]` | no | all configured phases | Phases to analyze |
| `--role {stop_bar,occupancy}` | no | `stop_bar` | Detector role to classify |
| `--bin-s S` | no | `2.0` | Width of a second-of-green bin, in seconds |
| `--bin-len M` | no | `15` | Minutes per aggregation bin |
| `--overlap` | no | off | Measure each phase's configured overlap (`Det_P<N>_Overlap`) instead |
| `--no-exclusions` | no | off | Ignore `TM_Exclusions` |
| `--max-green S` | no | `120.0` | Green-duration cap for plot heatmap bins (`0` = no cap) |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm left-turn-gap`

Per-green gap counts in opposing through traffic for permissive left turns (UDOT S-M9). Left/opposing movements are paired from `Det_P<N>_Direction`. Writes CSV + interactive HTML to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--left L [L ...]` | no | all | Left-turn movements to analyze, e.g. `EBL WBL` |
| `--bin-len M` | no | `15` | Minutes per bin (must divide 60) |
| `--edges LIST` | no | `1,3.3,3.7,7.4` | Comma-separated gap bin edges, in seconds |
| `--trend S` | no | `7.4` | Turnable-gap threshold for the trend line, in seconds |
| `--critical S` | no | lane rule from config | Critical-gap override, in seconds |
| `--gaps` | no | off | Also write the individual gap-event CSV (`LTG_Gaps_*.csv`) |
| `--no-exclusions` | no | off | Ignore `TM_Exclusions` |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm ped-delay`

Per-walk, binned, and per-plan pedestrian delay (UDOT S-M5). Writes CSV + interactive HTML to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--phases N [N ...]` | no | all configured phases | Phases to analyze |
| `--bin-len MINUTES` | no | `60` | Summary aggregation interval, in minutes |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm wait-time`

Per-window, binned, and per-plan vehicle wait time (UDOT S-M5). Writes CSV + interactive HTML to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--phases N [N ...]` | no | all configured phases | Phases to analyze |
| `--dropping {auto,on,off}` | no | `auto` | Whether phases use UDOT's dropping algorithm |
| `--max-wait SEC` | no | `360.0` | Cap on wait time for summary averages (`0` = no cap) |
| `--bin-len MINUTES` | no | `15` | Summary aggregation interval, in minutes |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm split-monitor`

UDOT split-monitor tables and plots (per-cycle services, per-plan statistics, timeline) for configured phases. Writes CSV + interactive HTML to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--phases N [N ...]` | no | all phases | Phases to analyze |
| `--percentiles A B` | no | `50.0 85.0` | Two split percentiles reported in stats |
| `--no-plot` | no | off | CSV only |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm preempt`

Preemption episodes (Code 105 family) from controller events, written as episode and summary CSV tables to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm clock-drift`

Decode pedestrian-call clock marks from the controller log, measure controller clock drift against the host clock, and identify clock-correction sets. The marker ped phases come from the `Clk_Behind`/`Clk_Ahead`/`Clk_Set` config. Writes CSV and an HTML plot to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--send-log PATH` | no | — | The head unit's `eos-time.jsonl` send log, for drift-against-host measurement |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm optimize`

Pick the cycle length C and splits that maximize saturated throughput `Σ 3600·N_p(s_p) / C` over the declared saturated phases, from measured cumulative discharge curves. Saturated phases are the engineer's declaration (`--saturated`); the end-slack classifier is printed only as an advisory. `--validate` instead tests the throughput model against the existing TOD plans. Writes CSV and interactive HTML plots to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start DATETIME` | yes | — | Period start (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--end DATETIME` | yes | — | Period end (local): `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` |
| `--saturated N [N ...]` | yes | — | Declared saturated phase numbers |
| `--plans ID [ID ...]` | no | all plans | Coordination plan IDs to filter cycles |
| `--pct PCT` | no | `1.0` | Percent of busiest modal-split cycles to keep (`100` = all) |
| `--split-tolerance TOL` | no | `0.10` | Split-duration tolerance around the target percentile |
| `--stratify` | no | off | Stratify discharge profiles by coordination plan |
| `--max-lost SEC` | no | `10.0` | Per-lane end-slack limit for advisory saturation |
| `--sat-threshold FRAC` | no | `0.8` | Threshold pass rate for advisory saturation |
| `--demand-stat {mean,peak}` | no | `mean` | Statistic for unsaturated-phase demand |
| `--default-min-split SEC` | no | `10.0` | Fallback minimum split |
| `--c-min SEC` | no | `60.0` | Shortest cycle scanned |
| `--c-max SEC` | no | `220.0` | Longest cycle scanned |
| `--c-step SEC` | no | `1.0` | Scan step |
| `--flat-tol-pct PCT` | no | `1.0` | Flat-band tolerance, percent of peak throughput |
| `--bin-len MIN` | no | `15` | Demand aggregation bin width, in minutes |
| `--include-missing` | no | off | Include partial/missing count bins when averaging demand |
| `--no-plot` | no | off | CSV only |
| `--validate` | no | off | Run model validation instead of the optimizer |
| `--min-plan-cycles N` | no | `30` | (`--validate`) Minimum complete cycles for a plan to be tested |
| `--split-cover-tol SEC` | no | `1.0` | (`--validate`) Split cover tolerance |
| `--rank-deadband-pct PCT` | no | `2.0` | (`--validate`) Ranking deadband percent |
| `--change-tol-pp PP` | no | `3.0` | (`--validate`) Magnitude tolerance in percentage points |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm discrepancies`

Co-located detector pair discrepancy analysis for a time window.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start ISO8601` | yes | — | Window start (local, e.g. `2024-06-01T06:00:00`) |
| `--end ISO8601` | yes | — | Window end, exclusive (local) |
| `--lag SEC` | no | `2.0` | Minimum disagreement duration counted as an anomaly |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--output` | no | off | Write results to CSV in the intersection's `outputs/` directory |
| `--verbose` | no | off | Full tracebacks |

## `atspm infer-detectors`

Propose a detector configuration for review from observed actuation behaviour. Never edits `int_cfg.csv` or the `config` table — it only writes a proposal to `outputs/`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start YYYY-MM-DD` | yes | — | Window start (local, inclusive) |
| `--end YYYY-MM-DD` | yes | — | Window end (local, inclusive) |
| `--min-actuations N` | no | `50` | Minimum uncensored on-intervals to classify a detector |
| `--all-phases` | no | off | Do not limit candidates to the `RB_*` ring phases |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm detector-health`

Run deterministic detector-health rules over raw events and activity profiles. Records findings into the `detector_findings` table and exports CSV plus HTML heatmaps to `outputs/`. The process exit code reflects the worst reported severity.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start YYYY-MM-DD` | yes | — | Window start (local, inclusive) |
| `--end YYYY-MM-DD` | no | `--start` | Window end (local, inclusive) |
| `--window {am,pm,day}` | no | `day` | Window to filter and report |
| `--min-severity {info,low,high}` | no | `low` | Minimum severity threshold to report |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm plot-detectors`

Interactive plot of co-located detector actuations with discrepancies highlighted.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start ISO8601` | yes | — | Window start (local) |
| `--end ISO8601` | yes | — | Window end, exclusive (local) |
| `--phases N [N ...]` | no | all configured pairs | Filter to specific phases |
| `--lag SEC` | no | `2.0` | Minimum disagreement duration for extended-disagreement classification |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm plot-coordination`

Interactive coordination/split diagram for a time window.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start ISO8601` | yes | — | Window start (local) |
| `--end ISO8601` | yes | — | Window end, exclusive (local) |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm plot-termination`

Interactive phase termination plot for a time window.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start ISO8601` | yes | — | Window start (local) |
| `--end ISO8601` | yes | — | Window end, exclusive (local) |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm plot-timing-actuation`

Visualise per-phase timing intervals (green, yellow, red), calls, pedestrian service, and detector actuations grouped by role, for a time window. The window is capped at 4 h, or at 24 h when narrowed with `--phases` or `--detectors`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection |
| `--start ISO8601` | yes | — | Window start (local) |
| `--end ISO8601` | yes | — | Window end, exclusive (local) |
| `--phases N [N ...]` | no | all | Filter to specific signal phases |
| `--detectors N [N ...]` | no | all | Filter to specific detector channels |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm video-calibrate-shapes`

Interactively draw/edit loop and stopbar shapes for one camera. Single-target only (`--target`/`--targetid`, no `--all`) — one calibration session is tied to one camera's video.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` | yes | — | Target selection (no `--all`) |
| `--camera NAME` | yes | — | Camera name, used as the shape-config filename stem (`<camera>_shapes.csv`) |
| `--video PATH` | yes | — | Video to calibrate against (first frame only); `.mp4` or `.ts`. Relative paths resolve against `<target>/video/`, absolute paths are used as-is |
| `--verbose` | no | off | Full tracebacks |

## `atspm video-overlay`

Render a video with live phase/overlap/detector status overlays, recoloring shapes drawn via `video-calibrate-shapes`. Single-target only.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` | yes | — | Target selection (no `--all`) |
| `--camera NAME` | yes | — | Camera name (matches the shape-config filename stem) |
| `--video PATH` | yes | — | Input video; `.mp4` (frame-decode recorder) or `.ts` (remux recorder). Relative paths resolve against `<target>/video/`, absolute paths are used as-is |
| `--start ISO8601` | yes | — | Real-world timestamp of the video's first frame (local time) |
| `--output PATH` | no | `<target>/outputs/<start-date>/<camera>_overlay_<start-time>.mp4` | Output video path. The overlay is re-encoded, so only writable containers are allowed (`.mp4`/`.m4v`/`.mov`/`.avi`); `.ts` is input-only |
| `--lookback MIN` | no | `10.0` | Minutes of event data fetched before/after the video window, for correct status at the clip's edges |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm video-sync`

Measure signal lamps in a recorded clip and align them against the database's controller phase/overlap states to recover an accurate first-frame `--start` timestamp and the camera's clock slip. Single-target only.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` | yes | — | Target selection (no `--all`) |
| `--camera NAME` | yes | — | Camera name (matches the shape-config filename stem) |
| `--video PATH` | yes | — | Input video; `.mp4` or `.ts`. Relative paths resolve against `<target>/video/`, absolute paths are used as-is |
| `--start-guess ISO8601` | yes | — | Estimated timestamp of the video's first frame (local time) |
| `--search SECONDS` | no | `30.0` | Half-window search duration, in seconds |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm video-locate-phase-change`

Auto-locate a phase's exact color-change time to correct a `--start` guess. Single-target only. Call once without `--observed-delta` to get a confirmation clip with a signed countdown label; read the value off the frame where the change actually happens, then call again with `--observed-delta <that value>` to get the corrected `--start`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` | yes | — | Target selection (no `--all`) |
| `--camera NAME` | yes | — | Camera name, used to name the output clip |
| `--video PATH` | yes | — | Input video; `.mp4` or `.ts`. Relative paths resolve against `<target>/video/`, absolute paths are used as-is |
| `--phase N` | yes | — | Signal phase number visible in the camera view |
| `--transition {green_to_yellow,yellow_to_red}` | no | auto-pick whichever occurs first | Pin the search to one edge |
| `--start ISO8601` | yes | — | Rough guess for the video's first-frame timestamp (local time) |
| `--min-offset SEC` | no | `5.0` | Only consider transitions at least this far into the video |
| `--window SEC` | no | `3.0` | Half-width of the confirmation clip |
| `--observed-delta SEC` | no | — | Signed value read off the clip's counter; when given, prints the corrected `--start` instead of rendering a clip |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm sync`

Copy intersection data between this machine's local drive and an archive drive (e.g. an external SSD), verifying every copy. Databases can't be queried over the 9p removable-media share, so the workflow is: `pull` a DB to local, work, then `push` results back. Set the archive root once per machine with `--archive-root ... --save` (persisted to `.atspm_sync.json`), or export `ATSPM_SYNC_ARCHIVE_ROOT`; it must point at the directory mirroring this project's `intersections/` folder on the archive drive.

The first positional argument selects the operation:

- **`status`** — show where each component lives and whether copies agree.
- **`pull`** — copy archive → local (default components `db,config`).
- **`push`** — copy local → archive (default `all`). With `--release`, the verified local copy is deleted afterward to free space.

| Flag | Required | Default | Description |
|---|---|---|---|
| `{status,pull,push}` | yes | — | Operation (positional) |
| `--target` / `--targetid` / `--all` | yes | — | Target selection (`--all` = union of local and archive for status/pull) |
| `--archive-root PATH` | no | env/config | Archive mirror of `intersections/` (overrides env var and config) |
| `--save` | no | off | Persist the resolved `--archive-root` to `.atspm_sync.json` |
| `--components LIST` | no | `db,config` (pull) / `all` (push, status) | Comma-separated groups, or `all`. Groups: `db`, `raw`, `outputs`, `video`, `config`, `other` |
| `--release` | no | off | `push` only: delete the local copy after a checksum-verified push |
| `--quick` | no | off | Verify by file size only, skipping SHA-256 (ignored for `--release`, which always checksums) |
| `--checksum` | no | off | `status` only: compare SHA-256 as well as size |
| `--dry-run` | no | off | Report what would be copied/released without changing anything |
| `--yes` | no | off | Skip the `--release` confirmation prompt |
| `--verbose` | no | off | Per-file progress and full tracebacks |

## Notes

- `counts`, `splits`, `aog`, `flow`, `infer-detectors`, and `detector-health` take plain `YYYY-MM-DD` dates; `discrepancies`, `plot-detectors`, `plot-coordination`, `plot-termination`, and `plot-timing-actuation` take full ISO-8601 datetimes (a time component is required). The analysis measures that accept a sub-day peak period (`critical`, `split-failures`, `approach-delay`, `approach-volume`, `yellow-red`, `green-time`, `left-turn-gap`, `ped-delay`, `wait-time`, `split-monitor`, `preempt`, `clock-drift`, `optimize`) take either `YYYY-MM-DD` or `YYYY-MM-DD HH:MM`.
- All subcommands with `--all` skip and log failed intersections rather than aborting the whole batch.
- The four `video-*` subcommands are single-target only (`--target`/`--targetid`, no `--all`) — one video file is one camera.
- `ingest-achd` is `--targetid`/`--all` only (no `--target`); its ids are namespaced under `intersections/achd/`.
