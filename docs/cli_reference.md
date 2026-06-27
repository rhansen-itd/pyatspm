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

Ingest `.datZ` files into `events` and compute `cycles`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` / `--all` | yes | — | Target selection (see above) |
| `--fill-gaps` | no | off | Gap-fill mode: scan **all** files and repair gaps, instead of only appending files newer than the last ingested span |
| `--batch-size N` | no | `50` | `.datZ` files committed per transaction |
| `--no-cycles` | no | off | Ingest raw events only; skip cycle detection |
| `--timezone TZ` | no | metadata.json value | Override timezone |

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

## `atspm video-calibrate-shapes`

Interactively draw/edit loop and stopbar shapes for one camera. Single-target only (`--target`/`--targetid`, no `--all`) — one calibration session is tied to one camera's video.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` | yes | — | Target selection (no `--all`) |
| `--camera NAME` | yes | — | Camera name, used as the shape-config filename stem (`<camera>_shapes.csv`) |
| `--video PATH` | yes | — | Video to calibrate against (first frame only). Relative paths resolve against `<target>/video/` |
| `--verbose` | no | off | Full tracebacks |

## `atspm video-overlay`

Render a video with live phase/overlap/detector status overlays, recoloring shapes drawn via `video-calibrate-shapes`. Single-target only.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` | yes | — | Target selection (no `--all`) |
| `--camera NAME` | yes | — | Camera name (matches the shape-config filename stem) |
| `--video PATH` | yes | — | Input video. Relative paths resolve against `<target>/video/` |
| `--start ISO8601` | yes | — | Real-world timestamp of the video's first frame (local time) |
| `--output PATH` | no | `<target>/outputs/<start-date>/<camera>_overlay_<start-time>.mp4` | Output video path |
| `--lookback MIN` | no | `10.0` | Minutes of event data fetched before/after the video window, for correct status at the clip's edges |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## `atspm video-locate-phase-change`

Auto-locate a phase's exact color-change time to correct a `--start` guess. Single-target only. Call once without `--observed-delta` to get a confirmation clip with a signed countdown label; read the value off the frame where the change actually happens, then call again with `--observed-delta <that value>` to get the corrected `--start`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--target` / `--targetid` | yes | — | Target selection (no `--all`) |
| `--camera NAME` | yes | — | Camera name, used to name the output clip |
| `--video PATH` | yes | — | Input video. Relative paths resolve against `<target>/video/` |
| `--phase N` | yes | — | Signal phase number visible in the camera view |
| `--transition {green_to_yellow,yellow_to_red}` | no | auto-pick whichever occurs first | Pin the search to one edge |
| `--start ISO8601` | yes | — | Rough guess for the video's first-frame timestamp (local time) |
| `--min-offset SEC` | no | `5.0` | Only consider transitions at least this far into the video |
| `--window SEC` | no | `3.0` | Half-width of the confirmation clip |
| `--observed-delta SEC` | no | — | Signed value read off the clip's counter; when given, prints the corrected `--start` instead of rendering a clip |
| `--timezone TZ` | no | metadata.json value | Override timezone |
| `--verbose` | no | off | Full tracebacks |

## Notes

- `counts`, `splits`, and `aog` take plain `YYYY-MM-DD` dates; `discrepancies` and the `plot-*` commands take full ISO-8601 datetimes (a time component is required).
- All subcommands with `--all` skip and log failed intersections rather than aborting the whole batch.
- The three `video-*` subcommands are single-target only (`--target`/`--targetid`, no `--all`) — one video file is one camera.
