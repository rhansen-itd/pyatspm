# pyATSPM CLI Reference

This document provides a comprehensive reference for the `atspm` command-line interface.

## `atspm setup`
Create a new intersection folder, metadata template, and config stub.

### Arguments:
* `--target` (Required): Intersection folder name, e.g. '2068_US-95_and_SH-8'.
* `--timezone`: IANA timezone for the new metadata.json (default: US/Mountain).

---

## `atspm process`
Ingest raw `.datZ` files for an intersection and compute cycles.

### Target Selection (Mutually Exclusive - Required)
* `--target`: Exact intersection folder name.
* `--targetid`: Intersection ID (prefix of folder name).
* `--all`: Process all intersections in the directory.

### Flags
* `--fill-gaps`: Enable Gap Fill mode: scan for and ingest historical gaps; scrub obsolete gap markers; surgically repair affected cycles.
* `--full`: Legacy alias for `--fill-gaps`.
* `--batch-size`: Number of `.datZ` files per transaction commit (default: 50).
* `--no-cycles`: Skip cycle processing; only ingest raw events.
* `--timezone`: Override the timezone from metadata.json (e.g. 'US/Pacific'). Useful for one-off corrections.

---

## `atspm report`
Generate ATSPM performance reports for one or more dates.

### Target Selection (Mutually Exclusive - Required)
* `--target`: Exact intersection folder name.
* `--targetid`: Intersection ID (prefix of folder name).
* `--all`: Generate reports for all intersections in the directory.

### Arguments & Flags
* `--dates` (Required): One or more local calendar dates to report on, e.g. `--dates 2026-02-19 2026-02-20`.
* `--backfill`: Run backfill_ring_phases() before generating reports. Safe to use repeatedly; fast when already complete.
* `--verbose`: Print full tracebacks for any per-date generation errors.

---

## `atspm discrepancies`
Analyze co-located detector discrepancies for a time window.

### Target Selection (Mutually Exclusive - Required)
* `--target`: Exact intersection folder name.
* `--targetid`: Intersection ID (prefix of folder name).
* `--all`: Analyze discrepancies for all intersections in the directory.

### Arguments & Flags
* `--start` (Required): Query window start (local time, ISO-8601). E.g. '2024-06-01T06:00:00'.
* `--end` (Required): Query window end, exclusive (local time, ISO-8601).
* `--lag`: Minimum disagreement duration in seconds (default: 2.0).
* `--timezone`: Override the timezone from metadata.json.
* `--output`: Write results to a CSV file in the intersection's outputs directory.
* `--verbose`: Print full tracebacks for any errors.

---

## `atspm plot-detectors`
Generate interactive detector comparison plots.

### Target Selection (Mutually Exclusive - Required)
* `--target`: Exact intersection folder name.
* `--targetid`: Intersection ID prefix (e.g. '2068').
* `--all`: Generate plots for all intersections in the directory.

### Arguments & Flags
* `--start` (Required): Window start (local time, ISO-8601). E.g. '2024-06-01T06:00:00'.
* `--end` (Required): Window end, exclusive (local time, ISO-8601).
* `--phases`: Filter to specific signal phases, e.g. `--phases 2 6`. Omit to include all configured pairs.
* `--lag`: Minimum disagreement duration (seconds) for extended-disagreement classification (default: 2.0).
* `--timezone`: Override the timezone from metadata.json.
* `--verbose`: Print full tracebacks for any errors.