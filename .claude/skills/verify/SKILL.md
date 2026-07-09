---
name: verify
description: Drive the atspm CLI against a throwaway intersection DB to observe a change working end-to-end.
---

# Verifying atspm changes end-to-end

The CLI entry point is `atspm` (installed from `[project.scripts]`; `.venv` is
already active in this repo). Every command resolves targets by walking up
from the **cwd** to find an `intersections/` directory — so build a scratch
project root anywhere and `cd` into it.

## Scratch intersection recipe

1. `mkdir -p <scratch>/proj/intersections/<ID>_Name`
2. `metadata.json` in that folder needs only:
   `{"intersection_name": "...", "db_filename": "<ID>_data.db", "intersection_id": "<ID>"}`
   (`devices.json` is only needed for retrieval commands.)
3. Create the DB with the app's own schema code — never hand-written DDL:
   - `atspm.data.manager.init_db(db_path)` → events/config/metadata/ingestion_log
   - instantiating `atspm.data.processing.CycleProcessor(db_path)` → cycles table
4. Seed via `DatabaseManager`: `insert_events([(ts, code, param), ...])`
   (include an `event_code == -1` row to exercise gap-rule paths), plus raw
   `INSERT INTO cycles (cycle_start) VALUES ...` and a minimal config row
   `INSERT INTO config (start_date, end_date) VALUES ('2020-01-01T00:00:00', '2022-01-01T00:00:00')`
   (RB_* columns optional; ring membership falls back to defaults).
5. Drive from the scratch root, e.g.
   `cd <scratch>/proj && atspm report --target <folder> --backfill --dates 2021-01-01`
6. Evidence: `sqlite3 <db> "SELECT cycle_start, r1_phases, r2_phases FROM cycles"`.

## Gotchas

- `CycleProcessor` timezone defaults to `US/Mountain` when the metadata table
  is empty; a UTC-midnight epoch lands on the *previous* local date, so
  per-date validation may report "no cycles" for the UTC date — harmless for
  backfill verification.
- `backfill_ring_phases` fetches events only up to `max(cycle_start) + 1s`,
  so the last pending cycle's greens are out of window and that row stays
  `None`/`None` (pre-existing behavior as of 2026-07; don't mistake it for
  your change).
- `atspm process` needs real `.datZ` files — for Functional Core changes,
  seed events directly and drive `report`/analysis commands instead.
