# CLAUDE.md — pyATSPM Project Constitution

## 1. Role & Objective
Expert Python Data Engineer and Traffic Operations Specialist building a high-performance ATSPM (Automated Traffic Signal Performance Measures) system from scratch.

- **Architecture:** Normalized SQLite ("One DB per Intersection") with a modern Python package structure (`src/atspm/`).
- **Philosophy:** Strict separation between data management (Imperative Shell) and mathematical logic (Functional Core).

## 2. Database Strategy
- **One SQLite file per intersection** (e.g., `2068_data.db`).
- **WAL mode:** always `PRAGMA journal_mode=WAL;` for concurrency.
- **No ORMs:** raw `sqlite3` for high-speed ingestion (writes). `pandas.read_sql` or `duckdb` for analysis (reads).

## 3. Implemented Schema
- **`events`** (raw data): `timestamp` (REAL), `event_code` (INT), `parameter` (INT). UNIQUE on `(timestamp, event_code, parameter)`.
- **`cycles`** (derived data): `cycle_start` (REAL PRIMARY KEY), `coord_plan` (INT), `detection_method` (TEXT), `r1_phases` (TEXT), `r2_phases` (TEXT).
- **`config`** (hybrid/temporal): `start_date`, `end_date`, dynamic columns (`TM_*`, `RB_*`, `Det_*`). `int_cfg.csv` is the single source of truth — it populates `config` via the hybrid schema transformation.
- **`metadata`** (static): `intersection_name`, `major_road_name`, `minor_road_name`, `agency_id`, etc.
- **`ingestion_log`** (state): `filename`, `processed_at`, `row_count`.

## 4. Architectural Style: Functional Core, Imperative Shell
- **Functional Core** (`src/atspm/analysis/`, `src/atspm/plotting/`): pure Python functions only. Input/output: DataFrames (standardized schema) or Plotly Figure objects. Strictly no side effects — no SQL connections, no file I/O, no `.write_html()`.
- **Imperative Shell** (`src/atspm/data/`, `src/atspm/reports/`): manages state/resources (DB connections, file paths). Fetches data via SQL, calls the Core, handles saving/rendering/logging.
- Engine classes (e.g., `SplitFailureEngine`, `DetectorEngine`) must mirror established patterns exactly (e.g., `AogEngine`, `PhaseEngine`, `CountsEngine`).

## 5. Refactoring & Coding Standards
- **Terminology sanitization (critical):**
  - No "legacy" or "orphan" anywhere in new code, comments, or docstrings; remove on sight when refactoring touched modules.
  - "pulse" = unconfirmed detector actuation; "isolated_pulse" is the correct term in analysis/plotting layers.
  - The previous compatibility adapter is called the "Events-with-cycles exporter."
- **Gap marker rule:** `event_code = -1` indicates a hard reset. Logic involving duration or sequential pairing MUST stop at a gap marker; never interpolate across it.
- **DRY helpers:** always import `resolve_pytz` from `utils.timezone` rather than reimplementing; use shared formatters from `utils.logging`; reuse existing internal functions (e.g., `_reconstruct_intervals`) rather than reimplementing logic.
- **Performance:** vectorize aggressively with Pandas/NumPy. No `iterrows()`/row iteration unless performance is proven irrelevant (especially in plotting code).
- **No speculative implementation:** don't infer or guess at existing implementations — work from authoritative source files actually present in the repo.

## 6. Visualization Standards
- **Library:** Plotly, exclusively. `write_html()` only ever in the imperative shell, never in the functional core.
- **Metadata integration:** titles must be dynamic using metadata, format `"{major_route} ({major_road_name}) & {minor_route} ({minor_road_name})"`. Handle missing values gracefully (omit parentheses/spaces if a road name is missing, omit parentheses around `minor_road_name` if `minor_route` is missing).
- Vectorized trace construction (e.g., `[start, end, None]` segment pattern); avoid dummy-trace legend hacks; keep hover colors consistent with the traces they describe.

## 7. User Interfaces & CLI
- **CLI implementation:** `argparse`, as implemented in `src/atspm/cli.py` (not Typer/Click).
- **CLI parity:** every new user-facing feature (ingestion, analysis, reporting) must include a corresponding CLI subcommand.
- Mutually exclusive target group pattern (`--target` / `--targetid` / `--all`) with batch `--all` support for new commands.

## 8. Output & Token Conservation
- No unsolicited docs: don't generate full READMEs or requirements files unless asked.
- Modular updates: only append/update affected sections of documentation (e.g., `ROADMAP.md`); use the live codebase as ground truth, not necessarily the existing README.
- Provide only the requested module/function, with Google-style docstrings.
- Validate before delivering: syntax/contract checks (schema correctness, aliased imports, fallback presence, gap-marker handling, CLI argument completeness) before code is considered done.

## Documentation Workflow

Code correctness comes first. Documentation (`README.md`, `docs/*.md`) is updated in a deliberate, separate pass — never automatically alongside a code change, even if the change seems to make a doc stale.

- **During coding tasks:** do not edit `README.md` or `docs/architecture.md`, `database_schema.md`, `configuration.md`, `cli_reference.md`, or `api_reference.md` unless explicitly asked to.
- **After finishing a coding task**, append at most one terse bullet per touched file to `docs/PENDING_DOC_CHANGES.md` — but only if the change is doc-relevant:
  - SQLite schema (`events`, `cycles`, `config`, `metadata`, `ingestion_log` — new/removed/renamed columns or tables)
  - A CLI subcommand or flag (`src/atspm/cli.py`) — added, removed, renamed, default changed
  - A public export in any `src/atspm/*/__init__.py`
  - The Functional Core / Imperative Shell boundary (e.g. I/O introduced into `analysis/` or `plotting/`)
  - Internal refactors, bug fixes with no signature/schema change, and perf tweaks are **not** logged — skip the bullet entirely.
  - Format: `- [path/to/file.py] one short phrase`. No elaboration, no rationale — the commit history holds that.
- **Doc sync, triggered only when the user explicitly asks** (e.g. "sync docs", "update the documentation"):
  1. Read `docs/PENDING_DOC_CHANGES.md` for the targeted list and the `Last doc sync: <sha>` marker.
  2. Cross-check completeness with `git diff <sha>..HEAD --stat -- src/` — catch anything doc-relevant that wasn't logged.
  3. Update only the affected `docs/*.md` / `README.md` sections.
  4. Clear the bullet list and bump `Last doc sync` to the current `HEAD` sha.

## Other agents in this repo
- `AGENTS.md` governs Jules, a separate autonomous agent scoped to repo-wide search/replace, naming/style enforcement, file migrations, and DRY refactors — it is explicitly forbidden from touching vectorization logic, Plotly code, SQL/schema, or functional core business logic. Don't assume its constraints apply here; this file is authoritative for Claude Code.
