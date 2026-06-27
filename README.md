# pyATSPM

A Python package for Automated Traffic Signal Performance Measures (ATSPM) analysis: ingest raw signal controller event logs (`.datZ`), store them in a normalized per-intersection SQLite database, and compute signal performance measures (Arrival on Green, counts, phase splits, detector discrepancies) with Plotly visualizations.

Built on two principles:
- **One SQLite database per intersection** — normalized, indexed, WAL-mode.
- **Functional Core / Imperative Shell** — pure analysis/plotting functions, kept separate from all I/O.

See [docs/architecture.md](docs/architecture.md) for how the layers fit together.

## Project Structure

```
pyatspm/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── architecture.md       # Functional Core / Imperative Shell, design principles
│   ├── database_schema.md    # events, cycles, config, metadata, ingestion_log tables
│   ├── configuration.md      # metadata.json and int_cfg.csv reference
│   ├── cli_reference.md      # every `atspm` subcommand and flag
│   ├── api_reference.md      # public functions/classes per package
│   └── ROADMAP.md
│
├── src/
│   └── atspm/
│       ├── data/           # Imperative Shell — DB I/O, ingestion, retrieval, engines
│       ├── analysis/       # Functional Core — pure transformations
│       ├── plotting/       # Functional Core — pure Plotly figure builders
│       ├── video/          # Imperative Shell (one exception) — OpenCV overlay rendering
│       ├── reports/        # Imperative Shell — report orchestration, HTML output
│       └── cli.py          # argparse CLI (entry point: `atspm`)
│
├── notebooks/              # exploratory analysis
│
└── intersections/          # per-intersection data (gitignored)
    └── 2068_US-95_and_SH-8/
        ├── metadata.json
        ├── int_cfg.csv
        ├── devices.json
        ├── 2068_data.db
        ├── raw_data/        # .datZ files go here
        ├── video/           # camera videos + <camera>_shapes.csv configs
        └── outputs/         # generated reports/CSVs
```

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9. Dependencies: `pandas`, `pytz`, `plotly`, `opencv-python`, `paramiko`, `scp` (see `pyproject.toml`/`requirements.txt`).

## Quickstart

```bash
# 1. Scaffold a new intersection
atspm setup --target 2068_US-95_and_SH-8 --timezone US/Mountain
# → fill in intersections/2068_US-95_and_SH-8/metadata.json and int_cfg.csv,
#   then drop .datZ files into raw_data/

# 2. Ingest events and detect cycles
atspm process --target 2068_US-95_and_SH-8

# 3. Generate reports for a date
atspm report --target 2068_US-95_and_SH-8 --dates 2025-01-15

# Or pull a specific measure straight to CSV
atspm aog --target 2068_US-95_and_SH-8 --start 2025-01-01 --end 2025-01-07 --phases 2 6
```

Every subcommand also accepts `--targetid <numeric id>` or `--all` instead of `--target` to run against one-by-ID or every intersection in `intersections/`. Full subcommand/flag reference: [docs/cli_reference.md](docs/cli_reference.md).

## Documentation

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Functional Core / Imperative Shell, data flow, design principles, terminology |
| [docs/database_schema.md](docs/database_schema.md) | Full SQLite schema, event codes, gap-marker rule |
| [docs/configuration.md](docs/configuration.md) | `metadata.json` and `int_cfg.csv` reference |
| [docs/cli_reference.md](docs/cli_reference.md) | Every `atspm` subcommand and flag |
| [docs/api_reference.md](docs/api_reference.md) | Public functions/classes for scripting against the package directly |

## Scripting Example

```python
from pathlib import Path
from datetime import datetime
from atspm.data import get_events_with_cycles_df, get_config_df

db_path = Path("intersections/2068_US-95_and_SH-8/2068_data.db")

df = get_events_with_cycles_df(
    db_path,
    start=datetime(2025, 1, 1),
    end=datetime(2025, 1, 2),
)

config = get_config_df(db_path, datetime(2025, 1, 1))
```
