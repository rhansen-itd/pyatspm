# pyATSPM Installation Guide

This guide describes how to install the `atspm` python package and CLI.

## Requirements
* Python 3.9 or higher

## Installation

### 1. Create a Virtual Environment (Recommended)
First, ensure you are in the project root directory where `pyproject.toml` and `requirements.txt` are located.
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
Install all required libraries for core analysis, database handling, and interactive plotting from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Install pyATSPM Package
To ensure the `atspm` command line interface (CLI) entry point is correctly registered, you must install the package itself.

It is highly recommended to do an "editable" install (`-e`). This means any future changes to the python code in `src/atspm/` will immediately be reflected without needing to reinstall the package:
```bash
pip install -e .
```

### 4. Verify Installation
Verify that the CLI is installed and accessible by running:
```bash
atspm --help
```
You should see the help documentation outlining the available commands (`setup`, `process`, `report`, `discrepancies`, and `plot-detectors`).