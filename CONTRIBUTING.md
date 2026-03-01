# Contributing to pyATSPM

We welcome contributions to pyATSPM!

## Getting Started
To get started, follow the installation instructions in [docs/INSTALL.md](docs/INSTALL.md) to set up your environment using `pip install -e .`.

## AI Agents & The Claude Constitution
This repository is frequently maintained and updated via autonomous AI agents. The instructions, role definitions, and strict rules for these agents are defined in the repository's "constitution", found in **`AGENTS.md`**.

**Important:** Before you start contributing, especially if you are using AI tools to assist your workflow, please read and abide by the rules outlined in `AGENTS.md`.

* **Allowed Tasks:** Repo-wide sweeps, consistent styling, DRY refactorings, boilerplate updates.
* **Forbidden Tasks:** DO NOT touch core pandas/numpy logic, visualization layers, or SQL logic without explicit instruction to do so.

## Submitting Pull Requests
1. Create a focused branch for your changes (`git checkout -b feature/my-new-feature`).
2. Keep your commits atomic and descriptive.
3. Be mindful of the **Functional Core / Imperative Shell** architecture. Separate pure transformations from state/database interactions as defined in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
4. Do not blindly overwrite large blocks of content without verifying.
5. Create a Pull Request against the main branch, detailing the changes and logic behind them.