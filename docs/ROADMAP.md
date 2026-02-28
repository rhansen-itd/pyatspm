\# pyatspm Roadmap \& Feature Backlog

\*(Last updated: February 2026)\*



## 1. AI Delegation Strategy
Use this matrix to decide which AI/tool to engage for each task type.

| Task Type                              | Best Tool                  | Best for Planning / Prompt Writing | Rationale / Notes                                                                 |
|----------------------------------------|----------------------------|-------------------------------------|-----------------------------------------------------------------------------------|
| Complex pandas/numpy vectorization     | Claude                     | Grok / Gemini (chatbot)             | Claude excels at analytical optimization; others help craft tight prompts.       |
| Detector state machines, SQL pipelines | Claude                     | Grok                                | Deep domain logic + architecture integrity.                                       |
| Repo-wide terminology sweeps & renames | Jules (agent)              | Grok / Gemini (chatbot)             | Full repo read/write access; zero copy-paste friction; catches buried references. |
| File migrations, bulk DRY refactors   | Jules (agent)              | Grok / Gemini (chatbot)             | "Set it and forget it" — renames files, updates imports/paths atomically.         |
| Small repo cleanups (imports, lint)    | Jules (agent)              | —                                   | Ongoing hygiene tasks (e.g. sort imports, remove unused vars) after big changes.  |
| CLI tools, new feature boilerplate     | Claude                     | Grok / Gemini                       | Claude keeps context for consistent Typer/Click style + feature integration.     |
| Dash / Streamlit / web prototypes      | Gemini (chatbot)           | Grok                                | Gemini strong at modern interactive dashboard scaffolding.                       |
| Documentation — initial major overhaul | Claude                     | —                                   | Deep project understanding from recent code work.                                 |
| Documentation — ongoing updates        | Gemini (chatbot) or Grok   | —                                   | Low token cost; feed diffs/changelogs for targeted patches.                      |



\## 2. Continuous Directives (The Claude Constitution)

\*Add these rules to your system prompts for Claude to ensure consistency across all future generation.\*



\* \*\*DRY Logging/Timezones:\*\* Never duplicate timezone handling or JSON log formatting code. Always import from `utils.timezone.resolve\_pytz` and the shared formatter from `utils.logging`.

\* \*\*Vectorization First:\*\* All new analysis/plotting modules must be vectorized where possible using `numpy`/`pandas` and include performance timing/logging.

\* \*\*CLI Parity:\*\* Every new user-facing feature (analysis, report, plot, ingestion) must expose a clean CLI command.

\* \*\*Terminology Standardization:\*\* Enforce "pulse" terminology globally. When renaming concepts (e.g., orphan -> pulse), do it consistently across code, comments, variables, docstrings, logs, and documentation.

\* \*\*Token Preservation:\*\* Documentation updates are allowed without explicit request \*only\* when a major architectural or feature change is made—and only the changed sections. Do not generate full `README` files or examples unless asked.



\## 3. Phase 1: Core Architecture \& Ingestion (High Priority)

\*Foundational cleanup and data pipeline improvements.\*



\* \*\*DRY Refactoring (Logging \& Timezones):\*\* Extract `\_resolve\_pytz` and JSON formatters into `utils/` modules. \*(Status: Prompt ready to deploy).\*

\* \*\*Direct Controller Integration (SCP Tools):\*\* Add SCP retrieval tools (e.g., `pyatspm fetch-datz`) to pull `.datz` files directly from controllers over the network.

\* \*\*Alternative Data Source Ingestion (CSV):\*\* Refactor the existing working prototype for CSV ingestion to fit seamlessly into the new SQL-based database architecture.



\## 4. Phase 2: The Detector Overhaul

\*Group these tasks into a single prompting session/sprint since they all touch detector configurations and logic.\*



\* \*\*Configuration Schema Update:\*\* Update the intersection configuration database schema to explicitly include paired detectors (used by the discrepancy engine and future plots).

\* \*\*Terminology Sweep:\*\* Execute the "orphan" to "pulse" rename across all detector analysis and data modules.

\* \*\*Detector Comparison Plot:\*\* Implement a new plot module based on the legacy version. Use the newly defined detector pairs to visually plot states side-by-side or overlaid.



\## 5. Phase 3: Advanced UDOT ATSPM Analytics

\*Rewrite, vectorize, and optimize legacy modules for the SQL backend.\*



\* \*\*Arrival on Red (AOR):\*\* Complete rewrite and vectorization for the SQL backend.

\* \*\*Split Failures:\*\* Complete rewrite and vectorization for the SQL backend.

\* \*\*Normalized Flow Rate Plot:\*\* Add a module that plots instantaneous and lost-time normalized vehicle flow rate from the start of green.

\* \*\*Remaining UDOT ATSPM Plots (Progressive Priority):\*\*

&nbsp;   \* Phase termination

&nbsp;   \* Approach volume / speed / delay

&nbsp;   \* Purdue coordination diagram

&nbsp;   \* Split monitor



\## 6. Phase 4: User Interface \& Documentation

\*Bringing it all together for the end-user.\*



\* \*\*Major Documentation Overhaul:\*\* Draft the first full User Guide covering architecture, installation, database configuration, CLI usage, and example outputs.

\* \*\*Web-Based GUI (Dash or Streamlit):\*\* Build a frontend application for interactive plotting. \*Recommendation: Start with Dash (Plotly) or Streamlit for the fastest path to interactive Python data dashboards without needing custom JS/HTML.\*

