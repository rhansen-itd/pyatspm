Last doc sync: bab4261368aa12d8ff92ac58ed7327efd17ef538

<!--
Queue, not a log — cleared on every doc sync. One bullet per doc-relevant
change, terse: `- [file/path.py] what changed, one phrase`.
Only log changes to: SQLite schema, CLI subcommands/flags, public
__init__.py exports, or the Functional Core/Imperative Shell boundary.
See CLAUDE.md "Documentation Workflow" for the rules.
-->

- [src/atspm/cli.py] new `flow` subcommand (split flow-rate tables + plots)
- [src/atspm/analysis/__init__.py] new exports `flow_rate`, `rate_profiles`
- [src/atspm/plotting/__init__.py] new export `plot_flow_profiles`
- [src/atspm/data/__init__.py] new exports `FlowRateEngine`, `get_flow_rate`
- [src/atspm/data/flow.py] new `Det_P<N>_Stopbar` config key (comma-separated stop-bar detector IDs)
- [src/atspm/cli.py] new `critical` subcommand (critical movement analysis)
- [src/atspm/analysis/__init__.py] new exports `ring_barrier_structure`, `movement_phase_map`, `phase_demand`, `critical_movement_analysis`
- [src/atspm/data/__init__.py] new exports `CriticalMovementEngine`, `get_critical_movements`
