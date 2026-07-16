Last doc sync: dd38a3837679d9b2c92052941802dbcc8239b805

<!--
Queue, not a log — cleared on every doc sync. One bullet per doc-relevant
change, terse: `- [file/path.py] what changed, one phrase`.
Only log changes to: SQLite schema, CLI subcommands/flags, public
__init__.py exports, or the Functional Core/Imperative Shell boundary.
See CLAUDE.md "Documentation Workflow" for the rules.
-->

- [src/atspm/analysis/cycles.py] validate_cycles: new optional gap_timestamps arg; assign_ring_phases green→cycle join now stops at -1 markers
