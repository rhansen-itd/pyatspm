Last doc sync: cba4a1d3749ae63489a538d90f0614e72b750baa

<!--
Queue, not a log — cleared on every doc sync. One bullet per doc-relevant
change, terse: `- [file/path.py] what changed, one phrase`.
Only log changes to: SQLite schema, CLI subcommands/flags, public
__init__.py exports, or the Functional Core/Imperative Shell boundary.
See CLAUDE.md "Documentation Workflow" for the rules.
-->

- [src/atspm/analysis/__init__.py] new export `parse_datz_header`
- [src/atspm/analysis/decoders.py] `parse_datz_bytes` now shifts the event base by the header's sub-minute offset
- [src/atspm/cli.py] `process` gains `--rebuild` (Path C, mutually exclusive with `--fill-gaps`) and `--yes`
- [src/atspm/data/__init__.py] new exports `MIN_PHASE_NUMBER` / `MAX_PHASE_NUMBER`; `resolve_stopbar_target` now rejects phases outside 1-16

