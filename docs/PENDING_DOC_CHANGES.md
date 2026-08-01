Last doc sync: ba3cac10fbd1db2b2dd186ea4d9cd916bee9e8b8

<!--
Queue, not a log — cleared on every doc sync. One bullet per doc-relevant
change, terse: `- [file/path.py] what changed, one phrase`.
Only log changes to: SQLite schema, CLI subcommands/flags, public
__init__.py exports, or the Functional Core/Imperative Shell boundary.
See CLAUDE.md "Documentation Workflow" for the rules.
-->

- [src/atspm/video/processor.py] accepts .ts input alongside .mp4; VideoOverlayResult gains timing_source
- [src/atspm/cli.py] video-overlay/-calibrate-shapes/-locate-phase-change --video document .ts input; --output restricted to writable containers


