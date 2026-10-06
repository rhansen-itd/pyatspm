Last doc sync: cc6c09691db893bd03d6b5c85e0cc414b8a3990e

<!--
Queue, not a log — cleared on every doc sync. One bullet per doc-relevant
change, terse: `- [file/path.py] what changed, one phrase`.
Only log changes to: SQLite schema, CLI subcommands/flags, public
__init__.py exports, or the Functional Core/Imperative Shell boundary.
See CLAUDE.md "Documentation Workflow" for the rules.
-->

- [src/atspm/cli.py] new pack-raw subcommand (--target/--targetid/--all, --include-current, --keep-loose, --dry-run, --verbose); sync push gains --pack
- [src/atspm/data/__init__.py] export DatzSource, PackResult, parse_datz_month, list_monthly_candidates, verify_archive, pack_monthly_archive, pack_intersection_raw
