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
- [src/atspm/cli.py] clock-drift gains --true-time (writes Clock_Model_*.csv, draws drift model)
- [src/atspm/data/reader.py] get_events_with_cycles_df gains true_time= (drift-corrected axis, needs Clk_* config)
- [src/atspm/analysis/__init__.py] export drift_model, to_true_time, apply_true_time
- [src/atspm/data/__init__.py] export load_drift_model
- [src/atspm/data/manager.py] int_cfg `Lanes:` rows import as config columns Lanes_{movement}, Lanes_{dir}_Layout
- [src/atspm/analysis/__init__.py] export parse_lane_config; phase_demand gains lanes=, outputs n_lanes/lane_source
- [src/atspm/cli.py] critical --basis per_lane divides by Lanes config lanes (detector proxy fallback)
- [src/atspm/analysis/true_time.py] drift_model interpolates hourly samples; MODEL_COLUMNS gains `segment`, rows are pieces
