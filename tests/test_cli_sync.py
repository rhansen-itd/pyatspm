# Tests for the `atspm sync` subcommand (local <-> archive data movement).
#
# Covered: argument parsing, archive-root resolution precedence (flag > env >
# config file) and its --save side effect, component-group validation, the
# intersection -> sync-item adapter's classification, and the --release
# confirmation gate (the destructive part).

import argparse
import io
import json

import pytest

from atspm import cli


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project root with an empty intersections/ dir, cwd-active."""
    (tmp_path / "intersections").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestSyncParser:

    def _parse(self, argv):
        return cli._build_parser().parse_args(argv)

    def test_status_all(self):
        args = self._parse(["sync", "status", "--all"])
        assert args.sync_action == "status" and args.all is True
        assert args.func is cli.handle_sync

    def test_pull_by_id_defaults(self):
        args = self._parse(["sync", "pull", "--targetid", "201"])
        assert args.sync_action == "pull" and args.targetid == "201"
        assert args.release is False and args.quick is False and args.dry_run is False

    def test_push_release_flags(self):
        args = self._parse(["sync", "push", "--target", "201_Foo", "--release", "--yes"])
        assert args.release is True and args.yes is True

    def test_target_group_is_required(self):
        with pytest.raises(SystemExit):
            self._parse(["sync", "status"])

    def test_target_and_all_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse(["sync", "pull", "--targetid", "201", "--all"])

    def test_invalid_action_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["sync", "frobnicate", "--all"])


# ---------------------------------------------------------------------------
# Archive-root resolution
# ---------------------------------------------------------------------------

class TestArchiveRoot:

    def test_flag_takes_precedence(self, project, monkeypatch):
        monkeypatch.setenv(cli._SYNC_ARCHIVE_ENV, str(project / "from_env"))
        args = argparse.Namespace(archive_root=str(project / "from_flag"), save=False)
        root = cli._get_archive_root(args, must_exist=False)
        assert root == (project / "from_flag").resolve()

    def test_env_used_when_no_flag(self, project, monkeypatch):
        monkeypatch.setenv(cli._SYNC_ARCHIVE_ENV, str(project / "from_env"))
        args = argparse.Namespace(archive_root=None, save=False)
        root = cli._get_archive_root(args, must_exist=False)
        assert root == (project / "from_env").resolve()

    def test_config_file_used_when_no_flag_or_env(self, project, monkeypatch):
        monkeypatch.delenv(cli._SYNC_ARCHIVE_ENV, raising=False)
        (project / cli._SYNC_CONFIG_FILENAME).write_text(
            json.dumps({"archive_root": str(project / "from_cfg")})
        )
        args = argparse.Namespace(archive_root=None, save=False)
        root = cli._get_archive_root(args, must_exist=False)
        assert root == (project / "from_cfg").resolve()

    def test_unset_dies(self, project, monkeypatch):
        monkeypatch.delenv(cli._SYNC_ARCHIVE_ENV, raising=False)
        args = argparse.Namespace(archive_root=None, save=False)
        with pytest.raises(SystemExit):
            cli._get_archive_root(args, must_exist=False)

    def test_save_persists_to_config(self, project, monkeypatch):
        monkeypatch.delenv(cli._SYNC_ARCHIVE_ENV, raising=False)
        target = project / "ssd" / "intersections"
        args = argparse.Namespace(archive_root=str(target), save=True)
        cli._get_archive_root(args, must_exist=False)

        saved = json.loads((project / cli._SYNC_CONFIG_FILENAME).read_text())
        assert saved["archive_root"] == str(target.resolve())

    def test_must_exist_dies_when_absent(self, project, monkeypatch):
        monkeypatch.delenv(cli._SYNC_ARCHIVE_ENV, raising=False)
        args = argparse.Namespace(archive_root=str(project / "nope"), save=False)
        with pytest.raises(SystemExit):
            cli._get_archive_root(args, must_exist=True)


# ---------------------------------------------------------------------------
# Component parsing
# ---------------------------------------------------------------------------

class TestParseComponents:

    def test_all_means_none(self):
        assert cli._parse_sync_components("all", default="db") is None

    def test_default_used_when_none(self):
        assert cli._parse_sync_components(None, default="db,config") == {"db", "config"}

    def test_explicit_subset(self):
        assert cli._parse_sync_components("raw,outputs", default="all") == {"raw", "outputs"}

    def test_unknown_group_dies(self):
        with pytest.raises(SystemExit):
            cli._parse_sync_components("db,bogus", default="all")


# ---------------------------------------------------------------------------
# Intersection -> sync-item adapter
# ---------------------------------------------------------------------------

class TestIntersectionSyncItems:

    def test_classifies_every_top_level_entry(self, project):
        t = project / "intersections" / "201_SH-55"
        (t / "raw_data").mkdir(parents=True)
        (t / "outputs").mkdir()
        (t / "video").mkdir()
        (t / "201_20261001_1209").mkdir()            # a per-run export dir -> "other"
        (t / "201_data.db").write_bytes(b"db")
        (t / "201_data.db-wal").write_bytes(b"wal")  # sidecar, not its own item
        (t / "metadata.json").write_text("{}")
        (t / "int_cfg.csv").write_text("x\n")

        archive_root = project / "ssd" / "intersections"
        items = {it.name: it for it in cli._intersection_sync_items("201_SH-55", archive_root)}

        assert items["201_data.db"].group == "db"
        assert items["201_data.db"].companions == ("-wal", "-shm")
        assert items["raw_data"].group == "raw" and items["raw_data"].kind == "dir"
        assert items["outputs"].group == "outputs"
        assert items["video"].group == "video"
        assert items["201_20261001_1209"].group == "other"
        assert items["metadata.json"].group == "config"
        assert items["int_cfg.csv"].group == "config"
        assert "201_data.db-wal" not in items          # carried as a companion
        # Archive paths mirror the same relative layout.
        assert items["raw_data"].archive == archive_root / "201_SH-55" / "raw_data"

    def test_unions_local_and_archive_entries(self, project):
        local_t = project / "intersections" / "201_SH-55"
        local_t.mkdir(parents=True)
        (local_t / "201_data.db").write_bytes(b"db")

        archive_t = project / "ssd" / "intersections" / "201_SH-55"
        archive_t.mkdir(parents=True)
        (archive_t / "raw_data").mkdir()               # exists only in the archive

        items = {it.name for it in cli._intersection_sync_items(
            "201_SH-55", project / "ssd" / "intersections")}
        assert {"201_data.db", "raw_data"} <= items


# ---------------------------------------------------------------------------
# Release confirmation gate
# ---------------------------------------------------------------------------

class TestConfirmRelease:

    def test_assume_yes_skips_prompt(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input",
                            lambda _: pytest.fail("must not prompt with --yes"))
        cli._confirm_release(["201_Foo"], assume_yes=True)
        assert capsys.readouterr().out == ""

    def test_decline_cancels(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO())
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        with pytest.raises(SystemExit):
            cli._confirm_release(["201_Foo"], assume_yes=False)

    def test_non_interactive_refuses_without_yes(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO())
        monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
        with pytest.raises(SystemExit):
            cli._confirm_release(["201_Foo"], assume_yes=False)
        assert "--yes" in capsys.readouterr().err
