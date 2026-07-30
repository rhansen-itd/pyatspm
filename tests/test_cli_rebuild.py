# Tests for the `atspm process --rebuild` safety gate.
#
# --rebuild deletes every events/cycles/ingestion_log row, so the parts worth
# pinning are the ones that stop it happening by accident: the confirmation
# prompt, the non-interactive refusal, and mutual exclusion with --fill-gaps.
# The deletion itself is covered by tests/data/test_manager_rebuild.py.

import argparse
import io

import pytest

from atspm import cli


class TestConfirmRebuild:

    def test_assume_yes_skips_the_prompt(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise AssertionError("input() must not be called with --yes")

        monkeypatch.setattr("builtins.input", _boom)
        cli._confirm_rebuild(["201_Main"], assume_yes=True)

        assert capsys.readouterr().out == ""

    def test_accepting_returns_and_lists_every_target(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO())
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _: "y")

        cli._confirm_rebuild(["201_Main", "315_Franklin"], assume_yes=False)

        out = capsys.readouterr().out
        assert "201_Main" in out
        assert "315_Franklin" in out
        assert "2 intersection(s)" in out

    @pytest.mark.parametrize("answer", ["y", "Y", "yes", "  YES  "])
    def test_affirmative_answers_proceed(self, monkeypatch, answer):
        monkeypatch.setattr("sys.stdin", io.StringIO())
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _: answer)

        cli._confirm_rebuild(["201_Main"], assume_yes=False)

    @pytest.mark.parametrize("answer", ["", "n", "no", "q", "yep"])
    def test_anything_else_cancels(self, monkeypatch, answer):
        monkeypatch.setattr("sys.stdin", io.StringIO())
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _: answer)

        with pytest.raises(SystemExit):
            cli._confirm_rebuild(["201_Main"], assume_yes=False)

    def test_non_interactive_stdin_refuses_without_yes(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO())
        monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
        monkeypatch.setattr(
            "builtins.input",
            lambda _: pytest.fail("must not prompt on non-interactive stdin"),
        )

        with pytest.raises(SystemExit):
            cli._confirm_rebuild(["201_Main"], assume_yes=False)

        assert "--yes" in capsys.readouterr().err


class TestProcessParserFlags:

    def _parse(self, argv):
        return cli._build_parser().parse_args(argv)

    def test_rebuild_defaults_to_false(self):
        args = self._parse(["process", "--targetid", "201"])
        assert args.rebuild is False
        assert args.yes is False

    def test_rebuild_and_yes_parse(self):
        args = self._parse(["process", "--targetid", "201", "--rebuild", "--yes"])
        assert args.rebuild is True
        assert args.yes is True

    def test_rebuild_conflicts_with_fill_gaps(self):
        with pytest.raises(SystemExit):
            self._parse(["process", "--targetid", "201", "--rebuild", "--fill-gaps"])

    def test_rebuild_works_with_all(self):
        args = self._parse(["process", "--all", "--rebuild"])
        assert args.all is True
        assert args.rebuild is True
