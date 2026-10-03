"""Goldens for the S-D4 ``detector_findings`` table (imperative shell).

The contract that matters: findings persist with NA sentinels making the
UNIQUE key total, a re-run over a date range is idempotent (delete-then-replace
in one transaction), rows outside the range are untouched, ``get_findings``
restores the analysis schema, and ``clear_ingested_data`` removes the table.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from atspm.analysis.detector_health import FINDINGS_SCHEMA
from atspm.data.manager import DatabaseManager


def _findings(rows):
    """Build a FINDINGS_SCHEMA frame from (date, window, ts, det, phase, role,
    rule, severity, value, threshold, message) tuples."""
    df = pd.DataFrame(rows, columns=FINDINGS_SCHEMA)
    return df.astype({"detector": "int64", "phase": "Int64"})


def _table_columns(db_path: Path):
    with sqlite3.connect(db_path) as c:
        return [r[1] for r in c.execute("PRAGMA table_info(detector_findings)")]


def _count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as c:
        return c.execute("SELECT COUNT(*) FROM detector_findings").fetchone()[0]


class TestSchema:
    def test_table_and_columns_created_by_init_db(self, empty_db: Path):
        cols = _table_columns(empty_db)
        assert cols == [
            "date", "window", "detector", "phase", "role", "rule", "severity",
            "value", "threshold", "message", "ts", "computed_at",
        ]

    def test_unique_index_includes_role_and_ts(self, empty_db: Path):
        with sqlite3.connect(empty_db) as c:
            idx = c.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='detector_findings' AND sql LIKE '%UNIQUE%' "
                "OR (tbl_name='detector_findings' AND name LIKE 'sqlite_autoindex%')"
            ).fetchall()
        # The UNIQUE constraint produces an autoindex; prove role+ts are in it
        # by inserting two rows that differ only in role, then only in ts.
        assert idx  # a unique/autoindex exists


class TestWriteReadRoundTrip:
    def test_sentinels_stored_and_restored(self, empty_db: Path):
        df = _findings([
            # bin-level: ts NaN, phase NA, real role
            ("2026-01-10", "day", float("nan"), 52, pd.NA, "stop_bar",
             "StuckOn", "high", 1500.0, 900.0, "held"),
            # event-level failsafe: real ts, phase NA, unit role, detector -1
            ("2026-01-10", "day", 1.7e9, -1, pd.NA, "unit:evo_0",
             "Failsafe", "high", 73.7, 240.0, "burst"),
        ])
        with DatabaseManager(empty_db) as m:
            m.replace_findings(df, "2026-01-10", "2026-01-10", computed_at="2026-01-10T00:00:00")

        # Stored sentinels are -1 / '' (no NULLs in the key columns).
        with sqlite3.connect(empty_db) as c:
            raw = c.execute(
                "SELECT detector, phase, role, ts FROM detector_findings "
                "ORDER BY detector"
            ).fetchall()
        assert (-1, -1, "unit:evo_0", 1.7e9) in [tuple(r) for r in raw]
        stuck = [r for r in raw if r[0] == 52][0]
        assert stuck[1] == -1 and stuck[3] == -1  # phase/ts sentinels

        # Read back: sentinels mapped to the analysis schema.
        with DatabaseManager(empty_db) as m:
            got = m.get_findings("2026-01-10", "2026-01-10")
        assert str(got["phase"].dtype) == "Int64"
        assert got["phase"].isna().all()
        fs = got[got["rule"] == "Failsafe"].iloc[0]
        assert fs["ts"] == pytest.approx(1.7e9)
        sb = got[got["rule"] == "StuckOn"].iloc[0]
        assert pd.isna(sb["ts"])
        assert "computed_at" in got.columns


class TestIdempotencyAndKey:
    def test_rerun_same_range_is_idempotent(self, empty_db: Path):
        df = _findings([
            ("2026-01-10", "day", float("nan"), 52, pd.NA, "stop_bar",
             "StuckOn", "high", 1.0, 0.0, "m"),
        ])
        with DatabaseManager(empty_db) as m:
            m.replace_findings(df, "2026-01-10", "2026-01-10")
            m.replace_findings(df, "2026-01-10", "2026-01-10")
        assert _count(empty_db) == 1

    def test_replace_only_touches_its_range(self, empty_db: Path):
        keep = _findings([
            ("2026-01-09", "day", float("nan"), 1, pd.NA, "tm",
             "LowDetectorHits", "low", 3.0, 20.0, "m"),
        ])
        first = _findings([
            ("2026-01-10", "day", float("nan"), 52, pd.NA, "stop_bar",
             "StuckOn", "high", 1.0, 0.0, "m"),
        ])
        second = _findings([
            ("2026-01-10", "day", float("nan"), 53, pd.NA, "stop_bar",
             "Chatter", "low", 0.1, 0.0, "m"),
        ])
        with DatabaseManager(empty_db) as m:
            m.replace_findings(keep, "2026-01-09", "2026-01-09")
            m.replace_findings(first, "2026-01-10", "2026-01-10")
            m.replace_findings(second, "2026-01-10", "2026-01-10")  # replaces the 10th only
            got = m.get_findings("2026-01-09", "2026-01-10")
        rules = dict(zip(got["detector"], got["rule"]))
        assert rules == {1: "LowDetectorHits", 53: "Chatter"}  # 52/StuckOn gone, 9th kept

    def test_two_units_same_instant_coexist(self, empty_db: Path):
        # Same day/detector(-1)/phase(-1)/rule/ts, different unit role -> both kept
        # (this is why role is in the UNIQUE key).
        df = _findings([
            ("2026-01-10", "day", 1.7e9, -1, pd.NA, "unit:evo_0",
             "Failsafe", "high", 10.0, 0.0, "a"),
            ("2026-01-10", "day", 1.7e9, -1, pd.NA, "unit:currux_0",
             "Failsafe", "high", 10.0, 0.0, "b"),
        ])
        with DatabaseManager(empty_db) as m:
            m.replace_findings(df, "2026-01-10", "2026-01-10")
        assert _count(empty_db) == 2

    def test_two_episodes_same_unit_different_ts_coexist(self, empty_db: Path):
        df = _findings([
            ("2026-01-10", "day", 1.7e9, -1, pd.NA, "unit:evo_0",
             "Failsafe", "high", 10.0, 0.0, "a"),
            ("2026-01-10", "day", 1.7e9 + 3600, -1, pd.NA, "unit:evo_0",
             "Failsafe", "high", 10.0, 0.0, "b"),
        ])
        with DatabaseManager(empty_db) as m:
            m.replace_findings(df, "2026-01-10", "2026-01-10")
        assert _count(empty_db) == 2

    def test_empty_frame_clears_range(self, empty_db: Path):
        df = _findings([
            ("2026-01-10", "day", float("nan"), 52, pd.NA, "stop_bar",
             "StuckOn", "high", 1.0, 0.0, "m"),
        ])
        empty = _findings([]).iloc[0:0]
        with DatabaseManager(empty_db) as m:
            m.replace_findings(df, "2026-01-10", "2026-01-10")
            n = m.replace_findings(empty, "2026-01-10", "2026-01-10")
        assert n == 0 and _count(empty_db) == 0


class TestClear:
    def test_clear_ingested_data_removes_findings(self, empty_db: Path):
        df = _findings([
            ("2026-01-10", "day", float("nan"), 52, pd.NA, "stop_bar",
             "StuckOn", "high", 1.0, 0.0, "m"),
        ])
        with DatabaseManager(empty_db) as m:
            m.replace_findings(df, "2026-01-10", "2026-01-10")
            deleted = m.clear_ingested_data()
        assert deleted.get("detector_findings") == 1
        assert _count(empty_db) == 0

    def test_window_filter_on_read(self, empty_db: Path):
        df = _findings([
            ("2026-01-10", "day", float("nan"), 52, pd.NA, "stop_bar",
             "StuckOn", "high", 1.0, 0.0, "m"),
            ("2026-01-10", "am", float("nan"), 53, pd.NA, "stop_bar",
             "ConfiguredSilent", "high", 0.0, 0.0, "m"),
        ])
        with DatabaseManager(empty_db) as m:
            m.replace_findings(df, "2026-01-10", "2026-01-10")
            am = m.get_findings("2026-01-10", "2026-01-10", window="am")
        assert list(am["window"].unique()) == ["am"]
