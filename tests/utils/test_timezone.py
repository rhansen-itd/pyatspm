"""Adversarial edge-case tests for timezone resolution (Functional Core).

Target: src/atspm/utils/timezone.py — resolve_pytz.
Pure function (logging aside): string in, pytz timezone out. The
documented fallback is pytz.utc with a logged warning — asserted here
via caplog, never invented.
"""

import logging
from datetime import datetime, timedelta

import pytest
import pytz

from atspm.utils.timezone import (
    DEFAULT_TIMEZONE,
    localize_naive,
    resolve_pytz,
    to_epoch,
)

LOGGER_NAME = "atspm.utils.timezone"


class TestResolvePytzValidNames:

    def test_valid_iana_name_resolves_to_matching_zone(self):
        tz = resolve_pytz("America/Boise")
        assert isinstance(tz, pytz.tzinfo.BaseTzInfo)
        assert tz.zone == "America/Boise"

    def test_valid_name_logs_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            resolve_pytz("US/Mountain")
        assert caplog.records == []

    def test_lowercase_name_resolves_case_insensitively(self):
        # pytz (2026.2) resolves names case-insensitively; a lowercase
        # spelling is NOT garbage — it canonicalizes, no fallback fires.
        tz = resolve_pytz("america/boise")
        assert tz.zone == "America/Boise"

    def test_fixed_offset_etc_zone_resolves_with_inverted_posix_sign(self):
        # Etc/GMT+7 is a valid IANA fixed-offset zone whose POSIX-style
        # sign is inverted: it means UTC-7, year-round, no DST.
        tz = resolve_pytz("Etc/GMT+7")
        assert tz.zone == "Etc/GMT+7"
        assert tz.utcoffset(datetime(2026, 1, 15)) == timedelta(hours=-7)
        assert tz.utcoffset(datetime(2026, 7, 15)) == timedelta(hours=-7)


class TestResolvePytzFallback:

    @pytest.mark.parametrize("bad", [None, ""])
    def test_missing_value_falls_back_to_utc_with_warning(self, bad, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            tz = resolve_pytz(bad)
        assert tz is pytz.utc
        assert len(caplog.records) == 1
        assert "No timezone provided" in caplog.records[0].message

    @pytest.mark.parametrize(
        "garbage",
        [
            "Not/AZone",
            "  ",              # truthy whitespace reaches pytz and fails there
            "UTC-07:00",       # raw offset strings are not IANA names
            "+07:00",
        ],
    )
    def test_unknown_zone_falls_back_to_utc_with_warning(self, garbage, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            tz = resolve_pytz(garbage)
        assert tz is pytz.utc
        assert len(caplog.records) == 1
        assert f"Unknown timezone '{garbage}'" in caplog.records[0].message

    def test_fallback_result_is_usable_for_localization(self):
        tz = resolve_pytz("definitely-not-a-zone")
        aware = tz.localize(datetime(2026, 7, 8, 12, 0))
        assert aware.utcoffset() == timedelta(0)


class TestResolvePytzDstAwareness:

    def test_resolved_zone_carries_dst_transitions(self):
        # resolve_pytz itself is name resolution only, but the object it
        # returns must be a full IANA zone, not a fixed offset: Boise is
        # UTC-7 (MST) in January and UTC-6 (MDT) in July.
        tz = resolve_pytz("America/Boise")
        assert tz.localize(datetime(2026, 1, 15, 12)).utcoffset() == timedelta(hours=-7)
        assert tz.localize(datetime(2026, 7, 15, 12)).utcoffset() == timedelta(hours=-6)

    def test_resolved_zone_rejects_nonexistent_spring_forward_time(self):
        # 2026-03-08 02:30 does not exist in America/Boise (clocks jump
        # 02:00 -> 03:00). A strict localize must raise, proving the zone
        # has real transition data.
        tz = resolve_pytz("America/Boise")
        with pytest.raises(pytz.exceptions.NonExistentTimeError):
            tz.localize(datetime(2026, 3, 8, 2, 30), is_dst=None)


class TestLocalizeNaive:

    def test_naive_takes_the_named_zone(self):
        aware = localize_naive(datetime(2026, 1, 15, 12), "America/Boise")
        assert aware.utcoffset() == timedelta(hours=-7)

    def test_aware_is_returned_untouched(self):
        original = pytz.timezone("America/New_York").localize(datetime(2026, 1, 15, 12))
        assert localize_naive(original, "America/Boise") is original

    def test_missing_zone_falls_back_to_utc(self):
        # resolve_pytz's documented fallback, inherited rather than re-invented.
        assert localize_naive(datetime(2026, 1, 15, 12), None).utcoffset() == timedelta(0)


class TestToEpoch:

    def test_naive_is_read_in_the_named_zone_not_the_host(self):
        # The whole point: this value must not depend on the machine's TZ.
        assert to_epoch(datetime(2026, 1, 15, 12), "America/Boise") == (
            pytz.timezone("America/Boise")
            .localize(datetime(2026, 1, 15, 12)).timestamp()
        )

    def test_aware_keeps_its_own_offset(self):
        aware = pytz.utc.localize(datetime(2026, 1, 15, 12))
        assert to_epoch(aware, "America/Boise") == aware.timestamp()

    def test_the_same_instant_in_two_zones_gives_one_epoch(self):
        boise = to_epoch(datetime(2026, 1, 15, 12), "America/Boise")
        utc = to_epoch(datetime(2026, 1, 15, 19), "UTC")
        assert boise == utc

    def test_dst_offset_is_applied_per_date(self):
        # A fixed-offset shortcut would make these differ by exactly 24h.
        winter = to_epoch(datetime(2026, 1, 15, 12), "America/Boise")
        summer = to_epoch(datetime(2026, 7, 15, 12), "America/Boise")
        assert (summer - winter) % 3600 == 0
        assert summer - winter != (datetime(2026, 7, 15) - datetime(2026, 1, 15)).total_seconds()


class TestDefaultTimezoneIsSingleSourced:
    """Every "what zone is this intersection?" fallback must give one answer.

    The regression: reader._resolve_timezone fell back to 'UTC' while the
    engines fell back to 'US/Mountain', so the same database was read in two
    different zones depending on which entry point you used.

    A *dropped* metadata table was never the divergent case — ``get_metadata``
    catches that itself and hands back the default. The two that did diverge
    are pinned below: a metadata row whose ``timezone`` is blank, and a
    database file that cannot be opened at all.
    """

    def _all_resolvers(self, db):
        """Every entry point's answer for *db*, keyed by name."""
        from atspm.data.aog import AogEngine
        from atspm.data.counts import CountEngine
        from atspm.data.critical import CriticalMovementEngine
        from atspm.data.flow import FlowRateEngine
        from atspm.data.manager import db_timezone
        from atspm.data.phases import PhaseEngine
        from atspm.data.reader import _resolve_timezone

        return {
            "db_timezone": db_timezone(db),
            "reader": _resolve_timezone(db),
            "counts": CountEngine(db).timezone,
            "phases": PhaseEngine(db).timezone,
            "aog": AogEngine(db).timezone,
            "flow": FlowRateEngine(db).timezone,
            "critical": CriticalMovementEngine(db).timezone,
        }

    def test_blank_timezone_column_resolves_the_same_everywhere(self, tmp_path):
        from atspm.data.manager import DatabaseManager

        db = tmp_path / "blank_tz.db"
        with DatabaseManager(db) as m:
            m.init_db()
            m.conn.execute(
                "INSERT INTO metadata (lock_id, timezone) VALUES (1, '')"
            )
            m.conn.commit()

        answers = self._all_resolvers(db)
        assert set(answers.values()) == {DEFAULT_TIMEZONE}, answers

    def test_unreadable_database_resolves_the_same_everywhere(self, tmp_path):
        db = tmp_path / "corrupt.db"
        db.write_text("this is not a sqlite database")

        answers = self._all_resolvers(db)
        assert set(answers.values()) == {DEFAULT_TIMEZONE}, answers

    def test_dropped_metadata_table_resolves_the_same_everywhere(self, tmp_path):
        from atspm.data.manager import DatabaseManager

        db = tmp_path / "no_metadata.db"
        with DatabaseManager(db) as m:
            m.init_db()
            m.conn.execute("DROP TABLE metadata")
            m.conn.commit()

        answers = self._all_resolvers(db)
        assert set(answers.values()) == {DEFAULT_TIMEZONE}, answers

    def test_a_recorded_zone_still_wins_over_the_default(self, tmp_path):
        from atspm.data.counts import CountEngine
        from atspm.data.manager import DatabaseManager, db_timezone
        from atspm.data.reader import _resolve_timezone

        db = tmp_path / "recorded.db"
        with DatabaseManager(db) as m:
            m.init_db()
            m.set_metadata(intersection_id="1", timezone="America/New_York")

        assert db_timezone(db) == "America/New_York"
        assert _resolve_timezone(db) == "America/New_York"
        assert CountEngine(db).timezone == "America/New_York"

    def test_an_explicit_override_wins_over_both(self, tmp_path):
        from atspm.data.manager import DatabaseManager, db_timezone

        db = tmp_path / "override.db"
        with DatabaseManager(db) as m:
            m.init_db()
            m.set_metadata(intersection_id="1", timezone="America/New_York")

        assert db_timezone(db, "Australia/Sydney") == "Australia/Sydney"

    def test_the_schema_column_default_matches_the_constant(self, tmp_path):
        """The DDL default and the Python fallback are one value, not two."""
        from atspm.data.manager import DatabaseManager

        db = tmp_path / "schema.db"
        with DatabaseManager(db) as m:
            m.init_db()
            cols = m.conn.execute("PRAGMA table_info(metadata)").fetchall()

        tz_default = next(c[4] for c in cols if c[1] == "timezone")
        assert tz_default == f"'{DEFAULT_TIMEZONE}'"
