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

from atspm.utils.timezone import resolve_pytz

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
