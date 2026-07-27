"""Window bounds are read in the intersection's zone, not the host machine's.

Target: src/atspm/data/reader.py — _bounds_to_epoch and the three public
entry points that use it (get_events_with_cycles_df, get_coordination_data,
check_data_quality), plus get_date_range.

The regression these guard: naive bounds used to reach the SQL query through
``datetime.timestamp()``, which resolves a naive value through the *host's*
zone. Reading a US/Mountain database from a UTC or Pacific machine therefore
returned a window shifted by the offset — and a shifted window is
indistinguishable from a short one. Each test here runs the identical query
under several fake host clocks and asserts the results do not move.

``TZ`` + ``time.tzset()`` is the only way to change what Python considers
"local"; it is process-global, so the fixture always restores it.
"""

import os
import time
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path

import pytest
import pytz

from atspm.data.manager import DatabaseManager
from atspm.data.reader import (
    check_data_quality,
    get_date_range,
    get_events_with_cycles_df,
)

from ..conftest import seed_events

#: The intersection's zone for every test in this module.
INT_TZ = "US/Mountain"

#: Host clocks to sweep. Includes one that matches INT_TZ (the only one the
#: old code got right) and one on the far side of UTC, so a sign error in the
#: conversion cannot pass by cancelling out.
HOST_ZONES = ["UTC", "America/Los_Angeles", "America/New_York", "America/Denver",
              "Australia/Sydney"]

#: Local wall-clock midnight of the seeded day, and the events hung off it.
LOCAL_DAY = datetime(2026, 1, 5)
DAY_START_EPOCH = pytz.timezone(INT_TZ).localize(LOCAL_DAY).timestamp()


@pytest.fixture
def host_zone():
    """Set the process's idea of local time, restoring it afterwards."""
    original = os.environ.get("TZ")

    def _set(zone: str) -> None:
        os.environ["TZ"] = zone
        time.tzset()

    yield _set

    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


@pytest.fixture
def seeded_db(empty_db: Path) -> Path:
    """A DB in US/Mountain with one event per hour through a local day.

    One event per hour means a window off by even a single hour returns a
    different row count, so an offset bug cannot hide behind a coarse
    assertion.
    """
    with DatabaseManager(empty_db) as manager:
        manager.set_metadata(intersection_id="999", timezone=INT_TZ)

    seed_events(
        empty_db,
        events=[(DAY_START_EPOCH + h * 3600.0, 82, 1) for h in range(24)],
    )
    return empty_db


class TestNaiveBoundsAreIntersectionLocal:

    @pytest.mark.parametrize("zone", HOST_ZONES)
    def test_naive_window_returns_the_same_rows_on_any_host(
        self, seeded_db: Path, host_zone, zone: str
    ):
        host_zone(zone)

        df = get_events_with_cycles_df(
            seeded_db, LOCAL_DAY, LOCAL_DAY + timedelta(days=1),
            timezone=INT_TZ,
        )

        assert len(df) == 24
        assert df["timestamp"].min().hour == 0
        assert df["timestamp"].max().hour == 23

    @pytest.mark.parametrize("zone", HOST_ZONES)
    def test_metadata_zone_is_used_when_no_timezone_argument(
        self, seeded_db: Path, host_zone, zone: str
    ):
        """A caller who names no zone still gets the intersection's, not the host's."""
        host_zone(zone)

        df = get_events_with_cycles_df(
            seeded_db, LOCAL_DAY, LOCAL_DAY + timedelta(days=1),
        )

        assert len(df) == 24

    @pytest.mark.parametrize("zone", HOST_ZONES)
    def test_check_data_quality_counts_the_same_window(
        self, seeded_db: Path, host_zone, zone: str
    ):
        host_zone(zone)

        result = check_data_quality(
            seeded_db, LOCAL_DAY, LOCAL_DAY + timedelta(days=1), timezone=INT_TZ,
        )

        assert result["event_count"] == 24

    @pytest.mark.parametrize("zone", HOST_ZONES)
    def test_date_range_reports_local_midnight_not_host_midnight(
        self, seeded_db: Path, host_zone, zone: str
    ):
        host_zone(zone)

        span = get_date_range(seeded_db)

        assert span is not None
        # The first seeded event sits exactly on local midnight; a host-clock
        # read would report some other hour (and possibly the day before).
        assert span["start"].hour == 0
        assert span["start"].date() == LOCAL_DAY.date()


class TestAwareBoundsKeepTheirOwnOffset:

    @pytest.mark.parametrize("zone", HOST_ZONES)
    def test_aware_bounds_are_unaffected_by_the_host(
        self, seeded_db: Path, host_zone, zone: str
    ):
        host_zone(zone)
        tz = pytz.timezone(INT_TZ)

        df = get_events_with_cycles_df(
            seeded_db,
            tz.localize(LOCAL_DAY),
            tz.localize(LOCAL_DAY + timedelta(days=1)),
            timezone=INT_TZ,
        )

        assert len(df) == 24

    def test_utc_aware_bounds_select_the_same_instants(self, seeded_db: Path):
        """The same window expressed in UTC must return the same rows."""
        start_utc = datetime.fromtimestamp(DAY_START_EPOCH, dt_timezone.utc)

        df = get_events_with_cycles_df(
            seeded_db, start_utc, start_utc + timedelta(days=1), timezone=INT_TZ,
        )

        assert len(df) == 24

    def test_an_explicit_timezone_does_not_re_shift_aware_bounds(
        self, seeded_db: Path
    ):
        """A naming mismatch must not double-apply an offset."""
        start_utc = datetime.fromtimestamp(DAY_START_EPOCH, dt_timezone.utc)

        as_utc_arg = get_events_with_cycles_df(
            seeded_db, start_utc, start_utc + timedelta(days=1), timezone="UTC",
        )
        as_local_arg = get_events_with_cycles_df(
            seeded_db, start_utc, start_utc + timedelta(days=1), timezone=INT_TZ,
        )

        assert len(as_utc_arg) == len(as_local_arg) == 24


class TestDaylightSavingBoundary:

    def test_spring_forward_day_is_23_hours_long(self, empty_db: Path):
        """US/Mountain 2026-03-08 has 23 hours; a fixed-offset conversion
        would fetch 24 and pull in an hour of the next day."""
        with DatabaseManager(empty_db) as manager:
            manager.set_metadata(intersection_id="999", timezone=INT_TZ)

        tz = pytz.timezone(INT_TZ)
        dst_day = datetime(2026, 3, 8)
        day_start = tz.localize(dst_day).timestamp()
        next_start = tz.localize(dst_day + timedelta(days=1)).timestamp()

        assert next_start - day_start == 23 * 3600

        # One event every hour of real elapsed time, plus one just inside the
        # following day that must not be picked up.
        seed_events(
            empty_db,
            events=[(day_start + h * 3600.0, 82, 1) for h in range(23)]
                   + [(next_start + 60.0, 82, 1)],
        )

        df = get_events_with_cycles_df(
            empty_db, dst_day, dst_day + timedelta(days=1), timezone=INT_TZ,
        )

        assert len(df) == 23
