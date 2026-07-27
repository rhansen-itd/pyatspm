"""Centralized timezone resolution utilities."""

import logging
from datetime import datetime
from typing import Optional

import pytz
from pytz import BaseTzInfo

logger = logging.getLogger(__name__)

#: The zone assumed for an intersection whose ``metadata`` table names none.
#: Every "what zone is this intersection?" fallback in the package resolves
#: here, so a database with no metadata cannot be read one way by the reader
#: and another by an engine.  It matches the ``metadata.timezone`` column
#: default in the schema and the ``atspm init`` CLI default.
#:
#: Distinct from :func:`resolve_pytz`'s UTC fallback, which answers a
#: different question — "this zone *name* is missing or unparseable" — and
#: stays UTC so an unusable name degrades to an unambiguous zone rather than
#: silently pretending to be Mountain.
DEFAULT_TIMEZONE = "US/Mountain"


def resolve_pytz(tz_string: Optional[str]) -> BaseTzInfo:
    """Resolve an IANA timezone name to a pytz timezone object.
    
    Falls back to UTC with a warning if the name is missing or unknown.
    
    Args:
        tz_string: IANA timezone (e.g. 'America/Boise', 'US/Mountain').
        
    Returns:
        pytz timezone object (always valid).
    """
    if not tz_string:
        logger.warning("No timezone provided; falling back to UTC.")
        return pytz.utc

    try:
        return pytz.timezone(tz_string)
    except pytz.UnknownTimeZoneError:
        logger.warning(
            f"Unknown timezone '{tz_string}'; falling back to UTC.",
            extra={"timezone": tz_string},
        )
        return pytz.utc


def localize_naive(dt: datetime, tz_string: Optional[str]) -> datetime:
    """Attach the intersection's timezone to a naive datetime.

    Naive datetimes anywhere in this package mean *intersection local wall
    clock*, never the clock of the machine running the code.  Python's own
    default is the opposite — ``datetime.timestamp()`` resolves a naive value
    through the host's zone — so every naive value bound for an epoch
    comparison must pass through here first.

    Args:
        dt: Datetime to localize.  Already-aware values are returned
            unchanged: they carry their own offset and need no help.
        tz_string: IANA timezone (e.g. ``'US/Mountain'``).  Resolved via
            :func:`resolve_pytz`, so an unknown or missing name falls back to
            UTC with a warning rather than raising.

    Returns:
        A timezone-aware datetime.
    """
    if dt.tzinfo is not None:
        return dt
    return resolve_pytz(tz_string).localize(dt)


def to_epoch(dt: datetime, tz_string: Optional[str]) -> float:
    """Convert a datetime to the UTC epoch float the ``events`` table stores.

    The single conversion point between wall-clock arguments and the REAL
    ``timestamp`` / ``cycle_start`` columns.  Calling ``dt.timestamp()``
    directly on a naive value is the bug this exists to prevent: it silently
    reads the host machine's zone, so a database from another zone comes back
    shifted by the offset — and a shifted window is indistinguishable from a
    short one.

    Args:
        dt: Window bound.  Naive values are read as *tz_string* local; aware
            values keep their own offset.
        tz_string: IANA timezone used to interpret naive values.

    Returns:
        UTC epoch seconds.
    """
    return localize_naive(dt, tz_string).timestamp()
