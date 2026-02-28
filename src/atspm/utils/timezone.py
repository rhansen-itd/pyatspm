"""Centralized timezone resolution utilities."""

import logging
from typing import Optional

import pytz
from pytz import BaseTzInfo

logger = logging.getLogger(__name__)


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