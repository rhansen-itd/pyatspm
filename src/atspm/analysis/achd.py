"""
ACHD Event-CSV Parser (Functional Core)

Pure transformations for the ACHD high-resolution event export format, the
CSV analogue of the ``.datZ`` binary log handled by :mod:`decoders`.  Like
that module, this one performs **no I/O**: it accepts already-read text /
DataFrames and returns standardised event DataFrames.  The imperative shell
(:mod:`atspm.data.achd_ingestion`) owns file reading, the database, and gap
insertion.

ACHD export layout
==================
Each ``{id}_Events_{YYYYMMDD}T0000.csv`` begins with four metadata lines::

    Signal,271 - Eagle Rd Ustick,
    Start time,"Sunday, 01 September 2024 00:00:00",
    End time,"Sunday, 15 September 2024 23:59:59",
    Total Events,"1,423,589",

followed by the header row and the event rows::

    Event Time, Event Code, Event Description, Event Parameter,
    09/12/24 00:00:00.000,81,Veh Det Off,9,

``Start``/``End time`` describe the *requested* export window, not the actual
data extent (the first real event above is nine days after the requested
start).  Spans are therefore derived from real event timestamps in the shell;
the header window is used only for file-level continuity decisions.

Event times are intersection-local wall-clock in the ``%m/%d/%y
%H:%M:%S.%f`` format.  They are localised and converted to UTC epoch seconds,
matching the ``timestamp`` basis of every other ingestion path.  The
non-existent spring-forward hour is shifted forward; the repeated fall-back
hour is taken as standard time (its two passes are indistinguishable once the
export is sorted by wall clock).  A continuous recording across either
transition keeps a monotonic UTC timebase and loses no rows — see
:func:`achd_events_from_frame`.

Package Location: src/atspm/analysis/achd.py
"""

import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from ..utils.timezone import resolve_pytz

# Number of metadata lines before the ``Event Time, …`` header row.
ACHD_HEADER_ROWS = 4

# Raw CSV column names (stripped of the leading spaces ACHD emits).
CSV_COL_TIME = "Event Time"
CSV_COL_CODE = "Event Code"
CSV_COL_PARAM = "Event Parameter"

# Event-time parse format, e.g. ``09/12/24 00:00:00.000``.
ACHD_TIME_FORMAT = "%m/%d/%y %H:%M:%S.%f"

# Metadata-line datetime format, e.g. ``Sunday, 01 September 2024 00:00:00``.
_META_TIME_FORMAT = "%A, %d %B %Y %H:%M:%S"


class AchdDecodingError(Exception):
    """Raised when an ACHD event CSV cannot be parsed into events.

    Mirrors :class:`atspm.analysis.decoders.DatZDecodingError` so the shell
    can treat a malformed ACHD file the same way it treats a bad ``.datZ``:
    log it and skip, without aborting the batch.
    """


@dataclass(frozen=True)
class AchdHeader:
    """Parsed ACHD metadata header.

    Attributes:
        signal_id:    Numeric signal identifier as a string (e.g. ``'271'``).
        signal_name:  Human-readable signal name (e.g. ``'Eagle Rd Ustick'``).
        start:        Requested export window start (naive local datetime), or
                      ``None`` when the line is absent/unparseable.
        end:          Requested export window end (naive local datetime), or
                      ``None``.
        total_events: Reported event count, or ``None`` when absent.
    """

    signal_id: Optional[str]
    signal_name: Optional[str]
    start: Optional[datetime]
    end: Optional[datetime]
    total_events: Optional[int]


def parse_achd_header(header_text: str) -> AchdHeader:
    """Parse the four-line ACHD metadata header from its raw text.

    Pure function: the shell reads the first :data:`ACHD_HEADER_ROWS` lines of
    a file and passes them here.  Parsing is tolerant — any field that is
    missing or malformed comes back as ``None`` rather than raising, because
    the event rows (not the header) are the authority on data extent.

    Args:
        header_text: The leading lines of an ACHD event CSV (at least the
            ``Signal`` / ``Start time`` / ``End time`` / ``Total Events``
            rows).  Extra trailing lines are ignored.

    Returns:
        An :class:`AchdHeader`.  ``signal_id`` is the leading numeric token of
        the ``Signal`` field; ``signal_name`` is the remainder with any
        ``'<id> - '`` prefix stripped.

    Example:
        >>> h = parse_achd_header('Signal,271 - Eagle Rd Ustick,\\n'
        ...                       'Start time,"Sunday, 01 September 2024 00:00:00",')
        >>> h.signal_id, h.signal_name
        ('271', 'Eagle Rd Ustick')
    """
    rows = list(csv_rows(header_text))
    fields = {}
    for row in rows:
        if len(row) >= 2 and row[0].strip():
            fields[row[0].strip().lower()] = row[1].strip()

    signal_raw = fields.get("signal", "")
    signal_id: Optional[str] = None
    signal_name: Optional[str] = None
    if signal_raw:
        m = re.match(r"\s*(\d+)\s*-\s*(.*)$", signal_raw)
        if m:
            signal_id, signal_name = m.group(1), m.group(2).strip()
        else:
            signal_name = signal_raw

    total: Optional[int] = None
    if fields.get("total events"):
        try:
            total = int(fields["total events"].replace(",", ""))
        except ValueError:
            total = None

    return AchdHeader(
        signal_id=signal_id,
        signal_name=signal_name,
        start=_parse_meta_datetime(fields.get("start time")),
        end=_parse_meta_datetime(fields.get("end time")),
        total_events=total,
    )


def _parse_meta_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse a metadata datetime like ``Sunday, 01 September 2024 00:00:00``."""
    if not value:
        return None
    try:
        return datetime.strptime(value, _META_TIME_FORMAT)
    except ValueError:
        return None


def csv_rows(text: str):
    """Yield CSV rows from *text* using the stdlib reader (quote-aware).

    Factored out so header parsing handles the quoted, comma-containing
    datetime fields the same way the row reader does.
    """
    import csv as _csv

    yield from _csv.reader(io.StringIO(text))


def achd_events_from_frame(raw: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Transform a raw ACHD event frame into the standard events schema.

    Pure DataFrame → DataFrame transform.  Input is the CSV body as read by
    the shell (``pd.read_csv(path, skiprows=ACHD_HEADER_ROWS)``); output
    matches the ``events`` table order ``[timestamp, event_code, parameter]``
    with a UTC-epoch ``timestamp`` (REAL), ready for insertion.

    Processing:
        1. Strip whitespace from column names (ACHD pads with a leading space)
           and select the three columns of interest.
        2. Parse ``Event Time`` and localise it to *timezone*, resolving the
           ambiguous fall-back hour from event order and shifting the
           non-existent spring-forward hour forward, then convert to UTC epoch
           seconds.  Vectorised throughout — no row iteration.
        3. Coerce ``Event Code`` / ``Event Parameter`` to integers, dropping
           rows where either is missing or non-numeric.
        4. Drop exact duplicate ``(timestamp, event_code, parameter)`` triples
           (overlapping exports repeat rows; the DB's UNIQUE/IGNORE constraint
           is the backstop, but de-duping here keeps span row-counts honest).

    Rows are returned in their original file order (ACHD exports are
    time-sorted); the shell sorts and inserts gap markers as needed.

    Args:
        raw: Raw ACHD event rows with ``Event Time`` / ``Event Code`` /
            ``Event Parameter`` columns (surrounding whitespace tolerated).
        timezone: IANA timezone of the intersection's wall clock
            (e.g. ``'US/Mountain'``).

    Returns:
        DataFrame with columns ``[timestamp, event_code, parameter]`` and
        dtypes ``float64, int64, int64``.  Empty input yields an empty frame
        with those columns.

    Raises:
        AchdDecodingError: If the required columns are absent, or the local
            timezone cannot localise the timestamps (e.g. an unresolvable
            ambiguous time).
    """
    cols = {str(c).strip(): c for c in raw.columns}
    missing = [c for c in (CSV_COL_TIME, CSV_COL_CODE, CSV_COL_PARAM) if c not in cols]
    if missing:
        raise AchdDecodingError(
            f"ACHD frame missing required column(s): {', '.join(missing)}"
        )

    empty = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="float64"),
            "event_code": pd.Series(dtype="int64"),
            "parameter": pd.Series(dtype="int64"),
        }
    )
    if raw.empty:
        return empty

    df = pd.DataFrame(
        {
            "timestamp": raw[cols[CSV_COL_TIME]],
            "event_code": pd.to_numeric(raw[cols[CSV_COL_CODE]], errors="coerce"),
            "parameter": pd.to_numeric(raw[cols[CSV_COL_PARAM]], errors="coerce"),
        }
    )
    df = df.dropna(subset=["event_code", "parameter"])
    if df.empty:
        return empty

    naive = pd.to_datetime(df["timestamp"], format=ACHD_TIME_FORMAT, errors="coerce")
    bad = naive.isna()
    if bad.any():
        naive = naive[~bad]
        df = df.loc[naive.index]
    if naive.empty:
        return empty

    tz = resolve_pytz(timezone)
    try:
        # DST edges: the non-existent spring-forward hour is shifted forward.
        # The repeated fall-back hour is ambiguous and, because ACHD exports
        # are sorted by local wall clock, its two passes interleave — so their
        # original order (and thus which is DST) is unrecoverable, and
        # ``ambiguous='infer'`` raises on the apparent back-and-forth. We pick
        # standard time for the whole repeated hour (``ambiguous=False``),
        # matching the prior pipeline's ``pytz.localize`` default: it never
        # drops data and keeps the UTC timebase monotonic. The cost is a <=1 h
        # mislabel of up to one hour of data, once a year, around 01:00 local
        # (the lowest-traffic hour) — negligible for ATSPM measures.
        localized = naive.dt.tz_localize(
            tz, ambiguous=False, nonexistent="shift_forward"
        )
    except Exception as exc:  # noqa: BLE001 - surface as a decoding failure
        raise AchdDecodingError(
            f"Failed to localise ACHD timestamps to {timezone!r}: {exc}"
        ) from exc

    # tz-aware wall clock → UTC epoch seconds.  Differencing against the
    # UTC-aware epoch keeps this correct and version-stable (Series.view was
    # removed in pandas 3.0).
    epoch = (localized - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    out = pd.DataFrame(
        {
            "timestamp": epoch.to_numpy(),
            "event_code": df["event_code"].astype("int64").to_numpy(),
            "parameter": df["parameter"].astype("int64").to_numpy(),
        }
    )
    out = out.drop_duplicates(subset=["timestamp", "event_code", "parameter"])
    return out.reset_index(drop=True)
