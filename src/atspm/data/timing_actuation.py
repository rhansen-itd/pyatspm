"""ATSPM Timing and Actuation Engine (Imperative Shell)

Orchestrates the visual detector check / timing-and-actuation plot by querying
events and findings from the SQLite database, resolving configuration, and
delegating intervals, row layout, and rendering to the Functional Core
(:mod:`atspm.analysis.timing_actuation` and :mod:`atspm.plotting.timing_actuation`).

Package Location: src/atspm/data/timing_actuation.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import pandas as pd

from .manager import DatabaseManager, db_timezone
from ..analysis.detector_roles import parse_detector_roles
from ..analysis.detector_health import apply_ignore, filter_min_severity, wd_ignore
from ..analysis.timing_actuation import (
    TIMING_CODES,
    ring_phase_order,
    timing_actuation_intervals,
    timing_actuation_rows,
)
from ..plotting.timing_actuation import plot_timing_actuation
from ..utils.timezone import to_epoch

MAX_WINDOW_S: float = 4 * 3600.0
MAX_NARROWED_WINDOW_S: float = 24 * 3600.0
FETCH_MARGIN_S: float = 900.0

# Accepted --start/--end string formats
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
)
_DATE_FORMAT = "%Y-%m-%d"


class TimingActuationEngine:
    """Queries the database and produces timing-and-actuation figures and intervals.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = TimingActuationEngine(Path("201_data.db"))

        # In-memory results
        results = engine.plot("2026-01-10 08:00", "2026-01-10 08:30")

        # Written to disk
        engine.plot(
            "2026-01-10 08:00", "2026-01-10 08:30",
            phases=[2, 4],
            output_dir=Path("./outputs"),
        )
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize the TimingActuationEngine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g., ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def plot(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        phases: Optional[Iterable[int]] = None,
        detectors: Optional[Iterable[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, object]:
        """Generate a timing-and-actuation plot for the specified window.

        Args:
            start: Window start — ``'YYYY-MM-DD'``, ``'YYYY-MM-DD HH:MM[:SS]'``,
                or a naive local ``datetime``.
            end: Window end (same formats).
            phases: Optional filter to specific phase numbers.
            detectors: Optional filter to specific detector channels.
            output_dir: When provided, write interactive HTML plot to this
                directory and set ``"html"`` in the returned dict to the Path.
                When ``None``, ``"html"`` is ``None``.

        Returns:
            Dict containing:
                - ``"figure"``: Plotly Figure object.
                - ``"rows"``: DataFrame of row layout.
                - ``"intervals"``: DataFrame of state intervals.
                - ``"marks"``: DataFrame of point marks.
                - ``"findings"``: DataFrame of overlaid findings, or None.
                - ``"html"``: Path to written HTML file, or None.

        Raises:
            ValueError: If ``end <= start``, if the window exceeds 4 h
                without phase/detector filters, or exceeds 24 h with filters.
        """
        start_dt, end_dt = self._parse_range(start, end)
        w0 = float(to_epoch(start_dt, self.timezone))
        w1 = float(to_epoch(end_dt, self.timezone))

        if w1 <= w0:
            raise ValueError(
                f"End time ({end_dt}) must be strictly after start time ({start_dt})."
            )

        phase_list = list(phases) if phases is not None else None
        detector_list = list(detectors) if detectors is not None else None
        is_narrowed = bool(phase_list) or bool(detector_list)
        window_duration = w1 - w0

        if not is_narrowed and window_duration > MAX_WINDOW_S:
            raise ValueError(
                f"Window duration ({window_duration:.0f} s) exceeds 4 h limit. "
                "Specify phases or detectors to allow up to 24 h."
            )
        if window_duration > MAX_NARROWED_WINDOW_S:
            raise ValueError(
                f"Window duration ({window_duration:.0f} s) exceeds 24 h limit."
            )

        data_start = w0 - FETCH_MARGIN_S
        data_end = w1 + FETCH_MARGIN_S

        with DatabaseManager(self.db_path) as mgr:
            events_df = mgr.query_events(
                start_time=data_start,
                end_time=data_end,
                event_codes=list(TIMING_CODES),
            )

        config = self._get_config(start_dt)
        roles = parse_detector_roles(config)
        phase_order = ring_phase_order(config)

        intervals_res = timing_actuation_intervals(
            events_df,
            window=(w0, w1),
            data_range=(data_start, data_end),
        )
        intervals = intervals_res["intervals"]
        marks = intervals_res["marks"]

        rows = timing_actuation_rows(
            roles=roles,
            intervals=intervals,
            marks=marks,
            phase_order=phase_order,
            phases=phase_list,
            detectors=detector_list,
        )

        d0 = start_dt.strftime("%Y-%m-%d")
        d1 = max(
            start_dt.date(),
            (end_dt - timedelta(microseconds=1)).date(),
        ).strftime("%Y-%m-%d")

        findings = None
        try:
            with DatabaseManager(self.db_path) as mgr:
                raw_findings = mgr.get_findings(d0, d1)
            if raw_findings is not None and not raw_findings.empty:
                ignored = apply_ignore(raw_findings, wd_ignore(config))
                filtered = filter_min_severity(ignored, "low")
                findings = filtered
        except Exception:
            findings = None

        with DatabaseManager(self.db_path) as mgr:
            metadata = mgr.get_metadata() or {}

        fig = plot_timing_actuation(
            rows=rows,
            intervals=intervals,
            marks=marks,
            window=(w0, w1),
            tz=self.timezone,
            metadata=metadata,
            findings=findings,
        )

        html_path: Optional[Path] = None
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = ""
            if phase_list:
                suffix += "_P" + "_".join(str(p) for p in phase_list)
            if detector_list:
                suffix += "_D" + "_".join(str(d) for d in detector_list)
            filename = (
                f"TimingActuation_{start_dt:%Y_%m_%d_%H%M}"
                f"-{end_dt:%Y_%m_%d_%H%M}{suffix}.html"
            )
            html_path = out_dir / filename
            fig.write_html(str(html_path))
            print(f"Wrote {filename}")

        return {
            "figure": fig,
            "rows": rows,
            "intervals": intervals,
            "marks": marks,
            "findings": findings,
            "html": html_path,
        }

    def _read_timezone(self) -> str:
        """Read the intersection timezone from the database."""
        return db_timezone(self.db_path)

    def _get_config(self, date: datetime) -> Dict[str, Any]:
        """Retrieve the active configuration dict for a given date.

        Args:
            date: Reference datetime (naive local) used to select the
                correct temporal config row.

        Returns:
            Config dict (may be empty if no config has been imported).
        """
        with DatabaseManager(self.db_path) as m:
            config = m.get_config_at_date(date)
        return config or {}

    @staticmethod
    def _parse_range(
        start: Union[str, datetime],
        end: Union[str, datetime],
    ) -> tuple[datetime, datetime]:
        """Coerce date or datetime strings to naive local datetimes.

        Args:
            start: Start date/datetime string or naive datetime.
            end: End date/datetime string or naive datetime.

        Returns:
            Tuple of ``(start_dt, end_dt)`` as naive datetimes.

        Raises:
            ValueError: When a string matches no accepted format.
        """
        def parse(value: Union[str, datetime], is_end: bool) -> datetime:
            if isinstance(value, datetime):
                return value
            for fmt in _DATETIME_FORMATS:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            parsed = datetime.strptime(value, _DATE_FORMAT)
            return parsed + timedelta(days=1) if is_end else parsed

        return parse(start, False), parse(end, True)


def get_timing_actuation(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    phases: Optional[Iterable[int]] = None,
    detectors: Optional[Iterable[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Dict[str, object]:
    """Convenience wrapper around :class:`TimingActuationEngine`.plot.

    Args:
        db_path: Path to the intersection SQLite database.
        start: Period start (local date or datetime).
        end: Period end (local date or datetime).
        phases: Optional filter to specific phase numbers.
        detectors: Optional filter to specific detector channels.
        output_dir: Write HTML plot and return its Path when provided.
        timezone: Override intersection timezone.

    Returns:
        Dict with keys ``figure``, ``rows``, ``intervals``, ``marks``,
        ``findings``, and ``html``.
    """
    return TimingActuationEngine(db_path, timezone=timezone).plot(
        start=start,
        end=end,
        phases=phases,
        detectors=detectors,
        output_dir=output_dir,
    )
