"""
ATSPM Report Generator (Imperative Shell)

Thin orchestration layer: resolves dates -> epoch bounds, calls reader.py
to fetch DataFrames, calls plotting functions to build figures, writes HTML.

No SQL lives here.  All data access goes through src/atspm/data/reader.py
(or domain-specific engines such as DetectorEngine).

Package Location: src/atspm/reports/generators.py

Usage::

    from pathlib import Path
    from atspm.reports.generators import PlotGenerator

    gen = PlotGenerator(
        db_path=Path("2068_data.db"),
        output_dir=Path("reports/2068"),
    )
    gen.generate_for_date("2025-06-15")
    # Writes:
    #   reports/2068/2025-06-15/Phase_Termination.html
    #   reports/2068/2025-06-15/Coordination_Split.html
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pytz

from ..data import reader
from ..plotting.termination import plot_termination
from ..plotting.coordination import plot_coordination

# ATSPM event codes needed for the termination plot
_TERM_CODES = [4, 5, 6, 21, 45, 105]


class PlotGenerator:
    """
    Generates and saves standard ATSPM HTML plots for a single intersection.

    Responsibilities
    ----------------
    - Resolve a local date string to UTC epoch bounds using the
      intersection's timezone from metadata.
    - Delegate all DB access to ``reader.py`` (standard plots) or to
      domain-specific engines (e.g. ``DetectorEngine``).
    - Call pure plotting functions from the functional core.
    - Write the resulting Plotly figures to HTML files.

    Args:
        db_path: Path to the intersection's SQLite database.
        output_dir: Root directory for report output.  A sub-directory named
            ``YYYY-MM-DD`` is created inside it for each date's reports.
    """

    def __init__(self, db_path: Path, output_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self._timezone: Optional[str] = None   # lazily resolved
        self._metadata: Optional[dict] = None  # lazily resolved

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_for_date(self, date_str: str) -> None:
        """
        Generate and save all standard plots for one local calendar date.

        Creates ``{output_dir}/{date_str}/`` if it does not exist.
        Saves:
        - ``Phase_Termination.html``
        - ``Coordination_Split.html``

        Errors in individual plots are caught and printed so that a failure
        in one plot does not prevent the other from being saved.

        Args:
            date_str: Local date in ``'YYYY-MM-DD'`` format.

        Raises:
            ValueError: If *date_str* is not in the expected format.
        """
        try:
            local_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError(
                f"Invalid date format '{date_str}'. Use 'YYYY-MM-DD'."
            )

        date_dir = self.output_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._get_metadata()
        tz_str = self._get_timezone()

        # Convert local date -> UTC-bounded datetimes for reader calls
        start_dt, end_dt = _local_date_to_utc_datetimes(local_date, tz_str)

        print(f"[{date_str}] Generating reports...")

        # ---- Termination plot ----
        try:
            self._generate_termination(
                date_str, start_dt, end_dt, metadata, date_dir, tz_str
            )
        except Exception as exc:
            print(f"[{date_str}] Termination plot FAILED: {exc}")

        # ---- Coordination plot ----
        try:
            self._generate_coordination(
                date_str, start_dt, end_dt, metadata, date_dir, tz_str
            )
        except Exception as exc:
            print(f"[{date_str}] Coordination plot FAILED: {exc}")

    def generate_date_range(self, start_date: str, end_date: str) -> None:
        """
        Generate reports for every local date in ``[start_date, end_date]``.

        Args:
            start_date: Inclusive start date in ``'YYYY-MM-DD'`` format.
            end_date: Inclusive end date in ``'YYYY-MM-DD'`` format.
        """
        from datetime import timedelta

        try:
            d_start = datetime.strptime(start_date, '%Y-%m-%d').date()
            d_end   = datetime.strptime(end_date,   '%Y-%m-%d').date()
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {exc}") from exc

        current = d_start
        while current <= d_end:
            self.generate_for_date(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

    # ------------------------------------------------------------------
    # Private plot generators
    # ------------------------------------------------------------------

    def _generate_termination(
        self,
        date_str: str,
        start_dt: datetime,
        end_dt: datetime,
        metadata: dict,
        date_dir: Path,
        tz_str: str,
    ) -> None:
        """
        Fetch termination events, build the figure, and write HTML.

        Args:
            date_str: Human-readable date (for log messages).
            start_dt: UTC-aware window start.
            end_dt: UTC-aware window end.
            metadata: Intersection metadata dict.
            date_dir: Output directory for this date.
            tz_str: Timezone string for timestamp conversion.
        """
        # Changed: Updated function call to get_events_with_cycles_df
        df_events = reader.get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_dt,
            end=end_dt,
            event_codes=_TERM_CODES,
            timezone=tz_str,
        )

        if df_events.empty:
            print(f"[{date_str}] Termination: no events - skipping")
            return

        fig = plot_termination(df_events=df_events, metadata=metadata)

        out_path = date_dir / 'Phase_Termination.html'
        fig.write_html(str(out_path))
        print(f"[{date_str}] Termination saved -> {out_path}")

    def _generate_coordination(
        self,
        date_str: str,
        start_dt: datetime,
        end_dt: datetime,
        metadata: dict,
        date_dir: Path,
        tz_str: str,
    ) -> None:
        """
        Fetch coordination data, build the figure, and write HTML.
        """
        df_cycles, df_signal, df_det = reader.get_coordination_data(
            db_path=self.db_path,
            start=start_dt,
            end=end_dt,
            timezone=tz_str,
        )

        if df_cycles.empty:
            print(f"[{date_str}] Coordination: no cycles - skipping")
            return

        if df_signal.empty:
            print(f"[{date_str}] Coordination: no signal events - skipping")
            return

        det_config = None
        if not df_det.empty:
            det_config = reader.get_det_config(
                db_path=self.db_path,
                date=start_dt,
            )
            if not det_config:
                det_config = None
                df_det = None
        else:
            df_det = None

        fig = plot_coordination(
            df_cycles=df_cycles,
            df_signal=df_signal,
            metadata=metadata,
            df_det=df_det,
            det_config=det_config,
        )

        out_path = date_dir / 'Coordination_Split.html'
        fig.write_html(str(out_path))
        print(f"[{date_str}] Coordination saved -> {out_path}")

    def _generate_detector_comparison(
        self,
        start_dt: datetime,
        end_dt: datetime,
        phases: Optional[List[int]] = None,
        lag_threshold_sec: float = 2.0,
    ) -> None:
        """Fetch detector data, build the comparison figure, and write HTML."""
        from ..data.detectors import DetectorEngine
        from ..plotting.detectors import plot_detector_comparison

        tz_str   = self._get_timezone()
        metadata = self._get_metadata()

        local_start = start_dt.astimezone(pytz.timezone(tz_str))
        date_str    = local_start.strftime("%Y-%m-%d")

        date_dir = self.output_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        engine = DetectorEngine(db_path=self.db_path, timezone=tz_str)

        try:
            events_df, anomalies_df, filtered_pairs = engine.get_plot_data(
                start=_strip_tz(start_dt),
                end=_strip_tz(end_dt),
                phases=phases,
                lag_threshold_sec=lag_threshold_sec,
            )
        except ValueError as exc:
            print(f"  [{date_str}] Detector Comparison: {exc} - skipping")
            return

        if not filtered_pairs:
            print(f"  [{date_str}] Detector Comparison: no detector pairs configured - skipping")
            return

        if events_df.empty:
            print(f"  [{date_str}] Detector Comparison: no detector events in window - skipping")
            return

        fig = plot_detector_comparison(
            events_df=events_df,
            anomalies_df=anomalies_df,
            detector_pairs=filtered_pairs,
            metadata=metadata,
        )

        if phases:
            phase_tag = "_Ph" + "_".join(str(p) for p in sorted(phases))
        else:
            phase_tag = ""
        filename = f"Detector_Comparison_{date_str}{phase_tag}.html"

        out_path = date_dir / filename
        fig.write_html(str(out_path))
        print(f"  [{date_str}] Detector Comparison saved -> {out_path}")

    # ------------------------------------------------------------------
    # Lazy metadata / timezone resolution
    # ------------------------------------------------------------------

    def _get_metadata(self) -> dict:
        """Fetch and cache intersection metadata from the database."""
        if self._metadata is None:
            from ..data.manager import DatabaseManager
            try:
                with DatabaseManager(self.db_path) as mgr:
                    self._metadata = mgr.get_metadata()
            except Exception:
                self._metadata = {}
            if not self._metadata:
                self._metadata = {'intersection_name': self.db_path.stem}
        return self._metadata

    def _get_timezone(self) -> str:
        """Return the intersection timezone string, reading from metadata once."""
        if self._timezone is None:
            meta = self._get_metadata()
            self._timezone = meta.get('timezone') or 'UTC'
        return self._timezone


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _local_date_to_utc_datetimes(
    local_date: 'date',
    tz_str: str,
) -> tuple[datetime, datetime]:
    """Convert a local calendar date to UTC-aware ``[start, end)`` datetimes."""
    from datetime import timedelta, time as dt_time

    tz = pytz.timezone(tz_str)
    local_midnight = tz.localize(datetime.combine(local_date, dt_time.min))
    start_utc = local_midnight.astimezone(pytz.utc)
    end_utc   = (local_midnight + timedelta(days=1)).astimezone(pytz.utc)
    return start_utc, end_utc


def _strip_tz(dt: datetime) -> datetime:
    """Return a naive copy of *dt* (drops tzinfo)."""
    return dt.replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Convenience entry-points
# ---------------------------------------------------------------------------

def generate_reports(
    db_path: Path,
    output_dir: Path,
    date_str: str,
) -> None:
    """Convenience function: create a ``PlotGenerator`` and run one date."""
    PlotGenerator(db_path=db_path, output_dir=output_dir).generate_for_date(date_str)