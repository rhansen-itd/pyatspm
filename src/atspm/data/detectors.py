"""
Co-Located Detector Discrepancy Data Engine (Imperative Shell)

Handles database I/O for the post-hoc detector health analysis.  Delegates
all computation to the Functional Core in ``src/atspm/analysis/detectors.py``.

Package Location: src/atspm/data/detectors.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .manager import DatabaseManager
from ..analysis.detectors import analyze_discrepancies

log = logging.getLogger(__name__)


class DetectorEngine:
    """Engine for co-located detector discrepancy analysis.

    Encapsulates database access patterns for the detector health feature.
    Follows the same architectural pattern as ``CountsEngine``:
    the engine owns the ``db_path`` / ``timezone`` context and exposes
    analysis methods that return DataFrames.

    Args:
        db_path:  Path to the intersection's SQLite database file.
        timezone: IANA timezone string used to interpret naive ``datetime``
                  arguments (e.g. ``'US/Mountain'``).  When ``None`` the
                  datetimes are treated as UTC-equivalent (``timestamp()``
                  is called without localisation).
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.timezone = timezone

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _localize_epoch(self, dt: datetime) -> float:
        """Convert a naive datetime to a UTC epoch float.

        Args:
            dt: Naive local datetime.

        Returns:
            UTC epoch float.
        """
        if self.timezone:
            try:
                import pytz
                tz = pytz.timezone(self.timezone)
                return tz.localize(dt).timestamp()
            except Exception:
                pass
        return dt.timestamp()

    def _fetch_events(
        self,
        manager: DatabaseManager,
        start_epoch: float,
        end_epoch: float,
        detector_pairs: List[Dict],
    ) -> pd.DataFrame:
        """Execute the optimised detector event query for a set of pairs.

        Fetches only Code-81/82 events for the relevant detector IDs plus
        all gap markers in the window.  Shared by both public methods so the
        SQL is never duplicated.

        Args:
            manager:        Open ``DatabaseManager`` context.
            start_epoch:    Window start (UTC epoch float, inclusive).
            end_epoch:      Window end (UTC epoch float, exclusive).
            detector_pairs: Filtered list of ``{"phase", "det_a", "det_b"}``
                            dicts.

        Returns:
            DataFrame with columns ``['timestamp', 'event_code', 'parameter']``,
            sorted by timestamp.  Empty DataFrame if no pairs supplied.
        """
        if not detector_pairs:
            return pd.DataFrame(columns=["timestamp", "event_code", "parameter"])

        det_ids: List[int] = list(
            {p["det_a"] for p in detector_pairs} |
            {p["det_b"] for p in detector_pairs}
        )
        det_ph = ", ".join("?" for _ in det_ids)

        sql = f"""
            SELECT timestamp, event_code, parameter
            FROM   events
            WHERE  timestamp >= ?
              AND  timestamp <  ?
              AND  (
                       (event_code IN (81, 82) AND parameter IN ({det_ph}))
                    OR  event_code = -1
                   )
            ORDER BY timestamp, event_code, parameter
        """
        params = [start_epoch, end_epoch] + det_ids
        return pd.read_sql_query(sql, manager.conn, params=params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_discrepancies(
        self,
        start: datetime,
        end: datetime,
        lag_threshold_sec: float = 2.0,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Fetch events and return detector discrepancy anomalies.

        Workflow:
            1. Resolves the active configuration at ``start`` to obtain the
               ``detector_pairs`` list (``[{"phase": int, "det_a": int,
               "det_b": int}, ...]``).
            2. Collects all unique detector IDs from the pairs to build a
               single, efficient SQL query (``event_code IN (81, 82)`` for
               those parameters, plus all gap markers).
            3. Passes the events DataFrame and pairs list to
               :func:`~atspm.analysis.detectors.analyze_discrepancies`.
            4. Optionally exports the result to
               ``output_dir/Discrepancies_{start}_{end}.csv``.

        Args:
            start: Query window start (naive local datetime).
            end: Query window end (naive local datetime, exclusive).
            lag_threshold_sec: Passed through to
                :func:`~atspm.analysis.detectors.analyze_discrepancies`.
                Defaults to ``2.0``.
            output_dir: When provided, the result DataFrame is written to a
                CSV file in this directory.  The directory is created if it
                does not exist.  The file is named
                ``Discrepancies_{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}.csv``.

        Returns:
            DataFrame of identified anomalies — see
            :func:`~atspm.analysis.detectors.analyze_discrepancies` for the
            full column schema.  Returns an empty DataFrame (same schema) when
            no anomalies are found or no detector pairs are configured.

        Raises:
            RuntimeError: If the database connection or query fails.
            ValueError: If no configuration is found for ``start``.
        """
        start_epoch = self._localize_epoch(start)
        end_epoch   = self._localize_epoch(end)

        with DatabaseManager(self.db_path) as manager:
            cfg = manager.get_config_at_date(start)
            if cfg is None:
                raise ValueError(
                    f"No configuration found for {start.isoformat()} "
                    f"in {self.db_path}"
                )

            detector_pairs = cfg.get("detector_pairs", [])
            if not detector_pairs:
                log.warning(
                    "No detector_pairs configured for %s at %s — "
                    "add Det_Ph<X>_Pairs rows to int_cfg.csv.",
                    self.db_path.name,
                    start.date(),
                )
                return analyze_discrepancies(pd.DataFrame(), [], lag_threshold_sec)

            events_df = self._fetch_events(
                manager, start_epoch, end_epoch, detector_pairs
            )

        result = analyze_discrepancies(
            events_df=events_df,
            detector_pairs=detector_pairs,
            lag_threshold_sec=lag_threshold_sec,
        )

        if output_dir is not None and not result.empty:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = (
                f"Discrepancies_"
                f"{start.strftime('%Y%m%d_%H%M%S')}_"
                f"{end.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            out_path = output_dir / fname
            result.to_csv(out_path, index=False)
            log.info("Discrepancy report written to %s", out_path)

        return result

    def get_plot_data(
        self,
        start: datetime,
        end: datetime,
        phases: Optional[List[int]] = None,
        lag_threshold_sec: float = 2.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """Fetch everything the detector comparison plot needs in one call.

        Workflow:
            1. Resolves the active configuration at ``start``.
            2. Filters ``detector_pairs`` to the requested ``phases`` (if
               supplied).
            3. Runs the optimised detector SQL query for all relevant IDs.
            4. Runs ``analyze_discrepancies`` on the events to produce
               ``anomalies_df``.
            5. Returns all three artefacts to the caller (imperative shell /
               ``PlotGenerator``) so the plotting function receives clean,
               pre-computed inputs.

        Args:
            start: Query window start (naive local datetime).
            end:   Query window end (naive local datetime, exclusive).
            phases: Optional list of signal phase numbers.  When provided,
                only pairs whose ``"phase"`` key appears in this list are
                included.  ``None`` means all configured pairs.
            lag_threshold_sec: Minimum disagreement duration in seconds passed
                to ``analyze_discrepancies``.  Defaults to ``2.0``.

        Returns:
            Tuple ``(events_df, anomalies_df, filtered_pairs)`` where:

            * **events_df** — raw detector events (Code 81/82) plus gap
              markers for the window; columns
              ``['timestamp', 'event_code', 'parameter']``.
            * **anomalies_df** — output of
              :func:`~atspm.analysis.detectors.analyze_discrepancies`;
              may be empty.
            * **filtered_pairs** — the ``detector_pairs`` list after phase
              filtering; list of ``{"phase", "det_a", "det_b"}`` dicts.

        Raises:
            ValueError: If no configuration is found for ``start``, or if
                ``phases`` is provided but none of the requested phases have
                configured pairs.
        """
        start_epoch = self._localize_epoch(start)
        end_epoch   = self._localize_epoch(end)

        with DatabaseManager(self.db_path) as manager:
            cfg = manager.get_config_at_date(start)
            if cfg is None:
                raise ValueError(
                    f"No configuration found for {start.isoformat()} "
                    f"in {self.db_path}"
                )

            all_pairs: List[Dict] = cfg.get("detector_pairs", [])

            if not all_pairs:
                log.warning(
                    "No detector_pairs configured for %s at %s.",
                    self.db_path.name,
                    start.date(),
                )
                empty_events = pd.DataFrame(
                    columns=["timestamp", "event_code", "parameter"]
                )
                empty_anomalies = analyze_discrepancies(
                    pd.DataFrame(), [], lag_threshold_sec
                )
                return empty_events, empty_anomalies, []

            # Phase filtering
            if phases is not None:
                phase_set = set(phases)
                filtered_pairs = [p for p in all_pairs if p["phase"] in phase_set]
                if not filtered_pairs:
                    raise ValueError(
                        f"No detector pairs found for phase(s) {sorted(phase_set)} "
                        f"in {self.db_path.name} at {start.date()}."
                    )
            else:
                filtered_pairs = list(all_pairs)

            events_df = self._fetch_events(
                manager, start_epoch, end_epoch, filtered_pairs
            )

        anomalies_df = analyze_discrepancies(
            events_df=events_df,
            detector_pairs=filtered_pairs,
            lag_threshold_sec=lag_threshold_sec,
        )

        return events_df, anomalies_df, filtered_pairs


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def get_detector_discrepancies(
    db_path: Path,
    start: datetime,
    end: datetime,
    lag_threshold_sec: float = 2.0,
    timezone: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Convenience wrapper: initialise a DetectorEngine and run discrepancy analysis.

    Suitable for one-off script usage or CLI dispatch.  For repeated calls
    on the same intersection, prefer constructing a :class:`DetectorEngine`
    directly to avoid the per-call metadata overhead.

    Args:
        db_path:           Path to the intersection's SQLite database file.
        start:             Query window start (naive local datetime).
        end:               Query window end (naive local datetime, exclusive).
        lag_threshold_sec: Minimum disagreement duration in seconds.
                           Defaults to ``2.0``.
        timezone:          IANA timezone string (e.g. ``'US/Mountain'``).
                           ``None`` treats datetimes as UTC-equivalent.
        output_dir:        When provided, the result is exported to a CSV in
                           this directory.

    Returns:
        DataFrame of identified anomalies (may be empty).

    Raises:
        FileNotFoundError: If ``db_path`` does not exist.
        ValueError:        If no configuration covers ``start``.
    """
    engine = DetectorEngine(db_path=db_path, timezone=timezone)
    return engine.get_discrepancies(
        start=start,
        end=end,
        lag_threshold_sec=lag_threshold_sec,
        output_dir=output_dir,
    )
