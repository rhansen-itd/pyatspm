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
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .manager import DatabaseManager
from ..utils.timezone import resolve_pytz, to_epoch
from ..analysis.detectors import analyze_discrepancies

log = logging.getLogger(__name__)

# Events are fetched this far past each window edge so an actuation that
# crosses the edge is reconstructed whole instead of being dropped (or
# leaving its partner's actuation looking like a disagreement).
_EDGE_MARGIN_SEC = 900.0

_EVENT_COLUMNS = ["timestamp", "event_code", "parameter"]

PairKey = Tuple[int, int, int]


def _pair_spans(
    configs: List[Dict],
    phases: Optional[Set[int]],
) -> Dict[PairKey, List[Tuple[float, float]]]:
    """Map each configured pair to the epoch spans during which it is active.

    Spans of consecutive configs that both define a pair are merged, so an
    anomaly crossing a config boundary is reported once.

    Args:
        configs: Output of ``DatabaseManager.get_configs_for_range``, sorted
            ascending, each carrying ``_epoch_start`` / ``_epoch_end`` and
            ``detector_pairs``.
        phases: When given, only pairs for these phases are kept.

    Returns:
        ``{(phase, det_a, det_b): [(start_epoch, end_epoch), ...]}``, ordered
        by phase and then first appearance.
    """
    spans: Dict[PairKey, List[Tuple[float, float]]] = {}
    for cfg in configs:
        c_start, c_end = cfg["_epoch_start"], cfg["_epoch_end"]
        if c_end <= c_start:
            continue
        for pair in cfg.get("detector_pairs", []):
            if phases is not None and pair["phase"] not in phases:
                continue
            key = (pair["phase"], pair["det_a"], pair["det_b"])
            key_spans = spans.setdefault(key, [])
            if key_spans and c_start <= key_spans[-1][1]:
                key_spans[-1] = (key_spans[-1][0], max(key_spans[-1][1], c_end))
            else:
                key_spans.append((c_start, c_end))
    return dict(sorted(spans.items(), key=lambda kv: kv[0][0]))


class DetectorEngine:
    """Engine for co-located detector discrepancy analysis.

    Encapsulates database access patterns for the detector health feature.
    Follows the same architectural pattern as ``CountsEngine``:
    the engine owns the ``db_path`` / ``timezone`` context and exposes
    analysis methods that return DataFrames.

    Args:
        db_path:  Path to the intersection's SQLite database file.
        timezone: IANA timezone string used to interpret naive ``datetime``
                  arguments (e.g. ``'US/Mountain'``).  When ``None`` naive
                  datetimes are read as UTC.  Aware datetimes always keep
                  their own offset.
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
        """Convert a window bound to a UTC epoch float.

        Args:
            dt: Window bound.  Naive datetimes are read as ``self.timezone``
                local; aware ones keep their own offset.

        Returns:
            UTC epoch float.
        """
        return to_epoch(dt, self.timezone)

    def _local_aware(self, dt: datetime) -> datetime:
        """Express a window bound as a pytz-aware datetime in ``self.timezone``.

        ``DatabaseManager.get_configs_for_range`` needs a pytz tzinfo (it
        calls ``localize``), whatever form the caller passed.

        Args:
            dt: Window bound, naive (intersection-local) or aware.

        Returns:
            The same instant, aware in the intersection's pytz zone.
        """
        tz = resolve_pytz(self.timezone)
        if dt.tzinfo is not None:
            dt = dt.astimezone(tz).replace(tzinfo=None)
        return tz.localize(dt)

    def _fetch_events(
        self,
        manager: DatabaseManager,
        start_epoch: float,
        end_epoch: float,
        detector_pairs: List[Dict],
    ) -> pd.DataFrame:
        """Execute the optimised detector event query for a set of pairs.

        Fetches only Code-81/82 events for the relevant detector IDs plus
        all gap markers in the window.

        Args:
            manager:        Open ``DatabaseManager`` context.
            start_epoch:    Fetch start (UTC epoch float, inclusive).
            end_epoch:      Fetch end (UTC epoch float, exclusive).
            detector_pairs: List of ``{"phase", "det_a", "det_b"}`` dicts.

        Returns:
            DataFrame with columns ``['timestamp', 'event_code', 'parameter']``,
            sorted by timestamp.  Empty DataFrame if no pairs supplied.
        """
        if not detector_pairs:
            return pd.DataFrame(columns=_EVENT_COLUMNS)

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

    def _run(
        self,
        start: datetime,
        end: datetime,
        phases: Optional[List[int]],
        lag_threshold_sec: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """Shared fetch-and-analyse path behind both public methods.

        Every config overlapping ``[start, end)`` contributes its pairs, each
        analysed only over the part of the window where it is configured.
        Events are fetched ``_EDGE_MARGIN_SEC`` past both edges; anomalies
        are kept when they overlap the window.

        Args:
            start: Window start (naive local, or aware).
            end:   Window end (naive local, or aware; exclusive).
            phases: Optional phase filter; ``None`` keeps all pairs.
            lag_threshold_sec: Passed to ``analyze_discrepancies``.

        Returns:
            ``(events_df, anomalies_df, pairs)`` -- see :meth:`get_plot_data`.

        Raises:
            ValueError: If no configuration overlaps the window, or if
                ``phases`` is given but none of them have configured pairs.
        """
        start_epoch = self._localize_epoch(start)
        end_epoch   = self._localize_epoch(end)
        empty_events    = pd.DataFrame(columns=_EVENT_COLUMNS)
        empty_anomalies = analyze_discrepancies(pd.DataFrame(), [], lag_threshold_sec)

        with DatabaseManager(self.db_path) as manager:
            configs = manager.get_configs_for_range(
                self._local_aware(start), self._local_aware(end)
            )
            if not configs:
                raise ValueError(
                    f"No configuration found for {start.isoformat()} – "
                    f"{end.isoformat()} in {self.db_path}"
                )

            spans = _pair_spans(configs, set(phases) if phases is not None else None)
            if not spans:
                if phases is not None:
                    raise ValueError(
                        f"No detector pairs found for phase(s) {sorted(set(phases))} "
                        f"in {self.db_path.name} for {start.date()}."
                    )
                log.warning(
                    "No detector_pairs configured for %s at %s — "
                    "add Det_P<X>_Pairs rows to int_cfg.csv.",
                    self.db_path.name,
                    start.date(),
                )
                return empty_events, empty_anomalies, []

            pairs = [{"phase": ph, "det_a": a, "det_b": b} for ph, a, b in spans]
            events_df = self._fetch_events(
                manager,
                start_epoch - _EDGE_MARGIN_SEC,
                end_epoch + _EDGE_MARGIN_SEC,
                pairs,
            )

        # Pairs active over identical spans are analysed together.
        by_spans: Dict[Tuple[Tuple[float, float], ...], List[Dict]] = {}
        for (ph, a, b), key_spans in spans.items():
            by_spans.setdefault(tuple(key_spans), []).append(
                {"phase": ph, "det_a": a, "det_b": b}
            )

        frames = [
            analyze_discrepancies(events_df, group, lag_threshold_sec, window=span)
            for key_spans, group in by_spans.items()
            for span in key_spans
        ]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return events_df, empty_anomalies, pairs

        anomalies_df = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates()
            .sort_values(["phase", "start_timestamp"])
            .reset_index(drop=True)
        )
        return events_df, anomalies_df, pairs

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
            1. Resolves every configuration overlapping ``[start, end)`` and
               the span over which each ``detector_pairs`` entry is active.
            2. Fetches Code-81/82 events for all paired detectors, plus gap
               markers, from ``_EDGE_MARGIN_SEC`` before ``start`` to the same
               margin after ``end``.
            3. Runs :func:`~atspm.analysis.detectors.analyze_discrepancies`
               per active span, keeping anomalies that overlap it.
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
            ValueError: If no configuration overlaps the window.
        """
        _, result, _ = self._run(start, end, None, lag_threshold_sec)

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

        Same workflow as :meth:`get_discrepancies`, with an optional phase
        filter, returning the raw events and pair list alongside the
        anomalies so the plotting function receives clean, pre-computed
        inputs.

        Args:
            start: Query window start (naive local datetime).
            end:   Query window end (naive local datetime, exclusive).
            phases: Optional list of signal phase numbers.  When provided,
                only pairs whose ``"phase"`` key appears in this list are
                included.  ``None`` means all configured pairs.
            lag_threshold_sec: Minimum disagreement duration in seconds passed
                to ``analyze_discrepancies``.  Defaults to ``2.0``.

        Returns:
            Tuple ``(events_df, anomalies_df, pairs)`` where:

            * **events_df** — raw detector events (Code 81/82) plus gap
              markers, extending ``_EDGE_MARGIN_SEC`` past both window edges;
              columns ``['timestamp', 'event_code', 'parameter']``.
            * **anomalies_df** — anomalies overlapping the window (see
              :func:`~atspm.analysis.detectors.analyze_discrepancies`);
              may be empty.
            * **pairs** — every pair configured at any point in the window
              (after phase filtering), ordered by phase; list of
              ``{"phase", "det_a", "det_b"}`` dicts.

        Raises:
            ValueError: If no configuration overlaps the window, or if
                ``phases`` is provided but none of the requested phases have
                configured pairs.
        """
        return self._run(start, end, phases, lag_threshold_sec)


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
