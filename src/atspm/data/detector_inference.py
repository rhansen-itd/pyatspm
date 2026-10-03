"""
ATSPM Detector Configuration Inference Engine (Imperative Shell)

Orchestrates detector configuration inference by querying raw events from the
SQLite database, resolving active configuration, and delegating classification
and diffing to the Functional Core (:mod:`atspm.analysis.detector_inference`).

Package Location: src/atspm/data/detector_inference.py

The engine proposes; it never writes config.  No int_cfg.csv writes, and no
config-table writes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .critical import CriticalMovementEngine
from .manager import DatabaseManager, db_timezone
from ..analysis.cycles import _parse_ring_groups
from ..analysis.detector_inference import diff_detector_roles, infer_detector_roles
from ..analysis.detector_roles import parse_detector_roles
from ..utils.timezone import to_epoch

_INFERENCE_CODES: List[int] = [-1, 1, 8, 9, 81, 82]


class DetectorInferenceEngine:
    """Queries the database and produces proposed detector roles and diffs.

    All date/time arguments are interpreted in the intersection's local
    timezone (read from the ``metadata`` table).

    Example::

        engine = DetectorInferenceEngine(Path("2068_data.db"))
        res = engine.infer("2023-11-14", "2023-11-16")
        print(res["proposed"])
        print(res["diff"])
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """Initialize the engine.

        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g., ``'US/Mountain'``).
                Defaults to the value stored in the ``metadata`` table,
                with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def infer(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        use_ring_config: bool = True,
        min_actuations: int = 50,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Infer detector configuration and compare with active config.

        Args:
            start: Start date/datetime string or naive datetime.
            end: End date/datetime string or naive datetime. A date-only
                end date covers the whole day.
            use_ring_config: When True, limit candidate phases to those in
                the active config's RB_R1/RB_R2 rings.
            min_actuations: Minimum uncensored actuations required to classify
                a detector.
            output_dir: Optional directory to write output CSVs. When None,
                returns the result dict.

        Returns:
            Dict containing ``'proposed'`` and ``'diff'`` DataFrames if
            *output_dir* is None, otherwise None.
        """
        start_dt, end_dt = CriticalMovementEngine._parse_range(start, end)
        start_epoch = to_epoch(start_dt, self.timezone)
        end_epoch = to_epoch(end_dt, self.timezone)

        with DatabaseManager(self.db_path) as m:
            events_df = m.query_events(start_epoch, end_epoch, _INFERENCE_CODES)

        config = self._get_config(start_dt)

        phases: Optional[List[int]] = None
        if use_ring_config and config:
            ring_phases = set()
            for key in ("RB_R1", "RB_R2"):
                val = config.get(key)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    for group in _parse_ring_groups(val):
                        ring_phases.update(group)
            if ring_phases:
                phases = sorted(ring_phases)

        proposed = infer_detector_roles(
            events_df,
            phases=phases,
            min_actuations=min_actuations,
        )

        active_counts: Dict[int, int] = {}
        if not events_df.empty and (events_df["event_code"] == 82).any():
            active_counts = (
                events_df[events_df["event_code"] == 82]
                .groupby("parameter")
                .size()
                .to_dict()
            )

        configured_roles = parse_detector_roles(config)
        diff = diff_detector_roles(proposed, configured_roles, active_counts)

        self._print_summary(diff)

        if output_dir is not None:
            self._write_outputs(proposed, diff, output_dir, start_dt, end_dt)
            return None

        return {"proposed": proposed, "diff": diff}

    def _read_timezone(self) -> str:
        """Read the intersection timezone from the database."""
        return db_timezone(self.db_path)

    def _get_config(self, date: datetime) -> dict:
        """Retrieve the active configuration dict for a given date.

        Args:
            date: Reference datetime (naive local) used to select the
                correct temporal config row.

        Returns:
            Config dict (empty dict if no config is found).
        """
        with DatabaseManager(self.db_path) as m:
            config = m.get_config_at_date(date)
        return config or {}

    def _print_summary(self, diff: pd.DataFrame) -> None:
        """Print summary counts and detail lines for flagged detectors.

        Args:
            diff: DataFrame returned by diff_detector_roles.
        """
        status_counts = diff["status"].value_counts().to_dict()
        status_str = ", ".join(f"{s}: {c}" for s, c in sorted(status_counts.items()))
        print(f"Summary: {status_str}")

        for row in diff.itertuples(index=False):
            if row.status in ("conflict", "new", "silent"):
                if pd.notna(row.proposed_phase):
                    prop_ph = f"P{row.proposed_phase}"
                elif row.candidates:
                    prop_ph = str(row.candidates)
                else:
                    prop_ph = "-"

                if prop_ph != "-":
                    p_disp = prop_ph if prop_ph.startswith("P") else f"P{prop_ph}"
                elif row.configured:
                    p_disp = "-"
                    for part in str(row.configured).split(","):
                        if " P" in part:
                            p_disp = "P" + part.split(" P")[-1]
                            break
                else:
                    p_disp = "-"

                cfg_disp = row.configured if row.configured else "-"
                role_disp = row.proposed_role if row.proposed_role else "-"
                conf_disp = row.confidence if row.confidence else "-"
                print(
                    f"  {p_disp} {row.detector}: {row.status} — "
                    f"configured {cfg_disp}; proposed {role_disp} {prop_ph} ({conf_disp})"
                )

    def _write_outputs(
        self,
        proposed: pd.DataFrame,
        diff: pd.DataFrame,
        output_dir: Union[str, Path],
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        """Write proposed and diff DataFrames to CSV files.

        Args:
            proposed: Inferred roles DataFrame.
            diff: Inferred vs configured diff DataFrame.
            output_dir: Destination directory.
            start_dt: Window start datetime.
            end_dt: Window end datetime.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def _stamp(dt: datetime) -> str:
            if (dt.hour, dt.minute) == (0, 0):
                return f"{dt:%Y_%m_%d}"
            return f"{dt:%Y_%m_%d_%H%M}"

        end_label = (
            end_dt - timedelta(days=1)
            if (end_dt.hour, end_dt.minute) == (0, 0)
            else end_dt
        )
        date_str = f"{_stamp(start_dt)}-{_stamp(end_label)}"

        prop_file = f"Detector_Inference_{date_str}.csv"
        diff_file = f"Detector_Inference_Diff_{date_str}.csv"

        proposed.to_csv(output_path / prop_file, index=False)
        print(f"Wrote {prop_file}")

        diff.to_csv(output_path / diff_file, index=False)
        print(f"Wrote {diff_file}")


def get_detector_inference(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    use_ring_config: bool = True,
    min_actuations: int = 50,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
    **kwargs,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Convenience wrapper around :class:`DetectorInferenceEngine`.infer.

    Args:
        db_path: Path to the intersection SQLite database.
        start: Window start date/datetime (local time).
        end: Window end date/datetime (local time).
        use_ring_config: Limit candidate phases to RB_* ring config.
        min_actuations: Minimum uncensored actuations to classify channel.
        output_dir: Optional directory to write output CSVs.
        timezone: Override intersection timezone.
        **kwargs: Additional keyword arguments passed to infer().

    Returns:
        Dict with "proposed" and "diff" DataFrames, or None if output_dir set.
    """
    return DetectorInferenceEngine(db_path, timezone=timezone).infer(
        start=start,
        end=end,
        use_ring_config=use_ring_config,
        min_actuations=min_actuations,
        output_dir=output_dir,
        **kwargs,
    )
