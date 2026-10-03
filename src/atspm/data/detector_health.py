"""
ATSPM Detector Health Engine (Imperative Shell)

Orchestrates detector health analysis by querying the SQLite database,
resolving temporal configuration, profiling detector activity, evaluating
deterministic rules, persisting findings into ``detector_findings``, and
producing reported views and heatmap plots.

Package Location: src/atspm/data/detector_health.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .manager import DatabaseManager, db_timezone
from .reader import get_events_with_cycles_df
from ..analysis.detector_activity import detector_activity_profile
from ..analysis.detector_health import (
    CONTROLLER_FAULT_CODES,
    FINDINGS_SCHEMA,
    HealthThresholds,
    apply_ignore,
    detector_health_findings,
    filter_min_severity,
    severity_exit_code,
    wd_ignore,
    wd_profile_windows,
    wd_reboot_windows,
    wd_thresholds,
    wd_units,
)
from ..analysis.detector_roles import parse_detector_roles
from ..analysis.timing_actuation import finding_plot_windows
from ..plotting.detector_health import plot_detector_health
from ..utils.timezone import resolve_pytz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Codes the profile + event rules need: detector on/off, gap marker, max-out,
# and Indiana controller-fault codes.
_ALL_HEALTH_CODES: List[int] = sorted(
    {-1, 5, 81, 82} | set(CONTROLLER_FAULT_CODES.keys())
)
# Event fetch margin (seconds) to measure releases/on-durations near day edges.
_EDGE_MARGIN_S: float = 3600.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DetectorHealthEngine:
    """Queries the database and evaluates detector-health rules.

    All date arguments are interpreted in the intersection's local timezone.
    Results are persisted to the ``detector_findings`` table and optionally
    written to CSV and HTML heatmap files.
    """

    def __init__(self, db_path: Path, timezone: Optional[str] = None) -> None:
        """
        Args:
            db_path: Path to the intersection SQLite database.
            timezone: Local timezone string (e.g. ``'US/Mountain'``).
                      Defaults to the value stored in the ``metadata`` table,
                      with a final fallback to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.timezone = timezone or self._read_timezone()

    def detector_health(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        window: str = "day",
        min_severity: str = "low",
        thresholds: Optional[HealthThresholds] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, object]:
        """Run detector-health evaluation and optionally write outputs.

        Args:
            start: Inclusive start date (``'YYYY-MM-DD'`` string or naive datetime).
            end: Inclusive end date (``'YYYY-MM-DD'`` string or naive datetime).
                 The query window extends to end-of-day.
            window: Reporting window filter (``'day'``, ``'am'``, ``'pm'``, or ``'all'``).
                    Default is ``'day'``.
            min_severity: Reporting severity threshold (``'info'``, ``'low'``, ``'high'``).
                          Default is ``'low'``.
            thresholds: Optional :class:`HealthThresholds` overrides.
            output_dir: Optional destination directory for CSV and HTML output files.

        Returns:
            Dictionary with keys:
            - ``"findings"``: DataFrame of all computed findings across the range.
            - ``"reported"``: DataFrame of reported findings (ignore-filtered,
              window-filtered, min_severity-filtered).
            - ``"exit_code"``: Integer exit code (0 = clean, 1 = low, 2 = high).
        """
        start_dt, end_dt = self._parse_range(start, end)
        date_start = start_dt.strftime("%Y-%m-%d")
        date_end = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        req_start_date = start_dt.date()
        req_end_date = (end_dt - timedelta(days=1)).date()

        # Fetch events once with edge margins
        start_fetch = start_dt - timedelta(seconds=_EDGE_MARGIN_S)
        end_fetch = end_dt + timedelta(seconds=_EDGE_MARGIN_S)
        events_df = get_events_with_cycles_df(
            db_path=self.db_path,
            start=start_fetch,
            end=end_fetch,
            event_codes=_ALL_HEALTH_CODES,
            timezone=self.timezone,
        )

        if events_df.empty:
            print(
                f"  ⚠️  DetectorHealth: no events found for {date_start} to {date_end}."
            )
            with DatabaseManager(self.db_path) as m:
                m.replace_findings(
                    pd.DataFrame(columns=FINDINGS_SCHEMA), date_start, date_end
                )
            empty_findings = pd.DataFrame(columns=FINDINGS_SCHEMA).astype(
                {"detector": "int64", "phase": "Int64"}
            )
            return {
                "findings": empty_findings,
                "reported": empty_findings,
                "exit_code": 0,
            }

        # Convert timestamp column to plain UTC epoch floats
        ts_col = events_df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(ts_col):
            events_df["timestamp"] = (
                ts_col.astype("int64").to_numpy(dtype=float) / 1e9
            )
        else:
            first_ts = ts_col.iloc[0]
            if hasattr(first_ts, "timestamp"):
                events_df["timestamp"] = np.array(
                    [t.timestamp() for t in ts_col], dtype=float
                )
            else:
                events_df["timestamp"] = ts_col.to_numpy(dtype=float)

        t0 = float(events_df["timestamp"].iloc[0])
        t1 = float(events_df["timestamp"].iloc[-1])
        zone = resolve_pytz(self.timezone)
        lo = datetime.fromtimestamp(t0, zone)
        hi = datetime.fromtimestamp(t1 + 1.0, zone)

        with DatabaseManager(self.db_path) as m:
            configs = m.get_configs_for_range(lo, hi)

        if not configs:
            configs = [{
                "_epoch_start": t0,
                "_epoch_end": t1 + 1.0,
            }]

        findings_parts: List[pd.DataFrame] = []
        profile_parts: List[pd.DataFrame] = []

        for cfg in configs:
            cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
            roles = parse_detector_roles(cfg_clean)
            windows = wd_profile_windows(cfg)
            th = thresholds or wd_thresholds(cfg)
            units, _types = wd_units(cfg)
            reboot = wd_reboot_windows(cfg)

            lo_epoch = cfg["_epoch_start"]
            hi_epoch = cfg["_epoch_end"]
            sub_events = events_df.loc[
                (events_df["timestamp"] >= lo_epoch)
                & (events_df["timestamp"] < hi_epoch)
            ]
            if sub_events.empty:
                continue

            cfg_start_date = datetime.fromtimestamp(lo_epoch, zone).date()
            cfg_end_date = datetime.fromtimestamp(
                max(lo_epoch, hi_epoch - 1.0), zone
            ).date()
            p_start = max(req_start_date, cfg_start_date)
            p_end = min(req_end_date, cfg_end_date)
            if p_start > p_end:
                continue

            profile = detector_activity_profile(
                sub_events,
                self.timezone,
                roles=roles,
                windows=windows,
                start_date=p_start,
                end_date=p_end,
            )
            if not profile.empty:
                profile_parts.append(profile)

            findings = detector_health_findings(
                profile,
                roles=roles,
                events_df=sub_events,
                tz=self.timezone,
                thresholds=th,
                reboot_windows=reboot,
                units=units,
            )
            if not findings.empty:
                findings_parts.append(findings)

        profile_all = (
            pd.concat(profile_parts, ignore_index=True)
            if profile_parts
            else pd.DataFrame()
        )
        all_findings = (
            pd.concat(findings_parts, ignore_index=True)
            if findings_parts
            else pd.DataFrame(columns=FINDINGS_SCHEMA).astype(
                {"detector": "int64", "phase": "Int64"}
            )
        )

        if not all_findings.empty:
            date_strs = all_findings["date"].astype(str)
            all_findings = all_findings.loc[
                (date_strs >= date_start) & (date_strs <= date_end)
            ].reset_index(drop=True)

        if not profile_all.empty:
            p_dates = profile_all["date"].astype(str)
            profile_all = profile_all.loc[
                (p_dates >= date_start) & (p_dates <= date_end)
            ].reset_index(drop=True)

        # 4. RecordCount rule: check coverage using ingestion_log spans
        with DatabaseManager(self.db_path) as m:
            spans_df = m.get_ingestion_spans()

        if not spans_df.empty:
            rc_findings = []
            cfg_latest = configs[-1] if configs else {}
            th_latest = thresholds or wd_thresholds(cfg_latest)
            profile_wins = wd_profile_windows(cfg_latest)

            cur_date = req_start_date
            while cur_date <= req_end_date:
                d_str = cur_date.isoformat()
                for win_name, (w_start, w_end) in profile_wins.items():
                    s_h, s_m = [int(p) for p in w_start.split(":")]
                    e_h, e_m = [int(p) for p in w_end.split(":")]
                    start_local = zone.localize(
                        datetime.combine(cur_date, datetime.min.time())
                        + timedelta(hours=s_h, minutes=s_m)
                    )
                    if e_h == 24:
                        end_local = zone.localize(
                            datetime.combine(
                                cur_date + timedelta(days=1), datetime.min.time()
                            )
                        )
                    else:
                        end_local = zone.localize(
                            datetime.combine(cur_date, datetime.min.time())
                            + timedelta(hours=e_h, minutes=e_m)
                        )
                    w_start_epoch = start_local.timestamp()
                    w_end_epoch = end_local.timestamp()
                    w_dur = w_end_epoch - w_start_epoch
                    if w_dur <= 0:
                        continue

                    overlaps = np.maximum(
                        0.0,
                        np.minimum(spans_df["span_end"].to_numpy(float), w_end_epoch)
                        - np.maximum(
                            spans_df["span_start"].to_numpy(float), w_start_epoch
                        ),
                    )
                    cov = float(np.clip(overlaps.sum() / w_dur, 0.0, 1.0))
                    if cov < th_latest.min_observed_share:
                        rc_findings.append({
                            "date": d_str,
                            "window": win_name,
                            "ts": np.nan,
                            "detector": -1,
                            "phase": pd.NA,
                            "role": "",
                            "rule": "RecordCount",
                            "severity": "low",
                            "value": cov,
                            "threshold": float(th_latest.min_observed_share),
                            "message": (
                                f"Logged coverage {cov:.1%} is below minimum "
                                f"observed share {th_latest.min_observed_share:.1%}"
                            ),
                        })
                cur_date += timedelta(days=1)

            if rc_findings:
                rc_df = pd.DataFrame(rc_findings, columns=FINDINGS_SCHEMA).astype({
                    "detector": "int64",
                    "phase": "Int64",
                    "value": "float64",
                    "threshold": "float64",
                })
                all_findings = pd.concat(
                    [all_findings, rc_df], ignore_index=True
                )

        # 5. Persist all findings (unfiltered)
        with DatabaseManager(self.db_path) as m:
            m.replace_findings(all_findings, date_start, date_end)

        # 6. Build the reported view
        cfg_latest = configs[-1] if configs else {}
        ignore = wd_ignore(cfg_latest)
        reported = apply_ignore(all_findings, ignore)
        if window not in (None, "all"):
            reported = reported.loc[reported["window"] == window].reset_index(
                drop=True
            )
        reported = filter_min_severity(reported, min_severity)
        exit_code = severity_exit_code(reported, min_severity)

        # Build timing_plot link column on reported findings
        with DatabaseManager(self.db_path) as m:
            metadata = m.get_metadata() or {}
        int_id = metadata.get("intersection_id")
        target_arg = f"--targetid {int_id}" if int_id else f"--target {self.db_path.parent.name}"

        if not reported.empty:
            links = finding_plot_windows(reported, events_df, self.timezone)
            zone = resolve_pytz(self.timezone)
            valid_mask = links["plot_start"].notna()

            start_s = pd.Series("", index=links.index, dtype=str)
            end_s = pd.Series("", index=links.index, dtype=str)

            if valid_mask.any():
                s_dt = pd.to_datetime(
                    links.loc[valid_mask, "plot_start"], unit="s", utc=True
                ).dt.tz_convert(zone)
                e_dt = pd.to_datetime(
                    links.loc[valid_mask, "plot_end"], unit="s", utc=True
                ).dt.tz_convert(zone)
                start_s.loc[valid_mask] = s_dt.dt.strftime("%Y-%m-%dT%H:%M:%S")
                end_s.loc[valid_mask] = e_dt.dt.strftime("%Y-%m-%dT%H:%M:%S")

            base_cmd = f"atspm plot-timing-actuation {target_arg} --start " + start_s + " --end " + end_s

            has_phase = links["plot_phase"].notna()
            has_det = (~has_phase) & links["plot_detector"].notna()

            suffix = pd.Series("", index=links.index, dtype=str)
            suffix.loc[has_phase] = " --phases " + links.loc[has_phase, "plot_phase"].astype(str)
            suffix.loc[has_det] = " --detectors " + links.loc[has_det, "plot_detector"].astype(str)

            full_cmd = base_cmd + suffix
            reported["timing_plot"] = full_cmd.where(valid_mask, "")
        else:
            reported["timing_plot"] = pd.Series([], dtype=str)

        # 7. Write outputs if output_dir provided
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            stamp = f"{date_start.replace('-', '_')}-{date_end.replace('-', '_')}"

            csv_file = output_path / f"DetectorHealth_Findings_{stamp}.csv"
            reported.to_csv(csv_file, index=False)
            print(f"Wrote {csv_file.name}")

            plot_win = window if window not in (None, "all") else "day"
            sub_prof = (
                profile_all.loc[profile_all["window"] == plot_win]
                if not profile_all.empty
                else profile_all
            )
            with DatabaseManager(self.db_path) as m:
                metadata = m.get_metadata() or {}
            fig = plot_detector_health(
                sub_prof, reported, metadata=metadata, window=plot_win
            )
            html_file = output_path / f"DetectorHealth_Heatmap_{stamp}.html"
            fig.write_html(str(html_file))
            print(f"Wrote {html_file.name}")

        # 8. Print short summary
        n_total = len(all_findings)
        n_ignored = len(all_findings) - len(apply_ignore(all_findings, ignore))
        n_rep = len(reported)
        print(f"\nDetector Health summary for {date_start} → {date_end}:")
        print(f"  Total findings:    {n_total}")
        print(f"  Ignored:           {n_ignored}")
        print(
            f"  Reported findings: {n_rep} (min_severity={min_severity}, window={window})"
        )
        if not reported.empty:
            print("  By rule:")
            for rule, count in reported["rule"].value_counts().items():
                print(f"    - {rule}: {count}")
            print("  By severity:")
            for sev, count in reported["severity"].value_counts().items():
                print(f"    - {sev}: {count}")

        return {
            "findings": all_findings,
            "reported": reported,
            "exit_code": exit_code,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_timezone(self) -> str:
        """Read the intersection timezone from the database."""
        return db_timezone(self.db_path)

    def _parse_range(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
    ) -> tuple[datetime, datetime]:
        """Coerce ``'YYYY-MM-DD'`` strings to naive midnight datetimes.

        The *end* date is extended by one day so the query window covers the
        full last calendar day, matching the ``counts`` / ``splits`` convention.

        Args:
            start: Start date string or naive datetime.
            end:   End date string or naive datetime.

        Returns:
            Tuple of ``(start_dt, end_dt)`` as naive datetimes.
        """
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        elif isinstance(end, datetime):
            end = end + timedelta(days=1)
        return start, end


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def get_detector_health(
    db_path: Path,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: str = "day",
    min_severity: str = "low",
    thresholds: Optional[HealthThresholds] = None,
    output_dir: Optional[Union[str, Path]] = None,
    timezone: Optional[str] = None,
) -> Dict[str, object]:
    """Convenience wrapper around :class:`DetectorHealthEngine`.detector_health.

    Args:
        db_path: Path to the intersection SQLite database.
        start: Inclusive start date/datetime (local time).
        end: Inclusive end date/datetime (local time).
        window: Reporting window (``'day'``, ``'am'``, ``'pm'``, or ``'all'``).
        min_severity: Minimum severity floor (``'info'``, ``'low'``, ``'high'``).
        thresholds: Optional rule threshold overrides.
        output_dir: Optional directory to write CSV and HTML heatmap.
        timezone: Override intersection timezone.

    Returns:
        Dict with keys ``"findings"``, ``"reported"``, and ``"exit_code"``.
    """
    engine = DetectorHealthEngine(db_path, timezone=timezone)
    return engine.detector_health(
        start=start,
        end=end,
        window=window,
        min_severity=min_severity,
        thresholds=thresholds,
        output_dir=output_dir,
    )
