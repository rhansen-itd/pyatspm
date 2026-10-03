"""S-D2 calibration sweep: detector-health rules over the corpus DBs.

Profiles every logged day at each site (``day`` + ``am`` windows), runs each
rule over a grid of thresholds and writes the flagged rows to
``docs/calibration/detector_health_<rule>.csv`` plus a summary table
(``detector_health_summary.csv``).  Reproduce with::

    .venv/bin/python docs/calibration/sweep_detector_health.py
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from atspm.analysis.detector_activity import UDOT_AM_WINDOW, detector_activity_profile
from atspm.analysis.detector_health import (
    DEFAULT_THRESHOLDS,
    _labelled,
    detector_health_findings,
    chatter,
    configured_silent,
    failsafe_findings,
    low_detector_hits,
    onset_bursts,
    stuck_on,
    unconfigured_detector,
)
from atspm.analysis.detector_roles import parse_detector_roles
from atspm.data.manager import DatabaseManager
from atspm.utils.timezone import resolve_pytz

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SITES = ["201", "313", "315", "701"]
WINDOWS = {"day": ("00:00", "24:00"), "am": UDOT_AM_WINDOW}
# Candidate reboot windows from the burst scan (201/701: 23:59:05-08,
# 00:00:01, 01:55:12-16 local).  Calibration input, not a default.
REBOOT = {"pre_midnight": ("23:58", "24:00"), "midnight": ("00:00", "00:03"),
          "early": ("01:54", "01:58")}

GRIDS = {
    "LowDetectorHits": ("low_hits_min", [10, 20, 50, 100]),
    "UnconfiguredDetector": ("unconfigured_min_act", [1, 20, 100]),
    "Chatter": ("chatter_peer_ratio", [2.0, 3.0, 5.0]),
    "Failsafe": ("burst_min_channels", [6, 8, 12, 16]),
}
STUCK_MIN = [5, 10, 15, 30, 60]
SHARES = [0.8, 0.9, 0.95]


def site_db(site: str) -> Path:
    return next((ROOT / "intersections").glob(f"{site}_*/{site}_data.db"))


def load(site: str):
    db = site_db(site)
    with sqlite3.connect(db) as c:
        events = pd.read_sql("SELECT timestamp, event_code, parameter FROM events ORDER BY timestamp", c)
    with DatabaseManager(db) as m:
        tz = m.get_timezone()
        t0, t1 = events["timestamp"].iloc[[0, -1]]
        zone = resolve_pytz(tz)
        lo = datetime.fromtimestamp(t0, zone)
        hi = datetime.fromtimestamp(t1 + 1, zone)
        configs = m.get_configs_for_range(lo, hi)
    return events, tz, configs


def profile_site(events, tz, configs):
    """Profile per config period; keep logged bins only.  Returns (profile, roles by date)."""
    parts, role_rows = [], []
    for cfg in configs:
        roles = parse_detector_roles({k: v for k, v in cfg.items() if not k.startswith("_")})
        lo, hi = cfg["_epoch_start"], cfg["_epoch_end"]
        sub = events[(events["timestamp"] >= lo) & (events["timestamp"] < hi)]
        if sub.empty:
            continue
        p = detector_activity_profile(sub, tz, roles, WINDOWS)
        p = p[p["observed_s"] > 0]
        parts.append((p, roles, sub))
    return parts


def main() -> None:
    flags = {k: [] for k in ("ConfiguredSilent", "LowDetectorHits", "UnconfiguredDetector",
                             "StuckOn", "Chatter", "Failsafe")}
    summary, dist, bursts_all, defaults = [], [], [], []
    for site in SITES:
        events, tz, configs = load(site)
        for p, roles, sub in profile_site(events, tz, configs):
            lab = _labelled(p, roles, "day")
            lab = lab[lab["share"] >= 0.9]
            dist.append(lab.assign(site=site)[["site", "date", "detector", "role", "configured",
                                               "n_act", "n_short", "n_censored", "max_on_s",
                                               "open_on_s", "min_off_gap_s"]])

            def add(rule, grid, f):
                if not f.empty:
                    flags[rule].append(f.assign(site=site, grid=grid))
                summary.append({"site": site, "rule": rule, "grid": grid, "flags": len(f),
                                "judged": int((p["window"] == "day").sum())})

            for share in SHARES:
                th = replace(DEFAULT_THRESHOLDS, min_observed_share=share)
                add("ConfiguredSilent", share, configured_silent(p, roles, th))
            for rule, fn in (("LowDetectorHits", low_detector_hits),
                             ("UnconfiguredDetector", unconfigured_detector),
                             ("Chatter", chatter)):
                field, grid = GRIDS[rule]
                for g in grid:
                    add(rule, g, fn(p, roles, replace(DEFAULT_THRESHOLDS, **{field: g})))
            for x in STUCK_MIN:
                th = replace(DEFAULT_THRESHOLDS,
                             stuck_on_s={r: x * 60.0 for r in DEFAULT_THRESHOLDS.stuck_on_s},
                             stuck_on_default_s=x * 60.0)
                add("StuckOn", x, stuck_on(p, roles, th))

            defaults.append(detector_health_findings(p, roles, sub, tz, reboot_windows=REBOOT)
                            .assign(site=site))
            b = onset_bursts(sub, 6, reassert_s=DEFAULT_THRESHOLDS.reassert_s)
            bursts_all.append(b.assign(site=site, tz=tz))
            for g in GRIDS["Failsafe"][1]:
                th = replace(DEFAULT_THRESHOLDS, burst_min_channels=g)
                add("Failsafe", g, failsafe_findings(sub, tz, roles, th, REBOOT, bursts=b))

    for rule, parts in flags.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        cols = ["site", "grid", "date", "window", "ts", "detector", "phase", "role",
                "severity", "value", "threshold", "message"]
        df.reindex(columns=cols).to_csv(OUT / f"detector_health_{rule}.csv", index=False)
    pd.DataFrame(summary).groupby(["rule", "grid", "site"])["flags"].sum().unstack("site") \
        .fillna(0).astype(int).to_csv(OUT / "detector_health_summary.csv")
    b = pd.concat(bursts_all, ignore_index=True)
    b["local"] = [pd.Timestamp(t, unit="s", tz="UTC").tz_convert(z).strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
                  for t, z in zip(b["ts"], b["tz"])]
    b.drop(columns="tz").to_csv(OUT / "detector_health_bursts.csv", index=False)
    pd.concat(dist, ignore_index=True).to_csv(OUT / "detector_health_daily.csv", index=False)
    print(pd.read_csv(OUT / "detector_health_summary.csv").to_string(index=False))

    d = pd.concat(defaults, ignore_index=True)
    d.to_csv(OUT / "detector_health_defaults.csv", index=False)
    print(d.groupby(["rule", "severity", "site"]).size().unstack("site").fillna(0).astype(int))
    verify(d)


def verify(d: pd.DataFrame) -> None:
    """Every labelled positive is found at the shipped defaults."""
    fs = d[(d["site"] == "201") & (d["rule"] == "Failsafe")]
    t = pd.to_datetime(fs["ts"], unit="s", utc=True).dt.tz_convert("US/Mountain")
    assert (t.dt.strftime("%Y-%m-%d %H:%M") == "2026-03-19 01:55").any(), "201 labelled failsafe"
    silent = set(d.loc[(d["site"] == "201") & (d["rule"] == "ConfiguredSilent"), "detector"])
    assert {60, 63, 64} <= silent, f"201 silent channels: {sorted(silent)}"
    big = d[(d["site"] == "315") & (d["rule"] == "Failsafe") & (d["value"] >= 42)]
    assert len(big) == 4 and (big["severity"] == "high").all(), "315 42-channel bursts"
    unc = set(d.loc[(d["site"] == "315") & (d["rule"] == "UnconfiguredDetector"), "detector"])
    s_d6 = {33, 41, 42, 43, 44, 47, 49, 57, 58, 59, 60, 63, 37, 53}  # design_detector_config_inference.md
    assert unc == s_d6, f"315 unconfigured zones: {sorted(unc)}"
    print("verify: all labelled positives found")


if __name__ == "__main__":
    main()
