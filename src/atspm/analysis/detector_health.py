"""
Detector Health Rules (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

Deterministic rules for *physical* detector states (S-D2), evaluated on the
per-detector activity profile (:func:`~atspm.analysis.detector_activity.
detector_activity_profile`) and, for failsafe and controller faults, on raw
events.  Statistical departures from normal (baseline drift, peer level) are
a separate layer (``traffic_anomaly``) and aren't handled here.

Every rule returns rows of :data:`FINDINGS_SCHEMA`.  Severity labels:
``info`` = expected or known (a scheduled reboot, an unconfigured zone);
``low`` = suspicious, worth a look; ``high`` = the detector is probably
failed and affecting operation.

Judgeable bins
--------------
Rules that read *absence* (configured-silent, low hits, held-on occupancy)
judge a bin only when ``observed_s / bin_s >= min_observed_share``: a log
gap isn't a silent detector.  Rules that read *present* evidence (a measured
long on-duration, actuations on an unconfigured channel, short pulses
against peers) don't need the gate.

Bursts and failsafe
-------------------
A sensor in failsafe drives a block of channels on in one decisecond.
:func:`onset_bursts` finds same-timestamp onsets.  A burst whose channels
had turned off within ``reassert_s`` before it is a **re-assertion**: the
controller logged off and on again for channels that were already on (701
does it at round hours, 201 just after midnight).  Those carry no onset
information and are excluded by that test, not by channel count.

Package Location: src/atspm/analysis/detector_health.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields as _dc_fields, replace
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.timezone import resolve_pytz
from .detector_activity import (
    MAX_SILENCE_S,
    UDOT_AM_WINDOW,
    _build_bins,
    _mark_silences,
)
from .detector_inference import _to_epoch, _windows_clear_of_gaps
from .detector_roles import ROLES

_GAP_CODE = -1
_CODE_DET_OFF = 81
_CODE_DET_ON = 82
_CODE_MAX_OUT = 5

SEVERITIES = ("info", "low", "high")

FINDINGS_SCHEMA = [
    "date", "window", "ts", "detector", "phase", "role", "rule", "severity",
    "value", "threshold", "message",
]
_FINDING_DTYPES = {
    "date": "object", "window": "str", "ts": "float64", "detector": "int64",
    "phase": "Int64", "role": "str", "rule": "str", "severity": "str",
    "value": "float64", "threshold": "float64", "message": "str",
}

BURST_SCHEMA = [
    "ts", "unit", "n_channels", "channels", "released_s", "n_censored",
    "n_reasserted", "reasserted",
]

#: Controller detector-fault codes (Indiana enumerations).  92 (ped detector
#: restored) is left out: the corpus logs it only as a state dump at restart.
CONTROLLER_FAULT_CODES: Mapping[int, Tuple[str, str]] = MappingProxyType({
    83: ("info", "detector restored"),
    84: ("high", "detector fault: other"),
    85: ("high", "detector fault: watchdog"),
    86: ("high", "detector fault: open loop"),
    87: ("high", "detector fault: shorted loop"),
    88: ("high", "detector fault: excessive change"),
    91: ("high", "pedestrian detector failed"),
})

# Roles a detector's on-duration is judged by; the longest legitimate hold
# among a detector's roles applies.  Presence zones hold through red.
_DEFAULT_STUCK_ON_S = MappingProxyType({
    "arrival": 900.0,
    "stop_bar": 900.0,
    "occupancy": 1800.0,
    "pairs": 1800.0,
    "tm": 900.0,
})
# Phase-keyed roles that form chatter peer groups (with tm movements).
_PEER_ROLES = ("arrival", "stop_bar", "occupancy")


@dataclass(frozen=True)
class HealthThresholds:
    """Rule thresholds.  Defaults are the S-D2 corpus calibration.

    Attributes:
        min_observed_share: Absence rules judge a bin only at or above this
            logged share of the bin.
        low_hits_min: LowDetectorHits flags ``0 < n_act < low_hits_min`` per
            day.
        unconfigured_min_act: UnconfiguredDetector flags at or above this
            many actuations per day.
        stuck_on_s: Per-role on-duration limit (seconds).  A detector with
            several roles gets the longest of its roles' limits.
        stuck_on_default_s: Limit for detectors with no role in
            ``stuck_on_s`` (unconfigured channels).
        held_occupancy: A judgeable bin at or above this occupancy is
            held on across the window.
        chatter_peer_ratio: Chatter flags a short-pulse share above this
            multiple of the peer median.
        chatter_share_floor: Peer median floor, so peers with no short
            pulses don't make every short pulse a finding.
        chatter_min_short: Minimum short pulses for a chatter finding.
        chatter_min_act: Minimum measured actuations for a detector to be
            judged or to count as a peer.
        burst_min_channels: Channels (per unit) turning on in one
            decisecond that make a failsafe burst.
        burst_merge_s: Bursts in one unit this close form one episode.
        reassert_s: A channel whose previous off is this recent is
            re-asserted, not a new onset.
        reboot_max_release_s: A burst inside a reboot window is ``info``
            only if its median release is at or under this.
        watchdog_corroborate_s: A watchdog-zone call within this of a
            failsafe episode corroborates it.
        maxout_lead_s: Max-outs this long before an episode count as
            during it.
    """

    min_observed_share: float = 0.9
    low_hits_min: int = 20
    unconfigured_min_act: int = 20
    stuck_on_s: Mapping[str, float] = field(default_factory=lambda: _DEFAULT_STUCK_ON_S)
    stuck_on_default_s: float = 1800.0
    held_occupancy: float = 0.99
    chatter_peer_ratio: float = 3.0
    chatter_share_floor: float = 0.02
    chatter_min_short: int = 50
    chatter_min_act: int = 50
    burst_min_channels: int = 12
    burst_merge_s: float = 10.0
    reassert_s: float = 1.0
    reboot_max_release_s: float = 240.0
    watchdog_corroborate_s: float = 60.0
    maxout_lead_s: float = 30.0


DEFAULT_THRESHOLDS = HealthThresholds()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_findings() -> pd.DataFrame:
    return pd.DataFrame(columns=FINDINGS_SCHEMA).astype(_FINDING_DTYPES)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_findings()
    return df[FINDINGS_SCHEMA].astype(_FINDING_DTYPES).reset_index(drop=True)


def _empty_bursts() -> pd.DataFrame:
    return pd.DataFrame({
        "ts": pd.Series(dtype="float64"), "unit": pd.Series(dtype="str"),
        "n_channels": pd.Series(dtype="int64"), "channels": pd.Series(dtype="object"),
        "released_s": pd.Series(dtype="float64"), "n_censored": pd.Series(dtype="int64"),
        "n_reasserted": pd.Series(dtype="int64"), "reasserted": pd.Series(dtype="bool"),
    })


def _detector_labels(roles: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per detector: joined role label, the single phase (else NA), watchdog-only flag."""
    if roles is None or roles.empty:
        return pd.DataFrame({
            "detector": pd.Series(dtype="int64"), "role": pd.Series(dtype="str"),
            "phase": pd.Series(dtype="Int64"), "watchdog_only": pd.Series(dtype="bool"),
        })
    rank = {r: i for i, r in enumerate(ROLES)}
    r = roles[["detector", "role", "phase"]].copy()
    r["_rank"] = r["role"].map(rank)
    r = r.sort_values(["detector", "_rank"], kind="stable")
    g = r.groupby("detector")
    out = pd.DataFrame({
        "role": g["role"].agg(lambda s: "+".join(dict.fromkeys(s))),
        "_n_phase": g["phase"].nunique(),
        "_phase": g["phase"].first(),
        "watchdog_only": g["role"].agg(lambda s: set(s) == {"watchdog"}),
    }).reset_index()
    out["phase"] = out["_phase"].where(out["_n_phase"] == 1).astype("Int64")
    return out[["detector", "role", "phase", "watchdog_only"]]


def _labelled(profile: pd.DataFrame, roles: Optional[pd.DataFrame], window: Optional[str]) -> pd.DataFrame:
    """Profile rows (one window, or all) with role label, phase and judgeable share."""
    p = profile if window is None else profile[profile["window"] == window]
    p = p.merge(_detector_labels(roles), on="detector", how="left")
    p["role"] = p["role"].fillna("unconfigured")
    p["watchdog_only"] = p["watchdog_only"].fillna(False).astype(bool)
    p["share"] = np.where(p["bin_s"] > 0, p["observed_s"] / p["bin_s"], 0.0)
    return p


def _judgeable(p: pd.DataFrame, th: HealthThresholds) -> pd.Series:
    return p["share"] >= th.min_observed_share


def _rows(p: pd.DataFrame, rule: str, severity, value, threshold, message) -> pd.DataFrame:
    return pd.DataFrame({
        "date": p["date"].to_numpy(), "window": p["window"].to_numpy(), "ts": np.nan,
        "detector": p["detector"].to_numpy(), "phase": p["phase"].to_numpy(),
        "role": p["role"].to_numpy(), "rule": rule, "severity": severity,
        "value": value, "threshold": threshold, "message": message,
    })


def _fmt_hours(s: pd.Series) -> pd.Series:
    return (s / 3600.0).round(1).astype(str)


# ---------------------------------------------------------------------------
# Profile rules
# ---------------------------------------------------------------------------


def configured_silent(profile: pd.DataFrame, roles: Optional[pd.DataFrame],
                      thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
                      window: str = "day") -> pd.DataFrame:
    """ConfiguredSilent: a configured channel with no actuation and no on-time.

    Watchdog-only channels are excluded (they're meant to be silent).  A
    channel stuck on has on-time and no onsets, so it's StuckOn, not silent.

    Args:
        profile: S-D1 activity profile.
        roles: Role table in effect (``parse_detector_roles``).
        thresholds: Rule thresholds.
        window: Profile window judged.

    Returns:
        Findings (:data:`FINDINGS_SCHEMA`), severity ``high``.
    """
    p = _labelled(profile, roles, window)
    p = p[_judgeable(p, thresholds) & p["configured"] & ~p["watchdog_only"]
          & (p["n_act"] == 0) & (p["on_time_s"] == 0)]
    if p.empty:
        return _empty_findings()
    msg = "no actuations in " + _fmt_hours(p["observed_s"]) + " h logged"
    return _finish(_rows(p, "ConfiguredSilent", "high", 0.0, 0.0, msg))


def low_detector_hits(profile: pd.DataFrame, roles: Optional[pd.DataFrame],
                      thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
                      window: str = "day") -> pd.DataFrame:
    """LowDetectorHits: a configured channel actuating, but fewer than k times.

    Args:
        profile: S-D1 activity profile.
        roles: Role table in effect.
        thresholds: ``low_hits_min`` is k.
        window: Profile window judged.

    Returns:
        Findings, severity ``low``.
    """
    k = thresholds.low_hits_min
    p = _labelled(profile, roles, window)
    p = p[_judgeable(p, thresholds) & p["configured"] & ~p["watchdog_only"]
          & (p["n_act"] > 0) & (p["n_act"] < k)]
    if p.empty:
        return _empty_findings()
    msg = p["n_act"].astype(str) + f" actuations (< {k})"
    return _finish(_rows(p, "LowDetectorHits", "low", p["n_act"].to_numpy(float), float(k), msg))


def unconfigured_detector(profile: pd.DataFrame, roles: Optional[pd.DataFrame],
                          thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
                          window: str = "day") -> pd.DataFrame:
    """UnconfiguredDetector: a channel with no role actuating at least k times.

    Args:
        profile: S-D1 activity profile.
        roles: Role table in effect.
        thresholds: ``unconfigured_min_act`` is k.
        window: Profile window judged.

    Returns:
        Findings, severity ``info`` (deliberate zones go on the ignore list).
    """
    k = thresholds.unconfigured_min_act
    p = _labelled(profile, roles, window)
    p = p[~p["configured"] & (p["n_act"] >= k)]
    if p.empty:
        return _empty_findings()
    msg = p["n_act"].astype(str) + " actuations on an unconfigured channel"
    return _finish(_rows(p, "UnconfiguredDetector", "info", p["n_act"].to_numpy(float), float(k), msg))


def _stuck_limit(p: pd.DataFrame, th: HealthThresholds) -> np.ndarray:
    """Longest legitimate hold among each row's roles."""
    limits = {**th.stuck_on_s}
    parts = p["role"].str.split("+")
    exploded = parts.explode()
    lim = exploded.map(lambda r: limits.get(r, np.nan)).groupby(level=0).max()
    return lim.reindex(p.index).fillna(th.stuck_on_default_s).to_numpy(float)


def stuck_on(profile: pd.DataFrame, roles: Optional[pd.DataFrame],
             thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
             duration_window: str = "day") -> pd.DataFrame:
    """StuckOn: an on-duration over the role's limit, or a window held on.

    Two tests:

    - **Duration** (``duration_window`` rows): ``max(max_on_s, open_on_s)``
      over the role limit.  ``open_on_s`` is a censored interval's observed
      lower bound, so a detector stuck on into a log gap still counts.
      Present evidence, so no judgeable gate.
    - **Held** (every window, judgeable bins): occupancy at or above
      ``held_occupancy``.  This catches the days after a long stuck-on
      began, which carry its on-time but not its onset.  A held row is
      dropped when the same detector-date already has a duration finding.

    Watchdog-only channels are excluded (see :func:`failsafe_findings`).

    Args:
        profile: S-D1 activity profile.
        roles: Role table in effect.
        thresholds: ``stuck_on_s``, ``stuck_on_default_s``,
            ``held_occupancy``.
        duration_window: Window whose rows carry the duration test.

    Returns:
        Findings, severity ``high``.
    """
    p = _labelled(profile, roles, None)
    p = p[~p["watchdog_only"]]
    if p.empty:
        return _empty_findings()

    d = p[p["window"] == duration_window].copy()
    d["limit"] = _stuck_limit(d, thresholds)
    d["longest"] = d[["max_on_s", "open_on_s"]].max(axis=1)
    d = d[d["longest"] > d["limit"]]
    open_note = np.where(d["open_on_s"].fillna(-1) >= d["max_on_s"].fillna(-1),
                         " (lower bound, no off logged)", "")
    dur = _rows(d, "StuckOn", "high", d["longest"].to_numpy(float), d["limit"].to_numpy(float),
                "on " + (d["longest"] / 60).round(1).astype(str) + " min" + open_note
                + " > " + (d["limit"] / 60).round(1).astype(str) + " min")

    h = p[_judgeable(p, thresholds) & (p["occupancy"] >= thresholds.held_occupancy)]
    if not h.empty and not dur.empty:
        hit = pd.MultiIndex.from_frame(dur[["date", "detector"]])
        h = h[~pd.MultiIndex.from_frame(h[["date", "detector"]]).isin(hit)]
    held = _rows(h, "StuckOn", "high", h["occupancy"].to_numpy(float), thresholds.held_occupancy,
                 "held on " + (h["occupancy"] * 100).round(1).astype(str) + "% of "
                 + h["window"] + " window")
    parts = [x for x in (dur, held) if not x.empty]
    return _finish(pd.concat(parts, ignore_index=True)) if parts else _empty_findings()


def _peer_groups(roles: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Membership ``detector, group``: ``tm:{movement}`` and ``{role}:P{phase}``."""
    if roles is None or roles.empty:
        return pd.DataFrame({"detector": pd.Series(dtype="int64"), "group": pd.Series(dtype="str")})
    tm = roles[roles["role"] == "tm"]
    ph = roles[roles["role"].isin(_PEER_ROLES)]
    groups = pd.concat([
        pd.DataFrame({"detector": tm["detector"], "group": "tm:" + tm["movement"].astype(str)}),
        pd.DataFrame({"detector": ph["detector"],
                      "group": ph["role"] + ":P" + ph["phase"].astype(str)}),
    ], ignore_index=True)
    return groups.drop_duplicates().reset_index(drop=True)


def chatter(profile: pd.DataFrame, roles: Optional[pd.DataFrame],
            thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
            window: str = "day") -> pd.DataFrame:
    """Chatter: a short-pulse share well above the detector's peers'.

    Never an absolute share: healthy count loops run 5–85 % short pulses
    depending on the loop.  The share is ``n_short / measured actuations``;
    peers are the other members of the same ``TM_*`` movement or the same
    phase and role, with at least ``chatter_min_act`` measured actuations
    that day.  A detector in several groups is compared to its most lenient
    group (the highest peer median).  Flag when the share exceeds
    ``chatter_peer_ratio × max(peer median, chatter_share_floor)`` with at
    least ``chatter_min_short`` short pulses.  Detectors with no qualifying
    peer aren't judged.  Severity is ``low``: ``min_off_gap_s`` is reported
    but can't raise it, because 52–100 % of healthy corpus detector-days
    have an off-gap under 0.2 s.

    Args:
        profile: S-D1 activity profile.
        roles: Role table in effect.
        thresholds: The ``chatter_*`` fields.
        window: Profile window judged.

    Returns:
        Findings, severity ``low``.
    """
    th = thresholds
    p = _labelled(profile, roles, window)
    p["measured"] = p["n_act"] - p["n_censored"]
    p = p[p["measured"] >= th.chatter_min_act]
    groups = _peer_groups(roles)
    if p.empty or groups.empty:
        return _empty_findings()
    p["sshare"] = p["n_short"] / p["measured"]

    m = p[["date", "detector", "sshare"]].merge(groups, on="detector")
    pairs = m.merge(m, on=["date", "group"], suffixes=("", "_peer"))
    pairs = pairs[pairs["detector"] != pairs["detector_peer"]]
    if pairs.empty:
        return _empty_findings()
    peer_med = (pairs.groupby(["date", "detector", "group"])["sshare_peer"].median()
                .groupby(["date", "detector"]).max().rename("peer_med").reset_index())
    q = p.merge(peer_med, on=["date", "detector"])
    q["limit"] = th.chatter_peer_ratio * np.maximum(q["peer_med"], th.chatter_share_floor)
    q = q[(q["sshare"] > q["limit"]) & (q["n_short"] >= th.chatter_min_short)]
    if q.empty:
        return _empty_findings()
    msg = ((q["sshare"] * 100).round(1).astype(str) + "% short pulses (" + q["n_short"].astype(str)
           + "), peers " + (q["peer_med"] * 100).round(1).astype(str) + "%; min off-gap "
           + q["min_off_gap_s"].round(2).astype(str) + " s")
    return _finish(_rows(q, "Chatter", "low",
                         q["sshare"].to_numpy(float), q["limit"].to_numpy(float), msg))


# ---------------------------------------------------------------------------
# Event rules
# ---------------------------------------------------------------------------


def _events_epoch(events_df: pd.DataFrame) -> pd.DataFrame:
    """Sorted epoch events, unmarked silences marked as gaps (as in S-D1)."""
    ev = pd.DataFrame({
        "timestamp": _to_epoch(events_df["timestamp"]),
        "event_code": events_df["event_code"].to_numpy(),
        "parameter": events_df["parameter"].to_numpy(),
    }).sort_values("timestamp", kind="stable").reset_index(drop=True)
    return _mark_silences(ev, MAX_SILENCE_S)


def _local_dates(ts: np.ndarray, zone) -> np.ndarray:
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert(zone).date


def controller_fault_findings(events_df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    """ControllerFault: the controller's own detector fault codes, passed through.

    One finding per local date, channel and code (:data:`CONTROLLER_FAULT_CODES`),
    with the count as ``value`` and the first occurrence as ``ts``.

    Args:
        events_df: Raw events.
        tz: IANA zone of the intersection.

    Returns:
        Findings; severity from :data:`CONTROLLER_FAULT_CODES`.
    """
    if events_df is None or events_df.empty:
        return _empty_findings()
    ev = _events_epoch(events_df)
    ev = ev[ev["event_code"].isin(list(CONTROLLER_FAULT_CODES))]
    if ev.empty:
        return _empty_findings()
    ev["date"] = _local_dates(ev["timestamp"].to_numpy(float), resolve_pytz(tz))
    g = ev.groupby(["date", "parameter", "event_code"])["timestamp"].agg(["size", "min"]).reset_index()
    sev = g["event_code"].map(lambda c: CONTROLLER_FAULT_CODES[c][0])
    text = g["event_code"].map(lambda c: CONTROLLER_FAULT_CODES[c][1])
    return _finish(pd.DataFrame({
        "date": g["date"], "window": "day", "ts": g["min"], "detector": g["parameter"],
        "phase": pd.NA, "role": np.where(g["event_code"] == 91, "ped", "detector"),
        "rule": "ControllerFault", "severity": sev, "value": g["size"].astype(float),
        "threshold": np.nan,
        "message": "code " + g["event_code"].astype(str) + " " + text + " ×" + g["size"].astype(str),
    }))


def onset_bursts(events_df: pd.DataFrame, min_channels: int = 6,
                 units: Optional[Mapping[str, Iterable[int]]] = None,
                 reassert_s: float = 1.0) -> pd.DataFrame:
    """Channels turning on in the same decisecond, per unit.

    For each onset (code 82) in a burst, its *release* is the time to the
    channel's next event when that event is an off (81) with no gap marker
    between; otherwise (another on, a gap marker, a silence over
    :data:`~atspm.analysis.detector_activity.MAX_SILENCE_S`, the data end)
    the release is censored.  A channel is *re-asserted* when its previous event is an
    off within ``reassert_s`` before the onset, with no gap marker between.

    Args:
        events_df: Raw events (``timestamp``, ``event_code``, ``parameter``).
        min_channels: Smallest burst returned (distinct channels in a unit).
        units: ``{unit: channels}``.  Channels in no unit form the unit
            ``"unassigned"``.  ``None`` puts every channel in unit ``"all"``.
        reassert_s: Re-assertion window (seconds).

    Returns:
        DataFrame with :data:`BURST_SCHEMA` columns, sorted by ``ts``, unit::

            ts            float64 – UTC epoch seconds of the burst
            unit          str
            n_channels    int64
            channels      object  – sorted tuple of channel ids
            released_s    float64 – median release, censored channels
                                    counting as never released; NaN when
                                    that median is censored
            n_censored    int64   – channels with a censored release
            n_reasserted  int64
            reasserted    bool    – more than half the channels re-asserted
    """
    if events_df is None or events_df.empty:
        return _empty_bursts()
    ev = _events_epoch(events_df)
    gaps = ev.loc[ev["event_code"] == _GAP_CODE, "timestamp"].to_numpy(float)
    det = ev[ev["event_code"].isin((_CODE_DET_ON, _CODE_DET_OFF))].sort_values(
        ["parameter", "timestamp"], kind="stable")
    if det.empty:
        return _empty_bursts()
    same = det["parameter"].shift(-1) == det["parameter"]
    same_prev = det["parameter"].shift(1) == det["parameter"]
    det = det.assign(
        next_t=det["timestamp"].shift(-1).where(same),
        next_code=det["event_code"].shift(-1).where(same),
        prev_t=det["timestamp"].shift(1).where(same_prev),
        prev_code=det["event_code"].shift(1).where(same_prev),
    )
    on = det[det["event_code"] == _CODE_DET_ON].copy()
    on["tick"] = np.round(on["timestamp"].to_numpy(float) * 10).astype(np.int64)

    if units is None:
        on["unit"] = "all"
    else:
        lookup = {int(ch): str(u) for u, chans in units.items() for ch in chans}
        on["unit"] = on["parameter"].map(lambda c: lookup.get(int(c), "unassigned"))

    size = on.groupby(["unit", "tick"])["parameter"].transform("nunique")
    on = on[size >= min_channels]
    if on.empty:
        return _empty_bursts()

    t = on["timestamp"].to_numpy(float)
    nxt = on["next_t"].to_numpy(float)
    has_off = (on["next_code"] == _CODE_DET_OFF).to_numpy()
    clear = np.zeros(len(on), bool)
    if has_off.any():
        clear[has_off] = _windows_clear_of_gaps(t[has_off], nxt[has_off], gaps)
    on["rel"] = np.where(clear, nxt - t, np.inf)
    on["cens"] = ~clear
    prv = on["prev_t"].to_numpy(float)
    was_off = (on["prev_code"] == _CODE_DET_OFF).to_numpy() & (t - prv <= reassert_s + 1e-6)
    rclear = np.zeros(len(on), bool)
    if was_off.any():
        rclear[was_off] = _windows_clear_of_gaps(prv[was_off], t[was_off], gaps)
    on["reas"] = rclear

    g = on.groupby(["unit", "tick"])
    out = pd.DataFrame({
        "ts": g["timestamp"].min(),
        "n_channels": g["parameter"].nunique(),
        "channels": g["parameter"].agg(lambda s: tuple(sorted(int(c) for c in set(s)))),
        "released_s": g["rel"].median(),
        "n_censored": g["cens"].sum(),
        "n_reasserted": g["reas"].sum(),
    }).reset_index()
    out["released_s"] = out["released_s"].replace(np.inf, np.nan)
    out["reasserted"] = out["n_reasserted"] * 2 > out["n_channels"]
    out = out.sort_values(["ts", "unit"], kind="stable").reset_index(drop=True)
    return out[BURST_SCHEMA].astype({"ts": "float64", "unit": "str", "n_channels": "int64",
                                     "released_s": "float64", "n_censored": "int64",
                                     "n_reasserted": "int64", "reasserted": "bool"})


def _in_windows(ts: np.ndarray, windows: Optional[Mapping[str, Tuple[str, str]]], zone) -> np.ndarray:
    """Name of the local window holding each *ts*, else ``""``."""
    names = np.full(len(ts), "", dtype=object)
    if not windows or not len(ts):
        return names
    dates = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in _local_dates(ts, zone)}))
    bins = _build_bins(dates, windows, zone)
    s, e = bins["bin_start"].to_numpy(), bins["bin_end"].to_numpy()
    hit = (ts[:, None] >= s[None, :]) & (ts[:, None] < e[None, :])
    any_hit = hit.any(axis=1)
    names[any_hit] = bins["window"].to_numpy()[hit[any_hit].argmax(axis=1)]
    return names


def failsafe_findings(events_df: pd.DataFrame, tz: Optional[str],
                      roles: Optional[pd.DataFrame] = None,
                      thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
                      reboot_windows: Optional[Mapping[str, Tuple[str, str]]] = None,
                      units: Optional[Mapping[str, Iterable[int]]] = None,
                      bursts: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Failsafe: block-on bursts, and watchdog-zone calls corroborated by them.

    Bursts of at least ``burst_min_channels`` in one unit, not re-asserted,
    are merged into episodes (``burst_merge_s``).  An episode is:

    - ``info``: inside a reboot window, median release at or under
      ``reboot_max_release_s``;
    - ``low``: inside a reboot window, release not observed (log gap or
      data end);
    - ``high``: otherwise (outside every window, or held too long).

    Watchdog zones (role ``watchdog``) are evidence, never a verdict.  A call
    within ``watchdog_corroborate_s`` of an episode is noted on it; any other
    call is a ``WatchdogCall`` finding (``low``, a suspected false call),
    one per channel and local date.

    Args:
        events_df: Raw events, including gap markers and, for the max-out
            note, code 5.
        tz: IANA zone of the intersection.
        roles: Role table in effect (for watchdog channels).
        thresholds: The ``burst_*``, ``reassert_s``, ``reboot_*``,
            ``watchdog_*`` and ``maxout_lead_s`` fields.
        reboot_windows: ``{name: ("HH:MM", "HH:MM")}`` local windows of
            scheduled sensor reboots (from ``WD:`` config).  None: no window.
        units: ``{unit: channels}`` detection-unit grouping.  None: every
            channel is one unit (the fallback until a site has unit rows).
        bursts: Precomputed :func:`onset_bursts` output, to skip recomputing.

    Returns:
        Findings: ``Failsafe`` rows (``detector`` -1, ``value`` channel
        count, ``ts`` episode start) and ``WatchdogCall`` rows.
    """
    th = thresholds
    if events_df is None or events_df.empty:
        return _empty_findings()
    zone = resolve_pytz(tz)
    ev = _events_epoch(events_df)
    if bursts is None:
        bursts = onset_bursts(ev, th.burst_min_channels, units, th.reassert_s)
    b = bursts[(~bursts["reasserted"]) & (bursts["n_channels"] >= th.burst_min_channels)]
    b = b.sort_values(["unit", "ts"], kind="stable")

    wd = set()
    if roles is not None and not roles.empty:
        wd = set(int(d) for d in roles.loc[roles["role"] == "watchdog", "detector"])
    wd_on = ev[(ev["event_code"] == _CODE_DET_ON) & ev["parameter"].isin(wd)]
    wd_t = wd_on["timestamp"].to_numpy(float)

    parts = []
    episodes = pd.DataFrame()
    if not b.empty:
        new = (b["unit"] != b["unit"].shift()) | (b["ts"].diff() > th.burst_merge_s)
        b = b.assign(ep=new.cumsum())
        g = b.groupby("ep")
        episodes = pd.DataFrame({
            "ts": g["ts"].min(), "unit": g["unit"].first(),
            "channels": g["channels"].agg(lambda s: tuple(sorted(set().union(*s)))),
            "released_s": g["released_s"].agg(lambda s: s.max(skipna=False)),
            "n_bursts": g.size(),
        }).reset_index(drop=True)
        episodes["n_channels"] = episodes["channels"].map(len)
        ts = episodes["ts"].to_numpy(float)
        win = _in_windows(ts, reboot_windows, zone)
        rel = episodes["released_s"].to_numpy(float)
        in_win = win != ""
        sev = np.where(in_win & (rel <= th.reboot_max_release_s), "info",
                       np.where(in_win & np.isnan(rel), "low", "high"))

        # Max-outs around each episode.
        mo = ev[ev["event_code"] == _CODE_MAX_OUT]
        mo_t, mo_p = mo["timestamp"].to_numpy(float), mo["parameter"].to_numpy()
        end = ts + np.where(np.isnan(rel), 0.0, rel)
        maxed = [sorted({int(x) for x in mo_p[(mo_t >= s - th.maxout_lead_s) & (mo_t <= e)]})
                 for s, e in zip(ts, end)]

        # Watchdog-zone calls near each episode.
        near = np.abs(wd_t[:, None] - ts[None, :]) <= th.watchdog_corroborate_s
        corroborated = [sorted({int(x) for x in wd_on["parameter"].to_numpy()[near[:, i]]})
                        for i in range(len(ts))]

        local = pd.to_datetime(ts, unit="s", utc=True).tz_convert(zone)
        clock = local.strftime("%H:%M:%S.%f").str[:-5]
        msg = [
            f"{n} channels on at {c} (unit {u}, {k} burst{'s' if k > 1 else ''}); "
            + (f"median release {r:.1f} s" if not np.isnan(r) else "release not observed")
            + (f"; inside reboot window '{w}'" if w else "")
            + (f"; phases maxed out: {', '.join(map(str, m))}" if m else "")
            + (f"; watchdog zones called: {', '.join(map(str, wz))}" if wz else "")
            for n, c, u, k, r, w, m, wz in zip(
                episodes["n_channels"], clock, episodes["unit"], episodes["n_bursts"],
                rel, win, maxed, corroborated)
        ]
        parts.append(pd.DataFrame({
            "date": local.date, "window": "day", "ts": ts, "detector": -1, "phase": pd.NA,
            "role": "unit:" + episodes["unit"], "rule": "Failsafe", "severity": sev,
            "value": episodes["n_channels"].astype(float), "threshold": float(th.burst_min_channels),
            "message": msg,
        }))

    # Uncorroborated watchdog calls.
    if len(wd_t):
        ep_t = episodes["ts"].to_numpy(float) if not episodes.empty else np.empty(0)
        lone = ~(np.abs(wd_t[:, None] - ep_t[None, :]) <= th.watchdog_corroborate_s).any(axis=1)
        w = wd_on[lone].assign(date=_local_dates(wd_t[lone], zone))
        if not w.empty:
            g = w.groupby(["date", "parameter"])["timestamp"].agg(["size", "min"]).reset_index()
            parts.append(pd.DataFrame({
                "date": g["date"], "window": "day", "ts": g["min"], "detector": g["parameter"],
                "phase": pd.NA, "role": "watchdog", "rule": "WatchdogCall", "severity": "low",
                "value": g["size"].astype(float), "threshold": np.nan,
                "message": g["size"].astype(str)
                + " watchdog-zone call(s) with no failsafe burst (suspected false call)",
            }))

    parts = [x for x in parts if not x.empty]
    if not parts:
        return _empty_findings()
    return _finish(pd.concat(parts, ignore_index=True))


# ---------------------------------------------------------------------------
# All rules
# ---------------------------------------------------------------------------


def detector_health_findings(
    profile: pd.DataFrame,
    roles: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    tz: Optional[str] = None,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
    reboot_windows: Optional[Mapping[str, Tuple[str, str]]] = None,
    units: Optional[Mapping[str, Iterable[int]]] = None,
) -> pd.DataFrame:
    """Run every deterministic detector-health rule.

    Profile rules: ConfiguredSilent, LowDetectorHits, UnconfiguredDetector,
    StuckOn, Chatter.  Event rules (only when *events_df* is given):
    Failsafe / WatchdogCall and ControllerFault.  LowDetectorHits is
    dropped on a detector-date that has a StuckOn finding.

    Args:
        profile: S-D1 activity profile (must include the ``day`` window).
        roles: Role table in effect.
        events_df: Raw events for the same period, or None.
        tz: IANA zone (needed with *events_df*).
        thresholds: Rule thresholds.
        reboot_windows: Scheduled sensor-reboot windows (local).
        units: Detection-unit grouping for failsafe; None = one unit.

    Returns:
        Findings (:data:`FINDINGS_SCHEMA`), sorted by date, detector, rule.
    """
    parts = [
        configured_silent(profile, roles, thresholds),
        low_detector_hits(profile, roles, thresholds),
        unconfigured_detector(profile, roles, thresholds),
        stuck_on(profile, roles, thresholds),
        chatter(profile, roles, thresholds),
    ]
    if events_df is not None and not events_df.empty:
        parts.append(failsafe_findings(events_df, tz, roles, thresholds, reboot_windows, units))
        parts.append(controller_fault_findings(events_df, tz))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _empty_findings()
    out = pd.concat(parts, ignore_index=True)
    # A stuck-on detector-date has few onsets by construction; StuckOn says why.
    stuck = out.loc[out["rule"] == "StuckOn", ["date", "detector"]]
    if not stuck.empty:
        key = pd.MultiIndex.from_frame(out[["date", "detector"]])
        out = out[~((out["rule"] == "LowDetectorHits") & key.isin(pd.MultiIndex.from_frame(stuck)))]
    out = out.sort_values(["date", "detector", "rule", "window", "ts"], kind="stable")
    return _finish(out)


# ---------------------------------------------------------------------------
# int_cfg WD-key parsing (config convention -> core arguments)
# ---------------------------------------------------------------------------
# Pure: a config dict in, plain structures out, so the S-D4 shell reads
# int_cfg once and hands the core already-parsed windows, units and
# thresholds.  Convention (owner, 2026-10-03), all under the ``WD:`` row
# category of ``int_cfg.csv`` (``_transform_config_column`` prefixes ``WD_``):
#
#   WD_Reboot     "23:58-24:00,00:00-00:03"               scheduled reboot windows
#   WD_PM         "16:30-17:30"                            peak-traffic window
#   WD_AM         "01:00-05:00"                            quiet-window override
#   WD_Units      "evo:[17-28],[29-40]; currux:[17-20]"    failure units, grouped by type
#   WD_Ignore     "52:StuckOn,60:ConfiguredSilent"         findings to suppress in reports
#   WD_Thresholds "low_hits_min=10,stuck_on_s.arrival=600" HealthThresholds overrides
#
# Malformed input raises ``ValueError`` (the shell surfaces it per
# intersection); it never prints, so the core stays side-effect free.

_DAY_WINDOW: Tuple[str, str] = ("00:00", "24:00")
_HHMM_RANGE_RE = re.compile(r"^(\d{1,2}:\d{2})-(\d{1,2}:\d{2})$")
_CHANNEL_GROUP_RE = re.compile(r"\[([^\]]*)\]")
_SEVERITY_RANK: Mapping[str, int] = MappingProxyType(
    {s: i for i, s in enumerate(SEVERITIES)}
)


def _is_blank(raw) -> bool:
    return raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip()


def _parse_window_ranges(raw) -> List[Tuple[str, str]]:
    """Comma-separated ``HH:MM-HH:MM`` local ranges -> list of ``(start, end)``."""
    out: List[Tuple[str, str]] = []
    if _is_blank(raw):
        return out
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = _HHMM_RANGE_RE.match(tok)
        if not m:
            raise ValueError(f"bad time range {tok!r} (want HH:MM-HH:MM)")
        out.append((m.group(1), m.group(2)))
    return out


def wd_reboot_windows(config: Optional[Mapping]) -> Optional[Dict[str, Tuple[str, str]]]:
    """Parse ``WD_Reboot`` into named failsafe reboot windows.

    Args:
        config: Active config dict (keys as stored, e.g. ``WD_Reboot``).

    Returns:
        ``{"reboot_0": (start, end), ...}``, or ``None`` when the key is
        absent or blank (no scheduled windows).
    """
    ranges = _parse_window_ranges((config or {}).get("WD_Reboot"))
    if not ranges:
        return None
    return {f"reboot_{i}": r for i, r in enumerate(ranges)}


def wd_profile_windows(config: Optional[Mapping]) -> Dict[str, Tuple[str, str]]:
    """Windows to profile: ``day`` always; ``am`` (``WD_AM`` or 01:00-05:00);
    ``pm`` only when ``WD_PM`` is set.

    Args:
        config: Active config dict.

    Returns:
        ``{name: ("HH:MM", "HH:MM")}`` for :func:`detector_activity_profile`.
    """
    config = config or {}
    windows: Dict[str, Tuple[str, str]] = {"day": _DAY_WINDOW}
    am = _parse_window_ranges(config.get("WD_AM"))
    windows["am"] = am[0] if am else UDOT_AM_WINDOW
    pm = _parse_window_ranges(config.get("WD_PM"))
    if pm:
        windows["pm"] = pm[0]
    return windows


def _parse_channel_list(body: str) -> List[int]:
    """``"17-20,25"`` -> ``[17, 18, 19, 20, 25]``."""
    chans: List[int] = []
    for tok in body.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            chans.extend(range(int(lo), int(hi) + 1))
        else:
            chans.append(int(tok))
    return chans


def wd_units(
    config: Optional[Mapping],
) -> Tuple[Optional[Dict[str, List[int]]], Dict[str, str]]:
    """Parse ``WD_Units`` into failure-unit channel groups and their types.

    Format ``"evo:[17-28],[29-40]; currux:[17-20],[21-24]"``: ``;`` splits
    unit *types*, each ``[...]`` is one physical device (an EVO sensor, a
    Currux/Thunder camera) — the unit that fails together in a failsafe
    block-on burst.  Channels inside a group are comma-separated ints or
    ``lo-hi`` ranges.

    Args:
        config: Active config dict.

    Returns:
        ``(units, types)``.  ``units`` maps a unit name to its channels
        (``{"evo_0": [17..28], "currux_0": [...]}``) for
        :func:`failsafe_findings`; ``types`` maps each unit name to its type.
        ``units`` is ``None`` when the key is absent — failsafe then treats the
        whole intersection as one unit.
    """
    raw = (config or {}).get("WD_Units")
    if _is_blank(raw):
        return None, {}
    units: Dict[str, List[int]] = {}
    types: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for seg in str(raw).split(";"):
        seg = seg.strip()
        if not seg:
            continue
        if ":" not in seg:
            raise ValueError(f"bad WD_Units segment {seg!r} (want 'type:[...],[...]')")
        utype, rest = (s.strip() for s in seg.split(":", 1))
        groups = _CHANNEL_GROUP_RE.findall(rest)
        if not groups:
            raise ValueError(f"no [channel] groups in WD_Units segment {seg!r}")
        for body in groups:
            idx = counts.get(utype, 0)
            counts[utype] = idx + 1
            name = f"{utype}_{idx}"
            units[name] = _parse_channel_list(body)
            types[name] = utype
    return units, types


def wd_ignore(config: Optional[Mapping]) -> List[Tuple[int, str]]:
    """Parse ``WD_Ignore`` = ``"52:StuckOn,60:ConfiguredSilent"``.

    Args:
        config: Active config dict.

    Returns:
        ``[(detector, rule), ...]`` (detector ``-1`` matches phase-level
        findings).  Empty when the key is absent.
    """
    raw = (config or {}).get("WD_Ignore")
    out: List[Tuple[int, str]] = []
    if _is_blank(raw):
        return out
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"bad WD_Ignore entry {tok!r} (want 'det:rule')")
        det, rule = (s.strip() for s in tok.split(":", 1))
        out.append((int(det), rule))
    return out


def apply_ignore(
    findings: pd.DataFrame, ignore: Iterable[Tuple[int, str]]
) -> pd.DataFrame:
    """Drop findings whose ``(detector, rule)`` is in *ignore*.

    Rule match is case-insensitive.  This is applied at read/report time, not
    at write time, so an ignored detector's findings are still recorded in the
    ``detector_findings`` table.

    Args:
        findings: Findings frame (:data:`FINDINGS_SCHEMA`).
        ignore: ``(detector, rule)`` pairs, e.g. from :func:`wd_ignore`.

    Returns:
        A filtered copy (the input when *ignore* is empty).
    """
    keys = {(int(d), str(r).lower()) for d, r in ignore}
    if findings is None or findings.empty or not keys:
        return findings
    pairs = pd.Series(
        list(zip(findings["detector"].astype("int64"),
                 findings["rule"].astype(str).str.lower())),
        index=findings.index,
    )
    return findings.loc[~pairs.isin(keys)].reset_index(drop=True)


def wd_thresholds(
    config: Optional[Mapping], base: HealthThresholds = DEFAULT_THRESHOLDS
) -> HealthThresholds:
    """Apply ``WD_Thresholds`` overrides onto *base*.

    Format ``"low_hits_min=10,chatter_peer_ratio=2,stuck_on_s.arrival=600"``:
    comma-separated ``field=value`` using :class:`HealthThresholds` field names
    verbatim.  A dotted ``mapping_field.key=value`` updates one entry of a
    mapping field (only ``stuck_on_s`` today).  Values are coerced to the
    field's type.

    Args:
        config: Active config dict.
        base: Thresholds to override (defaults to the corpus calibration).

    Returns:
        A new :class:`HealthThresholds` (``base`` unchanged when the key is
        absent).

    Raises:
        ValueError: An entry is malformed, names an unknown field, or dots a
            non-mapping field.
    """
    raw = (config or {}).get("WD_Thresholds")
    if _is_blank(raw):
        return base
    valid = {f.name for f in _dc_fields(base)}
    scalar: Dict[str, object] = {}
    mapping: Dict[str, Dict[str, float]] = {}
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"bad WD_Thresholds entry {tok!r} (want field=value)")
        key, val = (s.strip() for s in tok.split("=", 1))
        if "." in key:
            fld, subkey = key.split(".", 1)
            if fld not in valid or not isinstance(getattr(base, fld), Mapping):
                raise ValueError(f"WD_Thresholds: {fld!r} is not a mapping field")
            mapping.setdefault(fld, {})[subkey] = float(val)
        elif key in valid:
            current = getattr(base, key)
            if isinstance(current, bool):
                scalar[key] = val.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                scalar[key] = int(float(val))
            else:
                scalar[key] = float(val)
        else:
            raise ValueError(f"WD_Thresholds: unknown field {key!r}")
    for fld, sub in mapping.items():
        merged = dict(getattr(base, fld))
        merged.update(sub)
        scalar[fld] = merged
    return replace(base, **scalar) if scalar else base


def filter_min_severity(
    findings: pd.DataFrame, min_severity: str = "low"
) -> pd.DataFrame:
    """Keep findings at or above *min_severity* (the report/CSV view).

    The ``detector_findings`` table always keeps every severity; this only
    shapes what a report shows.

    Args:
        findings: Findings frame (:data:`FINDINGS_SCHEMA`).
        min_severity: One of :data:`SEVERITIES`.

    Returns:
        A filtered copy.
    """
    if findings is None or findings.empty:
        return findings
    floor = _SEVERITY_RANK.get(min_severity, _SEVERITY_RANK["low"])
    ranks = findings["severity"].map(_SEVERITY_RANK)
    return findings.loc[ranks >= floor].reset_index(drop=True)


def severity_exit_code(
    findings: pd.DataFrame, min_severity: str = "low"
) -> int:
    """Process exit code from the highest reported severity.

    *findings* should already be ignore-filtered.  Only findings at or above
    *min_severity* count.

    Args:
        findings: Findings frame (:data:`FINDINGS_SCHEMA`).
        min_severity: Severity floor.

    Returns:
        ``0`` nothing reported, ``1`` highest is ``low``, ``2`` highest is
        ``high``.  ``info`` never raises it above ``0``.
    """
    floor = _SEVERITY_RANK.get(min_severity, _SEVERITY_RANK["low"])
    if findings is None or findings.empty:
        return 0
    ranks = findings["severity"].map(_SEVERITY_RANK).dropna()
    ranks = ranks[ranks >= floor]
    return int(ranks.max()) if not ranks.empty else 0
