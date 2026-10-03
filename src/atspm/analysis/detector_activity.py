"""
Per-Detector Activity Profile (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

Summarises each detector channel's actuation behaviour per
``(detector, bin)``.  It's the input to the detector-health rules (S-D2):
stuck-on, chatter, silence, low hits and baseline drift.  Bins are local
calendar days crossed with named sub-day windows (``"day"`` = the whole day;
UDOT's Watchdog reads an AM window of 01:00–05:00, :data:`UDOT_AM_WINDOW`).
Bin edges are local wall-clock times resolved through
:func:`~atspm.utils.timezone.resolve_pytz`, so a DST day bin is 23 h or 25 h.

Intervals and censoring
-----------------------
On-intervals come from :func:`~atspm.analysis.detectors._reconstruct_intervals`
(code 82 on / 81 off, gap markers close an open interval).  An interval is
**censored** when it never logged its own off.  That happens either because
a gap marker (``event_code = -1``, clock-step markers included) falls between
on and off, or because the data ends while it's on.  A censored interval:

- counts as an actuation (``n_act``) and in ``n_censored``;
- is never *measured*: it adds nothing to ``max_on_s``, ``p95_on_s`` or
  ``n_short``, and no off-gap is taken from its end;
- still adds its *observed* span ``[on, censor point)`` to ``on_time_s``.
  The detector was seen to be on there.  Ingestion places a gap marker
  0.1 s after the last event before the gap, so the span stops where the
  log stops.  ``open_on_s`` reports that span's length, a lower bound on
  the true duration, so a detector stuck on into a gap or the end of the
  data still shows up.

Unobserved time is excluded from the occupancy denominator.  That's the
span from each gap marker to the next logged event, and everything before
the first event or after the last.  So ``occupancy = on_time_s /
observed_s``, and a bin with no observed time has ``occupancy`` NaN and
``on_at_start`` / ``on_at_end`` NA.  An off-gap is the time from a measured
off to the same detector's next on.  It is dropped if a gap marker lies
between them.

Interval statistics (``n_act``, ``n_censored``, ``max_on_s``, ``p95_on_s``,
``n_short``, ``open_on_s``, ``min_off_gap_s``) belong to the bin holding the
on-time (for an off-gap, the on that ends it), using the interval's full
duration.  ``on_time_s`` is clipped to each bin, so a 40-hour stuck-on
interval adds 1 actuation and a 40 h ``max_on_s`` to its start day, and
on-time to every day it covers.

Package Location: src/atspm/analysis/detector_activity.py
"""

from __future__ import annotations

import datetime as _dt
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.timezone import resolve_pytz
from .detector_inference import _to_epoch, _windows_clear_of_gaps
from .detectors import _reconstruct_intervals

_GAP_CODE = -1
_CODE_DET_OFF = 81
_CODE_DET_ON = 82

#: UDOT Watchdog's overnight window (local), used by its stuck-on style rules.
UDOT_AM_WINDOW: Tuple[str, str] = ("01:00", "05:00")
#: Default bins: the whole local day.
DEFAULT_WINDOWS: Mapping[str, Tuple[str, str]] = {"day": ("00:00", "24:00")}

# Timestamps are logged in tenths; absorb float noise when comparing to 0.1 s.
_SHORT_EPS = 1e-3

ACTIVITY_SCHEMA = [
    "date", "window", "bin_start", "bin_end", "bin_s", "observed_s",
    "detector", "configured", "n_act", "n_censored", "on_time_s", "occupancy",
    "max_on_s", "p95_on_s", "n_short", "min_off_gap_s", "open_on_s",
    "on_at_start", "on_at_end",
]
_DTYPES = {
    "date": "object", "window": "str", "bin_start": "float64",
    "bin_end": "float64", "bin_s": "float64", "observed_s": "float64",
    "detector": "int64", "configured": "bool", "n_act": "int64",
    "n_censored": "int64", "on_time_s": "float64", "occupancy": "float64",
    "max_on_s": "float64", "p95_on_s": "float64", "n_short": "int64",
    "min_off_gap_s": "float64", "open_on_s": "float64",
    "on_at_start": "boolean", "on_at_end": "boolean",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTIVITY_SCHEMA).astype(_DTYPES)


def _covered(starts: np.ndarray, ends: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Length of disjoint sorted spans ``[starts, ends)`` lying before each *t*."""
    if not len(starts):
        return np.zeros(len(t))
    cum = np.concatenate(([0.0], np.cumsum(ends - starts)))
    k = np.searchsorted(starts, t, side="right")
    overhang = np.where(k > 0, np.clip(ends[np.maximum(k - 1, 0)] - t, 0.0, None), 0.0)
    return cum[k] - overhang


def _inside(starts: np.ndarray, ends: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Whether each *t* lies in one of the disjoint sorted spans ``[starts, ends)``."""
    if not len(starts):
        return np.zeros(len(t), dtype=bool)
    k = np.searchsorted(starts, t, side="right") - 1
    return (k >= 0) & (t < ends[np.maximum(k, 0)])


def _observed_spans(t_ev: np.ndarray, gaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Spans of logged time: first..last event minus each gap marker → next event."""
    if not len(t_ev):
        return np.empty(0), np.empty(0)
    first, last = t_ev[0], t_ev[-1]
    g = gaps[(gaps > first) & (gaps < last)]
    nxt = t_ev[np.searchsorted(t_ev, g, side="right")]
    # Consecutive markers before one event share its time; keep the earliest.
    nxt, first_idx = np.unique(nxt, return_index=True)
    g = g[first_idx]
    starts = np.concatenate(([first], nxt))
    ends = np.concatenate((g, [last]))
    keep = ends > starts
    return starts[keep], ends[keep]


def _parse_clock(s: str) -> pd.Timedelta:
    hh, mm = s.split(":")
    return pd.Timedelta(hours=int(hh), minutes=int(mm))


def _build_bins(
    dates: pd.DatetimeIndex, windows: Mapping[str, Tuple[str, str]], tz
) -> pd.DataFrame:
    """One row per (date, window): local wall-clock edges as UTC epoch seconds.

    A nonexistent edge (inside a spring-forward hour) shifts forward to the
    first valid instant; an ambiguous edge (inside the repeated fall-back
    hour) resolves to its first occurrence.
    """
    frames = []
    epoch0 = pd.Timestamp(0, tz="UTC")
    for name, (start, end) in windows.items():
        s_off, e_off = _parse_clock(start), _parse_clock(end)
        if not (pd.Timedelta(0) <= s_off < e_off <= pd.Timedelta(hours=24)):
            raise ValueError(f"window {name!r}: need 00:00 <= start < end <= 24:00, got {start}-{end}")
        edges = []
        for off in (s_off, e_off):
            local = (dates + off).tz_localize(
                tz, ambiguous=np.ones(len(dates), dtype=bool), nonexistent="shift_forward"
            )
            edges.append(((local - epoch0) / pd.Timedelta(seconds=1)).to_numpy(float))
        frames.append(pd.DataFrame({
            "date": [d.date() for d in dates],
            "window": name,
            "bin_start": edges[0],
            "bin_end": edges[1],
        }))
    bins = pd.concat(frames, ignore_index=True)
    bins["bin_s"] = bins["bin_end"] - bins["bin_start"]
    return bins


def _bin_index(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of the disjoint sorted bin ``[starts, ends)`` holding each *x*, else -1."""
    k = np.searchsorted(starts, x, side="right") - 1
    ok = (k >= 0) & (x < ends[np.maximum(k, 0)])
    return np.where(ok, k, -1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detector_activity_profile(
    events_df: pd.DataFrame,
    tz: Optional[str],
    roles: Optional[pd.DataFrame] = None,
    windows: Optional[Mapping[str, Tuple[str, str]]] = None,
    start_date: Optional[_dt.date] = None,
    end_date: Optional[_dt.date] = None,
    short_pulse_s: float = 0.1,
) -> pd.DataFrame:
    """Per-detector activity statistics per local day and sub-day window.

    Args:
        events_df: Events with ``timestamp`` (UTC epoch seconds, or
            tz-aware/naive-UTC datetimes), ``event_code`` and ``parameter``.
            Detector channels come from codes 81/82.  Every non-gap event
            counts as logged time when building ``observed_s``, so pass the
            intersection's full event stream for the period, not a
            detector-only subset.
        tz: IANA zone of the intersection (``metadata.timezone``).
        roles: Optional role table from
            :func:`~atspm.analysis.detector_roles.parse_detector_roles`.
            Its detectors are flagged ``configured`` and get rows even when
            silent.  The caller chooses the config row that's in effect.
        windows: ``{name: ("HH:MM", "HH:MM")}`` local windows, start
            inclusive, end exclusive, with ``"24:00"`` for midnight at the
            end of the day.  Windows don't wrap past midnight.  Defaults to
            :data:`DEFAULT_WINDOWS` (``"day"`` only).
        start_date, end_date: Local dates, inclusive.  Default to the local
            dates of the first and last logged event.
        short_pulse_s: Measured on-durations at or below this count in
            ``n_short`` (chatter evidence).

    Returns:
        DataFrame with :data:`ACTIVITY_SCHEMA` columns, one row per
        ``(date, window, detector)``, sorted by window, date and detector.
        Detectors are every channel with 81/82 events plus every
        configured one.

            date          object   – local calendar date (datetime.date)
            window        str      – window name
            bin_start     float64  – UTC epoch seconds
            bin_end       float64  – UTC epoch seconds
            bin_s         float64  – wall length (23 h / 25 h on DST days)
            observed_s    float64  – logged seconds in the bin
            detector      int64
            configured    bool     – in ``roles``
            n_act         int64    – actuations (on-onsets) in the bin
            n_censored    int64    – of which never logged an off
            on_time_s     float64  – observed on-time clipped to the bin
            occupancy     float64  – on_time_s / observed_s (NaN if 0)
            max_on_s      float64  – longest measured on-duration
            p95_on_s      float64  – 95th percentile (linear) of measured
            n_short       int64    – measured on-durations ≤ short_pulse_s
            min_off_gap_s float64  – shortest measured off → next on
            open_on_s     float64  – longest observed span of a censored one
            on_at_start   boolean  – on at instant bin_start (NA unlogged)
            on_at_end     boolean  – on at instant bin_end (NA unlogged)

    Raises:
        ValueError: If a window's start isn't before its end, or either lies
            outside 00:00–24:00.
    """
    windows = dict(windows or DEFAULT_WINDOWS)
    zone = resolve_pytz(tz)

    if events_df is None or events_df.empty:
        ev = pd.DataFrame({"timestamp": np.empty(0), "event_code": np.empty(0, int),
                           "parameter": np.empty(0, int)})
    else:
        ev = pd.DataFrame({
            "timestamp": _to_epoch(events_df["timestamp"]),
            "event_code": events_df["event_code"].to_numpy(),
            "parameter": events_df["parameter"].to_numpy(),
        }).sort_values("timestamp", kind="stable").reset_index(drop=True)

    t = ev["timestamp"].to_numpy(float)
    code = ev["event_code"].to_numpy()
    par = ev["parameter"].to_numpy()
    gaps = t[code == _GAP_CODE]
    t_ev = t[code != _GAP_CODE]
    obs_s, obs_e = _observed_spans(t_ev, gaps)

    configured = (
        set(int(d) for d in roles["detector"].dropna().unique())
        if roles is not None and not roles.empty else set()
    )
    det_mask = np.isin(code, (_CODE_DET_ON, _CODE_DET_OFF))
    detectors = sorted(configured | set(int(d) for d in np.unique(par[det_mask])))

    if start_date is None or end_date is None:
        if not len(t_ev):
            return _empty()
        local = pd.to_datetime(t_ev[[0, -1]], unit="s", utc=True).tz_convert(zone)
        start_date = start_date or local[0].date()
        end_date = end_date or local[1].date()
    if not detectors or end_date < start_date:
        return _empty()

    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    bins = _build_bins(dates, windows, zone)
    bins["observed_s"] = _covered(obs_s, obs_e, bins["bin_end"].to_numpy()) - _covered(
        obs_s, obs_e, bins["bin_start"].to_numpy())
    b_start = bins["bin_start"].to_numpy()
    b_end = bins["bin_end"].to_numpy()
    start_logged = _inside(obs_s, obs_e, b_start)
    end_logged = _inside(obs_s, obs_e, b_end)

    # Per detector: intervals (measured + censored), and per-bin state.
    iv_parts, state_parts = [], []
    last_ts = t_ev[-1] if len(t_ev) else np.nan
    for det in detectors:
        iv = _reconstruct_intervals(ev, det)
        on = iv["on_ts"].to_numpy(float)
        off = iv["off_ts"].to_numpy(float)
        censored = np.isin(off, gaps)
        # _reconstruct_intervals drops an interval still open at the data end.
        d_codes = code[(det_mask & (par == det)) | (code == _GAP_CODE)]
        d_ts = t[(det_mask & (par == det)) | (code == _GAP_CODE)]
        if len(d_codes) and d_codes[-1] == _CODE_DET_ON:
            not_on = np.flatnonzero(d_codes != _CODE_DET_ON)
            open_on = d_ts[not_on[-1] + 1] if len(not_on) else d_ts[0]
            if open_on < last_ts:
                on = np.append(on, open_on)
                off = np.append(off, last_ts)
                censored = np.append(censored, True)

        dur = off - on
        prev_ok = np.concatenate(([False], ~censored[:-1])) if len(on) else np.empty(0, bool)
        prev_off = np.concatenate(([np.nan], off[:-1])) if len(on) else np.empty(0)
        gap_ok = prev_ok.copy()
        if prev_ok.any():
            gap_ok[prev_ok] = _windows_clear_of_gaps(prev_off[prev_ok], on[prev_ok], gaps)
        iv_parts.append(pd.DataFrame({
            "detector": det,
            "on": on,
            "censored": censored,
            "dur": np.where(censored, np.nan, dur),
            "open_s": np.where(censored, dur, np.nan),
            "off_gap": np.where(gap_ok, on - prev_off, np.nan),
        }))

        on_time = _covered(on, off, b_end) - _covered(on, off, b_start)
        state_parts.append(pd.DataFrame({
            "bin": np.arange(len(bins)),
            "detector": det,
            "on_time_s": on_time,
            "on_at_start": pd.array(np.where(start_logged, _inside(on, off, b_start), False),
                                    dtype="boolean"),
            "on_at_end": pd.array(np.where(end_logged, _inside(on, off, b_end), False),
                                  dtype="boolean"),
        }))
        state_parts[-1].loc[~start_logged, "on_at_start"] = pd.NA
        state_parts[-1].loc[~end_logged, "on_at_end"] = pd.NA

    ivs = pd.concat(iv_parts, ignore_index=True)
    state = pd.concat(state_parts, ignore_index=True)

    # Interval statistics into the bin holding each on, one window at a time.
    stat_parts = []
    on_all = ivs["on"].to_numpy(float)
    for name in windows:
        idx = np.flatnonzero(bins["window"].to_numpy() == name)
        k = _bin_index(on_all, b_start[idx], b_end[idx])
        sel = ivs.loc[k >= 0].assign(bin=idx[k[k >= 0]])
        if sel.empty:
            continue
        sel["short"] = sel["dur"] <= short_pulse_s + _SHORT_EPS
        stat_parts.append(sel.groupby(["detector", "bin"]).agg(
            n_act=("on", "size"),
            n_censored=("censored", "sum"),
            max_on_s=("dur", "max"),
            p95_on_s=("dur", lambda s: s.quantile(0.95)),
            n_short=("short", "sum"),
            min_off_gap_s=("off_gap", "min"),
            open_on_s=("open_s", "max"),
        ).reset_index())

    out = state.merge(bins.reset_index(names="bin"), on="bin")
    if stat_parts:
        out = out.merge(pd.concat(stat_parts, ignore_index=True), on=["detector", "bin"], how="left")
    else:
        for col in ("n_act", "n_censored", "max_on_s", "p95_on_s", "n_short",
                    "min_off_gap_s", "open_on_s"):
            out[col] = np.nan
    for col in ("n_act", "n_censored", "n_short"):
        out[col] = out[col].fillna(0)
    out["configured"] = out["detector"].isin(configured)
    obs = out["observed_s"].to_numpy(float)
    out["occupancy"] = np.where(obs > 0, out["on_time_s"] / np.where(obs > 0, obs, 1.0), np.nan)

    win_order = {name: i for i, name in enumerate(windows)}
    out = out.assign(_w=out["window"].map(win_order)).sort_values(
        ["_w", "bin_start", "detector"], kind="stable")
    return out[ACTIVITY_SCHEMA].astype(_DTYPES).reset_index(drop=True)
