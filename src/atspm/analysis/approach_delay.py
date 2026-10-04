"""
Arrivals on Red and Approach Delay (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

Cycle definition (UDOT PCD / approach delay)
--------------------------------------------
Each row is one *red-to-red* cycle of a phase, named by the green that
serves it::

    red_start ─── red ───▶ green_ts ── green ──▶ yellow_ts ─ yellow ─▶ next_red_start
    (previous green's                                                 (this green's
     end of yellow)                                                    end of yellow)

``red_start`` / ``next_red_start`` are ``yellow_end_ts`` from
``_build_phase_intervals``: Code 9, else Code 10.  Red clearance is part of
red, as the driver sees it.  Consecutive cycles tile time, so every arrival
falls in exactly one cycle.

An arrival is classed by the state it reaches the stop line in (detector
time + travel time):

* red    — ``[red_start, green_ts)``; delay = ``green_ts − t``
* green  — ``[green_ts, yellow_ts)``; delay 0
* yellow — ``[yellow_ts, next_red_start)``; delay 0

Delay per vehicle is total delay over *all* arrivals (UDOT), so it reads as
the mean stopped delay of the approach, not of red arrivals alone.

Gap Marker Rule (censoring)
---------------------------
A cycle is *censored* — reported, flagged, never counted — when:

* it is the first green of its gap-free segment (no preceding red start,
  or a gap marker lies between the previous green and this one);
* any Code 1 of the phase falls in ``(red_start, green_ts)``: a green whose
  end was lost (``_build_phase_intervals`` leaves it blank) was really
  served there, so a delay measured across it would be wrong.

Arrivals whose travel-time shift carries them across a gap marker are
dropped: their detector time and stop-line time sit in different segments.

Package Location: src/atspm/analysis/approach_delay.py
"""

from __future__ import annotations

from typing import List, Mapping, Union

import numpy as np
import pandas as pd

from .aog import _build_green_windows, _shift_detector_timestamps
from .detector_inference import _to_epoch

_GAP_CODE: int = -1
_CODE_GREEN: int = 1

CYCLE_SCHEMA = [
    "phase",
    "coord_plan",
    "cycle_start",
    "red_start",
    "green_ts",
    "yellow_ts",
    "next_red_start",
    "cycle_len",
    "red_dur",
    "green_dur",
    "yellow_dur",
    "censored",
    "arrivals",
    "arrivals_green",
    "arrivals_yellow",
    "arrivals_red",
    "aog_pct",
    "aoy_pct",
    "aor_pct",
    "total_delay_s",
    "delay_per_veh",
]

BIN_SCHEMA = [
    "time",
    "phase",
    "coord_plan",
    "n_cycles",
    "n_censored",
    "arrivals",
    "arrivals_green",
    "arrivals_yellow",
    "arrivals_red",
    "aog_pct",
    "aoy_pct",
    "aor_pct",
    "total_delay_s",
    "delay_per_veh",
    "total_delay_vh",
    "delay_vh_per_hr",
]

_COUNT_COLS = ["arrivals", "arrivals_green", "arrivals_yellow", "arrivals_red"]


def _shares(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three arrival shares and delay per vehicle (NaN when no arrivals)."""
    n = df["arrivals"].astype(float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for col, src in (("aog_pct", "arrivals_green"),
                         ("aoy_pct", "arrivals_yellow"),
                         ("aor_pct", "arrivals_red")):
            df[col] = np.where(n > 0, df[src].astype(float).to_numpy() / n, np.nan)
        df["delay_per_veh"] = np.where(
            n > 0, df["total_delay_s"].astype(float).to_numpy() / n, np.nan
        )
    return df


def approach_delay(
    events_df: pd.DataFrame,
    phase: int,
    detector_ids: List[int],
    travel_time_sec: Union[float, Mapping[int, float]] = 0.0,
) -> pd.DataFrame:
    """Per-cycle arrival shares (green/yellow/red) and approach delay for one phase.

    Args:
        events_df: Flat events DataFrame with columns
            ``[timestamp, event_code, parameter, cycle_start, coord_plan]``.
            Timestamps may be UTC epoch floats or tz-aware Timestamps (any
            resolution).  Gap markers (``event_code == -1``) must be present.
        phase: Signal phase number.
        detector_ids: Advance (``Det_P{phase}_Arrival``) detectors.
        travel_time_sec: Seconds from detector to stop line, added to each
            Code-82 time: one value for all detectors, or a
            ``{detector: seconds}`` mapping covering every detector.

    Returns:
        One row per red-to-red cycle (see module docstring), sorted by
        ``green_ts``, columns :data:`CYCLE_SCHEMA`::

            phase            int
            coord_plan       float   – plan at the serving green
            cycle_start      input dtype – detected (barrier) cycle of the green
            red_start        input dtype – previous green's end of yellow (NaT/NaN
                                       when none)
            green_ts, yellow_ts, next_red_start   input dtype
            cycle_len        float s – next_red_start − red_start
            red_dur, green_dur, yellow_dur   float s
            censored         bool
            arrivals, arrivals_green, arrivals_yellow, arrivals_red
                             Int64   – NA when censored
            aog_pct, aoy_pct, aor_pct   float – NaN when censored or no arrivals
            total_delay_s    float s – sum of red-arrival delays; NaN when censored
            delay_per_veh    float s – total_delay_s / arrivals

        Empty frame with this schema when no green windows exist or
        *detector_ids* is empty.

    Raises:
        ValueError: *travel_time_sec* is a mapping that misses a detector.
    """
    empty = pd.DataFrame(columns=CYCLE_SCHEMA)
    if events_df.empty or not detector_ids:
        return empty
    if isinstance(travel_time_sec, Mapping):
        missing = sorted(set(detector_ids) - set(travel_time_sec))
        if missing:
            raise ValueError(f"travel_time_sec has no value for detectors {missing}")

    win = _build_green_windows(events_df, phase)
    if win.empty:
        return empty
    win = win.sort_values("green_ts", kind="stable").reset_index(drop=True)
    n = len(win)

    G = _to_epoch(win["green_ts"])
    Y = _to_epoch(win["yellow_ts"])
    R = _to_epoch(win["yellow_end_ts"])
    seg = win["seg"].to_numpy()

    R_prev = np.full(n, np.nan)
    R_prev[1:] = R[:-1]
    valid = np.zeros(n, dtype=bool)
    valid[1:] = seg[1:] == seg[:-1]
    valid &= np.where(valid, R_prev <= G, False)

    # A Code 1 strictly inside (red_start, green_ts) is a served green whose
    # end was lost: the red did not run unbroken to this green.
    is_green = (events_df["event_code"] == _CODE_GREEN) & (events_df["parameter"] == phase)
    all_greens = np.sort(_to_epoch(events_df.loc[is_green, "timestamp"]))
    rp = np.where(valid, R_prev, G)
    n_between = (np.searchsorted(all_greens, G, side="left")
                 - np.searchsorted(all_greens, rp, side="right"))
    valid &= n_between == 0

    # --- arrivals at the stop line --------------------------------------------
    counts = np.zeros((n, 3), dtype=np.int64)           # green, yellow, red
    delay = np.zeros(n, dtype=np.float64)
    det = _shift_detector_timestamps(events_df, list(detector_ids), 0.0)
    if not det.empty:
        A_raw = _to_epoch(det["timestamp"])
        if isinstance(travel_time_sec, Mapping):
            off = det["parameter"].map(dict(travel_time_sec)).astype(float).to_numpy()
        else:
            off = np.full(len(det), float(travel_time_sec))
        A = A_raw + off
        gaps = np.sort(_to_epoch(events_df.loc[events_df["event_code"] == _GAP_CODE, "timestamp"]))
        same_side = (np.searchsorted(gaps, A_raw, side="right")
                     == np.searchsorted(gaps, A, side="right"))

        k = np.searchsorted(R, A, side="right")          # A in [R[k-1], R[k])
        kk = np.clip(k, 0, n - 1)
        take = same_side & (k >= 1) & (k < n) & valid[kk]
        A, kk = A[take], kk[take]
        red = A < G[kk]
        green = ~red & (A < Y[kk])
        yellow = ~red & ~green
        for j, m in enumerate((green, yellow, red)):
            counts[:, j] = np.bincount(kk[m], minlength=n)
        delay = np.bincount(kk[red], weights=G[kk[red]] - A[red], minlength=n)

    out = pd.DataFrame({
        "phase": int(phase),
        "coord_plan": win["coord_plan"].astype(float).to_numpy(),
        "cycle_start": win["cycle_start"],
        "red_start": win["yellow_end_ts"].shift(1),
        "green_ts": win["green_ts"],
        "yellow_ts": win["yellow_ts"],
        "next_red_start": win["yellow_end_ts"],
        "cycle_len": R - R_prev,
        "red_dur": G - R_prev,
        "green_dur": Y - G,
        "yellow_dur": R - Y,
        "censored": ~valid,
    })
    out["arrivals_green"] = counts[:, 0]
    out["arrivals_yellow"] = counts[:, 1]
    out["arrivals_red"] = counts[:, 2]
    out["arrivals"] = counts.sum(axis=1)
    out["total_delay_s"] = delay
    for col in _COUNT_COLS:
        out[col] = out[col].astype("Int64")
        out.loc[~valid, col] = pd.NA
    out.loc[~valid, "total_delay_s"] = np.nan
    out = _shares(out)

    return out[CYCLE_SCHEMA].round({
        "cycle_len": 2, "red_dur": 2, "green_dur": 2, "yellow_dur": 2,
        "aog_pct": 4, "aoy_pct": 4, "aor_pct": 4,
        "total_delay_s": 2, "delay_per_veh": 2,
    })


def bin_approach_delay(cycle_df: pd.DataFrame, bin_len: int = 15) -> pd.DataFrame:
    """Aggregate per-cycle approach delay into fixed time bins.

    Counts and delay are summed over uncensored cycles before any ratio is
    taken.  A cycle is binned by its serving green (``green_ts``).

    Args:
        cycle_df: Output of :func:`approach_delay`, possibly concatenated over
            phases.
        bin_len: Bin width in minutes.  Default 15.

    Returns:
        Columns :data:`BIN_SCHEMA`, sorted by ``phase, time``::

            time             Timestamp – bin start (UTC when input was epoch)
            phase            int
            coord_plan       float
            n_cycles         int   – uncensored cycles
            n_censored       int
            arrivals, arrivals_green, arrivals_yellow, arrivals_red   int
            aog_pct, aoy_pct, aor_pct   float – NaN when no arrivals
            total_delay_s    float – seconds
            delay_per_veh    float – total_delay_s / arrivals (NaN when none)
            total_delay_vh   float – vehicle-hours of delay in the bin
            delay_vh_per_hr  float – total_delay_vh scaled to an hourly rate

        A bin holding only censored cycles is kept, with zero counts and NaN
        rates, so lost coverage stays visible.
    """
    if cycle_df is None or cycle_df.empty:
        return pd.DataFrame(columns=BIN_SCHEMA)

    df = cycle_df.copy()
    g = df["green_ts"]
    if pd.api.types.is_datetime64_any_dtype(g):
        df["time"] = g.dt.floor(f"{bin_len}min")
    else:
        df["time"] = pd.to_datetime(g.astype(float), unit="s", utc=True).dt.floor(f"{bin_len}min")

    ok = ~df["censored"].astype(bool)
    df["_ok"] = ok.astype(int)
    df["_cens"] = (~ok).astype(int)
    for col in _COUNT_COLS:
        df[col] = df[col].fillna(0).astype(np.int64)
    df["total_delay_s"] = df["total_delay_s"].fillna(0.0)

    agg = df.groupby(["time", "phase", "coord_plan"], sort=False).agg(
        n_cycles=("_ok", "sum"),
        n_censored=("_cens", "sum"),
        arrivals=("arrivals", "sum"),
        arrivals_green=("arrivals_green", "sum"),
        arrivals_yellow=("arrivals_yellow", "sum"),
        arrivals_red=("arrivals_red", "sum"),
        total_delay_s=("total_delay_s", "sum"),
    ).reset_index()
    agg = _shares(agg)
    agg["total_delay_vh"] = agg["total_delay_s"] / 3600.0
    agg["delay_vh_per_hr"] = agg["total_delay_vh"] * (60.0 / bin_len)
    for col in ["n_cycles", "n_censored"] + _COUNT_COLS:
        agg[col] = agg[col].astype(int)

    agg = agg.sort_values(["phase", "time"], kind="stable").reset_index(drop=True)
    return agg[BIN_SCHEMA].round({
        "aog_pct": 4, "aoy_pct": 4, "aor_pct": 4, "total_delay_s": 2,
        "delay_per_veh": 2, "total_delay_vh": 4, "delay_vh_per_hr": 4,
    })
