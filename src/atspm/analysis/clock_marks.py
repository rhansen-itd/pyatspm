"""
Clock-Mark Decoder (Functional Core)

Pure functions only. No I/O, no SQL, no side effects.

Decodes the pedestrian-call pulses that ``eos_set_time`` (econ_itd_tools)
writes into the controller's own log on three ped phases the site does not
use.  Each pulse is a PedDetector On (code 90) / Off (code 89) pair on one
marker ped:

    behind  drift pulse, controller slow; held for |drift|
    ahead   drift pulse, controller fast; held for |drift|
    set     bracket around a front-panel clock correction; held for L host
            seconds (a whole multiple of 10) and so logged as L + shift

``drift`` is controller minus true (host) time, so true = label - drift.
The measured record behind every constant here (13 bench pulses, firmware
03.02.60) is in ``docs/ROADMAP.md``, item S1.

Gap Marker Rule:
    Pairing stops at comms-gap markers (``event_code = -1``,
    ``parameter = COMMS_GAP_PARAM``): data was lost there, so an ON before one
    is never paired with an OFF after it.  Backward-clock-step fences
    (``parameter = CLOCK_STEP_FENCE_PARAM``) are crossed, on the marker peds
    only.  A set bracket exists to span exactly such a step, and upstream
    sizes it (logged width >= 5 s) so every marker event keeps its order in
    label time across its own step; the marker peds carry nothing else.  An
    unmarked step (keypad set, power event) inside a pulse is not detectable
    from the log and can corrupt that one pulse's width.

Output frames hold UTC epoch floats in controller label time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .decoders import CLOCK_STEP_FENCE_PARAM


# ---------------------------------------------------------------------------
# Constants (bench-measured; see docs/ROADMAP.md, S1)
# ---------------------------------------------------------------------------

_GAP_CODE: int = -1
_PED_ON: int = 90
_PED_OFF: int = 89

# Marker peds must be distinct and in 9-16 (upstream refuses anything else).
_MARKER_PED_RANGE: Tuple[int, int] = (9, 16)

# Logged widths ran 0.0-0.3 s short of what was sent (mean 0.15), with
# +/-0.05 s precision on top.
_WIDTH_BIAS: float = 0.15
_BIAS_HALF_SPREAD: float = 0.15
_PRECISION: float = 0.05

# Drift pulses are capped at 30.0 s sent; a capped pulse logs 29.7-30.0 s.
_SATURATION_CAP: float = 30.0
_SATURATED_LOGGED: float = (
    _SATURATION_CAP - 2 * _BIAS_HALF_SPREAD - _PRECISION
)

# Set bracket: L is a whole multiple of 10 s, chosen so the logged width is
# >= 5 s; the first edit lands >= 1.4 s after ON.
_BRACKET_PERIOD: float = 10.0
_MIN_BRACKET_WIDTH: float = 5.0 - _WIDTH_BIAS - _BIAS_HALF_SPREAD - _PRECISION
_FIRST_EDIT_LEAD: float = 1.4

# Pulses of one run follow each other back to back (0.0-0.1 s apart on the
# bench); anything further apart is not the same run.
_RUN_CHAIN_GAP: float = 5.0

# The shift is the integer nearest -drift_pre; further than this means the
# bracket and its pre-set pulse disagree.
_SHIFT_TOLERANCE: float = 1.0

# Send-log pulses match logged ONs within this many seconds of label time.
_SEND_LOG_MATCH_TOL: float = 1.0

_DRIFT_COLUMNS = [
    "ts", "off", "ped", "width", "drift", "drift_lo", "drift_hi",
    "saturated", "role", "drift_host", "status",
]
_SET_COLUMNS = [
    "bracket_on", "bracket_off", "width", "drift_pre", "shift_est",
    "period", "shift", "step_lo", "step_hi", "shift_host", "status",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkerPeds:
    """The three ped phases one intersection's clock marks are written on."""

    behind: int
    ahead: int
    set: int

    @property
    def all(self) -> Tuple[int, int, int]:
        return (self.behind, self.ahead, self.set)


def marker_peds_from_config(config: Dict[str, Any]) -> Optional[MarkerPeds]:
    """Read the marker peds from the ``Clk_*`` config columns.

    There is no default, mirroring upstream: a site with no ``Clk:`` rows in
    ``int_cfg.csv`` has no clock marks to decode.

    Args:
        config: Active config dict (``Clk_Behind``, ``Clk_Ahead``,
            ``Clk_Set``).

    Returns:
        ``MarkerPeds``, or ``None`` when none of the three keys is set.

    Raises:
        ValueError: When only some keys are set, a value is not an integer,
            a ped is outside 9-16, or two roles share a ped.
    """
    keys = ("Clk_Behind", "Clk_Ahead", "Clk_Set")
    raw = [config.get(k) for k in keys]
    present = [v is not None and str(v).strip() not in ("", "nan") for v in raw]
    if not any(present):
        return None
    if not all(present):
        missing = [k for k, p in zip(keys, present) if not p]
        raise ValueError(f"Clock-mark config incomplete: missing {missing}")

    try:
        peds = [int(float(str(v).strip())) for v in raw]
    except ValueError as exc:
        raise ValueError(f"Clock-mark peds must be integers: {raw}") from exc

    lo, hi = _MARKER_PED_RANGE
    if any(not lo <= p <= hi for p in peds):
        raise ValueError(f"Clock-mark peds must be in {lo}-{hi}: {peds}")
    if len(set(peds)) != 3:
        raise ValueError(f"Clock-mark peds must be distinct: {peds}")
    return MarkerPeds(*peds)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def drop_marker_events(events_df: pd.DataFrame, peds: MarkerPeds) -> pd.DataFrame:
    """Remove every ped-phase event on the marker peds.

    The marks are ped calls (codes 89/90 plus a code 45 at each ON), so any
    pedestrian measure would read them as real calls.  Codes 21-23 and 45 on
    those peds are dropped too, so nothing depends on a walk never following.

    Args:
        events_df: Events with ``event_code`` and ``parameter`` columns.
        peds: The intersection's marker peds.

    Returns:
        ``events_df`` without those rows (index preserved).
    """
    ped_codes = (21, 22, 23, 45, _PED_OFF, _PED_ON)
    mask = (
        events_df["event_code"].isin(ped_codes)
        & events_df["parameter"].isin(peds.all)
    )
    return events_df.loc[~mask]


def send_log_pulses(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten ``eos-time.jsonl`` run records into one row per sent pulse.

    The head unit's send log times every pulse on the host clock.  A pulse
    logs at ``host + drift`` in controller label time, where ``drift`` is
    the pulse's own measurement for a drift pulse and the run's pre-set
    drift for the set bracket (it fires before any edit).

    Args:
        records: Parsed JSONL objects, one per run.

    Returns:
        DataFrame ``[ped, role, on_label, drift_host, shift_host]``; empty
        when no record carries pulses.
    """
    rows = []
    for rec in records:
        before = (rec.get("before") or {}).get("drift")
        for p in rec.get("pulses") or []:
            if p.get("role") == "set":
                drift, shift = before, p.get("shift_s")
            else:
                drift, shift = p.get("drift_s"), None
            if drift is None or p.get("on_epoch") is None:
                continue
            rows.append({
                "ped": int(p["ped"]),
                "role": p["role"],
                "on_label": float(p["on_epoch"]) + float(drift),
                "drift_host": float(p["drift_s"]) if p.get("drift_s") is not None else np.nan,
                "shift_host": float(shift) if shift is not None else np.nan,
            })
    return pd.DataFrame(
        rows, columns=["ped", "role", "on_label", "drift_host", "shift_host"]
    )


def pair_marker_pulses(events_df: pd.DataFrame, peds: MarkerPeds) -> pd.DataFrame:
    """Pair code 90 (ON) with code 89 (OFF) on the marker peds.

    Pairing is per ped and stops at comms-gap markers but crosses
    backward-clock-step fences (see the module docstring).  Each ON pairs
    with the next event on its ped only if that event is an OFF.

    Args:
        events_df: Events ``[timestamp, event_code, parameter]`` (UTC epoch
            floats), gap markers included.
        peds: The intersection's marker peds.

    Returns:
        DataFrame ``[ped, on, off, width, segment, status]``, one row per ON
        and one per OFF left without an ON, sorted by time.  ``status`` is
        ``'ok'``, ``'unpaired_on'`` or ``'unpaired_off'``; unpaired rows
        carry NaN for the missing side and the width.
    """
    code = events_df["event_code"]
    is_mark = code.isin((_PED_ON, _PED_OFF)) & events_df["parameter"].isin(peds.all)
    is_break = (code == _GAP_CODE) & (events_df["parameter"] != CLOCK_STEP_FENCE_PARAM)
    df = events_df.loc[is_mark | is_break, ["timestamp", "event_code", "parameter"]]
    if not is_mark.any():
        return pd.DataFrame(columns=["ped", "on", "off", "width", "segment", "status"])

    # Ties: an ON and its OFF can share a decisecond (logged width 0.0), so
    # ON sorts first; a break sorts after both.
    order = np.lexsort((-df["event_code"].to_numpy(), df["timestamp"].to_numpy()))
    df = df.iloc[order]
    df = df.assign(segment=(df["event_code"] == _GAP_CODE).cumsum().astype(np.int64))
    df = df.loc[df["event_code"] != _GAP_CODE]

    grp = df.groupby(["segment", "parameter"], sort=False)
    next_code = grp["event_code"].shift(-1)
    next_ts = grp["timestamp"].shift(-1)
    prev_code = grp["event_code"].shift(1)

    is_on = df["event_code"] == _PED_ON
    paired = is_on & (next_code == _PED_OFF)
    lone_off = (df["event_code"] == _PED_OFF) & (prev_code != _PED_ON)

    ons = df.loc[is_on]
    out_on = pd.DataFrame({
        "ped": ons["parameter"].to_numpy(),
        "on": ons["timestamp"].to_numpy(),
        "off": np.where(paired[is_on], next_ts[is_on], np.nan),
        "segment": ons["segment"].to_numpy(),
        "status": np.where(paired[is_on], "ok", "unpaired_on"),
    })
    offs = df.loc[lone_off]
    out_off = pd.DataFrame({
        "ped": offs["parameter"].to_numpy(),
        "on": np.nan,
        "off": offs["timestamp"].to_numpy(),
        "segment": offs["segment"].to_numpy(),
        "status": "unpaired_off",
    })
    out = pd.concat([out_on, out_off], ignore_index=True)
    out["ped"] = out["ped"].astype(np.int64)
    out["width"] = out["off"] - out["on"]
    sort_ts = out["on"].fillna(out["off"])
    out = out.iloc[np.argsort(sort_ts.to_numpy(), kind="stable")].reset_index(drop=True)
    return out[["ped", "on", "off", "width", "segment", "status"]]


def decode_clock_marks(
    events_df: pd.DataFrame,
    peds: MarkerPeds,
    send_log: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Decode drift samples and clock sets from the marker pulses.

    Drift pulses: ``|drift| = logged width + 0.15 s``, signed by ped
    (behind negative, ahead positive), with ``[drift_lo, drift_hi]`` spanning
    the 0.0-0.3 s bias spread and +/-0.05 s precision.  A pulse logged at
    29.65 s or more is saturated: only ``|drift| >= drift_lo`` is known.

    Set brackets: with ``shift_est = -drift_pre`` from the drift pulse just
    before, ``L = 10 * round((w - shift_est) / 10)`` and the exact whole-second
    ``shift = round(w - L)`` (positive = set forward).  The step itself lies
    in ``[step_lo, step_hi]``, in pre-step label time.

    The send log, when given, is matched by ped and label time.  It fills
    only what the pulses cannot decide and is otherwise kept for comparison
    in ``drift_host`` / ``shift_host``: a saturated drift pulse takes the
    logged drift, which then serves as ``drift_pre`` so the bracket still
    yields the shift; a set with no usable pre-set pulse takes the logged
    shift.  A set the bracket decodes but the
    send log contradicts is flagged rather than resolved either way.

    Args:
        events_df: Events ``[timestamp, event_code, parameter]`` (UTC epoch
            floats), gap markers included.
        peds: The intersection's marker peds.
        send_log: Optional output of ``send_log_pulses``.

    Returns:
        ``(drift_df, sets_df)``.

        **drift_df** - one row per drift pulse::

            ts, off      : ON / OFF label time
            ped, width   : marker ped, logged width (s)
            drift        : controller - true (s); NaN unless decodable
            drift_lo/hi  : bounds on drift (one side +/-inf when saturated)
            saturated    : bool
            role         : 'pre_set', 'residual' or 'check'
            drift_host   : the send log's measurement, if matched
            status       : 'ok', 'saturated', 'send_log', 'unpaired_on',
                           'unpaired_off'

        **sets_df** - one row per set bracket::

            bracket_on/off : label times
            width          : logged width w (s)
            drift_pre      : the pre-set pulse's drift; NaN if none decodable
            shift_est      : -drift_pre
            period         : L (s)
            shift          : whole seconds the clock moved; NaN unless decoded
            step_lo/hi     : label-time window holding the step(s)
            shift_host     : the send log's shift, if matched
            status         : 'ok', 'send_log', 'no_pre_pulse',
                             'pre_saturated', 'inconsistent',
                             'send_log_conflict', 'unpaired_on',
                             'unpaired_off'
    """
    pulses = pair_marker_pulses(events_df, peds)
    if send_log is not None and not send_log.empty and not pulses.empty:
        pulses = _match_send_log(pulses, send_log)
    else:
        pulses = pulses.assign(drift_host=np.nan, shift_host=np.nan)

    is_set = pulses["ped"] == peds.set
    drift_df = _decode_drift(pulses.loc[~is_set], peds)
    sets_df = _decode_sets(pulses.loc[is_set], drift_df)
    drift_df = _assign_roles(drift_df, sets_df)
    return drift_df[_DRIFT_COLUMNS], sets_df[_SET_COLUMNS]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _match_send_log(pulses: pd.DataFrame, send_log: pd.DataFrame) -> pd.DataFrame:
    """Attach ``drift_host`` / ``shift_host`` from the nearest sent pulse."""
    keyed = pulses.assign(_key=pulses["on"].fillna(pulses["off"]), _row=np.arange(len(pulses)))
    has_on = keyed["on"].notna()
    left = keyed.loc[has_on].sort_values("_key")
    right = send_log.sort_values("on_label")[["ped", "on_label", "drift_host", "shift_host"]]
    matched = pd.merge_asof(
        left, right, left_on="_key", right_on="on_label", by="ped",
        direction="nearest", tolerance=_SEND_LOG_MATCH_TOL,
    )
    out = keyed.merge(
        matched[["_row", "drift_host", "shift_host"]], on="_row", how="left"
    )
    return out.drop(columns=["_key", "_row"])


def _decode_drift(pulses: pd.DataFrame, peds: MarkerPeds) -> pd.DataFrame:
    """Turn behind/ahead pulses into signed drift samples."""
    w = pulses["width"].to_numpy(dtype=float)
    sign = np.where(pulses["ped"].to_numpy() == peds.ahead, 1.0, -1.0)
    ok = (pulses["status"] == "ok").to_numpy()
    saturated = ok & (w >= _SATURATED_LOGGED)
    decodable = ok & ~saturated

    mag_lo = np.maximum(w - _PRECISION, 0.0)
    mag_hi = np.where(saturated, np.inf, w + 2 * _BIAS_HALF_SPREAD + _PRECISION)
    bound_a, bound_b = sign * mag_lo, sign * mag_hi
    drift_lo = np.where(ok, np.minimum(bound_a, bound_b), np.nan)
    drift_hi = np.where(ok, np.maximum(bound_a, bound_b), np.nan)
    drift = np.where(decodable, sign * (w + _WIDTH_BIAS), np.nan)

    host = pulses["drift_host"].to_numpy(dtype=float)
    from_log = saturated & ~np.isnan(host)
    drift = np.where(from_log, host, drift)
    status = np.where(
        from_log, "send_log",
        np.where(saturated, "saturated", pulses["status"].to_numpy()),
    )
    return pd.DataFrame({
        "ts": pulses["on"].to_numpy(),
        "off": pulses["off"].to_numpy(),
        "ped": pulses["ped"].to_numpy(),
        "width": w,
        "drift": drift,
        "drift_lo": drift_lo,
        "drift_hi": drift_hi,
        "saturated": saturated,
        "segment": pulses["segment"].to_numpy(),
        "drift_host": host,
        "status": status,
    })


def _decode_sets(brackets: pd.DataFrame, drift_df: pd.DataFrame) -> pd.DataFrame:
    """Recover each bracket's whole-second shift from its pre-set pulse."""
    sets = pd.DataFrame({
        "bracket_on": brackets["on"].to_numpy(),
        "bracket_off": brackets["off"].to_numpy(),
        "width": brackets["width"].to_numpy(dtype=float),
        "segment": brackets["segment"].to_numpy(),
        "shift_host": brackets["shift_host"].to_numpy(dtype=float),
        "_pulse_status": brackets["status"].to_numpy(),
    })
    if sets.empty:
        return pd.DataFrame(columns=_SET_COLUMNS)

    # The pre-set pulse ends right before the bracket opens, same segment.
    # A saturated pulse still anchors the run (its drift is NaN unless the
    # send log filled it); an unpaired one does not.
    usable = drift_df["status"].isin(("ok", "saturated", "send_log"))
    pre = drift_df.loc[usable, ["off", "segment", "drift", "status"]]
    pre = pre.rename(columns={"off": "_pre_off", "drift": "drift_pre", "status": "_pre_status"})
    sets["_row"] = np.arange(len(sets))
    keyed = sets.loc[sets["bracket_on"].notna()].sort_values("bracket_on")
    matched = pd.merge_asof(
        keyed, pre.sort_values("_pre_off"),
        left_on="bracket_on", right_on="_pre_off", by="segment",
        direction="backward", tolerance=_RUN_CHAIN_GAP,
    )
    sets = sets.merge(
        matched[["_row", "drift_pre", "_pre_status"]], on="_row", how="left"
    )

    w = sets["width"].to_numpy()
    shift_est = -sets["drift_pre"].to_numpy(dtype=float)
    period = _BRACKET_PERIOD * np.round((w - shift_est) / _BRACKET_PERIOD)
    shift = np.round(w - period)

    pulse_ok = sets["_pulse_status"].to_numpy() == "ok"
    pre_status = sets["_pre_status"].to_numpy()
    has_pre = pd.notna(pre_status)
    pre_saturated = pre_status == "saturated"
    consistent = (
        (np.abs(shift - shift_est) <= _SHIFT_TOLERANCE)
        & (w >= _MIN_BRACKET_WIDTH)
        & (period >= _BRACKET_PERIOD)
    )
    decoded = pulse_ok & has_pre & ~pre_saturated & consistent

    host = sets["shift_host"].to_numpy()
    has_host = ~np.isnan(host)
    conflict = decoded & has_host & (shift != host)
    fill = pulse_ok & ~decoded & has_host & (~has_pre | pre_saturated)

    status = np.select(
        [~pulse_ok, conflict, decoded, fill, ~has_pre, pre_saturated],
        [sets["_pulse_status"].to_numpy(), "send_log_conflict", "ok",
         "send_log", "no_pre_pulse", "pre_saturated"],
        default="inconsistent",
    )
    final_shift = np.where(decoded & ~conflict, shift, np.where(fill, host, np.nan))
    # L follows from the shift once it is known: w = L + shift.
    final_period = np.where(
        np.isnan(final_shift), np.nan,
        _BRACKET_PERIOD * np.round((w - final_shift) / _BRACKET_PERIOD),
    )

    on = sets["bracket_on"].to_numpy()
    sets = sets.assign(
        shift_est=shift_est,
        period=final_period,
        shift=final_shift,
        step_lo=on + _FIRST_EDIT_LEAD,
        step_hi=on + final_period,
        status=status,
    )
    return sets


def _assign_roles(drift_df: pd.DataFrame, sets_df: pd.DataFrame) -> pd.DataFrame:
    """Label each drift pulse as pre-set, residual (post-set) or hourly check."""
    role = np.full(len(drift_df), "check", dtype=object)
    if not sets_df.empty and not drift_df.empty:
        on = drift_df["ts"].to_numpy(dtype=float)
        off = drift_df["off"].to_numpy(dtype=float)
        b_on = sets_df["bracket_on"].dropna().to_numpy(dtype=float)
        b_off = sets_df["bracket_off"].dropna().to_numpy(dtype=float)
        role[_within_before(off, b_on)] = "pre_set"
        role[_within_after(on, b_off)] = "residual"
    return drift_df.assign(role=role)


def _within_before(t: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """True where some anchor lies in ``[t, t + _RUN_CHAIN_GAP]``."""
    if anchors.size == 0:
        return np.zeros(t.shape, dtype=bool)
    anchors = np.sort(anchors)
    idx = np.searchsorted(anchors, t, side="left")
    nxt = anchors[np.minimum(idx, anchors.size - 1)]
    return (idx < anchors.size) & (nxt - t <= _RUN_CHAIN_GAP)


def _within_after(t: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """True where some anchor lies in ``[t - _RUN_CHAIN_GAP, t]``."""
    if anchors.size == 0:
        return np.zeros(t.shape, dtype=bool)
    anchors = np.sort(anchors)
    idx = np.searchsorted(anchors, t, side="right") - 1
    prv = anchors[np.maximum(idx, 0)]
    return (idx >= 0) & (t - prv <= _RUN_CHAIN_GAP)
