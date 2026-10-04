"""
Detector Role Table (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

One parser for every detector-bearing config key, so the shell engines
stop reimplementing key matching (and drifting on spellings).  The config
import turns ``int_cfg.csv`` rows into config columns::

    Plt: P{N} Arrival     →  Det_P{N}_Arrival     role 'arrival'
    Plt: P{N} Stop Bar    →  Det_P{N}_Stop_Bar    role 'stop_bar'
    Det: P{N} Stopbar     →  Det_P{N}_Stopbar     role 'stop_bar' (same role)
    Plt: P{N} Occupancy   →  Det_P{N}_Occupancy   role 'occupancy'
    Det: P{N} Pairs       →  Det_P{N}_Pairs       role 'pairs' (JSON pairs)
    TM:  {label}          →  TM_{label}           role 'tm'
    WD:  Sensor{K}        →  WD_Sensor{K}         role 'watchdog'

One non-role key rides with the arrival role (read by
:func:`arrival_travel_times`, never turned into role rows)::

    Det: P{N} Arrival Travel  →  Det_P{N}_Arrival_Travel  seconds, advance
                                 detector → stop line ("5.4", or one value
                                 per Det_P{N}_Arrival detector, in order)

and one that says which signal a phase's driver sees (read by
:func:`phase_overlaps`)::

    Det: P{N} Overlap  →  Det_P{N}_Overlap  overlap number ("1") or letter
                          ("A"); e.g. a protected left run as an FYA overlap

and one that says which approach a through phase serves (read by
:func:`phase_directions`)::

    Det: P{N} Direction  →  Det_P{N}_Direction  "NB", "SB", "EB" or "WB"; the
                            direction whose ``TM_{dir}T`` traffic phase N
                            serves (left-turn gap analysis, S-M9)

Role meanings (owner, 2026-10-02):

- ``arrival`` — advance detection, upstream of the stop line (AoG, PCD).
- ``stop_bar`` — short count loops *downstream* of the stop bar: counts and
  discharge flow rate.  Not presence.
- ``occupancy`` — the presence zone at the stop line, one channel per
  lane; split failures read it.
- ``pairs`` — redundancy pairs compared by the detector-comparison
  analysis; each member gets a row with the other as ``partner``.
- ``tm`` — turning-movement count detectors, labelled by movement.
- ``watchdog`` — zones no vehicle can call; a call is failsafe evidence.

Package Location: src/atspm/analysis/detector_roles.py
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Role names, in output sort order.
ROLES = ("arrival", "stop_bar", "occupancy", "pairs", "tm", "watchdog")

# Roles keyed by phase (``phase`` is never NA for these).
PHASE_ROLES = frozenset({"arrival", "stop_bar", "occupancy", "pairs"})

ROLE_SCHEMA = ["detector", "phase", "role", "movement", "partner", "key"]

_PHASE_KEY_RE = re.compile(
    r"^Det_P(\d+)_(Arrival|Stopbar|Stop_Bar|Occupancy|Pairs)$"
)
_SUFFIX_TO_ROLE = {
    "Arrival": "arrival",
    "Stopbar": "stop_bar",
    "Stop_Bar": "stop_bar",
    "Occupancy": "occupancy",
    "Pairs": "pairs",
}
_WATCHDOG_KEY_RE = re.compile(r"^WD_Sensor\d+$")
_TRAVEL_KEY_RE = re.compile(r"^Det_P(\d+)_Arrival_Travel$")
_OVERLAP_KEY_RE = re.compile(r"^Det_P(\d+)_Overlap$")
_DIRECTION_KEY_RE = re.compile(r"^Det_P(\d+)_Direction$")
DIRECTIONS = ("NB", "SB", "EB", "WB")
_TM_EXCLUDED_KEYS = frozenset({"TM_Exclusions"})

_DTYPES = {
    "detector": "int64",
    "phase": "Int64",
    "role": "str",
    "movement": "str",
    "partner": "Int64",
    "key": "str",
}


def _is_blank(raw: Any) -> bool:
    return raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip()


def _parse_id_list(raw: Any) -> List[int]:
    """Comma-separated detector IDs; non-integer tokens are dropped."""
    if _is_blank(raw):
        return []
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip().isdigit()]


def _parse_pairs(raw: Any) -> List[List[int]]:
    """``"[[a,b],[c,d]]"`` or a flat ``"[a,b]"``; malformed input gives ``[]``."""
    if _is_blank(raw):
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    if len(parsed) == 2 and all(isinstance(v, int) for v in parsed):
        parsed = [parsed]
    return [
        [int(e[0]), int(e[1])]
        for e in parsed
        if isinstance(e, list) and len(e) == 2 and all(isinstance(v, int) for v in e)
    ]


def _empty_roles() -> pd.DataFrame:
    return pd.DataFrame(columns=ROLE_SCHEMA).astype(_DTYPES)


def parse_detector_roles(config: Dict[str, Any]) -> pd.DataFrame:
    """Parse every detector-bearing config key into one role table.

    Both stop-bar spellings (``Det_P{N}_Stop_Bar`` and ``Det_P{N}_Stopbar``)
    map to ``'stop_bar'``; a detector listed under both for the same phase
    gives one row.  Blank values, non-integer tokens, malformed ``Pairs``
    JSON, ``TM_Exclusions`` and unrecognised ``Det_*`` / ``WD_*`` keys are
    ignored.  No inference is done: the table says what the config says.

    Args:
        config: Active config dict, e.g. from
            ``DatabaseManager.get_config_at_date``.

    Returns:
        DataFrame, one row per (detector, phase, role, partner, movement)::

            detector  int64
            phase     Int64  – NA for 'tm' and 'watchdog'
            role      str    – one of :data:`ROLES`
            movement  str    – the ``TM_*`` label: the row's own for 'tm';
                                for other roles the label of the one
                                ``TM_*`` key holding the detector, NaN when
                                none or several do
            partner   Int64  – the other detector of a 'pairs' row, else NA
            key       str    – the config key the row came from

        Sorted by role (in :data:`ROLES` order), phase, key, detector,
        partner.  Empty (with this schema) when nothing is configured.
    """
    rows: List[Dict[str, Any]] = []
    for key, raw in config.items():
        key = str(key)
        match = _PHASE_KEY_RE.match(key)
        if match:
            phase = int(match.group(1))
            role = _SUFFIX_TO_ROLE[match.group(2)]
            if role == "pairs":
                for a, b in _parse_pairs(raw):
                    rows.append({"detector": a, "phase": phase, "role": role,
                                 "movement": None, "partner": b, "key": key})
                    rows.append({"detector": b, "phase": phase, "role": role,
                                 "movement": None, "partner": a, "key": key})
            else:
                for det in _parse_id_list(raw):
                    rows.append({"detector": det, "phase": phase, "role": role,
                                 "movement": None, "partner": None, "key": key})
        elif key.startswith("TM_") and key not in _TM_EXCLUDED_KEYS:
            for det in _parse_id_list(raw):
                rows.append({"detector": det, "phase": None, "role": "tm",
                             "movement": key[3:], "partner": None, "key": key})
        elif _WATCHDOG_KEY_RE.match(key):
            for det in _parse_id_list(raw):
                rows.append({"detector": det, "phase": None, "role": "watchdog",
                             "movement": None, "partner": None, "key": key})

    if not rows:
        return _empty_roles()

    df = pd.DataFrame(rows, columns=ROLE_SCHEMA)

    # Movement label for non-tm rows: the single TM_* key holding the detector.
    tm = df.loc[df["role"] == "tm", ["detector", "movement"]].drop_duplicates()
    n_labels = tm.groupby("detector")["movement"].transform("size")
    unique_label = tm[n_labels == 1].set_index("detector")["movement"]
    not_tm = df["role"] != "tm"
    df["movement"] = df["movement"].astype(object)
    df.loc[not_tm, "movement"] = df.loc[not_tm, "detector"].map(unique_label).astype(object)

    df = df.astype(_DTYPES)
    df["_rank"] = df["role"].map({r: i for i, r in enumerate(ROLES)})
    df = (
        df.sort_values(["_rank", "phase", "key", "detector", "partner"],
                       na_position="last", kind="mergesort")
        .drop_duplicates(subset=["role", "phase", "detector", "partner", "movement"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    return df[ROLE_SCHEMA]


def detector_sets(roles: pd.DataFrame, role: str) -> Dict[int, frozenset]:
    """Per-phase detector sets for one phase-keyed role.

    Args:
        roles: Output of :func:`parse_detector_roles`.
        role: One of ``'arrival'``, ``'stop_bar'``, ``'occupancy'``,
            ``'pairs'`` (all detectors of the phase's pairs).

    Returns:
        ``{phase: frozenset(detector_ids)}`` for phases with at least one
        detector in *role*.

    Raises:
        ValueError: *role* is not phase-keyed (``'tm'``, ``'watchdog'``) or
            unknown.
    """
    if role not in PHASE_ROLES:
        raise ValueError(
            f"detector_sets needs a phase-keyed role {sorted(PHASE_ROLES)}, got {role!r}"
        )
    sub = roles.loc[roles["role"] == role, ["phase", "detector"]]
    return {
        int(phase): frozenset(int(d) for d in grp)
        for phase, grp in sub.groupby("phase")["detector"]
    }


def arrival_travel_times(config: Dict[str, Any]) -> Dict[int, Dict[int, float]]:
    """Per-detector travel times from the ``Det_P{N}_Arrival_Travel`` keys.

    The value is seconds from the advance detector to the stop line: one
    number applying to every ``Det_P{N}_Arrival`` detector of the phase, or a
    comma list aligned with the ``Det_P{N}_Arrival`` list in its listed
    order.  Phases without the key are absent (callers fall back to a
    global offset).

    Args:
        config: Active config dict, e.g. from
            ``DatabaseManager.get_config_at_date``.

    Returns:
        ``{phase: {detector: seconds}}``.

    Raises:
        ValueError: A travel value is not a non-negative number, its phase
            has no ``Det_P{N}_Arrival`` detectors, or the list length is
            neither 1 nor the number of arrival detectors.
    """
    out: Dict[int, Dict[int, float]] = {}
    for key, raw in config.items():
        match = _TRAVEL_KEY_RE.match(str(key))
        if not match or _is_blank(raw):
            continue
        phase = int(match.group(1))
        try:
            secs = [float(tok) for tok in str(raw).split(",") if tok.strip()]
        except ValueError:
            raise ValueError(f"{key}: not a number list: {raw!r}") from None
        if not secs or any(not np.isfinite(v) or v < 0 for v in secs):
            raise ValueError(f"{key}: travel times must be non-negative seconds: {raw!r}")
        dets = list(dict.fromkeys(_parse_id_list(config.get(f"Det_P{phase}_Arrival"))))
        if not dets:
            raise ValueError(f"{key} is set but Det_P{phase}_Arrival lists no detectors")
        if len(secs) == 1:
            secs = secs * len(dets)
        elif len(secs) != len(dets):
            raise ValueError(
                f"{key} has {len(secs)} values for {len(dets)} arrival detectors {dets}"
            )
        out[phase] = dict(zip(dets, secs))
    return out


def phase_overlaps(config: Dict[str, Any]) -> Dict[int, int]:
    """Overlap shown to a phase's movement, from the ``Det_P{N}_Overlap`` keys.

    Args:
        config: Active config dict, e.g. from
            ``DatabaseManager.get_config_at_date``.

    Returns:
        ``{phase: overlap_number}`` (A = 1 … P = 16).  Phases without the
        key are absent.

    Raises:
        ValueError: A value is neither an integer 1–16 nor a letter A–P.
    """
    out: Dict[int, int] = {}
    for key, raw in config.items():
        match = _OVERLAP_KEY_RE.match(str(key))
        if not match or _is_blank(raw):
            continue
        tok = str(raw).strip().upper()
        if tok.isdigit():
            num = int(tok)
        elif len(tok) == 1 and "A" <= tok <= "P":
            num = ord(tok) - ord("A") + 1
        else:
            num = 0
        if not 1 <= num <= 16:
            raise ValueError(f"{key}: overlap must be 1-16 or A-P, got {raw!r}")
        out[int(match.group(1))] = num
    return out


def phase_directions(config: Dict[str, Any]) -> Dict[int, str]:
    """Approach direction a through phase serves, from ``Det_P{N}_Direction``.

    Args:
        config: Active config dict, e.g. from
            ``DatabaseManager.get_config_at_date``.

    Returns:
        ``{phase: direction}``, direction one of :data:`DIRECTIONS`.  Phases
        without the key are absent.  Two phases may name the same direction
        only if the caller tolerates it; see
        ``left_turn_gap.through_phases``.

    Raises:
        ValueError: A value is not NB/SB/EB/WB (case-insensitive).
    """
    out: Dict[int, str] = {}
    for key, raw in config.items():
        match = _DIRECTION_KEY_RE.match(str(key))
        if not match or _is_blank(raw):
            continue
        tok = str(raw).strip().upper()
        if tok not in DIRECTIONS:
            raise ValueError(f"{key}: direction must be one of {DIRECTIONS}, got {raw!r}")
        out[int(match.group(1))] = tok
    return out
