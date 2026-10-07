"""
Critical Movement Analysis (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.
Input / output is DataFrames and plain Python scalars.

Algorithm overview
------------------
Classic critical movement analysis on a dual-ring, barriered controller:

1. :func:`ring_barrier_structure` — the ring/barrier concurrency structure
   from ``RB_R1`` / ``RB_R2`` config, cross-checked against the phase
   sequences actually observed in the ``cycles`` table
   (``r1_phases`` / ``r2_phases``).
2. :func:`movement_phase_map` — movement labels (``TM_*`` config) mapped to
   signal phases by intersecting each movement's detector set with the
   per-phase stop-bar detector sets (``Det_P{N}_Stopbar`` /
   ``Det_P{N}_Stop_Bar`` config).
3. :func:`phase_demand` — time-binned hourly movement counts summed into
   per-phase demand for the analysis period, divided per lane using the
   ``Lanes:`` config (:func:`parse_lane_config`) when present.
4. :func:`critical_movement_analysis` — the critical phase per concurrent
   slot and the critical path per barrier group: the ring whose summed
   demand (the required-time proxy) is larger.

Required-time proxy
-------------------
Demand — total vph or per-lane vphpl (``basis``) — stands in for required
green time, i.e. equal discharge rates are assumed across phases.  The
throughput optimizer replaces this proxy with measured discharge curves
(``atspm.analysis.flow``); this module's structural outputs (rings, barrier
groups, demand per phase) feed it unchanged.

Gap Marker Rule
---------------
No duration or sequential pairing is computed here.  The observed-sequence
input (``r1_phases`` / ``r2_phases``) is produced by
``atspm.analysis.cycles.assign_ring_phases``, which already bounds the
green-to-cycle join at gap markers; movement counts are aggregated
gap-aware by ``atspm.analysis.counts.vehicle_counts``.

Package Location: src/atspm/analysis/critical.py
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

from .counts import parse_movements_from_config
from .cycles import _parse_ring_groups
from .detector_roles import detector_sets, parse_detector_roles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NEMA-standard dual-ring fallback used when no RB_* config is present
_DEFAULT_R1_GROUPS: List[List[int]] = [[1, 2], [3, 4]]
_DEFAULT_R2_GROUPS: List[List[int]] = [[5, 6], [7, 8]]

# Output schemas
_STRUCTURE_SCHEMA = [
    "phase", "ring", "barrier_group", "position",
    "in_config", "observed_share", "source",
]
_MAP_SCHEMA = ["movement", "phase", "detectors", "n_detectors", "n_matched"]
_DEMAND_SCHEMA = [
    "phase", "movements", "n_detectors", "n_lanes", "lane_source",
    "demand_vph", "peak_vph", "demand_per_lane", "peak_per_lane",
]
_PHASE_SCHEMA = [
    "phase", "ring", "barrier_group", "position", "movements",
    "n_detectors", "n_lanes", "lane_source",
    "demand_vph", "demand_per_lane", "has_demand",
    "slot_critical", "on_critical_path",
]
_GROUP_SCHEMA = [
    "barrier_group", "ring", "phases", "n_phases",
    "demand_sum", "basis", "is_critical_path",
]

# Valid demand bases for critical_movement_analysis
DEMAND_BASES = ("per_lane", "total")

# Lane-count provenance in phase_demand output, most to least physical
LANE_SOURCES = ("layout", "movement", "detectors")

# Lane designations a Lanes_{dir}_Layout token may combine
_LANE_TURNS = frozenset("ULTR")

# Movement label → (approach, turn), e.g. 'EBT' → ('EB', 'T')
_MOVEMENT_RE = re.compile(r"^([A-Z]{2})([ULTR])$")


class LaneConfig(NamedTuple):
    """Parsed ``Lanes:`` config rows.

    Attributes:
        layouts: Approach (``'EB'``) → left-to-right lanes, each the
            frozenset of turns it serves (``'L|T|TR'`` →
            ``({'L'}, {'T'}, {'T', 'R'})``).
        counts: Movement label (``'EBT'``) → lanes serving it; a shared
            lane counts toward every movement it serves.
    """

    layouts: Dict[str, Tuple[frozenset, ...]]
    counts: Dict[str, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _observed_ring_share(cycles_df: pd.DataFrame, column: str) -> pd.Series:
    """Fraction of cycles in which each phase appears in one ring column.

    A phase re-served within a single cycle (listed twice in the same
    string) is counted once — the share measures presence, not frequency.

    Args:
        cycles_df: Cycles rows with the *column* of comma-joined phase
            strings (``"2,6"``) or the literal ``"None"``.
        column: ``'r1_phases'`` or ``'r2_phases'``.

    Returns:
        Float Series indexed by phase ID; empty when no phases appear.
    """
    if cycles_df.empty or column not in cycles_df.columns:
        return pd.Series(dtype=float)

    exploded = cycles_df[column].astype(str).str.split(",").explode()
    exploded = exploded.str.strip()
    exploded = exploded[exploded.str.isdigit()]

    if exploded.empty:
        return pd.Series(dtype=float)

    present = (
        exploded.astype(int)
        .reset_index()
        .drop_duplicates()  # one row per (cycle row, phase)
    )
    return present[column].value_counts() / float(len(cycles_df))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ring_barrier_structure(
    config: Dict[str, Any],
    cycles_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Derive the ring/barrier concurrency structure for an intersection.

    Structure source priority:

    1. ``RB_R1`` / ``RB_R2`` config values (pipe-delimited barrier groups,
       e.g. ``'1,2|3,4'``) — authoritative when present.
    2. NEMA-standard dual-ring fallback
       (R1 = ``1,2|3,4``, R2 = ``5,6|7,8``) with ``source='default'``.

    When *cycles_df* is provided, each structure row gains the fraction of
    observed cycles in which the phase actually ran (``observed_share``),
    and phases observed in a ring but absent from the configured structure
    are appended with ``in_config=False`` and no barrier-group placement —
    the caller should surface these as config/structure mismatches.

    Args:
        config: Active config dict, inspected for ``RB_R1`` / ``RB_R2``.
        cycles_df: Optional cycles rows with ``r1_phases`` / ``r2_phases``
            comma-joined phase strings (``'None'`` for an empty ring).

    Returns:
        DataFrame with one row per (ring, phase)::

            phase           int
            ring            int    – 1 or 2
            barrier_group   float  – 0-based group index; NaN when the
                                     phase is observed but not configured
            position        float  – 1-based order within the ring's group;
                                     NaN when unplaced
            in_config       bool
            observed_share  float  – fraction of cycles the phase ran in
                                     this ring; NaN when *cycles_df* absent
            source          str    – 'config' or 'default'
    """
    r1_groups = _parse_ring_groups(
        config.get("RB_R1") or config.get("RB_r1")
    )
    r2_groups = _parse_ring_groups(
        config.get("RB_R2") or config.get("RB_r2")
    )

    if not r1_groups and not r2_groups:
        r1_groups, r2_groups = _DEFAULT_R1_GROUPS, _DEFAULT_R2_GROUPS
        source = "default"
    else:
        source = "config"

    rows = [
        {
            "phase": phase,
            "ring": ring,
            "barrier_group": float(group_idx),
            "position": float(pos + 1),
            "in_config": source == "config",
        }
        for ring, groups in ((1, r1_groups), (2, r2_groups))
        for group_idx, group in enumerate(groups)
        for pos, phase in enumerate(group)
    ]
    structure = pd.DataFrame(rows, columns=_STRUCTURE_SCHEMA[:5])

    shares = {
        1: _observed_ring_share(cycles_df, "r1_phases"),
        2: _observed_ring_share(cycles_df, "r2_phases"),
    } if cycles_df is not None else {1: pd.Series(dtype=float),
                                     2: pd.Series(dtype=float)}

    structure["observed_share"] = [
        shares[ring].get(phase, 0.0 if cycles_df is not None else float("nan"))
        for ring, phase in zip(structure["ring"], structure["phase"])
    ]

    # Observed-but-unconfigured phases: appended, unplaced in any group.
    extra_rows = [
        {
            "phase": int(phase),
            "ring": ring,
            "barrier_group": float("nan"),
            "position": float("nan"),
            "in_config": False,
            "observed_share": share,
        }
        for ring in (1, 2)
        for phase, share in shares[ring].items()
        if not (
            (structure["ring"] == ring) & (structure["phase"] == phase)
        ).any()
    ]
    if extra_rows:
        structure = pd.concat(
            [structure, pd.DataFrame(extra_rows)], ignore_index=True
        )

    structure["source"] = source
    return (
        structure[_STRUCTURE_SCHEMA]
        .sort_values(["ring", "barrier_group", "position", "phase"])
        .reset_index(drop=True)
    )


def parse_lane_config(config: Dict[str, Any]) -> LaneConfig:
    """Parse ``Lanes_*`` config keys into per-approach layouts and counts.

    ``int_cfg.csv`` ``Lanes:`` rows import as ``Lanes_{movement}`` (lanes
    serving the movement, e.g. ``Lanes_EBT = '2'``) and
    ``Lanes_{dir}_Layout`` (left-to-right lane designations, e.g.
    ``Lanes_EB_Layout = 'L|T|TR'``).  Blank values are skipped.

    Args:
        config: Active config dict.

    Returns:
        :class:`LaneConfig`; both maps empty when no ``Lanes_*`` key is set.

    Raises:
        ValueError: On a layout token that isn't a non-repeating
            combination of ``U``/``L``/``T``/``R``, or a non-integer or
            negative movement lane count.
    """
    layouts: Dict[str, Tuple[frozenset, ...]] = {}
    counts: Dict[str, int] = {}
    for key, value in config.items():
        if not key.startswith("Lanes_"):
            continue
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if not text:
            continue
        name = key[len("Lanes_"):]

        if name.endswith("_Layout"):
            lanes = []
            for token in text.split("|"):
                token = token.strip().upper()
                turns = frozenset(token)
                if (not token or len(turns) != len(token)
                        or not turns <= _LANE_TURNS):
                    raise ValueError(
                        f"{key}: invalid lane designation {token!r} in "
                        f"{text!r} (expected U/L/T/R combinations, "
                        f"e.g. 'L|T|TR')."
                    )
                lanes.append(turns)
            layouts[name[:-len("_Layout")]] = tuple(lanes)
        else:
            try:
                value_f = float(text)
            except ValueError:
                value_f = -1.0
            n = int(value_f) if value_f.is_integer() else -1
            if n < 0:
                raise ValueError(
                    f"{key}: lane count must be a non-negative integer, "
                    f"got {text!r}."
                )
            counts[name] = n
    return LaneConfig(layouts, counts)


def _phase_lanes(
    movements: List[str],
    lanes: LaneConfig,
) -> Tuple[Optional[int], str]:
    """Count the physical lanes serving one phase's movements.

    Per approach, a ``Layout`` counts each **distinct** lane that serves
    any of the phase's turns, so a shared lane is never counted twice.
    Without a layout, the approach falls back to summing its movements'
    ``Lanes_{movement}`` counts.

    Args:
        movements: The phase's movement labels.
        lanes: Output of :func:`parse_lane_config`.

    Returns:
        ``(n_lanes, source)`` — ``source`` is ``'layout'`` when every
        approach had a layout, else ``'movement'``.  ``(None,
        'detectors')`` when any movement has no lane information or the
        total is zero.
    """
    turns_by_dir: Dict[str, set] = {}
    unparsed: List[str] = []
    for label in movements:
        match = _MOVEMENT_RE.match(label)
        if match and match.group(1) in lanes.layouts:
            turns_by_dir.setdefault(match.group(1), set()).add(match.group(2))
        else:
            unparsed.append(label)

    if any(label not in lanes.counts for label in unparsed):
        return None, "detectors"

    n_lanes = sum(
        sum(1 for lane in lanes.layouts[d] if lane & turns)
        for d, turns in turns_by_dir.items()
    ) + sum(lanes.counts[label] for label in unparsed)

    if n_lanes == 0:
        return None, "detectors"
    return n_lanes, "movement" if unparsed else "layout"


def movement_phase_map(config: Dict[str, Any]) -> pd.DataFrame:
    """Map movement labels to signal phases via stop-bar detector overlap.

    A movement (``TM_{label}`` detector list) is assigned to the phase
    whose stop-bar detector set (``Det_P{N}_Stopbar`` /
    ``Det_P{N}_Stop_Bar``) shares the most detectors with it.  A movement
    with no overlap anywhere, or with the same maximal overlap against two
    or more phases (ambiguous), is left unmapped (``phase`` = NA) — the
    caller should surface unmapped movements rather than guess.

    Args:
        config: Active config dict with ``TM_*`` movement keys and
            per-phase stop-bar detector keys.

    Returns:
        DataFrame with one row per movement::

            movement     str
            phase        Int64  – assigned phase, NA when unmapped
            detectors    str    – comma-joined movement detector IDs
            n_detectors  int
            n_matched    int    – detectors shared with the assigned phase
                                  (0 when unmapped)
    """
    movements = parse_movements_from_config(config)
    stopbar_sets = detector_sets(parse_detector_roles(config), "stop_bar")

    rows = []
    for label in sorted(movements):
        det_ids = movements[label]
        det_set = frozenset(det_ids)
        overlaps = {
            phase: len(det_set & sb_set)
            for phase, sb_set in stopbar_sets.items()
            if det_set & sb_set
        }

        phase: Any = pd.NA
        n_matched = 0
        if overlaps:
            best = max(overlaps.values())
            best_phases = [p for p, n in overlaps.items() if n == best]
            if len(best_phases) == 1:
                phase = best_phases[0]
                n_matched = best

        rows.append({
            "movement": label,
            "phase": phase,
            "detectors": ",".join(str(d) for d in det_ids),
            "n_detectors": len(det_set),
            "n_matched": n_matched,
        })

    result = pd.DataFrame(rows, columns=_MAP_SCHEMA)
    result["phase"] = result["phase"].astype("Int64")
    return result


def phase_demand(
    counts_df: pd.DataFrame,
    movement_map: pd.DataFrame,
    lanes: Optional[LaneConfig] = None,
) -> pd.DataFrame:
    """Aggregate time-binned hourly movement counts into per-phase demand.

    Per-bin movement rates are summed into per-phase time series first, so
    ``peak_vph`` reflects the phase's true busiest bin rather than a sum of
    per-movement peaks from different bins.

    Args:
        counts_df: Time-indexed DataFrame of movement columns holding
            **hourly flow rates** (``vehicle_counts(..., hourly=True)``).
            Non-movement columns (``TEV``, ``coverage``, ``data_quality``,
            detector IDs) are ignored.
        movement_map: Output of :func:`movement_phase_map`.  Unmapped
            movements (``phase`` = NA) are excluded.
        lanes: Output of :func:`parse_lane_config`.  ``None`` or empty
            uses the detector-count proxy for every phase.

    Returns:
        DataFrame with one row per phase::

            phase           int
            movements       str    – comma-joined contributing movements
            n_detectors     int    – distinct detectors across movements
            n_lanes         int    – per-lane divisor (see lane_source)
            lane_source     str    – 'layout': distinct lanes from the
                                     approach layouts; 'movement': summed
                                     Lanes_{movement} counts on at least
                                     one approach (a shared lane counts
                                     once per movement); 'detectors':
                                     no lane config covers every
                                     movement, n_lanes = n_detectors
            demand_vph      float  – mean of the per-phase bin rates
            peak_vph        float  – max of the per-phase bin rates
            demand_per_lane float  – demand_vph / n_lanes
            peak_per_lane   float  – peak_vph / n_lanes

        Empty (correct schema) when nothing is mapped or counted.
    """
    empty = pd.DataFrame(columns=_DEMAND_SCHEMA)

    mapped = movement_map.dropna(subset=["phase"])
    if counts_df.empty or mapped.empty:
        return empty

    available = mapped.loc[mapped["movement"].isin(counts_df.columns)]
    if available.empty:
        return empty

    phase_of = dict(zip(available["movement"], available["phase"].astype(int)))
    rates = counts_df[list(phase_of)].fillna(0.0)
    phase_ts = rates.T.groupby(rates.columns.map(phase_of)).sum().T

    lane_counts = {
        int(phase): len({
            int(tok)
            for dets in group["detectors"]
            for tok in dets.split(",")
            if tok.strip().isdigit()
        })
        for phase, group in available.groupby("phase")
    }
    movement_lists = (
        available.groupby(available["phase"].astype(int))["movement"]
        .agg(list)
        .reindex(phase_ts.columns)
    )
    n_detectors = [lane_counts[int(p)] for p in phase_ts.columns]

    if lanes is None or not (lanes.layouts or lanes.counts):
        lane_info = [(None, "detectors")] * len(n_detectors)
    else:
        lane_info = [_phase_lanes(mvs, lanes) for mvs in movement_lists]
    n_lanes = [
        n if n is not None else n_det
        for (n, _), n_det in zip(lane_info, n_detectors)
    ]

    result = pd.DataFrame({
        "phase": phase_ts.columns.astype(int),
        "movements": [",".join(mvs) for mvs in movement_lists],
        "n_detectors": n_detectors,
        "n_lanes": n_lanes,
        "lane_source": [source for _, source in lane_info],
        "demand_vph": phase_ts.mean().values,
        "peak_vph": phase_ts.max().values,
    })
    result["demand_per_lane"] = result["demand_vph"] / result["n_lanes"]
    result["peak_per_lane"] = result["peak_vph"] / result["n_lanes"]

    return (
        result[_DEMAND_SCHEMA]
        .round({"demand_vph": 1, "peak_vph": 1,
                "demand_per_lane": 1, "peak_per_lane": 1})
        .sort_values("phase")
        .reset_index(drop=True)
    )


def critical_movement_analysis(
    structure_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    basis: str = "per_lane",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify critical phases and the critical path per barrier group.

    Both rings must finish before a barrier crossing, so a barrier group's
    required time is the larger of its two ring sums; that ring is the
    group's critical path and its phases are the critical phases.  Within
    each concurrent slot — same barrier group and same position in each
    ring — the phase with the larger demand is additionally flagged
    ``slot_critical`` (positional pairing is nominal: actual dual-ring
    concurrency can slide within the group).

    Args:
        structure_df: Output of :func:`ring_barrier_structure`.
        demand_df: Output of :func:`phase_demand`.
        basis: ``'per_lane'`` (default) compares ``demand_per_lane`` —
            green time scales with per-lane demand; ``'total'`` compares
            ``demand_vph``.

    Returns:
        Tuple ``(phase_df, group_df)``:

        * ``phase_df`` — one row per structure phase::

              phase, ring, barrier_group, position, movements,
              n_detectors, n_lanes, lane_source,
              demand_vph, demand_per_lane,
              has_demand         bool – demand data existed for this phase
              slot_critical      bool
              on_critical_path   bool

        * ``group_df`` — one row per (barrier_group, ring)::

              barrier_group, ring, phases (comma str), n_phases,
              demand_sum (chosen basis), basis, is_critical_path

        Ties between rings resolve to the lower ring number.  Phases
        without a barrier-group placement (observed-but-unconfigured) are
        retained in ``phase_df`` but excluded from ``group_df``.

    Raises:
        ValueError: On an unknown *basis*.
    """
    if basis not in DEMAND_BASES:
        raise ValueError(
            f"Unknown basis {basis!r}; expected one of {DEMAND_BASES}."
        )

    if structure_df.empty:
        return (
            pd.DataFrame(columns=_PHASE_SCHEMA),
            pd.DataFrame(columns=_GROUP_SCHEMA),
        )

    value_col = "demand_per_lane" if basis == "per_lane" else "demand_vph"

    phase_df = structure_df.merge(
        demand_df[["phase", "movements", "n_detectors", "n_lanes",
                   "lane_source", "demand_vph", "demand_per_lane"]],
        on="phase",
        how="left",
    )
    phase_df["has_demand"] = phase_df["demand_vph"].notna()
    phase_df[["demand_vph", "demand_per_lane"]] = (
        phase_df[["demand_vph", "demand_per_lane"]].fillna(0.0)
    )
    phase_df["movements"] = phase_df["movements"].fillna("")
    phase_df[["n_detectors", "n_lanes"]] = (
        phase_df[["n_detectors", "n_lanes"]].fillna(0).astype(int)
    )
    phase_df["lane_source"] = phase_df["lane_source"].fillna("")

    placed = phase_df["barrier_group"].notna()

    # --- Critical path: per (group, ring) demand sums -------------------
    group_df = (
        phase_df.loc[placed]
        .groupby(["barrier_group", "ring"], as_index=False)
        .agg(
            phases=("phase", lambda s: ",".join(str(int(p)) for p in s)),
            n_phases=("phase", "size"),
            demand_sum=(value_col, "sum"),
        )
    )
    group_df["basis"] = basis

    # Larger ring sum wins; ties resolve to the lower ring number because
    # sort keeps ring order stable within each group.
    group_df = group_df.sort_values(["barrier_group", "ring"])
    critical_idx = (
        group_df.groupby("barrier_group")["demand_sum"].idxmax()
    )
    group_df["is_critical_path"] = group_df.index.isin(critical_idx)
    group_df = (
        group_df[_GROUP_SCHEMA]
        .round({"demand_sum": 1})
        .reset_index(drop=True)
    )

    critical_rings = set(
        zip(
            group_df.loc[group_df["is_critical_path"], "barrier_group"],
            group_df.loc[group_df["is_critical_path"], "ring"],
        )
    )
    phase_df["on_critical_path"] = [
        (bg, ring) in critical_rings
        for bg, ring in zip(phase_df["barrier_group"], phase_df["ring"])
    ]

    # --- Critical phase per concurrent slot ------------------------------
    slotted = placed & phase_df["position"].notna()
    slot_max = (
        phase_df.loc[slotted]
        .groupby(["barrier_group", "position"])[value_col]
        .transform("max")
    )
    phase_df["slot_critical"] = False
    phase_df.loc[slotted, "slot_critical"] = (
        phase_df.loc[slotted, value_col] >= slot_max
    ) & (phase_df.loc[slotted, value_col] > 0.0)

    phase_df = (
        phase_df[_PHASE_SCHEMA]
        .sort_values(["barrier_group", "ring", "position", "phase"])
        .reset_index(drop=True)
    )

    return phase_df, group_df
