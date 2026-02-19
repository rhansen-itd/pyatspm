"""
ATSPM Cycle Detection Logic (Functional Core)

Pure functions only. No I/O, no SQL, no side effects.
Input/output is DataFrames and plain dicts.

Package Location: src/atspm/analysis/cycles.py

Gap Marker Rule:
    Rows with event_code == -1 mark data discontinuities.  Detection
    functions (_detect_cycles_from_barriers, _detect_cycles_from_config)
    filter on specific event codes (31 and 1 respectively), so gap markers
    never enter those code paths and cannot produce false cycle detections.
    assign_events_to_cycles explicitly preserves gap marker rows with
    NaT / NaN cycle assignments so that downstream consumers are never
    handed a gap marker silently attributed to a real cycle.

Ring/Phase Assignment (Task 0):
    assign_ring_phases() adds r1_phases and r2_phases columns to a cycles
    DataFrame.  Assignment priority:
        1. RB_* columns in config (authoritative ring-barrier definition).
        2. Concurrency-based fallback matching legacy add_r1r2() behaviour.
    The raw comma-separated phase sequence string (e.g. "2,6") is stored
    directly – one string per cycle row – matching the schema target.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

_GAP_CODE: int = -1  # event_code value used for discontinuity markers

# ---------------------------------------------------------------------------
# Default legacy ring membership (mirrors add_r1r2 defaults in misc_tools.py)
# ---------------------------------------------------------------------------
_DEFAULT_R1: List[int] = [1, 2, 3, 4, 9, 10, 11, 12]
_DEFAULT_R2: List[int] = [5, 6, 7, 8, 13, 14, 15, 16]


class CycleDetectionError(Exception):
    """
    Raised when no valid cycle detection method is available or config is
    missing required Ring-Barrier data.
    """
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_cycles(
    events_df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate cycle start times, coordination plans, and ring/phase strings
    from raw events.

    Uses a two-path approach:
    - Path A: Code 31 barrier pulses if present and valid.
    - Path B: Ring-Barrier configuration fallback.

    After cycle detection, ring/phase strings are appended via
    :func:`assign_ring_phases`.

    Args:
        events_df: Raw events with columns [timestamp, event_code, parameter].
        config: Configuration dict.  May contain RB_R1 / RB_R2 entries and/or
            RB_* columns parsed from the config table.

    Returns:
        DataFrame with columns
        [cycle_start, coord_plan, detection_method, r1_phases, r2_phases],
        sorted by cycle_start.

    Raises:
        CycleDetectionError: If no detection method succeeds.
    """
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                'cycle_start', 'coord_plan', 'detection_method',
                'r1_phases', 'r2_phases',
            ]
        )

    has_valid_code31 = _check_code31_validity(events_df)

    if has_valid_code31:
        cycles_df = _detect_cycles_from_barriers(events_df)
        detection_method = 'barrier_pulse'
    else:
        cycles_df = _detect_cycles_from_config(events_df, config)
        detection_method = 'ring_barrier_config'

    if cycles_df.empty:
        cycles_df = pd.DataFrame([{
            'cycle_start': events_df.loc[
                events_df['event_code'] != _GAP_CODE, 'timestamp'
            ].min(),
            'detection_method': 'single_cycle_fallback'
        }])
    else:
        cycles_df['detection_method'] = detection_method

    # Merge coord plan using vectorized backward-fill join
    cycles_df = _merge_coordination_plan(cycles_df, events_df)
    cycles_df = cycles_df.sort_values('cycle_start').reset_index(drop=True)

    # Append ring/phase assignment columns
    cycles_df = assign_ring_phases(cycles_df, events_df, config)

    return cycles_df[[
        'cycle_start', 'coord_plan', 'detection_method',
        'r1_phases', 'r2_phases',
    ]]


def assign_ring_phases(
    cycles_df: pd.DataFrame,
    events_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add ``r1_phases`` and ``r2_phases`` columns to a cycles DataFrame.

    Assignment priority
    ------------------
    1. If *config* contains ``RB_R1`` and/or ``RB_R2`` entries the phases
       listed there are used as the authoritative ring membership lists.
    2. Otherwise the legacy concurrency-based defaults are used
       (R1 = [1,2,3,4,9,10,11,12], R2 = [5,6,7,8,13,14,15,16]).

    The resulting columns contain comma-separated phase number strings,
    e.g. ``"2,6"``, or ``"None"`` when no phases for that ring appeared in a
    given cycle.  This matches the schema column format and the legacy
    ``add_r1r2`` output.

    Gap markers (event_code == -1) in *events_df* are excluded before phase
    grouping so they cannot corrupt ring strings.

    Args:
        cycles_df: DataFrame with at least a ``cycle_start`` column (numeric
            UNIX timestamps or datetime).  Returned with two new columns.
        events_df: Raw events with columns
            [timestamp, event_code, parameter].  Code 1 (green start) rows are
            used to determine which phases appeared in each cycle.
        config: Configuration dict.  Inspected for ``RB_R1`` / ``RB_R2``
            keys to determine ring membership.

    Returns:
        *cycles_df* copy with ``r1_phases`` and ``r2_phases`` appended.

    Example SQL to build events_df for this call::

        -- Pull raw events for the analysis window
        SELECT timestamp, event_code, parameter
        FROM events
        WHERE timestamp BETWEEN :start AND :end
        ORDER BY timestamp;
    """
    cycles_out = cycles_df.copy()

    # Determine ring membership from config or legacy defaults
    r1_members, r2_members = _resolve_ring_membership(config)

    # Code 1 = green start; these define which phases ran in a cycle.
    # Exclude gap markers explicitly (they carry event_code == -1).
    green_events = events_df[
        (events_df['event_code'] == 1) &
        (events_df['event_code'] != _GAP_CODE)
    ][['timestamp', 'parameter']].copy()

    if green_events.empty or cycles_out.empty:
        cycles_out['r1_phases'] = 'None'
        cycles_out['r2_phases'] = 'None'
        return cycles_out

    # Assign each green event to its cycle via backward merge_asof.
    # This is O(n log n) and fully vectorized.
    cycles_sorted = cycles_out[['cycle_start']].sort_values('cycle_start').copy()
    green_sorted = green_events.sort_values('timestamp')

    # Align timestamp types – both must be the same dtype for merge_asof.
    # cycles may be datetime64 or float (UNIX epoch); normalise here.
    cs_col = cycles_sorted['cycle_start']
    ts_col = green_sorted['timestamp']

    if pd.api.types.is_datetime64_any_dtype(cs_col):
        if not pd.api.types.is_datetime64_any_dtype(ts_col):
            green_sorted = green_sorted.copy()
            green_sorted['timestamp'] = pd.to_datetime(
                green_sorted['timestamp'], unit='s', utc=True
            )
    else:
        # Keep as numeric
        pass

    assigned = pd.merge_asof(
        green_sorted,
        cycles_sorted.rename(columns={'cycle_start': '_cs'}),
        left_on='timestamp',
        right_on='_cs',
        direction='backward',
    )
    # Drop unmatched (events before the first cycle)
    assigned = assigned.dropna(subset=['_cs'])
    assigned['_cs'] = assigned['_cs']

    # Build per-cycle phase strings with vectorized groupby
    r1_set = set(r1_members)
    r2_set = set(r2_members)

    assigned['_phase'] = assigned['parameter'].astype(int)
    assigned['_in_r1'] = assigned['_phase'].isin(r1_set)
    assigned['_in_r2'] = assigned['_phase'].isin(r2_set)
    assigned['_phase_str'] = assigned['_phase'].astype(str)

    def _join_phases(series: pd.Series) -> str:
        phases = series.tolist()
        return ','.join(phases) if phases else 'None'

    # R1 strings
    r1_map = (
        assigned[assigned['_in_r1']]
        .groupby('_cs')['_phase_str']
        .agg(_join_phases)
        .rename('r1_phases')
    )

    # R2 strings
    r2_map = (
        assigned[assigned['_in_r2']]
        .groupby('_cs')['_phase_str']
        .agg(_join_phases)
        .rename('r2_phases')
    )

    # Merge back onto cycles
    cycles_out = cycles_out.merge(
        r1_map, left_on='cycle_start', right_index=True, how='left'
    )
    cycles_out = cycles_out.merge(
        r2_map, left_on='cycle_start', right_index=True, how='left'
    )

    cycles_out['r1_phases'] = cycles_out['r1_phases'].fillna('None')
    cycles_out['r2_phases'] = cycles_out['r2_phases'].fillna('None')

    return cycles_out


# ---------------------------------------------------------------------------
# Internal helpers – ring membership resolution
# ---------------------------------------------------------------------------

def _resolve_ring_membership(
    config: Dict[str, Any],
) -> Tuple[List[int], List[int]]:
    """
    Derive R1 / R2 phase membership lists from config.

    Priority
    --------
    1. ``RB_R1`` and ``RB_R2`` keys (pipe-delimited phase groups, e.g.
       ``"1,2|3,4"``).  All phases present across all groups are pooled per
       ring.
    2. Legacy defaults: R1=[1,2,3,4,9,10,11,12], R2=[5,6,7,8,13,14,15,16].

    Args:
        config: Configuration dict potentially containing ``RB_R1``/``RB_R2``.

    Returns:
        Tuple ``(r1_list, r2_list)`` of integer phase IDs.
    """
    def _flatten_ring(raw: Any) -> List[int]:
        """Parse 'group1_phase,group1_phase2|group2_phase,...' → flat list."""
        if not raw or (isinstance(raw, float) and pd.isna(raw)):
            return []
        phases: List[int] = []
        for group_str in str(raw).split('|'):
            for tok in group_str.split(','):
                tok = tok.strip()
                if tok.isdigit():
                    phases.append(int(tok))
        return phases

    r1_raw = config.get('RB_R1') or config.get('RB_r1')
    r2_raw = config.get('RB_R2') or config.get('RB_r2')

    r1 = _flatten_ring(r1_raw)
    r2 = _flatten_ring(r2_raw)

    if not r1 and not r2:
        # Fall back to legacy concurrency-based defaults
        return list(_DEFAULT_R1), list(_DEFAULT_R2)

    # Fill any missing ring with defaults rather than leaving empty
    if not r1:
        r1 = [p for p in _DEFAULT_R1 if p not in r2]
    if not r2:
        r2 = [p for p in _DEFAULT_R2 if p not in r1]

    return r1, r2


# ---------------------------------------------------------------------------
# Internal detection helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _segment_id(events_df: pd.DataFrame) -> pd.Series:
    """
    Assign a monotonically increasing segment ID to every row.

    The ID starts at 0 and increments at each gap marker (event_code == -1),
    so rows within the same continuous block of ingested data share an ID.
    Gap marker rows belong to the closing segment; callers drop them after
    this call.

    Args:
        events_df: DataFrame sorted by timestamp with an event_code column.

    Returns:
        Integer Series aligned with events_df's index.
    """
    return (events_df['event_code'] == _GAP_CODE).cumsum().astype(np.int32)


def _check_code31_validity(events_df: pd.DataFrame) -> bool:
    """
    Check whether Code 31 (barrier pulse) events are present and unambiguous.

    Gap markers are excluded before the check so they cannot produce false
    duplicate timestamps.

    Args:
        events_df: Raw events DataFrame.

    Returns:
        True if at least one Code 31 exists and none share a timestamp.
    """
    code31 = events_df[events_df['event_code'] == 31]
    if code31.empty:
        return False
    return not code31['timestamp'].duplicated().any()


def _detect_cycles_from_barriers(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect cycle starts using Code 31 barrier pulse events.

    A cycle begins when the barrier parameter wraps back to a lower value
    (Ring 2 -> Ring 1 transition).  The shift(1) comparison is performed
    within each data segment (grouped by _seg) so that a post-gap barrier
    is never compared against a pre-gap barrier parameter.  Normal detection
    resumes naturally after a gap -- the first post-gap barrier that
    constitutes a real wrap gets detected; events before that wrap get
    NaT for cycle_start, which is the correct representation of
    mid-cycle resumption after missing data.

    Args:
        events_df: Raw events DataFrame (may contain gap marker rows).

    Returns:
        DataFrame with column [cycle_start].
    """
    df = events_df.sort_values('timestamp').copy()
    df['_seg'] = _segment_id(df)
    barriers = df[df['event_code'] == 31].copy()

    if barriers.empty:
        return pd.DataFrame(columns=['cycle_start'])

    # shift(1) within each segment so the first barrier after a gap is
    # never compared against a pre-gap barrier parameter.
    barriers['_prev_param'] = barriers.groupby('_seg')['parameter'].shift(1)
    wrap_mask = barriers['parameter'] < barriers['_prev_param']
    cycle_starts = barriers.loc[wrap_mask, 'timestamp'].values

    return pd.DataFrame({'cycle_start': cycle_starts})


def _detect_cycles_from_config(
    events_df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Detect cycle starts using Ring-Barrier configuration (fallback path).

    Strategy:
    1. Parse Ring-Barrier sequences from config into barrier groups.
    2. Map each phase to its group index.
    3. Aggregate Code 1 events by (segment, timestamp) and detect wraps
       where min(current group) < max(previous group) within the same
       segment, indicating a return to an earlier barrier group.

    The prev_max_group shift is performed within each data segment so that
    the first post-gap timestamp is never compared against a pre-gap
    max_group.  Normal detection resumes naturally -- events before the
    first post-gap wrap get NaT for cycle_start.

    Args:
        events_df: Raw events DataFrame (may contain gap marker rows).
        config: Config dict with RB_R1 / RB_R2 keys.

    Returns:
        DataFrame with column [cycle_start].

    Raises:
        CycleDetectionError: If RB config is missing or unparseable.
    """
    barrier_groups = _parse_ring_barrier_config(config)

    if not barrier_groups:
        raise CycleDetectionError(
            "No valid Ring-Barrier configuration found. "
            "Config must contain 'RB_R1' and/or 'RB_R2' entries."
        )

    # Build phase -> group_index lookup (vectorised via map)
    phase_to_group: Dict[int, int] = {
        phase_id: group_idx
        for group_idx, phase_list in enumerate(barrier_groups)
        for phase_id in phase_list
    }

    # Assign segments before filtering so gap boundaries are preserved.
    df = events_df.sort_values('timestamp').copy()
    df['_seg'] = _segment_id(df)

    phase_events = df[df['event_code'] == 1].copy()
    if phase_events.empty:
        return pd.DataFrame(columns=['cycle_start'])

    phase_events['group_idx'] = phase_events['parameter'].map(phase_to_group)
    phase_events = phase_events.dropna(subset=['group_idx'])
    if phase_events.empty:
        return pd.DataFrame(columns=['cycle_start'])

    phase_events['group_idx'] = phase_events['group_idx'].astype(np.int8)

    # Aggregate per (segment, timestamp): min and max group_idx present.
    ts_agg = (
        phase_events.groupby(['_seg', 'timestamp'])['group_idx']
        .agg(min_group='min', max_group='max')
        .reset_index()
        .sort_values(['_seg', 'timestamp'])
    )

    # shift(1) within each segment: prev_max_group is NaN at every segment
    # boundary, so the first timestamp after a gap never triggers a wrap.
    ts_agg['prev_max_group'] = ts_agg.groupby('_seg')['max_group'].shift(1)
    wrap_mask = ts_agg['min_group'] < ts_agg['prev_max_group']
    cycle_starts = ts_agg.loc[wrap_mask, 'timestamp'].values

    return pd.DataFrame({'cycle_start': cycle_starts})


def _parse_ring_barrier_config(config: Dict[str, Any]) -> List[List[int]]:
    """
    Parse Ring-Barrier configuration into phase groups.

    Handles formats:
    - Single ring: RB_R1='6|4|8'           → [[6], [4], [8]]
    - Dual ring:   RB_R1='1,2|3,4',
                   RB_R2='5,6|7,8'          → [[1,2,5,6], [3,4,7,8]]

    Args:
        config: Configuration dict with optional RB_R1 / RB_R2 keys.

    Returns:
        List of phase groups; empty list if config is absent or unparseable.
    """
    def parse_ring(ring_str: Any) -> List[List[int]]:
        if not ring_str or (isinstance(ring_str, float) and pd.isna(ring_str)):
            return []
        groups = []
        for group_str in str(ring_str).split('|'):
            ids = [int(x) for x in group_str.split(',') if x.strip().isdigit()]
            if ids:
                groups.append(ids)
        return groups

    r1_groups = parse_ring(config.get('RB_R1'))
    r2_groups = parse_ring(config.get('RB_R2'))

    if not r1_groups and not r2_groups:
        return []

    # Merge R2 into R1 by position; extend if R2 is longer
    n = max(len(r1_groups), len(r2_groups))
    groups: List[List[int]] = [[] for _ in range(n)]

    for i, g in enumerate(r1_groups):
        groups[i].extend(g)
    for i, g in enumerate(r2_groups):
        groups[i].extend(g)

    return [g for g in groups if g]


def _merge_coordination_plan(
    cycles_df: pd.DataFrame,
    events_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign each cycle its active coordination plan via a vectorized
    backward-fill join.

    For each cycle_start timestamp, the coord plan is the parameter value
    of the most recent Code 131 or 132 event at or before that timestamp.
    Uses pd.merge_asof (O(n log n)) rather than a Python loop (O(n*m)).

    Args:
        cycles_df: DataFrame with column [cycle_start].
        events_df: Raw events DataFrame.

    Returns:
        cycles_df with column [coord_plan] added (float, 0.0 if unknown).
    """
    coord_events = (
        events_df[events_df['event_code'].isin([131, 132])]
        [['timestamp', 'parameter']]
        .sort_values('timestamp')
        .drop_duplicates(subset='timestamp', keep='last')
        .rename(columns={'parameter': 'coord_plan'})
    )

    cycles_sorted = cycles_df.sort_values('cycle_start').copy()

    if coord_events.empty:
        cycles_sorted['coord_plan'] = 0.0
        return cycles_sorted

    # merge_asof: for each cycle_start, find the last coord event ≤ cycle_start
    merged = pd.merge_asof(
        cycles_sorted,
        coord_events,
        left_on='cycle_start',
        right_on='timestamp',
        direction='backward'
    )

    merged['coord_plan'] = merged['coord_plan'].fillna(0.0).astype(float)
    merged = merged.drop(columns=['timestamp'], errors='ignore')

    return merged


# ---------------------------------------------------------------------------
# Utility functions (unchanged public API)
# ---------------------------------------------------------------------------

def validate_cycles(
    cycles_df: pd.DataFrame,
    min_cycle_length: float = 10.0,
    max_cycle_length: float = 300.0
) -> Tuple[bool, List[str]]:
    """
    Validate detected cycles for reasonableness.

    Args:
        cycles_df: Cycles DataFrame with [cycle_start] column.
        min_cycle_length: Minimum acceptable inter-cycle gap in seconds.
        max_cycle_length: Maximum acceptable inter-cycle gap in seconds.

    Returns:
        Tuple of (is_valid, list_of_warning_strings).
    """
    warnings_out: List[str] = []

    if cycles_df.empty:
        return True, []

    if cycles_df['cycle_start'].duplicated().any():
        warnings_out.append(
            f"Found {cycles_df['cycle_start'].duplicated().sum()} "
            "duplicate cycle start times"
        )

    sorted_starts = cycles_df['cycle_start'].sort_values()
    if not cycles_df['cycle_start'].is_monotonic_increasing:
        warnings_out.append("Cycles are not sorted by cycle_start")

    lengths = sorted_starts.diff().dropna()

    too_short = lengths < min_cycle_length
    if too_short.any():
        warnings_out.append(
            f"Found {too_short.sum()} cycles shorter than {min_cycle_length}s "
            f"(minimum: {lengths.min():.1f}s)"
        )

    too_long = lengths > max_cycle_length
    if too_long.any():
        warnings_out.append(
            f"Found {too_long.sum()} cycles longer than {max_cycle_length}s "
            f"(maximum: {lengths.max():.1f}s)"
        )

    return len(warnings_out) == 0, warnings_out


def get_cycle_stats(cycles_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for detected cycles.

    Args:
        cycles_df: Cycles DataFrame.

    Returns:
        Dictionary with keys: total_cycles, detection_methods, coord_plans,
        avg_cycle_length, std_cycle_length, min_cycle_length, max_cycle_length.
    """
    if cycles_df.empty:
        return {
            'total_cycles': 0,
            'detection_methods': {},
            'coord_plans': {},
            'avg_cycle_length': None,
            'std_cycle_length': None,
            'min_cycle_length': None,
            'max_cycle_length': None,
        }

    lengths = cycles_df['cycle_start'].sort_values().diff().dropna()

    return {
        'total_cycles': len(cycles_df),
        'detection_methods': cycles_df['detection_method'].value_counts().to_dict(),
        'coord_plans': cycles_df['coord_plan'].value_counts().to_dict(),
        'avg_cycle_length': float(lengths.mean()) if not lengths.empty else None,
        'std_cycle_length': float(lengths.std()) if not lengths.empty else None,
        'min_cycle_length': float(lengths.min()) if not lengths.empty else None,
        'max_cycle_length': float(lengths.max()) if not lengths.empty else None,
    }


def assign_events_to_cycles(
    events_df: pd.DataFrame,
    cycles_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign each event to its containing cycle via a backward merge_asof join.

    Args:
        events_df: Raw events DataFrame (must have [timestamp] column).
        cycles_df: Cycles DataFrame (must have [cycle_start, coord_plan]).

    Returns:
        events_df with [cycle_start, coord_plan] columns added.
    """
    if events_df.empty or cycles_df.empty:
        events_df = events_df.copy()
        events_df['cycle_start'] = pd.NaT
        events_df['coord_plan'] = 0.0
        return events_df

    events_sorted = events_df.sort_values('timestamp').copy()
    cycles_sorted = cycles_df.sort_values('cycle_start').copy()

    result = pd.merge_asof(
        events_sorted,
        cycles_sorted[['cycle_start', 'coord_plan']],
        left_on='timestamp',
        right_on='cycle_start',
        direction='backward'
    )

    first_cycle = cycles_sorted['cycle_start'].iloc[0]
    first_plan = cycles_sorted['coord_plan'].iloc[0]
    result['cycle_start'] = result['cycle_start'].fillna(first_cycle)
    result['coord_plan'] = result['coord_plan'].fillna(first_plan)

    # Gap marker rows must not be attributed to any real cycle.
    # Null out their cycle assignment so downstream consumers (comb_gyr_det,
    # phase_status, etc.) never treat a discontinuity row as a real event
    # belonging to the cycle that preceded the gap.
    gap_mask = result['event_code'] == _GAP_CODE
    if gap_mask.any():
        result.loc[gap_mask, 'cycle_start'] = pd.NaT
        result.loc[gap_mask, 'coord_plan'] = float('nan')

    return result
