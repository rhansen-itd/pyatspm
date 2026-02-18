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
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

_GAP_CODE: int = -1  # event_code value used for discontinuity markers


class CycleDetectionError(Exception):
    """
    Raised when no valid cycle detection method is available or config is
    missing required Ring-Barrier data.
    """
    pass


def calculate_cycles(
    events_df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate cycle start times and coordination plans from raw events.

    Uses a two-path approach:
    - Path A: Code 31 barrier pulses if present and valid.
    - Path B: Ring-Barrier configuration fallback.

    Args:
        events_df: Raw events with columns [timestamp, event_code, parameter].
        config: Configuration dict with RB_R1 / RB_R2 entries.

    Returns:
        DataFrame with columns [cycle_start, coord_plan, detection_method],
        sorted by cycle_start.

    Raises:
        CycleDetectionError: If no detection method succeeds.
    """
    if events_df.empty:
        return pd.DataFrame(columns=['cycle_start', 'coord_plan', 'detection_method'])

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
    return cycles_df[['cycle_start', 'coord_plan', 'detection_method']]


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
    # Including _seg in the groupby key keeps segments isolated.
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
    - Single ring: RB_R1='6|4|8'           â†’ [[6], [4], [8]]
    - Dual ring:   RB_R1='1,2|3,4',
                   RB_R2='5,6|7,8'          â†’ [[1,2,5,6], [3,4,7,8]]

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

    # merge_asof: for each cycle_start, find the last coord event â‰¤ cycle_start
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