"""
ATSPM Cycle Detection Logic (Functional Core)

This module handles cycle detection from raw event data using either:
- Path A: Standard barrier pulses (Code 31)
- Path B: Ring-Barrier configuration fallback

Package Location: src/atspm/analysis/cycles.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class CycleDetectionError(Exception):
    """
    Custom exception for cycle detection errors.
    
    Raised when:
    - Config is missing required Ring-Barrier data
    - No valid cycle detection method available
    """
    pass


def calculate_cycles(
    events_df: pd.DataFrame, 
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate cycle start times and coordination plans from raw events.
    
    Uses a two-path approach:
    - Path A: Standard barrier pulses (Code 31) if available and valid
    - Path B: Ring-Barrier configuration fallback for missing/bad Code 31
    
    Args:
        events_df: Raw events with columns ['timestamp', 'event_code', 'parameter']
        config: Configuration dict with Ring-Barrier data (RB_R1, RB_R2, etc.)
    
    Returns:
        DataFrame with columns: ['cycle_start', 'coord_plan', 'detection_method']
        Sorted by cycle_start
    
    Raises:
        CycleDetectionError: If cycle detection fails
    
    Example:
        >>> config = {'RB_R1': '1,2|3,4', 'RB_R2': '5,6|7,8'}
        >>> cycles_df = calculate_cycles(events_df, config)
        >>> cycles_df.head()
           cycle_start  coord_plan detection_method
        0   1609459200         1.0   barrier_pulse
        1   1609459280         1.0   barrier_pulse
    """
    if events_df.empty:
        return pd.DataFrame(columns=['cycle_start', 'coord_plan', 'detection_method'])
    
    # Step 1: Assess Code 31 quality
    has_valid_code31 = _check_code31_validity(events_df)
    
    # Step 2: Choose detection path
    if has_valid_code31:
        # Path A: Standard barrier pulse logic
        cycles_df = _detect_cycles_from_barriers(events_df)
        detection_method = 'barrier_pulse'
    else:
        # Path B: Ring-Barrier config fallback
        cycles_df = _detect_cycles_from_config(events_df, config)
        detection_method = 'ring_barrier_config'
    
    if cycles_df.empty:
        # No cycles detected - treat entire dataset as one cycle
        cycles_df = pd.DataFrame([{
            'cycle_start': events_df['timestamp'].min(),
            'detection_method': 'single_cycle_fallback'
        }])
    else:
        cycles_df['detection_method'] = detection_method
    
    # Step 3: Merge coordination plan information
    cycles_df = _merge_coordination_plan(cycles_df, events_df)
    
    # Step 4: Sort and clean
    cycles_df = cycles_df.sort_values('cycle_start').reset_index(drop=True)
    
    return cycles_df[['cycle_start', 'coord_plan', 'detection_method']]


def _check_code31_validity(events_df: pd.DataFrame) -> bool:
    """
    Check if Code 31 (barrier pulse) events are valid.
    
    Valid means:
    - At least one Code 31 exists
    - No duplicate Code 31s at same timestamp
    
    Args:
        events_df: Raw events DataFrame
    
    Returns:
        True if Code 31 data is valid for cycle detection
    """
    code31_events = events_df[events_df['event_code'] == 31]
    
    if len(code31_events) == 0:
        return False
    
    # Check for duplicate timestamps
    timestamp_counts = code31_events.groupby('timestamp').size()
    has_duplicates = (timestamp_counts > 1).any()
    
    return not has_duplicates


def _detect_cycles_from_barriers(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect cycle starts using Code 31 barrier pulse events.
    
    A cycle starts when the barrier parameter decreases
    (wraps back to Ring 1 from Ring 2).
    
    Args:
        events_df: Raw events DataFrame
    
    Returns:
        DataFrame with 'cycle_start' column
    """
    # Filter to Code 31 events
    barriers = events_df[events_df['event_code'] == 31].copy()
    
    if barriers.empty:
        return pd.DataFrame(columns=['cycle_start'])
    
    # Sort by timestamp
    barriers = barriers.sort_values('timestamp').reset_index(drop=True)
    
    # Detect cycle transitions: when parameter decreases
    barriers['next_param'] = barriers['parameter'].shift(-1)
    
    # Cycle starts where next parameter < current parameter
    # (Ring wraps back to beginning)
    cycle_transitions = barriers[barriers['next_param'] < barriers['parameter']]
    
    # The cycle_start is the timestamp of the LAST barrier before wrap
    # (or could be the first barrier of new cycle - depends on convention)
    # Based on legacy code, it's the timestamp where transition occurs
    cycles_df = pd.DataFrame({
        'cycle_start': cycle_transitions['timestamp'].values
    })
    
    return cycles_df


def _detect_cycles_from_config(
    events_df: pd.DataFrame, 
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Detect cycle starts using Ring-Barrier configuration fallback.
    
    Strategy:
    1. Parse Ring-Barrier sequences from config
    2. Map each phase to its group index
    3. Find transitions from later group to earlier group
    
    Args:
        events_df: Raw events DataFrame
        config: Configuration with RB_R1, RB_R2, etc.
    
    Returns:
        DataFrame with 'cycle_start' column
    
    Raises:
        CycleDetectionError: If config is missing or invalid
    """
    # Step 1: Parse Ring-Barrier configuration
    barrier_groups = _parse_ring_barrier_config(config)
    
    if not barrier_groups:
        raise CycleDetectionError(
            "No valid Ring-Barrier configuration found. "
            "Config must contain 'RB_R1' and/or 'RB_R2' entries."
        )
    
    # Step 2: Create phase-to-group mapping
    phase_to_group = {}
    for group_idx, phase_list in enumerate(barrier_groups):
        for phase_id in phase_list:
            phase_to_group[phase_id] = group_idx
    
    # Step 3: Filter for Code 1 (phase green start) events
    phase_events = events_df[events_df['event_code'] == 1].copy()
    
    if phase_events.empty:
        return pd.DataFrame(columns=['cycle_start'])
    
    # Map phases to groups
    phase_events['group_idx'] = phase_events['parameter'].map(phase_to_group)
    
    # Drop phases not in configuration
    phase_events = phase_events.dropna(subset=['group_idx'])
    phase_events['group_idx'] = phase_events['group_idx'].astype(int)
    
    if phase_events.empty:
        return pd.DataFrame(columns=['cycle_start'])
    
    # Sort by timestamp
    phase_events = phase_events.sort_values('timestamp').reset_index(drop=True)
    
    # Step 4: Detect cycle transitions
    # Group events by timestamp (multiple phases can start simultaneously)
    timestamp_groups = phase_events.groupby('timestamp').agg({
        'parameter': list,
        'group_idx': list
    }).reset_index()
    
    # Look for transitions from higher group to lower group
    timestamp_groups['prev_groups'] = timestamp_groups['group_idx'].shift(1)
    
    def is_cycle_transition(row):
        """Check if this timestamp represents a cycle wrap."""
        if pd.isna(row['prev_groups']) or not isinstance(row['prev_groups'], list):
            return False
        
        current_groups = row['group_idx']
        prev_groups = row['prev_groups']
        
        # Cycle transition: min(current) < max(previous)
        # This indicates wrap back to earlier group
        min_current = min(current_groups)
        max_previous = max(prev_groups)
        
        return min_current < max_previous
    
    timestamp_groups['is_transition'] = timestamp_groups.apply(
        is_cycle_transition, axis=1
    )
    
    # Extract cycle start timestamps
    cycle_starts = timestamp_groups[timestamp_groups['is_transition']]['timestamp'].values
    
    cycles_df = pd.DataFrame({
        'cycle_start': cycle_starts
    })
    
    return cycles_df


def _parse_ring_barrier_config(config: Dict[str, Any]) -> List[List[int]]:
    """
    Parse Ring-Barrier configuration into phase groups.
    
    Handles formats:
    - Single ring: RB_R1 = "6|4|8" → [[6], [4], [8]]
    - Dual ring: RB_R1 = "1,2|3,4", RB_R2 = "5,6|7,8" 
                 → [[1,2,5,6], [3,4,7,8]]
    
    Args:
        config: Configuration dict with RB_R1, RB_R2 keys
    
    Returns:
        List of phase groups (each group is a list of phase IDs)
    
    Example:
        >>> config = {'RB_R1': '1,2|3,4', 'RB_R2': '5,6|7,8'}
        >>> _parse_ring_barrier_config(config)
        [[1, 2, 5, 6], [3, 4, 7, 8]]
    """
    r1_str = config.get('RB_R1')
    r2_str = config.get('RB_R2')
    
    groups = []
    
    # Parse R1
    if r1_str and pd.notna(r1_str):
        r1_groups = str(r1_str).split('|')
        for group_str in r1_groups:
            # Parse comma-separated phases
            phase_ids = [
                int(x.strip()) 
                for x in group_str.split(',') 
                if x.strip().isdigit()
            ]
            if phase_ids:
                groups.append(phase_ids)
    
    # Parse R2 and merge with R1 groups
    if r2_str and pd.notna(r2_str):
        r2_groups = str(r2_str).split('|')
        
        for i, group_str in enumerate(r2_groups):
            # Parse comma-separated phases
            phase_ids = [
                int(x.strip()) 
                for x in group_str.split(',') 
                if x.strip().isdigit()
            ]
            
            if phase_ids:
                # Merge with corresponding R1 group
                if i < len(groups):
                    groups[i].extend(phase_ids)
                else:
                    groups.append(phase_ids)
    
    return groups


def _merge_coordination_plan(
    cycles_df: pd.DataFrame, 
    events_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge coordination plan information into cycles DataFrame.
    
    For each cycle, find the most recent Code 131/132 coordination
    plan change event.
    
    Args:
        cycles_df: DataFrame with cycle_start column
        events_df: Raw events DataFrame
    
    Returns:
        cycles_df with added 'coord_plan' column
    """
    # Extract coordination plan events (Code 131 or 132)
    coord_events = events_df[
        events_df['event_code'].isin([131, 132])
    ].copy()
    
    if coord_events.empty:
        # No coordination info - fill with 0
        cycles_df['coord_plan'] = 0.0
        return cycles_df
    
    # Sort by timestamp
    coord_events = coord_events.sort_values('timestamp')
    
    # For each cycle, find the most recent coord plan
    cycles_df = cycles_df.copy()
    cycles_df['coord_plan'] = 0.0
    
    for idx, row in cycles_df.iterrows():
        cycle_time = row['cycle_start']
        
        # Find most recent coord event before or at this cycle
        prior_coord = coord_events[
            coord_events['timestamp'] <= cycle_time
        ]
        
        if not prior_coord.empty:
            # Use the most recent one
            latest_coord = prior_coord.iloc[-1]
            cycles_df.loc[idx, 'coord_plan'] = float(latest_coord['parameter'])
    
    return cycles_df


def validate_cycles(
    cycles_df: pd.DataFrame,
    min_cycle_length: float = 10.0,
    max_cycle_length: float = 300.0
) -> Tuple[bool, List[str]]:
    """
    Validate detected cycles for reasonableness.
    
    Checks:
    - Cycle lengths are within acceptable range
    - No duplicate cycle starts
    - Cycles are properly sorted
    
    Args:
        cycles_df: Cycles DataFrame
        min_cycle_length: Minimum acceptable cycle length (seconds)
        max_cycle_length: Maximum acceptable cycle length (seconds)
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    
    Example:
        >>> is_valid, warnings = validate_cycles(cycles_df)
        >>> if not is_valid:
        ...     print("\\n".join(warnings))
    """
    warnings = []
    
    if cycles_df.empty:
        return True, []
    
    # Check for duplicates
    duplicates = cycles_df['cycle_start'].duplicated()
    if duplicates.any():
        dup_count = duplicates.sum()
        warnings.append(f"Found {dup_count} duplicate cycle start times")
    
    # Check sorting
    is_sorted = cycles_df['cycle_start'].is_monotonic_increasing
    if not is_sorted:
        warnings.append("Cycles are not sorted by cycle_start")
    
    # Calculate cycle lengths
    cycles_df = cycles_df.sort_values('cycle_start')
    cycle_lengths = cycles_df['cycle_start'].diff().dropna()
    
    # Check for unreasonably short cycles
    too_short = cycle_lengths < min_cycle_length
    if too_short.any():
        short_count = too_short.sum()
        min_found = cycle_lengths.min()
        warnings.append(
            f"Found {short_count} cycles shorter than {min_cycle_length}s "
            f"(minimum: {min_found:.1f}s)"
        )
    
    # Check for unreasonably long cycles
    too_long = cycle_lengths > max_cycle_length
    if too_long.any():
        long_count = too_long.sum()
        max_found = cycle_lengths.max()
        warnings.append(
            f"Found {long_count} cycles longer than {max_cycle_length}s "
            f"(maximum: {max_found:.1f}s)"
        )
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings


def get_cycle_stats(cycles_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for detected cycles.
    
    Args:
        cycles_df: Cycles DataFrame
    
    Returns:
        Dictionary with cycle statistics
    
    Example:
        >>> stats = get_cycle_stats(cycles_df)
        >>> print(f"Average cycle: {stats['avg_cycle_length']:.1f}s")
    """
    if cycles_df.empty:
        return {
            'total_cycles': 0,
            'detection_methods': {},
            'coord_plans': {},
            'avg_cycle_length': None,
            'std_cycle_length': None,
        }
    
    # Sort cycles
    cycles_df = cycles_df.sort_values('cycle_start')
    
    # Calculate cycle lengths
    cycle_lengths = cycles_df['cycle_start'].diff().dropna()
    
    # Detection methods breakdown
    method_counts = cycles_df['detection_method'].value_counts().to_dict()
    
    # Coordination plans breakdown
    coord_counts = cycles_df['coord_plan'].value_counts().to_dict()
    
    return {
        'total_cycles': len(cycles_df),
        'detection_methods': method_counts,
        'coord_plans': coord_counts,
        'avg_cycle_length': cycle_lengths.mean() if len(cycle_lengths) > 0 else None,
        'std_cycle_length': cycle_lengths.std() if len(cycle_lengths) > 0 else None,
        'min_cycle_length': cycle_lengths.min() if len(cycle_lengths) > 0 else None,
        'max_cycle_length': cycle_lengths.max() if len(cycle_lengths) > 0 else None,
    }


def assign_events_to_cycles(
    events_df: pd.DataFrame,
    cycles_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign each event to its corresponding cycle.
    
    Uses merge_asof for efficient timestamp-based assignment.
    
    Args:
        events_df: Raw events DataFrame (must have 'timestamp' column)
        cycles_df: Cycles DataFrame (must have 'cycle_start' column)
    
    Returns:
        events_df with added 'cycle_start' and 'coord_plan' columns
    
    Example:
        >>> events_with_cycles = assign_events_to_cycles(events_df, cycles_df)
        >>> # Now can group by cycle
        >>> cycle_groups = events_with_cycles.groupby('cycle_start')
    """
    if events_df.empty or cycles_df.empty:
        events_df['cycle_start'] = pd.NaT
        events_df['coord_plan'] = 0.0
        return events_df
    
    # Ensure both are sorted by timestamp/cycle_start
    events_df = events_df.sort_values('timestamp').copy()
    cycles_df = cycles_df.sort_values('cycle_start').copy()
    
    # Use merge_asof for efficient nearest-match join
    # Each event gets the cycle_start that is <= its timestamp
    result_df = pd.merge_asof(
        events_df,
        cycles_df[['cycle_start', 'coord_plan']],
        left_on='timestamp',
        right_on='cycle_start',
        direction='backward'
    )
    
    # Forward fill any NaN values (events before first cycle)
    result_df['cycle_start'] = result_df['cycle_start'].fillna(
        cycles_df['cycle_start'].iloc[0]
    )
    result_df['coord_plan'] = result_df['coord_plan'].fillna(
        cycles_df['coord_plan'].iloc[0]
    )
    
    return result_df