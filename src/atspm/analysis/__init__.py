"""
ATSPM Analysis Package (Functional Core)

This package contains pure transformation functions with no I/O.
All functions accept data structures (DataFrames, dicts, etc.) and
return transformed data.

Modules:
- decoders: Binary file parsing (DatZ format)
- cycles:   Cycle detection and barrier logic
- counts:   Vehicle and pedestrian count aggregations
- aog:      Arrival on Green calculations
- detector_roles: Detector role table and per-role detector sets
- detector_activity: Per-detector activity profile per local day/window
- clock_marks: eos_set_time drift/set marker decoding
"""

from .decoders import (
    DatZDecodingError,
    parse_datz_bytes,
    parse_datz_batch,
    parse_datz_header,
    validate_datz_file,
    estimate_event_count,
    insert_gap_marker,
    detect_corruption,
)

from .cycles import (
    CycleDetectionError,
    calculate_cycles,
    assign_ring_phases,
    validate_cycles,
    get_cycle_stats,
    assign_events_to_cycles,
)

from .counts import (
    vehicle_counts,
    ped_counts,
    parse_movements_from_config,
    parse_exclusions_from_config,
)

from .detectors import (
    analyze_discrepancies,
)

from .phases import (
    phase_splits,
)

from .aog import (
    arrival_on_green,
    bin_arrival_on_green,
)

from .flow import (
    discharge_profiles,
    flow_rate,
    rate_profiles,
    saturation_state,
)

from .critical import (
    ring_barrier_structure,
    movement_phase_map,
    phase_demand,
    critical_movement_analysis,
)

from .split_failures import split_failures, bin_split_failures
from .optimizer import optimize
from .optimizer_validation import validate_plans
from .clock_marks import (
    MarkerPeds,
    marker_peds_from_config,
    drop_marker_events,
    send_log_pulses,
    pair_marker_pulses,
    decode_clock_marks,
)

from .detector_roles import (
    parse_detector_roles,
    detector_sets,
)

from .detector_activity import detector_activity_profile

__all__ = [
    # Decoders
    'DatZDecodingError',
    'parse_datz_bytes',
    'parse_datz_batch',
    'parse_datz_header',
    'validate_datz_file',
    'estimate_event_count',
    'insert_gap_marker',
    'detect_corruption',
    # Cycles
    'CycleDetectionError',
    'calculate_cycles',
    'assign_ring_phases',
    'validate_cycles',
    'get_cycle_stats',
    'assign_events_to_cycles',
    # Counts
    'vehicle_counts',
    'ped_counts',
    'parse_movements_from_config',
    'parse_exclusions_from_config',
    # Detectors
    'analyze_discrepancies',
    # Phases
    'phase_splits',
    # AoG
    'arrival_on_green',
    'bin_arrival_on_green',
    # Flow
    'flow_rate',
    'rate_profiles',
    'discharge_profiles',
    'saturation_state',
    # Critical
    'ring_barrier_structure',
    'movement_phase_map',
    'phase_demand',
    'critical_movement_analysis',
    # Split failures
    'split_failures',
    'bin_split_failures',
    # Optimizer
    'optimize',
    'validate_plans',
    # Detector roles
    'parse_detector_roles',
    'detector_sets',
    # Detector activity
    'detector_activity_profile',
    # Clock marks
    'MarkerPeds',
    'marker_peds_from_config',
    'drop_marker_events',
    'send_log_pulses',
    'pair_marker_pulses',
    'decode_clock_marks',
]