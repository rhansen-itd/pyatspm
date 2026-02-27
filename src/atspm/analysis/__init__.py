"""
ATSPM Analysis Package (Functional Core)

This package contains pure transformation functions with no I/O.
All functions accept data structures (DataFrames, dicts, etc.) and
return transformed data.

Modules:
- decoders: Binary file parsing (DatZ format)
- cycles:   Cycle detection and barrier logic
- counts:   Vehicle and pedestrian count aggregations
"""

from .decoders import (
    DatZDecodingError,
    parse_datz_bytes,
    parse_datz_batch,
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
    analyze_colocated_discrepancies,
)

from .phases import (
    phase_splits,
)

__all__ = [
    # Decoders
    'DatZDecodingError',
    'parse_datz_bytes',
    'parse_datz_batch',
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
    'analyze_colocated_discrepancies',
    # Phases
    'phase_splits',
]