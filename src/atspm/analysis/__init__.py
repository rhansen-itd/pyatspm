"""
ATSPM Analysis Package (Functional Core)

This package contains pure transformation functions with no I/O.
All functions accept data structures (DataFrames, dicts, etc.) and
return transformed data.

Modules:
- decoders: Binary file parsing (DatZ format)
- cycles: Cycle detection and barrier logic
- moes: MOE calculations (planned)
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
    validate_cycles,
    get_cycle_stats,
    assign_events_to_cycles,
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
    'validate_cycles',
    'get_cycle_stats',
    'assign_events_to_cycles',
]