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
- detector_health: Deterministic detector-health rules and onset bursts
- clock_marks: eos_set_time drift/set marker decoding
- timing_actuation: Timing-and-actuation intervals, row layout, finding links
- preempt: Preemption request/service episodes and daily summary
- split_monitor: Programmed-plan timeline (131–149) and per-service split monitor
- yellow_red_actuations: Yellow and red actuations (UDOT YRA) per green-to-green cycle
- call_service: Call-to-service pairing; pedestrian delay and wait time (UDOT)
- green_time_utilization: Actuations per second-of-green bin (UDOT GTU)
- approach_volume: Directional volumes, peak hour, K- and D-factor (UDOT)
- left_turn_gap: Opposing-through gaps for left turns, binned per green (UDOT)
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

from .achd import (
    AchdDecodingError,
    AchdHeader,
    achd_events_from_frame,
    parse_achd_header,
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
from .approach_delay import approach_delay, bin_approach_delay

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
    arrival_travel_times,
    phase_overlaps,
    phase_directions,
)

from .detector_activity import detector_activity_profile
from .detector_health import (
    HealthThresholds,
    apply_ignore,
    detector_health_findings,
    filter_min_severity,
    onset_bursts,
    severity_exit_code,
    wd_ignore,
    wd_profile_windows,
    wd_reboot_windows,
    wd_thresholds,
    wd_units,
)
from .timing_actuation import (
    TIMING_CODES,
    finding_plot_windows,
    ring_phase_order,
    timing_actuation_intervals,
    timing_actuation_rows,
)
from .preempt import (
    PREEMPT_CODES,
    preempt_episodes,
    preempt_summary,
)
from .split_monitor import (
    plan_timeline,
    programmed_at,
    split_monitor,
    split_monitor_stats,
)
from .yellow_red_actuations import summarize_yellow_red, yellow_red_actuations
from .call_service import (
    ped_delay,
    summarize_ped_delay,
    summarize_wait_time,
    wait_time,
)
from .green_time_utilization import (
    green_time_utilization,
    summarize_gtu_bins,
    summarize_gtu_splits,
)
from .approach_volume import approach_volume, direction_detectors
from .left_turn_gap import (
    left_turn_gaps,
    left_turn_pairs,
    summarize_left_turn_gaps,
    through_phases,
)

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
    # ACHD CSV parser
    'AchdDecodingError',
    'AchdHeader',
    'achd_events_from_frame',
    'parse_achd_header',
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
    'arrival_travel_times',
    'phase_overlaps',
    'phase_directions',
    # Approach delay
    'approach_delay',
    'bin_approach_delay',
    # Detector activity
    'detector_activity_profile',
    'HealthThresholds',
    'detector_health_findings',
    'onset_bursts',
    'apply_ignore',
    'filter_min_severity',
    'severity_exit_code',
    'wd_ignore',
    'wd_profile_windows',
    'wd_reboot_windows',
    'wd_thresholds',
    'wd_units',
    # Clock marks
    'MarkerPeds',
    'marker_peds_from_config',
    'drop_marker_events',
    'send_log_pulses',
    'pair_marker_pulses',
    'decode_clock_marks',
    # Timing and actuation
    'TIMING_CODES',
    'finding_plot_windows',
    'ring_phase_order',
    'timing_actuation_intervals',
    'timing_actuation_rows',
    # Preemption
    'PREEMPT_CODES',
    'preempt_episodes',
    'preempt_summary',
    # Split monitor
    'plan_timeline',
    'programmed_at',
    'split_monitor',
    'split_monitor_stats',
    'yellow_red_actuations',
    'summarize_yellow_red',
    # Call-to-service pairing
    'ped_delay',
    'summarize_ped_delay',
    'wait_time',
    'summarize_wait_time',
    # Green time utilization
    'green_time_utilization',
    'summarize_gtu_bins',
    'summarize_gtu_splits',
    # Approach volume
    'approach_volume',
    'left_turn_gaps',
    'left_turn_pairs',
    'summarize_left_turn_gaps',
    'through_phases',
    'direction_detectors',
]