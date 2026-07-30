"""
ATSPM Data Package (Imperative Shell)

This package handles all I/O operations, database management,
and resource handling for the ATSPM system.

Modules:
- manager:    Database initialization and configuration import
- retrieval:  Device (controller/secondary) .datZ retrieval via SCP
- ingestion:  Event data ingestion from .datZ files
- processing: Cycle detection processing
- reader:     Adapter for reading data in flat format
- counts:     Vehicle and pedestrian count orchestration
- aog:        Arrival on Green calculation orchestration
- video:      Per-camera shape config (loop/stopbar) round-trip
"""

from .manager import DatabaseManager, init_db, import_config
from .retrieval import RetrievalEngine, run_retrieval
from .ingestion import IngestionEngine, run_ingestion
from .processing import CycleProcessor, run_cycle_processing
from .reader import (
    get_events_with_cycles_df,
    get_events_with_cycles_df_by_date,
    get_coordination_data,
    get_config_df,
    get_config_dict,
    get_det_config,
    get_date_range,
    get_available_dates,
    check_data_quality,
    convert_to_datetime,
)

from .counts import (
    CountEngine,
    get_vehicle_counts,
    get_ped_counts,
    get_combined_counts,
)

from.detectors import(
    DetectorEngine,
    get_detector_discrepancies,
)

from .phases import (
    PhaseEngine,
    get_phase_splits,
)

from .aog import (
    AogEngine,
    get_arrival_on_green,
)

from .flow import (
    FlowRateEngine,
    get_flow_rate,
)

from .critical import (
    CriticalMovementEngine,
    get_critical_movements,
)

from .video import (
    ShapeConfig,
    resolve_stopbar_target,
    OVERLAP_LETTER_MAP,
    MIN_PHASE_NUMBER,
    MAX_PHASE_NUMBER,
)

__all__ = [
    # Manager
    'DatabaseManager',
    'init_db',
    'import_config',
    # Retrieval
    'RetrievalEngine',
    'run_retrieval',
    # Ingestion
    'IngestionEngine',
    'run_ingestion',
    # Processing
    'CycleProcessor',
    'run_cycle_processing',
    # Reader
    'get_events_with_cycles_df',
    'get_events_with_cycles_df_by_date',
    'get_coordination_data',
    'get_config_df',
    'get_config_dict',
    'get_det_config',
    'get_date_range',
    'get_available_dates',
    'check_data_quality',
    'convert_to_datetime',
    #Detectors
    'DetectorEngine',
    'get_detector_discrepancies',
    # Counts
    'CountEngine',
    'get_vehicle_counts',
    'get_ped_counts',
    'get_combined_counts',
    # Phases
    'PhaseEngine',
    'get_phase_splits',
    #AoG
    'AogEngine',
    'get_arrival_on_green',
    # Flow
    'FlowRateEngine',
    'get_flow_rate',
    # Critical
    'CriticalMovementEngine',
    'get_critical_movements',
    # Video
    'ShapeConfig',
    'resolve_stopbar_target',
    'OVERLAP_LETTER_MAP',
    'MIN_PHASE_NUMBER',
    'MAX_PHASE_NUMBER',
]