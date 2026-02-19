"""
ATSPM Data Package (Imperative Shell)

This package handles all I/O operations, database management,
and resource handling for the ATSPM system.

Modules:
- manager:    Database initialization and configuration import
- ingestion:  Event data ingestion from .datZ files
- processing: Cycle detection processing
- reader:     Legacy adapter for reading data in flat format
- counts:     Vehicle and pedestrian count orchestration
"""

from .manager import DatabaseManager, init_db, import_config
from .ingestion import IngestionEngine, run_ingestion
from .processing import CycleProcessor, run_cycle_processing
from .reader import (
    get_legacy_dataframe,
    get_legacy_dataframe_by_date,
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

from .phases import (
    PhaseEngine,
    get_phase_splits,
)

__all__ = [
    # Manager
    'DatabaseManager',
    'init_db',
    'import_config',
    # Ingestion
    'IngestionEngine',
    'run_ingestion',
    # Processing
    'CycleProcessor',
    'run_cycle_processing',
    # Reader
    'get_legacy_dataframe',
    'get_legacy_dataframe_by_date',
    'get_coordination_data',
    'get_config_df',
    'get_config_dict',
    'get_det_config',
    'get_date_range',
    'get_available_dates',
    'check_data_quality',
    'convert_to_datetime',
    # Counts
    'CountEngine',
    'get_vehicle_counts',
    'get_ped_counts',
    'get_combined_counts',
    # Phases
    'PhaseEngine',
    'get_phase_splits',
]