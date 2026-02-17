"""
ATSPM Data Package (Imperative Shell)

This package handles all I/O operations, database management,
and resource handling for the ATSPM system.

Modules:
- manager: Database initialization and configuration import
- ingestion: Event data ingestion from .datZ files
- processing: Cycle detection processing
- reader: Legacy adapter for reading data in flat format
"""

from .manager import DatabaseManager, init_db, import_config
from .ingestion import IngestionEngine, run_ingestion
from .processing import CycleProcessor, run_cycle_processing
from .reader import (
    get_legacy_dataframe,
    get_legacy_dataframe_by_date,
    get_config_df,
    get_config_dict,
    get_date_range,
    get_available_dates,
    check_data_quality,
    preview_data,
    convert_to_datetime,
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
    'get_config_df',
    'get_config_dict',
    'get_date_range',
    'get_available_dates',
    'check_data_quality',
    'preview_data',
    'convert_to_datetime',
]