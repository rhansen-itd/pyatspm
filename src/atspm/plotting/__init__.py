"""
ATSPM Plotting Package (Functional Core)

Contains pure functions that accept DataFrames and metadata, and return
Plotly Figure objects. Strictly no side effects (no DB queries, no file I/O).
"""

from .termination import plot_termination
from .coordination import plot_coordination
from .detectors import plot_detector_comparison
from .flow import plot_flow_profiles
from .optimizer import (
    plot_allocation,
    plot_marginal_rates,
    plot_throughput_curve,
)
from .clock_marks import plot_clock_drift
from .split_failures import plot_split_failures
from .approach_delay import plot_approach_delay
from .yellow_red_actuations import plot_yellow_red
from .green_time_utilization import plot_green_time
from .call_service import plot_ped_delay, plot_wait_time
from .split_monitor import plot_split_monitor
from .detector_health import plot_detector_health
from .timing_actuation import plot_timing_actuation

__all__ = [
    'plot_termination',
    'plot_coordination',
    'plot_detector_comparison',
    'plot_flow_profiles',
    'plot_allocation',
    'plot_marginal_rates',
    'plot_throughput_curve',
    'plot_clock_drift',
    'plot_split_failures',
    'plot_approach_delay',
    'plot_yellow_red',
    'plot_green_time',
    'plot_ped_delay',
    'plot_wait_time',
    'plot_split_monitor',
    'plot_detector_health',
    'plot_timing_actuation',
]