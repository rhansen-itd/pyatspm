"""
ATSPM Plotting Package (Functional Core)

Contains pure functions that accept DataFrames and metadata, and return
Plotly Figure objects. Strictly no side effects (no DB queries, no file I/O).
"""

from .termination import plot_termination
from .coordination import plot_coordination
from .detectors import plot_detector_comparison

__all__ = [
    'plot_termination',
    'plot_coordination',
    'plot_detector_comparison',
]