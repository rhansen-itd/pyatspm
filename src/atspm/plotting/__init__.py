"""
ATSPM Plotting Package (Functional Core)

Pure plotting functions only – no SQL, no file I/O, no side effects.
Every public function accepts DataFrames / dicts and returns a
``plotly.graph_objects.Figure``.

Modules:
    termination:  Phase termination scatter plot (Gap Out / Max Out /
                  Force Off / Preempt / Pedestrian Service).
    coordination: Coordination / Split stacked-bar diagram with optional
                  arrival-offset slider (1 or 2 ring subplots).
"""

from .termination import plot_termination
from .coordination import plot_coordination

__all__ = [
    'plot_termination',
    'plot_coordination',
]