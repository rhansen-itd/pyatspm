"""
ATSPM Video Overlay Package (Imperative Shell, with one documented exception)

Renders a recorded intersection video with loop/stopbar shapes recolored
by live phase, overlap, and detector status.  A peer of ``analysis/``,
``data/``, ``plotting/``, and ``reports/`` rather than a submodule of
``plotting/`` -- OpenCV frame rendering is a different output shape (video
frames) and a different library (``plotting/`` is Plotly-exclusive per
``CLAUDE.md``).

Modules:
- overlay:    Pure(-ish) frame recoloring -- see overlay.py's documented
              Functional Core mutation exception.
- calibrate:  Interactive Tkinter+OpenCV shape calibration tool.
- processor:  Orchestrates DB reads, status lookups, and video I/O.

The turning-movement counter from the legacy ``spmfunctions`` tool
(``EnhancedIOUTracker``, YOLO/background-subtraction vehicle tracking) is
out of scope -- see ``docs/ROADMAP.md``'s "Future / deferred" section.
"""

from .overlay import draw_loop_overlay, draw_shape_overlay, draw_stopbar_overlay
from .calibrate import calibrate_shapes
from .processor import VideoOverlayResult, render_overlay

__all__ = [
    "draw_loop_overlay",
    "draw_shape_overlay",
    "draw_stopbar_overlay",
    "calibrate_shapes",
    "VideoOverlayResult",
    "render_overlay",
]
