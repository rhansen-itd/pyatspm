"""
Video Overlay — Frame Recoloring (Functional Core, with a documented exception)

Pure recoloring logic: given a single OpenCV frame, a shape definition, and
a precomputed status value (from ``atspm.analysis.video``), draw the
shape onto the frame in the right color.  No file I/O, no database access,
no video reads/writes.

Functional Core mutation exception
-----------------------------------
``cv2.polylines``/``cv2.fillPoly``/``cv2.line`` mutate the frame array in
place rather than returning a new one.  Copying every frame for strict
purity would be wasteful for video (this runs once per shape, per frame).
This is accepted as a documented exception to the Functional Core rule for
this module specifically, the same way ``architecture.md`` already
documents the ``.iterrows()`` exceptions in ``analysis/detectors.py`` and
``plotting/*`` by name, and ``plotting/detectors.py``'s
``_add_anomaly_overlays(fig, ...)`` already mutates a ``go.Figure`` in
place rather than returning a new one.

Package Location: src/atspm/video/overlay.py
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

_STOPBAR_COLOR_MAP = {
    "G":  (0, 255, 0),
    "Y":  (0, 255, 255),
    "R":  (0, 0, 255),
    "na": (128, 128, 128),
}

_LOOP_ON_OUTLINE = (255, 255, 255)
_LOOP_ON_ALPHA = 0.2


def draw_loop_overlay(frame: np.ndarray, shape: Dict[str, Any], is_on: bool) -> None:
    """Draw a detector loop shape, filled when on, outlined when off.

    Mutates ``frame`` in place (see module docstring).

    Args:
        frame: BGR image array to draw onto.
        shape: A ``"loop"``-type shape dict with ``points`` and ``color``.
        is_on: Detector On/Off state at this frame's timestamp.
    """
    pts = np.array(shape["points"], dtype=np.int32)
    base_color = shape.get("color", (0, 255, 0))

    if is_on:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], base_color)
        cv2.addWeighted(overlay, _LOOP_ON_ALPHA, frame, 1 - _LOOP_ON_ALPHA, 0, frame)
        outline_color = _LOOP_ON_OUTLINE
    else:
        outline_color = base_color

    cv2.polylines(frame, [pts], isClosed=True, color=outline_color, thickness=1)


def draw_stopbar_overlay(frame: np.ndarray, shape: Dict[str, Any], status: str) -> None:
    """Draw a stopbar line colored by its phase/overlap G/Y/R/na status.

    Mutates ``frame`` in place (see module docstring).

    Args:
        frame: BGR image array to draw onto.
        shape: A ``"stopbar"``-type shape dict with two ``points``.
        status: One of ``'G'``, ``'Y'``, ``'R'``, ``'na'`` (from
            ``atspm.analysis.video.phase_status_at_timestamps`` or
            ``overlap_status_at_timestamps``).
    """
    color = _STOPBAR_COLOR_MAP.get(status, _STOPBAR_COLOR_MAP["na"])
    pt1, pt2 = shape["points"]
    cv2.line(frame, pt1, pt2, color, thickness=3)


def draw_shape_overlay(frame: np.ndarray, shape: Dict[str, Any], status: Any) -> None:
    """Dispatch a single shape to the correct drawing function for its type.

    ``"approach"`` shapes belong to the out-of-scope turning-movement
    counter (see ``docs/ROADMAP.md``) and are intentionally not rendered
    here.

    Args:
        frame: BGR image array to draw onto. Mutated in place.
        shape: A shape dict as produced by ``atspm.data.video.ShapeConfig``.
        status: ``bool`` for ``"loop"`` shapes (detector On/Off), or a
            G/Y/R/na string for ``"stopbar"`` shapes.
    """
    if shape["type"] == "loop":
        draw_loop_overlay(frame, shape, bool(status))
    elif shape["type"] == "stopbar":
        draw_stopbar_overlay(frame, shape, status)
