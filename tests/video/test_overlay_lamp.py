"""Unit tests for signal lamp overlay drawing.

Target: src/atspm/video/overlay.py — draw_shape_overlay, draw_lamp_overlay.
"""

import numpy as np
import pytest

from atspm.video.overlay import draw_lamp_overlay, draw_shape_overlay


def test_lamp_dot_colors_and_untouched_lamp_position():
    lamp_pt = (50, 50)
    dot_center_y, dot_center_x = 40, 50
    lamp_shape = {
        "type": "lamp",
        "points": [lamp_pt],
        "color": (0, 255, 0),
        "input": None,
        "phase": 2,
        "name": "Phase 2 green",
        "indication": "green",
    }

    # Test G
    frame_g = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_shape_overlay(frame_g, lamp_shape, "G")
    assert tuple(frame_g[dot_center_y, dot_center_x]) == (0, 255, 0)
    assert tuple(frame_g[lamp_pt[1], lamp_pt[0]]) == (0, 0, 0)

    # Test R
    frame_r = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_shape_overlay(frame_r, lamp_shape, "R")
    assert tuple(frame_r[dot_center_y, dot_center_x]) == (0, 0, 255)
    assert tuple(frame_r[lamp_pt[1], lamp_pt[0]]) == (0, 0, 0)

    # Test na
    frame_na = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_shape_overlay(frame_na, lamp_shape, "na")
    assert tuple(frame_na[dot_center_y, dot_center_x]) == (128, 128, 128)
    assert tuple(frame_na[lamp_pt[1], lamp_pt[0]]) == (0, 0, 0)


def test_polygon_lamp_centroid_dot():
    pts = [(40, 40), (60, 40), (60, 60), (40, 60)]
    lamp_shape = {
        "type": "lamp",
        "points": pts,
        "color": (0, 255, 0),
        "input": None,
        "phase": 2,
        "name": "Poly lamp",
        "indication": "green",
    }
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_shape_overlay(frame, lamp_shape, "G")
    # Centroid is (50, 50), dot center is (50, 40)
    assert tuple(frame[40, 50]) == (0, 255, 0)
    # Centroid at (50, 50) is untouched
    assert tuple(frame[50, 50]) == (0, 0, 0)
