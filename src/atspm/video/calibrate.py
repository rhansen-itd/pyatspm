"""
Video Shape Calibration Tool (Imperative Shell)

Interactive, one-time-per-camera calibration: draw loop/stopbar/approach
shapes over a video's first frame, edit/drag/undo them, and save the
result as a :class:`atspm.data.video.ShapeConfig`.  Mouse-driven (OpenCV
window) with a few value-entry dialogs (Tkinter).  Not a batch operation --
there is deliberately no ``--all`` CLI path for this tool (see
``docs/ROADMAP.md``).

Ported from the legacy ``spmfunctions.video_processing.VideoProcessor
.draw_shapes_interface`` mouse/keyboard state machine, adapted to operate
on a standalone ``ShapeConfig`` instead of a video-processing god object.

Package Location: src/atspm/video/calibrate.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox

from ..data.video import ShapeConfig

_DIR_COLORS = {"N": (255, 0, 0), "E": (0, 255, 0), "S": (0, 0, 255), "W": (255, 255, 0)}
_DOT_RADIUS = 5


def _draw_shape_preview(img: np.ndarray, shape: Dict[str, Any]) -> None:
    """Render a shape in its configured (non-status) color, for the calibration preview."""
    if shape["type"] == "loop":
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=shape["color"], thickness=2)
    elif shape["type"] == "stopbar":
        pt1, pt2 = shape["points"]
        cv2.line(img, pt1, pt2, color=(0, 0, 255), thickness=2)
    elif shape["type"] == "approach":
        pt1, pt2 = shape["points"]
        color = _DIR_COLORS.get(shape["direction"], (128, 128, 128))
        cv2.line(img, pt1, pt2, color=color, thickness=3)
        mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
        cv2.putText(img, shape["direction"], (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def calibrate_shapes(
    video_path: Union[str, Path],
    shape_config: Optional[ShapeConfig] = None,
) -> ShapeConfig:
    """Open an interactive window to draw/edit loop, stopbar, and approach shapes.

    Args:
        video_path: Path to the camera video to calibrate against. Only the
            first frame is used as the drawing background.
        shape_config: Existing config to continue editing. When ``None``, a
            fresh config is created using the video's resolution.

    Returns:
        The (possibly newly-created) ``ShapeConfig``, with ``shapes``
        populated from the editing session and ``video_width``/
        ``video_height`` set to the video's actual resolution.

    Raises:
        ValueError: If the video cannot be opened or its first frame
            cannot be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Cannot read first frame")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    config = shape_config if shape_config is not None else ShapeConfig()
    config.video_width, config.video_height = width, height
    shapes: List[Dict[str, Any]] = config.shapes

    image = first_frame.copy()

    mode = "loop"
    color = (0, 255, 0)
    input_val = 1
    phase: Union[int, str] = 1
    current_shape: List[tuple] = []

    edit_mode = False
    current_edit_index = -1
    edit_shape_type: Optional[str] = None
    dragging_point: Optional[tuple] = None

    colors = {
        "Green": (0, 255, 0), "Blue": (255, 0, 0), "Red": (0, 0, 255),
        "Yellow": (255, 255, 0), "Magenta": (255, 0, 255), "Cyan": (0, 255, 255),
        "Black": (0, 0, 0),
    }
    color_index = 0

    root = tk.Tk()
    root.withdraw()
    instruction_window = tk.Toplevel()
    instruction_window.title("Draw Shapes Instructions")
    instruction_window.attributes("-topmost", True)
    instruction_window.resizable(False, False)
    instruction_text = (
        "Instructions:\n"
        "    - Press 'l' to switch to loop mode (4 points)\n"
        "    - Press 's' to switch to stop bar mode (2 points)\n"
        "    - Press 'a' to switch to approach mode (2 points, then enter N/E/S/W)\n"
        "    - Press 'c' to change color (for loops)\n"
        "    - Press 'i' to set input value (for loops)\n"
        "    - Press 'p' to set phase value (for stop bars)\n"
        "    - Press 'u' to undo last action\n"
        "    - Press 'e' to enter/exit edit mode\n"
        "    - In edit mode: 'n'/'p' to cycle next and previous shape, "
        "click near point to drag, 'i'/'p'/'d' to edit values\n"
        "    - Press 'q' when finished"
    )
    label = tk.Label(instruction_window, text=instruction_text, justify="left", padx=10, pady=10)
    label.pack()
    instruction_window.update_idletasks()
    screen_width = instruction_window.winfo_screenwidth()
    window_width = instruction_window.winfo_width()
    instruction_window.geometry(f"+{screen_width - window_width - 40}+40")

    def dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_shape
        if event == cv2.EVENT_LBUTTONDOWN:
            current_shape.append((x, y))
            if len(current_shape) == 4 and mode == "loop":
                shapes.append({
                    "type": "loop", "points": list(current_shape),
                    "color": color, "input": input_val,
                })
                current_shape = []
            elif len(current_shape) == 2 and mode == "stopbar":
                shapes.append({
                    "type": "stopbar", "points": list(current_shape), "phase": phase,
                })
                current_shape = []
            elif len(current_shape) == 2 and mode == "approach":
                direction = simpledialog.askstring(
                    "Direction", "Enter direction (N, E, S, W):", parent=root
                )
                if direction and direction.upper() in ["N", "E", "S", "W"]:
                    shapes.append({
                        "type": "approach", "points": list(current_shape),
                        "direction": direction.upper(),
                    })
                else:
                    print("Invalid direction. Use N, E, S, or W.")
                current_shape = []

    def mouse_callback_edit(event, x, y, flags, param):
        nonlocal dragging_point
        if not edit_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_edit_index == -1:
                return
            shape = shapes[current_edit_index]
            for i, pt in enumerate(shape["points"]):
                if dist((x, y), pt) <= _DOT_RADIUS * 2:
                    dragging_point = (current_edit_index, i)
                    print(f"Dragging point {i} of shape {current_edit_index}")
                    break
        elif event == cv2.EVENT_LBUTTONUP:
            if dragging_point:
                shape_idx, pt_idx = dragging_point
                shapes[shape_idx]["points"][pt_idx] = (x, y)
                print(f"Moved point {pt_idx} to ({x}, {y})")
                dragging_point = None

    cv2.namedWindow("Draw Shapes")
    cv2.setMouseCallback(
        "Draw Shapes",
        lambda e, x, y, f, p: mouse_callback(e, x, y, f, p) if not edit_mode
        else mouse_callback_edit(e, x, y, f, p),
    )

    while True:
        img_copy = image.copy()

        for idx, shape in enumerate(shapes):
            is_selected = edit_mode and idx == current_edit_index
            if is_selected:
                if shape["type"] == "loop":
                    pts = np.array(shape["points"], dtype=np.int32)
                    cv2.polylines(img_copy, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
                    for pt in shape["points"]:
                        cv2.circle(img_copy, pt, _DOT_RADIUS + 2, (255, 255, 255), -1)
                elif shape["type"] in ("stopbar", "approach"):
                    pt1, pt2 = shape["points"]
                    cv2.line(img_copy, pt1, pt2, color=(255, 255, 255), thickness=4)
                    cv2.circle(img_copy, pt1, _DOT_RADIUS + 2, (255, 255, 255), -1)
                    cv2.circle(img_copy, pt2, _DOT_RADIUS + 2, (255, 255, 255), -1)
                    if shape["type"] == "approach":
                        mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                        cv2.putText(img_copy, shape["direction"], (mid_x, mid_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                _draw_shape_preview(img_copy, shape)

        if not edit_mode and len(current_shape) > 0:
            for pt in current_shape:
                cv2.circle(img_copy, pt, 5, (0, 0, 0), -1)
            if len(current_shape) == 2:
                cv2.line(img_copy, current_shape[0], current_shape[1], (0, 0, 0), 2)
            elif len(current_shape) >= 3:
                pts = np.array(current_shape, dtype=np.int32)
                cv2.polylines(img_copy, [pts], isClosed=(len(current_shape) == 4 and mode == "loop"),
                              color=(0, 0, 0), thickness=1)

        mode_text = f"Mode: {mode}"
        if edit_mode:
            mode_text += " | EDIT MODE"
        if mode == "loop" and not edit_mode:
            color_name = next((name for name, rgb in colors.items() if rgb == color), str(color))
            mode_text += f" | Color: {color_name} | Input: {input_val}"
        elif not edit_mode:
            mode_text += f" | Phase: {phase}"
        cv2.putText(img_copy, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Draw Shapes", img_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key in (ord("l"), ord("s"), ord("a")):
            mode = {"l": "loop", "s": "stopbar", "a": "approach"}[chr(key)]
            edit_shape_type = None
            edit_mode = False
            current_edit_index = -1
            dragging_point = None

        elif key == ord("c") and mode == "loop" and not edit_mode:
            color_names = list(colors.keys())
            color_index = (color_index + 1) % len(color_names)
            color = colors[color_names[color_index]]

        elif key == ord("i") and not edit_mode:
            inp = simpledialog.askinteger("Input Value", "Enter input value (1-64):", minvalue=1, maxvalue=64)
            if inp is not None:
                input_val = inp

        elif key == ord("p") and not edit_mode:
            phase_input = simpledialog.askstring("Phase Value", "Enter phase value (1-16 or A-P):")
            if phase_input:
                phase_input = phase_input.strip().upper()
                if phase_input.isdigit():
                    p_val = int(phase_input)
                    if 1 <= p_val <= 16:
                        phase = p_val
                    else:
                        messagebox.showerror("Phase", "Phase must be between 1-16")
                elif len(phase_input) == 1 and "A" <= phase_input <= "P":
                    phase = f"OL{phase_input}"
                else:
                    messagebox.showerror("Phase", "Phase must be between 1-16 or A-P")

        elif key == ord("e"):
            if not edit_mode:
                edit_mode = True
                edit_shape_type = mode
                current_edit_index = next(
                    (i for i, s in enumerate(shapes) if s["type"] == edit_shape_type), -1
                )
                print(f"Entered edit mode for type: {edit_shape_type}. Selected shape {current_edit_index}")
            else:
                edit_mode = False
                current_edit_index = -1
                edit_shape_type = None
                dragging_point = None
                print("Exited edit mode.")

        elif edit_mode:
            if key == ord("i") and current_edit_index != -1:
                shape = shapes[current_edit_index]
                if shape["type"] == "loop":
                    inp = simpledialog.askinteger(
                        "Input Value", "Edit input value (1-64):",
                        initialvalue=shape.get("input", 1), minvalue=1, maxvalue=64,
                    )
                    if inp is not None:
                        shape["input"] = inp
            elif key == ord("p") and current_edit_index != -1:
                shape = shapes[current_edit_index]
                if shape["type"] == "stopbar":
                    phase_input = simpledialog.askstring(
                        "Phase Value", "Edit phase value (1-16 or A-P):",
                        initialvalue=str(shape.get("phase", "")),
                    )
                    if phase_input:
                        phase_input = phase_input.strip().upper()
                        if phase_input.isdigit():
                            p_val = int(phase_input)
                            if 1 <= p_val <= 16:
                                shape["phase"] = p_val
                            else:
                                messagebox.showerror("Phase", "Phase must be between 1-16")
                        elif len(phase_input) == 1 and "A" <= phase_input <= "P":
                            shape["phase"] = f"OL{phase_input}"
                        else:
                            messagebox.showerror("Phase", "Phase must be between 1-16 or A-P")
            elif key == ord("d") and current_edit_index != -1:
                shape = shapes[current_edit_index]
                if shape["type"] == "approach":
                    direction = simpledialog.askstring(
                        "Direction", "Edit direction (N/E/S/W):",
                        initialvalue=shape.get("direction", "N"),
                    )
                    if direction and direction.upper() in ["N", "E", "S", "W"]:
                        shape["direction"] = direction.upper()
                    else:
                        messagebox.showerror("Direction", "Must be N, E, S, or W")
            elif key in (ord("n"), ord("p")) and shapes:
                candidates = [i for i, s in enumerate(shapes) if s["type"] == edit_shape_type]
                if candidates:
                    try:
                        cur = candidates.index(current_edit_index)
                        step = 1 if key == ord("n") else -1
                        current_edit_index = candidates[(cur + step) % len(candidates)]
                    except ValueError:
                        current_edit_index = candidates[0]
                    print(f"Cycled to shape #{current_edit_index} ({shapes[current_edit_index]['type']})")

        elif key == ord("u"):
            if current_shape:
                current_shape.pop()
                print("Undone: Removed last point.")
            elif shapes:
                removed = shapes.pop()
                print(f"Undone: Removed last {removed['type'].title()} shape.")
            else:
                print("Nothing to undo.")

    cv2.destroyAllWindows()
    root.destroy()

    config.shapes = shapes
    return config
