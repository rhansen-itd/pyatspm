"""
Video Shape Calibration Tool (Imperative Shell)

Interactive, one-time-per-camera calibration: draw loop/stopbar shapes over
a video's first frame, edit/drag/undo them, and save the result as a
:class:`atspm.data.video.ShapeConfig`.  Mouse-driven (OpenCV window) with a
few value-entry dialogs (Tkinter).  Not a batch operation -- there is
deliberately no ``--all`` CLI path for this tool (see ``docs/ROADMAP.md``).

Ported from the legacy ``spmfunctions.video_processing.VideoProcessor
.draw_shapes_interface`` mouse/keyboard state machine, adapted to operate
on a standalone ``ShapeConfig`` instead of a video-processing god object.

Package Location: src/atspm/video/calibrate.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox

from ..data.video import ShapeConfig

_DOT_RADIUS = 5
_SNAP_RADIUS = 12
_BODY_HIT_TOLERANCE = 8


def _draw_shape_preview(img: np.ndarray, shape: Dict[str, Any]) -> None:
    """Render a shape in its configured (non-status) color, for the calibration preview."""
    if shape["type"] == "loop":
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=shape["color"], thickness=2)
    elif shape["type"] == "stopbar":
        pt1, pt2 = shape["points"]
        cv2.line(img, pt1, pt2, color=(0, 0, 255), thickness=2)


def _dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _point_segment_distance(pt: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> float:
    px, py = pt
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return _dist(pt, a)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return _dist(pt, (ax + t * dx, ay + t * dy))


def _shape_body_hit(pt: Tuple[int, int], shape: Dict[str, Any]) -> bool:
    """Whether *pt* lands on the body (not just a corner) of *shape*."""
    if shape["type"] == "stopbar":
        pt1, pt2 = shape["points"]
        return _point_segment_distance(pt, pt1, pt2) <= _BODY_HIT_TOLERANCE
    if shape["type"] == "loop":
        contour = np.array(shape["points"], dtype=np.float32)
        return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0
    return False


def _find_snap_target(
    pt: Tuple[int, int], shapes: List[Dict[str, Any]], exclude_idx: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Nearest point belonging to another shape within ``_SNAP_RADIUS``, if any."""
    best = None
    best_d = _SNAP_RADIUS
    for i, s in enumerate(shapes):
        if i == exclude_idx:
            continue
        for cand in s["points"]:
            d = _dist(pt, cand)
            if d < best_d:
                best_d, best = d, cand
    return best


def _whole_shape_snap_correction(
    points: List[Tuple[int, int]], shapes: List[Dict[str, Any]], exclude_idx: int,
) -> Optional[Tuple[int, int]]:
    """Translation that snaps the closest of *points* to a neighboring shape's point, if any."""
    best_d = _SNAP_RADIUS
    correction = None
    for p in points:
        target = _find_snap_target(p, shapes, exclude_idx)
        if target is not None:
            d = _dist(p, target)
            if d < best_d:
                best_d = d
                correction = (target[0] - p[0], target[1] - p[1])
    return correction


def calibrate_shapes(
    video_path: Union[str, Path],
    shape_config: Optional[ShapeConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> ShapeConfig:
    """Open an interactive window to draw/edit loop and stopbar shapes.

    Args:
        video_path: Path to the camera video to calibrate against. Only the
            first frame is used as the drawing background.
        shape_config: Existing config to continue editing. When ``None``, a
            fresh config is created using the video's resolution.
        save_path: Destination to write the config to. When set, pressing
            ``'w'`` saves immediately and ``'q'`` prompts to save before
            exiting. When ``None``, the session is purely in-memory -- the
            caller is responsible for persisting the returned config.

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
    save_path = Path(save_path) if save_path is not None else None

    image = first_frame.copy()

    mode = "loop"
    color = (0, 255, 0)
    input_val = 1
    phase: Union[int, str] = 1
    current_shape: List[tuple] = []
    snap_enabled = False

    edit_mode = False
    current_edit_index = -1
    edit_shape_type: Optional[str] = None
    dragging_point: Optional[tuple] = None
    whole_drag: Optional[Dict[str, Any]] = None

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
        "    - Press 'c' to change color (for loops)\n"
        "    - Press 'i' to set input value (for loops)\n"
        "    - Press 'p' to set phase value (for stop bars)\n"
        "    - Press 'g' to toggle point-snapping on/off\n"
        "    - Press 'u' to undo last action\n"
        "    - Press 'e' to enter/exit edit mode\n"
        "    - In edit mode: 'n'/'b' to cycle next/previous shape; "
        "drag a corner point to reshape; drag the shape body to move it "
        "(hold Ctrl while dragging to copy instead); 'i'/'p' to edit "
        "values; 'r' to rename\n"
        "    - Press 'w' to save now\n"
        "    - Press 'q' to finish (prompts to save)"
    )
    label = tk.Label(instruction_window, text=instruction_text, justify="left", padx=10, pady=10)
    label.pack()
    instruction_window.update_idletasks()
    screen_width = instruction_window.winfo_screenwidth()
    window_width = instruction_window.winfo_width()
    instruction_window.geometry(f"+{screen_width - window_width - 40}+40")

    def do_save() -> None:
        if save_path is None:
            print("No save path configured; nothing to save to.")
            return
        config.shapes = shapes
        config.save(save_path)
        print(f"Saved {len(shapes)} shapes to {save_path}")

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_shape
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x, y)
            if snap_enabled:
                target = _find_snap_target(pt, shapes)
                if target is not None:
                    pt = target
            current_shape.append(pt)
            if len(current_shape) == 4 and mode == "loop":
                shapes.append({
                    "type": "loop", "points": list(current_shape),
                    "color": color, "input": input_val, "name": None,
                })
                current_shape = []
            elif len(current_shape) == 2 and mode == "stopbar":
                shapes.append({
                    "type": "stopbar", "points": list(current_shape), "phase": phase,
                    "name": None,
                })
                current_shape = []

    def mouse_callback_edit(event, x, y, flags, param):
        nonlocal dragging_point, whole_drag, current_edit_index
        if not edit_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_edit_index == -1:
                return
            shape = shapes[current_edit_index]
            for i, pt in enumerate(shape["points"]):
                if _dist((x, y), pt) <= _DOT_RADIUS * 2:
                    dragging_point = (current_edit_index, i)
                    print(f"Dragging point {i} of shape {current_edit_index}")
                    return
            if _shape_body_hit((x, y), shape):
                is_copy = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
                if is_copy:
                    new_shape = dict(shape)
                    new_shape["points"] = list(shape["points"])
                    shapes.append(new_shape)
                    target_idx = len(shapes) - 1
                    current_edit_index = target_idx
                else:
                    target_idx = current_edit_index
                whole_drag = {
                    "shape_idx": target_idx,
                    "anchor": (x, y),
                    "orig_points": list(shapes[target_idx]["points"]),
                    "is_copy": is_copy,
                }
                print(f"{'Copying' if is_copy else 'Moving'} shape {target_idx}")
        elif event == cv2.EVENT_MOUSEMOVE:
            if whole_drag is not None:
                dx = x - whole_drag["anchor"][0]
                dy = y - whole_drag["anchor"][1]
                idx = whole_drag["shape_idx"]
                shapes[idx]["points"] = [
                    (px + dx, py + dy) for px, py in whole_drag["orig_points"]
                ]
        elif event == cv2.EVENT_LBUTTONUP:
            if dragging_point:
                shape_idx, pt_idx = dragging_point
                pt = (x, y)
                if snap_enabled:
                    target = _find_snap_target(pt, shapes, exclude_idx=shape_idx)
                    if target is not None:
                        pt = target
                shapes[shape_idx]["points"][pt_idx] = pt
                print(f"Moved point {pt_idx} to {pt}")
                dragging_point = None
            elif whole_drag is not None:
                idx = whole_drag["shape_idx"]
                if snap_enabled:
                    correction = _whole_shape_snap_correction(shapes[idx]["points"], shapes, idx)
                    if correction is not None:
                        shapes[idx]["points"] = [
                            (px + correction[0], py + correction[1])
                            for px, py in shapes[idx]["points"]
                        ]
                print(f"Finished {'copy' if whole_drag['is_copy'] else 'move'} of shape {idx}")
                whole_drag = None

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
                elif shape["type"] == "stopbar":
                    pt1, pt2 = shape["points"]
                    cv2.line(img_copy, pt1, pt2, color=(255, 255, 255), thickness=4)
                    cv2.circle(img_copy, pt1, _DOT_RADIUS + 2, (255, 255, 255), -1)
                    cv2.circle(img_copy, pt2, _DOT_RADIUS + 2, (255, 255, 255), -1)
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
        if snap_enabled:
            mode_text += " | Snap: ON"
        if mode == "loop" and not edit_mode:
            color_name = next((name for name, rgb in colors.items() if rgb == color), str(color))
            mode_text += f" | Color: {color_name} | Input: {input_val}"
        elif not edit_mode:
            mode_text += f" | Phase: {phase}"
        if edit_mode and current_edit_index != -1:
            name = shapes[current_edit_index].get("name")
            mode_text += f" | Name: {name or '(unnamed)'}"
        cv2.putText(img_copy, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Draw Shapes", img_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            should_exit = True
            if save_path is not None:
                choice = messagebox.askyesnocancel(
                    "Exit", "Save changes before exiting?", parent=root,
                )
                if choice is None:
                    should_exit = False
                elif choice:
                    do_save()
            if should_exit:
                break

        elif key == ord("w"):
            do_save()

        elif key == ord("g"):
            snap_enabled = not snap_enabled
            print(f"Snapping {'enabled' if snap_enabled else 'disabled'}.")

        elif key in (ord("l"), ord("s")):
            mode = {"l": "loop", "s": "stopbar"}[chr(key)]
            edit_shape_type = None
            edit_mode = False
            current_edit_index = -1
            dragging_point = None
            whole_drag = None

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
                whole_drag = None
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
            elif key == ord("r") and current_edit_index != -1:
                shape = shapes[current_edit_index]
                name = simpledialog.askstring(
                    "Name", "Enter element name:",
                    initialvalue=shape.get("name") or "", parent=root,
                )
                if name is not None:
                    shape["name"] = name.strip() or None
            elif key in (ord("n"), ord("b")) and shapes:
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
