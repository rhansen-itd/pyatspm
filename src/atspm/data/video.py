"""
ATSPM Video Shape Configuration (Imperative Shell)

Owns the per-camera shape-config CSV round-trip (loop/stopbar shapes drawn
over a fixed video resolution) and the small amount of identifier resolution
needed to connect a shape to the right event-code lookup in
``atspm.analysis.video``.

Shape CSV location convention
------------------------------
``intersections/<folder>/video/<camera>_shapes.csv`` -- a sibling of the
existing ``raw_data/``, ``outputs/``, and ``int_cfg.csv`` per-intersection
layout.  Unlike ``int_cfg.csv`` (signal-timing-period config), this file is
tied to a specific camera and its recorded ``video_width``/``video_height``
and is not date-versioned.

File layout
-----------
A 2-section CSV: a one-row metadata header (``video_width``/``video_height``)
followed by the per-shape table, so resolution is recorded once per file
rather than repeated on every shape row::

    video_width,video_height
    1920,1080
    type,points,color,input,phase,name
    loop,100,100;200,100;200,200;100,200,"0,255,0",3,,South Loop 3
    stopbar,50,300;250,300,"0,0,255",,2,Southbound Stop Bar

Overlap numbering
------------------
Overlap letters (``"OLA"``-``"OLP"``) map to numbers ``1``-``16`` (``A=1,
B=2, ...``), matching the Indiana/Purdue Hi-Res Logger Enumerations spec's
"Overlap # (as number A=1 B=2, etc)" parameter convention for event codes
61-66.  16, not 26 -- the legacy ``spmfunctions`` tool's ``range(26)`` was a
bug, not a spec requirement.

Package Location: src/atspm/data/video.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Overlap letter <-> number convention (fixed, global -- see module docstring)
# ---------------------------------------------------------------------------

OVERLAP_LETTER_MAP: Dict[str, int] = {f"OL{chr(ord('A') + i)}": i + 1 for i in range(16)}

# Valid signal phase numbers, matching the overlap range above: the
# Indiana/Purdue Hi-Res Logger Enumerations spec numbers both 1-16.
MIN_PHASE_NUMBER: int = 1
MAX_PHASE_NUMBER: int = 16

_META_FIELDS = ["video_width", "video_height"]
_CSV_FIELDS = ["type", "points", "color", "input", "phase", "name"]


def resolve_stopbar_target(phase_field: Union[int, str]) -> Tuple[str, int]:
    """Resolve a stopbar shape's ``phase`` field to a lookup target.

    Args:
        phase_field: Either an integer/numeric-string phase number
            (``1``-``16``), or an overlap letter code (``"OLA"``-``"OLP"``).

    Returns:
        A ``(kind, number)`` tuple where ``kind`` is ``"phase"`` or
        ``"overlap"``.

    Raises:
        ValueError: If ``phase_field`` is neither a valid phase number nor
            a recognised overlap letter, or if it is an integer outside the
            ``MIN_PHASE_NUMBER``-``MAX_PHASE_NUMBER`` range.
    """
    s = str(phase_field).strip().upper()
    if s in OVERLAP_LETTER_MAP:
        return ("overlap", OVERLAP_LETTER_MAP[s])
    try:
        number = int(s)
    except ValueError as exc:
        raise ValueError(
            f"Unrecognised stopbar phase field {phase_field!r}: not an "
            f"integer phase number or an OLA-OLP overlap code."
        ) from exc
    if not MIN_PHASE_NUMBER <= number <= MAX_PHASE_NUMBER:
        raise ValueError(
            f"Stopbar phase number {number} out of range: phases are "
            f"{MIN_PHASE_NUMBER}-{MAX_PHASE_NUMBER}."
        )
    return ("phase", number)


class ShapeConfig:
    """Loop/stopbar shape definitions for a single camera.

    Mirrors the legacy ``VideoProcessor`` shape list (a list of dicts with
    ``type`` in ``{"loop", "stopbar"}``), but as a standalone
    Imperative Shell class rather than bundled into a video-processing god
    object.
    """

    def __init__(
        self,
        shapes: Optional[List[Dict[str, Any]]] = None,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
    ) -> None:
        self.shapes: List[Dict[str, Any]] = shapes if shapes is not None else []
        self.video_width = video_width
        self.video_height = video_height

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ShapeConfig":
        """Load a shape CSV (round-trips with :meth:`save`).

        Args:
            path: Path to the ``<camera>_shapes.csv`` file.

        Returns:
            A populated ``ShapeConfig``.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Shape config not found: {path}")

        shapes: List[Dict[str, Any]] = []

        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # _META_FIELDS header row
            meta_row = next(reader, None) or []
            video_width = int(meta_row[0]) if len(meta_row) > 0 and meta_row[0] else None
            video_height = int(meta_row[1]) if len(meta_row) > 1 and meta_row[1] else None

            shape_header = next(reader, None) or _CSV_FIELDS
            for row in csv.DictReader(f, fieldnames=shape_header):
                points = []
                for pt_str in row["points"].split(";"):
                    x, y = map(int, pt_str.split(","))
                    points.append((x, y))

                color = tuple(map(int, row["color"].split(","))) if row["color"] else (0, 255, 0)

                shapes.append({
                    "type": row["type"],
                    "points": points,
                    "color": color,
                    "input": int(row["input"]) if row["input"] else None,
                    "phase": row["phase"] or None,
                    "name": row.get("name") or None,
                })

        return cls(shapes=shapes, video_width=video_width, video_height=video_height)

    def save(self, path: Union[str, Path]) -> None:
        """Write this shape config out in the same format :meth:`load` reads.

        Args:
            path: Destination path. Parent directory must already exist.
        """
        path = Path(path)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_META_FIELDS)
            writer.writerow([self.video_width, self.video_height])
            writer.writerow(_CSV_FIELDS)
            for shape in self.shapes:
                points_str = ";".join(f"{pt[0]},{pt[1]}" for pt in shape["points"])
                color = shape.get("color", (0, 0, 0))
                color_str = f"{color[0]},{color[1]},{color[2]}"
                writer.writerow([
                    shape["type"],
                    points_str,
                    color_str,
                    shape.get("input", ""),
                    shape.get("phase", ""),
                    shape.get("name", ""),
                ])

    def validate_resolution(self, actual_width: int, actual_height: int) -> None:
        """Reject a resolution mismatch between this config and a video.

        No rescaling is attempted -- shape coordinates are calibrated
        pixel-exact against a specific recorded resolution, and silently
        rescaling risks subtle, hard-to-notice stop-bar/loop misplacement.

        Args:
            actual_width: The video's actual frame width.
            actual_height: The video's actual frame height.

        Raises:
            ValueError: If the resolutions don't match.
        """
        if (self.video_width, self.video_height) != (actual_width, actual_height):
            raise ValueError(
                f"Shape config resolution ({self.video_width}x{self.video_height}) "
                f"does not match video resolution ({actual_width}x{actual_height}). "
                f"Re-calibrate shapes for this video, or use the matching camera config."
            )

    def relevant_phases(self) -> List[int]:
        """Phase numbers referenced by stopbar shapes (excluding overlaps)."""
        phases = set()
        for s in self.shapes:
            if s["type"] != "stopbar" or s["phase"] is None:
                continue
            kind, num = resolve_stopbar_target(s["phase"])
            if kind == "phase":
                phases.add(num)
        return sorted(phases)

    def relevant_overlaps(self) -> List[int]:
        """Overlap numbers (1-16) referenced by stopbar shapes."""
        overlaps = set()
        for s in self.shapes:
            if s["type"] != "stopbar" or s["phase"] is None:
                continue
            kind, num = resolve_stopbar_target(s["phase"])
            if kind == "overlap":
                overlaps.add(num)
        return sorted(overlaps)

    def relevant_detectors(self) -> List[int]:
        """Detector channel numbers referenced by loop shapes."""
        return sorted({
            s["input"] for s in self.shapes
            if s["type"] == "loop" and s["input"] is not None
        })
