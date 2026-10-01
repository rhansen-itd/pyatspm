"""Golden tests for the ``lamp`` shape type and the ``indication`` column.

Target: src/atspm/data/video.py — ShapeConfig (load/save, relevant_phases,
relevant_overlaps, lamp_shapes), LAMP_INDICATIONS.

FROZEN: written by Opus.  An implementation may not edit this file; if a
test looks wrong, stop and ask.
"""

import pytest

from atspm.data.video import LAMP_INDICATIONS, ShapeConfig

_OLD_HEADER = "video_width,video_height\n720,720\ntype,points,color,input,phase,name\n"
_NEW_HEADER = "video_width,video_height\n720,720\ntype,points,color,input,phase,name,indication\n"


def _write(tmp_path, text):
    p = tmp_path / "fisheye_shapes.csv"
    p.write_text(text)
    return p


def test_indications_constant():
    assert tuple(LAMP_INDICATIONS) == ("green", "yellow", "red")


def test_single_point_lamp_loads(tmp_path):
    p = _write(tmp_path, _NEW_HEADER + 'lamp,"643,417","0,255,0",,2,NB head,green\n')
    cfg = ShapeConfig.load(p)
    (lamp,) = cfg.shapes
    assert lamp["type"] == "lamp"
    assert lamp["points"] == [(643, 417)]
    assert lamp["phase"] == "2"
    assert lamp["indication"] == "green"
    assert lamp["name"] == "NB head"


def test_polygon_lamp_and_overlap_target(tmp_path):
    p = _write(tmp_path, _NEW_HEADER + 'lamp,"649,416;655,416;655,421;649,421","0,0,255",,OLB,,red\n')
    cfg = ShapeConfig.load(p)
    assert cfg.shapes[0]["points"] == [(649, 416), (655, 416), (655, 421), (649, 421)]
    assert cfg.relevant_overlaps() == [2]
    assert cfg.relevant_phases() == []


def test_indication_is_normalised_to_lower_case(tmp_path):
    p = _write(tmp_path, _NEW_HEADER + 'lamp,"643,417","0,255,0",,2,,Green\n')
    assert ShapeConfig.load(p).shapes[0]["indication"] == "green"


def test_lamp_phases_count_as_relevant(tmp_path):
    # The overlay must fetch phase events for a lamp-only config, so it can
    # draw the DB-state dot beside the real lamp.
    p = _write(tmp_path, _NEW_HEADER
               + 'lamp,"643,417","0,255,0",,2,,green\n'
               + 'stopbar,"10,10;20,20","0,0,0",,4,,\n')
    assert ShapeConfig.load(p).relevant_phases() == [2, 4]


def test_lamp_shapes_accessor_keeps_file_order(tmp_path):
    p = _write(tmp_path, _NEW_HEADER
               + 'lamp,"652,418","0,0,255",,2,,red\n'
               + 'loop,"1,1;2,1;2,2","255,0,0",38,,,\n'
               + 'lamp,"643,417","0,255,0",,2,,green\n')
    lamps = ShapeConfig.load(p).lamp_shapes()
    assert [s["indication"] for s in lamps] == ["red", "green"]


def test_old_file_without_indication_column_still_loads(tmp_path):
    p = _write(tmp_path, _OLD_HEADER
               + 'stopbar,"470,537;529,511","0,0,0",,4,\n'
               + 'loop,"429,579;431,628;441,628","255,0,0",38,,\n')
    cfg = ShapeConfig.load(p)
    assert len(cfg.shapes) == 2
    assert all(s.get("indication") is None for s in cfg.shapes)
    assert cfg.relevant_phases() == [4]
    assert cfg.relevant_detectors() == [38]


def test_round_trip_writes_indication_column(tmp_path):
    p = _write(tmp_path, _NEW_HEADER
               + 'lamp,"643,417","0,255,0",,2,NB head,green\n'
               + 'stopbar,"470,537;529,511","0,0,0",,4,,\n')
    cfg = ShapeConfig.load(p)
    out = tmp_path / "out.csv"
    cfg.save(out)
    lines = out.read_text().splitlines()
    assert lines[2] == "type,points,color,input,phase,name,indication"
    again = ShapeConfig.load(out)
    assert again.shapes[0]["indication"] == "green"
    assert again.shapes[0]["points"] == [(643, 417)]
    assert again.shapes[1].get("indication") is None
    assert again.relevant_phases() == [2, 4]


@pytest.mark.parametrize("row,match", [
    ('lamp,"643,417","0,255,0",,2,,blue\n', "indication"),
    ('lamp,"643,417","0,255,0",,2,,\n', "indication"),      # lamp without one
    ('lamp,"643,417","0,255,0",,,,green\n', "phase"),       # lamp without a target
    ('lamp,"643,417","0,255,0",,17,,green\n', "range"),     # resolve_stopbar_target's check
])
def test_bad_lamp_rows_are_rejected_at_load(tmp_path, row, match):
    p = _write(tmp_path, _NEW_HEADER + row)
    with pytest.raises(ValueError, match=match):
        ShapeConfig.load(p)
