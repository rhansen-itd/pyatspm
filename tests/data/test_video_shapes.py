"""Edge-case tests for the video shape-config CSV round-trip.

Target: src/atspm/data/video.py — ShapeConfig, resolve_stopbar_target,
OVERLAP_LETTER_MAP.  Imperative Shell, but pure file I/O: tmp_path only,
no DB, no video.
"""

import pytest

from atspm.data.video import OVERLAP_LETTER_MAP, ShapeConfig, resolve_stopbar_target

_HEADER = "video_width,video_height\n1920,1080\ntype,points,color,input,phase,name\n"


def _write(tmp_path, body, header=_HEADER):
    p = tmp_path / "cam1_shapes.csv"
    p.write_text(header + body)
    return p


class TestOverlapLetterMap:

    def test_sixteen_overlaps_a_through_p(self):
        # 16, not 26 — the legacy range(26) was a bug, not a spec requirement.
        assert len(OVERLAP_LETTER_MAP) == 16
        assert OVERLAP_LETTER_MAP["OLA"] == 1
        assert OVERLAP_LETTER_MAP["OLP"] == 16
        assert "OLQ" not in OVERLAP_LETTER_MAP


class TestResolveStopbarTarget:

    @pytest.mark.parametrize("field,expected", [
        ("OLA", ("overlap", 1)),
        ("olp", ("overlap", 16)),          # case-insensitive
        ("  OLB  ", ("overlap", 2)),       # whitespace-tolerant
        ("3", ("phase", 3)),
        (4, ("phase", 4)),
    ])
    def test_valid_fields(self, field, expected):
        assert resolve_stopbar_target(field) == expected

    @pytest.mark.parametrize("field", ["OLQ", "OL", "abc", "2.5", ""])
    def test_invalid_fields_raise(self, field):
        with pytest.raises(ValueError, match="Unrecognised stopbar phase field"):
            resolve_stopbar_target(field)

    def test_phase_number_is_not_range_checked(self):
        # Documented looseness: any integer is accepted as a phase number;
        # nothing validates it against the controller's real phase range.
        assert resolve_stopbar_target("99") == ("phase", 99)


class TestShapeConfigRoundTrip:

    _SHAPES = [
        {
            "type": "loop",
            "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
            "color": (0, 255, 0),
            "input": 3,
            "phase": None,
            "name": 'NB, loop "A"',   # comma and quotes must survive CSV
        },
        {
            "type": "stopbar",
            "points": [(5, 30), (25, 30)],
            "color": (0, 0, 255),
            "input": None,
            "phase": "OLB",
            "name": None,
        },
    ]

    def test_save_then_load_is_identity(self, tmp_path):
        p = tmp_path / "cam1_shapes.csv"
        ShapeConfig(shapes=self._SHAPES, video_width=1920, video_height=1080).save(p)
        back = ShapeConfig.load(p)
        assert (back.video_width, back.video_height) == (1920, 1080)
        assert back.shapes == self._SHAPES

    def test_double_round_trip_is_stable(self, tmp_path):
        p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
        ShapeConfig(shapes=self._SHAPES, video_width=1920, video_height=1080).save(p1)
        ShapeConfig.load(p1).save(p2)
        assert p1.read_text() == p2.read_text()


class TestShapeConfigLoadEdgeCases:

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ShapeConfig.load(tmp_path / "nope.csv")

    def test_empty_file_loads_as_empty_config(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        sc = ShapeConfig.load(p)
        assert sc.shapes == []
        assert sc.video_width is None and sc.video_height is None

    def test_header_only_file_loads_metadata_and_no_shapes(self, tmp_path):
        p = _write(tmp_path, "")
        sc = ShapeConfig.load(p)
        assert (sc.video_width, sc.video_height) == (1920, 1080)
        assert sc.shapes == []

    def test_blank_metadata_row_yields_none_resolution(self, tmp_path):
        p = _write(tmp_path, "", header="video_width,video_height\n,\n" + _HEADER.splitlines()[2] + "\n")
        sc = ShapeConfig.load(p)
        assert sc.video_width is None and sc.video_height is None

    def test_empty_color_field_falls_back_to_green(self, tmp_path):
        p = _write(tmp_path, 'loop,"10,10;20,20",,3,,x\n')
        sc = ShapeConfig.load(p)
        assert sc.shapes[0]["color"] == (0, 255, 0)

    def test_empty_input_phase_name_load_as_none(self, tmp_path):
        p = _write(tmp_path, 'loop,"10,10;20,20","0,255,0",,,\n')
        (shape,) = ShapeConfig.load(p).shapes
        assert shape["input"] is None
        assert shape["phase"] is None
        assert shape["name"] is None

    @pytest.mark.parametrize("bad_points", [
        "100",        # a point with no y coordinate
        "a;b",        # non-numeric points
        "",           # empty points cell
        "10,10;",     # trailing separator leaves an empty point
    ])
    def test_malformed_points_raise_valueerror(self, tmp_path, bad_points):
        p = _write(tmp_path, f'loop,{bad_points},"0,255,0",3,,x\n')
        with pytest.raises(ValueError):
            ShapeConfig.load(p)


class TestValidateResolution:

    def _cfg(self, w=1920, h=1080):
        return ShapeConfig(video_width=w, video_height=h)

    def test_exact_match_passes(self):
        self._cfg().validate_resolution(1920, 1080)  # no raise

    @pytest.mark.parametrize("actual", [
        (1921, 1080),   # width off by one
        (1919, 1080),
        (1920, 1081),   # height off by one
        (1920, 1079),
        (1080, 1920),   # transposed
    ])
    def test_any_mismatch_raises(self, actual):
        with pytest.raises(ValueError, match="does not match video resolution"):
            self._cfg().validate_resolution(*actual)

    def test_error_message_names_both_resolutions(self):
        with pytest.raises(ValueError, match=r"1920x1080.*1280x720"):
            self._cfg().validate_resolution(1280, 720)

    def test_unknown_config_resolution_never_validates(self):
        # A config whose metadata row was blank must reject every video
        # rather than silently pass.
        with pytest.raises(ValueError):
            self._cfg(w=None, h=None).validate_resolution(1920, 1080)


class TestRelevantTargets:

    _SHAPES = [
        {"type": "stopbar", "points": [(0, 0), (1, 1)], "color": (0, 0, 255), "input": None, "phase": "4", "name": None},
        {"type": "stopbar", "points": [(0, 0), (1, 1)], "color": (0, 0, 255), "input": None, "phase": "2", "name": None},
        {"type": "stopbar", "points": [(0, 0), (1, 1)], "color": (0, 0, 255), "input": None, "phase": "2", "name": None},   # duplicate
        {"type": "stopbar", "points": [(0, 0), (1, 1)], "color": (0, 0, 255), "input": None, "phase": "OLC", "name": None},
        {"type": "stopbar", "points": [(0, 0), (1, 1)], "color": (0, 0, 255), "input": None, "phase": None, "name": None},  # uncalibrated
        {"type": "loop", "points": [(0, 0), (1, 1)], "color": (0, 255, 0), "input": 7, "phase": None, "name": None},
        {"type": "loop", "points": [(0, 0), (1, 1)], "color": (0, 255, 0), "input": 3, "phase": None, "name": None},
        {"type": "loop", "points": [(0, 0), (1, 1)], "color": (0, 255, 0), "input": None, "phase": None, "name": None},     # uncalibrated
    ]

    def test_relevant_phases_sorted_deduped_overlaps_excluded(self):
        assert ShapeConfig(shapes=self._SHAPES).relevant_phases() == [2, 4]

    def test_relevant_overlaps(self):
        assert ShapeConfig(shapes=self._SHAPES).relevant_overlaps() == [3]

    def test_relevant_detectors_sorted_none_skipped(self):
        assert ShapeConfig(shapes=self._SHAPES).relevant_detectors() == [3, 7]

    def test_empty_config_yields_empty_lists(self):
        sc = ShapeConfig()
        assert sc.relevant_phases() == []
        assert sc.relevant_overlaps() == []
        assert sc.relevant_detectors() == []
