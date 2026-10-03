# Acceptance tests for the detector-inference shell engine and CLI
# (UDOT S-D6; spec docs/specs/detector_inference_shell.md).
#
# Opus-written; the implementation must make these pass without editing them.
# The inference itself is pinned in tests/analysis/test_detector_inference.py.

from pathlib import Path

import pandas as pd
import pytest

from atspm import cli
from atspm.analysis.detector_inference import diff_detector_roles, infer_detector_roles
from atspm.analysis.detector_roles import parse_detector_roles
from atspm.data.detector_inference import _INFERENCE_CODES, DetectorInferenceEngine
from atspm.data.manager import DatabaseManager
from tests.analysis.test_detector_inference import _synthetic

TZ = "US/Mountain"
START, END = "2023-11-14", "2023-11-16"        # covers the whole synthetic span
STAMP = "2023_11_14-2023_11_16"

CFG = {
    "RB_R1": "2|4",
    "Det_P2_Occupancy": "2",          # match
    "Det_P4_Stop_Bar": "4",           # conflict (it is P4 presence)
    "Det_P8_Occupancy": "77",         # silent
    "Det_P2_Arrival": "1",            # match
}


def _build_db(root: Path, events: pd.DataFrame, cfg=CFG) -> Path:
    db = root / "inf.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00", "end_date": None, **cfg})
        m.insert_events(list(events.itertuples(index=False, name=None)))
        m.conn.commit()
    return db


@pytest.fixture(scope="module")
def events():
    return _synthetic(days=2)


@pytest.fixture
def db(tmp_path, events):
    return _build_db(tmp_path, events)


def _config_rows(db: Path) -> int:
    with DatabaseManager(db) as m:
        return m.conn.execute("SELECT COUNT(*) FROM config").fetchone()[0]


class TestEngine:

    def test_codes(self):
        assert sorted(_INFERENCE_CODES) == [-1, 1, 8, 9, 81, 82]

    def test_reproduces_the_core(self, db, events):
        res = DetectorInferenceEngine(db).infer(START, END)
        exp = infer_detector_roles(events, phases=[2, 4])
        pd.testing.assert_frame_equal(res["proposed"].reset_index(drop=True), exp)

    def test_diff_against_active_config(self, db, events):
        res = DetectorInferenceEngine(db).infer(START, END)
        counts = events[events["event_code"] == 82].groupby("parameter").size().to_dict()
        exp = diff_detector_roles(infer_detector_roles(events, phases=[2, 4]),
                                  parse_detector_roles(CFG), counts)
        pd.testing.assert_frame_equal(res["diff"].reset_index(drop=True), exp)
        status = res["diff"].set_index("detector")["status"]
        assert status[2] == "match" and status[1] == "match"
        assert status[4] == "conflict" and status[77] == "silent"

    def test_ring_config_limits_candidate_phases(self, tmp_path):
        ev = _synthetic(days=2, p6_offset=0.0)             # P6 always starts with P2
        db = _build_db(tmp_path, ev)
        ring = DetectorInferenceEngine(db).infer(START, END)["proposed"].set_index("detector")
        assert ring.loc[2, "phase"] == 2                     # RB_R1 "2|4": 6 not a candidate
        every = DetectorInferenceEngine(db).infer(START, END, use_ring_config=False)
        every = every["proposed"].set_index("detector")
        assert pd.isna(every.loc[2, "phase"])
        assert every.loc[2, "candidates"] == "P2|P6"

    def test_no_ring_config_uses_every_phase(self, tmp_path):
        cfg = {k: v for k, v in CFG.items() if k != "RB_R1"}
        db = _build_db(tmp_path, _synthetic(days=2, p6_offset=0.0), cfg)
        res = DetectorInferenceEngine(db).infer(START, END)["proposed"].set_index("detector")
        assert res.loc[2, "candidates"] == "P2|P6"

    def test_min_actuations_is_passed_through(self, db):
        res = DetectorInferenceEngine(db).infer(START, END, min_actuations=10**9)
        assert res["proposed"].empty
        assert set(res["diff"]["status"]) <= {"silent", "low_volume"}

    def test_writes_csvs_and_never_touches_config(self, db, tmp_path, capsys):
        before = _config_rows(db)
        out = tmp_path / "out"
        assert DetectorInferenceEngine(db).infer(START, END, output_dir=out) is None
        assert (out / f"Detector_Inference_{STAMP}.csv").exists()
        assert (out / f"Detector_Inference_Diff_{STAMP}.csv").exists()
        diff = pd.read_csv(out / f"Detector_Inference_Diff_{STAMP}.csv")
        assert list(diff.columns)[:2] == ["detector", "status"]
        assert _config_rows(db) == before
        printed = capsys.readouterr().out
        assert "conflict" in printed and "silent" in printed

    def test_empty_window(self, db):
        res = DetectorInferenceEngine(db).infer("2010-01-01", "2010-01-02")
        assert res["proposed"].empty
        assert set(res["diff"]["status"]) == {"silent"}


class TestCli:

    @staticmethod
    def _parse(*argv):
        return cli._build_parser().parse_args(["infer-detectors", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "900", "--start", START, "--end", END)
        assert args.func is cli.handle_infer_detectors
        assert args.min_actuations == 50
        assert args.all_phases is False

    def test_options(self):
        args = self._parse("--all", "--start", START, "--end", END,
                           "--min-actuations", "20", "--all-phases")
        assert args.min_actuations == 20 and args.all_phases is True

    def test_target_group_is_required_and_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse("--start", START, "--end", END)
        with pytest.raises(SystemExit):
            self._parse("--all", "--targetid", "900", "--start", START, "--end", END)
