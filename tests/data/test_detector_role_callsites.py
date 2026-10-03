# Acceptance tests for the S-D0 call-site migration onto the detector role
# table (spec docs/specs/detector_roles_migration.md).
#
# Opus-written; the migration must make these pass without editing them.

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import atspm
from atspm.analysis import critical
from atspm.data.aog import AogEngine
from atspm.data.flow import FlowRateEngine
from atspm.data.manager import DatabaseManager
from atspm.data.reader import get_det_config
from atspm.data.split_failures import SplitFailureEngine
from tests.analysis.test_detector_roles import CFG_201, CFG_315

TZ = "US/Mountain"
SRC = Path(atspm.__file__).parent
REPO = SRC.parent.parent

CFG_201_PATHS = {
    "db": REPO / "intersections/201_SH-55_and_Banks-Lowman_Rd/201_data.db",
    "day": "2026-06-21",
}
CFG_315_PATHS = {
    "db": REPO / "intersections/315_US-20-26_Franklin_Rd_and_KCID_Rd/315_data.db",
    "day": "2025-12-15",
}


def _engine(cls, tmp_path):
    return cls(tmp_path / "unused.db", timezone=TZ)


# ---------------------------------------------------------------------------
# Per-engine resolvers on the real 201 / 315 config rows
# ---------------------------------------------------------------------------

class TestFlowResolver:
    def test_201(self, tmp_path):
        got = _engine(FlowRateEngine, tmp_path)._resolve_detector_map(CFG_201, None)
        assert got == {2: [60], 3: [64], 4: [41], 6: [43], 7: [40], 8: [63]}

    def test_315(self, tmp_path):
        got = _engine(FlowRateEngine, tmp_path)._resolve_detector_map(CFG_315, None)
        assert got == {1: [17], 2: [26, 27, 28], 3: [22], 4: [31], 5: [25],
                       6: [18, 19, 20], 7: [30], 8: [23]}

    @pytest.mark.parametrize("key", ["Det_P2_Stopbar", "Det_P2_Stop_Bar"])
    def test_either_spelling(self, tmp_path, key):
        got = _engine(FlowRateEngine, tmp_path)._resolve_detector_map({key: "9,8"}, None)
        assert got == {2: [8, 9]}

    def test_requested_phase_filter_and_warning(self, tmp_path, capsys):
        got = _engine(FlowRateEngine, tmp_path)._resolve_detector_map(CFG_201, [2, 5])
        assert got == {2: [60]}
        assert "phase 5 skipped" in capsys.readouterr().out


class TestAogResolver:
    def test_315(self, tmp_path):
        got = _engine(AogEngine, tmp_path)._resolve_detector_map(CFG_315, None)
        assert got == {2: [54, 55, 56], 6: [38, 39, 40]}

    def test_201(self, tmp_path):
        got = _engine(AogEngine, tmp_path)._resolve_detector_map(CFG_201, None)
        assert got == {2: [49], 6: [36]}

    def test_requested_phase_filter_and_warning(self, tmp_path, capsys):
        got = _engine(AogEngine, tmp_path)._resolve_detector_map(CFG_315, [6, 4])
        assert got == {6: [38, 39, 40]}
        assert "phase 4 skipped" in capsys.readouterr().out


class TestCriticalMovementMap:
    def test_201_movements_map_through_stop_bar(self):
        out = critical.movement_phase_map(CFG_201).set_index("movement")
        assert out.loc["NBT", "phase"] == 2
        assert out.loc["SBT", "phase"] == 6
        assert out.loc["WBT", "phase"] == 4
        assert out.loc["EBT", "phase"] == 8
        assert pd.isna(out.loc["WBR", "phase"])


# ---------------------------------------------------------------------------
# Coordination plot detector config, now built from the role table
# ---------------------------------------------------------------------------

def _config_db(tmp_path: Path, cfg: dict) -> Path:
    db = tmp_path / "cfg.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ)
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00",
                              "end_date": None, **cfg})
        m.conn.commit()
    return db


class TestDetConfig:
    def test_201(self, tmp_path):
        db = _config_db(tmp_path, CFG_201)
        got = get_det_config(db, datetime(2026, 6, 21))
        assert got == {
            "P2 Arrival": "49", "P6 Arrival": "36",
            "P2 Stop Bar": "60", "P3 Stop Bar": "64", "P4 Stop Bar": "41",
            "P6 Stop Bar": "43", "P7 Stop Bar": "40", "P8 Stop Bar": "63",
            "P2 Occupancy": "46", "P3 Occupancy": "53", "P4 Occupancy": "39",
            "P6 Occupancy": "33", "P7 Occupancy": "38", "P8 Occupancy": "51",
        }

    def test_both_spellings_give_one_stop_bar_entry(self, tmp_path):
        db = _config_db(tmp_path, {"Det_P2_Stopbar": "5,6", "Det_P2_Stop_Bar": "6,7",
                                   "Det_P2_Arrival": "12,11"})
        got = get_det_config(db, datetime(2026, 6, 21))
        assert got == {"P2 Arrival": "11,12", "P2 Stop Bar": "5,6,7"}


# ---------------------------------------------------------------------------
# One parser: no detector-key matching left outside detector_roles.py
# ---------------------------------------------------------------------------

# manager.py keeps _parse_detector_pairs (ordered pair list for the
# detector-comparison engine); everything else goes through the role table.
_ALLOWED = {"analysis/detector_roles.py", "data/manager.py"}

_KEY_MATCHING = [
    re.compile(r"Det_P\(\\d\+\)"),                      # regex literal on Det_P keys
    re.compile(r"""endswith\(\s*["']_(Arrival|Stopbar|Stop_Bar|Occupancy|Pairs)["']"""),
    re.compile(r"""startswith\(\s*["']Det_"""),
    re.compile(r"""\[\s*5\s*:\s*\w+\.index\("""),    # key[5:key.index("_Arrival")]
]


def test_no_detector_key_parsing_outside_role_module():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in _ALLOWED:
            continue
        text = path.read_text()
        for pat in _KEY_MATCHING:
            if pat.search(text):
                offenders.append(f"{rel}: {pat.pattern}")
    assert offenders == []


def test_critical_private_parsers_removed():
    for name in ("_parse_stopbar_sets", "_parse_occupancy_sets",
                 "_parse_detector_sets", "_STOPBAR_KEY_RE", "_OCCUPANCY_KEY_RE"):
        assert not hasattr(critical, name), name


def test_role_parser_is_exported():
    from atspm.analysis import detector_sets, parse_detector_roles  # noqa: F401


# ---------------------------------------------------------------------------
# Real data: flow and split failures find detectors on 201 / 315
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("site,phase", [(CFG_201_PATHS, 6), (CFG_315_PATHS, 2)])
def test_flow_on_real_db_is_not_empty(site, phase):
    if not site["db"].exists():
        pytest.skip("corpus DB not present")
    out = FlowRateEngine(site["db"]).flow(site["day"], site["day"], phases=[phase],
                                          make_plot=False)
    assert out and not out["cycle"].empty
    assert set(out["cycle"]["phase"]) == {phase}


def test_split_failures_on_315_reads_presence():
    db = CFG_315_PATHS["db"]
    if not db.exists():
        pytest.skip("corpus DB not present")
    out = SplitFailureEngine(db).split_failures(
        "2025-12-15 06:00", "2025-12-15 09:00", phases=[2], make_plot=False)
    assert set(out["lane"]["det"]) == {50, 51, 52}
