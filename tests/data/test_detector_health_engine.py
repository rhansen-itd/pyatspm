"""Acceptance gate for the S-D4 DetectorHealthEngine (imperative shell).

Opus-written; the engine itself is built by the delegated run.  These assert
the observable contract, not internal structure:

* findings computed by the core are persisted to ``detector_findings``;
* the WD_Ignore list is applied at *read/report* time, so an ignored finding is
  still recorded in the table;
* ``--min-severity`` / ``--window`` shape only the reported view;
* the exit code follows the highest reported severity;
* a re-run over the same range is idempotent.

A controlled silent-detector scenario gives deterministic ConfiguredSilent
(high) findings through the real rule core.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

from atspm.data.manager import DatabaseManager
# Built by the delegated S-D4 run; this import fails until the engine exists,
# which is the red state this acceptance gate starts in.
from atspm.data.detector_health import DetectorHealthEngine

TZ = "US/Mountain"
_MT = pytz.timezone(TZ)
DATE = "2026-01-10"


def _loc(s: str) -> float:
    return _MT.localize(pd.Timestamp(s).to_pydatetime(), is_dst=None).timestamp()


def _heartbeat(det: int, step_s: float = 600.0):
    """A detector pulsing across the whole local day so bins count as logged."""
    ts = np.arange(_loc(f"{DATE} 00:00:05"), _loc(f"{DATE} 23:59:55"), step_s)
    return [(t, 82, det) for t in ts] + [(t + 0.2, 81, det) for t in ts]


@pytest.fixture
def seeded_db(empty_db: Path) -> Path:
    """A day fully logged by detector 99, with 52 and 53 configured but silent."""
    from ..conftest import seed_events

    seed_events(empty_db, _heartbeat(99))
    with DatabaseManager(empty_db) as m:
        m.set_metadata(intersection_id="201", intersection_name="Scratch",
                       timezone=TZ, major_road_route="US-1", major_road_name="Main",
                       minor_road_route="SR-2", minor_road_name="Cross")
        cols = {
            "Det_P2_Stop_Bar": "52,53",  # configured, never actuate -> silent
            "TM_NBT": "99",              # detector 99 is configured, so no finding
            "WD_Ignore": "52:ConfiguredSilent",
        }
        for c in cols:
            m.add_config_column(c)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00", "end_date": None, **cols})
        m.conn.commit()
    return empty_db


def _db_pairs(db_path: Path):
    with DatabaseManager(db_path) as m:
        df = m.get_findings(DATE, DATE)
    return df


class TestEnginePersistAndReport:
    def test_findings_persisted_including_ignored(self, seeded_db: Path):
        eng = DetectorHealthEngine(seeded_db, timezone=TZ)
        eng.detector_health(DATE, DATE, window="day", min_severity="low")

        df = _db_pairs(seeded_db)
        pairs = set(zip(df["detector"], df["rule"]))
        # Both silent detectors are recorded, including the ignored one (52).
        assert (52, "ConfiguredSilent") in pairs
        assert (53, "ConfiguredSilent") in pairs

    def test_ignore_applied_to_reported_only(self, seeded_db: Path):
        eng = DetectorHealthEngine(seeded_db, timezone=TZ)
        result = eng.detector_health(DATE, DATE, window="day", min_severity="low")

        reported = result["reported"]
        rpairs = set(zip(reported["detector"], reported["rule"]))
        assert (53, "ConfiguredSilent") in rpairs       # kept
        assert (52, "ConfiguredSilent") not in rpairs    # ignored out of the report
        assert set(reported["window"]) <= {"day"}        # --window filters the view

    def test_exit_code_follows_highest_reported(self, seeded_db: Path):
        eng = DetectorHealthEngine(seeded_db, timezone=TZ)
        result = eng.detector_health(DATE, DATE, window="day", min_severity="low")
        assert result["exit_code"] == 2  # a reported ConfiguredSilent is high

    def test_min_severity_high_still_reports_high(self, seeded_db: Path):
        eng = DetectorHealthEngine(seeded_db, timezone=TZ)
        result = eng.detector_health(DATE, DATE, window="day", min_severity="high")
        assert (53, "ConfiguredSilent") in set(
            zip(result["reported"]["detector"], result["reported"]["rule"])
        )

    def test_rerun_is_idempotent(self, seeded_db: Path):
        eng = DetectorHealthEngine(seeded_db, timezone=TZ)
        eng.detector_health(DATE, DATE, window="day")
        n1 = len(_db_pairs(seeded_db))
        eng.detector_health(DATE, DATE, window="day")
        n2 = len(_db_pairs(seeded_db))
        assert n1 == n2 and n1 > 0

    def test_output_dir_writes_csv_and_html(self, seeded_db: Path, tmp_path: Path):
        out = tmp_path / "outputs"
        eng = DetectorHealthEngine(seeded_db, timezone=TZ)
        eng.detector_health(DATE, DATE, window="day", output_dir=out)
        assert any(p.suffix == ".csv" for p in out.iterdir())
        assert any(p.suffix == ".html" for p in out.iterdir())
