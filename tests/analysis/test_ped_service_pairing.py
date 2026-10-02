# Regression tests for call-to-service pairing (functional core).
#
# Found by volume_explorer against 701_data.db (2026-02-24 to 02-28): four
# services its exported workbook counts were missed. Three were a phase's
# first service, whose group id was shifted across the phase boundary; one
# was a call logged in the same decisecond as its walk. The termination
# plot's actuated/recall split shares the pairing and is pinned alongside.

import pandas as pd
import pytest

from atspm.analysis.counts import ped_counts
from atspm.plotting.termination import _classify_ped_service

T0 = pd.Timestamp("2026-02-24 07:00", tz="US/Mountain")


def _events(rows):
    df = pd.DataFrame(rows, columns=["s", "event_code", "parameter"])
    df["timestamp"] = T0 + pd.to_timedelta(df.pop("s"), unit="s")
    df["cycle_start"] = T0
    return df


def _total(rows) -> int:
    out = ped_counts(_events(rows), bin_len=60)
    return 0 if out.empty else int(out["Ped Total"].sum())


def _actuated_recall(rows):
    actuated, recall = _classify_ped_service(_events(rows))
    return len(actuated), len(recall)


# Phase 2 is served twice before phase 3's first call; the old cross-group
# shift handed phase 3's first row phase 2's running service count.
TWO_PHASES = [
    (0, 45, 2), (10, 21, 2), (100, 45, 2), (110, 21, 2),
    (200, 45, 3), (230, 21, 3),
]

SAME_DECISECOND = [(0, 21, 2), (0, 45, 2), (1.2, 45, 2)]


class TestPedCounts:

    def test_first_service_of_a_later_phase_is_counted(self):
        assert _total(TWO_PHASES) == 3

    def test_call_in_the_same_decisecond_as_its_walk_counts(self):
        assert _total(SAME_DECISECOND) == 1

    def test_recall_is_still_excluded(self):
        assert _total([(0, 45, 2), (10, 21, 2), (100, 21, 2)]) == 1

    def test_gap_marker_between_call_and_service_still_blocks(self):
        assert _total([(0, 45, 3), (5, -1, -1), (30, 21, 3)]) == 0


class TestTerminationClassification:

    def test_first_service_of_a_later_phase_is_actuated(self):
        assert _actuated_recall(TWO_PHASES) == (3, 0)

    def test_call_in_the_same_decisecond_as_its_walk_is_actuated(self):
        assert _actuated_recall(SAME_DECISECOND) == (1, 0)
