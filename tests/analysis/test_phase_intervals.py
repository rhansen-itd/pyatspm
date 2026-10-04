"""
Tests for atspm.analysis.phases._build_phase_intervals Code 12 handling.

Code 12 (Phase Inactive) during green is not a termination when a Code 8
follows: a thru phase parenting a start-delayed FYA overlap logs Code 12 at
the overlap's permissive start (315: Ph2 at the start of 2+A, 2 s into
green) while its own green continues.  A Code 12 with no Code 8 before the
next Code 1 is a dummy phase and keeps its include_no_clearance behaviour.
"""

import pandas as pd
import pytest

from atspm.analysis.phases import _build_phase_intervals


def _ph_df(events, phase=2, seg=0):
    """events: list of (timestamp, code) for one phase in one segment."""
    df = pd.DataFrame(events, columns=["timestamp", "event_code"])
    df["parameter"] = phase
    df["cycle_start"] = df["timestamp"].iloc[0]
    df["_seg"] = seg
    return df.sort_values("timestamp").reset_index(drop=True)


# Ph2 green at 0, Code 12 at +2 (FYA 2+A start), yellow 40, RC 44-46, 12 at 46.
_FYA_CYCLE = [(0.0, 1), (2.0, 12), (40.0, 8), (44.0, 9), (44.0, 10),
              (46.0, 11), (46.0, 12)]


class TestCode12InGreen:

    def test_fya_overlap_start_does_not_end_green(self):
        iv = _build_phase_intervals(_ph_df(_FYA_CYCLE))
        assert len(iv) == 1
        row = iv.iloc[0]
        assert row["green_dur"] == pytest.approx(40.0)
        assert row["clear_dur"] == pytest.approx(6.0)
        assert row["yellow_end_ts"] == pytest.approx(44.0)

    def test_fya_cycles_back_to_back(self):
        second = [(t + 100.0, c) for t, c in _FYA_CYCLE]
        iv = _build_phase_intervals(_ph_df(_FYA_CYCLE + second))
        assert list(iv["green_dur"]) == pytest.approx([40.0, 40.0])

    def test_fya_cycle_ignores_include_no_clearance(self):
        iv = _build_phase_intervals(_ph_df(_FYA_CYCLE),
                                    include_no_clearance=True)
        assert list(iv["green_dur"]) == pytest.approx([40.0])


class TestDummyPhase:
    # Code 1 -> Code 12 with no Code 8 before the next Code 1.
    _DUMMY = [(0.0, 1), (15.0, 12), (60.0, 1), (72.0, 12)]

    def test_dropped_by_default(self):
        assert _build_phase_intervals(_ph_df(self._DUMMY)).empty

    def test_green_only_when_requested(self):
        iv = _build_phase_intervals(_ph_df(self._DUMMY),
                                    include_no_clearance=True)
        assert list(iv["green_dur"]) == pytest.approx([15.0, 12.0])
        assert list(iv["clear_dur"]) == pytest.approx([0.0, 0.0])
        assert list(iv["clear_end_ts"]) == pytest.approx([15.0, 72.0])

    def test_first_code_12_is_the_end(self):
        events = [(0.0, 1), (15.0, 12), (16.0, 12), (60.0, 1)]
        iv = _build_phase_intervals(_ph_df(events), include_no_clearance=True)
        assert list(iv["green_dur"]) == pytest.approx([15.0])


class TestCode12InClearance:

    def test_code_12_during_yellow_ends_at_yellow(self):
        events = [(0.0, 1), (30.0, 8), (33.0, 12)]
        iv = _build_phase_intervals(_ph_df(events))
        assert len(iv) == 1
        assert iv.iloc[0]["clear_end_ts"] == pytest.approx(30.0)

    def test_code_12_after_end_yellow(self):
        events = [(0.0, 1), (30.0, 8), (34.0, 9), (34.5, 12)]
        iv = _build_phase_intervals(_ph_df(events))
        assert iv.iloc[0]["clear_dur"] == pytest.approx(4.0)


class TestReServiceAtEndOfRedClearance:
    """A phase re-served the instant its red clearance ends logs Code 11 and
    the next Code 1 in the same decisecond (201, 2026-03-18 18:17:45.6)."""

    _TWO = [(0.0, 1), (8.7, 8), (13.7, 9), (13.7, 10), (15.8, 1), (15.8, 11),
            (25.0, 8), (30.0, 9), (30.0, 10), (32.1, 11), (32.1, 12)]

    def test_both_greens_are_emitted(self):
        iv = _build_phase_intervals(_ph_df(self._TWO))
        assert list(iv["green_ts"]) == pytest.approx([0.0, 15.8])
        assert list(iv["clear_end_ts"]) == pytest.approx([15.8, 32.1])
        assert list(iv["yellow_end_ts"]) == pytest.approx([13.7, 30.0])

    def test_input_row_order_does_not_matter(self):
        swapped = [(15.8, 11) if e == (15.8, 1) else (15.8, 1) if e == (15.8, 11) else e
                   for e in self._TWO]
        iv = _build_phase_intervals(_ph_df(swapped))
        assert list(iv["green_ts"]) == pytest.approx([0.0, 15.8])
