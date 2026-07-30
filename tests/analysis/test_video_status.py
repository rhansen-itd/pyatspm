"""Adversarial edge-case tests for video point-in-time status lookups.

Target: src/atspm/analysis/video.py — phase_status_at_timestamps,
overlap_status_at_timestamps, detector_status_at_timestamps,
first_phase_transition_after.  Pure functions, no mocking, no DB.

Gap-rule audit result (CLAUDE.md §5), Fable #5:
    The residual risk documented in the module docstring — an inferred
    steady-red period whose gap crosses a segment boundary being mislabeled
    'R' — was probed exhaustively (every single/double marker placement in
    representative streams, including markers landing exactly on green_ts
    in every physical row order, plus 12,000 randomized trials) and CANNOT
    fire in the current implementation.  Structural reason: 'R' requires
    seg_arr[vi] == seg_arr[vi + 1], every emitted interval's events lie
    wholly within its segment's time span, so equal-segment neighbours can
    never bracket a marker.  Marker/event timestamp ties can only DROP an
    interval (widening 'na'), never leak a stale colour.  The tests in
    TestPhaseStatusGapRule pin that guarantee.

Genuine defect found instead, since fixed (TestPhaseRedClearance):
    a phase that serves red clearance (Codes 10/11) had its red-clearance
    period labeled 'Y', because _build_phase_intervals only emitted
    clear_end_ts = Code 11 (End Red Clearance) — correct for phase_splits'
    combined-YR reporting, wrong for a visual G/Y/R lookup.  It now also
    emits yellow_end_ts (Code 9, or Code 10 when no Code 9 was logged) and
    the lookup ends 'Y' there, matching what the overlap builder already
    did via Code 64.
"""

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.video import (
    detector_status_at_timestamps,
    first_phase_transition_after,
    overlap_status_at_timestamps,
    phase_status_at_timestamps,
)

T0 = 1_609_459_200.0  # 2021-01-01 00:00:00 UTC

PHASE = 2
OVERLAP = 1
DET = 5


def _events(rows) -> pd.DataFrame:
    """rows: iterable of (offset_sec, event_code, parameter)."""
    df = pd.DataFrame(
        [(T0 + off, code, param) for off, code, param in rows],
        columns=["timestamp", "event_code", "parameter"],
    )
    df["cycle_start"] = T0  # required by _build_phase_intervals; value unused here
    return df


def _q(*offsets) -> np.ndarray:
    return np.array([T0 + o for o in offsets], dtype=np.float64)


def _phase_at(rows, *offsets):
    return list(phase_status_at_timestamps(_events(rows), PHASE, _q(*offsets)))


def _overlap_at(rows, *offsets):
    return list(overlap_status_at_timestamps(_events(rows), OVERLAP, _q(*offsets)))


# Two complete phase intervals in one contiguous segment:
#   A: green 100, yellow 110, end-yellow 114, inactive 116
#   B: green 200, yellow 210, end-yellow 214, inactive 216
_TWO_PHASE_INTERVALS = [
    (100.0, 1, PHASE), (110.0, 8, PHASE), (114.0, 9, PHASE), (116.0, 12, PHASE),
    (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
]

# Overlap twin of the above (65 = Off, red indication still shown):
_TWO_OVERLAP_INTERVALS = [
    (100.0, 61, OVERLAP), (110.0, 63, OVERLAP), (114.0, 64, OVERLAP), (116.0, 65, OVERLAP),
    (200.0, 61, OVERLAP), (210.0, 63, OVERLAP), (214.0, 64, OVERLAP), (216.0, 65, OVERLAP),
]


class TestPhaseStatusBoundaries:
    """np.searchsorted(side='right') boundary matrix from the roadmap."""

    def test_before_first_interval_is_na(self):
        assert _phase_at(_TWO_PHASE_INTERVALS, 99.9) == ["na"]

    def test_exactly_on_green_onset_is_green(self):
        # side='right' includes equality: the frame at the onset instant
        # already shows the new colour.
        assert _phase_at(_TWO_PHASE_INTERVALS, 100.0) == ["G"]

    def test_exactly_on_yellow_onset_is_yellow(self):
        assert _phase_at(_TWO_PHASE_INTERVALS, 110.0) == ["Y"]

    def test_exactly_on_clearance_end_is_red(self):
        assert _phase_at(_TWO_PHASE_INTERVALS, 114.0) == ["R"]

    def test_inferred_red_between_intervals_same_segment(self):
        assert _phase_at(_TWO_PHASE_INTERVALS, 150.0) == ["R"]

    def test_exactly_on_next_green_flips_to_green(self):
        assert _phase_at(_TWO_PHASE_INTERVALS, 200.0) == ["G"]

    def test_after_last_interval_is_na_not_red(self):
        # No confirmed next green — the trailing red is unproven.  This is
        # why video/processor.py fetches lookahead_minutes past the video.
        assert _phase_at(_TWO_PHASE_INTERVALS, 214.0, 216.0, 500.0) == ["na", "na", "na"]

    def test_query_order_is_preserved_for_unsorted_input(self):
        assert _phase_at(_TWO_PHASE_INTERVALS, 150.0, 105.0, 500.0, 112.0) == [
            "R", "G", "na", "Y",
        ]

    def test_empty_events_frame_is_all_na(self):
        df = _events([]).iloc[0:0]
        assert list(phase_status_at_timestamps(df, PHASE, _q(100.0))) == ["na"]

    def test_empty_query_array_returns_empty(self):
        out = phase_status_at_timestamps(_events(_TWO_PHASE_INTERVALS), PHASE, _q())
        assert len(out) == 0

    def test_phase_with_no_events_is_all_na(self):
        assert list(
            phase_status_at_timestamps(_events(_TWO_PHASE_INTERVALS), 7, _q(105.0, 150.0))
        ) == ["na", "na"]

    def test_marker_only_frame_is_all_na(self):
        assert _phase_at([(100.0, -1, -1), (200.0, -1, -1)], 150.0) == ["na"]


class TestPhaseStatusGapRule:
    """CLAUDE.md §5: no status may forward-fill across an event_code == -1
    hard reset.  These pin the segment guard the module docstring describes."""

    # A complete interval, a marker at 150, then a complete interval:
    _GAPPED = [
        (100.0, 1, PHASE), (110.0, 8, PHASE), (114.0, 9, PHASE), (116.0, 12, PHASE),
        (150.0, -1, -1),
        (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
    ]

    def test_no_stale_red_after_gap_before_next_event(self):
        # The roadmap's priority assertion: a frame after the gap but before
        # the next real event must read 'na', never the stale pre-gap 'R'.
        assert _phase_at(self._GAPPED, 160.0, 199.9) == ["na", "na"]

    def test_no_inferred_red_before_the_gap_either(self):
        # Between A's clearance end and the marker the next interval is in
        # another segment, so no confirmed next green exists: 'na'.
        assert _phase_at(self._GAPPED, 120.0, 149.9) == ["na", "na"]

    def test_intervals_around_the_gap_still_resolve_internally(self):
        assert _phase_at(self._GAPPED, 105.0, 112.0, 205.0, 212.0) == ["G", "Y", "G", "Y"]

    def test_interval_straddling_a_marker_is_dropped_entirely(self):
        # Marker lands mid-interval (between yellow and end-yellow): the
        # interval's endpoints are definitionally unknown, so no part of it
        # may be coloured.
        rows = [
            (100.0, 1, PHASE), (110.0, 8, PHASE),
            (112.0, -1, -1),
            (114.0, 9, PHASE), (116.0, 12, PHASE),
            (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
        ]
        assert _phase_at(rows, 105.0, 111.0, 115.0, 150.0) == ["na", "na", "na", "na"]

    def test_marker_tied_exactly_on_green_ts_marker_row_first(self):
        # Marker at exactly B's green onset (ties are real: gap markers are
        # inserted at prev_last_event_ts + 0.1 and re-sorted).  Marker row
        # physically precedes the green row: B lands in the post-gap
        # segment and resolves; the pre-gap red span is 'na'.
        rows = [
            (100.0, 1, PHASE), (110.0, 8, PHASE), (114.0, 9, PHASE), (116.0, 12, PHASE),
            (200.0, -1, -1),
            (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
        ]
        assert _phase_at(rows, 150.0, 200.0, 205.0) == ["na", "G", "G"]

    def test_marker_tied_exactly_on_green_ts_marker_row_last(self):
        # Same tie, opposite physical order: B's green sorts into the
        # pre-gap segment, its clearance into the post-gap segment, so B is
        # dropped.  Conservative ('na' during B's real green) — but no
        # frame ever shows a colour whose evidence crosses the marker.
        rows = [
            (100.0, 1, PHASE), (110.0, 8, PHASE), (114.0, 9, PHASE), (116.0, 12, PHASE),
            (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
            (200.0, -1, -1),
        ]
        assert _phase_at(rows, 150.0, 205.0, 212.0) == ["na", "na", "na"]

    def test_double_marker_with_empty_segment_between(self):
        rows = [
            (100.0, 1, PHASE), (110.0, 8, PHASE), (114.0, 9, PHASE), (116.0, 12, PHASE),
            (130.0, -1, -1), (170.0, -1, -1),
            (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
        ]
        assert _phase_at(rows, 120.0, 150.0, 190.0) == ["na", "na", "na"]


class TestPhaseRedClearance:
    """Red clearance reads 'R', via yellow_end_ts (see module docstring)."""

    # A serves red clearance: end-yellow 114, begin RC 114, end RC 116.
    _RC_SERVED = [
        (100.0, 1, PHASE), (110.0, 8, PHASE), (114.0, 9, PHASE),
        (114.0, 10, PHASE), (116.0, 11, PHASE),
        (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
    ]

    # Same phase, no Code 9 logged: Code 10 is the only end-of-yellow signal.
    _RC_SERVED_NO_END_YELLOW = [
        (100.0, 1, PHASE), (110.0, 8, PHASE),
        (114.0, 10, PHASE), (116.0, 11, PHASE),
        (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
    ]

    def test_red_clearance_period_reads_red(self):
        assert _phase_at(self._RC_SERVED, 114.5, 115.9) == ["R", "R"]

    def test_yellow_ends_exactly_at_end_yellow(self):
        # Code 9 is the boundary: the instant before is still 'Y'.
        assert _phase_at(self._RC_SERVED, 113.9, 114.0) == ["Y", "R"]

    def test_begin_red_clearance_ends_yellow_without_a_code_9(self):
        # No Code 9 at all — Code 10 has to carry the boundary on its own.
        assert _phase_at(
            self._RC_SERVED_NO_END_YELLOW, 112.0, 113.9, 114.0, 115.9
        ) == ["Y", "Y", "R", "R"]

    def test_yellow_and_post_rc_red_are_correct(self):
        # Everything outside the RC window was already right.
        assert _phase_at(self._RC_SERVED, 105.0, 112.0, 116.0, 150.0) == ["G", "Y", "R", "R"]

    def test_no_red_clearance_phase_is_unaffected(self):
        # 1/8/9/12 termination: red starts at Code 9 as expected.
        assert _phase_at(_TWO_PHASE_INTERVALS, 114.5, 115.9) == ["R", "R"]


class TestPhaseStatusQuirks:
    """Characterizations of degenerate event sequences (documented, not fixed)."""

    def test_characterization_missing_end_yellow_labels_yellow_as_red(self):
        # 1 -> 8 -> 12 with no Code 9: clear_end falls back to the yellow
        # onset, so the real yellow period reads 'R'.  Pinned: the clearance
        # end is genuinely unlogged, but the direction of the guess (red,
        # the safe/terminal state) is worth keeping deliberate.
        rows = [
            (100.0, 1, PHASE), (110.0, 8, PHASE), (116.0, 12, PHASE),
            (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
        ]
        assert _phase_at(rows, 105.0, 112.0, 150.0) == ["G", "R", "R"]

    def test_characterization_green_without_yellow_is_dropped(self):
        # 1 -> 12 directly (include_no_clearance=False in the video path):
        # the green never becomes an interval, so its real green reads 'na'.
        rows = [
            (100.0, 1, PHASE), (116.0, 12, PHASE),
            (200.0, 1, PHASE), (210.0, 8, PHASE), (214.0, 9, PHASE), (216.0, 12, PHASE),
        ]
        assert _phase_at(rows, 105.0, 150.0, 205.0) == ["na", "na", "G"]


class TestOverlapStatus:
    """Overlap state machine (Codes 61/63/64/65/66) and its lookup."""

    def test_boundary_matrix_matches_phase_semantics(self):
        assert _overlap_at(
            _TWO_OVERLAP_INTERVALS, 99.9, 100.0, 110.0, 150.0, 200.0, 500.0
        ) == ["na", "G", "Y", "R", "G", "na"]

    def test_begin_red_clearance_ends_yellow(self):
        # Code 64 flips Y -> R immediately — the behaviour the phase path
        # lacks (see TestPhaseRedClearance).
        assert _overlap_at(_TWO_OVERLAP_INTERVALS, 113.9, 114.0, 115.0) == ["Y", "R", "R"]

    def test_dark_reads_na_from_dark_ts_until_next_green(self):
        # 66 = Dark: no active output — renders 'na', not red.
        rows = [
            (100.0, 61, OVERLAP), (110.0, 63, OVERLAP), (114.0, 64, OVERLAP), (116.0, 66, OVERLAP),
            (200.0, 61, OVERLAP), (210.0, 63, OVERLAP), (216.0, 65, OVERLAP),
        ]
        # Red clearance (64 -> 66) still reads 'R'; at/after dark_ts -> 'na'.
        assert _overlap_at(rows, 115.0, 116.0, 150.0, 205.0) == ["R", "na", "na", "G"]

    def test_yellow_straight_to_dark_sets_both_boundaries(self):
        # 63 -> 66 with no 64/65: yellow ends and dark begins at the same
        # instant; nothing in between may read 'R'.
        rows = [
            (100.0, 61, OVERLAP), (110.0, 63, OVERLAP), (116.0, 66, OVERLAP),
            (200.0, 61, OVERLAP), (210.0, 63, OVERLAP), (216.0, 65, OVERLAP),
        ]
        assert _overlap_at(rows, 112.0, 115.9, 116.0, 150.0) == ["Y", "Y", "na", "na"]

    def test_characterization_green_to_off_without_yellow_is_dropped(self):
        # 61 -> 65 directly: no yellow logged, interval dropped, real green
        # reads 'na' (mirrors the phase-path characterization).
        rows = [
            (100.0, 61, OVERLAP), (116.0, 65, OVERLAP),
            (200.0, 61, OVERLAP), (210.0, 63, OVERLAP), (216.0, 65, OVERLAP),
        ]
        assert _overlap_at(rows, 105.0, 150.0, 205.0) == ["na", "na", "G"]

    def test_gap_marker_between_overlap_intervals_blocks_red(self):
        rows = [
            (100.0, 61, OVERLAP), (110.0, 63, OVERLAP), (116.0, 65, OVERLAP),
            (150.0, -1, -1),
            (200.0, 61, OVERLAP), (210.0, 63, OVERLAP), (216.0, 65, OVERLAP),
        ]
        assert _overlap_at(rows, 120.0, 160.0, 199.9, 205.0) == ["na", "na", "na", "G"]

    def test_other_overlap_numbers_do_not_bleed_through(self):
        # Same codes, parameter 2 — overlap 1 must see nothing.
        rows = [(off, code, 2) for off, code, _ in _TWO_OVERLAP_INTERVALS]
        assert _overlap_at(rows, 105.0, 150.0) == ["na", "na"]

    def test_mid_stream_start_ignores_orphan_yellow(self):
        # Window opens mid-cycle: a yellow with no seen green must not
        # fabricate an interval.
        rows = [
            (110.0, 63, OVERLAP), (116.0, 65, OVERLAP),
            (200.0, 61, OVERLAP), (210.0, 63, OVERLAP), (216.0, 65, OVERLAP),
        ]
        assert _overlap_at(rows, 112.0, 150.0, 205.0) == ["na", "na", "G"]


class TestDetectorStatus:
    """detector_status_at_timestamps reuses analysis/detectors helpers."""

    _DET_ROWS = [
        (100.0, 82, DET), (120.0, 81, DET),   # ON 100-120
        (140.0, 82, DET),                     # ON, still open when...
        (150.0, -1, -1),                      # ...the hard reset closes it
        (200.0, 82, DET), (210.0, 81, DET),   # ON 200-210
    ]

    def test_on_off_and_boundaries(self):
        df = _events(self._DET_ROWS)
        out = detector_status_at_timestamps(df, DET, _q(90.0, 105.0, 130.0, 205.0, 215.0))
        assert list(out) == [False, True, False, True, False]

    def test_gap_marker_closes_open_interval(self):
        # ON at 140 with no OFF: the -1 reset closes it at the marker, so
        # the pre-gap ON holds right up to (not through) the reset.
        df = _events(self._DET_ROWS)
        out = detector_status_at_timestamps(df, DET, _q(145.0, 149.9, 150.0, 175.0))
        assert list(out) == [True, True, False, False]
        # Note: the boolean API cannot express "unknown" — a post-gap frame
        # before the next real event reads False (off), which at least never
        # shows the stale pre-gap ON state.

    def test_empty_events_is_all_off(self):
        df = _events([]).iloc[0:0]
        out = detector_status_at_timestamps(df, DET, _q(100.0))
        assert list(out) == [False]


class TestFirstPhaseTransitionAfter:

    _ROWS = [
        (100.0, 8, PHASE), (104.0, 9, PHASE),
        (150.0, 8, PHASE), (154.0, 9, PHASE),
        (150.0, 8, 3),  # other phase — must be ignored
    ]

    def test_after_ts_is_inclusive(self):
        assert first_phase_transition_after(_events(self._ROWS), PHASE, T0 + 100.0) == (
            "green_to_yellow", T0 + 100.0,
        )

    def test_auto_select_picks_earliest_edge(self):
        assert first_phase_transition_after(_events(self._ROWS), PHASE, T0 + 101.0) == (
            "yellow_to_red", T0 + 104.0,
        )

    def test_restricted_transition_skips_the_other_edge(self):
        assert first_phase_transition_after(
            _events(self._ROWS), PHASE, T0 + 101.0, "green_to_yellow"
        ) == ("green_to_yellow", T0 + 150.0)

    def test_no_match_returns_none(self):
        assert first_phase_transition_after(_events(self._ROWS), PHASE, T0 + 200.0) is None

    def test_other_phases_events_are_ignored(self):
        assert first_phase_transition_after(_events(self._ROWS), 3, T0 + 200.0) is None
        assert first_phase_transition_after(_events(self._ROWS), 3, T0 + 100.0) == (
            "green_to_yellow", T0 + 150.0,
        )

    def test_unrecognised_transition_raises(self):
        with pytest.raises(ValueError, match="Unrecognised transition"):
            first_phase_transition_after(_events(self._ROWS), PHASE, T0, "red_to_green")
