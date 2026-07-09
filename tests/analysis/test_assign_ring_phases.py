"""Regression tests for ring/phase assignment (Functional Core).

Target: src/atspm/analysis/cycles.py — assign_ring_phases.
Written BEFORE the Session D structural refactor to pin current behavior.

Characterized contract
----------------------
Input:  cycles_df with at least [cycle_start] (float epoch or datetime64),
        events_df with [timestamp, event_code, parameter], config dict that
        may hold RB_R1/RB_R2 (or lowercase RB_r1/RB_r2) ring definitions.
Output: a copy of cycles_df with r1_phases / r2_phases string columns
        appended — comma-joined phase numbers in chronological order
        (e.g. "2,6"), literal "None" when a ring has no phases in a cycle.
        Original row order, index, and extra columns are preserved.
Join:   code 1 (green start) events attach to cycles via
        pd.merge_asof(greens, cycles, direction='backward') with NO
        tolerance — each green goes to the most recent cycle_start at or
        before its timestamp; greens before the first cycle are dropped.

Gap-rule note (CLAUDE.md §5): gap marker rows themselves (event_code == -1)
can never become phases — the event_code == 1 filter excludes them.  The
backward join is additionally segment-constrained: segments increment at
each -1 marker, and a green only attaches to a cycle_start within its own
segment.  A green landing AFTER a hard reset yet BEFORE the first post-gap
cycle_start therefore attaches to no cycle at all — sequential pairing
never crosses the reset, no matter how close the marker sits to either
side (the rule is purely order-based; there is no time tolerance, so a
short mid-cycle blip and an hour-long outage bound the join identically).
Timestamp ties resolve independently of physical row order: a green or
cycle_start exactly at a marker timestamp belongs to the post-gap segment.
"""

import pandas as pd
import pytest

from atspm.analysis.cycles import assign_ring_phases, _merge_coordination_plan

T0 = 1_609_459_200.0  # 2021-01-01 00:00:00 UTC


def _cycles(starts, **extra) -> pd.DataFrame:
    return pd.DataFrame({'cycle_start': list(starts), **extra})


def _events(rows) -> pd.DataFrame:
    """rows: iterable of (timestamp, event_code, parameter)."""
    ts, codes, params = zip(*rows) if rows else ((), (), ())
    return pd.DataFrame({
        'timestamp': list(ts),
        'event_code': list(codes),
        'parameter': list(params),
    })


class TestDegenerateInputs:

    def test_empty_events_yields_none_strings(self):
        out = assign_ring_phases(_cycles([T0, T0 + 90.0]), _events([]), {})
        assert list(out['r1_phases']) == ['None', 'None']
        assert list(out['r2_phases']) == ['None', 'None']

    def test_empty_cycles_returns_empty_with_columns(self):
        out = assign_ring_phases(
            _cycles([]), _events([(T0 + 1, 1, 2)]), {}
        )
        assert out.empty
        assert 'r1_phases' in out.columns
        assert 'r2_phases' in out.columns

    def test_both_empty(self):
        out = assign_ring_phases(_cycles([]), _events([]), {})
        assert out.empty
        assert {'r1_phases', 'r2_phases'} <= set(out.columns)

    def test_events_with_no_code1_rows_yields_none_strings(self):
        events = _events([(T0 + 1, 31, 1), (T0 + 2, 131, 5)])
        out = assign_ring_phases(_cycles([T0]), events, {})
        assert list(out['r1_phases']) == ['None']
        assert list(out['r2_phases']) == ['None']

    def test_all_greens_before_first_cycle_are_dropped(self):
        # merge_asof direction='backward' leaves greens preceding the first
        # cycle_start unmatched; they must not be attributed anywhere.
        out = assign_ring_phases(
            _cycles([T0 + 100.0]), _events([(T0 + 5, 1, 2)]), {}
        )
        assert list(out['r1_phases']) == ['None']
        assert list(out['r2_phases']) == ['None']

    def test_input_frames_are_not_mutated(self):
        cycles = _cycles([T0])
        events = _events([(T0 + 1, 1, 2)])
        cycles_copy = cycles.copy()
        events_copy = events.copy()
        assign_ring_phases(cycles, events, {})
        pd.testing.assert_frame_equal(cycles, cycles_copy)
        pd.testing.assert_frame_equal(events, events_copy)


class TestDefaultRingMembership:

    def test_basic_two_ring_cycle(self):
        # Defaults: R1 = [1,2,3,4,9,10,11,12], R2 = [5,6,7,8,13,14,15,16].
        events = _events([(T0 + 5, 1, 2), (T0 + 6, 1, 6)])
        out = assign_ring_phases(_cycles([T0]), events, {})
        assert list(out['r1_phases']) == ['2']
        assert list(out['r2_phases']) == ['6']

    def test_ring1_only_cycle_gets_none_for_ring2(self):
        events = _events([(T0 + 5, 1, 2), (T0 + 10, 1, 4)])
        out = assign_ring_phases(_cycles([T0]), events, {})
        assert list(out['r1_phases']) == ['2,4']
        assert list(out['r2_phases']) == ['None']

    def test_ring2_only_cycle_gets_none_for_ring1(self):
        events = _events([(T0 + 5, 1, 6), (T0 + 10, 1, 8)])
        out = assign_ring_phases(_cycles([T0]), events, {})
        assert list(out['r1_phases']) == ['None']
        assert list(out['r2_phases']) == ['6,8']

    def test_phases_are_chronological_and_repeats_kept(self):
        # A phase served twice in one cycle appears twice, in time order.
        events = _events([
            (T0 + 5, 1, 4), (T0 + 10, 1, 2), (T0 + 20, 1, 4),
        ])
        out = assign_ring_phases(_cycles([T0]), events, {})
        assert list(out['r1_phases']) == ['4,2,4']

    def test_green_exactly_at_cycle_start_belongs_to_that_cycle(self):
        events = _events([(T0 + 90.0, 1, 2)])
        out = assign_ring_phases(_cycles([T0, T0 + 90.0]), events, {})
        assert list(out['r1_phases']) == ['None', '2']

    def test_multiple_cycles_partition_greens(self):
        events = _events([
            (T0 + 5, 1, 2), (T0 + 40, 1, 6),
            (T0 + 95, 1, 4), (T0 + 130, 1, 8),
        ])
        out = assign_ring_phases(_cycles([T0, T0 + 90.0]), events, {})
        assert list(out['r1_phases']) == ['2', '4']
        assert list(out['r2_phases']) == ['6', '8']


class TestConfigRingMembership:

    def test_rb_config_overrides_defaults(self):
        # Phase 6 belongs to R1 here, contrary to the defaults.
        config = {'RB_R1': '6|4', 'RB_R2': None}
        events = _events([(T0 + 5, 1, 6), (T0 + 10, 1, 4)])
        out = assign_ring_phases(_cycles([T0]), events, config)
        assert list(out['r1_phases']) == ['6,4']
        # R2 backfills with defaults minus R1 members (6 excluded).
        assert list(out['r2_phases']) == ['None']

    def test_lowercase_rb_keys_are_honored(self):
        config = {'RB_r1': '1,2|3,4', 'RB_r2': '5,6|7,8'}
        events = _events([(T0 + 5, 1, 3), (T0 + 6, 1, 7)])
        out = assign_ring_phases(_cycles([T0]), events, config)
        assert list(out['r1_phases']) == ['3']
        assert list(out['r2_phases']) == ['7']

    def test_phase_outside_both_rings_is_dropped(self):
        config = {'RB_R1': '1,2', 'RB_R2': '5,6'}
        events = _events([(T0 + 5, 1, 2), (T0 + 6, 1, 9), (T0 + 7, 1, 6)])
        out = assign_ring_phases(_cycles([T0]), events, config)
        assert list(out['r1_phases']) == ['2']
        assert list(out['r2_phases']) == ['6']

    def test_overlapping_ring_membership_appears_in_both(self):
        # A phase listed in both rings shows up in both strings.
        config = {'RB_R1': '2,3', 'RB_R2': '2,6'}
        events = _events([(T0 + 5, 1, 2), (T0 + 6, 1, 6)])
        out = assign_ring_phases(_cycles([T0]), events, config)
        assert list(out['r1_phases']) == ['2']
        assert list(out['r2_phases']) == ['2,6']


class TestFrameShapePreservation:

    def test_unsorted_cycles_keep_row_order_index_and_extra_columns(self):
        cycles = pd.DataFrame(
            {'cycle_start': [T0 + 100.0, T0], 'coord_plan': [7, 3]},
            index=[10, 20],
        )
        events = _events([(T0 + 5, 1, 2), (T0 + 105, 1, 6)])
        out = assign_ring_phases(cycles, events, {})
        assert list(out.index) == [10, 20]
        assert list(out['coord_plan']) == [7, 3]
        assert list(out['r1_phases']) == ['None', '2']
        assert list(out['r2_phases']) == ['6', 'None']

    def test_datetime_cycles_with_numeric_event_timestamps(self):
        cycles = pd.DataFrame(
            {'cycle_start': pd.to_datetime([T0], unit='s', utc=True)}
        )
        events = _events([(T0 + 5, 1, 2), (T0 + 6, 1, 6)])
        out = assign_ring_phases(cycles, events, {})
        assert list(out['r1_phases']) == ['2']
        assert list(out['r2_phases']) == ['6']


class TestGapMarkers:

    def test_gap_marker_row_never_becomes_a_phase(self):
        # The -1 row carries parameter 2; it must not leak into r1_phases.
        events = _events([(T0 + 5, 1, 2), (T0 + 50, -1, 2)])
        out = assign_ring_phases(_cycles([T0]), events, {})
        assert list(out['r1_phases']) == ['2']
        assert list(out['r2_phases']) == ['None']

    def test_gap_between_cycles_with_no_stranded_greens_is_clean(self):
        # Marker sits between the last pre-gap green and the first post-gap
        # cycle_start; every green lies inside a real cycle, so attribution
        # is unambiguous and correct.
        events = _events([
            (T0 + 5, 1, 2),
            (T0 + 200, -1, 0),
            (T0 + 3805, 1, 8),
        ])
        out = assign_ring_phases(_cycles([T0, T0 + 3800.0]), events, {})
        assert list(out['r1_phases']) == ['2', 'None']
        assert list(out['r2_phases']) == ['None', '8']

    def test_stranded_green_dropped_while_postgap_cycle_keeps_its_greens(self):
        # Session F guard (was the pre-fix characterization): the stranded
        # green (post-gap, pre-first-new-cycle) attaches nowhere, while
        # greens inside the post-gap cycle attach normally.
        events = _events([
            (T0 + 10, 1, 2),
            (T0 + 200, -1, 0),
            (T0 + 3700, 1, 4),   # stranded: post-gap, pre-first-new-cycle
            (T0 + 3810, 1, 8),
        ])
        out = assign_ring_phases(_cycles([T0, T0 + 3800.0]), events, {})
        assert list(out['r1_phases']) == ['2', 'None']
        assert list(out['r2_phases']) == ['None', '8']

    def test_stranded_postgap_greens_are_not_paired_across_the_reset(self):
        events = _events([
            (T0 + 10, 1, 2),
            (T0 + 200, -1, 0),
            (T0 + 3700, 1, 4),
        ])
        out = assign_ring_phases(_cycles([T0, T0 + 3800.0]), events, {})
        # Phase 4 crossed a hard reset relative to the pre-gap cycle and
        # precedes any post-gap cycle — attribute nowhere (CLAUDE.md §5).
        assert list(out['r1_phases']) == ['2', 'None']
        assert list(out['r2_phases']) == ['None', 'None']

    def test_midcycle_marker_stops_attribution_at_the_reset(self):
        # Session F behavior change: a green after a marker but before the
        # next cycle_start attaches nowhere, even when the marker is a
        # short mid-cycle blip.  This input is order-isomorphic to the
        # stranded-green case above (c0 < green < marker < green < c1 in
        # both; they differ only in second counts), so no order-based rule
        # can keep this green while dropping the stranded one — and the §5
        # gap rule has no time tolerance.  Pre-marker greens keep their
        # cycle; the post-marker green is conservatively unattributed.
        events = _events([
            (T0 + 10, 1, 2),
            (T0 + 30, -1, 0),
            (T0 + 60, 1, 6),
        ])
        out = assign_ring_phases(_cycles([T0, T0 + 90.0]), events, {})
        assert list(out['r1_phases']) == ['2', 'None']
        assert list(out['r2_phases']) == ['None', 'None']

    def test_greens_after_the_first_postgap_cycle_resume_normally(self):
        # Only the stranded window (marker -> next cycle_start) is dropped;
        # attribution inside post-gap cycles is unaffected, including for
        # a second gap later in the frame.
        events = _events([
            (T0 + 10, 1, 2),
            (T0 + 200, -1, 0),
            (T0 + 3805, 1, 4), (T0 + 3840, 1, 6),
            (T0 + 3900, -1, 0),
            (T0 + 3950, 1, 8),   # stranded again: before next cycle_start
            (T0 + 8010, 1, 1),
        ])
        cycles = _cycles([T0, T0 + 3800.0, T0 + 8000.0])
        out = assign_ring_phases(cycles, events, {})
        assert list(out['r1_phases']) == ['2', '4', '1']
        assert list(out['r2_phases']) == ['None', '6', 'None']

    @pytest.mark.parametrize('rows', [
        pytest.param(
            [(T0 + 10, 1, 2), (T0 + 50, -1, 0), (T0 + 50, 1, 6)],
            id='marker-row-first',
        ),
        pytest.param(
            [(T0 + 10, 1, 2), (T0 + 50, 1, 6), (T0 + 50, -1, 0)],
            id='green-row-first',
        ),
    ])
    def test_green_tied_with_marker_timestamp_is_postgap_in_both_row_orders(
        self, rows
    ):
        # A green at exactly the marker's timestamp belongs to the post-gap
        # segment regardless of physical row order (segments come from
        # searchsorted on marker timestamps, not row-order cumsum), so it
        # never attaches to the pre-gap cycle.
        out = assign_ring_phases(_cycles([T0, T0 + 90.0]), _events(rows), {})
        assert list(out['r1_phases']) == ['2', 'None']
        assert list(out['r2_phases']) == ['None', 'None']

    def test_datetime_cycles_with_marker_still_stop_at_the_reset(self):
        # Exercises the dtype-normalization branch: datetime64 cycles with
        # float event timestamps convert marker timestamps too, so the
        # segment constraint holds in either time base.
        cycles = pd.DataFrame({
            'cycle_start': pd.to_datetime([T0, T0 + 3800.0], unit='s', utc=True)
        })
        events = _events([
            (T0 + 10, 1, 2),
            (T0 + 200, -1, 0),
            (T0 + 3700, 1, 4),
            (T0 + 3810, 1, 8),
        ])
        out = assign_ring_phases(cycles, events, {})
        assert list(out['r1_phases']) == ['2', 'None']
        assert list(out['r2_phases']) == ['None', '8']

    def test_cycle_start_tied_with_marker_timestamp_owns_postgap_greens(self):
        # A cycle_start at exactly the marker's timestamp is post-gap, so
        # greens at/after it attach to it — resumption is not delayed.
        events = _events([
            (T0 + 10, 1, 2),
            (T0 + 90, -1, 0),
            (T0 + 95, 1, 6),
        ])
        out = assign_ring_phases(_cycles([T0, T0 + 90.0]), events, {})
        assert list(out['r1_phases']) == ['2', 'None']
        assert list(out['r2_phases']) == ['None', '6']


class TestCoordPlanBoundaries:
    """Guards for _merge_coordination_plan around the Code 131/132 bug fix
    (commit e2b0f82) — the area adjacent to this refactor."""

    def test_code131_backward_fills_onto_cycles(self):
        events = _events([(T0 - 10, 131, 3)])
        out = _merge_coordination_plan(_cycles([T0, T0 + 90.0]), events)
        assert list(out['coord_plan']) == [3.0, 3.0]

    def test_code132_is_ignored_for_coord_plan(self):
        # 132's parameter is a cycle length in seconds, not a plan ID.
        events = _events([(T0 - 10, 131, 3), (T0 - 10, 132, 120)])
        out = _merge_coordination_plan(_cycles([T0]), events)
        assert list(out['coord_plan']) == [3.0]

    def test_plan_change_boundary_splits_cycles(self):
        # Plan 3 before T0+90, plan 5 at exactly T0+90: the boundary cycle
        # takes the plan active AT its start (merge_asof backward includes
        # exact matches).
        events = _events([(T0 - 10, 131, 3), (T0 + 90.0, 131, 5)])
        out = _merge_coordination_plan(_cycles([T0, T0 + 90.0, T0 + 180.0]), events)
        assert list(out['coord_plan']) == [3.0, 5.0, 5.0]

    def test_no_coord_events_defaults_to_zero(self):
        out = _merge_coordination_plan(_cycles([T0]), _events([]))
        assert list(out['coord_plan']) == [0.0]
