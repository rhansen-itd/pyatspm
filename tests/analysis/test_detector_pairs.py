"""Regression tests for multi-pair detector comparison.

Targets: src/atspm/analysis/detectors.py (analyze_discrepancies) and
src/atspm/plotting/detectors.py (plot_detector_comparison).

Pins the pair-keyed contract: a detector shared by several pairs must have
each anomaly drawn in the band of the pair that produced it, disagreements
that flip direction split into separate rows, and a detector silent for the
whole window yields no anomalies.
"""

import pandas as pd

from atspm.analysis.detectors import analyze_discrepancies
from atspm.data.manager import _parse_detector_pairs
from atspm.plotting.detectors import plot_detector_comparison

# Det 42 is shared by pairs (42,3) and (42,2).
EVENTS = pd.DataFrame(
    [
        (100, 82, 42), (110, 81, 42), (100.2, 82, 3), (110.1, 81, 3),
        (200, 82, 42), (210, 81, 42),
        (300, 82, 2), (300.3, 81, 2),
        (400, 82, 42), (405, 81, 42), (405, 82, 3), (410, 81, 3),
    ],
    columns=["timestamp", "event_code", "parameter"],
)
PAIRS = [
    {"phase": 2, "det_a": 42, "det_b": 3},
    {"phase": 2, "det_a": 42, "det_b": 2},
]


def test_direction_flip_splits_disagreement():
    an = analyze_discrepancies(EVENTS, PAIRS[:1], 2.0)
    flip = an[an["start_timestamp"] >= 400]
    assert list(flip["on_det_id"]) == [42, 3]
    assert list(flip["end_timestamp"]) == [405.0, 410.0]


def test_duplicate_pair_analysed_once():
    once = analyze_discrepancies(EVENTS, PAIRS, 2.0)
    twice = analyze_discrepancies(EVENTS, PAIRS + PAIRS[:1], 2.0)
    pd.testing.assert_frame_equal(once, twice)


def test_silent_detector_yields_no_anomalies():
    only_a = EVENTS[EVENTS["parameter"] == 42]
    assert analyze_discrepancies(only_a, PAIRS[:1], 2.0).empty
    fig = plot_detector_comparison(only_a, pd.DataFrame(), PAIRS[:1])
    assert "Ph2 Det 3 (no data)" in fig.layout.yaxis.ticktext


def test_shared_detector_anomalies_stay_in_own_band():
    an = analyze_discrepancies(EVENTS, PAIRS, 2.0)
    fig = plot_detector_comparison(EVENTS, an, PAIRS, {"timezone": "UTC"})

    # First pair on top: (42,3) lanes at 5/4, (42,2) lanes at 2/1.
    assert list(fig.layout.yaxis.tickvals) == [5.0, 4.0, 2.0, 1.0]

    ext = an[an["anomaly_type"] == "extended_disagreement"]
    # Rectangles are filled traces: 5 outline corners + a None break each.
    rects = [
        (min(t.y[i:i + 5]), max(t.y[i:i + 5]))
        for t in fig.data if t.fill == "toself"
        for i in range(0, len(t.y), 6)
    ]
    top = [r for r in rects if 3.0 < r[0] and r[1] < 6.0]
    bottom = [r for r in rects if 0.0 < r[0] and r[1] < 3.0]
    assert len(top) == (ext["det_b_id"] == 3).sum()
    assert len(bottom) == (ext["det_b_id"] == 2).sum()
    assert len(top) + len(bottom) == len(rects) == len(ext)

    pulses = [t for t in fig.data if (t.name or "").startswith("Isolated Pulse")]
    assert len(pulses) == 1 and list(pulses[0].y) == [1.0]


def test_parse_pairs_accepts_flat_and_drops_repeats():
    assert _parse_detector_pairs({"Det_P2_Pairs": "[42,3]"}) == [
        {"phase": 2, "det_a": 42, "det_b": 3}
    ]
    parsed = _parse_detector_pairs({"Det_P2_Pairs": "[[42,3],[42,3],[42,2]]"})
    assert parsed == PAIRS


def test_window_keeps_overlapping_anomalies_only():
    an = analyze_discrepancies(EVENTS, PAIRS[:1], 2.0, window=(205.0, 402.0))
    # (200-210) and (400-405) overlap the window; the later flip half does not.
    assert list(an["start_timestamp"]) == [200.0, 400.0]


def test_actuation_crossing_window_edge_is_drawn_whole():
    # Both detectors ON from before the window start; padded events keep
    # the agreement intact instead of leaving a one-sided fragment.
    ev = pd.DataFrame(
        [(90, 82, 42), (90.5, 82, 3), (120, 81, 42), (120.2, 81, 3)],
        columns=["timestamp", "event_code", "parameter"],
    )
    window = (100.0, 200.0)
    an = analyze_discrepancies(ev, PAIRS[:1], 2.0, window=window)
    assert an.empty
    fig = plot_detector_comparison(ev, an, PAIRS[:1], {"timezone": "UTC"}, window=window)
    lane_a = next(t for t in fig.data if t.name == "Det A")
    # x is local wall-clock ms; UTC here, so ms = epoch * 1000.
    assert lane_a.x[0] == 90_000
    assert fig.layout.xaxis.range[0] == 100_000
    assert "(no data)" not in " ".join(fig.layout.yaxis.ticktext)


def test_gap_markers_drawn_inside_window():
    ev = pd.concat([EVENTS, pd.DataFrame(
        [(250.0, -1, 0), (900.0, -1, 0)], columns=EVENTS.columns,
    )])
    fig = plot_detector_comparison(ev, pd.DataFrame(), PAIRS, {"timezone": "UTC"},
                                   window=(0.0, 500.0))
    gap = next(t for t in fig.data if t.name == "Hard reset")
    assert [x for x in gap.x if x == x] == [250_000, 250_000]   # x == x drops NaN


def test_pair_summary_percentages():
    an = analyze_discrepancies(EVENTS, PAIRS, 2.0)
    fig = plot_detector_comparison(EVENTS, an, PAIRS, {"timezone": "UTC"},
                                   window=(0.0, 1000.0))
    texts = [a.text for a in fig.layout.annotations]
    # (42,3): 10 s + 5 s + 5 s of disagreement over 1000 s, no pulses.
    assert texts[0] == "2.0% disagree<br>0 pulses"
    # (42,2): 10 s + 10 s + 5 s, one pulse on det 2.
    assert texts[1] == "2.5% disagree<br>1 pulse"


def test_pair_spans_merge_across_config_change():
    from atspm.data.detectors import _pair_spans

    configs = [
        {"_epoch_start": 0.0, "_epoch_end": 50.0,
         "detector_pairs": [{"phase": 2, "det_a": 42, "det_b": 3}]},
        {"_epoch_start": 50.0, "_epoch_end": 100.0,
         "detector_pairs": [{"phase": 2, "det_a": 42, "det_b": 3},
                            {"phase": 1, "det_a": 49, "det_b": 20}]},
    ]
    spans = _pair_spans(configs, None)
    assert list(spans) == [(1, 49, 20), (2, 42, 3)]          # ordered by phase
    assert spans[(2, 42, 3)] == [(0.0, 50.0 + 50.0)]         # merged, not split
    assert spans[(1, 49, 20)] == [(50.0, 100.0)]
    assert list(_pair_spans(configs, {1})) == [(1, 49, 20)]
