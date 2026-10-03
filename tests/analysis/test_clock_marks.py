# Tests for the clock-mark decoder (functional core).
#
# Golden half: the 2026-09-30 bench files (10.70.10.51, firmware 03.02.60)
# go through the real ingestion path, and the head unit's send log
# (bench.jsonl) is the oracle. Synthetic half: the edge cases the bench
# didn't produce, built as event frames in label time.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.clock_marks import (
    MarkerPeds,
    decode_clock_marks,
    drop_marker_events,
    marker_peds_from_config,
    pair_marker_pulses,
    send_log_pulses,
)
from atspm.analysis.decoders import CLOCK_STEP_FENCE_PARAM, COMMS_GAP_PARAM
from atspm.data.ingestion import IngestionEngine
from atspm.data.manager import DatabaseManager

BENCH = Path(__file__).resolve().parents[1] / "fixtures" / "clock_marks_bench_2026_09_30"
PEDS = MarkerPeds(behind=15, ahead=16, set=14)


# ---------------------------------------------------------------------------
# Bench (golden)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bench_events(tmp_path_factory) -> pd.DataFrame:
    root = tmp_path_factory.mktemp("bench")
    raw = root / "raw_data"
    raw.mkdir()
    for f in sorted(BENCH.glob("*.datZ")):
        (raw / f.name).write_bytes(f.read_bytes())
    db = root / "bench.db"
    with DatabaseManager(db) as m:
        m.init_db()
    IngestionEngine(db, raw, timezone="US/Mountain").run()
    with DatabaseManager(db) as m:
        return pd.read_sql(
            "SELECT timestamp, event_code, parameter FROM events", m.conn
        )


@pytest.fixture(scope="module")
def bench_log() -> pd.DataFrame:
    with open(BENCH / "bench.jsonl") as fh:
        return send_log_pulses(json.loads(line) for line in fh)


class TestBenchSets:

    def test_all_four_sets_recover_the_exact_shift(self, bench_events, bench_log):
        _, sets = decode_clock_marks(bench_events, PEDS, bench_log)
        assert sets["shift"].tolist() == [-2, 3, 40, -40]
        assert sets["shift"].tolist() == sets["shift_host"].tolist()
        assert (sets["status"] == "ok").all()

    def test_periods_match_what_was_sent(self, bench_events, bench_log):
        _, sets = decode_clock_marks(bench_events, PEDS, bench_log)
        assert sets["period"].tolist() == [10, 20, 20, 50]

    def test_log_alone_decodes_the_unsaturated_sets_and_flags_the_rest(
        self, bench_events
    ):
        _, sets = decode_clock_marks(bench_events, PEDS)
        assert sets["shift"].iloc[:2].tolist() == [-2, 3]
        assert sets["status"].tolist() == ["ok", "ok", "pre_saturated", "pre_saturated"]
        assert sets["shift"].iloc[2:].isna().all()

    def test_step_window_holds_every_edit(self, bench_events, bench_log):
        # ENTER epochs are host time; the step window is pre-step label
        # time, so map each edit through its run's pre-set drift.
        _, sets = decode_clock_marks(bench_events, PEDS, bench_log)
        runs = [json.loads(l) for l in open(BENCH / "bench.jsonl")]
        runs = [r for r in runs if r["mode"] == "set"]
        for (_, row), run in zip(sets.iterrows(), runs):
            for edit in run["edits"]:
                label = edit["enter_epoch"] + run["before"]["drift"]
                assert row["step_lo"] <= label <= row["step_hi"]

    def test_the_minus_40_step_is_fenced_as_a_clock_step(self, bench_events):
        markers = bench_events.loc[bench_events["event_code"] == -1, "parameter"]
        assert markers.tolist() == [CLOCK_STEP_FENCE_PARAM]


class TestBenchDrift:

    def test_every_sample_brackets_the_host_measurement(self, bench_events, bench_log):
        drift, _ = decode_clock_marks(bench_events, PEDS, bench_log)
        assert len(drift) == 9  # 13 pulses sent in these files, 4 of them brackets
        clean = drift.loc[~drift["saturated"]]
        assert clean["drift_host"].notna().all()
        assert ((clean["drift_lo"] <= clean["drift_host"])
                & (clean["drift_host"] <= clean["drift_hi"])).all()
        assert (clean["drift"] - clean["drift_host"]).abs().max() < 0.2

    def test_saturated_pulses_take_the_send_log_value(self, bench_events, bench_log):
        drift, _ = decode_clock_marks(bench_events, PEDS, bench_log)
        sat = drift.loc[drift["saturated"]]
        assert sat["status"].tolist() == ["send_log", "send_log"]
        assert sat["drift"].tolist() == pytest.approx([-39.633, 40.291])

    def test_saturated_pulses_without_the_log_are_bounds_only(self, bench_events):
        drift, _ = decode_clock_marks(bench_events, PEDS)
        sat = drift.loc[drift["saturated"]]
        assert sat["status"].tolist() == ["saturated", "saturated"]
        assert sat["drift"].isna().all()
        behind, ahead = sat.iloc[0], sat.iloc[1]
        assert behind["drift_lo"] == -np.inf and behind["drift_hi"] <= -29.6
        assert ahead["drift_hi"] == np.inf and ahead["drift_lo"] >= 29.6

    def test_roles(self, bench_events):
        drift, _ = decode_clock_marks(bench_events, PEDS)
        assert drift["role"].tolist() == ["check"] + ["pre_set", "residual"] * 4


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------

T0 = 1_800_000_000.0


def _frame(rows):
    return pd.DataFrame(rows, columns=["timestamp", "event_code", "parameter"])


def _pulse(on, width, ped):
    return [(on, 90, ped), (on, 45, ped), (on + width, 89, ped)]


def _set_run(t, drift_pre, shift, period, residual=0.2, with_residual=True):
    """One daily-set run in label time: pre pulse, bracket, residual pulse.

    The bracket logs ``period + shift`` minus the 0.1 s mean shortfall, and
    the post-step labels move by ``shift``.
    """
    ped_pre = PEDS.ahead if drift_pre > 0 else PEDS.behind
    w_pre = round(abs(drift_pre) - 0.15, 1)
    rows = _pulse(t, w_pre, ped_pre)
    b_on = round(t + w_pre + 0.1, 1)
    b_off = round(b_on + period + shift - 0.1, 1)
    rows += _pulse(b_on, b_off - b_on, PEDS.set)
    if with_residual:
        rows += _pulse(round(b_off + 0.1, 1), residual, PEDS.ahead)
    return rows, b_on, b_off


class TestSyntheticSets:

    def test_small_backward_set_crosses_its_own_fence(self):
        # Busy controller, -2 s set: the fence lands inside the bracket in
        # sorted order (first edit 1.4 s after ON, labels step back 2 s).
        rows, b_on, _ = _set_run(T0, drift_pre=2.1, shift=-2, period=10)
        rows += [(b_on + 3.0 - 2.0 - 0.05, -1, CLOCK_STEP_FENCE_PARAM)]
        _, sets = decode_clock_marks(_frame(rows), PEDS)
        assert sets["shift"].tolist() == [-2]
        assert sets["status"].tolist() == ["ok"]

    def test_bracket_split_by_a_comms_gap_is_flagged_not_guessed(self):
        rows, b_on, _ = _set_run(T0, drift_pre=2.1, shift=-2, period=10)
        rows += [(b_on + 4.0, -1, COMMS_GAP_PARAM)]
        _, sets = decode_clock_marks(_frame(rows), PEDS)
        assert sets["status"].tolist() == ["unpaired_on", "unpaired_off"]
        assert sets["shift"].isna().all()

    def test_drift_pulse_split_by_a_comms_gap_is_flagged(self):
        rows = _pulse(T0, 3.0, PEDS.behind) + [(T0 + 1.0, -1, COMMS_GAP_PARAM)]
        drift, _ = decode_clock_marks(_frame(rows), PEDS)
        assert drift["status"].tolist() == ["unpaired_on", "unpaired_off"]
        assert drift["drift"].isna().all()

    def test_missing_post_set_pulse_still_decodes(self):
        rows, _, _ = _set_run(T0, drift_pre=-4.3, shift=4, period=10, with_residual=False)
        drift, sets = decode_clock_marks(_frame(rows), PEDS)
        assert sets["shift"].tolist() == [4]
        assert drift["role"].tolist() == ["pre_set"]

    def test_missing_pre_set_pulse(self):
        rows, _, _ = _set_run(T0, drift_pre=-4.3, shift=4, period=10)
        rows = [r for r in rows if r[2] != PEDS.behind]
        _, sets = decode_clock_marks(_frame(rows), PEDS)
        assert sets["status"].tolist() == ["no_pre_pulse"]
        assert sets["shift"].isna().all()

    def test_missing_pre_set_pulse_takes_the_send_log_shift(self):
        rows, b_on, _ = _set_run(T0, drift_pre=-4.3, shift=4, period=10)
        rows = [r for r in rows if r[2] != PEDS.behind]
        log = pd.DataFrame([{"ped": PEDS.set, "role": "set", "on_label": b_on + 0.3,
                             "drift_host": np.nan, "shift_host": 4.0}])
        _, sets = decode_clock_marks(_frame(rows), PEDS, log)
        assert sets["status"].tolist() == ["send_log"]
        assert sets["shift"].tolist() == [4]

    def test_send_log_contradicting_a_decoded_set_is_flagged(self):
        rows, b_on, _ = _set_run(T0, drift_pre=-4.3, shift=4, period=10)
        log = pd.DataFrame([{"ped": PEDS.set, "role": "set", "on_label": b_on,
                             "drift_host": np.nan, "shift_host": 5.0}])
        _, sets = decode_clock_marks(_frame(rows), PEDS, log)
        assert sets["status"].tolist() == ["send_log_conflict"]
        assert sets["shift"].isna().all()

    def test_bracket_disagreeing_with_its_pre_pulse_is_inconsistent(self):
        # Pre pulse says ~-4 s behind (shift +4); bracket logs 3 s (< 5 s).
        rows = _pulse(T0, 4.1, PEDS.behind) + _pulse(T0 + 4.2, 3.0, PEDS.set)
        _, sets = decode_clock_marks(_frame(rows), PEDS)
        assert sets["status"].tolist() == ["inconsistent"]
        assert sets["shift"].isna().all()

    def test_pulses_more_than_a_run_apart_do_not_chain(self):
        rows = _pulse(T0, 2.0, PEDS.behind) + _pulse(T0 + 60.0, 12.0, PEDS.set)
        drift, sets = decode_clock_marks(_frame(rows), PEDS)
        assert sets["status"].tolist() == ["no_pre_pulse"]
        assert drift["role"].tolist() == ["check"]


class TestSyntheticPairing:

    def test_zero_width_pulse_pairs(self):
        drift, _ = decode_clock_marks(_frame(_pulse(T0, 0.0, PEDS.ahead)), PEDS)
        assert drift["status"].tolist() == ["ok"]
        assert drift["drift"].tolist() == pytest.approx([0.15])
        assert drift["drift_lo"].iloc[0] == 0.0

    def test_repeated_on_leaves_the_first_unpaired(self):
        rows = [(T0, 90, PEDS.behind), (T0 + 5, 90, PEDS.behind), (T0 + 6, 89, PEDS.behind)]
        pulses = pair_marker_pulses(_frame(rows), PEDS)
        assert pulses["status"].tolist() == ["unpaired_on", "ok"]
        assert pulses["width"].iloc[1] == pytest.approx(1.0)

    def test_other_peds_and_codes_are_ignored(self):
        rows = _pulse(T0, 1.0, PEDS.behind) + [(T0 + 0.5, 90, 4), (T0 + 0.5, 89, 4),
                                                (T0 + 0.5, 82, PEDS.behind)]
        pulses = pair_marker_pulses(_frame(rows), PEDS)
        assert pulses["ped"].tolist() == [PEDS.behind]
        assert pulses["width"].tolist() == pytest.approx([1.0])

    def test_no_marker_events(self):
        drift, sets = decode_clock_marks(_frame([(T0, 1, 2), (T0 + 1, -1, -1)]), PEDS)
        assert drift.empty and sets.empty
        assert list(drift.columns)[:3] == ["ts", "off", "ped"]


class TestConfig:

    def test_no_keys_means_no_markers(self):
        assert marker_peds_from_config({"RB_R1": "1,2|3,4"}) is None

    def test_reads_all_three(self):
        cfg = {"Clk_Behind": "15", "Clk_Ahead": "16.0", "Clk_Set": 14}
        assert marker_peds_from_config(cfg) == MarkerPeds(15, 16, 14)

    @pytest.mark.parametrize("cfg", [
        {"Clk_Behind": "15", "Clk_Ahead": "16"},
        {"Clk_Behind": "15", "Clk_Ahead": "16", "Clk_Set": "8"},
        {"Clk_Behind": "15", "Clk_Ahead": "15", "Clk_Set": "14"},
        {"Clk_Behind": "x", "Clk_Ahead": "16", "Clk_Set": "14"},
    ])
    def test_rejects_partial_out_of_range_duplicate_or_junk(self, cfg):
        with pytest.raises(ValueError):
            marker_peds_from_config(cfg)

    def test_drop_marker_events_removes_ped_codes_on_marker_peds_only(self):
        rows = _pulse(T0, 1.0, PEDS.set) + [(T0, 21, PEDS.set), (T0, 21, 2),
                                             (T0, 1, PEDS.set), (T0, -1, -1)]
        kept = drop_marker_events(_frame(rows), PEDS)
        assert sorted(map(tuple, kept[["event_code", "parameter"]].values.tolist())) == [
            (-1, -1), (1, PEDS.set), (21, 2)]
