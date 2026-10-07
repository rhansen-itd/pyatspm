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
    MarkerPhases,
    decode_clock_marks,
    drop_marker_events,
    marker_phases_from_config,
    pair_marker_pulses,
    send_log_pulses,
)
from atspm.analysis.decoders import CLOCK_STEP_FENCE_PARAM, COMMS_GAP_PARAM
from atspm.data.ingestion import IngestionEngine
from atspm.data.manager import DatabaseManager

BENCH = Path(__file__).resolve().parents[1] / "fixtures" / "clock_marks_bench_2026_09_30"
MARKS = MarkerPhases(behind=15, ahead=16, set=14)


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
        _, sets = decode_clock_marks(bench_events, MARKS, bench_log)
        assert sets["shift"].tolist() == [-2, 3, 40, -40]
        assert sets["shift"].tolist() == sets["shift_host"].tolist()
        assert (sets["status"] == "ok").all()

    def test_periods_match_what_was_sent(self, bench_events, bench_log):
        _, sets = decode_clock_marks(bench_events, MARKS, bench_log)
        assert sets["period"].tolist() == [10, 20, 20, 50]

    def test_log_alone_decodes_the_unsaturated_sets_and_flags_the_rest(
        self, bench_events
    ):
        _, sets = decode_clock_marks(bench_events, MARKS)
        assert sets["shift"].iloc[:2].tolist() == [-2, 3]
        assert sets["status"].tolist() == ["ok", "ok", "pre_saturated", "pre_saturated"]
        assert sets["shift"].iloc[2:].isna().all()

    def test_step_window_holds_every_edit(self, bench_events, bench_log):
        # ENTER epochs are host time; the step window is pre-step label
        # time, so map each edit through its run's pre-set drift.
        _, sets = decode_clock_marks(bench_events, MARKS, bench_log)
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
        drift, _ = decode_clock_marks(bench_events, MARKS, bench_log)
        assert len(drift) == 9  # 13 pulses sent in these files, 4 of them brackets
        clean = drift.loc[~drift["saturated"]]
        assert clean["drift_host"].notna().all()
        assert ((clean["drift_lo"] <= clean["drift_host"])
                & (clean["drift_host"] <= clean["drift_hi"])).all()
        assert (clean["drift"] - clean["drift_host"]).abs().max() < 0.2

    def test_saturated_pulses_take_the_send_log_value(self, bench_events, bench_log):
        drift, _ = decode_clock_marks(bench_events, MARKS, bench_log)
        sat = drift.loc[drift["saturated"]]
        assert sat["status"].tolist() == ["send_log", "send_log"]
        assert sat["drift"].tolist() == pytest.approx([-39.633, 40.291])

    def test_saturated_pulses_without_the_log_are_bounds_only(self, bench_events):
        drift, _ = decode_clock_marks(bench_events, MARKS)
        sat = drift.loc[drift["saturated"]]
        assert sat["status"].tolist() == ["saturated", "saturated"]
        assert sat["drift"].isna().all()
        behind, ahead = sat.iloc[0], sat.iloc[1]
        assert behind["drift_lo"] == -np.inf and behind["drift_hi"] <= -29.6
        assert ahead["drift_hi"] == np.inf and ahead["drift_lo"] >= 29.6

    def test_roles(self, bench_events):
        drift, _ = decode_clock_marks(bench_events, MARKS)
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
    ped_pre = MARKS.ahead if drift_pre > 0 else MARKS.behind
    w_pre = round(abs(drift_pre) - 0.15, 1)
    rows = _pulse(t, w_pre, ped_pre)
    b_on = round(t + w_pre + 0.1, 1)
    b_off = round(b_on + period + shift - 0.1, 1)
    rows += _pulse(b_on, b_off - b_on, MARKS.set)
    if with_residual:
        rows += _pulse(round(b_off + 0.1, 1), residual, MARKS.ahead)
    return rows, b_on, b_off


class TestSyntheticSets:

    def test_small_backward_set_crosses_its_own_fence(self):
        # Busy controller, -2 s set: the fence lands inside the bracket in
        # sorted order (first edit 1.4 s after ON, labels step back 2 s).
        rows, b_on, _ = _set_run(T0, drift_pre=2.1, shift=-2, period=10)
        rows += [(b_on + 3.0 - 2.0 - 0.05, -1, CLOCK_STEP_FENCE_PARAM)]
        _, sets = decode_clock_marks(_frame(rows), MARKS)
        assert sets["shift"].tolist() == [-2]
        assert sets["status"].tolist() == ["ok"]

    def test_bracket_split_by_a_comms_gap_is_flagged_not_guessed(self):
        rows, b_on, _ = _set_run(T0, drift_pre=2.1, shift=-2, period=10)
        rows += [(b_on + 4.0, -1, COMMS_GAP_PARAM)]
        _, sets = decode_clock_marks(_frame(rows), MARKS)
        assert sets["status"].tolist() == ["unpaired_on", "unpaired_off"]
        assert sets["shift"].isna().all()

    def test_drift_pulse_split_by_a_comms_gap_is_flagged(self):
        rows = _pulse(T0, 3.0, MARKS.behind) + [(T0 + 1.0, -1, COMMS_GAP_PARAM)]
        drift, _ = decode_clock_marks(_frame(rows), MARKS)
        assert drift["status"].tolist() == ["unpaired_on", "unpaired_off"]
        assert drift["drift"].isna().all()

    def test_missing_post_set_pulse_still_decodes(self):
        rows, _, _ = _set_run(T0, drift_pre=-4.3, shift=4, period=10, with_residual=False)
        drift, sets = decode_clock_marks(_frame(rows), MARKS)
        assert sets["shift"].tolist() == [4]
        assert drift["role"].tolist() == ["pre_set"]

    def test_missing_pre_set_pulse(self):
        rows, _, _ = _set_run(T0, drift_pre=-4.3, shift=4, period=10)
        rows = [r for r in rows if r[2] != MARKS.behind]
        _, sets = decode_clock_marks(_frame(rows), MARKS)
        assert sets["status"].tolist() == ["no_pre_pulse"]
        assert sets["shift"].isna().all()

    def test_missing_pre_set_pulse_takes_the_send_log_shift(self):
        rows, b_on, _ = _set_run(T0, drift_pre=-4.3, shift=4, period=10)
        rows = [r for r in rows if r[2] != MARKS.behind]
        log = pd.DataFrame([{"phase": MARKS.set, "role": "set", "on_label": b_on + 0.3,
                             "drift_host": np.nan, "shift_host": 4.0}])
        _, sets = decode_clock_marks(_frame(rows), MARKS, log)
        assert sets["status"].tolist() == ["send_log"]
        assert sets["shift"].tolist() == [4]

    def test_send_log_contradicting_a_decoded_set_is_flagged(self):
        rows, b_on, _ = _set_run(T0, drift_pre=-4.3, shift=4, period=10)
        log = pd.DataFrame([{"phase": MARKS.set, "role": "set", "on_label": b_on,
                             "drift_host": np.nan, "shift_host": 5.0}])
        _, sets = decode_clock_marks(_frame(rows), MARKS, log)
        assert sets["status"].tolist() == ["send_log_conflict"]
        assert sets["shift"].isna().all()

    def test_bracket_disagreeing_with_its_pre_pulse_is_inconsistent(self):
        # Pre pulse says ~-4 s behind (shift +4); bracket logs 3 s (< 5 s).
        rows = _pulse(T0, 4.1, MARKS.behind) + _pulse(T0 + 4.2, 3.0, MARKS.set)
        _, sets = decode_clock_marks(_frame(rows), MARKS)
        assert sets["status"].tolist() == ["inconsistent"]
        assert sets["shift"].isna().all()

    def test_pulses_more_than_a_run_apart_do_not_chain(self):
        rows = _pulse(T0, 2.0, MARKS.behind) + _pulse(T0 + 60.0, 12.0, MARKS.set)
        drift, sets = decode_clock_marks(_frame(rows), MARKS)
        assert sets["status"].tolist() == ["no_pre_pulse"]
        assert drift["role"].tolist() == ["check"]


class TestSyntheticPairing:

    def test_zero_width_pulse_pairs(self):
        drift, _ = decode_clock_marks(_frame(_pulse(T0, 0.0, MARKS.ahead)), MARKS)
        assert drift["status"].tolist() == ["ok"]
        assert drift["drift"].tolist() == pytest.approx([0.15])
        assert drift["drift_lo"].iloc[0] == 0.0

    def test_repeated_on_leaves_the_first_unpaired(self):
        rows = [(T0, 90, MARKS.behind), (T0 + 5, 90, MARKS.behind), (T0 + 6, 89, MARKS.behind)]
        pulses = pair_marker_pulses(_frame(rows), MARKS)
        assert pulses["status"].tolist() == ["unpaired_on", "ok"]
        assert pulses["width"].iloc[1] == pytest.approx(1.0)

    def test_other_peds_and_codes_are_ignored(self):
        rows = _pulse(T0, 1.0, MARKS.behind) + [(T0 + 0.5, 90, 4), (T0 + 0.5, 89, 4),
                                                (T0 + 0.5, 82, MARKS.behind)]
        pulses = pair_marker_pulses(_frame(rows), MARKS)
        assert pulses["phase"].tolist() == [MARKS.behind]
        assert pulses["width"].tolist() == pytest.approx([1.0])

    def test_no_marker_events(self):
        drift, sets = decode_clock_marks(_frame([(T0, 1, 2), (T0 + 1, -1, -1)]), MARKS)
        assert drift.empty and sets.empty
        assert list(drift.columns)[:4] == ["ts", "off", "phase", "mark"]


class TestConfig:

    def test_no_keys_means_no_markers(self):
        assert marker_phases_from_config({"RB_R1": "1,2|3,4"}) is None

    def test_reads_all_three(self):
        cfg = {"Clk_Behind": "15", "Clk_Ahead": "16.0", "Clk_Set": 14}
        assert marker_phases_from_config(cfg) == MarkerPhases(15, 16, 14)

    @pytest.mark.parametrize("cfg", [
        {"Clk_Behind": "15", "Clk_Ahead": "16"},
        {"Clk_Behind": "15", "Clk_Ahead": "16", "Clk_Set": "8"},
        {"Clk_Behind": "15", "Clk_Ahead": "15", "Clk_Set": "14"},
        {"Clk_Behind": "x", "Clk_Ahead": "16", "Clk_Set": "14"},
    ])
    def test_rejects_partial_out_of_range_duplicate_or_junk(self, cfg):
        with pytest.raises(ValueError):
            marker_phases_from_config(cfg)

    def test_drop_marker_events_removes_ped_codes_on_marker_peds_only(self):
        rows = _pulse(T0, 1.0, MARKS.set) + [(T0, 21, MARKS.set), (T0, 21, 2),
                                             (T0, 1, MARKS.set), (T0, -1, -1)]
        kept = drop_marker_events(_frame(rows), MARKS)
        assert sorted(map(tuple, kept[["event_code", "parameter"]].values.tolist())) == [
            (-1, -1), (1, MARKS.set), (21, 2)]


# ---------------------------------------------------------------------------
# Phase-hold encoding (upstream from 2026-10-07)
# ---------------------------------------------------------------------------

def _hold(on, width, phase):
    return [(on, 41, phase), (on + width, 42, phase)]


def _as_holds(events: pd.DataFrame) -> pd.DataFrame:
    """The bench log re-encoded the way upstream now marks: 90/89 -> 41/42,
    and no code 45 (a hold places no ped call)."""
    on_mark = events["parameter"].isin(MARKS.all)
    ev = events.loc[~(on_mark & (events["event_code"] == 45))].copy()
    code = ev["event_code"]
    ev.loc[on_mark & (code == 90), "event_code"] = 41
    ev.loc[on_mark & (code == 89), "event_code"] = 42
    return ev


def _hold_records():
    """bench.jsonl as upstream now writes it: ``phase`` + ``mark``, no ``ped``."""
    runs = [json.loads(l) for l in open(BENCH / "bench.jsonl")]
    for run in runs:
        for p in run.get("pulses") or []:
            p["phase"] = p.pop("ped")
            p["mark"] = "hold"
            p["cib"] = p["phase"] + 79
    return runs


class TestPhaseHolds:

    def test_bench_decodes_the_same_as_holds(self, bench_events, bench_log):
        # Re-encoded ped widths carry the ped shortfall (up to 0.3 s), so the
        # two capped pulses (+/-40 s runs) log under the hold saturation
        # threshold; a real capped hold logs >= 29.8 s.  The first two runs
        # are compared: same shifts, drift moved by the bias difference.
        ped_drift, ped_sets = decode_clock_marks(bench_events, MARKS, bench_log)
        hold_log = send_log_pulses(_hold_records())
        drift, sets = decode_clock_marks(_as_holds(bench_events), MARKS, hold_log)
        assert (drift["mark"] == "hold").all() and (ped_drift["mark"] == "ped").all()
        assert sets["shift"].iloc[:2].tolist() == ped_sets["shift"].iloc[:2].tolist() == [-2, 3]
        assert (sets["status"].iloc[:2] == "ok").all()
        clean = (~ped_drift["saturated"]).to_numpy()
        sign = np.sign(ped_drift.loc[clean, "drift"].to_numpy())
        moved = ped_drift.loc[clean, "drift"].to_numpy() - drift.loc[clean, "drift"].to_numpy()
        assert moved.tolist() == pytest.approx((0.10 * sign).tolist())

    # Holds measured 2026-10-07 (sent -> logged, s): seven on the bench with
    # 313's database, one bench --drift, one each in the field at 313 / 701.
    HOLDS_MEASURED = [(2.00, 1.9), (1.01, 1.0), (5.00, 5.0), (0.50, 0.5), (3.00, 3.0),
                      (0.30, 0.3), (10.00, 10.0), (7.40, 7.2), (3.30, 3.3), (2.10, 2.1)]

    @pytest.mark.parametrize("sent, logged", HOLDS_MEASURED)
    def test_measured_holds_decode_within_bounds(self, sent, logged):
        drift, _ = decode_clock_marks(_frame(_hold(T0, logged, MARKS.ahead)), MARKS)
        row = drift.iloc[0]
        assert row["drift_lo"] - 1e-9 <= sent <= row["drift_hi"] + 1e-9
        assert abs(row["drift"] - sent) <= 0.15 + 1e-9

    def test_holds_saturate_at_their_own_threshold(self):
        rows = _hold(T0, 29.7, MARKS.ahead) + _pulse(T0 + 60, 29.7, MARKS.ahead)
        drift, _ = decode_clock_marks(_frame(rows), MARKS)
        assert drift["status"].tolist() == ["ok", "saturated"]

    def test_send_log_reads_phase_and_ped_records(self, bench_log):
        assert send_log_pulses(_hold_records()).equals(bench_log)

    def test_zero_width_hold_pairs(self):
        drift, _ = decode_clock_marks(_frame(_hold(T0, 0.0, MARKS.behind)), MARKS)
        assert drift["status"].tolist() == ["ok"]
        assert drift["drift"].tolist() == pytest.approx([-0.05])

    def test_a_log_spanning_the_switch_decodes_both(self):
        rows = _pulse(T0, 0.5, MARKS.behind) + _hold(T0 + 3600, 0.7, MARKS.behind)
        drift, _ = decode_clock_marks(_frame(rows), MARKS)
        assert drift["mark"].tolist() == ["ped", "hold"]
        assert drift["drift"].tolist() == pytest.approx([-0.65, -0.75])

    def test_encodings_never_pair_with_each_other(self):
        rows = [(T0, 90, MARKS.ahead), (T0 + 1.0, 42, MARKS.ahead)]
        pulses = pair_marker_pulses(_frame(rows), MARKS)
        assert pulses["status"].tolist() == ["unpaired_on", "unpaired_off"]
        assert pulses["mark"].tolist() == ["ped", "hold"]

    def test_holds_on_other_phases_are_ignored(self):
        rows = _hold(T0, 1.0, MARKS.set) + _hold(T0 + 0.5, 3.0, 4)
        pulses = pair_marker_pulses(_frame(rows), MARKS)
        assert pulses["phase"].tolist() == [MARKS.set]

    def test_drop_marker_events_removes_holds_on_marker_phases_only(self):
        rows = _hold(T0, 1.0, MARKS.ahead) + _hold(T0, 1.0, 2)
        kept = drop_marker_events(_frame(rows), MARKS)
        assert kept["parameter"].tolist() == [2, 2]
