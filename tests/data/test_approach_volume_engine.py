# Acceptance tests for the approach volume shell engine, plot and CLI
# (UDOT S-M8; spec docs/specs/approach_volume_shell.md).
#
# Opus-written; the implementation must make these pass without editing
# them.  The measure itself is pinned in tests/analysis/test_approach_volume.py.

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm import cli
from atspm.analysis.approach_volume import BIN_SCHEMA, DAY_SCHEMA, approach_volume
from atspm.analysis.counts import parse_movements_from_config
from atspm.data.counts import CountEngine
from atspm.data.manager import DatabaseManager
from atspm.utils.timezone import to_epoch

# Built by the delegated S-M8 run; these imports fail until they exist.
from atspm.data.approach_volume import ApproachVolumeEngine, get_approach_volume
from atspm.plotting.approach_volume import plot_approach_volume

TZ = "US/Mountain"
DAY = "2025-06-02"
STAMP = "2025_06_02"
SUB_START, SUB_END = f"{DAY} 07:00", f"{DAY} 09:00"
SUB_STAMP = "2025_06_02_0700-2025_06_02_0900"

# ---------------------------------------------------------------------------
# Synthetic intersection, one whole local day (96 bins of 15 min), every
# bin covered by ingestion_log.  Counts per bin (Code 82, evenly spaced):
#   NB  det 1  4, except 07:00..07:45 = 20, 30, 40, 10        total 468
#   SB  det 2  2, except 16:00..16:45 = 25 each               total 284
#   EB  det 4  3 everywhere                                   total 288
#   WB  det 5  configured, never actuates
# NB/SB: NB peak 07:00 (100, max bin 40), SB in that hour 8;
#        SB peak 16:00 (100), NB in that hour 16;
#        combined peak 16:00 (116), day total 752.
# EB/WB: every hour ties at 12 → 00:00; WB 0.
# Variants: a gap marker at 07:31 (bin 07:30 partial); TM_WBR sharing
# det 4 with EB plus an unparseable TM_Bike label; no TM_* keys at all.
# ---------------------------------------------------------------------------

_BINS = 96


def _per_bin():
    nb = np.full(_BINS, 4); nb[28:32] = [20, 30, 40, 10]
    sb = np.full(_BINS, 2); sb[64:68] = 25
    eb = np.full(_BINS, 3)
    return {1: nb, 2: sb, 4: eb}


def _local(hhmm: str) -> float:
    return to_epoch(datetime.strptime(f"{DAY} {hhmm}", "%Y-%m-%d %H:%M"), TZ)


def _build_db(root: Path, gap: bool = False, shared: bool = False,
              tm: bool = True) -> Path:
    t0 = _local("00:00")
    events = []
    for det, counts in _per_bin().items():
        for b, n in enumerate(counts):
            for j in range(int(n)):
                t = t0 + b * 900.0 + (j + 0.5) * 900.0 / n
                events += [(round(t, 1), 82, det), (round(t, 1) + 0.2, 81, det)]
    if gap:
        events.append((_local("07:31"), -1, -1))
    db = root / "av.db"
    with DatabaseManager(db) as m:
        m.init_db()
        m.set_metadata(intersection_name="Synthetic", timezone=TZ,
                       major_road_name="Main St", minor_road_name="Side St")
        cfg = {"RB_R1": "2|4"}
        if tm:
            cfg.update({"TM_NBT": "1", "TM_SBT": "2", "TM_EBT": "4", "TM_WBT": "5",
                        "TM_Exclusions": "[]"})
        if shared:
            cfg.update({"TM_WBR": "4", "TM_Bike": "9"})
        for col in cfg:
            m.add_config_column(col)
        m._insert_config_row({"start_date": "2000-01-01T00:00:00",
                              "end_date": None, **cfg})
        m.insert_events(sorted(events))
        m.conn.execute(
            "INSERT INTO ingestion_log (span_start, span_end, processed_at, "
            "row_count) VALUES (?, ?, '2025-06-03T00:00:00', ?)",
            (t0, t0 + 86400.0, len(events)),
        )
        m.conn.commit()
    return db


@pytest.fixture
def db(tmp_path) -> Path:
    return _build_db(tmp_path)


def _day(res, direction, pair):
    d = res["days"]
    out = d.loc[(d["direction"] == direction) & (d["pair"] == pair)]
    assert len(out) == 1, out
    return out.iloc[0]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:

    def test_result_keys_and_columns(self, db):
        res = ApproachVolumeEngine(db).approach_volume(DAY, DAY)
        assert set(res) == {"bins", "days"}
        assert list(res["bins"].columns) == BIN_SCHEMA
        assert list(res["days"].columns) == DAY_SCHEMA

    def test_whole_day_values(self, db):
        res = ApproachVolumeEngine(db).approach_volume(DAY, DAY)
        b = res["bins"]
        assert len(b) == _BINS * 6            # NB, SB, combined, EB, WB, combined
        assert b["complete"].all()
        nb = _day(res, "NB", "NB/SB")
        assert nb["total_volume"] == 468 and nb["complete_day"]
        assert nb["peak_start"] == pd.Timestamp(f"{DAY} 07:00", tz=TZ)
        assert nb["peak_volume"] == 100 and nb["peak_bin_volume"] == 40
        assert nb["phf"] == pytest.approx(0.625)
        assert nb["d_factor"] == pytest.approx(100 / 108)
        assert nb["k_factor"] == pytest.approx(108 / 752)
        sb = _day(res, "SB", "NB/SB")
        assert sb["peak_start"] == pd.Timestamp(f"{DAY} 16:00", tz=TZ)
        assert sb["d_factor"] == pytest.approx(100 / 116)
        c = _day(res, "combined", "NB/SB")
        assert c["peak_volume"] == 116 and c["total_volume"] == 752
        assert c["k_factor"] == pytest.approx(116 / 752)
        eb = _day(res, "EB", "EB/WB")
        assert eb["peak_start"] == pd.Timestamp(f"{DAY} 00:00", tz=TZ)
        assert eb["d_factor"] == pytest.approx(1.0)
        assert _day(res, "WB", "EB/WB")["total_volume"] == 0

    def test_matches_the_core_on_the_same_counts(self, db):
        res = ApproachVolumeEngine(db).approach_volume(DAY, DAY)
        with DatabaseManager(db) as m:
            cfg = m.get_config_at_date(datetime(2025, 6, 2))
        counts = CountEngine(db, TZ).vehicle_counts(DAY, DAY, bin_len=15,
                                                    include_detectors=True)
        bins, days = approach_volume(counts, parse_movements_from_config(cfg))
        pd.testing.assert_frame_equal(res["bins"].reset_index(drop=True), bins)
        pd.testing.assert_frame_equal(res["days"].reset_index(drop=True), days)

    def test_gap_marker_makes_its_bin_incomplete(self, tmp_path):
        res = ApproachVolumeEngine(_build_db(tmp_path, gap=True)).approach_volume(DAY, DAY)
        b = res["bins"]
        t = b.loc[b["time"] == pd.Timestamp(f"{DAY} 07:30", tz=TZ)]
        assert len(t) == 6 and not t["complete"].any() and t["vph"].isna().all()
        assert b["complete"].sum() == (_BINS - 1) * 6
        nb = _day(res, "NB", "NB/SB")
        assert nb["peak_start"] == pd.Timestamp(f"{DAY} 06:30", tz=TZ)
        assert nb["peak_volume"] == 58            # 4 + 4 + 20 + 30
        assert not nb["complete_day"] and pd.isna(nb["k_factor"])
        assert nb["total_volume"] == 468          # the bin's volume is kept

    def test_sub_day_window_has_no_k(self, db, capsys):
        res = ApproachVolumeEngine(db).approach_volume(SUB_START, SUB_END)
        b = res["bins"]
        assert b["time"].min() == pd.Timestamp(SUB_START, tz=TZ)
        assert b["time"].max() == pd.Timestamp(f"{DAY} 08:45", tz=TZ)
        nb = _day(res, "NB", "NB/SB")
        assert nb["peak_start"] == pd.Timestamp(SUB_START, tz=TZ)
        assert pd.isna(nb["k_factor"])
        assert "K-factor" in capsys.readouterr().out

    def test_bin_len(self, db):
        res = ApproachVolumeEngine(db).approach_volume(DAY, DAY, bin_len=60)
        nb = _day(res, "NB", "NB/SB")
        assert nb["n_bins"] == 24 and nb["peak_volume"] == 100
        assert res["bins"]["time"].nunique() == 24

    def test_bad_bin_len_raises_before_the_db(self, tmp_path):
        with pytest.raises(ValueError):
            ApproachVolumeEngine(tmp_path / "absent.db", timezone=TZ).approach_volume(
                DAY, DAY, bin_len=7)

    def test_silent_detector_warning(self, db, capsys):
        ApproachVolumeEngine(db).approach_volume(DAY, DAY)
        assert "WB detector(s) 5 logged no actuation" in capsys.readouterr().out

    def test_shared_detector_and_unparsed_label_warnings(self, tmp_path, capsys):
        res = ApproachVolumeEngine(_build_db(tmp_path, shared=True)).approach_volume(DAY, DAY)
        out = capsys.readouterr().out
        assert "detector 4 is configured in both EB and WB" in out
        assert "TM_Bike" in out
        assert _day(res, "WB", "EB/WB")["total_volume"] == 288

    def test_no_tm_config_returns_empty(self, tmp_path, capsys):
        eng = ApproachVolumeEngine(_build_db(tmp_path, tm=False))
        assert eng.approach_volume(DAY, DAY) == {}
        assert "TM_" in capsys.readouterr().out
        out = tmp_path / "out"
        assert eng.approach_volume(DAY, DAY, output_dir=out) is None
        assert not out.exists() or not list(out.iterdir())

    def test_no_events_returns_empty(self, db):
        assert ApproachVolumeEngine(db).approach_volume("2025-07-01", "2025-07-01") == {}

    def test_output_dir_writes_files(self, db, tmp_path):
        out = tmp_path / "out"
        assert ApproachVolumeEngine(db).approach_volume(DAY, DAY, output_dir=out) is None
        names = {p.name for p in out.iterdir()}
        assert names == {f"AV_Bins_15min_{STAMP}.csv", f"AV_Days_{STAMP}.csv",
                         f"AV_Chart_{STAMP}.html"}
        days = pd.read_csv(out / f"AV_Days_{STAMP}.csv")
        assert list(days.columns) == DAY_SCHEMA and len(days) == 6
        bins = pd.read_csv(out / f"AV_Bins_15min_{STAMP}.csv")
        assert list(bins.columns) == BIN_SCHEMA

    def test_sub_day_stamp_and_no_plot(self, db, tmp_path):
        out = tmp_path / "out"
        ApproachVolumeEngine(db).approach_volume(SUB_START, SUB_END, make_plot=False,
                                                 output_dir=out)
        names = {p.name for p in out.iterdir()}
        assert names == {f"AV_Bins_15min_{SUB_STAMP}.csv", f"AV_Days_{SUB_STAMP}.csv"}

    def test_convenience_wrapper(self, db):
        res = get_approach_volume(db, DAY, DAY, timezone=TZ)
        assert _day(res, "combined", "NB/SB")["peak_volume"] == 116


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


class TestPlot:

    @staticmethod
    def _frames(db):
        return ApproachVolumeEngine(db).approach_volume(DAY, DAY)

    def test_traces_per_pair(self, db):
        res = self._frames(db)
        fig = plot_approach_volume(res["bins"], res["days"], {"intersection_name": "X"})
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        for n in ("NB Volume", "SB Volume", "NB/SB Combined", "NB D-Factor",
                  "SB D-Factor", "EB Volume", "WB Volume", "EB/WB Combined",
                  "EB D-Factor", "WB D-Factor"):
            assert names.count(n) == 1, n
        nb = next(t for t in fig.data if t.name == "NB Volume")
        assert nb.line.shape == "hv"
        assert max(v for v in nb.y if v is not None and not np.isnan(v)) == 160.0
        dfac = next(t for t in fig.data if t.name == "NB D-Factor")
        assert dfac.line.dash == "dash"
        assert dfac.yaxis != nb.yaxis       # secondary axis

    def test_incomplete_bins_break_the_line(self, tmp_path):
        res = ApproachVolumeEngine(_build_db(tmp_path, gap=True)).approach_volume(DAY, DAY)
        fig = plot_approach_volume(res["bins"], res["days"])
        nb = next(t for t in fig.data if t.name == "NB Volume")
        assert sum(1 for v in nb.y if v is None or np.isnan(v)) == 1
        assert not nb.connectgaps

    def test_peak_hours_are_shaded(self, db):
        res = self._frames(db)
        fig = plot_approach_volume(res["bins"], res["days"])
        # One combined peak per (day, pair) with a peak: NB/SB and EB/WB.
        assert len(fig.layout.shapes) == 2

    def test_multi_day_and_one_sided(self):
        # Two days, EB only: no D-factor traces, one shaded peak per day.
        idx = pd.date_range(pd.Timestamp("2025-06-02", tz=TZ), periods=192,
                            freq="15min", name="Time")
        counts = pd.DataFrame({4: np.arange(192) % 7}, index=idx)
        bins, days = approach_volume(counts, {"EBT": [4]})
        fig = plot_approach_volume(bins, days)
        names = [t.name for t in fig.data]
        assert "EB Volume" in names and "EB/WB Combined" in names
        assert not any("D-Factor" in n for n in names)
        assert len(fig.layout.shapes) == 2

    def test_title_uses_metadata(self, db):
        res = self._frames(db)
        fig = plot_approach_volume(res["bins"], res["days"],
                                   {"major_road_name": "Main St", "minor_road_name": "Side St"})
        assert "Main St" in fig.layout.title.text
        assert "Approach Volume" in fig.layout.title.text

    def test_empty_frames_give_figure(self):
        fig = plot_approach_volume(pd.DataFrame(columns=BIN_SCHEMA),
                                   pd.DataFrame(columns=DAY_SCHEMA))
        assert isinstance(fig, go.Figure) and len(fig.data) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    @staticmethod
    def _parse(*argv):
        return cli._build_parser().parse_args(["approach-volume", *argv])

    def test_parses_with_defaults(self):
        args = self._parse("--targetid", "900", "--start", DAY, "--end", DAY)
        assert args.func is cli.handle_approach_volume
        assert args.bin_len == 15 and args.no_plot is False

    def test_target_group_is_required_and_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse("--start", DAY, "--end", DAY)
        with pytest.raises(SystemExit):
            self._parse("--all", "--targetid", "900", "--start", DAY, "--end", DAY)

    @staticmethod
    def _site(tmp_path, monkeypatch, **kw):
        folder = tmp_path / "intersections" / "900_Synthetic"
        folder.mkdir(parents=True)
        _build_db(folder, **kw)
        (folder / "metadata.json").write_text(json.dumps({
            "intersection_name": "Synthetic", "intersection_id": "900",
            "db_filename": "av.db", "timezone": TZ,
        }))
        monkeypatch.chdir(tmp_path)
        return folder

    def test_end_to_end(self, tmp_path, monkeypatch, capsys):
        folder = self._site(tmp_path, monkeypatch)
        args = self._parse("--targetid", "900", "--start", DAY, "--end", DAY, "--no-plot")
        args.func(args)
        days = pd.read_csv(folder / "outputs" / f"AV_Days_{STAMP}.csv")
        assert len(days) == 6
        assert not list((folder / "outputs").glob("*.html"))
        out = capsys.readouterr().out
        assert "NB/SB" in out and "EB/WB" in out
        assert "16:00" in out                 # combined NB/SB peak hour

    def test_no_tm_config_does_not_raise(self, tmp_path, monkeypatch, capsys):
        folder = self._site(tmp_path, monkeypatch, tm=False)
        args = self._parse("--targetid", "900", "--start", DAY, "--end", DAY)
        args.func(args)
        assert "TM_" in capsys.readouterr().out
        out = folder / "outputs"
        assert not out.exists() or not list(out.glob("AV_*"))
