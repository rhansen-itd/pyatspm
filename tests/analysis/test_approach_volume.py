"""Golden tests for Approach Volume (Functional Core).

Target: src/atspm/analysis/approach_volume.py (UDOT roadmap S-M8).  Opus-written.

Contract summary (UDOT ``ApproachVolumeService.cs``)
---------------------------------------------------
Input: binned raw Code 82 counts per detector (``CountEngine.vehicle_counts``
with ``include_detectors=True``) plus ``data_quality``.  Directions come
from the ``TM_*`` label prefix (NB/SB/EB/WB); pairs NB/SB and EB/WB.
vph = count × 60 / bin_len.  Peak hour: rolling hour of complete bins with
the largest volume, earliest on ties.  PHF = peak / (max bin × bins per
hour).  D = peak / (peak + opposing in that hour).  K = combined volume in
the row's peak hour / combined day total, complete local days only.
"""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atspm.analysis.approach_volume import (
    BIN_SCHEMA,
    DAY_SCHEMA,
    approach_volume,
    direction_detectors,
    unparsed_movements,
)

_TZ = "US/Mountain"
_MV = {"NBL": [1], "NBT": [2], "SBT": [3], "EBT": [4], "X_Ped": [9]}


def _grid(day="2025-12-15", n=96, bin_len=15):
    return pd.date_range(pd.Timestamp(day, tz=_TZ), periods=n, freq=f"{bin_len}min",
                         name="Time")


def _counts(cols, index=None, quality=None):
    """Detector-column frame; *cols* maps detector → array of counts."""
    idx = _grid() if index is None else index
    df = pd.DataFrame({d: np.asarray(v, dtype=np.int64) for d, v in cols.items()}, index=idx)
    df["NBL"] = 0.0  # movement columns are ignored
    if quality is not None:
        df["data_quality"] = quality
    return df


def _row(days, direction, pair=None, day=None):
    sel = days["direction"] == direction
    if pair is not None:
        sel &= days["pair"] == pair
    if day is not None:
        sel &= days["date"] == day
    out = days.loc[sel]
    assert len(out) == 1, out
    return out.iloc[0]


# ---------------------------------------------------------------------------
# Direction grouping
# ---------------------------------------------------------------------------

def test_direction_detectors_groups_by_prefix_and_dedupes():
    mv = {"NBL": [1], "NBT": [2, 1], "SBR": [3], "EBT": [4], "WBL": [4], "Ped": [9]}
    assert direction_detectors(mv) == {"NB": [1, 2], "SB": [3], "EB": [4], "WB": [4]}
    assert unparsed_movements(mv) == ["Ped"]


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def test_bins_schema_vph_and_d_split():
    n = 96
    nb1, nb2, sb = np.zeros(n), np.zeros(n), np.zeros(n)
    nb1[0], nb2[0], sb[0] = 2, 1, 1          # NB 3, SB 1 at 00:00
    bins, _ = approach_volume(_counts({1: nb1, 2: nb2, 3: sb, 4: np.zeros(n)}), _MV)
    assert list(bins.columns) == BIN_SCHEMA
    t0 = bins.loc[bins["time"] == _grid()[0]].set_index(["pair", "direction"])
    assert t0.loc[("NB/SB", "NB"), "volume"] == 3
    assert t0.loc[("NB/SB", "NB"), "vph"] == 12.0
    assert t0.loc[("NB/SB", "SB"), "d_split"] == pytest.approx(0.25)
    assert t0.loc[("NB/SB", "NB"), "d_split"] == pytest.approx(0.75)
    assert t0.loc[("NB/SB", "combined"), "volume"] == 4
    assert np.isnan(t0.loc[("NB/SB", "combined"), "d_split"])
    # Zero combined bin: split NA, not 0 (UDOT reports 0).
    t1 = bins.loc[bins["time"] == _grid()[1]].set_index(["pair", "direction"])
    assert np.isnan(t1.loc[("NB/SB", "NB"), "d_split"])
    assert t1.loc[("NB/SB", "NB"), "vph"] == 0.0
    # Row order within a pair: configured directions, then combined.
    assert bins["direction"].unique().tolist() == ["NB", "SB", "combined", "EB"]


def test_one_sided_pair_has_no_split_and_combined_equals_it():
    n = 96
    eb = np.arange(n) % 5
    bins, days = approach_volume(_counts({4: eb}), {"EBT": [4]})
    assert set(bins["direction"]) == {"EB", "combined"}
    assert bins["d_split"].isna().all()
    e, c = _row(days, "EB"), _row(days, "combined")
    assert e["peak_volume"] == c["peak_volume"]
    assert pd.isna(e["d_factor"])               # UDOT: 1.0
    assert e["k_factor"] == pytest.approx(c["k_factor"])


def test_missing_detector_column_counts_zero():
    n = 96
    bins, days = approach_volume(_counts({1: np.ones(n)}), {"NBT": [1, 77], "SBT": [78]})
    assert _row(days, "SB")["total_volume"] == 0
    assert _row(days, "NB")["total_volume"] == n


def test_incomplete_bin_keeps_volume_but_no_rate():
    n = 96
    q = np.array(["ok"] * n, dtype=object)
    q[5] = "partial"
    bins, _ = approach_volume(_counts({1: np.full(n, 2), 3: np.full(n, 2)}, quality=q), _MV)
    b5 = bins.loc[(bins["time"] == _grid()[5]) & (bins["direction"] == "NB")].iloc[0]
    assert b5["volume"] == 2 and not b5["complete"]
    assert np.isnan(b5["vph"]) and np.isnan(b5["d_split"])


# ---------------------------------------------------------------------------
# Peak hour, PHF, D, K
# ---------------------------------------------------------------------------

def _peak_day():
    """NB peaks 07:00–08:00 (10, 20, 30, 40 = 100), SB peaks 16:00 (4 × 25)."""
    n = 96
    nb, sb = np.ones(n, dtype=int), np.ones(n, dtype=int)
    nb[28:32] = [10, 20, 30, 40]          # 07:00 .. 07:45
    sb[28:32] = [5, 5, 5, 5]
    sb[64:68] = 25                         # 16:00 .. 16:45
    return nb, sb


def test_peak_hour_phf_d_and_k():
    nb, sb = _peak_day()
    _, days = approach_volume(_counts({2: nb, 3: sb}), _MV)
    assert list(days.columns) == DAY_SCHEMA
    tot = int(nb.sum() + sb.sum())

    n_ = _row(days, "NB")
    assert n_["peak_start"] == pd.Timestamp("2025-12-15 07:00", tz=_TZ)
    assert n_["peak_volume"] == 100 and n_["peak_bin_volume"] == 40
    assert n_["phf"] == pytest.approx(100 / 160)
    assert n_["pair_peak_volume"] == 120          # + SB 4 × 5
    assert n_["d_factor"] == pytest.approx(100 / 120)
    assert n_["k_factor"] == pytest.approx(120 / tot)
    assert n_["n_bins"] == 96 and n_["n_complete"] == 96 and n_["complete_day"]

    s_ = _row(days, "SB")
    assert s_["peak_start"] == pd.Timestamp("2025-12-15 16:00", tz=_TZ)
    assert s_["peak_volume"] == 100 and s_["phf"] == pytest.approx(1.0)
    assert s_["pair_peak_volume"] == 104          # + NB 4 × 1
    assert s_["d_factor"] == pytest.approx(100 / 104)

    c_ = _row(days, "combined", pair="NB/SB")
    assert c_["peak_volume"] == 120 and pd.isna(c_["d_factor"])
    assert c_["k_factor"] == pytest.approx(120 / tot)
    assert c_["total_volume"] == tot


def test_peak_tie_goes_to_earliest_hour():
    n = 96
    nb = np.zeros(n, dtype=int)
    nb[8:12] = 5     # 02:00
    nb[40:44] = 5    # 10:00
    _, days = approach_volume(_counts({2: nb}), {"NBT": [2]})
    assert _row(days, "NB")["peak_start"] == pd.Timestamp("2025-12-15 02:00", tz=_TZ)


def test_incomplete_bin_excludes_its_windows_and_k():
    nb, sb = _peak_day()
    q = np.array(["ok"] * 96, dtype=object)
    q[30] = "partial"                      # 07:30, inside the NB peak
    _, days = approach_volume(_counts({2: nb, 3: sb}, quality=q), _MV)
    n_ = _row(days, "NB")
    # Every window holding 07:30 is out: 06:45..07:30 starts.  Best left is
    # 07:45..08:45 = 40 + 1 + 1 + 1.
    assert n_["peak_start"] == pd.Timestamp("2025-12-15 07:45", tz=_TZ)
    assert n_["peak_volume"] == 43
    assert n_["n_complete"] == 95 and not n_["complete_day"]
    assert pd.isna(n_["k_factor"])
    assert not np.isnan(n_["d_factor"])
    # total_volume still counts every bin.
    assert n_["total_volume"] == int(nb.sum())


def test_absent_grid_bins_are_incomplete():
    nb, sb = _peak_day()
    keep = np.r_[0:30, 31:96]              # drop 07:30 row entirely
    df = _counts({2: nb, 3: sb}).iloc[keep]
    _, days = approach_volume(df, _MV)
    n_ = _row(days, "NB")
    assert n_["n_bins"] == 96 and n_["n_complete"] == 95
    assert n_["peak_start"] == pd.Timestamp("2025-12-15 07:45", tz=_TZ)
    assert pd.isna(n_["k_factor"])


def test_no_complete_hour_gives_no_peak():
    idx = _grid(n=3)                       # 45 min of data only
    _, days = approach_volume(_counts({2: [5, 5, 5]}, index=idx), {"NBT": [2]})
    n_ = _row(days, "NB")
    assert pd.isna(n_["peak_start"]) and pd.isna(n_["peak_volume"])
    assert pd.isna(n_["k_factor"]) and np.isnan(n_["phf"])
    assert n_["total_volume"] == 15 and n_["n_complete"] == 3


def test_partial_window_has_peak_but_no_k():
    idx = _grid(n=8)                       # 00:00 .. 02:00
    v = [1, 2, 3, 4, 5, 6, 7, 8]
    _, days = approach_volume(_counts({2: v}, index=idx), {"NBT": [2]})
    n_ = _row(days, "NB")
    assert n_["peak_start"] == pd.Timestamp("2025-12-15 01:00", tz=_TZ)
    assert n_["peak_volume"] == 26
    assert pd.isna(n_["k_factor"]) and not n_["complete_day"]


def test_hourly_bins():
    idx = _grid(n=24, bin_len=60)
    v = np.arange(24)
    bins, days = approach_volume(_counts({2: v}, index=idx), {"NBT": [2]}, bin_len=60)
    n_ = _row(days, "NB")
    assert n_["n_bins"] == 24 and n_["complete_day"]
    assert n_["peak_start"] == pd.Timestamp("2025-12-15 23:00", tz=_TZ)
    assert n_["phf"] == pytest.approx(1.0)
    assert bins.loc[bins["direction"] == "NB", "vph"].tolist() == list(map(float, v))


@pytest.mark.parametrize("bad", [0, 7, 45, 90])
def test_bin_len_must_divide_an_hour(bad):
    with pytest.raises(ValueError):
        approach_volume(_counts({2: np.ones(96)}), {"NBT": [2]}, bin_len=bad)


# ---------------------------------------------------------------------------
# Days
# ---------------------------------------------------------------------------

def test_days_are_local_and_windows_do_not_cross_midnight():
    idx = _grid(n=192)                     # 2025-12-15 and 12-16
    nb = np.zeros(192, dtype=int)
    nb[94:98] = 50                         # 23:30 .. 00:15 straddles midnight
    nb[100] = 1
    _, days = approach_volume(_counts({2: nb}, index=idx), {"NBT": [2]})
    d1 = _row(days, "NB", day=date(2025, 12, 15))
    d2 = _row(days, "NB", day=date(2025, 12, 16))
    assert d1["peak_volume"] == 100 and d1["peak_start"] == pd.Timestamp("2025-12-15 23:00", tz=_TZ)
    assert d2["peak_volume"] == 100 and d2["peak_start"] == pd.Timestamp("2025-12-16 00:00", tz=_TZ)
    assert d1["k_factor"] == pytest.approx(1.0)
    assert d2["k_factor"] == pytest.approx(100 / 101)


@pytest.mark.parametrize("day, n_bins", [("2026-03-08", 92), ("2025-11-02", 100)])
def test_dst_days_need_all_their_bins(day, n_bins):
    start = pd.Timestamp(day, tz=_TZ)
    idx = pd.date_range(start, start + pd.DateOffset(days=1), freq="15min",
                        inclusive="left", name="Time")
    assert len(idx) == n_bins
    _, days = approach_volume(_counts({2: np.ones(n_bins)}, index=idx), {"NBT": [2]})
    n_ = _row(days, "NB")
    assert n_["n_bins"] == n_bins and n_["complete_day"]
    assert n_["k_factor"] == pytest.approx(4 / n_bins)


def test_detector_in_both_directions_counts_in_both():
    n = 96
    bins, days = approach_volume(_counts({54: np.full(n, 2)}), {"EBL": [54], "WBR": [54]})
    assert _row(days, "EB")["total_volume"] == 2 * n
    assert _row(days, "WB")["total_volume"] == 2 * n
    assert _row(days, "combined")["total_volume"] == 4 * n


def test_no_quality_column_means_complete():
    _, days = approach_volume(_counts({2: np.ones(96)}), {"NBT": [2]})
    assert _row(days, "NB")["complete_day"]


def test_empty_inputs():
    b, d = approach_volume(pd.DataFrame(), _MV)
    assert list(b.columns) == BIN_SCHEMA and b.empty
    assert list(d.columns) == DAY_SCHEMA and d.empty
    b, d = approach_volume(_counts({2: np.ones(96)}), {"Ped": [2]})
    assert b.empty and d.empty


def test_dtypes():
    nb, sb = _peak_day()
    bins, days = approach_volume(_counts({2: nb, 3: sb}), _MV)
    assert bins["volume"].dtype == np.int64
    assert bins["complete"].dtype == bool
    for c in ("total_volume", "n_bins", "n_complete", "peak_volume",
              "peak_bin_volume", "pair_peak_volume"):
        assert str(days[c].dtype) == "Int64", c
    assert days["complete_day"].dtype == bool
    assert str(days["peak_start"].dt.tz) == _TZ


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2] / "intersections"
_DB315 = _ROOT / "315_US-20-26_Franklin_Rd_and_KCID_Rd" / "315_data.db"
_DB313 = _ROOT / "313_I-84_Ex_25_WB_and_SH-44" / "313_data.db"


def _corpus(db, day):
    if not db.exists():
        pytest.skip("corpus DB not present")
    from atspm.analysis.counts import parse_movements_from_config
    from atspm.data.counts import CountEngine
    from atspm.data.manager import DatabaseManager
    with DatabaseManager(db) as m:
        cfg = m.get_config_at_date(datetime.strptime(day, "%Y-%m-%d"))
    counts = CountEngine(db).vehicle_counts(day, day, bin_len=15, include_detectors=True)
    return approach_volume(counts, parse_movements_from_config(cfg))


def test_corpus_315_monday():
    # 2025-12-15: a whole, clean day.  NB/SB peaks in the morning, the
    # US-20-26 EB/WB arterial in the evening, EB the heavier direction.
    bins, days = _corpus(_DB315, "2025-12-15")
    assert days["complete_day"].all() and (days["n_bins"] == 96).all()
    ns = _row(days, "combined", pair="NB/SB")
    assert ns["peak_start"] == pd.Timestamp("2025-12-15 07:15", tz=_TZ)
    assert ns["peak_volume"] == 488 and ns["total_volume"] == 4372
    assert ns["k_factor"] == pytest.approx(488 / 4372)
    ew = _row(days, "combined", pair="EB/WB")
    assert ew["peak_start"] == pd.Timestamp("2025-12-15 16:30", tz=_TZ)
    assert ew["peak_volume"] == 1458 and ew["total_volume"] == 14206
    eb = _row(days, "EB")
    assert eb["peak_volume"] == 842 and eb["pair_peak_volume"] == 1458
    assert eb["d_factor"] == pytest.approx(842 / 1458)
    assert 0.08 < ew["k_factor"] < 0.12
    assert len(bins) == 96 * 6


def test_corpus_313_one_sided_and_silent_detectors():
    # 313 has no SB movement, and its EB/WB TM channels never log.
    _, days = _corpus(_DB313, "2026-08-10")
    nb = _row(days, "NB")
    assert nb["total_volume"] > 5000 and pd.isna(nb["d_factor"])
    assert 0.07 < nb["k_factor"] < 0.11
    ew = days.loc[days["pair"] == "EB/WB"]
    assert (ew["total_volume"] == 0).all() and ew["k_factor"].isna().all()
