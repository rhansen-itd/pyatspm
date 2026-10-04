"""
Approach Volume (Functional Core)

Pure functions only.  No I/O, no SQL, no side effects.

UDOT's Approach Volume measure (``ApproachVolumeService.cs`` and
``ApproachVolumeReportService.cs``, OpenSourceTransportation/Atspm v5) plots,
for each pair of opposing directions, the hourly volume of each direction
and of both combined, plus each direction's share of the combined volume
(the "D-factor" series), and tabulates the peak hour, peak hour factor,
K-factor and D-factor.

Pinned from the v5 source
-------------------------
* Detectors: on-events (Code 82) only.  UDOT draws one chart per detection
  type, *Advanced Count* (2) and *Lane-by-lane Count* (4), for approaches
  with a protected phase and vehicle lanes.  Here the volume comes from
  the ``TM_*`` count detectors (lane-by-lane count), grouped into
  directions by the label's ``NB``/``SB``/``EB``/``WB`` prefix.  Advanced
  count is not offered: the ``Det_P{N}_Arrival`` detectors are keyed by
  phase, and mapping a phase to a direction would need config that does
  not exist yet.
* Pairs: NB/SB and EB/WB, NB and EB primary (UDOT also pairs NE/SW and
  NW/SE; no corpus label uses a diagonal direction).
* Bins: 15 min by default (UDOT ``binSize`` preset), and ``60`` must be a
  multiple of the bin.  A bin's hourly rate is ``count × 60 / bin_len``.
* Peak hour: the rolling hour, starting at any bin, with the largest
  volume; ties go to the earliest.  Each direction and the combined
  volume has its own.  Peak hour factor = peak hour volume / (largest bin
  in that hour × bins per hour).
* D-factor: a direction's peak hour volume / (that + the opposing
  direction's volume in the same hour).  The D-factor series is, per bin,
  direction / combined.
* K-factor: for the combined row, combined peak hour volume / combined
  total; for a direction, the combined volume in *that direction's* peak
  hour / combined total.

Departures from UDOT
--------------------
* Per local calendar day.  UDOT computes one peak hour and one K over
  whatever window was asked for, so a multi-day window yields one peak
  across days and a K over several days' volume.  Here every summary row
  is one day of the index's timezone.
* K-factor is the design-hour share of the *daily* volume, so it is
  reported only for a complete day (every bin of the local day present
  and complete, DST days included).  UDOT divides by the window total
  whatever its length.
* An opposing direction with no detectors configured (313 has no SB)
  makes the D-factor and D-split NA; UDOT reports 1.0.  A zero
  denominator gives NA, where UDOT reports 0.
* A detector configured in both directions of a pair counts in both,
  as configured; :func:`direction_detectors` callers should warn.

Gap Marker Rule
---------------
Bin completeness comes from the ``data_quality`` column that
``CountEngine`` attaches: ``"ok"`` only when the bin is fully covered by
ingested data and holds no gap marker (``event_code == -1``).  Any other
label, and any grid bin absent from the input, makes the bin incomplete.
An incomplete bin keeps its raw ``volume`` but has NA ``vph`` and
``d_split``; a peak-hour window is considered only when all of its bins
are complete, and a day with an incomplete bin has no K-factor.

Package Location: src/atspm/analysis/approach_volume.py
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .call_service import _ratio

DEFAULT_BIN_LEN: int = 15

PAIRS: Tuple[Tuple[str, str], ...] = (("NB", "SB"), ("EB", "WB"))

COMBINED: str = "combined"

BIN_SCHEMA = ["time", "pair", "direction", "volume", "vph", "d_split", "complete"]

DAY_SCHEMA = [
    "date", "pair", "direction", "total_volume", "n_bins", "n_complete",
    "complete_day", "peak_start", "peak_volume", "peak_bin_volume", "phf",
    "pair_peak_volume", "k_factor", "d_factor",
]

_DIRECTION_RE = re.compile(r"^(NB|SB|EB|WB)")


def direction_detectors(movements: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Group ``TM_*`` movement detectors by approach direction.

    Args:
        movements: ``counts.parse_movements_from_config`` output, movement
            label → detector IDs (e.g. ``{"EBL": [25], "EBT": [26, 27]}``).

    Returns:
        Direction (``"NB"``, ``"SB"``, ``"EB"``, ``"WB"``) → sorted distinct
        detector IDs.  Labels without a direction prefix are skipped; see
        :func:`unparsed_movements`.
    """
    out: Dict[str, set] = {}
    for label, dets in movements.items():
        m = _DIRECTION_RE.match(label)
        if m:
            out.setdefault(m.group(1), set()).update(int(d) for d in dets)
    return {d: sorted(s) for d, s in out.items()}


def unparsed_movements(movements: Dict[str, List[int]]) -> List[str]:
    """Movement labels that carry no ``NB``/``SB``/``EB``/``WB`` prefix."""
    return sorted(label for label in movements if not _DIRECTION_RE.match(label))


def approach_volume(
    counts: pd.DataFrame,
    movements: Dict[str, List[int]],
    bin_len: int = DEFAULT_BIN_LEN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Directional volumes, peak hour, K-factor and D-factor per day.

    Args:
        counts: Binned detector counts indexed by tz-aware bin start, as
            ``CountEngine.vehicle_counts(bin_len=..., include_detectors=True)``
            returns them: one integer-named column per detector holding raw
            Code 82 counts (not hourly), plus an optional ``data_quality``
            column (``"ok"`` = complete).  Without ``data_quality`` every
            row present is complete.  Other columns are ignored, and a
            configured detector with no column counted zero.
        movements: Movement label → detector IDs (``TM_*`` config).
        bin_len: Bin width in minutes; must divide 60 and match *counts*.

    Returns:
        ``(bins, days)``.

        *bins* (:data:`BIN_SCHEMA`): one row per input bin, pair and
        direction (each configured direction of the pair, then
        ``"combined"``).  ``volume`` is the raw count (int), ``vph`` the
        hourly rate (NA when incomplete), ``d_split`` direction / combined
        (NA for the combined row, an incomplete bin, a zero combined bin or
        an unconfigured opposing direction).

        *days* (:data:`DAY_SCHEMA`): one row per local day of the index,
        pair and direction.  ``n_bins`` is the bin count of the local day
        (92/96/100 at 15 min), ``n_complete`` its complete bins;
        ``peak_start`` is NaT and the peak columns NA when no hour of
        complete bins exists; ``pair_peak_volume`` is the combined volume
        in this row's peak hour; ``k_factor`` needs ``complete_day``;
        ``d_factor`` is NA for the combined row.

    Raises:
        ValueError: *bin_len* is not a positive divisor of 60.
    """
    bin_len = int(bin_len)
    if bin_len <= 0 or 60 % bin_len:
        raise ValueError(f"bin_len must divide 60, got {bin_len}")

    dir_dets = direction_detectors(movements)
    pairs = [p for p in PAIRS if p[0] in dir_dets or p[1] in dir_dets]
    if counts.empty or not pairs:
        return _empty(BIN_SCHEMA), _empty(DAY_SCHEMA)

    counts = counts.sort_index()
    idx = counts.index
    if "data_quality" in counts.columns:
        complete = counts["data_quality"].eq("ok").to_numpy()
    else:
        complete = np.ones(len(counts), dtype=bool)

    vol = {
        d: counts.reindex(columns=dets).fillna(0).to_numpy(dtype=np.int64).sum(axis=1)
        for d, dets in dir_dets.items()
    }

    k = 60 // bin_len
    bin_frames, day_rows = [], []
    for primary, opposing in pairs:
        both = primary in vol and opposing in vol
        names = [d for d in (primary, opposing) if d in vol]
        series = {d: vol[d] for d in names}
        series[COMBINED] = sum(vol[d] for d in names)
        pair = f"{primary}/{opposing}"

        for d, v in series.items():
            split = (_ratio(v, series[COMBINED]) if both and d != COMBINED
                     else np.full(len(v), np.nan))
            bin_frames.append(pd.DataFrame({
                "time": idx,
                "pair": pair,
                "direction": d,
                "volume": v,
                "vph": np.where(complete, v * (60.0 / bin_len), np.nan),
                "d_split": np.where(complete, split, np.nan),
                "complete": complete,
            }))

        day_rows.extend(_day_rows(idx, complete, series, names, both, pair, bin_len, k))

    bins = pd.concat(bin_frames, ignore_index=True)[BIN_SCHEMA]
    days = pd.DataFrame(day_rows, columns=DAY_SCHEMA)
    for c in ("total_volume", "n_bins", "n_complete", "peak_volume",
              "peak_bin_volume", "pair_peak_volume"):
        days[c] = days[c].astype("Int64")
    days["complete_day"] = days["complete_day"].astype(bool)
    days["peak_start"] = pd.to_datetime(days["peak_start"])
    return bins, days


def _day_rows(idx, complete, series, names, both, pair, bin_len, k):
    """Summary rows for every local day of *idx*, one pair."""
    rows = []
    day_keys = idx.normalize()
    for day_start in day_keys.unique():
        next_start = day_start + pd.DateOffset(days=1)
        grid = pd.date_range(day_start, next_start, freq=f"{bin_len}min",
                             inclusive="left")
        sel = day_keys == day_start
        pos = grid.get_indexer(idx[sel])
        ok = pos >= 0
        grid_complete = np.zeros(len(grid), dtype=bool)
        grid_complete[pos[ok]] = complete[sel][ok]
        n_complete = int(grid_complete.sum())
        complete_day = n_complete == len(grid)

        # A window starting at grid position i covers bins i .. i+k-1.
        if len(grid) >= k:
            full = _window_sum(grid_complete.astype(np.int64), k) == k
        else:
            full = np.zeros(0, dtype=bool)

        grid_vol = {}
        for d, v in series.items():
            g = np.zeros(len(grid), dtype=np.int64)
            g[pos[ok]] = v[sel][ok]
            grid_vol[d] = g
        comb_win = _window_sum(grid_vol[COMBINED], k) if len(full) else full
        comb_total = int(grid_vol[COMBINED].sum())

        for d in names + [COMBINED]:
            g = grid_vol[d]
            row = {
                "date": day_start.date(), "pair": pair, "direction": d,
                "total_volume": int(g.sum()), "n_bins": len(grid),
                "n_complete": n_complete, "complete_day": complete_day,
                "peak_start": pd.NaT, "peak_volume": pd.NA,
                "peak_bin_volume": pd.NA, "phf": np.nan,
                "pair_peak_volume": pd.NA, "k_factor": np.nan, "d_factor": np.nan,
            }
            if full.any():
                win = _window_sum(g, k)
                cand = np.where(full, win, -1)
                i = int(np.argmax(cand))  # first maximum = earliest hour
                peak = int(win[i])
                peak_bin = int(g[i:i + k].max())
                pair_peak = int(comb_win[i])
                row.update(
                    peak_start=grid[i], peak_volume=peak, peak_bin_volume=peak_bin,
                    phf=peak / (peak_bin * k) if peak_bin > 0 else np.nan,
                    pair_peak_volume=pair_peak,
                )
                if complete_day and comb_total > 0:
                    row["k_factor"] = pair_peak / comb_total
                if both and d != COMBINED and pair_peak > 0:
                    row["d_factor"] = peak / pair_peak
            rows.append(row)
    return rows


def _window_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Sums of every *k* consecutive elements (``len(x) - k + 1`` values)."""
    c = np.concatenate(([0], np.cumsum(x)))
    return c[k:] - c[:-k]


def _empty(schema: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=schema)
