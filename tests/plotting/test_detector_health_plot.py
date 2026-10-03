"""Opus-written contract for the S-D4 detector-health heatmap (functional core).

The plot module is built by the delegated run; this import fails until it
exists, which is the red state this gate starts in.  Assertions are structural
and tolerant of reasonable implementation choices, but they pin the pieces the
spec requires: a detector x day heatmap, counts normalized to each detector's
own median, findings overlaid, a metadata-driven title, graceful missing names,
and a valid empty figure on empty input.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atspm.analysis.detector_health import FINDINGS_SCHEMA
# Built by the delegated S-D4 run.
from atspm.plotting.detector_health import plot_detector_health


def _profile(counts_by_det_day, window="day"):
    """counts_by_det_day: {detector: {date_str: n_act}} -> minimal profile frame."""
    rows = []
    for det, by_day in counts_by_det_day.items():
        for d, n in by_day.items():
            rows.append({
                "date": dt.date.fromisoformat(d), "window": window, "detector": det,
                "n_act": n, "bin_s": 86400.0, "observed_s": 86400.0,
            })
    return pd.DataFrame(rows)


def _findings(rows):
    df = pd.DataFrame(rows, columns=FINDINGS_SCHEMA)
    return df.astype({"detector": "int64", "phase": "Int64"})


META = {"major_road_route": "US-1", "major_road_name": "Main",
        "minor_road_route": "SR-2", "minor_road_name": "Cross"}


def _heatmaps(fig):
    return [t for t in fig.data if isinstance(t, go.Heatmap)]


class TestHeatmap:
    def test_returns_figure_with_heatmap(self):
        prof = _profile({52: {"2026-01-10": 100, "2026-01-11": 100}})
        fig = plot_detector_health(prof, _findings([]), META)
        assert isinstance(fig, go.Figure)
        assert _heatmaps(fig), "expected a Heatmap trace (detector x day)"

    def test_axes_are_detector_by_day(self):
        prof = _profile({52: {"2026-01-10": 100}, 53: {"2026-01-10": 50}})
        hm = _heatmaps(plot_detector_health(prof, _findings([]), META))[0]
        # two detectors on one axis, one day on the other
        dims = {len(np.atleast_1d(hm.x)), len(np.atleast_1d(hm.y))}
        assert dims == {1, 2}

    def test_counts_normalized_to_each_detectors_own_median(self):
        # Constant counts for a detector -> its row normalizes to ~1.0 everywhere.
        prof = _profile({52: {"2026-01-10": 100, "2026-01-11": 100, "2026-01-12": 100}})
        hm = _heatmaps(plot_detector_health(prof, _findings([]), META))[0]
        z = np.array(hm.z, dtype=float)
        assert np.allclose(z[~np.isnan(z)], 1.0)

    def test_findings_overlaid_when_present(self):
        prof = _profile({52: {"2026-01-10": 100, "2026-01-11": 0}})
        find = _findings([
            ("2026-01-11", "day", float("nan"), 52, pd.NA, "stop_bar",
             "ConfiguredSilent", "high", 0.0, 0.0, "silent"),
        ])
        fig = plot_detector_health(prof, find, META)
        # An overlay trace beyond the heatmap marks the flagged cell.
        assert len(fig.data) > len(_heatmaps(fig))

    def test_title_uses_metadata_location(self):
        prof = _profile({52: {"2026-01-10": 100}})
        fig = plot_detector_health(prof, _findings([]), META)
        title = (fig.layout.title.text or "")
        assert "Main" in title and "Cross" in title

    def test_missing_road_names_handled(self):
        prof = _profile({52: {"2026-01-10": 100}})
        fig = plot_detector_health(prof, _findings([]), {"major_road_route": "US-1"})
        assert isinstance(fig, go.Figure)  # no crash on absent names

    def test_empty_profile_returns_valid_figure(self):
        fig = plot_detector_health(_profile({}), _findings([]), META)
        assert isinstance(fig, go.Figure)
