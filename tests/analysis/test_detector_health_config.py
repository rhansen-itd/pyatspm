"""Goldens for the S-D4 int_cfg WD-key parsers (functional core, pure).

These settle the config conventions owner-agreed 2026-10-03 (design decision 1):
reboot/PM/AM windows, the single-cell WD_Units grouping, WD_Ignore, and
WD_Thresholds overrides, plus the severity/exit-code helpers the CLI uses.
"""

import math

import pandas as pd
import pytest

from atspm.analysis.detector_activity import UDOT_AM_WINDOW
from atspm.analysis.detector_health import (
    DEFAULT_THRESHOLDS,
    FINDINGS_SCHEMA,
    apply_ignore,
    filter_min_severity,
    severity_exit_code,
    wd_ignore,
    wd_profile_windows,
    wd_reboot_windows,
    wd_thresholds,
    wd_units,
)


# --------------------------------------------------------------------------- #
# Reboot / profile windows
# --------------------------------------------------------------------------- #

class TestWindows:
    def test_reboot_multi_with_midnight_end(self):
        cfg = {"WD_Reboot": "23:58-24:00, 00:00-00:03 ,01:54-01:58"}
        assert wd_reboot_windows(cfg) == {
            "reboot_0": ("23:58", "24:00"),
            "reboot_1": ("00:00", "00:03"),
            "reboot_2": ("01:54", "01:58"),
        }

    def test_reboot_absent_or_blank_is_none(self):
        assert wd_reboot_windows({}) is None
        assert wd_reboot_windows({"WD_Reboot": "   "}) is None
        assert wd_reboot_windows(None) is None

    def test_reboot_bad_range_raises(self):
        with pytest.raises(ValueError):
            wd_reboot_windows({"WD_Reboot": "2600"})

    def test_profile_windows_default_am_only(self):
        w = wd_profile_windows({})
        assert w["day"] == ("00:00", "24:00")
        assert w["am"] == UDOT_AM_WINDOW
        assert "pm" not in w

    def test_profile_windows_am_override_and_pm(self):
        w = wd_profile_windows({"WD_AM": "02:00-04:00", "WD_PM": "16:30-17:30"})
        assert w["am"] == ("02:00", "04:00")
        assert w["pm"] == ("16:30", "17:30")


# --------------------------------------------------------------------------- #
# Units: single-cell, grouped by type (owner-chosen format B)
# --------------------------------------------------------------------------- #

class TestUnits:
    def test_parse_multi_type_ranges_and_lists(self):
        cfg = {"WD_Units": "evo:[17-20],[21,23]; currux:[30-31]"}
        units, types = wd_units(cfg)
        assert units == {
            "evo_0": [17, 18, 19, 20],
            "evo_1": [21, 23],
            "currux_0": [30, 31],
        }
        assert types == {"evo_0": "evo", "evo_1": "evo", "currux_0": "currux"}

    def test_index_runs_across_segments_of_same_type(self):
        units, _ = wd_units({"WD_Units": "evo:[1-2]; evo:[3-4]"})
        assert set(units) == {"evo_0", "evo_1"}

    def test_absent_is_none_fallback(self):
        assert wd_units({}) == (None, {})
        assert wd_units({"WD_Units": ""}) == (None, {})

    def test_segment_without_type_raises(self):
        with pytest.raises(ValueError):
            wd_units({"WD_Units": "[17-20]"})

    def test_type_without_groups_raises(self):
        with pytest.raises(ValueError):
            wd_units({"WD_Units": "evo:17-20"})


# --------------------------------------------------------------------------- #
# Ignore list
# --------------------------------------------------------------------------- #

class TestIgnore:
    def test_parse_pairs(self):
        got = wd_ignore({"WD_Ignore": "52:StuckOn, 60:ConfiguredSilent ,-1:MaxOut"})
        assert got == [(52, "StuckOn"), (60, "ConfiguredSilent"), (-1, "MaxOut")]

    def test_absent_is_empty(self):
        assert wd_ignore({}) == []

    def test_bad_entry_raises(self):
        with pytest.raises(ValueError):
            wd_ignore({"WD_Ignore": "52"})

    def _findings(self):
        rows = [
            ("2026-01-10", "day", float("nan"), 52, pd.NA, "stop_bar", "StuckOn", "high", 1.0, 2.0, "m"),
            ("2026-01-10", "day", float("nan"), 60, pd.NA, "tm", "ConfiguredSilent", "high", 0.0, 0.0, "m"),
            ("2026-01-10", "day", float("nan"), 52, pd.NA, "stop_bar", "Chatter", "low", 0.1, 0.0, "m"),
        ]
        df = pd.DataFrame(rows, columns=FINDINGS_SCHEMA)
        return df.astype({"detector": "int64", "phase": "Int64"})

    def test_apply_ignore_drops_matching_pairs_case_insensitive(self):
        df = self._findings()
        out = apply_ignore(df, [(52, "stuckon"), (60, "ConfiguredSilent")])
        # 52:StuckOn and 60:ConfiguredSilent gone; 52:Chatter stays.
        assert list(zip(out["detector"], out["rule"])) == [(52, "Chatter")]

    def test_apply_ignore_empty_is_noop(self):
        df = self._findings()
        assert len(apply_ignore(df, [])) == len(df)


# --------------------------------------------------------------------------- #
# Thresholds overrides
# --------------------------------------------------------------------------- #

class TestThresholds:
    def test_absent_returns_base(self):
        assert wd_thresholds({}) is DEFAULT_THRESHOLDS

    def test_scalar_int_and_float_coercion(self):
        th = wd_thresholds({"WD_Thresholds": "low_hits_min=10,chatter_peer_ratio=2"})
        assert th.low_hits_min == 10 and isinstance(th.low_hits_min, int)
        assert th.chatter_peer_ratio == 2.0 and isinstance(th.chatter_peer_ratio, float)
        # Untouched fields keep the calibrated default.
        assert th.burst_min_channels == DEFAULT_THRESHOLDS.burst_min_channels

    def test_dotted_mapping_field_merges(self):
        th = wd_thresholds({"WD_Thresholds": "stuck_on_s.arrival=600"})
        assert th.stuck_on_s["arrival"] == 600.0
        # Other roles survive the merge.
        assert th.stuck_on_s["occupancy"] == DEFAULT_THRESHOLDS.stuck_on_s["occupancy"]

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError):
            wd_thresholds({"WD_Thresholds": "bogus=1"})

    def test_dotted_on_scalar_field_raises(self):
        with pytest.raises(ValueError):
            wd_thresholds({"WD_Thresholds": "low_hits_min.x=1"})


# --------------------------------------------------------------------------- #
# Severity filter and exit code
# --------------------------------------------------------------------------- #

def _sev_frame(severities):
    rows = [
        (f"2026-01-1{i}", "day", float("nan"), i, pd.NA, "r", "Rule", s, 0.0, 0.0, "m")
        for i, s in enumerate(severities)
    ]
    return pd.DataFrame(rows, columns=FINDINGS_SCHEMA).astype(
        {"detector": "int64", "phase": "Int64"}
    )


class TestSeverity:
    def test_filter_min_severity_default_low_hides_info(self):
        df = _sev_frame(["info", "low", "high"])
        out = filter_min_severity(df, "low")
        assert set(out["severity"]) == {"low", "high"}

    def test_filter_min_severity_high_only(self):
        df = _sev_frame(["info", "low", "high"])
        assert list(filter_min_severity(df, "high")["severity"]) == ["high"]

    @pytest.mark.parametrize(
        "severities,floor,expected",
        [
            ([], "low", 0),
            (["info"], "low", 0),         # info never raises above 0
            (["info"], "info", 0),
            (["info", "low"], "low", 1),
            (["low", "high"], "low", 2),
            (["high"], "info", 2),
        ],
    )
    def test_exit_code(self, severities, floor, expected):
        assert severity_exit_code(_sev_frame(severities), floor) == expected
