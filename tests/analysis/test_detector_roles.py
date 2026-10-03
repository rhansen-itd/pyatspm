# Golden tests for the detector role table (UDOT roadmap S-D0).
#
# Opus-written; implementations must make these pass without editing them.
# The 201 and 315 configs are the real rows as DatabaseManager.get_config_at_date
# returns them (201 on 2026-06-21, after the 2026-06-01 period start; 315 on
# 2025-12-15, with the owner-confirmed presence zones).

import pandas as pd
import pytest

from atspm.analysis.detector_roles import (
    PHASE_ROLES,
    ROLE_SCHEMA,
    ROLES,
    detector_sets,
    parse_detector_roles,
)

NA = None

CFG_201 = {
    "TM_EBT": "63", "TM_WBL": "40", "TM_WBT": "41", "TM_WBR": "42",
    "TM_NBT": "60", "TM_NBR": "61", "TM_SBT": "43", "TM_SBR": "45",
    "TM_Exclusions": "[]",
    "Det_P2_Arrival": "49", "Det_P6_Arrival": "36",
    "Det_P2_Occupancy": "46", "Det_P3_Occupancy": "53", "Det_P4_Occupancy": "39",
    "Det_P6_Occupancy": "33", "Det_P7_Occupancy": "38", "Det_P8_Occupancy": "51",
    "Det_P2_Stop_Bar": "60", "Det_P3_Stop_Bar": "64", "Det_P4_Stop_Bar": "41",
    "Det_P6_Stop_Bar": "43", "Det_P7_Stop_Bar": "40", "Det_P8_Stop_Bar": "63",
    "Det_P2_Pairs": "[[46,17]]", "Det_P4_Pairs": "[[39,22]]",
    "Det_P6_Pairs": "[[33,26]]", "Det_P7_Pairs": "[[38,24]]",
    "Det_P8_Pairs": "[[42,31]]",
    "WD_Sensor1": None, "WD_Sensor2": None, "WD_Sensor3": None,
    "Det_P1_Pairs": "[[43,29]]", "Det_P3_Pairs": "[[41,30]]",
}

# 201's 2020-01-01 → 2026-06-01 period: watchdog zones filled, P1/P3 pairs blank.
CFG_201_EARLY = {
    **CFG_201,
    "WD_Sensor1": "56", "WD_Sensor2": "57", "WD_Sensor3": "58",
    "Det_P1_Pairs": None, "Det_P3_Pairs": None,
}

CFG_315 = {
    "TM_EBL": "25", "TM_EBT": "26,27,28", "TM_EBR": "29", "TM_WBL": "17",
    "TM_WBT": "18,19,20", "TM_WBR": "21", "TM_NBL": "22", "TM_NBT": "23",
    "TM_NBR": "24", "TM_SBL": "30", "TM_SBT": "31", "TM_SBR": "32",
    "TM_Exclusions": "[]",
    "Det_P2_Arrival": "54,55,56", "Det_P6_Arrival": "38,39,40",
    "Det_P1_Stop_Bar": "17", "Det_P2_Stop_Bar": "26,27,28", "Det_P3_Stop_Bar": "22",
    "Det_P4_Stop_Bar": "31", "Det_P5_Stop_Bar": "25", "Det_P6_Stop_Bar": "18,19,20",
    "Det_P7_Stop_Bar": "30", "Det_P8_Stop_Bar": "23",
    "Det_P2_Occupancy": "50,51,52", "Det_P6_Occupancy": "34,35,36",
}

# (detector, phase, role, movement, partner, key)
EXPECTED_201 = [
    (49, 2, "arrival", NA, NA, "Det_P2_Arrival"),
    (36, 6, "arrival", NA, NA, "Det_P6_Arrival"),
    (60, 2, "stop_bar", "NBT", NA, "Det_P2_Stop_Bar"),
    (64, 3, "stop_bar", NA, NA, "Det_P3_Stop_Bar"),
    (41, 4, "stop_bar", "WBT", NA, "Det_P4_Stop_Bar"),
    (43, 6, "stop_bar", "SBT", NA, "Det_P6_Stop_Bar"),
    (40, 7, "stop_bar", "WBL", NA, "Det_P7_Stop_Bar"),
    (63, 8, "stop_bar", "EBT", NA, "Det_P8_Stop_Bar"),
    (46, 2, "occupancy", NA, NA, "Det_P2_Occupancy"),
    (53, 3, "occupancy", NA, NA, "Det_P3_Occupancy"),
    (39, 4, "occupancy", NA, NA, "Det_P4_Occupancy"),
    (33, 6, "occupancy", NA, NA, "Det_P6_Occupancy"),
    (38, 7, "occupancy", NA, NA, "Det_P7_Occupancy"),
    (51, 8, "occupancy", NA, NA, "Det_P8_Occupancy"),
    (29, 1, "pairs", NA, 43, "Det_P1_Pairs"),
    (43, 1, "pairs", "SBT", 29, "Det_P1_Pairs"),
    (17, 2, "pairs", NA, 46, "Det_P2_Pairs"),
    (46, 2, "pairs", NA, 17, "Det_P2_Pairs"),
    (30, 3, "pairs", NA, 41, "Det_P3_Pairs"),
    (41, 3, "pairs", "WBT", 30, "Det_P3_Pairs"),
    (22, 4, "pairs", NA, 39, "Det_P4_Pairs"),
    (39, 4, "pairs", NA, 22, "Det_P4_Pairs"),
    (26, 6, "pairs", NA, 33, "Det_P6_Pairs"),
    (33, 6, "pairs", NA, 26, "Det_P6_Pairs"),
    (24, 7, "pairs", NA, 38, "Det_P7_Pairs"),
    (38, 7, "pairs", NA, 24, "Det_P7_Pairs"),
    (31, 8, "pairs", NA, 42, "Det_P8_Pairs"),
    (42, 8, "pairs", "WBR", 31, "Det_P8_Pairs"),
    (63, NA, "tm", "EBT", NA, "TM_EBT"),
    (61, NA, "tm", "NBR", NA, "TM_NBR"),
    (60, NA, "tm", "NBT", NA, "TM_NBT"),
    (45, NA, "tm", "SBR", NA, "TM_SBR"),
    (43, NA, "tm", "SBT", NA, "TM_SBT"),
    (40, NA, "tm", "WBL", NA, "TM_WBL"),
    (42, NA, "tm", "WBR", NA, "TM_WBR"),
    (41, NA, "tm", "WBT", NA, "TM_WBT"),
]

EXPECTED_315 = [
    (54, 2, "arrival", NA, NA, "Det_P2_Arrival"),
    (55, 2, "arrival", NA, NA, "Det_P2_Arrival"),
    (56, 2, "arrival", NA, NA, "Det_P2_Arrival"),
    (38, 6, "arrival", NA, NA, "Det_P6_Arrival"),
    (39, 6, "arrival", NA, NA, "Det_P6_Arrival"),
    (40, 6, "arrival", NA, NA, "Det_P6_Arrival"),
    (17, 1, "stop_bar", "WBL", NA, "Det_P1_Stop_Bar"),
    (26, 2, "stop_bar", "EBT", NA, "Det_P2_Stop_Bar"),
    (27, 2, "stop_bar", "EBT", NA, "Det_P2_Stop_Bar"),
    (28, 2, "stop_bar", "EBT", NA, "Det_P2_Stop_Bar"),
    (22, 3, "stop_bar", "NBL", NA, "Det_P3_Stop_Bar"),
    (31, 4, "stop_bar", "SBT", NA, "Det_P4_Stop_Bar"),
    (25, 5, "stop_bar", "EBL", NA, "Det_P5_Stop_Bar"),
    (18, 6, "stop_bar", "WBT", NA, "Det_P6_Stop_Bar"),
    (19, 6, "stop_bar", "WBT", NA, "Det_P6_Stop_Bar"),
    (20, 6, "stop_bar", "WBT", NA, "Det_P6_Stop_Bar"),
    (30, 7, "stop_bar", "SBL", NA, "Det_P7_Stop_Bar"),
    (23, 8, "stop_bar", "NBT", NA, "Det_P8_Stop_Bar"),
    (50, 2, "occupancy", NA, NA, "Det_P2_Occupancy"),
    (51, 2, "occupancy", NA, NA, "Det_P2_Occupancy"),
    (52, 2, "occupancy", NA, NA, "Det_P2_Occupancy"),
    (34, 6, "occupancy", NA, NA, "Det_P6_Occupancy"),
    (35, 6, "occupancy", NA, NA, "Det_P6_Occupancy"),
    (36, 6, "occupancy", NA, NA, "Det_P6_Occupancy"),
    (25, NA, "tm", "EBL", NA, "TM_EBL"),
    (29, NA, "tm", "EBR", NA, "TM_EBR"),
    (26, NA, "tm", "EBT", NA, "TM_EBT"),
    (27, NA, "tm", "EBT", NA, "TM_EBT"),
    (28, NA, "tm", "EBT", NA, "TM_EBT"),
    (22, NA, "tm", "NBL", NA, "TM_NBL"),
    (24, NA, "tm", "NBR", NA, "TM_NBR"),
    (23, NA, "tm", "NBT", NA, "TM_NBT"),
    (30, NA, "tm", "SBL", NA, "TM_SBL"),
    (32, NA, "tm", "SBR", NA, "TM_SBR"),
    (31, NA, "tm", "SBT", NA, "TM_SBT"),
    (17, NA, "tm", "WBL", NA, "TM_WBL"),
    (21, NA, "tm", "WBR", NA, "TM_WBR"),
    (18, NA, "tm", "WBT", NA, "TM_WBT"),
    (19, NA, "tm", "WBT", NA, "TM_WBT"),
    (20, NA, "tm", "WBT", NA, "TM_WBT"),
]


def _rows(df: pd.DataFrame):
    """Rows as tuples with every missing value normalised to None."""
    out = []
    for rec in df[ROLE_SCHEMA].astype(object).itertuples(index=False):
        out.append(tuple(None if pd.isna(v) else v for v in rec))
    return out


class TestSchema:
    def test_columns_and_dtypes(self):
        df = parse_detector_roles(CFG_315)
        assert list(df.columns) == ROLE_SCHEMA
        assert df["detector"].dtype == "int64"
        assert df["phase"].dtype == "Int64"
        assert df["partner"].dtype == "Int64"
        for col in ("role", "movement", "key"):
            assert pd.api.types.is_string_dtype(df[col])

    def test_empty_config_keeps_schema(self):
        df = parse_detector_roles({})
        assert df.empty
        assert list(df.columns) == ROLE_SCHEMA
        assert df["phase"].dtype == "Int64"

    def test_index_is_range(self):
        df = parse_detector_roles(CFG_201)
        assert list(df.index) == list(range(len(df)))

    def test_roles_constant(self):
        assert ROLES == ("arrival", "stop_bar", "occupancy", "pairs", "tm", "watchdog")
        assert PHASE_ROLES == {"arrival", "stop_bar", "occupancy", "pairs"}


class TestRealConfigs:
    def test_201_golden(self):
        assert _rows(parse_detector_roles(CFG_201)) == EXPECTED_201

    def test_315_golden(self):
        assert _rows(parse_detector_roles(CFG_315)) == EXPECTED_315

    def test_201_early_period_has_watchdog_rows_and_no_p1_p3_pairs(self):
        rows = _rows(parse_detector_roles(CFG_201_EARLY))
        assert rows[-3:] == [
            (56, None, "watchdog", None, None, "WD_Sensor1"),
            (57, None, "watchdog", None, None, "WD_Sensor2"),
            (58, None, "watchdog", None, None, "WD_Sensor3"),
        ]
        pairs_phases = {r[1] for r in rows if r[2] == "pairs"}
        assert pairs_phases == {2, 4, 6, 7, 8}

    def test_201_stop_bar_sets(self):
        sets = detector_sets(parse_detector_roles(CFG_201), "stop_bar")
        assert sets == {2: {60}, 3: {64}, 4: {41}, 6: {43}, 7: {40}, 8: {63}}

    def test_315_presence_sets(self):
        sets = detector_sets(parse_detector_roles(CFG_315), "occupancy")
        assert sets == {2: {50, 51, 52}, 6: {34, 35, 36}}

    def test_config_order_does_not_matter(self):
        rev = dict(reversed(list(CFG_201.items())))
        assert _rows(parse_detector_roles(rev)) == EXPECTED_201


class TestStopBarSpellings:
    @pytest.mark.parametrize("key", ["Det_P2_Stop_Bar", "Det_P2_Stopbar"])
    def test_either_spelling_is_stop_bar(self, key):
        rows = _rows(parse_detector_roles({key: "5,6"}))
        assert rows == [(5, 2, "stop_bar", None, None, key),
                        (6, 2, "stop_bar", None, None, key)]

    def test_both_spellings_union_without_duplicates(self):
        df = parse_detector_roles({"Det_P2_Stopbar": "5,6", "Det_P2_Stop_Bar": "6,7"})
        assert detector_sets(df, "stop_bar") == {2: {5, 6, 7}}
        assert len(df) == 3
        # The shared detector keeps the first key in sort order.
        assert df.loc[df["detector"] == 6, "key"].tolist() == ["Det_P2_Stop_Bar"]

    def test_313_style_blank_stopbar_family_is_ignored(self):
        cfg = {f"Det_P{p}_Stopbar": None for p in range(1, 9)}
        cfg.update({f"Det_P{p}_Stop_Bar": "" for p in range(1, 9)})
        cfg["Det_P1_Occupancy"] = "43"
        assert _rows(parse_detector_roles(cfg)) == [
            (43, 1, "occupancy", None, None, "Det_P1_Occupancy"),
        ]


class TestValueParsing:
    def test_blank_values(self):
        cfg = {"Det_P2_Arrival": None, "Det_P4_Arrival": float("nan"),
               "Det_P6_Arrival": "", "Det_P8_Arrival": "  ", "TM_EBT": None}
        assert parse_detector_roles(cfg).empty

    def test_whitespace_and_junk_tokens(self):
        rows = _rows(parse_detector_roles({"Det_P2_Arrival": " 3 , x, 4,,-1 "}))
        assert [r[0] for r in rows] == [3, 4]

    def test_integer_value(self):
        rows = _rows(parse_detector_roles({"Det_P2_Arrival": 7}))
        assert rows == [(7, 2, "arrival", None, None, "Det_P2_Arrival")]

    def test_repeated_detector_in_one_key(self):
        assert len(parse_detector_roles({"Det_P2_Arrival": "3,3,4"})) == 2

    def test_unknown_keys_ignored(self):
        cfg = {"Det_P2_Presence": "1", "Det_Px_Arrival": "2", "Det_P2_arrival": "3",
               "TM_Exclusions": '[{"detector": 5}]', "WD_Ignore": "7:stuck_on",
               "WD_Reboot": "00:00", "RB_R1": "1,2|3,4", "Meta_x": "9"}
        assert parse_detector_roles(cfg).empty


class TestPairs:
    def test_flat_pair(self):
        rows = _rows(parse_detector_roles({"Det_P2_Pairs": "[42,3]"}))
        assert rows == [(3, 2, "pairs", None, 42, "Det_P2_Pairs"),
                        (42, 2, "pairs", None, 3, "Det_P2_Pairs")]

    def test_detector_in_two_pairs_and_repeat_dropped(self):
        rows = _rows(parse_detector_roles({"Det_P2_Pairs": "[[42,3],[42,3],[42,2]]"}))
        assert rows == [
            (2, 2, "pairs", None, 42, "Det_P2_Pairs"),
            (3, 2, "pairs", None, 42, "Det_P2_Pairs"),
            (42, 2, "pairs", None, 2, "Det_P2_Pairs"),
            (42, 2, "pairs", None, 3, "Det_P2_Pairs"),
        ]

    @pytest.mark.parametrize("raw", ["not json", "{}", "[[1,2,3]]", '[["a","b"]]', "[1]"])
    def test_malformed_pairs_ignored(self, raw):
        assert parse_detector_roles({"Det_P2_Pairs": raw}).empty

    def test_pairs_detector_set(self):
        sets = detector_sets(parse_detector_roles(CFG_201), "pairs")
        assert sets[2] == {46, 17}
        assert sets[1] == {43, 29}


class TestMovement:
    def test_detector_in_two_tm_keys_has_no_movement_elsewhere(self):
        cfg = {"TM_EBT": "5", "TM_EBR": "5", "Det_P2_Stop_Bar": "5"}
        rows = _rows(parse_detector_roles(cfg))
        assert (5, 2, "stop_bar", None, None, "Det_P2_Stop_Bar") in rows
        assert {r[3] for r in rows if r[2] == "tm"} == {"EBT", "EBR"}

    def test_tm_rows_have_no_phase(self):
        df = parse_detector_roles(CFG_315)
        assert df.loc[df["role"] == "tm", "phase"].isna().all()


class TestDetectorSets:
    @pytest.mark.parametrize("role", ["tm", "watchdog", "presence"])
    def test_non_phase_role_raises(self, role):
        with pytest.raises(ValueError):
            detector_sets(parse_detector_roles(CFG_201), role)

    def test_empty_table(self):
        assert detector_sets(parse_detector_roles({}), "arrival") == {}

    def test_values_are_plain_ints(self):
        sets = detector_sets(parse_detector_roles(CFG_315), "arrival")
        assert all(type(p) is int for p in sets)
        assert all(type(d) is int for s in sets.values() for d in s)
        assert all(isinstance(s, frozenset) for s in sets.values())
