"""Adversarial edge-case tests for the DatZ decoder (Functional Core).

Targets: src/atspm/analysis/decoders.py — parse_datz_bytes and friends.
Pure functions only: no mocking, no DB, no file I/O. Byte buffers are
constructed in-memory with zlib + struct.

Note on gap markers: the DatZ binary row format packs event_code as an
UNSIGNED char ('>BBH'), so a raw byte stream physically cannot encode
event_code == -1. Gap markers enter the pipeline post-decode via
insert_gap_marker(). The tests below therefore assert two things:
  1. 0xFF decodes to 255 — never silently aliased to a gap marker.
  2. A marker injected via insert_gap_marker() survives sorting and
     concatenation intact (not dropped, not merged, values preserved).
"""

import struct
import zlib

import pandas as pd
import pytest

from atspm.analysis.decoders import (
    DatZDecodingError,
    estimate_event_count,
    insert_gap_marker,
    parse_datz_batch,
    parse_datz_bytes,
    parse_datz_header,
    validate_datz_file,
)

BASE_TS = 1_609_459_200.0  # 2021-01-01 00:00:00 UTC
SCHEMA = ['timestamp', 'event_code', 'parameter']


def _pack_row(event_code: int, parameter: int, offset_deciseconds: int) -> bytes:
    """Pack one binary event row: Big-Endian UChar, UChar, UShort."""
    return struct.pack('>BBH', event_code, parameter, offset_deciseconds)


def _make_datz(payload: bytes, header: bytes = b"Some preamble\nPhases in use: 2,4,6,8\n") -> bytes:
    """Build a compressed DatZ buffer around a binary payload."""
    return zlib.compress(header + payload)


def _make_datz_with_header(payload: bytes, clock: str = "1/1/2021,00:00:00.5") -> bytes:
    """Build a compressed DatZ buffer carrying a real controller preamble.

    Args:
        payload: Binary event rows.
        clock:   ``<M/D/YYYY>,<HH:MM:SS.s>`` written into the
                 ``Controller Data Log Beginning`` line.
    """
    preamble = (
        b"1-1-2021 00:00:00.5,Version #:,3\n"
        b"1-1-2021 00:00:00.5,Controller Data Log Beginning:,"
        + clock.encode()
        + b"\n"
        b"1-1-2021 00:00:00.5,Phases in use:,1,2,3,4,5,6,7,8\n"
    )
    return zlib.compress(preamble + payload)


# ---------------------------------------------------------------------------
# parse_datz_bytes — malformed / truncated input
# ---------------------------------------------------------------------------

class TestParseDatzBytesMalformedInput:

    def test_empty_input_raises_decoding_error(self):
        with pytest.raises(DatZDecodingError, match="decompress"):
            parse_datz_bytes(b"", BASE_TS)

    def test_non_zlib_garbage_raises_decoding_error(self):
        with pytest.raises(DatZDecodingError, match="decompress"):
            parse_datz_bytes(b"\xde\xad\xbe\xef" * 16, BASE_TS)

    def test_truncated_compressed_stream_raises_decoding_error(self):
        good = _make_datz(_pack_row(1, 2, 100) * 20)
        with pytest.raises(DatZDecodingError, match="decompress"):
            parse_datz_bytes(good[: len(good) // 2], BASE_TS)

    def test_missing_marker_raises_decoding_error(self):
        raw = zlib.compress(b"no marker here\n" + _pack_row(1, 2, 100))
        with pytest.raises(DatZDecodingError, match="marker not found"):
            parse_datz_bytes(raw, BASE_TS)

    def test_marker_without_trailing_newline_raises_decoding_error(self):
        raw = zlib.compress(b"Phases in use: 2,4")
        with pytest.raises(DatZDecodingError, match="No newline"):
            parse_datz_bytes(raw, BASE_TS)

    def test_trailing_partial_record_raises_decoding_error(self):
        # 2 full rows + 2 stray bytes = 10 bytes, not divisible by 4
        payload = _pack_row(1, 2, 100) + _pack_row(8, 2, 150) + b"\x01\x02"
        with pytest.raises(DatZDecodingError, match="not divisible"):
            parse_datz_bytes(_make_datz(payload), BASE_TS)

    def test_single_stray_byte_payload_raises_decoding_error(self):
        with pytest.raises(DatZDecodingError, match="not divisible"):
            parse_datz_bytes(_make_datz(b"\xff"), BASE_TS)


# ---------------------------------------------------------------------------
# parse_datz_bytes — empty and boundary payloads
# ---------------------------------------------------------------------------

class TestParseDatzBytesBoundaries:

    def test_empty_payload_returns_empty_frame_with_schema(self):
        df = parse_datz_bytes(_make_datz(b""), BASE_TS)
        assert df.empty
        assert list(df.columns) == SCHEMA

    def test_all_zero_row_decodes_to_base_timestamp(self):
        df = parse_datz_bytes(_make_datz(_pack_row(0, 0, 0)), BASE_TS)
        assert len(df) == 1
        assert df.loc[0, 'timestamp'] == BASE_TS
        assert df.loc[0, 'event_code'] == 0
        assert df.loc[0, 'parameter'] == 0

    def test_max_byte_values_decode_at_type_ceilings(self):
        # UChar max = 255, UShort max = 65535 deciseconds = 6553.5 s
        df = parse_datz_bytes(_make_datz(_pack_row(255, 255, 65535)), BASE_TS)
        assert df.loc[0, 'event_code'] == 255
        assert df.loc[0, 'parameter'] == 255
        assert df.loc[0, 'timestamp'] == BASE_TS + 6553.5

    def test_0xff_event_code_is_255_never_negative_one(self):
        # The unsigned format cannot represent -1; 0xFF must decode to 255
        # and must never be conflated with the gap marker sentinel.
        df = parse_datz_bytes(_make_datz(b"\xff\xff\xff\xff"), BASE_TS)
        assert (df['event_code'] == 255).all()
        assert not (df['event_code'] == -1).any()

    def test_output_dtypes_are_float_and_int(self):
        df = parse_datz_bytes(_make_datz(_pack_row(82, 3, 7)), BASE_TS)
        assert pd.api.types.is_float_dtype(df['timestamp'])
        assert pd.api.types.is_integer_dtype(df['event_code'])
        assert pd.api.types.is_integer_dtype(df['parameter'])

    def test_newline_bytes_inside_payload_are_not_treated_as_delimiters(self):
        # Offset 0x0A0A = 2570 deciseconds embeds two newline bytes in the
        # record; the parser must slice by row size, not split on newlines.
        payload = _pack_row(10, 10, 0x0A0A) + _pack_row(1, 2, 0x0A00)
        df = parse_datz_bytes(_make_datz(payload), BASE_TS)
        assert len(df) == 2
        assert df.loc[0, 'timestamp'] == BASE_TS + 257.0
        assert df.loc[1, 'timestamp'] == BASE_TS + 256.0

    def test_input_row_order_is_preserved_even_when_offsets_decrease(self):
        # parse_datz_bytes does not sort; out-of-order offsets must come
        # back in stream order (sorting is parse_datz_batch's job).
        payload = _pack_row(1, 2, 300) + _pack_row(1, 6, 100) + _pack_row(8, 2, 200)
        df = parse_datz_bytes(_make_datz(payload), BASE_TS)
        assert list(df['timestamp']) == [BASE_TS + 30.0, BASE_TS + 10.0, BASE_TS + 20.0]
        assert list(df['parameter']) == [2, 6, 2]


# ---------------------------------------------------------------------------
# Gap marker survival (event_code == -1)
# ---------------------------------------------------------------------------

class TestGapMarkerSurvival:

    def test_gap_marker_survives_insertion_into_decoded_frame(self):
        payload = _pack_row(1, 2, 0) + _pack_row(8, 2, 600)
        df = parse_datz_bytes(_make_datz(payload), BASE_TS)
        out = insert_gap_marker(df, gap_timestamp=BASE_TS + 30.0)

        assert len(out) == len(df) + 1
        gap_rows = out[out['event_code'] == -1]
        assert len(gap_rows) == 1
        assert gap_rows.iloc[0]['parameter'] == -1
        assert gap_rows.iloc[0]['timestamp'] == BASE_TS + 30.0
        # Sorted into position between the two real events
        assert out['timestamp'].is_monotonic_increasing
        assert out.index[out['event_code'] == -1][0] == 1

    def test_gap_marker_not_merged_on_timestamp_collision(self):
        # A real event at the exact gap timestamp must not absorb or be
        # absorbed by the marker — both rows survive.
        df = parse_datz_bytes(_make_datz(_pack_row(82, 3, 100)), BASE_TS)
        collision_ts = BASE_TS + 10.0
        out = insert_gap_marker(df, gap_timestamp=collision_ts)

        assert len(out) == 2
        at_ts = out[out['timestamp'] == collision_ts]
        assert len(at_ts) == 2
        assert set(at_ts['event_code']) == {82, -1}

    def test_gap_marker_survives_downstream_sort_of_merged_files(self):
        # Simulates the shell's pattern: decode file A, mark the gap, decode
        # file B, concatenate and sort. The marker must survive intact.
        df_a = parse_datz_bytes(_make_datz(_pack_row(1, 2, 0)), BASE_TS)
        df_a = insert_gap_marker(df_a, gap_timestamp=BASE_TS + 60.0)
        df_b = parse_datz_bytes(_make_datz(_pack_row(1, 4, 0)), BASE_TS + 120.0)

        merged = (
            pd.concat([df_a, df_b], ignore_index=True)
            .sort_values('timestamp')
            .reset_index(drop=True)
        )
        assert (merged['event_code'] == -1).sum() == 1
        gap = merged[merged['event_code'] == -1].iloc[0]
        assert gap['timestamp'] == BASE_TS + 60.0
        assert gap['parameter'] == -1

    def test_gap_marker_insertion_into_empty_frame(self):
        empty = pd.DataFrame(columns=SCHEMA)
        out = insert_gap_marker(empty, gap_timestamp=BASE_TS)
        assert len(out) == 1
        assert out.iloc[0]['event_code'] == -1
        assert out.iloc[0]['parameter'] == -1


# ---------------------------------------------------------------------------
# validate_datz_file / estimate_event_count / parse_datz_batch
# ---------------------------------------------------------------------------

class TestValidateDatzFile:

    def test_valid_buffer_returns_true(self):
        assert validate_datz_file(_make_datz(_pack_row(1, 2, 0))) is True

    def test_garbage_returns_false_instead_of_raising(self):
        assert validate_datz_file(b"\x00\x01garbage") is False
        assert validate_datz_file(b"") is False

    def test_decompressible_but_markerless_returns_false(self):
        assert validate_datz_file(zlib.compress(b"nothing relevant")) is False


class TestEstimateEventCount:

    def test_count_matches_full_parse(self):
        payload = b"".join(_pack_row(1, p, p * 10) for p in range(1, 9))
        raw = _make_datz(payload)
        assert estimate_event_count(raw) == len(parse_datz_bytes(raw, BASE_TS))

    def test_invalid_input_returns_zero(self):
        assert estimate_event_count(b"not zlib") == 0
        assert estimate_event_count(zlib.compress(b"no marker")) == 0

    def test_partial_trailing_record_is_floored_not_raised(self):
        # Divisibility is not enforced here: 9 bytes // 4 = 2.
        raw = _make_datz(_pack_row(1, 2, 0) + _pack_row(8, 2, 10) + b"\xff")
        assert estimate_event_count(raw) == 2


class TestParseDatzBatch:

    def test_empty_batch_returns_empty_frame_with_schema(self):
        df = parse_datz_batch([])
        assert df.empty
        assert list(df.columns) == SCHEMA

    def test_batch_of_only_empty_payloads_returns_empty_schema(self):
        df = parse_datz_batch([(_make_datz(b""), BASE_TS)])
        assert df.empty
        assert list(df.columns) == SCHEMA

    def test_out_of_order_files_are_sorted_by_timestamp(self):
        late = (_make_datz(_pack_row(1, 4, 0)), BASE_TS + 900.0)
        early = (_make_datz(_pack_row(1, 2, 0)), BASE_TS)
        df = parse_datz_batch([late, early])
        assert df['timestamp'].is_monotonic_increasing
        assert list(df['parameter']) == [2, 4]

    def test_one_bad_file_fails_the_whole_batch(self):
        good = (_make_datz(_pack_row(1, 2, 0)), BASE_TS)
        bad = (b"corrupt", BASE_TS + 900.0)
        with pytest.raises(DatZDecodingError):
            parse_datz_batch([good, bad])

    def test_header_offset_applies_per_file_within_a_batch(self):
        first = (_make_datz_with_header(_pack_row(1, 2, 0), "1/1/2021,00:00:00.9"), BASE_TS)
        second = (_make_datz_with_header(_pack_row(1, 4, 0), "1/1/2021,00:15:00.2"),
                  BASE_TS + 900.0)
        df = parse_datz_batch([first, second])
        assert list(df['timestamp']) == [BASE_TS + 0.9, BASE_TS + 900.2]


# ---------------------------------------------------------------------------
# Header sub-minute offset (Controller Data Log Beginning)
# ---------------------------------------------------------------------------

class TestHeaderOffset:
    """Binary offsets are measured from the header instant, not the filename
    boundary, so the header's sub-minute delta shifts the event base."""

    def test_header_offset_shifts_every_event(self):
        payload = _pack_row(1, 2, 0) + _pack_row(8, 2, 150)
        df = parse_datz_bytes(_make_datz_with_header(payload, "1/1/2021,00:00:00.5"), BASE_TS)
        assert list(df['timestamp']) == [BASE_TS + 0.5, BASE_TS + 15.5]

    def test_full_second_offset_is_applied(self):
        df = parse_datz_bytes(
            _make_datz_with_header(_pack_row(1, 2, 0), "1/1/2021,00:00:01.0"), BASE_TS
        )
        assert df.loc[0, 'timestamp'] == BASE_TS + 1.0

    def test_zero_offset_leaves_base_unchanged(self):
        df = parse_datz_bytes(
            _make_datz_with_header(_pack_row(1, 2, 0), "1/1/2021,00:00:00.0"), BASE_TS
        )
        assert df.loc[0, 'timestamp'] == BASE_TS

    def test_within_file_durations_are_unaffected_by_the_shift(self):
        payload = _pack_row(1, 2, 100) + _pack_row(8, 2, 400)
        shifted = parse_datz_bytes(
            _make_datz_with_header(payload, "1/1/2021,00:00:00.7"), BASE_TS
        )
        unshifted = parse_datz_bytes(_make_datz(payload), BASE_TS)
        assert shifted['timestamp'].diff().iloc[1] == unshifted['timestamp'].diff().iloc[1]

    def test_missing_header_falls_back_to_supplied_base(self):
        df = parse_datz_bytes(_make_datz(_pack_row(1, 2, 0)), BASE_TS)
        assert df.loc[0, 'timestamp'] == BASE_TS

    def test_only_the_seconds_delta_is_used_not_the_absolute_header_time(self):
        # The header's HH:MM is deliberately ignored — the caller's boundary
        # is authoritative, so a mismatched hour must not move the base.
        df = parse_datz_bytes(
            _make_datz_with_header(_pack_row(1, 2, 0), "6/30/2026,17:45:00.3"), BASE_TS
        )
        assert df.loc[0, 'timestamp'] == BASE_TS + 0.3

    def test_out_of_range_seconds_falls_back_to_supplied_base(self):
        df = parse_datz_bytes(
            _make_datz_with_header(_pack_row(1, 2, 0), "1/1/2021,00:00:99.9"), BASE_TS
        )
        assert df.loc[0, 'timestamp'] == BASE_TS


class TestParseDatzHeader:

    def test_returns_all_fields_from_a_real_preamble(self):
        header = parse_datz_header(
            _make_datz_with_header(_pack_row(1, 2, 0), "7/29/2026,11:30:00.1")
        )
        assert header == {
            'year': 2026, 'month': 7, 'day': 29,
            'hour': 11, 'minute': 30, 'second_offset': 0.1,
        }

    def test_returns_none_when_header_line_absent(self):
        assert parse_datz_header(_make_datz(_pack_row(1, 2, 0))) is None

    def test_returns_none_for_impossible_hour(self):
        assert parse_datz_header(
            _make_datz_with_header(b"", "1/1/2021,42:00:00.1")
        ) is None

    def test_decompression_failure_raises_decoding_error(self):
        with pytest.raises(DatZDecodingError, match="decompress"):
            parse_datz_header(b"\xde\xad\xbe\xef" * 16)

    def test_does_not_require_a_binary_payload(self):
        header = parse_datz_header(_make_datz_with_header(b"", "1/1/2021,00:00:00.4"))
        assert header['second_offset'] == 0.4
