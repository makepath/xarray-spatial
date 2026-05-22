"""PackBits decode JIT kernel coverage (issue #2048).

``packbits_decompress`` was reworked from a pure-Python ``while`` loop into a
numba ``@ngjit`` kernel wrapped by a thin bytes-in / bytes-out shim. These
tests pin the kernel against PackBits' boundary conditions: the 128-byte run
length boundary that switches literal/replicate encodings, max-length runs,
the ``-128`` no-op sentinel, empty input, and the decompression-bomb cap.

The pre-existing PackBits coverage (``test_features.py`` round-trips and
``test_decompression_caps.py`` bomb guard) keeps passing too; this file just
fills in the bit-exact edge cases that the JIT rewrite is most likely to
regress on.
"""
from __future__ import annotations

import pytest

from xrspatial.geotiff._compression import packbits_compress, packbits_decompress

# -- Bit-exact decode against known PackBits encodings -----------------------


def test_packbits_decode_empty_input_returns_empty_bytes():
    assert packbits_decompress(b"") == b""


def test_packbits_decode_single_literal_byte():
    # Header 0x00 -> "copy next 1 byte literally".
    assert packbits_decompress(bytes([0x00, 0x42])) == b"\x42"


def test_packbits_decode_single_replicate_pair():
    # Header 0xFF (signed -1) -> "repeat next byte 2 times".
    assert packbits_decompress(bytes([0xFF, 0x42])) == b"\x42\x42"


def test_packbits_decode_noop_sentinel_is_skipped():
    # Header 0x80 (signed -128) is a no-op marker; surrounding data still decodes.
    assert packbits_decompress(bytes([0x80, 0x00, 0x42])) == b"\x42"


def test_packbits_decode_wikipedia_canonical_example():
    # From the PackBits Wikipedia article, normalized to a self-contained vector.
    encoded = bytes(
        [0xFE, 0xAA, 0x02, 0x80, 0x00, 0x2A, 0xFD, 0xAA,
         0x03, 0x80, 0x00, 0x2A, 0x22, 0xF7, 0xAA]
    )
    expected = (
        b"\xAA" * 3
        + b"\x80\x00\x2A"
        + b"\xAA" * 4
        + b"\x80\x00\x2A\x22"
        + b"\xAA" * 10
    )
    assert packbits_decompress(encoded) == expected


# -- Run-length boundary cases (128-byte switch) -----------------------------


def test_packbits_decode_max_literal_run_is_128_bytes():
    # Header 0x7F (127) -> 128 literal bytes follow.
    literal = bytes(range(128))
    encoded = bytes([0x7F]) + literal
    assert packbits_decompress(encoded) == literal


def test_packbits_decode_max_replicate_run_is_128_bytes():
    # Header 0x81 (signed -127) -> repeat next byte 128 times.
    encoded = bytes([0x81, 0x42])
    assert packbits_decompress(encoded) == b"\x42" * 128


def test_packbits_decode_back_to_back_max_runs():
    # Two back-to-back max-length runs land on the 128-byte boundary
    # and exercise the literal -> replicate switch with no gap.
    literal = bytes(range(128))
    encoded = bytes([0x7F]) + literal + bytes([0x81, 0x99])
    assert packbits_decompress(encoded) == literal + b"\x99" * 128


# -- Round-trip parity with the (untouched) compressor -----------------------


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"A",
        b"AAAA",
        b"ABCD" * 64,
        bytes(range(256)),
        b"\x00" * 1024,
        (b"long literal stretch with no runs at all 1234567890" * 17),
    ],
)
def test_packbits_roundtrip(payload):
    assert packbits_decompress(packbits_compress(payload)) == payload


# -- Decompression-bomb cap stays intact across the rewrite ------------------


def test_packbits_cap_rejects_oversized_expansion():
    # 0x81 0x42 produces 128 bytes from a 2-byte input; cap of 4 must reject.
    with pytest.raises(ValueError, match="packbits decode exceeded expected size"):
        packbits_decompress(bytes([0x81, 0x42]), expected_size=4)


def test_packbits_cap_allows_within_margin():
    # Cap is expected_size * 1.05 + 1 = 5; a 4-byte decode must pass.
    out = packbits_decompress(bytes([0xFF, 0x42, 0xFF, 0x43]), expected_size=4)
    assert out == b"BBCC"


@pytest.mark.parametrize(
    "expected_size, payload_bytes, should_pass",
    [
        # Cap = int(expected_size * 1.05) + 1.
        # expected_size=1 -> cap=2; 2-byte legitimate decode lands on the cap.
        (1, 2, True),
        # expected_size=100 -> cap=106; 106-byte decode equals the cap.
        (100, 106, True),
        # expected_size=100 -> cap=106; 107-byte decode trips the guard.
        (100, 107, False),
    ],
)
def test_packbits_cap_boundary(expected_size, payload_bytes, should_pass):
    # Encode `payload_bytes` zeros using replicate runs of 128 plus a tail.
    full_runs, tail = divmod(payload_bytes, 128)
    encoded = bytes([0x81, 0x00]) * full_runs
    if tail:
        # Replicate run of `tail` bytes: header = 257 - tail (for tail >= 2),
        # or a single literal byte (header 0x00) for tail == 1.
        if tail == 1:
            encoded += bytes([0x00, 0x00])
        else:
            encoded += bytes([257 - tail, 0x00])
    if should_pass:
        out = packbits_decompress(encoded, expected_size=expected_size)
        assert out == b"\x00" * payload_bytes
    else:
        with pytest.raises(ValueError, match="packbits decode exceeded"):
            packbits_decompress(encoded, expected_size=expected_size)


def test_packbits_no_cap_when_expected_size_is_zero():
    # expected_size=0 disables the cap (backward-compat path).
    out = packbits_decompress(bytes([0x81, 0x42]))
    assert out == b"\x42" * 128


# -- Truncated input must not read past src ----------------------------------


def test_packbits_truncated_literal_run_stops_at_src_end():
    # Header claims 4 literal bytes but only 2 follow. Decoder must not
    # read past the end of src.
    encoded = bytes([0x03, 0x01, 0x02])
    out = packbits_decompress(encoded)
    assert out == b"\x01\x02"


def test_packbits_truncated_replicate_header_stops_cleanly():
    # Replicate header without its data byte: decoder must terminate
    # rather than reading off the end.
    encoded = bytes([0xFE])
    out = packbits_decompress(encoded)
    assert out == b""
