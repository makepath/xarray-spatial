"""Pure-unit coverage for the codec helpers in ``_compression``.

Two groups of cases share this file because they fail in the same ways:

* PackBits decoder JIT kernel boundary cases (the 128-byte run boundary,
  the ``-128`` no-op sentinel, the decompression-bomb cap, truncated
  inputs).
* PackBits encoder JIT kernel round-trips against the decoder, kernel
  buffer mechanics, and worst-case output cap.

The matrix of write/read round-trips against the public ``to_geotiff`` /
``open_geotiff`` API lives separately under ``test_features.py`` and the
codec-specific files (``test_lz4.py``, ``test_lerc.py``, etc.); this file
exercises only the bytes-in / bytes-out kernels.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._compression import (_packbits_encode_kernel, packbits_compress,
                                            packbits_decompress)

# ---------------------------------------------------------------------------
# Decoder: bit-exact decode against known PackBits encodings
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Decoder: run-length boundary cases (128-byte switch)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Decoder: round-trip parity with the compressor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Decoder: decompression-bomb cap
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Decoder: truncated input handling
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Encoder: round-trip against the decoder
# ---------------------------------------------------------------------------


def _roundtrip(data: bytes) -> None:
    assert packbits_decompress(packbits_compress(data)) == data


class TestPackBitsEncodeRoundTrip:
    """Encode-decode parity across the regime boundaries of PackBits."""

    def test_empty(self):
        _roundtrip(b'')

    def test_length_one(self):
        _roundtrip(b'\x42')

    def test_length_two_same(self):
        _roundtrip(b'\xAA\xAA')

    def test_length_two_different(self):
        _roundtrip(b'\x00\xFF')

    def test_length_128_all_same(self):
        # Hits the inner run cap exactly.
        _roundtrip(b'\x55' * 128)

    def test_length_129_all_same(self):
        # Forces a second header byte after the 128-byte run cap.
        _roundtrip(b'\x55' * 129)

    def test_length_128_alternating(self):
        # Hits the literal cap exactly; no run ever forms.
        _roundtrip(bytes([i & 1 for i in range(128)]))

    def test_length_129_alternating(self):
        # Forces a second literal header after the 128-byte literal cap.
        _roundtrip(bytes([i & 1 for i in range(129)]))

    def test_alternating_short(self):
        _roundtrip(b'\x00\xFF\x00\xFF\x00\xFF')

    def test_run_of_three_at_end(self):
        # Boundary between literal scan and run detection.
        _roundtrip(b'\x01\x02\x03\xAA\xAA\xAA')

    def test_run_of_three_at_start(self):
        _roundtrip(b'\xAA\xAA\xAA\x01\x02\x03')

    def test_runs_and_literals_interleaved(self):
        data = b'\x00' * 100 + b'\xFF' * 50 + bytes(range(200))
        _roundtrip(data)

    @pytest.mark.parametrize("seed", [0, 1, 42, 12345])
    def test_random_bytes(self, seed):
        rng = np.random.default_rng(seed)
        data = rng.integers(0, 256, size=1024, dtype=np.uint8).tobytes()
        _roundtrip(data)

    def test_random_with_runs(self):
        # Mix runs of varying length with random literals.
        rng = np.random.default_rng(7)
        chunks = []
        for _ in range(20):
            if rng.random() < 0.5:
                val = int(rng.integers(0, 256))
                length = int(rng.integers(1, 200))
                chunks.append(bytes([val]) * length)
            else:
                length = int(rng.integers(1, 200))
                chunks.append(rng.integers(0, 256, size=length, dtype=np.uint8).tobytes())
        _roundtrip(b''.join(chunks))

    def test_all_zeros_large(self):
        data = b'\x00' * 10_000
        compressed = packbits_compress(data)
        # 10_000 bytes at run-cap 128 -> ceil(10_000 / 128) = 79 runs,
        # each 2 bytes => 158 bytes total.
        assert len(compressed) < len(data) // 50
        assert packbits_decompress(compressed) == data


# ---------------------------------------------------------------------------
# Encoder: low-level kernel surface
# ---------------------------------------------------------------------------


class TestPackBitsEncodeKernel:
    """The kernel itself is callable; sanity-check buffer mechanics."""

    def test_kernel_returns_length(self):
        src = np.array([1, 1, 1, 1, 1], dtype=np.uint8)
        dst = np.empty(2 * len(src) + 1, dtype=np.uint8)
        n = _packbits_encode_kernel(src, len(src), dst, len(dst))
        # 5 identical bytes -> one run: 2 bytes (header + value)
        assert n == 2
        # Header is the signed int8 (1 - 5) = -4, stored as 256 - 4 = 252
        assert dst[0] == 252
        assert dst[1] == 1

    def test_kernel_empty_input(self):
        src = np.empty(0, dtype=np.uint8)
        dst = np.empty(1, dtype=np.uint8)
        n = _packbits_encode_kernel(src, 0, dst, 1)
        assert n == 0

    def test_kernel_literal_golden(self):
        # Three distinct bytes encode as a single literal header (lit_len-1=2)
        # followed by the payload.
        src = np.array([0x10, 0x20, 0x30], dtype=np.uint8)
        dst = np.empty(2 * len(src) + 1, dtype=np.uint8)
        n = _packbits_encode_kernel(src, len(src), dst, len(dst))
        assert n == 4
        assert dst[0] == 2
        assert list(dst[1:4]) == [0x10, 0x20, 0x30]


# ---------------------------------------------------------------------------
# Encoder: worst-case output cap
# ---------------------------------------------------------------------------


class TestPackBitsEncodeBufferCap:
    """Output must always fit inside the worst-case allocation."""

    @pytest.mark.parametrize(
        "data",
        [
            b'',
            b'\x00',
            b'\x00\xFF',
            b'\x55' * 128,
            b'\x55' * 129,
            bytes([i & 1 for i in range(256)]),
            bytes(range(256)),
        ],
    )
    def test_output_within_cap(self, data):
        compressed = packbits_compress(data)
        # The wrapper allocates 2 * src_len + 1 bytes for the encode buffer;
        # the actual output must never exceed that bound.
        assert len(compressed) <= 2 * len(data) + 1

    def test_random_output_within_cap(self):
        rng = np.random.default_rng(2049)
        for _ in range(8):
            length = int(rng.integers(0, 4096))
            data = rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()
            compressed = packbits_compress(data)
            assert len(compressed) <= 2 * length + 1
