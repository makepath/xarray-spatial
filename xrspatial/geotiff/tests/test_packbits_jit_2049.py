"""JIT'd PackBits encoder: round-trip and edge-case coverage.

See issue #2049. ``packbits_compress`` was pure-Python; this file pins the
contract of the ``@ngjit`` rewrite against the decoder.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._compression import (_packbits_encode_kernel, packbits_compress,
                                            packbits_decompress)


def _roundtrip(data: bytes) -> None:
    assert packbits_decompress(packbits_compress(data)) == data


class TestPackBitsJITRoundTrip:
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


class TestPackBitsJITKernel:
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


class TestPackBitsJITBufferCap:
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
