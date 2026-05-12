"""Regression coverage for the vectorised ``unpack_bits`` (issue #1713).

The pre-vectorisation implementation walked the packed input byte by byte
in Python, which was ~100x slower than the numpy-strided equivalent on
realistic tile sizes. These tests pin the bit-for-bit equivalence
between the new code and an in-line reference implementation across the
sub-byte BitsPerSample values the reader recognises (``{1, 2, 4, 12}``),
including the short-input-buffer corner cases the original loops
handled.
"""
import numpy as np
import pytest

from xrspatial.geotiff._compression import unpack_bits


def _reference_unpack_bits(data: np.ndarray, bps: int,
                           pixel_count: int) -> np.ndarray:
    """Bit-for-bit copy of the original loop-based implementation.

    Kept here (rather than imported) so the test survives any future
    deletion of the loop-based code path.
    """
    if bps == 1:
        out = np.unpackbits(data)[:pixel_count]
        return out.astype(np.uint8)
    if bps == 2:
        out = np.empty(pixel_count, dtype=np.uint8)
        for i in range(min(len(data), (pixel_count + 3) // 4)):
            b = data[i]
            base = i * 4
            if base < pixel_count:
                out[base] = (b >> 6) & 0x03
            if base + 1 < pixel_count:
                out[base + 1] = (b >> 4) & 0x03
            if base + 2 < pixel_count:
                out[base + 2] = (b >> 2) & 0x03
            if base + 3 < pixel_count:
                out[base + 3] = b & 0x03
        return out
    if bps == 4:
        out = np.empty(pixel_count, dtype=np.uint8)
        for i in range(min(len(data), (pixel_count + 1) // 2)):
            b = data[i]
            base = i * 2
            if base < pixel_count:
                out[base] = (b >> 4) & 0x0F
            if base + 1 < pixel_count:
                out[base + 1] = b & 0x0F
        return out
    if bps == 12:
        out = np.empty(pixel_count, dtype=np.uint16)
        n_pairs = pixel_count // 2
        remainder = pixel_count % 2
        for i in range(n_pairs):
            off = i * 3
            if off + 2 < len(data):
                b0 = int(data[off])
                b1 = int(data[off + 1])
                b2 = int(data[off + 2])
                out[i * 2] = (b0 << 4) | (b1 >> 4)
                out[i * 2 + 1] = ((b1 & 0x0F) << 8) | b2
        if remainder and n_pairs * 3 + 1 < len(data):
            off = n_pairs * 3
            out[pixel_count - 1] = (
                (int(data[off]) << 4) | (int(data[off + 1]) >> 4)
            )
        return out
    raise ValueError(f"Unsupported bps: {bps}")


def _written_positions(bps: int, pixel_count: int, data_len: int) -> set:
    """Return the indices the original loop *wrote to*.

    The pre-vectorisation code used ``np.empty`` and then conditionally
    skipped writes when the input buffer was too short. The new code
    must agree on positions that the old code wrote; positions that
    were never written are pure ``np.empty`` garbage and must not be
    compared. This helper enumerates the written set so the test can
    stick to it.
    """
    positions: set[int] = set()
    if bps == 2:
        n_bytes = min(data_len, (pixel_count + 3) // 4)
        for i in range(n_bytes):
            base = i * 4
            for n in range(4):
                if base + n < pixel_count:
                    positions.add(base + n)
    elif bps == 4:
        n_bytes = min(data_len, (pixel_count + 1) // 2)
        for i in range(n_bytes):
            base = i * 2
            for n in range(2):
                if base + n < pixel_count:
                    positions.add(base + n)
    elif bps == 12:
        n_pairs = pixel_count // 2
        for i in range(n_pairs):
            off = i * 3
            if off + 2 < data_len:
                positions.add(i * 2)
                positions.add(i * 2 + 1)
        rem = pixel_count % 2
        if rem and n_pairs * 3 + 1 < data_len:
            positions.add(pixel_count - 1)
    elif bps == 1:
        # bps=1 covers every position via np.unpackbits.
        positions.update(range(pixel_count))
    return positions


# ----------------------------------------------------------------------
# Equivalence to the reference implementation across all sub-byte bps
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bps", [2, 4, 12])
@pytest.mark.parametrize("pixel_count", [0, 1, 2, 3, 4, 7, 8, 100, 10_000])
@pytest.mark.parametrize("data_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_vectorised_matches_reference(bps, pixel_count, data_factor):
    """Vectorised output equals the original for every covered position."""
    if bps == 2:
        bytes_per_pixel = 0.25
    elif bps == 4:
        bytes_per_pixel = 0.5
    else:  # bps == 12
        bytes_per_pixel = 1.5

    required = int(np.ceil(pixel_count * bytes_per_pixel))
    n_bytes = max(0, int(required * data_factor))
    rng = np.random.default_rng(seed=bps * 10_000 + pixel_count)
    data = rng.integers(0, 256, size=n_bytes, dtype=np.uint8)

    ref = _reference_unpack_bits(data, bps, pixel_count)
    new = unpack_bits(data, bps, pixel_count)

    assert ref.shape == new.shape
    assert ref.dtype == new.dtype

    for p in _written_positions(bps, pixel_count, len(data)):
        assert ref[p] == new[p], (
            f"bps={bps} pc={pixel_count} data_factor={data_factor}: "
            f"position {p} differs ref={ref[p]} new={new[p]}"
        )


# ----------------------------------------------------------------------
# Spot-check that bps=1 still returns the unpacked bit stream
# ----------------------------------------------------------------------

def test_bps1_unchanged():
    """bps=1 still routes through ``np.unpackbits`` and returns uint8."""
    data = np.array([0b10101100, 0b00001111], dtype=np.uint8)
    out = unpack_bits(data, 1, 16)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(
        out,
        np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                 dtype=np.uint8),
    )


# ----------------------------------------------------------------------
# Boundary case the original loop's strict-less-than guard cared about
# ----------------------------------------------------------------------

def test_bps12_three_byte_buffer_decodes_one_pair():
    """bps=12 with exactly 3 input bytes writes the single pair.

    The original loop's guard was ``off + 2 < len(data)``, which is
    satisfied for ``off=0, len(data)=3``. The vectorised implementation
    must keep the same decision and not skip the pair (this was the
    boundary case that surfaced during the rewrite).
    """
    # Two 12-bit values: 0x123 and 0x456 packed MSB-first into 3 bytes.
    data = np.array([0x12, 0x34, 0x56], dtype=np.uint8)
    out = unpack_bits(data, 12, 2)
    assert tuple(int(v) for v in out) == (0x123, 0x456)


def test_bps12_two_byte_buffer_no_pair_decoded():
    """A 2-byte buffer cannot satisfy ``off+2 < 2`` so no pair is written.

    Mirrors the original loop semantics. ``np.empty`` initial garbage
    at those positions is fine -- the test only asserts that the
    function does not crash and returns an array of the right shape.
    """
    data = np.array([0x12, 0x34], dtype=np.uint8)
    out = unpack_bits(data, 12, 2)
    assert out.shape == (2,)
    assert out.dtype == np.uint16


def test_unsupported_bps_raises():
    """Unknown sub-byte bps still raises a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported"):
        unpack_bits(np.zeros(10, dtype=np.uint8), 3, 10)
