"""Tests for LERC valid-mask handling on decode (issue #A4).

LERC's ``decode`` returns ``(rc, data, valid_mask, ...)``.  When a file
was encoded with a NoData mask, masked pixels in ``data`` are zero
filled.  Without consulting ``valid_mask`` the reader cannot tell those
zeros from real measurements, so any nodata-based downstream masking
silently misses them.

These tests cover:

* The low-level wrapper ``lerc_decompress_with_mask`` returns the mask
  and ``lerc_decompress`` still returns raw bytes for backwards compat.
* TIFF round-trip with a NaN nodata float file: NaN comes back at every
  masked position.
* TIFF round-trip with an explicit float sentinel (-9999): masked
  positions land on the sentinel.
* TIFF round-trip with an integer sentinel (uint16): masked positions
  land on the sentinel.
* Files with no mask round-trip bit-exactly (regression guard).
"""
from __future__ import annotations

import numpy as np
import pytest

lerc = pytest.importorskip("lerc")

from xrspatial.geotiff._compression import (  # noqa: E402
    LERC_AVAILABLE,
    lerc_decompress,
    lerc_decompress_with_mask,
)

pytestmark = pytest.mark.skipif(
    not LERC_AVAILABLE,
    reason="lerc not installed",
)


def _encode_with_mask(arr: np.ndarray, mask: np.ndarray | None,
                      max_z_error: float = 0.0) -> bytes:
    """Encode an array with LERC, optionally with a per-pixel valid mask."""
    has_mask = mask is not None
    result = lerc.encode(arr, 1, has_mask, mask, max_z_error, 1)
    assert result[0] == 0, f"lerc.encode returned error {result[0]}"
    return bytes(result[2])


# ---------------------------------------------------------------------------
# Wrapper-level tests (no TIFF writer involvement)
# ---------------------------------------------------------------------------

class TestLercDecompressWithMask:
    """Direct tests of the new ``lerc_decompress_with_mask`` wrapper."""

    def test_no_mask_returns_none(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        blob = _encode_with_mask(arr, None)
        decoded_bytes, mask = lerc_decompress_with_mask(blob)
        out = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(out, arr)
        assert mask is None

    def test_all_valid_mask_collapses_to_none(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        valid = np.ones((3, 4), dtype=np.uint8)
        blob = _encode_with_mask(arr, valid)
        decoded_bytes, mask = lerc_decompress_with_mask(blob)
        out = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(out, arr)
        # All-True mask carries no information, treated as no mask.
        assert mask is None

    def test_partial_mask_returns_array(self):
        arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        valid = np.array([[1, 0, 1]], dtype=np.uint8)
        blob = _encode_with_mask(arr, valid)
        decoded_bytes, mask = lerc_decompress_with_mask(blob)
        out = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(1, 3)
        # LERC zero-fills masked positions in the data array.
        np.testing.assert_array_equal(out, np.array([[1.0, 0.0, 3.0]],
                                                    dtype=np.float32))
        assert mask is not None
        np.testing.assert_array_equal(np.asarray(mask).reshape(1, 3),
                                      np.array([[1, 0, 1]], dtype=np.uint8))

    def test_legacy_decompress_drops_mask(self):
        """``lerc_decompress`` still returns just raw bytes."""
        arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        valid = np.array([[1, 0, 1]], dtype=np.uint8)
        blob = _encode_with_mask(arr, valid)
        out_bytes = lerc_decompress(blob)
        assert isinstance(out_bytes, bytes)
        out = np.frombuffer(out_bytes, dtype=np.float32).reshape(1, 3)
        np.testing.assert_array_equal(out, np.array([[1.0, 0.0, 3.0]],
                                                    dtype=np.float32))


# ---------------------------------------------------------------------------
# Reader-level (TIFF round-trip) tests
# ---------------------------------------------------------------------------

@pytest.fixture
def lerc_writer_with_mask(monkeypatch):
    """Patch ``lerc_compress`` to embed a valid-mask the writer can't pass.

    The writer hard-codes ``hasMask=False`` in its call to ``lerc.encode``
    because there's no plumbing for per-pixel mask data.  The fixture
    swaps in an encoder that derives the mask from a holder ``invalid``
    callable: given the (height, width) tile shape and the decoded
    pixel array it returns a uint8 valid mask (1=valid, 0=invalid) the
    same shape.  This lets the test specify masking by predicate so the
    mask matches whatever tile padding the writer emits.
    """
    holder = {"invalid": None}

    def _patched(data, width, height, samples=1,
                 dtype=np.dtype('float32'), max_z_error=0.0):
        if samples == 1:
            arr = np.frombuffer(data, dtype=dtype).reshape(height, width)
        else:
            arr = np.frombuffer(data, dtype=dtype).reshape(
                height, width, samples)
        invalid_pred = holder["invalid"]
        if invalid_pred is None:
            mask = None
            has_mask = False
        else:
            invalid = invalid_pred(arr)
            mask = np.where(invalid, np.uint8(0), np.uint8(1))
            has_mask = True
        result = lerc.encode(arr, samples, has_mask, mask, max_z_error, 1)
        if result[0] != 0:
            raise RuntimeError(
                f"LERC encode failed with error code {result[0]}")
        return bytes(result[2])

    monkeypatch.setattr(
        "xrspatial.geotiff._compression.lerc_compress", _patched,
    )
    return holder


class TestLercTiffRoundTripWithMask:
    """End-to-end TIFF round-trips that exercise the reader's mask merge."""

    def test_float32_nan_nodata(self, tmp_path, lerc_writer_with_mask):
        from xrspatial.geotiff._writer import write
        from xrspatial.geotiff._reader import read_to_array

        arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        # Mask positions (0, 1) and (5, 4); the rest stays valid.
        invalid_positions = {(0, 1), (5, 4)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_nan.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=float("nan"))

        out, _geo = read_to_array(path)
        for r in range(8):
            for c in range(8):
                if (r, c) in invalid_positions:
                    assert np.isnan(out[r, c]), \
                        f"expected NaN at ({r},{c}), got {out[r, c]}"
                else:
                    assert out[r, c] == arr[r, c], \
                        f"value mismatch at ({r},{c})"

    def test_float32_sentinel_nodata(self, tmp_path, lerc_writer_with_mask):
        from xrspatial.geotiff._writer import write
        from xrspatial.geotiff._reader import read_to_array

        arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        invalid_positions = {(0, 1), (3, 3), (7, 7)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_sentinel_f32.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=-9999.0)

        out, _geo = read_to_array(path)
        for r in range(8):
            for c in range(8):
                if (r, c) in invalid_positions:
                    assert out[r, c] == np.float32(-9999.0), \
                        f"expected sentinel at ({r},{c}), got {out[r, c]}"
                else:
                    assert out[r, c] == arr[r, c]

    def test_uint16_sentinel_nodata(self, tmp_path, lerc_writer_with_mask):
        from xrspatial.geotiff._writer import write
        from xrspatial.geotiff._reader import read_to_array

        arr = (np.arange(1, 65, dtype=np.uint16) * 100).reshape(8, 8)
        invalid_positions = {(0, 1), (4, 4)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_uint16.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=65535)

        out, _geo = read_to_array(path)
        for r in range(8):
            for c in range(8):
                if (r, c) in invalid_positions:
                    assert out[r, c] == np.uint16(65535)
                else:
                    assert out[r, c] == arr[r, c]

    def test_no_mask_roundtrip_bitexact(self, tmp_path):
        """Sanity: an all-valid LERC file (no mask) still round-trips."""
        from xrspatial.geotiff._writer import write
        from xrspatial.geotiff._reader import read_to_array

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / "lerc_no_mask.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8)

        out, _geo = read_to_array(path)
        np.testing.assert_array_equal(out, arr)
