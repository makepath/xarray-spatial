"""Tests for compression_level parameter in to_geotiff / write."""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_da(seed: int = 0, shape: tuple = (64, 64)) -> xr.DataArray:
    """Return a small float32 DataArray with reproducible content."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(shape).astype(np.float32)
    return xr.DataArray(arr, dims=['y', 'x'])


def _write_read(da, tmp_path, **kwargs) -> tuple[xr.DataArray, int]:
    """Write *da* to *tmp_path* with **kwargs, read back, return (da, file_size)."""
    to_geotiff(da, tmp_path, **kwargs)
    result = open_geotiff(tmp_path)
    size = os.path.getsize(tmp_path)
    return result, size


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Data survives write/read with various compression levels."""

    def test_zstd_level_1(self, tmp_path):
        da = _make_da(seed=1)
        path = str(tmp_path / 'zstd1.tif')
        result, _ = _write_read(da, path, compression='zstd', compression_level=1)
        np.testing.assert_allclose(result.values, da.values)

    def test_zstd_level_22(self, tmp_path):
        da = _make_da(seed=2)
        path = str(tmp_path / 'zstd22.tif')
        result, _ = _write_read(da, path, compression='zstd', compression_level=22)
        np.testing.assert_allclose(result.values, da.values)

    def test_deflate_level_1(self, tmp_path):
        da = _make_da(seed=3)
        path = str(tmp_path / 'deflate1.tif')
        result, _ = _write_read(da, path, compression='deflate', compression_level=1)
        np.testing.assert_allclose(result.values, da.values)

    def test_deflate_level_9(self, tmp_path):
        da = _make_da(seed=4)
        path = str(tmp_path / 'deflate9.tif')
        result, _ = _write_read(da, path, compression='deflate', compression_level=9)
        np.testing.assert_allclose(result.values, da.values)


# ---------------------------------------------------------------------------
# Higher level → smaller file
# ---------------------------------------------------------------------------

class TestLevelEffect:
    """Higher compression level produces a smaller or equal file."""

    def _make_compressible(self, shape=(512, 512)):
        """Smooth, highly compressible float32 array.

        Large + smooth so the level-9 vs level-1 gap is dominated by real
        compression work, not codec heuristic noise on tiny inputs.
        """
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        arr = (y + x).astype(np.float32)
        return xr.DataArray(arr, dims=['y', 'x'])

    def test_zstd_higher_level_not_larger(self, tmp_path):
        da = self._make_compressible()
        path_lo = str(tmp_path / 'zstd_lo.tif')
        path_hi = str(tmp_path / 'zstd_hi.tif')
        to_geotiff(da, path_lo, compression='zstd', compression_level=1)
        to_geotiff(da, path_hi, compression='zstd', compression_level=22)
        size_lo = os.path.getsize(path_lo)
        size_hi = os.path.getsize(path_hi)
        assert size_hi <= size_lo, (
            f"Expected level-22 file ({size_hi}) <= level-1 file ({size_lo})")

    def test_deflate_higher_level_not_larger(self, tmp_path):
        da = self._make_compressible()
        path_lo = str(tmp_path / 'deflate_lo.tif')
        path_hi = str(tmp_path / 'deflate_hi.tif')
        to_geotiff(da, path_lo, compression='deflate', compression_level=1)
        to_geotiff(da, path_hi, compression='deflate', compression_level=9)
        size_lo = os.path.getsize(path_lo)
        size_hi = os.path.getsize(path_hi)
        assert size_hi <= size_lo, (
            f"Expected level-9 file ({size_hi}) <= level-1 file ({size_lo})")


# ---------------------------------------------------------------------------
# Default (level=None) uses codec default
# ---------------------------------------------------------------------------

class TestDefaultLevel:
    def test_none_uses_default_zstd(self, tmp_path):
        da = _make_da(seed=5)
        path = str(tmp_path / 'zstd_default.tif')
        # Should not raise and should produce a valid file
        result, size = _write_read(da, path, compression='zstd', compression_level=None)
        assert size > 0
        np.testing.assert_allclose(result.values, da.values)

    def test_omitted_uses_default_deflate(self, tmp_path):
        da = _make_da(seed=6)
        path = str(tmp_path / 'deflate_default.tif')
        # compression_level not passed at all -- should use codec default
        result, size = _write_read(da, path, compression='deflate')
        assert size > 0
        np.testing.assert_allclose(result.values, da.values)


# ---------------------------------------------------------------------------
# LZW ignores level silently
# ---------------------------------------------------------------------------

class TestLZWIgnoresLevel:
    def test_lzw_with_level_does_not_raise(self, tmp_path):
        da = _make_da(seed=7)
        path = str(tmp_path / 'lzw_level.tif')
        # LZW has no level concept; passing one should succeed silently
        result, size = _write_read(da, path, compression='lzw', compression_level=5)
        assert size > 0
        np.testing.assert_allclose(result.values, da.values)


# ---------------------------------------------------------------------------
# Invalid levels raise ValueError
# ---------------------------------------------------------------------------

class TestInvalidLevels:
    def test_zstd_level_0_raises(self, tmp_path):
        da = _make_da()
        path = str(tmp_path / 'bad.tif')
        with pytest.raises(ValueError, match='compression_level'):
            to_geotiff(da, path, compression='zstd', compression_level=0)

    def test_zstd_level_23_raises(self, tmp_path):
        da = _make_da()
        path = str(tmp_path / 'bad.tif')
        with pytest.raises(ValueError, match='compression_level'):
            to_geotiff(da, path, compression='zstd', compression_level=23)

    def test_deflate_level_0_raises(self, tmp_path):
        da = _make_da()
        path = str(tmp_path / 'bad.tif')
        with pytest.raises(ValueError, match='compression_level'):
            to_geotiff(da, path, compression='deflate', compression_level=0)

    def test_deflate_level_10_raises(self, tmp_path):
        da = _make_da()
        path = str(tmp_path / 'bad.tif')
        with pytest.raises(ValueError, match='compression_level'):
            to_geotiff(da, path, compression='deflate', compression_level=10)

    def test_negative_level_raises(self, tmp_path):
        da = _make_da()
        path = str(tmp_path / 'bad.tif')
        with pytest.raises(ValueError, match='compression_level'):
            to_geotiff(da, path, compression='zstd', compression_level=-1)
