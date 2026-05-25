"""Tests for the ``max_z_error`` knob on LERC writes (#1510).

LERC supports a per-pixel error budget. ``max_z_error=0`` is lossless;
larger values trade bit-for-bit fidelity for smaller files, with the
guarantee that ``abs(decoded - original) <= max_z_error`` for every
pixel.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._compression import LERC_AVAILABLE

pytestmark = pytest.mark.skipif(
    not LERC_AVAILABLE,
    reason="lerc not installed",
)


def _smooth_surface(h: int = 64, w: int = 64) -> np.ndarray:
    """Smooth float32 surface that LERC can quantise efficiently."""
    yy, xx = np.meshgrid(
        np.linspace(0.0, 4.0, h, dtype=np.float32),
        np.linspace(0.0, 4.0, w, dtype=np.float32),
        indexing='ij',
    )
    return (np.sin(yy) + np.cos(xx) * 0.5).astype(np.float32) * 10.0


def _make_dataarray(arr: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(arr.shape[0]),
                'x': np.arange(arr.shape[1])},
        attrs={'crs': 4326},
    )


class TestLerclessLossless:
    """``max_z_error=0`` keeps the existing lossless behavior."""

    def test_lossless_roundtrip_bit_exact(self, tmp_path):
        arr = _smooth_surface()
        da = _make_dataarray(arr)
        path = str(tmp_path / 'lerc_lossless.tif')
        # Tier 3 codec (issue #2137); opt in so the test exercises the
        # encode path rather than the rejection gate.
        to_geotiff(da, path, compression='lerc', max_z_error=0.0,
                   allow_experimental_codecs=True)

        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, arr)


class TestLossyShrinksAndStaysWithinTolerance:
    """A non-zero budget shrinks the file and bounds the per-pixel error."""

    def test_lossy_smaller_and_bounded(self, tmp_path):
        arr = _smooth_surface(128, 128)
        da = _make_dataarray(arr)

        lossless_path = str(tmp_path / 'lerc_lossless.tif')
        lossy_path = str(tmp_path / 'lerc_lossy.tif')

        to_geotiff(da, lossless_path, compression='lerc', max_z_error=0.0,
                   allow_experimental_codecs=True)
        to_geotiff(da, lossy_path, compression='lerc', max_z_error=0.05,
                   allow_experimental_codecs=True)

        lossless_size = os.path.getsize(lossless_path)
        lossy_size = os.path.getsize(lossy_path)
        assert lossy_size < lossless_size, (
            f"expected lossy file to be smaller, "
            f"got lossless={lossless_size} lossy={lossy_size}")

        result = open_geotiff(lossy_path, allow_experimental_codecs=True).values
        max_err = float(np.max(np.abs(result - arr)))
        assert max_err <= 0.05 + 1e-7, f"per-pixel error {max_err} exceeds budget"


class TestStreamingDaskPath:
    """Dask-backed input takes the streaming write path."""

    def test_dask_lerc_with_max_z_error(self, tmp_path):
        dask = pytest.importorskip('dask.array')
        arr = _smooth_surface(64, 64)
        darr = dask.from_array(arr, chunks=(32, 32))
        da = xr.DataArray(
            darr, dims=['y', 'x'],
            coords={'y': np.arange(64), 'x': np.arange(64)},
            attrs={'crs': 4326},
        )
        path = str(tmp_path / 'lerc_dask.tif')
        to_geotiff(da, path, compression='lerc', max_z_error=0.05,
                   tile_size=32, allow_experimental_codecs=True)

        result = open_geotiff(path, allow_experimental_codecs=True).values
        max_err = float(np.max(np.abs(result - arr)))
        assert max_err <= 0.05 + 1e-7


class TestValidation:
    """Up-front validation rejects misuse."""

    def test_max_z_error_with_non_lerc_codec_raises(self, tmp_path):
        arr = _smooth_surface(16, 16)
        da = _make_dataarray(arr)
        path = str(tmp_path / 'should_not_exist.tif')
        with pytest.raises(ValueError, match="max_z_error"):
            to_geotiff(da, path, compression='zstd', max_z_error=0.1)

    def test_negative_max_z_error_raises(self, tmp_path):
        arr = _smooth_surface(16, 16)
        da = _make_dataarray(arr)
        path = str(tmp_path / 'should_not_exist.tif')
        with pytest.raises(ValueError, match="max_z_error"):
            to_geotiff(da, path, compression='lerc', max_z_error=-0.01,
                       allow_experimental_codecs=True)

    def test_max_z_error_zero_with_other_codec_is_allowed(self, tmp_path):
        # The default value 0.0 must not error out for any other codec.
        arr = _smooth_surface(16, 16)
        da = _make_dataarray(arr)
        path = str(tmp_path / 'zstd_default.tif')
        to_geotiff(da, path, compression='zstd', max_z_error=0.0)
        result = open_geotiff(path, allow_experimental_codecs=True).values
        np.testing.assert_array_equal(result, arr)
