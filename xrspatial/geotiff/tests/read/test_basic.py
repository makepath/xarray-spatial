"""Minimal reader paths: band validation, byte-band, eager read.

Consolidates the reader-side band-validation regression coverage
(formerly ``test_band_validation_1673.py``). The contract is that every
backend rejects out-of-range ``band`` arguments with the same typed
``IndexError`` so callers see consistent diagnostics regardless of
which path they pick.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def multiband_tiff_path(tmp_path):
    """4x6 three-band tiled tiff for band-validation tests."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(72, dtype=np.float32).reshape(4, 6, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            'band': [0, 1, 2],
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'mb_band_validation.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


class TestBandValidationLocal:
    """``read_to_array`` rejects out-of-range band indices."""

    def test_negative_band_rejected(self, multiband_tiff_path):
        """``band=-1`` no longer silently selects the last channel."""
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path
        with pytest.raises(IndexError, match="band=-1 out of range"):
            read_to_array(path, band=-1)

    def test_band_equal_to_samples_rejected(self, multiband_tiff_path):
        """``band=samples_per_pixel`` (off-by-one) raises a typed error."""
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path
        with pytest.raises(IndexError, match="band=3 out of range"):
            read_to_array(path, band=3)

    def test_band_far_above_samples_rejected(self, multiband_tiff_path):
        """A wildly out-of-range band index gives the same typed error."""
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path
        with pytest.raises(IndexError, match="band=103 out of range"):
            read_to_array(path, band=103)

    def test_valid_band_still_works(self, multiband_tiff_path):
        """Valid band indices keep working after the validation guard."""
        from xrspatial.geotiff._reader import read_to_array

        path, arr = multiband_tiff_path
        out, _ = read_to_array(path, band=1)
        np.testing.assert_array_equal(out, arr[:, :, 1])

    def test_band_none_returns_all_bands(self, multiband_tiff_path):
        """``band=None`` still returns the full multi-band array."""
        from xrspatial.geotiff._reader import read_to_array

        path, arr = multiband_tiff_path
        out, _ = read_to_array(path)
        np.testing.assert_array_equal(out, arr)


class TestBandValidationBackendParity:
    """Local eager and dask paths agree on the rejection contract."""

    def test_negative_band(self, multiband_tiff_path):
        """Both paths raise the same error for ``band=-1``."""
        from xrspatial.geotiff import read_geotiff_dask
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path

        with pytest.raises(IndexError) as eager_exc:
            read_to_array(path, band=-1)
        with pytest.raises(IndexError) as dask_exc:
            read_geotiff_dask(path, chunks=4, band=-1)

        assert "band=-1 out of range" in str(eager_exc.value)
        assert "band=-1 out of range" in str(dask_exc.value)

    def test_band_equal_to_samples(self, multiband_tiff_path):
        """Both paths agree on the off-by-one rejection."""
        from xrspatial.geotiff import read_geotiff_dask
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path

        with pytest.raises(IndexError) as eager_exc:
            read_to_array(path, band=3)
        with pytest.raises(IndexError) as dask_exc:
            read_geotiff_dask(path, chunks=4, band=3)

        assert "band=3 out of range" in str(eager_exc.value)
        assert "band=3 out of range" in str(dask_exc.value)
