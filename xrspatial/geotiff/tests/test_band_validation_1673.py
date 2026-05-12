"""Regression tests for issue #1673.

``read_to_array`` accepts a ``band`` argument and applies it to the
decoded array via ``arr[:, :, band]`` without validating the index.
Two failure modes follow:

* ``band=-1`` silently selects the last channel via numpy negative
  indexing. The public contract is "0-based non-negative index", so
  this is a silent semantic shift, not an explicit selection.
* ``band=N`` with ``N >= samples_per_pixel`` raises a raw numpy
  ``IndexError`` whose message ("index N is out of bounds for axis
  2 with size M") leaks the internal slice shape.

The dask path (``read_geotiff_dask``) and the GPU path both validate
``band`` up front and raise ``IndexError("band=N out of range for
M-band file.")``. These tests pin the local eager path to the same
contract so backend parity holds.
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
    p = tmp_path / 'mb_1673.tif'
    to_geotiff(da, str(p), tile_size=4)
    return str(p), arr


def test_read_to_array_negative_band_rejected(multiband_tiff_path):
    """``band=-1`` no longer silently selects the last channel."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path
    with pytest.raises(IndexError, match="band=-1 out of range"):
        read_to_array(path, band=-1)


def test_read_to_array_band_equal_to_samples_rejected(multiband_tiff_path):
    """``band=samples_per_pixel`` (off-by-one) raises a typed error."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path
    # File has 3 bands; valid indices are 0, 1, 2.
    with pytest.raises(IndexError, match="band=3 out of range"):
        read_to_array(path, band=3)


def test_read_to_array_band_far_above_samples_rejected(multiband_tiff_path):
    """A wildly out-of-range band index gives the same typed error."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path
    with pytest.raises(IndexError, match="band=103 out of range"):
        read_to_array(path, band=103)


def test_read_to_array_valid_band_still_works(multiband_tiff_path):
    """Valid band indices keep working after the validation guard."""
    from xrspatial.geotiff._reader import read_to_array

    path, arr = multiband_tiff_path
    out, _ = read_to_array(path, band=1)
    np.testing.assert_array_equal(out, arr[:, :, 1])


def test_read_to_array_band_none_still_returns_all_bands(multiband_tiff_path):
    """``band=None`` still returns the full multi-band array."""
    from xrspatial.geotiff._reader import read_to_array

    path, arr = multiband_tiff_path
    out, _ = read_to_array(path)
    np.testing.assert_array_equal(out, arr)


def test_backend_parity_negative_band(multiband_tiff_path):
    """Local eager and dask paths raise the same error for ``band=-1``."""
    from xrspatial.geotiff import read_geotiff_dask
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path

    with pytest.raises(IndexError) as eager_exc:
        read_to_array(path, band=-1)
    with pytest.raises(IndexError) as dask_exc:
        read_geotiff_dask(path, chunks=4, band=-1)

    # Same error type and same diagnostic substring; the dask message
    # is "band=-1 out of range for 3-band file." so any 0-based caller
    # gets identical signal regardless of which backend they pick.
    assert "band=-1 out of range" in str(eager_exc.value)
    assert "band=-1 out of range" in str(dask_exc.value)


def test_backend_parity_band_equal_to_samples(multiband_tiff_path):
    """Local eager and dask paths agree on the off-by-one rejection."""
    from xrspatial.geotiff import read_geotiff_dask
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path

    with pytest.raises(IndexError) as eager_exc:
        read_to_array(path, band=3)
    with pytest.raises(IndexError) as dask_exc:
        read_geotiff_dask(path, chunks=4, band=3)

    assert "band=3 out of range" in str(eager_exc.value)
    assert "band=3 out of range" in str(dask_exc.value)
