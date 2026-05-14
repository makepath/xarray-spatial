"""Regression tests for ``photometric='miniswhite'`` round-tripping.

The reader unconditionally inverts single-band MinIsWhite data at
``_reader._apply_photometric_miniswhite``. Before issue #1836 the writer
set the TIFF Photometric tag to 0 without inverting pixel values, so
``to_geotiff(..., photometric='miniswhite')`` followed by ``open_geotiff``
returned ``iinfo(dtype).max - original`` instead of the user's values.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _da(arr: np.ndarray) -> xr.DataArray:
    h, w = arr.shape
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(h, dtype=np.float64),
                'x': np.arange(w, dtype=np.float64)},
        attrs={'res': (1.0, 1.0)},
    )


def test_uint8_miniswhite_round_trip(tmp_path):
    arr = np.array([[0, 1, 127, 254, 255]], dtype=np.uint8)
    path = tmp_path / 'u8_msw_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    np.testing.assert_array_equal(np.asarray(r.values), arr)


def test_uint16_miniswhite_round_trip(tmp_path):
    info = np.iinfo(np.uint16)
    arr = np.array([[0, 1, info.max // 2, info.max - 1, info.max]],
                   dtype=np.uint16)
    path = tmp_path / 'u16_msw_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    np.testing.assert_array_equal(np.asarray(r.values), arr)


def test_float32_miniswhite_round_trip(tmp_path):
    arr = np.array([[-3.5, 0.0, 0.25, 7.5]], dtype=np.float32)
    path = tmp_path / 'f32_msw_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    np.testing.assert_allclose(np.asarray(r.values), arr)


def test_miniswhite_with_nodata_round_trip(tmp_path):
    arr = np.array([[10.0, np.nan, 20.0, 30.0]], dtype=np.float32)
    path = tmp_path / 'f32_msw_nd_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite', nodata=-9999.0)
    r = open_geotiff(str(path))
    out = np.asarray(r.values)
    assert np.isnan(out[0, 1]), (
        "nodata position must round-trip back to NaN after MinIsWhite "
        f"inversion; got {out!r}"
    )
    np.testing.assert_allclose(out[0, [0, 2, 3]], arr[0, [0, 2, 3]])


def test_miniswhite_int_passthrough(tmp_path):
    """The reader does not invert signed integer MinIsWhite data, so the
    writer must also pass it through unchanged. Otherwise the round-trip
    would silently corrupt signed data."""
    arr = np.array([[-5, -1, 0, 1, 5]], dtype=np.int16)
    path = tmp_path / 'i16_msw_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    np.testing.assert_array_equal(np.asarray(r.values), arr)


def test_miniswhite_rejected_with_cog(tmp_path):
    arr = np.zeros((512, 512), dtype=np.uint8)
    path = tmp_path / 'cog_msw_1836.tif'
    with pytest.raises(NotImplementedError, match='miniswhite'):
        to_geotiff(_da(arr), str(path), photometric='miniswhite', cog=True)


def test_miniswhite_rejected_with_explicit_overviews(tmp_path):
    arr = np.zeros((256, 256), dtype=np.uint8)
    path = tmp_path / 'ov_msw_1836.tif'
    with pytest.raises(NotImplementedError, match='miniswhite'):
        to_geotiff(_da(arr), str(path), photometric='miniswhite',
                   cog=True, overview_levels=[2, 4])
