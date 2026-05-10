"""Regression tests for issue #1553.

Two ``arr = arr.copy()`` calls on freshly-allocated buffers were
removed from the read path in ``xrspatial/geotiff/__init__.py``
(``_geotiff_to_xarray`` and ``_delayed_read_window``). The copies
were unnecessary because the source array was uniquely owned at that
point, so dropping them halves peak memory for a nodata-bearing read
without changing observed values.

These tests exercise the float and integer nodata sentinel paths on
both tiled and strip TIFFs, and on the dask-chunked read path that
flows through ``_delayed_read_window``. They confirm:

- Sentinel pixels become NaN on read.
- Non-sentinel pixels round-trip exactly.
- The dropped copies do not alias any caller-visible buffer (a second
  read of the same file returns the expected mask, proving we did not
  smear NaNs into shared state across calls).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _make_float_with_sentinel(h=24, w=24, sentinel=-9999.0,
                              dtype=np.float32):
    """Float array with a few sentinel pixels at known positions."""
    rng = np.random.default_rng(seed=1553)
    arr = rng.uniform(0.0, 100.0, size=(h, w)).astype(dtype)
    arr[0, 0] = sentinel
    arr[5, 7] = sentinel
    arr[-1, -1] = sentinel
    return arr


def _make_uint16_with_sentinel(h=24, w=24, sentinel=65535):
    """uint16 array with a few sentinel pixels at known positions."""
    rng = np.random.default_rng(seed=1554)
    arr = rng.integers(0, 1000, size=(h, w), dtype=np.uint16)
    arr[0, 0] = sentinel
    arr[5, 7] = sentinel
    arr[-1, -1] = sentinel
    return arr


def test_float_sentinel_strip_tiff_read(tmp_path):
    """Strip-organized float32 TIFF with sentinel nodata reads correctly."""
    src = _make_float_with_sentinel()
    path = str(tmp_path / 'issue_1553_float_strip.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, tiled=False,
               compression='deflate')

    out = open_geotiff(path)
    expected_mask = (src == np.float32(-9999.0))
    np.testing.assert_array_equal(np.isnan(out.data), expected_mask)
    finite = ~expected_mask
    np.testing.assert_allclose(out.data[finite], src[finite])
    assert out.attrs.get('nodata') == -9999.0


def test_float_sentinel_tiled_tiff_read(tmp_path):
    """Tiled float32 TIFF with sentinel nodata reads correctly."""
    src = _make_float_with_sentinel(h=64, w=64)
    path = str(tmp_path / 'issue_1553_float_tiled.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, tiled=True, tile_size=16,
               compression='deflate')

    out = open_geotiff(path)
    expected_mask = (src == np.float32(-9999.0))
    np.testing.assert_array_equal(np.isnan(out.data), expected_mask)
    finite = ~expected_mask
    np.testing.assert_allclose(out.data[finite], src[finite])


def test_uint16_sentinel_tiled_tiff_read(tmp_path):
    """Tiled uint16 TIFF with sentinel nodata is promoted to float+NaN."""
    src = _make_uint16_with_sentinel(h=48, w=48)
    path = str(tmp_path / 'issue_1553_uint16_tiled.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': 65535})
    to_geotiff(da, path, nodata=65535, tiled=True, tile_size=16,
               compression='deflate')

    out = open_geotiff(path)
    assert out.dtype.kind == 'f'
    expected_mask = (src == 65535)
    np.testing.assert_array_equal(np.isnan(out.data), expected_mask)
    finite = ~expected_mask
    np.testing.assert_array_equal(out.data[finite].astype(np.uint16),
                                  src[finite])


def test_repeat_reads_independent(tmp_path):
    """Repeated reads return independent arrays with the correct mask.

    If the dropped ``.copy()`` had been protecting against shared
    state across reads, the second read would either see corrupted
    values (NaN bleeding into other slots) or share a buffer with the
    first call. We mutate the first result and then re-read to verify
    the file's contents are not affected.
    """
    src = _make_float_with_sentinel()
    path = str(tmp_path / 'issue_1553_repeat_reads.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, compression='deflate')

    first = open_geotiff(path)
    expected_mask = (src == np.float32(-9999.0))
    np.testing.assert_array_equal(np.isnan(first.data), expected_mask)

    # Mutate the first read in place. If subsequent reads share state
    # with this buffer, the second read will look corrupted.
    first.data[1, 1] = np.nan
    first.data[2, 2] = 12345.0

    second = open_geotiff(path)
    np.testing.assert_array_equal(np.isnan(second.data), expected_mask)
    finite = ~expected_mask
    np.testing.assert_allclose(second.data[finite], src[finite])


def test_dask_chunked_float_sentinel_read(tmp_path):
    """Dask-chunked read of a float TIFF with sentinel nodata.

    Exercises ``_delayed_read_window`` — the second site where the
    defensive copy was dropped. Each chunk runs the per-window nodata
    rewrite path in a worker task.
    """
    src = _make_float_with_sentinel(h=64, w=64)
    path = str(tmp_path / 'issue_1553_dask_float.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, tiled=True, tile_size=16,
               compression='deflate')

    out = open_geotiff(path, chunks=16)
    materialised = out.compute().data
    expected_mask = (src == np.float32(-9999.0))
    np.testing.assert_array_equal(np.isnan(materialised), expected_mask)
    finite = ~expected_mask
    np.testing.assert_allclose(materialised[finite], src[finite])


def test_dask_chunked_uint16_sentinel_read(tmp_path):
    """Dask-chunked read of a uint16 TIFF promotes to float+NaN per chunk."""
    src = _make_uint16_with_sentinel(h=64, w=64)
    path = str(tmp_path / 'issue_1553_dask_uint16.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': 65535})
    to_geotiff(da, path, nodata=65535, tiled=True, tile_size=16,
               compression='deflate')

    out = open_geotiff(path, chunks=16)
    materialised = out.compute().data
    assert materialised.dtype.kind == 'f'
    expected_mask = (src == 65535)
    np.testing.assert_array_equal(np.isnan(materialised), expected_mask)
    finite = ~expected_mask
    np.testing.assert_array_equal(materialised[finite].astype(np.uint16),
                                  src[finite])


def test_writer_does_not_mutate_caller_input(tmp_path):
    """``to_geotiff`` must not mutate the caller's input array.

    The defensive copies kept in ``to_geotiff`` and
    ``_write_single_tile`` exist for this reason. This test guards
    against accidentally dropping them in a future audit pass.
    """
    src = np.array([
        [1.0, 2.0, np.nan],
        [np.nan, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)
    snapshot = src.copy()
    path = str(tmp_path / 'issue_1553_writer_no_mutate.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, compression='deflate')

    # Caller's buffer must still hold its original NaNs and finite
    # values; the writer must not have stamped the sentinel value into
    # the user's array in place.
    np.testing.assert_array_equal(np.isnan(src), np.isnan(snapshot))
    finite = ~np.isnan(snapshot)
    np.testing.assert_array_equal(src[finite], snapshot[finite])
