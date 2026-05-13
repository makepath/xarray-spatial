"""Regression tests for issue #1752.

Two public geotiff entry points used to accept size parameters without
checking they were positive:

* ``to_geotiff(..., tiled=True, tile_size=0)`` reached the tiled writer
  where ``math.ceil(width / tile_size)`` raised ``ZeroDivisionError``,
  with a traceback that did not name ``tile_size`` as the bad input.
* ``read_geotiff_dask(chunks=0)`` (or ``chunks=(0, N)``) propagated zero
  into dask's chunk math and surfaced as a confusing ``range()`` /
  empty-chunks error.

Both entry points now validate the size arguments up front and raise
``ValueError`` naming the parameter and the invalid value.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_geotiff_dask, to_geotiff


def _make_raster(tmp_path: str) -> str:
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(10), 'x': np.arange(10)},
        attrs={'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)},
    )
    path = os.path.join(tmp_path, 'raster.tif')
    to_geotiff(da, path)
    return path


# -- to_geotiff tile_size ---------------------------------------------------


def test_to_geotiff_tile_size_zero_raises(tmp_path):
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    da = xr.DataArray(arr, dims=['y', 'x'])
    out = os.path.join(str(tmp_path), 'out.tif')
    with pytest.raises(ValueError, match='tile_size'):
        to_geotiff(da, out, tiled=True, tile_size=0)


def test_to_geotiff_tile_size_negative_raises(tmp_path):
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    da = xr.DataArray(arr, dims=['y', 'x'])
    out = os.path.join(str(tmp_path), 'out.tif')
    with pytest.raises(ValueError, match='tile_size'):
        to_geotiff(da, out, tiled=True, tile_size=-1)


def test_to_geotiff_tile_size_non_int_raises(tmp_path):
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    da = xr.DataArray(arr, dims=['y', 'x'])
    out = os.path.join(str(tmp_path), 'out.tif')
    with pytest.raises(ValueError, match='tile_size'):
        to_geotiff(da, out, tiled=True, tile_size=256.0)


def test_to_geotiff_tile_size_16_writes(tmp_path):
    # ``tile_size=16`` is the smallest TIFF-spec-legal tile size. The
    # original 1752 regression checked ``tile_size=1`` here, but #1767
    # now requires multiples of 16 (TIFF 6 spec), so ``tile_size=1`` is
    # rejected. Keep a positive-path test at the new lower bound.
    arr = np.arange(256, dtype=np.float32).reshape(16, 16)
    da = xr.DataArray(arr, dims=['y', 'x'])
    out = os.path.join(str(tmp_path), 'out.tif')
    to_geotiff(da, out, tiled=True, tile_size=16)
    assert os.path.exists(out)


# -- read_geotiff_dask chunks ----------------------------------------------


def test_read_geotiff_dask_chunks_zero_raises(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='chunks'):
        read_geotiff_dask(path, chunks=0)


def test_read_geotiff_dask_chunks_negative_raises(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='chunks'):
        read_geotiff_dask(path, chunks=-1)


def test_read_geotiff_dask_chunks_tuple_zero_row_raises(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='chunks'):
        read_geotiff_dask(path, chunks=(0, 256))


def test_read_geotiff_dask_chunks_tuple_negative_col_raises(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='chunks'):
        read_geotiff_dask(path, chunks=(256, -1))


def test_read_geotiff_dask_chunks_tuple_wrong_length_raises(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='chunks'):
        read_geotiff_dask(path, chunks=(64, 64, 64))


def test_read_geotiff_dask_positive_int_chunks_works(tmp_path):
    path = _make_raster(str(tmp_path))
    arr = read_geotiff_dask(path, chunks=256)
    assert arr.shape == (10, 10)
    # Materialise to confirm the lazy graph is well-formed.
    np.asarray(arr)


def test_read_geotiff_dask_positive_tuple_chunks_works(tmp_path):
    path = _make_raster(str(tmp_path))
    arr = read_geotiff_dask(path, chunks=(4, 8))
    assert arr.shape == (10, 10)
    np.asarray(arr)


def test_read_geotiff_dask_numpy_int_scalar_chunks_works(tmp_path):
    # Numpy integer scalars (e.g. np.int64) should behave like plain
    # ``int`` for the scalar ``chunks`` form. The tuple branch already
    # accepts np.integer elements; the scalar branch was the gap.
    path = _make_raster(str(tmp_path))
    arr = read_geotiff_dask(path, chunks=np.int64(256))
    assert arr.shape == (10, 10)
    np.asarray(arr)


def test_read_geotiff_dask_numpy_int_tuple_chunks_works(tmp_path):
    path = _make_raster(str(tmp_path))
    arr = read_geotiff_dask(path, chunks=(np.int64(256), 256))
    assert arr.shape == (10, 10)
    np.asarray(arr)
