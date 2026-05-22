"""Tests for sparse-block GeoTIFF reads.

GDAL's ``SPARSE_OK=TRUE`` creation option records all-nodata tiles or
strips with ``TileByteCounts == 0`` and ``TileOffsets == 0`` (same for
strip layout).  The reader has to materialise such blocks as the file's
nodata value (or zero when no nodata is set), not crash, and not return
uninitialised memory.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')

from xrspatial.geotiff import open_geotiff  # noqa: E402
from xrspatial.geotiff._reader import read_to_array  # noqa: E402


def _write_sparse_tiled(path, *, dtype='uint16', nodata=0,
                        compress='DEFLATE', filled_value=100):
    """Write a 128x128 raster where only the top-left 64x64 tile is filled."""
    profile = {
        'driver': 'GTiff', 'dtype': dtype,
        'height': 128, 'width': 128, 'count': 1,
        'tiled': True, 'blockxsize': 64, 'blockysize': 64,
        'compress': compress, 'SPARSE_OK': 'TRUE',
    }
    if nodata is not None:
        profile['nodata'] = nodata
    fill = np.full((64, 64), filled_value, dtype=np.dtype(dtype))
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(fill, 1, window=rasterio.windows.Window(0, 0, 64, 64))


def _write_sparse_stripped(path, *, dtype='uint16', nodata=0):
    profile = {
        'driver': 'GTiff', 'dtype': dtype,
        'height': 128, 'width': 128, 'count': 1,
        'tiled': False, 'blockysize': 16,
        'compress': 'DEFLATE', 'SPARSE_OK': 'TRUE',
    }
    if nodata is not None:
        profile['nodata'] = nodata
    fill = np.full((32, 128), 200, dtype=np.dtype(dtype))
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(fill, 1, window=rasterio.windows.Window(0, 0, 128, 32))


class TestSparseTiles:
    def test_sparse_tile_with_nodata_round_trips(self, tmp_path):
        path = str(tmp_path / 'sparse_nodata.tif')
        _write_sparse_tiled(path, nodata=0)

        arr = open_geotiff(path)
        arr_np = np.asarray(arr)
        # Filled tile carries the data, sparse tiles are NaN (nodata=0
        # promoted by the accessor).
        assert arr_np[:64, :64].sum() == 64 * 64 * 100
        assert np.all(np.isnan(arr_np[:64, 64:]))
        assert np.all(np.isnan(arr_np[64:, :]))

    def test_sparse_tile_without_nodata_fills_zero(self, tmp_path):
        path = str(tmp_path / 'sparse_no_nodata.tif')
        _write_sparse_tiled(path, nodata=None)

        arr = open_geotiff(path)
        arr_np = np.asarray(arr)
        assert arr_np.dtype == np.uint16
        assert arr_np[:64, :64].sum() == 64 * 64 * 100
        assert arr_np[:64, 64:].sum() == 0
        assert arr_np[64:, :].sum() == 0

    def test_sparse_tile_raw_read_uses_nodata_sentinel(self, tmp_path):
        """``read_to_array`` returns raw values; sparse tiles == nodata."""
        path = str(tmp_path / 'sparse_raw.tif')
        _write_sparse_tiled(path, nodata=255)

        arr, geo = read_to_array(path)
        assert arr.dtype == np.uint16
        assert arr[:64, :64].sum() == 64 * 64 * 100
        assert np.all(arr[:64, 64:] == 255)
        assert np.all(arr[64:, :] == 255)


class TestSparseStrips:
    def test_sparse_strip_with_nodata(self, tmp_path):
        path = str(tmp_path / 'sparse_strips.tif')
        _write_sparse_stripped(path, nodata=0)

        arr = open_geotiff(path)
        arr_np = np.asarray(arr)
        assert arr_np[:32, :].sum() == 32 * 128 * 200
        # Empty strips below row 32 land on nodata -> NaN via accessor
        assert np.all(np.isnan(arr_np[32:, :]))


@pytest.mark.skipif(
    not pytest.importorskip('cupy', reason='cupy required'),
    reason='cupy required',
)
class TestSparseTilesGPU:
    def test_sparse_tile_gpu_round_trip(self, tmp_path):
        path = str(tmp_path / 'sparse_gpu.tif')
        _write_sparse_tiled(path, nodata=0)

        arr = open_geotiff(path, gpu=True)
        # GPU read now applies the high-level nodata mask (#1542): the
        # source uint16 raster is promoted to float64 and sentinel
        # values become NaN, matching the CPU eager path.
        host = arr.data.get()
        assert host.dtype == np.float64
        assert arr.attrs.get('nodata') == 0.0
        # Top-left tile holds the real data (100 in every cell).
        np.testing.assert_array_equal(host[:64, :64],
                                      np.full((64, 64), 100.0))
        # Sparse tiles + the bottom row of tiles become NaN via the mask.
        assert np.all(np.isnan(host[:64, 64:]))
        assert np.all(np.isnan(host[64:, :]))
