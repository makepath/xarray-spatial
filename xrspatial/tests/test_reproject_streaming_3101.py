"""Tests for the streaming fallback path of reproject() (issue #3101).

reproject() falls back to ``_reproject_streaming`` when the source is
larger than 512 MB and dask is not importable. That branch never runs in
CI organically (dask is installed and test rasters are small), so these
tests call the helpers directly with small inputs and compare against the
in-memory numpy path on the same output grid.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import xarray as xr

try:
    import pyproj  # noqa: F401
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

pytestmark = pytest.mark.skipif(
    not HAS_PYPROJ, reason="pyproj required for reproject tests"
)

# ``from xrspatial import reproject`` resolves to the function, not the
# package, so load the package module explicitly.
_reproject_mod = importlib.import_module('xrspatial.reproject')


def _make_raster(data, crs='EPSG:4326', x_range=(-5, 5), y_range=(45, 55)):
    h, w = data.shape[:2]
    y = np.linspace(y_range[1], y_range[0], h)  # north-up (descending)
    x = np.linspace(x_range[0], x_range[1], w)
    dims = ['y', 'x'] if data.ndim == 2 else ['y', 'x', 'band']
    coords = {'y': y, 'x': x}
    if data.ndim == 3:
        coords['band'] = np.arange(data.shape[2]) + 1
    return xr.DataArray(
        data, dims=dims, coords=coords,
        attrs={'crs': crs, 'nodata': np.nan},
    )


def _streaming_setup(raster, target_crs='EPSG:32633'):
    """Resolve the source/target geometry the way reproject() does."""
    from xrspatial.reproject._crs_utils import _resolve_crs
    from xrspatial.reproject._grid import _compute_output_grid

    src_crs = _resolve_crs(raster.attrs['crs'])
    tgt_crs = _resolve_crs(target_crs)
    src_bounds = _reproject_mod._source_bounds(raster)
    ydim, xdim = _reproject_mod._find_spatial_dims(raster)
    src_shape = (raster.sizes[ydim], raster.sizes[xdim])
    grid = _compute_output_grid(src_bounds, src_shape, src_crs, tgt_crs)
    return src_crs, tgt_crs, src_bounds, src_shape, grid


def _run_streaming(raster, tile_size, max_memory, resampling='bilinear',
                   precision=16):
    src_crs, tgt_crs, src_bounds, src_shape, grid = _streaming_setup(raster)
    return _reproject_mod._reproject_streaming(
        raster, src_bounds, src_shape, True,
        src_crs.to_wkt(), tgt_crs.to_wkt(),
        grid['bounds'], grid['shape'],
        resampling, float('nan'), precision,
        tile_size, _reproject_mod._parse_max_memory(max_memory),
        x_desc=False, band_nodata=None,
    )


class TestParseMaxMemory:
    """_parse_max_memory accepts ints and human-readable strings."""

    @pytest.mark.parametrize('value,expected', [
        (None, 1024 ** 3),               # default: 1 GB
        (12345, 12345),                  # int passthrough
        (12345.6, 12345),                # float truncates to int
        ('256KB', 256 * 1024),
        ('512MB', 512 * 1024 ** 2),
        ('4GB', 4 * 1024 ** 3),
        ('2TB', 2 * 1024 ** 4),
        ('1.5GB', int(1.5 * 1024 ** 3)),
        (' 1gb ', 1024 ** 3),            # whitespace + lowercase
        ('123', 123),                    # bare numeric string
    ])
    def test_parse(self, value, expected):
        assert _reproject_mod._parse_max_memory(value) == expected


class TestStreamingMatchesInMemory:
    """The streaming tile assembly must reproduce the in-memory result."""

    def _reference(self, raster):
        from xrspatial.reproject import reproject
        return reproject(raster, 'EPSG:32633')

    @staticmethod
    def _max_concurrent(max_memory, tile_size):
        # Mirror of the budget arithmetic in _reproject_streaming /
        # _process_tile_batch so the tests can assert which batching
        # branch they engage. If the tile_mem formula changes upstream,
        # these assertions flag the re-routed branch instead of both
        # tests silently exercising the same one.
        tile_mem = tile_size * tile_size * 8 * 4
        budget = _reproject_mod._parse_max_memory(max_memory)
        return max(1, budget // max(tile_mem, 1))

    def test_multi_tile_threaded(self):
        # 64x64 output split into 32x32 tiles; 1 GB budget keeps the
        # ThreadPoolExecutor branch active (max_concurrent >= 2).
        #
        # transform_precision=0 forces the exact pyproj path (one
        # Transformer per worker thread). The numba fast-path kernels are
        # parallel=True, and running them concurrently from this thread
        # pool aborts the interpreter on numba's workqueue threading
        # layer (macOS arm64 CI) -- that source bug is #3141. The numba
        # path stays covered by the serial tests below.
        assert self._max_concurrent('1GB', 32) >= 2
        data = np.random.RandomState(42).rand(64, 64)
        raster = _make_raster(data)
        from xrspatial.reproject import reproject
        ref = reproject(raster, 'EPSG:32633', transform_precision=0)
        out = _run_streaming(raster, tile_size=32, max_memory='1GB',
                             precision=0)
        assert out.shape == ref.shape
        # More than one tile, or the thread pool never gets a batch.
        assert ref.shape[0] > 32 or ref.shape[1] > 32
        np.testing.assert_allclose(out, ref.values, equal_nan=True)

    def test_multi_tile_serial(self):
        # max_memory=1 byte forces max_concurrent == 1, taking the serial
        # per-job loop instead of the thread pool.
        assert self._max_concurrent(1, 32) == 1
        data = np.random.RandomState(7).rand(64, 64)
        raster = _make_raster(data)
        ref = self._reference(raster)
        out = _run_streaming(raster, tile_size=32, max_memory=1)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref.values, equal_nan=True)

    def test_single_tile(self):
        # tile_size larger than the output grid: one job covers everything.
        data = np.random.RandomState(3).rand(32, 32)
        raster = _make_raster(data)
        ref = self._reference(raster)
        out = _run_streaming(raster, tile_size=4096, max_memory='1GB')
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref.values, equal_nan=True)

    def test_nearest_resampling_with_nan(self):
        # NaN cells survive the streaming assembly the same way they do
        # in-memory. max_memory=1 keeps this multi-tile run on the serial
        # loop so the parallel=True numba kernels are never entered
        # concurrently (#3141) while still covering the numba fast path.
        from xrspatial.reproject import reproject
        data = np.random.RandomState(11).rand(48, 48)
        data[10:14, 20:24] = np.nan
        raster = _make_raster(data)
        ref = reproject(raster, 'EPSG:32633', resampling='nearest')
        out = _run_streaming(raster, tile_size=20, max_memory=1,
                             resampling='nearest')
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref.values, equal_nan=True)


class TestStreaming3D:
    """3-D sources stream with the band axis intact (#3100, fixed in #3111)."""

    def test_3d_source_streams(self):
        from xrspatial.reproject import reproject
        data = np.random.RandomState(5).rand(32, 32, 3)
        raster = _make_raster(data)
        # max_memory=1 -> serial loop, keeping clear of the concurrent
        # parallel-kernel abort from #3141.
        out = _run_streaming(raster, tile_size=16, max_memory=1)
        # The streaming result must match the in-memory 3-D path:
        ref = reproject(raster, 'EPSG:32633')
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref.values, equal_nan=True)
