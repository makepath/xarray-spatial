"""Regression tests for issue #1876.

``read_geotiff_gpu(chunks=...)`` used to eagerly decode the full raster
into a single CuPy array and then call ``.chunk()`` on the resulting
DataArray, so peak GPU memory was the whole-image size even though the
docstring advertised an out-of-core pipeline. The function now
dispatches to a real Dask+CuPy graph that decodes one chunk window at
a time and uploads each block to the device, so peak GPU memory is
bounded by chunk size.

Two paths back that promise. When ``kvikio`` is available and the file
is a local, tiled, chunky, non-sparse GeoTIFF with trivial orientation,
each chunk task pulls its tile subset directly from disk to GPU via
GDS. Otherwise the per-chunk window is decoded on CPU via the existing
``read_geotiff_dask`` graph and uploaded with ``cupy.asarray``. Both
paths keep peak device memory at one chunk.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def _kvikio_available() -> bool:
    return importlib.util.find_spec("kvikio") is not None


_HAS_GPU = _gpu_available()
_HAS_KVIKIO = _kvikio_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)
_gds_only = pytest.mark.skipif(
    not (_HAS_GPU and _HAS_KVIKIO),
    reason="cupy + CUDA + kvikio required for GDS path",
)


@pytest.fixture
def small_raster_path_1876(tmp_path):
    from xrspatial.geotiff import to_geotiff
    import xarray as xr

    arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      attrs={'crs': 4326,
                             'transform': (1.0, 0, 0, 0, -1.0, 32.0)})
    path = str(tmp_path / 'small_raster_1876.tif')
    to_geotiff(da, path, compression='deflate', tile_size=16)
    return path


@pytest.fixture
def multi_band_path_1876(tmp_path):
    from xrspatial.geotiff import to_geotiff
    import xarray as xr

    rng = np.random.RandomState(42)
    arr = rng.rand(3, 32, 32).astype(np.float32)
    da = xr.DataArray(arr, dims=['band', 'y', 'x'],
                      attrs={'crs': 4326,
                             'transform': (1.0, 0, 0, 0, -1.0, 32.0)})
    path = str(tmp_path / 'multi_band_1876.tif')
    to_geotiff(da, path, compression='deflate', tile_size=16)
    return path


@_gpu_only
def test_read_geotiff_gpu_chunks_yields_dask_cupy_chunks(small_raster_path_1876):
    """Each block of the returned dask array must be a cupy array, not
    a numpy array and not a single eager cupy block."""
    import cupy
    import dask.array as da_mod

    from xrspatial.geotiff import read_geotiff_gpu

    result = read_geotiff_gpu(small_raster_path_1876, chunks=8)

    assert isinstance(result.data, da_mod.Array), (
        f"expected dask Array, got {type(result.data).__name__}"
    )
    assert isinstance(result.data._meta, cupy.ndarray), (
        f"expected cupy chunks, got meta={type(result.data._meta).__name__}"
    )
    assert result.data.numblocks == (4, 4)

    block = result.data.blocks[0, 0].compute()
    assert isinstance(block, cupy.ndarray)
    assert block.shape == (8, 8)


@_gpu_only
def test_read_geotiff_gpu_chunks_values_match_eager(small_raster_path_1876):
    """Lazy chunked result must equal the eager GPU result element-wise."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu

    eager = read_geotiff_gpu(small_raster_path_1876)
    chunked = read_geotiff_gpu(small_raster_path_1876, chunks=8)

    eager_np = cupy.asnumpy(eager.data)
    chunked_np = cupy.asnumpy(chunked.compute().data)
    np.testing.assert_array_equal(eager_np, chunked_np)


@_gpu_only
def test_read_geotiff_gpu_no_chunks_returns_eager_cupy(small_raster_path_1876):
    """``chunks=None`` keeps the eager GPU decode path."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu

    result = read_geotiff_gpu(small_raster_path_1876)

    assert isinstance(result.data, cupy.ndarray)


@_gpu_only
def test_open_geotiff_gpu_chunks_propagates_to_dask(small_raster_path_1876):
    """``open_geotiff(gpu=True, chunks=...)`` must return the same
    Dask+CuPy result as the direct read."""
    import cupy
    import dask.array as da_mod

    from xrspatial.geotiff import open_geotiff

    result = open_geotiff(small_raster_path_1876, gpu=True, chunks=8)

    assert isinstance(result.data, da_mod.Array)
    assert isinstance(result.data._meta, cupy.ndarray)


@_gpu_only
def test_read_geotiff_gpu_chunks_preserves_attrs(small_raster_path_1876):
    """Geo attrs (transform, crs) must survive the dask path."""
    from xrspatial.geotiff import read_geotiff_gpu

    result = read_geotiff_gpu(small_raster_path_1876, chunks=8)
    assert 'transform' in result.attrs
    assert 'crs' in result.attrs


@_gds_only
def test_read_geotiff_gpu_chunks_uses_gds_path_when_available(
        small_raster_path_1876, monkeypatch):
    """When kvikio is installed and the file qualifies (local + tiled +
    chunky + no sparse + orientation=1 + photometric!=0), each chunk
    task must call the direct disk->GPU decoder rather than detouring
    through ``read_geotiff_dask``."""
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gtmod

    direct_calls = {'n': 0}
    real_direct = gtmod._decode_window_gpu_direct

    def _spy(*args, **kwargs):
        direct_calls['n'] += 1
        return real_direct(*args, **kwargs)

    monkeypatch.setattr(gtmod, '_decode_window_gpu_direct', _spy)

    result = read_geotiff_gpu(small_raster_path_1876, chunks=8)
    result.compute()

    assert direct_calls['n'] == 16, (
        f"expected one disk->GPU call per chunk (4x4 = 16); "
        f"got {direct_calls['n']}"
    )


@_gpu_only
def test_read_geotiff_gpu_chunks_window_subset(small_raster_path_1876):
    """A window on the dask path produces the same values as a window
    on the eager path."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu

    eager = read_geotiff_gpu(small_raster_path_1876, window=(4, 4, 24, 28))
    chunked = read_geotiff_gpu(small_raster_path_1876, chunks=8,
                               window=(4, 4, 24, 28))

    eager_np = cupy.asnumpy(eager.data)
    chunked_np = cupy.asnumpy(chunked.compute().data)
    assert eager_np.shape == (20, 24)
    np.testing.assert_array_equal(eager_np, chunked_np)


@_gpu_only
def test_read_geotiff_gpu_chunks_multi_band(multi_band_path_1876):
    """Multi-band tiled files chunk along (y, x) with a band axis."""
    import cupy
    import dask.array as da_mod

    from xrspatial.geotiff import read_geotiff_gpu

    result = read_geotiff_gpu(multi_band_path_1876, chunks=16)
    assert isinstance(result.data, da_mod.Array)
    assert isinstance(result.data._meta, cupy.ndarray)
    assert result.sizes['band'] == 3

    eager = read_geotiff_gpu(multi_band_path_1876)
    eager_np = cupy.asnumpy(eager.data)
    chunked_np = cupy.asnumpy(result.compute().data)
    np.testing.assert_allclose(eager_np, chunked_np, rtol=1e-5)


@_gpu_only
def test_read_geotiff_gpu_chunks_single_band_selection(multi_band_path_1876):
    """``band=k`` collapses to a 2D Dask+CuPy DataArray."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu

    result = read_geotiff_gpu(multi_band_path_1876, chunks=16, band=1)
    assert result.ndim == 2
    assert isinstance(result.data._meta, cupy.ndarray)

    eager = read_geotiff_gpu(multi_band_path_1876, band=1)
    eager_np = cupy.asnumpy(eager.data)
    chunked_np = cupy.asnumpy(result.compute().data)
    np.testing.assert_allclose(eager_np, chunked_np, rtol=1e-5)


@_gpu_only
def test_read_geotiff_gpu_chunks_fallback_when_kvikio_absent(
        small_raster_path_1876, monkeypatch):
    """When kvikio is reported missing, the chunked path falls back to
    the CPU-decode + cupy.asarray graph and still produces a Dask+CuPy
    DataArray with correct values."""
    import cupy
    import importlib.util as _ilu

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gtmod

    original_find_spec = _ilu.find_spec

    def _fake_find_spec(name, *a, **kw):
        if name == 'kvikio':
            return None
        return original_find_spec(name, *a, **kw)

    monkeypatch.setattr(_ilu, 'find_spec', _fake_find_spec)

    direct_calls = {'n': 0}
    real_direct = gtmod._decode_window_gpu_direct

    def _spy(*args, **kwargs):
        direct_calls['n'] += 1
        return real_direct(*args, **kwargs)

    monkeypatch.setattr(gtmod, '_decode_window_gpu_direct', _spy)

    result = read_geotiff_gpu(small_raster_path_1876, chunks=8)
    computed = result.compute()
    assert direct_calls['n'] == 0
    assert isinstance(computed.data, cupy.ndarray)

    eager = read_geotiff_gpu(small_raster_path_1876)
    np.testing.assert_array_equal(
        cupy.asnumpy(eager.data), cupy.asnumpy(computed.data),
    )
