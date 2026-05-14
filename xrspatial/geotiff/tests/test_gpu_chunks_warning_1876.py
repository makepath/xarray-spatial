"""Regression tests for issue #1876.

``read_geotiff_gpu(chunks=...)`` and ``open_geotiff(gpu=True, chunks=...)``
decode the entire raster into a single CuPy array and then wrap it in a
Dask graph after the fact. The original docstring promised
out-of-core Dask+CuPy behaviour, which it does not deliver. Until lazy
per-chunk GPU decoding lands, the function emits a ``RuntimeWarning``
when a chunk shape smaller than the full raster is requested so users
do not assume the GPU memory footprint is bounded by chunk size.
"""
from __future__ import annotations

import importlib.util
import warnings

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


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
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


@_gpu_only
def test_read_geotiff_gpu_chunks_emits_warning(small_raster_path_1876):
    from xrspatial.geotiff import read_geotiff_gpu

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = read_geotiff_gpu(small_raster_path_1876, chunks=8)

    runtime_warnings = [w for w in caught
                        if issubclass(w.category, RuntimeWarning)
                        and 'chunks' in str(w.message).lower()]
    assert runtime_warnings, (
        "read_geotiff_gpu(chunks=...) should emit a RuntimeWarning "
        "about non-lazy decoding; got: "
        f"{[(w.category.__name__, str(w.message)) for w in caught]}"
    )
    assert 'out-of-core' in str(runtime_warnings[0].message).lower() or \
           'gpu memory' in str(runtime_warnings[0].message).lower()
    # The result should still be a chunked dask DataArray.
    import dask.array
    assert isinstance(result.data, dask.array.Array)


@_gpu_only
def test_read_geotiff_gpu_no_chunks_no_warning(small_raster_path_1876):
    from xrspatial.geotiff import read_geotiff_gpu

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        read_geotiff_gpu(small_raster_path_1876)

    chunks_warnings = [w for w in caught
                       if issubclass(w.category, RuntimeWarning)
                       and 'chunks' in str(w.message).lower()]
    assert not chunks_warnings, (
        "No chunks warning expected when chunks=None; got: "
        f"{[str(w.message) for w in chunks_warnings]}"
    )


@_gpu_only
def test_read_geotiff_gpu_chunks_equal_to_size_no_warning(small_raster_path_1876):
    """When chunks equal the full raster, there's no false out-of-core
    promise to warn about."""
    from xrspatial.geotiff import read_geotiff_gpu

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        read_geotiff_gpu(small_raster_path_1876, chunks=32)

    chunks_warnings = [w for w in caught
                       if issubclass(w.category, RuntimeWarning)
                       and 'chunks' in str(w.message).lower()]
    assert not chunks_warnings, (
        "No warning expected when chunk size equals raster size; got: "
        f"{[str(w.message) for w in chunks_warnings]}"
    )


@_gpu_only
def test_open_geotiff_gpu_chunks_warns(small_raster_path_1876):
    """The warning also fires through open_geotiff(gpu=True, chunks=...)."""
    from xrspatial.geotiff import open_geotiff

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        open_geotiff(small_raster_path_1876, gpu=True, chunks=8)

    runtime_warnings = [w for w in caught
                        if issubclass(w.category, RuntimeWarning)
                        and 'chunks' in str(w.message).lower()]
    assert runtime_warnings, (
        "open_geotiff(gpu=True, chunks=...) should propagate the "
        "RuntimeWarning from read_geotiff_gpu"
    )
