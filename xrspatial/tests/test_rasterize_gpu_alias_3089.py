"""Issue #3089: rasterize() renamed its GPU opt-in from ``use_cuda`` to
``gpu`` to match ``open_geotiff(gpu=True)``.

``use_cuda`` stays as a deprecated keyword alias: it still selects the GPU
backend but emits a ``DeprecationWarning``, and combining it with
``gpu=True`` raises ``TypeError``.  These tests pin the shim and the
positional-compatibility guarantee (``gpu`` occupies the slot ``use_cuda``
used to, so positional callers are unaffected).
"""
import inspect
import warnings

import numpy as np
import pytest

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

try:
    import cupy
    from numba import cuda
    has_gpu = cuda.is_available()
except ImportError:
    cupy = None
    has_gpu = False

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

requires_gpu = pytest.mark.skipif(
    not has_gpu, reason="CUDA GPU not available")


def _pairs():
    return [(box(0, 0, 5, 5), 1.0), (box(3, 3, 9, 9), 2.0)]


def _kw():
    return dict(width=10, height=10, bounds=(0, 0, 10, 10),
                merge='sum', fill=0)


def _to_numpy(result):
    data = result.data
    if hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)


def test_gpu_is_tenth_positional_param():
    """``gpu`` must sit exactly where ``use_cuda`` used to, so existing
    positional callers keep selecting the GPU backend."""
    params = list(inspect.signature(rasterize).parameters)
    assert params[9] == 'gpu'
    # the deprecated alias is appended last so it shifts nothing
    assert params[-1] == 'use_cuda'


def test_default_emits_no_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        rasterize(_pairs(), **_kw())


def test_use_cuda_false_warns_and_runs_cpu():
    with pytest.warns(DeprecationWarning, match='use gpu='):
        result = rasterize(_pairs(), use_cuda=False, **_kw())
    assert isinstance(result.data, np.ndarray)


def test_gpu_true_and_use_cuda_raises():
    with pytest.raises(TypeError, match="deprecated alias"):
        rasterize(_pairs(), gpu=True, use_cuda=True, **_kw())
    with pytest.raises(TypeError, match="deprecated alias"):
        rasterize(_pairs(), gpu=True, use_cuda=False, **_kw())


@requires_gpu
def test_gpu_true_matches_numpy():
    expected = rasterize(_pairs(), **_kw())
    result = rasterize(_pairs(), gpu=True, **_kw())
    assert isinstance(result.data, cupy.ndarray)
    np.testing.assert_array_equal(_to_numpy(result), expected.data)


@requires_gpu
def test_use_cuda_true_warns_and_matches_gpu_true():
    expected = rasterize(_pairs(), gpu=True, **_kw())
    with pytest.warns(DeprecationWarning, match='use gpu='):
        result = rasterize(_pairs(), use_cuda=True, **_kw())
    assert isinstance(result.data, cupy.ndarray)
    np.testing.assert_array_equal(_to_numpy(result), _to_numpy(expected))


@requires_gpu
def test_dask_cupy_via_gpu_and_alias():
    import dask.array as da

    lazy = rasterize(_pairs(), gpu=True, chunks=5, **_kw())
    assert isinstance(lazy.data, da.Array)
    assert isinstance(lazy.data._meta, cupy.ndarray)

    with pytest.warns(DeprecationWarning, match='use gpu='):
        lazy_alias = rasterize(_pairs(), use_cuda=True, chunks=5, **_kw())
    assert isinstance(lazy_alias.data._meta, cupy.ndarray)

    np.testing.assert_array_equal(
        _to_numpy(lazy.compute()), _to_numpy(lazy_alias.compute()))


@requires_gpu
def test_clip_polygon_dask_cupy_emits_no_deprecation_warning():
    """polygon_clip's internal rasterize call must use the new name."""
    import dask.array as da
    import xarray as xr

    from xrspatial.polygon_clip import clip_polygon

    data = cupy.ones((10, 10), dtype=cupy.float64)
    raster = xr.DataArray(
        da.from_array(data, chunks=(5, 5)),
        dims=['y', 'x'],
        coords={'y': np.linspace(9.5, 0.5, 10),
                'x': np.linspace(0.5, 9.5, 10)},
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        clipped = clip_polygon(raster, box(2, 2, 8, 8))
    assert clipped is not None
