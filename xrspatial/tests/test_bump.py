try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import xarray as xr

from xrspatial import bump
from xrspatial.tests.general_checks import (
    cuda_and_cupy_available,
    dask_array_available,
)
from xrspatial.utils import has_cuda_and_cupy


def test_bump():
    bumps = bump(20, 20)
    assert bumps is not None


def test_bump_agg_numpy():
    agg = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    np.random.seed(42)
    result = bump(agg=agg)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (20, 30)
    assert isinstance(result.data, np.ndarray)

    # determinism: same seed → same output
    np.random.seed(42)
    result2 = bump(agg=agg)
    np.testing.assert_array_equal(result.values, result2.values)


@cuda_and_cupy_available
def test_bump_agg_cupy():
    import cupy

    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_cp = agg_np.copy()
    agg_cp.data = cupy.asarray(agg_cp.data)

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_cp = bump(agg=agg_cp)

    assert isinstance(result_cp.data, cupy.ndarray)
    np.testing.assert_array_equal(result_np.values, result_cp.data.get())


@dask_array_available
def test_bump_agg_dask():
    """Single chunk (no internal boundaries) → bitwise match with numpy."""
    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dask = agg_np.copy()
    agg_dask.data = da.from_array(agg_dask.data, chunks=(20, 30))

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_dask = bump(agg=agg_dask)

    assert isinstance(result_dask.data, da.Array)
    np.testing.assert_array_equal(result_np.values, result_dask.values)


@dask_array_available
def test_bump_agg_dask_chunked():
    """Multiple chunks: verify laziness, shape, and approximate values."""
    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dask = agg_np.copy()
    agg_dask.data = da.from_array(agg_dask.data, chunks=(10, 15))

    np.random.seed(42)
    result_np = bump(agg=agg_np, spread=1)

    np.random.seed(42)
    result_dask = bump(agg=agg_dask, spread=1)

    assert isinstance(result_dask.data, da.Array)
    assert result_dask.shape == (20, 30)
    computed = result_dask.values
    # With spread=1, edge effects are minimal; total energy should be close
    np.testing.assert_allclose(computed.sum(), result_np.values.sum(), rtol=0.1)


@dask_array_available
@cuda_and_cupy_available
def test_bump_agg_dask_cupy():
    """Single chunk (no internal boundaries) → bitwise match with numpy."""
    import cupy

    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dc = agg_np.copy()
    agg_dc.data = da.from_array(
        cupy.asarray(agg_dc.data), chunks=(20, 30)
    )

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_dc = bump(agg=agg_dc)

    assert isinstance(result_dc.data, da.Array)
    computed = result_dc.data.compute()
    np.testing.assert_array_equal(result_np.values, computed.get())


def test_bump_preserves_coords():
    ys = np.linspace(0, 1, 10)
    xs = np.linspace(0, 2, 20)
    agg = xr.DataArray(
        np.zeros((10, 20)),
        dims=['lat', 'lon'],
        coords={'lat': ys, 'lon': xs},
    )
    result = bump(agg=agg)
    assert list(result.dims) == ['lat', 'lon']
    np.testing.assert_array_equal(result.coords['lat'].values, ys)
    np.testing.assert_array_equal(result.coords['lon'].values, xs)


def test_bump_agg_infers_shape():
    """When agg is given, width/height are inferred — no need to pass them."""
    agg = xr.DataArray(np.zeros((15, 25)), dims=['y', 'x'])
    np.random.seed(42)
    result = bump(agg=agg)
    assert result.shape == (15, 25)

    # Equivalent to explicit width/height
    np.random.seed(42)
    result2 = bump(width=25, height=15)
    np.testing.assert_array_equal(result.values, result2.values)
