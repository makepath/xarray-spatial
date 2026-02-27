"""Tests for dask+cupy backends: perlin, terrain, crosstab."""

import numpy as np
import xarray as xr

from xrspatial import generate_terrain, perlin
from xrspatial.tests.general_checks import cuda_and_cupy_available, dask_array_available
from xrspatial.utils import has_cuda_and_cupy


def _make_raster(shape=(50, 50), backend='numpy', chunks=(10, 10)):
    data = np.zeros(shape, dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)
    if 'dask' in backend:
        import dask.array as da
        raster.data = da.from_array(raster.data, chunks=chunks)
    return raster


# ---- perlin dask+cupy ----

@cuda_and_cupy_available
@dask_array_available
def test_perlin_dask_cupy():
    import cupy
    import dask.array as da

    cupy_raster = _make_raster(backend='cupy')
    result_cupy = perlin(cupy_raster)

    dask_cupy_raster = _make_raster(backend='dask+cupy')
    result_dask_cupy = perlin(dask_cupy_raster)

    assert isinstance(result_dask_cupy.data, da.Array)
    assert result_dask_cupy.shape == cupy_raster.shape

    computed = result_dask_cupy.data.compute()
    assert isinstance(computed, cupy.ndarray)

    vals = computed.get()
    assert vals.min() >= 0.0
    assert vals.max() <= 1.0

    np.testing.assert_allclose(
        result_cupy.data.get(), vals, rtol=1e-4, atol=1e-4,
    )


# ---- terrain dask+cupy ----

@cuda_and_cupy_available
@dask_array_available
def test_terrain_dask_cupy():
    import cupy
    import dask.array as da

    cupy_raster = _make_raster(backend='cupy')
    terrain_cupy = generate_terrain(cupy_raster)

    dask_cupy_raster = _make_raster(backend='dask+cupy')
    terrain_dask_cupy = generate_terrain(dask_cupy_raster)

    assert isinstance(terrain_dask_cupy.data, da.Array)
    assert terrain_dask_cupy.shape == cupy_raster.shape

    computed = terrain_dask_cupy.data.compute()
    assert isinstance(computed, cupy.ndarray)

    vals = computed.get()
    assert vals.min() >= 0.0

    np.testing.assert_allclose(
        terrain_cupy.data.get(), vals, rtol=1e-4, atol=1e-4,
    )


# ---- crosstab cupy ----

@cuda_and_cupy_available
def test_crosstab_cupy():
    import cupy
    from xrspatial.zonal import crosstab

    zones_np = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.float64)
    values_np = np.array([
        [10, 10, 20, 20],
        [10, 20, 20, 30],
        [30, 30, 10, 10],
        [30, 10, 20, 20],
    ], dtype=np.float64)

    zones_xr = xr.DataArray(zones_np, dims=['y', 'x'])
    values_xr = xr.DataArray(values_np, dims=['y', 'x'])
    df_numpy = crosstab(zones_xr, values_xr)

    zones_cupy = xr.DataArray(cupy.asarray(zones_np), dims=['y', 'x'])
    values_cupy = xr.DataArray(cupy.asarray(values_np), dims=['y', 'x'])
    df_cupy = crosstab(zones_cupy, values_cupy)

    # Both should be pandas DataFrames with identical content
    assert list(df_numpy.columns) == list(df_cupy.columns)
    np.testing.assert_array_equal(df_numpy.values, df_cupy.values)


# ---- crosstab dask+cupy ----

@cuda_and_cupy_available
@dask_array_available
def test_crosstab_dask_cupy():
    import cupy
    import dask.array as da
    from xrspatial.zonal import crosstab

    zones_np = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.float64)
    values_np = np.array([
        [10, 10, 20, 20],
        [10, 20, 20, 30],
        [30, 30, 10, 10],
        [30, 10, 20, 20],
    ], dtype=np.float64)

    zones_xr = xr.DataArray(zones_np, dims=['y', 'x'])
    values_xr = xr.DataArray(values_np, dims=['y', 'x'])
    df_numpy = crosstab(zones_xr, values_xr)

    zones_gpu = cupy.asarray(zones_np)
    values_gpu = cupy.asarray(values_np)
    zones_dask = xr.DataArray(
        da.from_array(zones_gpu, chunks=(2, 2)), dims=['y', 'x']
    )
    values_dask = xr.DataArray(
        da.from_array(values_gpu, chunks=(2, 2)), dims=['y', 'x']
    )
    df_dask_cupy = crosstab(zones_dask, values_dask)

    # dask case returns dask DataFrame
    df_computed = df_dask_cupy.compute()
    # sort both by zone for stable comparison
    df_numpy_sorted = df_numpy.sort_values('zone').reset_index(drop=True)
    df_computed_sorted = df_computed.sort_values('zone').reset_index(drop=True)

    assert list(df_numpy_sorted.columns) == list(df_computed_sorted.columns)
    np.testing.assert_array_equal(
        df_numpy_sorted.values, df_computed_sorted.values,
    )
