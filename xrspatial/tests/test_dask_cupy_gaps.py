"""Tests for dask+cupy backends: perlin, terrain, crosstab, trim, crop, apply."""

import numpy as np
import xarray as xr

from xrspatial import generate_terrain, perlin
from xrspatial.tests.general_checks import cuda_and_cupy_available, dask_array_available
from xrspatial.utils import has_cuda_and_cupy
from xrspatial.zonal import apply, crop, trim


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


# ---- trim: dask, cupy, dask+cupy ----

_TRIM_ARR = np.array([
    [0, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 4, 4, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
], dtype=np.int64)
_TRIM_EXPECTED_SHAPE = (3, 2)
_TRIM_EXPECTED = np.array([[4, 0], [4, 4], [1, 1]], dtype=np.int64)


@dask_array_available
def test_trim_dask():
    import dask.array as da

    raster = xr.DataArray(
        da.from_array(_TRIM_ARR, chunks=(3, 2)), dims=['y', 'x'],
    )
    result = trim(raster, values=(0,))
    assert result.shape == _TRIM_EXPECTED_SHAPE
    np.testing.assert_array_equal(result.data.compute(), _TRIM_EXPECTED)


@dask_array_available
def test_trim_dask_lazy():
    """trim() on a dask DataArray returns a dask-backed result (not computed)."""
    import dask.array as da

    raster = xr.DataArray(
        da.from_array(_TRIM_ARR, chunks=(3, 2)), dims=['y', 'x'],
    )
    result = trim(raster, values=(0,))
    assert isinstance(result.data, da.Array)


@cuda_and_cupy_available
def test_trim_cupy():
    import cupy

    raster = xr.DataArray(cupy.asarray(_TRIM_ARR), dims=['y', 'x'])
    result = trim(raster, values=(0,))
    assert result.shape == _TRIM_EXPECTED_SHAPE
    np.testing.assert_array_equal(result.data.get(), _TRIM_EXPECTED)


@cuda_and_cupy_available
@dask_array_available
def test_trim_dask_cupy():
    import cupy
    import dask.array as da

    gpu = cupy.asarray(_TRIM_ARR)
    raster = xr.DataArray(da.from_array(gpu, chunks=(3, 2)), dims=['y', 'x'])
    result = trim(raster, values=(0,))
    assert result.shape == _TRIM_EXPECTED_SHAPE
    computed = result.data.compute()
    assert isinstance(computed, cupy.ndarray)
    np.testing.assert_array_equal(computed.get(), _TRIM_EXPECTED)


# ---- crop: dask, cupy, dask+cupy ----

_CROP_ARR = np.array([
    [0, 4, 0, 3],
    [0, 4, 4, 3],
    [0, 1, 1, 3],
    [0, 1, 1, 3],
    [0, 0, 0, 0],
], dtype=np.int64)
_CROP_EXPECTED_SHAPE = (4, 3)
_CROP_EXPECTED = np.array([
    [4, 0, 3],
    [4, 4, 3],
    [1, 1, 3],
    [1, 1, 3],
], dtype=np.int64)


@dask_array_available
def test_crop_dask():
    import dask.array as da

    raster = xr.DataArray(
        da.from_array(_CROP_ARR, chunks=(3, 2)), dims=['y', 'x'],
    )
    result = crop(raster, raster, zones_ids=(1, 3))
    assert result.shape == _CROP_EXPECTED_SHAPE
    np.testing.assert_array_equal(result.data.compute(), _CROP_EXPECTED)


@dask_array_available
def test_crop_dask_lazy():
    """crop() on a dask DataArray returns a dask-backed result (not computed)."""
    import dask.array as da

    raster = xr.DataArray(
        da.from_array(_CROP_ARR, chunks=(3, 2)), dims=['y', 'x'],
    )
    result = crop(raster, raster, zones_ids=(1, 3))
    assert isinstance(result.data, da.Array)


@cuda_and_cupy_available
def test_crop_cupy():
    import cupy

    raster = xr.DataArray(cupy.asarray(_CROP_ARR), dims=['y', 'x'])
    result = crop(raster, raster, zones_ids=(1, 3))
    assert result.shape == _CROP_EXPECTED_SHAPE
    np.testing.assert_array_equal(result.data.get(), _CROP_EXPECTED)


@cuda_and_cupy_available
@dask_array_available
def test_crop_dask_cupy():
    import cupy
    import dask.array as da

    gpu = cupy.asarray(_CROP_ARR)
    raster = xr.DataArray(da.from_array(gpu, chunks=(3, 2)), dims=['y', 'x'])
    result = crop(raster, raster, zones_ids=(1, 3))
    assert result.shape == _CROP_EXPECTED_SHAPE
    computed = result.data.compute()
    assert isinstance(computed, cupy.ndarray)
    np.testing.assert_array_equal(computed.get(), _CROP_EXPECTED)


# ---- apply: cupy, dask+cupy, fallback ----

_APPLY_ZONES = np.array([
    [1, 1, 0, 2],
    [1, 1, 0, 2],
    [3, 3, 3, 2],
], dtype=np.int64)

_APPLY_VALUES = np.array([
    [10.0, 20.0, 30.0, 40.0],
    [50.0, 60.0, 70.0, 80.0],
    [90.0, 100.0, 110.0, 120.0],
], dtype=np.float64)


def _double(x):
    return x * 2


@cuda_and_cupy_available
def test_apply_cupy():
    import cupy

    zones_np = xr.DataArray(_APPLY_ZONES, dims=['y', 'x'])
    values_np = xr.DataArray(_APPLY_VALUES, dims=['y', 'x'])
    result_np = apply(zones_np, values_np, _double)

    zones_cupy = xr.DataArray(cupy.asarray(_APPLY_ZONES), dims=['y', 'x'])
    values_cupy = xr.DataArray(cupy.asarray(_APPLY_VALUES), dims=['y', 'x'])
    result_cupy = apply(zones_cupy, values_cupy, _double)

    assert isinstance(result_cupy.data, cupy.ndarray)
    np.testing.assert_allclose(result_cupy.data.get(), result_np.values)


@cuda_and_cupy_available
@dask_array_available
def test_apply_dask_cupy():
    import cupy
    import dask.array as da

    zones_np = xr.DataArray(_APPLY_ZONES, dims=['y', 'x'])
    values_np = xr.DataArray(_APPLY_VALUES, dims=['y', 'x'])
    result_np = apply(zones_np, values_np, _double)

    zones_gpu = cupy.asarray(_APPLY_ZONES)
    values_gpu = cupy.asarray(_APPLY_VALUES)
    zones_dask = xr.DataArray(
        da.from_array(zones_gpu, chunks=(2, 2)), dims=['y', 'x'],
    )
    values_dask = xr.DataArray(
        da.from_array(values_gpu, chunks=(2, 2)), dims=['y', 'x'],
    )
    result = apply(zones_dask, values_dask, _double)

    assert isinstance(result.data, da.Array)
    computed = result.data.compute()
    assert isinstance(computed, cupy.ndarray)
    np.testing.assert_allclose(computed.get(), result_np.values)


@cuda_and_cupy_available
def test_apply_cupy_fallback():
    """A func that CUDA can't compile still works via CPU fallback."""
    import cupy

    lookup = {10.0: 100.0, 50.0: 500.0}

    def _dict_func(x):
        return lookup.get(x, x)

    zones_np = xr.DataArray(_APPLY_ZONES, dims=['y', 'x'])
    values_np = xr.DataArray(_APPLY_VALUES, dims=['y', 'x'])
    result_np = apply(zones_np, values_np, _dict_func)

    zones_cupy = xr.DataArray(cupy.asarray(_APPLY_ZONES), dims=['y', 'x'])
    values_cupy = xr.DataArray(cupy.asarray(_APPLY_VALUES), dims=['y', 'x'])
    result_cupy = apply(zones_cupy, values_cupy, _dict_func)

    assert isinstance(result_cupy.data, cupy.ndarray)
    np.testing.assert_allclose(result_cupy.data.get(), result_np.values)


# ---- hotspots dask+cupy ----

@cuda_and_cupy_available
@dask_array_available
def test_hotspots_dask_cupy():
    import cupy
    import dask.array as da
    from xrspatial.convolution import custom_kernel
    from xrspatial.focal import hotspots

    rng = np.random.default_rng(42)
    np_data = rng.standard_normal((20, 20)).astype('f4')
    # Add hot/cold clusters
    np_data[2:5, 2:5] += 5.0
    np_data[15:18, 15:18] -= 5.0

    kernel = custom_kernel(np.ones((3, 3)))

    # numpy reference
    raster_np = xr.DataArray(np_data, dims=['y', 'x'])
    result_np = hotspots(raster_np, kernel)

    # dask+cupy
    gpu = cupy.asarray(np_data)
    raster_dask_cupy = xr.DataArray(
        da.from_array(gpu, chunks=(10, 10)), dims=['y', 'x'],
    )
    result_dc = hotspots(raster_dask_cupy, kernel)

    assert isinstance(result_dc.data, da.Array)
    computed = result_dc.data.compute()
    assert isinstance(computed, cupy.ndarray)
    np.testing.assert_array_equal(computed.get(), result_np.values)


# ---- emerging_hotspots dask+cupy ----

@cuda_and_cupy_available
@dask_array_available
def test_emerging_hotspots_dask_cupy():
    import cupy
    import dask.array as da
    from xrspatial.convolution import custom_kernel
    from xrspatial.emerging_hotspots import emerging_hotspots

    rng = np.random.default_rng(42)
    np_data = rng.standard_normal((5, 15, 15)).astype('f4')
    # Add an intensifying hot cluster
    for t in range(5):
        np_data[t, 3:6, 3:6] += 2.0 + t * 0.5

    kernel = custom_kernel(np.ones((3, 3)))

    # numpy reference
    raster_np = xr.DataArray(np_data, dims=['time', 'y', 'x'])
    ds_np = emerging_hotspots(raster_np, kernel)

    # dask+cupy
    gpu = cupy.asarray(np_data)
    raster_dc = xr.DataArray(
        da.from_array(gpu, chunks=(5, 8, 8)), dims=['time', 'y', 'x'],
    )
    ds_dc = emerging_hotspots(raster_dc, kernel).compute()

    for var in ('category', 'gi_zscore', 'gi_bin', 'trend_zscore', 'trend_pvalue'):
        dc_vals = ds_dc[var].data
        if isinstance(dc_vals, cupy.ndarray):
            dc_vals = dc_vals.get()
        np.testing.assert_allclose(
            dc_vals, ds_np[var].values, atol=1e-5,
            err_msg=f"mismatch in {var}",
        )
