import numpy as np
import pytest
import xarray as xr

from xrspatial.preview import preview
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ---- helpers ----

def _make_raster(rows, cols, backend='numpy', chunks=(50, 50)):
    rng = np.random.default_rng(986)
    data = rng.random((rows, cols)).astype(np.float32)
    return create_test_raster(data, backend=backend, chunks=chunks)


# ---- basic correctness ----

def test_numpy_basic():
    agg = _make_raster(200, 400)
    result = preview(agg, width=40)
    assert result.shape == (20, 40)
    assert result.dims == ('y', 'x')
    assert result.name == 'preview'
    assert len(result.coords['y']) == 20
    assert len(result.coords['x']) == 40


def test_explicit_height():
    agg = _make_raster(200, 400)
    result = preview(agg, width=50, height=25)
    assert result.shape == (25, 50)


def test_custom_name():
    agg = _make_raster(200, 400)
    result = preview(agg, width=40, name='thumbnail')
    assert result.name == 'thumbnail'


def test_small_raster_passthrough():
    """Rasters already smaller than target should pass through unchanged."""
    agg = _make_raster(5, 5)
    result = preview(agg, width=100)
    assert result is agg


def test_one_axis_small():
    """If only one axis needs reduction, only that axis shrinks."""
    agg = _make_raster(10, 500)
    result = preview(agg, width=50)
    # factor_x = 500//50 = 10, factor_y = 10//5 = 2
    assert result.shape[1] == 50
    assert result.shape[0] <= 10


# ---- NaN handling ----

def test_nan_blocks():
    """All-NaN blocks produce NaN; partial-NaN blocks produce valid means."""
    data = np.ones((100, 100), dtype=np.float32)
    # Top-left quadrant is NaN
    data[:50, :50] = np.nan
    agg = create_test_raster(data)
    result = preview(agg, width=10)

    vals = result.values
    # Top-left 5x5 should be NaN (all-NaN input blocks)
    assert np.all(np.isnan(vals[:5, :5]))
    # Bottom-right 5x5 should be 1.0 (all-ones input blocks)
    np.testing.assert_allclose(vals[5:, 5:], 1.0)


# ---- block averaging correctness ----

def test_coarsen_mean_values():
    """Verify that output values match manual block averaging."""
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data)
    result = preview(agg, width=5)

    # factor = 10 // 5 = 2, so each output pixel is the mean of a 2x2 block
    expected = data.reshape(5, 2, 5, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(result.values, expected)


# ---- nearest method ----

def test_nearest_basic():
    agg = _make_raster(200, 400)
    result = preview(agg, width=40, method='nearest')
    assert result.shape == (20, 40)
    assert result.dims == ('y', 'x')


def test_nearest_picks_strided_values():
    """Nearest should pick every Nth pixel, not average."""
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data)
    result = preview(agg, width=5, method='nearest')
    # factor = 2, so result[i, j] == data[2*i, 2*j]
    expected = data[::2, ::2]
    np.testing.assert_array_equal(result.values, expected)


@cuda_and_cupy_available
def test_nearest_cupy():
    agg = _make_raster(200, 400, backend='cupy')
    result = preview(agg, width=40, method='nearest')
    import cupy
    assert isinstance(result.data, cupy.ndarray)
    assert result.shape == (20, 40)


# ---- bilinear method ----

def test_bilinear_basic():
    agg = _make_raster(200, 400)
    result = preview(agg, width=40, method='bilinear')
    assert result.shape == (20, 40)
    assert result.dims == ('y', 'x')


def test_bilinear_smooth():
    """Bilinear output should differ from nearest (interpolation, not stride)."""
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data)
    nearest_result = preview(agg, width=5, method='nearest')
    bilinear_result = preview(agg, width=5, method='bilinear')
    # They should not be identical since bilinear interpolates
    assert not np.array_equal(nearest_result.values, bilinear_result.values)


@dask_array_available
def test_bilinear_dask():
    agg = _make_raster(200, 400, backend='dask', chunks=(50, 100))
    result = preview(agg, width=40, method='bilinear')
    computed = result.compute()
    assert computed.shape == (20, 40)


@cuda_and_cupy_available
def test_bilinear_cupy():
    agg = _make_raster(200, 400, backend='cupy')
    result = preview(agg, width=40, method='bilinear')
    import cupy
    assert isinstance(result.data, cupy.ndarray)
    assert result.shape == (20, 40)


# ---- dask backend ----

@dask_array_available
def test_dask_numpy_lazy():
    """Dask result should be lazy (not computed) until explicitly triggered."""
    import dask.array as da
    agg = _make_raster(200, 400, backend='dask', chunks=(50, 100))
    result = preview(agg, width=40)
    assert isinstance(result.data, da.Array)
    computed = result.compute()
    assert computed.shape == result.shape


@dask_array_available
def test_dask_matches_numpy():
    """Dask and numpy backends should produce the same result."""
    data = np.random.default_rng(986).random((200, 400)).astype(np.float32)
    np_agg = create_test_raster(data, backend='numpy')
    dk_agg = create_test_raster(data, backend='dask', chunks=(50, 100))

    np_result = preview(np_agg, width=40)
    dk_result = preview(dk_agg, width=40).compute()

    np.testing.assert_allclose(dk_result.values, np_result.values, rtol=1e-5)


# ---- cupy backend ----

@cuda_and_cupy_available
def test_cupy_basic():
    agg = _make_raster(200, 400, backend='cupy')
    result = preview(agg, width=40)
    import cupy
    assert isinstance(result.data, cupy.ndarray)
    assert result.shape[1] <= 40


@cuda_and_cupy_available
def test_dask_cupy():
    agg = _make_raster(200, 400, backend='dask+cupy', chunks=(50, 100))
    result = preview(agg, width=40)
    computed = result.compute()
    assert computed.shape[1] <= 40


# ---- Dataset support ----

def test_dataset():
    agg = _make_raster(200, 400)
    ds = xr.Dataset({'elev': agg, 'slope': agg * 2})
    result = preview(ds, width=40)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'elev', 'slope'}
    for var in result.data_vars:
        assert result[var].shape == (20, 40)


# ---- input validation ----

def test_rejects_invalid_method():
    agg = _make_raster(200, 400)
    with pytest.raises(ValueError, match='method must be one of'):
        preview(agg, width=40, method='cubic')


def test_rejects_non_dataarray():
    with pytest.raises(TypeError, match='must be an xarray.DataArray'):
        preview(np.ones((10, 10)))


def test_rejects_1d():
    agg = xr.DataArray(np.ones(10), dims=['x'])
    with pytest.raises(ValueError, match='2D'):
        preview(agg)


def test_rejects_non_numeric():
    agg = xr.DataArray(np.array([['a', 'b'], ['c', 'd']]), dims=['y', 'x'])
    with pytest.raises(ValueError, match='numeric'):
        preview(agg)


# ---- accessor ----

def test_accessor_dataarray():
    agg = _make_raster(200, 400)
    result = agg.xrs.preview(width=40)
    assert result.shape == (20, 40)


def test_accessor_dataset():
    agg = _make_raster(200, 400)
    ds = xr.Dataset({'elev': agg})
    result = ds.xrs.preview(width=40)
    assert isinstance(result, xr.Dataset)
    assert result['elev'].shape == (20, 40)
