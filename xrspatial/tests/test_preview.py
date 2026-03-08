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


# ---- exact output dimensions ----

def test_exact_height_trim():
    """Output should be trimmed to the exact requested height."""
    agg = _make_raster(200, 400)
    result = preview(agg, width=40, height=30)
    assert result.shape == (30, 40)


def test_exact_dimensions_odd_ratio():
    """Non-clean division should still produce exact target dims."""
    agg = _make_raster(199, 401)
    result = preview(agg, width=37, height=29)
    assert result.shape == (29, 37)


# ---- NaN handling ----

def test_all_nan_raster():
    """A fully NaN raster should return all-NaN output."""
    data = np.full((100, 200), np.nan, dtype=np.float32)
    agg = create_test_raster(data)
    result = preview(agg, width=20)
    assert np.all(np.isnan(result.values))


@dask_array_available
def test_nan_chunk_skip_dask():
    """All-NaN dask chunks should produce NaN output without error."""
    data = np.full((200, 400), np.nan, dtype=np.float32)
    data[100:, 200:] = 1.0  # only bottom-right quadrant is valid
    agg = create_test_raster(data, backend='dask', chunks=(100, 200))
    result = preview(agg, width=40).compute()
    # 3 of 4 chunks are all-NaN
    assert np.all(np.isnan(result.values[:10, :20]))   # top-left
    assert np.all(np.isnan(result.values[:10, 20:]))    # top-right
    assert np.all(np.isnan(result.values[10:, :20]))    # bottom-left
    np.testing.assert_allclose(result.values[10:, 20:], 1.0)  # bottom-right


@dask_array_available
def test_nan_chunk_skip_all_methods():
    """NaN-skip should work for all coarsen methods."""
    data = np.full((100, 200), np.nan, dtype=np.float32)
    data[50:, 100:] = 5.0
    agg = create_test_raster(data, backend='dask', chunks=(50, 100))
    for method in ('mean', 'median', 'max', 'min'):
        result = preview(agg, width=20, method=method).compute()
        assert np.all(np.isnan(result.values[:5, :10])), f'{method} failed'
        np.testing.assert_allclose(
            result.values[5:, 10:], 5.0, err_msg=f'{method} failed',
        )


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


# ---- median / max / min methods ----

def test_median_values():
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data)
    result = preview(agg, width=5, method='median')
    expected = np.median(
        data.reshape(5, 2, 5, 2).transpose(0, 2, 1, 3).reshape(5, 5, 4),
        axis=2,
    )
    np.testing.assert_allclose(result.values, expected)


def test_max_values():
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data)
    result = preview(agg, width=5, method='max')
    expected = data.reshape(5, 2, 5, 2).max(axis=(1, 3))
    np.testing.assert_allclose(result.values, expected)


def test_min_values():
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data)
    result = preview(agg, width=5, method='min')
    expected = data.reshape(5, 2, 5, 2).min(axis=(1, 3))
    np.testing.assert_allclose(result.values, expected)


@dask_array_available
def test_median_dask():
    agg = _make_raster(200, 400, backend='dask', chunks=(50, 100))
    result = preview(agg, width=40, method='median').compute()
    assert result.shape == (20, 40)


@dask_array_available
def test_max_dask():
    agg = _make_raster(200, 400, backend='dask', chunks=(50, 100))
    result = preview(agg, width=40, method='max').compute()
    assert result.shape == (20, 40)


@dask_array_available
def test_min_dask():
    agg = _make_raster(200, 400, backend='dask', chunks=(50, 100))
    result = preview(agg, width=40, method='min').compute()
    assert result.shape == (20, 40)


@cuda_and_cupy_available
def test_cupy_mean_values():
    """CuPy mean should produce true block averages, not stride subsampling."""
    import cupy

    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data, backend='cupy')
    result = preview(agg, width=5, method='mean')
    expected = data.reshape(5, 2, 5, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(result.data.get(), expected)


@cuda_and_cupy_available
def test_cupy_median():
    import cupy

    agg = _make_raster(200, 400, backend='cupy')
    result = preview(agg, width=40, method='median')
    assert isinstance(result.data, cupy.ndarray)
    assert result.shape == (20, 40)


@cuda_and_cupy_available
def test_cupy_max():
    import cupy

    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data, backend='cupy')
    result = preview(agg, width=5, method='max')
    expected = data.reshape(5, 2, 5, 2).max(axis=(1, 3))
    np.testing.assert_allclose(result.data.get(), expected)


@cuda_and_cupy_available
def test_cupy_min():
    import cupy

    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    agg = create_test_raster(data, backend='cupy')
    result = preview(agg, width=5, method='min')
    expected = data.reshape(5, 2, 5, 2).min(axis=(1, 3))
    np.testing.assert_allclose(result.data.get(), expected)


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


@dask_array_available
def test_bilinear_dask_is_lazy():
    """Bilinear on dask should produce a lazy result."""
    import dask.array as da

    agg = _make_raster(200, 400, backend='dask', chunks=(50, 100))
    result = preview(agg, width=40, method='bilinear')
    assert isinstance(result.data, da.Array)


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


@dask_array_available
def test_dask_chunk_alignment():
    """Chunk-aligned reduction should still produce correct results."""
    data = np.random.default_rng(42).random((200, 400)).astype(np.float32)
    np_agg = create_test_raster(data, backend='numpy')
    # Chunks not divisible by factor (factor=10, chunk=70)
    dk_agg = create_test_raster(data, backend='dask', chunks=(70, 70))

    np_result = preview(np_agg, width=40)
    dk_result = preview(dk_agg, width=40).compute()

    np.testing.assert_allclose(dk_result.values, np_result.values, rtol=1e-5)


@dask_array_available
def test_dask_snap_avoids_rechunk():
    """Factor should snap to a chunk divisor, avoiding rechunk layers."""
    import dask.array as da

    # chunks=70, width=30 -> initial factor=13, not a divisor of 70.
    # Nearest divisor of 70 to 13: 14.  Output dims shift slightly.
    agg = _make_raster(200, 400, backend='dask', chunks=(70, 70))
    result = preview(agg, width=30)
    assert isinstance(result.data, da.Array)
    graph = result.data.__dask_graph__()
    assert not any('rechunk' in k for k in graph.layers)
    computed = result.compute()
    assert computed.shape[0] > 0 and computed.shape[1] > 0


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


# ---- attrs propagation ----

def test_attrs_preserved():
    agg = _make_raster(200, 400)
    agg.attrs = {'crs': 'EPSG:4326', 'units': 'meters'}
    result = preview(agg, width=40)
    assert result.attrs == {'crs': 'EPSG:4326', 'units': 'meters'}


def test_attrs_preserved_nearest():
    agg = _make_raster(200, 400)
    agg.attrs = {'source': 'test'}
    result = preview(agg, width=40, method='nearest')
    assert result.attrs == {'source': 'test'}


def test_attrs_preserved_bilinear():
    agg = _make_raster(200, 400)
    agg.attrs = {'source': 'test'}
    result = preview(agg, width=40, method='bilinear')
    assert result.attrs == {'source': 'test'}


# ---- coordinate interpolation ----

def test_coords_non_uniform():
    """Non-uniform coordinates should be interpolated, not linspaced."""
    y = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81], dtype=np.float64)
    x = np.arange(20, dtype=np.float64)
    data = np.ones((10, 20), dtype=np.float32)
    agg = xr.DataArray(data, dims=['y', 'x'], coords={'y': y, 'x': x})
    result = preview(agg, width=10, method='nearest')
    # Nearest picks every 2nd coord; verify coords come from original
    np.testing.assert_array_equal(result.coords['y'].values, y[::2])


def test_coords_decreasing():
    """Decreasing coordinates (e.g. north-to-south lat) should work."""
    y = np.linspace(90, -90, 200)
    x = np.linspace(-180, 180, 400)
    data = np.ones((200, 400), dtype=np.float32)
    agg = xr.DataArray(data, dims=['y', 'x'], coords={'y': y, 'x': x})
    result = preview(agg, width=40)
    assert result.coords['y'].values[0] > result.coords['y'].values[-1]


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
