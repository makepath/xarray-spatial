import numpy as np
import pytest
import xarray as xr

from xrspatial import perlin
from xrspatial.tests.general_checks import cuda_and_cupy_available
from xrspatial.tests.general_checks import dask_array_available
from xrspatial.tests.general_checks import general_output_checks

from xrspatial.utils import has_cuda_and_cupy


def create_test_arr(backend='numpy'):
    W = 50
    H = 50
    data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])

    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend:
        import dask.array as da
        raster.data = da.from_array(raster.data, chunks=(10, 10))

    return raster


def test_perlin_cpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)
    general_output_checks(data_numpy, perlin_numpy)


@dask_array_available
def test_perlin_dask_cpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)
    general_output_checks(data_numpy, perlin_numpy)

    # dask
    data_dask = create_test_arr(backend='dask')
    perlin_dask = perlin(data_dask)
    general_output_checks(data_dask, perlin_dask)

    np.testing.assert_allclose(
        perlin_numpy.data, perlin_dask.data.compute(),
        rtol=1e-05, atol=1e-07, equal_nan=True
    )


@dask_array_available
def test_perlin_dask_does_not_persist_whole_array(monkeypatch):
    # Regression for issue #3469: the dask backends used dask.persist() to
    # cache the noise before reducing it, which forces every chunk resident
    # at once and OOMs at scale. The min/ptp reductions share the noise
    # subgraph within a single dask.compute call, so the persist is
    # unnecessary. Fail loudly if it ever comes back.
    import dask

    def _no_persist(*args, **kwargs):
        raise AssertionError(
            "perlin dask backend called dask.persist(); this materializes "
            "the whole noise array and reintroduces the OOM from #3469"
        )

    monkeypatch.setattr(dask, "persist", _no_persist)

    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)

    data_dask = create_test_arr(backend='dask')
    perlin_dask = perlin(data_dask)
    general_output_checks(data_dask, perlin_dask)

    np.testing.assert_allclose(
        perlin_numpy.data, perlin_dask.data.compute(),
        rtol=1e-05, atol=1e-07, equal_nan=True
    )


@cuda_and_cupy_available
def test_perlin_gpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)

    # cupy
    data_cupy = create_test_arr(backend='cupy')
    perlin_cupy = perlin(data_cupy)
    general_output_checks(data_cupy, perlin_cupy)
    np.testing.assert_allclose(
        perlin_numpy.data, perlin_cupy.data.get(),
        rtol=1e-4, atol=1e-6, equal_nan=True
    )


@cuda_and_cupy_available
@dask_array_available
def test_perlin_dask_gpu(monkeypatch):
    # The dask+cupy path must not call dask.persist either (issue #3469);
    # guard it here so the GPU backend is covered too.
    import dask

    def _no_persist(*args, **kwargs):
        raise AssertionError(
            "perlin dask+cupy backend called dask.persist(); this "
            "materializes the whole noise array and reintroduces the "
            "OOM from #3469"
        )

    monkeypatch.setattr(dask, "persist", _no_persist)

    # numpy baseline
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)

    # cupy baseline
    data_cupy = create_test_arr(backend='cupy')
    perlin_cupy = perlin(data_cupy)

    # dask + cupy
    data_dask_cupy = create_test_arr(backend='dask+cupy')
    perlin_dask_cupy = perlin(data_dask_cupy)
    general_output_checks(data_dask_cupy, perlin_dask_cupy)

    np.testing.assert_allclose(
        perlin_numpy.data, perlin_dask_cupy.data.compute().get(),
        rtol=1e-4, atol=1e-4, equal_nan=True
    )
    np.testing.assert_allclose(
        perlin_cupy.data.get(), perlin_dask_cupy.data.compute().get(),
        rtol=1e-4, atol=1e-4, equal_nan=True
    )


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32],
)
def test_perlin_rejects_integer_dtype(dtype):
    # Regression for issue #1232: integer-dtyped input silently produced
    # INT_MIN everywhere because the float noise was written in place and
    # then normalized by a zero ptp.  We now raise ValueError instead.
    data = np.zeros((20, 20), dtype=dtype)
    raster = xr.DataArray(data, dims=['y', 'x'])
    with pytest.raises(ValueError, match="floating-point dtype"):
        perlin(raster)


def test_perlin_preserves_coords():
    # Regression: perlin() preserved attrs but dropped the input's x/y
    # coords, yielding a DataArray that kept its `crs` attr with no
    # coordinates.  The output must carry the input coords through.
    H, W = 10, 12
    data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.linspace(10.0, 20.0, H),
                'x': np.linspace(100.0, 200.0, W)},
        attrs={'crs': 'EPSG:4326', 'res': (1.0, 1.0)},
    )
    result = perlin(raster)
    general_output_checks(raster, result)
    assert list(result.coords) == ['y', 'x']
    np.testing.assert_array_equal(result['y'].data, raster['y'].data)
    np.testing.assert_array_equal(result['x'].data, raster['x'].data)
    assert result.attrs == raster.attrs


@dask_array_available
def test_perlin_preserves_coords_dask():
    # The result wrapping is a single shared path, but the coord-bearing
    # assertion above only exercises numpy.  Lock in the dask backend too.
    import dask.array as da
    H, W = 10, 12
    data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(
        da.from_array(data, chunks=(5, 6)),
        dims=['y', 'x'],
        coords={'y': np.linspace(10.0, 20.0, H),
                'x': np.linspace(100.0, 200.0, W)},
        attrs={'crs': 'EPSG:4326', 'res': (1.0, 1.0)},
    )
    result = perlin(raster)
    general_output_checks(raster, result)
    assert list(result.coords) == ['y', 'x']
    np.testing.assert_array_equal(result['y'].data, raster['y'].data)
    np.testing.assert_array_equal(result['x'].data, raster['x'].data)
    assert result.attrs == raster.attrs


def test_perlin_float64_input():
    # float64 should still work (not just float32).
    data = np.zeros((20, 20), dtype=np.float64)
    raster = xr.DataArray(data, dims=['y', 'x'])
    result = perlin(raster)
    assert result.dtype == np.float64
    assert np.isfinite(result.data).all()
    # Normalized to [0, 1]
    assert result.data.min() >= 0.0
    assert result.data.max() <= 1.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_perlin_dask_cpu_preserves_dtype(dtype):
    # Regression: the dask backends used to hardcode float32, silently
    # downcasting float64 input while numpy/cupy preserved it.
    import dask.array as da
    data = da.from_array(np.zeros((20, 20), dtype=dtype), chunks=(10, 10))
    raster = xr.DataArray(data, dims=['y', 'x'])
    result = perlin(raster)
    assert result.dtype == dtype
    computed = result.data.compute()
    assert np.isfinite(computed).all()
    assert computed.min() >= 0.0
    assert computed.max() <= 1.0


@cuda_and_cupy_available
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_perlin_gpu_preserves_dtype(dtype):
    import cupy
    data = cupy.zeros((20, 20), dtype=dtype)
    raster = xr.DataArray(data, dims=['y', 'x'])
    result = perlin(raster)
    assert result.dtype == dtype


@cuda_and_cupy_available
@dask_array_available
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_perlin_dask_gpu_preserves_dtype(dtype):
    import cupy
    import dask.array as da
    data = da.from_array(cupy.zeros((20, 20), dtype=dtype), chunks=(10, 10))
    raster = xr.DataArray(data, dims=['y', 'x'])
    result = perlin(raster)
    assert result.dtype == dtype


# ---------------------------------------------------------------------------
# Parameter coverage (freq / seed / name)
# ---------------------------------------------------------------------------

def _zeros(h=30, w=30):
    return xr.DataArray(np.zeros((h, w), dtype=np.float32), dims=['y', 'x'])


def test_perlin_seed_changes_output():
    # Different seeds must produce different noise.
    a = perlin(_zeros(), seed=5)
    b = perlin(_zeros(), seed=9)
    assert not np.allclose(a.data, b.data)


def test_perlin_seed_is_deterministic():
    # Same seed must reproduce the same noise.
    a = perlin(_zeros(), seed=7)
    b = perlin(_zeros(), seed=7)
    np.testing.assert_array_equal(a.data, b.data)


def test_perlin_freq_changes_output():
    # A higher frequency must change the noise field.
    low = perlin(_zeros(), freq=(1, 1))
    high = perlin(_zeros(), freq=(4, 4))
    assert not np.allclose(low.data, high.data)


def test_perlin_name_param():
    # The name kwarg sets the output DataArray name.
    result = perlin(_zeros(), name='myperlin')
    assert result.name == 'myperlin'


# ---------------------------------------------------------------------------
# Geometric edge cases
# ---------------------------------------------------------------------------

def test_perlin_single_row():
    # 1xN strip works: the x axis still varies so ptp != 0.
    result = perlin(xr.DataArray(np.zeros((1, 10), dtype=np.float32),
                                 dims=['y', 'x']))
    assert result.shape == (1, 10)
    assert np.isfinite(result.data).all()
    assert result.data.min() >= 0.0
    assert result.data.max() <= 1.0


@pytest.mark.parametrize("shape", [(1, 1), (10, 1)])
def test_perlin_degenerate_single_column_all_nan(shape):
    # Pins current behavior: a 1x1 or Nx1 single-column raster has constant
    # noise down the column, so the ptp normalization divides by zero and the
    # whole output becomes NaN.  This is a known bug -- see the perlin
    # "all-NaN for 1x1 and Nx1 rasters" issue.  Update this test (to expect a
    # finite value or a raised error) once that is resolved.
    data = np.zeros(shape, dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])
    with np.errstate(invalid='ignore'):
        result = perlin(raster)
    assert result.shape == shape
    assert np.isnan(result.data).all()


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

def test_perlin_preserves_dims_and_attrs():
    src = xr.DataArray(np.zeros((10, 12), dtype=np.float32), dims=['lat', 'lon'],
                       attrs={'res': (0.5, 0.5), 'crs': 'EPSG:4326'})
    out = perlin(src)
    assert out.dims == ('lat', 'lon')
    assert out.attrs == src.attrs


def test_perlin_preserves_input_coords():
    # perlin() now carries the input coordinate variables through to the output
    # (see #3470).  The output keeps the same lat/lon coords as the input.
    src = xr.DataArray(np.zeros((10, 12), dtype=np.float32), dims=['lat', 'lon'])
    src['lat'] = np.arange(10)
    src['lon'] = np.arange(12)
    out = perlin(src)
    assert 'lat' in out.coords
    assert 'lon' in out.coords
    np.testing.assert_array_equal(out['lat'].values, src['lat'].values)
    np.testing.assert_array_equal(out['lon'].values, src['lon'].values)


def test_perlin_docstring_documents_name():
    # Regression for issue #3465: the docstring must document the `name`
    # parameter that exists in the signature.
    assert 'name : str' in perlin.__doc__
