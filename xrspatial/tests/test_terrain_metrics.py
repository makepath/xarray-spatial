import numpy as np
import pytest
import xarray as xr

from xrspatial import tri, tpi, roughness, landforms
from xrspatial.terrain_metrics import LANDFORM_CLASSES
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy,
                                            create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available,
                                            general_output_checks)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_surface():
    """Constant elevation — all three metrics should be 0 in the interior."""
    data = np.full((6, 8), 42.0, dtype=np.float64)
    return data


@pytest.fixture
def known_surface():
    """Hand-crafted 5×5 grid with pre-computed expected outputs.

    Grid:
        1  2  3  4  5
        6  7  8  9 10
       11 12 13 14 15
       16 17 18 19 20
       21 22 23 24 25
    """
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    return data


# ---------------------------------------------------------------------------
# Hand-computed expected values for the known_surface
# ---------------------------------------------------------------------------

def _expected_tri_known():
    """Compute expected TRI for interior cells of the 5×5 grid."""
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    out = np.full((5, 5), np.nan)
    for y in range(1, 4):
        for x in range(1, 4):
            center = data[y, x]
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    diff = data[y + dy, x + dx] - center
                    total += diff * diff
            out[y, x] = np.sqrt(total)
    return out


def _expected_tpi_known():
    """Compute expected TPI for interior cells of the 5×5 grid."""
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    out = np.full((5, 5), np.nan)
    for y in range(1, 4):
        for x in range(1, 4):
            center = data[y, x]
            total = 0.0
            count = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    total += data[y + dy, x + dx]
                    count += 1
            out[y, x] = center - total / count
    return out


def _expected_roughness_known():
    """Compute expected roughness for interior cells of the 5×5 grid."""
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    out = np.full((5, 5), np.nan)
    for y in range(1, 4):
        for x in range(1, 4):
            vmin = np.inf
            vmax = -np.inf
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    v = data[y + dy, x + dx]
                    if v < vmin:
                        vmin = v
                    if v > vmax:
                        vmax = v
            out[y, x] = vmax - vmin
    return out


# ---------------------------------------------------------------------------
# Flat surface tests
# ---------------------------------------------------------------------------

def test_tri_flat(flat_surface):
    agg = create_test_raster(flat_surface)
    result = tri(agg)
    interior = result.data[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 0.0, atol=1e-10)


def test_tpi_flat(flat_surface):
    agg = create_test_raster(flat_surface)
    result = tpi(agg)
    interior = result.data[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 0.0, atol=1e-10)


def test_roughness_flat(flat_surface):
    agg = create_test_raster(flat_surface)
    result = roughness(agg)
    interior = result.data[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Known-values tests
# ---------------------------------------------------------------------------

def test_tri_known(known_surface):
    agg = create_test_raster(known_surface)
    result = tri(agg)
    expected = _expected_tri_known()
    general_output_checks(agg, result, expected)


def test_tpi_known(known_surface):
    agg = create_test_raster(known_surface)
    result = tpi(agg)
    expected = _expected_tpi_known()
    general_output_checks(agg, result, expected)


def test_roughness_known(known_surface):
    agg = create_test_raster(known_surface)
    result = roughness(agg)
    expected = _expected_roughness_known()
    general_output_checks(agg, result, expected)


# ---------------------------------------------------------------------------
# NaN handling tests
# ---------------------------------------------------------------------------

def test_nan_propagation():
    """NaN in any neighbor should propagate to the output (GDAL behavior)."""
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, np.nan, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ])
    agg = create_test_raster(data)

    # Cell (1,2)=7.0 has NaN neighbor at (1,1) -> all three metrics NaN
    tri_result = tri(agg)
    assert np.isnan(tri_result.data[1, 2])

    tpi_result = tpi(agg)
    assert np.isnan(tpi_result.data[1, 2])

    roughness_result = roughness(agg)
    assert np.isnan(roughness_result.data[1, 2])

    # Cell (2,2)=11.0 also has NaN neighbor at (1,1) -> NaN
    assert np.isnan(tri_result.data[2, 2])

    # Cell (2,1)=10.0 has NaN neighbor at (1,1) -> NaN
    assert np.isnan(tpi_result.data[2, 1])

    # Cell with no NaN neighbors: (2,2) has NaN neighbor so skip;
    # Check (1,1) which is NaN center -> NaN output
    assert np.isnan(tri_result.data[1, 1])
    assert np.isnan(tpi_result.data[1, 1])


def test_no_nan_in_window():
    """Interior cells with no NaN neighbors should produce valid results."""
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    agg = create_test_raster(data)

    # Interior cell (2,2) = 13, all neighbors are valid
    tri_result = tri(agg)
    assert np.isfinite(tri_result.data[2, 2])

    tpi_result = tpi(agg)
    assert np.isfinite(tpi_result.data[2, 2])


# ---------------------------------------------------------------------------
# Edge NaN tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", [tri, tpi, roughness])
def test_nan_edges(func):
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = func(agg)
    # Edges should be NaN with default boundary='nan'
    np.testing.assert_array_equal(result.data[0, :], np.nan)
    np.testing.assert_array_equal(result.data[-1, :], np.nan)
    np.testing.assert_array_equal(result.data[:, 0], np.nan)
    np.testing.assert_array_equal(result.data[:, -1], np.nan)


# ---------------------------------------------------------------------------
# Cross-backend tests
# ---------------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("size", [(4, 6), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("func", [tri, tpi, roughness])
def test_numpy_equals_dask(random_data, func):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, func)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(4, 6), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("func", [tri, tpi, roughness])
def test_numpy_equals_cupy(random_data, func):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, func)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(4, 6), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("func", [tri, tpi, roughness])
def test_numpy_equals_dask_cupy(random_data, func):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, func,
                                  atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Boundary mode tests
# ---------------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("func", [tri, tpi, roughness])
def test_boundary_modes(func):
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data)
    dask_agg = create_test_raster(data, backend='dask+numpy')
    assert_boundary_mode_correctness(numpy_agg, dask_agg, func)


@dask_array_available
@pytest.mark.parametrize("func", [tri, tpi, roughness])
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((6, 8), (3, 4)),
    ((7, 9), (3, 3)),
    ((10, 15), (5, 5)),
    ((10, 15), (10, 15)),
    ((5, 5), (2, 2)),
])
def test_boundary_numpy_equals_dask(func, boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = func(numpy_agg, boundary=boundary)
    da_result = func(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("func", [tri, tpi, roughness])
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_flat(func, boundary):
    """Flat surface with non-nan boundary should produce all zeros, no NaN."""
    data = np.full((8, 10), 50.0, dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=(4, 5))
    np_result = func(numpy_agg, boundary=boundary)
    da_result = func(dask_agg, boundary=boundary)
    np.testing.assert_allclose(np_result.data, 0.0, atol=1e-10)
    np.testing.assert_allclose(np_result.data, da_result.data.compute(),
                               equal_nan=True, rtol=1e-6)


@pytest.mark.parametrize("func", [tri, tpi, roughness])
def test_boundary_invalid(func):
    data = np.ones((4, 5), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="boundary must be one of"):
        func(agg, boundary='invalid')


# ---------------------------------------------------------------------------
# Dtype preservation tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", [tri, tpi, roughness])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_dtype_acceptance(func, dtype):
    data = np.arange(20, dtype=dtype).reshape(4, 5)
    agg = create_test_raster(data)
    result = func(agg)
    assert result.shape == agg.shape
    assert result.dims == agg.dims


# ---------------------------------------------------------------------------
# Dataset support tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func,expected_name", [
    (tri, 'tri'),
    (tpi, 'tpi'),
    (roughness, 'roughness'),
])
def test_dataset_support(func, expected_name):
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    da1 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(2.5, 0, 6)
    da1['x'] = np.linspace(0, 3.5, 8)
    ds = xr.Dataset({'elev1': da1, 'elev2': da1 * 2})
    result = func(ds)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'elev1', 'elev2'}
    # Individual results should match calling on a DataArray directly
    for var in result.data_vars:
        expected = func(ds[var], name=var)
        np.testing.assert_allclose(
            result[var].data, expected.data, equal_nan=True)


# ---------------------------------------------------------------------------
# Landforms tests
# ---------------------------------------------------------------------------

def _gaussian_peak(rows=50, cols=60, sigma=8):
    """Gaussian peak centered in the raster."""
    y = np.arange(rows, dtype=np.float64)
    x = np.arange(cols, dtype=np.float64)
    Y, X = np.meshgrid(y, x, indexing='ij')
    return 100 * np.exp(-((Y - rows / 2)**2 + (X - cols / 2)**2)
                        / (2 * sigma**2))


def test_landforms_flat():
    """Flat surface should classify as plains (class 5)."""
    data = np.full((30, 40), 100.0, dtype=np.float64)
    agg = create_test_raster(data)
    result = landforms(agg, inner_radius=2, outer_radius=5)
    valid = ~np.isnan(result.data)
    np.testing.assert_array_equal(result.data[valid], 5.0)


def test_landforms_gaussian_peak():
    """Gaussian peak center should classify as mountain top."""
    data = _gaussian_peak()
    agg = create_test_raster(data)
    result = landforms(agg, inner_radius=2, outer_radius=8)
    center_class = result.data[25, 30]
    assert center_class == 10.0, (
        f"Expected class 10 (mountain top) at peak, got {center_class}")


def test_landforms_multiple_classes():
    """Varied terrain should produce several distinct classes."""
    rows, cols = 80, 100
    y = np.arange(rows, dtype=np.float64)
    x = np.arange(cols, dtype=np.float64)
    Y, X = np.meshgrid(y, x, indexing='ij')
    data = (50 * np.sin(Y / 8) * np.cos(X / 12)
            + 30 * np.sin(Y / 4) + 200)
    agg = create_test_raster(data)
    result = landforms(agg, inner_radius=2, outer_radius=8)
    unique = np.unique(result.data[~np.isnan(result.data)])
    assert len(unique) >= 3, f"Expected >= 3 classes, got {unique}"


def test_landforms_nan_propagation():
    """NaN in elevation should propagate to output."""
    data = np.random.default_rng(42).random((30, 40)) * 100
    data[15, 20] = np.nan
    agg = create_test_raster(data)
    result = landforms(agg, inner_radius=2, outer_radius=5)
    assert np.isnan(result.data[15, 20])


def test_landforms_input_nan_preserved():
    """Cells with NaN elevation should remain NaN in output."""
    data = np.random.default_rng(42).random((30, 40)) * 100
    data[5, 10] = np.nan
    data[20, 30] = np.nan
    agg = create_test_raster(data)
    result = landforms(agg, inner_radius=2, outer_radius=5)
    assert np.isnan(result.data[5, 10])
    assert np.isnan(result.data[20, 30])


def test_landforms_invalid_inner_radius():
    data = np.ones((10, 10), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="inner_radius must be >= 1"):
        landforms(agg, inner_radius=0)


def test_landforms_invalid_outer_radius():
    data = np.ones((10, 10), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="outer_radius.*must be greater"):
        landforms(agg, inner_radius=5, outer_radius=3)


def test_landforms_outer_radius_memory_guard():
    """Issue #1302: huge outer_radius must raise MemoryError before
    allocating the circular kernel."""
    data = np.ones((10, 10), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(MemoryError, match="outer_radius=200000"):
        landforms(agg, inner_radius=3, outer_radius=200000)


def test_landforms_inner_radius_memory_guard():
    """Issue #1302: huge inner_radius must also raise MemoryError."""
    data = np.ones((10, 10), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(MemoryError, match="inner_radius=200000"):
        landforms(agg, inner_radius=200000, outer_radius=300000)


def test_landforms_output_shape_and_attrs():
    data = np.random.default_rng(42).random((30, 40)) * 100
    agg = create_test_raster(data)
    result = landforms(agg, inner_radius=2, outer_radius=5)
    assert result.shape == agg.shape
    assert result.dims == agg.dims
    assert result.name == 'landforms'
    assert result.attrs == agg.attrs


def test_landforms_valid_class_range():
    """All non-NaN outputs should be in 1..10."""
    data = np.random.default_rng(42).random((30, 40)) * 100
    agg = create_test_raster(data)
    result = landforms(agg, inner_radius=2, outer_radius=5)
    valid = result.data[~np.isnan(result.data)]
    assert np.all((valid >= 1) & (valid <= 10))


def test_landforms_class_labels_complete():
    """LANDFORM_CLASSES should have entries for classes 1-10."""
    assert set(LANDFORM_CLASSES.keys()) == set(range(1, 11))


@dask_array_available
def test_landforms_numpy_equals_dask():
    """Numpy and dask backends should produce matching classifications."""
    data = np.random.default_rng(42).random((30, 40)) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask', chunks=(15, 20))
    np_result = landforms(numpy_agg, inner_radius=2, outer_radius=5)
    da_result = landforms(dask_agg, inner_radius=2, outer_radius=5)
    np_data = np_result.data
    da_data = da_result.data.compute()
    # Allow tiny fraction of boundary-cell mismatches from floating point
    valid = ~np.isnan(np_data) & ~np.isnan(da_data)
    matches = np.sum(np_data[valid] == da_data[valid])
    total = np.sum(valid)
    assert matches / total > 0.98, (
        f"Only {matches}/{total} cells match between numpy and dask")


def test_landforms_slope_threshold():
    """Adjusting slope_threshold should shift cells between class 5 and 6."""
    data = np.random.default_rng(42).random((30, 40)) * 100
    agg = create_test_raster(data)
    low_thresh = landforms(agg, inner_radius=2, outer_radius=5,
                           slope_threshold=1.0)
    high_thresh = landforms(agg, inner_radius=2, outer_radius=5,
                            slope_threshold=80.0)
    # With very low threshold: fewer plains (class 5), more open slopes (6)
    low_plains = np.sum(low_thresh.data == 5.0)
    high_plains = np.sum(high_thresh.data == 5.0)
    assert high_plains >= low_plains


def test_landforms_dataset_support():
    data = np.random.default_rng(42).random((30, 40)).astype(np.float64) * 100
    da1 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(14.5, 0, 30)
    da1['x'] = np.linspace(0, 19.5, 40)
    ds = xr.Dataset({'elev1': da1, 'elev2': da1 * 2})
    result = landforms(ds, inner_radius=2, outer_radius=5)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'elev1', 'elev2'}
