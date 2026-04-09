import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_less

from xrspatial import hillshade
from xrspatial.hillshade import _run_numpy
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy,
                                            create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available,
                                            general_output_checks)

from ..gpu_rtx import has_rtx


@pytest.fixture
def data_gaussian():
    _x = np.linspace(0, 50, 101)
    _y = _x.copy()
    _mean = 25
    _sdev = 5
    X, Y = np.meshgrid(_x, _y, sparse=True)
    x_fac = -np.power(X-_mean, 2)
    y_fac = -np.power(Y-_mean, 2)
    gaussian = np.exp((x_fac+y_fac)/(2*_sdev**2)) / (2.5*_sdev)
    return gaussian


def _make_raster(data, cellsize_x=1.0, cellsize_y=1.0):
    """Build an xr.DataArray with coordinates spaced at given cell sizes."""
    rows, cols = data.shape
    raster = xr.DataArray(
        data, dims=['y', 'x'],
        coords={
            'y': np.arange(rows) * cellsize_y,
            'x': np.arange(cols) * cellsize_x,
        },
    )
    return raster


def test_hillshade(data_gaussian):
    """
    Assert Simple Hillshade transfer function
    """
    da_gaussian = _make_raster(data_gaussian)
    da_gaussian_shade = hillshade(da_gaussian, name='hillshade_agg')
    general_output_checks(da_gaussian, da_gaussian_shade)
    assert da_gaussian_shade.name == 'hillshade_agg'
    assert da_gaussian_shade.mean() > 0
    assert da_gaussian_shade[60, 60] > 0


def test_hillshade_output_range():
    """Output should be in [0, 1] (plus NaN borders)."""
    rng = np.random.default_rng(42)
    data = rng.random((20, 20)) * 100
    raster = _make_raster(data)
    result = hillshade(raster)
    interior = result.values[1:-1, 1:-1]
    assert np.all(interior >= 0.0)
    assert np.all(interior <= 1.0)


def test_hillshade_flat_surface():
    """Flat surface should be uniformly lit (constant interior value)."""
    data = np.full((10, 10), 100.0)
    raster = _make_raster(data)
    result = hillshade(raster)
    interior = result.values[1:-1, 1:-1]
    assert np.all(np.isfinite(interior))
    # All interior values should be identical on a flat surface
    assert_allclose(interior, interior[0, 0], atol=1e-7)


def test_hillshade_resolution_sensitivity():
    """
    Different cell sizes must produce different results for the same
    raw elevation values.  The old code ignored resolution and would
    produce identical output regardless of cell size.
    """
    rng = np.random.default_rng(123)
    data = np.cumsum(rng.random((15, 15)), axis=0) * 100

    result_1m = _run_numpy(data, cellsize_x=1.0, cellsize_y=1.0)
    result_10m = _run_numpy(data, cellsize_x=10.0, cellsize_y=10.0)

    # Interior values should differ when cell size changes
    interior_1m = result_1m[1:-1, 1:-1]
    interior_10m = result_10m[1:-1, 1:-1]
    assert not np.allclose(interior_1m, interior_10m), \
        "hillshade should be sensitive to cell resolution"


def _horn_reference(data, cellsize_x, cellsize_y, azimuth, altitude):
    """Pure-numpy Horn's method reference for testing.

    Computes gradients with the 8-neighbor Sobel kernel and applies the
    GDAL hillshade formula.
    """
    az_rad = azimuth * np.pi / 180.
    alt_rad = altitude * np.pi / 180.

    # Horn / Sobel kernel gradients (same as GDAL gdaldem)
    a = data[2:, :-2]
    b = data[2:, 1:-1]
    c = data[2:, 2:]
    d = data[1:-1, :-2]
    f = data[1:-1, 2:]
    g = data[:-2, :-2]
    h = data[:-2, 1:-1]
    i = data[:-2, 2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y)

    xx_plus_yy = dz_dx**2 + dz_dy**2
    shaded = (
        np.sin(alt_rad) + np.cos(alt_rad) * (dz_dy * np.cos(az_rad) - dz_dx * np.sin(az_rad))
    ) / np.sqrt(1 + xx_plus_yy)
    return np.clip(shaded, 0.0, 1.0)


def test_hillshade_gdal_equivalence():
    """
    Verify that _run_numpy matches the GDAL hillshade formula with
    Horn's method (weighted 8-neighbor Sobel kernel) for gradients.
    """
    rng = np.random.default_rng(99)
    data = rng.random((20, 20)).astype(np.float32) * 500
    cellsize_x, cellsize_y = 30.0, 30.0
    azimuth, altitude = 315.0, 45.0

    ref = _horn_reference(data, cellsize_x, cellsize_y, azimuth, altitude)

    result = _run_numpy(data, azimuth=azimuth, angle_altitude=altitude,
                        cellsize_x=cellsize_x, cellsize_y=cellsize_y)

    interior = slice(1, -1)
    assert_allclose(result[interior, interior], ref, atol=1e-6)


def test_hillshade_horn_vs_central_diff():
    """Horn's method should differ from simple central differences on noisy data.

    This validates that the implementation actually uses the 8-neighbor Sobel
    kernel rather than 2-point np.gradient, which was the bug in #1175.
    """
    rng = np.random.default_rng(1175)
    # Smooth ramp with noise to amplify the difference between methods
    rows, cols = 64, 64
    base = np.tile(np.linspace(0, 500, cols), (rows, 1))
    noise = rng.normal(0, 2, (rows, cols))
    data = (base + noise).astype(np.float32)
    cellsize_x, cellsize_y = 30.0, 30.0
    azimuth, altitude = 315.0, 45.0

    # Our result (should be Horn)
    result = _run_numpy(data, azimuth=azimuth, angle_altitude=altitude,
                        cellsize_x=cellsize_x, cellsize_y=cellsize_y)
    interior = slice(1, -1)

    # Horn reference
    horn_ref = _horn_reference(data, cellsize_x, cellsize_y, azimuth, altitude)
    assert_allclose(result[interior, interior], horn_ref, atol=1e-6)

    # np.gradient reference (old method) -- should NOT match
    az_rad = azimuth * np.pi / 180.
    alt_rad = altitude * np.pi / 180.
    dy, dx = np.gradient(data, cellsize_y, cellsize_x)
    xx_plus_yy = dx**2 + dy**2
    old_shaded = (
        np.sin(alt_rad) + np.cos(alt_rad) * (dy * np.cos(az_rad) - dx * np.sin(az_rad))
    ) / np.sqrt(1 + xx_plus_yy)
    old_ref = np.clip(old_shaded, 0.0, 1.0)

    max_diff = np.max(np.abs(result[interior, interior] - old_ref[interior, interior]))
    assert max_diff > 0.01, (
        f"Expected meaningful difference between Horn and central-diff, "
        f"got max diff {max_diff}"
    )


@dask_array_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_hillshade_numpy_equals_dask_numpy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, hillshade)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_hillshade_gpu_equals_cpu(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, hillshade, rtol=1e-6)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_hillshade_numpy_equals_dask_cupy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, hillshade,
                                  atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not has_rtx(), reason="RTX not available")
def test_hillshade_rtx_with_shadows(data_gaussian):
    import cupy

    tall_gaussian = 400*data_gaussian
    cpu = hillshade(_make_raster(tall_gaussian))

    tall_gaussian_cupy = cupy.asarray(tall_gaussian)
    raster_gpu = _make_raster(tall_gaussian)
    raster_gpu.data = tall_gaussian_cupy
    rtx = hillshade(raster_gpu)
    rtx.data = cupy.asnumpy(rtx.data)

    assert cpu.shape == rtx.shape
    nhalf = cpu.shape[0] // 2

    # Quadrant nearest sun direction should be almost identical.
    quad_cpu = cpu.data[nhalf::, ::nhalf]
    quad_rtx = cpu.data[nhalf::, ::nhalf]
    assert_allclose(quad_cpu, quad_rtx, atol=0.03)

    # Opposite diagonal should be in shadow.
    diag_cpu = np.diagonal(cpu.data[::-1])[nhalf:]
    diag_rtx = np.diagonal(rtx.data[::-1])[nhalf:]
    assert_array_less(diag_rtx, diag_cpu + 1e-3)


@dask_array_available
def test_boundary_modes(data_gaussian):
    numpy_agg = create_test_raster(data_gaussian, backend='numpy')
    dask_agg = create_test_raster(data_gaussian, backend='dask+numpy')
    assert_boundary_mode_correctness(numpy_agg, dask_agg, hillshade)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((6, 8), (3, 4)),
    ((7, 9), (3, 3)),
    ((10, 15), (5, 5)),
    ((10, 15), (10, 15)),
    ((5, 5), (2, 2)),
])
def test_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64) * 500
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = hillshade(numpy_agg, boundary=boundary)
    da_result = hillshade(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_edges(boundary):
    """Non-nan modes produce no NaN output when source has no NaN."""
    rng = np.random.default_rng(99)
    data = rng.random((12, 14)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=(4, 7))
    np_result = hillshade(numpy_agg, boundary=boundary)
    da_result = hillshade(dask_agg, boundary=boundary)
    assert not np.any(np.isnan(np_result.data))
    assert not np.any(np.isnan(da_result.data.compute()))
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = _make_raster(data)
    with pytest.raises(ValueError, match="boundary must be one of"):
        hillshade(agg, boundary='invalid')
