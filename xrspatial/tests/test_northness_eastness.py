import numpy as np
import pytest

from xrspatial import aspect, eastness, northness
from xrspatial.tests.general_checks import (assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy,
                                            dask_array_available,
                                            create_test_raster,
                                            cuda_and_cupy_available,
                                            general_output_checks)


def _expected_northness(aspect_data):
    """Reference: cos(aspect_degrees) with flat cells (-1) as NaN."""
    out = np.cos(np.deg2rad(aspect_data))
    out[aspect_data == -1] = np.nan
    return out


def _expected_eastness(aspect_data):
    """Reference: sin(aspect_degrees) with flat cells (-1) as NaN."""
    out = np.sin(np.deg2rad(aspect_data))
    out[aspect_data == -1] = np.nan
    return out


# ---- Correctness against known values ----

def test_northness_correctness(elevation_raster):
    agg = create_test_raster(elevation_raster, backend='numpy')
    asp = aspect(agg)
    result = northness(agg)

    general_output_checks(agg, result, verify_dtype=False)
    assert result.name == 'northness'

    expected = _expected_northness(asp.data)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, equal_nan=True)


def test_eastness_correctness(elevation_raster):
    agg = create_test_raster(elevation_raster, backend='numpy')
    asp = aspect(agg)
    result = eastness(agg)

    general_output_checks(agg, result, verify_dtype=False)
    assert result.name == 'eastness'

    expected = _expected_eastness(asp.data)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, equal_nan=True)


# ---- Range checks: output must be in [-1, +1] ----

def test_northness_range(elevation_raster):
    agg = create_test_raster(elevation_raster, backend='numpy')
    result = northness(agg)
    valid = result.data[~np.isnan(result.data)]
    assert np.all(valid >= -1) and np.all(valid <= 1)


def test_eastness_range(elevation_raster):
    agg = create_test_raster(elevation_raster, backend='numpy')
    result = eastness(agg)
    valid = result.data[~np.isnan(result.data)]
    assert np.all(valid >= -1) and np.all(valid <= 1)


# ---- Cardinal direction spot checks ----

def test_cardinal_directions():
    """Build a surface that slopes in known directions and verify cos/sin."""
    # North-facing slope: higher in south, lower in north
    # 3x3 kernel needs a 5x5 raster to get one non-edge pixel
    north_facing = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [5, 5, 5, 5, 5],
        [10, 10, 10, 10, 10],
        [15, 15, 15, 15, 15],
    ], dtype=np.float32)
    agg = create_test_raster(north_facing)
    n = northness(agg).data[2, 2]
    e = eastness(agg).data[2, 2]
    # North-facing: aspect ~0 or 360, northness ~1, eastness ~0
    assert n > 0.9, f"Expected northness ~1 for north-facing, got {n}"
    assert abs(e) < 0.2, f"Expected eastness ~0 for north-facing, got {e}"

    # East-facing slope: higher in west, lower in east
    east_facing = np.array([
        [15, 10, 5, 0, 0],
        [15, 10, 5, 0, 0],
        [15, 10, 5, 0, 0],
        [15, 10, 5, 0, 0],
        [15, 10, 5, 0, 0],
    ], dtype=np.float32)
    agg = create_test_raster(east_facing)
    n = northness(agg).data[2, 2]
    e = eastness(agg).data[2, 2]
    # East-facing: aspect ~90, northness ~0, eastness ~1
    assert abs(n) < 0.2, f"Expected northness ~0 for east-facing, got {n}"
    assert e > 0.9, f"Expected eastness ~1 for east-facing, got {e}"


# ---- Flat surface → NaN ----

def test_flat_surface_gives_nan():
    data = np.ones((5, 5), dtype=np.float32)
    agg = create_test_raster(data)
    n = northness(agg)
    e = eastness(agg)
    # The center pixel is flat (aspect == -1), should be NaN
    assert np.isnan(n.data[2, 2])
    assert np.isnan(e.data[2, 2])


# ---- NaN propagation ----

def test_nan_propagation():
    data = np.ones((5, 5), dtype=np.float32)
    data[0, :] = np.nan  # NaN in first row
    agg = create_test_raster(data)
    n = northness(agg)
    e = eastness(agg)
    # First row should be NaN in output (edges plus NaN input)
    assert np.all(np.isnan(n.data[0, :]))
    assert np.all(np.isnan(e.data[0, :]))


# ---- Custom name ----

def test_custom_name(elevation_raster):
    agg = create_test_raster(elevation_raster, backend='numpy')
    n = northness(agg, name='my_north')
    e = eastness(agg, name='my_east')
    assert n.name == 'my_north'
    assert e.name == 'my_east'


# ---- Backend parity: Dask+NumPy ----

@dask_array_available
def test_northness_numpy_equals_dask(elevation_raster):
    numpy_agg = create_test_raster(elevation_raster, backend='numpy')
    dask_agg = create_test_raster(elevation_raster, backend='dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, northness)


@dask_array_available
def test_eastness_numpy_equals_dask(elevation_raster):
    numpy_agg = create_test_raster(elevation_raster, backend='numpy')
    dask_agg = create_test_raster(elevation_raster, backend='dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, eastness)


@dask_array_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_northness_numpy_equals_dask_random(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, northness)


@dask_array_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_eastness_numpy_equals_dask_random(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, eastness)


# ---- Backend parity: CuPy ----

@cuda_and_cupy_available
def test_northness_numpy_equals_cupy(elevation_raster):
    numpy_agg = create_test_raster(elevation_raster, backend='numpy')
    cupy_agg = create_test_raster(elevation_raster, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, northness, atol=1e-6, rtol=1e-6)


@cuda_and_cupy_available
def test_eastness_numpy_equals_cupy(elevation_raster):
    numpy_agg = create_test_raster(elevation_raster, backend='numpy')
    cupy_agg = create_test_raster(elevation_raster, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, eastness, atol=1e-6, rtol=1e-6)


# ---- Backend parity: Dask+CuPy ----

@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_northness_numpy_equals_dask_cupy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, northness, atol=1e-6, rtol=1e-6)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_eastness_numpy_equals_dask_cupy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, eastness, atol=1e-6, rtol=1e-6)
