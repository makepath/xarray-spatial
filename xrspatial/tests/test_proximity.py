from unittest.mock import patch

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial import allocation, direction, euclidean_distance, great_circle_distance, proximity
from xrspatial.proximity import _calc_direction
from xrspatial.tests.general_checks import general_output_checks, create_test_raster


def test_great_circle_distance():
    # invalid x_coord
    ys = [0, 0, -91, 91]
    xs = [-181, 181, 0, 0]
    for x, y in zip(xs, ys):
        with pytest.raises(Exception) as e_info:
            great_circle_distance(x1=0, x2=x, y1=0, y2=y)
            assert e_info


@pytest.fixture
def test_raster(backend):
    height, width = 4, 6
    # create test raster, all non-zero cells are unique,
    # this is to test allocation and direction against corresponding proximity
    data = np.asarray([[0., 0., 0., 0., 0., 2.],
                       [0., 0., 1., 0., 0., 0.],
                       [0., np.inf, 3., 0., 0., 0.],
                       [4., 0., 0., 0., np.nan, 0.]])
    _lon = np.linspace(-20, 20, width)
    _lat = np.linspace(20, -20, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(data, chunks=(4, 3))
    return raster


@pytest.fixture
def result_default_proximity():
    # DEFAULT SETTINGS
    expected_result = np.array([
        [20.82733247, 15.54920505, 13.33333333, 15.54920505,  8., 0.],
        [16., 8., 0., 8., 15.54920505, 13.33333333],
        [13.33333333, 8., 0., 8., 16., 24.],
        [0., 8., 13.33333333, 15.54920505, 20.82733247, 27.45501371]
    ], dtype=np.float32)
    return expected_result


@pytest.fixture
def result_target_proximity():
    target_values = [2, 3]
    expected_result = np.array([
        [31.09841011, 27.84081736, 24., 16., 8., 0.],
        [20.82733247, 15.54920505, 13.33333333, 15.54920505, 15.54920505, 13.33333333],
        [16., 8., 0., 8., 16., 24.],
        [20.82733247, 15.54920505, 13.33333333, 15.54920505, 20.82733247, 27.45501371]
    ], dtype=np.float32)
    return target_values, expected_result


@pytest.fixture
def result_manhattan_proximity():
    # distance_metric SETTING: MANHATTAN
    expected_result = np.array([
        [29.33333333, 21.33333333, 13.33333333, 16., 8., 0.],
        [16., 8., 0., 8., 16., 13.33333333],
        [13.33333333, 8., 0., 8., 16., 24.],
        [0., 8., 13.33333333, 21.33333333, 29.33333333, 37.33333333]
    ], dtype=np.float32)
    return expected_result


@pytest.fixture
def result_great_circle_proximity():
    # distance_metric SETTING: GREAT_CIRCLE
    expected_result = np.array([
        [2278099.27025501, 1717528.97437217, 1484259.87724365, 1673057.17235307, 836769.1780019, 0],
        [1768990.54084204, 884524.60324856, 0, 884524.60324856, 1717528.97437217, 1484259.87724365],
        [1484259.87724365, 884524.60324856, 0, 884524.60324856, 1768990.54084204, 2653336.85436932],
        [0, 836769.1780019, 1484259.87724365, 1717528.97437217, 2278099.27025501, 2986647.12982316]
    ], dtype=np.float32)
    return expected_result


@pytest.fixture
def result_max_distance_proximity():
    # max_distance setting
    max_distance = 10
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, 8., 0.],
        [np.nan, 8., 0., 8., np.nan, np.nan],
        [np.nan, 8., 0., 8., np.nan, np.nan],
        [0., 8., np.nan, np.nan, np.nan, np.nan]
    ], dtype=np.float32)
    return max_distance, expected_result


@pytest.fixture
def result_default_allocation():
    expected_result = np.array([
        [1., 1., 1., 1., 2., 2.],
        [1., 1., 1., 1., 2., 2.],
        [4., 3., 3., 3., 3., 3.],
        [4., 4., 3., 3., 3., 3.]
    ], dtype=np.float32)
    return expected_result


@pytest.fixture
def result_max_distance_allocation():
    # max_distance setting
    max_distance = 10
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, 2., 2.],
        [np.nan, 1., 1., 1., np.nan, np.nan],
        [np.nan, 3., 3., 3., np.nan, np.nan],
        [4., 4., np.nan, np.nan, np.nan, np.nan]
    ], dtype=np.float32)
    return max_distance, expected_result


@pytest.fixture
def result_default_direction():
    expected_result = np.array([
        [50.194427, 30.963757, 360., 329.03625, 90., 0.],
        [90., 90., 0., 270., 149.03624, 180.],
        [360., 90., 0., 270., 270., 270.],
        [0., 270., 180., 210.96376, 230.19443, 240.9454]
    ], dtype=np.float32)
    return expected_result


@pytest.fixture
def result_max_distance_direction():
    # max_distance setting
    max_distance = 10
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, 90., 0.],
        [np.nan, 90., 0., 270., np.nan, np.nan],
        [np.nan, 90., 0., 270., np.nan, np.nan],
        [0., 270., np.nan, np.nan, np.nan, np.nan]
    ], dtype=np.float32)
    return max_distance, expected_result


@pytest.fixture
def qgis_proximity_distance_target_values():
    target_values = [1]
    qgis_result = np.array([
        [1.802776, 1.414214, 1.118034, 1., 0.5, 0.],
        [1.581139, 1.118034, 0.707107, 0.5, 0.707107, 0.5],
        [1.118034, 1., 0.5, 0., 0.5, 1.],
        [0.707107, 0.5, 0.707107, 0.5, 0.707107, 1.118034],
        [0.5, 0., 0.5, 1., 1.118034, 1.414214],
        [0.707107, 0.5, 0.707107, 1.118034, 1., 1.],
        [0.5, 0., 0.5, 0.707107, 0.5, 0.5],
        [0.707107, 0.5, 0.707107, 0.5, 0., 0.]], dtype=np.float32)
    return target_values, qgis_result


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_default_proximity(test_raster, result_default_proximity):
    default_prox = proximity(test_raster, x='lon', y='lat')
    general_output_checks(test_raster, default_prox, result_default_proximity, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_target_proximity(test_raster, result_target_proximity):
    target_values, expected_result = result_target_proximity
    target_prox = proximity(test_raster, x='lon', y='lat', target_values=target_values)
    general_output_checks(test_raster, target_prox, expected_result, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_manhattan_proximity(test_raster, result_manhattan_proximity):
    manhattan_prox = proximity(test_raster, x='lon', y='lat', distance_metric='MANHATTAN')
    general_output_checks(
        test_raster, manhattan_prox, result_manhattan_proximity, verify_dtype=True
    )


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_great_circle_proximity(test_raster, result_great_circle_proximity):
    great_circle_prox = proximity(test_raster, x='lon', y='lat', distance_metric='GREAT_CIRCLE')
    general_output_checks(
        test_raster, great_circle_prox, result_great_circle_proximity, verify_dtype=True
    )


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_max_distance_proximity(test_raster, result_max_distance_proximity):
    max_distance, expected_result = result_max_distance_proximity
    max_distance_prox = proximity(test_raster, x='lon', y='lat', max_distance=max_distance)
    general_output_checks(test_raster, max_distance_prox, expected_result, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_default_allocation(test_raster, result_default_allocation):
    allocation_agg = allocation(test_raster, x='lon', y='lat')
    general_output_checks(test_raster, allocation_agg, result_default_allocation, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy'])
def test_default_allocation_against_proximity(test_raster, result_default_proximity):
    allocation_agg = allocation(test_raster, x='lon', y='lat')
    # check against corresponding proximity
    xcoords = allocation_agg['lon'].data
    ycoords = allocation_agg['lat'].data
    for y in range(test_raster.shape[0]):
        for x in range(test_raster.shape[1]):
            a = allocation_agg.data[y, x]
            py, px = np.where(test_raster.data == a)
            # non-zero cells in raster are unique, thus len(px)=len(py)=1
            d = euclidean_distance(xcoords[x], xcoords[px[0]], ycoords[y], ycoords[py[0]])
            np.testing.assert_allclose(result_default_proximity[y, x], d)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_max_distance_allocation(test_raster, result_max_distance_allocation):
    max_distance, expected_result = result_max_distance_allocation
    max_distance_alloc = allocation(test_raster, x='lon', y='lat', max_distance=max_distance)
    general_output_checks(test_raster, max_distance_alloc, expected_result, verify_dtype=True)


def test_calc_direction():
    n = 3
    x1, y1 = 1, 1
    output = np.zeros(shape=(n, n))
    for y2 in range(n):
        for x2 in range(n):
            output[y2, x2] = _calc_direction(x2, x1, y2, y1)

    expected_output = np.asarray([[135, 180, 225],
                                  [90,  0,   270],
                                  [45,  360, 315]])
    # set a tolerance of 1e-5
    tolerance = 1e-5
    assert (abs(output-expected_output) <= tolerance).all()


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_default_direction(test_raster, result_default_direction):
    direction_agg = direction(test_raster, x='lon', y='lat')
    general_output_checks(test_raster, direction_agg, result_default_direction)


@pytest.mark.parametrize("backend", ['numpy'])
def test_default_direction_against_allocation(test_raster, result_default_allocation):
    direction_agg = direction(test_raster, x='lon', y='lat')
    xcoords = direction_agg['lon'].data
    ycoords = direction_agg['lat'].data
    for y in range(test_raster.shape[0]):
        for x in range(test_raster.shape[1]):
            a = result_default_allocation.data[y, x]
            py, px = np.where(test_raster.data == a)
            # non-zero cells in raster are unique, thus len(px)=len(py)=1
            d = _calc_direction(xcoords[x], xcoords[px[0]], ycoords[y], ycoords[py[0]])
            np.testing.assert_allclose(direction_agg.data[y, x], d)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_max_distance_direction(test_raster, result_max_distance_direction):
    max_distance, expected_result = result_max_distance_direction
    max_distance_direction = direction(test_raster, x='lon', y='lat', max_distance=max_distance)
    general_output_checks(test_raster, max_distance_direction, expected_result, verify_dtype=True)


def test_proximity_distance_against_qgis(raster, qgis_proximity_distance_target_values):
    target_values, qgis_result = qgis_proximity_distance_target_values
    input_raster = create_test_raster(raster)

    # proximity by xrspatial
    xrspatial_result = proximity(input_raster, target_values=target_values)

    general_output_checks(input_raster, xrspatial_result)
    np.testing.assert_allclose(xrspatial_result.data, qgis_result.data, rtol=1e-05, equal_nan=True)


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_coord_arrays_are_lazy():
    """
    Test that coordinate arrays (xs, ys) are created as dask arrays
    when input is a dask array, avoiding memory issues with large rasters.

    This is a regression test for the issue where xs and ys were created
    as numpy arrays before checking if the input was a dask array,
    causing memory issues for large datasets.
    """
    from unittest.mock import patch

    height, width = 100, 120
    data = np.zeros((height, width), dtype=np.float64)
    # Add some target pixels
    data[10, 10] = 1.0
    data[50, 60] = 2.0
    data[90, 100] = 3.0

    _lon = np.linspace(-180, 180, width)
    _lat = np.linspace(90, -90, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    # Create dask-backed array with chunks
    raster.data = da.from_array(data, chunks=(25, 30))

    # Track calls to np.tile and np.repeat with the full raster shape
    original_tile = np.tile
    original_repeat = np.repeat
    large_numpy_array_created = []

    def tracking_tile(A, reps):
        result = original_tile(A, reps)
        # Check if result would be the size of the full coordinate grid
        if result.size >= height * width:
            large_numpy_array_created.append(('tile', result.shape))
        return result

    def tracking_repeat(a, repeats, axis=None):
        result = original_repeat(a, repeats, axis=axis)
        # Check if result would be the size of the full coordinate grid
        if result.size >= height * width:
            large_numpy_array_created.append(('repeat', result.shape))
        return result

    with patch.object(np, 'tile', tracking_tile):
        with patch.object(np, 'repeat', tracking_repeat):
            result = proximity(raster, x='lon', y='lat')

    # Verify no large numpy coordinate arrays were created
    assert len(large_numpy_array_created) == 0, (
        f"Large numpy arrays were created for coordinates: {large_numpy_array_created}. "
        "For dask inputs, coordinate arrays should be created using dask operations."
    )

    # Verify result is a dask array
    assert isinstance(result.data, da.Array), "Result should be a dask array"

    # Verify correctness by computing and checking a few values
    computed = result.compute()
    # Check that target pixels have distance 0
    assert computed.data[10, 10] == 0.0
    assert computed.data[50, 60] == 0.0
    assert computed.data[90, 100] == 0.0
    # Check that non-target pixels have positive distance
    assert computed.data[0, 0] > 0.0


def _make_kdtree_raster(height=20, width=30, chunks=(10, 15)):
    """Helper: build a small dask-backed raster with a few target pixels."""
    data = np.zeros((height, width), dtype=np.float64)
    data[3, 5] = 1.0
    data[12, 20] = 2.0
    data[18, 2] = 3.0
    _lon = np.linspace(0, 29, width)
    _lat = np.linspace(19, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=chunks)
    return raster


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("metric", ["EUCLIDEAN", "MANHATTAN"])
def test_proximity_dask_kdtree_matches_numpy(metric):
    """k-d tree dask result must match numpy result for the same raster."""
    raster = _make_kdtree_raster()
    numpy_raster = raster.copy()
    numpy_raster.data = raster.data.compute()

    numpy_result = proximity(numpy_raster, x='lon', y='lat',
                             distance_metric=metric)
    dask_result = proximity(raster, x='lon', y='lat',
                            distance_metric=metric)

    assert isinstance(dask_result.data, da.Array)
    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_no_large_arrays():
    """No full-raster-sized numpy arrays should be created in k-d tree path."""
    height, width = 100, 120
    data = np.zeros((height, width), dtype=np.float64)
    data[10, 10] = 1.0
    data[50, 60] = 2.0

    _lon = np.linspace(0, 119, width)
    _lat = np.linspace(99, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(25, 30))

    original_tile = np.tile
    original_repeat = np.repeat
    large_numpy_created = []

    def tracking_tile(A, reps):
        result = original_tile(A, reps)
        if result.size >= height * width:
            large_numpy_created.append(('tile', result.shape))
        return result

    def tracking_repeat(a, repeats, axis=None):
        result = original_repeat(a, repeats, axis=axis)
        if result.size >= height * width:
            large_numpy_created.append(('repeat', result.shape))
        return result

    with patch.object(np, 'tile', tracking_tile):
        with patch.object(np, 'repeat', tracking_repeat):
            result = proximity(raster, x='lon', y='lat')

    assert len(large_numpy_created) == 0, (
        f"Large numpy arrays created: {large_numpy_created}"
    )
    assert isinstance(result.data, da.Array)


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_with_target_values():
    """target_values filtering works through the k-d tree path."""
    raster = _make_kdtree_raster()
    numpy_raster = raster.copy()
    numpy_raster.data = raster.data.compute()

    target_values = [2, 3]
    numpy_result = proximity(numpy_raster, x='lon', y='lat',
                             target_values=target_values)
    dask_result = proximity(raster, x='lon', y='lat',
                            target_values=target_values)

    assert isinstance(dask_result.data, da.Array)
    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_no_targets():
    """No target pixels found → result is all NaN."""
    data = np.zeros((10, 10), dtype=np.float64)
    _lon = np.arange(10, dtype=np.float64)
    _lat = np.arange(10, dtype=np.float64)[::-1]
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(5, 5))

    result = proximity(raster, x='lon', y='lat')
    assert isinstance(result.data, da.Array)
    computed = result.values
    assert np.all(np.isnan(computed))


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_max_distance():
    """max_distance truncation works via distance_upper_bound in tree query."""
    raster = _make_kdtree_raster()
    numpy_raster = raster.copy()
    numpy_raster.data = raster.data.compute()

    max_dist = 5.0
    numpy_result = proximity(numpy_raster, x='lon', y='lat',
                             max_distance=max_dist)
    dask_result = proximity(raster, x='lon', y='lat',
                            max_distance=max_dist)

    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_fallback_no_scipy():
    """When cKDTree is None, falls back to single-chunk path."""
    import sys
    prox_mod = sys.modules['xrspatial.proximity']

    height, width = 8, 10
    data = np.zeros((height, width), dtype=np.float64)
    data[2, 3] = 1.0
    data[6, 8] = 2.0
    _lon = np.linspace(0, 9, width)
    _lat = np.linspace(7, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(4, 5))

    original_ckdtree = prox_mod.cKDTree
    try:
        prox_mod.cKDTree = None
        result = proximity(raster, x='lon', y='lat')
        assert isinstance(result.data, da.Array)
        # Should still produce correct results via fallback
        computed = result.values
        assert computed[2, 3] == 0.0
    finally:
        prox_mod.cKDTree = original_ckdtree


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_fallback_great_circle():
    """GREAT_CIRCLE metric falls back to single-chunk, not k-d tree."""
    import sys
    prox_mod = sys.modules['xrspatial.proximity']

    height, width = 8, 10
    data = np.zeros((height, width), dtype=np.float64)
    data[2, 3] = 1.0
    _lon = np.linspace(-10, 10, width)
    _lat = np.linspace(10, -10, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(4, 5))

    # Patch _process_dask_kdtree to detect if it's called
    kdtree_called = []
    original_fn = prox_mod._process_dask_kdtree

    def spy(*args, **kwargs):
        kdtree_called.append(True)
        return original_fn(*args, **kwargs)

    with patch.object(prox_mod, '_process_dask_kdtree', spy):
        result = proximity(raster, x='lon', y='lat',
                           distance_metric='GREAT_CIRCLE')

    assert len(kdtree_called) == 0, "k-d tree path should not be used for GREAT_CIRCLE"
    assert isinstance(result.data, da.Array)
