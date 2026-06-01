import inspect
from textwrap import dedent as textwrap_dedent
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
from xrspatial.tests.general_checks import (
    general_output_checks, create_test_raster, has_cuda_and_cupy,
)


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
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(4, 3))
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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_default_proximity(test_raster, result_default_proximity):
    default_prox = proximity(test_raster, x='lon', y='lat')
    general_output_checks(test_raster, default_prox, result_default_proximity, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_target_proximity(test_raster, result_target_proximity):
    target_values, expected_result = result_target_proximity
    target_prox = proximity(test_raster, x='lon', y='lat', target_values=target_values)
    general_output_checks(test_raster, target_prox, expected_result, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_manhattan_proximity(test_raster, result_manhattan_proximity):
    manhattan_prox = proximity(test_raster, x='lon', y='lat', distance_metric='MANHATTAN')
    general_output_checks(
        test_raster, manhattan_prox, result_manhattan_proximity, verify_dtype=True
    )


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_great_circle_proximity(test_raster, result_great_circle_proximity):
    great_circle_prox = proximity(test_raster, x='lon', y='lat', distance_metric='GREAT_CIRCLE')
    general_output_checks(
        test_raster, great_circle_prox, result_great_circle_proximity, verify_dtype=True
    )


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_max_distance_proximity(test_raster, result_max_distance_proximity):
    max_distance, expected_result = result_max_distance_proximity
    max_distance_prox = proximity(test_raster, x='lon', y='lat', max_distance=max_distance)
    general_output_checks(test_raster, max_distance_prox, expected_result, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_max_distance_direction(test_raster, result_max_distance_direction):
    max_distance, expected_result = result_max_distance_direction
    max_distance_direction = direction(test_raster, x='lon', y='lat', max_distance=max_distance)
    general_output_checks(test_raster, max_distance_direction, expected_result, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize("max_distance", [10, np.inf])
def test_output_metadata_consistent_across_backends(
    test_raster, func, max_distance
):
    # Regression: the declared (pre-compute) output dtype must be float32 and
    # .name must be None on every backend.  The bounded dask path previously
    # declared float64 meta, and dask backends leaked an internal op name
    # (e.g. "_trim-<hash>") into .name.
    result = func(test_raster, x='lon', y='lat', max_distance=max_distance)
    assert result.dtype == np.float32
    assert result.name is None
    assert result.dims == test_raster.dims
    assert result.attrs == test_raster.attrs


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


# ---------------------------------------------------------------------------
# Tiled KDTree fallback tests (memory-guarded path)
# ---------------------------------------------------------------------------

def _force_tiled_proximity(raster, **kwargs):
    """Run proximity with _available_memory_bytes mocked to force tiled path.

    Uses a counter-based side_effect:
      call 1 (_stream_target_counts cache budget): returns 1  → tiny cache
      call 2 (_process_dask_kdtree decision):      returns 1  → forces tiled
      call 3+ (_build_tiled_kdtree result check):  returns 10 GB → passes guard
    """
    call_count = [0]

    def _small_then_large():
        call_count[0] += 1
        if call_count[0] <= 2:
            return 1
        return 10 * 1024 ** 3

    with patch('xrspatial.proximity._available_memory_bytes',
               side_effect=_small_then_large):
        return proximity(raster, **kwargs)


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_tiled_matches_numpy():
    """Dense raster forced through tiled path must match numpy baseline."""
    height, width = 20, 30
    rng = np.random.RandomState(42)
    data = rng.choice([0.0, 1.0, 2.0], size=(height, width), p=[0.3, 0.4, 0.3])
    _lon = np.linspace(0, 29, width)
    _lat = np.linspace(19, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat

    numpy_result = proximity(raster, x='lon', y='lat')

    raster.data = da.from_array(data, chunks=(5, 10))
    dask_result = _force_tiled_proximity(raster, x='lon', y='lat')

    assert isinstance(dask_result.data, da.Array)
    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_tiled_manhattan():
    """Tiled path with MANHATTAN metric matches numpy."""
    height, width = 16, 20
    rng = np.random.RandomState(99)
    data = rng.choice([0.0, 1.0, 2.0], size=(height, width), p=[0.3, 0.4, 0.3])
    _lon = np.linspace(0, 19, width)
    _lat = np.linspace(15, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat

    numpy_result = proximity(raster, x='lon', y='lat',
                             distance_metric='MANHATTAN')

    raster.data = da.from_array(data, chunks=(4, 5))
    dask_result = _force_tiled_proximity(raster, x='lon', y='lat',
                                         distance_metric='MANHATTAN')

    assert isinstance(dask_result.data, da.Array)
    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_tiled_single_target():
    """One target in a corner, many chunks → exercises max ring expansion."""
    height, width = 20, 20
    data = np.zeros((height, width), dtype=np.float64)
    data[0, 0] = 1.0

    _lon = np.linspace(0, 19, width)
    _lat = np.linspace(19, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat

    numpy_result = proximity(raster, x='lon', y='lat')

    raster.data = da.from_array(data, chunks=(5, 5))
    dask_result = _force_tiled_proximity(raster, x='lon', y='lat')

    assert isinstance(dask_result.data, da.Array)
    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_tiled_all_targets():
    """Every pixel is a target → result should be all zeros."""
    height, width = 12, 12
    data = np.ones((height, width), dtype=np.float64)
    _lon = np.linspace(0, 11, width)
    _lat = np.linspace(11, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat

    raster.data = da.from_array(data, chunks=(4, 4))
    dask_result = _force_tiled_proximity(raster, x='lon', y='lat')

    assert isinstance(dask_result.data, da.Array)
    np.testing.assert_allclose(dask_result.values, 0.0)


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_tiled_no_targets():
    """No targets, forced tiled path → all NaN."""
    data = np.zeros((10, 10), dtype=np.float64)
    _lon = np.arange(10, dtype=np.float64)
    _lat = np.arange(10, dtype=np.float64)[::-1]
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(5, 5))

    # Even with tiny memory, zero targets should return early (all NaN)
    dask_result = _force_tiled_proximity(raster, x='lon', y='lat')
    assert isinstance(dask_result.data, da.Array)
    assert np.all(np.isnan(dask_result.values))


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_tiled_warns():
    """Verify ResourceWarning fires when tiled fallback is selected."""
    height, width = 10, 10
    data = np.zeros((height, width), dtype=np.float64)
    data[5, 5] = 1.0
    _lon = np.linspace(0, 9, width)
    _lat = np.linspace(9, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(5, 5))

    with pytest.warns(ResourceWarning, match="tiled KDTree fallback"):
        _force_tiled_proximity(raster, x='lon', y='lat')


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_kdtree_global_uses_cache():
    """Global path still works correctly after Phase 0 restructure."""
    raster = _make_kdtree_raster()
    numpy_raster = raster.copy()
    numpy_raster.data = raster.data.compute()

    numpy_result = proximity(numpy_raster, x='lon', y='lat')

    # Global path (default): _available_memory_bytes returns large value
    with patch('xrspatial.proximity._available_memory_bytes',
               return_value=10 * 1024**3):
        dask_result = proximity(raster, x='lon', y='lat')

    assert isinstance(dask_result.data, da.Array)
    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Phase-0 streaming count pass: row-batched compute (gh-2724)
# ---------------------------------------------------------------------------

def test_stream_target_counts_no_compute_in_inner_loop():
    """_stream_target_counts must not call .compute() inside the inner loop.

    Structural guard for gh-2724: the per-chunk count pass batches one
    dask.compute() per chunk-row, not one blocking .compute() per chunk.
    """
    import ast
    import inspect
    from xrspatial.proximity import _stream_target_counts

    src = inspect.getsource(_stream_target_counts)
    tree = ast.parse(textwrap_dedent(src))

    # Find .compute() attribute calls and the for-loops enclosing them.
    compute_calls = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.for_depth = 0

        def visit_For(self, node):
            self.for_depth += 1
            self.generic_visit(node)
            self.for_depth -= 1

        def visit_Call(self, node):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "compute":
                compute_calls.append(self.for_depth)
            self.generic_visit(node)

    _Visitor().visit(tree)

    assert compute_calls, "expected a compute() call in the count pass"
    # da.compute(...) is issued at the outer (row) loop level => for_depth == 1.
    # A regression to per-chunk .compute() would sit in the inner loop => 2.
    assert max(compute_calls) <= 1, (
        "compute() is called inside the inner chunk loop; the count pass "
        "should batch one da.compute() per chunk-row (gh-2724)."
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_stream_target_counts_matches_sequential_baseline():
    """Row-batched count pass matches a sequential per-chunk baseline.

    Verifies target_counts, total_targets, and the coords/values caches are
    byte-identical to the pre-gh-2724 sequential implementation.
    """
    from xrspatial.proximity import _stream_target_counts, _target_mask

    height, width = 24, 30
    rng = np.random.RandomState(7)
    data = rng.choice([0.0, 1.0, 2.0], size=(height, width),
                      p=[0.3, 0.4, 0.3])
    _lon = np.linspace(0, 29, width)
    _lat = np.linspace(23, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(6, 10))

    target_values = np.asarray([])
    chunks_y, chunks_x = raster.data.chunks

    counts, total, coords_cache, values_cache = _stream_target_counts(
        raster, target_values, _lat, _lon, chunks_y, chunks_x,
    )

    # Sequential reference (the pre-fix behaviour).
    n_tile_y, n_tile_x = len(chunks_y), len(chunks_x)
    y_off = np.concatenate([[0], np.cumsum(chunks_y)])
    x_off = np.concatenate([[0], np.cumsum(chunks_x)])
    ref_counts = np.zeros((n_tile_y, n_tile_x), dtype=np.int64)
    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            cd = raster.data.blocks[iy, ix].compute()
            mask = _target_mask(cd, target_values)
            rows, cols = np.where(mask)
            ref_counts[iy, ix] = len(rows)

    np.testing.assert_array_equal(counts, ref_counts)
    assert total == int(ref_counts.sum())
    # Cache holds the same target coords/values it would have under the
    # sequential pass (full raster easily fits the 25% budget here).
    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            cd = raster.data.blocks[iy, ix].compute()
            mask = _target_mask(cd, target_values)
            rows, cols = np.where(mask)
            if len(rows) == 0:
                assert (iy, ix) not in coords_cache
                continue
            assert (iy, ix) in coords_cache
            expected_coords = np.column_stack([
                _lat[y_off[iy] + rows],
                _lon[x_off[ix] + cols],
            ])
            np.testing.assert_array_equal(coords_cache[(iy, ix)],
                                          expected_coords)
            np.testing.assert_array_equal(
                values_cache[(iy, ix)],
                cd[rows, cols].astype(np.float32),
            )


# ---------------------------------------------------------------------------
# Allocation / Direction KDTree tests
# ---------------------------------------------------------------------------

def _check_allocation_consistency(raster_data, alloc_data, prox_data,
                                   x_coords, y_coords, metric):
    """Verify allocation is consistent: distance to allocated target == proximity."""
    from xrspatial.proximity import euclidean_distance, manhattan_distance
    dist_fn = euclidean_distance if metric == "EUCLIDEAN" else manhattan_distance

    h, w = raster_data.shape
    for y in range(h):
        for x in range(w):
            a = alloc_data[y, x]
            p = prox_data[y, x]
            if np.isnan(a):
                assert np.isnan(p), f"NaN allocation but non-NaN proximity at ({y},{x})"
                continue
            # Find target pixel(s) with this value
            ty, tx = np.where(raster_data == a)
            assert len(ty) > 0, f"Allocated value {a} not found in raster"
            min_d = min(
                dist_fn(x_coords[x], x_coords[tx[i]],
                        y_coords[y], y_coords[ty[i]])
                for i in range(len(ty))
            )
            np.testing.assert_allclose(p, min_d, rtol=1e-4,
                                       err_msg=f"at ({y},{x})")


def _check_direction_consistency(raster_data, dir_data, alloc_data,
                                  x_coords, y_coords):
    """Verify direction is consistent with allocation."""
    h, w = raster_data.shape
    for y in range(h):
        for x in range(w):
            d = dir_data[y, x]
            a = alloc_data[y, x]
            if np.isnan(d):
                assert np.isnan(a), f"NaN direction but non-NaN allocation at ({y},{x})"
                continue
            ty, tx = np.where(raster_data == a)
            assert len(ty) > 0
            expected = _calc_direction(x_coords[x], x_coords[tx[0]],
                                       y_coords[y], y_coords[ty[0]])
            # 0 and 360 are equivalent for north
            if abs(d - expected) > 359.0:
                np.testing.assert_allclose(
                    d % 360.0, expected % 360.0, atol=0.5,
                    err_msg=f"at ({y},{x})")
            else:
                np.testing.assert_allclose(d, expected, rtol=1e-4, atol=0.5,
                                           err_msg=f"at ({y},{x})")


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("metric", ["EUCLIDEAN", "MANHATTAN"])
def test_allocation_dask_kdtree_matches_numpy(metric):
    """k-d tree dask allocation is correct (consistent with proximity)."""
    raster = _make_kdtree_raster()

    dask_alloc = allocation(raster, x='lon', y='lat',
                            distance_metric=metric)
    dask_prox = proximity(raster, x='lon', y='lat',
                          distance_metric=metric)

    assert isinstance(dask_alloc.data, da.Array)
    _check_allocation_consistency(
        raster.data.compute(), dask_alloc.values, dask_prox.values,
        raster['lon'].values, raster['lat'].values, metric,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("metric", ["EUCLIDEAN", "MANHATTAN"])
def test_direction_dask_kdtree_matches_numpy(metric):
    """k-d tree dask direction is correct (consistent with allocation)."""
    raster = _make_kdtree_raster()

    dask_dir = direction(raster, x='lon', y='lat', distance_metric=metric)
    dask_alloc = allocation(raster, x='lon', y='lat', distance_metric=metric)

    assert isinstance(dask_dir.data, da.Array)
    _check_direction_consistency(
        raster.data.compute(), dask_dir.values, dask_alloc.values,
        raster['lon'].values, raster['lat'].values,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_allocation_dask_kdtree_with_target_values():
    """target_values filtering works through the k-d tree allocation path."""
    raster = _make_kdtree_raster()

    target_values = [2, 3]
    dask_alloc = allocation(raster, x='lon', y='lat',
                            target_values=target_values)
    dask_prox = proximity(raster, x='lon', y='lat',
                          target_values=target_values)

    assert isinstance(dask_alloc.data, da.Array)
    alloc_vals = dask_alloc.values
    valid_vals = np.isnan(alloc_vals) | np.isin(alloc_vals, target_values)
    assert np.all(valid_vals), "Allocation produced values outside target set"
    _check_allocation_consistency(
        raster.data.compute(), alloc_vals, dask_prox.values,
        raster['lon'].values, raster['lat'].values, "EUCLIDEAN",
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_direction_dask_kdtree_with_target_values():
    """target_values filtering works through the k-d tree direction path."""
    raster = _make_kdtree_raster()

    target_values = [2, 3]
    dask_dir = direction(raster, x='lon', y='lat',
                         target_values=target_values)
    dask_alloc = allocation(raster, x='lon', y='lat',
                            target_values=target_values)

    assert isinstance(dask_dir.data, da.Array)
    vals = dask_dir.values
    valid = vals[~np.isnan(vals)]
    assert np.all((valid >= 0) & (valid <= 360))
    _check_direction_consistency(
        raster.data.compute(), dask_dir.values, dask_alloc.values,
        raster['lon'].values, raster['lat'].values,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_allocation_dask_kdtree_no_targets():
    """No target pixels -> allocation result is all NaN."""
    data = np.zeros((10, 10), dtype=np.float64)
    _lon = np.arange(10, dtype=np.float64)
    _lat = np.arange(10, dtype=np.float64)[::-1]
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(5, 5))

    result = allocation(raster, x='lon', y='lat')
    assert isinstance(result.data, da.Array)
    assert np.all(np.isnan(result.values))


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_direction_dask_kdtree_no_targets():
    """No target pixels -> direction result is all NaN."""
    data = np.zeros((10, 10), dtype=np.float64)
    _lon = np.arange(10, dtype=np.float64)
    _lat = np.arange(10, dtype=np.float64)[::-1]
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(5, 5))

    result = direction(raster, x='lon', y='lat')
    assert isinstance(result.data, da.Array)
    assert np.all(np.isnan(result.values))


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_allocation_dask_kdtree_max_distance():
    """Pixels beyond max_distance are NaN in allocation via KDTree."""
    raster = _make_kdtree_raster()
    numpy_raster = raster.copy()
    numpy_raster.data = raster.data.compute()

    max_dist = 5.0
    numpy_result = allocation(numpy_raster, x='lon', y='lat',
                              max_distance=max_dist)
    dask_result = allocation(raster, x='lon', y='lat',
                             max_distance=max_dist)

    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_direction_dask_kdtree_max_distance():
    """Pixels beyond max_distance are NaN in direction via KDTree."""
    raster = _make_kdtree_raster()
    numpy_raster = raster.copy()
    numpy_raster.data = raster.data.compute()

    max_dist = 5.0
    numpy_result = direction(numpy_raster, x='lon', y='lat',
                             max_distance=max_dist)
    dask_result = direction(raster, x='lon', y='lat',
                            max_distance=max_dist)

    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_great_circle_dask_unbounded_memory_guard():
    """GREAT_CIRCLE with unbounded max_distance raises MemoryError when OOM."""
    height, width = 100, 100
    data = np.zeros((height, width), dtype=np.float64)
    data[50, 50] = 1.0
    _lon = np.linspace(-10, 10, width)
    _lat = np.linspace(10, -10, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(25, 25))

    with patch('xrspatial.proximity._available_memory_bytes', return_value=1):
        with pytest.raises(MemoryError, match="GREAT_CIRCLE"):
            proximity(raster, x='lon', y='lat',
                      distance_metric='GREAT_CIRCLE')


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_no_scipy_dask_unbounded_memory_guard():
    """No scipy + large raster + unbounded distance raises MemoryError."""
    import sys
    prox_mod = sys.modules['xrspatial.proximity']

    height, width = 100, 100
    data = np.zeros((height, width), dtype=np.float64)
    data[50, 50] = 1.0
    _lon = np.linspace(0, 99, width)
    _lat = np.linspace(99, 0, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(25, 25))

    original_ckdtree = prox_mod.cKDTree
    try:
        prox_mod.cKDTree = None
        with patch('xrspatial.proximity._available_memory_bytes',
                   return_value=1):
            with pytest.raises(MemoryError, match="scipy"):
                proximity(raster, x='lon', y='lat')
    finally:
        prox_mod.cKDTree = original_ckdtree


@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_return_annotation_consistency(func):
    # proximity, allocation, and direction share a signature and return
    # type; their return annotations should match.
    assert inspect.signature(func).return_annotation is xr.DataArray


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_great_circle_dask_bounded_matches_numpy(func):
    """Bounded GREAT_CIRCLE on a dask raster must match the numpy result.

    Regression test: the map_overlap padding used the raw degree cellsize
    while max_distance is in metres for GREAT_CIRCLE, producing an overlap
    depth of hundreds of thousands of pixels and raising ValueError on
    valid input. numpy and cupy backends handled the same call fine.
    """
    height, width = 16, 24
    rng = np.random.RandomState(7)
    data = (rng.rand(height, width) < 0.06).astype(np.float64)
    _lon = np.linspace(-20, 20, width)
    _lat = np.linspace(20, -20, height)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat

    # bounded, smaller than the corner-to-corner great-circle distance
    max_dist = 1.5e6
    numpy_result = func(raster, x='lon', y='lat',
                        max_distance=max_dist,
                        distance_metric='GREAT_CIRCLE')

    dask_raster = raster.copy()
    dask_raster.data = da.from_array(data, chunks=(8, 12))
    dask_result = func(dask_raster, x='lon', y='lat',
                       max_distance=max_dist,
                       distance_metric='GREAT_CIRCLE')

    assert isinstance(dask_result.data, da.Array)
    np.testing.assert_allclose(
        dask_result.values, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Coverage gaps closed for issue #2692 (test-only; no source change).
#
# All behaviour below was verified correct and cross-backend consistent on a
# CUDA host before these tests were committed. Nothing here pins a bug.
# ---------------------------------------------------------------------------

def _backend_raster(data, backend):
    """Build a lat/lon DataArray on the requested backend.

    Mirrors the ``test_raster`` fixture: 1D lat/lon coords, then the data
    buffer swapped to cupy and/or dask as requested. Degenerate (length-1)
    axes get a single coordinate so meshgrid math stays well defined.
    """
    height, width = data.shape
    _lon = np.linspace(-20, 20, width) if width > 1 else np.array([0.0])
    _lat = np.linspace(20, -20, height) if height > 1 else np.array([0.0])
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(data)
    if 'dask' in backend and da is not None:
        chunks = (max(height // 2, 1), max(width // 2, 1))
        raster.data = da.from_array(raster.data, chunks=chunks)
    return raster


# --- Cat 3: degenerate raster shapes (1x1, Nx1, 1xN) ----------------------

# (shape data, expected proximity, expected allocation, expected direction)
_DEGENERATE_CASES = {
    "1x1": (
        np.array([[1.0]]),
        np.array([[0.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
    ),
    "Nx1": (
        np.array([[0.0], [1.0], [0.0], [0.0]]),
        np.array([[13.333333], [0.0], [13.333333], [26.666666]],
                 dtype=np.float32),
        np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32),
        np.array([[360.0], [0.0], [180.0], [180.0]], dtype=np.float32),
    ),
    "1xN": (
        np.array([[0.0, 1.0, 0.0, 0.0]]),
        np.array([[13.333333, 0.0, 13.333333, 26.666666]], dtype=np.float32),
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[90.0, 0.0, 270.0, 270.0]], dtype=np.float32),
    ),
}


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("shape_name", list(_DEGENERATE_CASES))
def test_proximity_degenerate_shapes(backend, shape_name):
    data, expected, _, _ = _DEGENERATE_CASES[shape_name]
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    raster = _backend_raster(data, backend)
    result = proximity(raster, x='lon', y='lat')
    general_output_checks(raster, result, expected, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("shape_name", list(_DEGENERATE_CASES))
def test_allocation_degenerate_shapes(backend, shape_name):
    data, _, expected, _ = _DEGENERATE_CASES[shape_name]
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    raster = _backend_raster(data, backend)
    result = allocation(raster, x='lon', y='lat')
    general_output_checks(raster, result, expected, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("shape_name", list(_DEGENERATE_CASES))
def test_direction_degenerate_shapes(backend, shape_name):
    data, _, _, expected = _DEGENERATE_CASES[shape_name]
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    raster = _backend_raster(data, backend)
    result = direction(raster, x='lon', y='lat')
    general_output_checks(raster, result, expected)


# --- Cat 1/4: non-default metrics on allocation / direction ----------------
# proximity exercises MANHATTAN and GREAT_CIRCLE cross-backend; allocation and
# direction only ever ran EUCLIDEAN across backends. Pin both metrics on all
# four backends against the numpy baseline.

@pytest.fixture
def _metric_raster_data():
    return np.asarray([[0., 0., 0., 0., 0., 2.],
                       [0., 0., 1., 0., 0., 0.],
                       [0., np.inf, 3., 0., 0., 0.],
                       [4., 0., 0., 0., np.nan, 0.]])


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("metric", ['MANHATTAN', 'GREAT_CIRCLE'])
def test_allocation_metric_backends(backend, metric, _metric_raster_data):
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    numpy_raster = _backend_raster(_metric_raster_data, 'numpy')
    expected = allocation(
        numpy_raster, x='lon', y='lat', distance_metric=metric).data

    raster = _backend_raster(_metric_raster_data, backend)
    result = allocation(raster, x='lon', y='lat', distance_metric=metric)
    general_output_checks(raster, result, expected, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("metric", ['MANHATTAN', 'GREAT_CIRCLE'])
def test_direction_metric_backends(backend, metric, _metric_raster_data):
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    numpy_raster = _backend_raster(_metric_raster_data, 'numpy')
    expected = direction(
        numpy_raster, x='lon', y='lat', distance_metric=metric).data

    raster = _backend_raster(_metric_raster_data, backend)
    result = direction(raster, x='lon', y='lat', distance_metric=metric)
    general_output_checks(raster, result, expected)


# --- Cat 5: attrs preservation with realistic res / crs --------------------
# The shared test_raster fixture carries no attrs, so the attrs check in
# general_output_checks is vacuous. proximity also reads attrs['res'] for the
# bounded-dask padding (get_dataarray_resolution), so a res attr that matches
# the coordinate spacing must round-trip and still produce correct results.

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_proximity_preserves_attrs(backend, func):
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    data = np.zeros((8, 8), dtype=np.float64)
    data[1, 1] = 1.0
    data[6, 6] = 2.0
    raster = _backend_raster(data, backend)
    raster.attrs = {'res': (40.0 / 7, 40.0 / 7), 'crs': 'EPSG:5070',
                    'units': 'm'}

    result = func(raster, x='lon', y='lat')
    assert result.attrs == raster.attrs
    assert result.dims == raster.dims
    np.testing.assert_allclose(result['lon'].data, raster['lon'].data)
    np.testing.assert_allclose(result['lat'].data, raster['lat'].data)


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_res_attr_drives_bounded_dask_padding():
    """Bounded dask reads attrs['res'] for chunk padding; with a res attr that
    matches the coordinate spacing the result must equal the numpy baseline."""
    data = np.zeros((8, 8), dtype=np.float64)
    data[1, 1] = 1.0
    data[6, 6] = 2.0
    res = 40.0 / 7  # coords run -20..20 over 8 cells

    numpy_raster = _backend_raster(data, 'numpy')
    numpy_raster.attrs = {'res': (res, res)}
    expected = proximity(
        numpy_raster, x='lon', y='lat', max_distance=20.0).data

    dask_raster = _backend_raster(data, 'dask+numpy')
    dask_raster.attrs = {'res': (res, res)}
    result = proximity(dask_raster, x='lon', y='lat', max_distance=20.0)

    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(
        result.values, expected, equal_nan=True, rtol=1e-5)


@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_target_values_none_default_matches_empty_list(func):
    # target_values default switched from [] to a None sentinel; passing
    # None (the default) must behave exactly like passing an empty list.
    data = np.asarray([[0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 2., 0., 0.],
                       [0., 0., 0., 0., 0.]])
    n, m = data.shape
    raster = xr.DataArray(data, dims=['y', 'x'])
    raster['y'] = np.arange(n)[::-1]
    raster['x'] = np.arange(m)

    default_result = func(raster)
    explicit_result = func(raster, target_values=[])

    np.testing.assert_array_equal(
        np.nan_to_num(default_result.data),
        np.nan_to_num(explicit_result.data),
    )
