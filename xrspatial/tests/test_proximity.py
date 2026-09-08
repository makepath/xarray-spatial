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


def _simple_raster():
    data = np.asarray([[0., 0., 1.],
                       [0., 0., 0.],
                       [2., 0., 0.]])
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = np.linspace(-20, 20, 3)
    raster['lat'] = np.linspace(20, -20, 3)
    return raster


@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_invalid_distance_metric_raises(func):
    # A typoed / unsupported distance_metric must raise rather than
    # silently falling back to EUCLIDEAN (issue #2807).
    raster = _simple_raster()
    with pytest.raises(ValueError) as e_info:
        func(raster, x='lon', y='lat', distance_metric='euclidian')
    msg = str(e_info.value)
    assert 'euclidian' in msg
    # the error should list the valid options
    for metric in ('EUCLIDEAN', 'GREAT_CIRCLE', 'MANHATTAN'):
        assert metric in msg


@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize(
    "metric", ['EUCLIDEAN', 'GREAT_CIRCLE', 'MANHATTAN'])
def test_valid_distance_metric_runs(func, metric):
    # All documented metrics must keep working unchanged (issue #2807).
    raster = _simple_raster()
    result = func(raster, x='lon', y='lat', distance_metric=metric)
    assert result.shape == raster.shape


def test_great_circle_distance_returns_meters():
    # One degree of longitude at the equator spans ~111319.49 meters
    # (radius * radians(1) = 6378137 * pi/180). In kilometers this would
    # be ~111, so this fixed literal pins the unit down and keeps the
    # docstring's "meters" claim honest.
    dist = great_circle_distance(x1=0.0, x2=1.0, y1=0.0, y2=0.0)
    assert dist == pytest.approx(111319.49, abs=1e-2)


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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize("max_distance", [-1, -0.5, -np.inf, np.nan])
def test_invalid_max_distance_raises(test_raster, func, max_distance):
    # A negative or NaN max_distance is meaningless and used to produce
    # backend-dependent output (numpy squared it, the CUDA path compared
    # against it directly).  It must raise on every backend instead.
    with pytest.raises(ValueError, match="max_distance"):
        func(test_raster, x='lon', y='lat', max_distance=max_distance)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_zero_max_distance_keeps_meaning(test_raster, func):
    # Edge value: max_distance=0 is valid and means only target cells
    # qualify.  It must not raise.
    result = func(test_raster, x='lon', y='lat', max_distance=0)
    general_output_checks(test_raster, result)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize("target_values", [[np.inf], [-np.inf], [np.nan], [2, np.inf]])
def test_non_finite_target_values_raises(test_raster, func, target_values):
    # A non-finite target_values entry used to produce backend-dependent
    # output: numpy matched inf pixels and returned a real grid, while
    # dask/cupy masked non-finite pixels out and returned all-NaN for the
    # same raster (issue #2850).  It must raise on every backend instead.
    with pytest.raises(ValueError, match="target_values"):
        func(test_raster, x='lon', y='lat', target_values=target_values)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_non_numeric_target_values_raises(test_raster, func):
    # A non-numeric target_values can't index a raster; it must raise a clear
    # ValueError on every backend rather than a downstream TypeError (#2850).
    with pytest.raises(ValueError, match="target_values"):
        func(test_raster, x='lon', y='lat', target_values=['a', 'b'])


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_finite_target_values_run(test_raster, func):
    # The raster carries an inf and a nan pixel; the default (empty
    # target_values) path must keep ignoring them on every backend, and
    # explicit finite targets must keep working unchanged.
    result_default = func(test_raster, x='lon', y='lat')
    general_output_checks(test_raster, result_default)
    result_explicit = func(test_raster, x='lon', y='lat', target_values=[1, 3])
    general_output_checks(test_raster, result_explicit)


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


# ---------------------------------------------------------------------------
# Tie-breaking: when two targets are equidistant, every backend must pick the
# same one. The documented policy is "lowest flat (row-major) index wins".
# ---------------------------------------------------------------------------

@pytest.fixture
def tie_break_raster_data():
    # Targets at (1, 0)=1 and (1, 2)=2. The whole centre column is equidistant
    # to both. Target 1 sits at flat index 3, target 2 at flat index 5, so the
    # lowest-flat-index policy allocates the centre column to 1.
    return np.array([[0., 0., 0.],
                     [1., 0., 2.],
                     [0., 0., 0.]], dtype=np.float64)


@pytest.fixture
def tie_break_expected_allocation():
    return np.array([[1., 1., 2.],
                     [1., 1., 2.],
                     [1., 1., 2.]], dtype=np.float32)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_allocation_tie_break_lowest_flat_index(
        backend, tie_break_raster_data, tie_break_expected_allocation):
    raster = create_test_raster(
        tie_break_raster_data, backend=backend, dims=['lat', 'lon'],
        chunks=(1, 1),
    )
    result = allocation(raster, x='lon', y='lat')
    general_output_checks(
        raster, result, tie_break_expected_allocation, verify_dtype=True,
    )


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_direction_tie_break_matches_numpy(backend, tie_break_raster_data):
    # The numpy backend is the reference. Pin every other backend to it so the
    # direction angle chosen on a tie stays identical across backends.
    numpy_raster = create_test_raster(
        tie_break_raster_data, backend='numpy', dims=['lat', 'lon'],
    )
    expected = direction(numpy_raster, x='lon', y='lat').data

    raster = create_test_raster(
        tie_break_raster_data, backend=backend, dims=['lat', 'lon'],
        chunks=(1, 1),
    )
    result = direction(raster, x='lon', y='lat')
    general_output_checks(raster, result, expected)


@pytest.fixture
def tie_break_chunk_column_data():
    # Regression data for issue #3090. Pixel (2, 3) is equidistant to target
    # 2 at (1, 3) (flat index 8) and target 3 at (2, 2) (flat index 12), so
    # the documented lowest-flat-index policy picks 2. With chunks=(5, 3) the
    # raster splits into two chunk columns and target 3 sits in chunk (0, 0)
    # while target 2 sits in chunk (0, 1). The dask KDTree path used to
    # collect targets chunk by chunk, so its tree order was [1, 3, 2] instead
    # of the global row-major [1, 2, 3] and the tie resolved to 3.
    return np.array([[0., 0., 0., 0., 0.],
                     [0., 1., 0., 2., 0.],
                     [0., 0., 3., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]], dtype=np.float64)


@pytest.mark.parametrize("backend", ['dask+numpy', 'dask+cupy'])
def test_allocation_tie_break_chunk_column_order(
        backend, tie_break_chunk_column_data):
    numpy_raster = create_test_raster(
        tie_break_chunk_column_data, backend='numpy', dims=['lat', 'lon'],
    )
    expected = allocation(numpy_raster, x='lon', y='lat').data
    # The tied pixel must go to target 2 (lowest flat index).
    assert expected[2, 3] == 2.0

    raster = create_test_raster(
        tie_break_chunk_column_data, backend=backend, dims=['lat', 'lon'],
        chunks=(5, 3),
    )
    result = allocation(raster, x='lon', y='lat')
    general_output_checks(raster, result, expected, verify_dtype=True)


@pytest.mark.parametrize("backend", ['dask+numpy', 'dask+cupy'])
def test_direction_tie_break_chunk_column_order(
        backend, tie_break_chunk_column_data):
    numpy_raster = create_test_raster(
        tie_break_chunk_column_data, backend='numpy', dims=['lat', 'lon'],
    )
    expected = direction(numpy_raster, x='lon', y='lat').data

    raster = create_test_raster(
        tie_break_chunk_column_data, backend=backend, dims=['lat', 'lon'],
        chunks=(5, 3),
    )
    result = direction(raster, x='lon', y='lat')
    general_output_checks(raster, result, expected)


@pytest.fixture
def tie_break_nonlattice_raster_data():
    # Regression data for issue #3689. On a res-0.1 grid the linspace
    # coordinates are not exact float64 lattice points, so a pixel that is
    # geometrically equidistant to two targets sees float64 distances
    # separated by rounding noise (~1e-13) that vanishes at float32, the
    # precision of the output. The tie must be detected at float32 and
    # resolved to the lowest flat (row-major) index on every backend. The
    # numpy brute force already compared float32 distances, but the CUDA
    # kernel compared float64 and the cKDTree paths detected ties with exact
    # float64 equality, so each backend allocated such pixels to whichever
    # target its own float64 arithmetic favoured.
    data = np.zeros((40, 40), dtype=np.float64)
    data[27, 35] = 1.0   # target A, flat index 1115
    data[37, 25] = 2.0   # target B, flat index 1505
    return data


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("max_distance", [np.inf, 3.5])
@pytest.mark.parametrize("func", [allocation, direction])
def test_tie_break_float32_precision_nonlattice_grid(
        backend, max_distance, func, tie_break_nonlattice_raster_data):
    data = tie_break_nonlattice_raster_data
    res_attrs = {'res': (0.1, 0.1)}

    numpy_raster = create_test_raster(data, backend='numpy', attrs=res_attrs)

    # Sanity-check the fixture geometry: at least one pixel must be tied at
    # float32 while its float64 distances favour target B (the higher flat
    # index). Those are the pixels where a float64 comparison diverges from
    # the documented float32 tie-break.
    # Use the implementation's exact formula (sqrt of the sum of squares, see
    # euclidean_distance), not np.hypot: the two can differ by one float64
    # ulp, which is enough to flip a float32 tie in this regime.
    xs = numpy_raster['x'].data
    ys = numpy_raster['y'].data
    dx_a, dy_a = xs[None, :] - xs[35], ys[:, None] - ys[27]
    dx_b, dy_b = xs[None, :] - xs[25], ys[:, None] - ys[37]
    d_a = np.sqrt(dx_a * dx_a + dy_a * dy_a)
    d_b = np.sqrt(dx_b * dx_b + dy_b * dy_b)
    tied_f32 = np.float32(d_a) == np.float32(d_b)
    assert (tied_f32 & (d_b < d_a) & (d_a <= max_distance)).any()

    expected = func(numpy_raster, max_distance=max_distance).data

    # Every float32-tied pixel within range resolves to target A, the lowest
    # flat index: allocation returns A's value; direction returns the angle
    # toward A, which differs from the angle toward B by tens of degrees.
    in_range = tied_f32 & (np.float32(d_a) <= max_distance)
    if func is allocation:
        assert (expected[in_range] == 1.0).all()

    raster = create_test_raster(
        data, backend=backend, attrs=res_attrs, chunks=(16, 16))
    result = func(raster, max_distance=max_distance)
    general_output_checks(raster, result, expected)


def _raster_on_backend(raster, backend, chunks=(2, 2)):
    raster = raster.copy()
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=chunks)
    return raster


@pytest.fixture
def tie_break_proxy_gap_raster():
    # The brute-force kernel compares a cheap proxy (the squared distance)
    # and only rounds to float32 when the proxy beats the running best. Here
    # the proxies differ by a whole unit, 36000000 against 36000001, yet both
    # distances round to float32(6000.0). The float64-closer target B must
    # still lose to target A, the lower flat index (issue #3740).
    data = np.zeros((3, 3), dtype=np.float64)
    data[1, 2] = 1.0   # target A at (x=6000, y=1), flat index 5
    data[2, 1] = 2.0   # target B at (x=4800, y=3600), flat index 7
    raster = xr.DataArray(data, dims=['y', 'x'])
    raster['x'] = np.array([0.0, 4800.0, 6000.0])
    raster['y'] = np.array([0.0, 1.0, 3600.0])
    return raster


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [allocation, direction])
def test_tie_break_float32_beats_float64_proxy(
        backend, func, tie_break_proxy_gap_raster):
    numpy_raster = tie_break_proxy_gap_raster
    d_a = np.sqrt(6000.0 * 6000.0 + 1.0 * 1.0)
    d_b = np.sqrt(4800.0 * 4800.0 + 3600.0 * 3600.0)
    assert d_b < d_a
    assert np.float32(d_a) == np.float32(d_b)

    expected = func(numpy_raster).data
    if func is allocation:
        assert expected[0, 0] == 1.0
    else:
        assert expected[0, 0] == _calc_direction(0.0, 6000.0, 0.0, 1.0)

    raster = _raster_on_backend(numpy_raster, backend)
    result = func(raster)
    general_output_checks(raster, result, expected)


@pytest.mark.parametrize("which, bad_x, bad_y, point", [
    ('pixel', -181.0, None, 'first'),
    ('target', 181.0, None, 'second'),
    ('pixel', None, 91.0, 'first'),
    ('target', None, -91.0, 'second'),
])
def test_bruteforce_great_circle_range_check_messages(
        which, bad_x, bad_y, point):
    # GREAT_CIRCLE validates the coordinate grids once before the pixel loop
    # and must raise the same message the per-pair guard in
    # great_circle_distance raises for that coordinate (issue #3740).
    from xrspatial.proximity import GREAT_CIRCLE, PROXIMITY, _process_numpy_bruteforce

    img = np.zeros((2, 2))
    img[1, 1] = 1.0
    xs = np.tile(np.array([0.0, 1.0]), 2).reshape(2, 2)
    ys = np.repeat(np.array([0.0, 1.0]), 2).reshape(2, 2)
    row, col = (0, 0) if which == 'pixel' else (1, 1)
    if bad_x is not None:
        xs[row, col] = bad_x
        kwargs = dict(x1=0.0, x2=0.0, y1=0.0, y2=0.0)
        kwargs['x1' if point == 'first' else 'x2'] = bad_x
    else:
        ys[row, col] = bad_y
        kwargs = dict(x1=0.0, x2=0.0, y1=0.0, y2=0.0)
        kwargs['y1' if point == 'first' else 'y2'] = bad_y
    with pytest.raises(ValueError) as reference:
        great_circle_distance(**kwargs)
    with pytest.raises(ValueError) as kernel:
        _process_numpy_bruteforce(
            img, xs, ys, np.array([]), np.float32(np.inf),
            GREAT_CIRCLE, PROXIMITY)
    assert str(kernel.value) == str(reference.value)
    assert point in str(kernel.value)


def test_bruteforce_kernel_compiled_parallel():
    # prange over rows is a plain serial loop unless the kernel is compiled
    # with parallel=True; guard against that regressing (issue #3740).
    from xrspatial.proximity import _bruteforce_kernel
    assert _bruteforce_kernel.targetoptions.get('parallel') is True


def test_bruteforce_concurrent_launches_match_serial():
    # The parallel kernel is launched per chunk from dask worker threads and
    # is serialized behind _PARALLEL_KERNEL_LOCK; hammer it from a thread
    # pool and check every caller gets the single-threaded answer.
    from concurrent.futures import ThreadPoolExecutor

    # Use a raster large enough that the launches actually overlap.
    data = np.zeros((40, 40), dtype=np.float64)
    rng = np.random.default_rng(3740)
    data.flat[rng.choice(data.size, 50, replace=False)] = rng.integers(
        1, 6, 50)
    raster = create_test_raster(data, backend='numpy')
    expected = allocation(raster).data

    def one(_):
        return allocation(raster).data

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(32)))
    for result in results:
        np.testing.assert_array_equal(result, expected)


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
def test_allocation_tie_break_tiled_path():
    """The eager tiled KDTree fallback obeys the lowest-flat-index tie-break."""
    data = np.array([[0., 0., 0.],
                     [1., 0., 2.],
                     [0., 0., 0.]], dtype=np.float64)
    _lon = np.array([0., 1., 2.])
    _lat = np.array([2., 1., 0.])
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    raster.data = da.from_array(data, chunks=(1, 1))

    # Force the eager tiled KDTree path (see _force_tiled_proximity for the
    # counter semantics): tiny cache budget, force the tiled decision, then a
    # large value so the result-size guard passes.
    call_count = [0]

    def _small_then_large():
        call_count[0] += 1
        if call_count[0] <= 2:
            return 1
        return 10 * 1024 ** 3

    with patch('xrspatial.proximity._available_memory_bytes',
               side_effect=_small_then_large):
        result = allocation(raster, x='lon', y='lat')

    expected = np.array([[1., 1., 2.],
                         [1., 1., 2.],
                         [1., 1., 2.]], dtype=np.float32)
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_allocation_tie_break_tiled_path_chunk_column_order():
    """The tiled KDTree fallback ties row-major across chunk columns (#3090).

    Same geometry as test_allocation_tie_break_chunk_column_order: target 3
    is collected from an earlier chunk than target 2, so a chunk-major tree
    order would flip the tie at pixel (2, 3) to 3.
    """
    data = np.array([[0., 0., 0., 0., 0.],
                     [0., 1., 0., 2., 0.],
                     [0., 0., 3., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]], dtype=np.float64)
    _lon = np.arange(5, dtype=np.float64)
    _lat = np.arange(5, dtype=np.float64)[::-1]
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat

    numpy_expected = allocation(raster.copy(), x='lon', y='lat').values
    assert numpy_expected[2, 3] == 2.0

    raster.data = da.from_array(data, chunks=(5, 3))

    # Force the eager tiled KDTree path (see _force_tiled_proximity for the
    # counter semantics): tiny cache budget, force the tiled decision, then a
    # large value so the result-size guard passes.
    call_count = [0]

    def _small_then_large():
        call_count[0] += 1
        if call_count[0] <= 2:
            return 1
        return 10 * 1024 ** 3

    with patch('xrspatial.proximity._available_memory_bytes',
               side_effect=_small_then_large):
        result = allocation(raster, x='lon', y='lat')

    np.testing.assert_array_equal(result.values, numpy_expected)


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

    counts, total, coords_cache, values_cache, flat_cache = \
        _stream_target_counts(
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
            np.testing.assert_array_equal(
                flat_cache[(iy, ix)],
                (y_off[iy] + rows) * width + (x_off[ix] + cols),
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


@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_docstring_states_all_backends(func):
    # All three functions dispatch to numpy, cupy, dask+numpy, and dask+cupy
    # backends; the docstrings used to claim numpy and dask+numpy only
    # (issue #3091). This pins exact phrases to keep the old wording from
    # coming back; if the docs are reworded while staying accurate, update
    # the pinned strings rather than the docs.
    doc = func.__doc__
    assert "CuPy" in doc
    assert "Dask with CuPy" in doc
    assert "support NumPy backed, and Dask with NumPy backed" not in doc
    # The stray slope line that opened direction()'s docstring (#3091).
    assert "downward slope" not in doc


def test_allocation_docstring_example_matches_output():
    # The example in allocation()'s docstring predated the lowest-flat-index
    # tie-break and the float32 result dtype (issue #3091). Pin the printed
    # output to what the function returns.
    data = np.array([
        [0., 0., 0., 0., 0.],
        [0., 1., 0., 2., 0.],
        [0., 0., 3., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ])
    n, m = data.shape
    raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
    raster['y'] = np.arange(n)[::-1]
    raster['x'] = np.arange(m)

    result = allocation(raster)

    expected = np.array([[1., 1., 1., 2., 2.],
                         [1., 1., 1., 2., 2.],
                         [1., 1., 3., 2., 2.],
                         [1., 3., 3., 3., 2.],
                         [3., 3., 3., 3., 3.]], dtype=np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.values, expected)
    # Pixel (0, 2) ties between targets 1 and 2; the documented policy picks
    # the lowest flat index, which is target 1.
    assert result.values[0, 2] == 1.0


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
# Regression tests for issue #3108: bounded GREAT_CIRCLE on dask missed
# targets across the +/-180 antimeridian seam. The map_overlap halo is sized
# in array space, but great-circle distance is periodic in longitude, so a
# target near one edge of the array can be the nearest target of a pixel
# near the opposite edge. numpy/cupy (whole-raster brute force) found it;
# dask+numpy/dask+cupy returned NaN.
# ---------------------------------------------------------------------------

def _antimeridian_raster():
    lon = np.arange(-179.5, 180.0, 1.0)   # 360 columns spanning the seam
    lat = np.arange(-5.0, 6.0, 1.0)
    data = np.zeros((lat.size, lon.size))
    data[5, 0] = 1.0                       # target at (lat 0, lon -179.5)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = lon
    raster['lat'] = lat
    return raster


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("backend", ['dask+numpy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_great_circle_dask_antimeridian_matches_numpy(backend, func):
    """Bounded GREAT_CIRCLE dask sees targets across the +/-180 seam."""
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")

    raster = _antimeridian_raster()
    kwargs = dict(x='lon', y='lat', distance_metric='GREAT_CIRCLE',
                  max_distance=250_000.0)

    numpy_result = func(raster, **kwargs)
    # The wrap pixel at lon 179.5 is ~111 km from the target at lon -179.5,
    # within max_distance: it must be found, not NaN.
    assert np.isfinite(numpy_result.values[5, -1])

    chunk_data = raster.data
    if 'cupy' in backend:
        import cupy
        chunk_data = cupy.asarray(chunk_data)
    dask_raster = raster.copy(data=da.from_array(chunk_data, chunks=(11, 60)))
    dask_result = func(dask_raster, **kwargs)

    result_data = dask_result.data.compute()
    if 'cupy' in backend:
        result_data = result_data.get()
    np.testing.assert_allclose(
        result_data, numpy_result.values, rtol=1e-5, equal_nan=True,
    )


def test_great_circle_halo_folds_on_wrap_and_pole():
    """_halo_depth signals an x-axis fold when the chord bound cannot help."""
    from xrspatial.proximity import GREAT_CIRCLE, _halo_depth

    # Raster spanning the antimeridian: seam gap (1 degree) within reach.
    lon = np.arange(-179.5, 180.0, 1.0)
    lat = np.arange(-5.0, 6.0, 1.0)
    _, pad_x = _halo_depth(lon, lat, 250_000.0, GREAT_CIRCLE)
    assert pad_x > len(lon), "wrap-reachable raster must fold the x axis"

    # Pole-adjacent raster: the 180-degree chord at lat 89.75 is ~62 km,
    # within max_distance, so no longitude separation excludes a target.
    lon = np.arange(0.0, 180.0, 1.0)
    lat = np.arange(88.0, 90.0, 0.25)
    _, pad_x = _halo_depth(lon, lat, 500_000.0, GREAT_CIRCLE)
    assert pad_x > len(lon), "pole-adjacent raster must fold the x axis"

    # Regional raster far from seam and pole: finite halo, no fold.
    lon = np.arange(0.0, 10.0, 0.1)
    lat = np.arange(40.0, 45.0, 0.1)
    pad_y, pad_x = _halo_depth(lon, lat, 50_000.0, GREAT_CIRCLE)
    assert 0 < pad_x < len(lon)
    assert pad_y > 0

    # Descending coordinates (allowed by the monotonic check) give the same
    # halo: span and step are direction-independent.
    pad_y_desc, pad_x_desc = _halo_depth(
        lon[::-1], lat[::-1], 50_000.0, GREAT_CIRCLE)
    assert (pad_y_desc, pad_x_desc) == (pad_y, pad_x)

    # Descending wrap raster still folds.
    lon_desc = np.arange(-179.5, 180.0, 1.0)[::-1]
    lat_desc = np.arange(-5.0, 6.0, 1.0)[::-1]
    _, pad_x = _halo_depth(lon_desc, lat_desc, 250_000.0, GREAT_CIRCLE)
    assert pad_x > len(lon_desc)


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


# --- issue #2809: bounded-dask halo depth on irregular / degenerate coords --


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_bounded_dask_irregular_coords_matches_numpy(func):
    """Bounded dask must match numpy when coords are irregularly spaced.

    Regression for issue #2809 bug 1: the halo depth was taken from only the
    first coordinate pair. With a large leading gap and dense spacing
    afterwards, the overlap came out too thin and chunks dropped valid targets
    just past their boundary, so in-range cells turned to NaN under dask.
    """
    # First gap is 100, every later gap is 1 -> first-pair spacing would size
    # the halo at 0 pixels for max_distance=2.5, missing nearby targets.
    coords = np.array([0, 100, 101, 102, 103, 104, 105, 106], dtype=float)
    data = np.zeros((8, 8), dtype=np.float64)
    data[3, 3] = 1.0
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = coords
    raster['lat'] = coords

    expected = func(raster, x='lon', y='lat', max_distance=2.5).data

    dask_raster = raster.copy()
    dask_raster.data = da.from_array(data, chunks=(4, 2))
    result = func(dask_raster, x='lon', y='lat', max_distance=2.5)

    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(
        result.values, expected, equal_nan=True, rtol=1e-5)


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_bounded_dask_does_not_mutate_input(func):
    """Bounded dask must not rebind or rechunk the caller's .data (issue #2908).

    When the halo is deeper than a chunk, the bounded path folds that axis into
    a single chunk. The fold used to be assigned back to ``raster.data``, which
    rebound the caller's DataArray (the same mutation issue #2847 guards
    against) and left map_overlap reading a stale, un-folded array. Sizing the
    halo at 3 px against width-2 column chunks forces the column-axis fold.
    """
    coords = np.array([0, 100, 101, 102, 103, 104, 105, 106], dtype=float)
    data = np.zeros((8, 8), dtype=np.float64)
    data[3, 3] = 1.0
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = coords
    raster['lat'] = coords
    raster.data = da.from_array(data, chunks=(4, 2))

    data_before = raster.data
    chunks_before = raster.data.chunks

    func(raster, x='lon', y='lat', max_distance=2.5).compute()

    assert raster.data is data_before, "bounded dask rebound the input .data"
    assert raster.data.chunks == chunks_before, \
        "bounded dask rechunked the input"


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize("shape_name", ["1xN", "Nx1"])
def test_bounded_dask_single_row_or_col_matches_numpy(func, shape_name):
    """Bounded dask must not crash on 1xN / Nx1 rasters.

    Regression for issue #2809 bug 2: the halo code indexed coords[1] along
    both axes, so a single-row or single-column raster raised IndexError on
    the bounded dask path while numpy handled it fine.
    """
    if shape_name == "1xN":
        data = np.zeros((1, 6), dtype=np.float64)
        data[0, 2] = 1.0
        chunks = (1, 3)
    else:
        data = np.zeros((6, 1), dtype=np.float64)
        data[2, 0] = 1.0
        chunks = (3, 1)

    raster = _backend_raster(data, 'numpy')
    expected = func(raster, x='lon', y='lat', max_distance=2.5).data

    dask_raster = _backend_raster(data, 'numpy')
    dask_raster.data = da.from_array(data, chunks=chunks)
    result = func(dask_raster, x='lon', y='lat', max_distance=2.5)

    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(
        result.values, expected, equal_nan=True, rtol=1e-5)


# --- issue #2854: bounded-dask halo depth larger than an axis length -------


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_bounded_dask_skinny_raster_matches_numpy(func):
    """Bounded dask must not crash when the halo is deeper than an axis.

    Regression for issue #2854: on a skinny raster ``_halo_depth`` can return
    a pixel radius larger than the raster height/width. That depth went
    straight into ``da.map_overlap``, which rejects a depth larger than the
    array along that axis and raised ``ValueError: The overlapping depth ...
    is larger than your array ...``. A valid raster with a finite
    ``max_distance`` should still run and match the numpy backend.
    """
    data = np.zeros((3, 100), dtype=np.float64)
    data[1, 50] = 1.0
    xs = np.linspace(0, 99, 100)
    ys = np.linspace(0, 2, 3)

    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = xs
    raster['lat'] = ys
    expected = func(raster, x='lon', y='lat', max_distance=10).data

    dask_raster = raster.copy()
    dask_raster.data = da.from_array(data, chunks=(3, 100))
    result = func(dask_raster, x='lon', y='lat', max_distance=10)

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


# --- Cat 4: dim-order validation error path -------------------------------
# _process rejects a raster whose dims are not (y, x). The bounded-dask and
# kdtree paths all index xs/ys assuming that order, so a swapped raster must
# raise before any backend dispatch rather than silently transposing.

@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_wrong_dim_order_raises(func):
    data = np.zeros((4, 5), dtype=np.float64)
    data[1, 1] = 1.0
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = np.arange(5, dtype=np.float64)
    raster['lat'] = np.arange(4, dtype=np.float64)[::-1]

    # raster.dims is (lat, lon); passing x='lat', y='lon' makes the expected
    # order (lon, lat), which does not match -> ValueError.
    with pytest.raises(ValueError, match="should be named as coordinates"):
        func(raster, x='lat', y='lon')


# --- Cat 2: all-NaN raster (no targets) across all four backends ----------
# An all-NaN raster contains no finite, non-zero target pixels, so every
# output cell must be NaN. The dask no-target path is pinned above with an
# all-zero raster; this pins the all-NaN input (NaN is not finite, so it is
# never treated as a target) on every backend, including eager numpy/cupy.

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_all_nan_raster_all_nan_output(backend, func):
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    data = np.full((6, 6), np.nan, dtype=np.float64)
    raster = _backend_raster(data, backend)

    result = func(raster, x='lon', y='lat')

    out = result.data
    if da is not None and isinstance(out, da.Array):
        out = out.compute()
    if hasattr(out, 'get'):
        out = out.get()
    assert np.all(np.isnan(out))
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Issue #2812: GREAT_CIRCLE must match a brute-force nearest-target reference.
#
# The GDAL-style line-sweep propagates one nearest-target candidate between
# adjacent pixels, which assumes the nearest-target field is locally
# monotonic. That holds for EUCLIDEAN/MANHATTAN but not for GREAT_CIRCLE,
# which wraps in longitude and converges at the poles. On wide-longitude
# rasters the sweep latched onto a farther target and was off by ~1e6 metres.
# ---------------------------------------------------------------------------

def _brute_force_great_circle(data, lon, lat, process_mode):
    """Exact nearest-target reference for GREAT_CIRCLE on a lat/lon raster.

    process_mode is one of 'proximity', 'allocation', 'direction'. Mirrors the
    semantics of the proximity functions: non-zero finite pixels are targets,
    each output pixel reports the closest target under great-circle distance.
    """
    from xrspatial.proximity import _calc_direction

    h, w = data.shape
    mask = np.isfinite(data) & (data != 0)
    tr, tc = np.where(mask)
    out = np.full((h, w), np.nan, dtype=np.float32)
    if len(tr) == 0:
        return out
    for i in range(h):
        for j in range(w):
            best = np.inf
            best_k = -1
            for k in range(len(tr)):
                d = great_circle_distance(
                    lon[j], lon[tc[k]], lat[i], lat[tr[k]])
                if d < best:
                    best = d
                    best_k = k
            if process_mode == 'proximity':
                out[i, j] = best
            elif process_mode == 'allocation':
                out[i, j] = data[tr[best_k], tc[best_k]]
            else:
                out[i, j] = _calc_direction(
                    lon[j], lon[tc[best_k]], lat[i], lat[tr[best_k]])
    return out


@pytest.fixture
def _wide_longitude_raster_data():
    # Wide longitude span at low latitude: the worst-case family found by a
    # brute-force-vs-line-sweep sweep (off by ~5.9e6 m before the fix).
    width, height = 11, 6
    lon = np.linspace(-127.2, 173.3, width)
    lat = np.linspace(21.2, 19.9, height)
    data = np.zeros((height, width), dtype=np.float64)
    data[2, 1] = 1.0
    data[4, 0] = 2.0
    data[5, 4] = 3.0
    data[5, 6] = 4.0
    return data, lon, lat


def _wide_lon_raster(data, lon, lat, backend):
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = lon
    raster['lat'] = lat
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(3, 6))
    return raster


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize(
    "func, mode",
    [(proximity, 'proximity'),
     (allocation, 'allocation'),
     (direction, 'direction')],
)
def test_great_circle_matches_brute_force(
        backend, func, mode, _wide_longitude_raster_data):
    """GREAT_CIRCLE proximity/allocation/direction must match a brute-force
    nearest-target reference on a wide-longitude raster (issue #2812)."""
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    data, lon, lat = _wide_longitude_raster_data
    expected = _brute_force_great_circle(data, lon, lat, mode)

    raster = _wide_lon_raster(data, lon, lat, backend)
    result = func(raster, x='lon', y='lat', distance_metric='GREAT_CIRCLE')
    # rtol covers float32 rounding on million-metre distances; direction is
    # in degrees so use a small absolute tolerance there.
    if mode == 'direction':
        general_output_checks(raster, result, expected, rtol=1e-4)
    else:
        general_output_checks(raster, result, expected, rtol=1e-5)


def test_great_circle_numpy_off_by_more_than_a_metre_is_fixed(
        _wide_longitude_raster_data):
    """Direct numpy assertion that the old line-sweep error is gone.

    Before the fix this raster was off from brute force by ~5.9e6 metres.
    """
    data, lon, lat = _wide_longitude_raster_data
    expected = _brute_force_great_circle(data, lon, lat, 'proximity')

    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = lon
    raster['lat'] = lat
    result = proximity(
        raster, x='lon', y='lat', distance_metric='GREAT_CIRCLE').data

    assert np.nanmax(np.abs(result - expected)) < 1.0


# --- non-monotonic 1D coordinate rejection (issue #2851) ---


def _nonmonotonic_raster(backend, axis):
    """Build a raster whose `axis` ('lon' or 'lat') is non-monotonic.

    All other inputs are valid so the only reason _process can fail is the
    coordinate check.
    """
    data = np.asarray([[0., 0., 1., 0.],
                       [0., 0., 0., 0.],
                       [2., 0., 0., 0.],
                       [0., 0., 0., 3.]])
    lon = np.array([-20., -10., 0., 10.])
    lat = np.array([20., 10., 0., -10.])
    if axis == 'lon':
        lon = np.array([-20., 0., -10., 10.])  # not sorted
    else:
        lat = np.array([20., 0., 10., -10.])  # not sorted

    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = lon
    raster['lat'] = lat
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(2, 2))
    return raster


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize("axis", ['lon', 'lat'])
def test_nonmonotonic_coords_raise(backend, func, axis):
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("cupy not available")
    if 'dask' in backend and da is None:
        pytest.skip("dask not available")

    raster = _nonmonotonic_raster(backend, axis)
    with pytest.raises(ValueError, match="monotonic"):
        func(raster, x='lon', y='lat')


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_descending_coords_allowed(backend, result_default_proximity):
    """A strictly decreasing axis is monotonic and must be accepted.

    The default test raster already uses a descending lat axis; assert the
    standard backends still produce the reference proximity.
    """
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("cupy not available")
    if 'dask' in backend and da is None:
        pytest.skip("dask not available")

    data = np.asarray([[0., 0., 0., 0., 0., 2.],
                       [0., 0., 1., 0., 0., 0.],
                       [0., np.inf, 3., 0., 0., 0.],
                       [4., 0., 0., 0., np.nan, 0.]])
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = np.linspace(-20, 20, 6)   # ascending
    raster['lat'] = np.linspace(20, -20, 4)   # descending
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(4, 3))

    result = proximity(raster, x='lon', y='lat')
    general_output_checks(raster, result, result_default_proximity)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_single_element_axis_allowed(backend):
    """A length-1 axis has no order to violate and must not be rejected."""
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("cupy not available")
    if 'dask' in backend and da is None:
        pytest.skip("dask not available")

    data = np.asarray([[0., 1., 0., 2.]])
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = np.linspace(0, 30, 4)
    raster['lat'] = np.array([0.])
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(1, 2))

    result = proximity(raster, x='lon', y='lat')
    # no exception; finite proximity at the target columns
    assert result.shape == (1, 4)


# ---------------------------------------------------------------------------
# Regression: unbounded dask fallback must not mutate the caller's input
# (issue #2847). _process_dask used to do raster.data = raster.data.rechunk(),
# which rebound .data on the caller's DataArray for the GREAT_CIRCLE and
# no-scipy fallback paths.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("backend", ['dask+numpy', 'dask+cupy'])
@pytest.mark.parametrize("op", [proximity, allocation, direction])
def test_proximity_dask_great_circle_does_not_mutate_input(backend, op):
    """GREAT_CIRCLE unbounded fallback keeps the input .data untouched."""
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")

    data = np.zeros((8, 10), dtype=np.float64)
    data[2, 3] = 1.0
    data[6, 8] = 2.0
    raster = _backend_raster(data, backend)

    data_before = raster.data
    chunks_before = raster.data.chunks

    op(raster, x='lon', y='lat', distance_metric='GREAT_CIRCLE')

    assert raster.data is data_before, "proximity rebound the input .data"
    assert raster.data.chunks == chunks_before, "proximity rechunked the input"


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("backend", ['dask+numpy', 'dask+cupy'])
@pytest.mark.parametrize("op", [proximity, allocation, direction])
def test_proximity_dask_no_scipy_does_not_mutate_input(backend, op):
    """No-scipy unbounded fallback keeps the input .data untouched."""
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")

    import sys
    prox_mod = sys.modules['xrspatial.proximity']

    data = np.zeros((8, 10), dtype=np.float64)
    data[2, 3] = 1.0
    data[6, 8] = 2.0
    raster = _backend_raster(data, backend)

    data_before = raster.data
    chunks_before = raster.data.chunks

    original_ckdtree = prox_mod.cKDTree
    try:
        prox_mod.cKDTree = None
        op(raster, x='lon', y='lat')
    finally:
        prox_mod.cKDTree = original_ckdtree

    assert raster.data is data_before, "proximity rebound the input .data"
    assert raster.data.chunks == chunks_before, "proximity rechunked the input"


# test_proximity_linesweep_compiled_once (issue #3103) was removed along
# with the line-sweep kernel it pinned: PROXIMITY now runs an exact cKDTree
# query in plain Python (issue #3121), so there is no per-call numba
# recompilation left to guard against.


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_proximity_dask_coord_grid_graph_size():
    # Regression test for issue #3132: the bounded dask path used to build
    # the 2D coordinate grids with da.tile / da.repeat + rechunk, which cost
    # ~100 graph tasks per chunk (and the repeat term grew with raster
    # height). The chunk-aligned broadcast construction stays around ~60
    # tasks per chunk. Guard the per-chunk task budget so the tile/repeat
    # pattern cannot come back.
    n, c = 512, 64
    data = np.zeros((n, n), dtype=np.float64)
    data[5, 5] = 1.0
    data[400, 300] = 2.0
    raster = xr.DataArray(da.from_array(data, chunks=(c, c)), dims=['y', 'x'])
    raster['y'] = np.arange(n, dtype=np.float64)[::-1]
    raster['x'] = np.arange(n, dtype=np.float64)

    result = proximity(raster, max_distance=20)
    n_chunks = (n // c) ** 2
    n_tasks = len(result.data.__dask_graph__())
    assert n_tasks < 80 * n_chunks, (
        f"bounded proximity graph has {n_tasks} tasks for {n_chunks} chunks "
        f"({n_tasks / n_chunks:.1f}/chunk); coordinate-grid construction "
        f"regressed (issue #3132)"
    )

    # The broadcast grids must produce the same values as the numpy backend,
    # including on ragged chunks.
    numpy_raster = xr.DataArray(data, dims=['y', 'x'])
    numpy_raster['y'] = np.arange(n, dtype=np.float64)[::-1]
    numpy_raster['x'] = np.arange(n, dtype=np.float64)
    expected = proximity(numpy_raster, max_distance=20)
    np.testing.assert_allclose(
        result.compute().data, expected.data, equal_nan=True)

    ragged = xr.DataArray(da.from_array(data, chunks=(200, 150)),
                          dims=['y', 'x'])
    ragged['y'] = np.arange(n, dtype=np.float64)[::-1]
    ragged['x'] = np.arange(n, dtype=np.float64)
    ragged_result = proximity(ragged, max_distance=20)
    np.testing.assert_allclose(
        ragged_result.compute().data, expected.data, equal_nan=True)


# ---------------------------------------------------------------------------
# Coverage gaps closed for issue #3139 (test-only; no source change).
# All behaviour below was verified correct on a CUDA host before these tests
# were committed. Nothing here pins a bug.
# ---------------------------------------------------------------------------

# --- Cat 2: integer-dtype rasters across all four backends -----------------
# Every other raster in this file is float64, but integer class grids are the
# canonical allocation input. The bounded dask path pads chunks with
# boundary=np.nan; on an int array that pad casts to a garbage finite integer
# (e.g. INT_MIN), which qualifies as a non-zero "target" value. The only
# thing keeping those phantom pad targets from winning is that the
# coordinate-grid pads are real NaNs, so their distances come out NaN and
# never beat a finite one. Pin the int-raster result to the float64 numpy
# baseline on every backend, bounded (engages the halo) and unbounded.

_INT_RASTER_DATA = np.array([[0, 0, 0, 0, 0, 2],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 3, 0, 0, 0],
                             [4, 0, 0, 0, 0, 0]], dtype=np.int32)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize("max_distance", [np.inf, 10.0])
# int32: the NaN halo pad casts to INT_MIN, a phantom non-zero "target".
# uint8 (the usual land-cover dtype): the same cast lands on a different
# garbage value (0 on most platforms, which is a non-target). Both casts
# must leave the result identical to the float64 baseline.
@pytest.mark.parametrize("dtype", [np.int32, np.uint8])
def test_integer_raster_matches_float_baseline(backend, func, max_distance,
                                               dtype):
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("dask not available")

    float_raster = _backend_raster(
        _INT_RASTER_DATA.astype(np.float64), 'numpy')
    expected = func(
        float_raster, x='lon', y='lat', max_distance=max_distance).data

    raster = _backend_raster(_INT_RASTER_DATA.astype(dtype), backend)
    result = func(raster, x='lon', y='lat', max_distance=max_distance)
    general_output_checks(raster, result, expected, verify_dtype=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_integer_raster_explicit_target_values(backend, func):
    # target_values matching against integer pixels (int == int comparison in
    # _is_target_value / _target_mask / cp.isin) on every backend.
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("dask not available")

    float_raster = _backend_raster(
        _INT_RASTER_DATA.astype(np.float64), 'numpy')
    expected = func(
        float_raster, x='lon', y='lat', target_values=[2, 3]).data

    raster = _backend_raster(_INT_RASTER_DATA.copy(), backend)
    result = func(raster, x='lon', y='lat', target_values=[2, 3])
    general_output_checks(raster, result, expected, verify_dtype=True)


# --- Cat 1: bounded dask+cupy with non-default metrics ---------------------
# _process_dask_cupy (the chunked GPU path taken when max_distance is finite
# and smaller than the corner-to-corner distance) sizes its overlap halo from
# the active distance metric via _halo_depth. The metric-unit handling in that
# sizing is the #2809/#2854 bug family, but those regression tests only cover
# dask+numpy, and every existing finite-max_distance dask+cupy test uses
# EUCLIDEAN. Pin MANHATTAN (max_distance in degrees, corner distance 80) and
# GREAT_CIRCLE (max_distance in metres, corner distance ~6.0e6 m) against the
# numpy baseline, and assert the bounded GPU path is the one that ran.

@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize("metric, max_dist",
                         [('MANHATTAN', 10.0), ('GREAT_CIRCLE', 1.5e6)])
def test_bounded_dask_cupy_nondefault_metric_matches_numpy(
        func, metric, max_dist, _metric_raster_data):
    if not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if da is None:
        pytest.skip("dask not available")

    import sys
    prox_mod = sys.modules['xrspatial.proximity']

    numpy_raster = _backend_raster(_metric_raster_data, 'numpy')
    expected = func(numpy_raster, x='lon', y='lat', max_distance=max_dist,
                    distance_metric=metric).data

    raster = _backend_raster(_metric_raster_data, 'dask+cupy')

    bounded_gpu_calls = []
    original_fn = prox_mod._process_dask_cupy

    def spy(*args, **kwargs):
        bounded_gpu_calls.append(True)
        return original_fn(*args, **kwargs)

    with patch.object(prox_mod, '_process_dask_cupy', spy):
        result = func(raster, x='lon', y='lat', max_distance=max_dist,
                      distance_metric=metric)

    assert bounded_gpu_calls, (
        "expected the bounded dask+cupy path (_process_dask_cupy) to run; "
        "the call fell through to a fallback instead"
    )
    assert isinstance(result.data, da.Array)
    general_output_checks(raster, result, expected)


# --- Cat 3: empty (0-row / 0-col) rasters fail fast -------------------------
# A zero-size axis has no coordinate endpoints, so _process cannot compute
# max_possible_distance and the call fails fast (currently IndexError from
# the endpoint lookup). Pin that it raises rather than returning garbage; a
# future clearer ValueError would also satisfy this contract.

@pytest.mark.parametrize("func", [proximity, allocation, direction])
@pytest.mark.parametrize("shape", [(0, 5), (5, 0)])
def test_empty_raster_raises(func, shape):
    data = np.zeros(shape, dtype=np.float64)
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = np.arange(shape[1], dtype=np.float64)
    raster['lat'] = np.arange(shape[0], dtype=np.float64)[::-1]

    with pytest.raises((ValueError, IndexError)):
        func(raster, x='lon', y='lat')


# ---------------------------------------------------------------------------
# Issue #3121: EUCLIDEAN proximity on numpy missed the true nearest target.
#
# The GDAL-ported 4-pass line-sweep propagates one nearest-target candidate
# between adjacent pixels. On some target layouts the chain never delivers
# the true nearest target and the reported distance is too large (~7%
# relative in the repro below). The numpy backend now answers PROXIMITY for
# EUCLIDEAN/MANHATTAN with an exact cKDTree query (brute force when scipy is
# missing); the bounded dask+numpy chunk function reuses the numpy path, so
# it inherits the same fix.
# ---------------------------------------------------------------------------


def _exact_planar_proximity(data, xs_1d, ys_1d, metric, max_distance=np.inf):
    """Float64 exact nearest-target distance, the issue #3121 ground truth."""
    mask = np.isfinite(data) & (data != 0)
    tr, tc = np.where(mask)
    out = np.full(data.shape, np.nan, dtype=np.float64)
    if len(tr) == 0:
        return out
    ty, tx = ys_1d[tr], xs_1d[tc]
    for i in range(data.shape[0]):
        dy = ty - ys_1d[i]
        for j in range(data.shape[1]):
            dx = tx - xs_1d[j]
            if metric == 'MANHATTAN':
                d = (np.abs(dx) + np.abs(dy)).min()
            else:
                d = np.sqrt(dx * dx + dy * dy).min()
            if d <= max_distance:
                out[i, j] = d
    return out


def _issue_3121_raster(backend):
    data = np.zeros((8, 10))
    for i, j in [(0, 4), (0, 8), (1, 2), (2, 1), (6, 6)]:
        data[i, j] = 1.0
    raster = xr.DataArray(data, dims=['y', 'x'])
    raster['y'] = np.linspace(0, 10, 101)[12:20]   # pitch 0.1
    raster['x'] = np.linspace(0, 10, 103)[28:38]   # pitch ~0.098
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(4, 5))
    return raster


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("max_distance", [np.inf, 1.0])
def test_proximity_true_nearest_target_issue_3121(backend, max_distance):
    """Pixel (3, 4) is 0.28008 from the target at (1, 2); the line-sweep
    reported 0.3, the distance to the target at (0, 4). max_distance=1.0
    drives the bounded dask map_overlap path through the same check."""
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("dask not available")

    raster = _issue_3121_raster(backend)
    result = proximity(raster, max_distance=max_distance)

    out = result.data
    if da is not None and isinstance(out, da.Array):
        out = out.compute()
    if hasattr(out, 'get'):
        out = out.get()

    expected = _exact_planar_proximity(
        _issue_3121_raster('numpy').data,
        raster['x'].data, raster['y'].data, 'EUCLIDEAN', max_distance)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)
    assert result.dtype == np.float32


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
@pytest.mark.parametrize("metric", ['EUCLIDEAN', 'MANHATTAN'])
@pytest.mark.parametrize("max_distance", [np.inf, 3.0])
def test_proximity_random_raster_matches_ground_truth(
        backend, metric, max_distance):
    """~600 random targets on a 101x103 grid: every pixel must report the
    float64 exact nearest-target distance (issue #3121 found 4 pixels off
    by up to 0.02 on this kind of input)."""
    if 'dask' in backend and da is None:
        pytest.skip("dask not available")

    rng = np.random.default_rng(0)
    data = (rng.random((101, 103)) < 0.06).astype(np.float64)
    ys = np.linspace(0, 10, 101)
    xs = np.linspace(0, 10.2, 103)
    raster = xr.DataArray(data, dims=['y', 'x'])
    raster['y'] = ys
    raster['x'] = xs
    if 'dask' in backend:
        raster.data = da.from_array(data, chunks=(32, 40))

    result = proximity(raster, max_distance=max_distance,
                       distance_metric=metric)
    out = result.data
    if da is not None and isinstance(out, da.Array):
        out = out.compute()

    expected = _exact_planar_proximity(data, xs, ys, metric, max_distance)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)


def test_proximity_numpy_no_scipy_fallback_is_exact():
    """Without scipy the numpy backend falls back to the exact brute-force
    kernel, not the line-sweep."""
    import sys
    prox_mod = sys.modules['xrspatial.proximity']

    raster = _issue_3121_raster('numpy')
    original_ckdtree = prox_mod.cKDTree
    try:
        prox_mod.cKDTree = None
        with pytest.warns(UserWarning, match="brute-force"):
            result = proximity(raster)
    finally:
        prox_mod.cKDTree = original_ckdtree

    expected = _exact_planar_proximity(
        raster.data, raster['x'].data, raster['y'].data, 'EUCLIDEAN')
    np.testing.assert_allclose(result.data, expected, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# @supports_dataset: proximity / allocation / direction accept an xr.Dataset
# and process each data variable independently (test-only; no source change).
#
# All three functions are decorated with @supports_dataset and their
# docstrings document the Dataset-in / Dataset-out contract, but every other
# test in this file passes a DataArray. The shared decorator is exercised
# generically in test_dataset_support.py, but that file never lists
# proximity/allocation/direction, so the way these functions interact with it
# (per-variable _process dispatch, attrs/coords round-trip, and the
# result.name = None reset each function applies) is otherwise unpinned.
# Behaviour was verified correct on numpy and dask before these were added.
# ---------------------------------------------------------------------------

def _dataset_two_vars(backend):
    """Two-variable Dataset on the requested backend, shared lat/lon coords."""
    data_a = np.zeros((6, 6), dtype=np.float64)
    data_a[1, 1] = 1.0
    data_a[4, 4] = 2.0
    data_b = np.zeros((6, 6), dtype=np.float64)
    data_b[0, 0] = 3.0
    data_b[5, 2] = 4.0

    da_a = _backend_raster(data_a, backend)
    da_b = _backend_raster(data_b, backend)
    ds = xr.Dataset({'a': da_a, 'b': da_b}, attrs={'source': 'test'})
    return ds


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_dataset_input_processes_each_variable(backend, func):
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("dask not available")

    # Build per-variable expected results from the numpy backend so the
    # comparison array stays a plain ndarray (a cupy expected would force an
    # implicit host conversion in general_output_checks).
    numpy_ds = _dataset_two_vars('numpy')
    expected = {name: func(numpy_ds[name], x='lon', y='lat').data
                for name in ('a', 'b')}

    ds = _dataset_two_vars(backend)
    result = func(ds, x='lon', y='lat')

    # Dataset in -> Dataset out, same variables, attrs preserved.
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'a', 'b'}
    assert result.attrs == ds.attrs

    # Each variable must equal the standalone DataArray call on that variable.
    for name in ('a', 'b'):
        general_output_checks(ds[name], result[name], expected[name],
                              verify_dtype=True)
        # Coordinates round-trip on every variable.
        np.testing.assert_array_equal(
            result[name]['lon'].data, ds[name]['lon'].data)
        np.testing.assert_array_equal(
            result[name]['lat'].data, ds[name]['lat'].data)


# ---------------------------------------------------------------------------
# Issue #3389: the max_distance comparison must use the same precision on the
# CPU brute-force and the CUDA kernel. _distance casts to float32 before the
# CPU compares best_dist <= max_distance, while the kernel used to compare the
# float64 distance. A max_distance landing inside the float32 ulp gap of a
# target's distance then qualified the target on numpy but rejected it on cupy
# (or the reverse).
# ---------------------------------------------------------------------------


def test_max_distance_float32_boundary_gpu_matches_cpu():
    """GREAT_CIRCLE proximity at a float32-ulp max_distance agrees CPU/GPU."""
    if not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    import cupy

    lon = np.linspace(0, 39, 40)
    lat = np.array([0.0])
    data = np.zeros((1, 40), dtype=np.float64)
    data[0, 0] = 1.0
    j = 20
    d64 = great_circle_distance(lon[0], lon[j], lat[0], lat[0])
    # max_distance between the float32-rounded and the true float64 distance.
    md = (float(np.float32(d64)) + d64) / 2.0
    assert float(np.float32(d64)) < md < d64, "needs a real float32 ulp gap"

    kwargs = dict(x='lon', y='lat', distance_metric='GREAT_CIRCLE',
                  max_distance=md)
    numpy_raster = xr.DataArray(
        data, dims=['lat', 'lon'], coords={'lon': lon, 'lat': lat})
    numpy_val = proximity(numpy_raster, **kwargs).data[0, j]

    cupy_raster = xr.DataArray(
        cupy.asarray(data), dims=['lat', 'lon'],
        coords={'lon': lon, 'lat': lat})
    cupy_val = proximity(cupy_raster, **kwargs).data.get()[0, j]

    # Both keep the target: its float32 distance is <= max_distance.
    assert np.isnan(numpy_val) == np.isnan(cupy_val)
    np.testing.assert_allclose(numpy_val, cupy_val, rtol=1e-6)


def test_max_distance_float32_boundary_allocation_gpu_matches_cpu():
    """EUCLIDEAN allocation (brute-force on both CPU and GPU) agrees at the
    float32 ulp boundary."""
    if not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    import cupy

    lon = np.array([0.0, 0.1, 0.7])
    lat = np.array([0.0])
    data = np.zeros((1, 3), dtype=np.float64)
    data[0, 0] = 9.0
    j = 2
    d64 = euclidean_distance(lon[0], lon[j], lat[0], lat[0])
    md = (float(np.float32(d64)) + d64) / 2.0
    assert float(np.float32(d64)) < md < d64, "needs a real float32 ulp gap"

    kwargs = dict(x='lon', y='lat', max_distance=md)
    numpy_raster = xr.DataArray(
        data, dims=['lat', 'lon'], coords={'lon': lon, 'lat': lat})
    numpy_val = allocation(numpy_raster, **kwargs).data[0, j]

    cupy_raster = xr.DataArray(
        cupy.asarray(data), dims=['lat', 'lon'],
        coords={'lon': lon, 'lat': lat})
    cupy_val = allocation(cupy_raster, **kwargs).data.get()[0, j]

    assert np.isnan(numpy_val) == np.isnan(cupy_val)
    np.testing.assert_allclose(numpy_val, cupy_val, rtol=1e-6)


# ---------------------------------------------------------------------------
# Issue #3392: split off from #3389. The cKDTree backends (eager numpy
# PROXIMITY, and the dask paths for every mode) compared the float64 distance
# against max_distance, while the brute-force/CUDA kernels round the distance
# to float32 first. A max_distance landing inside the float32 ulp gap just
# below a target's distance then dropped the target to NaN on the cKDTree
# backends while the brute-force backends kept it. _inclusive_upper_bound now
# widens the bound by half a float32 ulp so the cKDTree decision matches.
# ---------------------------------------------------------------------------


def _ulp_raster(backend, data, lon, lat):
    """Build a (1, 3) raster on *backend* with explicit lon/lat coords.

    create_test_raster lays its own coords, but these tests need the exact
    0.0/0.1/0.7 longitudes that produce a float32 ulp gap, so build the
    backend arrays directly.
    """
    raster = xr.DataArray(
        data.copy(), dims=['lat', 'lon'], coords={'lon': lon, 'lat': lat})
    if backend == 'numpy':
        return raster
    if backend == 'dask+numpy':
        raster.data = da.from_array(data.copy(), chunks=(1, 2))
        return raster
    import cupy
    if backend == 'cupy':
        raster.data = cupy.asarray(data)
        return raster
    raster.data = da.from_array(cupy.asarray(data), chunks=(1, 2))  # dask+cupy
    return raster


def _ulp_value(agg, j):
    out = agg.data
    if da is not None and isinstance(out, da.Array):
        out = out.compute()
    if hasattr(out, 'get'):  # cupy
        out = out.get()
    return out[0, j]


@pytest.mark.parametrize(
    "backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("metric", ['EUCLIDEAN', 'MANHATTAN'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_float32_ulp_max_distance_backends_keep_target(func, metric, backend):
    # A target whose true float64 distance sits one float32-ulp above
    # max_distance, but whose float32-rounded distance is <= max_distance, must
    # be kept identically on every backend. The eager numpy result is the
    # reference; each backend has to agree with it.
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("dask is not installed")

    lon = np.array([0.0, 0.1, 0.7])
    lat = np.array([0.0])
    data = np.zeros((1, 3), dtype=np.float64)
    data[0, 0] = 9.0
    j = 2
    # lat is constant, so the target is one axis away: the distance is just the
    # longitude separation for both EUCLIDEAN (p=2) and MANHATTAN (p=1).
    d64 = abs(float(lon[j] - lon[0]))
    md = (float(np.float32(d64)) + d64) / 2.0
    assert float(np.float32(d64)) < md < d64, "needs a real float32 ulp gap"

    kwargs = dict(x='lon', y='lat', max_distance=md, distance_metric=metric)
    ref = func(_ulp_raster('numpy', data, lon, lat), **kwargs).data[0, j]
    val = _ulp_value(func(_ulp_raster(backend, data, lon, lat), **kwargs), j)

    # The target qualifies (its float32 distance is <= max_distance), so no
    # backend NaNs it out and all return the same value.
    assert not np.isnan(ref)
    assert np.isnan(ref) == np.isnan(val)
    np.testing.assert_allclose(ref, val, rtol=1e-6)


@pytest.mark.parametrize(
    "backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_float32_ulp_max_distance_just_below_excludes(func, backend):
    # The mirror case: when max_distance is below the target's float32-rounded
    # distance, the target is genuinely out of range and every backend must
    # NaN it out. Guards against the bound widening so far it admits targets
    # the brute-force kernels exclude.
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("dask is not installed")

    lon = np.array([0.0, 0.1, 0.7])
    lat = np.array([0.0])
    data = np.zeros((1, 3), dtype=np.float64)
    data[0, 0] = 9.0
    j = 2
    d32 = np.float32(0.7)
    md = float(np.nextafter(d32, np.float32(-np.inf)))  # one float32 ulp below
    assert md < float(d32)

    kwargs = dict(x='lon', y='lat', max_distance=md)
    val = _ulp_value(func(_ulp_raster(backend, data, lon, lat), **kwargs), j)
    assert np.isnan(val)


# ---------------------------------------------------------------------------
# Regression tests for issue #3442: EUCLIDEAN proximity/allocation/direction
# dropped a target sitting exactly at max_distance to NaN on the numpy and
# dask+numpy cKDTree backends, while cupy/dask+cupy (brute force) kept it.
# Most visible at max_distance=0, where a target pixel is distance 0 from
# itself and so must qualify. Root cause: cKDTree squares its (exclusive)
# distance_upper_bound for p=2, and a one-ulp widen of 0 underflows when
# squared, so the bound collapsed back to max_distance.
# ---------------------------------------------------------------------------

def _single_target_raster():
    data = np.zeros((3, 3), dtype=np.float64)
    data[1, 1] = 1.0
    raster = xr.DataArray(data, dims=['y', 'x'])
    raster['y'] = np.arange(3)[::-1].astype(float)
    raster['x'] = np.arange(3).astype(float)
    return raster


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("metric", ['EUCLIDEAN', 'MANHATTAN'])
def test_zero_max_distance_keeps_target_pixel(backend, metric):
    # A target pixel is distance 0 from itself, so max_distance=0 must keep
    # it (not NaN it out) on every backend. proximity -> 0, allocation ->
    # the target value, direction -> 0 (the source cell).
    if has_cuda_and_cupy() is False and 'cupy' in backend:
        pytest.skip("Requires CUDA and CuPy")

    raster = create_test_raster(
        _single_target_raster().data, backend=backend, dims=['y', 'x'],
        attrs={'res': (1.0, 1.0)}, chunks=(2, 2),
    )
    # create_test_raster lays its own coords but keeps the 1.0 at (1, 1).
    kwargs = dict(max_distance=0.0, distance_metric=metric)

    prox = proximity(raster, **kwargs)
    alloc = allocation(raster, **kwargs)
    direc = direction(raster, **kwargs)

    def _get(agg):
        d = agg.data
        if da is not None and isinstance(d, da.Array):
            d = d.compute()
        if has_cuda_and_cupy() and 'cupy' in backend:
            d = d.get()
        return d

    pd, ad, dd = _get(prox), _get(alloc), _get(direc)
    # Target pixel qualifies on every backend.
    assert pd[1, 1] == 0.0
    assert ad[1, 1] == 1.0
    assert dd[1, 1] == 0.0
    # Every non-target pixel is beyond max_distance=0, so it stays NaN.
    off = np.ones((3, 3), dtype=bool)
    off[1, 1] = False
    assert np.all(np.isnan(pd[off]))
    assert np.all(np.isnan(ad[off]))
    assert np.all(np.isnan(dd[off]))


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("func", [proximity, allocation, direction])
def test_zero_max_distance_numpy_matches_dask(func):
    # Regression: numpy/dask EUCLIDEAN used cKDTree and dropped the target at
    # max_distance=0; cupy/dask+cupy used brute force and kept it. Pin the two
    # CPU backends together so they cannot drift again.
    raster = _single_target_raster()
    numpy_result = func(raster, max_distance=0.0)

    dask_raster = raster.copy()
    dask_raster.data = da.from_array(raster.data, chunks=(2, 2))
    dask_result = func(dask_raster, max_distance=0.0)

    np.testing.assert_allclose(
        numpy_result.values, dask_result.values, equal_nan=True,
    )


@pytest.mark.skipif(da is None, reason="dask is not installed")
@pytest.mark.parametrize("metric", ['EUCLIDEAN', 'MANHATTAN'])
def test_exact_max_distance_keeps_boundary_pixel(metric):
    # A pixel whose true nearest-target distance equals max_distance exactly
    # must qualify (inclusive <= semantics), matching the brute-force
    # backends. The cKDTree bound is exclusive; the fix widens it.
    raster = _single_target_raster()
    # nearest-target distance from (1, 0) to the target at (1, 1) is 1.0 for
    # both EUCLIDEAN and MANHATTAN on this unit grid.
    md = 1.0

    numpy_result = proximity(raster, max_distance=md, distance_metric=metric)
    assert numpy_result.values[1, 0] == pytest.approx(md)

    dask_raster = raster.copy()
    dask_raster.data = da.from_array(raster.data, chunks=(2, 2))
    dask_result = proximity(dask_raster, max_distance=md,
                            distance_metric=metric)
    assert dask_result.values[1, 0] == pytest.approx(md)
