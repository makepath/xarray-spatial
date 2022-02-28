import itertools

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

try:
    import awkward as ak
except ImportError:
    ak = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import spatialpandas as sp
except ImportError:
    sp = None


from ..experimental.polygonize import polygonize


def assert_polygon_valid_and_get_area(polygon):
    # Assert that the polygon is valid and returns the signed area.
    assert isinstance(polygon, list)
    assert len(polygon) >= 1
    area = 0.0
    for i, boundary in enumerate(polygon):
        assert isinstance(boundary, np.ndarray)
        assert boundary.dtype == np.float64
        assert boundary.ndim == 2
        assert boundary.shape[0] > 3
        assert boundary.shape[1] == 2
        assert np.array_equal(boundary[0], boundary[-1])
        boundary_area = calc_boundary_area(boundary)
        if i == 0:
            assert boundary_area > 0.0
        else:
            assert boundary_area < 0.0
        area += boundary_area
    return area


def calc_boundary_area(boundary):
    # Shoelace formula (sum of cross products) for area of simple polygon
    # where first and last points are identical.  Positive area for points
    # ordered anticlockwise.
    x = boundary[:, 0]
    y = boundary[:, 1]
    return 0.5*(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


@pytest.fixture
def raster_2x2(dtype):
    return np.asarray([[0, 1], [1, 0]], dtype=dtype)


@pytest.fixture
def raster_3x3(dtype):
    return np.asarray([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=dtype)


@pytest.fixture
def raster_big_with_mask(dtype):
    shape = (40, 50)
    # The combination of random number seeds here gives a number of polygons
    # containing holes.
    rng = np.random.default_rng(28403)
    if np.issubdtype(dtype, np.integer):
        raster = rng.integers(low=0, high=2, size=shape, dtype=dtype)
    else:
        raster = rng.integers(low=0, high=2, size=shape).astype(dtype)
    rng = np.random.default_rng(384182)
    mask = rng.uniform(0, 1, size=shape) < 0.9
    return raster, mask


# Simple test with different results for both connectivities.
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_2x2(raster_2x2, connectivity):
    raster = xr.DataArray(raster_2x2)
    values, polygons = polygonize(
        raster, return_type="numpy", connectivity=connectivity)
    assert len(values) == len(polygons)
    areas = list(map(assert_polygon_valid_and_get_area, polygons))
    if connectivity == 4:
        assert_allclose(values, [0, 1, 1, 0])
        assert_allclose(areas, [1, 1, 1, 1])
    else:
        assert_allclose(values, [0, 1])
        assert_allclose(areas, [2, 2])
    assert_allclose(sum(areas), raster.size)


# Simple test with hole, using many different dtypes.
# Identical results for both connectivities.
@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_3x3(raster_3x3, connectivity):
    raster = xr.DataArray(raster_3x3)
    values, polygons = polygonize(
        raster, return_type="numpy", connectivity=connectivity)
    assert len(values) == len(polygons)
    areas = list(map(assert_polygon_valid_and_get_area, polygons))
    assert_allclose(values, [0, 1, 4])
    assert_allclose(areas, [7, 1, 1])
    assert_allclose(sum(areas), raster.size)


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_big_masked(raster_big_with_mask, connectivity):
    raster, mask = raster_big_with_mask
    raster = xr.DataArray(raster)
    mask = xr.DataArray(mask)
    values, polygons = polygonize(
        raster, mask=mask, return_type="numpy", connectivity=connectivity)
    assert len(values) == len(polygons)
    areas = list(map(assert_polygon_valid_and_get_area, polygons))
    areas = np.asarray(areas)
    values = np.asarray(values)
    values0 = values == 0
    values1 = values == 1
    if connectivity == 4:
        assert_allclose(np.sum(values0), 170)  # Number of polygons
        assert_allclose(np.sum(values1), 184)
    else:
        assert_allclose(np.sum(values0), 23)  # Number of polygons
        assert_allclose(np.sum(values1), 30)
    assert_allclose(np.sum(areas[values0]), 922)
    assert_allclose(np.sum(areas[values1]), 869)
    assert_allclose(sum(areas), mask.sum())


@pytest.mark.parametrize("shape", [(0, 0), (0, 1), (1, 0)])
def test_polygonize_too_small(shape):
    raster = np.full(shape, 1)
    raster = xr.DataArray(raster)
    msg = r"Raster array must be 2D with a shape of at least \(1, 1\)"
    with pytest.raises(ValueError, match=msg):
        _ = polygonize(raster)


@pytest.mark.skipif(ak is None, reason="awkward not installed")
@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_awkward(raster_3x3, connectivity):
    raster = xr.DataArray(raster_3x3)
    values, ak_array = polygonize(
        raster, return_type="awkward", connectivity=connectivity)
    assert_allclose(values, [0, 1, 4])
    assert isinstance(ak_array, ak.Array)


@pytest.mark.skipif(gpd is None, reason="geopandas not installed")
@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_geopandas(raster_3x3, connectivity):
    raster = xr.DataArray(raster_3x3)
    df = polygonize(
        raster, return_type="geopandas", connectivity=connectivity)
    assert isinstance(df, gpd.GeoDataFrame)
    assert_allclose(df.DN, [0, 1, 4])
    assert isinstance(df.geometry, gpd.GeoSeries)


@pytest.mark.skipif(sp is None, reason="spatialpandas not installed")
@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_spatialpandas(raster_3x3, connectivity):
    raster = xr.DataArray(raster_3x3)
    df = polygonize(
        raster, return_type="spatialpandas", connectivity=connectivity)
    assert isinstance(df, sp.GeoDataFrame)
    assert_allclose(df.DN, [0, 1, 4])
    assert isinstance(df.geometry, sp.GeoSeries)


@pytest.mark.parametrize("dtype", [np.uint64])
def test_polygonize_invalid_return_type(raster_3x3):
    raster = xr.DataArray(raster_3x3)
    return_type = "qwerty"
    msg = f"Invalid return_type '{return_type}'"
    with pytest.raises(ValueError, match=msg):
        polygonize(raster, return_type=return_type)


@pytest.mark.parametrize("transform", [
    (1, 0, 0, 0, 1, 0),
    (1.2, -0.3, 0.2, 1.4, 0.7, 0.1)])
@pytest.mark.parametrize("dtype", [np.int32])
def test_polygonize_transform(raster_3x3, transform):
    raster = xr.DataArray(raster_3x3)
    _, original = polygonize(raster)
    _, transformed = polygonize(raster, transform=transform)

    # Flatten list of lists.
    original = list(itertools.chain.from_iterable(original))
    transformed = list(itertools.chain.from_iterable(transformed))

    for o, t in zip(original, transformed):
        x = transform[0]*o[:, 0] + transform[1]*o[:, 1] + transform[2]
        y = transform[3]*o[:, 0] + transform[4]*o[:, 1] + transform[5]
        assert_allclose(x, t[:, 0])
        assert_allclose(y, t[:, 1])
