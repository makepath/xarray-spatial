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
    import dask.array as da
except ImportError:
    da = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import spatialpandas as sp
except ImportError:
    sp = None


from ..experimental.polygonize import polygonize
from .general_checks import cuda_and_cupy_available, dask_array_available


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


# --- Dask backend tests ---

def _area_by_value(values, polygons):
    """Helper: compute total area per pixel value from polygonize output."""
    area_map = {}
    for val, rings in zip(values, polygons):
        area = 0.0
        for i, ring in enumerate(rings):
            area += calc_boundary_area(ring)
        area_map.setdefault(val, 0.0)
        area_map[val] += area
    return area_map


@dask_array_available
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_dask_single_chunk(connectivity):
    """Single-chunk dask array should match numpy exactly."""
    data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
    raster_np = xr.DataArray(data)
    raster_dask = xr.DataArray(da.from_array(data, chunks=data.shape))

    vals_np, polys_np = polygonize(raster_np, connectivity=connectivity)
    vals_da, polys_da = polygonize(raster_dask, connectivity=connectivity)

    assert_allclose(vals_np, vals_da)
    assert len(polys_np) == len(polys_da)
    for pn, pd_ in zip(polys_np, polys_da):
        assert len(pn) == len(pd_)
        for rn, rd in zip(pn, pd_):
            assert_allclose(rn, rd)


@dask_array_available
def test_polygonize_dask_uniform():
    """4x4 all-ones raster in 2x2 chunks -> single polygon, area=16."""
    data = np.ones((4, 4), dtype=np.int32)
    raster = xr.DataArray(da.from_array(data, chunks=(2, 2)))
    values, polygons = polygonize(raster, connectivity=4)
    assert len(values) == 1
    assert values[0] == 1
    area = assert_polygon_valid_and_get_area(polygons[0])
    assert_allclose(area, 16.0)


@dask_array_available
@pytest.mark.parametrize("chunks", [(10, 10), (20, 25), (13, 17), (40, 50)])
def test_polygonize_dask_matches_numpy_big(chunks):
    """Per-value area sums from dask must match numpy for a masked raster."""
    shape = (40, 50)
    rng = np.random.default_rng(28403)
    data = rng.integers(low=0, high=2, size=shape, dtype=np.int64)
    rng = np.random.default_rng(384182)
    mask = (rng.uniform(0, 1, size=shape) < 0.9)

    raster_np = xr.DataArray(data)
    mask_np = xr.DataArray(mask)
    vals_np, polys_np = polygonize(
        raster_np, mask=mask_np, connectivity=4)
    areas_np = _area_by_value(vals_np, polys_np)

    raster_da = xr.DataArray(da.from_array(data, chunks=chunks))
    mask_da = xr.DataArray(da.from_array(mask, chunks=chunks))
    vals_da, polys_da = polygonize(
        raster_da, mask=mask_da, connectivity=4)
    areas_da = _area_by_value(vals_da, polys_da)

    for val in areas_np:
        assert val in areas_da, f"Value {val} missing from dask result"
        assert_allclose(areas_da[val], areas_np[val],
                        err_msg=f"Area mismatch for value {val}")


@dask_array_available
def test_polygonize_dask_holes_across_boundary():
    """Hole spanning a chunk boundary is preserved correctly."""
    # 6x6 raster: outer ring of 1s, inner 4x4 of 0s (= hole in the 1-polygon
    # when the 0-region is masked out).
    data = np.ones((6, 6), dtype=np.int32)
    data[1:5, 1:5] = 2
    # Chunk boundary at row 3 splits the inner block.
    raster = xr.DataArray(da.from_array(data, chunks=(3, 6)))

    values, polygons = polygonize(raster, connectivity=4)

    total_area = sum(
        assert_polygon_valid_and_get_area(p) for p in polygons)
    assert_allclose(total_area, 36.0)

    # The value-2 region should be a single polygon with area 16.
    areas_by_val = _area_by_value(values, polygons)
    assert_allclose(areas_by_val[2], 16.0)
    assert_allclose(areas_by_val[1], 20.0)


@dask_array_available
@pytest.mark.parametrize("transform", [
    (1, 0, 0, 0, 1, 0),
    (1.2, -0.3, 0.2, 1.4, 0.7, 0.1)])
def test_polygonize_dask_with_transform(transform):
    """Transform applied correctly after merge."""
    data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
    raster = xr.DataArray(da.from_array(data, chunks=(2, 2)))
    transform = np.array(transform, dtype=np.float64)

    # Get untransformed dask result.
    _, polys_no_t = polygonize(raster, connectivity=4)
    # Get transformed dask result.
    _, polys_t = polygonize(raster, connectivity=4, transform=transform)

    flat_no_t = list(itertools.chain.from_iterable(polys_no_t))
    flat_t = list(itertools.chain.from_iterable(polys_t))

    for o, t in zip(flat_no_t, flat_t):
        x = transform[0]*o[:, 0] + transform[1]*o[:, 1] + transform[2]
        y = transform[3]*o[:, 0] + transform[4]*o[:, 1] + transform[5]
        assert_allclose(x, t[:, 0])
        assert_allclose(y, t[:, 1])


@dask_array_available
@pytest.mark.skipif(gpd is None, reason="geopandas not installed")
def test_polygonize_dask_geopandas():
    """Dask backend + return_type='geopandas' works."""
    data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
    raster = xr.DataArray(da.from_array(data, chunks=(2, 2)))
    df = polygonize(raster, return_type="geopandas", connectivity=4)
    assert isinstance(df, gpd.GeoDataFrame)
    assert len(df) == 3
    assert set(df.DN) == {0, 1, 4}


# --- CuPy backend tests ---

@cuda_and_cupy_available
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_cupy_matches_numpy(connectivity):
    """CuPy 3x3 raster matches numpy exactly for both connectivities."""
    import cupy

    data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
    raster_np = xr.DataArray(data)
    raster_cp = xr.DataArray(cupy.asarray(data))

    vals_np, polys_np = polygonize(
        raster_np, connectivity=connectivity)
    vals_cp, polys_cp = polygonize(
        raster_cp, connectivity=connectivity)

    assert_allclose(vals_np, vals_cp)
    assert len(polys_np) == len(polys_cp)
    for pn, pc in zip(polys_np, polys_cp):
        assert len(pn) == len(pc)
        for rn, rc in zip(pn, pc):
            assert_allclose(rn, rc)


@cuda_and_cupy_available
def test_polygonize_cupy_masked():
    """CuPy with mask matches numpy."""
    import cupy

    data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
    mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.bool_)

    raster_np = xr.DataArray(data)
    mask_np = xr.DataArray(mask)
    vals_np, polys_np = polygonize(raster_np, mask=mask_np, connectivity=4)

    raster_cp = xr.DataArray(cupy.asarray(data))
    mask_cp = xr.DataArray(cupy.asarray(mask))
    vals_cp, polys_cp = polygonize(raster_cp, mask=mask_cp, connectivity=4)

    assert_allclose(vals_np, vals_cp)
    assert len(polys_np) == len(polys_cp)
    for pn, pc in zip(polys_np, polys_cp):
        assert len(pn) == len(pc)
        for rn, rc in zip(pn, pc):
            assert_allclose(rn, rc)


@cuda_and_cupy_available
@dask_array_available
@pytest.mark.parametrize("chunks", [(2, 2), (3, 3)])
def test_polygonize_dask_cupy_matches_numpy(chunks):
    """Dask+CuPy chunked raster matches numpy per-value areas."""
    import cupy

    data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)

    raster_np = xr.DataArray(data)
    vals_np, polys_np = polygonize(raster_np, connectivity=4)
    areas_np = _area_by_value(vals_np, polys_np)

    dask_data = da.from_array(cupy.asarray(data), chunks=chunks)
    raster_dcp = xr.DataArray(dask_data)
    vals_dcp, polys_dcp = polygonize(raster_dcp, connectivity=4)
    areas_dcp = _area_by_value(vals_dcp, polys_dcp)

    for val in areas_np:
        assert val in areas_dcp, f"Value {val} missing from dask+cupy result"
        assert_allclose(areas_dcp[val], areas_np[val],
                        err_msg=f"Area mismatch for value {val}")
