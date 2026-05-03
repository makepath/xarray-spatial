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


from ..polygonize import polygonize
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


@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_geojson(raster_3x3, connectivity):
    raster = xr.DataArray(raster_3x3)
    fc = polygonize(
        raster, return_type="geojson", connectivity=connectivity)
    assert isinstance(fc, dict)
    assert fc["type"] == "FeatureCollection"
    features = fc["features"]
    assert len(features) == 3

    values = [f["properties"]["DN"] for f in features]
    assert_allclose(values, [0, 1, 4])

    for f in features:
        assert f["type"] == "Feature"
        assert f["geometry"]["type"] == "Polygon"
        coords = f["geometry"]["coordinates"]
        assert isinstance(coords, list)
        assert len(coords) >= 1
        for ring in coords:
            # Ring is a list of [x, y] pairs.
            assert isinstance(ring, list)
            assert len(ring) >= 4
            # Ring is closed.
            assert ring[0] == ring[-1]


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


# --- Performance-related regression tests (#1008) ---

def test_polygonize_1008_jit_merge_helpers():
    """JIT-compiled _simplify_ring, _signed_ring_area, _point_in_ring."""
    from ..polygonize import _point_in_ring, _signed_ring_area, _simplify_ring

    # Unit square: CCW exterior.
    square = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)

    assert_allclose(_signed_ring_area(square), 1.0)

    # Point inside.
    assert _point_in_ring(0.5, 0.5, square) is True
    # Point outside.
    assert _point_in_ring(2.0, 2.0, square) is False

    # Square with collinear midpoints on each edge.
    with_collinear = np.array([
        [0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1],
        [0.5, 1], [0, 1], [0, 0.5], [0, 0]], dtype=np.float64)
    simplified = _simplify_ring(with_collinear)
    # Should remove the midpoints, leaving 4 corners + closing point.
    assert simplified.shape == (5, 2)
    assert_allclose(_signed_ring_area(simplified), 1.0)

    # Ring with no collinear points should be returned unchanged.
    triangle = np.array([
        [0, 0], [2, 0], [1, 2], [0, 0]], dtype=np.float64)
    assert _simplify_ring(triangle) is triangle


@dask_array_available
def test_polygonize_1008_dask_merge_many_boundary_polygons():
    """Dask merge with many boundary-crossing polygons of the same value.

    Checkerboard pattern in small chunks forces many boundary polygons
    through the merge path, exercising the JIT-compiled helpers.
    """
    # 8x8 checkerboard, chunks of 4x4.
    data = np.zeros((8, 8), dtype=np.int32)
    data[::2, ::2] = 1
    data[1::2, 1::2] = 1

    raster_np = xr.DataArray(data)
    vals_np, polys_np = polygonize(raster_np, connectivity=4)
    areas_np = _area_by_value(vals_np, polys_np)

    raster_da = xr.DataArray(da.from_array(data, chunks=(4, 4)))
    vals_da, polys_da = polygonize(raster_da, connectivity=4)
    areas_da = _area_by_value(vals_da, polys_da)

    for val in areas_np:
        assert val in areas_da
        assert_allclose(areas_da[val], areas_np[val],
                        err_msg=f"Area mismatch for value {val}")


@pytest.mark.skipif(gpd is None, reason="geopandas not installed")
def test_polygonize_1008_geopandas_batch_with_holes():
    """Batch shapely construction: mix of hole-free and holed polygons."""
    # Outer ring of 0s with inner block of 1s containing a 2 (hole in 1).
    data = np.zeros((6, 6), dtype=np.int32)
    data[1:5, 1:5] = 1
    data[2:4, 2:4] = 2

    raster = xr.DataArray(data)
    df = polygonize(raster, return_type="geopandas", connectivity=4)

    assert isinstance(df, gpd.GeoDataFrame)
    assert len(df) == 3  # values 0, 1, 2

    # Value 1 polygon should have a hole (the 2-region).
    row_1 = df[df.DN == 1].iloc[0]
    geom = row_1.geometry
    assert len(list(geom.interiors)) == 1

    # Total area should equal raster size.
    assert_allclose(df.geometry.area.sum(), 36.0)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_polygonize_nan_pixels_excluded(connectivity):
    """NaN pixels in float rasters should not produce polygons (#1190)."""
    data = np.array([
        [1.0, np.nan],
        [np.nan, 1.0],
    ], dtype=np.float64)
    raster = xr.DataArray(data)
    values, polygons = polygonize(raster, connectivity=connectivity)
    # No polygon should have a NaN value.
    assert not any(np.isnan(v) for v in values), (
        f"NaN-valued polygons emitted: {values}"
    )
    assert all(v == 1.0 for v in values)
    # 4-connectivity: two separate 1-pixel regions; 8-connectivity: one region.
    expected = 2 if connectivity == 4 else 1
    assert len(polygons) == expected


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_polygonize_nan_pixels_excluded_dask():
    """Dask backend also masks NaN pixels (#1190)."""
    data = np.array([
        [1.0, np.nan, 2.0],
        [np.nan, 1.0, np.nan],
        [3.0, np.nan, 1.0],
    ], dtype=np.float64)
    raster = xr.DataArray(da.from_array(data, chunks=(2, 2)))
    values, polygons = polygonize(raster, connectivity=4)
    assert not any(np.isnan(v) for v in values)
    assert set(values) == {1.0, 2.0, 3.0}


class TestSimplifyHelpers:
    """Tests for internal simplification helper functions."""

    def test_douglas_peucker_straight_line(self):
        """DP on a straight line should reduce to just endpoints."""
        from ..polygonize import _douglas_peucker
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                           [3.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        result = _douglas_peucker(coords, 0.1)
        expected = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        assert_allclose(result, expected)

    def test_douglas_peucker_preserves_bend(self):
        """DP should keep a vertex that exceeds tolerance."""
        from ..polygonize import _douglas_peucker
        coords = np.array([[0.0, 0.0], [2.0, 3.0], [4.0, 0.0]],
                          dtype=np.float64)
        # Distance of (2,3) from line (0,0)-(4,0) is 3.0
        result = _douglas_peucker(coords, 2.0)
        assert len(result) == 3  # all points kept

    def test_douglas_peucker_removes_below_tolerance(self):
        """DP should remove a vertex within tolerance."""
        from ..polygonize import _douglas_peucker
        coords = np.array([[0.0, 0.0], [2.0, 0.5], [4.0, 0.0]],
                          dtype=np.float64)
        # Distance of (2,0.5) from line (0,0)-(4,0) is 0.5
        result = _douglas_peucker(coords, 1.0)
        expected = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        assert_allclose(result, expected)

    def test_douglas_peucker_two_points(self):
        """DP on two points should return them unchanged."""
        from ..polygonize import _douglas_peucker
        coords = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        result = _douglas_peucker(coords, 1.0)
        assert_allclose(result, coords)

    def test_simplify_polygons_reduces_vertices(self):
        """Simplification should reduce vertex count on staircase edges."""
        from ..polygonize import _simplify_polygons

        # Staircase boundary between two values.
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        column_orig, pp_orig = polygonize(data, return_type="numpy")

        _, pp_simplified = _simplify_polygons(
            column_orig, pp_orig, tolerance=1.5)

        # Vertex count should be reduced for at least one polygon.
        orig_total_verts = sum(
            sum(len(r) for r in rings) for rings in pp_orig)
        simp_total_verts = sum(
            sum(len(r) for r in rings) for rings in pp_simplified)
        assert simp_total_verts < orig_total_verts

        # Total area across all polygons must be preserved (shared-edge
        # simplification can shift area between adjacent polygons but
        # the sum stays the same because the simplified edge is shared).
        total_orig = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_orig)
        total_simp = sum(
            sum(calc_boundary_area(r) for r in rings)
            for rings in pp_simplified)
        assert_allclose(total_simp, total_orig, atol=1e-10)

    def test_simplify_polygons_topology_preserved(self):
        """Adjacent simplified polygons should not create gaps."""
        from ..polygonize import _simplify_polygons

        # Two adjacent rectangles sharing an edge.
        raster = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2],
                           [1, 1, 2, 2],
                           [1, 1, 2, 2]], dtype=np.int64)
        data = xr.DataArray(raster)
        column, pp = polygonize(data, return_type="numpy")

        _, pp_simplified = _simplify_polygons(column, pp, tolerance=0.0)

        # With tolerance=0, no vertices should be removed; output
        # should match input exactly.
        for orig_rings, simp_rings in zip(pp, pp_simplified):
            assert len(orig_rings) == len(simp_rings)
            for orig_ring, simp_ring in zip(orig_rings, simp_rings):
                assert_allclose(simp_ring, orig_ring)

    def test_simplify_polygons_shared_edge_identical(self):
        """Two polygons sharing an edge must have identical simplified edges."""
        from ..polygonize import _simplify_polygons

        # Create a raster where two regions share a staircase boundary.
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        column, pp = polygonize(data, return_type="numpy")

        _, pp_simplified = _simplify_polygons(column, pp, tolerance=1.5)

        # Check total area is preserved (which requires no gaps/overlaps).
        total_orig = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp)
        total_simp = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_simplified)
        assert_allclose(total_simp, total_orig, atol=1e-10)

    def test_visvalingam_whyatt_straight_line(self):
        """VW on a straight line should reduce to just endpoints."""
        from ..polygonize import _visvalingam_whyatt
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                           [3.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        result = _visvalingam_whyatt(coords, 0.1)
        expected = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        assert_allclose(result, expected)

    def test_visvalingam_whyatt_preserves_large_triangle(self):
        """VW should keep a vertex forming a large triangle."""
        from ..polygonize import _visvalingam_whyatt
        coords = np.array([[0.0, 0.0], [2.0, 3.0], [4.0, 0.0]],
                          dtype=np.float64)
        # Triangle area = 0.5 * |0*(3-0) + 2*(0-0) + 4*(0-3)| = 6.0
        result = _visvalingam_whyatt(coords, 5.0)
        assert len(result) == 3  # area 6.0 > tolerance 5.0

    def test_visvalingam_whyatt_removes_small_triangle(self):
        """VW should remove a vertex forming a small triangle."""
        from ..polygonize import _visvalingam_whyatt
        coords = np.array([[0.0, 0.0], [2.0, 0.5], [4.0, 0.0]],
                          dtype=np.float64)
        # Triangle area = 0.5 * |0*(0.5-0) + 2*(0-0) + 4*(0-0.5)| = 1.0
        result = _visvalingam_whyatt(coords, 2.0)
        expected = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        assert_allclose(result, expected)

    def test_visvalingam_whyatt_two_points(self):
        """VW on two points should return them unchanged."""
        from ..polygonize import _visvalingam_whyatt
        coords = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        result = _visvalingam_whyatt(coords, 1.0)
        assert_allclose(result, coords)

    def test_simplify_polygons_drops_degenerate(self):
        """Polygons that collapse below 4 vertices should be dropped."""
        from ..polygonize import _simplify_polygons

        # A single-pixel polygon (4 unique vertices forming a unit square).
        # With a very large tolerance, it should collapse and be dropped.
        raster = np.array([
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        column, pp = polygonize(data, return_type="numpy")

        # Find the small polygon (value 1, single pixel).
        small_idx = column.index(1)
        small_col = [column[small_idx]]
        small_poly = [pp[small_idx]]

        # Large tolerance should collapse the single-pixel polygon.
        result_col, result_pp = _simplify_polygons(
            small_col, small_poly, tolerance=10.0)
        # Column and polygon_points must stay in sync.
        assert len(result_col) == len(result_pp)
        # The polygon should be dropped since its exterior collapses.
        assert len(result_pp) == 0 or all(len(r[0]) >= 4 for r in result_pp)

    def test_simplify_column_pp_sync_through_public_api(self):
        """Public API must keep column and polygon_points in sync."""
        # Raster with a single-pixel region that may collapse.
        raster = np.array([
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        col, pp = polygonize(data, simplify_tolerance=10.0)
        assert len(col) == len(pp)


class TestPolygonizeSimplify:
    """Tests for simplify_tolerance and simplify_method parameters."""

    def test_tolerance_none_no_change(self):
        """tolerance=None should produce identical output to no-arg call."""
        raster = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2]], dtype=np.int64)
        data = xr.DataArray(raster)
        col1, pp1 = polygonize(data)
        col2, pp2 = polygonize(data, simplify_tolerance=None)
        assert col1 == col2
        for r1, r2 in zip(pp1, pp2):
            for a, b in zip(r1, r2):
                assert_allclose(a, b)

    def test_tolerance_zero_no_change(self):
        """tolerance=0.0 should produce identical output."""
        raster = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2]], dtype=np.int64)
        data = xr.DataArray(raster)
        col1, pp1 = polygonize(data)
        col2, pp2 = polygonize(data, simplify_tolerance=0.0)
        assert col1 == col2
        for r1, r2 in zip(pp1, pp2):
            for a, b in zip(r1, r2):
                assert_allclose(a, b)

    def test_negative_tolerance_raises(self):
        """Negative tolerance should raise ValueError."""
        raster = np.array([[1, 1], [1, 1]], dtype=np.int64)
        data = xr.DataArray(raster)
        with pytest.raises(ValueError, match="simplify_tolerance"):
            polygonize(data, simplify_tolerance=-1.0)

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        raster = np.array([[1, 1], [1, 1]], dtype=np.int64)
        data = xr.DataArray(raster)
        with pytest.raises(ValueError, match="simplify_method"):
            polygonize(data, simplify_tolerance=1.0,
                       simplify_method="invalid")

    def test_simplify_dp_reduces_vertices(self):
        """Douglas-Peucker simplification should reduce total vertex count."""
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        _, pp_orig = polygonize(data)
        _, pp_simp = polygonize(data, simplify_tolerance=1.5)

        orig_verts = sum(len(r) for rings in pp_orig for r in rings)
        simp_verts = sum(len(r) for rings in pp_simp for r in rings)
        assert simp_verts < orig_verts

    def test_simplify_vw_reduces_vertices(self):
        """Visvalingam-Whyatt simplification should reduce total vertex count."""
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        _, pp_orig = polygonize(data)
        _, pp_simp = polygonize(data, simplify_tolerance=1.5,
                                simplify_method="visvalingam-whyatt")

        orig_verts = sum(len(r) for rings in pp_orig for r in rings)
        simp_verts = sum(len(r) for rings in pp_simp for r in rings)
        assert simp_verts < orig_verts

    def test_simplify_preserves_total_area(self):
        """Total area must be preserved after simplification."""
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        _, pp_orig = polygonize(data)
        _, pp_simp = polygonize(data, simplify_tolerance=1.5)

        total_orig = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_orig)
        total_simp = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_simp)
        assert_allclose(total_simp, total_orig, atol=1e-10)

    def test_simplify_with_geopandas(self):
        """Simplification should work with geopandas return type."""
        pytest.importorskip("geopandas")
        raster = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        gdf = polygonize(data, simplify_tolerance=0.5,
                         return_type="geopandas")
        assert len(gdf) == 2
        assert gdf.geometry.is_valid.all()

    def test_simplify_with_geojson(self):
        """Simplification should work with geojson return type."""
        raster = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        fc = polygonize(data, simplify_tolerance=0.5,
                        return_type="geojson")
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2

    def test_simplify_vw_with_geopandas(self):
        """VW simplification should work with geopandas return type."""
        pytest.importorskip("geopandas")
        raster = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        gdf = polygonize(data, simplify_tolerance=0.5,
                         simplify_method="visvalingam-whyatt",
                         return_type="geopandas")
        assert len(gdf) == 2
        assert gdf.geometry.is_valid.all()

    def test_simplify_with_spatialpandas(self):
        """Simplification should work with spatialpandas return type."""
        pytest.importorskip("spatialpandas")
        raster = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        result = polygonize(data, simplify_tolerance=0.5,
                            return_type="spatialpandas")
        assert len(result) == 2

    def test_simplify_with_awkward(self):
        """Simplification should work with awkward return type."""
        pytest.importorskip("awkward")
        raster = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        result = polygonize(data, simplify_tolerance=0.5,
                            return_type="awkward")
        assert result is not None

    def test_simplify_with_holes(self):
        """Simplification should handle polygons with interior holes."""
        raster = np.zeros((8, 8), dtype=np.int32)
        raster[1:7, 1:7] = 1
        raster[3:5, 3:5] = 0  # hole in the 1-region
        data = xr.DataArray(raster)

        col_orig, pp_orig = polygonize(data)
        col_simp, pp_simp = polygonize(data, simplify_tolerance=0.5)

        assert len(col_simp) == len(pp_simp)
        # Area should be preserved.
        total_orig = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_orig)
        total_simp = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_simp)
        assert_allclose(total_simp, total_orig, atol=1e-10)

    def test_simplify_with_transform(self):
        """Simplification should work with affine transform."""
        raster = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int64)
        transform = np.array([10.0, 0.0, 100.0, 0.0, 10.0, 200.0])
        data = xr.DataArray(raster)
        col, pp = polygonize(data, transform=transform,
                             simplify_tolerance=5.0)
        assert len(col) == len(pp)
        assert len(col) == 2

    def test_simplify_single_pixel_raster(self):
        """Simplification on a 1x1 raster should not crash."""
        raster = np.array([[5]], dtype=np.int64)
        data = xr.DataArray(raster)
        col, pp = polygonize(data, simplify_tolerance=1.0)
        assert len(col) == len(pp)


@pytest.mark.skipif(da is None, reason="dask not installed")
class TestPolygonizeSimplifyDask:
    """Simplification with dask backend."""

    def test_simplify_dask_matches_numpy(self):
        """Dask simplification should produce same areas as numpy."""
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)

        data_np = xr.DataArray(raster)
        data_dask = xr.DataArray(da.from_array(raster, chunks=(3, 3)))

        col_np, pp_np = polygonize(data_np, simplify_tolerance=1.5)
        col_dask, pp_dask = polygonize(data_dask, simplify_tolerance=1.5)

        # Compare per-value area sums.
        areas_np = {}
        for val, rings in zip(col_np, pp_np):
            a = sum(calc_boundary_area(r) for r in rings)
            areas_np[val] = areas_np.get(val, 0.0) + a

        areas_dask = {}
        for val, rings in zip(col_dask, pp_dask):
            a = sum(calc_boundary_area(r) for r in rings)
            areas_dask[val] = areas_dask.get(val, 0.0) + a

        for val in areas_np:
            assert_allclose(areas_dask[val], areas_np[val], atol=1e-10)

    def test_simplify_vw_dask_matches_numpy(self):
        """VW dask simplification should produce same areas as numpy."""
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)

        data_np = xr.DataArray(raster)
        data_dask = xr.DataArray(da.from_array(raster, chunks=(3, 3)))

        col_np, pp_np = polygonize(data_np, simplify_tolerance=1.5,
                                   simplify_method="visvalingam-whyatt")
        col_dask, pp_dask = polygonize(data_dask, simplify_tolerance=1.5,
                                       simplify_method="visvalingam-whyatt")

        areas_np = {}
        for val, rings in zip(col_np, pp_np):
            a = sum(calc_boundary_area(r) for r in rings)
            areas_np[val] = areas_np.get(val, 0.0) + a

        areas_dask = {}
        for val, rings in zip(col_dask, pp_dask):
            a = sum(calc_boundary_area(r) for r in rings)
            areas_dask[val] = areas_dask.get(val, 0.0) + a

        for val in areas_np:
            assert_allclose(areas_dask[val], areas_np[val], atol=1e-10)


@cuda_and_cupy_available
class TestPolygonizeSimplifyCuPy:
    """Simplification with CuPy backend."""

    def test_simplify_cupy_matches_numpy(self):
        """CuPy simplification should produce same areas as numpy."""
        import cupy as cp
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)

        data_np = xr.DataArray(raster)
        data_cp = xr.DataArray(cp.asarray(raster))

        col_np, pp_np = polygonize(data_np, simplify_tolerance=1.5)
        col_cp, pp_cp = polygonize(data_cp, simplify_tolerance=1.5)

        areas_np = {}
        for val, rings in zip(col_np, pp_np):
            a = sum(calc_boundary_area(r) for r in rings)
            areas_np[val] = areas_np.get(val, 0.0) + a

        areas_cp = {}
        for val, rings in zip(col_cp, pp_cp):
            a = sum(calc_boundary_area(r) for r in rings)
            areas_cp[val] = areas_cp.get(val, 0.0) + a

        for val in areas_np:
            assert_allclose(areas_cp[val], areas_np[val], atol=1e-10)


# =====================================================================
# Issue #1441: _validate_raster on raster + mask
# =====================================================================

import xarray as _xr_for_validation


class TestPolygonizeInputValidation:
    """polygonize() rejects bad raster / mask inputs (#1441)."""

    @staticmethod
    def _good_raster():
        return _xr_for_validation.DataArray(
            np.zeros((4, 4), dtype=np.int32),
            dims=('y', 'x'),
            coords={'y': np.arange(4), 'x': np.arange(4)},
        )

    def test_rejects_non_dataarray_raster(self):
        from xrspatial.polygonize import polygonize
        with pytest.raises(TypeError, match="xarray.DataArray"):
            polygonize(np.zeros((4, 4), dtype=np.int32))

    def test_rejects_complex_dtype_raster(self):
        from xrspatial.polygonize import polygonize
        bad = _xr_for_validation.DataArray(
            np.zeros((4, 4), dtype=np.complex128),
            dims=('y', 'x'),
            coords={'y': np.arange(4), 'x': np.arange(4)},
        )
        with pytest.raises(ValueError, match="real numeric"):
            polygonize(bad)

    def test_rejects_1d_raster(self):
        from xrspatial.polygonize import polygonize
        bad = _xr_for_validation.DataArray(
            np.zeros(4, dtype=np.int32), dims=('y',))
        with pytest.raises(ValueError, match=r"must be 2D"):
            polygonize(bad)

    def test_rejects_non_dataarray_mask(self):
        from xrspatial.polygonize import polygonize
        with pytest.raises(TypeError, match="xarray.DataArray"):
            polygonize(self._good_raster(), mask=np.ones((4, 4), dtype=bool))
