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


from ..polygonize import _polygonize_dask, polygonize
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


def _assert_polygonize_results_equal(result_a, result_b):
    """Assert two (values, polygons) results are element-wise identical."""
    vals_a, polys_a = result_a
    vals_b, polys_b = result_b
    assert_allclose(vals_a, vals_b)
    assert len(polys_a) == len(polys_b)
    for pa, pb in zip(polys_a, polys_b):
        assert len(pa) == len(pb)
        for ra, rb in zip(pa, pb):
            assert_allclose(ra, rb)


@dask_array_available
@pytest.mark.parametrize("batch_size", [1, 2, 3, 100])
def test_polygonize_dask_batch_size_invariant(batch_size):
    """Batched dask compute gives byte-identical output regardless of batch.

    The raster is split into 3x3 = 9 chunks, so batch_size=1 (one compute
    per chunk, the old serial behaviour) and larger batches all exercise
    the row-major task ordering.  Batching is a performance change only:
    every batch_size, including ones that force multiple batches, must
    produce the exact same values and polygon vertices as batch_size=1.
    """
    shape = (15, 18)
    rng = np.random.default_rng(28403)
    data = rng.integers(low=0, high=3, size=shape, dtype=np.int64)
    mask = rng.uniform(0, 1, size=shape) < 0.9

    raster_da = xr.DataArray(da.from_array(data, chunks=(5, 6)))
    mask_da = xr.DataArray(da.from_array(mask, chunks=(5, 6)))

    # batch_size=1 reproduces the pre-batching serial per-chunk loop.
    result_serial = _polygonize_dask(
        raster_da.data, mask_da.data, connectivity_8=False,
        transform=None, batch_size=1)
    result_batched = _polygonize_dask(
        raster_da.data, mask_da.data, connectivity_8=False,
        transform=None, batch_size=batch_size)

    _assert_polygonize_results_equal(result_serial, result_batched)


@dask_array_available
@pytest.mark.parametrize("batch_size", [1, 4, 100])
def test_polygonize_dask_batched_matches_numpy_areas(batch_size):
    """Batched multi-chunk dask areas match numpy for every batch size."""
    shape = (15, 18)
    rng = np.random.default_rng(28403)
    data = rng.integers(low=0, high=3, size=shape, dtype=np.int64)
    mask = rng.uniform(0, 1, size=shape) < 0.9

    vals_np, polys_np = polygonize(
        xr.DataArray(data), mask=xr.DataArray(mask), connectivity=4)
    areas_np = _area_by_value(vals_np, polys_np)

    raster_da = xr.DataArray(da.from_array(data, chunks=(5, 6)))
    mask_da = xr.DataArray(da.from_array(mask, chunks=(5, 6)))
    vals_da, polys_da = _polygonize_dask(
        raster_da.data, mask_da.data, connectivity_8=False,
        transform=None, batch_size=batch_size)
    areas_da = _area_by_value(vals_da, polys_da)

    assert set(areas_np) == set(areas_da)
    for val in areas_np:
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


@cuda_and_cupy_available
@pytest.mark.parametrize(
    "data",
    [
        # Adjacent near-equal values, the originally reported pattern.
        np.array([
            [1.0, 1.000001, 2.0],
            [1.000001, 1.0, 2.0],
            [3.0, 3.0, 2.0],
        ], dtype=np.float64),
        # Transitivity edge case: three values 1.0, 1.000009, 1.000018 where
        # 1.0 and 1.000018 are NOT pairwise close by ``_is_close`` but a
        # value-only grouping heuristic would chain them transitively.  The
        # CPU spatial CCL keeps them in separate regions when they are not
        # bridged by an adjacent intermediate pixel; the cupy backend must
        # match.
        np.array([
            [1.0, 1.000018],
            [1.000009, 3.0],
        ], dtype=np.float64),
    ],
    ids=["adjacent_near_equal", "transitive_non_adjacent"],
)
def test_polygonize_cupy_float_tolerance_matches_numpy_2151(data):
    """CuPy backend must use the same float tolerance as the numpy/numba
    path for grouping pixels into regions (#2151).

    Before the fix, the cupy backend binned pixels by exact value while
    the numpy/numba backend used ``_is_close`` (atol=1e-8, rtol=1e-5),
    yielding different polygon counts on float rasters with near-equal
    values.
    """
    import cupy

    raster_np = xr.DataArray(data)
    vals_np, polys_np = polygonize(raster_np, connectivity=4)

    raster_cp = xr.DataArray(cupy.asarray(data))
    vals_cp, polys_cp = polygonize(raster_cp, connectivity=4)

    # Both backends should return the same number of polygons.
    assert len(polys_np) == len(polys_cp), (
        f"polygon count differs: numpy={len(polys_np)} cupy={len(polys_cp)}; "
        f"numpy values={vals_np}, cupy values={vals_cp}"
    )
    # Per-value total area must agree across backends.
    areas_np = _area_by_value(vals_np, polys_np)
    areas_cp = _area_by_value(vals_cp, polys_cp)
    assert set(areas_np.keys()) == set(areas_cp.keys()), (
        f"value sets differ: numpy={set(areas_np.keys())} "
        f"cupy={set(areas_cp.keys())}"
    )
    for v, a in areas_np.items():
        assert_allclose(areas_cp[v], a, atol=1e-10)


# --- Dask float tolerance regression tests (#2171) ---

@dask_array_available
def test_polygonize_dask_float_tolerance_repro_2171():
    """The original `[[1.0, 1.000001]]` repro from #2171.

    The numpy path and the single-chunk Dask path return one polygon
    because `_is_close` (atol=1e-8, rtol=1e-5) groups the two near-equal
    values.  Before the fix the (1, 1) chunked Dask run returned two
    polygons because the chunk-stitching step keyed on raw float values.
    """
    data = np.array([[1.0, 1.000001]], dtype=np.float64)

    vals_np, polys_np = polygonize(xr.DataArray(data))
    vals_single, polys_single = polygonize(
        xr.DataArray(da.from_array(data, chunks=data.shape)))
    vals_multi, polys_multi = polygonize(
        xr.DataArray(da.from_array(data, chunks=(1, 1))))

    assert len(polys_np) == 1
    assert len(polys_single) == 1
    assert len(polys_multi) == 1

    # All three backends pick the same representative value.
    assert vals_np == vals_single == vals_multi


@dask_array_available
@pytest.mark.parametrize(
    "data,chunks",
    [
        # Single row, two near-equal cells, one cell per chunk.
        (np.array([[1.0, 1.000001]], dtype=np.float64), (1, 1)),
        # 2x2 with near-equal floats and a clearly different value,
        # each cell in its own chunk.
        (np.array([[1.0, 1.000001], [1.000001, 2.0]], dtype=np.float64),
         (1, 1)),
        # 3x3 with near-equal floats and a distinct value column.
        (np.array([
            [1.0, 1.000001, 2.0],
            [1.000001, 1.0, 2.0],
            [1.0, 1.0, 2.0],
        ], dtype=np.float64), (1, 1)),
        # Larger pattern at a slightly bigger chunk size to exercise
        # multiple boundary segments per chunk.
        (np.array([
            [1.0, 1.000001, 1.0, 2.0],
            [1.000001, 1.0, 1.000001, 2.0],
            [1.0, 1.000001, 1.0, 2.0],
            [1.000001, 1.0, 1.000001, 2.0],
        ], dtype=np.float64), (2, 2)),
    ],
    ids=["1x2_chunks_1x1", "2x2_chunks_1x1", "3x3_chunks_1x1",
         "4x4_chunks_2x2"],
)
def test_polygonize_dask_float_tolerance_matches_numpy_2171(data, chunks):
    """Multi-chunk Dask must match numpy on near-equal float inputs.

    The Dask chunk-stitching step must use the same `_is_close`
    tolerance the numpy/numba path uses inside a single chunk so the
    polygon count and per-value areas are independent of chunking.
    """
    raster_np = xr.DataArray(data)
    raster_da = xr.DataArray(da.from_array(data, chunks=chunks))

    vals_np, polys_np = polygonize(raster_np, connectivity=4)
    vals_da, polys_da = polygonize(raster_da, connectivity=4)

    assert len(polys_np) == len(polys_da), (
        f"polygon count differs: numpy={len(polys_np)} "
        f"dask={len(polys_da)}; numpy values={vals_np}, "
        f"dask values={vals_da}"
    )

    areas_np = _area_by_value(vals_np, polys_np)
    areas_da = _area_by_value(vals_da, polys_da)

    # Per-value areas must agree under the tolerance: dask may report a
    # nearby float (e.g. 1.000001 vs 1.0) as the bucket representative,
    # so match each numpy bucket to the closest dask bucket within
    # _is_close tolerance.
    atol = 1e-8
    rtol = 1e-5
    unmatched_da = dict(areas_da)
    for v_np, a_np in areas_np.items():
        match = None
        for v_da in unmatched_da:
            if abs(v_da - v_np) <= atol + rtol * abs(v_np):
                match = v_da
                break
        assert match is not None, (
            f"numpy value {v_np!r} not found in dask values "
            f"{list(unmatched_da.keys())}"
        )
        assert_allclose(unmatched_da.pop(match), a_np, atol=1e-10)
    assert not unmatched_da, f"extra dask buckets: {unmatched_da}"


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

    @staticmethod
    def _make_zigzag_coords(n):
        return np.array(
            [[float(i), float((i % 3) * 0.5)] for i in range(n)],
            dtype=np.float64,
        )

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

    def test_visvalingam_whyatt_nan_coords(self):
        """VW on all-NaN coords should return all points unchanged."""
        from ..polygonize import _visvalingam_whyatt
        coords = np.array(
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
             [np.nan, np.nan], [np.nan, np.nan]],
            dtype=np.float64)
        result = _visvalingam_whyatt(coords, 0.1)
        assert_allclose(result, coords)

    def test_visvalingam_whyatt_identical_coords(self):
        """VW on all-identical coords should simplify to just endpoints."""
        from ..polygonize import _visvalingam_whyatt
        coords = np.array(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float64)
        result = _visvalingam_whyatt(coords, 0.01)
        expected = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        assert_allclose(result, expected)

    def test_visvalingam_whyatt_heap_correctness(self):
        """Heap implementation must produce identical results to O(n^2) baseline."""
        from numba import njit

        from ..polygonize import _visvalingam_whyatt

        @njit
        def _triangle_area_numba(a, b, c):
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            cx, cy = c[0], c[1]
            return abs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0)

        @njit
        def _visvalingam_whyatt_o2(coords, tolerance):
            """Original O(n^2) implementation for correctness comparison."""
            n = len(coords)
            if n <= 2:
                return coords.copy()
            prev_idx = np.empty(n, dtype=np.int64)
            next_idx = np.empty(n, dtype=np.int64)
            for i in range(n):
                prev_idx[i] = i - 1
                next_idx[i] = i + 1
            prev_idx[0] = -1
            next_idx[n - 1] = -1
            areas = np.full(n, np.inf, dtype=np.float64)
            for i in range(1, n - 1):
                areas[i] = _triangle_area_numba(
                    coords[prev_idx[i]], coords[i], coords[next_idx[i]]
                )
            removed = np.zeros(n, dtype=np.bool_)
            remaining = n
            while remaining > 2:
                min_area = np.inf
                min_idx = -1
                for i in range(1, n - 1):
                    if not removed[i] and areas[i] < min_area:
                        min_area = areas[i]
                        min_idx = i
                if min_idx == -1 or min_area >= tolerance:
                    break
                removed[min_idx] = True
                remaining -= 1
                p = prev_idx[min_idx]
                nx_i = next_idx[min_idx]
                if p >= 0:
                    next_idx[p] = nx_i
                if nx_i >= 0 and nx_i < n:
                    prev_idx[nx_i] = p
                if p > 0 and prev_idx[p] >= 0:
                    new_area = _triangle_area_numba(
                        coords[prev_idx[p]], coords[p], coords[next_idx[p]]
                    )
                    areas[p] = max(new_area, min_area)
                if nx_i >= 0 and nx_i < n - 1 and next_idx[nx_i] >= 0:
                    new_area = _triangle_area_numba(
                        coords[prev_idx[nx_i]], coords[nx_i], coords[next_idx[nx_i]]
                    )
                    areas[nx_i] = max(new_area, min_area)
            count = 0
            for i in range(n):
                if not removed[i]:
                    count += 1
            result = np.empty((count, 2), dtype=np.float64)
            j = 0
            for i in range(n):
                if not removed[i]:
                    result[j, 0] = coords[i, 0]
                    result[j, 1] = coords[i, 1]
                    j += 1
            return result

        def make_coords(n, pattern="zigzag"):
            if pattern == "straight":
                return np.array([[float(i), 0.0] for i in range(n)], dtype=np.float64)
            if pattern == "zigzag":
                return np.array(
                    [[float(i), float((i % 3) * 0.5)] for i in range(n)],
                    dtype=np.float64,
                )
            if pattern == "triangle":
                pts = []
                for i in range(n):
                    angle = 2.0 * np.pi * i / n
                    pts.append([np.cos(angle), np.sin(angle)])
                return np.array(pts, dtype=np.float64)
            return np.array([[float(i), float(i * 0.1)] for i in range(n)], dtype=np.float64)

        sizes = [5, 10, 50, 100, 500]
        patterns = ["straight", "zigzag", "triangle", "linear"]
        tolerances = [0.01, 0.1, 1.0]

        for pattern in patterns:
            for n in sizes:
                for tol in tolerances:
                    coords = make_coords(n, pattern)
                    heap_result = _visvalingam_whyatt(coords, tol)
                    o2_result = _visvalingam_whyatt_o2(coords, tol)
                    assert heap_result.shape == o2_result.shape, (
                        f"Shape mismatch for {pattern} n={n} tol={tol}: "
                        f"heap={heap_result.shape} o2={o2_result.shape}"
                    )
                    assert_allclose(
                        heap_result, o2_result, rtol=1e-10,
                        err_msg=f"Output mismatch for {pattern} n={n} tol={tol}"
                    )

    def test_visvalingam_whyatt_no_regression_small_inputs(self):
        """Heap must not regress performance for small inputs (n < 1000).

        For typical polygon ring sizes, the heap overhead should not make
        the implementation more than 2x slower than the O(n^2) baseline.
        """
        import time

        from numba import njit

        from ..polygonize import _visvalingam_whyatt

        @njit
        def _triangle_area_numba(a, b, c):
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            cx, cy = c[0], c[1]
            return abs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0)

        @njit
        def _visvalingam_whyatt_o2(coords, tolerance):
            n = len(coords)
            if n <= 2:
                return coords.copy()
            prev_idx = np.empty(n, dtype=np.int64)
            next_idx = np.empty(n, dtype=np.int64)
            for i in range(n):
                prev_idx[i] = i - 1
                next_idx[i] = i + 1
            prev_idx[0] = -1
            next_idx[n - 1] = -1
            areas = np.full(n, np.inf, dtype=np.float64)
            for i in range(1, n - 1):
                areas[i] = _triangle_area_numba(
                    coords[prev_idx[i]], coords[i], coords[next_idx[i]]
                )
            removed = np.zeros(n, dtype=np.bool_)
            remaining = n
            while remaining > 2:
                min_area = np.inf
                min_idx = -1
                for i in range(1, n - 1):
                    if not removed[i] and areas[i] < min_area:
                        min_area = areas[i]
                        min_idx = i
                if min_idx == -1 or min_area >= tolerance:
                    break
                removed[min_idx] = True
                remaining -= 1
                p = prev_idx[min_idx]
                nx_i = next_idx[min_idx]
                if p >= 0:
                    next_idx[p] = nx_i
                if nx_i >= 0 and nx_i < n:
                    prev_idx[nx_i] = p
                if p > 0 and prev_idx[p] >= 0:
                    new_area = _triangle_area_numba(
                        coords[prev_idx[p]], coords[p], coords[next_idx[p]]
                    )
                    areas[p] = max(new_area, min_area)
                if nx_i >= 0 and nx_i < n - 1 and next_idx[nx_i] >= 0:
                    new_area = _triangle_area_numba(
                        coords[prev_idx[nx_i]], coords[nx_i], coords[next_idx[nx_i]]
                    )
                    areas[nx_i] = max(new_area, min_area)
            count = 0
            for i in range(n):
                if not removed[i]:
                    count += 1
            result = np.empty((count, 2), dtype=np.float64)
            j = 0
            for i in range(n):
                if not removed[i]:
                    result[j, 0] = coords[i, 0]
                    result[j, 1] = coords[i, 1]
                    j += 1
            return result

        def make_coords(n):
            return self._make_zigzag_coords(n)

        sizes = [50, 100, 200, 500]
        tolerance = 0.1
        iterations = 200

        for n in sizes:
            coords = make_coords(n)
            _visvalingam_whyatt(coords, tolerance)
            _visvalingam_whyatt_o2(coords, tolerance)

            start = time.perf_counter()
            for _ in range(iterations):
                _visvalingam_whyatt(coords, tolerance)
            heap_time = (time.perf_counter() - start) / iterations

            start = time.perf_counter()
            for _ in range(iterations):
                _visvalingam_whyatt_o2(coords, tolerance)
            o2_time = (time.perf_counter() - start) / iterations

            ratio = heap_time / o2_time if o2_time > 0 else 0
            # 200 iterations reduce measurement noise; sub-microsecond times
            # still have variance so allow up to 3x overhead at this scale.
            assert ratio < 3.0, (
                f"Heap is {ratio:.2f}x slower than O(n^2) for n={n}, "
                f"expected < 3x (heap={heap_time:.6f}s, o2={o2_time:.6f}s)"
            )

    def test_visvalingam_whyatt_completes_large_inputs(self):
        """_visvalingam_whyatt must complete on large inputs without hanging.

        Issue #2888: the previous timing-based subquadratic scaling test was
        flaky on loaded CI runners. This simplified smoke test verifies the
        function handles large inputs correctly without timing assertions.

        The O(n log n) complexity is guaranteed by the heap-based
        implementation and validated by correctness tests.
        """
        from ..polygonize import _visvalingam_whyatt

        sizes = [2000, 4000, 8000, 16000, 32000]
        tolerance = 0.1

        for n in sizes:
            coords = self._make_zigzag_coords(n)
            # Warm up call (triggers numba JIT compilation and cache population,
            # discarded; the second call below is the timed/measured one).
            _visvalingam_whyatt(coords, tolerance)
            result = _visvalingam_whyatt(coords, tolerance)
            assert result.shape[0] >= 2, (
                f"Expected at least 2 vertices for n={n}, got {result.shape[0]}"
            )
            assert result.shape[1] == 2
            # Verify meaningful simplification occurred: at minimum the
            # collinear upslope vertices (every 3rd interior point) are
            # removed, so output must be < 90 % of input.
            assert result.shape[0] < n * 0.9, (
                f"Expected < {0.9 * n:.0f} vertices for n={n}, "
                f"got {result.shape[0]} (ratio={result.shape[0]/n:.4f})"
            )


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

    @pytest.mark.parametrize("bad", [
        float("nan"), float("inf"), float("-inf")])
    def test_non_finite_tolerance_raises(self, bad):
        """nan/inf/-inf for simplify_tolerance must raise ValueError.

        Regression for #2575: ``nan`` silently disabled simplification
        (because ``nan > 0`` is False) and ``inf`` collapsed every
        polygon to empty output.  Mirror the atol/rtol validation
        contract: require finite + non-negative.
        """
        raster = np.array([[1, 1], [1, 1]], dtype=np.int64)
        data = xr.DataArray(raster)
        with pytest.raises(ValueError, match="simplify_tolerance"):
            polygonize(data, simplify_tolerance=bad)

    def test_zero_and_small_positive_tolerance_still_work(self):
        """0.0 and small positive tolerances must still be accepted.

        Regression for #2575: guard against an over-eager fix that
        bans non-positive values along with non-finite ones.
        """
        raster = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2]], dtype=np.int64)
        data = xr.DataArray(raster)
        # Both calls should succeed without raising.
        polygonize(data, simplify_tolerance=0.0)
        polygonize(data, simplify_tolerance=1e-9)

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


# =====================================================================
# Issue #2149: GeoDataFrame output drops input CRS
# =====================================================================


@pytest.mark.skipif(gpd is None, reason="geopandas not installed")
class TestPolygonizeCRSPropagation:
    """polygonize(return_type='geopandas') preserves the raster CRS (#2149)."""

    @staticmethod
    def _raster_with_attrs(**attrs):
        data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
        return xr.DataArray(
            data,
            dims=('y', 'x'),
            coords={'y': np.arange(3), 'x': np.arange(3)},
            attrs=attrs,
        )

    def test_crs_attr_epsg_string(self):
        raster = self._raster_with_attrs(crs="EPSG:4326")
        df = polygonize(raster, return_type="geopandas")
        assert df.crs is not None
        assert df.crs.to_epsg() == 4326

    def test_crs_attr_epsg_int(self):
        raster = self._raster_with_attrs(crs=3857)
        df = polygonize(raster, return_type="geopandas")
        assert df.crs is not None
        assert df.crs.to_epsg() == 3857

    def test_crs_wkt_attr(self):
        # Use pyproj if available to get a canonical WKT.
        pyproj = pytest.importorskip("pyproj")
        wkt = pyproj.CRS.from_epsg(4326).to_wkt()
        raster = self._raster_with_attrs(crs_wkt=wkt)
        df = polygonize(raster, return_type="geopandas")
        assert df.crs is not None
        assert df.crs.to_epsg() == 4326

    def test_no_crs_attr_yields_none(self):
        """A raster without CRS info should still produce a GeoDataFrame
        (with crs=None), not crash."""
        data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
        raster = xr.DataArray(data, dims=('y', 'x'))
        df = polygonize(raster, return_type="geopandas")
        assert df.crs is None

    def test_unparseable_crs_does_not_crash(self):
        """An invalid CRS attr should not break the polygonize call."""
        raster = self._raster_with_attrs(crs="not a real crs at all")
        # Should not raise; just yields no CRS on the GeoDataFrame.
        df = polygonize(raster, return_type="geopandas")
        assert isinstance(df, gpd.GeoDataFrame)

    def test_crs_prefers_attrs_over_rio(self):
        """When both attrs['crs'] and rio.crs exist, attrs wins."""
        pytest.importorskip("rioxarray")
        raster = self._raster_with_attrs(crs="EPSG:4326")
        # rio.write_crs stores the CRS on a spatial_ref coord, not in
        # attrs['crs'], so re-setting attrs['crs'] below is what makes
        # the precedence assertion meaningful.
        raster = raster.rio.write_crs("EPSG:3857")
        raster.attrs['crs'] = "EPSG:4326"
        df = polygonize(raster, return_type="geopandas")
        assert df.crs.to_epsg() == 4326

    def test_crs_from_rioxarray_only(self):
        """rioxarray's CRS should be picked up when no attrs are set."""
        pytest.importorskip("rioxarray")
        data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
        raster = xr.DataArray(
            data,
            dims=('y', 'x'),
            coords={'y': np.arange(3), 'x': np.arange(3)},
        )
        raster = raster.rio.write_crs("EPSG:3857")
        # rioxarray adds a spatial_ref coord but typically does not write
        # ``attrs['crs']`` on the DataArray itself.  Force-clear in case.
        raster.attrs.pop('crs', None)
        raster.attrs.pop('crs_wkt', None)
        df = polygonize(raster, return_type="geopandas")
        assert df.crs is not None
        assert df.crs.to_epsg() == 3857

    def test_no_crs_attr_simplify_still_works(self):
        """CRS propagation must not interact badly with simplify."""
        raster = self._raster_with_attrs(crs="EPSG:4326")
        df = polygonize(
            raster, return_type="geopandas", simplify_tolerance=0.5)
        assert df.crs is not None
        assert df.crs.to_epsg() == 4326


# --- transform attr propagation tests (issue #2536) ---
#
# polygonize used to propagate attrs['crs'] but ignore attrs['transform'].
# The result was a GeoDataFrame whose .crs claimed projected space while
# the geometries were in pixel space (silent misalignment).  The fix auto-
# detects attrs['transform'] (or rio.transform()) when no explicit
# transform= is passed, mirroring the CRS auto-detection.

@pytest.mark.skipif(gpd is None, reason="geopandas not installed")
class TestPolygonizeTransformPropagation:
    """polygonize auto-detects attrs['transform'] when not given one (#2536)."""

    # rasterio-ordered transform: 10 m pixels, origin at (1e6, 5e6).
    # _transform_points consumes this as
    #   x = a*col + b*row + c, y = d*col + e*row + f.
    _TRANSFORM = (10.0, 0.0, 1_000_000.0, 0.0, -10.0, 5_000_000.0)

    @staticmethod
    def _raster(**attrs):
        data = np.array([[1, 1, 2], [1, 1, 2], [2, 2, 2]], dtype=np.int32)
        return xr.DataArray(
            data,
            dims=('y', 'x'),
            coords={'y': np.arange(3), 'x': np.arange(3)},
            attrs=attrs,
        )

    def test_transform_attr_auto_detected_numpy(self):
        """attrs['transform'] is applied even when transform= is omitted."""
        raster = self._raster(transform=self._TRANSFORM)
        _, polys = polygonize(raster, return_type='numpy')
        # Polygons should span the CRS origin offset, not pixel space.
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        all_ys = np.concatenate([p[0][:, 1] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6
        assert all_xs.max() <= 1_000_000.0 + 30.0 + 1e-6
        # pixel_height is negative so y decreases from origin.
        assert all_ys.max() <= 5_000_000.0 + 1e-6
        assert all_ys.min() >= 5_000_000.0 - 30.0 - 1e-6

    def test_transform_attr_auto_detected_geopandas(self):
        """The geopandas path applies the auto-detected transform."""
        raster = self._raster(crs='EPSG:3857', transform=self._TRANSFORM)
        df = polygonize(raster, return_type='geopandas')
        assert df.crs is not None and df.crs.to_epsg() == 3857
        bounds = df.geometry.total_bounds  # [minx, miny, maxx, maxy]
        assert bounds[0] >= 1_000_000.0 - 1e-6
        assert bounds[2] <= 1_000_000.0 + 30.0 + 1e-6
        assert bounds[1] >= 5_000_000.0 - 30.0 - 1e-6
        assert bounds[3] <= 5_000_000.0 + 1e-6

    def test_explicit_transform_overrides_attr(self):
        """An explicit transform= wins over attrs['transform']."""
        raster = self._raster(transform=self._TRANSFORM)
        # Identity-ish: pixel space, no offset.  Output must be in 0..3.
        identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        _, polys = polygonize(raster, transform=identity,
                              return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        all_ys = np.concatenate([p[0][:, 1] for p in polys])
        assert all_xs.max() <= 3.0 + 1e-6
        assert all_ys.max() <= 3.0 + 1e-6

    def test_no_transform_info_yields_pixel_coords(self):
        """A raster with no transform info keeps pixel-space coords.

        Build a coordless raster here rather than using ``_raster()``:
        that helper sets integer-index coords (``np.arange(3)``), and
        since #2607 those coords are themselves a transform source
        (resolving to a -0.5 pixel-corner origin), so they would no
        longer exercise the "genuinely no transform info" path.
        """
        data = np.array([[1, 1, 2], [1, 1, 2], [2, 2, 2]], dtype=np.int32)
        raster = xr.DataArray(data, dims=('y', 'x'))  # no coords, no attrs
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        all_ys = np.concatenate([p[0][:, 1] for p in polys])
        assert all_xs.max() <= 3.0 + 1e-6
        assert all_ys.max() <= 3.0 + 1e-6

    def test_no_georef_marker_suppresses_auto_detect(self):
        """attrs['_xrspatial_no_georef']=True opts out of auto-detect even
        when attrs['transform'] is also set (defensive: should not happen
        in practice, but a malformed dict must not silently mis-transform).

        Note: ``xrspatial.geotiff`` never produces a raster with both attrs
        set simultaneously -- see ``xrspatial/geotiff/_attrs.py:782-785``,
        which only writes ``_xrspatial_no_georef=True`` when
        ``has_georef=False`` and in that branch does NOT write
        ``attrs['transform']``.  This test guards against a hand-built or
        future-third-party attrs dict that violates that invariant."""
        raster = self._raster(
            transform=self._TRANSFORM, _xrspatial_no_georef=True)
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.max() <= 3.0 + 1e-6

    def test_transform_attr_list_form_auto_detected(self):
        """attrs['transform'] stored as a list (not a tuple) still works."""
        raster = self._raster(transform=list(self._TRANSFORM))
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6

    def test_transform_attr_invalid_length_falls_back(self):
        """A malformed (wrong-length) attrs['transform'] is silently
        ignored, parallel to how an unparseable attrs['crs'] is dropped
        rather than raised.  The user did not opt in to that transform
        explicitly, so falling back to pixel coords is friendlier than
        breaking the call."""
        raster = self._raster(transform=(1.0, 0.0, 0.0))  # only 3 elements
        _, polys = polygonize(raster, return_type='numpy')
        # Falls back to pixel-space coords.
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.max() <= 3.0 + 1e-6

    def test_explicit_transform_invalid_length_still_raises(self):
        """An EXPLICIT transform= of wrong length must still raise --
        only the attrs path is silent on malformed input."""
        raster = self._raster()
        with pytest.raises(ValueError, match="Incorrect transform length"):
            polygonize(raster, transform=(1.0, 0.0, 0.0), return_type='numpy')

    def test_transform_attr_unparseable_falls_back(self):
        """A non-numeric attrs['transform'] is ignored, not raised."""
        raster = self._raster(transform=("not", "a", "transform"))
        # Falls back to pixel coords; no exception.
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.max() <= 3.0 + 1e-6

    def test_simplify_works_with_auto_detected_transform(self):
        """simplify_tolerance still works after auto-detected transform."""
        # Larger raster to leave room for simplification.
        data = np.zeros((10, 10), dtype=np.int32)
        data[2:8, 2:8] = 1
        raster = xr.DataArray(
            data,
            dims=('y', 'x'),
            attrs={'crs': 'EPSG:3857', 'transform': self._TRANSFORM},
        )
        df = polygonize(
            raster, return_type='geopandas', simplify_tolerance=1.0)
        assert df.crs.to_epsg() == 3857
        # Should land near the projected origin.
        assert df.geometry.total_bounds[0] >= 1_000_000.0 - 1e-6

    @pytest.mark.skipif(da is None, reason="dask not installed")
    def test_transform_attr_auto_detected_dask(self):
        """Dask backend honours auto-detected attrs['transform']."""
        data = np.array([[1, 1, 2], [1, 1, 2], [2, 2, 2]], dtype=np.int32)
        dask_data = da.from_array(data, chunks=(2, 2))
        raster = xr.DataArray(
            dask_data,
            dims=('y', 'x'),
            coords={'y': np.arange(3), 'x': np.arange(3)},
            attrs={'transform': self._TRANSFORM},
        )
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        all_ys = np.concatenate([p[0][:, 1] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6
        assert all_ys.max() <= 5_000_000.0 + 1e-6

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_transform_attr_works_with_both_connectivities(self, connectivity):
        """Auto-detected transform applies for connectivity=4 and 8 alike.

        The transform is applied inside ``_scan`` after region tracing, so
        the connectivity choice should not interact with the transform
        path.  Parametrize to catch any future change that decouples one
        connectivity from the transform call site.
        """
        raster = self._raster(transform=self._TRANSFORM)
        _, polys = polygonize(
            raster, connectivity=connectivity, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        all_ys = np.concatenate([p[0][:, 1] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6
        assert all_xs.max() <= 1_000_000.0 + 30.0 + 1e-6
        assert all_ys.max() <= 5_000_000.0 + 1e-6
        assert all_ys.min() >= 5_000_000.0 - 30.0 - 1e-6

    def test_transform_attr_with_mask(self):
        """Auto-detected transform composes with a mask= argument.

        The mask filters which pixels participate in polygonization but
        does not touch the transform path; the surviving polygons should
        still emerge in CRS space.
        """
        raster = self._raster(transform=self._TRANSFORM)
        # Mask out the right column (col index 2) so only the value-1
        # region in cols 0-1 remains.
        mask_data = np.array(
            [[True, True, False],
             [True, True, False],
             [False, False, False]], dtype=bool)
        mask = xr.DataArray(mask_data, dims=('y', 'x'))
        df = polygonize(raster, mask=mask, return_type='geopandas')
        assert len(df) == 1  # only the value-1 polygon remains
        # The surviving polygon should land at the CRS origin, not pixel 0.
        bounds = df.geometry.total_bounds
        assert bounds[0] >= 1_000_000.0 - 1e-6
        assert bounds[2] <= 1_000_000.0 + 20.0 + 1e-6
        assert bounds[3] <= 5_000_000.0 + 1e-6

    def test_rio_transform_auto_detected(self):
        """rio.transform() is used when attrs['transform'] is absent.

        rioxarray derives ``rio.transform()`` from the DataArray's x/y
        coords when they are present, so the test sets coords that
        encode the desired transform rather than calling
        ``rio.write_transform`` (whose stamped value is overridden by
        any existing coords).
        """
        pytest.importorskip("rioxarray")
        # 10 m pixels, origin at (1e6, 5e6), pixel_height negative.  The
        # coord values are pixel-centre coordinates, so the implied
        # upper-left corner is (1e6, 5e6).
        data = np.array([[1, 1, 2], [1, 1, 2], [2, 2, 2]], dtype=np.int32)
        xs = 1_000_000.0 + 10.0 * np.arange(3) + 5.0  # centres
        ys = 5_000_000.0 - 10.0 * np.arange(3) - 5.0
        raster = xr.DataArray(
            data,
            dims=('y', 'x'),
            coords={'y': ys, 'x': xs},
        )
        # Sanity: rio derives the right transform from the coords.
        derived = tuple(raster.rio.transform())[:6]
        assert derived == self._TRANSFORM, derived
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6


# --- coords-based transform auto-detection tests (issue #2607) ---
#
# polygonize auto-detected attrs['transform'] and rio.transform() but not
# the raster's own x/y coords -- the xarray/xrspatial standard
# georeferencing channel.  A coords-georeferenced raster with a crs attr
# produced a GeoDataFrame whose .crs lied about pixel-space geometries,
# the same misalignment #2536 fixed for the transform attr.

class TestPolygonizeCoordsTransform:
    """polygonize derives a transform from x/y coords when no other
    transform source is available (#2607)."""

    # 10 m pixels, centres so the implied corner origin is (1e6, 5e6).
    @staticmethod
    def _coords():
        xs = 1_000_000.0 + 10.0 * np.arange(3) + 5.0  # centres
        ys = 5_000_000.0 - 10.0 * np.arange(3) - 5.0
        return ys, xs

    @staticmethod
    def _data():
        return np.array([[1, 1, 2], [1, 1, 2], [2, 2, 2]], dtype=np.int32)

    def _raster(self, data=None, **attrs):
        ys, xs = self._coords()
        if data is None:
            data = self._data()
        return xr.DataArray(
            data, dims=('y', 'x'),
            coords={'y': ys, 'x': xs}, attrs=attrs)

    def test_coords_transform_numpy(self):
        """Coords-derived transform lands geometries in CRS space."""
        raster = self._raster()
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        all_ys = np.concatenate([p[0][:, 1] for p in polys])
        # Corner extent: centres span 1e6..1e6+20, so corners 1e6..1e6+30.
        assert all_xs.min() >= 1_000_000.0 - 1e-6
        assert all_xs.max() <= 1_000_000.0 + 30.0 + 1e-6
        assert all_ys.max() <= 5_000_000.0 + 1e-6
        assert all_ys.min() >= 5_000_000.0 - 30.0 - 1e-6

    @pytest.mark.skipif(gpd is None, reason="geopandas not installed")
    def test_coords_transform_geopandas_crs_matches_geometry(self):
        """The geopandas .crs no longer lies about pixel-space geometry."""
        raster = self._raster(crs='EPSG:3857')
        df = polygonize(raster, return_type='geopandas')
        assert df.crs is not None and df.crs.to_epsg() == 3857
        bounds = df.geometry.total_bounds  # [minx, miny, maxx, maxy]
        assert bounds[0] >= 1_000_000.0 - 1e-6
        assert bounds[2] <= 1_000_000.0 + 30.0 + 1e-6
        assert bounds[1] >= 5_000_000.0 - 30.0 - 1e-6
        assert bounds[3] <= 5_000_000.0 + 1e-6

    def test_transform_attr_beats_coords(self):
        """attrs['transform'] takes precedence over coords."""
        # Coords imply origin 1e6; the attr puts origin at 0.
        identity10 = (10.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        raster = self._raster(transform=identity10)
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        # Origin 0, 10 m pixels, 3 cols -> max 30, not ~1e6.
        assert all_xs.max() <= 30.0 + 1e-6

    def test_explicit_transform_beats_coords(self):
        """An explicit transform= overrides coords-derived transform."""
        raster = self._raster()
        identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        _, polys = polygonize(
            raster, transform=identity, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.max() <= 3.0 + 1e-6

    def test_irregular_coords_yield_no_coords_transform(self):
        """The coords fallback bails on unevenly spaced coords.

        ``_transform_from_coords`` is the no-rioxarray fallback (step 3 in
        ``_detect_raster_transform``).  When rioxarray is installed it owns
        step 2 and will average irregular spacing into a transform, so this
        guards the lower-level helper directly rather than the full pipeline.
        """
        from ..polygonize import _transform_from_coords
        ys, _ = self._coords()
        raster = xr.DataArray(
            self._data(), dims=('y', 'x'),
            coords={'y': ys, 'x': [0.0, 1.0, 5.0]})  # irregular x
        assert _transform_from_coords(raster) is None

    def test_single_pixel_coords_fall_back(self):
        """A length-1 coord dim has no spacing; fall back to pixel space."""
        raster = xr.DataArray(
            np.array([[7]], dtype=np.int32), dims=('y', 'x'),
            coords={'y': [5_000_000.0], 'x': [1_000_000.0]})
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.max() <= 2.0 + 1e-6  # nx padded to 2 internally

    def test_no_coords_yields_pixel_space(self):
        """No coords at all -> pixel-space output (unchanged behaviour)."""
        raster = xr.DataArray(self._data(), dims=('y', 'x'))
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.max() <= 3.0 + 1e-6

    def test_no_georef_marker_suppresses_coords(self):
        """_xrspatial_no_georef opts out of coords auto-detection too."""
        raster = self._raster(_xrspatial_no_georef=True)
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.max() <= 3.0 + 1e-6

    def test_coords_transform_with_non_yx_dim_names(self):
        """Coords detection uses the raster's actual dim names, not 'y'/'x'.

        get_dataarray_resolution treats dims[-1] as x and dims[-2] as y,
        so a raster dimensioned ('lat', 'lon') must still be georeferenced
        from its coords.
        """
        ys, xs = self._coords()
        raster = xr.DataArray(
            self._data(), dims=('lat', 'lon'),
            coords={'lat': ys, 'lon': xs})
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6

    @pytest.mark.skipif(da is None, reason="dask not installed")
    def test_coords_transform_dask(self):
        """Dask backend honours the coords-derived transform."""
        ys, xs = self._coords()
        raster = xr.DataArray(
            da.from_array(self._data(), chunks=(2, 2)),
            dims=('y', 'x'), coords={'y': ys, 'x': xs})
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6
        assert all_xs.max() <= 1_000_000.0 + 30.0 + 1e-6

    @cuda_and_cupy_available
    def test_coords_transform_cupy(self):
        """CuPy backend honours the coords-derived transform."""
        import cupy as cp
        ys, xs = self._coords()
        raster = xr.DataArray(
            cp.asarray(self._data()),
            dims=('y', 'x'), coords={'y': ys, 'x': xs})
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6
        assert all_xs.max() <= 1_000_000.0 + 30.0 + 1e-6

    @cuda_and_cupy_available
    def test_coords_transform_dask_cupy(self):
        """Dask+CuPy backend honours the coords-derived transform."""
        import cupy as cp
        ys, xs = self._coords()
        raster = xr.DataArray(
            da.from_array(cp.asarray(self._data()), chunks=(2, 2)),
            dims=('y', 'x'), coords={'y': ys, 'x': xs})
        _, polys = polygonize(raster, return_type='numpy')
        all_xs = np.concatenate([p[0][:, 0] for p in polys])
        assert all_xs.min() >= 1_000_000.0 - 1e-6
        assert all_xs.max() <= 1_000_000.0 + 30.0 + 1e-6


# --- atol / rtol tolerance parameter tests (issue #2173) ---
#
# The float pathway in _calculate_regions groups adjacent pixels using an
# absolute / relative tolerance.  Historically these constants were baked
# into _is_close; #2173 lifted them into the public polygonize() signature
# so a caller working with classified float rasters can opt into strict
# equality without touching internal symbols.

# Repro raster from the issue: three distinct float values, each pair
# within the default atol/rtol of the previous.  With the default
# tolerance (atol=1e-8, rtol=1e-5) all three pixels merge to one region;
# with atol=rtol=0 strict equality recovers three separate regions.
_REPRO_2173 = np.array([[1.0, 1.000009, 1.000018]], dtype=np.float64)


def test_polygonize_default_tolerance_merges_close_floats():
    """Default tolerance keeps backward-compatible float behavior (#2173)."""
    raster = xr.DataArray(_REPRO_2173)
    values, polygons = polygonize(raster)
    assert len(values) == 1
    assert_allclose(values[0], 1.0)
    # Single polygon spanning all three columns.
    total_area = sum(
        assert_polygon_valid_and_get_area(p) for p in polygons)
    assert_allclose(total_area, 3.0)


def test_polygonize_strict_float_equality():
    """atol=0 + rtol=0 -> exact-equality grouping for float rasters (#2173)."""
    raster = xr.DataArray(_REPRO_2173)
    values, polygons = polygonize(raster, atol=0.0, rtol=0.0)
    assert len(values) == 3
    assert_allclose(sorted(values), sorted(_REPRO_2173[0]))
    total_area = sum(
        assert_polygon_valid_and_get_area(p) for p in polygons)
    # Three 1x1 polygons -> total area 3.
    assert_allclose(total_area, 3.0)


def test_polygonize_integer_default_unchanged_by_atol():
    """Integer rasters always use strict equality; atol must not change
    that, even when set to a value that would merge neighbours in floats
    (#2173)."""
    data = np.array([[1, 2, 3]], dtype=np.int64)
    raster = xr.DataArray(data)
    # atol large enough to merge {1, 2, 3} if it were applied.
    values, polygons = polygonize(raster, atol=10.0, rtol=10.0)
    assert sorted(values) == [1, 2, 3]
    total_area = sum(
        assert_polygon_valid_and_get_area(p) for p in polygons)
    assert_allclose(total_area, 3.0)


def test_polygonize_integer_no_args_unchanged():
    """Default tolerance must still produce three polygons for ints with
    distinct values (#2173 regression check)."""
    data = np.array([[1, 2, 3]], dtype=np.int64)
    raster = xr.DataArray(data)
    values, _ = polygonize(raster)
    assert sorted(values) == [1, 2, 3]


def test_polygonize_negative_atol_raises():
    """Negative tolerances are rejected (#2173)."""
    raster = xr.DataArray(_REPRO_2173)
    with pytest.raises(ValueError, match="atol must be a non-negative"):
        polygonize(raster, atol=-1e-9)
    with pytest.raises(ValueError, match="rtol must be a non-negative"):
        polygonize(raster, rtol=-1e-9)


def test_polygonize_nan_atol_raises():
    """NaN tolerances are rejected (#2173 review).

    A NaN tolerance silently fails the `abs(diff) <= nan + ...` check for
    every pair, which would produce a raster of singleton polygons with no
    error.  Catch it up front so callers do not chase a phantom data
    corruption bug.
    """
    raster = xr.DataArray(_REPRO_2173)
    with pytest.raises(ValueError, match="atol must be a non-negative"):
        polygonize(raster, atol=float("nan"))
    with pytest.raises(ValueError, match="rtol must be a non-negative"):
        polygonize(raster, rtol=float("nan"))
    with pytest.raises(ValueError, match="atol must be a non-negative"):
        polygonize(raster, atol=float("inf"))
    with pytest.raises(ValueError, match="rtol must be a non-negative"):
        polygonize(raster, rtol=float("inf"))


def test_polygonize_custom_atol_intermediate():
    """An atol that lies between consecutive float steps should split the
    repro raster in a predictable way (#2173)."""
    # Steps in the repro are 9e-6.  An atol of 1e-6 (with rtol=0) is
    # smaller than the step so all three pixels become distinct regions.
    raster = xr.DataArray(_REPRO_2173)
    values, _ = polygonize(raster, atol=1e-6, rtol=0.0)
    assert len(values) == 3
    # An atol of 1e-4 covers the step and merges them back to one region.
    values, _ = polygonize(raster, atol=1e-4, rtol=0.0)
    assert len(values) == 1


@dask_array_available
def test_polygonize_dask_single_chunk_default_tolerance():
    """Default tolerance merge applies identically through the dask path
    for a single-chunk float raster (#2173)."""
    raster = xr.DataArray(da.from_array(_REPRO_2173, chunks=_REPRO_2173.shape))
    values, polygons = polygonize(raster)
    assert len(values) == 1
    assert_allclose(values[0], 1.0)
    total_area = sum(
        assert_polygon_valid_and_get_area(p) for p in polygons)
    assert_allclose(total_area, 3.0)


@dask_array_available
def test_polygonize_dask_single_chunk_strict_float():
    """atol=rtol=0 must propagate through the dask single-chunk path and
    produce three distinct float polygons (#2173)."""
    raster = xr.DataArray(da.from_array(_REPRO_2173, chunks=_REPRO_2173.shape))
    values, polygons = polygonize(raster, atol=0.0, rtol=0.0)
    assert len(values) == 3
    assert_allclose(sorted(values), sorted(_REPRO_2173[0]))


@dask_array_available
def test_polygonize_dask_multi_chunk_default_tolerance():
    """Multi-chunk dask path applies the default atol/rtol both per-chunk
    and at the chunk-stitching step, so close floats in adjacent chunks
    bucket together (#2171, #2173).

    Transitive bucketing across chunks must match numpy CCL within a
    single chunk (#2583).  With [1.0, 1.000009, 1.000018] split into
    single-pixel chunks, the middle pixel is within tolerance of both
    ends so all three pixels join one region -- the same answer numpy
    returns for the equivalent single-chunk input.
    """
    raster = xr.DataArray(da.from_array(_REPRO_2173, chunks=(1, 1)))
    values, polygons = polygonize(raster)
    # Numpy parity: single chunk numpy chains 1.0 -> 1.000009 -> 1.000018
    # into one region, dask must agree.
    v_np, _ = polygonize(xr.DataArray(_REPRO_2173))
    assert len(values) == len(v_np) == 1
    assert_allclose(sorted(values), sorted(v_np))
    total_area = sum(
        assert_polygon_valid_and_get_area(p) for p in polygons)
    assert_allclose(total_area, 3.0)


@dask_array_available
def test_polygonize_dask_multi_chunk_strict_float():
    """Multi-chunk dask path with atol=rtol=0 keeps every distinct float
    value separate, mirroring numpy strict behavior within each chunk
    (#2173)."""
    raster = xr.DataArray(da.from_array(_REPRO_2173, chunks=(1, 1)))
    values, polygons = polygonize(raster, atol=0.0, rtol=0.0)
    assert len(values) == 3
    assert_allclose(sorted(values), sorted(_REPRO_2173[0]))


@dask_array_available
def test_polygonize_dask_integer_atol_ignored():
    """Integer dask path must ignore atol/rtol just like the numpy path
    (#2173)."""
    data = np.array([[1, 2, 3]], dtype=np.int64)
    raster = xr.DataArray(da.from_array(data, chunks=(1, 1)))
    values, _ = polygonize(raster, atol=10.0, rtol=10.0)
    assert sorted(values) == [1, 2, 3]
