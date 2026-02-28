"""Tests for xrspatial.surface_distance."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.surface_distance import (
    surface_distance, surface_allocation, surface_direction,
)

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import cupy
except ImportError:
    cupy = None

from xrspatial.utils import has_cuda_and_cupy, is_cupy_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raster(data, backend='numpy', chunks=(3, 3), name='raster',
                 res=1.0):
    """Build a DataArray with y/x coords, optionally dask/cupy-backed."""
    h, w = data.shape
    arr = xr.DataArray(
        data.astype(np.float64),
        dims=['y', 'x'],
        coords={'y': np.arange(h, dtype=np.float64),
                'x': np.arange(w, dtype=np.float64)},
        attrs={'res': (float(res), float(res))},
        name=name,
    )
    if backend == 'dask+numpy':
        if da is None:
            pytest.skip("dask not installed")
        arr.data = da.from_array(arr.data, chunks=chunks)
    elif backend == 'cupy':
        if not has_cuda_and_cupy():
            pytest.skip("cupy/cuda not available")
        arr.data = cupy.asarray(arr.data)
    elif backend == 'dask+cupy':
        if da is None or not has_cuda_and_cupy():
            pytest.skip("dask or cupy/cuda not available")
        arr.data = da.from_array(cupy.asarray(arr.data), chunks=chunks)
    return arr


def _compute(arr):
    """Extract numpy data from any backend."""
    d = arr.data
    if da is not None and isinstance(d, da.Array):
        d = d.compute()
    if has_cuda_and_cupy() and is_cupy_array(d):
        d = d.get()
    return np.asarray(d)


# ---------------------------------------------------------------------------
# Tests — flat terrain (must match Euclidean proximity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_flat_terrain_matches_cost_distance(backend):
    """On zero-elevation, surface distance equals cost_distance with
    friction=1 (both use grid-graph Dijkstra)."""
    from xrspatial.cost_distance import cost_distance

    source = np.zeros((7, 7), dtype=np.float64)
    source[3, 3] = 1.0  # single target at centre
    elev = np.zeros((7, 7), dtype=np.float64)
    friction = np.ones((7, 7), dtype=np.float64)

    raster = _make_raster(source, backend=backend, chunks=(4, 4))
    elevation = _make_raster(elev, backend=backend, chunks=(4, 4))
    friction_da = _make_raster(friction, backend=backend, chunks=(4, 4))

    sd = surface_distance(raster, elevation)
    cd = cost_distance(raster, friction_da)

    sd_np = _compute(sd)
    cd_np = _compute(cd)

    np.testing.assert_allclose(sd_np, cd_np, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_flat_terrain_known_distances(backend):
    """On flat terrain, verify known grid-graph distances."""
    source = np.zeros((5, 5), dtype=np.float64)
    source[2, 2] = 1.0  # single target at centre
    elev = np.zeros((5, 5), dtype=np.float64)

    raster = _make_raster(source, backend=backend, chunks=(3, 3))
    elevation = _make_raster(elev, backend=backend, chunks=(3, 3))

    sd = _compute(surface_distance(raster, elevation))

    # Source pixel
    assert sd[2, 2] == 0.0
    # Cardinal neighbours: distance = 1.0
    assert sd[2, 3] == pytest.approx(1.0, abs=1e-5)
    assert sd[2, 1] == pytest.approx(1.0, abs=1e-5)
    assert sd[1, 2] == pytest.approx(1.0, abs=1e-5)
    assert sd[3, 2] == pytest.approx(1.0, abs=1e-5)
    # Diagonal neighbours: distance = sqrt(2)
    assert sd[1, 1] == pytest.approx(np.sqrt(2), abs=1e-5)
    assert sd[1, 3] == pytest.approx(np.sqrt(2), abs=1e-5)
    assert sd[3, 1] == pytest.approx(np.sqrt(2), abs=1e-5)
    assert sd[3, 3] == pytest.approx(np.sqrt(2), abs=1e-5)
    # Two cardinal steps: distance = 2.0
    assert sd[0, 2] == pytest.approx(2.0, abs=1e-5)
    assert sd[4, 2] == pytest.approx(2.0, abs=1e-5)
    assert sd[2, 0] == pytest.approx(2.0, abs=1e-5)
    assert sd[2, 4] == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests — steep terrain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_steep_terrain_increases_distance(backend):
    """Steep terrain should give longer surface distances than flat."""
    source = np.zeros((5, 5), dtype=np.float64)
    source[2, 2] = 1.0
    flat_elev = np.zeros((5, 5), dtype=np.float64)
    steep_elev = np.zeros((5, 5), dtype=np.float64)
    # Elevation ramp: 100m per cell in y direction
    for r in range(5):
        steep_elev[r, :] = r * 100.0

    raster = _make_raster(source, backend=backend, chunks=(3, 3))
    elev_flat = _make_raster(flat_elev, backend=backend, chunks=(3, 3))
    elev_steep = _make_raster(steep_elev, backend=backend, chunks=(3, 3))

    sd_flat = _compute(surface_distance(raster, elev_flat))
    sd_steep = _compute(surface_distance(raster, elev_steep))

    # All non-zero distances should be larger for steep terrain
    mask = np.isfinite(sd_flat) & (sd_flat > 0)
    assert np.all(sd_steep[mask] >= sd_flat[mask])
    # At least some distances should be strictly larger
    assert np.any(sd_steep[mask] > sd_flat[mask])


def test_45_degree_slope():
    """A 45-degree slope (dz = cellsize) gives sqrt(2)*cellsize per step."""
    source = np.zeros((1, 5), dtype=np.float64)
    source[0, 0] = 1.0
    # Elevation ramp: 1.0 per cell (with cellsize=1.0, slope=45 deg)
    elev = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])

    raster = _make_raster(source, res=1.0)
    elevation = _make_raster(elev, res=1.0)

    sd = _compute(surface_distance(raster, elevation))

    # Each cardinal step: sqrt(1^2 + 1^2) = sqrt(2)
    expected = np.array([0.0, np.sqrt(2), 2 * np.sqrt(2),
                         3 * np.sqrt(2), 4 * np.sqrt(2)], dtype=np.float32)
    np.testing.assert_allclose(sd[0], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests — NaN barriers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_nan_barrier(backend):
    """NaN elevation should block pathfinding."""
    source = np.zeros((3, 5), dtype=np.float64)
    source[1, 0] = 1.0
    elev = np.zeros((3, 5), dtype=np.float64)
    # Wall of NaN in column 2
    elev[:, 2] = np.nan

    raster = _make_raster(source, backend=backend, chunks=(3, 3))
    elevation = _make_raster(elev, backend=backend, chunks=(3, 3))

    sd = _compute(surface_distance(raster, elevation))

    # Columns 0-1 should be reachable
    assert np.isfinite(sd[1, 0])  # source
    assert np.isfinite(sd[1, 1])  # adjacent

    # Column 2 (barrier) should be NaN
    assert np.all(np.isnan(sd[:, 2]))

    # Columns 3-4 (behind barrier) should be NaN
    assert np.all(np.isnan(sd[:, 3:]))


def test_nan_elevation_source_ignored():
    """Source on NaN elevation should not be seeded."""
    source = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    elev = np.array([[np.nan, 0.0, 0.0]], dtype=np.float64)

    raster = _make_raster(source)
    elevation = _make_raster(elev)

    sd = _compute(surface_distance(raster, elevation))

    # Source is on NaN elevation, so nothing is reachable
    assert np.all(np.isnan(sd))


# ---------------------------------------------------------------------------
# Tests — allocation correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_allocation_consistency(backend):
    """Allocation must assign to the nearest target by surface distance."""
    source = np.zeros((5, 5), dtype=np.float64)
    source[0, 0] = 1.0
    source[4, 4] = 2.0
    elev = np.zeros((5, 5), dtype=np.float64)

    raster = _make_raster(source, backend=backend, chunks=(3, 3))
    elevation = _make_raster(elev, backend=backend, chunks=(3, 3))

    sd = _compute(surface_distance(raster, elevation))
    sa = _compute(surface_allocation(raster, elevation))

    # Source pixels should have alloc = their own value
    assert sa[0, 0] == 1.0
    assert sa[4, 4] == 2.0

    # The centre pixel should be allocated to whichever source is closer
    # Both are equidistant on flat terrain, so either 1 or 2 is acceptable
    assert sa[2, 2] in (1.0, 2.0)

    # Near source 1 should be allocated to source 1
    assert sa[0, 1] == 1.0
    assert sa[1, 0] == 1.0

    # Near source 2 should be allocated to source 2
    assert sa[4, 3] == 2.0
    assert sa[3, 4] == 2.0


# ---------------------------------------------------------------------------
# Tests — direction correctness
# ---------------------------------------------------------------------------


def test_direction_source_is_zero():
    """Source pixel direction should be 0."""
    source = np.zeros((3, 3), dtype=np.float64)
    source[1, 1] = 1.0
    elev = np.zeros((3, 3), dtype=np.float64)

    raster = _make_raster(source)
    elevation = _make_raster(elev)

    sd_dir = _compute(surface_direction(raster, elevation))

    assert sd_dir[1, 1] == 0.0


def test_direction_cardinal_points():
    """Check compass directions for 4 cardinal neighbours of a source."""
    source = np.zeros((3, 3), dtype=np.float64)
    source[1, 1] = 1.0
    elev = np.zeros((3, 3), dtype=np.float64)

    raster = _make_raster(source)
    elevation = _make_raster(elev)

    sd_dir = _compute(surface_direction(raster, elevation))

    # East of source (1, 2): direction should point west-ish (toward source)
    # Source is at x=1, pixel is at x=2. Direction from pixel to source
    # is toward west = 270
    assert sd_dir[1, 2] == pytest.approx(270.0, abs=1.0)

    # West of source (1, 0): direction should point east = 90
    assert sd_dir[1, 0] == pytest.approx(90.0, abs=1.0)

    # North of source (0, 1) with y increasing downward (row 0 = y=0):
    # Source at row 1, pixel at row 0.
    # dy = (src_row - pixel_row) * cellsize_y = (1 - 0) * 1 = 1 (south)
    # Direction to source is south = 180
    assert sd_dir[0, 1] == pytest.approx(180.0, abs=1.0)

    # South of source (2, 1): direction to source is north = 360
    assert sd_dir[2, 1] == pytest.approx(360.0, abs=1.0)


# ---------------------------------------------------------------------------
# Tests — max_distance clipping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_max_distance_clipping(backend):
    """Pixels beyond max_distance should be NaN."""
    source = np.zeros((5, 5), dtype=np.float64)
    source[2, 2] = 1.0
    elev = np.zeros((5, 5), dtype=np.float64)

    raster = _make_raster(source, backend=backend, chunks=(3, 3))
    elevation = _make_raster(elev, backend=backend, chunks=(3, 3))

    sd = _compute(surface_distance(raster, elevation, max_distance=1.5))

    # Source and immediate cardinal neighbours should be within 1.5
    assert sd[2, 2] == 0.0
    assert np.isfinite(sd[2, 3])  # distance 1.0
    assert np.isfinite(sd[2, 1])  # distance 1.0
    assert np.isfinite(sd[1, 2])  # distance 1.0
    assert np.isfinite(sd[3, 2])  # distance 1.0

    # Diagonal neighbours are sqrt(2) ≈ 1.414, still within 1.5
    assert np.isfinite(sd[1, 1])

    # Two steps away (distance 2.0) should be clipped
    assert np.isnan(sd[0, 2])
    assert np.isnan(sd[2, 0])
    assert np.isnan(sd[4, 2])
    assert np.isnan(sd[2, 4])


# ---------------------------------------------------------------------------
# Tests — target_values filtering
# ---------------------------------------------------------------------------


def test_target_values():
    """Only specified target values should be used as sources."""
    source = np.array([
        [0.0, 1.0, 0.0, 2.0, 0.0],
    ], dtype=np.float64)
    elev = np.zeros((1, 5), dtype=np.float64)

    raster = _make_raster(source)
    elevation = _make_raster(elev)

    # Only use value 2 as target
    sd = _compute(surface_distance(raster, elevation, target_values=[2]))

    # Pixel at col 3 (value 2) should be 0
    assert sd[0, 3] == 0.0
    # Pixel at col 1 (value 1) should NOT be 0 (not a target)
    assert sd[0, 1] > 0.0
    # Distance from col 1 to col 3 = 2.0
    assert sd[0, 1] == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests — connectivity
# ---------------------------------------------------------------------------


def test_connectivity_4_vs_8():
    """4-connectivity should give longer diagonal distances than 8."""
    source = np.zeros((5, 5), dtype=np.float64)
    source[0, 0] = 1.0
    elev = np.zeros((5, 5), dtype=np.float64)

    raster = _make_raster(source)
    elevation = _make_raster(elev)

    sd_8 = _compute(surface_distance(raster, elevation, connectivity=8))
    sd_4 = _compute(surface_distance(raster, elevation, connectivity=4))

    # With 4-conn, diagonal pixel (1,1) needs 2 steps = 2.0
    # With 8-conn, diagonal pixel (1,1) needs 1 step = sqrt(2)
    assert sd_4[1, 1] == pytest.approx(2.0, abs=1e-5)
    assert sd_8[1, 1] == pytest.approx(np.sqrt(2), abs=1e-5)


# ---------------------------------------------------------------------------
# Tests — validation
# ---------------------------------------------------------------------------


def test_invalid_connectivity():
    source = _make_raster(np.zeros((3, 3)))
    elev = _make_raster(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="connectivity"):
        surface_distance(source, elev, connectivity=6)


def test_shape_mismatch():
    source = _make_raster(np.zeros((3, 3)))
    elev = _make_raster(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="same shape"):
        surface_distance(source, elev)


def test_invalid_method():
    source = _make_raster(np.zeros((3, 3)))
    elev = _make_raster(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="method"):
        surface_distance(source, elev, method='fast')


# ---------------------------------------------------------------------------
# Tests — dask-specific
# ---------------------------------------------------------------------------


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_dask_matches_numpy():
    """Dask+numpy result must match numpy baseline."""
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    source[6, 7] = 2.0
    elev = np.random.default_rng(42).uniform(0, 100, (8, 10))

    raster_np = _make_raster(source, backend='numpy')
    elev_np = _make_raster(elev, backend='numpy')
    raster_dask = _make_raster(source, backend='dask+numpy', chunks=(4, 5))
    elev_dask = _make_raster(elev, backend='dask+numpy', chunks=(4, 5))

    np_result = _compute(surface_distance(raster_np, elev_np,
                                          max_distance=15.0))
    dask_result = _compute(surface_distance(raster_dask, elev_dask,
                                            max_distance=15.0))

    np.testing.assert_allclose(dask_result, np_result, rtol=1e-5,
                               equal_nan=True)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_dask_allocation_matches_numpy():
    """Dask+numpy allocation must match numpy baseline."""
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    source[6, 7] = 2.0
    elev = np.random.default_rng(42).uniform(0, 100, (8, 10))

    raster_np = _make_raster(source, backend='numpy')
    elev_np = _make_raster(elev, backend='numpy')
    raster_dask = _make_raster(source, backend='dask+numpy', chunks=(4, 5))
    elev_dask = _make_raster(elev, backend='dask+numpy', chunks=(4, 5))

    np_result = _compute(surface_allocation(raster_np, elev_np,
                                            max_distance=15.0))
    dask_result = _compute(surface_allocation(raster_dask, elev_dask,
                                              max_distance=15.0))

    np.testing.assert_allclose(dask_result, np_result, rtol=1e-5,
                               equal_nan=True)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_iterative_matches_numpy():
    """Dask iterative (unbounded) result must match numpy."""
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    source[6, 7] = 2.0
    rng = np.random.default_rng(42)
    elev = rng.uniform(0, 50, (8, 10))

    raster_np = _make_raster(source, backend='numpy')
    elev_np = _make_raster(elev, backend='numpy')
    raster_dask = _make_raster(source, backend='dask+numpy', chunks=(4, 5))
    elev_dask = _make_raster(elev, backend='dask+numpy', chunks=(4, 5))

    np_result = _compute(surface_distance(raster_np, elev_np))

    with pytest.warns(UserWarning, match="iterative"):
        dask_result = _compute(surface_distance(raster_dask, elev_dask))

    np.testing.assert_allclose(dask_result, np_result, rtol=1e-4,
                               equal_nan=True)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_iterative_allocation_matches_numpy():
    """Dask iterative allocation must match numpy."""
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    source[6, 7] = 2.0
    rng = np.random.default_rng(42)
    elev = rng.uniform(0, 50, (8, 10))

    raster_np = _make_raster(source, backend='numpy')
    elev_np = _make_raster(elev, backend='numpy')
    raster_dask = _make_raster(source, backend='dask+numpy', chunks=(4, 5))
    elev_dask = _make_raster(elev, backend='dask+numpy', chunks=(4, 5))

    np_result = _compute(surface_allocation(raster_np, elev_np))

    with pytest.warns(UserWarning, match="iterative"):
        dask_result = _compute(surface_allocation(raster_dask, elev_dask))

    np.testing.assert_allclose(dask_result, np_result, rtol=1e-4,
                               equal_nan=True)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_dask_returns_dask_array():
    """Result should be a dask array when input is dask."""
    source = np.zeros((5, 5), dtype=np.float64)
    source[2, 2] = 1.0
    elev = np.zeros((5, 5), dtype=np.float64)

    raster = _make_raster(source, backend='dask+numpy', chunks=(3, 3))
    elevation = _make_raster(elev, backend='dask+numpy', chunks=(3, 3))

    sd = surface_distance(raster, elevation, max_distance=5.0)
    assert isinstance(sd.data, da.Array)


# ---------------------------------------------------------------------------
# Tests — CuPy-specific (skipped if not available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="cupy/cuda not available")
def test_cupy_matches_numpy():
    """CuPy result must match numpy baseline."""
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    source[6, 7] = 2.0
    rng = np.random.default_rng(42)
    elev = rng.uniform(0, 100, (8, 10))

    raster_np = _make_raster(source, backend='numpy')
    elev_np = _make_raster(elev, backend='numpy')
    raster_cp = _make_raster(source, backend='cupy')
    elev_cp = _make_raster(elev, backend='cupy')

    np_result = _compute(surface_distance(raster_np, elev_np))
    cp_result = _compute(surface_distance(raster_cp, elev_cp))

    np.testing.assert_allclose(cp_result, np_result, rtol=1e-5,
                               equal_nan=True)


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="cupy/cuda not available")
def test_cupy_returns_cupy_array():
    """CuPy input should produce CuPy output."""
    source = np.zeros((5, 5), dtype=np.float64)
    source[2, 2] = 1.0
    elev = np.zeros((5, 5), dtype=np.float64)

    raster = _make_raster(source, backend='cupy')
    elevation = _make_raster(elev, backend='cupy')

    sd = surface_distance(raster, elevation)
    assert is_cupy_array(sd.data)


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="cupy/cuda not available")
def test_cupy_allocation_matches_numpy():
    """CuPy allocation must match numpy baseline."""
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    source[6, 7] = 2.0
    rng = np.random.default_rng(42)
    elev = rng.uniform(0, 100, (8, 10))

    raster_np = _make_raster(source, backend='numpy')
    elev_np = _make_raster(elev, backend='numpy')
    raster_cp = _make_raster(source, backend='cupy')
    elev_cp = _make_raster(elev, backend='cupy')

    np_result = _compute(surface_allocation(raster_np, elev_np))
    cp_result = _compute(surface_allocation(raster_cp, elev_cp))

    np.testing.assert_allclose(cp_result, np_result, rtol=1e-5,
                               equal_nan=True)


# ---------------------------------------------------------------------------
# Tests — multiple sources
# ---------------------------------------------------------------------------


def test_multiple_sources_nearest_wins():
    """Each pixel should be assigned to its nearest source."""
    source = np.zeros((1, 7), dtype=np.float64)
    source[0, 0] = 1.0
    source[0, 6] = 2.0
    elev = np.zeros((1, 7), dtype=np.float64)

    raster = _make_raster(source)
    elevation = _make_raster(elev)

    sd = _compute(surface_distance(raster, elevation))
    sa = _compute(surface_allocation(raster, elevation))

    # Pixel 0: dist 0 from source 1
    assert sd[0, 0] == 0.0
    assert sa[0, 0] == 1.0

    # Pixel 6: dist 0 from source 2
    assert sd[0, 6] == 0.0
    assert sa[0, 6] == 2.0

    # Pixel 3: equidistant (dist 3 from both)
    assert sd[0, 3] == pytest.approx(3.0, abs=1e-5)

    # Pixels 1, 2 closer to source 1
    assert sa[0, 1] == 1.0
    assert sa[0, 2] == 1.0

    # Pixels 4, 5 closer to source 2
    assert sa[0, 4] == 2.0
    assert sa[0, 5] == 2.0


# ---------------------------------------------------------------------------
# Tests — no sources
# ---------------------------------------------------------------------------


def test_no_sources_all_nan():
    """When there are no sources, all outputs should be NaN."""
    source = np.zeros((3, 3), dtype=np.float64)
    elev = np.zeros((3, 3), dtype=np.float64)

    raster = _make_raster(source)
    elevation = _make_raster(elev)

    sd = _compute(surface_distance(raster, elevation))
    sa = _compute(surface_allocation(raster, elevation))
    sd_dir = _compute(surface_direction(raster, elevation))

    assert np.all(np.isnan(sd))
    assert np.all(np.isnan(sa))
    assert np.all(np.isnan(sd_dir))


# ---------------------------------------------------------------------------
# Tests — geodesic mode (numpy only)
# ---------------------------------------------------------------------------


def test_geodesic_basic():
    """Basic geodesic test: horizontal distances should be in meters."""
    # Small grid centred at equator
    source = np.zeros((3, 3), dtype=np.float64)
    source[1, 1] = 1.0
    elev = np.zeros((3, 3), dtype=np.float64)

    h, w = source.shape
    lat = np.array([1.0, 0.0, -1.0])  # degrees
    lon = np.array([-1.0, 0.0, 1.0])

    raster = xr.DataArray(
        source,
        dims=['y', 'x'],
        coords={'y': lat, 'x': lon},
    )
    elevation = xr.DataArray(
        elev,
        dims=['y', 'x'],
        coords={'y': lat, 'x': lon},
    )

    sd = _compute(surface_distance(raster, elevation, method='geodesic'))

    # Source pixel should be 0
    assert sd[1, 1] == 0.0

    # Cardinal neighbours should be ~111 km (1 degree at equator)
    for pos in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        assert 100000 < sd[pos] < 130000  # roughly 100-130 km
