"""Tests for xrspatial.surface_distance."""

import warnings

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


def test_direction_follows_y_axis_orientation():
    """Bearings must use the y coordinate, not the row index (#3719).

    A north-up raster has y descending with the row index. Reading the row
    index as if it were y mirrors every bearing across the east-west axis.
    """
    source = np.zeros((3, 3), dtype=np.float64)
    source[1, 1] = 1.0
    elev = np.zeros((3, 3), dtype=np.float64)

    def _build(y_coords):
        return (
            xr.DataArray(source, dims=['y', 'x'],
                         coords={'y': y_coords,
                                 'x': np.arange(3, dtype=np.float64)}),
            xr.DataArray(elev, dims=['y', 'x'],
                         coords={'y': y_coords,
                                 'x': np.arange(3, dtype=np.float64)}),
        )

    # Descending y: row 0 is north of the source, so it points north (360).
    north_up = _compute(surface_direction(
        *_build(np.array([2.0, 1.0, 0.0]))))
    assert north_up[0, 1] == pytest.approx(360.0, abs=1.0)
    assert north_up[2, 1] == pytest.approx(180.0, abs=1.0)

    # Ascending y: row 0 is south of the source, so it points south (180).
    south_up = _compute(surface_direction(
        *_build(np.array([0.0, 1.0, 2.0]))))
    assert south_up[0, 1] == pytest.approx(180.0, abs=1.0)
    assert south_up[2, 1] == pytest.approx(360.0, abs=1.0)

    # East/west is unaffected by the y orientation.
    for out in (north_up, south_up):
        assert out[1, 0] == pytest.approx(90.0, abs=1.0)
        assert out[1, 2] == pytest.approx(270.0, abs=1.0)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
@pytest.mark.parametrize("func", [surface_distance, surface_allocation])
def test_distance_and_allocation_ignore_axis_direction(backend, func):
    """Only the bearing cares which way the axes run (#3719).

    _compute hands the backends signed cell sizes so _finalize_direction
    can build a compass bearing. That is safe only because every
    edge-cost consumer squares or abs()es the cell size first. Flipping
    the y axis must leave distance and allocation untouched apart from
    the row order.
    """
    source = np.zeros((6, 6), dtype=np.float64)
    source[1, 1] = 1.0
    source[4, 4] = 2.0
    elev = np.random.default_rng(11).uniform(0, 20, (6, 6))

    def _build(y_coords):
        arrays = []
        for data in (source, elev):
            arr = xr.DataArray(
                data.copy(), dims=['y', 'x'],
                coords={'y': y_coords, 'x': np.arange(6, dtype=np.float64)},
                attrs={'res': (1.0, 1.0)})
            if backend == 'dask+numpy':
                if da is None:
                    pytest.skip("dask not installed")
                arr.data = da.from_array(arr.data, chunks=(3, 3))
            arrays.append(arr)
        return arrays

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        up = _compute(func(*_build(np.arange(6, dtype=np.float64))))
        down = _compute(func(*_build(np.arange(6, dtype=np.float64)[::-1])))

    np.testing.assert_allclose(down, up, rtol=1e-6, equal_nan=True)


def test_direction_matches_proximity_direction():
    """surface_direction on flat terrain agrees with proximity.direction.

    Both document the same compass convention, so on zero relief with one
    source they must return the same bearings (#3719).
    """
    from xrspatial.proximity import direction

    source = np.zeros((5, 5), dtype=np.float64)
    source[3, 1] = 1.0
    elev = np.zeros((5, 5), dtype=np.float64)
    y_coords = np.array([40.0, 39.0, 38.0, 37.0, 36.0])  # descending
    x_coords = np.arange(5, dtype=np.float64)

    raster = xr.DataArray(source, dims=['y', 'x'],
                          coords={'y': y_coords, 'x': x_coords})
    elevation = xr.DataArray(elev, dims=['y', 'x'],
                             coords={'y': y_coords, 'x': x_coords})

    np.testing.assert_allclose(
        _compute(surface_direction(raster, elevation)),
        _compute(direction(raster)),
        rtol=1e-5,
    )


@pytest.mark.skipif(da is None, reason="dask not installed")
@pytest.mark.parametrize("max_distance", [np.inf, 4.0])
def test_dask_direction_matches_numpy(max_distance):
    """Dask direction must match numpy on every chunk (#3719).

    The unbounded path re-runs each tile with global source indices; those
    have to be rebased before the bearing is measured against the block's
    own index grid, or every chunk but the first is wrong.
    """
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    elev = np.random.default_rng(7).uniform(0, 5, (8, 10))

    raster_np = _make_raster(source, backend='numpy')
    elev_np = _make_raster(elev, backend='numpy')
    raster_dask = _make_raster(source, backend='dask+numpy', chunks=(4, 5))
    elev_dask = _make_raster(elev, backend='dask+numpy', chunks=(4, 5))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        np_result = _compute(surface_direction(
            raster_np, elev_np, max_distance=max_distance))
        dask_result = _compute(surface_direction(
            raster_dask, elev_dask, max_distance=max_distance))

    np.testing.assert_allclose(dask_result, np_result, rtol=1e-5,
                               equal_nan=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
@pytest.mark.parametrize("max_distance", [np.inf, 4.0])
def test_direction_nan_mask_matches_distance(backend, max_distance):
    """Direction is NaN exactly where distance is NaN (#3719).

    Guards the tiled dask path, where a chunk whose source lives in a
    neighbouring chunk must still get a bearing.
    """
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    elev = np.random.default_rng(7).uniform(0, 5, (8, 10))

    raster = _make_raster(source, backend=backend, chunks=(4, 5))
    elevation = _make_raster(elev, backend=backend, chunks=(4, 5))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dist = _compute(surface_distance(raster, elevation,
                                         max_distance=max_distance))
        bearing = _compute(surface_direction(raster, elevation,
                                             max_distance=max_distance))

    np.testing.assert_array_equal(np.isnan(bearing), np.isnan(dist))


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="cupy/cuda not available")
def test_cupy_direction_matches_numpy():
    """CuPy direction must match the numpy baseline (#3719)."""
    source = np.zeros((8, 10), dtype=np.float64)
    source[2, 3] = 1.0
    elev = np.random.default_rng(7).uniform(0, 5, (8, 10))

    np_result = _compute(surface_direction(
        _make_raster(source), _make_raster(elev)))
    cp_result = _compute(surface_direction(
        _make_raster(source, backend='cupy'),
        _make_raster(elev, backend='cupy')))

    np.testing.assert_allclose(cp_result, np_result, rtol=1e-5,
                               equal_nan=True)


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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy',
                                     'dask+cupy'])
@pytest.mark.parametrize("func", [surface_distance, surface_allocation,
                                  surface_direction])
@pytest.mark.parametrize("bad", [np.nan, -5.0])
def test_invalid_max_distance(backend, func, bad):
    """NaN / negative max_distance must raise, not diverge by backend.

    Before this check NaN slipped past the numpy kernel's `cost_u >
    max_distance` break (full unbounded surface) but blocked the CUDA
    kernel's `best <= max_distance` accept (seeds only), and a negative
    budget reached dask as a negative map_overlap depth.  See issue #3711.
    """
    data = np.zeros((4, 4))
    data[0, 0] = 1.0
    source = _make_raster(data, backend=backend)
    elev = _make_raster(np.zeros((4, 4)), backend=backend)
    with pytest.raises(ValueError, match="max_distance must be non-negative"):
        func(source, elev, max_distance=bad)


@pytest.mark.parametrize("bad", [1.0, [[1.0, 2.0], [3.0, 4.0]]])
def test_invalid_target_values_shape(bad):
    """Non-1D target_values must be rejected before reaching numba."""
    source = _make_raster(np.ones((3, 3)))
    elev = _make_raster(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="target_values must be a 1-D"):
        surface_distance(source, elev, target_values=bad)


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


def _serpentine_elevation(n):
    """NaN barriers everywhere except one winding open corridor.

    The corridor from (0, 0) is about n*n/2 pixel hops long, far more than
    the n+n the GPU relaxation used to allow itself.
    """
    elev = np.full((n, n), np.nan, dtype=np.float64)
    for r in range(0, n, 2):
        elev[r, :] = 0.0
    for r in range(1, n, 2):
        elev[r, n - 1 if (r // 2) % 2 == 0 else 0] = 0.0
    return elev


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="cupy/cuda not available")
@pytest.mark.parametrize(
    "mode", [surface_distance, surface_allocation, surface_direction])
def test_cupy_long_path_matches_numpy(mode):
    """CuPy must relax until it converges, not for a fixed H+W (#3721).

    A winding corridor needs far more relaxation passes than the raster is
    wide plus tall. Stopping early makes reachable pixels come back NaN,
    which reads as "no path exists".
    """
    n = 16
    elev = _serpentine_elevation(n)
    source = np.zeros((n, n), dtype=np.float64)
    source[0, 0] = 1.0

    np_result = _compute(mode(_make_raster(source), _make_raster(elev)))
    cp_result = _compute(mode(_make_raster(source, backend='cupy'),
                              _make_raster(elev, backend='cupy')))

    # The corridor is genuinely reachable end to end on the CPU.
    assert np.isfinite(np_result[n - 1, 0])
    assert int(np.isfinite(np_result).sum()) > 4 * n

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


# ---------------------------------------------------------------------------
# Metadata propagation (issue #3708)
# ---------------------------------------------------------------------------


def _metadata_backends():
    backends = ['numpy']
    if da is not None:
        backends.append('dask+numpy')
    if has_cuda_and_cupy():
        backends.append('cupy')
        if da is not None:
            backends.append('dask+cupy')
    return backends


@pytest.mark.parametrize("func", [surface_distance, surface_allocation,
                                  surface_direction])
@pytest.mark.parametrize("max_distance", [3.0, np.inf],
                         ids=['bounded', 'unbounded'])
@pytest.mark.parametrize("backend", _metadata_backends())
def test_output_name_consistent_across_backends(backend, max_distance, func):
    """Outputs must not adopt the dask graph token as .name.

    Without the post-construction reset the dask backends returned
    '_trim-<hash>' (bounded map_overlap route),
    'xrspatial.surface_*-<hash>' (unbounded iterative route) or
    'asarray-<hash>' (dask+cupy unbounded), while numpy and cupy returned
    None.  Same bug class as cost_distance #3344 and pathfinding #3652.
    """
    source = np.zeros((6, 6), dtype=np.float64)
    source[0, 0] = 1.0
    elev = np.arange(36, dtype=np.float64).reshape(6, 6) * 0.1
    raster = _make_raster(source, backend=backend)
    elevation = _make_raster(elev, backend=backend)

    result = func(raster, elevation, max_distance=max_distance)
    assert result.name is None


@pytest.mark.parametrize("func", [surface_distance, surface_allocation,
                                  surface_direction])
@pytest.mark.parametrize("backend", _metadata_backends())
def test_output_preserves_attrs_coords_dims(backend, func):
    """attrs, coords and dims come through unchanged on every backend."""
    source = np.zeros((6, 6), dtype=np.float64)
    source[0, 0] = 1.0
    elev = np.arange(36, dtype=np.float64).reshape(6, 6) * 0.1
    raster = _make_raster(source, backend=backend)
    elevation = _make_raster(elev, backend=backend)
    raster.attrs.update({'crs': 3857, 'nodatavals': (-9999.0,),
                         'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)})
    raster = raster.assign_coords(spatial_ref=0)

    result = func(raster, elevation, max_distance=3.0)

    assert result.dims == raster.dims
    assert result.attrs == raster.attrs
    assert set(result.coords) == set(raster.coords)
    for name in raster.coords:
        np.testing.assert_array_equal(result.coords[name].values,
                                      raster.coords[name].values)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------


class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        source = np.zeros((4, 4), dtype=np.float64)
        source[1, 1] = 1.0
        elev = np.zeros((4, 4), dtype=np.float64)
        raster = _make_raster(source)
        elevation = _make_raster(elev)

        # Mock available memory to 1 byte so even a 4x4 raster trips it.
        with patch(
            "xrspatial.surface_distance._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                surface_distance(raster, elevation)
            with pytest.raises(MemoryError, match="working memory"):
                surface_allocation(raster, elevation)
            with pytest.raises(MemoryError, match="working memory"):
                surface_direction(raster, elevation)

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        source = np.zeros((10, 10), dtype=np.float64)
        source[5, 5] = 1.0
        elev = np.zeros((10, 10), dtype=np.float64)
        raster = _make_raster(source)
        elevation = _make_raster(elev)
        # Should not raise -- 10x10 needs ~8 KB.
        result = surface_distance(raster, elevation)
        assert result.shape == (10, 10)

    def test_validation_error_takes_precedence(self):
        """Invalid args raise ValueError before the memory guard runs."""
        from unittest.mock import patch

        source = np.zeros((4, 4), dtype=np.float64)
        elev_wrong = np.zeros((5, 5), dtype=np.float64)
        raster = _make_raster(source)
        elevation = xr.DataArray(
            elev_wrong,
            dims=['y', 'x'],
            coords={'y': np.arange(5, dtype=np.float64),
                    'x': np.arange(5, dtype=np.float64)},
            attrs={'res': (1.0, 1.0)},
        )

        with patch(
            "xrspatial.surface_distance._available_memory_bytes",
            return_value=1,
        ):
            # Mismatched shapes raise ValueError before any allocation.
            with pytest.raises(ValueError, match="same shape"):
                surface_distance(raster, elevation)

            # Invalid connectivity raises ValueError too.
            elev_ok = _make_raster(np.zeros((4, 4), dtype=np.float64))
            with pytest.raises(ValueError, match="connectivity"):
                surface_distance(raster, elev_ok, connectivity=5)

    def test_dask_path_bounded_per_chunk(self):
        """Dask backend inherits the guard per-chunk (not on the full shape).

        A dask raster whose total footprint would trip the guard but whose
        per-chunk footprint fits comfortably should compute successfully.
        """
        if da is None:
            pytest.skip("dask not installed")

        from unittest.mock import patch

        # 200x200 total (~6.4 MB at 80 B/pixel) chunked at 20x20
        # (~32 KB per chunk).  Mock available memory to 1 MB: the full
        # array would exceed 50% of that, but each 20x20 chunk needs
        # only ~32 KB so per-chunk allocation passes.
        source = np.zeros((200, 200), dtype=np.float64)
        source[100, 100] = 1.0
        elev = np.zeros((200, 200), dtype=np.float64)
        raster = _make_raster(source, backend='dask+numpy', chunks=(20, 20))
        elevation = _make_raster(elev, backend='dask+numpy', chunks=(20, 20))

        with patch(
            "xrspatial.surface_distance._available_memory_bytes",
            return_value=1024 * 1024,  # 1 MB
        ):
            # max_distance=5 keeps map_overlap depth small (< chunk size).
            result = surface_distance(raster, elevation, max_distance=5.0)
            # Force a small compute window to prove per-chunk passes.
            _ = result.data[:4, :4].compute()

    def test_error_message_mentions_grid_size(self):
        """The error message names the grid dimensions and the dask alternative."""
        from unittest.mock import patch

        source = np.zeros((7, 11), dtype=np.float64)
        source[3, 5] = 1.0
        elev = np.zeros((7, 11), dtype=np.float64)
        raster = _make_raster(source)
        elevation = _make_raster(elev)

        with patch(
            "xrspatial.surface_distance._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="7x11"):
                surface_distance(raster, elevation)
            with pytest.raises(MemoryError, match="dask"):
                surface_distance(raster, elevation)
