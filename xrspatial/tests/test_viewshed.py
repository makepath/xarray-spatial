import dask.array as da
import numpy as np
import pytest
import xarray as xa

from xrspatial import viewshed
from xrspatial.tests.general_checks import general_output_checks
from xrspatial.utils import has_cuda_and_cupy
from xrspatial.viewshed import INVISIBLE, _calculate_event_row_col

from ..gpu_rtx import has_rtx


@pytest.fixture
def empty_agg():
    # create an empty DataArray of size 5x5 with coordinates
    H = 5
    W = 5
    x_range = (-20, 20)
    y_range = (-20, 20)

    dx = (x_range[1] - x_range[0]) / W
    dy = (y_range[1] - y_range[0]) / H
    xs = np.linspace(x_range[0] + dx / 2, x_range[1] - dx / 2, W)
    ys = np.linspace(y_range[0] + dy / 2, y_range[1] - dy / 2, H)

    agg = xa.DataArray(
        np.zeros((H, W), dtype='i8'),
        coords={'y': ys, 'x': xs},
        dims=['y', 'x'],
    )
    return agg


def test_viewshed_invalid_x_view(empty_agg):
    xs = empty_agg.coords['x'].values
    OBSERVER_X = xs[0] - 1
    OBSERVER_Y = 0
    with pytest.raises(Exception):
        viewshed(raster=empty_agg, x=OBSERVER_X, y=OBSERVER_Y, observer_elev=10)


def test_viewshed_invalid_y_view(empty_agg):
    ys = empty_agg.coords['y'].values
    OBSERVER_X = 0
    OBSERVER_Y = ys[-1] + 1
    with pytest.raises(Exception):
        viewshed(raster=empty_agg, x=OBSERVER_X, y=OBSERVER_Y, observer_elev=10)


def test_viewshed(empty_agg):

    H, W = empty_agg.shape

    # coordinates
    xs = empty_agg.coords['x'].values
    ys = empty_agg.coords['y'].values

    # define some values for observer's elevation to test
    OBS_ELEVS = [-1, 0, 1]
    TERRAIN_ELEV_AT_VP = [-1, 0, 1]

    # check if a matrix is symmetric
    def check_symmetric(matrix, rtol=1e-05, atol=1e-08):
        return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

    def get_matrices(y, x, height, width):
        # indexing 0 1 ... height-1 and 0 1 ... width-1
        height = height - 1
        width = width - 1

        # find first matrix's diagonal
        tmp = min(y, x)
        f_top_y, f_left_x = y - tmp, x - tmp

        tmp = min(height - y, width - x)
        f_bottom_y, f_right_x = y + tmp, x + tmp

        # find second matrix's antidiagonal
        tmp = min(y, width - x)
        s_top_y, s_right_x = y - tmp, x + tmp

        tmp = min(height - y, x)
        s_bottom_y, s_left_x = y + tmp, x - tmp

        return ((f_top_y, f_left_x, f_bottom_y + 1, f_right_x + 1),
                (s_top_y, s_left_x, s_bottom_y + 1, s_right_x + 1))

    # test on 3 scenarios:
    #   empty image.
    #   image with all 0s, except 1 cell with a negative value.
    #   image with all 0s, except 1 cell with a positive value.

    # for each scenario:
    #   if not empty image,
    #      observer is located at the same position as the non zero value.
    #   observer elevation can be: negative, zero, or positive.

    # assertion:
    #   angle at viewpoint is always 180.
    #   when the observer is above the terrain, all cells are visible.
    #   the symmetric property of observer's visibility.

    for obs_elev in OBS_ELEVS:
        for elev_at_vp in TERRAIN_ELEV_AT_VP:
            for col_id, x in enumerate(xs):
                for row_id, y in enumerate(ys):

                    empty_agg.values[row_id, col_id] = elev_at_vp
                    v = viewshed(raster=empty_agg, x=x, y=y, observer_elev=obs_elev)

                    # validate output properties
                    general_output_checks(empty_agg, v)

                    # angle at viewpoint is always 180
                    assert v[row_id, col_id] == 180

                    if obs_elev + elev_at_vp >= 0 and obs_elev >= abs(elev_at_vp):
                        # all cells are visible
                        assert (v.values > -1).all()

                    b1, b2 = get_matrices(row_id, col_id, H, W)
                    m1 = v.values[b1[0]:b1[2], b1[1]:b1[3]]
                    m2 = v.values[b2[0]:b2[2], b2[1]:b2[3]]

                    assert check_symmetric(m1)
                    assert check_symmetric(m2[::-1])

                    # empty image for next uses
                    empty_agg.values[row_id, col_id] = 0


@pytest.mark.parametrize("observer_elev", [5, 2])
@pytest.mark.parametrize("target_elev", [0, 1])
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_viewshed_flat(backend, observer_elev, target_elev):
    if backend == "cupy":
        if not has_rtx():
            pytest.skip("rtxpy not available")
        else:
            import cupy as cp

    x, y = 0, 0
    ny, nx = 5, 4
    arr = np.full((ny, nx), 1.3)
    xs = np.arange(nx)*0.5
    ys = np.arange(ny)*1.5
    if backend == "cupy":
        arr = cp.asarray(arr)
    xarr = xa.DataArray(arr, coords=dict(x=xs, y=ys), dims=["y", "x"])
    v = viewshed(
        xarr, x=x, y=y, observer_elev=observer_elev, target_elev=target_elev)
    if backend == "cupy":
        v.data = cp.asnumpy(v.data)
    xs2, ys2 = np.meshgrid(xs, ys)
    d_vert = observer_elev - target_elev
    d_horz = np.sqrt((xs2 - x)**2 + (ys2 - y)**2)
    angle = np.rad2deg(np.arctan2(d_horz, d_vert))
    # Don't want to compare value under observer.
    angle[0, 0] = v.data[0, 0]
    if backend == "numpy":
        np.testing.assert_allclose(v.data, angle)
    else:
        # Should do better with viewshed gpu output angles.
        mask = (v.data < 90)
        np.testing.assert_allclose(v.data[mask], angle[mask], atol=0.03)


# -------------------------------------------------------------------
# Dask backend tests
# -------------------------------------------------------------------

@pytest.mark.parametrize("observer_elev", [5, 2])
@pytest.mark.parametrize("target_elev", [0, 1])
def test_viewshed_dask_flat(observer_elev, target_elev):
    """Flat terrain: dask should match analytical formula."""
    x, y = 0, 0
    ny, nx = 5, 4
    arr = da.from_array(np.full((ny, nx), 1.3), chunks=(3, 2))
    xs = np.arange(nx) * 0.5
    ys = np.arange(ny) * 1.5
    xarr = xa.DataArray(arr, coords=dict(x=xs, y=ys), dims=["y", "x"])
    v = viewshed(xarr, x=x, y=y,
                 observer_elev=observer_elev, target_elev=target_elev)
    result = v.values

    xs2, ys2 = np.meshgrid(xs, ys)
    d_vert = observer_elev - target_elev
    d_horz = np.sqrt((xs2 - x) ** 2 + (ys2 - y) ** 2)
    expected = np.rad2deg(np.arctan2(d_horz, d_vert))
    expected[0, 0] = result[0, 0]  # observer cell
    np.testing.assert_allclose(result, expected)


def test_viewshed_dask_matches_numpy():
    """Dask should closely match numpy R2 sweep on small varied terrain."""
    np.random.seed(42)
    ny, nx = 12, 10
    terrain = np.random.uniform(0, 5, (ny, nx))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)

    # numpy reference
    raster_np = xa.DataArray(terrain.copy(), coords=dict(x=xs, y=ys),
                             dims=["y", "x"])
    v_np = viewshed(raster_np, x=5.0, y=6.0, observer_elev=10)

    # dask (will use Tier B — full compute + R2 — on this small grid)
    raster_da = xa.DataArray(
        da.from_array(terrain.copy(), chunks=(4, 5)),
        coords=dict(x=xs, y=ys), dims=["y", "x"])
    v_da = viewshed(raster_da, x=5.0, y=6.0, observer_elev=10)

    np.testing.assert_allclose(v_da.values, v_np.values)


def test_viewshed_dask_max_distance():
    """max_distance should produce partial viewshed within radius."""
    ny, nx = 20, 20
    arr_np = np.zeros((ny, nx))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)

    raster_da = xa.DataArray(
        da.from_array(arr_np, chunks=(10, 10)),
        coords=dict(x=xs, y=ys), dims=["y", "x"])
    v = viewshed(raster_da, x=10.0, y=10.0,
                 observer_elev=5, max_distance=5.0)
    result = v.values

    # Observer cell is visible
    assert result[10, 10] == 180.0

    # Cells far beyond max_distance should be INVISIBLE
    assert result[0, 0] == INVISIBLE
    assert result[19, 19] == INVISIBLE

    # Cells within max_distance should be visible (flat terrain, observer up)
    assert result[10, 12] > INVISIBLE  # 2 cells away
    assert result[8, 10] > INVISIBLE   # 2 cells away


def test_viewshed_dask_distance_sweep():
    """Force the distance-sweep path (Tier C) and verify flat terrain."""
    from unittest.mock import patch

    ny, nx = 10, 10
    arr_np = np.full((ny, nx), 0.0)
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)

    raster_da = xa.DataArray(
        da.from_array(arr_np, chunks=(5, 5)),
        coords=dict(x=xs, y=ys), dims=["y", "x"])

    # R2 needs 280*10*10=28000 bytes; Tier B requires <50% of avail.
    # Output grid is 10*10*8=800 bytes; memory guard requires <80% of avail.
    # 10000 bytes: skips Tier B (28000 > 5000) and passes guard (800 < 8000).
    with patch('xrspatial.viewshed._available_memory_bytes',
               return_value=10_000):
        v = viewshed(raster_da, x=5.0, y=5.0, observer_elev=5)

    result = v.values
    assert result[5, 5] == 180.0
    # All cells on flat terrain should be visible
    assert (result > INVISIBLE).all()


def test_viewshed_dask_max_distance_lazy_output():
    """max_distance on a large dask raster should produce a lazy output
    without allocating the full grid in memory."""
    ny, nx = 100_000, 100_000  # 80 GB if materialized
    # Don't actually create the data — just define a lazy dask array
    single_chunk = da.zeros((1000, 1000), chunks=(1000, 1000),
                            dtype=np.float64)
    # Tile to 100k x 100k via dask (no memory allocated)
    big = da.tile(single_chunk, (100, 100))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    raster = xa.DataArray(big, coords=dict(x=xs, y=ys), dims=["y", "x"])
    v = viewshed(raster, x=50000.0, y=50000.0,
                 observer_elev=5, max_distance=10.0)
    # Output should be a dask array (lazy), not numpy
    assert isinstance(v.data, da.Array)
    assert v.shape == (ny, nx)
    # Only compute a small slice to verify correctness
    # max_distance=10 → radius_cells=10 → window is obs ± 10
    center = v.isel(y=slice(49989, 50012), x=slice(49989, 50012)).values
    # Observer is at index 11 within this 23-cell slice
    assert center[11, 11] == 180.0  # observer cell
    # Cells within the circle should be visible (flat terrain, observer up)
    assert center[11, 13] > INVISIBLE  # 2 cells away
    # Corner (49989,49989) is sqrt(11^2+11^2) ≈ 15.6 from observer → outside
    assert center[0, 0] == INVISIBLE


def test_viewshed_numpy_max_distance():
    """max_distance should work on plain numpy raster too."""
    ny, nx = 20, 20
    arr_np = np.zeros((ny, nx))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)

    raster_np = xa.DataArray(arr_np, coords=dict(x=xs, y=ys),
                             dims=["y", "x"])
    v = viewshed(raster_np, x=10.0, y=10.0,
                 observer_elev=5, max_distance=5.0)
    result = v.values

    assert result[10, 10] == 180.0
    assert result[0, 0] == INVISIBLE
    assert result[19, 19] == INVISIBLE
    assert result[10, 12] > INVISIBLE


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_viewshed_max_distance_matches_full(backend):
    """max_distance results should match full viewshed within the radius."""
    if backend == "cupy":
        if not has_rtx():
            pytest.skip("rtxpy not available")
        else:
            import cupy as cp

    ny, nx = 10, 8
    np.random.seed(123)
    arr = np.random.uniform(0, 3, (ny, nx))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    if backend == "cupy":
        arr_backend = cp.asarray(arr)
    else:
        arr_backend = arr.copy()

    raster = xa.DataArray(arr_backend, coords=dict(x=xs, y=ys),
                          dims=["y", "x"])
    v_full = viewshed(raster, x=4.0, y=5.0, observer_elev=10)

    if backend == "cupy":
        arr_backend = cp.asarray(arr)
    else:
        arr_backend = arr.copy()
    raster2 = xa.DataArray(arr_backend, coords=dict(x=xs, y=ys),
                           dims=["y", "x"])
    v_dist = viewshed(raster2, x=4.0, y=5.0, observer_elev=10,
                      max_distance=3.5)

    full_vals = v_full.values if isinstance(v_full.data, np.ndarray) \
        else v_full.data.get()
    dist_vals = v_dist.values if isinstance(v_dist.data, np.ndarray) \
        else v_dist.data.get()

    # Within the circular radius, results should match
    obs_r, obs_c = 5, 4
    max_d = 3.5
    for r in range(max(0, obs_r - 3), min(ny, obs_r + 4)):
        for c in range(max(0, obs_c - 3), min(nx, obs_c + 4)):
            dr = (r - obs_r) * 1.0  # ns_res = 1
            dc = (c - obs_c) * 1.0  # ew_res = 1
            if np.sqrt(dr**2 + dc**2) > max_d:
                continue  # outside circle — correctly INVISIBLE
            np.testing.assert_allclose(
                dist_vals[r, c], full_vals[r, c],
                atol=0.03,
                err_msg=f"Mismatch at ({r},{c})")


def test_viewshed_dask_distance_sweep_target_elev():
    """Tier C distance sweep must not let target_elev contaminate the
    horizon profile.  Regression test: previously the horizon was updated
    with gradients that included target_elev, so intervening terrain
    appeared artificially tall and caused false occlusion."""
    from unittest.mock import patch

    ny, nx = 15, 20
    terrain = np.zeros((ny, nx))
    # Place a ridge at column 12 that is tall enough to block line-of-sight
    # to column 15 when target_elev=0, but not when target_elev=8.
    terrain[:, 12] = 6.0

    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)

    obs_x, obs_y = 5.0, 7.0
    obs_elev = 5

    # Numpy reference: with target_elev=8 the cells behind the ridge at
    # col 15 should be visible (target sticks up above the ridge).
    raster_np = xa.DataArray(terrain.copy(), coords=dict(x=xs, y=ys),
                             dims=["y", "x"])
    v_np = viewshed(raster_np, x=obs_x, y=obs_y,
                    observer_elev=obs_elev, target_elev=8)

    # Sanity: numpy says the cell behind the ridge IS visible.
    assert v_np.values[7, 15] > INVISIBLE, (
        "numpy reference should see cell (7,15) with target_elev=8")

    # Dask Tier C with same parameters
    raster_da = xa.DataArray(
        da.from_array(terrain.copy(), chunks=(5, 5)),
        coords=dict(x=xs, y=ys), dims=["y", "x"])

    with patch('xrspatial.viewshed._available_memory_bytes',
               return_value=10_000):
        v_dask = viewshed(raster_da, x=obs_x, y=obs_y,
                          observer_elev=obs_elev, target_elev=8)

    result = v_dask.values

    # The cell behind the ridge must be visible in Tier C too.
    assert result[7, 15] > INVISIBLE, (
        "Tier C distance sweep should see cell (7,15) with target_elev=8 "
        "(horizon must not include target_elev)")

    # Visible cells should have angles that roughly match numpy.
    vis_mask = (v_np.values > INVISIBLE) & (result > INVISIBLE)
    if vis_mask.any():
        np.testing.assert_allclose(
            result[vis_mask], v_np.values[vis_mask], atol=1.0,
            err_msg="Tier C angles diverge too far from numpy reference")


def test_viewshed_dask_tier_c_warns_and_quantifies_divergence():
    """Tier C (out-of-core distance sweep) is an approximate visibility
    model that does not match the exact numpy sweep cell-for-cell.

    Pins the documented contract for issue #2872:
    1. A UserWarning is emitted when the Tier C path runs, so users feeding
       large dask rasters into downstream visibility logic are not silently
       misled.
    2. The visibility-mask divergence vs the exact sweep is real and
       material on rough terrain (not "minor near occluded boundaries").
    """
    from unittest.mock import patch

    ny, nx = 20, 20
    rng = np.random.RandomState(42)
    terrain = rng.rand(ny, nx) * 10.0
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    obs_x, obs_y = 10.0, 10.0

    # Exact numpy reference.
    raster_np = xa.DataArray(terrain.copy(), coords=dict(x=xs, y=ys),
                             dims=["y", "x"])
    v_np = viewshed(raster_np, x=obs_x, y=obs_y)

    # Force Tier C: shrink the memory budget so Tier B (full compute + R2)
    # is skipped but the output-grid guard still passes.
    raster_da = xa.DataArray(
        da.from_array(terrain.copy(), chunks=(10, 10)),
        coords=dict(x=xs, y=ys), dims=["y", "x"])
    with patch('xrspatial.viewshed._available_memory_bytes',
               return_value=10_000):
        with pytest.warns(UserWarning,
                          match="approximate visibility model"):
            v_dask = viewshed(raster_da, x=obs_x, y=obs_y)

    exact_mask = v_np.values != INVISIBLE
    tier_c_mask = v_dask.values != INVISIBLE
    mismatches = int((exact_mask != tier_c_mask).sum())

    # The exact and Tier C masks must agree on the observer cell.
    obs_r, obs_c = 10, 10
    assert v_dask.values[obs_r, obs_c] == 180.0
    assert v_np.values[obs_r, obs_c] == 180.0

    # Document the divergence: this is the bug being pinned. On this seed
    # the two models disagree on 25 of 400 cells (~6%), well beyond "minor
    # near occluded boundaries". The assertion uses 10 as a floor below the
    # observed 25 to stay robust against small implementation drift. If a
    # future change makes Tier C exact, this assertion should be tightened
    # to parity instead.
    assert mismatches > 0, (
        "Tier C is expected to diverge from the exact sweep on rough "
        "terrain; if it now matches, update this test to assert parity")
    assert mismatches >= 10, (
        f"Tier C divergence ({mismatches} cells, observed 25) is expected "
        f"to be material on this random terrain, not minor")

    # Note: cumulative_viewshed calls viewshed once per observer, so the
    # Tier C warning can fire on each call. Python's default warning filter
    # dedupes by (message, category, module, lineno), so it surfaces once
    # per process rather than once per observer.


def test_viewshed_cpu_memory_guard():
    """_viewshed_cpu should refuse rasters that would blow past RAM.

    Regression test for the DoS vulnerability where the numpy path
    allocated ~500 bytes/pixel of working memory with no size check.
    """
    from unittest.mock import patch

    ny, nx = 100, 100
    terrain = np.zeros((ny, nx))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    raster = xa.DataArray(terrain, coords=dict(x=xs, y=ys), dims=["y", "x"])

    # 100x100 needs ~5 MB of working memory.  Report 1 MB available so
    # the 50% threshold (~500 kB) is exceeded and the guard trips.
    with patch('xrspatial.viewshed._available_memory_bytes',
               return_value=1_000_000):
        with pytest.raises(MemoryError, match="max_distance"):
            viewshed(raster, x=50.0, y=50.0, observer_elev=5)


def test_viewshed_cpu_memory_guard_passes_with_max_distance():
    """max_distance should bypass the memory guard by windowing first."""
    from unittest.mock import patch

    ny, nx = 100, 100
    terrain = np.zeros((ny, nx))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    raster = xa.DataArray(terrain, coords=dict(x=xs, y=ys), dims=["y", "x"])

    # Same tight memory budget as above — but max_distance=3 shrinks the
    # working window to a 7x7 tile, so the guard on the small window
    # does not trip.
    with patch('xrspatial.viewshed._available_memory_bytes',
               return_value=1_000_000):
        v = viewshed(raster, x=50.0, y=50.0, observer_elev=5,
                     max_distance=3.0)
    assert v.values[50, 50] == 180.0


@pytest.mark.parametrize("shape", [(1, 5), (5, 1)])
def test_viewshed_single_row_or_column(shape):
    """A 1xN or Nx1 raster must not divide by zero when computing the
    grid resolution.  Regression test: _viewshed_cpu used to compute
    ew_res/ns_res as range/(dim-1), which is 0/0 = NaN for a degenerate
    axis, silently marking every non-observer cell INVISIBLE on flat
    terrain where they should all be visible."""
    ny, nx = shape
    terrain = np.zeros((ny, nx))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    raster = xa.DataArray(terrain, coords=dict(x=xs, y=ys), dims=["y", "x"])

    obs_x = float(xs[nx // 2])
    obs_y = float(ys[ny // 2])
    v = viewshed(raster, x=obs_x, y=obs_y, observer_elev=5)

    # Observer cell is 180; on flat terrain with an elevated observer
    # every other cell is visible (no NaN poisoning the geometry).
    assert v.values[ny // 2, nx // 2] == 180.0
    assert (v.values > INVISIBLE).all()
    assert not np.isnan(v.values).any()


@pytest.mark.parametrize("shape", [(1, 7), (7, 1)])
def test_viewshed_single_row_or_column_max_distance(shape):
    """The max_distance window path routes a degenerate window back
    through _viewshed_cpu, so it must be guarded too."""
    ny, nx = shape
    terrain = np.zeros((ny, nx))
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    raster = xa.DataArray(terrain, coords=dict(x=xs, y=ys), dims=["y", "x"])

    obs_x = float(xs[nx // 2])
    obs_y = float(ys[ny // 2])
    v = viewshed(raster, x=obs_x, y=obs_y, observer_elev=5,
                 max_distance=2.0)

    assert v.values[ny // 2, nx // 2] == 180.0
    # A cell two steps along the populated axis is within max_distance and
    # visible on flat terrain.
    if nx > 1:
        assert v.values[0, nx // 2 + 2] > INVISIBLE
    else:
        assert v.values[ny // 2 + 2, 0] > INVISIBLE


def _make_raster(backend):
    """Build a small viewshed input raster for the given backend."""
    arr = np.array([
        [0, 0, 1, 0, 0],
        [1, 3, 0, 0, 0],
        [10, 2, 5, 2, -1],
        [11, 1, 2, 9, 0]], dtype=np.float64)
    xs = np.linspace(1, 5, 5)
    ys = np.linspace(1, 4, 4)
    attrs = {'res': (1.0, 1.0), 'crs': 'EPSG:4326'}

    if backend == "numpy":
        data = arr
    elif backend == "dask+numpy":
        data = da.from_array(arr, chunks=(2, 3))
    elif backend == "cupy":
        import cupy as cp
        data = cp.asarray(arr)
    elif backend == "dask+cupy":
        import cupy as cp
        data = da.from_array(cp.asarray(arr), chunks=(2, 3))
    else:
        raise ValueError(backend)

    return xa.DataArray(data, dims=['y', 'x'],
                        coords={'y': ys, 'x': xs},
                        attrs=attrs, name='elevation')


_METADATA_BACKENDS = [
    "numpy",
    "dask+numpy",
    pytest.param("cupy", marks=pytest.mark.skipif(
        not has_rtx(), reason="requires rtxpy for the GPU viewshed path")),
    pytest.param("dask+cupy", marks=pytest.mark.skipif(
        not has_rtx(), reason="requires rtxpy for the GPU viewshed path")),
]


@pytest.mark.parametrize("backend", _METADATA_BACKENDS)
@pytest.mark.parametrize("max_distance", [None, 3.0])
def test_viewshed_output_name_and_dtype_consistent(backend, max_distance):
    """Output .name and dtype must not depend on the backend (#2743).

    The default name is 'viewshed' on every backend, and the output dtype
    is float64 everywhere (the GPU path internally works in float32).
    """
    raster = _make_raster(backend)
    result = viewshed(raster, x=3, y=2, observer_elev=1,
                      max_distance=max_distance)

    assert result.name == "viewshed"
    assert result.dtype == np.float64
    assert result.dims == raster.dims
    # attrs/coords still pass through
    assert result.attrs == raster.attrs
    np.testing.assert_allclose(result.coords['x'].data,
                               raster.coords['x'].data)
    np.testing.assert_allclose(result.coords['y'].data,
                               raster.coords['y'].data)


@pytest.mark.parametrize("backend", _METADATA_BACKENDS)
def test_viewshed_custom_name(backend):
    """A user-supplied name is honoured on every backend."""
    raster = _make_raster(backend)
    result = viewshed(raster, x=3, y=2, observer_elev=1, name="my_vs")
    assert result.name == "my_vs"


@pytest.mark.parametrize("bad", [-1.0, -0.5, float("nan"),
                                 float("inf"), float("-inf")])
def test_viewshed_invalid_max_distance_raises(bad):
    """Negative or non-finite max_distance raises a clear ValueError (#2855).

    Validation lives at the public entry point, before backend dispatch,
    so a single numpy raster covers every backend. Previously these
    values fell through to confusing internal errors (e.g. "zero-size
    array to reduction operation minimum" or "cannot convert float NaN
    to integer").
    """
    raster = _make_raster("numpy")
    with pytest.raises(ValueError, match="max_distance must be a finite"):
        viewshed(raster, x=3, y=2, observer_elev=1, max_distance=bad)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
@pytest.mark.parametrize("good", [0.0, 3.0])
def test_viewshed_valid_max_distance_still_works(backend, good):
    """Finite max_distance >= 0 passes validation and returns a result."""
    raster = _make_raster(backend)
    result = viewshed(raster, x=3, y=2, observer_elev=1, max_distance=good)
    assert result.shape == raster.shape


@pytest.mark.parametrize("param", ["observer_elev", "target_elev"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"),
                                 "tall"])
def test_viewshed_invalid_elev_raises(param, bad):
    """Non-finite or non-numeric observer/target elevation raises a clear
    ValueError naming the offending parameter, before backend dispatch (#2794).

    Validation lives at the public entry point, so a single numpy raster
    covers every backend. Previously NaN/inf fell through to confusing
    downstream errors instead of a message pointing at the bad parameter.
    """
    raster = _make_raster("numpy")
    with pytest.raises(ValueError, match=f"{param} must be a finite number"):
        viewshed(raster, x=3, y=2, **{param: bad})


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_viewshed_invalid_elev_raises_before_dispatch(backend):
    """A dask raster also raises up front, confirming the guard runs before
    any backend dispatch rather than from deep inside the dask path."""
    raster = _make_raster(backend)
    with pytest.raises(ValueError, match="observer_elev must be a finite"):
        viewshed(raster, x=3, y=2, observer_elev=float("nan"))


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
@pytest.mark.parametrize("good", [-5.0, 0.0, 2.5])
def test_viewshed_valid_elev_still_works(backend, good):
    """Finite observer/target elevations (including negative) pass
    validation and return a result of the expected shape."""
    raster = _make_raster(backend)
    result = viewshed(raster, x=3, y=2, observer_elev=good, target_elev=good)
    assert result.shape == raster.shape


# -------------------------------------------------------------------
# dask+cupy backend tests
# -------------------------------------------------------------------

def _dask_cupy_raster(arr_np, xs, ys, chunks, attrs=None):
    """Build a dask-of-cupy DataArray for the requested terrain."""
    import cupy as cp
    data = da.from_array(cp.asarray(arr_np), chunks=chunks)
    return xa.DataArray(data, coords=dict(x=xs, y=ys), dims=["y", "x"],
                        attrs=attrs or {})


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="requires CUDA and cupy")
@pytest.mark.skipif(not has_rtx(), reason="rtxpy not available")
@pytest.mark.parametrize("observer_elev", [5, 2])
@pytest.mark.parametrize("target_elev", [0, 1])
def test_viewshed_dask_cupy_flat(observer_elev, target_elev):
    """dask+cupy on flat terrain should match the analytical angle formula.

    Exercises the dask+cupy dispatch path in ``_viewshed_dask`` (Tier B),
    which was previously never invoked by any test.
    """
    x, y = 0.0, 0.0
    ny, nx = 5, 4
    arr = np.full((ny, nx), 1.3)
    xs = np.arange(nx) * 0.5
    ys = np.arange(ny) * 1.5

    raster = _dask_cupy_raster(arr, xs, ys, chunks=(3, 2))
    v = viewshed(raster, x=x, y=y,
                 observer_elev=observer_elev, target_elev=target_elev)

    result = v.data.compute()
    if hasattr(result, "get"):
        result = result.get()

    xs2, ys2 = np.meshgrid(xs, ys)
    d_vert = observer_elev - target_elev
    d_horz = np.sqrt((xs2 - x) ** 2 + (ys2 - y) ** 2)
    angle = np.rad2deg(np.arctan2(d_horz, d_vert))
    # Don't compare the value under the observer.
    angle[0, 0] = result[0, 0]
    # GPU ray tracing introduces small angular errors on visible cells.
    mask = result < 90
    np.testing.assert_allclose(result[mask], angle[mask], atol=0.03)


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="requires CUDA and cupy")
@pytest.mark.skipif(not has_rtx(), reason="rtxpy not available")
def test_viewshed_dask_cupy_max_distance():
    """dask+cupy with max_distance windows the raster and runs on GPU."""
    ny, nx = 20, 20
    # Flat but non-zero: the RTX mesh builder rejects an all-zero raster
    # (no positive elevation to scale by, see issue #1378).
    arr = np.full((ny, nx), 1.3)
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)

    raster = _dask_cupy_raster(arr, xs, ys, chunks=(10, 10))
    v = viewshed(raster, x=10.0, y=10.0, observer_elev=5, max_distance=5.0)

    result = v.data.compute()
    if hasattr(result, "get"):
        result = result.get()

    assert result[10, 10] == 180.0          # observer cell
    assert result[0, 0] == INVISIBLE        # far outside the radius
    assert result[19, 19] == INVISIBLE
    assert result[10, 12] > INVISIBLE       # 2 cells away, flat, observer up


# -------------------------------------------------------------------
# Metadata preservation across backends
# -------------------------------------------------------------------

def _backend_raster(arr_np, xs, ys, backend, chunks, attrs):
    if backend == "numpy":
        data = arr_np
    elif backend == "cupy":
        import cupy as cp
        data = cp.asarray(arr_np)
    elif backend == "dask+numpy":
        data = da.from_array(arr_np, chunks=chunks)
    elif backend == "dask+cupy":
        import cupy as cp
        data = da.from_array(cp.asarray(arr_np), chunks=chunks)
    else:
        raise ValueError(backend)
    return xa.DataArray(data, coords=dict(x=xs, y=ys), dims=["y", "x"],
                        attrs=attrs)


@pytest.mark.parametrize("backend",
                         ["numpy", "cupy", "dask+numpy", "dask+cupy"])
@pytest.mark.parametrize("max_distance", [None, 2.0])
def test_viewshed_metadata_preserved(backend, max_distance):
    """attrs, coords, and dims survive every backend (and max_distance).

    viewshed reads the x/y coordinates to compute cell resolution, so a
    metadata regression would corrupt the result.  The numpy ``test_viewshed``
    asserts this via ``general_output_checks``; the cupy / dask / dask+cupy
    and max_distance paths previously had no such assertion.
    """
    if "cupy" in backend and not has_cuda_and_cupy():
        pytest.skip("requires CUDA and cupy")
    if "cupy" in backend and not has_rtx():
        pytest.skip("rtxpy not available")

    np.random.seed(7)
    ny, nx = 10, 8
    arr = np.random.uniform(0, 3, (ny, nx)).astype(np.float64)
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    attrs = {"res": (1.0, 1.0), "crs": "EPSG:5070"}

    raster = _backend_raster(arr, xs, ys, backend, chunks=(5, 4), attrs=attrs)
    v = viewshed(raster, x=4.0, y=5.0, observer_elev=10,
                 max_distance=max_distance)

    # attrs / dims / coords preserved regardless of backend.
    assert v.attrs == raster.attrs
    assert v.dims == raster.dims
    assert v.shape == raster.shape
    np.testing.assert_allclose(v.x.values, raster.x.values)
    np.testing.assert_allclose(v.y.values, raster.y.values)

    # Array-type preservation: dask+cupy currently materialises to a
    # numpy-backed dask array (the dask backend computes on CPU and
    # rewraps), so only assert exact type parity for the other backends.
    if backend != "dask+cupy":
        general_output_checks(raster, v)


# -------------------------------------------------------------------
# NaN / nodata input
# -------------------------------------------------------------------

# Every NaN position around the 5x5 grid, including the observer row to the
# right of the observer (2, 4), which used to raise ValueError: node not found.
_NAN_POSITIONS = [
    (1, 1), (0, 0), (4, 4), (2, 4), (2, 0), (0, 2), (4, 2), (3, 4), (2, 3),
]


@pytest.mark.parametrize("nan_pos", _NAN_POSITIONS)
def test_viewshed_nan_input_supported_positions(nan_pos):
    """A single NaN cell at any position runs without error (issue #2857).

    NaN cells generate no sweep events, so the sweep never tries to delete an
    uninserted node.  The NaN cell stays at its INVISIBLE fill value.
    """
    arr = np.full((5, 5), 2.0)
    arr[nan_pos] = np.nan
    xs = np.arange(5.0)
    ys = np.arange(5.0)
    raster = xa.DataArray(arr, coords=dict(x=xs, y=ys), dims=["y", "x"])

    v = viewshed(raster, x=2.0, y=2.0, observer_elev=5)

    # Observer cell is always 180.
    assert v.values[2, 2] == 180.0
    # The NaN cell must read as INVISIBLE so downstream `!= INVISIBLE`
    # visibility checks do not count it as visible.
    assert v.values[nan_pos] == INVISIBLE


def test_viewshed_nan_observer_row_right_does_not_crash():
    """Regression test for issue #2857.

    A NaN on the observer's row to the right of the observer used to raise
    ``ValueError: node not found`` from ``_delete_from_tree`` because the cell
    was never inserted into the status structure yet still emitted an EXITING
    event.
    """
    arr = np.full((5, 5), 2.0)
    arr[2, 4] = np.nan
    xs = np.arange(5.0)
    ys = np.arange(5.0)
    raster = xa.DataArray(arr, coords=dict(x=xs, y=ys), dims=["y", "x"])

    v = viewshed(raster, x=2.0, y=2.0, observer_elev=5)

    assert v.values[2, 2] == 180.0
    assert v.values[2, 4] == INVISIBLE


def test_viewshed_nan_does_not_leak_visibility_to_origin():
    """Skipped NaN cells must not leave events that mark cell (0, 0) visible.

    The event list is pre-allocated; dropping events for NaN cells without
    trimming the unused rows would sort all-zero rows as CENTER events at
    cell (0, 0).  A NaN at the observer cell's neighbour exercises that path.
    """
    arr = np.full((5, 5), 2.0)
    # Several NaNs so multiple event slots go unused.
    arr[0, 1] = np.nan
    arr[1, 0] = np.nan
    arr[3, 3] = np.nan
    xs = np.arange(5.0)
    ys = np.arange(5.0)
    raster = xa.DataArray(arr, coords=dict(x=xs, y=ys), dims=["y", "x"])

    no_nan = xa.DataArray(np.full((5, 5), 2.0), coords=dict(x=xs, y=ys),
                          dims=["y", "x"])

    v = viewshed(raster, x=2.0, y=2.0, observer_elev=5)
    base = viewshed(no_nan, x=2.0, y=2.0, observer_elev=5)

    # (0, 0) is a real cell here, not an artifact: its visibility must match
    # the NaN-free run (the NaNs are elsewhere and don't block the line).
    assert v.values[0, 0] == base.values[0, 0]
    for pos in [(0, 1), (1, 0), (3, 3)]:
        assert v.values[pos] == INVISIBLE


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_viewshed_nan_input_across_backends(backend):
    """NaN handling holds on the CPU-backed backends (numpy, dask Tier B).

    The cupy/RTX path is a separate implementation and is not covered here.
    """
    arr = np.full((5, 5), 2.0)
    arr[2, 4] = np.nan   # observer row, right of observer
    arr[1, 1] = np.nan
    xs = np.arange(5.0)
    ys = np.arange(5.0)

    if backend == "numpy":
        data = arr
    else:
        data = da.from_array(arr, chunks=(3, 2))
    raster = xa.DataArray(data, coords=dict(x=xs, y=ys), dims=["y", "x"])

    v = viewshed(raster, x=2.0, y=2.0, observer_elev=5)
    vals = v.values

    assert vals[2, 2] == 180.0
    assert vals[2, 4] == INVISIBLE
    assert vals[1, 1] == INVISIBLE


@pytest.mark.skipif(not has_rtx(), reason="rtxpy not available")
@pytest.mark.parametrize("fine_axis", ["x", "y"])
def test_viewshed_gpu_anisotropic_matches_cpu(fine_axis):
    """GPU viewshed must agree with the CPU sweep on an anisotropic raster.

    Regression test for issue #2861: the GPU mesh was built on integer grid
    coordinates and ignored ew_res / ns_res, so the ray tracer worked on a
    different terrain shape than the CPU sweep.  The mesh now uses the real
    cell resolution.  On flat terrain the visibility angles are driven
    purely by the horizontal geometry, so a resolution mix-up shows up
    directly as an angle mismatch against the CPU reference.  Both
    anisotropy orientations are checked so an ew_res / ns_res swap is
    caught.
    """
    import cupy as cp

    ny, nx = 9, 9
    terrain = np.full((ny, nx), 1.3)  # flat, positive (RTX needs maxH > 0)
    if fine_axis == "x":
        xs = np.arange(nx, dtype=float) * 1.0    # fine
        ys = np.arange(ny, dtype=float) * 7.0    # coarse
    else:
        xs = np.arange(nx, dtype=float) * 7.0    # coarse
        ys = np.arange(ny, dtype=float) * 1.0    # fine

    obs_x, obs_y, obs_elev = xs[4], ys[4], 5

    cpu = viewshed(
        xa.DataArray(terrain.copy(), coords=dict(x=xs, y=ys), dims=["y", "x"]),
        x=obs_x, y=obs_y, observer_elev=obs_elev).values
    gpu = viewshed(
        xa.DataArray(cp.asarray(terrain.copy()),
                     coords=dict(x=xs, y=ys), dims=["y", "x"]),
        x=obs_x, y=obs_y, observer_elev=obs_elev).data.get()

    # Same set of visible cells.
    assert ((cpu > INVISIBLE) == (gpu > INVISIBLE)).all()

    # Same visibility angles (skip the observer cell, fixed at 180).
    mask = (cpu > INVISIBLE) & (gpu > INVISIBLE)
    mask[4, 4] = False
    np.testing.assert_allclose(gpu[mask], cpu[mask], atol=0.05)


@pytest.mark.parametrize("backend", ["numpy", "cupy", "dask"])
@pytest.mark.parametrize("fine_axis", ["x", "y"])
def test_viewshed_max_distance_anisotropic(backend, fine_axis):
    """max_distance must not clip cells on anisotropic-resolution rasters.

    Regression test: the windowed path sized its analysis window from the
    coarser of ew_res / ns_res and used that single radius for both axes,
    so cells within max_distance along the finer axis were dropped from
    the window and returned INVISIBLE.  Both anisotropy orientations are
    checked so a mix-up between radius_rows / radius_cols is caught.
    """
    if backend == "cupy":
        if not has_rtx():
            pytest.skip("rtxpy not available")
        import cupy as cp

    # Strongly anisotropic: one axis spaced by 1, the other by 10.
    # Use a uniform nonzero elevation: flat enough that all in-range cells
    # stay visible, but the RTX mesh builder needs positive max elevation.
    ny, nx = 21, 21
    terrain = np.full((ny, nx), 1.0)
    if fine_axis == "x":
        xs = np.arange(nx, dtype=float) * 1.0    # fine
        ys = np.arange(ny, dtype=float) * 10.0   # coarse
    else:
        xs = np.arange(nx, dtype=float) * 10.0   # coarse
        ys = np.arange(ny, dtype=float) * 1.0    # fine

    obs_x, obs_y = xs[10], ys[10]
    obs_elev = 50

    base = xa.DataArray(terrain, coords=dict(x=xs, y=ys), dims=["y", "x"])
    full = viewshed(base, x=obs_x, y=obs_y, observer_elev=obs_elev)
    full_vals = full.values

    # max_distance=8 reaches 8 cells along the fine axis but < 1 along the
    # coarse axis, so the buggy single-radius window clipped the fine axis.
    arr = terrain.copy()
    if backend == "cupy":
        arr = cp.asarray(arr)
    raster = xa.DataArray(arr, coords=dict(x=xs, y=ys), dims=["y", "x"])
    if backend == "dask":
        raster = xa.DataArray(
            da.from_array(terrain.copy(), chunks=(7, 7)),
            coords=dict(x=xs, y=ys), dims=["y", "x"])

    v = viewshed(raster, x=obs_x, y=obs_y, observer_elev=obs_elev,
                 max_distance=8.0)
    if backend == "cupy":
        result = v.data.get()
    else:
        # numpy and dask both resolve through .values
        result = v.values

    # Cells within max_distance along the fine axis must be evaluated and
    # match the full viewshed, not be clipped to INVISIBLE.  obs is at
    # index 10 on both axes; step along the fine axis from the observer.
    for offset in (2, 5, 7):  # all < 8 cells along the fine axis
        if fine_axis == "x":
            rc = (10, 10 + offset)
        else:
            rc = (10 + offset, 10)
        assert result[rc] > INVISIBLE, (
            f"cell {rc} ({offset} fine-axis cells away) wrongly clipped")
        np.testing.assert_allclose(result[rc], full_vals[rc], atol=0.03)

    # A cell 9 fine-axis cells away (> max_distance) stays INVISIBLE.
    far = (10, 19) if fine_axis == "x" else (19, 10)
    assert result[far] == INVISIBLE


# -------------------------------------------------------------------
# Input-not-mutated regression tests (issue #2852)
# -------------------------------------------------------------------

def _build_int16_raster(backend):
    """5x5 int16 raster with a single non-zero cell, optionally cupy-backed."""
    ny, nx = 5, 5
    arr = np.zeros((ny, nx), dtype="int16")
    arr[2, 2] = 3
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    if backend == "cupy":
        import cupy as cp
        arr = cp.asarray(arr)
    elif backend == "dask":
        arr = da.from_array(arr, chunks=(3, 3))
    return xa.DataArray(arr, coords=dict(x=xs, y=ys), dims=["y", "x"])


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_viewshed_does_not_mutate_input(backend):
    """viewshed() must not change the caller's input dtype or data (#2852)."""
    raster = _build_int16_raster(backend)
    before = np.asarray(raster.data.compute() if backend == "dask"
                        else raster.data).copy()
    before_dtype = raster.dtype

    viewshed(raster, x=2, y=2, observer_elev=2)

    assert raster.dtype == before_dtype == np.dtype("int16")
    after = np.asarray(raster.data.compute() if backend == "dask"
                       else raster.data)
    np.testing.assert_array_equal(after, before)


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="Requires cupy")
@pytest.mark.skipif(has_rtx(), reason="Tests the no-rtx CuPy CPU fallback")
def test_viewshed_does_not_mutate_cupy_input():
    """CuPy fallback path must leave the caller's CuPy input unchanged (#2852).

    With cupy present but rtxpy absent, viewshed converts to numpy and runs on
    the CPU. The input must stay a CuPy array of its original dtype.
    """
    import cupy as cp
    raster = _build_int16_raster("cupy")
    before = raster.data.get().copy()

    viewshed(raster, x=2, y=2, observer_elev=2)

    assert isinstance(raster.data, cp.ndarray)
    assert raster.dtype == np.dtype("int16")
    np.testing.assert_array_equal(raster.data.get(), before)


# event types from xrspatial.viewshed
ENTERING_EVENT_2793 = 1
EXITING_EVENT_2793 = -1


def _event_positions_around_viewpoint_2793(vp_row, vp_col):
    # The eight neighbours plus the four axis-aligned cells two steps out,
    # so every quadrant/edge branch in _calculate_event_row_col is exercised.
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
    ]
    return [(vp_row + dr, vp_col + dc) for dr, dc in offsets]


def test_calculate_event_row_col_stays_within_one_cell_2793():
    """The corrected bounds guard must hold for every valid branch (#2793).

    _calculate_event_row_col returns a neighbour that should never drift more
    than one cell from the event cell. With the precedence bug fixed, the
    internal guard now enforces this; the assertions below verify the
    invariant directly for all quadrant and edge branches and both event
    types, so the guard cannot fire spuriously.
    """
    # No valid branch produces a drift > 1, so these checks document the
    # invariant the guard enforces; they do not drive the guard's raise.
    vp_row, vp_col = 5, 5
    for event_row, event_col in _event_positions_around_viewpoint_2793(
        vp_row, vp_col
    ):
        for event_type in (ENTERING_EVENT_2793, EXITING_EVENT_2793):
            y, x = _calculate_event_row_col(
                event_type, event_row, event_col, vp_row, vp_col
            )
            assert abs(x - event_col) <= 1
            assert abs(y - event_row) <= 1


def test_calculate_event_row_col_rejects_center_event_2793():
    """CENTER events are not valid input and must raise (#2793)."""
    with pytest.raises(ValueError):
        _calculate_event_row_col(0, 5, 5, 5, 5)


# -------------------------------------------------------------------
# Regular-grid validation (#2789)
# -------------------------------------------------------------------

def _build_grid_2789(backend, xs, ys):
    """5x5 raster with a single peak, given explicit x/y coordinates."""
    arr = np.zeros((len(ys), len(xs)), dtype="float64")
    arr[2, 2] = 5.0
    if backend == "cupy":
        import cupy as cp
        arr = cp.asarray(arr)
    elif backend == "dask":
        arr = da.from_array(arr, chunks=(3, 3))
    return xa.DataArray(arr, coords=dict(x=xs, y=ys), dims=["y", "x"])


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_viewshed_irregular_x_raises(backend):
    """Non-uniform x spacing must raise a clear ValueError naming the axis."""
    ys = np.arange(5, dtype=float)
    xs = np.array([0.0, 1.0, 2.0, 5.0, 6.0])  # gap between index 2 and 3
    raster = _build_grid_2789(backend, xs, ys)
    with pytest.raises(ValueError, match="x coordinates are not uniformly spaced"):
        viewshed(raster, x=2, y=2, observer_elev=2)


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_viewshed_irregular_y_raises(backend):
    """Non-uniform y spacing must raise a clear ValueError naming the axis."""
    xs = np.arange(5, dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    raster = _build_grid_2789(backend, xs, ys)
    with pytest.raises(ValueError, match="y coordinates are not uniformly spaced"):
        viewshed(raster, x=2, y=2, observer_elev=2)


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_viewshed_irregular_with_max_distance_raises(backend):
    """The windowed (max_distance) path is validated too (#2789)."""
    ys = np.arange(5, dtype=float)
    xs = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    raster = _build_grid_2789(backend, xs, ys)
    with pytest.raises(ValueError, match="x coordinates are not uniformly spaced"):
        viewshed(raster, x=2, y=2, observer_elev=2, max_distance=10)


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_viewshed_regular_grid_passes(backend):
    """A regularly spaced grid must not be rejected by the validation."""
    xs = np.arange(5, dtype=float)
    ys = np.arange(5, dtype=float)
    raster = _build_grid_2789(backend, xs, ys)
    result = viewshed(raster, x=2, y=2, observer_elev=2)
    general_output_checks(raster, result)


def test_viewshed_float_roundoff_regular_grid_passes():
    """Coordinates from linspace carry float roundoff but are still regular."""
    xs = np.linspace(-20, 20, 5)
    ys = np.linspace(-20, 20, 5)
    raster = _build_grid_2789("numpy", xs, ys)
    # Should not raise despite the diffs not being bit-identical.
    viewshed(raster, x=0, y=0, observer_elev=2)
