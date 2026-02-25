import dask.array as da
import datashader as ds
import numpy as np
import pandas as pd
import pytest
import xarray as xa

from xrspatial import viewshed
from xrspatial.tests.general_checks import general_output_checks
from xrspatial.viewshed import INVISIBLE

from ..gpu_rtx import has_rtx


@pytest.fixture
def empty_agg():
    # create an empty image of size 5*5
    H = 5
    W = 5

    canvas = ds.Canvas(plot_width=W, plot_height=H,
                       x_range=(-20, 20), y_range=(-20, 20))

    empty_df = pd.DataFrame({
       'x': np.random.normal(.5, 1, 0),
       'y': np.random.normal(.5, 1, 0)
    })
    agg = canvas.points(empty_df, 'x', 'y')
    return agg.astype('i8')


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
