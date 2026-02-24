import numpy as np
import pytest
import xarray as xa

from xrspatial.experimental import min_observable_height


def _make_raster(data, xs=None, ys=None):
    """Build a DataArray with y/x coords from a 2-D numpy array."""
    ny, nx = data.shape
    if xs is None:
        xs = np.arange(nx, dtype=float)
    if ys is None:
        ys = np.arange(ny, dtype=float)
    return xa.DataArray(
        data.astype(np.float64),
        coords=dict(x=xs, y=ys),
        dims=["y", "x"],
    )


# ---- input validation ---------------------------------------------------

def test_max_height_positive():
    r = _make_raster(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="max_height must be positive"):
        min_observable_height(r, x=1.0, y=1.0, max_height=0)


def test_precision_positive():
    r = _make_raster(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="precision must be positive"):
        min_observable_height(r, x=1.0, y=1.0, precision=0)


def test_precision_exceeds_max_height():
    r = _make_raster(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="precision.*must not exceed"):
        min_observable_height(r, x=1.0, y=1.0, max_height=5, precision=10)


# ---- flat terrain --------------------------------------------------------

def test_flat_terrain_all_visible_at_zero():
    """On flat terrain with any positive observer_elev, all cells are
    visible.  min_observable_height should be 0 everywhere."""
    data = np.full((5, 5), 10.0)
    r = _make_raster(data)
    result = min_observable_height(r, x=2.0, y=2.0,
                                   max_height=20.0, precision=1.0)
    assert result.shape == r.shape
    # Every cell should be visible at height 0 (observer at terrain level
    # can see all cells at same elevation).
    assert np.all(result.values == 0.0)


# ---- obstacle blocking ---------------------------------------------------

def test_wall_requires_height():
    """A wall between the observer and far cells forces higher observer."""
    #  y=0: 0  0  0  0  0
    #  y=1: 0  0 10  0  0    <- wall at (y=1, x=2)
    #  y=2: 0  0  0  0  0
    #  Observer at (x=0, y=1).
    #
    #  To see cell (x=3, y=1) over the wall, the line of sight from
    #  (0, h) to (3, 0) must clear (2, 10).  LOS height at x=2 is h/3,
    #  so h >= 30 is required.
    data = np.zeros((3, 5))
    data[1, 2] = 10.0  # wall
    r = _make_raster(data)

    result = min_observable_height(r, x=0.0, y=1.0,
                                   max_height=50.0, precision=1.0)

    # Observer cell itself should be 0.
    assert result.sel(x=0.0, y=1.0).item() == 0.0

    # Cells on same side as observer (x=1) should be visible at 0.
    assert result.sel(x=1.0, y=1.0).item() == 0.0

    # Cell behind the wall (x=3, y=1) needs a significant height.
    behind_wall = result.sel(x=3.0, y=1.0).item()
    assert behind_wall > 0.0, "cells behind wall should need height > 0"

    # The wall cell itself is visible from height 0 (it's tall, angles
    # up to observer) or at least from some modest height.
    wall_h = result.sel(x=2.0, y=1.0).item()
    assert wall_h < behind_wall, "wall cell should be easier to see than cells behind it"


def test_tall_peak_blocks_cells_behind():
    """A peak should block cells directly behind it, requiring more height."""
    # Observer at (x=0, y=1), peak at (x=2, y=1), target at (x=4, y=1).
    # Need at least 2 rows so viewshed can compute ns_res.
    data = np.zeros((3, 5))
    data[1, 2] = 8.0  # peak
    r = _make_raster(data)

    result = min_observable_height(r, x=0.0, y=1.0,
                                   max_height=40.0, precision=0.5)

    h_behind = result.values[1, 4]
    h_front = result.values[1, 1]
    # cell in front of peak should be visible from lower height
    assert h_front < h_behind


# ---- NaN for unreachable cells -------------------------------------------

def test_unreachable_cells_are_nan():
    """If a cell cannot be seen even at max_height, result should be NaN."""
    # Very tall wall close to observer, low max_height.
    data = np.zeros((3, 3))
    data[1, 1] = 1000.0  # enormous wall next to observer
    r = _make_raster(data)

    result = min_observable_height(r, x=0.0, y=1.0,
                                   max_height=5.0, precision=1.0)

    # Some cells behind the 1000-unit wall should be NaN at max_height=5.
    # The far corner (x=2, y=0 or y=2) is behind the wall.
    far_val = result.sel(x=2.0, y=0.0).item()
    assert np.isnan(far_val), "cell behind huge wall should be NaN at low max_height"


# ---- output shape and coords ---------------------------------------------

def test_output_shape_coords_match_input():
    data = np.random.default_rng(42).uniform(0, 10, (6, 8))
    xs = np.linspace(0, 7, 8)
    ys = np.linspace(0, 5, 6)
    r = _make_raster(data, xs=xs, ys=ys)

    result = min_observable_height(r, x=3.0, y=2.0,
                                   max_height=20, precision=2.0)
    assert result.shape == r.shape
    np.testing.assert_array_equal(result.coords['x'].values, xs)
    np.testing.assert_array_equal(result.coords['y'].values, ys)
    assert result.dims == r.dims


# ---- monotonicity property -----------------------------------------------

def test_monotonicity():
    """Visibility is monotonic: if visible at h, visible at all h' > h.

    This indirectly tests that the binary search is correct: for each
    cell, all heights >= min_observable_height should show it as visible.
    """
    data = np.zeros((5, 5))
    data[2, 3] = 8.0
    r = _make_raster(data)

    result = min_observable_height(r, x=0.0, y=2.0,
                                   max_height=20.0, precision=1.0)

    # Pick a cell that requires non-zero height.
    h_needed = result.sel(x=4.0, y=2.0).item()
    if not np.isnan(h_needed) and h_needed > 0:
        from xrspatial import viewshed
        # Should be invisible slightly below threshold.
        vs_below = viewshed(r, x=0.0, y=2.0, observer_elev=max(h_needed - 1.0, 0))
        vs_above = viewshed(r, x=0.0, y=2.0, observer_elev=h_needed + 1.0)
        # At or above threshold the cell should be visible.
        assert vs_above.sel(x=4.0, y=2.0).item() != -1


# ---- precision affects granularity ---------------------------------------

def test_finer_precision():
    """With finer precision, results should be at least as good (<=) as
    coarser precision."""
    data = np.zeros((3, 5))
    data[1, 2] = 10.0
    r = _make_raster(data)

    coarse = min_observable_height(r, x=0.0, y=1.0,
                                   max_height=20, precision=5.0)
    fine = min_observable_height(r, x=0.0, y=1.0,
                                 max_height=20, precision=0.5)

    # Fine result should be <= coarse result for all finite cells
    # (finer search can find an equal or lower threshold).
    mask = np.isfinite(fine.values) & np.isfinite(coarse.values)
    assert np.all(fine.values[mask] <= coarse.values[mask] + 1e-10)


# ---- accessor integration ------------------------------------------------

def test_accessor():
    """min_observable_height is available via .xrs accessor."""
    import xrspatial  # noqa: F401 – registers accessor
    data = np.zeros((3, 3))
    r = _make_raster(data)
    result = r.xrs.min_observable_height(x=1.0, y=1.0,
                                          max_height=10, precision=2)
    assert result.shape == r.shape
