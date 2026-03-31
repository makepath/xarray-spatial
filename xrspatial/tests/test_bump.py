try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import xarray as xr

from xrspatial import bump
from xrspatial.tests.general_checks import (
    cuda_and_cupy_available,
    dask_array_available,
)
from xrspatial.utils import has_cuda_and_cupy


def test_bump():
    bumps = bump(20, 20)
    assert bumps is not None


def test_bump_agg_numpy():
    agg = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    np.random.seed(42)
    result = bump(agg=agg)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (20, 30)
    assert isinstance(result.data, np.ndarray)

    # determinism: same seed → same output
    np.random.seed(42)
    result2 = bump(agg=agg)
    np.testing.assert_array_equal(result.values, result2.values)


@cuda_and_cupy_available
def test_bump_agg_cupy():
    import cupy

    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_cp = agg_np.copy()
    agg_cp.data = cupy.asarray(agg_cp.data)

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_cp = bump(agg=agg_cp)

    assert isinstance(result_cp.data, cupy.ndarray)
    np.testing.assert_array_equal(result_np.values, result_cp.data.get())


@dask_array_available
def test_bump_agg_dask():
    """Single chunk (no internal boundaries) → bitwise match with numpy."""
    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dask = agg_np.copy()
    agg_dask.data = da.from_array(agg_dask.data, chunks=(20, 30))

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_dask = bump(agg=agg_dask)

    assert isinstance(result_dask.data, da.Array)
    np.testing.assert_array_equal(result_np.values, result_dask.values)


@dask_array_available
def test_bump_agg_dask_chunked():
    """Multiple chunks: verify laziness, shape, and approximate values."""
    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dask = agg_np.copy()
    agg_dask.data = da.from_array(agg_dask.data, chunks=(10, 15))

    np.random.seed(42)
    result_np = bump(agg=agg_np, spread=1)

    np.random.seed(42)
    result_dask = bump(agg=agg_dask, spread=1)

    assert isinstance(result_dask.data, da.Array)
    assert result_dask.shape == (20, 30)
    computed = result_dask.values
    # With spread=1, edge effects are minimal; total energy should be close
    np.testing.assert_allclose(computed.sum(), result_np.values.sum(), rtol=0.1)


@dask_array_available
@cuda_and_cupy_available
def test_bump_agg_dask_cupy():
    """Single chunk (no internal boundaries) → bitwise match with numpy."""
    import cupy

    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dc = agg_np.copy()
    agg_dc.data = da.from_array(
        cupy.asarray(agg_dc.data), chunks=(20, 30)
    )

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_dc = bump(agg=agg_dc)

    assert isinstance(result_dc.data, da.Array)
    computed = result_dc.data.compute()
    np.testing.assert_array_equal(result_np.values, computed.get())


def test_bump_preserves_coords():
    ys = np.linspace(0, 1, 10)
    xs = np.linspace(0, 2, 20)
    agg = xr.DataArray(
        np.zeros((10, 20)),
        dims=['lat', 'lon'],
        coords={'lat': ys, 'lon': xs},
    )
    result = bump(agg=agg)
    assert list(result.dims) == ['lat', 'lon']
    np.testing.assert_array_equal(result.coords['lat'].values, ys)
    np.testing.assert_array_equal(result.coords['lon'].values, xs)


def test_bump_agg_infers_shape():
    """When agg is given, width/height are inferred — no need to pass them."""
    agg = xr.DataArray(np.zeros((15, 25)), dims=['y', 'x'])
    np.random.seed(42)
    result = bump(agg=agg)
    assert result.shape == (15, 25)

    # Equivalent to explicit width/height
    np.random.seed(42)
    result2 = bump(width=25, height=15)
    np.testing.assert_array_equal(result.values, result2.values)


def test_bump_decay_strongest_at_center_1102():
    """Pixels adjacent to center should be taller than pixels far from center.

    Regression test for #1102: decay formula was inverted (d2/s instead
    of (s-d2)/s), giving more height to farther pixels.
    """
    from xrspatial.bump import _finish_bump

    locs = np.array([[5, 5]], dtype=np.uint16)
    heights = np.array([10.0])
    out = _finish_bump(11, 11, locs, heights, spread=3)

    center = out[5, 5]
    adjacent = out[5, 6]  # 1 pixel away
    far = out[5, 8]       # 3 pixels away (edge of spread)

    assert center > adjacent > 0, f"center={center}, adjacent={adjacent}"
    assert adjacent > far, f"adjacent={adjacent}, far={far}"


def test_bump_spread_reaches_both_sides_1102():
    """Spread should reach pixels on both sides of center.

    Regression test for #1102: range upper bound excluded x+spread pixel,
    making the bump one pixel short on the positive side.
    """
    from xrspatial.bump import _finish_bump

    locs = np.array([[5, 5]], dtype=np.uint16)
    heights = np.array([10.0])
    out = _finish_bump(11, 11, locs, heights, spread=3)

    # Pixels 1 step away on BOTH sides should be reached
    assert out[5, 4] > 0, "left-1 should be > 0"
    assert out[5, 6] > 0, "right-1 should be > 0"
    assert out[4, 5] > 0, "up-1 should be > 0"
    assert out[6, 5] > 0, "down-1 should be > 0"

    # Pixels well outside spread should be 0
    assert out[5, 9] == 0, "well outside spread should be 0"
    assert out[0, 0] == 0, "corner should be 0"
