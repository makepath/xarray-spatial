# xrspatial/tests/test_hypsometric_integral.py
try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from .general_checks import create_test_raster

try:
    import cupy as cp
except ImportError:
    cp = None


def _to_numpy(result):
    """Extract numpy array from any backend result."""
    if da and isinstance(result.data, da.Array):
        result = result.compute()
    if cp is not None and isinstance(result.data, cp.ndarray):
        return result.data.get()
    return result.values


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def hi_zones(backend):
    """Two zones (1, 2) plus nodata (0).

    Zone 1: 5 cells — (0,1), (0,2), (0,3), (1,1), (1,2)
    Zone 2: 6 cells — (1,3), (2,1), (2,2), (2,3), (3,1), (3,2)
    Nodata: 4 cells — column 0 and (3,3)
    """
    data = np.array([
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 2, 2, 2],
        [0, 2, 2, 0],
    ], dtype=np.float64)
    return create_test_raster(data, backend, dims=['y', 'x'],
                              attrs={'res': (1.0, 1.0)}, chunks=(2, 2))


@pytest.fixture
def hi_values(backend):
    """Elevation values.

    Zone 1 cells: 10, 20, 30, 40, 50
      min=10, max=50, mean=30, HI=(30-10)/(50-10) = 0.5

    Zone 2 cells: 100, 60, 70, 80, 90, 95
      min=60, max=100, mean=82.5, HI=(82.5-60)/(100-60) = 0.5625
    """
    data = np.array([
        [999., 10., 20., 30.],
        [999., 40., 50., 100.],
        [999., 60., 70., 80.],
        [999., 90., 95., 999.],
    ], dtype=np.float64)
    return create_test_raster(data, backend, dims=['y', 'x'],
                              attrs={'res': (1.0, 1.0)}, chunks=(2, 2))


# --- basic -------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_basic(backend, hi_zones, hi_values):
    from xrspatial.zonal import hypsometric_integral

    result = hypsometric_integral(hi_zones, hi_values)

    assert isinstance(result, xr.DataArray)
    assert result.shape == hi_values.shape
    assert result.dims == hi_values.dims
    assert result.name == 'hypsometric_integral'

    out = _to_numpy(result)

    # zone 0 (nodata) cells should be NaN
    nodata_mask = np.array([
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, True],
    ])
    assert np.all(np.isnan(out[nodata_mask]))

    # zone 1: HI = 0.5
    z1_mask = np.array([
        [False, True, True, True],
        [False, True, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ])
    np.testing.assert_allclose(out[z1_mask], 0.5, rtol=1e-10)

    # zone 2: HI = 0.5625
    z2_mask = np.array([
        [False, False, False, False],
        [False, False, False, True],
        [False, True, True, True],
        [False, True, True, False],
    ])
    np.testing.assert_allclose(out[z2_mask], 0.5625, rtol=1e-10)


# --- edge cases --------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_flat_zone(backend):
    """A zone with all identical values has range=0, so HI should be NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[5.0, 5.0], [5.0, 5.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = _to_numpy(result)
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_nan_in_values(backend):
    """NaN elevation cells should be excluded from per-zone stats."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[10.0, np.nan], [20.0, 30.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = _to_numpy(result)

    # zone 1 finite values: 10, 20, 30 -> HI = (20-10)/(30-10) = 0.5
    # the NaN cell should remain NaN in output
    assert np.isnan(out[0, 1])
    np.testing.assert_allclose(out[0, 0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(out[1, 0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(out[1, 1], 0.5, rtol=1e-10)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_single_cell_zone(backend):
    """A zone with a single cell has range=0, so HI=NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 2]], dtype=np.float64), backend,
        chunks=(1, 2))
    values = create_test_raster(
        np.array([[10.0, 20.0]]), backend,
        chunks=(1, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = _to_numpy(result)
    # single cell -> range=0 -> NaN
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_all_nan_zone(backend):
    """A zone whose elevation cells are all NaN should produce NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [2, 2]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[np.nan, np.nan], [10.0, 20.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = _to_numpy(result)

    # zone 1: all NaN -> NaN
    assert np.all(np.isnan(out[0, :]))
    # zone 2: HI = (15-10)/(20-10) = 0.5
    np.testing.assert_allclose(out[1, :], 0.5, rtol=1e-10)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_nodata_none(backend):
    """When nodata=None, all zone IDs are included (even 0)."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[0, 0], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[10.0, 20.0], [30.0, 40.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=None)
    out = _to_numpy(result)

    # zone 0: HI = (15-10)/(20-10) = 0.5
    np.testing.assert_allclose(out[0, :], 0.5, rtol=1e-10)
    # zone 1: HI = (35-30)/(40-30) = 0.5
    np.testing.assert_allclose(out[1, :], 0.5, rtol=1e-10)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_accessor(backend, hi_zones, hi_values):
    """Verify the .xrs accessor method works."""
    result = hi_values.xrs.zonal_hypsometric_integral(hi_zones)
    assert isinstance(result, xr.DataArray)
    assert result.shape == hi_values.shape

    out = _to_numpy(result)
    z1_mask = np.array([
        [False, True, True, True],
        [False, True, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ])
    np.testing.assert_allclose(out[z1_mask], 0.5, rtol=1e-10)


# --- laziness (issue #2563) --------------------------------------------------

@pytest.mark.skipif(da is None, reason="dask not installed")
def test_hypsometric_integral_dask_returns_lazy(monkeypatch):
    """The dask backend must return a lazy graph — no eager compute on call."""
    import dask
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1, 2], [2, 2, 1]], dtype=np.float64),
        backend='dask+numpy', chunks=(2, 2),
    )
    values = create_test_raster(
        np.array([[10.0, 20, 5], [30, 40, 15]]),
        backend='dask+numpy', chunks=(2, 2),
    )

    calls = {'n': 0}
    orig_compute = dask.compute

    def counting_compute(*args, **kwargs):
        calls['n'] += 1
        return orig_compute(*args, **kwargs)

    monkeypatch.setattr(dask, "compute", counting_compute)

    result = hypsometric_integral(zones, values, nodata=None)

    assert isinstance(result.data, da.Array), \
        f"expected dask.Array, got {type(result.data).__name__}"
    assert calls['n'] == 0, \
        f"hypsometric_integral fired dask.compute {calls['n']} times during construction"


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_hypsometric_integral_dask_matches_numpy():
    """Eager `.compute()` of the dask result must match the numpy backend exactly."""
    from xrspatial.zonal import hypsometric_integral

    zones_arr = np.array([
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 2, 2, 2],
        [0, 2, 2, 0],
    ], dtype=np.float64)
    values_arr = np.array([
        [999., 10., 20., 30.],
        [999., 40., 50., 100.],
        [999., 60., 70., 80.],
        [999., 90., 95., 999.],
    ], dtype=np.float64)

    zones_np = create_test_raster(zones_arr, backend='numpy')
    values_np = create_test_raster(values_arr, backend='numpy')
    zones_d = create_test_raster(zones_arr, backend='dask+numpy', chunks=(2, 2))
    values_d = create_test_raster(values_arr, backend='dask+numpy', chunks=(2, 2))

    out_np = hypsometric_integral(zones_np, values_np).values
    out_dask = hypsometric_integral(zones_d, values_d).compute().values

    np.testing.assert_array_equal(out_np, out_dask)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_hypsometric_integral_dask_empty_lookup_is_lazy(monkeypatch):
    """When every zone is nodata, the result is still a lazy dask array of NaN."""
    import dask
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[0, 0], [0, 0]], dtype=np.float64),
        backend='dask+numpy', chunks=(2, 2),
    )
    values = create_test_raster(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        backend='dask+numpy', chunks=(2, 2),
    )

    calls = {'n': 0}
    orig_compute = dask.compute

    def counting_compute(*args, **kwargs):
        calls['n'] += 1
        return orig_compute(*args, **kwargs)

    monkeypatch.setattr(dask, "compute", counting_compute)

    result = hypsometric_integral(zones, values, nodata=0)

    assert isinstance(result.data, da.Array)
    assert calls['n'] == 0
    out = result.compute().values
    assert np.all(np.isnan(out))


def test_hypsometric_integral_list_of_pairs_zones():
    """Vector zones via list of (geometry, value) pairs."""
    from shapely.geometry import box
    from xrspatial.zonal import hypsometric_integral

    pytest.importorskip("shapely")
    pytest.importorskip("rasterio")

    values_data = np.array([
        [10., 20., 30.],
        [40., 50., 60.],
        [70., 80., 90.],
    ], dtype=np.float64)
    values = xr.DataArray(values_data, dims=['y', 'x'])
    values['y'] = [2.0, 1.0, 0.0]
    values['x'] = [0.0, 1.0, 2.0]
    values.attrs['res'] = (1.0, 1.0)

    # Zone 1 covers left half, zone 2 covers right half
    zones_pairs = [
        (box(-0.5, -0.5, 1.5, 2.5), 1),
        (box(1.5, -0.5, 2.5, 2.5), 2),
    ]

    result = hypsometric_integral(zones_pairs, values, nodata=0)
    assert isinstance(result, xr.DataArray)
    assert result.shape == values.shape


# --- backend-consistency regression (#2525) ---------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_hi_preserves_backend_2525(backend):
    """Regression: output backend must match input backend.

    Before the fix, `_hi_dask_cupy` converted chunks to numpy and called
    `_hi_dask_numpy`, returning a dask+numpy array even when the input was
    dask+cupy. Downstream dispatch via ArrayTypeFunctionMapping then routed
    through the numpy path silently. See issue #2525.
    """
    from xrspatial.utils import (
        has_cuda_and_cupy, has_dask_array,
        is_cupy_array, is_dask_cupy,
    )

    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and not has_dask_array():
        pytest.skip("Requires Dask")

    from xrspatial.zonal import hypsometric_integral

    zones_np = np.array([[1, 1, 2, 2],
                         [1, 1, 2, 2]], dtype=np.int32)
    values_np = np.array([[10.0, 20.0, 30.0, 40.0],
                          [15.0, 25.0, 35.0, 45.0]])

    zones = create_test_raster(zones_np, backend, dims=['y', 'x'],
                               chunks=(2, 2))
    values = create_test_raster(values_np, backend, dims=['y', 'x'],
                                chunks=(2, 2))

    result = hypsometric_integral(zones, values)
    assert isinstance(result, xr.DataArray)
    assert result.shape == values.shape

    if backend == 'numpy':
        assert isinstance(result.data, np.ndarray)
    elif backend == 'cupy':
        assert is_cupy_array(result.data)
    elif backend == 'dask+numpy':
        assert has_dask_array() and isinstance(result.data, da.Array)
        assert not is_dask_cupy(result)
        # confirm chunks are numpy
        sample = result.data.blocks[0, 0].compute()
        assert isinstance(sample, np.ndarray)
    elif backend == 'dask+cupy':
        assert is_dask_cupy(result), (
            "dask+cupy input must produce dask+cupy output (#2525)"
        )
        # confirm chunks are cupy
        sample = result.data.blocks[0, 0].compute()
        assert is_cupy_array(sample)
