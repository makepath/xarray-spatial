import numpy as np
import xarray as xr
import pytest
import warnings

import dask.array as da

from xrspatial import utils
from xrspatial.utils import validate_arrays

try:
    import cupy

    _has_cupy = True
    try:
        cupy.zeros(1)
    except Exception:
        _has_cupy = False
except ImportError:
    cupy = None
    _has_cupy = False


cupy_required = pytest.mark.skipif(
    not _has_cupy, reason="cupy unavailable in this environment"
)


def test_warn_if_unit_mismatch_degrees_horizontal_elevation_vertical(monkeypatch):
    """
    If coordinates look like degrees (lon/lat) and values look like elevation
    (e.g., meters), warn the user about a likely unit mismatch.
    """
    data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)

    # Coordinates in degrees (lon/lat-ish)
    y = np.linspace(5.0, 5.0025, 10)
    x = np.linspace(-74.93, -74.9275, 10)

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs={"units": "m"},  # elevation in meters
    )

    def fake_get_dataarray_resolution(arr):
        return float(x[1] - x[0]), float(y[1] - y[0])

    monkeypatch.setattr(utils, "get_dataarray_resolution", fake_get_dataarray_resolution)

    # Here we *do* expect a warning
    with pytest.warns(UserWarning, match="appears to have coordinates in degrees"):
        utils.warn_if_unit_mismatch(da)


def test_warn_if_unit_mismatch_func_name_in_message(monkeypatch):
    """
    The warning text names the calling function so an aspect() caller is not
    told to fix their `slope` call. Default stays `slope` for back-compat.
    Regression for issue #2782.
    """
    data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
    y = np.linspace(5.0, 5.0025, 10)
    x = np.linspace(-74.93, -74.9275, 10)
    da_2782 = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs={"units": "m"},
    )

    def fake_get_dataarray_resolution(arr):
        return float(x[1] - x[0]), float(y[1] - y[0])

    monkeypatch.setattr(
        utils, "get_dataarray_resolution", fake_get_dataarray_resolution
    )

    with pytest.warns(UserWarning, match="before calling `slope`"):
        utils.warn_if_unit_mismatch(da_2782)

    with pytest.warns(UserWarning, match="before calling `aspect`"):
        utils.warn_if_unit_mismatch(da_2782, func_name="aspect")


def test_warn_if_unit_mismatch_no_warning_for_projected_like_grid(monkeypatch):
    """
    If coordinates look like projected linear units (e.g., meters) and values
    look like elevation, we should NOT warn.
    """
    data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)

    # Coordinates in meters (projected-looking)
    y = np.arange(10) * 30.0              # 0, 30, 60, ...
    x = 500_000.0 + np.arange(10) * 30.0  # UTM-ish eastings

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs={"units": "m"},  # elevation in meters
    )

    def fake_get_dataarray_resolution(arr):
        return float(x[1] - x[0]), float(y[1] - y[0])  # 30 m

    monkeypatch.setattr(utils, "get_dataarray_resolution", fake_get_dataarray_resolution)

    # Capture warnings using the stdlib warnings module
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        utils.warn_if_unit_mismatch(da)

    assert len(w) == 0, "Expected no warnings for projected-like coordinates"


def test_warn_if_unit_mismatch_degrees_but_angle_vertical(monkeypatch):
    """
    If coordinates are in degrees but the DataArray itself looks like an angle
    (e.g., units='degrees'), we should NOT warn; slope/aspect outputs fall into
    this category.
    """
    data = np.linspace(0, 90, 10 * 10, dtype=float).reshape(10, 10)

    # Coordinates in degrees again
    y = np.linspace(5.0, 5.0025, 10)
    x = np.linspace(-74.93, -74.9275, 10)

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs={"units": "degrees"},  # angle, not elevation
    )

    def fake_get_dataarray_resolution(arr):
        return float(x[1] - x[0]), float(y[1] - y[0])

    monkeypatch.setattr(utils, "get_dataarray_resolution", fake_get_dataarray_resolution)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        utils.warn_if_unit_mismatch(da)

    assert len(w) == 0, "Expected no warnings when vertical units are angles"


# ---------------------------------------------------------------------------
# _validate_raster dtype handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_validate_raster_rejects_complex_dtype(dtype):
    """Complex dtypes are not real numeric and must be rejected."""
    raster = xr.DataArray(np.zeros((4, 4), dtype=dtype))
    with pytest.raises(ValueError, match="real numeric"):
        utils._validate_raster(raster, func_name="example")


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8],
)
def test_validate_raster_accepts_real_numeric_dtypes(dtype):
    """Integer and float dtypes pass the default numeric check."""
    raster = xr.DataArray(np.zeros((4, 4), dtype=dtype))
    # Should not raise.
    utils._validate_raster(raster, func_name="example")


def test_validate_arrays_dask_numpy_pair_passes():
    a = xr.DataArray(da.from_array(np.zeros((8, 8)), chunks=4))
    b = xr.DataArray(da.from_array(np.ones((8, 8)), chunks=4))
    validate_arrays(a, b)  # should not raise


def test_validate_arrays_numpy_pair_passes():
    a = xr.DataArray(np.zeros((6, 6)))
    b = xr.DataArray(np.ones((6, 6)))
    validate_arrays(a, b)  # should not raise


def test_validate_arrays_mixed_dask_numpy_and_eager_numpy_rejected():
    a = xr.DataArray(da.from_array(np.zeros((6, 6))))
    b = xr.DataArray(np.ones((6, 6)))
    with pytest.raises(ValueError, match="same backend"):
        validate_arrays(a, b)


@cupy_required
def test_validate_arrays_dask_cupy_pair_passes():
    a = xr.DataArray(da.from_array(cupy.zeros((8, 8)), chunks=4))
    b = xr.DataArray(da.from_array(cupy.ones((8, 8)), chunks=4))
    validate_arrays(a, b)  # should not raise


@cupy_required
def test_validate_arrays_mixed_dask_numpy_and_dask_cupy_rejected():
    a = xr.DataArray(da.from_array(np.zeros((8, 8))))
    b = xr.DataArray(da.from_array(cupy.zeros((8, 8))))
    with pytest.raises(ValueError, match="dask\\+numpy.*dask\\+cupy"):
        validate_arrays(a, b)


@cupy_required
def test_validate_arrays_mixed_eager_numpy_and_cupy_rejected():
    a = xr.DataArray(np.zeros((6, 6)))
    b = xr.DataArray(cupy.zeros((6, 6)))
    with pytest.raises(ValueError, match="numpy.*cupy"):
        validate_arrays(a, b)


# ---------------------------------------------------------------------------
# calc_res irregular-spacing warning (issue #2766)
# ---------------------------------------------------------------------------


def _grid(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return xr.DataArray(
        np.zeros((y.size, x.size)),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )


def test_calc_res_regular_grid_no_warning():
    raster = _grid([0, 1, 2, 3, 4], [0, 1, 2])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xres, yres = utils.calc_res(raster)
    assert (xres, yres) == (1.0, 1.0)
    assert len(w) == 0


def test_calc_res_irregular_x_warns():
    raster = _grid([0, 1, 2, 4, 8], [0, 1, 2])
    with pytest.warns(UserWarning, match="'x' coordinate is not evenly spaced"):
        xres, yres = utils.calc_res(raster)
    # averaged span: (8 - 0) / (5 - 1) == 2.0
    assert xres == 2.0
    assert yres == 1.0


def test_calc_res_irregular_y_warns():
    raster = _grid([0, 1, 2], [0, 1, 3, 7])
    with pytest.warns(UserWarning, match="'y' coordinate is not evenly spaced"):
        utils.calc_res(raster)


def test_calc_res_descending_axis_no_warning():
    # north-up rasters have a descending y axis: regular but negative steps
    raster = _grid([0, 1, 2, 3], [3, 2, 1, 0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xres, yres = utils.calc_res(raster)
    assert (xres, yres) == (1.0, 1.0)
    assert len(w) == 0


def test_calc_res_no_coords_no_warning():
    raster = xr.DataArray(np.zeros((3, 5)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xres, yres = utils.calc_res(raster)
    assert (xres, yres) == (1.0, 1.0)
    assert len(w) == 0


def test_calc_res_float_regular_grid_no_warning():
    # floating-point regular spacing must not trip the relative tolerance
    x = np.linspace(-74.93, -74.9275, 10)
    y = np.linspace(5.0, 5.0025, 10)
    raster = _grid(x, y)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        utils.calc_res(raster)
    assert len(w) == 0


def test_get_dataarray_resolution_irregular_with_res_attr_no_warning():
    # attrs['res'] is honored before calc_res, so no averaging warning fires
    raster = _grid([0, 1, 2, 4, 8], [0, 1, 2])
    raster.attrs["res"] = (1.0, 1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cx, cy = utils.get_dataarray_resolution(raster)
    assert (cx, cy) == (1.0, 1.0)
    assert len(w) == 0


def test_slope_irregular_coords_warns():
    # the user-facing symptom: planar slope averages cell size silently
    from xrspatial import slope

    x = [0, 1, 2, 4, 8]
    data = np.tile(np.asarray(x, dtype=float), (4, 1))
    raster = xr.DataArray(
        data, dims=("y", "x"), coords={"y": [0, 1, 2, 3], "x": x}
    )
    with pytest.warns(UserWarning, match="'x' coordinate is not evenly spaced"):
        slope(raster)
