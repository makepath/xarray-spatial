import numpy as np
import xarray as xr
import pytest
import warnings


from xrspatial import utils


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
