"""Tests for the xarray-spatial input-contract check (`xrs.validate`)."""
import numpy as np
import pytest
import xarray as xr

import xrspatial  # noqa: F401  (registers the .xrs accessor)
from xrspatial.validate import (
    DatasetValidationReport,
    ValidationReport,
    XrsContractError,
    validate,
    validate_dataset,
)

from .general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _compliant(name="dem"):
    """A raster that satisfies the whole contract."""
    data = np.ones((4, 4), dtype="float64")
    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"x": np.arange(4.0), "y": np.arange(4.0)},
        attrs={"crs": "EPSG:4326"},
        name=name,
    )


def _checks(report):
    return {i.check for i in report.issues}


# ---------------------------------------------------------------------------
# Compliant input
# ---------------------------------------------------------------------------

def test_compliant_raster_is_valid():
    report = _compliant().xrs.validate()
    assert isinstance(report, ValidationReport)
    assert report.is_valid is True
    assert bool(report) is True
    assert report.issues == []


# ---------------------------------------------------------------------------
# Error-level checks
# ---------------------------------------------------------------------------

def test_non_dataarray_is_error():
    report = validate([1, 2, 3])
    assert not report
    assert _checks(report) == {"type"}


@pytest.mark.parametrize("shape", [(5,), (2, 2, 2, 2)])
def test_wrong_ndim_is_error(shape):
    da = xr.DataArray(np.zeros(shape, dtype="float64"))
    report = da.xrs.validate()
    assert not report
    assert "ndim" in _checks(report)


def test_string_dtype_is_error():
    da = _compliant()
    da = da.astype("<U5")
    report = da.xrs.validate()
    assert not report
    assert "dtype" in _checks(report)


def test_complex_dtype_is_error():
    da = _compliant().astype("complex128")
    report = da.xrs.validate()
    assert not report
    assert "dtype" in _checks(report)


def test_non_numeric_coords_is_error():
    da = _compliant()
    da = da.assign_coords(x=np.array(["a", "b", "c", "d"]))
    report = da.xrs.validate()
    assert not report
    assert "coords_numeric" in _checks(report)


def test_non_monotonic_coords_is_error():
    da = _compliant()
    da = da.assign_coords(x=np.array([0.0, 2.0, 1.0, 3.0]))
    report = da.xrs.validate()
    assert not report
    assert "monotonic" in _checks(report)


def test_non_finite_cellsize_is_error():
    # A non-finite coordinate makes the inferred cell size non-finite,
    # which planar ops cannot use.
    da = _compliant()
    da = da.assign_coords(y=np.array([0.0, 1.0, 2.0, np.inf]))
    report = da.xrs.validate()
    assert not report
    assert "cellsize" in _checks(report)


# ---------------------------------------------------------------------------
# Warning-level checks
# ---------------------------------------------------------------------------

def test_uneven_spacing_is_warning():
    da = _compliant()
    da = da.assign_coords(x=np.array([0.0, 1.0, 2.0, 5.0]))
    report = da.xrs.validate()
    assert report.is_valid is True  # warning only
    assert "even_spacing" in _checks(report)


def test_missing_coords_is_warning():
    da = xr.DataArray(
        np.zeros((4, 4), dtype="float64"), dims=["y", "x"],
        attrs={"crs": "EPSG:4326"},
    )
    report = da.xrs.validate()
    assert report.is_valid is True
    assert "coords_present" in _checks(report)


def test_unconventional_dim_names_is_warning():
    da = xr.DataArray(
        np.zeros((4, 4), dtype="float64"), dims=["row", "col"],
        coords={"col": np.arange(4.0), "row": np.arange(4.0)},
        attrs={"crs": "EPSG:4326"},
    )
    report = da.xrs.validate()
    assert report.is_valid is True
    assert "spatial_dims" in _checks(report)


def test_projected_latlon_is_warning():
    da = xr.DataArray(
        np.zeros((3, 3), dtype="float64"), dims=["lat", "lon"],
        coords={"lat": [0.0, 1.0, 2.0], "lon": [500.0, 501.0, 502.0]},
        attrs={"crs": "EPSG:4326"},
    )
    report = da.xrs.validate()
    assert report.is_valid is True
    assert "geographic_range" in _checks(report)


def test_bare_latlon_dim_does_not_trigger_geographic_warning():
    # A dim named 'lat' with no real coordinate must not be checked
    # against the geographic range using xarray's synthesized indices.
    da = xr.DataArray(
        np.zeros((100, 4), dtype="float64"), dims=["lat", "lon"],
        attrs={"crs": "EPSG:4326"},
    )
    report = da.xrs.validate()
    assert "geographic_range" not in _checks(report)
    # the real problem (no coords) is still reported
    assert "coords_present" in _checks(report)


def test_missing_crs_is_warning():
    da = _compliant()
    da.attrs = {}
    report = da.xrs.validate()
    assert report.is_valid is True
    assert "crs" in _checks(report)


# ---------------------------------------------------------------------------
# Severity semantics
# ---------------------------------------------------------------------------

def test_warnings_only_is_valid():
    # missing coords + missing crs + odd dim names: all warnings.
    da = xr.DataArray(np.zeros((4, 4), dtype="float64"), dims=["row", "col"])
    report = da.xrs.validate()
    assert report.warnings
    assert not report.errors
    assert report.is_valid is True


def test_one_error_flips_validity():
    da = xr.DataArray(np.zeros((4, 4), dtype="<U3"), dims=["row", "col"])
    report = da.xrs.validate()
    assert report.errors
    assert report.is_valid is False


# ---------------------------------------------------------------------------
# raise_on_error
# ---------------------------------------------------------------------------

def test_raise_on_error_raises_and_names_every_error():
    da = _compliant().astype("<U5").assign_coords(
        x=np.array(["a", "b", "c", "d"])
    )
    report = da.xrs.validate()
    assert {"dtype", "coords_numeric"} <= _checks(report)
    with pytest.raises(XrsContractError) as exc:
        da.xrs.validate(raise_on_error=True)
    msg = str(exc.value)
    # the aggregated message names both error-level issues by their text
    assert "dtype is" in msg
    assert "not numeric" in msg


def test_raise_on_error_no_raise_for_warnings_only():
    da = _compliant()
    da.attrs = {}  # only a crs warning
    # should not raise
    report = da.xrs.validate(raise_on_error=True)
    assert report.is_valid is True


# ---------------------------------------------------------------------------
# Backends — structural check is identical and never materializes data
# ---------------------------------------------------------------------------

def _backend_raster(backend, dtype="float64"):
    data = np.ones((6, 6), dtype=dtype)
    # create_test_raster supplies res + crs attrs and y/x coords.
    return create_test_raster(data, backend=backend, name="dem")


@pytest.mark.parametrize("backend", ["numpy"])
def test_numpy_backend_compliant(backend):
    assert _backend_raster(backend).xrs.validate().is_valid


@dask_array_available
def test_dask_backend_stays_lazy():
    import dask.array as da
    raster = _backend_raster("dask+numpy")
    report = raster.xrs.validate()
    assert report.is_valid
    # validate must not have computed the dask graph.
    assert isinstance(raster.data, da.Array)


@dask_array_available
def test_dask_backend_reports_same_as_numpy():
    np_raster = _backend_raster("numpy", dtype="<U3")
    dask_raster = _backend_raster("dask+numpy", dtype="<U3")
    assert _checks(np_raster.xrs.validate()) == _checks(
        dask_raster.xrs.validate()
    )


@cuda_and_cupy_available
def test_cupy_backend_stays_on_device():
    import cupy
    raster = _backend_raster("cupy")
    report = raster.xrs.validate()
    assert report.is_valid
    assert isinstance(raster.data, cupy.ndarray)


@cuda_and_cupy_available
def test_dask_cupy_backend_compliant():
    raster = _backend_raster("dask+cupy")
    assert raster.xrs.validate().is_valid


# ---------------------------------------------------------------------------
# Dataset accessor
# ---------------------------------------------------------------------------

def test_dataset_per_variable_report():
    good = _compliant("dem")
    bad = good.astype("<U3").rename("labels")
    ds = xr.Dataset({"dem": good, "labels": bad})
    report = ds.xrs.validate()
    assert isinstance(report, DatasetValidationReport)
    assert report.is_valid is False
    assert report.reports["dem"].is_valid is True
    assert report.reports["labels"].is_valid is False
    assert "labels" in repr(report)


def test_dataset_all_compliant_is_valid():
    ds = xr.Dataset({"a": _compliant("a"), "b": _compliant("b")})
    assert ds.xrs.validate().is_valid


def test_dataset_raise_on_error():
    ds = xr.Dataset(
        {"dem": _compliant("dem"), "labels": _compliant("labels").astype("<U3")}
    )
    with pytest.raises(XrsContractError) as exc:
        ds.xrs.validate(raise_on_error=True)
    assert "labels" in str(exc.value)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

def test_text_repr_lists_messages_and_suggestions():
    da = _compliant().astype("<U5")
    report = da.xrs.validate()
    text = repr(report)
    assert "error" in text
    assert "dtype" in text
    assert "Fix:" in text


def test_html_repr_renders_table():
    da = _compliant().astype("<U5")
    html = da.xrs.validate()._repr_html_()
    assert "<table" in html
    assert "ValidationReport" in html


def test_compliant_repr_says_compliant():
    assert "compliant" in repr(_compliant().xrs.validate())


def test_validate_function_matches_accessor():
    da = _compliant().astype("<U5")
    assert _checks(validate(da)) == _checks(da.xrs.validate())
    ds = xr.Dataset({"dem": _compliant()})
    assert validate_dataset(ds).is_valid == ds.xrs.validate().is_valid
