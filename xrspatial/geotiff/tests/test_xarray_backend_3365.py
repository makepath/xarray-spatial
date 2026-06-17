"""xarray ``BackendEntrypoint`` for the native GeoTIFF reader.

Coverage for issue #3365: ``open_geotiff`` is exposed under xarray's
pluggable backend API so a GeoTIFF / COG / VRT source opens through the
standard entry point::

    xr.open_dataset("dem.tif", engine="xrspatial")

``open_geotiff`` returns a ``DataArray``; the backend promotes it to a
one-variable ``Dataset``. These tests drive the wrapper by passing the
entrypoint class as ``engine=`` so they exercise the worktree code
without depending on the installed ``xarray.backends`` entry point. A
separate test confirms the entry point registers once the package is
installed (the path CI exercises).
"""
from __future__ import annotations

import importlib.metadata
import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._xarray_backend import _DEFAULT_VARIABLE_NAME, GeoTIFFBackendEntrypoint
from xrspatial.geotiff.tests._helpers.tiff_builders import make_minimal_tiff


@pytest.fixture
def geo_tiff_path(tmp_path):
    """Write a stable-codec georeferenced GeoTIFF; yield its path."""
    payload = make_minimal_tiff(
        4, 4, np.dtype("float32"),
        geo_transform=(-120.0, 45.0, 0.001, -0.001),
        epsg=4326,
    )
    path = tmp_path / "backend_3365.tif"
    path.write_bytes(payload)
    return str(path)


def test_open_dataset_returns_one_variable_dataset(geo_tiff_path):
    ds = xr.open_dataset(geo_tiff_path, engine=GeoTIFFBackendEntrypoint)
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) == 1
    # Variable name comes from open_geotiff's default (the source stem).
    assert "backend_3365" in ds.data_vars


def test_dataset_matches_open_geotiff(geo_tiff_path):
    da = open_geotiff(geo_tiff_path)
    ds = xr.open_dataset(geo_tiff_path, engine=GeoTIFFBackendEntrypoint)
    var = ds[da.name]
    np.testing.assert_array_equal(var.values, da.values)
    assert var.dims == da.dims
    # Georeferencing survives the DataArray -> Dataset promotion.
    assert var.attrs.get("crs") == da.attrs.get("crs")
    assert var.attrs.get("transform") == da.attrs.get("transform")
    for coord in da.coords:
        np.testing.assert_array_equal(ds[coord].values, da[coord].values)


def test_file_like_source_falls_back_to_default_name():
    payload = make_minimal_tiff(4, 4, np.dtype("float32"))
    ds = xr.open_dataset(io.BytesIO(payload), engine=GeoTIFFBackendEntrypoint)
    # A file-like source has no path for open_geotiff to derive a name
    # from, so the backend uses its fallback variable name.
    assert _DEFAULT_VARIABLE_NAME in ds.data_vars


def test_backend_kwargs_forwarded_to_open_geotiff(geo_tiff_path):
    # backend_kwargs reach open_geotiff verbatim; default_name renames the
    # resulting data variable, which is an unambiguous signal the kwarg
    # made it through.
    ds = xr.open_dataset(
        geo_tiff_path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"default_name": "elevation"},
    )
    assert "elevation" in ds.data_vars


def test_top_level_chunks_gives_dask_backed_variable(geo_tiff_path):
    pytest.importorskip("dask")
    # ``chunks`` is reserved by xarray's open_dataset and cannot be passed
    # through backend_kwargs; the top-level argument wraps the returned
    # variable in dask.
    ds = xr.open_dataset(
        geo_tiff_path, engine=GeoTIFFBackendEntrypoint, chunks={},
    )
    assert ds["backend_3365"].chunks is not None


def test_drop_variables_removes_the_only_variable(geo_tiff_path):
    ds = xr.open_dataset(
        geo_tiff_path, engine=GeoTIFFBackendEntrypoint,
        drop_variables="backend_3365",
    )
    assert "backend_3365" not in ds.data_vars


def test_open_mfdataset(tmp_path):
    pytest.importorskip("dask")
    paths = []
    for i in range(2):
        payload = make_minimal_tiff(
            4, 4, np.dtype("float32"),
            geo_transform=(-120.0, 45.0, 0.001, -0.001),
            epsg=4326,
        )
        p = tmp_path / f"mf_3365_{i}.tif"
        p.write_bytes(payload)
        paths.append(str(p))

    # A shared default_name keeps every file's data variable identically
    # named, so the files concatenate into one variable along the new
    # dimension. Without it each file's variable takes its own stem and the
    # result has one variable per file instead.
    ds = xr.open_mfdataset(
        paths, engine=GeoTIFFBackendEntrypoint,
        combine="nested", concat_dim="tile",
        backend_kwargs={"default_name": "band_data"},
    )
    assert isinstance(ds, xr.Dataset)
    assert list(ds.data_vars) == ["band_data"]
    assert ds.sizes["tile"] == 2


@pytest.mark.parametrize(
    "name, expected",
    [
        ("dem.tif", True),
        ("dem.TIF", True),
        ("scene.tiff", True),
        ("mosaic.vrt", True),
        ("https://host/path/dem.tif?token=abc", True),
        ("data.nc", False),
        ("notes.txt", False),
        ("noextension", False),
    ],
)
def test_guess_can_open_extensions(name, expected):
    assert GeoTIFFBackendEntrypoint().guess_can_open(name) is expected


def test_guess_can_open_pathlike(tmp_path):
    p = tmp_path / "dem_3365.tif"
    assert GeoTIFFBackendEntrypoint().guess_can_open(p) is True


def test_guess_can_open_non_string_returns_false():
    assert GeoTIFFBackendEntrypoint().guess_can_open(io.BytesIO(b"")) is False


def test_entry_point_registered():
    """The ``xarray.backends`` entry point resolves to the backend class.

    Skips when the installed distribution metadata predates this change
    (e.g. an editable install made before the entry point was added).
    CI installs the branch fresh, so the assertion runs there.
    """
    eps = importlib.metadata.entry_points(group="xarray.backends")
    matches = [ep for ep in eps if ep.name == "xrspatial"]
    if not matches:
        pytest.skip(
            "xrspatial installed without the xarray.backends entry point; "
            "reinstall the package to register the 'xrspatial' engine.")
    assert matches[0].load() is GeoTIFFBackendEntrypoint
