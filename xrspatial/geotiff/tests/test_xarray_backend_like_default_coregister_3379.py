"""``like=`` defaults to a coregistered read through the engine (#3379).

Supplying a target grid via ``like=`` is the signal to snap onto it, so
the engine assumes ``coregister=True`` when the caller passes neither
``coregister`` nor ``auto_reproject``. These tests pin that default, the
two opt-outs (``coregister=False`` for a plain windowed read,
``auto_reproject=True`` for a reproject without grid snap), and that
``open_mfdataset`` composes onto the shared grid with ``like=`` alone.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._xarray_backend import GeoTIFFBackendEntrypoint


def _file_4326(tmp_path, name):
    """Write a 30x30 EPSG:4326 GeoTIFF; return its path."""
    height, width = 30, 30
    arr = np.arange(height * width, dtype=np.float32).reshape(height, width)
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    da = xr.DataArray(arr, dims=["y", "x"],
                      coords={"y": y, "x": x}, attrs={"crs": 4326})
    path = str(tmp_path / name)
    to_geotiff(da, path, compression="none")
    return path


def _template_4326(n=5):
    """A coarser, offset same-CRS grid inside the file footprint."""
    return xr.DataArray(
        np.zeros((n, n), dtype=np.float32),
        dims=["y", "x"],
        coords={"y": np.linspace(45.3, 44.7, n),
                "x": np.linspace(-120.3, -119.7, n)},
        attrs={"crs": 4326},
    )


def test_like_alone_coregisters_onto_template_grid(tmp_path):
    # like= without coregister snaps onto the template grid, exactly as
    # if coregister=True had been passed explicitly.
    path = _file_4326(tmp_path, "lk_3379_alone.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template},
    )
    var = ds[list(ds.data_vars)[0]]
    assert var.shape == template.shape
    assert np.allclose(var.coords["x"].values, template.coords["x"].values)
    assert np.allclose(var.coords["y"].values, template.coords["y"].values)


def test_like_alone_equals_explicit_coregister(tmp_path):
    # The whole point of the change: like= and like=+coregister=True are
    # the same read.
    path = _file_4326(tmp_path, "lk_3379_redundant.tif")
    template = _template_4326(5)
    implicit = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template},
    )
    explicit = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "coregister": True},
    )
    a = implicit[list(implicit.data_vars)[0]]
    b = explicit[list(explicit.data_vars)[0]]
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.coords["x"].values, b.coords["x"].values)
    np.testing.assert_array_equal(a.coords["y"].values, b.coords["y"].values)


def test_coregister_false_opts_out_to_windowed_read(tmp_path):
    # coregister=False is honoured: a plain windowed read at the file's
    # native resolution, so the result does NOT take the template shape.
    path = _file_4326(tmp_path, "lk_3379_optout.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "coregister": False},
    )
    var = ds[list(ds.data_vars)[0]]
    accessor_da = template.xrs.open_geotiff(path, coregister=False)
    assert var.shape != template.shape
    np.testing.assert_array_equal(var.values, accessor_da.values)


def test_auto_reproject_skips_default_coregister(tmp_path):
    # auto_reproject=True (no coregister) keeps the lighter reproject and
    # must not be silently upgraded to a grid-snapping coregister.
    path = _file_4326(tmp_path, "lk_3379_ar.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "auto_reproject": True},
    )
    var = ds[list(ds.data_vars)[0]]
    accessor_da = template.xrs.open_geotiff(path, auto_reproject=True)
    np.testing.assert_array_equal(var.values, accessor_da.values)
    # Same CRS here, so auto_reproject is a pure windowed read at native
    # resolution -- not the template's coarse grid.
    assert var.shape != template.shape


def test_dataset_like_alone_coregisters(tmp_path):
    # A Dataset target dispatches to the Dataset accessor; the default
    # coregister applies before that dispatch, so a bare Dataset like=
    # snaps onto its grid too.
    path = _file_4326(tmp_path, "lk_3379_ds.tif")
    template = _template_4326(5).to_dataset(name="elev")
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "var": "elev"},
    )
    var = ds[list(ds.data_vars)[0]]
    assert var.shape == template["elev"].shape
    assert np.allclose(var.coords["x"].values,
                       template.coords["x"].values)
    assert np.allclose(var.coords["y"].values,
                       template.coords["y"].values)


def test_open_mfdataset_like_alone_shares_grid(tmp_path):
    pytest.importorskip("dask")
    template = _template_4326(5)
    paths = [_file_4326(tmp_path, f"lk_3379_mf_{i}.tif") for i in range(2)]
    ds = xr.open_mfdataset(
        paths, engine=GeoTIFFBackendEntrypoint,
        combine="nested", concat_dim="tile",
        backend_kwargs={"like": template, "default_name": "band_data"},
    )
    assert list(ds.data_vars) == ["band_data"]
    assert ds.sizes["tile"] == 2
    assert np.allclose(ds.coords["x"].values, template.coords["x"].values)
    assert np.allclose(ds.coords["y"].values, template.coords["y"].values)
