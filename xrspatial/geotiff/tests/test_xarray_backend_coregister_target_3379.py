"""The reprojecting target rides on the mode parameter's value (#3379).

The ``xrspatial`` engine takes the target grid as the value of
``coregister`` or ``auto_reproject`` (a DataArray or Dataset), so
``coregister=template`` is the whole call -- no separate ``like=`` and no
redundant ``coregister=True``. These tests pin the two target-carrying
modes, the Dataset target with ``var=``, parity with the accessor, and
``open_mfdataset`` composing onto one shared grid.
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


def test_coregister_target_snaps_onto_grid(tmp_path):
    # coregister=<grid> is the whole call: snap onto the grid, matching the
    # accessor's coregister=True read.
    path = _file_4326(tmp_path, "cgt_3379_snap.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"coregister": template},
    )
    var = ds[list(ds.data_vars)[0]]
    accessor_da = template.xrs.open_geotiff(path, coregister=True)
    assert var.shape == template.shape
    assert np.allclose(var.coords["x"].values, template.coords["x"].values)
    assert np.allclose(var.coords["y"].values, template.coords["y"].values)
    np.testing.assert_array_equal(var.values, accessor_da.values)


def test_auto_reproject_target_keeps_native_resolution(tmp_path):
    # auto_reproject=<grid> reprojects onto the target's CRS but keeps the
    # file's resolution, so it does NOT take the template's coarse shape.
    path = _file_4326(tmp_path, "cgt_3379_ar.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"auto_reproject": template},
    )
    var = ds[list(ds.data_vars)[0]]
    accessor_da = template.xrs.open_geotiff(path, auto_reproject=True)
    np.testing.assert_array_equal(var.values, accessor_da.values)
    # Same CRS here, so auto_reproject is a windowed read at native
    # resolution -- not the template's 5x5 grid.
    assert var.shape != template.shape


def test_dataset_target_with_var(tmp_path):
    # A Dataset target dispatches to the Dataset accessor and honours var=.
    path = _file_4326(tmp_path, "cgt_3379_ds.tif")
    template = _template_4326(5).to_dataset(name="elev")
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"coregister": template, "var": "elev"},
    )
    var = ds[list(ds.data_vars)[0]]
    accessor_da = template.xrs.open_geotiff(path, coregister=True, var="elev")
    assert var.shape == template["elev"].shape
    np.testing.assert_array_equal(var.values, accessor_da.values)


def test_open_mfdataset_coregister_target_shares_grid(tmp_path):
    pytest.importorskip("dask")
    template = _template_4326(5)
    paths = [_file_4326(tmp_path, f"cgt_3379_mf_{i}.tif") for i in range(2)]
    ds = xr.open_mfdataset(
        paths, engine=GeoTIFFBackendEntrypoint,
        combine="nested", concat_dim="tile",
        backend_kwargs={"coregister": template, "default_name": "band_data"},
    )
    assert list(ds.data_vars) == ["band_data"]
    assert ds.sizes["tile"] == 2
    assert np.allclose(ds.coords["x"].values, template.coords["x"].values)
    assert np.allclose(ds.coords["y"].values, template.coords["y"].values)
