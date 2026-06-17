"""Coregistered reads through the xarray backend engine (issue #3376).

The ``xrspatial`` engine forwards to the standalone ``open_geotiff`` by
default, which has no target grid. Passing ``like=`` (a DataArray or
Dataset) routes the read through that object's ``.xrs.open_geotiff``
accessor instead, so ``coregister`` / ``auto_reproject`` / ``resampling``
become available through the standard ``xr.open_dataset`` API. These
tests pin that routing, the parity with the accessor, the variable
naming, the ``open_mfdataset`` composition, and the guard that turns the
opaque ``TypeError`` (from the standalone reader) into a pointed
``ValueError`` when a coregister kwarg arrives without ``like=``.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._xarray_backend import GeoTIFFBackendEntrypoint


def _file_4326(tmp_path, name, nodata=None):
    """Write a 30x30 EPSG:4326 GeoTIFF; return its path."""
    height, width = 30, 30
    arr = np.arange(height * width, dtype=np.float32).reshape(height, width)
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    attrs = {"crs": 4326}
    if nodata is not None:
        attrs["nodata"] = nodata
        arr[14:17, 14:17] = nodata
    da = xr.DataArray(arr, dims=["y", "x"],
                      coords={"y": y, "x": x}, attrs=attrs)
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


def _template_3857(n=6):
    """A grid in a different CRS overlapping the file footprint."""
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = tr.transform(-120.25, 45.25)
    x1, y1 = tr.transform(-119.75, 44.75)
    return xr.DataArray(
        np.zeros((n, n), dtype=np.float32),
        dims=["y", "x"],
        coords={"y": np.linspace(max(y0, y1), min(y0, y1), n),
                "x": np.linspace(min(x0, x1), max(x0, x1), n)},
        attrs={"crs": 3857},
    )


# ---------------------------------------------------------------------------
# like= routes to the coregister path
# ---------------------------------------------------------------------------

def test_coregister_via_engine_matches_template_grid(tmp_path):
    path = _file_4326(tmp_path, "cg_3376_same.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "coregister": True},
    )
    assert isinstance(ds, xr.Dataset)
    var = ds[list(ds.data_vars)[0]]
    assert var.shape == template.shape
    assert np.allclose(var.coords["x"].values, template.coords["x"].values)
    assert np.allclose(var.coords["y"].values, template.coords["y"].values)


def test_coregister_via_engine_matches_accessor(tmp_path):
    # The engine must produce exactly what the accessor produces; the
    # engine only promotes the DataArray to a one-variable Dataset.
    path = _file_4326(tmp_path, "cg_3376_parity.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "coregister": True},
    )
    accessor_da = template.xrs.open_geotiff(path, coregister=True)
    engine_da = ds[list(ds.data_vars)[0]]
    np.testing.assert_array_equal(engine_da.values, accessor_da.values)
    for coord in accessor_da.coords:
        np.testing.assert_array_equal(
            engine_da[coord].values, accessor_da[coord].values)


def test_coregister_via_engine_crs_mismatch(tmp_path):
    path = _file_4326(tmp_path, "cg_3376_mismatch.tif")
    template = _template_3857(6)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "coregister": True},
    )
    var = ds[list(ds.data_vars)[0]]
    assert var.shape == template.shape
    assert np.allclose(var.coords["x"].values, template.coords["x"].values)
    assert np.allclose(var.coords["y"].values, template.coords["y"].values)


def test_auto_reproject_via_engine(tmp_path):
    # auto_reproject (no coregister) keeps the file resolution but still
    # needs the target's bbox/CRS, so it routes through like= too.
    path = _file_4326(tmp_path, "ar_3376.tif")
    template = _template_3857(6)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "auto_reproject": True},
    )
    accessor_da = template.xrs.open_geotiff(path, auto_reproject=True)
    engine_da = ds[list(ds.data_vars)[0]]
    np.testing.assert_array_equal(engine_da.values, accessor_da.values)


def test_dataset_like_with_var(tmp_path):
    # A Dataset target dispatches to the Dataset accessor, which honours
    # the var= kwarg for backend/CRS inference.
    path = _file_4326(tmp_path, "cg_3376_dsvar.tif")
    template = _template_4326(5).to_dataset(name="elev")
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "coregister": True, "var": "elev"},
    )
    accessor_da = template.xrs.open_geotiff(path, coregister=True, var="elev")
    engine_da = ds[list(ds.data_vars)[0]]
    np.testing.assert_array_equal(engine_da.values, accessor_da.values)


# ---------------------------------------------------------------------------
# Variable naming follows the same default_name / stem rule
# ---------------------------------------------------------------------------

def test_coregister_variable_name_follows_stem(tmp_path):
    path = _file_4326(tmp_path, "cg_3376_stem.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "coregister": True},
    )
    assert "cg_3376_stem" in ds.data_vars


def test_coregister_default_name_renames_variable(tmp_path):
    path = _file_4326(tmp_path, "cg_3376_rename.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"like": template, "coregister": True,
                        "default_name": "elevation"},
    )
    assert "elevation" in ds.data_vars


# ---------------------------------------------------------------------------
# open_mfdataset composes onto one shared grid
# ---------------------------------------------------------------------------

def test_open_mfdataset_coregisters_onto_shared_grid(tmp_path):
    pytest.importorskip("dask")
    template = _template_4326(5)
    paths = [
        _file_4326(tmp_path, f"mf_3376_{i}.tif")
        for i in range(2)
    ]
    ds = xr.open_mfdataset(
        paths, engine=GeoTIFFBackendEntrypoint,
        combine="nested", concat_dim="tile",
        backend_kwargs={"like": template, "coregister": True,
                        "default_name": "band_data"},
    )
    assert list(ds.data_vars) == ["band_data"]
    assert ds.sizes["tile"] == 2
    # Every tile sits on the template grid.
    assert np.allclose(ds.coords["x"].values, template.coords["x"].values)
    assert np.allclose(ds.coords["y"].values, template.coords["y"].values)


# ---------------------------------------------------------------------------
# Guard: coregister kwargs without like= raise a pointed ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"coregister": True},
    {"auto_reproject": True},
    {"resampling": "bilinear"},
    {"var": "elev"},
])
def test_coregister_kwarg_without_like_raises(tmp_path, kwargs):
    path = _file_4326(tmp_path, "cg_3376_nolike.tif")
    with pytest.raises(ValueError, match="like="):
        xr.open_dataset(path, engine=GeoTIFFBackendEntrypoint,
                        backend_kwargs=kwargs)


def test_non_array_like_raises_typeerror(tmp_path):
    # A path or other non-array passed as like= would otherwise blow up on
    # `.xrs` with an opaque AttributeError; the guard names the right type.
    path = _file_4326(tmp_path, "cg_3376_badlike.tif")
    with pytest.raises(TypeError, match="DataArray or Dataset"):
        xr.open_dataset(path, engine=GeoTIFFBackendEntrypoint,
                        backend_kwargs={"like": "not_an_array",
                                        "coregister": True})


# ---------------------------------------------------------------------------
# GPU / .vrt rejections are inherited from the accessor path
# ---------------------------------------------------------------------------

def test_coregister_gpu_rejected_through_engine(tmp_path):
    # The unpack-and-reproject pipeline is CPU-only; the accessor raises on
    # gpu=True, and the engine inherits that rejection for free.
    path = _file_4326(tmp_path, "cg_3376_gpu.tif")
    template = _template_4326(5)
    with pytest.raises(ValueError, match="CPU only"):
        xr.open_dataset(
            path, engine=GeoTIFFBackendEntrypoint,
            backend_kwargs={"like": template, "coregister": True,
                            "gpu": True},
        )
