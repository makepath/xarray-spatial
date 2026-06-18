"""Reprojecting reads through the xarray backend engine (issue #3376).

The ``xrspatial`` engine forwards to the standalone ``open_geotiff`` by
default, which has no target grid. Passing the target as the value of
``coregister`` or ``auto_reproject`` (a DataArray or Dataset) routes the
read through that object's ``.xrs.open_geotiff`` accessor instead, so the
reprojecting reads become available through the standard
``xr.open_dataset`` API. These tests pin that routing, the parity with
the accessor, the variable naming, the ``open_mfdataset`` composition,
and the guards that turn an opaque error into a pointed ``ValueError``
when a reprojecting kwarg arrives without a target.
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
# a target on coregister=/auto_reproject= routes to the reprojecting path
# ---------------------------------------------------------------------------

def test_coregister_via_engine_matches_template_grid(tmp_path):
    path = _file_4326(tmp_path, "cg_3376_same.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"coregister": template},
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
        backend_kwargs={"coregister": template},
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
        backend_kwargs={"coregister": template},
    )
    var = ds[list(ds.data_vars)[0]]
    assert var.shape == template.shape
    assert np.allclose(var.coords["x"].values, template.coords["x"].values)
    assert np.allclose(var.coords["y"].values, template.coords["y"].values)


def test_auto_reproject_via_engine(tmp_path):
    # auto_reproject keeps the file resolution but still needs the target's
    # bbox/CRS, so the target rides on auto_reproject= itself.
    path = _file_4326(tmp_path, "ar_3376.tif")
    template = _template_3857(6)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"auto_reproject": template},
    )
    accessor_da = template.xrs.open_geotiff(path, auto_reproject=True)
    engine_da = ds[list(ds.data_vars)[0]]
    np.testing.assert_array_equal(engine_da.values, accessor_da.values)


def test_dataset_target_with_var(tmp_path):
    # A Dataset target dispatches to the Dataset accessor, which honours
    # the var= kwarg for backend/CRS inference.
    path = _file_4326(tmp_path, "cg_3376_dsvar.tif")
    template = _template_4326(5).to_dataset(name="elev")
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"coregister": template, "var": "elev"},
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
        backend_kwargs={"coregister": template},
    )
    assert "cg_3376_stem" in ds.data_vars


def test_coregister_default_name_renames_variable(tmp_path):
    path = _file_4326(tmp_path, "cg_3376_rename.tif")
    template = _template_4326(5)
    ds = xr.open_dataset(
        path, engine=GeoTIFFBackendEntrypoint,
        backend_kwargs={"coregister": template,
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
        backend_kwargs={"coregister": template,
                        "default_name": "band_data"},
    )
    assert list(ds.data_vars) == ["band_data"]
    assert ds.sizes["tile"] == 2
    # Every tile sits on the template grid.
    assert np.allclose(ds.coords["x"].values, template.coords["x"].values)
    assert np.allclose(ds.coords["y"].values, template.coords["y"].values)


# ---------------------------------------------------------------------------
# Guard: reprojecting kwargs without a target raise a pointed ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"coregister": True},
    {"auto_reproject": True},
    {"resampling": "bilinear"},
    {"var": "elev"},
])
def test_reprojecting_kwarg_without_target_raises(tmp_path, kwargs):
    # A bare bool mode or a lone modifier has no grid to read onto.
    path = _file_4326(tmp_path, "cg_3376_notarget.tif")
    with pytest.raises(ValueError, match="target grid"):
        xr.open_dataset(path, engine=GeoTIFFBackendEntrypoint,
                        backend_kwargs=kwargs)


def test_like_kwarg_removed_raises(tmp_path):
    # like= was removed; the engine names the replacement instead of letting
    # the standalone reader raise an opaque TypeError.
    path = _file_4326(tmp_path, "cg_3376_like.tif")
    template = _template_4326(5)
    with pytest.raises(ValueError, match="'like='.*removed"):
        xr.open_dataset(path, engine=GeoTIFFBackendEntrypoint,
                        backend_kwargs={"like": template})


@pytest.mark.parametrize("flag", ["coregister", "auto_reproject"])
def test_falsy_mode_flag_alone_reads_plain(tmp_path, flag):
    # A falsy mode flag means "no reprojecting read"; it must not leak into
    # the standalone reader (which has no such kwarg) as an opaque TypeError.
    path = _file_4326(tmp_path, f"cg_3376_falsy_{flag}.tif")
    ds = xr.open_dataset(path, engine=GeoTIFFBackendEntrypoint,
                         backend_kwargs={flag: False})
    plain = xr.open_dataset(path, engine=GeoTIFFBackendEntrypoint)
    a = ds[list(ds.data_vars)[0]]
    b = plain[list(plain.data_vars)[0]]
    assert a.shape == b.shape
    np.testing.assert_array_equal(a.values, b.values)


def test_target_on_both_modes_raises(tmp_path):
    # A grid on both coregister= and auto_reproject= is ambiguous.
    path = _file_4326(tmp_path, "cg_3376_both.tif")
    template = _template_4326(5)
    with pytest.raises(ValueError, match="exactly one"):
        xr.open_dataset(path, engine=GeoTIFFBackendEntrypoint,
                        backend_kwargs={"coregister": template,
                                        "auto_reproject": template})


def test_non_array_target_raises(tmp_path):
    # A truthy non-array on coregister= is not a grid, so it falls through to
    # the no-target guard rather than blowing up on `.xrs`.
    path = _file_4326(tmp_path, "cg_3376_badtarget.tif")
    with pytest.raises(ValueError, match="target grid"):
        xr.open_dataset(path, engine=GeoTIFFBackendEntrypoint,
                        backend_kwargs={"coregister": "not_an_array"})


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
            backend_kwargs={"coregister": template, "gpu": True},
        )
