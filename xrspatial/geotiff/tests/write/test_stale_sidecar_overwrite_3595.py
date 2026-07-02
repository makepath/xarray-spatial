"""Overwriting a GeoTIFF refreshes its PAM sidecar (#3595).

``to_geotiff`` writes a PAM ``<file>.aux.xml`` sidecar for categorical
rasters (#3483) and for ``color_ramp=`` statistics (#3537), and
``open_geotiff`` merges that sidecar back onto attrs. Before #3595 a write
that produced no sidecar of its own left a previous write's ``.aux.xml``
on disk, so re-reads attached the overwritten file's ``category_names`` /
``category_colors`` to the new pixels and GDAL/QGIS stretched the new data
with the old statistics. The writer now removes a pre-existing PAM sidecar
on every successful string-path write (matching GDAL's
``GDALDriver::QuietDelete`` behaviour) and re-creates it only when the new
write carries its own categories or statistics.

The QGIS ``.qml`` style sidecar is deliberately NOT removed: QGIS treats it
as user styling that persists across data updates, so only a new
``color_ramp=`` write replaces it.
"""
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

from .._helpers.markers import requires_gpu

pytest.importorskip("tifffile")

_NAMES = ["water", "forest", "urban"]


def _plain_da(dtype="float32"):
    """A continuous 2D DataArray with georef attrs and no categories."""
    data = np.arange(64, dtype=dtype).reshape(8, 8)
    return xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": 8.0 - np.arange(8) - 0.5, "x": np.arange(8) + 0.5},
        attrs={"transform": (1.0, 0.0, 0.0, 0.0, -1.0, 8.0), "crs": 4326},
    )


def _categorical_da():
    """A uint8 categorical DataArray carrying category attrs."""
    da = _plain_da(dtype="uint8")
    da.data[:] = da.data % 3
    da.attrs["category_names"] = list(_NAMES)
    return da


def test_plain_overwrite_removes_categorical_sidecar(tmp_path):
    """A plain overwrite drops the previous write's RAT sidecar."""
    path = str(tmp_path / "overwrite_cat_3595.tif")
    to_geotiff(_categorical_da(), path)
    assert os.path.exists(path + ".aux.xml")

    to_geotiff(_plain_da(), path)

    assert not os.path.exists(path + ".aux.xml")
    back = open_geotiff(path)
    assert "category_names" not in back.attrs
    assert "category_colors" not in back.attrs


def test_plain_overwrite_removes_stats_sidecar_keeps_qml(tmp_path):
    """A plain overwrite drops stale statistics but keeps the .qml style."""
    path = str(tmp_path / "overwrite_ramp_3595.tif")
    qml = os.path.splitext(path)[0] + ".qml"
    to_geotiff(_plain_da(), path, color_ramp="viridis")
    assert os.path.exists(path + ".aux.xml")
    assert os.path.exists(qml)

    to_geotiff(_plain_da() * 1000.0, path)

    assert not os.path.exists(path + ".aux.xml")
    # QGIS user styling persists across data updates by design.
    assert os.path.exists(qml)


def test_categorical_overwrite_replaces_stats_sidecar(tmp_path):
    """A categorical overwrite replaces old statistics with the new RAT."""
    path = str(tmp_path / "ramp_then_cat_3595.tif")
    to_geotiff(_plain_da(), path, color_ramp="viridis")

    to_geotiff(_categorical_da(), path)

    back = open_geotiff(path)
    assert back.attrs["category_names"] == _NAMES
    with open(path + ".aux.xml") as fh:
        assert "STATISTICS_MINIMUM" not in fh.read()


def test_color_ramp_overwrite_replaces_categorical_sidecar(tmp_path):
    """A color_ramp overwrite replaces the old RAT with fresh statistics."""
    path = str(tmp_path / "cat_then_ramp_3595.tif")
    to_geotiff(_categorical_da(), path)

    to_geotiff(_plain_da(), path, color_ramp="viridis")

    back = open_geotiff(path)
    assert "category_names" not in back.attrs
    with open(path + ".aux.xml") as fh:
        text = fh.read()
    assert "STATISTICS_MINIMUM" in text
    assert "CategoryNames" not in text


def test_multiband_color_ramp_overwrite_removes_sidecar(tmp_path):
    """Symbology no-ops on multiband, but the stale sidecar still goes away."""
    path = str(tmp_path / "cat_then_multiband_3595.tif")
    to_geotiff(_categorical_da(), path)

    band = _plain_da()
    multi = xr.concat([band, band + 1.0], dim="band")
    multi = multi.assign_coords(band=[1, 2])
    multi.attrs = dict(band.attrs)
    to_geotiff(multi, path, color_ramp="viridis")

    assert not os.path.exists(path + ".aux.xml")
    assert "category_names" not in open_geotiff(path).attrs


def test_foreign_sidecar_removed_on_fresh_write(tmp_path):
    """A pre-existing sidecar at a fresh output path is removed too."""
    path = str(tmp_path / "fresh_3595.tif")
    with open(path + ".aux.xml", "w") as fh:
        fh.write("<PAMDataset><PAMRasterBand band=\"1\"><CategoryNames>"
                 "<Category>stale</Category></CategoryNames>"
                 "</PAMRasterBand></PAMDataset>\n")

    to_geotiff(_plain_da(), path)

    assert not os.path.exists(path + ".aux.xml")
    assert "category_names" not in open_geotiff(path).attrs


def test_bare_ndarray_overwrite_removes_sidecar(tmp_path):
    """An ndarray (non-DataArray) overwrite also refreshes the sidecar."""
    path = str(tmp_path / "ndarray_3595.tif")
    to_geotiff(_categorical_da(), path)

    to_geotiff(np.arange(64, dtype="float32").reshape(8, 8), path)

    assert not os.path.exists(path + ".aux.xml")


def test_dask_overwrite_removes_sidecar(tmp_path):
    """The dask streaming write path refreshes the sidecar."""
    import dask.array as dsa

    path = str(tmp_path / "dask_3595.tif")
    to_geotiff(_categorical_da(), path)

    plain = _plain_da()
    plain = plain.copy(data=dsa.from_array(plain.data, chunks=(4, 4)))
    to_geotiff(plain, path)

    assert not os.path.exists(path + ".aux.xml")


def test_vrt_write_removes_stale_sidecar(tmp_path):
    """The VRT write path refreshes the sidecar next to the .vrt index.

    The VRT writer refuses to overwrite an existing tiles directory, so the
    stale sidecar here comes from a foreign/previous file at the same path
    rather than a same-path overwrite.
    """
    path = str(tmp_path / "mosaic_3595.vrt")
    with open(path + ".aux.xml", "w") as fh:
        fh.write("<PAMDataset><PAMRasterBand band=\"1\"><CategoryNames>"
                 "<Category>stale</Category></CategoryNames>"
                 "</PAMRasterBand></PAMDataset>\n")

    to_geotiff(_plain_da(), path)

    assert not os.path.exists(path + ".aux.xml")
    assert "category_names" not in open_geotiff(path).attrs


def test_failed_write_keeps_old_sidecar(tmp_path, monkeypatch):
    """A failed write leaves the old file and its sidecar consistent.

    The removal runs in ``_write_sidecars``, which only executes at the
    success return points, so a write that raises mid-pixel-write must not
    strip the sidecar that still describes the untouched old file. This
    pins that ordering against a refactor that moves the removal before
    the pixel write.
    """
    path = str(tmp_path / "failed_3595.tif")
    to_geotiff(_categorical_da(), path)
    assert os.path.exists(path + ".aux.xml")

    from xrspatial.geotiff._writers import eager as eager_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated mid-write failure")

    monkeypatch.setattr(eager_mod, "write", _boom)
    with pytest.raises(RuntimeError, match="simulated mid-write"):
        to_geotiff(_plain_da(), path)

    assert os.path.exists(path + ".aux.xml")
    assert open_geotiff(path).attrs["category_names"] == _NAMES


@requires_gpu
def test_gpu_overwrite_removes_sidecar(tmp_path):
    """The GPU (nvCOMP) write path refreshes the sidecar."""
    import cupy

    path = str(tmp_path / "gpu_3595.tif")
    to_geotiff(_categorical_da(), path)

    plain = _plain_da()
    plain = plain.copy(data=cupy.asarray(plain.data))
    to_geotiff(plain, path, gpu=True)

    assert not os.path.exists(path + ".aux.xml")
