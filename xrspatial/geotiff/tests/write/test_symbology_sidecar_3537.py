"""Continuous-raster symbology sidecars across write backends (#3537).

``to_geotiff(..., color_ramp=...)`` writes two sidecars next to a continuous
single-band raster so QGIS opens it with a color ramp: a QGIS ``<base>.qml``
style and ``STATISTICS_*`` in the PAM ``<file>.aux.xml``. The emit is wired into
the shared ``_write_sidecars()`` closure that every write path calls, so these
tests drive the numpy, dask, GPU, and dask+GPU backends to keep the emit (and
the cross-backend statistics) from regressing on any one branch. Unit tests
cover the building blocks (``_finite_stats``, ``build_qml``, ``qml_path``,
``resolve_ramp``, ``build_stats_pam_xml``) and the gating rules.
"""
import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._pam import build_stats_pam_xml
from xrspatial.geotiff._symbology import (
    _finite_stats, build_qml, qml_path, resolve_ramp)

from .._helpers.markers import requires_gpu

pytest.importorskip("tifffile")

# A smooth gradient with one NaN hole, so finite stats must skip the NaN.
_BASE = np.linspace(0.0, 100.0, 16 * 16).reshape(16, 16).astype("float32")
_BASE[0, 0] = np.nan


def _continuous_da(values):
    n = values.shape[0]
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.arange(n, dtype="float64"),
                "x": np.arange(values.shape[1], dtype="float64")},
        attrs={"crs": 4326},
    )


def _qml_renderer(path):
    """Parse a written ``.qml`` and return its <rasterrenderer> element."""
    root = ET.parse(path).getroot()
    return root.find(".//rasterrenderer")


# --------------------------------------------------------------------------
# cross-backend write + parity
# --------------------------------------------------------------------------

def test_numpy_write_emits_both_sidecars(tmp_path):
    path = str(tmp_path / "np.tif")
    to_geotiff(_continuous_da(_BASE), path, color_ramp="viridis")

    # QML uses the extension-replacing name; PAM stats use the appended name.
    assert os.path.exists(str(tmp_path / "np.qml"))
    assert not os.path.exists(path + ".qml")
    assert os.path.exists(path + ".aux.xml")

    rr = _qml_renderer(str(tmp_path / "np.qml"))
    assert rr.get("type") == "singlebandpseudocolor"
    assert rr.get("band") == "1"
    # NaN hole excluded -> min is the second sample, not nan.
    assert float(rr.get("classificationMin")) == pytest.approx(
        float(np.nanmin(_BASE)), abs=1e-5)
    assert float(rr.get("classificationMax")) == pytest.approx(100.0, abs=1e-5)
    # One ramp item per control point.
    items = rr.findall(".//colorrampshader/item")
    assert len(items) == len(resolve_ramp("viridis"))


def test_dask_write_emits_sidecars(tmp_path):
    import dask.array as dsa

    path = str(tmp_path / "dk.tif")
    to_geotiff(_continuous_da(dsa.from_array(_BASE, chunks=(8, 8))),
               path, color_ramp="plasma")
    assert os.path.exists(str(tmp_path / "dk.qml"))
    assert os.path.exists(path + ".aux.xml")


@requires_gpu
def test_gpu_write_emits_sidecars(tmp_path):
    import cupy

    path = str(tmp_path / "gpu.tif")
    to_geotiff(_continuous_da(cupy.asarray(_BASE)), path,
               gpu=True, color_ramp="magma")
    assert os.path.exists(str(tmp_path / "gpu.qml"))
    assert os.path.exists(path + ".aux.xml")


@requires_gpu
def test_dask_gpu_write_emits_sidecars(tmp_path):
    """dask+cupy (gpu=True over a dask array) hits the shared sidecar emit."""
    import cupy
    import dask.array as dsa

    path = str(tmp_path / "dgpu.tif")
    to_geotiff(_continuous_da(dsa.from_array(cupy.asarray(_BASE), chunks=(8, 8))),
               path, gpu=True, color_ramp="magma")
    assert os.path.exists(str(tmp_path / "dgpu.qml"))
    assert os.path.exists(path + ".aux.xml")


def test_finite_stats_backend_parity():
    """Stats agree across numpy / dask / cupy / dask+cupy."""
    import dask.array as dsa

    ref = _finite_stats(_continuous_da(_BASE), None)
    got = _finite_stats(_continuous_da(dsa.from_array(_BASE, chunks=(8, 8))), None)
    assert got == pytest.approx(ref, abs=1e-5)

    pytest.importorskip("cupy")
    import cupy
    from xrspatial.utils import has_cuda_and_cupy
    if not has_cuda_and_cupy():
        pytest.skip("no CUDA device")
    cg = _finite_stats(_continuous_da(cupy.asarray(_BASE)), None)
    cdg = _finite_stats(
        _continuous_da(dsa.from_array(cupy.asarray(_BASE), chunks=(8, 8))), None)
    assert cg == pytest.approx(ref, abs=1e-5)
    assert cdg == pytest.approx(ref, abs=1e-5)


# --------------------------------------------------------------------------
# _finite_stats unit behaviour
# --------------------------------------------------------------------------

def test_finite_stats_values():
    vmin, vmax, vmean, vstd = _finite_stats(_continuous_da(_BASE), None)
    finite = _BASE[np.isfinite(_BASE)]
    assert vmin == pytest.approx(float(finite.min()), abs=1e-5)
    assert vmax == pytest.approx(float(finite.max()), abs=1e-5)
    assert vmean == pytest.approx(float(finite.mean()), abs=1e-5)
    assert vstd == pytest.approx(float(finite.std()), abs=1e-5)  # ddof=0


def test_finite_stats_all_nan_returns_none():
    arr = np.full((4, 4), np.nan, dtype="float32")
    assert _finite_stats(_continuous_da(arr), None) is None


def test_finite_stats_excludes_nodata_sentinel():
    arr = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype="float32")
    vmin, vmax, _, _ = _finite_stats(_continuous_da(arr), nodata=-9999.0)
    assert vmin == pytest.approx(1.0)
    assert vmax == pytest.approx(3.0)


def test_finite_stats_constant_raster():
    arr = np.full((4, 4), 7.0, dtype="float32")
    vmin, vmax, vmean, vstd = _finite_stats(_continuous_da(arr), None)
    assert vmin == vmax == 7.0
    assert vstd == 0.0


# --------------------------------------------------------------------------
# QML / PAM building blocks
# --------------------------------------------------------------------------

def test_build_qml_structure():
    stops = resolve_ramp("viridis")
    xml = build_qml(stops, 10.0, 20.0)
    rr = ET.fromstring(xml).find(".//rasterrenderer")
    assert rr.get("type") == "singlebandpseudocolor"
    assert float(rr.get("classificationMin")) == pytest.approx(10.0)
    assert float(rr.get("classificationMax")) == pytest.approx(20.0)
    items = rr.findall(".//colorrampshader/item")
    assert len(items) == len(stops)
    # First/last item span the data range and carry the ramp endpoints.
    assert float(items[0].get("value")) == pytest.approx(10.0)
    assert float(items[-1].get("value")) == pytest.approx(20.0)
    assert items[0].get("color") == "#%02x%02x%02x" % stops[0][1]
    assert items[-1].get("color") == "#%02x%02x%02x" % stops[-1][1]


def test_qml_path_replaces_extension():
    assert qml_path("/a/b/dem.tif") == "/a/b/dem.qml"
    assert qml_path("/a/b/dem.tiff") == "/a/b/dem.qml"


def test_resolve_ramp_true_is_viridis():
    assert resolve_ramp(True) == resolve_ramp("viridis")
    assert resolve_ramp("VIRIDIS") == resolve_ramp("viridis")


def test_resolve_ramp_unknown_raises():
    with pytest.raises(ValueError, match="unknown color_ramp"):
        resolve_ramp("not-a-ramp")


def test_build_stats_pam_xml_keys():
    stats = {
        "STATISTICS_MINIMUM": 1.5,
        "STATISTICS_MAXIMUM": 9.0,
        "STATISTICS_MEAN": 5.25,
        "STATISTICS_STDDEV": 2.0,
    }
    band = ET.fromstring(build_stats_pam_xml(stats)).find(".//PAMRasterBand")
    assert band.get("band") == "1"
    found = {mdi.get("key"): float(mdi.text)
             for mdi in band.findall("./Metadata/MDI")}
    assert found == pytest.approx(stats)


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------

def test_categorical_wins_no_qml(tmp_path):
    """A categorical raster keeps the RAT sidecar; color_ramp is ignored."""
    cat = xr.DataArray(
        np.array([[0, 1], [1, 0]], dtype="int32"),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        attrs={"crs": 4326, "category_names": ["a", "b"]},
    )
    path = str(tmp_path / "cat.tif")
    to_geotiff(cat, path, color_ramp="viridis")
    assert not os.path.exists(str(tmp_path / "cat.qml"))
    assert "GDALRasterAttributeTable" in open(path + ".aux.xml").read()


def test_multiband_skipped(tmp_path):
    rgb = xr.DataArray(
        np.zeros((3, 4, 4), dtype="float32"),
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": np.arange(4.0), "x": np.arange(4.0)},
        attrs={"crs": 4326},
    )
    path = str(tmp_path / "rgb.tif")
    to_geotiff(rgb, path, color_ramp="viridis")
    assert not os.path.exists(str(tmp_path / "rgb.qml"))
    assert not os.path.exists(path + ".aux.xml")


def test_filelike_destination_ignored():
    buf = io.BytesIO()
    # No path to anchor sidecars; must not raise.
    to_geotiff(_continuous_da(_BASE), buf, color_ramp="viridis")
    assert buf.getbuffer().nbytes > 0


def test_unknown_ramp_raises_before_write(tmp_path):
    path = str(tmp_path / "bad.tif")
    with pytest.raises(ValueError, match="unknown color_ramp"):
        to_geotiff(_continuous_da(_BASE), path, color_ramp="nope")
    assert not os.path.exists(path)


def test_constant_raster_writes_stats_skips_qml(tmp_path):
    arr = np.full((8, 8), 42.0, dtype="float32")
    path = str(tmp_path / "flat.tif")
    to_geotiff(_continuous_da(arr), path, color_ramp="viridis")
    assert os.path.exists(path + ".aux.xml")          # stats still useful
    assert not os.path.exists(str(tmp_path / "flat.qml"))  # degenerate ramp


def test_pack_skips_symbology(tmp_path):
    """``pack=True`` rewrites pixels, so symbology is skipped (would mismatch)."""
    arr = np.linspace(0.0, 10.0, 8 * 8).reshape(8, 8).astype("float32")
    da = xr.DataArray(
        arr, dims=("y", "x"),
        coords={"y": np.arange(8.0), "x": np.arange(8.0)},
        attrs={"crs": 4326, "scale_factor": 0.1, "add_offset": 0.0,
               "mask_and_scale_dtype": "int16", "nodata": -1},
    )
    path = str(tmp_path / "packed.tif")
    to_geotiff(da, path, pack=True, color_ramp="viridis")
    assert not os.path.exists(str(tmp_path / "packed.qml"))
    assert not os.path.exists(path + ".aux.xml")


def test_vrt_write_emits_sidecars(tmp_path):
    """The .vrt write path also emits symbology via the shared closure."""
    import dask.array as dsa

    path = str(tmp_path / "mosaic.vrt")
    to_geotiff(_continuous_da(dsa.from_array(_BASE, chunks=(8, 8))),
               path, color_ramp="cividis")
    assert os.path.exists(str(tmp_path / "mosaic.qml"))
    assert os.path.exists(path + ".aux.xml")


def test_color_ramp_range_sets_bounds(tmp_path):
    path = str(tmp_path / "rng.tif")
    to_geotiff(_continuous_da(_BASE), path,
               color_ramp="viridis", color_ramp_range=(0.0, 50.0))
    rr = _qml_renderer(str(tmp_path / "rng.qml"))
    assert float(rr.get("classificationMin")) == pytest.approx(0.0)
    assert float(rr.get("classificationMax")) == pytest.approx(50.0)
    # range escape hatch writes only min/max stats (no mean/stddev pass).
    aux = open(path + ".aux.xml").read()
    assert "STATISTICS_MINIMUM" in aux and "STATISTICS_MEAN" not in aux


def test_dask_color_ramp_range_sets_bounds(tmp_path):
    """The range escape hatch exists for large dask graphs; assert its bounds
    and the skipped mean/stddev pass land on a dask-backed write."""
    import dask.array as dsa

    path = str(tmp_path / "dk_rng.tif")
    to_geotiff(_continuous_da(dsa.from_array(_BASE, chunks=(8, 8))), path,
               color_ramp="viridis", color_ramp_range=(0.0, 50.0))
    rr = _qml_renderer(str(tmp_path / "dk_rng.qml"))
    assert float(rr.get("classificationMin")) == pytest.approx(0.0)
    assert float(rr.get("classificationMax")) == pytest.approx(50.0)
    aux = open(path + ".aux.xml").read()
    assert "STATISTICS_MINIMUM" in aux and "STATISTICS_MEAN" not in aux


def test_3d_single_band_emits_sidecars(tmp_path):
    """A 3D raster with a length-1 band dim is single-band -> emit symbology."""
    band1 = xr.DataArray(
        _BASE.reshape(1, 16, 16),
        dims=("band", "y", "x"),
        coords={"band": [1], "y": np.arange(16.0), "x": np.arange(16.0)},
        attrs={"crs": 4326},
    )
    path = str(tmp_path / "b1.tif")
    to_geotiff(band1, path, color_ramp="viridis")
    assert os.path.exists(str(tmp_path / "b1.qml"))
    assert os.path.exists(path + ".aux.xml")


def test_nodata_attr_excluded_from_ramp_bounds(tmp_path):
    """attrs['nodata'] (no explicit nodata= kwarg) is excluded from the ramp
    stretch end-to-end, not just in the _finite_stats unit."""
    arr = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype="float32")
    da = _continuous_da(arr)
    da.attrs["nodata"] = -9999.0
    path = str(tmp_path / "nd.tif")
    to_geotiff(da, path, color_ramp="viridis")
    rr = _qml_renderer(str(tmp_path / "nd.qml"))
    assert float(rr.get("classificationMin")) == pytest.approx(1.0)
    assert float(rr.get("classificationMax")) == pytest.approx(3.0)
