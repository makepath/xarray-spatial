"""Split overloaded ``masked_nodata`` into separate lifecycle signals (#2135).

``attrs['masked_nodata']`` was a single boolean trying to describe a
multi-stage process (declared sentinel exists, masking step ran,
sentinel pixels actually present, dtype cast after masking). Issue
#2135 keeps the existing flag for backward compatibility and adds two
additive lifecycle attrs:

* ``nodata_pixels_present`` -- bool, ``True`` iff at least one pixel in
  the read window matched the declared sentinel before masking. Lets
  consumer code answer "any nodata in this tile" without rescanning.
  The dask path leaves this attr unset because a strict per-chunk
  reduction would force eager ``.compute()``.
* ``nodata_dtype_cast`` -- string dtype name (e.g. ``"float64"``),
  only emitted when the caller passed an explicit ``dtype=`` kwarg.
  Distinguishes float-because-masked from float-because-promoted.

These tests cover the eager / dask / GPU / VRT paths and pin the
emission rules so future changes do not silently drop or rename keys.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, read_geotiff_dask, to_geotiff


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


def _make_float_raster(path, sentinel=-9999.0, plant_sentinel=True):
    """Float32 raster: 2x3 with one (or zero) sentinel pixels."""
    if plant_sentinel:
        data = np.array(
            [[1.0, 2.0, sentinel], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
    else:
        data = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
    da = xr.DataArray(
        data,
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': sentinel},
    )
    to_geotiff(da, path)
    return da


def _make_int_raster(path, sentinel=30, plant_sentinel=True):
    """Int16 raster with sentinel optionally embedded."""
    if plant_sentinel:
        data = np.array([[10, 20, sentinel], [40, 50, 60]], dtype=np.int16)
    else:
        data = np.array([[10, 20, 25], [40, 50, 60]], dtype=np.int16)
    da = xr.DataArray(
        data,
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': sentinel},
    )
    to_geotiff(da, path)
    return da


# --- Eager numpy path ---------------------------------------------------


def test_eager_float_sentinel_present_masked(tmp_path):
    """Float file + sentinel embedded + mask_nodata=True:
    nodata_pixels_present=True, nodata_dtype_cast absent."""
    path = str(tmp_path / "tmp_2135_eager_float_present.tif")
    _make_float_raster(path)
    out = open_geotiff(path)
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is True
    assert 'nodata_dtype_cast' not in out.attrs


def test_eager_float_sentinel_absent_masked(tmp_path):
    """Float file + sentinel NOT embedded + mask_nodata=True:
    nodata_pixels_present=False."""
    path = str(tmp_path / "tmp_2135_eager_float_absent.tif")
    _make_float_raster(path, plant_sentinel=False)
    out = open_geotiff(path)
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is False


def test_eager_float_sentinel_present_unmasked(tmp_path):
    """Float file + sentinel embedded + mask_nodata=False:
    masking branch skipped but presence scan still runs."""
    path = str(tmp_path / "tmp_2135_eager_float_present_unmasked.tif")
    _make_float_raster(path)
    out = open_geotiff(path, mask_nodata=False)
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is True


def test_eager_int_sentinel_present(tmp_path):
    """Int file + sentinel embedded + mask_nodata=True:
    promotion fires, nodata_pixels_present=True."""
    path = str(tmp_path / "tmp_2135_eager_int_present.tif")
    _make_int_raster(path)
    out = open_geotiff(path)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is True


def test_eager_int_out_of_range_sentinel(tmp_path):
    """Int (uint16) file + sentinel out of range:
    no cast, nodata_pixels_present=False."""
    da = xr.DataArray(
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint16),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': -9999},
    )
    path = str(tmp_path / "tmp_2135_eager_int_oor.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('nodata') == -9999
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is False


def test_eager_dtype_cast_records_target(tmp_path):
    """``dtype=`` kwarg surfaces as nodata_dtype_cast."""
    path = str(tmp_path / "tmp_2135_eager_dtype_cast.tif")
    _make_int_raster(path)
    out = open_geotiff(path, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
    # Literal sentinel still in buffer (cast, not masked).
    assert 30.0 in out.values
    # Pixel-presence scan should still confirm the sentinel is there.
    assert out.attrs.get('nodata_pixels_present') is True


def test_eager_dtype_cast_absent_without_dtype_kwarg(tmp_path):
    """No ``dtype=`` kwarg: ``nodata_dtype_cast`` absent from attrs."""
    path = str(tmp_path / "tmp_2135_eager_no_dtype.tif")
    _make_float_raster(path)
    out = open_geotiff(path)
    assert 'nodata_dtype_cast' not in out.attrs


def test_eager_no_declared_sentinel(tmp_path):
    """File without GDAL_NODATA: no nodata-related attrs surface."""
    da = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2135_eager_no_sentinel.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert 'nodata' not in out.attrs
    assert 'masked_nodata' not in out.attrs
    assert 'nodata_pixels_present' not in out.attrs
    assert 'nodata_dtype_cast' not in out.attrs


# --- Dask path ----------------------------------------------------------


def test_dask_leaves_pixels_present_unset(tmp_path):
    """Dask path: per-chunk reduction would force eager compute, so
    ``nodata_pixels_present`` stays unset by design (#2135)."""
    path = str(tmp_path / "tmp_2135_dask_present.tif")
    _make_float_raster(path)
    out = read_geotiff_dask(path, chunks=2)
    assert out.attrs.get('masked_nodata') is True
    assert 'nodata_pixels_present' not in out.attrs


def test_dask_dtype_cast_records_target(tmp_path):
    """Dask path emits ``nodata_dtype_cast`` when caller passes dtype=."""
    path = str(tmp_path / "tmp_2135_dask_cast.tif")
    _make_int_raster(path)
    out = read_geotiff_dask(
        path, chunks=2, mask_nodata=False, dtype=np.float64,
    )
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
    assert 'nodata_pixels_present' not in out.attrs


def test_dask_no_dtype_cast_attr_absent(tmp_path):
    """Dask path without dtype=: nodata_dtype_cast absent."""
    path = str(tmp_path / "tmp_2135_dask_no_cast.tif")
    _make_float_raster(path)
    out = read_geotiff_dask(path, chunks=2)
    assert 'nodata_dtype_cast' not in out.attrs


# --- VRT path -----------------------------------------------------------


def _write_int_vrt(tmp_path, src_basename, vrt_basename, sentinel=30,
                   plant_sentinel=True):
    tifffile = pytest.importorskip("tifffile")
    src = str(tmp_path / src_basename)
    if plant_sentinel:
        data = np.array([[10, 20, sentinel], [40, 50, 60]], dtype=np.int16)
    else:
        data = np.array([[10, 20, 25], [40, 50, 60]], dtype=np.int16)
    tifffile.imwrite(src, data, metadata=None)
    vrt = str(tmp_path / vrt_basename)
    vrt_xml = f"""<VRTDataset rasterXSize="3" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Int16" band="1">
    <NoDataValue>{sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="3" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="3" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
    with open(vrt, 'w') as fh:
        fh.write(vrt_xml)
    return vrt


def test_vrt_int_sentinel_present_masked(tmp_path):
    """VRT int source + sentinel embedded + mask_nodata=True:
    helper promotes to float, nodata_pixels_present=True."""
    vrt = _write_int_vrt(
        tmp_path, "tmp_2135_vrt_src.tif", "tmp_2135_vrt_present.vrt",
    )
    out = open_geotiff(vrt)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is True


def test_vrt_int_sentinel_absent_masked(tmp_path):
    """VRT int source + sentinel NOT embedded + mask_nodata=True:
    helper does not promote, nodata_pixels_present=False."""
    vrt = _write_int_vrt(
        tmp_path, "tmp_2135_vrt_src_absent.tif",
        "tmp_2135_vrt_absent.vrt",
        plant_sentinel=False,
    )
    out = open_geotiff(vrt)
    assert out.dtype.kind == 'i'  # no promotion
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is False


def test_vrt_int_unmasked_still_scans(tmp_path):
    """VRT int + mask_nodata=False: presence scan still runs."""
    vrt = _write_int_vrt(
        tmp_path, "tmp_2135_vrt_src_unmasked.tif",
        "tmp_2135_vrt_unmasked.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False)
    assert out.dtype.kind == 'i'
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is True


def test_vrt_dtype_cast_records_target(tmp_path):
    """VRT + dtype=float64 + mask_nodata=False: cast attr surfaces."""
    vrt = _write_int_vrt(
        tmp_path, "tmp_2135_vrt_src_cast.tif",
        "tmp_2135_vrt_cast.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
    assert out.attrs.get('nodata_pixels_present') is True


# --- GPU path -----------------------------------------------------------


@_gpu_only
def test_gpu_float_sentinel_present_masked(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2135_gpu_float_present.tif")
    _make_float_raster(path)
    out = read_geotiff_gpu(path)
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is True


@_gpu_only
def test_gpu_int_sentinel_absent(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2135_gpu_int_absent.tif")
    _make_int_raster(path, plant_sentinel=False)
    out = read_geotiff_gpu(path)
    # No sentinel pixel: helper short-circuits, buffer stays int.
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is False


@_gpu_only
def test_gpu_dtype_cast_records_target(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2135_gpu_cast.tif")
    _make_int_raster(path)
    out = read_geotiff_gpu(path, mask_nodata=False, dtype=np.float64)
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
