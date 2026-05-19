"""``attrs['masked_nodata']`` must reflect whether masking ran (#2092).

The pre-fix ``_set_nodata_attrs`` set ``attrs['masked_nodata']``
purely from the final array dtype: any float output got ``True``.
With ``mask_nodata=False`` on a float file with a non-NaN sentinel
(e.g. -9999), the masking step is skipped, the buffer keeps the
literal sentinel pixels, but the attr still claimed ``True``. Any
downstream code that trusted the attr ("NaN means missing, sentinels
have been replaced") then treated the -9999 pixels as already-masked
valid data.

The fix threads the actual masking decision through to the helper
and computes it as ``mask_nodata and final_dtype.kind == 'f'`` for
the eager / dask / GPU paths. VRT keeps the dtype-driven rule
because its internal reader inlines float NaN-masking
unconditionally; with ``mask_nodata=False`` on an integer source
the dtype stays integer and the rule correctly reports False.

These tests cover eager numpy, dask, VRT, and the GPU path (gated
on cuda) for both directions: literal sentinel preserved when
masking is opt-out, NaN replacement when masking is opt-in.
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


def _make_float_raster_with_nodata(path):
    """Float32 raster with -9999 sentinel embedded in two cells."""
    da = xr.DataArray(
        np.array(
            [[1.0, 2.0, -9999.0],
             [4.0, -9999.0, 6.0]],
            dtype=np.float32,
        ),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': -9999.0},
    )
    to_geotiff(da, path)
    return da


# --- Eager numpy path ---------------------------------------------------


def test_eager_mask_nodata_false_reports_false(tmp_path):
    """Float file + nodata=-9999 + mask_nodata=False: buffer keeps
    literal sentinel pixels; attr must say False so downstream code
    knows the sentinel is still present (#2092 bug case)."""
    path = str(tmp_path / "tmp_2092_eager_unmasked.tif")
    _make_float_raster_with_nodata(path)

    out = open_geotiff(path, mask_nodata=False)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is False, (
        f"buffer holds literal -9999 pixels but attrs say "
        f"masked_nodata={out.attrs.get('masked_nodata')!r}")
    # And the literal sentinel really is in the buffer.
    assert -9999.0 in out.values


def test_eager_mask_nodata_true_reports_true(tmp_path):
    """Default behaviour: float file + nodata=-9999 + mask_nodata=True
    replaces -9999 with NaN. Attr must say True. Pin the canonical
    direction so the fix doesn't accidentally flip it everywhere."""
    path = str(tmp_path / "tmp_2092_eager_masked.tif")
    _make_float_raster_with_nodata(path)

    out = open_geotiff(path)  # mask_nodata defaults to True
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is True
    # -9999 replaced with NaN.
    assert np.isnan(out.values).sum() == 2
    assert -9999.0 not in out.values[~np.isnan(out.values)]


def test_eager_int_file_mask_nodata_true_no_match_reports_false(tmp_path):
    """Int file + mask_nodata=True but sentinel is out of range, so
    no pixel matches and no cast to float happens. Buffer stays int
    with literal sentinel; attr must say False. The pre-fix rule
    already got this right (dtype is int -> False); regression
    check that the new rule doesn't break it."""
    # uint16 file with a -9999 declared sentinel that cannot match
    # any pixel (uint16 range is 0..65535).
    da = xr.DataArray(
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint16),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': -9999},
    )
    path = str(tmp_path / "tmp_2092_int_oor_sentinel.tif")
    to_geotiff(da, path)

    out = open_geotiff(path)
    assert out.attrs.get('nodata') == -9999
    # No pixel matched, no cast, buffer stays uint16.
    assert out.dtype.kind == 'u'
    assert out.attrs.get('masked_nodata') is False


def test_eager_explicit_float_dtype_mask_off_reports_false(tmp_path):
    """Eager path: int file + mask_nodata=False + dtype=float64.
    The mask block at __init__.py is gated on ``mask_nodata``, then
    the explicit ``dtype=`` cast promotes the int buffer to float
    with literal sentinels still in it. The rule must report False
    (mask_nodata short-circuits the conjunction). Mirrors the dask
    edge in test_dask_explicit_float_dtype_mask_off_reports_false."""
    da = xr.DataArray(
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': 30},
    )
    path = str(tmp_path / "tmp_2092_eager_int_to_float_unmasked.tif")
    to_geotiff(da, path)

    out = open_geotiff(path, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    # The literal 30 is still in the float buffer (cast, not masked).
    assert 30.0 in out.values


# --- Dask path ----------------------------------------------------------


def test_dask_mask_nodata_false_reports_false(tmp_path):
    path = str(tmp_path / "tmp_2092_dask_unmasked.tif")
    _make_float_raster_with_nodata(path)

    out = read_geotiff_dask(path, chunks=2, mask_nodata=False)
    assert out.attrs.get('masked_nodata') is False
    computed = out.values
    assert -9999.0 in computed


def test_dask_mask_nodata_true_reports_true(tmp_path):
    path = str(tmp_path / "tmp_2092_dask_masked.tif")
    _make_float_raster_with_nodata(path)

    out = read_geotiff_dask(path, chunks=2)
    assert out.attrs.get('masked_nodata') is True
    computed = out.values
    assert np.isnan(computed).sum() == 2


def test_dask_explicit_float_dtype_mask_off_reports_false(tmp_path):
    """Edge: caller passes ``dtype=np.float64`` on an int file with
    ``mask_nodata=False``. Old rule would have said
    ``masked_nodata=True`` (final dtype is float) even though the
    buffer holds literal sentinel values. New rule correctly says
    False because the caller opted out of masking."""
    da = xr.DataArray(
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': 30},
    )
    path = str(tmp_path / "tmp_2092_dask_int_to_float_unmasked.tif")
    to_geotiff(da, path)

    out = read_geotiff_dask(
        path, chunks=2, mask_nodata=False, dtype=np.float64,
    )
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    computed = out.values
    # The literal 30 is still in the float buffer (cast, not masked).
    assert 30.0 in computed


# --- VRT path -----------------------------------------------------------


def _write_int_vrt(tmp_path, src_basename, vrt_basename, sentinel=30):
    """Build a single-band VRT pointing at an int16 source TIFF.

    The VRT XML follows the working pattern from
    ``test_vrt_multiband_int_nodata_1611``: ``GeoTransform`` plus
    explicit ``SrcRect`` and ``DstRect`` are required by the
    in-repo VRT reader; without them the reader returns a zero-fill
    buffer instead of decoding the source.
    """
    tifffile = pytest.importorskip("tifffile")
    src = str(tmp_path / src_basename)
    tifffile.imwrite(src, np.array(
        [[10, 20, 30], [40, 50, 60]], dtype=np.int16,
    ), metadata=None)
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


def test_vrt_int_source_mask_nodata_false_reports_false(tmp_path):
    """VRT with int source + mask_nodata=False: buffer stays int with
    literal sentinel; attr must say False. (Float VRT sources get
    inline NaN-masking from the VRT internal reader regardless of
    mask_nodata, so the int case is where the kwarg actually changes
    behaviour. This is the bug-class test for the VRT path.)"""
    vrt = _write_int_vrt(
        tmp_path,
        "tmp_2092_vrt_src.tif",
        "tmp_2092_vrt_unmasked.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False)
    assert out.dtype.kind == 'i', f"expected int dtype, got {out.dtype}"
    assert out.attrs.get('masked_nodata') is False
    # The literal sentinel is still in the buffer.
    assert 30 in out.values


def test_vrt_int_source_mask_nodata_true_reports_true(tmp_path):
    """VRT with int source + mask_nodata=True: helper promotes to
    float64 and replaces sentinel pixels with NaN. Attr says True.
    Baseline direction so the fix doesn't silently flip the canonical
    behaviour."""
    vrt = _write_int_vrt(
        tmp_path,
        "tmp_2092_vrt_src2.tif",
        "tmp_2092_vrt_masked.vrt",
    )
    out = open_geotiff(vrt)  # mask_nodata defaults to True
    assert out.dtype == np.float64, (
        f"expected float64 promotion, got {out.dtype}")
    assert out.attrs.get('masked_nodata') is True
    assert np.isnan(out.values).sum() == 1  # the lone 30 cell


def test_vrt_int_source_mask_off_with_float_cast_reports_false(tmp_path):
    """VRT int source + mask_nodata=False + dtype=float64 cast.

    Initial PR landed a dtype-only VRT rule (``arr.dtype.kind == 'f'``)
    which mis-claimed ``masked_nodata=True`` here -- the integer mask
    helper is skipped under ``mask_nodata=False``, then the explicit
    dtype cast promotes the int buffer to float64 with literal
    sentinels still in it. The fix reads the pre-cast dtype, so
    pre_cast is int and the attr is False. Pins the follow-up fix."""
    vrt = _write_int_vrt(
        tmp_path,
        "tmp_2092_vrt_src_cast.tif",
        "tmp_2092_vrt_unmasked_cast.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    # The literal 30 is still in the float buffer (cast, not masked).
    assert 30.0 in out.values


# --- GPU path -----------------------------------------------------------


@_gpu_only
def test_gpu_mask_nodata_false_reports_false(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2092_gpu_unmasked.tif")
    _make_float_raster_with_nodata(path)

    out = read_geotiff_gpu(path, mask_nodata=False)
    assert out.attrs.get('masked_nodata') is False


@_gpu_only
def test_gpu_mask_nodata_true_reports_true(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2092_gpu_masked.tif")
    _make_float_raster_with_nodata(path)

    out = read_geotiff_gpu(path)
    assert out.attrs.get('masked_nodata') is True
