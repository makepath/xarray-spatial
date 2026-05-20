"""``masked_nodata`` on VRT paths must honour ``mask_nodata=`` (#2159).

Both VRT call sites (eager at ``vrt.py:462`` and chunked at
``vrt.py:883``) used to compute ``attrs['masked_nodata']`` from the
buffer dtype alone:

    masked=(pre_cast_dtype.kind == 'f')          # eager
    masked=(declared_dtype.kind == 'f')          # chunked

That ignored the caller's ``mask_nodata`` opt-out. The dask, GPU, and
dask+GPU backends already gate on both signals
(``mask_nodata and dtype.kind == 'f'``); the contract at
``_attrs._set_nodata_attrs`` says the same. The VRT paths now match.

These tests pin the attr-side fix only. Sibling issue #2158 covers the
behavioural mismatch where the VRT internal reader still NaN-masks
float sources under ``mask_nodata=False``; this test file does not
assert on whether the buffer contains literal sentinels or NaN, only
on what the attr reports.

Coverage:
* Eager VRT: float source + ``mask_nodata=False`` reports ``False``.
* Chunked VRT: float source + ``mask_nodata=False`` reports ``False``.
* Eager VRT: float source + ``mask_nodata=True`` still reports ``True``
  (canonical direction, regression guard against an over-eager fix).
* Chunked VRT: float source + ``mask_nodata=True`` still reports
  ``True``.
* Both VRT paths: int source + ``mask_nodata=False`` reports ``False``
  (the pre-fix rule already got this right; keep it green).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_geotiff_dask


def _write_float_vrt(tmp_path, src_basename, vrt_basename, sentinel=-9999.0):
    """Build a single-band float32 VRT with a declared sentinel.

    Layout mirrors the working pattern from
    ``test_masked_nodata_attr_2092.py``: ``GeoTransform`` plus explicit
    ``SrcRect`` / ``DstRect`` are required by the in-repo VRT reader.
    """
    tifffile = pytest.importorskip("tifffile")
    src = str(tmp_path / src_basename)
    tifffile.imwrite(src, np.array(
        [[1.0, 2.0, sentinel],
         [4.0, sentinel, 6.0]],
        dtype=np.float32,
    ), metadata=None)
    vrt = str(tmp_path / vrt_basename)
    vrt_xml = f"""<VRTDataset rasterXSize="3" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
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


def _write_int_vrt(tmp_path, src_basename, vrt_basename, sentinel=30):
    """Single-band int16 VRT with a declared sentinel."""
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


# --- Eager VRT ----------------------------------------------------------


def test_vrt_eager_float_source_mask_off_reports_false(tmp_path):
    """Eager VRT + float source + ``mask_nodata=False`` must report
    ``masked_nodata=False``. Pre-fix rule (dtype alone) said ``True``."""
    vrt = _write_float_vrt(
        tmp_path,
        "tmp_2159_eager_float_src.tif",
        "tmp_2159_eager_unmasked.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is False, (
        f"caller opted out of masking but attrs say "
        f"masked_nodata={out.attrs.get('masked_nodata')!r}")


def test_vrt_eager_float_source_mask_on_reports_true(tmp_path):
    """Canonical direction: float source + masking on. The masking
    step runs, attr says True. Regression guard."""
    vrt = _write_float_vrt(
        tmp_path,
        "tmp_2159_eager_float_src_masked.tif",
        "tmp_2159_eager_masked.vrt",
    )
    out = open_geotiff(vrt)  # mask_nodata defaults to True
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is True


def test_vrt_eager_int_source_mask_off_reports_false(tmp_path):
    """Eager VRT + int source + ``mask_nodata=False``: integer helper
    skipped, dtype stays int, attr says False. Pre-fix rule already
    got this right (int dtype -> False); keep it green under the
    new ``mask_nodata and dtype.kind == 'f'`` rule."""
    vrt = _write_int_vrt(
        tmp_path,
        "tmp_2159_eager_int_src.tif",
        "tmp_2159_eager_int_unmasked.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False)
    assert out.dtype.kind == 'i'
    assert out.attrs.get('masked_nodata') is False


def test_vrt_eager_float_source_mask_off_with_cast_reports_false(tmp_path):
    """Eager VRT + float source + ``mask_nodata=False`` + ``dtype=float64``
    cast. Pre-fix used ``pre_cast_dtype.kind == 'f'`` so pre-cast is
    float anyway and the rule said True. New rule short-circuits on
    ``mask_nodata=False`` and says False. The caller-supplied cast is
    still recorded via ``nodata_dtype_cast``."""
    vrt = _write_float_vrt(
        tmp_path,
        "tmp_2159_eager_float_src_cast.tif",
        "tmp_2159_eager_unmasked_cast.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'


# --- Chunked VRT --------------------------------------------------------


def test_vrt_chunked_float_source_mask_off_reports_false(tmp_path):
    """Chunked VRT path (``chunks=`` triggers ``_read_vrt_chunked``)
    + float source + ``mask_nodata=False`` must report False."""
    vrt = _write_float_vrt(
        tmp_path,
        "tmp_2159_chunked_float_src.tif",
        "tmp_2159_chunked_unmasked.vrt",
    )
    out = read_geotiff_dask(vrt, chunks=2, mask_nodata=False)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is False, (
        f"chunked VRT path: caller opted out of masking but attrs say "
        f"masked_nodata={out.attrs.get('masked_nodata')!r}")


def test_vrt_chunked_float_source_mask_on_reports_true(tmp_path):
    """Canonical direction on the chunked path: masking on, attr True."""
    vrt = _write_float_vrt(
        tmp_path,
        "tmp_2159_chunked_float_src_masked.tif",
        "tmp_2159_chunked_masked.vrt",
    )
    out = read_geotiff_dask(vrt, chunks=2)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is True


def test_vrt_chunked_int_source_mask_off_reports_false(tmp_path):
    """Chunked VRT + int source + ``mask_nodata=False``. ``declared_dtype``
    stays integer because the masking-driven float-promotion gate
    earlier in the function is itself gated on ``mask_nodata``.
    The attr says False under both the old and the new rule."""
    vrt = _write_int_vrt(
        tmp_path,
        "tmp_2159_chunked_int_src.tif",
        "tmp_2159_chunked_int_unmasked.vrt",
    )
    out = read_geotiff_dask(vrt, chunks=2, mask_nodata=False)
    assert out.dtype.kind == 'i'
    assert out.attrs.get('masked_nodata') is False


def test_vrt_chunked_float_source_mask_off_with_cast_reports_false(tmp_path):
    """Chunked VRT + float source + ``mask_nodata=False`` + ``dtype=float64``
    cast. Same logic as the eager equivalent: caller opted out of
    masking, attr is False even though the lazy graph dtype is float."""
    vrt = _write_float_vrt(
        tmp_path,
        "tmp_2159_chunked_float_src_cast.tif",
        "tmp_2159_chunked_unmasked_cast.vrt",
    )
    out = read_geotiff_dask(
        vrt, chunks=2, mask_nodata=False, dtype=np.float64,
    )
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'


# --- Cross-backend invariant -------------------------------------------


def test_vrt_attr_matches_dask_backend_under_mask_off(tmp_path):
    """Both VRT backends should report the same ``masked_nodata`` as
    the regular dask backend does for an equivalent input. Pins the
    cross-backend invariant the contract at
    ``_attrs._set_nodata_attrs`` calls out."""
    # Float source, mask off, with an explicit float64 cast so the
    # buffer dtype is float on every backend.
    vrt = _write_float_vrt(
        tmp_path,
        "tmp_2159_xbackend_src.tif",
        "tmp_2159_xbackend.vrt",
    )

    eager = open_geotiff(vrt, mask_nodata=False, dtype=np.float64)
    chunked = read_geotiff_dask(
        vrt, chunks=2, mask_nodata=False, dtype=np.float64,
    )

    assert eager.attrs.get('masked_nodata') is False
    assert chunked.attrs.get('masked_nodata') is False
    assert (eager.attrs.get('masked_nodata')
            == chunked.attrs.get('masked_nodata'))
