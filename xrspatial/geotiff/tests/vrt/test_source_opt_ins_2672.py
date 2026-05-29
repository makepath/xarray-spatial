"""VRT source-read opt-in forwarding (issue #2672).

``read_vrt`` accepts ``allow_rotated`` and ``allow_invalid_nodata`` and
documents them as opt-ins, but until issue #2672 the eager and chunked
paths only forwarded the codec flags to the per-source GeoTIFF read. A
caller who passed ``allow_rotated=True`` or ``allow_invalid_nodata=True``
still hit a typed rejection when a source TIFF was rotated or carried a
non-representable integer ``GDAL_NODATA`` value.

These tests build a single-source VRT over a rotated source TIFF and over
a uint16 source TIFF with a NaN ``GDAL_NODATA``. Each case fails without
the opt-in and succeeds with it, on both the eager and chunked paths. A
final case confirms the opt-in keeps a source out of the
``missing_sources='warn'`` hole bucket.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from xrspatial.geotiff import GeoTIFFFallbackWarning, open_geotiff, read_vrt
from xrspatial.geotiff._errors import InvalidIntegerNodataError, RotatedTransformError

# Reuse the existing hand-rolled TIFF builders so this suite shares one
# byte layout with the non-VRT reader tests.
from ..read.test_crs import _write_rotated_tiff
from ..read.test_nodata import _build_uint16_tiff_1774


def _write_single_source_vrt(tmp_path, src_path, *, data_type, name) -> str:
    """Write a 2x2 single-source VRT over ``src_path``.

    The VRT GeoTransform is axis-aligned; only the *source* TIFF carries
    the property under test (rotation or invalid nodata).
    """
    vrt_path = tmp_path / f"{name}.vrt"
    vrt_path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
        f'  <VRTRasterBand dataType="{data_type}" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    return str(vrt_path)


def _rotated_source_vrt(tmp_path) -> str:
    src = tmp_path / "rotated_src_2672.tif"
    _write_rotated_tiff(str(src), np.array([[10, 20], [30, 40]], dtype='<u2'))
    return _write_single_source_vrt(
        tmp_path, str(src), data_type="UInt16", name="rotated_2672")


def _invalid_nodata_source_vrt(tmp_path) -> str:
    # ``_build_uint16_tiff_1774`` writes a 2x2 uint16 TIFF whose
    # GDAL_NODATA tag is the literal string "nan" -- non-representable
    # on an integer dtype, so the per-source read rejects it unless
    # ``allow_invalid_nodata=True`` is forwarded.
    src = _build_uint16_tiff_1774('nan', tmp_path)
    return _write_single_source_vrt(
        tmp_path, src, data_type="UInt16", name="invalid_nodata_2672")


# ---------------------------------------------------------------------------
# allow_invalid_nodata
# ---------------------------------------------------------------------------

def test_eager_invalid_nodata_rejected_without_opt_in(tmp_path):
    vrt = _invalid_nodata_source_vrt(tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_vrt(vrt)


def test_eager_invalid_nodata_accepted_with_opt_in(tmp_path):
    vrt = _invalid_nodata_source_vrt(tmp_path)
    da = read_vrt(vrt, allow_invalid_nodata=True)
    # NaN sentinel can't match any uint16 pixel, so the dtype survives
    # and the literal pixels come through unmasked.
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])


def test_chunked_invalid_nodata_rejected_without_opt_in(tmp_path):
    vrt = _invalid_nodata_source_vrt(tmp_path)
    da = open_geotiff(vrt, chunks=1)
    with pytest.raises(InvalidIntegerNodataError):
        da.compute()


def test_chunked_invalid_nodata_accepted_with_opt_in(tmp_path):
    vrt = _invalid_nodata_source_vrt(tmp_path)
    da = open_geotiff(vrt, chunks=1, allow_invalid_nodata=True)
    result = da.compute()
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result.values, [[10, 20], [30, 40]])


# ---------------------------------------------------------------------------
# allow_rotated
# ---------------------------------------------------------------------------

def test_eager_rotated_source_rejected_without_opt_in(tmp_path):
    vrt = _rotated_source_vrt(tmp_path)
    with pytest.raises(RotatedTransformError):
        read_vrt(vrt)


def test_eager_rotated_source_accepted_with_opt_in(tmp_path):
    vrt = _rotated_source_vrt(tmp_path)
    da = read_vrt(vrt, allow_rotated=True)
    # The source pixel grid is read without the georef assumption.
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])


def test_chunked_rotated_source_rejected_without_opt_in(tmp_path):
    vrt = _rotated_source_vrt(tmp_path)
    da = open_geotiff(vrt, chunks=1)
    with pytest.raises(RotatedTransformError):
        da.compute()


def test_chunked_rotated_source_accepted_with_opt_in(tmp_path):
    vrt = _rotated_source_vrt(tmp_path)
    da = open_geotiff(vrt, chunks=1, allow_rotated=True)
    result = da.compute()
    np.testing.assert_array_equal(result.values, [[10, 20], [30, 40]])


# ---------------------------------------------------------------------------
# missing_sources='warn' interaction
# ---------------------------------------------------------------------------

def test_missing_sources_warn_opt_in_avoids_false_hole(tmp_path):
    """The opt-in keeps a readable source out of the hole bucket.

    Without ``allow_invalid_nodata=True`` the source-metadata rejection
    is caught by the per-source ``except`` and reclassified as a hole
    under ``missing_sources='warn'``, silently dropping the tile. With
    the opt-in forwarded, the source reads cleanly and no hole is
    recorded.
    """
    vrt = _invalid_nodata_source_vrt(tmp_path)

    # Baseline: without the opt-in, warn mode turns the rejection into a
    # hole and emits the fallback warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        da_hole = read_vrt(vrt, missing_sources='warn')
    assert any(issubclass(w.category, GeoTIFFFallbackWarning) for w in caught)
    assert da_hole.attrs.get('vrt_holes')

    # With the opt-in forwarded, the source reads and no hole is recorded.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        da_ok = read_vrt(
            vrt, missing_sources='warn', allow_invalid_nodata=True)
    assert not any(
        issubclass(w.category, GeoTIFFFallbackWarning) for w in caught)
    assert not da_ok.attrs.get('vrt_holes')
    np.testing.assert_array_equal(da_ok.values, [[10, 20], [30, 40]])
