"""Regression tests for issue #2349.

Epic #2340 PR 5: unsupported GeoTIFF feature combinations fail loudly
with an actionable error message that names the feature, names the
offending input, and points the caller at the release tier map.

Unsupported features covered here:

* Full GDAL VRT parity (warped / pansharpened / processed / derived
  VRT subclasses, derived raster bands, kernel-filtered sources,
  unknown band sub-elements).
* Warped / reprojection VRTs (the GeoTransform check already lands on
  ``RotatedTransformError``; the regression here pins the existing
  behaviour so a future refactor cannot regress it back to a silent
  no-georef path).
* Rotated / sheared GeoTIFF write (the eager writer already rejects
  rotated ``attrs['transform']`` 6-tuples; the regression pins the
  message wording so callers can still match on it).
* Silent mixed-metadata flattening across stacked VRT sources
  (mismatched per-source nodata or AREA_OR_POINT registration).
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (RotatedTransformError, UnsupportedGeoTIFFFeatureError,
                               open_geotiff, to_geotiff)
from xrspatial.geotiff._vrt import parse_vrt, write_vrt


# ---------------------------------------------------------------------------
# VRT parse-time gates: subClass, derived raster bands, unknown band children.
# ---------------------------------------------------------------------------


def test_warped_vrt_subclass_rejected_at_parse():
    """A ``<VRTDataset subClass="VRTWarpedDataset">`` is not a plain mosaic.

    Without an explicit refusal the reader would dispatch on whatever
    simple sources the warped VRT happens to embed and drop the warping
    semantics silently. Pin the typed error and the feature-naming
    substring so the message stays actionable for callers grepping for
    "warped" / "subClass".
    """
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4" '
        'subClass="VRTWarpedDataset"></VRTDataset>'
    )
    with pytest.raises(UnsupportedGeoTIFFFeatureError, match="subClass"):
        parse_vrt(xml, '.')


def test_pansharpened_vrt_subclass_rejected_at_parse():
    """A ``<VRTDataset subClass="VRTPansharpenedDataset">`` is rejected too.

    The subClass check covers every GDAL VRT subclass uniformly, not
    just the warped one. Pin the pansharpened case so a caller who
    points read_vrt at a pansharpened VRT sees the same actionable
    failure rather than silently mis-reading.
    """
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4" '
        'subClass="VRTPansharpenedDataset"></VRTDataset>'
    )
    with pytest.raises(UnsupportedGeoTIFFFeatureError, match="VRTPansharpened"):
        parse_vrt(xml, '.')


def test_derived_rasterband_subclass_rejected_at_parse():
    """A ``<VRTRasterBand subClass="VRTDerivedRasterBand">`` is rejected.

    Derived raster bands declare a pixel-function expression evaluated
    over the sources. read_vrt has no pixel-function evaluator and
    would drop straight to the simple-source path, producing wrong
    output. Pin the typed error and the band number in the message.
    """
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4">'
        '  <VRTRasterBand band="1" dataType="Float32" '
        '   subClass="VRTDerivedRasterBand"></VRTRasterBand>'
        '</VRTDataset>'
    )
    with pytest.raises(UnsupportedGeoTIFFFeatureError,
                       match=r"band=1.*VRTDerivedRasterBand"):
        parse_vrt(xml, '.')


def test_kernel_filtered_source_rejected_at_parse():
    """``<KernelFilteredSource>`` is a known unsupported source type.

    The previous parser silently skipped every non-Simple/Complex tag
    inside ``<VRTRasterBand>``. The new gate enumerates the known
    output-altering children and raises on each. Pin the substring so
    the caller can match on "KernelFilteredSource" specifically.
    """
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4">'
        '  <VRTRasterBand band="1" dataType="Float32">'
        '    <KernelFilteredSource>'
        '      <SourceFilename relativeToVRT="1">src.tif</SourceFilename>'
        '    </KernelFilteredSource>'
        '  </VRTRasterBand>'
        '</VRTDataset>'
    )
    with pytest.raises(UnsupportedGeoTIFFFeatureError,
                       match="KernelFilteredSource"):
        parse_vrt(xml, '.')


def test_pansharpening_options_rejected_at_parse():
    """``<PansharpeningOptions>`` inside a band is rejected.

    PansharpeningOptions sits under VRTRasterBand for the pansharpened
    subClass case. The dataset-level subClass check fires first when
    the subClass attribute is present, but a malformed VRT that omits
    the subClass attribute still has to be rejected. Pin that path
    here.
    """
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4">'
        '  <VRTRasterBand band="1" dataType="Float32">'
        '    <PansharpeningOptions></PansharpeningOptions>'
        '  </VRTRasterBand>'
        '</VRTDataset>'
    )
    with pytest.raises(UnsupportedGeoTIFFFeatureError,
                       match="PansharpeningOptions"):
        parse_vrt(xml, '.')


def test_unknown_band_child_rejected_at_parse():
    """An unknown ``<VRTRasterBand>`` child element is rejected.

    The previous parser silently skipped any unknown tag. A future GDAL
    VRT extension that introduces a new pixel-altering element must
    not slip past this reader as a no-op; the catch-all branch raises
    rather than guess at semantics.
    """
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4">'
        '  <VRTRasterBand band="1" dataType="Float32">'
        '    <FutureVRTPixelMutator></FutureVRTPixelMutator>'
        '  </VRTRasterBand>'
        '</VRTDataset>'
    )
    with pytest.raises(UnsupportedGeoTIFFFeatureError,
                       match="FutureVRTPixelMutator"):
        parse_vrt(xml, '.')


def test_informational_band_children_still_pass(tmp_path):
    """``<Description>`` / ``<UnitType>`` / ``<Offset>`` / ``<Scale>`` skip.

    The gate enumerates known informational children that have no
    effect on the array bytes and must still be ignored silently.
    Pin the allow-list so a future refactor cannot regress it to
    "raise on everything" and break legitimate VRTs.
    """
    src = tmp_path / f'src_2349_info_{uuid.uuid4().hex[:6]}.tif'
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    y = np.arange(4, dtype=np.float64)
    x = np.arange(4, dtype=np.float64)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    to_geotiff(da, str(src), compression='none')
    xml = (
        f'<VRTDataset rasterXSize="4" rasterYSize="4">'
        f'  <SRS>EPSG:4326</SRS>'
        f'  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, 1.0</GeoTransform>'
        f'  <VRTRasterBand band="1" dataType="Float32">'
        f'    <Description>test</Description>'
        f'    <UnitType>m</UnitType>'
        f'    <NoDataValue>-9999</NoDataValue>'
        f'    <SimpleSource>'
        f'      <SourceFilename relativeToVRT="1">{src.name}</SourceFilename>'
        f'      <SourceBand>1</SourceBand>'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>'
        f'      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>'
        f'    </SimpleSource>'
        f'  </VRTRasterBand>'
        f'</VRTDataset>'
    )
    parsed = parse_vrt(xml, str(tmp_path))
    assert len(parsed.bands) == 1
    assert len(parsed.bands[0].sources) == 1


# ---------------------------------------------------------------------------
# VRT writer cross-source mixed-metadata gates.
# ---------------------------------------------------------------------------


def _unique_dir(tmp_path, label: str) -> str:
    d = tmp_path / f"vrt_2349_{label}_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return str(d)


def _write_source(path: str, *, px: float = 1.0, py: float = -1.0,
                  origin_x: float = 0.0, origin_y: float = 100.0,
                  nodata: float | int | None = -9999.0,
                  raster_type: str = 'area',
                  crs: int = 4326,
                  dtype=np.float32, h: int = 4, w: int = 4) -> None:
    arr = np.arange(h * w, dtype=dtype).reshape(h, w)
    y = origin_y + (np.arange(h) + 0.5) * py
    x = origin_x + (np.arange(w) + 0.5) * px
    attrs = {'crs': crs, 'raster_type': raster_type}
    if nodata is not None:
        attrs['nodata'] = nodata
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs=attrs)
    to_geotiff(da, path, compression='none', nodata=nodata)


def test_mixed_per_source_nodata_rejected(tmp_path):
    """Two sources with different nodata sentinels fail the write.

    Legacy ``write_vrt`` picked ``first['nodata']`` for every band and
    silently dropped the second source's sentinel. The fail-closed
    surface refuses; the caller can override by pinning the mosaic
    nodata via ``write_vrt(..., nodata=<value>)``. Pin the typed error
    and a substring naming the kwarg so the message stays actionable.
    """
    d = _unique_dir(tmp_path, "nodata")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_source(a, origin_x=0.0, nodata=-9999.0)
    _write_source(b, origin_x=4.0, nodata=-1.0)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(UnsupportedGeoTIFFFeatureError,
                       match=r"mixed.*nodata|nodata=\-9999"):
        write_vrt(vrt, [a, b])


def test_mixed_nodata_override_via_kwarg_passes(tmp_path):
    """``write_vrt(..., nodata=<value>)`` opts back into flatten-to-kwarg.

    The fail-closed default is the strict path; explicit caller intent
    via the ``nodata`` kwarg overrides it. Pin that the opt-out keeps
    working so the message is actionable rather than a dead-end.
    """
    d = _unique_dir(tmp_path, "nodata_ok")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_source(a, origin_x=0.0, nodata=-9999.0)
    _write_source(b, origin_x=4.0, nodata=-1.0)
    vrt = os.path.join(d, "out.vrt")
    write_vrt(vrt, [a, b], nodata=-9999.0)
    assert os.path.exists(vrt)


def test_mixed_raster_type_rejected(tmp_path):
    """Sources disagreeing on AREA_OR_POINT registration fail the write.

    The mosaic writes a single dataset-level AREA_OR_POINT, so silently
    flattening to the first source's value would shift the disagreeing
    source by half a pixel on read. Pin the typed error and the
    substring naming the mismatch.
    """
    d = _unique_dir(tmp_path, "raster_type")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_source(a, origin_x=0.0, raster_type='area')
    _write_source(b, origin_x=4.0, raster_type='point')
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(UnsupportedGeoTIFFFeatureError,
                       match=r"raster_type|AREA_OR_POINT"):
        write_vrt(vrt, [a, b])


# ---------------------------------------------------------------------------
# Rotated / sheared write gates: regression pins for the existing
# refusal at the eager writer entry point.
# ---------------------------------------------------------------------------


def test_eager_writer_rejects_rotated_6tuple_transform(tmp_path):
    """``attrs['transform']`` 6-tuple with non-zero ``b`` or ``d`` is refused.

    The eager writer emits an axis-aligned GeoTIFF; silently dropping
    the skew terms would place the raster at the wrong location. Pin
    the message wording so the existing match patterns in other
    fail-closed regressions keep matching.
    """
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        dims=['y', 'x'],
        attrs={'transform': (1.0, 0.5, 0.0, 0.0, -1.0, 0.0)},
    )
    path = tmp_path / f"rotated_2349_{uuid.uuid4().hex[:6]}.tif"
    with pytest.raises(ValueError, match=r"rotation/shear"):
        to_geotiff(da, str(path))


def test_eager_writer_rejects_rotated_affine_attr(tmp_path):
    """``attrs['rotated_affine']`` (set by reader on ``allow_rotated``) refused.

    The reader stamps the rotated 6-tuple on this attr when called with
    ``allow_rotated=True``. The writer has no ModelTransformationTag
    emit path, so a read-then-write round-trip would silently lose the
    rotation. Pin the refusal so the regression cannot regress into a
    silent identity-affine output.
    """
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        dims=['y', 'x'],
        attrs={'rotated_affine': (1.0, 0.5, 0.0, 0.0, -1.0, 0.0)},
    )
    path = tmp_path / f"rotated_affine_2349_{uuid.uuid4().hex[:6]}.tif"
    with pytest.raises(ValueError, match=r"rotated_affine"):
        to_geotiff(da, str(path))


# ---------------------------------------------------------------------------
# Warped / reprojection VRT gate: regression pin for the existing
# RotatedTransformError on a VRT with non-zero GeoTransform skew terms.
# ---------------------------------------------------------------------------


def test_vrt_with_skewed_geotransform_rejected(tmp_path):
    """A VRT GeoTransform with non-zero skew is rejected on read.

    The GDAL GeoTransform skew terms (positions 2 and 4 in the
    GDAL ordering) flag a warped / reprojection VRT or a rotated
    source. read_vrt has no resampler for the warped case; pin the
    existing typed error so a future refactor cannot regress to the
    silent no-georef fallback.
    """
    src = tmp_path / f'flat_2349_{uuid.uuid4().hex[:6]}.tif'
    arr = np.zeros((4, 4), dtype=np.float32)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': np.arange(4, dtype=np.float64),
                              'x': np.arange(4, dtype=np.float64)},
                      attrs={'crs': 4326})
    to_geotiff(da, str(src), compression='none')

    vrt = tmp_path / f'rotated_2349_{uuid.uuid4().hex[:6]}.vrt'
    vrt.write_text(
        f'<VRTDataset rasterXSize="4" rasterYSize="4">'
        f'  <SRS>EPSG:4326</SRS>'
        f'  <GeoTransform>0.0, 1.0, 0.5, 0.0, 0.0, -1.0</GeoTransform>'
        f'  <VRTRasterBand band="1" dataType="Float32">'
        f'    <SimpleSource>'
        f'      <SourceFilename relativeToVRT="1">{src.name}</SourceFilename>'
        f'      <SourceBand>1</SourceBand>'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>'
        f'      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>'
        f'    </SimpleSource>'
        f'  </VRTRasterBand>'
        f'</VRTDataset>'
    )
    with pytest.raises(RotatedTransformError, match=r"rotated affine"):
        open_geotiff(str(vrt))
