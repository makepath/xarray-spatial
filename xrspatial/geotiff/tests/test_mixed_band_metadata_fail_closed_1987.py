"""Fail-closed coverage for ``MixedBandMetadataError`` (issue #1987 PR 5).

A VRT can mosaic source bands that declare disagreeing per-band
``<NoDataValue>`` sentinels. The legacy reader picked band 0's sentinel
for the whole mosaic, which let band N's valid pixels collide with
band 0's sentinel after the flatten-to-scalar step. The fail-closed
default refuses the ambiguity; ``band_nodata='first'`` keeps the legacy
behaviour explicitly.

The validator check itself, the ``band_nodata=`` kwarg on
``read_vrt``, and the per-VRT-read context plumbing landed in
the #2031 bundle. This PR registers the check and forwards
``band_nodata`` through ``open_geotiff`` and ``read_geotiff_dask``.
The tests below pin the four entry-point contracts:

* default ``read_vrt`` raises on disagreeing per-band sentinels
* default ``read_vrt(chunks=...)`` raises before any task fires
* ``band_nodata='first'`` opts back into the legacy reader
* ``band_nodata`` raises on non-VRT sources for ``open_geotiff`` and
  ``read_geotiff_dask``
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import (GeoTIFFAmbiguousMetadataError, MixedBandMetadataError, open_geotiff,
                               read_geotiff_dask, read_vrt, to_geotiff)
from xrspatial.geotiff._writer import write


def _write_mixed_band_vrt(tmp_path, *, sentinel_a=65535, sentinel_b=65000):
    """Two uint16 sources with distinct sentinels, mosaiced into one VRT.

    ``[1, 1]`` is the sentinel pixel in each source so the legacy
    reader's flatten-to-first-band step would mis-mask band 1.
    """
    band0 = np.array([[1, 2], [3, sentinel_a]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, sentinel_b]], dtype=np.uint16)
    p0 = str(tmp_path / 'mixed_band_1987_a.tif')
    p1 = str(tmp_path / 'mixed_band_1987_b.tif')
    write(band0, p0, nodata=sentinel_a, compression='none', tiled=False)
    write(band1, p1, nodata=sentinel_b, compression='none', tiled=False)
    vrt_path = str(tmp_path / 'mixed_band_1987.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>{sentinel_a}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>{sentinel_b}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def _write_shared_sentinel_vrt(tmp_path, *, sentinel=65535):
    """Two uint16 sources mosaiced with the *same* per-band sentinel.

    The fail-closed check must not trigger on a VRT whose per-band
    nodata agrees; the check is about ambiguity, not about per-band
    declarations.
    """
    band0 = np.array([[1, 2], [3, sentinel]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, sentinel]], dtype=np.uint16)
    p0 = str(tmp_path / 'shared_band_1987_a.tif')
    p1 = str(tmp_path / 'shared_band_1987_b.tif')
    write(band0, p0, nodata=sentinel, compression='none', tiled=False)
    write(band1, p1, nodata=sentinel, compression='none', tiled=False)
    vrt_path = str(tmp_path / 'shared_band_1987.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>{sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>{sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def _write_one_band_no_sentinel_vrt(tmp_path, *, sentinel=65535):
    """One band declares a sentinel, the other does not.

    ``[b.nodata for b in vrt.bands]`` is ``[sentinel, None]``. The
    check considers ``None`` entries "no declaration" rather than a
    distinct sentinel, so only one concrete value is present and the
    check must accept.
    """
    band0 = np.array([[1, 2], [3, sentinel]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, 99]], dtype=np.uint16)
    p0 = str(tmp_path / 'one_band_sentinel_a.tif')
    p1 = str(tmp_path / 'one_band_sentinel_b.tif')
    write(band0, p0, nodata=sentinel, compression='none', tiled=False)
    write(band1, p1, nodata=None, compression='none', tiled=False)
    vrt_path = str(tmp_path / 'one_band_sentinel.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>{sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_read_vrt_rejects_mixed_per_band_nodata(tmp_path):
    """Default ``read_vrt`` raises on disagreeing per-band sentinels."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    with pytest.raises(MixedBandMetadataError) as exc_info:
        read_vrt(vrt_path)
    msg = str(exc_info.value)
    assert "65535" in msg and "65000" in msg
    assert "band_nodata='first'" in msg
    assert "#1987" in msg


def test_read_vrt_chunked_rejects_mixed_per_band_nodata(tmp_path):
    """The chunked dispatch validates before scheduling any task."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    with pytest.raises(MixedBandMetadataError):
        read_vrt(vrt_path, chunks=1)


def test_read_vrt_band_nodata_first_opts_back_in(tmp_path):
    """``band_nodata='first'`` keeps the legacy flatten-to-first path."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    r = read_vrt(vrt_path, band_nodata='first')
    # Legacy semantics: band 0's sentinel becomes ``attrs['nodata']``.
    assert r.attrs.get('nodata') == 65535.0


def test_read_vrt_shared_sentinel_accepts(tmp_path):
    """A VRT whose bands agree on a single sentinel is not ambiguous."""
    vrt_path = _write_shared_sentinel_vrt(tmp_path)
    r = read_vrt(vrt_path)
    assert r.attrs.get('nodata') == 65535.0


def test_read_vrt_only_one_band_declares_sentinel_accepts(tmp_path):
    """``None`` per-band entries count as "no declaration", not a value.

    A VRT with one declared sentinel and one undeclared band has only
    one concrete sentinel so the fail-closed check must accept.
    """
    vrt_path = _write_one_band_no_sentinel_vrt(tmp_path)
    # Should not raise.
    read_vrt(vrt_path)


def test_open_geotiff_propagates_mixed_band_rejection(tmp_path):
    """``open_geotiff`` on a ``.vrt`` source routes through ``read_vrt``."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    with pytest.raises(MixedBandMetadataError):
        open_geotiff(vrt_path)


def test_open_geotiff_band_nodata_first_passes_through(tmp_path):
    """``open_geotiff(band_nodata='first')`` forwards to ``read_vrt``."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    r = open_geotiff(vrt_path, band_nodata='first')
    assert r.attrs.get('nodata') == 65535.0


def test_read_geotiff_dask_band_nodata_first_passes_through(tmp_path):
    """``read_geotiff_dask`` on a ``.vrt`` source forwards the kwarg."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    r = read_geotiff_dask(vrt_path, chunks=1, band_nodata='first')
    assert r.attrs.get('nodata') == 65535.0


def test_open_geotiff_band_nodata_rejected_on_non_vrt_source(tmp_path):
    """``band_nodata`` is VRT-only; reject up front on plain GeoTIFFs.

    Same dispatch-kwarg pattern as ``missing_sources``, ``on_gpu_failure``,
    and ``max_cloud_bytes``: silently dropping a backend-specific kwarg
    is the bug class #1561 / #1605 / #1685 / #1795 / #1810 / #1974 closed
    on the rest of the surface.
    """
    arr = np.zeros((2, 2), dtype=np.uint16)
    arr_da = _wrap_2d(arr)
    p = tmp_path / 'plain.tif'
    to_geotiff(arr_da, str(p), compression='none', tiled=False)
    with pytest.raises(ValueError, match="band_nodata only applies to VRT"):
        open_geotiff(str(p), band_nodata='first')


def test_read_geotiff_dask_band_nodata_rejected_on_non_vrt_source(tmp_path):
    """``read_geotiff_dask`` matches ``open_geotiff``'s VRT-only guard."""
    arr = np.zeros((2, 2), dtype=np.uint16)
    arr_da = _wrap_2d(arr)
    p = tmp_path / 'plain.tif'
    to_geotiff(arr_da, str(p), compression='none', tiled=False)
    with pytest.raises(ValueError, match="band_nodata only applies to VRT"):
        read_geotiff_dask(str(p), chunks=1, band_nodata='first')


def test_mixed_band_metadata_error_subclasses_base(tmp_path):
    """``MixedBandMetadataError`` is catchable via the base class."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    with pytest.raises(GeoTIFFAmbiguousMetadataError):
        read_vrt(vrt_path)


def test_read_vrt_band_nodata_rejects_unknown_value(tmp_path):
    """Typos like ``band_nodata='firs'`` raise at the boundary.

    Without the value check, an unknown string degrades to strict mode
    (the registered check treats anything other than ``'first'`` as "no
    opt-out") and the caller sees ``MixedBandMetadataError`` from a call
    site that looked like an explicit opt-out. Mirrors the
    ``missing_sources`` value-validation pattern in ``read_vrt``.
    """
    vrt_path = _write_mixed_band_vrt(tmp_path)
    with pytest.raises(ValueError, match="band_nodata must be None or 'first'"):
        read_vrt(vrt_path, band_nodata='firs')


def test_open_geotiff_band_nodata_rejects_unknown_value(tmp_path):
    """``open_geotiff`` surfaces the same value-validation on VRT sources."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    with pytest.raises(ValueError, match="band_nodata must be None or 'first'"):
        open_geotiff(vrt_path, band_nodata='legacy')


def test_read_geotiff_dask_band_nodata_rejects_unknown_value(tmp_path):
    """``read_geotiff_dask`` forwards the value-validation on VRT sources."""
    vrt_path = _write_mixed_band_vrt(tmp_path)
    with pytest.raises(ValueError, match="band_nodata must be None or 'first'"):
        read_geotiff_dask(vrt_path, chunks=1, band_nodata='banana')


def _wrap_2d(arr):
    """Wrap a 2D numpy array as a minimal DataArray for ``to_geotiff``."""
    import xarray as xr
    h, w = arr.shape
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={
            'y': np.arange(h, dtype=np.float64),
            'x': np.arange(w, dtype=np.float64),
        },
        attrs={'crs': 4326,
               'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
