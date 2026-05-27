"""VRT metadata test suite.

Organised by sub-concern; helpers are prefixed (e.g. ``_holes_attr_*``)
so they do not collide across sections.

Sections:
* ``vrt_holes`` attr on missing-source reads
* ``masked_nodata`` attr honours ``mask_nodata`` kwarg
* Per-band ``<NoDataValue>`` selection
* SimpleSource ``<NODATA>0</NODATA>`` survives the falsy-zero bug
* Integer-with-nodata promotion through ``read_vrt``
* ``mask_nodata=False`` preserves float sentinels
* Tile-level metadata parity for VRT tiled writes
* VRT XML parsed once on the chunked path
* ``write_vrt`` escapes XML special characters
* XML size cap on eager ``read_vrt``
* XML size cap on chunked ``read_vrt``
* VRT metadata parity across backends
"""
from __future__ import annotations

import glob
import os
import pathlib
import pickle
import tempfile
import warnings

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (GeoTIFFFallbackWarning, MixedBandMetadataError, open_geotiff,
                               read_geotiff_dask, read_vrt, to_geotiff, write_vrt)
from xrspatial.geotiff._attrs import GEOREF_STATUS_FULL, GEOREF_STATUS_TRANSFORM_ONLY
from xrspatial.geotiff._errors import VRTUnsupportedError
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._vrt import parse_vrt
from xrspatial.geotiff._vrt import read_vrt as _source_nodata_zero_read_vrt_internal
from xrspatial.geotiff._vrt import read_vrt as _xml_size_cap_read_vrt_internal
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal
from xrspatial.geotiff._writer import write
from xrspatial.geotiff.tests.conftest import requires_gpu

# ---------------------------------------------------------------------------
# vrt_holes attr on missing-source reads
# ---------------------------------------------------------------------------


@pytest.fixture
def holes_attr_clear_strict_env(monkeypatch):
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)


@pytest.fixture
def holes_attr_set_strict_env(monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')


def _holes_attr_write_vrt_with_missing_source(vrt_path, missing_src) -> None:
    """Write a VRT with an Int32 band whose only source is missing.

    Integer ``dataType`` is the failure mode of interest here: the
    pre-fix lenient path zero-fills the output buffer (``fill = 0`` for
    integer dtypes) and the user cannot distinguish that hole from real
    zero-valued data. ``NoDataValue`` is omitted on purpose -- having
    one would let downstream code mask the hole and side-step the
    regression. See the module docstring.
    """
    vrt_path.write_text(
        f'<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        f'  <SRS></SRS>\n'
        f'  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        f'  <VRTRasterBand dataType="Int32" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{missing_src}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )


def test_skipped_source_records_vrt_holes_attr(holes_attr_clear_strict_env, tmp_path):
    """A VRT with a missing source returns a DataArray whose attrs
    carry a ``vrt_holes`` entry naming the source, band, dst_rect,
    and underlying error.

    Uses an Int32 VRT so the hole is zero-filled (the exact failure
    mode of interest): without the attr there is no way to tell
    the all-zeros tile from real data.
    """
    import numpy as np
    vrt_path = tmp_path / 'mosaic_1734_missing.vrt'
    missing_src = f'{tmp_path}/does_not_exist_1734.tif'
    _holes_attr_write_vrt_with_missing_source(vrt_path, missing_src)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', GeoTIFFFallbackWarning)
        da = read_vrt(str(vrt_path), missing_sources='warn')
    assert np.issubdtype(da.dtype, np.integer)
    assert (da.values == 0).all()
    assert 'vrt_holes' in da.attrs
    holes = da.attrs['vrt_holes']
    assert isinstance(holes, list)
    assert len(holes) == 1
    h = holes[0]
    assert h['source'].endswith('does_not_exist_1734.tif')
    assert h['band'] == 1
    assert h['dst_rect'] == (0, 0, 4, 4)
    assert 'error' in h
    assert h['error']


def test_no_holes_attr_when_all_sources_read(holes_attr_clear_strict_env, tmp_path):
    """A successful VRT read does not advertise an empty ``vrt_holes``
    attr; the key is omitted entirely so ``"vrt_holes" in attrs`` is a
    cheap completeness check."""
    import numpy as np
    import xarray as xr

    from xrspatial.geotiff import to_geotiff
    src_path = tmp_path / 'src_1734.tif'
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    da_src = xr.DataArray(arr, dims=['y', 'x'], coords={'y': np.linspace(3.5, 0.5, 4), 'x': np.linspace(0.5, 3.5, 4)}, attrs={'crs': 4326})  # noqa: E501
    to_geotiff(da_src, str(src_path), compression='none')
    vrt_path = tmp_path / 'mosaic_1734_ok.vrt'
    vrt_path.write_text(
        f'<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        f'  <SRS></SRS>\n'
        f'  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        f'  <VRTRasterBand dataType="Float32" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', GeoTIFFFallbackWarning)
        da = read_vrt(str(vrt_path))
    assert 'vrt_holes' not in da.attrs


def test_strict_mode_still_raises(holes_attr_set_strict_env, tmp_path):
    """Strict mode is unchanged: the missing source surfaces the
    underlying ``FileNotFoundError`` (an ``OSError`` subclass) from
    ``read_to_array`` instead of warning-and-skipping.

    Asserting the concrete exception class -- not a bare ``Exception``
    -- keeps the regression test honest: an unrelated bug somewhere in
    the read path that happens to raise a different exception will
    fail this test instead of silently satisfying it.
    """
    vrt_path = tmp_path / 'mosaic_1734_strict.vrt'
    missing_src = f'{tmp_path}/does_not_exist_1734_strict.tif'
    _holes_attr_write_vrt_with_missing_source(vrt_path, missing_src)
    with pytest.raises(FileNotFoundError, match='does_not_exist_1734_strict.tif'):
        read_vrt(str(vrt_path))


def test_warning_mentions_how_to_detect_holes(holes_attr_clear_strict_env, tmp_path):
    """The fallback warning now points callers at the attr or the
    strict env var so the recovery path is discoverable from a single
    captured warning."""
    vrt_path = tmp_path / 'mosaic_1734_msg.vrt'
    missing_src = f'{tmp_path}/does_not_exist_1734_msg.tif'
    _holes_attr_write_vrt_with_missing_source(vrt_path, missing_src)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        read_vrt(str(vrt_path), missing_sources='warn')
    fallback = [x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)]
    assert fallback, 'expected at least one GeoTIFFFallbackWarning'
    msg = ' '.join((str(x.message) for x in fallback))
    assert 'vrt_holes' in msg or 'XRSPATIAL_GEOTIFF_STRICT' in msg


# ---------------------------------------------------------------------------
# masked_nodata attr honours mask_nodata kwarg
# ---------------------------------------------------------------------------


def _masked_nodata_attr_write_float_vrt(tmp_path, src_basename, vrt_basename, sentinel=-9999.0):
    """Build a single-band float32 VRT with a declared sentinel.

    Layout mirrors the working pattern from
    ``test_masked_nodata_attr_2092.py``: ``GeoTransform`` plus explicit
    ``SrcRect`` / ``DstRect`` are required by the in-repo VRT reader.
    """
    tifffile = pytest.importorskip('tifffile')
    src = str(tmp_path / src_basename)
    tifffile.imwrite(src, np.array([[1.0, 2.0, sentinel], [4.0, sentinel, 6.0]], dtype=np.float32), metadata=None)  # noqa: E501
    vrt = str(tmp_path / vrt_basename)
    vrt_xml = f'<VRTDataset rasterXSize="3" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="Float32" band="1">\n    <NoDataValue>{sentinel}</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="3" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="3" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    with open(vrt, 'w') as fh:
        fh.write(vrt_xml)
    return vrt


def _masked_nodata_attr_write_int_vrt(tmp_path, src_basename, vrt_basename, sentinel=30):
    """Single-band int16 VRT with a declared sentinel."""
    tifffile = pytest.importorskip('tifffile')
    src = str(tmp_path / src_basename)
    tifffile.imwrite(src, np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16), metadata=None)
    vrt = str(tmp_path / vrt_basename)
    vrt_xml = f'<VRTDataset rasterXSize="3" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="Int16" band="1">\n    <NoDataValue>{sentinel}</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="3" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="3" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    with open(vrt, 'w') as fh:
        fh.write(vrt_xml)
    return vrt


def test_vrt_eager_float_source_mask_off_reports_false(tmp_path):
    """Eager VRT + float source + ``mask_nodata=False`` must report
    ``masked_nodata=False``. Pre-fix rule (dtype alone) said ``True``."""
    vrt = _masked_nodata_attr_write_float_vrt(tmp_path, 'tmp_2159_eager_float_src.tif', 'tmp_2159_eager_unmasked.vrt')  # noqa: E501
    out = open_geotiff(vrt, mask_nodata=False)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is False, f"caller opted out of masking but attrs say masked_nodata={out.attrs.get('masked_nodata')!r}"  # noqa: E501


def test_vrt_eager_float_source_mask_on_reports_true(tmp_path):
    """Canonical direction: float source + masking on. The masking
    step runs, attr says True. Regression guard."""
    vrt = _masked_nodata_attr_write_float_vrt(tmp_path, 'tmp_2159_eager_float_src_masked.tif', 'tmp_2159_eager_masked.vrt')  # noqa: E501
    out = open_geotiff(vrt)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is True


def test_vrt_eager_int_source_mask_off_reports_false(tmp_path):
    """Eager VRT + int source + ``mask_nodata=False``: integer helper
    skipped, dtype stays int, attr says False. Pre-fix rule already
    got this right (int dtype -> False); keep it green under the
    new ``mask_nodata and dtype.kind == 'f'`` rule."""
    vrt = _masked_nodata_attr_write_int_vrt(tmp_path, 'tmp_2159_eager_int_src.tif', 'tmp_2159_eager_int_unmasked.vrt')  # noqa: E501
    out = open_geotiff(vrt, mask_nodata=False)
    assert out.dtype.kind == 'i'
    assert out.attrs.get('masked_nodata') is False


def test_vrt_eager_float_source_mask_off_with_cast_reports_false(tmp_path):
    """Eager VRT + float source + ``mask_nodata=False`` + ``dtype=float64``
    cast. Pre-fix used ``pre_cast_dtype.kind == 'f'`` so pre-cast is
    float anyway and the rule said True. New rule short-circuits on
    ``mask_nodata=False`` and says False. The caller-supplied cast is
    still recorded via ``nodata_dtype_cast``."""
    vrt = _masked_nodata_attr_write_float_vrt(tmp_path, 'tmp_2159_eager_float_src_cast.tif', 'tmp_2159_eager_unmasked_cast.vrt')  # noqa: E501
    out = open_geotiff(vrt, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'


def test_vrt_chunked_float_source_mask_off_reports_false(tmp_path):
    """Chunked VRT path (``chunks=`` triggers ``_read_vrt_chunked``)
    + float source + ``mask_nodata=False`` must report False."""
    vrt = _masked_nodata_attr_write_float_vrt(tmp_path, 'tmp_2159_chunked_float_src.tif', 'tmp_2159_chunked_unmasked.vrt')  # noqa: E501
    out = read_geotiff_dask(vrt, chunks=2, mask_nodata=False)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is False, f"chunked VRT path: caller opted out of masking but attrs say masked_nodata = {out.attrs.get('masked_nodata')!r}"  # noqa: E501


def test_vrt_chunked_float_source_mask_on_reports_true(tmp_path):
    """Canonical direction on the chunked path: masking on, attr True."""
    vrt = _masked_nodata_attr_write_float_vrt(tmp_path, 'tmp_2159_chunked_float_src_masked.tif', 'tmp_2159_chunked_masked.vrt')  # noqa: E501
    out = read_geotiff_dask(vrt, chunks=2)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is True


def test_vrt_chunked_int_source_mask_off_reports_false(tmp_path):
    """Chunked VRT + int source + ``mask_nodata=False``. ``declared_dtype``
    stays integer because the masking-driven float-promotion gate
    earlier in the function is itself gated on ``mask_nodata``.
    The attr says False under both the old and the new rule."""
    vrt = _masked_nodata_attr_write_int_vrt(tmp_path, 'tmp_2159_chunked_int_src.tif', 'tmp_2159_chunked_int_unmasked.vrt')  # noqa: E501
    out = read_geotiff_dask(vrt, chunks=2, mask_nodata=False)
    assert out.dtype.kind == 'i'
    assert out.attrs.get('masked_nodata') is False


def test_vrt_chunked_float_source_mask_off_with_cast_reports_false(tmp_path):
    """Chunked VRT + float source + ``mask_nodata=False`` + ``dtype=float64``
    cast. Same logic as the eager equivalent: caller opted out of
    masking, attr is False even though the lazy graph dtype is float."""
    vrt = _masked_nodata_attr_write_float_vrt(tmp_path, 'tmp_2159_chunked_float_src_cast.tif', 'tmp_2159_chunked_unmasked_cast.vrt')  # noqa: E501
    out = read_geotiff_dask(vrt, chunks=2, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'


def test_vrt_attr_matches_dask_backend_under_mask_off(tmp_path):
    """Both VRT backends should report the same ``masked_nodata`` as
    the regular dask backend does for an equivalent input. Pins the
    cross-backend invariant the contract at
    ``_attrs._set_nodata_attrs`` calls out."""
    vrt = _masked_nodata_attr_write_float_vrt(tmp_path, 'tmp_2159_xbackend_src.tif', 'tmp_2159_xbackend.vrt')  # noqa: E501
    eager = open_geotiff(vrt, mask_nodata=False, dtype=np.float64)
    chunked = read_geotiff_dask(vrt, chunks=2, mask_nodata=False, dtype=np.float64)
    assert eager.attrs.get('masked_nodata') is False
    assert chunked.attrs.get('masked_nodata') is False
    assert eager.attrs.get('masked_nodata') == chunked.attrs.get('masked_nodata')


# ---------------------------------------------------------------------------
# per-band <NoDataValue> selection
# ---------------------------------------------------------------------------


def _band_nodata_write_two_band_per_band_nodata_vrt(tmp_path):
    """Two single-band uint16 sources, each with a distinct nodata
    sentinel, exposed as bands 1 and 2 of a hand-rolled VRT.
    """
    band0 = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, 65000]], dtype=np.uint16)
    p0 = str(tmp_path / 'vrt_band0_1598.tif')
    p1 = str(tmp_path / 'vrt_band1_1598.tif')
    write(band0, p0, nodata=65535, compression='none', tiled=False)
    write(band1, p1, nodata=65000, compression='none', tiled=False)
    vrt_path = str(tmp_path / 'two_band_per_band_nodata_1598.vrt')
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <NoDataValue>65535</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n  <VRTRasterBand dataType="UInt16" band="2">\n    <NoDataValue>65000</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_read_vrt_band0_uses_band0_nodata(tmp_path):
    """Sanity check the band-0 selection still works after the fix.

    Confirms the refactor did not flip the index.

    The fixture mosaics two bands with distinct per-band sentinels, so
    the default read raises ``MixedBandMetadataError``.
    The older flatten-to-first-band semantics this regression tests
    are still reachable via ``band_nodata='first'``; the opt-in surfaces
    at the call site that the test is exercising the legacy behaviour.
    """
    vrt_path = _band_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band=0, band_nodata='first')
    assert r.dtype == np.float64
    assert r.attrs.get('nodata') == 65535.0
    assert np.isnan(r.values[1, 1])
    assert r.values[0, 0] == 1


def test_read_vrt_band1_uses_band1_nodata(tmp_path):
    """The previously-broken case: band = 1 must use band 1's sentinel.

    Before the fix this returned dtype=uint16 with values=[[7,8],
    [9,65000]] and attrs['nodata']=65535.
    """
    vrt_path = _band_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band=1, band_nodata='first')
    assert r.dtype == np.float64, 'band=1 read kept uint16 dtype; per-band nodata regression.'
    assert r.attrs.get('nodata') == 65000.0, f"attrs['nodata'] was {r.attrs.get('nodata')}, expected 65000 from band 1's <NoDataValue>."  # noqa: E501
    assert np.isnan(r.values[1, 1]), "band 1's sentinel pixel was not NaN-masked; promotion ran against the wrong sentinel."  # noqa: E501
    assert r.values[0, 0] == 7
    assert r.values[1, 0] == 9


def test_read_vrt_no_band_keeps_band0_nodata_attr(tmp_path):
    """Unselected reads still surface band 0's sentinel.

    Multi-band VRTs with mixed sentinels return all bands stacked, and
    the canonical attr cannot encode per-band values; advertising
    band 0's sentinel matches the prior behavior and the documented
    "first band wins" contract for multi-band reads.
    """
    vrt_path = _band_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band_nodata='first')
    assert r.attrs.get('nodata') == 65535.0


def test_read_vrt_negative_band_raises(tmp_path):
    """Negative band indices used to be silently accepted via Python
    list indexing (``vrt.bands[-1]`` returned the last band) while the
    public reader's nodata lookup rejected them, producing band-N data
    with no nodata sentinel. They are now a clear ValueError up front.
    """
    vrt_path = _band_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    with pytest.raises(ValueError, match='band'):
        read_vrt(vrt_path, band=-1)


def test_read_vrt_out_of_range_band_raises(tmp_path):
    """Out-of-range band indices used to raise IndexError from deep in
    the read path. They are now a ValueError that names the available
    band count.
    """
    vrt_path = _band_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    with pytest.raises(ValueError, match='out of range'):
        read_vrt(vrt_path, band=5, band_nodata='first')


def test_read_vrt_non_integer_band_raises(tmp_path):
    """A non-int ``band`` would previously have raised TypeError on the
    list index. ValueError here matches the rest of the input
    validation surface.
    """
    vrt_path = _band_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    with pytest.raises(ValueError, match='band'):
        read_vrt(vrt_path, band='1')
    with pytest.raises(ValueError, match='band'):
        read_vrt(vrt_path, band=True)


# ---------------------------------------------------------------------------
# SimpleSource <NODATA>0</NODATA> survives
# ---------------------------------------------------------------------------


def _source_nodata_zero_write_source(tmp_path, arr, name='src_1655.tif'):
    """Write a small float32 GeoTIFF without a GDAL_NODATA tag."""
    p = str(tmp_path / name)
    write(arr, p, geo_transform=GeoTransform(origin_x=0.0, origin_y=0.0, pixel_width=1.0, pixel_height=-1.0), crs_epsg=4326, compression='none', tiled=False)  # noqa: E501
    return p


def _source_nodata_zero_vrt_with_source_nodata(tmp_path, src_path, nodata_xml, include_band_nodata=False, width=4, height=3, band_nodata='0.0'):  # noqa: E501
    """Write a single-band Float32 VRT with the supplied ``<NODATA>``
    on its SimpleSource. ``include_band_nodata`` controls whether a
    ``<NoDataValue>`` is emitted on the band as well.
    """
    band_nd_elem = f'<NoDataValue>{band_nodata}</NoDataValue>' if include_band_nodata else ''
    vrt_xml = f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n  <SRS>EPSG:4326</SRS>\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="Float32" band="1">\n    {band_nd_elem}\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n      <NODATA>{nodata_xml}</NODATA>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    vrt_path = str(tmp_path / 'src_zero_1655.vrt')
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


class TestVRTSourceNodataZero:
    """SimpleSource ``<NODATA>0</NODATA>`` must mask zeros to NaN."""

    def test_source_nodata_zero_no_band_nodata(self, tmp_path):
        """SimpleSource NODATA=0 with no band-level fallback masks zeros."""
        arr = np.array([[1.0, 0.0, 3.0, 0.0], [4.0, 0.0, 6.0, 7.0], [0.0, 8.0, 9.0, 10.0]], dtype=np.float32)  # noqa: E501
        src = _source_nodata_zero_write_source(tmp_path, arr)
        vrt = _source_nodata_zero_vrt_with_source_nodata(tmp_path, src, '0.0')
        result, _ = _source_nodata_zero_read_vrt_internal(vrt)
        assert int(np.isnan(result).sum()) == 4

    def test_source_nodata_zero_integer_xml(self, tmp_path):
        """``<NODATA>0</NODATA>`` (integer literal) also masks zeros."""
        arr = np.array([[1.0, 0.0, 3.0]], dtype=np.float32)
        src = _source_nodata_zero_write_source(tmp_path, arr, name='int_xml.tif')
        vrt = _source_nodata_zero_vrt_with_source_nodata(tmp_path, src, '0', width=3, height=1)
        result, _ = _source_nodata_zero_read_vrt_internal(vrt)
        assert int(np.isnan(result).sum()) == 1
        assert np.isnan(result[0, 1])

    def test_source_nodata_nonzero_unchanged(self, tmp_path):
        """SimpleSource NODATA != 0 keeps masking behaviour."""
        arr = np.array([[1.0, 0.0, 3.0, 0.0]], dtype=np.float32)
        src = _source_nodata_zero_write_source(tmp_path, arr, name='nonzero.tif')
        vrt = _source_nodata_zero_vrt_with_source_nodata(tmp_path, src, '1.0', width=4, height=1)
        result, _ = _source_nodata_zero_read_vrt_internal(vrt)
        assert int(np.isnan(result).sum()) == 1
        assert np.isnan(result[0, 0])

    def test_band_nodata_zero_still_honoured(self, tmp_path):
        """Band-level ``<NoDataValue>0</NoDataValue>`` keeps working."""
        arr = np.array([[1.0, 0.0, 3.0]], dtype=np.float32)
        src = _source_nodata_zero_write_source(tmp_path, arr, name='band_zero.tif')
        vrt_xml = f'<VRTDataset rasterXSize="3" rasterYSize="1">\n  <SRS>EPSG:4326</SRS>\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="Float32" band="1">\n    <NoDataValue>0.0</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="3" ySize="1"/>\n      <DstRect xOff="0" yOff="0" xSize="3" ySize="1"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
        vrt = str(tmp_path / 'band_zero_1655.vrt')
        with open(vrt, 'w') as f:
            f.write(vrt_xml)
        result, _ = _source_nodata_zero_read_vrt_internal(vrt)
        assert int(np.isnan(result).sum()) == 1
        assert np.isnan(result[0, 1])

    def test_source_nodata_zero_overrides_band(self, tmp_path):
        """SimpleSource NODATA=0 takes precedence over band NoDataValue=99."""
        arr = np.array([[1.0, 0.0, 99.0]], dtype=np.float32)
        src = _source_nodata_zero_write_source(tmp_path, arr, name='override.tif')
        vrt = _source_nodata_zero_vrt_with_source_nodata(tmp_path, src, '0.0', include_band_nodata=True, band_nodata='99.0', width=3, height=1)  # noqa: E501
        result, _ = _source_nodata_zero_read_vrt_internal(vrt)
        assert int(np.isnan(result).sum()) == 1
        assert np.isnan(result[0, 1])
        assert result[0, 2] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# integer-with-nodata promotion
# ---------------------------------------------------------------------------


def _int_nodata_write_uint16_with_nodata_tif(path, sentinel):
    """Write a small uint16 GeoTIFF with a nodata sentinel."""
    arr = np.array([[1, 2, 3], [sentinel, 5, 6]], dtype=np.uint16)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': np.arange(2), 'x': np.arange(3)}, attrs={'crs': 4326, 'nodata': sentinel})  # noqa: E501
    to_geotiff(da, path, compression='none', nodata=sentinel)
    return arr


def test_vrt_uint16_nodata_promotes_to_float64(tmp_path):
    """VRT route NaN-masks integer-with-nodata, matching open_geotiff."""
    tif = str(tmp_path / 'src_1564.tif')
    _int_nodata_write_uint16_with_nodata_tif(tif, sentinel=65535)
    eager = open_geotiff(tif)
    assert eager.dtype == np.float64
    assert np.isnan(eager.values[1, 0])
    vrt_path = str(tmp_path / 'src_1564.vrt')
    write_vrt(vrt_path, [tif])
    via_vrt = read_vrt(vrt_path)
    assert via_vrt.dtype == np.float64, f'VRT integer-with-nodata should promote to float64; got {via_vrt.dtype}'  # noqa: E501
    assert np.isnan(via_vrt.values[1, 0]), f'VRT sentinel pixel should be NaN; got {via_vrt.values[1, 0]} (literal sentinel survived)'  # noqa: E501
    assert via_vrt.attrs.get('nodata') == 65535.0


def test_vrt_uint16_no_nodata_keeps_dtype(tmp_path):
    """Without a nodata sentinel, the dtype stays integer."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': np.arange(2), 'x': np.arange(3)}, attrs={'crs': 4326})  # noqa: E501
    tif = str(tmp_path / 'src_no_nodata_1564.tif')
    to_geotiff(da, tif, compression='none')
    vrt_path = str(tmp_path / 'src_no_nodata_1564.vrt')
    write_vrt(vrt_path, [tif])
    via_vrt = read_vrt(vrt_path)
    assert via_vrt.dtype == np.uint16
    np.testing.assert_array_equal(via_vrt.values, arr)


def test_vrt_float_nodata_still_masks(tmp_path):
    """Regression guard: the existing float-with-nodata branch still
    works after the integer-branch addition."""
    arr = np.array([[1.0, 2.0, -9999.0], [4.0, -9999.0, 6.0]], dtype=np.float32)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': np.arange(2), 'x': np.arange(3)}, attrs={'crs': 4326, 'nodata': -9999.0})  # noqa: E501
    tif = str(tmp_path / 'srcf_1564.tif')
    to_geotiff(da, tif, compression='none', nodata=-9999.0)
    vrt_path = str(tmp_path / 'srcf_1564.vrt')
    write_vrt(vrt_path, [tif])
    via_vrt = read_vrt(vrt_path)
    assert via_vrt.dtype == np.float32
    assert np.isnan(via_vrt.values[0, 2])
    assert np.isnan(via_vrt.values[1, 1])


def _int_nodata_rewrite_vrt_nodata(vrt_path, new_nodata_text):
    """Rewrite the <NoDataValue> element of an existing VRT to a literal
    string so we can exercise fractional / out-of-range cases without
    going through ``write_vrt`` (which only accepts numeric values)."""
    with open(vrt_path, 'r') as f:
        xml = f.read()
    import re
    new_xml, n = re.subn('<NoDataValue>[^<]*</NoDataValue>', f'<NoDataValue>{new_nodata_text}</NoDataValue>', xml)  # noqa: E501
    assert n == 1, f'expected 1 NoDataValue element, found {n}'
    with open(vrt_path, 'w') as f:
        f.write(new_xml)


def test_vrt_fractional_nodata_is_not_masked(tmp_path):
    """Fractional VRT NoDataValue against an integer band must NOT mask:
    truncating to int would alias a real pixel value as nodata."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': np.arange(2), 'x': np.arange(3)}, attrs={'crs': 4326, 'nodata': 1})  # noqa: E501
    tif = str(tmp_path / 'frac_1564.tif')
    to_geotiff(da, tif, compression='none', nodata=1)
    vrt_path = str(tmp_path / 'frac_1564.vrt')
    write_vrt(vrt_path, [tif])
    _int_nodata_rewrite_vrt_nodata(vrt_path, '1.9')
    via_vrt = read_vrt(vrt_path)
    assert via_vrt.dtype == np.uint16, f'Fractional NoDataValue must not trigger integer masking (got dtype {via_vrt.dtype}, pixel @[0,0]={via_vrt.values[0, 0]})'  # noqa: E501
    np.testing.assert_array_equal(via_vrt.values, arr)


def test_vrt_out_of_range_nodata_is_not_masked(tmp_path):
    """NoDataValue outside the dtype range must NOT mask: casting would
    wrap and alias an in-range pixel."""
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': np.arange(2), 'x': np.arange(3)}, attrs={'crs': 4326, 'nodata': 0})  # noqa: E501
    tif = str(tmp_path / 'oor_1564.tif')
    to_geotiff(da, tif, compression='none', nodata=0)
    vrt_path = str(tmp_path / 'oor_1564.vrt')
    write_vrt(vrt_path, [tif])
    _int_nodata_rewrite_vrt_nodata(vrt_path, '-1')
    via_vrt = read_vrt(vrt_path)
    assert via_vrt.dtype == np.uint16, f'Out-of-range NoDataValue must not trigger integer masking (got dtype {via_vrt.dtype})'  # noqa: E501
    np.testing.assert_array_equal(via_vrt.values, arr)


def test_vrt_open_geotiff_parity_uint16_nodata(tmp_path):
    """open_geotiff routing a .vrt path should produce the same dtype
    and masked positions as a direct GeoTIFF read."""
    tif = str(tmp_path / 'parity_1564.tif')
    _int_nodata_write_uint16_with_nodata_tif(tif, sentinel=65535)
    direct = open_geotiff(tif)
    vrt_path = str(tmp_path / 'parity_1564.vrt')
    write_vrt(vrt_path, [tif])
    via_vrt = open_geotiff(vrt_path)
    assert direct.dtype == via_vrt.dtype
    np.testing.assert_array_equal(np.isnan(direct.values), np.isnan(via_vrt.values), err_msg='VRT route should NaN-mask the same pixels as direct read')  # noqa: E501
    mask = ~np.isnan(direct.values)
    np.testing.assert_array_equal(direct.values[mask], via_vrt.values[mask])


# ---------------------------------------------------------------------------
# mask_nodata=False preserves float sentinels
# ---------------------------------------------------------------------------


def _mask_nodata_float_write_float32_with_sentinel(tmp_path, sentinel=-9999.0, filename='float_2158.tif'):  # noqa: E501
    """float32 GeoTIFF with a non-NaN sentinel and matching pixels.

    The middle row has a literal ``-9999.0`` so the inline masking
    actually has something to rewrite.
    """
    band = np.array([[1.0, 2.0, 3.0], [4.0, sentinel, 6.0], [7.0, sentinel, 9.0]], dtype=np.float32)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return (p, band)


def _mask_nodata_float_write_float64_with_fractional_sentinel(tmp_path, sentinel=-9999.25, filename='float64_2158.tif'):  # noqa: E501
    """float64 GeoTIFF with a fractional sentinel.

    Float32's exact-cast rounding would clobber a fractional value
    like ``-9999.25``; the float64 path is the only one where the
    sentinel survives lossless.
    """
    band = np.array([[1.0, 2.0], [sentinel, 4.0]], dtype=np.float64)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return (p, band)


def _mask_nodata_float_build_vrt(tmp_path, source_path, vrt_dtype, nodata_value, filename='float_2158.vrt', shape=(3, 3)):  # noqa: E501
    """Hand-roll a single-source VRT pointing at the float source."""
    h, w = shape
    vrt_xml = f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="{vrt_dtype}" band="1">\n    <NoDataValue>{nodata_value}</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{source_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>\n      <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    p = str(tmp_path / filename)
    with open(p, 'w') as f:
        f.write(vrt_xml)
    return p


def test_default_mask_nodata_true_rewrites_float_sentinel(tmp_path):
    """The default behaviour (mask_nodata=True) still substitutes NaN.

    Pins the existing contract so the fix below does not regress the
    masking happy path.
    """
    src, _ = _mask_nodata_float_write_float32_with_sentinel(tmp_path)
    vrt = _mask_nodata_float_build_vrt(tmp_path, src, 'Float32', -9999.0)
    r = read_vrt(vrt)
    assert r.dtype == np.float32
    assert np.isnan(r.values[1, 1])
    assert np.isnan(r.values[2, 1])
    assert r.values[0, 0] == 1.0
    assert r.values[1, 0] == 4.0
    assert r.attrs.get('nodata') == -9999.0
    assert r.attrs.get('masked_nodata') is True


def test_eager_mask_nodata_false_preserves_float_sentinel(tmp_path):
    """Eager VRT path: ``mask_nodata=False`` keeps the literal sentinel.

    Previously this assertion failed -- the sentinel pixels were
    silently rewritten to NaN inside ``_vrt._read_data`` regardless
    of the kwarg.
    """
    src, original = _mask_nodata_float_write_float32_with_sentinel(tmp_path)
    vrt = _mask_nodata_float_build_vrt(tmp_path, src, 'Float32', -9999.0)
    r = read_vrt(vrt, mask_nodata=False)
    assert r.dtype == np.float32
    assert not np.isnan(r.values).any()
    assert r.values[1, 1] == np.float32(-9999.0)
    assert r.values[2, 1] == np.float32(-9999.0)
    np.testing.assert_array_equal(r.values, original)
    assert r.attrs.get('nodata') == -9999.0
    assert r.attrs.get('masked_nodata') is False


def test_chunked_mask_nodata_false_preserves_float_sentinel(tmp_path):
    """Chunked VRT path: ``mask_nodata=False`` keeps the literal sentinel.

    The chunked path used to call ``_read_vrt_internal`` from
    ``_vrt_chunk_read`` without forwarding the kwarg, so per-chunk
    decodes silently rewrote float sentinels too. The
    kwarg is now forwarded into the internal reader and both paths agree.
    """
    src, original = _mask_nodata_float_write_float32_with_sentinel(tmp_path)
    vrt = _mask_nodata_float_build_vrt(tmp_path, src, 'Float32', -9999.0)
    r = read_vrt(vrt, chunks=2, mask_nodata=False)
    assert r.dtype == np.float32
    computed = r.compute()
    assert not np.isnan(computed.values).any()
    assert computed.values[1, 1] == np.float32(-9999.0)
    assert computed.values[2, 1] == np.float32(-9999.0)
    np.testing.assert_array_equal(computed.values, original)
    assert computed.attrs.get('nodata') == -9999.0
    assert computed.attrs.get('masked_nodata') is False


def test_eager_and_chunked_agree_under_mask_nodata_false(tmp_path):
    """Cross-path parity: eager and chunked produce the same buffer.

    Previously the two paths could disagree because both rewrote
    the sentinel inline but at slightly different points in the
    pipeline. With the opt-out honored, both paths land on the
    untouched source array.
    """
    src, _ = _mask_nodata_float_write_float32_with_sentinel(tmp_path)
    vrt = _mask_nodata_float_build_vrt(tmp_path, src, 'Float32', -9999.0)
    eager = read_vrt(vrt, mask_nodata=False)
    chunked = read_vrt(vrt, chunks=2, mask_nodata=False).compute()
    np.testing.assert_array_equal(eager.values, chunked.values)
    assert eager.attrs.get('masked_nodata') == chunked.attrs.get('masked_nodata')


def test_mask_nodata_false_float64_fractional_sentinel(tmp_path):
    """A fractional sentinel survives the float64 opt-out path.

    Float32 would round ``-9999.25`` to the nearest representable
    value, so this corner is float64-only. With the opt-out honored
    the pixel keeps its exact bit pattern.
    """
    src, original = _mask_nodata_float_write_float64_with_fractional_sentinel(tmp_path)
    vrt = _mask_nodata_float_build_vrt(tmp_path, src, 'Float64', -9999.25, filename='float64_2158.vrt', shape=(2, 2))  # noqa: E501
    r = read_vrt(vrt, mask_nodata=False)
    assert r.dtype == np.float64
    assert r.values[1, 0] == -9999.25
    np.testing.assert_array_equal(r.values, original)


def test_masked_vs_unmasked_differ_only_at_sentinels(tmp_path):
    """``mask_nodata=True`` and ``=False`` differ only where the sentinel hits.

    Every pixel that is NaN in the masked output equals the declared
    sentinel in the unmasked output, and every non-sentinel pixel is
    bit-identical between the two reads. This pins the contract that
    the opt-out is a pure passthrough on the non-sentinel positions.
    """
    src, _ = _mask_nodata_float_write_float32_with_sentinel(tmp_path)
    vrt = _mask_nodata_float_build_vrt(tmp_path, src, 'Float32', -9999.0)
    masked = read_vrt(vrt).values
    unmasked = read_vrt(vrt, mask_nodata=False).values
    nan_positions = np.isnan(masked)
    sentinel_positions = unmasked == np.float32(-9999.0)
    np.testing.assert_array_equal(nan_positions, sentinel_positions)
    np.testing.assert_array_equal(masked[~nan_positions], unmasked[~sentinel_positions])


def _mask_nodata_float_write_uint16_with_sentinel(tmp_path, sentinel=65535, filename='uint16_2158.tif'):  # noqa: E501
    """uint16 GeoTIFF with a matching sentinel.

    Used to exercise the integer-source-feeding-float-VRT promotion at
    ``_vrt.py:1351-1390``. With ``mask_nodata=True`` the sentinel pixel
    surfaces as NaN in the float buffer; with ``mask_nodata=False`` the
    literal integer value flows through the int->float cast and lands
    as ``65535.0``.
    """
    band = np.array([[1, 2], [3, sentinel]], dtype=np.uint16)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return (p, band)


def test_int_source_float_vrt_mask_nodata_false_keeps_literal(tmp_path):
    """Integer source feeding a Float32 VRT preserves the literal sentinel.

    Pins the second branch of the inline masking opt-out.
    Before the fix, ``_vrt._read_data`` ran the int->float-with-NaN
    promotion unconditionally, so even ``mask_nodata=False`` lost the
    sentinel. After the fix the integer source pixel survives the
    int->float cast as ``65535.0`` and ``masked_nodata`` reflects
    that no masking ran.
    """
    src, _ = _mask_nodata_float_write_uint16_with_sentinel(tmp_path)
    vrt = _mask_nodata_float_build_vrt(tmp_path, src, 'Float32', 65535, filename='int_float_2158.vrt', shape=(2, 2))  # noqa: E501
    r = read_vrt(vrt, mask_nodata=False)
    assert r.dtype == np.float32
    assert not np.isnan(r.values).any()
    assert r.values[1, 1] == np.float32(65535.0)
    assert r.values[0, 0] == 1.0
    assert r.attrs.get('nodata') == 65535.0
    assert r.attrs.get('masked_nodata') is False


def test_int_source_float_vrt_default_still_promotes(tmp_path):
    """Default ``mask_nodata=True`` still NaN-masks the int->float promotion.

    Baseline that documents the default contract for the integer
    source path: the int->float NaN-promotion behavior is unchanged
    when the opt-out is not requested.
    """
    src, _ = _mask_nodata_float_write_uint16_with_sentinel(tmp_path)
    vrt = _mask_nodata_float_build_vrt(tmp_path, src, 'Float32', 65535, filename='int_float_default_2158.vrt', shape=(2, 2))  # noqa: E501
    r = read_vrt(vrt)
    assert r.dtype == np.float32
    assert np.isnan(r.values[1, 1])
    assert r.values[0, 0] == 1.0
    assert r.attrs.get('nodata') == 65535.0
    assert r.attrs.get('masked_nodata') is True


# ---------------------------------------------------------------------------
# tile metadata parity for VRT tiled writes
# ---------------------------------------------------------------------------


def _tiled_metadata_make_rioxarray_style(arr=None):
    """DataArray that looks like rioxarray output: nodata only via aliases."""
    if arr is None:
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        arr[0, 0] = -9999.0
    return xr.DataArray(arr, dims=('y', 'x'), coords={'y': np.arange(arr.shape[0], dtype=np.float64), 'x': np.arange(arr.shape[1], dtype=np.float64)}, attrs={'nodatavals': (-9999.0,), '_FillValue': -9999.0, 'crs': 4326, 'gdal_metadata': {'AREA_OR_POINT': 'Area', 'foo': 'bar'}, 'x_resolution': 96, 'y_resolution': 96, 'resolution_unit': 'inch', 'raster_type': 'point'})  # noqa: E501


def _tiled_metadata_first_tile_path(vrt_path):
    tiles_dir = vrt_path[:-len('.vrt')] + '_tiles'
    tiles = sorted(glob.glob(os.path.join(tiles_dir, '*.tif')))
    assert tiles, f'no per-tile .tif files under {tiles_dir}'
    return tiles[0]


class TestVrtTiledMetadataParity:

    def test_nodatavals_alias_propagates_to_tiles(self, tmp_path):
        da = _tiled_metadata_make_rioxarray_style()
        vrt = str(tmp_path / 'nodatavals.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        assert tile_da.attrs.get('nodata') == -9999.0

    def test_fill_value_alias_propagates_to_tiles(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        arr[0, 0] = -9999.0
        da = xr.DataArray(arr, dims=('y', 'x'), coords={'y': np.arange(8.0), 'x': np.arange(8.0)}, attrs={'_FillValue': -9999.0, 'crs': 4326})  # noqa: E501
        vrt = str(tmp_path / 'fillvalue.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        assert tile_da.attrs.get('nodata') == -9999.0

    def test_gdal_metadata_propagates_to_tiles(self, tmp_path):
        da = _tiled_metadata_make_rioxarray_style()
        vrt = str(tmp_path / 'gdal_meta.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        gm = tile_da.attrs.get('gdal_metadata')
        assert gm == {'AREA_OR_POINT': 'Area', 'foo': 'bar'}

    def test_resolution_tags_propagate_to_tiles(self, tmp_path):
        da = _tiled_metadata_make_rioxarray_style()
        vrt = str(tmp_path / 'resolution.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        assert tile_da.attrs.get('x_resolution') == 96.0
        assert tile_da.attrs.get('y_resolution') == 96.0
        assert tile_da.attrs.get('resolution_unit') == 'inch'

    def test_raster_type_point_propagates_to_tiles(self, tmp_path):
        da = _tiled_metadata_make_rioxarray_style()
        vrt = str(tmp_path / 'point.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        assert tile_da.attrs.get('raster_type') == 'point'

    def test_tif_vs_vrt_tile_metadata_parity(self, tmp_path):
        """Same DataArray, two destinations -- per-tile metadata matches."""
        da = _tiled_metadata_make_rioxarray_style()
        tif_path = str(tmp_path / 'parity.tif')
        vrt_path = str(tmp_path / 'parity.vrt')
        to_geotiff(da, tif_path, tile_size=16)
        to_geotiff(da, vrt_path, tile_size=16)
        tif_da = open_geotiff(tif_path)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt_path))
        keys = ('nodata', 'gdal_metadata', 'raster_type', 'x_resolution', 'y_resolution', 'resolution_unit')  # noqa: E501
        for k in keys:
            assert tif_da.attrs.get(k) == tile_da.attrs.get(k), f'{k} mismatch: tif = {tif_da.attrs.get(k)!r}, vrt-tile={tile_da.attrs.get(k)!r}'  # noqa: E501


class TestVrtTiledRichTagCoverage:
    """Cover the XML / extra_tags / friendly-tag paths the bare
    ``gdal_metadata`` dict assertion above does not exercise."""

    def test_gdal_metadata_xml_string_propagates_to_tiles(self, tmp_path):
        """``attrs['gdal_metadata_xml']`` (pre-built XML string) bypasses
        the dict->XML builder. Verify it still reaches per-tile files."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        xml = '<GDALMetadata>\n  <Item name="VRT_XML_KEY">vrt_xml_value</Item>\n</GDALMetadata>\n'
        da = xr.DataArray(arr, dims=('y', 'x'), coords={'y': np.arange(8.0), 'x': np.arange(8.0)}, attrs={'crs': 4326, 'gdal_metadata_xml': xml})  # noqa: E501
        vrt = str(tmp_path / 'gdal_xml.vrt')
        to_geotiff(da, vrt, tile_size=16, allow_experimental_codecs=True)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        gm = tile_da.attrs.get('gdal_metadata') or {}
        gm_xml = tile_da.attrs.get('gdal_metadata_xml') or ''
        assert gm.get('VRT_XML_KEY') == 'vrt_xml_value' or 'VRT_XML_KEY' in gm_xml, f'gdal_metadata_xml content lost on VRT-tile round-trip; gdal_metadata={gm!r}, gdal_metadata_xml={gm_xml!r}'  # noqa: E501

    def test_extra_tags_entry_propagates_to_tiles(self, tmp_path):
        """A user-supplied ``extra_tags`` entry (Software, tag 305)
        must round-trip through the VRT-tiled writer."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        software = 'xrspatial-vrt-test-1606'
        da = xr.DataArray(arr, dims=('y', 'x'), coords={'y': np.arange(8.0), 'x': np.arange(8.0)}, attrs={'crs': 4326, 'extra_tags': [(305, 2, len(software) + 1, software)]})  # noqa: E501
        vrt = str(tmp_path / 'extra_tags.vrt')
        to_geotiff(da, vrt, tile_size=16, allow_experimental_codecs=True)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        et = tile_da.attrs.get('extra_tags') or []
        tag_ids = {entry[0] for entry in et}
        assert 305 in tag_ids, f'Software (305) tag missing from VRT tile extra_tags; got tag ids {sorted(tag_ids)!r}'  # noqa: E501

    def test_image_description_friendly_attr_propagates_to_tiles(self, tmp_path):
        """``attrs['image_description']`` is folded into ``extra_tags``
        as tag 270 by ``_merge_friendly_extra_tags`` and then surfaces
        on read as ``attrs['image_description']``."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        da = xr.DataArray(arr, dims=('y', 'x'), coords={'y': np.arange(8.0), 'x': np.arange(8.0)}, attrs={'crs': 4326, 'image_description': 'vrt-tile-friendly-1606'})  # noqa: E501
        vrt = str(tmp_path / 'image_desc.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        assert tile_da.attrs.get('image_description') == 'vrt-tile-friendly-1606'


class TestVrtTiledMetadataDask:

    def test_nodatavals_alias_dask(self, tmp_path):
        pytest.importorskip('dask.array')
        import dask.array as dska
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        arr[0, 0] = -9999.0
        da_np = xr.DataArray(arr, dims=('y', 'x'), coords={'y': np.arange(8.0), 'x': np.arange(8.0)}, attrs={'nodatavals': (-9999.0,), 'crs': 4326, 'gdal_metadata': {'k': 'v'}})  # noqa: E501
        da = xr.DataArray(dska.from_array(arr, chunks=4), dims=da_np.dims, coords=da_np.coords, attrs=da_np.attrs)  # noqa: E501
        vrt = str(tmp_path / 'dask.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_tiled_metadata_first_tile_path(vrt))
        assert tile_da.attrs.get('nodata') == -9999.0
        assert tile_da.attrs.get('gdal_metadata') == {'k': 'v'}


# ---------------------------------------------------------------------------
# VRT XML parsed once on the chunked path
# ---------------------------------------------------------------------------


@pytest.fixture
def single_parse_two_by_two_vrt_1825():
    """4-tile mosaic via the to_geotiff(.vrt, ...) dask path."""
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    y = np.linspace(41.0, 40.0, 256)
    x = np.linspace(-106.0, -105.0, 256)
    raster = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1825_2x2_')
    vrt_path = os.path.join(td, 'mosaic_1825.vrt')
    to_geotiff(raster, vrt_path, tile_size=128)
    yield (vrt_path, arr)


@pytest.fixture
def single_parse_single_tile_vrt_1825():
    """One 64x64 float32 tile wrapped in a VRT."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    y = np.linspace(41.0, 40.0, 64)
    x = np.linspace(-106.0, -105.0, 64)
    raster = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1825_single_')
    tile_path = os.path.join(td, 'tile_1825.tif')
    to_geotiff(raster, tile_path)
    vrt_path = os.path.join(td, 'single_1825.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    yield (vrt_path, arr)


def test_chunked_path_parses_xml_once(monkeypatch, single_parse_two_by_two_vrt_1825):
    """Construction parses once, and ``.compute()`` adds zero parses.

    The previous implementation re-parsed inside every per-chunk task,
    so a 4x4 chunk grid produced 17 parses total. The
    dispatcher now parses once and threads the already-parsed VRTDataset
    through the task graph.
    """
    vrt_path, _ = single_parse_two_by_two_vrt_1825
    from xrspatial.geotiff import _vrt as vrt_module
    counter = {'parses': 0}
    real_parse = vrt_module.parse_vrt

    def counting_parse(*args, **kwargs):
        counter['parses'] += 1
        return real_parse(*args, **kwargs)
    monkeypatch.setattr(vrt_module, 'parse_vrt', counting_parse)
    result = read_vrt(vrt_path, chunks=(64, 64))
    assert counter['parses'] == 1, f"expected 1 parse during construction, got {counter['parses']}"
    computed = result.compute()
    assert counter['parses'] == 1, f"expected 1 parse total (construction only); got {counter['parses']} -- per-chunk tasks are still reparsing"  # noqa: E501
    assert computed.shape == (256, 256)
    assert computed.dtype == np.float32


def test_chunked_path_reads_xml_file_once(monkeypatch, single_parse_two_by_two_vrt_1825):
    """The chunked dispatcher reads the VRT XML file exactly once.

    Pin the file-read side too: previously every per-chunk task
    re-opened the .vrt file via ``_read_vrt_xml``. After the refactor
    only the dispatcher reads it.
    """
    vrt_path, _ = single_parse_two_by_two_vrt_1825
    from xrspatial.geotiff import _vrt as vrt_module
    counter = {'reads': 0}
    real_read_xml = vrt_module._read_vrt_xml

    def counting_read_xml(*args, **kwargs):
        counter['reads'] += 1
        return real_read_xml(*args, **kwargs)
    monkeypatch.setattr(vrt_module, '_read_vrt_xml', counting_read_xml)
    result = read_vrt(vrt_path, chunks=(64, 64))
    assert counter['reads'] == 1, f"expected 1 XML file read during construction, got {counter['reads']}"  # noqa: E501
    result.compute()
    assert counter['reads'] == 1, f"expected 1 XML file read total; got {counter['reads']} -- per-chunk tasks are still re-opening the .vrt file"  # noqa: E501


def test_parsed_vrt_is_picklable(single_parse_single_tile_vrt_1825):
    """The parsed VRTDataset round-trips through pickle.

    The chunked dispatcher embeds the parsed VRT into the dask graph,
    so dask must be able to serialise it for the distributed and
    process-pool schedulers. Pin picklability with the stdlib pickler
    (cloudpickle is a strict superset).
    """
    vrt_path, _ = single_parse_single_tile_vrt_1825
    from xrspatial.geotiff._vrt import _read_vrt_xml, parse_vrt
    xml_str = _read_vrt_xml(vrt_path)
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    vrt = parse_vrt(xml_str, vrt_dir)
    blob = pickle.dumps(vrt)
    restored = pickle.loads(blob)
    assert restored.width == vrt.width
    assert restored.height == vrt.height
    assert len(restored.bands) == len(vrt.bands)
    assert restored.bands[0].dtype == vrt.bands[0].dtype
    assert [s.filename for s in restored.bands[0].sources] == [s.filename for s in vrt.bands[0].sources]  # noqa: E501


def test_chunked_matches_eager_after_refactor(single_parse_two_by_two_vrt_1825):
    """Byte-identical eager vs chunked results after the helper consolidation.

    The eager path uses ``_apply_integer_sentinel_mask`` /
    ``_effective_dtype_for_bands`` / ``_sentinel_for_dtype`` from
    ``_vrt`` directly; the chunked path imports the same helpers. A
    regression in either call site would surface here.
    """
    vrt_path, original = single_parse_two_by_two_vrt_1825
    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=(64, 64)).compute()
    assert eager.dtype == chunked.dtype
    np.testing.assert_array_equal(eager.values, chunked.values)
    np.testing.assert_array_equal(eager.values, original)


def test_no_path_containment_revalidation_per_chunk(monkeypatch, single_parse_two_by_two_vrt_1825):
    """Per-chunk tasks skip the source-path containment check.

    ``parse_vrt`` is the only place that resolves and validates source
    paths against the VRT directory / ``XRSPATIAL_VRT_ALLOWED_ROOTS``.
    Because each task now receives the already-parsed VRT, ``parse_vrt``
    must not run during ``.compute()`` even when the graph is hydrated.
    """
    vrt_path, _ = single_parse_two_by_two_vrt_1825
    from xrspatial.geotiff import _vrt as vrt_module
    parse_calls = {'n': 0}
    real_parse = vrt_module.parse_vrt

    def counting_parse(*args, **kwargs):
        parse_calls['n'] += 1
        return real_parse(*args, **kwargs)
    monkeypatch.setattr(vrt_module, 'parse_vrt', counting_parse)
    result = read_vrt(vrt_path, chunks=(64, 64))
    parses_after_construction = parse_calls['n']
    da_arr = result.data
    if isinstance(da_arr, da.Array):
        _block = da_arr.blocks[0, 0].compute()
        assert _block.shape[0] > 0 and _block.shape[1] > 0
    assert parse_calls['n'] == parses_after_construction, f"per-block compute triggered extra parses ({parse_calls['n']} vs {parses_after_construction})"  # noqa: E501


def test_parsed_kwarg_does_not_mutate_caller_holes(single_parse_single_tile_vrt_1825):
    """``read_vrt(parsed=...)`` must not mutate the caller's ``holes``.

    The chunked dispatcher threads a single parsed ``VRTDataset`` into
    every per-chunk task. ``read_vrt`` appends skipped-source records to
    ``vrt.holes`` when a backing file is missing; without a defensive
    copy the appends would land on the dispatcher's shared object and
    leak across tasks (racy under the threaded scheduler, and
    cumulatively across calls if a caller ever reused the parsed
    object). Pin that ``parsed.holes`` stays untouched.
    """
    vrt_path, _ = single_parse_single_tile_vrt_1825
    from xrspatial.geotiff._vrt import _read_vrt_xml, parse_vrt
    from xrspatial.geotiff._vrt import read_vrt as _read_vrt_internal
    xml_str = _read_vrt_xml(vrt_path)
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    parsed = parse_vrt(xml_str, vrt_dir)
    parsed.bands[0].sources[0].filename = os.path.join(vrt_dir, 'gone.tif')
    holes_id_before = id(parsed.holes)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arr, returned = _read_vrt_internal(vrt_path, parsed=parsed, missing_sources='warn')
    assert parsed.holes == [], f'parsed.holes was mutated across the read; got {parsed.holes!r}'
    assert id(parsed.holes) == holes_id_before, "parsed.holes list object was replaced -- the caller's reference is now stale"  # noqa: E501
    assert len(returned.holes) == 1
    assert returned.holes[0]['source'].endswith('gone.tif')
    assert arr.shape == (64, 64)


# ---------------------------------------------------------------------------
# write_vrt escapes XML special chars
# ---------------------------------------------------------------------------


@pytest.fixture
def xml_escape_sample_tif(tmp_path):
    """Write a tiny GeoTIFF the VRT writer can introspect for metadata."""
    arr = np.zeros((4, 4), dtype=np.float32)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'nodata': -9999.0})
    path = str(tmp_path / 'src.tif')
    to_geotiff(da, path)
    return path


def test_crs_wkt_with_xml_special_chars_round_trips(xml_escape_sample_tif, tmp_path):
    """A WKT containing ``& < > " '`` must round-trip through write_vrt /
    parse_vrt unchanged (the entities are escaped on the way out and
    decoded on the way in)."""
    nasty_wkt = 'GEOGCS["spec & <chars> with "quotes" and \'apostrophes\'"]'
    vrt_path = str(tmp_path / 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [xml_escape_sample_tif], crs_wkt=nasty_wkt)
    with open(vrt_path, 'r') as fh:
        text = fh.read()
    parsed = parse_vrt(text, vrt_dir=str(tmp_path))
    assert parsed.crs_wkt == nasty_wkt


def test_crs_wkt_injection_does_not_change_raster_type(xml_escape_sample_tif, tmp_path):
    """The headline XML-injection case: a crafted WKT trying to close ``<SRS>``
    and inject ``<Metadata><MDI key="AREA_OR_POINT">Point</MDI>...``
    must NOT change ``raster_type`` from its default 'area' value."""
    injection = '</SRS><Metadata><MDI key="AREA_OR_POINT">Point</MDI></Metadata><SRS>'
    vrt_path = str(tmp_path / 'evil.vrt')
    _write_vrt_internal(vrt_path, [xml_escape_sample_tif], crs_wkt=injection)
    with open(vrt_path, 'r') as fh:
        text = fh.read()
    parsed = parse_vrt(text, vrt_dir=str(tmp_path))
    assert parsed.raster_type == 'area'
    assert parsed.crs_wkt == injection


def test_source_filename_with_ampersand_round_trips(tmp_path):
    """A source filename containing ``&`` must produce a VRT whose
    ``<SourceFilename>`` element decodes back to the original on-disk
    path (no double-escape, no corruption)."""
    arr = np.zeros((4, 4), dtype=np.float32)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': np.linspace(1, 0, 4), 'x': np.linspace(0, 1, 4)}, attrs={'nodata': -9999.0})  # noqa: E501
    src = str(tmp_path / 'a&b.tif')
    to_geotiff(da, src)
    vrt_path = str(tmp_path / 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [src])
    with open(vrt_path, 'r') as fh:
        text = fh.read()
    assert '&amp;' in text
    assert '<a&b' not in text
    parsed = parse_vrt(text, vrt_dir=str(tmp_path))
    assert len(parsed.bands) == 1
    assert len(parsed.bands[0].sources) == 1
    assert os.path.basename(parsed.bands[0].sources[0].filename) == 'a&b.tif'


def test_written_vrt_is_well_formed_xml(xml_escape_sample_tif, tmp_path):
    """Sanity check: the bytes written by write_vrt always parse cleanly
    as XML, even when crs_wkt carries every XML predefined entity."""
    nasty = '< & > " \''
    vrt_path = str(tmp_path / 'wf.vrt')
    _write_vrt_internal(vrt_path, [xml_escape_sample_tif], crs_wkt=nasty)
    import xml.etree.ElementTree as ET
    with open(vrt_path, 'r') as fh:
        ET.fromstring(fh.read())


# ---------------------------------------------------------------------------
# XML size cap on eager read_vrt
# ---------------------------------------------------------------------------


def _xml_size_cap_write_source(td: str) -> str:
    src_path = os.path.join(td, 'tmp_1815_src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path, compression='none')
    return src_path


def _xml_size_cap_write_vrt(td: str, *, pad_bytes: int = 0) -> str:
    """Write a VRT, optionally padded with a large XML comment."""
    vrt_path = os.path.join(td, 'tmp_1815_mosaic.vrt')
    comment = ''
    if pad_bytes > 0:
        comment = '<!-- ' + 'x' * pad_bytes + ' -->\n'
    vrt_xml = '<VRTDataset rasterXSize="10" rasterYSize="10">\n' + comment + '  <VRTRasterBand dataType="Byte" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="1">tmp_1815_src.tif</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_small_vrt_parses_under_default_cap(tmp_path):
    """A normal-sized VRT parses successfully with the default cap."""
    td = str(tmp_path)
    _xml_size_cap_write_source(td)
    vrt_path = _xml_size_cap_write_vrt(td)
    arr, _ = _xml_size_cap_read_vrt_internal(vrt_path)
    assert arr.shape == (10, 10)


def test_oversized_vrt_raises_value_error(tmp_path, monkeypatch):
    """A VRT padded past the cap raises ValueError naming the cap and env var."""
    td = str(tmp_path)
    _xml_size_cap_write_source(td)
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', '1024')
    vrt_path = _xml_size_cap_write_vrt(td, pad_bytes=4096)
    with pytest.raises(ValueError) as exc_info:
        _xml_size_cap_read_vrt_internal(vrt_path)
    msg = str(exc_info.value)
    assert 'XRSPATIAL_VRT_MAX_XML_BYTES' in msg
    assert '1,024' in msg


def test_raising_cap_lets_padded_vrt_parse(tmp_path, monkeypatch):
    """Setting the env var higher allows a padded VRT to parse."""
    td = str(tmp_path)
    _xml_size_cap_write_source(td)
    vrt_path = _xml_size_cap_write_vrt(td, pad_bytes=4096)
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', str(1024 * 1024))
    arr, _ = _xml_size_cap_read_vrt_internal(vrt_path)
    assert arr.shape == (10, 10)


@pytest.mark.parametrize('bad_value', ['not_a_number', '0', '-1', '-1024'])
def test_invalid_cap_raises_value_error(tmp_path, monkeypatch, bad_value):
    """Non-numeric, zero, or negative cap values produce a clear error."""
    td = str(tmp_path)
    _xml_size_cap_write_source(td)
    vrt_path = _xml_size_cap_write_vrt(td)
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', bad_value)
    with pytest.raises(ValueError, match='XRSPATIAL_VRT_MAX_XML_BYTES'):
        _xml_size_cap_read_vrt_internal(vrt_path)


# ---------------------------------------------------------------------------
# XML size cap on chunked read_vrt
# ---------------------------------------------------------------------------


def _xml_size_cap_chunked_write_source(td: str) -> str:
    src_path = os.path.join(td, 'tmp_1831_src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path, compression='none')
    return src_path


def _xml_size_cap_chunked_write_vrt(td: str, *, pad_bytes: int = 0) -> str:
    """Write a VRT, optionally padded with a large XML comment."""
    vrt_path = os.path.join(td, 'tmp_1831_mosaic.vrt')
    comment = ''
    if pad_bytes > 0:
        comment = '<!-- ' + 'x' * pad_bytes + ' -->\n'
    vrt_xml = '<VRTDataset rasterXSize="10" rasterYSize="10">\n' + comment + '  <VRTRasterBand dataType="Byte" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="1">tmp_1831_src.tif</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_chunked_read_vrt_honors_xml_cap(tmp_path, monkeypatch):
    """``read_vrt(chunks=...)`` rejects oversized VRT XML."""
    td = str(tmp_path)
    _xml_size_cap_chunked_write_source(td)
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', '1024')
    vrt_path = _xml_size_cap_chunked_write_vrt(td, pad_bytes=4096)
    with pytest.raises(ValueError) as exc_info:
        read_vrt(vrt_path, chunks=10)
    msg = str(exc_info.value)
    assert 'XRSPATIAL_VRT_MAX_XML_BYTES' in msg
    assert '1,024' in msg


def test_chunked_read_vrt_under_default_cap(tmp_path):
    """A normal-sized VRT parses successfully under the default cap."""
    td = str(tmp_path)
    _xml_size_cap_chunked_write_source(td)
    vrt_path = _xml_size_cap_chunked_write_vrt(td)
    arr = read_vrt(vrt_path, chunks=10)
    assert arr.shape == (10, 10)
    assert arr.dtype == np.uint8


def test_chunked_read_vrt_raised_cap_allows_padded(tmp_path, monkeypatch):
    """Raising ``XRSPATIAL_VRT_MAX_XML_BYTES`` lets a padded VRT parse."""
    td = str(tmp_path)
    _xml_size_cap_chunked_write_source(td)
    vrt_path = _xml_size_cap_chunked_write_vrt(td, pad_bytes=4096)
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', str(1024 * 1024))
    arr = read_vrt(vrt_path, chunks=10)
    assert arr.shape == (10, 10)


# ---------------------------------------------------------------------------
# VRT metadata parity across backends
# ---------------------------------------------------------------------------


_WGS84_WKT = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'  # noqa: E501


_VRT_OMITTED_ATTR_KEYS = frozenset({'extra_tags', 'image_description', 'extra_samples', 'gdal_metadata', 'gdal_metadata_xml', 'x_resolution', 'y_resolution', 'resolution_unit', 'colormap'})  # noqa: E501


_REPRESENTATION_KEYS = frozenset({'crs_wkt'})


_BACKEND_LIFECYCLE_KEYS = frozenset({'nodata_pixels_present'})


def _metadata_parity_write_single_source_vrt(tiff_path: str, vrt_path: str, *, width: int, height: int, dtype_xml: str = 'Float32', nodata: float | int | None = None, geo_transform: str | None = '0.0, 1.0, 0.0, 0.0, 0.0, -1.0', srs: str | None = None) -> None:  # noqa: E501
    """Write a 1-band VRT pointing at ``tiff_path``.

    Same writer style as ``test_vrt_finalization_parity_2162`` so the
    two test modules share fixture geometry conventions.
    """
    nodata_xml = f'    <NoDataValue>{nodata}</NoDataValue>\n' if nodata is not None else ''
    srs_xml = f'  <SRS>{srs}</SRS>\n' if srs is not None else ''
    gt_xml = f'  <GeoTransform>{geo_transform}</GeoTransform>\n' if geo_transform is not None else ''  # noqa: E501
    vrt_xml = f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n{gt_xml}{srs_xml}  <VRTRasterBand dataType="{dtype_xml}" band="1">\n{nodata_xml}    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{tiff_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)


def _metadata_parity_build_full_georef_vrt(tmp_path: pathlib.Path) -> str:
    """4x4 float32 single-source VRT with full georef + nodata."""
    import xarray as xr
    tiff = str(tmp_path / 'tmp_2321_full_src.tif')
    vrt = str(tmp_path / 'tmp_2321_full.vrt')
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    da = xr.DataArray(data, coords={'y': np.array([200.0, 199.0, 198.0, 197.0]), 'x': np.array([100.0, 101.0, 102.0, 103.0])}, dims=('y', 'x'), attrs={'crs': 4326})  # noqa: E501
    to_geotiff(da, tiff)
    _metadata_parity_write_single_source_vrt(tiff, vrt, width=4, height=4, dtype_xml='Float32', nodata=-9999.0, geo_transform='100.0, 1.0, 0.0, 200.0, 0.0, -1.0', srs=_WGS84_WKT)  # noqa: E501
    return vrt


def _metadata_parity_build_transform_only_vrt(tmp_path: pathlib.Path) -> str:
    """4x4 single-source VRT with transform but no SRS (CRS absent)."""
    import xarray as xr
    tiff = str(tmp_path / 'tmp_2321_tonly_src.tif')
    vrt = str(tmp_path / 'tmp_2321_tonly.vrt')
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    da = xr.DataArray(data, coords={'y': np.array([200.0, 199.0, 198.0, 197.0]), 'x': np.array([100.0, 101.0, 102.0, 103.0])}, dims=('y', 'x'))  # noqa: E501
    to_geotiff(da, tiff)
    _metadata_parity_write_single_source_vrt(tiff, vrt, width=4, height=4, dtype_xml='Float32', geo_transform='100.0, 1.0, 0.0, 200.0, 0.0, -1.0', srs=None)  # noqa: E501
    return vrt


def _metadata_parity_build_integer_with_nodata_vrt(tmp_path: pathlib.Path) -> str:
    """4x4 uint16 single-source VRT with declared nodata sentinel.

    Used for ``masked_nodata`` parity: the integer-with-sentinel source
    must promote to float64 with NaN-masked sentinel pixels in every
    backend and stamp ``attrs['masked_nodata']=True``.
    """
    src_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 65535], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint16)  # noqa: E501
    tiff = str(tmp_path / 'tmp_2321_int_src.tif')
    vrt = str(tmp_path / 'tmp_2321_int.vrt')
    write(src_arr, tiff, nodata=65535, compression='none', tiled=False)
    _metadata_parity_write_single_source_vrt(tiff, vrt, width=4, height=4, dtype_xml='UInt16', nodata=65535, geo_transform='0.0, 1.0, 0.0, 0.0, 0.0, -1.0', srs=_WGS84_WKT)  # noqa: E501
    return vrt


def _metadata_parity_read_eager_numpy(vrt_path: str):
    """Eager numpy via the dispatcher (mirrors public surface)."""
    return open_geotiff(vrt_path)


def _metadata_parity_read_dask(vrt_path: str):
    """Dask via the dispatcher, then ``compute()`` for value parity."""
    lazy = open_geotiff(vrt_path, chunks=2)
    return lazy.compute()


def _metadata_parity_read_dask_chunks_2(vrt_path: str):
    """Dask via the dispatcher, lazy (no compute).

    Used for negative-tests that pin the build-time raise contract
    (e.g., ``test_mixed_nodata_vrt_fails_closed_by_default``). Named
    at module scope so pytest test ids render as
    ``[dask_chunks_2-_metadata_parity_read_dask_chunks_2]`` rather than the cryptic
    ``[dask_chunks_2-<lambda>]`` an inline lambda would produce.
    """
    return open_geotiff(vrt_path, chunks=2)


def _metadata_parity_read_gpu_eager(vrt_path: str):
    """GPU eager via ``read_vrt(gpu=True)``.

    ``open_geotiff(..., gpu=True)`` rejects ``.vrt`` sources up front
    (the dispatcher routes ``.vrt`` to ``read_vrt`` and ``read_vrt``
    owns the ``gpu`` kwarg, see ``_backends/vrt.py``). Use the direct
    entry point here so the GPU eager path is exercised.
    """
    return read_vrt(vrt_path, gpu=True)


_BACKENDS = [pytest.param('numpy', _metadata_parity_read_eager_numpy, id='numpy'), pytest.param('dask', _metadata_parity_read_dask, id='dask'), pytest.param('gpu', _metadata_parity_read_gpu_eager, id='gpu', marks=requires_gpu)]  # noqa: E501


def _metadata_parity_comparable_attrs(attrs: dict) -> dict:
    """Filter attrs down to the cross-backend comparable subset.

    Drops the documented VRT-omitted keys (which may differ if one
    backend stamps a TIFF-specific key while another does not) and the
    representation-only keys (``crs_wkt``).
    """
    return {k: v for k, v in attrs.items() if k not in _VRT_OMITTED_ATTR_KEYS and k not in _REPRESENTATION_KEYS and (k not in _BACKEND_LIFECYCLE_KEYS)}  # noqa: E501


def _metadata_parity_to_numpy(arr) -> np.ndarray:
    """Return a host-side numpy view of ``arr.values`` regardless of
    backend.

    CuPy DataArrays have a ``.values`` accessor that triggers an
    implicit host transfer in some xarray versions but not others; use
    the explicit ``.data.get()`` path for cupy buffers per CLAUDE.md.
    """
    data = arr.data
    if hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_full_georef_vrt_attrs_match_eager_numpy(tmp_path, _label, reader):
    """Each non-numpy backend's attrs must match the eager numpy baseline.

    The full-georef VRT carries CRS, transform, nodata, and an
    integer-source-promotes-to-float lifecycle. Every attr the contract
    promises (``transform``, ``crs``, ``nodata``, ``masked_nodata``,
    ``georef_status``, ``raster_type``) must compare equal across
    backends. ``crs_wkt`` is compared via the ``crs`` integer instead
    because the WKT text can re-emit under pyproj normalisation.

    Without this assertion a backend regression that drops one of
    these attrs but still returns correct pixels would slip through
    every existing pixel-only test.
    """
    vrt = _metadata_parity_build_full_georef_vrt(tmp_path)
    baseline = _metadata_parity_read_eager_numpy(vrt)
    candidate = reader(vrt)
    base_attrs = _metadata_parity_comparable_attrs(dict(baseline.attrs))
    cand_attrs = _metadata_parity_comparable_attrs(dict(candidate.attrs))
    base_keys = set(base_attrs)
    cand_keys = set(cand_attrs)
    assert base_keys == cand_keys, f'Attr-key drift between numpy and {_label}: numpy-only={base_keys - cand_keys}, {_label}-only={cand_keys - base_keys}'  # noqa: E501
    differing = [k for k in base_keys if base_attrs[k] != cand_attrs[k]]
    assert not differing, f'Attr value drift between numpy and {_label}: {[(k, base_attrs[k], cand_attrs[k]) for k in differing]}'  # noqa: E501
    for key in ('transform', 'crs', 'georef_status'):
        assert key in cand_attrs, f'{_label} backend missing required attr {key!r}'
    assert cand_attrs['georef_status'] == GEOREF_STATUS_FULL
    assert cand_attrs['crs'] == 4326
    assert len(cand_attrs['transform']) == 6


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_full_georef_vrt_pixels_match_eager_numpy(tmp_path, _label, reader):
    """Pixel-value parity for the full-georef VRT.

    Twin of the attrs test above: a regression that fixed attrs but
    broke pixels (or vice versa) must surface on at least one of the
    two. Asserting both side-by-side keeps the surface explicit.
    """
    vrt = _metadata_parity_build_full_georef_vrt(tmp_path)
    base = _metadata_parity_to_numpy(_metadata_parity_read_eager_numpy(vrt))
    cand = _metadata_parity_to_numpy(reader(vrt))
    assert base.shape == cand.shape, f'shape drift numpy vs {_label}: {base.shape} vs {cand.shape}'
    np.testing.assert_array_equal(base, cand)


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_full_georef_vrt_coords_match_eager_numpy(tmp_path, _label, reader):
    """Coord-array parity for the full-georef VRT.

    The transform attr alone does not guarantee correct coords: the
    half-pixel AREA_OR_POINT shift can drift between backends. Compare
    the actual coord arrays so a coord regression surfaces directly.
    """
    vrt = _metadata_parity_build_full_georef_vrt(tmp_path)
    base = _metadata_parity_read_eager_numpy(vrt)
    cand = reader(vrt)
    assert list(cand.dims) == list(base.dims), f'dim drift numpy vs {_label}: {base.dims} vs {cand.dims}'  # noqa: E501
    for axis in ('y', 'x'):
        np.testing.assert_array_equal(np.asarray(cand[axis].values), np.asarray(base[axis].values))


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_transform_only_vrt_attrs_match_eager_numpy(tmp_path, _label, reader):
    """Same parity sweep on a transform-only VRT (no CRS).

    ``georef_status`` must be ``transform_only`` on every backend and
    ``attrs['crs']`` must be absent on every backend. A regression
    that emits a stale CRS from a TIFF-tag fallback would show up here
    as a key-set diff.
    """
    vrt = _metadata_parity_build_transform_only_vrt(tmp_path)
    baseline = _metadata_parity_read_eager_numpy(vrt)
    candidate = reader(vrt)
    base_attrs = _metadata_parity_comparable_attrs(dict(baseline.attrs))
    cand_attrs = _metadata_parity_comparable_attrs(dict(candidate.attrs))
    assert set(base_attrs) == set(cand_attrs)
    assert base_attrs == cand_attrs
    assert cand_attrs['georef_status'] == GEOREF_STATUS_TRANSFORM_ONLY
    assert 'crs' not in cand_attrs


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_integer_nodata_vrt_attrs_match_eager_numpy(tmp_path, _label, reader):
    """``masked_nodata`` and ``nodata`` lifecycle parity on integer VRT.

    The integer-with-sentinel source must promote to float on every
    backend and stamp ``attrs['masked_nodata']=True`` plus
    ``attrs['nodata']=65535`` (the original sentinel). A backend that
    forgets to stamp ``masked_nodata`` would silently mislead callers
    who branch on the attr to decide whether NaN is real or a mask.
    """
    vrt = _metadata_parity_build_integer_with_nodata_vrt(tmp_path)
    baseline = _metadata_parity_read_eager_numpy(vrt)
    candidate = reader(vrt)
    base_attrs = _metadata_parity_comparable_attrs(dict(baseline.attrs))
    cand_attrs = _metadata_parity_comparable_attrs(dict(candidate.attrs))
    assert set(base_attrs) == set(cand_attrs)
    assert base_attrs == cand_attrs
    assert cand_attrs.get('masked_nodata') is True
    assert cand_attrs.get('nodata') == 65535


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_integer_nodata_vrt_pixels_match_eager_numpy(tmp_path, _label, reader):
    """Pixel parity for the integer-VRT case.

    Twin of the attrs test so a backend regression that masks but
    forgets the attr (or stamps the attr but masks the wrong cell)
    fails one assertion or the other, never both silently.
    """
    vrt = _metadata_parity_build_integer_with_nodata_vrt(tmp_path)
    base = _metadata_parity_to_numpy(_metadata_parity_read_eager_numpy(vrt))
    cand = _metadata_parity_to_numpy(reader(vrt))
    assert base.shape == cand.shape
    np.testing.assert_array_equal(np.isnan(base), np.isnan(cand))
    base_finite = base[~np.isnan(base)]
    cand_finite = cand[~np.isnan(cand)]
    np.testing.assert_array_equal(base_finite, cand_finite)


def _metadata_parity_write_mixed_crs_vrt(tmp_path: pathlib.Path) -> str:
    """Two single-band sources with disagreeing CRS at the VRT.

    The VRT XML carries one SRS (WGS84) but the second underlying TIFF
    carries a UTM CRS. The fail-closed contract rejects this up front
    rather than silently flattening to the VRT-declared SRS. See
    ``test_mixed_crs_vrt_does_not_silently_flatten`` for the
    consumer-side pin.
    """
    import xarray as xr
    src0 = tmp_path / 'tmp_2321_mix_crs_src0.tif'
    src1 = tmp_path / 'tmp_2321_mix_crs_src1.tif'
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    da0 = xr.DataArray(data, coords={'y': np.array([200.0, 199.0, 198.0, 197.0]), 'x': np.array([100.0, 101.0, 102.0, 103.0])}, dims=('y', 'x'), attrs={'crs': 4326})  # noqa: E501
    da1 = xr.DataArray(data, coords={'y': np.array([200.0, 199.0, 198.0, 197.0]), 'x': np.array([104.0, 105.0, 106.0, 107.0])}, dims=('y', 'x'), attrs={'crs': 32633})  # noqa: E501
    to_geotiff(da0, str(src0))
    to_geotiff(da1, str(src1))
    vrt_path = tmp_path / 'tmp_2321_mixed_crs.vrt'
    vrt_xml = f'<VRTDataset rasterXSize="8" rasterYSize="4">\n  <GeoTransform>100.0, 1.0, 0.0, 200.0, 0.0, -1.0</GeoTransform>\n  <SRS>{_WGS84_WKT}</SRS>\n  <VRTRasterBand dataType="Float32" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src0}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n    </SimpleSource>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src1}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="4" yOff="0" xSize="4" ySize="4"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


def test_mixed_crs_vrt_does_not_silently_flatten(tmp_path):
    """A mixed-CRS VRT must not return a mosaic that silently inherits
    one source's CRS while pixels came from a CRS-incompatible source.

    ``validate_parsed_vrt`` opens each source
    TIFF and raises ``VRTUnsupportedError`` when any source CRS
    disagrees with the VRT-declared ``<SRS>``. The error message names
    both the offending source and the disagreeing CRS so the caller
    can locate the bad source without re-parsing the VRT XML.
    """
    vrt = _metadata_parity_write_mixed_crs_vrt(tmp_path)
    with pytest.raises(VRTUnsupportedError):
        read_vrt(vrt)


def _metadata_parity_write_mixed_nodata_vrt(tmp_path: pathlib.Path) -> str:
    """Two-band uint16 VRT with disagreeing per-band ``<NoDataValue>``.

    Mirrors the fixture in ``test_vrt_multiband_int_nodata_1611``: the
    fail-closed default (band_nodata=None) must raise
    ``MixedBandMetadataError``. The opt-out
    ``band_nodata='first'`` is the explicit escape hatch.
    """
    b0_arr = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
    b1_arr = np.array([[7, 8], [9, 65000]], dtype=np.uint16)
    p0 = tmp_path / 'tmp_2321_mix_nodata_b0.tif'
    p1 = tmp_path / 'tmp_2321_mix_nodata_b1.tif'
    write(b0_arr, str(p0), nodata=65535, compression='none', tiled=False)
    write(b1_arr, str(p1), nodata=65000, compression='none', tiled=False)
    vrt_path = tmp_path / 'tmp_2321_mix_nodata.vrt'
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <NoDataValue>65535</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n  <VRTRasterBand dataType="UInt16" band="2">\n    <NoDataValue>65000</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


@pytest.mark.parametrize('reader_label, reader', [('eager_numpy', _metadata_parity_read_eager_numpy), ('dask_chunks_2', _metadata_parity_read_dask_chunks_2)])  # noqa: E501
def test_mixed_nodata_vrt_fails_closed_by_default(tmp_path, reader_label, reader):
    """Per-band disagreeing nodata raises ``MixedBandMetadataError``
    by default on every backend route.

    The dask path's check fires at graph-build time (the metadata
    sweep runs before dask materialises any chunk). The eager path
    raises during the dispatcher's metadata validation. Both must
    refuse rather than flattening to band 0's sentinel.
    """
    vrt = _metadata_parity_write_mixed_nodata_vrt(tmp_path)
    with pytest.raises(MixedBandMetadataError):
        result = reader(vrt)
        if hasattr(result, 'compute'):
            result.compute()


def test_mixed_nodata_vrt_opt_in_first_succeeds(tmp_path):
    """``band_nodata='first'`` is the documented opt-out for the
    mixed-nodata fail-closed check.

    Positive pin so a future change that breaks the escape hatch
    surfaces here. The opt-out flattens to band 0's sentinel, which
    is the legacy behaviour callers may explicitly want.
    """
    vrt = _metadata_parity_write_mixed_nodata_vrt(tmp_path)
    result = read_vrt(vrt, band_nodata='first')
    assert result.shape == (2, 2, 2)


def _metadata_parity_write_unsupported_resample_vrt(tmp_path: pathlib.Path) -> str:
    """VRT with ``<ResampleAlg>Bilinear`` and a size-changing DstRect.

    A 4x4 source projected into a 2x2 destination with Bilinear must
    raise because the implementation only honours nearest-neighbour
    resampling at the placement site.
    """
    src_arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src_path = tmp_path / 'tmp_2321_resample_src.tif'
    write(src_arr, str(src_path), compression='none', tiled=False)
    vrt_path = tmp_path / 'tmp_2321_unsupported_resample.vrt'
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <ComplexSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <ResampleAlg>Bilinear</ResampleAlg>\n    </ComplexSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


def test_unsupported_resample_alg_raises(tmp_path):
    """A non-nearest resampling algorithm with a size-changing DstRect
    must raise ``NotImplementedError`` rather than return
    silently-nearest-sampled pixels mislabelled as Bilinear.

    The ``match=`` clause pins the algorithm name
    so an unrelated ``NotImplementedError`` from some other VRT code
    path cannot keep the test green. See ``_vrt.py`` for the existing
    raise that names the field. The centralised validator raises
    ``VRTUnsupportedError`` for the same case; the assertion below
    accepts either type.
    """
    vrt = _metadata_parity_write_unsupported_resample_vrt(tmp_path)
    with pytest.raises((NotImplementedError, VRTUnsupportedError), match='Bilinear'):
        read_vrt(vrt)


def _metadata_parity_write_bad_srcrect_vrt(tmp_path: pathlib.Path, *, x_size: int = -50) -> str:
    """VRT with a negative-size ``<SrcRect>``.

    The validator must reject this up front rather than
    swallow it in the missing-source ``try/except``.
    """
    src_arr = np.zeros((10, 10), dtype=np.uint8)
    src_path = tmp_path / 'tmp_2321_bad_srcrect_src.tif'
    to_geotiff(src_arr, str(src_path), compression='none')
    vrt_path = tmp_path / 'tmp_2321_bad_srcrect.vrt'
    vrt_xml = f'<VRTDataset rasterXSize="100" rasterYSize="100">\n  <VRTRasterBand dataType="Byte" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{x_size}" ySize="10"/>\n      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


def test_negative_srcrect_size_rejected(tmp_path):
    """Malformed ``SrcRect`` rejected with a ``ValueError`` (legacy
    path) or ``VRTUnsupportedError`` (centralised validator) that names
    the offending field.
    """
    vrt = _metadata_parity_write_bad_srcrect_vrt(tmp_path, x_size=-50)
    with pytest.raises((ValueError, VRTUnsupportedError), match='SrcRect.*negative'):
        read_vrt(vrt)


def _metadata_parity_write_bad_dstrect_vrt(tmp_path: pathlib.Path, *, x_size: int = -10) -> str:
    """VRT with a negative-size ``<DstRect>`` for the negative test.

    Mirrors the existing DstRect rejection; the regression
    coverage today targets oversized DstRects, this test pins the
    sister case for negative dimensions.
    """
    src_arr = np.zeros((10, 10), dtype=np.uint8)
    src_path = tmp_path / 'tmp_2321_bad_dstrect_src.tif'
    to_geotiff(src_arr, str(src_path), compression='none')
    vrt_path = tmp_path / 'tmp_2321_bad_dstrect.vrt'
    vrt_xml = f'<VRTDataset rasterXSize="100" rasterYSize="100">\n  <VRTRasterBand dataType="Byte" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n      <DstRect xOff="0" yOff="0" xSize="{x_size}" ySize="10"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


def test_negative_dstrect_size_rejected(tmp_path):
    """Malformed ``DstRect`` must not survive into the read path.

    Accept ``ValueError`` (today's posture; the SimpleSource DstRect
    validator raises ``VRT SimpleSource DstRect has negative size
    (...)`` before any pixel work begins). The ``match=`` clause pins
    the field name and the rejection reason so an unrelated
    ``ValueError`` from some other VRT code path cannot silently keep
    the test green. The centralised validator
    raises ``VRTUnsupportedError`` for the same case; both are accepted.
    """
    vrt = _metadata_parity_write_bad_dstrect_vrt(tmp_path, x_size=-10)
    with pytest.raises((ValueError, VRTUnsupportedError), match='DstRect.*negative'):
        read_vrt(vrt)


def _metadata_parity_write_missing_source_vrt(tmp_path: pathlib.Path, *, name: str = 'tmp_2321_missing.vrt') -> str:  # noqa: E501
    """VRT pointing at a single source path that does not exist.

    The dispatcher's static missing-source scan raises at
    construction time for both eager and dask routes when
    ``missing_sources='raise'`` is in effect.
    """
    vrt_path = tmp_path / name
    missing = tmp_path / 'tmp_2321_missing_src.tif'
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="Byte" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{missing}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    vrt_path.write_text(vrt_xml)
    assert not os.path.exists(str(missing)), 'fixture leak: missing-source path exists on disk'
    return str(vrt_path)


def test_missing_sources_raise_eager(tmp_path):
    """``missing_sources='raise'`` (the public default)
    must abort the read up front on the eager path."""
    vrt = _metadata_parity_write_missing_source_vrt(tmp_path, name='tmp_2321_miss_eager.vrt')
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        read_vrt(vrt)


def test_missing_sources_raise_dask(tmp_path):
    """``missing_sources='raise'`` (default) on the dask path raises
    at graph-build time, not at ``.compute()``.

    Pin both the build-time raise and the value path so a regression
    that defers the check to compute surfaces here.
    """
    vrt = _metadata_parity_write_missing_source_vrt(tmp_path, name='tmp_2321_miss_dask.vrt')
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        lazy = open_geotiff(vrt, chunks=2)
        lazy.compute()


def test_missing_sources_warn_records_holes(tmp_path):
    """``missing_sources='warn'`` is the documented escape hatch.

    The lenient path must emit ``GeoTIFFFallbackWarning`` and populate
    ``attrs['vrt_holes']`` so callers branching on the attr can detect
    a partial mosaic. This is the documented contract;
    the test pins it via the public ``read_vrt`` entry point so a
    regression in the warn-policy attr emission surfaces.

    The public API exposes ``'warn'`` as the lenient option (``'skip'``
    is used internally inside ``_vrt.read_vrt``). Use the documented
    public value here so the test pins the user-facing contract.
    """
    vrt = _metadata_parity_write_missing_source_vrt(tmp_path, name='tmp_2321_miss_warn.vrt')
    with pytest.warns(GeoTIFFFallbackWarning, match='could not be read'):
        result = read_vrt(vrt, missing_sources='warn')
    assert 'vrt_holes' in result.attrs, "missing_sources='warn' did not stamp attrs['vrt_holes']"
    holes = result.attrs['vrt_holes']
    assert len(holes) == 1
    assert isinstance(holes[0], dict), f'vrt_holes entry type drifted: {type(holes[0]).__name__}; #1734 documents a dict shape'  # noqa: E501
    hole_source = holes[0]['source']
    assert 'tmp_2321_missing_src.tif' in hole_source, f'hole source path drifted: {hole_source!r}'
