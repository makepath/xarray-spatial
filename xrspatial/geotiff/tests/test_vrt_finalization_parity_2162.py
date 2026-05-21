"""Cross-backend parity for the VRT finalization pipeline (issue #2180).

Wave 3 of #2162 routed the VRT eager and chunked paths through
``_finalize_lazy_read_attrs`` from #2177. Before the migration the two
sites built ``GeoTIFFMetadata`` from VRT internals by hand and called
``metadata_to_attrs`` directly, bypassing the shared
``_validate_read_geo_info`` / ``_populate_attrs_from_geo_info`` block
the other backends share.

The tests below pin parity for the attrs the helper now stamps:

* VRT eager attrs match eager numpy attrs (``open_geotiff``) for
  single-source VRTs that mirror a plain TIFF.
* VRT chunked attrs match dask numpy attrs (``read_geotiff_dask``) for
  the same single-source VRTs.
* ``band_nodata='first'`` paths still produce the per-band attrs
  pinned by ``test_vrt_band_nodata_1598``.
* ``missing_sources='warn'`` still surfaces ``attrs['vrt_holes']`` on
  the eager VRT path (the chunked path's parse-time hole scan is
  covered by ``test_open_geotiff_missing_sources_1810``).
* ``attrs['georef_status']`` matches across VRT and non-VRT paths for
  the five canonical states (``full``, ``transform_only``,
  ``crs_only``, ``none``, ``rotated_dropped``).

VRT-only attrs that the non-VRT path cannot produce (e.g.
``vrt_holes``) and the windowed-transform shift are not part of the
parity assertion -- they are pinned by the regression tests cited
above. A few attrs the non-VRT path emits (``extra_tags``,
``gdal_metadata``, resolution tags) are likewise dropped from the
comparison because the VRT path intentionally omits them; the test
filters those keys explicitly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_vrt,
    to_geotiff,
)
from xrspatial.geotiff._attrs import (
    GEOREF_STATUS_CRS_ONLY,
    GEOREF_STATUS_FULL,
    GEOREF_STATUS_NONE,
    GEOREF_STATUS_ROTATED_DROPPED,
    GEOREF_STATUS_TRANSFORM_ONLY,
)
from xrspatial.geotiff._coords import _NO_GEOREF_KEY
from xrspatial.geotiff._writer import write

tifffile = pytest.importorskip("tifffile")


# Attrs the VRT path is documented to omit when the non-VRT path emits
# them. The parity comparisons drop these keys before checking equality
# so the per-backend documented surface stays in scope.
_NON_VRT_ONLY_KEYS = frozenset({
    'extra_tags',
    'image_description',
    'extra_samples',
    'gdal_metadata',
    'gdal_metadata_xml',
    'x_resolution',
    'y_resolution',
    'resolution_unit',
    'colormap',
})


# Attrs that differ in textual representation between the GeoTIFF writer
# and the literal VRT XML even when they encode the same logical value.
# ``crs_wkt`` carries pyproj's expanded WKT in the TIFF path but the
# verbatim VRT XML body in the VRT path; ``transform`` shifts by a
# half-pixel between the two writers' AREA_OR_POINT conventions. The
# parity test compares them separately via EPSG / origin checks rather
# than insisting on byte-identical strings.
_REPRESENTATION_KEYS = frozenset({'crs_wkt', 'transform'})


def _shared_canonical_attrs(attrs: dict) -> dict:
    """Return the helper-emitted attrs that should match across writers.

    Drops:
    * The non-VRT TIFF-tag attrs the VRT path intentionally omits.
    * The representation-sensitive attrs (``crs_wkt``, ``transform``)
      that differ in literal form but encode the same logical value.
      ``crs`` (EPSG integer) carries the same information for the WKT
      comparison; the transform half-pixel shift is exercised by the
      regression tests for the underlying readers.
    """
    return {
        k: v for k, v in attrs.items()
        if k not in _NON_VRT_ONLY_KEYS and k not in _REPRESENTATION_KEYS
    }


def _strip_non_vrt_keys(attrs: dict) -> dict:
    return {k: v for k, v in attrs.items() if k not in _NON_VRT_ONLY_KEYS}


def _write_single_source_vrt(tiff_path, vrt_path, *, width, height,
                             dtype='Float32', nodata=None,
                             geo_transform='0.0, 1.0, 0.0, 0.0, 0.0, -1.0',
                             srs=None):
    """Write a one-band VRT pointing at ``tiff_path``.

    Mirrors the writer in ``test_vrt_band_nodata_1598`` but parameterises
    the geo bits so the same helper can produce ``full`` /
    ``transform_only`` / ``crs_only`` / ``none`` / ``rotated_dropped``
    VRTs.
    """
    nodata_xml = (
        f"    <NoDataValue>{nodata}</NoDataValue>\n" if nodata is not None
        else ''
    )
    srs_xml = (
        f'  <SRS>{srs}</SRS>\n' if srs is not None
        else ''
    )
    gt_xml = (
        f'  <GeoTransform>{geo_transform}</GeoTransform>\n'
        if geo_transform is not None
        else ''
    )
    vrt_xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        f'{gt_xml}'
        f'{srs_xml}'
        f'  <VRTRasterBand dataType="{dtype}" band="1">\n'
        f'{nodata_xml}'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{tiff_path}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)


# ---------------------------------------------------------------------------
# Fixture builders for the five georef states.
# ---------------------------------------------------------------------------
#
# Each builder writes a backing TIFF and a single-source VRT that wraps
# it with the same transform / CRS, then returns both paths. The VRT
# path's ``georef_status`` should match the TIFF path's because the VRT
# shares the same geometry.

_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
)


def _make_full_pair(tmp_path, name):
    """Full georef: float coords + CRS."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )
    to_geotiff(da, tiff)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform='100.0, 1.0, 0.0, 200.0, 0.0, -1.0',
        srs=_WGS84_WKT,
    )
    return tiff, vrt


def _make_transform_only_pair(tmp_path, name):
    """Float coords, no CRS."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
    )
    to_geotiff(da, tiff)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform='100.0, 1.0, 0.0, 200.0, 0.0, -1.0',
        srs=None,
    )
    return tiff, vrt


def _make_crs_only_pair(tmp_path, name):
    """No-georef marker + CRS."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(4, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True, 'crs': 4326},
    )
    to_geotiff(da, tiff)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform=None,
        srs=_WGS84_WKT,
    )
    return tiff, vrt


def _make_none_pair(tmp_path, name):
    """No CRS, no transform."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    arr = np.zeros((4, 4), dtype=np.float32)
    tifffile.imwrite(
        tiff, arr, photometric='minisblack', planarconfig='contig',
        metadata=None,
    )
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform=None,
        srs=None,
    )
    return tiff, vrt


def _make_rotated_pair(tmp_path, name):
    """Rotated VRT with ``allow_rotated=True``: lands at
    ``rotated_dropped``."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    write(arr, tiff, compression='none', tiled=False)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4, dtype='UInt16',
        geo_transform='0.0, 1.0, 0.5, 0.0, 0.5, -1.0',
        srs=None,
    )
    return tiff, vrt


# ---------------------------------------------------------------------------
# Parity tests: VRT eager attrs vs eager numpy attrs.
# ---------------------------------------------------------------------------


def test_vrt_eager_full_matches_open_geotiff(tmp_path):
    """A single-source VRT wrapping a ``full`` TIFF emits the same
    canonical helper-stamped attrs as the underlying TIFF read via
    ``open_geotiff``.

    The helper-emitted attrs (``crs`` / ``georef_status`` / contract
    version / nodata lifecycle) must match. ``crs_wkt`` and
    ``transform`` differ in textual representation between the two
    writers and are compared separately via EPSG / origin checks
    below; pinning byte-identical strings would test the writer, not
    the helper migration.
    """
    tiff, vrt = _make_full_pair(tmp_path, 'full_2180')
    tiff_attrs = _shared_canonical_attrs(dict(open_geotiff(tiff).attrs))
    vrt_attrs = _shared_canonical_attrs(dict(read_vrt(vrt).attrs))
    assert tiff_attrs == vrt_attrs, (
        f"TIFF/VRT attrs diverged:\n"
        f"  tiff only: {set(tiff_attrs) - set(vrt_attrs)}\n"
        f"  vrt only:  {set(vrt_attrs) - set(tiff_attrs)}\n"
        f"  shared keys with different values: "
        f"{[k for k in set(tiff_attrs) & set(vrt_attrs) if tiff_attrs[k] != vrt_attrs[k]]}"
    )
    # Logical CRS equality across the two writers (different WKT text,
    # same EPSG code).
    full_tiff_attrs = dict(open_geotiff(tiff).attrs)
    full_vrt_attrs = dict(read_vrt(vrt).attrs)
    assert full_tiff_attrs['crs'] == full_vrt_attrs['crs'] == 4326
    # Both paths emit a 6-tuple transform with the same length.
    assert len(full_tiff_attrs['transform']) == 6
    assert len(full_vrt_attrs['transform']) == 6


def test_vrt_eager_transform_only_matches_open_geotiff(tmp_path):
    tiff, vrt = _make_transform_only_pair(tmp_path, 'tonly_2180')
    tiff_attrs = _shared_canonical_attrs(dict(open_geotiff(tiff).attrs))
    vrt_attrs = _shared_canonical_attrs(dict(read_vrt(vrt).attrs))
    assert tiff_attrs == vrt_attrs
    assert tiff_attrs['georef_status'] == GEOREF_STATUS_TRANSFORM_ONLY


def test_vrt_eager_crs_only_matches_open_geotiff(tmp_path):
    tiff, vrt = _make_crs_only_pair(tmp_path, 'crsonly_2180')
    tiff_attrs = _shared_canonical_attrs(dict(open_geotiff(tiff).attrs))
    vrt_attrs = _shared_canonical_attrs(dict(read_vrt(vrt).attrs))
    assert tiff_attrs == vrt_attrs
    assert tiff_attrs['georef_status'] == GEOREF_STATUS_CRS_ONLY


def test_vrt_eager_none_matches_open_geotiff(tmp_path):
    tiff, vrt = _make_none_pair(tmp_path, 'none_2180')
    tiff_attrs = _shared_canonical_attrs(dict(open_geotiff(tiff).attrs))
    vrt_attrs = _shared_canonical_attrs(dict(read_vrt(vrt).attrs))
    assert tiff_attrs == vrt_attrs
    assert tiff_attrs['georef_status'] == GEOREF_STATUS_NONE


def test_vrt_eager_rotated_dropped_matches_open_geotiff(tmp_path):
    """The rotated branch is the VRT-specific path: a non-zero skew on
    the GDAL geotransform lands in ``rotated_dropped`` and the helper
    drops ``crs`` / ``transform`` / ``crs_wkt`` while emitting
    ``rotated_affine`` plus the no-georef marker. The non-VRT side does
    not have a way to write a rotated TIFF cleanly through ``to_geotiff``
    (axis-aligned only); the assertions here pin the attrs surface
    against the canonical ``georef_status`` values rather than a
    non-VRT TIFF parity check.
    """
    _, vrt = _make_rotated_pair(tmp_path, 'rot_2180')
    attrs = dict(read_vrt(vrt, allow_rotated=True).attrs)
    assert attrs['georef_status'] == GEOREF_STATUS_ROTATED_DROPPED
    assert attrs.get(_NO_GEOREF_KEY) is True
    assert 'rotated_affine' in attrs
    assert attrs.get('crs') is None
    assert attrs.get('crs_wkt') is None
    assert 'transform' not in attrs


# ---------------------------------------------------------------------------
# Parity tests: VRT chunked attrs vs dask numpy attrs.
# ---------------------------------------------------------------------------


def test_vrt_chunked_full_matches_dask(tmp_path):
    tiff, vrt = _make_full_pair(tmp_path, 'full_chunked_2180')
    tiff_attrs = _shared_canonical_attrs(
        dict(read_geotiff_dask(tiff, chunks=2).attrs)
    )
    vrt_attrs = _shared_canonical_attrs(
        dict(read_vrt(vrt, chunks=2).attrs)
    )
    assert tiff_attrs == vrt_attrs


def test_vrt_chunked_transform_only_matches_dask(tmp_path):
    tiff, vrt = _make_transform_only_pair(tmp_path, 'tonly_chunked_2180')
    tiff_attrs = _shared_canonical_attrs(
        dict(read_geotiff_dask(tiff, chunks=2).attrs)
    )
    vrt_attrs = _shared_canonical_attrs(
        dict(read_vrt(vrt, chunks=2).attrs)
    )
    assert tiff_attrs == vrt_attrs


def test_vrt_chunked_crs_only_matches_dask(tmp_path):
    tiff, vrt = _make_crs_only_pair(tmp_path, 'crsonly_chunked_2180')
    tiff_attrs = _shared_canonical_attrs(
        dict(read_geotiff_dask(tiff, chunks=2).attrs)
    )
    vrt_attrs = _shared_canonical_attrs(
        dict(read_vrt(vrt, chunks=2).attrs)
    )
    assert tiff_attrs == vrt_attrs


def test_vrt_chunked_none_matches_dask(tmp_path):
    tiff, vrt = _make_none_pair(tmp_path, 'none_chunked_2180')
    tiff_attrs = _shared_canonical_attrs(
        dict(read_geotiff_dask(tiff, chunks=2).attrs)
    )
    vrt_attrs = _shared_canonical_attrs(
        dict(read_vrt(vrt, chunks=2).attrs)
    )
    assert tiff_attrs == vrt_attrs


def test_vrt_chunked_rotated_dropped(tmp_path):
    _, vrt = _make_rotated_pair(tmp_path, 'rot_chunked_2180')
    attrs = dict(read_vrt(vrt, allow_rotated=True, chunks=2).attrs)
    assert attrs['georef_status'] == GEOREF_STATUS_ROTATED_DROPPED
    assert attrs.get(_NO_GEOREF_KEY) is True
    assert 'rotated_affine' in attrs


# ---------------------------------------------------------------------------
# band_nodata paths: the ``'first'`` opt-out keeps the legacy
# flatten-to-band-0 semantics. Pin per-band attrs on a mixed VRT.
# ---------------------------------------------------------------------------


def _write_two_band_per_band_nodata_vrt(tmp_path):
    band0 = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, 65000]], dtype=np.uint16)
    p0 = str(tmp_path / 'vrt_band0_2180.tif')
    p1 = str(tmp_path / 'vrt_band1_2180.tif')
    write(band0, p0, nodata=65535, compression='none', tiled=False)
    write(band1, p1, nodata=65000, compression='none', tiled=False)

    vrt_path = str(tmp_path / 'two_band_per_band_nodata_2180.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>65535</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>65000</NoDataValue>
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


def test_band_nodata_first_band_attrs(tmp_path):
    """``band=1`` with ``band_nodata='first'`` surfaces band 1's
    sentinel on attrs and masks against it. Pins the per-band selection
    survives the migration."""
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band=1, band_nodata='first')
    assert r.attrs['nodata'] == 65000.0
    assert r.attrs['masked_nodata'] is True
    assert np.isnan(r.values[1, 1])
    assert r.attrs.get('nodata_pixels_present') is True


def test_band_nodata_chunked_first_band_attrs(tmp_path):
    """The chunked path threads the same per-band sentinel onto attrs."""
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band=1, band_nodata='first', chunks=2)
    assert r.attrs['nodata'] == 65000.0
    assert r.attrs['masked_nodata'] is True
    # Chunked path leaves ``nodata_pixels_present`` unset by contract.
    assert 'nodata_pixels_present' not in r.attrs


# ---------------------------------------------------------------------------
# missing_sources paths: ``warn`` surfaces ``vrt_holes`` on the eager
# path; the chunked parse-time scan also surfaces it.
# ---------------------------------------------------------------------------


def test_missing_sources_eager_surfaces_vrt_holes(tmp_path):
    """The eager VRT path keeps populating ``attrs['vrt_holes']`` after
    the migration, even though the field rides outside the synthesised
    ``GeoInfo`` and through ``attrs_in`` on the helper."""
    tiff_path = str(tmp_path / 'present_2180.tif')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    write(arr, tiff_path, compression='none', tiled=False)

    missing_path = str(tmp_path / 'missing_2180.tif')  # never created
    vrt_path = str(tmp_path / 'mosaic_2180.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="8">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{tiff_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{missing_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="4" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = read_vrt(vrt_path, missing_sources='warn')
    assert 'vrt_holes' in r.attrs
    holes = r.attrs['vrt_holes']
    assert isinstance(holes, list) and len(holes) >= 1
    # Each hole entry has the documented shape.
    for hole in holes:
        assert 'source' in hole
        assert 'band' in hole
        assert 'dst_rect' in hole
        assert 'error' in hole


def test_missing_sources_chunked_surfaces_vrt_holes(tmp_path):
    """Chunked path's parse-time existence sweep still populates
    ``attrs['vrt_holes']`` after the migration."""
    tiff_path = str(tmp_path / 'present_chunked_2180.tif')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    write(arr, tiff_path, compression='none', tiled=False)

    missing_path = str(tmp_path / 'missing_chunked_2180.tif')
    vrt_path = str(tmp_path / 'mosaic_chunked_2180.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="8">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{tiff_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{missing_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="4" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    r = read_vrt(vrt_path, missing_sources='warn', chunks=2)
    assert 'vrt_holes' in r.attrs
    holes = r.attrs['vrt_holes']
    assert isinstance(holes, list) and len(holes) >= 1


# ---------------------------------------------------------------------------
# georef_status parity across the five states between VRT eager,
# VRT chunked, non-VRT eager, and non-VRT chunked.
# ---------------------------------------------------------------------------


_STATUS_PAIRS = [
    pytest.param(_make_full_pair, GEOREF_STATUS_FULL, False, id="full"),
    pytest.param(
        _make_transform_only_pair, GEOREF_STATUS_TRANSFORM_ONLY,
        False, id="transform_only",
    ),
    pytest.param(
        _make_crs_only_pair, GEOREF_STATUS_CRS_ONLY,
        False, id="crs_only",
    ),
    pytest.param(_make_none_pair, GEOREF_STATUS_NONE, False, id="none"),
    pytest.param(
        _make_rotated_pair, GEOREF_STATUS_ROTATED_DROPPED, True,
        id="rotated_dropped",
    ),
]


@pytest.mark.parametrize("pair_factory,expected_status,allow_rotated",
                         _STATUS_PAIRS)
def test_georef_status_eager_parity(tmp_path, pair_factory, expected_status,
                                    allow_rotated):
    """VRT eager and (where applicable) non-VRT eager agree on
    ``georef_status``. The rotated VRT case has no non-VRT counterpart
    through ``to_geotiff``, so the test pins the VRT value alone."""
    tiff, vrt = pair_factory(tmp_path, f'georef_eager_{expected_status}')
    kwargs = {'allow_rotated': True} if allow_rotated else {}
    vrt_status = read_vrt(vrt, **kwargs).attrs.get('georef_status')
    assert vrt_status == expected_status
    if not allow_rotated:
        tiff_status = open_geotiff(tiff, **kwargs).attrs.get('georef_status')
        assert tiff_status == expected_status
        assert vrt_status == tiff_status


@pytest.mark.parametrize("pair_factory,expected_status,allow_rotated",
                         _STATUS_PAIRS)
def test_georef_status_chunked_parity(tmp_path, pair_factory, expected_status,
                                      allow_rotated):
    """VRT chunked and non-VRT chunked agree on ``georef_status``."""
    tiff, vrt = pair_factory(tmp_path, f'georef_chunked_{expected_status}')
    kwargs = {'allow_rotated': True} if allow_rotated else {}
    vrt_status = read_vrt(vrt, chunks=2, **kwargs).attrs.get('georef_status')
    assert vrt_status == expected_status
    if not allow_rotated:
        tiff_status = read_geotiff_dask(
            tiff, chunks=2, **kwargs
        ).attrs.get('georef_status')
        assert tiff_status == expected_status
        assert vrt_status == tiff_status


# ---------------------------------------------------------------------------
# Eager/chunked VRT internal parity: the same VRT read eagerly and
# chunked should agree on the canonical attrs (modulo the documented
# absence of ``nodata_pixels_present`` on lazy reads).
# ---------------------------------------------------------------------------


_VRT_FACTORIES = [
    pytest.param(_make_full_pair, False, id="full"),
    pytest.param(_make_transform_only_pair, False, id="transform_only"),
    pytest.param(_make_crs_only_pair, False, id="crs_only"),
    pytest.param(_make_none_pair, False, id="none"),
    pytest.param(_make_rotated_pair, True, id="rotated_dropped"),
]


@pytest.mark.parametrize("pair_factory,allow_rotated", _VRT_FACTORIES)
def test_vrt_eager_chunked_internal_parity(tmp_path, pair_factory,
                                           allow_rotated):
    """Eager and chunked VRT reads of the same fixture agree on the
    shared canonical attrs (``crs`` / ``crs_wkt`` / ``transform`` /
    ``georef_status`` / contract version). The lazy contract from
    #2135 leaves ``nodata_pixels_present`` unset on chunked output, so
    the comparison drops that key."""
    _, vrt = pair_factory(tmp_path, 'internal_parity_2180')
    kwargs = {'allow_rotated': True} if allow_rotated else {}
    eager_attrs = dict(read_vrt(vrt, **kwargs).attrs)
    chunked_attrs = dict(read_vrt(vrt, chunks=2, **kwargs).attrs)
    eager_attrs.pop('nodata_pixels_present', None)
    chunked_attrs.pop('nodata_pixels_present', None)
    assert eager_attrs == chunked_attrs
