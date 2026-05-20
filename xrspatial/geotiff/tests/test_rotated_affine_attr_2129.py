"""``allow_rotated=True`` surfaces ``attrs['rotated_affine']`` (#2129).

Issue #2126 fixed the CRS-drop side of the ``allow_rotated=True``
contract, but the rotated 6-tuple itself was still unreachable from
public code: it lived on ``geo_info.transform.rotated_affine`` while
``geo_info`` was not on the returned DataArray. This file pins the
follow-up that emits ``attrs['rotated_affine']`` so callers can read
the rotated mapping without reaching into reader internals.

The attr:

* appears only when the source carried a rotated ``ModelTransformationTag``
  AND the caller passed ``allow_rotated=True``;
* is a rasterio-style 6-tuple ``(a, b, c, d, e, f)`` matching the
  ordering already documented on ``GeoTransform.rotated_affine``;
* is dropped on the way back through ``to_geotiff`` until the writer
  grows ``ModelTransformationTag`` support (#2115 follow-up). The
  drop is verified by feeding a synthesised attrs dict through
  ``attrs_to_metadata`` and asserting the parsed record does not
  carry the rotated 6-tuple forward.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._attrs import (
    _ATTRS_CONTRACT_VERSION,
    _populate_attrs_from_geo_info,
    attrs_to_metadata,
    geo_info_to_metadata,
)
from xrspatial.geotiff._geotags import GeoInfo, GeoTransform


_ROTATED_TUPLE = (8.66, -5.0, 100.0, 5.0, 8.66, 200.0)


def _rotated_geo_info(*, with_crs: bool = True) -> GeoInfo:
    """Build a GeoInfo that mimics the ``allow_rotated=True`` parser output."""
    t = GeoTransform(
        origin_x=0.0, origin_y=0.0,
        pixel_width=1.0, pixel_height=-1.0,
        rotated_affine=_ROTATED_TUPLE,
    )
    return GeoInfo(
        transform=t,
        has_georef=False,
        crs_epsg=4326 if with_crs else None,
        crs_wkt='GEOGCS["WGS 84"]' if with_crs else None,
    )


# ---------------------------------------------------------------------------
# Unit tests on the helpers (no file I/O)
# ---------------------------------------------------------------------------


def test_rotated_optin_emits_rotated_affine_tuple():
    gi = _rotated_geo_info()
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)

    assert attrs.get('rotated_affine') == _ROTATED_TUPLE
    # Sanity: rotated path still drops crs / transform (existing #2126
    # contract). Re-checked here so a regression to either branch
    # surfaces in the same test file as the new attr.
    assert 'crs' not in attrs
    assert 'transform' not in attrs


def test_rotated_optin_emits_rotated_affine_without_crs():
    gi = _rotated_geo_info(with_crs=False)
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)

    assert attrs.get('rotated_affine') == _ROTATED_TUPLE


def test_plain_no_georef_omits_rotated_affine():
    # Plain no-georef file: no transform tag, no rotated matrix. The new
    # attr must NOT appear on this path -- it is reserved for the
    # ``allow_rotated=True`` opt-in.
    gi = GeoInfo(transform=GeoTransform(), has_georef=False, crs_epsg=4326)
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)

    assert 'rotated_affine' not in attrs


def test_axis_aligned_read_omits_rotated_affine():
    gi = GeoInfo(
        transform=GeoTransform(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
        ),
        has_georef=True,
        crs_epsg=4326,
    )
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)

    assert 'rotated_affine' not in attrs


def test_rotated_affine_is_tuple_not_list():
    # ``GeoTransform.rotated_affine`` is documented as a tuple; the
    # public attr should respect the same type so downstream code can
    # rely on ``isinstance(attrs['rotated_affine'], tuple)``. The
    # parser tuple-casts the field even if a future change stores it
    # as a list or numpy sequence.
    gi = _rotated_geo_info()
    # Replace with a list to simulate a parser change.
    gi.transform.rotated_affine = list(_ROTATED_TUPLE)
    md = geo_info_to_metadata(gi)

    assert isinstance(md.rotated_affine, tuple)
    assert md.rotated_affine == _ROTATED_TUPLE


# ---------------------------------------------------------------------------
# Round-trip contract: the writer parser must drop ``rotated_affine``.
# ---------------------------------------------------------------------------


def test_attrs_to_metadata_drops_rotated_affine():
    """The write-side boundary parser intentionally does not carry the
    rotated 6-tuple forward; the writer would otherwise need a
    ``ModelTransformationTag`` emit path (#2115). Keeping it off the
    record ensures ``to_geotiff`` keeps writing a plain no-georef file
    until that follow-up lands."""
    attrs = {
        'rotated_affine': _ROTATED_TUPLE,
        '_xrspatial_no_georef': True,
        '_xrspatial_geotiff_contract': _ATTRS_CONTRACT_VERSION,
    }
    md = attrs_to_metadata(attrs)

    assert md.rotated_affine is None
    assert md.has_georef is False
    assert md.transform is None


# ---------------------------------------------------------------------------
# End-to-end via open_geotiff against a synthesised rotated GeoTIFF.
# ---------------------------------------------------------------------------


def _write_rotated_tiff(path, arr, *, epsg=None):
    """Write a rotated GeoTIFF with an optional GeoKey-encoded CRS.

    The 4x4 ``ModelTransformationTag`` matrix encodes a 30-degree
    rotation with 10-unit pixel spacing. Mirrors the helper in
    ``test_allow_rotated_crs_drop_2126.py`` so the two suites share the
    same synthetic file shape.
    """
    tifffile = pytest.importorskip("tifffile")
    cos30 = 0.8660254037844387
    sin30 = 0.5
    m = (
        10.0 * cos30, -10.0 * sin30, 0.0, 100.0,
        10.0 * sin30,  10.0 * cos30, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extratags = [
        # ModelTransformationTag (34264) -- DOUBLE, count=16.
        (34264, 12, 16, m, False),
    ]
    if epsg is not None:
        # GeoKeyDirectory: header (4 shorts) + one key for GeographicTypeGeoKey.
        geo_key_directory = (
            1, 1, 0, 1,
            2048, 0, 1, int(epsg),
        )
        extratags.append((34735, 3, 8, geo_key_directory, False))
    tifffile.imwrite(
        path, arr, photometric='minisblack',
        planarconfig='contig', extratags=extratags,
    )
    return m


def test_open_geotiff_rotated_emits_rotated_affine(tmp_path):
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    src = tmp_path / "tmp_2129_rotated_attr_eager.tif"
    m = _write_rotated_tiff(str(src), arr, epsg=4326)

    da = open_geotiff(str(src), allow_rotated=True)

    expected = (m[0], m[1], m[3], m[4], m[5], m[7])
    assert da.attrs.get('rotated_affine') == expected
    assert isinstance(da.attrs['rotated_affine'], tuple)
    # CRS attrs stay dropped on this path (#2126).
    assert 'crs' not in da.attrs
    assert 'transform' not in da.attrs


def test_open_geotiff_rotated_emits_rotated_affine_dask(tmp_path):
    arr = np.arange(40, dtype='<u2').reshape(5, 8)
    src = tmp_path / "tmp_2129_rotated_attr_dask.tif"
    m = _write_rotated_tiff(str(src), arr, epsg=4326)

    da = open_geotiff(str(src), allow_rotated=True, chunks=4)

    expected = (m[0], m[1], m[3], m[4], m[5], m[7])
    assert da.attrs.get('rotated_affine') == expected


def test_open_geotiff_plain_no_georef_omits_rotated_affine(tmp_path):
    """A plain TIFF with no transform tags must not grow a
    ``rotated_affine`` attr. The opt-in is rotation-specific."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(12, dtype='<u2').reshape(3, 4)
    src = tmp_path / "tmp_2129_plain_no_georef.tif"
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     planarconfig='contig')

    da = open_geotiff(str(src), allow_rotated=True)
    assert 'rotated_affine' not in da.attrs


# ---------------------------------------------------------------------------
# VRT path: rotated VRTs must surface the same attr (review follow-up).
# ---------------------------------------------------------------------------


def _write_rotated_vrt(tmp_path, src_path, *, gt_str, size=4):
    """Build a VRT pointing at ``src_path`` with a custom ``<GeoTransform>``.

    GDAL ``geo_transform`` ordering is
    ``(origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height)``;
    non-zero ``rot_x`` / ``rot_y`` is what the reader treats as a
    rotated VRT (see ``_vrt_is_rotated`` in ``_backends/vrt.py``).
    """
    vrt_xml = (
        f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}">\n'
        f'  <GeoTransform>{gt_str}</GeoTransform>\n'
        f'  <VRTRasterBand dataType="UInt16" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    vrt = tmp_path / "tmp_2129_rotated.vrt"
    vrt.write_text(vrt_xml)
    return str(vrt)


def test_open_geotiff_rotated_vrt_emits_rotated_affine(tmp_path):
    """A VRT with a rotated ``<GeoTransform>`` opened with
    ``allow_rotated=True`` lands in ``georef_status='rotated_dropped'``
    and must surface ``rotated_affine`` so callers can recover the
    mapping. Mirrors the non-VRT ``ModelTransformationTag`` path."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(16, dtype='<u2').reshape(4, 4)
    src = tmp_path / "tmp_2129_rotated_vrt_src.tif"
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     planarconfig='contig')

    # GDAL geo_transform: origin_x, res_x, rot_x, origin_y, rot_y, res_y.
    # Non-zero rotation terms (positions 2 and 4) trigger the rotated
    # path. rasterio Affine ordering: (a, b, c, d, e, f)
    # = (res_x, rot_x, origin_x, rot_y, res_y, origin_y).
    gt_str = "100.0, 10.0, 5.0, 200.0, -5.0, -10.0"
    expected = (10.0, 5.0, 100.0, -5.0, -10.0, 200.0)

    vrt_path = _write_rotated_vrt(tmp_path, str(src), gt_str=gt_str)
    da = open_geotiff(vrt_path, allow_rotated=True)

    assert da.attrs.get('rotated_affine') == expected
    assert isinstance(da.attrs['rotated_affine'], tuple)
    # Same drops as the non-VRT path (#2126).
    assert 'crs' not in da.attrs
    assert 'transform' not in da.attrs


def test_open_geotiff_rotated_vrt_emits_rotated_affine_dask(tmp_path):
    """Chunked VRT read path mirrors the eager path: ``rotated_affine``
    rides on the metadata record at the chunked build site too."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(16, dtype='<u2').reshape(4, 4)
    src = tmp_path / "tmp_2129_rotated_vrt_src_dask.tif"
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     planarconfig='contig')

    gt_str = "100.0, 10.0, 5.0, 200.0, -5.0, -10.0"
    expected = (10.0, 5.0, 100.0, -5.0, -10.0, 200.0)

    vrt_path = _write_rotated_vrt(tmp_path, str(src), gt_str=gt_str)
    da = open_geotiff(vrt_path, allow_rotated=True, chunks=2)

    assert da.attrs.get('rotated_affine') == expected


def test_open_geotiff_axis_aligned_vrt_omits_rotated_affine(tmp_path):
    """An axis-aligned VRT (zero rotation terms) round-trips via
    ``attrs['transform']`` and must NOT grow a ``rotated_affine``."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(16, dtype='<u2').reshape(4, 4)
    src = tmp_path / "tmp_2129_axis_vrt_src.tif"
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     planarconfig='contig')

    gt_str = "100.0, 10.0, 0.0, 200.0, 0.0, -10.0"
    vrt_path = _write_rotated_vrt(tmp_path, str(src), gt_str=gt_str)

    da = open_geotiff(vrt_path, allow_rotated=True)
    assert 'rotated_affine' not in da.attrs


def test_open_geotiff_axis_aligned_omits_rotated_affine(tmp_path):
    """An axis-aligned ``ModelTransformationTag`` round-trips via
    ``attrs['transform']``; the rotated attr must stay absent so
    downstream code does not branch on it for non-rotated reads."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(12, dtype='<u2').reshape(3, 4)
    src = tmp_path / "tmp_2129_axis_aligned.tif"
    m = (
        10.0, 0.0, 0.0, 100.0,
        0.0, -10.0, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    tifffile.imwrite(
        str(src), arr, photometric='minisblack', planarconfig='contig',
        extratags=[(34264, 12, 16, m, False)],
    )

    da = open_geotiff(str(src), allow_rotated=True)
    assert 'rotated_affine' not in da.attrs
    assert da.attrs.get('transform') == (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
