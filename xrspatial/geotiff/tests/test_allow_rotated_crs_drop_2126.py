"""``allow_rotated=True`` drops CRS attrs together with the transform (#2126).

Pre-fix, ``_populate_attrs_from_geo_info`` wrote ``crs`` / ``crs_wkt``
before checking ``has_georef``. On the ``allow_rotated=True`` opt-in
path (issue #2115) the transform was dropped but CRS attrs survived,
which mismatched the public docstring and misled downstream code that
gated on ``"crs" in da.attrs`` to mean "spatially meaningful".

The fix detects the rotated opt-in path (transform carries
``rotated_affine`` and ``has_georef`` is False) and skips the CRS
writes there. Plain no-georef files (no transform tags at all) still
keep their CRS attrs -- only the rotated path drops them.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._attrs import _populate_attrs_from_geo_info
from xrspatial.geotiff._geotags import GeoInfo, GeoTransform


def _rotated_geo_info(*, with_crs: bool = True) -> GeoInfo:
    """Build a GeoInfo that mimics the allow_rotated=True parser output."""
    t = GeoTransform(
        origin_x=0.0, origin_y=0.0,
        pixel_width=1.0, pixel_height=-1.0,
        rotated_affine=(8.66, -5.0, 100.0, 5.0, 8.66, 200.0),
    )
    return GeoInfo(
        transform=t,
        has_georef=False,
        crs_epsg=4326 if with_crs else None,
        crs_wkt='GEOGCS["WGS 84"]' if with_crs else None,
    )


def test_rotated_optin_drops_crs_epsg():
    gi = _rotated_geo_info()
    attrs = {}
    _populate_attrs_from_geo_info(attrs, gi)
    assert 'crs' not in attrs
    assert 'transform' not in attrs


def test_rotated_optin_drops_crs_wkt():
    gi = _rotated_geo_info()
    attrs = {}
    _populate_attrs_from_geo_info(attrs, gi)
    assert 'crs_wkt' not in attrs


def test_plain_no_georef_keeps_crs():
    # Plain no-georef file: no transform tag at all, but the GeoKey
    # directory carries a CRS. This case must still emit ``crs`` so
    # callers can display / reproject the array even without a transform.
    gi = GeoInfo(transform=GeoTransform(), has_georef=False, crs_epsg=4326)
    attrs = {}
    _populate_attrs_from_geo_info(attrs, gi)
    assert attrs.get('crs') == 4326
    assert 'transform' not in attrs


def test_axis_aligned_georef_keeps_crs_and_transform():
    gi = GeoInfo(
        transform=GeoTransform(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
        ),
        has_georef=True,
        crs_epsg=4326,
    )
    attrs = {}
    _populate_attrs_from_geo_info(attrs, gi)
    assert attrs.get('crs') == 4326
    assert attrs.get('transform') == (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)


# ---------------------------------------------------------------------------
# End-to-end via open_geotiff: write a rotated GeoTIFF with embedded CRS,
# read it with allow_rotated=True, confirm the attrs are clean.
# ---------------------------------------------------------------------------


def _write_rotated_tiff_with_geokeys(path, arr, *, epsg=4326):
    """Build a synthetic rotated GeoTIFF with a GeoKey-encoded CRS."""
    tifffile = pytest.importorskip("tifffile")
    # Rotated 4x4 model transformation tag (30-deg rotation, pixel size 10).
    cos30 = 0.8660254037844387
    sin30 = 0.5
    m = (
        10.0 * cos30, -10.0 * sin30, 0.0, 100.0,
        10.0 * sin30,  10.0 * cos30, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    # GeoKeyDirectory: header (4 shorts) + one key for GeographicTypeGeoKey
    # (key id 2048, location 0 -> short value, count 1, value EPSG).
    geo_key_directory = (
        1, 1, 0, 1,        # KeyDirectoryVersion=1, KeyRevision=1, MinorRevision=0, NumberOfKeys=1
        2048, 0, 1, int(epsg),
    )
    extratags = [
        # ModelTransformationTag (34264) -- DOUBLE, count=16.
        (34264, 12, 16, m, False),
        # GeoKeyDirectoryTag (34735) -- SHORT, count=8.
        (34735, 3, 8, geo_key_directory, False),
    ]
    tifffile.imwrite(
        path, arr, photometric='minisblack',
        planarconfig='contig', extratags=extratags,
    )


def test_open_geotiff_rotated_with_crs_drops_crs(tmp_path):
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    src = tmp_path / "tmp_2126_rotated_with_crs.tif"
    _write_rotated_tiff_with_geokeys(str(src), arr, epsg=4326)
    da = open_geotiff(str(src), allow_rotated=True)
    assert 'crs' not in da.attrs, (
        f"allow_rotated=True must drop attrs['crs']; got {da.attrs.get('crs')!r}"
    )
    assert 'crs_wkt' not in da.attrs, (
        f"allow_rotated=True must drop attrs['crs_wkt']; "
        f"got {da.attrs.get('crs_wkt')!r}"
    )
    assert da.attrs.get('transform') is None or da.attrs.get('transform') == 'rotated'


def test_open_geotiff_rotated_with_crs_dask_drops_crs(tmp_path):
    arr = np.arange(40, dtype='<u2').reshape(5, 8)
    src = tmp_path / "tmp_2126_rotated_with_crs_dask.tif"
    _write_rotated_tiff_with_geokeys(str(src), arr, epsg=4326)
    da = open_geotiff(str(src), allow_rotated=True, chunks=4)
    assert 'crs' not in da.attrs
    assert 'crs_wkt' not in da.attrs
