"""Regression tests for issue #1911.

``to_geotiff(..., nodata=True)`` used to be accepted without validation.
``bool`` is a subclass of ``int``, so the typo slipped past every
downstream ``isinstance(nodata, (int, float))`` guard. The geotag builder
then wrote ``str(True)`` -> ``"True"`` into GDAL_NODATA. No reader parses
that as numeric, so the round-trip silently dropped the sentinel. The
fix rejects ``bool`` / ``np.bool_`` early at the writer entry point with
a clear ``TypeError``, with a belt-and-braces copy of the same check in
``build_geo_tags`` for callers that bypass ``to_geotiff``.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import GeoTransform, build_geo_tags


@pytest.fixture
def uint8_da():
    """Small uint8 DataArray for nodata round-trip tests."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    return xr.DataArray(arr, dims=['y', 'x'])


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_to_geotiff_rejects_bool_nodata(uint8_da, tmp_path, bad):
    """``to_geotiff`` raises ``TypeError`` for any bool / np.bool_ nodata."""
    path = str(tmp_path / "tmp_1911_bool.tif")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        to_geotiff(uint8_da, path, nodata=bad)


@pytest.mark.parametrize(
    "good",
    [0, 0.0, -9999, 255, np.int16(-1), np.float32(0.5)],
)
def test_to_geotiff_accepts_numeric_nodata(uint8_da, tmp_path, good):
    """Numeric nodata sentinels still go through without complaint."""
    path = str(tmp_path / "tmp_1911_numeric.tif")
    # uint8 raster cannot hold negative sentinels at the pixel level, but
    # the writer must still accept them at the tag level (the on-disk
    # GDAL_NODATA string round-trips intact).
    to_geotiff(uint8_da, path, nodata=good)


def test_to_geotiff_accepts_none_nodata(uint8_da, tmp_path):
    """``nodata=None`` is the documented default and must keep working."""
    path = str(tmp_path / "tmp_1911_none.tif")
    to_geotiff(uint8_da, path, nodata=None)
    r = open_geotiff(path)
    # No nodata tag was written, so no nodata attribute appears on read.
    assert "nodata" not in r.attrs


def test_to_geotiff_numeric_nodata_roundtrips(tmp_path):
    """A valid numeric sentinel survives the to_geotiff / open_geotiff cycle."""
    arr = np.full((4, 4), 7, dtype=np.uint8)
    arr[0, 0] = 255  # mark one pixel as the sentinel
    da = xr.DataArray(arr, dims=['y', 'x'])
    path = str(tmp_path / "tmp_1911_roundtrip.tif")
    to_geotiff(da, path, nodata=255)

    r = open_geotiff(path)
    # Reader surfaces the numeric sentinel.
    assert "nodata" in r.attrs
    assert int(float(r.attrs["nodata"])) == 255


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_build_geo_tags_rejects_bool_nodata(bad):
    """The lower-level builder also rejects bool, in case ``to_geotiff`` is bypassed."""
    transform = GeoTransform()
    with pytest.raises(TypeError, match="nodata must be numeric"):
        build_geo_tags(transform, crs_epsg=4326, nodata=bad)


def test_build_geo_tags_accepts_numeric_nodata():
    """``build_geo_tags`` still writes numeric sentinels into the tag dict."""
    from xrspatial.geotiff._geotags import TAG_GDAL_NODATA
    transform = GeoTransform()
    tags = build_geo_tags(transform, crs_epsg=4326, nodata=-9999)
    assert tags[TAG_GDAL_NODATA] == "-9999"
