"""Regression tests for issue #1768.

`build_geo_tags` writes a user-defined CRS (no EPSG, WKT only) by
setting ``ProjectedCSType`` / ``GeographicType`` = 32767 and storing
the raw WKT in ``GTCitationGeoKey``. libgeotiff and GDAL can recover
the CRS by parsing the citation, but many other GeoTIFF readers treat
the citation as a free-form name and silently drop the projection.

The fix emits a ``UserWarning`` when the WKT-only path is taken so the
interop limitation is visible at write time. The citation bytes are
unchanged, so existing round-trips through this library still work.

Tests pin:

* the warning fires on the WKT-only path,
* the warning does NOT fire on the EPSG path,
* the GeoKeyDirectory + GeoAsciiParams bytes are still emitted (so
  libgeotiff / GDAL readers keep working).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import GeoTransform, build_geo_tags

_USER_DEFINED_WKT = (
    'PROJCS["User defined LCC 1768",'
    'GEOGCS["NAD83",'
    'DATUM["North American Datum 1983",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Lambert Conformal Conic 2SP"],'
    'PARAMETER["central_meridian",-100],'
    'PARAMETER["latitude_of_origin",40],'
    'PARAMETER["standard_parallel_1",30],'
    'PARAMETER["standard_parallel_2",50],'
    'UNIT["metre",1]]'
)


# ---------------------------------------------------------------------------
# Unit tests on build_geo_tags
# ---------------------------------------------------------------------------


def test_build_geo_tags_warns_on_wkt_only_path():
    """Calling build_geo_tags with crs_wkt and no crs_epsg fires a
    UserWarning pointing at EPSG as the preferred input."""
    t = GeoTransform(origin_x=0.0, origin_y=10.0,
                     pixel_width=1.0, pixel_height=-1.0)
    with pytest.warns(UserWarning, match=r"user-defined CRS via WKT only"):
        tags = build_geo_tags(t, crs_wkt=_USER_DEFINED_WKT)

    # The citation bytes must still be emitted -- the warning replaces
    # nothing, it only annotates the call. libgeotiff / GDAL readers
    # need GeoKeyDirectory + GeoAsciiParams to recover the CRS.
    assert 34735 in tags  # TAG_GEO_KEY_DIRECTORY
    assert 34737 in tags  # TAG_GEO_ASCII_PARAMS
    assert _USER_DEFINED_WKT in tags[34737]


def test_build_geo_tags_no_warning_on_epsg_path():
    """The EPSG path must not warn -- it emits the standard GeoKeys
    every reader understands."""
    t = GeoTransform(origin_x=0.0, origin_y=10.0,
                     pixel_width=1.0, pixel_height=-1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        tags = build_geo_tags(t, crs_epsg=4326)
    assert 34735 in tags


def test_build_geo_tags_no_warning_when_no_crs():
    """No CRS at all -> no warning (this is a benign georef-only case)."""
    t = GeoTransform(origin_x=0.0, origin_y=10.0,
                     pixel_width=1.0, pixel_height=-1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_geo_tags(t)


# ---------------------------------------------------------------------------
# End-to-end through to_geotiff
# ---------------------------------------------------------------------------


def _wkt_only_da():
    arr = np.ones((4, 4), dtype=np.float32)
    return xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 47.0, 4),
                'x': np.linspace(10.0, 13.0, 4)},
        attrs={'crs_wkt': _USER_DEFINED_WKT},
    )


def test_to_geotiff_warns_on_wkt_only_crs(tmp_path):
    """to_geotiff with attrs['crs_wkt'] only (no attrs['crs']) emits a
    UserWarning pointing the caller at EPSG."""
    da = _wkt_only_da()
    p = str(tmp_path / "wkt_only_1768.tif")

    with pytest.warns(UserWarning, match=r"user-defined CRS via WKT only"):
        to_geotiff(da, p, compression='none')

    # The file still carries the WKT in its citation -- the fix is a
    # warning, not a behavior change. Existing round-trips keep working.
    rd = open_geotiff(p)
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")


def test_to_geotiff_no_warning_on_epsg_crs(tmp_path):
    """The EPSG path must stay quiet -- it is the recommended input."""
    arr = np.ones((4, 4), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 47.0, 4),
                'x': np.linspace(10.0, 13.0, 4)},
        attrs={'crs': 4326},
    )
    p = str(tmp_path / "epsg_only_1768.tif")

    # No UserWarning matching the WKT-only message. Other warnings
    # (e.g. pyproj deprecations) are ignored by the match-filter.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_geotiff(da, p, compression='none')

    wkt_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "user-defined CRS via WKT only" in str(w.message)
    ]
    assert wkt_warnings == [], (
        f"EPSG path unexpectedly emitted the WKT-only warning: "
        f"{[str(w.message) for w in wkt_warnings]}"
    )


def test_wkt_only_citation_bytes_unchanged_after_fix(tmp_path):
    """The existing citation-WKT bytes must keep round-tripping. This
    regression-guards the contract from issue #1632: libgeotiff readers
    can still recover the CRS by parsing the citation."""
    tifffile = pytest.importorskip("tifffile")
    da = _wkt_only_da()
    p = str(tmp_path / "wkt_only_bytes_1768.tif")

    with warnings.catch_warnings():
        # We expect the warning here; suppress so the test focuses on
        # the byte-level contract.
        warnings.simplefilter("ignore", UserWarning)
        to_geotiff(da, p, compression='none')

    with tifffile.TiffFile(p) as tif:
        keys = tif.pages[0].tags.get(34735)  # GeoKeyDirectory
        ascii_tag = tif.pages[0].tags.get(34737)  # GeoAsciiParams
        # GeoKeyDirectory must contain at least the header (4 shorts) plus
        # KeyEntries for: model type, raster type, user-defined CRS
        # (ProjectedCSType or GeographicType = 32767), and the
        # GTCitationGeoKey pointer into GeoAsciiParams. Each entry is 4
        # shorts, so 4 + 4*4 = 20 shorts is the floor for the WKT-only
        # path.
        assert keys is not None
        assert len(keys.value) >= 20

        # Pin the user-defined CRS marker explicitly: one of
        # ProjectedCSType (3072) or GeographicType (2048) must be set to
        # 32767. The KeyEntries layout is [key_id, location, count,
        # value_or_offset] repeated; iterate as 4-tuples.
        entries = list(keys.value[4:])
        user_defined_crs_marker = False
        for i in range(0, len(entries), 4):
            key_id = entries[i]
            location = entries[i + 1]
            value = entries[i + 3]
            # location == 0 means the value is inline in the entry.
            if key_id in (2048, 3072) and location == 0 and value == 32767:
                user_defined_crs_marker = True
                break
        assert user_defined_crs_marker, (
            "GeoKeyDirectory must mark the CRS as user-defined (32767) "
            "via ProjectedCSType (3072) or GeographicType (2048)"
        )

        # Pin the citation payload: the full WKT must round-trip through
        # GeoAsciiParams, not just the leading ``PROJCS[``.
        assert ascii_tag is not None
        # tifffile strips the GeoTIFF terminator (``|`` or ``\x00``);
        # match the WKT body inclusive of the trailing closing bracket.
        assert _USER_DEFINED_WKT in ascii_tag.value
