"""Regression tests for issue #1632.

Files with a user-defined CRS (no EPSG, WKT stored in the GeoTIFF
citation under ``GEOKEY_*_CS_TYPE == 32767``) used to round-trip with
``attrs['crs_name']`` set but ``attrs['crs_wkt']`` and ``attrs['crs']``
unset. ``to_geotiff`` only consults the latter two, so a read -> write
cycle silently dropped the projection.

The fix promotes the citation to ``attrs['crs_wkt']`` whenever no EPSG
is resolved and the citation parses as WKT (starts with one of the
known WKT 1 / WKT 2 root keywords). ``crs_name`` stays populated for
back-compat. Tests pin the contract across all four read backends and
across the read -> write -> read round trip.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

tifffile = pytest.importorskip("tifffile")

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import _looks_like_wkt


# A user-defined Lambert Conformal Conic that pyproj cannot identify
# as a registered EPSG. Trimmed to keep test fixtures readable.
_USER_DEFINED_WKT = (
    'PROJCS["User defined LCC",'
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


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


def _write_user_defined_crs_tif(path):
    """Write a tiny GeoTIFF with WKT-only CRS and return the DataArray written."""
    arr = np.ones((4, 4), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 47.0, 4),
                'x': np.linspace(10.0, 13.0, 4)},
        attrs={'crs_wkt': _USER_DEFINED_WKT},
    )
    to_geotiff(da, path, compression='none')
    return da


# ---------------------------------------------------------------------------
# _looks_like_wkt unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    'PROJCS["foo"]',
    'GEOGCS["foo"]',
    'PROJCRS["foo"]',
    'GEOGCRS["foo"]',
    'COMPD_CS["foo"]',
    'COMPOUNDCRS["foo"]',
    'BOUNDCRS["foo"]',
    'VERT_CS["foo"]',
    'VERTCRS["foo"]',
    'LOCAL_CS["foo"]',
    '  PROJCS["leading whitespace"]',
    'projcs["lowercase"]',
])
def test_looks_like_wkt_positive(text):
    """Top-level WKT 1 / WKT 2 keywords parse as WKT."""
    assert _looks_like_wkt(text)


def test_looks_like_wkt_requires_bracket():
    """A keyword without the opening bracket is not WKT."""
    # "PROJCS" alone is a token, not a complete WKT element. The check
    # demands the bracket so plain-text references to WKT keywords in
    # human-readable names do not collide with the WKT path.
    assert not _looks_like_wkt("PROJCS")
    assert not _looks_like_wkt("PROJCS no bracket here")


@pytest.mark.parametrize("text", [
    None,
    '',
    'NAD83 / UTM Zone 12N',           # human-readable name, not WKT
    'epsg:4326',                       # urn-like
    'WGS 84',
    'Some random string',
    'PROJ string +proj=longlat +datum=WGS84',
    42,                                # non-string input
    b'PROJCS["bytes input"]',          # bytes, not str
])
def test_looks_like_wkt_negative(text):
    """Non-WKT inputs return False (including non-string types)."""
    assert not _looks_like_wkt(text)


# ---------------------------------------------------------------------------
# Read-side: backends emit crs_wkt for user-defined CRS files
# ---------------------------------------------------------------------------


def test_eager_emits_crs_wkt_for_user_defined_crs(tmp_path):
    """The eager numpy read populates attrs['crs_wkt'] when the file's
    citation carries WKT, even without an EPSG."""
    p = str(tmp_path / "user_defined_crs.tif")
    _write_user_defined_crs_tif(p)

    rd = open_geotiff(p)
    assert rd.attrs.get("crs") is None  # no EPSG
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")
    # crs_name is kept for back-compat
    assert rd.attrs.get("crs_name") == rd.attrs["crs_wkt"]


def test_dask_emits_crs_wkt_for_user_defined_crs(tmp_path):
    """The dask read path emits the same crs_wkt as numpy."""
    p = str(tmp_path / "user_defined_crs_dask.tif")
    _write_user_defined_crs_tif(p)

    rd = open_geotiff(p, chunks=4)
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")


@_gpu_only
def test_cupy_emits_crs_wkt_for_user_defined_crs(tmp_path):
    """The cupy / GPU read path emits the same crs_wkt as numpy."""
    p = str(tmp_path / "user_defined_crs_gpu.tif")
    _write_user_defined_crs_tif(p)

    rd = open_geotiff(p, gpu=True)
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")


@_gpu_only
def test_dask_cupy_emits_crs_wkt_for_user_defined_crs(tmp_path):
    """The dask+cupy read path emits the same crs_wkt as numpy."""
    p = str(tmp_path / "user_defined_crs_dask_gpu.tif")
    _write_user_defined_crs_tif(p)

    rd = open_geotiff(p, gpu=True, chunks=4)
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")


# ---------------------------------------------------------------------------
# Read -> write -> read round trip: WKT survives the second write
# ---------------------------------------------------------------------------


def test_user_defined_crs_round_trips_through_to_geotiff(tmp_path):
    """A read -> write of a user-defined CRS file keeps the projection.

    Pre-fix, ``to_geotiff(open_geotiff(src), dst)`` produced ``dst`` with
    no GeoKey CRS entries and no GeoAsciiParams tag because the read path
    only set ``attrs['crs_name']`` and the writer never consults that.
    """
    src = str(tmp_path / "round_trip_src.tif")
    _write_user_defined_crs_tif(src)

    rd = open_geotiff(src)
    dst = str(tmp_path / "round_trip_dst.tif")
    to_geotiff(rd, dst, compression='none')

    # The second file should carry the same WKT in its citation.
    rd2 = open_geotiff(dst)
    assert rd2.attrs.get("crs_wkt") == rd.attrs.get("crs_wkt")

    # And the raw GeoKey + ASCII tags must be present.
    with tifffile.TiffFile(dst) as tif:
        keys = tif.pages[0].tags.get(34735)  # GeoKeyDirectory
        ascii_tag = tif.pages[0].tags.get(34737)  # GeoAsciiParams
        assert keys is not None
        # GeoKeyDirectory header is 4 entries; a real CRS adds 3+ key
        # entries (model type, raster type, GTCitation -> ascii ref).
        assert len(keys.value) > 4
        assert ascii_tag is not None
        assert "PROJCS[" in ascii_tag.value


def test_epsg_crs_unchanged_by_fix(tmp_path):
    """The fix must not regress the EPSG path: files with attrs['crs'] = <int>
    should still emit both crs and crs_wkt on read."""
    arr = np.ones((4, 4), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 47.0, 4),
                'x': np.linspace(10.0, 13.0, 4)},
        attrs={'crs': 4326},
    )
    p = str(tmp_path / "epsg.tif")
    to_geotiff(da, p, compression='none')

    rd = open_geotiff(p)
    assert rd.attrs.get("crs") == 4326
    assert rd.attrs.get("crs_wkt") is not None
    # The WKT here is pyproj's canonical 4326 WKT; the citation is the
    # short EPSG-style name "WGS 84", not WKT, so crs_name should not
    # be promoted to crs_wkt.
    assert rd.attrs.get("crs_name") != rd.attrs.get("crs_wkt")


def test_human_readable_crs_name_not_promoted_to_crs_wkt(tmp_path):
    """A citation that is a human-readable name (not WKT) must stay in
    crs_name only. The _looks_like_wkt gate prevents accidental promotion."""
    # tifffile-built file with citation 'NAD83 / UTM Zone 12N' as the
    # citation, no EPSG. We can't easily build the GeoKey table from
    # scratch here without recapitulating extract_geo_info; instead we
    # exercise the path via the helper directly.
    assert not _looks_like_wkt("NAD83 / UTM Zone 12N")
    assert not _looks_like_wkt("WGS 84")
    assert not _looks_like_wkt("")
    assert not _looks_like_wkt(None)
