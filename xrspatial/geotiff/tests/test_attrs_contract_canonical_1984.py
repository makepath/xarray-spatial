"""Locking test for the canonical tier of the attrs contract.

Issue #1984, PR 4 of 7.

The attrs contract (see ``xrspatial/geotiff/_attrs.py`` and
``docs/source/user_guide/attrs_contract.rst``) splits every key the
read paths emit into three tiers. This file pins the *canonical* tier:
keys xrspatial owns and guarantees round-trip stable through
``to_geotiff`` -> ``open_geotiff``.

Sibling files cover the other tiers:

* ``test_attrs_contract_aliases_1984.py`` -- compatibility aliases.
* ``test_attrs_contract_passthrough_1984.py`` -- best-effort
  pass-through.
* ``test_attrs_contract_version_1984.py`` -- per-backend stamping of
  ``attrs['_xrspatial_geotiff_contract']`` (also canonical, kept in its
  own file because the assertion is per read path rather than
  per round-trip).

The canonical keys locked here:

* ``crs``                          -- EPSG integer code.
* ``crs_wkt``                      -- horizontal CRS WKT string.
* ``transform``                    -- rasterio-style 6-tuple.
* ``nodata``                       -- declared file sentinel (GDAL_NODATA).
* ``raster_type``                  -- 'point' (set explicitly) or absent
                                      (= 'area', the implicit default).
* ``extra_tags``                   -- list of (id, type, count, value)
                                      tuples for out-of-band TIFF tags.
* ``gdal_metadata``                -- dict parsed from GDAL_METADATA XML.
* ``gdal_metadata_xml``            -- raw GDAL_METADATA XML string.
* ``x_resolution``, ``y_resolution``,
  ``resolution_unit``              -- TIFF XResolution / YResolution /
                                      ResolutionUnit.
* ``_xrspatial_geotiff_contract``  -- integer contract version. Stamped
                                      on every read.

The fixture below sets every canonical key on a synthetic DataArray,
round-trips it through ``to_geotiff`` -> ``open_geotiff``, and the test
suite below asserts both presence and value equality per key. The
single-fixture shape is intentional: a future writer change that drops
one canonical key shows up here as one failing assertion rather than
being lost in a larger diff.

Issues #1985 (parity matrix) and #1986 (round-trip invariants) consume
this assertion list. If you add a key here, update the canonical block
in the contract page and the ``_attrs.py`` module docstring as well.
"""
from __future__ import annotations

import importlib.util
import re

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import _ATTRS_CONTRACT_VERSION


_CONTRACT_KEY = '_xrspatial_geotiff_contract'

# Every key the canonical tier guarantees round-trip stable. Keep the
# order consistent with the contract docs so a diff here lines up with
# a diff in ``attrs_contract.rst``.
#
# ``raster_type`` is canonical but absent from this constant: the
# implicit default 'area' is encoded as *absence* of the attr, so the
# "must be present after round-trip" check below cannot express it.
# The dedicated ``test_raster_type_*`` tests at the bottom of this
# file lock both branches.
_CANONICAL_KEYS = (
    'crs',
    'crs_wkt',
    'transform',
    'nodata',
    'extra_tags',
    'gdal_metadata',
    'gdal_metadata_xml',
    'x_resolution',
    'y_resolution',
    'resolution_unit',
    # Added in contract v3 (issue #2136). The fixture is georef + CRS
    # so the round-tripped value is the ``'full'`` literal; the
    # value-equality check lives on ``test_georef_status_roundtrip``
    # below to keep the membership and value assertions independent.
    'georef_status',
    _CONTRACT_KEY,
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

# Fixture values, written into ``attrs`` on the synthetic DataArray and
# compared to the read-back attrs after round-trip. ``transform`` is
# pinned by ``y`` / ``x`` coords so we expect that exact 6-tuple back.
_NODATA_SENTINEL = -9999.0
_X_RES = 300.0
_Y_RES = 300.0
_RES_UNIT = 'inch'
# The shared fixture leaves raster_type unset (= implicit 'area'). The
# point-specific test rebuilds the attrs dict so the two raster_type
# branches do not share state. Keep ``_GDAL_META`` free of an
# ``AREA_OR_POINT`` entry so the fixture stays consistent under both
# branches.
_GDAL_META = {'TIFFTAG_SOFTWARE': 'xrspatial-1984'}
# Software tag (305, ASCII). Picked because it is benign (no spatial
# interpretation, no security filter) and tifffile decodes it as-is.
# Count must include the trailing NUL byte. The Software identity also
# appears as ``gdal_metadata['TIFFTAG_SOFTWARE']`` above; the two are
# independent channels (a raw TIFF tag vs an entry in the GDAL_METADATA
# XML payload) and the writer does not synchronise them.
_SOFTWARE_STR = 'xrspatial-canonical-1984'
_EXTRA_TAGS = [(305, 2, len(_SOFTWARE_STR) + 1, _SOFTWARE_STR)]
# ``crs_wkt`` is round-tripped via attrs['crs'] (the EPSG code drives
# the writer), so we leave it for the reader to emit. Setting it on
# write is allowed but not required; the read-back value comes from the
# PROJ database.


def _make_canonical_da():
    """Build a synthetic DataArray exercising every canonical attr.

    Returns the DataArray plus the expected ``transform`` tuple so the
    test can assert on the round-tripped value without recomputing it.
    """
    h, w = 4, 4
    data = np.arange(h * w, dtype=np.float32).reshape(h, w)
    # Pin a non-identity transform so the round-trip check catches a
    # writer that drops the tiepoint / pixel-scale tags. Coords are
    # interpreted as pixel centres; the emitted transform's origin is
    # the top-left corner, so origin_x = 100 - 10/2 = 95 and
    # origin_y = 240 - (-10)/2 = 245.
    x = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float64)
    y = np.array([240.0, 230.0, 220.0, 210.0], dtype=np.float64)
    expected_transform = (10.0, 0.0, 95.0, 0.0, -10.0, 245.0)

    da = xr.DataArray(
        data, dims=('y', 'x'), coords={'y': y, 'x': x},
        attrs={
            'crs': 4326,
            'nodata': _NODATA_SENTINEL,
            'extra_tags': list(_EXTRA_TAGS),
            'gdal_metadata': dict(_GDAL_META),
            'x_resolution': _X_RES,
            'y_resolution': _Y_RES,
            'resolution_unit': _RES_UNIT,
        },
    )
    return da, expected_transform


@pytest.fixture
def canonical_roundtrip(tmp_path):
    """Round-trip the canonical fixture through write -> read.

    Returns ``(read_da, expected_transform)``. Scoped to one round-trip
    per test so per-key assertions stay independent and a single failure
    points at one key rather than cascading.
    """
    da, expected_transform = _make_canonical_da()
    path = str(tmp_path / 'attrs_contract_canonical.tif')
    to_geotiff(da, path)
    rd = open_geotiff(path)
    return rd, expected_transform


# ---------------------------------------------------------------------------
# Single-fixture coverage: every canonical key is present on read-back.
# ---------------------------------------------------------------------------


def test_every_canonical_key_present(canonical_roundtrip):
    """Pin the canonical key set after a round-trip.

    A writer that drops one canonical key (e.g. forgets to emit
    GDAL_METADATA) shows up here as one missing key rather than as a
    later equality failure with a less obvious cause.
    """
    rd, _ = canonical_roundtrip
    missing = sorted(k for k in _CANONICAL_KEYS if k not in rd.attrs)
    assert missing == [], (
        f"canonical attrs missing after round-trip: {missing}. "
        f"attrs keys present: {sorted(rd.attrs.keys())}"
    )


# ---------------------------------------------------------------------------
# Per-key value assertions: each canonical key round-trips by value.
# ---------------------------------------------------------------------------


def test_crs_roundtrip(canonical_roundtrip):
    rd, _ = canonical_roundtrip
    assert rd.attrs['crs'] == 4326


def test_crs_wkt_roundtrip(canonical_roundtrip):
    """``crs_wkt`` is reader-emitted from the EPSG code. Pin presence
    and the CRS-identity substring callers rely on. The exact WKT is
    PROJ-version dependent, so match ``WGS 84`` / ``WGS_1984`` / ``WGS-84``
    with one regex rather than a single literal."""
    rd, _ = canonical_roundtrip
    wkt = rd.attrs['crs_wkt']
    assert isinstance(wkt, str) and len(wkt) > 0
    assert re.search(r'WGS[\s_-]?84|WGS_1984', wkt), (
        f"crs_wkt round-trip lost the CRS identity: {wkt!r}"
    )


def test_transform_roundtrip(canonical_roundtrip):
    rd, expected_transform = canonical_roundtrip
    t = tuple(rd.attrs['transform'])
    assert t == pytest.approx(expected_transform), (
        f"transform round-trip mismatch.\n  expected: {expected_transform}\n"
        f"  got     : {t}"
    )


def test_nodata_roundtrip(canonical_roundtrip):
    rd, _ = canonical_roundtrip
    assert rd.attrs['nodata'] == _NODATA_SENTINEL


def test_extra_tags_roundtrip(canonical_roundtrip):
    """A non-friendly extra_tags entry (Software, 305) round-trips
    intact. The writer must preserve unknown tags so users can attach
    arbitrary metadata."""
    rd, _ = canonical_roundtrip
    got = rd.attrs['extra_tags']
    # Look up tag 305 specifically; ordering and any reader-added
    # entries are not part of the contract for this assertion.
    by_id = {t[0]: t for t in got}
    assert 305 in by_id, (
        f"Software tag (305) missing from read-back extra_tags: {got}"
    )
    tag_id, type_id, count, value = by_id[305]
    assert tag_id == 305
    assert type_id == 2  # TIFF ASCII
    assert value == _SOFTWARE_STR


def test_gdal_metadata_roundtrip(canonical_roundtrip):
    """The parsed dict survives the round-trip key-by-key. Allow extra
    entries the reader might inject (e.g. ``STATISTICS_*``) so this
    test is not a tripwire for unrelated reader changes."""
    rd, _ = canonical_roundtrip
    got = rd.attrs['gdal_metadata']
    assert isinstance(got, dict), f"gdal_metadata is not a dict: {got!r}"
    for k, v in _GDAL_META.items():
        assert got.get(k) == v, (
            f"gdal_metadata[{k!r}] mismatch.\n  expected: {v!r}\n"
            f"  got     : {got.get(k)!r}\n  full read-back: {got!r}"
        )


def test_gdal_metadata_xml_roundtrip(canonical_roundtrip):
    """The raw XML string is reconstructed by the writer from the
    ``gdal_metadata`` dict. Pin presence + the substring that proves
    our fixture survived; the exact XML formatting is writer-dependent."""
    rd, _ = canonical_roundtrip
    xml = rd.attrs['gdal_metadata_xml']
    assert isinstance(xml, str) and xml.startswith('<GDALMetadata>')
    assert 'xrspatial-1984' in xml, (
        f"gdal_metadata_xml lost the fixture marker: {xml!r}"
    )


def test_resolution_group_roundtrip(canonical_roundtrip):
    """``x_resolution`` / ``y_resolution`` / ``resolution_unit`` are
    written and read as one logical unit -- pin them together so a
    writer that drops one but keeps the others fails here."""
    rd, _ = canonical_roundtrip
    assert rd.attrs['x_resolution'] == pytest.approx(_X_RES)
    assert rd.attrs['y_resolution'] == pytest.approx(_Y_RES)
    assert rd.attrs['resolution_unit'] == _RES_UNIT


def test_contract_version_roundtrip(canonical_roundtrip):
    """``_xrspatial_geotiff_contract`` is stamped on every read; pin
    that the canonical fixture sees the current version. Per-backend
    coverage lives in ``test_attrs_contract_version_1984.py``."""
    rd, _ = canonical_roundtrip
    assert rd.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION


def test_georef_status_roundtrip(canonical_roundtrip):
    """``georef_status`` (issue #2136) is canonical from contract v3.
    The fixture sets ``crs`` + axis-aligned transform-from-coords, so
    the round-tripped value must be the ``'full'`` literal."""
    rd, _ = canonical_roundtrip
    assert rd.attrs['georef_status'] == 'full'


# ---------------------------------------------------------------------------
# Per-backend coverage for canonical-key *presence*.
#
# The 7-PR plan in issue #1984 asked for "one fixture per backend with
# explicit assertions that every canonical key is present and round-trips
# byte-equivalent through write -> read". Read-time backends share
# ``_populate_attrs_from_geo_info``, so per-key value round-trips are
# pinned once (above) on the eager numpy path. What the per-backend
# check guards against is a backend skipping the shared helper or
# building its attrs dict independently; the version-stamp test does
# this for one canonical key, and the loop below does it for the rest.
# ---------------------------------------------------------------------------


def _open_eager(path):
    return open_geotiff(path)


def _open_dask(path):
    return open_geotiff(path, chunks=2)


def _open_gpu(path):
    return open_geotiff(path, gpu=True)


def _open_dask_gpu(path):
    return open_geotiff(path, gpu=True, chunks=2)


_BACKEND_OPENERS = [
    pytest.param(_open_eager, id='eager-numpy'),
    pytest.param(_open_dask, id='dask-numpy'),
    pytest.param(_open_gpu, id='gpu', marks=_gpu_only),
    pytest.param(_open_dask_gpu, id='dask-gpu', marks=_gpu_only),
]


@pytest.mark.parametrize('opener', _BACKEND_OPENERS)
def test_canonical_keys_present_per_backend(tmp_path, opener):
    """Each read backend emits the full canonical key set.

    Writes the canonical fixture once with the eager writer (only the
    eager + dask writers exist; the GPU writer is exercised in its own
    parity tests), then re-reads it through every supported backend and
    asserts presence. Value-level round-trip is checked by the per-key
    tests above; this loop guards against a backend that bypasses
    ``_populate_attrs_from_geo_info``.
    """
    da, _ = _make_canonical_da()
    path = str(tmp_path / f'canonical_{opener.__name__}.tif')
    to_geotiff(da, path)

    rd = opener(path)
    missing = sorted(k for k in _CANONICAL_KEYS if k not in rd.attrs)
    assert missing == [], (
        f"{opener.__name__}: canonical attrs missing after round-trip: "
        f"{missing}. attrs keys present: {sorted(rd.attrs.keys())}"
    )


# ---------------------------------------------------------------------------
# ``raster_type`` lives outside the shared fixture because the canonical
# default ('area') is encoded as *absence* in attrs. The two branches need
# different fixtures.
# ---------------------------------------------------------------------------


def test_raster_type_area_omitted_on_roundtrip(tmp_path):
    """RasterPixelIsArea is the implicit default and is encoded as
    *absence* of ``attrs['raster_type']``. A DataArray with no
    ``raster_type`` attr must round-trip to a DataArray that still has
    no ``raster_type`` attr."""
    da, _ = _make_canonical_da()
    assert 'raster_type' not in da.attrs
    path = str(tmp_path / 'raster_type_area.tif')
    to_geotiff(da, path)
    rd = open_geotiff(path)
    assert 'raster_type' not in rd.attrs, (
        f"area is the implicit default but the reader emitted "
        f"raster_type={rd.attrs['raster_type']!r}"
    )


def test_raster_type_point_roundtrip(tmp_path):
    """``raster_type='point'`` is the only value the writer accepts via
    attrs; the reader emits it back on a round-trip."""
    da, _ = _make_canonical_da()
    da.attrs['raster_type'] = 'point'
    path = str(tmp_path / 'raster_type_point.tif')
    to_geotiff(da, path)
    rd = open_geotiff(path)
    assert rd.attrs.get('raster_type') == 'point'
