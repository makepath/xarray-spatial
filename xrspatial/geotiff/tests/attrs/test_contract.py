"""Attrs-contract tests for the GeoTIFF module.

Four dimensions of the attrs contract:

* ``canonical``    -- keys xrspatial owns and guarantees round-trip
                      stable through ``to_geotiff`` -> ``open_geotiff``.
* ``aliases``      -- compatibility aliases (rioxarray ``nodatavals``,
                      CF ``_FillValue``) honoured on read when canonical
                      ``nodata`` is absent.
* ``passthrough``  -- best-effort tier: writer does not consume, reader
                      rebuilds the value from a TIFF tag if one survived.
* ``version``      -- the ``attrs['_xrspatial_geotiff_contract']`` stamp,
                      emitted by every read backend (eager numpy, dask,
                      GPU, dask+GPU, VRT eager, VRT chunked).

Each dimension is a separate top-level section. Tests are parametrised
within each dimension and given descriptive IDs
(``canonical[GDAL_NODATA]``, ``alias[nodatavals->nodata]``, etc.).

The release-gate attrs-contract tests live in their own release-gate
module; parity-flavoured attrs tests live with the parity suite.
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import ConflictingNodataError
from xrspatial.geotiff import _attrs as _attrs_module
from xrspatial.geotiff import _read_vrt, open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import _ATTRS_CONTRACT_VERSION, _resolve_nodata_attr

from .._helpers.markers import requires_gpu

tifffile = pytest.importorskip("tifffile")


_CONTRACT_KEY = '_xrspatial_geotiff_contract'


# ===========================================================================
# Canonical tier
#
# Keys xrspatial owns; round-trip is guaranteed through write -> read.
# ===========================================================================

# Every key the canonical tier guarantees round-trip stable. Order
# mirrors ``docs/source/user_guide/attrs_contract.rst`` so a diff here
# lines up with a diff in the contract docs.
#
# ``raster_type`` is canonical but absent from this constant: the
# implicit default 'area' is encoded as *absence* of the attr, so the
# "must be present after round-trip" check below cannot express it.
# The dedicated ``test_raster_type_*`` tests below lock both branches.
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
    # Added in contract v3. The fixture is georef + CRS
    # so the round-tripped value is the ``'full'`` literal; value
    # equality lives on ``test_canonical[georef_status]`` below.
    'georef_status',
    _CONTRACT_KEY,
)

# Fixture values, written into ``attrs`` on the synthetic DataArray and
# compared to the read-back attrs after round-trip. ``transform`` is
# pinned by ``y`` / ``x`` coords so we expect that exact 6-tuple back.
_NODATA_SENTINEL = -9999.0
_X_RES = 300.0
_Y_RES = 300.0
_RES_UNIT = 'inch'
# Software tag (305, ASCII). Benign (no spatial interpretation, no
# security filter) and tifffile decodes it as-is. Count includes the
# trailing NUL byte. The Software identity also appears as
# ``gdal_metadata['TIFFTAG_SOFTWARE']``; the two are independent
# channels (raw TIFF tag vs an entry in the GDAL_METADATA XML payload)
# and the writer does not synchronise them.
_GDAL_META = {'TIFFTAG_SOFTWARE': 'xrspatial-1984'}
_SOFTWARE_STR = 'xrspatial-canonical-1984'
_EXTRA_TAGS = [(305, 2, len(_SOFTWARE_STR) + 1, _SOFTWARE_STR)]


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
    # The canonical fixture carries ``extra_tags``; the rich-tag gate
    # exempts attrs carrying ``_xrspatial_geotiff_contract`` (set by the
    # readers), but this fixture builds a fresh DataArray without the
    # marker, so the initial write must opt in.
    to_geotiff(da, path, allow_experimental_codecs=True)
    rd = open_geotiff(path)
    return rd, expected_transform


def test_canonical_every_key_present(canonical_roundtrip):
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


def _check_canonical_crs(rd, _expected_transform):
    assert rd.attrs['crs'] == 4326


def _check_canonical_crs_wkt(rd, _expected_transform):
    """``crs_wkt`` is reader-emitted from the EPSG code. Pin presence
    and the CRS-identity substring callers rely on. The exact WKT is
    PROJ-version dependent, so match ``WGS 84`` / ``WGS_1984`` /
    ``WGS-84`` with one regex rather than a single literal."""
    wkt = rd.attrs['crs_wkt']
    assert isinstance(wkt, str) and len(wkt) > 0
    assert re.search(r'WGS[\s_-]?84|WGS_1984', wkt), (
        f"crs_wkt round-trip lost the CRS identity: {wkt!r}"
    )


def _check_canonical_transform(rd, expected_transform):
    t = tuple(rd.attrs['transform'])
    assert t == pytest.approx(expected_transform), (
        f"transform round-trip mismatch.\n  expected: {expected_transform}\n"
        f"  got     : {t}"
    )


def _check_canonical_nodata(rd, _expected_transform):
    assert rd.attrs['nodata'] == _NODATA_SENTINEL


def _check_canonical_extra_tags(rd, _expected_transform):
    """A non-friendly extra_tags entry (Software, 305) round-trips
    intact. The writer must preserve unknown tags so users can attach
    arbitrary metadata."""
    got = rd.attrs['extra_tags']
    # Look up tag 305 specifically; ordering and any reader-added
    # entries are not part of the contract for this assertion.
    by_id = {t[0]: t for t in got}
    assert 305 in by_id, (
        f"Software tag (305) missing from read-back extra_tags: {got}"
    )
    tag_id, type_id, _count, value = by_id[305]
    assert tag_id == 305
    assert type_id == 2  # TIFF ASCII
    assert value == _SOFTWARE_STR


def _check_canonical_gdal_metadata(rd, _expected_transform):
    """The parsed dict survives the round-trip key-by-key. Allow extra
    entries the reader might inject (e.g. ``STATISTICS_*``) so this
    test is not a tripwire for unrelated reader changes."""
    got = rd.attrs['gdal_metadata']
    assert isinstance(got, dict), f"gdal_metadata is not a dict: {got!r}"
    for k, v in _GDAL_META.items():
        assert got.get(k) == v, (
            f"gdal_metadata[{k!r}] mismatch.\n  expected: {v!r}\n"
            f"  got     : {got.get(k)!r}\n  full read-back: {got!r}"
        )


def _check_canonical_gdal_metadata_xml(rd, _expected_transform):
    """The raw XML string is reconstructed by the writer from the
    ``gdal_metadata`` dict. Pin presence + the substring that proves
    our fixture survived; the exact XML formatting is writer-dependent.
    """
    xml = rd.attrs['gdal_metadata_xml']
    assert isinstance(xml, str) and xml.startswith('<GDALMetadata>')
    assert 'xrspatial-1984' in xml, (
        f"gdal_metadata_xml lost the fixture marker: {xml!r}"
    )


def _check_canonical_resolution_group(rd, _expected_transform):
    """``x_resolution`` / ``y_resolution`` / ``resolution_unit`` are
    written and read as one logical unit -- pin them together so a
    writer that drops one but keeps the others fails here."""
    assert rd.attrs['x_resolution'] == pytest.approx(_X_RES)
    assert rd.attrs['y_resolution'] == pytest.approx(_Y_RES)
    assert rd.attrs['resolution_unit'] == _RES_UNIT


def _check_canonical_contract_version(rd, _expected_transform):
    """``_xrspatial_geotiff_contract`` is stamped on every read; pin
    that the canonical fixture sees the current version. Per-backend
    coverage lives in the version section below."""
    assert rd.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION


def _check_canonical_georef_status(rd, _expected_transform):
    """``georef_status`` is canonical from contract v3.
    The fixture sets ``crs`` + axis-aligned transform-from-coords, so
    the round-tripped value is the ``'full'`` literal."""
    assert rd.attrs['georef_status'] == 'full'


# (id_suffix, check_callable). One row per canonical key with a
# value-level assertion. Parametrised so a single failure names the
# offending key in the test ID rather than firing inside a per-key
# helper buried in a generic test.
_CANONICAL_VALUE_CHECKS = [
    ('crs',                _check_canonical_crs),
    ('crs_wkt',            _check_canonical_crs_wkt),
    ('transform',          _check_canonical_transform),
    ('GDAL_NODATA',        _check_canonical_nodata),
    ('extra_tags',         _check_canonical_extra_tags),
    ('gdal_metadata',      _check_canonical_gdal_metadata),
    ('gdal_metadata_xml',  _check_canonical_gdal_metadata_xml),
    ('resolution_group',   _check_canonical_resolution_group),
    ('contract_version',   _check_canonical_contract_version),
    ('georef_status',      _check_canonical_georef_status),
]


@pytest.mark.parametrize(
    'check', [c[1] for c in _CANONICAL_VALUE_CHECKS],
    ids=[f'canonical[{c[0]}]' for c in _CANONICAL_VALUE_CHECKS],
)
def test_canonical_value_roundtrip(canonical_roundtrip, check):
    """Per-key value-equality check after a write -> read round-trip."""
    rd, expected_transform = canonical_roundtrip
    check(rd, expected_transform)


# Per-backend coverage for canonical-key *presence*.
#
# Read-time backends share ``_populate_attrs_from_geo_info``, so per-key
# value round-trips are pinned once (above) on the eager numpy path
# only -- value equality on the canonical fixture (which carries
# ``extra_tags`` + the experimental-codec opt-in) is exercised by the
# eager parametrize case at ``canonical[contract_version]``. The
# per-backend loop below guards against a backend that bypasses the
# shared attrs helper, not against a backend that emits the wrong
# value.


def _open_eager(path):
    return open_geotiff(path)


def _open_dask(path):
    return open_geotiff(path, chunks=2)


def _open_gpu(path):
    return open_geotiff(path, gpu=True)


def _open_dask_gpu(path):
    return open_geotiff(path, gpu=True, chunks=2)


# Each entry: (opener, label). The label is the suffix used in the
# parametrize id AND in the tmp_path filename, so a failing case names
# its backend in both places without leaking the leading-underscore
# function name into the filename.
_BACKEND_OPENERS = [
    pytest.param(_open_eager, 'eager-numpy', id='canonical[eager-numpy]'),
    pytest.param(_open_dask, 'dask-numpy', id='canonical[dask-numpy]'),
    pytest.param(_open_gpu, 'gpu', id='canonical[gpu]', marks=requires_gpu),
    pytest.param(
        _open_dask_gpu, 'dask-gpu',
        id='canonical[dask-gpu]', marks=requires_gpu),
]


@pytest.mark.parametrize('opener,label', _BACKEND_OPENERS)
def test_canonical_keys_present_per_backend(tmp_path, opener, label):
    """Each read backend emits the full canonical key set.

    Writes the canonical fixture once with the eager writer (only the
    eager + dask writers exist; the GPU writer is exercised in its own
    parity tests), then re-reads through every supported backend and
    asserts presence. Value-level round-trip is checked by the per-key
    cases above; this loop guards against a backend that bypasses
    ``_populate_attrs_from_geo_info``.
    """
    da, _ = _make_canonical_da()
    path = str(tmp_path / f'canonical_{label}.tif')
    # Fresh DataArray with ``extra_tags`` -> rich-tag opt-in required
    # (see ``canonical_roundtrip`` fixture for details).
    to_geotiff(da, path, allow_experimental_codecs=True)

    rd = opener(path)
    missing = sorted(k for k in _CANONICAL_KEYS if k not in rd.attrs)
    assert missing == [], (
        f"{label}: canonical attrs missing after round-trip: "
        f"{missing}. attrs keys present: {sorted(rd.attrs.keys())}"
    )


def _make_gdal_metadata_da():
    """Fresh DataArray (no contract marker) carrying only a dict
    ``gdal_metadata``. The writer turns this into on-disk GDAL XML, so it
    must trip the experimental rich-tag gate (#3320)."""
    data = np.arange(4, dtype=np.float32).reshape(2, 2)
    return xr.DataArray(
        data, dims=('y', 'x'),
        coords={'y': [240.0, 230.0], 'x': [100.0, 110.0]},
        attrs={'crs': 4326, 'gdal_metadata': {'AREA_OR_POINT': 'Area'}},
    )


def test_gdal_metadata_dict_requires_optin_3320(tmp_path):
    """A fresh DataArray whose only rich-tag attr is a dict
    ``gdal_metadata`` writes GDAL XML to disk, so ``to_geotiff`` must
    reject it without ``allow_experimental_codecs`` and accept it with
    the flag (#3320)."""
    da = _make_gdal_metadata_da()
    rejected = str(tmp_path / 'gdal_metadata_optin_3320_rejected.tif')
    with pytest.raises(ValueError, match=r"attrs\['gdal_metadata'\]"):
        to_geotiff(da, rejected)
    assert not os.path.exists(rejected)

    accepted = str(tmp_path / 'gdal_metadata_optin_3320_accepted.tif')
    to_geotiff(da, accepted, allow_experimental_codecs=True)
    assert os.path.exists(accepted)


def test_gdal_metadata_dict_roundtrip_writes_flag_free_3320(tmp_path):
    """A read-back DataArray carries the contract marker, so writing its
    ``gdal_metadata`` back is the canonical round-trip and stays
    flag-free even without the opt-in (#3320)."""
    da = _make_gdal_metadata_da()
    src = str(tmp_path / 'gdal_metadata_roundtrip_3320_src.tif')
    to_geotiff(da, src, allow_experimental_codecs=True)

    rd = open_geotiff(src)
    assert '_xrspatial_geotiff_contract' in rd.attrs
    out = str(tmp_path / 'gdal_metadata_roundtrip_3320_out.tif')
    to_geotiff(rd, out)
    assert os.path.exists(out)


# ``raster_type`` lives outside the shared fixture because the canonical
# default ('area') is encoded as *absence* in attrs. The two branches
# need different fixtures.


def test_canonical_raster_type_area_omitted_on_roundtrip(tmp_path):
    """RasterPixelIsArea is the implicit default and is encoded as
    *absence* of ``attrs['raster_type']``. A DataArray with no
    ``raster_type`` attr round-trips to a DataArray that still has no
    ``raster_type`` attr."""
    da, _ = _make_canonical_da()
    assert 'raster_type' not in da.attrs
    path = str(tmp_path / 'raster_type_area.tif')
    to_geotiff(da, path, allow_experimental_codecs=True)
    rd = open_geotiff(path)
    assert 'raster_type' not in rd.attrs, (
        f"area is the implicit default but the reader emitted "
        f"raster_type={rd.attrs['raster_type']!r}"
    )


def test_canonical_raster_type_point_roundtrip(tmp_path):
    """``raster_type='point'`` is the only value the writer accepts via
    attrs; the reader emits it back on a round-trip."""
    da, _ = _make_canonical_da()
    da.attrs['raster_type'] = 'point'
    path = str(tmp_path / 'raster_type_point.tif')
    to_geotiff(da, path, allow_experimental_codecs=True)
    rd = open_geotiff(path)
    assert rd.attrs.get('raster_type') == 'point'


# ===========================================================================
# Aliases tier
#
# Compatibility aliases (rioxarray ``nodatavals``, CF ``_FillValue``)
# are honoured on read when canonical ``nodata`` is absent. The write
# side emits only ``nodata``, never the aliases.
# ===========================================================================

# Arbitrary float-castable sentinel distinct from any data value below.
# The value itself does not matter; the tests assert it survives the
# alias resolver and the round-trip.
_ALIAS_SENTINEL = -9999.0


def _da_float(arr, **attrs):
    return xr.DataArray(
        arr, dims=["y", "x"],
        coords={"y": np.arange(arr.shape[0], dtype=np.float64),
                "x": np.arange(arr.shape[1], dtype=np.float64)},
        attrs=attrs,
    )


@pytest.fixture
def _arr_with_sentinel():
    return np.array(
        [[1.0, 2.0, _ALIAS_SENTINEL], [3.0, _ALIAS_SENTINEL, 5.0]],
        dtype=np.float32,
    )


# (id_suffix, alias_key). The two read-side aliases the resolver
# accepts when ``attrs['nodata']`` is absent.
_READ_ALIASES = [
    ('nodatavals->nodata', 'nodatavals'),
    ('_FillValue->nodata', '_FillValue'),
]


@pytest.mark.parametrize(
    'alias_key', [a[1] for a in _READ_ALIASES],
    ids=[f'alias[{a[0]}]' for a in _READ_ALIASES],
)
def test_alias_read_used_when_nodata_absent(
        tmp_path, _arr_with_sentinel, alias_key):
    """Pin: a foreign DataArray carrying only an alias round-trips
    through ``to_geotiff`` with the sentinel preserved.

    Covers both rioxarray ``nodatavals`` (tuple of one) and CF
    ``_FillValue`` (scalar) in one parametrised case.
    """
    if alias_key == 'nodatavals':
        alias_value = (_ALIAS_SENTINEL,)
    else:
        alias_value = _ALIAS_SENTINEL
    da = _da_float(_arr_with_sentinel, crs=4326, **{alias_key: alias_value})
    assert "nodata" not in da.attrs  # precondition: only the alias is set
    out = str(tmp_path / f"alias_{alias_key}.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _ALIAS_SENTINEL


def test_alias_canonical_nodata_wins_at_resolver(_arr_with_sentinel):
    """``_resolve_nodata_attr`` (the read-side chokepoint) still prefers
    canonical ``nodata`` over both aliases. This is the resolver-layer
    contract that downstream code consumes; the write-side fail-closed
    check is exercised below in
    ``test_alias_canonical_nodata_wins_at_write``."""
    canonical = -8888.0
    da = _da_float(
        _arr_with_sentinel, crs=4326,
        nodata=canonical,
        nodatavals=(_ALIAS_SENTINEL,),
        **{"_FillValue": -7777.0},
    )
    assert _resolve_nodata_attr(dict(da.attrs)) == canonical


def test_alias_canonical_nodata_wins_at_write(
        tmp_path, _arr_with_sentinel):
    """Pin: the legacy "canonical nodata silently wins on write" was
    replaced by a fail-closed raise. A DataArray with disagreeing
    ``nodata`` and ``nodatavals`` attrs refuses to write
    instead of dropping the alias. The opt-out is the explicit
    ``nodata=`` kwarg, which overrides both attrs."""
    da = _da_float(
        _arr_with_sentinel, crs=4326,
        nodata=-8888.0,
        nodatavals=(_ALIAS_SENTINEL,),
        **{"_FillValue": -7777.0},
    )

    out = str(tmp_path / "canonical_wins.tif")
    with pytest.raises(ConflictingNodataError):
        to_geotiff(da, out)

    # Explicit ``nodata=`` bypasses the check.
    to_geotiff(da, out, nodata=-8888.0)
    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == -8888.0


def test_alias_write_does_not_emit_aliases_when_canonical_present(
        tmp_path, _arr_with_sentinel):
    """Pin: round-tripping a DataArray that has only ``attrs['nodata']``
    set returns attrs containing ``nodata`` and not the two aliases.

    Reader paths emit a single canonical key; this guards against a
    future change that copies the resolved value into the rioxarray or
    CF alias slot as well (which would silently double-write the
    sentinel and confuse downstream consumers that special-case one
    alias over another)."""
    da = _da_float(_arr_with_sentinel, crs=4326, nodata=_ALIAS_SENTINEL)
    out = str(tmp_path / "canonical_only.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _ALIAS_SENTINEL
    assert "nodatavals" not in rd.attrs, (
        f"reader emitted rioxarray alias alongside canonical nodata: "
        f"attrs={sorted(rd.attrs)}"
    )
    assert "_FillValue" not in rd.attrs, (
        f"reader emitted CF alias alongside canonical nodata: "
        f"attrs={sorted(rd.attrs)}"
    )


def test_alias_nan_in_nodatavals_is_skipped():
    """Pin: NaN entries in ``attrs['nodatavals']`` are not a sentinel.

    NaN means the float NaN is already the implicit sentinel, which
    doesn't need a GDAL_NODATA tag. The resolver skips NaN entries and
    (in a mixed tuple) returns the first concrete numeric value."""
    # Pure-NaN tuple resolves to None.
    assert _resolve_nodata_attr({"nodatavals": (float("nan"),)}) is None
    # NaN followed by a concrete sentinel: the resolver skips NaN and
    # returns the concrete value.
    assert _resolve_nodata_attr(
        {"nodatavals": (float("nan"), _ALIAS_SENTINEL)}) == _ALIAS_SENTINEL
    # None entries are also skipped (rioxarray uses None for "no
    # sentinel on this band").
    assert _resolve_nodata_attr(
        {"nodatavals": (None, _ALIAS_SENTINEL)}) == _ALIAS_SENTINEL


# ===========================================================================
# Passthrough tier
#
# Writer does not consume; reader rebuilds the value from a TIFF tag
# if one survived.
# ===========================================================================

# Full set of pass-through keys defined by the contract. Contract v2
# trimmed this set to the three TIFF-tag-derived keys that actually
# round-trip via ``_merge_friendly_extra_tags``.
_ALL_PASSTHROUGH_KEYS = (
    'image_description',
    'extra_samples',
    'colormap',
)


# Attrs that the reader emitted under a ``DeprecationWarning`` in
# contract v1 and that contract v2 removed entirely.
_REMOVED_IN_V2_ATTRS = (
    'crs_name',
    'geog_citation',
    'datum_code',
    'angular_units',
    'linear_units',
    'semi_major_axis',
    'inv_flattening',
    'projection_code',
    'vertical_crs',
    'vertical_citation',
    'vertical_units',
    'colormap_rgba',
    'cmap',
)


def _make_passthrough_da(crs=None, attrs=None, shape=(4, 4), dtype=np.float32):
    """Build a minimal georeferenced DataArray with optional CRS and attrs."""
    data = np.ones(shape, dtype=dtype)
    h, w = shape
    coords = {
        'y': np.arange(h, 0, -1, dtype=np.float64),
        'x': np.arange(w, dtype=np.float64),
    }
    a = dict(attrs) if attrs else {}
    if crs is not None:
        a['crs'] = crs
    return xr.DataArray(data, dims=('y', 'x'), coords=coords, attrs=a)


def _passthrough_roundtrip(tmp_path, da, name='roundtrip.tif'):
    """Write ``da`` to ``tmp_path/name`` and read it back."""
    path = str(tmp_path / name)
    to_geotiff(da, path)
    return open_geotiff(path)


# (key, crs_to_use, value_set_on_write_or_None, expected_outcome)
#
# ``crs_to_use``: the CRS attached to the test DataArray; 4326 is fine
# for every remaining key.
# ``value_set_on_write``: the value the test sets on ``da.attrs``
# before write.
# ``expected``: ``'reconstructible'`` (key must be present in the
# read-back attrs and equal to the written value) or ``'dropped'`` (key
# must be absent). After contract v2, every row carries
# ``'reconstructible'``; the ``'dropped'`` arm is kept so a future
# addition can use it without restructuring the test.
_PASSTHROUGH_CASES = [
    # Non-GeoKey tag passthroughs. The writer folds these into
    # extra_tags via _merge_friendly_extra_tags, so the reader can
    # rebuild them.
    ('image_description', 4326, 'pr1984 fixture', 'reconstructible'),
    ('extra_samples',     4326, (1,),             'reconstructible'),
    # ``colormap`` round-trips as the raw uint16 triple list. The TIFF
    # ColorMap tag (320) stores RGB triples as uint16 values in the
    # 0-65535 range. Values below are written as-is and compared by
    # equality after the round-trip; if the writer ever rescales 8-bit
    # input to 16-bit (or vice versa), update this fixture rather than
    # the contract.
    (
        'colormap', 4326,
        tuple([0] * 256 + [128] * 256 + [255] * 256),
        'reconstructible',
    ),
]


def test_passthrough_cases_cover_all_keys():
    """``_PASSTHROUGH_CASES`` and ``_ALL_PASSTHROUGH_KEYS`` carry the
    same set in two forms. Pin them so a key added to one list and
    forgotten on the other fails here rather than silently skipping
    coverage in ``test_passthrough_dropped_when_no_crs``."""
    case_keys = {c[0] for c in _PASSTHROUGH_CASES}
    assert case_keys == set(_ALL_PASSTHROUGH_KEYS), (
        f"_PASSTHROUGH_CASES and _ALL_PASSTHROUGH_KEYS diverge.\n"
        f"  only in cases: {sorted(case_keys - set(_ALL_PASSTHROUGH_KEYS))}\n"
        f"  only in keys : {sorted(set(_ALL_PASSTHROUGH_KEYS) - case_keys)}"
    )


@pytest.mark.parametrize(
    'key,crs,value,expected',
    _PASSTHROUGH_CASES,
    ids=[f'passthrough[{c[0]}]' for c in _PASSTHROUGH_CASES],
)
def test_passthrough_key_roundtrip(tmp_path, key, crs, value, expected):
    """Lock per-key round-trip outcome for the pass-through attr tier."""
    attrs = {}
    if value is not None:
        attrs[key] = value
    da = _make_passthrough_da(crs=crs, attrs=attrs)
    # Single-band uint8 needed for the colormap tag to be valid in
    # TIFF.
    if key == 'colormap':
        da = _make_passthrough_da(crs=crs, attrs=attrs, dtype=np.uint8)

    rd = _passthrough_roundtrip(tmp_path, da, name=f'{key}.tif')

    if expected == 'reconstructible':
        assert key in rd.attrs, (
            f"pass-through key {key!r} was expected to round-trip but is "
            f"absent. attrs keys present: {sorted(rd.attrs.keys())}"
        )
        if value is not None:
            got = rd.attrs[key]
            if isinstance(value, tuple):
                assert tuple(got) == value, (
                    f"{key!r}: value mismatch on round-trip\n"
                    f"  written: {value}\n"
                    f"  read   : {got}"
                )
            else:
                assert got == value, (
                    f"{key!r}: value mismatch on round-trip\n"
                    f"  written: {value!r}\n"
                    f"  read   : {got!r}"
                )
    else:  # 'dropped'
        assert key not in rd.attrs, (
            f"pass-through key {key!r} was expected to drop on round-trip "
            f"(writer does not emit a tag the reader can rebuild it from) "
            f"but it is present with value {rd.attrs[key]!r}. If a writer "
            f"change started emitting this key, decide whether to promote "
            f"the key to canonical (issue #1984) and update this test."
        )


def test_passthrough_dropped_when_no_crs(tmp_path):
    """Files without a CRS do not surface any pass-through attrs."""
    da = _make_passthrough_da(crs=None)
    rd = _passthrough_roundtrip(tmp_path, da, name='no_crs.tif')

    present = sorted(k for k in _ALL_PASSTHROUGH_KEYS if k in rd.attrs)
    assert present == [], (
        f"pass-through keys leaked into a no-CRS round-trip: {present}. "
        f"All keys present: {sorted(rd.attrs.keys())}"
    )
    # Sanity: no CRS attrs either.
    assert 'crs' not in rd.attrs
    assert 'crs_wkt' not in rd.attrs


def test_passthrough_does_not_promote_to_canonical(tmp_path):
    """Setting legacy GeoKey-derived attrs without a CRS does not
    inject one.

    Contract v2 removed these keys from the reader's emission set, but
    a user with a hand-built ``DataArray`` may still
    set them. The writer treats them as advisory and never synthesises
    a CRS from them; this test pins that invariant.
    """
    # Mix of GeoKey-derived keys, but no ``crs`` / ``crs_wkt``. If the
    # writer ever started inferring a CRS from these (e.g. picking 4326
    # because angular_units == 'degree') this test would fail.
    attrs = {
        'crs_name': 'WGS 84',
        'geog_citation': 'WGS 84',
        'angular_units': 'degree',
        'linear_units': 'metre',
        'semi_major_axis': 6378137.0,
        'inv_flattening': 298.257223563,
        'datum_code': 6326,
    }
    da = _make_passthrough_da(crs=None, attrs=attrs)

    with warnings.catch_warnings():
        # The writer warns on user-defined-CRS WKT writes; we are not
        # on that path here, but suppress generously so a future
        # warning tweak does not turn this test into a warning
        # regression test.
        warnings.simplefilter('ignore')
        rd = _passthrough_roundtrip(
            tmp_path, da, name='no_crs_with_attrs.tif')

    assert 'crs' not in rd.attrs, (
        f"pass-through attrs caused the writer to synthesise a CRS: "
        f"crs={rd.attrs.get('crs')!r}. The contract says pass-through "
        f"attrs are advisory only; the writer must rely on attrs['crs'] "
        f"or attrs['crs_wkt'] to emit georeferencing."
    )
    assert 'crs_wkt' not in rd.attrs, (
        f"pass-through attrs caused the writer to synthesise a CRS WKT: "
        f"crs_wkt={rd.attrs.get('crs_wkt')!r}."
    )


def test_passthrough_removed_attrs_not_emitted(tmp_path):
    """Contract v2 removed 13 deprecated reader attrs.

    A freshly read DataArray does not carry any of them, even when the
    underlying GeoTIFF's GeoKey directory advertises the values. Pins
    the removal so a regression that re-adds an emit site fails here
    rather than silently leaking the attr back into the public surface.
    """
    da = _make_passthrough_da(crs=4326)
    rd = _passthrough_roundtrip(
        tmp_path, da, name='removed_attrs_no_emit.tif')

    leaked = sorted(k for k in _REMOVED_IN_V2_ATTRS if k in rd.attrs)
    assert leaked == [], (
        f"contract v2 attrs leaked into a fresh read: {leaked}. "
        f"All attrs keys present: {sorted(rd.attrs.keys())}. Issue "
        f"#2016 dropped these from the reader; re-emitting them is a "
        f"behaviour regression."
    )


def test_passthrough_removed_attrs_absent_after_roundtrip(tmp_path):
    """A write -> read cycle does not resurrect any v2-removed attr.

    Even when the input ``DataArray`` carries every removed attr as a
    legacy value, the writer ignores them and the reader never adds
    them back. The reopened DataArray's attrs is the public surface
    callers rely on.
    """
    legacy_payload = {
        'crs_name': 'WGS 84',
        'geog_citation': 'WGS 84',
        'datum_code': 6326,
        'angular_units': 'degree',
        'linear_units': 'metre',
        'semi_major_axis': 6378137.0,
        'inv_flattening': 298.257223563,
        'projection_code': 16033,
        'vertical_crs': 5703,
        'vertical_citation': 'NAVD88',
        'vertical_units': 'metre',
        'colormap_rgba': ((1.0, 0.0, 0.0, 1.0),),
        'cmap': 'tiff_palette_placeholder',
    }
    da = _make_passthrough_da(crs=4326, attrs=legacy_payload)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rd = _passthrough_roundtrip(
            tmp_path, da, name='removed_attrs_roundtrip.tif')

    resurrected = sorted(k for k in _REMOVED_IN_V2_ATTRS if k in rd.attrs)
    assert resurrected == [], (
        f"removed attrs survived a write -> read cycle: {resurrected}. "
        f"All attrs keys present: {sorted(rd.attrs.keys())}. The "
        f"reader drops these per contract v2 (issue #2016)."
    )


# Note: the ``passthrough[contract_version]`` stamp check that the old
# source file held is intentionally dropped here. The canonical fixture
# already exercises a stamped round-trip
# (``test_canonical_value_roundtrip[canonical[contract_version]]``) and
# the version section below owns per-backend stamp coverage.


# ===========================================================================
# Version tier
#
# Per-backend coverage of the ``_xrspatial_geotiff_contract`` stamp.
# ===========================================================================


def _write_small_tiff(path):
    """Write a small tiled float32 TIFF used by every read-path
    assertion."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        tile=(32, 32), compression='deflate', metadata=None,
    )
    return arr


def _write_minimal_vrt(vrt_path, source_name, *, height, width):
    """Write a VRT that references ``source_name`` as a single-band
    source."""
    vrt_path.write_text(
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_name}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_version_constant_is_current():
    """Pin the integer value so a careless bump shows up here first.

    Contract v3 added ``attrs['georef_status']`` to the canonical tier.
    Contract v4 added ``attrs['rotated_affine']`` for the
    ``allow_rotated=True`` opt-in path. Contract v5 added
    ``attrs['mask_and_scale_dtype']`` (the integer source dtype recorded
    on a ``mask_and_scale`` read, for ``to_geotiff(pack=True)``). Bumping
    past 5 should be paired with a docs update and a sibling test for the
    new key.
    """
    assert _ATTRS_CONTRACT_VERSION == 5


def test_version_module_docstring_matches_constant():
    """Guard against the docstring and the constant drifting apart.

    The ``_attrs.py`` module docstring spells out the current contract
    version inline (``The contract version is recorded in
    ``attrs['_xrspatial_geotiff_contract']`` (currently ``<N>``)``).
    A previous v3 -> v4 bump for the ``rotated_affine`` attr updated the
    constant but left the docstring at ``3``. This test parses the
    documented number out of the docstring and asserts
    it equals ``_ATTRS_CONTRACT_VERSION`` so the next drift gets caught
    in CI rather than in code review.
    """
    docstring = _attrs_module.__doc__
    assert docstring is not None, (
        "xrspatial.geotiff._attrs lost its module docstring; the contract "
        "documentation lives in that docstring and must be restored."
    )

    # Match the canonical phrasing
    # ``... ``attrs['_xrspatial_geotiff_contract']`` (currently
    # ``<N>``)`` while staying tolerant of trivial whitespace changes
    # around the parenthetical.
    match = re.search(
        r"attrs\['_xrspatial_geotiff_contract'\]``\s*\(currently\s*``(\d+)``\)",
        docstring,
    )
    assert match is not None, (
        "Could not find the documented contract version in the "
        "_attrs.py module docstring. Expected a phrase of the form "
        "``attrs['_xrspatial_geotiff_contract']`` (currently ``<N>``). "
        "Update the docstring or this test if the phrasing changed."
    )

    documented_version = int(match.group(1))
    assert documented_version == _ATTRS_CONTRACT_VERSION, (
        f"_attrs.py module docstring says the contract version is "
        f"{documented_version}, but _ATTRS_CONTRACT_VERSION is "
        f"{_ATTRS_CONTRACT_VERSION}. Update the docstring "
        f"'(currently ``{documented_version}``)' to "
        f"'(currently ``{_ATTRS_CONTRACT_VERSION}``)' so the two stay "
        f"in lockstep."
    )


# (id_suffix, opener_callable). Each opener takes a path and returns a
# DataArray; the stamp assertion is shared.
def _v_eager(path):
    return open_geotiff(path)


def _v_dask(path):
    return open_geotiff(path, chunks=32)


def _v_gpu(path):
    return open_geotiff(path, gpu=True)


def _v_dask_gpu(path):
    return open_geotiff(path, gpu=True, chunks=32)


# Each entry: (opener, label). Same shape as ``_BACKEND_OPENERS`` above
# so the filename matches the parametrize id.
_VERSION_TIFF_OPENERS = [
    pytest.param(_v_eager, 'eager-numpy', id='version[eager-numpy]'),
    pytest.param(_v_dask, 'dask-numpy', id='version[dask-numpy]'),
    pytest.param(_v_gpu, 'gpu', id='version[gpu]', marks=requires_gpu),
    pytest.param(
        _v_dask_gpu, 'dask-gpu',
        id='version[dask-gpu]', marks=requires_gpu),
]


@pytest.mark.parametrize('opener,label', _VERSION_TIFF_OPENERS)
def test_version_stamp_present_per_tiff_backend(tmp_path, opener, label):
    """Each TIFF read backend stamps the contract version."""
    path = str(tmp_path / f"contract_{label}.tif")
    _write_small_tiff(path)

    da = opener(path)
    assert da.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION


def _v_vrt_eager(path):
    return _read_vrt(path)


def _v_vrt_chunked(path):
    return _read_vrt(path, chunks=32)


_VERSION_VRT_OPENERS = [
    pytest.param(_v_vrt_eager, 'vrt-eager', id='version[vrt-eager]'),
    pytest.param(_v_vrt_chunked, 'vrt-chunked', id='version[vrt-chunked]'),
]


@pytest.mark.parametrize('opener,label', _VERSION_VRT_OPENERS)
def test_version_stamp_present_per_vrt_backend(tmp_path, opener, label):
    """Both VRT read paths stamp the contract version."""
    src = tmp_path / f"contract_{label}_source.tif"
    _write_small_tiff(str(src))
    vrt = tmp_path / f"contract_{label}.vrt"
    _write_minimal_vrt(vrt, os.path.basename(src), height=64, width=64)

    da = opener(str(vrt))
    assert da.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION
