"""Locking test for the best-effort pass-through tier of the attrs contract.

Issue #1984, PR 6 of 7.

The attrs contract has three tiers:

1. Canonical: writers consume the attr; round-trip is guaranteed.
2. Pass-through: writers do not consume the attr. The reader rebuilds
   the value from the GeoKey directory (or another TIFF tag) on read.
   Round-trip is best-effort: it works only when the writer happens to
   emit a tag the reader can rebuild the attr from.
3. Ignored: writer never touches; attr is dropped silently.

This file pins the *current* behaviour of every key in the pass-through
tier so future writer changes have to decide whether to promote a key to
canonical or to keep it dropping. The split between "reconstructible"
and "dropped on round-trip" is captured in the parametrisation below and
mirrored in PR 6's body so the next PR (canonical promotion) has a
shopping list.

Current state of every pass-through key, as locked here:

* Reconstructible (writer puts a TIFF tag the reader rebuilds from):
  - ``image_description``  -> tag 270 (ImageDescription)
  - ``extra_samples``      -> tag 338 (ExtraSamples)
  - ``colormap``           -> tag 320 (ColorMap, raw uint16 triples)

Contract v2 (issue #2016) removed the 13 GeoKey-derived and
matplotlib-colormap keys that v1 emitted on read under a
``DeprecationWarning`` (``crs_name``, ``geog_citation``,
``datum_code``, ``angular_units``, ``linear_units``,
``semi_major_axis``, ``inv_flattening``, ``projection_code``,
``vertical_crs``, ``vertical_citation``, ``vertical_units``,
``colormap_rgba``, ``cmap``). The reader no longer surfaces them as
attrs; ``test_removed_attrs_not_emitted`` and
``test_removed_attrs_absent_after_roundtrip`` lock that absence.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


# Full set of pass-through keys defined by the contract. Contract v2
# (issue #2016) trimmed this set to the three TIFF-tag-derived keys
# that actually round-trip via ``_merge_friendly_extra_tags``.
_ALL_PASSTHROUGH_KEYS = (
    'image_description',
    'extra_samples',
    'colormap',
)


# Attrs that the reader emitted under a ``DeprecationWarning`` in
# contract v1 and that contract v2 (issue #2016) removed entirely.
# ``test_removed_attrs_not_emitted`` pins their absence after a fresh
# read; ``test_removed_attrs_absent_after_roundtrip`` pins their
# absence after a write -> read cycle.
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


def _make_da(crs=None, attrs=None, shape=(4, 4), dtype=np.float32):
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


def _roundtrip(tmp_path, da, name='roundtrip.tif'):
    """Write ``da`` to ``tmp_path/name`` and read it back."""
    path = str(tmp_path / name)
    to_geotiff(da, path)
    return open_geotiff(path)


# (key, crs_to_use, value_set_on_write_or_None, expected_outcome)
#
# ``crs_to_use``: the CRS attached to the test DataArray; 4326 is
# fine for every remaining key.
#
# ``value_set_on_write``: the value the test sets on ``da.attrs`` before
# write.
#
# ``expected``: ``'reconstructible'`` (key must be present in the
# read-back attrs and equal to the written value) or ``'dropped'``
# (key must be absent). After contract v2 (issue #2016), every
# row carries ``'reconstructible'``; the ``'dropped'`` arm is kept
# so a future addition can use it without restructuring the test.
_PASSTHROUGH_CASES = [
    # Non-GeoKey tag passthroughs. The writer folds these into extra_tags
    # via _merge_friendly_extra_tags, so the reader can rebuild them.
    ('image_description', 4326,  'pr1984 fixture',   'reconstructible'),
    ('extra_samples',     4326,  (1,),               'reconstructible'),
    # ``colormap`` round-trips as the raw uint16 triple list.
    # The TIFF ColorMap tag (320) stores RGB triples as uint16 values in
    # the 0-65535 range. Values below are written as-is and compared
    # by-equality after the round-trip; if the writer ever rescales 8-bit
    # input to 16-bit (or vice versa), update this fixture rather than
    # the contract.
    ('colormap',          4326,  tuple([0] * 256 + [128] * 256 + [255] * 256),
                                                    'reconstructible'),
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
    ids=[c[0] for c in _PASSTHROUGH_CASES],
)
def test_passthrough_key_roundtrip(tmp_path, key, crs, value, expected):
    """Lock per-key round-trip outcome for the pass-through attr tier."""
    attrs = {}
    if value is not None:
        attrs[key] = value
    da = _make_da(crs=crs, attrs=attrs)
    # Single-band uint8 needed for the colormap tag to be valid in TIFF.
    if key == 'colormap':
        da = _make_da(crs=crs, attrs=attrs, dtype=np.uint8)

    rd = _roundtrip(tmp_path, da, name=f'{key}.tif')

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
    da = _make_da(crs=None)
    rd = _roundtrip(tmp_path, da, name='no_crs.tif')

    present = sorted(k for k in _ALL_PASSTHROUGH_KEYS if k in rd.attrs)
    assert present == [], (
        f"pass-through keys leaked into a no-CRS round-trip: {present}. "
        f"All keys present: {sorted(rd.attrs.keys())}"
    )
    # Sanity: no CRS attrs either.
    assert 'crs' not in rd.attrs
    assert 'crs_wkt' not in rd.attrs


def test_passthrough_does_not_promote_to_canonical(tmp_path):
    """Setting legacy GeoKey-derived attrs without a CRS must not inject one.

    Contract v2 (issue #2016) removed these keys from the reader's
    emission set, but a user with a hand-built ``DataArray`` may still
    set them. The writer must treat them as advisory and never synthesise
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
    da = _make_da(crs=None, attrs=attrs)

    with warnings.catch_warnings():
        # The writer warns on user-defined-CRS WKT writes; we are not on
        # that path here, but suppress generously so a future warning
        # tweak does not turn this test into a warning regression test.
        warnings.simplefilter('ignore')
        rd = _roundtrip(tmp_path, da, name='no_crs_with_attrs.tif')

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


def test_removed_attrs_not_emitted(tmp_path):
    """Contract v2 (issue #2016) removed 13 deprecated reader attrs.

    A freshly read DataArray must not carry any of them, even when the
    underlying GeoTIFF's GeoKey directory advertises the values. This
    test pins the removal so a regression that re-adds an emit site
    fails here rather than silently leaking the attr back into the
    public surface.
    """
    da = _make_da(crs=4326)
    rd = _roundtrip(tmp_path, da, name='removed_attrs_no_emit.tif')

    leaked = sorted(k for k in _REMOVED_IN_V2_ATTRS if k in rd.attrs)
    assert leaked == [], (
        f"contract v2 attrs leaked into a fresh read: {leaked}. "
        f"All attrs keys present: {sorted(rd.attrs.keys())}. Issue "
        f"#2016 dropped these from the reader; re-emitting them is a "
        f"behaviour regression."
    )


def test_removed_attrs_absent_after_roundtrip(tmp_path):
    """A write -> read cycle must not resurrect any v2-removed attr.

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
    da = _make_da(crs=4326, attrs=legacy_payload)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rd = _roundtrip(tmp_path, da, name='removed_attrs_roundtrip.tif')

    resurrected = sorted(k for k in _REMOVED_IN_V2_ATTRS if k in rd.attrs)
    assert resurrected == [], (
        f"removed attrs survived a write -> read cycle: {resurrected}. "
        f"All attrs keys present: {sorted(rd.attrs.keys())}. The "
        f"reader must drop these per contract v2 (issue #2016)."
    )


def test_contract_version_is_two(tmp_path):
    """``attrs['_xrspatial_geotiff_contract']`` is ``2`` on every read.

    The contract version is the user-visible signal that the removal
    landed. Downstream code branching on the integer needs the bump
    to fire here on every read path.
    """
    da = _make_da(crs=4326)
    rd = _roundtrip(tmp_path, da, name='contract_v2_signal.tif')

    assert rd.attrs.get('_xrspatial_geotiff_contract') == 2, (
        f"contract version stamp on a fresh read is "
        f"{rd.attrs.get('_xrspatial_geotiff_contract')!r}; issue "
        f"#2016 bumped it to 2."
    )
