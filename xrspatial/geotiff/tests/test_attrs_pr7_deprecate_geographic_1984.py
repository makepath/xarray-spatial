"""Deprecation-warning tests for the geographic-CRS GeoKey attrs.

Issue #1984, PR 7 of 7.

The six geographic-CRS GeoKey-derived attrs listed below were
documented in the contract as best-effort pass-through, but the
locking test in ``test_attrs_contract_passthrough_1984.py`` (issue
#1984, PR 6, merged as #2004) showed they never round-trip: the
writer's ``build_geo_tags`` only emits the primary
``GEOKEY_GEOGRAPHIC_TYPE`` plus citation, so the secondary GeoKeys
these attrs come from are never written.

PR 7 keeps emitting the attrs on read for one release cycle so
callers can migrate, but each emission now fires a
``DeprecationWarning``. This file pins that warning behaviour:

* One ``test_warns_<attr>`` per attr asserts a ``DeprecationWarning``
  with the canonical wording fires when ``_populate_attrs_from_geo_info``
  sees the matching ``GeoInfo`` field set.
* ``test_emission_still_present`` asserts the attr value still lands
  in ``attrs`` (i.e. PR 7 is warning-only; removal is a later PR).

The test drives ``_populate_attrs_from_geo_info`` directly with a
synthetic :class:`GeoInfo`. That bypasses ``open_geotiff`` and the
writer, both of which are irrelevant here: the contract change is on
the read-side attrs population step, not on the on-disk GeoKey set.
"""
from __future__ import annotations

import warnings

import pytest

from xrspatial.geotiff._attrs import (
    _DEPRECATED_GEOGRAPHIC_GEOKEY_ATTRS,
    _deprecated_geographic_geokey_warning,
    _populate_attrs_from_geo_info,
)
from xrspatial.geotiff._geotags import GeoInfo


# (attr_name, sample value). The attr name doubles as the ``GeoInfo``
# field name: every deprecated attr in this PR is stored on
# :class:`GeoInfo` under the same identifier it lands in ``attrs``
# under, so a single column is enough.
#
# Sample values mirror what the GeoTIFF spec would put in each
# secondary GeoKey for WGS 84 (EPSG:4326). The exact values do not
# matter for the warning assertion, but using realistic ones keeps the
# test useful as documentation.
_DEPRECATED_CASES = [
    ('crs_name',        'WGS 84'),
    ('geog_citation',   'WGS 84'),
    ('datum_code',      6326),
    ('angular_units',   'degree'),
    ('semi_major_axis', 6378137.0),
    ('inv_flattening',  298.257223563),
]


def _geo_info_with(**fields) -> GeoInfo:
    """Build a minimal :class:`GeoInfo` with only the given fields set.

    A bare ``GeoInfo()`` has every optional field at ``None`` so the
    other emission branches in ``_populate_attrs_from_geo_info`` stay
    quiet. Only the field under test is populated, which keeps the
    warning under test the only ``DeprecationWarning`` raised.
    """
    info = GeoInfo()
    for name, value in fields.items():
        setattr(info, name, value)
    return info


def test_deprecated_cases_cover_all_attrs():
    """The parametrised case list must enumerate every attr listed in
    ``_DEPRECATED_GEOGRAPHIC_GEOKEY_ATTRS``. A drift here would silently
    drop coverage for whichever attr was forgotten."""
    case_attrs = {c[0] for c in _DEPRECATED_CASES}
    assert case_attrs == set(_DEPRECATED_GEOGRAPHIC_GEOKEY_ATTRS), (
        f"deprecated cases drift from the module-level tuple:\n"
        f"  only in cases : {sorted(case_attrs - set(_DEPRECATED_GEOGRAPHIC_GEOKEY_ATTRS))}\n"
        f"  only in module: {sorted(set(_DEPRECATED_GEOGRAPHIC_GEOKEY_ATTRS) - case_attrs)}"
    )


def test_deprecated_cases_has_no_duplicates():
    """Length-equality guard so a duplicate row in ``_DEPRECATED_CASES``
    cannot be silently absorbed by the set comparison in
    ``test_deprecated_cases_cover_all_attrs``."""
    assert len(_DEPRECATED_CASES) == len(_DEPRECATED_GEOGRAPHIC_GEOKEY_ATTRS), (
        f"length mismatch: _DEPRECATED_CASES has {len(_DEPRECATED_CASES)} "
        f"rows but _DEPRECATED_GEOGRAPHIC_GEOKEY_ATTRS has "
        f"{len(_DEPRECATED_GEOGRAPHIC_GEOKEY_ATTRS)} entries. Likely cause: "
        f"a duplicate attr row in _DEPRECATED_CASES."
    )


@pytest.mark.parametrize(
    'attr,value',
    _DEPRECATED_CASES,
    ids=[c[0] for c in _DEPRECATED_CASES],
)
def test_warns_on_emission(attr, value):
    """Each deprecated geographic-GeoKey attr fires a DeprecationWarning
    with the canonical wording when ``_populate_attrs_from_geo_info``
    emits it."""
    info = _geo_info_with(**{attr: value})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        attrs: dict = {}
        _populate_attrs_from_geo_info(attrs, info)

    matching = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and _deprecated_geographic_geokey_warning(attr) == str(w.message)
    ]
    assert len(matching) == 1, (
        f"expected exactly one DeprecationWarning for {attr!r}; got "
        f"{[(w.category.__name__, str(w.message)) for w in caught]}"
    )


@pytest.mark.parametrize(
    'attr,value',
    _DEPRECATED_CASES,
    ids=[c[0] for c in _DEPRECATED_CASES],
)
def test_emission_still_present(attr, value):
    """Deprecation-period contract: the attr value still lands in attrs.

    Removal is a later PR. If a reader change drops the emission
    entirely while this test still expects presence, the failure here
    is the signal to bump the contract version and move the attr from
    the deprecated tier to the removed tier in the docstring."""
    info = _geo_info_with(**{attr: value})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        attrs: dict = {}
        _populate_attrs_from_geo_info(attrs, info)

    assert attr in attrs, (
        f"deprecated attr {attr!r} was dropped during PR 7's warning-only "
        f"phase. PR 7 keeps emitting; removal is scheduled for a later "
        f"release. attrs keys present: {sorted(attrs.keys())}"
    )
    assert attrs[attr] == value


def test_no_warning_when_field_absent():
    """A GeoInfo with none of the deprecated fields set fires no
    DeprecationWarning. Guards against an unconditional warning that
    would spam every read of every TIFF."""
    info = GeoInfo()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        attrs: dict = {}
        _populate_attrs_from_geo_info(attrs, info)

    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == [], (
        f"DeprecationWarning fired even though no deprecated field was "
        f"set on the GeoInfo: {[str(w.message) for w in dep]}"
    )


def test_warning_message_format():
    """Sanity-check the warning text shape so the canonical wording
    stays stable across the deprecation cycle."""
    msg = _deprecated_geographic_geokey_warning('crs_name')
    assert "xrspatial.geotiff" in msg
    assert "attrs['crs_name']" in msg
    assert "deprecated" in msg
    assert "round-trip" in msg
    assert "#1984" in msg


def test_warning_stacklevel_points_at_caller_file():
    """The ``DeprecationWarning`` filename should land on the caller's
    file, not on ``_attrs.py``.

    The emission helper computes ``stacklevel`` by walking past every
    ``xrspatial.geotiff*`` frame, so the warning reports the first
    external frame as its origin. The test file is outside that
    package, so ``w.filename`` should match ``__file__``. If a future
    refactor reintroduces a fixed ``stacklevel`` that is too small,
    the warning will be reattributed to one of the internal modules
    and this assertion will fail.

    Today the warning category is :class:`DeprecationWarning`, which
    Python silences by default for library code, so the stacklevel
    mostly affects test output. The pin lives here so a later switch
    to :class:`FutureWarning` does not regress the attribution
    silently.
    """
    info = _geo_info_with(crs_name='WGS 84')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        attrs: dict = {}
        _populate_attrs_from_geo_info(attrs, info)

    matches = [w for w in caught
               if issubclass(w.category, DeprecationWarning)]
    assert len(matches) == 1, [
        (w.category.__name__, str(w.message)) for w in caught
    ]
    assert matches[0].filename == __file__, (
        f"warning filename {matches[0].filename!r} (line "
        f"{matches[0].lineno}) does not match the test file "
        f"{__file__!r}; the stacklevel walk did not exit the "
        f"xrspatial.geotiff package."
    )
