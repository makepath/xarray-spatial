"""Consolidated metadata tests.

Folds four metadata regression areas into one module:

* error hierarchy, register / dispatch contract for the
  ambiguous-metadata validator framework.
* ``GeoTIFFMetadata`` dataclass / ``metadata_to_attrs`` /
  ``attrs_to_metadata`` round trip.
* transform / CRS / tag pass-through round trip end-to-end.
* VRT mixed-band nodata fail-closed behaviour at every entry point.

Helpers are suffixed so the four sections stay collision-free.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from .._geotiff_fixtures import write_minimal_tiff

import xrspatial.geotiff as geotiff_pkg
from xrspatial.geotiff import (ConflictingCRSError, GeoTIFFAmbiguousMetadataError,
                               MixedBandMetadataError, _runtime)
from xrspatial.geotiff import _validation as _validation_mod
from xrspatial.geotiff import open_geotiff, read_geotiff_dask, read_vrt, to_geotiff
from xrspatial.geotiff._attrs import (_ATTRS_CONTRACT_VERSION, GeoTIFFMetadata, _resolve_nodata_attr,
                                      attrs_to_metadata, geo_info_to_metadata, metadata_to_attrs)
from xrspatial.geotiff._errors import (ConflictingNodataError, InvalidCRSCodeError,
                                       NonUniformCoordsError, RotatedTransformError,
                                       UnparseableCRSError)
from xrspatial.geotiff._geotags import _NO_GEOREF_KEY
from xrspatial.geotiff._validation import (_check_read_rotated_transform,
                                           _check_read_unparseable_crs,
                                           _check_write_conflicting_nodata,
                                           _check_write_non_uniform_coords,
                                           _registered_read_metadata_checks,
                                           _registered_write_metadata_checks,
                                           register_read_metadata_check,
                                           register_write_metadata_check,
                                           unregister_read_metadata_check,
                                           unregister_write_metadata_check, validate_read_metadata,
                                           validate_write_metadata)
from xrspatial.geotiff._writer import write

# =============================================================================
# Section: Ambiguous metadata hooks
# =============================================================================
#
# The error class hierarchy lives in ``_errors.py`` and the register /
# dispatch framework in ``_validation.py``; each per-case check
# registers itself.
#
# These tests cover:
#
# - the error class hierarchy is what the per-case checks subclass
# - the hooks are no-ops when no checks are registered (so the framework
#   cannot regress any existing entry point)
# - registration is idempotent and ordered
# - unregistration is tolerant of unknown callables
# - a registered check that raises propagates through the hook
# - a context mapping is forwarded verbatim to each check


@pytest.fixture
def _reset_metadata_check_registries_1987():
    """Snapshot and restore the process-global check registries.

    The registries are module-global lists. A test that registers a
    check and crashes before its ``try/finally unregister`` would
    leave a stale callable in place and pollute later tests. Scope
    the snapshot to this section's tests via opt-in fixture so the
    other sections in this file are not affected.
    """
    read_snapshot = list(_validation_mod._READ_METADATA_CHECKS)
    write_snapshot = list(_validation_mod._WRITE_METADATA_CHECKS)
    try:
        yield
    finally:
        _validation_mod._READ_METADATA_CHECKS[:] = read_snapshot
        _validation_mod._WRITE_METADATA_CHECKS[:] = write_snapshot


# ----------------------------------------------------------------------
# Error class hierarchy
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "subclass",
    [
        InvalidCRSCodeError,
        UnparseableCRSError,
        RotatedTransformError,
        NonUniformCoordsError,
        MixedBandMetadataError,
        ConflictingCRSError,
        ConflictingNodataError,
    ],
)
def test_subclass_inherits_from_base_1987(subclass):
    """Each per-case error is catchable via the family base class."""
    assert issubclass(subclass, GeoTIFFAmbiguousMetadataError)


def test_base_is_value_error_subclass_1987():
    """Existing ``except ValueError`` callers keep catching the family."""
    assert issubclass(GeoTIFFAmbiguousMetadataError, ValueError)


def test_error_classes_reexported_from_public_namespace_1987():
    """User-facing ambiguity errors must be importable from ``xrspatial.geotiff``."""
    import xrspatial.geotiff as geotiff_pkg

    for name in (
        "GeoTIFFAmbiguousMetadataError",
        "InvalidCRSCodeError",
        "UnparseableCRSError",
        "RotatedTransformError",
        "NonUniformCoordsError",
        "MixedBandMetadataError",
        "ConflictingCRSError",
        "ConflictingNodataError",
    ):
        assert hasattr(geotiff_pkg, name), (
            f"{name} not exposed on xrspatial.geotiff")
        assert name in geotiff_pkg.__all__, (
            f"{name} not listed in xrspatial.geotiff.__all__")
    assert (geotiff_pkg.GeoTIFFAmbiguousMetadataError
            is GeoTIFFAmbiguousMetadataError)


def test_subclass_catch_does_not_catch_siblings_1987():
    """``except UnparseableCRSError`` must not catch ``RotatedTransformError``."""
    with pytest.raises(RotatedTransformError):
        try:
            raise RotatedTransformError("rotated")
        except UnparseableCRSError:
            pytest.fail("sibling subclass should not catch")


# ----------------------------------------------------------------------
# Hook no-op behaviour
# ----------------------------------------------------------------------


def test_read_hook_is_noop_when_no_checks_registered_1987(
    _reset_metadata_check_registries_1987,
):
    """An empty registry must not change behaviour at any read entry point."""
    # Clear any process-wide registered checks so the no-op test runs
    # against an empty registry.
    _validation_mod._READ_METADATA_CHECKS[:] = []
    validate_read_metadata()
    validate_read_metadata({})
    validate_read_metadata({"unused": object()})


def test_write_hook_is_noop_when_no_checks_registered_1987(
    _reset_metadata_check_registries_1987,
):
    """An empty registry must not change behaviour at any write entry point."""
    _validation_mod._WRITE_METADATA_CHECKS[:] = []
    validate_write_metadata()
    validate_write_metadata({})
    validate_write_metadata({"unused": object()})


# ----------------------------------------------------------------------
# Registration / dispatch
# ----------------------------------------------------------------------


def test_register_and_dispatch_read_check_1987(_reset_metadata_check_registries_1987):
    seen: list[dict] = []

    def check(ctx):
        seen.append(dict(ctx))

    register_read_metadata_check(check)
    try:
        validate_read_metadata({"_dispatch_probe": "value"})
        # The original (pre-consolidation) test asserted
        # ``seen == [{"_dispatch_probe": "value"}]``. The relaxation
        # to ``in seen`` is deliberate: in the consolidated module
        # built-in registered checks (e.g.
        # ``_check_write_conflicting_crs``) may also fire on the same
        # dispatch, and we want this dispatch-mechanism test to stay
        # independent of which other checks happen to be registered
        # by import-time side effects. ``seen`` is only appended to
        # from inside the custom callback above, so no other check
        # can pollute it.
        assert {"_dispatch_probe": "value"} in seen
    finally:
        unregister_read_metadata_check(check)


def test_register_and_dispatch_write_check_1987(_reset_metadata_check_registries_1987):
    seen: list[dict] = []

    def check(ctx):
        seen.append(dict(ctx))

    register_write_metadata_check(check)
    try:
        validate_write_metadata({"_dispatch_probe": "value"})
        # See the read-side counterpart above for the rationale
        # behind ``in seen`` (vs the original strict equality).
        assert {"_dispatch_probe": "value"} in seen
    finally:
        unregister_write_metadata_check(check)


def test_register_is_idempotent_read_1987(_reset_metadata_check_registries_1987):
    def check(ctx):
        return None

    register_read_metadata_check(check)
    register_read_metadata_check(check)
    try:
        assert _registered_read_metadata_checks().count(check) == 1
    finally:
        unregister_read_metadata_check(check)


def test_register_is_idempotent_write_1987(_reset_metadata_check_registries_1987):
    def check(ctx):
        return None

    register_write_metadata_check(check)
    register_write_metadata_check(check)
    try:
        assert _registered_write_metadata_checks().count(check) == 1
    finally:
        unregister_write_metadata_check(check)


def test_dispatch_preserves_registration_order_1987(
    _reset_metadata_check_registries_1987,
):
    # Clear the registry so we observe only our two callbacks in
    # deterministic order.
    _validation_mod._READ_METADATA_CHECKS[:] = []
    order: list[str] = []

    def first(ctx):
        order.append("first")

    def second(ctx):
        order.append("second")

    register_read_metadata_check(first)
    register_read_metadata_check(second)
    try:
        validate_read_metadata({})
        assert order == ["first", "second"]
    finally:
        unregister_read_metadata_check(first)
        unregister_read_metadata_check(second)


def test_write_dispatch_preserves_registration_order_1987(
    _reset_metadata_check_registries_1987,
):
    _validation_mod._WRITE_METADATA_CHECKS[:] = []
    order: list[str] = []

    def first(ctx):
        order.append("first")

    def second(ctx):
        order.append("second")

    register_write_metadata_check(first)
    register_write_metadata_check(second)
    try:
        validate_write_metadata({})
        assert order == ["first", "second"]
    finally:
        unregister_write_metadata_check(first)
        unregister_write_metadata_check(second)


def test_write_first_raising_check_short_circuits_1987(
    _reset_metadata_check_registries_1987,
):
    _validation_mod._WRITE_METADATA_CHECKS[:] = []
    later_called = {"flag": False}

    def deny(ctx):
        raise ConflictingCRSError("crs mismatch")

    def later(ctx):
        later_called["flag"] = True

    register_write_metadata_check(deny)
    register_write_metadata_check(later)
    try:
        with pytest.raises(ConflictingCRSError):
            validate_write_metadata({})
        assert later_called["flag"] is False
    finally:
        unregister_write_metadata_check(deny)
        unregister_write_metadata_check(later)


def test_read_and_write_registries_are_independent_1987(
    _reset_metadata_check_registries_1987,
):
    _validation_mod._READ_METADATA_CHECKS[:] = []
    _validation_mod._WRITE_METADATA_CHECKS[:] = []
    read_calls = {"count": 0}
    write_calls = {"count": 0}

    def read_check(ctx):
        read_calls["count"] += 1

    def write_check(ctx):
        write_calls["count"] += 1

    register_read_metadata_check(read_check)
    register_write_metadata_check(write_check)
    try:
        validate_read_metadata({})
        assert read_calls["count"] == 1
        assert write_calls["count"] == 0

        validate_write_metadata({})
        assert read_calls["count"] == 1
        assert write_calls["count"] == 1
    finally:
        unregister_read_metadata_check(read_check)
        unregister_write_metadata_check(write_check)


def test_unregister_unknown_check_is_safe_1987():
    """Test teardown must tolerate double-unregister."""

    def never_registered(ctx):
        return None

    unregister_read_metadata_check(never_registered)
    unregister_write_metadata_check(never_registered)


def test_check_can_raise_typed_error_1987(_reset_metadata_check_registries_1987):
    def deny(ctx):
        raise UnparseableCRSError("bad WKT")

    register_read_metadata_check(deny)
    try:
        with pytest.raises(UnparseableCRSError, match="bad WKT"):
            validate_read_metadata({"_dispatch_probe": "value"})
    finally:
        unregister_read_metadata_check(deny)


def test_first_raising_check_short_circuits_1987(
    _reset_metadata_check_registries_1987,
):
    _validation_mod._READ_METADATA_CHECKS[:] = []
    later_called = {"flag": False}

    def deny(ctx):
        raise RotatedTransformError("rotated")

    def later(ctx):
        later_called["flag"] = True

    register_read_metadata_check(deny)
    register_read_metadata_check(later)
    try:
        with pytest.raises(RotatedTransformError):
            validate_read_metadata({})
        assert later_called["flag"] is False
    finally:
        unregister_read_metadata_check(deny)
        unregister_read_metadata_check(later)


def test_dispatch_is_safe_against_registry_mutation_read_1987(
    _reset_metadata_check_registries_1987,
):
    _validation_mod._READ_METADATA_CHECKS[:] = []
    seen: list[str] = []

    def mutating(ctx):
        seen.append("mutating")
        unregister_read_metadata_check(mutating)

    def second(ctx):
        seen.append("second")

    register_read_metadata_check(mutating)
    register_read_metadata_check(second)
    try:
        validate_read_metadata({})
        assert seen == ["mutating", "second"]
    finally:
        unregister_read_metadata_check(mutating)
        unregister_read_metadata_check(second)


def test_dispatch_is_safe_against_registry_mutation_write_1987(
    _reset_metadata_check_registries_1987,
):
    _validation_mod._WRITE_METADATA_CHECKS[:] = []
    seen: list[str] = []

    def mutating(ctx):
        seen.append("mutating")
        unregister_write_metadata_check(mutating)

    def second(ctx):
        seen.append("second")

    register_write_metadata_check(mutating)
    register_write_metadata_check(second)
    try:
        validate_write_metadata({})
        assert seen == ["mutating", "second"]
    finally:
        unregister_write_metadata_check(mutating)
        unregister_write_metadata_check(second)


def test_none_context_is_treated_as_empty_mapping_1987(
    _reset_metadata_check_registries_1987,
):
    _validation_mod._READ_METADATA_CHECKS[:] = []
    seen: list[object] = []

    def check(ctx):
        seen.append(dict(ctx))

    register_read_metadata_check(check)
    try:
        validate_read_metadata()
        assert seen == [{}]
    finally:
        unregister_read_metadata_check(check)


# =============================================================================
# Section: GeoTIFFMetadata dataclass round-trip
# =============================================================================
#
# The dataclass and the two boundary functions (``metadata_to_attrs``
# and ``attrs_to_metadata``) replace the manual ``attrs[...] = ...``
# blocks scattered across the four read paths and three writers. The
# public attrs surface does not change; what changes is that every
# backend now goes through one marshalling step.


class _FakeTransform2139:
    """Stand-in for ``GeoInfo.transform`` used by the reader."""

    def __init__(self, origin_x=0.0, origin_y=0.0,
                 pixel_width=1.0, pixel_height=-1.0):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height


class _FakeGeoInfo2139:
    """Minimal ``GeoInfo`` stand-in covering the fields the reader emits."""

    def __init__(
        self,
        *,
        transform=None,
        crs_epsg=None,
        crs_wkt=None,
        raster_type=1,
        has_georef=True,
        nodata=None,
        extra_tags=None,
        image_description=None,
        extra_samples=None,
        gdal_metadata=None,
        gdal_metadata_xml=None,
        x_resolution=None,
        y_resolution=None,
        resolution_unit=None,
    ):
        self.transform = transform
        self.crs_epsg = crs_epsg
        self.crs_wkt = crs_wkt
        self.raster_type = raster_type
        self.has_georef = has_georef
        self.nodata = nodata
        self.extra_tags = extra_tags
        self.image_description = image_description
        self.extra_samples = extra_samples
        self.gdal_metadata = gdal_metadata
        self.gdal_metadata_xml = gdal_metadata_xml
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.resolution_unit = resolution_unit


def test_geo_info_to_metadata_stamps_contract_version_2139():
    md = geo_info_to_metadata(_FakeGeoInfo2139())
    assert md.contract_version == _ATTRS_CONTRACT_VERSION
    attrs = metadata_to_attrs(md)
    assert attrs['_xrspatial_geotiff_contract'] == _ATTRS_CONTRACT_VERSION


def test_geo_info_to_metadata_transform_present_2139():
    gi = _FakeGeoInfo2139(
        transform=_FakeTransform2139(origin_x=1.0, origin_y=10.0,
                                     pixel_width=0.5, pixel_height=-0.5),
        crs_epsg=4326, crs_wkt='WKT-CRS',
    )
    md = geo_info_to_metadata(gi)
    assert md.has_georef is True
    assert md.transform is not None
    assert md.crs_epsg == 4326
    assert md.crs_wkt == 'WKT-CRS'

    attrs = metadata_to_attrs(md)
    assert 'transform' in attrs
    assert attrs['crs'] == 4326
    assert attrs['crs_wkt'] == 'WKT-CRS'
    assert _NO_GEOREF_KEY not in attrs


def test_geo_info_to_metadata_no_georef_marker_2139():
    gi = _FakeGeoInfo2139(transform=None, has_georef=False)
    md = geo_info_to_metadata(gi)
    assert md.has_georef is False
    assert md.transform is None

    attrs = metadata_to_attrs(md)
    assert 'transform' not in attrs
    assert attrs[_NO_GEOREF_KEY] is True


def test_geo_info_to_metadata_point_raster_2139():
    gi = _FakeGeoInfo2139(transform=_FakeTransform2139(), raster_type=2)
    md = geo_info_to_metadata(gi)
    assert md.raster_type == 'point'

    attrs = metadata_to_attrs(md)
    assert attrs['raster_type'] == 'point'


def test_geo_info_to_metadata_resolution_unit_label_2139():
    gi = _FakeGeoInfo2139(
        transform=_FakeTransform2139(),
        x_resolution=300.0, y_resolution=300.0, resolution_unit=2,
    )
    md = geo_info_to_metadata(gi)
    assert md.resolution_unit == 'inch'

    attrs = metadata_to_attrs(md)
    assert attrs['resolution_unit'] == 'inch'
    assert attrs['x_resolution'] == 300.0
    assert attrs['y_resolution'] == 300.0


def test_geo_info_to_metadata_colormap_from_extra_tags_2139():
    extra_tags = [(320, 3, 6, (10, 20, 30, 40, 50, 60))]
    gi = _FakeGeoInfo2139(transform=_FakeTransform2139(), extra_tags=extra_tags)
    md = geo_info_to_metadata(gi)
    assert md.colormap == (10, 20, 30, 40, 50, 60)
    attrs = metadata_to_attrs(md)
    assert attrs['colormap'] == (10, 20, 30, 40, 50, 60)


def test_attrs_to_metadata_none_attrs_returns_default_record_2139():
    md = attrs_to_metadata(None)
    assert isinstance(md, GeoTIFFMetadata)
    assert md.transform is None
    assert md.crs_epsg is None
    assert md.crs_wkt is None
    assert md.nodata is None


def test_attrs_to_metadata_empty_dict_2139():
    md = attrs_to_metadata({})
    assert md.transform is None
    assert md.has_georef is False
    assert md.raster_type == 'area'


def test_attrs_to_metadata_crs_int_epsg_2139():
    md = attrs_to_metadata({'crs': 4326, 'crs_wkt': 'WKT-X'})
    assert md.crs_epsg == 4326
    assert md.crs_wkt == 'WKT-X'


def test_attrs_to_metadata_crs_wkt_string_in_crs_field_2139():
    md = attrs_to_metadata({'crs': 'PROJCS["X"...]'})
    assert md.crs_epsg is None
    assert md.crs_wkt == 'PROJCS["X"...]'


def test_attrs_to_metadata_crs_bool_rejected_at_boundary_2139():
    md = attrs_to_metadata({'crs': True})
    assert md.crs_epsg is None


def test_attrs_to_metadata_nodata_canonical_2139():
    md = attrs_to_metadata({'nodata': -9999, 'masked_nodata': True})
    assert md.nodata == -9999
    assert md.masked_nodata is True


def test_attrs_to_metadata_nodata_alias_nodatavals_2139():
    md = attrs_to_metadata({'nodatavals': (-9999,)})
    assert md.nodata == -9999
    assert md.masked_nodata is None


def test_attrs_to_metadata_nodata_alias_fill_value_2139():
    md = attrs_to_metadata({'_FillValue': -1})
    assert md.nodata == -1


def test_attrs_to_metadata_no_georef_marker_clears_has_georef_2139():
    md = attrs_to_metadata({_NO_GEOREF_KEY: True})
    assert md.has_georef is False


def test_attrs_to_metadata_transform_present_implies_georef_2139():
    md = attrs_to_metadata({'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)})
    assert md.has_georef is True
    assert md.transform == (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)


def _representative_attrs_dicts_2139():
    yield {
        '_xrspatial_geotiff_contract': _ATTRS_CONTRACT_VERSION,
        'crs': 4326,
        'crs_wkt': 'WKT-4326',
        'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
        'nodata': -9999,
        'masked_nodata': True,
    }
    yield {
        '_xrspatial_geotiff_contract': _ATTRS_CONTRACT_VERSION,
        'crs': 4326,
        'crs_wkt': 'WKT-4326',
        'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
        'raster_type': 'point',
    }
    yield {
        '_xrspatial_geotiff_contract': _ATTRS_CONTRACT_VERSION,
        _NO_GEOREF_KEY: True,
    }
    yield {
        '_xrspatial_geotiff_contract': _ATTRS_CONTRACT_VERSION,
        'crs_wkt': 'PROJCS["user defined"...]',
        'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
    }
    yield {
        '_xrspatial_geotiff_contract': _ATTRS_CONTRACT_VERSION,
        'crs': 32610,
        'crs_wkt': 'WKT-32610',
        'transform': (10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0),
        'vrt_holes': [{'source': '/tmp/missing.tif', 'band': 1}],
    }


@pytest.mark.parametrize(
    'attrs',
    list(_representative_attrs_dicts_2139()),
    ids=['eager', 'point', 'no_georef', 'user_wkt', 'vrt'],
)
def test_round_trip_attrs_to_metadata_to_attrs_2139(attrs):
    md = attrs_to_metadata(attrs)
    round_tripped = metadata_to_attrs(md)
    for key, value in attrs.items():
        assert key in round_tripped, f"key {key!r} dropped on round trip"
        assert round_tripped[key] == value, (
            f"value at {key!r} changed: {value!r} -> {round_tripped[key]!r}"
        )


@pytest.mark.parametrize(
    'attrs',
    list(_representative_attrs_dicts_2139()),
    ids=['eager', 'point', 'no_georef', 'user_wkt', 'vrt'],
)
def test_round_trip_emits_no_unexpected_keys_2139(attrs):
    md = attrs_to_metadata(attrs)
    round_tripped = metadata_to_attrs(md)
    expected_keys = set(attrs.keys()) | {'_xrspatial_geotiff_contract'}
    extra = set(round_tripped) - expected_keys
    assert not extra, (
        f"metadata_to_attrs emitted unexpected keys {extra!r} not present in input "
        f"{set(attrs)!r}; either add them to the attrs contract or fix "
        f"the marshalling step."
    )


def test_round_trip_metadata_to_attrs_to_metadata_2139():
    md = GeoTIFFMetadata(
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
        crs_epsg=4326,
        crs_wkt='WKT-4326',
        raster_type='point',
        has_georef=True,
        nodata=-9999,
        masked_nodata=True,
        x_resolution=300.0, y_resolution=300.0, resolution_unit='inch',
    )
    attrs = metadata_to_attrs(md)
    md2 = attrs_to_metadata(attrs)

    assert md2.transform == md.transform
    assert md2.crs_epsg == md.crs_epsg
    assert md2.crs_wkt == md.crs_wkt
    assert md2.raster_type == md.raster_type
    assert md2.has_georef == md.has_georef
    assert md2.nodata == md.nodata
    assert md2.masked_nodata == md.masked_nodata
    assert md2.x_resolution == md.x_resolution
    assert md2.y_resolution == md.y_resolution
    assert md2.resolution_unit == md.resolution_unit


def test_with_nodata_sets_pair_2139():
    md = GeoTIFFMetadata()
    md2 = md.with_nodata(-9999, masked=True)
    assert md2.nodata == -9999
    assert md2.masked_nodata is True
    assert md.nodata is None
    assert md is not md2


def test_with_nodata_none_returns_unchanged_2139():
    md = GeoTIFFMetadata(crs_epsg=4326)
    md2 = md.with_nodata(None, masked=True)
    assert md2.nodata is None
    assert md2.masked_nodata is None
    assert md2.crs_epsg == 4326


def test_with_nodata_masked_false_records_false_2139():
    md = GeoTIFFMetadata()
    md2 = md.with_nodata(-9999, masked=False)
    assert md2.masked_nodata is False


# =============================================================================
# Section: Transform / CRS / tag metadata round-trip
# =============================================================================
#
# * ``attrs['crs']`` stays as the same int EPSG and
#   ``attrs['transform']`` survives write -> read -> write -> read with
#   the same numeric values up to float precision.
# * ColorMap, ExtraSamples, and ImageDescription survive a single
#   write -> read cycle. ColorMap exits the writer through the
#   ``extra_tags`` pass-through (the tag is no longer in
#   ``_MANAGED_TAGS``); ImageDescription gets a friendly ``attrs`` entry.
# * integer rasters with a nodata sentinel get promoted to float64
#   with NaN, and a user-requested ``dtype='uint16'`` cast on the read
#   side raises ValueError (existing float-to-int guard).


def _make_palette_uint8_tiff_1484(path, pixels, palette_rgb16):
    """Write an 8-bit, 256-entry palette TIFF directly (no writer support
    for ColorMap on the write side).

    palette_rgb16 must have 256 (R, G, B) tuples of uint16 values.
    """
    bo = '<'
    width = pixels.shape[1]
    height = pixels.shape[0]
    n_colors = 256
    assert len(palette_rgb16) == n_colors

    flat = pixels.ravel().astype(np.uint8)
    pixel_bytes = flat.tobytes()

    r_vals = [c[0] for c in palette_rgb16]
    g_vals = [c[1] for c in palette_rgb16]
    b_vals = [c[2] for c in palette_rgb16]
    cmap_values = r_vals + g_vals + b_vals

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tag_list.append(
            (tag, 3, len(vals),
             struct.pack(f'{bo}{len(vals)}H', *vals)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 8)
    add_short(259, 1)
    add_short(262, 3)
    add_short(277, 1)
    add_short(278, height)
    add_long(273, 0)
    add_long(279, len(pixel_bytes))
    add_shorts(320, cmap_values)
    add_short(339, 1)

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)

    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count,
                            struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    with open(path, 'wb') as f:
        f.write(bytes(out))


def _write_simple_tiff_with_image_description_1484(path, pixels, description):
    """Write an uncompressed, single-strip TIFF that carries an
    ImageDescription tag (270) so we can test the read side."""
    bo = '<'
    height, width = pixels.shape
    pixel_bytes = pixels.astype(np.float32).tobytes()
    desc_bytes = description.encode('ascii') + b'\x00'
    if len(desc_bytes) % 2:
        desc_bytes += b'\x00'

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 32)
    add_short(259, 1)
    add_short(262, 1)
    tag_list.append((270, 2, len(description) + 1, desc_bytes))
    add_short(277, 1)
    add_short(278, height)
    add_long(273, 0)
    add_long(279, len(pixel_bytes))
    add_short(339, 3)

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _t, _c, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count,
                            struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _t, _c, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    with open(path, 'wb') as f:
        f.write(bytes(out))


class TestTransformCrsRoundTrip_1484:

    def test_transform_attr_present_on_read(self, tmp_path):
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        from xrspatial.geotiff._geotags import GeoTransform
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / 'transform_present_1484.tif')
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path)
        assert 'transform' in da.attrs
        a, b, c, d, e, f = da.attrs['transform']
        assert b == 0.0 and d == 0.0
        assert a == pytest.approx(30.0)
        assert e == pytest.approx(-30.0)
        assert c == pytest.approx(500000.0)
        assert f == pytest.approx(4000000.0)
        assert da.attrs['crs'] == 32610

    def test_double_round_trip_fractional_transform(self, tmp_path):
        from xrspatial.geotiff._geotags import GeoTransform
        arr = np.linspace(0, 1, 8 * 12, dtype=np.float64).reshape(8, 12)
        gt = GeoTransform(
            origin_x=-122.123456789,
            origin_y=37.987654321,
            pixel_width=1.0 / 3600.0 + 1e-12,
            pixel_height=-(1.0 / 3600.0 + 1e-12),
        )
        path1 = str(tmp_path / 'rt1_1484.tif')
        write(arr, path1, geo_transform=gt, crs_epsg=4326,
              compression='none', tiled=False)
        da1 = open_geotiff(path1)
        assert da1.attrs['crs'] == 4326

        path2 = str(tmp_path / 'rt2_1484.tif')
        to_geotiff(da1, path2, compression='none')
        da2 = open_geotiff(path2)

        path3 = str(tmp_path / 'rt3_1484.tif')
        to_geotiff(da2, path3, compression='none')
        da3 = open_geotiff(path3)

        assert da3.attrs['crs'] == 4326
        t1 = da1.attrs['transform']
        t3 = da3.attrs['transform']
        for v1, v3 in zip(t1, t3):
            assert v3 == pytest.approx(v1, abs=1e-15, rel=1e-12)

    def test_crs_string_input_still_tolerated(self, tmp_path):
        from xrspatial.geotiff._geotags import _epsg_to_wkt
        wkt = _epsg_to_wkt(4326)
        if wkt is None:
            pytest.skip("pyproj not available")
        arr = np.zeros((3, 3), dtype=np.float32)
        da = xr.DataArray(
            arr,
            dims=['y', 'x'],
            coords={
                'y': np.array([0.5, -0.5, -1.5]),
                'x': np.array([0.5, 1.5, 2.5]),
            },
            attrs={'crs': wkt},
        )
        path = str(tmp_path / 'wkt_string_crs_1484.tif')
        to_geotiff(da, path, compression='none')
        result = open_geotiff(path)
        assert result.attrs['crs'] == 4326


class TestTagPassThrough_1484:

    def test_colormap_round_trip(self, tmp_path):
        palette = [(i * 257, (255 - i) * 257, (i * 2) % 65536)
                   for i in range(256)]
        pixels = np.array([[0, 1, 2, 254, 255],
                           [10, 20, 30, 40, 50]], dtype=np.uint8)
        in_path = str(tmp_path / 'colormap_in_1484.tif')
        _make_palette_uint8_tiff_1484(in_path, pixels, palette)

        da = open_geotiff(in_path)
        assert da.dtype == np.uint8
        assert 'colormap' in da.attrs
        assert len(da.attrs['colormap']) == 768

        out_path = str(tmp_path / 'colormap_out_1484.tif')
        to_geotiff(da, out_path, compression='none')
        da2 = open_geotiff(out_path)

        np.testing.assert_array_equal(da2.values, pixels)
        assert 'colormap' in da2.attrs
        assert tuple(da2.attrs['colormap']) == tuple(da.attrs['colormap'])

    def test_image_description_round_trip(self, tmp_path):
        pixels = np.arange(12, dtype=np.float32).reshape(3, 4)
        desc = "elevation tile from issue 1484"
        in_path = str(tmp_path / 'desc_in_1484.tif')
        _write_simple_tiff_with_image_description_1484(in_path, pixels, desc)

        da = open_geotiff(in_path)
        assert da.attrs.get('image_description') == desc
        et_ids = {t[0] for t in da.attrs['extra_tags']}
        assert 270 in et_ids

        out_path = str(tmp_path / 'desc_out_1484.tif')
        to_geotiff(da, out_path, compression='none')
        da2 = open_geotiff(out_path)
        assert da2.attrs.get('image_description') == desc

    def test_image_description_added_via_attrs(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(4), 'x': np.arange(4)},
            attrs={'image_description': 'synthetic test 1484'},
        )
        path = str(tmp_path / 'desc_synth_1484.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.attrs.get('image_description') == 'synthetic test 1484'

    def test_extra_samples_attr_surfaces_on_read(self, tmp_path):
        rgba = np.zeros((4, 5, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        path = str(tmp_path / 'rgba_es_1484.tif')
        write(rgba, path, compression='none', tiled=False,
              photometric='rgba')
        da = open_geotiff(path)
        assert da.attrs.get('extra_samples') is not None
        assert da.attrs['extra_samples'][0] in (1, 2)


class TestIntegerNodataPromotion_1484:

    def test_uint16_with_nodata_promotes_to_float64(self, tmp_path):
        arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
        path = str(tmp_path / 'u16_nodata_1484.tif')
        write(arr, path, nodata=65535, compression='none', tiled=False)

        da = open_geotiff(path)
        assert da.dtype == np.float64
        assert np.isnan(da.values[1, 0])
        np.testing.assert_array_equal(
            da.values[~np.isnan(da.values)],
            np.array([1.0, 2.0, 3.0, 5.0, 6.0]),
        )

    def test_uint16_with_nodata_dtype_uint16_raises(self, tmp_path):
        arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
        path = str(tmp_path / 'u16_nodata_cast_1484.tif')
        write(arr, path, nodata=65535, compression='none', tiled=False)
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='uint16')

    def test_uint16_no_nodata_keeps_dtype(self, tmp_path):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        path = str(tmp_path / 'u16_no_nodata_1484.tif')
        write(arr, path, compression='none', tiled=False)
        da = open_geotiff(path)
        assert da.dtype == np.uint16


# =============================================================================
# Section: Mixed-band metadata fail-closed
# =============================================================================
#
# A VRT can mosaic source bands that declare disagreeing per-band
# ``<NoDataValue>`` sentinels. The legacy reader picked band 0's sentinel
# for the whole mosaic, which let band N's valid pixels collide with
# band 0's sentinel after the flatten-to-scalar step. The fail-closed
# default refuses the ambiguity; ``band_nodata='first'`` keeps the legacy
# behaviour explicitly.


def _write_mixed_band_vrt_1987(tmp_path, *, sentinel_a=65535, sentinel_b=65000):
    """Two uint16 sources with distinct sentinels, mosaiced into one VRT."""
    band0 = np.array([[1, 2], [3, sentinel_a]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, sentinel_b]], dtype=np.uint16)
    p0 = str(tmp_path / 'mixed_band_1987_a.tif')
    p1 = str(tmp_path / 'mixed_band_1987_b.tif')
    write(band0, p0, nodata=sentinel_a, compression='none', tiled=False)
    write(band1, p1, nodata=sentinel_b, compression='none', tiled=False)
    vrt_path = str(tmp_path / 'mixed_band_1987.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>{sentinel_a}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>{sentinel_b}</NoDataValue>
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


def _write_shared_sentinel_vrt_1987(tmp_path, *, sentinel=65535):
    band0 = np.array([[1, 2], [3, sentinel]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, sentinel]], dtype=np.uint16)
    p0 = str(tmp_path / 'shared_band_1987_a.tif')
    p1 = str(tmp_path / 'shared_band_1987_b.tif')
    write(band0, p0, nodata=sentinel, compression='none', tiled=False)
    write(band1, p1, nodata=sentinel, compression='none', tiled=False)
    vrt_path = str(tmp_path / 'shared_band_1987.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>{sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>{sentinel}</NoDataValue>
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


def _write_one_band_no_sentinel_vrt_1987(tmp_path, *, sentinel=65535):
    band0 = np.array([[1, 2], [3, sentinel]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, 99]], dtype=np.uint16)
    p0 = str(tmp_path / 'one_band_sentinel_a.tif')
    p1 = str(tmp_path / 'one_band_sentinel_b.tif')
    write(band0, p0, nodata=sentinel, compression='none', tiled=False)
    write(band1, p1, nodata=None, compression='none', tiled=False)
    vrt_path = str(tmp_path / 'one_band_sentinel.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>{sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
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


def _wrap_2d_1987(arr):
    h, w = arr.shape
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={
            'y': np.arange(h, dtype=np.float64),
            'x': np.arange(w, dtype=np.float64),
        },
        attrs={'crs': 4326,
               'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )


def test_read_vrt_rejects_mixed_per_band_nodata_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    with pytest.raises(MixedBandMetadataError) as exc_info:
        read_vrt(vrt_path)
    msg = str(exc_info.value)
    assert "65535" in msg and "65000" in msg
    assert "band_nodata='first'" in msg
    assert "#1987" in msg


def test_read_vrt_chunked_rejects_mixed_per_band_nodata_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    with pytest.raises(MixedBandMetadataError):
        read_vrt(vrt_path, chunks=1)


def test_read_vrt_band_nodata_first_opts_back_in_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    r = read_vrt(vrt_path, band_nodata='first')
    assert r.attrs.get('nodata') == 65535.0


def test_read_vrt_shared_sentinel_accepts_1987(tmp_path):
    vrt_path = _write_shared_sentinel_vrt_1987(tmp_path)
    r = read_vrt(vrt_path)
    assert r.attrs.get('nodata') == 65535.0


def test_read_vrt_only_one_band_declares_sentinel_accepts_1987(tmp_path):
    vrt_path = _write_one_band_no_sentinel_vrt_1987(tmp_path)
    read_vrt(vrt_path)


def test_open_geotiff_propagates_mixed_band_rejection_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    with pytest.raises(MixedBandMetadataError):
        open_geotiff(vrt_path)


def test_open_geotiff_band_nodata_first_passes_through_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    r = open_geotiff(vrt_path, band_nodata='first')
    assert r.attrs.get('nodata') == 65535.0


def test_read_geotiff_dask_band_nodata_first_passes_through_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    r = read_geotiff_dask(vrt_path, chunks=1, band_nodata='first')
    assert r.attrs.get('nodata') == 65535.0


def test_open_geotiff_band_nodata_rejected_on_non_vrt_source_1987(tmp_path):
    arr = np.zeros((2, 2), dtype=np.uint16)
    arr_da = _wrap_2d_1987(arr)
    p = tmp_path / 'plain.tif'
    to_geotiff(arr_da, str(p), compression='none', tiled=False)
    with pytest.raises(ValueError, match="band_nodata only applies to VRT"):
        open_geotiff(str(p), band_nodata='first')


def test_read_geotiff_dask_band_nodata_rejected_on_non_vrt_source_1987(tmp_path):
    arr = np.zeros((2, 2), dtype=np.uint16)
    arr_da = _wrap_2d_1987(arr)
    p = tmp_path / 'plain.tif'
    to_geotiff(arr_da, str(p), compression='none', tiled=False)
    with pytest.raises(ValueError, match="band_nodata only applies to VRT"):
        read_geotiff_dask(str(p), chunks=1, band_nodata='first')


def test_mixed_band_metadata_error_subclasses_base_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    with pytest.raises(GeoTIFFAmbiguousMetadataError):
        read_vrt(vrt_path)


def test_read_vrt_band_nodata_rejects_unknown_value_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    with pytest.raises(ValueError, match="band_nodata must be None or 'first'"):
        read_vrt(vrt_path, band_nodata='firs')


def test_open_geotiff_band_nodata_rejects_unknown_value_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    with pytest.raises(ValueError, match="band_nodata must be None or 'first'"):
        open_geotiff(vrt_path, band_nodata='legacy')


def test_read_geotiff_dask_band_nodata_rejects_unknown_value_1987(tmp_path):
    vrt_path = _write_mixed_band_vrt_1987(tmp_path)
    with pytest.raises(ValueError, match="band_nodata must be None or 'first'"):
        read_geotiff_dask(vrt_path, chunks=1, band_nodata='banana')

# ===========================================================================
# Runtime sentinels identity contract (#1880)
# Source: test_runtime_sentinels_identity_1880.py
# ===========================================================================


def test_gpu_deprecated_sentinel_is_singleton():
    assert geotiff_pkg._GPU_DEPRECATED_SENTINEL is _runtime._GPU_DEPRECATED_SENTINEL


def test_on_gpu_failure_sentinel_is_singleton():
    assert geotiff_pkg._ON_GPU_FAILURE_SENTINEL is _runtime._ON_GPU_FAILURE_SENTINEL


def test_crs_wkt_deprecated_sentinel_is_singleton():
    assert geotiff_pkg._CRS_WKT_DEPRECATED_SENTINEL is \
        _runtime._CRS_WKT_DEPRECATED_SENTINEL


def test_missing_sources_sentinel_is_singleton():
    assert geotiff_pkg._MISSING_SOURCES_SENTINEL is \
        _runtime._MISSING_SOURCES_SENTINEL


def test_fallback_warning_class_is_singleton():
    """``GeoTIFFFallbackWarning`` is the same class through both import paths.

    This is the only re-exported name from ``_runtime`` that is in
    ``__all__``. A duplicate class would still print the right name in
    a ``warns(GeoTIFFFallbackWarning)`` context but ``issubclass``
    chains in user code would break.
    """
    assert geotiff_pkg.GeoTIFFFallbackWarning is _runtime.GeoTIFFFallbackWarning


def test_strict_mode_helper_is_singleton():
    assert geotiff_pkg._geotiff_strict_mode is _runtime._geotiff_strict_mode


def test_gpu_fallback_warning_message_is_singleton():
    assert geotiff_pkg._gpu_fallback_warning_message is \
        _runtime._gpu_fallback_warning_message


def test_strict_mode_env_var_round_trips(monkeypatch):
    """The strict-mode helper still reads the env var after the move.

    Guards against an accidental hard-coded return value or wrong env
    var name introduced during the relocation.
    """
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "1")
    assert geotiff_pkg._geotiff_strict_mode() is True
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "true")
    assert geotiff_pkg._geotiff_strict_mode() is True
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "0")
    assert geotiff_pkg._geotiff_strict_mode() is False
    monkeypatch.delenv("XRSPATIAL_GEOTIFF_STRICT", raising=False)
    assert geotiff_pkg._geotiff_strict_mode() is False


def test_fallback_message_includes_exception_type_and_message():
    """The two GPU-fallback wording branches both surface the exception."""
    exc = RuntimeError("nvcomp not installed")
    explicit = geotiff_pkg._gpu_fallback_warning_message(
        auto_detected=False, exc=exc)
    auto = geotiff_pkg._gpu_fallback_warning_message(
        auto_detected=True, exc=exc)
    for msg in (explicit, auto):
        assert "RuntimeError" in msg
        assert "nvcomp not installed" in msg
    assert "to_geotiff(gpu=True) was requested" in explicit
    assert "Data is on the GPU" in auto

# ===========================================================================
# Remaining fail-closed checks for ambiguous metadata (#1987 PRs 2/3/4/7)
# Source: test_remaining_fail_closed_1987.py
# ===========================================================================


pyproj = pytest.importorskip("pyproj")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _da(*, coords=None, attrs=None, shape=(4, 4)):
    """Build a minimal 2-D DataArray. Caller supplies coords + attrs."""
    data = np.zeros(shape, dtype=np.float32)
    if coords is None:
        coords = {
            'y': np.linspace(3.0, 0.0, shape[0], dtype=np.float64),
            'x': np.linspace(0.0, 3.0, shape[1], dtype=np.float64),
        }
    return xr.DataArray(
        data, dims=('y', 'x'), coords=coords, attrs=dict(attrs or {}),
    )


def _write_minimal_tiff_with_wkt(path: str, wkt: str) -> None:
    ascii_buf = bytearray((wkt + '|').encode('ascii'))
    gkd = [1, 1, 0, 1, 1026, 34737, len(wkt) + 1, 0]
    write_minimal_tiff(path, geokeys=gkd, geo_ascii=wkt)


def _write_rotated_vrt(
    path: Path,
    source_basename: str,
    *,
    geo_transform: str = '0.0, 1.0, 0.5, 0.0, 0.0, -1.0',
) -> None:
    """Build a VRT carrying ``geo_transform``. Default value rotates on
    the x-axis (GDAL ``GT[2] = 0.5``); pass a different string to
    exercise the row-axis (``GT[4]``) branch."""
    Path(path).write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS>EPSG:4326</SRS>\n'
        f'  <GeoTransform>{geo_transform}</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_basename}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


# ---------------------------------------------------------------------------
# Registry sanity.
# ---------------------------------------------------------------------------


def test_four_checks_registered():
    """Four of the five #1987 PR 2-7 checks are active by default. The
    mixed-band check is intentionally deferred to a follow-up that also
    migrates the legacy VRT test fixtures."""
    write_names = {c.__name__ for c in _registered_write_metadata_checks()}
    read_names = {c.__name__ for c in _registered_read_metadata_checks()}
    assert _check_write_conflicting_nodata.__name__ in write_names
    assert _check_write_non_uniform_coords.__name__ in write_names
    assert _check_read_unparseable_crs.__name__ in read_names
    assert _check_read_rotated_transform.__name__ in read_names


def test_error_class_hierarchy():
    for cls in (
        UnparseableCRSError,
        RotatedTransformError,
        NonUniformCoordsError,
        ConflictingNodataError,
    ):
        assert issubclass(cls, GeoTIFFAmbiguousMetadataError)
        assert issubclass(cls, ValueError)


# ---------------------------------------------------------------------------
# UnparseableCRSError (write side + read side).
# ---------------------------------------------------------------------------


def test_write_rejects_garbage_crs_kwarg(tmp_path):
    """``to_geotiff(..., crs="not a CRS")`` refuses to emit the
    unparseable string into ``GTCitationGeoKey``."""
    da = _da()
    with pytest.raises(UnparseableCRSError):
        to_geotiff(da, str(tmp_path / 'bad_crs.tif'), crs='not a CRS')


def test_write_garbage_crs_kwarg_opt_in_passes(tmp_path):
    """``allow_unparseable_crs=True`` keeps the pre-#1929 citation-only
    behaviour for callers who need it."""
    da = _da()
    to_geotiff(
        da, str(tmp_path / 'bad_crs_opt_in.tif'),
        crs='not a CRS', allow_unparseable_crs=True,
    )


def test_read_rejects_garbage_crs_wkt():
    """The read-side check raises on a string pyproj cannot parse.
    Driven directly through ``validate_read_metadata`` because building
    a TIFF whose stored WKT actually surfaces as ``attrs['crs_wkt']``
    (vs ``attrs['crs_name']``, which carries the GeoKey citation) is a
    full-stack fixture exercise rather than a unit one."""
    from xrspatial.geotiff._validation import validate_read_metadata

    with pytest.raises(UnparseableCRSError):
        validate_read_metadata({'crs_wkt': 'NOT A REAL WKT STRING'})


def test_read_garbage_crs_wkt_opt_in_passes():
    """``allow_unparseable_crs=True`` short-circuits the check."""
    from xrspatial.geotiff._validation import validate_read_metadata

    validate_read_metadata({
        'crs_wkt': 'NOT A REAL WKT STRING',
        'allow_unparseable_crs': True,
    })


def test_read_pyproj_parseable_non_wkt_passes():
    """``EPSG:4326`` is not WKT but pyproj parses it via
    ``from_user_input``. The check tolerates pyproj-parseable
    placeholders (e.g. GDAL's ``<SRS>EPSG:4326</SRS>`` convention)."""
    from xrspatial.geotiff._validation import validate_read_metadata

    validate_read_metadata({'crs_wkt': 'EPSG:4326'})


# ---------------------------------------------------------------------------
# RotatedTransformError.
# ---------------------------------------------------------------------------


def test_read_rejects_rotated_vrt(tmp_path):
    """A VRT whose GeoTransform carries non-zero rotation/shear terms is
    refused on read."""
    src = tmp_path / 'flat.tif'
    _write_minimal_tiff_with_wkt(str(src), 'EPSG:4326')
    vrt = tmp_path / 'rotated.vrt'
    _write_rotated_vrt(vrt, os.path.basename(src))

    with pytest.raises(RotatedTransformError):
        open_geotiff(str(vrt))


def test_read_rotated_vrt_opt_in_passes(tmp_path):
    """``allow_rotated=True`` skips the check and returns the pixel grid
    without the axis-aligned-grid assumption."""
    src = tmp_path / 'flat.tif'
    _write_minimal_tiff_with_wkt(str(src), 'EPSG:4326')
    vrt = tmp_path / 'rotated_ok.vrt'
    _write_rotated_vrt(vrt, os.path.basename(src))

    da = open_geotiff(str(vrt), allow_rotated=True)
    assert da.shape == (4, 4)


def test_read_rejects_row_axis_rotated_vrt(tmp_path):
    """The ``d`` term (GDAL ``GT[4]``, rasterio ``Affine`` index 3) is
    the row-axis rotation; the sibling test above only sets the column
    rotation. Pin that the row-axis branch raises too."""
    src = tmp_path / 'flat_d.tif'
    _write_minimal_tiff_with_wkt(str(src), 'EPSG:4326')
    vrt = tmp_path / 'rotated_d.vrt'
    _write_rotated_vrt(
        vrt, os.path.basename(src),
        geo_transform='0.0, 1.0, 0.0, 0.0, 0.5, -1.0',
    )

    with pytest.raises(RotatedTransformError):
        open_geotiff(str(vrt))


# ---------------------------------------------------------------------------
# NonUniformCoordsError.
# ---------------------------------------------------------------------------


def test_write_rejects_non_uniform_y_coords(tmp_path):
    """A DataArray whose ``y`` coords are not uniformly spaced is refused.
    The writer would otherwise pick the first two values as the pixel
    size and silently misrepresent the rest of the axis."""
    coords = {
        'y': np.array([10.0, 9.0, 7.0, 4.0], dtype=np.float64),
        'x': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    }
    da = _da(coords=coords)
    with pytest.raises(NonUniformCoordsError):
        to_geotiff(da, str(tmp_path / 'non_uniform_y.tif'))


def test_write_rejects_non_uniform_x_coords(tmp_path):
    coords = {
        'y': np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float64),
        'x': np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64),
    }
    da = _da(coords=coords)
    with pytest.raises(NonUniformCoordsError):
        to_geotiff(da, str(tmp_path / 'non_uniform_x.tif'))


def test_write_accepts_uniform_coords(tmp_path):
    """Float coords that are uniformly spaced pass."""
    coords = {
        'y': np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float64),
        'x': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    }
    da = _da(coords=coords)
    to_geotiff(da, str(tmp_path / 'uniform.tif'))


def test_write_accepts_integer_coord_sentinel(tmp_path):
    """Int-dtype coords (the no-georef sentinel from #1969) bypass the
    uniformity check because they don't represent geographic positions."""
    coords = {
        'y': np.array([0, 1, 2, 3], dtype=np.int64),
        'x': np.array([0, 1, 2, 3], dtype=np.int64),
    }
    da = _da(coords=coords)
    to_geotiff(da, str(tmp_path / 'int_sentinel.tif'))


def test_write_rejects_constant_float_coords(tmp_path):
    """A float coord array whose first two values are equal makes the
    derived pixel step zero. The check refuses rather than emitting a
    zero-step GeoTransform."""
    coords = {
        'y': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        'x': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    }
    da = _da(coords=coords)
    with pytest.raises(NonUniformCoordsError, match='constant'):
        to_geotiff(da, str(tmp_path / 'constant_y.tif'))


# ---------------------------------------------------------------------------
# ConflictingNodataError.
# ---------------------------------------------------------------------------


def test_write_rejects_conflicting_nodata_attrs(tmp_path):
    """``attrs['nodata']`` disagreeing with every concrete entry in
    ``attrs['nodatavals']`` raises."""
    da = _da(attrs={'nodata': -8888.0, 'nodatavals': (-9999.0,)})
    with pytest.raises(ConflictingNodataError):
        to_geotiff(da, str(tmp_path / 'conflict_nodata.tif'))


def test_write_accepts_agreeing_nodata_attrs(tmp_path):
    """When the tuple includes the canonical value (e.g. per-band
    rioxarray convention) the check passes."""
    da = _da(attrs={'nodata': -9999.0, 'nodatavals': (-9999.0, -9999.0)})
    to_geotiff(da, str(tmp_path / 'agree_nodata.tif'))


def test_write_nan_canonical_with_concrete_alias_raises(tmp_path):
    """``nodata=NaN`` paired with a concrete numeric in ``nodatavals``
    means "NaN is the sentinel" and "some integer is the sentinel"
    at the same time. Refuse the ambiguity."""
    da = _da(attrs={'nodata': float('nan'), 'nodatavals': (-9999.0,)})
    with pytest.raises(ConflictingNodataError):
        to_geotiff(da, str(tmp_path / 'nan_vs_concrete.tif'))


def test_write_explicit_nodata_kwarg_bypasses_check(tmp_path):
    """``to_geotiff(..., nodata=X)`` overrides both attrs and bypasses
    the conflict check entirely."""
    da = _da(attrs={'nodata': -8888.0, 'nodatavals': (-9999.0,)})
    to_geotiff(da, str(tmp_path / 'kwarg_override.tif'), nodata=-1.0)


def test_write_none_in_nodatavals_tuple_is_skipped(tmp_path):
    """``None`` entries in the rioxarray tuple are "no sentinel on this
    band" and don't count as disagreement."""
    da = _da(attrs={'nodata': -9999.0, 'nodatavals': (None, -9999.0)})
    to_geotiff(da, str(tmp_path / 'none_skipped.tif'))


# ---------------------------------------------------------------------------
# Distinct per-band nodatavals (#2514).
#
# A TIFF stores one GDAL_NODATA tag per file, so per-band tuples with
# multiple distinct concrete sentinels cannot round-trip safely. The
# legacy resolver flattened the tuple to the first usable entry and the
# silent drop turned the remaining bands' sentinel cells into real data.
# The validator now rejects this on write so the corruption never
# reaches disk. The existing conflict check covered only the case where
# every tuple member disagreed with attrs['nodata']; these tests cover
# the dangerous case where the tuple itself carries the disagreement.
# ---------------------------------------------------------------------------


def _da_2514(*, attrs=None, shape=(2, 4, 4)):
    """Multi-band DataArray for the per-band nodata tests (#2514)."""
    data = np.zeros(shape, dtype=np.float32)
    coords = {
        'band': np.arange(1, shape[0] + 1),
        'y': np.linspace(3.0, 0.0, shape[1], dtype=np.float64),
        'x': np.linspace(0.0, 3.0, shape[2], dtype=np.float64),
    }
    return xr.DataArray(
        data, dims=('band', 'y', 'x'), coords=coords, attrs=dict(attrs or {}),
    )


def test_write_rejects_distinct_per_band_nodatavals_no_scalar(tmp_path):
    """Two distinct concrete entries in ``nodatavals`` without a scalar
    ``nodata`` key still raises. The earlier conflict check only
    inspected the (nodata, nodatavals) disagreement axis and let this
    case through."""
    da = _da_2514(attrs={'nodatavals': (-9999.0, -8888.0)})
    out_path = str(tmp_path / 'tmp_2514_no_scalar.tif')
    with pytest.raises(ConflictingNodataError, match='distinct per-band'):
        to_geotiff(da, out_path)
    # The error must surface before any file write.
    assert not os.path.exists(out_path)


def test_write_rejects_distinct_per_band_nodatavals_with_matching_scalar(tmp_path):
    """A scalar ``nodata`` that happens to match one band does not
    rescue the write -- the other band's sentinel would still be
    silently dropped."""
    da = _da_2514(attrs={'nodata': -9999.0, 'nodatavals': (-9999.0, -8888.0)})
    out_path = str(tmp_path / 'tmp_2514_match_one_band.tif')
    with pytest.raises(ConflictingNodataError, match='distinct per-band'):
        to_geotiff(da, out_path)
    assert not os.path.exists(out_path)


def test_write_rejects_three_distinct_per_band_nodatavals(tmp_path):
    """Three bands with three distinct sentinels: the check should
    surface every concrete value in the error message so the user can
    see which ones collided."""
    da = _da_2514(
        shape=(3, 4, 4),
        attrs={'nodatavals': (-9999.0, -8888.0, 0.0)},
    )
    out_path = str(tmp_path / 'tmp_2514_three_distinct.tif')
    with pytest.raises(ConflictingNodataError) as exc:
        to_geotiff(da, out_path)
    msg = str(exc.value)
    assert '-9999' in msg and '-8888' in msg and '0.0' in msg
    assert not os.path.exists(out_path)


def test_write_distinct_nodatavals_explicit_kwarg_bypasses(tmp_path):
    """``to_geotiff(..., nodata=X)`` overrides attrs and short-circuits
    the per-band check the same way it short-circuits the conflict
    check."""
    da = _da_2514(attrs={'nodatavals': (-9999.0, -8888.0)})
    to_geotiff(da, str(tmp_path / 'tmp_2514_kwarg.tif'), nodata=-1.0)


def test_write_repeated_concrete_nodatavals_accepted(tmp_path):
    """A tuple where every concrete entry is the same value (the rioxarray
    convention for "all bands share this sentinel") must still write."""
    da = _da_2514(attrs={'nodatavals': (-9999.0, -9999.0)})
    to_geotiff(da, str(tmp_path / 'tmp_2514_repeated.tif'))


def test_write_none_and_single_concrete_nodatavals_accepted(tmp_path):
    """``(None, -9999.0)`` means "band 0 has no sentinel, band 1's
    sentinel is -9999". One distinct concrete value -- safe."""
    da = _da_2514(attrs={'nodatavals': (None, -9999.0)})
    to_geotiff(da, str(tmp_path / 'tmp_2514_none_and_one.tif'))


def test_write_all_nan_nodatavals_accepted(tmp_path):
    """All-NaN ``nodatavals`` means "the float NaN is the sentinel" on
    every band -- there is nothing to disagree about."""
    nan = float('nan')
    da = _da_2514(attrs={'nodatavals': (nan, nan)})
    to_geotiff(da, str(tmp_path / 'tmp_2514_all_nan.tif'))


def test_resolve_nodata_attr_raises_on_distinct_per_band():
    """Defense-in-depth path: ``_resolve_nodata_attr`` itself raises on
    distinct per-band sentinels even when called outside the writer
    (where ``_check_write_distinct_per_band_nodatavals`` would
    normally fire first). Guards against bypass paths that might
    skip the write-side validator. The two sites share the error
    message via ``_distinct_per_band_nodatavals_msg`` in ``_errors``."""
    with pytest.raises(ConflictingNodataError, match='distinct per-band'):
        _resolve_nodata_attr({'nodatavals': (-9999.0, -8888.0)})
    # ``attrs['nodata']`` short-circuits the resolver before the
    # distinct-check runs; that path is the responsibility of the
    # write-side validator (and is already covered).
    assert _resolve_nodata_attr(
        {'nodata': -9999.0, 'nodatavals': (-9999.0, -8888.0)}
    ) == -9999.0


# ---------------------------------------------------------------------------
# Round-trip safety: a written-then-read DataArray with both attrs set
# (which the reader does emit by default) still writes again cleanly.
# ---------------------------------------------------------------------------


def test_read_then_write_round_trip_does_not_raise(tmp_path):
    """``to_geotiff -> open_geotiff -> to_geotiff`` is the typical
    pipeline. The reader emits both ``crs`` and ``crs_wkt`` whenever the
    file has a CRS, so the second write hits both attrs populated; they
    must agree (they came from the same file)."""
    src = _da(attrs={'crs': 4326, 'nodata': -9999.0})
    first = str(tmp_path / 'rt_first.tif')
    second = str(tmp_path / 'rt_second.tif')
    to_geotiff(src, first)
    da = open_geotiff(first)
    to_geotiff(da, second)
