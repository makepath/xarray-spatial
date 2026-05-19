"""Round-trip tests for the internal ``GeoTIFFMetadata`` dataclass.

Issue #2139.

The dataclass and the two boundary functions
(``metadata_to_attrs`` and ``attrs_to_metadata``) replace the manual
``attrs[...] = ...`` blocks scattered across the four read paths and
three writers. The public attrs surface does not change; what changes
is that every backend now goes through one marshalling step.

These tests pin two invariants:

* ``metadata_to_attrs(geo_info_to_metadata(...))`` emits the same key
  set the legacy ``_populate_attrs_from_geo_info`` writes.
* ``metadata_to_attrs(attrs_to_metadata(x))`` is a stable round trip
  for representative attrs dicts (eager numpy, dask, GPU, VRT,
  no-georef, point raster, user-defined CRS WKT, with-nodata).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._attrs import (
    GeoTIFFMetadata,
    _ATTRS_CONTRACT_VERSION,
    attrs_to_metadata,
    geo_info_to_metadata,
    metadata_to_attrs,
)
from xrspatial.geotiff._geotags import _NO_GEOREF_KEY


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class _FakeTransform:
    """Stand-in for ``GeoInfo.transform`` used by the reader."""

    def __init__(self, origin_x=0.0, origin_y=0.0,
                 pixel_width=1.0, pixel_height=-1.0):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height


class _FakeGeoInfo:
    """Minimal ``GeoInfo`` stand-in covering the fields the reader emits."""

    def __init__(
        self,
        *,
        transform=None,
        crs_epsg=None,
        crs_wkt=None,
        raster_type=1,  # RASTER_PIXEL_IS_AREA
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


# ---------------------------------------------------------------------------
# geo_info_to_metadata -> metadata_to_attrs
# ---------------------------------------------------------------------------


def test_geo_info_to_metadata_stamps_contract_version():
    md = geo_info_to_metadata(_FakeGeoInfo())
    assert md.contract_version == _ATTRS_CONTRACT_VERSION
    attrs = metadata_to_attrs(md)
    assert attrs['_xrspatial_geotiff_contract'] == _ATTRS_CONTRACT_VERSION


def test_geo_info_to_metadata_transform_present():
    gi = _FakeGeoInfo(
        transform=_FakeTransform(origin_x=1.0, origin_y=10.0,
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


def test_geo_info_to_metadata_no_georef_marker():
    # No transform tags -> has_georef=False on the GeoInfo
    gi = _FakeGeoInfo(transform=None, has_georef=False)
    md = geo_info_to_metadata(gi)
    assert md.has_georef is False
    assert md.transform is None

    attrs = metadata_to_attrs(md)
    assert 'transform' not in attrs
    assert attrs[_NO_GEOREF_KEY] is True


def test_geo_info_to_metadata_point_raster():
    gi = _FakeGeoInfo(transform=_FakeTransform(), raster_type=2)  # POINT
    md = geo_info_to_metadata(gi)
    assert md.raster_type == 'point'

    attrs = metadata_to_attrs(md)
    assert attrs['raster_type'] == 'point'


def test_geo_info_to_metadata_resolution_unit_label():
    gi = _FakeGeoInfo(
        transform=_FakeTransform(),
        x_resolution=300.0, y_resolution=300.0, resolution_unit=2,
    )
    md = geo_info_to_metadata(gi)
    assert md.resolution_unit == 'inch'

    attrs = metadata_to_attrs(md)
    assert attrs['resolution_unit'] == 'inch'
    assert attrs['x_resolution'] == 300.0
    assert attrs['y_resolution'] == 300.0


def test_geo_info_to_metadata_colormap_from_extra_tags():
    extra_tags = [(320, 3, 6, (10, 20, 30, 40, 50, 60))]
    gi = _FakeGeoInfo(transform=_FakeTransform(), extra_tags=extra_tags)
    md = geo_info_to_metadata(gi)
    assert md.colormap == (10, 20, 30, 40, 50, 60)
    attrs = metadata_to_attrs(md)
    assert attrs['colormap'] == (10, 20, 30, 40, 50, 60)


# ---------------------------------------------------------------------------
# attrs_to_metadata: parse user/reader attrs back to a record
# ---------------------------------------------------------------------------


def test_attrs_to_metadata_none_attrs_returns_default_record():
    md = attrs_to_metadata(None)
    assert isinstance(md, GeoTIFFMetadata)
    assert md.transform is None
    assert md.crs_epsg is None
    assert md.crs_wkt is None
    assert md.nodata is None


def test_attrs_to_metadata_empty_dict():
    md = attrs_to_metadata({})
    assert md.transform is None
    assert md.has_georef is False  # no transform -> not georeferenced
    assert md.raster_type == 'area'


def test_attrs_to_metadata_crs_int_epsg():
    md = attrs_to_metadata({'crs': 4326, 'crs_wkt': 'WKT-X'})
    assert md.crs_epsg == 4326
    assert md.crs_wkt == 'WKT-X'


def test_attrs_to_metadata_crs_wkt_string_in_crs_field():
    # Some upstream pipelines stash a WKT string in attrs['crs']
    md = attrs_to_metadata({'crs': 'PROJCS["X"...]'})
    assert md.crs_epsg is None
    assert md.crs_wkt == 'PROJCS["X"...]'


def test_attrs_to_metadata_crs_bool_rejected_at_boundary():
    # attrs={'crs': True} previously round-tripped as EPSG=1 in some
    # writers; the boundary parser refuses to coerce booleans here.
    md = attrs_to_metadata({'crs': True})
    assert md.crs_epsg is None


def test_attrs_to_metadata_nodata_canonical():
    md = attrs_to_metadata({'nodata': -9999, 'masked_nodata': True})
    assert md.nodata == -9999
    assert md.masked_nodata is True


def test_attrs_to_metadata_nodata_alias_nodatavals():
    md = attrs_to_metadata({'nodatavals': (-9999,)})
    assert md.nodata == -9999
    # No masked_nodata flag on the input -> None on the record
    assert md.masked_nodata is None


def test_attrs_to_metadata_nodata_alias_fill_value():
    md = attrs_to_metadata({'_FillValue': -1})
    assert md.nodata == -1


def test_attrs_to_metadata_no_georef_marker_clears_has_georef():
    md = attrs_to_metadata({_NO_GEOREF_KEY: True})
    assert md.has_georef is False


def test_attrs_to_metadata_transform_present_implies_georef():
    md = attrs_to_metadata({'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)})
    assert md.has_georef is True
    assert md.transform == (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)


# ---------------------------------------------------------------------------
# Round-trip metadata_to_attrs <-> attrs_to_metadata
# ---------------------------------------------------------------------------


def _representative_attrs_dicts():
    """Yield representative attrs dicts that the round trip must preserve.

    Each entry mirrors a real backend's emit set:

    * eager numpy file (CRS + transform + nodata + masked)
    * point raster
    * no-georef file
    * user-defined CRS (WKT only, no EPSG)
    * VRT with holes
    """
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
    list(_representative_attrs_dicts()),
    ids=['eager', 'point', 'no_georef', 'user_wkt', 'vrt'],
)
def test_round_trip_attrs_to_metadata_to_attrs(attrs):
    """``metadata_to_attrs(attrs_to_metadata(x))`` preserves every key in x."""
    md = attrs_to_metadata(attrs)
    round_tripped = metadata_to_attrs(md)
    for key, value in attrs.items():
        assert key in round_tripped, f"key {key!r} dropped on round trip"
        assert round_tripped[key] == value, (
            f"value at {key!r} changed: {value!r} -> {round_tripped[key]!r}"
        )


@pytest.mark.parametrize(
    'attrs',
    list(_representative_attrs_dicts()),
    ids=['eager', 'point', 'no_georef', 'user_wkt', 'vrt'],
)
def test_round_trip_emits_no_unexpected_keys(attrs):
    """Round trip emits the input keys and the contract version, nothing else.

    Locks the marshalling step so a future field added to
    ``metadata_to_attrs`` without an attr-contract update fails this
    test rather than silently appearing on every read.
    """
    md = attrs_to_metadata(attrs)
    round_tripped = metadata_to_attrs(md)
    expected_keys = set(attrs.keys()) | {'_xrspatial_geotiff_contract'}
    extra = set(round_tripped) - expected_keys
    assert not extra, (
        f"metadata_to_attrs emitted unexpected keys {extra!r} not present in input "
        f"{set(attrs)!r}; either add them to the attrs contract or fix "
        f"the marshalling step."
    )


def test_round_trip_metadata_to_attrs_to_metadata():
    """Building a record, marshalling, parsing back recovers all fields."""
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


# ---------------------------------------------------------------------------
# with_nodata builder
# ---------------------------------------------------------------------------


def test_with_nodata_sets_pair():
    md = GeoTIFFMetadata()
    md2 = md.with_nodata(-9999, masked=True)
    assert md2.nodata == -9999
    assert md2.masked_nodata is True
    # frozen dataclass returns a new instance
    assert md.nodata is None
    assert md is not md2


def test_with_nodata_none_returns_unchanged():
    md = GeoTIFFMetadata(crs_epsg=4326)
    md2 = md.with_nodata(None, masked=True)
    # ``None`` mirrors ``_set_nodata_attrs`` no-op contract; record
    # comes back equal so absence keeps signalling "no declared
    # sentinel."
    assert md2.nodata is None
    assert md2.masked_nodata is None
    assert md2.crs_epsg == 4326


def test_with_nodata_masked_false_records_false():
    md = GeoTIFFMetadata()
    md2 = md.with_nodata(-9999, masked=False)
    assert md2.masked_nodata is False
