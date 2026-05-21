"""Issue #2210 -- ``_validate_read_geo_info`` and
``_finalize_lazy_read_attrs`` forward ``band_nodata`` /
``band_nodata_values``.

Before #2210 these helpers had no per-band-nodata context, so the
``_check_read_mixed_band_metadata`` registry entry dispatched as a
no-op when the VRT call sites routed validation through the shared
helper. The signatures now accept the kwargs and forward them to
``validate_read_metadata`` so a VRT carrying disagreeing per-band
sentinels is rejected by the helper-routed call too.

These tests pin the new kwargs on both helpers directly (no VRT read).
The VRT pre-read inline call already had this context and stays in
place; its post-read counterpart is what these tests cover.
"""
from __future__ import annotations

import inspect

import pytest

from xrspatial.geotiff._attrs import (
    _finalize_lazy_read_attrs,
    _validate_read_geo_info,
)
from xrspatial.geotiff._errors import MixedBandMetadataError


class _FakeTransform:
    def __init__(self, origin_x=0.0, origin_y=10.0,
                 pixel_width=1.0, pixel_height=-1.0,
                 rotated_affine=None):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.rotated_affine = rotated_affine


class _FakeGeoInfo:
    def __init__(self, *, crs_wkt='EPSG:4326', has_georef=True):
        self.transform = _FakeTransform()
        self.crs_epsg = 4326
        self.crs_wkt = crs_wkt
        self.raster_type = 1
        self.has_georef = has_georef
        self.nodata = None
        self.extra_tags = None
        self.image_description = None
        self.extra_samples = None
        self.gdal_metadata = None
        self.gdal_metadata_xml = None
        self.x_resolution = None
        self.y_resolution = None
        self.resolution_unit = None


def test_validate_helper_signature_includes_band_nodata_kwargs():
    sig = inspect.signature(_validate_read_geo_info)
    assert 'band_nodata' in sig.parameters
    assert 'band_nodata_values' in sig.parameters
    # Keyword-only so the wave-1 positional signature stays stable.
    assert sig.parameters['band_nodata'].kind == inspect.Parameter.KEYWORD_ONLY
    assert (sig.parameters['band_nodata_values'].kind
            == inspect.Parameter.KEYWORD_ONLY)


def test_lazy_finalize_signature_includes_band_nodata_kwargs():
    sig = inspect.signature(_finalize_lazy_read_attrs)
    assert 'band_nodata' in sig.parameters
    assert 'band_nodata_values' in sig.parameters
    assert sig.parameters['band_nodata'].kind == inspect.Parameter.KEYWORD_ONLY


def test_validate_helper_rejects_mixed_band_sentinels():
    gi = _FakeGeoInfo()

    with pytest.raises(MixedBandMetadataError):
        _validate_read_geo_info(
            gi,
            band_nodata=None,
            band_nodata_values=[-9999.0, 0.0],
        )


def test_validate_helper_accepts_single_sentinel_band_list():
    # All bands carry the same sentinel -- the check passes.
    gi = _FakeGeoInfo()

    _validate_read_geo_info(
        gi,
        band_nodata=None,
        band_nodata_values=[-9999.0, -9999.0, -9999.0],
    )


def test_validate_helper_band_nodata_first_opts_out():
    # ``band_nodata='first'`` keeps the legacy flatten-to-first-band
    # behaviour explicitly even when the bands disagree.
    gi = _FakeGeoInfo()

    _validate_read_geo_info(
        gi,
        band_nodata='first',
        band_nodata_values=[-9999.0, 0.0],
    )


def test_validate_helper_omits_band_kwargs_short_circuits():
    # Non-VRT callers omit the kwargs entirely; the check short-circuits
    # because ``band_nodata_values`` defaults to ``None`` (falsy).
    gi = _FakeGeoInfo()

    _validate_read_geo_info(gi)  # no kwargs -- no raise


def test_lazy_finalize_routes_band_nodata_through_validator():
    # Mixed-band sentinels routed through the lazy finalizer surface as
    # the same error class the VRT pre-read inline call raises, so the
    # helper-routed post-read check is no longer a no-op (#2210).
    gi = _FakeGeoInfo()

    with pytest.raises(MixedBandMetadataError):
        _finalize_lazy_read_attrs(
            geo_info=gi,
            nodata=None,
            mask_nodata=False,
            graph_dtype='float64',
            window=None,
            band_nodata=None,
            band_nodata_values=[-9999.0, 0.0],
        )


def test_lazy_finalize_band_nodata_first_opts_out():
    # Same as above with the explicit opt-in to legacy semantics. The
    # finalizer returns attrs without raising.
    gi = _FakeGeoInfo()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        window=None,
        band_nodata='first',
        band_nodata_values=[-9999.0, 0.0],
    )
    assert attrs['georef_status'] == 'full'


def test_lazy_finalize_without_band_kwargs_unchanged():
    # The default kwargs match the pre-#2210 behaviour: no mixed-band
    # dispatch is meaningful because ``band_nodata_values`` is None.
    gi = _FakeGeoInfo()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        window=None,
    )
    assert attrs['georef_status'] == 'full'
    assert attrs['nodata'] == -9999
