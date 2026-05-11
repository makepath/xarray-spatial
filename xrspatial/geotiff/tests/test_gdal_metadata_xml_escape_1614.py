"""Tests for #1614: _build_gdal_metadata_xml escapes XML special chars.

The serialiser previously interpolated keys and values with plain
f-strings, which corrupted the document on any value containing
``& < > " '`` and let a caller-crafted key inject extra attributes
into the ``<Item>`` element. This file pins the escaped behaviour:
all five XML special characters round-trip through write/read
without loss, and attribute slots refuse injection attempts.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from xrspatial.geotiff._geotags import (
    _build_gdal_metadata_xml,
    _parse_gdal_metadata,
)


class TestBuildGdalMetadataXMLEscape:
    """Verify each XML special character round-trips intact."""

    @pytest.mark.parametrize(
        "payload",
        [
            "A & B",
            "value < 5",
            "value > 5",
            'quote "x"',
            "apos 'x'",
            "&<>\"'",            # all five at once
            "no specials here",  # control: plain text still works
        ],
    )
    def test_value_round_trip_for_special_chars(self, payload):
        meta = {"NOTE": payload}
        xml = _build_gdal_metadata_xml(meta)
        # Document parses as well-formed XML
        ET.fromstring(xml)
        # Round-trip via the package's own parser reproduces the value
        assert _parse_gdal_metadata(xml) == meta

    @pytest.mark.parametrize(
        "payload",
        [
            "A & B",
            "value < 5",
            'quote "x"',
        ],
    )
    def test_per_band_value_round_trip(self, payload):
        meta = {("STAT", 0): payload}
        xml = _build_gdal_metadata_xml(meta)
        ET.fromstring(xml)
        assert _parse_gdal_metadata(xml) == meta

    def test_attribute_injection_via_name_blocked(self):
        """A key containing a quote cannot inject extra attributes."""
        injection_key = 'foo" malicious="bar'
        meta = {injection_key: "value"}
        xml = _build_gdal_metadata_xml(meta)

        root = ET.fromstring(xml)
        items = root.findall("Item")
        assert len(items) == 1
        # Only the legitimate ``name`` attribute exists; the injection
        # attempt did not create a second attribute.
        assert set(items[0].attrib.keys()) == {"name"}
        assert items[0].attrib["name"] == injection_key
        assert items[0].text == "value"
        # Round-trip reproduces the original key intact.
        assert _parse_gdal_metadata(xml) == meta

    def test_element_injection_via_value_blocked(self):
        """A value with closing-tag syntax cannot inject a sibling Item."""
        meta = {"OUTER": "</Item><Item name=\"X\">stuff</Item><!--"}
        xml = _build_gdal_metadata_xml(meta)

        root = ET.fromstring(xml)
        items = root.findall("Item")
        # Only the one Item we wrote -- the injection payload sits inside
        # its text content, not as a new element.
        assert len(items) == 1
        assert items[0].attrib == {"name": "OUTER"}
        assert _parse_gdal_metadata(xml) == meta

    def test_unicode_value_unaffected(self):
        """Non-ASCII text passes through unchanged (only XML specials escape)."""
        meta = {"DESC": "élévation ☃"}
        xml = _build_gdal_metadata_xml(meta)
        ET.fromstring(xml)
        assert _parse_gdal_metadata(xml) == meta

    def test_existing_clean_string_format_unchanged(self):
        """Plain keys/values still emit the original double-quoted form."""
        xml = _build_gdal_metadata_xml({"DataType": "Generic"})
        # quoteattr picks double quotes when the string has no double
        # quote of its own, so the legacy assertion ``name="DataType"``
        # still holds and downstream tools see the same output.
        assert '<Item name="DataType">Generic</Item>' in xml


class TestToGeotiffMetadataRoundTrip:
    """End-to-end: special-char metadata survives a write/read cycle."""

    def test_to_geotiff_special_chars_round_trip(self, tmp_path):
        import xarray as xr
        from xrspatial.geotiff import open_geotiff, to_geotiff

        arr = np.ones((4, 4), dtype=np.float32)
        da = xr.DataArray(arr, dims=["y", "x"])
        da.attrs["gdal_metadata"] = {
            "NOTE": "value < 5 & ok",
            ("STAT", 0): 'quote "x"',
        }

        path = tmp_path / "meta.tif"
        to_geotiff(da, str(path))

        opened = open_geotiff(str(path))
        round_tripped = opened.attrs.get("gdal_metadata")
        assert round_tripped is not None, opened.attrs
        # Both dataset-level and per-band entries survive intact.
        assert round_tripped.get("NOTE") == "value < 5 & ok"
        assert round_tripped.get(("STAT", 0)) == 'quote "x"'
