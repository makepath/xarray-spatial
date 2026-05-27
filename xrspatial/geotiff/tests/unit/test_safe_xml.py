"""_build_gdal_metadata_xml escapes XML special chars.

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

from xrspatial.geotiff._geotags import _build_gdal_metadata_xml, _parse_gdal_metadata


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


# =============================================================================
# Section: Safe extra_tags filter
# =============================================================================
#
# Original: ``test_extra_tags_safe_filter_1657.py``.
#
# Before the fix, reading an overview level (NewSubfileType=1) or any
# TIFF with a SubIFDs entry leaked those tags into
# ``attrs['extra_tags']`` because they were not in ``_MANAGED_TAGS``.
# Writing the DataArray back via ``to_geotiff`` or
# ``write_geotiff_gpu`` then re-emitted them on the output IFD,
# producing:
#
# * A primary IFD wrongly marked as a reduced-resolution overview
#   (``NewSubfileType=1``), so GDAL / rasterio skip it when picking the
#   primary image.
# * Stale absolute byte offsets in ``SubIFDs`` that point into the new
#   file's pixel data, crashing readers that follow the chain.
#
# The fix adds both tags to ``_MANAGED_TAGS`` (read-side filter) and to
# ``_DANGEROUS_EXTRA_TAG_IDS`` in ``_writer.py`` (write-side
# belt-and-braces guard so caller-supplied ``attrs['extra_tags']``
# still produces a clean file).
#
# These tests pin the contract on every available backend.
from xrspatial.geotiff import open_geotiff, to_geotiff  # noqa: E402

from .._helpers.markers import requires_gpu as _requires_gpu_1657  # noqa: E402

tifffile_1657 = pytest.importorskip("tifffile")


def _make_cog_1657(path) -> None:
    """Write a small COG with overviews so each backend can read an overview."""
    import xarray as xr
    da = xr.DataArray(
        np.arange(512 * 512, dtype=np.float32).reshape(512, 512),
        dims=['y', 'x'],
        coords={'y': np.arange(512) * -0.5 + 10.0,
                'x': np.arange(512) * 0.5 - 10.0},
        attrs={'crs': 4326},
    )
    to_geotiff(da, str(path), cog=True,
               tile_size=64, overview_levels=[2, 4])


def _read_subfile_type_1657(path) -> int | None:
    """Return the NewSubfileType value on page 0 of a TIFF, or None if absent."""
    with tifffile_1657.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        tag = page.tags.get('NewSubfileType')
        return None if tag is None else int(tag.value)


def test_overview_read_does_not_leak_newsubfiletype_numpy_1657(tmp_path):
    cog_path = tmp_path / 'cog.tif'
    _make_cog_1657(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1)
    extra = ov.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 254 for t in extra), (
        f"NewSubfileType (tag 254) leaked into attrs['extra_tags']: {extra}"
    )


def test_overview_read_does_not_leak_newsubfiletype_dask_1657(tmp_path):
    cog_path = tmp_path / 'cog.tif'
    _make_cog_1657(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1, chunks=32)
    extra = ov.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 254 for t in extra), (
        f"NewSubfileType (tag 254) leaked under dask: {extra}"
    )


@_requires_gpu_1657
def test_overview_read_does_not_leak_newsubfiletype_cupy_1657(tmp_path):
    cog_path = tmp_path / 'cog.tif'
    _make_cog_1657(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1, gpu=True)
    extra = ov.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 254 for t in extra), (
        f"NewSubfileType (tag 254) leaked under cupy: {extra}"
    )


@_requires_gpu_1657
def test_overview_read_does_not_leak_newsubfiletype_dask_cupy_1657(tmp_path):
    cog_path = tmp_path / 'cog.tif'
    _make_cog_1657(cog_path)
    ov = open_geotiff(
        str(cog_path), overview_level=1, gpu=True, chunks=32)
    extra = ov.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 254 for t in extra), (
        f"NewSubfileType (tag 254) leaked under dask+cupy: {extra}"
    )


def test_subifds_does_not_leak_into_attrs_1657(tmp_path):
    """tifffile writes SubIFDs by default on multi-page TIFFs.

    Anything carrying tag 330 must not surface in ``attrs['extra_tags']``
    because the byte offsets are file-absolute and cannot be replayed
    into a rewritten file.
    """
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = tmp_path / 'subifd.tif'
    with tifffile_1657.TiffWriter(str(path)) as tw:
        tw.write(data, tile=(32, 32), subifds=1)
        tw.write(data[::2, ::2], tile=(32, 32), subfiletype=1)

    da = open_geotiff(str(path))
    extra = da.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 330 for t in extra), (
        f"SubIFDs (tag 330) leaked into attrs['extra_tags']: {extra}"
    )


def test_overview_roundtrip_primary_ifd_clean_numpy_1657(tmp_path):
    cog_path = tmp_path / 'cog.tif'
    _make_cog_1657(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1)
    out = tmp_path / 'out_numpy.tif'
    to_geotiff(ov, str(out))
    sft = _read_subfile_type_1657(out)
    assert sft in (None, 0), (
        f"Round-tripped overview produced NewSubfileType={sft} on the "
        f"primary IFD (expected None or 0)."
    )


def test_overview_roundtrip_primary_ifd_clean_dask_1657(tmp_path):
    cog_path = tmp_path / 'cog.tif'
    _make_cog_1657(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1, chunks=32)
    out = tmp_path / 'out_dask.tif'
    to_geotiff(ov, str(out))
    sft = _read_subfile_type_1657(out)
    assert sft in (None, 0), (
        f"Dask round-tripped overview produced NewSubfileType={sft}."
    )


@_requires_gpu_1657
def test_overview_roundtrip_primary_ifd_clean_cupy_1657(tmp_path):
    cog_path = tmp_path / 'cog.tif'
    _make_cog_1657(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1, gpu=True)
    out = tmp_path / 'out_cupy.tif'
    to_geotiff(ov, str(out))
    sft = _read_subfile_type_1657(out)
    assert sft in (None, 0), (
        f"Cupy round-tripped overview produced NewSubfileType={sft}."
    )


def test_writer_filters_caller_supplied_newsubfiletype_1657(tmp_path):
    import xarray as xr
    da = xr.DataArray(
        np.zeros((32, 32), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.arange(32) * -0.5 + 10.0,
                'x': np.arange(32) * 0.5 - 10.0},
        attrs={
            'crs': 4326,
            'extra_tags': [(254, 4, 1, 1)],
        },
    )
    out = tmp_path / 'with_dangerous_extra_tag.tif'
    to_geotiff(da, str(out), allow_experimental_codecs=True)
    sft = _read_subfile_type_1657(out)
    assert sft in (None, 0), (
        f"Writer accepted dangerous extra_tags[254]={sft}, expected None/0."
    )


def test_writer_filters_caller_supplied_subifds_1657(tmp_path):
    import xarray as xr
    da = xr.DataArray(
        np.zeros((32, 32), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.arange(32) * -0.5 + 10.0,
                'x': np.arange(32) * 0.5 - 10.0},
        attrs={
            'crs': 4326,
            'extra_tags': [(330, 4, 2, (999999, 888888))],
        },
    )
    out = tmp_path / 'with_subifds.tif'
    to_geotiff(da, str(out), allow_experimental_codecs=True)
    with tifffile_1657.TiffFile(str(out)) as tf:
        sub = tf.pages[0].tags.get('SubIFDs')
        assert sub is None, (
            f"Writer emitted SubIFDs={sub.value}, should have filtered it."
        )


def test_writer_keeps_benign_extra_tags_1657(tmp_path):
    import xarray as xr
    da = xr.DataArray(
        np.zeros((32, 32), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.arange(32) * -0.5 + 10.0,
                'x': np.arange(32) * 0.5 - 10.0},
        attrs={
            'crs': 4326,
            'extra_tags': [
                (254, 4, 1, 1),
                (305, 2, 12, 'tifffile.py'),
            ],
        },
    )
    out = tmp_path / 'mixed_extra_tags.tif'
    to_geotiff(da, str(out), allow_experimental_codecs=True)
    with tifffile_1657.TiffFile(str(out)) as tf:
        page = tf.pages[0]
        assert page.tags.get('NewSubfileType') is None
        software = page.tags.get('Software')
        assert software is not None, (
            "Benign extra_tag (305 Software) was filtered too -- "
            "filter is too aggressive."
        )
        assert 'tifffile' in str(software.value)
