"""Regression tests for issue #1607: write_vrt must XML-escape
caller-supplied text (CRS WKT, source filenames) so a value carrying
XML special characters cannot break the generated VRT or inject extra
elements that change how the VRT parses when read back.
"""
from __future__ import annotations

import os
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._vrt import write_vrt, parse_vrt


@pytest.fixture
def sample_tif(tmp_path):
    """Write a tiny GeoTIFF the VRT writer can introspect for metadata."""
    arr = np.zeros((4, 4), dtype=np.float32)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'nodata': -9999.0},
    )
    path = str(tmp_path / 'src.tif')
    to_geotiff(da, path)
    return path


def test_crs_wkt_with_xml_special_chars_round_trips(sample_tif, tmp_path):
    """A WKT containing ``& < > " '`` must round-trip through write_vrt /
    parse_vrt unchanged (the entities are escaped on the way out and
    decoded on the way in)."""
    nasty_wkt = 'GEOGCS["spec & <chars> with \"quotes\" and \'apostrophes\'"]'
    vrt_path = str(tmp_path / 'mosaic.vrt')
    write_vrt(vrt_path, [sample_tif], crs_wkt=nasty_wkt)

    with open(vrt_path, 'r') as fh:
        text = fh.read()

    parsed = parse_vrt(text, vrt_dir=str(tmp_path))
    assert parsed.crs_wkt == nasty_wkt


def test_crs_wkt_injection_does_not_change_raster_type(sample_tif, tmp_path):
    """The headline #1607 case: a crafted WKT trying to close ``<SRS>``
    and inject ``<Metadata><MDI key="AREA_OR_POINT">Point</MDI>...``
    must NOT change ``raster_type`` from its default 'area' value."""
    injection = (
        '</SRS><Metadata><MDI key="AREA_OR_POINT">Point</MDI>'
        '</Metadata><SRS>'
    )
    vrt_path = str(tmp_path / 'evil.vrt')
    write_vrt(vrt_path, [sample_tif], crs_wkt=injection)

    with open(vrt_path, 'r') as fh:
        text = fh.read()

    parsed = parse_vrt(text, vrt_dir=str(tmp_path))
    assert parsed.raster_type == 'area'
    # And the injection round-trips as literal text inside <SRS>.
    assert parsed.crs_wkt == injection


def test_source_filename_with_ampersand_round_trips(tmp_path):
    """A source filename containing ``&`` must produce a VRT whose
    ``<SourceFilename>`` element decodes back to the original on-disk
    path (no double-escape, no corruption)."""
    # Build a TIFF on disk whose filename has an ampersand.
    arr = np.zeros((4, 4), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(1, 0, 4), 'x': np.linspace(0, 1, 4)},
        attrs={'nodata': -9999.0},
    )
    src = str(tmp_path / 'a&b.tif')
    to_geotiff(da, src)

    vrt_path = str(tmp_path / 'mosaic.vrt')
    write_vrt(vrt_path, [src])

    with open(vrt_path, 'r') as fh:
        text = fh.read()
    # The on-disk text must carry the escaped form, not the raw '&'.
    assert '&amp;' in text
    assert '<a&b' not in text  # no raw ampersand inside a tag context

    parsed = parse_vrt(text, vrt_dir=str(tmp_path))
    # Exactly one source recorded, pointing at the original path
    # (after the parser's realpath canonicalisation).
    assert len(parsed.bands) == 1
    assert len(parsed.bands[0].sources) == 1
    assert os.path.basename(parsed.bands[0].sources[0].filename) == 'a&b.tif'


def test_written_vrt_is_well_formed_xml(sample_tif, tmp_path):
    """Sanity check: the bytes written by write_vrt always parse cleanly
    as XML, even when crs_wkt carries every XML predefined entity."""
    nasty = '< & > " \''
    vrt_path = str(tmp_path / 'wf.vrt')
    write_vrt(vrt_path, [sample_tif], crs_wkt=nasty)

    # Use stdlib ElementTree (not parse_vrt) so we exercise the
    # well-formedness check directly without the DOCTYPE-rejection layer.
    import xml.etree.ElementTree as ET
    with open(vrt_path, 'r') as fh:
        ET.fromstring(fh.read())  # raises on malformed XML
