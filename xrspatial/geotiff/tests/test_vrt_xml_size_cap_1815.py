"""VRT XML reads must be bounded to avoid unbounded memory allocation.

Regression test for issue #1815: ``read_vrt`` previously called
``f.read()`` on the VRT XML path with no size limit, so a multi-gigabyte
XML file would consume all that memory before parsing. The fix adds a
64 MiB default cap, overridable via ``XRSPATIAL_VRT_MAX_XML_BYTES``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._vrt import read_vrt


def _write_source(td: str) -> str:
    src_path = os.path.join(td, 'tmp_1815_src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path,
               compression='none')
    return src_path


def _write_vrt(td: str, *, pad_bytes: int = 0) -> str:
    """Write a VRT, optionally padded with a large XML comment."""
    vrt_path = os.path.join(td, 'tmp_1815_mosaic.vrt')
    comment = ''
    if pad_bytes > 0:
        comment = '<!-- ' + ('x' * pad_bytes) + ' -->\n'
    vrt_xml = (
        '<VRTDataset rasterXSize="10" rasterYSize="10">\n'
        + comment +
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">'
        'tmp_1815_src.tif</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_small_vrt_parses_under_default_cap(tmp_path):
    """A normal-sized VRT parses successfully with the default cap."""
    td = str(tmp_path)
    _write_source(td)
    vrt_path = _write_vrt(td)
    arr, _ = read_vrt(vrt_path)
    assert arr.shape == (10, 10)


def test_oversized_vrt_raises_value_error(tmp_path, monkeypatch):
    """A VRT padded past the cap raises ValueError naming the cap and env var."""
    td = str(tmp_path)
    _write_source(td)
    # Set a small cap (1 KiB) and pad the comment past it.
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', '1024')
    vrt_path = _write_vrt(td, pad_bytes=4096)
    with pytest.raises(ValueError) as exc_info:
        read_vrt(vrt_path)
    msg = str(exc_info.value)
    assert 'XRSPATIAL_VRT_MAX_XML_BYTES' in msg
    assert '1,024' in msg


def test_raising_cap_lets_padded_vrt_parse(tmp_path, monkeypatch):
    """Setting the env var higher allows a padded VRT to parse."""
    td = str(tmp_path)
    _write_source(td)
    vrt_path = _write_vrt(td, pad_bytes=4096)
    # Default cap of 64 MiB is more than enough; verify with an explicit
    # higher cap too.
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', str(1024 * 1024))
    arr, _ = read_vrt(vrt_path)
    assert arr.shape == (10, 10)


@pytest.mark.parametrize('bad_value', ['not_a_number', '0', '-1', '-1024'])
def test_invalid_cap_raises_value_error(tmp_path, monkeypatch, bad_value):
    """Non-numeric, zero, or negative cap values produce a clear error."""
    td = str(tmp_path)
    _write_source(td)
    vrt_path = _write_vrt(td)
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', bad_value)
    with pytest.raises(ValueError, match='XRSPATIAL_VRT_MAX_XML_BYTES'):
        read_vrt(vrt_path)
