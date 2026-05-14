"""VRT XML reads from the chunked dispatcher must honor the size cap.

Regression test for issue #1831: ``read_vrt(path, chunks=...)`` (added in
#1822) parsed the VRT XML with an unbounded ``open().read()``, bypassing
the 64 MiB cap that #1818 added in ``_vrt._read_vrt_xml``. An attacker
supplying a multi-GB VRT file plus a chunked workflow would exhaust
host memory before any parser-side guard fired.

The eager path is already covered by ``test_vrt_xml_size_cap_1815.py``;
this file pins the same behavior for ``chunks=``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from xrspatial.geotiff import read_vrt, to_geotiff


def _write_source(td: str) -> str:
    src_path = os.path.join(td, 'tmp_1831_src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path,
               compression='none')
    return src_path


def _write_vrt(td: str, *, pad_bytes: int = 0) -> str:
    """Write a VRT, optionally padded with a large XML comment."""
    vrt_path = os.path.join(td, 'tmp_1831_mosaic.vrt')
    comment = ''
    if pad_bytes > 0:
        comment = '<!-- ' + ('x' * pad_bytes) + ' -->\n'
    vrt_xml = (
        '<VRTDataset rasterXSize="10" rasterYSize="10">\n'
        + comment +
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">'
        'tmp_1831_src.tif</SourceFilename>\n'
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


def test_chunked_read_vrt_honors_xml_cap(tmp_path, monkeypatch):
    """``read_vrt(chunks=...)`` rejects oversized VRT XML."""
    td = str(tmp_path)
    _write_source(td)
    # 1 KiB cap, 4 KiB pad. The cap message must reference the env var so
    # operators know how to raise it.
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', '1024')
    vrt_path = _write_vrt(td, pad_bytes=4096)
    with pytest.raises(ValueError) as exc_info:
        read_vrt(vrt_path, chunks=10)
    msg = str(exc_info.value)
    assert 'XRSPATIAL_VRT_MAX_XML_BYTES' in msg
    assert '1,024' in msg


def test_chunked_read_vrt_under_default_cap(tmp_path):
    """A normal-sized VRT parses successfully under the default cap."""
    td = str(tmp_path)
    _write_source(td)
    vrt_path = _write_vrt(td)
    arr = read_vrt(vrt_path, chunks=10)
    assert arr.shape == (10, 10)
    assert arr.dtype == np.uint8


def test_chunked_read_vrt_raised_cap_allows_padded(tmp_path, monkeypatch):
    """Raising ``XRSPATIAL_VRT_MAX_XML_BYTES`` lets a padded VRT parse."""
    td = str(tmp_path)
    _write_source(td)
    vrt_path = _write_vrt(td, pad_bytes=4096)
    monkeypatch.setenv('XRSPATIAL_VRT_MAX_XML_BYTES', str(1024 * 1024))
    arr = read_vrt(vrt_path, chunks=10)
    assert arr.shape == (10, 10)
