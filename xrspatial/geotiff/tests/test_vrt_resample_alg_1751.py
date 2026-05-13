"""Regression tests for issue #1751.

``read_vrt`` parses ``<ResampleAlg>`` from a ``ComplexSource`` but always
calls ``_resample_nearest`` during placement, regardless of the parsed
algorithm.  A VRT declaring ``Bilinear`` / ``Cubic`` / ``Average`` /
``Mode`` therefore receives nearest-sampled pixels mislabelled as the
requested algorithm -- silently wrong analytics.

The fix raises ``NotImplementedError`` at the resample call site when
the source declares an unsupported algorithm and the SrcRect/DstRect
sizes actually differ (a 1:1 placement is nearest-equivalent and stays
permissive).  Nearest, NearestNeighbour, NEAR, empty, and an absent
``<ResampleAlg>`` element are all accepted.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._vrt import read_vrt
from xrspatial.geotiff._writer import write


def _write_src(tmp_path) -> str:
    """Write a 4x4 uint16 source TIFF and return its path."""
    src = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)
    return src_path


def _write_vrt(tmp_path, xml: str, name: str = 'test.vrt') -> str:
    p = str(tmp_path / name)
    with open(p, 'w') as f:
        f.write(xml)
    return p


def _vrt_xml(src_path: str, *, alg_elem: str,
             dst_x: int = 2, dst_y: int = 2) -> str:
    """Render a VRT XML with a 4x4 SrcRect and configurable DstRect+Alg.

    ``alg_elem`` is the raw ``<ResampleAlg>...</ResampleAlg>`` element
    to splice into the ``<ComplexSource>``, or the empty string to
    omit it entirely.
    """
    return f"""<VRTDataset rasterXSize="{dst_x}" rasterYSize="{dst_y}">
  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="{dst_x}" ySize="{dst_y}"/>
      {alg_elem}
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"""


@pytest.mark.parametrize('alg', ['Bilinear', 'Cubic', 'CubicSpline',
                                 'Lanczos', 'Average', 'Mode'])
def test_unsupported_resample_alg_raises(tmp_path, alg):
    """A ComplexSource declaring any non-nearest algorithm with a size
    change must raise ``NotImplementedError`` rather than return
    silently nearest-sampled pixels."""
    src_path = _write_src(tmp_path)
    xml = _vrt_xml(src_path,
                   alg_elem=f'<ResampleAlg>{alg}</ResampleAlg>')
    vrt_path = _write_vrt(tmp_path, xml, f'{alg.lower()}.vrt')

    with pytest.raises(NotImplementedError) as excinfo:
        read_vrt(vrt_path)
    msg = str(excinfo.value)
    assert alg in msg
    assert '1751' in msg


def test_unsupported_resample_alg_case_insensitive(tmp_path):
    """Algorithm names are matched case-insensitively: ``bilinear``
    (lowercase) is the same unsupported request as ``Bilinear``."""
    src_path = _write_src(tmp_path)
    xml = _vrt_xml(src_path,
                   alg_elem='<ResampleAlg>bilinear</ResampleAlg>')
    vrt_path = _write_vrt(tmp_path, xml, 'lower.vrt')

    with pytest.raises(NotImplementedError, match='bilinear'):
        read_vrt(vrt_path)


@pytest.mark.parametrize('alg', ['Nearest', 'NearestNeighbour',
                                 'NearestNeighbor', 'NEAR', 'nearest',
                                 'NEAREST', ''])
def test_nearest_variants_accepted(tmp_path, alg):
    """Nearest (and its case / spelling variants, plus empty text) is
    the implemented algorithm and must round-trip without raising."""
    src_path = _write_src(tmp_path)
    xml = _vrt_xml(src_path,
                   alg_elem=f'<ResampleAlg>{alg}</ResampleAlg>')
    vrt_path = _write_vrt(tmp_path, xml, f'near_{alg or "empty"}.vrt')

    arr, _ = read_vrt(vrt_path)
    assert arr.shape == (2, 2)


def test_missing_resample_alg_accepted(tmp_path):
    """Absent ``<ResampleAlg>`` (GDAL's nearest default) must still
    round-trip without raising."""
    src_path = _write_src(tmp_path)
    xml = _vrt_xml(src_path, alg_elem='')
    vrt_path = _write_vrt(tmp_path, xml, 'absent.vrt')

    arr, _ = read_vrt(vrt_path)
    assert arr.shape == (2, 2)


def test_bilinear_at_same_size_does_not_raise(tmp_path):
    """A ``Bilinear`` declaration with matching SrcRect/DstRect sizes
    is nearest-equivalent (no resample step runs) so the read is
    accepted.  This pins down the resample-site placement of the
    check -- a parse-time check would have rejected this case too."""
    src_path = _write_src(tmp_path)
    # SrcRect 4x4, DstRect 4x4: needs_resample is False, no kernel
    # call, no silent wrongness.
    xml = _vrt_xml(src_path,
                   alg_elem='<ResampleAlg>Bilinear</ResampleAlg>',
                   dst_x=4, dst_y=4)
    vrt_path = _write_vrt(tmp_path, xml, 'bilinear_1to1.vrt')

    arr, _ = read_vrt(vrt_path)
    assert arr.shape == (4, 4)
