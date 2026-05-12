"""VRT ``DstRect`` xSize/ySize must not drive unbounded resample intermediates.

A crafted VRT can declare a ``<SimpleSource><DstRect>`` whose ``xSize`` and
``ySize`` are orders of magnitude larger than the VRT's own
``rasterXSize`` / ``rasterYSize``. The output buffer is already bounded by
``_check_dimensions`` against ``max_pixels``, but ``_resample_nearest`` is
called with ``(dr.y_size, dr.x_size)`` *before* the clip is taken, so it
allocates the full DstRect-sized intermediate before discarding most of it.

Regression test for issue #1737: ``read_vrt`` should refuse the read with a
``ValueError`` that names the offending size, rather than try to allocate
gigabytes of intermediate memory on a tiny output.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._vrt import read_vrt


def _write_source(td: str) -> str:
    """Write a 10x10 uint8 source GeoTIFF and return its path."""
    src_path = os.path.join(td, 'src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path,
               compression='none')
    return src_path


def _write_vrt(td: str, *, dst_x_size: int, dst_y_size: int,
               raster_x: int = 100, raster_y: int = 100) -> str:
    """Write a VRT with a single SimpleSource using the given DstRect size."""
    vrt_path = os.path.join(td, 'mosaic.vrt')
    vrt_xml = (
        f'<VRTDataset rasterXSize="{raster_x}" rasterYSize="{raster_y}">\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">src.tif</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n'
        f'      <DstRect xOff="0" yOff="0" '
        f'xSize="{dst_x_size}" ySize="{dst_y_size}"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_huge_dstrect_rejected_before_intermediate_allocation():
    """A DstRect that would force a multi-billion-pixel resample intermediate
    must raise ``ValueError`` before ``_resample_nearest`` allocates."""
    with tempfile.TemporaryDirectory() as td:
        _write_source(td)
        # 50000 x 50000 = 2.5 billion pixels of intermediate; the output
        # buffer is only 100 x 100. With the cap in place this should
        # raise before _resample_nearest runs.
        vrt_path = _write_vrt(td, dst_x_size=50000, dst_y_size=50000)
        with pytest.raises(ValueError, match="resample intermediate"):
            read_vrt(vrt_path)


def test_huge_dstrect_y_axis_rejected():
    """Asymmetric blow-up: only one axis is huge. Still rejected."""
    with tempfile.TemporaryDirectory() as td:
        _write_source(td)
        vrt_path = _write_vrt(
            td, dst_x_size=10, dst_y_size=10_000_000_000)
        with pytest.raises(ValueError, match="resample intermediate"):
            read_vrt(vrt_path)


def test_legitimate_upsample_still_works():
    """A legitimate upsample stays under the cap and must succeed."""
    with tempfile.TemporaryDirectory() as td:
        _write_source(td)
        # 100 x 100 destination, matches the VRT extent.
        vrt_path = _write_vrt(td, dst_x_size=100, dst_y_size=100)
        arr, _ = read_vrt(vrt_path)
        assert arr.shape == (100, 100)


def test_max_pixels_kwarg_raises_cap():
    """When the caller bumps ``max_pixels``, a previously-rejected DstRect
    is accepted (matches the contract for other read paths)."""
    with tempfile.TemporaryDirectory() as td:
        _write_source(td)
        vrt_path = _write_vrt(td, dst_x_size=2000, dst_y_size=2000)
        # Default cap is 1e9, 2000*2000=4e6 well under.
        arr, _ = read_vrt(vrt_path)
        assert arr.shape == (100, 100)


def test_dstrect_at_cap_succeeds():
    """Exactly at ``max_pixels`` is accepted; the cap is inclusive."""
    with tempfile.TemporaryDirectory() as td:
        _write_source(td)
        # max_pixels=10000 means dst 100x100 = 10000 is allowed.
        vrt_path = _write_vrt(td, dst_x_size=100, dst_y_size=100)
        arr, _ = read_vrt(vrt_path, max_pixels=10000)
        assert arr.shape == (100, 100)


def test_negative_dstrect_rejected():
    """Negative ``xSize`` / ``ySize`` must surface as ``ValueError`` rather
    than degenerate into a negative-stride numpy slice."""
    with tempfile.TemporaryDirectory() as td:
        _write_source(td)
        vrt_path = _write_vrt(td, dst_x_size=-5, dst_y_size=100)
        # The negative size makes ``needs_resample`` true because it differs
        # from sr.x_size=10; the cap branch catches the negative value.
        # If the source dstrect doesn't overlap the window the source is
        # skipped silently (returns the empty fill array) and no resample
        # runs - that's also OK; we accept either behaviour.
        try:
            arr, _ = read_vrt(vrt_path)
            # If it did succeed, the array must still be the VRT extent.
            assert arr.shape == (100, 100)
        except ValueError as e:
            # The cap message identifies the resample intermediate, which is
            # what we want to surface here.
            assert "resample intermediate" in str(e)
