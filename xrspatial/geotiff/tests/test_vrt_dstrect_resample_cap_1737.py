"""VRT ``DstRect`` xSize/ySize must not drive unbounded resample intermediates.

A crafted VRT can declare a ``<SimpleSource><DstRect>`` whose ``xSize`` and
``ySize`` are orders of magnitude larger than the VRT's own
``rasterXSize`` / ``rasterYSize``. Originally (issue #1737) ``read_vrt``
called ``_resample_nearest(src_arr, dr.y_size, dr.x_size)`` *before* clipping,
allocating the full DstRect-sized intermediate before discarding most of it,
so the read was refused with a ``ValueError`` naming the offending size.

After issue #1704 the resample path reads only the source subset that feeds
the clipped destination sub-window, so the intermediate is bounded by the
caller's window (and by the VRT extent) rather than the raw DstRect. The
huge-DstRect attack vector is therefore neutralised by the read path itself,
not by the per-source pixel-budget guard. The per-source guard still applies
to the clipped sub-window, which is now what the cap is measured against.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._vrt import read_vrt


def _write_source(td: str) -> str:
    """Write a 10x10 uint8 source GeoTIFF and return its path.

    Stripped (non-tiled) so the source read does not allocate a 256x256
    tile that trips ``_check_dimensions`` under the small ``max_pixels``
    values these tests pass.
    """
    src_path = os.path.join(td, 'src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path,
               compression='none', tiled=False)
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


def test_huge_dstrect_no_longer_allocates_full_intermediate():
    """After #1704 the windowed read clips a 50000x50000 DstRect down to
    the 100x100 VRT extent, so the resample intermediate is 100x100 and
    no longer hits the pixel-budget cap. The earlier behaviour rejected
    the read up front; the new behaviour just returns the assembled
    100x100 mosaic.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, dst_x_size=50000, dst_y_size=50000)
        arr, _ = read_vrt(vrt_path)
        assert arr.shape == (100, 100)


def test_huge_dstrect_y_axis_clipped_to_extent():
    """Asymmetric blow-up: ``ySize`` declared as 10 billion but the VRT
    extent caps the clipped sub-window at 100 rows. Read succeeds with
    the bounded intermediate."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(
            td, dst_x_size=10, dst_y_size=10_000_000_000)
        arr, _ = read_vrt(vrt_path)
        assert arr.shape == (100, 100)


def test_legitimate_upsample_still_works():
    """A legitimate upsample stays under the cap and must succeed."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        # 100 x 100 destination, matches the VRT extent.
        vrt_path = _write_vrt(td, dst_x_size=100, dst_y_size=100)
        arr, _ = read_vrt(vrt_path)
        assert arr.shape == (100, 100)


def test_per_source_cap_bites_when_sub_window_exceeds_budget():
    """The per-source pixel-budget guard applies to the clipped
    sub-window, not the raw DstRect. Pick a VRT and ``max_pixels`` where
    the sub-window itself exceeds the cap so the per-source check fires
    even after the windowed-read change.

    The output buffer dimension check (``_check_dimensions``) is also
    bounded by ``max_pixels``, so to isolate the per-source branch we
    request a window whose sub-window product crosses the cap. Both
    guards use the same threshold; the per-source one provides defence
    in depth.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, dst_x_size=2000, dst_y_size=2000,
                              raster_x=2000, raster_y=2000)
        # Sub-window is 2000x2000 = 4e6 pixels.  Cap of 1e6 rejects.
        with pytest.raises(ValueError, match="resample intermediate|safety limit"):
            read_vrt(vrt_path, max_pixels=1_000_000)
        # Bump the cap above 4e6: accepted.
        arr, _ = read_vrt(vrt_path, max_pixels=4_000_000)
        assert arr.shape == (2000, 2000)


def test_per_source_cap_inclusive_boundary():
    """The per-source cap is inclusive: exactly ``max_pixels`` succeeds,
    one below rejects. Mirrors the boundary the original #1737 test
    pinned down, on the new sub-window semantics."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, dst_x_size=100, dst_y_size=100,
                              raster_x=100, raster_y=100)
        # Sub-window is 100x100 = 10_000 pixels.
        with pytest.raises(ValueError, match="resample intermediate|safety limit"):
            read_vrt(vrt_path, max_pixels=9_999)
        arr, _ = read_vrt(vrt_path, max_pixels=10000)
        assert arr.shape == (100, 100)


def test_negative_dstrect_rejected():
    """Negative ``xSize`` / ``ySize`` must surface as ``ValueError``
    rather than be silently skipped by the overlap check.  The error
    message must call out the malformed negative size, not the pixel
    budget."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, dst_x_size=-5, dst_y_size=100)
        with pytest.raises(ValueError, match="negative size"):
            read_vrt(vrt_path)


def test_negative_dstrect_y_size_rejected():
    """Negative ``ySize`` is also rejected with the same tailored error."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, dst_x_size=100, dst_y_size=-5)
        with pytest.raises(ValueError, match="negative size"):
            read_vrt(vrt_path)
