"""Regression tests for issue #1704.

When a VRT ``<SimpleSource>`` has a ``<SrcRect>`` size that differs from
its ``<DstRect>`` size (i.e. the source feeds an up- or downsample into
its destination cell), ``read_vrt`` used to decode the full SrcRect from
disk even when the caller passed a tiny ``window=`` clipping into the
middle. For very large source rects this caused multi-GB decodes and
resample intermediates.

The fix inverts the nearest-neighbour mapping: for the clipped
destination sub-window, compute the smallest SrcRect-relative range of
rows / cols that ``_resample_nearest`` would gather from, read only
that, and resample directly into the sub-window output. The result is
byte-identical to the old "read full, resample full, then slice" path.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from xrspatial.geotiff._vrt import read_vrt
from xrspatial.geotiff._writer import write


def _write_vrt_xml(tmp_path, xml: str, name: str) -> str:
    p = str(tmp_path / name)
    with open(p, 'w') as f:
        f.write(xml)
    return p


def _write_src(tmp_path, arr: np.ndarray, name: str = 'tmp_1704_src.tif') -> str:
    src_path = str(tmp_path / name)
    write(arr, src_path, compression='none', tiled=False)
    return src_path


def _single_source_vrt(src_path: str, *,
                       raster_x: int, raster_y: int,
                       src_x: int, src_y: int,
                       src_xsize: int, src_ysize: int,
                       dst_x: int, dst_y: int,
                       dst_xsize: int, dst_ysize: int,
                       dtype: str = "UInt16",
                       nodata: str | None = None) -> str:
    nodata_xml = f"      <NODATA>{nodata}</NODATA>\n" if nodata is not None else ""
    return (
        f'<VRTDataset rasterXSize="{raster_x}" rasterYSize="{raster_y}">\n'
        f'  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
        f'  <VRTRasterBand dataType="{dtype}" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="{src_x}" yOff="{src_y}" '
        f'xSize="{src_xsize}" ySize="{src_ysize}"/>\n'
        f'      <DstRect xOff="{dst_x}" yOff="{dst_y}" '
        f'xSize="{dst_xsize}" ySize="{dst_ysize}"/>\n'
        f'{nodata_xml}'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )


# ---------------------------------------------------------------------------
# Parity: byte-identical to full-then-slice for upsample and downsample
# ---------------------------------------------------------------------------

def test_upsample_window_matches_full_then_slice(tmp_path):
    """4x upsample, then read a small window from the middle. The
    windowed read must equal the full read sliced at the same offsets."""
    src = (np.arange(10 * 10, dtype=np.uint16).reshape(10, 10) + 1)
    src_path = _write_src(tmp_path, src)
    vrt_xml = _single_source_vrt(
        src_path,
        raster_x=40, raster_y=40,
        src_x=0, src_y=0, src_xsize=10, src_ysize=10,
        dst_x=0, dst_y=0, dst_xsize=40, dst_ysize=40,
    )
    vrt_path = _write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_up.vrt')

    full, _ = read_vrt(vrt_path)
    assert full.shape == (40, 40)
    # Pick a middle clip; covers a region with non-trivial neighbour mapping.
    r0, c0, r1, c1 = 7, 11, 33, 29
    windowed, _ = read_vrt(vrt_path, window=(r0, c0, r1, c1))
    np.testing.assert_array_equal(windowed, full[r0:r1, c0:c1])


def test_downsample_window_matches_full_then_slice(tmp_path):
    """4x downsample, windowed read parity with full-then-slice."""
    src = (np.arange(40 * 40, dtype=np.uint16).reshape(40, 40) + 1)
    src_path = _write_src(tmp_path, src)
    vrt_xml = _single_source_vrt(
        src_path,
        raster_x=10, raster_y=10,
        src_x=0, src_y=0, src_xsize=40, src_ysize=40,
        dst_x=0, dst_y=0, dst_xsize=10, dst_ysize=10,
    )
    vrt_path = _write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_down.vrt')

    full, _ = read_vrt(vrt_path)
    assert full.shape == (10, 10)
    r0, c0, r1, c1 = 2, 3, 9, 8
    windowed, _ = read_vrt(vrt_path, window=(r0, c0, r1, c1))
    np.testing.assert_array_equal(windowed, full[r0:r1, c0:c1])


# ---------------------------------------------------------------------------
# Non-integer ratio: floor-rounding edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r0,c0,r1,c1", [
    (0, 0, 11, 11),     # full extent
    (1, 1, 10, 10),     # inner shrink
    (3, 2, 7, 9),       # off-centre
    (0, 0, 1, 1),       # single pixel at origin
    (10, 10, 11, 11),   # single pixel at corner
    (5, 0, 6, 11),      # one-row strip
    (0, 5, 11, 6),      # one-col strip
])
def test_non_integer_ratio_7_to_11_window_parity(tmp_path, r0, c0, r1, c1):
    """SrcRect 7x7, DstRect 11x11 (irrational ratio 7/11). The
    nearest-neighbour mapping has uneven step sizes so the inverse
    mapping has to handle each output index individually; this is the
    case that breaks Option-2 "resample sub-shape into sub-shape"
    implementations.
    """
    src = (np.arange(7 * 7, dtype=np.uint16).reshape(7, 7) + 100)
    src_path = _write_src(tmp_path, src)
    vrt_xml = _single_source_vrt(
        src_path,
        raster_x=11, raster_y=11,
        src_x=0, src_y=0, src_xsize=7, src_ysize=7,
        dst_x=0, dst_y=0, dst_xsize=11, dst_ysize=11,
    )
    vrt_path = _write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_7_11.vrt')

    full, _ = read_vrt(vrt_path)
    windowed, _ = read_vrt(vrt_path, window=(r0, c0, r1, c1))
    np.testing.assert_array_equal(windowed, full[r0:r1, c0:c1])


# ---------------------------------------------------------------------------
# Edge alignment
# ---------------------------------------------------------------------------

def test_window_starting_at_origin(tmp_path):
    src = (np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) + 1)
    src_path = _write_src(tmp_path, src)
    vrt_xml = _single_source_vrt(
        src_path,
        raster_x=20, raster_y=20,
        src_x=0, src_y=0, src_xsize=8, src_ysize=8,
        dst_x=0, dst_y=0, dst_xsize=20, dst_ysize=20,
    )
    vrt_path = _write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_origin.vrt')
    full, _ = read_vrt(vrt_path)
    windowed, _ = read_vrt(vrt_path, window=(0, 0, 5, 5))
    np.testing.assert_array_equal(windowed, full[0:5, 0:5])


def test_window_ending_at_last_pixel(tmp_path):
    src = (np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) + 1)
    src_path = _write_src(tmp_path, src)
    vrt_xml = _single_source_vrt(
        src_path,
        raster_x=20, raster_y=20,
        src_x=0, src_y=0, src_xsize=8, src_ysize=8,
        dst_x=0, dst_y=0, dst_xsize=20, dst_ysize=20,
    )
    vrt_path = _write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_last.vrt')
    full, _ = read_vrt(vrt_path)
    windowed, _ = read_vrt(vrt_path, window=(15, 15, 20, 20))
    np.testing.assert_array_equal(windowed, full[15:20, 15:20])


def test_window_crossing_multiple_sources(tmp_path):
    """Two SimpleSources tiled side by side, each with non-1:1 SrcRect /
    DstRect. A window that spans both sources must equal the full read
    sliced over the same range. Both sources go through the new windowed
    resample path.
    """
    left = (np.arange(5 * 5, dtype=np.uint16).reshape(5, 5) + 1)
    right = (np.arange(5 * 5, dtype=np.uint16).reshape(5, 5) + 1000)
    left_path = _write_src(tmp_path, left, 'tmp_1704_left.tif')
    right_path = _write_src(tmp_path, right, 'tmp_1704_right.tif')
    vrt_xml = (
        '<VRTDataset rasterXSize="20" rasterYSize="10">\n'
        '  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
        '  <VRTRasterBand dataType="UInt16" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{left_path}</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="5" ySize="5"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n'
        '    </SimpleSource>\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{right_path}</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="5" ySize="5"/>\n'
        '      <DstRect xOff="10" yOff="0" xSize="10" ySize="10"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    vrt_path = _write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_multi.vrt')

    full, _ = read_vrt(vrt_path)
    assert full.shape == (10, 20)
    # Window crosses x=10 boundary so both sources are clipped, not
    # one-or-the-other.
    windowed, _ = read_vrt(vrt_path, window=(2, 7, 8, 14))
    np.testing.assert_array_equal(windowed, full[2:8, 7:14])


# ---------------------------------------------------------------------------
# Nodata: masking happens on the read buffer; sub-window read still masks.
# ---------------------------------------------------------------------------

def test_nodata_round_trip_through_window(tmp_path):
    """SimpleSource with ``<NODATA>``; the sentinel inside the windowed
    region must surface as NaN in a float-typed VRT. Both the full read
    and the windowed read must agree on which pixels are NaN.
    """
    src = (np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) + 1).astype(np.uint16)
    # Sprinkle the sentinel through the source so the sub-window catches
    # at least one masked pixel under the 2x upsample.
    src[3, 4] = 65535
    src[5, 2] = 65535
    src_path = _write_src(tmp_path, src)
    vrt_xml = _single_source_vrt(
        src_path,
        raster_x=16, raster_y=16,
        src_x=0, src_y=0, src_xsize=8, src_ysize=8,
        dst_x=0, dst_y=0, dst_xsize=16, dst_ysize=16,
        dtype="Float32",
        nodata="65535",
    )
    vrt_path = _write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_nodata.vrt')

    full, _ = read_vrt(vrt_path)
    windowed, _ = read_vrt(vrt_path, window=(4, 4, 12, 12))
    # NaN equality needs assert_array_equal with equal_nan=True.
    np.testing.assert_array_equal(windowed, full[4:12, 4:12])
    # Sanity: the sub-window contains at least one NaN, otherwise the
    # test is vacuous.
    assert np.isnan(windowed).any()


# ---------------------------------------------------------------------------
# Read-bound assertion: only the minimal source sub-rect is decoded.
# ---------------------------------------------------------------------------

def test_only_minimal_source_rect_is_read(tmp_path):
    """Patch ``read_to_array`` to record the windows requested. Under
    the new path the source window must be much smaller than the full
    SrcRect when the caller asks for a small sub-window.
    """
    src = (np.arange(40 * 40, dtype=np.uint16).reshape(40, 40) + 1)
    src_path = _write_src(tmp_path, src)
    vrt_xml = _single_source_vrt(
        src_path,
        raster_x=160, raster_y=160,
        src_x=0, src_y=0, src_xsize=40, src_ysize=40,
        dst_x=0, dst_y=0, dst_xsize=160, dst_ysize=160,
    )
    vrt_path = _write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_bound.vrt')

    seen_windows: list[tuple[int, int, int, int]] = []
    # ``read_vrt`` does ``from ._reader import read_to_array`` at call
    # time, so the spy must live on ``_reader`` (the module that owns
    # the name), not on ``_vrt``.
    from xrspatial.geotiff import _reader as _reader_mod
    real_read = _reader_mod.read_to_array

    def spy(filename, *, window, **kw):
        seen_windows.append(tuple(window))
        return real_read(filename, window=window, **kw)

    with mock.patch.object(_reader_mod, 'read_to_array', spy):
        # Window is 8x8 pixels in destination coords starting at (80, 80).
        # Mapping back through floor((d+0.5)*40/160) = floor((d+0.5)/4)
        # gives source rows / cols 20..21 (inclusive) so the read should
        # be 2x2 source pixels, not 40x40.
        arr, _ = read_vrt(vrt_path, window=(80, 80, 88, 88))

    assert arr.shape == (8, 8)
    assert len(seen_windows) == 1
    r0, c0, r1, c1 = seen_windows[0]
    read_h = r1 - r0
    read_w = c1 - c0
    assert read_h < 10, (
        f"expected a small source row range, got {read_h} rows; "
        f"the full SrcRect is 40 rows so the fix is not reducing the read."
    )
    assert read_w < 10
