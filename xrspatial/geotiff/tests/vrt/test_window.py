"""Consolidated VRT window / scaling / tiling / chunking tests.

Folds nine issue-numbered VRT test files into one place. Each section
preserves its originating file's helpers, fixtures and assertions;
helpers are prefixed (e.g. ``_window_validation_*``) so the folds do
not collide. Test names dropped their trailing issue numbers where
the originating file already namespaced them.

Sections:
* Window kwarg validation (#1697)
* Resample + window inverse parity (#1704)
* DstRect resample-time cap (#1737)
* Scaled SrcRect / DstRect nearest resampling (#1694)
* Per-source tile-size sanity check (#1823)
* Per-source max-pixel cap (#1796)
* Lazy chunks construction (#1814)
* Shared parsed VRTDataset in the chunked graph (#1923)
* Tiled VRT writer uses the threaded scheduler (#1714)

See ``CLUSTER_AUDIT_PR6.md`` for the file:test -> section:test mapping.
"""
from __future__ import annotations

import glob
import os
import tempfile
import uuid
import warnings
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_vrt, to_geotiff
from xrspatial.geotiff._reader import PixelSafetyLimitError, read_to_array
from xrspatial.geotiff._vrt import _resample_nearest
from xrspatial.geotiff._vrt import read_vrt as _dstrect_cap_read_vrt_internal
from xrspatial.geotiff._vrt import read_vrt as _resample_window_inverse_read_vrt_internal
from xrspatial.geotiff._vrt import read_vrt as _scaled_rects_read_vrt_internal
from xrspatial.geotiff._vrt import read_vrt as _source_tile_check_read_vrt_internal
from xrspatial.geotiff._vrt import read_vrt as _window_validation_read_vrt_internal
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal
from xrspatial.geotiff._writer import write

# ---------------------------------------------------------------------------
# window kwarg validation (#1697)
# Originally: test_vrt_window_validation_1697.py
# ---------------------------------------------------------------------------


def _window_validation_unique_dir(tmp_path, label: str) -> str:
    """Return a sub-path under ``tmp_path`` with a uuid suffix so
    parallel test workers cannot collide on the same name."""
    d = tmp_path / f'vrt_1697_{label}_{uuid.uuid4().hex[:8]}'
    d.mkdir()
    return str(d)


def _window_validation_write_tif(path: str, size: int = 4) -> None:
    """Write a ``size``x``size`` float32 GeoTIFF the VRT can wrap."""
    arr = np.arange(size * size, dtype=np.float32).reshape(size, size)
    y = np.linspace(float(size) - 0.5, 0.5, size)
    x = np.linspace(0.5, float(size) - 0.5, size)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326})
    to_geotiff(da, path, compression='none')


def _window_validation_write_vrt(vrt_path: str, source_filename: str, size: int = 4) -> None:
    """Write a single-band VRT of dimension ``size``x``size`` pointing
    at ``source_filename`` (relative to the VRT directory)."""
    xml = f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}">\n  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n  <VRTRasterBand dataType="Float32" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="1">{source_filename}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    with open(vrt_path, 'w') as f:
        f.write(xml)


@pytest.fixture
def window_validation_vrt_4x4(tmp_path):
    """Return a path to a 4x4 single-band VRT wrapping a 4x4 TIFF."""
    d = _window_validation_unique_dir(tmp_path, 'fixture')
    tif = os.path.join(d, 'data.tif')
    _window_validation_write_tif(tif, size=4)
    vrt = os.path.join(d, 'mosaic.vrt')
    _window_validation_write_vrt(vrt, 'data.tif', size=4)
    return vrt


def test_negative_r0_raises_value_error(window_validation_vrt_4x4):
    """``r0 < 0`` raises ValueError instead of being clamped to 0."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(-1, 0, 2, 2))


def test_negative_c0_raises_value_error(window_validation_vrt_4x4):
    """``c0 < 0`` raises ValueError instead of being clamped to 0."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(0, -1, 2, 2))


def test_r1_past_bottom_edge_raises_value_error(window_validation_vrt_4x4):
    """``r1 > vrt.height`` raises instead of being clamped to vrt.height."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(0, 0, 5, 4))


def test_c1_past_right_edge_raises_value_error(window_validation_vrt_4x4):
    """``c1 > vrt.width`` raises instead of being clamped to vrt.width."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(0, 0, 4, 5))


def test_window_past_both_edges_raises_value_error(window_validation_vrt_4x4):
    """Windows past both right and bottom edges raise the same error."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(0, 0, 10, 10))


def test_zero_size_row_window_raises_value_error(window_validation_vrt_4x4):
    """``r0 == r1`` produces a zero-height window and must raise."""
    with pytest.raises(ValueError, match='non-positive size'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(2, 0, 2, 4))


def test_zero_size_col_window_raises_value_error(window_validation_vrt_4x4):
    """``c0 == c1`` produces a zero-width window and must raise."""
    with pytest.raises(ValueError, match='non-positive size'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(0, 2, 4, 2))


def test_fully_zero_size_window_raises_value_error(window_validation_vrt_4x4):
    """``r0 == r1 and c0 == c1`` raises (current code returned a (0, 0) array)."""
    with pytest.raises(ValueError, match='non-positive size'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(2, 2, 2, 2))


def test_inverted_row_window_raises_value_error(window_validation_vrt_4x4):
    """``r0 > r1`` is degenerate and must raise."""
    with pytest.raises(ValueError, match='non-positive size'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(3, 0, 1, 4))


def test_inverted_col_window_raises_value_error(window_validation_vrt_4x4):
    """``c0 > c1`` is degenerate and must raise."""
    with pytest.raises(ValueError, match='non-positive size'):
        _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(0, 3, 4, 1))


def test_full_extent_window_still_works(window_validation_vrt_4x4):
    """A window covering the full VRT extent still reads the full array."""
    arr, _ = _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(0, 0, 4, 4))
    assert arr.shape == (4, 4)


def test_interior_window_still_works(window_validation_vrt_4x4):
    """An interior window returns the requested subset shape."""
    arr, _ = _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(1, 1, 3, 3))
    assert arr.shape == (2, 2)


def test_edge_aligned_window_still_works(window_validation_vrt_4x4):
    """A window that touches but does not exceed the edges is accepted."""
    arr, _ = _window_validation_read_vrt_internal(window_validation_vrt_4x4, window=(2, 2, 4, 4))
    assert arr.shape == (2, 2)


def test_none_window_still_returns_full_array(window_validation_vrt_4x4):
    """``window=None`` still returns the full VRT extent."""
    arr, _ = _window_validation_read_vrt_internal(window_validation_vrt_4x4)
    assert arr.shape == (4, 4)


def test_vrt_and_local_paths_share_window_validation(tmp_path):
    """Same bad window rejected on both VRT and local-TIFF paths with the
    same error class and message shape (one word swap is fine)."""
    d = _window_validation_unique_dir(tmp_path, 'parity')
    tif = os.path.join(d, 'data.tif')
    _window_validation_write_tif(tif, size=4)
    vrt = os.path.join(d, 'mosaic.vrt')
    _window_validation_write_vrt(vrt, 'data.tif', size=4)
    bad_window = (-1, 0, 2, 2)
    with pytest.raises(ValueError) as vrt_exc:
        _window_validation_read_vrt_internal(vrt, window=bad_window)
    with pytest.raises(ValueError) as local_exc:
        read_to_array(tif, window=bad_window)
    vrt_msg = str(vrt_exc.value)
    local_msg = str(local_exc.value)
    assert 'window=' in vrt_msg
    assert 'window=' in local_msg
    assert '4x4' in vrt_msg
    assert '4x4' in local_msg
    assert 'extent' in vrt_msg
    assert 'extent' in local_msg
    assert 'non-positive size' in vrt_msg
    assert 'non-positive size' in local_msg
    assert 'VRT extent' in vrt_msg
    assert 'source extent' in local_msg


# ---------------------------------------------------------------------------
# resample+window inverse parity (#1704)
# Originally: test_vrt_resample_window_inverse_1704.py
# ---------------------------------------------------------------------------


def _resample_window_inverse_write_vrt_xml(tmp_path, xml: str, name: str) -> str:
    p = str(tmp_path / name)
    with open(p, 'w') as f:
        f.write(xml)
    return p


def _resample_window_inverse_write_src(tmp_path, arr: np.ndarray, name: str = 'tmp_1704_src.tif') -> str:  # noqa: E501
    src_path = str(tmp_path / name)
    write(arr, src_path, compression='none', tiled=False)
    return src_path


def _resample_window_inverse_single_source_vrt(src_path: str, *, raster_x: int, raster_y: int, src_x: int, src_y: int, src_xsize: int, src_ysize: int, dst_x: int, dst_y: int, dst_xsize: int, dst_ysize: int, dtype: str = 'UInt16', nodata: str | None = None) -> str:  # noqa: E501
    nodata_xml = f'      <NODATA>{nodata}</NODATA>\n' if nodata is not None else ''
    return f'<VRTDataset rasterXSize="{raster_x}" rasterYSize="{raster_y}">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="{dtype}" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="{src_x}" yOff="{src_y}" xSize="{src_xsize}" ySize="{src_ysize}"/>\n      <DstRect xOff="{dst_x}" yOff="{dst_y}" xSize="{dst_xsize}" ySize="{dst_ysize}"/>\n{nodata_xml}    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501


def test_upsample_window_matches_full_then_slice(tmp_path):
    """4x upsample, then read a small window from the middle. The
    windowed read must equal the full read sliced at the same offsets."""
    src = np.arange(10 * 10, dtype=np.uint16).reshape(10, 10) + 1
    src_path = _resample_window_inverse_write_src(tmp_path, src)
    vrt_xml = _resample_window_inverse_single_source_vrt(src_path, raster_x=40, raster_y=40, src_x=0, src_y=0, src_xsize=10, src_ysize=10, dst_x=0, dst_y=0, dst_xsize=40, dst_ysize=40)  # noqa: E501
    vrt_path = _resample_window_inverse_write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_up.vrt')
    full, _ = _resample_window_inverse_read_vrt_internal(vrt_path)
    assert full.shape == (40, 40)
    r0, c0, r1, c1 = (7, 11, 33, 29)
    windowed, _ = _resample_window_inverse_read_vrt_internal(vrt_path, window=(r0, c0, r1, c1))
    np.testing.assert_array_equal(windowed, full[r0:r1, c0:c1])


def test_downsample_window_matches_full_then_slice(tmp_path):
    """4x downsample, windowed read parity with full-then-slice."""
    src = np.arange(40 * 40, dtype=np.uint16).reshape(40, 40) + 1
    src_path = _resample_window_inverse_write_src(tmp_path, src)
    vrt_xml = _resample_window_inverse_single_source_vrt(src_path, raster_x=10, raster_y=10, src_x=0, src_y=0, src_xsize=40, src_ysize=40, dst_x=0, dst_y=0, dst_xsize=10, dst_ysize=10)  # noqa: E501
    vrt_path = _resample_window_inverse_write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_down.vrt')
    full, _ = _resample_window_inverse_read_vrt_internal(vrt_path)
    assert full.shape == (10, 10)
    r0, c0, r1, c1 = (2, 3, 9, 8)
    windowed, _ = _resample_window_inverse_read_vrt_internal(vrt_path, window=(r0, c0, r1, c1))
    np.testing.assert_array_equal(windowed, full[r0:r1, c0:c1])


@pytest.mark.parametrize('r0,c0,r1,c1', [(0, 0, 11, 11), (1, 1, 10, 10), (3, 2, 7, 9), (0, 0, 1, 1), (10, 10, 11, 11), (5, 0, 6, 11), (0, 5, 11, 6)])  # noqa: E501
def test_non_integer_ratio_7_to_11_window_parity(tmp_path, r0, c0, r1, c1):
    """SrcRect 7x7, DstRect 11x11 (irrational ratio 7/11). The
    nearest-neighbour mapping has uneven step sizes so the inverse
    mapping has to handle each output index individually; this is the
    case that breaks Option-2 "resample sub-shape into sub-shape"
    implementations.
    """
    src = np.arange(7 * 7, dtype=np.uint16).reshape(7, 7) + 100
    src_path = _resample_window_inverse_write_src(tmp_path, src)
    vrt_xml = _resample_window_inverse_single_source_vrt(src_path, raster_x=11, raster_y=11, src_x=0, src_y=0, src_xsize=7, src_ysize=7, dst_x=0, dst_y=0, dst_xsize=11, dst_ysize=11)  # noqa: E501
    vrt_path = _resample_window_inverse_write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_7_11.vrt')
    full, _ = _resample_window_inverse_read_vrt_internal(vrt_path)
    windowed, _ = _resample_window_inverse_read_vrt_internal(vrt_path, window=(r0, c0, r1, c1))
    np.testing.assert_array_equal(windowed, full[r0:r1, c0:c1])


def test_window_starting_at_origin(tmp_path):
    src = np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) + 1
    src_path = _resample_window_inverse_write_src(tmp_path, src)
    vrt_xml = _resample_window_inverse_single_source_vrt(src_path, raster_x=20, raster_y=20, src_x=0, src_y=0, src_xsize=8, src_ysize=8, dst_x=0, dst_y=0, dst_xsize=20, dst_ysize=20)  # noqa: E501
    vrt_path = _resample_window_inverse_write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_origin.vrt')
    full, _ = _resample_window_inverse_read_vrt_internal(vrt_path)
    windowed, _ = _resample_window_inverse_read_vrt_internal(vrt_path, window=(0, 0, 5, 5))
    np.testing.assert_array_equal(windowed, full[0:5, 0:5])


def test_window_ending_at_last_pixel(tmp_path):
    src = np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) + 1
    src_path = _resample_window_inverse_write_src(tmp_path, src)
    vrt_xml = _resample_window_inverse_single_source_vrt(src_path, raster_x=20, raster_y=20, src_x=0, src_y=0, src_xsize=8, src_ysize=8, dst_x=0, dst_y=0, dst_xsize=20, dst_ysize=20)  # noqa: E501
    vrt_path = _resample_window_inverse_write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_last.vrt')
    full, _ = _resample_window_inverse_read_vrt_internal(vrt_path)
    windowed, _ = _resample_window_inverse_read_vrt_internal(vrt_path, window=(15, 15, 20, 20))
    np.testing.assert_array_equal(windowed, full[15:20, 15:20])


def test_window_crossing_multiple_sources(tmp_path):
    """Two SimpleSources tiled side by side, each with non-1:1 SrcRect /
    DstRect. A window that spans both sources must equal the full read
    sliced over the same range. Both sources go through the new windowed
    resample path.
    """
    left = np.arange(5 * 5, dtype=np.uint16).reshape(5, 5) + 1
    right = np.arange(5 * 5, dtype=np.uint16).reshape(5, 5) + 1000
    left_path = _resample_window_inverse_write_src(tmp_path, left, 'tmp_1704_left.tif')
    right_path = _resample_window_inverse_write_src(tmp_path, right, 'tmp_1704_right.tif')
    vrt_xml = f'<VRTDataset rasterXSize="20" rasterYSize="10">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{left_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="5" ySize="5"/>\n      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n    </SimpleSource>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{right_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="5" ySize="5"/>\n      <DstRect xOff="10" yOff="0" xSize="10" ySize="10"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    vrt_path = _resample_window_inverse_write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_multi.vrt')
    full, _ = _resample_window_inverse_read_vrt_internal(vrt_path)
    assert full.shape == (10, 20)
    windowed, _ = _resample_window_inverse_read_vrt_internal(vrt_path, window=(2, 7, 8, 14))
    np.testing.assert_array_equal(windowed, full[2:8, 7:14])


def test_nodata_round_trip_through_window(tmp_path):
    """SimpleSource with ``<NODATA>``; the sentinel inside the windowed
    region must surface as NaN in a float-typed VRT. Both the full read
    and the windowed read must agree on which pixels are NaN.
    """
    src = (np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) + 1).astype(np.uint16)
    src[3, 4] = 65535
    src[5, 2] = 65535
    src_path = _resample_window_inverse_write_src(tmp_path, src)
    vrt_xml = _resample_window_inverse_single_source_vrt(src_path, raster_x=16, raster_y=16, src_x=0, src_y=0, src_xsize=8, src_ysize=8, dst_x=0, dst_y=0, dst_xsize=16, dst_ysize=16, dtype='Float32', nodata='65535')  # noqa: E501
    vrt_path = _resample_window_inverse_write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_nodata.vrt')
    full, _ = _resample_window_inverse_read_vrt_internal(vrt_path)
    windowed, _ = _resample_window_inverse_read_vrt_internal(vrt_path, window=(4, 4, 12, 12))
    np.testing.assert_array_equal(windowed, full[4:12, 4:12])
    assert np.isnan(windowed).any()


def test_only_minimal_source_rect_is_read(tmp_path):
    """Patch ``read_to_array`` to record the windows requested. Under
    the new path the source window must be much smaller than the full
    SrcRect when the caller asks for a small sub-window.
    """
    src = np.arange(40 * 40, dtype=np.uint16).reshape(40, 40) + 1
    src_path = _resample_window_inverse_write_src(tmp_path, src)
    vrt_xml = _resample_window_inverse_single_source_vrt(src_path, raster_x=160, raster_y=160, src_x=0, src_y=0, src_xsize=40, src_ysize=40, dst_x=0, dst_y=0, dst_xsize=160, dst_ysize=160)  # noqa: E501
    vrt_path = _resample_window_inverse_write_vrt_xml(tmp_path, vrt_xml, 'tmp_1704_bound.vrt')
    seen_windows: list[tuple[int, int, int, int]] = []
    from xrspatial.geotiff import _reader as _reader_mod
    real_read = _reader_mod.read_to_array

    def spy(filename, *, window, **kw):
        seen_windows.append(tuple(window))
        return real_read(filename, window=window, **kw)
    with mock.patch.object(_reader_mod, 'read_to_array', spy):
        arr, _ = _resample_window_inverse_read_vrt_internal(vrt_path, window=(80, 80, 88, 88))
    assert arr.shape == (8, 8)
    assert len(seen_windows) == 1
    r0, c0, r1, c1 = seen_windows[0]
    read_h = r1 - r0
    read_w = c1 - c0
    assert read_h < 10, f'expected a small source row range, got {read_h} rows; the full SrcRect is 40 rows so the fix is not reducing the read.'  # noqa: E501
    assert read_w < 10


# ---------------------------------------------------------------------------
# DstRect resample cap (#1737)
# Originally: test_vrt_dstrect_resample_cap_1737.py
# ---------------------------------------------------------------------------


def _dstrect_cap_write_source(td: str) -> str:
    """Write a 10x10 uint8 source GeoTIFF and return its path.

    Stripped (non-tiled) so the source read does not allocate a 256x256
    tile that trips ``_check_dimensions`` under the small ``max_pixels``
    values these tests pass.
    """
    src_path = os.path.join(td, 'src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path, compression='none', tiled=False)
    return src_path


def _dstrect_cap_write_vrt(td: str, *, dst_x_size: int, dst_y_size: int, raster_x: int = 100, raster_y: int = 100) -> str:  # noqa: E501
    """Write a VRT with a single SimpleSource using the given DstRect size."""
    vrt_path = os.path.join(td, 'mosaic.vrt')
    vrt_xml = f'<VRTDataset rasterXSize="{raster_x}" rasterYSize="{raster_y}">\n  <VRTRasterBand dataType="Byte" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="1">src.tif</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n      <DstRect xOff="0" yOff="0" xSize="{dst_x_size}" ySize="{dst_y_size}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
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
        _dstrect_cap_write_source(td)
        vrt_path = _dstrect_cap_write_vrt(td, dst_x_size=50000, dst_y_size=50000)
        arr, _ = _dstrect_cap_read_vrt_internal(vrt_path)
        assert arr.shape == (100, 100)


def test_huge_dstrect_y_axis_clipped_to_extent():
    """Asymmetric blow-up: ``ySize`` declared as 10 billion but the VRT
    extent caps the clipped sub-window at 100 rows. Read succeeds with
    the bounded intermediate."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _dstrect_cap_write_source(td)
        vrt_path = _dstrect_cap_write_vrt(td, dst_x_size=10, dst_y_size=10000000000)
        arr, _ = _dstrect_cap_read_vrt_internal(vrt_path)
        assert arr.shape == (100, 100)


def test_legitimate_upsample_still_works():
    """A legitimate upsample stays under the cap and must succeed."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _dstrect_cap_write_source(td)
        vrt_path = _dstrect_cap_write_vrt(td, dst_x_size=100, dst_y_size=100)
        arr, _ = _dstrect_cap_read_vrt_internal(vrt_path)
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
        _dstrect_cap_write_source(td)
        vrt_path = _dstrect_cap_write_vrt(td, dst_x_size=2000, dst_y_size=2000, raster_x=2000, raster_y=2000)  # noqa: E501
        with pytest.raises(ValueError, match='resample intermediate|safety limit'):
            _dstrect_cap_read_vrt_internal(vrt_path, max_pixels=1000000)
        arr, _ = _dstrect_cap_read_vrt_internal(vrt_path, max_pixels=4000000)
        assert arr.shape == (2000, 2000)


def test_per_source_cap_inclusive_boundary():
    """The per-source cap is inclusive: exactly ``max_pixels`` succeeds,
    one below rejects. Mirrors the boundary the original #1737 test
    pinned down, on the new sub-window semantics."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _dstrect_cap_write_source(td)
        vrt_path = _dstrect_cap_write_vrt(td, dst_x_size=100, dst_y_size=100, raster_x=100, raster_y=100)  # noqa: E501
        with pytest.raises(ValueError, match='resample intermediate|safety limit'):
            _dstrect_cap_read_vrt_internal(vrt_path, max_pixels=9999)
        arr, _ = _dstrect_cap_read_vrt_internal(vrt_path, max_pixels=10000)
        assert arr.shape == (100, 100)


def test_negative_dstrect_rejected():
    """Negative ``xSize`` / ``ySize`` must surface as ``ValueError``
    rather than be silently skipped by the overlap check.  The error
    message must call out the malformed negative size, not the pixel
    budget."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _dstrect_cap_write_source(td)
        vrt_path = _dstrect_cap_write_vrt(td, dst_x_size=-5, dst_y_size=100)
        with pytest.raises(ValueError, match='negative size'):
            _dstrect_cap_read_vrt_internal(vrt_path)


def test_negative_dstrect_y_size_rejected():
    """Negative ``ySize`` is also rejected with the same tailored error."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _dstrect_cap_write_source(td)
        vrt_path = _dstrect_cap_write_vrt(td, dst_x_size=100, dst_y_size=-5)
        with pytest.raises(ValueError, match='negative size'):
            _dstrect_cap_read_vrt_internal(vrt_path)


# ---------------------------------------------------------------------------
# scaled SrcRect/DstRect resampling (#1694)
# Originally: test_vrt_scaled_rects_1694.py
# ---------------------------------------------------------------------------


def _scaled_rects_write_vrt(tmp_path, xml: str, name: str = 'test.vrt') -> str:
    p = str(tmp_path / name)
    with open(p, 'w') as f:
        f.write(xml)
    return p


def test_downsample_4x4_to_2x2_does_not_raise_and_uses_nearest(tmp_path):
    """SrcRect 4x4 -> DstRect 2x2: result is (2,2), nearest-neighbour.

    Before the fix the source (4,4) array was assigned directly into the
    (2,2) destination slice, raising the broadcast error documented in
    issue #1694.
    """
    src = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    vrt_path = _scaled_rects_write_vrt(tmp_path, vrt_xml, 'down.vrt')
    result, _ = _scaled_rects_read_vrt_internal(vrt_path)
    assert result.shape == (2, 2), f'expected (2,2), got {result.shape}; resample step missing.'
    expected = np.array([[src[1, 1], src[1, 3]], [src[3, 1], src[3, 3]]], dtype=np.uint16)
    np.testing.assert_array_equal(result, expected)


def test_upsample_2x2_to_4x4_repeats_each_source_pixel(tmp_path):
    """SrcRect 2x2 -> DstRect 4x4: each source pixel repeated 2x2.

    Before the fix only the top-left 2x2 of the destination was written
    and the rest stayed at the fill value (0 for integer, NaN for
    float).
    """
    src = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)
    vrt_xml = f'<VRTDataset rasterXSize="4" rasterYSize="4">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    vrt_path = _scaled_rects_write_vrt(tmp_path, vrt_xml, 'up.vrt')
    result, _ = _scaled_rects_read_vrt_internal(vrt_path)
    assert result.shape == (4, 4)
    expected = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint16)
    np.testing.assert_array_equal(result, expected)
    assert not (result == 0).any(), 'upsample left zero-filled cells; resample not propagated.'


def test_non_integer_scale_3x3_to_2x2_no_holes(tmp_path):
    """Non-integer source / destination ratio: covers index-mapping path.

    With src=(3,3) -> dst=(2,2), neither integer-ratio fast path applies.
    Confirms the general nearest-neighbour gather produces the correct
    shape, no holes, no out-of-bounds writes.
    """
    src = np.arange(9, dtype=np.uint16).reshape(3, 3)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.5, 0.0, 0.0, 0.0, -1.5</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="3" ySize="3"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    vrt_path = _scaled_rects_write_vrt(tmp_path, vrt_xml, 'nonint.vrt')
    result, _ = _scaled_rects_read_vrt_internal(vrt_path)
    assert result.shape == (2, 2)
    expected = np.array([[src[0, 0], src[0, 2]], [src[2, 0], src[2, 2]]], dtype=np.uint16)
    np.testing.assert_array_equal(result, expected)


def test_per_band_scale_mix(tmp_path):
    """Mixed: band 1 downsampled, band 2 at native resolution.

    Both bands must land in the right places without a broadcast error
    and without bleeding band 1's resampled values into band 2.
    """
    band1_src = (np.arange(16, dtype=np.uint16) * 10).reshape(4, 4)
    band2_src = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    p1 = str(tmp_path / 'b1.tif')
    p2 = str(tmp_path / 'b2.tif')
    write(band1_src, p1, compression='none', tiled=False)
    write(band2_src, p2, compression='none', tiled=False)
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n  <VRTRasterBand dataType="UInt16" band="2">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p2}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    vrt_path = _scaled_rects_write_vrt(tmp_path, vrt_xml, 'mix.vrt')
    result, _ = _scaled_rects_read_vrt_internal(vrt_path)
    assert result.shape == (2, 2, 2)
    expected_b1 = np.array([[band1_src[1, 1], band1_src[1, 3]], [band1_src[3, 1], band1_src[3, 3]]], dtype=np.uint16)  # noqa: E501
    np.testing.assert_array_equal(result[..., 0], expected_b1)
    np.testing.assert_array_equal(result[..., 1], band2_src)


def test_window_on_downsampled_source_returns_correct_subwindow(tmp_path):
    """``window=(0,0,1,1)`` on a 4x4 -> 2x2 source returns the (0,0) cell.

    The destination cell maps to the source pixel that the resample
    routine would sample for that location.  Confirms the clip-after-
    resample ordering: clipping in source coordinates first (as the old
    code effectively did) would feed the wrong source slice into the
    resampler.
    """
    src = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    vrt_path = _scaled_rects_write_vrt(tmp_path, vrt_xml, 'win.vrt')
    result, _ = _scaled_rects_read_vrt_internal(vrt_path, window=(0, 0, 1, 1))
    assert result.shape == (1, 1)
    assert result[0, 0] == src[1, 1]


def test_nodata_preserved_across_downsample(tmp_path):
    """Source sentinel pixels survive the resample as NaN in the result.

    Source is uint16 with sentinel=65535.  Pixels at the sampled-from
    positions whose values are 65535 must appear as NaN in the float64
    VRT output.
    """
    sentinel = np.uint16(65535)
    src = np.array([[10, 20, 30, 40], [50, sentinel, 70, sentinel], [90, 100, 110, 120], [130, sentinel, 150, sentinel]], dtype=np.uint16)  # noqa: E501
    src_path = str(tmp_path / 'src_nd.tif')
    write(src, src_path, nodata=int(sentinel), compression='none', tiled=False)
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>\n  <VRTRasterBand dataType="Float64" band="1">\n    <NoDataValue>-9999</NoDataValue>\n    <ComplexSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <NODATA>65535</NODATA>\n    </ComplexSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    vrt_path = _scaled_rects_write_vrt(tmp_path, vrt_xml, 'nd.vrt')
    result, _ = _scaled_rects_read_vrt_internal(vrt_path)
    assert result.shape == (2, 2)
    assert result.dtype == np.float64
    assert np.isnan(result).all(), f'sentinel did not survive resample as NaN; got {result!r}'


def test_nodata_with_mixed_sentinel_and_valid_pixels(tmp_path):
    """Mixed sentinel / valid source -> mixed NaN / valid destination.

    Confirms the mask resamples *with* the data, not against the
    pre-resampled source.
    """
    sentinel = np.uint16(65535)
    src = np.zeros((4, 4), dtype=np.uint16)
    src[1, 1] = 11
    src[1, 3] = sentinel
    src[3, 1] = 31
    src[3, 3] = sentinel
    src_path = str(tmp_path / 'src_mixed.tif')
    write(src, src_path, nodata=int(sentinel), compression='none', tiled=False)
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>\n  <VRTRasterBand dataType="Float64" band="1">\n    <NoDataValue>-9999</NoDataValue>\n    <ComplexSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <NODATA>65535</NODATA>\n    </ComplexSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    vrt_path = _scaled_rects_write_vrt(tmp_path, vrt_xml, 'nd_mixed.vrt')
    result, _ = _scaled_rects_read_vrt_internal(vrt_path)
    assert result.shape == (2, 2)
    assert result[0, 0] == 11.0
    assert np.isnan(result[0, 1])
    assert result[1, 0] == 31.0
    assert np.isnan(result[1, 1])


@pytest.mark.parametrize('shape', [(0, 5), (5, 0), (0, 0)])
def test_resample_nearest_rejects_empty_source(shape):
    """``_resample_nearest`` raises ValueError on an empty source array.

    A SimpleSource with ``SrcRect xSize=0`` or ``ySize=0`` -- or a
    windowed read that clamps to an empty slice -- would otherwise feed
    a zero-dim array to the integer-ratio fast paths, which compute
    ``out_h % src_h`` and divide by ``src_h``/``src_w`` and so would
    raise an opaque ``ZeroDivisionError``.  Surface the bad input with
    a clear ``ValueError`` instead.
    """
    src_arr = np.zeros(shape, dtype=np.float64)
    with pytest.raises(ValueError, match='empty source array'):
        _resample_nearest(src_arr, 2, 2)


# ---------------------------------------------------------------------------
# per-source tile-size sanity check (#1823)
# Originally: test_vrt_source_tile_check_1823.py
# ---------------------------------------------------------------------------


def _source_tile_check_write_normal_tile_source(td: str) -> str:
    """10x10 uint8 source -- ``to_geotiff`` pads to a 256x256 tile."""
    src = os.path.join(td, 'src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src, compression='none')
    return src


def _source_tile_check_write_vrt(td: str, *, dst_x_size: int, dst_y_size: int, raster_x: int = 100, raster_y: int = 100, src_x_size: int = 10, src_y_size: int = 10) -> str:  # noqa: E501
    vrt = os.path.join(td, 'mosaic.vrt')
    xml = f'<VRTDataset rasterXSize="{raster_x}" rasterYSize="{raster_y}">\n  <VRTRasterBand dataType="Byte" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="1">src.tif</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{src_x_size}" ySize="{src_y_size}"/>\n      <DstRect xOff="0" yOff="0" xSize="{dst_x_size}" ySize="{dst_y_size}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'  # noqa: E501
    with open(vrt, 'w') as f:
        f.write(xml)
    return vrt


class TestPerTileCheckDoesNotUseCallerBudget:
    """Per-tile dim sanity must not reject normal 256x256 source tiles
    when the caller's ``max_pixels`` is a small output-budget value."""

    def test_normal_tile_source_with_small_max_pixels(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _source_tile_check_write_normal_tile_source(td)
            vrt = _source_tile_check_write_vrt(td, dst_x_size=100, dst_y_size=100)
            arr, _ = _source_tile_check_read_vrt_internal(vrt, max_pixels=10000)
            assert arr.shape == (100, 100)

    def test_normal_tile_source_with_tiny_max_pixels(self):
        """An output budget below a single tile must still succeed when
        the requested output window itself fits."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _source_tile_check_write_normal_tile_source(td)
            vrt = _source_tile_check_write_vrt(td, dst_x_size=5, dst_y_size=5, raster_x=5, raster_y=5)  # noqa: E501
            arr, _ = _source_tile_check_read_vrt_internal(vrt, max_pixels=100)
            assert arr.shape == (5, 5)


class TestOutputWindowCheckStillEnforced:
    """The output-window check still rejects a read whose VRT extent
    exceeds ``max_pixels``. After #1704 the source read is bounded by
    the clipped destination sub-window, so the per-source guard now
    rarely fires; the top-level ``_check_dimensions`` call against the
    output extent catches over-budget requests up front. The #1796
    protection (tiny VRT cannot force huge source decode) is preserved
    structurally.
    """

    def test_output_extent_exceeds_max_pixels_still_rejected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            src = os.path.join(td, 'src.tif')
            to_geotiff(np.arange(64, dtype=np.uint8).reshape(8, 8), src, compression='none', tiled=False)  # noqa: E501
            vrt = _source_tile_check_write_vrt(td, dst_x_size=8, dst_y_size=8, raster_x=8, raster_y=8, src_x_size=4, src_y_size=4)  # noqa: E501
            with pytest.raises(ValueError, match='exceed|safety limit'):
                _source_tile_check_read_vrt_internal(vrt, max_pixels=4)


class TestPerTileCheckStillRejectsCraftedHeader:
    """A pathological ``TileWidth``/``TileLength`` must still fail at
    the per-tile sanity check, which uses ``MAX_PIXELS_DEFAULT``."""

    def test_per_tile_check_caps_at_default(self, monkeypatch):
        """Lower ``MAX_PIXELS_DEFAULT`` to verify the per-tile call site
        is wired to it (rather than to the caller's budget)."""
        from xrspatial.geotiff import _reader as reader_mod
        monkeypatch.setattr(reader_mod, 'MAX_PIXELS_DEFAULT', 100)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _source_tile_check_write_normal_tile_source(td)
            vrt = _source_tile_check_write_vrt(td, dst_x_size=100, dst_y_size=100)
            with pytest.raises(PixelSafetyLimitError, match='65,536'):
                _source_tile_check_read_vrt_internal(vrt, max_pixels=1000000000)


# ---------------------------------------------------------------------------
# per-source max-pixel cap (#1796)
# Originally: test_vrt_source_max_pixels_1796.py
# ---------------------------------------------------------------------------


def test_tiny_vrt_with_huge_srcrect_now_reads_minimally(tmp_path):
    """A 1x1 VRT pointing at a 4x4 SrcRect now reads only the one source
    pixel that maps to the single output pixel, so ``max_pixels=1`` is
    no longer exceeded. Locks in the structural improvement from #1704."""
    src = tmp_path / 'tmp_1796_source.tif'
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    to_geotiff(data, str(src), compression='none')
    vrt = tmp_path / 'tmp_1796_source_cap.vrt'
    vrt.write_text(
        f'<VRTDataset rasterXSize="1" rasterYSize="1">\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{os.path.basename(src)}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="1" ySize="1"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    arr = read_vrt(str(vrt), max_pixels=1)
    assert arr.shape == (1, 1)


def test_source_cap_still_fires_when_sub_window_exceeds_budget(tmp_path):
    """The per-source pixel-budget guard still rejects a sub-window that
    exceeds ``max_pixels``. With the sub-window-bounded read, the cap is
    measured against the clipped destination region rather than the raw
    SrcRect; the protection from #1796 carries over to that new
    measurement.
    """
    src = tmp_path / 'tmp_1796_big_source.tif'
    data = np.arange(64, dtype=np.uint8).reshape(8, 8)
    to_geotiff(data, str(src), compression='none', tiled=False)
    vrt = tmp_path / 'tmp_1796_big_cap.vrt'
    vrt.write_text(
        f'<VRTDataset rasterXSize="8" rasterYSize="8">\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{os.path.basename(src)}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="8" ySize="8"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with pytest.raises(ValueError, match='exceed|safety limit'):
        read_vrt(str(vrt), max_pixels=4)


# ---------------------------------------------------------------------------
# lazy chunks construction (#1814)
# Originally: test_vrt_lazy_chunks_1814.py
# ---------------------------------------------------------------------------


def _lazy_chunks_gpu_available() -> bool:
    try:
        import cupy
    except ImportError:
        return False
    try:
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _lazy_chunks_gpu_available()


@pytest.fixture
def lazy_chunks_single_tile_vrt():
    """One 128x128 float32 tile wrapped in a VRT."""
    arr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    y = np.linspace(41.0, 40.0, 128)
    x = np.linspace(-106.0, -105.0, 128)
    raster = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1814_single_')
    tile_path = os.path.join(td, 'tile.tif')
    to_geotiff(raster, tile_path)
    vrt_path = os.path.join(td, 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    yield (vrt_path, arr)


@pytest.fixture
def lazy_chunks_two_by_two_vrt():
    """4-tile mosaic via the to_geotiff(.vrt, ...) dask path."""
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    y = np.linspace(41.0, 40.0, 256)
    x = np.linspace(-106.0, -105.0, 256)
    raster = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1814_2x2_')
    vrt_path = os.path.join(td, 'mosaic.vrt')
    to_geotiff(raster, vrt_path, tile_size=128)
    yield (vrt_path, arr)


@pytest.fixture
def lazy_chunks_multiband_vrt():
    """3-band single-tile VRT."""
    rng = np.random.default_rng(1814)
    arr = rng.random((64, 64, 3), dtype=np.float32)
    y = np.linspace(41.0, 40.0, 64)
    x = np.linspace(-106.0, -105.0, 64)
    raster = xr.DataArray(arr, dims=['y', 'x', 'band'], coords={'y': y, 'x': x, 'band': np.arange(3)}, attrs={'crs': 4326})  # noqa: E501
    td = tempfile.mkdtemp(prefix='tmp_1814_mb_')
    tile_path = os.path.join(td, 'tile.tif')
    to_geotiff(raster, tile_path)
    vrt_path = os.path.join(td, 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    yield (vrt_path, arr)


def test_chunks_builds_dask_array_with_multiple_blocks(lazy_chunks_two_by_two_vrt):
    """``read_vrt(chunks=(N,N))`` returns a dask-backed DataArray
    whose underlying array has more than one chunk along each spatial
    axis. Before the fix the array was numpy-backed under
    ``result.chunk()``, so this asserts the new lazy graph is in
    play.
    """
    vrt_path, _ = lazy_chunks_two_by_two_vrt
    result = read_vrt(vrt_path, chunks=(64, 64))
    assert isinstance(result.data, da.Array), f'expected dask Array, got {type(result.data).__name__}'  # noqa: E501
    assert result.data.numblocks == (4, 4), f'expected 4x4 blocks, got {result.data.numblocks}'


def test_chunks_is_lazy_does_not_call_internal_reader(monkeypatch, lazy_chunks_two_by_two_vrt):
    """Construction-time call count of the internal VRT reader is zero;
    after ``.compute()`` it equals the chunk count.
    """
    vrt_path, _ = lazy_chunks_two_by_two_vrt
    from xrspatial.geotiff import _vrt as vrt_module
    counter = {'calls': 0}
    real_read = vrt_module.read_vrt

    def counting_read(*args, **kwargs):
        counter['calls'] += 1
        return real_read(*args, **kwargs)
    monkeypatch.setattr(vrt_module, 'read_vrt', counting_read)
    result = read_vrt(vrt_path, chunks=(64, 64))
    assert counter['calls'] == 0, f"_read_vrt_internal called {counter['calls']} times before .compute(); the chunked path leaked an eager decode"  # noqa: E501
    computed = result.compute()
    assert counter['calls'] == 16, f"expected 16 per-chunk decodes after compute, got {counter['calls']}"  # noqa: E501
    assert computed.shape == (256, 256)


def test_chunked_compute_matches_eager(lazy_chunks_two_by_two_vrt):
    vrt_path, _ = lazy_chunks_two_by_two_vrt
    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=(64, 64)).compute()
    assert eager.shape == chunked.shape
    assert np.array_equal(eager.values, chunked.values), 'chunked compute diverged from eager read'
    np.testing.assert_array_equal(eager['x'].values, chunked['x'].values)
    np.testing.assert_array_equal(eager['y'].values, chunked['y'].values)
    assert eager.attrs.get('transform') == chunked.attrs.get('transform')
    assert eager.attrs.get('crs') == chunked.attrs.get('crs')


def test_chunked_single_tile_matches_eager(lazy_chunks_single_tile_vrt):
    """Single-tile VRT (one source) should still match eager when
    chunked. Exercises the path where many chunk windows hit the
    same single source.
    """
    vrt_path, _ = lazy_chunks_single_tile_vrt
    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=(32, 32)).compute()
    assert np.array_equal(eager.values, chunked.values)


def test_chunks_task_cap_raises(lazy_chunks_two_by_two_vrt):
    """``chunks=(1, 1)`` on a 256x256 VRT would build 65,536 tasks,
    blowing past the 50,000-task cap. The reader should refuse with
    a ValueError that names ``chunks=`` and suggests a larger size.
    """
    vrt_path, _ = lazy_chunks_two_by_two_vrt
    with pytest.raises(ValueError, match='chunks=.*task'):
        read_vrt(vrt_path, chunks=(1, 1))


def test_window_plus_chunks_matches_eager(lazy_chunks_two_by_two_vrt):
    """When both ``window=`` and ``chunks=`` are passed, the dask
    graph must tile the window (not the full VRT extent). The output
    shape and pixel values match an eager windowed read.
    """
    vrt_path, _ = lazy_chunks_two_by_two_vrt
    window = (32, 48, 160, 192)
    eager = read_vrt(vrt_path, window=window)
    chunked = read_vrt(vrt_path, window=window, chunks=(64, 64))
    assert isinstance(chunked.data, da.Array)
    assert chunked.data.numblocks == (2, 3), f'expected (2, 3) numblocks over the window, got {chunked.data.numblocks}'  # noqa: E501
    computed = chunked.compute()
    assert computed.shape == eager.shape == (128, 144)
    assert np.array_equal(eager.values, computed.values)


@pytest.mark.skipif(not _HAS_GPU, reason='cupy + CUDA required')
def test_gpu_plus_chunks_returns_dask_on_cupy(lazy_chunks_two_by_two_vrt):
    """``read_vrt(gpu=True, chunks=...)`` must build a dask graph whose
    blocks are cupy-backed (not numpy that gets cupy-wrapped at
    compute time on the host).
    """
    import cupy
    vrt_path, _ = lazy_chunks_two_by_two_vrt
    result = read_vrt(vrt_path, gpu=True, chunks=(64, 64))
    assert isinstance(result.data, da.Array)
    assert isinstance(result.data._meta, cupy.ndarray), f'expected cupy _meta, got {type(result.data._meta).__module__}.{type(result.data._meta).__name__}'  # noqa: E501
    computed = result.compute()
    assert isinstance(computed.data, cupy.ndarray)


def test_multiband_plus_chunks_preserves_band_dim(lazy_chunks_multiband_vrt):
    """3-band VRT read with ``chunks=`` keeps the band dimension on
    every block and the assembled DataArray.
    """
    vrt_path, src = lazy_chunks_multiband_vrt
    result = read_vrt(vrt_path, chunks=(32, 32))
    assert isinstance(result.data, da.Array)
    assert result.dims == ('y', 'x', 'band')
    assert result.shape == (64, 64, 3)
    assert result.data.chunks[2] == (3,)
    computed = result.compute()
    np.testing.assert_allclose(computed.values, src, rtol=0, atol=0)


def test_chunked_propagates_vrt_holes_when_source_missing(lazy_chunks_two_by_two_vrt):
    """When a source referenced by the VRT does not exist on disk and
    the caller opts into the lenient ``missing_sources='warn'`` path,
    the chunked reader must populate ``attrs['vrt_holes']`` with the
    same schema the eager reader uses, so callers can branch on
    ``"vrt_holes" in da.attrs`` regardless of which code path produced
    the DataArray.

    Note: the default ``missing_sources='raise'`` raises at build time
    under #2265, so this test exercises the explicit ``'warn'`` opt-in.
    """
    import warnings

    from xrspatial.geotiff import GeoTIFFFallbackWarning
    from xrspatial.geotiff._reader import _mmap_cache
    vrt_path, _ = lazy_chunks_two_by_two_vrt
    vrt_dir = os.path.dirname(vrt_path)
    tile_files = []
    for root, _dirs, files in os.walk(vrt_dir):
        for f in files:
            if f.endswith('.tif'):
                tile_files.append(os.path.join(root, f))
    assert len(tile_files) >= 1
    _mmap_cache.clear()
    os.unlink(tile_files[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', GeoTIFFFallbackWarning)
        result = read_vrt(vrt_path, chunks=(64, 64), missing_sources='warn')
    assert 'vrt_holes' in result.attrs, 'chunked path dropped vrt_holes contract from #1734'
    holes = result.attrs['vrt_holes']
    assert isinstance(holes, list) and len(holes) >= 1
    entry = holes[0]
    assert set(entry.keys()) >= {'source', 'band', 'dst_rect', 'error'}
    assert isinstance(entry['dst_rect'], tuple)
    assert len(entry['dst_rect']) == 4


def test_chunked_no_vrt_holes_attr_when_complete(lazy_chunks_two_by_two_vrt):
    """When every source is on disk the chunked reader must not set
    ``attrs['vrt_holes']`` (eager parity: empty hole list is omitted).
    """
    vrt_path, _ = lazy_chunks_two_by_two_vrt
    result = read_vrt(vrt_path, chunks=(64, 64))
    assert 'vrt_holes' not in result.attrs


def test_chunked_integer_no_nodata_keeps_source_dtype():
    """A uint16 source with no <NoDataValue> declared must produce a
    uint16 chunked DataArray, not float64. The eager path stays integer
    in this case because its runtime ``mask.any()`` is False; the
    chunked path approximates with a static "any band declares nodata?"
    check, which yields the same answer here.
    """
    arr = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    y = np.linspace(41.0, 40.0, 128)
    x = np.linspace(-106.0, -105.0, 128)
    raster = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1814_uint16_nonodata_')
    tile_path = os.path.join(td, 'tile.tif')
    to_geotiff(raster, tile_path)
    vrt_path = os.path.join(td, 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    result = read_vrt(vrt_path, chunks=(32, 32))
    assert result.dtype == np.uint16, f'expected uint16 (source dtype), got {result.dtype}; chunked path promoted to float64 despite no declared nodata'  # noqa: E501
    computed = result.compute()
    assert computed.dtype == np.uint16
    np.testing.assert_array_equal(computed.values, arr)


# ---------------------------------------------------------------------------
# shared parsed VRT in chunked graph (#1923)
# Originally: test_vrt_chunked_shared_dataset_1923.py
# ---------------------------------------------------------------------------


def _chunked_shared_dataset_make_tile_vrt(tmp_path, n_tiles_per_side=4):
    """Build a small multi-source VRT for testing the chunked path.

    Each source is a 64x64 tile written to a temp directory; the VRT
    stitches them into a ``(64*N, 64*N)`` mosaic. ``N=4`` keeps the
    fixture cheap while still producing a multi-source VRT whose
    embedded metadata is measurable.
    """
    tile_dir = os.path.join(tmp_path, 'tiles')
    os.makedirs(tile_dir, exist_ok=True)
    tile_size = 64
    sources = []
    for r in range(n_tiles_per_side):
        for c in range(n_tiles_per_side):
            arr = np.full((tile_size, tile_size), fill_value=r * n_tiles_per_side + c, dtype=np.float32)  # noqa: E501
            ox = c * tile_size
            oy = -(r * tile_size)
            da = xr.DataArray(arr, dims=['y', 'x'], attrs={'transform': (1.0, 0.0, ox, 0.0, -1.0, oy)})  # noqa: E501
            path = os.path.join(tile_dir, f'tile_{r}_{c}.tif')
            to_geotiff(da, path, compression='deflate', tiled=True, tile_size=64)
            sources.append((path, r, c, tile_size))
    vrt_path = os.path.join(tmp_path, 'mosaic.vrt')
    width = n_tiles_per_side * tile_size
    height = n_tiles_per_side * tile_size
    lines = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">', '<SRS></SRS>', '<GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>', '<VRTRasterBand dataType="Float32" band="1">']  # noqa: E501
    for path, r, c, ts in sources:
        lines.extend(['<SimpleSource>', f'<SourceFilename relativeToVRT="0">{path}</SourceFilename>', '<SourceBand>1</SourceBand>', f'<SrcRect xOff="0" yOff="0" xSize="{ts}" ySize="{ts}"/>', f'<DstRect xOff="{c * ts}" yOff="{r * ts}" xSize="{ts}" ySize="{ts}"/>', '</SimpleSource>'])  # noqa: E501
    lines.extend(['</VRTRasterBand>', '</VRTDataset>'])
    with open(vrt_path, 'w') as f:
        f.write('\n'.join(lines))
    return (vrt_path, len(sources))


def test_vrt_chunked_dataset_is_shared_graph_input(tmp_path):
    """Issue #1923: parsed VRTDataset is wrapped as a single Delayed.

    Walks each ``_vrt_chunk_read`` task's kwargs dict in the dask graph
    and verifies that ``parsed_vrt`` is NOT an inline ``VRTDataset``
    instance (the pre-fix shape). With the fix, ``parsed_vrt`` is
    routed through ``dask.delayed(vrt, pure=True)`` so each task's
    ``kwargs['parsed_vrt']`` is a graph reference (an ``Alias`` /
    ``TaskRef``-style placeholder pointing into a single shared
    ``from-value`` layer) rather than a literal embedded dataset.

    Under the synchronous / threaded scheduler tasks are not pickled
    at all, so an embedded copy is harmless in that path. The bug
    surfaces under the distributed / multi-process scheduler where
    each task is serialised independently and the full dataset is
    shipped once per task -- so the structural shape of the graph,
    not in-process behaviour, is what matters.
    """
    from xrspatial.geotiff._vrt import VRTDataset
    vrt_path, n_sources = _chunked_shared_dataset_make_tile_vrt(str(tmp_path), n_tiles_per_side=4)
    result = read_vrt(vrt_path, chunks=32)
    graph = result.__dask_graph__()
    assert n_sources == 16, 'fixture build sanity check'
    chunk_task_count = 0
    embedded_vrt_count = 0
    for layer_name, layer in graph.layers.items():
        if '_vrt_chunk_read' not in layer_name:
            continue
        for _key, task in layer.items():
            kwargs = getattr(task, 'kwargs', None)
            if kwargs is None:
                continue
            parsed_vrt = kwargs.get('parsed_vrt')
            if parsed_vrt is None:
                continue
            chunk_task_count += 1
            if isinstance(parsed_vrt, VRTDataset):
                embedded_vrt_count += 1
    assert chunk_task_count > 1, f'fixture sanity: expected multiple chunk tasks, got {chunk_task_count}'  # noqa: E501
    assert embedded_vrt_count == 0, f"#1923 regression: {embedded_vrt_count} of {chunk_task_count} _vrt_chunk_read tasks still embed an inline VRTDataset in kwargs['parsed_vrt']. The fix wraps the dataset in dask.delayed(vrt, pure=True) so kwargs['parsed_vrt'] should be a TaskRef-style graph reference, not a VRTDataset."  # noqa: E501


def test_vrt_chunked_decode_unchanged_after_shared_wrap(tmp_path):
    """The shared-Delayed wrap must not change decoded pixel values."""
    vrt_path, _ = _chunked_shared_dataset_make_tile_vrt(str(tmp_path), n_tiles_per_side=3)
    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=32).compute()
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(chunked))


def test_vrt_chunked_band_kwarg_still_validates(tmp_path):
    """Wrapping the dataset must not change band validation behaviour."""
    vrt_path, _ = _chunked_shared_dataset_make_tile_vrt(str(tmp_path), n_tiles_per_side=2)
    with pytest.raises(ValueError):
        read_vrt(vrt_path, chunks=32, band=5)


# ---------------------------------------------------------------------------
# tiled writer uses threaded scheduler (#1714)
# Originally: test_vrt_tiled_scheduler_1714.py
# ---------------------------------------------------------------------------


def _tiled_scheduler_make_dask_da(h: int = 32, w: int = 32, chunk: int = 8) -> xr.DataArray:
    """Return a dask-backed 2D DataArray with ``chunk``-sized chunks.

    Using ``da.from_array`` on a pre-built numpy array gives clean
    ``(chunk, chunk)`` chunking. ``da.arange(...).reshape(...)`` keeps a
    chunk size of 1 along the new axis, which produces a confusing test
    setup.
    """
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return xr.DataArray(da.from_array(arr, chunks=(chunk, chunk)), dims=['y', 'x'])


def test_vrt_tiled_uses_threaded_scheduler():
    """_write_vrt_tiled passes ``scheduler='threads'`` to dask.compute."""
    da_arr = _tiled_scheduler_make_dask_da()
    with tempfile.TemporaryDirectory(prefix='vrt_sched_1714_', ignore_cleanup_errors=True) as td:
        vrt = os.path.join(td, 'sched_check.vrt')
        captured = {}
        real_compute = dask.compute

        def spy(*args, **kwargs):
            captured['scheduler'] = kwargs.get('scheduler')
            return real_compute(*args, **kwargs)
        with patch.object(dask, 'compute', side_effect=spy) as p:
            to_geotiff(da_arr, vrt)
            assert p.called, '_write_vrt_tiled never invoked dask.compute'
        assert captured.get('scheduler') == 'threads', f"Expected scheduler='threads' on the VRT-tiled write but got {captured.get('scheduler')!r}"  # noqa: E501


def test_vrt_tiled_threaded_write_produces_all_tiles():
    """All expected tile files exist after the threaded write."""
    da_arr = _tiled_scheduler_make_dask_da(h=32, w=32, chunk=8)
    with tempfile.TemporaryDirectory(prefix='vrt_sched_1714_', ignore_cleanup_errors=True) as td:
        vrt = os.path.join(td, 'tile_count.vrt')
        to_geotiff(da_arr, vrt)
        tiles_dir = os.path.join(td, 'tile_count_tiles')
        tiles = sorted(glob.glob(os.path.join(tiles_dir, '*.tif')))
        assert len(tiles) == 16, f'Expected 16 tile files, got {len(tiles)} in {tiles_dir}'


def test_vrt_tiled_threaded_write_is_deterministic():
    """Threaded scheduler must not introduce write ordering races.

    Each delayed task writes to its own file path, so the threaded
    scheduler is safe. Run the same write twice and compare byte
    contents of every tile to catch any accidental race regression.
    """
    da_arr = _tiled_scheduler_make_dask_da(h=32, w=32, chunk=8)

    def _write_and_collect(vrt_path: str) -> dict[str, bytes]:
        to_geotiff(da_arr, vrt_path)
        stem = os.path.splitext(os.path.basename(vrt_path))[0]
        tiles_dir = os.path.join(os.path.dirname(vrt_path), stem + '_tiles')
        return {os.path.basename(p): Path(p).read_bytes() for p in sorted(glob.glob(os.path.join(tiles_dir, '*.tif')))}  # noqa: E501
    with tempfile.TemporaryDirectory(prefix='vrt_sched_1714_', ignore_cleanup_errors=True) as td1:
        with tempfile.TemporaryDirectory(prefix='vrt_sched_1714_', ignore_cleanup_errors=True) as td2:  # noqa: E501
            tiles1 = _write_and_collect(os.path.join(td1, 'run1.vrt'))
            tiles2 = _write_and_collect(os.path.join(td2, 'run2.vrt'))
    assert set(tiles1) == set(tiles2), f'Tile file set differs between runs: {set(tiles1) ^ set(tiles2)}'  # noqa: E501
    for name, blob1 in tiles1.items():
        assert blob1 == tiles2[name], f'Tile {name} differs between runs (race condition?)'


# ---------------------------------------------------------------------------
# VRT-tail window / chunking folds (cluster 13, #2437)
# ---------------------------------------------------------------------------
#
# Two originally-standalone files folded here, both exercising the
# windowed / chunked read paths this module already covers:
#
# * read_vrt(chunks=...) lazy-window construction (#1798): chunk layout
#   matches eager values, build does not decode sources, and an
#   excessive task count is rejected.
# * read_geotiff_dask('.vrt') kwarg forwarding (#1795): the direct dask
#   entry point forwards window / band / max_pixels through to read_vrt.


def _vrttail_write_single_band_vrt(vrt_path, source_name):
    """One-band 6x4 Float32 VRT wrapping ``source_name`` (relative)."""
    vrt_path.write_text(
        '<VRTDataset rasterXSize="6" rasterYSize="4">\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_name}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def _vrttail_write_multi_band_vrt(vrt_path, source_name, *, bands):
    """``bands``-band 6x4 Float32 VRT wrapping ``source_name`` (relative)."""
    band_xml = []
    for i in range(bands):
        band_xml.append(
            f'  <VRTRasterBand dataType="Float32" band="{i + 1}">\n'
            '    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{source_name}'
            '</SourceFilename>\n'
            f'      <SourceBand>{i + 1}</SourceBand>\n'
            '      <SrcRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
        )
    vrt_path.write_text(
        '<VRTDataset rasterXSize="6" rasterYSize="4">\n'
        + ''.join(band_xml)
        + '</VRTDataset>\n'
    )


class TestVrtTailLazyChunks:
    """read_vrt(chunks=...) builds lazy window tasks (#1798)."""

    def test_chunks_matches_eager_values(self, tmp_path):
        arr = np.arange(24, dtype=np.float32).reshape(4, 6)
        src = tmp_path / "tmp_1798_source.tif"
        to_geotiff(arr, str(src), compression='none')
        vrt = tmp_path / "tmp_1798_source.vrt"
        _vrttail_write_single_band_vrt(vrt, os.path.basename(src))

        eager = read_vrt(str(vrt))
        lazy = read_vrt(str(vrt), chunks=2)

        assert lazy.data.chunks == ((2, 2), (2, 2, 2))
        np.testing.assert_array_equal(lazy.compute().values, eager.values)

    def test_chunks_does_not_read_sources_during_construction(self, tmp_path):
        """The chunked path runs a cheap ``os.path.exists`` sweep at build
        but must not open or decode any source file.

        Pairing the missing source with ``missing_sources='warn'`` lets
        the build succeed; the assertion is that no decode-time warnings
        (which would only fire if a source were actually read) leak out
        during construction.
        """
        vrt = tmp_path / "tmp_1798_missing_source.vrt"
        _vrttail_write_single_band_vrt(vrt, "missing.tif")

        with warnings.catch_warnings(record=True) as caught:
            lazy = read_vrt(str(vrt), chunks=2, missing_sources="warn")

        assert caught == []
        assert hasattr(lazy.data, 'compute')

    def test_chunks_rejects_excessive_task_count(self, tmp_path):
        vrt = tmp_path / "tmp_1798_huge_extent.vrt"
        vrt.write_text(
            '<VRTDataset rasterXSize="100000" rasterYSize="100000">\n'
            '  <VRTRasterBand dataType="Byte" band="1"/>\n'
            '</VRTDataset>\n'
        )
        with pytest.raises(ValueError, match="task cap"):
            read_vrt(str(vrt), chunks=1, max_pixels=20_000_000_000)


class TestVrtTailDirectDaskKwargs:
    """read_geotiff_dask('.vrt') forwards VRT kwargs (#1795)."""

    def test_forwards_window_and_band(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
        src = tmp_path / "tmp_1797_source.tif"
        to_geotiff(arr, str(src), compression='none')
        vrt = tmp_path / "tmp_1797_source.vrt"
        _vrttail_write_multi_band_vrt(vrt, os.path.basename(src), bands=2)

        got = read_geotiff_dask(
            str(vrt), chunks=2, window=(1, 2, 4, 6), band=1,
        )
        assert got.shape == (3, 4)
        np.testing.assert_array_equal(got.values, arr[1:4, 2:6, 1])

    def test_forwards_max_pixels(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.arange(24, dtype=np.float32).reshape(4, 6)
        src = tmp_path / "tmp_1797_source_cap.tif"
        to_geotiff(arr, str(src), compression='none')
        vrt = tmp_path / "tmp_1797_source_cap.vrt"
        _vrttail_write_single_band_vrt(vrt, os.path.basename(src))

        with pytest.raises(ValueError, match="exceed"):
            read_geotiff_dask(str(vrt), chunks=2, max_pixels=10)
