"""``allow_rotated=True`` works for rotated GeoTIFFs (issue #2115).

Before the fix, ``open_geotiff(..., allow_rotated=True)`` raised
``NotImplementedError`` for a rotated ``ModelTransformationTag``
because the geotag parser short-circuited before the validator's
opt-out could be consulted. The VRT path honoured the kwarg; the
GeoTIFF path did not.

These tests pin the current behaviour:

* Default (``allow_rotated=False``) raises ``RotatedTransformError``
  (issue #2267; previously ``NotImplementedError``) so an existing
  caller that does not know about the opt-out keeps the safety net.
* ``allow_rotated=True`` drops the georef and reads the pixel grid
  without raising. The pixel values match what the file actually holds.
* The rotated 6-tuple is preserved on ``geo_info.transform.rotated_affine``
  so callers / attrs-roundtrip code can see the matrix.
* Falls through to the same path on dask reads.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._errors import RotatedTransformError
from xrspatial.geotiff._geotags import (
    GeoTransform,
    TAG_MODEL_TRANSFORMATION,
    _extract_transform,
)
from xrspatial.geotiff._header import IFD, IFDEntry


def _make_ifd_with_rotated_transform(matrix_16: tuple) -> IFD:
    """Build a synthetic IFD that carries only a rotated ModelTransformationTag.

    The other geo-related tags stay absent so the transform branch is the
    only one exercised; this keeps the unit tests independent of the full
    geo-key plumbing.
    """
    ifd = IFD()
    ifd.entries[TAG_MODEL_TRANSFORMATION] = IFDEntry(
        tag=TAG_MODEL_TRANSFORMATION,
        type_id=12,  # DOUBLE
        count=16,
        value=tuple(matrix_16),
    )
    return ifd


# A 30-degree rotation with pixel size 10, origin (100, 200).
# Row-major 4x4 in GeoTIFF order:
#   x = M[0]*col + M[1]*row + M[2]*z + M[3]
#   y = M[4]*col + M[5]*row + M[6]*z + M[7]
_COS30 = 0.8660254037844387
_SIN30 = 0.5
_ROTATED_M = (
    10.0 * _COS30, -10.0 * _SIN30, 0.0, 100.0,  # x row
    10.0 * _SIN30, 10.0 * _COS30, 0.0, 200.0,  # y row
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def test_extract_transform_rejects_rotated_by_default():
    ifd = _make_ifd_with_rotated_transform(_ROTATED_M)
    with pytest.raises(RotatedTransformError, match="rotation"):
        _extract_transform(ifd)


def test_extract_transform_allow_rotated_returns_no_georef():
    ifd = _make_ifd_with_rotated_transform(_ROTATED_M)
    gt, has_georef = _extract_transform(ifd, allow_rotated=True)
    assert isinstance(gt, GeoTransform)
    assert has_georef is False
    # The rotated 6-tuple should be preserved for downstream consumers.
    expected = (
        _ROTATED_M[0], _ROTATED_M[1], _ROTATED_M[3],
        _ROTATED_M[4], _ROTATED_M[5], _ROTATED_M[7],
    )
    assert gt.rotated_affine is not None
    assert gt.rotated_affine == pytest.approx(expected)


def test_extract_transform_allow_rotated_passes_through_axis_aligned():
    """Axis-aligned files should be unchanged by the opt-in path."""
    aligned = (
        10.0, 0.0, 0.0, 100.0,
        0.0, -10.0, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    ifd = _make_ifd_with_rotated_transform(aligned)
    gt, has_georef = _extract_transform(ifd, allow_rotated=True)
    assert has_georef is True
    assert gt.pixel_width == 10.0
    assert gt.pixel_height == -10.0
    assert gt.origin_x == 100.0
    assert gt.origin_y == 200.0
    assert gt.rotated_affine is None


def _write_rotated_tiff(path, arr: np.ndarray) -> None:
    """Write a minimal little-endian TIFF with a rotated ModelTransformationTag.

    Single-band, single-strip, uncompressed; just enough to exercise the
    reader's rotation path without depending on rasterio/GDAL.
    """
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    # Layout:
    #   header (8 bytes)
    #   strip pixels (h*w*2)
    #   IFD (entry_count u16, N*12 bytes entries, next_ifd u32)
    #   transform doubles area (16*8)
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    entries = []
    # Each entry is (tag, type_id, count, value_or_offset).
    # When the value fits in 4 bytes it is inlined, otherwise it is an
    # offset to the value area. We use the value-area only for
    # ModelTransformationTag and the strip-offset for cleanliness.
    entries.append((256, 3, 1, w))      # ImageWidth (SHORT, fits inline)
    entries.append((257, 3, 1, h))      # ImageLength
    entries.append((258, 3, 1, 16))     # BitsPerSample = 16
    entries.append((259, 3, 1, 1))      # Compression = none
    entries.append((262, 3, 1, 1))      # Photometric = BlackIsZero
    entries.append((273, 4, 1, header_size))  # StripOffsets
    entries.append((277, 3, 1, 1))      # SamplesPerPixel
    entries.append((278, 3, 1, h))      # RowsPerStrip
    entries.append((279, 4, 1, strip_size))  # StripByteCounts
    entries.append((339, 3, 1, 1))      # SampleFormat = unsigned int
    entries.append((TAG_MODEL_TRANSFORMATION, 12, 16, transform_off))

    entries.sort(key=lambda e: e[0])
    ifd_bytes = struct.pack('<H', len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:  # SHORT
            ifd_bytes += struct.pack('<HHIHH', tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack('<HHII', tag, type_id, count, val)
    ifd_bytes += struct.pack('<I', 0)  # next IFD

    with open(path, 'wb') as f:
        # Header: little-endian, magic 42, IFD offset
        f.write(struct.pack('<HHI', 0x4949, 42, ifd_off))
        f.write(arr.tobytes())
        f.write(struct.pack('<16d', *_ROTATED_M))
        f.write(ifd_bytes)


def test_open_geotiff_rotated_default_raises(tmp_path):
    src = tmp_path / "rotated_2115_default.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(str(src), arr)
    with pytest.raises(RotatedTransformError, match="rotation"):
        open_geotiff(str(src))


def test_open_geotiff_rotated_allow_rotated_reads_pixels(tmp_path):
    src = tmp_path / "rotated_2115_optin.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(str(src), arr)
    da = open_geotiff(str(src), allow_rotated=True)
    assert da.shape == arr.shape
    np.testing.assert_array_equal(da.values, arr)
    # Without a usable georef the reader should not emit a transform
    # attribute that pretends to be axis-aligned. The rotated 6-tuple
    # is preserved on geo_info via the parser, which downstream attrs
    # roundtrip code can pick up; the public attrs themselves remain
    # consistent with no-georef behaviour.
    # The default attrs include the rotated affine if the writer chose
    # to expose it; here we just confirm there is no axis-aligned
    # transform that would mislead spatial ops.
    assert da.attrs.get('transform') in (None, 'rotated')


def test_open_geotiff_rotated_default_raises_with_dask(tmp_path):
    src = tmp_path / "rotated_2115_dask_default.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(str(src), arr)
    with pytest.raises(RotatedTransformError, match="rotation"):
        open_geotiff(str(src), chunks=2)


def test_open_geotiff_rotated_allow_rotated_with_dask(tmp_path):
    src = tmp_path / "rotated_2115_dask_optin.tif"
    arr = np.arange(40, dtype='<u2').reshape(5, 8)
    _write_rotated_tiff(str(src), arr)
    da = open_geotiff(str(src), allow_rotated=True, chunks=4)
    assert da.shape == arr.shape
    np.testing.assert_array_equal(da.compute().values, arr)


# ---------------------------------------------------------------------------
# HTTP path honours allow_rotated end-to-end.
# ---------------------------------------------------------------------------
def _start_http_server(directory):
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **kw):
            return

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_open_geotiff_http_rotated_default_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = tmp_path / "rotated_2115_http_default.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(str(src), arr)
    httpd = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/{src.name}"
        with pytest.raises(RotatedTransformError, match="rotation"):
            open_geotiff(url)
    finally:
        httpd.shutdown()


def test_open_geotiff_http_rotated_allow_rotated_reads_pixels(
        tmp_path, monkeypatch):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = tmp_path / "rotated_2115_http_optin.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(str(src), arr)
    httpd = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/{src.name}"
        da = open_geotiff(url, allow_rotated=True)
        assert da.shape == arr.shape
        np.testing.assert_array_equal(da.values, arr)
    finally:
        httpd.shutdown()
