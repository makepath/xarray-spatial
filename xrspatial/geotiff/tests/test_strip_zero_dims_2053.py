"""Regression tests for issue #2053.

The stripped TIFF read paths previously trusted ``ImageWidth``,
``ImageLength``, and ``SamplesPerPixel`` straight off the IFD. A
malformed file with any of those set to 0 (or with a count interpreted
as a negative-cast-to-huge-unsigned) would flow past the dimension
check, since :func:`xrspatial.geotiff._reader._check_dimensions` only
enforces the upper bound and the post-window clamp would collapse the
output to an empty array.

The fix is a two-layer defense:

1. :func:`_check_source_dimensions` rejects ``<= 0`` on width, height,
   or samples.
2. Both stripped read paths (``_read_strips`` for local files and
   ``_fetch_decode_cog_http_strips`` for HTTP COGs) call it right after
   reading the IFD, before any window clamping.

Tiled paths already validate through ``validate_tile_layout`` in
``_header.py``; these tests pin that parity so a future change to the
tiled path can't silently regress the stripped path.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._header import (TAG_IMAGE_LENGTH, TAG_IMAGE_WIDTH, TAG_SAMPLES_PER_PIXEL,
                                       parse_header)
from xrspatial.geotiff._reader import _check_source_dimensions

# ---------------------------------------------------------------------------
# Helpers: locate and patch a tag value inside a classic-TIFF IFD entry
# ---------------------------------------------------------------------------


def _find_ifd_entry_offset(buf: bytes, tag_id: int) -> int:
    """Return the byte offset of the IFD entry for ``tag_id``.

    Classic TIFF only. The IFD entry layout is 12 bytes:
    ``tag(2) + type(2) + count(4) + value/offset(4)``. We use the
    parsed header's ``first_ifd_offset``, then scan the entries.
    """
    header = parse_header(buf)
    assert not header.is_bigtiff, "helper only handles classic TIFF"
    bo = header.byte_order
    ifd_off = header.first_ifd_offset
    num_entries = struct.unpack_from(f'{bo}H', buf, ifd_off)[0]
    entry_base = ifd_off + 2
    for i in range(num_entries):
        entry_off = entry_base + i * 12
        tag = struct.unpack_from(f'{bo}H', buf, entry_off)[0]
        if tag == tag_id:
            return entry_off
    raise KeyError(f"Tag {tag_id} not found in IFD")


def _patch_inline_long(buf: bytearray, tag_id: int, new_value: int) -> None:
    """Patch the inline LONG value of an IFD entry to ``new_value``.

    Assumes the entry already stores its value inline (count=1 with a
    4-byte-or-smaller type). For ``ImageWidth`` / ``ImageLength``
    written as LONG (type=4, count=1) by the standard writer this
    holds.
    """
    header = parse_header(bytes(buf))
    bo = header.byte_order
    entry_off = _find_ifd_entry_offset(bytes(buf), tag_id)
    type_id = struct.unpack_from(f'{bo}H', buf, entry_off + 2)[0]
    count = struct.unpack_from(f'{bo}I', buf, entry_off + 4)[0]
    assert count == 1, (
        f"helper only supports count=1 entries; got count={count} "
        f"for tag {tag_id}"
    )
    value_off = entry_off + 8
    if type_id == 4:  # LONG
        struct.pack_into(f'{bo}I', buf, value_off, new_value & 0xFFFFFFFF)
    elif type_id == 3:  # SHORT (2 bytes; upper 2 bytes of slot are padding)
        struct.pack_into(f'{bo}H', buf, value_off, new_value & 0xFFFF)
    else:
        raise AssertionError(
            f"unsupported type_id={type_id} for tag {tag_id}; helper handles "
            f"LONG and SHORT only"
        )


def _make_valid_stripped(tmp_path, *, height=16, width=8):
    """Write a small valid stripped TIFF and return its bytes + path."""
    arr = xr.DataArray(
        np.arange(height * width, dtype=np.uint8).reshape(height, width),
        dims=['y', 'x'],
    )
    path = str(tmp_path / 'valid_stripped_2053.tif')
    to_geotiff(arr, path, compression='none', tiled=False)
    with open(path, 'rb') as f:
        return bytearray(f.read()), path


def _make_valid_tiled(tmp_path, *, height=32, width=32, tile_size=16):
    """Write a small valid tiled TIFF and return its bytes + path."""
    arr = xr.DataArray(
        np.arange(height * width, dtype=np.uint8).reshape(height, width),
        dims=['y', 'x'],
    )
    path = str(tmp_path / 'valid_tiled_2053.tif')
    to_geotiff(arr, path, compression='none', tiled=True, tile_size=tile_size)
    with open(path, 'rb') as f:
        return bytearray(f.read()), path


# ---------------------------------------------------------------------------
# Unit tests on the helper itself
# ---------------------------------------------------------------------------

class TestCheckSourceDimensions:
    """The validator must reject every flavor of non-positive input."""

    def test_zero_width_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(0, 16, 1)

    def test_zero_height_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(16, 0, 1)

    def test_zero_samples_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(16, 16, 0)

    def test_negative_width_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(-1, 16, 1)

    def test_negative_height_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(16, -1, 1)

    def test_negative_samples_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(16, 16, -1)

    def test_all_positive_passes(self):
        # No exception => pass
        _check_source_dimensions(1, 1, 1)
        _check_source_dimensions(1024, 1024, 3)

    def test_error_message_contains_each_value(self):
        with pytest.raises(ValueError) as excinfo:
            _check_source_dimensions(0, 5, 7)
        msg = str(excinfo.value)
        assert "ImageWidth=0" in msg
        assert "ImageLength=5" in msg
        assert "SamplesPerPixel=7" in msg


# ---------------------------------------------------------------------------
# End-to-end: malformed stripped TIFFs are rejected by open_geotiff
# ---------------------------------------------------------------------------

class TestStrippedZeroDimsRejected:

    def test_zero_image_width_rejected(self, tmp_path):
        buf, _ = _make_valid_stripped(tmp_path)
        _patch_inline_long(buf, TAG_IMAGE_WIDTH, 0)
        bad_path = tmp_path / 'zero_width.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            open_geotiff(str(bad_path))

    def test_zero_image_length_rejected(self, tmp_path):
        buf, _ = _make_valid_stripped(tmp_path)
        _patch_inline_long(buf, TAG_IMAGE_LENGTH, 0)
        bad_path = tmp_path / 'zero_height.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            open_geotiff(str(bad_path))

    def test_zero_samples_per_pixel_rejected(self, tmp_path):
        buf, _ = _make_valid_stripped(tmp_path)
        # SamplesPerPixel is written as SHORT (type=3) by the writer.
        _patch_inline_long(buf, TAG_SAMPLES_PER_PIXEL, 0)
        bad_path = tmp_path / 'zero_samples.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            open_geotiff(str(bad_path))

    def test_negative_width_via_signed_cast_rejected(self, tmp_path):
        """A 32-bit pattern that looks like a negative signed int.

        Real-world TIFFs store ImageWidth as an unsigned LONG, so a
        "negative" value would surface as a huge unsigned int. Either
        the strict ``<= 0`` check rejects it directly, or the
        upper-bound ``_check_dimensions`` rejects it as oversized.
        Either error is acceptable here; the test pins that the file
        does not silently produce an empty array.
        """
        buf, _ = _make_valid_stripped(tmp_path)
        # 0xFFFFFFFF = -1 as int32, ~4.29B as uint32. Larger than
        # MAX_PIXELS_DEFAULT so the upper-bound check fires regardless.
        _patch_inline_long(buf, TAG_IMAGE_WIDTH, 0xFFFFFFFF)
        bad_path = tmp_path / 'huge_width.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError):
            open_geotiff(str(bad_path))


# ---------------------------------------------------------------------------
# Valid windowed-empty reads must keep working (option A in the design)
# ---------------------------------------------------------------------------

class TestWindowedEmptyStillAllowed:
    """The new check sits *before* window clamping. A caller passing
    a window entirely outside the image is still allowed to receive an
    empty result; the strict check only applies to source IFD dims.
    """

    def test_windowed_outside_image_returns_empty_not_error(self, tmp_path):
        buf, path = _make_valid_stripped(tmp_path, height=16, width=8)
        # Read the file through open_geotiff's window kwarg if it
        # supports one; otherwise call the lower-level reader directly.
        # We use the lower-level _read_strips because open_geotiff
        # doesn't expose a window kwarg consistently across versions.
        from xrspatial.geotiff._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
        from xrspatial.geotiff._header import parse_all_ifds
        from xrspatial.geotiff._reader import _read_strips

        data = bytes(buf)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        bps = resolve_bits_per_sample(ifd.bits_per_sample)
        dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)

        # Window starting at the image's bottom-right corner. After
        # clamping (r0 clamps up to height, c0 clamps up to width when
        # r1/c1 also clamp down), the post-window dims are (0, 0).
        # The image is 16 high x 8 wide; this picks a zero-area window
        # along the bottom edge. Use a window that doesn't exceed the
        # image dimensions on the lower bound (otherwise existing
        # negative-dim handling kicks in).
        edge_window = (ifd.height, 0, ifd.height, ifd.width)
        arr = _read_strips(data, ifd, header, dtype, window=edge_window)
        # r0 = 16 (clamped), r1 = 16 -> out_h = 0; c spans 0..8 -> out_w = 8.
        assert arr.shape[0] == 0, (
            f"expected zero-height array from edge window, got shape "
            f"{arr.shape}"
        )
        # The source dim check must NOT have rejected the valid source
        # IFD with width=8, height=16, samples=1; only the post-window
        # output is empty.


# ---------------------------------------------------------------------------
# Parity check: tiled path was already protected; pin it
# ---------------------------------------------------------------------------

class TestTiledParityPinned:
    """``validate_tile_layout`` already rejects zero w/h on tiled
    files. This pins that behavior so any refactor of the tiled
    validator that drops the check would surface here, not in
    production.
    """

    def test_tiled_zero_width_rejected(self, tmp_path):
        buf, _ = _make_valid_tiled(tmp_path)
        _patch_inline_long(buf, TAG_IMAGE_WIDTH, 0)
        bad_path = tmp_path / 'tiled_zero_width.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid"):
            open_geotiff(str(bad_path))

    def test_tiled_zero_height_rejected(self, tmp_path):
        buf, _ = _make_valid_tiled(tmp_path)
        _patch_inline_long(buf, TAG_IMAGE_LENGTH, 0)
        bad_path = tmp_path / 'tiled_zero_height.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid"):
            open_geotiff(str(bad_path))


# ---------------------------------------------------------------------------
# HTTP path: a malformed stripped COG over HTTP must also reject
# ---------------------------------------------------------------------------

class _StaticBytesHTTPSource:
    """Minimal ``_HTTPSource`` stand-in backed by a static buffer."""
    def __init__(self, buf: bytes):
        self._buf = buf
        self.read_all_called = False

    def read_range(self, start: int, length: int) -> bytes:
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        self.read_all_called = True
        return self._buf

    def read_ranges_coalesced(self, ranges, *, max_workers=8,
                              gap_threshold=0,
                              max_coalesced_range_bytes=None):
        return [self._buf[s:s + le] for (s, le) in ranges]

    def close(self):
        pass


class TestHTTPStrippedZeroDimsRejected:

    def test_zero_image_width_over_http_rejected(self, tmp_path, monkeypatch):
        buf, _ = _make_valid_stripped(tmp_path, height=64, width=32)
        _patch_inline_long(buf, TAG_IMAGE_WIDTH, 0)
        bad_bytes = bytes(buf)

        from xrspatial.geotiff import _reader as reader_mod
        monkeypatch.setattr(
            reader_mod, '_HTTPSource',
            lambda url, **kw: _StaticBytesHTTPSource(bad_bytes))

        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            reader_mod._read_cog_http('http://mock/bad.tif')

    def test_zero_image_length_over_http_rejected(self, tmp_path,
                                                  monkeypatch):
        buf, _ = _make_valid_stripped(tmp_path, height=64, width=32)
        _patch_inline_long(buf, TAG_IMAGE_LENGTH, 0)
        bad_bytes = bytes(buf)

        from xrspatial.geotiff import _reader as reader_mod
        monkeypatch.setattr(
            reader_mod, '_HTTPSource',
            lambda url, **kw: _StaticBytesHTTPSource(bad_bytes))

        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            reader_mod._read_cog_http('http://mock/bad.tif')
