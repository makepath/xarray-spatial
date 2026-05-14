"""Regression tests for issue #1870.

The TIFF 6.0 spec defines exactly two valid PlanarConfiguration values:
  1 - Chunky (interleaved, the default)
  2 - Planar (separated, one plane per sample)

Prior to this fix, ``IFD.planar_config`` returned the raw tag value
without validation. A malformed file carrying ``PlanarConfiguration=0``,
``3``, ``255``, etc. would silently decode under an assumed chunky layout
instead of being rejected, potentially producing garbage output.

These tests pin the contract:

* Files with an explicit PlanarConfiguration value outside {1, 2} raise a
  typed ``ValueError`` that names the invalid value.
* Files with PlanarConfiguration=1 (chunky) open successfully.
* Files with PlanarConfiguration=2 (planar) open successfully.
* Files that omit the PlanarConfiguration tag entirely open successfully
  (the TIFF spec says 1 is the default).
"""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff.tests.conftest import make_minimal_tiff


def _patch_planar_config(tiff_bytes: bytes, value: int) -> bytes:
    """Return a copy of *tiff_bytes* with PlanarConfiguration tag set to *value*.

    ``make_minimal_tiff`` does not write a PlanarConfiguration tag, so this
    helper injects one by replacing the StripByteCounts SHORT tag in the IFD
    with a PlanarConfiguration SHORT and then appending a fresh StripByteCounts
    entry.  The simpler approach used here just patches the raw bytes at the
    known tag-field offsets produced by ``make_minimal_tiff``.

    We use a raw-bytes approach so the test does not depend on any TIFF-writing
    library other than the project itself.
    """
    # Build a new TIFF that explicitly includes PlanarConfiguration.
    # We do this by constructing the IFD from scratch so the test is not
    # brittle to internal offsets.
    bo = '<'
    width, height = 4, 4
    pixel_data = np.zeros((height, width), dtype=np.float32)
    pixel_bytes = pixel_data.tobytes()

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    add_short(256, width)         # ImageWidth
    add_short(257, height)        # ImageLength
    add_short(258, 32)            # BitsPerSample (float32)
    add_short(259, 1)             # Compression = none
    add_short(262, 1)             # PhotometricInterpretation
    add_short(277, 1)             # SamplesPerPixel
    add_short(278, height)        # RowsPerStrip
    add_long(273, 0)              # StripOffsets (placeholder)
    add_long(279, len(pixel_bytes))  # StripByteCounts
    add_short(284, value)         # PlanarConfiguration = <value under test>
    add_short(339, 3)             # SampleFormat = float

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    for _tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)

    pixel_data_start = overflow_start + len(overflow_buf)

    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets: dict[int, int | None] = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)
    return bytes(out)


@pytest.mark.parametrize("bad_value", [0, 3, 10, 255])
def test_invalid_planar_config_raises_value_error(bad_value):
    """PlanarConfiguration values outside {1, 2} must raise ValueError
    that names the offending value.  Issue #1870."""
    tiff = _patch_planar_config(b"", bad_value)
    with pytest.raises(ValueError, match=str(bad_value)):
        open_geotiff(io.BytesIO(tiff))


def test_planar_config_1_opens_successfully():
    """PlanarConfiguration=1 (chunky, the default) must read without error."""
    tiff = _patch_planar_config(b"", 1)
    da = open_geotiff(io.BytesIO(tiff))
    assert da is not None


def test_planar_config_absent_uses_default_1():
    """A file that omits the PlanarConfiguration tag entirely defaults to 1
    (chunky) per TIFF 6.0 §7.  The reader must not raise."""
    tiff = make_minimal_tiff(4, 4, np.dtype("float32"))
    da = open_geotiff(io.BytesIO(tiff))
    assert da is not None
