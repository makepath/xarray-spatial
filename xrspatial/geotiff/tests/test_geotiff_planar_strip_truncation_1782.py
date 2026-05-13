"""Regression tests for issue #1782.

``_read_strips`` pre-flighted the strip table against
``ceil(height/RowsPerStrip)``, which is the count for
``PlanarConfiguration=1`` (chunky). For ``PlanarConfiguration=2`` (separate)
the table must hold ``strips_per_band * samples_per_pixel`` entries, one
run per sample plane. The old planar branch silently skipped strips whose
global index walked off the end of the truncated table, leaving regions
of the ``np.empty`` output uninitialised.

These tests pin the contract:

* a planar=2 RGB strip table truncated to a single band's worth of strips
  raises a typed ``ValueError`` that names the planar layout and the
  expected count;
* a correctly-formed planar=2 RGB strip table reads back the per-band
  pixel values it was written with;
* a chunky single-band file with a truncated strip table still raises
  the existing typed error (no regression on the non-planar path).

Tests build minimal TIFFs from raw bytes via ``struct.pack`` to keep them
independent of tifffile / rasterio / gdal.
"""
from __future__ import annotations

import io
import math
import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff.tests.conftest import make_minimal_tiff


# ---------------------------------------------------------------------------
# Hand-rolled planar=2 stripped TIFF builder
# ---------------------------------------------------------------------------

def _make_planar2_stripped_tiff(
    width: int,
    height: int,
    bands: int,
    data: np.ndarray,
    *,
    rows_per_strip: int,
    truncate_strip_table_to: int | None = None,
) -> bytes:
    """Build an uncompressed ``PlanarConfiguration=2`` stripped TIFF.

    ``data`` is shaped ``(bands, height, width)``. ``rows_per_strip``
    controls how many image rows each strip covers (per band).

    If ``truncate_strip_table_to`` is set, both ``StripOffsets`` (273)
    and ``StripByteCounts`` (279) are written with only that many entries
    instead of the full ``strips_per_band * bands`` count. This is the
    corruption pattern from issue #1782.
    """
    bo = '<'
    assert data.shape == (bands, height, width)
    dtype = data.dtype
    bps = dtype.itemsize * 8
    sf = 1  # unsigned int

    strips_per_band = math.ceil(height / rows_per_strip)

    # planar=2: every strip for band 0, then every strip for band 1, ...
    strip_blobs: list[bytes] = []
    for b in range(bands):
        for s in range(strips_per_band):
            r0 = s * rows_per_strip
            r1 = min(r0 + rows_per_strip, height)
            strip = data[b, r0:r1, :]
            strip_blobs.append(strip.tobytes())

    pixel_bytes = b''.join(strip_blobs)
    full_byte_counts = [len(b) for b in strip_blobs]
    full_num = len(strip_blobs)

    # The strip table that gets *written* to the header. When
    # ``truncate_strip_table_to`` is set, the header advertises fewer
    # entries than the file actually contains.
    if truncate_strip_table_to is None:
        table_len = full_num
    else:
        table_len = truncate_strip_table_to
    table_byte_counts = full_byte_counts[:table_len]

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_shorts(tag, vals):
        tag_list.append(
            (tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals))
        )

    def add_longs(tag, vals):
        tag_list.append(
            (tag, 4, len(vals), struct.pack(f'{bo}{len(vals)}I', *vals))
        )

    add_short(256, width)
    add_short(257, height)
    add_shorts(258, [bps] * bands)            # BitsPerSample
    add_short(259, 1)                         # Compression = none
    add_short(262, 2 if bands >= 3 else 1)    # Photometric (RGB or BlackIsZero)
    add_longs(273, [0] * table_len)           # StripOffsets placeholder
    add_short(277, bands)                     # SamplesPerPixel
    add_short(278, rows_per_strip)            # RowsPerStrip
    add_longs(279, table_byte_counts)         # StripByteCounts (matches table_len)
    add_short(284, 2)                         # PlanarConfiguration = Separate
    add_shorts(339, [sf] * bands)             # SampleFormat per band

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4

    # First pass: compute overflow size and the pixel-data start.
    overflow_buf = bytearray()
    for _tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
    overflow_start = ifd_start + ifd_size
    pixel_data_start = overflow_start + len(overflow_buf)

    # Patch StripOffsets with real byte positions for the *advertised*
    # entries only. (When truncated, only the first ``table_len`` strips
    # are addressable; the rest of the pixel bytes still sit in the file
    # but are unreachable through the strip table.)
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            offs = []
            pos = 0
            for i in range(table_len):
                offs.append(pixel_data_start + pos)
                pos += full_byte_counts[i]
            new_raw = struct.pack(f'{bo}{table_len}I', *offs)
            patched.append((tag, typ, count, new_raw))
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

    out.extend(struct.pack(f'{bo}I', 0))  # next IFD = none
    out.extend(overflow_buf)
    out.extend(pixel_bytes)
    return bytes(out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _rgb_4x4() -> np.ndarray:
    """Deterministic 3-band 4x4 uint8 raster shaped (3, 4, 4)."""
    rng = np.random.RandomState(1782)
    return rng.randint(0, 256, size=(3, 4, 4), dtype=np.uint8)


def test_planar_strip_table_truncated_raises_typed_error():
    """Issue #1782: planar=2 strip table holding only one band's worth of
    strips must raise ``ValueError`` naming the planar layout and the
    expected entry count, instead of silently returning a partially
    initialised buffer.
    """
    data = _rgb_4x4()
    rps = 2
    strips_per_band = math.ceil(data.shape[1] / rps)  # = 2
    # Advertise only one band's worth of strips (2 entries) instead of
    # the required strips_per_band * bands = 6.
    tiff = _make_planar2_stripped_tiff(
        width=4,
        height=4,
        bands=3,
        data=data,
        rows_per_strip=rps,
        truncate_strip_table_to=strips_per_band,
    )

    with pytest.raises(ValueError) as exc:
        open_geotiff(io.BytesIO(tiff))

    msg = str(exc.value).lower()
    assert "planar" in msg, f"error should mention planar layout: {exc.value!r}"
    # The error must surface the *real* expected count, not the chunky one.
    assert "6" in str(exc.value), (
        f"error should name the planar expected count (6 = 2 strips x 3 "
        f"samples): {exc.value!r}"
    )


def test_planar_strip_table_complete_reads_correct_pixels():
    """A correctly-formed planar=2 RGB stripped TIFF must read back the
    per-band pixel values it was written with. Guards against the planar
    branch losing pixels once the pre-flight check is tightened.
    """
    data = _rgb_4x4()
    tiff = _make_planar2_stripped_tiff(
        width=4,
        height=4,
        bands=3,
        data=data,
        rows_per_strip=2,
        truncate_strip_table_to=None,  # full strip table
    )

    da = open_geotiff(io.BytesIO(tiff))
    arr = np.asarray(da)

    # The CPU reader returns (y, x, band) for multi-band stripped files.
    # Compare against the writer-side (band, y, x) layout transposed the
    # same way.
    expected = np.transpose(data, (1, 2, 0))
    if arr.ndim == 3 and arr.shape == expected.shape:
        np.testing.assert_array_equal(arr, expected)
    elif arr.ndim == 3 and arr.shape == data.shape:
        np.testing.assert_array_equal(arr, data)
    else:
        raise AssertionError(
            f"unexpected output shape {arr.shape}; expected one of "
            f"{expected.shape} (y,x,band) or {data.shape} (band,y,x)"
        )


def test_chunky_single_band_truncated_strip_table_still_raises():
    """Regression guard for the non-planar path. A chunky (planar=1)
    single-band TIFF whose strip table is shorter than ``ceil(height/rps)``
    must keep raising the existing typed ``ValueError`` after the
    planar-aware tightening.
    """
    # Reuse the project's minimal-TIFF builder, then truncate its
    # ``StripByteCounts`` count field. ``test_fuzz_hypothesis_1661`` uses
    # the same byte offset (110) for this corruption.
    base = make_minimal_tiff(4, 4, np.dtype('float32'))
    mut = bytearray(base)
    mut[110] = 0  # zero StripByteCounts count -> truncates the strip table

    with pytest.raises(ValueError):
        open_geotiff(io.BytesIO(bytes(mut)))
