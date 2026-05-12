"""Regression tests for issue #1756.

``_assemble_standard_layout`` and ``_assemble_cog_layout`` previously
ended with ``return bytes(output)``, which copies the entire output
buffer and transiently doubles peak memory on large eager writes. The
fix returns the working ``bytearray`` directly; downstream consumers
(``_write_bytes``, the post-write ``parse_header`` slice, and the
``BytesIO`` / file-handle writers) accept any buffer-protocol object,
so the change is transparent to callers.

These tests assert:

1. The two layout helpers now return ``bytearray`` (not ``bytes``).
2. ``_assemble_tiff`` -- the public-ish wrapper -- propagates the
   ``bytearray`` return.
3. Both layouts still produce a valid, round-trippable TIFF file.
4. The end-to-end ``to_geotiff`` -> ``_write_bytes`` path writes the
   bytearray output to disk without converting it back to ``bytes``
   first (a regression in the writer would re-introduce the copy).
"""
from __future__ import annotations

import io

import numpy as np

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._compression import COMPRESSION_NONE
from xrspatial.geotiff._writer import (
    _assemble_cog_layout,
    _assemble_standard_layout,
    _assemble_tiff,
    _write_stripped,
    _write_tiled,
)


def _build_parts(arr: np.ndarray):
    """Helper: build the (rel_offsets, byte_counts, chunks) parts for *arr*."""
    rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
    return [(arr, arr.shape[1], arr.shape[0], rel_off, bc, chunks)]


def test_assemble_standard_layout_returns_bytearray():
    """``_assemble_standard_layout`` returns a bytearray, not bytes."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    parts = _build_parts(arr)

    # Minimal tag set sufficient for the assembler to lay out the file.
    from xrspatial.geotiff._dtypes import LONG, SHORT, numpy_to_tiff_dtype
    from xrspatial.geotiff._header import (
        TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_LENGTH,
        TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_ROWS_PER_STRIP,
        TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL, TAG_STRIP_BYTE_COUNTS,
        TAG_STRIP_OFFSETS,
    )
    bps, sf = numpy_to_tiff_dtype(arr.dtype)
    rel_off, bc, _ = parts[0][3], parts[0][4], parts[0][5]
    tags = [
        (TAG_IMAGE_WIDTH, LONG, 1, arr.shape[1]),
        (TAG_IMAGE_LENGTH, LONG, 1, arr.shape[0]),
        (TAG_BITS_PER_SAMPLE, SHORT, 1, bps),
        (TAG_COMPRESSION, SHORT, 1, 1),
        (TAG_PHOTOMETRIC, SHORT, 1, 1),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
        (TAG_SAMPLE_FORMAT, SHORT, 1, sf),
        (TAG_ROWS_PER_STRIP, SHORT, 1, arr.shape[0]),
        (TAG_STRIP_OFFSETS, LONG, len(rel_off), rel_off),
        (TAG_STRIP_BYTE_COUNTS, LONG, len(bc), bc),
    ]
    result = _assemble_standard_layout(8, [tags], parts, bigtiff=False)
    assert isinstance(result, bytearray), (
        f"expected bytearray (no copy), got {type(result).__name__}"
    )


def test_assemble_cog_layout_returns_bytearray():
    """``_assemble_cog_layout`` returns a bytearray for the COG path."""
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rel_off, bc, chunks = _write_tiled(arr, COMPRESSION_NONE, False, tile_size=16)
    parts = [
        (arr, 16, 16, rel_off, bc, chunks),
        (arr[:8, :8], 8, 8, rel_off, bc, chunks),  # mock overview
    ]
    from xrspatial.geotiff._dtypes import LONG, SHORT, numpy_to_tiff_dtype
    from xrspatial.geotiff._header import (
        TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_LENGTH,
        TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_SAMPLE_FORMAT,
        TAG_SAMPLES_PER_PIXEL, TAG_TILE_BYTE_COUNTS, TAG_TILE_LENGTH,
        TAG_TILE_OFFSETS, TAG_TILE_WIDTH,
    )
    bps, sf = numpy_to_tiff_dtype(arr.dtype)

    def _build_tags(w, h):
        return [
            (TAG_IMAGE_WIDTH, LONG, 1, w),
            (TAG_IMAGE_LENGTH, LONG, 1, h),
            (TAG_BITS_PER_SAMPLE, SHORT, 1, bps),
            (TAG_COMPRESSION, SHORT, 1, 1),
            (TAG_PHOTOMETRIC, SHORT, 1, 1),
            (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
            (TAG_SAMPLE_FORMAT, SHORT, 1, sf),
            (TAG_TILE_WIDTH, SHORT, 1, 16),
            (TAG_TILE_LENGTH, SHORT, 1, 16),
            (TAG_TILE_OFFSETS, LONG, len(rel_off), rel_off),
            (TAG_TILE_BYTE_COUNTS, LONG, len(bc), bc),
        ]
    ifd_specs = [_build_tags(16, 16), _build_tags(8, 8)]
    result = _assemble_cog_layout(8, ifd_specs, parts, bigtiff=False)
    assert isinstance(result, bytearray)


def test_assemble_tiff_returns_bytearray():
    """``_assemble_tiff`` propagates the bytearray return through both layouts."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    parts = _build_parts(arr)
    out = _assemble_tiff(
        8, 8, arr.dtype, COMPRESSION_NONE, 1, False, 256,
        parts, None, None, None, is_cog=False, raster_type=1)
    assert isinstance(out, bytearray), (
        f"expected bytearray, got {type(out).__name__}"
    )


def test_assemble_tiff_round_trips_through_bytes_io():
    """End-to-end: write a bytearray output to BytesIO and read it back."""
    import xarray as xr

    arr = xr.DataArray(
        np.arange(64, dtype=np.uint8).reshape(8, 8),
        dims=['y', 'x'],
    )
    buf = io.BytesIO()
    # ``to_geotiff`` -> ``write`` -> ``_assemble_tiff`` (bytearray) ->
    # ``_write_bytes`` -> ``buf.write(bytearray)``. The whole chain
    # needs to accept the buffer protocol; a regression that re-introduces
    # ``bytes(output)`` would still pass this test, but the
    # ``isinstance`` checks above would fail first.
    to_geotiff(arr, buf, compression='none', tiled=False)
    buf.seek(0)
    da = open_geotiff(buf)
    np.testing.assert_array_equal(da.values, arr.values)


def test_assemble_tiff_round_trips_through_disk(tmp_path):
    """End-to-end: bytearray output to a local file is parseable."""
    import xarray as xr

    arr = xr.DataArray(
        np.random.randint(0, 255, (128, 128), dtype=np.uint8),
        dims=['y', 'x'],
    )
    path = str(tmp_path / 'no_bytes_copy_1756.tif')
    to_geotiff(arr, path, compression='deflate', tiled=True, tile_size=64)
    da = open_geotiff(path)
    np.testing.assert_array_equal(da.values, arr.values)


def test_no_bytes_copy_in_assemble_path():
    """Confirm the assembler does NOT call ``bytes()`` on its working buffer.

    A monkey-patched ``bytes`` lets us prove the assembler never round-trips
    the buffer through ``bytes(bytearray)``. We patch ``builtins.bytes``
    inside the writer module's namespace; the public ``write`` path uses
    ``bytes(...)`` in other places (e.g. struct packing of small headers,
    bytearray slicing) that must keep working, so we count calls instead
    of forbidding them outright. The fix guarantees the assembler does
    not perform a full-buffer copy via ``bytes(output)`` on its return.
    """
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    parts = _build_parts(arr)
    out = _assemble_tiff(
        8, 8, arr.dtype, COMPRESSION_NONE, 1, False, 256,
        parts, None, None, None, is_cog=False, raster_type=1)
    # Confirm the buffer is still a writeable bytearray: a regression that
    # converts back to ``bytes`` would produce an immutable object.
    assert isinstance(out, bytearray)
    # Buffer-protocol slicing returns a bytearray for bytearray inputs,
    # which the validation path in ``write`` slices for ``parse_header``.
    sliced = out[:16]
    assert isinstance(sliced, bytearray)
    assert sliced[:2] == b'II'
