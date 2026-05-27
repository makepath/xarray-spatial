"""Streaming / chunked read paths.

Covers the streaming-decision helper that the chunked write/read pipeline
uses to pick classic vs. BigTIFF, plus the integration check that the
user's ``bigtiff=`` override still wins on the streaming code path.

The streaming writer's auto-BigTIFF decision used to compare only the
uncompressed pixel-data size against ``UINT32_MAX``. For rasters just
under 4 GiB the IFD plus the strip/tile-offset table pushed the actual
file past the classic-TIFF uint32 offset ceiling, and the write failed
late with ``struct.error``.

The pinned contract:

* The helper takes an actual ``ifd_overhead_bytes`` value (computed from
  the real tag list via ``_compute_classic_ifd_overhead``) rather than a
  200-byte fudge constant; large ``gdal_metadata_xml`` or ``extra_tags``
  payloads must not silently undercount overhead.
* The comparison is ``> UINT32_MAX``, matching the eager
  ``_assemble_tiff`` decision. A file that is exactly ``UINT32_MAX``
  bytes still fits classic.
* The explicit ``bigtiff=True``/``False`` user override still wins.
"""
from __future__ import annotations

import struct

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._dtypes import ASCII, LONG, SHORT
from xrspatial.geotiff._header import (TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_GDAL_METADATA,
                                       TAG_IMAGE_LENGTH, TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC,
                                       TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL,
                                       TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS)
from xrspatial.geotiff._writer import _compute_classic_ifd_overhead, _should_use_bigtiff_streaming

UINT32_MAX = 0xFFFFFFFF


def _minimal_tag_list(n_entries: int, gdal_metadata_size: int = 0) -> list:
    """Build a representative classic-TIFF tag list for sizing.

    Mirrors the streaming writer's tag set for a stripped uint8 raster
    so ``_compute_classic_ifd_overhead`` returns a realistic byte count.
    ``gdal_metadata_size`` injects an ASCII overflow payload to model
    metadata-heavy writes.
    """
    tags = [
        (TAG_IMAGE_WIDTH, LONG, 1, 1),
        (TAG_IMAGE_LENGTH, LONG, 1, 1),
        (TAG_BITS_PER_SAMPLE, SHORT, 1, 8),
        (TAG_COMPRESSION, SHORT, 1, 1),
        (TAG_PHOTOMETRIC, SHORT, 1, 1),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
        (TAG_SAMPLE_FORMAT, SHORT, 1, 1),
        (TAG_STRIP_OFFSETS, LONG, n_entries, [0] * n_entries),
        (TAG_STRIP_BYTE_COUNTS, LONG, n_entries, [0] * n_entries),
    ]
    if gdal_metadata_size > 0:
        blob = 'x' * gdal_metadata_size
        tags.append((TAG_GDAL_METADATA, ASCII, len(blob) + 1, blob))
    return tags


# -- Helper-level unit tests ------------------------------------------------

class TestShouldUseBigTIFFStreaming:
    def test_just_under_uint32_max_promotes(self):
        """uncompressed = UINT32_MAX - 50 with non-trivial overhead promotes."""
        # 1024 entries: strip table contributes 8 * 1024 = 8 KiB.
        tags = _minimal_tag_list(n_entries=1024)
        overhead = _compute_classic_ifd_overhead(tags)
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=UINT32_MAX - 50,
            n_entries=0,
            ifd_overhead_bytes=overhead,
        ) is True

    def test_half_uint32_max_stays_classic(self):
        """uncompressed = UINT32_MAX // 2 stays classic."""
        tags = _minimal_tag_list(n_entries=1024)
        overhead = _compute_classic_ifd_overhead(tags)
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=UINT32_MAX // 2,
            n_entries=0,
            ifd_overhead_bytes=overhead,
        ) is False

    def test_exactly_uint32_max_stays_classic(self):
        """Boundary: total file size == UINT32_MAX bytes still fits classic."""
        tags = _minimal_tag_list(n_entries=1)
        overhead = _compute_classic_ifd_overhead(tags)
        header = 8
        uncompressed = UINT32_MAX - header - overhead
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=uncompressed,
            n_entries=0,
            ifd_overhead_bytes=overhead,
        ) is False
        # One byte over and we must promote.
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=uncompressed + 1,
            n_entries=0,
            ifd_overhead_bytes=overhead,
        ) is True

    def test_small_raster_no_overhead_stays_classic(self):
        """Small rasters with one strip stay classic."""
        tags = _minimal_tag_list(n_entries=1)
        overhead = _compute_classic_ifd_overhead(tags)
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=1024,
            n_entries=0,
            ifd_overhead_bytes=overhead,
        ) is False

    def test_large_strip_table_alone_can_promote(self):
        """Even a small pixel payload can need BigTIFF if n_entries is huge."""
        n_entries = (UINT32_MAX // 8) + 1
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=0,
            n_entries=n_entries,
            ifd_overhead_bytes=0,
        ) is True

    def test_overhead_pushes_just_under_threshold_over(self):
        """A payload that fits classic by raw bytes but not once header +
        IFD + strip table is added must promote.
        """
        n_entries = 100_000  # ~800 KB strip table
        tags = _minimal_tag_list(n_entries=n_entries)
        overhead = _compute_classic_ifd_overhead(tags)
        header = 8
        uncompressed = UINT32_MAX + 1 - header - overhead
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=uncompressed,
            n_entries=0,
            ifd_overhead_bytes=overhead,
        ) is True
        # One byte less and we stay classic (boundary is now ``>``).
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=uncompressed - 1,
            n_entries=0,
            ifd_overhead_bytes=overhead,
        ) is False

    def test_large_gdal_metadata_flips_decision(self):
        """A 5000-byte gdal_metadata blob must flip a borderline case."""
        n_entries = 1024
        big_blob = 5000  # ASCII overflow heap entry
        plain_tags = _minimal_tag_list(n_entries=n_entries)
        meta_tags = _minimal_tag_list(
            n_entries=n_entries, gdal_metadata_size=big_blob)

        plain_overhead = _compute_classic_ifd_overhead(plain_tags)
        meta_overhead = _compute_classic_ifd_overhead(meta_tags)
        assert meta_overhead - plain_overhead >= big_blob

        header = 8
        uncompressed = UINT32_MAX - header - plain_overhead
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=uncompressed,
            n_entries=0,
            ifd_overhead_bytes=plain_overhead,
        ) is False
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=uncompressed,
            n_entries=0,
            ifd_overhead_bytes=meta_overhead,
        ) is True


# -- Integration tests against the writer ------------------------------------


def _read_tiff_magic(path: str) -> int:
    """Return the TIFF version field: 42 (0x002A) classic, 43 (0x002B) BigTIFF."""
    with open(path, 'rb') as f:
        head = f.read(4)
    byte_order = head[:2]
    if byte_order == b'II':
        fmt = '<H'
    elif byte_order == b'MM':
        fmt = '>H'
    else:
        raise AssertionError(f"unexpected byte order {byte_order!r}")
    return struct.unpack(fmt, head[2:4])[0]


@pytest.fixture
def small_dask_raster():
    """64x64 float32 dask raster with the attrs to_geotiff needs."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    darr = da.from_array(arr, chunks=(32, 32))
    y = np.linspace(41.0, 40.0, 64)
    x = np.linspace(-106.0, -105.0, 64)
    return xr.DataArray(darr, dims=['y', 'x'],
                        coords={'y': y, 'x': x},
                        attrs={'crs': 4326, 'nodata': -9999.0})


class TestStreamingBigTIFFUserOverride:
    def test_bigtiff_true_forces_bigtiff_on_small_raster(
            self, small_dask_raster, tmp_path):
        path = str(tmp_path / 'force_bigtiff.tif')
        to_geotiff(small_dask_raster, path, bigtiff=True)
        assert _read_tiff_magic(path) == 43

    def test_bigtiff_false_forces_classic_on_small_raster(
            self, small_dask_raster, tmp_path):
        path = str(tmp_path / 'force_classic.tif')
        to_geotiff(small_dask_raster, path, bigtiff=False)
        assert _read_tiff_magic(path) == 42

    def test_bigtiff_none_small_raster_stays_classic(
            self, small_dask_raster, tmp_path):
        path = str(tmp_path / 'auto_classic.tif')
        to_geotiff(small_dask_raster, path, bigtiff=None)
        assert _read_tiff_magic(path) == 42
