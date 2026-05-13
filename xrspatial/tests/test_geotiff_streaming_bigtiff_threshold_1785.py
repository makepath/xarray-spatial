"""Regression tests for issue #1785.

The streaming writer's auto-BigTIFF decision used to compare only the
uncompressed pixel-data size against ``UINT32_MAX``. For rasters just
under 4 GiB the IFD plus the strip/tile-offset table pushed the actual
file past the classic-TIFF uint32 offset ceiling, and the write failed
late with ``struct.error``.

These tests pin the corrected decision: the helper accounts for IFD and
strip-table overhead, and the comparison is ``>=`` because
``UINT32_MAX`` itself is not a referenceable offset in classic TIFF.
The explicit ``bigtiff=True``/``False`` user override must still win.
"""
from __future__ import annotations

import struct

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._writer import _should_use_bigtiff_streaming


UINT32_MAX = 0xFFFFFFFF


# -- Helper-level unit tests ------------------------------------------------

class TestShouldUseBigTIFFStreaming:
    def test_just_under_uint32_max_promotes(self):
        """uncompressed = UINT32_MAX - 50 with non-trivial overhead promotes.

        Even ~50 bytes of slack disappears once IFD + strip-table overhead
        is added, so this case must promote to BigTIFF.
        """
        # 1024 entries => 8 KiB strip-table overhead, well above 50 bytes.
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=UINT32_MAX - 50,
            n_entries=1024,
        ) is True

    def test_half_uint32_max_stays_classic(self):
        """uncompressed = UINT32_MAX // 2 stays classic."""
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=UINT32_MAX // 2,
            n_entries=1024,
        ) is False

    def test_exactly_uint32_max_promotes(self):
        """Boundary: uncompressed == UINT32_MAX promotes because of ``>=``.

        Classic TIFF cannot reference an offset equal to UINT32_MAX (the
        last valid offset is UINT32_MAX - 1), so equality must promote.
        """
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=UINT32_MAX,
            n_entries=1,
        ) is True

    def test_small_raster_no_overhead_stays_classic(self):
        """Small rasters with one strip stay classic."""
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=1024,
            n_entries=1,
        ) is False

    def test_large_strip_table_alone_can_promote(self):
        """Even a small pixel payload can need BigTIFF if n_entries is huge.

        This documents the strip-table contribution: ~536 M entries puts
        the table itself near 4 GiB. Not realistic in practice, but it
        proves the overhead is wired through.
        """
        # Choose n_entries so that 8 * n_entries alone is close to UINT32_MAX.
        n_entries = (UINT32_MAX // 8) + 1
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=0,
            n_entries=n_entries,
        ) is True

    def test_overhead_pushes_just_under_threshold_over(self):
        """Regression: a payload that fits classic by raw bytes but not
        once header + IFD + strip table is added must promote.
        """
        # Pick overhead components large enough that the sum crosses
        # UINT32_MAX even though uncompressed_bytes < UINT32_MAX.
        n_entries = 100_000  # 800 KB strip table
        uncompressed = UINT32_MAX - (8 + 200 + n_entries * 8)
        # Sum equals exactly UINT32_MAX -> promote (>=).
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=uncompressed,
            n_entries=n_entries,
        ) is True
        # One byte less and we stay classic.
        assert _should_use_bigtiff_streaming(
            uncompressed_bytes=uncompressed - 1,
            n_entries=n_entries,
        ) is False


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
        path = str(tmp_path / 'force_bigtiff_1785.tif')
        to_geotiff(small_dask_raster, path, bigtiff=True)
        assert _read_tiff_magic(path) == 43

    def test_bigtiff_false_forces_classic_on_small_raster(
            self, small_dask_raster, tmp_path):
        path = str(tmp_path / 'force_classic_1785.tif')
        to_geotiff(small_dask_raster, path, bigtiff=False)
        assert _read_tiff_magic(path) == 42

    def test_bigtiff_none_small_raster_stays_classic(
            self, small_dask_raster, tmp_path):
        path = str(tmp_path / 'auto_classic_1785.tif')
        to_geotiff(small_dask_raster, path, bigtiff=None)
        assert _read_tiff_magic(path) == 42
