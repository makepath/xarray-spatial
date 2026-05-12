"""Regression test for issue #1736.

The uncompressed tiled branch of ``xrspatial.geotiff._writer._write_tiled``
previously allocated a contiguous ``bytearray`` plus a memoryview
``(n_tiles * tw * th * bytes_per_sample * samples)`` bytes long at the
top of the loop and never read either back. Tile bytes were still
built via ``tile_arr.tobytes()`` and appended to a list. The dead
buffer roughly doubled peak memory for an uncompressed write.

The fix is a pure deletion. The tests below cover both behaviours
worth pinning:

* round-trip fidelity (writing an uncompressed tiled GeoTIFF must
  still read back identically with no holes between tiles); and
* peak-memory shape, by snapshotting ``tracemalloc`` peak across a
  direct ``_write_tiled`` call. The current implementation lands at
  roughly ``1.06x`` the raw raster size; the dead bytearray pushed it
  to ``~2.07x``. The threshold below (``1.5x``) catches any
  reintroduction of that allocation with comfortable headroom for
  unrelated implementation changes.
"""
from __future__ import annotations

import os
import tracemalloc
import uuid

import numpy as np
import xarray as xr

from xrspatial.geotiff import to_geotiff, open_geotiff
from xrspatial.geotiff._compression import COMPRESSION_NONE
from xrspatial.geotiff._writer import _write_tiled


# Peak ``tracemalloc`` size, in multiples of the input raster size, that
# the uncompressed branch of ``_write_tiled`` must stay under. The dead
# bytearray drove peak to ~2.07x; the current implementation sits at
# ~1.06-1.12x across the cases below. 1.5x leaves room for unrelated
# refactors while still firmly catching the regression.
_PEAK_RATIO_LIMIT = 1.5


def test_uncompressed_tiled_round_trip_exact(tmp_path):
    rng = np.random.RandomState(20260512)
    h, w = 96, 144
    data = rng.randint(0, 200, size=(h, w)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x'])

    p = str(tmp_path / f"tmp_1736_uncomp_{uuid.uuid4().hex[:8]}.tif")
    to_geotiff(da, p, tiled=True, tile_size=32, compression='none')
    assert os.path.exists(p)

    out = open_geotiff(p)
    np.testing.assert_array_equal(out.data, data)
    assert out.shape == (h, w)


def test_uncompressed_tiled_round_trip_partial_edge_tiles(tmp_path):
    """Tile size that does not divide width/height exercises the
    zero-padded edge-tile branch inside the loop."""
    rng = np.random.RandomState(20260513)
    h, w = 50, 70  # 32 does not divide either; edges pad
    data = rng.randint(0, 60000, size=(h, w)).astype(np.uint16)
    da = xr.DataArray(data, dims=['y', 'x'])

    p = str(tmp_path / f"tmp_1736_edge_{uuid.uuid4().hex[:8]}.tif")
    to_geotiff(da, p, tiled=True, tile_size=32, compression='none')

    out = open_geotiff(p)
    np.testing.assert_array_equal(out.data, data)


def test_uncompressed_tiled_round_trip_multiband(tmp_path):
    rng = np.random.RandomState(20260514)
    h, w, b = 48, 80, 3
    data = rng.randint(0, 200, size=(h, w, b)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    p = str(tmp_path / f"tmp_1736_multi_{uuid.uuid4().hex[:8]}.tif")
    to_geotiff(da, p, tiled=True, tile_size=16, compression='none')

    out = open_geotiff(p)
    np.testing.assert_array_equal(out.data, data)


def _peak_ratio_for_write_tiled(data: np.ndarray, tile_size: int) -> float:
    """Return ``tracemalloc`` peak / ``data.nbytes`` for one
    ``_write_tiled`` call against the uncompressed branch.

    Allocations made before this call are excluded from peak by the
    ``reset_peak`` step, so the ratio reflects what ``_write_tiled``
    itself adds.
    """
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        _write_tiled(data, COMPRESSION_NONE, 1, tile_size=tile_size)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / data.nbytes


def test_uncompressed_tiled_peak_memory_single_band():
    """Peak memory for the uncompressed branch should stay below
    ``_PEAK_RATIO_LIMIT * raster_bytes``.  Reintroducing the dead
    ``bytearray(n_tiles * tile_bytes)`` would push the ratio to ~2x
    and fail this check."""
    h, w = 1024, 1024  # 1 MB raw, exact tile divisor -> no edge padding
    data = np.random.RandomState(20260512).randint(
        0, 255, size=(h, w), dtype=np.uint8,
    )
    ratio = _peak_ratio_for_write_tiled(data, tile_size=256)
    assert ratio < _PEAK_RATIO_LIMIT, (
        f"_write_tiled peak memory {ratio:.2f}x raster exceeds the "
        f"{_PEAK_RATIO_LIMIT}x cap; the dead bytearray from #1736 may "
        f"have been reintroduced."
    )


def test_uncompressed_tiled_peak_memory_multiband():
    """3-band variant of the peak-memory check. ``samples == 3``
    triples the would-be dead buffer, so this case is even more
    sensitive to a regression."""
    h, w = 1024, 1024
    data = np.random.RandomState(20260513).randint(
        0, 255, size=(h, w, 3), dtype=np.uint8,
    )
    ratio = _peak_ratio_for_write_tiled(data, tile_size=256)
    assert ratio < _PEAK_RATIO_LIMIT, (
        f"_write_tiled peak memory {ratio:.2f}x raster exceeds the "
        f"{_PEAK_RATIO_LIMIT}x cap; the dead bytearray from #1736 may "
        f"have been reintroduced."
    )
