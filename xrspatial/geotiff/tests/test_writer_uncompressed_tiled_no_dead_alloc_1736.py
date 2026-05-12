"""Regression test for issue #1736.

The uncompressed tiled-write branch of ``_compress_tiles`` previously
allocated a contiguous ``bytearray`` plus a memoryview ``(n_tiles *
tw * th * bytes_per_sample * samples)`` bytes long at the top of the
loop and never read either back. Tile bytes were still built via
``tile_arr.tobytes()`` and appended to a list. The dead buffer roughly
doubled peak memory for an uncompressed write.

The fix is a pure deletion. This test pins the round-trip so a future
refactor that re-introduces a real contiguous buffer keeps the same
external behaviour: writing an uncompressed tiled GeoTIFF must still
read back identically with no holes between tiles.
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import xarray as xr

from xrspatial.geotiff import to_geotiff, open_geotiff


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
