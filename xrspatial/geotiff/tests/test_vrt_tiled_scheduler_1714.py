"""Verify ``_write_vrt_tiled`` runs tile writes on dask's threaded scheduler.

Issue #1714: the prior code called ``dask.compute(*delayed_tasks,
scheduler='synchronous')`` which serialised independent tile writes on
the calling thread. Switching to the threaded scheduler reduces wall
time by ~33% on a 256-tile zstd write. These tests pin the new
scheduler choice and confirm the output is correct (no concurrent-write
races, all tiles present, content matches).
"""
from __future__ import annotations

import glob
import os
import tempfile
from unittest.mock import patch

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff


def _make_dask_da(h: int = 32, w: int = 32, chunk: int = 8) -> xr.DataArray:
    """Return a dask-backed 2D DataArray with ``chunk``-sized chunks.

    Using ``da.from_array`` on a pre-built numpy array gives clean
    ``(chunk, chunk)`` chunking. ``da.arange(...).reshape(...)`` keeps a
    chunk size of 1 along the new axis, which produces a confusing test
    setup.
    """
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return xr.DataArray(
        da.from_array(arr, chunks=(chunk, chunk)),
        dims=["y", "x"],
    )


def test_vrt_tiled_uses_threaded_scheduler():
    """_write_vrt_tiled passes ``scheduler='threads'`` to dask.compute."""
    da_arr = _make_dask_da()
    with tempfile.TemporaryDirectory(prefix="vrt_sched_1714_") as td:
        vrt = os.path.join(td, "sched_check.vrt")

        # Wrap dask.compute so we can record the scheduler kwarg the
        # writer chose. The dask module is imported inside
        # _write_vrt_tiled, so we patch on the module object directly.
        captured = {}
        real_compute = dask.compute

        def spy(*args, **kwargs):
            captured["scheduler"] = kwargs.get("scheduler")
            return real_compute(*args, **kwargs)

        with patch.object(dask, "compute", side_effect=spy) as p:
            to_geotiff(da_arr, vrt)
            assert p.called, "_write_vrt_tiled never invoked dask.compute"

        assert captured.get("scheduler") == "threads", (
            "Expected scheduler='threads' on the VRT-tiled write but "
            f"got {captured.get('scheduler')!r}"
        )


def test_vrt_tiled_threaded_write_produces_all_tiles():
    """All expected tile files exist after the threaded write."""
    da_arr = _make_dask_da(h=32, w=32, chunk=8)  # 4x4 = 16 tiles
    with tempfile.TemporaryDirectory(prefix="vrt_sched_1714_") as td:
        vrt = os.path.join(td, "tile_count.vrt")
        to_geotiff(da_arr, vrt)
        tiles_dir = os.path.join(td, "tile_count_tiles")
        tiles = sorted(glob.glob(os.path.join(tiles_dir, "*.tif")))
        assert len(tiles) == 16, (
            f"Expected 16 tile files, got {len(tiles)} in {tiles_dir}"
        )


def test_vrt_tiled_threaded_write_is_deterministic():
    """Threaded scheduler must not introduce write ordering races.

    Each delayed task writes to its own file path, so the threaded
    scheduler is safe. Run the same write twice and compare byte
    contents of every tile to catch any accidental race regression.
    """
    da_arr = _make_dask_da(h=32, w=32, chunk=8)

    def _write_and_collect(vrt_path: str) -> dict[str, bytes]:
        to_geotiff(da_arr, vrt_path)
        stem = os.path.splitext(os.path.basename(vrt_path))[0]
        tiles_dir = os.path.join(os.path.dirname(vrt_path), stem + "_tiles")
        return {
            os.path.basename(p): open(p, "rb").read()
            for p in sorted(glob.glob(os.path.join(tiles_dir, "*.tif")))
        }

    with tempfile.TemporaryDirectory(prefix="vrt_sched_1714_") as td1:
        with tempfile.TemporaryDirectory(prefix="vrt_sched_1714_") as td2:
            tiles1 = _write_and_collect(os.path.join(td1, "run1.vrt"))
            tiles2 = _write_and_collect(os.path.join(td2, "run2.vrt"))

    assert set(tiles1) == set(tiles2), (
        "Tile file set differs between runs: "
        f"{set(tiles1) ^ set(tiles2)}"
    )
    for name, blob1 in tiles1.items():
        assert blob1 == tiles2[name], (
            f"Tile {name} differs between runs (race condition?)"
        )
