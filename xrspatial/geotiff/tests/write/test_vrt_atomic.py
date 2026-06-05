"""Transactional VRT tiled writes (issue #2669).

A tiled ``.vrt`` write must be atomic: a successful write leaves a
complete tile directory plus a readable VRT, and a write that fails
partway through leaves no poisoned partial output so a retry can
succeed.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._writers import eager


def _make_da(n=32):
    """A georeferenced DataArray that tiles into a 2x2 grid at
    ``tile_size=16`` (TIFF 6 requires tile sizes to be multiples of 16)."""
    arr = np.arange(n * n, dtype=np.float32).reshape(n, n)
    return xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.linspace(n - 0.5, 0.5, n),
                'x': np.linspace(0.5, n - 0.5, n)},
        attrs={'crs': 4326},
    )


def _tiles_dir_for(vrt_path):
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    stem = os.path.splitext(os.path.basename(vrt_path))[0]
    return os.path.join(vrt_dir, stem + '_tiles')


def test_successful_tiled_write_produces_readable_vrt(tmp_path):
    """A clean tiled write yields the tile directory and a VRT that reads
    back to the original values."""
    da = _make_da()
    vrt_path = str(tmp_path / 'atomic_2669_ok.vrt')

    to_geotiff(da, vrt_path, tiled=True, tile_size=16, compression='none')

    tiles_dir = _tiles_dir_for(vrt_path)
    assert os.path.isfile(vrt_path)
    assert os.path.isdir(tiles_dir)
    tiles = [f for f in os.listdir(tiles_dir) if f.endswith('.tif')]
    assert len(tiles) == 4  # 2x2 grid

    # No staging dir left behind.
    leftovers = [f for f in os.listdir(tmp_path)
                 if '.tmp-' in f]
    assert leftovers == []

    result = open_geotiff(vrt_path)
    np.testing.assert_array_almost_equal(
        np.asarray(result.data).squeeze(), da.values, decimal=5)


def test_failed_tiled_write_leaves_no_poisoned_output_and_retry_succeeds(
        tmp_path, monkeypatch):
    """A tile write that raises partway through must leave neither the
    final tile dir nor a staging dir behind, and a subsequent retry must
    succeed."""
    da = _make_da()
    vrt_path = str(tmp_path / 'atomic_2669_fail.vrt')
    tiles_dir = _tiles_dir_for(vrt_path)

    real_write_single_tile = eager._write_single_tile
    calls = {'n': 0}

    def flaky_write_single_tile(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError("simulated tile write failure")
        return real_write_single_tile(*args, **kwargs)

    monkeypatch.setattr(eager, '_write_single_tile',
                        flaky_write_single_tile)

    with pytest.raises(RuntimeError, match="simulated tile write failure"):
        to_geotiff(da, vrt_path, tiled=True, tile_size=16,
                   compression='none')

    # No final tile dir, no VRT, no staging dir.
    assert not os.path.exists(tiles_dir)
    assert not os.path.exists(vrt_path)
    leftovers = [f for f in os.listdir(tmp_path) if '.tmp-' in f]
    assert leftovers == [], f"poisoned staging dirs left: {leftovers}"

    # Retry with a healthy writer must succeed -- the leftover-state guard
    # no longer blocks it because nothing partial was committed.
    monkeypatch.setattr(eager, '_write_single_tile', real_write_single_tile)
    to_geotiff(da, vrt_path, tiled=True, tile_size=16, compression='none')

    assert os.path.isfile(vrt_path)
    assert os.path.isdir(tiles_dir)
    result = open_geotiff(vrt_path)
    np.testing.assert_array_almost_equal(
        np.asarray(result.data).squeeze(), da.values, decimal=5)


def test_early_validation_failure_leaves_no_staging_dir(tmp_path):
    """A write that fails validation *after* the staging dir is created
    (here: the 2D-only check rejecting a 3D array) must still remove the
    staging dir, so no ``*.tmp-*`` leftover blocks a later retry (#2964).

    This failure happens before any tile is written, so it exercises the
    early-failure path the per-tile tests do not.
    """
    arr = np.arange(2 * 16 * 16, dtype=np.float32).reshape(2, 16, 16)
    da = xr.DataArray(
        arr,
        dims=['band', 'y', 'x'],
        coords={'y': np.linspace(15.5, 0.5, 16),
                'x': np.linspace(0.5, 15.5, 16)},
        attrs={'crs': 4326},
    )
    vrt_path = str(tmp_path / 'atomic_2964_3d.vrt')

    with pytest.raises(ValueError, match="2D arrays only"):
        to_geotiff(da, vrt_path, tiled=True, tile_size=16,
                   compression='none')

    leftovers = [f for f in os.listdir(tmp_path) if '.tmp-' in f]
    assert leftovers == [], f"staging dir leaked on validation failure: {leftovers}"
    assert not os.path.exists(_tiles_dir_for(vrt_path))
    assert not os.path.exists(vrt_path)


def test_preexisting_empty_tiles_dir_is_reused(tmp_path):
    """An empty ``*_tiles`` directory left over from a prior aborted run
    passes the non-empty leftover-state guard, so the write must promote
    the staging dir into that slot rather than raising."""
    da = _make_da()
    vrt_path = str(tmp_path / 'atomic_2669_empty.vrt')
    tiles_dir = _tiles_dir_for(vrt_path)
    os.makedirs(tiles_dir)  # empty, pre-existing

    to_geotiff(da, vrt_path, tiled=True, tile_size=16, compression='none')

    assert os.path.isfile(vrt_path)
    assert os.path.isdir(tiles_dir)
    tiles = [f for f in os.listdir(tiles_dir) if f.endswith('.tif')]
    assert len(tiles) == 4
    result = open_geotiff(vrt_path)
    np.testing.assert_array_almost_equal(
        np.asarray(result.data).squeeze(), da.values, decimal=5)


def test_dask_failed_tiled_write_cleans_up(tmp_path, monkeypatch):
    """Same atomicity guarantee on the dask streaming path."""
    pytest.importorskip("dask.array")
    da = _make_da().chunk({'y': 16, 'x': 16})
    vrt_path = str(tmp_path / 'atomic_2669_dask.vrt')
    tiles_dir = _tiles_dir_for(vrt_path)

    real_write_single_tile = eager._write_single_tile
    calls = {'n': 0}

    def flaky_write_single_tile(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError("simulated dask tile failure")
        return real_write_single_tile(*args, **kwargs)

    monkeypatch.setattr(eager, '_write_single_tile',
                        flaky_write_single_tile)

    with pytest.raises(RuntimeError):
        to_geotiff(da, vrt_path, tiled=True, compression='none')

    assert not os.path.exists(tiles_dir)
    assert not os.path.exists(vrt_path)
    leftovers = [f for f in os.listdir(tmp_path) if '.tmp-' in f]
    assert leftovers == []

    # Retry succeeds.
    monkeypatch.setattr(eager, '_write_single_tile', real_write_single_tile)
    to_geotiff(da, vrt_path, tiled=True, compression='none')
    assert os.path.isfile(vrt_path)
    assert os.path.isdir(tiles_dir)
