"""Parallel per-tile compression in the streaming write path (P4).

The streaming tile-write path used to compress tiles serially inside each
horizontal segment. The non-streaming path already fans compress out to a
``ThreadPoolExecutor`` sized at ``os.cpu_count()``. These tests confirm
the streaming path now follows the same pattern: round-trip pixels are
unchanged, more than one worker thread participates, and a moderate
deflate write completes within a generous wall-time budget.
"""
from __future__ import annotations

import inspect
import os
import threading
import time

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _writer as writer_mod
from xrspatial.geotiff import open_geotiff, to_geotiff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataarray(shape, dtype=np.float32, seed=20260508):
    rng = np.random.default_rng(seed)
    if np.issubdtype(np.dtype(dtype), np.floating):
        arr = rng.random(shape, dtype=dtype)
        nodata = -9999.0
    else:
        info = np.iinfo(dtype)
        arr = rng.integers(info.min // 2 if info.min < 0 else 1,
                           info.max // 2,
                           size=shape, dtype=dtype)
        # Pick a sentinel that is in-range for the integer dtype.
        nodata = 0 if info.min >= 0 else -1
    if len(shape) == 2:
        h, w = shape
    else:
        h, w = shape[:2]
    y = np.linspace(41.0, 40.0, h)
    x = np.linspace(-106.0, -105.0, w)
    return xr.DataArray(arr, dims=['y', 'x'][:len(shape)],
                        coords={'y': y, 'x': x},
                        attrs={'crs': 4326, 'nodata': nodata})


# ---------------------------------------------------------------------------
# Round-trip correctness across dtype/compression/tile-size combos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'dtype,compression,tile_size',
    [
        (np.float32, 'deflate', 256),
        (np.float32, 'zstd', 128),
        (np.float32, 'lzw', 256),
        (np.uint16, 'deflate', 256),
        (np.uint8, 'deflate', 128),
        (np.float32, 'none', 256),
    ],
)
def test_streaming_write_round_trip_unchanged(
        dtype, compression, tile_size, tmp_path):
    """Streaming write must be bit-exact vs. the eager write (and vs. input).

    Picks shapes that force multiple horizontal segments by setting a
    small ``streaming_buffer_bytes`` so the parallel-compress branch
    actually fires.
    """
    shape = (600, 700)
    da = _make_dataarray(shape, dtype=dtype)
    # small chunks so dask path is taken; small buffer so segments split
    dask_da = da.chunk({'y': 200, 'x': 200})

    eager_path = str(tmp_path / 'eager.tif')
    stream_path = str(tmp_path / 'stream.tif')

    to_geotiff(da, eager_path,
               compression=compression, tile_size=tile_size)

    # Force multi-segment by limiting buffer to a couple of tile columns,
    # if the underlying write_streaming exposes the kwarg.
    sig = inspect.signature(writer_mod.write_streaming)
    extra = {}
    if 'streaming_buffer_bytes' in sig.parameters:
        # ~3 tile columns at 256-tile widths of float32 -> few hundred KB.
        extra['streaming_buffer_bytes'] = 256 * 1024
    to_geotiff(dask_da, stream_path,
               compression=compression, tile_size=tile_size, **extra)

    eager = open_geotiff(eager_path)
    stream = open_geotiff(stream_path)

    # Bit-exact pixel match: streaming vs eager
    np.testing.assert_array_equal(eager.values, stream.values)
    # And streaming vs the source array (lossless codecs only)
    np.testing.assert_array_equal(stream.values, da.values)


# ---------------------------------------------------------------------------
# Parallelism is observable: multiple thread IDs participate
# ---------------------------------------------------------------------------

def test_streaming_write_parallelism_observed(monkeypatch, tmp_path):
    """Confirm more than one worker thread runs ``_compress_block``.

    Strategy: wrap the real function so each call records the current
    thread ID, then write a raster sized to produce many tiles inside a
    single segment. After the write, assert at least 2 unique thread IDs
    were observed.

    Force ``os.cpu_count()`` to 4 in the global ``os`` module for the
    duration of the test so the assertion stays deterministic on
    single-core CI containers (where the real cpu_count would size the
    pool to 1 and the test would fail for environment reasons). The
    writer imports ``os`` lazily inside the function, so patching the
    global module is sufficient.
    """
    monkeypatch.setattr(os, 'cpu_count', lambda: 4)

    real_compress = writer_mod._compress_block
    seen_threads: set[int] = set()
    lock = threading.Lock()

    def recording_compress(*args, **kwargs):
        # Hold each call long enough that the pool cannot serialise it
        # away on a single core.
        tid = threading.get_ident()
        with lock:
            seen_threads.add(tid)
        # Tiny synthetic delay so concurrent submissions overlap.
        time.sleep(0.005)
        return real_compress(*args, **kwargs)

    monkeypatch.setattr(writer_mod, '_compress_block', recording_compress)

    # Many tiles in a single segment: 16 tile columns x 8 tile rows = 128
    # tiles, well above any cpu count.
    shape = (16 * 256, 16 * 256)
    da = _make_dataarray(shape, dtype=np.float32, seed=42)
    dask_da = da.chunk({'y': 256 * 4, 'x': 256 * 4})

    path = str(tmp_path / 'parallel_check.tif')
    to_geotiff(dask_da, path,
               compression='deflate', tile_size=256)

    assert len(seen_threads) > 1, (
        f"Expected >1 worker threads in streaming compress, "
        f"saw {len(seen_threads)}: {seen_threads}")


# ---------------------------------------------------------------------------
# Optional wall-clock regression guard (env-gated)
# ---------------------------------------------------------------------------

def test_streaming_write_perf_sanity(tmp_path):
    """Opt-in tripwire: a 4096x4096 deflate streaming write under a
    generous wall-clock threshold. Skipped by default because shared CI
    runners, CPU throttling, debug builds, and slow filesystems all
    make absolute timings flaky. Set ``XRSPATIAL_RUN_PERF_TESTS=1`` to
    enable. The deterministic regression coverage lives in
    :func:`test_streaming_write_parallelism_observed` -- this is just a
    sanity bound on top.
    """
    if os.environ.get('XRSPATIAL_RUN_PERF_TESTS') != '1':
        pytest.skip(
            "set XRSPATIAL_RUN_PERF_TESTS=1 to run wall-clock perf tests")

    shape = (4096, 4096)
    da = _make_dataarray(shape, dtype=np.float32, seed=2026)
    dask_da = da.chunk({'y': 1024, 'x': 1024})

    path = str(tmp_path / 'perf_4k.tif')

    t0 = time.perf_counter()
    to_geotiff(dask_da, path, compression='deflate', tile_size=256)
    elapsed = time.perf_counter() - t0

    result = open_geotiff(path)
    assert result.shape == shape

    assert elapsed < 5.0, (
        f"Streaming 4096x4096 deflate write took {elapsed:.2f}s, "
        f"expected <5s (regression guard)")
