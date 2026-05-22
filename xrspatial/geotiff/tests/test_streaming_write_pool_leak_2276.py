"""ThreadPoolExecutor leak on mid-stream failure in tiled writes (#2276).

The streaming tiled-write path in ``_write_streaming`` builds a
``ThreadPoolExecutor`` for parallel per-tile compression. Before this
fix, ``shutdown`` only ran after the tile-row loop completed -- if any
mid-stream step raised (compression error, dask compute error, file
write error), the shutdown was skipped and worker threads survived the
failure path.

These tests inject a failure mid-stream and assert that:

1. No worker threads owned by the writer's pool remain alive after the
   call returns, and
2. The injected exception propagates cleanly out of ``to_geotiff`` (no
   "swallowed" failures), and
3. The pool the writer constructed is shut down (``_shutdown`` flag
   set).

We monkey-patch ``ThreadPoolExecutor`` inside ``_writer`` to capture
the pool ``_write_streaming`` constructs and inspect its state after
the failure path. The test also walks ``threading.enumerate()`` to
check that no threads with the writer's distinctive
``thread_name_prefix`` (``_TILE_POOL_THREAD_PREFIX`` from the writer
module) remain. Dask spins up its own ``ThreadPoolExecutor`` instances
during ``.compute()`` -- those use a different prefix and are
deliberately kept alive as singletons, so filtering on the writer's
prefix avoids false positives.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff import _writer as writer_mod


# Re-use the writer's own constant so the test does not silently drift
# if the prefix ever changes on the writer side. ``_writer`` exposes
# ``_TILE_POOL_THREAD_PREFIX`` for exactly this purpose (#2276).
_WRITER_POOL_PREFIX = writer_mod._TILE_POOL_THREAD_PREFIX


def _make_dataarray(shape, dtype=np.float32, seed=20260521):
    rng = np.random.default_rng(seed)
    arr = rng.random(shape, dtype=dtype)
    h, w = shape
    y = np.linspace(41.0, 40.0, h)
    x = np.linspace(-106.0, -105.0, w)
    return xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x},
        attrs={'crs': 4326, 'nodata': -9999.0})


def _list_writer_pool_worker_threads():
    """Return live threads owned by the writer's tile-compress pool.

    Dask spins up its own ``ThreadPoolExecutor`` instances (the
    ``Dask-Offload`` singleton and the threaded scheduler) that survive
    deliberately, so filtering on the writer's distinctive prefix
    avoids false positives.
    """
    return [t for t in threading.enumerate()
            if t.name.startswith(_WRITER_POOL_PREFIX) and t.is_alive()]


@pytest.fixture
def captured_pools(monkeypatch):
    """Capture only the ``ThreadPoolExecutor`` instances ``_writer``
    constructs (filtered by ``thread_name_prefix``).

    ``_write_streaming`` does ``from concurrent.futures import
    ThreadPoolExecutor`` inside the function body, so we patch the
    symbol on the ``concurrent.futures`` module the import resolves
    through. Dask also constructs its own ``ThreadPoolExecutor``
    instances during ``.compute()``; those use a different
    ``thread_name_prefix`` and are filtered out here.
    """
    pools = []
    real_cls = ThreadPoolExecutor

    class _RecordingPool(real_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            prefix = kwargs.get('thread_name_prefix', '')
            if prefix.startswith(_WRITER_POOL_PREFIX):
                pools.append(self)

    import concurrent.futures as _cf
    monkeypatch.setattr(_cf, 'ThreadPoolExecutor', _RecordingPool)
    return pools


def test_pool_shutdown_on_compress_failure(
        captured_pools, monkeypatch, tmp_path):
    """A raise inside ``_compress_block`` must shut the pool down."""
    # Force cpu_count high enough that ``_use_pool`` triggers.
    monkeypatch.setattr(os, 'cpu_count', lambda: 4)

    pre_existing_pool_threads = _list_writer_pool_worker_threads()
    pre_existing_names = {t.name for t in pre_existing_pool_threads}

    real_compress = writer_mod._compress_block
    call_count = {'n': 0}
    lock = threading.Lock()

    class _InjectedError(RuntimeError):
        pass

    def failing_compress(*args, **kwargs):
        # Let the first few calls succeed so the executor genuinely
        # spins up worker threads, then raise from inside a worker.
        with lock:
            call_count['n'] += 1
            n = call_count['n']
        if n >= 3:
            raise _InjectedError(
                f"injected failure on _compress_block call #{n}")
        return real_compress(*args, **kwargs)

    monkeypatch.setattr(writer_mod, '_compress_block', failing_compress)

    # Sized to produce many tiles per segment so the parallel branch
    # fires (pool only used when ``n_seg_tiles > 1``).
    shape = (8 * 256, 8 * 256)
    da = _make_dataarray(shape)
    dask_da = da.chunk({'y': 256 * 2, 'x': 256 * 2})

    out_path = str(tmp_path / 'tmp_2276_compress_fail.tif')

    with pytest.raises(_InjectedError):
        to_geotiff(dask_da, out_path,
                   compression='deflate', tile_size=256)

    # At least one pool should have been constructed by _write_streaming.
    assert len(captured_pools) >= 1, (
        "Expected _write_streaming to construct a ThreadPoolExecutor; "
        "none were captured.")

    # Every captured pool must be shut down by the time we get here.
    for idx, pool in enumerate(captured_pools):
        assert pool._shutdown, (
            f"Captured pool #{idx} was NOT shut down after the "
            f"mid-stream failure -- ThreadPoolExecutor leak.")

    # Give workers a moment to actually exit after shutdown(wait=True);
    # _shutdown=True plus shutdown(wait=True) should already mean
    # threads have joined, but defensive sleep avoids a race on slow
    # CI runners.
    deadline = time.monotonic() + 2.0
    leaked = []
    while time.monotonic() < deadline:
        current = _list_writer_pool_worker_threads()
        leaked = [t for t in current if t.name not in pre_existing_names]
        if not leaked:
            break
        time.sleep(0.05)

    assert not leaked, (
        f"ThreadPoolExecutor worker threads still alive after failed "
        f"streaming write: {[t.name for t in leaked]}")


def test_pool_shutdown_on_file_write_failure(
        captured_pools, monkeypatch, tmp_path):
    """A raise from the sequential file-write step (after the parallel
    compress has already run for a segment) must still shut the pool
    down. This covers the second class of mid-stream failure: the
    pool's work finished cleanly but the consumer of those compressed
    buffers failed before the loop reached the bottom of the function.
    """
    monkeypatch.setattr(os, 'cpu_count', lambda: 4)

    pre_existing = {t.name for t in _list_writer_pool_worker_threads()}

    class _InjectedWriteError(IOError):
        pass

    # Wrap ``os.fdopen`` so the file object's ``write`` raises after a
    # configurable number of calls. The streaming writer opens the
    # output file via ``os.fdopen(fd, 'wb')`` once and then calls
    # ``f.write(...)`` for each header, IFD chunk, and compressed
    # tile. Letting a generous number of early writes through gets us
    # past the header/IFD and into the per-tile write loop where the
    # pool is actively in use.
    real_fdopen = os.fdopen
    write_count = {'n': 0}

    def wrapping_fdopen(fd, mode='r', *args, **kwargs):
        f = real_fdopen(fd, mode, *args, **kwargs)
        real_write = f.write

        def counting_write(data):
            write_count['n'] += 1
            if write_count['n'] >= 12:
                raise _InjectedWriteError(
                    f"injected file-write failure on call "
                    f"#{write_count['n']}")
            return real_write(data)

        f.write = counting_write
        return f

    monkeypatch.setattr(os, 'fdopen', wrapping_fdopen)

    shape = (8 * 256, 8 * 256)
    da = _make_dataarray(shape)
    dask_da = da.chunk({'y': 512, 'x': 512})

    out_path = str(tmp_path / 'tmp_2276_write_fail.tif')

    with pytest.raises(_InjectedWriteError):
        to_geotiff(dask_da, out_path,
                   compression='deflate', tile_size=256)

    assert len(captured_pools) >= 1, (
        "Expected _write_streaming to construct a writer pool before "
        "the file-write failure.")
    for idx, pool in enumerate(captured_pools):
        assert pool._shutdown, (
            f"Captured pool #{idx} not shut down after file-write "
            f"failure -- ThreadPoolExecutor leak.")

    deadline = time.monotonic() + 2.0
    leaked = []
    while time.monotonic() < deadline:
        leaked = [t for t in _list_writer_pool_worker_threads()
                  if t.name not in pre_existing]
        if not leaked:
            break
        time.sleep(0.05)
    assert not leaked, (
        f"Pool worker threads leaked after file-write failure: "
        f"{[t.name for t in leaked]}")


def test_pool_shutdown_on_happy_path(
        captured_pools, monkeypatch, tmp_path):
    """Regression guard: the success path must still shut the pool
    down -- the ``try/finally`` rewrite must not regress the original
    behaviour."""
    monkeypatch.setattr(os, 'cpu_count', lambda: 4)

    shape = (4 * 256, 4 * 256)
    da = _make_dataarray(shape)
    dask_da = da.chunk({'y': 512, 'x': 512})

    out_path = str(tmp_path / 'tmp_2276_happy.tif')
    to_geotiff(dask_da, out_path,
               compression='deflate', tile_size=256)

    assert len(captured_pools) >= 1
    for idx, pool in enumerate(captured_pools):
        assert pool._shutdown, (
            f"Captured pool #{idx} not shut down on the success path.")
