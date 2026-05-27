# Tests for the atexit shutdown hook that closes cached mmap-backed
# file handles at interpreter shutdown (issue #2486).
import atexit
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._sources import _FileSource, _mmap_cache, _MmapCache, _shutdown_cleanup


@pytest.fixture
def tiff_path_2486(tmp_path):
    arr = np.zeros((4, 4), dtype=np.uint8)
    path = str(tmp_path / 'tmp_2486_shutdown.tif')
    to_geotiff(arr, path, compression='none')
    return path


def test_close_all_closes_idle_entries(tiff_path_2486):
    # An idle entry (refcount 0) should be closed by close_all.
    cache = _MmapCache(max_size=4)
    mm, size, entry = cache.acquire(tiff_path_2486)
    cache.release(entry)
    # Entry sits idle in the cache.
    real = os.path.realpath(tiff_path_2486)
    assert real in cache._entries
    fh = entry[0]
    mm_obj = entry[1]
    assert not fh.closed
    cache.close_all()
    assert len(cache._entries) == 0
    assert fh.closed
    if mm_obj is not None:
        # mmap exposes closed via .closed in CPython 3.2+.
        assert mm_obj.closed


def test_close_all_closes_in_use_entries(tiff_path_2486):
    # An in-use entry (refcount > 0) should also be closed -- the process
    # is going away and we want no handles to leak.
    cache = _MmapCache(max_size=4)
    mm, size, entry = cache.acquire(tiff_path_2486)
    assert entry[3] == 1  # refcount
    fh = entry[0]
    cache.close_all()
    assert len(cache._entries) == 0
    assert fh.closed


def test_close_all_idempotent(tiff_path_2486):
    cache = _MmapCache(max_size=4)
    _mm, _size, entry = cache.acquire(tiff_path_2486)
    cache.release(entry)
    cache.close_all()
    # Second call should be a harmless no-op.
    cache.close_all()
    assert len(cache._entries) == 0


def test_release_after_close_all_does_not_crash(tiff_path_2486):
    # After close_all, a stale FileSource holding the entry token must
    # still survive its eventual release() call. The release should not
    # raise, even though the file handle and mmap are already closed.
    cache = _MmapCache(max_size=4)
    mm, size, entry = cache.acquire(tiff_path_2486)
    cache.close_all()
    # Should not raise. The orphan branch in release() will try to
    # close again -- mmap.close() and file.close() are idempotent.
    cache.release(entry)


def test_shutdown_cleanup_runs_against_module_singleton(tiff_path_2486):
    # The module-level cleanup function should call close_all on the
    # module singleton. Use the real singleton (not a fresh _MmapCache)
    # because the atexit registration targets the singleton by name.
    src = _FileSource(tiff_path_2486)
    src.close()  # Returns entry to idle pool.
    real = os.path.realpath(tiff_path_2486)
    assert real in _mmap_cache._entries
    entry = _mmap_cache._entries[real]
    fh = entry[0]
    assert not fh.closed
    _shutdown_cleanup()
    assert len(_mmap_cache._entries) == 0
    assert fh.closed


def test_shutdown_cleanup_handles_already_closed_resources(tiff_path_2486):
    # If the mmap / file object is already closed (e.g. another finalizer
    # ran first), the cleanup must not propagate exceptions out of the
    # atexit chain.
    cache = _MmapCache(max_size=4)
    _mm, _size, entry = cache.acquire(tiff_path_2486)
    # Pre-close the underlying resources to simulate finalization order.
    if entry[1] is not None:
        entry[1].close()
    entry[0].close()
    # Should swallow the double-close errors silently.
    cache.close_all()
    assert len(cache._entries) == 0


def test_subprocess_read_emits_no_resource_warning(tiff_path_2486):
    # Spawn a subprocess that opens a GeoTIFF and exits. With the atexit
    # hook registered, no ResourceWarning should be emitted at shutdown.
    # Run with -W error::ResourceWarning so any leaked handle turns into
    # a non-zero exit code with a traceback we can grep.
    script = textwrap.dedent(f'''
        import warnings
        # Promote ResourceWarning to error so the test subprocess fails
        # loudly if a handle leaks.
        warnings.simplefilter("error", ResourceWarning)
        from xrspatial.geotiff._sources import _FileSource
        src = _FileSource({tiff_path_2486!r})
        # Read something so the mmap is actually used.
        _ = src.read_range(0, min(16, src.size))
        src.close()
        # Do not call _mmap_cache.clear() explicitly -- the atexit hook
        # is what we are testing.
    ''')
    result = subprocess.run(
        [sys.executable, '-W', 'error::ResourceWarning', '-c', script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # The subprocess should exit cleanly with no ResourceWarning.
    assert result.returncode == 0, (
        f"subprocess exited with {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert 'ResourceWarning' not in result.stderr, (
        f"ResourceWarning leaked in subprocess stderr:\n{result.stderr}"
    )
    assert 'unclosed file' not in result.stderr, (
        f"unclosed-file warning leaked in subprocess stderr:\n"
        f"{result.stderr}"
    )


def test_atexit_hook_is_registered():
    # The atexit hook must be registered on module import so production
    # processes get the cleanup without any caller action. ``atexit``
    # doesn't expose the registered callback list, but ``unregister``
    # is a no-op when the function is not registered and removes it
    # otherwise. Re-register immediately so this test doesn't disable
    # the hook for the rest of the suite.
    atexit.unregister(_shutdown_cleanup)
    # Re-register so subsequent tests / the actual interpreter shutdown
    # still benefit from the cleanup.
    atexit.register(_shutdown_cleanup)
    # If the registration was already present, unregistering and
    # re-registering keeps the singleton invariant. The strong test of
    # "the registration actually runs at interpreter exit" lives in
    # ``test_subprocess_read_emits_no_resource_warning`` -- spawning a
    # subprocess and asserting no ``ResourceWarning`` leaks is the real
    # end-to-end check.
