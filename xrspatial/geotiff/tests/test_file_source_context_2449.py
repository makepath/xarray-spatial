# Tests for _FileSource context manager protocol (issue #2449).
import os
import struct

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._sources import _FileSource, _mmap_cache


@pytest.fixture
def tiff_path(tmp_path):
    arr = np.zeros((4, 4), dtype=np.uint8)
    path = str(tmp_path / 'fs_ctx_2449.tif')
    to_geotiff(arr, path, compression='none')
    return path


def _refcount(path):
    """Look up the cache refcount for *path*, or None if not cached."""
    real = os.path.realpath(path)
    entry = _mmap_cache._entries.get(real)
    return None if entry is None else entry[3]


def test_enter_returns_self(tiff_path):
    src = _FileSource(tiff_path)
    try:
        assert src.__enter__() is src
    finally:
        src.close()


def test_exit_releases_entry(tiff_path):
    _mmap_cache.clear()
    with _FileSource(tiff_path) as src:
        assert _refcount(tiff_path) == 1
        # mmap is usable inside the block
        assert len(src.read_all()) == src.size
    # After the with block, refcount returns to 0 (entry stays cached).
    assert _refcount(tiff_path) == 0


def test_exit_releases_on_exception(tiff_path):
    _mmap_cache.clear()
    with pytest.raises(struct.error):
        with _FileSource(tiff_path):
            assert _refcount(tiff_path) == 1
            struct.unpack('>I', b'')
    assert _refcount(tiff_path) == 0


def test_double_close_safe(tiff_path):
    with _FileSource(tiff_path) as src:
        src.close()
        # __exit__ will call close() again; must not raise or
        # over-decrement the cache refcount.
        assert _refcount(tiff_path) == 0
    assert _refcount(tiff_path) == 0


def test_nested_with_shares_cache_entry(tiff_path):
    _mmap_cache.clear()
    with _FileSource(tiff_path):
        assert _refcount(tiff_path) == 1
        with _FileSource(tiff_path):
            assert _refcount(tiff_path) == 2
        assert _refcount(tiff_path) == 1
    assert _refcount(tiff_path) == 0
