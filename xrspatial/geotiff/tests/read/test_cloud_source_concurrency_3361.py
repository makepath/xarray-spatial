"""Regression tests for the _CloudSource shared-handle data race (#3361).

``_CloudSource.read_range`` used to do ``fs.open(path, 'rb')`` followed by
``seek`` + ``read``. For backends whose ``open`` returns a shared,
process-global file object -- notably fsspec's ``MemoryFileSystem``, where
every open of a path hands back the same stored buffer -- concurrent
windowed reads raced on that single cursor. Under the free-threaded
interpreter (3.14t) this corrupted tile bytes (``zlib.error: incorrect
header check``) or read from a handle another thread had just closed
(``ValueError: I/O operation on closed file``). The GIL masked the race,
so the integration repro (``chunks=`` open of a ``memory://`` COG) only
fails on the free-threaded lane.

The fix routes reads through the stateless ``cat_file`` ranged API. These
tests pin that contract deterministically (without depending on the
free-threaded interpreter) by driving ``read_range`` against a filesystem
whose ``open`` returns a shared handle and whose ``seek`` is barrier-
synchronised, so a shared-cursor implementation provably returns the wrong
bytes while the ``cat_file`` implementation does not.
"""
import threading

import pytest

fsspec = pytest.importorskip("fsspec")

from xrspatial.geotiff._sources import _CloudSource  # noqa: E402


class _SharedHandle:
    """A single buffer with one shared cursor, like an fsspec MemoryFile."""

    def __init__(self, buf, seek_barrier=None):
        self._buf = buf
        self.pos = 0
        self._seek_barrier = seek_barrier

    def seek(self, pos, whence=0):
        assert whence == 0
        self.pos = pos
        # Force every concurrent reader to finish seeking before any of
        # them reads. On a shared cursor this guarantees at least one read
        # observes another thread's offset -- the exact race the fix
        # removes -- making the test deterministic under the GIL.
        if self._seek_barrier is not None:
            self._seek_barrier.wait()

    def read(self, n=-1):
        return self._buf[self.pos:self.pos + n] if n >= 0 else self._buf[self.pos:]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _SharedHandleFS:
    """Filesystem stub: ``open`` returns one shared handle; ``cat_file``
    is the stateless ranged read that the fix relies on."""

    def __init__(self, buf, seek_barrier=None):
        self._data = buf
        self._handle = _SharedHandle(buf, seek_barrier=seek_barrier)

    def open(self, path, mode='rb'):
        return self._handle

    def cat_file(self, path, start=None, end=None):
        return self._data[start:end]

    def size(self, path):
        return len(self._data)


def _cloud_source_over(fs):
    src = _CloudSource.__new__(_CloudSource)
    src._url = "memory://stub/x.bin"
    src._fs = fs
    src._path = "stub/x.bin"
    src._size = len(fs._data)
    return src


def test_cloud_source_concurrent_read_range_no_shared_cursor_3361():
    """Concurrent ``read_range`` calls must each return their own bytes
    even when the backend's ``open`` exposes a shared cursor."""
    data = bytes((i % 251) for i in range(1000))
    barrier = threading.Barrier(2)
    src = _cloud_source_over(_SharedHandleFS(data, seek_barrier=barrier))

    results = {}

    def worker(start):
        results[start] = src.read_range(start, 100)

    threads = [threading.Thread(target=worker, args=(s,)) for s in (0, 500)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # A shared-cursor implementation makes both reads land on whichever
    # seek ran last, so at least one of these is wrong. ``cat_file`` keeps
    # them independent.
    assert results[0] == data[0:100]
    assert results[500] == data[500:600]


def test_cloud_source_read_range_matches_read_semantics_3361():
    """``cat_file``-based ``read_range`` keeps ``seek``+``read`` semantics,
    including the EOF clamp the COG header prefetch relies on."""
    data = b"ABCDE"
    src = _cloud_source_over(_SharedHandleFS(data))
    assert src.read_range(0, 5) == b"ABCDE"
    assert src.read_range(0, 100) == b"ABCDE"   # length past EOF clamps
    assert src.read_range(3, 100) == b"DE"
    assert src.read_range(2, 0) == b""
    assert src.read_range(5, 10) == b""         # start at EOF


def test_cloud_source_read_range_real_memory_fs_3361():
    """End-to-end against a real fsspec ``memory://`` source: every range
    reads correctly through the live ``MemoryFileSystem`` (whose ``open``
    really does return a shared handle)."""
    import os
    import uuid

    fs = fsspec.filesystem("memory")
    data = bytes((i % 256) for i in range(4096))
    key = f"test3361-{os.getpid()}-{uuid.uuid4().hex[:8]}/x.bin"
    fs.pipe_file(key, data)
    try:
        src = _CloudSource("memory://" + key)
        for start, length in [(0, 16), (100, 200), (4000, 500), (4096, 8)]:
            assert src.read_range(start, length) == data[start:start + length]
        assert src.read_all() == data
    finally:
        fs.rm_file(key)
