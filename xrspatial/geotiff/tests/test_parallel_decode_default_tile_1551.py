"""Regression test for issue #1551.

The parallel tile decode gate in ``_read_tiles`` previously used a strict
``tile_pixels > 64 * 1024`` comparison, which excluded the default
``tile_size=256`` (256 * 256 == 64 * 1024). Tiles at the default size are
the most common in real GeoTIFFs, so the parallel path almost never ran.
The fix is one character: ``>`` becomes ``>=``.

These tests assert the dispatch by monkey-patching
``concurrent.futures.ThreadPoolExecutor`` and confirming it gets
constructed at the default tile_size.
"""
from __future__ import annotations

import concurrent.futures
import threading

import numpy as np

from xrspatial.geotiff._dtypes import tiff_dtype_to_numpy
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import _read_tiles

from .conftest import make_minimal_tiff


class _PoolSpy:
    """Drop-in replacement for ThreadPoolExecutor that records calls.

    Wraps the real executor so tests still get correct decoded output;
    only the construction is observed. Each instance writes its
    construction (and the test's thread id) into the supplied list so
    pytest-xdist runs do not cross-contaminate.
    """

    def __init__(self, record, real_cls):
        self._record = record
        self._real_cls = real_cls
        self._real = None

    def __call__(self, *args, **kwargs):
        self._record.append({
            'args': args,
            'kwargs': dict(kwargs),
            'thread': threading.get_ident(),
        })
        self._real = self._real_cls(*args, **kwargs)
        return self._real


def _build_tiled_tiff(tile_size: int, tiles_across: int, tiles_down: int) -> bytes:
    """Build a tiled TIFF with deterministic pixel content."""
    width = tile_size * tiles_across
    height = tile_size * tiles_down
    pixel_data = np.arange(width * height, dtype=np.float32).reshape(height, width)
    return make_minimal_tiff(
        width, height, np.dtype('float32'),
        pixel_data=pixel_data,
        tiled=True,
        tile_size=tile_size,
    )


def _decode(data: bytes) -> np.ndarray:
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    ifd = ifds[0]
    dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)
    return _read_tiles(data, ifd, header, dtype)


def test_parallel_engages_at_default_tile_size_256(monkeypatch):
    """At tile_size=256 with multiple tiles, parallel decode must run.

    256 * 256 == 64 * 1024 exactly, so this is the boundary case the
    old strict ``>`` excluded.
    """
    record: list[dict] = []
    real_cls = concurrent.futures.ThreadPoolExecutor
    monkeypatch.setattr(
        concurrent.futures,
        'ThreadPoolExecutor',
        _PoolSpy(record, real_cls),
    )

    # 2x2 tile grid at the default tile_size=256 -> 4 tiles, each 64K pixels.
    data = _build_tiled_tiff(tile_size=256, tiles_across=2, tiles_down=2)
    arr = _decode(data)

    # Output is correct.
    assert arr.shape == (512, 512)
    assert arr.dtype == np.float32
    assert arr[0, 0] == 0.0
    assert arr[-1, -1] == float(512 * 512 - 1)

    # Most importantly, the parallel path engaged.
    assert len(record) == 1, (
        f"Expected ThreadPoolExecutor to be constructed exactly once, "
        f"got {len(record)} (parallel path did not run for default tile_size)"
    )


def test_sequential_for_small_tiles(monkeypatch):
    """At tile_size=128 (16K pixels), parallel must NOT run.

    The threshold is meant to avoid pool overhead for small tiles, and
    128*128 = 16384 < 64*1024. This guards against an over-eager fix.
    """
    record: list[dict] = []
    real_cls = concurrent.futures.ThreadPoolExecutor
    monkeypatch.setattr(
        concurrent.futures,
        'ThreadPoolExecutor',
        _PoolSpy(record, real_cls),
    )

    data = _build_tiled_tiff(tile_size=128, tiles_across=4, tiles_down=4)
    arr = _decode(data)

    assert arr.shape == (512, 512)
    assert len(record) == 0, (
        f"Expected sequential decode below threshold, but "
        f"ThreadPoolExecutor was constructed {len(record)} time(s)"
    )


def test_sequential_when_only_one_tile(monkeypatch):
    """A single tile must stay on the sequential path even at large size."""
    record: list[dict] = []
    real_cls = concurrent.futures.ThreadPoolExecutor
    monkeypatch.setattr(
        concurrent.futures,
        'ThreadPoolExecutor',
        _PoolSpy(record, real_cls),
    )

    data = _build_tiled_tiff(tile_size=256, tiles_across=1, tiles_down=1)
    arr = _decode(data)

    assert arr.shape == (256, 256)
    assert len(record) == 0, (
        f"Single-tile reads must stay sequential, but the pool was "
        f"constructed {len(record)} time(s)"
    )
