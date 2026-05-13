"""Round-trip and threshold tests for the parallel strip/tile writer (#1800)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from xrspatial.geotiff._writer import (
    _PARALLEL_MIN_BYTES,
    _write_stripped,
    _write_tiled,
    write,
)
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._compression import (
    COMPRESSION_DEFLATE,
    COMPRESSION_NONE,
    _HAVE_LIBDEFLATE,
    deflate_compress,
)


# -- Strip writer parity --------------------------------------------------


def _make_data(h, w, dtype=np.float32, pattern='gradient'):
    """Reproducible array used across tests."""
    n = h * w
    if pattern == 'gradient':
        return np.arange(n, dtype=dtype).reshape(h, w)
    rng = np.random.RandomState(1800)
    arr = rng.rand(h, w) * 1000
    return arr.astype(dtype)


@pytest.mark.parametrize('compression', ['deflate', 'lzw', 'zstd'])
@pytest.mark.parametrize('predictor', [False, True])
def test_strip_writer_round_trip_large(tmp_path, compression, predictor):
    """Multi-strip writes round-trip bit-identically through the parallel path."""
    expected = _make_data(1024, 768, pattern='random')
    path = str(tmp_path / f'parallel_strip_1800_{compression}_{predictor}.tif')
    write(expected, path, compression=compression, tiled=False,
          predictor=predictor)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.int16, np.int32,
                                   np.float32, np.float64])
def test_strip_writer_dtypes(tmp_path, dtype):
    """Parallel strip path preserves every supported numeric dtype."""
    if np.issubdtype(dtype, np.floating):
        expected = _make_data(800, 400, dtype=dtype, pattern='random')
    else:
        info = np.iinfo(dtype)
        rng = np.random.RandomState(1800)
        expected = rng.randint(info.min, info.max,
                               size=(800, 400), dtype=dtype)
    path = str(tmp_path / f'parallel_strip_1800_dtype_{dtype.__name__}.tif')
    write(expected, path, compression='deflate', tiled=False)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)


def test_strip_writer_small_takes_sequential_path(tmp_path):
    """Below the byte threshold the parallel strip path is skipped.

    The sequential branch is functionally identical, so the round-trip
    check just guards against the threshold logic accidentally breaking
    the small-payload case.
    """
    expected = _make_data(32, 64, pattern='gradient')
    assert expected.nbytes < _PARALLEL_MIN_BYTES
    path = str(tmp_path / 'small_seq_strip_1800.tif')
    write(expected, path, compression='deflate', tiled=False)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)


def test_strip_writer_thread_pool_used_when_large(monkeypatch):
    """A multi-MiB strip write must dispatch through ThreadPoolExecutor."""
    expected = _make_data(2048, 2048, dtype=np.float32, pattern='random')
    assert expected.nbytes > _PARALLEL_MIN_BYTES

    used = {'pool': False}

    import concurrent.futures as cf

    class _Probe(cf.ThreadPoolExecutor):
        def __init__(self, *a, **kw):
            used['pool'] = True
            super().__init__(*a, **kw)

    # The writer does ``from concurrent.futures import ThreadPoolExecutor``
    # inside the function, so patching the module attribute is enough.
    monkeypatch.setattr(cf, 'ThreadPoolExecutor', _Probe)

    rel, bc, blobs = _write_stripped(
        expected, COMPRESSION_DEFLATE, predictor=1, rows_per_strip=256)
    assert used['pool'], 'parallel strip writer should have used ThreadPoolExecutor'
    # And the output should still round-trip
    import zlib
    decoded = b''.join(zlib.decompress(b) for b in blobs)
    rt = np.frombuffer(decoded, dtype=np.float32).reshape(expected.shape)
    np.testing.assert_array_equal(rt, expected)


def test_strip_writer_uncompressed_stays_sequential(monkeypatch):
    """``compression='none'`` never dispatches to the thread pool."""
    expected = _make_data(2048, 2048, dtype=np.float32, pattern='gradient')
    assert expected.nbytes > _PARALLEL_MIN_BYTES

    used = {'pool': False}

    import concurrent.futures as cf

    class _Probe(cf.ThreadPoolExecutor):
        def __init__(self, *a, **kw):
            used['pool'] = True
            super().__init__(*a, **kw)

    monkeypatch.setattr(cf, 'ThreadPoolExecutor', _Probe)
    _write_stripped(expected, COMPRESSION_NONE, predictor=1, rows_per_strip=256)
    assert not used['pool'], 'uncompressed strip writer must stay sequential'


# -- Tile writer adaptive threshold ---------------------------------------


def test_tile_writer_large_tile_size_parallelizes(monkeypatch):
    """A 2048x2048 deflate write with tile_size=1024 (n_tiles=4) must run
    in parallel after the threshold fix.

    Pre-fix, ``n_tiles <= 4`` shoved this case onto the serial path even
    though the payload was 16 MiB; that produced ~8x slower writes.
    """
    expected = _make_data(2048, 2048, dtype=np.float32, pattern='random')
    assert expected.nbytes > _PARALLEL_MIN_BYTES

    used = {'pool': False}

    import concurrent.futures as cf

    class _Probe(cf.ThreadPoolExecutor):
        def __init__(self, *a, **kw):
            used['pool'] = True
            super().__init__(*a, **kw)

    monkeypatch.setattr(cf, 'ThreadPoolExecutor', _Probe)
    _write_tiled(
        expected, COMPRESSION_DEFLATE, predictor=1, tile_size=1024)
    assert used['pool'], (
        'tile writer with tile_size=1024 on 2048x2048 (n_tiles=4, 16 MiB) '
        'must parallelize after the adaptive-threshold change'
    )


def test_tile_writer_small_payload_stays_sequential(monkeypatch):
    """A small raster keeps the sequential path even with n_tiles > 2."""
    expected = _make_data(128, 128, dtype=np.float32, pattern='gradient')
    assert expected.nbytes < _PARALLEL_MIN_BYTES

    used = {'pool': False}

    import concurrent.futures as cf

    class _Probe(cf.ThreadPoolExecutor):
        def __init__(self, *a, **kw):
            used['pool'] = True
            super().__init__(*a, **kw)

    monkeypatch.setattr(cf, 'ThreadPoolExecutor', _Probe)
    _write_tiled(
        expected, COMPRESSION_DEFLATE, predictor=1, tile_size=32)
    assert not used['pool']


# -- libdeflate backend ----------------------------------------------------


def test_deflate_compress_zlib_wire_compatible():
    """Output is decompressible by stdlib zlib regardless of backend."""
    import zlib
    raw = (np.arange(1024, dtype=np.uint8) % 251).tobytes() * 64
    compressed = deflate_compress(raw, level=6)
    assert zlib.decompress(compressed) == raw


def test_deflate_compress_fallback_when_libdeflate_missing(monkeypatch):
    """When the deflate package is absent we route through stdlib zlib unchanged."""
    import zlib

    import xrspatial.geotiff._compression as comp_mod
    monkeypatch.setattr(comp_mod, '_HAVE_LIBDEFLATE', False)
    monkeypatch.setattr(comp_mod, '_deflate', None)
    # Reset the one-shot warning latch so the test path is exercised cleanly.
    monkeypatch.setattr(comp_mod, '_zlib_fallback_warned', True)

    raw = b'1800-deflate-fallback' * 4096
    blob = comp_mod.deflate_compress(raw, level=6)
    assert zlib.decompress(blob) == raw
    # Exact byte equality to ``zlib.compress`` at the same level (the
    # fallback path is a direct call).
    assert blob == zlib.compress(raw, 6)


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_deflate_compress_uses_libdeflate_when_available():
    """When the deflate package is installed, output stays wire-compatible."""
    import zlib
    raw = (np.arange(8192, dtype=np.uint8) % 251).tobytes() * 16
    blob = deflate_compress(raw, level=6)
    assert zlib.decompress(blob) == raw


# -- End-to-end via write() ------------------------------------------------


def test_write_strip_deflate_round_trip_multi_strip(tmp_path):
    """Drive the writer entrypoint with a multi-strip deflate payload.

    The reader doesn't care which path produced the bytes; this guards
    the full write pipeline (predictor on, multiple strips).
    """
    expected = _make_data(900, 700, dtype=np.float32, pattern='random')
    path = str(tmp_path / 'e2e_strip_1800.tif')
    write(expected, path, compression='deflate', tiled=False, predictor=True)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)


def test_write_tiled_deflate_large_tile_round_trip(tmp_path):
    """tile_size=1024 on 2048x2048 must round-trip through the parallel path."""
    expected = _make_data(2048, 2048, dtype=np.float32, pattern='random')
    path = str(tmp_path / 'e2e_tile1024_1800.tif')
    write(expected, path, compression='deflate', tiled=True, tile_size=1024)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)
