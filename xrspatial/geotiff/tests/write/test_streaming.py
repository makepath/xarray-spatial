"""Streaming and parallel writer paths.

Covers the dask-backed streaming write surface and the thread-pool
machinery that backs it:

- the parallel strip / tile writer and its byte-threshold gate, plus the
  libdeflate backend fallback;
- streaming round-trip correctness, geo-metadata preservation, edge
  cases, multiband layouts, BigTIFF / error handling, COG fallback, and
  the ``streaming_buffer_bytes`` horizontal-segmentation budget;
- parallel per-tile compression inside the streaming path -- bit-exact
  round trip, observable multi-thread participation, and an opt-in
  wall-clock tripwire;
- ThreadPoolExecutor shutdown on mid-stream failure so a compress / file
  -write error never leaks worker threads.

Section banners below mark the topical sub-areas.
"""
from __future__ import annotations

import inspect
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _writer as writer_mod
from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._compression import (_HAVE_LIBDEFLATE, COMPRESSION_DEFLATE, COMPRESSION_NONE,
                                            LERC_AVAILABLE, LZ4_AVAILABLE, deflate_compress)
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import _PARALLEL_MIN_BYTES, _write_stripped, _write_tiled, write

# -------------------------------------------------------------------------
# Section: parallel strip / tile writer
# -------------------------------------------------------------------------


def _make_data_1800(h, w, dtype=np.float32, pattern='gradient'):
    """Reproducible array used across the parallel-writer tests."""
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
    expected = _make_data_1800(1024, 768, pattern='random')
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
        expected = _make_data_1800(800, 400, dtype=dtype, pattern='random')
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
    expected = _make_data_1800(32, 64, pattern='gradient')
    assert expected.nbytes < _PARALLEL_MIN_BYTES
    path = str(tmp_path / 'small_seq_strip_1800.tif')
    write(expected, path, compression='deflate', tiled=False)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)


def test_strip_writer_thread_pool_used_when_large(monkeypatch):
    """A multi-MiB strip write must dispatch through ThreadPoolExecutor."""
    expected = _make_data_1800(2048, 2048, dtype=np.float32, pattern='random')
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
    expected = _make_data_1800(2048, 2048, dtype=np.float32, pattern='gradient')
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


def test_tile_writer_large_tile_size_parallelizes(monkeypatch):
    """A 2048x2048 deflate write with tile_size=1024 (n_tiles=4) must run
    in parallel after the threshold fix.

    Pre-fix, ``n_tiles <= 4`` shoved this case onto the serial path even
    though the payload was 16 MiB; that produced ~8x slower writes.
    """
    expected = _make_data_1800(2048, 2048, dtype=np.float32, pattern='random')
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
    expected = _make_data_1800(128, 128, dtype=np.float32, pattern='gradient')
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


def test_write_strip_deflate_round_trip_multi_strip(tmp_path):
    """Drive the writer entrypoint with a multi-strip deflate payload.

    The reader doesn't care which path produced the bytes; this guards
    the full write pipeline (predictor on, multiple strips).
    """
    expected = _make_data_1800(900, 700, dtype=np.float32, pattern='random')
    path = str(tmp_path / 'e2e_strip_1800.tif')
    write(expected, path, compression='deflate', tiled=False, predictor=True)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)


def test_write_tiled_deflate_large_tile_round_trip(tmp_path):
    """tile_size=1024 on 2048x2048 must round-trip through the parallel path."""
    expected = _make_data_1800(2048, 2048, dtype=np.float32, pattern='random')
    path = str(tmp_path / 'e2e_tile1024_1800.tif')
    write(expected, path, compression='deflate', tiled=True, tile_size=1024)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)


# -------------------------------------------------------------------------
# Section: streaming write round-trip
# -------------------------------------------------------------------------

@pytest.fixture
def sample_raster():
    """200x200 float32 raster with coords and CRS."""
    arr = np.random.default_rng(1084).random((200, 200), dtype=np.float32)
    y = np.linspace(41.0, 40.0, 200)
    x = np.linspace(-106.0, -105.0, 200)
    return xr.DataArray(arr, dims=['y', 'x'],
                        coords={'y': y, 'x': x},
                        attrs={'crs': 4326, 'nodata': -9999.0})


@pytest.fixture
def dask_raster(sample_raster):
    return sample_raster.chunk({'y': 100, 'x': 100})


class TestStreamingRoundTrip:
    def test_tiled_zstd(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'tiled_zstd_1084.tif')
        to_geotiff(dask_raster, path, compression='zstd')
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tiled_deflate(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'tiled_deflate_1084.tif')
        to_geotiff(dask_raster, path, compression='deflate')
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tiled_lzw(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'tiled_lzw_1084.tif')
        to_geotiff(dask_raster, path, compression='lzw')
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tiled_uncompressed(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'tiled_none_1084.tif')
        to_geotiff(dask_raster, path, compression='none')
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_stripped(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'stripped_1084.tif')
        to_geotiff(dask_raster, path, tiled=False)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_predictor(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'pred_1084.tif')
        to_geotiff(dask_raster, path, predictor=True)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_compression_level(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'level_1084.tif')
        to_geotiff(dask_raster, path, compression='deflate',
                   compression_level=1)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_matches_eager_write(self, sample_raster, dask_raster, tmp_path):
        """Streaming and eager paths should produce identical pixel data."""
        eager_path = str(tmp_path / 'eager_1084.tif')
        stream_path = str(tmp_path / 'stream_1084.tif')

        to_geotiff(sample_raster, eager_path)   # numpy -> eager
        to_geotiff(dask_raster, stream_path)     # dask -> streaming

        eager = open_geotiff(eager_path)
        stream = open_geotiff(stream_path)
        np.testing.assert_array_equal(eager.values, stream.values)


class TestStreamingGeoMetadata:
    def test_crs_preserved(self, dask_raster, tmp_path):
        path = str(tmp_path / 'crs_1084.tif')
        to_geotiff(dask_raster, path)
        result = open_geotiff(path)
        assert result.attrs.get('crs') == 4326

    def test_nodata_preserved(self, dask_raster, tmp_path):
        path = str(tmp_path / 'nd_1084.tif')
        to_geotiff(dask_raster, path)
        result = open_geotiff(path)
        assert float(result.attrs.get('nodata')) == pytest.approx(-9999.0)

    def test_coordinates_preserved(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'coords_1084.tif')
        to_geotiff(dask_raster, path)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.coords['x'].values, sample_raster.coords['x'].values,
            decimal=6)
        np.testing.assert_array_almost_equal(
            result.coords['y'].values, sample_raster.coords['y'].values,
            decimal=6)


class TestStreamingEdgeCases:
    def test_nan_to_nodata(self, tmp_path):
        """NaN pixels should round-trip through the nodata sentinel."""
        arr = np.ones((100, 100), dtype=np.float32)
        arr[10:20, 10:20] = np.nan
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0})
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'nan_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path, masked=True)

        assert np.isnan(result.values[15, 15])
        assert result.values[0, 0] == pytest.approx(1.0)

    def test_single_chunk(self, sample_raster, tmp_path):
        """Single chunk = whole array, but still goes through streaming."""
        dask_da = sample_raster.chunk({'y': 200, 'x': 200})
        path = str(tmp_path / 'single_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_uneven_chunks(self, tmp_path):
        """Chunks that don't divide evenly into tile_size."""
        arr = np.arange(150 * 170, dtype=np.float32).reshape(150, 170)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 64, 'x': 64})

        path = str(tmp_path / 'uneven_1084.tif')
        to_geotiff(dask_da, path, tile_size=128)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_small_raster(self, tmp_path):
        """Raster smaller than one tile."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 2, 'x': 2})

        path = str(tmp_path / 'tiny_1084.tif')
        to_geotiff(dask_da, path, tile_size=256)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_uint16(self, tmp_path):
        arr = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'u16_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_int32(self, tmp_path):
        arr = np.arange(10000, dtype=np.int32).reshape(100, 100)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'i32_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_float64(self, tmp_path):
        arr = np.random.default_rng(1084).random((80, 80))
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 40, 'x': 40})

        path = str(tmp_path / 'f64_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, arr)


class TestStreamingMultiband:
    def test_3d_band_last(self, tmp_path):
        """3D array with (y, x, band) layout."""
        arr = np.random.default_rng(1084).random(
            (100, 100, 3), dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x', 'band'])
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'band_last_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, arr, decimal=5)

    def test_3d_band_first(self, tmp_path):
        """Band-first (band, y, x) DataArray gets transposed automatically."""
        arr = np.random.default_rng(1084).random(
            (3, 100, 100), dtype=np.float32)
        da = xr.DataArray(arr, dims=['band', 'y', 'x'])
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'band_first_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        # Result is (y, x, band), so compare transposed
        np.testing.assert_array_almost_equal(
            result.values, np.moveaxis(arr, 0, -1), decimal=5)


class TestStreamingBigTiffAndErrors:
    def test_forced_bigtiff(self, tmp_path):
        """bigtiff=True on a small array should produce a valid BigTIFF."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 4, 'x': 4})

        path = str(tmp_path / 'bigtiff_1084.tif')
        to_geotiff(dask_da, path, bigtiff=True)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_cloud_uri_raises(self, tmp_path):
        """Streaming to cloud storage should raise NotImplementedError."""
        arr = np.ones((10, 10), dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 5, 'x': 5})

        with pytest.raises(NotImplementedError, match='cloud'):
            to_geotiff(dask_da, 's3://bucket/file.tif')


class TestCogFallback:
    def test_cog_with_dask_still_works(self, sample_raster, tmp_path):
        """cog=True with dask input should fall through to eager compute."""
        dask_da = sample_raster.chunk({'y': 100, 'x': 100})
        path = str(tmp_path / 'cog_1084.tif')
        to_geotiff(dask_da, path, cog=True)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)


class TestStreamingBufferBudget:
    """Horizontal segmentation when a tile-row exceeds the buffer budget."""

    def test_round_trip_wide_raster(self, tmp_path):
        """Tight buffer forces multi-segment compute; output must be byte-equal."""
        rng = np.random.default_rng(1485)
        arr = rng.random((1024, 8192), dtype=np.float64)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 256, 'x': 1024})

        # Default budget: one segment covers the whole row (control)
        ref_path = str(tmp_path / 'wide_default_1485.tif')
        to_geotiff(dask_da, ref_path, compression='zstd')
        ref = open_geotiff(ref_path).values

        # Tight budget: forces ~2-tile segments per tile-row
        seg_path = str(tmp_path / 'wide_segmented_1485.tif')
        to_geotiff(dask_da, seg_path, compression='zstd',
                   streaming_buffer_bytes=2 * 256 * 256 * 8)  # 2 tile cols
        seg = open_geotiff(seg_path).values

        np.testing.assert_array_equal(ref, arr)
        np.testing.assert_array_equal(seg, arr)

    def test_tight_4mb_budget_succeeds(self, tmp_path):
        """A 4 MB cap on a wide raster must succeed without OOM."""
        # 256 rows * 8192 cols * 8 bytes = 16 MB per tile-row.
        # 4 MB budget forces splitting each tile-row into segments.
        arr = np.arange(256 * 8192, dtype=np.float64).reshape(256, 8192)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 256, 'x': 2048})

        path = str(tmp_path / 'tight_budget_1485.tif')
        to_geotiff(dask_da, path, compression='zstd',
                   streaming_buffer_bytes=4 * 1024 * 1024)

        result = open_geotiff(path).values
        np.testing.assert_array_equal(result, arr)

    def test_smaller_than_one_tile_clamps(self, tmp_path):
        """Budget below one tile must still produce a valid file (clamped)."""
        arr = np.arange(256 * 1024, dtype=np.float32).reshape(256, 1024)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 256, 'x': 256})

        path = str(tmp_path / 'tiny_budget_1485.tif')
        # 1 byte is well below one tile (256*256*4 = 262144 bytes)
        to_geotiff(dask_da, path, compression='zstd',
                   streaming_buffer_bytes=1)

        result = open_geotiff(path).values
        np.testing.assert_array_equal(result, arr)

    def test_multiband_segmentation(self, tmp_path):
        """3-band float64 raster with horizontal segmentation."""
        rng = np.random.default_rng(1485)
        arr = rng.random((512, 2048, 3), dtype=np.float64)
        da = xr.DataArray(arr, dims=['y', 'x', 'band'])
        dask_da = da.chunk({'y': 256, 'x': 512, 'band': 3})

        path = str(tmp_path / 'multiband_1485.tif')
        # Force ~2 tile-cols per segment
        to_geotiff(dask_da, path, compression='zstd',
                   streaming_buffer_bytes=2 * 256 * 256 * 8 * 3)

        result = open_geotiff(path).values
        np.testing.assert_array_almost_equal(result, arr, decimal=10)

    def test_overlap_source_respects_buffer_3007(self, tmp_path, monkeypatch):
        """A map_overlap source must not pull far more than the buffer.

        Slicing one tile-row out of a ``slope`` (map_overlap, depth 1)
        result materialises every source chunk-row the band touches plus
        a one-chunk halo. When the source chunks are taller than the tile
        the old budget -- sized from the output tile height -- left the
        whole tile-row in a single segment and the compute pulled several
        times the buffer. Bound peak source bytes per compute (#3007).
        """
        import dask.array as da

        from xrspatial import slope

        height, width, chunk = 1024, 8192, 512  # chunks taller than tile=256
        base = da.random.random(
            (height, width), chunks=(chunk, chunk)).astype('float32')

        materialized = []  # bytes of each source chunk as dask pulls it

        def _record(block):
            materialized.append(block.nbytes)
            return block

        base = base.map_blocks(_record, dtype='float32')
        y = np.linspace(40.0, 39.0, height)
        x = np.linspace(-105.0, -104.0, width)
        lazy = slope(xr.DataArray(base, dims=['y', 'x'],
                                  coords={'y': y, 'x': x}))

        peak = {'bytes': 0}
        orig_compute = da.Array.compute

        def spy_compute(self, *args, **kwargs):
            before = sum(materialized)
            result = orig_compute(self, *args, **kwargs)
            peak['bytes'] = max(peak['bytes'], sum(materialized) - before)
            return result

        monkeypatch.setattr(da.Array, 'compute', spy_compute)

        buf = 8 * 1024 * 1024  # 8 MB
        path = str(tmp_path / 'overlap_budget_3007.tif')
        to_geotiff(lazy, path, compression='zstd',
                   streaming_buffer_bytes=buf)

        # streaming_buffer_bytes is a soft cap; the column halo of the 2D
        # overlap adds a bounded couple of source chunk-columns on top of
        # the row budget. Allow 2x slack. The unfixed writer pulled ~4x
        # (one full-width 2-chunk-row band == 32 MB) and trips this.
        assert peak['bytes'] <= 2 * buf, (
            f"peak source bytes {peak['bytes']} exceeded 2x buffer {buf}")

    def test_overlap_source_roundtrip_small_buffer_3007(self, tmp_path):
        """slope() of a tall-chunk wide raster round-trips under a tight cap.

        Exercises the source-aware segmentation end to end: the lazy
        slope written with a small buffer must match the eager slope
        (#3007).
        """
        import dask.array as da

        from xrspatial import slope

        height, width, chunk = 768, 4096, 512
        rng = np.random.default_rng(3007)
        npdata = rng.random((height, width)).astype('float32')
        y = np.linspace(40.0, 39.0, height)
        x = np.linspace(-105.0, -104.0, width)

        eager = slope(xr.DataArray(npdata, dims=['y', 'x'],
                                   coords={'y': y, 'x': x}))
        base = da.from_array(npdata, chunks=(chunk, chunk))
        lazy = slope(xr.DataArray(base, dims=['y', 'x'],
                                  coords={'y': y, 'x': x}))

        path = str(tmp_path / 'overlap_roundtrip_3007.tif')
        to_geotiff(lazy, path, compression='zstd',
                   streaming_buffer_bytes=4 * 1024 * 1024)

        back = open_geotiff(path).values
        np.testing.assert_allclose(back, eager.values,
                                   rtol=1e-5, atol=1e-5, equal_nan=True)


def test_max_streaming_row_span_3007():
    """The row-span helper accounts for the source chunk grid + halo."""
    from xrspatial.geotiff._writer import _max_streaming_row_span

    # A 256-tall band landing in a 512-tall chunk pulls the neighbour
    # chunk-row too -> 1024 source rows.
    assert _max_streaming_row_span((512, 512), 256, 1024) == 1024
    # Uniform 256 chunks: a middle band touches three chunk-rows (its own
    # plus one halo each side).
    assert _max_streaming_row_span((256, 256, 256, 256), 256, 1024) == 768
    # A single chunk-row spanning the whole height has no neighbour.
    assert _max_streaming_row_span((256,), 256, 256) == 256


# -------------------------------------------------------------------------
# Section: parallel per-tile streaming compress (P4)
# -------------------------------------------------------------------------

def _make_dataarray_parallel(shape, dtype=np.float32, seed=20260508):
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
    da = _make_dataarray_parallel(shape, dtype=dtype)
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
    da = _make_dataarray_parallel(shape, dtype=np.float32, seed=42)
    dask_da = da.chunk({'y': 256 * 4, 'x': 256 * 4})

    path = str(tmp_path / 'parallel_check.tif')
    to_geotiff(dask_da, path,
               compression='deflate', tile_size=256)

    assert len(seen_threads) > 1, (
        f"Expected >1 worker threads in streaming compress, "
        f"saw {len(seen_threads)}: {seen_threads}")


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
    da = _make_dataarray_parallel(shape, dtype=np.float32, seed=2026)
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


# -------------------------------------------------------------------------
# Section: tile-pool shutdown on mid-stream failure
# -------------------------------------------------------------------------

# Re-use the writer's own constant so the test does not silently drift
# if the prefix ever changes on the writer side. ``_writer`` exposes
# ``_TILE_POOL_THREAD_PREFIX`` for exactly this purpose.
_WRITER_POOL_PREFIX = writer_mod._TILE_POOL_THREAD_PREFIX


def _make_dataarray_2276(shape, dtype=np.float32, seed=20260521):
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
    da = _make_dataarray_2276(shape)
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
    da = _make_dataarray_2276(shape)
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
    da = _make_dataarray_2276(shape)
    dask_da = da.chunk({'y': 512, 'x': 512})

    out_path = str(tmp_path / 'tmp_2276_happy.tif')
    to_geotiff(dask_da, out_path,
               compression='deflate', tile_size=256)

    assert len(captured_pools) >= 1
    for idx, pool in enumerate(captured_pools):
        assert pool._shutdown, (
            f"Captured pool #{idx} not shut down on the success path.")

# ===========================================================================
# Streaming dask-write codecs (LERC/LZ4/packbits gap sweep)
# Source: test_streaming_codecs_2026_05_11.py
# ===========================================================================


pytest.importorskip("dask")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def float_raster():
    """200x200 float32 raster shaped to exceed a single chunk."""
    rng = np.random.default_rng(20260511)
    arr = (rng.standard_normal((200, 200)) * 50.0).astype(np.float32)
    return xr.DataArray(arr, dims=['y', 'x'])


@pytest.fixture
def dask_float_raster(float_raster):
    return float_raster.chunk({'y': 64, 'x': 64})


@pytest.fixture
def uint8_raster():
    """200x200 uint8 raster.

    packbits compresses runs of equal bytes, so we keep modest entropy
    rather than a uniform random fill -- the test still pins round-trip,
    not compression ratio.
    """
    rng = np.random.default_rng(20260511)
    arr = (rng.integers(0, 16, size=(200, 200), dtype=np.uint8))
    return xr.DataArray(arr, dims=['y', 'x'])


@pytest.fixture
def dask_uint8_raster(uint8_raster):
    return uint8_raster.chunk({'y': 64, 'x': 64})


# ---------------------------------------------------------------------------
# LERC: lossless + lossy parity with eager writer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LERC_AVAILABLE, reason="lerc not installed")
class TestStreamingLerc:
    def test_lossless_round_trip(self, float_raster, dask_float_raster,
                                 tmp_path):
        """Dask + LERC (max_z_error=0) round-trips exactly."""
        path = str(tmp_path / 'stream_lerc_lossless.tif')
        # Tier 3 codec (issue #2137); opt in to exercise the encode path.
        to_geotiff(dask_float_raster, path, compression='lerc',
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        # LERC with max_z_error=0 is lossless for float32 sources.
        np.testing.assert_array_equal(result.values, float_raster.values)

    def test_lossy_respects_max_z_error(self, float_raster, dask_float_raster,
                                        tmp_path):
        """Dask + LERC with non-zero max_z_error keeps every pixel within bound."""
        max_z = 0.1
        path = str(tmp_path / 'stream_lerc_lossy.tif')
        to_geotiff(dask_float_raster, path,
                   compression='lerc', max_z_error=max_z,
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        max_diff = float(np.abs(result.values - float_raster.values).max())
        assert max_diff <= max_z + 1e-7, (
            f"LERC lossy stream write exceeded max_z_error budget: "
            f"{max_diff} > {max_z}")

    def test_streaming_matches_eager(self, float_raster, dask_float_raster,
                                     tmp_path):
        """Pixel-identical output between eager and streaming LERC writers.

        This is the parity guarantee: both paths feed the same
        ``_compress_block`` with the same ``max_z_error``, so the
        encoded byte stream and decoded pixels should match exactly.
        """
        eager_path = str(tmp_path / 'eager_lerc.tif')
        stream_path = str(tmp_path / 'stream_lerc.tif')
        to_geotiff(float_raster, eager_path,
                   compression='lerc', max_z_error=0.05,
                   allow_experimental_codecs=True)
        to_geotiff(dask_float_raster, stream_path,
                   compression='lerc', max_z_error=0.05,
                   allow_experimental_codecs=True)
        eager = open_geotiff(eager_path, allow_experimental_codecs=True).values
        stream = open_geotiff(stream_path, allow_experimental_codecs=True).values
        np.testing.assert_array_equal(eager, stream)


# ---------------------------------------------------------------------------
# LZ4: round-trip parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LZ4_AVAILABLE, reason="lz4 not installed")
class TestStreamingLz4:
    def test_round_trip(self, float_raster, dask_float_raster, tmp_path):
        path = str(tmp_path / 'stream_lz4.tif')
        # Tier 3 codec (issue #2137); opt in to exercise the encode path.
        to_geotiff(dask_float_raster, path, compression='lz4',
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, float_raster.values)

    def test_streaming_matches_eager(self, float_raster, dask_float_raster,
                                     tmp_path):
        eager_path = str(tmp_path / 'eager_lz4.tif')
        stream_path = str(tmp_path / 'stream_lz4.tif')
        to_geotiff(float_raster, eager_path, compression='lz4',
                   allow_experimental_codecs=True)
        to_geotiff(dask_float_raster, stream_path, compression='lz4',
                   allow_experimental_codecs=True)
        eager = open_geotiff(eager_path, allow_experimental_codecs=True).values
        stream = open_geotiff(stream_path, allow_experimental_codecs=True).values
        np.testing.assert_array_equal(eager, stream)


# ---------------------------------------------------------------------------
# packbits: round-trip on uint8 (the dtype packbits is designed for)
# ---------------------------------------------------------------------------

class TestStreamingPackbits:
    def test_round_trip_uint8(self, uint8_raster, dask_uint8_raster, tmp_path):
        path = str(tmp_path / 'stream_packbits.tif')
        to_geotiff(dask_uint8_raster, path, compression='packbits')
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, uint8_raster.values)

    def test_streaming_matches_eager(self, uint8_raster, dask_uint8_raster,
                                     tmp_path):
        eager_path = str(tmp_path / 'eager_packbits.tif')
        stream_path = str(tmp_path / 'stream_packbits.tif')
        to_geotiff(uint8_raster, eager_path, compression='packbits')
        to_geotiff(dask_uint8_raster, stream_path, compression='packbits')
        eager = open_geotiff(eager_path, allow_experimental_codecs=True).values
        stream = open_geotiff(stream_path, allow_experimental_codecs=True).values
        np.testing.assert_array_equal(eager, stream)


# ---------------------------------------------------------------------------
# Cubic overview resampling (scipy code path)
# ---------------------------------------------------------------------------

class TestCubicOverview:
    """Pin the scipy-based cubic resampler in ``_block_reduce_2d``.

    The other overview methods (mean / nearest / min / max / median / mode)
    have direct test coverage in ``test_cog.py``. The ``cubic`` branch
    only ran in production code; this test ensures the scipy import,
    zoom call, and dtype-preservation cast all work end-to-end.
    """

    def test_cubic_overview_round_trip(self, tmp_path):
        pytest.importorskip("scipy")
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'cubic_overview.tif')
        to_geotiff(arr, path,
                   compression='deflate',
                   tile_size=16,
                   tiled=True,
                   cog=True,
                   overview_levels=[2],
                   overview_resampling='cubic')

        # Full-resolution data is preserved exactly.
        full = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(full.values, arr)

        # Overview is half-size and same dtype (the cubic branch ends in
        # ``.astype(arr2d.dtype)``).
        ov = open_geotiff(path, overview_level=1)
        assert ov.shape == (8, 8)
        assert ov.dtype == arr.dtype

        # Cubic resampling values lie within the source range (no
        # ringing escape on a monotonic ramp).
        assert float(ov.values.min()) >= float(arr.min()) - 1.0
        assert float(ov.values.max()) <= float(arr.max()) + 1.0

    def test_cubic_distinct_from_mean(self, tmp_path):
        """Cubic and mean produce different overview pixels on a non-linear ramp.

        Sanity-check that the cubic branch actually engages (the test
        would still pass on a perfectly linear ramp because both
        methods reduce to the same value, so use a quadratic).
        """
        pytest.importorskip("scipy")
        x = np.arange(16, dtype=np.float32)
        y = x[:, None]
        arr = (x * x + y * y * 0.5).astype(np.float32)

        cubic_path = str(tmp_path / 'cubic_q.tif')
        mean_path = str(tmp_path / 'mean_q.tif')
        to_geotiff(arr, cubic_path, compression='deflate', tile_size=16,
                   tiled=True, cog=True, overview_levels=[2],
                   overview_resampling='cubic')
        to_geotiff(arr, mean_path, compression='deflate', tile_size=16,
                   tiled=True, cog=True, overview_levels=[2],
                   overview_resampling='mean')

        cubic_ov = open_geotiff(cubic_path, allow_experimental_codecs=True, overview_level=1).values
        mean_ov = open_geotiff(mean_path, allow_experimental_codecs=True, overview_level=1).values
        assert not np.array_equal(cubic_ov, mean_ov), (
            "cubic and mean overview should differ on a non-linear ramp")
