"""Tests for the geotiff polish bundle (issue #1488).

Covers ten low-severity audit items:

* C-1 -- early ``compression`` validation in ``to_geotiff``
* C-2 -- read dispatch leaves ``read_geotiff_dask`` with a defensive
  ``.vrt`` fallback that delegates to ``read_vrt``
* C-5 -- ``write_vrt`` docstring lists kwargs (rejects unknown ones)
* C-6 -- predictor doc covers True/2 equivalence and 3=fp
* C-7 -- ``tile_size`` warns when ``tiled=False`` and non-default
* P-3 -- mmap cache eviction (LRU + env var override)
* P-4 -- decode parallelism gate based on tile pixel count
* P-5 -- dask read raises with a hint instead of silently rescaling
* P-6 -- GPU memory pre-check raises before the cupy call
* P-9 -- COG auto-overview generation capped at 8 levels
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff, read_geotiff_dask, write_vrt
from xrspatial.geotiff._reader import _MmapCache, read_to_array
from xrspatial.geotiff._writer import _MAX_OVERVIEW_LEVELS, write


# ---------------------------------------------------------------------------
# C-1: early compression validation
# ---------------------------------------------------------------------------

class TestC1CompressionValidation:
    def test_unknown_compression_raises_at_top(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'bogus_1488.tif')
        with pytest.raises(ValueError, match="Unknown compression"):
            to_geotiff(arr, path, compression='bzip2')

    def test_known_compression_passes(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'ok_1488.tif')
        # Should not raise.
        to_geotiff(arr, path, compression='deflate')
        assert os.path.exists(path)

    def test_validation_runs_before_vrt_dispatch(self, tmp_path):
        # Bad compression on a .vrt path should still surface from the
        # up-front check, not from deeper in _write_vrt_tiled.
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'bogus_1488.vrt')
        with pytest.raises(ValueError, match="Unknown compression"):
            to_geotiff(arr, path, compression='nope')


# ---------------------------------------------------------------------------
# C-2: read dispatch comment (behavior verification)
# ---------------------------------------------------------------------------

class TestC2ReadDispatch:
    def test_read_geotiff_dask_handles_vrt_directly(self, tmp_path):
        # Build a 2-tile VRT and confirm read_geotiff_dask routes to the
        # VRT reader without trying to parse XML as TIFF.
        from xrspatial.geotiff import write_vrt as wv
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        a_path = str(tmp_path / 'a_1488.tif')
        b_path = str(tmp_path / 'b_1488.tif')
        # Two tiles side-by-side via geo-transform attrs would normally be
        # generated upstream; for this test we just need both files to
        # share a CRS and the writer's default transform.
        write(arr, a_path, compression='none')
        write(arr, b_path, compression='none')
        vrt_path = str(tmp_path / 'mosaic_1488.vrt')
        wv(vrt_path, [a_path, b_path])

        result = read_geotiff_dask(vrt_path, chunks=8)
        assert result.dims == ('y', 'x')


# ---------------------------------------------------------------------------
# C-5: write_vrt kwargs documented
# ---------------------------------------------------------------------------

class TestC5WriteVrtKwargs:
    def test_known_kwargs_accepted(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        a_path = str(tmp_path / 'a_c5_1488.tif')
        write(arr, a_path, compression='none')
        vrt_path = str(tmp_path / 'mosaic_c5_1488.vrt')
        # All three documented kwargs should be accepted.
        write_vrt(vrt_path, [a_path], relative=False, crs_wkt=None,
                  nodata=-9999.0)
        assert os.path.exists(vrt_path)

    def test_unknown_kwarg_raises_typeerror(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        a_path = str(tmp_path / 'a_c5b_1488.tif')
        write(arr, a_path, compression='none')
        vrt_path = str(tmp_path / 'mosaic_c5b_1488.vrt')
        with pytest.raises(TypeError):
            write_vrt(vrt_path, [a_path], not_a_real_kwarg=True)

    def test_docstring_lists_kwargs(self):
        # Defensive: the whole point of C-5 is the docstring -- guard
        # against future regressions.
        assert 'relative' in write_vrt.__doc__
        assert 'crs_wkt' in write_vrt.__doc__
        assert 'nodata' in write_vrt.__doc__


# ---------------------------------------------------------------------------
# C-6: predictor docstring polish
# ---------------------------------------------------------------------------

class TestC6PredictorDoc:
    def test_to_geotiff_doc_covers_predictor_modes(self):
        doc = to_geotiff.__doc__
        # True and 2 documented as equivalent
        assert 'True' in doc and '2' in doc
        # 3 documented as fp predictor for floats
        assert '3' in doc and 'float' in doc.lower()


# ---------------------------------------------------------------------------
# C-7: tile_size warning in strip mode
# ---------------------------------------------------------------------------

class TestC7TileSizeWarn:
    def test_warning_when_tile_size_set_with_tiled_false(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'strip_1488.tif')
        with pytest.warns(UserWarning, match="tile_size.*ignored"):
            to_geotiff(arr, path, tiled=False, tile_size=128,
                       compression='none')

    def test_no_warning_when_tile_size_default(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'strip2_1488.tif')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            to_geotiff(arr, path, tiled=False, compression='none')

    def test_no_warning_when_tiled_true(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'tiled_1488.tif')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            to_geotiff(arr, path, tiled=True, tile_size=128,
                       compression='none')


# ---------------------------------------------------------------------------
# P-3: mmap LRU cache cap
# ---------------------------------------------------------------------------

class TestP3MmapLRU:
    def test_cap_evicts_oldest_idle_entry(self, tmp_path):
        cache = _MmapCache(max_size=2)
        files = []
        for i in range(3):
            p = tmp_path / f'p3_{i}_1488.bin'
            p.write_bytes(b'x' * 32)
            files.append(str(p))

        # Acquire and release each: all become idle.
        for f in files:
            cache.acquire(f)
            cache.release(f)

        # Cache should be at the cap (2), with the oldest evicted.
        assert len(cache._entries) == 2
        oldest = os.path.realpath(files[0])
        assert oldest not in cache._entries

    def test_inuse_entries_not_evicted(self, tmp_path):
        cache = _MmapCache(max_size=1)
        a = tmp_path / 'p3_a_1488.bin'
        a.write_bytes(b'a' * 32)
        b = tmp_path / 'p3_b_1488.bin'
        b.write_bytes(b'b' * 32)

        cache.acquire(str(a))  # rc=1, in use
        cache.acquire(str(b))  # would exceed cap, but a is in use
        cache.release(str(b))  # b idle now

        # a still in cache because rc > 0.
        assert os.path.realpath(str(a)) in cache._entries

    def test_env_var_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_MMAP_CACHE_SIZE', '5')
        # New cache picks up the env setting.
        cache = _MmapCache()
        assert cache._max_size == 5

    def test_fingerprint_mismatch_invalidates_idle_entry(self, tmp_path):
        """Platform-agnostic check that the invalidation logic itself
        runs. The end-to-end ``os.replace`` and truncate scenarios are
        POSIX-only because Windows blocks the precondition, but the
        fingerprint comparison is the actual line of code that prevents
        stale reads -- exercise it directly by mutating the cached
        fingerprint to simulate a file that changed underneath us."""
        cache = _MmapCache()
        p = tmp_path / 'fingerprint_1488.bin'
        p.write_bytes(b'A' * 16)

        mm1, _ = cache.acquire(str(p))
        cache.release(str(p))

        real = os.path.realpath(str(p))
        # Force the cached fingerprint to a value the file cannot match,
        # which mimics any real-world change (replace, truncate, rename
        # back) without depending on the OS allowing those operations
        # while we hold an mmap.
        cache._entries[real][4] = (-1, -1, -1)

        # Next acquire must notice the mismatch and re-mmap. The mm
        # object identity should change; the new entry's fingerprint
        # should match the live file.
        mm2, _ = cache.acquire(str(p))
        try:
            assert mm2 is not mm1
            assert cache._entries[real][4] == cache._file_ident(real)
        finally:
            cache.release(str(p))

    @pytest.mark.skipif(
        sys.platform.startswith('win'),
        reason="Windows holds a file lock while a path is mmapped, "
               "so os.replace fails before the cache lookup runs. "
               "The stale-cache bug this test guards against can only "
               "occur on POSIX.",
    )
    def test_replaced_file_invalidates_idle_entry(self, tmp_path):
        """``os.replace`` swaps the inode under a stable path. The cache
        must spot the new inode and re-mmap rather than serve stale bytes.
        Reproduces a real ``to_geotiff`` round-trip pattern -- the writer
        does an atomic rename, so a re-read after overwriting the same
        path used to return data from the unlinked old file."""
        cache = _MmapCache()
        p = tmp_path / 'replaced_1488.bin'
        p.write_bytes(b'A' * 16)

        mm1, _ = cache.acquire(str(p))
        first = bytes(mm1[:16])
        assert first == b'A' * 16
        cache.release(str(p))

        # Atomic-rename style replacement: same path, new inode.
        new = tmp_path / 'replaced_1488.bin.tmp'
        new.write_bytes(b'B' * 16)
        os.replace(str(new), str(p))

        mm2, _ = cache.acquire(str(p))
        assert bytes(mm2[:16]) == b'B' * 16
        cache.release(str(p))

    @pytest.mark.skipif(
        sys.platform.startswith('win'),
        reason="Windows blocks open(path, 'wb') while the path is "
               "mmapped, so the in-place rewrite fails before the "
               "cache invalidation check. POSIX-only scenario.",
    )
    def test_truncated_file_invalidates_idle_entry(self, tmp_path):
        """In-place truncate-and-overwrite (``open(p, 'wb')``) preserves the
        inode but changes size and mtime, so the cache must still
        invalidate."""
        cache = _MmapCache()
        p = tmp_path / 'truncate_1488.bin'
        p.write_bytes(b'A' * 32)

        mm1, sz1 = cache.acquire(str(p))
        assert sz1 == 32
        cache.release(str(p))

        # In-place rewrite with a smaller payload.
        import time
        time.sleep(0.01)  # ensure st_mtime_ns advances
        with open(str(p), 'wb') as fh:
            fh.write(b'C' * 8)

        mm2, sz2 = cache.acquire(str(p))
        assert sz2 == 8
        assert bytes(mm2[:8]) == b'C' * 8
        cache.release(str(p))


# ---------------------------------------------------------------------------
# P-4: decode parallelism threshold
# ---------------------------------------------------------------------------

class TestP4ParallelThreshold:
    def test_uncompressed_large_tiles_round_trip(self, tmp_path):
        # The threshold is purely an internal heuristic; the test confirms
        # the new gate doesn't regress correctness on uncompressed files
        # with multiple large tiles (which previously skipped the pool).
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(512, 512), dtype=np.uint8)
        path = str(tmp_path / 'p4_1488.tif')
        write(arr, path, compression='none', tiled=True, tile_size=256)
        out, _ = read_to_array(path)
        np.testing.assert_array_equal(out, arr)

    def test_small_tile_path_still_correct(self, tmp_path):
        # 64x64 tiles of 64x64 image: tile_pixels = 4096 < 64K, so the
        # gate stays sequential. Round-trip must still match.
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        path = str(tmp_path / 'p4b_1488.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=64)
        out, _ = read_to_array(path)
        np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# P-5: dask task cap
# ---------------------------------------------------------------------------

class TestP5DaskTaskCap:
    def test_excessive_chunks_raises_with_hint(self, tmp_path):
        # 1024x1024 image with chunks=2 => ~262K chunks, well over 50K cap.
        arr = np.zeros((1024, 1024), dtype=np.uint8)
        path = str(tmp_path / 'p5_1488.tif')
        write(arr, path, compression='none', tiled=True, tile_size=64)
        with pytest.raises(ValueError, match=r"50,000-task cap"):
            read_geotiff_dask(path, chunks=2)

    def test_normal_chunks_pass(self, tmp_path):
        arr = np.zeros((512, 512), dtype=np.uint8)
        path = str(tmp_path / 'p5b_1488.tif')
        write(arr, path, compression='none', tiled=True, tile_size=128)
        result = read_geotiff_dask(path, chunks=128)
        assert result.shape == (512, 512)


# ---------------------------------------------------------------------------
# P-6: GPU memory pre-check
# ---------------------------------------------------------------------------

class TestP6GpuMemoryCheck:
    def test_helper_raises_when_request_exceeds_free(self, monkeypatch):
        cupy = pytest.importorskip('cupy')
        from xrspatial.geotiff import _gpu_decode

        # Stub memGetInfo to a tiny free budget so the check trips.
        class _FakeRuntime:
            @staticmethod
            def memGetInfo():
                return (1024, 8 * 1024 * 1024 * 1024)  # 1KB free

        monkeypatch.setattr(cupy.cuda, 'runtime', _FakeRuntime,
                            raising=False)
        with pytest.raises(MemoryError, match="GPU out of memory"):
            _gpu_decode._check_gpu_memory(10 * 1024 * 1024,
                                          what="test buffer")

    def test_helper_noop_for_zero_or_negative(self):
        from xrspatial.geotiff import _gpu_decode
        # Should not even try to query CUDA.
        _gpu_decode._check_gpu_memory(0, what="empty")
        _gpu_decode._check_gpu_memory(-100, what="negative")

    def test_helper_silent_when_cupy_unavailable(self, monkeypatch):
        # When cupy isn't importable, the helper falls through silently
        # so the real allocation can produce its own error.
        from xrspatial.geotiff import _gpu_decode
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'cupy':
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        _gpu_decode._check_gpu_memory(10**9, what="probe")  # no raise


# ---------------------------------------------------------------------------
# P-9: COG auto-overview level cap
# ---------------------------------------------------------------------------

class TestP9OverviewCap:
    def test_max_levels_constant(self):
        assert _MAX_OVERVIEW_LEVELS == 8

    def test_auto_overview_capped(self, tmp_path):
        # 4096x4096 with tile_size=64 would generate ceil(log2(4096/64))
        # = 6 levels (under the cap), so use a tinier tile_size to push
        # the natural count past 8.
        arr = np.zeros((4096, 4096), dtype=np.uint8)
        path = str(tmp_path / 'p9_1488.tif')
        # tile_size=4 -> log2(4096/4)=10 natural levels, but cap is 8.
        write(arr, path, compression='none', tiled=True, tile_size=4,
              cog=True)

        # Re-open and count IFDs (overviews + full-res).
        from xrspatial.geotiff._header import parse_header, parse_all_ifds
        from xrspatial.geotiff._reader import _FileSource
        src = _FileSource(path)
        try:
            data = src.read_all()
            header = parse_header(data)
            ifds = parse_all_ifds(data, header)
        finally:
            src.close()

        # 1 full-res IFD + at most 8 overview IFDs.
        assert len(ifds) <= 1 + _MAX_OVERVIEW_LEVELS

    def test_explicit_overview_levels_not_capped(self, tmp_path):
        # When the caller passes overview_levels explicitly, the cap is
        # not applied -- they get exactly what they asked for.
        arr = np.zeros((1024, 1024), dtype=np.uint8)
        path = str(tmp_path / 'p9b_1488.tif')
        write(arr, path, compression='none', tiled=True, tile_size=64,
              cog=True, overview_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        from xrspatial.geotiff._header import parse_header, parse_all_ifds
        from xrspatial.geotiff._reader import _FileSource
        src = _FileSource(path)
        try:
            data = src.read_all()
            header = parse_header(data)
            ifds = parse_all_ifds(data, header)
        finally:
            src.close()

        # 10 explicit overviews + 1 full-res = 11 IFDs.
        assert len(ifds) == 11
