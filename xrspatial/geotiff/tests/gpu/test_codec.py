"""GPU codec coverage: nvCOMP, nvJPEG / nvJPEG2000, JPEG, LERC, predictor.

This module gathers the GPU codec tests into one home. Sections in
source-order below:

* ``test_nvcomp_batch_compress_batched_1712.py`` -- batched nvCOMP
  compress: single contiguous output alloc + single batched D2H concat.
* ``test_nvcomp_batch_upload_p3.py`` -- batched H2D upload on the
  nvCOMP decompress side; cumulative-sum offset pattern.
* ``test_nvcomp_decompress_cumsum_offsets_1950.py`` -- decompress-side
  prefix-sum offsets via ``np.cumsum`` rather than a Python loop.
* ``test_nvcomp_from_device_bufs_single_alloc_1659.py`` -- single
  contiguous output buffer for the device-buf nvCOMP path.
* ``test_nvjpeg_encode_stream_sync_2212.py`` -- the per-tile encode-
  loop sync uses ``Stream.null.synchronize()`` (not
  ``Device().synchronize()``).
* ``test_nvjpeg2k_single_alloc_2107.py`` -- pool the per-tile alloc +
  per-tile sync in ``_try_nvjpeg2k_batch_decode``.
* ``test_jpeg_gpu_1549.py`` -- nvJPEG output-format constants match
  the SDK; cross-backend pixel parity + context survival.
* ``test_lerc_valid_mask_gpu.py`` -- the GPU LERC tile-decode path
  honours the file's valid-mask, matching the CPU reader.
* ``test_predictor2_big_endian_gpu_1517.py`` -- byte-swap helper +
  predictor=2 BE files match CPU baseline.
* ``test_predictor3_int_dtype_gpu_1933.py`` -- predictor=3 + integer
  SampleFormat is rejected at every GPU entry point.
* ``test_gpu_jpeg_interop_reject_issue_D_1845.py`` -- the GPU writer
  rejects ``compression='jpeg'`` by default and emits a
  ``GeoTIFFFallbackWarning`` on the opt-in.

Every test in this module is gated through the shared ``requires_gpu``
marker from ``_helpers/markers.py``. Module-level helpers carry the
source issue number suffix (e.g. ``_write_jpeg_rgb_tiff_1549``) so
sibling sections stay collision-free.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import pathlib
import re
import tempfile
import time
import uuid
import warnings as _warnings

import numpy as np
import pytest
import xarray as xr

from .._helpers.markers import gpu_available, requires_gpu

# Aliased so the per-file ``_gpu_only`` decorators read the same as
# before the consolidation; the underlying check is the shared
# ``requires_gpu`` marker.
_gpu_only = requires_gpu
needs_cupy = requires_gpu

# A handful of sections additionally gate on optional libraries (tifffile,
# imagecodecs, nvJPEG, etc.). Those gates layer on top of ``requires_gpu``
# below; they need separate skipif decorators because the missing-library
# reason text is informative.
_HAS_GPU = gpu_available()
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_HAS_PIL = importlib.util.find_spec("PIL") is not None
_HAS_IMAGECODECS = importlib.util.find_spec("imagecodecs") is not None


# ============================================================
# Section: nvCOMP batched compress
# ============================================================
# Source: test_nvcomp_batch_compress_batched_1712.py
#
# The pre-fix function allocated compressed-output device buffers one
# ``cupy.empty`` per tile and then read each tile back to host with one
# ``.get()`` per tile. Both patterns serialised on the default CUDA
# stream and were dominant in large-N writes. The fix folds both into
# a single contiguous device allocation + a single batched D2H concat-
# and-``.get()``. These tests pin the new shape and confirm the deflate
# / zstd GPU write paths still round-trip end-to-end.

# nvCOMP is the entry point that exercises this code path.
from xrspatial.geotiff import _gpu_decode  # noqa: E402


def test_no_per_tile_cupy_empty_in_compressed_pool_1712():
    """The per-tile cupy.empty list comprehension is gone (#1712)."""
    source = inspect.getsource(_gpu_decode._nvcomp_batch_compress)
    assert "cupy.empty(max_cs, dtype=cupy.uint8) for _ in range" not in source, (
        "_nvcomp_batch_compress regressed to per-tile cupy.empty "
        "allocations for the compressed output pool. See #1712."
    )


def test_no_per_tile_get_in_result_loop_1712():
    """The per-tile ``d_comp_bufs[i][:cs].get().tobytes()`` is gone (#1712)."""
    source = inspect.getsource(_gpu_decode._nvcomp_batch_compress)
    bad_fragment = "d_comp_bufs[i][:cs].get().tobytes()"
    assert bad_fragment not in source, (
        "_nvcomp_batch_compress regressed to per-tile .get().tobytes() "
        "D2H readback. See #1712."
    )


@requires_gpu
@pytest.mark.parametrize("compression", ["deflate", "zstd"])
def test_gpu_write_roundtrip_after_batched_compress_1712(compression):
    """GPU compress path round-trips uncorrupted for deflate + zstd."""
    import cupy

    from xrspatial.geotiff import open_geotiff, write_geotiff_gpu

    rng = np.random.default_rng(seed=1712)
    arr_cpu = rng.random((512, 512), dtype=np.float32)
    arr_gpu = cupy.asarray(arr_cpu)
    darr = xr.DataArray(arr_gpu, dims=["y", "x"])

    with tempfile.TemporaryDirectory(prefix="nvcomp_batch_1712_") as td:
        path = os.path.join(td, f"roundtrip_{compression}.tif")
        try:
            write_geotiff_gpu(
                darr, path,
                compression=compression,
                tiled=True,
                tile_size=64,
            )
        except RuntimeError as e:
            pytest.skip(f"nvCOMP unavailable for {compression}: {e}")

        back = open_geotiff(path)
        np.testing.assert_allclose(back.values, arr_cpu, rtol=0, atol=0)


@requires_gpu
def test_gpu_write_zero_tile_edge_case_1712():
    """A 0-tile compress returns an empty list without indexing into None."""
    import cupy

    from xrspatial.geotiff import open_geotiff, write_geotiff_gpu

    arr_gpu = cupy.zeros((32, 32), dtype=cupy.float32)
    darr = xr.DataArray(arr_gpu, dims=["y", "x"])
    with tempfile.TemporaryDirectory(prefix="nvcomp_batch_1712_") as td:
        path = os.path.join(td, "tiny.tif")
        try:
            write_geotiff_gpu(darr, path, compression="zstd",
                              tiled=True, tile_size=32)
        except RuntimeError as e:
            pytest.skip(f"nvCOMP unavailable: {e}")
        back = open_geotiff(path)
        assert back.shape == (32, 32)


# ============================================================
# Section: nvCOMP batched H2D upload
# ============================================================
# Source: test_nvcomp_batch_upload_p3.py
#
# The decompress-side fast path used to do one ``cupy.asarray`` per
# compressed tile. The fix concatenates all tiles into a single host
# buffer, performs one H2D transfer, and derives per-tile device
# pointers via ``base_ptr + offsets``.


def _kvikio_nvcomp_importable_p3() -> bool:
    """True iff ``import kvikio.nvcomp`` actually succeeds."""
    try:
        import kvikio.nvcomp  # noqa: F401
    except Exception:
        return False
    return True


def _nvcomp_path_available_p3() -> bool:
    """True when at least one nvCOMP backend is loadable on this host."""
    if not _HAS_GPU:
        return False
    try:
        from xrspatial.geotiff._gpu_decode import _get_nvcomp
    except Exception:
        return False
    if _get_nvcomp() is not None:
        return True
    return _kvikio_nvcomp_importable_p3()


_HAS_NVCOMP_P3 = _nvcomp_path_available_p3()
_nvcomp_only_p3 = pytest.mark.skipif(
    not (_HAS_GPU and _HAS_TIFFFILE and _HAS_NVCOMP_P3),
    reason="cupy + CUDA + tifffile + (libnvcomp or kvikio.nvcomp) required",
)


def _write_deflate_tiled_p3(path, arr, tile=(256, 256)):
    import tifffile
    tifffile.imwrite(
        str(path), arr, compression="deflate", tile=tile,
    )


def _wrap_nvcomp_with_call_recorder_p3(monkeypatch):
    """Replace ``_try_nvcomp_batch_decompress`` with a recording wrapper."""
    from xrspatial.geotiff import _gpu_decode

    records: list[tuple[int, bool]] = []
    original = _gpu_decode._try_nvcomp_batch_decompress

    def _recording(compressed_tiles, tile_bytes, compression):
        result = original(compressed_tiles, tile_bytes, compression)
        records.append((compression, result is not None))
        return result

    monkeypatch.setattr(
        _gpu_decode,
        '_try_nvcomp_batch_decompress',
        _recording,
        raising=True,
    )
    return records


@_nvcomp_only_p3
@pytest.mark.parametrize("size,tile", [
    (256, (128, 128)),    # 4 tiles
    (1024, (256, 256)),   # 16 tiles
    (2048, (128, 128)),   # 256 tiles -- matches the audit measurement
])
def test_nvcomp_batch_upload_correctness_p3(tmp_path, monkeypatch, size, tile):
    """GPU decode of Deflate-tiled TIFFs is bit-exact vs CPU."""
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260508)
    arr = rng.randint(0, 4096, size=(size, size), dtype=np.uint16)

    name = f"deflate_{size}_{tile[0]}_{uuid.uuid4().hex[:8]}.tif"
    path = tmp_path / name
    _write_deflate_tiled_p3(path, arr, tile=tile)

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    records = _wrap_nvcomp_with_call_recorder_p3(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)

    assert any(success for _, success in records), (
        "_try_nvcomp_batch_decompress was never invoked or always returned "
        f"None; records={records}. The optimised path was not exercised, so "
        f"this test would pass even if the rewrite were broken."
    )


@_nvcomp_only_p3
def test_nvcomp_kvikio_fallback_skips_zstd_p3(monkeypatch):
    """ZSTD-compressed input must NOT take the kvikio DeflateManager path."""
    import xrspatial.geotiff._gpu_decode as _gpu_decode

    if not _kvikio_nvcomp_importable_p3():
        pytest.skip("kvikio.nvcomp not importable; the kvikio branch "
                    "is never entered on this host")
    monkeypatch.setattr(_gpu_decode, '_get_nvcomp', lambda: None)

    result = _gpu_decode._try_nvcomp_batch_decompress(
        compressed_tiles=[b'\x28\xb5\x2f\xfd' + b'\x00' * 16],
        tile_bytes=1024,
        compression=50000,  # ZSTD
    )
    assert result is None, (
        "_try_nvcomp_batch_decompress returned non-None for ZSTD via the "
        "kvikio fallback; this would feed ZSTD bytes through DeflateManager "
        "and produce garbage."
    )


@_nvcomp_only_p3
def test_nvcomp_batch_upload_perf_regression_guard_p3(tmp_path, monkeypatch):
    """Sanity guard: 2048x2048 Deflate-tiled GPU decode finishes quickly."""
    from xrspatial.geotiff import read_geotiff_gpu

    rng = np.random.RandomState(20260508)
    arr = rng.randint(0, 4096, size=(2048, 2048), dtype=np.uint16)
    path = tmp_path / f"deflate_2048_perf_{uuid.uuid4().hex[:8]}.tif"
    _write_deflate_tiled_p3(path, arr, tile=(128, 128))

    # Warm up.
    _ = read_geotiff_gpu(str(path))

    records = _wrap_nvcomp_with_call_recorder_p3(monkeypatch)
    t0 = time.perf_counter()
    out = read_geotiff_gpu(str(path))
    elapsed = time.perf_counter() - t0

    assert any(success for _, success in records), (
        "nvCOMP fast-path did not run during the timed call; the threshold "
        f"is meaningless without it. Records: {records}"
    )

    assert elapsed < 0.2, (
        f"read_geotiff_gpu on 2048x2048 deflate-tiled TIFF took "
        f"{elapsed * 1000:.1f} ms (threshold 200 ms) -- possible "
        f"regression in the nvCOMP batched H2D upload path"
    )
    assert out.shape == (2048, 2048)


# ============================================================
# Section: nvCOMP decompress cumsum offsets
# ============================================================
# Source: test_nvcomp_decompress_cumsum_offsets_1950.py
#
# ``_try_nvcomp_batch_decompress`` used to compute its per-tile host
# prefix-sum offsets via a Python ``for`` loop. The fix swaps in
# ``np.cumsum(sizes, out=offsets[1:])`` to align with the sibling
# batched-D2H helper and the compress-side prefix sum.


def test_nvcomp_decompress_uses_cumsum_for_offsets_1950():
    """Source-level guard against reintroducing the Python for loop."""
    src_path = pathlib.Path(__file__).parent.parent.parent / "_gpu_decode.py"
    src = src_path.read_text()

    cumsum_call = re.compile(
        r"np\.cumsum\(\s*comp_sizes_arr\[:-1\]\s*,\s*"
        r"out\s*=\s*comp_offsets_h\[1:\]\s*\)"
    )
    assert cumsum_call.search(src), (
        "decompress upload block should use "
        "``np.cumsum(comp_sizes_arr[:-1], out=comp_offsets_h[1:])`` for "
        "prefix-sum offsets, aligning with _batched_d2h_to_bytes "
        "(issue #1950)."
    )
    legacy_loop = re.compile(
        r"for\s+i\s+in\s+range\(\s*1\s*,\s*n_tiles\s*\)\s*:\s*\n"
        r"\s*comp_offsets_h\[i\]"
    )
    assert not legacy_loop.search(src), (
        "decompress upload block should no longer compute prefix-sum "
        "offsets with a Python for loop (issue #1950)."
    )


def test_cumsum_matches_loop_prefix_sum_1950():
    """Equivalence between the vectorised cumsum and the prior loop."""
    rng = np.random.RandomState(1950)
    n = 1024
    sizes = rng.randint(100, 100_000, size=n).astype(np.int64)

    offsets_cumsum = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(sizes[:-1], out=offsets_cumsum[1:])

    offsets_loop = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        offsets_loop[i] = offsets_loop[i - 1] + sizes[i - 1]

    np.testing.assert_array_equal(offsets_cumsum, offsets_loop)


@pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None,
    reason="cupy required for nvCOMP path",
)
def test_nvcomp_batch_decompress_roundtrip_1950():
    """End-to-end check: a deflate-tiled raster still decodes correctly."""
    if os.environ.get("XRSPATIAL_GEOTIFF_STRICT_GPU") != "1":
        pytest.skip(
            "set XRSPATIAL_GEOTIFF_STRICT_GPU=1 to exercise the nvCOMP "
            "prefix-sum site; without it the GPU path may fall back to "
            "a CPU codec and bypass this regression."
        )
    try:
        import cupy
    except ImportError:
        pytest.skip("cupy not importable")
    if not cupy.cuda.is_available():
        pytest.skip("CUDA device not available")

    from xrspatial.geotiff import open_geotiff, to_geotiff

    rng = np.random.RandomState(1950)
    height, width = 1024, 1024
    arr = rng.rand(height, width).astype(np.float32)
    da = xr.DataArray(
        arr, dims=["y", "x"],
        coords={"y": np.arange(height), "x": np.arange(width)},
        attrs={"crs": 4326},
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tmp_1950_deflate.tif")
        to_geotiff(da, path, compression="deflate", tile_size=256)

        result = open_geotiff(path, gpu=True)
        assert result.shape == (height, width)
        decoded = cupy.asnumpy(result.data) if hasattr(
            result.data, "get") else np.asarray(result.data)

    np.testing.assert_allclose(decoded, arr, atol=0, rtol=0)


# ============================================================
# Section: nvCOMP from-device-bufs single alloc
# ============================================================
# Source: test_nvcomp_from_device_bufs_single_alloc_1659.py
#
# ``_try_nvcomp_from_device_bufs`` used to allocate N separate
# ``cupy.empty(tile_bytes)`` output buffers and run ``cupy.concatenate``
# after the nvCOMP decompress kernel returned. The fix matches the
# single-contiguous-buffer + pointer-offset pattern.

from xrspatial.geotiff._gpu_decode import _try_nvcomp_from_device_bufs  # noqa: E402


def _nvcomp_available_1659() -> bool:
    from xrspatial.geotiff._gpu_decode import _get_nvcomp
    return _get_nvcomp() is not None


@requires_gpu
def test_unsupported_codec_short_circuits_before_allocation_1659():
    """Non-ZSTD codecs must return None without allocating output buffers."""
    import cupy

    d_tiles = [cupy.zeros(1024, dtype=cupy.uint8) for _ in range(4)]
    assert _try_nvcomp_from_device_bufs(d_tiles, 1024, 8) is None


@requires_gpu
def test_no_nvcomp_lib_returns_none_1659(monkeypatch):
    """When the nvCOMP library is missing, the function must return None."""
    import cupy

    from xrspatial.geotiff import _gpu_decode

    monkeypatch.setattr(_gpu_decode, "_get_nvcomp", lambda: None)

    d_tiles = [cupy.zeros(1024, dtype=cupy.uint8)]
    assert _try_nvcomp_from_device_bufs(d_tiles, 1024, 50000) is None


@requires_gpu
def test_memory_guard_runs_with_full_decomp_size_1659(monkeypatch):
    """The single-buffer allocation must be size-checked before cupy.empty."""
    import cupy

    from xrspatial.geotiff import _gpu_decode

    seen = {"total_bytes": None, "what": None, "called": False}

    def fake_check(required_bytes, what="tile buffer"):
        seen["total_bytes"] = int(required_bytes)
        seen["what"] = what
        seen["called"] = True
        raise MemoryError("simulated OOM")

    monkeypatch.setattr(_gpu_decode, "_get_nvcomp", lambda: object())
    monkeypatch.setattr(_gpu_decode, "_check_gpu_memory", fake_check)

    n_tiles = 8
    tile_bytes = 65536
    d_tiles = [cupy.zeros(128, dtype=cupy.uint8) for _ in range(n_tiles)]

    with pytest.raises(MemoryError):
        _try_nvcomp_from_device_bufs(d_tiles, tile_bytes, 50000)

    assert seen["called"], "_check_gpu_memory was not called"
    expected_bytes = n_tiles * tile_bytes
    assert seen["total_bytes"] == expected_bytes, (
        f"expected total {expected_bytes}, got {seen['total_bytes']}"
    )
    assert "decompressed" in seen["what"] or "nvCOMP" in seen["what"], (
        f"unhelpful 'what' label: {seen['what']!r}"
    )


@pytest.mark.skipif(
    not _HAS_GPU or not _nvcomp_available_1659(),
    reason="cupy + CUDA + nvCOMP shared lib required",
)
def test_zstd_decompress_roundtrip_returns_single_contiguous_buffer_1659():
    """End-to-end: feed real ZSTD-compressed device buffers in."""
    import cupy
    import zstandard as zstd

    rng = np.random.default_rng(seed=1659)
    tile_bytes = 4096
    n_tiles = 8

    cctx = zstd.ZstdCompressor()
    host_tiles = [rng.integers(0, 256, size=tile_bytes, dtype=np.uint8)
                  for _ in range(n_tiles)]
    compressed = [cctx.compress(t.tobytes()) for t in host_tiles]
    d_tiles = [cupy.asarray(np.frombuffer(c, dtype=np.uint8))
               for c in compressed]

    result = _try_nvcomp_from_device_bufs(d_tiles, tile_bytes, 50000)

    if result is None:
        pytest.skip("nvCOMP returned None; library may be unusable on this host")

    assert isinstance(result, cupy.ndarray)
    assert result.dtype == cupy.uint8
    assert result.shape == (n_tiles * tile_bytes,)
    assert result.flags.c_contiguous

    host_out = result.get()
    for i, expected in enumerate(host_tiles):
        decoded = host_out[i * tile_bytes:(i + 1) * tile_bytes]
        assert np.array_equal(decoded, expected), (
            f"tile {i} decoded payload differs from input"
        )


@requires_gpu
def test_no_orphan_decomp_buffers_after_call_1659(monkeypatch):
    """A successful call returns a single contiguous buffer."""
    import cupy

    from xrspatial.geotiff import _gpu_decode

    monkeypatch.setattr(_gpu_decode, "_get_nvcomp",
                        lambda: _FakeNvcompLib_1659())

    n_tiles = 4
    tile_bytes = 2048
    d_tiles = [cupy.zeros(64, dtype=cupy.uint8) for _ in range(n_tiles)]
    result = _try_nvcomp_from_device_bufs(d_tiles, tile_bytes, 50000)

    assert result is not None
    assert isinstance(result, cupy.ndarray)
    assert result.size == n_tiles * tile_bytes
    assert result.flags.c_contiguous
    assert result.dtype == cupy.uint8


class _FakeNvcompLib_1659:
    """Stand-in for the nvCOMP CDLL handle used in tests."""

    def __getattr__(self, name):
        if name == 'nvcompBatchedZstdDecompressGetTempSizeAsync':
            return _fake_temp_size_fn_1659
        if name == 'nvcompBatchedZstdDecompressAsync':
            return _fake_decompress_fn_1659
        raise AttributeError(name)


def _fake_temp_size_fn_1659(n, tile_bytes, opts, p_temp_size, total):
    """Stub for nvcompBatchedZstdDecompressGetTempSizeAsync."""
    p_temp_size._obj.value = 1
    return 0


def _fake_decompress_fn_1659(*args):
    """Stub for nvcompBatchedZstdDecompressAsync (success)."""
    return 0


# ============================================================
# Section: nvJPEG encode stream-null sync
# ============================================================
# Source: test_nvjpeg_encode_stream_sync_2212.py
#
# Replace ``Device().synchronize()`` inside the per-tile encode loops
# in ``_nvjpeg_batch_encode`` and ``_nvjpeg2k_batch_encode`` with
# ``Stream.null.synchronize()`` so the per-tile sync is scoped to the
# default stream rather than the whole device.


def _function_source_2212(func):
    src = inspect.getsource(func)
    start_line = func.__code__.co_firstlineno
    return src, start_line


def _parent_map_2212(tree: ast.AST) -> dict:
    mapping: dict = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            mapping[id(child)] = parent
    return mapping


def _inside_for_loop_2212(node: ast.AST, parents: dict) -> bool:
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.For):
            return True
        cur = parents.get(id(cur))
    return False


def _device_synchronize_lines_2212(tree: ast.AST, start_line: int,
                                   parents: dict, *, only_in_loop: bool):
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != 'synchronize':
            continue
        parent_call = func.value
        if not isinstance(parent_call, ast.Call):
            continue
        if not isinstance(parent_call.func, ast.Attribute):
            continue
        if parent_call.func.attr != 'Device':
            continue
        if only_in_loop and not _inside_for_loop_2212(node, parents):
            continue
        if not only_in_loop and _inside_for_loop_2212(node, parents):
            continue
        out.append(start_line + node.lineno - 1)
    return out


def _stream_null_synchronize_lines_2212(tree: ast.AST, start_line: int,
                                        parents: dict, *, only_in_loop: bool):
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != 'synchronize':
            continue
        chain = func.value
        if isinstance(chain, ast.Call):
            continue
        if not isinstance(chain, ast.Attribute):
            continue
        found_stream_null = False
        cur = chain
        if cur.attr == 'null':
            inner = cur.value
            if isinstance(inner, ast.Attribute) and inner.attr == 'Stream':
                found_stream_null = True
        if not found_stream_null:
            continue
        if only_in_loop and not _inside_for_loop_2212(node, parents):
            continue
        if not only_in_loop and _inside_for_loop_2212(node, parents):
            continue
        out.append(start_line + node.lineno - 1)
    return out


class TestNvjpegEncodeStreamSync_2212:
    """Structural assertions on the encoder sync fix (no GPU required)."""

    def setup_method(self):
        from xrspatial.geotiff import _gpu_decode
        self._fn = _gpu_decode._nvjpeg_batch_encode
        src, start = _function_source_2212(self._fn)
        self._src = src
        self._start_line = start
        self._tree = ast.parse(src)
        self._parents = _parent_map_2212(self._tree)

    def test_no_device_synchronize_inside_encode_loop(self):
        offending = _device_synchronize_lines_2212(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert offending == [], (
            "_nvjpeg_batch_encode contains cupy.cuda.Device().synchronize() "
            f"calls inside a for-loop at file lines {offending}. The fix "
            "in #2212 scopes the per-tile sync to the default stream via "
            "cupy.cuda.Stream.null.synchronize()."
        )

    def test_stream_null_synchronize_present(self):
        found = _stream_null_synchronize_lines_2212(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert len(found) >= 1, (
            "_nvjpeg_batch_encode no longer calls "
            "cupy.cuda.Stream.null.synchronize() inside the encode loop."
        )


class TestNvjpeg2kEncodeStreamSync_2212:
    """Structural assertions on the nvJPEG2000 encoder sync fix."""

    def setup_method(self):
        from xrspatial.geotiff import _gpu_decode
        self._fn = _gpu_decode._nvjpeg2k_batch_encode
        src, start = _function_source_2212(self._fn)
        self._src = src
        self._start_line = start
        self._tree = ast.parse(src)
        self._parents = _parent_map_2212(self._tree)

    def test_no_device_synchronize_inside_encode_loop(self):
        offending = _device_synchronize_lines_2212(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert offending == [], (
            "_nvjpeg2k_batch_encode contains Device().synchronize() inside "
            f"a for-loop at file lines {offending}. The fix in #2212 "
            "requires Stream.null.synchronize()."
        )

    def test_stream_null_synchronize_present(self):
        found = _stream_null_synchronize_lines_2212(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert len(found) >= 1, (
            "_nvjpeg2k_batch_encode no longer calls "
            "Stream.null.synchronize() inside the encode loop."
        )


class TestDecodeReferencePattern_2212:
    """The decoder pattern is the contract we mirror. Pin it as the reference."""

    def setup_method(self):
        from xrspatial.geotiff import _gpu_decode
        self._fn = _gpu_decode._try_nvjpeg_batch_decode
        src, start = _function_source_2212(self._fn)
        self._src = src
        self._start_line = start
        self._tree = ast.parse(src)
        self._parents = _parent_map_2212(self._tree)

    def test_decoder_uses_stream_null_sync_in_loop(self):
        found = _stream_null_synchronize_lines_2212(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert len(found) >= 1, (
            "_try_nvjpeg_batch_decode no longer uses "
            "Stream.null.synchronize() inside the decode loop."
        )


# ============================================================
# Section: nvJPEG2000 single-alloc pool
# ============================================================
# Source: test_nvjpeg2k_single_alloc_2107.py
#
# Replace per-tile / per-component ``cupy.empty`` allocations and per-
# tile ``Device().synchronize()`` inside the decode loop with a single
# contiguous device pool and a single batch-end sync.


def _function_source_2107(func):
    src = inspect.getsource(func)
    start_line = func.__code__.co_firstlineno
    return src, start_line


def _inside_for_loop_2107(node: ast.AST, parents: dict) -> bool:
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.For):
            return True
        cur = parents.get(id(cur))
    return False


def _parent_map_2107(tree: ast.AST) -> dict:
    mapping: dict = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            mapping[id(child)] = parent
    return mapping


class TestNvjpeg2kSingleAllocStructural_2107:
    """Structural assertions on the refactored helper (no GPU required)."""

    def setup_method(self):
        from xrspatial.geotiff import _gpu_decode

        self._fn = _gpu_decode._try_nvjpeg2k_batch_decode
        src, start = _function_source_2107(self._fn)
        self._src = src
        self._start_line = start
        self._tree = ast.parse(src)
        self._parents = _parent_map_2107(self._tree)

    def test_no_cupy_empty_inside_decode_loop(self):
        """``cupy.empty`` must NOT appear inside the per-tile ``for`` loop."""
        offending = []
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != 'empty':
                continue
            if (not isinstance(func.value, ast.Name)
                    or func.value.id not in ('cupy', 'cp')):
                continue
            if _inside_for_loop_2107(node, self._parents):
                offending.append(self._start_line + node.lineno - 1)
        assert offending == [], (
            f"_try_nvjpeg2k_batch_decode contains cupy.empty(...) calls "
            f"inside a for-loop at file lines {offending}. The refactor "
            f"in #2107 moved every output allocation outside the per-tile "
            f"loop."
        )

    def test_no_device_synchronize_inside_decode_loop(self):
        """``Device().synchronize()`` must NOT live inside the decode loop."""
        offending = []
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != 'synchronize':
                continue
            parent_call = func.value
            if (not isinstance(parent_call, ast.Call)
                    or not isinstance(parent_call.func, ast.Attribute)
                    or parent_call.func.attr != 'Device'):
                continue
            if _inside_for_loop_2107(node, self._parents):
                offending.append(self._start_line + node.lineno - 1)
        assert offending == [], (
            f"_try_nvjpeg2k_batch_decode contains Device().synchronize() "
            f"calls inside a for-loop at file lines {offending}. The "
            f"refactor in #2107 keeps exactly one batch-end sync outside "
            f"the loop."
        )

    def test_pool_allocation_present(self):
        """Source contains the expected pool buffer name and slab math."""
        assert 'd_comp_pool' in self._src, (
            "_try_nvjpeg2k_batch_decode no longer references the shared "
            "d_comp_pool buffer; the refactor in #2107 is missing or "
            "reverted."
        )
        assert 'per_tile_comp_bytes' in self._src, (
            "_try_nvjpeg2k_batch_decode no longer references "
            "per_tile_comp_bytes."
        )

    def test_check_gpu_memory_guard_present(self):
        """The pool allocation must be guarded by ``_check_gpu_memory``."""
        assert '_check_gpu_memory(' in self._src, (
            "_try_nvjpeg2k_batch_decode no longer calls _check_gpu_memory."
        )


class TestNvjpeg2kLibAbsentShortCircuit_2107:
    """When the shared library is missing, the function returns None."""

    def test_returns_none_when_lib_missing(self, monkeypatch):
        from xrspatial.geotiff import _gpu_decode

        monkeypatch.setattr(_gpu_decode, '_get_nvjpeg2k', lambda: None)

        result = _gpu_decode._try_nvjpeg2k_batch_decode(
            compressed_tiles=[b''],
            tile_width=8,
            tile_height=8,
            dtype=np.dtype('uint8'),
            samples=1,
        )
        assert result is None

    def test_returns_none_for_unsupported_dtype(self, monkeypatch):
        """Unsupported dtypes short-circuit before any device allocation."""
        from xrspatial.geotiff import _gpu_decode

        class _FakeLib:
            def __init__(self):
                self.calls = []

            def nvjpeg2kCreateSimple(self, *_args):
                return 0

            def nvjpeg2kDecodeStateCreate(self, *_args):
                return 0

            def nvjpeg2kStreamCreate(self, *_args):
                return 0

            def nvjpeg2kDecodeParamsCreate(self, *_args):
                return 0

            def nvjpeg2kDecodeParamsDestroy(self, *_args):
                self.calls.append('params_destroy')

            def nvjpeg2kStreamDestroy(self, *_args):
                self.calls.append('stream_destroy')

            def nvjpeg2kDecodeStateDestroy(self, *_args):
                self.calls.append('state_destroy')

            def nvjpeg2kDestroy(self, *_args):
                self.calls.append('handle_destroy')

        fake = _FakeLib()
        monkeypatch.setattr(_gpu_decode, '_get_nvjpeg2k', lambda: fake)

        result = _gpu_decode._try_nvjpeg2k_batch_decode(
            compressed_tiles=[b''],
            tile_width=8,
            tile_height=8,
            dtype=np.dtype('float32'),
            samples=1,
        )
        assert result is None
        assert fake.calls == [
            'params_destroy',
            'stream_destroy',
            'state_destroy',
            'handle_destroy',
        ]


@requires_gpu
class TestNvjpeg2kPoolWithCupy_2107:
    """Lightweight cupy-only smoke tests for the pool layout."""

    def test_pool_slabs_are_non_overlapping(self):
        """Tile-component slabs into the pool must not overlap."""
        cupy = pytest.importorskip('cupy')

        n_tiles = 4
        tile_width = 32
        tile_height = 32
        samples = 3
        dtype = np.dtype('uint16')
        pitch = tile_width * dtype.itemsize
        per_tile_comp_bytes = samples * tile_height * pitch
        pool = cupy.empty(n_tiles * per_tile_comp_bytes, dtype=cupy.uint8)

        seen = set()
        for i in range(n_tiles):
            tile_pool_start = i * per_tile_comp_bytes
            for c in range(samples):
                start = tile_pool_start + c * tile_height * pitch
                end = start + tile_height * pitch
                for byte in range(start, end):
                    assert byte not in seen, (
                        f"pool byte {byte} appears in two slabs "
                        f"(tile={i}, comp={c}); per-tile slab math is "
                        f"wrong."
                    )
                    seen.add(byte)
        assert len(seen) == int(pool.nbytes)


# ============================================================
# Section: nvJPEG output-format constants
# ============================================================
# Source: test_jpeg_gpu_1549.py
#
# Off-by-two on the ``nvjpegOutputFormat_t`` constants in
# ``_gpu_decode.py`` caused ``cudaErrorIllegalAddress`` on 3-band JPEG
# TIFFs and silently-wrong pixels on single-band JPEG TIFFs.


def _nvjpeg_available_1549() -> bool:
    """True when libnvjpeg.so loads on this host."""
    if not _HAS_GPU:
        return False
    try:
        from xrspatial.geotiff._gpu_decode import _get_nvjpeg
        return _get_nvjpeg() is not None
    except Exception:
        return False


_HAS_NVJPEG_1549 = _nvjpeg_available_1549()

_gpu_only_1549 = pytest.mark.skipif(
    not (_HAS_GPU and _HAS_TIFFFILE and _HAS_PIL
         and _HAS_IMAGECODECS and _HAS_NVJPEG_1549),
    reason="cupy + CUDA + tifffile + Pillow + imagecodecs + nvJPEG required",
)


def _write_jpeg_rgb_tiff_1549(path: str, seed: int = 0,
                              noise: bool = True) -> np.ndarray:
    """Write a 3-band 256x256 tiled JPEG TIFF using tifffile."""
    import tifffile
    if noise:
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    else:
        ys, xs = np.mgrid[0:256, 0:256].astype(np.int32)
        r = (ys + xs) // 2
        g = ys
        b = xs
        arr = np.stack([r, g, b], axis=2).clip(0, 255).astype(np.uint8)
    tifffile.imwrite(path, arr, photometric='rgb', tile=(128, 128),
                     compression='jpeg')
    return arr


def _write_jpeg_gray_tiff_1549(path: str, seed: int = 42) -> np.ndarray:
    """Write a 1-band 256x256 tiled JPEG TIFF using tifffile."""
    import tifffile
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    tifffile.imwrite(path, arr, photometric='minisblack', tile=(128, 128),
                     compression='jpeg')
    return arr


@_gpu_only_1549
def test_rgb_jpeg_gpu_no_crash_1549(tmp_path, monkeypatch):
    """3-band JPEG must not raise CUDARuntimeError on GPU read."""
    import cupy

    from xrspatial.geotiff import _gpu_decode, read_geotiff_gpu

    spy = {"calls": 0, "successes": 0}
    original = _gpu_decode._try_nvjpeg_batch_decode

    def wrapped(*args, **kwargs):
        spy["calls"] += 1
        result = original(*args, **kwargs)
        if result is not None:
            spy["successes"] += 1
        return result

    monkeypatch.setattr(_gpu_decode, "_try_nvjpeg_batch_decode", wrapped)

    path = str(tmp_path / "rgb_jpeg_1549.tif")
    _write_jpeg_rgb_tiff_1549(path)

    arr = read_geotiff_gpu(path, gpu='strict', allow_internal_only_jpeg=True)
    assert isinstance(arr.data, cupy.ndarray)
    decoded = arr.data.get()
    assert decoded.shape == (256, 256, 3)
    assert decoded.dtype == np.uint8

    assert spy["calls"] >= 1, (
        "nvJPEG branch was never called -- test did not exercise the "
        "code path the #1549 fix lives on"
    )
    assert spy["successes"] >= 1, (
        "nvJPEG returned None -- CPU Pillow fallback ran and the fix was "
        "not exercised"
    )


@_gpu_only_1549
def test_rgb_jpeg_gpu_matches_cpu_1549(tmp_path):
    """GPU pixels must be within JPEG decoder tolerance of CPU pixels."""
    from xrspatial.geotiff import open_geotiff

    path = str(tmp_path / "rgb_jpeg_match_1549.tif")
    _write_jpeg_rgb_tiff_1549(path, noise=False)

    cpu = open_geotiff(path, allow_internal_only_jpeg=True)
    gpu = open_geotiff(path, gpu=True, allow_internal_only_jpeg=True)
    assert cpu.shape == gpu.shape == (256, 256, 3)

    cpu_arr = np.asarray(cpu.data)
    gpu_arr = np.asarray(gpu.data.get())

    diff = np.abs(cpu_arr.astype(int) - gpu_arr.astype(int))
    assert diff.mean() < 1.0, f"mean diff {diff.mean():.3f} too large"
    assert diff.max() < 8, f"max diff {diff.max()} too large"


@_gpu_only_1549
def test_grayscale_jpeg_gpu_matches_cpu_1549(tmp_path):
    """Single-band JPEG GPU read must also produce correct pixels."""
    from xrspatial.geotiff import open_geotiff

    path = str(tmp_path / "gray_jpeg_1549.tif")
    _write_jpeg_gray_tiff_1549(path)

    cpu = open_geotiff(path, allow_internal_only_jpeg=True)
    gpu = open_geotiff(path, gpu=True, allow_internal_only_jpeg=True)
    assert cpu.shape == gpu.shape == (256, 256)

    cpu_arr = np.asarray(cpu.data)
    gpu_arr = np.asarray(gpu.data.get())
    diff = np.abs(cpu_arr.astype(int) - gpu_arr.astype(int))
    assert diff.max() <= 2, (
        f"grayscale max diff {diff.max()} indicates corruption, "
        f"not just rounding"
    )


@_gpu_only_1549
def test_cuda_context_survives_after_jpeg_gpu_read_1549(tmp_path):
    """Verify the CUDA context is healthy after a GPU JPEG read."""
    import cupy

    from xrspatial.geotiff import open_geotiff

    path = str(tmp_path / "rgb_ctx_1549.tif")
    _write_jpeg_rgb_tiff_1549(path)

    arr = open_geotiff(path, gpu=True, allow_internal_only_jpeg=True)
    _ = arr.data.get()

    x = cupy.arange(1024, dtype=cupy.float32)
    s = float(cupy.sum(x).item())
    assert s == 1023 * 1024 / 2

    other_path = str(tmp_path / "other_1549.tif")
    _write_jpeg_gray_tiff_1549(other_path, seed=7)
    other = open_geotiff(other_path, gpu=True, allow_internal_only_jpeg=True)
    assert other.shape == (256, 256)
    assert other.dtype == np.uint8


# ============================================================
# Section: LERC valid-mask GPU
# ============================================================
# Source: test_lerc_valid_mask_gpu.py
#
# The CPU LERC reader honours the LERC valid-mask. The GPU LERC tile-
# decode path used to discard the mask. These tests confirm the GPU
# path now matches the CPU path for representative mask combinations.

# Module-level skip: this whole section is LERC-only.
lerc_lerc = pytest.importorskip("lerc", reason="lerc required for LERC GPU tests")

from xrspatial.geotiff._compression import LERC_AVAILABLE  # noqa: E402

_gpu_only_lerc = pytest.mark.skipif(
    not (_HAS_GPU and LERC_AVAILABLE),
    reason="cupy + CUDA + lerc required",
)


@pytest.fixture
def lerc_writer_with_mask_gpu(monkeypatch):
    """Patch ``lerc_compress`` to embed a valid-mask the writer can't pass."""
    holder = {"invalid": None}

    def _patched(data, width, height, samples=1,
                 dtype=np.dtype('float32'), max_z_error=0.0):
        if samples == 1:
            arr = np.frombuffer(data, dtype=dtype).reshape(height, width)
        else:
            arr = np.frombuffer(data, dtype=dtype).reshape(
                height, width, samples)
        invalid_pred = holder["invalid"]
        if invalid_pred is None:
            mask = None
            has_mask = False
        else:
            invalid = invalid_pred(arr)
            mask = np.where(invalid, np.uint8(0), np.uint8(1))
            has_mask = True
        result = lerc_lerc.encode(
            arr, samples, has_mask, mask, max_z_error, 1,
        )
        if result[0] != 0:
            raise RuntimeError(
                f"LERC encode failed with error code {result[0]}")
        return bytes(result[2])

    monkeypatch.setattr(
        "xrspatial.geotiff._compression.lerc_compress", _patched,
    )
    return holder


def _read_cpu_gpu_lerc(path):
    """Read *path* with both readers and return ``(cpu_array, gpu_host_array)``."""
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    cpu, _geo = read_to_array(path, allow_experimental_codecs=True)
    gpu_da = read_geotiff_gpu(
        path, gpu='strict', allow_experimental_codecs=True,
    )
    gpu_host = gpu_da.data.get()
    return cpu, gpu_host


def _restore_sentinel_lerc(arr, nodata):
    """Replace NaN positions in *arr* with *nodata* for bit-exact compare."""
    if nodata is None or arr.dtype.kind != 'f' or np.isnan(nodata):
        return arr
    out = arr.copy()
    out[np.isnan(out)] = arr.dtype.type(nodata)
    return out


@_gpu_only_lerc
class TestGpuLercValidMask:
    """End-to-end TIFF round-trips comparing GPU vs CPU output."""

    def test_float32_nan_nodata(self, tmp_path, lerc_writer_with_mask_gpu):
        """Float32 LERC + NaN nodata: GPU output matches CPU output."""
        from xrspatial.geotiff._writer import write

        arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        invalid_positions = {(0, 1), (5, 4)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask_gpu["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_nan_gpu.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=float("nan"))

        cpu, gpu = _read_cpu_gpu_lerc(path)
        for (r, c) in invalid_positions:
            assert np.isnan(cpu[r, c])
            assert np.isnan(gpu[r, c])
        cpu_valid = np.where(np.isnan(cpu), 0.0, cpu)
        gpu_valid = np.where(np.isnan(gpu), 0.0, gpu)
        np.testing.assert_array_equal(cpu_valid, gpu_valid)

    def test_float32_sentinel_nodata(self, tmp_path, lerc_writer_with_mask_gpu):
        """Float32 LERC + sentinel nodata (-9999): GPU matches CPU."""
        from xrspatial.geotiff._writer import write

        arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        invalid_positions = {(0, 1), (3, 3), (7, 7)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask_gpu["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_sentinel_f32_gpu.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=-9999.0)

        cpu, gpu = _read_cpu_gpu_lerc(path)
        gpu_with_sentinel = _restore_sentinel_lerc(gpu, -9999.0)
        np.testing.assert_array_equal(cpu, gpu_with_sentinel)
        for (r, c) in invalid_positions:
            assert np.isnan(gpu[r, c])
            assert gpu_with_sentinel[r, c] == np.float32(-9999.0)

    def test_uint16_sentinel_nodata(self, tmp_path, lerc_writer_with_mask_gpu):
        """Uint16 LERC + sentinel nodata (65535): GPU matches CPU."""
        from xrspatial.geotiff._writer import write

        arr = (np.arange(1, 65, dtype=np.uint16) * 100).reshape(8, 8)
        invalid_positions = {(0, 1), (4, 4)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask_gpu["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_uint16_gpu.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=65535)

        cpu, gpu = _read_cpu_gpu_lerc(path)
        assert gpu.dtype == np.float64
        gpu_no_nan = np.where(np.isnan(gpu), 65535.0, gpu)
        gpu_u16 = gpu_no_nan.astype(np.uint16)
        np.testing.assert_array_equal(cpu, gpu_u16)
        for (r, c) in invalid_positions:
            assert np.isnan(gpu[r, c])
            assert gpu_u16[r, c] == np.uint16(65535)

    def test_no_mask_roundtrip_bitexact(self, tmp_path):
        """All-valid LERC (no encoded mask): GPU and CPU agree bit-exact."""
        from xrspatial.geotiff._writer import write

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / "lerc_no_mask_gpu.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8)

        cpu, gpu = _read_cpu_gpu_lerc(path)
        np.testing.assert_array_equal(cpu, arr)
        np.testing.assert_array_equal(gpu, arr)


# ============================================================
# Section: predictor=2 big-endian GPU
# ============================================================
# Source: test_predictor2_big_endian_gpu_1517.py
#
# Predictor=2 BE files used to come back with wrong values on the GPU
# tiled path. The per-dtype predictor kernels now byte-swap the buffer
# before running the prefix-sum.

_gpu_only_1517 = pytest.mark.skipif(
    not (_HAS_GPU and _HAS_TIFFFILE),
    reason="cupy + CUDA + tifffile required",
)


def _block_cpu_fallback_1517(monkeypatch):
    """Make any call to ``read_to_array`` from ``read_geotiff_gpu`` fail loudly."""
    from xrspatial.geotiff._backends import gpu as gpu_backend

    def _no_fallback(*args, **kwargs):
        raise AssertionError(
            "read_geotiff_gpu fell back to read_to_array; "
            "the GPU decode path was not exercised."
        )

    monkeypatch.setattr(
        gpu_backend, '_read_to_array', _no_fallback, raising=True,
    )


@_gpu_only_1517
def test_gpu_predictor2_big_endian_int32_tiled_reproducer_1517(tmp_path, monkeypatch):
    """Exact reproducer from issue #1517: BE int32 tiled deflate + pred=2."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260507)
    arr = rng.randint(
        -1_000_000, 1_000_000, size=(32, 48), dtype=np.int64
    ).astype(np.int32)

    path = tmp_path / "be_pred2_int32.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2,
        compression="deflate", tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    _block_cpu_fallback_1517(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(np.int32)
    assert gpu_da.data.dtype.isnative
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only_1517
@pytest.mark.parametrize(
    "dtype",
    [np.uint16, np.int16, np.uint32, np.int32],
)
def test_gpu_predictor2_big_endian_dtypes_tiled_1517(tmp_path, monkeypatch, dtype):
    """BE predictor=2 tiled files match CPU baseline across dtypes."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260508)
    info = np.iinfo(dtype)
    arr = rng.randint(
        max(info.min, -1_000_000),
        min(info.max, 1_000_000),
        size=(32, 48),
        dtype=np.int64,
    ).astype(dtype)

    path = tmp_path / f"be_pred2_{np.dtype(dtype).name}.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2,
        compression="deflate", tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    _block_cpu_fallback_1517(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(dtype)
    assert gpu_da.data.dtype.isnative
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only_1517
def test_gpu_predictor2_big_endian_stripped_uint16_1517(tmp_path):
    """Stripped BE predictor=2 files take the CPU fallback but stay correct."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260509)
    arr = rng.randint(0, 60000, size=(32, 48), dtype=np.uint16)

    path = tmp_path / "be_pred2_uint16_strip.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2, compression="deflate",
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(np.uint16)
    assert gpu_da.data.dtype.isnative
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only_1517
def test_gpu_predictor2_little_endian_still_works_1517(tmp_path, monkeypatch):
    """LE predictor=2 must still round-trip after the BE fix."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260510)
    arr = rng.randint(
        -1_000_000, 1_000_000, size=(32, 48), dtype=np.int64
    ).astype(np.int32)

    path = tmp_path / "le_pred2_int32.tif"
    tifffile.imwrite(
        str(path), arr, byteorder="<", predictor=2,
        compression="deflate", tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    _block_cpu_fallback_1517(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only_1517
def test_gpu_predictor3_big_endian_still_works_1517(tmp_path, monkeypatch):
    """Floating-point predictor BE must still match CPU after the fix."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260511)
    arr = rng.standard_normal((32, 48)).astype(np.float32)

    path = tmp_path / "be_pred3_float32.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=3,
        compression="deflate", tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    _block_cpu_fallback_1517(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


def test_swap_byte_lanes_numpy_bps2_1517():
    """The byte-swap helper reverses bytes per sample on a numpy buffer."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.array([0x01, 0x02, 0x03, 0x04], dtype=np.uint8)
    _swap_byte_lanes(buf, 2)
    np.testing.assert_array_equal(buf, np.array([0x02, 0x01, 0x04, 0x03],
                                                dtype=np.uint8))


def test_swap_byte_lanes_numpy_bps4_1517():
    """bps=4: full byte reversal within each 4-byte sample."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.array([0x01, 0x02, 0x03, 0x04,
                    0x05, 0x06, 0x07, 0x08], dtype=np.uint8)
    _swap_byte_lanes(buf, 4)
    np.testing.assert_array_equal(
        buf, np.array([0x04, 0x03, 0x02, 0x01,
                       0x08, 0x07, 0x06, 0x05], dtype=np.uint8))


def test_swap_byte_lanes_numpy_bps8_1517():
    """bps=8: full byte reversal within each 8-byte sample."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    sample = np.arange(1, 9, dtype=np.uint8)
    buf = np.tile(sample, 2).copy()
    _swap_byte_lanes(buf, 8)
    np.testing.assert_array_equal(
        buf, np.tile(sample[::-1], 2))


def test_swap_byte_lanes_uint8_noop_1517():
    """bps=1 must be a no-op."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.array([1, 2, 3], dtype=np.uint8)
    _swap_byte_lanes(buf, 1)
    np.testing.assert_array_equal(buf, np.array([1, 2, 3], dtype=np.uint8))


def test_swap_byte_lanes_rejects_unsupported_bps_1517():
    """Unsupported bps values raise ValueError rather than corrupt data."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.zeros(6, dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported bps"):
        _swap_byte_lanes(buf, 3)


def test_swap_byte_lanes_rejects_misaligned_size_1517():
    """Buffer size must be a multiple of bps."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.zeros(5, dtype=np.uint8)
    with pytest.raises(ValueError, match="not a multiple"):
        _swap_byte_lanes(buf, 2)


def test_swap_byte_lanes_numpy_is_zero_temp_1517():
    """The numpy path must mutate the original buffer without realloc."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.array([0x01, 0x02, 0x03, 0x04], dtype=np.uint8)
    addr_before = buf.ctypes.data
    _swap_byte_lanes(buf, 2)
    assert buf.ctypes.data == addr_before
    np.testing.assert_array_equal(buf, np.array([0x02, 0x01, 0x04, 0x03],
                                                dtype=np.uint8))


@_gpu_only_1517
@pytest.mark.parametrize("bps,dtype", [
    (2, np.uint16),
    (4, np.uint32),
    (8, np.uint64),
])
def test_swap_byte_lanes_cupy_kernel_1517(bps, dtype):
    """The cupy path runs the CUDA kernel and matches numpy.byteswap."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    rng = np.random.RandomState(20260512 + bps)
    n_samples = 1024
    src = rng.randint(0, np.iinfo(dtype).max, size=n_samples,
                      dtype=np.uint64).astype(dtype)
    expected = src.byteswap()

    d_buf = cupy.asarray(src.view(np.uint8))
    addr_before = int(d_buf.data.ptr)
    _swap_byte_lanes(d_buf, bps)
    addr_after = int(d_buf.data.ptr)

    assert addr_after == addr_before, "kernel must operate in place"
    np.testing.assert_array_equal(
        d_buf.get().view(dtype), expected,
    )


@_gpu_only_1517
def test_swap_byte_lanes_cupy_uint8_noop_1517():
    """bps=1 leaves cupy buffers untouched (no kernel launch)."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    src = np.arange(16, dtype=np.uint8)
    d_buf = cupy.asarray(src)
    _swap_byte_lanes(d_buf, 1)
    np.testing.assert_array_equal(d_buf.get(), src)


# ============================================================
# Section: predictor=3 + integer SampleFormat rejection on GPU
# ============================================================
# Source: test_predictor3_int_dtype_gpu_1933.py
#
# ``_validate_predictor_sample_format`` is wired into every IFD-read
# site. This section closes the GPU coverage gap for the two GPU
# validator call sites (tiled eager + GDS chunked).

from xrspatial.geotiff._compression import COMPRESSION_NONE  # noqa: E402
from xrspatial.geotiff._dtypes import LONG, SHORT, numpy_to_tiff_dtype  # noqa: E402
from xrspatial.geotiff._header import (TAG_BITS_PER_SAMPLE, TAG_COMPRESSION,  # noqa: E402
                                       TAG_IMAGE_LENGTH, TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC,
                                       TAG_PREDICTOR, TAG_ROWS_PER_STRIP, TAG_SAMPLE_FORMAT,
                                       TAG_SAMPLES_PER_PIXEL, TAG_STRIP_BYTE_COUNTS,
                                       TAG_STRIP_OFFSETS, TAG_TILE_BYTE_COUNTS, TAG_TILE_LENGTH,
                                       TAG_TILE_OFFSETS, TAG_TILE_WIDTH)
from xrspatial.geotiff._writer import _assemble_standard_layout, _write_stripped  # noqa: E402


def _build_predictor3_uint32_stripped_tiff_1933(arr: np.ndarray) -> bytes:
    """Build a stripped TIFF: predictor=3 + uint32 SampleFormat=1."""
    rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
    bits_per_sample, _ = numpy_to_tiff_dtype(arr.dtype)
    tags = [
        (TAG_IMAGE_WIDTH, LONG, 1, arr.shape[1]),
        (TAG_IMAGE_LENGTH, LONG, 1, arr.shape[0]),
        (TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample),
        (TAG_COMPRESSION, SHORT, 1, COMPRESSION_NONE),
        (TAG_PHOTOMETRIC, SHORT, 1, 1),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
        (TAG_SAMPLE_FORMAT, SHORT, 1, 1),
        (TAG_PREDICTOR, SHORT, 1, 3),
        (TAG_ROWS_PER_STRIP, SHORT, 1, arr.shape[0]),
        (TAG_STRIP_OFFSETS, LONG, len(rel_off), rel_off),
        (TAG_STRIP_BYTE_COUNTS, LONG, len(bc), bc),
    ]
    parts = [(arr, arr.shape[1], arr.shape[0], rel_off, bc, chunks)]
    return _assemble_standard_layout(8, [tags], parts, bigtiff=False)


def _build_predictor3_uint32_tiled_tiff_1933(
    arr: np.ndarray, tile_w: int = 16, tile_h: int = 16,
) -> bytes:
    """Build a tiled malformed TIFF: predictor=3 + uint32 SampleFormat=1."""
    bits_per_sample, _ = numpy_to_tiff_dtype(arr.dtype)
    h, w = arr.shape

    tiles_across = (w + tile_w - 1) // tile_w
    tiles_down = (h + tile_h - 1) // tile_h
    tiles: list[bytes] = []
    rel_off: list[int] = []
    bc: list[int] = []
    offset = 0
    for tr in range(tiles_down):
        for tc in range(tiles_across):
            r0 = tr * tile_h
            c0 = tc * tile_w
            r1 = min(r0 + tile_h, h)
            c1 = min(c0 + tile_w, w)
            tile_slice = arr[r0:r1, c0:c1]
            if tile_slice.shape != (tile_h, tile_w):
                padded = np.zeros((tile_h, tile_w), dtype=arr.dtype)
                padded[: tile_slice.shape[0], : tile_slice.shape[1]] = (
                    tile_slice)
                tile_arr = padded
            else:
                tile_arr = np.ascontiguousarray(tile_slice)
            chunk = tile_arr.tobytes()
            rel_off.append(offset)
            bc.append(len(chunk))
            tiles.append(chunk)
            offset += len(chunk)

    tags = [
        (TAG_IMAGE_WIDTH, LONG, 1, w),
        (TAG_IMAGE_LENGTH, LONG, 1, h),
        (TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample),
        (TAG_COMPRESSION, SHORT, 1, COMPRESSION_NONE),
        (TAG_PHOTOMETRIC, SHORT, 1, 1),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
        (TAG_SAMPLE_FORMAT, SHORT, 1, 1),
        (TAG_PREDICTOR, SHORT, 1, 3),
        (TAG_TILE_WIDTH, LONG, 1, tile_w),
        (TAG_TILE_LENGTH, LONG, 1, tile_h),
        (TAG_TILE_OFFSETS, LONG, len(rel_off), rel_off),
        (TAG_TILE_BYTE_COUNTS, LONG, len(bc), bc),
    ]
    parts = [(arr, w, h, rel_off, bc, tiles)]
    return _assemble_standard_layout(8, [tags], parts, bigtiff=False)


@requires_gpu
class TestGPUEagerRejectsMalformedFile_1933:
    """``read_geotiff_gpu`` rejects predictor=3 + integer SampleFormat."""

    def test_gpu_eager_stripped_raises(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint32)
        path = tmp_path / "pred3_uint32_stripped.tif"
        path.write_bytes(_build_predictor3_uint32_stripped_tiff_1933(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_gpu(str(path))

    def test_gpu_eager_tiled_raises(self, tmp_path):
        """Tiled layout hits the tiled GPU validator at gpu.py:443."""
        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.arange(256, dtype=np.uint32).reshape(16, 16)
        path = tmp_path / "pred3_uint32_tiled.tif"
        path.write_bytes(_build_predictor3_uint32_tiled_tiff_1933(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_gpu(str(path))

    def test_gpu_dispatcher_eager_raises(self, tmp_path):
        """``open_geotiff(gpu=True)`` dispatcher rejects the file."""
        from xrspatial.geotiff import open_geotiff

        arr = np.arange(64, dtype=np.uint32).reshape(8, 8)
        path = tmp_path / "pred3_uint32_dispatch.tif"
        path.write_bytes(_build_predictor3_uint32_stripped_tiff_1933(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            open_geotiff(str(path), gpu=True)


@requires_gpu
class TestGPUChunkedRejectsMalformedFile_1933:
    """The dask+GPU paths also reject predictor=3 + integer."""

    def test_read_geotiff_gpu_chunked_stripped_raises(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.arange(64, dtype=np.uint32).reshape(8, 8)
        path = tmp_path / "pred3_uint32_chunked_str.tif"
        path.write_bytes(_build_predictor3_uint32_stripped_tiff_1933(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_gpu(str(path), chunks=4)

    def test_read_geotiff_gpu_chunked_tiled_raises(self, tmp_path):
        """Tiled chunked path with KvikIO available exercises gpu.py:999."""
        pytest.importorskip("kvikio")

        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.arange(256, dtype=np.uint32).reshape(16, 16)
        path = tmp_path / "pred3_uint32_chunked_tiled.tif"
        path.write_bytes(_build_predictor3_uint32_tiled_tiff_1933(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_gpu(str(path), chunks=16)

    def test_open_geotiff_chunks_gpu_dispatcher_raises(self, tmp_path):
        """``open_geotiff(chunks=, gpu=True)`` dispatcher rejects the file."""
        from xrspatial.geotiff import open_geotiff

        arr = np.arange(256, dtype=np.uint32).reshape(16, 16)
        path = tmp_path / "pred3_uint32_chunked_dispatch.tif"
        path.write_bytes(_build_predictor3_uint32_tiled_tiff_1933(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            open_geotiff(str(path), chunks=8, gpu=True)


@requires_gpu
class TestValidPredictor3StillWorksOnGPU_1933:
    """A legitimate predictor=3 + float32 tiled file still decodes on GPU."""

    def test_predictor3_float32_gpu_round_trip(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_gpu, to_geotiff

        arr = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
        path = tmp_path / "pred3_float32_tiled.tif"
        to_geotiff(
            arr, str(path), compression="deflate", predictor=3,
            tiled=True, tile_size=16,
        )

        result = read_geotiff_gpu(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result.data.get(), arr)

    def test_predictor3_float32_dask_gpu_round_trip(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_gpu, to_geotiff

        arr = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
        path = tmp_path / "pred3_float32_dask.tif"
        to_geotiff(
            arr, str(path), compression="deflate", predictor=3,
            tiled=True, tile_size=16,
        )

        result = read_geotiff_gpu(str(path), chunks=8)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result.compute().data.get(), arr)


@requires_gpu
class TestErrorMessageStable_1933:
    """The GPU error wording matches the eager/dask wording."""

    def test_gpu_error_message_matches_eager(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

        arr = np.arange(64, dtype=np.uint32).reshape(8, 8)
        path = tmp_path / "pred3_uint32_msg.tif"
        path.write_bytes(_build_predictor3_uint32_stripped_tiff_1933(arr))

        with pytest.raises(ValueError) as exc_eager:
            open_geotiff(str(path))
        with pytest.raises(ValueError) as exc_gpu:
            read_geotiff_gpu(str(path))

        assert str(exc_eager.value) == str(exc_gpu.value), (
            "GPU and eager paths must surface the same Predictor=3 "
            "error message so callers can use a single except branch."
        )


# ============================================================
# Section: GPU writer rejects JPEG without opt-in
# ============================================================
# Source: test_gpu_jpeg_interop_reject_issue_D_1845.py
#
# ``write_geotiff_gpu`` mirrors ``to_geotiff`` and rejects
# ``compression='jpeg'`` by default. ``allow_internal_only_jpeg=True``
# opts in and emits ``GeoTIFFFallbackWarning``.

from xrspatial.geotiff import GeoTIFFFallbackWarning, write_geotiff_gpu  # noqa: E402


def _make_rgb_uint8_da_1845() -> xr.DataArray:
    """64x64x3 uint8 RGB raster suitable for the JPEG encode path."""
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return xr.DataArray(
        arr,
        dims=("y", "x", "band"),
        coords={
            "y": np.arange(64, dtype=np.float64),
            "x": np.arange(64, dtype=np.float64),
            "band": np.array([1, 2, 3], dtype=np.int32),
        },
    )


def test_write_geotiff_gpu_rejects_jpeg_without_opt_in_1845(tmp_path):
    """``compression='jpeg'`` without the opt-in raises ``ValueError``."""
    da = _make_rgb_uint8_da_1845()
    path = str(tmp_path / "rejected_issue_D_1845.tif")

    with pytest.raises(ValueError, match="JPEGTables"):
        write_geotiff_gpu(da, path, compression='jpeg')


def test_write_geotiff_gpu_rejects_jpeg_message_mentions_alternatives_1845(tmp_path):
    """The rejection error mentions the same alternative codecs."""
    da = _make_rgb_uint8_da_1845()
    path = str(tmp_path / "rejected_msg_issue_D_1845.tif")

    with pytest.raises(ValueError) as exc:
        write_geotiff_gpu(da, path, compression='jpeg')

    msg = str(exc.value)
    assert "deflate" in msg
    assert "zstd" in msg


def test_write_geotiff_gpu_rejects_jpeg_case_insensitive_1845(tmp_path):
    """Upper-case ``compression='JPEG'`` is rejected too."""
    da = _make_rgb_uint8_da_1845()
    path = str(tmp_path / "rejected_upper_issue_D_1845.tif")

    with pytest.raises(ValueError, match="JPEGTables"):
        write_geotiff_gpu(da, path, compression='JPEG')


@requires_gpu
def test_write_geotiff_gpu_jpeg_opt_in_emits_warning_1845(tmp_path):
    """``allow_internal_only_jpeg=True`` emits ``GeoTIFFFallbackWarning``."""
    da = _make_rgb_uint8_da_1845()
    path = str(tmp_path / "opt_in_issue_D_1845.tif")

    with pytest.warns(GeoTIFFFallbackWarning, match="JPEGTables"):
        write_geotiff_gpu(
            da, path,
            compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


@requires_gpu
def test_write_geotiff_gpu_non_jpeg_unaffected_by_flag_1845(tmp_path):
    """Setting ``allow_internal_only_jpeg=True`` on a non-JPEG codec is a no-op."""
    da = _make_rgb_uint8_da_1845()
    path = str(tmp_path / "non_jpeg_flag_issue_D_1845.tif")

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", GeoTIFFFallbackWarning)
        write_geotiff_gpu(
            da, path,
            compression='zstd',
            allow_internal_only_jpeg=True,
        )
