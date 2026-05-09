"""Regression tests for batched host->device upload in the nvCOMP path.

Performance audit P3: ``_try_nvcomp_batch_decompress`` previously did one
``cupy.asarray`` per compressed tile, costing ~6.07 ms for 256 x 64 KB
tiles. The fix concatenates all tiles into a single host buffer, performs
one H2D transfer, and derives per-tile device pointers via
``base_ptr + offsets`` -- mirroring the pattern at
``_gpu_decode.py`` L1714-1722 in the LZW/Deflate path. Measured ~1.66x
speedup that scales worse with more tiles.

The tests skip cleanly when neither libnvcomp nor kvikio.nvcomp is
available on the host -- without one of those, the GPU decoder falls
back to the per-tile numba kernel and the changed code path is not
exercised, so a passing test would be misleading. When the path *is*
available, the correctness test additionally asserts that
``_try_nvcomp_batch_decompress`` returned a non-None CuPy array, so a
silent fall-through to the slow path counts as a test failure.
"""
from __future__ import annotations

import importlib.util
import time
import uuid

import numpy as np
import pytest


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def _kvikio_nvcomp_importable() -> bool:
    """True iff ``import kvikio.nvcomp`` actually succeeds.

    ``importlib.util.find_spec`` may report kvikio as installed even when
    the underlying ``libkvikio.so`` is missing, so we attempt the real
    import here.
    """
    try:
        import kvikio.nvcomp  # noqa: F401
    except Exception:
        return False
    return True


def _nvcomp_path_available() -> bool:
    """True when at least one nvCOMP backend is loadable on this host.

    The optimised code path runs only when either the C nvCOMP library
    (``libnvcomp.so``) or ``kvikio.nvcomp`` is importable. Without one
    of those, ``_try_nvcomp_batch_decompress`` always returns None and
    timing/correctness tests would silently exercise the slower fallback
    decoder.
    """
    if not _gpu_available():
        return False
    try:
        from xrspatial.geotiff._gpu_decode import _get_nvcomp
    except Exception:
        return False
    if _get_nvcomp() is not None:
        return True
    return _kvikio_nvcomp_importable()


_HAS_GPU = _gpu_available()
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_HAS_NVCOMP = _nvcomp_path_available()
_nvcomp_only = pytest.mark.skipif(
    not (_HAS_GPU and _HAS_TIFFFILE and _HAS_NVCOMP),
    reason="cupy + CUDA + tifffile + (libnvcomp or kvikio.nvcomp) required",
)


def _write_deflate_tiled(path, arr, tile=(256, 256)):
    import tifffile
    tifffile.imwrite(
        str(path), arr, compression="deflate", tile=tile,
    )


def _wrap_nvcomp_with_call_recorder(monkeypatch):
    """Replace ``_try_nvcomp_batch_decompress`` with a wrapper that records
    each (compression, returned_non_none) call. Returns the records list."""
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


@_nvcomp_only
@pytest.mark.parametrize("size,tile", [
    (256, (128, 128)),    # 4 tiles
    (1024, (256, 256)),   # 16 tiles
    (2048, (128, 128)),   # 256 tiles -- matches the audit measurement
])
def test_nvcomp_batch_upload_correctness(tmp_path, monkeypatch, size, tile):
    """GPU decode of Deflate-tiled TIFFs is bit-exact vs CPU after the
    batched H2D upload rewrite, AND the nvCOMP fast-path actually ran."""
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260508)
    arr = rng.randint(0, 4096, size=(size, size), dtype=np.uint16)

    name = f"deflate_{size}_{tile[0]}_{uuid.uuid4().hex[:8]}.tif"
    path = tmp_path / name
    _write_deflate_tiled(path, arr, tile=tile)

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    records = _wrap_nvcomp_with_call_recorder(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)

    assert any(success for _, success in records), (
        "_try_nvcomp_batch_decompress was never invoked or always returned "
        f"None; records={records}. The optimised path was not exercised, so "
        f"this test would pass even if the rewrite were broken."
    )


@_nvcomp_only
def test_nvcomp_kvikio_fallback_skips_zstd(monkeypatch):
    """When the C nvCOMP lib is missing and kvikio is the only backend,
    ZSTD-compressed input must NOT take the kvikio DeflateManager path
    (which would strip a fake zlib header and try to decode ZSTD frames
    as Deflate). It must return None so the caller can fall through."""
    import xrspatial.geotiff._gpu_decode as _gpu_decode

    # Force the libnvcomp branch off so the kvikio fallback is the only
    # path. If kvikio.nvcomp isn't importable, the fallback returns
    # None for an unrelated reason -- skip in that case.
    if not _kvikio_nvcomp_importable():
        pytest.skip("kvikio.nvcomp not importable; the kvikio branch "
                    "is never entered on this host")
    monkeypatch.setattr(_gpu_decode, '_get_nvcomp', lambda: None)

    # Pass any bytes; the gate must reject ZSTD before any decode work.
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


@_nvcomp_only
def test_nvcomp_batch_upload_perf_regression_guard(tmp_path, monkeypatch):
    """Sanity guard: 2048x2048 Deflate-tiled GPU decode finishes under a
    generous threshold WHILE going through the nvCOMP fast-path. Skipping
    when nvCOMP isn't available is handled by ``_nvcomp_only``."""
    from xrspatial.geotiff import read_geotiff_gpu

    rng = np.random.RandomState(20260508)
    arr = rng.randint(0, 4096, size=(2048, 2048), dtype=np.uint16)
    path = tmp_path / f"deflate_2048_perf_{uuid.uuid4().hex[:8]}.tif"
    _write_deflate_tiled(path, arr, tile=(128, 128))

    # Warm up: first call may JIT-compile kernels and load CUDA libs.
    _ = read_geotiff_gpu(str(path))

    records = _wrap_nvcomp_with_call_recorder(monkeypatch)
    t0 = time.perf_counter()
    out = read_geotiff_gpu(str(path))
    elapsed = time.perf_counter() - t0

    assert any(success for _, success in records), (
        "nvCOMP fast-path did not run during the timed call; the threshold "
        f"is meaningless without it. Records: {records}"
    )

    # Generous regression threshold; the per-tile upload version was
    # ~6 ms just for H2D so anything well above 200 ms is a real
    # regression somewhere in the decode pipeline.
    assert elapsed < 0.2, (
        f"read_geotiff_gpu on 2048x2048 deflate-tiled TIFF took "
        f"{elapsed * 1000:.1f} ms (threshold 200 ms) -- possible "
        f"regression in the nvCOMP batched H2D upload path"
    )
    assert out.shape == (2048, 2048)
