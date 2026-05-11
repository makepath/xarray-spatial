"""GPU regression test for issue #1549.

Reading a 3-band tiled JPEG GeoTIFF with ``open_geotiff(..., gpu=True)``
crashed inside the nvJPEG decode kernel with::

    cupy.cuda.runtime.CUDARuntimeError: cudaErrorIllegalAddress

The crash was sticky -- it poisoned the CUDA context so every later GPU
call in the same process also failed.  Root cause: the
``nvjpegOutputFormat_t`` constants in ``_gpu_decode.py`` were defined
two values lower than the SDK's enum.  The wrapper sent ``3`` thinking
it was ``NVJPEG_OUTPUT_RGBI`` (interleaved RGB), but ``3`` is
``NVJPEG_OUTPUT_RGB`` (planar) in the real SDK.  nvJPEG then wrote the
G and B planes through ``nvjpegImage.channel[1]`` and
``channel[2]``, which the wrapper had set to NULL for an interleaved
output buffer, producing an out-of-bounds GPU write inside
``ycbcr_to_format_kernel_roi``.

The same off-by-two affected the single-band path: it sent ``5``
(thinking ``NVJPEG_OUTPUT_UNCHANGED``) which is actually
``NVJPEG_OUTPUT_RGBI`` -- nvJPEG produced 3-byte-per-pixel output into a
1-byte-per-pixel buffer, returning visibly wrong pixels rather than
crashing.

These tests build the exact reproducer from the issue, decode it on GPU,
and verify (a) the decode does not crash, (b) the GPU pixels match the
CPU pixels within the typical libjpeg/nvjpeg rounding tolerance, and
(c) the CUDA context survives a follow-up GPU read of an unrelated
file.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _gpu_available() -> bool:
    """True when cupy is importable and CUDA is initialised."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def _nvjpeg_available() -> bool:
    """True when libnvjpeg.so loads on this host.

    Without nvJPEG the GPU pipeline silently falls back to CPU Pillow
    decode, so the regression for issue #1549 (an out-of-bounds write
    inside the nvJPEG kernel) would never be exercised. Skip rather
    than test a path the bug never lived on.
    """
    if not _gpu_available():
        return False
    try:
        from xrspatial.geotiff._gpu_decode import _get_nvjpeg
        return _get_nvjpeg() is not None
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_HAS_PIL = importlib.util.find_spec("PIL") is not None
# tifffile.imwrite(compression='jpeg') delegates the codec to imagecodecs
# (or libjpeg via Pillow on some installs); when neither is wired up the
# write raises and the suite would error instead of skipping cleanly.
_HAS_IMAGECODECS = importlib.util.find_spec("imagecodecs") is not None
_HAS_NVJPEG = _nvjpeg_available()

_gpu_only = pytest.mark.skipif(
    not (_HAS_GPU and _HAS_TIFFFILE and _HAS_PIL
         and _HAS_IMAGECODECS and _HAS_NVJPEG),
    reason="cupy + CUDA + tifffile + Pillow + imagecodecs + nvJPEG required",
)


def _write_jpeg_rgb_tiff(path: str, seed: int = 0,
                          noise: bool = True) -> np.ndarray:
    """Write a 3-band 256x256 tiled JPEG TIFF using tifffile.

    tifffile emits a complete JFIF stream (SOI + APP0 + DQT + DHT + SOF0
    + ...) per tile rather than a JPEGTables-style abbreviated stream,
    so the GPU decode path runs without any extra splice step.

    With ``noise=True`` the payload is uniform random bytes -- the
    worst-case input for JPEG compression and useful for the
    crash-reproducer test which only cares about safe completion.
    With ``noise=False`` the payload is a smooth gradient where
    libjpeg and nvjpeg agree to within a few LSBs per pixel; that lets
    the cross-backend match assertion run with a tight tolerance.
    """
    import tifffile
    if noise:
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    else:
        # Smooth gradient: per-channel ramp + cross terms.
        ys, xs = np.mgrid[0:256, 0:256].astype(np.int32)
        r = (ys + xs) // 2
        g = ys
        b = xs
        arr = np.stack([r, g, b], axis=2).clip(0, 255).astype(np.uint8)
    tifffile.imwrite(path, arr, photometric='rgb', tile=(128, 128),
                     compression='jpeg')
    return arr


def _write_jpeg_gray_tiff(path: str, seed: int = 42) -> np.ndarray:
    """Write a 1-band 256x256 tiled JPEG TIFF using tifffile."""
    import tifffile
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    tifffile.imwrite(path, arr, photometric='minisblack', tile=(128, 128),
                     compression='jpeg')
    return arr


@_gpu_only
def test_rgb_jpeg_gpu_no_crash(tmp_path, monkeypatch):
    """3-band JPEG must not raise CUDARuntimeError on GPU read.

    Uses ``gpu='strict'`` so the original
    ``cudaErrorIllegalAddress`` would propagate up the stack instead of
    being swallowed by the auto-mode CPU fallback.  Without strict the
    bug presented as a ``RuntimeWarning: GPU decode failed (...);
    falling back to CPU`` and the returned array was the CPU array, so
    the assertions below would still pass even with the bug present.

    Also spies on ``_try_nvjpeg_batch_decode`` to fail loudly if the
    decode took a CPU fallback path instead of nvJPEG.  Without this
    guard the test would pass on a system whose nvJPEG returned None for
    any reason, defeating the point of the regression test.
    """
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff import _gpu_decode
    import cupy

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
    _write_jpeg_rgb_tiff(path)

    arr = read_geotiff_gpu(path, gpu='strict')
    # Materialise the GPU buffer so any deferred kernel actually runs
    # and surface any sticky error from the decode pipeline.
    assert isinstance(arr.data, cupy.ndarray)
    decoded = arr.data.get()
    assert decoded.shape == (256, 256, 3)
    assert decoded.dtype == np.uint8

    assert spy["calls"] >= 1, (
        "nvJPEG branch was never called — test did not exercise the "
        "code path the #1549 fix lives on"
    )
    assert spy["successes"] >= 1, (
        "nvJPEG returned None — CPU Pillow fallback ran and the fix was "
        "not exercised"
    )


@_gpu_only
def test_rgb_jpeg_gpu_matches_cpu(tmp_path):
    """GPU pixels must be within JPEG decoder tolerance of CPU pixels.

    With a smooth gradient input the libjpeg (CPU) and nvjpeg (GPU)
    decoders agree to a couple of LSBs per pixel.  The off-by-two
    constant bug scrambled channels enough to push the mean diff above
    60 and the max diff above 200, so a tight bound here pins both the
    constant fix and the per-tile sync that keeps the multi-tile
    decode deterministic.
    """
    from xrspatial.geotiff import open_geotiff

    path = str(tmp_path / "rgb_jpeg_match_1549.tif")
    _write_jpeg_rgb_tiff(path, noise=False)

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)
    assert cpu.shape == gpu.shape == (256, 256, 3)

    cpu_arr = np.asarray(cpu.data)
    gpu_arr = np.asarray(gpu.data.get())

    diff = np.abs(cpu_arr.astype(int) - gpu_arr.astype(int))
    assert diff.mean() < 1.0, f"mean diff {diff.mean():.3f} too large"
    assert diff.max() < 8, f"max diff {diff.max()} too large"


@_gpu_only
def test_grayscale_jpeg_gpu_matches_cpu(tmp_path):
    """Single-band JPEG GPU read must also produce correct pixels.

    With the off-by-two constants the single-band path silently produced
    wrong output (each Y value duplicated three times then wrapped) -- a
    quieter failure mode than the 3-band crash but a corruption
    nonetheless.
    """
    from xrspatial.geotiff import open_geotiff

    path = str(tmp_path / "gray_jpeg_1549.tif")
    _write_jpeg_gray_tiff(path)

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)
    assert cpu.shape == gpu.shape == (256, 256)

    cpu_arr = np.asarray(cpu.data)
    gpu_arr = np.asarray(gpu.data.get())
    diff = np.abs(cpu_arr.astype(int) - gpu_arr.astype(int))
    # For grayscale there is no chroma involved, so libjpeg and nvjpeg
    # only diverge by IDCT rounding (typically <= 1 LSB).
    assert diff.max() <= 2, (
        f"grayscale max diff {diff.max()} indicates corruption, "
        f"not just rounding"
    )


@_gpu_only
def test_cuda_context_survives_after_jpeg_gpu_read(tmp_path):
    """Verify the CUDA context is healthy after a GPU JPEG read.

    Before the fix, the failing nvJPEG kernel left the context in an
    error state so every later GPU call in the same process raised
    ``cudaErrorIllegalAddress`` -- even unrelated allocations.  This
    test reads the JPEG on GPU, then performs a small follow-up GPU
    operation and an unrelated GPU read and asserts both succeed.
    """
    import cupy
    from xrspatial.geotiff import open_geotiff

    path = str(tmp_path / "rgb_ctx_1549.tif")
    _write_jpeg_rgb_tiff(path)

    arr = open_geotiff(path, gpu=True)
    _ = arr.data.get()

    # Plain CuPy op -- this is the call that used to surface the sticky
    # cudaErrorIllegalAddress on its first allocation.
    x = cupy.arange(1024, dtype=cupy.float32)
    s = float(cupy.sum(x).item())
    assert s == 1023 * 1024 / 2

    # Unrelated GPU TIFF read -- closes the loop on the issue's
    # "every later GPU call fails" symptom.
    other_path = str(tmp_path / "other_1549.tif")
    _write_jpeg_gray_tiff(other_path, seed=7)
    other = open_geotiff(other_path, gpu=True)
    assert other.shape == (256, 256)
    assert other.dtype == np.uint8
