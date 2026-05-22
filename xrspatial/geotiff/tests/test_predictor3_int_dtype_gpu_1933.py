"""GPU + dask+GPU backend coverage for issue #1933.

#1933 added ``_validate_predictor_sample_format`` and wired it into
every IFD-read site (eager numpy, dask, GPU tiled, GPU stripped). The
eager and dask paths are covered by ``test_predictor3_int_dtype_1933``;
this module closes the GPU coverage gap.

The validator is invoked at two GPU sites:

* ``_backends/gpu.py:443`` -- the tiled eager GPU read path. Reached when
  the file is tiled and ``bps == file_dtype.itemsize * 8`` (so the
  bps_mismatch fallback at line 358 does not take over).
* ``_backends/gpu.py:999`` -- the GDS chunked GPU path
  (``_read_geotiff_gpu_chunked_gds``). Reached when the file qualifies
  for direct disk->GPU decode.

The stripped GPU path falls back to CPU via ``_read_to_array`` and the
CPU-side validator there fires; the dask+GPU non-GDS path delegates to
``read_geotiff_dask`` which has its own validator (covered by the
existing dask test). The two NEW call sites have no targeted tests.

A regression dropping either of those two validator calls would let
malformed predictor=3 + integer tiled files decode silently to
garbage bytes on GPU. The eager-test asserts the error path is wired
on CPU; this module asserts the GPU dispatcher path is wired too.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff._compression import COMPRESSION_NONE
from xrspatial.geotiff._dtypes import LONG, SHORT, numpy_to_tiff_dtype
from xrspatial.geotiff._header import (TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_LENGTH,
                                       TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_PREDICTOR,
                                       TAG_ROWS_PER_STRIP, TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL,
                                       TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS,
                                       TAG_TILE_BYTE_COUNTS, TAG_TILE_LENGTH, TAG_TILE_OFFSETS,
                                       TAG_TILE_WIDTH)
from xrspatial.geotiff._writer import _assemble_standard_layout, _write_stripped


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
pytestmark = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)


def _build_predictor3_uint32_stripped_tiff(arr: np.ndarray) -> bytes:
    """Build a stripped TIFF: predictor=3 + uint32 SampleFormat=1.

    Mirrors the helper in ``test_predictor3_int_dtype_1933`` so the GPU
    coverage gap can be exercised against the same shape of malformed
    file the eager test uses. Compression is COMPRESSION_NONE so the
    strip bytes are exactly the raw integer values.
    """
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


def _build_predictor3_uint32_tiled_tiff(
    arr: np.ndarray, tile_w: int = 16, tile_h: int = 16,
) -> bytes:
    """Build a tiled malformed TIFF: predictor=3 + uint32 SampleFormat=1.

    The tiled layout is the one that reaches the GPU validator at
    ``_backends/gpu.py:443`` (no bps_mismatch fallback). Tile size is
    16x16, the smallest tifffile/standard tile size.
    """
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


class TestGPUEagerRejectsMalformedFile:
    """``read_geotiff_gpu`` rejects predictor=3 + integer SampleFormat."""

    def test_gpu_eager_stripped_raises(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint32)
        path = tmp_path / "pred3_uint32_stripped.tif"
        path.write_bytes(_build_predictor3_uint32_stripped_tiff(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_gpu(str(path))

    def test_gpu_eager_tiled_raises(self, tmp_path):
        """Tiled layout hits the tiled GPU validator at gpu.py:443.

        Distinct from the stripped fallback path -- a regression
        dropping the line 443 call would leak through this test
        because the stripped path's validator lives in
        ``_read_to_array`` and would still raise.
        """
        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.arange(256, dtype=np.uint32).reshape(16, 16)
        path = tmp_path / "pred3_uint32_tiled.tif"
        path.write_bytes(_build_predictor3_uint32_tiled_tiff(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_gpu(str(path))

    def test_gpu_dispatcher_eager_raises(self, tmp_path):
        """``open_geotiff(gpu=True)`` dispatcher rejects the file."""
        from xrspatial.geotiff import open_geotiff

        arr = np.arange(64, dtype=np.uint32).reshape(8, 8)
        path = tmp_path / "pred3_uint32_dispatch.tif"
        path.write_bytes(_build_predictor3_uint32_stripped_tiff(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            open_geotiff(str(path), gpu=True)


class TestGPUChunkedRejectsMalformedFile:
    """The dask+GPU paths also reject predictor=3 + integer."""

    def test_read_geotiff_gpu_chunked_stripped_raises(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.arange(64, dtype=np.uint32).reshape(8, 8)
        path = tmp_path / "pred3_uint32_chunked_str.tif"
        path.write_bytes(_build_predictor3_uint32_stripped_tiff(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_gpu(str(path), chunks=4)

    def test_read_geotiff_gpu_chunked_tiled_raises(self, tmp_path):
        """Tiled chunked path with KvikIO available exercises gpu.py:999.

        Gated on ``kvikio`` so the GDS qualification path
        (``_read_geotiff_gpu_chunked_gds``) is the branch actually
        taken. Without KvikIO the dispatcher falls back to the CPU
        dask path and the line-999 validator is never reached, which
        leaves the targeted call site untested. The CPU fallback
        rejection is already covered by the eager/dask tests in
        ``test_predictor3_int_dtype_1933``.
        """
        pytest.importorskip("kvikio")

        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.arange(256, dtype=np.uint32).reshape(16, 16)
        path = tmp_path / "pred3_uint32_chunked_tiled.tif"
        path.write_bytes(_build_predictor3_uint32_tiled_tiff(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_gpu(str(path), chunks=16)

    def test_open_geotiff_chunks_gpu_dispatcher_raises(self, tmp_path):
        """``open_geotiff(chunks=, gpu=True)`` dispatcher rejects the file."""
        from xrspatial.geotiff import open_geotiff

        arr = np.arange(256, dtype=np.uint32).reshape(16, 16)
        path = tmp_path / "pred3_uint32_chunked_dispatch.tif"
        path.write_bytes(_build_predictor3_uint32_tiled_tiff(arr))
        with pytest.raises(ValueError, match="Predictor=3"):
            open_geotiff(str(path), chunks=8, gpu=True)


class TestValidPredictor3StillWorksOnGPU:
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


class TestErrorMessageStable:
    """The GPU error wording matches the eager/dask wording.

    Cross-backend error parity is a real concern -- a regression that
    fired the validator on GPU but with a different message would force
    callers to special-case the backend on ``except ValueError``.
    """

    def test_gpu_error_message_matches_eager(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

        arr = np.arange(64, dtype=np.uint32).reshape(8, 8)
        path = tmp_path / "pred3_uint32_msg.tif"
        path.write_bytes(_build_predictor3_uint32_stripped_tiff(arr))

        with pytest.raises(ValueError) as exc_eager:
            open_geotiff(str(path))
        with pytest.raises(ValueError) as exc_gpu:
            read_geotiff_gpu(str(path))

        assert str(exc_eager.value) == str(exc_gpu.value), (
            "GPU and eager paths must surface the same Predictor=3 "
            "error message so callers can use a single except branch."
        )
