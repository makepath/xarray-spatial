"""GPU backend coverage for issue #1941 (float16 read).

#1941 added float16 auto-promotion on read by making
``tiff_dtype_to_numpy(16, SAMPLE_FORMAT_FLOAT)`` return ``float32`` and
adding the on-disk ``tiff_storage_dtype`` companion. The eager numpy and
dask paths are covered by ``test_float16_read_1941.py``; this module
closes the GPU and dask+GPU coverage gap.

A regression that:

* dropped the ``bps_mismatch`` stripped/odd-bps fallback at
  ``_backends/gpu.py:357`` would route float16 stripped reads through
  the tiled GPU decoder and mis-decode the half-precision samples;
* dropped the ``bps_first == 16 and sample_format == SAMPLE_FORMAT_FLOAT``
  early-out at ``_backends/gpu.py:791`` in ``_gds_chunk_path_available``
  would send tiled float16 chunked reads down the kvikIO GDS path and
  mis-stride the buffer;
* dropped the entry at ``(16, SAMPLE_FORMAT_FLOAT) -> float32`` in
  ``tiff_dtype_to_numpy`` would surface as ``ValueError("Unsupported
  BitsPerSample=16, SampleFormat=3")`` from the GPU read paths.

Every test ships through ``read_geotiff_gpu`` directly or through
``open_geotiff(..., gpu=True)`` so the dispatcher path is also wired in.
``cuda-unavailable`` builds skip the suite via the project's standard
``CUDA_AVAILABLE`` gate.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


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
    not _HAS_GPU, reason="cupy + CUDA required for GPU float16 read tests",
)


@pytest.fixture
def float16_stripped_tif(tmp_path):
    """Stripped float16 GeoTIFF: triggers the bps_mismatch CPU fallback.

    ``tifffile.imwrite`` without ``tile=`` produces a stripped layout, so
    the GPU reader hits ``bps_mismatch=True`` (file_dtype.itemsize*8 == 32
    but bps == 16) and falls back to ``_read_to_array`` on CPU before
    copying to device.
    """
    tifffile = pytest.importorskip("tifffile")
    arr = np.array(
        [[0.0, 1.0, 2.0, 3.0],
         [-1.0, -2.0, -3.0, -4.0],
         [0.5, 1.5, 2.5, 3.5],
         [100.0, 200.0, 300.0, 400.0]],
        dtype=np.float16,
    )
    path = tmp_path / "f16_stripped.tif"
    tifffile.imwrite(str(path), arr, compression=None)
    return path, arr


@pytest.fixture
def float16_tiled_tif(tmp_path):
    """Multi-tile float16 GeoTIFF: 32x32 image, 16x16 tiles (2x2 grid).

    Tiled and deflate-compressed. The 2x2 tile grid exercises inter-tile
    reassembly in the decoder path so a regression that mis-stitched
    adjacent tiles would surface here. ``bps_mismatch`` short-circuits
    the tiled GPU decode path and routes through the CPU decoder; the
    GDS path is also gated off via ``_gds_chunk_path_available``
    returning False for (bps=16, sf=3).
    """
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(1024, dtype=np.float16).reshape(32, 32)
    path = tmp_path / "f16_tiled.tif"
    tifffile.imwrite(
        str(path), arr, compression="deflate", tile=(16, 16))
    return path, arr


@pytest.fixture
def float16_tiled_uncompressed_tif(tmp_path):
    """Tiled uncompressed float16 GeoTIFF.

    Mirrors ``float16_tiled_tif`` but with ``compression=None`` so the
    tile-decode path is exercised without an extra deflate codec call.
    Tile size 16 is the smallest tifffile allows.
    """
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(256, dtype=np.float16).reshape(16, 16)
    path = tmp_path / "f16_tiled_none.tif"
    tifffile.imwrite(
        str(path), arr, compression=None, tile=(16, 16))
    return path, arr


class TestEagerGPUReadFloat16:
    """``read_geotiff_gpu`` returns float32 for stripped float16 input."""

    def test_read_geotiff_gpu_stripped_returns_float32(
        self, float16_stripped_tif
    ):
        from xrspatial.geotiff import read_geotiff_gpu

        path, arr = float16_stripped_tif
        result = read_geotiff_gpu(str(path))
        assert result.dtype == np.float32, (
            f"GPU read of float16 must return float32, got {result.dtype}"
        )
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))

    def test_read_geotiff_gpu_tiled_returns_float32(
        self, float16_tiled_tif
    ):
        from xrspatial.geotiff import read_geotiff_gpu

        path, arr = float16_tiled_tif
        result = read_geotiff_gpu(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))

    def test_read_geotiff_gpu_tiled_uncompressed_returns_float32(
        self, float16_tiled_uncompressed_tif
    ):
        from xrspatial.geotiff import read_geotiff_gpu

        path, arr = float16_tiled_uncompressed_tif
        result = read_geotiff_gpu(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))

    def test_open_geotiff_gpu_dispatcher_float16(self, float16_tiled_tif):
        """``open_geotiff(gpu=True)`` dispatches correctly for float16."""
        from xrspatial.geotiff import open_geotiff

        path, arr = float16_tiled_tif
        result = open_geotiff(str(path), gpu=True)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))


class TestGPUWindowedFloat16:
    """Windowed GPU reads honour the bps_mismatch fallback path."""

    def test_read_geotiff_gpu_windowed_stripped(self, float16_stripped_tif):
        from xrspatial.geotiff import read_geotiff_gpu

        path, arr = float16_stripped_tif
        result = read_geotiff_gpu(str(path), window=(0, 0, 2, 2))
        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(
            result.data.get(), arr[:2, :2].astype(np.float32))

    def test_read_geotiff_gpu_windowed_tiled(self, float16_tiled_tif):
        from xrspatial.geotiff import read_geotiff_gpu

        path, arr = float16_tiled_tif
        result = read_geotiff_gpu(str(path), window=(0, 0, 8, 8))
        assert result.dtype == np.float32
        assert result.shape == (8, 8)
        np.testing.assert_array_equal(
            result.data.get(), arr[:8, :8].astype(np.float32))


class TestDaskGPUFloat16:
    """``open_geotiff(chunks=, gpu=True)`` decodes float16 correctly."""

    def test_dask_gpu_tiled_float16(self, float16_tiled_tif):
        from xrspatial.geotiff import open_geotiff

        path, arr = float16_tiled_tif
        result = open_geotiff(str(path), chunks=8, gpu=True)
        assert result.dtype == np.float32, (
            f"dask+GPU read of float16 must return float32, got {result.dtype}"
        )
        # Compute the dask array; under dask+cupy, .compute() yields a
        # cupy-backed DataArray, so the .data.get() step pulls to host.
        computed = result.compute()
        np.testing.assert_array_equal(
            computed.data.get(), arr.astype(np.float32))

    def test_read_geotiff_gpu_chunks_kwarg_float16(self, float16_tiled_tif):
        """``read_geotiff_gpu(chunks=)`` also routes correctly."""
        from xrspatial.geotiff import read_geotiff_gpu

        path, arr = float16_tiled_tif
        result = read_geotiff_gpu(str(path), chunks=8)
        assert result.dtype == np.float32
        computed = result.compute()
        np.testing.assert_array_equal(
            computed.data.get(), arr.astype(np.float32))


class TestGDSPathGatedOffForFloat16:
    """``_gds_chunk_path_available`` returns False for (bps=16, sf=3).

    Direct structural test of the gating logic added in #1941 to keep the
    KvikIO GDS chunked path from mis-decoding half-precision tiles. A
    regression dropping the float16 guard would silently corrupt every
    chunked GPU read of a float16 source.
    """

    def test_gds_path_gated_off_for_float16(self, float16_tiled_tif):
        pytest.importorskip("kvikio", exc_type=ImportError)

        from xrspatial.geotiff._backends.gpu import _gds_chunk_path_available
        from xrspatial.geotiff._header import parse_all_ifds, parse_header

        path, _ = float16_tiled_tif
        with open(str(path), "rb") as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]

        # Sanity-check fixture: tiled, bps=16, sample_format=3 (float)
        from xrspatial.geotiff._dtypes import SAMPLE_FORMAT_FLOAT
        assert ifd.is_tiled, "fixture sanity: tiled layout expected"
        # Mirror the production unpacking pattern at gpu.py:791
        # (bps_first[0] if bps_first else 0) so an empty BitsPerSample
        # tuple would not raise IndexError here.
        bps_first = ifd.bits_per_sample
        if isinstance(bps_first, tuple):
            bps = bps_first[0] if bps_first else 0
        else:
            bps = bps_first
        assert bps == 16, "fixture sanity: bps=16 expected"
        assert ifd.sample_format == SAMPLE_FORMAT_FLOAT

        result = _gds_chunk_path_available(
            str(path), ifd, has_sparse_tile=False, orientation=1)
        assert result is False, (
            "_gds_chunk_path_available must return False for "
            "(bps=16, sf=float) so the GDS chunked path does not "
            "mis-decode half-precision tiles."
        )

    def test_gds_path_allowed_for_float32_tiled(self, tmp_path):
        """Sanity: GDS path remains allowed for a float32 tiled file.

        Pins that the float16 guard at gpu.py:791 fires only on
        (bps=16, sf=float), not on every tiled float file. A regression
        widening the guard to all floats would silently disable the
        GDS path on every float32 tiled COG.
        """
        tifffile = pytest.importorskip("tifffile")
        pytest.importorskip("kvikio", exc_type=ImportError)

        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = tmp_path / "f32_tiled.tif"
        tifffile.imwrite(
            str(path), arr, compression="deflate", tile=(16, 16))

        from xrspatial.geotiff._backends.gpu import _gds_chunk_path_available
        from xrspatial.geotiff._header import parse_all_ifds, parse_header

        with open(str(path), "rb") as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)

        result = _gds_chunk_path_available(
            str(path), ifds[0], has_sparse_tile=False, orientation=1)
        assert result is True, (
            "_gds_chunk_path_available must remain True for "
            "(bps=32, sf=float) tiled files so the kvikio GDS chunk "
            "path still applies."
        )


class TestBackendParityFloat16:
    """All four backends agree pixel-exact on float16 input."""

    def test_eager_numpy_equals_gpu(self, float16_tiled_tif):
        from xrspatial.geotiff import open_geotiff

        path, _ = float16_tiled_tif
        cpu = open_geotiff(str(path))
        gpu = open_geotiff(str(path), gpu=True)

        assert cpu.dtype == gpu.dtype == np.float32
        np.testing.assert_array_equal(np.asarray(cpu), gpu.data.get())

    def test_eager_numpy_equals_dask_gpu(self, float16_tiled_tif):
        from xrspatial.geotiff import open_geotiff

        path, _ = float16_tiled_tif
        cpu = open_geotiff(str(path))
        dask_gpu = open_geotiff(str(path), chunks=8, gpu=True).compute()

        assert cpu.dtype == dask_gpu.dtype == np.float32
        np.testing.assert_array_equal(
            np.asarray(cpu), dask_gpu.data.get())

    def test_dask_numpy_equals_dask_gpu(self, float16_tiled_tif):
        from xrspatial.geotiff import open_geotiff, read_geotiff_dask

        path, _ = float16_tiled_tif
        dask_cpu = read_geotiff_dask(str(path), chunks=8).compute()
        dask_gpu = open_geotiff(str(path), chunks=8, gpu=True).compute()

        np.testing.assert_array_equal(
            np.asarray(dask_cpu), dask_gpu.data.get())


class TestPredictor3Float16GPU:
    """Predictor=3 + float16 on disk also decodes correctly on GPU."""

    def test_predictor3_float16_gpu_round_trip(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        pytest.importorskip("imagecodecs")  # required for predictor=3

        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.linspace(-1.0, 1.0, 16).astype(np.float16).reshape(4, 4)
        path = tmp_path / "pred3_f16.tif"
        tifffile.imwrite(
            str(path), arr, predictor=3, compression="deflate")

        result = read_geotiff_gpu(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))
