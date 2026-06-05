"""Reader dtype handling.

Covers:

* The ``dtype=`` kwarg on ``open_geotiff`` (eager + dask, float -> float
  / int -> int casts, float -> int rejection).
* IEEE half-precision auto-promotion to float32 on read (eager + dask).
* The same float16 promotion on ``_read_geotiff_gpu`` and
  ``open_geotiff(gpu=True)``.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _read_geotiff_dask, open_geotiff, to_geotiff
from xrspatial.geotiff._dtypes import (SAMPLE_FORMAT_FLOAT, SAMPLE_FORMAT_INT, SAMPLE_FORMAT_UINT,
                                       tiff_dtype_to_numpy, tiff_storage_dtype)

from .._helpers.markers import requires_gpu as _gpu_only

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def float64_tif(tmp_path):
    """Write a float64 GeoTIFF for dtype cast tests."""
    arr = np.random.default_rng(99).random((80, 80)).astype(np.float64)
    y = np.linspace(40.0, 41.0, 80)
    x = np.linspace(-105.0, -104.0, 80)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    path = str(tmp_path / 'dtype_f64.tif')
    to_geotiff(da, path, compression='none')
    return path, arr


@pytest.fixture
def uint16_tif(tmp_path):
    """Write a uint16 GeoTIFF for dtype cast tests."""
    arr = np.random.default_rng(77).integers(0, 10000, (60, 60),
                                             dtype=np.uint16)
    y = np.linspace(40.0, 41.0, 60)
    x = np.linspace(-105.0, -104.0, 60)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    path = str(tmp_path / 'dtype_u16.tif')
    to_geotiff(da, path, compression='none')
    return path, arr


@pytest.fixture
def float16_tif(tmp_path):
    """Write a small float16 GeoTIFF using tifffile.

    tifffile encodes numpy float16 with ``BitsPerSample=16`` and
    ``SampleFormat=3``, which is what an external rasterio / GDAL caller
    would produce.
    """
    tifffile = pytest.importorskip("tifffile")
    arr = np.array(
        [[0.0, 1.0, 2.0, 3.0],
         [-1.0, -2.0, -3.0, -4.0],
         [0.5, 1.5, 2.5, 3.5],
         [100.0, 200.0, 300.0, 400.0]],
        dtype=np.float16,
    )
    path = tmp_path / "f16.tif"
    tifffile.imwrite(str(path), arr, compression=None)
    return path, arr


@pytest.fixture
def float16_stripped_tif(tmp_path):
    """Stripped float16 GeoTIFF: triggers the bps_mismatch CPU fallback."""
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
    """Multi-tile float16 GeoTIFF: 32x32 image, 16x16 tiles (2x2 grid)."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(1024, dtype=np.float16).reshape(32, 32)
    path = tmp_path / "f16_tiled.tif"
    tifffile.imwrite(
        str(path), arr, compression="deflate", tile=(16, 16))
    return path, arr


@pytest.fixture
def float16_tiled_uncompressed_tif(tmp_path):
    """Tiled uncompressed float16 GeoTIFF."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(256, dtype=np.float16).reshape(16, 16)
    path = tmp_path / "f16_tiled_none.tif"
    tifffile.imwrite(
        str(path), arr, compression=None, tile=(16, 16))
    return path, arr


# ---------------------------------------------------------------------------
# dtype= kwarg on open_geotiff (eager)
# ---------------------------------------------------------------------------


class TestDtypeEager:
    def test_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype='float32')
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result.values, orig.astype(np.float32), decimal=6)

    def test_float64_to_float16(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype=np.float16)
        assert result.dtype == np.float16

    def test_uint16_to_int32(self, uint16_tif):
        path, orig = uint16_tif
        result = open_geotiff(path, dtype='int32')
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result.values, orig.astype(np.int32))

    def test_uint16_to_uint8(self, uint16_tif):
        path, _ = uint16_tif
        result = open_geotiff(path, dtype='uint8')
        assert result.dtype == np.uint8

    def test_float_to_int_raises(self, float64_tif):
        path, _ = float64_tif
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='int32')

    def test_dtype_none_preserves_native(self, float64_tif):
        path, _ = float64_tif
        result = open_geotiff(path, dtype=None)
        assert result.dtype == np.float64

    def test_int_with_nodata_float_to_int_raises(self, tmp_path):
        """uint16 file with nodata: nodata masking promotes to float64, so float->int validation fires."""  # noqa: E501
        arr = np.array([[1, 2], [3, 9999]], dtype=np.uint16)
        y = np.linspace(40.0, 41.0, 2)
        x = np.linspace(-105.0, -104.0, 2)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          coords={'y': y, 'x': x},
                          attrs={'crs': 4326, 'nodata': 9999.0})
        path = str(tmp_path / 'dtype_nodata_int_eager.tif')
        to_geotiff(da, path, compression='none')
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='int32', masked=True)


# ---------------------------------------------------------------------------
# dtype= kwarg on open_geotiff (dask)
# ---------------------------------------------------------------------------


class TestDtypeDask:
    def test_float64_to_float32_dask(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype='float32', chunks=40)
        assert result.dtype == np.float32
        computed = result.values
        np.testing.assert_array_almost_equal(
            computed, orig.astype(np.float32), decimal=6)

    def test_chunks_are_target_dtype(self, float64_tif):
        path, _ = float64_tif
        result = open_geotiff(path, dtype='float32', chunks=40)
        assert result.data.dtype == np.float32

    def test_float_to_int_raises_dask(self, float64_tif):
        path, _ = float64_tif
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='int32', chunks=40)

    def test_int_with_nodata_float_to_int_raises_dask(self, tmp_path):
        """uint16 file with nodata: nodata masking promotes to float64, so float->int validation fires."""  # noqa: E501
        arr = np.array([[1, 2], [3, 9999]], dtype=np.uint16)
        y = np.linspace(40.0, 41.0, 2)
        x = np.linspace(-105.0, -104.0, 2)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          coords={'y': y, 'x': x},
                          attrs={'crs': 4326, 'nodata': 9999.0})
        path = str(tmp_path / 'dtype_nodata_int_dask.tif')
        to_geotiff(da, path, compression='none')
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='int32', chunks=2, masked=True)


# ---------------------------------------------------------------------------
# Float16 dtype-map: auto-promotion on read
# ---------------------------------------------------------------------------


class TestFloat16DtypeMap:
    """The dtype map auto-promotes float16 on read."""

    def test_tiff_dtype_to_numpy_float16(self):
        assert tiff_dtype_to_numpy(16, SAMPLE_FORMAT_FLOAT) == np.float32

    def test_tiff_storage_dtype_float16(self):
        assert tiff_storage_dtype(16, SAMPLE_FORMAT_FLOAT) == np.float16

    def test_tiff_storage_dtype_delegates_for_non_promoted(self):
        # Non-promoted keys behave identically.
        for bps, sf in [
            (8, SAMPLE_FORMAT_UINT),
            (16, SAMPLE_FORMAT_UINT),
            (16, SAMPLE_FORMAT_INT),
            (32, SAMPLE_FORMAT_FLOAT),
            (64, SAMPLE_FORMAT_FLOAT),
        ]:
            assert tiff_storage_dtype(bps, sf) == tiff_dtype_to_numpy(bps, sf)


# ---------------------------------------------------------------------------
# Float16 eager + dask reads
# ---------------------------------------------------------------------------


class TestEagerFloat16Read:
    """``open_geotiff`` decodes an external float16 file to float32."""

    def test_open_geotiff_returns_float32(self, float16_tif):
        path, arr = float16_tif
        result = open_geotiff(str(path))
        assert result.dtype == np.float32
        # Float16 values fit exactly in float32, so equality is well-defined.
        np.testing.assert_array_equal(result.values, arr.astype(np.float32))

    def test_open_geotiff_dask_returns_float32(self, float16_tif):
        path, arr = float16_tif
        result = _read_geotiff_dask(str(path), chunks=2)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.compute().values, arr.astype(np.float32))


class TestPredictor3Float16:
    """Predictor=3 + float16 on disk also decodes correctly."""

    def test_predictor3_float16_round_trip(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        pytest.importorskip("imagecodecs")  # required for predictor=3
        arr = np.linspace(-1.0, 1.0, 16).astype(np.float16).reshape(4, 4)
        path = tmp_path / "pred3_f16.tif"
        tifffile.imwrite(
            str(path), arr, predictor=3, compression="deflate")

        result = open_geotiff(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.values, arr.astype(np.float32))


class TestFloat16RegressionGuards:
    """The float16 promotion did not change non-float16 behaviour."""

    def test_float32_still_float32(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = tmp_path / "f32.tif"
        tifffile.imwrite(str(path), arr)

        result = open_geotiff(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result.values, arr)

    def test_float64_still_float64(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        arr = np.arange(16, dtype=np.float64).reshape(4, 4)
        path = tmp_path / "f64.tif"
        tifffile.imwrite(str(path), arr)

        result = open_geotiff(str(path))
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result.values, arr)

    def test_uint16_still_uint16(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
        path = tmp_path / "u16.tif"
        tifffile.imwrite(str(path), arr)

        result = open_geotiff(str(path))
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result.values, arr)


# ---------------------------------------------------------------------------
# Float16 GPU read paths
# ---------------------------------------------------------------------------


class TestEagerGPUReadFloat16:
    """``_read_geotiff_gpu`` returns float32 for stripped float16 input."""

    @_gpu_only
    def test_read_geotiff_gpu_stripped_returns_float32(
        self, float16_stripped_tif
    ):
        from xrspatial.geotiff import _read_geotiff_gpu

        path, arr = float16_stripped_tif
        result = _read_geotiff_gpu(str(path))
        assert result.dtype == np.float32, (
            f"GPU read of float16 must return float32, got {result.dtype}"
        )
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))

    @_gpu_only
    def test_read_geotiff_gpu_tiled_returns_float32(
        self, float16_tiled_tif
    ):
        from xrspatial.geotiff import _read_geotiff_gpu

        path, arr = float16_tiled_tif
        result = _read_geotiff_gpu(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))

    @_gpu_only
    def test_read_geotiff_gpu_tiled_uncompressed_returns_float32(
        self, float16_tiled_uncompressed_tif
    ):
        from xrspatial.geotiff import _read_geotiff_gpu

        path, arr = float16_tiled_uncompressed_tif
        result = _read_geotiff_gpu(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))

    @_gpu_only
    def test_open_geotiff_gpu_dispatcher_float16(self, float16_tiled_tif):
        """``open_geotiff(gpu=True)`` dispatches correctly for float16."""
        path, arr = float16_tiled_tif
        result = open_geotiff(str(path), gpu=True)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))


class TestGPUWindowedFloat16:
    """Windowed GPU reads honour the bps_mismatch fallback path."""

    @_gpu_only
    def test_read_geotiff_gpu_windowed_stripped(self, float16_stripped_tif):
        from xrspatial.geotiff import _read_geotiff_gpu

        path, arr = float16_stripped_tif
        result = _read_geotiff_gpu(str(path), window=(0, 0, 2, 2))
        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(
            result.data.get(), arr[:2, :2].astype(np.float32))

    @_gpu_only
    def test_read_geotiff_gpu_windowed_tiled(self, float16_tiled_tif):
        from xrspatial.geotiff import _read_geotiff_gpu

        path, arr = float16_tiled_tif
        result = _read_geotiff_gpu(str(path), window=(0, 0, 8, 8))
        assert result.dtype == np.float32
        assert result.shape == (8, 8)
        np.testing.assert_array_equal(
            result.data.get(), arr[:8, :8].astype(np.float32))


class TestDaskGPUFloat16:
    """``open_geotiff(chunks=, gpu=True)`` decodes float16 correctly."""

    @_gpu_only
    def test_dask_gpu_tiled_float16(self, float16_tiled_tif):
        path, arr = float16_tiled_tif
        result = open_geotiff(str(path), chunks=8, gpu=True)
        assert result.dtype == np.float32, (
            f"dask+GPU read of float16 must return float32, got {result.dtype}"
        )
        computed = result.compute()
        np.testing.assert_array_equal(
            computed.data.get(), arr.astype(np.float32))

    @_gpu_only
    def test_read_geotiff_gpu_chunks_kwarg_float16(self, float16_tiled_tif):
        """``_read_geotiff_gpu(chunks=)`` also routes correctly."""
        from xrspatial.geotiff import _read_geotiff_gpu

        path, arr = float16_tiled_tif
        result = _read_geotiff_gpu(str(path), chunks=8)
        assert result.dtype == np.float32
        computed = result.compute()
        np.testing.assert_array_equal(
            computed.data.get(), arr.astype(np.float32))


class TestGDSPathGatedOffForFloat16:
    """``_gds_chunk_path_available`` returns False for (bps=16, sf=3)."""

    @_gpu_only
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

        assert ifd.is_tiled, "fixture sanity: tiled layout expected"
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

    @_gpu_only
    def test_gds_path_allowed_for_float32_tiled(self, tmp_path):
        """Sanity: GDS path remains allowed for a float32 tiled file."""
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

    @_gpu_only
    def test_eager_numpy_equals_gpu(self, float16_tiled_tif):
        path, _ = float16_tiled_tif
        cpu = open_geotiff(str(path))
        gpu = open_geotiff(str(path), gpu=True)

        assert cpu.dtype == gpu.dtype == np.float32
        np.testing.assert_array_equal(np.asarray(cpu), gpu.data.get())

    @_gpu_only
    def test_eager_numpy_equals_dask_gpu(self, float16_tiled_tif):
        path, _ = float16_tiled_tif
        cpu = open_geotiff(str(path))
        dask_gpu = open_geotiff(str(path), chunks=8, gpu=True).compute()

        assert cpu.dtype == dask_gpu.dtype == np.float32
        np.testing.assert_array_equal(
            np.asarray(cpu), dask_gpu.data.get())

    @_gpu_only
    def test_dask_numpy_equals_dask_gpu(self, float16_tiled_tif):
        path, _ = float16_tiled_tif
        dask_cpu = _read_geotiff_dask(str(path), chunks=8).compute()
        dask_gpu = open_geotiff(str(path), chunks=8, gpu=True).compute()

        np.testing.assert_array_equal(
            np.asarray(dask_cpu), dask_gpu.data.get())


class TestPredictor3Float16GPU:
    """Predictor=3 + float16 on disk also decodes correctly on GPU."""

    @_gpu_only
    def test_predictor3_float16_gpu_round_trip(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        pytest.importorskip("imagecodecs")  # required for predictor=3

        from xrspatial.geotiff import _read_geotiff_gpu

        arr = np.linspace(-1.0, 1.0, 16).astype(np.float16).reshape(4, 4)
        path = tmp_path / "pred3_f16.tif"
        tifffile.imwrite(
            str(path), arr, predictor=3, compression="deflate")

        result = _read_geotiff_gpu(str(path))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result.data.get(), arr.astype(np.float32))
