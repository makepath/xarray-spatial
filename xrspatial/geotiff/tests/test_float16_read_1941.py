"""Regression tests for issue #1941.

External GeoTIFFs that store IEEE half-precision floats (``BitsPerSample
=16`` + ``SampleFormat=3``) used to raise ``ValueError("Unsupported
BitsPerSample=16, SampleFormat=3")`` from ``tiff_dtype_to_numpy``. The
writer auto-promotes float16 inputs to float32 before encoding, so the
write side could not produce such a file, but reads from rasterio /
GDAL / tifffile-produced files broke read-parity.

The fix:

* ``tiff_dtype_to_numpy(16, 3)`` returns ``np.float32`` (symmetric with
  the writer's auto-promotion).
* A new ``tiff_storage_dtype`` returns ``np.float16`` for the same key
  so the byte-view in ``_decode_strip_or_tile`` reads the raw 2-byte
  samples correctly before casting to float32.
* The GPU paths fall back to CPU decode when bps != dtype.itemsize * 8,
  matching the existing stripped-layout fallback.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_geotiff_dask
from xrspatial.geotiff._dtypes import (
    SAMPLE_FORMAT_FLOAT,
    SAMPLE_FORMAT_INT,
    SAMPLE_FORMAT_UINT,
    tiff_dtype_to_numpy,
    tiff_storage_dtype,
)


class TestDtypeMap:
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
        result = read_geotiff_dask(str(path), chunks=2)
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


class TestRegressionGuards:
    """The promotion did not change non-float16 behaviour."""

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
