"""Dask streaming write coverage for degenerate raster shapes and NaN / Inf.

The eager numpy write path (``test_edge_cases.py``) covers 1x1, 1xN, Nx1
rasters end-to-end plus all-NaN, all-Inf, and -Inf inputs. The
``write_geotiff_gpu`` path got the same shapes via the test coverage
sweep pass 5 (``test_degenerate_shapes_backends_2026_05_11.py``). The
dask streaming write path through ``to_geotiff`` on a dask-backed
DataArray (#1084) had no matching coverage: ``test_streaming_write.py``
hits 100x100 with a NaN block and a 2x2 ``test_small_raster`` but
nothing single-pixel-row / single-pixel-column, nothing all-NaN, and
nothing Inf / -Inf.

A regression in the dask streaming tile-row segmenter (#1485) on a
1-pixel-tall raster, or in the streaming nodata-mask coercion on an
all-NaN chunk, would not surface from any other path. Both of those
code branches are reached only when the input is a dask-backed
DataArray.

Pass 14 (2026-05-15) closes the gap:

* Cat 3 HIGH -- 1x1, 1xN, Nx1 round-trips through the dask streaming
  writer with chunk sizes that match the raster shape and chunk sizes
  smaller than the raster (so the chunk boundary genuinely splits the
  array).
* Cat 2 HIGH -- all-NaN dask streaming write with a finite nodata
  sentinel: the writer must mask every NaN to the sentinel during
  streaming (the eager path's equivalent is
  ``test_edge_cases.TestNanAndInfHandling.test_all_nan``).
* Cat 2 MEDIUM -- mixed NaN / +Inf / -Inf dask streaming write: +Inf
  and -Inf are valid IEEE-754 float values and must round-trip
  bit-exactly through the streaming pipeline. Only NaN is treated as
  nodata.
* Cat 2 MEDIUM -- all-Inf and all -Inf dask streaming writes.
* Cat 4 MEDIUM -- ``predictor=3`` (floating-point predictor) on a
  small dask raster: the streaming write path threads ``predictor=``
  through to each tile-row encode, and the float predictor branch had
  no direct streaming-write coverage (``test_streaming_write.py``
  covers ``predictor=True`` only).
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


# ---------------------------------------------------------------------------
# Cat 3: 1x1, 1xN, Nx1 dask streaming writes
# ---------------------------------------------------------------------------


class TestStreamingWrite1x1:
    """A single-pixel dask raster must round-trip through the streaming writer."""

    def test_1x1_chunk_matches_shape(self, tmp_path):
        arr = np.array([[42.0]], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 1})
        path = str(tmp_path / '1x1_a.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert result.shape == (1, 1)
        assert result.values[0, 0] == pytest.approx(42.0)

    def test_1x1_with_nodata_attr(self, tmp_path):
        """``attrs['nodata']`` must round-trip even for a 1x1 raster."""
        arr = np.array([[7.5]], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0}).chunk({'y': 1, 'x': 1})
        path = str(tmp_path / '1x1_nodata.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert result.shape == (1, 1)
        assert result.values[0, 0] == pytest.approx(7.5)
        assert result.attrs.get('nodata') == pytest.approx(-9999.0)

    def test_1x1_uint16(self, tmp_path):
        arr = np.array([[255]], dtype=np.uint16)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 1})
        path = str(tmp_path / '1x1_u16.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert result.shape == (1, 1)
        assert int(result.values[0, 0]) == 255


class TestStreamingWrite1xN:
    """A 1-pixel-tall raster exercises the single-tile-row streaming path."""

    def test_1xN_single_chunk(self, tmp_path):
        arr = np.arange(10, dtype=np.float32).reshape(1, 10)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 10})
        path = str(tmp_path / '1xN_a.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_1xN_chunks_split_columns(self, tmp_path):
        """Chunk grid splits the row into multiple column-chunks."""
        arr = np.arange(20, dtype=np.float32).reshape(1, 20)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 7})
        path = str(tmp_path / '1xN_b.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_1xN_wide_segmented_by_buffer(self, tmp_path):
        """Wide single row segmented by streaming_buffer_bytes (#1485)."""
        arr = np.arange(64, dtype=np.float32).reshape(1, 64)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 16})
        path = str(tmp_path / '1xN_seg.tif')
        # Tiny streaming buffer so the segmenter splits the tile-row.
        to_geotiff(da, path, tile_size=16,
                   streaming_buffer_bytes=1)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)


class TestStreamingWriteNx1:
    """A 1-pixel-wide raster exercises the column-degenerate streaming path."""

    def test_Nx1_single_chunk(self, tmp_path):
        arr = np.arange(10, dtype=np.float32).reshape(10, 1)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 10, 'x': 1})
        path = str(tmp_path / 'Nx1_a.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_Nx1_chunks_split_rows(self, tmp_path):
        """Chunk grid splits the column into multiple row-chunks."""
        arr = np.arange(20, dtype=np.float32).reshape(20, 1)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 7, 'x': 1})
        path = str(tmp_path / 'Nx1_b.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)


# ---------------------------------------------------------------------------
# Cat 2: NaN / Inf dask streaming writes
# ---------------------------------------------------------------------------


class TestStreamingWriteAllNan:
    """All-NaN dask raster must mask every pixel to the nodata sentinel."""

    def test_all_nan_with_sentinel(self, tmp_path):
        arr = np.full((8, 8), np.nan, dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0}).chunk({'y': 4, 'x': 4})
        path = str(tmp_path / 'allnan.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        # Every pixel must round-trip back to NaN (sentinel -> NaN on read).
        assert np.isnan(result.values).all()
        # Sentinel must be preserved in attrs.
        assert result.attrs.get('nodata') == pytest.approx(-9999.0)

    def test_all_nan_default_nodata(self, tmp_path):
        """``attrs['nodata']`` omitted -- the streaming writer must still
        accept the all-NaN input. The reader cannot mask without a
        sentinel so the float NaN survives in the file."""
        arr = np.full((4, 4), np.nan, dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 2, 'x': 2})
        path = str(tmp_path / 'allnan_nosen.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert np.isnan(result.values).all()


class TestStreamingWriteMixedNanInf:
    """Mixed NaN / +Inf / -Inf in a single dask raster.

    The streaming writer must (a) replace NaN with the nodata sentinel,
    (b) leave +Inf and -Inf untouched (they are valid IEEE-754 floats).
    """

    def test_mixed_nan_plus_minus_inf(self, tmp_path):
        arr = np.array([
            [1.0, np.nan, 3.0, 4.0],
            [np.inf, 6.0, -np.inf, 8.0],
            [9.0, 10.0, np.nan, 12.0],
            [13.0, np.inf, 15.0, -np.inf],
        ], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0}).chunk({'y': 2, 'x': 2})
        path = str(tmp_path / 'mixed.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        # NaN positions round-trip to NaN.
        assert np.isnan(result.values[0, 1])
        assert np.isnan(result.values[2, 2])
        # +Inf and -Inf round-trip verbatim.
        assert result.values[1, 0] == np.inf
        assert result.values[3, 1] == np.inf
        assert result.values[1, 2] == -np.inf
        assert result.values[3, 3] == -np.inf
        # Finite values stay finite.
        assert result.values[0, 0] == pytest.approx(1.0)
        assert result.values[2, 0] == pytest.approx(9.0)


class TestStreamingWriteAllInf:
    """All +Inf and all -Inf dask streaming writes.

    +Inf and -Inf are valid IEEE-754 floats; the streaming writer
    should pass them through unchanged. The reader keeps Inf as Inf
    because the nodata mask only matches the sentinel value, not Inf.
    """

    def test_all_plus_inf(self, tmp_path):
        arr = np.full((4, 4), np.inf, dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 2, 'x': 2})
        path = str(tmp_path / 'allposinf.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert (result.values == np.inf).all()

    def test_all_minus_inf(self, tmp_path):
        arr = np.full((4, 4), -np.inf, dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 2, 'x': 2})
        path = str(tmp_path / 'allneginf.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert (result.values == -np.inf).all()


# ---------------------------------------------------------------------------
# Cat 4: predictor=3 floating-point predictor through dask streaming write
# ---------------------------------------------------------------------------


class TestStreamingWriteFloatPredictor:
    """``predictor=3`` (TIFF float predictor) on a small dask raster.

    The streaming writer threads ``predictor=`` through to every tile-row
    encode. ``test_streaming_write.py`` covers ``predictor=True`` (=2)
    only; the float predictor 3 branch lacked direct streaming
    coverage. Verify lossless float32 round-trip plus the dtype-guard
    on int input.
    """

    def test_predictor3_float32_round_trip(self, tmp_path):
        rng = np.random.default_rng(2026_05_15)
        arr = rng.random((40, 40), dtype=np.float32) * 100.0
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 16, 'x': 16})
        path = str(tmp_path / 'pred3_f32.tif')
        to_geotiff(da, path, compression='deflate', predictor=3,
                   tile_size=16)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_predictor3_float64_round_trip(self, tmp_path):
        rng = np.random.default_rng(2026_05_15)
        arr = rng.random((32, 32), dtype=np.float64) * 100.0
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 16, 'x': 16})
        path = str(tmp_path / 'pred3_f64.tif')
        to_geotiff(da, path, compression='deflate', predictor=3,
                   tile_size=16)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_predictor3_int_input_rejected(self, tmp_path):
        """``predictor=3`` requires float dtype; int input must raise."""
        arr = np.arange(32 * 32, dtype=np.int32).reshape(32, 32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 16, 'x': 16})
        path = str(tmp_path / 'pred3_i32.tif')
        with pytest.raises(ValueError, match='predictor'):
            to_geotiff(da, path, compression='deflate', predictor=3,
                       tile_size=16)
