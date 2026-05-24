"""Streaming dask-write coverage for codecs not in the original #1084 matrix.

The original streaming-write test suite (``test_streaming_write.py``)
covers ``none / deflate / lzw / zstd`` round-trips and the predictor flag.
The LERC, LZ4, and packbits codecs route through the same
``_compress_block`` helper as the eager writer but had no
dask-streaming-write coverage before today. This file pins:

* Dask streaming write + LERC (lossless and lossy ``max_z_error``)
  produces identical output to the eager writer.
* Dask streaming write + LZ4 round-trips a float32 raster.
* Dask streaming write + packbits round-trips a uint8 raster. packbits
  operates on the raw byte stream so any dtype is technically supported;
  uint8 is the variant exercised here and the one packbits was designed
  for (run-length encoding of byte runs).
* COG output with ``overview_resampling='cubic'`` round-trips through
  scipy.ndimage.zoom (the only overview method that takes a separate
  code path in ``_block_reduce_2d``).

Coverage-gap sweep 2026-05-11: closes the dask-streaming codec gap
remaining after PR #1565.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("dask")

from xrspatial.geotiff import open_geotiff, to_geotiff  # noqa: E402
from xrspatial.geotiff._compression import LERC_AVAILABLE, LZ4_AVAILABLE  # noqa: E402

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
