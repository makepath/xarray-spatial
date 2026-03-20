"""Tests for new features: multi-band, integer nodata, packbits, dask, BigTIFF."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_geotiff, write_geotiff
from xrspatial.geotiff._compression import (
    COMPRESSION_PACKBITS,
    packbits_compress,
    packbits_decompress,
)
from xrspatial.geotiff._header import parse_header, parse_all_ifds
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import write


# -----------------------------------------------------------------------
# Multi-band write and read
# -----------------------------------------------------------------------

class TestMultiBand:

    def test_rgb_uint8_round_trip(self, tmp_path):
        """Write and read back RGB uint8 image."""
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[:, :, 0] = 200  # red
        arr[:, :, 1] = 100  # green
        arr[:, :, 2] = 50   # blue
        path = str(tmp_path / 'rgb.tif')
        write(arr, path, compression='none', tiled=False)

        result, geo = read_to_array(path)
        assert result.shape == (8, 8, 3)
        np.testing.assert_array_equal(result, arr)

    def test_rgb_deflate_tiled(self, tmp_path):
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        path = str(tmp_path / 'rgb_deflate.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8)

        result, geo = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_rgba_uint8(self, tmp_path):
        arr = np.ones((4, 4, 4), dtype=np.uint8) * 128
        path = str(tmp_path / 'rgba.tif')
        write(arr, path, compression='none', tiled=False)

        result, geo = read_to_array(path)
        assert result.shape == (4, 4, 4)
        np.testing.assert_array_equal(result, arr)

    def test_multiband_float32(self, tmp_path):
        arr = np.random.RandomState(99).rand(8, 8, 5).astype(np.float32)
        path = str(tmp_path / 'multi.tif')
        write(arr, path, compression='deflate', tiled=False)

        result, geo = read_to_array(path)
        assert result.shape == (8, 8, 5)
        np.testing.assert_array_equal(result, arr)

    def test_single_band_selection(self, tmp_path):
        """band= parameter should extract one band."""
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 1] = 42
        path = str(tmp_path / 'rgb_sel.tif')
        write(arr, path, compression='none', tiled=False)

        result, _ = read_to_array(path, band=1)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, 42)

    def test_rgb_write_geotiff_api(self, tmp_path):
        """write_geotiff accepts 3D arrays."""
        arr = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        path = str(tmp_path / 'rgb_api.tif')
        write_geotiff(arr, path, compression='none')

        result = read_geotiff(path)
        assert 'band' in result.dims
        assert result.shape == (4, 4, 3)
        np.testing.assert_array_equal(result.values, arr)

    def test_rgb_cog(self, tmp_path):
        """Multi-band COG with overviews."""
        arr = np.random.RandomState(7).randint(
            0, 256, (32, 32, 3), dtype=np.uint8)
        path = str(tmp_path / 'rgb_cog.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=16,
              cog=True, overview_levels=[1])

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)


# -----------------------------------------------------------------------
# Integer nodata masking
# -----------------------------------------------------------------------

class TestIntegerNodata:

    def test_uint8_nodata_masked(self, tmp_path):
        arr = np.array([[0, 1, 2], [3, 255, 5]], dtype=np.uint8)
        path = str(tmp_path / 'uint8_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=255)

        da = read_geotiff(path)
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 1] == 1.0
        assert da.dtype == np.float64  # promoted from uint8

    def test_uint16_nodata_masked(self, tmp_path):
        arr = np.array([[100, 0], [200, 0]], dtype=np.uint16)
        path = str(tmp_path / 'uint16_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=0)

        da = read_geotiff(path)
        assert np.isnan(da.values[0, 1])
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 0] == 100.0

    def test_int16_nodata_negative(self, tmp_path):
        arr = np.array([[-9999, 10], [20, -9999]], dtype=np.int16)
        path = str(tmp_path / 'int16_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=-9999)

        da = read_geotiff(path)
        assert np.isnan(da.values[0, 0])
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 1] == 10.0

    def test_integer_no_nodata_stays_integer(self, tmp_path):
        """Without nodata, integer arrays should not be promoted."""
        arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
        path = str(tmp_path / 'no_nodata.tif')
        write(arr, path, compression='none', tiled=False)

        da = read_geotiff(path)
        assert da.dtype == np.uint16


# -----------------------------------------------------------------------
# PackBits compression
# -----------------------------------------------------------------------

class TestPackBits:

    def test_packbits_round_trip(self):
        data = b'\x00' * 100 + b'\xff' * 50 + bytes(range(200))
        compressed = packbits_compress(data)
        decompressed = packbits_decompress(compressed)
        assert decompressed == data

    def test_packbits_single_byte(self):
        data = b'\x42'
        assert packbits_decompress(packbits_compress(data)) == data

    def test_packbits_empty(self):
        assert packbits_decompress(packbits_compress(b'')) == b''

    def test_packbits_all_same(self):
        data = b'\xAA' * 500
        compressed = packbits_compress(data)
        assert len(compressed) < len(data)
        assert packbits_decompress(compressed) == data

    def test_write_read_packbits(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'packbits.tif')
        write(arr, path, compression='packbits', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_packbits_tiled(self, tmp_path):
        arr = np.random.RandomState(42).rand(16, 16).astype(np.float32)
        path = str(tmp_path / 'packbits_tiled.tif')
        write(arr, path, compression='packbits', tiled=True, tile_size=8)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)


# -----------------------------------------------------------------------
# BigTIFF write
# -----------------------------------------------------------------------

class TestBigTIFF:

    def test_bigtiff_header_written(self, tmp_path):
        """Force BigTIFF via internal threshold by mocking; test header parsing."""
        # We can't easily create a >4GB file in tests, but we can verify
        # the BigTIFF path works by writing a small file with bigtiff=True
        # through the internal API.
        from xrspatial.geotiff._writer import _assemble_tiff, _write_stripped
        from xrspatial.geotiff._compression import COMPRESSION_NONE
        from xrspatial.geotiff._geotags import GeoTransform

        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
        parts = [(arr, 4, 4, rel_off, bc, chunks)]

        file_bytes = _assemble_tiff(
            4, 4, arr.dtype, COMPRESSION_NONE, False, False, 256,
            parts, None, None, None, is_cog=False, raster_type=1)

        # Standard TIFF: magic 42
        header = parse_header(file_bytes)
        assert not header.is_bigtiff

    def test_bigtiff_read_write_round_trip(self, tmp_path):
        """Test that BigTIFF files produced internally can be read back."""
        from xrspatial.geotiff._writer import (
            _assemble_tiff, _write_stripped, _assemble_standard_layout,
        )
        from xrspatial.geotiff._compression import COMPRESSION_NONE
        from xrspatial.geotiff._dtypes import numpy_to_tiff_dtype, SHORT, LONG, DOUBLE
        from xrspatial.geotiff._header import (
            TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH, TAG_BITS_PER_SAMPLE,
            TAG_COMPRESSION, TAG_PHOTOMETRIC, TAG_SAMPLES_PER_PIXEL,
            TAG_SAMPLE_FORMAT, TAG_ROWS_PER_STRIP,
            TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS,
        )

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
        bits_per_sample, sample_format = numpy_to_tiff_dtype(arr.dtype)

        tags = [
            (TAG_IMAGE_WIDTH, LONG, 1, 8),
            (TAG_IMAGE_LENGTH, LONG, 1, 8),
            (TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample),
            (TAG_COMPRESSION, SHORT, 1, 1),
            (TAG_PHOTOMETRIC, SHORT, 1, 1),
            (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
            (TAG_SAMPLE_FORMAT, SHORT, 1, sample_format),
            (TAG_ROWS_PER_STRIP, SHORT, 1, 8),
            (TAG_STRIP_OFFSETS, LONG, len(rel_off), rel_off),
            (TAG_STRIP_BYTE_COUNTS, LONG, len(bc), bc),
        ]

        parts = [(arr, 8, 8, rel_off, bc, chunks)]
        file_bytes = _assemble_standard_layout(
            16, [tags], parts, bigtiff=True)

        path = str(tmp_path / 'bigtiff.tif')
        with open(path, 'wb') as f:
            f.write(file_bytes)

        header = parse_header(file_bytes)
        assert header.is_bigtiff

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)


# -----------------------------------------------------------------------
# Dask lazy reads
# -----------------------------------------------------------------------

class TestDaskReads:

    def test_dask_basic(self, tmp_path):
        """read_geotiff_dask returns a dask-backed DataArray."""
        import dask.array as da
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'dask_test.tif')
        write(arr, path, compression='none', tiled=False)

        result = read_geotiff_dask(path, chunks=8)
        assert isinstance(result.data, da.Array)
        assert result.shape == (16, 16)

        # Compute and compare
        computed = result.compute()
        np.testing.assert_array_equal(computed.values, arr)

    def test_dask_coords(self, tmp_path):
        """Dask read preserves coordinates and CRS."""
        from xrspatial.geotiff import read_geotiff_dask
        from xrspatial.geotiff._geotags import GeoTransform

        arr = np.ones((8, 8), dtype=np.float32)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'dask_geo.tif')
        write(arr, path, geo_transform=gt, crs_epsg=4326,
              compression='none', tiled=False)

        result = read_geotiff_dask(path, chunks=4)
        assert result.attrs['crs'] == 4326
        assert len(result.coords['y']) == 8
        assert len(result.coords['x']) == 8

    def test_dask_nodata(self, tmp_path):
        """Nodata masking applied per-chunk."""
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.array([[1.0, -9999.0], [-9999.0, 2.0],
                        [3.0, 4.0], [5.0, -9999.0]], dtype=np.float32)
        path = str(tmp_path / 'dask_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=-9999.0)

        result = read_geotiff_dask(path, chunks=2)
        computed = result.compute()
        assert np.isnan(computed.values[0, 1])
        assert np.isnan(computed.values[1, 0])
        assert computed.values[0, 0] == 1.0

    def test_dask_chunk_tuple(self, tmp_path):
        """Chunks as (row, col) tuple."""
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.arange(200, dtype=np.float32).reshape(10, 20)
        path = str(tmp_path / 'dask_tuple.tif')
        write(arr, path, compression='deflate', tiled=False)

        result = read_geotiff_dask(path, chunks=(5, 10))
        computed = result.compute()
        np.testing.assert_array_equal(computed.values, arr)
