"""Edge case tests for invalid, corrupt, and boundary-condition inputs."""
from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._compression import (
    COMPRESSION_DEFLATE,
    COMPRESSION_LZW,
    COMPRESSION_NONE,
    compress,
    decompress,
    deflate_decompress,
    lzw_compress,
    lzw_decompress,
)
from xrspatial.geotiff._dtypes import numpy_to_tiff_dtype, tiff_dtype_to_numpy
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import write


# -----------------------------------------------------------------------
# Writer: invalid inputs
# -----------------------------------------------------------------------

class TestWriteInvalidInputs:
    """Writer should reject or gracefully handle bad inputs."""

    def test_4d_array(self, tmp_path):
        arr = np.zeros((2, 3, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            to_geotiff(arr, str(tmp_path / 'bad.tif'))

    def test_1d_array(self, tmp_path):
        arr = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D"):
            to_geotiff(arr, str(tmp_path / 'bad.tif'))

    def test_0d_scalar(self, tmp_path):
        arr = np.float32(42.0)
        with pytest.raises(ValueError, match="Expected 2D"):
            to_geotiff(arr, str(tmp_path / 'bad.tif'))

    def test_unsupported_compression(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        # ``to_geotiff`` validates ``compression`` up-front (#1488). The
        # earlier "Unsupported compression" message comes from the deeper
        # ``_compression_tag`` and is now only seen when callers reach
        # the writer directly. Both phrasings are acceptable.
        with pytest.raises(ValueError, match="(Unknown|Unsupported) compression"):
            to_geotiff(arr, str(tmp_path / 'bad.tif'), compression='webp')

    def test_complex_dtype(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.complex64)
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            to_geotiff(arr, str(tmp_path / 'bad.tif'))

    def test_bool_dtype_auto_promoted(self, tmp_path):
        """Bool arrays are auto-promoted to uint8."""
        arr = np.array([[True, False], [False, True]])
        path = str(tmp_path / 'bool.tif')
        to_geotiff(arr, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr.astype(np.uint8))


# -----------------------------------------------------------------------
# Writer: boundary-condition data values
# -----------------------------------------------------------------------

class TestWriteSpecialValues:
    """Writer should handle NaN, Inf, and extreme values."""

    def test_all_nan(self, tmp_path):
        arr = np.full((4, 4), np.nan, dtype=np.float32)
        path = str(tmp_path / 'all_nan.tif')
        write(arr, path, compression='none', tiled=False)

        result, _ = read_to_array(path)
        assert np.all(np.isnan(result))

    def test_nan_and_inf(self, tmp_path):
        arr = np.array([[np.nan, np.inf], [-np.inf, 0.0]], dtype=np.float32)
        path = str(tmp_path / 'nan_inf.tif')
        write(arr, path, compression='none', tiled=False)

        result, _ = read_to_array(path)
        assert np.isnan(result[0, 0])
        assert np.isposinf(result[0, 1])
        assert np.isneginf(result[1, 0])
        assert result[1, 1] == 0.0

    def test_nan_with_deflate(self, tmp_path):
        arr = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)
        path = str(tmp_path / 'nan_deflate.tif')
        write(arr, path, compression='deflate', tiled=False)

        result, _ = read_to_array(path)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 1])
        assert result[0, 1] == 1.0
        assert result[1, 0] == 2.0

    def test_nan_with_lzw(self, tmp_path):
        arr = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)
        path = str(tmp_path / 'nan_lzw.tif')
        write(arr, path, compression='lzw', tiled=False)

        result, _ = read_to_array(path)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 1])

    def test_float32_extremes(self, tmp_path):
        finfo = np.finfo(np.float32)
        arr = np.array([[finfo.max, finfo.min],
                        [finfo.tiny, -finfo.tiny]], dtype=np.float32)
        path = str(tmp_path / 'extremes.tif')
        write(arr, path, compression='deflate', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_uint16_full_range(self, tmp_path):
        arr = np.array([[0, 65535], [1, 65534]], dtype=np.uint16)
        path = str(tmp_path / 'uint16_range.tif')
        write(arr, path, compression='none', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_int16_negative(self, tmp_path):
        arr = np.array([[-32768, 32767], [-1, 0]], dtype=np.int16)
        path = str(tmp_path / 'int16.tif')
        write(arr, path, compression='none', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_all_zeros(self, tmp_path):
        arr = np.zeros((8, 8), dtype=np.float32)
        path = str(tmp_path / 'zeros.tif')
        write(arr, path, compression='deflate', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_all_same_value(self, tmp_path):
        arr = np.full((16, 16), 42.5, dtype=np.float32)
        path = str(tmp_path / 'constant.tif')
        write(arr, path, compression='lzw', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)


# -----------------------------------------------------------------------
# Writer: boundary-condition shapes
# -----------------------------------------------------------------------

class TestWriteBoundaryShapes:
    """Test extreme and non-aligned image dimensions."""

    def test_single_pixel(self, tmp_path):
        arr = np.array([[42.0]], dtype=np.float32)
        path = str(tmp_path / '1x1.tif')
        write(arr, path, compression='none', tiled=False)

        result, _ = read_to_array(path)
        assert result.shape == (1, 1)
        assert result[0, 0] == 42.0

    def test_single_row(self, tmp_path):
        arr = np.arange(10, dtype=np.float32).reshape(1, 10)
        path = str(tmp_path / '1x10.tif')
        write(arr, path, compression='deflate', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_single_column(self, tmp_path):
        arr = np.arange(10, dtype=np.float32).reshape(10, 1)
        path = str(tmp_path / '10x1.tif')
        write(arr, path, compression='deflate', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_non_tile_aligned(self, tmp_path):
        """Image dimensions not divisible by tile size."""
        arr = np.arange(35, dtype=np.float32).reshape(5, 7)
        path = str(tmp_path / 'non_aligned.tif')
        write(arr, path, compression='none', tiled=True, tile_size=4)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_tile_larger_than_image(self, tmp_path):
        """Tile size larger than the image."""
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        path = str(tmp_path / 'big_tile.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=256)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_odd_dimensions_all_compressions(self, tmp_path):
        """Non-power-of-2 dimensions with every compression."""
        arr = np.random.RandomState(99).rand(13, 17).astype(np.float32)
        for comp in ['none', 'deflate', 'lzw']:
            path = str(tmp_path / f'odd_{comp}.tif')
            write(arr, path, compression=comp, tiled=False)
            result, _ = read_to_array(path)
            np.testing.assert_array_equal(result, arr)

    def test_very_wide_single_row_tiled(self, tmp_path):
        """1 row, many columns, tiled layout."""
        arr = np.arange(500, dtype=np.float32).reshape(1, 500)
        path = str(tmp_path / 'wide.tif')
        write(arr, path, compression='none', tiled=True, tile_size=64)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_very_tall_single_column_tiled(self, tmp_path):
        """Many rows, 1 column, tiled layout."""
        arr = np.arange(500, dtype=np.float32).reshape(500, 1)
        path = str(tmp_path / 'tall.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=64)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_predictor_single_pixel(self, tmp_path):
        """Predictor on a 1x1 image (no pixels to difference against)."""
        arr = np.array([[7.5]], dtype=np.float32)
        path = str(tmp_path / 'pred_1x1.tif')
        write(arr, path, compression='deflate', tiled=False, predictor=True)

        result, _ = read_to_array(path)
        assert result[0, 0] == pytest.approx(7.5)


# -----------------------------------------------------------------------
# Reader: corrupt / truncated files
# -----------------------------------------------------------------------

class TestReadCorruptFiles:
    """Reader should raise clear errors on malformed input."""

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / 'empty.tif')
        with open(path, 'wb') as f:
            pass  # 0 bytes
        with pytest.raises((ValueError, Exception)):
            read_to_array(path)

    def test_too_short_for_header(self, tmp_path):
        path = str(tmp_path / 'short.tif')
        with open(path, 'wb') as f:
            f.write(b'II\x2a\x00')  # only 4 bytes, need 8
        with pytest.raises((ValueError, Exception)):
            read_to_array(path)

    def test_random_bytes(self, tmp_path):
        path = str(tmp_path / 'random.tif')
        with open(path, 'wb') as f:
            f.write(b'\xde\xad\xbe\xef' * 100)
        with pytest.raises(ValueError, match="Invalid TIFF"):
            read_to_array(path)

    def test_valid_header_but_no_ifd(self, tmp_path):
        """TIFF header pointing to IFD beyond file end."""
        path = str(tmp_path / 'no_ifd.tif')
        # Valid LE TIFF header pointing to offset 99999 which doesn't exist
        with open(path, 'wb') as f:
            f.write(b'II')
            f.write(struct.pack('<H', 42))
            f.write(struct.pack('<I', 99999))
        with pytest.raises((ValueError, Exception)):
            read_to_array(path)

    def test_truncated_pixel_data(self, tmp_path):
        """Valid TIFF structure but pixel data truncated."""
        from .conftest import make_minimal_tiff
        full_data = make_minimal_tiff(4, 4, np.dtype('float32'))
        # Truncate: remove last 32 bytes of pixel data
        truncated = full_data[:-32]
        path = str(tmp_path / 'truncated.tif')
        with open(path, 'wb') as f:
            f.write(truncated)
        # Should either raise or return partial/garbage data (not crash)
        try:
            arr, _ = read_to_array(path)
            # If it doesn't raise, the shape should still be correct
            assert arr.shape == (4, 4)
        except (ValueError, struct.error, IndexError):
            pass  # acceptable: raises on truncated data

    def test_ifd_offset_zero(self, tmp_path):
        """TIFF header with first IFD offset = 0."""
        path = str(tmp_path / 'zero_ifd.tif')
        with open(path, 'wb') as f:
            f.write(b'II')
            f.write(struct.pack('<H', 42))
            f.write(struct.pack('<I', 0))
        with pytest.raises((ValueError, Exception)):
            read_to_array(path)

    def test_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            read_to_array('/nonexistent/path/fake.tif')


# -----------------------------------------------------------------------
# Reader: corrupt compressed data
# -----------------------------------------------------------------------

class TestReadCorruptCompression:
    """Decompressor should raise on corrupt streams, not segfault."""

    def test_corrupt_deflate(self):
        with pytest.raises((zlib.error, Exception)):
            deflate_decompress(b'\x00\x01\x02\x03\x04')

    def test_truncated_deflate(self):
        valid = zlib.compress(b'hello world')
        with pytest.raises((zlib.error, Exception)):
            deflate_decompress(valid[:5])

    def test_corrupt_lzw(self):
        """Random bytes fed to LZW decoder should not crash."""
        bad_data = bytes(range(256))
        # Should not segfault; may return garbage or short output
        result = lzw_decompress(bad_data, 1000)
        assert isinstance(result, np.ndarray)

    def test_lzw_zero_length(self):
        result = lzw_decompress(b'', 0)
        assert len(result) == 0

    def test_decompress_unsupported_code(self):
        with pytest.raises(ValueError, match="Unsupported compression"):
            decompress(b'data', 999)


# -----------------------------------------------------------------------
# Header parser: edge cases
# -----------------------------------------------------------------------

class TestHeaderEdgeCases:

    def test_invalid_byte_order(self):
        with pytest.raises(ValueError, match="Invalid TIFF byte order"):
            parse_header(b'XX\x2a\x00\x08\x00\x00\x00')

    def test_invalid_magic(self):
        with pytest.raises(ValueError, match="Invalid TIFF magic"):
            parse_header(b'II\x99\x00\x08\x00\x00\x00')

    def test_only_two_bytes(self):
        with pytest.raises(ValueError, match="Not enough data"):
            parse_header(b'II')

    def test_bigtiff_header_too_short(self):
        # BigTIFF magic (43) but not enough bytes for full header
        data = b'II' + struct.pack('<H', 43) + b'\x00\x00'
        with pytest.raises(ValueError, match="Not enough data"):
            parse_header(data)


# -----------------------------------------------------------------------
# Dtype edge cases
# -----------------------------------------------------------------------

class TestDtypeEdgeCases:

    def test_unsupported_bits_per_sample(self):
        with pytest.raises(ValueError, match="Unsupported BitsPerSample"):
            tiff_dtype_to_numpy(3, 1)  # 3-bit not supported

    def test_sub_byte_dtypes_supported(self):
        """1, 2, 4, and 12-bit map to uint8/uint16."""
        assert tiff_dtype_to_numpy(1, 1) == np.dtype('uint8')
        assert tiff_dtype_to_numpy(4, 1) == np.dtype('uint8')
        assert tiff_dtype_to_numpy(12, 1) == np.dtype('uint16')

    def test_unsupported_sample_format(self):
        with pytest.raises(ValueError, match="Unsupported BitsPerSample"):
            tiff_dtype_to_numpy(32, 99)  # format 99 doesn't exist

    def test_complex_dtype_write(self):
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            numpy_to_tiff_dtype(np.dtype('complex128'))

    def test_float16_auto_promoted_to_float32(self):
        """float16 on disk (bps=16 + SampleFormat=3) is exposed as float32.

        The writer auto-promotes float16 inputs to float32 before
        encoding, and the reader matches that contract: a file with
        bps=16 + SampleFormat=3 (an externally produced half-precision
        TIFF) returns float32 to the user. See issue #1941 for the
        original read-side asymmetry.
        """
        assert tiff_dtype_to_numpy(16, 3) == np.float32


# -----------------------------------------------------------------------
# Public API: edge cases
# -----------------------------------------------------------------------

class TestPublicAPIEdgeCases:

    def test_write_list_input(self, tmp_path):
        """Python list should be converted via np.asarray."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        path = str(tmp_path / 'from_list.tif')
        to_geotiff(data, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, np.array(data, dtype=np.float64))

    def test_write_integer_list(self, tmp_path):
        """Integer Python list."""
        data = [[1, 2], [3, 4]]
        path = str(tmp_path / 'int_list.tif')
        to_geotiff(data, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, np.array(data))

    def test_read_nonexistent(self):
        with pytest.raises((FileNotFoundError, OSError)):
            open_geotiff('/tmp/nonexistent_file_abc123.tif')

    def test_write_dataarray_no_coords(self, tmp_path):
        """DataArray with dimension coords but no explicit y/x."""
        da = xr.DataArray(np.ones((3, 3), dtype=np.float32), dims=['y', 'x'])
        path = str(tmp_path / 'no_coords.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, da.values)

    def test_write_f_contiguous_array(self, tmp_path):
        """Fortran-order array (column-major)."""
        arr = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
        assert arr.flags['F_CONTIGUOUS']
        path = str(tmp_path / 'f_order.tif')
        to_geotiff(arr, path, compression='deflate')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_write_non_contiguous_slice(self, tmp_path):
        """Non-contiguous array (e.g. every other row)."""
        base = np.arange(64, dtype=np.float32).reshape(8, 8)
        sliced = base[::2, ::2]  # 4x4, non-contiguous
        assert not sliced.flags['C_CONTIGUOUS']
        path = str(tmp_path / 'strided.tif')
        to_geotiff(sliced, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, sliced)

    def test_crs_in_attrs_with_coords(self, tmp_path):
        """CRS from DataArray attrs round-trips through the file."""
        y = np.linspace(45.0, 44.0, 4)
        x = np.linspace(-120.0, -119.0, 4)
        da = xr.DataArray(np.ones((4, 4), dtype=np.float32),
                          dims=['y', 'x'], coords={'y': y, 'x': x},
                          attrs={'crs': 4326})
        path = str(tmp_path / 'crs_coords.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.attrs['crs'] == 4326

    def test_crs_kwarg_no_coords(self, tmp_path):
        """Explicit crs= kwarg preserved even without coordinates."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'crs_kwarg.tif')
        to_geotiff(arr, path, crs=32610, compression='none')

        result = open_geotiff(path)
        assert result.attrs['crs'] == 32610

    def test_crs_and_nodata_no_coords(self, tmp_path):
        """Both CRS and nodata preserved without coordinates."""
        arr = np.array([[1.0, -9999.0], [-9999.0, 2.0]], dtype=np.float32)
        path = str(tmp_path / 'both.tif')
        to_geotiff(arr, path, crs=4326, nodata=-9999.0, compression='none')

        result = open_geotiff(path)
        assert result.attrs['crs'] == 4326
        assert np.isnan(result.values[0, 1])
        assert result.values[1, 1] == 2.0

    def test_crs_projected(self, tmp_path):
        """Projected CRS (UTM) round-trips."""
        y = np.linspace(4500000.0, 4499000.0, 4)
        x = np.linspace(500000.0, 501000.0, 4)
        da = xr.DataArray(np.ones((4, 4), dtype=np.float32),
                          dims=['y', 'x'], coords={'y': y, 'x': x},
                          attrs={'crs': 32610})
        path = str(tmp_path / 'utm.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.attrs['crs'] == 32610

    def test_pixel_is_point_round_trip(self, tmp_path):
        """PixelIsPoint raster_type preserves origin correctly."""
        # Use truly uniform coords; hand-typed fractional decimals had
        # ~3.6e-3 relative deviation between steps and silently produced
        # a wrong transform before issue #1720 enforced regularity.
        y = np.linspace(41.0, 40.999167, 4)
        x = np.linspace(-75.0, -74.999167, 4)
        da = xr.DataArray(
            np.arange(16, dtype=np.float32).reshape(4, 4),
            dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 4326, 'raster_type': 'point'},
        )
        path = str(tmp_path / 'point.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.attrs.get('raster_type') == 'point'
        assert result.attrs['crs'] == 4326
        # Coordinates should match exactly -- origin IS the pixel center
        np.testing.assert_array_almost_equal(result.coords['x'].values, x, decimal=4)
        np.testing.assert_array_almost_equal(result.coords['y'].values, y, decimal=4)

    def test_pixel_is_area_default(self, tmp_path):
        """Default raster_type is PixelIsArea (no attr set)."""
        y = np.linspace(45.0, 44.0, 4)
        x = np.linspace(-120.0, -119.0, 4)
        da = xr.DataArray(np.ones((4, 4), dtype=np.float32),
                          dims=['y', 'x'], coords={'y': y, 'x': x},
                          attrs={'crs': 4326})
        path = str(tmp_path / 'area.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert 'raster_type' not in result.attrs  # default, not stored
        np.testing.assert_array_almost_equal(result.coords['x'].values, x, decimal=6)
        np.testing.assert_array_almost_equal(result.coords['y'].values, y, decimal=6)

    def test_no_crs_no_nodata(self, tmp_path):
        """No CRS or nodata -- attrs should be empty."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'bare.tif')
        to_geotiff(arr, path, compression='none')

        result = open_geotiff(path)
        assert 'crs' not in result.attrs
        assert 'nodata' not in result.attrs

    def test_nodata_nan_in_attrs(self, tmp_path):
        """DataArray with nodata=NaN should round-trip."""
        arr = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'], attrs={'nodata': np.nan})
        path = str(tmp_path / 'nodata_nan.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert np.isnan(result.values[0, 1])
        assert result.values[0, 0] == 1.0

    def test_nodata_string_numeric(self, tmp_path):
        """Nodata value stored as string in attrs (common from rioxarray)."""
        arr = np.array([[1.0, -9999.0], [-9999.0, 2.0]], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0})
        path = str(tmp_path / 'nodata_str.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        # The nodata sentinel should be masked to NaN on read
        assert np.isnan(result.values[0, 1])
        assert np.isnan(result.values[1, 0])
        assert result.values[0, 0] == 1.0
        assert result.values[1, 1] == 2.0


# -----------------------------------------------------------------------
# Compression: boundary conditions
# -----------------------------------------------------------------------

class TestCompressionBoundary:

    def test_compress_empty(self):
        for comp in [COMPRESSION_NONE, COMPRESSION_DEFLATE, COMPRESSION_LZW]:
            result = compress(b'', comp)
            if comp == COMPRESSION_NONE:
                assert result == b''
            else:
                assert isinstance(result, bytes)

    def test_decompress_empty_none(self):
        result = decompress(b'', COMPRESSION_NONE)
        assert len(result) == 0

    def test_single_byte_lzw(self):
        data = b'\x42'
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, 1)
        assert decompressed.tobytes() == data

    def test_single_byte_deflate(self):
        data = b'\x42'
        compressed = zlib.compress(data)
        assert deflate_decompress(compressed) == data

    def test_lzw_round_trip_all_byte_values(self):
        """All 256 byte values."""
        data = bytes(range(256))
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, 256)
        assert decompressed.tobytes() == data

    def test_lzw_round_trip_highly_repetitive(self):
        """Maximally compressible data."""
        data = b'\x00' * 100000
        compressed = lzw_compress(data)
        assert len(compressed) < len(data)  # should actually compress
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_deflate_round_trip_random(self):
        """Incompressible random data."""
        rng = np.random.RandomState(42)
        data = bytes(rng.randint(0, 256, size=10000, dtype=np.uint8))
        compressed = zlib.compress(data)
        assert deflate_decompress(compressed) == data


# -----------------------------------------------------------------------
# Write + read round-trip: all dtype x compression combos
# -----------------------------------------------------------------------

class TestRoundTripMatrix:
    """Systematic coverage of every dtype x compression x layout combination."""

    @pytest.mark.parametrize("dtype", [
        np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32,
        np.float32, np.float64,
    ])
    @pytest.mark.parametrize("compression", ['none', 'deflate', 'lzw'])
    def test_dtype_compression_stripped(self, dtype, compression, tmp_path):
        arr = np.arange(16, dtype=dtype).reshape(4, 4)
        path = str(tmp_path / f'{dtype.__name__}_{compression}.tif')
        write(arr, path, compression=compression, tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.uint16])
    def test_dtype_tiled(self, dtype, tmp_path):
        arr = np.arange(64, dtype=dtype).reshape(8, 8)
        path = str(tmp_path / f'tiled_{dtype.__name__}.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=4)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)
