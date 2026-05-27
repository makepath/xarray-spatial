"""Minimal reader paths: band validation, byte-band, eager read.

Consolidates the reader-side band-validation regression coverage
(formerly ``test_band_validation_1673.py``). The contract is that every
backend rejects out-of-range ``band`` arguments with the same typed
``IndexError`` so callers see consistent diagnostics regardless of
which path they pick.
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning, open_geotiff, to_geotiff
from xrspatial.geotiff import write_vrt as _write_vrt_1810
from xrspatial.geotiff._dtypes import tiff_dtype_to_numpy
from xrspatial.geotiff._geotags import RASTER_PIXEL_IS_POINT, TAG_GEO_ASCII_PARAMS, extract_geo_info
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import (_BytesIOSource, _read_strips, _read_tiles, _read_to_array,
                                       read_to_array)
from xrspatial.geotiff._sources import _FileSource, _mmap_cache
from xrspatial.geotiff._writer import write

from ..conftest import make_minimal_tiff


@pytest.fixture
def multiband_tiff_path(tmp_path):
    """4x6 three-band tiled tiff for band-validation tests."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(72, dtype=np.float32).reshape(4, 6, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            'band': [0, 1, 2],
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'mb_band_validation.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


class TestBandValidationLocal:
    """``read_to_array`` rejects out-of-range band indices."""

    def test_negative_band_rejected(self, multiband_tiff_path):
        """``band=-1`` no longer silently selects the last channel."""
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path
        with pytest.raises(IndexError, match="band=-1 out of range"):
            read_to_array(path, band=-1)

    def test_band_equal_to_samples_rejected(self, multiband_tiff_path):
        """``band=samples_per_pixel`` (off-by-one) raises a typed error."""
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path
        with pytest.raises(IndexError, match="band=3 out of range"):
            read_to_array(path, band=3)

    def test_band_far_above_samples_rejected(self, multiband_tiff_path):
        """A wildly out-of-range band index gives the same typed error."""
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path
        with pytest.raises(IndexError, match="band=103 out of range"):
            read_to_array(path, band=103)

    def test_valid_band_still_works(self, multiband_tiff_path):
        """Valid band indices keep working after the validation guard."""
        from xrspatial.geotiff._reader import read_to_array

        path, arr = multiband_tiff_path
        out, _ = read_to_array(path, band=1)
        np.testing.assert_array_equal(out, arr[:, :, 1])

    def test_band_none_returns_all_bands(self, multiband_tiff_path):
        """``band=None`` still returns the full multi-band array."""
        from xrspatial.geotiff._reader import read_to_array

        path, arr = multiband_tiff_path
        out, _ = read_to_array(path)
        np.testing.assert_array_equal(out, arr)


class TestBandValidationBackendParity:
    """Local eager and dask paths agree on the rejection contract."""

    def test_negative_band(self, multiband_tiff_path):
        """Both paths raise the same error for ``band=-1``."""
        from xrspatial.geotiff import read_geotiff_dask
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path

        with pytest.raises(IndexError) as eager_exc:
            read_to_array(path, band=-1)
        with pytest.raises(IndexError) as dask_exc:
            read_geotiff_dask(path, chunks=4, band=-1)

        assert "band=-1 out of range" in str(eager_exc.value)
        assert "band=-1 out of range" in str(dask_exc.value)

    def test_band_equal_to_samples(self, multiband_tiff_path):
        """Both paths agree on the off-by-one rejection."""
        from xrspatial.geotiff import read_geotiff_dask
        from xrspatial.geotiff._reader import read_to_array

        path, _ = multiband_tiff_path

        with pytest.raises(IndexError) as eager_exc:
            read_to_array(path, band=3)
        with pytest.raises(IndexError) as dask_exc:
            read_geotiff_dask(path, chunks=4, band=3)

        assert "band=3 out of range" in str(eager_exc.value)
        assert "band=3 out of range" in str(dask_exc.value)


# =============================================================================
# Section: open_geotiff missing_sources (#1810)
# =============================================================================
#
# Original: ``test_open_geotiff_missing_sources_1810.py``.
#
# ``open_geotiff`` did not accept ``missing_sources`` and did not
# forward it to ``read_vrt`` when the source was a VRT. The
# api-consistency sweep on 2026-05-13 flagged that ``read_vrt`` had
# gained a ``missing_sources='warn'|'raise'`` policy kwarg (#1806) but
# the documented dispatcher entry point ``open_geotiff`` did not
# expose it. Same class of dispatcher-drops-backend-kwarg bug as
# #1561, #1605, #1685, #1795.


def _write_missing_source_vrt_1810(path):
    """Write a VRT pointing at a non-existent source file."""
    path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">missing.tif'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_open_geotiff_accepts_missing_sources_1810():
    """The signature exposes ``missing_sources`` so IDE autocomplete and
    ``inspect.signature`` see the kwarg without parsing the docstring.
    """
    sig = inspect.signature(open_geotiff)
    assert 'missing_sources' in sig.parameters


def test_open_geotiff_vrt_missing_sources_warn_1810(tmp_path):
    vrt = tmp_path / "tmp_1810_missing_warn.vrt"
    _write_missing_source_vrt_1810(vrt)

    with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
        da = open_geotiff(str(vrt), missing_sources='warn')

    assert 'vrt_holes' in da.attrs
    assert da.attrs['vrt_holes'][0]['source'].endswith('missing.tif')


def test_open_geotiff_vrt_missing_sources_raise_1810(tmp_path):
    vrt = tmp_path / "tmp_1810_missing_raise.vrt"
    _write_missing_source_vrt_1810(vrt)

    with pytest.raises((OSError, ValueError)):
        open_geotiff(str(vrt), missing_sources='raise')


def test_open_geotiff_vrt_missing_sources_validates_policy_1810(tmp_path):
    vrt = tmp_path / "tmp_1810_missing_bad_policy.vrt"
    _write_missing_source_vrt_1810(vrt)

    with pytest.raises(ValueError, match="missing_sources"):
        open_geotiff(str(vrt), missing_sources='ignore')


def test_open_geotiff_rejects_missing_sources_on_tif_1810(tmp_path):
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(4, dtype=np.float64),
                'x': np.arange(4, dtype=np.float64)},
        attrs={'crs': 4326},
    )
    tif_path = tmp_path / "single_tile.tif"
    to_geotiff(da, str(tif_path))

    with pytest.raises(ValueError, match="missing_sources only applies"):
        open_geotiff(str(tif_path), missing_sources='raise')


def test_open_geotiff_vrt_without_missing_sources_kwarg_still_works_1810(tmp_path):
    arr_a = np.arange(16, dtype=np.uint16).reshape(4, 4)
    da_a = xr.DataArray(
        arr_a, dims=['y', 'x'],
        coords={'y': np.array([0.5, 1.5, 2.5, 3.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        attrs={'crs': 4326},
    )
    tile_a = tmp_path / "tile_a.tif"
    to_geotiff(da_a, str(tile_a))

    arr_b = np.arange(16, 32, dtype=np.uint16).reshape(4, 4)
    da_b = xr.DataArray(
        arr_b, dims=['y', 'x'],
        coords={'y': np.array([0.5, 1.5, 2.5, 3.5]),
                'x': np.array([4.5, 5.5, 6.5, 7.5])},
        attrs={'crs': 4326},
    )
    tile_b = tmp_path / "tile_b.tif"
    to_geotiff(da_b, str(tile_b))

    vrt_path = tmp_path / "mosaic_1810.vrt"
    _write_vrt_1810(str(vrt_path), [str(tile_a), str(tile_b)])

    da = open_geotiff(str(vrt_path))
    assert da.shape == (4, 8)


# =============================================================================
# Section: Reader strips/tiles/array (low-level) and partial-tile validation
# =============================================================================
#
# Original: ``test_reader.py`` (general low-level reader coverage) and
# the ``#1486`` partial-tile validation block within it.
#
# Covers ``_read_strips`` / ``_read_tiles`` / ``read_to_array`` happy
# paths and the truncated-tile rejection contract that turns opaque
# numpy reshape errors into clear "size mismatch" diagnostics.


class TestReadStrips_reader:
    def test_float32_sequential(self):
        """Read a simple float32 stripped TIFF and verify pixel values."""
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        data = make_minimal_tiff(4, 4, np.dtype('float32'), pixel_data=expected)

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        arr = _read_strips(data, ifd, header, dtype)
        np.testing.assert_array_equal(arr, expected)

    def test_uint16(self):
        expected = np.arange(20, dtype=np.uint16).reshape(4, 5)
        data = make_minimal_tiff(5, 4, np.dtype('uint16'), pixel_data=expected)

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        arr = _read_strips(data, ifd, header, dtype)
        np.testing.assert_array_equal(arr, expected)

    def test_windowed_read(self):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(8, 8, np.dtype('float32'), pixel_data=expected)

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        window = (2, 3, 6, 7)  # rows 2-5, cols 3-6
        arr = _read_strips(data, ifd, header, dtype, window=window)
        np.testing.assert_array_equal(arr, expected[2:6, 3:7])


class TestReadTiles_reader:
    def test_tiled_float32(self):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(
            8, 8, np.dtype('float32'),
            pixel_data=expected,
            tiled=True,
            tile_size=4,
        )

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        arr = _read_tiles(data, ifd, header, dtype)
        np.testing.assert_array_equal(arr, expected)

    def test_tiled_windowed(self):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(
            8, 8, np.dtype('float32'),
            pixel_data=expected,
            tiled=True,
            tile_size=4,
        )

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        window = (1, 2, 5, 6)
        arr = _read_tiles(data, ifd, header, dtype, window=window)
        np.testing.assert_array_equal(arr, expected[1:5, 2:6])


class TestReadToArray_reader:
    def test_local_file(self, tmp_path):
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        tiff_data = make_minimal_tiff(4, 4, np.dtype('float32'), pixel_data=expected)
        path = str(tmp_path / 'test.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        arr, geo_info = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_geo_info(self, tmp_path):
        tiff_data = make_minimal_tiff(
            4, 4, np.dtype('float32'),
            geo_transform=(-120.0, 45.0, 0.001, -0.001),
            epsg=4326,
        )
        path = str(tmp_path / 'geo_test.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        arr, geo_info = read_to_array(path)
        assert geo_info.crs_epsg == 4326
        assert geo_info.transform.origin_x == pytest.approx(-120.0)


class TestPartialTileValidation_1486:
    """Issue #1486: corrupt tile/strip data should raise a clear error.

    Without validation a truncated deflate stream causes numpy.reshape to
    raise an opaque "cannot reshape array of size N" with no hint of which
    tile is at fault.  These tests pin the new behaviour: a clear ValueError
    naming the size mismatch.
    """

    def _zero_out_last_tile_1486(self, path):
        """Replace the last tile's compressed bytes with zeros so deflate
        decodes a short stream."""
        with open(path, 'rb') as f:
            data = bytearray(f.read())
        header = parse_header(bytes(data))
        ifds = parse_all_ifds(bytes(data), header)
        ifd = ifds[0]
        if ifd.tile_offsets is not None:
            offsets = ifd.tile_offsets
            counts = ifd.tile_byte_counts
        else:
            offsets = ifd.strip_offsets
            counts = ifd.strip_byte_counts
        last_off = offsets[-1]
        last_count = counts[-1]
        zero_stream = b'\x78\x9c\x03\x00\x00\x00\x00\x01'
        padded = zero_stream + b'\x00' * max(0, last_count - len(zero_stream))
        for i, b in enumerate(padded[:last_count]):
            data[last_off + i] = b
        with open(path, 'wb') as f:
            f.write(bytes(data))

    def test_truncated_tile_raises_clear_error(self, tmp_path):
        from xrspatial.geotiff._writer import write

        pixels = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'truncated_1486.tif')
        write(pixels, path, compression='deflate', tiled=True, tile_size=8)

        self._zero_out_last_tile_1486(path)

        with pytest.raises(ValueError) as exc:
            read_to_array(path)
        msg = str(exc.value)
        assert 'size mismatch' in msg
        assert 'expected' in msg
        assert 'truncated or corrupt' in msg

    def test_valid_edge_tile_still_works(self, tmp_path):
        """Edge tiles in a valid file decompress to full tile size; the
        validation should not flag this as corrupt."""
        from xrspatial.geotiff._writer import write

        pixels = np.arange(81, dtype=np.float32).reshape(9, 9)
        path = str(tmp_path / 'edge_tiles_1486.tif')
        write(pixels, path, compression='deflate', tiled=True, tile_size=4)

        arr, _ = read_to_array(path)
        np.testing.assert_array_equal(arr, pixels)

# ===========================================================================
# Accuracy regressions (#1081)
# Source: test_accuracy_1081.py
# ===========================================================================


def _make_pixel_is_point_tiff(tmp_path, width=8, height=8):
    """Create a GeoTIFF with PixelIsPoint raster type via the writer."""
    from xrspatial.geotiff._geotags import GeoTransform

    arr = np.arange(width * height, dtype=np.float32).reshape(height, width)
    path = str(tmp_path / 'point_1081.tif')
    write(
        arr, path,
        geo_transform=GeoTransform(
            origin_x=10.0, origin_y=50.0,
            pixel_width=0.001, pixel_height=-0.001,
        ),
        crs_epsg=4326,
        compression='none',
        tiled=False,
        raster_type=RASTER_PIXEL_IS_POINT,
    )
    return path


# -----------------------------------------------------------------------
# Bug 1: Windowed read + PixelIsPoint
# -----------------------------------------------------------------------

class TestWindowedReadPixelIsPoint:

    def test_full_read_pixel_is_point_no_offset(self, tmp_path):
        """Full read of PixelIsPoint file should NOT add half-pixel offset."""
        path = _make_pixel_is_point_tiff(tmp_path)
        da = open_geotiff(path)
        # For PixelIsPoint, coordinates should be exactly at the tiepoint
        # origin (10.0) without any 0.5*pixel_width offset.
        assert da.attrs.get('raster_type') == 'point'
        assert float(da.coords['x'].values[0]) == pytest.approx(10.0)
        assert float(da.coords['y'].values[0]) == pytest.approx(50.0)

    def test_windowed_read_pixel_is_point_no_offset(self, tmp_path):
        """Windowed read of PixelIsPoint file should match full-read coords."""
        path = _make_pixel_is_point_tiff(tmp_path)
        da_full = open_geotiff(path)
        da_win = open_geotiff(path, window=(2, 2, 6, 6))

        # The windowed-read x/y should match the corresponding slice
        # of the full-read coordinates.
        np.testing.assert_allclose(
            da_win.coords['x'].values,
            da_full.coords['x'].values[2:6],
        )
        np.testing.assert_allclose(
            da_win.coords['y'].values,
            da_full.coords['y'].values[2:6],
        )

    def test_windowed_read_pixel_is_area_has_offset(self, tmp_path):
        """Windowed read of PixelIsArea should still apply half-pixel offset."""
        from xrspatial.geotiff._geotags import GeoTransform

        arr = np.ones((8, 8), dtype=np.float32)
        path = str(tmp_path / 'area_1081.tif')
        write(
            arr, path,
            geo_transform=GeoTransform(
                origin_x=10.0, origin_y=50.0,
                pixel_width=0.001, pixel_height=-0.001,
            ),
            crs_epsg=4326,
            compression='none',
            tiled=False,
        )
        da_full = open_geotiff(path)
        da_win = open_geotiff(path, window=(2, 2, 6, 6))

        np.testing.assert_allclose(
            da_win.coords['x'].values,
            da_full.coords['x'].values[2:6],
        )
        np.testing.assert_allclose(
            da_win.coords['y'].values,
            da_full.coords['y'].values[2:6],
        )


# -----------------------------------------------------------------------
# Bug 2: CRS WKT loss on write
# -----------------------------------------------------------------------

# A custom WKT that has no EPSG code -- represents a local engineering grid
_CUSTOM_WKT = (
    'LOCAL_CS["Local Grid",'
    'LOCAL_DATUM["Local",10000],'
    'UNIT["metre",1],'
    'AXIS["Easting",EAST],'
    'AXIS["Northing",NORTH]]'
)


class TestCrsWktRoundTrip:

    def test_wkt_survives_round_trip(self, tmp_path):
        """Custom WKT CRS should be preserved in GeoAsciiParamsTag."""
        arr = np.ones((4, 4), dtype=np.float32)
        da = xr.DataArray(
            arr,
            dims=['y', 'x'],
            coords={
                'y': np.arange(4, dtype=np.float64),
                'x': np.arange(4, dtype=np.float64),
            },
            attrs={'crs_wkt': _CUSTOM_WKT},
        )
        path = str(tmp_path / 'wkt_1081.tif')
        to_geotiff(da, path)

        # Read back the raw tags and check GeoAsciiParamsTag
        import mmap
        with open(path, 'rb') as f:
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            header = parse_header(data)
            ifds = parse_all_ifds(data, header)
            geo_info = extract_geo_info(ifds[0], data, header.byte_order)
        finally:
            data.close()

        # The GeoKey directory should have a user-defined CRS (32767)
        assert geo_info.crs_epsg is None or geo_info.crs_epsg == 32767

    def test_wkt_crs_param_survives(self, tmp_path):
        """crs= param with WKT string should be written when no EPSG."""
        arr = np.ones((4, 4), dtype=np.float32)
        da = xr.DataArray(
            arr,
            dims=['y', 'x'],
            coords={
                'y': np.arange(4, dtype=np.float64),
                'x': np.arange(4, dtype=np.float64),
            },
        )
        path = str(tmp_path / 'wkt_param_1081.tif')
        to_geotiff(da, path, crs=_CUSTOM_WKT)

        # Verify the GeoAsciiParams tag was written
        import mmap
        with open(path, 'rb') as f:
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            header = parse_header(data)
            ifds = parse_all_ifds(data, header)
            ifd = ifds[0]
            # Check for TAG_GEO_ASCII_PARAMS (34737) in IFD entries
            has_ascii_params = TAG_GEO_ASCII_PARAMS in ifd.entries
        finally:
            data.close()

        assert has_ascii_params, "GeoAsciiParamsTag should contain WKT"

    def test_epsg_crs_still_works(self, tmp_path):
        """EPSG CRS should still work as before (no WKT fallback)."""
        arr = np.ones((4, 4), dtype=np.float32)
        da = xr.DataArray(
            arr,
            dims=['y', 'x'],
            coords={
                'y': np.arange(4, dtype=np.float64),
                'x': np.arange(4, dtype=np.float64),
            },
        )
        path = str(tmp_path / 'epsg_1081.tif')
        to_geotiff(da, path, crs=4326)

        da_back = open_geotiff(path)
        assert da_back.attrs.get('crs') == 4326


# -----------------------------------------------------------------------
# Bug 3: NaN not restored to nodata sentinel on write
# -----------------------------------------------------------------------

class TestNodataRestore:

    def test_nan_restored_to_sentinel_float(self, tmp_path):
        """NaN pixels should be written as the nodata sentinel, not NaN."""
        arr = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
        da = xr.DataArray(
            arr,
            dims=['y', 'x'],
            coords={
                'y': np.arange(2, dtype=np.float64),
                'x': np.arange(2, dtype=np.float64),
            },
            attrs={'nodata': -9999.0},
        )
        path = str(tmp_path / 'nodata_restore_1081.tif')
        to_geotiff(da, path)

        # Read raw pixel data (before nodata masking) to verify sentinel
        raw_arr, geo_info = read_to_array(path)
        # The pixel that was NaN should now be -9999.0
        assert raw_arr[1, 0] == pytest.approx(-9999.0)
        assert not np.isnan(raw_arr[1, 0])

    def test_nan_nodata_sentinel_is_nan(self, tmp_path):
        """When nodata is NaN, pixels should stay as NaN (no conversion)."""
        arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        da = xr.DataArray(
            arr,
            dims=['y', 'x'],
            coords={
                'y': np.arange(2, dtype=np.float64),
                'x': np.arange(2, dtype=np.float64),
            },
            attrs={'nodata': float('nan')},
        )
        path = str(tmp_path / 'nan_nodata_1081.tif')
        to_geotiff(da, path)

        raw_arr, _ = read_to_array(path)
        assert np.isnan(raw_arr[0, 1])

    def test_full_round_trip_preserves_nodata(self, tmp_path):
        """open_geotiff -> to_geotiff round-trip should preserve nodata."""
        from xrspatial.geotiff._geotags import GeoTransform

        # Write a file with integer nodata sentinel
        arr = np.array([[1, 2], [0, 4]], dtype=np.int16)
        path1 = str(tmp_path / 'src_1081.tif')
        write(
            arr, path1,
            geo_transform=GeoTransform(0.0, 0.0, 1.0, -1.0),
            crs_epsg=4326,
            nodata=0,
            compression='none',
            tiled=False,
        )

        # Read it (nodata=0 -> NaN)
        da = open_geotiff(path1)
        assert np.isnan(da.values[1, 0])
        assert da.attrs['nodata'] == 0

        # Write it back
        path2 = str(tmp_path / 'dst_1081.tif')
        to_geotiff(da, path2)

        # Read raw data and check sentinel is restored
        # Note: the array was promoted to float64, so nodata=0 becomes 0.0
        raw, geo = read_to_array(path2)
        assert raw[1, 0] == pytest.approx(0.0)
        assert not np.isnan(raw[1, 0])

    def test_no_nodata_attr_no_conversion(self, tmp_path):
        """Arrays without nodata attr should not have NaN converted."""
        arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        da = xr.DataArray(
            arr,
            dims=['y', 'x'],
            coords={
                'y': np.arange(2, dtype=np.float64),
                'x': np.arange(2, dtype=np.float64),
            },
        )
        path = str(tmp_path / 'no_nodata_1081.tif')
        to_geotiff(da, path)

        raw_arr, _ = read_to_array(path)
        assert np.isnan(raw_arr[0, 1])

# ===========================================================================
# BytesIO file-like read/write (#1511)
# Source: test_bytesio_source.py
# ===========================================================================


def _gpu_available() -> bool:
    """True when cupy imports AND a CUDA runtime is initialised.

    Mirrors the helper used in other geotiff GPU tests so the BytesIO
    GPU-writer tests skip cleanly on hosts where CuPy is installed but
    CUDA is unavailable (Copilot review on #1653).
    """
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


def _make_da(height=32, width=40, dtype=np.float32):
    arr = np.arange(height * width, dtype=dtype).reshape(height, width)
    # Simple geotransform with negative pixel_height (north-up)
    x = np.arange(width, dtype=np.float64) * 0.5 + 100.0 + 0.25
    y = np.arange(height, dtype=np.float64) * (-0.5) + 50.0 - 0.25
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='test',
        attrs={
            'crs': 4326,
            'transform': (0.5, 0.0, 100.0, 0.0, -0.5, 50.0),
        },
    )
    return da


class TestBytesIORoundTrip:
    def test_round_trip_basic(self):
        """to_geotiff(BytesIO) + open_geotiff(BytesIO) round-trips data and crs."""
        da = _make_da()
        buf = io.BytesIO()
        to_geotiff(da, buf, compression='deflate')
        buf.seek(0)
        # Read back from a fresh BytesIO so the read path has to seek itself
        round_tripped = open_geotiff(io.BytesIO(buf.getvalue()))

        np.testing.assert_array_equal(round_tripped.values, da.values)
        assert round_tripped.attrs.get('crs') == 4326
        # Transform tuple preserved verbatim
        assert round_tripped.attrs.get('transform') == da.attrs['transform']

    def test_round_trip_uint8(self):
        """uint8 round-trip without compression."""
        da = _make_da(dtype=np.uint8)
        buf = io.BytesIO()
        to_geotiff(da, buf, compression='none')
        result = open_geotiff(io.BytesIO(buf.getvalue()))
        np.testing.assert_array_equal(result.values, da.values)


class TestBytesIOWindowedRead:
    def test_windowed_read(self):
        """Windowed read from a BytesIO source returns the right slice."""
        da = _make_da(height=64, width=64)
        buf = io.BytesIO()
        to_geotiff(da, buf, compression='deflate', tiled=True, tile_size=16)

        # Window covering tile-aligned and unaligned regions
        window = (8, 12, 40, 48)
        arr, _ = read_to_array(io.BytesIO(buf.getvalue()), window=window)
        expected = da.values[8:40, 12:48]
        np.testing.assert_array_equal(arr, expected)


class TestBytesIORejectsCog:
    def test_cog_with_file_like_rejected(self):
        """cog=True with a file-like destination raises ValueError."""
        da = _make_da()
        buf = io.BytesIO()
        with pytest.raises(ValueError, match='cog=True'):
            to_geotiff(da, buf, cog=True)


class TestBytesIORejectsVrt:
    def test_vrt_extension_with_file_like_is_treated_as_geotiff(self):
        """A file-like destination cannot carry a VRT extension marker.

        VRT output is filesystem-only (it writes per-tile sidecar GeoTIFFs
        to a directory), so passing a buffer with an unrelated name should
        not select the VRT path. The check uses isinstance(path, str), so a
        BytesIO simply gets treated as a plain GeoTIFF destination.
        """
        # No way to "name" a BytesIO as .vrt, so this asserts the gate works:
        # passing a BytesIO produces a normal GeoTIFF.
        da = _make_da()
        buf = io.BytesIO()
        to_geotiff(da, buf)
        # Header magic for classic TIFF is II*\0 or MM\0*
        head = buf.getvalue()[:4]
        assert head in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+')

    def test_explicit_vrt_path_string_with_file_like_data_works(self, tmp_path):
        """A real .vrt path with file-like data is unrelated to this PR.

        This test just sanity-checks that string-path VRT handling still
        rejects cog=True (existing behaviour) -- the regression we worry
        about is the writer accidentally taking the VRT branch when
        ``path`` is a buffer.
        """
        da = _make_da()
        with pytest.raises(ValueError, match='cog'):
            to_geotiff(da, str(tmp_path / 'out.vrt'), cog=True)


class TestBytesIOConcurrentReads:
    def test_concurrent_windowed_reads(self):
        """Many threads reading disjoint windows from one BytesIOSource see
        consistent bytes (lock around seek+read prevents cursor races)."""
        da = _make_da(height=128, width=128)
        buf = io.BytesIO()
        to_geotiff(da, buf, compression='deflate', tiled=True, tile_size=16)
        encoded = buf.getvalue()

        # All threads share one source instance.
        shared = io.BytesIO(encoded)
        src = _BytesIOSource(shared)

        # Pick byte ranges scattered through the file.
        size = src.size
        rng = np.random.default_rng(0)
        ranges = []
        for _ in range(64):
            start = int(rng.integers(0, max(1, size - 32)))
            length = int(rng.integers(1, min(128, size - start) + 1))
            ranges.append((start, length))

        # Compute the truth values single-threaded.
        truth = [encoded[s:s + n] for s, n in ranges]

        results = [None] * len(ranges)
        errors: list[BaseException] = []

        def worker(i):
            try:
                s, n = ranges[i]
                results[i] = src.read_range(s, n)
            except BaseException as e:  # pragma: no cover - defensive
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(len(ranges))))

        assert not errors, errors
        for got, want in zip(results, truth):
            assert got == want


# ---------------------------------------------------------------------------
# PR #1512 review followups: pathlib.Path normalisation, GPU+file-like reject,
# truncate-on-rewrite semantics, _is_file_like requires tell.
# ---------------------------------------------------------------------------


def _make_small_da():
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    return xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs': 4326},
    )


def test_pathlib_path_write_and_read_roundtrip(tmp_path):
    """``pathlib.Path`` should behave identically to ``str``."""
    p = tmp_path / 'roundtrip.tif'
    da = _make_small_da()
    to_geotiff(da, p)  # Path, not str
    out = open_geotiff(p)  # Path, not str
    np.testing.assert_array_equal(out.values, da.values)


def test_pathlib_path_vrt_routes_to_read_vrt(tmp_path):
    """``Path('x.vrt')`` must dispatch to the VRT path, not the TIFF reader."""
    import pathlib

    da = _make_small_da()
    tif_path = tmp_path / 'src.tif'
    to_geotiff(da, str(tif_path))

    vrt_path = tmp_path / 'mosaic.vrt'
    vrt_path.write_text(f"""<VRTDataset rasterXSize="8" rasterYSize="8">
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{tif_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="8" ySize="8"/>
      <DstRect xOff="0" yOff="0" xSize="8" ySize="8"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
""")
    out = open_geotiff(pathlib.Path(vrt_path))
    np.testing.assert_array_equal(out.values, da.values)


def test_pathlib_path_derives_name(tmp_path):
    """``name`` should derive from ``Path`` stem, same as for ``str``."""
    p = tmp_path / 'mydata.tif'
    to_geotiff(_make_small_da(), p)
    out = open_geotiff(p)
    assert out.name == 'mydata'


def test_to_geotiff_rejects_gpu_with_file_like():
    """``gpu=True`` + file-like is untested territory; reject up front."""
    da = _make_small_da()
    buf = io.BytesIO()
    with pytest.raises(ValueError, match='gpu=True is not supported'):
        to_geotiff(da, buf, gpu=True)


def test_to_geotiff_buffer_rewritten_in_place():
    """Reusing the same BytesIO should overwrite, not append.

    Mirrors ``to_geotiff('/tmp/x.tif', ...)`` followed by another call to
    the same path -- the second call replaces, not concatenates. PR #1512
    review found that file-like writes used to ``write()`` at the cursor
    so two writes to one buffer produced two TIFFs back-to-back.
    """
    buf = io.BytesIO()
    to_geotiff(_make_small_da(), buf)
    first_size = len(buf.getvalue())
    assert first_size > 0

    # Write a second, different DataArray to the same buffer.
    arr2 = np.full((4, 4), 7.0, dtype=np.float32)
    da2 = xr.DataArray(
        arr2, dims=['y', 'x'],
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        attrs={'crs': 4326},
    )
    to_geotiff(da2, buf)
    second_size = len(buf.getvalue())

    # The second TIFF should fully replace the first, not append.
    out = open_geotiff(io.BytesIO(buf.getvalue()))
    np.testing.assert_array_equal(out.values, arr2)
    # And the buffer must be smaller than first+second (i.e. no concat).
    assert second_size < first_size + 100


def test_is_file_like_requires_tell():
    """Objects with read+seek but no tell are not accepted as file-like.

    ``_BytesIOSource`` needs ``tell`` to compute the buffer size. We refuse
    seekable-but-not-tellable inputs at the gate rather than crashing inside
    ``__init__``.
    """
    from xrspatial.geotiff._reader import _is_file_like

    class ReadSeekNoTell:
        def read(self, n=-1):
            return b''

        def seek(self, *a, **k):
            return 0

    assert _is_file_like(io.BytesIO(b'x')) is True
    assert _is_file_like(ReadSeekNoTell()) is False


class TestWriteGeotiffGpuBytesIO:
    """Regression coverage for ``write_geotiff_gpu`` file-like behaviour.

    ``to_geotiff(gpu=True, ...)`` always rejects BytesIO destinations paired
    with ``cog=True`` (the auto-dispatch path's existing guard). The explicit
    GPU writer used to silently accept that combo and produce a COG into the
    buffer, so the two entry points disagreed on what ``to_geotiff(gpu=True,
    cog=True, path=BytesIO)`` does. These tests pin the mirrored gate added
    by issue #1652 and confirm the non-cog file-like path still works.
    """

    @_gpu_only
    def test_cog_with_bytesio_rejected_1652(self):
        import cupy
        da = xr.DataArray(
            cupy.asarray(np.random.rand(64, 64).astype(np.float32)),
            dims=['y', 'x'],
            coords={'y': np.arange(64.0), 'x': np.arange(64.0)},
            attrs={'crs': 4326},
        )
        from xrspatial.geotiff import write_geotiff_gpu

        buf = io.BytesIO()
        with pytest.raises(ValueError, match='cog=True'):
            write_geotiff_gpu(da, buf, cog=True)

    @_gpu_only
    def test_cog_with_bytesio_error_matches_to_geotiff_1652(self):
        """The error string must match ``to_geotiff``'s gate verbatim so
        downstream callers can rely on a single message (Copilot review
        on #1653)."""
        import cupy
        da = xr.DataArray(
            cupy.asarray(np.random.rand(64, 64).astype(np.float32)),
            dims=['y', 'x'],
            coords={'y': np.arange(64.0), 'x': np.arange(64.0)},
            attrs={'crs': 4326},
        )
        from xrspatial.geotiff import write_geotiff_gpu

        # to_geotiff's canonical message; mirrored verbatim in
        # write_geotiff_gpu's gate.
        expected = (
            "cog=True is not supported for file-like destinations. "
            "Pass a string path or write to BytesIO without cog=True."
        )

        buf = io.BytesIO()
        with pytest.raises(ValueError) as exc_info:
            write_geotiff_gpu(da, buf, cog=True)
        assert str(exc_info.value) == expected

        # And the CPU writer raises the same string for parity.
        with pytest.raises(ValueError) as exc_info_cpu:
            to_geotiff(_make_da(), io.BytesIO(), cog=True)
        assert str(exc_info_cpu.value) == expected

    @_gpu_only
    def test_invalid_path_type_raises_typeerror_1652(self):
        """Mirror to_geotiff's TypeError for non-str, non-file-like paths
        so callers see identical behaviour from both entry points."""
        import cupy
        da = xr.DataArray(
            cupy.asarray(np.random.rand(64, 64).astype(np.float32)),
            dims=['y', 'x'],
            coords={'y': np.arange(64.0), 'x': np.arange(64.0)},
            attrs={'crs': 4326},
        )
        from xrspatial.geotiff import write_geotiff_gpu

        with pytest.raises(TypeError, match="path must be a str"):
            write_geotiff_gpu(da, 42)  # int is neither str nor file-like

    @_gpu_only
    def test_non_cog_bytesio_still_works_1652(self):
        import cupy
        arr_cpu = np.random.rand(64, 64).astype(np.float32)
        da = xr.DataArray(
            cupy.asarray(arr_cpu),
            dims=['y', 'x'],
            coords={'y': np.arange(64.0), 'x': np.arange(64.0)},
            attrs={'crs': 4326},
        )
        from xrspatial.geotiff import write_geotiff_gpu

        buf = io.BytesIO()
        # Non-cog file-like write is still supported on the explicit GPU
        # writer; only cog=True is gated.
        write_geotiff_gpu(da, buf)
        assert len(buf.getvalue()) > 0

        # Verify it round-trips through open_geotiff
        rd = open_geotiff(io.BytesIO(buf.getvalue()))
        np.testing.assert_allclose(np.asarray(rd.values), arr_cpu)

# ===========================================================================
# Eager source close-on-error (#2322)
# Source: test_eager_source_close_on_error_2322.py
# ===========================================================================


def _make_tiff_bytes() -> bytes:
    """Build a small valid TIFF in memory for happy-path baselines."""
    arr = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        attrs={'crs': 4326,
               'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
    buf = io.BytesIO()
    to_geotiff(da, buf, compression='none')
    return buf.getvalue()


class _FailingSource:
    """Fake source whose ``read_all()`` raises.

    Records the number of ``close()`` calls so the test can verify
    cleanup ran on the exception path. Mirrors the ``_CloseTracker``
    pattern in ``test_cog_http_close_on_error_1816.py`` but as a
    standalone fake rather than a delegating wrapper, because the
    failure happens inside ``read_all()`` itself (no real source ever
    succeeds in this scenario).
    """

    def __init__(self, exc: Exception):
        self._exc = exc
        self.close_count = 0
        self.read_all_count = 0

    def read_all(self):
        self.read_all_count += 1
        raise self._exc

    def close(self):
        self.close_count += 1


def test_file_source_closed_when_read_all_raises(tmp_path):
    """``_FileSource.read_all()`` failure path still runs ``close()``.

    Patches ``_FileSource`` so the constructor returns a fake whose
    ``read_all()`` raises. The fix wraps the eager read in the
    ``try/finally`` that calls ``src.close()``, so the close count
    must be exactly 1 even though ``read_all()`` raised.
    """
    path = str(tmp_path / "tmp_2322_cleanup_file.tif")
    # File must exist so ``_coerce_path`` does not bail early — even
    # though the patched ``_FileSource`` never reads from it.
    with open(path, 'wb') as f:
        f.write(_make_tiff_bytes())

    fake = _FailingSource(OSError("simulated local read failure"))

    with patch(
        'xrspatial.geotiff._reader._FileSource',
        return_value=fake,
    ):
        with pytest.raises(OSError, match="simulated local read failure"):
            _read_to_array(path)

    assert fake.read_all_count == 1, (
        "read_all() was not invoked; the test setup is wrong")
    assert fake.close_count == 1, (
        "src.close() was not called on the exception path; "
        "the try/finally guard around read_all() is missing")


def test_bytesio_source_closed_when_read_all_raises():
    """``_BytesIOSource.read_all()`` failure path still runs ``close()``.

    Same shape as the file-source test, but via the file-like
    branch (``_is_file_like(source)`` returns True for ``BytesIO``).
    """
    buf = io.BytesIO(_make_tiff_bytes())

    fake = _FailingSource(RuntimeError("simulated buffer read failure"))

    with patch(
        'xrspatial.geotiff._reader._BytesIOSource',
        return_value=fake,
    ):
        with pytest.raises(RuntimeError, match="simulated buffer read"):
            _read_to_array(buf)

    assert fake.read_all_count == 1
    assert fake.close_count == 1, (
        "src.close() was not called on the exception path for the "
        "BytesIO/file-like branch")


def test_cloud_source_closed_when_read_all_raises():
    """``_CloudSource.read_all()`` failure path still runs ``close()``.

    The cloud branch is the original motivating case: a fsspec read
    that fails mid-download must still tear down whatever state the
    source holds. Today ``_CloudSource.close()`` is a no-op, so this
    test exists primarily to lock in the structural guard before any
    real resource is added.

    Patches both ``_is_fsspec_uri`` (to route the input string into
    the cloud branch) and ``_CloudSource`` (to return a fake that
    raises in ``read_all()``). ``_resolve_max_cloud_bytes`` is
    bypassed by passing ``max_cloud_bytes=None`` so the
    pre-``read_all`` size check does not consume the close() call
    on its own error path (that branch is already exception-safe and
    is exercised separately in ``test_cloud_read_byte_limit_1928``).
    """
    fake = _FailingSource(OSError("simulated S3 mid-download failure"))

    with patch(
        'xrspatial.geotiff._reader._is_fsspec_uri',
        return_value=True,
    ), patch(
        'xrspatial.geotiff._reader._CloudSource',
        return_value=fake,
    ):
        with pytest.raises(OSError, match="simulated S3 mid-download"):
            _read_to_array("s3://fake-bucket/fake-key.tif",
                           max_cloud_bytes=None)

    assert fake.read_all_count == 1
    assert fake.close_count == 1, (
        "src.close() was not called on the exception path for the "
        "fsspec/cloud branch; this is the main case the fix targets")

# ===========================================================================
# TIFF Orientation tag decode (#1503)
# Source: test_orientation.py
# ===========================================================================


tifffile = pytest.importorskip("tifffile")


# Eight orientation values defined by TIFF 6.0.
_ORIENTATIONS = [1, 2, 3, 4, 5, 6, 7, 8]


def _write_with_orientation(path, arr, orientation):
    """Write *arr* to *path* with the given Orientation tag value.

    tifffile's ``imwrite`` does not expose Orientation as a kwarg, but
    the ``extratags`` parameter accepts (tag_id, dtype_code, count, value,
    write_once) tuples that get emitted into the IFD verbatim. ``H`` is
    the unsigned short (TIFF type 3) struct code.
    """
    tifffile.imwrite(
        str(path),
        arr,
        extratags=[(274, 'H', 1, orientation, True)],
    )


def _expected_for_orientation(stored, orientation):
    """Return what *stored* should look like after applying *orientation*.

    Mirrors the spec table in :func:`xrspatial.geotiff._reader._apply_orientation`.
    """
    if orientation == 1:
        return stored
    if orientation == 2:
        return stored[:, ::-1]
    if orientation == 3:
        return stored[::-1, ::-1]
    if orientation == 4:
        return stored[::-1, :]
    if orientation == 5:
        return stored.T
    if orientation == 6:
        return stored.T[:, ::-1]
    if orientation == 7:
        return stored.T[::-1, ::-1]
    if orientation == 8:
        return stored.T[::-1, :]
    raise AssertionError(orientation)


@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_orientation_matches_spec(tmp_path, orientation):
    """open_geotiff applies the spec-defined transform for each orientation."""
    # Asymmetric data (different height and width, distinct row/column
    # values) so any axis swap or flip shows up as a clear mismatch.
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    expected = _expected_for_orientation(arr, orientation)
    got = open_geotiff(str(path))

    assert got.values.shape == expected.shape, (
        f"orientation={orientation}: shape mismatch "
        f"got={got.values.shape} expected={expected.shape}"
    )
    np.testing.assert_array_equal(got.values, expected)


@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_orientation_coords_match_post_orientation_shape(
    tmp_path, orientation
):
    """y/x coordinate arrays size matches the post-orientation array shape."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_coords_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    da = open_geotiff(str(path))

    h, w = da.values.shape
    assert da.coords['y'].shape == (h,)
    assert da.coords['x'].shape == (w,)


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_orientation_5_to_8_swap_dims(tmp_path, orientation):
    """Orientations 5-8 swap rows and columns relative to the stored shape."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)  # h=4, w=6
    path = tmp_path / f"orient_swap_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    da = open_geotiff(str(path))

    # File stores h=4, w=6. After orientation 5-8 the displayed shape is
    # (6, 4) -- width and height swap.
    assert da.values.shape == (6, 4)


def test_orientation_default_unchanged(tmp_path):
    """A file without an Orientation tag defaults to 1 (no transform)."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / "no_orient.tif"
    tifffile.imwrite(str(path), arr)

    da = open_geotiff(str(path))
    np.testing.assert_array_equal(da.values, arr)


def test_orientation_with_window_raises(tmp_path):
    """Windowed read on a non-default orientation raises ValueError."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / "orient2_window.tif"
    _write_with_orientation(path, arr, 2)

    with pytest.raises(ValueError, match="orientation"):
        read_to_array(str(path), window=(0, 0, 2, 2))

    with pytest.raises(ValueError, match="orientation"):
        open_geotiff(str(path), window=(0, 0, 2, 2))


def test_orientation_1_with_window_still_works(tmp_path):
    """Default orientation (1) with window= keeps working as before."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / "orient1_window.tif"
    _write_with_orientation(path, arr, 1)

    da = open_geotiff(str(path), window=(0, 0, 2, 3))
    assert da.values.shape == (2, 3)
    np.testing.assert_array_equal(da.values, arr[:2, :3])


@pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
def test_orientation_tag_not_passed_through_extra_tags(tmp_path, orientation):
    """Tag 274 must not survive on the returned DataArray.

    Without this, a read+write round-trip would re-emit the original
    Orientation tag even though the pixel buffer is already remapped --
    downstream readers would apply the orientation a second time.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_passthrough_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    da = open_geotiff(str(path))
    extra = da.attrs.get('extra_tags') or []
    tag_ids = [t[0] if isinstance(t, tuple) else t for t in extra]
    assert 274 not in tag_ids, (
        f"orientation={orientation}: tag 274 leaked into extra_tags={extra}"
    )


def test_orientation_round_trip_does_not_double_apply(tmp_path):
    """open_geotiff -> to_geotiff -> open_geotiff returns the same array.

    Concretely: a file written with orientation=4 reads as flipped
    (correct), the writer emits a normal file (no orientation tag), and
    a second read returns the same array. If tag 274 leaked through
    extra_tags, the second read would apply orientation=4 again.
    """
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path1 = tmp_path / "orient4_in.tif"
    _write_with_orientation(path1, arr, 4)

    da1 = open_geotiff(str(path1))

    path2 = tmp_path / "orient4_out.tif"
    to_geotiff(da1, str(path2))
    da2 = open_geotiff(str(path2))

    np.testing.assert_array_equal(da2.values, da1.values)


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_orientation_5_to_8_raise_on_georef(tmp_path, orientation):
    """Axis-swap orientations on georef'd files raise NotImplementedError.

    Orientations 5-8 require a per-orientation origin shift plus a
    rotation that the axis-aligned GeoTransform cannot represent.
    The reader used to swap pixel_width/pixel_height and warn; that
    produced silently wrong coords on georef'd files (issue #1765).
    The reader now refuses the file instead.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_georef_raise_1765_{orientation}.tif"
    # ModelPixelScale + GeoKeyDirectory pair pointing at EPSG:4326
    # makes the reader treat this as a georeferenced file.
    tifffile.imwrite(
        str(path), arr,
        extratags=[
            (274, 'H', 1, orientation, True),
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), True),
            (34735, 'H', 12, (
                1, 1, 0, 2,
                1024, 0, 1, 2,
                2048, 0, 1, 4326,
            ), True),
        ],
    )

    with pytest.raises(NotImplementedError, match=str(orientation)):
        open_geotiff(str(path))


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_orientation_5_to_8_transform_only_raises(tmp_path, orientation):
    """``has_georef`` without CRS still triggers the raise.

    A TIFF carrying ModelPixelScale + ModelTiepoint but no
    GeoKeyDirectory has ``has_georef=True`` and ``crs_epsg=None``. The
    pixel-size swap alone misses the per-orientation origin shift, so
    refusing is the honest contract regardless of CRS tagging.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_transform_only_1765_{orientation}.tif"
    tifffile.imwrite(
        str(path), arr,
        extratags=[
            (274, 'H', 1, orientation, True),
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), True),
        ],
    )

    with pytest.raises(NotImplementedError, match=str(orientation)):
        open_geotiff(str(path))


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_orientation_5_to_8_no_geo_still_swaps(tmp_path, orientation):
    """Without georef, orientations 5-8 still do the axis swap.

    No geographic claim to violate, so the existing transpose path is
    preserved (regression guard for the #1765 fix not over-reaching).
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_no_geo_1765_{orientation}.tif"
    tifffile.imwrite(
        str(path), arr,
        extratags=[(274, 'H', 1, orientation, True)],
    )

    da = open_geotiff(str(path))
    expected = _expected_for_orientation(arr, orientation)
    assert da.values.shape == expected.shape
    np.testing.assert_array_equal(da.values, expected)


def test_orientation_1_georef_unchanged_1765(tmp_path):
    """orientation=1 on a georef'd file still reads normally.

    Regression guard: the #1765 raise must be scoped to 5-8, not fire
    on any georeferenced file.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / "orient_georef_1_1765.tif"
    tifffile.imwrite(
        str(path), arr,
        extratags=[
            (274, 'H', 1, 1, True),
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), True),
            (34735, 'H', 12, (
                1, 1, 0, 2,
                1024, 0, 1, 2,
                2048, 0, 1, 4326,
            ), True),
        ],
    )

    da = open_geotiff(str(path))
    np.testing.assert_array_equal(da.values, arr)


# ---------------------------------------------------------------------------
# Geographic coordinate updates for mirror-flip orientations (issue #1537)
# ---------------------------------------------------------------------------
#
# Orientations 2/3/4 flip the array horizontally, both axes, or vertically.
# The reader used to apply the buffer flip but leave the y/x coord arrays
# computed from the original transform, so xarray label-based lookups
# returned the wrong pixel for georeferenced files.

_GEOREF_EXTRA_AREA = [
    (33550, 'd', 3, (1.0, 1.0, 0.0)),
    (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0)),
    (34735, 'H', 12, (
        1, 1, 0, 2,
        1024, 0, 1, 2,
        2048, 0, 1, 4326,
    )),
]
_GEOREF_EXTRA_POINT = [
    (33550, 'd', 3, (1.0, 1.0, 0.0)),
    (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0)),
    (34735, 'H', 16, (
        1, 1, 0, 3,
        1024, 0, 1, 2,
        1025, 0, 1, 2,    # PixelIsPoint
        2048, 0, 1, 4326,
    )),
]


def _write_with_orient_and_georef(path, arr, orientation, raster_type='area'):
    """Write *arr* with Orientation tag + EPSG:4326 georef."""
    extras = (_GEOREF_EXTRA_POINT if raster_type == 'point'
              else _GEOREF_EXTRA_AREA)
    tifffile.imwrite(
        str(path), arr,
        extratags=[(274, 'H', 1, orientation, True)] + [
            (tag, dt, count, val, True) for (tag, dt, count, val) in extras
        ],
    )


@pytest.mark.parametrize('orientation', [2, 3, 4])
def test_orient_2_3_4_coords_track_pixel_flip_area(tmp_path, orientation):
    """Mirror-flip orientations: a label-based lookup at a fixed (x, y)
    must return the same pixel value regardless of the file's orientation.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)  # h=4, w=6

    # Reference: orientation=1 (no transform) tells us the geographic
    # coords for each file pixel under PixelIsArea. Pick three corners
    # so x-flip, y-flip, and combined flips each have a target.
    ref_path = tmp_path / 'orient_ref.tif'
    _write_with_orient_and_georef(ref_path, arr, 1)
    ref = open_geotiff(str(ref_path))
    # Pixel (0, 0): top-left, value 0, at x=100.5, y=49.5
    # Pixel (0, 5): top-right, value 5, at x=105.5, y=49.5
    # Pixel (3, 0): bottom-left, value 18, at x=100.5, y=46.5
    # Pixel (3, 5): bottom-right, value 23, at x=105.5, y=46.5
    targets = [
        (100.5, 49.5, 0),
        (105.5, 49.5, 5),
        (100.5, 46.5, 18),
        (105.5, 46.5, 23),
    ]
    for x, y, expected in targets:
        assert int(ref.sel(x=x, y=y).item()) == expected

    # Now check the same coords under the flipped orientation.
    path = tmp_path / f'orient_{orientation}.tif'
    _write_with_orient_and_georef(path, arr, orientation)
    da = open_geotiff(str(path))
    for x, y, expected in targets:
        got = int(da.sel(x=x, y=y).item())
        assert got == expected, (
            f'orientation={orientation}: sel(x={x}, y={y}) returned {got}, '
            f'expected {expected} (mirror-flip lost the pixel-to-coord '
            f'binding)'
        )


@pytest.mark.parametrize('orientation', [2, 3, 4])
def test_orient_2_3_4_coords_track_pixel_flip_point(tmp_path, orientation):
    """Same coord-fidelity check for PixelIsPoint files."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)

    ref_path = tmp_path / 'orient_ref_pt.tif'
    _write_with_orient_and_georef(ref_path, arr, 1, raster_type='point')
    ref = open_geotiff(str(ref_path))
    # PixelIsPoint: pixel (0, 0) center = (100, 50), pixel (0, 5) = (105, 50)
    # pixel (3, 0) = (100, 47), pixel (3, 5) = (105, 47)
    targets = [
        (100.0, 50.0, 0),
        (105.0, 50.0, 5),
        (100.0, 47.0, 18),
        (105.0, 47.0, 23),
    ]
    for x, y, expected in targets:
        assert int(ref.sel(x=x, y=y).item()) == expected

    path = tmp_path / f'orient_{orientation}_pt.tif'
    _write_with_orient_and_georef(path, arr, orientation, raster_type='point')
    da = open_geotiff(str(path))
    for x, y, expected in targets:
        got = int(da.sel(x=x, y=y).item())
        assert got == expected, (
            f'PixelIsPoint orient={orientation}: sel(x={x}, y={y})={got}, '
            f'expected {expected}'
        )


@pytest.mark.parametrize(
    'orientation,expected_first_x,expected_first_y',
    [
        # x_first=100.5 means the leftmost displayed column carries the
        # original left-edge geographic coordinate; 105.5 means it carries
        # the original right-edge coordinate (i.e. the coord array got
        # flipped along x).
        (2, 105.5, 49.5),  # x flipped, y unchanged
        (3, 105.5, 46.5),  # both flipped
        (4, 100.5, 46.5),  # x unchanged, y flipped
    ],
)
def test_orient_2_3_4_coord_arrays(
    tmp_path, orientation, expected_first_x, expected_first_y,
):
    """The first y/x coord lands on the right edge after a flip."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f'orient_arr_{orientation}.tif'
    _write_with_orient_and_georef(path, arr, orientation)

    da = open_geotiff(str(path))
    np.testing.assert_allclose(da.x[0].item(), expected_first_x)
    np.testing.assert_allclose(da.y[0].item(), expected_first_y)


def test_orient_2_3_4_no_geo_still_uses_pixel_coords(tmp_path):
    """Without georef, the legacy integer pixel coord behaviour is preserved.

    A file without ModelPixelScale / ModelTiepoint stays on integer
    coords regardless of orientation; the fix must not start fabricating
    coordinates for un-georeferenced files.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    for orientation in (2, 3, 4):
        path = tmp_path / f'orient_no_geo_{orientation}.tif'
        tifffile.imwrite(
            str(path), arr,
            extratags=[(274, 'H', 1, orientation, True)],
        )
        da = open_geotiff(str(path))
        # Coords default to 0..N-1 when has_georef is False; the
        # orientation transform fix must not change that.
        assert da.x.values.dtype.kind in ('i', 'u'), (
            f'orient={orientation}: x coords drifted off integer '
            f'(dtype={da.x.values.dtype})'
        )


def test_orient_2_3_4_no_geo_does_not_modify_transform(tmp_path):
    """Non-georef file: transform attr stays at the default after orient 2/3/4.

    Earlier draft fabricated origin/sign for the default GeoTransform on
    un-georeferenced files because the gating only checked
    ``transform is not None`` and the dataclass default is non-None.
    Downstream consumers rightly fall back to integer pixel coords for
    these files, but exposing a fake transform via attrs misleads any
    direct attrs reader. Gate must be ``has_georef``.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    # Default GeoTransform on a fresh file the writer leaves untouched.
    baseline_path = tmp_path / "orient_no_geo_baseline.tif"
    tifffile.imwrite(
        str(baseline_path), arr,
        extratags=[(274, 'H', 1, 1, True)],
    )
    baseline_transform = open_geotiff(str(baseline_path)).attrs.get('transform')

    for orientation in (2, 3, 4):
        path = tmp_path / f"orient_no_geo_xform_{orientation}.tif"
        tifffile.imwrite(
            str(path), arr,
            extratags=[(274, 'H', 1, orientation, True)],
        )
        da = open_geotiff(str(path))
        assert da.attrs.get('transform') == baseline_transform, (
            f"orient={orientation} on a non-georef file modified "
            f"attrs['transform']: got {da.attrs.get('transform')}, "
            f"expected baseline {baseline_transform}"
        )


def test_orientation_with_band_selection_returns_2d(tmp_path):
    """band= followed by an orientation transpose returns a 2D array.

    Regression: an earlier draft applied orientation before slicing the
    band, which wasted memory and produced confusing intermediates.
    Now band slicing happens first.
    """
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    path = tmp_path / "orient5_rgb.tif"
    tifffile.imwrite(
        str(path), rgb, photometric='rgb',
        extratags=[(274, 'H', 1, 5, True)],
    )

    # Orientation 5 transposes spatial axes, so output spatial shape is
    # (6, 4). Band 1 returns just that channel.
    da = open_geotiff(str(path), band=1)
    assert da.values.shape == (6, 4)
    expected = rgb[..., 1].T  # band 1 then transpose
    np.testing.assert_array_equal(da.values, expected)

# ===========================================================================
# _FileSource context-manager protocol (#2449)
# Source: test_file_source_context_2449.py
# ===========================================================================


@pytest.fixture
def tiff_path(tmp_path):
    arr = np.zeros((4, 4), dtype=np.uint8)
    path = str(tmp_path / 'fs_ctx_2449.tif')
    to_geotiff(arr, path, compression='none')
    return path


def _refcount(path):
    """Look up the cache refcount for *path*, or None if not cached."""
    real = os.path.realpath(path)
    entry = _mmap_cache._entries.get(real)
    return None if entry is None else entry[3]


def test_enter_returns_self(tiff_path):
    src = _FileSource(tiff_path)
    try:
        assert src.__enter__() is src
    finally:
        src.close()


def test_exit_releases_entry(tiff_path):
    _mmap_cache.clear()
    with _FileSource(tiff_path) as src:
        assert _refcount(tiff_path) == 1
        # mmap is usable inside the block
        assert len(src.read_all()) == src.size
    # After the with block, refcount returns to 0 (entry stays cached).
    assert _refcount(tiff_path) == 0


def test_exit_releases_on_exception(tiff_path):
    _mmap_cache.clear()
    with pytest.raises(struct.error):
        with _FileSource(tiff_path):
            assert _refcount(tiff_path) == 1
            struct.unpack('>I', b'')
    assert _refcount(tiff_path) == 0


def test_double_close_safe(tiff_path):
    with _FileSource(tiff_path) as src:
        src.close()
        # __exit__ will call close() again; must not raise or
        # over-decrement the cache refcount.
        assert _refcount(tiff_path) == 0
    assert _refcount(tiff_path) == 0


def test_nested_with_shares_cache_entry(tiff_path):
    _mmap_cache.clear()
    with _FileSource(tiff_path):
        assert _refcount(tiff_path) == 1
        with _FileSource(tiff_path):
            assert _refcount(tiff_path) == 2
        assert _refcount(tiff_path) == 1
    assert _refcount(tiff_path) == 0
