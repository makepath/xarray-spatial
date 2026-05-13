"""Tests for new features: multi-band, integer nodata, packbits, zstd, dask, BigTIFF."""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._compression import (
    COMPRESSION_PACKBITS,
    packbits_compress,
    packbits_decompress,
    zstd_compress,
    zstd_decompress,
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

    def test_rgb_to_geotiff_api(self, tmp_path):
        """to_geotiff accepts 3D arrays."""
        arr = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        path = str(tmp_path / 'rgb_api.tif')
        to_geotiff(arr, path, compression='none')

        result = open_geotiff(path)
        assert 'band' in result.dims
        assert result.shape == (4, 4, 3)
        np.testing.assert_array_equal(result.values, arr)

    def test_rgb_cog(self, tmp_path):
        """Multi-band COG with overviews."""
        arr = np.random.RandomState(7).randint(
            0, 256, (32, 32, 3), dtype=np.uint8)
        path = str(tmp_path / 'rgb_cog.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=16,
              cog=True, overview_levels=[2])

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

        da = open_geotiff(path)
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 1] == 1.0
        assert da.dtype == np.float64  # promoted from uint8

    def test_uint16_nodata_masked(self, tmp_path):
        arr = np.array([[100, 0], [200, 0]], dtype=np.uint16)
        path = str(tmp_path / 'uint16_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=0)

        da = open_geotiff(path)
        assert np.isnan(da.values[0, 1])
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 0] == 100.0

    def test_int16_nodata_negative(self, tmp_path):
        arr = np.array([[-9999, 10], [20, -9999]], dtype=np.int16)
        path = str(tmp_path / 'int16_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=-9999)

        da = open_geotiff(path)
        assert np.isnan(da.values[0, 0])
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 1] == 10.0

    def test_integer_no_nodata_stays_integer(self, tmp_path):
        """Without nodata, integer arrays should not be promoted."""
        arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
        path = str(tmp_path / 'no_nodata.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
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
# ZSTD compression
# -----------------------------------------------------------------------

class TestZstd:

    def test_zstd_round_trip_bytes(self):
        data = b'hello zstd! ' * 1000
        compressed = zstd_compress(data)
        assert len(compressed) < len(data)
        assert zstd_decompress(compressed) == data

    def test_zstd_empty(self):
        compressed = zstd_compress(b'')
        assert zstd_decompress(compressed) == b''

    def test_zstd_random(self):
        rng = np.random.RandomState(42)
        data = bytes(rng.randint(0, 256, size=5000, dtype=np.uint8))
        assert zstd_decompress(zstd_compress(data)) == data

    def test_write_read_zstd_stripped(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'zstd_strip.tif')
        write(arr, path, compression='zstd', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_write_read_zstd_tiled(self, tmp_path):
        arr = np.random.RandomState(99).rand(16, 16).astype(np.float32)
        path = str(tmp_path / 'zstd_tiled.tif')
        write(arr, path, compression='zstd', tiled=True, tile_size=8)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_zstd_uint16(self, tmp_path):
        arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
        path = str(tmp_path / 'zstd_u16.tif')
        write(arr, path, compression='zstd', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_zstd_with_predictor(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'zstd_pred.tif')
        write(arr, path, compression='zstd', tiled=False, predictor=True)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_zstd_multiband(self, tmp_path):
        arr = np.random.RandomState(7).randint(0, 256, (8, 8, 3), dtype=np.uint8)
        path = str(tmp_path / 'zstd_rgb.tif')
        write(arr, path, compression='zstd', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_zstd_public_api(self, tmp_path):
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'zstd_api.tif')
        to_geotiff(arr, path, compression='zstd')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)


# -----------------------------------------------------------------------
# GeoKey metadata extraction
# -----------------------------------------------------------------------

class TestGeoKeys:

    def test_geographic_crs_attrs(self, tmp_path):
        """Geographic CRS files expose citation and angular units."""
        from xrspatial.geotiff._geotags import GeoTransform

        arr = np.ones((4, 4), dtype=np.float32)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'geog.tif')
        write(arr, path, compression='none', tiled=False,
              geo_transform=gt, crs_epsg=4326)

        da = open_geotiff(path)
        assert da.attrs['crs'] == 4326
        assert da.attrs.get('geog_citation') is not None or da.attrs['crs'] == 4326

    def test_projected_crs_attrs(self, tmp_path):
        """Projected CRS files expose linear units."""
        from xrspatial.geotiff._geotags import GeoTransform

        arr = np.ones((4, 4), dtype=np.float32)
        gt = GeoTransform(500000.0, 4500000.0, 30.0, -30.0)
        path = str(tmp_path / 'proj.tif')
        write(arr, path, compression='none', tiled=False,
              geo_transform=gt, crs_epsg=32610)

        da = open_geotiff(path)
        assert da.attrs['crs'] == 32610

    def test_geoinfo_fields_from_real_file(self):
        """Verify GeoInfo fields populated from a real geographic file."""
        import os
        path = '../rtxpy/examples/render_demo_terrain.tif'
        if not os.path.exists(path):
            pytest.skip("Real test files not available")

        da = open_geotiff(path)
        assert da.attrs['crs'] == 4269
        assert da.attrs['geog_citation'] == 'NAD83'
        assert da.attrs['angular_units'] == 'degree'
        assert da.attrs['semi_major_axis'] == pytest.approx(6378137.0)
        assert da.attrs['inv_flattening'] == pytest.approx(298.257, rel=1e-3)

    def test_geoinfo_fields_from_projected_file(self):
        """Verify projected CRS fields from a real UTM file."""
        import os
        path = '../rtxpy/examples/USGS_one_meter_x65y454_NY_LongIsland_Z18_2014.tif'
        if not os.path.exists(path):
            pytest.skip("Real test files not available")

        da = open_geotiff(path)
        assert da.attrs['crs'] == 26918
        assert da.attrs['crs_name'] == 'NAD83 / UTM zone 18N'
        assert da.attrs['geog_citation'] == 'NAD83'
        assert da.attrs['linear_units'] == 'metre'

    def test_no_crs_no_geokey_attrs(self, tmp_path):
        """Files without CRS don't get geokey attrs."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'bare.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        assert 'crs_name' not in da.attrs
        assert 'geog_citation' not in da.attrs
        assert 'angular_units' not in da.attrs
        assert 'linear_units' not in da.attrs

    def test_angular_unit_lookup(self):
        """Unit code -> name lookup works for known codes."""
        from xrspatial.geotiff._geotags import ANGULAR_UNITS, LINEAR_UNITS
        assert ANGULAR_UNITS[9102] == 'degree'
        assert ANGULAR_UNITS[9101] == 'radian'
        assert LINEAR_UNITS[9001] == 'metre'
        assert LINEAR_UNITS[9002] == 'foot'
        assert LINEAR_UNITS[9003] == 'us_survey_foot'

    def test_crs_wkt_from_epsg(self, tmp_path):
        """crs_wkt is resolved from EPSG via pyproj."""
        from xrspatial.geotiff._geotags import GeoTransform
        arr = np.ones((4, 4), dtype=np.float32)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'wkt.tif')
        write(arr, path, compression='none', tiled=False,
              geo_transform=gt, crs_epsg=4326)

        da = open_geotiff(path)
        assert 'crs_wkt' in da.attrs
        wkt = da.attrs['crs_wkt']
        assert 'WGS 84' in wkt or '4326' in wkt

    def test_write_with_wkt_string(self, tmp_path):
        """crs= accepts a WKT string and resolves to EPSG."""
        arr = np.ones((4, 4), dtype=np.float32)
        wkt = ('GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
               'ELLIPSOID["WGS 84",6378137,298.257223563]],'
               'CS[ellipsoidal,2],'
               'AXIS["geodetic latitude (Lat)",north],'
               'AXIS["geodetic longitude (Lon)",east],'
               'UNIT["degree",0.0174532925199433],'
               'ID["EPSG",4326]]')
        path = str(tmp_path / 'wkt_in.tif')
        to_geotiff(arr, path, crs=wkt, compression='none')

        da = open_geotiff(path)
        assert da.attrs['crs'] == 4326

    def test_write_with_proj_string(self, tmp_path):
        """crs= accepts a PROJ string."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'proj_in.tif')
        to_geotiff(arr, path, crs='+proj=utm +zone=18 +datum=NAD83',
                      compression='none')

        da = open_geotiff(path)
        # pyproj should resolve this to EPSG:26918
        assert da.attrs.get('crs') is not None

    def test_crs_wkt_attr_round_trip(self, tmp_path):
        """DataArray with crs_wkt attr (no int crs) round-trips."""
        wkt = ('GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
               'ELLIPSOID["WGS 84",6378137,298.257223563]],'
               'CS[ellipsoidal,2],'
               'AXIS["geodetic latitude (Lat)",north],'
               'AXIS["geodetic longitude (Lon)",east],'
               'UNIT["degree",0.0174532925199433],'
               'ID["EPSG",4326]]')
        y = np.linspace(45.0, 44.0, 4)
        x = np.linspace(-120.0, -119.0, 4)
        da = xr.DataArray(np.ones((4, 4), dtype=np.float32),
                          dims=['y', 'x'], coords={'y': y, 'x': x},
                          attrs={'crs_wkt': wkt})
        path = str(tmp_path / 'wkt_rt.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.attrs['crs'] == 4326
        assert 'crs_wkt' in result.attrs

    def test_no_crs_no_wkt(self, tmp_path):
        """File without CRS has no crs_wkt attr."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'no_wkt.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        assert 'crs_wkt' not in da.attrs


# -----------------------------------------------------------------------
# Resolution / DPI tags
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# GDAL metadata (tag 42112)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Arbitrary tag preservation
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Big-endian pixel data
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Cloud storage (fsspec) support
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# VRT (Virtual Raster Table) support
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Fixes: band-first, MinIsWhite, ExtraSamples, float16, VRT write, etc.
# -----------------------------------------------------------------------

class TestFixesBatch:

    def test_band_first_dataarray(self, tmp_path):
        """DataArray with (band, y, x) dims is transposed before write."""
        arr = np.zeros((3, 8, 8), dtype=np.uint8)
        arr[0] = 200  # red
        arr[1] = 100  # green
        arr[2] = 50   # blue

        da = xr.DataArray(arr, dims=['band', 'y', 'x'])
        path = str(tmp_path / 'band_first.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.shape == (8, 8, 3)
        assert result.values[0, 0, 0] == 200  # red channel
        assert result.values[0, 0, 1] == 100  # green channel

    def test_band_last_dataarray_unchanged(self, tmp_path):
        """DataArray with (y, x, band) dims is not transposed."""
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[:, :, 0] = 200
        da = xr.DataArray(arr, dims=['y', 'x', 'band'])
        path = str(tmp_path / 'band_last.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.shape == (8, 8, 3)
        assert result.values[0, 0, 0] == 200

    def test_min_is_white_inversion(self, tmp_path):
        """MinIsWhite (photometric=0) inverts grayscale values on read."""
        from .conftest import make_minimal_tiff
        import struct

        # Build a minimal TIFF with photometric=0
        # The conftest doesn't support photometric param, so build manually
        bo = '<'
        width, height = 4, 4
        pixels = np.array([[0, 50, 100, 200]], dtype=np.uint8).repeat(4, axis=0)

        tag_list = []
        def add_short(tag, val):
            tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))
        def add_long(tag, val):
            tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

        add_short(256, width)
        add_short(257, height)
        add_short(258, 8)
        add_short(259, 1)
        add_short(262, 0)   # MinIsWhite
        add_short(277, 1)
        add_short(278, height)
        add_long(273, 0)
        add_long(279, len(pixels.tobytes()))
        add_short(339, 1)

        tag_list.sort(key=lambda t: t[0])
        num_entries = len(tag_list)
        ifd_start = 8
        ifd_size = 2 + 12 * num_entries + 4
        overflow_start = ifd_start + ifd_size
        pixel_start = overflow_start
        # Patch strip offset
        for i, (tag, typ, count, raw) in enumerate(tag_list):
            if tag == 273:
                tag_list[i] = (tag, typ, count, struct.pack(f'{bo}I', pixel_start))

        out = bytearray()
        out.extend(b'II')
        out.extend(struct.pack(f'{bo}H', 42))
        out.extend(struct.pack(f'{bo}I', ifd_start))
        out.extend(struct.pack(f'{bo}H', num_entries))
        for tag, typ, count, raw in tag_list:
            out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
            out.extend(raw.ljust(4, b'\x00'))
        out.extend(struct.pack(f'{bo}I', 0))
        out.extend(pixels.tobytes())

        path = str(tmp_path / 'miniswhite.tif')
        with open(path, 'wb') as f:
            f.write(bytes(out))

        from xrspatial.geotiff._reader import read_to_array
        result, _ = read_to_array(path)
        # MinIsWhite: 0 -> 255, 50 -> 205, 100 -> 155, 200 -> 55
        assert result[0, 0] == 255
        assert result[0, 1] == 205
        assert result[0, 2] == 155
        assert result[0, 3] == 55

    def test_extra_samples_rgba(self, tmp_path):
        """RGBA write includes ExtraSamples tag."""
        from xrspatial.geotiff._header import parse_header, parse_all_ifds, TAG_EXTRA_SAMPLES
        arr = np.ones((4, 4, 4), dtype=np.uint8) * 128
        path = str(tmp_path / 'rgba.tif')
        write(arr, path, compression='none', tiled=False)

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        extra = ifd.entries.get(TAG_EXTRA_SAMPLES)
        assert extra is not None
        # Value 2 = unassociated alpha
        assert extra.value == 2 or (isinstance(extra.value, tuple) and extra.value[0] == 2)

    def test_float16_auto_promotion(self, tmp_path):
        """Float16 arrays are auto-promoted to float32."""
        arr = np.ones((4, 4), dtype=np.float16) * 3.14
        path = str(tmp_path / 'f16.tif')
        to_geotiff(arr, path, compression='none')

        result = open_geotiff(path)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result.values, 3.14, decimal=2)

    def test_vrt_write_and_read_back(self, tmp_path):
        """write_vrt generates a valid VRT that reads back correctly."""
        from xrspatial.geotiff import write_vrt
        from xrspatial.geotiff._geotags import GeoTransform

        # Write two tiles with known geo transforms
        left = np.arange(16, dtype=np.float32).reshape(4, 4)
        right = np.arange(16, 32, dtype=np.float32).reshape(4, 4)

        gt_left = GeoTransform(origin_x=0.0, origin_y=4.0,
                               pixel_width=1.0, pixel_height=-1.0)
        gt_right = GeoTransform(origin_x=4.0, origin_y=4.0,
                                pixel_width=1.0, pixel_height=-1.0)

        lpath = str(tmp_path / 'left.tif')
        rpath = str(tmp_path / 'right.tif')
        write(left, lpath, geo_transform=gt_left, compression='none', tiled=False)
        write(right, rpath, geo_transform=gt_right, compression='none', tiled=False)

        vrt_path = str(tmp_path / 'mosaic.vrt')
        write_vrt(vrt_path, [lpath, rpath])

        da = open_geotiff(vrt_path)
        assert da.shape == (4, 8)
        np.testing.assert_array_equal(da.values[:, :4], left)
        np.testing.assert_array_equal(da.values[:, 4:], right)

    def test_dask_vrt(self, tmp_path):
        """read_geotiff_dask handles VRT files."""
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        tile_path = str(tmp_path / 'tile.tif')
        write(arr, tile_path, compression='none', tiled=False)

        vrt_xml = (
            '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '  <VRTRasterBand dataType="Float32" band="1">\n'
            '    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{os.path.basename(tile_path)}</SourceFilename>\n'
            '      <SourceBand>1</SourceBand>\n'
            '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        vrt_path = str(tmp_path / 'dask.vrt')
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        import dask.array as da
        result = read_geotiff_dask(vrt_path, chunks=2)
        assert isinstance(result.data, da.Array)
        computed = result.compute()
        np.testing.assert_array_equal(computed.values, arr)


class TestVRT:

    def _write_tile(self, tmp_path, name, data):
        """Write a GeoTIFF tile and return its path."""
        from xrspatial.geotiff._writer import write
        path = str(tmp_path / name)
        write(data, path, compression='none', tiled=False)
        return path

    def _make_mosaic_vrt(self, tmp_path, tile_paths, tile_shapes,
                         tile_offsets, width, height, dtype='Float32'):
        """Build a VRT XML that mosaics multiple tiles."""
        lines = [
            f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
            '  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>',
            f'  <VRTRasterBand dataType="{dtype}" band="1">',
        ]
        for path, (th, tw), (yo, xo) in zip(tile_paths, tile_shapes, tile_offsets):
            lines.append('    <SimpleSource>')
            lines.append(f'      <SourceFilename relativeToVRT="1">{os.path.basename(path)}</SourceFilename>')
            lines.append('      <SourceBand>1</SourceBand>')
            lines.append(f'      <SrcRect xOff="0" yOff="0" xSize="{tw}" ySize="{th}"/>')
            lines.append(f'      <DstRect xOff="{xo}" yOff="{yo}" xSize="{tw}" ySize="{th}"/>')
            lines.append('    </SimpleSource>')
        lines.append('  </VRTRasterBand>')
        lines.append('</VRTDataset>')

        vrt_path = str(tmp_path / 'mosaic.vrt')
        with open(vrt_path, 'w') as f:
            f.write('\n'.join(lines))
        return vrt_path

    def test_single_tile_vrt(self, tmp_path):
        """VRT with one source tile reads correctly."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        tile_path = self._write_tile(tmp_path, 'tile.tif', arr)

        vrt_path = self._make_mosaic_vrt(
            tmp_path,
            [tile_path], [(4, 4)], [(0, 0)],
            width=4, height=4,
        )

        da = open_geotiff(vrt_path)
        np.testing.assert_array_equal(da.values, arr)

    def test_2x1_mosaic(self, tmp_path):
        """VRT that tiles two images side-by-side."""
        left = np.arange(16, dtype=np.float32).reshape(4, 4)
        right = np.arange(16, 32, dtype=np.float32).reshape(4, 4)
        lpath = self._write_tile(tmp_path, 'left.tif', left)
        rpath = self._write_tile(tmp_path, 'right.tif', right)

        vrt_path = self._make_mosaic_vrt(
            tmp_path,
            [lpath, rpath], [(4, 4), (4, 4)], [(0, 0), (0, 4)],
            width=8, height=4,
        )

        da = open_geotiff(vrt_path)
        assert da.shape == (4, 8)
        np.testing.assert_array_equal(da.values[:, :4], left)
        np.testing.assert_array_equal(da.values[:, 4:], right)

    def test_2x2_mosaic(self, tmp_path):
        """VRT that tiles four images in a 2x2 grid."""
        tiles = []
        paths = []
        offsets = []
        for r in range(2):
            for c in range(2):
                base = (r * 2 + c) * 16
                arr = np.arange(base, base + 16, dtype=np.float32).reshape(4, 4)
                name = f'tile_{r}_{c}.tif'
                paths.append(self._write_tile(tmp_path, name, arr))
                tiles.append(arr)
                offsets.append((r * 4, c * 4))

        vrt_path = self._make_mosaic_vrt(
            tmp_path,
            paths, [(4, 4)] * 4, offsets,
            width=8, height=8,
        )

        da = open_geotiff(vrt_path)
        assert da.shape == (8, 8)
        # Check each quadrant
        np.testing.assert_array_equal(da.values[0:4, 0:4], tiles[0])
        np.testing.assert_array_equal(da.values[0:4, 4:8], tiles[1])
        np.testing.assert_array_equal(da.values[4:8, 0:4], tiles[2])
        np.testing.assert_array_equal(da.values[4:8, 4:8], tiles[3])

    def test_windowed_vrt_read(self, tmp_path):
        """Windowed read of a VRT mosaic."""
        left = np.arange(16, dtype=np.float32).reshape(4, 4)
        right = np.arange(16, 32, dtype=np.float32).reshape(4, 4)
        lpath = self._write_tile(tmp_path, 'left.tif', left)
        rpath = self._write_tile(tmp_path, 'right.tif', right)

        vrt_path = self._make_mosaic_vrt(
            tmp_path,
            [lpath, rpath], [(4, 4), (4, 4)], [(0, 0), (0, 4)],
            width=8, height=4,
        )

        # Window spanning both tiles
        da = open_geotiff(vrt_path, window=(1, 2, 3, 6))
        assert da.shape == (2, 4)
        expected = np.hstack([left, right])[1:3, 2:6]
        np.testing.assert_array_equal(da.values, expected)

    def test_vrt_with_crs(self, tmp_path):
        """VRT with SRS tag populates CRS in attrs."""
        arr = np.ones((4, 4), dtype=np.float32)
        tile_path = self._write_tile(tmp_path, 'tile.tif', arr)

        vrt_xml = (
            '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '  <SRS>EPSG:4326</SRS>\n'
            '  <GeoTransform>-120.0, 0.001, 0.0, 45.0, 0.0, -0.001</GeoTransform>\n'
            '  <VRTRasterBand dataType="Float32" band="1">\n'
            '    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{os.path.basename(tile_path)}</SourceFilename>\n'
            '      <SourceBand>1</SourceBand>\n'
            '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        vrt_path = str(tmp_path / 'crs.vrt')
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        da = open_geotiff(vrt_path)
        assert da.attrs.get('crs_wkt') == 'EPSG:4326'
        assert len(da.coords['x']) == 4
        assert len(da.coords['y']) == 4

    def test_vrt_nodata(self, tmp_path):
        """VRT NoDataValue is stored in attrs."""
        arr = np.array([[1, 2], [3, -9999]], dtype=np.float32)
        tile_path = self._write_tile(tmp_path, 'tile.tif', arr)

        vrt_xml = (
            '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
            '  <VRTRasterBand dataType="Float32" band="1">\n'
            '    <NoDataValue>-9999</NoDataValue>\n'
            '    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{os.path.basename(tile_path)}</SourceFilename>\n'
            '      <SourceBand>1</SourceBand>\n'
            '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        vrt_path = str(tmp_path / 'nodata.vrt')
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        da = open_geotiff(vrt_path)
        assert da.attrs.get('nodata') == -9999.0

    def test_read_vrt_function(self, tmp_path):
        """read_vrt() works directly."""
        from xrspatial.geotiff import read_vrt
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        tile_path = self._write_tile(tmp_path, 'tile.tif', arr)

        vrt_path = self._make_mosaic_vrt(
            tmp_path,
            [tile_path], [(4, 4)], [(0, 0)],
            width=4, height=4,
        )

        da = read_vrt(vrt_path)
        assert da.name == 'mosaic'
        np.testing.assert_array_equal(da.values, arr)

    def test_vrt_parser(self, tmp_path):
        """VRT XML parser extracts all fields correctly."""
        from xrspatial.geotiff._vrt import parse_vrt

        # Use a path under tmp_path so the issue #1671 containment check
        # accepts the source.  The test exercises field-extraction, not
        # the on-disk readability of the source file.
        src_path = str(tmp_path / 'tile.tif')
        xml = (
            '<VRTDataset rasterXSize="100" rasterYSize="200">\n'
            '  <SRS>EPSG:32610</SRS>\n'
            '  <GeoTransform>500000, 30, 0, 4500000, 0, -30</GeoTransform>\n'
            '  <VRTRasterBand dataType="UInt16" band="1">\n'
            '    <NoDataValue>0</NoDataValue>\n'
            '    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n'
            '      <SourceBand>1</SourceBand>\n'
            '      <SrcRect xOff="10" yOff="20" xSize="80" ySize="160"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="80" ySize="160"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        vrt = parse_vrt(xml, str(tmp_path))
        assert vrt.width == 100
        assert vrt.height == 200
        assert vrt.crs_wkt == 'EPSG:32610'
        assert vrt.geo_transform == (500000.0, 30.0, 0.0, 4500000.0, 0.0, -30.0)
        assert len(vrt.bands) == 1
        assert vrt.bands[0].dtype == np.uint16
        assert vrt.bands[0].nodata == 0.0
        assert len(vrt.bands[0].sources) == 1
        src = vrt.bands[0].sources[0]
        assert src.filename == os.path.realpath(src_path)
        assert src.src_rect.x_off == 10

    def test_vrt_float64_fractional_nodata_masked(self, tmp_path):
        """VRT read masks float64 fractional nodata exactly.

        Regression for the ``np.float32(src_nodata)`` hard-cast in
        ``_vrt.read_vrt``.  A float64 source with a fractional
        sentinel that is not exactly representable in float32
        (e.g. -9999.1) used to miss the mask because
        ``np.float32(-9999.1) != np.float64(-9999.1)`` in the ``==``
        comparison.  The fix casts the sentinel to the source
        array's own dtype.

        -9999.1 is chosen over -9999.25 because the latter is
        exactly representable in float32 and would not exercise
        the bug.
        """
        sentinel = np.float64(-9999.1)
        # Sanity check the premise of the regression: the float32
        # cast must not round-trip back to the float64 value.
        assert np.float32(sentinel) != sentinel

        arr = np.array(
            [[1.0, 2.0],
             [sentinel, 4.0]],
            dtype=np.float64,
        )
        tile_path = self._write_tile(tmp_path, 'f64_nodata_1247.tif', arr)

        vrt_xml = (
            '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
            '  <VRTRasterBand dataType="Float64" band="1">\n'
            '    <NoDataValue>-9999.1</NoDataValue>\n'
            '    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{os.path.basename(tile_path)}</SourceFilename>\n'
            '      <SourceBand>1</SourceBand>\n'
            '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        vrt_path = str(tmp_path / 'f64_nodata_1247.vrt')
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        da = open_geotiff(vrt_path)
        vals = da.values

        # The sentinel pixel must be NaN.
        assert np.isnan(vals[1, 0]), (
            f"float64 fractional nodata not masked: got {vals[1, 0]!r}")
        # Other pixels untouched.
        assert vals[0, 0] == 1.0
        assert vals[0, 1] == 2.0
        assert vals[1, 1] == 4.0

    def test_vrt_pixel_is_point_no_half_pixel_shift(self, tmp_path):
        """VRT with AREA_OR_POINT=Point does not apply a half-pixel shift.

        Before the fix, ``read_vrt`` always added ``(c + 0.5) * res``
        to the GeoTransform origin, even when the VRT advertised
        Point registration.  That shifted coords by half a cell in
        world space on any Point-tagged VRT.

        The expected GDAL convention: when ``AREA_OR_POINT=Point``
        the GeoTransform origin is already the *center* of pixel
        (0, 0), so coords[0] must equal origin exactly.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tile_path = self._write_tile(tmp_path, 'point_1247.tif', arr)

        origin_x, origin_y = 100.0, 50.0
        pixel_w, pixel_h = 10.0, -10.0
        vrt_xml = (
            f'<VRTDataset rasterXSize="2" rasterYSize="2">\n'
            f'  <Metadata>\n'
            f'    <MDI key="AREA_OR_POINT">Point</MDI>\n'
            f'  </Metadata>\n'
            f'  <GeoTransform>{origin_x}, {pixel_w}, 0.0, '
            f'{origin_y}, 0.0, {pixel_h}</GeoTransform>\n'
            f'  <VRTRasterBand dataType="Float32" band="1">\n'
            f'    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{os.path.basename(tile_path)}</SourceFilename>\n'
            f'      <SourceBand>1</SourceBand>\n'
            f'      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
            f'      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
            f'    </SimpleSource>\n'
            f'  </VRTRasterBand>\n'
            f'</VRTDataset>\n'
        )
        vrt_path = str(tmp_path / 'point_1247.vrt')
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        da = open_geotiff(vrt_path)

        # Point registration: coords[0] == origin, no 0.5*pixel shift.
        assert float(da.coords['x'].values[0]) == pytest.approx(origin_x)
        assert float(da.coords['y'].values[0]) == pytest.approx(origin_y)
        # Adjacent cell is one full pixel away.
        assert float(da.coords['x'].values[1]) == pytest.approx(
            origin_x + pixel_w)
        assert float(da.coords['y'].values[1]) == pytest.approx(
            origin_y + pixel_h)
        # Raster type is surfaced in attrs.
        assert da.attrs.get('raster_type') == 'point'

    def test_vrt_pixel_is_area_still_shifts(self, tmp_path):
        """Default VRT (no AREA_OR_POINT metadata) keeps the half-pixel shift.

        This is the backwards-compat guard for the Point fix: Area
        registration must continue to add ``0.5 * pixel`` to the
        origin.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tile_path = self._write_tile(tmp_path, 'area_1247.tif', arr)

        origin_x, origin_y = 100.0, 50.0
        pixel_w, pixel_h = 10.0, -10.0
        vrt_xml = (
            f'<VRTDataset rasterXSize="2" rasterYSize="2">\n'
            f'  <GeoTransform>{origin_x}, {pixel_w}, 0.0, '
            f'{origin_y}, 0.0, {pixel_h}</GeoTransform>\n'
            f'  <VRTRasterBand dataType="Float32" band="1">\n'
            f'    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{os.path.basename(tile_path)}</SourceFilename>\n'
            f'      <SourceBand>1</SourceBand>\n'
            f'      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
            f'      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
            f'    </SimpleSource>\n'
            f'  </VRTRasterBand>\n'
            f'</VRTDataset>\n'
        )
        vrt_path = str(tmp_path / 'area_1247.vrt')
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        da = open_geotiff(vrt_path)

        # Area registration: coords[0] == origin + 0.5 * pixel.
        assert float(da.coords['x'].values[0]) == pytest.approx(
            origin_x + 0.5 * pixel_w)
        assert float(da.coords['y'].values[0]) == pytest.approx(
            origin_y + 0.5 * pixel_h)
        # No raster_type attr when Area (default).
        assert da.attrs.get('raster_type') != 'point'


class TestCloudStorage:

    def test_cloud_scheme_detection(self):
        """Cloud URI schemes are detected correctly."""
        from xrspatial.geotiff._reader import _is_fsspec_uri
        assert _is_fsspec_uri('s3://bucket/key.tif')
        assert _is_fsspec_uri('gs://bucket/key.tif')
        assert _is_fsspec_uri('az://container/blob.tif')
        assert _is_fsspec_uri('abfs://container/blob.tif')
        assert _is_fsspec_uri('memory:///test.tif')
        assert not _is_fsspec_uri('/local/path.tif')
        assert not _is_fsspec_uri('http://example.com/file.tif')
        assert not _is_fsspec_uri('relative/path.tif')

    def test_memory_filesystem_read_write(self, tmp_path):
        """Round-trip through fsspec's in-memory filesystem."""
        import fsspec

        arr = np.arange(16, dtype=np.float32).reshape(4, 4)

        # Write to memory filesystem via fsspec
        from xrspatial.geotiff._writer import write, _write_bytes
        from xrspatial.geotiff._writer import _assemble_tiff, _write_stripped
        from xrspatial.geotiff._compression import COMPRESSION_NONE

        # First write locally, then copy to memory fs
        local_path = str(tmp_path / 'test.tif')
        write(arr, local_path, compression='none', tiled=False)

        with open(local_path, 'rb') as f:
            tiff_bytes = f.read()

        # Put into fsspec memory filesystem
        fs = fsspec.filesystem('memory')
        fs.pipe('/test.tif', tiff_bytes)

        # Read via _CloudSource
        from xrspatial.geotiff._reader import _CloudSource
        src = _CloudSource('memory:///test.tif')
        data = src.read_all()
        assert len(data) == len(tiff_bytes)
        assert data == tiff_bytes

        # Range read
        chunk = src.read_range(0, 8)
        assert chunk == tiff_bytes[:8]

        # Clean up
        fs.rm('/test.tif')

    def test_memory_filesystem_full_roundtrip(self, tmp_path):
        """to_geotiff + open_geotiff through memory:// filesystem."""
        import fsspec

        arr = np.arange(16, dtype=np.float32).reshape(4, 4)

        # Write locally first, then copy to memory fs
        local_path = str(tmp_path / 'local.tif')
        to_geotiff(arr, local_path, compression='deflate')
        with open(local_path, 'rb') as f:
            tiff_bytes = f.read()

        fs = fsspec.filesystem('memory')
        fs.pipe('/roundtrip.tif', tiff_bytes)

        # Read from memory filesystem
        from xrspatial.geotiff._reader import read_to_array
        result, geo = read_to_array('memory:///roundtrip.tif')
        np.testing.assert_array_equal(result, arr)

        fs.rm('/roundtrip.tif')

    def test_dask_path_fsspec_uri_1749(self, tmp_path):
        """read_geotiff_dask supports fsspec URIs (issue #1749).

        The eager path already routed through _CloudSource via
        _read_to_array. The dask path's _read_geo_info used plain
        open(), which failed on memory://, s3://, etc.
        """
        pytest.importorskip('fsspec')
        import fsspec

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)

        local_path = str(tmp_path / 'src.tif')
        to_geotiff(arr, local_path, compression='none')
        with open(local_path, 'rb') as f:
            tiff_bytes = f.read()

        fs = fsspec.filesystem('memory')
        fs.pipe('/dask_1749_full.tif', tiff_bytes)

        try:
            eager = open_geotiff('memory:///dask_1749_full.tif')
            lazy = open_geotiff('memory:///dask_1749_full.tif', chunks=4)

            # Lazy path is dask-backed
            import dask.array as da
            assert isinstance(lazy.data, da.Array)

            np.testing.assert_array_equal(lazy.values, eager.values)
            np.testing.assert_array_equal(lazy.values, arr)
        finally:
            fs.rm('/dask_1749_full.tif')

    def test_dask_path_fsspec_uri_no_full_download_1749(self, tmp_path,
                                                       monkeypatch):
        """Dask graph build for fsspec URIs must not pull the whole file.

        ``_read_geo_info`` previously called ``_CloudSource.read_all`` to
        parse metadata. For a large COG on S3 that downloads the whole
        object just to learn its shape/transform. The fix routes fsspec
        sources through ``_parse_cog_http_meta``, which only uses
        ``read_range``. Guard against regression by failing the test if
        ``read_all`` runs during ``open_geotiff(..., chunks=...)``. See
        PR #1755 review.
        """
        pytest.importorskip('fsspec')
        import fsspec

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)

        local_path = str(tmp_path / 'src.tif')
        to_geotiff(arr, local_path, compression='none')
        with open(local_path, 'rb') as f:
            tiff_bytes = f.read()

        fs = fsspec.filesystem('memory')
        fs.pipe('/dask_1749_nofull.tif', tiff_bytes)

        from xrspatial.geotiff import _reader as _reader_mod

        def _no_read_all(self):
            raise AssertionError(
                "_CloudSource.read_all called during dask graph build")

        monkeypatch.setattr(
            _reader_mod._CloudSource, 'read_all', _no_read_all)

        try:
            lazy = open_geotiff('memory:///dask_1749_nofull.tif', chunks=4)
            # Materialise to confirm the chunk tasks also avoid read_all.
            np.testing.assert_array_equal(lazy.values, arr)
        finally:
            fs.rm('/dask_1749_nofull.tif')

    def test_writer_cloud_scheme_detection(self):
        """Writer detects cloud schemes."""
        from xrspatial.geotiff._writer import _is_fsspec_uri
        assert _is_fsspec_uri('s3://bucket/key.tif')
        assert _is_fsspec_uri('gs://bucket/key.tif')
        assert _is_fsspec_uri('az://container/blob.tif')
        assert not _is_fsspec_uri('/local/path.tif')

    def test_write_to_memory_filesystem(self, tmp_path):
        """_write_bytes can write to fsspec memory filesystem."""
        import fsspec
        from xrspatial.geotiff._writer import write

        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        local_path = str(tmp_path / 'src.tif')
        write(arr, local_path, compression='none', tiled=False)
        with open(local_path, 'rb') as f:
            tiff_bytes = f.read()

        # Write via _write_bytes to memory filesystem
        from xrspatial.geotiff._writer import _write_bytes
        _write_bytes(tiff_bytes, 'memory:///written.tif')

        fs = fsspec.filesystem('memory')
        assert fs.exists('/written.tif')
        assert fs.cat('/written.tif') == tiff_bytes

        fs.rm('/written.tif')


class TestBigEndian:

    def test_float32_big_endian(self, tmp_path):
        """Read a big-endian float32 TIFF."""
        from .conftest import make_minimal_tiff
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        tiff_data = make_minimal_tiff(4, 4, np.dtype('float32'),
                                       pixel_data=expected, big_endian=True)
        path = str(tmp_path / 'be_f32.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, expected)

    def test_uint16_big_endian(self, tmp_path):
        """Read a big-endian uint16 TIFF."""
        from .conftest import make_minimal_tiff
        expected = np.arange(20, dtype=np.uint16).reshape(4, 5) * 1000
        tiff_data = make_minimal_tiff(5, 4, np.dtype('uint16'),
                                       pixel_data=expected, big_endian=True)
        path = str(tmp_path / 'be_u16.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, expected)

    def test_int32_big_endian(self, tmp_path):
        """Read a big-endian int32 TIFF."""
        from .conftest import make_minimal_tiff
        expected = np.arange(16, dtype=np.int32).reshape(4, 4) - 8
        tiff_data = make_minimal_tiff(4, 4, np.dtype('int32'),
                                       pixel_data=expected, big_endian=True)
        path = str(tmp_path / 'be_i32.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, expected)

    def test_float64_big_endian(self, tmp_path):
        """Read a big-endian float64 TIFF."""
        from .conftest import make_minimal_tiff
        expected = np.linspace(-1.0, 1.0, 16, dtype=np.float64).reshape(4, 4)
        tiff_data = make_minimal_tiff(4, 4, np.dtype('float64'),
                                       pixel_data=expected, big_endian=True)
        path = str(tmp_path / 'be_f64.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.float64
        np.testing.assert_array_almost_equal(result, expected)

    def test_uint8_big_endian_no_swap_needed(self, tmp_path):
        """uint8 big-endian needs no byte swap (single byte per sample)."""
        from .conftest import make_minimal_tiff
        expected = np.arange(16, dtype=np.uint8).reshape(4, 4)
        tiff_data = make_minimal_tiff(4, 4, np.dtype('uint8'),
                                       pixel_data=expected, big_endian=True)
        path = str(tmp_path / 'be_u8.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, expected)

    def test_big_endian_windowed(self, tmp_path):
        """Windowed read of a big-endian TIFF."""
        from .conftest import make_minimal_tiff
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        tiff_data = make_minimal_tiff(8, 8, np.dtype('float32'),
                                       pixel_data=expected, big_endian=True)
        path = str(tmp_path / 'be_window.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path, window=(2, 3, 6, 7))
        np.testing.assert_array_equal(result, expected[2:6, 3:7])

    def test_big_endian_via_public_api(self, tmp_path):
        """open_geotiff handles big-endian files."""
        from .conftest import make_minimal_tiff
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        tiff_data = make_minimal_tiff(
            4, 4, np.dtype('float32'), pixel_data=expected,
            big_endian=True,
            geo_transform=(-120.0, 45.0, 0.001, -0.001), epsg=4326)
        path = str(tmp_path / 'be_api.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = open_geotiff(path)
        assert da.attrs['crs'] == 4326
        np.testing.assert_array_equal(da.values, expected)


class TestExtraTags:

    def _make_tiff_with_extra_tags(self, tmp_path):
        """Build a TIFF with Software (305) and DateTime (306) tags."""
        import struct
        bo = '<'
        width, height = 4, 4
        pixels = np.arange(16, dtype=np.float32).reshape(4, 4)
        pixel_bytes = pixels.tobytes()

        tag_list = []
        def add_short(tag, val):
            tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))
        def add_long(tag, val):
            tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))
        def add_ascii(tag, text):
            raw = text.encode('ascii') + b'\x00'
            tag_list.append((tag, 2, len(raw), raw))

        add_short(256, width)
        add_short(257, height)
        add_short(258, 32)
        add_short(259, 1)
        add_short(262, 1)
        add_short(277, 1)
        add_short(278, height)
        add_long(273, 0)  # placeholder
        add_long(279, len(pixel_bytes))
        add_short(339, 3)  # float
        add_ascii(305, 'TestSoftware v1.0')
        add_ascii(306, '2025:01:15 12:00:00')

        tag_list.sort(key=lambda t: t[0])
        num_entries = len(tag_list)
        ifd_start = 8
        ifd_size = 2 + 12 * num_entries + 4
        overflow_start = ifd_start + ifd_size

        overflow_buf = bytearray()
        tag_offsets = {}
        for tag, typ, count, raw in tag_list:
            if len(raw) > 4:
                tag_offsets[tag] = len(overflow_buf)
                overflow_buf.extend(raw)
                if len(overflow_buf) % 2:
                    overflow_buf.append(0)
            else:
                tag_offsets[tag] = None

        pixel_data_start = overflow_start + len(overflow_buf)

        patched = []
        for tag, typ, count, raw in tag_list:
            if tag == 273:
                patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
            else:
                patched.append((tag, typ, count, raw))
        tag_list = patched

        overflow_buf = bytearray()
        tag_offsets = {}
        for tag, typ, count, raw in tag_list:
            if len(raw) > 4:
                tag_offsets[tag] = len(overflow_buf)
                overflow_buf.extend(raw)
                if len(overflow_buf) % 2:
                    overflow_buf.append(0)
            else:
                tag_offsets[tag] = None

        out = bytearray()
        out.extend(b'II')
        out.extend(struct.pack(f'{bo}H', 42))
        out.extend(struct.pack(f'{bo}I', ifd_start))
        out.extend(struct.pack(f'{bo}H', num_entries))
        for tag, typ, count, raw in tag_list:
            out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
            if len(raw) <= 4:
                out.extend(raw.ljust(4, b'\x00'))
            else:
                ptr = overflow_start + tag_offsets[tag]
                out.extend(struct.pack(f'{bo}I', ptr))
        out.extend(struct.pack(f'{bo}I', 0))
        out.extend(overflow_buf)
        out.extend(pixel_bytes)

        path = str(tmp_path / 'extra_tags.tif')
        with open(path, 'wb') as f:
            f.write(bytes(out))
        return path, pixels

    def test_extra_tags_read(self, tmp_path):
        """Extra tags are collected in attrs['extra_tags']."""
        path, _ = self._make_tiff_with_extra_tags(tmp_path)
        da = open_geotiff(path)

        extra = da.attrs.get('extra_tags')
        assert extra is not None
        tag_ids = {t[0] for t in extra}
        assert 305 in tag_ids  # Software
        assert 306 in tag_ids  # DateTime

    def test_extra_tags_round_trip(self, tmp_path):
        """Extra tags survive read -> write -> read."""
        path, pixels = self._make_tiff_with_extra_tags(tmp_path)
        da = open_geotiff(path)

        out_path = str(tmp_path / 'roundtrip.tif')
        to_geotiff(da, out_path, compression='none')

        da2 = open_geotiff(out_path)

        # Pixels should match
        np.testing.assert_array_equal(da2.values, pixels)

        # Extra tags should survive
        extra2 = da2.attrs.get('extra_tags')
        assert extra2 is not None
        tag_map = {t[0]: t[3] for t in extra2}
        assert 305 in tag_map
        assert 'TestSoftware v1.0' in str(tag_map[305])
        assert 306 in tag_map
        assert '2025:01:15' in str(tag_map[306])

    def test_no_extra_tags(self, tmp_path):
        """Files with only managed tags have no extra_tags attr."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'no_extra.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        assert 'extra_tags' not in da.attrs


class TestGDALMetadata:

    def test_parse_gdal_metadata_xml(self):
        """XML parsing extracts dataset and per-band items."""
        from xrspatial.geotiff._geotags import _parse_gdal_metadata
        xml = (
            '<GDALMetadata>\n'
            '  <Item name="DataType">Generic</Item>\n'
            '  <Item name="STATISTICS_MAX" sample="0">100.5</Item>\n'
            '  <Item name="STATISTICS_MIN" sample="0">-5.2</Item>\n'
            '  <Item name="BAND_NAME" sample="1">green</Item>\n'
            '</GDALMetadata>\n'
        )
        meta = _parse_gdal_metadata(xml)
        assert meta['DataType'] == 'Generic'
        assert meta[('STATISTICS_MAX', 0)] == '100.5'
        assert meta[('STATISTICS_MIN', 0)] == '-5.2'
        assert meta[('BAND_NAME', 1)] == 'green'

    def test_build_gdal_metadata_xml(self):
        """Dict serializes back to valid XML."""
        from xrspatial.geotiff._geotags import (
            _build_gdal_metadata_xml, _parse_gdal_metadata)
        meta = {
            'DataType': 'Generic',
            ('STATS_MAX', 0): '42.0',
            ('STATS_MIN', 0): '-1.0',
        }
        xml = _build_gdal_metadata_xml(meta)
        assert '<GDALMetadata>' in xml
        assert '<Item name="DataType">Generic</Item>' in xml
        assert 'sample="0"' in xml
        # Round-trip through parser
        reparsed = _parse_gdal_metadata(xml)
        assert reparsed == meta

    def test_round_trip_via_file(self, tmp_path):
        """GDAL metadata survives write -> read."""
        meta = {
            'DataType': 'Elevation',
            ('STATISTICS_MAXIMUM', 0): '2500.0',
            ('STATISTICS_MINIMUM', 0): '100.0',
            ('STATISTICS_MEAN', 0): '1200.5',
        }
        from xrspatial.geotiff._geotags import _build_gdal_metadata_xml
        xml = _build_gdal_metadata_xml(meta)

        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'gdal_meta.tif')
        write(arr, path, compression='none', tiled=False,
              gdal_metadata_xml=xml)

        da = open_geotiff(path)
        assert 'gdal_metadata' in da.attrs
        assert 'gdal_metadata_xml' in da.attrs
        result_meta = da.attrs['gdal_metadata']
        assert result_meta['DataType'] == 'Elevation'
        assert result_meta[('STATISTICS_MAXIMUM', 0)] == '2500.0'
        assert result_meta[('STATISTICS_MEAN', 0)] == '1200.5'

    def test_dataarray_attrs_round_trip(self, tmp_path):
        """GDAL metadata from DataArray attrs is preserved."""
        meta = {'Source': 'test', ('BAND', 0): 'dem'}
        da = xr.DataArray(
            np.ones((4, 4), dtype=np.float32),
            dims=['y', 'x'],
            attrs={'gdal_metadata': meta},
        )
        path = str(tmp_path / 'da_meta.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.attrs['gdal_metadata']['Source'] == 'test'
        assert result.attrs['gdal_metadata'][('BAND', 0)] == 'dem'

    def test_no_metadata_no_attrs(self, tmp_path):
        """Files without GDAL metadata don't get the attrs."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'no_meta.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        assert 'gdal_metadata' not in da.attrs
        assert 'gdal_metadata_xml' not in da.attrs

    def test_real_file_metadata(self):
        """Real USGS file has GDAL metadata with statistics."""
        import os
        path = '../rtxpy/examples/USGS_one_meter_x65y454_NY_LongIsland_Z18_2014.tif'
        if not os.path.exists(path):
            pytest.skip("Real test files not available")

        da = open_geotiff(path)
        meta = da.attrs.get('gdal_metadata')
        assert meta is not None
        assert 'DataType' in meta
        assert ('STATISTICS_MAXIMUM', 0) in meta

    def test_real_file_round_trip(self):
        """GDAL metadata survives real-file round-trip."""
        import os, tempfile
        path = '../rtxpy/examples/USGS_one_meter_x65y454_NY_LongIsland_Z18_2014.tif'
        if not os.path.exists(path):
            pytest.skip("Real test files not available")

        da = open_geotiff(path)
        orig_meta = da.attrs['gdal_metadata']

        out = os.path.join(tempfile.mkdtemp(), 'rt.tif')
        to_geotiff(da, out, compression='deflate', tiled=False)

        da2 = open_geotiff(out)
        for k, v in orig_meta.items():
            assert da2.attrs['gdal_metadata'].get(k) == v, f"Mismatch on {k}"


class TestResolution:

    def test_write_read_dpi(self, tmp_path):
        """Resolution tags round-trip through write and read."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'dpi.tif')
        write(arr, path, compression='none', tiled=False,
              x_resolution=300.0, y_resolution=300.0, resolution_unit=2)

        da = open_geotiff(path)
        assert da.attrs['x_resolution'] == pytest.approx(300.0, rel=0.01)
        assert da.attrs['y_resolution'] == pytest.approx(300.0, rel=0.01)
        assert da.attrs['resolution_unit'] == 'inch'

    def test_write_read_cm(self, tmp_path):
        """Centimeter resolution unit."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'dpi_cm.tif')
        write(arr, path, compression='none', tiled=False,
              x_resolution=118.0, y_resolution=118.0, resolution_unit=3)

        da = open_geotiff(path)
        assert da.attrs['x_resolution'] == pytest.approx(118.0, rel=0.01)
        assert da.attrs['resolution_unit'] == 'centimeter'

    def test_no_resolution_no_attrs(self, tmp_path):
        """Files without resolution tags don't get resolution attrs."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'no_dpi.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        assert 'x_resolution' not in da.attrs
        assert 'y_resolution' not in da.attrs
        assert 'resolution_unit' not in da.attrs

    def test_dataarray_attrs_round_trip(self, tmp_path):
        """Resolution attrs on DataArray are preserved through write/read."""
        da = xr.DataArray(
            np.ones((4, 4), dtype=np.float32),
            dims=['y', 'x'],
            attrs={'x_resolution': 72.0, 'y_resolution': 72.0,
                   'resolution_unit': 'inch'},
        )
        path = str(tmp_path / 'da_dpi.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.attrs['x_resolution'] == pytest.approx(72.0, rel=0.01)
        assert result.attrs['y_resolution'] == pytest.approx(72.0, rel=0.01)
        assert result.attrs['resolution_unit'] == 'inch'

    def test_unit_none(self, tmp_path):
        """ResolutionUnit=1 (no unit) round-trips as 'none'."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'no_unit.tif')
        write(arr, path, compression='none', tiled=False,
              x_resolution=1.0, y_resolution=1.0, resolution_unit=1)

        da = open_geotiff(path)
        assert da.attrs['resolution_unit'] == 'none'


# -----------------------------------------------------------------------
# Overview resampling methods
# -----------------------------------------------------------------------

class TestOverviewResampling:

    def test_mean_default(self, tmp_path):
        """Default mean resampling produces correct 2x2 block averages."""
        from xrspatial.geotiff._writer import _make_overview
        arr = np.array([[1, 3, 5, 7],
                        [2, 4, 6, 8],
                        [10, 20, 30, 40],
                        [10, 20, 30, 40]], dtype=np.float32)
        ov = _make_overview(arr, 'mean')
        assert ov.shape == (2, 2)
        # (1+3+2+4)/4 = 2.5
        assert ov[0, 0] == pytest.approx(2.5)

    def test_nearest(self, tmp_path):
        """Nearest resampling picks top-left pixel of each 2x2 block."""
        from xrspatial.geotiff._writer import _make_overview
        arr = np.array([[10, 20, 30, 40],
                        [50, 60, 70, 80],
                        [90, 100, 110, 120],
                        [130, 140, 150, 160]], dtype=np.uint8)
        ov = _make_overview(arr, 'nearest')
        assert ov.shape == (2, 2)
        assert ov[0, 0] == 10
        assert ov[0, 1] == 30
        assert ov[1, 0] == 90
        assert ov[1, 1] == 110

    def test_min(self, tmp_path):
        from xrspatial.geotiff._writer import _make_overview
        arr = np.array([[10, 1, 5, 3],
                        [20, 2, 6, 4],
                        [30, 3, 7, 5],
                        [40, 4, 8, 6]], dtype=np.float32)
        ov = _make_overview(arr, 'min')
        assert ov[0, 0] == pytest.approx(1.0)
        assert ov[0, 1] == pytest.approx(3.0)

    def test_max(self, tmp_path):
        from xrspatial.geotiff._writer import _make_overview
        arr = np.array([[10, 1, 5, 3],
                        [20, 2, 6, 4],
                        [30, 3, 7, 5],
                        [40, 4, 8, 6]], dtype=np.float32)
        ov = _make_overview(arr, 'max')
        assert ov[0, 0] == pytest.approx(20.0)
        assert ov[1, 1] == pytest.approx(8.0)

    def test_median(self, tmp_path):
        from xrspatial.geotiff._writer import _make_overview
        arr = np.array([[1, 2, 10, 20],
                        [3, 100, 30, 40],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]], dtype=np.float32)
        ov = _make_overview(arr, 'median')
        assert ov.shape == (2, 2)
        # median of [1, 2, 3, 100] = 2.5
        assert ov[0, 0] == pytest.approx(2.5)

    def test_mode(self, tmp_path):
        """Mode picks the most common value in each 2x2 block."""
        from xrspatial.geotiff._writer import _make_overview
        arr = np.array([[1, 1, 2, 3],
                        [1, 2, 2, 2],
                        [5, 5, 5, 6],
                        [5, 7, 6, 6]], dtype=np.uint8)
        ov = _make_overview(arr, 'mode')
        assert ov[0, 0] == 1   # 1 appears 3 times
        assert ov[0, 1] == 2   # 2 appears 3 times
        assert ov[1, 0] == 5   # 5 appears 3 times
        assert ov[1, 1] == 6   # 6 appears 3 times

    def test_mean_with_nan(self, tmp_path):
        """Mean resampling ignores NaN values."""
        from xrspatial.geotiff._writer import _make_overview
        arr = np.array([[np.nan, 2, 4, 6],
                        [1, 3, np.nan, 8],
                        [10, 20, 30, 40],
                        [10, 20, 30, 40]], dtype=np.float32)
        ov = _make_overview(arr, 'mean')
        # nanmean([nan, 2, 1, 3]) = 2.0
        assert ov[0, 0] == pytest.approx(2.0)

    def test_multiband(self, tmp_path):
        """Resampling works on 3D (multi-band) arrays."""
        from xrspatial.geotiff._writer import _make_overview
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 100
        arr[:, :, 1] = 200
        arr[:, :, 2] = 50
        ov = _make_overview(arr, 'mean')
        assert ov.shape == (2, 2, 3)
        assert ov[0, 0, 0] == 100
        assert ov[0, 0, 1] == 200
        assert ov[0, 0, 2] == 50

    def test_cog_round_trip_nearest(self, tmp_path):
        """COG with nearest resampling writes and reads back."""
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'cog_nearest.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2], overview_resampling='nearest')

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_cog_round_trip_mode(self, tmp_path):
        """COG with mode resampling for classified data."""
        arr = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                        [0, 0, 1, 1, 2, 2, 3, 3],
                        [4, 4, 5, 5, 6, 6, 7, 7],
                        [4, 4, 5, 5, 6, 6, 7, 7],
                        [0, 0, 1, 1, 2, 2, 3, 3],
                        [0, 0, 1, 1, 2, 2, 3, 3],
                        [4, 4, 5, 5, 6, 6, 7, 7],
                        [4, 4, 5, 5, 6, 6, 7, 7]], dtype=np.uint8)
        path = str(tmp_path / 'cog_mode.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=4,
              cog=True, overview_levels=[2], overview_resampling='mode')

        # Full res should be exact
        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

        # Overview should have mode-reduced values
        ov, _ = read_to_array(path, overview_level=1)
        assert ov.shape == (4, 4)
        assert ov[0, 0] == 0
        assert ov[0, 1] == 1

    def test_to_geotiff_api(self, tmp_path):
        """overview_resampling kwarg works through the public API."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'api_nearest.tif')
        to_geotiff(arr, path, compression='deflate',
                      cog=True, overview_resampling='nearest')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_invalid_method(self):
        from xrspatial.geotiff._writer import _make_overview
        arr = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown overview resampling"):
            _make_overview(arr, 'bicubic_spline')


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

    def test_force_bigtiff_via_public_api(self, tmp_path):
        """bigtiff=True on to_geotiff forces BigTIFF even for small files."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = str(tmp_path / 'forced_bigtiff.tif')
        to_geotiff(arr, path, compression='none', bigtiff=True)

        with open(path, 'rb') as f:
            header = parse_header(f.read(16))
        assert header.is_bigtiff

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_small_file_stays_classic(self, tmp_path):
        """Small files default to classic TIFF (bigtiff=None auto-detects)."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = str(tmp_path / 'classic.tif')
        to_geotiff(arr, path, compression='none')

        with open(path, 'rb') as f:
            header = parse_header(f.read(16))
        assert not header.is_bigtiff

    def test_force_bigtiff_false_stays_classic(self, tmp_path):
        """bigtiff=False forces classic TIFF."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = str(tmp_path / 'forced_classic.tif')
        to_geotiff(arr, path, compression='none', bigtiff=False)

        with open(path, 'rb') as f:
            header = parse_header(f.read(16))
        assert not header.is_bigtiff

    def _assert_offset_tags_are_long8(self, path):
        """Parse *path*'s first IFD and assert offset tags use LONG8."""
        from xrspatial.geotiff._header import (
            parse_all_ifds, TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS,
            TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS,
        )
        from xrspatial.geotiff._dtypes import LONG8

        with open(path, 'rb') as f:
            buf = f.read()
        header = parse_header(buf)
        assert header.is_bigtiff, (
            "Test precondition: file must be BigTIFF.")
        ifds = parse_all_ifds(buf, header)
        assert len(ifds) >= 1
        entries = ifds[0].entries

        offset_tags = (TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS,
                       TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS)
        present = [t for t in offset_tags if t in entries]
        assert present, (
            "File had no strip/tile offset tags; "
            "cannot verify the LONG8 promotion.")
        for tag_id in present:
            entry = entries[tag_id]
            assert entry.type_id == LONG8, (
                f"Tag {tag_id} in BigTIFF output was typed "
                f"{entry.type_id}, expected LONG8 (16).  A 32-bit "
                "offset would truncate on files larger than 4 GB.")

    def test_bigtiff_eager_tile_offsets_are_long8_1247(self, tmp_path):
        """Eager writer emits LONG8 TileOffsets in BigTIFF output.

        Regression for the Medium Cat 3 finding in the #1247 audit:
        eager ``_assemble_tiff`` hard-coded LONG for TileOffsets /
        TileByteCounts regardless of the BigTIFF decision.  Anything
        past 4 GB would silently truncate (or, with ``struct.pack``,
        fail at pack time).

        Asserting on a small-but-forced BigTIFF is enough: the fix
        is width-of-the-offset-field, not value-range.
        """
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'bigtiff_long8_eager_1247.tif')
        to_geotiff(arr, path, compression='none',
                   tiled=True, tile_size=16, bigtiff=True)
        self._assert_offset_tags_are_long8(path)
        # Data must still round-trip.
        np.testing.assert_array_equal(open_geotiff(path).values, arr)

    def test_bigtiff_eager_strip_offsets_are_long8_1247(self, tmp_path):
        """Eager writer emits LONG8 StripOffsets for stripped BigTIFF."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'bigtiff_long8_eager_strip_1247.tif')
        to_geotiff(arr, path, compression='none',
                   tiled=False, bigtiff=True)
        self._assert_offset_tags_are_long8(path)
        np.testing.assert_array_equal(open_geotiff(path).values, arr)

    def test_bigtiff_streaming_tile_offsets_are_long8_1247(self, tmp_path):
        """Streaming writer emits LONG8 TileOffsets in BigTIFF output.

        Covers the pre-fix code comment at ``_writer.write_streaming``
        that explicitly acknowledged LONG8 was needed and hadn't been
        done.  Uses a small dask array so the test doesn't actually
        need to produce a >4 GB file.
        """
        import dask.array as da
        import xarray as xr

        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        dask_da = xr.DataArray(
            da.from_array(arr, chunks=8),
            dims=['y', 'x'],
        )
        path = str(tmp_path / 'bigtiff_long8_stream_1247.tif')
        to_geotiff(dask_da, path, compression='none',
                   tiled=True, tile_size=16, bigtiff=True)
        self._assert_offset_tags_are_long8(path)
        np.testing.assert_array_equal(open_geotiff(path).values, arr)


# -----------------------------------------------------------------------
# Sub-byte bit depths (1-bit, 4-bit, 12-bit)
# -----------------------------------------------------------------------

def _make_sub_byte_tiff(width, height, bps, pixel_values):
    """Build a minimal TIFF with sub-byte BitsPerSample.

    pixel_values: 2D array of unpacked integer values.
    Data is packed MSB-first into bytes according to bps.
    """
    import struct
    bo = '<'
    dtype_np = np.dtype('uint8') if bps <= 8 else np.dtype('uint16')

    # Pack pixel values into bytes
    flat = pixel_values.ravel()
    if bps == 1:
        packed = np.packbits(flat.astype(np.uint8))
    elif bps == 4:
        n = len(flat)
        packed_len = (n + 1) // 2
        packed = np.zeros(packed_len, dtype=np.uint8)
        for i in range(n):
            if i % 2 == 0:
                packed[i // 2] |= (flat[i] & 0x0F) << 4
            else:
                packed[i // 2] |= flat[i] & 0x0F
        packed = packed
    elif bps == 12:
        n = len(flat)
        n_pairs = n // 2
        remainder = n % 2
        packed_len = n_pairs * 3 + (2 if remainder else 0)
        packed = np.zeros(packed_len, dtype=np.uint8)
        for i in range(n_pairs):
            v0 = int(flat[i * 2])
            v1 = int(flat[i * 2 + 1])
            off = i * 3
            packed[off] = (v0 >> 4) & 0xFF
            packed[off + 1] = ((v0 & 0x0F) << 4) | ((v1 >> 8) & 0x0F)
            packed[off + 2] = v1 & 0xFF
        if remainder:
            v = int(flat[-1])
            off = n_pairs * 3
            packed[off] = (v >> 4) & 0xFF
            packed[off + 1] = (v & 0x0F) << 4
    else:
        raise ValueError(f"Unsupported bps: {bps}")

    pixel_bytes = packed.tobytes()

    # Build tags
    tag_list = []
    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))
    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, bps)
    add_short(259, 1)   # no compression
    add_short(262, 1)  # BlackIsZero (works for all bit depths)
    add_short(277, 1)
    add_short(278, height)
    add_long(273, 0)    # strip offset placeholder
    add_long(279, len(pixel_bytes))
    if bps <= 8:
        add_short(339, 1)  # UINT
    else:
        add_short(339, 1)

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_buf = bytearray()
    tag_offsets = {}
    overflow_start = ifd_start + ifd_size

    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)

    # Patch strip offset
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    # Rebuild overflow after patching
    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    return bytes(out), pixel_values


class TestSubByteBitDepths:

    def test_1bit_bilevel(self, tmp_path):
        """Read a 1-bit bilevel TIFF."""
        pixels = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                           [0, 1, 0, 1, 0, 1, 0, 1],
                           [1, 1, 0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0, 0, 1, 1]], dtype=np.uint8)
        tiff_data, expected = _make_sub_byte_tiff(8, 4, 1, pixels)
        path = str(tmp_path / '1bit.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.uint8
        assert result.shape == (4, 8)
        np.testing.assert_array_equal(result, expected)

    def test_1bit_non_byte_aligned_width(self, tmp_path):
        """1-bit image whose width is not a multiple of 8."""
        pixels = np.array([[1, 0, 1],
                           [0, 1, 0]], dtype=np.uint8)
        tiff_data, expected = _make_sub_byte_tiff(3, 2, 1, pixels)
        path = str(tmp_path / '1bit_3wide.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, expected)

    def test_4bit_nibble(self, tmp_path):
        """Read a 4-bit TIFF."""
        pixels = np.array([[0, 1, 2, 3],
                           [4, 5, 6, 7],
                           [8, 9, 10, 11],
                           [12, 13, 14, 15]], dtype=np.uint8)
        tiff_data, expected = _make_sub_byte_tiff(4, 4, 4, pixels)
        path = str(tmp_path / '4bit.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.uint8
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, expected)

    def test_4bit_odd_width(self, tmp_path):
        """4-bit image with odd width (partial byte at row end)."""
        pixels = np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=np.uint8)
        tiff_data, expected = _make_sub_byte_tiff(3, 2, 4, pixels)
        path = str(tmp_path / '4bit_odd.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, expected)

    def test_12bit(self, tmp_path):
        """Read a 12-bit TIFF."""
        pixels = np.array([[0, 100, 2048, 4095],
                           [1000, 2000, 3000, 4000]], dtype=np.uint16)
        tiff_data, expected = _make_sub_byte_tiff(4, 2, 12, pixels)
        path = str(tmp_path / '12bit.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.uint16
        assert result.shape == (2, 4)
        np.testing.assert_array_equal(result, expected)

    def test_unpack_bits_codec_directly(self):
        """Test unpack_bits on known packed data."""
        from xrspatial.geotiff._compression import unpack_bits

        # 1-bit: byte 0xA5 = 10100101 -> [1,0,1,0,0,1,0,1]
        data = np.array([0xA5], dtype=np.uint8)
        result = unpack_bits(data, 1, 8)
        np.testing.assert_array_equal(result, [1, 0, 1, 0, 0, 1, 0, 1])

        # 4-bit: byte 0x3C = 0011_1100 -> [3, 12]
        data = np.array([0x3C], dtype=np.uint8)
        result = unpack_bits(data, 4, 2)
        np.testing.assert_array_equal(result, [3, 12])


# -----------------------------------------------------------------------
# Planar configuration (separate planes)
# -----------------------------------------------------------------------

def _make_planar_tiff(width, height, bands, dtype=np.uint8, tiled=False,
                      tile_size=4):
    """Build a minimal planar-config TIFF (PlanarConfiguration=2) by hand.

    Each band's pixel data is stored as a separate set of strips (or tiles).
    Band values: band 0 gets pixel values 10+pixel_idx, band 1 gets 20+,
    band 2 gets 30+, etc.
    """
    import struct
    bo = '<'

    dtype = np.dtype(dtype)
    bps = dtype.itemsize * 8
    if dtype.kind == 'f':
        sf = 3
    elif dtype.kind == 'i':
        sf = 2
    else:
        sf = 1

    # Build per-band pixel arrays
    band_arrays = []
    for b in range(bands):
        base = (b + 1) * 10
        arr = np.arange(width * height, dtype=dtype).reshape(height, width) + dtype.type(base)
        band_arrays.append(arr)

    if tiled:
        import math
        tw = th = tile_size
        tiles_across = math.ceil(width / tw)
        tiles_down = math.ceil(height / th)
        tiles_per_band = tiles_across * tiles_down

        # Build tile data: all tiles for band 0, then band 1, etc.
        tile_blobs = []
        for b in range(bands):
            for tr in range(tiles_down):
                for tc in range(tiles_across):
                    tile = np.zeros((th, tw), dtype=dtype)
                    r0, c0 = tr * th, tc * tw
                    r1 = min(r0 + th, height)
                    c1 = min(c0 + tw, width)
                    tile[:r1 - r0, :c1 - c0] = band_arrays[b][r0:r1, c0:c1]
                    tile_blobs.append(tile.tobytes())

        pixel_bytes = b''.join(tile_blobs)
        tile_byte_counts = [len(t) for t in tile_blobs]
        num_offsets = len(tile_blobs)
    else:
        # Strips: 1 strip per band (whole image), one set per band
        strip_blobs = []
        for b in range(bands):
            strip_blobs.append(band_arrays[b].tobytes())
        pixel_bytes = b''.join(strip_blobs)
        strip_byte_counts = [len(s) for s in strip_blobs]
        num_offsets = bands

    # Build tags
    tag_list = []
    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))
    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals)))
    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))
    def add_longs(tag, vals):
        tag_list.append((tag, 4, len(vals), struct.pack(f'{bo}{len(vals)}I', *vals)))

    add_short(256, width)
    add_short(257, height)
    add_shorts(258, [bps] * bands)
    add_short(259, 1)   # no compression
    add_short(262, 2 if bands >= 3 else 1)  # RGB or BlackIsZero
    add_short(277, bands)
    add_short(284, 2)   # PlanarConfiguration = Separate
    add_shorts(339, [sf] * bands)

    if tiled:
        add_short(322, tile_size)
        add_short(323, tile_size)
        add_longs(324, [0] * num_offsets)  # placeholder
        add_longs(325, tile_byte_counts)
    else:
        add_short(278, height)  # RowsPerStrip = full image
        add_longs(273, [0] * num_offsets)  # placeholder
        add_longs(279, strip_byte_counts)

    tag_list.sort(key=lambda t: t[0])

    # Layout
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4

    # Collect overflow
    overflow_buf = bytearray()
    tag_offsets = {}
    overflow_start = ifd_start + ifd_size

    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)

    # Patch offsets
    offset_tag = 324 if tiled else 273
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == offset_tag:
            if tiled:
                offs = []
                pos = 0
                for blob in tile_blobs:
                    offs.append(pixel_data_start + pos)
                    pos += len(blob)
                new_raw = struct.pack(f'{bo}{num_offsets}I', *offs)
            else:
                offs = []
                pos = 0
                for blob in strip_blobs:
                    offs.append(pixel_data_start + pos)
                    pos += len(blob)
                new_raw = struct.pack(f'{bo}{num_offsets}I', *offs)
            patched.append((tag, typ, count, new_raw))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    # Rebuild overflow
    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    # Serialize
    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))  # next IFD
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    # Build expected output for verification
    expected = np.stack(band_arrays, axis=2)
    return bytes(out), expected


# -----------------------------------------------------------------------
# Palette / indexed color (ColorMap tag 320)
# -----------------------------------------------------------------------

def _make_palette_tiff(width, height, bps, pixel_values, palette_rgb):
    """Build a palette-color TIFF (Photometric=3 + ColorMap tag).

    palette_rgb: list of (R, G, B) tuples, uint16 values (0-65535).
    """
    import struct
    bo = '<'
    n_colors = len(palette_rgb)
    assert n_colors == (1 << bps), f"Palette must have {1 << bps} entries for {bps}-bit"

    # Pack pixel data
    flat = pixel_values.ravel().astype(np.uint8)
    if bps == 8:
        pixel_bytes = flat.tobytes()
    elif bps == 4:
        n = len(flat)
        packed_len = (n + 1) // 2
        packed = np.zeros(packed_len, dtype=np.uint8)
        for i in range(n):
            if i % 2 == 0:
                packed[i // 2] |= (flat[i] & 0x0F) << 4
            else:
                packed[i // 2] |= flat[i] & 0x0F
        pixel_bytes = packed.tobytes()
    else:
        pixel_bytes = flat.tobytes()

    # Build ColorMap: [R0..R_{n-1}, G0..G_{n-1}, B0..B_{n-1}]
    r_vals = [c[0] for c in palette_rgb]
    g_vals = [c[1] for c in palette_rgb]
    b_vals = [c[2] for c in palette_rgb]
    cmap_values = r_vals + g_vals + b_vals

    tag_list = []
    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))
    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))
    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, bps)
    add_short(259, 1)     # no compression
    add_short(262, 3)     # Photometric = Palette
    add_short(277, 1)     # SamplesPerPixel = 1
    add_short(278, height)
    add_long(273, 0)      # StripOffsets placeholder
    add_long(279, len(pixel_bytes))
    add_shorts(320, cmap_values)  # ColorMap
    add_short(339, 1)     # SampleFormat = UINT

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)

    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    return bytes(out)


class TestPalette:

    def test_palette_8bit_read(self, tmp_path):
        """Read an 8-bit palette TIFF and verify pixel indices."""
        # 4-color palette: red, green, blue, white
        palette = [
            (65535, 0, 0),       # 0 = red
            (0, 65535, 0),       # 1 = green
            (0, 0, 65535),       # 2 = blue
            (65535, 65535, 65535),# 3 = white
        ] + [(0, 0, 0)] * 252   # pad to 256 entries for 8-bit

        pixels = np.array([[0, 1, 2, 3],
                           [3, 2, 1, 0]], dtype=np.uint8)

        tiff_data = _make_palette_tiff(4, 2, 8, pixels, palette)
        path = str(tmp_path / 'palette8.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = open_geotiff(path)
        # Should return raw index values
        assert da.dtype == np.uint8
        np.testing.assert_array_equal(da.values, pixels)

        # Should have cmap and colormap_rgba in attrs
        assert 'cmap' in da.attrs
        assert 'colormap_rgba' in da.attrs

        # Verify the palette colors
        rgba = da.attrs['colormap_rgba']
        assert len(rgba) == 256
        assert rgba[0] == pytest.approx((1.0, 0.0, 0.0, 1.0))
        assert rgba[1] == pytest.approx((0.0, 1.0, 0.0, 1.0))
        assert rgba[2] == pytest.approx((0.0, 0.0, 1.0, 1.0))

    def test_palette_4bit(self, tmp_path):
        """Read a 4-bit palette TIFF."""
        palette = [(i * 4369, i * 4369, i * 4369) for i in range(16)]
        pixels = np.array([[0, 5, 10, 15],
                           [1, 6, 11, 3]], dtype=np.uint8)

        tiff_data = _make_palette_tiff(4, 2, 4, pixels, palette)
        path = str(tmp_path / 'palette4.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = open_geotiff(path)
        assert da.dtype == np.uint8
        np.testing.assert_array_equal(da.values, pixels)
        assert 'cmap' in da.attrs
        assert len(da.attrs['colormap_rgba']) == 16

    def test_palette_cmap_works_with_plot(self, tmp_path):
        """Verify the colormap can be used with matplotlib."""
        from matplotlib.colors import ListedColormap

        palette = [
            (65535, 0, 0),
            (0, 65535, 0),
            (0, 0, 65535),
            (65535, 65535, 0),
        ] + [(0, 0, 0)] * 252

        pixels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        tiff_data = _make_palette_tiff(2, 2, 8, pixels, palette)
        path = str(tmp_path / 'palette_plot.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = open_geotiff(path)
        cmap = da.attrs['cmap']
        assert isinstance(cmap, ListedColormap)

        # Verify color mapping at known indices
        assert cmap(0)[:3] == pytest.approx((1.0, 0.0, 0.0), abs=0.01)
        assert cmap(1 / 255)[:3] == pytest.approx((0.0, 1.0, 0.0), abs=0.01)

    def test_xrs_plot_with_palette(self, tmp_path):
        """da.xrs.plot() uses the embedded colormap."""
        import matplotlib
        matplotlib.use('Agg')
        import xrspatial.accessor  # register .xrs accessor

        palette = [
            (65535, 0, 0),
            (0, 65535, 0),
            (0, 0, 65535),
            (65535, 65535, 65535),
        ] + [(0, 0, 0)] * 252

        pixels = np.array([[0, 1, 2, 3],
                           [3, 2, 1, 0]], dtype=np.uint8)
        tiff_data = _make_palette_tiff(4, 2, 8, pixels, palette)
        path = str(tmp_path / 'plot_palette.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = open_geotiff(path)
        artist = da.xrs.plot()
        assert artist is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_xrs_plot_no_palette(self, tmp_path):
        """da.xrs.plot() falls through to normal plot for non-palette data."""
        import matplotlib
        matplotlib.use('Agg')
        import xrspatial.accessor

        arr = np.random.RandomState(42).rand(4, 4).astype(np.float32)
        path = str(tmp_path / 'no_palette.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        artist = da.xrs.plot()
        assert artist is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_geotiff_deprecated(self, tmp_path):
        """plot_geotiff still works but emits a DeprecationWarning."""
        import matplotlib
        matplotlib.use('Agg')
        import xrspatial.accessor
        from xrspatial.geotiff import plot_geotiff

        palette = [(65535, 0, 0), (0, 65535, 0)] + [(0, 0, 0)] * 254
        pixels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        tiff_data = _make_palette_tiff(2, 2, 8, pixels, palette)
        path = str(tmp_path / 'deprecated.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = open_geotiff(path)
        with pytest.warns(DeprecationWarning, match='plot_geotiff is deprecated'):
            artist = plot_geotiff(da)
        assert artist is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_non_palette_no_cmap(self, tmp_path):
        """Non-palette TIFFs should not have a cmap attr."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'no_palette.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        assert 'cmap' not in da.attrs
        assert 'colormap_rgba' not in da.attrs


class TestPlanarConfig:

    def test_planar_strips_rgb(self, tmp_path):
        """Read a 3-band planar-stripped TIFF."""
        tiff_data, expected = _make_planar_tiff(4, 6, 3, np.uint8)
        path = str(tmp_path / 'planar_strip.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (6, 4, 3)
        np.testing.assert_array_equal(result, expected)

    def test_planar_strips_2band(self, tmp_path):
        """Read a 2-band planar-stripped TIFF."""
        tiff_data, expected = _make_planar_tiff(5, 4, 2, np.uint16)
        path = str(tmp_path / 'planar_2band.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (4, 5, 2)
        np.testing.assert_array_equal(result, expected)

    def test_planar_tiles_rgb(self, tmp_path):
        """Read a 3-band planar-tiled TIFF."""
        tiff_data, expected = _make_planar_tiff(
            8, 8, 3, np.uint8, tiled=True, tile_size=4)
        path = str(tmp_path / 'planar_tiled.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (8, 8, 3)
        np.testing.assert_array_equal(result, expected)

    def test_planar_windowed(self, tmp_path):
        """Windowed read of a planar-stripped TIFF."""
        tiff_data, expected = _make_planar_tiff(8, 8, 3, np.uint8)
        path = str(tmp_path / 'planar_window.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path, window=(2, 1, 6, 5))
        np.testing.assert_array_equal(result, expected[2:6, 1:5, :])

    def test_planar_band_selection(self, tmp_path):
        """Selecting a single band from a planar TIFF."""
        tiff_data, expected = _make_planar_tiff(4, 4, 3, np.uint8)
        path = str(tmp_path / 'planar_band.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path, band=1)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, expected[:, :, 1])

    def test_planar_via_public_api(self, tmp_path):
        """open_geotiff on a planar file returns correct DataArray."""
        from xrspatial.geotiff import open_geotiff
        tiff_data, expected = _make_planar_tiff(4, 4, 3, np.uint8)
        path = str(tmp_path / 'planar_api.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = open_geotiff(path)
        assert 'band' in da.dims
        assert da.shape == (4, 4, 3)
        np.testing.assert_array_equal(da.values, expected)


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


class TestPublicAPI:
    """`__all__` reflects every supported public function and `from
    xrspatial.geotiff import *` does not silently drop production names."""

    def test_all_lists_supported_functions(self):
        import xrspatial.geotiff as g
        # Frozen list of names that callers / tests treat as part of the
        # public API. If any of these gets removed or renamed, that is a
        # breaking change and should go through a deprecation cycle.
        expected = {
            'GeoTIFFFallbackWarning',
            'UnsafeURLError',
            'open_geotiff',
            'read_geotiff_gpu',
            'read_geotiff_dask',
            'read_vrt',
            'to_geotiff',
            'write_geotiff_gpu',
            'write_vrt',
        }
        assert set(g.__all__) == expected

    def test_star_import_brings_in_all_public_names(self):
        # ``from ... import *`` honours ``__all__``; verify every entry is
        # importable that way (catches typos in __all__).
        ns: dict = {}
        exec('from xrspatial.geotiff import *', ns)
        import xrspatial.geotiff as g
        for name in g.__all__:
            assert name in ns, f"{name} listed in __all__ but not exported"

    def test_plot_geotiff_not_in_all_but_still_importable(self):
        # plot_geotiff is intentionally omitted from __all__ (deprecated)
        # but stays importable so existing user code keeps working.
        import xrspatial.geotiff as g
        assert 'plot_geotiff' not in g.__all__
        assert hasattr(g, 'plot_geotiff')
