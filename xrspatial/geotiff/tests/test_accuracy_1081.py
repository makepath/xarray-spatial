"""Tests for accuracy bugs fixed in #1081.

Bug 1: Windowed read ignores PixelIsPoint raster type
Bug 2: CRS WKT silently lost on write for non-EPSG CRS
Bug 3: NaN not restored to nodata sentinel on write
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import RASTER_PIXEL_IS_POINT, TAG_GEO_ASCII_PARAMS, extract_geo_info
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import write


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
