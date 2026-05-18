"""Round-trip tests for transform / crs / tag metadata (issue #1484).

These cover findings M-1 through M-4 from the geotiff metadata audit:

* M-1 / M-2: ``attrs['crs']`` stays as the same int EPSG and
  ``attrs['transform']`` survives write -> read -> write -> read with
  the same numeric values up to float precision.
* M-3: ColorMap, ExtraSamples, and ImageDescription survive a single
  write -> read cycle. ColorMap exits the writer through the
  ``extra_tags`` pass-through (the tag is no longer in
  ``_MANAGED_TAGS``); ImageDescription gets a friendly ``attrs`` entry.
* M-4: integer rasters with a nodata sentinel get promoted to float64
  with NaN, and a user-requested ``dtype='uint16'`` cast on the read
  side raises ValueError (existing float-to-int guard).
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_palette_uint8_tiff(path, pixels, palette_rgb16):
    """Write an 8-bit, 256-entry palette TIFF directly (no writer support
    for ColorMap on the write side).

    palette_rgb16 must have 256 (R, G, B) tuples of uint16 values.
    """
    bo = '<'
    width = pixels.shape[1]
    height = pixels.shape[0]
    n_colors = 256
    assert len(palette_rgb16) == n_colors

    flat = pixels.ravel().astype(np.uint8)
    pixel_bytes = flat.tobytes()

    r_vals = [c[0] for c in palette_rgb16]
    g_vals = [c[1] for c in palette_rgb16]
    b_vals = [c[2] for c in palette_rgb16]
    cmap_values = r_vals + g_vals + b_vals

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tag_list.append(
            (tag, 3, len(vals),
             struct.pack(f'{bo}{len(vals)}H', *vals)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 8)        # bits per sample
    add_short(259, 1)        # no compression
    add_short(262, 3)        # photometric = palette
    add_short(277, 1)        # samples per pixel = 1
    add_short(278, height)   # rows per strip
    add_long(273, 0)         # strip offsets placeholder
    add_long(279, len(pixel_bytes))
    add_shorts(320, cmap_values)  # ColorMap
    add_short(339, 1)        # sample format = uint

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _typ, _count, raw in tag_list:
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
            patched.append((tag, typ, count,
                            struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _typ, _count, raw in tag_list:
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

    with open(path, 'wb') as f:
        f.write(bytes(out))


def _write_simple_tiff_with_image_description(path, pixels, description):
    """Write an uncompressed, single-strip TIFF that carries an
    ImageDescription tag (270) so we can test the read side."""
    bo = '<'
    height, width = pixels.shape
    pixel_bytes = pixels.astype(np.float32).tobytes()
    desc_bytes = description.encode('ascii') + b'\x00'
    if len(desc_bytes) % 2:
        desc_bytes += b'\x00'

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 32)
    add_short(259, 1)
    add_short(262, 1)
    tag_list.append((270, 2, len(description) + 1, desc_bytes))
    add_short(277, 1)
    add_short(278, height)
    add_long(273, 0)
    add_long(279, len(pixel_bytes))
    add_short(339, 3)  # sample format = float

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _t, _c, raw in tag_list:
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
            patched.append((tag, typ, count,
                            struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _t, _c, raw in tag_list:
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

    with open(path, 'wb') as f:
        f.write(bytes(out))


# ---------------------------------------------------------------------------
# M-1 / M-2: transform & crs round-trip stability
# ---------------------------------------------------------------------------

class TestTransformCrsRoundTrip:

    def test_transform_attr_present_on_read(self, tmp_path):
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        from xrspatial.geotiff._geotags import GeoTransform
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / 'transform_present_1484.tif')
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path)
        assert 'transform' in da.attrs
        a, b, c, d, e, f = da.attrs['transform']
        assert b == 0.0 and d == 0.0
        assert a == pytest.approx(30.0)
        assert e == pytest.approx(-30.0)
        assert c == pytest.approx(500000.0)
        assert f == pytest.approx(4000000.0)
        assert da.attrs['crs'] == 32610

    def test_double_round_trip_fractional_transform(self, tmp_path):
        """Fractional pixel size + non-grid origin: writing twice must not
        drift the transform. This is the case ``_coords_to_transform`` can
        miss because ``x[1] - x[0]`` is recomputed from already-rounded
        coords."""
        from xrspatial.geotiff._geotags import GeoTransform
        arr = np.linspace(0, 1, 8 * 12, dtype=np.float64).reshape(8, 12)
        gt = GeoTransform(
            origin_x=-122.123456789,
            origin_y=37.987654321,
            pixel_width=1.0 / 3600.0 + 1e-12,  # ~ 1 arc-second + tiny offset
            pixel_height=-(1.0 / 3600.0 + 1e-12),
        )
        path1 = str(tmp_path / 'rt1_1484.tif')
        write(arr, path1, geo_transform=gt, crs_epsg=4326,
              compression='none', tiled=False)
        da1 = open_geotiff(path1)
        assert da1.attrs['crs'] == 4326

        path2 = str(tmp_path / 'rt2_1484.tif')
        to_geotiff(da1, path2, compression='none')
        da2 = open_geotiff(path2)

        path3 = str(tmp_path / 'rt3_1484.tif')
        to_geotiff(da2, path3, compression='none')
        da3 = open_geotiff(path3)

        # CRS stays an int EPSG unchanged across cycles
        assert da3.attrs['crs'] == 4326
        # Transform tuple equal up to float precision
        t1 = da1.attrs['transform']
        t3 = da3.attrs['transform']
        for v1, v3 in zip(t1, t3):
            assert v3 == pytest.approx(v1, abs=1e-15, rel=1e-12)

    def test_crs_string_input_still_tolerated(self, tmp_path):
        """Backward compat: passing a WKT string in attrs['crs'] still works
        on the write side. open_geotiff turns it back into an int EPSG."""
        import xarray as xr
        from xrspatial.geotiff._geotags import _epsg_to_wkt
        wkt = _epsg_to_wkt(4326)
        if wkt is None:
            pytest.skip("pyproj not available")
        arr = np.zeros((3, 3), dtype=np.float32)
        da = xr.DataArray(
            arr,
            dims=['y', 'x'],
            coords={
                'y': np.array([0.5, -0.5, -1.5]),
                'x': np.array([0.5, 1.5, 2.5]),
            },
            attrs={'crs': wkt},
        )
        path = str(tmp_path / 'wkt_string_crs_1484.tif')
        to_geotiff(da, path, compression='none')
        result = open_geotiff(path)
        assert result.attrs['crs'] == 4326


# ---------------------------------------------------------------------------
# M-3: tag pass-through (ColorMap, ImageDescription, ExtraSamples)
# ---------------------------------------------------------------------------

class TestTagPassThrough:

    def test_colormap_round_trip(self, tmp_path):
        palette = [(i * 257, (255 - i) * 257, (i * 2) % 65536)
                   for i in range(256)]
        pixels = np.array([[0, 1, 2, 254, 255],
                           [10, 20, 30, 40, 50]], dtype=np.uint8)
        in_path = str(tmp_path / 'colormap_in_1484.tif')
        _make_palette_uint8_tiff(in_path, pixels, palette)

        # Contract v2 (issue #2016) removed the ``cmap`` /
        # ``colormap_rgba`` emit sites; the read no longer fires a
        # ``DeprecationWarning`` on Photometric=3 fixtures and the
        # ``catch_warnings`` shim is no longer needed here.
        da = open_geotiff(in_path)
        assert da.dtype == np.uint8
        assert 'colormap' in da.attrs
        # Raw uint16 ColorMap: 3 * 256 = 768 entries
        assert len(da.attrs['colormap']) == 768

        # Round-trip through to_geotiff: ColorMap rides extra_tags
        out_path = str(tmp_path / 'colormap_out_1484.tif')
        to_geotiff(da, out_path, compression='none')
        da2 = open_geotiff(out_path)

        np.testing.assert_array_equal(da2.values, pixels)
        assert 'colormap' in da2.attrs
        assert tuple(da2.attrs['colormap']) == tuple(da.attrs['colormap'])

    def test_image_description_round_trip(self, tmp_path):
        pixels = np.arange(12, dtype=np.float32).reshape(3, 4)
        desc = "elevation tile from issue 1484"
        in_path = str(tmp_path / 'desc_in_1484.tif')
        _write_simple_tiff_with_image_description(in_path, pixels, desc)

        da = open_geotiff(in_path)
        assert da.attrs.get('image_description') == desc
        # Also reachable through extra_tags by tag id 270
        et_ids = {t[0] for t in da.attrs['extra_tags']}
        assert 270 in et_ids

        out_path = str(tmp_path / 'desc_out_1484.tif')
        to_geotiff(da, out_path, compression='none')
        da2 = open_geotiff(out_path)
        assert da2.attrs.get('image_description') == desc

    def test_image_description_added_via_attrs(self, tmp_path):
        """Setting attrs['image_description'] on a fresh DataArray flows
        through to the output file even when extra_tags is empty."""
        import xarray as xr
        arr = np.zeros((4, 4), dtype=np.float32)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(4), 'x': np.arange(4)},
            attrs={'image_description': 'synthetic test 1484'},
        )
        path = str(tmp_path / 'desc_synth_1484.tif')
        to_geotiff(da, path, compression='none')

        result = open_geotiff(path)
        assert result.attrs.get('image_description') == 'synthetic test 1484'

    def test_extra_samples_attr_surfaces_on_read(self, tmp_path):
        """A 4-band RGBA write produces ExtraSamples internally; reading
        it back surfaces the codes as ``attrs['extra_samples']``. Since
        issue #1769 the RGBA interpretation is opt-in via
        ``photometric='rgba'``; the default is now MinIsBlack and would
        emit ExtraSamples=[0,0,0]."""
        rgba = np.zeros((4, 5, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        path = str(tmp_path / 'rgba_es_1484.tif')
        write(rgba, path, compression='none', tiled=False,
              photometric='rgba')
        da = open_geotiff(path)
        assert da.attrs.get('extra_samples') is not None
        # Code 2 = unassociated alpha, per the writer
        assert da.attrs['extra_samples'][0] in (1, 2)


# ---------------------------------------------------------------------------
# M-4: integer-with-nodata dtype promotion
# ---------------------------------------------------------------------------

class TestIntegerNodataPromotion:

    def test_uint16_with_nodata_promotes_to_float64(self, tmp_path):
        arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
        path = str(tmp_path / 'u16_nodata_1484.tif')
        write(arr, path, nodata=65535, compression='none', tiled=False)

        da = open_geotiff(path)
        assert da.dtype == np.float64
        assert np.isnan(da.values[1, 0])
        np.testing.assert_array_equal(
            da.values[~np.isnan(da.values)],
            np.array([1.0, 2.0, 3.0, 5.0, 6.0]),
        )

    def test_uint16_with_nodata_dtype_uint16_raises(self, tmp_path):
        """Promotion happens before the user-requested dtype check, so
        passing dtype='uint16' on an integer-with-nodata raster hits the
        float-to-int guard and raises ValueError."""
        arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
        path = str(tmp_path / 'u16_nodata_cast_1484.tif')
        write(arr, path, nodata=65535, compression='none', tiled=False)
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='uint16')

    def test_uint16_no_nodata_keeps_dtype(self, tmp_path):
        """Without a nodata sentinel, no promotion; original dtype stays."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        path = str(tmp_path / 'u16_no_nodata_1484.tif')
        write(arr, path, compression='none', tiled=False)
        da = open_geotiff(path)
        assert da.dtype == np.uint16
