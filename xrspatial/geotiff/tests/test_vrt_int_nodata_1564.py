"""VRT integer-with-nodata parity tests for issue #1564.

Before the fix, ``read_vrt`` set ``attrs['nodata']`` on the returned
DataArray but left the integer sentinel value (e.g. 65535) intact in the
pixel array. The eager ``open_geotiff`` path, the dask path, and both
GPU paths all promote integer-with-nodata rasters to float64 and replace
the sentinel with NaN -- the VRT route was the lone divergence.

These tests pin the contract that ``read_vrt`` mirrors that promotion.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    read_vrt,
    to_geotiff,
    write_vrt,
)


def _write_uint16_with_nodata_tif(path, sentinel):
    """Write a small uint16 GeoTIFF with a nodata sentinel."""
    arr = np.array([[1, 2, 3], [sentinel, 5, 6]], dtype=np.uint16)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.arange(2), 'x': np.arange(3)},
        attrs={'crs': 4326, 'nodata': sentinel},
    )
    to_geotiff(da, path, compression='none', nodata=sentinel)
    return arr


def test_vrt_uint16_nodata_promotes_to_float64(tmp_path):
    """VRT route NaN-masks integer-with-nodata, matching open_geotiff."""
    tif = str(tmp_path / 'src_1564.tif')
    _write_uint16_with_nodata_tif(tif, sentinel=65535)

    eager = open_geotiff(tif)
    assert eager.dtype == np.float64
    assert np.isnan(eager.values[1, 0])

    vrt_path = str(tmp_path / 'src_1564.vrt')
    write_vrt(vrt_path, [tif])
    via_vrt = read_vrt(vrt_path)

    assert via_vrt.dtype == np.float64, (
        f"VRT integer-with-nodata should promote to float64; "
        f"got {via_vrt.dtype}"
    )
    assert np.isnan(via_vrt.values[1, 0]), (
        f"VRT sentinel pixel should be NaN; got "
        f"{via_vrt.values[1, 0]} (literal sentinel survived)"
    )
    assert via_vrt.attrs.get('nodata') == 65535.0


def test_vrt_uint16_no_nodata_keeps_dtype(tmp_path):
    """Without a nodata sentinel, the dtype stays integer."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.arange(2), 'x': np.arange(3)},
        attrs={'crs': 4326},
    )
    tif = str(tmp_path / 'src_no_nodata_1564.tif')
    to_geotiff(da, tif, compression='none')

    vrt_path = str(tmp_path / 'src_no_nodata_1564.vrt')
    write_vrt(vrt_path, [tif])
    via_vrt = read_vrt(vrt_path)

    assert via_vrt.dtype == np.uint16
    np.testing.assert_array_equal(via_vrt.values, arr)


def test_vrt_float_nodata_still_masks(tmp_path):
    """Regression guard: the existing float-with-nodata branch still
    works after the integer-branch addition."""
    arr = np.array([[1.0, 2.0, -9999.0], [4.0, -9999.0, 6.0]],
                   dtype=np.float32)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.arange(2), 'x': np.arange(3)},
        attrs={'crs': 4326, 'nodata': -9999.0},
    )
    tif = str(tmp_path / 'srcf_1564.tif')
    to_geotiff(da, tif, compression='none', nodata=-9999.0)

    vrt_path = str(tmp_path / 'srcf_1564.vrt')
    write_vrt(vrt_path, [tif])
    via_vrt = read_vrt(vrt_path)

    assert via_vrt.dtype == np.float32
    assert np.isnan(via_vrt.values[0, 2])
    assert np.isnan(via_vrt.values[1, 1])


def _rewrite_vrt_nodata(vrt_path, new_nodata_text):
    """Rewrite the <NoDataValue> element of an existing VRT to a literal
    string so we can exercise fractional / out-of-range cases without
    going through ``write_vrt`` (which only accepts numeric values)."""
    with open(vrt_path, 'r') as f:
        xml = f.read()
    import re
    new_xml, n = re.subn(
        r'<NoDataValue>[^<]*</NoDataValue>',
        f'<NoDataValue>{new_nodata_text}</NoDataValue>',
        xml,
    )
    assert n == 1, f'expected 1 NoDataValue element, found {n}'
    with open(vrt_path, 'w') as f:
        f.write(new_xml)


def test_vrt_fractional_nodata_is_not_masked(tmp_path):
    """Fractional VRT NoDataValue against an integer band must NOT mask:
    truncating to int would alias a real pixel value as nodata."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(2), 'x': np.arange(3)},
        attrs={'crs': 4326, 'nodata': 1},
    )
    tif = str(tmp_path / 'frac_1564.tif')
    to_geotiff(da, tif, compression='none', nodata=1)

    vrt_path = str(tmp_path / 'frac_1564.vrt')
    write_vrt(vrt_path, [tif])
    _rewrite_vrt_nodata(vrt_path, '1.9')

    via_vrt = read_vrt(vrt_path)
    assert via_vrt.dtype == np.uint16, (
        'Fractional NoDataValue must not trigger integer masking '
        f'(got dtype {via_vrt.dtype}, pixel @[0,0]={via_vrt.values[0, 0]})'
    )
    np.testing.assert_array_equal(via_vrt.values, arr)


def test_vrt_out_of_range_nodata_is_not_masked(tmp_path):
    """NoDataValue outside the dtype range must NOT mask: casting would
    wrap and alias an in-range pixel."""
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(2), 'x': np.arange(3)},
        attrs={'crs': 4326, 'nodata': 0},
    )
    tif = str(tmp_path / 'oor_1564.tif')
    to_geotiff(da, tif, compression='none', nodata=0)

    vrt_path = str(tmp_path / 'oor_1564.vrt')
    write_vrt(vrt_path, [tif])
    # uint16 wraps -1 to 65535, which is in-range but a different sentinel.
    # Reject it rather than silently masking the wrong value.
    _rewrite_vrt_nodata(vrt_path, '-1')

    via_vrt = read_vrt(vrt_path)
    assert via_vrt.dtype == np.uint16, (
        'Out-of-range NoDataValue must not trigger integer masking '
        f'(got dtype {via_vrt.dtype})'
    )
    np.testing.assert_array_equal(via_vrt.values, arr)


def test_vrt_open_geotiff_parity_uint16_nodata(tmp_path):
    """open_geotiff routing a .vrt path should produce the same dtype
    and masked positions as a direct GeoTIFF read."""
    tif = str(tmp_path / 'parity_1564.tif')
    _write_uint16_with_nodata_tif(tif, sentinel=65535)

    direct = open_geotiff(tif)

    vrt_path = str(tmp_path / 'parity_1564.vrt')
    write_vrt(vrt_path, [tif])
    via_vrt = open_geotiff(vrt_path)

    assert direct.dtype == via_vrt.dtype
    np.testing.assert_array_equal(
        np.isnan(direct.values), np.isnan(via_vrt.values),
        err_msg='VRT route should NaN-mask the same pixels as direct read',
    )
    # Non-sentinel pixels equal
    mask = ~np.isnan(direct.values)
    np.testing.assert_array_equal(
        direct.values[mask], via_vrt.values[mask],
    )
