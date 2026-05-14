"""Regression tests for VRT writer 64-bit integer dtype handling.

``write_vrt`` (and ``to_geotiff(da, "*.vrt")`` by extension) previously
mapped signed 64-bit source rasters to ``Int32`` and unsigned 64-bit
source rasters to ``Byte`` because the dtype lookup had no entry for
``bps=64`` and fell back to the small-int default. The VRT reader has
explicit ``UInt64`` / ``Int64`` support (see issue #1783), so the loss
happened on write -- silently truncating uint64 values to ``[0, 255]``.

See issue #1833.
"""
from __future__ import annotations

import re

import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _da(arr: np.ndarray) -> xr.DataArray:
    h, w = arr.shape
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(h, dtype=np.float64),
                'x': np.arange(w, dtype=np.float64)},
        attrs={'res': (1.0, 1.0)},
    )


def _read_vrt_dtype_attr(vrt_path: str) -> str:
    """Extract the ``dataType`` attribute from the emitted VRT XML."""
    with open(vrt_path) as f:
        xml = f.read()
    m = re.search(r'dataType="([^"]+)"', xml)
    assert m is not None, f"no dataType attribute in VRT:\n{xml}"
    return m.group(1)


def test_uint64_vrt_writer_declares_uint64(tmp_path):
    big = np.iinfo(np.uint64).max
    arr = np.array([[1, 2], [big - 7, big]], dtype=np.uint64)
    vrt = tmp_path / 'u64_1833.vrt'
    to_geotiff(_da(arr), str(vrt))
    assert _read_vrt_dtype_attr(str(vrt)) == 'UInt64'


def test_int64_vrt_writer_declares_int64(tmp_path):
    info = np.iinfo(np.int64)
    arr = np.array([[info.min, -1], [0, info.max]], dtype=np.int64)
    vrt = tmp_path / 'i64_1833.vrt'
    to_geotiff(_da(arr), str(vrt))
    assert _read_vrt_dtype_attr(str(vrt)) == 'Int64'


def test_uint64_vrt_round_trip(tmp_path):
    big = np.iinfo(np.uint64).max
    arr = np.array([[1, 2], [big - 7, big]], dtype=np.uint64)
    vrt = tmp_path / 'u64_rt_1833.vrt'
    to_geotiff(_da(arr), str(vrt))
    r = open_geotiff(str(vrt))
    assert r.dtype == np.uint64
    np.testing.assert_array_equal(np.asarray(r.values), arr)


def test_int64_vrt_round_trip(tmp_path):
    info = np.iinfo(np.int64)
    arr = np.array([[info.min, -1], [0, info.max]], dtype=np.int64)
    vrt = tmp_path / 'i64_rt_1833.vrt'
    to_geotiff(_da(arr), str(vrt))
    r = open_geotiff(str(vrt))
    assert r.dtype == np.int64
    np.testing.assert_array_equal(np.asarray(r.values), arr)
