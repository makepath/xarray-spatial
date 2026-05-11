"""Regression tests for issue #1597.

``read_geotiff_dask`` on an integer raster with an in-range nodata
sentinel used to silently lose the mask when the sentinel only appeared
in non-first chunks. Per-chunk dtype divergence (uint16 vs float64)
caused dask concatenation to preallocate from the first chunk's actual
dtype, casting float64 chunks back to int and converting NaN to 0.

The fix threads the resolved ``target_dtype`` (the dask graph's
declared dtype) unconditionally through ``_delayed_read_window`` so
every chunk lands as float64 regardless of whether its mask hit.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._writer import write


@pytest.fixture
def uint16_with_sentinel_only_in_corner(tmp_path):
    """Write a uint16 8x8 TIFF whose nodata sentinel is in the
    bottom-right 2x2 quadrant. With ``chunks=4`` the top-left chunk
    never sees a sentinel and used to keep its uint16 dtype.
    """
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8) + 1
    arr[6:8, 6:8] = 65535
    path = str(tmp_path / 'uint16_corner_sentinel_1597.tif')
    write(arr, path, nodata=65535, compression='none', tiled=False)
    return path, arr


def test_eager_promotes_to_float64_and_masks(uint16_with_sentinel_only_in_corner):
    """Baseline: the eager path produces float64 with 4 NaNs."""
    path, _ = uint16_with_sentinel_only_in_corner
    eager = open_geotiff(path)
    assert eager.dtype == np.float64
    assert np.isnan(eager.values).sum() == 4
    assert np.isnan(eager.values[6:8, 6:8]).all()


def test_dask_chunks_4_matches_eager(uint16_with_sentinel_only_in_corner):
    """The dask compute result matches the eager path bit-for-bit.

    Before the fix this returned a uint16 array with 0s where the
    sentinel had been, because dask coerced the late-arriving float64
    chunk back to uint16 at concat time.
    """
    path, _ = uint16_with_sentinel_only_in_corner
    eager = open_geotiff(path)
    dk = open_geotiff(path, chunks=4)
    assert dk.dtype == np.float64
    computed = dk.compute()
    assert computed.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(computed.values),
                                  np.isnan(eager.values))
    finite = ~np.isnan(eager.values)
    np.testing.assert_array_equal(computed.values[finite],
                                  eager.values[finite])


def test_dask_chunks_2_per_chunk_dtype_uniform(
        uint16_with_sentinel_only_in_corner):
    """Every dask chunk returns float64 regardless of mask hit.

    Iterates the delayed blocks and asserts each one computes to
    float64; the regression had the first chunk's actual data come back
    as uint16 because the mask never matched there.
    """
    path, _ = uint16_with_sentinel_only_in_corner
    dk = open_geotiff(path, chunks=2)
    blocks = dk.data.to_delayed().flatten()
    for i, block in enumerate(blocks):
        chunk = block.compute()
        assert chunk.dtype == np.float64, (
            f"chunk {i} computed as {chunk.dtype}, expected float64; "
            f"per-chunk dtype divergence is the #1597 regression."
        )


def test_dask_keeps_dtype_for_out_of_range_sentinel(tmp_path):
    """Out-of-range sentinels (uint16 + nodata=-9999) stay uint16.

    The fix should not regress #1581: when the sentinel cannot match
    any pixel, no float64 promotion is needed and the dask path keeps
    the file's native dtype.
    """
    arr = np.array([[1, 2, 3, 4]] * 4, dtype=np.uint16)
    path = str(tmp_path / 'uint16_out_of_range_1597.tif')
    write(arr, path, nodata=-9999, compression='none', tiled=False)

    dk = open_geotiff(path, chunks=2)
    assert dk.dtype == np.uint16
    result = dk.compute()
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result.values, arr)


def test_dask_float_input_with_sentinel_in_one_chunk(tmp_path):
    """Float rasters with sentinel in non-first chunk also stay float.

    The float path doesn't promote dtype, but it does in-place NaN
    substitution. Verify the substitution holds for chunks with and
    without the sentinel.
    """
    arr = np.arange(64, dtype=np.float32).reshape(8, 8) + 1
    arr[6:8, 6:8] = -9999.0
    path = str(tmp_path / 'float_corner_sentinel_1597.tif')
    write(arr, path, nodata=-9999, compression='none', tiled=False)

    eager = open_geotiff(path)
    dk = open_geotiff(path, chunks=4).compute()
    np.testing.assert_array_equal(np.isnan(dk.values),
                                  np.isnan(eager.values))
