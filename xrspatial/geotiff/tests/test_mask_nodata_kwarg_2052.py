"""Regression tests for issue #2052.

``open_geotiff(path, dtype="uint16")`` on a uint16 file whose nodata
sentinel matches actual pixels used to raise ``ValueError`` from the
``_validate_dtype_cast`` float64-to-uint16 guard: the masking block
ran first and promoted the array to float64, then the dtype= cast
rejected the float-to-int conversion. The docstring at
``xrspatial/geotiff/__init__.py`` promised "Pass ``dtype=...`` to keep
the source dtype", but for integer rasters with a matching sentinel
that contract was unreachable.

The fix adds ``mask_nodata: bool = True`` to the public reader entry
points. Passing ``mask_nodata=False`` skips the sentinel-to-NaN step so
the source dtype survives; ``attrs['nodata']`` still carries the raw
sentinel either way.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._writer import write


@pytest.fixture
def uint16_with_matching_sentinel(tmp_path):
    """uint16 TIFF where nodata=0 and the array has zeros in it."""
    arr = np.array([[0, 100, 200, 300],
                    [400, 500, 0, 600],
                    [700, 800, 900, 0],
                    [0, 1100, 1200, 1300]], dtype=np.uint16)
    path = str(tmp_path / 'uint16_match_2052.tif')
    write(arr, path, nodata=0, compression='none', tiled=False)
    return path, arr


@pytest.fixture
def uint16_no_match(tmp_path):
    """uint16 TIFF whose nodata sentinel is not present in any pixel."""
    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8]], dtype=np.uint16)
    path = str(tmp_path / 'uint16_nomatch_2052.tif')
    write(arr, path, nodata=65535, compression='none', tiled=False)
    return path, arr


@pytest.fixture
def float32_tiff(tmp_path):
    """float32 TIFF with NaN nodata."""
    arr = np.array([[1.0, 2.0, np.nan],
                    [4.0, np.nan, 6.0]], dtype=np.float32)
    path = str(tmp_path / 'float32_2052.tif')
    write(arr, path, nodata=float('nan'), compression='none', tiled=False)
    return path, arr


def test_regression_dtype_uint16_was_unreachable(
        uint16_with_matching_sentinel):
    """Without the kwarg, ``dtype="uint16"`` raises on a matching sentinel.

    Baseline that documents the broken contract: this is the original
    failure mode reported in the issue. The ``mask_nodata=False``
    branch below is the fix.
    """
    path, _ = uint16_with_matching_sentinel
    with pytest.raises(ValueError):
        open_geotiff(path, dtype='uint16')


def test_mask_nodata_false_preserves_uint16(uint16_with_matching_sentinel):
    """``mask_nodata=False`` keeps the uint16 source dtype."""
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, dtype='uint16', mask_nodata=False)
    assert da.dtype == np.uint16
    # Raw sentinels survive in the data.
    np.testing.assert_array_equal(da.values, arr)
    # The declared sentinel is still surfaced for downstream maskers.
    assert da.attrs['nodata'] == 0


def test_mask_nodata_false_no_dtype_kwarg(uint16_with_matching_sentinel):
    """Without ``dtype=``, the source dtype is preserved as-is."""
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, mask_nodata=False)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, arr)
    assert da.attrs['nodata'] == 0


def test_default_mask_nodata_true_still_promotes(
        uint16_with_matching_sentinel):
    """Default ``mask_nodata=True`` keeps the existing behaviour."""
    path, _ = uint16_with_matching_sentinel
    da = open_geotiff(path)
    assert da.dtype == np.float64
    assert np.isnan(da.values).sum() == 4
    # Sentinel positions should be NaN.
    assert np.isnan(da.values[0, 0])
    assert np.isnan(da.values[1, 2])


def test_no_match_both_modes_agree(uint16_no_match):
    """When the sentinel does not match any pixel, both modes return the
    same uint16 array (no promotion needed in either case).
    """
    path, arr = uint16_no_match
    masked = open_geotiff(path)
    unmasked = open_geotiff(path, mask_nodata=False)
    assert masked.dtype == np.uint16
    assert unmasked.dtype == np.uint16
    np.testing.assert_array_equal(masked.values, arr)
    np.testing.assert_array_equal(unmasked.values, arr)


def test_float_file_mask_nodata_false_keeps_data(float32_tiff):
    """For a float32 file with NaN nodata, ``mask_nodata=False`` is a
    no-op: the sentinel is NaN so the inline mask would do nothing
    anyway, and the float dtype is preserved either way.
    """
    path, arr = float32_tiff
    masked = open_geotiff(path)
    unmasked = open_geotiff(path, mask_nodata=False)
    assert masked.dtype == np.float32
    assert unmasked.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(masked.values),
                                  np.isnan(arr))
    np.testing.assert_array_equal(np.isnan(unmasked.values),
                                  np.isnan(arr))


def test_dtype_cast_preservation_uint8(tmp_path):
    """Casting to a different integer dtype also works with the opt-out.

    The reader keeps the source dtype (uint16) via ``mask_nodata=False``,
    then ``dtype="uint32"`` casts integer-to-integer, which is allowed.
    """
    arr = np.array([[0, 100, 200],
                    [300, 0, 500]], dtype=np.uint16)
    path = str(tmp_path / 'uint16_to_uint32_2052.tif')
    write(arr, path, nodata=0, compression='none', tiled=False)

    da = open_geotiff(path, dtype='uint32', mask_nodata=False)
    assert da.dtype == np.uint32
    np.testing.assert_array_equal(da.values, arr.astype(np.uint32))


def test_dask_path_mask_nodata_false(uint16_with_matching_sentinel):
    """The dask path honours the kwarg too: integer source dtype survives.

    Without this, ``read_geotiff_dask`` would still promote the dask
    graph dtype to float64 and force the per-chunk cast.
    """
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, chunks=2, mask_nodata=False)
    assert da.dtype == np.uint16
    computed = da.compute()
    assert computed.dtype == np.uint16
    np.testing.assert_array_equal(computed.values, arr)
    assert computed.attrs['nodata'] == 0


def test_dask_path_default_still_promotes(uint16_with_matching_sentinel):
    """The dask default (``mask_nodata=True``) still promotes to float64."""
    path, _ = uint16_with_matching_sentinel
    da = open_geotiff(path, chunks=2)
    assert da.dtype == np.float64
    computed = da.compute()
    assert np.isnan(computed.values).sum() == 4


def test_dask_dtype_cast_with_opt_out(uint16_with_matching_sentinel):
    """``dtype="uint16"`` + ``mask_nodata=False`` works on the dask path."""
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, chunks=2, dtype='uint16', mask_nodata=False)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, arr)
