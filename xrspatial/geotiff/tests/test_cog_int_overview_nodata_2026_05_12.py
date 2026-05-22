"""COG overview generation respects the nodata sentinel for integer rasters.

Companion to issue #1613 (float COG overview poisoning). Before this fix,
``to_geotiff(int_data, cog=True, nodata=N)`` ran the overview reduction with
the sentinel still present in the integer-cast float64 block. The nan-aware
reduction (``np.nanmean`` / nanmin / nanmax / nanmedian) averaged the
sentinel into surrounding valid pixels and produced overview values that
the reader could not mask -- they did not equal the sentinel, so the
int-to-NaN mask in ``open_geotiff`` left them as silent garbage.

These tests pin the contract that the CPU writer (and the GPU mirror in
``_block_reduce_2d_gpu``) skip the integer sentinel during overview
reduction, so the resulting pyramid only contains real measurements and
the sentinel value.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


# ---------------------------------------------------------------------------
# Unit-level: _block_reduce_2d on integer dtypes
# ---------------------------------------------------------------------------

def _int_block_partial_sentinel(sentinel, dtype):
    """4x4 integer raster where the right two columns of each row pair
    mix valid and sentinel cells. Block (0, 1) has (100, 100, sentinel,
    sentinel); block (1, 1) has (200, 200, sentinel, sentinel)."""
    arr = np.array([
        [100, 100, 100, 100],
        [100, 100, sentinel, sentinel],
        [200, 200, 200, 200],
        [200, 200, sentinel, sentinel],
    ], dtype=dtype)
    return arr


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
@pytest.mark.parametrize('dtype,sentinel', [
    (np.uint8, 255),
    (np.uint16, 65535),
    (np.int16, -9999),
    (np.int32, -2_000_000_000),
])
def test_block_reduce_int_sentinel_masked(method, dtype, sentinel):
    """Integer overview reductions must skip sentinel cells.

    Before the fix, mean produced averages like ``(100+sentinel)/2`` cast
    back to the integer dtype -- a non-sentinel value that the reader
    leaves untouched. The fix masks the sentinel to NaN before the
    reduction so nan-aware aggregation skips it.
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = _int_block_partial_sentinel(sentinel, dtype)
    out = _block_reduce_2d(arr, method, nodata=sentinel)

    # Every block now has at least one valid 100/200; result should equal
    # the valid value (since for mean/min/max/median over {100, 100} is
    # 100, and over {200, 200} is 200). Neither block has any cell that
    # isn't 100, 200, or sentinel, so the output must be a subset of
    # {100, 200}.
    assert out.dtype == arr.dtype
    out_vals = set(out.flatten().tolist())
    assert out_vals.issubset({100, 200}), (
        f"method={method} dtype={dtype} sentinel={sentinel} "
        f"produced poisoned values: {out_vals - {100, 200}}"
    )


@pytest.mark.parametrize('dtype,sentinel', [
    (np.uint16, 65535),
    (np.int16, -9999),
])
def test_block_reduce_int_all_sentinel_block(dtype, sentinel):
    """A 2x2 block that's entirely sentinel reduces to the sentinel.

    Without the post-reduction NaN-to-sentinel rewrite in the integer
    branch, the all-NaN block from nanmean would cast to undefined
    integer behaviour (zero or INT_MIN depending on platform).
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.array([
        [100,      100,      sentinel, sentinel],
        [100,      100,      sentinel, sentinel],
        [200,      200,      200,      200],
        [200,      200,      200,      200],
    ], dtype=dtype)

    out = _block_reduce_2d(arr, 'mean', nodata=sentinel)
    assert out.dtype == arr.dtype
    # Top-right block is all-sentinel; output must be the sentinel
    assert out[0, 1] == sentinel
    # Other blocks contain only valid values
    assert out[0, 0] == 100
    assert out[1, 0] == 200
    assert out[1, 1] == 200


def test_block_reduce_int_no_nodata_unchanged():
    """Without ``nodata``, the integer reduction code path stays unchanged.

    Regression check: the fix must not alter the no-sentinel case.
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=np.int16)

    out = _block_reduce_2d(arr, 'mean')
    # Block (0,0) = mean(1,2,5,6) = 3.5 -> round -> 4
    # Block (0,1) = mean(3,4,7,8) = 5.5 -> round -> 6
    # Block (1,0) = mean(9,10,13,14) = 11.5 -> round -> 12
    # Block (1,1) = mean(11,12,15,16) = 13.5 -> round -> 14
    expected = np.array([[4, 6], [12, 14]], dtype=np.int16)
    np.testing.assert_array_equal(out, expected)


def test_block_reduce_int_out_of_range_sentinel_noop():
    """A sentinel outside the dtype's range is a no-op (no mask applied).

    Mirrors the ``_int_nodata_in_range`` gating in ``_reader.py``: a
    uint16 file with GDAL_NODATA="-9999" cannot match any decoded pixel,
    so the reduction proceeds without the mask. This keeps the fix from
    raising OverflowError on the dtype cast.
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    # uint16 with nodata=-9999: out of range, no-op
    arr = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ], dtype=np.uint16)
    out = _block_reduce_2d(arr, 'mean', nodata=-9999)
    # Should produce the same result as without the kwarg
    expected = _block_reduce_2d(arr, 'mean')
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# End-to-end: to_geotiff + open_geotiff round trip
# ---------------------------------------------------------------------------

@pytest.fixture
def _int_cog_inputs(tmp_path):
    """uint16 raster, full of 100 with a 65x65 sentinel patch."""
    H, W = 256, 256
    data = np.full((H, W), 100, dtype=np.uint16)
    data[64:129, 64:129] = 65535
    da = xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': np.arange(H, dtype=np.float64),
                'x': np.arange(W, dtype=np.float64)},
        attrs={'crs': 4326},
    )
    return da, tmp_path


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
def test_cpu_int_cog_overview_not_poisoned(_int_cog_inputs, method):
    """End-to-end: integer COG overview pyramid contains only valid values.

    Before the fix, the level-1 read contained values like 16459 and
    32818 -- nan-aware-mean of (sentinel, 100, 100, 100) and (sentinel,
    sentinel, 100, 100) cast back to uint16. The reader can't mask them
    because they don't equal 65535.
    """
    from xrspatial.geotiff import open_geotiff, to_geotiff

    da, tmp_path = _int_cog_inputs
    path = str(tmp_path / f'int_overview_{method}_2026_05_12.tif')
    to_geotiff(da, path, nodata=65535, cog=True,
               overview_levels=[2], overview_resampling=method)

    ov = open_geotiff(path, overview_level=1)
    arr = np.asarray(ov.data)
    unique = set(int(v) for v in np.unique(arr) if not np.isnan(v))
    poisoned = unique - {100, 65535}
    assert not poisoned, (
        f"method={method} produced poisoned overview values: {poisoned}"
    )


def test_cpu_int_cog_overview_3band_not_poisoned(tmp_path):
    """3-band integer COG: same fix applies via the 3D _make_overview branch."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    H, W = 256, 256
    data = np.full((H, W, 3), 100, dtype=np.uint16)
    data[64:129, 64:129, :] = 65535
    da = xr.DataArray(
        data,
        dims=('y', 'x', 'band'),
        coords={'y': np.arange(H, dtype=np.float64),
                'x': np.arange(W, dtype=np.float64),
                'band': [0, 1, 2]},
        attrs={'crs': 4326},
    )

    path = str(tmp_path / 'int_overview_3band_2026_05_12.tif')
    to_geotiff(da, path, nodata=65535, cog=True,
               overview_levels=[2], overview_resampling='mean')

    ov = open_geotiff(path, overview_level=1)
    arr = np.asarray(ov.data)
    unique = set(int(v) for v in np.unique(arr) if not np.isnan(v))
    poisoned = unique - {100, 65535}
    assert not poisoned, (
        f"3-band integer overview produced poisoned values: {poisoned}"
    )


def test_cpu_int_cog_no_nodata_unchanged(tmp_path):
    """No nodata kwarg: integer overview path stays as it was."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    H, W = 256, 256
    data = np.full((H, W), 100, dtype=np.uint16)
    data[100:200, 100:200] = 50
    da = xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': np.arange(H, dtype=np.float64),
                'x': np.arange(W, dtype=np.float64)},
        attrs={'crs': 4326},
    )

    path = str(tmp_path / 'int_overview_no_nodata_2026_05_12.tif')
    to_geotiff(da, path, cog=True,
               overview_levels=[2], overview_resampling='mean')

    ov = open_geotiff(path, overview_level=1)
    arr = np.asarray(ov.data)
    # No sentinel, so every overview pixel is a real average of 50 / 100.
    # Block-boundary pixels are weighted means: (50,50,50,100)/4 = 62.5 -> 63
    unique = set(int(v) for v in np.unique(arr))
    # Must contain at least 50 and 100; boundary-mixing averages allowed.
    assert 50 in unique
    assert 100 in unique


# ---------------------------------------------------------------------------
# GPU mirror
# ---------------------------------------------------------------------------

@_gpu_only
@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
@pytest.mark.parametrize('dtype,sentinel', [
    (np.uint16, 65535),
    (np.int16, -9999),
])
def test_gpu_block_reduce_int_sentinel_masked(method, dtype, sentinel):
    """GPU mirror of the CPU integer sentinel-mask fix."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr = _int_block_partial_sentinel(sentinel, dtype)
    cpu_arr = cupy.asarray(arr)
    out_gpu = _block_reduce_2d_gpu(cpu_arr, method, nodata=sentinel)
    out = out_gpu.get()

    assert out.dtype == arr.dtype
    out_vals = set(out.flatten().tolist())
    assert out_vals.issubset({100, 200}), (
        f"GPU method={method} dtype={dtype} produced poisoned values: "
        f"{out_vals - {100, 200}}"
    )


@_gpu_only
@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
def test_gpu_cpu_int_overview_byte_match(method):
    """CPU and GPU integer overview reductions agree byte-for-byte.

    Same parity contract as #1623 (cubic). Without the GPU fix, the GPU
    pyramid would carry poisoned values while the CPU pyramid carried
    sentinels -- two backends disagreeing on identical input.
    """
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = _int_block_partial_sentinel(-9999, np.int16)
    cpu_out = _block_reduce_2d(arr, method, nodata=-9999)
    gpu_out = _block_reduce_2d_gpu(
        cupy.asarray(arr), method, nodata=-9999).get()

    np.testing.assert_array_equal(cpu_out, gpu_out)
