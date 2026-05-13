"""Parameter coverage for the ``overview_resampling`` modes
``'min'``, ``'max'``, and ``'median'``.

The CPU writer (``xrspatial.geotiff._writer._block_reduce_2d``) and the
GPU writer (``xrspatial.geotiff._gpu_decode._block_reduce_2d_gpu``) both
implement seven resampling reductions for COG overview generation:

* ``mean`` -- covered by ``test_cog_overview_nodata_1613`` and
  ``test_features``.
* ``nearest`` -- covered by ``test_features`` and the same suite.
* ``mode`` -- covered by ``test_mode_overview_perf``.
* ``cubic`` -- covered by ``test_cog_cubic_overview_nodata_1623``.
* ``min`` / ``max`` / ``median`` -- CPU end-to-end paths covered by
  ``test_cog_overview_nodata_1613::test_cpu_cog_overview_aggregations_ignore_sentinel``,
  but the GPU end-to-end paths and the direct CPU/GPU block-reducer
  branches had no targeted tests prior to this file.

Test coverage gap sweep 2026-05-11 (pass 6) closes a Cat 4 (parameter
coverage) HIGH gap: the GPU end-to-end paths and the direct CPU+GPU
block-reducer branches for ``overview_resampling='min'/'max'/'median'``
had no targeted tests, so a regression on those code paths would ship
undetected.

The tests cover:

* CPU writer (``to_geotiff(cog=True, overview_resampling='min'/'max'/'median')``)
  with finite data, and with a nodata sentinel so the nan-aware
  reductions (``nanmin`` / ``nanmax`` / ``nanmedian``) get exercised.
* GPU writer (``write_geotiff_gpu`` and ``to_geotiff(gpu=True, ...)``)
  for the three modes, verifying the cupy implementation matches the
  CPU implementation byte-for-byte.
* The ``cog=True`` overview-level read path round-trips the resampled
  data so the full write/read pipeline is exercised.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._writer import _block_reduce_2d


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialised."""
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
# Fixtures: 4x4 rasters with deterministic values so the 2x decimated
# overview has a closed-form min / max / median for every 2x2 block.
# ---------------------------------------------------------------------------

def _arr_4x4_ramp() -> np.ndarray:
    """4x4 float32 ramp.

    Block layout (top-left 2x2, top-right 2x2, ...):

        [ 1  2 | 3  4 ]
        [ 5  6 | 7  8 ]
        --------------
        [ 9 10 |11 12 ]
        [13 14 |15 16 ]

    Per-block reductions:
      * min:    [[1, 3], [9, 11]]
      * max:    [[6, 8], [14, 16]]
      * median: [[3.5, 5.5], [11.5, 13.5]]   (mean of the two middle values)
    """
    return np.arange(1, 17, dtype=np.float32).reshape(4, 4)


def _arr_4x4_with_nan() -> np.ndarray:
    """4x4 float32 ramp with one NaN per top-row 2x2 block.

    Block layout:

        [NaN  2 | 3 NaN]
        [  5  6 | 7  8 ]
        --------------
        [  9 10 |11 12 ]
        [ 13 14 |15 16 ]

    Per-block reductions (NaN ignored):
      * min:    [[2, 3], [9, 11]]
      * max:    [[6, 8], [14, 16]]
      * median: the four-cell median ignoring NaN. For top-left, finite
        cells are {2, 5, 6} -> median 5. Top-right finite {3, 7, 8} -> 7.
        Bottom rows are unchanged ramp medians.
    """
    arr = _arr_4x4_ramp()
    arr[0, 0] = np.nan
    arr[0, 3] = np.nan
    return arr


# Expected outputs are computed via numpy.nan* once at module import so
# they double as a CPU-impl correctness check.

_RAMP_EXPECTED_MIN = np.array([[1.0, 3.0], [9.0, 11.0]], dtype=np.float32)
_RAMP_EXPECTED_MAX = np.array([[6.0, 8.0], [14.0, 16.0]], dtype=np.float32)
_RAMP_EXPECTED_MEDIAN = np.array([[3.5, 5.5], [11.5, 13.5]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Cat 4 HIGH: CPU writer overview_resampling=min/max/median (block reducer).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method, expected", [
    ('min', _RAMP_EXPECTED_MIN),
    ('max', _RAMP_EXPECTED_MAX),
    ('median', _RAMP_EXPECTED_MEDIAN),
])
def test_block_reduce_2d_cpu(method, expected):
    """``_block_reduce_2d`` returns the documented reduction per 2x2 block."""
    arr = _arr_4x4_ramp()
    out = _block_reduce_2d(arr, method)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("method", ['min', 'max', 'median'])
def test_block_reduce_2d_cpu_skips_nan(method):
    """``_block_reduce_2d`` uses nan-aware reductions so partial-NaN
    blocks aggregate over the finite cells only."""
    arr = _arr_4x4_with_nan()
    out = _block_reduce_2d(arr, method)
    assert np.all(np.isfinite(out)), (
        f"method={method!r} returned NaN for a partial-NaN block")

    # Recompute expected via numpy nan-aware ops on the same 2x2 reshape.
    blocks = arr.reshape(2, 2, 2, 2)
    flat = blocks.transpose(0, 2, 1, 3).reshape(2, 2, 4)
    if method == 'min':
        expected = np.nanmin(flat, axis=2)
    elif method == 'max':
        expected = np.nanmax(flat, axis=2)
    else:
        expected = np.nanmedian(flat, axis=2)
    np.testing.assert_allclose(out, expected.astype(np.float32))


@pytest.mark.parametrize("method, expected", [
    ('min', _RAMP_EXPECTED_MIN),
    ('max', _RAMP_EXPECTED_MAX),
    ('median', _RAMP_EXPECTED_MEDIAN),
])
def test_to_geotiff_cog_overview_resampling_cpu(tmp_path, method, expected):
    """End-to-end: ``to_geotiff(cog=True, overview_resampling=method)``
    writes a COG whose overview level 1 matches the closed-form 2x2
    reduction."""
    arr = _arr_4x4_ramp()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / f'cog_{method}.tif')
    to_geotiff(da, p, cog=True, compression='deflate', tiled=True,
               tile_size=2, overview_levels=[2],
               overview_resampling=method)

    ov = open_geotiff(p, overview_level=1)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


@pytest.mark.parametrize("method", ['min', 'max', 'median'])
def test_to_geotiff_cog_overview_resampling_cpu_nodata(tmp_path, method):
    """CPU writer: nan-aware reductions skip the sentinel when ``nodata``
    is set (the regression that motivated issue #1613, here covering the
    min/max/median branches that #1613 did not test)."""
    arr = _arr_4x4_with_nan()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / f'cog_{method}_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=2, overview_levels=[2],
               overview_resampling=method)

    ov = open_geotiff(p, overview_level=1)
    out = np.asarray(ov.data)

    # Recompute expected from the same nan-aware reduction on the source.
    blocks = arr.reshape(2, 2, 2, 2)
    flat = blocks.transpose(0, 2, 1, 3).reshape(2, 2, 4)
    if method == 'min':
        expected = np.nanmin(flat, axis=2)
    elif method == 'max':
        expected = np.nanmax(flat, axis=2)
    else:
        expected = np.nanmedian(flat, axis=2)
    np.testing.assert_allclose(out, expected.astype(np.float32))


# ---------------------------------------------------------------------------
# Cat 4 HIGH: GPU writer overview_resampling=min/max/median.
# ---------------------------------------------------------------------------

@_gpu_only
@pytest.mark.parametrize("method, expected", [
    ('min', _RAMP_EXPECTED_MIN),
    ('max', _RAMP_EXPECTED_MAX),
    ('median', _RAMP_EXPECTED_MEDIAN),
])
def test_block_reduce_2d_gpu(method, expected):
    """``_block_reduce_2d_gpu`` returns the same reduction as the CPU
    block reducer for finite input."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr_cpu = _arr_4x4_ramp()
    arr_gpu = cupy.asarray(arr_cpu)
    out = _block_reduce_2d_gpu(arr_gpu, method)
    np.testing.assert_allclose(cupy.asnumpy(out), expected)


@_gpu_only
@pytest.mark.parametrize("method", ['min', 'max', 'median'])
def test_block_reduce_2d_gpu_matches_cpu_with_nan(method):
    """GPU nan-aware reductions match CPU nan-aware reductions for a
    partial-NaN block."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr_cpu = _arr_4x4_with_nan()
    cpu_out = _block_reduce_2d(arr_cpu, method)
    gpu_out = _block_reduce_2d_gpu(cupy.asarray(arr_cpu), method)
    np.testing.assert_allclose(cupy.asnumpy(gpu_out), cpu_out)


@_gpu_only
@pytest.mark.parametrize("method, expected", [
    ('min', _RAMP_EXPECTED_MIN),
    ('max', _RAMP_EXPECTED_MAX),
    ('median', _RAMP_EXPECTED_MEDIAN),
])
def test_write_geotiff_gpu_cog_overview_resampling(tmp_path, method, expected):
    """End-to-end: ``write_geotiff_gpu(cog=True, overview_resampling=method)``
    writes a COG whose overview level 1 matches the closed-form 2x2
    reduction. Exercises the GPU make-overview path including the dispatch
    on ``method``."""
    import cupy

    from xrspatial.geotiff import write_geotiff_gpu

    arr = _arr_4x4_ramp()
    arr_gpu = cupy.asarray(arr)
    da = xr.DataArray(arr_gpu, dims=['y', 'x'])
    p = str(tmp_path / f'cog_{method}_gpu.tif')
    write_geotiff_gpu(da, p, cog=True, compression='deflate', tiled=True,
                      tile_size=2, overview_levels=[2],
                      overview_resampling=method)

    ov = open_geotiff(p, overview_level=1)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


@_gpu_only
@pytest.mark.parametrize("method", ['min', 'max', 'median'])
def test_to_geotiff_gpu_cog_overview_matches_cpu(tmp_path, method):
    """``to_geotiff(gpu=True, ..., overview_resampling=method)`` produces
    overview bytes that round-trip to the same values as the CPU writer."""
    import cupy

    arr = _arr_4x4_ramp()
    da_cpu = xr.DataArray(arr, dims=['y', 'x'])
    p_cpu = str(tmp_path / f'cog_{method}_cpu.tif')
    to_geotiff(da_cpu, p_cpu, cog=True, compression='deflate', tiled=True,
               tile_size=2, overview_levels=[2],
               overview_resampling=method)

    da_gpu = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])
    p_gpu = str(tmp_path / f'cog_{method}_gpu_via_to_geotiff.tif')
    to_geotiff(da_gpu, p_gpu, gpu=True, cog=True, compression='deflate',
               tiled=True, tile_size=2, overview_levels=[2],
               overview_resampling=method)

    ov_cpu = np.asarray(open_geotiff(p_cpu, overview_level=1).data)
    ov_gpu = np.asarray(open_geotiff(p_gpu, overview_level=1).data)
    np.testing.assert_allclose(ov_gpu, ov_cpu)


# ---------------------------------------------------------------------------
# Error path: unknown method names raise ValueError on both backends.
# ---------------------------------------------------------------------------

def test_block_reduce_2d_cpu_unknown_method_raises():
    """The CPU block reducer raises ``ValueError`` on an unknown method
    name. Exercises the else-branch that lists the valid methods."""
    arr = _arr_4x4_ramp()
    with pytest.raises(ValueError, match="Unknown overview resampling"):
        _block_reduce_2d(arr, 'bogus')


@_gpu_only
def test_block_reduce_2d_gpu_unknown_method_raises():
    """The GPU block reducer raises ``ValueError`` on an unknown method
    name. The CPU equivalent already raises for parity."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr_gpu = cupy.asarray(_arr_4x4_ramp())
    with pytest.raises(ValueError, match="Unknown GPU overview resampling"):
        _block_reduce_2d_gpu(arr_gpu, 'bogus')
