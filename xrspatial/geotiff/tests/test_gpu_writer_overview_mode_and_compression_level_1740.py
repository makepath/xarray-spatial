"""Coverage for ``write_geotiff_gpu`` ``overview_resampling='mode'`` and
``compression_level=`` accepted-but-ignored contract.

Test coverage gap sweep 2026-05-12. Closes issue #1740.

``write_geotiff_gpu`` and ``to_geotiff(gpu=True)`` accept seven
``overview_resampling`` modes (``mean``, ``nearest``, ``min``, ``max``,
``median``, ``mode``, ``cubic``). Six have direct end-to-end coverage in
the existing suite; ``mode`` is the odd one out -- the dedicated branch
in ``_block_reduce_2d_gpu`` (``xrspatial/geotiff/_gpu_decode.py:3051-3056``)
and the matching ``write_geotiff_gpu(cog=True, overview_resampling='mode')``
/ ``to_geotiff(gpu=True, ..., overview_resampling='mode')`` paths had no
targeted tests. ``test_mode_overview_perf.py`` exercises the CPU
``_block_reduce_2d`` helper, not the GPU one.

If someone dropped the ``mode`` dispatch from ``_block_reduce_2d_gpu``,
the function would fall through to the ``mean`` reshape branch and emit
wrong overview pixels for integer rasters; the existing suite would not
catch the regression.

``write_geotiff_gpu(compression_level=...)`` is documented as "Accepted
for API compatibility but currently ignored -- nvCOMP does not expose
level control". ``to_geotiff`` threads the kwarg through unchanged. The
CPU writer rejects out-of-range ``compression_level`` values with
``ValueError``; the GPU writer should not, per the docstring. No test
pinned that asymmetry, so wiring the GPU writer up to the CPU range
validator would silently break every ``to_geotiff(gpu=True,
compression_level=X)`` caller for in-range levels and noisily for
out-of-range ones.

The tests are *additive*: no source changes are made.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    to_geotiff,
    write_geotiff_gpu,
)
from xrspatial.geotiff._writer import _block_reduce_2d


# ---------------------------------------------------------------------------
# GPU gating
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    """True when cupy is importable and CUDA is initialised."""
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
# Fixtures
# ---------------------------------------------------------------------------


def _mode_4x4_uint8() -> np.ndarray:
    """4x4 uint8 raster with a deterministic mode per 2x2 block.

    Block layout:

        [ 1  1 | 5  5 ]
        [ 1  9 | 5  5 ]
        --------------
        [ 2  2 | 3  3 ]
        [ 2  7 | 3  8 ]

    Mode of each 2x2 block:
        [[ 1, 5 ],
         [ 2, 3 ]]
    """
    return np.array(
        [
            [1, 1, 5, 5],
            [1, 9, 5, 5],
            [2, 2, 3, 3],
            [2, 7, 3, 8],
        ],
        dtype=np.uint8,
    )


_MODE_4x4_EXPECTED = np.array([[1, 5], [2, 3]], dtype=np.uint8)


def _mode_8x8_uint8() -> np.ndarray:
    """8x8 uint8 raster -- big enough for two tiles at tile_size=4 with
    a deterministic mode per 2x2 block on the level-1 overview."""
    rng = np.random.default_rng(seed=1740)
    # Use a small categorical range so ties are common; the GPU mode
    # branch falls back to the CPU implementation, so the result must
    # match _block_reduce_2d bit-for-bit.
    return rng.integers(0, 4, size=(8, 8), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Direct _block_reduce_2d_gpu(method='mode') call.
# ---------------------------------------------------------------------------


@_gpu_only
def test_block_reduce_2d_gpu_mode_matches_cpu_4x4():
    """``_block_reduce_2d_gpu(method='mode')`` falls back to the CPU
    helper and must produce the same output as the CPU reducer on the
    same input."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr = _mode_4x4_uint8()
    arr_gpu = cupy.asarray(arr)
    out_gpu = _block_reduce_2d_gpu(arr_gpu, 'mode')
    np.testing.assert_array_equal(cupy.asnumpy(out_gpu), _MODE_4x4_EXPECTED)


@_gpu_only
def test_block_reduce_2d_gpu_mode_matches_cpu_random_8x8():
    """Bit-for-bit parity with the CPU helper across a random 8x8 input
    with many ties (small categorical range)."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr = _mode_8x8_uint8()
    cpu_out = _block_reduce_2d(arr, 'mode')
    gpu_out = _block_reduce_2d_gpu(cupy.asarray(arr), 'mode')
    np.testing.assert_array_equal(cupy.asnumpy(gpu_out), cpu_out)
    # Output dtype is preserved.
    assert gpu_out.dtype == cpu_out.dtype == arr.dtype


@_gpu_only
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32])
def test_block_reduce_2d_gpu_mode_dtype_preserved(dtype):
    """Mode is dtype-preserving; the CPU fallback must not promote to
    float on the GPU branch."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    info = np.iinfo(dtype)
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(max(info.min, 0), min(info.max, 7) + 1,
                       size=(8, 8), dtype=dtype)
    out = _block_reduce_2d_gpu(cupy.asarray(arr), 'mode')
    assert out.dtype == arr.dtype


# ---------------------------------------------------------------------------
# End-to-end: write_geotiff_gpu(cog=True, overview_resampling='mode').
# ---------------------------------------------------------------------------


@_gpu_only
def test_write_geotiff_gpu_cog_overview_resampling_mode(tmp_path):
    """``write_geotiff_gpu(cog=True, overview_resampling='mode')`` writes
    a COG whose level-1 overview matches the closed-form 2x2 mode
    reduction.

    Exercises the full GPU make-overview path including the dispatch on
    ``method == 'mode'`` in ``_block_reduce_2d_gpu``.
    """
    import cupy

    arr = _mode_4x4_uint8()
    arr_gpu = cupy.asarray(arr)
    da = xr.DataArray(
        arr_gpu, dims=['y', 'x'],
        coords={'y': np.arange(4.0, 0, -1), 'x': np.arange(4.0)},
    )
    p = str(tmp_path / 'cog_mode_gpu_1740.tif')
    write_geotiff_gpu(
        da, p, cog=True, compression='deflate', tiled=True,
        tile_size=4, overview_levels=[2],
        overview_resampling='mode',
    )

    # Full-res round-trip remains exact.
    full = open_geotiff(p)
    np.testing.assert_array_equal(np.asarray(full.data), arr)

    # Level-1 overview matches the closed-form 2x2 mode reduction.
    ov = open_geotiff(p, overview_level=1)
    assert ov.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(ov.data), _MODE_4x4_EXPECTED)


@_gpu_only
def test_to_geotiff_gpu_cog_overview_resampling_mode(tmp_path):
    """``to_geotiff(gpu=True, cog=True, overview_resampling='mode')``
    threads through to the GPU writer and produces the same overview as
    the explicit ``write_geotiff_gpu`` call."""
    import cupy

    arr = _mode_4x4_uint8()
    da = xr.DataArray(
        cupy.asarray(arr), dims=['y', 'x'],
        coords={'y': np.arange(4.0, 0, -1), 'x': np.arange(4.0)},
    )
    p = str(tmp_path / 'cog_mode_to_geotiff_gpu_1740.tif')
    to_geotiff(
        da, p, gpu=True, cog=True, compression='deflate', tiled=True,
        tile_size=4, overview_levels=[2],
        overview_resampling='mode',
    )

    ov = open_geotiff(p, overview_level=1)
    assert ov.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(ov.data), _MODE_4x4_EXPECTED)


@_gpu_only
def test_gpu_vs_cpu_mode_overview_pixel_parity(tmp_path):
    """The GPU writer's ``mode`` overview bytes match the CPU writer's
    output pixel-for-pixel.

    The GPU branch routes through the CPU ``_block_reduce_2d`` helper,
    so any drift between the two would either mean the GPU dispatch is
    swapping methods or the routing has changed. Both are regressions
    the read-back tests above might miss on small inputs.
    """
    import cupy

    arr = _mode_8x8_uint8()

    da_cpu = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
    )
    p_cpu = str(tmp_path / 'cog_mode_cpu_1740.tif')
    to_geotiff(
        da_cpu, p_cpu, cog=True, compression='deflate', tiled=True,
        tile_size=4, overview_levels=[2],
        overview_resampling='mode',
    )

    da_gpu = xr.DataArray(
        cupy.asarray(arr), dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
    )
    p_gpu = str(tmp_path / 'cog_mode_gpu_via_to_geotiff_1740.tif')
    to_geotiff(
        da_gpu, p_gpu, gpu=True, cog=True, compression='deflate', tiled=True,
        tile_size=4, overview_levels=[2],
        overview_resampling='mode',
    )

    ov_cpu = np.asarray(open_geotiff(p_cpu, overview_level=1).data)
    ov_gpu = np.asarray(open_geotiff(p_gpu, overview_level=1).data)
    np.testing.assert_array_equal(ov_gpu, ov_cpu)


# ---------------------------------------------------------------------------
# compression_level=... accepted-but-ignored contract on the GPU writer.
#
# The CPU writer validates compression_level against the codec's range
# (deflate 1-9, zstd 1-22, lz4 0-16). The GPU writer docstring says the
# kwarg is "Accepted for API compatibility but currently ignored". A
# regression that started forwarding the level to a range validator
# would change behaviour against the contract.
# ---------------------------------------------------------------------------


def _gpu_compression_level_da(arr: np.ndarray):
    """Wrap *arr* in a CuPy-backed DataArray with monotonic y/x coords."""
    import cupy
    return xr.DataArray(
        cupy.asarray(arr), dims=['y', 'x'],
        coords={
            'y': np.arange(arr.shape[0], dtype=np.float64)[::-1],
            'x': np.arange(arr.shape[1], dtype=np.float64),
        },
    )


@_gpu_only
@pytest.mark.parametrize("compression, in_range_level", [
    ('zstd', 1),
    ('zstd', 22),
    ('deflate', 1),
    ('deflate', 9),
])
def test_write_geotiff_gpu_compression_level_in_range_accepted(
        tmp_path, compression, in_range_level):
    """In-range ``compression_level`` values are accepted and the file
    still round-trips.

    The kwarg is documented as ignored, but ``to_geotiff(gpu=True, ...,
    compression_level=N)`` callers expect the level to flow through
    without error when N is a value the CPU writer would also accept.
    """
    arr = np.random.RandomState(1740).rand(32, 32).astype(np.float32)
    da = _gpu_compression_level_da(arr)
    p = str(tmp_path / f'level_in_{compression}_{in_range_level}_1740.tif')

    # Must not raise.
    write_geotiff_gpu(
        da, p, compression=compression,
        compression_level=in_range_level,
        tile_size=32,
    )

    out = open_geotiff(p)
    np.testing.assert_allclose(np.asarray(out.data), arr)


@_gpu_only
@pytest.mark.parametrize("compression, oor_level", [
    # CPU writer would reject these; the GPU writer must not.
    ('zstd', 999),
    ('zstd', -5),
    ('deflate', 50),
    ('deflate', 0),
])
def test_write_geotiff_gpu_compression_level_out_of_range_accepted(
        tmp_path, compression, oor_level):
    """Out-of-range ``compression_level`` values do not raise on the GPU
    writer (contrast with the CPU writer, which raises ``ValueError``).

    Pins the docstring contract: "Accepted for API compatibility but
    currently ignored -- nvCOMP does not expose level control". Without
    this test, a regression that wired the GPU writer up to the CPU
    range validator would change behaviour against the doc and break
    every ``to_geotiff(gpu=True, compression_level=X)`` caller for the
    in-range case and noisily for the out-of-range one.
    """
    arr = np.random.RandomState(1740).rand(32, 32).astype(np.float32)
    da = _gpu_compression_level_da(arr)
    p = str(tmp_path / f'level_oor_{compression}_{oor_level}_1740.tif')

    # Must not raise -- the GPU writer ignores the level.
    write_geotiff_gpu(
        da, p, compression=compression,
        compression_level=oor_level,
        tile_size=32,
    )

    # And the file must still round-trip; the ignored level cannot
    # corrupt the output.
    out = open_geotiff(p)
    np.testing.assert_allclose(np.asarray(out.data), arr)


@_gpu_only
def test_to_geotiff_gpu_compression_level_out_of_range_accepted(tmp_path):
    """``to_geotiff(gpu=True, compression_level=X)`` threads the kwarg
    through to ``write_geotiff_gpu`` and must not raise on an out-of-range
    value (the GPU writer ignores it).

    Without this test the dispatcher could be rewritten to validate
    levels before forwarding and silently break the documented contract.
    """
    arr = np.random.RandomState(1740).rand(32, 32).astype(np.float32)
    da = _gpu_compression_level_da(arr)
    p = str(tmp_path / 'level_oor_to_geotiff_gpu_1740.tif')

    # Must not raise even though 999 is out of zstd's 1-22 range.
    to_geotiff(
        da, p, gpu=True, compression='zstd',
        compression_level=999, tiled=True, tile_size=32,
    )

    out = open_geotiff(p)
    np.testing.assert_allclose(np.asarray(out.data), arr)


@_gpu_only
def test_to_geotiff_cpu_compression_level_out_of_range_raises(tmp_path):
    """Companion check: the CPU writer rejects out-of-range
    ``compression_level``. Pins the asymmetry the GPU tests above rely
    on -- if the CPU writer ever stopped raising, the GPU "ignored"
    contract becomes uninteresting and the GPU tests above would silently
    pass for the wrong reason.
    """
    arr = np.random.RandomState(1740).rand(32, 32).astype(np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={
            'y': np.arange(32.0, 0, -1),
            'x': np.arange(32.0),
        },
    )
    p = str(tmp_path / 'level_oor_to_geotiff_cpu_1740.tif')
    with pytest.raises(ValueError, match=r"compression_level=999"):
        to_geotiff(da, p, compression='zstd', compression_level=999,
                   tiled=True, tile_size=32)
