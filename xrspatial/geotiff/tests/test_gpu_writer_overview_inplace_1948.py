"""Regression tests for issue #1948.

The COG overview generation loop inside ``write_geotiff_gpu`` used to
replace the in-place NaN-to-sentinel rewrite with
``current = current.copy(); current[nan_mask] = sentinel``. Every call
to ``make_overview_gpu`` returns a freshly allocated cupy buffer
(``_block_reduce_2d_gpu`` ends in ``cupy.nan*`` / ``cupy.around(...).astype(...)`` /
``cropped[::2, ::2].copy()``, the 3-D path ends in ``cupy.stack``), so
nothing else aliases the buffer between the return and the rewrite. The
``.copy()`` allocated a second chunk-sized GPU buffer per overview
level for no semantic gain.

The fix swaps the rewrite to ``cupy.putmask(current, nan_mask, sentinel)``
which mutates the existing buffer in place. This drops one chunk-sized
device allocation per overview level and mirrors the in-place sentinel
rewrite ``_apply_nodata_mask_gpu`` adopted in #1934.

Two guards here:

1. Correctness -- a COG write with a sentinel-bearing float raster
   round-trips through the GPU writer and back to the CPU reader with
   the sentinel preserved in every overview level.
2. Structural -- the writer source uses ``cupy.putmask`` and no longer
   carries the redundant ``current = current.copy()`` line in the
   overview loop.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import xarray as xr

from xrspatial.geotiff.tests.conftest import requires_gpu as _gpu_only


def _make_float_raster_with_nodata(height: int, width: int, sentinel: float):
    """Build a float32 raster whose sentinel pixels would poison overviews.

    Reserve a contiguous all-sentinel sub-rectangle so that every 2x
    decimation step still contains at least one fully-sentinel 2x2
    block. ``_block_reduce_2d_gpu`` masks the sentinel back to NaN
    before reducing, so a partially-sentinel block returns a valid mean
    and the overview pixel never becomes NaN; only a fully-sentinel
    block triggers the NaN->sentinel rewrite branch the fix touches.
    """
    import cupy

    arr = np.random.RandomState(1948).rand(height, width).astype(np.float32)
    # Bottom-right quadrant becomes all-sentinel so every 2x reduction
    # of that region stays all-sentinel and the NaN rewrite branch
    # fires at every overview level.
    arr[height // 2:, width // 2:] = sentinel
    return cupy.asarray(arr)


def test_gpu_writer_overview_loop_uses_putmask_1948():
    """Source-level guard against the redundant ``current.copy()``.

    The fix replaces the ``current = current.copy(); current[nan_mask] = ...``
    pair with ``cupy.putmask(current, nan_mask, ...)`` so the buffer
    ``make_overview_gpu`` returned is mutated in place. This guard
    fires if anyone reintroduces the redundant copy.
    """
    import pathlib

    src_path = pathlib.Path(__file__).parent.parent / "_writers" / "gpu.py"
    src = src_path.read_text()
    # The overview loop should call cupy.putmask, not current.copy() + indexed write.
    assert "cupy.putmask(" in src, (
        "write_geotiff_gpu overview loop should use cupy.putmask "
        "for the in-place NaN->sentinel rewrite (issue #1948)."
    )
    # Confirm the in-place pattern is the one inside the overview branch,
    # not somewhere unrelated. The pattern's location follows the
    # ``make_overview_gpu`` call and the ``nan_mask = cupy.isnan(current)``
    # gate; assert the order is preserved.
    idx_overview = src.find("make_overview_gpu(current")
    idx_putmask = src.find("cupy.putmask(", idx_overview)
    assert idx_overview != -1 and idx_putmask != -1, (
        "could not locate the overview-loop in-place rewrite"
    )
    # The legacy two-line pattern would have ``current = current.copy()``
    # right before the indexed write. Ensure the overview branch no
    # longer contains that exact line. Anchor the slice on the next
    # statement after the inner ``while`` (``parts.append(...)``) so
    # the window tracks the real loop body instead of a fixed
    # character count that drifts as surrounding code changes.
    idx_parts_append = src.find("parts.append(", idx_overview)
    assert idx_parts_append != -1, (
        "could not locate the ``parts.append(`` sentinel that closes "
        "the overview-loop body"
    )
    overview_branch = src[idx_overview:idx_parts_append]
    assert "current = current.copy()" not in overview_branch, (
        "overview loop should no longer copy the cupy buffer before "
        "the in-place sentinel rewrite (issue #1948)."
    )


@_gpu_only
def test_gpu_writer_cog_overview_sentinel_roundtrip_1948():
    """COG write -> CPU read preserves the sentinel through every overview.

    Regression check on the correctness of the in-place rewrite. If
    ``cupy.putmask`` ever drifts away from the ``current.copy()`` +
    indexed-write semantics (e.g. broadcasts the sentinel as the wrong
    dtype) the round-trip would surface NaN where the sentinel should
    be, breaking external-reader masking. The test is gated on a GPU
    because the overview loop only runs on the GPU writer code path.
    """
    from xrspatial.geotiff import open_geotiff, write_geotiff_gpu

    sentinel = np.float32(-9999.0)
    height, width = 1024, 1024
    arr_gpu = _make_float_raster_with_nodata(height, width, float(sentinel))

    da = xr.DataArray(
        arr_gpu, dims=["y", "x"],
        coords={"y": np.arange(height), "x": np.arange(width)},
        attrs={"crs": 4326, "nodata": float(sentinel)},
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tmp_1948_cog.tif")
        write_geotiff_gpu(
            da, path, compression="deflate", tile_size=256,
            cog=True, overview_levels=[2, 4],
        )

        # Read full-resolution and the two overview levels.
        full = open_geotiff(path)
        ov1 = open_geotiff(path, overview_level=1)
        ov2 = open_geotiff(path, overview_level=2)

    # Full-resolution: sentinel pixels survive as NaN (the read path
    # masks the sentinel back to NaN since attrs['nodata'] is set).
    assert np.isnan(full.values).any(), (
        "full-resolution overview lost its sentinel pixels"
    )
    assert ov1.shape == (height // 2, width // 2), (
        f"overview level 1 shape {ov1.shape}, expected "
        f"({height // 2}, {width // 2})"
    )
    assert ov2.shape == (height // 4, width // 4), (
        f"overview level 2 shape {ov2.shape}, expected "
        f"({height // 4}, {width // 4})"
    )
    # Each overview must still mask the sentinel. With the prior
    # ``current.copy()`` pattern the test passed because the buffer was
    # copied; with ``cupy.putmask`` the test still passes because the
    # buffer is mutated in place. This guard fires only if the rewrite
    # is dropped entirely or sentinel dtype/promotion semantics change.
    assert np.isnan(ov1.values).any(), (
        "overview level 1 lost its sentinel pixels after the in-place rewrite"
    )
    assert np.isnan(ov2.values).any(), (
        "overview level 2 lost its sentinel pixels after the in-place rewrite"
    )


@_gpu_only
def test_gpu_writer_overview_uses_make_overview_gpu_fresh_buffer_1948():
    """``make_overview_gpu`` returns a freshly allocated cupy buffer.

    The in-place rewrite contract relies on ``make_overview_gpu``
    handing back a buffer that no caller-visible state aliases. This
    guard exercises every overview-method path inside
    ``_block_reduce_2d_gpu`` and confirms the returned buffer is a
    different cupy.ndarray than the input.
    """
    import cupy

    from xrspatial.geotiff._gpu_decode import make_overview_gpu

    rng = np.random.RandomState(1948)
    src_np = rng.rand(64, 64).astype(np.float32)
    src_gpu = cupy.asarray(src_np)

    for method in ("mean", "nearest", "min", "max", "median"):
        out = make_overview_gpu(src_gpu, method=method, nodata=None)
        assert int(out.data.ptr) != int(src_gpu.data.ptr), (
            f"make_overview_gpu(method={method!r}) returned an aliased buffer"
        )
        assert out.shape == (32, 32), (
            f"make_overview_gpu(method={method!r}) returned shape {out.shape}"
        )
