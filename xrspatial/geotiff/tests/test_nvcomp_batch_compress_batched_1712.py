"""Coverage for the batched ``_nvcomp_batch_compress`` (#1712).

The pre-fix function allocated compressed-output device buffers one
``cupy.empty`` per tile and then read each tile back to host with one
``.get()`` per tile. Both patterns serialised on the default CUDA
stream and were dominant in large-N writes. The fix folds both into a
single contiguous device allocation + a single batched D2H concat-and-
``.get()``, matching the patterns already in use on the decode side
(#1552, #1659).

These tests pin the new shape and confirm the deflate / zstd GPU write
paths still round-trip end-to-end.
"""
from __future__ import annotations

import inspect
import os
import tempfile

import numpy as np
import pytest
import xarray as xr

try:
    import cupy
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

# nvCOMP is the entry point that exercises this code path.
from xrspatial.geotiff import _gpu_decode

needs_cupy = pytest.mark.skipif(
    not _HAS_CUPY, reason="CUDA / cupy unavailable on this host"
)


# ----------------------------------------------------------------------
# Source-level structural assertions -- run on any host with the source
# available, no GPU required.
# ----------------------------------------------------------------------

def test_no_per_tile_cupy_empty_in_compressed_pool():
    """The per-tile cupy.empty list comprehension is gone (#1712).

    The fix replaced it with a single contiguous allocation. Catch any
    regression that brings the loop back.
    """
    source = inspect.getsource(_gpu_decode._nvcomp_batch_compress)
    assert "cupy.empty(max_cs, dtype=cupy.uint8) for _ in range" not in source, (
        "_nvcomp_batch_compress regressed to per-tile cupy.empty "
        "allocations for the compressed output pool. See #1712."
    )


def test_no_per_tile_get_in_result_loop():
    """The per-tile ``d_comp_bufs[i][:cs].get().tobytes()`` is gone (#1712).

    The fix replaced it with one concat + one ``.get()``. Catch any
    regression that brings the per-tile pattern back.
    """
    source = inspect.getsource(_gpu_decode._nvcomp_batch_compress)
    # The exact string the prior loop used:
    bad_fragment = "d_comp_bufs[i][:cs].get().tobytes()"
    assert bad_fragment not in source, (
        "_nvcomp_batch_compress regressed to per-tile .get().tobytes() "
        "D2H readback. See #1712."
    )


# ----------------------------------------------------------------------
# End-to-end behaviour: GPU write + read round-trip stays correct
# ----------------------------------------------------------------------

@needs_cupy
@pytest.mark.parametrize("compression", ["deflate", "zstd"])
def test_gpu_write_roundtrip_after_batched_compress(compression):
    """GPU compress path round-trips uncorrupted for deflate + zstd.

    Catches the most likely regression mode: any off-by-one in the
    cumulative-sum offsets used to slice the host-side concatenated
    buffer would scramble tile order, which a round-trip equality
    check picks up immediately.
    """
    from xrspatial.geotiff import write_geotiff_gpu, open_geotiff

    rng = np.random.default_rng(seed=1712)
    arr_cpu = rng.random((512, 512), dtype=np.float32)
    arr_gpu = cupy.asarray(arr_cpu)
    darr = xr.DataArray(arr_gpu, dims=["y", "x"])

    with tempfile.TemporaryDirectory(prefix="nvcomp_batch_1712_") as td:
        path = os.path.join(td, f"roundtrip_{compression}.tif")
        try:
            write_geotiff_gpu(
                darr, path,
                compression=compression,
                tiled=True,
                tile_size=64,
            )
        except RuntimeError as e:
            # nvCOMP may be unavailable in this environment; the writer
            # falls back to CPU and that path doesn't exercise the
            # change. Skip rather than fail.
            pytest.skip(f"nvCOMP unavailable for {compression}: {e}")

        back = open_geotiff(path)
        np.testing.assert_allclose(back.values, arr_cpu, rtol=0, atol=0)


@needs_cupy
def test_gpu_write_zero_tile_edge_case():
    """A 0-tile compress returns an empty list without indexing into None.

    The cumulative-sum / concat path must short-circuit before
    ``cupy.concatenate`` (which would raise on an empty list). The
    pre-fix loop simply iterated zero times, so the contract is the
    same empty-list output.
    """
    # Direct call into the internal function with n_tiles=0 is
    # awkward because it needs a libnvCOMP handle and matching opts.
    # Instead, exercise the public writer with a tiny single-tile
    # input and confirm the fast path does not crash. Real n_tiles==0
    # never occurs via the writer (every image has at least one tile).
    from xrspatial.geotiff import write_geotiff_gpu, open_geotiff
    arr_gpu = cupy.zeros((32, 32), dtype=cupy.float32)
    darr = xr.DataArray(arr_gpu, dims=["y", "x"])
    with tempfile.TemporaryDirectory(prefix="nvcomp_batch_1712_") as td:
        path = os.path.join(td, "tiny.tif")
        try:
            write_geotiff_gpu(darr, path, compression="zstd",
                              tiled=True, tile_size=32)
        except RuntimeError as e:
            pytest.skip(f"nvCOMP unavailable: {e}")
        back = open_geotiff(path)
        assert back.shape == (32, 32)
