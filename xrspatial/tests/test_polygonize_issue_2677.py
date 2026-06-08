"""Dask 8-connectivity float polygonize parity at large rtol (#2677).

numpy connected-component labelling consults an 8-connected diagonal
neighbour *conditionally*: a pixel only looks at its SW diagonal when its
W neighbour did not match, and at its SE diagonal only when its S
neighbour did not match (see the ``connectivity_8`` block in
``_calculate_regions``).  The dask cross-chunk merge in
``_cells_close_directed`` previously honoured every diagonal pair
unconditionally, so at a large ``rtol`` it could union two regions that
numpy keeps separate when the intervening orthogonal neighbour already
matched.  The result was a chunked 8-conn float raster reporting fewer
polygons (and different DN values) than the unchunked input.

This is distinct from the #2666 rtol scan-order orientation: it fails
across chunkings that do not split the diverging pixels, so it is the
diagonal-pairing decision, not the cross-chunk boundary close test, that
was wrong.  The fix carries per-cell ``w_match`` / ``s_match`` flags plus
a global boundary-value map so the merge consults a diagonal only when
numpy would, for both in-chunk and cross-chunk orthogonal neighbours.

These tests pin numpy/dask parity for the issue repro across every
chunking, and add a deterministic fuzz over random float rasters so the
8-conn merge stays chunk-invariant.
"""
import itertools

import numpy as np
import pytest
import xarray as xr

try:
    import cupy
except ImportError:
    cupy = None

try:
    import dask.array as da
except ImportError:
    da = None

from ..polygonize import polygonize
from .general_checks import cuda_and_cupy_available, dask_array_available


def _to_dask(arr, chunks):
    return xr.DataArray(da.from_array(arr, chunks=chunks))


def _to_dask_cupy(arr, chunks):
    return xr.DataArray(da.from_array(cupy.asarray(arr), chunks=chunks))


def _signature(values, geoms):
    """Order-independent polygon signature: DN value plus exterior-ring
    min corner and vertex count.  Stronger than a bare count + DN multiset
    because it also pins where each polygon sits and how big it is.
    """
    sig = []
    for v, rings in zip(values, geoms):
        ext = rings[0]
        sig.append((
            round(float(v), 9),
            round(float(ext[:, 0].min()), 6),
            round(float(ext[:, 1].min()), 6),
            len(ext),
        ))
    return sorted(sig)


def _all_chunkings(shape):
    ny, nx = shape
    return sorted(set(itertools.product(range(1, ny + 1), range(1, nx + 1))))


# The exact repro from the issue: numpy reports 6 polygons (8-conn,
# atol=0, rtol=0.1); the unpatched dask merge reported fewer because it
# unioned the value-2.206 region into the value-2.098 region through a
# diagonal numpy never consults.
_REPRO = np.array(
    [
        [2.098, 2.43, 2.206, 2.09],
        [1.847, 2.292, 1.875, 2.784],
        [2.927, 1.767, 2.583, 2.058],
        [2.136, 2.851, 1.142, 1.174],
    ],
    dtype=np.float64,
)


@dask_array_available
class TestReproChunkInvariant:
    """The issue repro must match the unchunked numpy partition for every
    chunking, not just the count.
    """

    def test_numpy_reference_polygon_count(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO), atol=0.0, rtol=0.1, connectivity=8)
        assert len(v_np) == 6

    @pytest.mark.parametrize("chunks", _all_chunkings(_REPRO.shape))
    def test_dask_matches_numpy(self, chunks):
        v_np, g_np = polygonize(
            xr.DataArray(_REPRO), atol=0.0, rtol=0.1, connectivity=8)
        v_dk, g_dk = polygonize(
            _to_dask(_REPRO, chunks=chunks),
            atol=0.0, rtol=0.1, connectivity=8)
        assert len(v_dk) == len(v_np), (
            f"count mismatch chunks={chunks}: "
            f"numpy={len(v_np)} dask={len(v_dk)}"
        )
        assert _signature(v_dk, g_dk) == _signature(v_np, g_np)


@dask_array_available
class TestEightConnFuzzParity:
    """Random float rasters: every chunking of an 8-conn polygonize must
    reproduce the unchunked numpy polygon partition.  Deterministic via a
    fixed seed -- no wall-clock assertions.
    """

    # Total numpy polygon count over the fixed-seed sweep, recorded per
    # rtol.  Pinning it means a numpy-side CCL change is caught here
    # directly, rather than silently shifting the dask parity reference.
    _EXPECTED_NUMPY_TOTAL = {0.0: 639, 0.05: 492, 0.1: 352, 0.2: 200}

    @pytest.mark.parametrize("rtol", [0.0, 0.05, 0.1, 0.2])
    def test_parity_8conn(self, rtol):
        rng = np.random.default_rng(2677)
        numpy_total = 0
        for _ in range(40):
            ny = int(rng.integers(3, 6))
            nx = int(rng.integers(3, 6))
            arr = np.round(rng.uniform(1.0, 3.0, size=(ny, nx)), 3)
            v_np, g_np = polygonize(
                xr.DataArray(arr), atol=0.0, rtol=rtol, connectivity=8)
            numpy_total += len(v_np)
            ref = _signature(v_np, g_np)
            for chunks in _all_chunkings((ny, nx)):
                v_dk, g_dk = polygonize(
                    _to_dask(arr, chunks=chunks),
                    atol=0.0, rtol=rtol, connectivity=8)
                assert _signature(v_dk, g_dk) == ref, (
                    f"diverge arr={arr.tolist()} rtol={rtol} "
                    f"chunks={chunks}"
                )
        expected = self._EXPECTED_NUMPY_TOTAL[rtol]
        assert expected is None or numpy_total == expected, (
            f"numpy 8-conn reference count changed for rtol={rtol}: "
            f"got {numpy_total}, expected {expected}"
        )


@dask_array_available
class TestFourConnUnaffected:
    """4-connectivity already matched numpy after #2675; the diagonal
    guard must leave it untouched.
    """

    @pytest.mark.parametrize("chunks", _all_chunkings(_REPRO.shape))
    def test_4conn_repro_unchanged(self, chunks):
        v_np, g_np = polygonize(
            xr.DataArray(_REPRO), atol=0.0, rtol=0.1, connectivity=4)
        v_dk, g_dk = polygonize(
            _to_dask(_REPRO, chunks=chunks),
            atol=0.0, rtol=0.1, connectivity=4)
        assert _signature(v_dk, g_dk) == _signature(v_np, g_np)


@dask_array_available
class TestIntegerEightConnUnaffected:
    """Integer 8-conn rasters use strict equality; the float diagonal
    guard must not change their chunk-invariant result.
    """

    def test_integer_chunk_invariance(self):
        arr = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]], dtype=np.int32)
        v_np, _ = polygonize(xr.DataArray(arr), connectivity=8)
        for chunks in _all_chunkings(arr.shape):
            v_dk, _ = polygonize(
                _to_dask(arr, chunks=chunks), connectivity=8)
            assert sorted(v_dk) == sorted(v_np)


@cuda_and_cupy_available
@dask_array_available
class TestReproDaskCuPy:
    """dask+cupy shares the cross-chunk merge path, so the repro must
    match numpy there too.
    """

    @pytest.mark.parametrize("chunks", [(2, 2), (1, 4), (4, 1)])
    def test_repro_dask_cupy(self, chunks):
        v_np, g_np = polygonize(
            xr.DataArray(_REPRO), atol=0.0, rtol=0.1, connectivity=8)
        v_dc, g_dc = polygonize(
            _to_dask_cupy(_REPRO, chunks=chunks),
            atol=0.0, rtol=0.1, connectivity=8)
        assert _signature(v_dc, g_dc) == _signature(v_np, g_np)
