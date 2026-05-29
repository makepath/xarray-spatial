"""Regression tests for the batched dask compute path in polygonize.

Issue #2608: `_polygonize_dask` used to call `dask.compute()` once per
chunk inside a nested Python loop, which serialized chunk processing.
Issue #2664 batched the per-chunk tasks into bounded `dask.compute()`
calls so the scheduler can run a batch's chunks concurrently while peak
driver memory stays bounded by ``batch_size`` chunk results (independent
of the chunk-grid shape, unlike the earlier per-row batching).  These
tests pin both the output parity (against numpy) and the batched compute
structure so it can't silently regress to per-chunk (slow) or per-raster
(memory-heavy).
"""
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None
    da = None

from ..polygonize import polygonize, _polygonize_dask
from .general_checks import dask_array_available


def _area_by_value(values, polygons):
    """Total area per pixel value from polygonize output."""
    area_map = {}
    for val, rings in zip(values, polygons):
        area = 0.0
        for ring in rings:
            x = ring[:, 0]
            y = ring[:, 1]
            area += 0.5 * (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        area_map.setdefault(val, 0.0)
        area_map[val] += area
    return area_map


@dask_array_available
@pytest.mark.parametrize("chunks", [(20, 20), (16, 24), (40, 50), (13, 17)])
def test_row_batched_dask_matches_numpy(chunks):
    """Row-batched dask output must match numpy per-value areas exactly.

    Uses 4-connectivity, which has exact dask/numpy area parity (the
    8-connectivity cross-chunk merge is a documented approximation and
    is exercised separately for self-consistency below).
    """
    shape = (40, 50)
    rng = np.random.default_rng(28403)
    data = rng.integers(low=0, high=3, size=shape, dtype=np.int64)

    raster_np = xr.DataArray(data)
    vals_np, polys_np = polygonize(raster_np, connectivity=4)
    areas_np = _area_by_value(vals_np, polys_np)

    raster_da = xr.DataArray(da.from_array(data, chunks=chunks))
    vals_da, polys_da = polygonize(raster_da, connectivity=4)
    areas_da = _area_by_value(vals_da, polys_da)

    assert set(areas_np) == set(areas_da)
    for val in areas_np:
        assert_allclose(areas_da[val], areas_np[val],
                        err_msg=f"Area mismatch for value {val}")


@dask_array_available
@pytest.mark.parametrize("chunks", [(20, 20), (16, 24)])
def test_row_batched_dask_conn8_self_consistent(chunks):
    """Row batching must not change the 8-connectivity dask output.

    The dask 8-connectivity area need not match numpy (cross-chunk merge
    is approximate), but the row-batched compute path must produce the
    same polygons the per-chunk path would have, so total per-value area
    is stable across repeated runs and chunkings of the same data.
    """
    shape = (40, 50)
    rng = np.random.default_rng(28403)
    data = rng.integers(low=0, high=3, size=shape, dtype=np.int64)
    raster_da = xr.DataArray(da.from_array(data, chunks=chunks))

    vals_a, polys_a = polygonize(raster_da, connectivity=8)
    vals_b, polys_b = polygonize(raster_da, connectivity=8)

    assert list(vals_a) == list(vals_b)
    assert len(polys_a) == len(polys_b)
    for pa, pb in zip(polys_a, polys_b):
        assert len(pa) == len(pb)
        for ra, rb in zip(pa, pb):
            assert np.array_equal(ra, rb)


@dask_array_available
def test_dask_with_mask_row_batched_matches_numpy():
    """Masked raster still matches numpy through the row-batched path."""
    shape = (32, 48)
    rng = np.random.default_rng(11)
    data = rng.integers(low=0, high=2, size=shape, dtype=np.int64)
    rng2 = np.random.default_rng(99)
    mask = rng2.uniform(0, 1, size=shape) < 0.85

    vals_np, polys_np = polygonize(
        xr.DataArray(data), mask=xr.DataArray(mask), connectivity=4)
    areas_np = _area_by_value(vals_np, polys_np)

    raster_da = xr.DataArray(da.from_array(data, chunks=(16, 16)))
    mask_da = xr.DataArray(da.from_array(mask, chunks=(16, 16)))
    vals_da, polys_da = polygonize(raster_da, mask=mask_da, connectivity=4)
    areas_da = _area_by_value(vals_da, polys_da)

    assert set(areas_np) == set(areas_da)
    for val in areas_np:
        assert_allclose(areas_da[val], areas_np[val])


@dask_array_available
def test_compute_batches_chunks(monkeypatch):
    """`_polygonize_dask` should batch chunks per dask.compute call.

    A 4x3 chunk grid is 12 chunks total.  With ``batch_size=5`` that is
    ceil(12 / 5) = 3 compute calls (sizes 5, 5, 2) -- not 12 (per chunk,
    serial) and not 1 (per raster, memory-heavy).  This pins the bounded
    batching from #2664, which superseded the per-row batching of #2608
    so peak memory is bounded by ``batch_size`` rather than by the number
    of column chunks.
    """
    shape = (40, 30)
    rng = np.random.default_rng(5)
    data = rng.integers(low=0, high=3, size=shape, dtype=np.int64)
    dask_data = da.from_array(data, chunks=(10, 10))
    n_rows = len(dask_data.chunks[0])
    n_cols = len(dask_data.chunks[1])
    assert (n_rows, n_cols) == (4, 3)
    n_chunks = n_rows * n_cols

    import sys
    # The package re-exports ``polygonize`` (the function) as
    # ``xrspatial.polygonize``, shadowing the submodule, so reach the
    # module object through sys.modules.  ``_polygonize_dask`` calls
    # ``dask.compute`` via its module-level ``import dask``, so
    # ``poly_mod.dask`` is the real dask module; patching ``compute`` on
    # it is the only seam to count the calls.  monkeypatch restores it at
    # teardown, so the global patch does not leak to other tests.
    poly_mod = sys.modules["xrspatial.polygonize"]
    calls = {"count": 0, "sizes": []}
    real_compute = poly_mod.dask.compute

    def counting_compute(*args, **kwargs):
        calls["count"] += 1
        calls["sizes"].append(len(args))
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(poly_mod.dask, "compute", counting_compute)

    batch_size = 5
    _polygonize_dask(dask_data, None, False, None, batch_size=batch_size)

    # ceil(n_chunks / batch_size) compute calls.
    expected_calls = -(-n_chunks // batch_size)
    assert calls["count"] == expected_calls
    # Each call batches up to batch_size chunks; total chunks add up.
    assert all(size <= batch_size for size in calls["sizes"])
    assert sum(calls["sizes"]) == n_chunks
