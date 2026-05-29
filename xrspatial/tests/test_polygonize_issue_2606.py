"""Regression tests for #2606.

The dask cross-chunk merge for ``connectivity=8`` used to fill in the
diagonal notch where two same-value regions meet only at a corner across
a chunk boundary.  ``_merge_polygon_rings`` traced the notch as a
separate hole ring, but ``_group_rings_into_polygons`` dropped that hole
because it tested containment using the hole's first vertex -- which sits
on the exterior boundary at the pinch point, so ``_point_in_ring``
returned False.  The dropped hole left the merged exterior covering one
extra cell, so the total polygon area came out larger than the raster and
the geometry self-intersected differently from numpy.

These tests pin numpy/dask area parity for ``connectivity=8`` across
dask+numpy and (where available) dask+cupy.
"""
import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from shapely.geometry import Polygon

from xrspatial.polygonize import polygonize

from .general_checks import cuda_and_cupy_available, dask_array_available


# The minimal reproducer from the bug report: two value=1 regions meet
# only diagonally across the row-2 chunk boundary.
_REPRO = np.array(
    [[1, 1, 1, 0],
     [1, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 1, 0, 1]],
    dtype=np.int32,
)


def _per_value_area(values, polygons):
    by_val = {}
    for val, rings in zip(values, polygons):
        # Exterior minus holes -- a dropped hole shows up here.
        area = Polygon(rings[0], rings[1:]).area
        by_val[float(val)] = by_val.get(float(val), 0.0) + area
    return by_val


@dask_array_available
def test_repro_total_area_matches_raster():
    """Total polygon area must equal the raster cell count, not exceed it."""
    rn = xr.DataArray(_REPRO.copy())
    rd = xr.DataArray(da.from_array(_REPRO.copy(), chunks=(2, 4)))

    vn, pn = polygonize(rn, connectivity=8)
    vd, pd = polygonize(rd, connectivity=8)

    np_total = sum(Polygon(r[0], r[1:]).area for r in pn)
    dk_total = sum(Polygon(r[0], r[1:]).area for r in pd)

    assert np_total == pytest.approx(_REPRO.size)
    assert dk_total == pytest.approx(np_total)


@dask_array_available
def test_repro_per_value_area_parity():
    rn = xr.DataArray(_REPRO.copy())
    rd = xr.DataArray(da.from_array(_REPRO.copy(), chunks=(2, 4)))

    vn, pn = polygonize(rn, connectivity=8)
    vd, pd = polygonize(rd, connectivity=8)

    area_np = _per_value_area(vn, pn)
    area_dk = _per_value_area(vd, pd)
    assert set(area_np) == set(area_dk)
    for val in area_np:
        assert area_dk[val] == pytest.approx(area_np[val], abs=1e-12)


@dask_array_available
@pytest.mark.parametrize("seed", range(20))
def test_random_8conn_area_parity(seed):
    """Random integer rasters with random chunkings: 8-conn area parity."""
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(3, 9)), int(rng.integers(3, 9)))
    data = rng.integers(0, 3, size=shape).astype(np.int32)
    chunks = (int(rng.integers(1, shape[0] + 1)),
              int(rng.integers(1, shape[1] + 1)))

    rn = xr.DataArray(data.copy())
    rd = xr.DataArray(da.from_array(data.copy(), chunks=chunks))

    vn, pn = polygonize(rn, connectivity=8)
    vd, pd = polygonize(rd, connectivity=8)

    area_np = _per_value_area(vn, pn)
    area_dk = _per_value_area(vd, pd)
    assert set(area_np) == set(area_dk), f"seed={seed} value set mismatch"
    for val in area_np:
        assert area_dk[val] == pytest.approx(area_np[val], abs=1e-12), (
            f"seed={seed} chunks={chunks} value {val} area mismatch")


@cuda_and_cupy_available
@dask_array_available
def test_repro_dask_cupy_area_parity():
    import cupy

    rn = xr.DataArray(_REPRO.copy())
    rdc = xr.DataArray(
        da.from_array(cupy.asarray(_REPRO.copy()), chunks=(2, 4)))

    vn, pn = polygonize(rn, connectivity=8)
    vc, pc = polygonize(rdc, connectivity=8)

    area_np = _per_value_area(vn, pn)
    area_dc = _per_value_area(vc, pc)
    assert set(area_np) == set(area_dc)
    for val in area_np:
        assert area_dc[val] == pytest.approx(area_np[val], abs=1e-12)
