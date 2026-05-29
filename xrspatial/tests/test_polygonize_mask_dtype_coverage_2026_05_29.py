"""Mask dtype coverage for polygonize (deep-sweep test-coverage, Cat 4).

polygonize's ``mask`` parameter is documented to accept bool, integer, or
float values (see the public docstring: "Pixels to include should have
mask values of 1 or True, pixels to exclude should have 0 or False").
Every pre-existing test passes only a ``bool`` mask, so the integer- and
float-mask code paths were never exercised.

Inside the algorithm the mask is consumed as a truth value:
``_calculate_regions`` indexes ``mask[ij]``, and the float-raster branch of
``_polygonize_numpy`` does ``mask = mask & nan_mask``. A regression that
mishandled a non-bool mask would silently change which pixels participate.

What these tests pin:

* Integer masks (int32 / int64) produce results identical to the
  equivalent bool mask, on every backend (numpy, cupy, dask+numpy,
  dask+cupy), for both raster dtypes and both connectivities.  Each
  backend is compared against ITS OWN bool-mask output, so the only
  variable under test is the mask dtype -- not the unrelated
  numpy-vs-dask polygon-representation differences.

* Float masks: integer rasters accept a float mask, but a float-dtype
  mask on a float-dtype raster currently raises ``TypeError`` from the
  ``mask & nan_mask`` line (xrspatial/polygonize.py:918).  That is a
  source bug tracked in #2623; the float-mask cases are marked xfail so
  the expectation is recorded and the test flips green automatically
  once the source is fixed.

CUDA-dependent variants are guarded with the project's GPU-skip
decorator.
"""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

try:
    import dask.array as da
except ImportError:
    da = None

from ..polygonize import polygonize
from .general_checks import cuda_and_cupy_available, dask_array_available


# Integer raster with a value that forms a hole, plus a 0/1 mask pattern
# whose 0 cell excludes the centre pixel so the mask actually changes the
# output.
_IDATA = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
# Float raster: exercises the float-only ``mask & nan_mask`` branch.
_FDATA = np.array([[1.0, 1.0, 2.0], [1.0, 3.0, 2.0], [1.0, 1.0, 2.0]],
                  dtype=np.float64)
_MASK_PATTERN = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int64)


def _result_signature(values, polygons):
    """Backend-independent signature: sorted (value, ring-vertex-counts).

    Region ordering and exact coordinates can differ across backends, so
    reduce each polygon to its value plus the per-ring vertex counts and
    sort.  That is stable enough to detect a mask that lets the wrong
    pixels through (different polygon count / different ring structure)
    while tolerating benign backend ordering differences.
    """
    sig = []
    for val, rings in zip(values, polygons):
        ring_lens = tuple(len(r) for r in rings)
        sig.append((float(val), ring_lens))
    return sorted(sig)


def _make_raster(data, backend, chunks=(2, 2)):
    if backend == "numpy":
        return xr.DataArray(data)
    if backend == "dask":
        return xr.DataArray(da.from_array(data, chunks=chunks))
    if backend == "cupy":
        import cupy
        return xr.DataArray(cupy.asarray(data))
    if backend == "dask_cupy":
        import cupy
        return xr.DataArray(da.from_array(cupy.asarray(data), chunks=chunks))
    raise ValueError(backend)


def _bool_reference_same_backend(data, backend, connectivity):
    """Polygonize with a bool mask on the SAME backend as the test."""
    raster = _make_raster(data, backend)
    mask = _make_raster(_MASK_PATTERN.astype(np.bool_), backend)
    return polygonize(raster, mask=mask, connectivity=connectivity)


# ---------------------------------------------------------------------------
# Integer masks: must match the equivalent bool mask on every backend.
# ---------------------------------------------------------------------------

_INT_MASK_DTYPES = [np.int32, np.int64]

_BACKEND_MARKS = {
    "numpy": [],
    "dask": [dask_array_available],
    "cupy": [cuda_and_cupy_available],
    "dask_cupy": [cuda_and_cupy_available, dask_array_available],
}


def _backend_params():
    params = []
    for backend, marks in _BACKEND_MARKS.items():
        params.append(pytest.param(backend, marks=marks))
    return params


@pytest.mark.parametrize("backend", _backend_params())
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("mask_dtype", _INT_MASK_DTYPES)
@pytest.mark.parametrize(
    "data", [_IDATA, _FDATA], ids=["int_raster", "float_raster"])
def test_integer_mask_matches_bool_mask(
        data, mask_dtype, connectivity, backend):
    """An integer mask equals the equivalent bool mask, same backend."""
    ref_vals, ref_polys = _bool_reference_same_backend(
        data, backend, connectivity)
    ref_sig = _result_signature(ref_vals, ref_polys)

    raster = _make_raster(data, backend)
    mask = _make_raster(_MASK_PATTERN.astype(mask_dtype), backend)
    vals, polys = polygonize(raster, mask=mask, connectivity=connectivity)

    assert _result_signature(vals, polys) == ref_sig


# ---------------------------------------------------------------------------
# Float masks.
# ---------------------------------------------------------------------------

_FLOAT_MASK_DTYPES = [np.float32, np.float64]


@pytest.mark.parametrize("backend", _backend_params())
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("mask_dtype", _FLOAT_MASK_DTYPES)
def test_float_mask_on_integer_raster_matches_bool(
        mask_dtype, backend, connectivity):
    """A float mask works on an integer raster (no ``mask & nan_mask``)."""
    ref_vals, ref_polys = _bool_reference_same_backend(
        _IDATA, backend, connectivity)
    ref_sig = _result_signature(ref_vals, ref_polys)

    raster = _make_raster(_IDATA, backend)
    mask = _make_raster(_MASK_PATTERN.astype(mask_dtype), backend)
    vals, polys = polygonize(raster, mask=mask, connectivity=connectivity)

    assert _result_signature(vals, polys) == ref_sig


@pytest.mark.xfail(
    reason="float mask on float raster raises TypeError from "
           "mask & nan_mask (polygonize.py:918); source bug #2623",
    raises=TypeError, strict=True)
@pytest.mark.parametrize("backend", _backend_params())
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("mask_dtype", _FLOAT_MASK_DTYPES)
def test_float_mask_on_float_raster_2623(mask_dtype, backend, connectivity):
    """A float mask on a float raster should polygonize, but currently
    crashes (#2623).  xfail-strict so this flips to a pass once the
    source casts the mask to bool before ``& nan_mask``."""
    ref_vals, ref_polys = _bool_reference_same_backend(
        _FDATA, backend, connectivity)
    ref_sig = _result_signature(ref_vals, ref_polys)

    raster = _make_raster(_FDATA, backend)
    mask = _make_raster(_MASK_PATTERN.astype(mask_dtype), backend)
    vals, polys = polygonize(raster, mask=mask, connectivity=connectivity)

    assert _result_signature(vals, polys) == ref_sig


# ---------------------------------------------------------------------------
# Sanity anchor: a non-bool mask with a 0 cell really drops that pixel.
# ---------------------------------------------------------------------------

def test_nonbool_mask_actually_excludes_pixels():
    """Without this, "matches the bool reference" would be vacuous if the
    bool reference itself were broken.  Pin that masking the centre cell
    removes the value-3 region from the float raster."""
    raster = xr.DataArray(_FDATA)
    mask = xr.DataArray(_MASK_PATTERN.astype(np.int32))  # int mask, 0 centre
    vals, _ = polygonize(raster, mask=mask, connectivity=4)
    assert 3.0 not in [float(v) for v in vals]

    vals_nomask, _ = polygonize(raster, connectivity=4)
    assert 3.0 in [float(v) for v in vals_nomask]


@pytest.mark.parametrize("mask_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_integer_mask_exact_coords_numpy(mask_dtype, connectivity):
    """Tighter numpy check: an integer mask reproduces the bool mask's
    exact polygon coordinates, not just the vertex-count signature.

    The signature comparison used elsewhere tolerates benign backend
    ordering differences but could miss a mask that swapped which pixels
    pass while keeping the same ring shape.  On a single backend (numpy)
    the region order is deterministic, so assert the full coordinate
    arrays match.
    """
    raster = xr.DataArray(_IDATA)
    ref_vals, ref_polys = polygonize(
        raster, mask=xr.DataArray(_MASK_PATTERN.astype(np.bool_)),
        connectivity=connectivity)

    vals, polys = polygonize(
        raster, mask=xr.DataArray(_MASK_PATTERN.astype(mask_dtype)),
        connectivity=connectivity)

    assert [float(v) for v in vals] == [float(v) for v in ref_vals]
    assert len(polys) == len(ref_polys)
    for rings, ref_rings in zip(polys, ref_polys):
        assert len(rings) == len(ref_rings)
        for ring, ref_ring in zip(rings, ref_rings):
            assert_array_equal(ring, ref_ring)
