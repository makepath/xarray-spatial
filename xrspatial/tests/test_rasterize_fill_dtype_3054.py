"""Regression coverage for issue #3054 (metadata sweep).

``rasterize()`` casts its float64 work buffer to the output dtype at the
end of every backend, while the attrs block stores the original ``fill``
verbatim.  Two cases used to slip past the old NaN-only-integer guard and
leave the burned array disagreeing with its ``nodata`` / ``_FillValue`` /
``nodatavals`` attrs:

  * An out-of-range integer fill: ``dtype=np.uint8, fill=-9999`` burned
    241 but advertised -9999.
  * A boolean dtype: ``np.issubdtype(np.bool_, np.integer)`` is False, so
    ``dtype=np.bool_, fill=np.nan`` slipped through, turning every
    unwritten pixel into ``True`` with no nodata attrs at all.

The guard now rejects any fill the output dtype cannot represent exactly
(integer overflow and boolean included).  These tests pin that across all
four backends so a refactor that re-introduces the silent cast surfaces
in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import cupy  # noqa: F401
    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import dask.array  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

try:
    from numba import cuda
    has_cuda = has_cupy and cuda.is_available()
except Exception:
    has_cuda = False

if has_shapely:
    from xrspatial.rasterize import rasterize

skip_no_shapely = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed")
skip_no_dask = pytest.mark.skipif(
    not has_dask, reason="dask not installed")
skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")


def _square():
    return [(box(2, 2, 8, 8), 1.0)]


# --------------------------------------------------------------------------
# Out-of-range integer fill (the uint8 / -9999 -> 241 reproduction).
# --------------------------------------------------------------------------

@skip_no_shapely
@pytest.mark.parametrize("dt,fill", [
    (np.uint8, -9999),
    (np.uint8, 256),
    (np.int8, 200),
    (np.uint16, -1),
])
def test_out_of_range_int_fill_raises_numpy(dt, fill):
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=fill, dtype=dt)


@skip_no_shapely
def test_huge_int_fill_raises_cleanly():
    """A fill wider than the platform C long is rejected, not crashed on.

    ``np.array(2**100)`` is an object array whose ``astype(<int>)`` raises
    ``OverflowError`` rather than ``ValueError``; the guard catches it so
    the user sees the same actionable message instead of a stray
    ``OverflowError``.
    """
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=2 ** 100, dtype=np.int64)


@skip_no_shapely
@skip_no_dask
def test_out_of_range_int_fill_raises_dask_numpy():
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=-9999, dtype=np.uint8, chunks=5)


@skip_no_shapely
@skip_no_cuda
def test_out_of_range_int_fill_raises_cupy():
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=-9999, dtype=np.uint8, gpu=True)


@skip_no_shapely
@skip_no_cuda
@skip_no_dask
def test_out_of_range_int_fill_raises_dask_cupy():
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=-9999, dtype=np.uint8, gpu=True, chunks=5)


@skip_no_shapely
def test_in_range_int_fill_round_trips():
    """A fill the dtype can hold is burned and advertised consistently."""
    r = rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=200, dtype=np.uint8)
    assert r.dtype == np.uint8
    assert r.values[0, 0] == 200
    assert r.attrs.get('nodata') == 200
    assert r.attrs.get('_FillValue') == 200
    assert r.attrs.get('nodatavals') == (200,)


# --------------------------------------------------------------------------
# Boolean dtype (NaN -> True everywhere, no nodata attrs).
# --------------------------------------------------------------------------

@skip_no_shapely
@pytest.mark.parametrize("fill", [np.nan, 2, -1])
def test_bool_dtype_unrepresentable_fill_raises_numpy(fill):
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=fill, dtype=np.bool_)


@skip_no_shapely
@skip_no_dask
def test_bool_dtype_nan_fill_raises_dask_numpy():
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=np.nan, dtype=np.bool_, chunks=5)


@skip_no_shapely
@skip_no_cuda
def test_bool_dtype_nan_fill_raises_cupy():
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=np.nan, dtype=np.bool_, gpu=True)


@skip_no_shapely
@skip_no_cuda
@skip_no_dask
def test_bool_dtype_nan_fill_raises_dask_cupy():
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=np.nan, dtype=np.bool_, gpu=True, chunks=5)


@skip_no_shapely
@pytest.mark.parametrize("fill", [False, True, 0, 1])
def test_bool_dtype_representable_fill_round_trips(fill):
    """``fill=False``/``True`` (and 0/1) are exact for bool and accepted."""
    r = rasterize(_square(), width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=fill, dtype=np.bool_)
    assert r.dtype == np.bool_
    assert bool(r.values[0, 0]) == bool(fill)
    # Non-False fill emits the nodata triplet; False is falsy so the
    # existing ``if not fill_is_nan``/``nodata`` block keys off the value
    # exactly like any other sentinel.
    assert r.attrs.get('nodata') == fill
    assert r.attrs.get('_FillValue') == fill
    assert r.attrs.get('nodatavals') == (fill,)
