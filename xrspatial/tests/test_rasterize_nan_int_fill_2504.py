"""Regression coverage for issue #2504 (metadata sweep, Cat 4).

``rasterize(..., dtype=<int>)`` with the default ``fill=np.nan`` used to
silently coerce NaN to a platform-specific sentinel
(``np.iinfo(dtype).min`` on x86, ``0`` elsewhere) and emit no
``_FillValue`` / ``nodata`` / ``nodatavals`` attr to mark unwritten
pixels.  Downstream consumers (geotiff writer, rioxarray masks) had no
sentinel to key off and treated unwritten cells as valid data.  The
sweep replaced the cast with an upfront ``ValueError``.  These tests
pin that behaviour across all four backends so a refactor that
re-introduces the silent cast surfaces in CI.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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


INT_DTYPES = [np.int8, np.int16, np.int32, np.int64,
              np.uint8, np.uint16, np.uint32, np.uint64]


@skip_no_shapely
@pytest.mark.parametrize("dt", INT_DTYPES)
def test_int_dtype_default_nan_fill_raises_numpy(dt):
    """Numpy backend: ``dtype=<int>`` + default ``fill=np.nan`` raises."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), 1.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  dtype=dt)


@skip_no_shapely
@pytest.mark.parametrize("dt", [np.int16, np.int32, np.uint8])
def test_like_int_dtype_default_nan_fill_raises_numpy(dt):
    """``like`` with integer dtype trips the same guard."""
    x = np.linspace(0.5, 9.5, 10)
    y = np.linspace(9.5, 0.5, 10)
    like = xr.DataArray(
        np.zeros((10, 10), dtype=dt), dims=['y', 'x'],
        coords={'y': y, 'x': x}, attrs={'crs': 'EPSG:32610'})
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), 1.0)], like=like)


@skip_no_shapely
@skip_no_dask
def test_int_dtype_default_nan_fill_raises_dask_numpy():
    """Dask+numpy backend trips the guard before graph construction."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), 1.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  dtype=np.int32, chunks=5)


@skip_no_shapely
@skip_no_cuda
def test_int_dtype_default_nan_fill_raises_cupy():
    """CuPy backend trips the guard before any device allocation."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), 1.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  dtype=np.int32, gpu=True)


@skip_no_shapely
@skip_no_cuda
@skip_no_dask
def test_int_dtype_default_nan_fill_raises_dask_cupy():
    """Dask+CuPy backend trips the guard before any device allocation."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), 1.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  dtype=np.int32, gpu=True, chunks=5)


@skip_no_shapely
def test_explicit_int_fill_with_int_dtype_emits_nodata_triplet():
    """Explicit int fill round-trips through nodata / _FillValue / nodatavals."""
    r = rasterize([(box(2, 2, 8, 8), 1.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=-9999, dtype=np.int32)
    assert r.dtype == np.int32
    assert r.values[0, 0] == -9999
    # Sentinel lands on all three keys so downstream code can resolve it
    # via whichever convention it uses (xrspatial's `nodata`, the CF
    # alias `_FillValue`, or rioxarray's per-band `nodatavals`).
    assert r.attrs.get('nodata') == -9999
    assert r.attrs.get('_FillValue') == -9999
    assert r.attrs.get('nodatavals') == (-9999,)


@skip_no_shapely
def test_float_dtype_default_nan_fill_unaffected():
    """The guard only fires on integer dtypes; float dtypes accept NaN."""
    r = rasterize([(box(2, 2, 8, 8), 1.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10))
    assert r.dtype == np.float64
    assert np.isnan(r.values[0, 0])
    # Default NaN fill on a float dtype intentionally emits no nodata
    # keys (NaN is its own marker).  Pin that behaviour stays unchanged.
    assert 'nodata' not in r.attrs
    assert '_FillValue' not in r.attrs
    assert 'nodatavals' not in r.attrs


@skip_no_shapely
def test_int_dtype_explicit_nan_fill_also_raises():
    """``fill=np.nan`` passed explicitly trips the same guard."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), 1.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=np.nan, dtype=np.int32)


@skip_no_shapely
def test_int_dtype_numpy_typed_nan_fill_raises():
    """``np.float32(np.nan)`` is still NaN and still trips the guard.

    ``isinstance(np.float32(np.nan), float)`` is False, so a naive
    ``isinstance(fill, float)`` check would let this slip through.  The
    guard normalises through ``float()`` then ``np.isnan`` to catch
    every NaN flavour.
    """
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), 1.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=np.float32(np.nan), dtype=np.int32)
