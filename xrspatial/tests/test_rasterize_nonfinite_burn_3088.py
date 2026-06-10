"""Regression coverage for issue #3088 (security sweep, Cat 6).

``rasterize(..., dtype=<int>)`` with a non-finite *burn value* (NaN or
+-inf) used to silently cast the burned pixels to a platform sentinel
(INT_MIN, -2147483648 for int32) on every backend, and ``dtype=bool``
collapsed them to ``True``.  The fill-value guard (#2504 / #3059) and
the safe-integer guard (#3056) both missed this case: the former only
looks at ``fill``, the latter only examines finite values.  The fix
rejects non-finite burn values up front when the resolved output dtype
is integer or bool, before any backend dispatch.  NaN burns into float
dtypes are intended NaN propagation (#2255) and must keep working.
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

try:
    import geopandas as gpd
    has_geopandas = True
except ImportError:
    has_geopandas = False

if has_shapely:
    from xrspatial.rasterize import rasterize

skip_no_shapely = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed")
skip_no_dask = pytest.mark.skipif(
    not has_dask, reason="dask not installed")
skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")
skip_no_geopandas = pytest.mark.skipif(
    not has_geopandas, reason="geopandas not installed")


INT_DTYPES = [np.int8, np.int16, np.int32, np.int64,
              np.uint8, np.uint16, np.uint32, np.uint64]
NONFINITE = [np.nan, np.inf, -np.inf]


@skip_no_shapely
@pytest.mark.parametrize("dt", INT_DTYPES)
def test_nan_burn_int_dtype_raises_numpy(dt):
    """Numpy backend: NaN burn value + integer dtype raises up front."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), float('nan'))],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=dt)


@skip_no_shapely
@pytest.mark.parametrize("value", NONFINITE)
def test_nonfinite_burn_int_dtype_raises_numpy(value):
    """inf and -inf trip the same guard as NaN."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), value)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=np.int32)


@skip_no_shapely
def test_nan_burn_bool_dtype_raises():
    """bool output dtype rejects NaN burns (would collapse to True)."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), float('nan'))],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=False, dtype=bool)


@skip_no_shapely
def test_mixed_finite_and_nan_burn_raises():
    """One bad value among finite ones is enough to raise."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(0, 0, 4, 4), 7.0),
                   (box(5, 5, 9, 9), float('nan'))],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=np.int32)


@skip_no_shapely
def test_nan_burn_like_int_template_raises():
    """``like`` with an integer dtype resolves to int and trips the guard."""
    x = np.linspace(0.5, 9.5, 10)
    y = np.linspace(9.5, 0.5, 10)
    like = xr.DataArray(
        np.zeros((10, 10), dtype=np.int32), dims=['y', 'x'],
        coords={'y': y, 'x': x})
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), float('nan'))], like=like, fill=0)


@skip_no_shapely
@skip_no_dask
def test_nan_burn_int_dtype_raises_dask_numpy():
    """Dask+numpy backend trips the guard before graph construction."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), float('nan'))],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=np.int32, chunks=5)


@skip_no_shapely
@skip_no_cuda
def test_nan_burn_int_dtype_raises_cupy():
    """CuPy backend trips the guard before any device allocation."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), float('nan'))],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=np.int32, use_cuda=True)


@skip_no_shapely
@skip_no_cuda
@skip_no_dask
def test_nan_burn_int_dtype_raises_dask_cupy():
    """Dask+CuPy backend trips the guard before any device allocation."""
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize([(box(2, 2, 8, 8), float('nan'))],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=np.int32, use_cuda=True, chunks=5)


@skip_no_shapely
@pytest.mark.parametrize("value", NONFINITE)
def test_nonfinite_burn_float_dtype_unaffected(value):
    """Float dtypes keep propagating NaN / inf burns (#2255 behaviour)."""
    r = rasterize([(box(2, 2, 8, 8), value)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0.0)
    assert r.dtype == np.float64
    if np.isnan(value):
        assert np.isnan(r.values[5, 5])
    else:
        assert r.values[5, 5] == value
    assert r.values[0, 0] == 0.0


@skip_no_shapely
@skip_no_geopandas
def test_nan_attribute_column_gdf_int_dtype_raises():
    """The realistic entry point: a GeoDataFrame burn column with a
    missing value plus an integer output dtype.  ``_parse_input`` builds
    props_array via ``.values.astype(float64)`` on this path (vs
    ``float()`` per item for pairs), so pin it separately."""
    gdf = gpd.GeoDataFrame({
        'value': [7.0, np.nan],
        'geometry': [box(0, 0, 4, 4), box(5, 5, 9, 9)],
    })
    with pytest.raises(ValueError, match="cannot be represented"):
        rasterize(gdf, column='value',
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=np.int32)


@skip_no_shapely
def test_finite_burn_int_dtype_still_works():
    """Finite burn values into integer dtypes are unaffected by the guard."""
    r = rasterize([(box(2, 2, 8, 8), 7.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=np.int32, merge='sum')
    assert r.dtype == np.int32
    assert r.values[5, 5] == 7
    assert r.values[0, 0] == 0
