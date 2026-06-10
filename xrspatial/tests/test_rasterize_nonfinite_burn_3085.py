"""Regression coverage for issue #3085.

``rasterize()`` computes in float64 and casts to the output dtype at the
end.  A non-finite burn value (NaN or +/-inf, e.g. a GeoDataFrame column
with missing data) against an integer output dtype used to land on a
platform sentinel (NaN -> -2147483648 for int32) -- silently on the numpy
backend, whose final cast suppresses the RuntimeWarning.  Against bool the
cast collapses NaN to ``True``.

The fix rejects non-finite burn values up front with a ``ValueError`` that
names the offending value, mirroring the NaN-fill guard (#2504) and the
unsafe-integer guard (#3056).  ``merge='count'`` is exempt: it burns
overlap counts and never reads property values.  Float dtypes are
unaffected.  These tests pin that behaviour across all four backends.
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
    import geopandas as gpd
    has_geopandas = True
except ImportError:
    has_geopandas = False

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
skip_no_geopandas = pytest.mark.skipif(
    not has_geopandas, reason="geopandas not installed")
skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")

_MATCH = "is not finite and cannot be represented"


@skip_no_shapely
def test_repro_nan_burn_int_raises_numpy():
    """The issue reproduction raises instead of burning INT_MIN."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), np.nan), (box(5, 5, 10, 10), 7.0)],
                  width=10, height=10, bounds=(0, 0, 10, 10),
                  fill=0, dtype=np.int32)


@skip_no_shapely
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("dt", [np.int16, np.int32, np.uint8, np.int64])
def test_nonfinite_burn_raises_for_int_dtypes(bad, dt):
    """NaN and +/-inf are rejected for every integer dtype."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), bad)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=dt)


@skip_no_shapely
def test_nan_burn_bool_dtype_raises():
    """bool output rejects NaN burns (cast would collapse to True)."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), np.nan)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=False, dtype=np.bool_)


@skip_no_shapely
def test_error_names_offending_value():
    """The message reports the offending value so users can find it."""
    with pytest.raises(ValueError, match=r"\binf\b"):
        rasterize([(box(0, 0, 5, 5), np.inf)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int32)


@skip_no_shapely
def test_float_dtype_nan_burn_unaffected():
    """Float outputs keep accepting NaN burns; NaN is representable."""
    r = rasterize([(box(0, 0, 5, 5), np.nan)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0.0, dtype=np.float64)
    assert np.isnan(r.values).all()


@skip_no_shapely
def test_count_merge_with_nan_props_still_works():
    """merge='count' never burns props, so NaN attributes stay usable."""
    r = rasterize([(box(0, 0, 5, 5), np.nan), (box(0, 0, 5, 5), np.nan)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int32, merge='count')
    assert r.dtype == np.int32
    assert int(r.values[0, 0]) == 2


@skip_no_shapely
def test_finite_burn_int_dtype_unaffected():
    """Ordinary finite values are not touched by the guard."""
    r = rasterize([(box(0, 0, 5, 5), 42)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int32)
    assert int(r.values[0, 0]) == 42


@skip_no_shapely
def test_like_integer_dtype_trips_guard():
    """An integer-dtype ``like`` template trips the same guard."""
    x = np.linspace(0.5, 4.5, 5)
    y = np.linspace(4.5, 0.5, 5)
    like = xr.DataArray(
        np.zeros((5, 5), dtype=np.int32), dims=['y', 'x'],
        coords={'y': y, 'x': x})
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), np.nan)], like=like, fill=0)


@skip_no_shapely
@skip_no_geopandas
def test_geodataframe_column_with_missing_data_trips_guard():
    """The GeoDataFrame ``column`` path (NaN from missing data) raises."""
    gdf = gpd.GeoDataFrame(
        {'zone': [1.0, np.nan]},
        geometry=[box(0, 0, 2, 2), box(3, 3, 5, 5)])
    with pytest.raises(ValueError, match=_MATCH):
        rasterize(gdf, width=4, height=4, bounds=(0, 0, 5, 5),
                  column='zone', fill=0, dtype=np.int32)


@skip_no_shapely
@skip_no_geopandas
def test_geodataframe_columns_path_trips_guard():
    """The multi-column ``columns`` burn path trips the guard."""
    gdf = gpd.GeoDataFrame(
        {'a': [np.nan], 'b': [1.0]}, geometry=[box(0, 0, 5, 5)])
    with pytest.raises(ValueError, match=_MATCH):
        rasterize(gdf, width=4, height=4, bounds=(0, 0, 5, 5),
                  columns=['a', 'b'], fill=0, dtype=np.int64, merge='sum')


@skip_no_shapely
@skip_no_dask
def test_nan_burn_int_raises_dask_numpy():
    """Dask+numpy backend trips the guard before graph construction."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), np.nan)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int32, chunks=2)


@skip_no_shapely
@skip_no_cuda
def test_nan_burn_int_raises_cupy():
    """CuPy backend trips the guard before any device allocation."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), np.nan)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int32, use_cuda=True)


@skip_no_shapely
@skip_no_cuda
@skip_no_dask
def test_nan_burn_int_raises_dask_cupy():
    """Dask+CuPy backend trips the guard before any device allocation."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), np.nan)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int32, use_cuda=True, chunks=2)
