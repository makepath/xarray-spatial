"""Regression coverage for issue #3056.

``rasterize()`` computes in float64 and only casts to the requested
``dtype`` at the end.  ``_parse_input`` also casts every burn value to
float64 up front (props_array is float64, GeoDataFrame columns go through
``.astype(np.float64)``, and ``(geom, value)`` pairs go through
``float()``).  Integers with magnitude above ``2**53 - 1`` are not exactly
representable in float64, so an ID like ``2**53 + 1`` used to silently land
on ``2**53`` -- off by one -- whenever the output dtype was an integer
type.  Zone IDs, parcel IDs, and uint64 identifiers are exactly the values
that exceed this range.

The fix rejects such burn values up front with a ``ValueError`` that names
the offending value, mirroring the NaN-fill-vs-integer-dtype guard.  These
tests pin that behaviour across all four backends, and confirm safe-range
integers and float dtypes are unaffected.
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

# First integer that float64 cannot represent exactly.  float(UNSAFE) ==
# 2**53, so the burned value would be off by one without the guard.
UNSAFE = 2 ** 53 + 1
# Largest integer float64 holds exactly; must pass through untouched.
MAX_SAFE = 2 ** 53 - 1

_MATCH = "exceeds the range of integers that float64 can represent"


@skip_no_shapely
def test_repro_unsafe_int_raises_numpy():
    """The issue reproduction raises instead of returning an off-by-one."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), UNSAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int64)


@skip_no_shapely
@pytest.mark.parametrize("dt", [np.int64, np.uint64])
def test_unsafe_int_raises_for_64bit_dtypes(dt):
    """int64 and uint64 outputs both reject the unsafe value."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), UNSAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=dt)


@skip_no_shapely
def test_negative_unsafe_int_raises():
    """The magnitude check catches large negative values too."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), -UNSAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int64)


@skip_no_shapely
def test_error_names_offending_value():
    """The message reports the offending value so users can find it."""
    with pytest.raises(ValueError, match=r"9007199254740992"):
        rasterize([(box(0, 0, 5, 5), UNSAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int64)


@skip_no_shapely
def test_max_safe_int_burns_exactly_numpy():
    """2**53 - 1 is exactly representable and must round-trip untouched."""
    r = rasterize([(box(0, 0, 5, 5), MAX_SAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int64)
    assert r.dtype == np.int64
    assert int(r.values[0, 0]) == MAX_SAFE


@skip_no_shapely
def test_small_int_unaffected():
    """Ordinary small integer IDs are not touched by the guard."""
    r = rasterize([(box(0, 0, 5, 5), 42)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int32)
    assert int(r.values[0, 0]) == 42


@skip_no_shapely
def test_float_dtype_accepts_unsafe_value():
    """Float outputs are unaffected: the float64 value is what was asked for."""
    r = rasterize([(box(0, 0, 5, 5), float(UNSAFE))],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=np.nan, dtype=np.float64)
    assert r.dtype == np.float64
    # float64 already collapses UNSAFE to 2**53; that is the documented
    # float behaviour, not a rasterize defect, so no error is raised.
    assert r.values[0, 0] == float(UNSAFE)


@skip_no_shapely
def test_like_integer_dtype_trips_guard():
    """An integer-dtype ``like`` template trips the same guard."""
    x = np.linspace(0.5, 4.5, 5)
    y = np.linspace(4.5, 0.5, 5)
    like = xr.DataArray(
        np.zeros((5, 5), dtype=np.int64), dims=['y', 'x'],
        coords={'y': y, 'x': x})
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), UNSAFE)], like=like, fill=0)


@skip_no_shapely
@skip_no_geopandas
def test_geodataframe_column_path_trips_guard():
    """The GeoDataFrame ``column`` burn path trips the guard."""
    gdf = gpd.GeoDataFrame({'zone': [UNSAFE]}, geometry=[box(0, 0, 5, 5)])
    with pytest.raises(ValueError, match=_MATCH):
        rasterize(gdf, width=4, height=4, bounds=(0, 0, 5, 5),
                  column='zone', fill=0, dtype=np.int64)


@skip_no_shapely
@skip_no_geopandas
def test_geodataframe_columns_path_trips_guard():
    """The multi-column ``columns`` burn path trips the guard."""
    gdf = gpd.GeoDataFrame(
        {'a': [UNSAFE], 'b': [1]}, geometry=[box(0, 0, 5, 5)])
    with pytest.raises(ValueError, match=_MATCH):
        rasterize(gdf, width=4, height=4, bounds=(0, 0, 5, 5),
                  columns=['a', 'b'], fill=0, dtype=np.int64, merge='sum')


@skip_no_shapely
@skip_no_dask
def test_unsafe_int_raises_dask_numpy():
    """Dask+numpy backend trips the guard before graph construction."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), UNSAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int64, chunks=2)


@skip_no_shapely
@skip_no_dask
def test_max_safe_int_burns_exactly_dask_numpy():
    """Safe-range value round-trips on the dask+numpy backend too."""
    r = rasterize([(box(0, 0, 5, 5), MAX_SAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int64, chunks=2)
    assert int(r.compute().values[0, 0]) == MAX_SAFE


@skip_no_shapely
@skip_no_cuda
def test_unsafe_int_raises_cupy():
    """CuPy backend trips the guard before any device allocation."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), UNSAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int64, gpu=True)


@skip_no_shapely
@skip_no_cuda
@skip_no_dask
def test_unsafe_int_raises_dask_cupy():
    """Dask+CuPy backend trips the guard before any device allocation."""
    with pytest.raises(ValueError, match=_MATCH):
        rasterize([(box(0, 0, 5, 5), UNSAFE)],
                  width=4, height=4, bounds=(0, 0, 5, 5),
                  fill=0, dtype=np.int64, gpu=True, chunks=2)
