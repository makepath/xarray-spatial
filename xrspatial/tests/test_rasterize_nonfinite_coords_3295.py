"""Regression coverage for issue #3295.

``rasterize()`` used to cast float pixel coordinates to int32 without
checking finiteness or range.  A geometry with a NaN vertex survived the
horizontal-edge filter (``NaN != NaN`` is True) and turned into a phantom
full-height edge; an inf vertex burned pixels left of the polygon's real
x-extent; and finite coordinates more than 2**31 pixels outside the
bounds overflowed the int32 cast, so a polygon entirely off-raster burned
a phantom block on the numpy/cupy backends while the dask backends burned
nothing (the tile bbox filter excluded it), making the backends disagree.

The fix drops geometries containing non-finite coordinates with a warning
(in ``_classify_geometries``, shared by all four backends and the
all_touched path), and clamps pixel-space row values to a narrow band
around the raster before the int32 casts in ``_extract_edges_vectorized``
so off-raster edges are filtered instead of wrapping.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

try:
    from shapely.geometry import (
        GeometryCollection, LineString, Point, Polygon,
    )
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

_WARN_MATCH = "non-finite coordinates"

_KW = dict(width=10, height=10, bounds=(0, 0, 10, 10), fill=0.0)


def _nan_poly():
    return Polygon([(2, 2), (8, 2), (8, float('nan')), (2, 8)])


def _inf_poly():
    return Polygon([(2, 2), (8, 2), (8, float('inf')), (2, 8)])


def _ok_poly():
    return Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])


def _off_raster_overflow_poly():
    # Entirely below the raster, ~3e9 pixels away: far enough that the
    # pixel-space row position overflows int32.
    return Polygon([(2, -2.5e9), (8, -2.5e9), (8, -3e9), (2, -3e9)])


def _values(result):
    data = result.data
    if has_dask and isinstance(data, dask.array.Array):
        data = data.compute()
    if has_cupy and isinstance(data, cupy.ndarray):
        data = data.get()
    return np.asarray(data)


@skip_no_shapely
@pytest.mark.parametrize("bad_geom", [
    pytest.param(lambda: _nan_poly(), id="nan-polygon"),
    pytest.param(lambda: _inf_poly(), id="inf-polygon"),
    pytest.param(lambda: LineString([(1, 1), (float('nan'), 5)]),
                 id="nan-line"),
    pytest.param(lambda: Point(float('nan'), 5), id="nan-point"),
])
def test_nonfinite_geometry_dropped_with_warning(bad_geom):
    """A geometry with a NaN/inf coordinate burns nothing and warns."""
    with pytest.warns(UserWarning, match=_WARN_MATCH):
        result = rasterize([(bad_geom(), 7.0)], **_KW)
    assert np.count_nonzero(_values(result)) == 0


@skip_no_shapely
def test_nan_polygon_all_touched_consistent():
    """all_touched used to burn 13 boundary pixels for a NaN polygon
    while the plain scanline path burned none; both now drop it."""
    with pytest.warns(UserWarning, match=_WARN_MATCH):
        result = rasterize([(_nan_poly(), 7.0)], all_touched=True, **_KW)
    assert np.count_nonzero(_values(result)) == 0


@skip_no_shapely
def test_nonfinite_dropped_but_good_geometries_kept():
    with pytest.warns(UserWarning, match=_WARN_MATCH):
        result = rasterize([(_nan_poly(), 7.0), (_ok_poly(), 1.0)], **_KW)
    v = _values(result)
    assert np.count_nonzero(v == 1.0) == 36
    assert np.count_nonzero(v == 7.0) == 0


@skip_no_shapely
def test_geometrycollection_bad_leaf_dropped_good_leaf_kept():
    """The slow path checks per leaf: one bad member does not take the
    whole collection down."""
    gc = GeometryCollection([_ok_poly(), _nan_poly()])
    with pytest.warns(UserWarning, match=_WARN_MATCH):
        result = rasterize([(gc, 7.0)], **_KW)
    assert np.count_nonzero(_values(result) == 7.0) == 36


@skip_no_shapely
def test_mixed_collection_and_toplevel_nonfinite_warns_once():
    """A GeometryCollection in the input routes everything through the
    slow path; a top-level non-finite geometry must produce exactly one
    warning, not one from each detection path."""
    gc = GeometryCollection([_ok_poly()])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rasterize([(_nan_poly(), 7.0), (gc, 1.0)], **_KW)
    nonfinite_warnings = [
        w for w in caught if _WARN_MATCH in str(w.message)]
    assert len(nonfinite_warnings) == 1
    v = _values(result)
    assert np.count_nonzero(v == 1.0) == 36
    assert np.count_nonzero(v == 7.0) == 0


@skip_no_shapely
def test_int32_overflow_off_raster_polygon_burns_nothing():
    """The issue reproduction: used to burn a 60-pixel phantom block."""
    result = rasterize([(_off_raster_overflow_poly(), 9.0)], **_KW)
    assert np.count_nonzero(_values(result)) == 0


@skip_no_shapely
def test_int32_overflow_partially_on_raster_polygon_exact():
    """A polygon reaching into the raster from ~3e9 pixels below fills
    exactly its in-bounds region."""
    poly = Polygon([(2, 2), (8, 2), (8, -3e9), (2, -3e9)])
    result = rasterize([(poly, 5.0)], **_KW)
    expected = np.zeros((10, 10))
    expected[8:10, 2:8] = 5.0
    np.testing.assert_array_equal(_values(result), expected)


@skip_no_shapely
def test_overflow_coordinates_do_not_warn():
    """Finite far-away coordinates are legal input; only non-finite
    coordinates warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rasterize([(_off_raster_overflow_poly(), 9.0)], **_KW)


@skip_no_shapely
def test_clean_input_no_warning_and_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = rasterize([(_ok_poly(), 7.0)], **_KW)
    assert np.count_nonzero(_values(result) == 7.0) == 36


@skip_no_shapely
@skip_no_dask
def test_dask_numpy_parity_for_overflow_polygon():
    eager = rasterize([(_off_raster_overflow_poly(), 9.0)], **_KW)
    lazy = rasterize([(_off_raster_overflow_poly(), 9.0)], chunks=5, **_KW)
    np.testing.assert_array_equal(_values(eager), _values(lazy))


@skip_no_shapely
@skip_no_dask
def test_dask_nonfinite_dropped_good_kept():
    with pytest.warns(UserWarning, match=_WARN_MATCH):
        result = rasterize(
            [(_nan_poly(), 7.0), (_ok_poly(), 1.0)], chunks=5, **_KW)
    v = _values(result)
    assert np.count_nonzero(v == 1.0) == 36
    assert np.count_nonzero(v == 7.0) == 0


@skip_no_shapely
@skip_no_cuda
@pytest.mark.parametrize("chunks", [
    pytest.param(None, id="cupy"),
    pytest.param(5, id="dask-cupy"),
])
def test_gpu_overflow_polygon_burns_nothing(chunks):
    result = rasterize([(_off_raster_overflow_poly(), 9.0)],
                       gpu=True, chunks=chunks, **_KW)
    assert np.count_nonzero(_values(result)) == 0


@skip_no_shapely
@skip_no_cuda
@pytest.mark.parametrize("chunks", [
    pytest.param(None, id="cupy"),
    pytest.param(5, id="dask-cupy"),
])
def test_gpu_nonfinite_dropped(chunks):
    with pytest.warns(UserWarning, match=_WARN_MATCH):
        result = rasterize([(_nan_poly(), 7.0), (_ok_poly(), 1.0)],
                           gpu=True, chunks=chunks, **_KW)
    v = _values(result)
    assert np.count_nonzero(v == 1.0) == 36
    assert np.count_nonzero(v == 7.0) == 0
