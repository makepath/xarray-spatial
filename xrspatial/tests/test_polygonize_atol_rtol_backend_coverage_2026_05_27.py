"""Cross-backend parity tests for polygonize() atol / rtol kwargs (#2537).

Closes a Cat 4 MEDIUM parameter-coverage gap surfaced by the test-coverage
sweep on 2026-05-27.  The atol / rtol kwargs route through every backend:

  * numpy   : _polygonize_numpy(values, mask, conn8, transform, atol, rtol)
  * cupy    : _polygonize_cupy(...) forwards atol/rtol to _polygonize_numpy
              on float dtypes (polygonize.py:823-825 and 829-830) and
              documents them as no-ops on the integer GPU CCL path
  * dask    : _polygonize_dask(...) plumbs atol/rtol through
              dask.delayed(_polygonize_chunk)(..., atol, rtol) and into
              _group_boundary_polygons(..., atol, rtol) (replaced the
              legacy _bucket_key_for_value bucket in #2583) for the
              spatial-topology + value-closeness cross-chunk merge
  * dask+cupy: same _polygonize_dask path -- cupy chunks are converted to
              numpy inside _polygonize_chunk via _to_numpy(block) at
              polygonize.py:859-861, so atol/rtol again reach
              _polygonize_numpy unchanged

Existing tests pin every behaviour on numpy and dask+numpy.  The cupy and
dask+cupy backends were untested for non-default atol / rtol; a regression
in those dispatchers that silently dropped the kwargs would change the
float polygon count and was not caught by any pre-existing test.

This file pins three behaviours per GPU backend so a future dispatcher
refactor that loses the kwargs surfaces immediately:

  1. atol=0, rtol=0 on a float raster recovers strict equality and the
     polygon count matches the numpy strict-equality result.
  2. An intermediate atol value lying between two known float steps
     picks the same split as the numpy reference does.
  3. Integer rasters ignore atol / rtol entirely (matches the existing
     test_polygonize_integer_default_unchanged_by_atol and
     test_polygonize_dask_integer_atol_ignored numpy / dask pins).
"""
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

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


# Three distinct float values where each consecutive pair sits within the
# default atol/rtol but the outer pair does not.  Reused from the existing
# #2173 tests so the numpy reference results are well-documented.
_REPRO_2173 = np.array([[1.0, 1.000009, 1.000018]], dtype=np.float64)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ring_area(ring):
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def _areas_by_value(values, polygons, atol=1e-12, rtol=0.0):
    """Bucket polygon areas by value, merging keys that compare close.

    Used for cross-backend comparisons where dask may report 1.000009 as
    the representative for a bucket that numpy reported as 1.0 (and vice
    versa).
    """
    out = {}
    for val, rings in zip(values, polygons):
        area = sum(_ring_area(r) for r in rings)
        # Match against an existing close key, otherwise create a new one.
        matched = None
        for k in out:
            if abs(k - val) <= atol + rtol * abs(k):
                matched = k
                break
        if matched is None:
            out[val] = area
        else:
            out[matched] += area
    return out


def _to_cupy_array(arr):
    return xr.DataArray(cupy.asarray(arr))


def _to_dask_cupy_array(arr, chunks):
    return xr.DataArray(da.from_array(cupy.asarray(arr), chunks=chunks))


def _to_dask_array(arr, chunks):
    return xr.DataArray(da.from_array(arr, chunks=chunks))


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM: atol=rtol=0 strict equality on cupy / dask+cupy float
# ---------------------------------------------------------------------------


@cuda_and_cupy_available
class TestStrictFloatEqualityCuPy:
    """atol=0, rtol=0 must reach _polygonize_numpy through the cupy path.

    The cupy backend forwards atol/rtol to _polygonize_numpy on float
    dtypes (polygonize.py:823-825).  A dispatcher regression that dropped
    the kwargs would re-introduce the default tolerance (atol=1e-8,
    rtol=1e-5) which merges all three pixels of _REPRO_2173 into a single
    region.  Strict equality must produce three distinct polygons matching
    the numpy reference.
    """

    def test_three_distinct_values(self):
        # numpy reference: three polygons under strict equality.
        v_np, p_np = polygonize(
            xr.DataArray(_REPRO_2173), atol=0.0, rtol=0.0)
        assert len(v_np) == 3

        v_cp, p_cp = polygonize(
            _to_cupy_array(_REPRO_2173), atol=0.0, rtol=0.0)

        assert len(v_cp) == 3, (
            f"cupy strict-equality polygon count mismatch: "
            f"numpy={len(v_np)} cupy={len(v_cp)}; cupy values={v_cp}"
        )
        assert_allclose(sorted(v_cp), sorted(_REPRO_2173[0]))

        a_np = _areas_by_value(v_np, p_np)
        a_cp = _areas_by_value(v_cp, p_cp)
        assert set(a_np) == set(a_cp)
        for k in a_np:
            assert_allclose(a_cp[k], a_np[k], atol=1e-10)

    def test_default_tolerance_still_merges(self):
        """Sanity pin: with defaults the cupy path still merges to one.

        Without this companion test, a regression where the cupy path
        always passed atol=0,rtol=0 would silently pass the strict test
        above and break backward compatibility on the default path.
        """
        v_cp, p_cp = polygonize(_to_cupy_array(_REPRO_2173))
        assert len(v_cp) == 1
        assert_allclose(v_cp[0], 1.0)


@cuda_and_cupy_available
@dask_array_available
class TestStrictFloatEqualityDaskCuPy:
    """atol=0, rtol=0 must reach _polygonize_numpy through dask+cupy.

    The dask+cupy path goes through _polygonize_dask -> _polygonize_chunk
    -> _polygonize_numpy.  atol/rtol thread through dask.delayed and the
    cross-chunk merge runs _group_boundary_polygons(..., atol, rtol)
    (replaced the legacy _bucket_key_for_value bucket in #2583).  Pin
    both single-chunk (no merge) and multi-chunk (merge engaged) cases.
    """

    def test_single_chunk(self):
        v_np, p_np = polygonize(
            xr.DataArray(_REPRO_2173), atol=0.0, rtol=0.0)

        v_dc, p_dc = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=_REPRO_2173.shape),
            atol=0.0, rtol=0.0)

        assert len(v_dc) == 3, (
            f"dask+cupy single-chunk strict-equality polygon count "
            f"mismatch: numpy={len(v_np)} dask+cupy={len(v_dc)}; "
            f"dask+cupy values={v_dc}"
        )
        assert_allclose(sorted(v_dc), sorted(_REPRO_2173[0]))

    def test_multi_chunk_merge(self):
        """Each pixel its own chunk -- the cross-chunk merge bucket
        must see atol=rtol=0 too, otherwise the default tolerance would
        re-merge the near-equal pixels at the stitching step."""
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_2173), atol=0.0, rtol=0.0)

        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=(1, 1)),
            atol=0.0, rtol=0.0)

        assert len(v_dc) == 3, (
            f"dask+cupy multi-chunk strict-equality polygon count "
            f"mismatch: numpy={len(v_np)} dask+cupy={len(v_dc)}; "
            f"dask+cupy values={v_dc}"
        )
        assert_allclose(sorted(v_dc), sorted(_REPRO_2173[0]))

    def test_default_tolerance_still_merges_multi_chunk(self):
        """Multi-chunk dask+cupy parity with single-chunk numpy under the
        default tolerance (#2583).

        Numpy CCL chains 1.0 -> 1.000009 -> 1.000018 within a single
        chunk because the middle pixel is within tolerance of both ends.
        After #2583, the dask cross-chunk merge groups boundary polygons
        by spatial-topology + value-closeness union-find, so the same
        transitive chain applies across chunk boundaries: all three
        single-pixel chunks collapse to one region with one DN value,
        matching numpy.
        """
        v_np, _ = polygonize(xr.DataArray(_REPRO_2173))
        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=(1, 1)))
        assert len(v_dc) == len(v_np) == 1
        assert_allclose(sorted(v_dc), sorted(v_np))


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM: intermediate atol picks predictable splits on GPU backends
# ---------------------------------------------------------------------------


@cuda_and_cupy_available
class TestIntermediateAtolCuPy:
    """atol values between the float steps must pick the same split as numpy.

    The repro has steps of 9e-6 between consecutive pixels.  An atol of
    1e-6 (rtol=0) is smaller than the step so every pixel is distinct;
    an atol of 1e-4 covers the step and merges everything to one region.
    """

    def test_small_atol_three_polygons(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_2173), atol=1e-6, rtol=0.0)
        v_cp, _ = polygonize(
            _to_cupy_array(_REPRO_2173), atol=1e-6, rtol=0.0)
        assert len(v_cp) == len(v_np) == 3

    def test_large_atol_one_polygon(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_2173), atol=1e-4, rtol=0.0)
        v_cp, _ = polygonize(
            _to_cupy_array(_REPRO_2173), atol=1e-4, rtol=0.0)
        assert len(v_cp) == len(v_np) == 1


@cuda_and_cupy_available
@dask_array_available
class TestIntermediateAtolDaskCuPy:
    """Same intermediate-atol invariant on dask+cupy."""

    def test_small_atol_three_polygons_single_chunk(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_2173), atol=1e-6, rtol=0.0)
        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=_REPRO_2173.shape),
            atol=1e-6, rtol=0.0)
        assert len(v_dc) == len(v_np) == 3

    def test_small_atol_three_polygons_multi_chunk(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_2173), atol=1e-6, rtol=0.0)
        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=(1, 1)),
            atol=1e-6, rtol=0.0)
        assert len(v_dc) == len(v_np) == 3

    def test_large_atol_one_polygon_single_chunk(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_2173), atol=1e-4, rtol=0.0)
        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=_REPRO_2173.shape),
            atol=1e-4, rtol=0.0)
        assert len(v_dc) == len(v_np) == 1

    def test_large_atol_one_polygon_multi_chunk(self):
        """Large atol on a multi-chunk dask+cupy raster exercises the
        cross-chunk merge (_group_boundary_polygons in polygonize.py,
        formerly _bucket_key_for_value before #2583) with non-default
        atol.  A regression that hard-coded the default atol inside the
        merge would split the result into multiple polygons here.
        """
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_2173), atol=1e-4, rtol=0.0)
        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=(1, 1)),
            atol=1e-4, rtol=0.0)
        assert len(v_dc) == len(v_np) == 1


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM: integer rasters ignore atol/rtol on cupy and dask+cupy
# ---------------------------------------------------------------------------


@cuda_and_cupy_available
class TestIntegerAtolIgnoredCuPy:
    """Integer cupy rasters always use strict equality.

    Mirrors test_polygonize_integer_default_unchanged_by_atol on the
    numpy backend.  The integer cupy path uses _calculate_regions_cupy
    (per-value GPU CCL) which is strict by construction; atol/rtol are
    documented no-ops there.
    """

    def test_huge_atol_keeps_integers_distinct(self):
        data = np.array([[1, 2, 3]], dtype=np.int64)
        v_cp, p_cp = polygonize(
            _to_cupy_array(data), atol=10.0, rtol=10.0)
        assert sorted(v_cp) == [1, 2, 3]
        total_area = sum(_ring_area(r) for rings in p_cp for r in rings)
        assert_allclose(total_area, 3.0)


@cuda_and_cupy_available
@dask_array_available
class TestIntegerAtolIgnoredDaskCuPy:
    """Integer dask+cupy rasters always use strict equality.

    Mirrors test_polygonize_dask_integer_atol_ignored on the dask+numpy
    backend.
    """

    def test_huge_atol_keeps_integers_distinct_single_chunk(self):
        data = np.array([[1, 2, 3]], dtype=np.int64)
        v_dc, p_dc = polygonize(
            _to_dask_cupy_array(data, chunks=data.shape),
            atol=10.0, rtol=10.0)
        assert sorted(v_dc) == [1, 2, 3]
        total_area = sum(_ring_area(r) for rings in p_dc for r in rings)
        assert_allclose(total_area, 3.0)

    def test_huge_atol_keeps_integers_distinct_multi_chunk(self):
        data = np.array([[1, 2, 3]], dtype=np.int64)
        v_dc, _ = polygonize(
            _to_dask_cupy_array(data, chunks=(1, 1)),
            atol=10.0, rtol=10.0)
        assert sorted(v_dc) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Eager-cupy non-default rtol coverage (rtol-only variant, atol=0)
# ---------------------------------------------------------------------------


@cuda_and_cupy_available
class TestRtolCuPy:
    """Non-default rtol on float cupy raster must reach _polygonize_numpy.

    atol=0 isolates the rtol branch.  The relative-tolerance threshold
    around 1.0 with rtol=1e-3 is 1e-3, which comfortably covers the
    18e-6 outer gap in the repro -> single polygon.
    """

    def test_large_rtol_merges_all(self):
        v_cp, _ = polygonize(
            _to_cupy_array(_REPRO_2173), atol=0.0, rtol=1e-3)
        assert len(v_cp) == 1

    def test_small_rtol_keeps_distinct(self):
        # rtol=1e-7 with atol=0 -> threshold ~1e-7 around value 1.0,
        # smaller than the 9e-6 step -> three polygons.
        v_cp, _ = polygonize(
            _to_cupy_array(_REPRO_2173), atol=0.0, rtol=1e-7)
        assert len(v_cp) == 3


# ---------------------------------------------------------------------------
# dask+cupy non-default rtol coverage (rtol-only variant, atol=0)
# ---------------------------------------------------------------------------


@cuda_and_cupy_available
@dask_array_available
class TestRtolDaskCuPy:
    """Non-default rtol on float dask+cupy raster must reach
    _polygonize_numpy through _polygonize_chunk and
    _group_boundary_polygons (replaced the legacy _bucket_key_for_value
    in #2583).

    Mirrors TestRtolCuPy on the dask+cupy backend.  Multi-chunk variants
    additionally exercise the cross-chunk merge with rtol>0.
    """

    def test_large_rtol_merges_all_single_chunk(self):
        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=_REPRO_2173.shape),
            atol=0.0, rtol=1e-3)
        assert len(v_dc) == 1

    def test_large_rtol_merges_all_multi_chunk(self):
        """Per-pixel chunks force the merge bucket to honour rtol=1e-3."""
        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=(1, 1)),
            atol=0.0, rtol=1e-3)
        assert len(v_dc) == 1

    def test_small_rtol_keeps_distinct_multi_chunk(self):
        v_dc, _ = polygonize(
            _to_dask_cupy_array(_REPRO_2173, chunks=(1, 1)),
            atol=0.0, rtol=1e-7)
        assert len(v_dc) == 3
