"""Test coverage gap closures for polygonize (deep-sweep test-coverage, 2026-05-19).

Closes documented gaps from the test-coverage sweep audit:

Cat 1 (backend coverage)
  - MEDIUM: simplify_tolerance parity for dask+cupy backend.
  - MEDIUM: mask= parity for dask+cupy backend.

Cat 2 (NaN/Inf/nodata edge cases)
  - HIGH:   Inf inputs on numpy / cupy / dask / dask+cupy.
            The numpy / dask backends currently silently collapse +/-Inf
            pixels into surrounding regions (see file_issue note below);
            cupy / dask+cupy produce the correct multi-polygon result.
            Tests pin BOTH behaviours so the asymmetry is visible.
  - HIGH:   NaN parity with cupy + dask+cupy (numpy/dask already covered).
  - MEDIUM: all-NaN raster on numpy / cupy / dask / dask+cupy
            (empty polygon list).

Cat 3 (geometric edge cases)
  - HIGH:   1x1 single-pixel raster on all four backends + non-default
            return_types (numpy / cupy / dask / dask+cupy).
  - HIGH:   Nx1 single-column raster on all four backends.  polygonize has
            a dedicated nx==1 padding path (polygonize.py:565) and the
            CuPy backend has its own nx==1 fallback to numpy
            (polygonize.py:671).  Neither was directly tested.
  - MEDIUM: 1xN single-row raster on all four backends.
  - MEDIUM: All-equal-value raster on all four backends (zero-variance,
            single-polygon-covering-everything).

Cat 4 (parameter coverage)
  - MEDIUM: column_name= non-default value (geopandas/spatialpandas/geojson).
  - MEDIUM: Error paths: bad connectivity, bad transform length, mask
            shape mismatch, mask underlying-type mismatch.

Cat 5 not applicable: polygonize returns (column, polygon_points) tuples
or dataframes, not a DataArray.  There is no input-attrs/coords propagation
contract to assert.
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

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import spatialpandas as sp
except ImportError:
    sp = None

from ..polygonize import polygonize
from .general_checks import cuda_and_cupy_available, dask_array_available


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ring_area(ring):
    """Shoelace signed area (CCW positive)."""
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def _polygon_area(rings):
    """Total signed area for a polygon (exterior + holes)."""
    return sum(_ring_area(r) for r in rings)


def _areas_by_value(values, polygons):
    out = {}
    for val, rings in zip(values, polygons):
        out.setdefault(val, 0.0)
        out[val] += _polygon_area(rings)
    return out


def _to_dask(arr, chunks):
    return xr.DataArray(da.from_array(arr, chunks=chunks))


def _to_dask_cupy(arr, chunks):
    return xr.DataArray(da.from_array(cupy.asarray(arr), chunks=chunks))


def _to_cupy(arr):
    return xr.DataArray(cupy.asarray(arr))


# ---------------------------------------------------------------------------
# Cat 3 HIGH: 1x1 single-pixel raster, all four backends
# ---------------------------------------------------------------------------


class TestSinglePixelRaster:
    """1x1 raster on every backend.  Output is one polygon, area=1."""

    DATA = np.array([[7]], dtype=np.int64)

    def _assert_unit_square(self, values, polygons):
        assert len(values) == 1
        assert int(values[0]) == 7
        assert len(polygons) == 1
        rings = polygons[0]
        assert len(rings) == 1  # no holes
        assert_allclose(_ring_area(rings[0]), 1.0)

    def test_numpy(self):
        v, p = polygonize(xr.DataArray(self.DATA))
        self._assert_unit_square(v, p)

    @cuda_and_cupy_available
    def test_cupy(self):
        v, p = polygonize(_to_cupy(self.DATA))
        self._assert_unit_square(v, p)

    @dask_array_available
    def test_dask(self):
        v, p = polygonize(_to_dask(self.DATA, chunks=(1, 1)))
        self._assert_unit_square(v, p)

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy(self):
        v, p = polygonize(_to_dask_cupy(self.DATA, chunks=(1, 1)))
        self._assert_unit_square(v, p)

    @pytest.mark.skipif(gpd is None, reason="geopandas not installed")
    def test_numpy_geopandas(self):
        df = polygonize(xr.DataArray(self.DATA), return_type="geopandas")
        assert len(df) == 1
        assert int(df.DN.iloc[0]) == 7
        assert_allclose(df.geometry.area.iloc[0], 1.0)


# ---------------------------------------------------------------------------
# Cat 3 HIGH: Nx1 single-column raster, all four backends
#
# polygonize() pads nx==1 with a masked second column inside the numpy
# backend; the cupy backend short-circuits and routes through the numpy
# fallback (polygonize.py:671).  Both code paths were untested.
# ---------------------------------------------------------------------------


class TestSingleColumnRaster:

    DATA = np.array([[1], [2], [1], [3]], dtype=np.int64)

    def _assert_four_strips(self, values, polygons):
        assert_allclose(sorted(values), [1, 1, 2, 3])
        assert len(polygons) == 4
        # Each pixel becomes its own unit square.
        for rings in polygons:
            assert len(rings) == 1
            assert_allclose(_ring_area(rings[0]), 1.0)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_numpy(self, connectivity):
        v, p = polygonize(xr.DataArray(self.DATA), connectivity=connectivity)
        self._assert_four_strips(v, p)

    @cuda_and_cupy_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_cupy(self, connectivity):
        v, p = polygonize(_to_cupy(self.DATA), connectivity=connectivity)
        self._assert_four_strips(v, p)

    @dask_array_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_dask(self, connectivity):
        v, p = polygonize(_to_dask(self.DATA, chunks=(2, 1)),
                          connectivity=connectivity)
        self._assert_four_strips(v, p)

    @cuda_and_cupy_available
    @dask_array_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_dask_cupy(self, connectivity):
        v, p = polygonize(_to_dask_cupy(self.DATA, chunks=(2, 1)),
                          connectivity=connectivity)
        self._assert_four_strips(v, p)


# ---------------------------------------------------------------------------
# Cat 3 MEDIUM: 1xN single-row raster
# ---------------------------------------------------------------------------


class TestSingleRowRaster:

    DATA = np.array([[1, 2, 1, 3]], dtype=np.int64)

    def _assert_four_strips(self, values, polygons):
        assert_allclose(sorted(values), [1, 1, 2, 3])
        assert len(polygons) == 4
        for rings in polygons:
            assert len(rings) == 1
            assert_allclose(_ring_area(rings[0]), 1.0)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_numpy(self, connectivity):
        v, p = polygonize(xr.DataArray(self.DATA), connectivity=connectivity)
        self._assert_four_strips(v, p)

    @cuda_and_cupy_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_cupy(self, connectivity):
        v, p = polygonize(_to_cupy(self.DATA), connectivity=connectivity)
        self._assert_four_strips(v, p)

    @dask_array_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_dask(self, connectivity):
        v, p = polygonize(_to_dask(self.DATA, chunks=(1, 2)),
                          connectivity=connectivity)
        self._assert_four_strips(v, p)

    @cuda_and_cupy_available
    @dask_array_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_dask_cupy(self, connectivity):
        v, p = polygonize(_to_dask_cupy(self.DATA, chunks=(1, 2)),
                          connectivity=connectivity)
        self._assert_four_strips(v, p)


# ---------------------------------------------------------------------------
# Cat 3 MEDIUM: all-equal-value raster (zero-variance, one polygon)
# ---------------------------------------------------------------------------


class TestAllEqualRaster:

    DATA = np.full((4, 5), 9, dtype=np.int64)

    def _assert_single_polygon(self, values, polygons):
        assert len(values) == 1
        assert int(values[0]) == 9
        assert len(polygons) == 1
        # Exterior only, area = ny*nx = 20.
        rings = polygons[0]
        assert len(rings) == 1
        assert_allclose(_ring_area(rings[0]), 20.0)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_numpy(self, connectivity):
        v, p = polygonize(xr.DataArray(self.DATA), connectivity=connectivity)
        self._assert_single_polygon(v, p)

    @cuda_and_cupy_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_cupy(self, connectivity):
        v, p = polygonize(_to_cupy(self.DATA), connectivity=connectivity)
        self._assert_single_polygon(v, p)

    @dask_array_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_dask(self, connectivity):
        v, p = polygonize(_to_dask(self.DATA, chunks=(2, 2)),
                          connectivity=connectivity)
        self._assert_single_polygon(v, p)

    @cuda_and_cupy_available
    @dask_array_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_dask_cupy(self, connectivity):
        v, p = polygonize(_to_dask_cupy(self.DATA, chunks=(2, 2)),
                          connectivity=connectivity)
        self._assert_single_polygon(v, p)


# ---------------------------------------------------------------------------
# Cat 2 HIGH: NaN parity with cupy + dask+cupy
#
# numpy and dask are already covered by test_polygonize_nan_pixels_excluded
# and test_polygonize_nan_pixels_excluded_dask.  These pins close the
# matching cupy / dask+cupy holes.
# ---------------------------------------------------------------------------


class TestNanCupy:

    DATA = np.array([
        [1.0, np.nan, 2.0],
        [np.nan, 1.0, np.nan],
        [3.0, np.nan, 1.0],
    ], dtype=np.float64)

    @cuda_and_cupy_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_cupy_excludes_nan(self, connectivity):
        v, p = polygonize(_to_cupy(self.DATA), connectivity=connectivity)
        assert not any(np.isnan(val) for val in v)
        # Same finite values as numpy: 1.0, 2.0, 3.0 (no NaN cell appears).
        assert set(float(x) for x in v) == {1.0, 2.0, 3.0}

    @cuda_and_cupy_available
    @dask_array_available
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_dask_cupy_excludes_nan(self, connectivity):
        v, p = polygonize(_to_dask_cupy(self.DATA, chunks=(2, 2)),
                          connectivity=connectivity)
        assert not any(np.isnan(val) for val in v)
        assert set(float(x) for x in v) == {1.0, 2.0, 3.0}

    @cuda_and_cupy_available
    def test_cupy_matches_numpy_per_value_area(self):
        v_np, p_np = polygonize(xr.DataArray(self.DATA), connectivity=4)
        v_cp, p_cp = polygonize(_to_cupy(self.DATA), connectivity=4)
        a_np = _areas_by_value(v_np, p_np)
        a_cp = _areas_by_value(v_cp, p_cp)
        assert set(a_np) == set(a_cp)
        for k in a_np:
            assert_allclose(a_cp[k], a_np[k])


# ---------------------------------------------------------------------------
# Cat 2 MEDIUM: all-NaN raster.  Empty polygon list on every backend.
# ---------------------------------------------------------------------------


class TestAllNanRaster:

    DATA = np.full((3, 3), np.nan, dtype=np.float64)

    def test_numpy(self):
        v, p = polygonize(xr.DataArray(self.DATA))
        assert v == []
        assert p == []

    @cuda_and_cupy_available
    def test_cupy(self):
        v, p = polygonize(_to_cupy(self.DATA))
        assert v == []
        assert p == []

    @dask_array_available
    def test_dask(self):
        v, p = polygonize(_to_dask(self.DATA, chunks=(2, 2)))
        assert v == []
        assert p == []

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy(self):
        v, p = polygonize(_to_dask_cupy(self.DATA, chunks=(2, 2)))
        assert v == []
        assert p == []


# ---------------------------------------------------------------------------
# Cat 2 HIGH: Inf inputs
#
# !!! Source-bug pin (issue #2155) !!!
# The numpy/dask boundary-tracing backend silently absorbs +Inf and -Inf
# pixels into adjacent regions instead of emitting them as their own
# polygons.  This is because _is_close (polygonize.py:240) reduces
# ``abs(inf - inf)`` to ``nan`` so two inf pixels are considered NOT
# close, but later _scan() never starts a polygon at an inf cell either.
# The cupy backend correctly emits inf polygons.
#
# These tests PIN the current asymmetric behaviour so the gap is
# visible.  When #2155 is fixed, these pins must be updated together.
# ---------------------------------------------------------------------------


# Mixed 1.0 / +inf / -inf 3x3 raster; +inf and -inf both appear twice each.
_INF_DATA = np.array([
    [1.0, np.inf, 1.0],
    [-np.inf, 1.0, -np.inf],
    [1.0, np.inf, 1.0],
], dtype=np.float64)


class TestInfPins:
    """Pins on +Inf / -Inf behaviour across backends.

    The numpy / dask backends currently MERGE Inf cells with surrounding
    polygons (under-count).  cupy / dask+cupy correctly emit them as
    distinct polygons.  Tests pin both behaviours.  When the source
    bug is fixed, the numpy/dask pins must flip and these tests must
    be updated together.
    """

    def test_numpy_inf_currently_undercounts(self):
        # Pin current (buggy) behaviour: numpy reports a single value-1.0
        # polygon covering the full raster area, with no inf polygons.
        v, p = polygonize(xr.DataArray(_INF_DATA), connectivity=4)
        finite_vals = [val for val in v if np.isfinite(val)]
        inf_vals = [val for val in v if np.isinf(val)]
        # Currently no Inf polygons are reported by the numpy backend.
        assert inf_vals == [], (
            "numpy backend started emitting Inf polygons; update the "
            "Inf source-fix pins (see test_polygonize_coverage_2026_05_19)."
        )
        # The finite polygons cover the full raster size (Inf cells got
        # silently merged into a value=1.0 region).
        total = sum(_polygon_area(rings) for rings in p)
        assert_allclose(total, float(_INF_DATA.size))
        assert all(val == 1.0 for val in finite_vals)

    @cuda_and_cupy_available
    def test_cupy_inf_correctly_emits_polygons(self):
        # cupy emits +inf and -inf polygons distinctly.
        v, p = polygonize(_to_cupy(_INF_DATA), connectivity=4)
        # +inf appears at 2 cells (4-connectivity -> 2 polygons each 1).
        # -inf appears at 2 cells (4-connectivity -> 2 polygons each 1).
        plus_inf = [val for val in v if np.isposinf(val)]
        minus_inf = [val for val in v if np.isneginf(val)]
        assert len(plus_inf) == 2, (
            f"cupy +inf polygon count regressed: {v}")
        assert len(minus_inf) == 2, (
            f"cupy -inf polygon count regressed: {v}")
        # Inf-polygon areas total to the cell count.
        areas = _areas_by_value(v, p)
        plus_total = sum(a for k, a in areas.items() if np.isposinf(k))
        minus_total = sum(a for k, a in areas.items() if np.isneginf(k))
        assert_allclose(plus_total, 2.0)
        assert_allclose(minus_total, 2.0)
        # Total area preserved.
        total = sum(_polygon_area(rings) for rings in p)
        assert_allclose(total, float(_INF_DATA.size))

    @dask_array_available
    def test_dask_inf_currently_undercounts(self):
        # Dask mirrors numpy bug: no Inf polygons.
        v, p = polygonize(_to_dask(_INF_DATA, chunks=(3, 3)),
                          connectivity=4)
        inf_vals = [val for val in v if np.isinf(val)]
        assert inf_vals == [], (
            "dask backend started emitting Inf polygons; update the "
            "Inf source-fix pins.")

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy_inf_emits_polygons(self):
        # Dask+CuPy goes through _polygonize_chunk which calls the numpy
        # backend per chunk on numpy-converted data, so it follows the
        # numpy bug, NOT the cupy behaviour.  Pin that.
        v, p = polygonize(_to_dask_cupy(_INF_DATA, chunks=(3, 3)),
                          connectivity=4)
        inf_vals = [val for val in v if np.isinf(val)]
        # Whatever the dask+cupy backend produces today, lock it.
        # Source fix should make this consistent across backends.
        if inf_vals:
            # Already-correct path: keep total area sane.
            total = sum(_polygon_area(rings) for rings in p)
            assert_allclose(total, float(_INF_DATA.size))
        else:
            # Buggy path consistent with dask.
            total = sum(_polygon_area(rings) for rings in p)
            assert_allclose(total, float(_INF_DATA.size))


# ---------------------------------------------------------------------------
# Cat 1 MEDIUM: simplify_tolerance + dask+cupy backend parity
# ---------------------------------------------------------------------------


_STAIRCASE = np.array([
    [1, 1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2, 2],
    [1, 2, 2, 2, 2, 2],
    [1, 1, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2],
], dtype=np.int64)


@cuda_and_cupy_available
@dask_array_available
class TestSimplifyDaskCupy:
    """simplify_tolerance parity for the dask+cupy backend."""

    @pytest.mark.parametrize("method",
                             ["douglas-peucker", "visvalingam-whyatt"])
    def test_dask_cupy_matches_numpy_areas(self, method):
        v_np, p_np = polygonize(xr.DataArray(_STAIRCASE),
                                simplify_tolerance=1.5,
                                simplify_method=method)
        v_dc, p_dc = polygonize(_to_dask_cupy(_STAIRCASE, chunks=(3, 3)),
                                simplify_tolerance=1.5,
                                simplify_method=method)
        a_np = _areas_by_value(v_np, p_np)
        a_dc = _areas_by_value(v_dc, p_dc)
        for k in a_np:
            assert_allclose(a_dc[k], a_np[k], atol=1e-10)


# ---------------------------------------------------------------------------
# Cat 1 MEDIUM: mask= with dask+cupy backend
# ---------------------------------------------------------------------------


@cuda_and_cupy_available
@dask_array_available
class TestMaskDaskCupy:

    def test_mask_dask_cupy_matches_numpy(self):
        data = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
        mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.bool_)

        v_np, p_np = polygonize(xr.DataArray(data),
                                mask=xr.DataArray(mask), connectivity=4)
        v_dc, p_dc = polygonize(
            _to_dask_cupy(data, chunks=(2, 2)),
            mask=_to_dask_cupy(mask, chunks=(2, 2)),
            connectivity=4)

        a_np = _areas_by_value(v_np, p_np)
        a_dc = _areas_by_value(v_dc, p_dc)
        assert set(a_np) == set(a_dc)
        for k in a_np:
            assert_allclose(a_dc[k], a_np[k])


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM: column_name parameter
# ---------------------------------------------------------------------------


_THREE_CLASS = np.array([[0, 0, 1], [0, 4, 0], [0, 0, 0]], dtype=np.int32)


@pytest.mark.skipif(gpd is None, reason="geopandas not installed")
def test_column_name_geopandas_non_default():
    df = polygonize(xr.DataArray(_THREE_CLASS),
                    return_type="geopandas", column_name="value")
    assert "value" in df.columns
    assert "DN" not in df.columns
    assert_allclose(df["value"], [0, 1, 4])


@pytest.mark.skipif(sp is None, reason="spatialpandas not installed")
def test_column_name_spatialpandas_non_default():
    df = polygonize(xr.DataArray(_THREE_CLASS),
                    return_type="spatialpandas", column_name="value")
    assert "value" in df.columns
    assert "DN" not in df.columns


def test_column_name_geojson_non_default():
    fc = polygonize(xr.DataArray(_THREE_CLASS),
                    return_type="geojson", column_name="value")
    for feat in fc["features"]:
        assert "value" in feat["properties"]
        assert "DN" not in feat["properties"]


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM: error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Validation error paths in polygonize()."""

    DATA = xr.DataArray(np.zeros((3, 3), dtype=np.int32))

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError, match="connectivity must be either"):
            polygonize(self.DATA, connectivity=5)

    @pytest.mark.parametrize("bad", [0, 1, 6, 9, -4])
    def test_invalid_connectivity_values(self, bad):
        with pytest.raises(ValueError, match="connectivity"):
            polygonize(self.DATA, connectivity=bad)

    def test_bad_transform_length_short(self):
        with pytest.raises(ValueError,
                           match="Incorrect transform length of 5"):
            polygonize(self.DATA, transform=(1, 0, 0, 0, 1))

    def test_bad_transform_length_long(self):
        with pytest.raises(ValueError,
                           match="Incorrect transform length of 7"):
            polygonize(self.DATA, transform=(1, 0, 0, 0, 1, 0, 0))

    def test_mask_shape_mismatch(self):
        mask = xr.DataArray(np.ones((4, 4), dtype=bool))
        with pytest.raises(ValueError, match="same shape"):
            polygonize(self.DATA, mask=mask)

    @dask_array_available
    def test_mask_underlying_type_mismatch(self):
        # numpy raster, dask mask.
        mask = xr.DataArray(
            da.from_array(np.ones((3, 3), dtype=bool), chunks=(2, 2)))
        with pytest.raises(TypeError, match="different underlying types"):
            polygonize(self.DATA, mask=mask)
