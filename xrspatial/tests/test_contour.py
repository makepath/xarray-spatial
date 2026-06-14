import warnings
import numpy as np
import pytest
import xarray as xr

from collections import defaultdict

from xrspatial.contour import contours, _contours_numpy, _stitch_segments
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ramp(ny=5, nx=6):
    """Simple left-to-right linear ramp: values 0..nx-1 repeated down rows."""
    return np.tile(np.arange(nx, dtype=np.float64), (ny, 1))


def _make_peak():
    """Small 5x5 raster with a peak in the center."""
    data = np.array([
        [0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0.],
        [0., 1., 2., 1., 0.],
        [0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0.],
    ], dtype=np.float64)
    return data


def _segments_by_level(results, decimals=8):
    """Decompose contour polylines into canonicalized segments per level.

    Each segment is stored with its smaller endpoint first so the result is
    direction-independent, and segments are sorted for stable comparison
    across backends.
    """
    by_level = defaultdict(list)
    for level, coords in results:
        for i in range(len(coords) - 1):
            p0 = (round(coords[i, 0], decimals), round(coords[i, 1], decimals))
            p1 = (round(coords[i + 1, 0], decimals),
                  round(coords[i + 1, 1], decimals))
            by_level[level].append((min(p0, p1), max(p0, p1)))
    return {lvl: sorted(segs) for lvl, segs in by_level.items()}


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestMarchingSquaresBasic:

    def test_no_contours_flat(self):
        """A flat raster produces no contour lines at any level outside it."""
        data = np.ones((4, 4), dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5, 1.5])
        assert isinstance(result, list)
        # Level 0.5 is below entire surface, 1.5 is above -> no crossings.
        assert len(result) == 0

    def test_single_level_ramp(self):
        """A horizontal ramp should produce vertical contour lines."""
        data = _make_ramp(ny=5, nx=6)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[2.5])
        assert len(result) > 0
        # With res=0.5 x_coords = [0, 0.5, 1.0, 1.5, 2.0, 2.5].
        # The crossing between col indices 2 and 3 maps to x = 1.25.
        expected_x = 1.25
        for level, coords in result:
            assert level == 2.5
            np.testing.assert_allclose(coords[:, 1], expected_x, atol=1e-10)

    def test_multiple_levels_ramp(self):
        """Multiple levels on a ramp produce one line per level."""
        data = _make_ramp(ny=5, nx=6)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.5, 2.5, 3.5])
        levels_found = sorted(set(lvl for lvl, _ in result))
        assert levels_found == [1.5, 2.5, 3.5]

    def test_peak_contour_is_closed(self):
        """A contour fully inside the raster should form a closed ring."""
        # Use a larger peak so the 0.5 contour doesn't hit the boundary.
        data = np.array([
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 1., 1., 0., 0.],
            [0., 0., 1., 2., 1., 0., 0.],
            [0., 0., 1., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5])
        assert len(result) >= 1
        for level, coords in result:
            # Closed ring: first point equals last point.
            np.testing.assert_allclose(coords[0], coords[-1], atol=1e-10)

    def test_peak_contour_at_1_5(self):
        """Contour at level 1.5 around a 2-valued peak."""
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.5])
        assert len(result) >= 1
        y_min = float(agg.coords[agg.dims[0]].values.min())
        y_max = float(agg.coords[agg.dims[0]].values.max())
        x_min = float(agg.coords[agg.dims[1]].values.min())
        x_max = float(agg.coords[agg.dims[1]].values.max())
        for level, coords in result:
            assert level == 1.5
            # All points should be inside the raster coordinate bounds.
            assert np.all(coords[:, 0] >= y_min - 1e-10)
            assert np.all(coords[:, 0] <= y_max + 1e-10)
            assert np.all(coords[:, 1] >= x_min - 1e-10)
            assert np.all(coords[:, 1] <= x_max + 1e-10)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:

    def test_all_nan(self):
        """All-NaN raster produces no contours."""
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.0])
        assert result == []

    def test_all_nan_auto_levels_no_warning(self):
        """All-NaN raster with auto levels must not emit RuntimeWarning (#2795)."""
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = contours(agg)
            assert result == []
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0, (
                f"RuntimeWarning emitted on all-NaN auto-level path: "
                f"{[str(x.message) for x in runtime_warnings]}"
            )

    @dask_array_available
    def test_all_nan_auto_levels_no_warning_dask(self):
        """All-NaN dask raster with auto levels must not emit RuntimeWarning (#2795)."""
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='dask+numpy', chunks=(2, 2))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = contours(agg)
            assert result == []
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0, (
                f"RuntimeWarning emitted on all-NaN dask auto-level path: "
                f"{[str(x.message) for x in runtime_warnings]}"
            )

    @cuda_and_cupy_available
    def test_all_nan_auto_levels_no_warning_cupy(self):
        """All-NaN cupy raster with auto levels must not emit RuntimeWarning (#2795)."""
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='cupy')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = contours(agg)
            assert result == []
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0, (
                f"RuntimeWarning emitted on all-NaN cupy auto-level path: "
                f"{[str(x.message) for x in runtime_warnings]}"
            )

    @dask_array_available
    @cuda_and_cupy_available
    def test_all_nan_auto_levels_no_warning_dask_cupy(self):
        """All-NaN dask+cupy raster with auto levels must not emit RuntimeWarning (#2795)."""
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='dask+cupy', chunks=(2, 2))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = contours(agg)
            assert result == []
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0, (
                f"RuntimeWarning emitted on all-NaN dask+cupy auto-level path: "
                f"{[str(x.message) for x in runtime_warnings]}"
            )

    def test_partial_nan(self):
        """Contours skip quads with NaN corners."""
        data = _make_ramp(ny=5, nx=6)
        data[0, :] = np.nan  # top row is NaN
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[2.5])
        assert len(result) > 0
        # y_coords = [2.0, 1.5, 1.0, 0.5, 0.0] (decreasing with res=0.5).
        # NaN row is row 0 (y=2.0).  All contour points must stay at y <= 1.5.
        nan_row_y = agg.coords[agg.dims[0]].values[0]
        for level, coords in result:
            assert np.all(coords[:, 0] < nan_row_y + 1e-10)


# ---------------------------------------------------------------------------
# Non-finite (inf) handling in automatic level generation (issue #2797)
# ---------------------------------------------------------------------------

def _make_ramp_with_inf():
    """Left-to-right ramp with one +inf and one -inf cell in the corner."""
    data = _make_ramp(ny=5, nx=10)
    data[0, 0] = np.inf
    data[1, 0] = -np.inf
    return data


class TestAutoLevelsInf:

    def test_auto_levels_ignore_inf(self):
        """+/-inf must not poison auto-generated levels (#2797)."""
        data = _make_ramp_with_inf()
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, n_levels=5)
        # The finite ramp spans 0..9, so auto-levels still produce contours.
        assert len(result) > 0
        for level, _ in result:
            assert np.isfinite(level)

    def test_auto_levels_match_finite_range(self):
        """Levels come from the finite min/max, not the inf extremes."""
        data = _make_ramp_with_inf()
        finite = _make_ramp(ny=5, nx=10)
        agg = create_test_raster(data, backend='numpy')
        finite_agg = create_test_raster(finite, backend='numpy')
        inf_levels = sorted({lvl for lvl, _ in contours(agg, n_levels=5)})
        finite_levels = sorted(
            {lvl for lvl, _ in contours(finite_agg, n_levels=5)}
        )
        assert inf_levels == finite_levels

    def test_all_inf_returns_empty(self):
        """An entirely non-finite raster yields no contours, no crash."""
        data = np.full((4, 4), np.inf, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg)
        assert result == []

    def test_explicit_levels_unaffected_by_inf(self):
        """Explicit levels bypass the range computation entirely."""
        data = _make_ramp_with_inf()
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[4.5])
        assert len(result) > 0

    @dask_array_available
    def test_auto_levels_ignore_inf_dask(self):
        """Dask backend ignores inf in the lazy nanmin/nanmax path."""
        data = _make_ramp_with_inf()
        np_agg = create_test_raster(data, backend='numpy')
        dask_agg = create_test_raster(
            data, backend='dask+numpy', chunks=(3, 4)
        )
        np_levels = sorted({lvl for lvl, _ in contours(np_agg, n_levels=5)})
        dk_levels = sorted({lvl for lvl, _ in contours(dask_agg, n_levels=5)})
        assert len(dk_levels) > 0
        assert np_levels == dk_levels

    @cuda_and_cupy_available
    def test_auto_levels_ignore_inf_cupy(self):
        """CuPy backend ignores inf in the nanmin/nanmax path."""
        data = _make_ramp_with_inf()
        np_agg = create_test_raster(data, backend='numpy')
        cupy_agg = create_test_raster(data, backend='cupy')
        np_levels = sorted({lvl for lvl, _ in contours(np_agg, n_levels=5)})
        cp_levels = sorted({lvl for lvl, _ in contours(cupy_agg, n_levels=5)})
        assert len(cp_levels) > 0
        assert np_levels == cp_levels

    @dask_array_available
    @cuda_and_cupy_available
    def test_auto_levels_ignore_inf_dask_cupy(self):
        """Dask+CuPy backend ignores inf in the lazy nanmin/nanmax path."""
        data = _make_ramp_with_inf()
        np_agg = create_test_raster(data, backend='numpy')
        dc_agg = create_test_raster(
            data, backend='dask+cupy', chunks=(3, 4)
        )
        np_levels = sorted({lvl for lvl, _ in contours(np_agg, n_levels=5)})
        dc_levels = sorted({lvl for lvl, _ in contours(dc_agg, n_levels=5)})
        assert len(dc_levels) > 0
        assert np_levels == dc_levels


# ---------------------------------------------------------------------------
# n_levels validation contract (issue #2895)
# ---------------------------------------------------------------------------

class TestNLevelsValidation:

    def test_n_levels_zero_raises(self):
        """n_levels=0 must raise instead of silently returning nothing."""
        agg = create_test_raster(_make_ramp(), backend='numpy')
        with pytest.raises(ValueError, match="n_levels must be >= 1"):
            contours(agg, n_levels=0)

    def test_n_levels_negative_raises(self):
        """n_levels=-1 must raise a clear out-of-range error."""
        agg = create_test_raster(_make_ramp(), backend='numpy')
        with pytest.raises(ValueError, match="n_levels must be >= 1"):
            contours(agg, n_levels=-1)

    def test_n_levels_float_raises_clear_typeerror(self):
        """A non-integer n_levels raises a clear TypeError naming the
        parameter, not a raw numpy 'cannot be interpreted as an integer'."""
        agg = create_test_raster(_make_ramp(), backend='numpy')
        with pytest.raises(TypeError, match="n_levels must be an integer"):
            contours(agg, n_levels=2.5)

    def test_n_levels_bool_raises(self):
        """bool is an int subclass but is not a valid level count."""
        agg = create_test_raster(_make_ramp(), backend='numpy')
        with pytest.raises(TypeError, match="n_levels must be an integer"):
            contours(agg, n_levels=True)

    def test_n_levels_one_produces_single_level(self):
        """n_levels=1 is valid and yields exactly one contour level."""
        agg = create_test_raster(_make_ramp(ny=5, nx=10), backend='numpy')
        result = contours(agg, n_levels=1)
        levels = sorted({lvl for lvl, _ in result})
        assert len(levels) == 1

    def test_explicit_levels_skip_n_levels_validation(self):
        """When explicit levels are given, n_levels is unused and an
        otherwise-invalid value must not be rejected."""
        agg = create_test_raster(_make_ramp(ny=5, nx=6), backend='numpy')
        # n_levels=0 would raise on the auto branch; with explicit levels
        # it is ignored and the call must succeed.
        result = contours(agg, levels=[2.5], n_levels=0)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_minimum_raster(self):
        """2x2 raster (one quad) with a crossing."""
        data = np.array([[0., 1.], [1., 2.]], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5])
        assert len(result) >= 1

    def test_level_at_exact_value(self):
        """Level exactly matching cell values still works."""
        data = np.array([
            [0., 1., 2.],
            [1., 2., 3.],
            [2., 3., 4.],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[2.0])
        # Should not crash; some segments may be degenerate.
        assert isinstance(result, list)

    def test_auto_levels(self):
        """When levels is None, n_levels evenly spaced levels are chosen."""
        data = _make_ramp(ny=5, nx=10)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, n_levels=5)
        assert len(result) > 0
        levels_found = sorted(set(lvl for lvl, _ in result))
        assert len(levels_found) <= 5

    def test_invalid_ndim(self):
        """3D input raises ValueError."""
        data = np.ones((3, 3, 3), dtype=np.float64)
        agg = xr.DataArray(data, dims=['z', 'y', 'x'])
        with pytest.raises(ValueError, match="2D"):
            contours(agg, levels=[0.5])

    def test_too_small(self):
        """1-row raster raises ValueError."""
        data = np.ones((1, 5), dtype=np.float64)
        agg = xr.DataArray(data, dims=['y', 'x'])
        with pytest.raises(ValueError, match="at least 2"):
            contours(agg, levels=[0.5])

    def test_complex_dtype_rejected(self):
        """Complex input raises instead of silently dropping the imaginary part."""
        data = np.ones((3, 3), dtype=np.complex128)
        agg = xr.DataArray(data, dims=['y', 'x'])
        with pytest.raises(ValueError, match="real numeric"):
            contours(agg, levels=[0.5])

    def test_non_dataarray_rejected(self):
        """A plain ndarray raises a clear TypeError, not a late dispatch error."""
        data = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(TypeError, match="xarray.DataArray"):
            contours(data, levels=[0.5])

    def test_invalid_return_type_rejected(self):
        """A bad return_type raises ValueError for a normal raster."""
        data = _make_ramp(ny=5, nx=6)
        agg = create_test_raster(data, backend='numpy')
        with pytest.raises(ValueError, match="Invalid return_type"):
            contours(agg, levels=[2.5], return_type="bad")

    def test_invalid_return_type_rejected_all_nan(self):
        """A bad return_type raises even on the all-non-finite path.

        Previously the all-non-finite early return handed back an empty
        GeoDataFrame for any non-'numpy' value instead of raising.
        """
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        with pytest.raises(ValueError, match="Invalid return_type"):
            contours(agg, return_type="bad")

    def test_invalid_return_type_checked_before_data_work(self):
        """return_type is validated before level computation / extraction.

        A degenerate all-NaN raster (which would otherwise take the
        early-return path) still raises on the bad argument.
        """
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        with pytest.raises(ValueError, match="Invalid return_type"):
            contours(agg, levels=[0.5], return_type="bad")

    @dask_array_available
    def test_invalid_return_type_no_dask_compute(self, monkeypatch):
        """An invalid return_type on a Dask input must raise before any
        compute, nanmin/nanmax, or backend dispatch (#2788).
        """
        import dask
        import dask.array as da

        data = _make_ramp(ny=5, nx=6)
        agg = create_test_raster(data, backend='dask+numpy', chunks=(3, 3))

        # Wrap dask compute to detect if it is ever called.
        compute_called = False
        _original_compute = dask.compute

        def spy_compute(*args, **kwargs):
            nonlocal compute_called
            compute_called = True
            return _original_compute(*args, **kwargs)

        monkeypatch.setattr(dask, 'compute', spy_compute)

        with pytest.raises(ValueError, match="Invalid return_type"):
            contours(agg, levels=[2.5], return_type="bogus")

        assert not compute_called, (
            "dask.compute was called before the invalid return_type raise"
        )

    def test_valid_return_types_accepted(self):
        """The two valid return_type values still work."""
        data = _make_ramp(ny=5, nx=6)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[2.5], return_type="numpy")
        assert isinstance(result, list)

        gpd = pytest.importorskip("geopandas")
        gdf = contours(agg, levels=[2.5], return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)


# ---------------------------------------------------------------------------
# Segment stitching
# ---------------------------------------------------------------------------

class TestStitching:

    def test_empty_segments(self):
        """No segments produces no lines."""
        seg_r = np.empty((0, 2), dtype=np.float64)
        seg_c = np.empty((0, 2), dtype=np.float64)
        lines = _stitch_segments(seg_r, seg_c, 0)
        assert lines == []

    def test_single_segment(self):
        """One segment produces one line with two points."""
        seg_r = np.array([[0.0, 1.0]], dtype=np.float64)
        seg_c = np.array([[0.5, 0.5]], dtype=np.float64)
        lines = _stitch_segments(seg_r, seg_c, 1)
        assert len(lines) == 1
        assert lines[0].shape == (2, 2)

    def test_connected_segments(self):
        """Three connected segments produce one line with four points."""
        seg_r = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
        ], dtype=np.float64)
        seg_c = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float64)
        lines = _stitch_segments(seg_r, seg_c, 3)
        assert len(lines) == 1
        assert lines[0].shape[0] == 4


# ---------------------------------------------------------------------------
# Backend equivalence
# ---------------------------------------------------------------------------

class TestBackendEquivalence:

    def _numpy_result(self, data, levels):
        agg = create_test_raster(data, backend='numpy')
        return contours(agg, levels=levels)

    def _collect_segments(self, results):
        """Collect all individual segments as a sorted set for comparison.

        Each polyline is decomposed into its constituent segments, each
        segment is canonicalized (smaller endpoint first), and the full
        set is sorted for stable comparison across backends.
        """
        from collections import defaultdict
        by_level = defaultdict(list)
        DECIMALS = 8
        for level, coords in results:
            for i in range(len(coords) - 1):
                p0 = (round(coords[i, 0], DECIMALS),
                      round(coords[i, 1], DECIMALS))
                p1 = (round(coords[i + 1, 0], DECIMALS),
                      round(coords[i + 1, 1], DECIMALS))
                seg = (min(p0, p1), max(p0, p1))
                by_level[level].append(seg)
        # Sort segments within each level for stable comparison.
        return {lvl: sorted(segs) for lvl, segs in by_level.items()}

    @dask_array_available
    def test_numpy_equals_dask(self, elevation_raster_no_nans):
        data = elevation_raster_no_nans
        levels = [300.0, 500.0, 700.0]
        numpy_agg = create_test_raster(data, backend='numpy')
        dask_agg = create_test_raster(data, backend='dask+numpy', chunks=(4, 3))

        np_result = contours(numpy_agg, levels=levels)
        dk_result = contours(dask_agg, levels=levels)

        np_segs = self._collect_segments(np_result)
        dk_segs = self._collect_segments(dk_result)

        assert set(np_segs.keys()) == set(dk_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == dk_segs[lvl], (
                f"Segment mismatch at level {lvl}")

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self, elevation_raster_no_nans):
        data = elevation_raster_no_nans
        levels = [300.0, 500.0, 700.0]
        numpy_agg = create_test_raster(data, backend='numpy')
        cupy_agg = create_test_raster(data, backend='cupy')

        np_result = contours(numpy_agg, levels=levels)
        cp_result = contours(cupy_agg, levels=levels)

        np_segs = self._collect_segments(np_result)
        cp_segs = self._collect_segments(cp_result)

        assert set(np_segs.keys()) == set(cp_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == cp_segs[lvl], (
                f"Segment mismatch at level {lvl}")

    @dask_array_available
    @cuda_and_cupy_available
    def test_numpy_equals_dask_cupy(self, elevation_raster_no_nans):
        data = elevation_raster_no_nans
        levels = [300.0, 500.0, 700.0]
        numpy_agg = create_test_raster(data, backend='numpy')
        dask_cupy_agg = create_test_raster(data, backend='dask+cupy',
                                           chunks=(4, 3))

        np_result = contours(numpy_agg, levels=levels)
        dc_result = contours(dask_cupy_agg, levels=levels)

        np_segs = self._collect_segments(np_result)
        dc_segs = self._collect_segments(dc_result)

        assert set(np_segs.keys()) == set(dc_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == dc_segs[lvl], (
                f"Segment mismatch at level {lvl}")


# ---------------------------------------------------------------------------
# Integer dtype: NaN halo regression (issue #3020)
# ---------------------------------------------------------------------------

class TestIntegerDtypeCollar:
    """boundary=np.nan can't fill an integer halo, so dask used to leave a
    border collar that numpy never produced.  These guard the float cast in
    _overlap_for_contours.
    """

    LEVELS = [5.0, 10.0, 15.0]
    # Cover several integer widths/signedness; the int-min halo fill differs
    # per dtype, so a phantom crossing would show up regardless.
    INT_DTYPES = [np.int16, np.int32, np.int64, np.uint8]

    def _int_ramp(self, dtype, ny=20, nx=20):
        # Edge columns straddle the levels, so any phantom halo crossing
        # shows up as a frame around the raster.
        return np.tile(np.arange(nx), (ny, 1)).astype(dtype)

    @dask_array_available
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_int_dask_equals_numpy(self, dtype):
        data = self._int_ramp(dtype)
        np_agg = create_test_raster(data, backend='numpy')
        dk_agg = create_test_raster(data, backend='dask+numpy', chunks=(7, 7))

        np_segs = _segments_by_level(contours(np_agg, levels=self.LEVELS))
        dk_segs = _segments_by_level(contours(dk_agg, levels=self.LEVELS))

        assert set(np_segs.keys()) == set(dk_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == dk_segs[lvl], (
                f"Integer dask result diverges from numpy at level {lvl}")

    @dask_array_available
    def test_int_dask_no_border_collar(self):
        # A vertical ramp at these levels is a set of straight vertical lines.
        # A collar would push the bounding box out to the raster edges and
        # inflate the total length.
        pytest.importorskip("geopandas")
        data = self._int_ramp(np.int32)
        np_agg = create_test_raster(data, backend='numpy')
        dk_agg = create_test_raster(data, backend='dask+numpy', chunks=(7, 7))

        g_np = contours(np_agg, levels=self.LEVELS, return_type='geopandas')
        g_dk = contours(dk_agg, levels=self.LEVELS, return_type='geopandas')

        assert g_dk.length.sum() == pytest.approx(g_np.length.sum())
        np.testing.assert_allclose(g_dk.total_bounds, g_np.total_bounds)

    @dask_array_available
    @cuda_and_cupy_available
    def test_int_dask_cupy_equals_numpy(self):
        data = self._int_ramp(np.int32)
        np_agg = create_test_raster(data, backend='numpy')
        dc_agg = create_test_raster(data, backend='dask+cupy', chunks=(7, 7))

        np_segs = _segments_by_level(contours(np_agg, levels=self.LEVELS))
        dc_segs = _segments_by_level(contours(dc_agg, levels=self.LEVELS))

        assert set(np_segs.keys()) == set(dc_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == dc_segs[lvl], (
                f"Integer dask+cupy result diverges from numpy at level {lvl}")


# ---------------------------------------------------------------------------
# Return type: GeoDataFrame
# ---------------------------------------------------------------------------

class TestGeoDataFrame:

    def test_geopandas_return(self):
        pytest.importorskip("geopandas")
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        gdf = contours(agg, levels=[0.5, 1.5], return_type="geopandas")

        import geopandas as gpd
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert 'level' in gdf.columns
        assert 'geometry' in gdf.columns
        assert len(gdf) > 0

    def test_geopandas_propagates_crs(self):
        """A populated geopandas result carries the input raster's CRS."""
        pytest.importorskip("geopandas")
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')  # attrs include a crs
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        assert len(gdf) > 0
        assert gdf.crs == agg.attrs['crs']

    def test_geopandas_no_georef_marker_suppresses_crs(self):
        """attrs['_xrspatial_no_georef']=True suppresses CRS propagation.

        contours() resolves its CRS through polygonize._detect_raster_crs,
        which returns None when the geotiff reader's no-georeference
        marker is set (#3293).  The geometries are not georeferenced on
        that path, so attaching the CRS would misrepresent them.
        """
        pytest.importorskip("geopandas")
        data = _make_peak()
        agg = xr.DataArray(
            data, dims=['y', 'x'],
            attrs={'crs': 4326, '_xrspatial_no_georef': True})
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        assert len(gdf) > 0
        assert gdf.crs is None

    def test_geopandas_empty_result_keeps_crs(self):
        """Levels with no crossings return an empty GeoDataFrame with the CRS.

        Regression for #2700: gpd.GeoDataFrame(records, crs=crs) raised
        ValueError when records was empty and crs was not None.
        """
        pytest.importorskip("geopandas")
        import geopandas as gpd
        data = np.ones((4, 4), dtype=np.float64)  # flat -> no crossings
        agg = create_test_raster(data, backend='numpy')  # attrs include a crs
        gdf = contours(agg, levels=[5.0], return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0
        assert 'level' in gdf.columns
        assert 'geometry' in gdf.columns
        assert gdf.crs == agg.attrs['crs']

    def test_geopandas_all_nan_keeps_crs(self):
        """All-NaN input with auto levels keeps the CRS on the empty frame.

        Regression for #2700: the all-NaN early-return path dropped the CRS.
        """
        pytest.importorskip("geopandas")
        import geopandas as gpd
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')  # attrs include a crs
        gdf = contours(agg, return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0
        assert gdf.crs == agg.attrs['crs']

    def test_geopandas_empty_result_no_crs(self):
        """An empty result with no input CRS returns an empty frame, no crash."""
        pytest.importorskip("geopandas")
        import geopandas as gpd
        data = np.ones((4, 4), dtype=np.float64)
        agg = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(2, 0, 4), 'x': np.linspace(0, 2, 4)},
        )
        gdf = contours(agg, levels=[5.0], return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0
        assert gdf.crs is None

    def test_invalid_return_type(self):
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        with pytest.raises(ValueError, match="Invalid return_type"):
            contours(agg, levels=[0.5], return_type="invalid")


# ---------------------------------------------------------------------------
# Accessor integration
# ---------------------------------------------------------------------------

class TestAccessor:

    def test_dataarray_accessor(self):
        import xrspatial  # noqa: F401 -- registers accessors
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        result = agg.xrs.contours(levels=[0.5])
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Memory guard (#1240)
# ---------------------------------------------------------------------------


class TestMemoryGuard:

    def test_rejects_oversize_numpy(self, monkeypatch):
        """contours() raises MemoryError when segment buffers exceed budget."""
        import xrspatial.contour as contour_mod

        # Force available memory to 1 MB so even a small raster trips the
        # guard.  A 1000x1000 raster needs ~32 MB per level's buffers.
        monkeypatch.setattr(
            contour_mod, '_available_memory_bytes', lambda: 1 * 1024 * 1024
        )

        data = np.zeros((1000, 1000), dtype=np.float64)
        agg = xr.DataArray(data, dims=['y', 'x'])

        with pytest.raises(MemoryError, match="segment buffers per level"):
            contours(agg, levels=[0.5])

    def test_allows_within_budget(self, monkeypatch):
        """A small raster stays under the guard and returns normally."""
        import xrspatial.contour as contour_mod

        monkeypatch.setattr(
            contour_mod, '_available_memory_bytes', lambda: 8 * 1024 ** 3
        )

        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.5])
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Reference validation against skimage.measure.find_contours
# ---------------------------------------------------------------------------

class TestReferenceValidation:

    def test_matches_skimage(self):
        """Compare our contours against skimage as a reference."""
        skimage_measure = pytest.importorskip("skimage.measure")

        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 1., 2., 1., 0.],
            [0., 2., 4., 2., 0.],
            [0., 1., 2., 1., 0.],
            [0., 0., 0., 0., 0.],
        ], dtype=np.float64)

        level = 1.5
        agg = create_test_raster(data, backend='numpy')
        our_result = contours(agg, levels=[level])

        sk_contours = skimage_measure.find_contours(data, level)

        # Both should find contour lines at this level.
        assert len(our_result) > 0
        assert len(sk_contours) > 0

        # Transform our coordinate-space points back to array indices for
        # comparison with skimage (which returns array index coordinates).
        y_coords = agg.coords[agg.dims[0]].values
        x_coords = agg.coords[agg.dims[1]].values
        y_idx = np.arange(len(y_coords), dtype=np.float64)
        x_idx = np.arange(len(x_coords), dtype=np.float64)

        our_points_idx = []
        for _, coords in our_result:
            pts = np.empty_like(coords)
            pts[:, 0] = np.interp(coords[:, 0], y_coords[::-1], y_idx[::-1]) \
                if y_coords[0] > y_coords[-1] \
                else np.interp(coords[:, 0], y_coords, y_idx)
            pts[:, 1] = np.interp(coords[:, 1], x_coords, x_idx)
            our_points_idx.append(pts)
        our_points = np.vstack(our_points_idx)

        # Every skimage contour point should be close to some point of ours.
        for sk_line in sk_contours:
            for pt in sk_line:
                dists = np.linalg.norm(our_points - pt, axis=1)
                assert np.min(dists) < 0.5, (
                    f"skimage point {pt} not near any of our contour points"
                )


# ---------------------------------------------------------------------------
# Infinity handling (#2704)
# ---------------------------------------------------------------------------

class TestInfHandling:

    def _inf_peak(self, value):
        """5x5 raster of ones with a single +/-inf cell in the center."""
        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., value, 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.],
        ], dtype=np.float64)
        return data

    def test_inf_far_level_no_crossing(self):
        """A level the inf cell sits above on all sides still traces the
        surrounding ring without touching the inf quad's interpolation."""
        data = self._inf_peak(np.inf)
        agg = create_test_raster(data, backend='numpy')
        # Level 0.5 crosses the outer 0/1 boundary.  The four quads that
        # touch the inf cell have all corners >= 0.5 (1 and inf), so they
        # are the all-above case (idx 15) and are skipped before any
        # interpolation runs.
        result = contours(agg, levels=[0.5])
        assert len(result) >= 1
        for level, coords in result:
            assert np.isfinite(coords).all(), (
                "level 0.5 ring should not include the inf quad")

    def test_inf_corner_no_nan_coords(self):
        """A finite level near a +inf cell must not leak NaN coordinates.

        Regression for #2704: the NaN-skip guard in the kernel used ``x != x``
        which does not catch infinity; fixed by using ``np.isfinite``.
        """
        data = self._inf_peak(np.inf)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.5])
        for level, coords in result:
            assert np.isfinite(coords).all(), (
                f"non-finite coordinate in contour at level {level}: {coords}")

    def test_neg_inf_corner_no_nan_coords(self):
        """A finite level near a -inf cell must not leak NaN coordinates.

        Regression for #2704: same fix as test_inf_corner_no_nan_coords.
        """
        data = self._inf_peak(-np.inf)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5])
        for level, coords in result:
            assert np.isfinite(coords).all(), (
                f"non-finite coordinate in contour at level {level}: {coords}")

    def test_mixed_inf(self):
        """Multiple infinities of opposite signs must not produce NaN."""
        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., np.inf, 1., -np.inf, 0.],
            [0., 1., 1., 1., 0.],
            [0., -np.inf, 1., np.inf, 0.],
            [0., 0., 0., 0., 0.],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5])
        for level, coords in result:
            assert np.isfinite(coords).all(), \
                f"Non-finite coordinates found at level {level}"

    def test_all_inf_quad(self):
        """A 2x2 raster with all corners infinite produces no contours."""
        data = np.full((2, 2), np.inf, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.0])
        assert result == []


# ---------------------------------------------------------------------------
# CRS propagation to GeoDataFrame output (#2704 audit, Cat 5)
# ---------------------------------------------------------------------------

class TestCRSPropagation:

    def test_geopandas_crs_from_attrs(self):
        """return_type='geopandas' copies agg.attrs['crs'] onto the gdf."""
        pytest.importorskip("geopandas")
        data = _make_peak()
        # create_test_raster sets attrs={'res': ..., 'crs': 'EPSG: 5070'}.
        agg = create_test_raster(data, backend='numpy')
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        # GeoPandas normalizes the CRS; check it resolves to EPSG:5070.
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 5070

    def test_geopandas_no_crs_attr(self):
        """A raster with no crs attr yields a GeoDataFrame with crs None."""
        gpd = pytest.importorskip("geopandas")
        data = _make_peak()
        agg = xr.DataArray(data, dims=['y', 'x'])
        agg['y'] = np.linspace(2.0, 0.0, data.shape[0])
        agg['x'] = np.linspace(0.0, 2.0, data.shape[1])
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs is None

    # CRS-resolver parity with polygonize (#2893). contours must use the
    # same resolution order as polygonize._detect_raster_crs:
    #   attrs['crs'] -> attrs['crs_wkt'] -> raster.rio.crs -> None.

    @staticmethod
    def _bare_raster():
        """A peak raster with explicit coords and no CRS metadata."""
        data = _make_peak()
        agg = xr.DataArray(data, dims=['y', 'x'])
        agg['y'] = np.linspace(2.0, 0.0, data.shape[0])
        agg['x'] = np.linspace(0.0, 2.0, data.shape[1])
        return agg

    def test_geopandas_crs_from_crs_wkt(self):
        """A raster with only attrs['crs_wkt'] still georeferences the gdf.

        Previously contours read only attrs['crs'], so a crs_wkt-only raster
        produced an unprojected GeoDataFrame.
        """
        pytest.importorskip("geopandas")
        from pyproj import CRS

        agg = self._bare_raster()
        agg.attrs['crs_wkt'] = CRS.from_epsg(5070).to_wkt()
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 5070

    def test_geopandas_crs_attr_precedence(self):
        """attrs['crs'] wins over attrs['crs_wkt'] when both are present."""
        pytest.importorskip("geopandas")
        from pyproj import CRS

        agg = self._bare_raster()
        agg.attrs['crs'] = 'EPSG:5070'
        agg.attrs['crs_wkt'] = CRS.from_epsg(4326).to_wkt()
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 5070

    def test_geopandas_no_crs_info(self):
        """A raster with no CRS info yields a GeoDataFrame with crs None."""
        gpd = pytest.importorskip("geopandas")
        agg = self._bare_raster()
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs is None

    @staticmethod
    def _wkt_4326():
        from pyproj import CRS
        return {'crs_wkt': CRS.from_epsg(4326).to_wkt()}

    @pytest.mark.parametrize("attrs_factory", [
        pytest.param(lambda: {'crs': 'EPSG:5070'}, id="crs"),
        pytest.param(lambda: TestCRSPropagation._wkt_4326(), id="crs_wkt"),
        pytest.param(lambda: {}, id="no_crs"),
    ])
    def test_geopandas_crs_matches_detect_raster_crs(self, attrs_factory):
        """contours resolves the same CRS polygonize would for one raster."""
        pytest.importorskip("geopandas")
        from pyproj import CRS

        from xrspatial.polygonize import _detect_raster_crs

        agg = self._bare_raster()
        agg.attrs.update(attrs_factory())
        gdf = contours(agg, levels=[1.5], return_type="geopandas")

        expected = _detect_raster_crs(agg)
        if expected is None:
            assert gdf.crs is None
        else:
            assert gdf.crs == CRS.from_user_input(expected)


# ---------------------------------------------------------------------------
# Non-default dim names: index -> coordinate transform (#2704 audit, Cat 5)
# ---------------------------------------------------------------------------

class TestNonDefaultDims:

    def test_lat_lon_dims_coordinate_transform(self):
        """Output coordinates map into the lat/lon coordinate space.

        contours() reads agg.dims[0]/[1] coords to convert array indices
        to coordinate values, so non-y/x dim names must still work and the
        coordinates must land inside the lat/lon ranges.
        """
        data = _make_peak()
        ny, nx = data.shape
        lat = np.linspace(40.0, 30.0, ny)   # decreasing, like a north-up DEM
        lon = np.linspace(-100.0, -90.0, nx)
        agg = xr.DataArray(data, dims=['lat', 'lon'])
        agg['lat'] = lat
        agg['lon'] = lon

        result = contours(agg, levels=[1.5])
        assert len(result) >= 1
        for level, coords in result:
            # coords are (lat, lon) in the input coordinate space.
            assert np.all(coords[:, 0] >= lat.min() - 1e-9)
            assert np.all(coords[:, 0] <= lat.max() + 1e-9)
            assert np.all(coords[:, 1] >= lon.min() - 1e-9)
            assert np.all(coords[:, 1] <= lon.max() + 1e-9)

    def test_lat_lon_matches_yx_equivalent(self):
        """Renaming y/x to lat/lon (same coord values) gives same output."""
        data = _make_peak()
        ny, nx = data.shape
        y = np.linspace(2.0, 0.0, ny)
        x = np.linspace(0.0, 2.5, nx)

        yx = xr.DataArray(data, dims=['y', 'x'])
        yx['y'] = y
        yx['x'] = x

        ll = xr.DataArray(data, dims=['lat', 'lon'])
        ll['lat'] = y
        ll['lon'] = x

        r_yx = contours(yx, levels=[1.5])
        r_ll = contours(ll, levels=[1.5])

        assert len(r_yx) == len(r_ll)
        for (lvl_a, c_a), (lvl_b, c_b) in zip(r_yx, r_ll):
            assert lvl_a == lvl_b
            np.testing.assert_allclose(c_a, c_b)


# ---------------------------------------------------------------------------
# Degenerate geometry at exact-level corners (issue #2892)
# ---------------------------------------------------------------------------

def _make_checkerboard(n=4, lo=0.0, hi=1.0):
    """n x n checkerboard alternating between lo and hi."""
    board = np.indices((n, n)).sum(axis=0) % 2
    return np.where(board == 0, lo, hi).astype(np.float64)


def _assert_no_degenerate_numpy(result):
    """Every numpy polyline has at least two distinct vertices, no repeats."""
    for level, coords in result:
        # No two consecutive points are identical.
        if len(coords) >= 2:
            diffs = np.abs(np.diff(coords, axis=0)).sum(axis=1)
            assert np.all(diffs > 0), (
                f"repeated consecutive point at level {level}: {coords}"
            )
        # At least two distinct vertices (non-zero extent).
        distinct = np.unique(np.round(coords, 10), axis=0)
        assert len(distinct) >= 2, (
            f"single-point polyline at level {level}: {coords}"
        )


def _assert_no_degenerate_geopandas(gdf):
    """No geometry is zero-length or invalid in Shapely."""
    for geom in gdf.geometry:
        assert geom.length > 0, f"zero-length geometry: {geom.wkt}"
        assert geom.is_valid, f"invalid geometry: {geom.wkt}"


def _assert_no_duplicate_lines(result):
    """Assert that no two contour lines share the same geometry at the same level.

    Each polyline is decomposed into canonical segments (smaller endpoint first,
    rounded to 10 decimals). Two lines at the same level are duplicates if they
    have the same set of canonical segments.
    """
    by_level = defaultdict(list)
    DECIMALS = 10
    for level, coords in result:
        segs = []
        for i in range(len(coords) - 1):
            p0 = (round(coords[i, 0], DECIMALS), round(coords[i, 1], DECIMALS))
            p1 = (round(coords[i + 1, 0], DECIMALS), round(coords[i + 1, 1], DECIMALS))
            segs.append((min(p0, p1), max(p0, p1)))
        by_level[level].append(tuple(sorted(segs)))

    for level, line_signatures in by_level.items():
        seen = set()
        for sig in line_signatures:
            assert sig not in seen, (
                f"Duplicate contour line at level {level}"
            )
            seen.add(sig)


class TestDegenerateExactLevel:
    """A corner exactly equal to the level must not poison the output.

    Corners are classified with ``>= level`` (treated as above), so the
    fix is to drop the zero-length / single-point segments that would
    otherwise collapse onto that corner.  The rule must hold identically
    on every backend.
    """

    def test_checkerboard_numpy_no_zero_length(self):
        data = _make_checkerboard(4, lo=0.0, hi=1.0)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.0])
        _assert_no_degenerate_numpy(result)

    def test_checkerboard_numpy_geopandas_valid(self):
        pytest.importorskip("geopandas")
        data = _make_checkerboard(4, lo=0.0, hi=1.0)
        agg = create_test_raster(data, backend='numpy')
        gdf = contours(agg, levels=[1.0], return_type="geopandas")
        _assert_no_degenerate_geopandas(gdf)

    def test_checkerboard_equality_consistent(self):
        """The level lands on every 'hi' corner; orientation must not matter.

        Two checkerboards that differ only by which phase carries the
        exact-level value must both yield clean (degenerate-free) output.
        """
        for lo, hi in [(0.0, 1.0), (1.0, 2.0), (2.0, 1.0)]:
            data = _make_checkerboard(4, lo=lo, hi=hi)
            agg = create_test_raster(data, backend='numpy')
            result = contours(agg, levels=[1.0])
            _assert_no_degenerate_numpy(result)

    @dask_array_available
    def test_checkerboard_dask_no_zero_length(self):
        data = _make_checkerboard(4, lo=0.0, hi=1.0)
        agg = create_test_raster(data, backend='dask+numpy', chunks=(2, 2))
        result = contours(agg, levels=[1.0])
        _assert_no_degenerate_numpy(result)

    @dask_array_available
    def test_checkerboard_dask_geopandas_valid(self):
        pytest.importorskip("geopandas")
        data = _make_checkerboard(4, lo=0.0, hi=1.0)
        agg = create_test_raster(data, backend='dask+numpy', chunks=(2, 2))
        gdf = contours(agg, levels=[1.0], return_type="geopandas")
        _assert_no_degenerate_geopandas(gdf)

    @dask_array_available
    def test_checkerboard_numpy_matches_dask(self):
        """numpy and dask agree that the checkerboard yields no geometry."""
        data = _make_checkerboard(4, lo=0.0, hi=1.0)
        np_agg = create_test_raster(data, backend='numpy')
        dk_agg = create_test_raster(
            data, backend='dask+numpy', chunks=(2, 2)
        )
        np_res = contours(np_agg, levels=[1.0])
        dk_res = contours(dk_agg, levels=[1.0])
        _assert_no_degenerate_numpy(np_res)
        _assert_no_degenerate_numpy(dk_res)
        assert len(np_res) == len(dk_res)

    def test_genuine_contour_survives(self):
        """The degenerate filter must not drop real crossings.

        A ramp that crosses the level mid-edge still produces a valid,
        non-zero-length contour.
        """
        pytest.importorskip("geopandas")
        data = _make_ramp(ny=5, nx=6)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[2.5])
        assert len(result) > 0
        _assert_no_degenerate_numpy(result)
        gdf = contours(agg, levels=[2.5], return_type="geopandas")
        assert len(gdf) > 0
        _assert_no_degenerate_geopandas(gdf)

    def test_plateau_no_duplicate_geometries(self):
        """A flat interior plateau must not produce duplicate contour lines.

        Regression for #2790: when a plateau's interior cells all equal the
        contour level, overlapping chunk boundaries (dask) or saddle-case
        disambiguation (numpy) can emit the same polyline twice.
        """
        data = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=float)
        np_agg = create_test_raster(data, backend='numpy')
        np_result = contours(np_agg, levels=[1.0])

        # All returned lines must be geometrically unique.
        _assert_no_duplicate_lines(np_result)

    @dask_array_available
    def test_plateau_no_duplicate_geometries_dask(self):
        """Same plateau test with dask backend.

        Regression for #2790: overlapping chunk boundaries can emit
        duplicate polylines that must be deduplicated.
        """
        data = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=float)
        dk_agg = create_test_raster(data, backend='dask+numpy', chunks=(2, 2))
        dk_result = contours(dk_agg, levels=[1.0])

        _assert_no_duplicate_lines(dk_result)

    def test_plateau_no_duplicate_geometries_geopandas(self):
        """Verify no duplicate geometries in GeoDataFrame output.

        Regression for #2790: deduplication must apply before
        GeoDataFrame construction.
        """
        pytest.importorskip("geopandas")
        data = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=float)
        np_agg = create_test_raster(data, backend='numpy')
        gdf = contours(np_agg, levels=[1.0], return_type="geopandas")

        # Each row's geometry must be unique.
        seen = set()
        for _, row in gdf.iterrows():
            geom_key = tuple(round(v, 10) for c in row.geometry.coords for v in c)
            assert geom_key not in seen, (
                f"Duplicate geometry in GeoDataFrame at level {row.level}"
            )
            seen.add(geom_key)


class TestDeduplicateLines:

    def test_deduplicate_lines_removes_exact_duplicates(self):
        """_deduplicate_lines removes identical polylines at the same level."""
        from xrspatial.contour import _deduplicate_lines
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        results = [(1.0, coords.copy()), (1.0, coords.copy())]
        deduped = _deduplicate_lines(results)
        assert len(deduped) == 1
        assert deduped[0][0] == 1.0
        np.testing.assert_allclose(deduped[0][1], coords)

    def test_deduplicate_lines_keeps_different_levels(self):
        """_deduplicate_lines keeps lines at different levels even with same geometry."""
        from xrspatial.contour import _deduplicate_lines
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        results = [(1.0, coords.copy()), (2.0, coords.copy())]
        deduped = _deduplicate_lines(results)
        assert len(deduped) == 2

    def test_deduplicate_lines_removes_reverse_duplicates(self):
        """_deduplicate_lines removes polylines that trace the same segments in reverse."""
        from xrspatial.contour import _deduplicate_lines
        fwd = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        rev = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        results = [(1.0, fwd.copy()), (1.0, rev.copy())]
        deduped = _deduplicate_lines(results)
        assert len(deduped) == 1


# ---------------------------------------------------------------------------
# Cross-backend parity with NaN input (#3044)
# ---------------------------------------------------------------------------

class TestNaNBackendParity:
    """A raster with NaN cells must trace identical segments on every backend.

    The numpy backend skips quads with a non-finite corner in the interior;
    the dask backend pads each chunk with a NaN halo and stitches across
    chunk boundaries.  The existing backend-equivalence tests use a no-NaN
    fixture, so nothing pins numpy/cupy/dask parity when NaN cells sit next
    to a chunk edge.  This guards that path.
    """

    LEVELS = [2.5, 5.5, 8.5]

    def _partial_nan_ramp(self, ny=10, nx=12):
        # Left-to-right ramp so every level crosses, then punch a NaN edge
        # row and an interior NaN cell.  The interior NaN lands inside a
        # non-edge chunk so a halo crossing would diverge from numpy.
        data = np.tile(np.arange(nx, dtype=np.float64), (ny, 1))
        data[0, :] = np.nan      # NaN edge row
        data[5, 6] = np.nan      # interior NaN cell
        return data

    @dask_array_available
    def test_nan_dask_equals_numpy(self):
        data = self._partial_nan_ramp()
        np_agg = create_test_raster(data, backend='numpy')
        dk_agg = create_test_raster(data, backend='dask+numpy', chunks=(4, 4))

        np_segs = _segments_by_level(contours(np_agg, levels=self.LEVELS))
        dk_segs = _segments_by_level(contours(dk_agg, levels=self.LEVELS))

        assert set(np_segs.keys()) == set(dk_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == dk_segs[lvl], (
                f"NaN-input dask result diverges from numpy at level {lvl}")

    @cuda_and_cupy_available
    def test_nan_cupy_equals_numpy(self):
        data = self._partial_nan_ramp()
        np_agg = create_test_raster(data, backend='numpy')
        cp_agg = create_test_raster(data, backend='cupy')

        np_segs = _segments_by_level(contours(np_agg, levels=self.LEVELS))
        cp_segs = _segments_by_level(contours(cp_agg, levels=self.LEVELS))

        assert set(np_segs.keys()) == set(cp_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == cp_segs[lvl], (
                f"NaN-input cupy result diverges from numpy at level {lvl}")

    @dask_array_available
    @cuda_and_cupy_available
    def test_nan_dask_cupy_equals_numpy(self):
        data = self._partial_nan_ramp()
        np_agg = create_test_raster(data, backend='numpy')
        dc_agg = create_test_raster(data, backend='dask+cupy', chunks=(4, 4))

        np_segs = _segments_by_level(contours(np_agg, levels=self.LEVELS))
        dc_segs = _segments_by_level(contours(dc_agg, levels=self.LEVELS))

        assert set(np_segs.keys()) == set(dc_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == dc_segs[lvl], (
                f"NaN-input dask+cupy result diverges from numpy at level {lvl}")
