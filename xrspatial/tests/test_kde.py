"""Tests for xrspatial.kde (kernel density estimation)."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.kde import kde, line_density, _silverman_bandwidth

from xrspatial.tests.general_checks import (
    dask_array_available,
    cuda_and_cupy_available,
)

try:
    import dask.array as da
except ImportError:
    da = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def point_cluster():
    """Tight cluster of points at the origin -- easy to validate."""
    rng = np.random.default_rng(1143)
    x = rng.normal(0, 1, 200)
    y = rng.normal(0, 1, 200)
    return x, y


@pytest.fixture
def simple_grid():
    """Small 16x16 template grid centred on the origin."""
    return xr.DataArray(
        np.zeros((16, 16), dtype=np.float64),
        dims=['y', 'x'],
        coords={
            'y': np.linspace(-4, 4, 16),
            'x': np.linspace(-4, 4, 16),
        },
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestKDEBasic:
    """Core KDE functionality on numpy arrays."""

    def test_output_shape_from_width_height(self, point_cluster):
        x, y = point_cluster
        result = kde(x, y, bandwidth=1.0, width=20, height=30)
        assert result.shape == (30, 20)
        assert result.dims == ('y', 'x')

    def test_output_shape_from_template(self, point_cluster, simple_grid):
        x, y = point_cluster
        result = kde(x, y, bandwidth=1.0, template=simple_grid)
        assert result.shape == simple_grid.shape
        np.testing.assert_array_equal(result.coords['y'], simple_grid.coords['y'])
        np.testing.assert_array_equal(result.coords['x'], simple_grid.coords['x'])

    def test_density_positive(self, point_cluster):
        x, y = point_cluster
        result = kde(x, y, bandwidth=1.0, width=16, height=16)
        assert float(result.min()) >= 0.0
        assert float(result.sum()) > 0.0

    def test_peak_near_data_centre(self, point_cluster, simple_grid):
        """Density peak should be near (0, 0) for a standard-normal cluster."""
        x, y = point_cluster
        result = kde(x, y, bandwidth=1.0, template=simple_grid)
        flat_idx = int(result.values.argmax())
        peak_row, peak_col = divmod(flat_idx, result.shape[1])
        y_peak = float(result.coords['y'].values[peak_row])
        x_peak = float(result.coords['x'].values[peak_col])
        assert abs(x_peak) < 2.0
        assert abs(y_peak) < 2.0

    def test_name_propagated(self, point_cluster):
        x, y = point_cluster
        result = kde(x, y, bandwidth=1.0, name='density', width=8, height=8)
        assert result.name == 'density'


class TestKernelTypes:
    """Each kernel produces valid, non-zero output."""

    @pytest.mark.parametrize('kernel', ['gaussian', 'epanechnikov', 'quartic'])
    def test_kernel_produces_output(self, point_cluster, kernel):
        x, y = point_cluster
        result = kde(x, y, bandwidth=1.0, kernel=kernel, width=16, height=16)
        assert float(result.sum()) > 0.0

    def test_compact_kernels_are_zero_outside_bandwidth(self):
        """A single point at origin, quartic kernel: pixels beyond bw should be 0."""
        result = kde(
            [0.0], [0.0], bandwidth=1.0, kernel='quartic',
            x_range=(-3, 3), y_range=(-3, 3), width=32, height=32,
        )
        xs = result.coords['x'].values
        ys = result.coords['y'].values
        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                dist2 = xs[c] ** 2 + ys[r] ** 2
                if dist2 > 1.0:
                    assert result.values[r, c] == 0.0

    def test_gaussian_nonzero_everywhere(self):
        """Gaussian kernel should have non-zero density everywhere (within cutoff)."""
        result = kde(
            [0.0], [0.0], bandwidth=1.0, kernel='gaussian',
            x_range=(-2, 2), y_range=(-2, 2), width=8, height=8,
        )
        assert float(result.min()) > 0.0


class TestBandwidth:
    """Bandwidth selection and validation."""

    def test_silverman_auto(self, point_cluster):
        x, y = point_cluster
        bw = _silverman_bandwidth(x, y)
        assert bw > 0

    def test_silverman_keyword(self, point_cluster):
        x, y = point_cluster
        result = kde(x, y, bandwidth='silverman', width=8, height=8)
        assert float(result.sum()) > 0.0

    def test_wider_bandwidth_smoother(self, point_cluster, simple_grid):
        """Wider bandwidth should produce a smoother (lower peak) surface."""
        x, y = point_cluster
        narrow = kde(x, y, bandwidth=0.5, template=simple_grid)
        wide = kde(x, y, bandwidth=2.0, template=simple_grid)
        assert float(narrow.max()) > float(wide.max())

    def test_negative_bandwidth_raises(self, point_cluster):
        x, y = point_cluster
        with pytest.raises(ValueError, match='positive'):
            kde(x, y, bandwidth=-1.0, width=8, height=8)

    def test_invalid_bandwidth_string_raises(self, point_cluster):
        x, y = point_cluster
        with pytest.raises(ValueError, match='silverman'):
            kde(x, y, bandwidth='invalid', width=8, height=8)


class TestWeights:
    """Weighted KDE."""

    def test_double_weights_double_density(self, simple_grid):
        x = np.array([0.0, 1.0, -1.0])
        y = np.array([0.0, 1.0, -1.0])
        r1 = kde(x, y, weights=[1, 1, 1], bandwidth=1.0, template=simple_grid)
        r2 = kde(x, y, weights=[2, 2, 2], bandwidth=1.0, template=simple_grid)
        np.testing.assert_allclose(r2.values, r1.values * 2.0, rtol=1e-12)

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match='same length'):
            kde([0, 1], [0, 1], weights=[1], bandwidth=1.0, width=4, height=4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_point(self):
        result = kde([5.0], [5.0], bandwidth=1.0, width=8, height=8)
        assert result.shape == (8, 8)
        assert float(result.sum()) > 0.0

    def test_two_identical_points(self):
        result = kde([0.0, 0.0], [0.0, 0.0], bandwidth=1.0, width=8, height=8)
        assert float(result.sum()) > 0.0

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match='kernel must be'):
            kde([0], [0], kernel='unknown', bandwidth=1.0, width=4, height=4)

    def test_mismatched_xy_raises(self):
        with pytest.raises(ValueError, match='same length'):
            kde([0, 1], [0], bandwidth=1.0, width=4, height=4)

    def test_template_not_dataarray_raises(self):
        with pytest.raises(TypeError, match='xr.DataArray'):
            kde([0], [0], bandwidth=1.0, template=np.zeros((4, 4)))

    def test_template_not_2d_raises(self):
        t = xr.DataArray(np.zeros((4, 4, 4)), dims=['z', 'y', 'x'])
        with pytest.raises(ValueError, match='2-D'):
            kde([0], [0], bandwidth=1.0, template=t)


# ---------------------------------------------------------------------------
# Memory guard (#1287)
# ---------------------------------------------------------------------------

class TestMemoryGuard:
    """kde() and line_density() must reject grids that would OOM."""

    def test_kde_huge_grid_raises_memory_error(self):
        # 1Mx1M float64 = 8 TB, well above any realistic budget.
        with pytest.raises(MemoryError, match='width/height'):
            kde([0.0], [0.0], bandwidth=1.0,
                width=1_000_000, height=1_000_000)

    def test_line_density_huge_grid_raises_memory_error(self):
        with pytest.raises(MemoryError, match='width/height'):
            line_density([0.0], [0.0], [1.0], [1.0],
                         bandwidth=1.0,
                         width=1_000_000, height=1_000_000)

    def test_kde_normal_grid_passes(self):
        # Sanity: typical grid sizes still work.
        result = kde([0.0], [0.0], bandwidth=1.0, width=256, height=256)
        assert result.shape == (256, 256)

    def test_line_density_normal_grid_passes(self):
        result = line_density([0.0], [0.0], [1.0], [1.0],
                              bandwidth=0.5, width=256, height=256)
        assert result.shape == (256, 256)

    def test_kde_memory_error_names_parameter(self):
        """Error message should hint at width/height so the user can fix it."""
        with pytest.raises(MemoryError) as exc:
            kde([0.0], [0.0], bandwidth=1.0,
                width=1_000_000, height=1_000_000)
        # Message should mention either dimension or the dask escape hatch.
        assert 'width' in str(exc.value) or 'height' in str(exc.value)


# ---------------------------------------------------------------------------
# Descending coordinate templates (#1198)
# ---------------------------------------------------------------------------

class TestDescendingCoordinates:
    """KDE must produce correct results when template coords are descending.

    Geospatial rasters commonly have row 0 at the top (descending y).
    Before the fix in #1198, negative dy/dx caused the bounding-box
    index math to produce lo > hi, so the inner loops never ran.
    """

    @pytest.fixture
    def asc_template(self):
        return xr.DataArray(
            np.zeros((16, 16), dtype=np.float64),
            dims=['y', 'x'],
            coords={
                'y': np.linspace(-4, 4, 16),
                'x': np.linspace(-4, 4, 16),
            },
        )

    @pytest.fixture
    def desc_y_template(self):
        """Row 0 = top (descending y, ascending x)."""
        return xr.DataArray(
            np.zeros((16, 16), dtype=np.float64),
            dims=['y', 'x'],
            coords={
                'y': np.linspace(4, -4, 16),
                'x': np.linspace(-4, 4, 16),
            },
        )

    @pytest.fixture
    def desc_x_template(self):
        """Ascending y, descending x."""
        return xr.DataArray(
            np.zeros((16, 16), dtype=np.float64),
            dims=['y', 'x'],
            coords={
                'y': np.linspace(-4, 4, 16),
                'x': np.linspace(4, -4, 16),
            },
        )

    @pytest.fixture
    def desc_both_template(self):
        """Both axes descending."""
        return xr.DataArray(
            np.zeros((16, 16), dtype=np.float64),
            dims=['y', 'x'],
            coords={
                'y': np.linspace(4, -4, 16),
                'x': np.linspace(4, -4, 16),
            },
        )

    def test_descending_y_nonzero(self, desc_y_template):
        result = kde([0.0], [0.0], bandwidth=1.0, template=desc_y_template)
        assert float(result.sum()) > 0.0

    def test_descending_y_matches_ascending(self, asc_template, desc_y_template):
        """Descending-y result should equal ascending-y result, row-flipped."""
        r_asc = kde([0.0], [0.0], bandwidth=1.0, template=asc_template)
        r_desc = kde([0.0], [0.0], bandwidth=1.0, template=desc_y_template)
        np.testing.assert_allclose(
            r_desc.values[::-1], r_asc.values, rtol=1e-12,
        )

    def test_descending_x_matches_ascending(self, asc_template, desc_x_template):
        """Descending-x result should equal ascending-x result, col-flipped."""
        r_asc = kde([0.0], [0.0], bandwidth=1.0, template=asc_template)
        r_desc = kde([0.0], [0.0], bandwidth=1.0, template=desc_x_template)
        np.testing.assert_allclose(
            r_desc.values[:, ::-1], r_asc.values, rtol=1e-12,
        )

    def test_both_descending_matches_ascending(self, asc_template,
                                                desc_both_template):
        r_asc = kde([0.0], [0.0], bandwidth=1.0, template=asc_template)
        r_both = kde([0.0], [0.0], bandwidth=1.0, template=desc_both_template)
        np.testing.assert_allclose(
            r_both.values[::-1, ::-1], r_asc.values, rtol=1e-12,
        )

    @pytest.mark.parametrize('kernel', ['epanechnikov', 'quartic'])
    def test_compact_kernels_descending_y(self, asc_template,
                                           desc_y_template, kernel):
        """Compact kernels match exactly (no cutoff-boundary rounding)."""
        pts_x = [0.0, 1.0, -1.0]
        pts_y = [0.0, 1.0, -1.0]
        r_asc = kde(pts_x, pts_y, bandwidth=1.0, kernel=kernel,
                    template=asc_template)
        r_desc = kde(pts_x, pts_y, bandwidth=1.0, kernel=kernel,
                     template=desc_y_template)
        np.testing.assert_allclose(
            r_desc.values[::-1], r_asc.values, rtol=1e-12,
        )

    def test_gaussian_descending_y_multipoint(self, asc_template,
                                               desc_y_template):
        """Gaussian with multiple points: allow tiny diffs at cutoff edge.

        The 4*bw box cutoff means int() truncation can include/exclude
        a different fringe pixel when spacing flips sign. The affected
        values are near exp(-8) ~ 3e-4, so atol=1e-5 is plenty tight.
        """
        pts_x = [0.0, 1.0, -1.0]
        pts_y = [0.0, 1.0, -1.0]
        r_asc = kde(pts_x, pts_y, bandwidth=1.0, kernel='gaussian',
                    template=asc_template)
        r_desc = kde(pts_x, pts_y, bandwidth=1.0, kernel='gaussian',
                     template=desc_y_template)
        np.testing.assert_allclose(
            r_desc.values[::-1], r_asc.values, atol=1e-5,
        )

    def test_line_density_descending_y_quartic(self, asc_template,
                                                desc_y_template):
        """Compact kernel: exact match with descending y."""
        r_asc = line_density([0.0], [0.0], [1.0], [1.0],
                             bandwidth=0.5, kernel='quartic',
                             template=asc_template)
        r_desc = line_density([0.0], [0.0], [1.0], [1.0],
                              bandwidth=0.5, kernel='quartic',
                              template=desc_y_template)
        np.testing.assert_allclose(
            r_desc.values[::-1], r_asc.values, rtol=1e-12,
        )

    def test_line_density_descending_y_gaussian(self, asc_template,
                                                 desc_y_template):
        """Gaussian: allow tiny diffs at the 4*bw cutoff edge."""
        r_asc = line_density([0.0], [0.0], [1.0], [1.0],
                             bandwidth=0.5, kernel='gaussian',
                             template=asc_template)
        r_desc = line_density([0.0], [0.0], [1.0], [1.0],
                              bandwidth=0.5, kernel='gaussian',
                              template=desc_y_template)
        np.testing.assert_allclose(
            r_desc.values[::-1], r_asc.values, atol=1e-4,
        )

    def test_line_density_descending_x(self, asc_template, desc_x_template):
        r_asc = line_density([0.0], [0.0], [1.0], [1.0],
                             bandwidth=0.5, kernel='quartic',
                             template=asc_template)
        r_desc = line_density([0.0], [0.0], [1.0], [1.0],
                              bandwidth=0.5, kernel='quartic',
                              template=desc_x_template)
        np.testing.assert_allclose(
            r_desc.values[:, ::-1], r_asc.values, rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Line density
# ---------------------------------------------------------------------------

class TestLineDensity:

    def test_basic_line(self):
        result = line_density(
            [0.0], [0.0], [1.0], [1.0],
            bandwidth=0.5, width=16, height=16,
        )
        assert result.shape == (16, 16)
        assert float(result.sum()) > 0.0

    def test_name(self):
        result = line_density(
            [0], [0], [1], [1],
            bandwidth=0.5, name='roads', width=8, height=8,
        )
        assert result.name == 'roads'

    def test_parallel_lines_higher_density_between(self):
        """Two parallel horizontal lines: density should be higher between them."""
        x1 = np.array([0, 0])
        y1 = np.array([-1, 1])
        x2 = np.array([2, 2])
        y2 = np.array([-1, 1])
        result = line_density(
            x1, y1, x2, y2,
            bandwidth=1.0,
            x_range=(-1, 3), y_range=(-3, 3),
            width=16, height=32,
        )
        # Centre row should have higher density than edge rows
        mid_row = result.shape[0] // 2
        centre_density = float(result.values[mid_row, :].mean())
        edge_density = float(result.values[0, :].mean())
        assert centre_density > edge_density

    def test_mismatched_arrays_raises(self):
        with pytest.raises(ValueError, match='same length'):
            line_density([0, 1], [0], [1], [1], bandwidth=1.0, width=4, height=4)

    def test_weighted_lines(self):
        r1 = line_density([0], [0], [1], [1], weights=[1],
                          bandwidth=0.5, width=8, height=8)
        r2 = line_density([0], [0], [1], [1], weights=[3],
                          bandwidth=0.5, width=8, height=8)
        np.testing.assert_allclose(r2.values, r1.values * 3.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------

@dask_array_available
class TestDaskParity:
    """Dask+numpy results should match pure numpy."""

    def _make_dask_template(self, simple_grid, chunks=(8, 8)):
        return simple_grid.copy(data=da.from_array(simple_grid.values, chunks=chunks))

    @pytest.mark.parametrize('kernel', ['gaussian', 'epanechnikov', 'quartic'])
    def test_dask_matches_numpy(self, point_cluster, simple_grid, kernel):
        x, y = point_cluster
        np_result = kde(x, y, bandwidth=1.0, kernel=kernel, template=simple_grid)
        dask_template = self._make_dask_template(simple_grid)
        dask_result = kde(x, y, bandwidth=1.0, kernel=kernel, template=dask_template)
        assert isinstance(dask_result.data, da.Array)
        np.testing.assert_allclose(
            dask_result.values, np_result.values, rtol=1e-4,
        )

    def test_compact_kernel_exact_match(self, point_cluster, simple_grid):
        """Compact kernels should match exactly since there are no cutoff boundary effects."""
        x, y = point_cluster
        np_result = kde(x, y, bandwidth=1.0, kernel='quartic', template=simple_grid)
        dask_template = self._make_dask_template(simple_grid)
        dask_result = kde(x, y, bandwidth=1.0, kernel='quartic', template=dask_template)
        np.testing.assert_allclose(
            dask_result.values, np_result.values, rtol=1e-12,
        )


@cuda_and_cupy_available
class TestCuPyParity:
    """CuPy results should match numpy."""

    def _make_cupy_template(self, simple_grid):
        import cupy
        return simple_grid.copy(data=cupy.asarray(simple_grid.values))

    @pytest.mark.parametrize('kernel', ['gaussian', 'epanechnikov', 'quartic'])
    def test_cupy_matches_numpy(self, point_cluster, simple_grid, kernel):
        x, y = point_cluster
        np_result = kde(x, y, bandwidth=1.0, kernel=kernel, template=simple_grid)
        cupy_template = self._make_cupy_template(simple_grid)
        cupy_result = kde(x, y, bandwidth=1.0, kernel=kernel, template=cupy_template)
        import cupy
        result_np = cupy_result.data.get()
        # Gaussian: CPU uses a 4*bw box cutoff, GPU sums all points.
        # Compact kernels match exactly.
        tol = 1e-2 if kernel == 'gaussian' else 1e-6
        np.testing.assert_allclose(result_np, np_result.values, rtol=tol)
