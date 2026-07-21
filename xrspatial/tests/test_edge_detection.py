import inspect
import re

import numpy as np
import pytest
import xarray as xr

from xrspatial.edge_detection import (LAPLACIAN_KERNEL, PREWITT_X, PREWITT_Y, SOBEL_X, SOBEL_Y,
                                      laplacian, prewitt_x, prewitt_y, sobel_x, sobel_y)
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_numpy_equals_dask_numpy, create_test_raster,
                                            cuda_and_cupy_available, dask_array_available,
                                            general_output_checks)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ramp_data():
    """5x6 ramp: values increase left-to-right, constant top-to-bottom."""
    return np.arange(30, dtype=np.float64).reshape(5, 6) % 6


@pytest.fixture
def numpy_agg(ramp_data):
    return create_test_raster(ramp_data, backend='numpy')


@pytest.fixture
def dask_agg(ramp_data):
    return create_test_raster(ramp_data, backend='dask+numpy', chunks=(5, 6))


# ---------------------------------------------------------------------------
# Kernel sanity checks
# ---------------------------------------------------------------------------

class TestKernels:
    def test_sobel_x_shape_and_sum(self):
        assert SOBEL_X.shape == (3, 3)
        assert SOBEL_X.sum() == 0

    def test_sobel_y_shape_and_sum(self):
        assert SOBEL_Y.shape == (3, 3)
        assert SOBEL_Y.sum() == 0

    def test_prewitt_x_shape_and_sum(self):
        assert PREWITT_X.shape == (3, 3)
        assert PREWITT_X.sum() == 0

    def test_prewitt_y_shape_and_sum(self):
        assert PREWITT_Y.shape == (3, 3)
        assert PREWITT_Y.sum() == 0

    def test_laplacian_shape_and_sum(self):
        assert LAPLACIAN_KERNEL.shape == (3, 3)
        assert LAPLACIAN_KERNEL.sum() == 0

    def test_sobel_xy_transpose_relationship(self):
        np.testing.assert_array_equal(SOBEL_X.T, SOBEL_Y)

    def test_prewitt_xy_transpose_relationship(self):
        np.testing.assert_array_equal(PREWITT_X.T, PREWITT_Y)


# ---------------------------------------------------------------------------
# Correctness against hand-computed values
# ---------------------------------------------------------------------------

class TestCorrectnessNumpy:
    """Verify interior cells against manually computed convolutions."""

    def test_sobel_x_on_ramp(self, numpy_agg):
        result = sobel_x(numpy_agg)
        # ramp repeats [0,1,2,3,4,5] per row, so horizontal gradient is constant
        # interior cell (2,3): sum of kernel * data = 4*(-1+1) + 4*(-2+2) ... = 4
        # (every interior pixel on this ramp has the same Sobel-X response)
        interior = result.data[1:-1, 1:-1]
        assert not np.any(np.isnan(interior))
        # all interior values should be equal (uniform horizontal gradient)
        np.testing.assert_allclose(interior, interior[0, 0])

    def test_sobel_y_on_ramp(self, numpy_agg):
        result = sobel_y(numpy_agg)
        interior = result.data[1:-1, 1:-1]
        assert not np.any(np.isnan(interior))
        # vertical gradient of a repeating row-pattern is zero at interior
        # cells where top/bottom rows are identical
        # rows are: [0,1,2,3,4,5] repeated, so dz_dy = 0
        np.testing.assert_allclose(interior, 0, atol=1e-6)

    def test_laplacian_on_constant(self):
        data = np.ones((5, 6), dtype=np.float64) * 7.0
        agg = create_test_raster(data, backend='numpy')
        result = laplacian(agg)
        interior = result.data[1:-1, 1:-1]
        # Laplacian of a constant field is zero
        np.testing.assert_allclose(interior, 0, atol=1e-6)

    def test_prewitt_x_on_ramp(self, numpy_agg):
        result = prewitt_x(numpy_agg)
        interior = result.data[1:-1, 1:-1]
        assert not np.any(np.isnan(interior))
        # uniform horizontal gradient -> constant response
        np.testing.assert_allclose(interior, interior[0, 0])

    def test_prewitt_y_on_ramp(self, numpy_agg):
        result = prewitt_y(numpy_agg)
        interior = result.data[1:-1, 1:-1]
        # vertical gradient is zero for repeating rows
        np.testing.assert_allclose(interior, 0, atol=1e-6)


# ---------------------------------------------------------------------------
# Golden values pinned against scipy.ndimage
# ---------------------------------------------------------------------------

class TestGoldenScipyParity:
    """Pin parity with scipy.ndimage using hardcoded reference outputs.

    Expected arrays were generated with scipy.ndimage 1.16.1:
    ``ndi.sobel(data, axis=1)``, ``ndi.sobel(data, axis=0)``,
    ``ndi.prewitt(data, axis=1)``, ``ndi.prewitt(data, axis=0)`` and
    ``ndi.laplace(data)``, all with scipy's default ``mode='reflect'``,
    which matches xrspatial's ``boundary='reflect'``. The values are
    integer-exact for this input, so comparisons use equality.
    """

    DATA = np.array([[3, 1, 4, 1, 5, 9],
                     [2, 6, 5, 3, 5, 8],
                     [9, 7, 9, 3, 2, 3],
                     [8, 4, 6, 2, 6, 4],
                     [3, 3, 8, 3, 2, 7]], dtype=np.float64)

    EXPECTED = {
        'sobel_x': np.array([[-2, 6, -3, 3, 29, 15],
                             [4, 7, -10, -6, 18, 11],
                             [-4, 1, -13, -14, 7, 3],
                             [-10, 1, -8, -13, 8, 2],
                             [-4, 13, -2, -18, 14, 13]], dtype=np.float64),
        'sobel_y': np.array([[2, 10, 9, 5, 1, -3],
                             [24, 23, 18, 6, -10, -21],
                             [16, 3, -1, 0, -3, -11],
                             [-22, -15, -6, -1, 4, 12],
                             [-16, -5, 4, 0, -4, 5]], dtype=np.float64),
        'prewitt_x': np.array([[0, 5, -3, 2, 21, 11],
                               [0, 4, -7, -6, 13, 8],
                               [-2, 1, -9, -7, 7, 2],
                               [-6, 3, -6, -13, 6, 4],
                               [-4, 8, -2, -12, 10, 8]], dtype=np.float64),
        'prewitt_y': np.array([[3, 5, 8, 3, 1, -2],
                               [18, 17, 13, 4, -7, -15],
                               [10, 5, -2, 1, -4, -7],
                               [-16, -11, -5, -1, 4, 8],
                               [-11, -4, 2, -1, 0, 2]], dtype=np.float64),
        'laplacian': np.array([[-3, 10, -5, 9, 0, -5],
                               [12, -9, 2, 2, -2, -7],
                               [-10, 0, -15, 4, 9, 5],
                               [-8, 8, -1, 10, -14, 4],
                               [5, 6, -12, 3, 10, -8]], dtype=np.float64),
    }

    @pytest.mark.parametrize('func,key', [
        (sobel_x, 'sobel_x'),
        (sobel_y, 'sobel_y'),
        (prewitt_x, 'prewitt_x'),
        (prewitt_y, 'prewitt_y'),
        (laplacian, 'laplacian'),
    ])
    def test_golden_reflect_full_array(self, func, key):
        agg = create_test_raster(self.DATA, backend='numpy')
        result = func(agg, boundary='reflect')
        np.testing.assert_array_equal(result.data, self.EXPECTED[key])

    @pytest.mark.parametrize('func,key', [
        (sobel_x, 'sobel_x'),
        (sobel_y, 'sobel_y'),
        (prewitt_x, 'prewitt_x'),
        (prewitt_y, 'prewitt_y'),
        (laplacian, 'laplacian'),
    ])
    def test_golden_nan_interior(self, func, key):
        # default boundary='nan': interior pixels are unaffected by the
        # boundary mode, so they must equal the reference interior
        agg = create_test_raster(self.DATA, backend='numpy')
        result = func(agg)
        np.testing.assert_array_equal(
            result.data[1:-1, 1:-1], self.EXPECTED[key][1:-1, 1:-1])


# ---------------------------------------------------------------------------
# Output metadata / backend checks
# ---------------------------------------------------------------------------

class TestOutputMetadata:
    @pytest.mark.parametrize('func,expected_name', [
        (sobel_x, 'sobel_x'),
        (sobel_y, 'sobel_y'),
        (laplacian, 'laplacian'),
        (prewitt_x, 'prewitt_x'),
        (prewitt_y, 'prewitt_y'),
    ])
    def test_default_name(self, numpy_agg, func, expected_name):
        result = func(numpy_agg)
        assert result.name == expected_name

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_custom_name(self, numpy_agg, func):
        result = func(numpy_agg, name='custom')
        assert result.name == 'custom'

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_output_checks(self, numpy_agg, func):
        result = func(numpy_agg)
        general_output_checks(numpy_agg, result, verify_attrs=True)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNanHandling:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_nan_edges_default_boundary(self, numpy_agg, func):
        result = func(numpy_agg)
        # edges should be NaN with default boundary='nan'
        assert np.all(np.isnan(result.data[0, :]))
        assert np.all(np.isnan(result.data[-1, :]))
        assert np.all(np.isnan(result.data[:, 0]))
        assert np.all(np.isnan(result.data[:, -1]))

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_nan_in_input_propagates(self, func):
        data = np.ones((5, 6), dtype=np.float64)
        data[2, 3] = np.nan
        agg = create_test_raster(data, backend='numpy')
        result = func(agg, boundary='reflect')
        # per the docstring: every output cell whose 3x3 neighborhood
        # contains the NaN becomes NaN, and no other cell does
        expected_nan = np.zeros((5, 6), dtype=bool)
        expected_nan[1:4, 2:5] = True
        np.testing.assert_array_equal(np.isnan(result.data), expected_nan)

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_all_nan_input(self, func):
        data = np.full((5, 6), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = func(agg)
        assert np.all(np.isnan(result.data))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_single_value_raster(self, func):
        # 3x3 is the minimum for a 3x3 kernel (only 1 interior cell)
        data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = func(agg)
        assert result.shape == (3, 3)
        # only center cell is non-NaN with boundary='nan'
        assert not np.isnan(result.data[1, 1])
        # edges are NaN
        assert np.isnan(result.data[0, 0])

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_constant_field_gives_zero(self, func):
        data = np.full((7, 7), 42.0, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = func(agg)
        interior = result.data[1:-1, 1:-1]
        np.testing.assert_allclose(interior, 0, atol=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_non_dataarray_raises_type_error(self, func):
        # plain numpy array should be rejected with a clean TypeError,
        # not an AttributeError from agg.data
        with pytest.raises(TypeError, match='must be an xarray.DataArray'):
            func(np.zeros((5, 5), dtype=np.float64))

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_1d_dataarray_raises_value_error(self, func):
        # 1-D input should raise ValueError, not fail later inside the kernel
        agg = xr.DataArray(np.zeros(10, dtype=np.float64), dims=['x'])
        with pytest.raises(ValueError, match='2D'):
            func(agg)

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_3d_dataarray_raises_value_error(self, func):
        agg = xr.DataArray(np.zeros((3, 5, 6), dtype=np.float64),
                           dims=['z', 'y', 'x'])
        with pytest.raises(ValueError, match='2D'):
            func(agg)


# ---------------------------------------------------------------------------
# Boundary modes
# ---------------------------------------------------------------------------

class TestBoundaryModes:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_boundary_modes(self, func):
        data = np.random.RandomState(1038).rand(8, 10).astype(np.float64)
        np_agg = create_test_raster(data, backend='numpy')
        da_agg = create_test_raster(data, backend='dask+numpy', chunks=(8, 10))
        assert_boundary_mode_correctness(np_agg, da_agg, func)


# ---------------------------------------------------------------------------
# Docstring contract
# ---------------------------------------------------------------------------

class TestDocstrings:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_docstring_params_match_signature(self, func):
        # Every parameter documented in the "Parameters" section must exist
        # in the signature (and vice versa, in the same order).
        sig_params = list(inspect.signature(func).parameters)
        doc = inspect.getdoc(func)
        lines = doc.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.strip() == 'Parameters')
        documented = []
        for ln in lines[start + 2:]:
            if ln.strip() in ('Returns', 'References', 'Notes', 'Examples'):
                break
            m = re.match(r'^(\w+)\s*:', ln)
            if m:
                documented.append(m.group(1))
        assert documented == sig_params, (
            f'{func.__name__}: documented params {documented} != '
            f'signature params {sig_params}'
        )

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_docstring_has_examples_section(self, func):
        doc = inspect.getdoc(func)
        assert any(ln.strip() == 'Examples' for ln in doc.splitlines()), (
            f'{func.__name__} docstring has no Examples section'
        )

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, prewitt_x, prewitt_y])
    def test_docstring_states_cross_correlation(self, func):
        # convolve_2d applies kernels by cross-correlation, which for the
        # antisymmetric Sobel/Prewitt kernels differs from convolution by
        # sign. The docstrings used to say "convolving", which predicts the
        # negated result. Pin the wording so it does not come back.
        doc = inspect.getdoc(func)
        assert 'cross-correlat' in doc
        assert 'by convolving' not in doc


# ---------------------------------------------------------------------------
# Dask backend
# ---------------------------------------------------------------------------

@dask_array_available
class TestDaskNumpy:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_numpy_equals_dask(self, numpy_agg, dask_agg, func):
        assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, func)


# ---------------------------------------------------------------------------
# CuPy backend (skip if no GPU)
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
class TestCuPy:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_numpy_equals_cupy(self, ramp_data, func):
        from xrspatial.tests.general_checks import assert_numpy_equals_cupy
        np_agg = create_test_raster(ramp_data, backend='numpy')
        cu_agg = create_test_raster(ramp_data, backend='cupy')
        assert_numpy_equals_cupy(np_agg, cu_agg, func)

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_numpy_equals_dask_cupy(self, ramp_data, func):
        from xrspatial.tests.general_checks import assert_numpy_equals_dask_cupy
        np_agg = create_test_raster(ramp_data, backend='numpy')
        dcu_agg = create_test_raster(ramp_data, backend='dask+cupy', chunks=(5, 6))
        assert_numpy_equals_dask_cupy(np_agg, dcu_agg, func)


# ---------------------------------------------------------------------------
# Multi-chunk dask (#3682) -- the tests above use a single chunk equal to
# the full array, which never exercises chunk-boundary stitching in the
# map_overlap path.
# ---------------------------------------------------------------------------

@dask_array_available
class TestDaskMultiChunk:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_numpy_equals_dask_multichunk(self, func):
        data = np.random.RandomState(3682).rand(9, 11).astype(np.float64)
        np_agg = create_test_raster(data, backend='numpy')
        da_agg = create_test_raster(data, backend='dask+numpy', chunks=(4, 5))
        assert_numpy_equals_dask_numpy(np_agg, da_agg, func)

    @cuda_and_cupy_available
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_numpy_equals_dask_cupy_multichunk(self, func):
        from xrspatial.tests.general_checks import assert_numpy_equals_dask_cupy
        data = np.random.RandomState(3682).rand(9, 11).astype(np.float64)
        np_agg = create_test_raster(data, backend='numpy')
        dcu_agg = create_test_raster(data, backend='dask+cupy', chunks=(4, 5))
        assert_numpy_equals_dask_cupy(np_agg, dcu_agg, func)


# ---------------------------------------------------------------------------
# NaN at the raster edge, on every backend and boundary mode (#3682) --
# earlier NaN tests only place a NaN at an interior cell on the numpy
# backend. NaN-input backend divergence has shipped before (kde, #3628).
# ---------------------------------------------------------------------------

def _nan_edge_data():
    data = np.random.RandomState(42).rand(8, 10).astype(np.float64)
    data[0, 0] = np.nan   # corner: interacts with boundary padding
    data[3, 4] = np.nan   # interior
    return data


class TestNanAtBoundary:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    @pytest.mark.parametrize('boundary', ['nan', 'nearest', 'reflect', 'wrap'])
    def test_corner_nan_propagates(self, func, boundary):
        result = func(create_test_raster(_nan_edge_data()), boundary=boundary)
        # the corner NaN reaches its kernel neighborhood
        assert np.isnan(result.data[1, 1])
        # cells outside both NaN neighborhoods stay finite
        assert np.isfinite(result.data[6, 7])

    @dask_array_available
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    @pytest.mark.parametrize('boundary', ['nan', 'nearest', 'reflect', 'wrap'])
    def test_numpy_equals_dask_multichunk(self, func, boundary):
        data = _nan_edge_data()
        expected = func(create_test_raster(data), boundary=boundary)
        da_agg = create_test_raster(data, backend='dask+numpy', chunks=(3, 4))
        result = func(da_agg, boundary=boundary)
        np.testing.assert_allclose(
            expected.data, result.data.compute(), equal_nan=True)

    @cuda_and_cupy_available
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    @pytest.mark.parametrize('boundary', ['nan', 'nearest', 'reflect', 'wrap'])
    def test_numpy_equals_cupy(self, func, boundary):
        data = _nan_edge_data()
        expected = func(create_test_raster(data), boundary=boundary)
        cu_agg = create_test_raster(data, backend='cupy')
        result = func(cu_agg, boundary=boundary)
        np.testing.assert_allclose(
            expected.data, result.data.get(), equal_nan=True)

    @cuda_and_cupy_available
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    @pytest.mark.parametrize('boundary', ['nan', 'nearest', 'reflect', 'wrap'])
    def test_numpy_equals_dask_cupy(self, func, boundary):
        data = _nan_edge_data()
        expected = func(create_test_raster(data), boundary=boundary)
        dcu_agg = create_test_raster(data, backend='dask+cupy', chunks=(3, 4))
        result = func(dcu_agg, boundary=boundary)
        np.testing.assert_allclose(
            expected.data, result.data.compute().get(), equal_nan=True)


# ---------------------------------------------------------------------------
# Inf handling (#3682) -- an inf cell propagates +/-inf where the kernel
# weight has a single sign and NaN where inf meets -inf (or a zero weight).
# ---------------------------------------------------------------------------

# per-operator expected 3x3 neighborhood around a +inf cell in a field of
# ones: +/-inf where the kernel weight over the inf cell has a single sign,
# NaN where the weight is zero (0 * inf) or opposing infs meet.
INF_NEIGHBORHOODS = [
    (sobel_x, [[np.inf, np.nan, -np.inf]] * 3),
    (sobel_y, [[np.inf] * 3, [np.nan] * 3, [-np.inf] * 3]),
    (prewitt_x, [[np.inf, np.nan, -np.inf]] * 3),
    (prewitt_y, [[np.inf] * 3, [np.nan] * 3, [-np.inf] * 3]),
    (laplacian, [[np.nan, np.inf, np.nan],
                 [np.inf, -np.inf, np.inf],
                 [np.nan, np.inf, np.nan]]),
]


class TestInfHandling:
    @pytest.mark.parametrize('func,expected_block', INF_NEIGHBORHOODS)
    def test_inf_neighborhood(self, func, expected_block):
        data = np.ones((5, 6), dtype=np.float64)
        data[2, 3] = np.inf
        result = func(create_test_raster(data), boundary='reflect')
        # 3x3 neighborhood centred on the inf cell
        np.testing.assert_array_equal(result.data[1:4, 2:5], expected_block)
        # cells outside the neighborhood are unaffected: constant field -> 0
        assert result.data[0, 0] == 0
        assert result.data[4, 0] == 0

    @pytest.mark.parametrize('func,expected_block', INF_NEIGHBORHOODS)
    def test_negative_inf(self, func, expected_block):
        data = np.ones((5, 6), dtype=np.float64)
        data[2, 3] = -np.inf
        result = func(create_test_raster(data), boundary='reflect')
        # a -inf cell produces the +inf expected block negated (NaN unchanged)
        np.testing.assert_array_equal(
            result.data[1:4, 2:5], np.negative(expected_block))
        assert result.data[0, 0] == 0


# ---------------------------------------------------------------------------
# Degenerate shapes (#3682) -- 1x1, strips, and empty rasters.
# ---------------------------------------------------------------------------

class TestDegenerateShapes:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_1x1_nan_boundary(self, func):
        agg = create_test_raster(np.array([[5.0]]))
        result = func(agg)
        assert result.shape == (1, 1)
        assert np.isnan(result.data[0, 0])

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_1x1_reflect(self, func):
        agg = create_test_raster(np.array([[5.0]]))
        result = func(agg, boundary='reflect')
        # reflected padding makes the neighborhood constant -> zero response
        np.testing.assert_array_equal(result.data, [[0.0]])

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_row_strip(self, func):
        data = np.arange(6, dtype=np.float64).reshape(1, 6)
        result_nan = func(create_test_raster(data))
        assert result_nan.shape == (1, 6)
        assert np.all(np.isnan(result_nan.data))
        result_reflect = func(create_test_raster(data), boundary='reflect')
        assert not np.any(np.isnan(result_reflect.data))

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_column_strip(self, func):
        data = np.arange(6, dtype=np.float64).reshape(6, 1)
        result = func(create_test_raster(data), boundary='reflect')
        assert result.shape == (6, 1)
        assert not np.any(np.isnan(result.data))

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_empty_raster(self, func):
        agg = xr.DataArray(np.zeros((0, 5), dtype=np.float64), dims=['y', 'x'])
        result = func(agg)
        assert result.shape == (0, 5)


# ---------------------------------------------------------------------------
# Invalid boundary and dim-name propagation (#3682)
# ---------------------------------------------------------------------------

class TestBoundaryValidation:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_invalid_boundary_raises(self, func):
        agg = create_test_raster(np.ones((5, 6), dtype=np.float64))
        with pytest.raises(ValueError, match='boundary must be one of'):
            func(agg, boundary='invalid')


class TestDimNamePropagation:
    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    def test_custom_dims_preserved(self, func):
        agg = create_test_raster(np.ones((5, 6), dtype=np.float64),
                                 dims=['lat', 'lon'])
        result = func(agg)
        assert result.dims == ('lat', 'lon')
        np.testing.assert_allclose(result['lat'].data, agg['lat'].data)
        np.testing.assert_allclose(result['lon'].data, agg['lon'].data)


# ---------------------------------------------------------------------------
# Wide-integer precision (issue-3680)
# ---------------------------------------------------------------------------
# convolve_2d promotes integer inputs to float32, whose 24-bit mantissa
# cannot separate integers above 2**24, so unit-step gradients on large
# int32/int64 values silently collapsed to zero. edge_detection now
# pre-promotes 32/64-bit integers to float64.


def _to_numpy(data):
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return data


class TestWideIntegerPrecision:
    @pytest.fixture
    def wide_int32_data(self):
        # unit-step ramp on an offset beyond float32's 24-bit mantissa
        return (np.arange(30).reshape(5, 6) % 6 + 100_000_000).astype(np.int32)

    @pytest.mark.parametrize('func', [sobel_x, sobel_y, laplacian, prewitt_x, prewitt_y])
    @pytest.mark.parametrize('backend', ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
    def test_large_int32_matches_float64_reference(self, wide_int32_data, func, backend):
        from xrspatial.tests.general_checks import has_cuda_and_cupy, has_dask_array
        if 'cupy' in backend and not has_cuda_and_cupy():
            pytest.skip("Requires CUDA and CuPy")
        if 'dask' in backend and not has_dask_array():
            pytest.skip("Requires Dask")
        ref = func(create_test_raster(
            wide_int32_data.astype(np.float64), backend='numpy'))
        agg = create_test_raster(wide_int32_data, backend=backend, chunks=(5, 6))
        result = _to_numpy(func(agg).data)
        np.testing.assert_allclose(result, ref.data, equal_nan=True)

    def test_large_int32_gradient_nonzero(self, wide_int32_data):
        # regression guard for the exact silent failure from issue-3680:
        # the old float32 path returned 0.0 at every interior cell
        result = sobel_x(create_test_raster(wide_int32_data, backend='numpy'))
        interior = result.data[1:-1, 1:-1]
        assert np.all(interior != 0)

    def test_large_int64_matches_float64_reference(self):
        data = (np.arange(30).reshape(5, 6) % 6
                + 10_000_000_000).astype(np.int64)
        ref = sobel_x(create_test_raster(data.astype(np.float64),
                                         backend='numpy'))
        result = sobel_x(create_test_raster(data, backend='numpy'))
        np.testing.assert_allclose(result.data, ref.data, equal_nan=True)

    @pytest.mark.parametrize('dtype,expected', [
        (np.int8, np.float32),
        (np.uint8, np.float32),
        (np.int16, np.float32),
        (np.uint16, np.float32),
        (np.int32, np.float64),
        (np.uint32, np.float64),
        (np.int64, np.float64),
        (np.uint64, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ])
    def test_output_dtype_by_input_dtype(self, dtype, expected):
        data = np.ones((5, 6), dtype=dtype)
        result = laplacian(create_test_raster(data, backend='numpy'))
        assert result.dtype == expected
