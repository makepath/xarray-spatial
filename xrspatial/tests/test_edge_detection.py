import numpy as np
import pytest
import xarray as xr

from xrspatial.edge_detection import (
    LAPLACIAN_KERNEL, PREWITT_X, PREWITT_Y, SOBEL_X, SOBEL_Y,
    laplacian, prewitt_x, prewitt_y, sobel_x, sobel_y,
)
from xrspatial.tests.general_checks import (
    assert_boundary_mode_correctness,
    assert_numpy_equals_dask_numpy,
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)

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
        # NaN should propagate to neighbors
        assert np.isnan(result.data[2, 3])

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
