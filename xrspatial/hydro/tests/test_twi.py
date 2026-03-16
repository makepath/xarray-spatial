"""Tests for xrspatial.twi (Topographic Wetness Index)."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.twi import twi
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_twi_rasters(flow_accum_data, slope_data, backend='numpy',
                      chunks=(3, 3), res=(1.0, 1.0)):
    """Create paired flow_accum and slope DataArrays for testing."""
    attrs = {'res': res}
    fa = create_test_raster(flow_accum_data, backend=backend, name='fa',
                            attrs=attrs, chunks=chunks)
    sl = create_test_raster(slope_data, backend=backend, name='slope',
                            attrs=attrs, chunks=chunks)
    return fa, sl


class TestTWIKnownValues:
    """Test TWI against hand-computed values."""

    def test_known_values(self):
        # 3x3 grid: flow_accum=10, slope=45°
        # cellsize = sqrt(1*1) = 1
        # sca = 10 * 1 = 10
        # tan(45°) = 1.0
        # TWI = ln(10 / 1) = ln(10) ≈ 2.3026
        fa_data = np.full((3, 3), 10.0, dtype=np.float64)
        sl_data = np.full((3, 3), 45.0, dtype=np.float64)
        fa, sl = _make_twi_rasters(fa_data, sl_data)
        result = twi(fa, sl)
        expected = np.log(10.0 / np.tan(np.radians(45.0)))
        np.testing.assert_allclose(result.data, expected, rtol=1e-10)

    def test_varying_slope(self):
        # Single row: accum=100, slopes=[1, 10, 30, 45, 89]
        fa_data = np.full((1, 5), 100.0, dtype=np.float64)
        sl_data = np.array([[1.0, 10.0, 30.0, 45.0, 89.0]], dtype=np.float64)
        fa, sl = _make_twi_rasters(fa_data, sl_data)
        result = twi(fa, sl)
        for i, s in enumerate([1.0, 10.0, 30.0, 45.0, 89.0]):
            expected = np.log(100.0 / np.tan(np.radians(s)))
            np.testing.assert_allclose(result.data[0, i], expected, rtol=1e-10)


class TestTWIFlatSlopeClamp:
    """Verify no inf/NaN when slope is zero."""

    def test_flat_slope_clamp(self):
        fa_data = np.full((3, 3), 50.0, dtype=np.float64)
        sl_data = np.zeros((3, 3), dtype=np.float64)
        fa, sl = _make_twi_rasters(fa_data, sl_data)
        result = twi(fa, sl)
        assert not np.any(np.isinf(result.data))
        assert not np.any(np.isnan(result.data))
        # Should be ln(50 / tan(0.001°))
        tan_min = np.tan(np.radians(0.001))
        expected = np.log(50.0 / tan_min)
        np.testing.assert_allclose(result.data, expected, rtol=1e-10)

    def test_very_small_slope(self):
        fa_data = np.full((2, 2), 1.0, dtype=np.float64)
        sl_data = np.full((2, 2), 1e-10, dtype=np.float64)
        fa, sl = _make_twi_rasters(fa_data, sl_data)
        result = twi(fa, sl)
        assert not np.any(np.isinf(result.data))
        assert not np.any(np.isnan(result.data))


class TestTWINanPropagation:
    """NaN in either input should produce NaN output."""

    def test_nan_in_flow_accum(self):
        fa_data = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
        sl_data = np.full((2, 2), 10.0, dtype=np.float64)
        fa, sl = _make_twi_rasters(fa_data, sl_data)
        result = twi(fa, sl)
        assert np.isnan(result.data[0, 1])
        assert not np.isnan(result.data[0, 0])

    def test_nan_in_slope(self):
        fa_data = np.full((2, 2), 10.0, dtype=np.float64)
        sl_data = np.array([[10.0, 10.0], [np.nan, 10.0]], dtype=np.float64)
        fa, sl = _make_twi_rasters(fa_data, sl_data)
        result = twi(fa, sl)
        assert np.isnan(result.data[1, 0])
        assert not np.isnan(result.data[0, 0])

    def test_nan_in_both(self):
        fa_data = np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float64)
        sl_data = np.array([[10.0, np.nan], [10.0, 10.0]], dtype=np.float64)
        fa, sl = _make_twi_rasters(fa_data, sl_data)
        result = twi(fa, sl)
        assert np.isnan(result.data[0, 0])
        assert np.isnan(result.data[0, 1])
        assert not np.isnan(result.data[1, 1])


class TestTWINonSquareCells:
    """Test with non-square cell sizes."""

    def test_rectangular_cells(self):
        fa_data = np.full((3, 3), 10.0, dtype=np.float64)
        sl_data = np.full((3, 3), 45.0, dtype=np.float64)
        fa, sl = _make_twi_rasters(fa_data, sl_data, res=(2.0, 3.0))
        result = twi(fa, sl)
        cellsize = np.sqrt(2.0 * 3.0)
        expected = np.log(10.0 * cellsize / np.tan(np.radians(45.0)))
        np.testing.assert_allclose(result.data, expected, rtol=1e-10)


@dask_array_available
class TestTWIDask:
    """Cross-backend: numpy vs dask."""

    @pytest.mark.parametrize("chunks", [
        (2, 2), (3, 3), (1, 5), (5, 1), (5, 5),
    ])
    def test_numpy_equals_dask(self, chunks):
        np.random.seed(42)
        fa_data = np.random.uniform(1, 1000, (5, 5)).astype(np.float64)
        sl_data = np.random.uniform(0.1, 60, (5, 5)).astype(np.float64)

        fa_np, sl_np = _make_twi_rasters(fa_data, sl_data, backend='numpy')
        fa_da, sl_da = _make_twi_rasters(fa_data, sl_data, backend='dask',
                                          chunks=chunks)

        result_np = twi(fa_np, sl_np)
        result_da = twi(fa_da, sl_da)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
class TestTWICuPy:
    """Cross-backend: numpy vs cupy."""

    def test_numpy_equals_cupy(self):
        np.random.seed(42)
        fa_data = np.random.uniform(1, 1000, (5, 5)).astype(np.float64)
        sl_data = np.random.uniform(0.1, 60, (5, 5)).astype(np.float64)

        fa_np, sl_np = _make_twi_rasters(fa_data, sl_data, backend='numpy')
        fa_cu, sl_cu = _make_twi_rasters(fa_data, sl_data, backend='cupy')

        result_np = twi(fa_np, sl_np)
        result_cu = twi(fa_cu, sl_cu)

        np.testing.assert_allclose(
            result_np.data, result_cu.data.get(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
@dask_array_available
class TestTWIDaskCuPy:
    """Cross-backend: numpy vs dask+cupy."""

    def test_numpy_equals_dask_cupy(self):
        np.random.seed(42)
        fa_data = np.random.uniform(1, 1000, (5, 5)).astype(np.float64)
        sl_data = np.random.uniform(0.1, 60, (5, 5)).astype(np.float64)

        fa_np, sl_np = _make_twi_rasters(fa_data, sl_data, backend='numpy')
        fa_dc, sl_dc = _make_twi_rasters(fa_data, sl_data, backend='dask+cupy',
                                          chunks=(3, 3))

        result_np = twi(fa_np, sl_np)
        result_dc = twi(fa_dc, sl_dc)

        np.testing.assert_allclose(
            result_np.data, result_dc.data.compute().get(),
            equal_nan=True, rtol=1e-10)
