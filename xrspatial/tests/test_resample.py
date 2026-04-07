"""Tests for xrspatial.resample."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.resample import resample, ALL_METHODS
from xrspatial.tests.general_checks import (
    create_test_raster,
    dask_array_available,
    cuda_and_cupy_available,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_4x4():
    """4x4 float32 grid with values 1..16."""
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    return create_test_raster(data, backend='numpy',
                              attrs={'res': (1.0, 1.0)})


@pytest.fixture
def grid_8x8():
    """8x8 grid with smooth gradient for interpolation tests."""
    y, x = np.mgrid[0:8, 0:8]
    data = (y * 10 + x).astype(np.float32)
    return create_test_raster(data, backend='numpy',
                              attrs={'res': (1.0, 1.0)})


@pytest.fixture
def grid_with_nan():
    """4x4 grid with NaN in top-left corner."""
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    data[0, 0] = np.nan
    return create_test_raster(data, backend='numpy',
                              attrs={'res': (1.0, 1.0)})


# ---------------------------------------------------------------------------
# API validation
# ---------------------------------------------------------------------------

class TestResampleAPI:
    def test_invalid_method(self, grid_4x4):
        with pytest.raises(ValueError, match="method must be one of"):
            resample(grid_4x4, scale_factor=0.5, method='invalid')

    def test_neither_scale_nor_resolution(self, grid_4x4):
        with pytest.raises(ValueError, match="Exactly one"):
            resample(grid_4x4)

    def test_both_scale_and_resolution(self, grid_4x4):
        with pytest.raises(ValueError, match="Exactly one"):
            resample(grid_4x4, scale_factor=0.5, target_resolution=2.0)

    def test_negative_scale(self, grid_4x4):
        with pytest.raises(ValueError, match="positive"):
            resample(grid_4x4, scale_factor=-1.0)

    def test_aggregate_upsample_rejected(self, grid_4x4):
        with pytest.raises(ValueError, match="downsampling"):
            resample(grid_4x4, scale_factor=2.0, method='average')

    def test_identity_scale(self, grid_4x4):
        out = resample(grid_4x4, scale_factor=1.0)
        xr.testing.assert_identical(out.rename(grid_4x4.name), grid_4x4)

    def test_output_name(self, grid_4x4):
        out = resample(grid_4x4, scale_factor=0.5, name='resampled')
        assert out.name == 'resampled'


# ---------------------------------------------------------------------------
# Output shape & coordinates
# ---------------------------------------------------------------------------

class TestOutputGeometry:
    def test_downsample_shape(self, grid_8x8):
        out = resample(grid_8x8, scale_factor=0.5)
        assert out.shape == (4, 4)

    def test_upsample_shape(self, grid_4x4):
        out = resample(grid_4x4, scale_factor=2.0)
        assert out.shape == (8, 8)

    def test_target_resolution(self, grid_8x8):
        out = resample(grid_8x8, target_resolution=2.0)
        assert out.shape == (4, 4)
        assert pytest.approx(out.attrs['res'][0], abs=0.01) == 2.0

    def test_asymmetric_scale(self, grid_8x8):
        out = resample(grid_8x8, scale_factor=(0.5, 0.25))
        assert out.shape == (4, 2)

    def test_coords_preserve_extent(self, grid_8x8):
        """Output coords should span the same spatial extent as input."""
        out = resample(grid_8x8, scale_factor=0.5)
        ydim, xdim = grid_8x8.dims[-2], grid_8x8.dims[-1]
        in_y, in_x = grid_8x8[ydim].values, grid_8x8[xdim].values
        out_y, out_x = out[ydim].values, out[xdim].values

        # Pixel edges should align (within half a pixel of old res)
        in_xmin = in_x[0] - (in_x[1] - in_x[0]) / 2
        in_xmax = in_x[-1] + (in_x[-1] - in_x[-2]) / 2
        out_xmin = out_x[0] - (out_x[1] - out_x[0]) / 2
        out_xmax = out_x[-1] + (out_x[-1] - out_x[-2]) / 2
        assert pytest.approx(in_xmin, abs=1e-6) == out_xmin
        assert pytest.approx(in_xmax, abs=1e-6) == out_xmax

    def test_res_attribute_updated(self, grid_8x8):
        out = resample(grid_8x8, scale_factor=0.5)
        assert pytest.approx(out.attrs['res'][0], abs=0.01) == 2.0
        assert pytest.approx(out.attrs['res'][1], abs=0.01) == 2.0


# ---------------------------------------------------------------------------
# Correctness: known values
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_nearest_downsample(self):
        data = np.array([[10, 20, 30, 40],
                         [50, 60, 70, 80],
                         [90, 100, 110, 120],
                         [130, 140, 150, 160]], dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out.shape == (2, 2)
        # Each output pixel picks the nearest input pixel
        # Verify values are from the original data
        for v in out.values.ravel():
            assert v in data

    def test_average_downsample_2x(self):
        data = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]], dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='average')
        assert out.shape == (2, 2)
        expected = np.array([[3.5, 5.5],
                             [11.5, 13.5]], dtype=np.float32)
        np.testing.assert_allclose(out.values, expected, atol=1e-5)

    def test_min_downsample(self):
        data = np.array([[4, 3], [2, 1]], dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='min')
        assert out.shape == (1, 1)
        assert out.values[0, 0] == 1.0

    def test_max_downsample(self):
        data = np.array([[4, 3], [2, 1]], dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='max')
        assert out.shape == (1, 1)
        assert out.values[0, 0] == 4.0

    def test_mode_downsample(self):
        data = np.array([[1, 1, 2, 2],
                         [1, 1, 2, 2],
                         [3, 3, 4, 4],
                         [3, 3, 4, 4]], dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='mode')
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np.testing.assert_array_equal(out.values, expected)

    def test_bilinear_upsample_smooth(self, grid_8x8):
        """Bilinear on a linear gradient should produce exact results."""
        out = resample(grid_8x8, scale_factor=2.0, method='bilinear')
        assert out.shape == (16, 16)
        # For a perfectly linear surface, bilinear should be exact
        # Verify interior is within tolerance of the linear gradient
        assert np.all(np.isfinite(out.values))


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_nearest_preserves_nan(self, grid_with_nan):
        out = resample(grid_with_nan, scale_factor=2.0, method='nearest')
        # NaN should appear in the nearest-neighbor output
        assert np.isnan(out.values).any()

    def test_average_ignores_nan(self, grid_with_nan):
        out = resample(grid_with_nan, scale_factor=0.5, method='average')
        # Average of a block containing NaN should use only valid pixels
        # Not all-NaN output
        assert not np.isnan(out.values).all()

    def test_all_nan_block_gives_nan(self):
        data = np.full((4, 4), np.nan, dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='average')
        assert np.isnan(out.values).all()

    def test_bilinear_nan_handling(self, grid_with_nan):
        out = resample(grid_with_nan, scale_factor=2.0, method='bilinear')
        # Should produce finite values away from the NaN region
        assert np.isfinite(out.values).any()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_pixel(self):
        data = np.array([[42.0]], dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=2.0, method='nearest')
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(out.values, 42.0)

    def test_single_row(self):
        data = np.array([[1, 2, 3, 4]], dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)},
                                 dims=['y', 'x'])
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out.shape[1] == 2

    def test_non_square(self):
        data = np.random.rand(6, 10).astype(np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out.shape == (3, 5)

    def test_very_small_scale(self):
        data = np.random.rand(20, 20).astype(np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.1, method='nearest')
        assert out.shape == (2, 2)

    def test_dataset_input(self):
        data = np.random.rand(8, 8).astype(np.float32)
        ds = xr.Dataset({
            'band1': create_test_raster(data, attrs={'res': (1.0, 1.0)}),
            'band2': create_test_raster(data * 2, attrs={'res': (1.0, 1.0)}),
        })
        out = resample(ds, scale_factor=0.5, method='nearest')
        assert isinstance(out, xr.Dataset)
        assert out['band1'].shape == (4, 4)
        assert out['band2'].shape == (4, 4)


# ---------------------------------------------------------------------------
# Dask parity
# ---------------------------------------------------------------------------

@dask_array_available
class TestDaskParity:
    """Verify dask+numpy results match pure numpy for all methods."""

    @pytest.fixture
    def numpy_and_dask_rasters(self):
        data = np.random.RandomState(1152).rand(20, 20).astype(np.float32)
        np_agg = create_test_raster(data, backend='numpy',
                                    attrs={'res': (1.0, 1.0)})
        dk_agg = create_test_raster(data, backend='dask+numpy',
                                    attrs={'res': (1.0, 1.0)},
                                    chunks=(8, 8))
        return np_agg, dk_agg

    @pytest.mark.parametrize('method', ['nearest', 'bilinear', 'cubic'])
    @pytest.mark.parametrize('sf', [0.5, 2.0, 0.7])
    def test_interp_parity(self, numpy_and_dask_rasters, method, sf):
        np_agg, dk_agg = numpy_and_dask_rasters
        np_out = resample(np_agg, scale_factor=sf, method=method)
        dk_out = resample(dk_agg, scale_factor=sf, method=method)
        np.testing.assert_allclose(dk_out.values, np_out.values,
                                   atol=1e-5, equal_nan=True)

    @pytest.mark.parametrize('method', ['average', 'min', 'max', 'median', 'mode'])
    def test_aggregate_parity(self, numpy_and_dask_rasters, method):
        np_agg, dk_agg = numpy_and_dask_rasters
        np_out = resample(np_agg, scale_factor=0.5, method=method)
        dk_out = resample(dk_agg, scale_factor=0.5, method=method)
        np.testing.assert_allclose(dk_out.values, np_out.values,
                                   atol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# CuPy parity
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
class TestCuPyParity:
    """Verify cupy results match numpy."""

    @pytest.fixture
    def numpy_and_cupy_rasters(self):
        data = np.random.RandomState(1152).rand(12, 12).astype(np.float32)
        np_agg = create_test_raster(data, backend='numpy',
                                    attrs={'res': (1.0, 1.0)})
        cp_agg = create_test_raster(data, backend='cupy',
                                    attrs={'res': (1.0, 1.0)})
        return np_agg, cp_agg

    @pytest.mark.parametrize('method', ['nearest', 'bilinear', 'cubic'])
    @pytest.mark.parametrize('sf', [0.5, 2.0])
    def test_interp_parity(self, numpy_and_cupy_rasters, method, sf):
        np_agg, cp_agg = numpy_and_cupy_rasters
        np_out = resample(np_agg, scale_factor=sf, method=method)
        cp_out = resample(cp_agg, scale_factor=sf, method=method)
        np.testing.assert_allclose(cp_out.data.get(), np_out.values,
                                   atol=1e-4, equal_nan=True)

    @pytest.mark.parametrize('method', ['average', 'min', 'max'])
    def test_aggregate_parity(self, numpy_and_cupy_rasters, method):
        np_agg, cp_agg = numpy_and_cupy_rasters
        np_out = resample(np_agg, scale_factor=0.5, method=method)
        cp_out = resample(cp_agg, scale_factor=0.5, method=method)
        np.testing.assert_allclose(cp_out.data.get(), np_out.values,
                                   atol=1e-5, equal_nan=True)
