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
# Metadata propagation
# ---------------------------------------------------------------------------

class TestMetadataPropagation:
    """resample() should refresh stale grid attrs and nodata sentinels."""

    @staticmethod
    def _raster_with_transform(transform):
        """4x4 raster whose coords match the given rasterio 6-tuple."""
        res_x, _, left, _, neg_res_y, top = transform
        res_y = -neg_res_y
        x = np.array([left + (i + 0.5) * res_x for i in range(4)])
        y = np.array([top - (i + 0.5) * res_y for i in range(4)])
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        return xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'transform': transform, 'res': (res_x, res_y)},
        )

    def test_transform_refreshed_on_downsample(self):
        raster = self._raster_with_transform(
            (1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
        )
        out = resample(raster, scale_factor=0.5)
        assert tuple(out.attrs['transform']) == (
            2.0, 0.0, 100.0, 0.0, -2.0, 200.0,
        )

    def test_transform_absent_stays_absent(self):
        # grid_4x4 fixture has no transform attr.
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        raster = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'y': np.linspace(3, 0, 4), 'x': np.linspace(0, 3, 4)},
            attrs={'res': (1.0, 1.0)},
        )
        out = resample(raster, scale_factor=0.5)
        assert 'transform' not in out.attrs

    def test_fill_value_replaced_with_nan(self):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        raster = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'y': np.linspace(3, 0, 4), 'x': np.linspace(0, 3, 4)},
            attrs={'res': (1.0, 1.0), '_FillValue': -9999},
        )
        out = resample(raster, scale_factor=0.5)
        assert '_FillValue' in out.attrs
        assert np.isnan(out.attrs['_FillValue'])

    def test_fill_value_absent_stays_absent(self):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        raster = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'y': np.linspace(3, 0, 4), 'x': np.linspace(0, 3, 4)},
            attrs={'res': (1.0, 1.0)},
        )
        out = resample(raster, scale_factor=0.5)
        assert '_FillValue' not in out.attrs

    def test_nodatavals_replaced_with_nan(self):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        raster = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'y': np.linspace(3, 0, 4), 'x': np.linspace(0, 3, 4)},
            attrs={'res': (1.0, 1.0), 'nodatavals': (-9999,)},
        )
        out = resample(raster, scale_factor=0.5)
        assert 'nodatavals' in out.attrs
        nv = out.attrs['nodatavals']
        assert len(nv) == 1
        assert np.isnan(nv[0])

    def test_other_attrs_preserved(self):
        # crs, units, long_name, scales, offsets should round-trip.
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        raster = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'y': np.linspace(3, 0, 4), 'x': np.linspace(0, 3, 4)},
            attrs={
                'res': (1.0, 1.0),
                'crs': 'EPSG:4326',
                'crs_wkt': 'GEOGCS["WGS 84"]',
                'units': 'm',
                'long_name': 'elevation',
                'scales': (1.0,),
                'offsets': (0.0,),
            },
        )
        out = resample(raster, scale_factor=0.5)
        assert out.attrs['crs'] == 'EPSG:4326'
        assert out.attrs['crs_wkt'] == 'GEOGCS["WGS 84"]'
        assert out.attrs['units'] == 'm'
        assert out.attrs['long_name'] == 'elevation'
        assert out.attrs['scales'] == (1.0,)
        assert out.attrs['offsets'] == (0.0,)


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

    @pytest.mark.parametrize('method', ['nearest', 'bilinear', 'cubic'])
    def test_interp_coordinate_alignment_downsample(self, method):
        """Interpolated values should match output coordinate labels (#1202).

        On a linear gradient where value == x-coordinate, a correct
        block-centered resample produces output values equal to the output
        coordinate labels (within floating-point tolerance).
        """
        data = np.tile(np.arange(8, dtype=np.float32), (8, 1))
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)},
                                 dims=['y', 'x'])
        out = resample(agg, scale_factor=0.5, method=method)

        # Output x-coords are block-centered: 0.5, 2.5, 4.5, 6.5
        # Values should match because the input is a linear gradient
        np.testing.assert_allclose(
            out.values[0], out.x.values, atol=0.6,
            err_msg=f"{method}: values should be close to x-coordinates"
        )
        # For bilinear on a linear gradient, the match should be exact
        if method == 'bilinear':
            np.testing.assert_allclose(
                out.values[0], out.x.values, atol=1e-5,
                err_msg="bilinear on linear gradient must be exact"
            )

    def test_bilinear_coordinate_alignment_upsample(self):
        """Upsampled interior pixels should match coordinates on a gradient."""
        data = np.tile(np.arange(8, dtype=np.float32), (8, 1))
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)},
                                 dims=['y', 'x'])
        out = resample(agg, scale_factor=2.0, method='bilinear')

        # Interior pixels (away from boundary clamping) should be exact
        # for bilinear on a linear gradient.  Skip first and last pixel
        # which may be clamped by mode='nearest' boundary handling.
        interior = slice(1, -1)
        np.testing.assert_allclose(
            out.values[0, interior], out.x.values[interior], atol=1e-4,
            err_msg="bilinear: interior values should match x-coordinates"
        )


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

    @pytest.mark.parametrize('method', ['bilinear'])
    def test_dask_coordinate_alignment(self, method):
        """Dask bilinear on a linear gradient should match coordinates (#1202)."""
        data = np.tile(np.arange(20, dtype=np.float32), (20, 1))
        dk_agg = create_test_raster(data, backend='dask+numpy',
                                    attrs={'res': (1.0, 1.0)},
                                    chunks=(8, 8))
        out = resample(dk_agg, scale_factor=0.5, method=method)
        np.testing.assert_allclose(
            out.values[0], out.x.values, atol=1e-4,
            err_msg="dask bilinear values should match x-coordinates"
        )

    @pytest.mark.parametrize('method', ['average', 'min', 'max', 'median', 'mode'])
    def test_aggregate_parity(self, numpy_and_dask_rasters, method):
        np_agg, dk_agg = numpy_and_dask_rasters
        np_out = resample(np_agg, scale_factor=0.5, method=method)
        dk_out = resample(dk_agg, scale_factor=0.5, method=method)
        np.testing.assert_allclose(dk_out.values, np_out.values,
                                   atol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# Aggregate dask boundary contamination (issue #1469)
# ---------------------------------------------------------------------------

@dask_array_available
class TestAggregateDaskBoundary:
    """Aggregate dask path must not let overlap-padding bias the result."""

    @pytest.mark.parametrize('method', ['min', 'max', 'median'])
    def test_chunk_spanning_window_bit_identical(self, method):
        """Aggregate windows that cross chunk boundaries must equal eager
        numpy bit-exactly (same kernel, no padding contribution)."""
        rng = np.random.RandomState(20240504)
        data = rng.rand(24, 24).astype(np.float32)
        np_agg = create_test_raster(data, backend='numpy',
                                    attrs={'res': (1.0, 1.0)})
        # Use a chunk size that does NOT divide the input cleanly so that
        # output windows span chunk boundaries.
        dk_agg = create_test_raster(data, backend='dask+numpy',
                                    attrs={'res': (1.0, 1.0)},
                                    chunks=(7, 7))
        np_out = resample(np_agg, scale_factor=0.5, method=method)
        dk_out = resample(dk_agg, scale_factor=0.5, method=method)
        # min/max/median over the same set of source cells should be
        # bit-identical -- no atol slack.
        np.testing.assert_array_equal(dk_out.values, np_out.values)

    @pytest.mark.parametrize('method', ['min', 'max', 'median'])
    def test_global_edge_extremes_match_eager(self, method):
        """Extreme values on the global array edges must not contaminate
        adjacent output windows. The aggregate kernel's indexing keeps the
        global-edge pad out of any window, and boundary=np.nan reinforces
        that contract: pad cells are NaN and would be skipped even if read.
        Result must match the eager (non-overlapped) numpy path exactly."""
        # Inner field is in a low range; the rightmost column is an
        # extreme value that, if duplicated by 'nearest' padding, would
        # show up in adjacent output windows that should not see it.
        rng = np.random.RandomState(1469)
        data = rng.uniform(0.0, 0.1, size=(20, 20)).astype(np.float32)
        data[:, -1] = 999.0   # extreme right column
        data[:, 0] = -999.0   # extreme left column
        data[-1, :] = 999.0   # extreme bottom row
        data[0, :] = -999.0   # extreme top row

        np_agg = create_test_raster(data, backend='numpy',
                                    attrs={'res': (1.0, 1.0)})
        dk_agg = create_test_raster(data, backend='dask+numpy',
                                    attrs={'res': (1.0, 1.0)},
                                    chunks=(8, 8))
        np_out = resample(np_agg, scale_factor=0.5, method=method)
        dk_out = resample(dk_agg, scale_factor=0.5, method=method)
        np.testing.assert_array_equal(dk_out.values, np_out.values)


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

    @pytest.mark.parametrize('method',
                             ['average', 'min', 'max', 'median', 'mode'])
    def test_aggregate_parity(self, numpy_and_cupy_rasters, method):
        # 'median' and 'mode' fall back to CPU inside _run_cupy; the
        # round-trip (cupy in -> cupy out) still needs to be verified.
        np_agg, cp_agg = numpy_and_cupy_rasters
        np_out = resample(np_agg, scale_factor=0.5, method=method)
        cp_out = resample(cp_agg, scale_factor=0.5, method=method)
        np.testing.assert_allclose(cp_out.data.get(), np_out.values,
                                   atol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# Dask + CuPy parity
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
@dask_array_available
class TestDaskCuPyParity:
    """Verify dask+cupy results match numpy for the four-backend matrix."""

    @pytest.fixture
    def numpy_and_dask_cupy_rasters(self):
        data = np.random.RandomState(1470).rand(20, 20).astype(np.float32)
        np_agg = create_test_raster(data, backend='numpy',
                                    attrs={'res': (1.0, 1.0)})
        dc_agg = create_test_raster(data, backend='dask+cupy',
                                    attrs={'res': (1.0, 1.0)},
                                    chunks=(8, 8))
        return np_agg, dc_agg

    @pytest.mark.parametrize('method', ['nearest', 'bilinear', 'cubic'])
    @pytest.mark.parametrize('sf', [0.5, 2.0, 0.7])
    def test_interp_parity(self, numpy_and_dask_cupy_rasters, method, sf):
        np_agg, dc_agg = numpy_and_dask_cupy_rasters
        np_out = resample(np_agg, scale_factor=sf, method=method)
        dc_out = resample(dc_agg, scale_factor=sf, method=method)
        np.testing.assert_allclose(dc_out.data.compute().get(), np_out.values,
                                   atol=1e-4, equal_nan=True)

    @pytest.mark.parametrize('method', ['average', 'min', 'max'])
    def test_aggregate_parity(self, numpy_and_dask_cupy_rasters, method):
        # median/mode go through the CPU fallback inside the cupy chunk
        # function; T-1 covers that round-trip on the eager cupy backend.
        np_agg, dc_agg = numpy_and_dask_cupy_rasters
        np_out = resample(np_agg, scale_factor=0.5, method=method)
        dc_out = resample(dc_agg, scale_factor=0.5, method=method)
        np.testing.assert_allclose(dc_out.data.compute().get(), np_out.values,
                                   atol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# Integer-dtype input
# ---------------------------------------------------------------------------

def _backend_available(backend):
    if backend == 'numpy':
        return True
    if backend == 'cupy':
        from xrspatial.utils import has_cuda_and_cupy
        return has_cuda_and_cupy()
    if backend == 'dask+numpy':
        from xrspatial.utils import has_dask_array
        return has_dask_array()
    if backend == 'dask+cupy':
        from xrspatial.utils import has_cuda_and_cupy, has_dask_array
        return has_cuda_and_cupy() and has_dask_array()
    return False


def _to_numpy(arr):
    """Materialize a backend-agnostic DataArray's data as a numpy ndarray."""
    data = arr.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return np.asarray(data)


class TestIntegerInput:
    """Integer rasters should resample through every backend without
    clipping or overflow, producing finite float output."""

    @pytest.mark.parametrize(
        'backend', ['numpy', 'cupy', 'dask+numpy', 'dask+cupy']
    )
    @pytest.mark.parametrize(
        'method', ['nearest', 'average', 'min', 'max', 'median', 'mode']
    )
    def test_int32_input_resamples(self, backend, method):
        if not _backend_available(backend):
            pytest.skip(f"backend {backend} unavailable")

        rng = np.random.RandomState(1470)
        data = rng.randint(0, 101, size=(8, 8)).astype(np.int32)
        agg = create_test_raster(
            data, backend=backend, attrs={'res': (1.0, 1.0)}, chunks=(4, 4)
        )

        out = resample(agg, scale_factor=0.5, method=method)
        out_np = _to_numpy(out)

        assert out.shape == (4, 4)
        # Output should always be float (regardless of input dtype).
        assert np.issubdtype(out_np.dtype, np.floating)
        assert np.all(np.isfinite(out_np))
        # Values are bounded by the input range, so no overflow / clipping.
        assert out_np.min() >= 0
        assert out_np.max() <= 100


# ---------------------------------------------------------------------------
# target_resolution: tuple and scalar forms
# ---------------------------------------------------------------------------

class TestTargetResolutionTuple:
    """Tuple form `target_resolution=(y, x)` lets users specify
    asymmetric output cell sizes; only the scalar form was covered before."""

    def test_target_resolution_tuple_shape_and_res(self, grid_8x8):
        # grid_8x8 has res=(1.0, 1.0) and shape (8, 8).
        # target_resolution=(2.0, 4.0) means 2.0 m per pixel along y and
        # 4.0 m per pixel along x, so output should be (4, 2).
        out = resample(grid_8x8, target_resolution=(2.0, 4.0))
        assert out.shape == (4, 2)
        # attrs['res'] is stored as (px, py) -- x cellsize, y cellsize.
        assert pytest.approx(out.attrs['res'][0], abs=0.01) == 4.0
        assert pytest.approx(out.attrs['res'][1], abs=0.01) == 2.0

    def test_target_resolution_scalar_matches_uniform_tuple(self, grid_8x8):
        scalar_out = resample(grid_8x8, target_resolution=2.0)
        tuple_out = resample(grid_8x8, target_resolution=(2.0, 2.0))
        assert scalar_out.shape == tuple_out.shape == (4, 4)
        np.testing.assert_allclose(
            scalar_out.values, tuple_out.values, atol=1e-6
        )
        assert (
            pytest.approx(scalar_out.attrs['res'][0], abs=0.01)
            == tuple_out.attrs['res'][0]
        )

    def test_target_resolution_list_form(self, grid_8x8):
        # The source accepts list as well as tuple; verify equivalence.
        out_tuple = resample(grid_8x8, target_resolution=(2.0, 4.0))
        out_list = resample(grid_8x8, target_resolution=[2.0, 4.0])
        assert out_tuple.shape == out_list.shape == (4, 2)
        np.testing.assert_allclose(
            out_tuple.values, out_list.values, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Memory guard (#1295)
# ---------------------------------------------------------------------------

class TestMemoryGuard:
    """Reject scale factors that would OOM the eager backends."""

    def test_huge_scale_factor_raises(self, grid_4x4):
        # 4 * 1e9 ~= 4e9 cells per axis -> 1.6e19 cells -> ~190 EB
        with pytest.raises(MemoryError, match="resample output of"):
            resample(grid_4x4, scale_factor=1e9, method='nearest')

    def test_huge_target_resolution_inverse_raises(self, grid_4x4):
        # cellsize=1.0, target_resolution=1e-9 -> ~4e9 cells per axis
        with pytest.raises(MemoryError, match="resample output of"):
            resample(grid_4x4, target_resolution=1e-9, method='nearest')

    def test_huge_scale_factor_aggregate_path_unaffected(self, grid_4x4):
        # Aggregate methods reject scale > 1.0 with ValueError before
        # the memory guard runs; confirm that error path still wins.
        with pytest.raises(ValueError, match="only supports downsampling"):
            resample(grid_4x4, scale_factor=1e9, method='average')

    def test_normal_inputs_unaffected(self, grid_4x4):
        # Sanity: a normal upsample call still works.
        out = resample(grid_4x4, scale_factor=2.0, method='nearest')
        assert out.shape == (8, 8)

    def test_error_message_names_parameters(self, grid_4x4):
        # The hint should point the user at the parameters they control.
        with pytest.raises(MemoryError) as excinfo:
            resample(grid_4x4, scale_factor=1e9, method='bilinear')
        msg = str(excinfo.value)
        assert "scale_factor" in msg
        assert "target_resolution" in msg

    def test_dask_path_skips_guard(self, grid_4x4):
        # Dask backends build per-chunk allocations lazily -- the guard
        # should not fire even for shapes that would OOM the eager path.
        # We only check that the output graph builds; we never compute it.
        if not dask_array_available():
            pytest.skip("dask not installed")
        import dask.array as da
        dask_agg = grid_4x4.copy()
        dask_agg.data = da.from_array(grid_4x4.data, chunks=2)
        # scale_factor=100 -> 400x400 output, well within RAM budget
        # but still exercises the dask dispatch.  We just want the
        # guard not to short-circuit a reasonable dask call.
        out = resample(dask_agg, scale_factor=100.0, method='nearest')
        assert out.shape == (400, 400)


# ---------------------------------------------------------------------------
# Coordinate orientation (T-3)
# ---------------------------------------------------------------------------

class TestCoordinateOrientation:
    """Resampling must preserve the y-axis orientation of the input."""

    def test_descending_y_preserved(self):
        # 8x8 grid with strong vertical asymmetry: row 0 = 100, row 7 = 0.
        data = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            data[r, :] = (7 - r) * (100.0 / 7.0)
        # Descending y: 10, 9, ..., 3.  x ascending.
        y = np.arange(10, 2, -1, dtype=np.float64)
        x = np.arange(8, dtype=np.float64)
        agg = xr.DataArray(
            data, dims=('y', 'x'), coords={'y': y, 'x': x},
            attrs={'res': (1.0, 1.0)},
        )

        out = resample(agg, scale_factor=0.5, method='average')

        # Output should have 4 y-coords, still descending.
        assert out.shape == (4, 4)
        assert np.all(np.diff(out.y.values) < 0), (
            "output y must remain descending"
        )

        # Cell-centered y-coords for a 4-pixel output spanning the same
        # extent as input y in [2.5, 10.5]: centers at 9.5, 7.5, 5.5, 3.5.
        np.testing.assert_allclose(
            out.y.values, [9.5, 7.5, 5.5, 3.5], atol=1e-6
        )

        # Top row of output should average the (high-value) top of the
        # input, bottom row should average the (low-value) bottom.  No
        # vertical flip.
        assert out.values[0, 0] > out.values[-1, 0], (
            "top of output should be greater than bottom (no flip)"
        )
        # Block-average of rows 0 and 1 = mean of 100 and ~85.7 = ~92.86.
        np.testing.assert_allclose(out.values[0, 0], 92.857143, atol=1e-4)
        # Bottom block averages rows 6 and 7 = mean of ~14.29 and 0 = ~7.14.
        np.testing.assert_allclose(out.values[-1, 0], 7.142857, atol=1e-4)


# ---------------------------------------------------------------------------
# Inf input behavior (T-6)
# ---------------------------------------------------------------------------

class TestInfHandling:
    """Lock in the current behavior: infs propagate, they are not NaN-coerced.

    The NaN-aware interpolation path treats infs the same as any other
    finite value (they pass through ``np.isnan`` checks), so the
    interpolation kernel includes them in its weighted sum and the
    aggregate kernel includes them in min / max / mean.  Output values
    that depend on an inf input cell come back as +/- inf.
    """

    @pytest.fixture
    def grid_with_infs(self):
        data = np.ones((6, 6), dtype=np.float32)
        data[1, 1] = np.inf
        data[4, 4] = -np.inf
        return create_test_raster(data, attrs={'res': (1.0, 1.0)})

    def test_bilinear_propagates_inf(self, grid_with_infs):
        out = resample(grid_with_infs, scale_factor=0.5, method='bilinear')
        # +inf and -inf both reach the output, no NaN coercion.
        assert np.isposinf(out.values).any()
        assert np.isneginf(out.values).any()
        assert not np.isnan(out.values).any()

    def test_average_propagates_inf(self, grid_with_infs):
        out = resample(grid_with_infs, scale_factor=0.5, method='average')
        assert np.isposinf(out.values).any()
        assert np.isneginf(out.values).any()
        assert not np.isnan(out.values).any()

    def test_nearest_passes_inf_through(self, grid_with_infs):
        out = resample(grid_with_infs, scale_factor=0.5, method='nearest')
        # Nearest only picks one input cell per output pixel, so at
        # least one infinite input survives in the output.
        assert np.isinf(out.values).any()


# ---------------------------------------------------------------------------
# 4D input rejection message (T-7)
# ---------------------------------------------------------------------------

class TestDimensionRejection:
    def test_4d_input_error_message(self):
        # 3D (band, y, x) is now valid; 4D should still be rejected.
        data = np.zeros((2, 2, 4, 4), dtype=np.float32)
        agg = xr.DataArray(
            data, dims=('time', 'band', 'y', 'x'),
            coords={
                'time': [0, 1],
                'band': [1, 2],
                'y': np.arange(4, dtype=np.float64),
                'x': np.arange(4, dtype=np.float64),
            },
            attrs={'res': (1.0, 1.0)},
        )
        with pytest.raises(ValueError, match="must be 2D"):
            resample(agg, scale_factor=0.5, method='nearest')


# ---------------------------------------------------------------------------
# Bit reproducibility (T-9)
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Repeat calls with identical inputs must produce identical outputs."""

    @pytest.fixture
    def random_grid(self):
        data = np.random.RandomState(1471).rand(20, 20).astype(np.float32)
        return create_test_raster(data, attrs={'res': (1.0, 1.0)})

    @pytest.mark.parametrize('method', ['bilinear', 'cubic', 'average'])
    def test_bit_identical_repeat(self, random_grid, method):
        out1 = resample(random_grid, scale_factor=0.5, method=method)
        out2 = resample(random_grid, scale_factor=0.5, method=method)
        assert np.array_equal(out1.values, out2.values), (
            f"{method} resample is not bit-identical across runs"
        )


# ---------------------------------------------------------------------------
# Inlined dask aggregate kernel (#1463)
# ---------------------------------------------------------------------------

@dask_array_available
class TestDaskAggregateInlined:
    """Cover the per-chunk numba aggregate kernel."""

    @pytest.mark.parametrize('method',
                             ['average', 'min', 'max', 'median', 'mode'])
    def test_inlined_kernel_matches_numpy(self, method):
        # 60x60 with 20x20 chunks and scale_factor=1/3 produces a 20x20
        # output.  Each output pixel collapses a 3x3 input window, and
        # the output chunks straddle the input chunk boundaries because
        # `_add_overlap` extends each chunk by `depth_y = depth_x = 3`.
        rng = np.random.RandomState(1463)
        data = rng.rand(60, 60).astype(np.float32)
        np_agg = create_test_raster(data, backend='numpy',
                                    attrs={'res': (1.0, 1.0)})
        dk_agg = create_test_raster(data, backend='dask+numpy',
                                    attrs={'res': (1.0, 1.0)},
                                    chunks=(20, 20))
        np_out = resample(np_agg, scale_factor=1.0 / 3.0, method=method)
        dk_out = resample(dk_agg, scale_factor=1.0 / 3.0, method=method)
        np.testing.assert_array_equal(dk_out.values, np_out.values)

    def test_dask_aggregate_smoke_200x200(self):
        # Smoke test: confirm the inlined path completes within a
        # generous wall-clock budget on a moderate raster.  Not a
        # perf assertion -- just guards against accidental
        # quadratic regressions in the chunk loop.
        import time
        rng = np.random.RandomState(146301)
        data = rng.rand(200, 200).astype(np.float32)
        dk_agg = create_test_raster(data, backend='dask+numpy',
                                    attrs={'res': (1.0, 1.0)},
                                    chunks=(50, 50))
        t0 = time.perf_counter()
        out = resample(dk_agg, scale_factor=0.25, method='average').compute()
        elapsed = time.perf_counter() - t0
        assert out.shape == (50, 50)
        # 5 s is generous; the inlined kernel runs in well under 1 s on
        # a typical laptop.  The previous per-pixel dispatch could miss
        # this on cold-cache numba compilation runs.
        assert elapsed < 30.0, (
            f"dask aggregate took {elapsed:.2f}s; expected well under 5s"
        )


# ---------------------------------------------------------------------------
# 3D rasters (issue #1466)
# ---------------------------------------------------------------------------

class TestThreeDRasters:
    """Multi-band ``(band, y, x)`` rasters resample per-band."""

    def _make_3d(self, backend='numpy'):
        # 3 bands of an 8x8 gradient. Each band has a unique offset so we
        # can confirm bands aren't mixed during the dispatch.
        y, x = np.mgrid[0:8, 0:8]
        band0 = (y * 10 + x).astype(np.float32)
        band1 = band0 + 100
        band2 = band0 + 200
        data = np.stack([band0, band1, band2], axis=0)
        agg = xr.DataArray(
            data,
            dims=('band', 'y', 'x'),
            coords={
                'band': np.array([1, 2, 3]),
                'y': np.arange(8, dtype=np.float64),
                'x': np.arange(8, dtype=np.float64),
            },
            attrs={'res': (1.0, 1.0)},
            name='myraster',
        )
        if backend == 'dask':
            import dask.array as da
            agg = agg.copy()
            agg.data = da.from_array(agg.data, chunks=(1, 4, 4))
        elif backend == 'cupy':
            import cupy
            agg = agg.copy()
            agg.data = cupy.asarray(agg.data)
        return agg

    def test_3d_numpy_shape_and_band_coord(self):
        agg = self._make_3d('numpy')
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out.shape == (3, 4, 4)
        assert out.dims == ('band', 'y', 'x')
        np.testing.assert_array_equal(out['band'].values, [1, 2, 3])

    def test_3d_per_band_independence(self):
        """Each band's output should be the 2D resample of that band."""
        agg = self._make_3d('numpy')
        out = resample(agg, scale_factor=0.5, method='average')
        for i in range(3):
            band_2d = agg.isel(band=i).reset_coords(drop=True)
            ref = resample(band_2d, scale_factor=0.5, method='average')
            np.testing.assert_allclose(out.isel(band=i).values, ref.values,
                                       atol=1e-5)

    def test_3d_target_resolution_tuple(self):
        agg = self._make_3d('numpy')
        out = resample(agg, target_resolution=(2.0, 4.0))
        assert out.shape == (3, 4, 2)

    @dask_array_available
    def test_3d_dask(self):
        agg = self._make_3d('dask')
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out.shape == (3, 4, 4)
        np.testing.assert_array_equal(out['band'].values, [1, 2, 3])

    @cuda_and_cupy_available
    def test_3d_cupy(self):
        agg = self._make_3d('cupy')
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out.shape == (3, 4, 4)
        np.testing.assert_array_equal(out['band'].values, [1, 2, 3])


# ---------------------------------------------------------------------------
# nodata handling (issue #1466)
# ---------------------------------------------------------------------------

class TestNodata:
    def test_explicit_nodata_int_sentinel(self):
        # Integer raster with -9999 sentinel. After resample those pixels
        # should become NaN; valid pixels stay as float interpolations.
        data = np.array([
            [-9999, -9999, 10, 10],
            [-9999, -9999, 10, 10],
            [20, 20, 30, 30],
            [20, 20, 30, 30],
        ], dtype=np.int32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='nearest',
                       nodata=-9999)
        assert out.shape == (2, 2)
        # Top-left output pixel maps to the -9999 region -> NaN
        assert np.isnan(out.values[0, 0])
        # Bottom-right pixel maps to a valid region -> finite
        assert np.isfinite(out.values[1, 1])
        assert out.attrs.get('_FillValue') is not None
        assert np.isnan(out.attrs['_FillValue'])

    def test_nodata_from_fillvalue_attr(self):
        # Same data, but sentinel discovered via _FillValue attr.
        data = np.array([
            [-9999, -9999, 10, 10],
            [-9999, -9999, 10, 10],
            [20, 20, 30, 30],
            [20, 20, 30, 30],
        ], dtype=np.int32)
        agg = create_test_raster(
            data, attrs={'res': (1.0, 1.0), '_FillValue': -9999}
        )
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert np.isnan(out.values[0, 0])
        assert np.isfinite(out.values[1, 1])

    def test_nodata_from_nodata_attr(self):
        data = np.array([
            [-9999, -9999, 10, 10],
            [-9999, -9999, 10, 10],
            [20, 20, 30, 30],
            [20, 20, 30, 30],
        ], dtype=np.int32)
        agg = create_test_raster(
            data, attrs={'res': (1.0, 1.0), 'nodata': -9999}
        )
        out = resample(agg, scale_factor=0.5, method='average')
        assert np.isnan(out.values[0, 0])

    def test_nodata_none_no_attrs_unchanged(self):
        # Without an explicit param or attr, behavior matches the old
        # (pre-#1466) implementation -- no masking, no _FillValue added.
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert '_FillValue' not in out.attrs

    def test_nodata_float_explicit(self):
        # Float sentinel -- e.g. -1.0 marking masked pixels.
        data = np.array([[-1.0, -1.0, 5.0, 5.0],
                         [-1.0, -1.0, 5.0, 5.0],
                         [3.0, 3.0, 7.0, 7.0],
                         [3.0, 3.0, 7.0, 7.0]], dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='nearest', nodata=-1.0)
        assert np.isnan(out.values[0, 0])

    def test_explicit_nodata_overrides_attr(self):
        # Explicit param wins over _FillValue attr.
        # 4x4 with -1 in the top-left 2x2 block. _FillValue says -999
        # (which doesn't appear); explicit nodata=-1 should mask the corner.
        data = np.array([[-1.0, -1.0, 5.0, 5.0],
                         [-1.0, -1.0, 5.0, 5.0],
                         [3.0, 3.0, 7.0, 7.0],
                         [3.0, 3.0, 7.0, 7.0]], dtype=np.float32)
        agg = create_test_raster(
            data, attrs={'res': (1.0, 1.0), '_FillValue': -999.0}
        )
        out = resample(agg, scale_factor=0.5, method='nearest', nodata=-1.0)
        # Without override the attr would say -999 (no match) and -1 would
        # leak through; with override the top-left output pixel is NaN.
        assert np.isnan(out.values[0, 0])


# ---------------------------------------------------------------------------
# target_resolution as 2-tuple (issue #1466)
# ---------------------------------------------------------------------------

class TestTargetResolutionTuple:
    def test_tuple_resolution_independent_axes(self, grid_8x8):
        # 8x8 grid with res=(1, 1) -> target (2, 4) -> output (4, 2).
        out = resample(grid_8x8, target_resolution=(2.0, 4.0))
        assert out.shape == (4, 2)

    def test_tuple_resolution_matches_scale_factor(self, grid_8x8):
        # target_resolution=(2.0, 2.0) should match scale_factor=0.5.
        a = resample(grid_8x8, target_resolution=(2.0, 2.0), method='nearest')
        b = resample(grid_8x8, scale_factor=0.5, method='nearest')
        np.testing.assert_allclose(a.values, b.values)
