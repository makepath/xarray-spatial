"""Test coverage gap closures for xrspatial.resample (deep-sweep 2026-05-27).

Adds tests for gaps the deep-sweep test-coverage audit surfaced:

- Cat 3 HIGH: Nx1 single-column raster across all four backends + all
  methods (existing tests covered 1x1 and 1xN but not Nx1).
- Cat 2 MEDIUM: NaN parity between numpy and cupy / dask+cupy
  backends (existing parity tests used random data without NaNs).
- Cat 3 MEDIUM: all-equal-value raster across methods (zero-variance
  input that could expose divide-by-zero in weight-aware kernels).
- Cat 5 MEDIUM: non-default dim names (lat/lon) propagate through
  resample without being renamed to y/x.
- Cat 3 MEDIUM: empty raster behavior pin (raises a clear ValueError
  before reaching coordinate indexing).

Source untouched -- this is a test-only coverage closure.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.resample import resample
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ---------------------------------------------------------------------------
# Cat 3 HIGH -- Nx1 single-column raster
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
    data = arr.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return np.asarray(data)


class TestSingleColumnRaster:
    """Nx1 single-column input. Kernel boundary degeneracy: width is 1
    so every window has unit horizontal extent. Existing tests covered
    1x1 and 1xN strips but no Nx1 case until now."""

    @pytest.mark.parametrize(
        'backend', ['numpy', 'cupy', 'dask+numpy', 'dask+cupy']
    )
    @pytest.mark.parametrize(
        'method', ['nearest', 'bilinear', 'cubic', 'average',
                   'min', 'max', 'median', 'mode']
    )
    def test_nx1_downsample(self, backend, method):
        if not _backend_available(backend):
            pytest.skip(f"backend {backend} unavailable")
        data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        agg = create_test_raster(
            data, backend=backend, dims=['y', 'x'],
            attrs={'res': (1.0, 1.0)}, chunks=(2, 1),
        )
        out = resample(agg, scale_factor=0.5, method=method)
        out_np = _to_numpy(out)
        assert out.shape == (2, 1)
        assert np.all(np.isfinite(out_np))
        # Output bounded by input range (no overflow / extrapolation)
        assert out_np.min() >= 1.0 and out_np.max() <= 4.0

    @pytest.mark.parametrize(
        'backend', ['numpy', 'cupy', 'dask+numpy', 'dask+cupy']
    )
    def test_nx1_upsample_nearest(self, backend):
        if not _backend_available(backend):
            pytest.skip(f"backend {backend} unavailable")
        data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        agg = create_test_raster(
            data, backend=backend, dims=['y', 'x'],
            attrs={'res': (1.0, 1.0)}, chunks=(2, 1),
        )
        # scale_factor=(2.0, 1.0) keeps the single column width fixed and
        # only upsamples along y, otherwise scale=2.0 would also double
        # the x axis to width 2 (since the formula rounds 1*2 = 2).
        out = resample(agg, scale_factor=(2.0, 1.0), method='nearest')
        out_np = _to_numpy(out)
        assert out.shape == (8, 1)
        # Nearest upsample preserves source values
        for v in out_np.ravel():
            assert v in data.ravel()

    def test_nx1_parity_numpy_vs_backends(self):
        """Pin Nx1 output equality across all available backends."""
        data = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]],
                        dtype=np.float32)
        np_agg = create_test_raster(
            data, backend='numpy', dims=['y', 'x'],
            attrs={'res': (1.0, 1.0)},
        )
        np_out = _to_numpy(resample(np_agg, scale_factor=0.5, method='average'))

        for backend in ['cupy', 'dask+numpy', 'dask+cupy']:
            if not _backend_available(backend):
                continue
            agg = create_test_raster(
                data, backend=backend, dims=['y', 'x'],
                attrs={'res': (1.0, 1.0)}, chunks=(3, 1),
            )
            out = _to_numpy(resample(agg, scale_factor=0.5, method='average'))
            np.testing.assert_allclose(
                out, np_out, atol=1e-5, equal_nan=True,
                err_msg=f"Nx1 average parity failed for {backend}",
            )


# ---------------------------------------------------------------------------
# Cat 2 MEDIUM -- NaN parity across cupy / dask+cupy backends
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
class TestCuPyNaNParity:
    """Existing TestCuPyParity used random data without NaN cells. The
    weight-aware NaN handling has independent numpy and cupy code paths
    (_nan_aware_interp_np vs _nan_aware_interp_cupy); a divergence in
    the weight-mask gate or spline-prefilter prepad would not surface
    from the no-NaN parity tests."""

    @pytest.fixture
    def numpy_and_cupy_nan_rasters(self):
        rng = np.random.RandomState(2026)
        data = rng.rand(12, 12).astype(np.float32)
        # Scatter NaNs so both interior and near-edge cells are masked.
        data[1, 1] = np.nan
        data[6, 6] = np.nan
        data[0, 4] = np.nan      # edge cell
        data[11, 11] = np.nan    # corner cell
        np_agg = create_test_raster(
            data, backend='numpy', attrs={'res': (1.0, 1.0)},
        )
        cp_agg = create_test_raster(
            data, backend='cupy', attrs={'res': (1.0, 1.0)},
        )
        return np_agg, cp_agg

    @pytest.mark.parametrize('method', ['nearest', 'bilinear', 'cubic'])
    @pytest.mark.parametrize('sf', [0.5, 2.0])
    def test_interp_nan_parity(self, numpy_and_cupy_nan_rasters, method, sf):
        np_agg, cp_agg = numpy_and_cupy_nan_rasters
        np_out = resample(np_agg, scale_factor=sf, method=method).values
        cp_out = resample(cp_agg, scale_factor=sf, method=method).data.get()
        np.testing.assert_allclose(
            cp_out, np_out, atol=1e-4, equal_nan=True,
            err_msg=f"cupy NaN parity failed for {method} sf={sf}",
        )

    @pytest.mark.parametrize('method', ['average', 'min', 'max'])
    def test_aggregate_nan_parity(self, numpy_and_cupy_nan_rasters, method):
        np_agg, cp_agg = numpy_and_cupy_nan_rasters
        np_out = resample(np_agg, scale_factor=0.5, method=method).values
        cp_out = resample(cp_agg, scale_factor=0.5, method=method).data.get()
        np.testing.assert_allclose(
            cp_out, np_out, atol=1e-5, equal_nan=True,
            err_msg=f"cupy NaN aggregate parity failed for {method}",
        )


@cuda_and_cupy_available
@dask_array_available
class TestDaskCuPyNaNParity:
    """Same NaN-parity check on the dask+cupy backend so the chunked
    GPU map_blocks path is also exercised with masked cells."""

    @pytest.fixture
    def numpy_and_dask_cupy_nan_rasters(self):
        rng = np.random.RandomState(2027)
        data = rng.rand(20, 20).astype(np.float32)
        data[2, 2] = np.nan
        data[10, 10] = np.nan
        data[0, 9] = np.nan
        data[19, 19] = np.nan
        np_agg = create_test_raster(
            data, backend='numpy', attrs={'res': (1.0, 1.0)},
        )
        dc_agg = create_test_raster(
            data, backend='dask+cupy', attrs={'res': (1.0, 1.0)},
            chunks=(8, 8),
        )
        return np_agg, dc_agg

    @pytest.mark.parametrize('method', ['nearest', 'bilinear', 'cubic'])
    def test_interp_nan_parity(self, numpy_and_dask_cupy_nan_rasters, method):
        np_agg, dc_agg = numpy_and_dask_cupy_nan_rasters
        np_out = resample(np_agg, scale_factor=0.5, method=method).values
        dc_out = resample(dc_agg, scale_factor=0.5,
                          method=method).data.compute().get()
        np.testing.assert_allclose(
            dc_out, np_out, atol=1e-4, equal_nan=True,
            err_msg=f"dask+cupy NaN parity failed for {method}",
        )

    @pytest.mark.parametrize('method', ['average', 'min', 'max'])
    def test_aggregate_nan_parity(self, numpy_and_dask_cupy_nan_rasters,
                                  method):
        np_agg, dc_agg = numpy_and_dask_cupy_nan_rasters
        np_out = resample(np_agg, scale_factor=0.5, method=method).values
        dc_out = resample(dc_agg, scale_factor=0.5,
                          method=method).data.compute().get()
        np.testing.assert_allclose(
            dc_out, np_out, atol=1e-5, equal_nan=True,
            err_msg=f"dask+cupy NaN aggregate parity failed for {method}",
        )


# ---------------------------------------------------------------------------
# Cat 3 MEDIUM -- all-equal-value raster (zero variance / zero gradient)
# ---------------------------------------------------------------------------

class TestAllEqualRaster:
    """Constant-value input has zero gradient: spline coefficient pre-filter
    should produce zeros except for the DC term, the weight-aware divisor
    should stay at 1.0, and aggregate kernels should return the constant.
    A divide-by-zero in the weight gate (e.g. dropping np.maximum(z_wt,
    1e-10)) would surface as NaN here."""

    @pytest.mark.parametrize(
        'method', ['nearest', 'bilinear', 'cubic', 'average',
                   'min', 'max', 'median', 'mode']
    )
    def test_constant_input_constant_output(self, method):
        data = np.full((8, 8), 5.0, dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method=method)
        np.testing.assert_allclose(out.values, 5.0, atol=1e-5)

    @pytest.mark.parametrize(
        'method', ['nearest', 'bilinear', 'cubic']
    )
    def test_constant_upsample(self, method):
        # Aggregate methods reject scale > 1.0, so only interpolation
        # methods upsample.
        data = np.full((4, 4), 3.0, dtype=np.float32)
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=2.0, method=method)
        np.testing.assert_allclose(out.values, 3.0, atol=1e-5)

    def test_constant_with_nan_block_aggregate(self):
        # The average aggregate is NaN-aware (see _agg_mean / _nan_aware_*
        # in resample.py): a 2x2 window containing one NaN and three 7s
        # collapses to 7, not NaN. Other cells are pure 7.
        data = np.full((4, 4), 7.0, dtype=np.float32)
        data[0, 0] = np.nan
        agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        out = resample(agg, scale_factor=0.5, method='average').values
        np.testing.assert_allclose(out, 7.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Cat 5 MEDIUM -- non-default dim names (lat/lon) propagate
# ---------------------------------------------------------------------------

class TestNonDefaultDimNames:
    """resample should not silently rename `lat`/`lon` to `y`/`x`. The
    public API uses agg.dims and rebuilds the output with the same names
    -- this test locks the contract so a regression would surface."""

    def test_lat_lon_dims_preserved(self):
        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        agg = xr.DataArray(
            data,
            dims=['lat', 'lon'],
            coords={
                'lat': np.linspace(7, 0, 8),
                'lon': np.linspace(0, 7, 8),
            },
            attrs={'res': (1.0, 1.0)},
        )
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out.dims == ('lat', 'lon')
        assert 'lat' in out.coords and 'lon' in out.coords
        assert out.shape == (4, 4)

    def test_latitude_longitude_dims_preserved(self):
        # CF-style long names.
        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        agg = xr.DataArray(
            data,
            dims=['latitude', 'longitude'],
            coords={
                'latitude': np.linspace(7, 0, 8),
                'longitude': np.linspace(0, 7, 8),
            },
            attrs={'res': (1.0, 1.0)},
        )
        out = resample(agg, scale_factor=0.5, method='average')
        assert out.dims == ('latitude', 'longitude')

    def test_3d_band_lat_lon_dims_preserved(self):
        # 3D path should preserve the leading-dim name too.
        data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        agg = xr.DataArray(
            data,
            dims=['channel', 'lat', 'lon'],
            coords={
                'channel': [1, 2, 3],
                'lat': np.linspace(3, 0, 4),
                'lon': np.linspace(0, 3, 4),
            },
            attrs={'res': (1.0, 1.0)},
        )
        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out.dims == ('channel', 'lat', 'lon')
        np.testing.assert_array_equal(out['channel'].values, [1, 2, 3])

    def test_dim_attrs_preserved(self):
        # Per-dim attrs (e.g. units) should round-trip.
        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        agg = xr.DataArray(
            data,
            dims=['lat', 'lon'],
            coords={
                'lat': np.linspace(7, 0, 8),
                'lon': np.linspace(0, 7, 8),
            },
            attrs={'res': (1.0, 1.0)},
        )
        agg['lat'].attrs['units'] = 'degrees_north'
        agg['lon'].attrs['units'] = 'degrees_east'

        out = resample(agg, scale_factor=0.5, method='nearest')
        assert out['lat'].attrs.get('units') == 'degrees_north'
        assert out['lon'].attrs.get('units') == 'degrees_east'


# ---------------------------------------------------------------------------
# Cat 3 MEDIUM -- empty raster (0 rows or 0 cols) behavior pin
# ---------------------------------------------------------------------------

class TestEmptyRasterRejected:
    """Resample on a zero-area raster raises a clear ValueError up front,
    before reaching the output-coordinate rebuild that would otherwise
    surface an opaque IndexError (vals[0] on an empty coord array)."""

    def test_zero_rows_raises(self):
        data = np.zeros((0, 4), dtype=np.float32)
        agg = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.zeros(0), 'x': np.arange(4, dtype=np.float64)},
            attrs={'res': (1.0, 1.0)},
        )
        with pytest.raises(ValueError, match='non-empty spatial'):
            resample(agg, scale_factor=0.5, method='nearest')

    def test_zero_cols_raises(self):
        data = np.zeros((4, 0), dtype=np.float32)
        agg = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.arange(4, dtype=np.float64), 'x': np.zeros(0)},
            attrs={'res': (1.0, 1.0)},
        )
        with pytest.raises(ValueError, match='non-empty spatial'):
            resample(agg, scale_factor=0.5, method='nearest')


# ---------------------------------------------------------------------------
# Issue #2547 -- dask cubic on arrays smaller than the prefilter depth
# ---------------------------------------------------------------------------

class TestDaskCubicSmallInput:
    """Inputs smaller than the cubic prefilter depth used to crash the dask
    path with ``ValueError: The overlapping depth 16 is larger than your
    array N``. The fix clamps overlap depth per axis to the array size, so
    every shape below should run end-to-end and match the eager numpy
    reference within float32 tolerance."""

    # Once depth is clamped to ``axis_total - 1``, ``_ensure_min_chunksize``
    # always collapses the clamped axis to a single chunk (its min_size is
    # ``2 * (axis_total - 1) + 1 > axis_total``). A user-supplied multi-chunk
    # layout on a small axis therefore folds into one chunk anyway, so the
    # parametrization below only varies shape, not user chunking.
    @pytest.mark.parametrize('backend', ['dask+numpy', 'dask+cupy'])
    @pytest.mark.parametrize(
        'shape,chunks',
        [
            ((8, 1), (2, 1)),    # Nx1 column, smaller than depth on rows
            ((1, 8), (1, 2)),    # 1xN row, smaller than depth on cols
            ((4, 4), (4, 4)),    # tiny single-chunk array
            ((8, 8), (4, 4)),    # both axes smaller than depth=16
        ],
    )
    def test_cubic_small_input(self, backend, shape, chunks):
        if not _backend_available(backend):
            pytest.skip(f"backend {backend} unavailable")
        h, w = shape
        data = np.arange(h * w, dtype=np.float32).reshape(h, w) + 1.0
        agg = create_test_raster(
            data, backend=backend, dims=['y', 'x'],
            attrs={'res': (1.0, 1.0)}, chunks=chunks,
        )
        out = resample(agg, scale_factor=0.5, method='cubic')
        out_np = _to_numpy(out)

        ref = create_test_raster(
            data, backend='numpy', dims=['y', 'x'],
            attrs={'res': (1.0, 1.0)},
        )
        ref_np = _to_numpy(resample(ref, scale_factor=0.5, method='cubic'))

        assert out_np.shape == ref_np.shape
        assert np.all(np.isfinite(out_np))
        # Boundary IIR transient differs once depth is clamped below 16,
        # so allow a loose tolerance for the small-input case.
        np.testing.assert_allclose(out_np, ref_np, rtol=1e-3, atol=1e-3)
