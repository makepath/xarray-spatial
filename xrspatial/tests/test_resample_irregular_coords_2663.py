"""Coordinate-geometry validation tests for xrspatial.resample (issue #2663).

``resample`` only supports regular, monotonic rasters: ``calc_res`` derives
the input resolution from the full coordinate extent while the output coords
are rebuilt from first/last neighbour spacing. On an irregular or
non-monotonic grid those two views disagree and the function used to silently
produce inconsistent output geometry (e.g. ``x=[0, 1, 4]`` with
``target_resolution=1.0`` yielded width 6 and coords ``[0,1,2,3,4,5]``,
spilling past the input range ``[0, 4]``).

The function now rejects such inputs up front with a clear ``ValueError``.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.resample import resample
from xrspatial.utils import has_cuda_and_cupy, has_dask_array


def _backend_raster(data, *, y, x, backend):
    """Build a (y, x) DataArray with the given coords on a chosen backend."""
    raster = xr.DataArray(
        np.asarray(data, dtype=np.float64),
        dims=['y', 'x'],
        coords={'y': np.asarray(y, dtype=np.float64),
                'x': np.asarray(x, dtype=np.float64)},
    )
    if 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)
    if 'dask' in backend:
        import dask.array as da
        raster.data = da.from_array(raster.data, chunks=(2, 2))
    return raster


# Backends to exercise. Coordinate validation reads numpy-backed coords
# regardless of data backend, so we drive every available backend to prove
# the guard fires before any backend dispatch happens.
BACKENDS = ['numpy']
if has_dask_array():
    BACKENDS.append('dask+numpy')
if has_cuda_and_cupy():
    BACKENDS.append('cupy')
    if has_dask_array():
        BACKENDS.append('dask+cupy')


@pytest.mark.parametrize('backend', BACKENDS)
class TestIrregularCoordsRejected:
    def test_irregular_x_spacing(self, backend):
        data = np.arange(9.).reshape(3, 3)
        agg = _backend_raster(data, y=[0, 1, 2], x=[0, 1, 4], backend=backend)
        with pytest.raises(
            ValueError,
            match=r"coordinate 'x' must be evenly spaced",
        ):
            resample(agg, target_resolution=1.0)

    def test_irregular_y_spacing(self, backend):
        data = np.arange(9.).reshape(3, 3)
        agg = _backend_raster(data, y=[0, 1, 4], x=[0, 1, 2], backend=backend)
        with pytest.raises(
            ValueError,
            match=r"coordinate 'y' must be evenly spaced",
        ):
            resample(agg, target_resolution=1.0)

    def test_non_monotonic_x(self, backend):
        data = np.arange(9.).reshape(3, 3)
        agg = _backend_raster(data, y=[0, 1, 2], x=[0, 2, 1], backend=backend)
        with pytest.raises(
            ValueError,
            match=r"coordinate 'x' must be strictly monotonic",
        ):
            resample(agg, target_resolution=1.0)

    def test_non_monotonic_y(self, backend):
        data = np.arange(9.).reshape(3, 3)
        agg = _backend_raster(data, y=[0, 2, 1], x=[0, 1, 2], backend=backend)
        with pytest.raises(
            ValueError,
            match=r"coordinate 'y' must be strictly monotonic",
        ):
            resample(agg, target_resolution=1.0)

    def test_irregular_rejected_for_scale_factor_too(self, backend):
        # The guard runs before the scale_factor / target_resolution split,
        # so an irregular grid is rejected regardless of which knob is used.
        data = np.arange(9.).reshape(3, 3)
        agg = _backend_raster(data, y=[0, 1, 2], x=[0, 1, 4], backend=backend)
        with pytest.raises(ValueError, match=r"evenly spaced"):
            resample(agg, scale_factor=2.0)


@pytest.mark.parametrize('backend', BACKENDS)
class TestRegularCoordsStillWork:
    def test_ascending_regular(self, backend):
        data = np.arange(16.).reshape(4, 4)
        agg = _backend_raster(
            data, y=[0, 1, 2, 3], x=[0, 1, 2, 3], backend=backend,
        )
        out = resample(agg, scale_factor=0.5)
        assert out.shape == (2, 2)

    def test_descending_y_regular(self, backend):
        # North-up rasters have a descending y axis; that is regular and
        # must remain accepted.
        data = np.arange(16.).reshape(4, 4)
        agg = _backend_raster(
            data, y=[3, 2, 1, 0], x=[0, 1, 2, 3], backend=backend,
        )
        out = resample(agg, target_resolution=0.5)
        assert out.shape == (8, 8)

    def test_float_jitter_within_tolerance(self, backend):
        # Tiny floating-point jitter on an otherwise regular grid must not
        # trip the evenly-spaced check.
        x = np.array([0.0, 1.0, 2.0, 3.0])
        x = x + np.array([0.0, 1e-10, -1e-10, 5e-11])
        data = np.arange(16.).reshape(4, 4)
        agg = _backend_raster(data, y=[0, 1, 2, 3], x=x, backend=backend)
        out = resample(agg, scale_factor=0.5)
        assert out.shape == (2, 2)


def test_missing_coords_not_validated():
    # An input without spatial coords has nothing to validate; the existing
    # code paths handle it, so the guard must not raise.
    data = np.arange(16.).reshape(4, 4)
    agg = xr.DataArray(data.astype(np.float64), dims=['y', 'x'])
    out = resample(agg, scale_factor=2.0)
    assert out.shape == (8, 8)


def test_single_pixel_axis_skipped():
    # A length-1 axis has no spacing to check; pair it with a regular axis
    # and a scale_factor so the call reaches output construction.
    data = np.arange(3.).reshape(1, 3)
    agg = xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': [0.0], 'x': [0.0, 1.0, 2.0]},
    )
    out = resample(agg, scale_factor=(1.0, 2.0))
    assert out.shape[-1] == 6
