"""dask+numpy nodata-masking value parity for xrspatial.resample (#3073).

`resample` replaces nodata sentinels with NaN before resampling via
`_apply_nodata_mask`, which dispatches through xarray's `.where()` /
`.astype()` per backend. The existing dask+numpy nodata tests only cover
the error-raising paths (`TestNodataOutOfRange`) and the identity-attr
refresh (`TestIdentityNodataMetadata`); none assert that sentinel cells
become NaN with values matching the numpy backend after a real
(non-identity) resample. These tests close that gap.

cupy / dask+cupy are intentionally not covered: on xarray 2025.12 +
cupy 13.6, `DataArray.where()` on a cupy backend raises
`AttributeError: module 'cupy' has no attribute 'astype'` (an
xarray/cupy Array-API mismatch affecting every `.where` on a cupy array,
not a resample bug). Source untouched -- this is a test-only closure.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.resample import resample
from xrspatial.tests.general_checks import (
    create_test_raster,
    dask_array_available,
)


def _to_numpy(arr):
    data = arr.data
    if hasattr(data, 'compute'):
        data = data.compute()
    return np.asarray(data)


# A 4x4 grid whose top-left 2x2 block is the sentinel; downsampling by 0.5
# collapses each 2x2 block to one output pixel, so out[0, 0] maps entirely
# onto the sentinel block and must become NaN.
def _sentinel_grid(sentinel, dtype):
    return np.array([
        [sentinel, sentinel, 10, 10],
        [sentinel, sentinel, 10, 10],
        [20,       20,       30, 30],
        [20,       20,       30, 30],
    ], dtype=dtype)


@dask_array_available
class TestDaskNodataMaskingParity:
    """dask+numpy resample with a nodata sentinel masks the same cells,
    with the same values, as the numpy backend.
    """

    @pytest.mark.parametrize('method', ['nearest', 'average'])
    @pytest.mark.parametrize(
        'sentinel, dtype',
        [(-9999, np.int32), (-1.0, np.float32)],
        ids=['int_-9999', 'float_-1.0'],
    )
    def test_explicit_nodata_arg_parity(self, method, sentinel, dtype):
        data = _sentinel_grid(sentinel, dtype)
        np_agg = create_test_raster(
            data, backend='numpy', attrs={'res': (1.0, 1.0)})
        dk_agg = create_test_raster(
            data, backend='dask+numpy', attrs={'res': (1.0, 1.0)},
            chunks=(2, 2))

        np_out = resample(np_agg, scale_factor=0.5, method=method,
                          nodata=sentinel)
        dk_out = resample(dk_agg, scale_factor=0.5, method=method,
                          nodata=sentinel)

        np_vals = np_out.values
        dk_vals = _to_numpy(dk_out)
        # The all-sentinel block became NaN on both backends...
        assert np.isnan(np_vals[0, 0])
        assert np.isnan(dk_vals[0, 0])
        # ...and the valid corner stayed finite.
        assert np.isfinite(dk_vals[1, 1])
        # ...and the two backends agree everywhere (NaN positions included).
        np.testing.assert_allclose(dk_vals, np_vals, equal_nan=True,
                                   atol=1e-5)

    @pytest.mark.parametrize('method', ['nearest', 'average'])
    @pytest.mark.parametrize(
        'attr_key, sentinel, dtype',
        [
            ('_FillValue', -9999, np.int32),
            ('nodata', -9999, np.int32),
            ('_FillValue', -1.0, np.float32),
        ],
        ids=['fillvalue_int', 'nodata_attr_int', 'fillvalue_float'],
    )
    def test_attr_nodata_parity(self, method, attr_key, sentinel, dtype):
        data = _sentinel_grid(sentinel, dtype)
        attrs = {'res': (1.0, 1.0), attr_key: sentinel}
        np_agg = create_test_raster(data, backend='numpy', attrs=dict(attrs))
        dk_agg = create_test_raster(
            data, backend='dask+numpy', attrs=dict(attrs), chunks=(2, 2))

        np_out = resample(np_agg, scale_factor=0.5, method=method)
        dk_out = resample(dk_agg, scale_factor=0.5, method=method)

        np_vals = np_out.values
        dk_vals = _to_numpy(dk_out)
        assert np.isnan(np_vals[0, 0])
        assert np.isnan(dk_vals[0, 0])
        assert np.isfinite(dk_vals[1, 1])
        np.testing.assert_allclose(dk_vals, np_vals, equal_nan=True,
                                   atol=1e-5)
        # The masked-to-NaN output advertises NaN as the new sentinel.
        assert np.isnan(dk_out.attrs['_FillValue'])
