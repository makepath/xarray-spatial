"""Coverage for the cupy eager aggregate CPU fallback (issue #2615).

`_run_cupy` (xrspatial/resample.py) picks a GPU reshape+reduce path only
when the downsample factor is an integer and the method is one of
``average`` / ``min`` / ``max``. Everything else -- ``median`` / ``mode``,
and ``average`` / ``min`` / ``max`` at a non-integer factor -- copies the
array to the host, runs the numpy ``_AGG_FUNCS`` kernel, and copies the
result back.

The existing ``TestCuPyParity.test_aggregate_parity`` downsamples a 12x12
raster by 0.5, so the factor is exactly 2. That hits the GPU path for
average/min/max and the host fallback for median/mode. The average/min/max
host fallback for a *non-integer* factor was never exercised: a regression
in the ``fy == int(fy)`` branch test or in the host round-trip for that
path would have shipped undetected.

Source untouched -- test-only coverage closure.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.resample import resample
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
)


@cuda_and_cupy_available
class TestCuPyAggregateNonIntegerFactor:
    """average / min / max on a non-integer downsample factor route
    through the host fallback inside ``_run_cupy``; pin parity with the
    numpy reference so a fallback regression surfaces."""

    @pytest.fixture
    def numpy_and_cupy_rasters(self):
        # 10x10 at scale 0.3 -> 3x3 output -> factor 10/3 = 3.33 (non-integer).
        # This is the branch the integer-factor GPU fast path skips.
        data = np.random.RandomState(2615).rand(10, 10).astype(np.float32)
        np_agg = create_test_raster(data, backend='numpy',
                                    attrs={'res': (1.0, 1.0)})
        cp_agg = create_test_raster(data, backend='cupy',
                                    attrs={'res': (1.0, 1.0)})
        return np_agg, cp_agg

    @pytest.mark.parametrize('method', ['average', 'min', 'max'])
    def test_non_integer_factor_fallback_parity(self, numpy_and_cupy_rasters,
                                                method):
        np_agg, cp_agg = numpy_and_cupy_rasters
        # Guard the premise: 10 / round(10 * 0.3) must be non-integer so we
        # are genuinely on the fallback branch, not the GPU reshape path.
        out_h = max(1, round(10 * 0.3))
        assert 10 / out_h != int(10 / out_h)

        np_out = resample(np_agg, scale_factor=0.3, method=method)
        cp_out = resample(cp_agg, scale_factor=0.3, method=method)
        assert cp_out.shape == np_out.shape == (out_h, out_h)
        np.testing.assert_allclose(
            cp_out.data.get(), np_out.values, atol=1e-5, equal_nan=True,
            err_msg=f"cupy non-integer-factor fallback parity failed for "
                    f"{method}",
        )

    @pytest.mark.parametrize('method', ['average', 'min', 'max'])
    def test_non_integer_factor_fallback_with_nan(self, method):
        # The host fallback feeds the NaN-aware numpy kernels; make sure a
        # masked cell behaves the same as the eager numpy path.
        data = np.random.RandomState(26151).rand(10, 10).astype(np.float32)
        data[2, 3] = np.nan
        data[7, 8] = np.nan
        # Same premise guard as the no-NaN test: keep this on the
        # non-integer-factor host fallback, not the GPU reshape path.
        out_h = max(1, round(10 * 0.3))
        assert 10 / out_h != int(10 / out_h)
        np_agg = create_test_raster(data, backend='numpy',
                                    attrs={'res': (1.0, 1.0)})
        cp_agg = create_test_raster(data, backend='cupy',
                                    attrs={'res': (1.0, 1.0)})
        np_out = resample(np_agg, scale_factor=0.3, method=method)
        cp_out = resample(cp_agg, scale_factor=0.3, method=method)
        np.testing.assert_allclose(
            cp_out.data.get(), np_out.values, atol=1e-5, equal_nan=True,
            err_msg=f"cupy non-integer-factor NaN fallback parity failed for "
                    f"{method}",
        )
