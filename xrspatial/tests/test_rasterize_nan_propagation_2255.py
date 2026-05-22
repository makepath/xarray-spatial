"""Cross-backend NaN-propagation pins for ``rasterize`` ``merge='max'`` / ``'min'``.

Issue #2255: before the fix the GPU paths silently dropped NaN burns
(``atomicMax`` / ``atomicMin`` compare with ``<`` / ``>``, both False
for NaN operands), and the CPU path dropped a NaN burn that arrived
*after* a finite burn (``NaN > finite`` is False, so the kernel kept
the prior finite pixel).  The PR aligned all four backends on strict
NaN propagation: any NaN burn, regardless of input order, makes the
pixel NaN under max / min.

These tests pin that contract across numpy / cupy / dask+numpy /
dask+cupy so a future regression (e.g. swapping ``fmax`` / ``fmin``
back in, or restoring the -inf-init optimisation without the NaN-mask
finalize) fails loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

try:
    import cupy
    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import dask  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

try:
    from numba import cuda
    has_cuda = has_cupy and cuda.is_available()
except Exception:
    has_cuda = False

skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")
skip_no_dask = pytest.mark.skipif(
    not has_dask, reason="dask not installed")


def _materialise(result):
    """Compute (if dask) and copy to host (if cupy) so callers see ndarray."""
    data = result.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if has_cupy and isinstance(data, cupy.ndarray):
        return cupy.asnumpy(data)
    return np.asarray(data)


ALL_BACKENDS = [
    pytest.param('numpy', {}, id='numpy'),
    pytest.param('cupy', {'use_cuda': True},
                 marks=skip_no_cuda, id='cupy'),
    pytest.param('dask_numpy', {'chunks': (5, 5)},
                 marks=skip_no_dask, id='dask_numpy'),
    pytest.param('dask_cupy', {'use_cuda': True, 'chunks': (5, 5)},
                 marks=[skip_no_cuda, skip_no_dask], id='dask_cupy'),
]


class TestNaNPropagationMaxMin:
    """Any NaN burn poisons ``max`` / ``min``, regardless of order."""

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_nan_first_then_finite_max(self, backend_name, kw):
        r = rasterize(
            [(box(0, 0, 10, 5), np.nan), (box(0, 0, 10, 5), 1.0)],
            width=10, height=5, bounds=(0, 0, 10, 5),
            fill=0, merge='max', **kw)
        data = _materialise(r)
        assert np.all(np.isnan(data)), (
            f"{backend_name}: expected all-NaN, got {data}")

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_finite_then_nan_max(self, backend_name, kw):
        """CPU previously kept the finite value here ('NaN > finite' is
        False so the order-dependent kernel dropped the later NaN).
        After the fix the NaN burn explicitly poisons the pixel."""
        r = rasterize(
            [(box(0, 0, 10, 5), 1.0), (box(0, 0, 10, 5), np.nan)],
            width=10, height=5, bounds=(0, 0, 10, 5),
            fill=0, merge='max', **kw)
        data = _materialise(r)
        assert np.all(np.isnan(data)), (
            f"{backend_name}: expected all-NaN, got {data}")

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_single_nan_burn_max(self, backend_name, kw):
        """Lone NaN burn lands as NaN, not -inf (GPU init) and not the
        fill (CPU default)."""
        r = rasterize([(box(0, 0, 10, 5), np.nan)],
                      width=10, height=5, bounds=(0, 0, 10, 5),
                      fill=0, merge='max', **kw)
        data = _materialise(r)
        assert np.all(np.isnan(data)), (
            f"{backend_name}: expected NaN, got {data}")

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_nan_first_then_finite_min(self, backend_name, kw):
        """Same poisoning contract under ``merge='min'``."""
        r = rasterize(
            [(box(0, 0, 10, 5), np.nan), (box(0, 0, 10, 5), 1.0)],
            width=10, height=5, bounds=(0, 0, 10, 5),
            fill=0, merge='min', **kw)
        data = _materialise(r)
        assert np.all(np.isnan(data)), (
            f"{backend_name}: expected all-NaN, got {data}")

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_single_nan_burn_min(self, backend_name, kw):
        """Single NaN burn under ``merge='min'`` -> NaN (was +inf on GPU)."""
        r = rasterize([(box(0, 0, 10, 5), np.nan)],
                      width=10, height=5, bounds=(0, 0, 10, 5),
                      fill=0, merge='min', **kw)
        data = _materialise(r)
        assert np.all(np.isnan(data)), (
            f"{backend_name}: expected NaN, got {data}")

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_partial_nan_partial_finite_max(self, backend_name, kw):
        """Non-overlapping polygons: NaN poisons only its own pixels,
        finite polygon's pixels keep their finite max.  Guards against a
        regression where the NaN-mask leaks across the raster."""
        r = rasterize(
            [(box(0, 0, 5, 5), np.nan), (box(5, 0, 10, 5), 1.0)],
            width=10, height=5, bounds=(0, 0, 10, 5),
            fill=0, merge='max', **kw)
        data = _materialise(r)
        assert np.all(np.isnan(data[:, :5])), (
            f"{backend_name}: left half should be NaN")
        np.testing.assert_array_equal(data[:, 5:], np.ones((5, 5)))

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_untouched_pixels_get_fill_under_max(self, backend_name, kw):
        """Pixels with no burn retain the fill, even when other pixels
        saw NaN burns.  Guards against the NaN-mask bleeding into
        untouched cells during finalize."""
        r = rasterize([(box(0, 0, 5, 5), np.nan)],
                      width=10, height=5, bounds=(0, 0, 10, 5),
                      fill=-9.0, merge='max', **kw)
        data = _materialise(r)
        assert np.all(np.isnan(data[:, :5]))
        np.testing.assert_array_equal(data[:, 5:], np.full((5, 5), -9.0))
