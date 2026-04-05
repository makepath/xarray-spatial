"""Verify that dask-backed DataArrays stay lazy through fully-lazy functions.

Each test calls a function with a dask input and asserts the output is still a
dask collection (no accidental .compute()).  We do NOT test correctness here --
other test modules cover that.  We only test that the task graph survives.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.tests.general_checks import create_test_raster, dask_array_available

try:
    import dask
    import dask.array as da
except ImportError:
    da = None


pytestmark = dask_array_available


def _is_lazy(result):
    """Return True if *result* is a dask-backed DataArray (not computed)."""
    if isinstance(result, xr.DataArray):
        return dask.is_dask_collection(result)
    if isinstance(result, da.Array):
        return True
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elev():
    """8x6 dask-backed elevation raster."""
    data = np.array([
        [870.5, 283.0, 845.3,  51.2, 990.8, 600.6],
        [704.2, 242.2, 429.3, 779.9, 193.3, 984.7],
        [226.6, 815.7, 290.6,  76.5, 820.9,  32.3],
        [344.8, 256.3, 806.8, 602.0, 721.2, 497.0],
        [185.4, 834.1, 387.1, 716.0,  49.6, 753.0],
        [302.4, 151.5, 442.3, 358.5, 659.8, 447.1],
        [148.0, 819.2, 469.0, 977.1, 597.7, 999.1],
        [268.2, 626.0, 840.3, 448.3, 859.3, 528.0],
    ], dtype=np.float32)
    return create_test_raster(data, backend='dask+numpy',
                              attrs={'res': (1, 1)}, chunks=(4, 3))


@pytest.fixture
def classify_elev():
    """Larger raster for classification functions that need more data."""
    rng = np.random.default_rng(12345)
    data = rng.random((20, 20), dtype=np.float32) * 1000
    return create_test_raster(data, backend='dask+numpy',
                              attrs={'res': (1, 1)}, chunks=(10, 10))


# ---------------------------------------------------------------------------
# Terrain metrics
# ---------------------------------------------------------------------------

class TestTerrainLazy:
    def test_slope(self, elev):
        from xrspatial import slope
        assert _is_lazy(slope(elev))

    def test_aspect(self, elev):
        from xrspatial import aspect
        assert _is_lazy(aspect(elev))

    def test_curvature(self, elev):
        from xrspatial import curvature
        assert _is_lazy(curvature(elev))

    def test_hillshade(self, elev):
        from xrspatial import hillshade
        assert _is_lazy(hillshade(elev))

    def test_northness(self, elev):
        from xrspatial import northness
        assert _is_lazy(northness(elev))

    def test_eastness(self, elev):
        from xrspatial import eastness
        assert _is_lazy(eastness(elev))


# ---------------------------------------------------------------------------
# Focal
# ---------------------------------------------------------------------------

class TestFocalLazy:
    def test_mean(self, elev):
        from xrspatial import mean
        assert _is_lazy(mean(elev))

    def test_apply(self, elev):
        from xrspatial.focal import apply as focal_apply
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        assert _is_lazy(focal_apply(elev, kernel))

    def test_focal_stats(self, elev):
        from xrspatial.focal import focal_stats
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        result = focal_stats(elev, kernel)
        assert _is_lazy(result)

    def test_hotspots(self, elev):
        from xrspatial.focal import hotspots
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        assert _is_lazy(hotspots(elev, kernel))


# ---------------------------------------------------------------------------
# Classification (partially lazy -- returns dask after computing small stats)
# ---------------------------------------------------------------------------

class TestClassifyLazy:
    def test_binary(self, classify_elev):
        from xrspatial import binary
        assert _is_lazy(binary(classify_elev, values=[0, 500]))

    def test_reclassify(self, classify_elev):
        from xrspatial import reclassify
        assert _is_lazy(reclassify(classify_elev,
                                   bins=[250, 500, 750, 1000],
                                   new_values=[1, 2, 3, 4]))

    def test_quantile(self, classify_elev):
        from xrspatial import quantile
        assert _is_lazy(quantile(classify_elev, k=4))

    def test_natural_breaks(self, classify_elev):
        from xrspatial import natural_breaks
        assert _is_lazy(natural_breaks(classify_elev, k=4))

    def test_equal_interval(self, classify_elev):
        from xrspatial import equal_interval
        assert _is_lazy(equal_interval(classify_elev, k=4))

    def test_std_mean(self, classify_elev):
        from xrspatial import std_mean
        assert _is_lazy(std_mean(classify_elev))

    def test_head_tail_breaks(self, classify_elev):
        from xrspatial import head_tail_breaks
        assert _is_lazy(head_tail_breaks(classify_elev))

    def test_percentiles(self, classify_elev):
        from xrspatial import percentiles
        assert _is_lazy(percentiles(classify_elev))

    def test_maximum_breaks(self, classify_elev):
        from xrspatial import maximum_breaks
        assert _is_lazy(maximum_breaks(classify_elev, k=4))

    def test_box_plot(self, classify_elev):
        from xrspatial import box_plot
        assert _is_lazy(box_plot(classify_elev))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalizeLazy:
    def test_rescale(self, elev):
        from xrspatial import rescale
        assert _is_lazy(rescale(elev))

    def test_standardize(self, elev):
        from xrspatial import standardize
        assert _is_lazy(standardize(elev))
