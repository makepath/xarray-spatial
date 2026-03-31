"""Tests for issue #1106: multi-metric output ordering bug.

When glcm_texture() receives metrics in non-standard order, the coordinate
labels must still match the actual data.  The kernel writes output slots in
VALID_METRICS order, so the labels need to follow.
"""
try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial.glcm import glcm_texture, VALID_METRICS, _sorted_metrics
from xrspatial.tests.general_checks import (
    create_test_raster,
    dask_array_available,
)


# ---- _sorted_metrics helper ----

def test_sorted_metrics_standard_order():
    assert _sorted_metrics(['contrast', 'entropy']) == ['contrast', 'entropy']


def test_sorted_metrics_reversed():
    assert _sorted_metrics(['entropy', 'contrast']) == ['contrast', 'entropy']


def test_sorted_metrics_all():
    shuffled = list(reversed(VALID_METRICS))
    assert _sorted_metrics(shuffled) == list(VALID_METRICS)


def test_sorted_metrics_single():
    assert _sorted_metrics(['homogeneity']) == ['homogeneity']


# ---- Core regression test: label matches data ----

@pytest.fixture
def random_8x8():
    rng = np.random.default_rng(99)
    return rng.random((8, 8))


def test_reversed_order_matches_solo(random_8x8):
    """Requesting metrics in reversed order should still label them correctly."""
    agg = create_test_raster(random_8x8)

    contrast_solo = glcm_texture(agg, metric='contrast', window_size=3,
                                 levels=16, angle=0)
    homogeneity_solo = glcm_texture(agg, metric='homogeneity', window_size=3,
                                    levels=16, angle=0)

    # Reversed order: homogeneity before contrast
    multi = glcm_texture(agg, metric=['homogeneity', 'contrast'],
                         window_size=3, levels=16, angle=0)

    np.testing.assert_allclose(
        multi.sel(metric='contrast').values,
        contrast_solo.values,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        multi.sel(metric='homogeneity').values,
        homogeneity_solo.values,
        rtol=1e-10,
    )


def test_arbitrary_order_matches_solo(random_8x8):
    """Any permutation of metrics should label outputs correctly."""
    agg = create_test_raster(random_8x8)
    metrics_order = ['entropy', 'contrast', 'homogeneity']

    multi = glcm_texture(agg, metric=metrics_order, window_size=3,
                         levels=16, angle=0)

    for m in metrics_order:
        solo = glcm_texture(agg, metric=m, window_size=3, levels=16, angle=0)
        np.testing.assert_allclose(
            multi.sel(metric=m).values,
            solo.values,
            rtol=1e-10, equal_nan=True,
        )


def test_all_metrics_reversed(random_8x8):
    """All six metrics reversed should still produce correct labels."""
    agg = create_test_raster(random_8x8)
    reversed_metrics = list(reversed(VALID_METRICS))

    multi = glcm_texture(agg, metric=reversed_metrics, window_size=3,
                         levels=16, angle=0)

    for m in VALID_METRICS:
        solo = glcm_texture(agg, metric=m, window_size=3, levels=16, angle=0)
        np.testing.assert_allclose(
            multi.sel(metric=m).values,
            solo.values,
            rtol=1e-10, equal_nan=True,
        )


def test_standard_order_still_works(random_8x8):
    """Standard order (which worked before the fix) should still be fine."""
    agg = create_test_raster(random_8x8)
    metrics = ['contrast', 'dissimilarity']

    multi = glcm_texture(agg, metric=metrics, window_size=3,
                         levels=16, angle=0)

    for m in metrics:
        solo = glcm_texture(agg, metric=m, window_size=3, levels=16, angle=0)
        np.testing.assert_allclose(
            multi.sel(metric=m).values,
            solo.values,
            rtol=1e-10, equal_nan=True,
        )


def test_angle_none_with_reversed_order(random_8x8):
    """angle=None averaging should also respect metric ordering."""
    agg = create_test_raster(random_8x8)
    metrics = ['energy', 'contrast']

    multi = glcm_texture(agg, metric=metrics, window_size=3, levels=16)

    for m in metrics:
        solo = glcm_texture(agg, metric=m, window_size=3, levels=16)
        np.testing.assert_allclose(
            multi.sel(metric=m).values,
            solo.values,
            rtol=1e-10, equal_nan=True,
        )


# ---- Dask backend ----

@dask_array_available
def test_dask_reversed_order_matches_numpy(random_8x8):
    """Dask backend with reversed metric order should match numpy results."""
    numpy_agg = create_test_raster(random_8x8)
    dask_agg = create_test_raster(random_8x8, backend='dask+numpy',
                                  chunks=(4, 4))
    metrics = ['entropy', 'contrast']

    np_result = glcm_texture(numpy_agg, metric=metrics, window_size=3,
                             levels=16, angle=0)
    da_result = glcm_texture(dask_agg, metric=metrics, window_size=3,
                             levels=16, angle=0)

    for m in metrics:
        np.testing.assert_allclose(
            np_result.sel(metric=m).values,
            da_result.sel(metric=m).values,
            rtol=1e-10, equal_nan=True,
        )
