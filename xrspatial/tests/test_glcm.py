try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial.glcm import glcm_texture, VALID_METRICS, _quantize
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)


# ---- Fixtures ----

@pytest.fixture
def texture_data():
    """10x10 random raster for GLCM tests."""
    rng = np.random.default_rng(963)
    return rng.random((10, 10)).astype(np.float64)


@pytest.fixture
def checkerboard():
    """8x8 checkerboard pattern -- high contrast texture."""
    board = np.zeros((8, 8), dtype=np.float64)
    board[::2, 1::2] = 1.0
    board[1::2, ::2] = 1.0
    return board


@pytest.fixture
def uniform():
    """8x8 constant raster -- zero texture."""
    return np.full((8, 8), 42.0, dtype=np.float64)


# ---- Validation ----

def test_glcm_rejects_non_dataarray():
    with pytest.raises(TypeError, match="xarray.DataArray"):
        glcm_texture(np.zeros((5, 5)))


def test_glcm_rejects_even_window():
    agg = xr.DataArray(np.zeros((5, 5)), dims=['y', 'x'])
    with pytest.raises(ValueError, match="odd"):
        glcm_texture(agg, window_size=4)


def test_glcm_rejects_small_window():
    agg = xr.DataArray(np.zeros((5, 5)), dims=['y', 'x'])
    with pytest.raises(ValueError, match=">= 3"):
        glcm_texture(agg, window_size=1)


def test_glcm_rejects_bad_metric():
    agg = xr.DataArray(np.zeros((8, 8)), dims=['y', 'x'])
    with pytest.raises(ValueError, match="unknown metric"):
        glcm_texture(agg, metric='bogus')


def test_glcm_rejects_bad_angle():
    agg = xr.DataArray(np.zeros((8, 8)), dims=['y', 'x'])
    with pytest.raises(ValueError, match="angle"):
        glcm_texture(agg, angle=30)


def test_glcm_rejects_bad_levels():
    agg = xr.DataArray(np.zeros((8, 8)), dims=['y', 'x'])
    with pytest.raises(ValueError, match=">= 2"):
        glcm_texture(agg, levels=1)
    with pytest.raises(ValueError, match="<= 256"):
        glcm_texture(agg, levels=300)


def test_glcm_rejects_3d():
    agg = xr.DataArray(np.zeros((3, 5, 5)), dims=['band', 'y', 'x'])
    with pytest.raises(ValueError, match="2D"):
        glcm_texture(agg)


def test_glcm_rejects_window_larger_than_raster():
    # window_size larger than the raster would iterate window_size**2 times
    # per pixel without producing any new information -- CPU-time DoS.
    agg = xr.DataArray(np.zeros((10, 10)), dims=['y', 'x'])
    with pytest.raises(ValueError, match="window_size"):
        glcm_texture(agg, window_size=1_000_001)


def test_glcm_rejects_window_larger_than_min_dim():
    # Non-square raster: cap is min(rows, cols).
    agg = xr.DataArray(np.zeros((20, 5)), dims=['y', 'x'])
    with pytest.raises(ValueError, match="<= 5"):
        glcm_texture(agg, window_size=7)


def test_glcm_window_size_equal_to_raster_ok():
    # Boundary case: window_size exactly equal to min(rows, cols) is allowed.
    agg = xr.DataArray(np.ones((5, 5)), dims=['y', 'x'])
    result = glcm_texture(agg, metric='contrast', window_size=5, levels=4)
    assert result.shape == (5, 5)


def test_glcm_rejects_distance_larger_than_half_window():
    # distance greater than window_size // 2 places every pair outside the
    # window, so no GLCM entries get populated. Reject upfront.
    agg = xr.DataArray(np.zeros((10, 10)), dims=['y', 'x'])
    with pytest.raises(ValueError, match="distance"):
        glcm_texture(agg, window_size=5, distance=10**9)


def test_glcm_distance_equal_to_half_window_ok():
    # Boundary case: distance == window_size // 2 is the largest separation
    # where a pair can still fit inside the window.
    agg = xr.DataArray(np.ones((8, 8)), dims=['y', 'x'])
    result = glcm_texture(agg, metric='contrast', window_size=5,
                          distance=2, levels=4)
    assert result.shape == (8, 8)


# ---- Quantize ----

def test_quantize_basic():
    data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    q = _quantize(data, levels=5)
    np.testing.assert_array_equal(q, [0, 1, 2, 3, 4])


def test_quantize_nan():
    data = np.array([0.0, np.nan, 1.0])
    q = _quantize(data, levels=4)
    assert q[0] == 0
    assert q[1] == -1
    assert q[2] == 3


def test_quantize_constant():
    data = np.array([5.0, 5.0, 5.0])
    q = _quantize(data, levels=8)
    np.testing.assert_array_equal(q, [0, 0, 0])


# ---- Single metric output ----

def test_single_metric_returns_2d(texture_data):
    agg = create_test_raster(texture_data)
    result = glcm_texture(agg, metric='contrast', window_size=3, levels=16)
    assert result.ndim == 2
    assert result.shape == agg.shape
    assert result.dims == agg.dims


def test_multi_metric_returns_3d(texture_data):
    agg = create_test_raster(texture_data)
    metrics = ['contrast', 'homogeneity', 'entropy']
    result = glcm_texture(agg, metric=metrics, window_size=3, levels=16)
    assert result.ndim == 3
    assert result.shape == (3, 10, 10)
    assert result.dims[0] == 'metric'
    assert list(result.coords['metric'].values) == metrics


# ---- Known-value tests ----

def test_uniform_raster_zero_contrast(uniform):
    agg = create_test_raster(uniform)
    result = glcm_texture(agg, metric='contrast', window_size=3, levels=16)
    # Uniform raster: all pixels quantize to the same level.
    # GLCM is concentrated on the diagonal -> contrast = 0.
    interior = result.values[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 0.0, atol=1e-10)


def test_uniform_raster_max_energy(uniform):
    agg = create_test_raster(uniform)
    result = glcm_texture(agg, metric='energy', window_size=3, levels=16)
    interior = result.values[1:-1, 1:-1]
    # All probability in one cell -> energy = 1.0
    np.testing.assert_allclose(interior, 1.0, atol=1e-10)


def test_checkerboard_has_high_contrast(checkerboard):
    agg = create_test_raster(checkerboard)
    contrast = glcm_texture(agg, metric='contrast', window_size=3,
                            levels=2, angle=0)
    # Every horizontal pair is (0,1) or (1,0) -> all off-diagonal.
    # Contrast should be close to maximum for 2 levels: (1-0)^2 = 1.
    interior = contrast.values[1:-1, 1:-1]
    assert np.all(interior > 0.5)


# ---- NaN handling ----

def test_all_nan_raster():
    # Regression test for issue #1408: an all-NaN window has no valid pairs,
    # so every metric should be NaN. Both the single-angle and angle=None
    # paths must agree -- previously angle=None returned 0.0 because the
    # averaging step replaced per-angle NaN with 0 before dividing by 4.
    data = np.full((6, 6), np.nan)
    agg = create_test_raster(data)
    for angle in [None, 0, 45, 90, 135]:
        for m in VALID_METRICS:
            result = glcm_texture(agg, metric=m, window_size=3, levels=8,
                                  angle=angle)
            assert result.shape == (6, 6)
            assert np.all(np.isnan(result.values)), (
                f"metric={m} angle={angle} produced finite values "
                f"on all-NaN input: {np.unique(result.values)}"
            )


def test_partial_nan(texture_data):
    data = texture_data.copy()
    data[0, :] = np.nan
    data[:, 0] = np.nan
    agg = create_test_raster(data)
    result = glcm_texture(agg, metric='contrast', window_size=3, levels=16)
    assert result.shape == (10, 10)
    # Interior cells away from NaN border should have valid values
    assert np.isfinite(result.values[3, 3])


def test_angle_none_partial_nan_uses_valid_angles():
    # Regression test for issue #1408: when only some angles produce a valid
    # value at a pixel, the angle=None result should be the mean of the valid
    # angles only -- not (sum / 4) with NaN treated as 0.
    rng = np.random.default_rng(7)
    data = rng.random((8, 8))
    agg = create_test_raster(data)

    avg = glcm_texture(agg, metric='correlation', window_size=3, levels=4)
    per_angle = []
    for ang in [0, 45, 90, 135]:
        r = glcm_texture(agg, metric='correlation', window_size=3,
                         levels=4, angle=ang)
        per_angle.append(r.values)
    expected = np.nanmean(per_angle, axis=0)
    np.testing.assert_allclose(avg.values, expected, rtol=1e-10,
                               equal_nan=True)


# ---- Single-cell / tiny rasters ----

def test_single_cell():
    # Regression test for issue #1408: a 1x1 raster has no valid pairs at any
    # angle, so the result must be NaN for both single-angle and angle=None.
    data = np.array([[5.0]])
    agg = create_test_raster(data)
    for angle in [None, 0, 45, 90, 135]:
        result = glcm_texture(agg, metric='contrast', window_size=3,
                              levels=4, angle=angle)
        assert result.shape == (1, 1)
        assert np.isnan(result.values[0, 0]), (
            f"angle={angle} produced finite value {result.values[0, 0]} "
            "on 1x1 raster (no valid pairs)"
        )


# ---- Angle parameter ----

def test_each_angle_produces_output(texture_data):
    agg = create_test_raster(texture_data)
    for ang in [0, 45, 90, 135]:
        result = glcm_texture(agg, metric='contrast', window_size=3,
                              levels=16, angle=ang)
        assert result.shape == (10, 10)
        assert np.any(np.isfinite(result.values))


def test_angle_none_averages(texture_data):
    agg = create_test_raster(texture_data)
    # angle=None averages over angles that produced a valid (non-NaN) value
    # at each pixel. Pixels where every angle is NaN stay NaN.
    avg = glcm_texture(agg, metric='contrast', window_size=3, levels=16)
    per_angle = []
    for ang in [0, 45, 90, 135]:
        r = glcm_texture(agg, metric='contrast', window_size=3,
                         levels=16, angle=ang)
        per_angle.append(r.values)
    expected = np.nanmean(per_angle, axis=0)
    np.testing.assert_allclose(avg.values, expected, rtol=1e-10,
                               equal_nan=True)


# ---- All metrics produce finite values ----

def test_all_metrics_produce_finite(texture_data):
    agg = create_test_raster(texture_data)
    for m in VALID_METRICS:
        result = glcm_texture(agg, metric=m, window_size=3, levels=16, angle=0)
        interior = result.values[2:-2, 2:-2]
        if m == 'correlation':
            # correlation can be NaN when std=0
            continue
        assert np.all(np.isfinite(interior)), f"{m} has non-finite interior"


# ---- Cross-backend: numpy vs dask ----

@dask_array_available
def test_numpy_equals_dask(texture_data):
    numpy_agg = create_test_raster(texture_data)
    dask_agg = create_test_raster(texture_data, backend='dask+numpy',
                                  chunks=(5, 5))
    for m in VALID_METRICS:
        np_result = glcm_texture(numpy_agg, metric=m, window_size=3,
                                 levels=16, angle=0)
        da_result = glcm_texture(dask_agg, metric=m, window_size=3,
                                 levels=16, angle=0)
        assert isinstance(da_result.data, da.Array)
        np.testing.assert_allclose(
            np_result.values, da_result.data.compute(),
            rtol=1e-10, equal_nan=True,
        )


@dask_array_available
def test_numpy_equals_dask_angle_none(texture_data):
    numpy_agg = create_test_raster(texture_data)
    dask_agg = create_test_raster(texture_data, backend='dask+numpy',
                                  chunks=(5, 5))
    np_result = glcm_texture(numpy_agg, metric=['contrast', 'homogeneity'],
                             window_size=5, levels=32)
    da_result = glcm_texture(dask_agg, metric=['contrast', 'homogeneity'],
                             window_size=5, levels=32)
    np.testing.assert_allclose(
        np_result.values, da_result.data.compute(),
        rtol=1e-10, equal_nan=True,
    )


@dask_array_available
@pytest.mark.parametrize("size,chunks", [
    ((8, 10), (4, 5)),
    ((7, 9), (3, 3)),
    ((12, 12), (6, 4)),
])
def test_dask_various_chunk_sizes(texture_data, size, chunks):
    rng = np.random.default_rng(963)
    data = rng.random(size).astype(np.float64)
    numpy_agg = create_test_raster(data)
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = glcm_texture(numpy_agg, metric='contrast', window_size=3,
                             levels=16, angle=0)
    da_result = glcm_texture(dask_agg, metric='contrast', window_size=3,
                             levels=16, angle=0)
    np.testing.assert_allclose(
        np_result.values, da_result.data.compute(),
        rtol=1e-10, equal_nan=True,
    )


# ---- Cross-backend: GPU ----

@cuda_and_cupy_available
def test_numpy_equals_cupy(texture_data):
    numpy_agg = create_test_raster(texture_data)
    cupy_agg = create_test_raster(texture_data, backend='cupy')
    np_result = glcm_texture(numpy_agg, metric='contrast', window_size=3,
                             levels=16, angle=0)
    cp_result = glcm_texture(cupy_agg, metric='contrast', window_size=3,
                             levels=16, angle=0)
    np.testing.assert_allclose(
        np_result.values, cp_result.data.get(),
        rtol=1e-10, equal_nan=True,
    )


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy(texture_data):
    numpy_agg = create_test_raster(texture_data)
    dask_cupy_agg = create_test_raster(texture_data, backend='dask+cupy',
                                       chunks=(5, 5))
    np_result = glcm_texture(numpy_agg, metric='contrast', window_size=3,
                             levels=16, angle=0)
    dcp_result = glcm_texture(dask_cupy_agg, metric='contrast', window_size=3,
                              levels=16, angle=0)
    np.testing.assert_allclose(
        np_result.values, dcp_result.data.compute().get(),
        rtol=1e-10, equal_nan=True,
    )


# ---- Accessor ----

def test_accessor(texture_data):
    import xrspatial  # noqa: registers accessor
    agg = create_test_raster(texture_data)
    result = agg.xrs.glcm_texture(metric='contrast', window_size=3, levels=16)
    assert result.shape == (10, 10)
    direct = glcm_texture(agg, metric='contrast', window_size=3, levels=16)
    np.testing.assert_allclose(result.values, direct.values)
