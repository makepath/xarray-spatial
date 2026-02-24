import numpy as np
import pytest
import xarray as xr

from xrspatial.mahalanobis import mahalanobis
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)


# --- fixtures ---

def _band_data():
    """3 correlated bands, 8x4, with a few NaNs."""
    rng = np.random.default_rng(42)
    h, w = 8, 4
    # generate correlated bands via a mixing matrix
    raw = rng.standard_normal((3, h * w))
    mix = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ])
    correlated = mix @ raw  # (3, pixels)
    bands = [correlated[i].reshape(h, w).astype(np.float64) for i in range(3)]
    # inject NaNs at a few different positions
    bands[0][0, 0] = np.nan
    bands[1][3, 2] = np.nan
    bands[2][7, 3] = np.nan
    return bands


@pytest.fixture
def band_arrays():
    """Raw numpy band data."""
    return _band_data()


@pytest.fixture
def numpy_bands(band_arrays):
    return [create_test_raster(b, backend='numpy') for b in band_arrays]


@pytest.fixture
def dask_bands(band_arrays):
    return [create_test_raster(b, backend='dask+numpy', chunks=(4, 2))
            for b in band_arrays]


# --- analytical correctness ---

def test_identity_covariance_equals_euclidean():
    """With identity covariance, Mahalanobis == Euclidean from mean."""
    rng = np.random.default_rng(123)
    h, w = 6, 5
    band_data = [rng.standard_normal((h, w)) for _ in range(3)]
    bands = [xr.DataArray(b, dims=['y', 'x']) for b in band_data]

    mu = np.zeros(3)
    inv_cov = np.eye(3)

    result = mahalanobis(bands, mean=mu, inv_cov=inv_cov)

    # expected: sqrt(sum of squares across bands)
    stacked = np.stack(band_data, axis=0)
    expected = np.sqrt(np.sum(stacked ** 2, axis=0))
    np.testing.assert_allclose(result.values, expected, rtol=1e-12)


def test_known_covariance():
    """Verify against manual per-pixel computation."""
    h, w = 4, 3
    rng = np.random.default_rng(99)
    b1 = rng.standard_normal((h, w))
    b2 = rng.standard_normal((h, w))
    bands = [xr.DataArray(b, dims=['y', 'x']) for b in [b1, b2]]

    mu = np.array([0.5, -0.3])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    inv_cov = np.linalg.inv(cov)

    result = mahalanobis(bands, mean=mu, inv_cov=inv_cov)

    # manual per-pixel
    expected = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            x = np.array([b1[i, j], b2[i, j]])
            diff = x - mu
            expected[i, j] = np.sqrt(diff @ inv_cov @ diff)

    np.testing.assert_allclose(result.values, expected, rtol=1e-12)


# --- user-provided vs auto-computed stats ---

def test_user_stats_match_auto(numpy_bands):
    """User-provided stats should give same result as auto-computed."""
    auto_result = mahalanobis(numpy_bands)

    # extract auto-computed stats by recomputing
    from xrspatial.mahalanobis import _compute_stats_numpy
    mu, inv_cov = _compute_stats_numpy([b.data for b in numpy_bands])

    manual_result = mahalanobis(numpy_bands, mean=mu, inv_cov=inv_cov)
    np.testing.assert_allclose(
        auto_result.values, manual_result.values, rtol=1e-12
    )


# --- cross-backend: numpy ---

def test_mahalanobis_numpy(band_arrays):
    bands = [create_test_raster(b, backend='numpy') for b in band_arrays]
    result = mahalanobis(bands)
    general_output_checks(bands[0], result, verify_dtype=False)
    assert result.shape == bands[0].shape
    # NaN positions: (0,0), (3,2), (7,3)
    assert np.isnan(result.values[0, 0])
    assert np.isnan(result.values[3, 2])
    assert np.isnan(result.values[7, 3])
    # non-NaN positions should be finite positive
    valid = ~np.isnan(result.values)
    assert np.all(result.values[valid] >= 0)


# --- cross-backend: dask+numpy ---

@dask_array_available
def test_mahalanobis_dask_matches_numpy(band_arrays):
    np_bands = [create_test_raster(b, backend='numpy') for b in band_arrays]
    dk_bands = [create_test_raster(b, backend='dask+numpy', chunks=(4, 2))
                for b in band_arrays]

    np_result = mahalanobis(np_bands)
    dk_result = mahalanobis(dk_bands)

    general_output_checks(dk_bands[0], dk_result, verify_dtype=False)
    np.testing.assert_allclose(
        np_result.values, dk_result.values, rtol=1e-10, equal_nan=True
    )


# --- cross-backend: cupy ---

@cuda_and_cupy_available
def test_mahalanobis_cupy_matches_numpy(band_arrays):
    np_bands = [create_test_raster(b, backend='numpy') for b in band_arrays]
    cu_bands = [create_test_raster(b, backend='cupy') for b in band_arrays]

    np_result = mahalanobis(np_bands)
    cu_result = mahalanobis(cu_bands)

    general_output_checks(cu_bands[0], cu_result, verify_dtype=False)
    np.testing.assert_allclose(
        np_result.values, cu_result.data.get(), rtol=1e-10, equal_nan=True
    )


# --- cross-backend: dask+cupy ---

@cuda_and_cupy_available
def test_mahalanobis_dask_cupy_matches_numpy(band_arrays):
    np_bands = [create_test_raster(b, backend='numpy') for b in band_arrays]
    dc_bands = [create_test_raster(b, backend='dask+cupy', chunks=(4, 2))
                for b in band_arrays]

    np_result = mahalanobis(np_bands)
    dc_result = mahalanobis(dc_bands)

    general_output_checks(dc_bands[0], dc_result, verify_dtype=False)
    np.testing.assert_allclose(
        np_result.values, dc_result.data.compute().get(),
        rtol=1e-10, equal_nan=True
    )


# --- NaN handling ---

def test_nan_propagation():
    """NaN in any band at a pixel → NaN output."""
    h, w = 4, 4
    b1 = np.ones((h, w))
    b2 = np.ones((h, w)) * 2.0
    b1[1, 2] = np.nan
    b2[3, 0] = np.nan
    bands = [xr.DataArray(b, dims=['y', 'x']) for b in [b1, b2]]

    mu = np.array([1.0, 2.0])
    inv_cov = np.eye(2)
    result = mahalanobis(bands, mean=mu, inv_cov=inv_cov)

    assert np.isnan(result.values[1, 2])
    assert np.isnan(result.values[3, 0])
    # all other pixels: distance = 0 (all at mean)
    mask = np.ones((h, w), dtype=bool)
    mask[1, 2] = False
    mask[3, 0] = False
    np.testing.assert_allclose(result.values[mask], 0.0, atol=1e-14)


# --- error cases ---

def test_error_single_band():
    b = xr.DataArray(np.ones((4, 4)), dims=['y', 'x'])
    with pytest.raises(ValueError, match="At least 2 bands"):
        mahalanobis([b])


def test_error_mismatched_shapes():
    b1 = xr.DataArray(np.ones((4, 4)), dims=['y', 'x'])
    b2 = xr.DataArray(np.ones((4, 5)), dims=['y', 'x'])
    with pytest.raises(ValueError, match="equal shapes"):
        mahalanobis([b1, b2])


def test_error_only_mean_provided():
    b1 = xr.DataArray(np.ones((4, 4)), dims=['y', 'x'])
    b2 = xr.DataArray(np.ones((4, 4)) * 2, dims=['y', 'x'])
    with pytest.raises(ValueError, match="both.*mean.*inv_cov"):
        mahalanobis([b1, b2], mean=np.zeros(2))


def test_error_only_inv_cov_provided():
    b1 = xr.DataArray(np.ones((4, 4)), dims=['y', 'x'])
    b2 = xr.DataArray(np.ones((4, 4)) * 2, dims=['y', 'x'])
    with pytest.raises(ValueError, match="both.*mean.*inv_cov"):
        mahalanobis([b1, b2], inv_cov=np.eye(2))


def test_error_singular_covariance():
    """Identical bands → singular covariance."""
    data = np.arange(16, dtype=np.float64).reshape(4, 4)
    b1 = xr.DataArray(data, dims=['y', 'x'])
    b2 = xr.DataArray(data.copy(), dims=['y', 'x'])  # identical
    with pytest.raises(ValueError, match="singular"):
        mahalanobis([b1, b2])


def test_error_mean_wrong_shape():
    b1 = xr.DataArray(np.ones((4, 4)), dims=['y', 'x'])
    b2 = xr.DataArray(np.ones((4, 4)) * 2, dims=['y', 'x'])
    with pytest.raises(ValueError, match="mean.*shape"):
        mahalanobis([b1, b2], mean=np.zeros(3), inv_cov=np.eye(2))


def test_error_inv_cov_wrong_shape():
    b1 = xr.DataArray(np.ones((4, 4)), dims=['y', 'x'])
    b2 = xr.DataArray(np.ones((4, 4)) * 2, dims=['y', 'x'])
    with pytest.raises(ValueError, match="inv_cov.*shape"):
        mahalanobis([b1, b2], mean=np.zeros(2), inv_cov=np.eye(3))


# --- output metadata ---

def test_output_metadata():
    rng = np.random.default_rng(7)
    data1 = rng.standard_normal((6, 5))
    data2 = rng.standard_normal((6, 5))
    b1 = create_test_raster(data1, backend='numpy', name='band1')
    b2 = create_test_raster(data2, backend='numpy', name='band2')

    result = mahalanobis([b1, b2], name='my_mahal')

    assert result.name == 'my_mahal'
    assert result.dims == b1.dims
    assert result.attrs == b1.attrs
    for coord in b1.coords:
        assert np.all(b1[coord] == result[coord])


# --- accessor ---

def test_accessor():
    import xrspatial  # noqa: F401 — registers accessors

    rng = np.random.default_rng(55)
    b1 = xr.DataArray(rng.standard_normal((5, 4)), dims=['y', 'x'])
    b2 = xr.DataArray(rng.standard_normal((5, 4)), dims=['y', 'x'])
    b3 = xr.DataArray(rng.standard_normal((5, 4)), dims=['y', 'x'])

    # accessor: self is first band, other_bands is list of remaining
    acc_result = b1.xrs.mahalanobis([b2, b3])
    direct_result = mahalanobis([b1, b2, b3])

    np.testing.assert_allclose(
        acc_result.values, direct_result.values, rtol=1e-12
    )
