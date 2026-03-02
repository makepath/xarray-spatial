try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial import generate_terrain
from xrspatial.tests.general_checks import cuda_and_cupy_available
from xrspatial.tests.general_checks import dask_array_available
from xrspatial.utils import has_cuda_and_cupy


def create_test_arr(backend='numpy'):
    W = 50
    H = 50
    data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])

    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    # TODO: restructure dask test cases to use skips if da is None
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(10, 10))

    return raster


# ---------------------------------------------------------------------------
# Basic terrain generation (existing behaviour)
# ---------------------------------------------------------------------------

def test_terrain_cpu():
    data_numpy = create_test_arr()
    terrain_numpy = generate_terrain(data_numpy)
    assert isinstance(terrain_numpy, xr.DataArray)


@dask_array_available
def test_terrain_dask_cpu():
    data_numpy = create_test_arr()
    terrain_numpy = generate_terrain(data_numpy)
    data_dask = create_test_arr(backend='dask')
    terrain_dask = generate_terrain(data_dask)
    assert isinstance(terrain_dask.data, da.Array)

    terrain_dask = terrain_dask.compute()
    np.testing.assert_allclose(terrain_numpy.data, terrain_dask.data, rtol=1e-05, atol=1e-07)


@cuda_and_cupy_available
def test_terrain_gpu():
    data_numpy = create_test_arr()
    terrain_numpy = generate_terrain(data_numpy)

    data_cupy = create_test_arr(backend='cupy')
    terrain_cupy = generate_terrain(data_cupy)

    np.testing.assert_allclose(terrain_numpy.data, terrain_cupy.data.get(), rtol=1e-05, atol=1e-07)


# ---------------------------------------------------------------------------
# Lacunarity / persistence
# ---------------------------------------------------------------------------

def test_lacunarity_persistence_changes_output():
    """Non-default lacunarity/persistence should produce different output."""
    data = create_test_arr()
    default = generate_terrain(data)
    changed = generate_terrain(create_test_arr(), lacunarity=3.0, persistence=0.4)
    assert not np.allclose(default.data, changed.data)


@dask_array_available
def test_lacunarity_dask_matches_numpy():
    data_np = create_test_arr()
    t_np = generate_terrain(data_np, lacunarity=3.0, persistence=0.4)
    data_dask = create_test_arr(backend='dask')
    t_dask = generate_terrain(data_dask, lacunarity=3.0, persistence=0.4)
    np.testing.assert_allclose(t_np.data, t_dask.compute().data, rtol=1e-5, atol=1e-7)


@cuda_and_cupy_available
def test_lacunarity_gpu_matches_numpy():
    data_np = create_test_arr()
    t_np = generate_terrain(data_np, lacunarity=3.0, persistence=0.4)
    data_cupy = create_test_arr(backend='cupy')
    t_cupy = generate_terrain(data_cupy, lacunarity=3.0, persistence=0.4)
    np.testing.assert_allclose(t_np.data, t_cupy.data.get(), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Adaptive octaves
# ---------------------------------------------------------------------------

def test_adaptive_octaves():
    """octaves=None should work and use ceil(log2(min(H,W)))."""
    data = create_test_arr()
    terrain = generate_terrain(data, octaves=None)
    assert isinstance(terrain, xr.DataArray)
    assert terrain.shape == (50, 50)


# ---------------------------------------------------------------------------
# Ridged multifractal noise
# ---------------------------------------------------------------------------

def test_ridged_differs_from_fbm():
    data = create_test_arr()
    fbm = generate_terrain(data, noise_mode='fbm')
    ridged = generate_terrain(create_test_arr(), noise_mode='ridged')
    assert not np.allclose(fbm.data, ridged.data)


@dask_array_available
def test_ridged_dask_matches_numpy():
    t_np = generate_terrain(create_test_arr(), noise_mode='ridged')
    t_dask = generate_terrain(create_test_arr(backend='dask'), noise_mode='ridged')
    np.testing.assert_allclose(t_np.data, t_dask.compute().data, rtol=1e-5, atol=1e-7)


@cuda_and_cupy_available
def test_ridged_gpu_matches_numpy():
    t_np = generate_terrain(create_test_arr(), noise_mode='ridged')
    t_cupy = generate_terrain(create_test_arr(backend='cupy'), noise_mode='ridged')
    np.testing.assert_allclose(t_np.data, t_cupy.data.get(), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Domain warping
# ---------------------------------------------------------------------------

def test_warp_zero_is_noop():
    """warp_strength=0 should match the unwarped result."""
    data = create_test_arr()
    no_warp = generate_terrain(data)
    with_zero_warp = generate_terrain(create_test_arr(), warp_strength=0.0)
    np.testing.assert_allclose(no_warp.data, with_zero_warp.data)


def test_warp_changes_output():
    data = create_test_arr()
    no_warp = generate_terrain(data)
    warped = generate_terrain(create_test_arr(), warp_strength=0.5)
    assert not np.allclose(no_warp.data, warped.data)


@dask_array_available
def test_warp_dask_matches_numpy():
    t_np = generate_terrain(create_test_arr(), warp_strength=0.5)
    t_dask = generate_terrain(create_test_arr(backend='dask'), warp_strength=0.5)
    np.testing.assert_allclose(t_np.data, t_dask.compute().data, rtol=1e-5, atol=1e-7)


@cuda_and_cupy_available
def test_warp_gpu_matches_numpy():
    t_np = generate_terrain(create_test_arr(), warp_strength=0.5)
    t_cupy = generate_terrain(create_test_arr(backend='cupy'), warp_strength=0.5)
    np.testing.assert_allclose(t_np.data, t_cupy.data.get(), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Worley blending
# ---------------------------------------------------------------------------

def test_worley_zero_is_noop():
    """worley_blend=0 should match the default result."""
    data = create_test_arr()
    no_worley = generate_terrain(data)
    with_zero_worley = generate_terrain(create_test_arr(), worley_blend=0.0)
    np.testing.assert_allclose(no_worley.data, with_zero_worley.data)


def test_worley_changes_output():
    data = create_test_arr()
    no_worley = generate_terrain(data)
    with_worley = generate_terrain(create_test_arr(), worley_blend=0.2)
    assert not np.allclose(no_worley.data, with_worley.data)


@dask_array_available
def test_worley_dask_matches_numpy():
    t_np = generate_terrain(create_test_arr(), worley_blend=0.2)
    t_dask = generate_terrain(create_test_arr(backend='dask'), worley_blend=0.2)
    np.testing.assert_allclose(t_np.data, t_dask.compute().data, rtol=1e-5, atol=1e-7)


@cuda_and_cupy_available
def test_worley_gpu_matches_numpy():
    t_np = generate_terrain(create_test_arr(), worley_blend=0.2)
    t_cupy = generate_terrain(create_test_arr(backend='cupy'), worley_blend=0.2)
    np.testing.assert_allclose(t_np.data, t_cupy.data.get(), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Erosion
# ---------------------------------------------------------------------------

def test_erosion_lowers_peaks():
    """Eroded terrain should have a lower (or equal) max than uneroded."""
    data = create_test_arr()
    uneroded = generate_terrain(data, zfactor=1000)
    eroded = generate_terrain(create_test_arr(), zfactor=1000,
                              erode=True, erosion_iterations=5000)
    # erosion moves mass around -- max should generally decrease
    assert eroded.data.max() <= uneroded.data.max() + 1e-3


def test_erosion_preserves_shape():
    data = create_test_arr()
    eroded = generate_terrain(data, erode=True, erosion_iterations=1000)
    assert eroded.shape == (50, 50)


# ---------------------------------------------------------------------------
# Combined features
# ---------------------------------------------------------------------------

def test_all_features_combined():
    """Smoke test: all features enabled at once."""
    data = create_test_arr()
    terrain = generate_terrain(
        data,
        noise_mode='ridged',
        warp_strength=0.4,
        worley_blend=0.1,
        lacunarity=2.5,
        persistence=0.45,
        octaves=8,
        erode=True,
        erosion_iterations=1000,
    )
    assert isinstance(terrain, xr.DataArray)
    assert terrain.shape == (50, 50)
    assert np.isfinite(terrain.data).all()


@dask_array_available
def test_all_features_dask():
    """Smoke test: all features enabled with dask backend."""
    data = create_test_arr(backend='dask')
    terrain = generate_terrain(
        data,
        noise_mode='ridged',
        warp_strength=0.4,
        worley_blend=0.1,
        lacunarity=2.5,
        persistence=0.45,
        octaves=8,
        erode=True,
        erosion_iterations=1000,
    )
    assert isinstance(terrain, xr.DataArray)
    terrain_computed = terrain.compute()
    assert terrain_computed.shape == (50, 50)
    assert np.isfinite(terrain_computed.data).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_noise_mode():
    data = create_test_arr()
    with pytest.raises(ValueError, match="noise_mode"):
        generate_terrain(data, noise_mode='invalid')


def test_negative_octaves():
    data = create_test_arr()
    with pytest.raises(ValueError, match="octaves"):
        generate_terrain(data, octaves=0)
