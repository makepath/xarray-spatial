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
# Chunk-boundary continuity (edge effects)
# ---------------------------------------------------------------------------

@dask_array_available
def test_warp_dask_different_chunks_match():
    """Warped terrain with different chunk sizes should produce the same result."""
    data_np = create_test_arr()
    t_np = generate_terrain(data_np, warp_strength=0.5)

    # use a chunk size that doesn't evenly divide 50
    data_dask = create_test_arr()
    data_dask.data = da.from_array(data_dask.data, chunks=(13, 17))
    t_dask = generate_terrain(data_dask, warp_strength=0.5)
    np.testing.assert_allclose(t_np.data, t_dask.compute().data, rtol=1e-5, atol=1e-7)


@dask_array_available
def test_worley_dask_different_chunks_match():
    """Worley blended terrain should match across chunk boundaries."""
    data_np = create_test_arr()
    t_np = generate_terrain(data_np, worley_blend=0.2)

    data_dask = create_test_arr()
    data_dask.data = da.from_array(data_dask.data, chunks=(13, 17))
    t_dask = generate_terrain(data_dask, worley_blend=0.2)
    np.testing.assert_allclose(t_np.data, t_dask.compute().data, rtol=1e-5, atol=1e-7)


@dask_array_available
def test_ridged_warp_worley_dask_matches_numpy():
    """All features combined should match numpy vs dask with odd chunk sizes."""
    data_np = create_test_arr()
    t_np = generate_terrain(
        data_np, noise_mode='ridged', warp_strength=0.4, worley_blend=0.1,
    )
    data_dask = create_test_arr()
    data_dask.data = da.from_array(data_dask.data, chunks=(13, 17))
    t_dask = generate_terrain(
        data_dask, noise_mode='ridged', warp_strength=0.4, worley_blend=0.1,
    )
    np.testing.assert_allclose(t_np.data, t_dask.compute().data, rtol=1e-5, atol=1e-7)


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


# =====================================================================
# Issue #1443: memory guard + scalar validation
# =====================================================================

import xarray as _xr_test


class TestTerrainMemoryAndValidation:

    @staticmethod
    def _template(h=8, w=8):
        return _xr_test.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': np.arange(h), 'x': np.arange(w)},
        )

    def test_numpy_memory_guard(self):
        from unittest.mock import patch
        from xrspatial.terrain import generate_terrain
        with patch(
            "xrspatial.terrain._available_memory_bytes", return_value=1
        ):
            with pytest.raises(MemoryError, match="scratch memory"):
                generate_terrain(self._template())

    def test_numpy_memory_guard_message_dimensions(self):
        from unittest.mock import patch
        from xrspatial.terrain import generate_terrain
        with patch(
            "xrspatial.terrain._available_memory_bytes", return_value=1
        ):
            with pytest.raises(MemoryError, match="8x8"):
                generate_terrain(self._template(8, 8))

    @pytest.mark.parametrize("lac", [0, -1.0, float('inf'), float('nan')])
    def test_lacunarity_rejected(self, lac):
        from xrspatial.terrain import generate_terrain
        with pytest.raises(ValueError, match="lacunarity"):
            generate_terrain(self._template(), lacunarity=lac)

    @pytest.mark.parametrize("per", [0, -0.5, float('inf'), float('nan')])
    def test_persistence_rejected(self, per):
        from xrspatial.terrain import generate_terrain
        with pytest.raises(ValueError, match="persistence"):
            generate_terrain(self._template(), persistence=per)


# =====================================================================
# Issue #3474: preserve the caller's coords / chunks / res / crs
# =====================================================================

def _georef_arr(H=40, W=30, res=30, backend='numpy'):
    """A georeferenced template: real y/x coords plus crs / res / units."""
    ys = np.arange(H) * res
    xs = np.arange(W) * res
    data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(
        data, coords={'y': ys, 'x': xs}, dims=('y', 'x'),
        name='study_area',
        attrs={'res': (res, res), 'crs': 'EPSG:5070', 'units': 'meters'},
    )
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(13, 17))
    return raster


def test_generate_terrain_preserves_caller_coords_and_attrs():
    src = _georef_arr()
    terrain = generate_terrain(src)

    np.testing.assert_array_equal(terrain.y.data, src.y.data)
    np.testing.assert_array_equal(terrain.x.data, src.x.data)
    assert terrain.attrs['res'] == (30, 30)
    assert terrain.attrs['crs'] == 'EPSG:5070'
    assert terrain.attrs['units'] == 'meters'


def test_generate_terrain_accessor_preserves_georeference():
    src = _georef_arr()
    terrain = src.xrs.generate_terrain()

    np.testing.assert_array_equal(terrain.y.data, src.y.data)
    np.testing.assert_array_equal(terrain.x.data, src.x.data)
    assert terrain.attrs['res'] == (30, 30)
    assert terrain.attrs['crs'] == 'EPSG:5070'


def test_generate_terrain_bare_template_keeps_synthetic_grid():
    """No coords / attrs on the input -> old (x_range, y_range) behaviour."""
    bare = create_test_arr()  # 50x50, dims only, no coords / attrs
    terrain = generate_terrain(bare, x_range=(0, 500), y_range=(0, 500))

    assert 'crs' not in terrain.attrs
    dx = 500 / 50
    np.testing.assert_allclose(terrain.x.data[0], dx / 2)
    np.testing.assert_allclose(terrain.attrs['res'], (dx, dx))


def test_generate_terrain_coords_without_res_attr():
    """Coords but no res attr -> res derived from coord spacing."""
    ys = np.arange(40) * 25.0
    xs = np.arange(30) * 25.0
    src = xr.DataArray(np.zeros((40, 30), dtype=np.float32),
                       coords={'y': ys, 'x': xs}, dims=('y', 'x'))
    terrain = generate_terrain(src)

    np.testing.assert_array_equal(terrain.x.data, xs)
    np.testing.assert_allclose(terrain.attrs['res'], (25.0, 25.0))


@dask_array_available
def test_generate_terrain_dask_preserves_chunks_coords_attrs():
    src = _georef_arr(backend='dask')
    terrain = generate_terrain(src)

    assert isinstance(terrain.data, da.Array)
    assert terrain.data.chunks == src.data.chunks
    np.testing.assert_array_equal(terrain.y.data, src.y.data)
    np.testing.assert_array_equal(terrain.x.data, src.x.data)
    assert terrain.attrs['crs'] == 'EPSG:5070'
    assert terrain.attrs['res'] == (30, 30)


@cuda_and_cupy_available
def test_generate_terrain_cupy_preserves_coords_attrs():
    src = _georef_arr(backend='cupy')
    terrain = generate_terrain(src)

    np.testing.assert_array_equal(terrain.y.data, src.y.data)
    np.testing.assert_array_equal(terrain.x.data, src.x.data)
    assert terrain.attrs['crs'] == 'EPSG:5070'
    assert terrain.attrs['res'] == (30, 30)


@cuda_and_cupy_available
@dask_array_available
def test_generate_terrain_dask_cupy_preserves_coords_attrs():
    src = _georef_arr(backend='dask+cupy')
    terrain = generate_terrain(src)

    assert isinstance(terrain.data, da.Array)
    assert terrain.data.chunks == src.data.chunks
    np.testing.assert_array_equal(terrain.y.data, src.y.data)
    np.testing.assert_array_equal(terrain.x.data, src.x.data)
    assert terrain.attrs['crs'] == 'EPSG:5070'
    assert terrain.attrs['res'] == (30, 30)


# =====================================================================
# Issue #3525: an all-NaN template (e.g. from_template) must be filled,
# not propagated.  data * 0 left NaN * 0 == NaN on numpy / cupy.
# =====================================================================

def _nan_template(backend='numpy'):
    """An empty all-NaN grid, like from_template() returns."""
    data = np.full((50, 50), np.nan, dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])
    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=(10, 10))
    return raster


def test_terrain_all_nan_template_is_finite():
    """All-NaN input must produce all-finite terrain, not all-NaN."""
    terrain = generate_terrain(_nan_template())
    assert np.isfinite(terrain.data).all()


def test_terrain_all_nan_template_matches_zeros_template():
    """generate_terrain ignores input values: an all-NaN template and an
    all-zeros template must yield identical terrain."""
    from_nan = generate_terrain(_nan_template())
    from_zeros = generate_terrain(create_test_arr())
    np.testing.assert_array_equal(from_nan.data, from_zeros.data)


@dask_array_available
def test_terrain_all_nan_template_dask_matches_numpy():
    t_np = generate_terrain(_nan_template())
    t_dask = generate_terrain(_nan_template(backend='dask')).compute()
    assert np.isfinite(t_dask.data).all()
    np.testing.assert_allclose(t_np.data, t_dask.data, rtol=1e-5, atol=1e-7)


@cuda_and_cupy_available
def test_terrain_all_nan_template_cupy_matches_numpy():
    t_np = generate_terrain(_nan_template())
    t_cupy = generate_terrain(_nan_template(backend='cupy'))
    assert np.isfinite(t_cupy.data.get()).all()
    np.testing.assert_allclose(t_np.data, t_cupy.data.get(),
                               rtol=1e-5, atol=1e-7)


@cuda_and_cupy_available
@dask_array_available
def test_terrain_all_nan_template_dask_cupy_matches_numpy():
    t_np = generate_terrain(_nan_template())
    t_dc = generate_terrain(_nan_template(backend='dask+cupy')).compute()
    assert np.isfinite(t_dc.data.get()).all()
    np.testing.assert_allclose(t_np.data, t_dc.data.get(),
                               rtol=1e-4, atol=1e-4)


# =====================================================================
# Issue #3574: the dask backends regenerate every cell from coordinates
# and must not materialize the template's values.  Mapping over a
# da.empty_like skeleton drops the template (e.g. from_template's
# da.full(nan)) out of the graph entirely.
# =====================================================================

def _poisoned_template(backend='dask'):
    """A dask template whose blocks raise if they are ever computed.

    generate_terrain regenerates terrain from coordinates and never reads
    these values, so its result must compute without tripping the poison.
    """
    def _boom(block):
        raise AssertionError("template values were materialized")

    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        base = da.zeros((50, 50), chunks=(10, 10), dtype=cupy.float32,
                        meta=cupy.array((), dtype=cupy.float32))
        poisoned = da.map_blocks(_boom, base, dtype=cupy.float32,
                                 meta=cupy.array((), dtype=cupy.float32))
    else:
        base = da.zeros((50, 50), chunks=(10, 10), dtype=np.float32)
        poisoned = da.map_blocks(_boom, base, dtype=np.float32,
                                 meta=np.array((), dtype=np.float32))
    return xr.DataArray(poisoned, dims=['y', 'x'])


@dask_array_available
@pytest.mark.parametrize("worley_blend", [0.0, 0.2])
def test_terrain_dask_does_not_materialize_template(worley_blend):
    """The template's values must never be read; mapping over an empty
    skeleton keeps them out of the graph (both map_blocks call sites)."""
    terrain = generate_terrain(_poisoned_template(), worley_blend=worley_blend)
    result = terrain.compute()  # raises if the template were materialized
    assert np.isfinite(result.data).all()


@dask_array_available
def test_terrain_dask_skeleton_matches_numpy():
    """Swapping the template for an empty skeleton must not change output."""
    expected = generate_terrain(create_test_arr()).data
    actual = generate_terrain(_poisoned_template()).compute().data
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


@cuda_and_cupy_available
@dask_array_available
@pytest.mark.parametrize("worley_blend", [0.0, 0.2])
def test_terrain_dask_cupy_does_not_materialize_template(worley_blend):
    terrain = generate_terrain(_poisoned_template(backend='dask+cupy'),
                               worley_blend=worley_blend)
    result = terrain.compute()
    assert np.isfinite(result.data.get()).all()


# ---------------------------------------------------------------------------
# Fused numpy fast path (worley off) -- independent correctness anchor
# ---------------------------------------------------------------------------

def _reference_terrain(height, width, seed, noise_mode='fbm', octaves=16,
                       lacunarity=2.0, persistence=0.5, zfactor=4000):
    """Recompute the worley-off terrain pipeline from the trusted, un-fused
    _perlin building block.

    The fused numpy kernel and the dask+numpy path both go through
    _gen_terrain, so dask-vs-numpy parity cannot catch a bug in the kernel
    itself, and the GPU parity tests are skipped on CPU-only CI. This mirrors
    _gen_terrain + _terrain_numpy with x_range/y_range scaled to (0, 1) (the
    default full_extent) so it stays an independent reference.
    """
    from xrspatial.perlin import _make_perm_table, _perlin

    linx = np.linspace(0, 1, width, endpoint=False, dtype=np.float32)
    liny = np.linspace(0, 1, height, endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(linx, liny)

    hm = np.zeros((height, width), dtype=np.float32)
    norm = sum(persistence ** i for i in range(octaves))
    if noise_mode == 'ridged':
        weight = np.ones((height, width), dtype=np.float32)
        for i in range(octaves):
            amp = persistence ** i
            freq = lacunarity ** i
            noise = _perlin(_make_perm_table(seed + i), x * freq, y * freq)
            noise = 1.0 - np.abs(noise)
            noise = noise * noise
            noise *= weight
            weight = np.clip(noise, 0, 1)
            hm += noise * amp
    else:
        for i in range(octaves):
            amp = persistence ** i
            freq = lacunarity ** i
            hm += _perlin(_make_perm_table(seed + i), x * freq, y * freq) * amp

    hm /= norm
    hm = hm ** 3
    hm = np.clip(hm, -1, 1)
    hm = (hm + 1) / 2
    hm[hm < 0.3] = 0
    hm *= zfactor
    return hm


# rtol is the real drift guard; the loose atol only absorbs the ridged
# feedback path's ~6e-3 absolute gap on zfactor-scaled values.  Do not
# tighten atol or the ridged case flakes.
_REF_RTOL = 1e-4
_REF_ATOL = 2e-2


@pytest.mark.parametrize('noise_mode', ['fbm', 'ridged'])
def test_fused_numpy_matches_reference(noise_mode):
    data = xr.DataArray(np.zeros((60, 80), dtype=np.float32), dims=['y', 'x'])
    out = generate_terrain(data, seed=10, noise_mode=noise_mode)
    ref = _reference_terrain(60, 80, seed=10, noise_mode=noise_mode)
    np.testing.assert_allclose(out.data, ref, rtol=_REF_RTOL, atol=_REF_ATOL)


def test_fused_numpy_matches_reference_nondefault_params():
    data = xr.DataArray(np.zeros((60, 80), dtype=np.float32), dims=['y', 'x'])
    out = generate_terrain(data, seed=7, octaves=10,
                           lacunarity=2.3, persistence=0.45)
    ref = _reference_terrain(60, 80, seed=7, octaves=10,
                             lacunarity=2.3, persistence=0.45)
    np.testing.assert_allclose(out.data, ref, rtol=_REF_RTOL, atol=_REF_ATOL)
