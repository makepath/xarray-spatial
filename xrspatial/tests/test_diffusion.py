import numpy as np
import pytest
import xarray as xr

from xrspatial.diffusion import diffuse
from xrspatial.tests.general_checks import (
    create_test_raster,
    general_output_checks,
)


# ---- helpers ----

def _make_hotspot(shape=(21, 21), center=None, value=100.0, background=0.0):
    """Create a raster with a single hot cell."""
    data = np.full(shape, background, dtype=np.float64)
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)
    data[center] = value
    return data


# ---- correctness tests ----

def test_single_step_known_values():
    """One step of diffusion on a 3x3 field with a hot center."""
    data = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})

    # dt = 0.25 * 1^2 / 1.0 = 0.25, dt_over_dx2 = 0.25
    # laplacian at center = 0+0+0+0 - 4*4 = -16
    # new center = 4 + 0.25 * 1.0 * (-16) = 0.0
    # corners stay 0 (boundary=nan, so neighbors outside are nan -> no update
    # possible, but interior cells with all-finite neighbors do update)
    result = diffuse(agg, diffusivity=1.0, steps=1, boundary='nearest')
    out = result.values

    # With 'nearest' boundary, edge cells pad with their own value (0).
    # Center: 4 + 0.25*(0+0+0+0 - 16) = 0
    # Edge cells e.g. (0,1): 0 + 0.25*(0+0+0+4 - 0) = 1.0
    # Corner (0,0): 0 + 0.25*(0+0+0+0 - 0) = 0
    assert out[1, 1] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(0.0)


def test_uniform_field_unchanged():
    """A uniform field should not change under diffusion."""
    data = np.full((10, 10), 42.0)
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
    result = diffuse(agg, diffusivity=0.5, steps=10, boundary='nearest')
    np.testing.assert_allclose(result.values, 42.0, atol=1e-10)


def test_conservation_wrap_boundary():
    """With wrap boundary, total mass should be conserved."""
    data = _make_hotspot((15, 15))
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
    result = diffuse(agg, diffusivity=0.5, steps=20, boundary='wrap')
    initial_sum = np.nansum(data)
    final_sum = np.nansum(result.values)
    np.testing.assert_allclose(final_sum, initial_sum, rtol=1e-10)


def test_nan_propagation():
    """NaN cells should remain NaN and not pollute neighbors (boundary=nearest)."""
    data = np.ones((5, 5))
    data[2, 2] = np.nan
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
    result = diffuse(agg, diffusivity=1.0, steps=1, boundary='nearest')
    assert np.isnan(result.values[2, 2])
    # Neighbors of NaN see NaN in the stencil; with the ngjit kernel
    # the Laplacian includes NaN so those cells become NaN too.
    # This is the expected behavior (NaN spreads to direct neighbors).


def test_spatially_varying_diffusivity():
    """Different diffusivity values produce different results."""
    data = np.zeros((7, 7))
    data[3, 3] = 10.0
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})

    alpha_arr = np.ones((7, 7))
    alpha_low = alpha_arr.copy()
    alpha_low[:] = 0.1
    alpha_high = alpha_arr.copy()
    alpha_high[:] = 1.0

    d_low = create_test_raster(alpha_low, attrs={'res': (1.0, 1.0)})
    d_high = create_test_raster(alpha_high, attrs={'res': (1.0, 1.0)})

    # Use a fixed small dt that's stable for both
    dt = 0.05
    r_low = diffuse(agg, diffusivity=d_low, steps=5, dt=dt, boundary='nearest')
    r_high = diffuse(agg, diffusivity=d_high, steps=5, dt=dt, boundary='nearest')

    # Higher diffusivity should spread the hotspot more (lower peak)
    assert r_high.values[3, 3] < r_low.values[3, 3]


def test_auto_dt():
    """Auto dt should satisfy the CFL condition."""
    data = _make_hotspot((11, 11))
    agg = create_test_raster(data, attrs={'res': (2.0, 2.0)})

    # dx=2, alpha=0.5 -> dt = 0.25 * 4 / 0.5 = 2.0
    # With auto dt, 10 steps should complete without blowup
    result = diffuse(agg, diffusivity=0.5, steps=10, boundary='nearest')
    assert np.all(np.isfinite(result.values))
    assert result.values.max() <= 100.0  # should not exceed initial max


def test_multiple_steps_vs_loop():
    """Running steps=N should equal running steps=1 N times."""
    data = _make_hotspot((9, 9))
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})

    result_batch = diffuse(agg, diffusivity=0.5, steps=5, boundary='nearest')

    # manual loop
    current = agg
    for _ in range(5):
        current = diffuse(current, diffusivity=0.5, steps=1, boundary='nearest')

    np.testing.assert_allclose(
        result_batch.values, current.values, rtol=1e-10
    )


# ---- output structure tests ----

def test_output_preserves_metadata():
    """Output should keep dims, coords, and attrs from input."""
    data = _make_hotspot((8, 8))
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
    result = diffuse(agg, diffusivity=1.0, steps=1, boundary='nearest')
    general_output_checks(agg, result, verify_attrs=True)


# ---- dask backend ----

try:
    import dask.array as _da
    _has_dask = True
except ImportError:
    _has_dask = False

dask_array_available = pytest.mark.skipif(
    not _has_dask, reason='dask not available'
)


@dask_array_available
def test_dask_matches_numpy():
    """Dask backend should produce the same result as numpy."""
    data = _make_hotspot((11, 11))
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1.0, 1.0)})
    dask_agg = create_test_raster(data, backend='dask', attrs={'res': (1.0, 1.0)},
                                  chunks=(6, 6))

    np_result = diffuse(numpy_agg, diffusivity=0.5, steps=5, boundary='nearest')
    dk_result = diffuse(dask_agg, diffusivity=0.5, steps=5, boundary='nearest')

    general_output_checks(dask_agg, dk_result, verify_attrs=True)
    np.testing.assert_allclose(
        np_result.values, dk_result.data.compute(), rtol=1e-10
    )


@dask_array_available
def test_dask_wrap_boundary():
    """Dask wrap boundary should conserve mass like numpy."""
    data = _make_hotspot((15, 15))
    dask_agg = create_test_raster(data, backend='dask', attrs={'res': (1.0, 1.0)},
                                  chunks=(8, 8))
    result = diffuse(dask_agg, diffusivity=0.5, steps=10, boundary='wrap')
    final_sum = np.nansum(result.data.compute())
    np.testing.assert_allclose(final_sum, np.nansum(data), rtol=1e-10)


# ---- validation tests ----

def test_invalid_steps():
    data = np.ones((5, 5))
    agg = create_test_raster(data)
    with pytest.raises((TypeError, ValueError)):
        diffuse(agg, steps=0)


def test_invalid_diffusivity():
    data = np.ones((5, 5))
    agg = create_test_raster(data)
    with pytest.raises(ValueError):
        diffuse(agg, diffusivity=-1.0)


def test_invalid_boundary():
    data = np.ones((5, 5))
    agg = create_test_raster(data)
    with pytest.raises(ValueError):
        diffuse(agg, boundary='invalid')


def test_diffusivity_shape_mismatch():
    data = np.ones((5, 5))
    agg = create_test_raster(data)
    bad_alpha = xr.DataArray(np.ones((3, 3)))
    with pytest.raises(ValueError, match='does not match'):
        diffuse(agg, diffusivity=bad_alpha)


def test_non_dataarray_input():
    with pytest.raises(TypeError):
        diffuse(np.ones((5, 5)))


# ---- security guards (issue #1267) ----

def test_steps_upper_bound_rejected():
    """`steps` above the safety cap should raise ValueError."""
    data = np.ones((4, 4))
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match='steps'):
        diffuse(agg, steps=10**12)


def test_max_steps_constant_documented():
    """The documented step cap should match _MAX_STEPS."""
    from xrspatial.diffusion import _MAX_STEPS
    assert _MAX_STEPS == 100_000


def test_oversize_raster_raises_memory_error(monkeypatch):
    """A raster whose buffers exceed the memory budget should raise."""
    import xrspatial.diffusion as _diff

    # Pretend the host has only 16 MB of available RAM.
    monkeypatch.setattr(_diff, '_available_memory_bytes',
                        lambda: 16 * 1024 * 1024)

    # 4096x4096 float64 buffers = ~512 MB peak working set, well over
    # 50% of 16 MB.
    data = np.ones((4096, 4096), dtype=np.float64)
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})

    with pytest.raises(MemoryError, match='diffuse'):
        diffuse(agg, diffusivity=1.0, steps=1, boundary='nearest')


def test_dask_scalar_diffusivity_does_not_materialize_alpha(monkeypatch):
    """Dask + scalar diffusivity must not call np.full at agg.shape.

    Regression for issue #1267: the public API used to call
    np.full(agg.shape, alpha) up front, even when the dispatched
    backend was dask + scalar (which can pass the scalar through).
    A 100kx100k input would force an 80 GB numpy alloc.
    """
    if not _has_dask:
        pytest.skip('dask not available')

    import xrspatial.diffusion as _diff

    calls = []
    real_full = np.full

    def tracking_full(shape, *args, **kwargs):
        calls.append(tuple(shape) if hasattr(shape, '__iter__') else (shape,))
        return real_full(shape, *args, **kwargs)

    monkeypatch.setattr(_diff.np, 'full', tracking_full)

    data = np.ones((64, 64), dtype=np.float64)
    dask_agg = create_test_raster(data, backend='dask',
                                  attrs={'res': (1.0, 1.0)},
                                  chunks=(16, 16))

    diffuse(dask_agg, diffusivity=0.5, steps=2,
            boundary='nearest').data.compute()

    full_shape_calls = [c for c in calls if c == data.shape]
    assert full_shape_calls == [], (
        f"np.full was called with the full raster shape {data.shape}: "
        f"{full_shape_calls}. Scalar diffusivity on a dask path should "
        f"not materialise a full numpy alpha array."
    )


def test_dask_scalar_matches_dask_array_diffusivity():
    """Scalar-alpha and uniform-array-alpha dask paths should agree."""
    if not _has_dask:
        pytest.skip('dask not available')

    data = _make_hotspot((11, 11))
    dask_agg = create_test_raster(data, backend='dask',
                                  attrs={'res': (1.0, 1.0)},
                                  chunks=(6, 6))
    alpha_xr = create_test_raster(np.full((11, 11), 0.5),
                                  backend='dask',
                                  attrs={'res': (1.0, 1.0)},
                                  chunks=(6, 6))

    r_scalar = diffuse(dask_agg, diffusivity=0.5, steps=3,
                       boundary='nearest')
    r_array = diffuse(dask_agg, diffusivity=alpha_xr, steps=3,
                      boundary='nearest')

    np.testing.assert_allclose(
        r_scalar.data.compute(), r_array.data.compute(), rtol=1e-12
    )


def test_oversize_raster_raises_before_alpha_alloc(monkeypatch):
    """The memory guard fires before np.full(agg.shape) runs."""
    import xrspatial.diffusion as _diff

    monkeypatch.setattr(_diff, '_available_memory_bytes',
                        lambda: 16 * 1024 * 1024)

    full_calls = []
    real_full = np.full

    def tracking_full(shape, *args, **kwargs):
        full_calls.append(shape)
        return real_full(shape, *args, **kwargs)

    monkeypatch.setattr(_diff.np, 'full', tracking_full)

    data = np.ones((4096, 4096), dtype=np.float64)
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})

    with pytest.raises(MemoryError):
        diffuse(agg, diffusivity=1.0, steps=1, boundary='nearest')

    oversize_full_calls = [
        c for c in full_calls
        if hasattr(c, '__iter__') and tuple(c) == data.shape
    ]
    assert oversize_full_calls == [], (
        f"np.full was called with the oversize shape {data.shape} "
        f"before the memory guard: {oversize_full_calls}"
    )


# ---- CFL stability tests ----

def test_dt_above_cfl_bound_raises():
    """User-supplied dt above the CFL bound must raise ValueError.

    With dx=1 and alpha=1, the stable bound is 0.25; dt=1.0 is 4x too large
    and would diverge silently to Inf/NaN over many steps.
    """
    data = np.ones((10, 10)) + 0.01 * np.arange(100).reshape(10, 10)
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
    with pytest.raises(ValueError, match='CFL'):
        diffuse(agg, diffusivity=1.0, steps=10, dt=1.0, boundary='nearest')


def test_dt_at_cfl_bound_accepted():
    """dt exactly at the CFL bound is the auto-dt choice and must be allowed."""
    data = _make_hotspot((11, 11))
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
    # alpha=1, dx=1 -> cfl_max = 0.25
    result = diffuse(agg, diffusivity=1.0, steps=5, dt=0.25, boundary='nearest')
    assert np.all(np.isfinite(result.values))


def test_auto_dt_still_works():
    """The CFL guard must not break the dt=None path."""
    data = _make_hotspot((11, 11))
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})
    result = diffuse(agg, diffusivity=1.0, steps=10, dt=None, boundary='nearest')
    assert np.all(np.isfinite(result.values))


def test_cfl_bound_uses_max_alpha():
    """For spatially varying alpha, the CFL bound is set by max(alpha)."""
    data = _make_hotspot((7, 7))
    agg = create_test_raster(data, attrs={'res': (1.0, 1.0)})

    alpha_arr = np.full((7, 7), 0.1)
    alpha_arr[3, 3] = 2.0  # peak alpha at the center
    alpha = create_test_raster(alpha_arr, attrs={'res': (1.0, 1.0)})

    # cfl_max = 0.25 * 1 / 2.0 = 0.125; dt=0.2 should be rejected
    with pytest.raises(ValueError, match='CFL'):
        diffuse(agg, diffusivity=alpha, steps=5, dt=0.2, boundary='nearest')

    # dt at the bound for max alpha is fine
    result = diffuse(agg, diffusivity=alpha, steps=5, dt=0.125,
                     boundary='nearest')
    assert np.all(np.isfinite(result.values))
