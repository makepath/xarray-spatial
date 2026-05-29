"""Test-coverage sweep additions for ``xrspatial.reproject`` (2026-05-27).

Closes two backend-coverage gaps left after the 2026-05-10 sweep:

1. ``bounds_policy`` (added in #2187) only had numpy and dask+numpy
   coverage. The cupy and dask+cupy backends were not exercised, so a
   regression that broke the bounds-derivation heuristics on a GPU input
   would ship undetected. ``bounds_policy`` is computed before the
   backend split inside ``_compute_output_grid``, but the surrounding
   ``reproject()`` call still routes through the backend-specific
   chunk worker -- pinning the policy through cupy / dask+cupy locks in
   that the policy keyword survives every dispatch.

2. ``test_reproject_handles_inf_input`` covered only the eager numpy
   path. ``+/-Inf`` pixels are common in real raster data (often as the
   result of an upstream divide or a corrupt source), and the cupy,
   dask, and dask+cupy chunk workers each have their own copy of the
   bilinear / cubic resampler math. A regression that started raising
   on Inf in any one backend would not surface from the existing test.
   Together the four new tests pin that every backend completes without
   crashing on +/-Inf input.

Both gaps map to **Cat 1 -- Backend coverage** in the sweep template
(``MEDIUM`` severity: a real bug could ship undetected, but the public
contract on the failure paths is "implementation-defined" so the test
asserts on geometry, not pixel values).
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import has_cuda_and_cupy

try:
    import pyproj  # noqa: F401
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import cupy as cp
except ImportError:
    cp = None

# Gate GPU tests on a real CUDA runtime probe, not just that ``cupy`` imports.
# An import-only check leaves tests erroring with ``cudaErrorInsufficientDriver``
# on hosts where ``cupy`` is installed but the driver is missing or too old.
HAS_CUPY = has_cuda_and_cupy()


pytestmark = pytest.mark.skipif(
    not HAS_PYPROJ, reason="pyproj required for reproject tests"
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _benign_geographic(h=32, w=32, seed=0):
    """Mid-latitude raster, well clear of any projection singularity."""
    data = np.random.RandomState(seed).rand(h, w).astype(np.float32)
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, h),
                'x': np.linspace(-5, 5, w)},
        attrs={'crs': 'EPSG:4326'},
    )


def _global_geographic(h=50, w=100, seed=0):
    """Global raster that triggers the auto blow-up heuristic when
    projected to Web Mercator (polar singularity on the y axis)."""
    data = np.random.RandomState(seed).rand(h, w).astype(np.float32)
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.linspace(90, -90, h),
                'x': np.linspace(-180, 180, w)},
        attrs={'crs': 'EPSG:4326'},
    )


def _inf_raster(h=32, w=32):
    """Raster with one +Inf and one -Inf cell at fixed positions."""
    data = np.ones((h, w), dtype=np.float64)
    data[0, 0] = np.inf
    data[1, 1] = -np.inf
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, h), 'x': np.linspace(-5, 5, w)},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )


# ---------------------------------------------------------------------------
# Gap 1: bounds_policy on cupy / dask+cupy backends
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
class TestBoundsPolicyCupy:
    """bounds_policy must thread through the cupy backend.

    The numpy and dask+numpy paths are already covered by
    ``TestBoundsPolicy``. Without these tests, a regression that dropped
    the kwarg from the cupy / dask+cupy dispatch would not surface.
    """

    def test_raw_policy_cupy_backend(self):
        """bounds_policy='raw' returns a finite output on a cupy raster."""
        from xrspatial.reproject import reproject

        host = _benign_geographic()
        raster = host.copy(data=cp.asarray(host.values))
        out = reproject(raster, 'EPSG:3857', bounds_policy='raw')
        # Output stays on GPU.
        assert hasattr(out.data, 'device') or hasattr(out.data, 'get'), (
            "expected cupy-backed output, got "
            f"{type(out.data).__name__}"
        )
        # Sanity: at least one finite pixel.
        arr = out.data.get() if hasattr(out.data, 'get') else np.asarray(out.data)
        assert np.isfinite(arr).any()

    def test_clamp_policy_cupy_matches_numpy(self):
        """Bounds policy effect must match numpy bit-for-bit on the same
        grid (the kwarg only changes ``_compute_output_grid``, not the
        per-pixel resampler)."""
        from xrspatial.reproject import reproject

        host = _global_geographic()
        gpu = host.copy(data=cp.asarray(host.values))
        # Pin to a fixed bounds via clamp to avoid 2/98 percentile randomness
        # on tiny grids.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            np_out = reproject(host, 'EPSG:3857', bounds_policy='clamp')
            gpu_out = reproject(gpu, 'EPSG:3857', bounds_policy='clamp')

        np.testing.assert_allclose(
            np_out.coords['x'].values, gpu_out.coords['x'].values,
            rtol=1e-10, atol=1e-10,
        )
        np.testing.assert_allclose(
            np_out.coords['y'].values, gpu_out.coords['y'].values,
            rtol=1e-10, atol=1e-10,
        )
        np_vals = np_out.values
        gpu_vals = (gpu_out.data.get() if hasattr(gpu_out.data, 'get')
                    else np.asarray(gpu_out.data))
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(gpu_vals))
        finite = np.isfinite(np_vals)
        if finite.any():
            np.testing.assert_allclose(
                np_vals[finite], gpu_vals[finite], rtol=1e-5, atol=1e-5,
            )

    def test_invalid_policy_rejected_cupy(self):
        """Validation must fire before the backend dispatch on cupy too."""
        from xrspatial.reproject import reproject

        host = _benign_geographic()
        raster = host.copy(data=cp.asarray(host.values))
        with pytest.raises(ValueError, match=r"bounds_policy"):
            reproject(raster, 'EPSG:3857', bounds_policy='bogus')


@pytest.mark.skipif(
    not (HAS_CUPY and HAS_DASK), reason="cupy and dask required",
)
class TestBoundsPolicyDaskCupy:
    """bounds_policy must thread through the dask+cupy backend."""

    def test_raw_policy_dask_cupy_backend(self):
        """bounds_policy='raw' works with a dask+cupy raster."""
        from xrspatial.reproject import reproject

        host = _benign_geographic()
        gpu_chunked = da.from_array(cp.asarray(host.values), chunks=(16, 16))
        raster = host.copy(data=gpu_chunked)
        out = reproject(raster, 'EPSG:3857', bounds_policy='raw')
        # The dask+cupy fast path materializes the GPU result eagerly
        # when the output fits in VRAM (see ``_reproject_dask_cupy``).
        # The chunked fallback returns a lazy dask array. Either is
        # acceptable here -- the contract is that the data stays on the
        # GPU and at least one cell is finite.
        if hasattr(out.data, 'dask'):
            arr = out.compute().data
        else:
            arr = out.data
        assert isinstance(arr, cp.ndarray), (
            f"expected cupy-backed output, got {type(arr).__name__}"
        )
        host_arr = arr.get()
        assert np.isfinite(host_arr).any()

    def test_clamp_policy_dask_cupy_matches_numpy(self):
        """bounds_policy='clamp' produces the same output grid as the numpy
        path on a dask+cupy raster."""
        from xrspatial.reproject import reproject

        host = _global_geographic()
        gpu_chunked = da.from_array(cp.asarray(host.values), chunks=(25, 50))
        raster = host.copy(data=gpu_chunked)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            np_out = reproject(host, 'EPSG:3857', bounds_policy='clamp')
            dgpu_raw = reproject(raster, 'EPSG:3857', bounds_policy='clamp')

        # Materialize the dask+cupy result if still lazy.
        if hasattr(dgpu_raw.data, 'dask'):
            dgpu_out = dgpu_raw.compute()
        else:
            dgpu_out = dgpu_raw

        np.testing.assert_allclose(
            np_out.coords['x'].values, dgpu_out.coords['x'].values,
            rtol=1e-10, atol=1e-10,
        )
        np.testing.assert_allclose(
            np_out.coords['y'].values, dgpu_out.coords['y'].values,
            rtol=1e-10, atol=1e-10,
        )

    def test_invalid_policy_rejected_dask_cupy(self):
        """Validation must fire before backend dispatch on dask+cupy too."""
        from xrspatial.reproject import reproject

        host = _benign_geographic()
        gpu_chunked = da.from_array(cp.asarray(host.values), chunks=(16, 16))
        raster = host.copy(data=gpu_chunked)
        with pytest.raises(ValueError, match=r"bounds_policy"):
            reproject(raster, 'EPSG:3857', bounds_policy='bogus')


# ---------------------------------------------------------------------------
# Gap 2: Inf input handling across all backends
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DASK, reason="dask required")
def test_reproject_inf_input_dask_backend():
    """+/-Inf input on the dask+numpy backend must not crash.

    Pins the dask map_blocks chunk worker's behaviour on Inf cells. The
    output may propagate Inf or coerce to NaN -- both are acceptable --
    but the call must return a valid 2-D DataArray with the requested
    output geometry.
    """
    from xrspatial.reproject import reproject

    raster = _inf_raster()
    raster = raster.copy(data=da.from_array(raster.values, chunks=(16, 16)))
    out = reproject(raster, 'EPSG:32633')
    assert hasattr(out.data, 'dask'), "expected dask-backed output"
    computed = out.compute()
    assert computed.ndim == 2
    assert computed.shape[0] >= 1 and computed.shape[1] >= 1


@pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
def test_reproject_inf_input_cupy_backend():
    """+/-Inf input on the cupy backend must not crash."""
    from xrspatial.reproject import reproject

    raster = _inf_raster()
    raster = raster.copy(data=cp.asarray(raster.values))
    out = reproject(raster, 'EPSG:32633')
    assert out.ndim == 2
    arr = out.data.get() if hasattr(out.data, 'get') else np.asarray(out.data)
    assert arr.shape[0] >= 1 and arr.shape[1] >= 1


@pytest.mark.skipif(
    not (HAS_CUPY and HAS_DASK), reason="cupy and dask required",
)
def test_reproject_inf_input_dask_cupy_backend():
    """+/-Inf input on the dask+cupy backend must not crash.

    The dask+cupy fast path returns the result eagerly when it fits in
    VRAM. The fallback returns a lazy dask array. Both are acceptable;
    the contract checked here is that no kernel raises on +/-Inf cells.
    """
    from xrspatial.reproject import reproject

    raster = _inf_raster()
    raster = raster.copy(
        data=da.from_array(cp.asarray(raster.values), chunks=(16, 16))
    )
    out = reproject(raster, 'EPSG:32633')
    if hasattr(out.data, 'dask'):
        out = out.compute()
    assert out.ndim == 2
    arr = (out.data.get() if hasattr(out.data, 'get')
           else np.asarray(out.data))
    assert arr.shape[0] >= 1 and arr.shape[1] >= 1


@pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
def test_reproject_all_inf_cupy_backend():
    """All-Inf raster on cupy must produce a valid output.

    The eager numpy path already pins +/-Inf at scattered positions;
    this pins the harder case where the source has no finite cells at
    all and every output pixel must come from a non-finite source.
    """
    from xrspatial.reproject import reproject

    data = np.full((16, 16), np.inf, dtype=np.float64)
    host = xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, 16),
                'x': np.linspace(-5, 5, 16)},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )
    raster = host.copy(data=cp.asarray(data))
    out = reproject(raster, 'EPSG:32633')
    assert out.ndim == 2
    arr = out.data.get() if hasattr(out.data, 'get') else np.asarray(out.data)
    # Every cell is either Inf or NaN; no finite values can appear.
    finite_mask = np.isfinite(arr)
    assert not finite_mask.any(), (
        f"all-Inf input should not produce finite outputs; "
        f"got {finite_mask.sum()} finite cell(s)"
    )
