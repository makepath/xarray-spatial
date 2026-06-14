"""Regression tests for #3281.

The output-size dask promotion added in #3278 only applied to numpy
inputs (``if not is_dask and not is_cupy`` in ``reproject()``). An
in-memory cupy raster with a very large output still allocated the full
working set on the GPU and could fail with a device OOM.

The fix drops the ``not is_cupy`` exclusion so a large-output cupy input
is wrapped in a cupy-backed dask array and routed through
``_reproject_dask_cupy``, which has its own GPU VRAM check and a chunked
``map_blocks`` fallback. The result-type contract mirrors ``merge()``:
an output that still fits in VRAM stays eager cupy, while one that
exceeds VRAM becomes a chunked dask result.
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import has_cuda_and_cupy

try:
    import pyproj  # noqa: F401
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

HAS_CUPY = has_cuda_and_cupy()

pytestmark = pytest.mark.skipif(
    not (HAS_PYPROJ and HAS_CUPY),
    reason="cupy + CUDA device and pyproj required for GPU promotion tests",
)

# ``from xrspatial import reproject`` resolves to the function, not the
# package, so load the package module explicitly to reach the helpers.
_reproject_mod = importlib.import_module('xrspatial.reproject')


def _make_cupy_raster():
    import cupy as cp

    h, w = 60, 60
    arr = np.random.RandomState(0).rand(h, w).astype('float64')
    y = np.linspace(55, 45, h)  # north-up (descending)
    x = np.linspace(-5, 5, w)
    return xr.DataArray(
        cp.asarray(arr),
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )


def test_inmemory_cupy_below_threshold_stays_eager(monkeypatch):
    """A small cupy output keeps the eager in-memory cupy path."""
    import cupy as cp

    import dask.array as _da

    r = _make_cupy_raster()
    out = _reproject_mod.reproject(r, target_crs='EPSG:32633')

    assert not isinstance(out.data, _da.Array)
    assert isinstance(out.data, cp.ndarray)


def test_inmemory_cupy_promotion_routes_through_dask_cupy(monkeypatch):
    """Above the threshold a cupy input promotes to the dask+cupy path.

    Before #3281 the ``not is_cupy`` guard sent a large-output cupy raster
    straight to ``_reproject_inmemory_cupy`` (full GPU allocation). The
    fix routes it through ``_reproject_dask_cupy`` instead.
    """
    r = _make_cupy_raster()

    routed = {'dask_cupy': False, 'inmemory_cupy': False}
    orig_dask_cupy = _reproject_mod._reproject_dask_cupy
    orig_inmem_cupy = _reproject_mod._reproject_inmemory_cupy

    def spy_dask_cupy(*args, **kwargs):
        routed['dask_cupy'] = True
        return orig_dask_cupy(*args, **kwargs)

    def spy_inmem_cupy(*args, **kwargs):
        routed['inmemory_cupy'] = True
        return orig_inmem_cupy(*args, **kwargs)

    monkeypatch.setattr(_reproject_mod, '_reproject_dask_cupy', spy_dask_cupy)
    monkeypatch.setattr(
        _reproject_mod, '_reproject_inmemory_cupy', spy_inmem_cupy
    )
    # Force promotion without building a genuinely huge GPU array.
    monkeypatch.setattr(_reproject_mod, '_REPROJECT_OOM_THRESHOLD', 1)

    _reproject_mod.reproject(r, target_crs='EPSG:32633')

    assert routed['dask_cupy'], "promoted cupy input should hit the dask+cupy path"
    assert not routed['inmemory_cupy'], (
        "promoted cupy input must not fall back to the eager in-memory path"
    )


def test_inmemory_cupy_vram_fallback_yields_dask_with_parity(monkeypatch):
    """When the output exceeds VRAM the result is a chunked dask array.

    The chunked ``map_blocks`` fallback must match the eager cupy path
    numerically while keeping peak memory bounded by the chunk size.
    """
    import cupy as cp

    import dask.array as _da

    r = _make_cupy_raster()
    eager = _reproject_mod.reproject(r, target_crs='EPSG:32633')

    # Force promotion, then force the VRAM check to report "does not fit".
    monkeypatch.setattr(_reproject_mod, '_REPROJECT_OOM_THRESHOLD', 1)

    class _FakeDevice:
        @property
        def mem_info(self):
            return (1, 1)  # 1 byte free -> estimated output never fits

    monkeypatch.setattr(cp.cuda, 'Device', lambda *a, **k: _FakeDevice())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chunked = _reproject_mod.reproject(
            r, target_crs='EPSG:32633', chunk_size=16
        )

    assert isinstance(chunked.data, _da.Array)
    # Blocks compute to cupy even though dask advertises a numpy meta.
    assert type(chunked.data.blocks[0, 0].compute()).__module__ == 'cupy'

    eager_np = eager.data.get()
    chunked_np = chunked.data.compute().get()
    assert eager_np.shape == chunked_np.shape
    np.testing.assert_allclose(eager_np, chunked_np, equal_nan=True)
