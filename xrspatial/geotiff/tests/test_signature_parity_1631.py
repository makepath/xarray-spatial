"""Regression test for #1631: public write_vrt / write_geotiff_gpu
signature and docstring parity vs to_geotiff.

Three drifts were flagged by the api-consistency sweep on 2026-05-11:

1. ``write_vrt(vrt_path, source_files, **kwargs)`` swallowed every kwarg
   into ``**kwargs``. The docstring documented ``relative``, ``crs_wkt``,
   ``nodata``, but ``inspect.signature`` and IDE autocomplete saw nothing.
2. ``write_geotiff_gpu``'s ``overview_resampling`` docstring omitted
   ``'cubic'``; ``to_geotiff`` lists it and ``make_overview_gpu`` accepts
   it (falling back to CPU).
3. ``write_geotiff_gpu(data, ...)`` lacked the type hint that
   ``to_geotiff(data, ...)`` has.

This module pins each of those three guarantees against future drift.
"""
from __future__ import annotations

import importlib.util
import inspect
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)


def _gpu_available() -> bool:
    """True when cupy imports and CUDA is initialised."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)


def test_write_vrt_signature_exposes_documented_kwargs():
    """``inspect.signature(write_vrt)`` reports the three accepted kwargs.

    Prior to #1631 the public wrapper used ``**kwargs``, so
    ``inspect.signature`` only saw ``vrt_path`` and ``source_files``.
    """
    sig = inspect.signature(write_vrt)
    params = sig.parameters
    assert 'relative' in params
    assert 'crs_wkt' in params
    assert 'nodata' in params
    # Defaults must match _vrt.write_vrt
    assert params['relative'].default is True
    assert params['crs_wkt'].default is None
    assert params['nodata'].default is None
    # No more catch-all VAR_KEYWORD
    kinds = {p.kind for p in params.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds


def test_write_vrt_unknown_kwarg_rejected_at_public_level(tmp_path):
    """A typo'd kwarg now raises ``TypeError`` from the public function
    rather than from deep inside ``_vrt.write_vrt``.
    """
    arr = np.zeros((8, 8), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    tif_path = str(tmp_path / 't.tif')
    to_geotiff(da, tif_path)

    with pytest.raises(TypeError, match='typo_kwarg'):
        write_vrt(str(tmp_path / 't.vrt'), [tif_path], typo_kwarg=1)


def test_write_vrt_accepts_documented_kwargs(tmp_path):
    """Each documented kwarg round-trips through the explicit signature."""
    arr = np.zeros((8, 8), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    tif_path = str(tmp_path / 't.tif')
    to_geotiff(da, tif_path)

    vrt_path = str(tmp_path / 't.vrt')
    out = write_vrt(
        vrt_path, [tif_path],
        relative=False, crs_wkt=None, nodata=-9999.0,
    )
    assert out == vrt_path
    assert os.path.exists(vrt_path)


def test_write_geotiff_gpu_docstring_lists_cubic():
    """``overview_resampling`` docstring includes ``'cubic'`` so it
    matches ``to_geotiff`` and the underlying ``make_overview_gpu``.
    """
    doc = write_geotiff_gpu.__doc__
    assert doc is not None
    # Find the overview_resampling block
    assert 'overview_resampling' in doc
    # The block must mention cubic
    block_start = doc.index('overview_resampling')
    block_end = doc.index('bigtiff', block_start)
    block = doc[block_start:block_end]
    assert 'cubic' in block


def test_write_geotiff_gpu_data_has_type_hint():
    """``data`` parameter is annotated, matching ``to_geotiff(data, ...)``.

    The annotation also covers ``np.ndarray`` because the implementation
    accepts numpy inputs (uploaded via ``cupy.asarray(np.asarray(data))``)
    and the test suite exercises that path (e.g.
    ``test_backend_kwarg_parity_1561.py`` passes a numpy ``dummy``).
    """
    sig = inspect.signature(write_geotiff_gpu)
    data_param = sig.parameters['data']
    assert data_param.annotation is not inspect.Parameter.empty
    # The annotation is a forward reference under ``from __future__ import
    # annotations``; just confirm it mentions the documented types.
    ann_str = str(data_param.annotation)
    assert 'DataArray' in ann_str
    assert 'cupy' in ann_str
    assert 'ndarray' in ann_str  # numpy parity vs to_geotiff


@_gpu_only
def test_write_geotiff_gpu_cubic_overview_round_trip(tmp_path):
    """``overview_resampling='cubic'`` works on the GPU writer.

    Sanity check that the docstring update is not advertising an
    unsupported codec. ``make_overview_gpu`` falls back to the CPU
    cubic implementation for parity with the CPU writer.
    """
    import cupy

    arr_cpu = np.random.RandomState(0).rand(256, 256).astype(np.float32)
    arr_gpu = cupy.asarray(arr_cpu)
    da_gpu = xr.DataArray(
        arr_gpu, dims=['y', 'x'],
        coords={'y': np.arange(256.0, 0, -1), 'x': np.arange(256.0)},
    )
    path = str(tmp_path / 'cog.tif')
    write_geotiff_gpu(
        da_gpu, path,
        cog=True, tile_size=64, overview_resampling='cubic',
    )
    # Overview level 1 = 1/2 resolution
    ov = open_geotiff(path, overview_level=1)
    assert ov.shape == (128, 128)
