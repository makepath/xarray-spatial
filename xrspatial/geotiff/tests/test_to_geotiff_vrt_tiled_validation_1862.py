"""Regression tests for issue #1862.

``to_geotiff(..., '.vrt', tiled=False, tile_size=0)`` previously warned that
``tile_size`` was ignored, then crashed with ``ZeroDivisionError`` inside
``_write_vrt_tiled`` because the VRT writer always tiles. The ``tiled=False``
flag was never honored on the VRT path, and ``tile_size`` was only validated
when ``tiled=True``, so an invalid ``tile_size=0`` slipped through.

``to_geotiff`` now refuses ``tiled=False`` for ``.vrt`` paths up front with a
``ValueError``, and validates ``tile_size`` unconditionally on the VRT
branch so callers get a clear error before the writer divides by it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff


def _make_da(shape=(64, 64)):
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return xr.DataArray(arr, dims=['y', 'x'])


def test_vrt_rejects_tiled_false_1862(tmp_path):
    """``tiled=False`` is not a valid request for VRT output."""
    da = _make_da()
    out = os.path.join(str(tmp_path), 'vrt_tiled_false_1862.vrt')
    with pytest.raises(ValueError, match='tiled=False is not compatible'):
        to_geotiff(da, out, tiled=False)


def test_vrt_tiled_false_zero_tile_size_raises_value_error_1862(tmp_path):
    """``tiled=False`` plus ``tile_size=0`` must raise ``ValueError``,
    not the previous ``ZeroDivisionError`` from inside the writer."""
    da = _make_da()
    out = os.path.join(
        str(tmp_path), 'vrt_tiled_false_zero_1862.vrt')
    with pytest.raises(ValueError) as exc:
        to_geotiff(da, out, tiled=False, tile_size=0)
    # Either the tiled=False guard or the tile_size validator may fire
    # first; both produce ValueError, never ZeroDivisionError.
    assert not isinstance(exc.value, ZeroDivisionError)


def test_vrt_zero_tile_size_default_tiled_raises_value_error_1862(tmp_path):
    """With the default ``tiled=True``, ``tile_size=0`` must surface from
    the shared ``_validate_tile_size`` check, not a deep ``ZeroDivisionError``.
    """
    da = _make_da()
    out = os.path.join(
        str(tmp_path), 'vrt_default_tiled_zero_1862.vrt')
    with pytest.raises(ValueError, match='tile_size'):
        to_geotiff(da, out, tile_size=0)


def test_vrt_default_args_still_succeeds_1862(tmp_path):
    """Sanity: the default-args VRT write path is unaffected by the fix."""
    da = _make_da()
    out = os.path.join(str(tmp_path), 'vrt_default_1862.vrt')
    to_geotiff(da, out)
    assert os.path.exists(out)
