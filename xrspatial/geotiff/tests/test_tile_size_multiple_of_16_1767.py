"""Regression tests for issue #1767.

``to_geotiff(..., tiled=True, tile_size=...)`` previously accepted any
positive integer tile size. The TIFF 6 spec requires TileWidth and
TileLength to be multiples of 16, so values like ``tile_size=17``
produced files that the in-repo reader round-tripped but that strict
TIFF tools (libtiff, GDAL) may reject. ``to_geotiff`` now refuses
non-multiples of 16 when ``tiled=True`` and suggests the nearest
valid value(s).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff


def _make_da(shape=(32, 32)):
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return xr.DataArray(arr, dims=['y', 'x'])


def test_tile_size_17_rejected_1767(tmp_path):
    """``tile_size=17`` is not a multiple of 16 and must be rejected."""
    da = _make_da()
    out = os.path.join(str(tmp_path), 'tile_size_17_1767.tif')
    with pytest.raises(ValueError) as exc:
        to_geotiff(da, out, tiled=True, tile_size=17)
    msg = str(exc.value)
    assert 'tile_size' in msg
    assert '17' in msg
    # Hint should suggest nearest valid choices (16 and 32).
    assert '16' in msg and '32' in msg


def test_tile_size_1_rejected_1767(tmp_path):
    """``tile_size=1`` was accepted previously; now rejected because
    1 is not a multiple of 16."""
    da = _make_da((16, 16))
    out = os.path.join(str(tmp_path), 'tile_size_1_1767.tif')
    with pytest.raises(ValueError, match=r'tile_size.*multiple of 16'):
        to_geotiff(da, out, tiled=True, tile_size=1)


def test_tile_size_default_256_works_1767(tmp_path):
    """The default ``tile_size=256`` is a multiple of 16 and must work."""
    da = _make_da((256, 256))
    out = os.path.join(str(tmp_path), 'tile_size_256_1767.tif')
    to_geotiff(da, out, tiled=True, tile_size=256)
    assert os.path.exists(out)


def test_tile_size_512_works_1767(tmp_path):
    da = _make_da((512, 512))
    out = os.path.join(str(tmp_path), 'tile_size_512_1767.tif')
    to_geotiff(da, out, tiled=True, tile_size=512)
    assert os.path.exists(out)


def test_tile_size_128_works_1767(tmp_path):
    da = _make_da((128, 128))
    out = os.path.join(str(tmp_path), 'tile_size_128_1767.tif')
    to_geotiff(da, out, tiled=True, tile_size=128)
    assert os.path.exists(out)


def test_tile_size_16_works_1767(tmp_path):
    """The smallest legal tile size is 16."""
    da = _make_da((32, 32))
    out = os.path.join(str(tmp_path), 'tile_size_16_1767.tif')
    to_geotiff(da, out, tiled=True, tile_size=16)
    assert os.path.exists(out)


def test_tile_size_17_with_tiled_false_passes_1767(tmp_path):
    """``tiled=False`` ignores ``tile_size`` entirely; multiple-of-16
    validation must not fire there."""
    da = _make_da()
    out = os.path.join(str(tmp_path), 'tile_size_17_strip_1767.tif')
    # ``tiled=False`` emits a warning when a non-default tile_size is
    # passed; we only care that no ValueError fires.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        to_geotiff(da, out, tiled=False, tile_size=17)
    assert os.path.exists(out)


def test_tile_size_24_message_suggests_16_and_32_1767(tmp_path):
    """Error message names both nearest valid multiples (lower & upper)."""
    da = _make_da()
    out = os.path.join(str(tmp_path), 'tile_size_24_1767.tif')
    with pytest.raises(ValueError) as exc:
        to_geotiff(da, out, tiled=True, tile_size=24)
    msg = str(exc.value)
    assert '16' in msg
    assert '32' in msg


def test_tile_size_8_message_suggests_16_only_1767(tmp_path):
    """For ``tile_size < 16`` only the upper neighbour (16) is valid."""
    da = _make_da()
    out = os.path.join(str(tmp_path), 'tile_size_8_1767.tif')
    with pytest.raises(ValueError) as exc:
        to_geotiff(da, out, tiled=True, tile_size=8)
    msg = str(exc.value)
    assert '16' in msg
    # 0 is not a valid tile size and should not appear as a suggestion.
    assert 'tile_size=0' not in msg
