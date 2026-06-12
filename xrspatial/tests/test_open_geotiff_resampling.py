"""Tests for the ``resampling`` parameter of ``.xrs.open_geotiff``
(issue #3067 - auto-pick resampling for the auto_reproject step)."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.accessor import _resolve_resampling
from xrspatial.geotiff import to_geotiff


# ---------------------------------------------------------------------------
# _resolve_resampling unit tests (no file IO)
# ---------------------------------------------------------------------------

def _da(dtype, colormap=None):
    da = xr.DataArray(np.zeros((2, 2), dtype=dtype), dims=['y', 'x'])
    if colormap is not None:
        da.attrs['colormap'] = colormap
    return da


def test_auto_float_no_colormap_picks_bilinear():
    assert _resolve_resampling('auto', _da('float32')) == 'bilinear'


def test_auto_integer_picks_nearest():
    # integer-without-colormap also fires the is-this-really-categorical
    # advisory alongside the 'nearest' choice
    with pytest.warns(UserWarning, match="nearest"):
        assert _resolve_resampling('auto', _da('int16')) == 'nearest'
    with pytest.warns(UserWarning, match="nearest"):
        assert _resolve_resampling('auto', _da('uint8')) == 'nearest'


def test_auto_float_with_colormap_picks_nearest():
    assert _resolve_resampling('auto', _da('float32', colormap=(1, 2, 3))) \
        == 'nearest'


@pytest.mark.parametrize('mode', ['nearest', 'bilinear', 'cubic'])
def test_explicit_mode_returned_unchanged(mode):
    # explicit modes ignore the data characteristics entirely
    assert _resolve_resampling(mode, _da('int16')) == mode
    assert _resolve_resampling(mode, _da('float32')) == mode


# ---------------------------------------------------------------------------
# End-to-end forwarding into reproject(), via a spy
# ---------------------------------------------------------------------------

def _mismatch_template_and_file(tmp_path, dtype, name):
    """File in EPSG:4326, caller template in EPSG:3857 over same box."""
    height, width = 30, 30
    arr = np.arange(height * width, dtype=dtype).reshape(height, width)
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    file_da = xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326},
    )
    path = str(tmp_path / name)
    to_geotiff(file_da, path, compression='none')

    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = tr.transform(-120.25, 45.25)
    x1, y1 = tr.transform(-119.75, 44.75)
    template = xr.DataArray(
        np.zeros((6, 6), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(max(y0, y1), min(y0, y1), 6),
                'x': np.linspace(min(x0, x1), max(x0, x1), 6)},
        attrs={'crs': 3857},
    )
    return template, path


@pytest.fixture
def reproject_spy(monkeypatch):
    """Patch reproject() to record its resampling kwarg and echo input."""
    calls = {}

    def _spy(raster, *args, **kwargs):
        calls['resampling'] = kwargs.get('resampling')
        calls['count'] = calls.get('count', 0) + 1
        return raster

    # ``xrspatial.reproject`` the attribute is the function (it shadows the
    # submodule), but ``_open_geotiff_windowed`` does ``from .reproject
    # import reproject``, which resolves through the submodule. Patch the
    # attribute on the submodule object from sys.modules.
    import importlib
    rmod = importlib.import_module('xrspatial.reproject')
    monkeypatch.setattr(rmod, 'reproject', _spy)
    return calls


def test_auto_float_forwards_bilinear(tmp_path, reproject_spy):
    template, path = _mismatch_template_and_file(
        tmp_path, np.float32, 'rs3067_float.tif')
    template.xrs.open_geotiff(path, auto_reproject=True)
    assert reproject_spy['resampling'] == 'bilinear'


def test_auto_integer_forwards_nearest(tmp_path, reproject_spy):
    template, path = _mismatch_template_and_file(
        tmp_path, np.int16, 'rs3067_int.tif')
    with pytest.warns(UserWarning, match="nearest"):
        template.xrs.open_geotiff(path, auto_reproject=True)
    assert reproject_spy['resampling'] == 'nearest'


def test_explicit_override_forwarded(tmp_path, reproject_spy):
    template, path = _mismatch_template_and_file(
        tmp_path, np.int16, 'rs3067_override.tif')
    template.xrs.open_geotiff(path, auto_reproject=True, resampling='cubic')
    assert reproject_spy['resampling'] == 'cubic'


def test_categorical_real_reproject_preserves_class_ids(tmp_path):
    # End-to-end (no spy): an integer class raster reprojected under
    # 'auto' uses nearest, so every output value is an original class ID
    # rather than an averaged in-between value.
    classes = np.array([10, 20, 30, 40], dtype=np.int16)
    template, path = _mismatch_template_and_file(
        tmp_path, np.int16, 'rs3067_classids.tif')
    # Rewrite the file with only a few distinct class values
    height, width = 30, 30
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    block = np.tile(classes, (height, width // classes.size + 1))[:, :width]
    file_da = xr.DataArray(
        block.astype(np.int16), dims=['y', 'x'],
        coords={'y': y, 'x': x}, attrs={'crs': 4326},
    )
    to_geotiff(file_da, path, compression='none')

    with pytest.warns(UserWarning, match="nearest"):
        result = template.xrs.open_geotiff(path, auto_reproject=True)
    vals = np.asarray(result.data)
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert np.isin(finite, classes).all()


# ---------------------------------------------------------------------------
# Validation and no-op behaviour
# ---------------------------------------------------------------------------

def test_invalid_resampling_raises(tmp_path):
    template, path = _mismatch_template_and_file(
        tmp_path, np.float32, 'rs3067_bad.tif')
    with pytest.raises(ValueError, match="resampling must be one of"):
        template.xrs.open_geotiff(path, auto_reproject=True,
                                  resampling='bogus')


def test_no_reproject_when_crs_matches(tmp_path, reproject_spy):
    # Caller and file share CRS: no reproject, resampling unused.
    height, width = 12, 12
    arr = np.arange(height * width, dtype=np.float32).reshape(height, width)
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    file_da = xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326},
    )
    path = str(tmp_path / 'rs3067_match.tif')
    to_geotiff(file_da, path, compression='none')

    template = xr.DataArray(
        np.zeros((6, 6), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': y[3:9], 'x': x[3:9]},
        attrs={'crs': 4326},
    )
    # passing resampling here is accepted and simply ignored
    template.xrs.open_geotiff(path, auto_reproject=True, resampling='nearest')
    assert reproject_spy.get('count', 0) == 0
