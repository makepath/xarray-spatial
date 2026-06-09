"""Tests for ``.xrs.open_geotiff(coregister=True)`` (issue #3069 -
unpack + reproject + resample onto the caller's exact grid)."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.accessor import _caller_grid
from xrspatial.geotiff import to_geotiff


# ---------------------------------------------------------------------------
# _caller_grid unit tests
# ---------------------------------------------------------------------------

def test_caller_grid_reconstructs_centers():
    # reproject emits linspace(left+res/2, right-res/2, W); the bounds
    # _caller_grid returns must map those back onto the caller's coords.
    x = np.linspace(-100.0, -95.0, 6)
    y = np.linspace(45.0, 40.0, 6)
    da = xr.DataArray(np.zeros((6, 6)), dims=['y', 'x'],
                      coords={'y': y, 'x': x})
    (left, bottom, right, top), W, H = _caller_grid(da)
    res_x = (right - left) / W
    res_y = (top - bottom) / H
    ox = np.linspace(left + res_x / 2, right - res_x / 2, W)
    oy = np.linspace(top - res_y / 2, bottom + res_y / 2, H)
    assert np.allclose(ox, x)
    assert np.allclose(oy, y)
    assert (W, H) == (6, 6)


def test_caller_grid_single_cell_needs_resolution():
    da = xr.DataArray(np.zeros((1, 1)), dims=['y', 'x'],
                      coords={'y': [44.0], 'x': [-97.0]})
    with pytest.raises(ValueError, match="single-cell"):
        _caller_grid(da)
    # with a transform it can infer the cell size
    da.attrs['transform'] = (0.01, 0, -97.0, 0, -0.01, 44.0)
    (left, bottom, right, top), W, H = _caller_grid(da)
    assert (W, H) == (1, 1)
    assert np.isclose(right - left, 0.01)


# ---------------------------------------------------------------------------
# Helpers: build a file + a template on a different grid
# ---------------------------------------------------------------------------

def _file_4326(tmp_path, dtype, name, nodata=None):
    height, width = 30, 30
    arr = np.arange(height * width, dtype=dtype).reshape(height, width)
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    attrs = {'crs': 4326}
    if nodata is not None:
        attrs['nodata'] = nodata
        # place the sentinel in the centre so it falls inside the
        # template's windowed footprint (which covers the interior)
        arr[14:17, 14:17] = nodata
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x}, attrs=attrs)
    path = str(tmp_path / name)
    to_geotiff(da, path, compression='none')
    return path


def _template_3857(n=6):
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = tr.transform(-120.25, 45.25)
    x1, y1 = tr.transform(-119.75, 44.75)
    return xr.DataArray(
        np.zeros((n, n), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(max(y0, y1), min(y0, y1), n),
                'x': np.linspace(min(x0, x1), max(x0, x1), n)},
        attrs={'crs': 3857},
    )


# ---------------------------------------------------------------------------
# Exact grid match
# ---------------------------------------------------------------------------

def test_coregister_matches_grid_crs_mismatch(tmp_path):
    path = _file_4326(tmp_path, np.float32, 'cg_mismatch.tif')
    template = _template_3857(6)
    out = template.xrs.open_geotiff(path, coregister=True)
    assert out.shape == template.shape
    assert np.allclose(out.coords['x'].values, template.coords['x'].values)
    assert np.allclose(out.coords['y'].values, template.coords['y'].values)


def test_coregister_matches_grid_same_crs(tmp_path):
    # Same CRS but a coarser, offset template: coregister still snaps.
    path = _file_4326(tmp_path, np.float32, 'cg_samecrs.tif')
    template = xr.DataArray(
        np.zeros((5, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(45.3, 44.7, 5),
                'x': np.linspace(-120.3, -119.7, 5)},
        attrs={'crs': 4326},
    )
    out = template.xrs.open_geotiff(path, coregister=True)
    assert out.shape == template.shape
    assert np.allclose(out.coords['x'].values, template.coords['x'].values)
    assert np.allclose(out.coords['y'].values, template.coords['y'].values)


def test_coregister_crs_less_file(tmp_path):
    # A file with a transform but no CRS is treated as already in the
    # template's CRS: coregister resamples onto the template grid rather
    # than failing reproject's source-CRS detection.
    height, width = 20, 20
    arr = np.arange(height * width, dtype=np.float32).reshape(height, width)
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    # no 'crs' attr -> to_geotiff writes a transform-only (CRS-less) file
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x})
    path = str(tmp_path / 'cg_nocrs.tif')
    to_geotiff(da, path, compression='none')

    template = xr.DataArray(
        np.zeros((8, 8), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(45.3, 44.7, 8),
                'x': np.linspace(-120.3, -119.7, 8)},
        attrs={'crs': 4326},
    )
    out = template.xrs.open_geotiff(path, coregister=True)
    assert out.shape == template.shape
    assert np.allclose(out.coords['x'].values, template.coords['x'].values)
    assert np.allclose(out.coords['y'].values, template.coords['y'].values)


def test_coregister_dask_template(tmp_path):
    path = _file_4326(tmp_path, np.float32, 'cg_dask.tif')
    template = _template_3857(6).chunk({'y': 3, 'x': 3})
    out = template.xrs.open_geotiff(path, coregister=True)
    assert np.allclose(out.coords['x'].values, template.coords['x'].values)
    assert np.allclose(out.coords['y'].values, template.coords['y'].values)


# ---------------------------------------------------------------------------
# unpack (renamed from mask_and_scale)
# ---------------------------------------------------------------------------

def test_coregister_forwards_unpack(tmp_path, monkeypatch):
    # coregister must read with unpack=True. Spy on the real open_geotiff
    # and confirm the kwarg, then delegate.
    path = _file_4326(tmp_path, np.float32, 'cg_ms.tif')
    template = _template_3857(6)

    import xrspatial.geotiff as gt
    real = gt.open_geotiff
    seen = {}

    def spy(src, **kw):
        seen['unpack'] = kw.get('unpack')
        return real(src, **kw)

    monkeypatch.setattr(gt, 'open_geotiff', spy)
    template.xrs.open_geotiff(path, coregister=True)
    assert seen['unpack'] is True


def test_coregister_masks_nodata_to_nan(tmp_path):
    # Coregister onto a template that matches the file grid, so the
    # masked sentinel cells map one-to-one to output NaNs (no resample
    # blending to hide the hole).
    height, width = 20, 20
    arr = np.arange(height * width, dtype=np.int16).reshape(height, width)
    arr[8:12, 8:12] = -9999
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x},
                      attrs={'crs': 4326, 'nodata': -9999})
    path = str(tmp_path / 'cg_nodata.tif')
    to_geotiff(da, path, compression='none')

    template = xr.DataArray(
        np.zeros((height, width), dtype=np.float32),
        dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326},
    )
    out = template.xrs.open_geotiff(path, coregister=True, resampling='nearest')
    vals = np.asarray(out.data)
    assert np.issubdtype(vals.dtype, np.floating)
    assert np.isnan(vals).any()


# ---------------------------------------------------------------------------
# resampling reuse
# ---------------------------------------------------------------------------

def test_coregister_categorical_preserves_class_ids(tmp_path):
    # Integer class raster (no nodata, no scale) -> 'auto' uses nearest,
    # so coregistered output holds only original class IDs.
    classes = np.array([10, 20, 30, 40], dtype=np.int16)
    height, width = 30, 30
    block = np.tile(classes, (height, width // classes.size + 1))[:, :width]
    y = np.linspace(45.5, 44.5, height)
    x = np.linspace(-120.5, -119.5, width)
    da = xr.DataArray(block.astype(np.int16), dims=['y', 'x'],
                      coords={'y': y, 'x': x}, attrs={'crs': 4326})
    path = str(tmp_path / 'cg_classes.tif')
    to_geotiff(da, path, compression='none')

    out = _template_3857(6).xrs.open_geotiff(path, coregister=True)
    vals = np.asarray(out.data)
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert np.isin(finite, classes).all()


# ---------------------------------------------------------------------------
# gpu / vrt guard
# ---------------------------------------------------------------------------

def test_coregister_gpu_guard(tmp_path):
    path = _file_4326(tmp_path, np.float32, 'cg_gpu.tif')
    template = _template_3857(6)
    with pytest.raises(ValueError, match="not supported with gpu"):
        template.xrs.open_geotiff(path, coregister=True, gpu=True)


def test_coregister_vrt_guard(tmp_path):
    # The guard fires before any read, so the .vrt need not exist.
    template = _template_3857(6)
    with pytest.raises(ValueError, match="not supported with gpu"):
        template.xrs.open_geotiff('nonexistent.vrt', coregister=True)
