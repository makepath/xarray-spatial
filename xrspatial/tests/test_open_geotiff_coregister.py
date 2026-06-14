"""Tests for ``.xrs.open_geotiff(coregister=True)`` (issue #3069 -
unpack + reproject + resample onto the caller's exact grid)."""
from __future__ import annotations

import warnings

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
# output chunking follows the caller (#3234)
# ---------------------------------------------------------------------------

def test_coregister_dask_template_keeps_caller_chunks(tmp_path):
    # reproject defaults its dask output chunks to 512 (one whole-array
    # chunk for a small template); the coregistered result must instead
    # carry the caller's chunk layout. Asymmetric chunks prove the
    # (y, x) ordering.
    path = _file_4326(tmp_path, np.float32, 'cg_chunks_3234.tif')
    template = _template_3857(6).chunk({'y': 3, 'x': 2})
    out = template.xrs.open_geotiff(path, coregister=True)
    assert out.data.chunksize == (3, 2)


def test_coregister_explicit_chunks_kwarg(tmp_path):
    # numpy template + explicit chunks=: the windowed read is dask, and
    # the coregistered output keeps the requested chunking instead of
    # reverting to reproject's default.
    path = _file_4326(tmp_path, np.float32, 'cg_chunks_kw_3234.tif')
    template = _template_3857(6)
    out = template.xrs.open_geotiff(path, coregister=True, chunks=2)
    assert out.data.chunksize == (2, 2)


def test_coregister_chunks_kwarg_tuple_and_np_integer(tmp_path):
    # the (row, col) tuple form is what _compute_chunk_layout unpacks
    # directly, and np.integer scalars are valid for the read so they
    # must be coerced before reaching reproject.
    path = _file_4326(tmp_path, np.float32, 'cg_chunks_forms_3234.tif')
    template = _template_3857(6)
    out = template.xrs.open_geotiff(path, coregister=True, chunks=(3, 2))
    assert out.data.chunksize == (3, 2)
    out = template.xrs.open_geotiff(path, coregister=True,
                                    chunks=np.int64(2))
    assert out.data.chunksize == (2, 2)


def test_auto_reproject_dask_template_keeps_caller_chunks(tmp_path):
    # the auto_reproject branch reprojects onto an auto-computed grid,
    # but the output chunk layout should still follow the caller.
    path = _file_4326(tmp_path, np.float32, 'ar_chunks_3234.tif')
    template = _template_3857(6).chunk({'y': 3, 'x': 3})
    out = template.xrs.open_geotiff(path, auto_reproject=True)
    assert out.data.chunksize == (3, 3)


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

    # integer dtype + no colormap also triggers the is-this-really-
    # categorical advisory; the class IDs are the point of this test.
    with pytest.warns(UserWarning, match="nearest"):
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


# ---------------------------------------------------------------------------
# sparse-coverage warning (issue #3246)
# ---------------------------------------------------------------------------

def test_coregister_warns_when_file_covers_small_fraction(tmp_path):
    # 1x1 degree file against a 10x10 degree template: ~1% coverage.
    path = _file_4326(tmp_path, np.float32, 'cg3246_small.tif')
    template = xr.DataArray(
        np.zeros((100, 100), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 40.0, 100),
                'x': np.linspace(-125.0, -115.0, 100)},
        attrs={'crs': 4326},
    )
    with pytest.warns(UserWarning, match="covers only"):
        out = template.xrs.open_geotiff(path, coregister=True)
    # The warning changes nothing about the result itself.
    assert out.shape == template.shape
    assert np.allclose(out.coords['x'].values, template.coords['x'].values)
    assert np.allclose(out.coords['y'].values, template.coords['y'].values)


def test_coregister_warns_dask_template(tmp_path):
    # The motivating case for #3246: a large dask-backed template. The
    # check runs on metadata before the read, so the warning fires
    # without computing, and the result stays lazy with caller chunks.
    path = _file_4326(tmp_path, np.float32, 'cg3246_dask.tif')
    template = xr.DataArray(
        np.zeros((100, 100), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 40.0, 100),
                'x': np.linspace(-125.0, -115.0, 100)},
        attrs={'crs': 4326},
    ).chunk({'y': 50, 'x': 50})
    with pytest.warns(UserWarning, match="covers only"):
        out = template.xrs.open_geotiff(path, coregister=True)
    assert out.data.chunksize == (50, 50)


def test_coregister_full_coverage_no_warning(tmp_path):
    # Template sits entirely inside the file footprint: no warning.
    path = _file_4326(tmp_path, np.float32, 'cg3246_full.tif')
    template = _template_3857(6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        template.xrs.open_geotiff(path, coregister=True)
    assert not [w for w in caught if "covers only" in str(w.message)]


def test_coregister_warns_small_fraction_crs_mismatch(tmp_path):
    # Coverage is computed after the caller bbox is transformed into the
    # file CRS, so the warning also fires across a CRS mismatch.
    from pyproj import Transformer
    path = _file_4326(tmp_path, np.float32, 'cg3246_mismatch.tif')
    tr = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = tr.transform(-125.0, 50.0)
    x1, y1 = tr.transform(-115.0, 40.0)
    template = xr.DataArray(
        np.zeros((100, 100), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(max(y0, y1), min(y0, y1), 100),
                'x': np.linspace(min(x0, x1), max(x0, x1), 100)},
        attrs={'crs': 3857},
    )
    with pytest.warns(UserWarning, match="covers only"):
        template.xrs.open_geotiff(path, coregister=True)


# ---------------------------------------------------------------------------
# overview-aware windowing + auto overview selection
# ---------------------------------------------------------------------------

def _cog_4326(tmp_path, name):
    # 256x256 smooth gradient COG over ~44..46N / -121..-119E with
    # internal overviews (tile_size=64 -> levels at 128 and 64).
    height = width = 256
    arr = np.add.outer(np.arange(height, dtype=np.float32),
                       np.arange(width, dtype=np.float32))
    y = np.linspace(45.999, 44.001, height)
    x = np.linspace(-120.999, -119.001, width)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x}, attrs={'crs': 4326})
    path = str(tmp_path / name)
    to_geotiff(da, path, cog=True, tile_size=64)
    return path


def _interior_template_4326(n=8):
    return xr.DataArray(
        np.zeros((n, n), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(45.4, 44.6, n),
                'x': np.linspace(-120.4, -119.6, n)},
        attrs={'crs': 4326},
    )


def test_coregister_explicit_overview_level(tmp_path):
    # An explicit overview_level= must window in that overview's pixel
    # space; computing the window against the full-resolution transform
    # reads the wrong region of the overview (or falls outside it).
    path = _cog_4326(tmp_path, 'cg_ov_explicit.tif')
    template = _interior_template_4326(8)
    full = template.xrs.open_geotiff(path, coregister=True,
                                     overview_level=0)
    out = template.xrs.open_geotiff(path, coregister=True,
                                    overview_level=1)
    assert np.allclose(out.coords['x'].values, template.coords['x'].values)
    assert np.allclose(out.coords['y'].values, template.coords['y'].values)
    vals = np.asarray(out.data)
    assert not np.isnan(vals).any()
    # The template sits inside the file, the source is a smooth gradient:
    # the overview read must land within resampling tolerance of the
    # full-resolution read.
    assert np.allclose(vals, np.asarray(full.data), atol=2.0)


def _spy_read_overview_level(monkeypatch):
    # Capture the overview_level the windowed read ends up using.
    import xrspatial.geotiff as gt
    real = gt.open_geotiff
    seen = {}

    def spy(src, **kw):
        seen['overview_level'] = kw.get('overview_level')
        return real(src, **kw)

    monkeypatch.setattr(gt, 'open_geotiff', spy)
    return seen


def test_coregister_auto_selects_overview(tmp_path, monkeypatch):
    # A caller grid much coarser than the file: coregister should read
    # the coarsest overview that is still finer than the caller grid
    # (256px file, 8px template -> 32x decimation -> level 2 = 64px)
    # instead of decoding the full-resolution pixels.
    path = _cog_4326(tmp_path, 'cg_ov_auto.tif')
    template = _interior_template_4326(8)
    seen = _spy_read_overview_level(monkeypatch)
    full = template.xrs.open_geotiff(path, coregister=True,
                                     overview_level=0)
    out = template.xrs.open_geotiff(path, coregister=True)
    assert seen['overview_level'] == 2
    vals = np.asarray(out.data)
    assert not np.isnan(vals).any()
    assert np.allclose(vals, np.asarray(full.data), atol=8.0)


def test_coregister_auto_overview_crs_mismatch(tmp_path, monkeypatch):
    # The caller resolution that drives level selection is estimated in
    # file-CRS units from the transformed bbox; a 3857 template over a
    # 4326 file exercises that conversion (the same-CRS tests run it
    # with an identity transform only).
    from pyproj import Transformer
    path = _cog_4326(tmp_path, 'cg_ov_auto_3857.tif')
    tr = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = tr.transform(-120.4, 45.4)
    x1, y1 = tr.transform(-119.6, 44.6)
    template = xr.DataArray(
        np.zeros((8, 8), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(max(y0, y1), min(y0, y1), 8),
                'x': np.linspace(min(x0, x1), max(x0, x1), 8)},
        attrs={'crs': 3857},
    )
    seen = _spy_read_overview_level(monkeypatch)
    out = template.xrs.open_geotiff(path, coregister=True)
    # ~0.1 deg per template cell vs 0.0078 deg file pixels: level 2
    # (0.031 deg) is the coarsest that still meets the template.
    assert seen['overview_level'] == 2
    vals = np.asarray(out.data)
    assert not np.isnan(vals).any()


def test_coregister_explicit_overview_level_wins_over_auto(tmp_path,
                                                           monkeypatch):
    path = _cog_4326(tmp_path, 'cg_ov_explicit_wins.tif')
    template = _interior_template_4326(8)
    seen = _spy_read_overview_level(monkeypatch)
    template.xrs.open_geotiff(path, coregister=True, overview_level=1)
    assert seen['overview_level'] == 1
    # overview_level=0 is an explicit full-resolution opt-out.
    template.xrs.open_geotiff(path, coregister=True, overview_level=0)
    assert seen['overview_level'] == 0


def test_coregister_no_overviews_falls_back_to_full_res(tmp_path,
                                                        monkeypatch):
    # A plain (non-COG) TIFF has no overviews: a coarse caller grid must
    # fall back to the full-resolution read, not raise.
    path = _file_4326(tmp_path, np.float32, 'cg_ov_plain.tif')
    template = _template_3857(3)  # 30px file onto 3px template: 10x
    seen = _spy_read_overview_level(monkeypatch)
    out = template.xrs.open_geotiff(path, coregister=True)
    assert seen['overview_level'] is None
    assert out.shape == template.shape


def test_coregister_no_overview_when_res_similar(tmp_path, monkeypatch):
    # Template resolution close to the file's: stay at full resolution.
    path = _cog_4326(tmp_path, 'cg_ov_similar.tif')
    template = _interior_template_4326(200)
    seen = _spy_read_overview_level(monkeypatch)
    template.xrs.open_geotiff(path, coregister=True)
    assert seen['overview_level'] is None


# ---------------------------------------------------------------------------
# fail-fast guards
# ---------------------------------------------------------------------------

def test_coregister_zero_overlap_raises_clear_error(tmp_path):
    # A file fully disjoint from the caller grid would come back all
    # NaN; raise up front with a message that names the problem instead
    # of the low-level "window is outside the source extent" error.
    path = _file_4326(tmp_path, np.float32, 'cg_disjoint.tif')
    template = xr.DataArray(
        np.zeros((6, 6), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(10.0, 9.0, 6),
                'x': np.linspace(30.0, 31.0, 6)},
        attrs={'crs': 4326},
    )
    with pytest.raises(ValueError, match="does not overlap"):
        template.xrs.open_geotiff(path, coregister=True)


def test_coregister_allow_rotated_guard(tmp_path):
    # allow_rotated=True reads a rotated file as an ungeoreferenced
    # pixel grid; reproject would then treat its y/x coords as
    # axis-aligned map coordinates and produce silently wrong
    # georeferencing. Refuse the combination up front (so the guard
    # fires before any read).
    template = _template_3857(6)
    with pytest.raises(ValueError, match="allow_rotated"):
        template.xrs.open_geotiff('unused.tif', coregister=True,
                                  allow_rotated=True)


def test_coregister_irregular_template_raises(tmp_path):
    # _caller_grid derives the output grid from first/last coords and a
    # uniform spacing; an irregular template would silently misalign.
    path = _file_4326(tmp_path, np.float32, 'cg_irregular.tif')
    x = np.linspace(-120.3, -119.7, 8)
    x[3] += 0.05  # well past any float tolerance
    template = xr.DataArray(
        np.zeros((8, 8), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(45.3, 44.7, 8), 'x': x},
        attrs={'crs': 4326},
    )
    with pytest.raises(ValueError, match="regular"):
        template.xrs.open_geotiff(path, coregister=True)


def test_coregister_template_orientation_raises(tmp_path):
    # reproject emits x-ascending / y-descending output; a template in
    # any other orientation cannot line up cell-for-cell, so refuse it
    # rather than returning coords in a different order than the caller.
    path = _file_4326(tmp_path, np.float32, 'cg_orient.tif')
    base = _interior_template_4326(8)
    flipped_x = base.assign_coords(x=base.coords['x'].values[::-1])
    with pytest.raises(ValueError, match="ascending"):
        flipped_x.xrs.open_geotiff(path, coregister=True)
    flipped_y = base.assign_coords(y=base.coords['y'].values[::-1])
    with pytest.raises(ValueError, match="descending"):
        flipped_y.xrs.open_geotiff(path, coregister=True)


def test_coregister_antimeridian_template_warns(tmp_path, monkeypatch):
    # A caller grid straddling the dateline transforms into file-CRS
    # longitudes near +180 and -180; min/max windowing then covers the
    # full file width. That stays correct (a superset read) but is easy
    # to hit by accident and reads far more than the caller footprint,
    # so warn -- and skip auto overview selection, because the inflated
    # bbox span would otherwise pick an overview far coarser than the
    # caller's real resolution.
    from pyproj import Transformer
    height = width = 256
    arr = np.add.outer(np.arange(height, dtype=np.float32),
                       np.arange(width, dtype=np.float32))
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(59.99, 50.01, height),
                'x': np.linspace(-179.99, 179.99, width)},
        attrs={'crs': 4326},
    )
    path = str(tmp_path / 'cg_antimeridian.tif')
    to_geotiff(da, path, cog=True, tile_size=64)

    # UTM zone 60N caller crossing 180E (lon 179..181).
    tr = Transformer.from_crs(4326, 32660, always_xy=True)
    x0, y0 = tr.transform(179.0, 54.0)
    x1, y1 = tr.transform(181.0, 56.0)
    template = xr.DataArray(
        np.zeros((8, 8), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.linspace(max(y0, y1), min(y0, y1), 8),
                'x': np.linspace(min(x0, x1), max(x0, x1), 8)},
        attrs={'crs': 32660},
    )
    seen = _spy_read_overview_level(monkeypatch)
    with pytest.warns(UserWarning, match="antimeridian"):
        out = template.xrs.open_geotiff(path, coregister=True)
    assert seen['overview_level'] is None
    assert out.shape == template.shape
    assert np.isfinite(np.asarray(out.data)).any()


def test_coregister_auto_nearest_integer_warns(tmp_path):
    # 'auto' picks 'nearest' for integer rasters on the assumption they
    # are categorical. For an integer DEM that silently throws away
    # interpolation quality, so say what was chosen and how to override.
    path = _file_4326(tmp_path, np.int16, 'cg_intwarn.tif')
    template = _template_3857(6)
    with pytest.warns(UserWarning, match="nearest"):
        template.xrs.open_geotiff(path, coregister=True)
    # An explicit choice is not second-guessed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        template.xrs.open_geotiff(path, coregister=True,
                                  resampling='nearest')
    assert not [w for w in caught if "resampling" in str(w.message)]
