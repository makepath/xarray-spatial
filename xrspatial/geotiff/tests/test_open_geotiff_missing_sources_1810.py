"""Regression test for #1810: ``open_geotiff`` did not accept
``missing_sources`` and did not forward it to ``read_vrt`` when the
source was a VRT.

The api-consistency sweep on 2026-05-13 flagged that ``read_vrt`` had
gained a ``missing_sources='warn'|'raise'`` policy kwarg (#1806) but
the documented dispatcher entry point ``open_geotiff`` did not expose
it. Callers wanting strict failure had to call ``read_vrt`` directly.
Same class of dispatcher-drops-backend-kwarg bug as #1561, #1605,
#1685, #1795.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning, open_geotiff, to_geotiff, write_vrt


def _write_missing_source_vrt(path):
    """Write a VRT pointing at a non-existent source file.

    Mirrors the helper in ``test_vrt_missing_sources_policy_1799.py`` so
    the warn-vs-raise branches surface the same way through the
    dispatcher as they do via the direct ``read_vrt`` call.
    """
    path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">missing.tif'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_open_geotiff_accepts_missing_sources():
    """The signature exposes ``missing_sources`` so IDE autocomplete and
    ``inspect.signature`` see the kwarg without parsing the docstring.
    """
    sig = inspect.signature(open_geotiff)
    assert 'missing_sources' in sig.parameters


def test_open_geotiff_vrt_missing_sources_warn(tmp_path):
    """``missing_sources='warn'`` matches the historic behavior:
    ``GeoTIFFFallbackWarning`` is emitted and ``attrs['vrt_holes']`` is
    populated. Going through the dispatcher must produce the same
    result as calling ``read_vrt`` directly.
    """
    vrt = tmp_path / "tmp_1810_missing_warn.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
        da = open_geotiff(str(vrt), missing_sources='warn')

    assert 'vrt_holes' in da.attrs
    assert da.attrs['vrt_holes'][0]['source'].endswith('missing.tif')


def test_open_geotiff_vrt_missing_sources_raise(tmp_path):
    """``missing_sources='raise'`` propagates through the dispatcher and
    fails fast instead of silently returning a partial mosaic.
    """
    vrt = tmp_path / "tmp_1810_missing_raise.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.raises((OSError, ValueError)):
        open_geotiff(str(vrt), missing_sources='raise')


def test_open_geotiff_vrt_missing_sources_validates_policy(tmp_path):
    """The validator on ``read_vrt`` fires through the dispatcher path
    so callers get the same parameter-named error from either entry
    point.
    """
    vrt = tmp_path / "tmp_1810_missing_bad_policy.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.raises(ValueError, match="missing_sources"):
        open_geotiff(str(vrt), missing_sources='ignore')


def test_open_geotiff_rejects_missing_sources_on_tif(tmp_path):
    """``missing_sources`` is VRT-only; passing it with a ``.tif`` source
    raises rather than silently doing nothing. Mirrors the
    ``on_gpu_failure`` rejection pattern.
    """
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(4, dtype=np.float64),
                'x': np.arange(4, dtype=np.float64)},
        attrs={'crs': 4326},
    )
    tif_path = tmp_path / "single_tile.tif"
    to_geotiff(da, str(tif_path))

    with pytest.raises(ValueError, match="missing_sources only applies"):
        open_geotiff(str(tif_path), missing_sources='raise')


def test_open_geotiff_vrt_without_missing_sources_kwarg_still_works(tmp_path):
    """Omitting ``missing_sources`` is a no-op: behaviour matches the
    pre-fix dispatcher (and the ``read_vrt`` default of ``'warn'``).
    The existing two-tile VRT exercises the happy path so callers
    upgrading from #1806 do not see a regression.
    """
    arr_a = np.arange(16, dtype=np.uint16).reshape(4, 4)
    da_a = xr.DataArray(
        arr_a, dims=['y', 'x'],
        coords={'y': np.array([0.5, 1.5, 2.5, 3.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        attrs={'crs': 4326},
    )
    tile_a = tmp_path / "tile_a.tif"
    to_geotiff(da_a, str(tile_a))

    arr_b = np.arange(16, 32, dtype=np.uint16).reshape(4, 4)
    da_b = xr.DataArray(
        arr_b, dims=['y', 'x'],
        coords={'y': np.array([0.5, 1.5, 2.5, 3.5]),
                'x': np.array([4.5, 5.5, 6.5, 7.5])},
        attrs={'crs': 4326},
    )
    tile_b = tmp_path / "tile_b.tif"
    to_geotiff(da_b, str(tile_b))

    vrt_path = tmp_path / "mosaic_1810.vrt"
    write_vrt(str(vrt_path), [str(tile_a), str(tile_b)])

    da = open_geotiff(str(vrt_path))
    assert da.shape == (4, 8)
