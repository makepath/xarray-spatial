"""Regression test for issue #1734.

Under the lenient default (``XRSPATIAL_GEOTIFF_STRICT`` unset),
``read_vrt`` warns once per unreadable source and continues, producing
a mosaic with zero-filled holes for integer VRTs that downstream code
cannot distinguish from real data. The warning is easy to miss in a
pipeline that ignores ``UserWarning``s.

This module pins the new behaviour: the returned DataArray now carries
an ``attrs['vrt_holes']`` list describing each skipped source so
callers can detect a partial mosaic with a single attribute lookup.
Strict mode is unchanged and still raises.
"""
from __future__ import annotations

import warnings

import pytest

from xrspatial.geotiff import GeoTIFFFallbackWarning, read_vrt


@pytest.fixture
def clear_strict_env(monkeypatch):
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)


@pytest.fixture
def set_strict_env(monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')


def _write_vrt_with_missing_source(vrt_path, missing_src) -> None:
    vrt_path.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS></SRS>\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <NoDataValue>-9999</NoDataValue>\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{missing_src}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_skipped_source_records_vrt_holes_attr(
    clear_strict_env, tmp_path,
):
    """A VRT with a missing source returns a DataArray whose attrs
    carry a ``vrt_holes`` entry naming the source, band, dst_rect,
    and underlying error."""
    vrt_path = tmp_path / 'mosaic_1734_missing.vrt'
    missing_src = f'{tmp_path}/does_not_exist_1734.tif'
    _write_vrt_with_missing_source(vrt_path, missing_src)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', GeoTIFFFallbackWarning)
        da = read_vrt(str(vrt_path))

    assert 'vrt_holes' in da.attrs
    holes = da.attrs['vrt_holes']
    assert isinstance(holes, list)
    assert len(holes) == 1
    h = holes[0]
    assert h['source'].endswith('does_not_exist_1734.tif')
    assert h['band'] == 1
    assert h['dst_rect'] == (0, 0, 4, 4)
    assert 'error' in h
    assert h['error']  # non-empty


def test_no_holes_attr_when_all_sources_read(clear_strict_env, tmp_path):
    """A successful VRT read does not advertise an empty ``vrt_holes``
    attr; the key is omitted entirely so ``"vrt_holes" in attrs`` is a
    cheap completeness check."""
    import numpy as np
    import xarray as xr
    from xrspatial.geotiff import to_geotiff

    # Write a real source the VRT can reference.
    src_path = tmp_path / 'src_1734.tif'
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    da_src = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(3.5, 0.5, 4),
                'x': np.linspace(0.5, 3.5, 4)},
        attrs={'crs': 4326},
    )
    to_geotiff(da_src, str(src_path), compression='none')

    vrt_path = tmp_path / 'mosaic_1734_ok.vrt'
    vrt_path.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS></SRS>\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error', GeoTIFFFallbackWarning)
        da = read_vrt(str(vrt_path))

    assert 'vrt_holes' not in da.attrs


def test_strict_mode_still_raises(set_strict_env, tmp_path):
    """Strict mode is unchanged: the missing source raises and no
    DataArray is returned."""
    vrt_path = tmp_path / 'mosaic_1734_strict.vrt'
    missing_src = f'{tmp_path}/does_not_exist_1734_strict.tif'
    _write_vrt_with_missing_source(vrt_path, missing_src)

    with pytest.raises(Exception):
        read_vrt(str(vrt_path))


def test_warning_mentions_how_to_detect_holes(clear_strict_env, tmp_path):
    """The fallback warning now points callers at the attr or the
    strict env var so the recovery path is discoverable from a single
    captured warning."""
    vrt_path = tmp_path / 'mosaic_1734_msg.vrt'
    missing_src = f'{tmp_path}/does_not_exist_1734_msg.tif'
    _write_vrt_with_missing_source(vrt_path, missing_src)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        read_vrt(str(vrt_path))

    fallback = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert fallback, "expected at least one GeoTIFFFallbackWarning"
    msg = ' '.join(str(x.message) for x in fallback)
    assert 'vrt_holes' in msg or 'XRSPATIAL_GEOTIFF_STRICT' in msg
