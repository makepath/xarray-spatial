"""Regression test for #1860: the public ``read_vrt`` and
``open_geotiff(.vrt)`` default ``missing_sources`` to ``'raise'``, matching
the internal ``_vrt.read_vrt`` default set in #1843.

Before #1860 the public wrapper defaulted to ``'warn'``, which silently
overrode the internal ``'raise'`` default and let unreadable backing
sources produce zero-fill holes on integer rasters with no exception.
Callers that want the lenient partial-mosaic behaviour pass
``missing_sources='warn'`` explicitly.
"""
from __future__ import annotations

import pytest

from xrspatial.geotiff import GeoTIFFFallbackWarning, open_geotiff, read_vrt


def _write_missing_source_vrt(path):
    path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">missing_1860.tif'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_public_read_vrt_default_raises_on_unreadable_source(tmp_path):
    """Public ``read_vrt`` with no ``missing_sources`` kwarg must raise.

    Before #1860 the default was ``'warn'`` and the call returned a
    partial mosaic with ``attrs['vrt_holes']`` instead of raising. With
    the default aligned to the internal ``_vrt.read_vrt`` default of
    ``'raise'``, the unreadable source must now halt the call.
    """
    vrt = tmp_path / "tmp_1860_public_default_raise.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.raises((OSError, ValueError)):
        read_vrt(str(vrt))


def test_open_geotiff_vrt_default_raises_on_unreadable_source(tmp_path):
    """``open_geotiff(vrt_path)`` with no ``missing_sources`` kwarg must
    raise on an unreadable backing source.

    ``open_geotiff`` forwards ``missing_sources`` to ``read_vrt`` only
    when the caller passed it explicitly; otherwise the public
    ``read_vrt`` default applies. With that default now ``'raise'``, the
    silent-degradation path is closed for ``open_geotiff`` callers too.
    """
    vrt = tmp_path / "tmp_1860_open_geotiff_default_raise.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.raises((OSError, ValueError)):
        open_geotiff(str(vrt))


def test_public_read_vrt_explicit_warn_preserves_lenient_behaviour(tmp_path):
    """``missing_sources='warn'`` is still the escape hatch for partial
    mosaics on the public ``read_vrt`` API.

    The warning fires, the call returns, and ``attrs['vrt_holes']`` is
    populated with the skipped source record. Pinning this keeps the
    historical contract available to callers that opt in.
    """
    vrt = tmp_path / "tmp_1860_public_explicit_warn.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
        da = read_vrt(str(vrt), missing_sources='warn')

    assert 'vrt_holes' in da.attrs
    assert da.attrs['vrt_holes'][0]['source'].endswith('missing_1860.tif')


def test_open_geotiff_vrt_explicit_warn_preserves_lenient_behaviour(tmp_path):
    """``open_geotiff(vrt_path, missing_sources='warn')`` still produces
    a partial mosaic with the hole record on the DataArray attrs.

    The forwarding branch in ``open_geotiff`` only runs when the caller
    explicitly passes ``missing_sources``; this test pins that branch
    against regressions.
    """
    vrt = tmp_path / "tmp_1860_open_geotiff_explicit_warn.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
        da = open_geotiff(str(vrt), missing_sources='warn')

    assert 'vrt_holes' in da.attrs
    assert da.attrs['vrt_holes'][0]['source'].endswith('missing_1860.tif')
