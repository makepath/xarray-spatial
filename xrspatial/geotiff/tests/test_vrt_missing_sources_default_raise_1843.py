"""Regression test for #1843: the internal ``read_vrt`` in ``_vrt.py``
defaults to ``missing_sources='raise'`` so an unreadable source halts
the call instead of leaving a silent zero-fill hole on integer rasters.

Callers wanting the historical lenient behaviour pass
``missing_sources='warn'`` explicitly. The strict-mode env var
``XRSPATIAL_GEOTIFF_STRICT=1`` continues to force-raise across the
whole module (orthogonal axis, not affected by this change).
"""
from __future__ import annotations

import pytest

from xrspatial.geotiff import GeoTIFFFallbackWarning
from xrspatial.geotiff._vrt import read_vrt as _read_vrt_internal


def _write_missing_source_vrt(path):
    path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">missing_1843.tif'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_read_vrt_default_raises_on_unreadable_source(tmp_path):
    """Without an explicit ``missing_sources`` kwarg, an unreadable
    backing source must raise rather than silently zero-fill.

    This is the behaviour change from #1843. Before this commit the
    default was ``'warn'`` and a missing ``Byte`` tile produced a hole
    of zero pixels that was indistinguishable from real data unless
    the caller checked ``attrs['vrt_holes']``.
    """
    vrt = tmp_path / "tmp_1843_default_raise.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.raises((OSError, ValueError)):
        _read_vrt_internal(str(vrt))


def test_read_vrt_explicit_warn_preserves_lenient_behaviour(tmp_path):
    """``missing_sources='warn'`` is still the escape hatch for callers
    that want partial mosaics with ``vrt.holes`` populated.

    Pinning the lenient path here keeps the historical contract
    available to callers who opt in. The warning and the hole record
    must both still surface.
    """
    vrt = tmp_path / "tmp_1843_explicit_warn.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
        arr, parsed = _read_vrt_internal(str(vrt), missing_sources='warn')

    assert arr.shape == (2, 2)
    assert len(parsed.holes) == 1
    assert parsed.holes[0]['source'].endswith('missing_1843.tif')


def test_read_vrt_strict_env_still_raises_under_warn(monkeypatch, tmp_path):
    """``XRSPATIAL_GEOTIFF_STRICT=1`` continues to force-raise even
    when the caller explicitly asks for the lenient ``'warn'`` policy.

    The strict env var is a module-wide override (see #1662); it must
    still win over per-call ``missing_sources='warn'`` so CI runs with
    strict mode catch partial mosaics regardless of caller settings.
    """
    vrt = tmp_path / "tmp_1843_strict_env.vrt"
    _write_missing_source_vrt(vrt)

    monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "1")

    with pytest.raises((OSError, ValueError)):
        _read_vrt_internal(str(vrt), missing_sources='warn')
