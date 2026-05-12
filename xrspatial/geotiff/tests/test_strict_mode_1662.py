"""Regression tests for #1662: typed warnings + ``XRSPATIAL_GEOTIFF_STRICT``.

The audit in issue #1662 flagged ten ``except Exception: return None``
sites in the geotiff module that silently swallowed errors. Each site
now emits a ``GeoTIFFFallbackWarning`` (or re-raises in strict mode).
This module pins the contract:

* Default mode: a fallback returns ``None`` (or skips) and a
  ``GeoTIFFFallbackWarning`` is emitted with the original exception
  type and message.
* ``XRSPATIAL_GEOTIFF_STRICT=1``: the same code paths re-raise the
  original exception. CI can flip the env var to fail loudly on any
  silent fallback.

The GPU helper sites in ``_gpu_decode.py`` are exercised indirectly
through ``read_geotiff_gpu`` when CuPy is available; tests gated on
``importlib.util.find_spec('cupy')`` are skipped otherwise.
"""
from __future__ import annotations

import importlib.util
import os
import warnings

import pytest

from xrspatial.geotiff import (
    GeoTIFFFallbackWarning,
    _geotiff_strict_mode,
    _wkt_to_epsg,
)
from xrspatial.geotiff._geotags import _epsg_to_wkt


@pytest.fixture
def clear_strict_env(monkeypatch):
    """Ensure the strict-mode env var is unset for the default-mode test."""
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)


@pytest.fixture
def set_strict_env(monkeypatch):
    """Set ``XRSPATIAL_GEOTIFF_STRICT=1`` for strict-mode tests."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')


# ---------------------------------------------------------------------------
# _geotiff_strict_mode() helper
# ---------------------------------------------------------------------------

def test_strict_mode_default_false(clear_strict_env):
    assert _geotiff_strict_mode() is False


@pytest.mark.parametrize('value', ['1', 'true', 'True', 'yes', 'YES'])
def test_strict_mode_truthy_values(monkeypatch, value):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', value)
    assert _geotiff_strict_mode() is True


@pytest.mark.parametrize('value', ['0', 'false', 'no', '', 'maybe'])
def test_strict_mode_falsy_values(monkeypatch, value):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', value)
    assert _geotiff_strict_mode() is False


# ---------------------------------------------------------------------------
# _wkt_to_epsg
# ---------------------------------------------------------------------------

def test_wkt_to_epsg_default_warns_returns_none(clear_strict_env):
    """In default mode a broken WKT input warns and returns None."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = _wkt_to_epsg('not-a-valid-wkt-string-1662')

    assert result is None
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    assert '_wkt_to_epsg failed' in str(fallback_warnings[0].message)


def test_wkt_to_epsg_strict_reraises(set_strict_env):
    """Under XRSPATIAL_GEOTIFF_STRICT=1, _wkt_to_epsg re-raises."""
    with pytest.raises(Exception):
        _wkt_to_epsg('not-a-valid-wkt-string-1662')


def test_wkt_to_epsg_valid_input_no_warning(clear_strict_env):
    """A real WKT string returns its EPSG without any warning."""
    pyproj = pytest.importorskip('pyproj')
    wkt = pyproj.CRS.from_epsg(4326).to_wkt()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = _wkt_to_epsg(wkt)

    assert result == 4326
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert fallback_warnings == []


# ---------------------------------------------------------------------------
# _epsg_to_wkt
# ---------------------------------------------------------------------------

def test_epsg_to_wkt_default_warns_returns_none(clear_strict_env):
    """An unknown EPSG warns and returns None in default mode."""
    pytest.importorskip('pyproj')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # EPSG 999999 is unassigned; pyproj raises CRSError.
        result = _epsg_to_wkt(999999)

    assert result is None
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    assert '_epsg_to_wkt' in str(fallback_warnings[0].message)


def test_epsg_to_wkt_strict_reraises(set_strict_env):
    """Strict mode re-raises rather than warning."""
    pytest.importorskip('pyproj')
    with pytest.raises(Exception):
        _epsg_to_wkt(999999)


# ---------------------------------------------------------------------------
# VRT source skip
# ---------------------------------------------------------------------------

def test_vrt_missing_source_default_warns_then_continues(
    clear_strict_env, tmp_path,
):
    """A VRT referencing a missing source file warns once and skips it."""
    from xrspatial.geotiff import read_vrt

    vrt_path = tmp_path / 'mosaic_1662_missing.vrt'
    vrt_path.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS></SRS>\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <NoDataValue>-9999</NoDataValue>\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{tmp_path}/does_not_exist_1662.tif</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path))

    # The mosaic should still load (with a hole) and one warning should
    # describe the skipped source.
    assert da.shape == (4, 4)
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) >= 1
    msgs = ' '.join(str(x.message) for x in fallback_warnings)
    assert 'VRT source' in msgs
    assert 'does_not_exist_1662' in msgs


def test_vrt_missing_source_strict_raises(set_strict_env, tmp_path):
    """In strict mode the missing source surfaces as an exception."""
    from xrspatial.geotiff import read_vrt

    vrt_path = tmp_path / 'mosaic_1662_missing_strict.vrt'
    vrt_path.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS></SRS>\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <NoDataValue>-9999</NoDataValue>\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{tmp_path}/does_not_exist_1662_strict.tif</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )

    with pytest.raises(Exception):
        read_vrt(str(vrt_path))


# ---------------------------------------------------------------------------
# _warn_or_raise_gpu_fallback pins the warn-vs-raise contract for every
# GPU helper site flagged in #1662. Exercised directly so the test does
# not depend on having a working CUDA/GDS/nvCOMP stack.
# ---------------------------------------------------------------------------

CUPY_AVAILABLE = importlib.util.find_spec('cupy') is not None


def test_warn_or_raise_gpu_fallback_default_warns(clear_strict_env):
    """Default mode emits one GeoTIFFFallbackWarning carrying type + msg."""
    from xrspatial.geotiff._gpu_decode import _warn_or_raise_gpu_fallback

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _warn_or_raise_gpu_fallback(
            "_try_nvjpeg_batch_decode", RuntimeError("bogus 1662"))

    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    msg = str(fallback_warnings[0].message)
    assert '_try_nvjpeg_batch_decode' in msg
    assert 'RuntimeError' in msg
    assert 'bogus 1662' in msg


def test_warn_or_raise_gpu_fallback_strict_reraises(set_strict_env):
    """Strict mode re-raises the original exception."""
    from xrspatial.geotiff._gpu_decode import _warn_or_raise_gpu_fallback

    with pytest.raises(RuntimeError, match='bogus 1662 strict'):
        _warn_or_raise_gpu_fallback(
            "_try_nvjpeg_batch_decode", RuntimeError("bogus 1662 strict"))


# ---------------------------------------------------------------------------
# read_geotiff_gpu on_gpu_failure='auto' + env var integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="cupy required for read_geotiff_gpu fallback test",
)
def test_read_geotiff_gpu_env_var_promotes_to_strict(set_strict_env, tmp_path):
    """With on_gpu_failure='auto' but XRSPATIAL_GEOTIFF_STRICT=1, a GPU
    decode failure surfaces instead of falling back to CPU."""
    from xrspatial.geotiff import read_geotiff_gpu

    # A non-existent path triggers a failure before any decode runs;
    # the env var should still bubble it up.
    bogus = str(tmp_path / 'no_such_file_1662_promote.tif')
    with pytest.raises(Exception):
        read_geotiff_gpu(bogus)
