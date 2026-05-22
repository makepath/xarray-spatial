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
through ``read_geotiff_gpu`` when CuPy + CUDA are available; tests
gated on ``_gpu_available()`` are skipped otherwise.
"""
from __future__ import annotations

import importlib.util
import warnings

import pytest

from xrspatial.geotiff import GeoTIFFFallbackWarning, _geotiff_strict_mode, _wkt_to_epsg
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
    missing_src = f'{tmp_path}/does_not_exist_1662.tif'
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

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Public ``read_vrt`` defaults to ``missing_sources='raise'``
        # since #1860. Opt back into the lenient warn-then-continue
        # behaviour to keep exercising the warning path.
        da = read_vrt(str(vrt_path), missing_sources='warn')

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
    missing_src = f'{tmp_path}/does_not_exist_1662_strict.tif'
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

    with pytest.raises(Exception):
        read_vrt(str(vrt_path))


# ---------------------------------------------------------------------------
# _warn_or_raise_gpu_fallback pins the warn-vs-raise contract for every
# GPU helper site flagged in #1662. Exercised directly so the test does
# not depend on having a working CUDA/GDS/nvCOMP stack.
# ---------------------------------------------------------------------------


def test_warn_or_raise_gpu_fallback_default_warns(clear_strict_env):
    """Default mode emits one GeoTIFFFallbackWarning carrying type + msg."""
    from xrspatial.geotiff._gpu_decode import _warn_or_raise_gpu_fallback

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = _warn_or_raise_gpu_fallback(
            "_try_nvjpeg_batch_decode", RuntimeError("bogus 1662"))

    # Default mode returns False so the caller falls back to None.
    assert result is False
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    msg = str(fallback_warnings[0].message)
    assert '_try_nvjpeg_batch_decode' in msg
    assert 'RuntimeError' in msg
    assert 'bogus 1662' in msg


def test_warn_or_raise_gpu_fallback_strict_returns_true(set_strict_env):
    """Strict mode returns True so the caller can ``raise`` itself.

    The helper deliberately does not raise here -- re-raising the
    captured exception from inside this frame would clobber the
    original traceback. Returning True lets each call site bubble the
    live exception up via a bare ``raise`` from its own ``except``
    block.
    """
    from xrspatial.geotiff._gpu_decode import _warn_or_raise_gpu_fallback

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = _warn_or_raise_gpu_fallback(
            "_try_nvjpeg_batch_decode", RuntimeError("bogus 1662 strict"))

    assert result is True
    # No warning should be emitted in strict mode.
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert fallback_warnings == []


def test_warn_or_raise_gpu_fallback_preserves_traceback(set_strict_env):
    """The call-site pattern preserves the original exception traceback.

    The helper returns True in strict mode; the caller re-raises with a
    bare ``raise``. The resulting traceback should point at the original
    failure (``raise RuntimeError(...)`` below), not at the helper.
    """
    from xrspatial.geotiff._gpu_decode import _warn_or_raise_gpu_fallback

    def site():
        try:
            raise RuntimeError("bogus 1662 traceback")
        except Exception as e:
            if _warn_or_raise_gpu_fallback("_dummy_stage", e):
                raise
            return None

    with pytest.raises(RuntimeError, match='bogus 1662 traceback') as excinfo:
        site()
    # The deepest traceback frame should be ``site``'s ``raise
    # RuntimeError`` line, not ``_warn_or_raise_gpu_fallback``.
    tb = excinfo.tb
    while tb.tb_next is not None:
        tb = tb.tb_next
    assert tb.tb_frame.f_code.co_name == 'site'


# ---------------------------------------------------------------------------
# read_geotiff_gpu on_gpu_failure='auto' + env var integration
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()


@pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required for read_geotiff_gpu fallback test",
)
def test_read_geotiff_gpu_env_var_promotes_to_strict(monkeypatch, tmp_path):
    """With on_gpu_failure='auto' but XRSPATIAL_GEOTIFF_STRICT=1, a GPU
    decode failure surfaces instead of falling back to CPU.

    The seam: ``read_geotiff_gpu`` does a local
    ``from ._gpu_decode import gpu_decode_tiles_from_file`` inside the
    function body, so rebinding the attribute on the
    ``xrspatial.geotiff._gpu_decode`` module is picked up on the next
    call. Stubbing it to raise lets us exercise both branches against
    a real on-disk TIF without needing a broken GPU stack.
    """
    import cupy as cp
    import numpy as np
    import xarray as xr

    from xrspatial.geotiff import _gpu_decode, read_geotiff_gpu, to_geotiff

    # 1. Write a small valid TIF so the metadata parse succeeds and we
    # reach the GPU decode stage.
    h, w = 16, 16
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w)
    y = np.arange(h, dtype=np.float64) * -1.0 + 100.0
    x = np.arange(w, dtype=np.float64) * 1.0 + 0.0
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x})
    p = str(tmp_path / 'strict_promote_1662.tif')
    to_geotiff(da, p, crs=4326)

    sentinel = "bogus 1662 promote"

    def _boom(*args, **kwargs):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _boom)
    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles', _boom)

    # 2. Default mode: XRSPATIAL_GEOTIFF_STRICT unset, the failure
    # should be absorbed and the CPU fallback should return a
    # CuPy-backed DataArray.
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = read_geotiff_gpu(p)
    assert isinstance(result, xr.DataArray)
    assert isinstance(result.data, cp.ndarray)

    # 3. With XRSPATIAL_GEOTIFF_STRICT=1, the same call should re-raise
    # the patched RuntimeError instead of falling back.
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')
    with pytest.raises(RuntimeError, match=sentinel):
        read_geotiff_gpu(p)
