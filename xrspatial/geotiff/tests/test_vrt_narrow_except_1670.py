"""Regression tests for #1670: narrow ``except Exception`` in VRT reader.

``read_vrt`` historically wrapped each source read in ``except Exception``,
which under default mode swallowed any subclass of ``Exception`` -- including
``RuntimeError``, ``MemoryError``, and other bugs unrelated to the "source
file is unreadable" fallback the catch was meant to cover.

This module pins the narrowed contract:

* I/O errors (``OSError`` and its subclasses), parse errors
  (``ValueError``, ``struct.error``), and codec-library decode errors
  (``zlib.error`` for deflate; ``zstandard.ZstdError`` when zstandard is
  installed) warn-and-continue in default mode and re-raise under
  ``XRSPATIAL_GEOTIFF_STRICT=1``.
* Other exception types (``RuntimeError`` here as a stand-in for real bugs)
  propagate in both modes so callers and CI see them.
"""
from __future__ import annotations

import struct
import warnings
import zlib

import pytest

from xrspatial.geotiff import GeoTIFFFallbackWarning


@pytest.fixture
def clear_strict_env(monkeypatch):
    """Ensure XRSPATIAL_GEOTIFF_STRICT is unset for default-mode tests."""
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)


@pytest.fixture
def set_strict_env(monkeypatch):
    """Set XRSPATIAL_GEOTIFF_STRICT=1 for strict-mode tests."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')


def _write_simple_vrt(tmp_path, src_path, name='mosaic_1670.vrt'):
    """Write a 4x4 VRT pointing at a single source path."""
    vrt_path = tmp_path / name
    vrt_path.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS></SRS>\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <NoDataValue>-9999</NoDataValue>\n'
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
    return vrt_path


def _patch_read_to_array(monkeypatch, exc):
    """Make ``read_to_array`` inside ``_vrt`` raise ``exc`` on every call.

    The VRT reader does a local ``from ._reader import read_to_array`` inside
    ``read_vrt``, so we patch the source attribute and the import will pick up
    the stub.
    """
    from xrspatial.geotiff import _reader

    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(_reader, 'read_to_array', _boom)


def test_runtime_error_propagates_default_mode(
    clear_strict_env, monkeypatch, tmp_path,
):
    """A non-I/O bug (RuntimeError) must NOT be absorbed by the VRT fallback.

    Default mode used to catch ``Exception``, which swallowed real bugs. The
    narrowed catch only handles I/O / parse errors, so a RuntimeError raised
    from ``read_to_array`` should bubble straight up.
    """
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670.tif'
    src_path.write_bytes(b'placeholder')  # parse will be replaced by stub
    vrt_path = _write_simple_vrt(tmp_path, str(src_path))

    _patch_read_to_array(monkeypatch, RuntimeError("synthetic 1670"))

    with pytest.raises(RuntimeError, match='synthetic 1670'):
        read_vrt(str(vrt_path))


def test_runtime_error_propagates_strict_mode(
    set_strict_env, monkeypatch, tmp_path,
):
    """Strict mode already propagates everything; double-check RuntimeError."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_strict.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_rt_strict.vrt')

    _patch_read_to_array(monkeypatch, RuntimeError("synthetic 1670 strict"))

    with pytest.raises(RuntimeError, match='synthetic 1670 strict'):
        read_vrt(str(vrt_path))


def test_file_not_found_warns_and_continues(
    clear_strict_env, monkeypatch, tmp_path,
):
    """FileNotFoundError is the canonical "source is unreadable" case."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_fnf.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_fnf.vrt')

    _patch_read_to_array(
        monkeypatch, FileNotFoundError("synthetic missing 1670"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path))

    # Mosaic still loads (with a hole) and the skipped source is reported.
    assert da.shape == (4, 4)
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) >= 1
    msgs = ' '.join(str(x.message) for x in fallback_warnings)
    assert 'VRT source' in msgs
    assert 'FileNotFoundError' in msgs


def test_file_not_found_strict_reraises(
    set_strict_env, monkeypatch, tmp_path,
):
    """Strict mode re-raises the FileNotFoundError."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_fnf_strict.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_fnf_strict.vrt')

    _patch_read_to_array(
        monkeypatch, FileNotFoundError("synthetic missing 1670 strict"))

    with pytest.raises(FileNotFoundError, match='synthetic missing 1670 strict'):
        read_vrt(str(vrt_path))


def test_value_error_warns_and_continues(
    clear_strict_env, monkeypatch, tmp_path,
):
    """A typed ValueError from parse_header / parse_ifd is a documented case."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_val.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_val.vrt')

    _patch_read_to_array(
        monkeypatch, ValueError("bad header 1670"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path))

    assert da.shape == (4, 4)
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) >= 1
    msgs = ' '.join(str(x.message) for x in fallback_warnings)
    assert 'VRT source' in msgs
    assert 'ValueError' in msgs


def test_value_error_strict_reraises(
    set_strict_env, monkeypatch, tmp_path,
):
    """Strict mode re-raises the ValueError."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_val_strict.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_val_strict.vrt')

    _patch_read_to_array(
        monkeypatch, ValueError("bad header 1670 strict"))

    with pytest.raises(ValueError, match='bad header 1670 strict'):
        read_vrt(str(vrt_path))


def test_struct_error_warns_and_continues(
    clear_strict_env, monkeypatch, tmp_path,
):
    """struct.error still leaks from some parse paths; catch it too.

    Future PRs may convert these to ValueError, but until then the VRT
    reader should be robust to the raw ``struct.error`` exception so a
    single malformed source does not abort the whole mosaic.
    """
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_se.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_se.vrt')

    _patch_read_to_array(
        monkeypatch, struct.error("short buffer 1670"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path))

    assert da.shape == (4, 4)
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) >= 1
    msgs = ' '.join(str(x.message) for x in fallback_warnings)
    assert 'VRT source' in msgs
    assert 'error' in msgs  # struct.error type name is 'error'


def test_permission_error_warns_and_continues(
    clear_strict_env, monkeypatch, tmp_path,
):
    """PermissionError is an OSError subclass, so it goes through the fallback."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_perm.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_perm.vrt')

    _patch_read_to_array(
        monkeypatch, PermissionError("denied 1670"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path))

    assert da.shape == (4, 4)
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) >= 1


def test_memory_error_propagates_default_mode(
    clear_strict_env, monkeypatch, tmp_path,
):
    """MemoryError is a real failure, not an "unreadable source"; propagate."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_mem.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_mem.vrt')

    _patch_read_to_array(monkeypatch, MemoryError("OOM 1670"))

    with pytest.raises(MemoryError, match='OOM 1670'):
        read_vrt(str(vrt_path))


def test_zlib_error_warns_and_continues(
    clear_strict_env, monkeypatch, tmp_path,
):
    """A ``zlib.error`` from a corrupt deflate tile should warn-and-skip.

    The narrowed catch in PR #1675 covered only ``(OSError, ValueError,
    struct.error)``; ``zlib.error`` is a direct ``Exception`` subclass and
    used to abort the whole mosaic when a deflate tile was corrupt. The
    follow-up adds codec-library decode errors to the allowlist so the
    historical warn-and-skip behaviour is restored for corrupt payloads.
    """
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_zlib.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_zlib.vrt')

    _patch_read_to_array(monkeypatch, zlib.error("synthetic deflate 1670"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path))

    assert da.shape == (4, 4)
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) >= 1
    msgs = ' '.join(str(x.message) for x in fallback_warnings)
    assert 'VRT source' in msgs
    assert 'error' in msgs  # zlib.error type name is 'error'


def test_zlib_error_strict_reraises(
    set_strict_env, monkeypatch, tmp_path,
):
    """Under ``XRSPATIAL_GEOTIFF_STRICT=1`` a corrupt deflate tile re-raises."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_zlib_strict.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_zlib_strict.vrt')

    _patch_read_to_array(
        monkeypatch, zlib.error("synthetic deflate 1670 strict"))

    with pytest.raises(zlib.error, match='synthetic deflate 1670 strict'):
        read_vrt(str(vrt_path))


def test_zstd_error_warns_and_continues_if_available(
    clear_strict_env, monkeypatch, tmp_path,
):
    """``zstandard.ZstdError`` from a corrupt ZSTD tile should warn-and-skip.

    Skipped when ``zstandard`` is not installed -- the wrapper path that
    raises this exception is unreachable in that case and there's no class
    to instantiate. See ``_CODEC_DECODE_EXCEPTIONS`` in ``_vrt.py``.
    """
    pytest.importorskip('zstandard')
    from zstandard import ZstdError
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / 'src_1670_zstd.tif'
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(
        tmp_path, str(src_path), name='mosaic_1670_zstd.vrt')

    _patch_read_to_array(monkeypatch, ZstdError("synthetic zstd 1670"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path))

    assert da.shape == (4, 4)
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) >= 1
    msgs = ' '.join(str(x.message) for x in fallback_warnings)
    assert 'VRT source' in msgs
    assert 'ZstdError' in msgs
