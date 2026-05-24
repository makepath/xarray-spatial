"""Opt-in gates for experimental and internal-only GeoTIFF paths (#2352).

Background
----------
Issue #2340 tiers the GeoTIFF release contract into Stable / Advanced /
Experimental / Internal-only. PR 1 of the epic (#2348) lined up the
``SUPPORTED_FEATURES`` constant with that tier shape. PR 4 (this issue,
#2352) extends the writer-side opt-in shape onto every Experimental /
Internal-only path that did not yet have one.

What this file pins
-------------------
* Read-side codec gate (LERC / JPEG2000 / J2K / LZ4 / JPEG-in-TIFF):
  ``open_geotiff`` / ``read_geotiff_dask`` / ``read_geotiff_gpu``
  reject a source whose Compression tag selects an experimental or
  internal-only codec unless the caller passes the matching flag
  (``allow_experimental_codecs=True`` or ``allow_internal_only_jpeg=
  True``). The writer already enforces these flags; the read side
  matches the same shape.
* Writer rich-tag gate: ``to_geotiff`` / ``write_geotiff_gpu`` reject
  a DataArray whose attrs carry ``gdal_metadata_xml`` or ``extra_tags``
  unless the caller passes ``allow_experimental_codecs=True``. Both
  attrs ride the Experimental tier in ``SUPPORTED_FEATURES`` because
  the bytes are written verbatim and downstream interop depends on the
  payload.
* Each rejection message names the missing flag, the feature, and the
  tier so the call site can be fixed in one line.
* Signature checks pin the new kwargs on the public entry points.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (open_geotiff, read_geotiff_dask, read_geotiff_gpu,
                               to_geotiff, write_geotiff_gpu)
from xrspatial.geotiff._attrs import (_COMPRESSION_TAG_TO_NAME, _validate_read_codec_optin,
                                       _validate_write_rich_tag_optin)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_float32_da(h: int = 32, w: int = 32) -> xr.DataArray:
    """Small float32 raster used for the write-side gate."""
    rng = np.random.RandomState(0)
    arr = rng.standard_normal((h, w)).astype(np.float32)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": np.arange(h, dtype=np.float64),
            "x": np.arange(w, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )


def _write_test_tif(tmp_path, compression: str,
                    *, allow_experimental_codecs=False,
                    allow_internal_only_jpeg=False,
                    dtype=np.float32):
    """Write a small file with the requested codec so the read side has
    a real target. Returns the file path. Skips when the optional
    encoder dependency is missing."""
    h = w = 32
    rng = np.random.RandomState(0)
    if dtype == np.uint8:
        arr = rng.randint(0, 256, size=(h, w), dtype=np.uint8)
    else:
        arr = rng.standard_normal((h, w)).astype(dtype)
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": np.arange(h, dtype=np.float64),
            "x": np.arange(w, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )
    path = os.path.join(str(tmp_path), f'src_{compression}.tif')
    try:
        to_geotiff(
            da, path, compression=compression,
            allow_experimental_codecs=allow_experimental_codecs,
            allow_internal_only_jpeg=allow_internal_only_jpeg,
        )
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"optional encoder missing for {compression}: {e}")
    return path


# ---------------------------------------------------------------------------
# Signature tests: every public read entry point exposes the new flags.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [open_geotiff, read_geotiff_dask, read_geotiff_gpu])
def test_read_signature_has_codec_optin(fn):
    """``open_geotiff`` / ``read_geotiff_dask`` / ``read_geotiff_gpu``
    expose ``allow_experimental_codecs=False`` and
    ``allow_internal_only_jpeg=False``. The default is ``False`` so
    accidental removal of the gate would surface here.
    """
    params = inspect.signature(fn).parameters
    assert 'allow_experimental_codecs' in params, fn.__name__
    assert params['allow_experimental_codecs'].default is False
    assert 'allow_internal_only_jpeg' in params, fn.__name__
    assert params['allow_internal_only_jpeg'].default is False


# ---------------------------------------------------------------------------
# Helper unit tests: the validators raise on the codec / attrs surface
# without an opt-in and accept the call with one. These do not require
# disk IO.
# ---------------------------------------------------------------------------


def test_validate_read_codec_optin_accepts_stable_codecs():
    """A stable codec (deflate / none / lzw / zstd / packbits) does not
    require any opt-in regardless of the flag values.
    """
    for tag in (1, 5, 8, 32773, 50000):  # none, lzw, deflate, packbits, zstd
        _validate_read_codec_optin(
            tag,
            allow_experimental_codecs=False,
            allow_internal_only_jpeg=False,
        )


@pytest.mark.parametrize("codec_name", ['lerc', 'jpeg2000', 'lz4'])
def test_validate_read_codec_optin_rejects_experimental(codec_name):
    """LERC / JPEG2000 / LZ4 raise ``ValueError`` whose message names
    ``allow_experimental_codecs`` so the caller can find the flag from
    the error itself.
    """
    tag = {
        v: k for k, v in _COMPRESSION_TAG_TO_NAME.items()
    }[codec_name]
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        _validate_read_codec_optin(
            tag,
            allow_experimental_codecs=False,
            allow_internal_only_jpeg=False,
        )


def test_validate_read_codec_optin_rejects_jpeg():
    """JPEG-in-TIFF raises ``ValueError`` whose message names
    ``allow_internal_only_jpeg`` -- the dedicated flag, NOT
    ``allow_experimental_codecs``. The two flags do not collapse.
    """
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        _validate_read_codec_optin(
            7,  # COMPRESSION_JPEG
            allow_experimental_codecs=False,
            allow_internal_only_jpeg=False,
        )
    # ``allow_experimental_codecs=True`` does NOT cover JPEG.
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        _validate_read_codec_optin(
            7,
            allow_experimental_codecs=True,
            allow_internal_only_jpeg=False,
        )


def test_validate_read_codec_optin_accepts_jpeg_with_flag():
    """With ``allow_internal_only_jpeg=True`` the read-side gate lets
    JPEG-in-TIFF through.
    """
    _validate_read_codec_optin(
        7,
        allow_experimental_codecs=False,
        allow_internal_only_jpeg=True,
    )


@pytest.mark.parametrize("codec_name", ['lerc', 'jpeg2000', 'lz4'])
def test_validate_read_codec_optin_accepts_experimental_with_flag(codec_name):
    """With ``allow_experimental_codecs=True`` the read-side gate lets
    LERC / JPEG2000 / LZ4 through.
    """
    tag = {
        v: k for k, v in _COMPRESSION_TAG_TO_NAME.items()
    }[codec_name]
    _validate_read_codec_optin(
        tag,
        allow_experimental_codecs=True,
        allow_internal_only_jpeg=False,
    )


def test_validate_read_codec_optin_message_names_feature_and_tier():
    """The rejection message names the codec, the missing flag, the
    SUPPORTED_FEATURES tier, and the parent epic so a reader can fix
    the call site without grepping the source.
    """
    with pytest.raises(ValueError) as exc:
        _validate_read_codec_optin(
            34887,  # LERC
            allow_experimental_codecs=False,
            allow_internal_only_jpeg=False,
        )
    msg = str(exc.value)
    assert 'lerc' in msg
    assert 'allow_experimental_codecs' in msg
    assert 'experimental' in msg
    assert '#2340' in msg


def test_validate_write_rich_tag_optin_accepts_empty_attrs():
    """No rich-tag attrs and no opt-in: the writer gate is a no-op."""
    _validate_write_rich_tag_optin(
        {}, allow_experimental_codecs=False)


def test_validate_write_rich_tag_optin_rejects_gdal_metadata_xml():
    """``attrs['gdal_metadata_xml']`` triggers the gate; rejection
    message names the attr and the opt-in flag.
    """
    with pytest.raises(ValueError, match='gdal_metadata_xml'):
        _validate_write_rich_tag_optin(
            {'gdal_metadata_xml': '<GDALMetadata/>'},
            allow_experimental_codecs=False,
        )


def test_validate_write_rich_tag_optin_rejects_extra_tags():
    """``attrs['extra_tags']`` triggers the gate; rejection message
    names the attr and the opt-in flag.
    """
    with pytest.raises(ValueError, match='extra_tags'):
        _validate_write_rich_tag_optin(
            {'extra_tags': [(700, 1, 0, b'')]},
            allow_experimental_codecs=False,
        )


def test_validate_write_rich_tag_optin_accepts_with_flag():
    """``allow_experimental_codecs=True`` accepts both rich-tag attrs."""
    _validate_write_rich_tag_optin(
        {'gdal_metadata_xml': '<GDALMetadata/>',
         'extra_tags': [(700, 1, 0, b'')]},
        allow_experimental_codecs=True,
    )


def test_validate_write_rich_tag_optin_exempts_round_trip():
    """An attrs dict carrying the ``_xrspatial_geotiff_contract`` marker
    came from an xrspatial read; round-tripping it back through
    ``to_geotiff`` is the canonical contract from #1984 and must not
    require a new flag. The marker is the gate's exemption signal.
    """
    _validate_write_rich_tag_optin(
        {'gdal_metadata_xml': '<GDALMetadata/>',
         'extra_tags': [(700, 1, 0, b'')],
         '_xrspatial_geotiff_contract': 2},
        allow_experimental_codecs=False,
    )


# ---------------------------------------------------------------------------
# Read end-to-end: write an experimental-codec file via the existing
# writer opt-in, then assert open_geotiff refuses to read it without the
# matching opt-in and succeeds with it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ['lerc', 'lz4'])
def test_open_geotiff_rejects_experimental_codec(tmp_path, codec):
    """A file written with LERC or LZ4 raises ``ValueError`` on read
    by default; the message names ``allow_experimental_codecs``.
    """
    path = _write_test_tif(
        tmp_path, codec, allow_experimental_codecs=True)
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        open_geotiff(path)


@pytest.mark.parametrize("codec", ['lerc', 'lz4'])
def test_open_geotiff_accepts_experimental_codec_with_flag(tmp_path, codec):
    """``allow_experimental_codecs=True`` lets the read through and
    returns a DataArray with the expected shape.
    """
    path = _write_test_tif(
        tmp_path, codec, allow_experimental_codecs=True)
    try:
        da = open_geotiff(path, allow_experimental_codecs=True)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"optional decoder missing for {codec}: {e}")
    assert da.shape == (32, 32)


def test_open_geotiff_rejects_jpeg2000(tmp_path):
    """JPEG2000 is experimental and requires the same opt-in as LERC /
    LZ4. ``j2k`` is an alias the writer maps to the same codec, so
    only one source file is needed.
    """
    path = _write_test_tif(
        tmp_path, 'jpeg2000', allow_experimental_codecs=True,
        dtype=np.uint8)
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        open_geotiff(path)


def test_open_geotiff_rejects_jpeg_internal_only(tmp_path):
    """JPEG-in-TIFF is internal-only; the dedicated flag
    ``allow_internal_only_jpeg`` is the gate. Mirrors the writer side
    where ``allow_experimental_codecs`` does NOT cover JPEG.
    """
    path = _write_test_tif(
        tmp_path, 'jpeg', allow_internal_only_jpeg=True,
        dtype=np.uint8)
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        open_geotiff(path)
    # ``allow_experimental_codecs=True`` does NOT unlock JPEG-in-TIFF
    # on the read side either.
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        open_geotiff(path, allow_experimental_codecs=True)


def test_open_geotiff_accepts_jpeg_internal_only_with_flag(tmp_path):
    """``allow_internal_only_jpeg=True`` lets the read through."""
    path = _write_test_tif(
        tmp_path, 'jpeg', allow_internal_only_jpeg=True,
        dtype=np.uint8)
    da = open_geotiff(path, allow_internal_only_jpeg=True)
    assert da.shape == (32, 32)


def test_read_geotiff_dask_rejects_experimental_codec(tmp_path):
    """The dask read path fires the gate at graph build, before any
    chunk task is scheduled.
    """
    path = _write_test_tif(
        tmp_path, 'lz4', allow_experimental_codecs=True)
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        read_geotiff_dask(path, chunks=16)


def test_read_geotiff_dask_accepts_experimental_codec_with_flag(tmp_path):
    """``allow_experimental_codecs=True`` lets the dask graph build."""
    path = _write_test_tif(
        tmp_path, 'lz4', allow_experimental_codecs=True)
    try:
        da = read_geotiff_dask(
            path, chunks=16, allow_experimental_codecs=True)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"optional decoder missing: {e}")
    assert da.shape == (32, 32)


# ---------------------------------------------------------------------------
# Writer rich-tag attrs: gdal_metadata_xml / extra_tags require the
# experimental opt-in.
# ---------------------------------------------------------------------------


def test_to_geotiff_rejects_gdal_metadata_xml_without_flag(tmp_path):
    """A DataArray whose attrs carry ``gdal_metadata_xml`` is rejected
    by ``to_geotiff`` unless the caller passes
    ``allow_experimental_codecs=True``. The message names the attr.
    """
    da = _make_float32_da()
    da.attrs['gdal_metadata_xml'] = (
        '<GDALMetadata><Item name="STATISTICS_MIN">0</Item>'
        '</GDALMetadata>'
    )
    path = os.path.join(str(tmp_path), 'rich_xml.tif')
    with pytest.raises(ValueError, match='gdal_metadata_xml'):
        to_geotiff(da, path)


def test_to_geotiff_rejects_extra_tags_without_flag(tmp_path):
    """Same shape as the ``gdal_metadata_xml`` case but for
    ``attrs['extra_tags']``. Both surfaces feed the same on-disk path
    and ride the same Experimental tier.
    """
    da = _make_float32_da()
    da.attrs['extra_tags'] = [(700, 1, 0, b'')]
    path = os.path.join(str(tmp_path), 'rich_extra.tif')
    with pytest.raises(ValueError, match='extra_tags'):
        to_geotiff(da, path)


def test_to_geotiff_accepts_rich_tags_with_flag(tmp_path):
    """``allow_experimental_codecs=True`` lets both attrs through and
    the write completes.
    """
    da = _make_float32_da()
    da.attrs['gdal_metadata_xml'] = (
        '<GDALMetadata><Item name="STATISTICS_MIN">0</Item>'
        '</GDALMetadata>'
    )
    da.attrs['extra_tags'] = [(700, 1, 0, b'')]
    path = os.path.join(str(tmp_path), 'rich_optin.tif')
    out = to_geotiff(da, path, allow_experimental_codecs=True)
    assert out == path
    assert os.path.exists(path)


def test_write_geotiff_gpu_rejects_rich_tags_without_flag(tmp_path):
    """The GPU writer mirrors ``to_geotiff`` so the two writers expose
    a consistent surface; the rejection fires before any GPU work and
    does not depend on cupy being installed.
    """
    da = _make_float32_da()
    da.attrs['gdal_metadata_xml'] = (
        '<GDALMetadata><Item name="STATISTICS_MIN">0</Item>'
        '</GDALMetadata>'
    )
    path = os.path.join(str(tmp_path), 'rich_gpu.tif')
    with pytest.raises(ValueError, match='gdal_metadata_xml'):
        write_geotiff_gpu(da, path)


# ---------------------------------------------------------------------------
# Already-gated paths: pin the existing behaviour so a future refactor
# that drops a flag fails this file rather than passing in CI.
# ---------------------------------------------------------------------------


def test_allow_rotated_default_raises_already_gated(tmp_path):
    """``allow_rotated=False`` (the default) raises on a rotated read.
    Pinned here so the Experimental + Internal-only opt-in inventory
    in PR 4 lives next to the existing ``allow_rotated`` /
    ``allow_unparseable_crs`` gates and a future refactor cannot drop
    one of them without failing this file.

    The PR 1 audit (#2348) demoted ``reader.allow_rotated`` from
    advanced to experimental, so the gate already matches the epic.
    """
    # A signature pin is enough -- the actual rotated-read behaviour is
    # covered by the existing test_allow_rotated_geotiff_2115.py suite.
    params = inspect.signature(open_geotiff).parameters
    assert 'allow_rotated' in params
    assert params['allow_rotated'].default is False


def test_allow_unparseable_crs_default_raises_already_gated():
    """``allow_unparseable_crs=False`` (the default) raises on an
    unparseable CRS string. The PR 1 audit (#2348) demoted
    ``reader.allow_unparseable_crs`` to experimental, so the gate
    already matches the epic. Pin the signature here next to the new
    PR 4 opt-ins so the inventory lives in one file.
    """
    params = inspect.signature(open_geotiff).parameters
    assert 'allow_unparseable_crs' in params
    assert params['allow_unparseable_crs'].default is False


def test_gpu_read_requires_explicit_optin():
    """GPU read is Experimental in ``SUPPORTED_FEATURES`` and the
    opt-in is the boolean ``gpu=True`` kwarg. Pin the default here so
    a future refactor cannot flip GPU read to auto-on.
    """
    params = inspect.signature(open_geotiff).parameters
    assert 'gpu' in params
    assert params['gpu'].default is False


def test_gpu_write_requires_explicit_optin():
    """GPU write is Experimental and gates on ``gpu=True`` /
    ``gpu=None`` (auto-detect from CuPy data). Pin the default here:
    ``None`` is the documented auto-detect sentinel and ``False`` /
    ``True`` are the explicit selectors. A flip to ``True`` default
    would silently route every NumPy write through the GPU pipeline.
    """
    params = inspect.signature(to_geotiff).parameters
    assert 'gpu' in params
    assert params['gpu'].default is None
