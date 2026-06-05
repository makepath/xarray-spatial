"""Photometric interpretation and codec-tier API surface tests.

This file groups two areas that share a *concern* rather than runtime
logic: how the reader/writer expose photometric and codec-tier choices
through the public API.

Section 1 -- ``MinIsWhite`` read/write semantics
    The reader unconditionally inverts single-band MinIsWhite pixels.
    The writer must pre-invert pixels so a write/read round-trip is
    identity. Three regression bundles live here:

    * uint/float MinIsWhite with ``GDAL_NODATA`` masking. The mask must
      compare against the *post-inversion* sentinel; comparing against
      the original sentinel mis-masks pixels that collide with
      ``iinfo.max - sentinel``.
    * writer round-trip for the unsigned and float MinIsWhite paths,
      including the rejection branches (signed-int writer, dask writer,
      COG/overview writer, contradictory ``extra_tags`` photometric).
    * signed-MinIsWhite rejection on both read and write paths. The
      message must include ``SampleFormat``, ``BitsPerSample``, and
      ``Photometric`` so callers can diagnose without digging into the
      spec.

Section 2 -- ``allow_internal_only_jpeg`` API tier acceptance
    The CPU dispatcher ``to_geotiff`` exposes the same opt-in flag the
    GPU writer ``_write_geotiff_gpu`` does, so callers on the auto-
    dispatch path can reach the experimental internal-reader-only JPEG
    encode. The tests pin signature parity, the default-rejection
    message, the opt-in warning + round-trip, and the GPU dispatch
    forwarding.

    Section 2 shares a home with Section 1 because both exercise the
    photometric / codec-tier API surface: the writer's MinIsWhite gate
    and the dispatcher's JPEG gate are sibling pieces of the same
    "advertised tag must match the encode contract" surface. They do
    *not* share runtime logic; the boundary comment below restates this
    so future readers do not look for shared helpers across the two.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning, _write_geotiff_gpu, open_geotiff, to_geotiff
from xrspatial.geotiff._header import TAG_PHOTOMETRIC, TAG_SAMPLE_FORMAT, parse_header

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# Shared GPU gate. Reuses the helper-marker pattern; we keep a local
# probe so the MinIsWhite GPU regressions can stay co-located.
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:  # pragma: no cover - import-time errors only
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


# ===========================================================================
# Section 1: MinIsWhite read/write semantics
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers (MinIsWhite)
# ---------------------------------------------------------------------------


def _write_miniswhite_tiff(path: str, stored: np.ndarray, nodata_str: str,
                           tiled: bool = False) -> None:
    """Write a MinIsWhite TIFF with an inline ``GDAL_NODATA`` tag."""
    extratags = [("GDAL_NODATA", "s", 0, f"{nodata_str}\0", True)]
    kwargs = {"photometric": "miniswhite", "extratags": extratags}
    if tiled:
        kwargs["tile"] = (16, 16)
    tifffile.imwrite(path, stored, **kwargs)


def _da(arr: np.ndarray, attrs_extra=None) -> xr.DataArray:
    """Wrap an ndarray as a ``to_geotiff``-compatible DataArray.

    Square 1xN strips trigger the degenerate-axis rule; opt into the
    borrow-from-other-axis fallback for those shapes so the writer
    accepts the call.
    """
    h, w = arr.shape
    attrs = {'res': (1.0, 1.0)}
    if h == 1 or w == 1:
        attrs['assume_square_pixels_for_degenerate_axis'] = True
    if attrs_extra:
        attrs.update(attrs_extra)
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(h, dtype=np.float64),
                'x': np.arange(w, dtype=np.float64)},
        attrs=attrs,
    )


def _patch_tag_value(data: bytearray, tag: int, value: int) -> None:
    """Rewrite the inline value slot of *tag* in the first IFD.

    Used to forge a signed-int MinIsWhite TIFF on the read side: write a
    normal MinIsBlack int16 file, then flip the Photometric tag to
    MinIsWhite on disk.
    """
    header = parse_header(bytes(data))
    bo = header.byte_order
    ifd_offset = header.first_ifd_offset
    num_entries = struct.unpack_from(f"{bo}H", data, ifd_offset)[0]
    entry_offset = ifd_offset + 2
    for i in range(num_entries):
        eo = entry_offset + i * 12
        cur_tag = struct.unpack_from(f"{bo}H", data, eo)[0]
        if cur_tag != tag:
            continue
        type_id = struct.unpack_from(f"{bo}H", data, eo + 2)[0]
        count = struct.unpack_from(f"{bo}I", data, eo + 4)[0]
        assert type_id == 3 and count == 1, (
            f"tag {tag} not the expected SHORT/count=1 layout: "
            f"type={type_id} count={count}"
        )
        struct.pack_into(f"{bo}H", data, eo + 8, value)
        return
    raise AssertionError(f"tag {tag} not found in IFD")


def _patch_first_tile_sparse(path: str) -> None:
    """Zero TileOffsets[0] and TileByteCounts[0] in place.

    Handles SHORT and LONG type encodings for both tags.
    """
    from xrspatial.geotiff._header import parse_header as _parse_header

    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    header = _parse_header(bytes(data))
    bo = header.byte_order
    ifd_offset = header.first_ifd_offset
    num_entries = struct.unpack_from(f"{bo}H", data, ifd_offset)[0]
    entry_offset = ifd_offset + 2

    targets = {324, 325}  # TileOffsets, TileByteCounts
    for i in range(num_entries):
        eo = entry_offset + i * 12
        tag = struct.unpack_from(f"{bo}H", data, eo)[0]
        if tag not in targets:
            continue
        type_id = struct.unpack_from(f"{bo}H", data, eo + 2)[0]
        count = struct.unpack_from(f"{bo}I", data, eo + 4)[0]
        if type_id == 4:  # LONG
            total = count * 4
            if total <= 4:
                struct.pack_into(f"{bo}I", data, eo + 8, 0)
            else:
                ptr = struct.unpack_from(f"{bo}I", data, eo + 8)[0]
                struct.pack_into(f"{bo}I", data, ptr, 0)
        elif type_id == 3:  # SHORT
            total = count * 2
            if total <= 4:
                struct.pack_into(f"{bo}H", data, eo + 8, 0)
            else:
                ptr = struct.unpack_from(f"{bo}I", data, eo + 8)[0]
                struct.pack_into(f"{bo}H", data, ptr, 0)

    with open(path, "wb") as fh:
        fh.write(data)


def _forge_signed_miniswhite_tif(tmp_path, name: str) -> str:
    """Write a normal int16 MinIsBlack file, then flip Photometric to 0."""
    arr = np.array([[-5, -1, 0, 1, 5]], dtype=np.int16)
    path = tmp_path / name
    to_geotiff(_da(arr), str(path))
    raw = bytearray(path.read_bytes())
    _patch_tag_value(raw, TAG_PHOTOMETRIC, 0)
    header = parse_header(bytes(raw))
    bo = header.byte_order
    ifd_offset = header.first_ifd_offset
    num_entries = struct.unpack_from(f"{bo}H", raw, ifd_offset)[0]
    found_sf = None
    for i in range(num_entries):
        eo = ifd_offset + 2 + i * 12
        cur_tag = struct.unpack_from(f"{bo}H", raw, eo)[0]
        if cur_tag == TAG_SAMPLE_FORMAT:
            found_sf = struct.unpack_from(f"{bo}H", raw, eo + 8)[0]
            break
    assert found_sf == 2, (
        f"expected forged file to already declare SampleFormat=2 (signed), "
        f"got {found_sf}"
    )
    path.write_bytes(bytes(raw))
    return str(path)


# ---------------------------------------------------------------------------
# 1.1 -- read path: MinIsWhite + nodata interaction
#
# Before the fix, the reader inverted MinIsWhite pixels before applying
# the nodata mask. The mask then compared against the *original*
# sentinel, so:
#   * stored-sentinel pixels survived as ``iinfo.max - sentinel``
#   * pixels equal to ``iinfo.max - sentinel`` got masked to NaN
# Every backend (eager numpy, dask, eager GPU) shared the bug. The fix
# stashes the post-inversion sentinel on ``geo_info._mask_nodata`` and
# routes every backend's mask through it, leaving ``attrs['nodata']``
# at the original sentinel for write-time round-trip.
# ---------------------------------------------------------------------------


def _uint8_case():
    stored = np.array([[0, 100, 200], [50, 0, 255]], dtype=np.uint8)
    expected = np.array(
        [[np.nan, 155.0, 55.0], [205.0, np.nan, 0.0]], dtype=np.float64
    )
    return stored, expected


def _uint16_case():
    stored = np.array([[0, 1000, 32000], [65535, 50000, 0]], dtype=np.uint16)
    expected = np.array(
        [[65535.0, 64535.0, 33535.0], [np.nan, 15535.0, 65535.0]],
        dtype=np.float64,
    )
    return stored, expected


def _float32_case():
    stored = np.array(
        [[0.0, 100.0, -9999.0], [50.0, 9999.0, 200.0]], dtype=np.float32
    )
    expected = np.array(
        [[-0.0, -100.0, np.nan], [-50.0, -9999.0, -200.0]], dtype=np.float32
    )
    return stored, expected


@pytest.mark.parametrize(
    "case_factory,nodata_str,sentinel",
    [
        pytest.param(_uint8_case, "0", 0, id="miniswhite_read[uint8-nodata-0]"),
        pytest.param(_uint16_case, "65535", 65535,
                     id="miniswhite_read[uint16-nodata-max]"),
        pytest.param(_float32_case, "-9999", -9999,
                     id="miniswhite_read[float32-nodata-neg9999]"),
    ],
)
def test_eager_numpy_miniswhite_nodata(
        tmp_path, case_factory, nodata_str, sentinel):
    """Eager numpy read masks against the post-inversion sentinel."""
    stored, expected = case_factory()
    path = str(tmp_path / "mw_eager.tif")
    _write_miniswhite_tiff(path, stored, nodata_str)

    arr = open_geotiff(path, masked=True)

    assert arr.attrs["nodata"] == sentinel
    np.testing.assert_array_equal(arr.values, expected)


@pytest.mark.parametrize(
    "case_factory,nodata_str,sentinel",
    [
        pytest.param(_uint8_case, "0", 0, id="miniswhite_read[uint8-dask]"),
        pytest.param(_float32_case, "-9999", -9999,
                     id="miniswhite_read[float32-dask]"),
    ],
)
def test_dask_miniswhite_nodata(tmp_path, case_factory, nodata_str, sentinel):
    """Dask read computes to the same post-inversion mask."""
    stored, expected = case_factory()
    path = str(tmp_path / "mw_dask.tif")
    _write_miniswhite_tiff(path, stored, nodata_str)

    arr = open_geotiff(path, chunks=2, masked=True).compute()

    assert arr.attrs["nodata"] == sentinel
    np.testing.assert_array_equal(arr.values, expected)


def test_eager_miniswhite_uint8_no_collision(tmp_path):
    """Sentinel that does not collide with any stored pixel after
    inversion still produces the expected NaN placement. The pre-fix
    bug only surfaced on collisions; this case must stay green to
    confirm no regression on the non-colliding path."""
    stored = np.array([[7, 100, 200], [50, 7, 230]], dtype=np.uint8)
    path = str(tmp_path / "mw_no_collision.tif")
    _write_miniswhite_tiff(path, stored, "7")
    expected = np.array(
        [[np.nan, 155.0, 55.0], [205.0, np.nan, 25.0]], dtype=np.float64
    )

    arr = open_geotiff(path, masked=True)

    np.testing.assert_array_equal(arr.values, expected)


def test_eager_miniswhite_no_nodata_stays_integer(tmp_path):
    """With no nodata, inversion stays in the integer dtype: there is
    no need to cast to float because nothing has to express NaN."""
    stored = np.array([[0, 50, 100, 200]], dtype=np.uint8).repeat(4, axis=0)
    path = str(tmp_path / "mw_no_nodata.tif")
    tifffile.imwrite(path, stored, photometric="miniswhite")

    arr = open_geotiff(path)

    assert arr.dtype == np.uint8
    np.testing.assert_array_equal(arr.values, 255 - stored)


def test_miniswhite_backend_parity_uint8_nodata_zero(tmp_path):
    """Every available backend agrees on the same MinIsWhite input."""
    stored, expected = _uint8_case()
    path = str(tmp_path / "mw_parity.tif")
    _write_miniswhite_tiff(path, stored, "0", tiled=True)

    eager = open_geotiff(path, masked=True).values
    dask_result = open_geotiff(path, chunks=2, masked=True).compute().values
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(dask_result, expected)
    np.testing.assert_array_equal(eager, dask_result)
    if _HAS_GPU:
        gpu = open_geotiff(path, gpu=True, masked=True).data.get()
        np.testing.assert_array_equal(gpu, expected)
        np.testing.assert_array_equal(eager, gpu)


@_gpu_only
def test_gpu_eager_miniswhite_uint8_nodata_zero(tmp_path):
    stored, expected = _uint8_case()
    path = str(tmp_path / "mw_uint8_gpu.tif")
    _write_miniswhite_tiff(path, stored, "0", tiled=True)

    arr = open_geotiff(path, gpu=True, masked=True)

    assert arr.attrs["nodata"] == 0
    np.testing.assert_array_equal(arr.data.get(), expected)


@_gpu_only
def test_gpu_sparse_tile_miniswhite_nodata_zero(tmp_path):
    """GPU CPU-fallback path must use the post-MinIsWhite sentinel.

    Build a tiled MinIsWhite uint8 raster with ``GDAL_NODATA=0``, then
    patch the first tile's TileOffsets/TileByteCounts to 0 so the read
    routes through the sparse-tile CPU-fallback branch of
    ``_read_geotiff_gpu``. Pre-fix this branch discarded
    ``_mask_nodata`` and masked against the original sentinel.
    """
    stored = np.zeros((32, 32), dtype=np.uint8)
    stored[:16, :16] = 0     # tile 0 (sparse) -- value irrelevant
    stored[:16, 16:] = 100   # tile 1
    stored[16:, :16] = 200   # tile 2
    stored[16:, 16:] = 255   # tile 3, collides with inverted sentinel

    path = str(tmp_path / "mw_sparse_gpu.tif")
    _write_miniswhite_tiff(path, stored, "0", tiled=True)
    _patch_first_tile_sparse(path)

    arr = open_geotiff(path, gpu=True, masked=True)
    out = arr.data.get()

    assert np.all(np.isnan(out[:16, :16]))
    assert np.all(out[:16, 16:] == 155.0)
    assert np.all(out[16:, :16] == 55.0)
    assert np.all(out[16:, 16:] == 0.0)
    assert arr.attrs["nodata"] == 0


# ---------------------------------------------------------------------------
# 1.2 -- writer round-trip: ``to_geotiff(..., photometric='miniswhite')``
#
# Writer pre-inverts pixels for unsigned-int and float dtypes. Signed
# dtypes are rejected (see 1.3). Dask, COG, and overview write paths
# do not pre-invert per tile and are also rejected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr,assert_kind",
    [
        pytest.param(
            np.array([[0, 1, 127, 254, 255]], dtype=np.uint8),
            "equal",
            id="miniswhite_roundtrip[uint8]",
        ),
        pytest.param(
            np.array([[0, 1, np.iinfo(np.uint16).max // 2,
                       np.iinfo(np.uint16).max - 1,
                       np.iinfo(np.uint16).max]], dtype=np.uint16),
            "equal",
            id="miniswhite_roundtrip[uint16]",
        ),
        pytest.param(
            np.array([[-3.5, 0.0, 0.25, 7.5]], dtype=np.float32),
            "allclose",
            id="miniswhite_roundtrip[float32]",
        ),
    ],
)
def test_miniswhite_round_trip(tmp_path, arr, assert_kind):
    """Write/read identity for the dtypes the writer accepts."""
    path = tmp_path / "rt.tif"
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    if assert_kind == "equal":
        np.testing.assert_array_equal(np.asarray(r.values), arr)
    else:
        np.testing.assert_allclose(np.asarray(r.values), arr)


def test_miniswhite_float_with_nodata_round_trips_nan(tmp_path):
    """A float NaN pixel round-trips back to NaN through the writer's
    sentinel substitution."""
    arr = np.array([[10.0, np.nan, 20.0, 30.0]], dtype=np.float32)
    path = tmp_path / "rt_float_nd.tif"
    to_geotiff(_da(arr), str(path), photometric='miniswhite', nodata=-9999.0)
    r = open_geotiff(str(path), masked=True)
    out = np.asarray(r.values)
    assert np.isnan(out[0, 1]), (
        "nodata position must round-trip back to NaN after MinIsWhite "
        f"inversion; got {out!r}"
    )
    np.testing.assert_allclose(out[0, [0, 2, 3]], arr[0, [0, 2, 3]])


def test_miniswhite_uint16_in_range_nodata_round_trips_nan(tmp_path):
    """Unsigned-nodata inversion branch: an in-range integer sentinel
    must round-trip to NaN. The on-disk sentinel byte is
    ``iinfo.max - 9999 = 55536``, which collides with real ``55536``
    pixels only when the reader masks against the correct value."""
    arr = np.array([[10, 9999, 200, 65000]], dtype=np.uint16)
    path = tmp_path / "rt_u16_nd.tif"
    da_in = xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(1, dtype=np.float64),
                'x': np.arange(4, dtype=np.float64)},
        attrs={'res': (1.0, 1.0), 'nodata': 9999,
               'assume_square_pixels_for_degenerate_axis': True},
    )
    to_geotiff(da_in, str(path), photometric='miniswhite', nodata=9999)
    r = open_geotiff(str(path), masked=True)
    out = np.asarray(r.values)
    assert np.isnan(out[0, 1]), (
        f"in-range uint nodata must round-trip to NaN through the "
        f"writer-side sentinel inversion; got {out!r}"
    )
    np.testing.assert_array_equal(
        out[0, [0, 2, 3]].astype(np.uint16), arr[0, [0, 2, 3]]
    )


def test_miniswhite_rejected_with_cog_no_overviews(tmp_path):
    """COG path cannot pre-invert per tile."""
    arr = np.zeros((512, 512), dtype=np.uint8)
    path = tmp_path / "cog_msw.tif"
    with pytest.raises(NotImplementedError, match='miniswhite'):
        to_geotiff(_da(arr), str(path), photometric='miniswhite', cog=True)


def test_miniswhite_rejected_with_explicit_overviews_no_cog(tmp_path):
    """Isolate the ``overview_levels`` branch: cog=False so only the
    overview gate fires."""
    from xrspatial.geotiff._writer import write as _eager_write
    arr = np.zeros((256, 256), dtype=np.uint8)
    path = str(tmp_path / "ov_msw.tif")
    with pytest.raises(NotImplementedError, match='miniswhite'):
        _eager_write(arr, path, photometric='miniswhite',
                     cog=False, overview_levels=[2])


def test_miniswhite_rejected_on_dask_path(tmp_path):
    """``to_geotiff`` on a dask-backed DataArray refuses miniswhite
    because the streaming writer does not pre-invert per tile."""
    dask_array = pytest.importorskip("dask.array")
    arr = dask_array.zeros((64, 64), dtype=np.uint8, chunks=32)
    da_in = xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(64, dtype=np.float64),
                'x': np.arange(64, dtype=np.float64)},
        attrs={'res': (1.0, 1.0)},
    )
    path = tmp_path / "dask_msw.tif"
    with pytest.raises(NotImplementedError, match='miniswhite'):
        to_geotiff(da_in, str(path), photometric='miniswhite')


def test_miniswhite_rejected_with_extra_tags_photometric_override(tmp_path):
    """An ``extra_tags`` TAG_PHOTOMETRIC override that disagrees with
    the kwarg-resolved photometric must be refused."""
    from xrspatial.geotiff._dtypes import SHORT
    from xrspatial.geotiff._header import TAG_PHOTOMETRIC as _TAG_PHOTOMETRIC
    from xrspatial.geotiff._writer import write as _eager_write
    arr = np.array([[10, 20]], dtype=np.uint8)
    path = str(tmp_path / "extras_msw.tif")
    with pytest.raises(ValueError, match='TAG_PHOTOMETRIC'):
        _eager_write(
            arr, path,
            photometric='auto',
            extra_tags=[(_TAG_PHOTOMETRIC, SHORT, 1, 0)],
        )


# ---------------------------------------------------------------------------
# 1.3 -- signed-MinIsWhite rejection on read and write
#
# Pre-fix the reader and writer silently passed signed-int pixels
# through unchanged for ``Photometric=0``, so the file round-tripped
# inside xrspatial but disagreed with every other TIFF consumer. Both
# paths now raise ``NotImplementedError`` with a message that lists
# SampleFormat / BitsPerSample / Photometric.
#
# Preserve the exact ``NotImplementedError`` class and the message
# substring assertions; the issue-number string in the error message
# is the writer's diagnostic, not a test-side marker, so it stays.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected_bps",
    [
        pytest.param(np.int8, 8, id="signed_miniswhite_rejected[int8]"),
        pytest.param(np.int16, 16, id="signed_miniswhite_rejected[int16]"),
        pytest.param(np.int32, 32, id="signed_miniswhite_rejected[int32]"),
        pytest.param(np.int64, 64, id="signed_miniswhite_rejected[int64]"),
    ],
)
def test_write_signed_miniswhite_rejected(tmp_path, dtype, expected_bps):
    """Every signed dtype is refused at write time with a diagnostic
    message that lists SampleFormat, BitsPerSample, and Photometric."""
    info = np.iinfo(dtype)
    arr = np.array([[info.min, -1, 0, 1, info.max]], dtype=dtype)
    path = tmp_path / f"i{expected_bps}_msw.tif"
    with pytest.raises(NotImplementedError) as excinfo:
        to_geotiff(_da(arr), str(path), photometric='miniswhite')
    msg = str(excinfo.value)
    assert '2278' in msg, msg
    assert 'MinIsWhite' in msg, msg
    assert 'SampleFormat=2' in msg, msg
    assert f'BitsPerSample={expected_bps}' in msg, msg
    assert str(np.dtype(dtype)) in msg, msg


def test_write_signed_miniswhite_no_partial_file(tmp_path):
    """The rejection fires before any bytes are written."""
    arr = np.array([[-5, -1, 0, 1, 5]], dtype=np.int16)
    path = tmp_path / "i16_msw_no_partial.tif"
    with pytest.raises(NotImplementedError):
        to_geotiff(_da(arr), str(path), photometric='miniswhite')
    assert not path.exists(), (
        f"writer should not leave a partial file on a pre-write "
        f"rejection, but {path} exists"
    )


def test_write_signed_miniswhite_int16_error_message_shape(tmp_path):
    """Pin the same diagnostic shape used elsewhere in xrspatial:
    SampleFormat, BitsPerSample, dtype string, plus the issue marker
    that ships inside the error message."""
    arr = np.array([[-5, -1, 0, 1, 5]], dtype=np.int16)
    path = tmp_path / "i16_msw_msg.tif"
    with pytest.raises(NotImplementedError) as excinfo:
        to_geotiff(_da(arr), str(path), photometric='miniswhite')
    msg = str(excinfo.value)
    assert '2278' in msg
    assert 'MinIsWhite' in msg
    assert 'SampleFormat' in msg
    assert 'BitsPerSample' in msg
    assert 'int16' in msg


def test_read_signed_miniswhite_rejected(tmp_path):
    """Reading a forged signed-int MinIsWhite TIFF raises with the
    SampleFormat / BitsPerSample / Photometric values in the message
    instead of returning silently-wrong pixels."""
    path = _forge_signed_miniswhite_tif(tmp_path, "i16_msw_read.tif")
    with pytest.raises(NotImplementedError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert '2278' in msg, msg
    assert 'MinIsWhite' in msg, msg
    assert 'SampleFormat=2' in msg, msg
    assert 'BitsPerSample=16' in msg, msg
    assert 'int16' in msg, msg


def test_read_signed_miniswhite_rejected_on_gpu_path(tmp_path):
    """GPU decode path mirrors the CPU rejection so the backends do
    not diverge silently on a forged file."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device available")
    except Exception as exc:  # pragma: no cover - CI without CUDA
        pytest.skip(f"cupy importable but CUDA unusable: {exc}")
    path = _forge_signed_miniswhite_tif(tmp_path, "i16_msw_gpu.tif")
    with pytest.raises(NotImplementedError) as excinfo:
        open_geotiff(path, gpu=True)
    msg = str(excinfo.value)
    assert '2278' in msg, msg
    assert 'MinIsWhite' in msg, msg


@pytest.mark.parametrize(
    "arr_factory,assert_kind",
    [
        pytest.param(
            lambda: np.array([[0, 1, 127, 254, 255]], dtype=np.uint8),
            "equal",
            id="signed_miniswhite_nonregression[uint8]",
        ),
        pytest.param(
            lambda: np.array([[-3.5, 0.0, 0.25, 7.5]], dtype=np.float32),
            "allclose",
            id="signed_miniswhite_nonregression[float32]",
        ),
    ],
)
def test_unsigned_and_float_miniswhite_still_round_trip(
        tmp_path, arr_factory, assert_kind):
    """The unsigned and float paths are unaffected by the signed
    rejection; pin the non-regression here so the rejection branch
    cannot accidentally widen."""
    arr = arr_factory()
    path = tmp_path / "nonreg.tif"
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    out = np.asarray(open_geotiff(str(path)).values)
    if assert_kind == "equal":
        np.testing.assert_array_equal(out, arr)
    else:
        np.testing.assert_allclose(out, arr)


# ===========================================================================
# Section 2: ``allow_internal_only_jpeg`` API tier acceptance
#
# Boundary note: this section shares a home with the MinIsWhite tests
# above because both exercise the photometric / codec-tier API surface
# (the advertised tag must match the encode contract), *not* because
# they share runtime logic. The MinIsWhite gate lives in the writer's
# photometric branch; the JPEG opt-in lives in the dispatcher's
# compression branch. Treat them as siblings, not cousins. Do not
# pull helpers across the section boundary.
# ===========================================================================


def _make_rgb_uint8_da() -> xr.DataArray:
    """64x64x3 uint8 RGB raster suitable for the JPEG encode path."""
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return xr.DataArray(
        arr,
        dims=("y", "x", "band"),
        coords={
            "y": np.arange(64, dtype=np.float64),
            "x": np.arange(64, dtype=np.float64),
            "band": np.array([1, 2, 3], dtype=np.int32),
        },
    )


def test_to_geotiff_signature_has_allow_internal_only_jpeg():
    """``to_geotiff`` must expose ``allow_internal_only_jpeg`` so callers
    on the auto-dispatch path can reach the same opt-in
    ``_write_geotiff_gpu`` exposes."""
    params = inspect.signature(to_geotiff).parameters
    assert "allow_internal_only_jpeg" in params, (
        "to_geotiff must accept allow_internal_only_jpeg for parity "
        "with _write_geotiff_gpu."
    )
    assert params["allow_internal_only_jpeg"].default is False


def test_to_geotiff_and_gpu_writer_share_kwarg_default():
    """Both writers use the same default so a caller forwarding the
    kwarg from one to the other sees identical behaviour."""
    eager_default = inspect.signature(
        to_geotiff).parameters["allow_internal_only_jpeg"].default
    gpu_default = inspect.signature(
        _write_geotiff_gpu).parameters["allow_internal_only_jpeg"].default
    assert eager_default == gpu_default == False  # noqa: E712


@pytest.mark.parametrize(
    "expect_match,assertion_target",
    [
        pytest.param("JPEGTables", "raises_value_error",
                     id="allow_internal_only_jpeg[default-rejects]"),
        pytest.param("allow_internal_only_jpeg", "message_mentions_flag",
                     id="allow_internal_only_jpeg[error-mentions-flag]"),
    ],
)
def test_to_geotiff_rejects_jpeg_without_opt_in(
        tmp_path, expect_match, assertion_target):
    """Without the opt-in, ``compression='jpeg'`` raises ``ValueError``,
    and the message points at the opt-in flag plus its issue marker so
    callers can find the escape hatch."""
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "rejected.tif")
    if assertion_target == "raises_value_error":
        with pytest.raises(ValueError, match=expect_match):
            to_geotiff(da, path, compression='jpeg')
    else:
        with pytest.raises(ValueError) as exc:
            to_geotiff(da, path, compression='jpeg')
        msg = str(exc.value)
        assert expect_match in msg
        assert "1845" in msg


def test_to_geotiff_jpeg_opt_in_emits_warning_and_writes(tmp_path):
    """Setting the opt-in lets the write proceed and emits
    ``GeoTIFFFallbackWarning`` so callers know the file may not
    round-trip through external readers. xrspatial's own reader still
    decodes it via the internal JFIF path."""
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "opt_in.tif")
    with pytest.warns(GeoTIFFFallbackWarning, match="JPEGTables"):
        to_geotiff(
            da, path,
            compression='jpeg',
            allow_internal_only_jpeg=True,
        )
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    decoded = open_geotiff(path, allow_internal_only_jpeg=True)
    assert decoded.shape == da.shape


def test_to_geotiff_non_jpeg_unaffected_by_flag(tmp_path):
    """The flag must stay JPEG-specific so callers defensively passing
    it to either writer do not pay a cost on other codecs."""
    import warnings as _warnings

    da = _make_rgb_uint8_da()
    path = str(tmp_path / "non_jpeg_flag.tif")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", GeoTIFFFallbackWarning)
        to_geotiff(
            da, path,
            compression='zstd',
            allow_internal_only_jpeg=True,
        )
    assert os.path.exists(path)


def test_to_geotiff_gpu_dispatch_forwards_allow_internal_only_jpeg(
        tmp_path, monkeypatch):
    """``to_geotiff(gpu=True, allow_internal_only_jpeg=True)`` forwards
    the kwarg into ``_write_geotiff_gpu``. Without this the GPU writer
    would refuse the encode after the CPU dispatcher accepted it."""
    captured: dict = {}

    def _fake_write_geotiff_gpu(data, path, **kwargs):
        captured['data'] = data
        captured['path'] = path
        captured['kwargs'] = kwargs
        with open(path, 'wb') as f:
            f.write(b'')

    monkeypatch.setattr(
        'xrspatial.geotiff._writers.eager._write_geotiff_gpu',
        _fake_write_geotiff_gpu,
    )

    da = _make_rgb_uint8_da()
    path = str(tmp_path / "gpu_forward.tif")
    to_geotiff(
        da, path,
        gpu=True,
        compression='jpeg',
        allow_internal_only_jpeg=True,
    )

    assert 'kwargs' in captured, (
        "to_geotiff must dispatch to _write_geotiff_gpu when gpu=True"
    )
    assert captured['kwargs'].get('allow_internal_only_jpeg') is True, (
        "to_geotiff must forward allow_internal_only_jpeg unchanged "
        "into _write_geotiff_gpu."
    )
    assert captured['kwargs'].get('compression') == 'jpeg'


def test_to_geotiff_gpu_dispatch_emits_single_jpeg_opt_in_warning(
        tmp_path, monkeypatch, recwarn):
    """When the GPU writer handles the encode it emits the opt-in
    warning. The CPU dispatcher must not emit its own, otherwise the
    same opt-in disclosure surfaces twice for one encode."""
    import warnings as _warnings

    def _fake_write_geotiff_gpu(data, path, **kwargs):
        if kwargs.get('compression') == 'jpeg' and kwargs.get(
                'allow_internal_only_jpeg'):
            _warnings.warn(
                "_write_geotiff_gpu jpeg opt-in (stub).",
                GeoTIFFFallbackWarning,
                stacklevel=2,
            )
        with open(path, 'wb') as f:
            f.write(b'')

    monkeypatch.setattr(
        'xrspatial.geotiff._writers.eager._write_geotiff_gpu',
        _fake_write_geotiff_gpu,
    )

    da = _make_rgb_uint8_da()
    path = str(tmp_path / "gpu_single_warn.tif")
    to_geotiff(
        da, path,
        gpu=True,
        compression='jpeg',
        allow_internal_only_jpeg=True,
    )

    jpeg_warns = [w for w in recwarn.list
                  if issubclass(w.category, GeoTIFFFallbackWarning)]
    assert len(jpeg_warns) == 1, (
        "to_geotiff(gpu=True, compression='jpeg', "
        "allow_internal_only_jpeg=True) must emit exactly one "
        "GeoTIFFFallbackWarning; got "
        f"{[str(w.message) for w in jpeg_warns]}"
    )
