"""Issue #2278: signed-integer MinIsWhite is rejected on both paths.

Before #2278 the reader (``_decode._apply_photometric_miniswhite``) and
writer (``_encode._apply_photometric_miniswhite_invert``) silently
passed signed-int pixels through unchanged for ``Photometric=0``. The
result round-tripped inside xrspatial but produced files whose pixel
values disagreed with the on-disk ``Photometric`` tag against every
other TIFF consumer (GDAL, libtiff, ImageMagick).

Both paths now raise ``NotImplementedError`` with a message that lists
SampleFormat / BitsPerSample / Photometric so callers can diagnose the
rejection without digging into the TIFF spec. The existing unsigned and
float MinIsWhite paths are left intact and still round-trip.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._header import (
    TAG_PHOTOMETRIC,
    TAG_SAMPLE_FORMAT,
    parse_header,
)


def _da(arr: np.ndarray, attrs_extra=None) -> xr.DataArray:
    """Wrap *arr* as an ``open_geotiff``-compatible DataArray.

    Square 1xN strips are common in the existing miniswhite tests so we
    keep the same degenerate-axis opt-in as
    ``test_miniswhite_writer_roundtrip_1836.py`` (issue #2214).
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

    Both ``TAG_PHOTOMETRIC`` (262) and ``TAG_SAMPLE_FORMAT`` (339) are
    written as a single SHORT (type 3, count 1) by the xrspatial writer,
    so the on-disk value lives in the IFD entry's first two bytes of the
    8-byte value slot. This helper avoids the type/length checks in
    ``_tiff_surgery.patch_byte_counts`` for that simpler case.
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


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype,expected_bps",
    [(np.int8, 8), (np.int16, 16), (np.int32, 32), (np.int64, 64)],
)
def test_write_signed_miniswhite_rejected_2278(tmp_path, dtype, expected_bps):
    """Every signed dtype is refused at write time with a diagnostic
    message that lists SampleFormat, BitsPerSample, and Photometric."""
    info = np.iinfo(dtype)
    arr = np.array([[info.min, -1, 0, 1, info.max]], dtype=dtype)
    path = tmp_path / f'i{expected_bps}_msw_2278_{tmp_path.name}.tif'
    with pytest.raises(NotImplementedError) as excinfo:
        to_geotiff(_da(arr), str(path), photometric='miniswhite')
    msg = str(excinfo.value)
    assert '2278' in msg, msg
    assert 'MinIsWhite' in msg, msg
    assert 'SampleFormat=2' in msg, msg
    assert f'BitsPerSample={expected_bps}' in msg, msg
    assert str(np.dtype(dtype)) in msg, msg


def test_write_signed_miniswhite_does_not_partial_write_2278(tmp_path):
    """The rejection fires before any bytes are written, so a failed
    write does not leave a half-finished file on disk."""
    arr = np.array([[-5, -1, 0, 1, 5]], dtype=np.int16)
    path = tmp_path / 'i16_msw_2278_no_partial.tif'
    with pytest.raises(NotImplementedError):
        to_geotiff(_da(arr), str(path), photometric='miniswhite')
    assert not path.exists(), (
        f"writer should not leave a partial file on a pre-write "
        f"rejection, but {path} exists"
    )


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

def _forge_signed_miniswhite_tif(tmp_path, name: str) -> str:
    """Write a normal int16 MinIsBlack file then flip its Photometric
    tag from 1 (MinIsBlack) to 0 (MinIsWhite) in place. The resulting
    file is exactly what the issue describes: SampleFormat=2 (signed),
    Photometric=0 (MinIsWhite), which xrspatial used to read back
    silently as the un-inverted MinIsBlack pixels."""
    arr = np.array([[-5, -1, 0, 1, 5]], dtype=np.int16)
    path = tmp_path / name
    # Default photometric='auto' is MinIsBlack (1) for single-band data,
    # so the writer accepts it and emits a normal signed-int TIFF.
    to_geotiff(_da(arr), str(path))
    raw = bytearray(path.read_bytes())
    _patch_tag_value(raw, TAG_PHOTOMETRIC, 0)
    # SampleFormat is already 2 for int16, but assert via the patch
    # helper so the test would fail loud if the writer ever changes.
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


def test_read_signed_miniswhite_rejected_2278(tmp_path):
    """Reading a forged signed-int MinIsWhite TIFF now raises with the
    SampleFormat / BitsPerSample / Photometric values in the message
    instead of returning silently-wrong pixels."""
    path = _forge_signed_miniswhite_tif(tmp_path, 'i16_msw_read_2278.tif')
    with pytest.raises(NotImplementedError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert '2278' in msg, msg
    assert 'MinIsWhite' in msg, msg
    assert 'SampleFormat=2' in msg, msg
    assert 'BitsPerSample=16' in msg, msg
    assert 'int16' in msg, msg


# ---------------------------------------------------------------------------
# Non-regression: unsigned and float MinIsWhite still round-trip
# ---------------------------------------------------------------------------

def test_unsigned_miniswhite_still_round_trips_2278(tmp_path):
    """The unsigned path is unaffected by the signed rejection."""
    arr = np.array([[0, 1, 127, 254, 255]], dtype=np.uint8)
    path = tmp_path / 'u8_msw_nonreg_2278.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    out = np.asarray(open_geotiff(str(path)).values)
    np.testing.assert_array_equal(out, arr)


def test_float_miniswhite_still_round_trips_2278(tmp_path):
    """The float path is unaffected by the signed rejection."""
    arr = np.array([[-3.5, 0.0, 0.25, 7.5]], dtype=np.float32)
    path = tmp_path / 'f32_msw_nonreg_2278.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    out = np.asarray(open_geotiff(str(path)).values)
    np.testing.assert_allclose(out, arr)
