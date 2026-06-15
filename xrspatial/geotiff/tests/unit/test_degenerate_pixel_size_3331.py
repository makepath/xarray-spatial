"""Reject a zero or non-finite ModelPixelScale on read (issue #3331).

The direct-TIFF read path built coordinate arrays as
``arange(N) * pixel_width + origin`` straight from the
``ModelPixelScale`` / ``ModelTransformation`` diagonal, with no
finite-nonzero check. A zero pixel size collapsed the whole axis onto
the origin; a NaN / Inf one filled it with NaN / Inf. Either way the
read returned a degenerate raster with no error.

The VRT read path already rejected a zero ``res_x`` / ``res_y``
(``VRTUnsupportedError``) and the writer rejected a zero-step coord axis
(``NonUniformCoordsError``). This test pins the matching rejection on the
direct-TIFF read path: ``DegeneratePixelSizeError``.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff._errors import DegeneratePixelSizeError, GeoTIFFAmbiguousMetadataError
from xrspatial.geotiff._geotags import _extract_transform
from xrspatial.geotiff._header import parse_all_ifds, parse_header

_BO = '<'


def _assemble_tiff(extra_geo_tags: list[tuple]) -> bytes:
    """Build a 2x2 single-strip TIFF with the given geo tags.

    ``extra_geo_tags`` is a list of ``(tag, doubles_tuple)`` entries
    appended as TIFF DOUBLE (type 12) arrays.
    """
    width, height = 2, 2
    pixels = np.zeros((height, width), dtype=np.uint8)

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{_BO}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{_BO}I', val)))

    def add_doubles(tag, vals):
        tag_list.append(
            (tag, 12, len(vals), struct.pack(f'{_BO}{len(vals)}d', *vals)))

    add_short(256, width)            # ImageWidth
    add_short(257, height)           # ImageLength
    add_short(258, 8)                # BitsPerSample
    add_short(259, 1)                # Compression: none
    add_short(262, 1)                # PhotometricInterpretation
    add_short(277, 1)                # SamplesPerPixel
    add_short(278, height)           # RowsPerStrip
    add_long(273, 0)                 # StripOffsets (placeholder)
    add_long(279, len(pixels.tobytes()))  # StripByteCounts
    add_short(339, 1)                # SampleFormat
    for tag, vals in extra_geo_tags:
        add_doubles(tag, list(vals))

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4

    overflow = bytearray()
    overflow_offsets = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_offsets[tag] = ifd_start + ifd_size + len(overflow)
            overflow.extend(raw)
            if len(overflow) % 2:
                overflow.append(0)

    pixel_start = ifd_start + ifd_size + len(overflow)

    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{_BO}I', pixel_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{_BO}H', 42))
    out.extend(struct.pack(f'{_BO}I', ifd_start))
    out.extend(struct.pack(f'{_BO}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{_BO}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            out.extend(struct.pack(f'{_BO}I', overflow_offsets[tag]))
    out.extend(struct.pack(f'{_BO}I', 0))
    out.extend(overflow)
    out.extend(pixels.tobytes())
    return bytes(out)


def _tiff_with_pixel_scale(sx: float, sy: float, with_tiepoint: bool) -> bytes:
    """TIFF with ModelPixelScale (33550), optionally + ModelTiepoint (33922)."""
    tags = [(33550, (sx, sy, 0.0))]
    if with_tiepoint:
        tags.append((33922, (0.0, 0.0, 0.0, 500000.0, 4500000.0, 0.0)))
    return _assemble_tiff(tags)


def _tiff_with_transformation(sx: float, sy: float) -> bytes:
    """TIFF with an axis-aligned ModelTransformation (34264) diagonal."""
    matrix = (
        sx, 0.0, 0.0, 500000.0,
        0.0, sy, 0.0, 4500000.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    return _assemble_tiff([(34264, matrix)])


def _extract(data: bytes):
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    return _extract_transform(ifds[0])


_BAD = [
    pytest.param(0.0, 30.0, id="zero_width"),
    pytest.param(30.0, 0.0, id="zero_height"),
    pytest.param(float('nan'), 30.0, id="nan_width"),
    pytest.param(30.0, float('nan'), id="nan_height"),
    pytest.param(float('inf'), 30.0, id="inf_width"),
    pytest.param(30.0, float('-inf'), id="neg_inf_height"),
]


@pytest.mark.parametrize("sx,sy", _BAD)
def test_pixel_scale_only_rejected(sx, sy):
    with pytest.raises(DegeneratePixelSizeError) as exc:
        _extract(_tiff_with_pixel_scale(sx, sy, with_tiepoint=False))
    assert 'ModelPixelScaleTag' in str(exc.value)


@pytest.mark.parametrize("sx,sy", _BAD)
def test_pixel_scale_with_tiepoint_rejected(sx, sy):
    with pytest.raises(DegeneratePixelSizeError) as exc:
        _extract(_tiff_with_pixel_scale(sx, sy, with_tiepoint=True))
    assert 'ModelPixelScaleTag' in str(exc.value)


@pytest.mark.parametrize("sx,sy", _BAD)
def test_transformation_diagonal_rejected(sx, sy):
    with pytest.raises(DegeneratePixelSizeError) as exc:
        _extract(_tiff_with_transformation(sx, sy))
    assert 'ModelTransformationTag' in str(exc.value)


def test_degenerate_pixel_size_is_ambiguous_metadata_subclass():
    # Existing ``except ValueError`` / ``except
    # GeoTIFFAmbiguousMetadataError`` callers must keep catching this.
    assert issubclass(DegeneratePixelSizeError, GeoTIFFAmbiguousMetadataError)
    assert issubclass(DegeneratePixelSizeError, ValueError)


@pytest.mark.parametrize("builder", [
    lambda: _tiff_with_pixel_scale(30.0, 30.0, with_tiepoint=False),
    lambda: _tiff_with_pixel_scale(30.0, 30.0, with_tiepoint=True),
    lambda: _tiff_with_transformation(30.0, -30.0),
])
def test_finite_nonzero_pixel_size_still_accepted(builder):
    transform, has_georef = _extract(builder())
    assert has_georef is True
    assert transform.pixel_width == pytest.approx(30.0)
    assert abs(transform.pixel_height) == pytest.approx(30.0)


def test_open_geotiff_rejects_zero_pixel_scale(tmp_path):
    # End-to-end through the public read entry point.
    path = tmp_path / 'degenerate_scale_3331.tif'
    path.write_bytes(_tiff_with_pixel_scale(0.0, 30.0, with_tiepoint=True))
    from xrspatial.geotiff import open_geotiff
    with pytest.raises(DegeneratePixelSizeError):
        open_geotiff(str(path))
