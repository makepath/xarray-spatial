"""Shared GeoTIFF fixture builder for the test suite.

Consolidates the hand-built TIFF helpers that were duplicated across
test modules, exposing a single parameterised ``write_minimal_tiff``
that supports GeoKey directory and/or GeoAsciiParams payloads.

Migrated from:
  - ``unit/test_geotags.py::_write_tiff_with_geokeys``  (#2417)
  - ``unit/test_metadata.py::_write_minimal_tiff_with_wkt`` (#1987)
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_minimal_tiff(
    path: str,
    *,
    geokeys: list[int] | None = None,
    geo_ascii: str | None = None,
) -> None:
    """Build a 4x4 float32 TIFF with optional GeoKey directory and/or
    GeoAsciiParams tags, then write it to ``path``.

    Parameters
    ----------
    path : str
        Destination file path.
    geokeys : list of int, optional
        SHORT values for the GeoKeyDirectory tag (34735). If given, the
        tag is emitted directly; no automatic header is prepended.
    geo_ascii : str, optional
        ASCII string for the GeoAsciiParams tag (34737). A trailing ``|``
        separator and NULL terminator are added automatically.
    """
    bo = '<'
    pixels = np.zeros((4, 4), dtype=np.float32).tobytes()

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals),
                         struct.pack(f'{bo}{len(vals)}H', *vals)))

    def add_doubles(tag, vals):
        tag_list.append((tag, 12, len(vals),
                         struct.pack(f'{bo}{len(vals)}d', *vals)))

    def add_ascii(tag, raw_bytes):
        if not raw_bytes.endswith(b'\x00'):
            raw_bytes = raw_bytes + b'\x00'
        tag_list.append((tag, 2, len(raw_bytes), raw_bytes))

    add_short(256, 4)
    add_short(257, 4)
    add_short(258, 32)
    add_short(259, 1)
    add_short(262, 1)
    add_short(277, 1)
    add_short(339, 3)
    add_short(278, 4)
    add_long(273, 0)
    add_long(279, len(pixels))
    add_doubles(33550, [1.0, 1.0, 0.0])
    add_doubles(33922, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    if geokeys is not None:
        add_shorts(34735, geokeys)
    if geo_ascii is not None:
        ascii_buf = bytearray((geo_ascii + '|').encode('ascii'))
        add_ascii(34737, bytes(ascii_buf))

    tag_list.sort(key=lambda t: t[0])

    n = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * n + 4
    overflow_start = ifd_start + ifd_size
    overflow_buf = bytearray()
    tag_offsets: dict[int, int | None] = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None
    pixel_data_start = overflow_start + len(overflow_buf)
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            raw = struct.pack(f'{bo}I', pixel_data_start)
        patched.append((tag, typ, count, raw))
    tag_list = patched
    out = bytearray(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', n))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            out.extend(struct.pack(f'{bo}I', overflow_start + tag_offsets[tag]))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixels)
    Path(path).write_bytes(bytes(out))
