"""DeprecationWarning tests for the vertical-CRS GeoKey attrs (PR7 of #1984).

The writer in ``xrspatial.geotiff._geotags.build_geo_tags`` does not emit
the vertical GeoKey block, so ``attrs['vertical_crs']`` /
``attrs['vertical_citation']`` / ``attrs['vertical_units']`` set on a
DataArray are silently dropped on round-trip. The PR6 pass-through
locking test (``test_attrs_contract_passthrough_1984.py``) pins that
behaviour as "dropped".

PR7 deprecates these three read-side emissions for one release cycle
before removal. Each test here builds a minimal TIFF carrying the
relevant vertical GeoKey and asserts that ``open_geotiff`` raises a
``DeprecationWarning`` whose message points at issue #1984. A final
test confirms the attrs still appear on the returned DataArray during
the deprecation window so existing consumers keep working.
"""
from __future__ import annotations

import struct
import warnings

import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff.tests.conftest import make_minimal_tiff


# GeoKey IDs from xrspatial.geotiff._geotags
_GEOKEY_VERTICAL_CS_TYPE = 4096
_GEOKEY_VERTICAL_CITATION = 4097
_GEOKEY_VERTICAL_UNITS = 4099

# TIFF tag IDs
_TAG_GEO_KEY_DIRECTORY = 34735
_TAG_GEO_ASCII_PARAMS = 34737


def _build_tiff_with_vertical_geokeys(
    *,
    vertical_epsg: int | None = None,
    vertical_citation: str | None = None,
    vertical_units_code: int | None = None,
) -> bytes:
    """Build a minimal float32 TIFF with horizontal EPSG 4326 and optional
    vertical GeoKey entries.

    Starts from ``make_minimal_tiff(epsg=4326)`` and rewrites the
    GeoKeyDirectory tag (34735) to append the requested vertical block.
    The format is the standard GeoTIFF GeoKey directory:
    ``(version, revision, minor, n_keys)`` header followed by ``n_keys``
    quadruples of ``(key_id, location, count, value_or_offset)``. A
    ``location`` of 0 means the value is inline; ``location == 34737``
    points into GeoAsciiParams.
    """
    # Start with the minimal 4326 TIFF.
    base = make_minimal_tiff(
        4, 4,
        geo_transform=(-120.0, 45.0, 0.001, -0.001),
        epsg=4326,
    )

    # Find the GeoKeyDirectory tag (34735) in the IFD and replace it.
    # The IFD lives at offset 8 in the byte stream produced by
    # ``make_minimal_tiff``. We rebuild a fresh TIFF with the extra
    # vertical entries appended to the GeoKey directory instead of
    # surgically editing ``base``; this keeps the byte layout valid.
    # The simplest approach: build a new TIFF using a small ad-hoc
    # serializer that mirrors ``make_minimal_tiff`` but takes a
    # pre-built GeoKey directory and optional GeoAsciiParams payload.

    width = height = 4
    import numpy as np

    pixel_data = np.zeros((height, width), dtype=np.float32)
    pixel_bytes = pixel_data.tobytes()

    # Build the GeoKey directory: start with the minimal 4326 keys
    # (model_type=2 geographic + GEOKEY_GEOGRAPHIC_CS_TYPE=2048 -> 4326),
    # then append any requested vertical entries.
    # Header: version=1, revision=1, minor=0, n_keys (filled later).
    gkd_entries: list[int] = [
        1024, 0, 1, 2,          # GTModelType -> 2 (geographic)
        2048, 0, 1, 4326,       # GeographicTypeGeoKey -> 4326
    ]

    geo_ascii_buf = bytearray()

    if vertical_epsg is not None:
        gkd_entries.extend([
            _GEOKEY_VERTICAL_CS_TYPE, 0, 1, int(vertical_epsg),
        ])

    if vertical_citation is not None:
        s = vertical_citation + '|'  # GeoTIFF citation strings end with '|'
        ascii_offset = len(geo_ascii_buf)
        geo_ascii_buf.extend(s.encode('ascii'))
        gkd_entries.extend([
            _GEOKEY_VERTICAL_CITATION,
            _TAG_GEO_ASCII_PARAMS,
            len(s),
            ascii_offset,
        ])

    if vertical_units_code is not None:
        gkd_entries.extend([
            _GEOKEY_VERTICAL_UNITS, 0, 1, int(vertical_units_code),
        ])

    n_keys = len(gkd_entries) // 4
    # Prepend header
    gkd = [1, 1, 0, n_keys] + gkd_entries

    # Now serialize a minimal TIFF carrying the standard image tags and
    # this enriched GeoKey directory plus a GeoAsciiParams tag when
    # populated. Reuses the layout strategy of ``make_minimal_tiff``.
    bo = '<'

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals)))

    def add_doubles(tag, vals):
        tag_list.append((tag, 12, len(vals), struct.pack(f'{bo}{len(vals)}d', *vals)))

    def add_ascii(tag, raw_bytes: bytes):
        # ASCII type, count includes the terminating NUL.
        if not raw_bytes.endswith(b'\x00'):
            raw_bytes = raw_bytes + b'\x00'
        tag_list.append((tag, 2, len(raw_bytes), raw_bytes))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 32)            # BitsPerSample
    add_short(259, 1)             # Compression none
    add_short(262, 1)             # Photometric BlackIsZero
    add_short(277, 1)             # SamplesPerPixel
    add_short(339, 3)             # SampleFormat IEEE float
    add_short(278, height)        # RowsPerStrip
    add_long(273, 0)              # StripOffsets placeholder
    add_long(279, len(pixel_bytes))  # StripByteCounts
    add_doubles(33550, [0.001, 0.001, 0.0])
    add_doubles(33922, [0.0, 0.0, 0.0, -120.0, 45.0, 0.0])
    add_shorts(_TAG_GEO_KEY_DIRECTORY, gkd)
    if geo_ascii_buf:
        add_ascii(_TAG_GEO_ASCII_PARAMS, bytes(geo_ascii_buf))

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
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

    # Patch StripOffsets now that we know where pixel data lives.
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    # Silence the unused-import lint for ``base`` -- we deliberately
    # rebuild rather than patch.
    _ = base
    return bytes(out)


def _write(tmp_path, blob: bytes, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(blob)
    return str(path)


def test_vertical_crs_emits_deprecation_warning(tmp_path):
    """Reading a TIFF with VerticalCSTypeGeoKey warns about
    ``attrs['vertical_crs']`` deprecation."""
    blob = _build_tiff_with_vertical_geokeys(vertical_epsg=5703)
    path = _write(tmp_path, blob, 'vertical_crs.tif')

    with pytest.warns(DeprecationWarning, match=r"vertical_crs.*#1984"):
        open_geotiff(path)


def test_vertical_citation_emits_deprecation_warning(tmp_path):
    """Reading a TIFF with VerticalCitationGeoKey warns about
    ``attrs['vertical_citation']`` deprecation."""
    blob = _build_tiff_with_vertical_geokeys(vertical_citation='NAVD88')
    path = _write(tmp_path, blob, 'vertical_citation.tif')

    with pytest.warns(DeprecationWarning, match=r"vertical_citation.*#1984"):
        open_geotiff(path)


def test_vertical_units_emits_deprecation_warning(tmp_path):
    """Reading a TIFF with VerticalUnitsGeoKey warns about
    ``attrs['vertical_units']`` deprecation."""
    # 9001 is the GeoTIFF code for linear metre, mapped to 'metre' on read.
    blob = _build_tiff_with_vertical_geokeys(vertical_units_code=9001)
    path = _write(tmp_path, blob, 'vertical_units.tif')

    with pytest.warns(DeprecationWarning, match=r"vertical_units.*#1984"):
        open_geotiff(path)


def test_vertical_attrs_still_emit_during_deprecation(tmp_path):
    """Deprecation does not yet remove the attrs: a read of a TIFF with
    all three vertical GeoKeys still surfaces all three keys on
    ``DataArray.attrs``."""
    blob = _build_tiff_with_vertical_geokeys(
        vertical_epsg=5703,
        vertical_citation='NAVD88',
        vertical_units_code=9001,
    )
    path = _write(tmp_path, blob, 'vertical_all.tif')

    with warnings.catch_warnings():
        # The whole point of this test is that the attrs still emit;
        # the per-attr warning is asserted in the three tests above, so
        # silence it here to keep this test focused on the attr values.
        warnings.simplefilter('ignore', DeprecationWarning)
        da = open_geotiff(path)

    assert da.attrs.get('vertical_crs') == 5703
    assert da.attrs.get('vertical_citation') == 'NAVD88'
    assert da.attrs.get('vertical_units') == 'metre'
