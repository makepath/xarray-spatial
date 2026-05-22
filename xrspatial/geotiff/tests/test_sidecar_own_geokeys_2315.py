"""Sidecar IFDs that declare their own georef payload (issue #2315).

The local eager reader and the metadata-only path both swap pixel bytes
over to the sidecar when the selected IFD lives there, but pre-#2315
they always read georef tags from the base file. That is fine under
the usual GDAL convention -- a ``.tif.ovr`` sidecar carries no
geokeys -- but a sidecar that does declare its own
GeoKeyDirectory / ModelPixelScale / ModelTiepoint /
ModelTransformation had its tags parsed against the wrong buffer.

These tests cover the two contracted behaviors:

* When the sidecar IFD declares its own georef payload, the sidecar's
  transform wins over the base file's.
* When the sidecar IFD has no georef tags (the GDAL convention),
  today's inheritance from the base file is preserved.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _read_geo_info, open_geotiff
from xrspatial.geotiff._geotags import _ifd_has_georef_payload
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._writers.eager import to_geotiff


def _make_dataarray(values: np.ndarray, *, origin_x: float, origin_y: float,
                    pixel_w: float, pixel_h: float,
                    epsg: int = 4326) -> xr.DataArray:
    """Build a 2D DataArray whose attrs survive ``to_geotiff``."""
    h, w = values.shape
    xs = origin_x + (np.arange(w) + 0.5) * pixel_w
    ys = origin_y + (np.arange(h) + 0.5) * pixel_h
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": ys, "x": xs})
    da.attrs["crs"] = epsg
    return da


def _write_pair(tmp_path, *, base_geo, sidecar_geo, sidecar_has_geokeys):
    """Write base.tif and (optionally geokey-bearing) sidecar.tif.ovr.

    ``base_geo`` and ``sidecar_geo`` are kwarg dicts forwarded to
    :func:`_make_dataarray`. When ``sidecar_has_geokeys`` is False the
    sidecar file is rewritten to strip its GeoKey / Model* tags so the
    file represents the "GDAL convention" (no geokeys on the sidecar
    IFD). The byte-level strip operates on the parsed IFD entries: we
    rewrite the IFD's entry count and entries array to exclude the
    four georef tag IDs.
    """
    base_arr = np.full((8, 8), 1, dtype=np.uint16)
    sidecar_arr = np.full((4, 4), 2, dtype=np.uint16)

    base_path = tmp_path / "base_2315.tif"
    side_path = tmp_path / "base_2315.tif.ovr"

    base_da = _make_dataarray(base_arr, **base_geo)
    to_geotiff(base_da, str(base_path), tiled=False, compression="none")

    side_da = _make_dataarray(sidecar_arr, **sidecar_geo)
    to_geotiff(side_da, str(side_path), tiled=False, compression="none")

    # Real GDAL ``.tif.ovr`` files mark every IFD as a reduced-res
    # overview (NewSubfileType bit 0 set). ``to_geotiff`` writes a
    # standalone full-res TIFF so the tag is absent; insert one so the
    # sidecar IFD is treated as an overview by the inheritance helper.
    _mark_first_ifd_as_overview(side_path)

    if not sidecar_has_geokeys:
        _strip_georef_tags(side_path)

    return base_path, side_path


def _mark_first_ifd_as_overview(path):
    """Set NewSubfileType=1 (reduced-resolution overview) on the first IFD.

    The tag is added or rewritten so the sidecar IFD looks like a real
    GDAL external overview rather than a freshly written full-res TIFF.
    Inline value, type LONG (3 = uint16 not enough -- the spec uses
    type 4 / uint32).
    """
    import struct

    raw = bytearray(path.read_bytes())
    assert raw[:2] == b"II", "test fixture assumes little-endian classic TIFF"
    first_ifd_offset = struct.unpack_from("<I", raw, 4)[0]
    n_entries = struct.unpack_from("<H", raw, first_ifd_offset)[0]

    # Look for an existing NewSubfileType (254). If present, rewrite to
    # value=1. Otherwise insert a new entry and shift the trailing
    # entries forward, keeping the entries sorted by tag id.
    NSF = 254
    for i in range(n_entries):
        entry_off = first_ifd_offset + 2 + i * 12
        tag = struct.unpack_from("<H", raw, entry_off)[0]
        if tag == NSF:
            struct.pack_into("<HHII", raw, entry_off, NSF, 4, 1, 1)
            path.write_bytes(bytes(raw))
            return

    # Build the new entry.
    new_entry = struct.pack("<HHII", NSF, 4, 1, 1)

    # The IFD entries are sorted by tag id. 254 sorts before every tag
    # ``to_geotiff`` emits in practice (the smallest the writer uses is
    # 256 / ImageWidth), so the new entry goes at position 0. Assert
    # the invariant so a future writer change that emits a tag <= 254
    # fails this fixture loudly instead of silently producing an
    # out-of-order IFD that the reader could later reject. (Review nit
    # on #2315.)
    if n_entries > 0:
        first_tag = struct.unpack_from("<H", raw, first_ifd_offset + 2)[0]
        assert first_tag > NSF, (
            f"test fixture invariant: first emitted tag must be > {NSF}, "
            f"got {first_tag}"
        )
    insert_pos = first_ifd_offset + 2

    next_ifd_off_pos = first_ifd_offset + 2 + n_entries * 12
    block_end = next_ifd_off_pos + 4

    # Slide every byte from ``insert_pos`` to end-of-file forward by 12
    # bytes. This makes room for the new entry and keeps value blocks
    # that lived past the entries array intact at their new offsets.
    tail = bytes(raw[insert_pos:])
    raw[insert_pos:insert_pos] = b"\x00" * 12  # carve a 12-byte hole
    # raw is now 12 bytes longer; the hole sits at insert_pos..insert_pos+12.
    # The original tail bytes are at insert_pos+12..end now.

    struct.pack_into("<H", raw, first_ifd_offset, n_entries + 1)
    raw[insert_pos:insert_pos + 12] = new_entry

    # Every absolute byte offset stored in an IFD entry that points
    # at or past ``insert_pos`` is off by -12 now. Bump them.
    _bump_offsets(raw, first_ifd_offset, n_entries + 1, insert_pos, 12)

    # Also bump the stripoffsets / tileoffsets if their value is inline
    # (count==1) but small enough to still point at a now-shifted
    # location. ``_bump_offsets`` already handles the offset-vs-inline
    # case correctly via type_size*count > 4. For count==1 LONG (typ=4,
    # size=4) the value IS the byte offset and total=4 is inline, so
    # the helper above skipped it. Fix it explicitly here.
    _bump_inline_long_offsets(raw, first_ifd_offset, n_entries + 1,
                              insert_pos, 12)

    path.write_bytes(bytes(raw))


def _bump_inline_long_offsets(raw, ifd_offset, n_entries, threshold, delta):
    """Bump inline LONG offsets stored in StripOffsets / TileOffsets.

    For these tags the *value* itself is a byte offset into the file.
    When count==1, the value sits inline in the entry's value field
    and ``_bump_offsets`` skips it (because total <= 4). Patch it here.
    """
    import struct

    OFFSET_TAGS = {273, 324}  # StripOffsets, TileOffsets
    for i in range(n_entries):
        entry_off = ifd_offset + 2 + i * 12
        tag = struct.unpack_from("<H", raw, entry_off)[0]
        if tag not in OFFSET_TAGS:
            continue
        type_id = struct.unpack_from("<H", raw, entry_off + 2)[0]
        count = struct.unpack_from("<I", raw, entry_off + 4)[0]
        if type_id == 4 and count == 1:
            val = struct.unpack_from("<I", raw, entry_off + 8)[0]
            if val >= threshold:
                struct.pack_into("<I", raw, entry_off + 8, val + delta)
        elif type_id == 3 and count == 1:
            val = struct.unpack_from("<H", raw, entry_off + 8)[0]
            if val >= threshold:
                struct.pack_into("<H", raw, entry_off + 8, val + delta)


def _bump_offsets(raw, ifd_offset, n_entries, threshold, delta):
    """Add ``delta`` to every byte offset stored in the IFD >= threshold.

    Type sizes per TIFF 6.0: BYTE/SBYTE/ASCII/UNDEFINED=1, SHORT/SSHORT=2,
    LONG/SLONG/FLOAT=4, RATIONAL/SRATIONAL/DOUBLE=8.
    """
    import struct

    TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
                 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    for i in range(n_entries):
        entry_off = ifd_offset + 2 + i * 12
        type_id = struct.unpack_from("<H", raw, entry_off + 2)[0]
        count = struct.unpack_from("<I", raw, entry_off + 4)[0]
        ts = TYPE_SIZE.get(type_id, 1)
        total = ts * count
        if total <= 4:
            continue  # inline value, no offset to bump
        off = struct.unpack_from("<I", raw, entry_off + 8)[0]
        if off >= threshold:
            struct.pack_into("<I", raw, entry_off + 8, off + delta)

    # Bump the next-IFD offset if needed.
    next_ifd_off_pos = ifd_offset + 2 + n_entries * 12
    next_off = struct.unpack_from("<I", raw, next_ifd_off_pos)[0]
    if next_off >= threshold and next_off != 0:
        struct.pack_into("<I", raw, next_ifd_off_pos, next_off + delta)


def _strip_georef_tags(path):
    """Rewrite ``path`` so its first IFD carries no georef tags.

    The four georef-bearing tag IDs (33550, 33922, 34264, 34735) are
    removed from the IFD entries array. The IFD's entry count field
    is decremented accordingly. The trailing value blocks the removed
    tags pointed at are left in place: bounds checking happens at the
    entry level, and any orphan bytes after the IFD are ignored by
    every IFD walker in this package.

    Little-endian classic TIFF assumed (matches what ``to_geotiff``
    writes by default).
    """
    import struct

    raw = bytearray(path.read_bytes())
    if raw[:2] != b"II":
        raise AssertionError(
            "test fixture: expected little-endian TIFF from to_geotiff()")
    assert struct.unpack_from("<H", raw, 2)[0] == 42, "classic TIFF expected"
    first_ifd_offset = struct.unpack_from("<I", raw, 4)[0]

    n_entries = struct.unpack_from("<H", raw, first_ifd_offset)[0]
    GEOREF_TAGS = {33550, 33922, 34264, 34735}
    keep = []
    for i in range(n_entries):
        entry_off = first_ifd_offset + 2 + i * 12
        tag = struct.unpack_from("<H", raw, entry_off)[0]
        if tag in GEOREF_TAGS:
            continue
        keep.append(raw[entry_off:entry_off + 12])

    # Rewrite entry count + entries.  The post-entries 4-byte
    # ``next-IFD-offset`` field needs to stay where the truncated
    # entries array ends.
    next_ifd_off_pos = first_ifd_offset + 2 + n_entries * 12
    next_ifd_off = raw[next_ifd_off_pos:next_ifd_off_pos + 4]

    new_n = len(keep)
    struct.pack_into("<H", raw, first_ifd_offset, new_n)
    write_pos = first_ifd_offset + 2
    for entry in keep:
        raw[write_pos:write_pos + 12] = entry
        write_pos += 12
    raw[write_pos:write_pos + 4] = next_ifd_off
    # Zero out the trailing bytes from the original entries array so a
    # downstream walker that scans past ``next_ifd_off`` does not pick
    # up stale entry data.
    tail_start = write_pos + 4
    tail_end = next_ifd_off_pos + 4
    if tail_end > tail_start:
        raw[tail_start:tail_end] = b"\x00" * (tail_end - tail_start)

    path.write_bytes(bytes(raw))


# ---------------------------------------------------------------------------
# Helper unit test: the georef-payload detector recognises the four tags.
# ---------------------------------------------------------------------------
def test_ifd_has_georef_payload_true_for_pixel_scale(tmp_path):
    p = tmp_path / "p_2315.tif"
    da = _make_dataarray(
        np.zeros((4, 4), dtype=np.uint16),
        origin_x=0.0, origin_y=0.0, pixel_w=1.0, pixel_h=-1.0,
    )
    to_geotiff(da, str(p), tiled=False, compression="none")
    data = p.read_bytes()
    hdr = parse_header(data)
    ifds = parse_all_ifds(data, hdr)
    assert _ifd_has_georef_payload(ifds[0])


def test_ifd_has_georef_payload_false_after_strip(tmp_path):
    p = tmp_path / "p_2315_strip.tif"
    da = _make_dataarray(
        np.zeros((4, 4), dtype=np.uint16),
        origin_x=0.0, origin_y=0.0, pixel_w=1.0, pixel_h=-1.0,
    )
    to_geotiff(da, str(p), tiled=False, compression="none")
    _strip_georef_tags(p)
    data = p.read_bytes()
    hdr = parse_header(data)
    ifds = parse_all_ifds(data, hdr)
    assert not _ifd_has_georef_payload(ifds[0])


# ---------------------------------------------------------------------------
# Sidecar carries its own geokeys -> the sidecar's georef wins.
# ---------------------------------------------------------------------------
def test_sidecar_with_own_geokeys_wins_eager(tmp_path):
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, side = _write_pair(tmp_path,
                             base_geo=base_geo, sidecar_geo=side_geo,
                             sidecar_has_geokeys=True)

    da = open_geotiff(str(base), overview_level=1)
    # Sidecar transform wins.  Rasterio-style 6-tuple:
    # ``(pixel_width, 0, origin_x, 0, pixel_height, origin_y)``.
    t = da.attrs["transform"]
    assert t[0] == pytest.approx(2.5)
    assert t[2] == pytest.approx(500.0)
    assert t[4] == pytest.approx(-2.5)
    assert t[5] == pytest.approx(800.0)
    # Sidecar EPSG wins too (sidecar CRS is 3857, base is 4326).
    assert da.attrs.get("crs") == 3857


def test_sidecar_with_own_geokeys_wins_metadata_only(tmp_path):
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(tmp_path,
                          base_geo=base_geo, sidecar_geo=side_geo,
                          sidecar_has_geokeys=True)

    geo_info, h, w, _, _ = _read_geo_info(str(base), overview_level=1)
    assert (h, w) == (4, 4)
    assert geo_info.transform.origin_x == pytest.approx(500.0)
    assert geo_info.transform.origin_y == pytest.approx(800.0)
    assert geo_info.transform.pixel_width == pytest.approx(2.5)
    assert geo_info.transform.pixel_height == pytest.approx(-2.5)
    assert geo_info.crs_epsg == 3857


# ---------------------------------------------------------------------------
# Sidecar has no geokeys -> today's behavior is preserved (inherit base).
# ---------------------------------------------------------------------------
def test_sidecar_without_geokeys_inherits_from_base_eager(tmp_path):
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    # The sidecar's georef bytes will be stripped after write, so the
    # numbers in ``side_geo`` are irrelevant to the assertion.
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(tmp_path,
                          base_geo=base_geo, sidecar_geo=side_geo,
                          sidecar_has_geokeys=False)

    da = open_geotiff(str(base), overview_level=1)
    # The sidecar IFD has no georef payload; the inheritance helper
    # pulls the transform from the base IFD and rescales for the
    # overview's reduction factor. Origin stays unchanged
    # (PixelIsArea), pixel size scales by 8/4 = 2.
    t = da.attrs["transform"]
    assert t[0] == pytest.approx(20.0)
    assert t[2] == pytest.approx(100.0)
    assert t[4] == pytest.approx(-20.0)
    assert t[5] == pytest.approx(200.0)
    assert da.attrs.get("crs") == 4326


def test_sidecar_without_geokeys_inherits_from_base_metadata_only(tmp_path):
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(tmp_path,
                          base_geo=base_geo, sidecar_geo=side_geo,
                          sidecar_has_geokeys=False)

    geo_info, h, w, _, _ = _read_geo_info(str(base), overview_level=1)
    assert (h, w) == (4, 4)
    # Base-inherited transform with rescaled pixel size for the 8->4
    # reduction factor.
    assert geo_info.transform.origin_x == pytest.approx(100.0)
    assert geo_info.transform.origin_y == pytest.approx(200.0)
    assert geo_info.transform.pixel_width == pytest.approx(20.0)
    assert geo_info.transform.pixel_height == pytest.approx(-20.0)
    assert geo_info.crs_epsg == 4326
