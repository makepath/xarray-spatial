"""Deprecation tests for the projected-CRS GeoKey pass-through attrs.

Issue #1984, PR 7 (projected-attrs slice).

PR 6 (#2004) locked the pass-through tier of the attrs contract and
flagged ``linear_units`` and ``projection_code`` as DROPPED on a write +
read round-trip: ``xrspatial.geotiff._geotags.build_geo_tags`` only
emits the primary ``GEOKEY_PROJECTED_CS_TYPE`` (3072) and never the
secondary projected GeoKeys (``GEOKEY_PROJECTION`` 3074,
``GEOKEY_PROJ_LINEAR_UNITS`` 3076), so the reader has nothing to
rebuild the attrs from.

This file pins the deprecation behaviour for one release cycle:

* ``linear_units`` and ``projection_code`` are still emitted by the
  read path when the source file carries the secondary GeoKey.
* Each emission triggers a ``DeprecationWarning`` pointing at issue
  #1984 so downstream code can migrate before the attrs are removed.

Removal will land in a follow-up after the deprecation cycle.
"""
from __future__ import annotations

import struct
import warnings

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff

from .conftest import make_minimal_tiff


def _make_projected_tiff_with_secondary_geokeys(
    *,
    epsg: int = 32633,
    include_linear_units: bool = False,
    include_projection_code: bool = False,
    linear_units_code: int = 9001,   # metre
    projection_code_value: int = 16033,
) -> bytes:
    """Build a TIFF carrying a projected CRS plus selected secondary GeoKeys.

    Builds on :func:`make_minimal_tiff` to produce a stripped float32
    TIFF with ModelPixelScale + ModelTiepoint and a GeoKeyDirectory
    (tag 34735) that carries the primary
    ``GEOKEY_PROJECTED_CS_TYPE`` (3072) entry plus optional secondary
    ``GEOKEY_PROJECTION`` (3074) and ``GEOKEY_PROJ_LINEAR_UNITS``
    (3076) entries. ``make_minimal_tiff`` itself only emits the primary
    GeoKey, so we rebuild and replace tag 34735 in the output bytes.
    """
    # Start with the baseline projected TIFF (primary GeoKey only).
    base = make_minimal_tiff(
        4, 4, np.dtype('float32'),
        geo_transform=(500000.0, 4649776.0, 30.0, -30.0),
        epsg=epsg,
    )

    # Compose the full GeoKey directory we want to inject. Format per
    # the GeoTIFF spec for tag 34735: header [version, rev_major,
    # rev_minor, num_keys] followed by 4-short tuples
    # [key_id, tiff_tag_location, count, value_or_offset].
    keys: list[tuple[int, int, int, int]] = []
    # GTModelTypeGeoKey -> Projected
    keys.append((1024, 0, 1, 1))
    # GTRasterTypeGeoKey -> PixelIsArea (default but explicit)
    keys.append((1025, 0, 1, 1))
    # ProjectedCSTypeGeoKey
    keys.append((3072, 0, 1, epsg))
    if include_projection_code:
        keys.append((3074, 0, 1, projection_code_value))
    if include_linear_units:
        keys.append((3076, 0, 1, linear_units_code))

    num_keys = len(keys)
    gkd: list[int] = [1, 1, 0, num_keys]
    for k in keys:
        gkd.extend(k)

    # Locate tag 34735 in the original IFD and overwrite its short
    # buffer in place. ``make_minimal_tiff`` always uses little-endian
    # ('II', magic 42) and a standard 8-byte header, so the IFD starts
    # at offset 8 with a uint16 entry count followed by 12-byte
    # records.
    out = bytearray(base)
    ifd_off = struct.unpack_from('<I', out, 4)[0]
    n_entries = struct.unpack_from('<H', out, ifd_off)[0]
    entries_off = ifd_off + 2

    new_gkd_bytes = struct.pack(f'<{len(gkd)}H', *gkd)

    for i in range(n_entries):
        rec_off = entries_off + i * 12
        tag_id, typ, count = struct.unpack_from('<HHI', out, rec_off)
        if tag_id != 34735:
            continue
        # Original GKD value lives at the overflow offset stored in the
        # last 4 bytes of the record. The conftest writer guarantees the
        # overflow buffer is contiguous and word-aligned; appending the
        # new GKD at end-of-file and repointing the entry is the
        # simplest way to keep all other tag offsets stable.
        new_count = len(gkd)
        new_offset = len(out)
        # Pad to word alignment to be safe.
        if new_offset % 2:
            out.append(0)
            new_offset = len(out)
        out.extend(new_gkd_bytes)
        struct.pack_into(
            '<HHI', out, rec_off, tag_id, typ, new_count
        )
        struct.pack_into('<I', out, rec_off + 8, new_offset)
        break
    else:
        raise AssertionError(
            "baseline TIFF did not include tag 34735 (GeoKeyDirectory)"
        )

    return bytes(out)


def _write_tmp_tiff(tmp_path, name: str, payload: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def test_linear_units_emits_deprecation_warning(tmp_path):
    """Reading a projected TIFF with ``ProjLinearUnitsGeoKey`` fires
    a ``DeprecationWarning`` for ``attrs['linear_units']``."""
    payload = _make_projected_tiff_with_secondary_geokeys(
        epsg=32633, include_linear_units=True,
    )
    path = _write_tmp_tiff(tmp_path, 'linear_units.tif', payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        open_geotiff(path)

    matched = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "attrs['linear_units']" in str(w.message)
    ]
    assert matched, (
        "expected a DeprecationWarning mentioning attrs['linear_units']; "
        f"captured warnings: {[(w.category.__name__, str(w.message)) for w in caught]}"
    )
    msg = str(matched[0].message)
    assert '#1984' in msg, (
        f"deprecation message should reference issue #1984: {msg!r}"
    )


def test_projection_code_emits_deprecation_warning(tmp_path):
    """Reading a projected TIFF with ``ProjectionGeoKey`` fires a
    ``DeprecationWarning`` for ``attrs['projection_code']``."""
    payload = _make_projected_tiff_with_secondary_geokeys(
        epsg=32633, include_projection_code=True,
    )
    path = _write_tmp_tiff(tmp_path, 'projection_code.tif', payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        open_geotiff(path)

    matched = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "attrs['projection_code']" in str(w.message)
    ]
    assert matched, (
        "expected a DeprecationWarning mentioning attrs['projection_code']; "
        f"captured warnings: {[(w.category.__name__, str(w.message)) for w in caught]}"
    )
    msg = str(matched[0].message)
    assert '#1984' in msg, (
        f"deprecation message should reference issue #1984: {msg!r}"
    )


def test_deprecated_projected_attrs_still_emitted(tmp_path):
    """During the deprecation cycle the attrs are still populated.

    Removing emission is a separate follow-up; this test guards
    against an accidental early removal.
    """
    payload = _make_projected_tiff_with_secondary_geokeys(
        epsg=32633,
        include_linear_units=True,
        include_projection_code=True,
    )
    path = _write_tmp_tiff(tmp_path, 'both.tif', payload)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        da = open_geotiff(path)

    assert 'linear_units' in da.attrs, (
        "attrs['linear_units'] disappeared during deprecation cycle; "
        "emission removal is a separate follow-up to this PR."
    )
    assert da.attrs['linear_units'] == 'metre'

    assert 'projection_code' in da.attrs, (
        "attrs['projection_code'] disappeared during deprecation cycle; "
        "emission removal is a separate follow-up to this PR."
    )
    assert da.attrs['projection_code'] == 16033
