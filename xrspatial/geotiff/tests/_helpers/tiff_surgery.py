"""In-place TIFF byte-surgery helpers shared by security-cap tests.

The local strip / tile byte-cap tests and the GPU per-tile byte-cap
test both need to forge a TIFF whose declared ``TileByteCounts`` (tag
325) or ``StripByteCounts`` (tag 279) entries exceed the production
cap. They each parse the leading IFD and rewrite every matching tag's
value array in place. Keeping two near-identical copies of that
surgery in two test files invited drift, so the helpers now live here.

Not part of the public API; used only by the test suite.
"""
from __future__ import annotations

import struct


def patch_byte_counts(data: bytearray, tag: int, value: int) -> None:
    """Rewrite every entry for *tag* in the first IFD of *data*.

    Parameters
    ----------
    data : bytearray
        Mutable TIFF file bytes (entire file). Mutated in place.
    tag : int
        ``325`` for ``TileByteCounts`` or ``279`` for ``StripByteCounts``.
        Other tags work mechanically but the helper exists for those two.
    value : int
        New value to stamp into every byte-count entry. For ``SHORT``
        (type 3) entries the value is clipped to ``0xFFFF`` because the
        on-disk slot is 16-bit; tests that need a multi-MB value must
        ensure the source file was written with a ``LONG`` (type 4) tag.

    Raises
    ------
    AssertionError
        When ``tag`` is not present in the first IFD.
    """
    from xrspatial.geotiff._header import parse_header

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
        if type_id == 4:  # LONG
            total = count * 4
            if total <= 4:
                for k in range(count):
                    struct.pack_into(f"{bo}I", data, eo + 8 + k * 4, value)
            else:
                ptr = struct.unpack_from(f"{bo}I", data, eo + 8)[0]
                for k in range(count):
                    struct.pack_into(f"{bo}I", data, ptr + k * 4, value)
        elif type_id == 3:  # SHORT
            clipped = min(value, 0xFFFF)
            total = count * 2
            if total <= 4:
                for k in range(count):
                    struct.pack_into(
                        f"{bo}H", data, eo + 8 + k * 2, clipped)
            else:
                ptr = struct.unpack_from(f"{bo}I", data, eo + 8)[0]
                for k in range(count):
                    struct.pack_into(
                        f"{bo}H", data, ptr + k * 2, clipped)
        return
    raise AssertionError(f"tag {tag} not found in IFD")
