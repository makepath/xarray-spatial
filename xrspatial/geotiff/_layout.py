"""TIFF/COG layout validation and byte-budget helpers.

This module is private to :mod:`xrspatial.geotiff`. It collects the
read-side layout helpers that were previously colocated with the
top-level reader: pixel-count safety limits, source-dimension
validation, sparse-tile/strip handling, the per-image byte budget used
to bound HTTP body reads, and the IFD-extent probe used by the COG
HTTP prefetch grow loop.

The helpers are intentionally free of decode (codec) and transport
(HTTP / mmap / fsspec) concerns. Decode lives in :mod:`._decode`,
transport lives in :mod:`._sources`, top-level orchestration lives in
:mod:`._reader`.

Source: PR-H of the GeoTIFF refactor epic, issue #2247.
"""
from __future__ import annotations

import math

import numpy as np

from ._decode import _int_nodata_in_range
from ._geotags import _parse_nodata_str as _parse_nd
from ._header import IFD
from ._sources import _max_tile_bytes_from_env

# ---------------------------------------------------------------------------
# Allocation guard: reject TIFF dimensions that would exhaust memory
# ---------------------------------------------------------------------------

#: Default maximum total pixel count (width * height * samples).
#: ~1 billion pixels, which is ~4 GB for float32 single-band.
#: Override per-call via the ``max_pixels`` keyword argument.
MAX_PIXELS_DEFAULT = 1_000_000_000


class PixelSafetyLimitError(ValueError):
    """Raised when a requested TIFF allocation exceeds max_pixels."""


def _check_dimensions(width, height, samples, max_pixels):
    """Raise PixelSafetyLimitError if the request exceeds *max_pixels*."""
    total = width * height * samples
    if total > max_pixels:
        raise PixelSafetyLimitError(
            f"TIFF image dimensions ({width} x {height} x {samples} = "
            f"{total:,} pixels) exceed the safety limit of "
            f"{max_pixels:,} pixels.  Pass a larger max_pixels value to "
            f"read_to_array() if this file is legitimate."
        )


def _check_source_dimensions(width, height, samples):
    """Validate the source IFD dimensions of a TIFF before any windowing.

    Companion to :func:`_check_dimensions`, which only enforces the
    upper bound. The stripped read paths read ``width``,  ``height``,
    and ``samples_per_pixel`` straight off the IFD and then clamp the
    output window to those values, so a malformed file with
    ``ImageWidth = 0`` (or a negative value, which would parse as a
    huge unsigned int but can also surface via signed-cast errors)
    would produce an empty array silently. The tiled paths are already
    protected by :func:`validate_tile_layout` in ``_header.py``; this
    helper closes the same gap for the stripped path. Issue #2053.
    """
    if width <= 0 or height <= 0 or samples <= 0:
        raise ValueError(
            f"Invalid TIFF dimensions: ImageWidth={width}, "
            f"ImageLength={height}, SamplesPerPixel={samples} "
            f"(all must be > 0)"
        )


# ---------------------------------------------------------------------------
# Sparse tile / strip handling
# ---------------------------------------------------------------------------


def _sparse_fill_value(ifd: IFD, dtype: np.dtype):
    """Resolve the fill value for sparse tiles/strips.

    A sparse TIFF entry has TileByteCounts/StripByteCounts == 0 (and
    typically the matching Offset == 0). GDAL emits these for SPARSE_OK
    files where blocks containing only the nodata value are omitted.
    The reader is expected to materialise such blocks as nodata, or
    zero when nodata is unset (the default per the GDAL convention).
    """
    nodata_str = ifd.nodata_str
    if nodata_str is not None:
        # Try ``int`` first so 64-bit sentinels survive without the
        # float64 round-trip; fall back to ``float`` for NaN / Inf /
        # scientific notation / fractional values.  See issue #1847.
        parsed = _parse_nd(nodata_str)
        if parsed is not None:
            if dtype.kind == 'f':
                return dtype.type(parsed)
            if isinstance(parsed, int):
                if _int_nodata_in_range(parsed, dtype):
                    return dtype.type(parsed)
            elif not math.isnan(parsed) and not math.isinf(parsed):
                if float(parsed).is_integer():
                    nodata_int = int(parsed)
                    if _int_nodata_in_range(nodata_int, dtype):
                        return dtype.type(nodata_int)
    return dtype.type(0)


def _has_sparse(byte_counts) -> bool:
    """Return True if any tile/strip is empty (byte_count == 0)."""
    if byte_counts is None:
        return False
    for bc in byte_counts:
        if bc == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-image byte budget (HTTP body bound for stripped TIFFs)
# ---------------------------------------------------------------------------

#: Slack added to the strip-table byte budget for the TIFF header,
#: trailing IFD chain, ExifIFD, GeoKey directory, GDAL_METADATA, and any
#: ICC profile or XMP packet. 4 MiB is comfortable for real-world COGs
#: (the prefetch path already tolerates up to ``MAX_HTTP_HEADER_BYTES``
#: of header bytes) while still bounding the body away from gigabyte
#: scale. Issue #2051.
_FULL_IMAGE_BUDGET_HEADER_SLACK = 4 * 1024 * 1024


def _compute_full_image_byte_budget(offsets, byte_counts) -> int:
    """Compute an upper bound on the legitimate HTTP body size for a stripped TIFF.

    A stripped TIFF body is laid out as: [TIFF header + IFDs + tag value
    arrays] followed by strip payloads at the offsets listed in
    ``StripOffsets``. The largest byte index any strip references is
    ``max(offset + byte_count)`` across the strip table; the body cannot
    legitimately extend past that point plus a small tail for trailing
    metadata. We add :data:`_FULL_IMAGE_BUDGET_HEADER_SLACK` to cover the
    header prologue (which lives at offset 0) and any tags that follow
    the last strip. The cap is loose by design -- it exists to reject
    bodies that are orders of magnitude larger than the file claims to
    be, not to second-guess legitimate layouts.

    If the strip table is missing or empty (sparse-only, malformed),
    fall back to the per-strip safety cap so the read is still bounded.
    Issue #2051.
    """
    fallback = _max_tile_bytes_from_env() + _FULL_IMAGE_BUDGET_HEADER_SLACK
    if not offsets or not byte_counts:
        return fallback
    max_end = 0
    for off, bc in zip(offsets, byte_counts):
        try:
            end = int(off) + int(bc)
        except (TypeError, ValueError):
            continue
        if end > max_end:
            max_end = end
    if max_end <= 0:
        return fallback
    return max_end + _FULL_IMAGE_BUDGET_HEADER_SLACK


# ---------------------------------------------------------------------------
# COG HTTP IFD-extent probe (grow loop)
# ---------------------------------------------------------------------------


def _ifd_required_extent(ifds: list[IFD]) -> int:
    """Return the highest byte offset the parsed IFDs reference.

    Used to decide whether the prefetch buffer is large enough to hold the
    entire IFD chain plus every out-of-line tag value. We compare this
    against ``len(data)`` in :func:`_parse_cog_http_meta`; if it exceeds the
    buffer, the chain is truncated and the caller must grow and retry.

    The walk re-derives each tag's value-area placement directly from the
    IFD layout (entry table base + entry slot) rather than re-parsing the
    raw bytes. For out-of-line tags ``parse_ifd`` already resolved the
    pointer and validated ``ptr + size <= len(data)``; the *interesting*
    extent for the grow loop is the next-IFD pointer of the chain tail,
    plus an "is there a next IFD we have not yet seen" probe.
    """
    if not ifds:
        return 0

    required = 0
    # Last IFD's next_ifd_offset: 0 means end-of-chain; anything else
    # points at an IFD we haven't parsed yet because it sat past the
    # buffer (parse_all_ifds stops on offset >= len(data)).
    tail_next = ifds[-1].next_ifd_offset
    if tail_next != 0:
        # Need at least enough bytes to reach the next IFD header. Pad
        # by a small amount so parse_ifd can read the num_entries field
        # without truncation -- the actual entry table is bounded by the
        # parser's own checks on the next grow iteration.
        required = max(required, tail_next + 64)

    # Out-of-line tag values are already parsed (parse_ifd bounds-checked
    # ptr + total_size <= len(data) before reading). For grow logic we
    # only need to ensure those checks did not *fail*; a thrown
    # ValueError surfaces in parse_all_ifds and is handled by the loop.
    return required
