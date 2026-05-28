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
"""
from __future__ import annotations

import importlib.util
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

#: Fallback bytes-per-pixel used by ``_gb_hint`` when the caller has
#: not resolved the decoded dtype yet. float32 matches the constant
#: docstring above and the dtype the decoder produces for the common
#: single-band float case. Real call sites pass the actual dtype so
#: the GB hint reflects the allocation that would have happened.
_DEFAULT_HINT_BYTES = 4


def _gb_hint(pixels, dtype=None):
    """Return a ``~X.XX GB at N bytes/pixel`` string for ``pixels``.

    ``dtype`` is the numpy dtype of the would-be allocation. When
    provided, the hint uses ``dtype.itemsize`` so f64 reports 8
    bytes/pixel, u8 reports 1, etc. When omitted, the hint falls
    back to ``_DEFAULT_HINT_BYTES`` (float32) for older call sites
    that have not threaded the dtype through yet.

    Uses decimal GB (10**9 bytes) so the printed number matches the
    ``~4 GB`` figure quoted in the :data:`MAX_PIXELS_DEFAULT`
    docstring above and the convention reported by ``df`` / ``du`` /
    most cloud object stores.
    """
    if dtype is None:
        bpp = _DEFAULT_HINT_BYTES
    else:
        bpp = int(np.dtype(dtype).itemsize)
    gb = (pixels * bpp) / 1_000_000_000
    return f"~{gb:.2f} GB at {bpp} bytes/pixel"


def _suggest_chunk_side(max_pixels, samples):
    """Pick a square chunksize whose pixel count fits comfortably under
    ``max_pixels`` for the given band count. Capped at 1024 so the hint
    stays near common COG tile sizes (256 / 512 / 1024)."""
    samples = max(1, int(samples))
    if max_pixels <= 0:
        return 1
    side = int(math.isqrt(max_pixels // samples))
    return max(1, min(1024, side))


def _recovery_hint(max_pixels, samples):
    """Build the multi-line recovery hint appended to PixelSafetyLimitError.

    Always suggests ``window=`` and ``bbox=`` for reading a sub-region.
    Suggests ``chunks=`` when dask is installed; otherwise recommends
    installing dask so chunked reads become available.
    ``importlib.util.find_spec`` avoids importing dask just to format
    an error.
    """
    chunk = _suggest_chunk_side(max_pixels, samples)
    lines = [
        "To read this file, try one of:",
        "  * Pass a larger max_pixels= if the allocation is acceptable.",
        f"  * Read a sub-region with window=(r0, c0, r1, c1), "
        f"e.g. window=(0, 0, {chunk}, {chunk}).",
        "  * Read a geographic sub-region with "
        "bbox=(x_min, y_min, x_max, y_max) in the file's CRS.",
    ]
    if importlib.util.find_spec('dask') is not None:
        lines.append(
            f"  * Read lazily in chunks with chunks={chunk} so each "
            f"decoded buffer stays under max_pixels."
        )
    else:
        lines.append(
            "  * Install dask (`pip install dask` or "
            "`conda install -c conda-forge dask`) to read the file "
            f"lazily via chunks={chunk}."
        )
    return "\n".join(lines)


class PixelSafetyLimitError(ValueError):
    """Raised when a requested TIFF allocation exceeds max_pixels."""


def _check_dimensions(width, height, samples, max_pixels, dtype=None):
    """Raise PixelSafetyLimitError if the request exceeds *max_pixels*.

    ``dtype`` is forwarded to :func:`_gb_hint` so the byte-allocation
    estimate in the error message reflects the actual dtype that
    would have been allocated. Optional for callers that do not have
    the dtype resolved yet (the hint then falls back to float32).
    """
    total = width * height * samples
    if total > max_pixels:
        raise PixelSafetyLimitError(
            f"TIFF image dimensions ({width} x {height} x {samples} = "
            f"{total:,} pixels, {_gb_hint(total, dtype)}) exceed the "
            f"safety limit of {max_pixels:,} pixels "
            f"({_gb_hint(max_pixels, dtype)}).\n"
            f"{_recovery_hint(max_pixels, samples)}"
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
    helper closes the same gap for the stripped path.
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
        # scientific notation / fractional values.
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
#: scale.
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
