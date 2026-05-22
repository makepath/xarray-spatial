"""GeoTIFF IFD assembly and BigTIFF/COG layout helpers (private).

This module is private to :mod:`xrspatial.geotiff`. It owns the
TIFF/BigTIFF IFD-encoding primitives and the layout-planning helpers.
The pixel encoding kernels (strip/tile compression, photometric and
predictor encode) and the top-level write entry points live in
:mod:`xrspatial.geotiff._writer`.

The eager writer (``_write`` -> ``_assemble_tiff``) drives the high-
level entry point :func:`_assemble_tiff`. The streaming writer
(``_write_streaming``) composes the lower-level helpers directly
(``_build_ifd``, ``_compute_classic_ifd_overhead``,
``_should_use_bigtiff_streaming``, ``_promote_offsets_to_long8``) and
never calls ``_assemble_tiff``.

Functions exported from here are also re-exported from
``xrspatial.geotiff._writer`` so existing internal call sites and
tests that import them from the original module keep working.

Internal contract: ``_assemble_tiff`` resolves
``_resolve_photometric``, ``_OVERRIDABLE_AUTO_TAG_IDS``,
``_DANGEROUS_EXTRA_TAG_IDS``, and the four layout helpers it dispatches
to (``_compute_classic_ifd_overhead``, ``_promote_offsets_to_long8``,
``_assemble_cog_layout``, ``_assemble_standard_layout``) through the
``_writer`` module at call time. That preserves the pre-extraction
monkeypatch semantics for tests that patch ``_writer._compute_classic_ifd_overhead``
and friends. Do not inline those calls back to the module-local names
without also updating the affected tests.
"""
from __future__ import annotations

import struct

import numpy as np

from ._compression import COMPRESSION_NONE
from ._dtypes import ASCII, DOUBLE, LONG, LONG8, RATIONAL, SHORT, numpy_to_tiff_dtype
from ._geotags import (TAG_GDAL_NODATA, TAG_GEO_ASCII_PARAMS, TAG_GEO_KEY_DIRECTORY,
                       TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT, TAG_MODEL_TRANSFORMATION,
                       GeoTransform, build_geo_tags)
from ._header import (TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_EXTRA_SAMPLES, TAG_GDAL_METADATA,
                      TAG_IMAGE_LENGTH, TAG_IMAGE_WIDTH, TAG_NEW_SUBFILE_TYPE, TAG_PHOTOMETRIC,
                      TAG_PREDICTOR, TAG_RESOLUTION_UNIT, TAG_ROWS_PER_STRIP, TAG_SAMPLE_FORMAT,
                      TAG_SAMPLES_PER_PIXEL, TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS,
                      TAG_TILE_BYTE_COUNTS, TAG_TILE_LENGTH, TAG_TILE_OFFSETS, TAG_TILE_WIDTH,
                      TAG_X_RESOLUTION, TAG_Y_RESOLUTION)

# Byte order: always write little-endian.
BO = '<'


def _float_to_rational(val):
    """Convert a float to a TIFF RATIONAL (numerator, denominator) pair."""
    if val == int(val):
        return (int(val), 1)
    # Use a denominator of 10000 for reasonable precision
    den = 10000
    num = int(round(val * den))
    return (num, den)


def _serialize_tag_value(type_id, count, values):
    """Serialize tag values to bytes."""
    if type_id == ASCII:
        if isinstance(values, str):
            return values.encode('ascii') + b'\x00'
        return values + b'\x00'
    elif type_id == SHORT:
        if isinstance(values, (list, tuple)):
            return struct.pack(f'{BO}{count}H', *values)
        return struct.pack(f'{BO}H', values)
    elif type_id == LONG:
        if isinstance(values, (list, tuple)):
            return struct.pack(f'{BO}{count}I', *values)
        return struct.pack(f'{BO}I', values)
    elif type_id == LONG8:
        # BigTIFF 64-bit unsigned.  Used for StripOffsets / TileOffsets
        # (and their byte-count siblings) in files larger than 4 GB.
        if isinstance(values, (list, tuple)):
            return struct.pack(f'{BO}{count}Q', *values)
        return struct.pack(f'{BO}Q', values)
    elif type_id == RATIONAL:
        # RATIONAL = two LONGs (numerator, denominator) per value
        if isinstance(values, (list, tuple)) and isinstance(values[0], (list, tuple)):
            parts = []
            for num, den in values:
                parts.extend([int(num), int(den)])
            return struct.pack(f'{BO}{count * 2}I', *parts)
        else:
            num, den = _float_to_rational(float(values))
            return struct.pack(f'{BO}II', num, den)
    elif type_id == DOUBLE:
        if isinstance(values, (list, tuple)):
            return struct.pack(f'{BO}{count}d', *values)
        return struct.pack(f'{BO}d', values)
    else:
        if isinstance(values, bytes):
            return values
        return struct.pack(f'{BO}I', values)


def _pack_tag_value(tag_id: int, type_id: int, count: int,
                    values, overflow_buf: bytearray,
                    overflow_base: int, bigtiff: bool = False) -> bytes:
    """Pack a single IFD entry.

    Standard TIFF: 12 bytes (tag:2, type:2, count:4, value:4).
    BigTIFF: 20 bytes (tag:2, type:2, count:8, value:8).
    """
    val_bytes = _serialize_tag_value(type_id, count, values)

    # For ASCII, count is the actual byte length
    if type_id == ASCII:
        count = len(val_bytes)

    inline_max = 8 if bigtiff else 4

    if bigtiff:
        entry = struct.pack(f'{BO}HHQ', tag_id, type_id, count)
    else:
        entry = struct.pack(f'{BO}HHI', tag_id, type_id, count)

    if len(val_bytes) <= inline_max:
        value_field = val_bytes.ljust(inline_max, b'\x00')
    else:
        offset = overflow_base + len(overflow_buf)
        if bigtiff:
            value_field = struct.pack(f'{BO}Q', offset)
        else:
            value_field = struct.pack(f'{BO}I', offset)
        overflow_buf.extend(val_bytes)
        if len(overflow_buf) % 2:
            overflow_buf.append(0)

    return entry + value_field


def _build_ifd(tags: list[tuple], overflow_base: int,
               bigtiff: bool = False) -> tuple[bytes, bytes]:
    """Build a complete IFD block.

    Parameters
    ----------
    tags : list of (tag_id, type_id, count, values)
        Tags sorted by tag_id.
    overflow_base : int
        Where overflow data starts in the file.

    Returns
    -------
    (ifd_bytes, overflow_bytes)
    """
    # Sort by tag ID (TIFF spec requires this)
    tags = sorted(tags, key=lambda t: t[0])

    num_entries = len(tags)
    overflow_buf = bytearray()

    if bigtiff:
        ifd_parts = [struct.pack(f'{BO}Q', num_entries)]
    else:
        ifd_parts = [struct.pack(f'{BO}H', num_entries)]

    for tag_id, type_id, count, values in tags:
        entry = _pack_tag_value(tag_id, type_id, count, values,
                                overflow_buf, overflow_base, bigtiff=bigtiff)
        ifd_parts.append(entry)

    # Next IFD offset (0 = no more IFDs, will be patched for COG)
    if bigtiff:
        ifd_parts.append(struct.pack(f'{BO}Q', 0))
    else:
        ifd_parts.append(struct.pack(f'{BO}I', 0))

    return b''.join(ifd_parts), bytes(overflow_buf)


# Tags whose LONG encoding must become LONG8 in BigTIFF output.
_BIGTIFF_OFFSET_TAGS = frozenset({
    TAG_STRIP_OFFSETS,
    TAG_STRIP_BYTE_COUNTS,
    TAG_TILE_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
})


def _promote_offsets_to_long8(tags: list) -> list:
    """Retype strip/tile offset and byte-count tags from LONG to LONG8.

    Used when switching to BigTIFF output: 32-bit offsets cannot
    address past 4 GB, so the offset arrays must be emitted as
    LONG8 (uint64).  Non-offset tags pass through unchanged.
    """
    out = []
    for tag_id, type_id, count, values in tags:
        if tag_id in _BIGTIFF_OFFSET_TAGS and type_id == LONG:
            out.append((tag_id, LONG8, count, values))
        else:
            out.append((tag_id, type_id, count, values))
    return out


def _assemble_standard_layout(header_size: int,
                              ifd_specs: list,
                              pixel_data_parts: list,
                              bigtiff: bool = False) -> bytearray:
    """Assemble standard TIFF layout (one IFD at a time).

    Returns the assembled output as a ``bytearray``. The caller writes
    it via ``_write_bytes`` (which accepts any buffer-protocol object)
    and may slice it for header validation. Returning the bytearray
    directly avoids the peak-memory doubling that ``bytes(output)``
    would impose on multi-GB writes (issue #1756).
    """
    output = bytearray()
    entry_size = 20 if bigtiff else 12

    # TIFF header
    output.extend(b'II')  # little-endian
    if bigtiff:
        output.extend(struct.pack(f'{BO}H', 43))   # BigTIFF magic
        output.extend(struct.pack(f'{BO}H', 8))    # offset size
        output.extend(struct.pack(f'{BO}H', 0))    # padding
        output.extend(struct.pack(f'{BO}Q', 0))    # first IFD offset placeholder
    else:
        output.extend(struct.pack(f'{BO}H', 42))   # magic
        output.extend(struct.pack(f'{BO}I', 0))    # first IFD offset placeholder

    for level_idx, (tags, (_arr, _lw, _lh, rel_offsets, byte_counts, comp_chunks)) in enumerate(
            zip(ifd_specs, pixel_data_parts)):

        ifd_offset = len(output)

        if level_idx == 0:
            if bigtiff:
                struct.pack_into(f'{BO}Q', output, 8, ifd_offset)
            else:
                struct.pack_into(f'{BO}I', output, 4, ifd_offset)

        num_entries = len(tags)
        count_size = 8 if bigtiff else 2
        next_size = 8 if bigtiff else 4
        ifd_block_size = count_size + entry_size * num_entries + next_size
        overflow_base = ifd_offset + ifd_block_size

        ifd_bytes, overflow_bytes = _build_ifd(tags, overflow_base, bigtiff=bigtiff)

        pixel_data_offset = overflow_base + len(overflow_bytes)

        patched_tags = []
        for tag_id, type_id, count, values in tags:
            if tag_id in (TAG_STRIP_OFFSETS, TAG_TILE_OFFSETS):
                actual_offsets = [pixel_data_offset + ro for ro in rel_offsets]
                patched_tags.append((tag_id, type_id, count, actual_offsets))
            else:
                patched_tags.append((tag_id, type_id, count, values))

        ifd_bytes, overflow_bytes = _build_ifd(patched_tags, overflow_base,
                                               bigtiff=bigtiff)

        output.extend(ifd_bytes)
        output.extend(overflow_bytes)
        # Extend directly from chunk list (no intermediate join copy)
        for chunk in comp_chunks:
            output.extend(chunk)

        # Patch next IFD pointer if there are more levels
        if level_idx < len(ifd_specs) - 1:
            next_ifd_offset = len(output)
            next_ptr_pos = ifd_offset + count_size + entry_size * num_entries
            if bigtiff:
                struct.pack_into(f'{BO}Q', output, next_ptr_pos, next_ifd_offset)
            else:
                struct.pack_into(f'{BO}I', output, next_ptr_pos, next_ifd_offset)

    return output


def _assemble_cog_layout(header_size: int,
                         ifd_specs: list,
                         pixel_data_parts: list,
                         bigtiff: bool = False) -> bytearray:
    """Assemble COG layout: all IFDs first, then all pixel data.

    Returns the assembled output as a ``bytearray``; see
    :func:`_assemble_standard_layout` for the rationale (issue #1756).
    """
    entry_size = 20 if bigtiff else 12
    count_size = 8 if bigtiff else 2
    next_size = 8 if bigtiff else 4

    # First pass: compute IFD sizes
    ifd_blocks = []
    for tags in ifd_specs:
        num_entries = len(tags)
        ifd_block_size = count_size + entry_size * num_entries + next_size
        _, overflow = _build_ifd(tags, 0, bigtiff=bigtiff)
        ifd_blocks.append((ifd_block_size, len(overflow)))

    total_ifd_size = sum(bs + ov for bs, ov in ifd_blocks)
    pixel_data_start = header_size + total_ifd_size

    # COG block-order requirement (issue #2308): on-disk pixel data must
    # run smallest-overview first, then progressively larger overviews,
    # with the main-resolution image's blocks last. ``pixel_data_parts``
    # arrives in ``[main, ov_factor_2, ov_factor_4, ...]`` order (full
    # res first, then overviews of increasing decimation), so the
    # emission order is the overview tail reversed (smallest factor =
    # largest decimation = last entry, emitted first) followed by index
    # 0 (main resolution). The IFD chain still walks ``[main, ov1, ov2,
    # ...]`` -- only the byte-level placement of the tile/strip blocks
    # changes, which is what external validators like rio-cogeo check.
    n_parts = len(pixel_data_parts)
    if n_parts > 1:
        pixel_emission_order = list(range(n_parts - 1, 0, -1)) + [0]
    else:
        # In normal use, _assemble_tiff routes single-IFD outputs to
        # ``_assemble_standard_layout``; the upstream gate
        # ``is_cog and len(ifd_specs) > 1`` guarantees n_parts >= 2 here.
        # The single-entry fallback keeps the helper safe for direct
        # unit tests that exercise the COG layout with one IFD.
        pixel_emission_order = [0]

    # Second pass: pixel data offsets per level. We walk the emission
    # order to accumulate offsets, then store them indexed by the
    # original level so the third pass can look up each IFD's tile/strip
    # base directly.
    level_pixel_offsets = [0] * n_parts
    current_pixel_offset = pixel_data_start
    for emit_idx in pixel_emission_order:
        _arr, _lw, _lh, _rel, _bc, comp_chunks = pixel_data_parts[emit_idx]
        level_pixel_offsets[emit_idx] = current_pixel_offset
        current_pixel_offset += sum(len(c) for c in comp_chunks)

    # Third pass: build IFDs with correct offsets
    output = bytearray()
    output.extend(b'II')
    if bigtiff:
        output.extend(struct.pack(f'{BO}H', 43))
        output.extend(struct.pack(f'{BO}H', 8))
        output.extend(struct.pack(f'{BO}H', 0))
        output.extend(struct.pack(f'{BO}Q', header_size))
    else:
        output.extend(struct.pack(f'{BO}H', 42))
        output.extend(struct.pack(f'{BO}I', header_size))

    current_ifd_pos = header_size
    for level_idx, (tags, (_arr, _lw, _lh, rel_offsets, byte_counts, comp_chunks)) in enumerate(
            zip(ifd_specs, pixel_data_parts)):

        pixel_base = level_pixel_offsets[level_idx]

        patched_tags = []
        for tag_id, type_id, count, values in tags:
            if tag_id in (TAG_STRIP_OFFSETS, TAG_TILE_OFFSETS):
                actual_offsets = [pixel_base + ro for ro in rel_offsets]
                patched_tags.append((tag_id, type_id, count, actual_offsets))
            else:
                patched_tags.append((tag_id, type_id, count, values))

        num_entries = len(patched_tags)
        ifd_block_size = count_size + entry_size * num_entries + next_size
        overflow_base = current_ifd_pos + ifd_block_size

        ifd_bytes, overflow_bytes = _build_ifd(patched_tags, overflow_base,
                                               bigtiff=bigtiff)

        # Patch next IFD offset
        if level_idx < len(ifd_specs) - 1:
            next_ifd_pos = current_ifd_pos + ifd_block_size + len(overflow_bytes)
            ifd_ba = bytearray(ifd_bytes)
            next_ptr_pos = count_size + entry_size * num_entries
            if bigtiff:
                struct.pack_into(f'{BO}Q', ifd_ba, next_ptr_pos, next_ifd_pos)
            else:
                struct.pack_into(f'{BO}I', ifd_ba, next_ptr_pos, next_ifd_pos)
            ifd_bytes = bytes(ifd_ba)

        output.extend(ifd_bytes)
        output.extend(overflow_bytes)
        current_ifd_pos = len(output)

    # Append all pixel data in COG-compliant order: smallest overview
    # first, then progressively larger, with the main resolution image
    # last (issue #2308).
    for emit_idx in pixel_emission_order:
        _arr, _lw, _lh, _rel, _bc, comp_chunks = pixel_data_parts[emit_idx]
        for chunk in comp_chunks:
            output.extend(chunk)

    return output


def _compute_classic_ifd_overhead(tags: list) -> int:
    """Return the on-disk size of the classic-TIFF IFD for ``tags``.

    Sums the fixed IFD block (entry count + 12 bytes per entry + next-IFD
    pointer) and the variable overflow heap (values whose serialised size
    exceeds the 4-byte inline limit, including ASCII strings such as
    ``gdal_metadata`` and user-supplied ``extra_tags``).

    The heap size is recovered by building the IFD with
    ``_build_ifd(tags, overflow_base=0, bigtiff=False)`` and measuring the
    returned overflow buffer; this matches the bytes the streaming writer
    will actually emit, with no fudge constant.
    """
    num_tags = len(tags)
    # classic IFD: 2-byte count + 12-byte entries + 4-byte next-IFD pointer
    ifd_block_size = 2 + 12 * num_tags + 4
    _, overflow_bytes = _build_ifd(tags, overflow_base=0, bigtiff=False)
    return ifd_block_size + len(overflow_bytes)


def _should_use_bigtiff_streaming(uncompressed_bytes: int,
                                  n_entries: int,
                                  ifd_overhead_bytes: int,
                                  header_size_classic: int = 8) -> bool:
    """Decide whether the streaming writer must emit BigTIFF.

    Classic TIFF stores offsets as uint32, so the file size addressable
    via classic offsets is at most ``UINT32_MAX`` bytes (offsets run
    ``0..UINT32_MAX - 1``). The streaming writer appends pixel data after
    the header and IFD, so the final file size is
    ``header + ifd + overflow + strip_table + uncompressed_bytes``.

    The comparison is ``> UINT32_MAX`` to match the eager
    ``_assemble_tiff`` decision (``estimated_file_size > UINT32_MAX``):
    a file that is exactly ``UINT32_MAX`` bytes still fits classic.

    See issue #1785 and the Copilot review on PR #1787: the previous
    helper applied a 200-byte fudge for IFD overhead, which silently
    underestimated when ``gdal_metadata_xml`` or large ``extra_tags``
    pushed the actual overflow heap well past that constant.

    Parameters
    ----------
    uncompressed_bytes : int
        Total pixel-data bytes that will be written after the IFD.
    n_entries : int
        Number of strip or tile entries; each contributes a LONG offset
        (4 bytes) plus a LONG byte-count (4 bytes) to the overflow heap.
        Pass ``0`` if ``ifd_overhead_bytes`` already covers the strip
        table (the streaming-writer caller does this by passing the
        actual tag list through ``_compute_classic_ifd_overhead``).
    ifd_overhead_bytes : int
        Classic-TIFF IFD size: fixed entry block plus variable overflow
        heap (ASCII metadata, geo tags, strip/tile offset arrays, etc.).
        Computed via ``_compute_classic_ifd_overhead(tags)``.
    header_size_classic : int, optional
        Classic-TIFF header size (8 bytes).
    """
    # strip/tile-table overhead is 8 bytes per entry (LONG offset + LONG
    # byte count). If the caller already accounted for the offset arrays
    # inside ``ifd_overhead_bytes`` they should pass n_entries=0.
    strip_table_overhead = n_entries * 8
    reserved_overhead = (
        header_size_classic + ifd_overhead_bytes + strip_table_overhead
    )
    UINT32_MAX = 0xFFFFFFFF
    # ``> UINT32_MAX`` matches the eager path's
    # ``estimated_file_size > UINT32_MAX`` check in ``_assemble_tiff``.
    return uncompressed_bytes + reserved_overhead > UINT32_MAX


def _assemble_tiff(width: int, height: int, dtype: np.dtype,
                   compression: int, predictor: int,
                   tiled: bool, tile_size: int,
                   pixel_data_parts: list[tuple],
                   geo_transform: GeoTransform | None,
                   crs_epsg: int | None,
                   nodata,
                   is_cog: bool = False,
                   raster_type: int = 1,
                   crs_wkt: str | None = None,
                   gdal_metadata_xml: str | None = None,
                   extra_tags: list | None = None,
                   x_resolution: float | None = None,
                   y_resolution: float | None = None,
                   resolution_unit: int | None = None,
                   force_bigtiff: bool | None = None,
                   photometric='auto') -> bytearray:
    """Assemble a complete TIFF file.

    This function is owned by ``_writer.py`` semantically; it lives in
    ``_write_layout.py`` only to keep the IFD-encoding helpers
    co-located. It resolves ``_resolve_photometric``,
    ``_OVERRIDABLE_AUTO_TAG_IDS``, ``_DANGEROUS_EXTRA_TAG_IDS``, and
    the four layout helpers it dispatches to through the ``_writer``
    module at call time, so existing test monkeypatches against
    ``_writer.*`` keep working unchanged.

    Parameters
    ----------
    pixel_data_parts : list of (array, width, height, rel_offsets, byte_counts, comp_data)
        One entry per resolution level (full res first, then overviews).
    is_cog : bool
        If True, layout IFDs contiguously at file start (COG layout).
    raster_type : int
        1 = PixelIsArea, 2 = PixelIsPoint.

    Returns
    -------
    bytearray
        Complete TIFF file. The bytearray is returned directly rather
        than copied into an immutable ``bytes`` object so multi-GB
        writes do not transiently double peak memory; downstream
        consumers (``_write_bytes``, ``parse_header`` for the
        post-write validation slice) accept the buffer protocol so the
        type change is transparent. See issue #1756.
    """
    # Photometric / extra-samples resolution and the writer-side tag
    # filter live in ``_writer.py`` (next to the public ``photometric``
    # kwarg validation). Import lazily here to keep this module free of
    # an import cycle with ``_writer``. Look up the IFD-overhead and
    # layout helpers through ``_writer`` as well so existing tests that
    # monkeypatch ``_writer._compute_classic_ifd_overhead`` (etc.) keep
    # their override semantics from the pre-extraction layout.
    from . import _writer as _writer_mod
    _DANGEROUS_EXTRA_TAG_IDS = _writer_mod._DANGEROUS_EXTRA_TAG_IDS
    _OVERRIDABLE_AUTO_TAG_IDS = _writer_mod._OVERRIDABLE_AUTO_TAG_IDS
    _resolve_photometric = _writer_mod._resolve_photometric
    _compute_classic_ifd_overhead_fn = _writer_mod._compute_classic_ifd_overhead
    _promote_offsets_to_long8_fn = _writer_mod._promote_offsets_to_long8
    _assemble_cog_layout_fn = _writer_mod._assemble_cog_layout
    _assemble_standard_layout_fn = _writer_mod._assemble_standard_layout

    bits_per_sample, sample_format = numpy_to_tiff_dtype(dtype)

    # Determine samples per pixel from the pixel data
    first_arr = pixel_data_parts[0][0]
    samples_per_pixel = first_arr.shape[2] if first_arr.ndim == 3 else 1

    # Build geo tags
    geo_tags_dict = {}
    if geo_transform is not None:
        geo_tags_dict = build_geo_tags(
            geo_transform, crs_epsg, nodata, raster_type=raster_type,
            crs_wkt=crs_wkt)
    else:
        # No spatial reference -- still write CRS and nodata if provided
        if crs_epsg is not None or crs_wkt is not None or nodata is not None:
            geo_tags_dict = build_geo_tags(
                GeoTransform(), crs_epsg, nodata, raster_type=raster_type,
                crs_wkt=crs_wkt,
            )
            # Remove the default pixel scale / tiepoint tags since we
            # have no real transform -- keep only GeoKeys and NODATA.
            geo_tags_dict.pop(TAG_MODEL_PIXEL_SCALE, None)
            geo_tags_dict.pop(TAG_MODEL_TIEPOINT, None)

    # Compression tag for predictor
    pred_val = predictor if compression != COMPRESSION_NONE else 1

    # Resolve photometric interpretation once so primary IFD and any
    # overviews carry the same values. A user-supplied ``extra_tags``
    # entry of (TAG_PHOTOMETRIC, ...) or (TAG_EXTRA_SAMPLES, ...)
    # overrides the writer's chosen value at every level. See issue
    # #1769.
    auto_photometric, auto_extras = _resolve_photometric(
        photometric, samples_per_pixel)
    user_photometric_override = None
    user_extras_override = None
    if extra_tags is not None:
        for _et in extra_tags:
            if _et[0] not in _OVERRIDABLE_AUTO_TAG_IDS:
                continue
            if _et[0] == TAG_PHOTOMETRIC:
                user_photometric_override = _et
            elif _et[0] == TAG_EXTRA_SAMPLES:
                user_extras_override = _et

    # Build IFDs for each resolution level
    ifd_specs = []
    for level_idx, (arr, lw, lh, rel_offsets, byte_counts, comp_data) in enumerate(
            pixel_data_parts):
        tags = []

        # Mark overview IFDs as reduced-resolution images (TIFF tag 254).
        # GDAL/rasterio use this tag to identify overview sub-IFDs.
        if level_idx > 0:
            tags.append((TAG_NEW_SUBFILE_TYPE, LONG, 1, 1))

        tags.append((TAG_IMAGE_WIDTH, LONG, 1, lw))
        tags.append((TAG_IMAGE_LENGTH, LONG, 1, lh))
        if samples_per_pixel > 1:
            tags.append((TAG_BITS_PER_SAMPLE, SHORT, samples_per_pixel,
                         [bits_per_sample] * samples_per_pixel))
        else:
            tags.append((TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample))
        tags.append((TAG_COMPRESSION, SHORT, 1, compression))
        # Photometric: caller-controlled via the ``photometric`` kwarg
        # (default 'auto' = MinIsBlack for any band count, so a 4-band
        # raster is not silently tagged as RGB+alpha). Issue #1769.
        if user_photometric_override is not None:
            tags.append(user_photometric_override)
        else:
            tags.append((TAG_PHOTOMETRIC, SHORT, 1, auto_photometric))
        tags.append((TAG_SAMPLES_PER_PIXEL, SHORT, 1, samples_per_pixel))
        if samples_per_pixel > 1:
            tags.append((TAG_SAMPLE_FORMAT, SHORT, samples_per_pixel,
                         [sample_format] * samples_per_pixel))
        else:
            tags.append((TAG_SAMPLE_FORMAT, SHORT, 1, sample_format))

        # ExtraSamples: count matches samples_per_pixel - bands consumed
        # by Photometric. User override (from extra_tags) wins.
        if user_extras_override is not None:
            tags.append(user_extras_override)
        elif auto_extras:
            tags.append((TAG_EXTRA_SAMPLES, SHORT, len(auto_extras),
                         list(auto_extras)))

        if pred_val != 1:
            tags.append((TAG_PREDICTOR, SHORT, 1, pred_val))

        # Resolution / DPI tags
        if x_resolution is not None:
            tags.append((TAG_X_RESOLUTION, RATIONAL, 1, x_resolution))
        if y_resolution is not None:
            tags.append((TAG_Y_RESOLUTION, RATIONAL, 1, y_resolution))
        if resolution_unit is not None:
            tags.append((TAG_RESOLUTION_UNIT, SHORT, 1, resolution_unit))

        if tiled:
            tags.append((TAG_TILE_WIDTH, SHORT, 1, tile_size))
            tags.append((TAG_TILE_LENGTH, SHORT, 1, tile_size))
            # Placeholder offsets/counts -- will be patched
            tags.append((TAG_TILE_OFFSETS, LONG, len(rel_offsets), rel_offsets))
            tags.append((TAG_TILE_BYTE_COUNTS, LONG, len(byte_counts), byte_counts))
        else:
            rows_per_strip = 256
            if lh <= rows_per_strip:
                rows_per_strip = lh
            tags.append((TAG_ROWS_PER_STRIP, SHORT, 1, rows_per_strip))
            tags.append((TAG_STRIP_OFFSETS, LONG, len(rel_offsets), rel_offsets))
            tags.append((TAG_STRIP_BYTE_COUNTS, LONG, len(byte_counts), byte_counts))

        # Geo tags only on first IFD
        if level_idx == 0:
            for gtag, gval in geo_tags_dict.items():
                if gtag == TAG_MODEL_PIXEL_SCALE:
                    tags.append((gtag, DOUBLE, 3, list(gval)))
                elif gtag == TAG_MODEL_TIEPOINT:
                    tags.append((gtag, DOUBLE, 6, list(gval)))
                elif gtag == TAG_MODEL_TRANSFORMATION:
                    tags.append((gtag, DOUBLE, 16, list(gval)))
                elif gtag == TAG_GEO_KEY_DIRECTORY:
                    tags.append((gtag, SHORT, len(gval), list(gval)))
                elif gtag == TAG_GEO_ASCII_PARAMS:
                    tags.append((gtag, ASCII, len(str(gval)) + 1, str(gval)))
                elif gtag == TAG_GDAL_NODATA:
                    tags.append((gtag, ASCII, len(str(gval)) + 1, str(gval)))

            # GDALMetadata XML (tag 42112)
            if gdal_metadata_xml is not None:
                tags.append((TAG_GDAL_METADATA, ASCII,
                             len(gdal_metadata_xml) + 1, gdal_metadata_xml))

            # Extra tags (pass-through from source file)
            if extra_tags is not None:
                # Compute existing tag IDs once; update as we append to keep
                # this loop O(len(extra_tags) + len(tags)) instead of O(N*M).
                # See issue #1657 for the filter rationale.
                existing_ids = {t[0] for t in tags}
                for etag_id, etype_id, ecount, evalue in extra_tags:
                    if (etag_id not in existing_ids
                            and etag_id not in _DANGEROUS_EXTRA_TAG_IDS):
                        tags.append((etag_id, etype_id, ecount, evalue))
                        existing_ids.add(etag_id)

        ifd_specs.append(tags)

    # --- Determine if BigTIFF is needed ---
    # Classic TIFF uses 32-bit offsets (max ~4.29 GB). Estimate total
    # file size including headers, IFDs, overflow heap, and all pixel
    # data; switch to BigTIFF if any offset could exceed 2^32. The IFD
    # overhead is the exact bytes ``_build_ifd`` would emit, summed
    # across all IFDs. The earlier fixed 1 KB-per-IFD fudge
    # under-promoted near the 4 GiB boundary when ``gdal_metadata_xml``
    # or ``extra_tags`` pushed the overflow heap past that constant
    # (#1905). Shares ``_compute_classic_ifd_overhead`` with the
    # streaming writer's BigTIFF decision (#1785, #1787).
    total_pixel_data = sum(sum(len(c) for c in chunks)
                           for _, _, _, _, _, chunks in pixel_data_parts)
    ifd_overhead = sum(
        _compute_classic_ifd_overhead_fn(tags) for tags in ifd_specs
    )
    estimated_file_size = 8 + ifd_overhead + total_pixel_data

    UINT32_MAX = 0xFFFFFFFF  # 4,294,967,295
    if force_bigtiff is not None:
        bigtiff = force_bigtiff
    else:
        bigtiff = estimated_file_size > UINT32_MAX

    header_size = 16 if bigtiff else 8

    # In BigTIFF, StripOffsets / TileOffsets and their byte-count
    # siblings must use 64-bit offsets.  The ifd_specs above were
    # built with LONG (uint32) because bigtiff wasn't yet decided;
    # promote them to LONG8 here.  This is the write-side counterpart
    # of the 64-bit offset handling in _header.parse_ifd.
    if bigtiff:
        ifd_specs = [_promote_offsets_to_long8_fn(tags) for tags in ifd_specs]

    if is_cog and len(ifd_specs) > 1:
        return _assemble_cog_layout_fn(header_size, ifd_specs, pixel_data_parts,
                                       bigtiff=bigtiff)
    else:
        return _assemble_standard_layout_fn(header_size, ifd_specs, pixel_data_parts,
                                            bigtiff=bigtiff)
