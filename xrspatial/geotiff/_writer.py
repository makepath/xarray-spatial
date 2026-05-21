"""GeoTIFF/COG writer (private).

This module is private to :mod:`xrspatial.geotiff`. The supported public
write entry points are :func:`xrspatial.geotiff.to_geotiff`,
:func:`xrspatial.geotiff.write_geotiff_gpu`, and
:func:`xrspatial.geotiff.write_vrt`. Direct callers of the helpers
defined here bypass the DataArray-level validation that the public
wrappers run (``transform`` derivation, ``masked_nodata`` handling,
``band``-first dim reordering, ...) and must accept the resulting byte
divergence. See issue #2138.

For source modules inside :mod:`xrspatial.geotiff`, the canonical
internal names for the array-level write functions are
:func:`_write` and :func:`_write_streaming`. The non-underscored
:func:`write` and :func:`write_streaming` are kept as aliases for
internal call sites that pre-date the rename and may be removed in a
future release.
"""
from __future__ import annotations

import math
import struct
import warnings

import numpy as np

from ._compression import (
    COMPRESSION_DEFLATE,
    COMPRESSION_JPEG,
    COMPRESSION_JPEG2000,
    COMPRESSION_LERC,
    COMPRESSION_LZ4,
    COMPRESSION_LZW,
    COMPRESSION_NONE,
    COMPRESSION_PACKBITS,
    COMPRESSION_ZSTD,
    compress,
    fp_predictor_encode,
    jpeg_compress,
    predictor_encode,
)
from ._dtypes import (
    DOUBLE,
    RATIONAL,
    SHORT,
    LONG,
    ASCII,
    numpy_to_tiff_dtype,
    TIFF_TYPE_SIZES,
)
from ._geotags import (
    GeoTransform,
    build_geo_tags,
    TAG_GEO_ASCII_PARAMS,
    TAG_GEO_KEY_DIRECTORY,
    TAG_GDAL_NODATA,
    TAG_MODEL_PIXEL_SCALE,
    TAG_MODEL_TIEPOINT,
    TAG_MODEL_TRANSFORMATION,
)
from ._header import (
    TAG_NEW_SUBFILE_TYPE,
    TAG_IMAGE_WIDTH,
    TAG_IMAGE_LENGTH,
    TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION,
    TAG_PHOTOMETRIC,
    TAG_SAMPLES_PER_PIXEL,
    TAG_SAMPLE_FORMAT,
    TAG_STRIP_OFFSETS,
    TAG_ROWS_PER_STRIP,
    TAG_STRIP_BYTE_COUNTS,
    TAG_SUB_IFDS,
    TAG_X_RESOLUTION,
    TAG_Y_RESOLUTION,
    TAG_RESOLUTION_UNIT,
    TAG_TILE_WIDTH,
    TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
    TAG_EXTRA_SAMPLES,
    TAG_PREDICTOR,
    TAG_GDAL_METADATA,
)
# IFD-assembly and BigTIFF / COG layout helpers live in
# ``_write_layout.py``. Re-export them here so internal call sites
# (``_write``, ``_write_streaming``) and external importers (the
# ``_writers`` subpackage, tests, ``_gpu_decode``) keep using the
# ``xrspatial.geotiff._writer`` import path.
from ._write_layout import (  # noqa: F401
    BO,
    _BIGTIFF_OFFSET_TAGS,
    _assemble_cog_layout,
    _assemble_standard_layout,
    _assemble_tiff,
    _build_ifd,
    _compute_classic_ifd_overhead,
    _float_to_rational,
    _pack_tag_value,
    _promote_offsets_to_long8,
    _serialize_tag_value,
    _should_use_bigtiff_streaming,
)

# Tag IDs the writer must never accept from ``extra_tags``. NewSubfileType
# (254) is a per-IFD status flag the writer emits on its own for overview
# IFDs; copying a level-1 source value onto a level-0 destination would
# mis-mark the primary IFD as a reduced-resolution overview. SubIFDs
# (330) carries absolute byte offsets, which become garbage after a
# rewrite. The read side now filters both via ``_MANAGED_TAGS``; this
# constant is the writer-side belt-and-braces guard. See issue #1657.
_DANGEROUS_EXTRA_TAG_IDS = frozenset({TAG_NEW_SUBFILE_TYPE, TAG_SUB_IFDS})

# Tag IDs whose writer-auto value can be overridden by a user
# ``extra_tags`` entry of the same id. Restricted to the photometric
# interpretation tags so callers cannot accidentally clobber tags
# carrying computed offsets, dimensions, or layout. See issue #1769.
_OVERRIDABLE_AUTO_TAG_IDS = frozenset({TAG_PHOTOMETRIC, TAG_EXTRA_SAMPLES})

# TIFF Photometric Interpretation values (TIFF 6 spec, tag 262).
PHOTOMETRIC_MINISBLACK = 1
PHOTOMETRIC_RGB = 2

# Friendly names accepted by the ``photometric`` writer kwarg. ``'auto'``
# defaults to MinIsBlack so scientific multispectral rasters (e.g.
# R,G,B,NIR) round-trip without being silently tagged as RGB+alpha.
# ``'rgba'`` is a convenience for "RGB plus an unassociated-alpha extra
# sample".  See issue #1769.
_PHOTOMETRIC_NAME_MAP = {
    'auto': 'auto',
    'minisblack': PHOTOMETRIC_MINISBLACK,
    'miniswhite': 0,
    'rgb': PHOTOMETRIC_RGB,
    'rgba': 'rgba',
}


def _invert_nodata_for_miniswhite(nodata, dtype: np.dtype):
    """Invert a nodata sentinel for MinIsWhite writes.

    The reader's mask path (see ``_reader._miniswhite_inverted_nodata``)
    treats the stored sentinel as living in the on-disk display domain,
    so the writer pre-inverts the user-supplied sentinel alongside the
    pixels. After the writer pre-inversion, the on-disk sentinel byte
    matches the on-disk pixel byte that represents "missing", and the
    reader inverts both back to the user domain to drive the NaN mask.
    Returns ``nodata`` unchanged for signed integer pixels, NaN
    sentinels, and unsigned sentinels that are out-of-range or
    fractional or non-finite -- matching the reader exactly. ``+/-inf``
    on a float sentinel is negated (the reader does the same). See
    issue #1836.
    """
    if nodata is None:
        return nodata
    if dtype.kind == 'u':
        if not np.isfinite(nodata):
            return nodata
        if not float(nodata).is_integer():
            return nodata
        vi = int(nodata)
        info = np.iinfo(dtype)
        if not (info.min <= vi <= info.max):
            return nodata
        return info.max - vi
    if dtype.kind == 'f':
        if np.isnan(nodata):
            return nodata
        return -float(nodata)
    return nodata


def _apply_photometric_miniswhite_invert(
    arr: np.ndarray, resolved_photometric: int, samples_per_pixel: int,
) -> np.ndarray:
    """Mirror the reader's MinIsWhite inversion on the writer side.

    The reader unconditionally inverts single-band ``photometric == 0``
    data via ``_reader._apply_photometric_miniswhite``. Without a
    matching writer-side inversion, ``to_geotiff(da, photometric=
    'miniswhite')`` silently corrupts pixel values on the round trip.
    See issue #1836.

    Returns the pre-inverted array (a new array) so that the reader's
    inversion restores the original values. Multi-band data and signed
    integer data pass through unchanged, matching the reader.
    """
    if resolved_photometric != 0 or samples_per_pixel != 1:
        return arr
    if arr.dtype.kind == 'u':
        return np.iinfo(arr.dtype).max - arr
    if arr.dtype.kind == 'f':
        return -arr
    return arr


def _resolve_photometric(photometric, samples_per_pixel: int):
    """Resolve the ``photometric`` writer kwarg to a TIFF photometric int
    and the matching ExtraSamples list.

    Returns ``(photometric_int, extra_samples_list)`` where
    ``extra_samples_list`` has one entry per band beyond what the
    photometric model accounts for (empty when no ExtraSamples tag is
    needed).
    """
    if isinstance(photometric, str):
        key = photometric.lower()
        if key not in _PHOTOMETRIC_NAME_MAP:
            valid = sorted(_PHOTOMETRIC_NAME_MAP)
            raise ValueError(
                f"photometric={photometric!r} is not a valid name; "
                f"expected one of {valid} or an int.")
        resolved = _PHOTOMETRIC_NAME_MAP[key]
    elif isinstance(photometric, (int, np.integer)) and not isinstance(
            photometric, bool):
        resolved = int(photometric)
    else:
        raise TypeError(
            f"photometric must be a str or int, got "
            f"{type(photometric).__name__}")

    if resolved == 'auto':
        # MinIsBlack default for every band count. Scientific
        # multispectral rasters fall through here; previous behaviour
        # silently called any 4th band an alpha channel.
        n_extra = max(0, samples_per_pixel - 1)
        return PHOTOMETRIC_MINISBLACK, [0] * n_extra

    if resolved == 'rgba':
        if samples_per_pixel < 4:
            raise ValueError(
                f"photometric='rgba' requires at least 4 bands, got "
                f"samples_per_pixel={samples_per_pixel}.")
        n_extra = samples_per_pixel - 3
        return PHOTOMETRIC_RGB, [2] + [0] * (n_extra - 1)

    photo_int = int(resolved)
    if photo_int == PHOTOMETRIC_RGB and samples_per_pixel < 3:
        raise ValueError(
            f"photometric=RGB requires at least 3 bands, got "
            f"samples_per_pixel={samples_per_pixel}.")
    consumed = 3 if photo_int == PHOTOMETRIC_RGB else 1
    if samples_per_pixel > consumed:
        return photo_int, [0] * (samples_per_pixel - consumed)
    return photo_int, []


def _reject_disagreeing_photometric_override(
    extra_tags, resolved_photo: int, samples: int, photometric
) -> None:
    """Reject an ``extra_tags`` entry that overrides ``TAG_PHOTOMETRIC``
    across the MinIsWhite boundary for a single-band raster.

    The single-band MinIsWhite path requires the writer to pre-invert
    pixels (and the nodata sentinel) so the round-trip matches what the
    reader unconditionally inverts. An ``extra_tags`` entry that flips
    ``TAG_PHOTOMETRIC`` between MinIsWhite (0) and anything else makes
    the on-disk tag advertise one model while the bytes were
    pre-processed for the other -- the round-trip silently corrupts.

    The eager and streaming writers both call this guard before any
    pre-inversion runs. Only the MinIsWhite-crossing single-band case
    is rejected; multi-band rasters and non-crossing overrides (e.g.
    photometric='minisblack' with extra_tags=[(262, SHORT, 1, 1)])
    pass through unchanged. Issues #2073 / #1769 / #1836.
    """
    if extra_tags is None:
        return
    override = None
    for _et in extra_tags:
        if _et[0] == TAG_PHOTOMETRIC:
            override = int(_et[3])
            break
    if override is None:
        return
    if override == resolved_photo:
        return
    if not (override == 0 or resolved_photo == 0):
        return
    if samples != 1:
        return
    raise ValueError(
        f"extra_tags TAG_PHOTOMETRIC override ({override}) "
        f"disagrees with photometric={photometric!r} for a "
        f"single-band raster where MinIsWhite (photometric=0) "
        f"requires writer-side pixel inversion. The override would "
        f"either pre-invert pixels for a non-MinIsWhite tag or skip "
        f"inversion for a MinIsWhite tag. Pass photometric= directly "
        f"instead, or drop the override."
    )


def normalize_predictor(predictor, dtype, compression: int) -> int:
    """Normalize a user-supplied predictor value to a TIFF predictor int.

    Accepts ``False``/``True`` (legacy) and integers ``1``/``2``/``3``.
    Returns ``1`` (no predictor), ``2`` (horizontal differencing), or ``3``
    (floating-point predictor).
    """
    if predictor is False or predictor == 0:
        return 1
    if predictor is True or predictor == 2:
        return 2
    if predictor == 1:
        return 1
    if predictor == 3:
        if np.dtype(dtype).kind != 'f':
            raise ValueError(
                "predictor=3 (floating-point) requires float data, "
                f"got dtype={np.dtype(dtype)}")
        return 3
    raise ValueError(
        f"predictor must be False/True or 1/2/3, got {predictor!r}")


def _apply_predictor_encode(buf: np.ndarray, predictor: int,
                            width: int, height: int,
                            bytes_per_sample: int, samples: int) -> np.ndarray:
    """Apply the chosen predictor to a flat uint8 buffer.

    Files always go to disk in little-endian order (see ``BO``), so
    ``predictor_encode`` is invoked with ``byte_order='<'``.
    """
    if predictor == 2:
        return predictor_encode(buf, width, height,
                                bytes_per_sample, samples=samples,
                                byte_order=BO)
    if predictor == 3:
        return fp_predictor_encode(buf, width * samples, height,
                                   bytes_per_sample)
    return buf


def _compression_tag(compression_name: str) -> int:
    """Convert compression name to TIFF tag value."""
    _map = {
        'none': COMPRESSION_NONE,
        'deflate': COMPRESSION_DEFLATE,
        'lzw': COMPRESSION_LZW,
        'jpeg': COMPRESSION_JPEG,
        'packbits': COMPRESSION_PACKBITS,
        'zstd': COMPRESSION_ZSTD,
        'lz4': COMPRESSION_LZ4,
        'jpeg2000': COMPRESSION_JPEG2000,
        'j2k': COMPRESSION_JPEG2000,
        'lerc': COMPRESSION_LERC,
    }
    name = compression_name.lower()
    if name not in _map:
        raise ValueError(f"Unsupported compression: {compression_name!r}. "
                         f"Use one of: {list(_map.keys())}")
    return _map[name]


OVERVIEW_METHODS = ('mean', 'nearest', 'min', 'max', 'median', 'mode', 'cubic')

#: Maximum number of overview levels generated by auto-overview mode in COG
#: writes. 8 halvings = 1/256 of the original resolution, which is enough
#: for any practical raster. Pass ``overview_levels=[...]`` explicitly to
#: override.
_MAX_OVERVIEW_LEVELS = 8

#: Total uncompressed payload (bytes) below which the strip and tile
#: writers stay sequential. The thread-pool startup cost dominates on
#: small rasters; above this size the per-block compression cost more
#: than pays for it. 4 MiB was chosen empirically on a 20-core box:
#: parallel becomes a net win around ~2 MiB, and the 4 MiB margin keeps
#: a few-tile / two-strip layout from incurring a slowdown.
_PARALLEL_MIN_BYTES = 4 * 1024 * 1024


def _validate_overview_levels(overview_levels, height=None, width=None):
    """Validate and normalise an explicit ``overview_levels`` list.

    Each entry is a decimation factor relative to full resolution.
    Factors must be strictly increasing integers >= 2, and each must
    be a power of two because the underlying block reducer only does
    2x decimation per step (issue #1766 — prior to that fix, the values
    were ignored and only the list length mattered).

    When ``height`` and ``width`` are supplied, each factor is also
    checked against the input shape: a factor F is feasible only if
    ``height // F >= 1`` and ``width // F >= 1``. Asking for a factor
    that would reduce the source below 1 pixel in either dimension
    raises ``ValueError`` instead of silently writing a zero-sized
    overview IFD.

    Returns the validated list of ints. ``None`` passes through so the
    caller can run its auto-generation path.
    """
    if overview_levels is None:
        return None
    if not isinstance(overview_levels, (list, tuple)):
        raise ValueError(
            f"overview_levels must be a list or tuple of ints, got "
            f"{type(overview_levels).__name__}.")
    if len(overview_levels) == 0:
        return []
    cleaned = []
    prev = 1
    for i, level in enumerate(overview_levels):
        # Reject bools explicitly; ``bool`` is an ``int`` subclass and
        # ``True``/``False`` would otherwise sneak through the integer
        # check below.
        if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
            raise ValueError(
                f"overview_levels[{i}] must be an int >= 2, got "
                f"{level!r} (type {type(level).__name__}).")
        level = int(level)
        if level < 2:
            raise ValueError(
                f"overview_levels[{i}] must be >= 2 (1 is the original "
                f"full-resolution band), got {level}.")
        if level <= prev:
            raise ValueError(
                f"overview_levels must be strictly increasing, got "
                f"{list(overview_levels)} (entry at index {i} is "
                f"{level}, previous was {prev}).")
        # Power-of-two check: the underlying ``_make_overview`` only
        # halves, so reaching factor N takes log2(N) halvings and N
        # must be a power of two for the cumulative factor to land on
        # the requested value exactly.
        if (level & (level - 1)) != 0:
            raise ValueError(
                f"overview_levels[{i}]={level} is not a power of two. "
                f"Only power-of-two decimation factors are supported "
                f"(2, 4, 8, 16, ...).")
        # Shape feasibility: refuse factors that would shrink the
        # raster below 1 pixel. ``_block_reduce_2d`` halves via
        # ``(dim // 2) * 2`` which produces a zero-sized array once
        # ``dim < 2``, and chaining further halvings keeps it at zero.
        # Without this check the writer silently emits zero-sized
        # overview IFDs.
        if height is not None and width is not None:
            if height // level < 1 or width // level < 1:
                raise ValueError(
                    f"overview_levels[{i}]={level} is too large for "
                    f"input shape ({height}, {width}); decimation "
                    f"would produce a zero-sized overview.")
        cleaned.append(level)
        prev = level
    return cleaned


def _resolve_int_nodata(dtype, nodata):
    """Return ``int(nodata)`` if it is representable as *dtype*, else None.

    Folds the three checks that gate the integer sentinel-to-NaN mask in
    :func:`_block_reduce_2d` into one call: ``dtype`` is integer, the
    sentinel is finite and integer-valued, and the integer fits the
    dtype range. Out-of-range pairs like ``uint16`` + ``GDAL_NODATA=-9999``
    return None so the caller stays a no-op rather than tripping
    ``OverflowError`` on the dtype cast. Mirrors
    ``_int_nodata_in_range`` in ``_reader.py``.
    """
    if nodata is None or dtype.kind not in ('i', 'u'):
        return None
    if not np.isfinite(nodata) or not float(nodata).is_integer():
        return None
    nodata_int = int(nodata)
    info = np.iinfo(dtype)
    if info.min <= nodata_int <= info.max:
        return nodata_int
    return None


def _replicate_pad_2d(src, h2, w2):
    """Pad ``src`` to ``(h2, w2)`` by replicating its trailing row/col.

    Used by the cubic overview branch to extend an odd-sized input to
    even dimensions so a 0.5-zoom lands on the GDAL ceil shape and the
    spline kernel sees a valid neighbourhood at the trailing edge. The
    helper is dtype-preserving (no float promotion) so integer inputs
    stay integer through the pad. Returns ``src`` unchanged when no
    padding is needed.
    """
    h, w = src.shape
    if (h, w) == (h2, w2):
        return src
    padded = np.empty((h2, w2), dtype=src.dtype)
    padded[:h, :w] = src
    if h2 != h:
        padded[h:, :w] = src[h - 1:h, :]
    if w2 != w:
        padded[:h, w:] = src[:, w - 1:w]
    if h2 != h and w2 != w:
        padded[h:, w:] = src[h - 1, w - 1]
    return padded


def _block_reduce_2d(arr2d, method, nodata=None):
    """2x block-reduce a single 2D plane using *method*.

    Output dimensions follow GDAL's ceil semantics (5x5 -> 3x3): every
    base pixel contributes to the overview. Inputs with an odd height
    or width are padded with NaN (or float-promoted and NaN-padded for
    integer dtypes) along the trailing edge so the 2x2 block reshape
    still works; the nan-aware aggregations then exclude the padded
    cells from the residual blocks. ``nearest`` strides directly,
    ``cubic`` replicates the last row/col so the spline kernel sees a
    valid neighbourhood at the trailing edge, and ``mode`` ignores
    NaN-padded cells when counting. Even-sized inputs take the
    existing fast path with no padding overhead (issue #2105).

    When ``nodata`` is supplied, cells that equal the sentinel are
    treated as NaN during the reduction so the ``nan*`` aggregation
    routines correctly skip them. The float branch keeps any all-
    sentinel block as NaN so the caller's post-overview loop can
    rewrite it back to the sentinel; the integer branch rewrites NaN
    back to the sentinel before the dtype cast so the cast is
    well-defined (the caller's post-overview loop only handles the
    float case). The ``nearest`` and ``mode`` methods do NOT mask the
    sentinel: ``nearest`` returns the top-left pixel of each 2x2 block
    and ``mode`` returns the most-frequent value, so the sentinel can
    be selected as the overview pixel if it occupies that position
    (``nearest``) or is the most frequent value in the block
    (``mode``). Mean / median / min / max / cubic all mask the
    sentinel before reduction. The ``cubic`` branch honours ``nodata``
    by masking the sentinel to NaN, running cubic with
    ``prefilter=False`` to keep the kernel local, and rewriting any
    NaN in the output back to the sentinel before returning (issue
    #1623). Cubic on integer dtypes follows the same path via a
    float64 promotion so NaN can carry through the spline, with a
    ``np.round(...).astype(arr2d.dtype)`` at the end to keep the cast
    well-defined (issue #1975).
    """
    h, w = arr2d.shape
    # GDAL-style ceil semantics keep the trailing row/col when the input
    # is odd-sized. Floor-and-crop drops those pixels and produces an
    # overview that no longer covers the source extent (issue #2105).
    oh, ow = (h + 1) // 2, (w + 1) // 2
    h2, w2 = 2 * oh, 2 * ow
    need_pad = (h2, w2) != (h, w)

    if method == 'nearest':
        # Top-left pixel of each 2x2 block. For odd-sized inputs the
        # trailing 1x2 / 2x1 / 1x1 residual block's top-left is still
        # the source pixel at (2*i, 2*j), so direct striding gives the
        # correct ceil-shape output without padding. The trailing
        # ``.copy()`` materialises the strided view so the caller owns
        # a contiguous buffer (downstream nodata-mask rewrites mutate
        # the result in place); without it those writes would also
        # touch ``arr2d``.
        return arr2d[::2, ::2].copy()

    if method == 'cubic':
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError(
                "scipy is required for cubic overview resampling. "
                "Install it with: pip install scipy")
        # When ``nodata`` is supplied on a float array, the writer has
        # already rewritten NaN to the sentinel value upstream. Feeding
        # that sentinel-poisoned array straight into ``zoom`` blends the
        # sentinel into neighbouring cells and produces ringing
        # artefacts near nodata borders (issue #1623, same root cause
        # as #1613 but for the cubic branch).
        #
        # Mask the sentinel back to NaN before the spline so the
        # interpolation does not treat it as signal, run cubic with
        # ``prefilter=False`` so a single NaN does not poison the entire
        # row/column (the default B-spline prefilter is global), then
        # rewrite any NaN in the result back to the sentinel so the
        # on-disk overview keeps the same convention as the
        # full-resolution band. The ``prefilter=False`` switch only
        # fires when a sentinel was actually found in the input, so the
        # default cubic semantics still apply to inputs without nodata.
        #
        # Integer rasters take the same path via a float64 promotion so
        # NaN can carry through the spline; the result is rewritten
        # back to the sentinel and rounded before casting to the source
        # integer dtype (issue #1975, integer mirror of #1623).
        #
        # Odd-sized inputs are first edge-replicated to even dimensions
        # so the 0.5 zoom factor lands on the GDAL ceil shape and the
        # spline sees a valid neighbourhood at the trailing edge.
        # Replicating sentinel values is fine: they are masked to NaN
        # in the steps below alongside any in-bounds sentinel, and the
        # post-zoom NaN->sentinel restore stamps the trailing overview
        # pixels back to the sentinel as expected (issue #2105). The
        # ``_replicate_pad_2d`` helper is module-level so the cubic
        # path does not redefine a closure on every call.

        if (nodata is not None
                and arr2d.dtype.kind == 'f'
                and not np.isnan(nodata)):
            try:
                sentinel = arr2d.dtype.type(nodata)
            except (OverflowError, ValueError):
                sentinel = None
            if sentinel is not None:
                mask = arr2d == sentinel
                if mask.any():
                    src = _replicate_pad_2d(arr2d, h2, w2)
                    src_mask = src == sentinel
                    masked = np.where(src_mask, np.float64('nan'), src)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        result = zoom(masked, 0.5, order=3,
                                      prefilter=False)
                    nan_mask = np.isnan(result)
                    if nan_mask.any():
                        result = result.copy()
                        result[nan_mask] = float(nodata)
                    return result.astype(arr2d.dtype)
                # No sentinel pixels present; fall through to the
                # default cubic path below so the spline runs without
                # the prefilter=False NaN-safety branch.
        nodata_int = _resolve_int_nodata(arr2d.dtype, nodata)
        if nodata_int is not None:
            sentinel = arr2d.dtype.type(nodata_int)
            mask = arr2d == sentinel
            if mask.any():
                src = _replicate_pad_2d(arr2d, h2, w2)
                src_mask = src == sentinel
                masked = np.where(src_mask, np.float64('nan'),
                                  src.astype(np.float64))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    result = zoom(masked, 0.5, order=3,
                                  prefilter=False)
                nan_mask = np.isnan(result)
                if nan_mask.any():
                    result = np.where(nan_mask, float(nodata_int),
                                      result)
                return np.round(result).astype(arr2d.dtype)
            # No sentinel pixels present in this integer array; fall
            # through to the default cubic path below.
        src = _replicate_pad_2d(arr2d, h2, w2)
        return zoom(src, 0.5, order=3).astype(arr2d.dtype)

    if method == 'mode':
        # Most-common value per 2x2 block (useful for classified rasters).
        # Vectorized: sort each 4-cell block, then for each position count
        # how many cells equal it. argmax picks the leftmost max-count
        # position, which (post-sort) is the smallest tied value, matching
        # the prior np.unique+argmax tie-break behavior ("lowest wins").
        #
        # Odd-sized inputs are float-promoted and NaN-padded so the
        # residual block has at least one valid cell plus NaN cells that
        # never match themselves (NaN != NaN). Counts for the NaN cells
        # stay at zero, so argmax picks a valid cell, and numpy's sort
        # places NaN at the end so the "lowest valid wins" tie-break
        # still holds (issue #2105). The result is cast back to the
        # source dtype.
        if need_pad:
            padded = np.full((h2, w2), np.nan, dtype=np.float64)
            padded[:h, :w] = arr2d
            blocks = padded.reshape(oh, 2, ow, 2).transpose(0, 2, 1, 3).reshape(oh, ow, 4)
            srt = np.sort(blocks, axis=-1)
            counts = np.zeros(srt.shape, dtype=np.int8)
            for i in range(4):
                counts[..., i] = np.sum(srt == srt[..., i:i + 1], axis=-1)
            pick = np.argmax(counts, axis=-1)
            picked = np.take_along_axis(srt, pick[..., None], axis=-1).squeeze(-1)
            if arr2d.dtype.kind in ('i', 'u'):
                return np.round(picked).astype(arr2d.dtype)
            return picked.astype(arr2d.dtype)
        blocks = arr2d.reshape(oh, 2, ow, 2).transpose(0, 2, 1, 3).reshape(oh, ow, 4)
        srt = np.sort(blocks, axis=-1)
        counts = np.empty_like(srt, dtype=np.int8)
        for i in range(4):
            counts[..., i] = np.sum(srt == srt[..., i:i + 1], axis=-1)
        pick = np.argmax(counts, axis=-1)
        return np.take_along_axis(srt, pick[..., None], axis=-1).squeeze(-1)

    # Block reshape for mean/min/max/median. Odd-sized inputs are padded
    # with NaN along the trailing edge so the reshape works on every
    # input size; the padded cells are naturally excluded by the
    # nan-aware aggregations below (issue #2105). The padded buffer
    # keeps the source's float dtype (float32 inputs stay float32) so
    # an odd-shape read does not pay a 2x intermediate-memory cost for
    # a silent promote to float64.
    if arr2d.dtype.kind == 'f':
        if need_pad:
            padded = np.full((h2, w2), np.nan, dtype=arr2d.dtype)
            padded[:h, :w] = arr2d
            blocks = padded.reshape(oh, 2, ow, 2)
        else:
            blocks = arr2d.reshape(oh, 2, ow, 2)
        # When a sentinel was used in place of NaN by an upstream
        # NaN-to-sentinel rewrite, mask it back to NaN here so nanmean /
        # nanmin / nanmax / nanmedian honour the missing-data semantic.
        # Without this the sentinel value participates in the reduction
        # and poisons the overview (issue #1613). Match the upstream
        # NaN->sentinel rewrite gate (``not np.isnan(nodata)``) so that
        # ``nodata=+/-inf`` is masked here too.
        if nodata is not None and not np.isnan(nodata):
            try:
                sentinel = arr2d.dtype.type(nodata)
            except (OverflowError, ValueError):
                sentinel = None
            if sentinel is not None:
                mask = blocks == sentinel
                if mask.any():
                    # ``np.where(mask, nan, blocks)`` produces a fresh
                    # array so the caller's input is not mutated.
                    blocks = np.where(mask, np.float64('nan'), blocks)
    else:
        if need_pad:
            blocks = np.full((h2, w2), np.nan, dtype=np.float64)
            blocks[:h, :w] = arr2d
            blocks = blocks.reshape(oh, 2, ow, 2)
        else:
            blocks = arr2d.astype(np.float64).reshape(oh, 2, ow, 2)
        # Integer rasters with a sentinel still need the NaN-mask the
        # float branch above applies. The sentinel-equal cells in the
        # original integer array are computed at the dtype's native
        # width below (so a 64-bit sentinel near INT64_MAX still
        # matches, which a float-cast comparison would miss) and the
        # mask is then broadcast into the float64 ``blocks`` view.
        # Without this, nanmean / nanmin / nanmax / nanmedian average
        # the sentinel value into surrounding valid cells and produce
        # overview pixels that the read side cannot remap because they
        # no longer equal the sentinel. Gate on the sentinel being
        # representable in the source integer dtype (mirrors
        # ``_int_nodata_in_range`` in ``_reader.py``) so an out-of-
        # range sentinel pair like ``uint16`` + ``GDAL_NODATA=-9999``
        # stays a no-op rather than tripping ``OverflowError`` on the
        # dtype cast.
        nodata_int = _resolve_int_nodata(arr2d.dtype, nodata)
        if nodata_int is not None:
            sentinel = arr2d.dtype.type(nodata_int)
            if need_pad:
                # Compute the sentinel mask at the integer's native width
                # before padding, so a 64-bit sentinel near INT64_MAX (which
                # is not exactly representable in float64) still matches.
                # The mask is then padded with False to the same (h2, w2)
                # shape as the float-promoted blocks view, so the residual
                # padded cells (already NaN in ``blocks``) are not also
                # rewritten via the where below.
                int_mask = np.zeros((h2, w2), dtype=bool)
                int_mask[:h, :w] = arr2d == sentinel
                mask = int_mask.reshape(oh, 2, ow, 2)
            else:
                # Compare against the original integer block view so the
                # equality runs at the integer's native width (avoids any
                # float-cast rounding on adjacent values). The boolean
                # mask broadcasts into the float64 block layout below.
                int_blocks = arr2d.reshape(oh, 2, ow, 2)
                mask = int_blocks == sentinel
            if mask.any():
                blocks = np.where(mask, np.float64('nan'), blocks)

    # nanmean / nanmin / nanmax / nanmedian emit RuntimeWarning when a
    # 2x2 block is all-NaN (typical at nodata borders). The all-NaN
    # output is the desired signal that the caller rewrites to the
    # sentinel, so suppress the warning locally to keep COG writes quiet.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if method == 'mean':
            result = np.nanmean(blocks, axis=(1, 3))
        elif method == 'min':
            result = np.nanmin(blocks, axis=(1, 3))
        elif method == 'max':
            result = np.nanmax(blocks, axis=(1, 3))
        elif method == 'median':
            flat = blocks.transpose(0, 2, 1, 3).reshape(oh, ow, 4)
            result = np.nanmedian(flat, axis=2)
        else:
            raise ValueError(
                f"Unknown overview resampling method: {method!r}. "
                f"Use one of: {OVERVIEW_METHODS}")

    if arr2d.dtype.kind != 'f':
        # All-sentinel 2x2 blocks come back as NaN from the nan-aware
        # reduction; cast NaN to an integer dtype is undefined (varies
        # between platforms / produces zero or INT_MIN). Rewrite those
        # back to the sentinel before the cast so the integer overview
        # pyramid carries the same masking convention as the
        # full-resolution band. The float branch relies on the caller's
        # post-overview rewrite in ``write()``; integer dtypes skip that
        # branch because ``current.dtype.kind == 'f'`` is False, so we
        # close the loop here.
        nan_mask = np.isnan(result)
        if nan_mask.any():
            nodata_int = _resolve_int_nodata(arr2d.dtype, nodata)
            if nodata_int is not None:
                result = np.where(nan_mask, float(nodata_int), result)
        return np.round(result).astype(arr2d.dtype)
    return result.astype(arr2d.dtype)


def _make_overview(arr: np.ndarray, method: str = 'mean',
                   nodata=None) -> np.ndarray:
    """Generate a 2x decimated overview.

    Parameters
    ----------
    arr : np.ndarray
        2D or 3D (height, width, bands) array.
    method : str
        Resampling method: 'mean' (default), 'nearest', 'min', 'max',
        'median', 'mode', or 'cubic'.
    nodata : scalar or None
        When supplied, cells equal to the sentinel are masked back to
        NaN before the reduction so the sentinel does not bias the
        result. Applies to both float dtypes (issue #1613, extended to
        ``cubic`` in #1623) and integer dtypes (the mean / min / max /
        median reductions used to average the sentinel into surrounding
        valid pixels and produce overview values that the reader could
        not mask). Ignored for ``nearest`` / ``mode`` methods (no
        averaging occurs).

    Returns
    -------
    np.ndarray
        Half-resolution array.
    """
    if arr.ndim == 3:
        bands = [_block_reduce_2d(arr[:, :, b], method, nodata=nodata)
                 for b in range(arr.shape[2])]
        return np.stack(bands, axis=2)
    return _block_reduce_2d(arr, method, nodata=nodata)


# ---------------------------------------------------------------------------
# Tag serialization helpers (``_float_to_rational``, ``_serialize_tag_value``,
# ``_pack_tag_value``, ``_build_ifd``) live in ``_write_layout.py`` and are
# re-exported above for backwards compatibility.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Strip writer
# ---------------------------------------------------------------------------

def _prepare_strip(data, i, rows_per_strip, height, width, samples, dtype,
                   bytes_per_sample, predictor: int, compression,
                   compression_level=None, max_z_error: float = 0.0,
                   gil_friendly: bool = False):
    """Extract and compress a single strip. Thread-safe."""
    r0 = i * rows_per_strip
    r1 = min(r0 + rows_per_strip, height)
    strip_rows = r1 - r0

    if compression == COMPRESSION_JPEG:
        strip_data = np.ascontiguousarray(data[r0:r1]).tobytes()
        return jpeg_compress(strip_data, width, strip_rows, samples)
    if predictor != 1 and compression != COMPRESSION_NONE:
        strip_arr = np.ascontiguousarray(data[r0:r1])
        buf = strip_arr.view(np.uint8).ravel().copy()
        buf = _apply_predictor_encode(
            buf, predictor, width, strip_rows, bytes_per_sample, samples)
        strip_data = buf.tobytes()
    else:
        strip_data = np.ascontiguousarray(data[r0:r1]).tobytes()

    if compression == COMPRESSION_JPEG2000:
        from ._compression import jpeg2000_compress
        return jpeg2000_compress(
            strip_data, width, strip_rows, samples=samples, dtype=dtype)
    if compression == COMPRESSION_LERC:
        from ._compression import lerc_compress
        return lerc_compress(
            strip_data, width, strip_rows, samples=samples, dtype=dtype,
            max_z_error=max_z_error)
    if compression_level is None:
        return compress(strip_data, compression, gil_friendly=gil_friendly)
    return compress(strip_data, compression, level=compression_level,
                    gil_friendly=gil_friendly)


def _write_stripped(data: np.ndarray, compression: int, predictor: int,
                    rows_per_strip: int = 256,
                    compression_level: int | None = None,
                    max_z_error: float = 0.0) -> tuple[list, list, list]:
    """Compress data as strips.

    For compressed formats (deflate, lzw, zstd, lz4, ...) strips are
    compressed in parallel using a thread pool: zlib, zstandard, lz4,
    and the Numba LZW kernel all release the GIL during compression.

    Returns
    -------
    (offsets_placeholder, byte_counts, compressed_chunks)
        offsets are relative to the start of the compressed data block.
        compressed_chunks is a list of bytes objects (one per strip).
    """
    height, width = data.shape[:2]
    samples = data.shape[2] if data.ndim == 3 else 1
    dtype = data.dtype
    bytes_per_sample = dtype.itemsize

    num_strips = math.ceil(height / rows_per_strip)

    total_bytes = int(data.nbytes)

    # Sequential path: uncompressed, few strips, or small payload.  The
    # threshold mirrors the tile writer so we don't pay thread-pool
    # overhead on tiny rasters.
    use_parallel = (
        compression != COMPRESSION_NONE
        and num_strips > 2
        and total_bytes > _PARALLEL_MIN_BYTES
    )

    if not use_parallel:
        strips = []
        rel_offsets = []
        byte_counts = []
        current_offset = 0
        for i in range(num_strips):
            compressed = _prepare_strip(
                data, i, rows_per_strip, height, width, samples, dtype,
                bytes_per_sample, predictor, compression,
                compression_level, max_z_error,
            )
            rel_offsets.append(current_offset)
            byte_counts.append(len(compressed))
            strips.append(compressed)
            current_offset += len(compressed)
        return rel_offsets, byte_counts, strips

    # Parallel strip compression -- zlib/zstd/lz4/LZW all release the GIL.
    # ``gil_friendly=True`` keeps deflate on stdlib zlib here: the
    # ``deflate`` (libdeflate) binding holds the GIL during compress, so
    # 8 threads run effectively serially through it. Sequential callers
    # still get libdeflate's per-call speedup (~3x).
    from concurrent.futures import ThreadPoolExecutor
    import os

    n_workers = min(num_strips, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        compressed_strips = list(pool.map(
            lambda i: _prepare_strip(
                data, i, rows_per_strip, height, width, samples, dtype,
                bytes_per_sample, predictor, compression,
                compression_level, max_z_error,
                gil_friendly=True,
            ),
            range(num_strips),
        ))

    rel_offsets = []
    byte_counts = []
    current_offset = 0
    for cs in compressed_strips:
        rel_offsets.append(current_offset)
        byte_counts.append(len(cs))
        current_offset += len(cs)

    return rel_offsets, byte_counts, compressed_strips


# ---------------------------------------------------------------------------
# Tile writer
# ---------------------------------------------------------------------------

def _prepare_tile(data, tr, tc, th, tw, height, width, samples, dtype,
                  bytes_per_sample, predictor: int, compression,
                  compression_level=None, max_z_error: float = 0.0,
                  gil_friendly: bool = False):
    """Extract, pad, and compress a single tile.  Thread-safe."""
    r0 = tr * th
    c0 = tc * tw
    r1 = min(r0 + th, height)
    c1 = min(c0 + tw, width)
    actual_h = r1 - r0
    actual_w = c1 - c0

    tile_slice = data[r0:r1, c0:c1]

    if actual_h < th or actual_w < tw:
        if data.ndim == 3:
            padded = np.empty((th, tw, samples), dtype=dtype)
        else:
            padded = np.empty((th, tw), dtype=dtype)
        padded[:actual_h, :actual_w] = tile_slice
        if actual_h < th:
            padded[actual_h:, :] = 0
        if actual_w < tw:
            padded[:actual_h, actual_w:] = 0
        tile_arr = padded
    else:
        tile_arr = np.ascontiguousarray(tile_slice)

    if compression == COMPRESSION_JPEG:
        tile_data = tile_arr.tobytes()
        return jpeg_compress(tile_data, tw, th, samples)
    elif predictor != 1 and compression != COMPRESSION_NONE:
        buf = tile_arr.view(np.uint8).ravel().copy()
        buf = _apply_predictor_encode(
            buf, predictor, tw, th, bytes_per_sample, samples)
        tile_data = buf.tobytes()
    else:
        tile_data = tile_arr.tobytes()

    if compression == COMPRESSION_JPEG2000:
        from ._compression import jpeg2000_compress
        return jpeg2000_compress(
            tile_data, tw, th, samples=samples, dtype=dtype)
    if compression == COMPRESSION_LERC:
        from ._compression import lerc_compress
        return lerc_compress(
            tile_data, tw, th, samples=samples, dtype=dtype,
            max_z_error=max_z_error)
    if compression_level is None:
        return compress(tile_data, compression, gil_friendly=gil_friendly)
    return compress(tile_data, compression, level=compression_level,
                    gil_friendly=gil_friendly)


def _write_tiled(data: np.ndarray, compression: int, predictor: int,
                 tile_size: int = 256,
                 compression_level: int | None = None,
                 max_z_error: float = 0.0) -> tuple[list, list, list]:
    """Compress data as tiles, using parallel compression.

    For compressed formats (deflate, lzw, zstd), tiles are compressed
    in parallel using a thread pool.  zlib, zstandard, and our Numba
    LZW all release the GIL.

    Returns
    -------
    (relative_offsets, byte_counts, compressed_chunks)
        compressed_chunks is a list of bytes objects (one per tile).
    """
    height, width = data.shape[:2]
    samples = data.shape[2] if data.ndim == 3 else 1
    dtype = data.dtype
    bytes_per_sample = dtype.itemsize

    tw = tile_size
    th = tile_size
    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)
    n_tiles = tiles_across * tiles_down

    if compression == COMPRESSION_NONE:
        # Uncompressed: build tiles one at a time. An earlier version
        # pre-allocated a contiguous ``bytearray(n_tiles * tile_bytes)``
        # buffer here on the theory that we'd copy each tile into it
        # directly, but the loop below ended up calling ``tobytes()``
        # per tile anyway and never read the buffer. That left a dead
        # allocation roughly the size of the full uncompressed raster
        # alongside the actual tile list, doubling peak memory and
        # turning OOM-marginal writes into OOM-failing ones (#1736).
        tiles = []
        rel_offsets = []
        byte_counts = []
        current_offset = 0

        for tr in range(tiles_down):
            for tc in range(tiles_across):
                r0 = tr * th
                c0 = tc * tw
                r1 = min(r0 + th, height)
                c1 = min(c0 + tw, width)
                actual_h = r1 - r0
                actual_w = c1 - c0

                tile_slice = data[r0:r1, c0:c1]
                if actual_h < th or actual_w < tw:
                    if data.ndim == 3:
                        padded = np.zeros((th, tw, samples), dtype=dtype)
                    else:
                        padded = np.zeros((th, tw), dtype=dtype)
                    padded[:actual_h, :actual_w] = tile_slice
                    tile_arr = padded
                else:
                    tile_arr = np.ascontiguousarray(tile_slice)

                chunk = tile_arr.tobytes()
                rel_offsets.append(current_offset)
                byte_counts.append(len(chunk))
                tiles.append(chunk)
                current_offset += len(chunk)

        return rel_offsets, byte_counts, tiles

    # Sequential path: very few tiles, or small total payload. A previous
    # ``n_tiles <= 4`` cutoff sent ``tile_size=1024`` writes on a 2048x2048
    # image down the serial path (n_tiles=4) and made them ~8x slower than
    # the parallel path. Switching to a bytes-based threshold lets
    # large-tile writes parallelize while still skipping the pool on
    # small rasters where its setup cost dominates.
    if n_tiles <= 2 or int(data.nbytes) <= _PARALLEL_MIN_BYTES:
        tiles = []
        rel_offsets = []
        byte_counts = []
        current_offset = 0
        for tr in range(tiles_down):
            for tc in range(tiles_across):
                compressed = _prepare_tile(
                    data, tr, tc, th, tw, height, width,
                    samples, dtype, bytes_per_sample, predictor, compression,
                    compression_level, max_z_error,
                )
                rel_offsets.append(current_offset)
                byte_counts.append(len(compressed))
                tiles.append(compressed)
                current_offset += len(compressed)
        return rel_offsets, byte_counts, tiles

    # Parallel tile compression -- zlib/zstd/LZW all release the GIL
    from concurrent.futures import ThreadPoolExecutor
    import os

    n_workers = min(n_tiles, os.cpu_count() or 4)
    tile_indices = [(tr, tc) for tr in range(tiles_down)
                    for tc in range(tiles_across)]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _prepare_tile, data, tr, tc, th, tw, height, width,
                samples, dtype, bytes_per_sample, predictor, compression,
                compression_level, max_z_error, True,
            )
            for tr, tc in tile_indices
        ]
        compressed_tiles = [f.result() for f in futures]

    rel_offsets = []
    byte_counts = []
    current_offset = 0
    for ct in compressed_tiles:
        rel_offsets.append(current_offset)
        byte_counts.append(len(ct))
        current_offset += len(ct)

    return rel_offsets, byte_counts, compressed_tiles


# ---------------------------------------------------------------------------
# File assembly (``_assemble_tiff``, ``_assemble_standard_layout``,
# ``_assemble_cog_layout``, ``_promote_offsets_to_long8``, and
# ``_BIGTIFF_OFFSET_TAGS``) lives in ``_write_layout.py`` and is
# re-exported above for backwards compatibility.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Array-level write entry points (module-private; see module docstring)
# ---------------------------------------------------------------------------


def _validate_lowlevel_write_kwargs(*,
                                    compression,
                                    allow_internal_only_jpeg: bool,
                                    max_z_error,
                                    crs_epsg,
                                    crs_wkt,
                                    allow_unparseable_crs: bool,
                                    entry_point: str) -> None:
    """Push-down byte-affecting validation for ``_write`` / ``_write_streaming``.

    Centralises the checks that the public :func:`to_geotiff` wrapper
    runs but that the array-level entry points used to skip, so a
    direct caller cannot quietly produce a different file. See issue
    #2138 for the gap inventory. The checks are byte-affecting: each
    of them changes the bytes on disk (or refuses to write garbage),
    so they belong with the array-level writer rather than the
    DataArray wrapper.

    Parameters
    ----------
    compression : str or other
        Codec name. Validated against :data:`_VALID_COMPRESSIONS` and
        the JPEG-in-TIFF opt-in gate. Non-string values are not
        rejected here; ``_compression_tag`` raises downstream.
    allow_internal_only_jpeg : bool
        If False (the default), ``compression='jpeg'`` is rejected
        because the encoder writes JFIF tiles without the
        ``JPEGTables`` tag (issue #1845).
    max_z_error : float
        Per-pixel LERC error budget. Must be ``>= 0`` and is only
        meaningful with ``compression='lerc'``.
    crs_epsg : int or None
        EPSG kwarg. ``bool`` is rejected because it would otherwise
        be silently written as ``EPSG=1`` / ``EPSG=0``.
    crs_wkt : str or None
        WKT fallback. Validated against the structural ``_looks_like_wkt``
        gate unless ``allow_unparseable_crs=True``.
    allow_unparseable_crs : bool
        Opt-in to keep pre-#1929 behaviour for unparseable CRS strings.
    entry_point : str
        Name of the calling function ('_write' or '_write_streaming'),
        used in error messages so the source of the rejection is clear.
    """
    from ._attrs import _VALID_COMPRESSIONS
    from ._crs import _validate_crs_fallback

    # Gap #1: compression name validation against the canonical list.
    # ``_compression_tag`` already raises on unknown names with a list,
    # but only ``str`` inputs reach it; non-string values would land in
    # ``compression_name.lower()`` and surface as ``AttributeError``.
    if isinstance(compression, str):
        if compression.lower() not in _VALID_COMPRESSIONS:
            raise ValueError(
                f"Unknown compression {compression!r} (in {entry_point}). "
                f"Valid options: {list(_VALID_COMPRESSIONS)}.")
    elif compression is not None:
        # Unreachable from ``to_geotiff`` (which only forwards ``str``
        # or the default), but direct callers can hit this. Without the
        # explicit guard the downstream ``compression.lower()`` would
        # surface as ``AttributeError`` instead of a typed error.
        raise TypeError(
            f"compression must be a str (in {entry_point}); "
            f"got {type(compression).__name__}.")

    # Gap #2: JPEG opt-in gate. The encoder writes self-contained JFIF
    # streams without the JPEGTables tag (347), so the file decodes
    # through xrspatial but not through libtiff / GDAL / rasterio.
    # ``to_geotiff`` already enforces this; surface the same gate here
    # so direct callers cannot silently produce a non-interop file.
    if (isinstance(compression, str)
            and compression.lower() == 'jpeg'
            and not allow_internal_only_jpeg):
        raise ValueError(
            "compression='jpeg' is not supported: the encoder writes "
            "self-contained JFIF streams without the required "
            "JPEGTables tag (347), so other readers (libtiff, GDAL, "
            "rasterio) reject the file. Use 'deflate', 'zstd', or "
            "'lzw' instead. Pass allow_internal_only_jpeg=True to "
            "opt in to the experimental internal-reader-only path "
            "(issue #1845).")

    # Gap #3: max_z_error must be >= 0 and only applies with LERC.
    # ``_write_tiled`` / ``_write_stripped`` silently ignore the value
    # on non-LERC codecs, which lets a typo or a stale arg slip past.
    if max_z_error is not None and max_z_error < 0:
        raise ValueError(
            f"max_z_error must be >= 0, got {max_z_error} "
            f"(in {entry_point})")
    if (max_z_error is not None and max_z_error != 0
            and (not isinstance(compression, str)
                 or compression.lower() != 'lerc')):
        raise ValueError(
            f"max_z_error is only valid with compression='lerc' "
            f"(in {entry_point}); got compression={compression!r}.")

    # Gap #4: ``crs_epsg`` must not be ``bool``. ``bool`` is an ``int``
    # subclass in Python, so ``crs_epsg=True`` would otherwise be
    # written as ``EPSG=1`` (and ``False`` as ``EPSG=0``) -- neither
    # resolves with any CRS database. Mirrors the public
    # ``_validate_crs_arg`` gate.
    if isinstance(crs_epsg, bool):
        raise ValueError(
            f"crs_epsg must be an int (EPSG code) or None (in "
            f"{entry_point}); got bool ({crs_epsg!r}). bool is an int "
            f"subclass in Python, so True/False would otherwise be "
            f"written as EPSG=1 / EPSG=0.")

    # Gap #9: refuse to land an unvalidatable string in
    # GTCitationGeoKey unless the caller opts in. ``to_geotiff`` runs
    # this against the post-``_wkt_to_epsg`` fallback; ``_write`` only
    # sees the raw ``crs_wkt`` kwarg, so apply the same structural
    # check here. This is a strict subset of the upstream gate -- the
    # upstream call already rejected the same input, so this is a
    # no-op when ``to_geotiff`` is the caller.
    if crs_wkt is not None and crs_epsg is None:
        _validate_crs_fallback(crs_wkt, allow_unparseable_crs)


def _write(data: np.ndarray, path: str, *,
          geo_transform: GeoTransform | None = None,
          crs_epsg: int | None = None,
          crs_wkt: str | None = None,
          nodata=None,
          compression: str = 'zstd',
          compression_level: int | None = None,
          tiled: bool = True,
          tile_size: int = 256,
          predictor: bool | int = False,
          cog: bool = False,
          overview_levels: list[int] | None = None,
          overview_resampling: str = 'mean',
          raster_type: int = 1,
          x_resolution: float | None = None,
          y_resolution: float | None = None,
          resolution_unit: int | None = None,
          gdal_metadata_xml: str | None = None,
          extra_tags: list | None = None,
          bigtiff: bool | None = None,
          max_z_error: float = 0.0,
          photometric='auto',
          restore_sentinel: bool = True,
          allow_internal_only_jpeg: bool = False,
          allow_unparseable_crs: bool = False) -> None:
    """Write a numpy array as a GeoTIFF or COG.

    Parameters
    ----------
    data : np.ndarray
        2D array (height x width).
    path : str
        Output file path.
    geo_transform : GeoTransform or None
        Pixel-to-coordinate mapping.
    crs_epsg : int or None
        EPSG code.
    crs_wkt : str or None
        WKT string. Used only when ``crs_epsg`` is None.
    nodata : float, int, or None
        NoData value.
    compression : str
        Codec name. One of ``'none'``, ``'deflate'``, ``'lzw'``,
        ``'jpeg'``, ``'packbits'``, ``'zstd'``, ``'lz4'``,
        ``'jpeg2000'`` (alias ``'j2k'``), or ``'lerc'``.
        ``'jpeg'`` is only valid for ``uint8`` data with 1 or 3 bands;
        any other dtype or band count raises ``ValueError``.
    compression_level : int or None
        Effort level forwarded to the codec. None uses each codec's
        default. Valid ranges: deflate 1-9, zstd 1-22, lz4 0-16.
        Codecs without a level concept (lzw, packbits, jpeg) accept any
        value and ignore it.
    tiled : bool
        Use tiled layout (vs strips).
    tile_size : int
        Tile width and height.
    predictor : bool or int
        TIFF predictor. ``False``/``0``/``1`` -> none, ``True``/``2`` ->
        horizontal differencing, ``3`` -> floating-point predictor
        (float dtypes only).
    cog : bool
        Write as Cloud Optimized GeoTIFF.
    overview_levels : list of int or None
        Decimation factors for the overview pyramid, expressed as a
        list of power-of-two integers strictly greater than 1
        (``[2, 4, 8]`` writes overviews at 1/2, 1/4 and 1/8 of the
        full resolution). The list must be strictly increasing.
        Non-power-of-two values raise ``ValueError`` because the
        underlying block reducer only halves per step. Only used if
        ``cog=True``. If None and ``cog=True``, levels auto-generate
        as ``[2, 4, 8, ...]`` until the next halving would fall below
        ``tile_size`` (capped at 8 levels).
    overview_resampling : str
        Resampling method for overviews: ``'mean'`` (default),
        ``'nearest'``, ``'min'``, ``'max'``, ``'median'``, ``'mode'``,
        or ``'cubic'``.
    raster_type : int
        TIFF ``GTRasterTypeGeoKey`` value. ``1`` (default) = PixelIsArea,
        ``2`` = PixelIsPoint.
    x_resolution, y_resolution : float or None
        Pixels per ``resolution_unit`` along each axis. Written into the
        TIFF XResolution / YResolution tags.
    resolution_unit : int or None
        TIFF ResolutionUnit tag. ``1`` = none, ``2`` = inch, ``3`` = cm.
    gdal_metadata_xml : str or None
        Raw XML payload written to the ``GDAL_METADATA`` tag. Used to
        round-trip arbitrary GDAL-style metadata.
    extra_tags : list or None
        Additional TIFF tags to emit, as a list of
        ``(tag_id, type_id, count, value)`` tuples.
    bigtiff : bool or None
        Force BigTIFF (64-bit offsets). None auto-promotes when the
        estimated file size would exceed the classic-TIFF 4 GB limit.
    max_z_error : float
        Per-pixel error budget for LERC compression. ``0.0`` (default)
        is lossless. Only valid with ``compression='lerc'``.
    allow_internal_only_jpeg : bool
        Opt in to the experimental JPEG-in-TIFF path. The encoder
        writes self-contained JFIF tiles without the ``JPEGTables``
        tag (347), so external readers (libtiff, GDAL, rasterio)
        reject the file. ``False`` (default) rejects
        ``compression='jpeg'`` with a clear error. See issue #1845.
    allow_unparseable_crs : bool
        Opt in to writing a ``crs_wkt`` string that does not parse as
        WKT and does not resolve via pyproj into ``GTCitationGeoKey``
        (pre-#1929 behaviour). Default ``False`` refuses to land
        unvalidatable strings in the citation field.

    Notes
    -----
    This is a module-private array-level entry point. The supported
    public surface is :func:`xrspatial.geotiff.to_geotiff`. Several
    DataArray-level checks (3D dim-order rejection, ``band``-first
    reorder, fail-closed georeferenced-transform, ``masked_nodata``
    handling) are deliberately *not* performed here -- they need
    ``data.dims`` / ``data.attrs`` state that the array-level entry
    point does not have. Direct callers that need those checks should
    use :func:`xrspatial.geotiff.to_geotiff` instead. See issue #2138.
    """
    # Issue #2075: reject empty spatial shapes before any IFD layout
    # math runs. ``to_geotiff`` already guards this for DataArray inputs,
    # but ``_write`` is also called directly by tests and by the GPU
    # path, so guard here too. ``_write`` always receives band-last
    # arrays (eager moveaxis ran upstream), so the ndim-based pair
    # picked by ``_validate_writer_spatial_shape`` without ``dims`` is
    # correct.
    from ._validation import _validate_writer_spatial_shape
    _validate_writer_spatial_shape(
        getattr(data, 'shape', None), entry_point="_write")

    # Issue #2138: push down byte-affecting validation that the public
    # ``to_geotiff`` wrapper used to perform on its own. The wrapper
    # still runs the same checks upstream, so this is a no-op when
    # ``to_geotiff`` is the caller; the gate matters for direct callers
    # of ``_write`` (the GPU CPU-fallback path, internal tests, and
    # downstream code that imports the array-level entry point).
    _validate_lowlevel_write_kwargs(
        compression=compression,
        allow_internal_only_jpeg=allow_internal_only_jpeg,
        max_z_error=max_z_error,
        crs_epsg=crs_epsg,
        crs_wkt=crs_wkt,
        allow_unparseable_crs=allow_unparseable_crs,
        entry_point="_write",
    )

    # Issue #2138 gap #7: auto-promote ``float16`` and ``bool_`` before
    # the dtype mapper. ``to_geotiff`` already does this upstream; the
    # push-down here lets direct callers feed unsupported dtypes without
    # tripping ``numpy_to_tiff_dtype`` with a less actionable error.
    # The guard on ``isinstance(data, np.ndarray)`` is intentional:
    # ``_write`` is the numpy-array entry point, so the dask streaming
    # path goes through ``_write_streaming`` (which handles ``out_dtype``
    # promotion separately). A bare ``dask.array`` reaching ``_write``
    # is already a misuse; the dtype mapper below will surface the
    # mismatch.
    if isinstance(data, np.ndarray):
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        elif data.dtype == np.bool_:
            data = data.astype(np.uint8)

    # Issue #2138 gap #5: defensive copy on the NaN-to-sentinel rewrite.
    # ``to_geotiff`` already copies for DataArray inputs, but a direct
    # caller passing a NaN-containing float array would otherwise have
    # their buffer mutated (numpy arrays are passed by reference). The
    # rewrite is gated on ``restore_sentinel`` so DataArrays carrying
    # ``masked_nodata=False`` (read with ``mask_nodata=False``) keep
    # their literal sentinel bytes untouched (#1988).
    if (isinstance(data, np.ndarray)
            and nodata is not None
            and data.dtype.kind == 'f'
            and not np.isnan(nodata)
            and restore_sentinel):
        nan_mask = np.isnan(data)
        if nan_mask.any():
            data = data.copy()
            data[nan_mask] = data.dtype.type(nodata)

    comp_tag = _compression_tag(compression)
    pred_int = normalize_predictor(predictor, data.dtype, comp_tag)

    # JPEG validation: only uint8, 1 or 3 bands
    if comp_tag == COMPRESSION_JPEG:
        samples = data.shape[2] if data.ndim == 3 else 1
        if data.dtype != np.uint8:
            raise ValueError(
                f"JPEG compression requires uint8 data, got {data.dtype}. "
                f"JPEG is lossy and only supports 8-bit unsigned data.")
        if samples not in (1, 3):
            raise ValueError(
                f"JPEG compression requires 1 or 3 bands, got {samples}")

    # MinIsWhite (photometric=0) requires a writer-side inversion to mirror
    # the reader's unconditional inversion of single-band MinIsWhite data
    # (see _reader._apply_photometric_miniswhite). Without this, written
    # values do not round-trip. The nodata sentinel is inverted alongside
    # the pixels so that the on-disk sentinel byte matches the on-disk
    # pixel byte that means "missing" -- the reader's existing mask logic
    # (issue #1809) then identifies the correct positions and rewrites
    # them to NaN. Issue #1836.
    _samples = data.shape[2] if data.ndim == 3 else 1
    _resolved_photo, _ = _resolve_photometric(photometric, _samples)
    _reject_disagreeing_photometric_override(
        extra_tags, _resolved_photo, _samples, photometric
    )
    if _resolved_photo == 0 and _samples == 1:
        if cog or overview_levels is not None:
            raise NotImplementedError(
                "photometric='miniswhite' is not supported with "
                "cog=True or explicit overview_levels: overview reducers "
                "('min', 'max', 'mode', ...) do not commute with the "
                "pixel inversion, so summary statistics would not match "
                "the user-domain values. Write without overviews, or "
                "use photometric='minisblack' / 'auto'.")
        data = _apply_photometric_miniswhite_invert(
            data, _resolved_photo, _samples)
        if nodata is not None:
            nodata = _invert_nodata_for_miniswhite(nodata, data.dtype)

    # Build pixel data parts
    parts = []

    # Full resolution
    if tiled:
        rel_off, bc, comp_data = _write_tiled(data, comp_tag, pred_int, tile_size,
                                               compression_level=compression_level,
                                               max_z_error=max_z_error)
    else:
        rel_off, bc, comp_data = _write_stripped(data, comp_tag, pred_int,
                                                  compression_level=compression_level,
                                                  max_z_error=max_z_error)

    h, w = data.shape[:2]
    parts.append((data, w, h, rel_off, bc, comp_data))

    # Overviews
    if cog:
        if overview_levels is None:
            # Auto-generate: keep halving until < tile_size, capped at 8 levels.
            # 8 halvings = 1/256 resolution, which is more than enough for
            # interactive zoom on any realistic raster. Past that, overview
            # write cost dominates without benefiting consumers. The list
            # holds actual decimation factors (2, 4, 8, ...) so the loop
            # below treats auto-generated and user-supplied lists
            # identically (issue #1766).
            overview_levels = []
            oh, ow = h, w
            factor = 2
            while oh > tile_size and ow > tile_size and len(overview_levels) < _MAX_OVERVIEW_LEVELS:
                oh //= 2
                ow //= 2
                if oh > 0 and ow > 0:
                    overview_levels.append(factor)
                    factor *= 2
        else:
            # Validate explicit lists. Each entry is a power-of-two
            # decimation factor >= 2, strictly increasing, and feasible
            # for the input shape. The previous behaviour silently
            # ignored the values and used the list length as the
            # halving count (issue #1766).
            overview_levels = _validate_overview_levels(
                overview_levels, height=h, width=w)

        # Overview reductions need the *unmasked* float array so that
        # ``np.nanmean`` / ``np.nanmin`` / ``np.nanmax`` / ``np.nanmedian``
        # honour the sentinel as missing-data. The CPU writer's caller
        # (``to_geotiff``) currently rewrites NaN to ``nodata`` before
        # ``write()`` runs (so the on-disk full-resolution tile bytes
        # match the sentinel-aware reader). We pass ``nodata`` into
        # ``_make_overview`` here so the reducer masks the sentinel back
        # to NaN before averaging; without this, the sentinel poisons
        # the overview (issue #1613). After reduction any block that was
        # all-sentinel comes back as NaN; we rewrite those NaNs back to
        # ``nodata`` below so the on-disk overview tiles use the same
        # sentinel convention as the full-resolution band (external
        # readers without NaN awareness still see a well-defined pixel).
        current = data
        cumulative_factor = 1
        for target_factor in overview_levels:
            # Halve repeatedly until the cumulative decimation matches
            # the requested factor. Validation has already established
            # that ``target_factor`` is a power of two and strictly
            # greater than ``cumulative_factor``.
            while cumulative_factor < target_factor:
                current = _make_overview(current, method=overview_resampling,
                                         nodata=nodata)
                cumulative_factor *= 2
                # Rewrite any NaN produced by the all-sentinel reduction
                # back to the sentinel so the overview pyramid carries the
                # same masking convention as the full-resolution band. The
                # original ``data`` already underwent the NaN->sentinel
                # rewrite upstream, so the only new NaNs here come from the
                # reducer itself.
                if (nodata is not None
                        and current.dtype.kind == 'f'
                        and not np.isnan(nodata)
                        and restore_sentinel):
                    nan_mask = np.isnan(current)
                    if nan_mask.any():
                        current = current.copy()
                        current[nan_mask] = current.dtype.type(nodata)
            oh, ow = current.shape[:2]
            if tiled:
                o_off, o_bc, o_data = _write_tiled(current, comp_tag, pred_int,
                                                    tile_size,
                                                    compression_level=compression_level,
                                                    max_z_error=max_z_error)
            else:
                o_off, o_bc, o_data = _write_stripped(current, comp_tag, pred_int,
                                                       compression_level=compression_level,
                                                       max_z_error=max_z_error)
            parts.append((current, ow, oh, o_off, o_bc, o_data))

    file_bytes = _assemble_tiff(
        w, h, data.dtype, comp_tag, pred_int, tiled, tile_size,
        parts, geo_transform, crs_epsg, nodata, is_cog=cog,
        raster_type=raster_type, crs_wkt=crs_wkt,
        gdal_metadata_xml=gdal_metadata_xml,
        extra_tags=extra_tags,
        x_resolution=x_resolution, y_resolution=y_resolution,
        resolution_unit=resolution_unit,
        force_bigtiff=bigtiff,
        photometric=photometric,
    )

    _write_bytes(file_bytes, path)


# Backward-compatible alias for internal call sites that pre-date the
# rename to :func:`_write`. New code inside ``xrspatial.geotiff`` should
# import :func:`_write` directly. See issue #2138 and the module
# docstring.
write = _write


def _compress_block(arr, block_w, block_h, samples, dtype, bytes_per_sample,
                    predictor: int, compression, compression_level=None,
                    max_z_error: float = 0.0, gil_friendly: bool = False):
    """Compress a tile or strip.  *arr* must be contiguous and correctly sized."""
    if compression == COMPRESSION_JPEG:
        return jpeg_compress(arr.tobytes(), block_w, block_h, samples)

    if predictor != 1 and compression != COMPRESSION_NONE:
        buf = arr.view(np.uint8).ravel().copy()
        buf = _apply_predictor_encode(
            buf, predictor, block_w, block_h, bytes_per_sample, samples)
        raw_data = buf.tobytes()
    else:
        raw_data = arr.tobytes()

    if compression == COMPRESSION_JPEG2000:
        from ._compression import jpeg2000_compress
        return jpeg2000_compress(raw_data, block_w, block_h,
                                 samples=samples, dtype=dtype)
    if compression == COMPRESSION_LERC:
        from ._compression import lerc_compress
        return lerc_compress(raw_data, block_w, block_h,
                             samples=samples, dtype=dtype,
                             max_z_error=max_z_error)
    if compression_level is None:
        return compress(raw_data, compression, gil_friendly=gil_friendly)
    return compress(raw_data, compression, level=compression_level,
                    gil_friendly=gil_friendly)


# ---------------------------------------------------------------------------
# Streaming writer (dask -> monolithic TIFF without full materialisation)
# ---------------------------------------------------------------------------

# ``_compute_classic_ifd_overhead`` and
# ``_should_use_bigtiff_streaming`` live in ``_write_layout.py`` and are
# re-exported above for backwards compatibility.



def _write_streaming(dask_data, path: str, *,
                    geo_transform: 'GeoTransform | None' = None,
                    crs_epsg: int | None = None,
                    crs_wkt: str | None = None,
                    nodata=None,
                    compression: str = 'zstd',
                    compression_level: int | None = None,
                    tiled: bool = True,
                    tile_size: int = 256,
                    predictor: bool | int = False,
                    raster_type: int = 1,
                    x_resolution: float | None = None,
                    y_resolution: float | None = None,
                    resolution_unit: int | None = None,
                    gdal_metadata_xml: str | None = None,
                    extra_tags: list | None = None,
                    bigtiff: bool | None = None,
                    streaming_buffer_bytes: int = 256 * 1024 * 1024,
                    max_z_error: float = 0.0,
                    photometric='auto',
                    restore_sentinel: bool = True,
                    allow_internal_only_jpeg: bool = False,
                    allow_unparseable_crs: bool = False) -> None:
    """Write a dask array as a GeoTIFF by streaming pixel data.

    For tiled output, each tile-row is computed in horizontal segments
    that fit within ``streaming_buffer_bytes``. Most rasters fit in a
    single segment per tile-row, matching the previous behaviour. Wide
    rasters get bounded peak memory at the cost of more dask compute
    calls.

    Peak materialised memory is approximately
    ``min(streaming_buffer_bytes, tile_height * width * bytes_per_sample
    * samples)`` for tiled output, or
    ``rows_per_strip * width * bytes_per_sample * samples`` for stripped
    output (no horizontal segmentation in strip mode).

    After all pixel data is written the IFD offset and byte-count arrays
    are patched in place.

    Parameters
    ----------
    streaming_buffer_bytes : int
        Soft cap on bytes materialised per dask compute call when
        writing tiles. Defaults to 256 MB. Values smaller than one tile
        column are clamped up to one tile column.
    allow_internal_only_jpeg : bool
        Opt in to the experimental JPEG-in-TIFF path. See ``_write`` for
        the rationale. Default ``False``.
    allow_unparseable_crs : bool
        Opt in to writing an unparseable ``crs_wkt`` string into
        ``GTCitationGeoKey``. Default ``False``.

    Notes
    -----
    This is a module-private array-level entry point. The supported
    public surface is :func:`xrspatial.geotiff.to_geotiff`. See
    :func:`_write` and issue #2138 for the full rationale.
    """
    import os
    import tempfile

    # Fail fast for unsupported destinations
    if _is_fsspec_uri(path):
        raise NotImplementedError(
            "Streaming dask write to cloud storage is not yet supported. "
            "Use .compute() first or write to a .vrt file.")

    # Issue #2075: reject empty spatial shapes before tile/strip count
    # math (``math.ceil(width / tw)`` etc. below at the layout block)
    # silently produces zero entries. ``to_geotiff`` already validates
    # this upstream, but direct callers of ``_write_streaming`` go
    # through here too.
    from ._validation import _validate_writer_spatial_shape
    _validate_writer_spatial_shape(
        getattr(dask_data, 'shape', None), entry_point="_write_streaming")

    # Issue #2138: push-down validation for byte-affecting kwargs.
    # ``to_geotiff`` runs these upstream as well, so this is a no-op
    # on that path; the gate matters for direct callers.
    _validate_lowlevel_write_kwargs(
        compression=compression,
        allow_internal_only_jpeg=allow_internal_only_jpeg,
        max_z_error=max_z_error,
        crs_epsg=crs_epsg,
        crs_wkt=crs_wkt,
        allow_unparseable_crs=allow_unparseable_crs,
        entry_point="_write_streaming",
    )

    height, width = dask_data.shape[:2]
    samples = dask_data.shape[2] if dask_data.ndim == 3 else 1
    dtype = dask_data.dtype

    # MinIsWhite pre-inversion (issue #1836) runs per-array in the eager
    # ``write`` path. The streaming dask path materialises one tile-row
    # at a time, so applying the inversion correctly would require
    # threading the transform through every per-tile segment. That
    # plumbing is out of scope for the round-trip fix; refuse the
    # combination so callers do not silently get inverted on-disk values.
    # Callers can ``.compute()`` first and use the eager ``write`` path.
    _resolved_photo_ds, _ = _resolve_photometric(photometric, samples)
    if _resolved_photo_ds == 0 and samples == 1:
        raise NotImplementedError(
            "photometric='miniswhite' on a dask-backed array is not "
            "supported: the streaming writer would have to thread the "
            "writer-side pixel inversion through every tile segment to "
            "match the reader's unconditional MinIsWhite inversion "
            "(issue #1836). Call ``.compute()`` first to use the eager "
            "writer, or write with photometric='minisblack' / 'auto'.")
    # The kwarg guard above only catches photometric='miniswhite'. An
    # ``extra_tags`` entry of ``(TAG_PHOTOMETRIC, ...)`` silently
    # overrides the IFD tag further down, so the writer must reject the
    # MinIsWhite-crossing single-band case the same way the eager
    # writer does. Issue #2073.
    _reject_disagreeing_photometric_override(
        extra_tags, _resolved_photo_ds, samples, photometric
    )

    # Match the eager path's dtype promotion
    out_dtype = dtype
    if out_dtype == np.float16:
        out_dtype = np.float32
    elif out_dtype == np.bool_:
        out_dtype = np.uint8

    bits_per_sample, sample_format = numpy_to_tiff_dtype(out_dtype)
    bytes_per_sample = out_dtype.itemsize
    comp_tag = _compression_tag(compression)
    pred_int = normalize_predictor(predictor, out_dtype, comp_tag)

    if comp_tag == COMPRESSION_JPEG:
        if out_dtype != np.uint8:
            raise ValueError(
                f"JPEG compression requires uint8 data, got {out_dtype}.")
        if samples not in (1, 3):
            raise ValueError(
                f"JPEG compression requires 1 or 3 bands, got {samples}")

    # Layout parameters
    if tiled:
        tw = th = tile_size
        tiles_across = math.ceil(width / tw)
        tiles_down = math.ceil(height / th)
        n_entries = tiles_across * tiles_down
    else:
        rows_per_strip = min(256, height)
        n_entries = math.ceil(height / rows_per_strip)

    # BigTIFF detection has to wait until the full tag list is built so
    # that variable-length payloads (gdal_metadata, geo tags, user
    # extra_tags) feed into the IFD-overhead calculation. Build the tag
    # list assuming classic offsets first, then decide BigTIFF, then
    # promote the strip/tile offset arrays to LONG8 if needed. See
    # issue #1785 and the Copilot review on PR #1787.
    uncompressed_bytes = height * width * bytes_per_sample * samples

    # ---- Build tag list (mirrors _assemble_tiff for level 0) ----
    # Start with classic offset types; the offset arrays are promoted to
    # LONG8 below once BigTIFF is chosen.
    use_bigtiff = bool(bigtiff) if bigtiff is not None else False
    tags = []
    tags.append((TAG_IMAGE_WIDTH, LONG, 1, width))
    tags.append((TAG_IMAGE_LENGTH, LONG, 1, height))
    if samples > 1:
        tags.append((TAG_BITS_PER_SAMPLE, SHORT, samples,
                     [bits_per_sample] * samples))
    else:
        tags.append((TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample))
    tags.append((TAG_COMPRESSION, SHORT, 1, comp_tag))
    # Photometric: caller-controlled, default 'auto' -> MinIsBlack so a
    # 4-band raster is not silently tagged as RGB+alpha. A user
    # ``extra_tags`` entry of (TAG_PHOTOMETRIC, ...) or
    # (TAG_EXTRA_SAMPLES, ...) overrides the writer's chosen value.
    # See issue #1769.
    auto_photometric, auto_extras = _resolve_photometric(
        photometric, samples)
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
    if user_photometric_override is not None:
        tags.append(user_photometric_override)
    else:
        tags.append((TAG_PHOTOMETRIC, SHORT, 1, auto_photometric))
    tags.append((TAG_SAMPLES_PER_PIXEL, SHORT, 1, samples))
    if samples > 1:
        tags.append((TAG_SAMPLE_FORMAT, SHORT, samples,
                     [sample_format] * samples))
    else:
        tags.append((TAG_SAMPLE_FORMAT, SHORT, 1, sample_format))

    if user_extras_override is not None:
        tags.append(user_extras_override)
    elif auto_extras:
        tags.append((TAG_EXTRA_SAMPLES, SHORT, len(auto_extras),
                     list(auto_extras)))

    pred_val = pred_int if comp_tag != COMPRESSION_NONE else 1
    if pred_val != 1:
        tags.append((TAG_PREDICTOR, SHORT, 1, pred_val))

    if x_resolution is not None:
        tags.append((TAG_X_RESOLUTION, RATIONAL, 1, x_resolution))
    if y_resolution is not None:
        tags.append((TAG_Y_RESOLUTION, RATIONAL, 1, y_resolution))
    if resolution_unit is not None:
        tags.append((TAG_RESOLUTION_UNIT, SHORT, 1, resolution_unit))

    # Layout tags with placeholder offsets / byte-counts. Use classic
    # LONG (uint32) here; if the auto-BigTIFF decision below promotes
    # the file, ``_promote_offsets_to_long8`` retypes these to LONG8.
    # A caller-forced ``bigtiff=True`` is also resolved at that point.
    offset_type = LONG
    placeholder = [0] * n_entries
    if tiled:
        tags.append((TAG_TILE_WIDTH, SHORT, 1, tile_size))
        tags.append((TAG_TILE_LENGTH, SHORT, 1, tile_size))
        tags.append((TAG_TILE_OFFSETS, offset_type, n_entries, list(placeholder)))
        tags.append((TAG_TILE_BYTE_COUNTS, offset_type, n_entries, list(placeholder)))
    else:
        tags.append((TAG_ROWS_PER_STRIP, SHORT, 1, rows_per_strip))
        tags.append((TAG_STRIP_OFFSETS, offset_type, n_entries, list(placeholder)))
        tags.append((TAG_STRIP_BYTE_COUNTS, offset_type, n_entries, list(placeholder)))

    # Geo tags
    geo_tags_dict = {}
    if geo_transform is not None:
        geo_tags_dict = build_geo_tags(
            geo_transform, crs_epsg, nodata, raster_type=raster_type,
            crs_wkt=crs_wkt)
    elif crs_epsg is not None or crs_wkt is not None or nodata is not None:
        geo_tags_dict = build_geo_tags(
            GeoTransform(), crs_epsg, nodata, raster_type=raster_type,
            crs_wkt=crs_wkt)
        geo_tags_dict.pop(TAG_MODEL_PIXEL_SCALE, None)
        geo_tags_dict.pop(TAG_MODEL_TIEPOINT, None)

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

    if gdal_metadata_xml is not None:
        tags.append((TAG_GDAL_METADATA, ASCII,
                     len(gdal_metadata_xml) + 1, gdal_metadata_xml))

    if extra_tags is not None:
        existing_ids = {t[0] for t in tags}
        for etag_id, etype_id, ecount, evalue in extra_tags:
            # Skip dangerous tags (NewSubfileType, SubIFDs) that would
            # mis-mark the IFD or carry stale offsets. See issue #1657.
            if (etag_id not in existing_ids
                    and etag_id not in _DANGEROUS_EXTRA_TAG_IDS):
                tags.append((etag_id, etype_id, ecount, evalue))

    # ---- BigTIFF decision (auto path) ----
    # Compute the real classic-TIFF IFD overhead from the actual tag
    # list, including overflow heap (gdal_metadata, geo ascii params,
    # strip/tile offset arrays, user extra_tags). This replaces the
    # 200-byte fudge constant the original PR used; with metadata-heavy
    # writes that constant silently underestimated overhead and let
    # sub-4 GiB rasters overflow classic offsets late in the write.
    # See issue #1785 and the Copilot review on PR #1787.
    if bigtiff is None:
        ifd_overhead_bytes = _compute_classic_ifd_overhead(tags)
        # n_entries=0 because the strip/tile offset arrays are already
        # inside ``tags`` and therefore in ``ifd_overhead_bytes``.
        use_bigtiff = _should_use_bigtiff_streaming(
            uncompressed_bytes,
            n_entries=0,
            ifd_overhead_bytes=ifd_overhead_bytes,
            header_size_classic=8,
        )

    header_size = 16 if use_bigtiff else 8

    # Promote the strip/tile offset arrays to LONG8 once BigTIFF is set.
    if use_bigtiff:
        tags = _promote_offsets_to_long8(tags)

    # ---- Pre-compute IFD reservation size ----
    sorted_tags = sorted(tags, key=lambda t: t[0])
    entry_size = 20 if use_bigtiff else 12
    count_size = 8 if use_bigtiff else 2
    next_size = 8 if use_bigtiff else 4
    num_tags = len(sorted_tags)
    ifd_block_size = count_size + entry_size * num_tags + next_size
    overflow_base = header_size + ifd_block_size
    _, placeholder_overflow = _build_ifd(sorted_tags, overflow_base,
                                          bigtiff=use_bigtiff)
    pixel_data_start = overflow_base + len(placeholder_overflow)

    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tif.tmp')

    try:
        # -- Pass 1: write header + placeholder IFD + streaming pixel data --
        actual_offsets = []
        actual_counts = []
        current_offset = pixel_data_start

        with os.fdopen(fd, 'wb') as f:
            # Header
            f.write(b'II')
            if use_bigtiff:
                f.write(struct.pack(f'{BO}H', 43))
                f.write(struct.pack(f'{BO}H', 8))
                f.write(struct.pack(f'{BO}H', 0))
                f.write(struct.pack(f'{BO}Q', header_size))
            else:
                f.write(struct.pack(f'{BO}H', 42))
                f.write(struct.pack(f'{BO}I', header_size))

            # Placeholder IFD + overflow
            ifd_bytes, overflow_bytes = _build_ifd(
                sorted_tags, overflow_base, bigtiff=use_bigtiff)
            f.write(ifd_bytes)
            f.write(overflow_bytes)

            # Stream pixel data
            if tiled:
                # Decide how many tile-columns we can buffer at once.
                # bytes_per_full_tile_row = tile_h * width * dtype * samples;
                # if it fits the budget we buffer the whole row (matches
                # original behaviour). Otherwise segment horizontally,
                # always at tile boundaries to keep slicing aligned.
                bytes_per_tile_col = (
                    th * tw * bytes_per_sample * samples)
                bytes_per_full_row = bytes_per_tile_col * tiles_across
                if bytes_per_full_row <= streaming_buffer_bytes:
                    tiles_per_segment = tiles_across
                else:
                    tiles_per_segment = max(
                        1, streaming_buffer_bytes // bytes_per_tile_col)

                # Hoist the compression thread pool over the entire tiled
                # write. Re-creating the executor per segment paid the
                # thread-startup cost on every horizontal stripe and
                # offset the parallel speedup on wide rasters; a single
                # pool reused across all segments avoids that overhead.
                # Skip the pool when compression is uncompressed (no
                # C-level work to release the GIL on) or when the host
                # has only one usable core.
                from concurrent.futures import ThreadPoolExecutor
                _pool_workers = min(tiles_per_segment, os.cpu_count() or 4)
                _use_pool = (comp_tag != COMPRESSION_NONE
                             and _pool_workers > 1)
                tile_pool = (ThreadPoolExecutor(max_workers=_pool_workers)
                             if _use_pool else None)

                for tr in range(tiles_down):
                    r0 = tr * th
                    r1 = min(r0 + th, height)
                    actual_h = r1 - r0

                    for seg_start in range(0, tiles_across, tiles_per_segment):
                        seg_end = min(seg_start + tiles_per_segment,
                                       tiles_across)
                        seg_c0 = seg_start * tw
                        seg_c1 = min(seg_end * tw, width)

                        # Compute just this horizontal segment
                        if dask_data.ndim == 3:
                            seg_np = np.asarray(
                                dask_data[r0:r1, seg_c0:seg_c1, :].compute())
                        else:
                            seg_np = np.asarray(
                                dask_data[r0:r1, seg_c0:seg_c1].compute())
                        if hasattr(seg_np, 'get'):
                            seg_np = seg_np.get()

                        if seg_np.dtype != out_dtype:
                            seg_np = seg_np.astype(out_dtype)

                        # NaN -> nodata sentinel
                        if (nodata is not None and seg_np.dtype.kind == 'f'
                                and not np.isnan(nodata)
                                and restore_sentinel):
                            nan_mask = np.isnan(seg_np)
                            if nan_mask.any():
                                seg_np = seg_np.copy()
                                seg_np[nan_mask] = seg_np.dtype.type(nodata)

                        # Build tile arrays for this segment
                        seg_tile_arrs = []
                        for tc in range(seg_start, seg_end):
                            c0 = tc * tw
                            c1 = min(c0 + tw, width)
                            actual_w = c1 - c0

                            local_c0 = c0 - seg_c0
                            local_c1 = c1 - seg_c0
                            tile_slice = seg_np[:, local_c0:local_c1]

                            if actual_h < th or actual_w < tw:
                                if seg_np.ndim == 3:
                                    padded = np.zeros((th, tw, samples),
                                                      dtype=out_dtype)
                                else:
                                    padded = np.zeros((th, tw), dtype=out_dtype)
                                padded[:actual_h, :actual_w] = tile_slice
                                tile_arr = padded
                            else:
                                tile_arr = np.ascontiguousarray(tile_slice)

                            seg_tile_arrs.append(tile_arr)

                        # Parallel compress on the hoisted ``tile_pool``
                        # when it exists. zlib/zstd/LZW release the GIL,
                        # so threading actually parallelises the C-level
                        # work. Peak memory while the segment is in
                        # flight covers BOTH the uncompressed
                        # ``seg_tile_arrs`` (one full tile per column,
                        # released after the futures resolve) AND the
                        # compressed buffers ``seg_compressed`` (held
                        # until the sequential write loop drains them).
                        # Both lists are bounded by ``tiles_per_segment``
                        # which the streaming buffer cap sets; fall
                        # through to a serial path when the pool is None
                        # (no compression / single core) or when only
                        # one tile sits in this segment.
                        n_seg_tiles = len(seg_tile_arrs)
                        if tile_pool is None or n_seg_tiles <= 1:
                            seg_compressed = [
                                _compress_block(
                                    ta, tw, th, samples, out_dtype,
                                    bytes_per_sample, pred_int, comp_tag,
                                    compression_level, max_z_error)
                                for ta in seg_tile_arrs
                            ]
                        else:
                            futures = [
                                tile_pool.submit(
                                    _compress_block,
                                    ta, tw, th, samples, out_dtype,
                                    bytes_per_sample, pred_int, comp_tag,
                                    compression_level, max_z_error,
                                    True)
                                for ta in seg_tile_arrs
                            ]
                            seg_compressed = [
                                fut.result() for fut in futures]

                        # Sequential file write to preserve on-disk tile order
                        for compressed in seg_compressed:
                            actual_offsets.append(current_offset)
                            actual_counts.append(len(compressed))
                            f.write(compressed)
                            current_offset += len(compressed)

                        del seg_np, seg_tile_arrs, seg_compressed

                if tile_pool is not None:
                    tile_pool.shutdown(wait=True)
            else:
                # Strip layout
                for i in range(n_entries):
                    r0 = i * rows_per_strip
                    r1 = min(r0 + rows_per_strip, height)
                    strip_rows = r1 - r0

                    if dask_data.ndim == 3:
                        strip_np = np.asarray(
                            dask_data[r0:r1, :, :].compute())
                    else:
                        strip_np = np.asarray(dask_data[r0:r1, :].compute())
                    if hasattr(strip_np, 'get'):
                        strip_np = strip_np.get()

                    if strip_np.dtype != out_dtype:
                        strip_np = strip_np.astype(out_dtype)

                    if (nodata is not None and strip_np.dtype.kind == 'f'
                            and not np.isnan(nodata)
                            and restore_sentinel):
                        nan_mask = np.isnan(strip_np)
                        if nan_mask.any():
                            strip_np = strip_np.copy()
                            strip_np[nan_mask] = strip_np.dtype.type(nodata)

                    compressed = _compress_block(
                        np.ascontiguousarray(strip_np),
                        width, strip_rows, samples, out_dtype,
                        bytes_per_sample, pred_int, comp_tag,
                        compression_level, max_z_error)

                    actual_offsets.append(current_offset)
                    actual_counts.append(len(compressed))
                    f.write(compressed)
                    current_offset += len(compressed)

                    del strip_np

        # -- Pass 2: patch IFD with actual offsets.  Reuse the type
        # chosen at tag-build time (LONG for classic, LONG8 for
        # BigTIFF) so the patch stays width-consistent with the
        # placeholders reserved in pass 1.
        patched_tags = []
        for tag_id, type_id, count, values in sorted_tags:
            if tag_id in (TAG_TILE_OFFSETS, TAG_STRIP_OFFSETS):
                patched_tags.append((tag_id, type_id, n_entries, actual_offsets))
            elif tag_id in (TAG_TILE_BYTE_COUNTS, TAG_STRIP_BYTE_COUNTS):
                patched_tags.append((tag_id, type_id, n_entries, actual_counts))
            else:
                patched_tags.append((tag_id, type_id, count, values))

        with open(tmp_path, 'r+b') as f:
            f.seek(header_size)
            ifd_bytes, overflow_bytes = _build_ifd(
                patched_tags, overflow_base, bigtiff=use_bigtiff)
            f.write(ifd_bytes)
            f.write(overflow_bytes)

        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# Backward-compatible alias for internal call sites that pre-date the
# rename to :func:`_write_streaming`. New code inside
# ``xrspatial.geotiff`` should import :func:`_write_streaming` directly.
# See issue #2138.
write_streaming = _write_streaming


def _is_fsspec_uri(path) -> bool:
    """Check if a path is a fsspec-compatible URI (string only)."""
    if not isinstance(path, str):
        return False
    if path.startswith(('http://', 'https://')):
        return False
    return '://' in path


def _write_bytes(file_bytes: bytes | bytearray, path) -> None:
    """Write bytes to a local file (atomic), cloud storage (via fsspec),
    or any binary file-like object exposing ``write``.

    Accepts either ``bytes`` or ``bytearray`` so the eager assembler
    can hand its working buffer through without a copy (issue #1756);
    ``file.write``, ``BytesIO.write``, and ``fsspec`` ``open(..., 'wb')``
    all accept the buffer protocol.
    """
    import os

    # File-like destination: match string-path "overwrite" semantics
    # (writing to '/tmp/x.tif' twice produces a one-TIFF file, not two
    # concatenated). Rewind+truncate when the buffer supports it so a
    # caller reusing the same BytesIO across writes doesn't end up with
    # silently appended TIFFs. The caller still owns the buffer's
    # lifetime; we don't close it.
    if not isinstance(path, str) and hasattr(path, 'write'):
        if hasattr(path, 'seek') and hasattr(path, 'truncate'):
            try:
                path.seek(0)
                path.truncate(0)
            except (OSError, AttributeError):
                pass
        path.write(file_bytes)
        return

    if _is_fsspec_uri(path):
        try:
            import fsspec
        except ImportError:
            raise ImportError(
                "fsspec is required to write to cloud storage. "
                "Install it with: pip install fsspec")
        fs, fspath = fsspec.core.url_to_fs(path)
        with fs.open(fspath, 'wb') as f:
            f.write(file_bytes)
        return

    # Local file: write to temp file then atomically rename
    import tempfile
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tif.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(file_bytes)
        os.replace(tmp_path, path)  # atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
