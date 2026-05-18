"""GeoTIFF/COG writer."""
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
    LONG8,
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


# Byte order: always write little-endian
BO = '<'


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


def _block_reduce_2d(arr2d, method, nodata=None):
    """2x block-reduce a single 2D plane using *method*.

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
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    cropped = arr2d[:h2, :w2]
    oh, ow = h2 // 2, w2 // 2

    if method == 'nearest':
        # Top-left pixel of each 2x2 block
        return cropped[::2, ::2].copy()

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
                    masked = np.where(mask, np.float64('nan'), arr2d)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        result = zoom(masked, 0.5, order=3,
                                      prefilter=False)
                    nan_mask = np.isnan(result)
                    if nan_mask.any():
                        result = result.copy()
                        result[nan_mask] = float(nodata)
                    return result.astype(arr2d.dtype)
        nodata_int = _resolve_int_nodata(arr2d.dtype, nodata)
        if nodata_int is not None:
            sentinel = arr2d.dtype.type(nodata_int)
            mask = arr2d == sentinel
            if mask.any():
                masked = np.where(mask, np.float64('nan'),
                                  arr2d.astype(np.float64))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    result = zoom(masked, 0.5, order=3,
                                  prefilter=False)
                nan_mask = np.isnan(result)
                if nan_mask.any():
                    result = np.where(nan_mask, float(nodata_int),
                                      result)
                return np.round(result).astype(arr2d.dtype)
        return zoom(arr2d, 0.5, order=3).astype(arr2d.dtype)

    if method == 'mode':
        # Most-common value per 2x2 block (useful for classified rasters).
        # Vectorized: sort each 4-cell block, then for each position count
        # how many cells equal it. argmax picks the leftmost max-count
        # position, which (post-sort) is the smallest tied value, matching
        # the prior np.unique+argmax tie-break behavior ("lowest wins").
        blocks = cropped.reshape(oh, 2, ow, 2).transpose(0, 2, 1, 3).reshape(oh, ow, 4)
        srt = np.sort(blocks, axis=-1)
        counts = np.empty_like(srt, dtype=np.int8)
        for i in range(4):
            counts[..., i] = np.sum(srt == srt[..., i:i + 1], axis=-1)
        pick = np.argmax(counts, axis=-1)
        return np.take_along_axis(srt, pick[..., None], axis=-1).squeeze(-1)

    # Block reshape for mean/min/max/median
    if arr2d.dtype.kind == 'f':
        blocks = cropped.reshape(oh, 2, ow, 2)
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
        blocks = cropped.astype(np.float64).reshape(oh, 2, ow, 2)
        # Integer rasters with a sentinel need the same NaN-mask the float
        # branch above applies: without it, nanmean / nanmin / nanmax /
        # nanmedian average the sentinel value into surrounding valid
        # cells and produce overview pixels that are neither the sentinel
        # nor any real measurement. The read-side int-to-NaN mask in
        # ``open_geotiff`` only catches exact sentinel hits, so the
        # poisoned values survive as silent garbage at every zoom level
        # above 0. Gate on the sentinel being representable in the
        # source integer dtype (mirrors ``_int_nodata_in_range`` in
        # ``_reader.py``) so an out-of-range sentinel pair like
        # ``uint16`` + ``GDAL_NODATA="-9999"`` stays a no-op rather than
        # tripping ``OverflowError`` on the dtype cast.
        nodata_int = _resolve_int_nodata(arr2d.dtype, nodata)
        if nodata_int is not None:
            sentinel = arr2d.dtype.type(nodata_int)
            # Compare against the original integer block view so the
            # equality runs at the integer's native width (avoids any
            # float-cast rounding on adjacent values). The boolean
            # mask broadcasts into the float64 block layout below.
            int_blocks = cropped.reshape(oh, 2, ow, 2)
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
# Tag serialization
# ---------------------------------------------------------------------------

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
# File assembly
# ---------------------------------------------------------------------------

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

    Parameters
    ----------
    pixel_data_parts : list of (array, width, height, relative_offsets, byte_counts, compressed_data)
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
    for level_idx, (arr, lw, lh, rel_offsets, byte_counts, comp_data) in enumerate(pixel_data_parts):
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
        _compute_classic_ifd_overhead(tags) for tags in ifd_specs
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
        ifd_specs = [_promote_offsets_to_long8(tags) for tags in ifd_specs]

    if is_cog and len(ifd_specs) > 1:
        return _assemble_cog_layout(header_size, ifd_specs, pixel_data_parts,
                                    bigtiff=bigtiff)
    else:
        return _assemble_standard_layout(header_size, ifd_specs, pixel_data_parts,
                                         bigtiff=bigtiff)


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

    # Second pass: pixel data offsets per level
    current_pixel_offset = pixel_data_start
    level_pixel_offsets = []
    for _arr, _lw, _lh, rel_offsets, byte_counts, comp_chunks in pixel_data_parts:
        level_pixel_offsets.append(current_pixel_offset)
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

    # Append all pixel data
    for _arr, _lw, _lh, _rel_offsets, _byte_counts, comp_chunks in pixel_data_parts:
        for chunk in comp_chunks:
            output.extend(chunk)

    return output


# ---------------------------------------------------------------------------
# Public write function
# ---------------------------------------------------------------------------

def write(data: np.ndarray, path: str, *,
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
          restore_sentinel: bool = True) -> None:
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
    """
    # Issue #2075: reject empty spatial shapes before any IFD layout
    # math runs. ``to_geotiff`` already guards this for DataArray inputs,
    # but ``write`` is also called directly by tests and by the GPU
    # path, so guard here too. ``write`` always receives band-last
    # arrays (eager moveaxis ran upstream), so the ndim-based pair
    # picked by ``_validate_writer_spatial_shape`` without ``dims`` is
    # correct.
    from ._validation import _validate_writer_spatial_shape
    _validate_writer_spatial_shape(
        getattr(data, 'shape', None), entry_point="write")

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


def write_streaming(dask_data, path: str, *,
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
                    restore_sentinel: bool = True) -> None:
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
    # this upstream, but direct callers of ``write_streaming`` go
    # through here too.
    from ._validation import _validate_writer_spatial_shape
    _validate_writer_spatial_shape(
        getattr(dask_data, 'shape', None), entry_point="write_streaming")

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
