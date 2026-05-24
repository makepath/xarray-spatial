"""Virtual Raster Table (VRT) reader.

Parses GDAL VRT XML files and assembles a virtual raster from one or
more source GeoTIFF files using windowed reads.
"""
from __future__ import annotations

import math
import os
import struct
import zlib
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from typing import Union
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

import numpy as np

from ._errors import UnsupportedGeoTIFFFeatureError
from ._safe_xml import safe_fromstring


def _codec_decode_exceptions() -> tuple[type[BaseException], ...]:
    """Return the tuple of codec-specific decode exceptions worth swallowing.

    ``read_to_array`` dispatches to per-codec wrappers in
    :mod:`._compression`. Most of those wrappers raise ``ValueError`` on
    malformed input (LZW, PackBits, LERC pre-decode bomb check, JPEG 2000
    fail-closed shape check), but a few codecs leak their library's
    native exception class through the wrapper:

    * ``zlib.error`` from ``zlib.decompress`` for deflate / adobe-deflate
      payloads.
    * ``zstandard.ZstdError`` from ``zstandard.stream_reader.read`` when
      a ZSTD frame is corrupt.

    These are recoverable per-source failures -- they mean "this tile's
    compressed payload is bad", not "the program is broken" -- so they
    belong in the same warn-and-skip catch as ``OSError`` / ``ValueError``
    / ``struct.error``. ``RuntimeError`` (raised by lz4 frame decoder,
    LERC error-code translation, and glymur on malformed JP2) is
    deliberately NOT included: it can come from real bugs as easily as
    from corruption, so it stays in the propagate-and-fail bucket.

    ``zstandard`` is an optional dependency; if it's not installed the
    decoder path is unreachable and there's no exception class to catch.
    """
    excs: list[type[BaseException]] = [zlib.error]
    try:  # pragma: no cover - depends on optional install
        from zstandard import ZstdError
        excs.append(ZstdError)
    except ImportError:
        pass
    return tuple(excs)


# Computed once at import: tuple of codec exception classes to catch in
# the per-source read fallback below. Defined at module scope so the
# import-time work doesn't repeat on every VRT source.
_CODEC_DECODE_EXCEPTIONS = _codec_decode_exceptions()


# Environment variable name used to opt in to absolute source paths
# outside the VRT directory.  ``os.pathsep``-separated list of
# directory roots (``:`` on POSIX, ``;`` on Windows).  Empty entries
# are ignored.  See ``parse_vrt`` for the containment policy.
_ALLOWED_ROOTS_ENV = 'XRSPATIAL_VRT_ALLOWED_ROOTS'


# Cap on the VRT XML file read. VRTs are XML metadata only (pixel data lives
# in source TIFFs); a 50k-source VRT is around 25 MB, so 64 MiB is a safe
# default that still rejects pathological inputs without scanning the whole
# file into memory.
_MAX_XML_BYTES_ENV = 'XRSPATIAL_VRT_MAX_XML_BYTES'
_DEFAULT_MAX_XML_BYTES = 64 * 1024 * 1024


def _get_vrt_max_xml_bytes() -> int:
    """Return the cap on VRT XML file size, in bytes."""
    raw = os.environ.get(_MAX_XML_BYTES_ENV)
    if raw is None or raw == '':
        return _DEFAULT_MAX_XML_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{_MAX_XML_BYTES_ENV} must be a positive integer, got "
            f"{raw!r}")
    if value <= 0:
        raise ValueError(
            f"{_MAX_XML_BYTES_ENV} must be a positive integer, got "
            f"{value}")
    return value


def _read_vrt_xml(vrt_path: str) -> str:
    """Read a VRT XML file with a bounded total size."""
    cap = _get_vrt_max_xml_bytes()
    chunks = []
    total = 0
    with open(vrt_path, 'rb') as f:
        while True:
            remaining = cap + 1 - total
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise ValueError(
                    f"VRT XML at {vrt_path!r} exceeds the "
                    f"{cap:,}-byte cap. Raise the cap by setting "
                    f"{_MAX_XML_BYTES_ENV} if this file is legitimate.")
            chunks.append(chunk)
    return b''.join(chunks).decode('utf-8')


def _allowed_source_roots() -> list[str]:
    """Return the operator-supplied allowlist of trusted source roots.

    Returns the canonical ``realpath`` of each entry so the containment
    check can compare against the resolved source path directly. Empty
    entries (from stray ``os.pathsep`` separators) are dropped.
    """
    raw = os.environ.get(_ALLOWED_ROOTS_ENV, '')
    roots = []
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        roots.append(os.path.realpath(entry))
    return roots


def _path_is_under(path: str, root: str) -> bool:
    """Return True when ``path`` lives under ``root``.

    Both inputs must already be canonical (``os.path.realpath`` applied).
    Uses ``os.path.commonpath`` for robustness against prefix-collisions
    (``/foo`` vs ``/foobar``) which a naive ``startswith`` check misses.
    """
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        # Different drives on Windows, or one of the paths is empty.
        return False


def _xml_text(value) -> str:
    """Escape *value* for safe inclusion as XML element text.

    Handles the five XML predefined entities (``& < > " '``). Returns the
    empty string when ``value`` is ``None``.
    """
    if value is None:
        return ""
    return _xml_escape(str(value), {'"': "&quot;", "'": "&apos;"})


def _xml_attr(value) -> str:
    """Quote *value* for use as an XML attribute value.

    Wraps in matching quotes and escapes the predefined entities. Returns
    ``'""'`` when ``value`` is ``None``.
    """
    if value is None:
        return '""'
    return _xml_quoteattr(str(value))


# Mapping from GDAL VRT dataType names to NumPy dtypes.
#
# ``Int8`` is the GDAL 3.7+ signed-byte name.  ``UInt64`` / ``Int64`` are
# the GDAL 3.5+ 64-bit integer names; both round-trip losslessly through
# numpy's native ``uint64`` / ``int64``.  Complex types (``CInt16``,
# ``CInt32``, ``CFloat32``, ``CFloat64``) are deliberately absent: the
# reader has no complex-data code path, so an unknown ``dataType`` raises
# in :func:`parse_vrt` rather than silently dropping the imaginary
# component by falling back to ``Float32``.  See issue #1783.
_DTYPE_MAP = {
    'Byte': np.uint8,
    'Int8': np.int8,
    'UInt16': np.uint16,
    'Int16': np.int16,
    'UInt32': np.uint32,
    'Int32': np.int32,
    'UInt64': np.uint64,
    'Int64': np.int64,
    'Float32': np.float32,
    'Float64': np.float64,
}


@dataclass
class _Rect:
    """Pixel rectangle: (x_off, y_off, x_size, y_size)."""
    x_off: int
    y_off: int
    x_size: int
    y_size: int


@dataclass
class _Source:
    """A single source region within a VRT band."""
    filename: str
    band: int  # 1-based
    src_rect: _Rect
    dst_rect: _Rect
    # Parsed from ``<NODATA>`` on the source.  ``int`` for integer-dtype
    # bands so 64-bit sentinels survive without float64 rounding (see
    # :func:`_parse_band_nodata`); ``float`` for floating-dtype bands
    # and the legacy fallback.  See issue #1783 follow-up.
    nodata: Union[int, float, None] = None
    # ComplexSource extras
    scale: float | None = None
    offset: float | None = None
    # GDAL ``<ComplexSource><ResampleAlg>`` values are ``NearestNeighbour``
    # (default), ``Bilinear``, ``Cubic``, ``CubicSpline``, ``Lanczos``,
    # ``Average``, ``Mode``.  Only nearest-neighbour is implemented in
    # the placement step (issue #1694).  When ``SrcRect`` and ``DstRect``
    # sizes differ (resampling actually required), the read raises
    # ``NotImplementedError`` for any non-nearest algorithm rather than
    # silently substituting nearest (issue #1751); a non-nearest alg with
    # matching rect sizes is a no-op and passes through.  Higher-quality
    # resamplers are tracked for follow-up.
    resample_alg: str | None = None


@dataclass
class _VRTBand:
    """A single band in a VRT dataset."""
    band_num: int  # 1-based
    dtype: np.dtype
    # Parsed from ``<NoDataValue>``.  ``int`` for integer-dtype bands so
    # 64-bit sentinels (``2**64 - 1``, ``INT64_MIN``) survive without
    # float64 rounding; ``float`` for floating-dtype bands.  See
    # :func:`_parse_band_nodata` and the issue #1783 follow-up.
    nodata: Union[int, float, None] = None
    sources: list[_Source] = field(default_factory=list)
    color_interp: str | None = None


@dataclass
class VRTDataset:
    """Parsed Virtual Raster Table."""
    width: int
    height: int
    crs_wkt: str | None = None
    geo_transform: tuple | None = None  # (origin_x, res_x, skew_x, origin_y, skew_y, res_y)
    bands: list[_VRTBand] = field(default_factory=list)
    # GDAL raster registration metadata.  'area' (default) means the
    # GeoTransform origin is the top-left *corner* of pixel (0, 0) and
    # pixel-center coords need the usual half-pixel shift.  'point'
    # means the origin is already at the *center* of pixel (0, 0) and
    # coords must be emitted without the shift.  Parsed from
    # ``<Metadata><MDI key="AREA_OR_POINT">Point</MDI></Metadata>``.
    raster_type: str = 'area'  # 'area' or 'point'
    # Per-load record of sources skipped by ``read_vrt`` when called
    # with ``missing_sources='warn'`` (the lenient opt-in; the default
    # since #1843 is ``'raise'``, and strict mode always raises). Each
    # entry is a dict with ``source``, ``band`` (1-based), ``dst_rect``
    # (xoff, yoff, xsize, ysize), and ``error`` keys. Empty when no
    # sources failed to read. Populated by :func:`read_vrt` so callers
    # on the warn path can detect holes by attribute lookup instead of
    # parsing the ``GeoTIFFFallbackWarning`` message (issue #1734).
    holes: list[dict] = field(default_factory=list)


def _parse_rect(elem) -> _Rect:
    """Parse a SrcRect or DstRect element."""
    return _Rect(
        x_off=int(float(elem.get('xOff', 0))),
        y_off=int(float(elem.get('yOff', 0))),
        x_size=int(float(elem.get('xSize', 0))),
        y_size=int(float(elem.get('ySize', 0))),
    )


def _text(elem, tag, default=None):
    """Get text content of a child element."""
    child = elem.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return default


def _parse_band_nodata(text: str | None,
                       dtype: np.dtype) -> Union[int, float, None]:
    """Parse a VRT ``<NoDataValue>`` / ``<NODATA>`` string for a given band dtype.

    Returns ``None`` when ``text`` is empty/None.

    For floating dtypes the value is parsed as ``float`` (so ``nan`` /
    ``inf`` and scientific notation work as before).

    For integer dtypes the value is parsed as ``int`` whenever possible so
    that 64-bit sentinels (``2**64 - 1`` for ``UInt64``, ``INT64_MIN`` for
    ``Int64``) round-trip exactly rather than collapsing through a float64
    that cannot represent them.  GDAL is happy to write integer nodata in
    scientific notation or with a trailing ``.0``; if a direct ``int()``
    fails we fall back to ``int(float(text))`` but only accept that result
    when the float is finite, integer-valued, and representable in the
    target dtype.  Out-of-range / non-integer values for an integer dtype
    fall back to the parsed ``float`` (matching the eager-TIFF reader's
    behaviour in :func:`_resolve_masked_fill`: the sentinel can never
    match a real pixel, but ``attrs['nodata']`` still surfaces the raw
    value for write round-trip).  See issue #1783 follow-up.
    """
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    if np.issubdtype(dtype, np.integer):
        # Try the cheap path first: a plain integer literal (decimal,
        # binary, hex with explicit base would be unusual but ``int()``
        # without ``base`` rejects anything other than base-10 numeric
        # strings, which is what GDAL writes).
        try:
            return int(text)
        except ValueError:
            pass
        # Fall back: scientific notation or trailing ``.0``.  Only accept
        # when the float is exactly integer-valued and fits the dtype, so
        # we never silently truncate a fractional value or wrap a
        # sentinel that overflows the band's dtype.
        try:
            v = float(text)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v) or not v.is_integer():
            return v
        info = np.iinfo(dtype)
        nodata_int = int(v)
        if info.min <= nodata_int <= info.max:
            return nodata_int
        return v

    # Floating / other dtypes: parse as float (preserves NaN, Inf,
    # scientific notation).
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


# VRTRasterBand child elements that read_vrt explicitly supports or
# tolerates. Hoisted to module scope so the parse_vrt band loop does
# not rebuild the frozensets on every iteration (epic #2340 / PR 5).
#
# ``_INFORMATIONAL_BAND_TAGS`` -- pure-metadata children that have no
# effect on the assembled array. Silently ignored.
#
# ``_SOURCE_BAND_TAGS`` -- the source types whose pixel-placement
# semantics this reader implements.
#
# ``_UNSUPPORTED_BAND_TAGS`` -- known GDAL VRT children that alter the
# computed output (kernel filters, derived band expressions, mask /
# pansharpening sources). Rejected with ``UnsupportedGeoTIFFFeatureError``
# so a VRT relying on them cannot read as a plain mosaic and lose its
# declared computation. Anything that is none of the three -- e.g. a
# future GDAL VRT extension -- also raises, via the catch-all branch in
# ``parse_vrt``.
_INFORMATIONAL_BAND_TAGS = frozenset({
    'NoDataValue', 'ColorInterp', 'Description', 'Offset',
    'Scale', 'Metadata', 'UnitType', 'CategoryNames',
    'ColorTable', 'Histograms', 'GDALRasterAttributeTable',
    'ColorEntry', 'HideNoDataValue',
    # VRT band-level overview declarations. The reader does not load
    # VRT-declared overviews (overview_level= is honoured by the
    # underlying GeoTIFF source reader, not by VRT-level overview
    # tags), so keeping these informational preserves pre-#2349
    # behaviour for VRTs emitted by GDAL with declared overviews.
    'OverviewList', 'Overview',
})
_SOURCE_BAND_TAGS = frozenset({'SimpleSource', 'ComplexSource'})
_UNSUPPORTED_BAND_TAGS = frozenset({
    'KernelFilteredSource', 'AveragedSource', 'NoDataFromMaskSource',
    'PixelFunctionType', 'PixelFunctionLanguage',
    'PixelFunctionCode', 'PixelFunctionArguments',
    'SourceTransferType', 'MaskBand', 'PansharpeningOptions',
})

# Dataset-level (``<VRTDataset>`` children, sibling of ``<VRTRasterBand>``)
# elements that signal a feature this reader does not implement. The band
# loop never sees these because they sit at the dataset level; sweep the
# root for them after the ``subClass`` check. See epic #2340 / PR 5.
_UNSUPPORTED_DATASET_TAGS = frozenset({
    # Per-dataset alpha / mask band. GDAL writes this as a sibling of
    # the VRTRasterBand entries; the previous parser ignored it and
    # silently dropped the per-pixel mask.
    'MaskBand',
    # Ground-control-point list for warped reprojection.
    'GCPList',
    # Pansharpening setup at the dataset level (separate from the
    # band-level <PansharpeningOptions>).
    'PansharpeningOptions',
})


def parse_vrt(xml_str: str, vrt_dir: str = '.') -> VRTDataset:
    """Parse a VRT XML string into a VRTDataset.

    Parameters
    ----------
    xml_str : str
        VRT XML content.
    vrt_dir : str
        Directory of the VRT file, for resolving relative source paths.

    Returns
    -------
    VRTDataset

    Raises
    ------
    ValueError
        When a ``SourceFilename`` resolves outside ``vrt_dir`` and
        outside every entry in ``XRSPATIAL_VRT_ALLOWED_ROOTS``.  See
        :func:`read_vrt` for the full containment policy.
    """
    # ``safe_fromstring`` refuses DOCTYPE declarations so a crafted VRT
    # cannot trigger XML entity expansion (billion-laughs) attacks
    # against the reader. See issue #1579.
    root = safe_fromstring(xml_str)

    # Pre-compute the trusted roots once per parse.  ``vrt_dir`` is
    # always trusted; the allowlist from ``XRSPATIAL_VRT_ALLOWED_ROOTS``
    # is only consulted for absolute sources (``relativeToVRT='0'``).
    # See issue #1671 for the path-traversal hardening.
    vrt_root = os.path.realpath(vrt_dir)
    allowed_roots = _allowed_source_roots()

    # Refuse any GDAL VRT subclass beyond the plain dataset. The mosaic
    # reader implements the simple ``<VRTDataset>`` case only; warped /
    # pansharpened / processed / derived subclasses depend on a separate
    # pipeline (reprojection, kernel filters, derived expressions) that
    # this release does not ship. A silently-ignored ``subClass`` would
    # produce wrong output: the reader would dispatch on whatever simple
    # sources the subclassed VRT happens to embed and skip the subclass
    # semantics entirely. Name the offending attribute in the error and
    # point at the release tier map. See epic #2340 / PR 5 (#2349).
    _sub_class = root.get('subClass')
    if _sub_class:
        raise UnsupportedGeoTIFFFeatureError(
            f"VRTDataset declares subClass={_sub_class!r}, which is not "
            f"a supported feature in this release. read_vrt implements "
            f"plain ``<VRTDataset>`` mosaics only; warped / pansharpened "
            f"/ processed / derived VRT subclasses are listed as "
            f"unsupported in `xrspatial.geotiff.SUPPORTED_FEATURES` "
            f"(see epic #2340). Materialise the warped or derived output "
            f"with an external tool (e.g. ``gdalwarp``) and read the "
            f"resulting GeoTIFF instead."
        )

    # Sweep the dataset root for known unsupported sibling elements.
    # The band loop below only walks ``<VRTRasterBand>`` children, so a
    # ``<MaskBand>`` / ``<GCPList>`` / dataset-level
    # ``<PansharpeningOptions>`` would otherwise slip through silently
    # and drop its declared semantics (per-pixel mask, ground control
    # points, pansharpening setup). See epic #2340 / PR 5.
    for _ds_child in root:
        _ds_tag = _ds_child.tag
        if isinstance(_ds_tag, str) and _ds_tag in _UNSUPPORTED_DATASET_TAGS:
            raise UnsupportedGeoTIFFFeatureError(
                f"VRTDataset declares a <{_ds_tag}> element, which is "
                f"not a supported feature in this release. read_vrt "
                f"implements plain ``<VRTDataset>`` mosaics over "
                f"``<VRTRasterBand>`` children only; mask bands, GCP "
                f"lists, and dataset-level pansharpening setup are "
                f"listed as unsupported in "
                f"`xrspatial.geotiff.SUPPORTED_FEATURES` (epic #2340)."
            )

    width = int(root.get('rasterXSize', 0))
    height = int(root.get('rasterYSize', 0))

    # CRS
    crs_wkt = _text(root, 'SRS')

    # GeoTransform: "origin_x, res_x, skew_x, origin_y, skew_y, res_y"
    gt_str = _text(root, 'GeoTransform')
    geo_transform = None
    if gt_str:
        parts = [float(x.strip()) for x in gt_str.split(',')]
        if len(parts) == 6:
            geo_transform = tuple(parts)

    # Registration metadata (AREA_OR_POINT).  GDAL stores this as
    # ``<Metadata><MDI key="AREA_OR_POINT">Point</MDI></Metadata>``
    # at the dataset level.  Default is Area.
    raster_type = 'area'
    for md_elem in root.findall('Metadata'):
        if md_elem.get('domain') not in (None, '', 'default'):
            continue  # skip domain-scoped metadata (IMAGE_STRUCTURE etc.)
        for mdi in md_elem.findall('MDI'):
            if mdi.get('key') == 'AREA_OR_POINT':
                txt = (mdi.text or '').strip().lower()
                if txt == 'point':
                    raster_type = 'point'

    # Bands
    bands = []
    for band_elem in root.findall('VRTRasterBand'):
        band_num = int(band_elem.get('band', 1))
        # ``subClass="VRTDerivedRasterBand"`` (and other future subclasses)
        # signal a band-level computation that read_vrt does not
        # implement. The plain ``<VRTRasterBand>`` mosaic case has no
        # ``subClass`` attribute. Reject so the derived expression is
        # not silently dropped down to whatever sources happen to be
        # listed. Epic #2340 / PR 5.
        _band_sub_class = band_elem.get('subClass')
        if _band_sub_class:
            raise UnsupportedGeoTIFFFeatureError(
                f"VRTRasterBand band={band_num} declares "
                f"subClass={_band_sub_class!r}, which is not supported "
                f"by read_vrt. Only the plain ``<VRTRasterBand>`` "
                f"mosaic case is implemented; derived / pixel-function "
                f"bands are listed as unsupported in "
                f"`xrspatial.geotiff.SUPPORTED_FEATURES` (epic #2340)."
            )
        # Distinguish "attribute missing" (GDAL default: Float32) from
        # "attribute present but unsupported".  The previous
        # ``_DTYPE_MAP.get(dtype_name, np.float32)`` collapsed both cases
        # to ``Float32``, so a VRT declaring ``CFloat32`` silently lost
        # its imaginary component and one declaring a typo (``Flaot32``)
        # silently produced wrong values.  See issue #1783.
        dtype_name = band_elem.get('dataType')
        if dtype_name is None:
            dtype_name = 'Float32'
        if dtype_name not in _DTYPE_MAP:
            supported = ', '.join(sorted(_DTYPE_MAP))
            raise ValueError(
                f"VRTRasterBand band={band_num} declares "
                f"dataType={dtype_name!r}, which is not supported by "
                f"read_vrt. Supported dataType values: {supported}. "
                f"Complex dataType values (CInt16, CInt32, CFloat32, "
                f"CFloat64) are not supported because the reader has no "
                f"complex-data code path; falling back to Float32 would "
                f"silently drop the imaginary component."
            )
        dtype = np.dtype(_DTYPE_MAP[dtype_name])
        # Parse <NoDataValue> with dtype-aware precision.  For integer
        # bands (including UInt64 / Int64) the sentinel is parsed as an
        # ``int`` so values like ``2**64 - 1`` or ``INT64_MIN`` round-trip
        # exactly through the masking pipeline; ``float(text)`` would
        # collapse those to a representable float64 and break exact-
        # equality masks downstream.  See issue #1783 follow-up.
        nodata_str = _text(band_elem, 'NoDataValue')
        nodata = _parse_band_nodata(nodata_str, dtype)
        color_interp = _text(band_elem, 'ColorInterp')

        # Band-children classification is driven by the module-scope
        # ``_INFORMATIONAL_BAND_TAGS`` / ``_SOURCE_BAND_TAGS`` /
        # ``_UNSUPPORTED_BAND_TAGS`` frozensets above. The pure-metadata
        # ones stay ignored; the output-altering ones (kernel filters,
        # derived band expressions, pansharpening) get rejected so a
        # VRT relying on them cannot read as a plain mosaic and lose
        # its declared computation. See epic #2340 / PR 5.
        sources = []
        for src_elem in band_elem:
            tag = src_elem.tag
            if tag in _INFORMATIONAL_BAND_TAGS:
                continue
            if tag in _UNSUPPORTED_BAND_TAGS:
                raise UnsupportedGeoTIFFFeatureError(
                    f"VRTRasterBand band={band_num} contains a "
                    f"<{tag}> element, which declares a VRT feature "
                    f"(kernel-filtered / derived / pansharpened / "
                    f"mask-band source) that read_vrt does not "
                    f"implement. Only ``<SimpleSource>`` and "
                    f"``<ComplexSource>`` are supported. See "
                    f"`xrspatial.geotiff.SUPPORTED_FEATURES` for the "
                    f"release tier map (epic #2340)."
                )
            if tag not in _SOURCE_BAND_TAGS:
                # Unknown band child that is neither informational nor a
                # known unsupported-feature marker. Refuse rather than
                # silently dropping the element: a future GDAL VRT
                # extension that alters pixel computation must not pass
                # through this reader as if it were a no-op.
                raise UnsupportedGeoTIFFFeatureError(
                    f"VRTRasterBand band={band_num} contains an unknown "
                    f"<{tag}> element. read_vrt only recognises "
                    f"``<SimpleSource>`` and ``<ComplexSource>`` sources "
                    f"plus informational metadata children; an unknown "
                    f"element could change the declared pixel output. "
                    f"See `xrspatial.geotiff.SUPPORTED_FEATURES` for "
                    f"the release tier map (epic #2340)."
                )

            filename = _text(src_elem, 'SourceFilename') or ''
            relative = src_elem.find('SourceFilename')
            is_relative = (relative is not None and
                           relative.get('relativeToVRT', '0') == '1')
            if is_relative and not os.path.isabs(filename):
                filename = os.path.join(vrt_dir, filename)
            # Canonicalize so ``..`` segments and symlinks both resolve
            # to their target before the containment check below.
            filename = os.path.realpath(filename)

            # Containment policy (issue #1671):
            #
            # * ``relativeToVRT='1'`` declares the source lives under the
            #   VRT directory.  Honour that intent even when the allowlist
            #   would otherwise cover the resolved path -- a relative
            #   source that escapes ``vrt_dir`` is always an attack
            #   primitive ('..' segments, symlink swap, etc.).
            # * Absolute sources are accepted when they resolve under
            #   ``vrt_dir`` (matches the writer's ``relative=False`` round
            #   trip) or under any explicit ``XRSPATIAL_VRT_ALLOWED_ROOTS``
            #   entry.
            #
            # In every other case raise ``ValueError`` so a crafted VRT
            # cannot read arbitrary files the process has access to.
            if is_relative:
                trusted = [vrt_root]
            else:
                trusted = [vrt_root, *allowed_roots]
            if not any(_path_is_under(filename, r) for r in trusted):
                # Use direct interpolation (not !r) for ``filename`` and
                # ``vrt_root`` so Windows paths render with single
                # backslashes rather than the doubled escapes repr emits.
                # The explicit quotes keep the boundary readable when the
                # path has embedded spaces. ``allowed_roots`` stays as !r
                # for its list-of-strings render.
                raise ValueError(
                    f"VRT SourceFilename '{filename}' resolves outside "
                    f"the VRT directory ('{vrt_root}') and is not "
                    f"covered by any {_ALLOWED_ROOTS_ENV} entry "
                    f"({allowed_roots!r}). Refusing to read; see issue "
                    f"#1671."
                )

            src_band = int(_text(src_elem, 'SourceBand') or '1')

            src_rect_elem = src_elem.find('SrcRect')
            dst_rect_elem = src_elem.find('DstRect')
            if src_rect_elem is None or dst_rect_elem is None:
                continue

            src_rect = _parse_rect(src_rect_elem)
            dst_rect = _parse_rect(dst_rect_elem)

            # Per-source ``<NODATA>`` honours the band dtype too -- a
            # uint64 source feeding a UInt64 band needs the full 64-bit
            # sentinel preserved, not a float64-rounded approximation.
            # See issue #1783 follow-up.
            src_nodata_str = _text(src_elem, 'NODATA')
            src_nodata = _parse_band_nodata(src_nodata_str, dtype)

            # ComplexSource extras
            scale = None
            offset = None
            resample_alg = None
            if tag == 'ComplexSource':
                scale_str = _text(src_elem, 'ScaleOffset')
                offset_str = _text(src_elem, 'ScaleRatio')
                # Note: GDAL uses ScaleOffset=offset, ScaleRatio=scale
                if offset_str:
                    scale = float(offset_str)
                if scale_str:
                    offset = float(scale_str)
                # ``<ResampleAlg>`` records the requested resampler for
                # the placement step.  ``read_vrt`` only implements
                # nearest-neighbour today, so when ``SrcRect`` and
                # ``DstRect`` sizes differ the resample site raises
                # ``NotImplementedError`` for any non-nearest algorithm
                # rather than returning silently-mislabelled pixels;
                # sources with matching rect sizes do not resample and
                # pass through regardless of this tag.  See issues
                # #1694 and #1751.
                resample_alg = _text(src_elem, 'ResampleAlg')

            sources.append(_Source(
                filename=filename,
                band=src_band,
                src_rect=src_rect,
                dst_rect=dst_rect,
                nodata=src_nodata,
                scale=scale,
                offset=offset,
                resample_alg=resample_alg,
            ))

        bands.append(_VRTBand(
            band_num=band_num,
            dtype=dtype,
            nodata=nodata,
            sources=sources,
            color_interp=color_interp,
        ))

    return VRTDataset(
        width=width,
        height=height,
        crs_wkt=crs_wkt,
        geo_transform=geo_transform,
        bands=bands,
        raster_type=raster_type,
    )


# GDAL ``<ResampleAlg>`` values that are equivalent to nearest-neighbour
# (or that explicitly request it).  Comparison is case-insensitive and
# done on the stripped element text.  Empty / missing text is also
# treated as nearest because the GDAL default for a SimpleSource with
# no ``ResampleAlg`` child is nearest.  See issue #1751.
_NEAREST_RESAMPLE_ALGS = frozenset({
    '', 'nearest', 'nearestneighbour', 'nearestneighbor', 'near',
})


def _check_resample_alg_supported(resample_alg: str | None,
                                  filename: str) -> None:
    """Raise ``NotImplementedError`` for unsupported VRT resample algorithms.

    ``ComplexSource`` may declare ``<ResampleAlg>`` values such as
    ``Bilinear``, ``Cubic``, ``CubicSpline``, ``Lanczos``, ``Average``,
    or ``Mode``.  ``read_vrt`` only implements nearest-neighbour
    placement, so a VRT that asks for any of the higher-quality
    resamplers would silently return nearest-sampled pixels mislabelled
    as the requested algorithm.  Refuse the read instead of returning
    quietly wrong numbers.  See issue #1751.
    """
    if resample_alg is None:
        return
    norm = resample_alg.strip().lower()
    if norm in _NEAREST_RESAMPLE_ALGS:
        return
    raise NotImplementedError(
        f"VRT ComplexSource for '{filename}' requests "
        f"<ResampleAlg>{resample_alg}</ResampleAlg>, but read_vrt only "
        f"implements nearest-neighbour resampling. Returning "
        f"nearest-sampled pixels would silently mislabel them as "
        f"'{resample_alg}'. Re-export the VRT with "
        f"<ResampleAlg>Nearest</ResampleAlg> or matching SrcRect/DstRect "
        f"sizes if nearest-neighbour is acceptable. See issue #1751."
    )


def _resample_nearest(src_arr: np.ndarray,
                      out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resample ``src_arr`` to ``(out_h, out_w)``.

    Mirrors GDAL's ``SimpleSource`` semantics: each output pixel samples
    the source pixel whose centre is closest in the source grid. The
    centre-of-pixel mapping is ``src_idx = (out_idx + 0.5) * src_size /
    out_size``; ``floor`` of that, clamped into range, gives the index.

    Used by :func:`read_vrt` to honour ``<SrcRect>``/``<DstRect>``
    scaling.  See issue #1694 -- before this helper, a source array was
    pasted directly into the destination with no resample step, which
    raised a broadcast error for downsampled sources and left holes for
    upsampled ones.

    Integer-ratio cases short-circuit to ``np.repeat`` (integer
    upsample) or strided slicing (integer downsample) to avoid building
    an index array for the common GDAL ``SrcRect`` shapes.
    """
    src_h, src_w = src_arr.shape[:2]
    # Reject an empty source up front: a SimpleSource with SrcRect
    # ``xSize=0`` / ``ySize=0`` (or a windowed read that clamps to an
    # empty slice) would otherwise reach the integer-ratio fast paths
    # below and divide / modulo by zero on ``src_h`` or ``src_w``. Raise
    # an explicit ValueError so the bad VRT surfaces with a clear cause
    # instead of an opaque ZeroDivisionError.
    if src_h == 0 or src_w == 0:
        raise ValueError(
            "_resample_nearest received an empty source array; "
            "the VRT SourceFilename probably has SrcRect with zero size"
        )
    if src_h == out_h and src_w == out_w:
        return src_arr
    # Fast paths for integer ratios -- common when a VRT downsamples by
    # 2x / 4x or upsamples by an integer factor.  ``np.repeat`` is a
    # cheaper nearest-neighbour upsample than the gather below.
    if (out_h % src_h == 0 and out_w % src_w == 0
            and out_h >= src_h and out_w >= src_w):
        ry = out_h // src_h
        rx = out_w // src_w
        return np.repeat(np.repeat(src_arr, ry, axis=0), rx, axis=1)
    if (src_h % out_h == 0 and src_w % out_w == 0
            and src_h >= out_h and src_w >= out_w):
        sy = src_h // out_h
        sx = src_w // out_w
        # Sample the centre of each output pixel's source block (offset
        # by half a block, floored) so the integer-ratio downsample
        # agrees with the general non-integer path for the same shapes.
        return src_arr[sy // 2::sy, sx // 2::sx][:out_h, :out_w]
    # General case: build per-axis nearest indices and gather.  ``floor``
    # plus a clamp matches GDAL's nearest-neighbour rounding for
    # non-integer ratios.
    y_idx = np.minimum(
        np.floor((np.arange(out_h) + 0.5) * src_h / out_h).astype(np.intp),
        src_h - 1,
    )
    x_idx = np.minimum(
        np.floor((np.arange(out_w) + 0.5) * src_w / out_w).astype(np.intp),
        src_w - 1,
    )
    return src_arr[y_idx[:, None], x_idx[None, :]]


def _nn_src_index(out_idx: int, src_size: int, out_size: int) -> int:
    """Return the source pixel index a single output pixel samples.

    Implements the same ``floor((out_idx + 0.5) * src_size / out_size)``
    rule as :func:`_resample_nearest`, with the same clamp into
    ``[0, src_size - 1]``. Used to compute the inverse mapping from a
    clipped destination sub-window back to the minimal source rect that
    feeds it. See issue #1704.
    """
    idx = int(math.floor((out_idx + 0.5) * src_size / out_size))
    if idx < 0:
        return 0
    if idx > src_size - 1:
        return src_size - 1
    return idx


def _resample_nearest_window(src_sub: np.ndarray,
                             src_origin_y: int, src_origin_x: int,
                             full_src_h: int, full_src_w: int,
                             full_dst_h: int, full_dst_w: int,
                             dst_r0: int, dst_c0: int,
                             dst_r1: int, dst_c1: int) -> np.ndarray:
    """Nearest-neighbour resample a source sub-rect into a destination
    sub-window, using the same per-pixel mapping as
    :func:`_resample_nearest` applied to the full source rect.

    ``src_sub`` is the slice of the full source rect starting at
    ``(src_origin_y, src_origin_x)`` in full-source coordinates. The
    output covers destination rows ``[dst_r0, dst_r1)`` and columns
    ``[dst_c0, dst_c1)`` in full-destination coordinates. Mapping each
    output pixel back through ``floor((out + 0.5) * src_size /
    out_size)`` reproduces the index ``_resample_nearest`` would compute
    on the full source array; subtracting the origin then indexes into
    ``src_sub``. This yields the same numbers as resampling the full
    source rect and slicing ``[dst_r0:dst_r1, dst_c0:dst_c1]`` afterward.
    See issue #1704.
    """
    out_h = dst_r1 - dst_r0
    out_w = dst_c1 - dst_c0
    # ``floor((dst_* + 0.5) * src_size / dst_size)`` with the same clamp
    # to ``src_size - 1`` as ``_resample_nearest`` uses on the full rect.
    y_full = np.minimum(
        np.floor((np.arange(dst_r0, dst_r1) + 0.5)
                 * full_src_h / full_dst_h).astype(np.intp),
        full_src_h - 1,
    )
    x_full = np.minimum(
        np.floor((np.arange(dst_c0, dst_c1) + 0.5)
                 * full_src_w / full_dst_w).astype(np.intp),
        full_src_w - 1,
    )
    y_idx = y_full - src_origin_y
    x_idx = x_full - src_origin_x
    sub_h, sub_w = src_sub.shape[:2]
    # Defensive clamp: callers compute the sub-window via the same
    # mapping so y_idx / x_idx already lie in ``[0, sub_h)`` /
    # ``[0, sub_w)``, but a rounding edge with very large sizes could
    # nudge an index by one. Clip to keep the gather inside the buffer.
    np.clip(y_idx, 0, sub_h - 1, out=y_idx)
    np.clip(x_idx, 0, sub_w - 1, out=x_idx)
    if out_h == 0 or out_w == 0:
        return src_sub[:0, :0]
    return src_sub[y_idx[:, None], x_idx[None, :]]


# ---------------------------------------------------------------------------
# Shared helpers used by both the eager VRT read path (this module) and the
# chunked dask path in ``xrspatial.geotiff.__init__._read_vrt_chunked``.
# Centralised here so both call sites agree on dtype promotion, sentinel
# masking, and effective-dtype computation. See issue #1825.
# ---------------------------------------------------------------------------


def _sentinel_for_dtype(nodata_val, dtype):
    """Return ``dtype``-cast sentinel for ``nodata_val`` or ``None``.

    ``None`` is returned when the value cannot be represented in ``dtype``
    (non-integer dtype, out-of-range, non-finite, or fractional). Mirrors
    the gating used by other read paths via ``_int_nodata_in_range``.

    A plain Python ``int`` ``nodata_val`` is handled without going through
    ``float`` first, so 64-bit sentinels such as ``2**64 - 1`` (``UInt64``
    max) and ``-2**63`` (``Int64`` min) round-trip without the float64
    rounding that pushes them past the dtype's representable range.
    ``_parse_band_nodata`` parses integer-band ``<NoDataValue>`` directly
    as ``int`` to feed this path. See issue #1783 follow-up.

    This is the single shared implementation used by both the eager path
    in :func:`read_vrt` and the chunked path in
    :func:`xrspatial.geotiff._read_vrt_chunked`; previously a closure in
    the eager path and a module-level twin in the chunked path duplicated
    the logic (issue #1825).
    """
    if nodata_val is None or dtype.kind not in ('u', 'i'):
        return None
    info = np.iinfo(dtype)
    # Fast/exact path: ``nodata_val`` is already an integer. Avoids the
    # float64 round-trip that loses precision near the int64 / uint64
    # extremes. ``bool`` is a subclass of ``int`` -- treat True/False as
    # a 1/0 sentinel rather than rejecting outright, matching the
    # existing ``int(float(...))`` behaviour.
    if isinstance(nodata_val, (int, np.integer)) and not isinstance(
            nodata_val, bool):
        nodata_int = int(nodata_val)
        if info.min <= nodata_int <= info.max:
            return dtype.type(nodata_int)
        return None
    try:
        nodata_f = float(nodata_val)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(nodata_f) and nodata_f.is_integer()
            and info.min <= nodata_f <= info.max):
        return None
    return dtype.type(int(nodata_f))


def _effective_dtype_for_bands(selected_bands) -> np.dtype:
    """Return the output buffer dtype that holds every selected band losslessly.

    Computes ``np.result_type`` over each band's effective dtype, where
    each band's effective dtype is widened to ``float64`` if any of its
    ``ComplexSource`` declarations apply a non-identity ``ScaleRatio``
    (``scale``) or ``ScaleOffset`` (``offset``). Mirrors the historic
    inline computation in :func:`read_vrt` and matches the parse-time
    declared dtype the chunked path emits up front. Issue #1825.
    """
    if not selected_bands:
        raise ValueError(
            "VRT has no <VRTRasterBand> elements; cannot determine "
            "output dtype"
        )
    effective_dtypes = []
    for vrt_band in selected_bands:
        eff = vrt_band.dtype
        for src in vrt_band.sources:
            scaled = src.scale is not None and src.scale != 1.0
            offset = src.offset is not None and src.offset != 0.0
            if scaled or offset:
                eff = np.dtype(np.float64)
                break
        effective_dtypes.append(eff)
    return np.result_type(*effective_dtypes)


def _apply_integer_sentinel_mask(arr, vrt, band):
    """NaN-mask integer sentinels in a freshly decoded VRT buffer.

    Mirrors the post-decode integer-promotion branch the eager path
    applies after :func:`read_vrt` and the chunked path applies inside
    each per-chunk task. Walks the relevant ``vrt.bands`` entry / entries,
    promotes ``arr`` to ``float64`` on the first sentinel hit, and rewrites
    matching pixels to ``NaN`` in place on the promoted view.

    Returns the (possibly promoted) ``arr``. The internal reader already
    NaN-masks float source arrays inline; this helper only fires for
    integer-dtype outputs paired with an integer ``<NoDataValue>``.
    Issue #1825.
    """
    if arr.dtype.kind not in ('u', 'i'):
        return arr
    if arr.ndim == 3 and band is None and vrt.bands:
        int_arr = arr
        int_dtype = int_arr.dtype
        for i, vrt_band in enumerate(vrt.bands):
            if i >= int_arr.shape[-1]:
                break
            sentinel = _sentinel_for_dtype(vrt_band.nodata, int_dtype)
            if sentinel is None:
                continue
            mask = int_arr[..., i] == sentinel
            if not mask.any():
                continue
            if arr.dtype != np.float64:
                arr = arr.astype(np.float64)
            arr[..., i][mask] = np.nan
        return arr
    band_idx = band if band is not None else 0
    nodata = None
    if vrt.bands and 0 <= band_idx < len(vrt.bands):
        nodata = vrt.bands[band_idx].nodata
    if nodata is None:
        return arr
    sentinel = _sentinel_for_dtype(nodata, arr.dtype)
    if sentinel is None:
        return arr
    mask = arr == sentinel
    if mask.any():
        arr = arr.astype(np.float64)
        arr[mask] = np.nan
    return arr


def _apply_integer_sentinel_mask_with_presence(arr, vrt, band):
    """Same as :func:`_apply_integer_sentinel_mask` but also reports presence.

    Returns ``(arr, pixels_present)`` where ``pixels_present`` is a bool
    indicating whether any sentinel pixel was found in the integer
    buffer before masking. Used by the VRT eager and chunked paths to
    surface ``attrs['nodata_pixels_present']`` (issue #2135) without
    rescanning the (possibly promoted) float buffer.

    Kept as a sibling of :func:`_apply_integer_sentinel_mask` rather
    than collapsed into one helper with an opt-in flag because the
    original helper is shared with the chunked-VRT per-task path that
    cannot afford an extra Python-side branch per chunk. A future
    refactor can unify them once a third variant is needed.
    """
    if arr.dtype.kind not in ('u', 'i'):
        return arr, False
    if arr.ndim == 3 and band is None and vrt.bands:
        int_arr = arr
        int_dtype = int_arr.dtype
        any_hit = False
        for i, vrt_band in enumerate(vrt.bands):
            if i >= int_arr.shape[-1]:
                break
            sentinel = _sentinel_for_dtype(vrt_band.nodata, int_dtype)
            if sentinel is None:
                continue
            mask = int_arr[..., i] == sentinel
            if not mask.any():
                continue
            any_hit = True
            if arr.dtype != np.float64:
                arr = arr.astype(np.float64)
            arr[..., i][mask] = np.nan
        return arr, any_hit
    band_idx = band if band is not None else 0
    nodata = None
    if vrt.bands and 0 <= band_idx < len(vrt.bands):
        nodata = vrt.bands[band_idx].nodata
    if nodata is None:
        return arr, False
    sentinel = _sentinel_for_dtype(nodata, arr.dtype)
    if sentinel is None:
        return arr, False
    mask = arr == sentinel
    present = bool(mask.any())
    if present:
        arr = arr.astype(np.float64)
        arr[mask] = np.nan
    return arr, present


def _scan_for_sentinel(arr, vrt, band):
    """Detect whether ``arr`` contains any pixel matching a declared sentinel.

    Used by the VRT eager path when ``mask_nodata=False`` so
    ``attrs['nodata_pixels_present']`` (issue #2135) still surfaces a
    meaningful answer even though the masking branch was skipped. The
    multi-band case checks each band against its own ``<NoDataValue>``;
    the single-band case checks the selected band's sentinel.

    Returns a plain bool. ``False`` covers the no-declared-sentinel and
    sentinel-out-of-range cases (mirrors the masking-branch gates).
    """
    if arr.ndim == 3 and band is None and vrt.bands:
        for i, vrt_band in enumerate(vrt.bands):
            if i >= arr.shape[-1]:
                break
            if vrt_band.nodata is None:
                continue
            if arr.dtype.kind == 'f':
                if np.isnan(vrt_band.nodata):
                    if bool(np.isnan(arr[..., i]).any()):
                        return True
                else:
                    if bool((arr[..., i]
                             == arr.dtype.type(vrt_band.nodata)).any()):
                        return True
            else:
                sentinel = _sentinel_for_dtype(vrt_band.nodata, arr.dtype)
                if sentinel is None:
                    continue
                if bool((arr[..., i] == sentinel).any()):
                    return True
        return False
    band_idx = band if band is not None else 0
    nodata = None
    if vrt.bands and 0 <= band_idx < len(vrt.bands):
        nodata = vrt.bands[band_idx].nodata
    if nodata is None:
        return False
    if arr.dtype.kind == 'f':
        if np.isnan(nodata):
            return bool(np.isnan(arr).any())
        return bool((arr == arr.dtype.type(nodata)).any())
    sentinel = _sentinel_for_dtype(nodata, arr.dtype)
    if sentinel is None:
        return False
    return bool((arr == sentinel).any())


def read_vrt(vrt_path: str, *, window=None,
             band: int | None = None,
             max_pixels: int | None = None,
             missing_sources: str = 'raise',
             parsed: VRTDataset | None = None,
             mask_nodata: bool = True,
             ) -> tuple[np.ndarray, VRTDataset]:
    """Read a VRT file by assembling pixel data from its source files.

    Do not call this symbol directly from external code. Release-contract
    tier (epic #2340): this is the [internal-only] pixel-assembly
    helper. The public surface lives in
    ``xrspatial.geotiff.read_vrt`` (re-exported from
    ``_backends/vrt.py``) and carries the [advanced] tier; this
    function is what that public wrapper calls into. See
    ``docs/source/reference/release_gate_geotiff.rst`` and
    ``docs/source/reference/geotiff_release_contract.rst`` for the
    contract. Direct calls into this symbol bypass the dispatcher-level
    validation in ``_validate_dispatch_kwargs`` and are not part of the
    public API.

    Parameters
    ----------
    vrt_path : str
        [internal-only] Path to the .vrt file.
    window : tuple or None
        [internal-only] (row_start, col_start, row_stop, col_stop) for
        windowed read.
    band : int or None
        [internal-only] Band index (0-based). None returns all bands.
    max_pixels : int or None
        [internal-only] Maximum allowed pixel count
        (width * height * samples) for the assembled VRT region. None
        uses the reader default.
    missing_sources : {'raise', 'warn'}, default 'raise'
        [internal-only] Policy for unreadable source files referenced
        by the VRT.
        ``'raise'`` (the default) fails immediately on an unreadable
        source so a partial mosaic never surfaces silently. This matches
        the rest of the geotiff module's up-front rejection of malformed
        input and avoids the ambiguity of a zero-fill hole on an integer
        raster without a nodata sentinel (see issue #1843). Prior to
        #1843 the default was ``'warn'``; callers that relied on the
        lenient behaviour should pass ``missing_sources='warn'``
        explicitly.
        ``'warn'`` is the opt-in escape hatch for partial mosaics: it
        emits ``GeoTIFFFallbackWarning`` and records the skipped source
        on ``vrt.holes`` (surfaced as ``attrs['vrt_holes']`` on the
        public DataArray).
        ``XRSPATIAL_GEOTIFF_STRICT=1`` forces a raise across the whole
        module regardless of this kwarg (see issue #1662).
    parsed : VRTDataset or None
        [internal-only] Pre-parsed VRT structure. When supplied,
        ``vrt_path`` is not
        re-read or re-parsed and the source-path containment check is
        skipped (the supplied ``VRTDataset`` is assumed to have been
        produced by :func:`parse_vrt` already, which performs the check).
        Used by the chunked dask path (issue #1825) so each per-chunk
        task can skip the redundant XML parse and allowlist validation.
    mask_nodata : bool, default True
        [internal-only] If True (the default), float source bands have
        their declared
        nodata sentinel rewritten to NaN inline during assembly, and
        integer sources feeding a float-dataType VRT have their
        sentinel rewritten to NaN as part of the int->float
        placement. If False, both inline masking branches are skipped
        so the literal sentinel value survives to the public backend
        layer, which can then honor a caller's ``mask_nodata=False``
        opt-out symmetrically for float and integer source dtypes.
        See issue #2158.

    Returns
    -------
    (np.ndarray, VRTDataset) tuple
    """
    from ._reader import PixelSafetyLimitError, read_to_array

    if parsed is not None:
        # Shallow-copy with a fresh ``holes`` list. ``read_vrt`` appends
        # to ``vrt.holes`` on missing/unreadable sources, and under
        # chunked dispatch (issue #1825) the same ``parsed`` instance is
        # threaded into every per-chunk task. Mutating the shared list
        # would leak skipped-source records across tasks (racy growth
        # under the threaded scheduler, and cumulative duplication
        # across calls if a caller ever reused the parsed object). The
        # dataclass replace is O(1) over a handful of fields; the bands
        # / sources / dtypes references are intentionally shared.
        vrt = _dc_replace(parsed, holes=[])
    else:
        xml_str = _read_vrt_xml(vrt_path)
        vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
        vrt = parse_vrt(xml_str, vrt_dir)
    if missing_sources not in ('warn', 'raise'):
        raise ValueError(
            f"missing_sources must be 'warn' or 'raise', got "
            f"{missing_sources!r}")

    # Validate ``band`` against the parsed band count. Python list
    # indexing would silently accept negative values (``vrt.bands[-1]``
    # returns the last band) and raise an opaque ``IndexError`` for
    # out-of-range positive values, neither of which is the contract the
    # public API documents (``band`` is a 0-based non-negative index). Reject
    # both up front with a clear ``ValueError`` so callers do not get
    # band-N data paired with band-0's nodata sentinel or a downstream
    # IndexError from deep in the read path.
    if band is not None:
        if not isinstance(band, (int, np.integer)) or isinstance(band, bool):
            raise ValueError(
                f"band must be a non-negative int, got {band!r}")
        if band < 0 or band >= len(vrt.bands):
            raise ValueError(
                f"band index {band} out of range for VRT with "
                f"{len(vrt.bands)} band(s)")

    # Validate ``window`` against the VRT's parsed extent before any
    # source reads. Prior to #1697, per-source reads silently clamped
    # out-of-bounds windows to the source extent and returned a smaller
    # array, which then mismatched caller-built coord arrays in
    # ``open_geotiff`` and surfaced as an opaque
    # ``CoordinateValidationError`` far from the real cause. We now
    # reject invalid windows up front. Mirrors the local-path validator
    # in ``read_to_array`` (#1634) and the HTTP path validator in
    # ``_read_cog_http`` (#1669) so all backends agree on the contract.
    # See issue #1697.
    if window is not None:
        r0, c0, r1, c1 = window
        if (r0 < 0 or c0 < 0
                or r1 > vrt.height or c1 > vrt.width
                or r0 >= r1 or c0 >= c1):
            raise ValueError(
                f"window={window} is outside the VRT extent "
                f"({vrt.height}x{vrt.width}) or has non-positive size.")
    else:
        r0, c0, r1, c1 = 0, 0, vrt.height, vrt.width

    out_h = r1 - r0
    out_w = c1 - c0

    from ._reader import MAX_PIXELS_DEFAULT, _check_dimensions
    if max_pixels is None:
        max_pixels = MAX_PIXELS_DEFAULT
    n_bands = len([vrt.bands[band]] if band is not None else vrt.bands)
    _check_dimensions(out_w, out_h, n_bands, max_pixels)

    # Select bands
    if band is not None:
        selected_bands = [vrt.bands[band]]
    else:
        selected_bands = vrt.bands

    # Allocate output.
    #
    # The output buffer dtype must be wide enough to hold every selected
    # band losslessly. A naive ``selected_bands[0].dtype`` would silently
    # truncate any wider band's values when the per-band placement at the
    # ``result[...] = src_arr[...]`` line below casts to the buffer dtype.
    # Two sources of widening matter:
    #
    # * Heterogeneous declared dtypes across bands (e.g. ``Byte`` + ``Float32``).
    # * ``ComplexSource`` ``ScaleRatio`` / ``ScaleOffset`` promote source
    #   values to ``float64`` before placement (see the ``# Apply
    #   ComplexSource scaling`` block later in this function); the
    #   destination has to be float-typed too, otherwise the fractional
    #   part is lost.
    #
    # ``np.result_type`` produces the common dtype that holds every
    # contributing dtype without loss. For an all-integer VRT it stays
    # integer; for mixes it widens to the common dtype (typically
    # ``float64`` when integer and floating-point bands mix, but it can
    # be a narrower float -- e.g. ``float32`` under NumPy 2.x for
    # ``uint8`` + ``float32`` -- or ``complex128`` when complex bands
    # are present). See issue #1696.
    #
    # Guard against a malformed VRT with zero ``<VRTRasterBand>``
    # elements: ``np.result_type()`` with no args raises a generic
    # "at least one array or dtype is required" message that gives the
    # caller no hint about the underlying cause. The helper raises
    # ``ValueError`` for the empty case with that explicit message.
    dtype = _effective_dtype_for_bands(selected_bands)
    fill = np.nan if dtype.kind in ('f', 'c') else 0
    if len(selected_bands) == 1:
        result = np.full((out_h, out_w), fill, dtype=dtype)
    else:
        result = np.full((out_h, out_w, len(selected_bands)), fill, dtype=dtype)

    for band_idx, vrt_band in enumerate(selected_bands):
        nodata = vrt_band.nodata

        for src in vrt_band.sources:
            # Compute overlap between source's destination rect and our window
            dr = src.dst_rect
            sr = src.src_rect

            # Reject malformed negative DstRect sizes up front, before the
            # overlap math.  A negative ``xSize`` / ``ySize`` would otherwise
            # produce ``dst_*1 < dst_*0`` and the source would be silently
            # skipped by the overlap continue below; we'd rather surface the
            # malformed VRT to the caller than treat it as a no-op tile.
            # See issue #1737.
            if dr.x_size < 0 or dr.y_size < 0:
                raise ValueError(
                    f"VRT SimpleSource DstRect has negative size "
                    f"(xSize={dr.x_size}, ySize={dr.y_size}); "
                    f"DstRect sizes must be non-negative."
                )

            # Reject malformed SrcRect values up front, before the overlap
            # math and before the lenient source-read try/except below. A
            # negative ``xSize`` / ``ySize`` would otherwise reach
            # ``read_to_array`` as a bad window and be silently swallowed by
            # the missing-source fallback (which is meant for unreadable
            # files, not malformed XML rectangles); a negative ``xOff`` /
            # ``yOff`` likewise produces an out-of-range window that the
            # fallback would turn into a zero-filled hole. See issue #1784.
            if sr.x_size < 0 or sr.y_size < 0:
                raise ValueError(
                    f"VRT SimpleSource SrcRect has negative size "
                    f"(xSize={sr.x_size}, ySize={sr.y_size}); "
                    f"SrcRect sizes must be non-negative."
                )
            if sr.x_off < 0 or sr.y_off < 0:
                raise ValueError(
                    f"VRT SimpleSource SrcRect has negative offset "
                    f"(xOff={sr.x_off}, yOff={sr.y_off}); "
                    f"SrcRect offsets must be non-negative."
                )

            # Destination rect in virtual raster coordinates
            dst_r0 = dr.y_off
            dst_c0 = dr.x_off
            dst_r1 = dr.y_off + dr.y_size
            dst_c1 = dr.x_off + dr.x_size

            # Clip to window
            clip_r0 = max(dst_r0, r0)
            clip_c0 = max(dst_c0, c0)
            clip_r1 = min(dst_r1, r1)
            clip_c1 = min(dst_c1, c1)

            if clip_r0 >= clip_r1 or clip_c0 >= clip_c1:
                continue  # no overlap

            # SrcRect / DstRect scaling.  When sizes match (the common
            # SimpleSource case GDAL writes for tile mosaics), the source
            # rect maps 1:1 to the destination rect and we can issue a
            # windowed read for only the overlap region -- cheaper than
            # decoding the whole source rect.  When the sizes differ the
            # source band is being upsampled or downsampled into its
            # destination cell, so we read the full source rect, apply
            # nodata masking, resample to ``(dr.y_size, dr.x_size)`` with
            # nearest-neighbour (matching GDAL's SimpleSource semantics),
            # and *then* take the clip subwindow.  Reading the whole
            # source rect first keeps the resample math local to the
            # source/destination shapes; trying to resample a windowed
            # source slice independently introduces fence-post errors at
            # the clip boundary and breaks the integer-ratio fast paths.
            # See issue #1694.
            needs_resample = (sr.y_size != dr.y_size
                              or sr.x_size != dr.x_size)
            if needs_resample:
                # Clip-relative destination sub-window, in full-DstRect
                # coordinates.  These are the destination rows / cols
                # the caller actually wants out of this source.
                sub_r0 = clip_r0 - dst_r0
                sub_c0 = clip_c0 - dst_c0
                sub_r1 = clip_r1 - dst_r0
                sub_c1 = clip_c1 - dst_c0

                # Invert the nearest-neighbour mapping: find the smallest
                # SrcRect-relative range of rows / cols that
                # ``_resample_nearest`` would gather from when producing
                # destination rows ``[sub_r0, sub_r1)`` and columns
                # ``[sub_c0, sub_c1)``.  Each destination pixel ``d``
                # samples source pixel ``floor((d + 0.5) * src / dst)``
                # clamped to ``src - 1``; taking that mapping at the two
                # endpoints bounds the source range we have to read.
                # ``+ 1`` on the stop makes the range half-open, matching
                # the read_to_array window convention. See issue #1704.
                src_row_min = _nn_src_index(
                    sub_r0, sr.y_size, dr.y_size)
                src_row_max = _nn_src_index(
                    sub_r1 - 1, sr.y_size, dr.y_size)
                src_col_min = _nn_src_index(
                    sub_c0, sr.x_size, dr.x_size)
                src_col_max = _nn_src_index(
                    sub_c1 - 1, sr.x_size, dr.x_size)
                read_r0 = sr.y_off + src_row_min
                read_c0 = sr.x_off + src_col_min
                read_r1 = sr.y_off + src_row_max + 1
                read_c1 = sr.x_off + src_col_max + 1

                # Cap the resample intermediate before the source read.
                # ``_resample_nearest_window`` below allocates an output
                # of ``(sub_r1 - sub_r0, sub_c1 - sub_c0)`` pixels, which
                # is bounded by the user's window.  The cap previously
                # protected against a crafted VRT declaring a huge
                # DstRect even on a tiny output; with the windowed read
                # the intermediate is now bounded by the window, so the
                # pixel-budget check applies to the clipped sub-window
                # rather than the full DstRect.  See issues #1737 and
                # #1704.
                sub_dst_h = sub_r1 - sub_r0
                sub_dst_w = sub_c1 - sub_c0
                if (sub_dst_w > max_pixels
                        or sub_dst_h > max_pixels
                        or sub_dst_w * sub_dst_h > max_pixels):
                    raise ValueError(
                        f"VRT SimpleSource DstRect "
                        f"(xSize={dr.x_size}, ySize={dr.y_size}) requires "
                        f"a resample intermediate of "
                        f"{sub_dst_w * sub_dst_h:,} pixels for the "
                        f"requested window, which exceeds the safety "
                        f"limit of {max_pixels:,} pixels. Pass a larger "
                        f"max_pixels= to read_vrt() if this file is "
                        f"legitimate."
                    )
            else:
                read_r0 = sr.y_off + (clip_r0 - dst_r0)
                read_c0 = sr.x_off + (clip_c0 - dst_c0)
                read_r1 = sr.y_off + (clip_r1 - dst_r0)
                read_c1 = sr.x_off + (clip_c1 - dst_c0)

            # Read from source file using windowed read.
            #
            # Narrow the catch to the exception families ``read_to_array``
            # actually documents/raises for an unreadable or malformed
            # source: ``OSError`` (and subclasses ``FileNotFoundError`` /
            # ``PermissionError``) for I/O problems, ``ValueError`` for the
            # typed parse errors from ``parse_header`` / ``parse_ifd`` and
            # friends, ``struct.error`` which still leaks from a few parse
            # paths until that work lands, and the codec-library decode
            # exceptions enumerated in :data:`_CODEC_DECODE_EXCEPTIONS`
            # (``zlib.error`` for corrupt deflate tiles, plus
            # ``zstandard.ZstdError`` when zstandard is installed).
            # ``RuntimeError``, ``MemoryError``, and other non-I/O bugs
            # should NOT be absorbed by the "skip the tile" fallback --
            # they signal real failures and need to surface to the
            # caller. See issues #1670 and PR #1675.
            try:
                src_arr, _ = read_to_array(
                    src.filename,
                    window=(read_r0, read_c0, read_r1, read_c1),
                    band=src.band - 1,  # convert 1-based to 0-based
                    max_pixels=max_pixels,
                )
            except (
                OSError, ValueError, struct.error,
            ) + _CODEC_DECODE_EXCEPTIONS as e:
                if isinstance(e, PixelSafetyLimitError):
                    raise
                # Default (``missing_sources='raise'``) surfaces the read
                # failure immediately so a partial mosaic never ships
                # silently. ``XRSPATIAL_GEOTIFF_STRICT=1`` forces the same
                # raise module-wide regardless of the kwarg (#1662).
                #
                # The ``missing_sources='warn'`` opt-in keeps the legacy
                # lenient path: emit ``GeoTIFFFallbackWarning`` once per
                # unreadable source and record the skip on ``vrt.holes``,
                # which the public ``read_vrt`` surfaces as
                # ``attrs['vrt_holes']`` on the returned DataArray. That
                # path is for callers that explicitly want a holey mosaic
                # plus a machine-readable list of which sources failed --
                # zero-filled holes in an integer VRT are otherwise
                # indistinguishable from real data without a nodata
                # sentinel, which is why the default was flipped to raise
                # in #1843. See also issues #1734 and #1843.
                import warnings

                from . import GeoTIFFFallbackWarning, _geotiff_strict_mode
                if missing_sources == 'raise' or _geotiff_strict_mode():
                    raise
                warnings.warn(
                    f"VRT source {src.filename!r} could not be read "
                    f"({type(e).__name__}: {e}); skipping. The output "
                    f"mosaic will have a hole at this tile. Inspect "
                    f"``DataArray.attrs['vrt_holes']`` or set "
                    f"missing_sources='raise' or "
                    f"XRSPATIAL_GEOTIFF_STRICT=1 to raise instead.",
                    GeoTIFFFallbackWarning,
                    stacklevel=2,
                )
                vrt.holes.append({
                    'source': src.filename,
                    'band': src.band,
                    'dst_rect': (dr.x_off, dr.y_off,
                                 dr.x_size, dr.y_size),
                    'error': f"{type(e).__name__}: {e}",
                })
                continue  # skip missing/unreadable sources

            # Handle source nodata.  Cast the sentinel to the *source*
            # dtype so the equality test round-trips exactly: a float64
            # source with a fractional nodata (e.g. -9999.25) would
            # previously miss the mask because ``np.float32(-9999.25)``
            # rounds to the nearest float32 and then compares unequal
            # to the float64 pixel value.  Use an explicit ``is not None``
            # check so a legitimate ``<NODATA>0</NODATA>`` survives the
            # fallback: the earlier ``src.nodata or nodata`` shortcut treated
            # ``0.0`` as falsy and silently replaced it with the band-level
            # sentinel (issue #1655).
            #
            # Apply nodata masking *before* the resample.  Nearest-neighbour
            # carries the sentinel value through unchanged, which is what
            # we want -- the resampled pixel inherits the NaN (or integer
            # sentinel) of whichever source pixel it sampled.  Resampling
            # first and masking after would still work for the float case
            # but would force the integer-into-float promotion to allocate
            # the resampled-shape buffer before discovering the mask,
            # which is a waste when the source rect is much larger than
            # the destination rect.
            # ``mask_nodata=False`` skips every inline sentinel rewrite
            # below so the public backend's opt-out is honored
            # symmetrically across float and integer source dtypes.
            # Without this gate the float branch at the ``kind == 'f'``
            # arm masked unconditionally, and the integer-feeding-float
            # branch below promoted to NaN unconditionally, so a caller
            # passing ``mask_nodata=False`` to ``read_vrt`` still lost
            # the literal sentinel pixels. See issue #2158.
            src_nodata = src.nodata if src.nodata is not None else nodata
            if mask_nodata and src_nodata is not None and src_arr.dtype.kind == 'f':
                src_arr = src_arr.copy()
                sentinel = src_arr.dtype.type(src_nodata)
                src_arr[src_arr == sentinel] = np.nan
            elif (mask_nodata
                    and src_nodata is not None
                    and src_arr.dtype.kind in ('u', 'i')
                    and result.dtype.kind == 'f'):
                # Integer source feeding a float-dataType VRT.  Without
                # this branch the source's sentinel value (e.g. 65535
                # for uint16) flows through the int->float cast at the
                # ``result[...] = src_arr[...]`` placement below as a
                # literal float value, so downstream NaN-aware code
                # sees the sentinel as valid data.  Gate the cast on
                # the sentinel being representable in the source dtype
                # so out-of-range sentinels (e.g. uint16 file paired
                # with GDAL_NODATA="-9999") stay no-op rather than
                # tripping OverflowError on ``dtype.type(int(...))``.
                # See issue #1616.
                # Accept an int sentinel directly so 64-bit values
                # (``2**64 - 1`` for UInt64, ``INT64_MIN`` for Int64)
                # don't get clobbered by ``float(...).is_integer()``
                # losing precision near the edges.  Fall back to the
                # float parse for legacy float-typed nodata.
                nodata_int: int | None = None
                if (isinstance(src_nodata, (int, np.integer))
                        and not isinstance(src_nodata, bool)):
                    nodata_int = int(src_nodata)
                else:
                    try:
                        nodata_f = float(src_nodata)
                    except (TypeError, ValueError):
                        nodata_f = None
                    if (nodata_f is not None
                            and np.isfinite(nodata_f)
                            and nodata_f.is_integer()):
                        nodata_int = int(nodata_f)
                if nodata_int is not None:
                    info = np.iinfo(src_arr.dtype)
                    if info.min <= nodata_int <= info.max:
                        sentinel = src_arr.dtype.type(nodata_int)
                        mask = src_arr == sentinel
                        if mask.any():
                            src_arr = src_arr.astype(result.dtype)
                            src_arr[mask] = result.dtype.type('nan')

            # Apply ComplexSource scaling
            if src.scale is not None and src.scale != 1.0:
                src_arr = src_arr.astype(np.float64) * src.scale
            if src.offset is not None and src.offset != 0.0:
                src_arr = src_arr.astype(np.float64) + src.offset

            if needs_resample:
                # Refuse VRTs that request a non-nearest resample
                # algorithm.  ``_resample_nearest_window`` below is the
                # only implemented kernel; silently substituting it for
                # ``Bilinear`` / ``Cubic`` / ``Average`` / ``Mode``
                # would return wrong numbers under the user's
                # configured algorithm name.  See issue #1751.
                _check_resample_alg_supported(src.resample_alg, src.filename)
                # Resample only the destination sub-window the caller
                # asked for.  ``_resample_nearest_window`` walks each
                # destination pixel in ``[sub_*0, sub_*1)`` through the
                # same ``floor((d + 0.5) * src / dst)`` mapping as
                # ``_resample_nearest`` would use on the full rect, then
                # subtracts the sub-source origin to index into
                # ``src_arr`` (which is the minimal sub-rect read above).
                # The output is byte-identical to resampling the full
                # rect and slicing ``[sub_r0:sub_r1, sub_c0:sub_c1]``
                # afterwards. See issue #1704.
                src_arr = _resample_nearest_window(
                    src_arr,
                    src_row_min, src_col_min,
                    sr.y_size, sr.x_size,
                    dr.y_size, dr.x_size,
                    sub_r0, sub_c0, sub_r1, sub_c1,
                )

            # Place into output
            out_r0 = clip_r0 - r0
            out_c0 = clip_c0 - c0
            out_r1 = out_r0 + src_arr.shape[0]
            out_c1 = out_c0 + src_arr.shape[1]

            # Handle size mismatch from rounding
            actual_h = min(src_arr.shape[0], out_r1 - out_r0)
            actual_w = min(src_arr.shape[1], out_c1 - out_c0)

            if len(selected_bands) == 1:
                result[out_r0:out_r0 + actual_h,
                       out_c0:out_c0 + actual_w] = src_arr[:actual_h, :actual_w]
            else:
                result[out_r0:out_r0 + actual_h,
                       out_c0:out_c0 + actual_w,
                       band_idx] = src_arr[:actual_h, :actual_w]

    return result, vrt


# ---------------------------------------------------------------------------
# VRT writer
# ---------------------------------------------------------------------------

_NP_TO_VRT_DTYPE = {v: k for k, v in _DTYPE_MAP.items()}


def _vrt_dtype_name_for(bps, sample_format):
    """Map TIFF ``BitsPerSample`` + ``SampleFormat`` to a GDAL VRT dtype.

    Routes through ``_dtypes.tiff_dtype_to_numpy`` so the VRT header
    matches what the reader actually decodes (including ``bps=12, sf=1``
    -> ``uint16``). Raises ``ValueError`` when the resolved numpy dtype
    has no GDAL VRT name rather than silently falling back to ``Byte``.

    Parameters
    ----------
    bps : int
        Raw ``BitsPerSample`` (already resolved to a scalar; pass through
        ``resolve_bits_per_sample`` first if you have a sequence).
    sample_format : int or sequence of int
        Raw ``SampleFormat`` tag value. Sequences are normalised through
        ``resolve_sample_format``.

    Returns
    -------
    str
        A GDAL ``dataType`` name (``Byte``, ``UInt16``, ``Float32`` ...).
    """
    from ._dtypes import resolve_sample_format, tiff_dtype_to_numpy

    sf = resolve_sample_format(sample_format)
    np_dtype = tiff_dtype_to_numpy(bps, sf)
    try:
        return _NP_TO_VRT_DTYPE[np_dtype.type]
    except KeyError:
        raise ValueError(
            f"Cannot map numpy dtype {np_dtype} (from bps={bps}, "
            f"sample_format={sf}) to a GDAL VRT dataType. Supported "
            f"VRT dtypes are: {sorted(_NP_TO_VRT_DTYPE.values())}."
        )


def _check_no_rotated_source_transforms(sources_meta: list[dict]) -> None:
    """Reject VRT writer sources whose transform has non-zero skew (#2349).

    The writer emits an axis-aligned mosaic GeoTransform with ``0.0``
    skew slots. A source whose own GeoTIFF declared a rotated /
    sheared affine cannot be placed correctly on that mosaic; the
    on-disk VRT would point at the source with the rotation silently
    dropped, mis-locating the tile. Refuse rather than write a
    mis-aligned mosaic.

    The rotated 6-tuple is stamped on ``GeoTransform.rotated_affine``
    by the reader when ``allow_rotated=True`` -- see issue #2129. A
    plain axis-aligned source has ``rotated_affine=None`` and skips
    the check.
    """
    for m in sources_meta:
        ra = getattr(m['transform'], 'rotated_affine', None)
        if ra is None:
            continue
        # ``rotated_affine`` is the rasterio 6-tuple
        # ``(pixel_width, b, origin_x, d, pixel_height, origin_y)``;
        # positions 1 and 3 are the skew terms.
        try:
            b = float(ra[1])
            d = float(ra[3])
        except (IndexError, TypeError, ValueError):
            b = d = 0.0
        if b != 0.0 or d != 0.0:
            raise UnsupportedGeoTIFFFeatureError(
                f"VRT source {m['path']!r} has a rotated / sheared "
                f"affine transform (b={b!r}, d={d!r}); rotated source "
                f"transforms are not a supported feature in this "
                f"release. write_vrt emits an axis-aligned mosaic and "
                f"would silently drop the skew terms, mis-placing the "
                f"source on the virtual raster. Reproject the source "
                f"to an axis-aligned grid before adding it to the "
                f"mosaic. See `xrspatial.geotiff.SUPPORTED_FEATURES` "
                f"for the release tier map (epic #2340)."
            )


def _check_no_mixed_raster_type(sources_meta: list[dict]) -> None:
    """Reject VRT writer sources whose AREA_OR_POINT disagrees (#2349).

    The mosaic VRT emits a single dataset-level ``AREA_OR_POINT``
    value. Silent flattening to ``first['raster_type']`` would shift
    the disagreeing source by half a pixel on read, because
    PixelIsArea origin is the pixel's edge and PixelIsPoint origin is
    the pixel's centre.
    """
    if not sources_meta:
        return
    first = sources_meta[0]
    first_raster_type = first.get('raster_type')
    for m in sources_meta[1:]:
        m_raster_type = m.get('raster_type')
        if (m_raster_type is not None and first_raster_type is not None
                and m_raster_type != first_raster_type):
            raise UnsupportedGeoTIFFFeatureError(
                f"VRT source {m['path']!r} declares raster_type="
                f"{m_raster_type!r} which does not match the first "
                f"source ({first_raster_type!r}). Silent flattening of "
                f"mixed AREA_OR_POINT registration across stacked "
                f"sources is not supported; the mosaic would emit a "
                f"single dataset-level value and shift the disagreeing "
                f"source by half a pixel. Re-tag the sources so they "
                f"share the same pixel registration before mosaicing. "
                f"See `xrspatial.geotiff.SUPPORTED_FEATURES` "
                f"(epic #2340)."
            )


def _nodata_values_agree(a, b) -> bool:
    """Return True iff two nodata sentinels are the same value.

    Float NaN must be compared via :func:`math.isnan` because
    ``float('nan') != float('nan')`` is ``True`` and would otherwise
    flag two sources that both carry the standard float NaN sentinel
    as a mismatch. ``None`` (no sentinel declared) is symmetric: two
    Nones agree. Integer / float cross-type equality
    (``-9999 == -9999.0``) is fine as-is.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        a_is_nan = isinstance(a, float) and math.isnan(a)
        b_is_nan = isinstance(b, float) and math.isnan(b)
    except (TypeError, ValueError):
        a_is_nan = b_is_nan = False
    if a_is_nan and b_is_nan:
        return True
    if a_is_nan or b_is_nan:
        return False
    return a == b


def _check_no_mixed_nodata(sources_meta: list[dict], *,
                           caller_nodata) -> None:
    """Reject VRT writer sources whose nodata sentinels disagree (#2349).

    The legacy writer silently picked ``first['nodata']`` for every
    band of every source, so a tile whose own sentinel was different
    had its "valid" pixels masked by the mosaic-level
    ``<NoDataValue>`` on read. Fail closed unless the caller pinned
    the mosaic nodata explicitly via the ``nodata`` kwarg. Mirrors
    the read-side ``MixedBandMetadataError`` opt-out pattern.
    """
    if caller_nodata is not None or not sources_meta:
        return
    first = sources_meta[0]
    first_nodata = first.get('nodata')
    for m in sources_meta[1:]:
        m_nodata = m.get('nodata')
        if not _nodata_values_agree(m_nodata, first_nodata):
            raise UnsupportedGeoTIFFFeatureError(
                f"VRT source {m['path']!r} declares nodata="
                f"{m_nodata!r} which does not match the first source "
                f"({first_nodata!r}). Silent flattening of mixed "
                f"per-source nodata sentinels across stacked sources "
                f"is not supported; the writer would emit the first "
                f"source's value as a single dataset-level "
                f"``<NoDataValue>`` and the disagreeing source's "
                f"sentinel would become a valid-looking pixel on "
                f"read. Either pin the mosaic nodata explicitly with "
                f"``write_vrt(..., nodata=<value>)`` or re-tag the "
                f"sources so they agree. See "
                f"`xrspatial.geotiff.SUPPORTED_FEATURES` (epic #2340)."
            )


def write_vrt(vrt_path: str, source_files: list[str], *,
              relative: bool = True,
              crs_wkt: str | None = None,
              nodata: float | int | None = None) -> str:
    """Generate a VRT file that mosaics multiple GeoTIFF tiles.

    Do not call this symbol directly from external code. Release-contract
    tier (epic #2340): this is the [internal-only] VRT XML emitter.
    The public surface lives in ``xrspatial.geotiff.write_vrt``
    (re-exported from ``_writers/vrt.py``) and carries the [advanced]
    tier; this function is what that public wrapper calls into. See
    ``docs/source/reference/release_gate_geotiff.rst`` and
    ``docs/source/reference/geotiff_release_contract.rst`` for the
    contract.

    Each source file is placed in the virtual raster based on its
    geo transform. All sources must share the same pixel size, dtype
    (sample format + bits-per-sample), band count, and CRS. Mismatches
    raise ``ValueError`` rather than producing a misplaced or mis-typed
    mosaic.

    Parameters
    ----------
    vrt_path : str
        [internal-only] Output .vrt file path.
    source_files : list of str
        [internal-only] Paths to the source GeoTIFF files.
    relative : bool
        [internal-only] Store source paths relative to the VRT file.
    crs_wkt : str or None
        [internal-only] CRS as WKT string. If None, taken from the
        first source.
    nodata : float, int, or None
        [internal-only] NoData value applied to every band of the
        mosaic. Caller-supplied value takes precedence; when ``None``,
        the first source's per-band nodata is used. Integer sentinels
        (e.g. ``65535`` for uint16, ``-9999`` for int32) are accepted
        so the surface lines up with the ``nodata`` kwarg on
        ``to_geotiff`` and ``write_geotiff_gpu``.

    Returns
    -------
    str
        Path to the written VRT file.
    """
    from ._dtypes import resolve_bits_per_sample
    from ._geotags import extract_geo_info
    from ._header import parse_all_ifds, parse_header
    from ._reader import _FileSource

    # Defense-in-depth: the public ``write_vrt`` wrapper in
    # ``_writers/vrt.py`` already rejects bool / non-numeric ``nodata`` via
    # ``_validate_nodata_arg`` so callers see the parity error from the
    # entry point. Repeat the check here so direct callers of the
    # internal symbol (and any future refactor that splits the public
    # wrapper) cannot regress to ``<NoDataValue>True</NoDataValue>``. See
    # issue #1921.
    if isinstance(nodata, (bool, np.bool_)):
        raise TypeError(
            f"nodata must be numeric (int or float), got {nodata!r}")

    if not source_files:
        raise ValueError("source_files must not be empty")

    # Read metadata from all sources
    sources_meta = []
    for src_path in source_files:
        src = _FileSource(src_path)
        data = src.read_all()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        geo = extract_geo_info(ifd, data, header.byte_order)
        src.close()

        bps = resolve_bits_per_sample(ifd.bits_per_sample)

        sources_meta.append({
            'path': src_path,
            'width': ifd.width,
            'height': ifd.height,
            'bands': ifd.samples_per_pixel,
            'dtype': np.dtype(_DTYPE_MAP.get(
                {v: k for k, v in _DTYPE_MAP.items()}.get(
                    np.dtype(
                        f'{"f" if ifd.sample_format == 3 else ("i" if ifd.sample_format == 2 else "u")}'  # noqa: E501
                        f'{bps // 8}'
                    ).type,
                    'Float32'),
                np.float32)),
            'bps': bps,
            'sample_format': ifd.sample_format,
            'transform': geo.transform,
            'crs_wkt': geo.crs_wkt,
            'nodata': geo.nodata,
            'raster_type': geo.raster_type,
        })

    first = sources_meta[0]
    res_x = first['transform'].pixel_width
    res_y = first['transform'].pixel_height

    # Enforce the docstring contract: every source must agree with the
    # first on pixel size, sample format + bits-per-sample (i.e. dtype),
    # band count, and CRS WKT. Without this, build_vrt would silently
    # produce a syntactically valid VRT that misplaces or mis-types data
    # downstream (issue #1733).
    #
    # Pixel size is compared with a small relative tolerance: TIFFs
    # written by different tools occasionally round the GeoTransform
    # slightly, and the existing mosaic-extent math rounds the final
    # raster size anyway. Sample format + bps must match exactly because
    # the VRT dtype is taken from ``first`` and applied to every band.
    _PIXEL_SIZE_RTOL = 1e-6

    def _pixel_size_mismatch(a: float, b: float) -> bool:
        # Use a relative tolerance on the magnitude rather than a direct
        # ``a == b`` check so that minor float rounding between tools is
        # ignored. ``pixel_height`` is negative for the common north-up
        # case; a flipped sign would change the magnitude ratio above
        # the tolerance and so is correctly treated as a mismatch (which
        # is the desired behavior, since opposite-sign sources do not
        # stack into a valid VRT).
        if a == b:
            return False
        denom = max(abs(a), abs(b))
        if denom == 0.0:
            return abs(a - b) > 0.0
        return abs(a - b) / denom > _PIXEL_SIZE_RTOL

    # Group the new "fail-loudly on unsupported feature combinations"
    # gates (epic #2340 / PR 5) in one block so the cross-source policy
    # is easy to read end-to-end:
    #
    #   1. Reject any source whose declared transform carries a
    #      non-zero skew term (rotated / sheared affine). write_vrt
    #      emits an axis-aligned GeoTransform, so a rotated source
    #      would be silently mis-placed on the mosaic.
    #   2. Reject mixed per-source AREA_OR_POINT registration. The
    #      mosaic writes a single dataset-level value; flattening to
    #      ``first`` would shift the disagreeing source by half a
    #      pixel on read.
    #   3. Reject mixed per-source nodata sentinels unless the caller
    #      pinned the mosaic value via the ``nodata`` kwarg. Mirrors
    #      the read-side ``MixedBandMetadataError`` opt-out pattern.
    _check_no_rotated_source_transforms(sources_meta)
    _check_no_mixed_raster_type(sources_meta)
    _check_no_mixed_nodata(sources_meta, caller_nodata=nodata)

    # Pixel size, sample format, band count, and CRS share the
    # documented "all sources must agree with first" contract (#1733).
    # These remain inline here because they predate #2349 and have
    # their own match-pattern tests.
    first_crs = first.get('crs_wkt')
    for m in sources_meta[1:]:
        t = m['transform']
        if _pixel_size_mismatch(t.pixel_width, res_x) \
                or _pixel_size_mismatch(t.pixel_height, res_y):
            raise ValueError(
                f"VRT source {m['path']!r} has pixel size "
                f"({t.pixel_width}, {t.pixel_height}) which does not "
                f"match the first source ({res_x}, {res_y}). All sources "
                f"in a VRT must share the same pixel size."
            )
        if m['sample_format'] != first['sample_format'] \
                or m['bps'] != first['bps']:
            raise ValueError(
                f"VRT source {m['path']!r} has sample_format="
                f"{m['sample_format']}, bps={m['bps']} which does not "
                f"match the first source (sample_format="
                f"{first['sample_format']}, bps={first['bps']}). All "
                f"sources in a VRT must share the same dtype."
            )
        if m['bands'] != first['bands']:
            raise ValueError(
                f"VRT source {m['path']!r} has {m['bands']} band(s) "
                f"which does not match the first source "
                f"({first['bands']} band(s)). All sources in a VRT "
                f"must share the same band count."
            )
        m_crs = m.get('crs_wkt')
        # Treat asymmetric CRS (one set, one missing/empty) as a
        # mismatch too. If we only flagged when both were set, a source
        # that lost its CRS during a re-write would silently inherit the
        # first source's CRS in the VRT header and could end up tagged
        # with the wrong projection.
        if (first_crs or m_crs) and m_crs != first_crs:
            raise ValueError(
                f"VRT source {m['path']!r} has CRS WKT that does not "
                f"match the first source. All sources in a VRT must "
                f"share the same CRS."
            )

    # Compute the bounding box of all sources
    all_x0, all_y0, all_x1, all_y1 = [], [], [], []
    for m in sources_meta:
        t = m['transform']
        x0 = t.origin_x
        y0 = t.origin_y
        x1 = x0 + m['width'] * t.pixel_width
        y1 = y0 + m['height'] * t.pixel_height
        all_x0.append(min(x0, x1))
        all_y0.append(min(y0, y1))
        all_x1.append(max(x0, x1))
        all_y1.append(max(y0, y1))

    mosaic_x0 = min(all_x0)
    mosaic_y_top = max(all_y1)  # top edge (y increases upward in geo)
    mosaic_x1 = max(all_x1)
    mosaic_y_bottom = min(all_y0)

    total_w = int(round((mosaic_x1 - mosaic_x0) / abs(res_x)))
    total_h = int(round((mosaic_y_top - mosaic_y_bottom) / abs(res_y)))

    # Determine VRT dtype via the central TIFF-to-numpy resolver so the
    # VRT header agrees with what the reader will actually decode. The
    # previous local if/elif/else ladder had no entry for sub-byte or
    # 12-bit unsigned samples (reader promotes ``bps=12, sf=1`` to
    # ``uint16``), so a VRT over a valid 12-bit source got tagged
    # ``Byte`` and could be truncated by downstream GDAL readers. Issue
    # #1914.
    vrt_dtype_name = _vrt_dtype_name_for(first['bps'], first['sample_format'])

    srs = crs_wkt or first.get('crs_wkt') or ''
    nd = nodata if nodata is not None else first.get('nodata')

    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    n_bands = first['bands']

    # Build XML.  Every interpolated text value is run through _xml_text
    # (or _xml_attr for attribute slots) before concatenation so that a
    # caller-supplied CRS WKT or a source filename containing XML
    # special characters (``< > & " '``) cannot break the document or
    # inject extra elements.  Numeric fields (offsets, sizes, pixel
    # scales) are emitted from int / float literals and need no
    # escaping.  See issue #1607.
    lines = [f'<VRTDataset rasterXSize="{int(total_w)}" rasterYSize="{int(total_h)}">']
    if srs:
        lines.append(f'  <SRS>{_xml_text(srs)}</SRS>')
    lines.append(f'  <GeoTransform>{mosaic_x0}, {res_x}, 0.0, '
                 f'{mosaic_y_top}, 0.0, {res_y}</GeoTransform>')

    for band_num in range(1, n_bands + 1):
        lines.append(
            f'  <VRTRasterBand dataType={_xml_attr(vrt_dtype_name)} '
            f'band="{int(band_num)}">')
        if nd is not None:
            lines.append(f'    <NoDataValue>{_xml_text(nd)}</NoDataValue>')

        for m in sources_meta:
            t = m['transform']
            # Top edge of this source in geo coords -- origin_y is the
            # *top* only for north-up rasters (pixel_height < 0).  Sources
            # with ascending y (pixel_height > 0) place origin_y at the
            # bottom, so the top is origin_y + height * pixel_height.
            src_top = max(
                t.origin_y,
                t.origin_y + m['height'] * t.pixel_height,
            )
            src_left = min(
                t.origin_x,
                t.origin_x + m['width'] * t.pixel_width,
            )
            # Pixel offset in the virtual raster
            dst_x_off = int(round((src_left - mosaic_x0) / abs(res_x)))
            dst_y_off = int(round((mosaic_y_top - src_top) / abs(res_y)))

            fname = m['path']
            rel_attr = '0'
            if relative:
                try:
                    fname = os.path.relpath(fname, vrt_dir)
                    # VRT XML uses forward slashes regardless of platform
                    fname = fname.replace('\\', '/')
                    rel_attr = '1'
                except ValueError:
                    pass  # different drives on Windows

            lines.append('    <SimpleSource>')
            lines.append(
                f'      <SourceFilename relativeToVRT="{rel_attr}">'
                f'{_xml_text(fname)}</SourceFilename>')
            lines.append(f'      <SourceBand>{int(band_num)}</SourceBand>')
            lines.append(
                f'      <SrcRect xOff="0" yOff="0" '
                f'xSize="{int(m["width"])}" ySize="{int(m["height"])}"/>')
            lines.append(
                f'      <DstRect xOff="{int(dst_x_off)}" '
                f'yOff="{int(dst_y_off)}" '
                f'xSize="{int(m["width"])}" ySize="{int(m["height"])}"/>')
            lines.append('    </SimpleSource>')

        lines.append('  </VRTRasterBand>')

    lines.append('</VRTDataset>')

    xml = '\n'.join(lines) + '\n'
    with open(vrt_path, 'w') as f:
        f.write(xml)

    return vrt_path
