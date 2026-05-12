"""Virtual Raster Table (VRT) reader.

Parses GDAL VRT XML files and assembles a virtual raster from one or
more source GeoTIFF files using windowed reads.
"""
from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass, field
from xml.sax.saxutils import escape as _xml_escape, quoteattr as _xml_quoteattr

import numpy as np

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

# Lazy imports to avoid circular dependency
_DTYPE_MAP = {
    'Byte': np.uint8,
    'UInt16': np.uint16,
    'Int16': np.int16,
    'UInt32': np.uint32,
    'Int32': np.int32,
    'Float32': np.float32,
    'Float64': np.float64,
    'Int8': np.int8,
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
    nodata: float | None = None
    # ComplexSource extras
    scale: float | None = None
    offset: float | None = None


@dataclass
class _VRTBand:
    """A single band in a VRT dataset."""
    band_num: int  # 1-based
    dtype: np.dtype
    nodata: float | None = None
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
        dtype_name = band_elem.get('dataType', 'Float32')
        dtype = np.dtype(_DTYPE_MAP.get(dtype_name, np.float32))
        nodata_str = _text(band_elem, 'NoDataValue')
        nodata = float(nodata_str) if nodata_str else None
        color_interp = _text(band_elem, 'ColorInterp')

        sources = []
        for src_elem in band_elem:
            tag = src_elem.tag
            if tag not in ('SimpleSource', 'ComplexSource'):
                continue

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

            src_nodata_str = _text(src_elem, 'NODATA')
            src_nodata = float(src_nodata_str) if src_nodata_str else None

            # ComplexSource extras
            scale = None
            offset = None
            if tag == 'ComplexSource':
                scale_str = _text(src_elem, 'ScaleOffset')
                offset_str = _text(src_elem, 'ScaleRatio')
                # Note: GDAL uses ScaleOffset=offset, ScaleRatio=scale
                if offset_str:
                    scale = float(offset_str)
                if scale_str:
                    offset = float(scale_str)

            sources.append(_Source(
                filename=filename,
                band=src_band,
                src_rect=src_rect,
                dst_rect=dst_rect,
                nodata=src_nodata,
                scale=scale,
                offset=offset,
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


def read_vrt(vrt_path: str, *, window=None,
             band: int | None = None,
             max_pixels: int | None = None) -> tuple[np.ndarray, VRTDataset]:
    """Read a VRT file by assembling pixel data from its source files.

    Parameters
    ----------
    vrt_path : str
        Path to the .vrt file.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) for windowed read.
    band : int or None
        Band index (0-based). None returns all bands.

    Returns
    -------
    (np.ndarray, VRTDataset) tuple
    """
    from ._reader import read_to_array

    with open(vrt_path, 'r') as f:
        xml_str = f.read()

    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    vrt = parse_vrt(xml_str, vrt_dir)

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

    if window is not None:
        r0, c0, r1, c1 = window
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(vrt.height, r1)
        c1 = min(vrt.width, c1)
    else:
        r0, c0, r1, c1 = 0, 0, vrt.height, vrt.width

    out_h = r1 - r0
    out_w = c1 - c0

    from ._reader import _check_dimensions, MAX_PIXELS_DEFAULT
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
    #   values to ``float64`` before placement (see L562-565); the destination
    #   has to be float-typed too, otherwise the fractional part is lost.
    #
    # ``np.result_type`` produces the narrowest dtype that holds every
    # contributing dtype, so an all-integer VRT stays integer and only mixes
    # widen to float64. See issue #1696.
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
    dtype = np.result_type(*effective_dtypes)
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

            # Map back to source coordinates
            # Scale factor: source pixels per destination pixel
            scale_y = sr.y_size / dr.y_size if dr.y_size > 0 else 1.0
            scale_x = sr.x_size / dr.x_size if dr.x_size > 0 else 1.0

            src_r0 = sr.y_off + int((clip_r0 - dst_r0) * scale_y)
            src_c0 = sr.x_off + int((clip_c0 - dst_c0) * scale_x)
            src_r1 = sr.y_off + int((clip_r1 - dst_r0) * scale_y)
            src_c1 = sr.x_off + int((clip_c1 - dst_c0) * scale_x)

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
                    window=(src_r0, src_c0, src_r1, src_c1),
                    band=src.band - 1,  # convert 1-based to 0-based
                )
            except (
                OSError, ValueError, struct.error,
            ) + _CODEC_DECODE_EXCEPTIONS as e:
                # Under XRSPATIAL_GEOTIFF_STRICT=1, surface the read failure
                # so partial mosaics are caught in CI. Default mode warns
                # once per missing source then continues, preserving the
                # historical behaviour. See issue #1662.
                import warnings
                from . import _geotiff_strict_mode, GeoTIFFFallbackWarning
                if _geotiff_strict_mode():
                    raise
                warnings.warn(
                    f"VRT source {src.filename!r} could not be read "
                    f"({type(e).__name__}: {e}); skipping. The output "
                    f"mosaic will have a hole at this tile.",
                    GeoTIFFFallbackWarning,
                    stacklevel=2,
                )
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
            src_nodata = src.nodata if src.nodata is not None else nodata
            if src_nodata is not None and src_arr.dtype.kind == 'f':
                src_arr = src_arr.copy()
                sentinel = src_arr.dtype.type(src_nodata)
                src_arr[src_arr == sentinel] = np.nan
            elif (src_nodata is not None
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
                try:
                    nodata_f = float(src_nodata)
                except (TypeError, ValueError):
                    nodata_f = None
                if (nodata_f is not None
                        and np.isfinite(nodata_f)
                        and nodata_f.is_integer()):
                    info = np.iinfo(src_arr.dtype)
                    nodata_int = int(nodata_f)
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


def write_vrt(vrt_path: str, source_files: list[str], *,
              relative: bool = True,
              crs_wkt: str | None = None,
              nodata: float | int | None = None) -> str:
    """Generate a VRT file that mosaics multiple GeoTIFF tiles.

    Each source file is placed in the virtual raster based on its
    geo transform. Files must share the same CRS and pixel size.

    Parameters
    ----------
    vrt_path : str
        Output .vrt file path.
    source_files : list of str
        Paths to the source GeoTIFF files.
    relative : bool
        Store source paths relative to the VRT file.
    crs_wkt : str or None
        CRS as WKT string. If None, taken from the first source.
    nodata : float, int, or None
        NoData value. If None, taken from the first source. Integer
        sentinels (e.g. ``65535`` for uint16, ``-9999`` for int32) are
        accepted so the surface lines up with the ``nodata`` kwarg on
        ``to_geotiff`` and ``write_geotiff_gpu``.

    Returns
    -------
    str
        Path to the written VRT file.
    """
    from ._reader import read_to_array
    from ._header import parse_header, parse_all_ifds
    from ._geotags import extract_geo_info
    from ._reader import _FileSource
    from ._dtypes import resolve_bits_per_sample

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
                    np.dtype(f'{"f" if ifd.sample_format == 3 else ("i" if ifd.sample_format == 2 else "u")}{bps // 8}').type,
                    'Float32'),
                np.float32)),
            'bps': bps,
            'sample_format': ifd.sample_format,
            'transform': geo.transform,
            'crs_wkt': geo.crs_wkt,
            'nodata': geo.nodata,
        })

    first = sources_meta[0]
    res_x = first['transform'].pixel_width
    res_y = first['transform'].pixel_height

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

    # Determine VRT dtype
    sf = first['sample_format']
    bps = first['bps']
    if sf == 3:
        vrt_dtype_name = 'Float64' if bps == 64 else 'Float32'
    elif sf == 2:
        vrt_dtype_name = {8: 'Int8', 16: 'Int16', 32: 'Int32'}.get(bps, 'Int32')
    else:
        vrt_dtype_name = {8: 'Byte', 16: 'UInt16', 32: 'UInt32'}.get(bps, 'Byte')

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
            # Pixel offset in the virtual raster
            dst_x_off = int(round((t.origin_x - mosaic_x0) / abs(res_x)))
            dst_y_off = int(round((mosaic_y_top - t.origin_y) / abs(res_y)))

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
