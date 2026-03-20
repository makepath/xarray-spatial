"""TIFF/COG reader: tile/strip assembly, windowed reads, HTTP range requests."""
from __future__ import annotations

import math
import mmap
import threading
import urllib.request

import numpy as np

from ._compression import (
    COMPRESSION_NONE,
    decompress,
    fp_predictor_decode,
    predictor_decode,
    unpack_bits,
)
from ._dtypes import SUB_BYTE_BPS, tiff_dtype_to_numpy
from ._geotags import GeoInfo, GeoTransform, extract_geo_info
from ._header import IFD, TIFFHeader, parse_all_ifds, parse_header


# ---------------------------------------------------------------------------
# Data source abstraction
# ---------------------------------------------------------------------------

class _MmapCache:
    """Thread-safe, reference-counted mmap cache.

    Multiple threads reading the same file share a single read-only mmap.
    The mmap is closed when the last reference is released.
    mmap slicing on a read-only mapping is thread-safe (no seek involved).
    """

    def __init__(self):
        self._lock = threading.Lock()
        # path -> (fh, mm, refcount)
        self._entries: dict[str, tuple] = {}

    def acquire(self, path: str):
        """Get or create a read-only mmap for *path*. Returns (mm, size)."""
        import os
        real = os.path.realpath(path)
        with self._lock:
            if real in self._entries:
                fh, mm, size, rc = self._entries[real]
                self._entries[real] = (fh, mm, size, rc + 1)
                return mm, size

            fh = open(real, 'rb')
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(0)
            if size > 0:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                mm = None
            self._entries[real] = (fh, mm, size, 1)
            return mm, size

    def release(self, path: str):
        """Decrement the reference count; close the mmap when it hits zero."""
        import os
        real = os.path.realpath(path)
        with self._lock:
            entry = self._entries.get(real)
            if entry is None:
                return
            fh, mm, size, rc = entry
            rc -= 1
            if rc <= 0:
                del self._entries[real]
                if mm is not None:
                    mm.close()
                fh.close()
            else:
                self._entries[real] = (fh, mm, size, rc)


# Module-level cache shared across all reads
_mmap_cache = _MmapCache()


class _FileSource:
    """Local file data source using a shared, thread-safe mmap cache."""

    def __init__(self, path: str):
        self._path = path
        self._mm, self._size = _mmap_cache.acquire(path)

    def read_range(self, start: int, length: int) -> bytes:
        if self._mm is not None:
            return self._mm[start:start + length]
        return b''

    def read_all(self):
        """Return mmap object (supports slicing, struct.unpack_from, len)."""
        if self._mm is not None:
            return self._mm
        return b''

    @property
    def size(self) -> int:
        return self._size

    def close(self):
        _mmap_cache.release(self._path)


def _get_http_pool():
    """Return a module-level urllib3 PoolManager, or None if unavailable."""
    global _http_pool
    if _http_pool is not None:
        return _http_pool
    try:
        import urllib3
        _http_pool = urllib3.PoolManager(
            num_pools=10,
            maxsize=10,
            retries=urllib3.Retry(total=2, backoff_factor=0.1),
        )
        return _http_pool
    except ImportError:
        return None


_http_pool = None


class _HTTPSource:
    """HTTP data source using range requests with connection reuse.

    Uses urllib3.PoolManager when available (reuses TCP connections and
    TLS sessions across range requests to the same host). Falls back to
    stdlib urllib.request if urllib3 is not installed.
    """

    def __init__(self, url: str):
        self._url = url
        self._size = None
        self._pool = _get_http_pool()

    def read_range(self, start: int, length: int) -> bytes:
        end = start + length - 1
        if self._pool is not None:
            resp = self._pool.request(
                'GET', self._url,
                headers={'Range': f'bytes={start}-{end}'},
            )
            return resp.data
        # Fallback: stdlib
        req = urllib.request.Request(
            self._url,
            headers={'Range': f'bytes={start}-{end}'},
        )
        with urllib.request.urlopen(req) as resp:
            return resp.read()

    def read_all(self) -> bytes:
        if self._pool is not None:
            resp = self._pool.request('GET', self._url)
            return resp.data
        with urllib.request.urlopen(self._url) as resp:
            return resp.read()

    @property
    def size(self) -> int | None:
        return self._size

    def close(self):
        pass


_CLOUD_SCHEMES = ('s3://', 'gs://', 'az://', 'abfs://')


def _is_fsspec_uri(path: str) -> bool:
    """Check if a path is a fsspec-compatible URI (not http/https/local)."""
    if path.startswith(('http://', 'https://')):
        return False
    return '://' in path


class _CloudSource:
    """Cloud storage data source using fsspec.

    Supports S3, GCS, Azure Blob Storage, and any other fsspec backend.
    Requires the appropriate library (s3fs, gcsfs, adlfs) to be installed.
    """

    def __init__(self, url: str, **storage_options):
        try:
            import fsspec
        except ImportError:
            raise ImportError(
                "fsspec is required to read from cloud storage. "
                "Install it with: pip install fsspec")
        self._url = url
        self._fs, self._path = fsspec.core.url_to_fs(url, **storage_options)
        self._size = self._fs.size(self._path)

    def read_range(self, start: int, length: int) -> bytes:
        with self._fs.open(self._path, 'rb') as f:
            f.seek(start)
            return f.read(length)

    def read_all(self) -> bytes:
        with self._fs.open(self._path, 'rb') as f:
            return f.read()

    @property
    def size(self) -> int:
        return self._size

    def close(self):
        pass


def _open_source(source: str):
    """Open a data source (local file, URL, or cloud path)."""
    if source.startswith(('http://', 'https://')):
        return _HTTPSource(source)
    if _is_fsspec_uri(source):
        return _CloudSource(source)
    return _FileSource(source)


def _apply_predictor(chunk: np.ndarray, pred: int, width: int,
                     height: int, bytes_per_sample: int) -> np.ndarray:
    """Apply the appropriate predictor decode to decompressed data."""
    if pred == 2:
        return predictor_decode(chunk, width, height, bytes_per_sample)
    elif pred == 3:
        return fp_predictor_decode(chunk, width, height, bytes_per_sample)
    return chunk


def _packed_byte_count(pixel_count: int, bps: int) -> int:
    """Compute the number of packed bytes for sub-byte bit depths."""
    return (pixel_count * bps + 7) // 8


def _decode_strip_or_tile(data_slice, compression, width, height, samples,
                          bps, bytes_per_sample, is_sub_byte, dtype, pred,
                          byte_order='<'):
    """Decompress, apply predictor, unpack sub-byte, and reshape a strip/tile.

    Parameters
    ----------
    byte_order : str
        '<' for little-endian, '>' for big-endian.  When the file byte
        order differs from the system's native order, pixel data is
        byte-swapped after decompression.

    Returns an array shaped (height, width) or (height, width, samples).
    """
    pixel_count = width * height * samples
    if is_sub_byte:
        expected = _packed_byte_count(pixel_count, bps)
    else:
        expected = pixel_count * bytes_per_sample

    chunk = decompress(data_slice, compression, expected,
                       width=width, height=height, samples=samples)

    if pred in (2, 3) and not is_sub_byte:
        if not chunk.flags.writeable:
            chunk = chunk.copy()
        chunk = _apply_predictor(chunk, pred, width, height,
                                 bytes_per_sample * samples)

    if is_sub_byte:
        pixels = unpack_bits(chunk, bps, pixel_count)
    else:
        # Use the file's byte order for the view, then convert to native
        file_dtype = dtype.newbyteorder(byte_order)
        pixels = chunk.view(file_dtype)
        if file_dtype.byteorder not in ('=', '|', _NATIVE_ORDER):
            pixels = pixels.astype(dtype)

    if samples > 1:
        return pixels.reshape(height, width, samples)
    return pixels.reshape(height, width)


import sys as _sys
_NATIVE_ORDER = '<' if _sys.byteorder == 'little' else '>'


# ---------------------------------------------------------------------------
# Strip reader
# ---------------------------------------------------------------------------

def _read_strips(data: bytes, ifd: IFD, header: TIFFHeader,
                 dtype: np.dtype, window=None) -> np.ndarray:
    """Read a strip-organized TIFF image.

    Parameters
    ----------
    data : bytes
        Full file data.
    ifd : IFD
        Parsed IFD for this image.
    header : TIFFHeader
        File header.
    dtype : np.dtype
        Output pixel dtype.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) or None for full image.

    Returns
    -------
    np.ndarray with shape (height, width) or windowed subset.
    """
    width = ifd.width
    height = ifd.height
    samples = ifd.samples_per_pixel
    compression = ifd.compression
    rps = ifd.rows_per_strip
    offsets = ifd.strip_offsets
    byte_counts = ifd.strip_byte_counts
    pred = ifd.predictor
    bps = ifd.bits_per_sample
    if isinstance(bps, tuple):
        bps = bps[0]
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS

    if offsets is None or byte_counts is None:
        raise ValueError("Missing strip offsets or byte counts")

    planar = ifd.planar_config  # 1=chunky (interleaved), 2=planar (separate)

    # Determine output region
    if window is not None:
        r0, c0, r1, c1 = window
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(height, r1)
        c1 = min(width, c1)
    else:
        r0, c0, r1, c1 = 0, 0, height, width

    out_h = r1 - r0
    out_w = c1 - c0

    if samples > 1:
        result = np.empty((out_h, out_w, samples), dtype=dtype)
    else:
        result = np.empty((out_h, out_w), dtype=dtype)

    if planar == 2 and samples > 1:
        strips_per_band = math.ceil(height / rps)
        first_strip = r0 // rps
        last_strip = min((r1 - 1) // rps, strips_per_band - 1)

        for band_idx in range(samples):
            band_offset = band_idx * strips_per_band
            for strip_idx in range(first_strip, last_strip + 1):
                global_idx = band_offset + strip_idx
                if global_idx >= len(offsets):
                    continue
                strip_row = strip_idx * rps
                strip_rows = min(rps, height - strip_row)
                if strip_rows <= 0:
                    continue

                strip_data = data[offsets[global_idx]:offsets[global_idx] + byte_counts[global_idx]]
                strip_pixels = _decode_strip_or_tile(
                    strip_data, compression, width, strip_rows, 1,
                    bps, bytes_per_sample, is_sub_byte, dtype, pred,
                    byte_order=header.byte_order)

                src_r0 = max(r0 - strip_row, 0)
                src_r1 = min(r1 - strip_row, strip_rows)
                dst_r0 = max(strip_row - r0, 0)
                dst_r1 = dst_r0 + (src_r1 - src_r0)
                if dst_r1 > dst_r0:
                    result[dst_r0:dst_r1, :, band_idx] = strip_pixels[src_r0:src_r1, c0:c1]
    else:
        first_strip = r0 // rps
        last_strip = min((r1 - 1) // rps, len(offsets) - 1)

        for strip_idx in range(first_strip, last_strip + 1):
            strip_row = strip_idx * rps
            strip_rows = min(rps, height - strip_row)
            if strip_rows <= 0:
                continue

            strip_data = data[offsets[strip_idx]:offsets[strip_idx] + byte_counts[strip_idx]]
            strip_pixels = _decode_strip_or_tile(
                strip_data, compression, width, strip_rows, samples,
                bps, bytes_per_sample, is_sub_byte, dtype, pred,
                byte_order=header.byte_order)

            src_r0 = max(r0 - strip_row, 0)
            src_r1 = min(r1 - strip_row, strip_rows)
            dst_r0 = max(strip_row - r0, 0)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            if dst_r1 > dst_r0:
                result[dst_r0:dst_r1] = strip_pixels[src_r0:src_r1, c0:c1]

    return result


# ---------------------------------------------------------------------------
# Tile reader
# ---------------------------------------------------------------------------

def _read_tiles(data: bytes, ifd: IFD, header: TIFFHeader,
                dtype: np.dtype, window=None) -> np.ndarray:
    """Read a tile-organized TIFF image.

    Parameters
    ----------
    data : bytes
        Full file data.
    ifd : IFD
        Parsed IFD for this image.
    header : TIFFHeader
        File header.
    dtype : np.dtype
        Output pixel dtype.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) or None for full image.

    Returns
    -------
    np.ndarray with shape (height, width) or windowed subset.
    """
    width = ifd.width
    height = ifd.height
    tw = ifd.tile_width
    th = ifd.tile_height
    samples = ifd.samples_per_pixel
    compression = ifd.compression
    pred = ifd.predictor
    bps = ifd.bits_per_sample
    if isinstance(bps, tuple):
        bps = bps[0]
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS

    offsets = ifd.tile_offsets
    byte_counts = ifd.tile_byte_counts
    if offsets is None or byte_counts is None:
        raise ValueError("Missing tile offsets or byte counts")

    planar = ifd.planar_config
    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)

    if window is not None:
        r0, c0, r1, c1 = window
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(height, r1)
        c1 = min(width, c1)
    else:
        r0, c0, r1, c1 = 0, 0, height, width

    out_h = r1 - r0
    out_w = c1 - c0

    _alloc = np.zeros if window is not None else np.empty
    if samples > 1:
        result = _alloc((out_h, out_w, samples), dtype=dtype)
    else:
        result = _alloc((out_h, out_w), dtype=dtype)

    tile_row_start = r0 // th
    tile_row_end = min(math.ceil(r1 / th), tiles_down)
    tile_col_start = c0 // tw
    tile_col_end = min(math.ceil(c1 / tw), tiles_across)

    band_count = samples if (planar == 2 and samples > 1) else 1
    tiles_per_band = tiles_across * tiles_down

    for band_idx in range(band_count):
        band_tile_offset = band_idx * tiles_per_band if band_count > 1 else 0
        tile_samples = 1 if band_count > 1 else samples

        for tr in range(tile_row_start, tile_row_end):
            for tc in range(tile_col_start, tile_col_end):
                tile_idx = band_tile_offset + tr * tiles_across + tc
                if tile_idx >= len(offsets):
                    continue

                tile_data = data[offsets[tile_idx]:offsets[tile_idx] + byte_counts[tile_idx]]
                tile_pixels = _decode_strip_or_tile(
                    tile_data, compression, tw, th, tile_samples,
                    bps, bytes_per_sample, is_sub_byte, dtype, pred,
                    byte_order=header.byte_order)

                tile_r0 = tr * th
                tile_c0 = tc * tw

                src_r0 = max(r0 - tile_r0, 0)
                src_c0 = max(c0 - tile_c0, 0)
                src_r1 = min(r1 - tile_r0, th)
                src_c1 = min(c1 - tile_c0, tw)

                dst_r0 = max(tile_r0 - r0, 0)
                dst_c0 = max(tile_c0 - c0, 0)

                actual_tile_h = min(th, height - tile_r0)
                actual_tile_w = min(tw, width - tile_c0)
                src_r1 = min(src_r1, actual_tile_h)
                src_c1 = min(src_c1, actual_tile_w)
                dst_r1 = dst_r0 + (src_r1 - src_r0)
                dst_c1 = dst_c0 + (src_c1 - src_c0)

                if dst_r1 > dst_r0 and dst_c1 > dst_c0:
                    src_slice = tile_pixels[src_r0:src_r1, src_c0:src_c1]
                    if band_count > 1:
                        result[dst_r0:dst_r1, dst_c0:dst_c1, band_idx] = src_slice
                    else:
                        result[dst_r0:dst_r1, dst_c0:dst_c1] = src_slice

    return result


# ---------------------------------------------------------------------------
# COG HTTP reader
# ---------------------------------------------------------------------------

def _read_cog_http(url: str, overview_level: int | None = None,
                   band: int | None = None) -> tuple[np.ndarray, GeoInfo]:
    """Read a COG via HTTP range requests.

    Parameters
    ----------
    url : str
        HTTP(S) URL to the COG file.
    overview_level : int or None
        Which overview to read (0 = full res, 1 = first overview, etc.).
    band : int
        Band index (0-based, for multi-band files).

    Returns
    -------
    (array, geo_info) tuple
    """
    source = _HTTPSource(url)

    # Initial fetch: get header + IFDs (COGs put metadata first)
    header_bytes = source.read_range(0, 16384)

    header = parse_header(header_bytes)
    ifds = parse_all_ifds(header_bytes, header)

    # If we didn't get all IFDs, try a larger fetch
    if len(ifds) == 0:
        header_bytes = source.read_range(0, 65536)
        ifds = parse_all_ifds(header_bytes, header)

    if len(ifds) == 0:
        raise ValueError("No IFDs found in COG")

    # Select IFD based on overview level
    ifd_idx = 0
    if overview_level is not None:
        ifd_idx = min(overview_level, len(ifds) - 1)
    ifd = ifds[ifd_idx]

    bps = ifd.bits_per_sample
    if isinstance(bps, tuple):
        bps = bps[0]
    dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
    geo_info = extract_geo_info(ifd, header_bytes, header.byte_order)

    # COGs are tiled -- fetch individual tiles
    if not ifd.is_tiled:
        # Fallback: fetch entire file
        all_data = source.read_all()
        arr = _read_strips(all_data, ifd, header, dtype)
        source.close()
        return arr, geo_info

    width = ifd.width
    height = ifd.height
    tw = ifd.tile_width
    th = ifd.tile_height
    samples = ifd.samples_per_pixel
    compression = ifd.compression
    pred = ifd.predictor
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS

    offsets = ifd.tile_offsets
    byte_counts = ifd.tile_byte_counts

    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)

    if samples > 1:
        result = np.empty((height, width, samples), dtype=dtype)
    else:
        result = np.empty((height, width), dtype=dtype)

    for tr in range(tiles_down):
        for tc in range(tiles_across):
            tile_idx = tr * tiles_across + tc
            if tile_idx >= len(offsets):
                continue

            off = offsets[tile_idx]
            bc = byte_counts[tile_idx]
            if bc == 0:
                continue

            tile_data = source.read_range(off, bc)
            tile_pixels = _decode_strip_or_tile(
                tile_data, compression, tw, th, samples,
                bps, bytes_per_sample, is_sub_byte, dtype, pred,
                byte_order=header.byte_order)

            # Place tile
            y0 = tr * th
            x0 = tc * tw
            y1 = min(y0 + th, height)
            x1 = min(x0 + tw, width)
            actual_h = y1 - y0
            actual_w = x1 - x0
            result[y0:y1, x0:x1] = tile_pixels[:actual_h, :actual_w]

    source.close()
    return result, geo_info


# ---------------------------------------------------------------------------
# Main read function
# ---------------------------------------------------------------------------

def read_to_array(source: str, *, window=None, overview_level: int | None = None,
                  band: int | None = None) -> tuple[np.ndarray, GeoInfo]:
    """Read a GeoTIFF/COG to a numpy array.

    Parameters
    ----------
    source : str
        File path or URL.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop).
    overview_level : int or None
        Overview level (0 = full res).
    band : int
        Band index for multi-band files.

    Returns
    -------
    (np.ndarray, GeoInfo) tuple
    """
    if source.startswith(('http://', 'https://')):
        return _read_cog_http(source, overview_level=overview_level, band=band)

    # Local file or cloud storage: read all bytes then parse
    if _is_fsspec_uri(source):
        src = _CloudSource(source)
    else:
        src = _FileSource(source)
    data = src.read_all()

    try:
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)

        if len(ifds) == 0:
            raise ValueError("No IFDs found in TIFF file")

        # Select IFD
        ifd_idx = 0
        if overview_level is not None:
            ifd_idx = min(overview_level, len(ifds) - 1)
        ifd = ifds[ifd_idx]

        bps = ifd.bits_per_sample
        if isinstance(bps, tuple):
            bps = bps[0]
        dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        geo_info = extract_geo_info(ifd, data, header.byte_order)

        if ifd.is_tiled:
            arr = _read_tiles(data, ifd, header, dtype, window)
        else:
            arr = _read_strips(data, ifd, header, dtype, window)

        # For multi-band with band selection, extract single band
        if arr.ndim == 3 and ifd.samples_per_pixel > 1 and band is not None:
            arr = arr[:, :, band]

        # MinIsWhite (photometric=0): invert single-band grayscale values
        if ifd.photometric == 0 and ifd.samples_per_pixel == 1:
            if arr.dtype.kind == 'u':
                arr = np.iinfo(arr.dtype).max - arr
            elif arr.dtype.kind == 'f':
                arr = -arr
    finally:
        src.close()

    return arr, geo_info
