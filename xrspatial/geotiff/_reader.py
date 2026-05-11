"""TIFF/COG reader: tile/strip assembly, windowed reads, HTTP range requests."""
from __future__ import annotations

import math
import mmap
import os as _os_module
import threading
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._compression import (
    COMPRESSION_LERC,
    COMPRESSION_NONE,
    decompress,
    fp_predictor_decode,
    lerc_decompress_with_mask,
    predictor_decode,
    unpack_bits,
)
from ._dtypes import SUB_BYTE_BPS, resolve_bits_per_sample, tiff_dtype_to_numpy
from ._geotags import (
    GeoInfo,
    GeoTransform,
    RASTER_PIXEL_IS_POINT,
    extract_geo_info,
)
from ._header import (
    IFD,
    TIFFHeader,
    parse_all_ifds,
    parse_header,
    select_overview_ifd,
    validate_tile_layout,
)

# ---------------------------------------------------------------------------
# Allocation guard: reject TIFF dimensions that would exhaust memory
# ---------------------------------------------------------------------------

#: Default maximum total pixel count (width * height * samples).
#: ~1 billion pixels, which is ~4 GB for float32 single-band.
#: Override per-call via the ``max_pixels`` keyword argument.
MAX_PIXELS_DEFAULT = 1_000_000_000


def _check_dimensions(width, height, samples, max_pixels):
    """Raise ValueError if the requested allocation exceeds *max_pixels*."""
    total = width * height * samples
    if total > max_pixels:
        raise ValueError(
            f"TIFF image dimensions ({width} x {height} x {samples} = "
            f"{total:,} pixels) exceed the safety limit of "
            f"{max_pixels:,} pixels.  Pass a larger max_pixels value to "
            f"read_to_array() if this file is legitimate."
        )


#: Default per-tile compressed-byte cap for HTTP COG reads. A crafted
#: ``TileByteCounts`` entry can declare arbitrarily many bytes, and the
#: HTTP path then tries to fetch and buffer that many bytes from the
#: server before it ever decompresses. 256 MiB tolerates legitimate
#: large tiles (RGB JPEG2000 at very high resolution can land in the
#: tens of MB) while keeping the fetch bounded. Override via the
#: ``XRSPATIAL_COG_MAX_TILE_BYTES`` environment variable.
MAX_TILE_BYTES_DEFAULT = 256 << 20  # 256 MiB


def _max_tile_bytes_from_env() -> int:
    """Read the per-tile byte cap from the environment, or fall back to the default."""
    raw = _os_module.environ.get('XRSPATIAL_COG_MAX_TILE_BYTES')
    if raw is None:
        return MAX_TILE_BYTES_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return MAX_TILE_BYTES_DEFAULT
    return max(1, val)


# ---------------------------------------------------------------------------
# Data source abstraction
# ---------------------------------------------------------------------------

#: Soft cap on the number of mmap entries the reader keeps open at once.
#: When the cache size exceeds this, the least-recently-used *idle* entry
#: (refcount 0) is closed. In-use entries are never evicted. Override via
#: the ``XRSPATIAL_GEOTIFF_MMAP_CACHE_SIZE`` environment variable.
_DEFAULT_MMAP_CACHE_SIZE = 32


def _mmap_cache_size_from_env() -> int:
    """Read the cache size cap from the environment, falling back to the default."""
    raw = _os_module.environ.get('XRSPATIAL_GEOTIFF_MMAP_CACHE_SIZE')
    if raw is None:
        return _DEFAULT_MMAP_CACHE_SIZE
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MMAP_CACHE_SIZE
    return max(1, val)


class _MmapCache:
    """Thread-safe, reference-counted, bounded LRU mmap cache.

    Multiple threads reading the same file share a single read-only mmap.
    The cache keeps idle (refcount 0) mmaps around so repeated opens of the
    same file avoid the cost of re-mapping. When the number of entries
    exceeds the cap (default 32, or ``XRSPATIAL_GEOTIFF_MMAP_CACHE_SIZE``),
    the least-recently-used *idle* entry is evicted. Entries with active
    references are never evicted.

    mmap slicing on a read-only mapping is thread-safe (no seek involved).
    """

    def __init__(self, max_size: int | None = None):
        self._lock = threading.Lock()
        # path -> entry list. Each entry is
        # [fh, mm, size, refcount, ident, orphaned]
        #
        # ``ident`` is (st_ino, st_size, st_mtime_ns) used to spot files that
        # were replaced (e.g. via ``os.replace`` on an atomic write) at the
        # same path. ``orphaned`` is True once the entry has been removed
        # from ``self._entries`` (typically because the underlying file was
        # replaced). An orphaned entry is no longer the cache slot for the
        # path, but live ``_FileSource`` instances still hold the entry list
        # by reference and decrement *its* refcount on release. This keeps
        # holders of the old mmap unaffected by any new acquires for the
        # same path. ``OrderedDict`` gives LRU semantics via move_to_end.
        self._entries: OrderedDict[str, list] = OrderedDict()
        self._max_size = (max_size if max_size is not None
                          else _mmap_cache_size_from_env())

    @staticmethod
    def _file_ident(path: str):
        """Return a (st_ino, st_size, st_mtime_ns) tuple for *path* or None."""
        try:
            st = _os_module.stat(path)
        except OSError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    @staticmethod
    def _close_entry_locked(entry):
        """Close the file handle and mmap for *entry* (must be idle)."""
        if entry[1] is not None:
            entry[1].close()
        entry[0].close()

    def acquire(self, path: str):
        """Get or create a read-only mmap for *path*.

        Returns ``(mm, size, entry)``. The opaque ``entry`` token must be
        passed back to :meth:`release` so the matching reference count is
        decremented even after the cache slot has been replaced (e.g. by an
        atomic file overwrite at the same path).
        """
        real = _os_module.path.realpath(path)
        with self._lock:
            entry = self._entries.get(real)
            ident = self._file_ident(real)
            if entry is not None:
                # If the file at this path has been replaced (different inode,
                # size, or mtime) the cached mmap is stale. Drop the entry so
                # we re-open below. If the old entry is still in use by other
                # callers, leave their mmap valid -- they still hold a
                # reference -- but mark it orphaned so a later release of
                # *that* entry closes its own resources rather than touching
                # the new cache slot.
                if ident is not None and entry[4] != ident:
                    self._entries.pop(real)
                    entry[5] = True  # orphaned
                    if entry[3] <= 0:
                        self._close_entry_locked(entry)
                    entry = None

            if entry is not None:
                entry[3] += 1
                self._entries.move_to_end(real)
                return entry[1], entry[2], entry

            fh = open(real, 'rb')
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(0)
            if size > 0:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                mm = None
            # Re-stat after opening so size matches the mmap we built.
            ident = self._file_ident(real) or (0, size, 0)
            new_entry = [fh, mm, size, 1, ident, False]
            self._entries[real] = new_entry
            self._evict_locked()
            return mm, size, new_entry

    def release(self, entry):
        """Decrement the reference count for the supplied entry token.

        When the count hits zero on a still-cached entry, it stays cached
        (keyed by realpath) until LRU eviction or :meth:`clear`. When the
        count hits zero on an orphaned entry, its file handle and mmap are
        closed immediately because no further callers can reach it.
        """
        with self._lock:
            entry[3] -= 1
            if entry[3] > 0:
                return
            if entry[5]:
                # Orphaned: not in the dict; close now.
                self._close_entry_locked(entry)
                return
            # Find the path so we can move it to the LRU tail. The entry
            # identity is unique per realpath while non-orphaned, so a
            # linear search over a small dict is fine.
            for key, ent in self._entries.items():
                if ent is entry:
                    self._entries.move_to_end(key)
                    break
            self._evict_locked()

    def _evict_locked(self):
        """Drop oldest *idle* entries until the cache is at or below the cap."""
        if len(self._entries) <= self._max_size:
            return
        # Walk from the front (oldest); only close idle (refcount 0) entries.
        # An in-use entry can still happen to be at the front if the same
        # file was acquired long ago and held; skip it.
        to_drop = []
        for key, entry in list(self._entries.items()):
            if len(self._entries) - len(to_drop) <= self._max_size:
                break
            if entry[3] <= 0:
                to_drop.append(key)
        for key in to_drop:
            entry = self._entries.pop(key)
            self._close_entry_locked(entry)

    def clear(self):
        """Close and drop all idle entries (used by tests)."""
        with self._lock:
            for key in [k for k, v in self._entries.items() if v[3] <= 0]:
                entry = self._entries.pop(key)
                self._close_entry_locked(entry)


# Module-level cache shared across all reads
_mmap_cache = _MmapCache()


class _FileSource:
    """Local file data source using a shared, thread-safe mmap cache."""

    def __init__(self, path: str):
        self._path = path
        self._mm, self._size, self._entry = _mmap_cache.acquire(path)

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
        if self._entry is not None:
            _mmap_cache.release(self._entry)
            self._entry = None


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


# ---------------------------------------------------------------------------
# HTTP range coalescing
# ---------------------------------------------------------------------------

#: Default gap threshold (bytes) for merging adjacent COG tile ranges into a
#: single GET. COG tiles are stored sequentially, so most adjacent ranges
#: differ by zero (back-to-back) or a few bytes; 1 MB tolerates small holes
#: caused by interleaved overview/mask data without ballooning over-fetch.
#: Most tiles are well under 1 MB compressed, so the coalesced GET stays
#: O(num_tiles) bytes plus at most one threshold of slack between tiles.
COALESCE_GAP_THRESHOLD_DEFAULT = 1 << 20  # 1 MB


def coalesce_ranges(
    ranges: list[tuple[int, int]],
    gap_threshold: int = COALESCE_GAP_THRESHOLD_DEFAULT,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Merge nearby ``(offset, length)`` ranges into fewer larger ones.

    Parameters
    ----------
    ranges : list of (offset, length)
        Per-tile byte ranges to fetch. Order is preserved in the
        ``mapping`` output so callers can reassemble per-tile bytes.
    gap_threshold : int
        Maximum gap, in bytes, between two adjacent ranges before they
        are merged. A gap of zero means perfectly back-to-back; larger
        gaps trade some over-fetch for fewer round-trips.

    Returns
    -------
    merged : list of (start, length)
        Coalesced ranges, sorted by ``start``. Issue one GET per entry.
    mapping : list of (merged_idx, rel_offset, length)
        For each input range (in input order), the index of the merged
        range its bytes live in, the offset within that merged range,
        and the original length. Use with :func:`split_coalesced_bytes`.

    Notes
    -----
    Empty input returns ``([], [])``. Negative gap thresholds disable
    merging entirely (every input becomes its own merged range).
    """
    if not ranges:
        return [], []

    # Tag each input with its original index so we can rebuild mapping.
    indexed = sorted(
        ((off, length, i) for i, (off, length) in enumerate(ranges)),
        key=lambda t: t[0],
    )

    merged: list[tuple[int, int]] = []
    # mapping[input_idx] -> (merged_idx, rel_offset, length)
    mapping: list[tuple[int, int, int]] = [(0, 0, 0)] * len(ranges)

    cur_start, cur_length, first_idx = indexed[0]
    cur_end = cur_start + cur_length
    members = [(first_idx, cur_start, cur_length)]

    for off, length, orig_idx in indexed[1:]:
        gap = off - cur_end
        if gap_threshold >= 0 and gap <= gap_threshold:
            # Extend current merged range. Gaps may be negative if a
            # later-listed range overlaps an earlier one; clamp so the
            # merged length covers both.
            new_end = max(cur_end, off + length)
            cur_length = new_end - cur_start
            cur_end = new_end
            members.append((orig_idx, off, length))
        else:
            merged_idx = len(merged)
            merged.append((cur_start, cur_length))
            for orig, m_off, m_len in members:
                mapping[orig] = (merged_idx, m_off - cur_start, m_len)
            cur_start, cur_length, cur_end = off, length, off + length
            members = [(orig_idx, off, length)]

    merged_idx = len(merged)
    merged.append((cur_start, cur_length))
    for orig, m_off, m_len in members:
        mapping[orig] = (merged_idx, m_off - cur_start, m_len)

    return merged, mapping


def split_coalesced_bytes(
    merged_bytes: list[bytes],
    mapping: list[tuple[int, int, int]],
) -> list[bytes]:
    """Slice merged-GET payloads back into per-tile bytes using *mapping*.

    Inverse of :func:`coalesce_ranges`. ``merged_bytes[i]`` must be the
    bytes returned by the GET for the ``i``th merged range; the output
    is one bytes object per original input range, in input order.
    """
    out: list[bytes] = [b''] * len(mapping)
    for orig_idx, (merged_idx, rel_off, length) in enumerate(mapping):
        chunk = merged_bytes[merged_idx]
        out[orig_idx] = chunk[rel_off:rel_off + length]
    return out


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

    def read_ranges(
        self,
        ranges: list[tuple[int, int]],
        max_workers: int = 8,
    ) -> list[bytes]:
        """Fetch multiple ranges concurrently using a thread pool.

        Each ``(start, length)`` pair is fetched with its own range request,
        but requests run in parallel so total wall time is bounded by the
        slowest worker rather than ``len(ranges) * RTT``.

        Returns the bytes for each range in input order.
        """
        if not ranges:
            return []
        if len(ranges) == 1:
            start, length = ranges[0]
            return [self.read_range(start, length)]

        workers = min(max_workers, len(ranges))
        results: list[bytes | None] = [None] * len(ranges)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_idx = {
                ex.submit(self.read_range, start, length): i
                for i, (start, length) in enumerate(ranges)
            }
            for fut in future_to_idx:
                idx = future_to_idx[fut]
                results[idx] = fut.result()

        return results  # type: ignore[return-value]

    def read_ranges_coalesced(
        self,
        ranges: list[tuple[int, int]],
        max_workers: int = 8,
        gap_threshold: int = COALESCE_GAP_THRESHOLD_DEFAULT,
    ) -> list[bytes]:
        """Fetch *ranges* using merged GETs where adjacent ranges allow it.

        Wrapper around :meth:`read_ranges` that first calls
        :func:`coalesce_ranges` to group nearby ranges into fewer larger
        GETs, then splits the responses back per-input via
        :func:`split_coalesced_bytes`. Returns bytes in input order, same
        as :meth:`read_ranges`.

        Setting *gap_threshold* to a negative number disables merging
        and falls back to one GET per input range.
        """
        if not ranges:
            return []
        merged, mapping = coalesce_ranges(ranges, gap_threshold=gap_threshold)
        merged_bytes = self.read_ranges(merged, max_workers=max_workers)
        return split_coalesced_bytes(merged_bytes, mapping)

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
    if not isinstance(path, str):
        return False
    if path.startswith(('http://', 'https://')):
        return False
    return '://' in path


def _is_file_like(obj) -> bool:
    """Return True if obj exposes a binary file-like interface (read+seek+tell).

    ``tell`` is required because :class:`_BytesIOSource` uses it to compute
    the buffer length via seek-to-end. ``os.PathLike`` instances don't
    expose ``read``/``seek``/``tell`` and are excluded here so that
    :func:`_coerce_path` can convert them to ``str`` upstream.
    """
    return (
        not isinstance(obj, str)
        and hasattr(obj, 'read')
        and hasattr(obj, 'seek')
        and hasattr(obj, 'tell')
    )


def _coerce_path(source):
    """Normalize ``os.PathLike`` (e.g. ``pathlib.Path``) to ``str``.

    Strings and binary file-likes pass through unchanged. Used at the top
    of every public reader/writer entry so that ``Path('mosaic.vrt')``
    dispatches to the VRT path, ``Path('x.tif')`` derives a ``name``, etc.
    """
    if isinstance(source, _os_module.PathLike):
        return _os_module.fspath(source)
    return source


class _BytesIOSource:
    """Data source backed by an in-memory or any seekable binary file-like.

    Wraps a `BytesIO` or any object exposing ``read``/``seek`` so the reader
    can issue windowed byte reads without touching the filesystem. Concurrent
    callers (e.g. parallel tile decode) are serialized through a lock around
    the seek+read pair so they don't race on the underlying buffer's cursor.
    """

    def __init__(self, fileobj):
        # _is_file_like (the gate that lets us reach this constructor)
        # already requires read/seek/tell, so we can call tell() directly
        # rather than guarding it. We do still defend against tell raising
        # on a closed/detached buffer with an informative error.
        self._fh = fileobj
        self._lock = threading.Lock()
        try:
            cur = fileobj.tell()
            fileobj.seek(0, 2)
            self._size = fileobj.tell()
            fileobj.seek(cur)
        except (OSError, ValueError) as e:
            raise ValueError(
                f"file-like source is not usable for size measurement: "
                f"{type(e).__name__}: {e}"
            ) from e

    def read_range(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b''
        with self._lock:
            self._fh.seek(start)
            return self._fh.read(length)

    def read_all(self):
        with self._lock:
            self._fh.seek(0)
            return self._fh.read()

    @property
    def size(self) -> int:
        return self._size

    def close(self):
        # Don't close the caller's buffer -- they own it.
        self._fh = None


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


def _open_source(source):
    """Open a data source (local file, URL, cloud path, or file-like)."""
    source = _coerce_path(source)
    if _is_file_like(source):
        return _BytesIOSource(source)
    if not isinstance(source, str):
        raise TypeError(
            f"source must be a str path/URL or a binary file-like object "
            f"with read+seek methods, got {type(source).__name__}")
    if source.startswith(('http://', 'https://')):
        return _HTTPSource(source)
    if _is_fsspec_uri(source):
        return _CloudSource(source)
    return _FileSource(source)


def _apply_predictor(chunk: np.ndarray, pred: int, width: int,
                     height: int, bytes_per_sample: int,
                     samples: int = 1,
                     byte_order: str = '<') -> np.ndarray:
    """Apply the appropriate predictor decode to decompressed data.

    ``width``, ``height``, ``bytes_per_sample``, and ``samples`` describe
    the raw pixel layout before predictor inversion: ``width * samples``
    samples per row, each ``bytes_per_sample`` bytes wide.

    Predictor=2 (horizontal differencing) operates at the *sample* level
    per TIFF Technical Note (libtiff/GDAL convention): the difference is
    taken between adjacent same-component samples in the sample's
    natural bit width, with stride equal to ``samples`` samples.  A
    byte-wise implementation drops the inter-byte carry for multi-byte
    samples and produces wrong values.

    Predictor=3 (floating-point) byte-swizzles each row into
    ``bytes_per_sample`` interleaved lanes of length ``width * samples``,
    per TIFF Technical Note 3.  The un-transpose stage has to put the
    MSB lane at the file's high-order byte position, which differs for
    big- vs little-endian files; ``byte_order`` carries that.
    """
    if pred == 2:
        return predictor_decode(chunk, width, height,
                                bytes_per_sample, samples=samples,
                                byte_order=byte_order)
    elif pred == 3:
        return fp_predictor_decode(chunk, width * samples, height,
                                   bytes_per_sample,
                                   big_endian=(byte_order == '>'))
    return chunk


def _packed_byte_count(pixel_count: int, bps: int) -> int:
    """Compute the number of packed bytes for sub-byte bit depths."""
    return (pixel_count * bps + 7) // 8


def _int_nodata_in_range(nodata_int: int, dtype: np.dtype) -> bool:
    """Return True iff *nodata_int* is representable as *dtype*.

    Used to gate ``dtype.type(int(...))`` casts that would otherwise raise
    ``OverflowError`` on real-world files that pair an unsigned dtype with
    a negative GDAL_NODATA sentinel (e.g. uint16 + ``-9999``). When the
    sentinel cannot be represented, the file's pixels can never match it,
    so the caller should treat the sentinel as a no-op for value matching
    (still surfacing it via ``attrs['nodata']`` so write round-trips
    preserve the original tag).
    """
    if dtype.kind not in ('u', 'i'):
        return False
    info = np.iinfo(dtype)
    return info.min <= nodata_int <= info.max


def _resolve_masked_fill(nodata_str: str | None, dtype: np.dtype):
    """Resolve the value to use when restoring LERC-masked pixels.

    Mirrors :func:`_sparse_fill_value` but defaults to NaN for floating
    dtypes when the file does not declare a nodata sentinel.  Float
    rasters with no GDAL_NODATA tag still benefit from NaN propagation
    because LERC's zero fill would silently masquerade as a real
    measurement at z == 0.

    Note: integer dtypes with no GDAL_NODATA tag fall back to ``0``,
    which is the same value LERC zero-fills masked pixels with -- in
    that case the mask application is intentionally a no-op.  We avoid
    inventing an integer sentinel (e.g. iinfo.max) because doing so
    would silently change pixel values for files that never declared
    one, breaking downstream consumers that key off the original data.

    Out-of-range integer sentinels (e.g. ``uint16`` paired with
    ``GDAL_NODATA="-9999"``, common on legacy GDAL files) cannot be
    represented in the file dtype and so cannot match any decoded
    pixel; we fall back to ``0`` rather than raising ``OverflowError``
    on the dtype cast.
    """
    if nodata_str is not None:
        try:
            v = float(nodata_str)
            if dtype.kind == 'f':
                return dtype.type(v)
            if not math.isnan(v) and not math.isinf(v):
                nodata_int = int(v)
                if _int_nodata_in_range(nodata_int, dtype):
                    return dtype.type(nodata_int)
        except (TypeError, ValueError):
            pass
    if dtype.kind == 'f':
        return dtype.type(np.nan)
    return dtype.type(0)


def _decode_strip_or_tile(data_slice, compression, width, height, samples,
                          bps, bytes_per_sample, is_sub_byte, dtype, pred,
                          byte_order='<', jpeg_tables=None,
                          masked_fill=None):
    """Decompress, apply predictor, unpack sub-byte, and reshape a strip/tile.

    Parameters
    ----------
    byte_order : str
        '<' for little-endian, '>' for big-endian.  When the file byte
        order differs from the system's native order, pixel data is
        byte-swapped after decompression.
    jpeg_tables : bytes or None
        Raw bytes of the file's JPEGTables tag (347), or None if the file
        doesn't have one. GDAL-style tiled JPEG TIFFs store DQT/DHT tables
        once in this tag and each tile is a JPEG fragment that depends on
        them; the JPEG decoder splices the tables in before handing the
        tile to libjpeg. Ignored for non-JPEG compressions.
    masked_fill : scalar or None
        Fill value written into pixels that the LERC valid-mask flags as
        invalid.  Only consulted for ``compression == COMPRESSION_LERC``
        when the decoder returns a non-trivial mask; ignored for every
        other codec.  Callers should compute it once per IFD via
        :func:`_resolve_masked_fill` (typically NaN for float dtypes or
        the parsed ``GDAL_NODATA`` sentinel).  When ``None``, masked
        pixels are left at LERC's zero fill.

    Returns an array shaped (height, width) or (height, width, samples).
    """
    pixel_count = width * height * samples
    if is_sub_byte:
        expected = _packed_byte_count(pixel_count, bps)
    else:
        expected = pixel_count * bytes_per_sample

    lerc_mask = None
    if compression == COMPRESSION_LERC:
        # LERC needs special handling: lerc.decode also returns a
        # valid-mask which the generic decompress() dispatcher discards.
        # We capture it here so masked pixels can be restored to nodata
        # below, instead of leaking LERC's zero fill into the output.
        # Forward ``expected`` so the wrapper rejects bombs at the
        # blob-header level rather than after the full buffer is
        # materialised (issue #1625).
        decoded_bytes, lerc_mask = lerc_decompress_with_mask(
            data_slice, expected_size=expected)
        chunk = np.frombuffer(decoded_bytes, dtype=np.uint8)
    else:
        chunk = decompress(data_slice, compression, expected,
                           width=width, height=height, samples=samples,
                           jpeg_tables=jpeg_tables)

    # Validate the decompressed byte count.  A truncated deflate stream or a
    # buggy compressor can produce fewer or more bytes than expected.  Without
    # this check the downstream reshape raises an opaque "cannot reshape array
    # of size N into shape (h, w)" that hides which tile/strip broke.  Edge
    # tiles in a valid TIFF still decompress to the full tile_height x
    # tile_width (the caller slices the top-left region), so this only fires
    # on genuine corruption.
    if chunk.size != expected:
        raise ValueError(
            f"Decompressed tile/strip size mismatch: expected {expected} "
            f"bytes for a {width} x {height} x {samples} block "
            f"(bps={bps}, compression={compression}), got {chunk.size}. "
            f"The TIFF data is likely truncated or corrupt."
        )

    if pred in (2, 3) and not is_sub_byte:
        if not chunk.flags.writeable:
            chunk = chunk.copy()
        chunk = _apply_predictor(chunk, pred, width, height,
                                 bytes_per_sample, samples=samples,
                                 byte_order=byte_order)

    if is_sub_byte:
        pixels = unpack_bits(chunk, bps, pixel_count)
    else:
        # Use the file's byte order for the view, then convert to native
        file_dtype = dtype.newbyteorder(byte_order)
        pixels = chunk.view(file_dtype)
        if file_dtype.byteorder not in ('=', '|', _NATIVE_ORDER):
            pixels = pixels.astype(dtype)

    if samples > 1:
        out = pixels.reshape(height, width, samples)
    else:
        out = pixels.reshape(height, width)

    # Restore nodata in positions LERC flagged as invalid.  LERC
    # zero-fills masked pixels in the data array, which would otherwise
    # be indistinguishable from real zero readings downstream.
    if lerc_mask is not None and masked_fill is not None:
        mask_arr = np.asarray(lerc_mask)
        if mask_arr.ndim == 2 and out.ndim == 3:
            mask_arr = mask_arr[..., None]
        invalid = np.broadcast_to(mask_arr == 0, out.shape)
        if invalid.any():
            if not out.flags.writeable:
                out = out.copy()
            np.putmask(out, invalid, masked_fill)
    return out


import sys as _sys
_NATIVE_ORDER = '<' if _sys.byteorder == 'little' else '>'


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
        try:
            v = float(nodata_str)
            if dtype.kind == 'f':
                return dtype.type(v)
            if not math.isnan(v) and not math.isinf(v):
                nodata_int = int(v)
                if _int_nodata_in_range(nodata_int, dtype):
                    return dtype.type(nodata_int)
        except (TypeError, ValueError):
            pass
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
# Strip reader
# ---------------------------------------------------------------------------

def _read_strips(data: bytes, ifd: IFD, header: TIFFHeader,
                 dtype: np.dtype, window=None,
                 max_pixels: int = MAX_PIXELS_DEFAULT) -> np.ndarray:
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
    max_pixels : int
        Maximum allowed pixel count (width * height * samples).

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
    bps = resolve_bits_per_sample(ifd.bits_per_sample)
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)

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

    _check_dimensions(out_w, out_h, samples, max_pixels)

    # Sparse strips (StripByteCounts == 0) must materialise as nodata or 0
    # rather than be decoded.  Pre-fill the result so any skipped strips
    # land on a known fill value.
    sparse = _has_sparse(byte_counts)
    if sparse:
        fill = _sparse_fill_value(ifd, dtype)
        if samples > 1:
            result = np.full((out_h, out_w, samples), fill, dtype=dtype)
        else:
            result = np.full((out_h, out_w), fill, dtype=dtype)
    elif samples > 1:
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
                if byte_counts[global_idx] == 0:
                    # Sparse strip: result is already pre-filled.
                    continue
                strip_row = strip_idx * rps
                strip_rows = min(rps, height - strip_row)
                if strip_rows <= 0:
                    continue

                strip_data = data[offsets[global_idx]:offsets[global_idx] + byte_counts[global_idx]]
                strip_pixels = _decode_strip_or_tile(
                    strip_data, compression, width, strip_rows, 1,
                    bps, bytes_per_sample, is_sub_byte, dtype, pred,
                    byte_order=header.byte_order,
                    jpeg_tables=jpeg_tables,
                    masked_fill=masked_fill)

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
            if byte_counts[strip_idx] == 0:
                # Sparse strip: result is already pre-filled.
                continue

            strip_data = data[offsets[strip_idx]:offsets[strip_idx] + byte_counts[strip_idx]]
            strip_pixels = _decode_strip_or_tile(
                strip_data, compression, width, strip_rows, samples,
                bps, bytes_per_sample, is_sub_byte, dtype, pred,
                byte_order=header.byte_order,
                jpeg_tables=jpeg_tables,
                masked_fill=masked_fill)

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
                dtype: np.dtype, window=None,
                max_pixels: int = MAX_PIXELS_DEFAULT) -> np.ndarray:
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
    max_pixels : int
        Maximum allowed pixel count (width * height * samples).

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
    bps = resolve_bits_per_sample(ifd.bits_per_sample)
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)

    offsets = ifd.tile_offsets
    byte_counts = ifd.tile_byte_counts
    if offsets is None or byte_counts is None:
        raise ValueError("Missing tile offsets or byte counts")

    if tw <= 0 or th <= 0:
        raise ValueError(
            f"Invalid tile dimensions: TileWidth={tw}, TileLength={th}")

    # Reject crafted tile dims that would force huge per-tile allocations.
    # A single tile's decoded bytes must also fit under the pixel budget.
    _check_dimensions(tw, th, samples, max_pixels)

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

    _check_dimensions(out_w, out_h, samples, max_pixels)

    # Reject malformed TIFFs whose declared tile grid exceeds the number of
    # supplied TileOffsets entries. Silent skipping in the CPU loop below
    # would mask the problem, and the GPU path reads OOB. See issue #1219.
    validate_tile_layout(ifd)

    # Sparse tiles (TileByteCounts == 0) must materialise as nodata or 0
    # rather than be decoded.  Pre-fill the result so any skipped tiles
    # land on a known fill value; otherwise sparse regions would leak
    # uninitialised memory (full-image read) or stay zeroed regardless
    # of the file's nodata setting (windowed read).
    sparse = _has_sparse(byte_counts)
    if sparse:
        fill = _sparse_fill_value(ifd, dtype)
        if samples > 1:
            result = np.full((out_h, out_w, samples), fill, dtype=dtype)
        else:
            result = np.full((out_h, out_w), fill, dtype=dtype)
    else:
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

    # Build list of tiles to decode.  Sparse tiles (byte_count==0) are
    # skipped here -- the result is pre-filled with the sparse fill value.
    tile_jobs = []
    for band_idx in range(band_count):
        band_tile_offset = band_idx * tiles_per_band if band_count > 1 else 0
        tile_samples = 1 if band_count > 1 else samples

        for tr in range(tile_row_start, tile_row_end):
            for tc in range(tile_col_start, tile_col_end):
                tile_idx = band_tile_offset + tr * tiles_across + tc
                if tile_idx >= len(offsets):
                    continue
                if byte_counts[tile_idx] == 0:
                    continue
                tile_jobs.append((band_idx, tr, tc, tile_idx, tile_samples))

    # Decode tiles in parallel when the work per tile is large enough to
    # outweigh the thread-pool overhead. Uncompressed multi-tile reads also
    # benefit because numpy frombuffer + slice copies aren't free at large
    # tile sizes. Threshold (~64K decoded pixels per tile) was picked to
    # avoid pool overhead on small 64x64 / 128x128 tile reads. The bound
    # is inclusive so the default tile_size=256 (256*256 == 64*1024) lands
    # on the parallel path -- a strict `>` excluded the most common tile
    # size in practice (issue #1551).
    n_tiles = len(tile_jobs)
    tile_pixels = tw * th
    use_parallel = (n_tiles > 1 and tile_pixels >= 64 * 1024)

    def _decode_one(job):
        band_idx, tr, tc, tile_idx, tile_samples = job
        tile_data = data[offsets[tile_idx]:offsets[tile_idx] + byte_counts[tile_idx]]
        return _decode_strip_or_tile(
            tile_data, compression, tw, th, tile_samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor
        import os as _os
        n_workers = min(n_tiles, _os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            decoded = list(pool.map(_decode_one, tile_jobs))
    else:
        decoded = [_decode_one(job) for job in tile_jobs]

    # Place decoded tiles into the output array
    for (band_idx, tr, tc, tile_idx, tile_samples), tile_pixels in zip(tile_jobs, decoded):
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

def _parse_cog_http_meta(
    source: _HTTPSource,
    overview_level: int | None = None,
) -> tuple[TIFFHeader, IFD, GeoInfo, bytes]:
    """Fetch + parse the leading IFDs of an HTTP COG once.

    Issues one (or rarely two) range GET(s) for the leading 16 KB / 64 KB
    of the file, parses the header and IFD list, and returns the selected
    IFD plus the raw header bytes (kept for ``extract_geo_info`` callers
    that might want the IFD's tag offsets).

    Pulled out of :func:`_read_cog_http` so :func:`read_geotiff_dask`
    can parse metadata once per graph rather than once per chunk task
    (P5: each delayed task used to fire its own 16 KB header GET).
    """
    header_bytes = source.read_range(0, 16384)
    header = parse_header(header_bytes)
    ifds = parse_all_ifds(header_bytes, header)

    # parse_all_ifds bails the moment it walks past the bytes we
    # fetched, so a header GET that lands short of the first IFD's
    # offset returns an empty list. Retry with a larger window in that
    # case; this is *not* a partial-IFD recovery (overviews chained
    # past the first 16 KiB are still loaded lazily by other readers).
    if len(ifds) == 0:
        header_bytes = source.read_range(0, 65536)
        ifds = parse_all_ifds(header_bytes, header)

    if len(ifds) == 0:
        raise ValueError("No IFDs found in COG")

    ifd = select_overview_ifd(ifds, overview_level)
    geo_info = extract_geo_info(ifd, header_bytes, header.byte_order)
    return header, ifd, geo_info, header_bytes


def _read_cog_http(url: str, overview_level: int | None = None,
                   band: int | None = None,
                   max_pixels: int = MAX_PIXELS_DEFAULT,
                   ) -> tuple[np.ndarray, GeoInfo]:
    """Read a COG via HTTP range requests.

    Tile fetches run concurrently through a small thread pool so that the
    total wall time is bounded by the slowest tile request rather than
    ``num_tiles * RTT``. The pool size can be overridden with the
    ``XRSPATIAL_COG_HTTP_WORKERS`` environment variable (default 8).

    Parameters
    ----------
    url : str
        HTTP(S) URL to the COG file.
    overview_level : int or None
        Which overview to read (0 = full res, 1 = first overview, etc.).
    band : int
        Band index (0-based, for multi-band files).
    max_pixels : int
        Maximum allowed pixel count (width * height * samples).

    Returns
    -------
    (array, geo_info) tuple
    """
    source = _HTTPSource(url)
    header, ifd, geo_info, header_bytes = _parse_cog_http_meta(
        source, overview_level=overview_level)

    arr = _fetch_decode_cog_http_tiles(
        source, header, ifd, max_pixels=max_pixels, window=None)
    source.close()
    return arr, geo_info


def _fetch_decode_cog_http_tiles(
    source: _HTTPSource,
    header: TIFFHeader,
    ifd: IFD,
    *,
    max_pixels: int = MAX_PIXELS_DEFAULT,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Fetch and decode the tiles of a tiled COG over HTTP.

    Pulled out of :func:`_read_cog_http` so that callers with
    pre-parsed metadata (notably :func:`read_geotiff_dask`) can reuse a
    single IFD parse across many tile-fetch calls. When *window* is
    given, only tiles intersecting the window are fetched + decoded;
    the result is sized to the (clamped) window rather than the full
    image. Coalescing of adjacent ranges still applies.
    """
    bps = resolve_bits_per_sample(ifd.bits_per_sample)
    dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
    if not ifd.is_tiled:
        # Stripped HTTP COG: fall back to a full read. Window is honoured
        # by slicing the decoded array.
        all_data = source.read_all()
        arr = _read_strips(all_data, ifd, header, dtype)
        if window is not None:
            r0, c0, r1, c1 = window
            return arr[r0:r1, c0:c1]
        return arr

    width = ifd.width
    height = ifd.height
    tw = ifd.tile_width
    th = ifd.tile_height
    samples = ifd.samples_per_pixel
    compression = ifd.compression
    pred = ifd.predictor
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)

    offsets = ifd.tile_offsets
    byte_counts = ifd.tile_byte_counts

    if tw <= 0 or th <= 0:
        raise ValueError(
            f"Invalid tile dimensions: TileWidth={tw}, TileLength={th}")

    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)

    # Cap the *materialised* pixel count, not the declared image size.
    # A windowed HTTP read of a multi-billion-pixel COG only allocates
    # the window, so capping the full image would reject legitimate
    # tiled reads. The full-image cap still applies for whole-file
    # reads (window is None). The single-tile budget always applies.
    if window is None:
        _check_dimensions(width, height, samples, max_pixels)
    _check_dimensions(tw, th, samples, max_pixels)

    # Reject malformed TIFFs whose declared tile grid exceeds the supplied
    # TileOffsets length. See issue #1219.
    validate_tile_layout(ifd)

    if window is None:
        r0_out, c0_out, r1_out, c1_out = 0, 0, height, width
    else:
        r0_out, c0_out, r1_out, c1_out = window
        r0_out = max(0, r0_out)
        c0_out = max(0, c0_out)
        r1_out = min(height, r1_out)
        c1_out = min(width, c1_out)

    out_h = r1_out - r0_out
    out_w = c1_out - c0_out
    _check_dimensions(out_w, out_h, samples, max_pixels)

    # Sparse tiles (TileByteCounts == 0) need to land on the file's nodata
    # value (or 0 if unset) rather than uninitialised memory.  Detect them
    # up front so the result buffer is pre-filled before tile placement.
    sparse = _has_sparse(byte_counts)
    if sparse:
        fill = _sparse_fill_value(ifd, dtype)
        if samples > 1:
            result = np.full((out_h, out_w, samples), fill, dtype=dtype)
        else:
            result = np.full((out_h, out_w), fill, dtype=dtype)
    elif samples > 1:
        result = np.empty((out_h, out_w, samples), dtype=dtype)
    else:
        result = np.empty((out_h, out_w), dtype=dtype)

    tile_row_start = r0_out // th
    tile_row_end = min(math.ceil(r1_out / th), tiles_down)
    tile_col_start = c0_out // tw
    tile_col_end = min(math.ceil(c1_out / tw), tiles_across)

    # Pass 1: collect every tile's range and where it lands in the output.
    # Empty tiles (byte_count == 0) and any tile_idx beyond the offsets
    # array are skipped here so the fetch list stays exactly aligned with
    # the placements list.
    #
    # Each tile's compressed size is checked against the cap returned by
    # _max_tile_bytes_from_env() (default MAX_TILE_BYTES_DEFAULT, 256 MiB)
    # before the fetch list is built. A crafted COG can claim arbitrarily
    # large TileByteCounts; without this guard the HTTP layer would issue
    # a Range request sized by the attacker's value (issue #1536). The cap
    # is overridable via XRSPATIAL_COG_MAX_TILE_BYTES; the local-mmap path
    # is naturally bounded by file size and does not need this check.
    max_tile_bytes = _max_tile_bytes_from_env()
    fetch_ranges: list[tuple[int, int]] = []
    placements: list[tuple[int, int]] = []  # (tr, tc) per fetched tile
    for tr in range(tile_row_start, tile_row_end):
        for tc in range(tile_col_start, tile_col_end):
            tile_idx = tr * tiles_across + tc
            if tile_idx >= len(offsets):
                continue
            off = offsets[tile_idx]
            bc = byte_counts[tile_idx]
            if bc == 0:
                continue
            if bc > max_tile_bytes:
                raise ValueError(
                    f"TIFF tile {tile_idx} declares "
                    f"TileByteCount={bc:,} bytes, which exceeds the HTTP "
                    f"COG safety cap of {max_tile_bytes:,} bytes. The "
                    f"file is malformed or attempting denial-of-service. "
                    f"Override via XRSPATIAL_COG_MAX_TILE_BYTES if this "
                    f"file is legitimate."
                )
            fetch_ranges.append((off, bc))
            placements.append((tr, tc))

    # Pass 2: fetch all tile bytes in parallel. Worker pool size is tunable
    # via XRSPATIAL_COG_HTTP_WORKERS so users on very slow links can dial
    # it up without code changes.
    #
    # COG tile offsets are sorted and usually back-to-back, so we coalesce
    # adjacent ranges into fewer larger GETs (P2). The 1 MB gap threshold
    # tolerates small interleaved metadata between tiles without dragging
    # in unrelated overview data. Set XRSPATIAL_COG_COALESCE_GAP=-1 to
    # disable merging (one GET per tile, the legacy behaviour).
    try:
        workers = max(1, int(_os_module.environ.get('XRSPATIAL_COG_HTTP_WORKERS', '8')))
    except ValueError:
        workers = 8
    try:
        gap = int(_os_module.environ.get(
            'XRSPATIAL_COG_COALESCE_GAP',
            str(COALESCE_GAP_THRESHOLD_DEFAULT)))
    except ValueError:
        gap = COALESCE_GAP_THRESHOLD_DEFAULT
    tile_bytes_list = source.read_ranges_coalesced(
        fetch_ranges, max_workers=workers, gap_threshold=gap)

    # Pass 3: decode each tile and place it (clipped to the window).
    for (tr, tc), tile_data in zip(placements, tile_bytes_list):
        tile_pixels = _decode_strip_or_tile(
            tile_data, compression, tw, th, samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

        # Tile position in image coordinates.
        ty0 = tr * th
        tx0 = tc * tw
        ty1 = ty0 + th
        tx1 = tx0 + tw

        # Intersect with the requested window.
        iy0 = max(ty0, r0_out)
        ix0 = max(tx0, c0_out)
        iy1 = min(ty1, r1_out)
        ix1 = min(tx1, c1_out)
        if iy1 <= iy0 or ix1 <= ix0:
            continue

        # Source slice within the decoded tile pixels.
        sy0 = iy0 - ty0
        sx0 = ix0 - tx0
        sy1 = sy0 + (iy1 - iy0)
        sx1 = sx0 + (ix1 - ix0)

        # Destination slice within the output buffer.
        dy0 = iy0 - r0_out
        dx0 = ix0 - c0_out
        dy1 = iy1 - r0_out
        dx1 = ix1 - c0_out

        result[dy0:dy1, dx0:dx1] = tile_pixels[sy0:sy1, sx0:sx1]

    return result


# ---------------------------------------------------------------------------
# Main read function
# ---------------------------------------------------------------------------

def _apply_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Reorient a decoded TIFF array according to the Orientation tag (274).

    The TIFF 6.0 spec defines eight orientations describing where the
    *first row* and *first column* of the stored data sit relative to the
    visual top-left of the image:

    ===  =================  ========================================
     1   top-left           identity (default, no transform)
     2   top-right          mirror horizontally (flip columns)
     3   bottom-right       rotate 180 degrees
     4   bottom-left        mirror vertically (flip rows)
     5   left-top           transpose (rows<->columns)
     6   right-top          rotate 90 clockwise
     7   right-bottom       transverse (anti-transpose)
     8   left-bottom        rotate 90 counter-clockwise
    ===  =================  ========================================

    Values 5-8 swap rows and columns: the file's stored width becomes the
    output's height and vice versa.

    The input ``arr`` is shaped ``(height, width)`` or
    ``(height, width, samples)``. Multi-band 3D arrays only have their
    first two axes transformed; the sample axis is preserved.
    """
    if orientation == 1:
        return arr
    if orientation == 2:
        return np.ascontiguousarray(arr[:, ::-1])
    if orientation == 3:
        return np.ascontiguousarray(arr[::-1, ::-1])
    if orientation == 4:
        return np.ascontiguousarray(arr[::-1, :])
    # Orientations 5-8 swap rows and columns.
    if arr.ndim == 3:
        # Transpose only the spatial axes; keep the sample axis trailing.
        if orientation == 5:
            return np.ascontiguousarray(arr.transpose(1, 0, 2))
        if orientation == 6:
            return np.ascontiguousarray(arr.transpose(1, 0, 2)[:, ::-1])
        if orientation == 7:
            return np.ascontiguousarray(arr.transpose(1, 0, 2)[::-1, ::-1])
        if orientation == 8:
            return np.ascontiguousarray(arr.transpose(1, 0, 2)[::-1, :])
    else:
        if orientation == 5:
            return np.ascontiguousarray(arr.T)
        if orientation == 6:
            return np.ascontiguousarray(arr.T[:, ::-1])
        if orientation == 7:
            return np.ascontiguousarray(arr.T[::-1, ::-1])
        if orientation == 8:
            return np.ascontiguousarray(arr.T[::-1, :])
    raise ValueError(
        f"Invalid TIFF Orientation tag value: {orientation} "
        f"(must be 1-8 per TIFF 6.0)"
    )


def read_to_array(source, *, window=None, overview_level: int | None = None,
                  band: int | None = None,
                  max_pixels: int = MAX_PIXELS_DEFAULT,
                  ) -> tuple[np.ndarray, GeoInfo]:
    """Read a GeoTIFF/COG to a numpy array.

    Parameters
    ----------
    source : str or binary file-like
        File path, URL, or a file-like object with ``read``/``seek``.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop).
    overview_level : int or None
        Overview level (0 = full res).
    band : int
        Band index for multi-band files.
    max_pixels : int
        Maximum allowed total pixel count (width * height * samples).
        Prevents memory exhaustion from crafted TIFF headers.
        Default is 1 billion (~4 GB for float32 single-band).

    Returns
    -------
    (np.ndarray, GeoInfo) tuple
    """
    source = _coerce_path(source)
    if isinstance(source, str) and source.startswith(('http://', 'https://')):
        return _read_cog_http(source, overview_level=overview_level, band=band,
                              max_pixels=max_pixels)

    # Local file, cloud storage, or file-like buffer: read all bytes then parse
    if _is_file_like(source):
        src = _BytesIOSource(source)
    elif _is_fsspec_uri(source):
        src = _CloudSource(source)
    else:
        src = _FileSource(source)
    data = src.read_all()

    try:
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)

        if len(ifds) == 0:
            raise ValueError("No IFDs found in TIFF file")

        # Select IFD, skipping any mask IFDs
        ifd = select_overview_ifd(ifds, overview_level)

        bps = resolve_bits_per_sample(ifd.bits_per_sample)
        dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        geo_info = extract_geo_info(ifd, data, header.byte_order)

        # Orientation tag (274): values 2-8 mean the stored pixel order
        # differs from display order. We need to remap the array post
        # decode. A windowed read against a non-default orientation has
        # ambiguous semantics (does the window refer to file pixels or
        # display pixels?) so we reject that combo rather than guess.
        # ``read_geotiff_dask`` chunks the file by issuing windowed reads,
        # so this check also rejects ``chunks=`` for non-default
        # orientation; the error mentions both so the failure is easy to
        # diagnose if it surfaces under dask.
        orientation = ifd.orientation
        if orientation != 1 and window is not None:
            raise ValueError(
                f"Orientation tag (274) is {orientation}; windowed reads "
                f"(window=...) and dask-chunked reads (chunks=...) are not "
                f"supported for non-default orientation. Read the full "
                f"array first, then slice."
            )

        if ifd.is_tiled:
            arr = _read_tiles(data, ifd, header, dtype, window,
                              max_pixels=max_pixels)
        else:
            arr = _read_strips(data, ifd, header, dtype, window,
                               max_pixels=max_pixels)

        # Extract the requested band before reorienting so we work on a
        # smaller 2D array rather than reorienting a full multi-band cube
        # only to slice it afterwards.
        if arr.ndim == 3 and ifd.samples_per_pixel > 1 and band is not None:
            arr = arr[:, :, band]

        if orientation != 1:
            # Use the *file* dimensions (before orientation) for the
            # transform-flip math below. After ``_apply_orientation`` the
            # array shape may swap (orientations 5-8), so capture them now.
            file_h = arr.shape[0]
            file_w = arr.shape[1]
            arr = _apply_orientation(arr, orientation)
            # The pixel buffer was just remapped; the transform that maps
            # display pixels back to geographic coordinates needs the
            # matching remap or the y/x coords still describe the file's
            # original layout.
            #
            # Orientations 2-4 are pure mirror flips: the array shape stays
            # the same, but the displayed origin moves to the opposite
            # edge along whichever axes were flipped. Update origin and
            # sign of the affected pixel scale so xarray coords land on
            # the right geographic positions.
            #
            # Orientations 5-8 swap rows and columns. Pixel sizes swap
            # axes so coord array lengths match the new shape. Signs are
            # preserved rather than coerced to north-up since some
            # legitimate files use a non-standard sign convention
            # (south-up, west-up). For 6/7/8 (rotations + flips, not a
            # pure transpose) the swap is geometrically inexact for
            # georef'd files: a strict implementation would also adjust
            # origin and re-sign per axis. Those files are vanishingly
            # rare in practice (TIFF Orientation 5-8 with a meaningful
            # ModelTransformation); warn so the user knows to verify.
            t = geo_info.transform
            # Only georeferenced files have a meaningful transform to flip.
            # Plain TIFFs with an Orientation tag but no GeoTIFF tags get
            # their pixel buffer remapped above; their default transform
            # is left untouched and the downstream consumer falls back to
            # integer pixel coords.
            if not geo_info.has_georef:
                pass
            elif orientation in (2, 3, 4):
                # PixelIsPoint tiepoints are at pixel centers, so the
                # opposite-edge pixel sits ``(N-1) * step`` away. PixelIsArea
                # tiepoints are at pixel edges, so the opposite edge is
                # ``N * step`` away. The two cases collapse to a single
                # formula below by switching the offset.
                if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
                    x_shift = file_w - 1
                    y_shift = file_h - 1
                else:
                    x_shift = file_w
                    y_shift = file_h
                new_origin_x = t.origin_x
                new_origin_y = t.origin_y
                new_px_w = t.pixel_width
                new_px_h = t.pixel_height
                if orientation in (2, 3):  # x flipped
                    new_origin_x = t.origin_x + x_shift * t.pixel_width
                    new_px_w = -t.pixel_width
                if orientation in (3, 4):  # y flipped
                    new_origin_y = t.origin_y + y_shift * t.pixel_height
                    new_px_h = -t.pixel_height
                geo_info.transform = GeoTransform(
                    origin_x=new_origin_x,
                    origin_y=new_origin_y,
                    pixel_width=new_px_w,
                    pixel_height=new_px_h,
                )
            elif orientation in (5, 6, 7, 8):
                geo_info.transform = GeoTransform(
                    origin_x=t.origin_x,
                    origin_y=t.origin_y,
                    pixel_width=t.pixel_height,
                    pixel_height=t.pixel_width,
                )
                if (geo_info.crs_epsg is not None
                        or geo_info.crs_wkt is not None):
                    import warnings
                    warnings.warn(
                        f"Orientation {orientation} swaps spatial axes on "
                        f"a georeferenced file; the returned coords are "
                        f"shape-correct but the geographic transform may "
                        f"need manual adjustment.",
                        stacklevel=2,
                    )

        # MinIsWhite (photometric=0): invert single-band grayscale values
        if ifd.photometric == 0 and ifd.samples_per_pixel == 1:
            if arr.dtype.kind == 'u':
                arr = np.iinfo(arr.dtype).max - arr
            elif arr.dtype.kind == 'f':
                arr = -arr
    finally:
        src.close()

    return arr, geo_info
