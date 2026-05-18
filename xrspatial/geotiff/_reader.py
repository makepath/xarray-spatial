"""TIFF/COG reader: tile/strip assembly, windowed reads, HTTP range requests."""
from __future__ import annotations

import math
import mmap
import os as _os_module
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import urllib3

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
    extract_geo_info_with_overview_inheritance,
)
from ._header import (
    IFD,
    TIFFHeader,
    parse_all_ifds,
    parse_header,
    select_overview_ifd,
    validate_tile_layout,
)
from ._validation import _validate_predictor_sample_format

# ---------------------------------------------------------------------------
# Allocation guard: reject TIFF dimensions that would exhaust memory
# ---------------------------------------------------------------------------

#: Default maximum total pixel count (width * height * samples).
#: ~1 billion pixels, which is ~4 GB for float32 single-band.
#: Override per-call via the ``max_pixels`` keyword argument.
MAX_PIXELS_DEFAULT = 1_000_000_000

#: Default byte ceiling for eager reads from fsspec cloud sources
#: (``s3://``, ``gs://``, ``az://``, ``abfs://``, ``memory://``, ...).
#: ``_CloudSource`` knows the object size up front via ``fsspec.size()``,
#: so checking against this budget runs before any data is downloaded.
#: 256 MiB is comfortable for typical demo COGs while bounding the blast
#: radius of a crafted or oversized remote object. Override per call
#: with the ``max_cloud_bytes`` kwarg, or env-wide with
#: ``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES``. Pass ``max_cloud_bytes=None``
#: to skip the check entirely (the pre-#1928 behaviour). See issue #1928.
MAX_CLOUD_BYTES_DEFAULT = 256 * 1024 * 1024

#: Sentinel for "caller did not pass ``max_cloud_bytes``". Distinguishes
#: that case from ``max_cloud_bytes=None`` (caller explicitly disabling
#: the check) so the env-var override has somewhere to land.
_MAX_CLOUD_BYTES_SENTINEL = object()


def _resolve_max_cloud_bytes(max_cloud_bytes):
    """Return the effective cloud byte budget.

    Precedence:
    1. Explicit kwarg (including ``None`` to disable) wins.
    2. ``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES`` env var, if set to a
       positive int.
    3. :data:`MAX_CLOUD_BYTES_DEFAULT`.
    """
    if max_cloud_bytes is not _MAX_CLOUD_BYTES_SENTINEL:
        return max_cloud_bytes
    env = _os_module.environ.get('XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES')
    if env:
        try:
            v = int(env)
        except ValueError:
            return MAX_CLOUD_BYTES_DEFAULT
        if v > 0:
            return v
    return MAX_CLOUD_BYTES_DEFAULT


class PixelSafetyLimitError(ValueError):
    """Raised when a requested TIFF allocation exceeds max_pixels."""


class CloudSizeLimitError(ValueError):
    """Raised when an eager fsspec read would exceed ``max_cloud_bytes``.

    Distinct from :class:`PixelSafetyLimitError` because the cloud check
    runs against the compressed object size before any TIFF header parse,
    so the pixel-count message would be misleading.
    """


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


#: Default per-tile (or per-strip) compressed-byte cap. A crafted
#: ``TileByteCounts`` / ``StripByteCounts`` entry can declare arbitrarily
#: many bytes. On HTTP, the reader would issue a Range GET sized by the
#: attacker's value; on local files, mmap slicing is bounded by the file
#: size but a small compressed slice can still decompress (deflate/zstd/
#: lzw) into hundreds of MiB. 256 MiB tolerates legitimate large tiles
#: (RGB JPEG2000 at very high resolution can land in the tens of MB)
#: while keeping the fetch / decode bounded. Override via the
#: ``XRSPATIAL_COG_MAX_TILE_BYTES`` environment variable. Issues #1536
#: (HTTP) and #1664 (local).
MAX_TILE_BYTES_DEFAULT = 256 << 20  # 256 MiB


def _max_tile_bytes_from_env() -> int:
    """Read the per-tile byte cap from the environment, or fall back to the default.

    Non-integer, empty, zero, or negative values all fall back to
    ``MAX_TILE_BYTES_DEFAULT``. Matches the policy used by the HTTP
    timeout helpers so callers don't accidentally set an unreachable
    1-byte cap with ``XRSPATIAL_COG_MAX_TILE_BYTES=-1``.
    """
    raw = _os_module.environ.get('XRSPATIAL_COG_MAX_TILE_BYTES')
    if raw is None:
        return MAX_TILE_BYTES_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return MAX_TILE_BYTES_DEFAULT
    return val if val > 0 else MAX_TILE_BYTES_DEFAULT


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
    """Return the module-level urllib3 PoolManager, building it on first call."""
    global _http_pool
    if _http_pool is not None:
        return _http_pool
    _http_pool = urllib3.PoolManager(
        num_pools=10,
        maxsize=10,
        retries=urllib3.Retry(
            total=2,
            backoff_factor=0.1,
            # Redirects are *not* delegated to urllib3 -- they're
            # followed manually in ``_HTTPSource._request`` so each
            # ``Location`` runs through ``_validate_http_url`` before
            # the next GET. Issue #1664.
            redirect=False,
        ),
    )
    return _http_pool


_http_pool = None


# ---------------------------------------------------------------------------
# SSRF defenses for _HTTPSource (issue #1664)
# ---------------------------------------------------------------------------

#: Maximum number of redirects to follow when fetching a TIFF over HTTP.
_HTTP_MAX_REDIRECTS = 5

#: Default connect / read timeouts (seconds) for HTTP TIFF fetches.
_HTTP_CONNECT_TIMEOUT_DEFAULT = 10.0
_HTTP_READ_TIMEOUT_DEFAULT = 30.0

#: URL schemes that ``_HTTPSource`` accepts. The HTTP source is a Range
#: GET implementation backed by urllib3, which only speaks ``http`` and
#: ``https`` -- widening here would just push the failure to connect time.
#: fsspec handles every other ``scheme://`` and is routed separately by
#: :func:`_open_source`.
_HTTP_ALLOWED_SCHEMES = ('http', 'https')


def _http_allow_private_hosts() -> bool:
    """Return True if loopback / link-local / private IPs are allowed."""
    raw = _os_module.environ.get('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS')
    if raw is None:
        return False
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def _http_timeout_from_env(var_name: str, default: float) -> float:
    """Parse a positive-float timeout from the named env var, or fall back."""
    raw = _os_module.environ.get(var_name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return val if val > 0 else default


def _http_connect_timeout() -> float:
    return _http_timeout_from_env(
        'XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT',
        _HTTP_CONNECT_TIMEOUT_DEFAULT,
    )


def _http_read_timeout() -> float:
    return _http_timeout_from_env(
        'XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT',
        _HTTP_READ_TIMEOUT_DEFAULT,
    )


class UnsafeURLError(ValueError):
    """Raised when an HTTP URL fails the SSRF allow-list check.

    Subclasses ``ValueError`` so existing callers that catch ``ValueError``
    on bad input keep working. Carries the offending URL on ``.url`` for
    structured logging.
    """

    def __init__(self, msg: str, url: str | None = None):
        super().__init__(msg)
        self.url = url


def _ip_is_private(ip_str: str) -> bool:
    """Return True if *ip_str* is a loopback, link-local, or private IP.

    Covers both IPv4 and IPv6. Multicast and unspecified addresses are
    treated as unsafe (no legitimate reason to GET a TIFF from them, and
    cloud metadata sometimes lives behind link-local IPv6).
    """
    import ipaddress
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Not a literal IP -- caller must resolve it first.
        return False
    # ``is_private`` is True for RFC1918 (10/8, 172.16/12, 192.168/16),
    # the IPv6 ULAs (fc00::/7), and -- in stdlib >= 3.4 -- also for
    # loopback / link-local. Stay explicit so we don't depend on subtle
    # behaviour across Python versions.
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def _validate_http_url(url: str) -> str | None:
    """Reject URLs that would let ``_HTTPSource`` reach unsafe destinations.

    Enforces:

    * scheme in ``_HTTP_ALLOWED_SCHEMES`` (http / https)
    * hostname resolves to at least one non-loopback, non-link-local,
      non-private IP (override via ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS``)
    * hostname is non-empty

    Raises :class:`UnsafeURLError` (a ``ValueError`` subclass) on any of
    the above. Issue #1664.

    Returns the first resolved IP literal so the caller can pin the
    actual TCP connection to that exact address. Without pinning, the
    HTTP source resolves the hostname a second time at connect-time,
    leaving a DNS-rebind window: a hostile resolver can return a public
    IP here and a private IP at connect. Returns ``None`` when the
    escape hatch ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` is set, in
    which case the caller falls back to urllib3's default DNS path.
    Issues #1664 (validation) and #1846 (pinning).
    """
    import socket
    from urllib.parse import urlparse

    if not isinstance(url, str) or not url:
        raise UnsafeURLError(
            "HTTP source requires a non-empty URL string", url=url)

    parsed = urlparse(url)
    scheme = (parsed.scheme or '').lower()
    if scheme not in _HTTP_ALLOWED_SCHEMES:
        raise UnsafeURLError(
            f"URL scheme {scheme!r} is not in the allow-list "
            f"{_HTTP_ALLOWED_SCHEMES}. Only HTTP(S) is supported; other "
            f"schemes are dispatched via fsspec. URL: {url!r}",
            url=url,
        )

    host = parsed.hostname
    if not host:
        raise UnsafeURLError(
            f"URL {url!r} has no hostname", url=url)

    if _http_allow_private_hosts():
        # Escape hatch: skip resolution and skip pinning. Callers that
        # opt into private hosts knowingly trade the DNS-rebind defence
        # for the ability to hit localhost/dev services without having
        # to pre-resolve. ``None`` tells the caller to use the default
        # urllib3 DNS path.
        return None

    # Resolve and reject if any resolved IP is in a private/loopback/link-
    # local/multicast range. Rejecting on *any* match (rather than all)
    # prevents DNS-rebind tricks that return both a public and a private
    # IP for the same name. socket.getaddrinfo handles IPv4, IPv6, and
    # literal IP strings uniformly.
    try:
        infos = socket.getaddrinfo(host, parsed.port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise UnsafeURLError(
            f"could not resolve host {host!r}: {e}", url=url) from e

    first_safe_ip: str | None = None
    for info in infos:
        sockaddr = info[4]
        # sockaddr is (ip, port) for AF_INET and (ip, port, flow, scope)
        # for AF_INET6 -- the IP is always index 0.
        ip_str = sockaddr[0]
        # IPv6 scoped addresses come back as 'fe80::1%eth0' -- strip the
        # zone id before passing to ipaddress.
        if '%' in ip_str:
            ip_str = ip_str.split('%', 1)[0]
        if _ip_is_private(ip_str):
            raise UnsafeURLError(
                f"host {host!r} resolves to {ip_str!r}, which is in a "
                f"loopback / link-local / private range. Set "
                f"XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1 to allow.",
                url=url,
            )
        if first_safe_ip is None:
            first_safe_ip = ip_str

    # Defensive: ``getaddrinfo`` returning an empty list would be
    # unusual, but if it did we have nothing to pin to.
    if first_safe_ip is None:
        raise UnsafeURLError(
            f"host {host!r} produced no usable IP addresses", url=url)
    return first_safe_ip


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

#: Per-tile pixel count at and above which the local and HTTP tile-read paths
#: spread codec decode across a ``ThreadPoolExecutor``. Below this, pool
#: startup costs outweigh the parallelism win (issue #1551). Bound is inclusive
#: so the default ``tile_size=256`` (256*256 == 64*1024) lands on the parallel
#: path. Used by both ``_read_tiles`` and ``_fetch_decode_cog_http_tiles``.
_PARALLEL_DECODE_PIXEL_THRESHOLD = 64 * 1024


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


# ---------------------------------------------------------------------------
# Pinned-IP urllib3 connection (issue #1846)
# ---------------------------------------------------------------------------
#
# Security: ``_validate_http_url`` resolves the hostname and rejects any URL
# that lands on a private / loopback / link-local IP. Without the pinning
# below, urllib3 would resolve the hostname *again* at connect time. A
# hostile DNS server can return a public IP at validation time and a
# private IP at connect time, bypassing the guard (DNS rebinding, TOCTOU).
#
# To close that gap we build a custom urllib3 connection that:
#
# 1. Opens the TCP socket to the validated IP literal (via
#    ``socket.create_connection`` directly, so we never re-consult DNS).
# 2. Leaves ``self.host`` set to the original hostname, which is what
#    urllib3 writes into the HTTP ``Host`` header (needed for virtual
#    hosting on shared hosts).
# 3. Leaves ``self.server_hostname`` set to the original hostname, which
#    is what urllib3 feeds into TLS SNI and into certificate hostname
#    verification (so HTTPS cert validation still checks the cert was
#    issued for the hostname the caller asked for, not for the IP).
#
# Residual scope:
# - Each redirect hop is freshly resolved and freshly pinned. The pin
#   does not persist across hostname changes; each hop gets its own
#   validate-and-pin pair.
# - An attacker who legitimately controls multiple public IPs on a
#   hostname can still influence which one we pick (we take the first).
#   They cannot make us connect to a private IP.


def _build_pinned_connection_classes():
    """Build pinned ``HTTPConnection`` / ``HTTPSConnection`` subclasses.

    Built lazily on first use so the urllib3 connection submodules are
    only imported when ``_HTTPSource`` is actually exercised. The
    subclasses override ``_new_conn`` to dial the validated IP directly.
    """
    import socket as _socket
    from urllib3.connection import HTTPConnection, HTTPSConnection
    from urllib3.exceptions import (
        ConnectTimeoutError,
        NameResolutionError,
        NewConnectionError,
    )

    class _PinnedHTTPConnection(HTTPConnection):
        """``HTTPConnection`` that dials a fixed IP, ignoring DNS.

        ``pinned_ip`` is set after construction (urllib3 builds the
        connection through ``ConnectionCls(host=..., port=..., ...)``
        without passing custom kwargs, so we attach the pin via a
        per-pool factory rather than via __init__).
        """

        pinned_ip: str | None = None

        def _new_conn(self) -> _socket.socket:
            ip = self.pinned_ip
            if ip is None:
                # Should never happen for pools we build, but fall
                # back to default behaviour rather than crash.
                return super()._new_conn()
            try:
                sock = _socket.create_connection(
                    (ip, self.port),
                    self.timeout,
                    source_address=self.source_address,
                )
            except _socket.gaierror as e:
                # Pinning to a literal IP shouldn't trigger DNS, but
                # IPv6 literals can still fail to resolve into a
                # sockaddr on misconfigured stacks.
                raise NameResolutionError(self.host, self, e) from e
            except _socket.timeout as e:
                raise ConnectTimeoutError(
                    self,
                    f"Connection to {self.host} ({ip}) timed out. "
                    f"(connect timeout={self.timeout})",
                ) from e
            except OSError as e:
                raise NewConnectionError(
                    self,
                    f"Failed to establish a new connection to "
                    f"{self.host} ({ip}): {e}",
                ) from e
            # Apply the socket options urllib3 normally sets (nodelay
            # etc.). Mirrors HTTPConnection._new_conn behaviour.
            for opt in self.socket_options or []:
                sock.setsockopt(*opt)
            return sock

    class _PinnedHTTPSConnection(HTTPSConnection):
        """HTTPS version: dial the pinned IP, keep SNI on the hostname."""

        pinned_ip: str | None = None

        def _new_conn(self) -> _socket.socket:
            ip = self.pinned_ip
            if ip is None:
                return super()._new_conn()
            try:
                sock = _socket.create_connection(
                    (ip, self.port),
                    self.timeout,
                    source_address=self.source_address,
                )
            except _socket.gaierror as e:
                raise NameResolutionError(self.host, self, e) from e
            except _socket.timeout as e:
                raise ConnectTimeoutError(
                    self,
                    f"Connection to {self.host} ({ip}) timed out. "
                    f"(connect timeout={self.timeout})",
                ) from e
            except OSError as e:
                raise NewConnectionError(
                    self,
                    f"Failed to establish a new connection to "
                    f"{self.host} ({ip}): {e}",
                ) from e
            for opt in self.socket_options or []:
                sock.setsockopt(*opt)
            return sock

    return _PinnedHTTPConnection, _PinnedHTTPSConnection


_pinned_conn_classes = None


def _get_pinned_conn_classes():
    """Return cached (PinnedHTTPConn, PinnedHTTPSConn) tuple."""
    global _pinned_conn_classes
    if _pinned_conn_classes is None:
        _pinned_conn_classes = _build_pinned_connection_classes()
    return _pinned_conn_classes


def _make_pinned_pool(scheme: str, host: str, port: int, pinned_ip: str,
                     connect_timeout: float, read_timeout: float):
    """Build a urllib3 ConnectionPool whose connections dial *pinned_ip*.

    The pool's ``host`` stays the original hostname so the HTTP ``Host``
    header and TLS SNI / cert verification use the name, not the IP.
    """
    import urllib3

    HTTPConn, HTTPSConn = _get_pinned_conn_classes()

    if scheme == 'https':
        # Subclass the connection so we can stamp ``pinned_ip`` on the
        # class -- urllib3 instantiates it via ``ConnectionCls(host=...,
        # port=..., ...)`` and there's no straightforward kwarg to pass
        # extra attributes. A per-pool subclass is the cleanest hook.
        class _Conn(HTTPSConn):
            pass
        _Conn.pinned_ip = pinned_ip
        pool = urllib3.HTTPSConnectionPool(
            host=host,
            port=port,
            timeout=urllib3.Timeout(
                connect=connect_timeout, read=read_timeout),
            maxsize=10,
            block=False,
            retries=urllib3.Retry(
                total=2, backoff_factor=0.1, redirect=False),
            # ``server_hostname`` is what becomes the TLS SNI string
            # and the name urllib3 verifies the cert against. We keep
            # it set to the original hostname so cert validation still
            # checks the name, not the IP literal.
            server_hostname=host,
        )
        pool.ConnectionCls = _Conn
        return pool

    class _Conn(HTTPConn):
        pass
    _Conn.pinned_ip = pinned_ip
    pool = urllib3.HTTPConnectionPool(
        host=host,
        port=port,
        timeout=urllib3.Timeout(
            connect=connect_timeout, read=read_timeout),
        maxsize=10,
        block=False,
        retries=urllib3.Retry(
            total=2, backoff_factor=0.1, redirect=False),
    )
    pool.ConnectionCls = _Conn
    return pool


class _HTTPSource:
    """HTTP data source using range requests with connection reuse.

    Uses :class:`urllib3.PoolManager` for the unpinned escape-hatch path
    and a per-hop pinned ``HTTP[S]ConnectionPool`` for the default path,
    so TCP and TLS state is reused across range requests to the same host.
    urllib3 is a hard install dependency; there is no stdlib fallback.
    The stdlib ``urllib.request`` path was removed in #2050 because it
    re-resolved the hostname at request time, defeating the IP pin that
    closes the DNS-rebinding TOCTOU from #1846.
    """

    def __init__(self, url: str):
        # Security: ``_validate_http_url`` runs the SSRF allow-list
        # (scheme + host) and returns the validated IP literal so we
        # can pin the actual TCP connection to that exact address.
        # Without pinning there is a DNS-rebind TOCTOU: urllib3 would
        # resolve the hostname a second time at connect-time, and a
        # hostile resolver can flip from public to private IP between
        # the two lookups. The escape hatch
        # ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` returns ``None``
        # here -- we then fall back to urllib3's default DNS path.
        # UnsafeURLError subclasses ValueError so callers that already
        # catch ValueError keep working. Issues #1664, #1846.
        self._pinned_ip = _validate_http_url(url)
        self._url = url
        self._size = None
        # Connection-pool manager is still shared across instances for
        # the unpinned escape-hatch path. The pinned path builds its
        # own ``HTTP[S]ConnectionPool`` per (scheme, host, port, ip)
        # tuple and caches it on ``self`` so subsequent range requests
        # to the same hop reuse TCP/TLS state.
        self._pool = _get_http_pool()
        self._pinned_pools: dict[tuple, object] = {}
        self._connect_timeout = _http_connect_timeout()
        self._read_timeout = _http_read_timeout()

    def _urllib3_timeout(self):
        """Build a :class:`urllib3.Timeout` for this source."""
        return urllib3.Timeout(
            connect=self._connect_timeout, read=self._read_timeout)

    def _get_pinned_pool(self, scheme: str, host: str, port: int | None,
                         pinned_ip: str):
        """Return (creating if needed) a pinned pool for this hop.

        Pools are cached per (scheme, host, port, ip) tuple so range
        requests against the same URL reuse the TCP/TLS connection.
        Redirect hops to a different hostname get their own pool with
        their own pin.
        """
        if port is None:
            port = 443 if scheme == 'https' else 80
        key = (scheme, host, port, pinned_ip)
        pool = self._pinned_pools.get(key)
        if pool is None:
            pool = _make_pinned_pool(
                scheme, host, port, pinned_ip,
                self._connect_timeout, self._read_timeout)
            self._pinned_pools[key] = pool
        return pool

    def _request(self, headers: dict | None = None,
                 preload_content: bool = True):
        """Issue a GET with manual, validated redirect following.

        urllib3's built-in redirect follower has no validation hook, so
        we set ``redirect=False`` and walk the chain ourselves. Each
        ``Location`` runs through :func:`_validate_http_url` before the
        next GET, defeating a public-to-private 3xx bounce. Cap at
        :data:`_HTTP_MAX_REDIRECTS` hops. Issue #1664.

        Security: each hop also gets the resolved IP pinned into the
        connection's TCP target. The pin closes the DNS-rebind window
        that exists between ``getaddrinfo`` in the validator and the
        second ``getaddrinfo`` urllib3 would otherwise do at connect
        time. Issue #1846.

        ``preload_content=False`` returns a streaming response: the body
        is not buffered into ``resp.data`` and the caller must drain it
        via ``resp.stream(...)``. Used by :meth:`read_all` when a
        ``max_bytes`` budget is in play, so the body is bounded
        on-the-wire instead of being fully allocated before the cap is
        checked. Issue #2051.
        """
        from urllib.parse import urljoin
        timeout = self._urllib3_timeout()
        current_url = self._url
        current_pin = self._pinned_ip
        for _ in range(_HTTP_MAX_REDIRECTS + 1):
            pool = self._pool_for_request(current_url, current_pin)
            resp = pool.request(
                'GET', current_url,
                headers=headers,
                timeout=timeout,
                redirect=False,
                preload_content=preload_content,
            )
            if 300 <= resp.status < 400 and resp.status != 304:
                location = resp.headers.get('Location')
                if not location:
                    return resp
                # Release the redirect response's connection back to
                # the pool. ``preload_content=True`` (the default) drains
                # the body for us, but the streaming path
                # (``preload_content=False``, used by ``read_all`` with a
                # byte budget) leaves the connection borrowed -- if we
                # do not release it here, subsequent hops will allocate
                # fresh connections every time.
                if not preload_content:
                    try:
                        resp.release_conn()
                    except Exception:  # noqa: BLE001
                        pass
                # Resolve relative ``Location`` against the URL we just
                # requested, not against ``self._url``: chained
                # redirects can land us on a different origin.
                next_url = urljoin(current_url, location)
                # Re-validate and re-pin for the new hop. If the new
                # hop is a different hostname, this gives us a fresh
                # IP to pin to; if the escape hatch is set, this
                # returns ``None`` and we fall back to unpinned.
                current_pin = _validate_http_url(next_url)
                current_url = next_url
                continue
            return resp
        raise UnsafeURLError(
            f"More than {_HTTP_MAX_REDIRECTS} HTTP redirects "
            f"starting from {self._url!r}",
            url=self._url,
        )

    def _pool_for_request(self, url: str, pinned_ip: str | None):
        """Pick the right pool for *url*: pinned if we have an IP,
        otherwise the shared default ``PoolManager``.

        Tests that monkeypatch ``self._pool`` to a mock keep working
        because we still consult ``self._pool`` when no pin is set.
        """
        if pinned_ip is None:
            return self._pool
        from urllib.parse import urlparse
        parsed = urlparse(url)
        scheme = (parsed.scheme or '').lower()
        host = parsed.hostname or ''
        # If a test has swapped ``self._pool`` for a mock, honour that
        # mock for hops where the test wants to script responses. We
        # detect the mock by checking whether ``self._pool`` is the
        # module-level urllib3 PoolManager. Anything else (e.g. the
        # ``_MockPool`` in the SSRF tests) wins so existing tests stay
        # decoupled from this change.
        if self._pool is not _http_pool:
            return self._pool
        return self._get_pinned_pool(scheme, host, parsed.port, pinned_ip)

    def read_range(self, start: int, length: int) -> bytes:
        # Match the ``b''``-for-non-positive-length convention used by
        # other source implementations (e.g. ``_BytesIOSource``).
        # Without this guard, ``Range: bytes=<start>-<start-1>`` goes on
        # the wire, which is an invalid range and triggers a 416 from
        # well-behaved servers (or worse, an arbitrarily large 200 body
        # from misbehaving ones).
        if length <= 0:
            return b''
        end = start + length - 1
        headers = {'Range': f'bytes={start}-{end}'}
        resp = self._request(headers=headers)
        return self._validate_range_response(
            status=resp.status,
            content_range=resp.headers.get('Content-Range'),
            data=resp.data,
            start=start,
            length=length,
        )

    @staticmethod
    def _validate_range_response(*, status, content_range, data,
                                 start: int, length: int) -> bytes:
        """Reject HTTP responses that do not satisfy the Range request.

        Without this, three things can go wrong silently (issue #1735):

        - the server returns a 4xx/5xx body (urllib3 by default does not
          raise on non-2xx, so the bytes would be handed to the caller);
        - the server ignores ``Range`` for a non-zero start and returns
          the whole object as a 200 with no ``Content-Range``, handing
          the codec wrong-offset bytes;
        - the body is shorter or longer than what the server advertised
          via ``Content-Range``, which surfaces later inside a decoder
          as an opaque error far from the real cause.

        A short response near EOF is legitimate: a fixed-size header
        prefetch (e.g. ``read_range(0, 16384)``) will hit the end of a
        smaller file and the server returns ``Content-Range: bytes
        0-(size-1)/size``. The validator accepts that as long as the
        Content-Range starts at the requested offset and the body
        length matches what Content-Range advertises.

        Returns the validated bytes, sliced to at most ``length`` bytes
        in the "server ignored Range and returned the whole object as
        200" case (start == 0, no Content-Range). Other branches return
        ``data`` unchanged.
        """
        if status is None or status not in (200, 206):
            raise OSError(
                f"HTTP range request returned status {status}; expected "
                f"206 Partial Content (or 200 with full body)."
            )
        if content_range is None:
            # No Content-Range. A 200 with no Content-Range is only safe
            # when the caller asked for the beginning of the object; for
            # any other ``start`` the bytes returned do not correspond
            # to the requested offset.
            if status == 206:
                raise OSError(
                    "HTTP 206 response missing Content-Range header."
                )
            if start != 0:
                raise OSError(
                    f"HTTP server returned status 200 with no "
                    f"Content-Range for a range request starting at "
                    f"byte {start}; refusing to use the body as a "
                    f"range fetch."
                )
            if len(data) < min(length, 1):
                # Empty body but caller wanted bytes.
                raise OSError("HTTP 200 response body is empty.")
            # Server ignored Range and returned the full object as 200.
            # The implicit contract is "at most ``length`` bytes"; slice
            # so a 16 KiB prefetch against a 2 GiB object doesn't drag
            # the whole thing into memory.
            if len(data) > length:
                return data[:length]
            return data
        # Parse ``bytes <start>-<end>/<total-or-*>``. Reject anything
        # that does not start at the requested offset; allow ``end`` to
        # be lower than requested when EOF was hit.
        try:
            unit, _, spec = content_range.partition(' ')
            rng, _, _total = spec.partition('/')
            cr_start_s, _, cr_end_s = rng.partition('-')
            cr_start = int(cr_start_s)
            cr_end = int(cr_end_s)
        except ValueError:
            raise OSError(
                f"HTTP Content-Range header {content_range!r} could "
                f"not be parsed."
            ) from None
        if unit != 'bytes':
            raise OSError(
                f"HTTP Content-Range unit {unit!r} is not 'bytes'."
            )
        if cr_start != start:
            raise OSError(
                f"HTTP Content-Range {content_range!r} starts at byte "
                f"{cr_start}, but the request started at byte {start}."
            )
        if cr_end < cr_start:
            raise OSError(
                f"HTTP Content-Range {content_range!r} has end "
                f"({cr_end}) below start ({cr_start})."
            )
        if cr_end - cr_start + 1 > length:
            raise OSError(
                f"HTTP Content-Range {content_range!r} advertises more "
                f"bytes than were requested (length={length})."
            )
        expected_len = cr_end - cr_start + 1
        if len(data) != expected_len:
            raise OSError(
                f"HTTP range body length {len(data)} does not match "
                f"the {expected_len} bytes advertised by "
                f"Content-Range {content_range!r}."
            )
        return data

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

    def read_all(self, max_bytes: int | None = None) -> bytes:
        """Fetch the full body, optionally bounded by ``max_bytes``.

        ``max_bytes`` caps both the advertised ``Content-Length`` (rejected
        up front before any bytes are read into memory) and the actual
        body size (streamed and aborted once ``max_bytes + 1`` bytes have
        arrived). The ``+ 1`` is the over-shoot detector: a body that
        exactly matches the cap passes, but a server that ignores or
        lies about ``Content-Length`` and streams more bytes is caught
        as soon as the first extra byte lands.

        Without a cap, a tiny TIFF header (e.g. 100x100) that survives
        :func:`_check_dimensions` can still be served as a multi-gigabyte
        HTTP body and the whole body is allocated before TIFF parsing
        gets a chance to reject it. Issue #2051.

        ``max_bytes=None`` preserves the legacy unbounded behaviour for
        callers that already gate the read upstream (e.g. cloud reads
        gated by :data:`max_cloud_bytes`).
        """
        if max_bytes is None:
            return self._request().data
        # Stream the body so the cap is enforced before the bytes land
        # in memory. ``preload_content=False`` makes urllib3 hand us
        # the response without buffering ``resp.data``.
        resp = self._request(preload_content=False)
        try:
            self._check_content_length(resp.headers, max_bytes)
            return self._read_capped(resp, max_bytes)
        finally:
            try:
                resp.release_conn()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _check_content_length(headers, max_bytes: int) -> None:
        """Reject a response whose advertised ``Content-Length`` exceeds the cap.

        This is the cheap pre-flight check; we still cap the actual read
        below in case the server omits the header or lies about it.
        """
        raw = None
        try:
            raw = headers.get('Content-Length')
        except AttributeError:
            return
        if raw is None:
            return
        try:
            declared = int(raw)
        except (TypeError, ValueError):
            return
        if declared > max_bytes:
            raise OSError(
                f"HTTP response declares Content-Length={declared:,} "
                f"bytes, which exceeds the byte budget of "
                f"{max_bytes:,} bytes computed from the TIFF strip "
                f"table. The file is malformed or attempting "
                f"denial-of-service. Issue #2051."
            )

    @staticmethod
    def _read_capped(resp, max_bytes: int) -> bytes:
        """Stream-read a urllib3 response, aborting past ``max_bytes``.

        Read at most ``max_bytes + 1`` bytes. The extra byte is the
        over-shoot probe: if it arrives the server lied or omitted
        ``Content-Length`` and tried to send a larger body. Raise
        :class:`OSError` so callers that already handle network failures
        also handle this.
        """
        chunks: list[bytes] = []
        received = 0
        for chunk in resp.stream(amt=65536, decode_content=True):
            if not chunk:
                continue
            chunks.append(chunk)
            received += len(chunk)
            if received > max_bytes:
                raise OSError(
                    f"HTTP response body exceeded the byte budget of "
                    f"{max_bytes:,} bytes (received {received:,} bytes "
                    f"before abort). The server likely ignored or lied "
                    f"about Content-Length. Issue #2051."
                )
        return b''.join(chunks)

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

    def read_ranges(
        self,
        ranges: list[tuple[int, int]],
        max_workers: int = 8,
    ) -> list[bytes]:
        """Fetch multiple ranges concurrently using a thread pool.

        Mirrors :meth:`_HTTPSource.read_ranges` so that
        :func:`_fetch_decode_cog_http_tiles` can drive a cloud source
        the same way it drives an HTTP source. See PR #1755.
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

        Mirrors :meth:`_HTTPSource.read_ranges_coalesced` so the tiled
        COG decode path can coalesce neighbouring tiles when reading
        from object storage.
        """
        if not ranges:
            return []
        merged, mapping = coalesce_ranges(ranges, gap_threshold=gap_threshold)
        merged_bytes = self.read_ranges(merged, max_workers=max_workers)
        return split_coalesced_bytes(merged_bytes, mapping)

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
        # Try ``int`` first so 64-bit sentinels survive without the
        # float64 round-trip; fall back to ``float`` for NaN / Inf /
        # scientific notation / fractional values.  See issue #1847.
        from ._geotags import _parse_nodata_str as _parse_nd
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
        # Use the file's byte order for the view, then convert to native.
        # The view dtype must match the on-disk sample width: float16
        # files (bps=16 + SampleFormat=3) are auto-promoted to float32
        # for the user-visible array, but the raw bytes have to be
        # viewed as float16 first then cast (#1941). Detect the
        # promotion via the bps-vs-dtype.itemsize mismatch so the
        # surrounding pipeline stays unchanged for byte-equal cases.
        if dtype.itemsize * 8 != bps and bps == 16 and dtype.kind == 'f':
            storage_dtype = np.dtype('float16').newbyteorder(byte_order)
            pixels = chunk.view(storage_dtype).astype(dtype)
        else:
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
        # Try ``int`` first so 64-bit sentinels survive without the
        # float64 round-trip; fall back to ``float`` for NaN / Inf /
        # scientific notation / fractional values.  See issue #1847.
        from ._geotags import _parse_nodata_str as _parse_nd
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
    # Source-IFD dim check (issue #2053). The tiled path is already
    # covered by ``validate_tile_layout``; this is its stripped-path
    # parity. Run before any window clamping so a malformed
    # ``ImageWidth=0`` IFD fails at the source rather than collapsing
    # to an empty post-clamp window.
    _check_source_dimensions(width, height, samples)
    compression = ifd.compression
    rps = ifd.rows_per_strip
    offsets = ifd.strip_offsets
    byte_counts = ifd.strip_byte_counts
    pred = ifd.predictor
    _validate_predictor_sample_format(pred, ifd.sample_format)
    bps = resolve_bits_per_sample(ifd.bits_per_sample)
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)

    if offsets is None or byte_counts is None:
        raise ValueError("Missing strip offsets or byte counts")

    # Per-strip compressed-byte cap (issue #1664). Mirrors the HTTP path:
    # a crafted ``StripByteCounts`` can declare a huge value and even
    # though mmap slicing on the local path is bounded by the file size,
    # the slice is still passed into the decompressor which can expand
    # a few KiB of crafted deflate/zstd into gigabytes of decoded output.
    # Override via ``XRSPATIAL_COG_MAX_TILE_BYTES`` (the env var is shared
    # with the tile path because the budget is the same).
    max_tile_bytes = _max_tile_bytes_from_env()
    for _strip_idx, _bc in enumerate(byte_counts):
        if _bc > max_tile_bytes:
            raise ValueError(
                f"TIFF strip {_strip_idx} declares "
                f"StripByteCount={_bc:,} bytes, which exceeds the "
                f"per-strip safety cap of {max_tile_bytes:,} bytes. "
                f"The file is malformed or attempting denial-of-service. "
                f"Override via XRSPATIAL_COG_MAX_TILE_BYTES if this file "
                f"is legitimate."
            )

    # A corrupt header can report RowsPerStrip=0, which would divide by zero
    # below.  Reject it as a typed parse error rather than letting the
    # ZeroDivisionError leak out to the caller.
    if rps is None or rps <= 0:
        raise ValueError(f"Invalid RowsPerStrip: {rps!r}")

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

    # StripByteCounts must have at least one entry per strip; a corrupt count
    # field can shrink it.  Detect the mismatch after the dimension safety
    # check so an oversized header raises the safety-limit error first, then
    # raise a typed ValueError here instead of IndexError when the loop
    # indexes past the end.
    #
    # For PlanarConfiguration=2 (separate / planar) each sample plane has its
    # own run of strips, so the table must hold strips_per_band * samples
    # entries.  PlanarConfiguration=1 (chunky) interleaves samples within a
    # single run of strips_per_band entries.
    strips_per_band = (height + rps - 1) // rps
    if planar == 2 and samples > 1:
        n_strips_expected = strips_per_band * samples
        if len(offsets) < n_strips_expected or len(byte_counts) < n_strips_expected:
            raise ValueError(
                f"Strip table truncated for planar layout "
                f"(PlanarConfiguration=2): expected "
                f"{n_strips_expected} entries "
                f"({strips_per_band} strips x {samples} samples), got "
                f"offsets={len(offsets)}, byte_counts={len(byte_counts)}")
    else:
        n_strips_expected = strips_per_band
        if len(offsets) < n_strips_expected or len(byte_counts) < n_strips_expected:
            raise ValueError(
                f"Strip table truncated: expected {n_strips_expected} entries, "
                f"got offsets={len(offsets)}, byte_counts={len(byte_counts)}")

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
        first_strip = r0 // rps
        last_strip = min((r1 - 1) // rps, strips_per_band - 1)

        for band_idx in range(samples):
            band_offset = band_idx * strips_per_band
            for strip_idx in range(first_strip, last_strip + 1):
                global_idx = band_offset + strip_idx
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
    _validate_predictor_sample_format(pred, ifd.sample_format)
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

    # Reject crafted tile dims (e.g. TileWidth = 2**31). This guards the
    # TIFF header against malformed values; it is not the caller's output
    # budget. The output-window check below uses ``max_pixels`` and is
    # what enforces the user's per-call memory cap. The source-read path
    # under ``read_vrt`` (#1796) relies on that output check to honour a
    # small caller ``max_pixels`` against a normal-tile source; see
    # #1823.
    _check_dimensions(tw, th, samples, MAX_PIXELS_DEFAULT)

    # Per-tile compressed-byte cap (issue #1664). Same env var as the
    # HTTP path. mmap slicing is bounded by the file size, but the slice
    # gets handed to the decompressor, and a small slice can balloon
    # into gigabytes through deflate / zstd / lzw / lerc.
    max_tile_bytes = _max_tile_bytes_from_env()
    for _tile_idx, _bc in enumerate(byte_counts):
        if _bc > max_tile_bytes:
            raise ValueError(
                f"TIFF tile {_tile_idx} declares "
                f"TileByteCount={_bc:,} bytes, which exceeds the "
                f"per-tile safety cap of {max_tile_bytes:,} bytes. "
                f"The file is malformed or attempting denial-of-service. "
                f"Override via XRSPATIAL_COG_MAX_TILE_BYTES if this file "
                f"is legitimate."
            )

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
    # tile sizes. Threshold is shared with the HTTP COG path below
    # (issue #1551).
    n_tiles = len(tile_jobs)
    tile_pixels = tw * th
    use_parallel = (n_tiles > 1 and tile_pixels >= _PARALLEL_DECODE_PIXEL_THRESHOLD)

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

#: Initial prefetch size for ``_parse_cog_http_meta``. Sized for the common
#: case (a single-IFD COG with modest GeoTIFF tags) so the fast path is a
#: single range GET.
INITIAL_HTTP_HEADER_BYTES = 16 * 1024

#: Upper bound on how far ``_parse_cog_http_meta`` will grow its prefetch
#: buffer before giving up. 4 MiB comfortably covers deep pyramids whose
#: IFD chains plus tag arrays (TileOffsets, GeoAsciiParams, GDAL_METADATA)
#: extend far past the initial fetch window. See issue #1718.
MAX_HTTP_HEADER_BYTES = 4 * 1024 * 1024


def _ifd_required_extent(
    ifds: list[IFD], header: TIFFHeader, data_len: int,
) -> int:
    """Return the highest byte offset the parsed IFDs reference.

    Used to decide whether the prefetch buffer is large enough to hold the
    entire IFD chain plus every out-of-line tag value. We compare this
    against ``len(data)`` in :func:`_parse_cog_http_meta`; if it exceeds the
    buffer, the chain is truncated and the caller must grow and retry.

    The walk re-derives each tag's value-area placement directly from the
    IFD layout (entry table base + entry slot) rather than re-parsing the
    raw bytes. For out-of-line tags ``parse_ifd`` already resolved the
    pointer and validated ``ptr + size <= data_len``; the *interesting*
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


def _parse_cog_http_meta(
    source: _HTTPSource,
    overview_level: int | None = None,
) -> tuple[TIFFHeader, IFD, GeoInfo, bytes]:
    """Fetch + parse the leading IFDs of an HTTP COG once.

    The fast path is a single 16 KiB range GET. When the IFD chain or its
    out-of-line tag arrays extend past that window the buffer is doubled
    and reparsed until either the chain is fully resolved or the cap at
    :data:`MAX_HTTP_HEADER_BYTES` is reached. Real COGs whose pyramid
    metadata legitimately exceeds the cap need a different strategy
    (lazy per-IFD reads); the cap exists to bound a malformed-file blast
    radius rather than to constrain valid pyramids.

    Pulled out of :func:`_read_cog_http` so :func:`read_geotiff_dask`
    can parse metadata once per graph rather than once per chunk task
    (P5: each delayed task used to fire its own 16 KB header GET).
    """
    fetch_size = INITIAL_HTTP_HEADER_BYTES
    header_bytes = source.read_range(0, fetch_size)
    header = parse_header(header_bytes)

    last_len = len(header_bytes)
    ifds: list[IFD] = []
    while True:
        try:
            ifds = parse_all_ifds(header_bytes, header)
            required = _ifd_required_extent(ifds, header, len(header_bytes))
            # Chain is fully resolved when every IFD parsed cleanly and
            # the tail next_ifd_offset is reachable within the buffer
            # (required == 0 means end-of-chain).
            if ifds and required <= len(header_bytes):
                break
        except ValueError:
            # parse_ifd raises when an out-of-line tag points past the
            # buffer. Treat it the same as a truncated chain: grow and
            # retry. If we are already at the cap and still failing, let
            # the next iteration's cap check raise a clear error.
            ifds = []

        if fetch_size >= MAX_HTTP_HEADER_BYTES:
            raise ValueError(
                f"COG IFD chain or tag arrays extend past "
                f"MAX_HTTP_HEADER_BYTES={MAX_HTTP_HEADER_BYTES} bytes; "
                f"the file may be malformed or its pyramid metadata is "
                f"unusually large for HTTP prefetch")
        fetch_size = min(fetch_size * 2, MAX_HTTP_HEADER_BYTES)
        header_bytes = source.read_range(0, fetch_size)
        # Server returned the same number of bytes as last time: we have
        # hit EOF on the underlying file. No point growing further; if
        # the IFD chain still doesn't resolve, the file is truncated.
        if len(header_bytes) == last_len:
            try:
                ifds = parse_all_ifds(header_bytes, header)
            except ValueError:
                ifds = []
            break
        last_len = len(header_bytes)

    if len(ifds) == 0:
        raise ValueError("No IFDs found in COG")

    ifd = select_overview_ifd(ifds, overview_level)
    # When the requested IFD is an overview that lacks its own geokeys
    # (the common case for COG writers, including this package's
    # ``to_geotiff``), inherit and rescale the georef from the level-0
    # IFD so overview reads do not silently lose CRS / transform.
    # See issue #1640.
    geo_info = extract_geo_info_with_overview_inheritance(
        ifd, ifds, header_bytes, header.byte_order)
    return header, ifd, geo_info, header_bytes


def _read_cog_http(url: str, overview_level: int | None = None,
                   band: int | None = None,
                   max_pixels: int = MAX_PIXELS_DEFAULT,
                   window: tuple[int, int, int, int] | None = None,
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
    window : tuple or None
        ``(row_start, col_start, row_stop, col_stop)``. Forwarded to
        ``_fetch_decode_cog_http_tiles`` so HTTP reads honour the same
        windowed contract as the local-file path. See issue #1669.

    Returns
    -------
    (array, geo_info) tuple
    """
    source = _HTTPSource(url)
    # Issue #1816: wrap everything after the ``_HTTPSource`` construction
    # in try/finally so ``source.close()`` runs even when header parsing,
    # validation, fetch/decode, or orientation/photometric post-processing
    # raises. ``_HTTPSource.close()`` is a no-op today, but a future
    # resource-holding source would leak on the error path without this.
    # ``close()`` is idempotent, so the explicit pre-raise ``source.close()``
    # calls in the validation blocks below stay as-is.
    try:
        header, ifd, geo_info, header_bytes = _parse_cog_http_meta(
            source, overview_level=overview_level)

        # Mirror the local-path orientation guard in ``read_to_array``: a
        # windowed read against a non-default Orientation tag (274) has
        # ambiguous semantics (does the window refer to file pixels or to
        # display pixels?) and the HTTP path does not yet implement
        # ``_apply_orientation``. Reject the combination here so HTTP and
        # local reads agree on the contract for oriented TIFFs instead of
        # silently returning a different region or pixel order. See PR
        # #1680 review feedback on issue #1669.
        if ifd.orientation != 1 and window is not None:
            source.close()
            raise ValueError(
                f"Orientation tag (274) is {ifd.orientation}; windowed reads "
                f"(window=...) and dask-chunked reads (chunks=...) are not "
                f"supported for non-default orientation. Read the full "
                f"array first, then slice."
            )

        # Validate ``window`` against the selected IFD's extent before the
        # tile fetch is built. Without this, the helper silently clamps an
        # out-of-bounds window and returns a smaller array, mismatching
        # ``open_geotiff``'s caller-built coord arrays. Mirrors the
        # local-path validator in ``read_to_array`` (#1634).
        if window is not None:
            w_r0, w_c0, w_r1, w_c1 = window
            if (w_r0 < 0 or w_c0 < 0
                    or w_r1 > ifd.height or w_c1 > ifd.width
                    or w_r0 >= w_r1 or w_c0 >= w_c1):
                source.close()
                raise ValueError(
                    f"window={window} is outside the source extent "
                    f"({ifd.height}x{ifd.width}) or has non-positive size.")

        # Validate ``band`` against the selected IFD's sample count before
        # the tile fetch. Without this, ``band=-1`` silently picks the last
        # channel via numpy negative indexing and ``band>=samples_per_pixel``
        # leaks a raw numpy ``IndexError``; on a single-band file ``band=N``
        # (N != 0) is dropped on the floor because the post-decode slice
        # below is gated on ``arr.ndim == 3 and samples_per_pixel > 1``.
        # Mirrors the local-path validator in ``read_to_array`` so all
        # backends agree on the contract: 0-based non-negative index only.
        # ``source.close()`` is called for symmetry with the success-path
        # teardown below; it is a no-op on ``_HTTPSource`` today (the
        # urllib3 ``PoolManager`` is shared module-level, not per-source)
        # but a future resource-holding source will need it. See issue #1695.
        if band is not None:
            # Reject ``bool`` (and ``np.bool_``) up front; ``isinstance(True, int)``
            # is True in Python so ``True < samples_per_pixel`` evaluates without
            # raising and silently reads band 1. ``np.bool_`` is not a subclass of
            # ``bool`` so it needs its own check to match the VRT path's
            # rejection. See #1786.
            if isinstance(band, (bool, np.bool_)):
                source.close()
                raise ValueError(
                    f"band must be a non-negative int, got {band!r}")
            # Reject non-integer numeric types and anything else that
            # would slip past the bool guard. ``band=0.0`` passes
            # ``0 <= 0.0 < n_bands`` and silently selects band 0 on a
            # single-band file or raises a raw numpy ``IndexError`` from
            # deep in the read path on multi-band files; ``band="0"``
            # fails the comparison with an opaque ``TypeError``. The VRT
            # paths already enforce this; mirror them here. See #1910.
            if not isinstance(band, (int, np.integer)):
                source.close()
                raise TypeError(
                    f"band must be a non-negative int, got {band!r}")
            if ifd.samples_per_pixel <= 1:
                if band != 0:
                    source.close()
                    raise IndexError(
                        f"band={band} requested on a single-band file.")
            elif not 0 <= band < ifd.samples_per_pixel:
                source.close()
                raise IndexError(
                    f"band={band} out of range for "
                    f"{ifd.samples_per_pixel}-band file.")

        arr = _fetch_decode_cog_http_tiles(
            source, header, ifd, max_pixels=max_pixels, window=window)

        # Mirror the local-path band selection in ``read_to_array``: extract
        # the requested band only after the array is materialised so the
        # multi-band tile decode can populate every plane first. ``band``
        # outside the valid range raises ``IndexError`` the same as numpy.
        if arr.ndim == 3 and ifd.samples_per_pixel > 1 and band is not None:
            arr = arr[:, :, band]

        # Apply Orientation tag (274) so HTTP reads return the same pixel
        # order and transform as the local-file path. Only the full-read
        # branch reaches here; the windowed-read branch is rejected above
        # for non-default orientation. See issue #1717.
        if ifd.orientation != 1:
            arr, geo_info = _apply_orientation_with_geo(
                arr, geo_info, ifd.orientation)

        if ifd.photometric == 0 and ifd.samples_per_pixel == 1:
            # Stash the inverted sentinel on geo_info so the caller's
            # sentinel-to-NaN mask runs against the post-MinIsWhite value
            # while ``attrs['nodata']`` keeps the original sentinel for
            # round-trip on write (issue #1809).
            inverted_nodata = _miniswhite_inverted_nodata(
                geo_info.nodata, ifd, arr.dtype)
            geo_info._mask_nodata = inverted_nodata
        arr = _apply_photometric_miniswhite(arr, ifd)
    finally:
        source.close()

    return arr, geo_info


def _fetch_decode_cog_http_strips(
    source: _HTTPSource,
    header: TIFFHeader,
    ifd: IFD,
    dtype: np.dtype,
    bps: int,
    *,
    max_pixels: int = MAX_PIXELS_DEFAULT,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Fetch and decode the strips of a stripped TIFF over HTTP.

    Stripped HTTP companion to :func:`_fetch_decode_cog_http_tiles`. When
    *window* is given, only the strip byte-ranges that intersect the
    window are fetched + decoded; the result is sized to the (clamped)
    window rather than the full image, so a small window read of a
    multi-billion-pixel stripped file does not download the whole
    raster. Adjacent strip ranges are coalesced via
    :meth:`_HTTPSource.read_ranges_coalesced` the same way the tiled
    path does. ``max_pixels`` is applied to the *materialised* pixel
    count (window for windowed reads, full image otherwise) so a small
    caller cap on a tiny window passes a large source the same way the
    tiled branch does (#1823). When *window* is None, the function
    falls back to ``source.read_all()`` and dispatches to
    :func:`_read_strips`; the caller's ``max_pixels`` is threaded
    through so the full-image dim check honours the user's cap.
    See issues #1664 and #1823 for the safety contract this restores.
    """
    width = ifd.width
    height = ifd.height
    samples = ifd.samples_per_pixel
    # Source-IFD dim check (issue #2053). Mirror of the local-path
    # check in ``_read_strips`` so HTTP COG reads of a malformed
    # stripped file fail at the source rather than collapsing to an
    # empty post-clamp window. Tiled paths already get the equivalent
    # check from ``validate_tile_layout``.
    _check_source_dimensions(width, height, samples)
    compression = ifd.compression
    rps = ifd.rows_per_strip
    offsets = ifd.strip_offsets
    byte_counts = ifd.strip_byte_counts
    pred = ifd.predictor
    _validate_predictor_sample_format(pred, ifd.sample_format)
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)
    planar = ifd.planar_config

    if offsets is None or byte_counts is None:
        raise ValueError("Missing strip offsets or byte counts")
    if rps is None or rps <= 0:
        raise ValueError(f"Invalid RowsPerStrip: {rps!r}")

    # Per-strip compressed-byte cap (#1664). A crafted ``StripByteCounts``
    # entry can request an unbounded HTTP Range GET or decompress a few
    # KiB into gigabytes. The cap applies to strips we actually fetch:
    # - Full-image path: validated inside ``_read_strips`` over every
    #   strip (full file is materialised regardless).
    # - Windowed path: validated inside the fetch-range loop below so a
    #   small window only fails on strips it intersects -- mirrors the
    #   tiled HTTP path's per-tile check (#1851).
    max_tile_bytes = _max_tile_bytes_from_env()

    # Full-image read: keep the legacy ``read_all`` + ``_read_strips``
    # path so anything _read_strips already validates (sparse strips,
    # strip-table truncation, LERC masked_fill, per-strip byte cap, etc.)
    # stays in one place. Just thread the caller's ``max_pixels`` through
    # so the dim check uses their cap instead of the default 1B.
    if window is None:
        _check_dimensions(width, height, samples, max_pixels)
        # Bound the HTTP body to the byte size implied by the TIFF strip
        # table. Without this cap, a tiny declared raster (which sails
        # past ``_check_dimensions``) can still pull a multi-gigabyte
        # body off the wire and into memory before ``_read_strips``
        # gets a chance to reject anything. The strip table tells us
        # the maximum legitimate byte offset; anything beyond that is
        # either a malformed file or a hostile server. Issue #2051.
        max_bytes = _compute_full_image_byte_budget(offsets, byte_counts)
        all_data = source.read_all(max_bytes=max_bytes)
        return _read_strips(all_data, ifd, header, dtype,
                            max_pixels=max_pixels)

    # Windowed read: fetch only the strips that intersect the window.
    r0, c0, r1, c1 = window
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(height, r1)
    c1 = min(width, c1)
    out_h = r1 - r0
    out_w = c1 - c0
    _check_dimensions(out_w, out_h, samples, max_pixels)

    strips_per_band = (height + rps - 1) // rps
    if planar == 2 and samples > 1:
        n_strips_expected = strips_per_band * samples
        if (len(offsets) < n_strips_expected
                or len(byte_counts) < n_strips_expected):
            raise ValueError(
                f"Strip table truncated for planar layout "
                f"(PlanarConfiguration=2): expected "
                f"{n_strips_expected} entries "
                f"({strips_per_band} strips x {samples} samples), got "
                f"offsets={len(offsets)}, byte_counts={len(byte_counts)}")
    else:
        n_strips_expected = strips_per_band
        if (len(offsets) < n_strips_expected
                or len(byte_counts) < n_strips_expected):
            raise ValueError(
                f"Strip table truncated: expected "
                f"{n_strips_expected} entries, got "
                f"offsets={len(offsets)}, byte_counts={len(byte_counts)}")

    first_strip = r0 // rps
    last_strip = min((r1 - 1) // rps, strips_per_band - 1)

    # Sparse strips (StripByteCounts == 0) must materialise as nodata or 0,
    # mirroring the local strip path. Detect sparsity over the *whole*
    # strip table so an empty strip outside the window does not change
    # the windowed allocation contract.
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

    # Pass 1: build the list of byte ranges + placements. Skip sparse
    # strips and any strips whose intersected row range is empty.
    band_count = samples if (planar == 2 and samples > 1) else 1
    strip_samples = 1 if band_count > 1 else samples
    fetch_ranges: list[tuple[int, int]] = []
    placements: list[tuple[int, int]] = []
    for band_idx in range(band_count):
        band_offset = band_idx * strips_per_band if band_count > 1 else 0
        for strip_idx in range(first_strip, last_strip + 1):
            global_idx = band_offset + strip_idx
            if global_idx >= len(offsets):
                continue
            bc = byte_counts[global_idx]
            if bc == 0:
                # Sparse strip: result is already pre-filled above.
                continue
            # Per-strip byte cap, scoped to strips the window actually
            # fetches (#1851). Mirrors the per-tile check in
            # ``_fetch_decode_cog_http_tiles`` so a window over a benign
            # strip is not rejected because some unrelated strip in the
            # file exceeds the cap.
            if bc > max_tile_bytes:
                raise ValueError(
                    f"TIFF strip {global_idx} declares "
                    f"StripByteCount={bc:,} bytes, which exceeds the "
                    f"per-strip safety cap of {max_tile_bytes:,} bytes. "
                    f"The file is malformed or attempting denial-of-service. "
                    f"Override via XRSPATIAL_COG_MAX_TILE_BYTES if this file "
                    f"is legitimate."
                )
            fetch_ranges.append((offsets[global_idx], bc))
            placements.append((band_idx, strip_idx))

    # Pass 2: fetch the strip bytes, coalescing adjacent ranges (mirrors
    # the tiled HTTP path; see #1823 / coalescing rationale on line ~2145).
    try:
        workers = max(1, int(
            _os_module.environ.get('XRSPATIAL_COG_HTTP_WORKERS', '8')))
    except ValueError:
        workers = 8
    try:
        gap = int(_os_module.environ.get(
            'XRSPATIAL_COG_COALESCE_GAP',
            str(COALESCE_GAP_THRESHOLD_DEFAULT)))
    except ValueError:
        gap = COALESCE_GAP_THRESHOLD_DEFAULT
    if fetch_ranges:
        strip_bytes_list = source.read_ranges_coalesced(
            fetch_ranges, max_workers=workers, gap_threshold=gap)
    else:
        strip_bytes_list = []

    # Pass 3: decode each strip and place its intersection with the window.
    for (band_idx, strip_idx), strip_data in zip(placements, strip_bytes_list):
        strip_row = strip_idx * rps
        strip_rows = min(rps, height - strip_row)
        if strip_rows <= 0:
            continue

        # Per-strip decoded-dimension cap (#1851). Mirrors the per-tile
        # ``_check_dimensions(tw, th, samples, MAX_PIXELS_DEFAULT)`` in
        # the tiled HTTP path: a tiny window intersecting an oversized
        # strip would otherwise force ``_decode_strip_or_tile`` to
        # allocate ``width * strip_rows * strip_samples`` bytes before
        # the window clip. Use ``MAX_PIXELS_DEFAULT`` rather than the
        # caller's ``max_pixels`` so a small output-window budget does
        # not reject normal strip sizes.
        _check_dimensions(width, strip_rows, strip_samples,
                          MAX_PIXELS_DEFAULT)

        strip_pixels = _decode_strip_or_tile(
            strip_data, compression, width, strip_rows, strip_samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

        src_r0 = max(r0 - strip_row, 0)
        src_r1 = min(r1 - strip_row, strip_rows)
        dst_r0 = max(strip_row - r0, 0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        if dst_r1 <= dst_r0:
            continue

        if band_count > 1:
            # Planar=2 strip holds one band; place into the per-band slot.
            result[dst_r0:dst_r1, :, band_idx] = (
                strip_pixels[src_r0:src_r1, c0:c1])
        else:
            result[dst_r0:dst_r1] = strip_pixels[src_r0:src_r1, c0:c1]

    return result


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
        return _fetch_decode_cog_http_strips(
            source, header, ifd, dtype, bps,
            max_pixels=max_pixels, window=window,
        )

    width = ifd.width
    height = ifd.height
    tw = ifd.tile_width
    th = ifd.tile_height
    samples = ifd.samples_per_pixel
    planar = ifd.planar_config
    compression = ifd.compression
    pred = ifd.predictor
    _validate_predictor_sample_format(pred, ifd.sample_format)
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
    # reads (window is None). The per-tile dim check below guards the
    # TIFF header against absurd ``TileWidth`` / ``TileLength`` values
    # (e.g. 2**31) and uses ``MAX_PIXELS_DEFAULT`` so a caller's small
    # ``max_pixels`` -- intended as an output-window budget -- does not
    # reject normal 256x256 tiles. See #1823.
    if window is None:
        _check_dimensions(width, height, samples, max_pixels)
    _check_dimensions(tw, th, samples, MAX_PIXELS_DEFAULT)

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

    # ``PlanarConfiguration=2`` stores one tile sequence per band,
    # concatenated in TileOffsets. ``tiles_per_band`` selects the right
    # slab when computing ``tile_idx``; ``band_count == 1`` for chunky
    # files keeps the original single-loop fetch behaviour. Mirrors the
    # local ``_read_tiles`` path (#1669).
    band_count = samples if (planar == 2 and samples > 1) else 1
    tiles_per_band = tiles_across * tiles_down
    # Per-tile sample count: planar=2 tiles hold one band each, planar=1
    # tiles interleave ``samples`` components per pixel.
    tile_samples = 1 if band_count > 1 else samples

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
    # is overridable via XRSPATIAL_COG_MAX_TILE_BYTES. The local-mmap path
    # applies the same cap in _read_tiles / _read_strips (issue #1664).
    max_tile_bytes = _max_tile_bytes_from_env()
    fetch_ranges: list[tuple[int, int]] = []
    # Placement record: (band_idx, tr, tc). band_idx is 0 for chunky
    # files; for planar=2 it indicates which sample axis slot the
    # decoded tile fills.
    placements: list[tuple[int, int, int]] = []
    for band_idx in range(band_count):
        band_tile_offset = (band_idx * tiles_per_band
                            if band_count > 1 else 0)
        for tr in range(tile_row_start, tile_row_end):
            for tc in range(tile_col_start, tile_col_end):
                tile_idx = band_tile_offset + tr * tiles_across + tc
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
                placements.append((band_idx, tr, tc))

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
    #
    # Codec decode (deflate, zstd, LZW, ...) releases the GIL inside the
    # C extension, so a thread pool over the per-tile decode actually
    # overlaps codec work across cores. The local-file path in
    # ``_read_tiles`` uses the same pattern with a 64K-pixel threshold to
    # skip the pool-startup cost on small tiles; mirror that gate here so
    # HTTP COG reads of wide windows benefit from the same parallelism
    # rather than serialising the decode after a parallel fetch. The
    # placement loop that copies pixels into ``result`` stays serial to
    # avoid contending writes to the output buffer.
    n_decode_tiles = len(placements)
    decode_in_parallel = (
        n_decode_tiles > 1 and tw * th >= _PARALLEL_DECODE_PIXEL_THRESHOLD)

    def _decode_one(tile_data):
        return _decode_strip_or_tile(
            tile_data, compression, tw, th, tile_samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

    if decode_in_parallel:
        from concurrent.futures import ThreadPoolExecutor
        n_decode_workers = min(n_decode_tiles, _os_module.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_decode_workers) as pool:
            decoded_tiles = list(pool.map(_decode_one, tile_bytes_list))
    else:
        decoded_tiles = [_decode_one(tile_data) for tile_data in tile_bytes_list]

    for (band_idx, tr, tc), tile_pixels in zip(placements, decoded_tiles):
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

        if band_count > 1:
            # Planar=2 tile holds one band; place into the per-band slot
            # of the (out_h, out_w, samples) result. ``tile_pixels`` from
            # ``_decode_strip_or_tile`` with ``samples=1`` is 2D.
            result[dy0:dy1, dx0:dx1, band_idx] = tile_pixels[sy0:sy1, sx0:sx1]
        else:
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


def _apply_orientation_with_geo(
    arr: np.ndarray, geo_info: GeoInfo, orientation: int,
) -> tuple[np.ndarray, GeoInfo]:
    """Apply Orientation tag to ``arr`` and update ``geo_info`` to match.

    Shared helper used by the local-file and HTTP COG paths so both
    return the same pixel order and transform for a given file. See
    issue #1717 for the HTTP-path parity break this consolidates.
    """
    if orientation == 1:
        return arr, geo_info
    # Use the *file* dimensions (before orientation) for the transform
    # math below. After ``_apply_orientation`` the array shape may swap
    # (orientations 5-8), so capture them now.
    file_h = arr.shape[0]
    file_w = arr.shape[1]
    arr = _apply_orientation(arr, orientation)
    t = geo_info.transform
    if not geo_info.has_georef:
        pass
    elif orientation in (2, 3, 4):
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
        # ``has_georef`` is True whenever ModelTransformation,
        # ModelPixelScale, or ModelTiepoint is present, even without a
        # CRS. The pixel-size swap below cannot express the
        # per-orientation origin shift plus rotation these orientations
        # require, so the x/y coords would be wrong whether or not a
        # CRS tag accompanies the transform. Refuse the file in that
        # case rather than warn and return silently wrong coords.
        raise NotImplementedError(
            f"TIFF Orientation {orientation} on a georeferenced file "
            f"requires a per-orientation origin shift plus a rotation "
            f"that the axis-aligned GeoTransform used here cannot "
            f"represent, so the returned x/y coords would be wrong. "
            f"Reproject the file with another tool (e.g. GDAL) or "
            f"strip the Orientation tag before reading. See issue "
            f"#1765."
        )
    return arr, geo_info


def _apply_photometric_miniswhite(arr: np.ndarray, ifd: IFD) -> np.ndarray:
    """Apply TIFF MinIsWhite inversion for single-band grayscale images."""
    if ifd.photometric != 0 or ifd.samples_per_pixel != 1:
        return arr
    if arr.dtype.kind == 'u':
        return np.iinfo(arr.dtype).max - arr
    if arr.dtype.kind == 'f':
        return -arr
    return arr


def _miniswhite_inverted_nodata(nodata, ifd: IFD, dtype: np.dtype):
    """Return the nodata sentinel value after MinIsWhite inversion.

    When the reader applied MinIsWhite (``photometric == 0``,
    ``samples_per_pixel == 1``), the original integer sentinel ``s`` is
    rewritten to ``iinfo(dtype).max - s`` and the float sentinel ``s`` to
    ``-s``.  Downstream nodata-to-NaN masks must compare against the
    inverted sentinel rather than the original, otherwise they flag the
    wrong pixels: inverted real data colliding with the original
    sentinel value is incorrectly masked while the real nodata cells
    keep their inverted-sentinel value (issue #1809).

    Returns the inverted nodata sentinel, or the original ``nodata``
    when MinIsWhite was not applied / not applicable.  Non-finite or
    out-of-range nodata is returned unchanged so callers' downstream
    skip-the-mask logic stays unchanged.
    """
    if nodata is None:
        return nodata
    if ifd.photometric != 0 or ifd.samples_per_pixel != 1:
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


def read_to_array(source, *, window=None, overview_level: int | None = None,
                  band: int | None = None,
                  max_pixels: int = MAX_PIXELS_DEFAULT,
                  max_cloud_bytes=_MAX_CLOUD_BYTES_SENTINEL,
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
    max_cloud_bytes : int or None, optional
        Byte ceiling for eager reads from fsspec sources (``s3://``,
        ``gs://``, ``az://``, ``abfs://``, ``memory://``, ...). The
        compressed object size is checked against this budget before any
        bytes are downloaded. Default is :data:`MAX_CLOUD_BYTES_DEFAULT`
        (256 MiB), overridable via the
        ``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES`` env var. Pass ``None`` to
        skip the check entirely (pre-#1928 behaviour). The HTTP path
        already reads only what it needs via range requests and is not
        subject to this limit. See issue #1928.

    Returns
    -------
    (np.ndarray, GeoInfo) tuple
    """
    source = _coerce_path(source)
    if isinstance(source, str) and source.startswith(('http://', 'https://')):
        return _read_cog_http(source, overview_level=overview_level, band=band,
                              max_pixels=max_pixels, window=window)

    # Local file, cloud storage, or file-like buffer: read all bytes then parse
    if _is_file_like(source):
        src = _BytesIOSource(source)
    elif _is_fsspec_uri(source):
        src = _CloudSource(source)
        # Check the compressed object size before any bytes are
        # downloaded. ``_CloudSource.__init__`` already fetched the size
        # via ``fsspec.size()``, so this is free. See issue #1928.
        cloud_budget = _resolve_max_cloud_bytes(max_cloud_bytes)
        if cloud_budget is not None:
            size = src.size
            if size is None:
                src.close()
                raise CloudSizeLimitError(
                    f"Cloud source {source!r} reports unknown size; "
                    f"refusing to download to avoid an unbounded read. "
                    f"Pass max_cloud_bytes=None to disable the size "
                    f"check for this source. Raising the byte limit "
                    f"does not help when the source size is unknown.")
            if size > cloud_budget:
                src.close()
                raise CloudSizeLimitError(
                    f"Cloud source {source!r} is {size:,} bytes, which "
                    f"exceeds max_cloud_bytes={cloud_budget:,}. Eager "
                    f"reads pull the full object before any TIFF header "
                    f"parse; raise max_cloud_bytes (or set "
                    f"XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES) if the file is "
                    f"legitimate, pass max_cloud_bytes=None to disable "
                    f"the check, or use chunks=... for a windowed dask "
                    f"read.")
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
        # Inherit georef from level 0 when an overview IFD lacks its own
        # geokeys (issue #1640). For overview_level=0 (or None) this is a
        # no-op: the helper short-circuits when the IFD is not a
        # NewSubfileType=overview entry.
        geo_info = extract_geo_info_with_overview_inheritance(
            ifd, ifds, data, header.byte_order)

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

        # Validate ``window`` against the selected IFD's extent. Without
        # this, ``_read_tiles`` / ``_read_strips`` silently clamp an
        # out-of-bounds window and return a smaller array, which then
        # mismatches caller-built coord arrays in ``open_geotiff`` and
        # surfaces as an opaque ``CoordinateValidationError``. Raising
        # here matches the dask path's pre-flight validator (see
        # ``read_geotiff_dask`` in ``__init__.py``) so all backends
        # agree on the contract. Reuses the IFD already parsed above,
        # so callers pay no extra metadata-parse cost (file-like
        # sources are read once instead of twice). See issue #1634.
        if window is not None:
            w_r0, w_c0, w_r1, w_c1 = window
            if (w_r0 < 0 or w_c0 < 0
                    or w_r1 > ifd.height or w_c1 > ifd.width
                    or w_r0 >= w_r1 or w_c0 >= w_c1):
                raise ValueError(
                    f"window={window} is outside the source extent "
                    f"({ifd.height}x{ifd.width}) or has non-positive size.")

        # Validate ``band`` against the selected IFD's sample count.
        # Without this, ``band=-1`` silently selects the last channel
        # via numpy negative indexing and ``band>=samples_per_pixel``
        # leaks a raw numpy ``IndexError`` with the internal slice
        # shape. Mirrors the dask path's pre-flight validator (see
        # ``read_geotiff_dask`` in ``__init__.py``), the GPU path, and
        # the HTTP path (``_read_cog_http`` above, as of issue #1695)
        # so all backends agree on the contract: 0-based non-negative
        # index only. See issue #1673.
        ifd_samples = ifd.samples_per_pixel
        if band is not None:
            # Reject ``bool`` and ``np.bool_`` before the range check.
            # ``isinstance(True, int)`` is True in Python and
            # ``True < ifd_samples`` evaluates as ``1``, so without this
            # guard ``band=True`` silently reads band 1 and ``band=False``
            # reads band 0. ``np.bool_`` is not a subclass of ``bool`` so it
            # needs its own check to match the VRT path's existing
            # rejection. See #1786.
            if isinstance(band, (bool, np.bool_)):
                raise ValueError(
                    f"band must be a non-negative int, got {band!r}")
            # Reject non-integer numeric types and anything else that
            # would slip past the bool guard. ``band=0.0`` passes
            # ``0 <= 0.0 < n_bands`` and silently selects band 0 on a
            # single-band file or raises a raw numpy ``IndexError`` from
            # deep in the read path on multi-band files. The VRT paths
            # already enforce this; mirror them here. See #1910.
            if not isinstance(band, (int, np.integer)):
                raise TypeError(
                    f"band must be a non-negative int, got {band!r}")
            if ifd_samples <= 1:
                if band != 0:
                    raise IndexError(
                        f"band={band} requested on a single-band file.")
            elif not 0 <= band < ifd_samples:
                raise IndexError(
                    f"band={band} out of range for {ifd_samples}-band file.")

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
            arr, geo_info = _apply_orientation_with_geo(
                arr, geo_info, orientation)

        if ifd.photometric == 0 and ifd.samples_per_pixel == 1:
            # The MinIsWhite inversion rewrites the original sentinel
            # value, so any downstream nodata-to-NaN mask must compare
            # against the inverted sentinel instead.  Stash the inverted
            # sentinel on geo_info as a private attribute so callers can
            # apply the mask post-inversion while keeping the original
            # sentinel on ``geo_info.nodata`` for the attrs round-trip
            # (issue #1809).
            inverted_nodata = _miniswhite_inverted_nodata(
                geo_info.nodata, ifd, arr.dtype)
            arr = _apply_photometric_miniswhite(arr, ifd)
            geo_info._mask_nodata = inverted_nodata
    finally:
        src.close()

    return arr, geo_info
