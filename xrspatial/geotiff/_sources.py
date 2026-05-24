"""Data source abstractions for the GeoTIFF reader.

This module is private to :mod:`xrspatial.geotiff`. It owns the
read-side I/O layer: mmap-backed local files, urllib3-backed HTTP with
SSRF defences and DNS-rebind pinning, fsspec cloud paths, and an
in-memory BytesIO adapter. The decode pipeline in ``_reader`` consumes
these source objects through a small interface:

* ``read_range(start, length) -> bytes`` -- implemented by every source.
* ``read_all() -> bytes | mmap`` -- implemented by every source.
* ``size`` property -- implemented by every source.
* ``close()`` -- implemented by every source.
* ``read_ranges(ranges, max_workers=...) -> list[bytes]`` -- HTTP and
  cloud only. ``_FileSource`` and ``_BytesIOSource`` do not implement
  this; callers that want parallel range fetches must dispatch on the
  source type or fall back to looping ``read_range``.
* ``read_ranges_coalesced(ranges, max_workers=..., gap_threshold=...)
  -> list[bytes]`` -- HTTP and cloud only, same caveat.

The module also contains the byte-range coalescing helpers
(:func:`coalesce_ranges`, :func:`split_coalesced_bytes`) used by the
HTTP and cloud sources to merge nearby tile fetches into fewer GETs.

History: extracted from ``_reader.py`` in PR-E of the GeoTIFF refactor
epic (issue #2228). Behaviour-preserving move; no public API change.
"""
from __future__ import annotations

import mmap
import os as _os_module
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import urllib3

# ---------------------------------------------------------------------------
# Cloud byte budget (eager fsspec reads)
# ---------------------------------------------------------------------------

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


class CloudSizeLimitError(ValueError):
    """Raised when an eager fsspec read would exceed ``max_cloud_bytes``.

    Distinct from :class:`xrspatial.geotiff._reader.PixelSafetyLimitError`
    because the cloud check runs against the compressed object size before
    any TIFF header parse, so the pixel-count message would be misleading.
    """


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
# Local mmap-backed source
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


# Module-level cache shared across all reads. This is a *singleton*: there
# is one shared mmap cache per Python process so concurrent ``_FileSource``
# instances for the same path share a single mmap and a single refcount.
# ``_reader.py`` re-exports this exact name so legacy access via
# ``xrspatial.geotiff._reader._mmap_cache.clear()`` (e.g. in
# test_vrt_lazy_chunks_1814) still reaches the same instance. A future
# refactor that adds another ``_MmapCache()`` here, or moves this object
# to a third module, must keep the single-instance contract or
# ``_FileSource`` will leak mmaps between cache instances.
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


# ---------------------------------------------------------------------------
# HTTP source: pool, timeouts, SSRF defences (issue #1664)
# ---------------------------------------------------------------------------


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

#: Default upper bound (bytes) on any single coalesced range. The gap
#: threshold alone does not bound the *total* over-fetch: a tile table
#: with N entries whose offsets are spaced just under ``gap_threshold``
#: apart will chain into one merged range of size ~N * gap_threshold,
#: even when each individual tile is tiny and passes the per-tile cap.
#: This cap seals the current merged range and starts a new one once
#: extending it would exceed the limit. Override via the
#: ``XRSPATIAL_COG_MAX_COALESCED_RANGE_BYTES`` environment variable.
#: Issue #2266.
MAX_COALESCED_RANGE_BYTES_DEFAULT = MAX_TILE_BYTES_DEFAULT  # 256 MiB


def _max_coalesced_range_bytes_from_env() -> int:
    """Read the coalesced-range cap from the environment, or use the default.

    Non-integer, empty, zero, or negative values all fall back to
    ``MAX_COALESCED_RANGE_BYTES_DEFAULT``. Mirrors the policy used by
    :func:`_max_tile_bytes_from_env` so callers can not accidentally set
    an unreachable 1-byte cap.
    """
    raw = _os_module.environ.get('XRSPATIAL_COG_MAX_COALESCED_RANGE_BYTES')
    if raw is None:
        return MAX_COALESCED_RANGE_BYTES_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return MAX_COALESCED_RANGE_BYTES_DEFAULT
    return val if val > 0 else MAX_COALESCED_RANGE_BYTES_DEFAULT


def coalesce_ranges(
    ranges: list[tuple[int, int]],
    gap_threshold: int = COALESCE_GAP_THRESHOLD_DEFAULT,
    max_coalesced_range_bytes: int | None = None,
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
    max_coalesced_range_bytes : int or None, optional
        Upper bound on any single merged range. When extending the
        current merged range would push its length above this cap, the
        current range is sealed and a new one is started instead. This
        bounds the *total* over-fetch even when many small ranges are
        spaced just under ``gap_threshold`` apart. ``None`` (the
        default) reads the cap from
        ``XRSPATIAL_COG_MAX_COALESCED_RANGE_BYTES`` (falling back to
        :data:`MAX_COALESCED_RANGE_BYTES_DEFAULT`, 256 MiB). A
        non-positive value disables the cap. Issue #2266.

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
    merging entirely (every input becomes its own merged range). When a
    single input range already exceeds ``max_coalesced_range_bytes`` it
    is still emitted intact -- the per-tile cap in
    :func:`_max_tile_bytes_from_env` is the right place to reject
    oversized individual tiles; this cap only governs how greedily
    *separate* tiles are stitched together.
    """
    if not ranges:
        return [], []

    if max_coalesced_range_bytes is None:
        max_coalesced_range_bytes = _max_coalesced_range_bytes_from_env()

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
        # Gaps may be negative if a later-listed range overlaps an
        # earlier one; clamp ``new_end`` so the merged length covers
        # both. ``candidate_length`` is the length the merged range
        # would have if we extended it to include this input. We use
        # it both to decide whether the merge is allowed under the
        # size cap and (when it is) to update ``cur_length``.
        new_end = max(cur_end, off + length)
        candidate_length = new_end - cur_start
        size_ok = (
            max_coalesced_range_bytes <= 0
            or candidate_length <= max_coalesced_range_bytes
        )
        if gap_threshold >= 0 and gap <= gap_threshold and size_ok:
            # Extend current merged range.
            cur_length = candidate_length
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
    from urllib3.exceptions import ConnectTimeoutError, NameResolutionError, NewConnectionError

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
                # fresh connections every time. Drain first so urllib3
                # can return the connection to the pool instead of
                # closing it; a 3xx body is bounded by Content-Length so
                # the drain is cheap.
                if not preload_content:
                    try:
                        resp.drain_conn()
                    except Exception:  # noqa: BLE001
                        pass
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

    #: Absolute ceiling for a ``read_range(0, length)`` whose server
    #: ignores ``Range`` and returns a 200 with the full object body. Up
    #: to this size we tolerate the misbehaviour (the response is sliced
    #: down to ``length``); beyond it we refuse to buffer. The value is
    #: large enough to cover a typical sidecar/header prefetch served
    #: from a Range-blind origin (kilobytes to a few MiB) but well below
    #: the multi-gigabyte regime the OOM guard exists to prevent.
    #:
    #: This is the worst-case *transient* buffer per ``read_range``
    #: call. :meth:`read_ranges` fans out across up to 8 threads, so a
    #: full pool against a Range-blind server can hold roughly
    #: ``8 * _RANGE_IGNORED_FULL_OBJECT_CAP`` (~128 MiB) of body bytes
    #: in flight at once. That is still bounded -- the pre-#2264 path
    #: had no per-call bound at all -- but worth keeping in mind if a
    #: future caller wants to scale ``max_workers`` higher. Issue #2264.
    _RANGE_IGNORED_FULL_OBJECT_CAP = 16 * 1024 * 1024

    def read_range(self, start: int, length: int) -> bytes:
        """Fetch ``[start, start + length)`` from the remote object.

        Sends ``Range: bytes={start}-{start + length - 1}`` and validates
        the response on three axes before returning:

        - status is 200 or 206 (anything else raises ``OSError``);
        - the advertised ``Content-Length`` (or the streamed body, if no
          header) fits inside the per-call byte budget -- ``length``
          bytes for a 206 or non-zero start, or
          :attr:`_RANGE_IGNORED_FULL_OBJECT_CAP` for the ``start=0`` +
          200-with-no-Content-Range fallback path;
        - the ``Content-Range`` header (when present) actually starts at
          ``start``.

        The body is streamed with ``preload_content=False`` so an
        oversize response cannot land in ``resp.data`` before the cap is
        applied; :meth:`_read_capped` aborts past the byte budget.

        Raises ``OSError`` when the server's body is past the budget,
        the status code is wrong, or the ``Content-Range`` does not line
        up with what was requested. Issue #2264.
        """
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
        # Stream the body so a server that ignores ``Range`` and returns
        # the full object as 200 cannot drag arbitrarily many bytes into
        # process memory before the cap is applied. Without
        # ``preload_content=False`` urllib3 buffers the full response into
        # ``resp.data`` *before* this method returns, so the slice in
        # ``_validate_range_response`` only ever ran after the body was
        # already resident. Mirrors the ``read_all`` streaming path that
        # #2051 introduced. Issue #2264.
        resp = self._request(headers=headers, preload_content=False)
        try:
            content_range = resp.headers.get('Content-Range')
            content_length = resp.headers.get('Content-Length')
            status = resp.status
            # Choose the byte budget for ``_read_capped``. A 206 with
            # Content-Range is bounded by ``length`` (the validator
            # below also re-checks this against the advertised range).
            # A 200 without Content-Range means the server ignored
            # ``Range``; allow the full-object slack only when the
            # caller asked from ``start=0`` (the only case where the
            # full body offsets line up with what was requested).
            if content_range is None and start == 0:
                cap = self._RANGE_IGNORED_FULL_OBJECT_CAP
            else:
                cap = length
            self._check_range_content_length(content_length, cap)
            data = self._read_capped(resp, cap)
            # Validate inside the try so we hold the response alive while
            # status/headers are inspected. ``release_conn`` in the
            # finally happens after the validator returns its bytes (or
            # raises).
            return self._validate_range_response(
                status=status,
                content_range=content_range,
                data=data,
                start=start,
                length=length,
            )
        finally:
            try:
                resp.release_conn()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _check_range_content_length(raw, cap: int) -> None:
        """Reject a range response whose advertised ``Content-Length``
        exceeds the byte budget for the call.

        For a ``Range: bytes=S-E`` request, a well-behaved server returns
        either a 206 with body length ``E - S + 1`` or a 200 with the
        full object body. A server that ignores ``Range`` and returns a
        multi-gigabyte 200 advertises that size up front; rejecting on
        the header lets us bail before any body bytes are read.

        Missing or unparseable ``Content-Length`` returns silently --
        the streaming cap in :meth:`_read_capped` is the real defence
        and will catch an over-sized body whether the header was honest,
        dishonest, or absent. Issue #2264.
        """
        if raw is None:
            return
        try:
            declared = int(raw)
        except (TypeError, ValueError):
            return
        if declared > cap:
            raise OSError(
                f"HTTP range response declares Content-Length="
                f"{declared:,} bytes, which exceeds the byte budget of "
                f"{cap:,} bytes for this request. The server likely "
                f"ignored the Range header and returned a multi-MiB "
                f"body. Issue #2264."
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
            # the buffered body down. ``read_range`` has already capped
            # the buffer at :attr:`_RANGE_IGNORED_FULL_OBJECT_CAP` (16
            # MiB) via the streaming preflight, so ``data`` arriving here
            # is bounded even when the server lied about the body size;
            # the slice below just enforces the caller's contract.
            # Issue #2264.
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
        max_coalesced_range_bytes: int | None = None,
    ) -> list[bytes]:
        """Fetch *ranges* using merged GETs where adjacent ranges allow it.

        Wrapper around :meth:`read_ranges` that first calls
        :func:`coalesce_ranges` to group nearby ranges into fewer larger
        GETs, then splits the responses back per-input via
        :func:`split_coalesced_bytes`. Returns bytes in input order, same
        as :meth:`read_ranges`.

        Setting *gap_threshold* to a negative number disables merging
        and falls back to one GET per input range.

        *max_coalesced_range_bytes* caps the size of any single merged
        GET. ``None`` (the default) reads the cap from
        ``XRSPATIAL_COG_MAX_COALESCED_RANGE_BYTES`` and otherwise uses
        :data:`MAX_COALESCED_RANGE_BYTES_DEFAULT`. See
        :func:`coalesce_ranges` for details. Issue #2266.
        """
        if not ranges:
            return []
        merged, mapping = coalesce_ranges(
            ranges,
            gap_threshold=gap_threshold,
            max_coalesced_range_bytes=max_coalesced_range_bytes,
        )
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
        :func:`xrspatial.geotiff._reader._check_dimensions` can still be
        served as a multi-gigabyte HTTP body and the whole body is
        allocated before TIFF parsing gets a chance to reject it.
        Issue #2051.

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

        Missing or unparseable ``Content-Length`` returns silently --
        the streaming cap in :meth:`_read_capped_urllib3` /
        :meth:`_read_capped_stdlib` is the real defence and will catch
        an over-sized body whether the header was honest, dishonest, or
        absent.
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
                    f"about Content-Length. Issues #2051, #2264."
                )
        return b''.join(chunks)

    @property
    def size(self) -> int | None:
        return self._size

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Source dispatch helpers
# ---------------------------------------------------------------------------

_CLOUD_SCHEMES = ('s3://', 'gs://', 'az://', 'abfs://')


def _is_http_source(source) -> bool:
    """Return True if ``source`` is an HTTP(S) URL, case-insensitively.

    Centralized so every routing call site in ``xrspatial/geotiff/``
    classifies the scheme the same way. Before this helper existed,
    each call site did ``source.startswith(('http://', 'https://'))``,
    which is case-sensitive and let ``HTTP://example.internal/...``
    (uppercase) slip past :class:`_HTTPSource` and the SSRF allow-list
    in :func:`_validate_http_url`. Per RFC 3986 section 3.1 URI schemes
    are case-insensitive, so any uppercase / mixed-case variant has to
    route through the same validator as ``http`` / ``https``.

    Non-string inputs (``None``, ``bytes``, ``os.PathLike``, file-like
    objects) return ``False`` so callers can drop the surrounding
    ``isinstance(_, str)`` check where they want to. Issues #2323 / #2332.
    """
    if not isinstance(source, str) or not source:
        return False
    # ``urlparse`` strips off the scheme cleanly even for unusual inputs
    # (e.g. ``HTTP:`` with no ``//``) and avoids the prefix-tuple trap.
    return urlparse(source).scheme.lower() in ('http', 'https')


# Back-compat alias: earlier patches (#2323) shipped this same helper under
# the name ``_is_http_url`` and downstream tests / re-exports still use that
# name. Keep the alias so importers and the regression tests stay green.
_is_http_url = _is_http_source


def _is_fsspec_uri(path: str) -> bool:
    """Check if a path is a fsspec-compatible URI (not http/https/local).

    Excludes http(s) case-insensitively so uppercase URLs cannot dodge the
    SSRF allow-list and pinned DNS in :class:`_HTTPSource` (issues #2323 /
    #2332).
    """
    if not isinstance(path, str):
        return False
    if _is_http_source(path):
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
        max_coalesced_range_bytes: int | None = None,
    ) -> list[bytes]:
        """Fetch *ranges* using merged GETs where adjacent ranges allow it.

        Mirrors :meth:`_HTTPSource.read_ranges_coalesced` so the tiled
        COG decode path can coalesce neighbouring tiles when reading
        from object storage. ``max_coalesced_range_bytes`` caps the
        size of any single merged GET; see :func:`coalesce_ranges`.
        Issue #2266.
        """
        if not ranges:
            return []
        merged, mapping = coalesce_ranges(
            ranges,
            gap_threshold=gap_threshold,
            max_coalesced_range_bytes=max_coalesced_range_bytes,
        )
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
    if _is_http_source(source):
        return _HTTPSource(source)
    if _is_fsspec_uri(source):
        return _CloudSource(source)
    return _FileSource(source)
