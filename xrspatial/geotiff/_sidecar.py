"""External `.tif.ovr` sidecar discovery for overview reading.

GDAL and rasterio write overview pyramids to a sibling ``<path>.ovr``
file by default (`gdaladdo -ro` and `rasterio` `--overview-resampling`
when the source itself is not a COG). The sidecar is a standalone TIFF
whose IFDs are the continuation of the base file's pyramid: base IFD 0
is overview level 0, sidecar IFD 0 is level 1, sidecar IFD 1 is level
2, and so on.

This module discovers the sidecar next to a local-file, HTTP, or
fsspec source and parses its IFDs, returning the inputs the reader
needs to switch to the sidecar's bytes/header when the requested
overview level lives there. Discovery is gated on a cheap existence
check (local stat, HTTP HEAD, or fsspec ``exists``); a missing sidecar
returns ``None`` and the caller falls back to base-file-only behaviour.
"""
from __future__ import annotations

import mmap
import os
from typing import NamedTuple, Union

from ._header import IFD, TIFFHeader, parse_all_ifds, parse_header
# Single source of truth for the fsspec gate; mirroring it here drifts
# silently the moment ``_reader._is_fsspec_uri`` learns a new scheme.
# ``_reader`` imports ``_sidecar`` lazily (inside functions), so this
# top-level import does not form a cycle at module load time.
from ._reader import _is_fsspec_uri
from ._sources import _is_http_source

#: Type of the bytes-like buffer a sidecar carries: an mmap for local
#: files, bytes for HTTP / fsspec downloads. Narrowed from ``object``
#: so the reader sees the actual variants when slicing IFD data.
SidecarBuffer = Union[mmap.mmap, bytes]


class SidecarOverviews(NamedTuple):
    """Bytes, header, and IFDs from a sibling ``.tif.ovr`` sidecar."""

    data: SidecarBuffer
    header: TIFFHeader
    ifds: list[IFD]
    path: str


def _is_http_url(source: str) -> bool:
    """Case-insensitive HTTP(S) scheme test for sidecar routing.

    Delegates to :func:`xrspatial.geotiff._sources._is_http_source` so
    the SSRF-relevant routing decision matches the rest of the package.
    """
    return _is_http_source(source)


def find_sidecar(source) -> str | None:
    """Return the path / URL of a sibling ``.ovr`` sidecar if one exists.

    Scopes:

    * Local file paths -- probe with :func:`os.path.isfile`.
    * HTTP / HTTPS URLs -- issue a 1-byte Range GET to ``<url>.ovr``
      through the eager reader's ``_HTTPSource`` so the probe inherits
      SSRF validation, IP pinning, the shared urllib3 pool, and manual
      redirect re-validation. See :func:`_probe_http`.
    * fsspec URIs (``s3://``, ``gs://``, ``az://``, ``memory://`` ...)
      -- call ``fsspec.AbstractFileSystem.exists`` on ``<uri>.ovr``.
    * File-like buffers (``io.BytesIO``, etc.) -- no sidecar concept,
      return ``None``.

    Discovery failures are silent: any network error, missing fsspec,
    or unreadable path returns ``None`` so the caller falls back to
    base-file-only behaviour. The existence check is bounded and does
    not download or open the sidecar itself; :func:`load_sidecar`
    handles the actual read once the path is known.
    """
    if not isinstance(source, str):
        return None
    candidate = source + ".ovr"
    if "://" not in source:
        return candidate if os.path.isfile(candidate) else None
    if _is_http_url(source):
        return _probe_http(candidate)
    if _is_fsspec_uri(source):
        return _probe_fsspec(candidate)
    return None


def _probe_http(url: str) -> str | None:
    """Return ``url`` if an existence probe succeeds via ``_HTTPSource``.

    Routes through :class:`xrspatial.geotiff._reader._HTTPSource` so the
    probe inherits the same defences the eager reader applies to every
    HTTP fetch:

    * SSRF allow-list via :func:`_validate_http_url` -- loopback /
      link-local / private hostnames are rejected unless
      ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` is set.
    * IP pinning so a DNS-rebind between probe and download cannot
      flip the target.
    * Shared urllib3 ``PoolManager`` with retries and connection reuse,
      so opening many sidecar-bearing files in a row keeps TCP/TLS
      state warm instead of re-handshaking per probe.
    * ``redirect=False`` with manual re-validation of every ``Location``
      header, so a 302 from a public host to a private one is rejected
      the same way the eager reader rejects it.

    Existence is probed with a 1-byte Range GET (``bytes=0-0``) rather
    than HEAD because HEAD support is inconsistent across CDNs and
    object stores; a 1-byte GET works everywhere Range works (which is
    the same precondition the eager COG reader assumes) and the
    one-byte payload is cheaper than the headers urllib3 already
    fetches for either method.

    Any failure -- ``UnsafeURLError``, 4xx/5xx, network timeout,
    missing dependency -- returns ``None`` so the caller falls back
    to base-only behaviour without raising.
    """
    try:
        from ._reader import _HTTPSource
        src = _HTTPSource(url)
        src.read_range(0, 1)
        return url
    except Exception:
        return None


def _probe_fsspec(uri: str) -> str | None:
    """Return ``uri`` if fsspec reports the object exists."""
    try:
        import fsspec
        fs, path = fsspec.core.url_to_fs(uri)
        return uri if fs.exists(path) else None
    except Exception:
        return None


def load_sidecar(path: str,
                 *,
                 max_cloud_bytes: int | None = None,
                 ) -> SidecarOverviews:
    """Open and parse a sidecar ``.ovr`` file.

    Accepts local file paths, HTTP / HTTPS URLs, and fsspec URIs.
    Local paths are mmap'd; remote sources are downloaded once via the
    matching transport (HTTP via :mod:`urllib`, fsspec URIs via the
    fsspec filesystem). The IFD list is the sidecar's full IFD chain
    in file order; the reader treats them as overview levels (the
    first sidecar IFD is level 1 when the base file holds only a
    full-resolution IFD, level 2 when the base file already carries
    one internal overview, and so on).

    Parameters
    ----------
    path
        Sidecar path or URL returned by :func:`find_sidecar`.
    max_cloud_bytes
        Byte ceiling applied to HTTP and fsspec downloads. Mirrors the
        base-file ``max_cloud_bytes`` budget that ``read_to_array`` and
        ``_CloudSource`` enforce so a hostile or malformed sidecar can
        not bypass the cap a caller already set on the source. ``None``
        (the default) means unbounded -- matches the base-file semantics
        when the caller passes ``max_cloud_bytes=None`` explicitly.
        Ignored on the local-file path because mmap does not allocate
        the file.

    The returned ``data`` is either an ``mmap`` (local) or ``bytes``
    (remote). Callers should close the mmap variant via
    ``data.close()`` when present; the bytes case is no-op.

    Raises
    ------
    CloudSizeLimitError
        If the sidecar's size exceeds ``max_cloud_bytes`` on either
        transport. fsspec checks the declared size up front via
        ``fs.size()``; HTTP catches the ``OSError`` that
        :meth:`_HTTPSource.read_all` raises from its
        ``Content-Length`` pre-check or streaming overshoot detector
        and re-raises it as ``CloudSizeLimitError`` so callers see a
        single exception type for "sidecar too big". Non-budget HTTP
        failures (connection reset, DNS error, etc.) pass through as
        ``OSError`` unchanged.
    """
    if "://" not in path:
        f = open(path, "rb")
        try:
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        finally:
            f.close()
    elif _is_http_url(path):
        # Reuse the eager reader's HTTP source so the sidecar download
        # inherits SSRF validation, IP pinning, the shared urllib3
        # PoolManager, and manual redirect re-validation. See
        # ``_probe_http`` for the threat model the indirection closes.
        # ``max_bytes`` here closes a separate gap: without it, the
        # sidecar fetch ignores the ``max_cloud_bytes`` budget the
        # caller set on the base file and a hostile server can serve a
        # multi-GB ``.ovr`` to OOM the process.
        from ._reader import CloudSizeLimitError, _HTTPSource
        try:
            data = _HTTPSource(path).read_all(max_bytes=max_cloud_bytes)
        except OSError as e:
            # ``_HTTPSource.read_all(max_bytes=...)`` raises ``OSError``
            # with "byte budget" in the message for both the
            # Content-Length pre-check and the streaming overshoot probe
            # (see ``_HTTPSource._check_content_length`` and
            # ``_read_capped``). Translate to ``CloudSizeLimitError`` so
            # the HTTP and fsspec branches surface the same exception
            # type for the same failure mode. Other ``OSError``
            # subtypes (connection reset, DNS, etc.) pass through
            # untouched.
            if max_cloud_bytes is not None and "byte budget" in str(e):
                raise CloudSizeLimitError(
                    f"Sidecar {path!r} exceeds "
                    f"max_cloud_bytes={max_cloud_bytes:,}. Raise "
                    f"max_cloud_bytes (or set "
                    f"XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES) if the sidecar "
                    f"is legitimate, or pass max_cloud_bytes=None on the "
                    f"source to disable the check."
                ) from e
            raise
    else:
        # fsspec URI. Stat the sidecar first so an oversized object is
        # rejected before any bytes hit memory, mirroring the
        # ``_CloudSource`` size guard.
        import fsspec
        fs, fs_path = fsspec.core.url_to_fs(path)
        if max_cloud_bytes is not None:
            size = fs.size(fs_path)
            from ._reader import CloudSizeLimitError
            if size is None:
                raise CloudSizeLimitError(
                    f"Sidecar {path!r} reports unknown size; refusing "
                    f"to download to avoid an unbounded read. Pass "
                    f"max_cloud_bytes=None on the source to disable the "
                    f"check for this sidecar."
                )
            if size > max_cloud_bytes:
                raise CloudSizeLimitError(
                    f"Sidecar {path!r} is {size:,} bytes, which exceeds "
                    f"max_cloud_bytes={max_cloud_bytes:,}. Raise "
                    f"max_cloud_bytes (or set "
                    f"XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES) if the sidecar "
                    f"is legitimate, or pass max_cloud_bytes=None on the "
                    f"source to disable the check."
                )
        with fs.open(fs_path, "rb") as f:
            data = f.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    return SidecarOverviews(data=data, header=header, ifds=ifds, path=path)


def close_sidecar(sidecar: SidecarOverviews | None) -> None:
    """Close the sidecar's data buffer if it holds a ``close`` method.

    Mmap data buffers need explicit close; ``bytes`` from a remote
    download does not. Safe to call with ``None``.
    """
    if sidecar is None:
        return
    closer = getattr(sidecar.data, "close", None)
    if closer is None:
        return
    try:
        closer()
    except Exception:
        pass


def attach_sidecar_origin(
    ifds: list[IFD],
    data: SidecarBuffer,
    header: TIFFHeader,
) -> dict[int, tuple[SidecarBuffer, TIFFHeader]]:
    """Return a mapping from ``id(ifd)`` to its ``(data, header)`` origin.

    The reader looks up the selected IFD in this mapping and slices
    the corresponding buffer for strip/tile reads. IFDs not in the
    mapping fall through to the base-file ``data`` / ``header`` the
    caller already has in scope.

    Returns a fresh dict per call rather than mutating the parsed IFD
    instances. Mutation via ``object.__setattr__`` would persist on the
    IFD across requests if a future caller cached or reused the parsed
    pyramid list. A side dict is request-scoped and discarded with the
    surrounding ``read_to_array`` call frame.
    """
    return {id(ifd): (data, header) for ifd in ifds}


def discover_remote_sidecar(
    source,
    base_ifds: list[IFD],
    *,
    overview_level: int | None = None,
    max_cloud_bytes: int | None = None,
) -> tuple[list[IFD], 'SidecarOverviews | None', set[int]]:
    """Probe for a sibling ``.ovr`` and return a merged IFD list.

    Wraps :func:`find_sidecar` and :func:`load_sidecar` so the HTTP /
    fsspec dask metadata path, the eager HTTP path, and
    ``_read_geo_info``'s fsspec branch can resolve ``overview_level``
    against the same merged pyramid the eager local/fsspec reader
    already does.

    Returns:

    * ``merged_ifds`` -- the base IFD list with sidecar IFDs appended
      (or unchanged when no sidecar is present).
    * ``sidecar`` -- the :class:`SidecarOverviews` instance the caller
      must close, or ``None``.
    * ``sidecar_ifd_ids`` -- a set of ``id()`` values for the sidecar
      IFDs. Callers route per-chunk reads to ``sidecar.path`` when the
      selected IFD's ``id()`` is in this set; non-sidecar IFDs fall
      through to the original source.

    ``overview_level`` is the caller's requested level. It is threaded
    into :func:`handle_sidecar_parse_failure` so a download/parse
    failure follows the same policy the eager local and fsspec readers
    use: an explicit level the base file alone cannot serve raises with
    the real parse error chained, anything else warns and falls back.

    Discovery is silent only for the probe itself: probes that fail
    (404, missing fsspec, timeout, etc.) return
    ``([...base_ifds], None, set())``. A successful probe whose download
    or parse subsequently fails -- for example a hostile server that
    returns 200 to every URL but the bytes are not a valid TIFF, or a
    transient network error during download -- is routed through
    :func:`handle_sidecar_parse_failure` so it warns (and falls back to
    base-only) instead of being swallowed, matching the eager
    local/fsspec readers. A misbehaving server still cannot poison an
    otherwise valid base read unless the caller pinned an
    ``overview_level`` only the sidecar could serve.

    Raises
    ------
    CloudSizeLimitError
        Re-raised verbatim when the sidecar download breaches
        ``max_cloud_bytes`` (the one exception that survives the
        otherwise warn-and-fall-back path). The budget is a caller-set
        contract; surfacing the breach lets the caller distinguish "no
        sidecar" from "sidecar too big" rather than silently dropping
        the level and reading a stale base-only pyramid. Set
        ``max_cloud_bytes=None`` to disable the budget.
    ValueError
        Raised by :func:`handle_sidecar_parse_failure` (chained from the
        underlying parse error) when ``overview_level`` is explicitly
        requested and exceeds the base IFD count, so a level that
        depends on the sidecar surfaces the real parse cause instead of
        degrading to a generic out-of-range error.

    Local file paths are accepted too; the helper is shape-agnostic.
    """
    sidecar_path = find_sidecar(source)
    if sidecar_path is None:
        return list(base_ifds), None, set()
    # Local-import to avoid a top-level cycle with ``_reader``. Same
    # rationale as the ``_HTTPSource`` import inside ``_probe_http``.
    from ._reader import CloudSizeLimitError
    try:
        sidecar = load_sidecar(sidecar_path, max_cloud_bytes=max_cloud_bytes)
    except CloudSizeLimitError:
        # Budget breach is a deliberate caller-visible failure mode --
        # propagate so the caller learns about it rather than silently
        # dropping the sidecar level and reading a stale base-only
        # pyramid. See ``load_sidecar`` for the error contract.
        raise
    except Exception as exc:
        # Probe succeeded but the download / parse failed. Route through
        # the shared policy the eager CPU/GPU and fsspec metadata paths
        # use: an explicit ``overview_level`` the base file alone cannot
        # serve re-raises with the real parse error chained; anything
        # else warns and falls back to base-only. This replaces the
        # previous silent swallow so a malformed remote sidecar is no
        # longer dropped without a trace. Issue #3022.
        handle_sidecar_parse_failure(
            exc, sidecar_path, overview_level,
            base_ifd_count=len(base_ifds),
        )
        return list(base_ifds), None, set()
    merged = list(base_ifds) + list(sidecar.ifds)
    sidecar_ifd_ids = {id(ifd) for ifd in sidecar.ifds}
    return merged, sidecar, sidecar_ifd_ids


def handle_sidecar_parse_failure(
    exc: BaseException,
    sidecar_path: str,
    overview_level: int | None,
    base_ifd_count: int,
) -> None:
    """React to a sidecar load/parse failure consistently across readers.

    Called by the eager CPU, eager GPU, and metadata-only paths after
    catching a non-:class:`CloudSizeLimitError` exception from
    :func:`load_sidecar`. The shared policy is:

    * If the caller explicitly requested a level the base file alone
      cannot serve (``overview_level >= base_ifd_count``), raise a
      ``ValueError`` chained from ``exc`` so the traceback shows the
      real sidecar parse failure. The release contract
      (``geotiff_release_contract.md`` row ``reader.sidecar_ovr``) and
      the in-tree warning text both promise this case surfaces the
      cause rather than degrading to a misleading "out of range" raise
      from :func:`select_overview_ifd`. Issue #2484.

    * Otherwise (``None``, ``0``, or any level the base IFD chain can
      serve), emit a ``RuntimeWarning`` and return so the caller can
      fall back to base-only behaviour.

    Gating on ``len(base_ifd_count)`` rather than the cruder
    ``overview_level >= 1`` matters when the base TIFF carries
    internal overview IFDs of its own: a request for ``level=1`` is
    reachable from the base file in that case, and forcing a raise
    just because a sibling sidecar happens to be corrupt would deny
    the user a perfectly valid read.

    Parameters
    ----------
    exc
        The exception raised by :func:`load_sidecar`.
    sidecar_path
        Path or URL of the sidecar that failed to parse. Embedded in
        both the raised error message and the fallback warning so the
        user can find the offending file.
    overview_level
        The caller's requested overview level. ``None`` and ``0`` map
        to silent-fallback. Higher values trigger the raise only when
        they exceed ``base_ifd_count``.
    base_ifd_count
        Number of IFDs already parsed from the base file's pyramid.
        A request inside ``[0, base_ifd_count)`` is satisfiable without
        the sidecar and so does not trigger the raise.
    """
    if (
        overview_level is not None
        and overview_level >= base_ifd_count
    ):
        raise ValueError(
            f"Cannot read overview_level={overview_level} from "
            f"external sidecar {sidecar_path!r}: sidecar parse failed "
            f"with {type(exc).__name__}: {exc}"
        ) from exc
    import warnings
    warnings.warn(
        f"Ignoring unreadable sidecar {sidecar_path!r}: "
        f"{type(exc).__name__}: {exc}. Falling back to "
        f"base-file-only read. Delete the .ovr file or pass "
        f"overview_level>=1 to surface the parse error.",
        RuntimeWarning,
        stacklevel=3,
    )
