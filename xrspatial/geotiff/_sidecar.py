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
    return source.startswith(("http://", "https://"))


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
      ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` is set (#1664).
    * IP pinning so a DNS-rebind between probe and download cannot
      flip the target (#1846).
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
        the file. Issue #2121.

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
        # multi-GB ``.ovr`` to OOM the process. Issue #2121.
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
        # ``_CloudSource`` size guard at ``_reader.py:3239-3260``.
        # Issue #2121.
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
