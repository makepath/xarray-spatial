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
from typing import NamedTuple

from ._header import IFD, TIFFHeader, parse_all_ifds, parse_header


class SidecarOverviews(NamedTuple):
    """Bytes, header, and IFDs from a sibling ``.tif.ovr`` sidecar."""

    data: object  # bytes-like (mmap or bytes)
    header: TIFFHeader
    ifds: list[IFD]
    path: str


def _is_http_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _is_fsspec_uri(source: str) -> bool:
    # Same gate ``_reader._is_fsspec_uri`` uses: any ``scheme://`` that is
    # not http / https / local-mmap-able. We avoid importing the reader
    # helper here to keep the dependency direction one-way.
    if "://" not in source:
        return False
    if _is_http_url(source):
        return False
    return True


def find_sidecar(source) -> str | None:
    """Return the path / URL of a sibling ``.ovr`` sidecar if one exists.

    Scopes:

    * Local file paths -- probe with :func:`os.path.isfile`.
    * HTTP / HTTPS URLs -- issue a single HEAD request to
      ``<url>.ovr``; treat any 2xx as "exists".
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
    """Return ``url`` if a HEAD request reports an existing object."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        # 10 second timeout matches the eager HTTP reader's defaults;
        # a stuck remote should not block sidecar discovery indefinitely.
        with urllib.request.urlopen(req, timeout=10) as resp:
            return url if 200 <= resp.status < 300 else None
    except Exception:
        # 404 (urllib raises HTTPError) and any network error reach
        # here; either way the sidecar is unavailable from our point
        # of view and we fall back to base-only.
        return None


def _probe_fsspec(uri: str) -> str | None:
    """Return ``uri`` if fsspec reports the object exists."""
    try:
        import fsspec
        fs, path = fsspec.core.url_to_fs(uri)
        return uri if fs.exists(path) else None
    except Exception:
        return None


def load_sidecar(path: str) -> SidecarOverviews:
    """Open and parse a sidecar ``.ovr`` file.

    Accepts local file paths, HTTP / HTTPS URLs, and fsspec URIs.
    Local paths are mmap'd; remote sources are downloaded once via the
    matching transport (HTTP via :mod:`urllib`, fsspec URIs via the
    fsspec filesystem). The IFD list is the sidecar's full IFD chain
    in file order; the reader treats them as overview levels (the
    first sidecar IFD is level 1 when the base file holds only a
    full-resolution IFD, level 2 when the base file already carries
    one internal overview, and so on).

    The returned ``data`` is either an ``mmap`` (local) or ``bytes``
    (remote). Callers should close the mmap variant via
    ``data.close()`` when present; the bytes case is no-op.
    """
    if "://" not in path:
        f = open(path, "rb")
        try:
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        finally:
            f.close()
    elif _is_http_url(path):
        import urllib.request
        with urllib.request.urlopen(path, timeout=30) as resp:
            data = resp.read()
    else:
        # fsspec URI
        import fsspec
        with fsspec.open(path, "rb") as f:
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


def attach_sidecar_origin(ifds: list[IFD],
                          data: object,
                          header: TIFFHeader) -> None:
    """Tag each IFD in ``ifds`` with its source bytes and header.

    The reader checks ``ifd._source_data`` / ``ifd._source_header`` to
    decide which buffer to slice for strip/tile reads. Untagged IFDs
    fall through to the base-file ``data`` / ``header`` the caller
    already has in scope.
    """
    for ifd in ifds:
        # Use ``object.__setattr__`` so this works whether IFD is a
        # plain dataclass or a frozen one in a future revision.
        object.__setattr__(ifd, "_source_data", data)
        object.__setattr__(ifd, "_source_header", header)
