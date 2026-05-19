"""External `.tif.ovr` sidecar discovery for overview reading.

GDAL and rasterio write overview pyramids to a sibling ``<path>.ovr``
file by default (`gdaladdo -ro` and `rasterio` `--overview-resampling`
when the source itself is not a COG). The sidecar is a standalone TIFF
whose IFDs are the continuation of the base file's pyramid: base IFD 0
is overview level 0, sidecar IFD 0 is level 1, sidecar IFD 1 is level
2, and so on.

This module discovers the sidecar next to a local-file source and
parses its IFDs, returning the inputs needed for the reader to switch
to the sidecar's bytes/header when the selected overview level lives
there. Cloud / HTTP / file-like sources are not in scope for the first
cut (see issue #2112): they would need sidecar URLs to be discoverable
in their respective namespaces, which is not a TIFF concern. Callers
that hit a non-local source skip the sidecar discovery entirely and
fall back to base-file-only behaviour.
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


def find_sidecar(source) -> str | None:
    """Return the path to a sibling ``.ovr`` sidecar if one exists.

    Only local file paths are considered. File-like buffers and fsspec
    / HTTP URIs are out of scope (their sidecar conventions differ and
    discovery is non-trivial); they return ``None`` so callers fall
    back to base-file-only behaviour.

    The sidecar lives at ``<source>.ovr`` per the GDAL convention. The
    function returns the absolute path on success and ``None`` if the
    sidecar does not exist or the source is not a local file path.
    """
    if not isinstance(source, str):
        return None
    # fsspec URIs (s3://, gs://, az://, memory://, http://, https://)
    # share the str type with local paths but should not be probed via
    # ``os.path.exists``: the latter would either succeed against a
    # local file that happens to share the name or fail and we cannot
    # safely fetch a remote sidecar from this layer.
    if "://" in source:
        return None
    sidecar = source + ".ovr"
    if not os.path.isfile(sidecar):
        return None
    return sidecar


def load_sidecar(path: str) -> SidecarOverviews:
    """Open and parse a sidecar ``.ovr`` file.

    The returned ``data`` is an ``mmap`` object the caller must close
    (or wrap in a ``try / finally`` next to its own data close). The
    IFD list is the sidecar's full IFD chain in file order; the reader
    treats them as overview levels (the first sidecar IFD is level 1
    when the base file holds only a full-resolution IFD, level 2 when
    the base file already carries one internal overview, and so on).
    """
    f = open(path, "rb")
    try:
        # ``mmap`` over the whole file matches the eager reader's
        # ``read_to_array`` path for local sources; the lifetime is
        # owned by the caller.
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    finally:
        f.close()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    return SidecarOverviews(data=data, header=header, ifds=ifds, path=path)


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
