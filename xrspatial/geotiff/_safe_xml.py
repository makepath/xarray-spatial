"""Safe XML parsing helpers for VRT and GDALMetadata payloads.

`xml.etree.ElementTree.fromstring` is built on top of expat and, by
default, expands internal entities. A crafted VRT file or a hostile
TIFF carrying a billion-laughs payload in the GDALMetadata tag (42112)
can OOM the host process when read via the standard API (CWE-776,
GHSA-class "XML Entity Expansion"). External entities (SYSTEM/PUBLIC)
are already blocked by expat, but internal entity DoS is not.

This module exposes a single :func:`safe_fromstring` helper that:

* Rejects any input declaring a ``<!DOCTYPE`` (and therefore any
  ``<!ENTITY ...>`` definitions) before expat ever sees the bytes.
* Falls back to :mod:`defusedxml.ElementTree` when that library is
  installed, layering a second, audited defence (defusedxml also
  blocks external entities, processing instructions on untrusted
  input, and a few other XML pitfalls).
* Otherwise falls back to :func:`xml.etree.ElementTree.fromstring`,
  which is fine once DOCTYPEs are pre-rejected: expat exposes no
  external-entity fetch by default, and without a DTD there is no
  way to declare an entity that would expand to more than the
  literal bytes.

VRT and GDALMetadata XML never contain DTDs in legitimate files, so
the pre-rejection is loss-free for real-world inputs.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as _ET

# A DOCTYPE declaration is the only path to internal entity expansion in
# stdlib XML, so refuse the whole input the moment we see one. The
# regex is intentionally permissive on whitespace because spec-compliant
# parsers accept blanks between ``<!`` and ``DOCTYPE``. We do not look
# inside comments / CDATA: those constructs cannot legally hold a
# DOCTYPE declaration anyway, and trying to be clever about them is how
# parser-confusion bugs get introduced.
_DOCTYPE_RE = re.compile(r'<!\s*DOCTYPE', re.IGNORECASE)


def _decode_bytes(data: bytes) -> str:
    """Decode XML bytes to text using BOM / XML-declaration detection.

    Mirrors the encoding-detection precedence expat would apply, so a
    UTF-16- or UTF-32-encoded payload cannot smuggle a DOCTYPE past
    the ASCII-only scanner. Falls back to UTF-8 with replacement when
    no BOM or leading null bytes hint at a wider encoding.
    """
    if data[:4] in (b'\x00\x00\xfe\xff', b'\xff\xfe\x00\x00'):
        return data.decode('utf-32')
    if data[:2] in (b'\xfe\xff', b'\xff\xfe'):
        return data.decode('utf-16')
    if data[:3] == b'\xef\xbb\xbf':
        return data.decode('utf-8-sig')
    # No BOM: a leading or alternating null byte gives away UTF-16/32.
    if len(data) >= 4:
        if data[:2] == b'\x00\x00':
            return data.decode('utf-32-be', errors='replace')
        if data[2:4] == b'\x00\x00':
            return data.decode('utf-32-le', errors='replace')
        if data[0] == 0:
            return data.decode('utf-16-be', errors='replace')
        if data[1] == 0:
            return data.decode('utf-16-le', errors='replace')
    return data.decode('utf-8', errors='replace')


def _reject_doctype(data: bytes | str) -> None:
    """Raise ValueError if *data* declares a DOCTYPE.

    Accepts both ``bytes`` and ``str`` so callers don't have to encode
    upstream. Bytes are decoded with BOM/encoding detection so that a
    UTF-16 or UTF-32 encoded payload cannot evade the ASCII scanner.
    Empty / None inputs are passed through and handled by the
    downstream parser.
    """
    if data is None:
        return
    if isinstance(data, str):
        probe = data
    else:
        probe = _decode_bytes(bytes(data))
    if _DOCTYPE_RE.search(probe):
        raise ValueError(
            "XML input contains a DOCTYPE declaration; this is refused "
            "to prevent XML entity expansion (billion-laughs) attacks. "
            "VRT and GDALMetadata payloads never need a DTD."
        )


def safe_fromstring(text: str | bytes):
    """Parse *text* into an ElementTree root, refusing DTDs / entities.

    Returns the parsed root :class:`xml.etree.ElementTree.Element`. Raises
    ``ValueError`` if the input declares a DOCTYPE, or whatever the
    underlying parser raises on malformed input (typically
    :class:`xml.etree.ElementTree.ParseError`).
    """
    _reject_doctype(text)
    try:
        # Prefer defusedxml when available -- it adds belt-and-braces
        # defences (external entity / network fetch / processing
        # instruction handling) on top of the DOCTYPE rejection.
        from defusedxml import ElementTree as _defused_ET
        return _defused_ET.fromstring(text)
    except ImportError:
        return _ET.fromstring(text)
