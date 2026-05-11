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
* Otherwise uses :class:`xml.etree.ElementTree.XMLParser` directly,
  which is fine once DOCTYPEs are pre-rejected: the parser exposes no
  external-entity fetch and refuses anything else that would expand
  to more than the literal bytes.

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
_DOCTYPE_RE = re.compile(rb'<!\s*DOCTYPE', re.IGNORECASE)


def _reject_doctype(data: bytes | str) -> None:
    """Raise ValueError if *data* declares a DOCTYPE.

    Accepts both ``bytes`` and ``str`` so callers don't have to encode
    upstream. Empty / None inputs are passed through and handled by the
    downstream parser.
    """
    if data is None:
        return
    if isinstance(data, str):
        probe = data.encode('utf-8', errors='ignore')
    else:
        probe = bytes(data)
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
