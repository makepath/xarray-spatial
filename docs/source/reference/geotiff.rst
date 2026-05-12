..  _reference.geotiff:

***************
GeoTIFF / COG
***************

Reading
=======
.. autosummary::
    :toctree: _autosummary

    xrspatial.geotiff.open_geotiff
    xrspatial.geotiff.read_vrt

Writing
=======
.. autosummary::
    :toctree: _autosummary

    xrspatial.geotiff.to_geotiff
    xrspatial.geotiff.write_geotiff_gpu
    xrspatial.geotiff.write_vrt

Security and I/O limits
=======================

``open_geotiff`` and the underlying reader enforce several limits to
keep crafted or hostile inputs from exhausting memory or reaching
internal network targets. All limits have safe defaults; advanced users
can override them via environment variables.

Per-tile / per-strip compressed-byte cap
----------------------------------------

A crafted TIFF can declare arbitrarily large ``TileByteCounts`` or
``StripByteCounts``. Both the HTTP fetcher (which would issue a Range
GET sized by the attacker's value) and the local-file decoder (where a
small compressed slice can balloon under deflate / zstd / lzw) reject
any tile or strip whose declared size exceeds the cap.

* Default: 256 MiB
* Override: ``XRSPATIAL_COG_MAX_TILE_BYTES`` (positive integer, bytes).
  Non-integer, empty, zero, or negative values are ignored and fall back
  to the default. Set above your largest legitimate tile or strip size.
* Exception: ``ValueError`` ("safety cap")

HTTP SSRF defenses
------------------

When ``open_geotiff`` is given an ``http://`` or ``https://`` URL, the
reader rejects URLs that would let a service-side caller probe internal
infrastructure. Other ``scheme://`` strings are dispatched through
fsspec and are not covered by these checks.

* Scheme allow-list: ``http`` and ``https`` only.
* Host filtering: hostnames that resolve to a loopback (``127.0.0.0/8``,
  ``::1``), link-local (``169.254.0.0/16``, ``fe80::/10``), or RFC1918
  private range are rejected. Override via
  ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1``. The check rejects on
  *any* resolved IP being unsafe, which also blocks DNS-rebind tricks.
* Redirect handling: at most 5 redirects per request. Each ``Location``
  is re-validated against the same scheme and host filter, so a public
  URL cannot 3xx-redirect into private space. Requires ``urllib3``; on
  the stdlib fallback the same cap and re-validation are enforced via
  a custom redirect handler.
* Timeouts: 10 s connect, 30 s read by default. Override via
  ``XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT`` and
  ``XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT`` (positive float, seconds).
* Exception: :class:`xrspatial.geotiff.UnsafeURLError` (a
  ``ValueError`` subclass).

If you run an integration test against a local HTTP server (e.g.
``http.server`` bound to ``127.0.0.1``), set
``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` for the duration of the
test.

Strict mode (``XRSPATIAL_GEOTIFF_STRICT``)
==========================================

Several internal helpers historically returned ``None`` when something went
wrong: pyproj failing to parse a WKT string, a VRT source file being
missing, a GPU helper (GDS, nvCOMP, nvJPEG, nvJPEG2000) hitting a CUDA or
library error. These now emit :class:`xrspatial.geotiff.GeoTIFFFallbackWarning`
with the original exception type and message.

Set ``XRSPATIAL_GEOTIFF_STRICT=1`` (or ``true``, ``yes``) to promote those
warnings into raised exceptions. The same env var also forces
``read_geotiff_gpu(on_gpu_failure='auto')`` to behave like
``on_gpu_failure='strict'`` so CI can fail loudly when the GPU fast path
silently falls back to CPU.

.. code-block:: bash

    XRSPATIAL_GEOTIFF_STRICT=1 pytest xrspatial/geotiff/tests/

See issue #1662 for the audit and the full list of affected call sites.
