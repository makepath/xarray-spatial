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

COG validator CI gate
=====================

``to_geotiff(..., cog=True)`` is validated against the external
`rio-cogeo <https://github.com/cogeotiff/rio-cogeo>`_ and GDAL's
``validate_cloud_optimized_geotiff`` sample (from
`gdal/swig/python/gdal-utils/osgeo_utils/samples
<https://github.com/OSGeo/gdal/blob/master/swig/python/gdal-utils/osgeo_utils/samples/validate_cloud_optimized_geotiff.py>`_)
on every PR. A dedicated Linux job (``pytest-cog-validator``)
installs rio-cogeo and the GDAL Python bindings from conda-forge,
sets ``XRSPATIAL_REQUIRE_COG_VALIDATOR=1``, and runs the compliance
suite in ``xrspatial/geotiff/tests/test_cog_writer_compliance.py``.
With the env var set, a missing validator dependency is a hard
failure instead of a silent skip, so a misconfigured install step
cannot quietly let the gate pass. Contributors without rio-cogeo
or GDAL installed locally are unaffected: the env var is unset on
their machines and the optional validator step still skips cleanly.
See issue #2302 for the gate's design rationale.

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

Degenerate-axis writes (1xN / Nx1)
==================================

A DataArray whose spatial coords cover one row or one column has no
pixel-size signal on the length-1 axis (``coord[1] - coord[0]`` is
undefined). The writers used to borrow the non-degenerate axis's
spacing for the degenerate one (issue #1945), which silently invented
the wrong pixel size whenever the source raster was not square. A 30 m
by 10 m source written as a 1xN strip wrote out as 30 m by 30 m, and
downstream slope / proximity / zonal math then trusted a wrong
transform. See issue #2214.

The writers now fail closed in that case. A 1xN or Nx1 ``DataArray``
with spatial coords on both axes but no explicit transform raises
``ValueError``. Two ways to keep the write:

* Supply the affine on ``attrs['transform']`` (rasterio 6-tuple
  ``(px, 0, ox, 0, py, oy)``). This is the recommended path; it
  round-trips bit-exactly.
* Opt in to the borrow-from-other-axis fallback with
  ``attrs['assume_square_pixels_for_degenerate_axis'] = True``. Only
  set this when the source raster is known to be square -- the writer
  will copy the magnitude of the non-degenerate axis onto the
  degenerate one. The flag must be the boolean ``True`` (not a truthy
  string) so a stray attrs value can't accidentally re-enable the
  silent-invent path.

Multi-row / multi-column writes are unaffected. 1x1 inputs still
require ``attrs['transform']`` because neither axis has a step.

VRT missing sources
===================

``read_vrt`` accepts ``missing_sources='warn'`` or ``'raise'``. The default
``'raise'`` (since #1860) fails the read immediately if any source file
referenced by the VRT does not exist on disk. Both the eager and chunked
dispatchers honour this at construction time -- chunked callers do not
have to wait until ``compute()`` to learn the VRT is broken (#2265).
The static missing-source sweep is scoped to the requested ``window=``
and ``band=`` so a windowed or band-restricted read that does not depend
on a missing source still succeeds.

Pass ``missing_sources='warn'`` to opt into the lenient path: unreadable
source files emit :class:`xrspatial.geotiff.GeoTIFFFallbackWarning`, the
returned DataArray carries ``attrs['vrt_holes']``, and the mosaic is
returned with holes left as the band's nodata sentinel (or zero on
integer bands without a sentinel). ``XRSPATIAL_GEOTIFF_STRICT=1``
forces the raise in ``'warn'`` mode too, so CI environments can enforce
fail-fast behavior globally.
