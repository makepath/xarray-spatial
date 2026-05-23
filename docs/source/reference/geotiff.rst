..  _reference.geotiff:

***************
GeoTIFF / COG
***************

Stable COG contract
===================

As of the #2286 production-readiness wave, the local COG read and write
paths are tagged ``stable`` in
:data:`xrspatial.geotiff.SUPPORTED_FEATURES`. ``SUPPORTED_FEATURES['writer.cog']``
and ``SUPPORTED_FEATURES['reader.local_cog']`` both report ``stable``;
``SUPPORTED_FEATURES['reader.http_cog']`` stays ``advanced`` while the
HTTP transport surface is contracted separately.

The contract covers:

* Axis-aligned 2D / 3D rasters.
* CPU writer and CPU reader paths.
* Stable codecs only: ``none``, ``deflate``, ``lzw``, ``zstd``,
  ``packbits``.
* Internal overviews only.
* Normal CRS, transform, dtype, nodata, band, and
  pixel-is-area / pixel-is-point behavior.

The promotion is backed by the writer compliance suite (#2292), the
cross-backend parity gate (#2293), and the per-tile byte-budget contract
(#2294 / #2298). These tests run on every CI build so a regression in
the stable surface fails the build rather than silently shipping.

Outside the stable contract
----------------------------

The following combinations stay outside the stable contract. They still
work where they did before and are still tested, but they keep their
existing tier (``advanced``, ``experimental``, or ``internal_only``) and
the corresponding caveats:

* GPU COG read / write.
* Experimental codecs (``lerc``, ``jpeg2000`` / ``j2k``, ``lz4``).
* Internal-only ``jpeg``.
* Rotated transforms.
* External ``.tif.ovr`` sidecars.
* File-like destinations with ``cog=True``.
* BigTIFF COG (tracked separately).
* HTTP / range COG (tracked separately; see the byte-budget contract in
  #2298).

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

VRT support matrix (issue #2321)
================================

VRT reads sit at the ``advanced`` tier in
:data:`xrspatial.geotiff.SUPPORTED_FEATURES` (``reader.vrt``).
``open_geotiff``, ``read_vrt``, and ``write_vrt`` all target the same
narrow subset of GDAL's VRT spec. The reference below is the canonical
contract; the three docstrings echo it.

Supported
---------

* Simple GDAL VRT mosaics whose ``<SourceFilename>`` entries point at
  GeoTIFF files. The VRT XML must resolve to source paths under the
  VRT's own directory (or under a root listed in
  ``XRSPATIAL_VRT_ALLOWED_ROOTS``); see the source-path containment
  note on ``read_vrt`` (#1671).
* Sources that agree on CRS, transform orientation (axis-aligned,
  same sign on the y step), pixel size, dtype, and band count. The
  read rejects mismatch with ``MixedBandMetadataError`` /
  ``ValueError`` rather than silently flattening.
* Windowed reads via ``window=(row_start, col_start, row_stop,
  col_stop)``. Eager and dask paths shift coords and
  ``attrs['transform']`` together so a windowed eager read and a
  windowed dask read agree on metadata.
* Lazy / dask reads over the same subset via ``chunks=``. Construction
  parses the VRT XML and runs a parse-time existence sweep over every
  referenced source so a missing file is surfaced at graph build, not
  at ``compute()`` time (#2265).
* Explicit ``nodata``. The default (``band_nodata=None``) rejects a VRT
  whose bands declare disagreeing per-band ``<NoDataValue>`` sentinels
  with ``MixedBandMetadataError``. ``band_nodata='first'`` opts back
  into the legacy flatten-to-band-0 behaviour explicitly (#1987).
* ``missing_sources='raise'`` (the default since #1860). Pass
  ``missing_sources='warn'`` to opt into the lenient partial-mosaic
  path; see "VRT missing sources" below.

Non-goals (intentionally unsupported)
-------------------------------------

* Warped / reprojection VRTs (``<VRTDataset subClass="VRTWarpedDataset">``).
* Arbitrary resampling beyond the tested subset. The VRT reader honours
  only the small set of resampling rules its test corpus covers; other
  modes raise rather than silently picking a default.
* Mixed CRS, resolution, dtype, or band metadata across sources without
  an explicit opt-in. The default behaviour is to fail closed.
* Nested VRTs (a ``<SourceFilename>`` that itself points at a ``.vrt``).
* Complex source / mask band / alpha band structures
  (``<ComplexSource>`` with arbitrary scale and offset,
  ``<MaskBand>``, ``<AlphaBand>``).
* Full GDAL VRT parity. The contract above is the supported surface;
  anything outside it is on a best-effort basis at most and is allowed
  to raise.

Safe usage
----------

A simple mosaic over two compatible GeoTIFF tiles, read eagerly with
the fail-closed defaults:

.. code-block:: python

    from xrspatial.geotiff import open_geotiff, write_vrt

    # Write a VRT that mosaics two tiles. Both tiles share CRS,
    # pixel size, dtype, and band count.
    vrt_path = write_vrt(
        'mosaic.vrt',
        source_files=['tile_west.tif', 'tile_east.tif'],
    )

    # Read with the defaults: missing_sources='raise',
    # band_nodata=None (fail closed on disagreeing per-band sentinels).
    da = open_geotiff(vrt_path)

Intentionally raises
--------------------

Pointing the read at a VRT whose source tiles disagree on their
per-band nodata sentinels triggers the fail-closed check:

.. code-block:: python

    from xrspatial.geotiff import open_geotiff, MixedBandMetadataError

    # tile_a.tif declares nodata=-9999, tile_b.tif declares nodata=0.
    # The default band_nodata=None rejects the mosaic rather than
    # flattening to one sentinel.
    try:
        open_geotiff('mixed_nodata.vrt')
    except MixedBandMetadataError:
        # Pass band_nodata='first' to opt back into the legacy
        # flatten-to-band-0 semantics, or fix the source tiles.
        pass

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

BigTIFF COG (issue #2303)
=========================

A COG larger than the classic-TIFF 4 GiB offset ceiling needs the
BigTIFF wrapper (magic ``43``, 8-byte offsets, 20-byte IFD entries).
``to_geotiff(..., cog=True)`` auto-promotes to BigTIFF when the
estimated file size exceeds ``UINT32_MAX`` (0xFFFFFFFF bytes); callers
can force the wrapper with ``bigtiff=True`` even on small rasters when
they want a stable layout for downstream tooling that probes the magic
byte. The same threshold and force-flag rules apply whether the output
is a plain GeoTIFF or a COG.

``SUPPORTED_FEATURES['writer.bigtiff_cog']`` is currently ``advanced``.
The external-interop gate lives in
``xrspatial/geotiff/tests/test_bigtiff_cog_compliance_2286.py`` and
covers the BigTIFF-specific layout (header, IFDs, tile and overview
offset tables), one lossless integer codec, one lossless float codec,
single-band and 3-band, one overview level, plus an auto-promotion row
that drives the threshold via the IFD-overhead helper rather than
allocating a multi-gigabyte buffer. Promotion to ``stable`` follows the
same release-cycle soak rule as the rest of the COG surface.
