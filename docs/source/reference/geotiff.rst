..  _reference.geotiff:

***************
GeoTIFF / COG
***************

.. seealso::

   :ref:`reference.geotiff_release_contract` -- the user-facing release
   contract that defines what each support tier promises and lists every
   feature in :data:`xrspatial.geotiff.SUPPORTED_FEATURES` against its
   tier.

   :ref:`reference.geotiff_release_gate` -- the release gate / audit
   checklist that lists every promised feature on this page, its tier,
   its one-line acceptance, and the regression test that locks it.

   :ref:`user_guide.attrs_contract` -- the user-guide page that defines
   which attrs keys are canonical, which are aliases, and which are
   pass-through, and the round-trip guarantees that apply to each
   tier.

GPU support (experimental)
==========================

The GPU read and write paths are tagged ``experimental`` in
:data:`xrspatial.geotiff.SUPPORTED_FEATURES`. Both
``SUPPORTED_FEATURES['reader.gpu']`` and
``SUPPORTED_FEATURES['writer.gpu']`` report ``experimental``: the paths
work and are covered by tests, but the surface can shift without a
deprecation window. The GPU paths are not a release blocker -- a
regression on a GPU row does not fail the build the way a regression
on the stable CPU surface does.

What you can expect:

* GPU read and write produce the same pixels and the same canonical
  attrs as the CPU path on the supported codec subset. The eager and
  dask GPU readers are covered by
  ``xrspatial/geotiff/tests/golden_corpus/test_gpu.py`` and
  ``xrspatial/geotiff/tests/golden_corpus/test_dask_gpu.py``.
* Integer and float nodata sentinels survive the GPU round-trip; see
  ``xrspatial/geotiff/tests/gpu/test_reader.py``.
* On GPU failure the reader emits
  :class:`xrspatial.geotiff.GeoTIFFFallbackWarning` and falls back to
  CPU unless ``on_gpu_failure='strict'`` or
  ``XRSPATIAL_GEOTIFF_STRICT=1`` is set; see
  ``xrspatial/geotiff/tests/gpu/test_kernels_and_kwargs.py``.

What you should NOT rely on:

* GPU support for every codec on the CPU path. ``allow_experimental_codecs``
  does NOT widen the GPU codec set; on the GPU writer, codecs outside the
  GPU-supported set route through a CPU fallback inside the GPU writer
  (``_write_geotiff_gpu``) rather than executing on the GPU. Locked by
  ``xrspatial/geotiff/tests/gpu/test_writer.py``.
* GPU promotion to ``stable`` inside this release cycle. See the GPU
  rows in :ref:`reference.geotiff_release_gate` for the current tier
  and the regression tests behind each row.

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

What the stable COG contract does NOT cover is the overview pyramid
customisation surface itself. The presence of an internal overview
pyramid is part of the stable COG layout, but the
``overview_levels`` and ``overview_resampling`` knobs on
``to_geotiff`` (and the pyramid bytes the resampling kernels produce)
are tracked as ``advanced`` under
``SUPPORTED_FEATURES['writer.overviews']``. The two registry entries
exist precisely so that the COG layout and the pyramid-generation
knobs can promote independently; the ``to_geotiff`` docstring marks
``cog=True`` as stable but flags ``overview_levels`` and
``overview_resampling`` as advanced for the same reason.

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

Restricting a read to stable sources
------------------------------------

The ``stable_only=True`` opt-in restricts a read to stable-tier sources.
Under it, ``open_geotiff`` rejects advanced-tier sources before any fetch
or decode: a ``.vrt`` mosaic raises ``VRTStableSourcesOnlyError``, and an
HTTP(S) URL or fsspec URI (``s3://``, ``gs://``, ``memory://``) raises
``RemoteStableSourcesOnlyError``. Both subclass
``GeoTIFFAmbiguousMetadataError``. The eager, dask, and GPU paths all run
this check, so a remote source cannot slip through on a non-VRT path. Pass
``allow_experimental_codecs=True`` to opt back into the advanced and
experimental tiers, or drop ``stable_only`` for the default behaviour.
Local-file reads always pass.

Rotated and sheared transforms
==============================

Read posture. ``open_geotiff`` rejects a file whose affine transform
has non-zero rotation or shear coefficients by default. Pass
``allow_rotated=True`` to opt in: the read then surfaces the rotated
6-tuple on ``attrs['rotated_affine']`` and drops ``attrs['crs']`` so
downstream math cannot silently mix a rotated grid with an
axis-aligned CRS. The dropped-CRS rule is locked by
``xrspatial/geotiff/tests/read/test_crs.py``. The
HTTP dask path honours the same opt-in via
``xrspatial/geotiff/tests/test_http_dask_allow_rotated_2130.py``.
Without ``allow_rotated=True`` the read raises a typed error; see
``xrspatial/geotiff/tests/test_rotated_typed_error_2267.py``.

Write posture. ``to_geotiff`` rejects a DataArray carrying
``attrs['rotated_affine']`` unless the caller also passes
``drop_rotation=True``. With the opt-in, the writer drops the rotated
affine and writes an axis-aligned file from the coords. This is
locked by ``xrspatial/geotiff/tests/test_to_geotiff_drop_rotation_2216.py``.
A rotated or skewed 6-tuple supplied through ``attrs['transform']``
or through a VRT source is also rejected; see
``xrspatial/geotiff/tests/release_gates/test_features.py``
(``test_eager_writer_rejects_rotated_6tuple_transform`` and
``test_vrt_with_skewed_geotransform_rejected``).

Failure-closed combinations. The following inputs raise rather than
silently emit a mislabeled raster:

* Rotated read without ``allow_rotated=True`` -- raises across eager,
  dask, and windowed paths
  (``xrspatial/geotiff/tests/release_gates/test_stable_features.py``,
  ``Negative cases`` section).
* Rotated write without ``drop_rotation=True`` -- raises ``ValueError``
  (``xrspatial/geotiff/tests/test_to_geotiff_drop_rotation_2216.py``).
* Rotated or skewed source inside a VRT -- raises at parse
  (``xrspatial/geotiff/tests/test_vrt_unsupported_2370.py``).

Nodata lifecycle
================

This page summarises the read / write contract. The full lifecycle
of every attrs key, including which keys are canonical, which are
aliases, and which are pass-through, lives in
:ref:`user_guide.attrs_contract`. Do not duplicate that page here;
this section is the brief.

* Integer nodata. The on-disk sentinel survives the read bit-exact
  and is preserved on the next write. ``attrs['nodata']`` carries
  the sentinel as a Python ``int``. Out-of-range sentinels for the
  band dtype are rejected at write
  (``xrspatial/geotiff/tests/write/test_nodata.py``).
* Float nodata. The on-disk sentinel is recorded on
  ``attrs['nodata']`` and surfaces as NaN in pixel data only when the
  read promotes via ``masked=True``. The default is ``masked=False``
  (matching rioxarray's ``open_rasterio``), so by default the raw float
  sentinel passes through and downstream callers can branch on the exact
  value; pass ``masked=True`` to get NaN-masked output.
  ``xrspatial/geotiff/tests/write/test_nodata.py`` pins this split.
  (``mask_nodata`` is a deprecated alias of ``masked``.)
* NaN nodata. A file that declares ``nodata=NaN`` is read with NaN in
  both ``attrs['nodata']`` and pixel data (NaN propagates either way).
* ``attrs['masked_nodata']``. Every read sets a boolean lifecycle
  signal: ``True`` when the read produced NaN-masked output distinct
  from the on-disk sentinel, ``False`` when pixel data carries the
  raw sentinel. The signal is part of the canonical attrs contract;
  ``xrspatial/geotiff/tests/read/test_nodata.py`` pins
  the canonical form and
  ``xrspatial/geotiff/tests/vrt/test_metadata.py``
  covers the VRT mosaic case.
* Mixed-band nodata. A VRT whose sources declare disagreeing per-band
  nodata sentinels raises ``MixedBandMetadataError`` by default. Pass
  ``band_nodata='first'`` to opt back into the legacy flatten-to-band-0
  behaviour; see ``xrspatial/geotiff/tests/vrt/test_metadata.py``.

The lifecycle is locked end-to-end by
``xrspatial/geotiff/tests/read/test_nodata.py``.

Reading
=======
``open_geotiff`` is the single read entry point. The backend follows the
parameters: ``gpu=True`` returns a CuPy-backed array, ``chunks=N`` returns a
lazy dask array, and a ``.vrt`` source reads a mosaic.

.. autosummary::
    :toctree: _autosummary

    xrspatial.geotiff.open_geotiff

Writing
=======
``to_geotiff`` is the single write entry point (``gpu=True`` or CuPy data
selects the GPU path; a ``.vrt`` output path writes tiles plus an index).
Writing to a ``.vrt`` path is how you produce a VRT mosaic; the underlying
index emitter is internal.

.. autosummary::
    :toctree: _autosummary

    xrspatial.geotiff.to_geotiff

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
suite in ``xrspatial/geotiff/tests/write/test_cog.py``.
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

Remote-read safety limits and env vars
--------------------------------------

The reader applies a layered budget to every remote ``http://`` or
``https://`` read so a single hostile file cannot exhaust memory or
turn the process into a port scanner. The knobs are:

* ``max_cloud_bytes`` (kwarg) / ``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES``
  (env). Per-call total byte budget for a remote read. The kwarg wins
  over the env var; the env var wins over the built-in default. Pass
  ``max_cloud_bytes=None`` to disable the cap on a single call. Locked
  by ``xrspatial/geotiff/tests/integration/test_http_sources.py``
  (max_cloud_bytes_dispatcher and max_cloud_bytes_annot sections, plus
  the http_read_all_bounded section).
* ``XRSPATIAL_COG_MAX_TILE_BYTES``. Per-tile / per-strip compressed
  byte cap (default 256 MiB). Locked by
  ``xrspatial/geotiff/tests/read/test_tiling.py``,
  ``xrspatial/geotiff/tests/test_cloud_read_byte_limit_1928.py``, and
  ``xrspatial/geotiff/tests/read/test_tiling.py``.
* ``XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT`` and
  ``XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT``. Per-request connect / read
  timeouts in seconds. Positive floats only; other values fall back
  to the defaults (10 s and 30 s). Range coalescing inside one read
  shares a single connection so the connect timeout applies once per
  host, not once per range.
* ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS``. Set to ``1`` (or
  ``true`` / ``yes``) to disable the private-host reject. Off by
  default; locked by
  ``xrspatial/geotiff/tests/integration/test_http_sources.py``
  (ssrf_hardening, dns_rebinding, and uppercase_scheme_ssrf sections).
* ``XRSPATIAL_VRT_ALLOWED_ROOTS``. Colon-separated list of additional
  directory roots that a VRT is allowed to reference. The default
  containment rule (sources must live under the VRT's directory) is
  locked by ``xrspatial/geotiff/tests/test_vrt_path_containment_1671.py``.
* ``XRSPATIAL_GEOTIFF_STRICT``. Promotes the fallback warnings into
  raised exceptions, including the GPU-fallback path; see the next
  section.

The same byte budget applies to sidecar fetches, not just the parent
file (``xrspatial/geotiff/tests/integration/test_sidecar.py``,
sidecar_max_cloud_bytes section).

Strict mode (``XRSPATIAL_GEOTIFF_STRICT``)
==========================================

Several internal helpers historically returned ``None`` when something went
wrong: pyproj failing to parse a WKT string, a VRT source file being
missing, a GPU helper (GDS, nvCOMP, nvJPEG, nvJPEG2000) hitting a CUDA or
library error. These now emit :class:`xrspatial.geotiff.GeoTIFFFallbackWarning`
with the original exception type and message.

Set ``XRSPATIAL_GEOTIFF_STRICT=1`` (or ``true``, ``yes``) to promote those
warnings into raised exceptions. The same env var also forces
``open_geotiff(gpu=True, on_gpu_failure='auto')`` to behave like
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

.. _reference.geotiff.vrt_support_matrix:

VRT support matrix (issue #2321)
================================

VRT reads sit at the ``advanced`` tier in
:data:`xrspatial.geotiff.SUPPORTED_FEATURES` (``reader.vrt``).
``open_geotiff`` (on a ``.vrt`` source) and ``to_geotiff`` (to a ``.vrt``
output) both target the same narrow subset of GDAL's VRT spec. The
reference below is the canonical contract; the docstrings echo it.

Supported
---------

* Simple GDAL VRT mosaics whose ``<SourceFilename>`` entries point at
  GeoTIFF files. The VRT XML must resolve to source paths under the
  VRT's own directory (or under a root listed in
  ``XRSPATIAL_VRT_ALLOWED_ROOTS``); see the source-path containment
  note on the VRT reader (``_read_vrt``) (#1671).
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

Write a chunked DataArray to a ``.vrt`` path (``to_geotiff`` tiles the
array and emits the index), then read it back eagerly with the
fail-closed defaults:

.. code-block:: python

    from xrspatial.geotiff import open_geotiff, to_geotiff

    # ``da`` is a 2D dask-backed DataArray with crs / transform set.
    # Writing to a .vrt produces a directory of tiled GeoTIFFs plus
    # the VRT index that references them.
    to_geotiff(da, 'mosaic.vrt')

    # Read with the defaults: missing_sources='raise',
    # band_nodata=None (fail closed on disagreeing per-band sentinels).
    back = open_geotiff('mosaic.vrt')

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

``open_geotiff`` accepts ``missing_sources='warn'`` or ``'raise'`` for
``.vrt`` sources. The default
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

Under chunked dispatch (``chunks=`` set), ``attrs['vrt_holes']`` records
statically-detectable missing-file holes only (#2989). A source file
that exists but fails to decode at compute time -- corrupt, truncated,
or a codec error -- reads as a hole in the per-chunk worker and emits a
:class:`xrspatial.geotiff.GeoTIFFFallbackWarning` then, but cannot be
reduced back onto the parent DataArray's ``attrs['vrt_holes']`` without
eagerly decoding every source (which would defeat lazy reading). So the
chunked ``attrs['vrt_holes']`` is a lower bound, not the eager reader's
exhaustive list. The chunked ``'warn'`` path emits one
``GeoTIFFFallbackWarning`` up front when the requested window touches any
existing source, so do not treat the absence of ``attrs['vrt_holes']``
as proof of a complete mosaic. Use ``missing_sources='raise'`` to fail
closed if you need a hard guarantee.

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
``xrspatial/geotiff/tests/write/test_bigtiff.py`` and
covers the BigTIFF-specific layout (header, IFDs, tile and overview
offset tables), one lossless integer codec, one lossless float codec,
single-band and 3-band, one overview level, plus an auto-promotion row
that drives the threshold via the IFD-overhead helper rather than
allocating a multi-gigabyte buffer. Promotion to ``stable`` follows the
same release-cycle soak rule as the rest of the COG surface.

Known unsupported combinations
==============================

The combinations below fail closed today: they raise a typed error
rather than emit a possibly-wrong raster. Each row names the
regression test that locks the behaviour.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Combination
     - Regression test
   * - ``to_geotiff(cog=True, tiled=False)``
     - ``xrspatial/geotiff/tests/write/test_cog.py``
   * - ``to_geotiff(cog=True, tile_size <= 0)``
     - ``xrspatial/geotiff/tests/write/test_cog.py``
   * - Warped VRT
       (``<VRTDataset subClass="VRTWarpedDataset">`` or
       ``<VRTRasterBand subClass="VRTWarpedRasterBand">``)
     - ``xrspatial/geotiff/tests/test_vrt_unsupported_2370.py``,
       ``xrspatial/geotiff/tests/test_vrt_capability_validator_2371.py``
   * - Nested VRT (a ``<SourceFilename>`` that resolves to a ``.vrt``)
     - ``xrspatial/geotiff/tests/test_vrt_unsupported_2370.py``
       (``test_nested_vrt_source_raises``,
       ``test_nested_vrt_open_geotiff_raises``)
   * - Mixed-CRS VRT (sources disagree on CRS without an opt-in)
     - ``xrspatial/geotiff/tests/test_vrt_unsupported_2370.py``,
       ``xrspatial/geotiff/tests/test_vrt_capability_validator_2371.py``
   * - Mixed per-band nodata across VRT sources (default
       ``band_nodata=None``)
     - ``xrspatial/geotiff/tests/vrt/test_metadata.py``,
       ``xrspatial/geotiff/tests/release_gates/test_features.py``
       (``test_mixed_per_source_nodata_rejected``)
   * - Rotated read without ``allow_rotated=True``
     - ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
       (``Negative cases`` section),
       ``xrspatial/geotiff/tests/test_rotated_typed_error_2267.py``
   * - Rotated write without ``drop_rotation=True``
     - ``xrspatial/geotiff/tests/test_to_geotiff_drop_rotation_2216.py``,
       ``xrspatial/geotiff/tests/release_gates/test_features.py``
       (``test_eager_writer_rejects_rotated_6tuple_transform``,
       ``test_eager_writer_rejects_rotated_affine_attr``)
   * - Skewed VRT geotransform
     - ``xrspatial/geotiff/tests/release_gates/test_features.py``
       (``test_vrt_with_skewed_geotransform_rejected``)
   * - Complex source / mask band / alpha band in a VRT
     - ``xrspatial/geotiff/tests/test_vrt_unsupported_2370.py``,
       ``xrspatial/geotiff/tests/test_vrt_capability_validator_2371.py``
   * - VRT source path escapes the VRT directory tree
     - ``xrspatial/geotiff/tests/test_vrt_path_containment_1671.py``
   * - 1xN / Nx1 write without ``attrs['transform']`` or
       ``assume_square_pixels_for_degenerate_axis=True``
     - ``xrspatial/geotiff/tests/test_degenerate_pixel_size_2214.py``;
       see also "Degenerate-axis writes" above.
   * - HTTP read against a private / loopback / link-local host
       without ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1``
     - ``xrspatial/geotiff/tests/integration/test_http_sources.py``
       (ssrf_hardening and dns_rebinding sections)
   * - Duplicate IFD tag id within one IFD (TIFF 6.0 forbids this; the
       parser raises ``DuplicateIFDTagError`` rather than letting the
       last duplicate win silently)
     - ``xrspatial/geotiff/tests/unit/test_ifd.py``
       (Section 2b, ``test_duplicate_*``)
   * - Unsupported feature flags more broadly (codec, layout, and
       writer combos that ``SUPPORTED_FEATURES`` does not promise)
     - ``xrspatial/geotiff/tests/release_gates/test_features.py``

This list is the prose mirror of the negative rows in
:ref:`reference.geotiff_release_gate`. When a row gets promoted or
removed, update both pages in the same PR so the docs and the runtime
constant stay in sync.
