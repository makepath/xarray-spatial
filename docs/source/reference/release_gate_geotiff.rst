.. _reference.geotiff_release_gate:

**************************************
GeoTIFF release gate / audit checklist
**************************************

.. note::

   This page is the audit trail for the GeoTIFF promises that release notes
   are allowed to make. Every promised tier below points to at least one
   regression test that locks the behaviour. If a row is missing a test, the
   release notes cannot promise that tier.

   The tier strings come from :data:`xrspatial.geotiff.SUPPORTED_FEATURES`.
   ``stable`` means the contract is gated by CI and a regression breaks the
   build. ``advanced`` means the feature is tested and works but the surface
   may shift before a 1.0 release. ``experimental`` means the feature is
   covered but not promised; behaviour can change without a deprecation
   window. ``internal_only`` means the feature exists for one specific
   internal use case and is not part of the public surface.

   Each row also names the epic that owns the row so a reviewer can trace
   any acceptance statement back to the issue that defined it: ``#2286``
   (COG promotion), ``#2321`` (VRT contract and release hardening),
   ``#2340`` (feature tiering), ``#2341`` (correctness and backend parity),
   ``#2342`` (conservative VRT subset), and ``#2344`` (remote / source
   safety hardening). The doc-readiness epic that owns this checklist
   itself is ``#2345``.

How a maintainer runs this gate
===============================

This section is the operational entry point for the release review. Walk it
once per release-tag candidate before signing off.

Selecting the gate suite
------------------------

Every regression test cited in the rows below lives under
``xrspatial/geotiff/tests/``. The gate is the union of those files. The
shortest invocation is:

.. code-block:: bash

   pytest xrspatial/geotiff/tests/

To run only the release-gate-tagged subset that backs this checklist, use
the ``release_gate`` marker. Every gate test lives under a single registry
file (``xrspatial/geotiff/tests/release_gates/test_stable_features.py``)
so the selector picks the registry up exactly:

.. code-block:: bash

   pytest xrspatial/geotiff/tests/release_gates/ -m release_gate

The ``-m release_gate`` selector also works from the wider tests root and
returns the same set of tests.

GPU rows live behind the standard CUDA fixtures. They auto-skip when
``cupy`` or a CUDA device is unavailable, so the same command runs on a
CPU-only host. The GPU rows are tagged ``experimental``; their skip is not
a release blocker (see the decision rule below).

The cross-cutting meta-gates (the consolidated
``xrspatial/geotiff/tests/release_gates/test_stable_features.py`` plus
``test_supported_features_tiers_2137.py``) are part of the same suite.
They fail if a row in this checklist names a feature key that is missing
from :data:`xrspatial.geotiff.SUPPORTED_FEATURES` or a test file that
does not exist.

Handling skipped rows
---------------------

A skipped row is not the same as a passing row. Before signing off:

* Confirm the skip reason is one of: GPU not present, optional codec
  library not installed (``codec.lerc``, ``codec.jpeg2000``, ``codec.j2k``,
  ``codec.lz4``), or the COG validator opt-in
  (``XRSPATIAL_REQUIRE_COG_VALIDATOR=1``) is intentionally off in this
  environment.
* Anything skipped for a reason other than the three above is a blocker.
  Treat ``ImportError``, ``ModuleNotFoundError``, or environment-error
  skips inside the gate suite as failures unless the row is already
  tagged ``experimental``.
* The ``xfail`` rows inside the ``Negative cases`` section of
  ``release_gates/test_stable_features.py`` are intentional pins for
  follow-up work (see the in-file section docstring). A newly-passing
  ``xfail`` is also a signal: the linked follow-up has landed, the row
  should be re-tiered in this PR, and the ``xfail`` marker on the test
  should be removed in the same commit so the gate cannot silently
  regress.

Promote / demote decision rule
------------------------------

Use this rule to decide what to do with a row after the gate suite runs:

* All rows for a tier pass and the row's acceptance statement still matches
  the implementation: keep the tier. No edit required.
* A row currently at ``stable`` regresses (the cited test fails on a fresh
  ``main``-tracking checkout): the release is blocked. Either fix the
  regression in this release or demote the row to ``advanced`` and update
  :data:`xrspatial.geotiff.SUPPORTED_FEATURES` in the same PR. A ``stable``
  row cannot ship red.
* A row at ``advanced`` regresses: the release is not blocked, but the row
  is demoted to ``experimental`` in the same PR. Release notes cannot keep
  promising ``advanced`` while the regression is open.
* A row at ``experimental`` regresses: note it in the release summary; no
  tier change is required. ``experimental`` rows are explicitly not
  release-blocking.
* A row at ``experimental`` or ``advanced`` has been green for a full
  release cycle and the cited test covers every supported backend: it is a
  promotion candidate. The promotion edit goes in a PR that updates both
  the row here and :data:`xrspatial.geotiff.SUPPORTED_FEATURES` together;
  the two cannot drift.

If a row's acceptance statement no longer matches the test (the test was
amended without updating the row), fix the row in this PR before tagging.
A row that lies about its own gate is worse than a missing row.

A note on sub-gate rows
-----------------------

Some rows in the tables below are sub-gates of a broader feature key
(for example ``reader.http`` -- SSRF defense, or ``writer.cog`` --
tile-layout pre-flight). A sub-gate row can carry a stricter tier than
its parent feature row. ``reader.http`` is ``advanced`` because the
HTTP read surface as a whole still has unresolved questions (redirect
handling, retry policy), but the SSRF fail-closed defense inside it is
a hard gate at ``stable``: SSRF cannot regress, even on an ``advanced``
read path. Sub-gate tier stricter than parent is intentional, not
drift, and does not need to be reconciled in an audit pass.

Cross-references
----------------

* Before promoting a feature from ``advanced`` to ``stable``, add a row
  here and update :data:`xrspatial.geotiff.SUPPORTED_FEATURES` in the same
  PR so the docs and the runtime constant agree.
* When deprecating or removing a feature, update both the row here and the
  ``SUPPORTED_FEATURES`` entry in the same PR.
* The parity gate in
  ``xrspatial/geotiff/tests/test_supported_features_tiers_2137.py`` already
  asserts every codec key is tiered; this checklist extends that to the
  reader / writer / VRT / HTTP / GPU surfaces.

Local GeoTIFF read and write
============================

.. list-table::
   :header-rows: 1
   :widths: 18 12 30 28 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - ``reader.local_file``
     - stable
     - Round-trip a local GeoTIFF: pixel bytes, ``transform``, ``crs``, and
       ``nodata`` all survive read.
     - ``xrspatial/geotiff/tests/parity/test_pixel_equality.py``,
       ``xrspatial/geotiff/tests/parity/test_backend_matrix.py``
     - `#2341`_
   * - ``reader.windowed``
     - stable
     - ``open_geotiff(window=(x0, y0, w, h))`` returns the requested
       pixel sub-rectangle for tiled and stripped layouts; out-of-bounds
       and zero-area windows raise rather than silently clamp; coords
       on georeferenced inputs match the eager full-read slice.
     - ``xrspatial/geotiff/tests/unit/test_input_validation.py``,
       ``xrspatial/geotiff/tests/read/test_georef.py``
     - `#2340`_
   * - ``reader.windowed`` -- shifted-transform parity (eager + dask)
     - stable
     - For each representative file, a window strictly interior to the
       raster returns the expected shape, coords that are a bit-exact
       slice of the unwindowed read, an ``attrs['transform']`` equal to
       ``T_full * Affine.translation(col_off, row_off)`` (no float
       drift), and the canonical non-transform release attrs unchanged.
       Covered for both ``open_geotiff(window=...)`` and
       ``read_geotiff_dask(window=...)``.
     - ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
       (windowed-reads section)
     - `#2341`_
   * - ``reader.dask``
     - stable
     - ``open_geotiff(chunks=...)`` returns a Dask-backed
       :class:`xarray.DataArray` that computes to the same pixels,
       coords, and ``attrs`` as the eager numpy read.
     - ``xrspatial/geotiff/tests/parity/test_backend_matrix.py``,
       ``xrspatial/geotiff/tests/parity/test_pixel_equality.py``
     - `#2341`_
   * - ``reader.dask`` -- eager / dask parity
     - stable
     - ``open_geotiff(path)`` and ``read_geotiff_dask(path)`` return the
       same pixels, ``dims``, ``coords``, and the seven release-attr
       keys (``transform``, ``crs``, ``crs_wkt``, ``nodata``,
       ``masked_nodata``, ``georef_status``, ``raster_type``) across
       four scenarios: integer-nodata, float-NaN-nodata, MinIsWhite,
       and the ``mask_nodata=False`` raw-sentinel branch of the
       nodata lifecycle.
     - ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
       (eager / dask full parity section)
     - `#2341`_
   * - ``writer.local_file``
     - stable
     - ``to_geotiff`` writes a file that ``open_geotiff`` reads back
       bit-exact for every stable codec.
     - ``xrspatial/geotiff/tests/write/test_cog.py``,
       ``xrspatial/geotiff/tests/parity/test_backend_matrix.py``
     - `#2341`_
   * - ``writer.overviews``
     - advanced
     - Internal overview IFDs round-trip; the reader can pick a level.
     - ``xrspatial/geotiff/tests/integration/test_dask_pipeline.py``,
       ``xrspatial/geotiff/tests/write/test_overview.py``
     - `#2286`_
   * - ``writer.bigtiff``
     - advanced
     - ``bigtiff=True`` (or auto-promotion above 4 GiB) writes a file with
       magic ``43``, 8-byte offsets, and 20-byte IFD entries.
     - ``xrspatial/geotiff/tests/test_eager_bigtiff_overhead_exact_1905.py``,
       ``xrspatial/geotiff/tests/test_to_geotiff_bigtiff_doc_1683.py``
     - `#2340`_
   * - ``writer.gdal_metadata_xml``
     - experimental
     - ``attrs['gdal_metadata_xml']`` is escaped before serialization and
       does not corrupt the IFD when round-tripped.
     - ``xrspatial/geotiff/tests/unit/test_safe_xml.py``
     - `#2340`_
   * - ``writer.extra_tags``
     - experimental
     - ``attrs['extra_tags']`` filters out reserved tag ids before write.
     - ``xrspatial/geotiff/tests/test_extra_tags_safe_filter_1657.py``
     - `#2340`_
   * - Codec ``none`` / ``deflate`` / ``lzw`` / ``zstd`` / ``packbits``
     - stable
     - Lossless byte-for-byte round-trip on integer and float dtypes.
     - ``xrspatial/geotiff/tests/test_supported_features_tiers_2137.py``,
       ``xrspatial/geotiff/tests/read/test_compression.py``
     - `#2340`_
   * - Stable codec round-trip (read / write / read)
     - stable
     - For every stable codec * promised dtype combination, a full
       write / read / write / read cycle preserves byte-exact pixels
       (NaN-aware for float) and the canonical release attrs. See
       the cited test for the codec, dtype, and attr-key matrix.
     - ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
       (codec round-trip section)
     - `#2341`_
   * - Codec ``lerc`` / ``jpeg2000`` / ``j2k`` / ``lz4``
     - experimental
     - Rejected by default; accepted with
       ``allow_experimental_codecs=True``; emits
       :class:`xrspatial.geotiff.GeoTIFFFallbackWarning` once per call.
     - ``xrspatial/geotiff/tests/test_supported_features_tiers_2137.py``
     - `#2340`_
   * - Codec ``jpeg``
     - internal_only
     - Rejected by default; accepted only with
       ``allow_internal_only_jpeg=True`` (does NOT collapse into the
       ``allow_experimental_codecs`` switch).
     - ``xrspatial/geotiff/tests/test_supported_features_tiers_2137.py``,
       ``xrspatial/geotiff/tests/unit/test_photometric.py``
     - `#2340`_

Cloud-optimized GeoTIFF (COG)
=============================

.. list-table::
   :header-rows: 1
   :widths: 18 12 30 28 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - ``writer.cog``
     - stable
     - ``to_geotiff(cog=True)`` writes an IFD-first tiled file with internal
       overviews that ``rio-cogeo`` accepts (CI-gated by
       ``XRSPATIAL_REQUIRE_COG_VALIDATOR=1``).
     - ``xrspatial/geotiff/tests/write/test_cog.py``
     - `#2286`_
   * - ``reader.local_cog``
     - stable
     - Local COG with overview IFDs decodes byte-for-byte through eager and
       dask paths.
     - ``xrspatial/geotiff/tests/write/test_cog.py``,
       ``xrspatial/geotiff/tests/test_golden_corpus_overview_cog_1930.py``
     - `#2286`_
   * - ``reader.http_cog``
     - advanced
     - Range-request COG read honours the per-tile byte-count cap and the
       SSRF / private-host filter.
     - ``xrspatial/geotiff/tests/integration/test_http_sources.py``
     - `#2344`_
   * - ``writer.bigtiff_cog``
     - advanced
     - BigTIFF + COG combination passes the dedicated compliance suite
       (header magic, IFDs, tile and overview offset tables).
     - ``xrspatial/geotiff/tests/write/test_bigtiff.py``
     - `#2286`_
   * - ``writer.cog`` -- tile-layout pre-flight (``cog=True, tiled=False``)
     - stable
     - Raises ``ValueError`` at the writer entry point regardless of dtype
       or codec.
     - ``xrspatial/geotiff/tests/write/test_cog.py``
     - `#2286`_
   * - ``writer.cog`` -- tile-size pre-flight (non-positive ``tile_size``)
     - stable
     - Non-positive tile sizes raise ``ValueError`` regardless of the
       ``tiled`` flag.
     - ``xrspatial/geotiff/tests/write/test_cog.py``
     - `#2286`_

HTTP / fsspec reads
===================

.. list-table::
   :header-rows: 1
   :widths: 18 12 30 28 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - ``reader.http``
     - advanced
     - ``http://`` / ``https://`` URLs dispatch through ``_HTTPSource`` and
       apply the SSRF / private-host filter; uppercase schemes
       (``HTTP://``, ``HTTPS://``) route the same way (case-insensitive
       scheme routing, ``#2323``).
     - ``xrspatial/geotiff/tests/integration/test_http_sources.py``,
       ``xrspatial/geotiff/tests/test_golden_corpus_http_1930.py``
     - `#2344`_
   * - ``reader.fsspec``
     - advanced
     - Non-HTTP schemes (``s3://``, ``gs://``, ``file://``) dispatch through
       fsspec; HTTP(S) schemes do not silently fall through.
     - ``xrspatial/geotiff/tests/test_golden_corpus_fsspec_1930.py``
     - `#2344`_
   * - ``reader.http`` -- SSRF defense
     - stable
     - URLs resolving to loopback, link-local, or RFC1918 ranges raise
       :class:`xrspatial.geotiff.UnsafeURLError` unless
       ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` is set.
     - ``xrspatial/geotiff/tests/integration/test_http_sources.py``
       (ssrf_hardening and dns_rebinding sections),
       ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
       (HTTP SSRF presence gate)
     - `#2344`_
   * - ``reader.http_cog`` -- per-tile byte-count cap
     - stable
     - Tile or strip declared sizes exceeding ``XRSPATIAL_COG_MAX_TILE_BYTES``
       (default 256 MiB) raise ``ValueError``.
     - ``xrspatial/geotiff/tests/integration/test_http_sources.py``,
       ``xrspatial/geotiff/tests/read/test_tiling.py``
     - `#2344`_
   * - ``max_cloud_bytes`` dispatcher pass-through
     - stable
     - ``open_geotiff(max_cloud_bytes=...)`` forwards to every read backend
       (no silent drop).
     - ``xrspatial/geotiff/tests/integration/test_http_sources.py``
       (max_cloud_bytes_dispatcher and max_cloud_bytes_annot sections)
     - `#2344`_

Nodata lifecycle
================

.. list-table::
   :header-rows: 1
   :widths: 22 12 30 24 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - Nodata round-trip (read -> write)
     - stable
     - The sentinel survives read and write across every backend; integer
       sentinels are preserved bit-exact, float sentinels surface as NaN
       only when ``mask_nodata=True``.
     - ``xrspatial/geotiff/tests/read/test_nodata.py``
     - `#2341`_
   * - ``attrs['masked_nodata']`` lifecycle signal
     - stable
     - ``masked_nodata`` records whether the read produced NaN-masked output
       distinct from the on-disk sentinel; mixed-band VRT inputs honour the
       split.
     - ``xrspatial/geotiff/tests/vrt/test_metadata.py``,
       ``xrspatial/geotiff/tests/test_mask_nodata_gpu_vrt_2052.py``
     - `#2341`_
   * - Mixed-band metadata reject
     - stable
     - Mixed nodata across bands fails closed unless an explicit opt-in
       resolves the ambiguity.
     - ``xrspatial/geotiff/tests/test_ambiguous_metadata_hooks_1987.py``,
       ``xrspatial/geotiff/tests/test_conflicting_crs_write_1987.py``
     - `#2341`_
   * - VRT mixed-band nodata fail-closed
     - stable
     - VRT sources with conflicting per-band nodata raise rather than
       silently flatten.
     - ``xrspatial/geotiff/tests/vrt/test_metadata.py``,
       ``xrspatial/geotiff/tests/vrt/test_dtype_conversion.py``
     - `#2342`_

attrs contract
==============

.. list-table::
   :header-rows: 1
   :widths: 22 12 30 24 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - Contract version stamp
     - stable
     - Every read stamps ``attrs['_xrspatial_geotiff_contract']`` so
       downstream callers can branch on the version.
     - ``xrspatial/geotiff/tests/attrs/test_contract.py``
     - `#2341`_
   * - Canonical attrs after read
     - stable
     - ``transform``, ``crs``, ``crs_wkt``, ``nodata``, ``georef_status``,
       ``raster_type`` appear in canonical form on every backend.
     - ``xrspatial/geotiff/tests/attrs/test_contract.py``,
       ``xrspatial/geotiff/tests/parity/test_backend_matrix.py``
     - `#2341`_
   * - Attrs pass-through on write
     - stable
     - User-supplied attrs survive write round-trips; reserved keys are
       not silently dropped.
     - ``xrspatial/geotiff/tests/attrs/test_contract.py``
     - `#2341`_
   * - ``georef_status`` canonical signal
     - stable
     - ``attrs['georef_status']`` reports whether CRS and transform were
       both parsed, partially parsed, or absent.
     - ``xrspatial/geotiff/tests/attrs/test_contract.py``
     - `#2341`_
   * - ``reader.allow_rotated`` (``allow_rotated=True`` drops ``crs``)
     - experimental
     - Rotated reads surface ``rotated_affine`` and drop ``crs`` so
       downstream math cannot silently mix a rotated grid with an
       axis-aligned CRS.
     - ``xrspatial/geotiff/tests/read/test_crs.py``
     - `#2340`_
   * - ``reader.allow_unparseable_crs``
     - experimental
     - ``allow_unparseable_crs=True`` lets the reader return a DataArray
       when the CRS WKT does not parse; the missing CRS surfaces in
       ``attrs['georef_status']`` rather than silently as a corrupt
       value.
     - ``xrspatial/geotiff/tests/test_crs_fail_closed_1929.py``,
       ``xrspatial/geotiff/tests/test_crs_fail_closed_gpu_1929.py``,
       ``xrspatial/geotiff/tests/test_remaining_fail_closed_1987.py``
     - `#2340`_

VRT supported subset
====================

.. note::

   VRT is the ``advanced`` tier. It covers simple GDAL VRT mosaics over
   GeoTIFF sources with compatible CRS, transform orientation, pixel size,
   dtype, and band count. Warped / reprojection VRTs, mixed CRS without an
   opt-in, nested VRTs, arbitrary resampling beyond the tested subset, and
   complex source / mask / alpha semantics are explicit non-goals (see
   ``#2321``, ``#2342``, and the VRT prose in :ref:`reference.geotiff`).

.. list-table::
   :header-rows: 1
   :widths: 26 12 28 22 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - ``reader.vrt`` -- simple mosaic
     - advanced
     - VRT over compatible GeoTIFF sources returns the same pixels and
       attrs through eager and dask paths.
     - ``xrspatial/geotiff/tests/vrt/test_parity.py``,
       ``xrspatial/geotiff/tests/test_golden_corpus_vrt_1930.py``
     - `#2342`_
   * - VRT default ``missing_sources='raise'``
     - stable
     - Missing source files fail at construction, not at compute.
     - ``xrspatial/geotiff/tests/vrt/test_missing_sources.py``
     - `#2342`_
   * - VRT ``missing_sources='warn'`` opt-in
     - advanced
     - Holes surface as the band sentinel, ``attrs['vrt_holes']`` is set,
       and a :class:`GeoTIFFFallbackWarning` is emitted.
     - ``xrspatial/geotiff/tests/vrt/test_metadata.py``,
       ``xrspatial/geotiff/tests/vrt/test_missing_sources.py``
     - `#2342`_
   * - VRT source / dest rectangle validation
     - stable
     - Out-of-bounds source or destination rectangles raise at construction.
     - ``xrspatial/geotiff/tests/vrt/test_validation.py``,
       ``xrspatial/geotiff/tests/vrt/test_window.py``
     - `#2342`_
   * - VRT path containment
     - stable
     - Relative source paths are constrained to the VRT's directory tree
       and cannot escape via ``..``.
     - ``xrspatial/geotiff/tests/vrt/test_validation.py``
     - `#2344`_
   * - VRT resampling algorithm allow-list
     - advanced
     - Unsupported resampling identifiers are rejected; supported ones
       (``nearest``, ``bilinear``, ``cubic``) round-trip pixels through
       eager and dask.
     - ``xrspatial/geotiff/tests/vrt/test_dtype_conversion.py``,
       ``xrspatial/geotiff/tests/vrt/test_window.py``
     - `#2342`_
   * - VRT dtype / band layout consistency
     - stable
     - Mixed dtype, mixed band count, or mismatched 12-bit-vs-16-bit
       sources raise rather than coerce.
     - ``xrspatial/geotiff/tests/vrt/test_dtype_conversion.py``
     - `#2342`_
   * - VRT lazy / chunked read parity
     - advanced
     - Chunked VRT reads return the same shape, coords, attrs, and values
       as eager reads on the supported subset.
     - ``xrspatial/geotiff/tests/vrt/test_window.py``
     - `#2342`_
   * - VRT single-parse contract
     - stable
     - VRT XML is parsed once per read; chunked callers do not re-parse
       per-chunk.
     - ``xrspatial/geotiff/tests/vrt/test_metadata.py``
     - `#2321`_
   * - VRT narrow exception surface
     - stable
     - VRT-specific failures surface as typed exceptions rather than as
       generic ``Exception``.
     - ``xrspatial/geotiff/tests/vrt/test_validation.py``
     - `#2321`_
   * - VRT presence gate
     - stable
     - At least one regression test exists for every promised VRT
       behaviour (this row is a meta-gate on the rows above).
     - ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
       (VRT presence meta-gate)
     - `#2321`_
   * - ``write_vrt``
     - advanced
     - Writer rejects source-incompatibility cases at the writer boundary.
     - ``xrspatial/geotiff/tests/vrt/test_validation.py``
     - `#2342`_

Sidecar and overview interactions
=================================

.. list-table::
   :header-rows: 1
   :widths: 22 12 30 24 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - ``reader.sidecar_ovr``
     - advanced
     - External ``.tif.ovr`` sidecars produce the same georef status and
       CRS attrs as inline-overview sources.
     - ``xrspatial/geotiff/tests/integration/test_sidecar.py``
       (sidecar_ovr and sidecar_own_geokeys sections)
     - `#2286`_
   * - Overview metadata survival (internal COG and ``.ovr`` sidecar)
     - stable
     - For both internal-COG and external ``.ovr`` sources at factors
       ``[2, 4]``, every overview level agrees with the base on ``crs``,
       ``crs_wkt``, ``georef_status``, ``raster_type``, ``nodata``, and
       ``masked_nodata``; ``transform`` scales pixel size by the level
       factor with the origin preserved. Covered through the eager and
       dask read paths.
     - ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
       (overview / sidecar metadata section)
     - `#2341`_
   * - Remote sidecar byte order
     - stable
     - Sidecar ``.ovr`` files fetched over HTTP honour the sidecar's own
       header byte order, not the parent file's.
     - ``xrspatial/geotiff/tests/integration/test_sidecar.py``
       (remote_sidecar_byte_order section)
     - `#2344`_
   * - Remote sidecar chunked read
     - advanced
     - Chunked dask reads can resolve remote sidecars without
       materializing the full file.
     - ``xrspatial/geotiff/tests/integration/test_sidecar.py``
       (remote_sidecar_chunked section)
     - `#2344`_
   * - Sidecar ``max_cloud_bytes``
     - stable
     - The cloud byte budget applies to sidecar fetches, not just the
       parent file.
     - ``xrspatial/geotiff/tests/integration/test_sidecar.py``
       (sidecar_max_cloud_bytes section)
     - `#2344`_

GPU paths (experimental)
========================

.. note::

   GPU read and write are tagged ``experimental`` in
   :data:`xrspatial.geotiff.SUPPORTED_FEATURES`. Behaviour can change
   without a deprecation window. The tests below pin the documented
   behaviour but a regression here is not a release blocker.

.. list-table::
   :header-rows: 1
   :widths: 22 12 30 24 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - ``reader.gpu``
     - experimental
     - GPU read returns the same pixels and attrs as the CPU path on the
       golden corpus where the GPU path is exercised.
     - ``xrspatial/geotiff/tests/test_golden_corpus_gpu_1930.py``,
       ``xrspatial/geotiff/tests/test_golden_corpus_dask_gpu_1930.py``
     - `#2341`_
   * - ``reader.gpu`` -- fallback warning
     - experimental
     - GPU read errors emit :class:`GeoTIFFFallbackWarning` and fall back
       to CPU unless ``on_gpu_failure='strict'`` or
       ``XRSPATIAL_GEOTIFF_STRICT=1`` is set.
     - ``xrspatial/geotiff/tests/test_gpu_strict_fallback_1516.py``,
       ``xrspatial/geotiff/tests/test_gpu_fallback_forwards_kwargs_2238.py``
     - `#2340`_
   * - ``writer.gpu``
     - experimental
     - GPU write produces a file the CPU reader can decode bit-exact on
       the supported codec subset.
     - ``xrspatial/geotiff/tests/gpu/test_writer.py``
     - `#2340`_
   * - GPU nodata handling
     - experimental
     - Integer and float nodata sentinels survive the GPU read / write
       round-trip.
     - ``xrspatial/geotiff/tests/test_gpu_nodata_1542.py``,
       ``xrspatial/geotiff/tests/read/test_nodata.py``
     - `#2341`_

Internal-only surfaces (not promised)
=====================================

.. list-table::
   :header-rows: 1
   :widths: 22 12 30 24 12

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
     - Epic
   * - Codec ``jpeg``
     - internal_only
     - Lossy 8-bit codec retained for one internal use case. Opt-in via
       ``allow_internal_only_jpeg=True``; not covered by
       ``allow_experimental_codecs``.
     - ``xrspatial/geotiff/tests/unit/test_photometric.py``,
       ``xrspatial/geotiff/tests/test_gpu_jpeg_interop_reject_issue_D_1845.py``
     - `#2340`_

Cross-cutting CI gates
======================

These gates are not tier rows but they back the rest of the checklist.

* ``test_supported_features_tiers_2137.py`` -- every codec in
  ``_VALID_COMPRESSIONS`` has a ``SUPPORTED_FEATURES`` tier, and the writer
  rejects experimental and internal-only codecs without their respective
  opt-in flags. Owning epic: `#2340`_.
* ``parity/test_backend_matrix.py`` and
  ``parity/test_pixel_equality.py`` -- cross-backend pixel and
  metadata parity across the 4 read backends (numpy, cupy, dask+numpy,
  dask+cupy) on the golden corpus. Owning epic: `#2341`_.
* ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
  (``Cross-cutting meta-gates`` section) -- meta-gate that asserts every
  promised VRT behaviour in this checklist resolves to a real test file
  and a real ``SUPPORTED_FEATURES`` entry. Owning epic: `#2321`_.
* ``xrspatial/geotiff/tests/release_gates/test_stable_features.py``
  (``Negative cases`` section) -- negative cross-cutting gate. Pins that
  ambiguous metadata fails closed at every promised read entry point:
  conflicting CRS between header and ``.aux.xml`` PAM sidecar (xfail
  until PAM sidecar support lands), integer nodata sentinel that cannot
  be honoured on a float-promoted raster (xfail against ``#1774``
  follow-up), rotated transform without ``allow_rotated=True`` uniformly
  across eager / dask / windowed paths, and mixed-tier VRT children when
  stable-only is requested (xfail against epic `#2342`_). Owning epic:
  `#2341`_.

Owning epics
============

The release gate rows above point at one of the following epics. Each
link resolves on GitHub; the title is kept here so a reader walking the
file does not need to leave the page to confirm scope.

.. _#2286: https://github.com/xarray-contrib/xarray-spatial/issues/2286
.. _#2321: https://github.com/xarray-contrib/xarray-spatial/issues/2321
.. _#2340: https://github.com/xarray-contrib/xarray-spatial/issues/2340
.. _#2341: https://github.com/xarray-contrib/xarray-spatial/issues/2341
.. _#2342: https://github.com/xarray-contrib/xarray-spatial/issues/2342
.. _#2344: https://github.com/xarray-contrib/xarray-spatial/issues/2344

* `#2286`_ -- Promote COG support to ready / stable with compliance and
  parity gates.
* `#2321`_ -- GeoTIFF release hardening: define and lock down supported
  VRT contract.
* `#2340`_ -- GeoTIFF release contract and feature tiering.
* `#2341`_ -- GeoTIFF correctness and backend parity release gate.
* `#2342`_ -- Conservative VRT support contract for GeoTIFF release.
* `#2344`_ -- GeoTIFF remote / source safety hardening.
