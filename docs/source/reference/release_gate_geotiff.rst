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

   See parent issue ``#2321`` for the release hardening epic. Sub-PRs 1
   through 5 contribute the VRT contract, the centralized validator, the
   metadata parity tests, the backend parity matrix, and the case-insensitive
   HTTP scheme routing that this checklist references.

How to use this page
====================

* Before tagging a release, walk every row and confirm the cited regression
  test still exists and still runs in CI.
* When promoting a feature from ``advanced`` to ``stable``, add a row here
  and update :data:`xrspatial.geotiff.SUPPORTED_FEATURES` in the same PR so
  the docs and the runtime constant agree.
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
   :widths: 20 15 35 30

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
   * - ``reader.local_file``
     - stable
     - Round-trip a local GeoTIFF: pixel bytes, ``transform``, ``crs``, and
       ``nodata`` all survive read.
     - ``xrspatial/geotiff/tests/test_backend_pixel_parity_matrix_1813.py``,
       ``xrspatial/geotiff/tests/test_backend_parity_matrix.py``
   * - ``writer.local_file``
     - stable
     - ``to_geotiff`` writes a file that ``open_geotiff`` reads back
       bit-exact for every stable codec.
     - ``xrspatial/geotiff/tests/test_cog_writer_compliance.py``,
       ``xrspatial/geotiff/tests/test_attrs_finalization_parity_2211.py``
   * - ``writer.overviews``
     - advanced
     - Internal overview IFDs round-trip; the reader can pick a level.
     - ``xrspatial/geotiff/tests/test_dask_overview_level.py``,
       ``xrspatial/geotiff/tests/test_cog_overview_nodata_1613.py``
   * - ``writer.bigtiff``
     - advanced
     - ``bigtiff=True`` (or auto-promotion above 4 GiB) writes a file with
       magic ``43``, 8-byte offsets, and 20-byte IFD entries.
     - ``xrspatial/geotiff/tests/test_eager_bigtiff_overhead_exact_1905.py``,
       ``xrspatial/geotiff/tests/test_to_geotiff_bigtiff_doc_1683.py``
   * - ``writer.gdal_metadata_xml``
     - experimental
     - ``attrs['gdal_metadata_xml']`` is escaped before serialization and
       does not corrupt the IFD when round-tripped.
     - ``xrspatial/geotiff/tests/test_gdal_metadata_xml_escape_1614.py``
   * - ``writer.extra_tags``
     - experimental
     - ``attrs['extra_tags']`` filters out reserved tag ids before write.
     - ``xrspatial/geotiff/tests/test_extra_tags_safe_filter_1657.py``
   * - Codec ``none`` / ``deflate`` / ``lzw`` / ``zstd`` / ``packbits``
     - stable
     - Lossless byte-for-byte round-trip on integer and float dtypes.
     - ``xrspatial/geotiff/tests/test_supported_features_tiers_2137.py``,
       ``xrspatial/geotiff/tests/test_compression.py``
   * - Codec ``lerc`` / ``jpeg2000`` / ``j2k`` / ``lz4``
     - experimental
     - Rejected by default; accepted with
       ``allow_experimental_codecs=True``; emits
       :class:`xrspatial.geotiff.GeoTIFFFallbackWarning` once per call.
     - ``xrspatial/geotiff/tests/test_supported_features_tiers_2137.py``
   * - Codec ``jpeg``
     - internal_only
     - Rejected by default; accepted only with
       ``allow_internal_only_jpeg=True`` (does NOT collapse into the
       ``allow_experimental_codecs`` switch).
     - ``xrspatial/geotiff/tests/test_supported_features_tiers_2137.py``,
       ``xrspatial/geotiff/tests/test_to_geotiff_allow_internal_only_jpeg_parity.py``

Cloud-optimized GeoTIFF (COG)
=============================

.. list-table::
   :header-rows: 1
   :widths: 20 15 35 30

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
   * - ``writer.cog``
     - stable
     - ``to_geotiff(cog=True)`` writes an IFD-first tiled file with internal
       overviews that ``rio-cogeo`` accepts (CI-gated by
       ``XRSPATIAL_REQUIRE_COG_VALIDATOR=1``).
     - ``xrspatial/geotiff/tests/test_cog_writer_compliance.py``,
       ``xrspatial/geotiff/tests/test_cog_parity_2286.py``
   * - ``reader.local_cog``
     - stable
     - Local COG with overview IFDs decodes byte-for-byte through eager and
       dask paths.
     - ``xrspatial/geotiff/tests/test_cog.py``,
       ``xrspatial/geotiff/tests/test_golden_corpus_overview_cog_1930.py``
   * - ``reader.http_cog``
     - advanced
     - Range-request COG read honours the per-tile byte-count cap and the
       SSRF / private-host filter.
     - ``xrspatial/geotiff/tests/test_cog_http_concurrent.py``,
       ``xrspatial/geotiff/tests/test_cog_http_parallel_decode_2026_05_15.py``,
       ``xrspatial/geotiff/tests/test_cog_http_close_on_error_1816.py``
   * - ``writer.bigtiff_cog``
     - advanced
     - BigTIFF + COG combination passes the dedicated compliance suite
       (header magic, IFDs, tile and overview offset tables).
     - ``xrspatial/geotiff/tests/test_bigtiff_cog_compliance_2286.py``
   * - ``to_geotiff(cog=True, tiled=False)``
     - rejected at writer boundary
     - Raises ``ValueError`` at the writer entry point regardless of dtype
       or codec.
     - ``xrspatial/geotiff/tests/test_cog_requires_tiled_2312.py``
   * - ``to_geotiff(cog=True, tile_size=<= 0>)``
     - rejected at writer boundary
     - Non-positive tile sizes raise ``ValueError`` regardless of the
       ``tiled`` flag.
     - ``xrspatial/geotiff/tests/test_cog_tile_size_hang_2311.py``

HTTP / fsspec reads
===================

.. list-table::
   :header-rows: 1
   :widths: 20 15 35 30

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
   * - ``reader.http``
     - advanced
     - ``http://`` / ``https://`` URLs dispatch through ``_HTTPSource`` and
       apply the SSRF / private-host filter; uppercase schemes
       (``HTTP://``, ``HTTPS://``) route the same way (see ``#2321``
       sub-PR 5).
     - ``xrspatial/geotiff/tests/test_http_read_all_bounded_2051.py``,
       ``xrspatial/geotiff/tests/test_golden_corpus_http_1930.py``,
       ``xrspatial/geotiff/tests/test_http_dask_allow_rotated_2130.py``
   * - ``reader.fsspec``
     - advanced
     - Non-HTTP schemes (``s3://``, ``gs://``, ``file://``) dispatch through
       fsspec; HTTP(S) schemes do not silently fall through.
     - ``xrspatial/geotiff/tests/test_golden_corpus_fsspec_1930.py``
   * - HTTP SSRF defense
     - stable
     - URLs resolving to loopback, link-local, or RFC1918 ranges raise
       :class:`xrspatial.geotiff.UnsafeURLError` unless
       ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` is set.
     - ``xrspatial/geotiff/tests/test_ssrf_hardening_1664.py``,
       ``xrspatial/geotiff/tests/test_dns_rebinding_pin_issue_1846.py``,
       ``xrspatial/geotiff/tests/test_release_gate_2321.py``
       (HTTP SSRF presence gate)
   * - HTTP per-tile byte-count cap
     - stable
     - Tile or strip declared sizes exceeding ``XRSPATIAL_COG_MAX_TILE_BYTES``
       (default 256 MiB) raise ``ValueError``.
     - ``xrspatial/geotiff/tests/test_cloud_read_byte_limit_1928.py``,
       ``xrspatial/geotiff/tests/test_gpu_tile_byte_cap_2026_05_18.py``
   * - ``max_cloud_bytes`` dispatcher pass-through
     - stable
     - ``open_geotiff(max_cloud_bytes=...)`` forwards to every read backend
       (no silent drop).
     - ``xrspatial/geotiff/tests/test_max_cloud_bytes_dispatcher_silent_drop_2026_05_15.py``,
       ``xrspatial/geotiff/tests/test_open_geotiff_max_cloud_bytes_annot_2106.py``

Nodata lifecycle
================

.. list-table::
   :header-rows: 1
   :widths: 25 15 35 25

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
   * - Nodata round-trip (read -> write)
     - stable
     - The sentinel survives read and write across every backend; integer
       sentinels are preserved bit-exact, float sentinels surface as NaN
       only when ``mask_nodata=True``.
     - ``xrspatial/geotiff/tests/test_nodata_lifecycle_attrs_2135.py``,
       ``xrspatial/geotiff/tests/test_nodata_lifecycle_parity_2211.py``
   * - ``attrs['masked_nodata']`` lifecycle signal
     - stable
     - ``masked_nodata`` records whether the read produced NaN-masked output
       distinct from the on-disk sentinel; mixed-band VRT inputs honour the
       split.
     - ``xrspatial/geotiff/tests/test_vrt_masked_nodata_attr_2159.py``,
       ``xrspatial/geotiff/tests/test_mask_nodata_gpu_vrt_2052.py``
   * - Mixed-band metadata reject
     - stable
     - Mixed nodata across bands fails closed unless an explicit opt-in
       resolves the ambiguity.
     - ``xrspatial/geotiff/tests/test_ambiguous_metadata_hooks_1987.py``,
       ``xrspatial/geotiff/tests/test_conflicting_crs_write_1987.py``
   * - VRT mixed-band nodata
     - rejected
     - VRT sources with conflicting per-band nodata raise rather than
       silently flatten.
     - ``xrspatial/geotiff/tests/test_vrt_band_nodata_1598.py``,
       ``xrspatial/geotiff/tests/test_vrt_int_nodata_1564.py``,
       ``xrspatial/geotiff/tests/test_vrt_multiband_int_nodata_1611.py``

attrs contract
==============

.. list-table::
   :header-rows: 1
   :widths: 25 15 35 25

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
   * - Contract version stamp
     - stable
     - Every read stamps ``attrs['_xrspatial_geotiff_contract']`` so
       downstream callers can branch on the version.
     - ``xrspatial/geotiff/tests/test_attrs_contract_version_1984.py``
   * - Canonical attrs after read
     - stable
     - ``transform``, ``crs``, ``crs_wkt``, ``nodata``, ``georef_status``,
       ``raster_type`` appear in canonical form on every backend.
     - ``xrspatial/geotiff/tests/test_attrs_contract_canonical_1984.py``,
       ``xrspatial/geotiff/tests/test_attrs_parity_1548.py``
   * - Attrs pass-through on write
     - stable
     - User-supplied attrs survive write round-trips; reserved keys are
       not silently dropped.
     - ``xrspatial/geotiff/tests/test_attrs_contract_passthrough_1984.py``,
       ``xrspatial/geotiff/tests/test_attrs_contract_aliases_1984.py``
   * - ``georef_status`` canonical signal
     - stable
     - ``attrs['georef_status']`` reports whether CRS and transform were
       both parsed, partially parsed, or absent.
     - ``xrspatial/geotiff/tests/test_attrs_contract_canonical_1984.py``
   * - ``reader.allow_rotated`` (``allow_rotated=True`` drops ``crs``)
     - advanced
     - Rotated reads surface ``rotated_affine`` and drop ``crs`` so
       downstream math cannot silently mix a rotated grid with an
       axis-aligned CRS.
     - ``xrspatial/geotiff/tests/test_allow_rotated_crs_drop_2126.py``,
       ``xrspatial/geotiff/tests/test_allow_rotated_no_crs_2122.py``,
       ``xrspatial/geotiff/tests/test_allow_rotated_geotiff_2115.py``
   * - ``reader.allow_unparseable_crs``
     - advanced
     - ``allow_unparseable_crs=True`` lets the reader return a DataArray
       when the CRS WKT does not parse; the missing CRS surfaces in
       ``attrs['georef_status']`` rather than silently as a corrupt
       value.
     - ``xrspatial/geotiff/tests/test_crs_fail_closed_1929.py``,
       ``xrspatial/geotiff/tests/test_crs_fail_closed_gpu_1929.py``,
       ``xrspatial/geotiff/tests/test_remaining_fail_closed_1987.py``

VRT supported subset
====================

.. note::

   VRT is the ``advanced`` tier. It covers simple GDAL VRT mosaics over
   GeoTIFF sources with compatible CRS, transform orientation, pixel size,
   dtype, and band count. Warped / reprojection VRTs, mixed CRS without an
   opt-in, nested VRTs, arbitrary resampling beyond the tested subset, and
   complex source / mask / alpha semantics are explicit non-goals (see
   ``#2321`` and the VRT prose in :ref:`reference.geotiff`).

.. list-table::
   :header-rows: 1
   :widths: 30 15 30 25

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
   * - ``reader.vrt`` -- simple mosaic
     - advanced
     - VRT over compatible GeoTIFF sources returns the same pixels and
       attrs through eager and dask paths.
     - ``xrspatial/geotiff/tests/test_vrt_backend_coverage_2026_05_11.py``,
       ``xrspatial/geotiff/tests/test_golden_corpus_vrt_1930.py``,
       ``xrspatial/geotiff/tests/test_vrt_finalization_parity_2162.py``
   * - VRT default ``missing_sources='raise'``
     - stable
     - Missing source files fail at construction, not at compute.
     - ``xrspatial/geotiff/tests/test_vrt_missing_sources_default_raise_1843.py``,
       ``xrspatial/geotiff/tests/test_read_vrt_default_missing_sources_1860.py``,
       ``xrspatial/geotiff/tests/test_vrt_chunked_missing_raise_at_build_2265.py``
   * - VRT ``missing_sources='warn'`` opt-in
     - advanced
     - Holes surface as the band sentinel, ``attrs['vrt_holes']`` is set,
       and a :class:`GeoTIFFFallbackWarning` is emitted.
     - ``xrspatial/geotiff/tests/test_vrt_holes_attr_1734.py``,
       ``xrspatial/geotiff/tests/test_vrt_missing_sources_policy_1799.py``,
       ``xrspatial/geotiff/tests/test_vrt_chunked_missing_sources_1799.py``
   * - VRT source / dest rectangle validation
     - stable
     - Out-of-bounds source or destination rectangles raise at construction.
     - ``xrspatial/geotiff/tests/test_geotiff_vrt_srcrect_validation_1784.py``,
       ``xrspatial/geotiff/tests/test_vrt_scaled_rects_1694.py``,
       ``xrspatial/geotiff/tests/test_vrt_dstrect_resample_cap_1737.py``
   * - VRT path containment
     - stable
     - Relative source paths are constrained to the VRT's directory tree
       and cannot escape via ``..``.
     - ``xrspatial/geotiff/tests/test_vrt_path_containment_1671.py``
   * - VRT resampling algorithm allow-list
     - advanced
     - Unsupported resampling identifiers are rejected; supported ones
       (``nearest``, ``bilinear``, ``cubic``) round-trip pixels through
       eager and dask.
     - ``xrspatial/geotiff/tests/test_vrt_resample_alg_1751.py``,
       ``xrspatial/geotiff/tests/test_vrt_resample_window_inverse_1704.py``
   * - VRT dtype / band layout consistency
     - stable
     - Mixed dtype, mixed band count, or mismatched 12-bit-vs-16-bit
       sources raise rather than coerce.
     - ``xrspatial/geotiff/tests/test_vrt_dtype_1783.py``,
       ``xrspatial/geotiff/tests/test_vrt_dtype_12bit_1914.py``,
       ``xrspatial/geotiff/tests/test_vrt_multiband_dtype_1696.py``
   * - VRT lazy / chunked read parity
     - advanced
     - Chunked VRT reads return the same shape, coords, attrs, and values
       as eager reads on the supported subset.
     - ``xrspatial/geotiff/tests/test_vrt_lazy_chunks_1814.py``,
       ``xrspatial/geotiff/tests/test_read_vrt_lazy_chunks_1798.py``,
       ``xrspatial/geotiff/tests/test_vrt_chunked_shared_dataset_1923.py``
   * - VRT single-parse contract
     - stable
     - VRT XML is parsed once per read; chunked callers do not re-parse
       per-chunk.
     - ``xrspatial/geotiff/tests/test_vrt_single_parse_1825.py``
   * - VRT narrow exception surface
     - stable
     - VRT-specific failures surface as typed exceptions rather than as
       generic ``Exception``.
     - ``xrspatial/geotiff/tests/test_vrt_narrow_except_1670.py``
   * - VRT presence gate
     - stable
     - At least one regression test exists for every promised VRT
       behaviour (this row is a meta-gate on the rows above).
     - ``xrspatial/geotiff/tests/test_release_gate_2321.py``
   * - ``write_vrt``
     - advanced
     - Writer rejects source-incompatibility cases at the writer boundary.
     - ``xrspatial/geotiff/tests/test_to_geotiff_vrt_tiled_validation_1862.py``

Sidecar and overview interactions
=================================

.. list-table::
   :header-rows: 1
   :widths: 25 15 35 25

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
   * - ``reader.sidecar_ovr``
     - advanced
     - External ``.tif.ovr`` sidecars produce the same georef status and
       CRS attrs as inline-overview sources.
     - ``xrspatial/geotiff/tests/test_sidecar_ovr_2112.py``,
       ``xrspatial/geotiff/tests/test_sidecar_own_geokeys_2315.py``
   * - Remote sidecar byte order
     - stable
     - Sidecar ``.ovr`` files fetched over HTTP honour the sidecar's own
       header byte order, not the parent file's.
     - ``xrspatial/geotiff/tests/test_remote_sidecar_byte_order_2314.py``
   * - Remote sidecar chunked read
     - advanced
     - Chunked dask reads can resolve remote sidecars without
       materializing the full file.
     - ``xrspatial/geotiff/tests/test_remote_sidecar_chunked_2239.py``
   * - Sidecar ``max_cloud_bytes``
     - stable
     - The cloud byte budget applies to sidecar fetches, not just the
       parent file.
     - ``xrspatial/geotiff/tests/test_sidecar_max_cloud_bytes_2121.py``

GPU paths (experimental)
========================

.. note::

   GPU read and write are tagged ``experimental`` in
   :data:`xrspatial.geotiff.SUPPORTED_FEATURES`. Behaviour can change
   without a deprecation window. The tests below pin the documented
   behaviour but a regression here is not a release blocker.

.. list-table::
   :header-rows: 1
   :widths: 25 15 35 25

   * - Feature
     - Tier
     - One-line acceptance
     - Regression test
   * - ``reader.gpu``
     - experimental
     - GPU read returns the same pixels and attrs as the CPU path on the
       golden corpus where the GPU path is exercised.
     - ``xrspatial/geotiff/tests/test_golden_corpus_gpu_1930.py``,
       ``xrspatial/geotiff/tests/test_golden_corpus_dask_gpu_1930.py``
   * - GPU fallback warning
     - experimental
     - GPU read errors emit :class:`GeoTIFFFallbackWarning` and fall back
       to CPU unless ``on_gpu_failure='strict'`` or
       ``XRSPATIAL_GEOTIFF_STRICT=1`` is set.
     - ``xrspatial/geotiff/tests/test_gpu_strict_fallback_1516.py``,
       ``xrspatial/geotiff/tests/test_gpu_fallback_forwards_kwargs_2238.py``
   * - ``writer.gpu``
     - experimental
     - GPU write produces a file the CPU reader can decode bit-exact on
       the supported codec subset.
     - ``xrspatial/geotiff/tests/test_gpu_writer_attrs_1563.py``,
       ``xrspatial/geotiff/tests/test_to_geotiff_gpu_fallback_1674.py``
   * - GPU nodata handling
     - experimental
     - Integer and float nodata sentinels survive the GPU read / write
       round-trip.
     - ``xrspatial/geotiff/tests/test_gpu_nodata_1542.py``,
       ``xrspatial/geotiff/tests/test_apply_nodata_mask_gpu_inplace_1934.py``

Internal-only surfaces (not promised)
=====================================

.. list-table::
   :header-rows: 1
   :widths: 25 15 35 25

   * - Feature
     - Tier
     - Note
     - Regression test
   * - Codec ``jpeg``
     - internal_only
     - Lossy 8-bit codec retained for one internal use case. Opt-in via
       ``allow_internal_only_jpeg=True``; not covered by
       ``allow_experimental_codecs``.
     - ``xrspatial/geotiff/tests/test_to_geotiff_allow_internal_only_jpeg_parity.py``,
       ``xrspatial/geotiff/tests/test_gpu_jpeg_interop_reject_issue_D_1845.py``

Cross-cutting CI gates
======================

These gates are not tier rows but they back the rest of the checklist.

* ``test_supported_features_tiers_2137.py`` -- every codec in
  ``_VALID_COMPRESSIONS`` has a ``SUPPORTED_FEATURES`` tier, and the writer
  rejects experimental and internal-only codecs without their respective
  opt-in flags.
* ``test_backend_parity_matrix.py`` and
  ``test_backend_pixel_parity_matrix_1813.py`` -- cross-backend pixel and
  metadata parity across the 4 read backends (numpy, cupy, dask+numpy,
  dask+cupy) on the golden corpus.
* ``test_release_gate_2321.py`` -- meta-gate that asserts every promised
  VRT behaviour in this checklist resolves to a real test file and a real
  ``SUPPORTED_FEATURES`` entry.

Placeholder PR cross-references
===============================

The sub-PRs of ``#2321`` cited in this checklist are listed below. When a
sub-PR lands its number replaces the placeholder, both here and in the
parent issue's tracking comment.

* Sub-PR 1 (publish the VRT contract): ``(see #2321)``
* Sub-PR 2 (centralize VRT capability validation): ``(see #2321)``
* Sub-PR 3 (VRT metadata parity tests): ``(see #2321)``
* Sub-PR 4 (backend parity for VRT + sidecar / overview): ``(see #2321)``
* Sub-PR 5 (case-insensitive HTTP(S) scheme routing): #2326
* Sub-PR 6 (this release gate / audit checklist): tracked by sub-issue
  ``#2331``.
