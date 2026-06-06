..  _user_guide.geotiff_safe_io:

***********************
Safe GeoTIFF IO usage
***********************

This page is the user-facing answer to "is this safe to rely on?" for
:mod:`xrspatial.geotiff`. It explains which entry points to prefer,
how to read the tier vocabulary the module publishes, which codecs and
COG combinations sit inside the stable contract, the fail-closed errors
a caller will hit, and the env vars / kwargs that bound remote reads.

The page does not claim full GDAL / VRT / GPU parity. Where a feature
is tested but the public surface is not yet pinned, it is called out as
``advanced`` or ``experimental`` and a caller should treat it as such.

.. contents:: On this page
   :local:
   :depth: 2


Entry points
============

The public IO surface lives at ``xrspatial.geotiff``. Five names cover
the read and write paths:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry point
     - What it does
   * - :func:`xrspatial.geotiff.open_geotiff`
     - The read entry point. A path or a binary file-like is the only
       required argument. Pass ``chunks=N`` for a dask-backed lazy
       read; pass ``gpu=True`` for a CuPy-backed eager read; combine
       both for a dask + CuPy read. Returns a 2D
       :class:`xarray.DataArray` for single-band input and a 3D one for
       multi-band input. The binary file-like form is restricted to the
       eager numpy reader; dask, GPU, VRT, and remote-URL paths require
       a string. A ``.vrt`` source reads a GDAL mosaic (tier:
       ``advanced``) over a documented subset of the GDAL VRT schema;
       unsupported features raise ``VRTUnsupportedError`` or
       :class:`xrspatial.geotiff.UnsupportedGeoTIFFFeatureError` at
       graph-build time rather than producing wrong pixels. Both error
       classes live in :mod:`xrspatial.geotiff._errors`.
   * - :func:`xrspatial.geotiff.to_geotiff`
     - Write a DataArray to a local path. Pass ``cog=True`` for a
       Cloud-optimized GeoTIFF layout. Pass ``allow_experimental_codecs=True``
       to opt into ``lerc``, ``jpeg2000`` / ``j2k``, or ``lz4``; pass
       ``allow_internal_only_jpeg=True`` to opt into the
       internal-only ``jpeg`` codec. Pass ``gpu=True`` (or pass
       CuPy-backed data) for the GPU writer (tier: ``experimental``);
       use the CPU path for anything you round-trip through external
       tools.

A dask-backed read is just ``open_geotiff(source, chunks=...)`` -- there
is no separate ``read_geotiff_dask`` name on the public surface. The
internal helper exists for backend wiring; callers should go through
``open_geotiff``.


Tier vocabulary
===============

:data:`xrspatial.geotiff.SUPPORTED_FEATURES` is a dict that maps every
feature name on the public surface to one of four tier strings. Read
the tier before depending on a feature in production:

* ``stable`` -- the path a new user should be on. Covered by the
  cross-backend parity matrix and a release-gate test. A regression
  here fails CI. Safe to rely on for the supported release.
* ``advanced`` -- works and is tested, but the caller should know what
  they are signing up for. Cloud cost, partial VRT mosaics, rotated
  transforms dropping on write, BigTIFF promotion, and ``.tif.ovr``
  sidecar discovery all live here. No kwarg gate; the docstring
  carries an ``Advanced:`` marker.
* ``experimental`` -- works in our tests, no claim about external
  interop or numerical parity across backends. GPU read and write,
  rotated-transform escape hatches, and Tier 3 codecs sit here. Tier 3
  codecs additionally require ``allow_experimental_codecs=True`` on the
  writer.
* ``internal_only`` -- the strictest tier. The output does not round-trip
  through libtiff / GDAL / rasterio. ``codec.jpeg`` is the only entry
  today and requires its own ``allow_internal_only_jpeg=True`` opt-in;
  ``allow_experimental_codecs`` does not cover it.

To check a feature at runtime::

    from xrspatial.geotiff import SUPPORTED_FEATURES

    if SUPPORTED_FEATURES.get('writer.cog') != 'stable':
        # The release you are on has not promoted COG writes.
        # Fall back to a plain GeoTIFF write or pin a known release.
        ...

The full tier map and the rationale for each entry live in
:ref:`reference.geotiff_release_contract`. The release-gate audit table
that ties each ``stable`` promise to a regression test lives in
:ref:`reference.geotiff_release_gate`.


Recommended codecs
==================

Five codecs are tagged ``stable`` and form the lossless contract:

* ``none`` -- no compression (``COMPRESSION_NONE`` in the TIFF spec).
* ``deflate`` -- DEFLATE.
* ``lzw`` -- LZW.
* ``packbits`` -- PackBits.
* ``zstd`` -- Zstandard.

Each of these is lossless and round-trips byte-for-byte for integer and
float dtypes through the CPU writer and CPU reader. If you do not have
a reason to pick something else, write with one of these.

The following codecs are tagged ``experimental`` and require
``allow_experimental_codecs=True`` on :func:`xrspatial.geotiff.to_geotiff`:

* ``lerc`` -- Limited Error Raster Compression.
* ``jpeg2000`` and ``j2k`` -- JPEG 2000.
* ``lz4`` -- LZ4.

The ``jpeg`` codec is tagged ``internal_only``. It does not round-trip
through libtiff / GDAL / rasterio and the writer rejects it unless the
caller passes ``allow_internal_only_jpeg=True``. The general
``allow_experimental_codecs=True`` flag does not unlock it.

A file falls outside the stable codec contract whenever it uses a
non-``stable`` codec, or whenever it is read or written through a
non-``stable`` path (GPU, BigTIFF COG, HTTP COG, file-like destinations
with ``cog=True``).


COG output
==========

Pass ``cog=True`` to :func:`xrspatial.geotiff.to_geotiff` to write a
Cloud-optimized GeoTIFF. The writer emits an IFD-first, tiled layout
with internal overviews using a lossless codec.

The stable COG contract covers:

* Axis-aligned 2D / 3D rasters.
* CPU writer and CPU reader paths (``writer.cog`` and
  ``reader.local_cog`` are both ``stable``).
* Stable codecs only.
* Internal overviews only -- no ``.tif.ovr`` sidecars in the stable
  layout.
* Normal CRS, transform, dtype, nodata, band, and
  pixel-is-area / pixel-is-point round-trip.

The following combinations stay outside the stable contract even when
``cog=True`` is set:

* GPU COG read or write -- ``writer.gpu`` and ``reader.gpu`` are
  ``experimental``.
* Experimental codecs (``lerc``, ``jpeg2000`` / ``j2k``, ``lz4``) and
  the internal-only ``jpeg`` codec.
* Rotated transforms -- read-side ``allow_rotated=True`` is
  ``experimental``, and the writer drops rotation terms on round-trip.
* External ``.tif.ovr`` sidecars (``reader.sidecar_ovr`` is
  ``advanced``).
* File-like destinations with ``cog=True``.
* BigTIFF COG (``writer.bigtiff_cog`` is ``advanced``).
* HTTP / range COG (``reader.http_cog`` is ``advanced``).

If your pipeline relies on any of these, pin the xrspatial release and
treat the behaviour as opt-in rather than as part of the stable
contract.


Fail-closed errors
==================

The reader and writer raise typed errors instead of guessing when the
input is ambiguous or unsupported. The hierarchy lives in
:mod:`xrspatial.geotiff`. Every entry below subclasses
:class:`ValueError`, so existing ``except ValueError`` callers keep
catching them. Every entry except
:class:`~xrspatial.geotiff.UnsupportedGeoTIFFFeatureError` also subclasses
:class:`~xrspatial.geotiff.GeoTIFFAmbiguousMetadataError`, which catches
the ambiguous-metadata family at once.
:class:`~xrspatial.geotiff.UnsupportedGeoTIFFFeatureError` is a direct
``ValueError`` subclass and sits outside that family on purpose --
"we refuse this input" is distinct from "the input is malformed".

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Error
     - Meaning
     - Opt-in
   * - :class:`~xrspatial.geotiff.InvalidCRSCodeError`
     - The CRS code does not resolve to a known authority entry.
     - Pass a valid EPSG code or full WKT.
   * - :class:`~xrspatial.geotiff.UnparseableCRSError`
     - The CRS string cannot be parsed as WKT or an authority code.
     - ``allow_unparseable_crs=True`` (experimental).
   * - :class:`~xrspatial.geotiff.RotatedTransformError`
     - The affine transform has non-zero rotation / shear terms.
     - ``allow_rotated=True`` (experimental). The opt-in returns the
       pixel grid without the geospatial assumption.
   * - :class:`~xrspatial.geotiff.NonUniformCoordsError`
     - The DataArray coords on write imply a non-uniform pixel grid.
     - Regrid the array to uniform spacing first.
   * - :class:`~xrspatial.geotiff.MixedBandMetadataError`
     - A VRT declares conflicting per-band metadata (most often
       disagreeing nodata sentinels).
     - ``band_nodata='first'`` to keep the legacy "use band 0" behaviour
       explicitly.
   * - :class:`~xrspatial.geotiff.ConflictingCRSError`
     - ``attrs['crs']`` and ``attrs['crs_wkt']`` do not canonicalise to
       the same WKT on write.
     - Resolve the conflict in caller code before writing.
   * - :class:`~xrspatial.geotiff.InconsistentGeoKeysError`
     - The source's GeoKey directory is internally contradictory:
       ``ModelTypeGeoKey`` disagrees with the type-specific keys
       actually populated (a projected model with only
       ``GeographicTypeGeoKey``, or a geographic model carrying
       ``ProjectedCSTypeGeoKey``). A projected file that also names its
       base geographic CRS is the normal shape and reads without error.
     - No opt-in. Fix the source's GeoKey directory so the model type
       matches the CRS codes it declares.
   * - :class:`~xrspatial.geotiff.ConflictingNodataError`
     - ``attrs['nodata']`` and ``attrs['nodatavals']`` disagree on
       write.
     - Resolve in caller code; the writer will not pick one silently.
   * - ``VRTUnsupportedError``
     - The parsed VRT declares a feature the read pipeline does not
       honour (CRS / dtype / band / nodata / transform / pixel-size /
       window / resampling mismatch).
     - No opt-in. Either fix the VRT or read the sources directly.
   * - :class:`~xrspatial.geotiff.UnknownCRSModelTypeError`
     - The writer cannot classify an EPSG code as geographic or
       projected.
     - Pass a code pyproj can resolve, or install pyproj.
   * - :class:`~xrspatial.geotiff.NonRepresentableEPSGCRSError`
     - The integer EPSG code resolves to a compound (horizontal +
       vertical) CRS, which the writer cannot represent in a single
       ``GeographicTypeGeoKey`` or ``ProjectedCSTypeGeoKey`` slot.
     - Pass the full compound CRS as WKT to take the user-defined CRS
       fallback path, or pass the horizontal sub-CRS EPSG directly if
       the vertical component is not needed.
   * - :class:`~xrspatial.geotiff.UnsupportedGeoTIFFFeatureError`
     - The input declares a feature the GeoTIFF module does not
       implement (warped / reprojection VRTs, pansharpened or derived
       VRT subclasses, non-zero skew on a VRT mosaic source transform,
       and so on).
     - No opt-in. The error message names the feature and the source
       that triggered it.
   * - :class:`~xrspatial.geotiff.DuplicateIFDTagError`
     - An IFD declares the same tag id more than once. TIFF 6.0
       forbids duplicate tags; without this check a malformed or
       adversarial file could change the parsed ``ImageWidth``,
       ``ImageLength``, ``Compression``, CRS, transform, or nodata
       silently because the legacy parser let the last duplicate win.
     - No opt-in. Either fix the source file or rewrite it through a
       conforming TIFF writer (rasterio, GDAL, libtiff). The error
       message names the duplicated tag id and the byte offsets of the
       two conflicting entries.

Remote-read safety limits
=========================

When :func:`xrspatial.geotiff.open_geotiff` is pointed at an
``http://``, ``https://``, ``s3://``, ``gs://``, ``az://``, or
``memory://`` URI, the reader applies several bounded-read guards
before fetching pixel bytes.

Byte budget
-----------

The reader caps the total bytes pulled from a remote source via the
``max_cloud_bytes`` kwarg on
:func:`~xrspatial.geotiff.open_geotiff`. The resolution order is:

1. The ``max_cloud_bytes`` kwarg, if the caller passed one.
2. The ``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES`` env var, if it is set to a
   positive integer.
3. The module default, 256 MiB. The constant lives at
   :data:`xrspatial.geotiff._sources.MAX_CLOUD_BYTES_DEFAULT`.

Pass ``max_cloud_bytes=None`` to disable the cap explicitly when the
caller has another reason to trust the source. The cap is a guard
against an unintended full-file fetch; it is not a substitute for an
explicit window or chunked read.

Private-host rejection
----------------------

HTTP / HTTPS reads resolve the URL's host and reject any address that
maps to a private, loopback, link-local, or otherwise non-public IP.
The check is on by default and exists to keep an SSRF-style request
from reaching an internal service. Set
``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` to opt out when the caller
is intentionally targeting a host on a private network.

Timeouts
--------

Two env vars control the HTTP timeouts on remote reads:

* ``XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT`` -- connect timeout in
  seconds.
* ``XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT`` -- read timeout in seconds.

Both fall back to the module default when unset.

Strict mode
-----------

``XRSPATIAL_GEOTIFF_STRICT=1`` flips several "warn and continue" sites
to "raise". The flag affects CRS resolution, VRT validation, and a
handful of decode-side fallback paths. Use it in CI when you want a
hard failure on metadata that the default path would tolerate.

Other env vars
--------------

* ``XRSPATIAL_GEOTIFF_MMAP_CACHE_SIZE`` -- caps the mmap cache size for
  local-file reads. Default 32.

The full list of env vars lives in the source under
:mod:`xrspatial.geotiff._sources` and :mod:`xrspatial.geotiff._runtime`.
The user-facing names above cover everything a caller normally
configures.


See also
========

* :ref:`reference.geotiff` -- the API reference for every public name on
  the module, including signatures, kwargs, and the stable COG contract
  text.
* :ref:`reference.geotiff_release_contract` -- the user-facing release
  contract that enumerates every feature in
  :data:`xrspatial.geotiff.SUPPORTED_FEATURES` against its tier.
* :ref:`reference.geotiff_release_gate` -- the release-gate audit
  checklist that ties each ``stable`` promise to a regression test.
* :ref:`user_guide.attrs_contract` -- the round-trip contract for the
  ``DataArray.attrs`` mapping that the reader emits and the writer
  consumes.
