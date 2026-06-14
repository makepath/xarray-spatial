"""Attrs / metadata helpers and TIFF tag constants for the geotiff entry points.

Bundle: everything that fills ``DataArray.attrs`` on a read or unpacks
attrs back into TIFF tags on a write. Called from every read backend
(via ``_populate_attrs_from_geo_info``) and every writer (via
``_resolve_nodata_attr`` / ``_extract_rich_tags``), so a single
canonical version keeps the four read paths and three writers in
lockstep.

Includes the ``_LEVEL_RANGES`` / ``_VALID_COMPRESSIONS`` constants
used by ``to_geotiff`` for friendly up-front validation, and the
``_TIFF_BYTE`` / ``_TIFF_ASCII`` / ``_TIFF_SHORT`` /
``_RESOLUTION_UNIT_IDS`` tag-id maps used when synthesising extra-tag
entries.

Extracted in step 5 of issue #1813.

Attrs contract (issue #1984)
----------------------------

The keys written into ``DataArray.attrs`` by the read paths fall into
three tiers. Writers honour the same split: canonical keys are emitted,
compatibility aliases are read but never written when the canonical key
is present, and pass-through keys are kept when the writer can
reconstruct them from canonical state.

The contract version is recorded in ``attrs['_xrspatial_geotiff_contract']``
(currently ``5``). Consumers can branch on this integer if the tier
split changes in a future release.

Canonical (xrspatial owns these; round-trip stable):

- ``crs``: EPSG integer code for the horizontal CRS. Dropped on rotated
  reads opened with ``allow_rotated=True`` (issue #2122) -- the array is
  treated as a no-georef pixel grid in that case.
- ``crs_wkt``: WKT string for the horizontal CRS. Present on read whenever
  any CRS information is available. Dropped on rotated reads opened with
  ``allow_rotated=True`` (issue #2122), in lockstep with ``crs``.
- ``transform``: rasterio-style 6-tuple
  ``(pixel_width, 0.0, origin_x, 0.0, pixel_height, origin_y)``. Omitted
  for files with no GeoTIFF transform tags (ModelTransformation,
  ModelPixelScale, or ModelTiepoint), and for rotated reads opened with
  ``allow_rotated=True`` (axis-aligned 6-tuple would silently drop the
  rotation terms).
- ``rotated_affine`` (#2129): rasterio-style 6-tuple
  ``(a, b, c, d, e, f)`` capturing the full ``ModelTransformationTag``
  on the ``allow_rotated=True`` opt-in path. Only emitted when the
  source carried a rotated / sheared transform; absent on plain
  no-georef reads and on axis-aligned reads (which already round-trip
  via ``transform``). Read-only -- the writer drops it on the way out
  until ``to_geotiff`` learns to emit ``ModelTransformationTag``
  (issue #2115 follow-up).
- ``nodata``: declared file sentinel as stored in the GDAL_NODATA tag.
  Set whenever the source declares one, as a scalar of the source
  dtype, regardless of whether the in-memory array is float-with-NaN
  or int-with-sentinels.
- ``masked_nodata``: boolean flag paired with ``nodata``. ``True`` iff
  the in-memory array is float dtype and the reader's sentinel-to-NaN
  step ran; ``False`` iff the array still carries the literal integer
  sentinel. Only emitted when ``nodata`` is set; absence is the
  "no declared sentinel" signal. See ``_set_nodata_attrs``. The flag
  tracks whether masking ran, not whether any sentinel pixel matched:
  a masked read of a maskable integer source promotes to float and sets
  ``True`` even when zero pixels match, so the eager and dask paths agree
  for the same input (issue #2990). Use ``nodata_pixels_present`` for the
  did-any-pixel-match question.
- ``nodata_pixels_present`` (#2135): bool, only emitted when
  ``nodata`` is set and the backend computed the answer cheaply.
  True iff the read window contained at least one pixel matching the
  declared sentinel before masking. Lets QA and writer code answer
  "are any sentinel pixels in this tile" without scanning the buffer.
  The dask path leaves this unset because a strict per-chunk
  reduction would force an eager ``.compute()``.
- ``nodata_dtype_cast`` (#2135): string dtype name (e.g.
  ``"float64"``), only emitted when ``nodata`` is set and the caller
  passed an explicit ``dtype=`` kwarg. Records that a post-mask cast
  happened so consumers can tell float-because-masked from
  float-because-promoted.
- ``mask_and_scale_dtype`` (#3064, contract v5): string dtype name of
  the source (e.g. ``"int8"``). Emitted when masking and / or
  ``mask_and_scale`` promoted an integer array to float on read, and
  on ``mask_and_scale`` reads of float sources that carry a declared
  sentinel or a non-identity scale / offset (#3080) -- float sources
  are never promoted, but recording their width keeps
  ``to_geotiff(pack=True)`` from widening float32 to float64.
  ``pack`` reads it to restore the on-disk dtype.
- ``raster_type``: ``'area'`` (implicit / RasterPixelIsArea) or ``'point'``
  (explicit / RasterPixelIsPoint).
- ``georef_status``: one of ``'full'``, ``'transform_only'``, ``'crs_only'``,
  ``'none'``, ``'rotated_dropped'``. Single attr that encodes the five
  distinct states the reader can land in when CRS / transform tags are
  combined. See :func:`_compute_georef_status` for the decision table and
  issue #2136 for the rationale. The attr is additive: ``crs`` / ``crs_wkt``
  / ``transform`` / ``_xrspatial_no_georef`` remain present with unchanged
  semantics so existing consumers keep working.
- ``extra_tags``: list of ``(tag_id, type_id, count, value)`` tuples for
  TIFF tags outside the structured set. Omitted when no out-of-band
  tags are present.
- ``gdal_metadata``: dict parsed from the GDAL_METADATA XML tag.
- ``gdal_metadata_xml``: raw GDAL_METADATA XML string. Writers prefer this
  over ``gdal_metadata`` when both are present.
- ``x_resolution``, ``y_resolution``, ``resolution_unit``: TIFF
  XResolution / YResolution / ResolutionUnit values.
- ``_xrspatial_geotiff_contract``: integer version of this contract.

Compatibility alias (read for ecosystem interop; writers must not emit
when the canonical key is present):

- ``nodatavals``: rioxarray per-band tuple form of ``nodata``.
- ``_FillValue``: CF-convention name for ``nodata``.

Best-effort pass-through (preserved when the writer can reconstruct
from canonical state, otherwise dropped on round-trip):

- ``image_description``: TIFF ImageDescription tag.
- ``extra_samples``: TIFF ExtraSamples tag.
- ``colormap``: raw uint16 RGB triples from the TIFF ColorMap tag (320),
  attached to single-band paletted images.

Removed in contract v2 (issue #2016):

The following keys were emitted by older xrspatial releases under a
deprecation warning and have been removed from the reader as of
contract version ``2``. Reads no longer surface them on
``DataArray.attrs``; downstream code that accessed them via
``attrs[key]`` will now see ``KeyError``. Switch to ``attrs.get(key)``
or derive the value from ``crs`` / ``crs_wkt`` with :mod:`pyproj`.

* Geographic-CRS GeoKey attrs: ``crs_name``, ``geog_citation``,
  ``datum_code``, ``angular_units``, ``semi_major_axis``,
  ``inv_flattening``.
* Projected-CRS GeoKey attrs: ``linear_units``, ``projection_code``.
* Vertical-CRS GeoKey attrs: ``vertical_crs``, ``vertical_citation``,
  ``vertical_units``.
* Colormap variants: ``colormap_rgba``, ``cmap``. Reshape
  ``attrs['colormap']`` to ``(n_colors, 3)`` and append an alpha
  channel in caller code, or construct a
  :class:`matplotlib.colors.ListedColormap` from
  ``attrs['colormap']`` in caller code.

Migration recipe (the canonical replacement is ``crs`` / ``crs_wkt``
plus a one-liner with :mod:`pyproj` when a derived value is needed)::

    from pyproj import CRS
    crs = CRS.from_wkt(attrs['crs_wkt'])  # or CRS.from_epsg(attrs['crs'])

    # Geographic
    crs.name                                 # crs_name
    crs.datum.to_epsg()                      # datum_code
    crs.ellipsoid.semi_major_metre           # semi_major_axis
    crs.ellipsoid.inverse_flattening         # inv_flattening
    # geog_citation / angular_units: best-effort derive from
    # ``crs`` / ``crs.axis_info``; the original GeoKey citation text
    # is not generally recoverable.

    # Projected
    crs.coordinate_system.axis_list[0].unit_name   # linear_units
    crs.to_epsg()                                  # projection_code

    # Vertical
    crs.sub_crs_list[-1].to_epsg()                 # vertical_crs
    crs.sub_crs_list[-1].name                      # vertical_citation
    crs.sub_crs_list[-1].axis_info[0].unit_name    # vertical_units

See ``docs/source/user_guide/attrs_contract.rst`` for the full
migration notes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import xarray as xr

from ._coords import coords_from_geo_info as _coords_from_geo_info
from ._coords import resolve_georef as _resolve_georef
from ._coords import transform_tuple_from_pixel_geometry as _transform_tuple_from_pixel_geometry
from ._errors import (ConflictingNodataError, MalformedScaleOffsetError,
                      _distinct_per_band_nodatavals_msg)
from ._geotags import (_NO_GEOREF_KEY, GEOKEY_GEOGRAPHIC_TYPE, GEOKEY_MODEL_TYPE,
                       GEOKEY_PROJECTED_CS_TYPE, RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT)

# Per-codec valid compression-level ranges, used by ``to_geotiff`` for
# friendly up-front validation. Codecs not listed here either reject any
# level (e.g. ``packbits``) or accept any value (e.g. ``lzw``); the
# writer enforces those rules at codec-tag time.
_LEVEL_RANGES = {
    'deflate': (1, 9),
    'zstd': (1, 22),
    'lz4': (0, 16),
}

# Names accepted by ``compression=`` in :func:`to_geotiff`.  Kept in sync with
# ``_compression_tag`` in ``_writer.py``.  Validated up-front so users see a
# friendly error rather than the deeper traceback from ``_compression_tag``.
_VALID_COMPRESSIONS = (
    'none', 'deflate', 'lzw', 'jpeg', 'packbits', 'zstd', 'lz4',
    'jpeg2000', 'j2k', 'lerc',
)


# Tiered feature inventory for the public geotiff surface.
# Defined in ``_attrs.py`` (not the package ``__init__.py``) so the writers
# can import it at module scope without a circular dependency: the package
# ``__init__`` already imports the writers. The package re-exports
# ``SUPPORTED_FEATURES`` so the public API stays
# ``xrspatial.geotiff.SUPPORTED_FEATURES``.
#
# See ``xrspatial/geotiff/__init__.py`` for the per-tier semantics; the
# inline comments here track the codec/reader/writer split used by the
# user-guide notebook table.
#
# COG entries are split into three keys so the writer, local reader, and
# HTTP reader can move between tiers on their own track. The three code
# paths have different stability profiles: the writer emits a
# self-contained COG layout, the local reader parses overview IFDs out of
# an on-disk file, and the HTTP reader adds range-request fetching and
# partial-IFD handling on top of that. Folding them into a single ``cog``
# key would tie a promotion of any one path to a promotion of all three.
# The keys carry no behaviour today; production code still gates on the
# underlying writer / reader options.
#
# Stable COG contract
# -------------------
# The local COG paths (``writer.cog`` and ``reader.local_cog``) are tagged
# ``stable``. The promotion is backed by the writer compliance suite, the
# cross-backend parity gate, and the per-tile byte-budget contract that
# pins HTTP fetch behaviour.
#
# The stable COG contract covers:
#
# * Axis-aligned 2D / 3D rasters.
# * CPU writer and CPU reader paths.
# * Lossless codecs: ``none``, ``deflate``, ``lzw``, ``zstd``, ``packbits``.
# * Internal overviews only (no ``.tif.ovr`` sidecars).
# * Normal CRS, transform, dtype, nodata, band, and
#   pixel-is-area / pixel-is-point round-trip.
#
# The following combinations stay outside the stable contract and keep
# their existing tier:
#
# * GPU COG read / write (``writer.gpu``, ``reader.gpu``).
# * Experimental codecs (``lerc``, ``jpeg2000``, ``j2k``, ``lz4``) and the
#   internal-only ``jpeg`` codec.
# * Rotated transforms (``reader.allow_rotated``), demoted from
#   ``advanced`` to ``experimental`` -- see the alignment block
#   immediately below.
# * External ``.tif.ovr`` sidecars (``reader.sidecar_ovr``).
# * File-like destinations with ``cog=True``.
# * BigTIFF COG (tracked separately).
# * HTTP / range COG (``reader.http_cog``); see the per-key comment below.
#
# Release-contract alignment
# ---------------------------
# The release contract tiers the public geotiff surface as Stable /
# Advanced / Experimental / Internal-only. The mapping below is the
# source of truth that the docs page, the user-guide notebook, and the
# writer gates read from, so it has to match the contract.
#
# Reconciliation notes:
#
# * Add ``reader.windowed`` at ``stable``. Windowed reads have a
#   release-gate suite (``test_window_*`` + ``test_no_georef_windowed_coords_1710``
#   and the GPU window cases) and the contract places them in Stable.
# * Add ``reader.dask`` at ``stable``. Dask reads are parity-tested
#   against the eager numpy reader via
#   ``test_backend_parity_matrix.py`` and ``test_backend_full_parity_2211.py``,
#   which is the bar the contract asks for (dask reads only where
#   parity-tested).
# * Demote ``reader.allow_rotated`` from ``advanced`` to ``experimental``.
#   The opt-in lets a caller bypass the read-side rotated-transform
#   check that ``RotatedTransformError`` defends; rotated writes are
#   Unsupported and the read-side escape hatch sits naturally in
#   Experimental.
# * Demote ``reader.allow_unparseable_crs`` from ``advanced`` to
#   ``experimental``. Explicit-CRS permissive escape hatches sit in
#   Experimental.
SUPPORTED_FEATURES = {
    # Codecs. Tier 1 lossless integer + float byte-for-byte round-trip.
    'codec.none': 'stable',
    'codec.deflate': 'stable',
    'codec.lzw': 'stable',
    'codec.packbits': 'stable',
    'codec.zstd': 'stable',
    # Tier 3 codecs: require ``allow_experimental_codecs=True``.
    'codec.lerc': 'experimental',
    'codec.jpeg2000': 'experimental',
    'codec.j2k': 'experimental',
    'codec.lz4': 'experimental',
    # Tier 4 codec: requires the dedicated ``allow_internal_only_jpeg``
    # opt-in. Not covered by ``allow_experimental_codecs``.
    'codec.jpeg': 'internal_only',
    # Read paths.
    'reader.local_file': 'stable',
    # Windowed reads: release-gate covered by the window-read
    # suite in ``xrspatial/geotiff/tests/`` (eager numpy, dask, GPU,
    # HTTP, and VRT branches each have a window-validation test).
    'reader.windowed': 'stable',
    # Dask reads: parity-tested against the eager numpy reader
    # by ``test_backend_parity_matrix.py`` and ``test_backend_full_parity_2211.py``,
    # which is the gate the contract asks for. GPU-backed dask reads remain
    # under ``reader.gpu`` and stay experimental.
    'reader.dask': 'stable',
    'reader.fsspec': 'advanced',
    'reader.http': 'advanced',
    'reader.vrt': 'advanced',
    'reader.sidecar_ovr': 'advanced',
    # Permissive escape hatches: demoted from ``advanced`` to
    # ``experimental``. Both opt-ins let a caller bypass a read-side
    # check (rotated transform / unparseable CRS) the writer normally
    # rejects, so they sit in Experimental.
    'reader.allow_rotated': 'experimental',
    'reader.allow_unparseable_crs': 'experimental',
    # COG reader paths: split out from the previous single COG concept
    # (which only carried ``writer.cog``) so the local and HTTP reader
    # variants can promote independently of the writer and of each other.
    #
    # ``reader.local_cog`` is ``stable``: on-disk overview-IFD parsing
    # is covered by the compliance suite and the parity gate.
    #
    # ``reader.http_cog`` stays ``advanced``. The HTTP path layers
    # range-request fetching, range coalescing, an SSRF / private-host
    # filter, a per-tile byte-count cap, and partial-IFD handling on
    # top of the local COG reader. The byte-budget contract pins
    # per-tile fetch behaviour, but the transport-side
    # surface -- redirect handling, network error reporting, cache /
    # retry policy -- is not yet contracted at the ``stable`` bar.
    # Tracked for separate promotion.
    'reader.local_cog': 'stable',
    'reader.http_cog': 'advanced',
    'reader.gpu': 'experimental',
    # ``unpack=True`` scale/offset decode on ``open_geotiff`` (CF-style
    # packed integers -> float, nodata sentinel -> NaN). Sits at
    # ``experimental`` rather than ``advanced``: the GPU / dask+GPU
    # branches gained round-trip coverage only recently (#3112 pack
    # crash fix, #3114 coverage), so no behavioural promise is made yet.
    'reader.unpack': 'experimental',
    # ``coregister=True`` on the ``.xrs.open_geotiff`` accessor: an
    # unpack-and-reproject read that resamples the file onto the
    # caller's exact grid. Sits at ``experimental``: CPU-only (raises
    # with ``gpu=True`` / ``.vrt`` sources), forces ``unpack=True``,
    # and the surface (resampling heuristic, NaN fill outside the file
    # footprint) can shift without a deprecation window. See the
    # "Coregistered reads" section in
    # ``docs/source/reference/geotiff.rst``.
    'reader.coregister': 'experimental',
    # Write paths.
    'writer.local_file': 'stable',
    # ``writer.cog`` is ``stable``: the CPU writer emits a
    # spec-conforming COG layout (IFD-first, tiled, internal
    # overviews, lossless codec) covered by the compliance suite
    # and the cross-backend parity gate. GPU writes,
    # BigTIFF COG, file-like destinations, and experimental codecs
    # stay outside the contract; see the block comment above.
    'writer.cog': 'stable',
    'writer.overviews': 'advanced',
    'writer.bigtiff': 'advanced',
    'writer.gpu': 'experimental',
    'writer.gdal_metadata_xml': 'experimental',
    'writer.extra_tags': 'experimental',
    # ``pack=True`` on ``to_geotiff``: inverse of ``reader.unpack``
    # (re-apply scale/offset, restore the recorded integer dtype, fill
    # NaN back to the nodata sentinel). Tracked at the same
    # ``experimental`` tier as ``reader.unpack`` -- the pair shares the
    # scale/offset attr contract and the recently-closed GPU gaps
    # (#3112, #3114).
    'writer.pack': 'experimental',
    # BigTIFF COG writer surface.
    # Tracked separately from ``writer.bigtiff`` and ``writer.cog`` because
    # the BigTIFF + COG combination has its own external-interop surface
    # (8-byte offsets in tile/overview tables, BigTIFF-form IFDs, COG
    # layout invariants). Stays ``advanced`` even when every row of
    # ``tests/write/test_bigtiff.py`` passes -- promotion
    # to ``stable`` happens after the gate has lived in CI for a release
    # cycle. See the BigTIFF COG section in
    # ``docs/source/reference/geotiff.rst``.
    'writer.bigtiff_cog': 'advanced',
}


# Tier 3 codec names (lower-cased) gated behind
# ``allow_experimental_codecs`` on the writers. Derived from
# ``SUPPORTED_FEATURES`` so the gate cannot drift from the docs.
_EXPERIMENTAL_CODECS = frozenset(
    name.split('.', 1)[1].lower()
    for name, tier in SUPPORTED_FEATURES.items()
    if name.startswith('codec.') and tier == 'experimental'
)


# Map TIFF compression tag values to codec names so the read-side opt-in
# gate can name the codec in the rejection message without each call
# site repeating the integer-to-name table. The keys
# are the TIFF 6 Compression tag values (tag 259) used inside
# ``_compression.py``; the values match the codec names that appear on
# the ``SUPPORTED_FEATURES`` keys (``codec.<name>``).
_COMPRESSION_TAG_TO_NAME = {
    1: 'none',
    5: 'lzw',
    7: 'jpeg',
    8: 'deflate',
    32773: 'packbits',
    # Adobe Deflate (32946) decodes through the same zlib path as
    # plain Deflate (8) and is collapsed onto the same codec name on
    # purpose: both tags share the stable-tier classification in
    # ``SUPPORTED_FEATURES`` (``codec.deflate``). A future Adobe-
    # Deflate-specific tier would need its own ``codec.<name>`` entry
    # AND its own mapping line here; the collapse is deliberate, not
    # a passthrough.
    32946: 'deflate',
    34712: 'jpeg2000',
    34887: 'lerc',
    50000: 'zstd',
    50004: 'lz4',
}


def _validate_read_codec_optin(
    compression: int,
    *,
    allow_experimental_codecs: bool,
    allow_internal_only_jpeg: bool,
    entry_point: str = "open_geotiff",
) -> None:
    """Reject experimental / internal-only codecs on the read side.

    Mirrors the writer-side gate in ``_writers/eager.py`` /
    ``_writers/gpu.py`` so a caller cannot decode a file produced with
    an experimental or internal-only codec without naming the matching
    opt-in flag at the call site. The flag and feature both appear in
    the rejection message so the caller learns the name from the error
    rather than the docs.

    The writer side already carries these gates; this helper extends
    the same shape to the read entry points.

    Parameters
    ----------
    compression : int
        TIFF Compression tag value (tag 259) from the parsed IFD.
    allow_experimental_codecs : bool
        Opt-in for Tier 3 read paths (LERC, JPEG2000 / J2K, LZ4).
    allow_internal_only_jpeg : bool
        Opt-in for Tier 4 read path (JPEG-in-TIFF). Does not collapse
        into ``allow_experimental_codecs`` for the same reason as on
        the writer: internal-only is a stricter tier and keeps its own
        dedicated flag.
    entry_point : str
        Name of the public read function for the rejection message.
    """
    codec_name = _COMPRESSION_TAG_TO_NAME.get(int(compression))
    if codec_name is None:
        # Unknown compression tags are validated separately by the
        # decoder; the opt-in gate only fires for codecs the reader
        # otherwise accepts.
        return
    if codec_name == 'jpeg' and not allow_internal_only_jpeg:
        raise ValueError(
            f"{entry_point}: source uses compression='jpeg' (TIFF "
            "tag 259 = 7), which is internal-only: the encoder writes "
            "self-contained JFIF tiles without the TIFF JPEGTables tag "
            "(347), so the read path is not interoperable with libtiff "
            "/ GDAL / rasterio. Pass allow_internal_only_jpeg=True to "
            "opt in to the internal-reader-only decode path. See "
            "SUPPORTED_FEATURES tier 'internal_only' for codec.jpeg "
            "(epic #2340, original gate #1845).")
    if codec_name in _EXPERIMENTAL_CODECS and not allow_experimental_codecs:
        raise ValueError(
            f"{entry_point}: source uses compression={codec_name!r} "
            "which is experimental on the read side: cross-backend "
            "numerical parity is not claimed and reader support across "
            "GDAL versions is uneven. Pass allow_experimental_codecs="
            "True to opt in, or re-encode the source with a stable "
            "lossless codec ('deflate', 'zstd', or 'lzw'). See "
            f"SUPPORTED_FEATURES tier 'experimental' for codec.{codec_name} "
            "(epic #2340, original writer gate #2137).")


# Writer rich-tag attrs that ride the Experimental tier.
# ``writer.gdal_metadata_xml`` and ``writer.extra_tags`` carry
# free-form payloads through to the on-disk TIFF; their interop with
# other readers (rasterio, libtiff, GDAL) depends on the payload and is
# not part of the release promise. The opt-in keeps the surface narrow
# without removing the capability.
#
# Round-trip exemption: when the attrs carry the
# ``_xrspatial_geotiff_contract`` marker, they came from a previous
# xrspatial read. The reader populated ``gdal_metadata_xml`` /
# ``extra_tags`` from the source file; gating the write would force
# every read-then-write caller to opt in. Skip the gate on
# round-tripped attrs so the canonical contract stays a no-flag
# operation. The gate still fires when a caller adds those attrs to a
# fresh DataArray that did not come from a read.
def _validate_write_rich_tag_optin(
    attrs: dict,
    *,
    gdal_metadata_xml_kwarg: object = None,
    extra_tags_kwarg: object = None,
    allow_experimental_codecs: bool,
    entry_point: str = "to_geotiff",
) -> None:
    """Reject writes that include ``gdal_metadata_xml`` or ``extra_tags``
    unless the caller opted in via ``allow_experimental_codecs=True``.

    Mirrors the existing codec-flag shape so the rejection names the
    same opt-in the caller already learned from the LERC / J2K / LZ4
    paths. Round-tripped attrs (carrying the
    ``_xrspatial_geotiff_contract`` marker) are exempt so the canonical
    attrs round-trip stays a no-flag operation; the gate fires only when
    a caller constructs a fresh DataArray with one of the rich-tag attrs
    set.
    """
    if allow_experimental_codecs:
        return
    # Round-trip exemption: a DataArray that came from
    # ``open_geotiff`` / ``_read_geotiff_dask`` / ``_read_geotiff_gpu``
    # carries the contract marker. Writing it back is the canonical
    # round-trip and should not require a new flag.
    #
    # This is a soft gate by design: a caller who hand-builds an
    # attrs dict with the contract marker could bypass it. Forging
    # the marker is a deliberate act, and the alternative (gating
    # every read-then-write call) would break the canonical attrs
    # round-trip that downstream code already depends on. The hard
    # guarantee is "fresh DataArrays carrying these attrs need the
    # opt-in"; the soft exemption keeps round-trips frictionless.
    if '_xrspatial_geotiff_contract' in attrs:
        return
    triggered: list[str] = []
    if attrs.get('gdal_metadata_xml') is not None:
        triggered.append("attrs['gdal_metadata_xml']")
    if attrs.get('extra_tags') is not None:
        triggered.append("attrs['extra_tags']")
    if gdal_metadata_xml_kwarg is not None:
        triggered.append('gdal_metadata_xml kwarg')
    if extra_tags_kwarg is not None:
        triggered.append('extra_tags kwarg')
    if not triggered:
        return
    raise ValueError(
        f"{entry_point}: {', '.join(triggered)} pass-through is "
        "experimental: the on-disk bytes are written verbatim and "
        "interop with other readers (rasterio, libtiff, GDAL) depends "
        "on the payload. Pass allow_experimental_codecs=True to opt "
        "in to the rich-tag write path, or drop the attr before the "
        "write. See SUPPORTED_FEATURES tier 'experimental' for "
        "writer.gdal_metadata_xml / writer.extra_tags (epic #2340).")


# TIFF type ids needed when synthesizing extra_tags entries from attrs.
_TIFF_BYTE = 1
_TIFF_ASCII = 2
_TIFF_SHORT = 3


# Contract version emitted on every read; bumped when the attrs contract
# changes. Downstream code reads ``attrs['_xrspatial_geotiff_contract']``
# to learn which attrs-contract revision produced the array. See
# ``docs/source/user_guide/attrs_contract.rst``.
#
# Version 2 drops the 13 deprecated GeoKey-derived and
# matplotlib-colormap attrs that v1 still emitted under a
# ``DeprecationWarning``. Downstream code that read those keys via
# ``attrs[key]`` now sees ``KeyError`` rather than the deprecated value.
#
# Version 3 adds ``attrs['georef_status']`` to the canonical
# tier. Existing keys (``crs``, ``crs_wkt``, ``transform``, the
# ``_xrspatial_no_georef`` marker) keep their pre-v3 shape so downstream
# code that branches on them still works; the new attr is additive and
# disambiguates ``crs_only`` from ``none`` and ``rotated_dropped`` from
# the truly-no-transform case.
#
# Version 4 adds ``attrs['rotated_affine']`` for the
# ``allow_rotated=True`` opt-in path. The 6-tuple is read-only -- the
# writer drops it on round-trip until ``to_geotiff`` grows a
# ``ModelTransformationTag`` emit path. Existing keys keep their pre-v4
# shape.
#
# Version 5 adds ``attrs['mask_and_scale_dtype']``: the integer source
# dtype recorded when ``mask_nodata`` / ``mask_and_scale`` promoted the
# array to float on read. ``to_geotiff(pack=True)`` reads it to restore
# the on-disk dtype. Existing keys keep their pre-v5 shape. #3080 later
# extended emission (same key, same shape, still v5) to float sources on
# ``mask_and_scale`` reads that carry a sentinel or a non-identity
# scale / offset, so ``pack`` keeps float32 sources float32.
_ATTRS_CONTRACT_VERSION = 5


# Canonical ``attrs['georef_status']`` values. One attr
# encodes the five distinct states the reader can land in when CRS and
# transform tags are combined; downstream code can branch on this rather
# than reconstructing the state from the union of ``crs``, ``crs_wkt``,
# ``transform``, and ``_xrspatial_no_georef``.
GEOREF_STATUS_FULL = 'full'
GEOREF_STATUS_TRANSFORM_ONLY = 'transform_only'
GEOREF_STATUS_CRS_ONLY = 'crs_only'
GEOREF_STATUS_NONE = 'none'
GEOREF_STATUS_ROTATED_DROPPED = 'rotated_dropped'

# Public frozenset of every valid ``georef_status`` value. Exposed so
# downstream code can validate user-set values without hard-coding the
# five-string list (e.g. ``status in GEOREF_STATUS_VALUES``).
GEOREF_STATUS_VALUES = frozenset({
    GEOREF_STATUS_FULL,
    GEOREF_STATUS_TRANSFORM_ONLY,
    GEOREF_STATUS_CRS_ONLY,
    GEOREF_STATUS_NONE,
    GEOREF_STATUS_ROTATED_DROPPED,
})


# String identifiers (used in xrspatial attrs) -> TIFF ResolutionUnit tag ids.
_RESOLUTION_UNIT_IDS = {'none': 1, 'inch': 2, 'centimeter': 3}


# Reverse map of ``_RESOLUTION_UNIT_IDS``: TIFF ResolutionUnit tag ids back to
# the xrspatial string identifiers. Used by ``geo_info_to_metadata`` so the
# read side stores the same string label the writer expects on the way out.
# Derived from ``_RESOLUTION_UNIT_IDS`` so the forward and reverse maps cannot
# drift if a new unit is added.
_RESOLUTION_UNIT_NAMES = {v: k for k, v in _RESOLUTION_UNIT_IDS.items()}


@dataclass(frozen=True)
class GeoTIFFMetadata:
    """Typed internal record for GeoTIFF read/write metadata.

    Mirrors the public attrs contract documented in this module's
    docstring field-for-field. Read paths build a ``GeoTIFFMetadata``
    once and call :func:`metadata_to_attrs` at the DataArray-construction
    boundary; write paths call :func:`attrs_to_metadata` once at entry
    and read fields off the record instead of re-resolving from
    ``data.attrs``.

    The public attrs surface is unchanged. The dataclass exists so the
    four read paths (eager numpy, dask+numpy, GPU, VRT) and three writers
    stop building / parsing the same dict by hand.
    """

    # Spatial reference
    transform: tuple | None = None
    crs_epsg: int | None = None
    crs_wkt: str | None = None
    raster_type: str = 'area'
    has_georef: bool = True

    # NoData semantics
    nodata: Any = None
    masked_nodata: bool | None = None

    # Pass-through TIFF tags
    extra_tags: list | None = None
    image_description: str | None = None
    extra_samples: tuple | None = None
    colormap: tuple | None = None

    # GDAL_METADATA
    gdal_metadata: dict | None = None
    gdal_metadata_xml: str | None = None

    # Resolution tags
    x_resolution: float | None = None
    y_resolution: float | None = None
    resolution_unit: str | None = None

    # VRT-only
    vrt_holes: list | None = None

    # Canonical reader-state classifier. Carried on the
    # record so the eager / dask / GPU / VRT read paths all stamp it via
    # the same :func:`metadata_to_attrs` marshalling step instead of
    # branching on attrs after the dict has been built.
    georef_status: str | None = None

    # Rotated 6-tuple from ``ModelTransformationTag`` on the
    # ``allow_rotated=True`` opt-in path. Carried on the
    # record so the eager / dask / GPU / VRT read paths emit
    # ``attrs['rotated_affine']`` through the same marshalling step.
    # Read-only: :func:`attrs_to_metadata` intentionally does NOT
    # populate this field from incoming attrs so the writer keeps
    # dropping the rotation on round-trip until ``to_geotiff`` learns to
    # emit ``ModelTransformationTag``.
    rotated_affine: tuple | None = None

    # Contract version stamped on read
    contract_version: int = _ATTRS_CONTRACT_VERSION

    def with_nodata(self, nodata, *, masked: bool) -> 'GeoTIFFMetadata':
        """Return a copy with ``nodata`` and ``masked_nodata`` set.

        Mirrors :func:`_set_nodata_attrs`: when ``nodata is None`` the
        record is returned unchanged so absence keeps signalling "no
        declared sentinel"; otherwise ``masked`` is coerced to ``bool``
        and stored as ``masked_nodata``.
        """
        if nodata is None:
            return self
        return replace(self, nodata=nodata, masked_nodata=bool(masked))


def geo_info_to_metadata(geo_info, *, window=None) -> GeoTIFFMetadata:
    """Build a :class:`GeoTIFFMetadata` from a reader's ``GeoInfo``.

    Centralises the field-by-field copy previously done inline in
    :func:`_populate_attrs_from_geo_info`. ``window`` carries the
    ``(r0, c0, r1, c1)`` tuple for windowed reads and is applied to the
    emitted transform.
    """
    has_georef = getattr(geo_info, 'has_georef', True)
    src_t = geo_info.transform

    transform = None
    if src_t is not None and has_georef:
        transform = _transform_tuple_from_pixel_geometry(
            src_t.origin_x, src_t.origin_y,
            src_t.pixel_width, src_t.pixel_height,
            window=window,
        )

    # ``allow_rotated=True`` opt-in path: the parser returns a
    # GeoTransform with ``rotated_affine`` set and ``has_georef=False``.
    # The rotated 6-tuple cannot be expressed as an axis-aligned
    # rasterio transform, so the writer cannot round-trip it via
    # ``attrs['transform']``. Per the documented contract on
    # ``open_geotiff(allow_rotated=True)``, CRS attrs are dropped on
    # this path too -- otherwise downstream code that gates on
    # ``"crs" in da.attrs`` treats the array as spatially meaningful
    # while the actual mapping is gone.
    rotated_optin = (
        src_t is not None
        and getattr(src_t, 'rotated_affine', None) is not None
        and not has_georef
    )
    crs_epsg = None if rotated_optin else geo_info.crs_epsg
    crs_wkt = None if rotated_optin else geo_info.crs_wkt

    # Surface the rotated 6-tuple on the public attrs so
    # downstream code that knows how to handle rotated rasters can read
    # it without diving into the internal ``GeoInfo`` / ``GeoTransform``
    # objects. Tuple cast normalises lists or numpy sequences coming
    # from the parser into the documented ``tuple`` shape.
    rotated_affine_tuple = (
        tuple(src_t.rotated_affine) if rotated_optin else None
    )

    raster_type = (
        'point' if geo_info.raster_type == RASTER_PIXEL_IS_POINT else 'area')

    resolution_unit = None
    if geo_info.resolution_unit is not None:
        resolution_unit = _RESOLUTION_UNIT_NAMES.get(
            geo_info.resolution_unit, str(geo_info.resolution_unit))

    colormap = None
    if geo_info.extra_tags is not None:
        for _tag_id, _tt, _tc, _tv in geo_info.extra_tags:
            if _tag_id == 320:  # TAG_COLORMAP
                colormap = _tv
                break

    return GeoTIFFMetadata(
        transform=transform,
        crs_epsg=crs_epsg,
        crs_wkt=crs_wkt,
        raster_type=raster_type,
        has_georef=bool(has_georef and src_t is not None),
        extra_tags=geo_info.extra_tags,
        image_description=geo_info.image_description,
        extra_samples=geo_info.extra_samples,
        colormap=colormap,
        gdal_metadata=geo_info.gdal_metadata,
        gdal_metadata_xml=geo_info.gdal_metadata_xml,
        x_resolution=geo_info.x_resolution,
        y_resolution=geo_info.y_resolution,
        resolution_unit=resolution_unit,
        # ``georef_status`` is computed off the unmodified
        # ``geo_info`` rather than the post-branch metadata fields so a
        # future change to which fields the record carries cannot
        # accidentally shift the status value. The VRT inline path uses
        # ``_compute_georef_status_from_parts`` to fill this field
        # without synthesising a ``GeoInfo``.
        georef_status=_compute_georef_status(geo_info),
        rotated_affine=rotated_affine_tuple,
        contract_version=_ATTRS_CONTRACT_VERSION,
    )


def metadata_to_attrs(md: GeoTIFFMetadata) -> dict:
    """Build the public attrs dict from a :class:`GeoTIFFMetadata` record.

    The output is identical to what
    :func:`_populate_attrs_from_geo_info` writes today, so the read-path
    swap is a behaviour no-op when the source record comes from
    :func:`geo_info_to_metadata`.
    """
    attrs: dict = {'_xrspatial_geotiff_contract': md.contract_version}

    # ``georef_status`` is stamped before the optional CRS /
    # transform branches so the value reflects the reader's state
    # decision (computed off ``geo_info``) rather than which fields
    # happened to land in the emitted dict.
    if md.georef_status is not None:
        attrs['georef_status'] = md.georef_status

    if md.crs_epsg is not None:
        attrs['crs'] = md.crs_epsg
    if md.crs_wkt is not None:
        attrs['crs_wkt'] = md.crs_wkt
    if md.raster_type == 'point':
        attrs['raster_type'] = 'point'

    # Three states on the (transform, has_georef) pair:
    #
    # * transform set + has_georef -> emit ``attrs['transform']``.
    # * not has_georef             -> emit ``attrs[_NO_GEOREF_KEY] = True``.
    # * transform None + has_georef -> emit neither.
    #
    # The third state is the eager VRT path's contract: the VRT builder
    # constructs the record with ``has_georef=True`` but leaves
    # ``transform=None`` because the inline VRT code stamps the
    # rasterio-ordered transform tuple onto the dict a few lines later
    # (after the GPU transfer / dtype cast / nodata mask steps). Removing
    # this branch would re-introduce a duplicate transform write or, worse,
    # would emit ``_NO_GEOREF_KEY`` on a georef'd VRT array.
    if md.transform is not None and md.has_georef:
        attrs['transform'] = md.transform
    elif not md.has_georef:
        attrs[_NO_GEOREF_KEY] = True

    # ``rotated_affine`` rides alongside the
    # ``_xrspatial_no_georef`` marker on the ``allow_rotated=True`` path
    # so callers can recover the rotated mapping. Only set on read; the
    # writer-side :func:`attrs_to_metadata` deliberately does not parse
    # it back, so a read-then-write round-trip drops the rotation until
    # the writer grows ``ModelTransformationTag`` emit support.
    if md.rotated_affine is not None:
        attrs['rotated_affine'] = md.rotated_affine

    if md.nodata is not None:
        attrs['nodata'] = md.nodata
        attrs['masked_nodata'] = bool(md.masked_nodata)

    if md.gdal_metadata is not None:
        attrs['gdal_metadata'] = md.gdal_metadata
    if md.gdal_metadata_xml is not None:
        attrs['gdal_metadata_xml'] = md.gdal_metadata_xml

    if md.extra_tags is not None:
        attrs['extra_tags'] = md.extra_tags
    if md.image_description is not None:
        attrs['image_description'] = md.image_description
    if md.extra_samples is not None:
        attrs['extra_samples'] = md.extra_samples

    if md.x_resolution is not None:
        attrs['x_resolution'] = md.x_resolution
    if md.y_resolution is not None:
        attrs['y_resolution'] = md.y_resolution
    if md.resolution_unit is not None:
        attrs['resolution_unit'] = md.resolution_unit

    if md.colormap is not None:
        attrs['colormap'] = md.colormap

    if md.vrt_holes:
        attrs['vrt_holes'] = md.vrt_holes

    return attrs


def attrs_to_metadata(attrs) -> GeoTIFFMetadata:
    """Parse a (possibly user-supplied) attrs dict into a metadata record.

    Honours the alias resolution :func:`_resolve_nodata_attr` already
    implements. ``attrs`` may be a plain dict, an
    ``xarray.core.utils.Frozen``, or ``None``.

    Boundary contract -- this function parses leniently and lets the
    writer enforce strict validation against the parsed record:

    * ``attrs['crs']=True`` lands as ``crs_epsg=None`` rather than
      raising. ``_validate_crs_arg`` (called from the writers) is the
      validator that should reject bool values; the boundary parser
      only needs to keep the bad value out of the record so the writer
      sees ``crs_epsg=None`` and falls through to ``crs_wkt``. See
      ``write/test_crs.py`` (CRS argument validation section).
    * ``transform`` is coerced via ``tuple(...)`` with no length or
      numeric-type check. ``_transform_from_attr`` is the canonical
      validator and runs inside the writer.
    """
    if attrs is None:
        attrs = {}

    raster_type = 'point' if attrs.get('raster_type') == 'point' else 'area'

    transform = attrs.get('transform')
    no_georef = bool(attrs.get(_NO_GEOREF_KEY))
    has_georef = (transform is not None) and not no_georef

    nodata = _resolve_nodata_attr(attrs)
    masked_attr = attrs.get('masked_nodata')
    masked_nodata: bool | None
    if nodata is None:
        masked_nodata = None
    elif masked_attr is None:
        masked_nodata = None
    else:
        masked_nodata = bool(masked_attr)

    crs_epsg = None
    crs_wkt = attrs.get('crs_wkt') if isinstance(attrs.get('crs_wkt'), str) else None
    crs_attr = attrs.get('crs')
    if isinstance(crs_attr, str):
        # ``attrs['crs']`` carries a WKT string under some pipelines; fold
        # it into ``crs_wkt`` here so the writer's resolve step does not
        # have to re-branch on type.
        if crs_wkt is None:
            crs_wkt = crs_attr
    elif crs_attr is not None and not isinstance(crs_attr, bool):
        # ``isinstance(True, int)`` is True; reject bool here so the
        # writer's _validate_crs_arg gate is not bypassed at the
        # boundary.
        try:
            crs_epsg = int(crs_attr)
        except (TypeError, ValueError):
            crs_epsg = None

    contract_version = attrs.get(
        '_xrspatial_geotiff_contract', _ATTRS_CONTRACT_VERSION)

    return GeoTIFFMetadata(
        transform=tuple(transform) if transform is not None else None,
        crs_epsg=crs_epsg,
        crs_wkt=crs_wkt,
        raster_type=raster_type,
        has_georef=has_georef,
        nodata=nodata,
        masked_nodata=masked_nodata,
        extra_tags=attrs.get('extra_tags'),
        image_description=attrs.get('image_description'),
        extra_samples=attrs.get('extra_samples'),
        colormap=attrs.get('colormap'),
        gdal_metadata=attrs.get('gdal_metadata'),
        gdal_metadata_xml=attrs.get('gdal_metadata_xml'),
        x_resolution=attrs.get('x_resolution'),
        y_resolution=attrs.get('y_resolution'),
        resolution_unit=attrs.get('resolution_unit'),
        vrt_holes=attrs.get('vrt_holes'),
        georef_status=attrs.get('georef_status'),
        contract_version=contract_version,
    )


def _extent_to_window(transform, file_height, file_width,
                      y_min, y_max, x_min, x_max):
    """Convert geographic extent to pixel window (row_start, col_start, row_stop, col_stop).

    Clamps to file bounds.
    """
    # Pixel coords from geographic coords
    col_start = (x_min - transform.origin_x) / transform.pixel_width
    col_stop = (x_max - transform.origin_x) / transform.pixel_width

    row_start = (y_max - transform.origin_y) / transform.pixel_height
    row_stop = (y_min - transform.origin_y) / transform.pixel_height

    # pixel_height is typically negative, so row_start/row_stop may be swapped
    if row_start > row_stop:
        row_start, row_stop = row_stop, row_start
    if col_start > col_stop:
        col_start, col_stop = col_stop, col_start

    row_start = max(0, int(np.floor(row_start)))
    col_start = max(0, int(np.floor(col_start)))
    row_stop = min(file_height, int(np.ceil(row_stop)))
    col_stop = min(file_width, int(np.ceil(col_stop)))

    return (row_start, col_start, row_stop, col_stop)


def _should_restore_nan_sentinel(attrs) -> bool:
    """Return True iff the writer should NaN-to-sentinel rewrite the array.

    Pairs with :func:`_set_nodata_attrs`. The reader stores a boolean in
    ``attrs['masked_nodata']`` describing whether the in-memory array
    went through the sentinel-to-NaN promotion. The writer reads it back
    here to decide whether the inverse rewrite is appropriate:

    * ``masked_nodata`` missing -> default True. Legacy behaviour:
      any float array with NaN and a declared sentinel gets the NaN
      pixels rewritten to the sentinel value. This is what every
      xrspatial caller has relied on for years and what every external
      DataArray that does not carry the new attr should still see.
    * ``masked_nodata=True`` -> True. The reader masked the sentinel
      into NaN; the writer reverses the step.
    * ``masked_nodata=False`` -> False. The reader did NOT mask (the
      array still carries the literal sentinel, or is float for an
      unrelated reason). Any NaN present in the array did not come
      from sentinel-masking, and rewriting it to the integer sentinel
      would corrupt data the user wrote there for other reasons.

    The ``attrs`` argument may be a plain dict, an
    ``xarray.core.utils.Frozen``, or ``None`` (the GPU writer's
    positional-cupy branch has no DataArray to read from). ``None`` and
    non-mapping inputs fall back to the True default.
    """
    if attrs is None:
        return True
    try:
        value = attrs.get('masked_nodata')
    except AttributeError:
        return True
    # Treat anything other than literal ``False`` as the True default.
    # The flag is the boolean ``True``/``False`` per the contract, but
    # we narrow on identity rather than truthiness so a stray ``0`` /
    # ``''`` (which a future maintainer might assume is "off" under a
    # truthiness rule) does not silently disable the sentinel rewrite.
    # The identity check is deliberate: do not refactor to ``not value``.
    return value is not False


def _set_nodata_attrs(
    attrs: dict,
    nodata,
    *,
    masked: bool,
    pixels_present: bool | None = None,
    dtype_cast: str | None = None,
) -> None:
    """Set the nodata lifecycle attrs on a read.

    ``masked`` is the actual mask-decision the read path made: True iff
    sentinel pixels in the in-memory buffer have been replaced with NaN
    (or the buffer is NaN-aware as a result of the reader's masking
    step). False iff the literal sentinel values are still present in
    the buffer.

    ``pixels_present`` is a lifecycle signal. If
    not ``None``, the read path computed whether the read window
    contained at least one pixel matching the declared sentinel before
    masking; the value is forwarded to ``attrs['nodata_pixels_present']``
    so consumers can answer "any nodata in this tile" without scanning
    the buffer. Pass ``None`` when the backend cannot cheaply produce
    the value (e.g. dask, where a strict per-chunk reduction would
    force eager compute).

    ``dtype_cast`` is the second lifecycle signal.
    If the caller passed an explicit ``dtype=`` kwarg, the backend
    forwards the resolved target dtype string (e.g. ``"float64"``) so
    consumers can distinguish "float because masking promoted it" from
    "float because the caller cast it". ``None`` means no caller cast
    happened; the attr is omitted in that case.

    Contract (splits the two meanings previously fused into
    ``attrs['nodata']``):

    * ``attrs['nodata']`` -- declared file sentinel, as a scalar of the
      source dtype. Set whenever the source declared one, regardless of
      whether the array is float-with-NaN or int-with-sentinels.
    * ``attrs['masked_nodata']`` -- the ``masked`` value the caller
      passed, coerced to bool. Only emitted when ``nodata is not
      None``; absence of the flag means there is no declared sentinel.
    * ``attrs['nodata_pixels_present']`` (additive) -- bool,
      only emitted when ``nodata is not None`` and ``pixels_present``
      is not ``None``. Tracks whether the read window contained any
      sentinel pixel before masking.
    * ``attrs['nodata_dtype_cast']`` (additive) -- string dtype
      name (e.g. ``"float64"``), only emitted when ``nodata is not
      None`` and ``dtype_cast`` is not ``None``. Records that a
      caller-requested cast happened after masking.

    The helper used to infer ``masked`` from the final array
    dtype, which lied when ``mask_nodata=False`` left literal sentinel
    values in a float buffer; downstream code that trusted the attr
    treated those literal values as already-NaN. The eager, dask, GPU,
    and VRT paths now compute ``masked`` as
    ``mask_nodata and final_dtype.kind == 'f'``.
    """
    if nodata is None:
        return
    attrs['nodata'] = nodata
    attrs['masked_nodata'] = bool(masked)
    if pixels_present is not None:
        attrs['nodata_pixels_present'] = bool(pixels_present)
    if dtype_cast is not None:
        attrs['nodata_dtype_cast'] = str(dtype_cast)


def _validate_read_geo_info(
    geo_info,
    *,
    window=None,
    allow_rotated: bool = False,
    allow_unparseable_crs: bool = False,
    band_nodata: str | None = None,
    band_nodata_values: list | None = None,
) -> None:
    """Run the read-side ambiguous-metadata checks against ``geo_info``.

    Centralised helper so the eager numpy, dask, GPU, and VRT read
    paths run the same checks before constructing the returned
    DataArray. Forwards ``allow_rotated`` / ``allow_unparseable_crs``
    to the registered checks (``_check_read_rotated_transform`` and
    ``_check_read_unparseable_crs`` today; sibling checks attach via
    the registry).

    ``band_nodata`` and ``band_nodata_values`` are the VRT-specific
    context for ``_check_read_mixed_band_metadata``. Non-VRT callers
    omit them and the mixed-band check short-circuits because
    ``band_nodata_values`` is falsy. VRT callers thread the kwargs
    through ``_finalize_lazy_read_attrs`` so the helper-routed call
    runs the same surface as the VRT pre-read inline check, instead
    of dispatching the mixed-band check as a no-op.

    Raises whichever ``GeoTIFFAmbiguousMetadataError`` subclass a
    registered check picks. The hook is a no-op when no check is
    registered, so callers can use this helper unconditionally without
    coupling each backend to the current check list.

    Note: the transform tuple built here is always axis-aligned
    (``b == 0`` / ``d == 0``) because ``_transform_tuple_from_pixel_geometry``
    only carries origin + pixel size, and the upstream TIFF reader
    rejects rotated ``ModelTransformationTag`` entries with
    ``RotatedTransformError`` in ``_geotags._extract_transform_and_georef``
    before we reach this helper (previously a plain
    ``NotImplementedError``). Both the GeoTIFF and VRT entry points
    therefore raise the same typed error -- the rotated-transform check
    in this helper still fires only on the VRT path (which builds its
    context from the GDAL ``geo_transform`` via
    ``_gdal_geotransform_to_affine_tuple``), but the contract a caller
    sees is identical across both.
    """
    from ._validation import validate_read_metadata
    transform_for_check = (
        _transform_tuple_from_pixel_geometry(
            geo_info.transform.origin_x,
            geo_info.transform.origin_y,
            geo_info.transform.pixel_width,
            geo_info.transform.pixel_height,
            window=window,
        )
        if (geo_info.transform is not None
            and getattr(geo_info, 'has_georef', True))
        else None
    )
    # Pull the GeoKey shape onto the context so
    # ``_check_read_inconsistent_geokeys`` can audit
    # ModelType / ProjectedCSType / GeographicType for contradictions.
    # ``geo_info.geokeys`` is the parsed dict; missing keys map to
    # ``None`` so the check treats the slot as "not declared" rather
    # than as a default-zero, which would otherwise look like
    # ``ModelType = undefined`` instead of "no model type tag at all".
    raw_geokeys = getattr(geo_info, 'geokeys', None) or {}
    model_type_ctx = raw_geokeys.get(GEOKEY_MODEL_TYPE)
    proj_cs_ctx = raw_geokeys.get(GEOKEY_PROJECTED_CS_TYPE)
    geog_ctx = raw_geokeys.get(GEOKEY_GEOGRAPHIC_TYPE)
    validate_read_metadata({
        'allow_rotated': allow_rotated,
        'allow_unparseable_crs': allow_unparseable_crs,
        'transform': transform_for_check,
        'crs_wkt': geo_info.crs_wkt,
        'model_type': model_type_ctx,
        'projected_cs_type': proj_cs_ctx,
        'geographic_type': geog_ctx,
        'band_nodata': band_nodata,
        'band_nodata_values': band_nodata_values,
    })


def _compute_georef_status(geo_info) -> str:
    """Classify ``geo_info`` into one of the five ``georef_status`` values.

    See the module docstring for the full rationale. The
    decision table:

    ============================  =================  ===============
    transform tags                CRS present        georef_status
    ============================  =================  ===============
    axis-aligned                  yes                ``full``
    axis-aligned                  no                 ``transform_only``
    absent                        yes                ``crs_only``
    absent                        no                 ``none``
    rotated, dropped              either             ``rotated_dropped``
    ============================  =================  ===============

    "CRS present" is signalled by either ``geo_info.crs_epsg`` or
    ``geo_info.crs_wkt`` being non-None. The rotated-dropped branch
    fires when the upstream reader saw a rotated
    ``ModelTransformationTag`` and was opened with ``allow_rotated=True``;
    that path returns ``has_georef=False`` with the rotated 6-tuple on
    ``geo_info.transform.rotated_affine``. The check is on
    ``rotated_affine`` rather than the surrounding state so a future
    reader change cannot accidentally re-route a real "no transform"
    file into the rotated bucket.

    The eager numpy, dask, and three GPU read sites (chunked / eager /
    tile in ``_backends/gpu.py``) all call this through
    :func:`_populate_attrs_from_geo_info`. The two VRT inline branches
    (eager + chunked in ``_backends/vrt.py``) call
    :func:`_compute_georef_status_from_parts` directly because they
    build their attrs dict from a different dataclass and would have to
    synthesise a fake ``GeoInfo`` to reuse this helper. Keep all the
    call sites in lockstep through one of the two helpers.

    Implementation note: routes through
    :func:`xrspatial.geotiff._coords.resolve_georef` so the read-side
    bucket decision lives in one place. The resolver's status strings
    are the same canonical values exported here (``GEOREF_STATUS_*``).
    """
    return _resolve_georef(geo_info=geo_info).georef_status


def _compute_georef_status_from_parts(
    *,
    has_transform: bool,
    has_crs: bool,
    rotated_dropped: bool = False,
) -> str:
    """Compute ``georef_status`` from raw booleans rather than a ``GeoInfo``.

    The VRT inline branches do not build a ``GeoInfo`` instance: they
    parse the VRT XML straight into ``geo_transform`` / ``crs_wkt``
    fields on a different dataclass. Calling :func:`_compute_georef_status`
    from those sites would require synthesising a fake ``GeoInfo`` for
    each branch. This helper takes the underlying booleans directly so
    the VRT paths and the ``_populate_attrs_from_geo_info`` path share
    the same decision rule without the intermediate object.

    Implementation note: kept as a thin boolean shim so VRT
    inline branches do not have to build a fake ``GeoInfo`` to feed
    :func:`resolve_georef`. The decision table here mirrors the
    reader-path branch in :func:`resolve_georef`.
    """
    if rotated_dropped:
        return GEOREF_STATUS_ROTATED_DROPPED
    if has_transform and has_crs:
        return GEOREF_STATUS_FULL
    if has_transform:
        return GEOREF_STATUS_TRANSFORM_ONLY
    if has_crs:
        return GEOREF_STATUS_CRS_ONLY
    return GEOREF_STATUS_NONE


def _populate_attrs_from_geo_info(attrs: dict, geo_info, *, window=None) -> None:
    """Populate ``attrs`` with all GeoTIFF metadata from ``geo_info``.

    Centralised so the eager numpy, dask, and GPU read paths emit the
    same attrs keys for the same input file. Mutates ``attrs`` in place.

    The ``nodata`` / ``masked_nodata`` pair is intentionally NOT set
    here because each caller pairs them with its own nodata-masking step
    via :func:`_set_nodata_attrs`. The pair carries two distinct
    signals: ``nodata`` is the declared file sentinel (always set when
    the source declared one), and ``masked_nodata`` is a boolean for
    whether the in-memory array has been NaN-masked.

    ``window`` is a ``(r0, c0, r1, c1)`` tuple for windowed reads; when
    set, the emitted ``attrs['transform']`` shifts the origin to the
    window's top-left. The eager path and the dask path (which threads
    ``window=`` through ``_read_geotiff_dask``) both pass
    the outer window through this helper so the resulting DataArray
    advertises the windowed transform. The GPU path does not currently
    expose a windowed read, so it passes ``window=None``.

    ``attrs['_xrspatial_geotiff_contract']`` is stamped unconditionally
    as the first step. Any pre-existing value on the passed-in dict is
    overwritten with the current ``_ATTRS_CONTRACT_VERSION``; callers
    pass freshly built dicts, so this is the intended behaviour.
    """
    # Compatibility shim: build a typed :class:`GeoTIFFMetadata` once
    # and fold it into the caller's dict. The two routes
    # (:func:`metadata_to_attrs` and the legacy field-by-field writes)
    # produce the same attrs surface; centralising on the record lets
    # the VRT path emit the same field set without copying this block.
    # The ``allow_rotated=True`` opt-in CRS-drop is handled
    # inside ``geo_info_to_metadata``. ``georef_status`` rides
    # on the record so the VRT path can stamp it via the same
    # marshalling step. See ``metadata_to_attrs``.
    md = geo_info_to_metadata(geo_info, window=window)
    attrs.update(metadata_to_attrs(md))


def _resolve_nodata_attr(attrs: dict):
    """Resolve a NoData sentinel from DataArray attrs.

    xrspatial's own readers always emit ``attrs['nodata']`` (a scalar),
    so that key is checked first for a clean intra-library round-trip.
    Falls back to two ecosystem conventions on miss:

    * ``attrs['nodatavals']`` -- rioxarray's per-band tuple. Returns
      the first entry that is not None, not non-numeric, and not NaN.
      In practice this is band 0 for almost every real file; the skip
      logic only matters when band 0 is missing a sentinel (NaN /
      None) while a later band declares one. Bands with mixed concrete
      sentinels are uncommon and would need an explicit ``nodata=``
      argument anyway.
    * ``attrs['_FillValue']`` -- CF-style xarray pipelines.

    Returns ``None`` when none of the keys carry a usable value. NaN
    entries in ``nodatavals`` are skipped rather than treated as a
    sentinel (NaN means "the float NaN is the sentinel", which is
    already the default and doesn't need a GDAL_NODATA tag).
    """
    nodata = attrs.get('nodata')
    if nodata is not None:
        try:
            float(nodata)
        except (TypeError, ValueError) as e:
            raise ValueError(_nodata_attr_non_numeric_msg('nodata', nodata)) from e
        return nodata

    vals = attrs.get('nodatavals')
    if vals is not None:
        try:
            seq = list(vals)
        except TypeError:
            seq = [vals]
        saw_non_numeric = False
        # Collect usable (numeric, non-NaN, non-None) entries first so we
        # can detect distinct per-band sentinels rather than silently
        # picking the first one. Issue #2514: the writer cannot represent
        # multiple sentinels in a single GDAL_NODATA tag, so flattening a
        # tuple like ``(-9999.0, -8888.0)`` to ``-9999.0`` drops band 1's
        # sentinel and turns its missing-data cells into real values on
        # the resulting file. The write-side validator catches this at
        # the boundary, but the resolver is also called from the writer
        # itself after validation, so a defensive check here guards
        # against accidental bypass paths.
        usable: list[tuple[float, Any]] = []
        for v in seq:
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                saw_non_numeric = True
                continue
            if np.isnan(fv):
                continue
            usable.append((fv, v))
        if usable:
            distinct: list[float] = []
            for fv, _ in usable:
                if not any(fv == d for d in distinct):
                    distinct.append(fv)
            if len(distinct) > 1:
                # Defense-in-depth raise: the writer-side validator
                # ``_check_write_distinct_per_band_nodatavals`` normally
                # fires first, but the resolver is also called from the
                # writer itself after validation. Share the message via
                # the helper in ``_errors`` so the two sites stay in sync.
                raise ConflictingNodataError(
                    _distinct_per_band_nodatavals_msg(vals, distinct)
                )
            return usable[0][1]
        # A tuple where every entry is non-numeric is almost certainly a
        # user error (typo, stringified sentinel) rather than a legitimate
        # "no sentinel" signal. Warn so the caller sees it, but still fall
        # through to the rest of the resolution chain rather than raising:
        # the rest of the function's contract is "skip non-numeric entries".
        if saw_non_numeric:
            warnings.warn(
                f"attrs['nodatavals']={vals!r} contained only non-numeric "
                f"entries; no usable sentinel could be resolved from it. "
                f"Pass ``nodata=`` explicitly or fix the attr.",
                UserWarning,
                stacklevel=2,
            )

    fill = attrs.get('_FillValue')
    if fill is not None:
        try:
            ffv = float(fill)
        except (TypeError, ValueError) as e:
            raise ValueError(_nodata_attr_non_numeric_msg('_FillValue', fill)) from e
        if np.isnan(ffv):
            return None
        return fill

    return None


def _nodata_attr_non_numeric_msg(attr_name: str, value) -> str:
    """Error string shared by the ``attrs['nodata']`` and ``attrs['_FillValue']``
    non-numeric branches in ``_resolve_nodata_attr``."""
    return (
        f"attrs[{attr_name!r}]={value!r} is not numeric "
        f"({type(value).__name__}). The writer needs a numeric "
        f"sentinel to compare against pixel values; passing a "
        f"non-numeric value would otherwise crash inside "
        f"``np.isnan`` with an opaque ufunc error. Drop the "
        f"attr, replace it with a numeric sentinel, or pass "
        f"``nodata=`` explicitly (issue #1973)."
    )


def _merge_friendly_extra_tags(extra_tags_list, attrs: dict) -> list | None:
    """Combine ``attrs['extra_tags']`` with friendly tag attrs.

    Synthesizes ``(tag_id, type_id, count, value)`` entries from
    ``attrs['image_description']`` (270, ASCII),
    ``attrs['extra_samples']`` (338, SHORT) and ``attrs['colormap']``
    (320, SHORT). An entry already present in ``extra_tags`` wins, so
    a verbatim round-trip stays byte-identical.
    """
    existing = list(extra_tags_list) if extra_tags_list else []
    seen_ids = {t[0] for t in existing}

    img_desc = attrs.get('image_description')
    if img_desc is not None and 270 not in seen_ids:
        s = str(img_desc)
        existing.append((270, _TIFF_ASCII, len(s) + 1, s))
        seen_ids.add(270)

    extra_samples = attrs.get('extra_samples')
    if extra_samples is not None and 338 not in seen_ids:
        try:
            vals = tuple(int(x) for x in extra_samples)
        except (TypeError, ValueError):
            vals = None
        if vals:
            value = vals if len(vals) > 1 else vals[0]
            existing.append((338, _TIFF_SHORT, len(vals), value))
            seen_ids.add(338)

    colormap = attrs.get('colormap')
    if colormap is not None and 320 not in seen_ids:
        try:
            cmap_vals = tuple(int(x) for x in colormap)
        except (TypeError, ValueError):
            cmap_vals = None
        if cmap_vals:
            value = cmap_vals if len(cmap_vals) > 1 else cmap_vals[0]
            existing.append((320, _TIFF_SHORT, len(cmap_vals), value))
            seen_ids.add(320)

    return existing or None


def _extract_rich_tags(attrs: dict) -> dict:
    """Extract the rich-tag set forwarded by the writers to ``write(...)``.

    Centralises the bookkeeping shared by :func:`to_geotiff`,
    :func:`_write_vrt_tiled`, and :func:`_write_geotiff_gpu`:

    * ``raster_type`` -- mapped from ``attrs['raster_type']`` ('point'
      becomes :data:`RASTER_PIXEL_IS_POINT`; everything else stays
      :data:`RASTER_PIXEL_IS_AREA`).
    * ``gdal_metadata_xml`` -- prefers ``attrs['gdal_metadata_xml']``;
      falls back to building XML from ``attrs['gdal_metadata']`` when
      it is a dict.
    * ``extra_tags`` -- ``attrs['extra_tags']`` folded with the friendly
      tag attrs (image_description / extra_samples / colormap) via
      :func:`_merge_friendly_extra_tags`.
    * ``x_resolution`` / ``y_resolution`` -- pass-through.
    * ``resolution_unit`` -- string label mapped to the integer tag id.

    Returns a kwargs dict ready to splat into ``write(...)``: every key
    matches the corresponding parameter name on
    :func:`xrspatial.geotiff._writer.write`.
    """
    raster_type = (RASTER_PIXEL_IS_POINT
                   if attrs.get('raster_type') == 'point'
                   else RASTER_PIXEL_IS_AREA)

    gdal_meta_xml = attrs.get('gdal_metadata_xml')
    if gdal_meta_xml is None:
        gdal_meta_dict = attrs.get('gdal_metadata')
        if isinstance(gdal_meta_dict, dict):
            from ._geotags import _build_gdal_metadata_xml
            gdal_meta_xml = _build_gdal_metadata_xml(gdal_meta_dict)

    extra_tags_list = _merge_friendly_extra_tags(
        attrs.get('extra_tags'), attrs)

    res_unit = None
    unit_str = attrs.get('resolution_unit')
    if unit_str is not None:
        res_unit = _RESOLUTION_UNIT_IDS.get(str(unit_str), None)

    return {
        'raster_type': raster_type,
        'gdal_metadata_xml': gdal_meta_xml,
        'extra_tags': extra_tags_list,
        'x_resolution': attrs.get('x_resolution'),
        'y_resolution': attrs.get('y_resolution'),
        'resolution_unit': res_unit,
    }


def _apply_eager_nodata_mask(arr, *, mask_sentinel, mask_nodata):
    """Apply the nodata-to-NaN mask on an eager (host-side) numpy buffer.

    Mirrors the inline block in ``open_geotiff`` so the eager helper can
    share one implementation. Returns ``(arr, nodata_pixels_present)``
    where ``arr`` is promoted from an integer dtype to float64 whenever
    ``mask_nodata`` is set and the sentinel is maskable (finite, integer,
    in-range), independent of whether any pixel matches. This matches the
    dask path (which declares float64 up front from the same gate) and
    rioxarray's ``masked=True`` (issue #2990). ``nodata_pixels_present``
    is the bool used to populate ``attrs['nodata_pixels_present']``;
    ``None`` means "no scan was appropriate for this dtype / sentinel
    combination."

    The sentinel is taken as the ``mask_sentinel`` parameter rather than
    being read from ``geo_info``. Three GPU eager sites derive it three
    different ways (``_mw_mask_nodata`` local, the CPU-fallback
    ``_cpu_fallback_geo._mask_nodata``, raw ``nodata``), so the helper
    accepts the sentinel value directly.
    """
    nodata_pixels_present: bool | None = None
    if mask_sentinel is None:
        return arr, nodata_pixels_present
    if mask_nodata:
        if arr.dtype.kind == 'f':
            if not np.isnan(mask_sentinel):
                mask_f = arr == arr.dtype.type(mask_sentinel)
                nodata_pixels_present = bool(mask_f.any())
                if nodata_pixels_present:
                    arr[mask_f] = np.nan
            else:
                # NaN-only sentinel on a float buffer: ``mask_nodata`` is
                # a no-op, but downstream may want to know if any NaN
                # pixels already exist in the source so the attr stays
                # informative.
                nodata_pixels_present = bool(np.isnan(arr).any())
        elif arr.dtype.kind in ('u', 'i'):
            # Integer arrays: convert to float to represent NaN. Gate on
            # finite + integer + in-range so a sentinel that cannot match
            # an integer pixel resolves to ``False`` rather than crashing
            # in the equality cast (mirrors the eager block in
            # ``open_geotiff``).
            if (np.isfinite(mask_sentinel)
                    and float(mask_sentinel).is_integer()):
                nodata_int = int(mask_sentinel)
                info = np.iinfo(arr.dtype)
                if info.min <= nodata_int <= info.max:
                    # Compute the sentinel mask at the integer dtype's
                    # native width BEFORE the float64 promotion. For
                    # int64/uint64 sentinels above 2**53 (INT64_MAX,
                    # UINT64_MAX, ...) the promotion rounds nearby valid
                    # values onto the sentinel's float64 representation,
                    # so a post-promotion compare masked valid pixels to
                    # NaN and diverged from the dask / GPU-chunked / VRT
                    # paths, which all compare at native width
                    # (issue #3098).
                    mask = arr == arr.dtype.type(nodata_int)
                    nodata_pixels_present = bool(mask.any())
                    # Promote to float64 whenever the sentinel is
                    # maskable, independent of whether any pixel matches.
                    # The dask path declares float64 up front from the
                    # same finite/integer/in-range gate (see
                    # ``_read_geotiff_dask``'s ``sentinel_fits_buffer``
                    # branch), and rioxarray's ``masked=True`` always
                    # promotes an integer source to float. Gating the
                    # promotion on a matching pixel made the eager output
                    # diverge (uint16 + ``masked_nodata=False``) from the
                    # lazy output (float64 + ``masked_nodata=True``) for
                    # the same file when no sentinel pixel was present
                    # (issue #2990). ``nodata_pixels_present`` still
                    # records whether a pixel matched.
                    arr = arr.astype(np.float64)
                    if nodata_pixels_present:
                        arr[mask] = np.nan
                else:
                    nodata_pixels_present = False
            else:
                nodata_pixels_present = False
    else:
        # ``mask_nodata=False``: do not rewrite pixels, but still surface
        # ``attrs['nodata_pixels_present']`` so callers know whether
        # literal sentinel pixels survive in the buffer.
        if arr.dtype.kind == 'f':
            if np.isnan(mask_sentinel):
                nodata_pixels_present = bool(np.isnan(arr).any())
            else:
                nodata_pixels_present = bool(
                    (arr == arr.dtype.type(mask_sentinel)).any()
                )
        elif arr.dtype.kind in ('u', 'i'):
            if (np.isfinite(mask_sentinel)
                    and float(mask_sentinel).is_integer()):
                nodata_int = int(mask_sentinel)
                info = np.iinfo(arr.dtype)
                if info.min <= nodata_int <= info.max:
                    nodata_pixels_present = bool(
                        (arr == arr.dtype.type(nodata_int)).any()
                    )
                else:
                    nodata_pixels_present = False
            else:
                nodata_pixels_present = False
    return arr, nodata_pixels_present


def _extract_scale_offset(gdal_metadata, band=None, *, malformed=False):
    """Pull SCALE / OFFSET from parsed GDAL_METADATA for ``mask_and_scale``.

    Returns ``(scale, offset)`` floats, defaulting to ``(1.0, 0.0)`` when the
    source carries no scale / offset. GDAL stores these in the GDAL_METADATA
    XML either as dataset-level ``SCALE`` / ``OFFSET`` items or per-band items
    keyed ``(name, band_index)`` by :func:`_parse_gdal_metadata`.

    Dataset-level values apply uniformly to the whole array and always win.
    Failing those, the per-band values are used:

    * When ``band`` selects a single band, that band's per-band value is used
      (or the default when that band carries none).
    * When ``band`` is ``None`` the full array is returned, so a single pair
      has to cover every band. If the per-band values disagree, applying any
      one of them silently corrupts the other bands, so this raises
      :class:`MixedBandMetadataError`. Uniform per-band values (or a single
      band's worth) are returned unchanged.

    Raises :class:`MalformedScaleOffsetError` when a ``SCALE`` or ``OFFSET``
    item is present but does not parse as a float, when ``SCALE`` is zero
    or non-finite, or when ``OFFSET`` is non-finite (issue #3104). An
    absent key keeps the 1.0 / 0.0 identity default.

    ``malformed=True`` signals that the source carried a GDAL_METADATA XML
    payload that did not parse (see :func:`_parse_gdal_metadata_strict`).
    Because the unparseable payload could have declared a scale / offset
    that is now lost, ``mask_and_scale`` fails closed with a
    :class:`MalformedScaleOffsetError` rather than reading the raw pixels
    as if no scaling were declared.
    """
    scale, offset = 1.0, 0.0
    # Check ``malformed`` before the empty-dict short-circuit below: an
    # unparseable payload yields ``gdal_metadata == {}``, so a guard that
    # returned the identity default on an empty dict first would silence
    # the rejection. Keep this check at the top.
    if malformed:
        raise MalformedScaleOffsetError(
            "GDAL_METADATA XML is malformed and could not be parsed. "
            "unpack=True cannot honour the scale / offset it may "
            "declare, so the read is refused rather than returning raw, "
            "unscaled pixels."
        )
    if not gdal_metadata:
        return scale, offset

    def _coerce(name, raw):
        # A key that is present but unparseable is rejected -- ``mask_and_scale``
        # asked us to honour the metadata, so a malformed value must not be
        # silently dropped and the raw pixels read as if clean.
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise MalformedScaleOffsetError(
                f"GDAL_METADATA {name} is not a number: {raw!r}. "
                "unpack=True cannot honour a malformed "
                f"{name}."
            ) from None

    def _resolve(name, default):
        # Dataset-level value applies to every band uniformly.
        if name in gdal_metadata:
            return _coerce(name, gdal_metadata[name])

        per_band = {
            key[1]: val
            for key, val in gdal_metadata.items()
            if isinstance(key, tuple) and len(key) == 2 and key[0] == name
        }
        if not per_band:
            return default

        if band is not None:
            # A specific band was selected upstream, so use its value
            # (or the default when that band carries none).
            if band not in per_band:
                return default
            return _coerce(name, per_band[band])

        coerced = [_coerce(name, per_band[i]) for i in sorted(per_band)]
        distinct = sorted(set(coerced))
        if len(distinct) > 1:
            from ._errors import MixedBandMetadataError
            raise MixedBandMetadataError(
                f"unpack=True but the source declares distinct "
                f"per-band {name} values {distinct!r}. Applying one band's "
                f"{name} to the whole array would silently corrupt the other "
                f"bands. Select a single band with band= to read it with its "
                f"own {name}, or drop unpack."
            )
        return distinct[0]

    scale = _resolve('SCALE', 1.0)
    offset = _resolve('OFFSET', 0.0)
    # A zero or non-finite SCALE (and a non-finite OFFSET) parses as a
    # float but cannot be honoured: ``data * 0 + offset`` collapses every
    # pixel to the offset, ``data * nan`` destroys the array, and the
    # inverse transform in ``_pack`` divides by SCALE. Treat these like
    # the unparseable case above and fail closed (issue #3104).
    if scale == 0.0 or not np.isfinite(scale):
        raise MalformedScaleOffsetError(
            f"GDAL_METADATA SCALE is {scale!r}. mask_and_scale=True "
            "cannot honour a zero or non-finite SCALE: applying it "
            "destroys the data and it has no inverse for pack=True."
        )
    if not np.isfinite(offset):
        raise MalformedScaleOffsetError(
            f"GDAL_METADATA OFFSET is {offset!r}. mask_and_scale=True "
            "cannot honour a non-finite OFFSET."
        )
    return scale, offset


def _pack_fill_nan(chunk, fill):
    """NaN -> sentinel fill for ``_pack``'s float targets, at the buffer
    level. Integer targets route through :func:`_pack_restore_int`
    instead, which assigns the sentinel at the target dtype's native
    width (issue #3264).

    ``DataArray.fillna`` routes through xarray's ``duck_array_ops.where``,
    which breaks on cupy-backed arrays (issue #3112): eager cupy hits
    xarray calling ``cupy.astype`` (AttributeError on cupy 13.6), and a
    dask+cupy graph feeds the host-side 0-d fill array into
    ``cupy.where`` (TypeError). Filling through a boolean mask on the
    raw buffer sidesteps ``where`` entirely: ``np.isnan`` dispatches to
    cupy via ``__array_ufunc__`` and the masked assignment takes a
    plain Python scalar on numpy and cupy alike.
    """
    if chunk.dtype.kind != 'f':
        # Integer buffers cannot hold NaN; nothing to fill (matches
        # ``fillna``'s no-op on non-float data).
        return chunk
    out = chunk.copy()
    out[np.isnan(out)] = float(fill)
    return out


def _pack_restore_int(chunk, fill, tgt_name):
    """NaN -> sentinel fill plus integer restore for ``_pack``, at the
    target dtype's native width.

    Filling the float64 buffer first and casting afterwards corrupts
    64-bit sentinels above 2**53: ``float(INT64_MAX)`` rounds to 2**63,
    the cast overflows, and the holes land on INT64_MIN (0 for
    UINT64_MAX) while the GDAL_NODATA tag still declares the original
    sentinel, so a masked re-read returns them as valid pixels (issue
    #3264). Instead, park the holes at zero for the round + cast, then
    assign the sentinel as a Python int -- exact at any width, the same
    way the eager and GPU writers fill via ``dtype.type(nodata)``.
    Works on numpy and cupy chunks alike: ``np.isnan`` dispatches on
    the array type and the masked assignments take Python scalars.

    Runs :func:`_pack_guard_int_range` between the round and the cast:
    this is the only integer restore the sentinel path takes, and the
    chunk is already integer-typed by the time ``_pack``'s body-level
    guard block runs, so the range / finiteness check (issue #3260)
    must happen here. ``+/-Inf`` is not NaN and therefore survives the
    hole mask; the parked holes hold 0.0, in range for every integer
    dtype, so they pass the guard.
    """
    tgt = np.dtype(tgt_name)
    if chunk.dtype.kind != 'f':
        # Integer buffers cannot hold NaN; nothing to fill.
        return chunk.astype(tgt)
    mask = np.isnan(chunk)
    out = chunk.copy()
    out[mask] = 0.0
    out = out.round()
    info = np.iinfo(tgt)
    _pack_guard_int_range(out, tgt.name, float(info.min),
                          float(int(info.max) + 1))
    out = out.astype(tgt)
    out[mask] = int(fill)
    return out


def _pack_guard_int_range(chunk, tgt_name, lo, hi_excl):
    """Per-chunk range / finiteness guard for ``_pack``'s integer restore.

    Runs after the round and before the ``astype``. Raises ``ValueError``
    when the chunk holds non-finite values or values outside the target
    integer dtype's range -- the cast would silently wrap them into
    valid-looking integers (issue #3260). Returns the chunk unchanged
    when clean.

    ``lo`` is ``iinfo.min`` and ``hi_excl`` is ``iinfo.max + 1``, both as
    floats. The exclusive upper bound matters for the 64-bit dtypes:
    ``float(iinfo.max)`` rounds up to ``2**63`` / ``2**64``, so a
    ``chunk > float(iinfo.max)`` test would pass a value that still wraps
    in the cast. ``iinfo.max + 1`` is a power of two for every integer
    dtype and therefore exact in float64. ``iinfo.min`` is likewise a
    power of two (or zero) and exact.

    Mapped over dask blocks so the scan fuses with the write's single
    compute (same shape as :func:`_pack_guard_no_nan`); numpy and cupy
    buffers are checked eagerly at ``_pack`` call time. ``np.isfinite``
    and the comparisons dispatch on the array type, so the one function
    covers all four backends.
    """
    if chunk.dtype.kind == 'f':
        bad = ~np.isfinite(chunk)
        bad |= chunk < lo
        bad |= chunk >= hi_excl
        if bool(bad.any()):
            raise ValueError(
                f"pack=True: data contains values that the packed "
                f"integer dtype {tgt_name} cannot represent (valid "
                f"range {int(lo)}..{int(hi_excl) - 1} after the "
                f"scale/offset reversal) or that are not finite; the "
                f"cast would silently wrap them. Clip the data to the "
                f"packed range or rescale before writing.")
    return chunk


def _pack_guard_no_nan(chunk, tgt_name):
    """Per-chunk NaN guard for ``_pack``'s no-sentinel integer restore.

    Mapped over dask-backed data so the scan runs inside the write's
    single compute instead of forcing a second full execution of the
    upstream graph at ``to_geotiff(pack=True)`` call time (issue
    #3235). Returns the chunk unchanged when clean; raises
    ``ValueError`` when it contains NaN. Works on numpy and cupy
    chunks alike -- ``np.isnan`` dispatches on the array type and the
    ``bool(...)`` reduction is a scalar sync per chunk.
    """
    if chunk.dtype.kind == 'f' and bool(np.isnan(chunk).any()):
        raise ValueError(
            f"pack=True: cannot restore integer dtype {tgt_name}: "
            "NaN pixels are present but no nodata sentinel is "
            "declared to fill them.")
    return chunk


def _pack(data, nodata=None):
    """Re-pack an ``unpack=True`` read for writing -- inverse of
    :func:`_extract_scale_offset`'s forward direction.

    Reverses the scale / offset applied on read (``(data - add_offset) /
    scale_factor``), fills NaN back to the nodata sentinel, and
    casts to the source dtype recorded in
    ``attrs['mask_and_scale_dtype']`` (falling back to the dtype of an
    integer ``attrs['nodata']`` for arrays read by an xrspatial release
    predating contract v5, and to the buffer's own dtype otherwise).
    Returns a new DataArray; ``data`` is left untouched.

    ``nodata`` is the writer's explicit ``nodata=`` kwarg. When given, it
    overrides the attrs sentinel as the NaN fill value, matching the
    "kwarg overrides attrs" precedence the GDAL_NODATA tag already
    follows -- otherwise the holes get filled with one value while the
    tag declares another and a ``masked=True`` re-read returns the holes
    as valid pixels (#3168). The pre-v5 dtype-width fallback still keys
    off the attrs sentinel: it is stored in the source dtype, while the
    kwarg is a plain Python number with no width information.

    The SCALE / OFFSET GDAL_METADATA tags are kept: the written file stores
    the raw packed integers and re-declares the scale, so reopening with
    ``unpack=True`` unpacks to the same values rather than scaling a
    second time. (The double-scale bug the round-trip caveat warns about
    comes from writing the *already-scaled* floats with the tags still on;
    reversing the scale first, as here, makes keeping the tags correct.)
    Per-band ``(SCALE, i)`` / ``(OFFSET, i)`` entries are an exception:
    after a band-subset read they still describe the *source's* band
    indices, which the written file no longer has (#3161). They are
    rewritten as dataset-level entries carrying the reversed pair -- the
    one pair that was applied to every pixel of this array -- so the
    packed file unpacks correctly whatever bands it carries.

    Raises ``ValueError`` when ``data`` carries no unpack state to
    reverse, when an integer dtype must be restored but NaN pixels are
    present with no declared sentinel to fill them, when the ``nodata``
    kwarg cannot be represented in the packed integer dtype (out of range
    or fractional) -- the cast would wrap or round the fill value away
    from what the GDAL_NODATA tag declares -- or when any pixel's packed
    value is non-finite or falls outside the packed integer dtype's
    range, where the cast would silently wrap it into a valid-looking
    integer (issue #3260).

    Error timing for the NaN-with-no-sentinel and packed-range cases
    depends on the
    backing: numpy-backed data raises here, at call time. dask-backed
    data defers the scan into the graph (one per-chunk check fused with
    the write's single compute) so the upstream graph is not executed a
    second time just to look for NaN (issue #3235); the ``ValueError``
    then surfaces from the compute that consumes the packed array,
    typically the ``to_geotiff`` write itself.
    """
    attrs = dict(data.attrs)
    scale = attrs.get('scale_factor', 1.0)
    offset = attrs.get('add_offset', 0.0)
    target = attrs.get('mask_and_scale_dtype')
    attrs_nodata = _resolve_nodata_attr(attrs)
    nodata_kwarg = nodata
    if nodata is None:
        nodata = attrs_nodata

    if ('scale_factor' not in attrs and target is None
            and not attrs.get('masked_nodata')):
        raise ValueError(
            "pack=True but the array carries no unpack state to "
            "reverse (no scale_factor / mask_and_scale_dtype / "
            "masked_nodata). It was not produced by "
            "open_geotiff(unpack=True).")

    # ``_extract_scale_offset`` rejects these on read (issue #3104), so
    # they can only appear via hand-edited attrs. Refuse rather than
    # divide by zero below and silently write a corrupt file.
    if scale == 0 or not np.isfinite(scale) or not np.isfinite(offset):
        raise ValueError(
            f"pack=True cannot reverse scale_factor={scale!r} / "
            f"add_offset={offset!r}: the inverse transform divides by "
            "scale_factor, so it must be non-zero and finite, and "
            "add_offset must be finite.")

    out = (data - offset) / scale

    if (target is None and attrs_nodata is not None
            and np.asarray(attrs_nodata).dtype.kind in ('i', 'u')):
        # Best-effort recovery for pre-v5 reads of integer sources: the
        # attrs sentinel is stored in the source dtype, so its width is
        # the source width. Float sentinels are excluded -- a reader
        # stores them as plain Python floats, so their width says
        # float64 regardless of the source's width (issue #3080); the
        # buffer below is the better signal because float sources are
        # never promoted on read. (The ``nodata`` kwarg carries no width
        # information, so it never feeds this fallback.)
        target = np.asarray(attrs_nodata).dtype.name
    tgt = np.dtype(target) if target is not None else np.dtype(str(out.dtype))

    if nodata_kwarg is not None and tgt.kind in ('i', 'u'):
        # The attrs sentinel fits the source dtype by construction (the
        # reader validated it against GDAL_NODATA). The kwarg has no such
        # guarantee: an out-of-range or fractional value would wrap /
        # round in the fill below, silently recreating the fill-vs-tag
        # disagreement this override exists to prevent. Compare as
        # integers, not through float(): float64 cannot represent 64-bit
        # values above 2**53, so a float round-trip rejected INT64_MAX
        # even though it fits int64 exactly (issue #3264).
        info = np.iinfo(tgt)
        try:
            iv = int(nodata_kwarg)
            representable = (iv == nodata_kwarg
                             and info.min <= iv <= info.max)
        except (OverflowError, ValueError):
            representable = False  # inf / nan
        if not representable:
            raise ValueError(
                f"pack=True: nodata={nodata_kwarg!r} cannot be represented "
                f"in the packed integer dtype {tgt.name} (valid range "
                f"{info.min}..{info.max}). The filled holes would not match "
                f"the GDAL_NODATA tag; pass a sentinel that fits the packed "
                f"dtype.")

    if nodata is not None:
        # Restore the masked sentinel for every dtype. Integers can't hold
        # NaN at all; floats would otherwise write NaN pixels under a
        # GDAL_NODATA tag that still declares the sentinel, leaving an
        # inconsistent file that non-masking readers misread (#3078).
        # Not ``fillna``: that routes through xarray's where(), which
        # crashes on cupy-backed arrays (#3112). The helpers fill at the
        # buffer level on all four backends; dask backings keep the fill
        # lazy via the same map_blocks shape as the no-sentinel guard
        # below.
        if tgt.kind in ('i', 'u'):
            # Integer restore: fill, round, and cast in one step so the
            # sentinel is assigned at the target dtype's native width.
            # Filling the float64 buffer first corrupts 64-bit sentinels
            # above 2**53 -- INT64_MAX wrapped to INT64_MIN in the cast
            # while GDAL_NODATA kept declaring INT64_MAX (issue #3264).
            if hasattr(out.data, 'dask'):
                out = out.copy(data=out.data.map_blocks(
                    _pack_restore_int, nodata, tgt.name,
                    dtype=tgt, meta=out.data._meta.astype(tgt)))
            else:
                out = out.copy(
                    data=_pack_restore_int(out.data, nodata, tgt.name))
        elif hasattr(out.data, 'dask'):
            out = out.copy(data=out.data.map_blocks(
                _pack_fill_nan, nodata,
                dtype=out.dtype, meta=out.data._meta))
        else:
            out = out.copy(data=_pack_fill_nan(out.data, nodata))
    elif tgt.kind in ('i', 'u'):
        # No sentinel exists to fill an integer's holes, so a NaN pixel
        # must abort the pack: the ``astype`` below would silently wrap
        # it into a valid-looking integer. This guard runs on every
        # no-sentinel integer pack, success path included. numpy-backed
        # data scans eagerly and raises here, at call time. dask-backed
        # data maps the guard into the graph instead: an eager
        # ``isnull().any()`` executed the whole upstream graph once for
        # the scan and the writer executed it again for the pixels
        # (issue #3235), so the per-chunk check now raises from the
        # write's single compute.
        if hasattr(out.data, 'dask'):
            out = out.copy(data=out.data.map_blocks(
                _pack_guard_no_nan, tgt.name,
                dtype=out.dtype, meta=out.data._meta))
        else:
            # Buffer-level scan, not ``bool(out.isnull().any())``: the
            # xarray reduction materialises through ``np.asarray`` and
            # raised TypeError on cupy-backed eager input, crashing
            # every no-sentinel integer pack on the gpu backend before
            # the scan could answer (#3260). ``_pack_guard_no_nan``
            # dispatches ``np.isnan`` on the array type, so numpy and
            # cupy take the same path here.
            _pack_guard_no_nan(out.data, tgt.name)

    # The integer-with-sentinel path above already rounded and cast at
    # the chunk level; its ``out`` is integer-typed here, so the round
    # is skipped and the astype is a no-op cast.
    if tgt.kind in ('i', 'u') and out.dtype.kind == 'f':
        out = out.round()
        # Out-of-range and non-finite packed values would silently wrap
        # in the astype below, exactly the corruption the NaN guard
        # above exists to prevent: a finite value whose packed form
        # exceeds the dtype range wraps (40000 -> -25536 in int16), and
        # +/-Inf casts to a platform-defined integer (issue #3260).
        # Runs after the round so a value like 32767.4 that rounds back
        # into range is not rejected. Same eager/lazy split as the NaN
        # guard: numpy / cupy raise here, dask raises from the write's
        # single compute.
        info = np.iinfo(tgt)
        lo = float(info.min)
        hi_excl = float(int(info.max) + 1)
        if hasattr(out.data, 'dask'):
            out = out.copy(data=out.data.map_blocks(
                _pack_guard_int_range, tgt.name, lo, hi_excl,
                dtype=out.dtype, meta=out.data._meta))
        else:
            _pack_guard_int_range(out.data, tgt.name, lo, hi_excl)

    out = out.astype(tgt)

    # Rewrite stale per-band SCALE / OFFSET entries (#3161). After a
    # band-subset read the kept GDAL_METADATA still describes the source's
    # band indices, so re-reading the packed file would raise
    # MixedBandMetadataError or silently apply another band's scale. The
    # reversed (scale, offset) pair is the single pair that was applied to
    # every pixel of this array, so dataset-level entries carrying it are
    # correct for any band layout (and _extract_scale_offset gives them
    # precedence). Dataset-level-only metadata is already index-free and
    # stays verbatim, preserving the raw XML. Scoped to arrays that carry
    # unpack state: a plain masked read never applied the per-band scale,
    # so collapsing its entries would destroy valid source metadata.
    # Other per-band items (band descriptions, STATISTICS_*) are left
    # as-is on purpose: they don't affect pixel values and can't be
    # re-indexed without knowing which band was read.
    gdal_md = attrs.get('gdal_metadata')
    if (isinstance(gdal_md, dict)
            and ('scale_factor' in attrs or 'mask_and_scale_dtype' in attrs)):
        per_band_keys = [
            key for key in gdal_md
            if (isinstance(key, tuple) and len(key) == 2
                and key[0] in ('SCALE', 'OFFSET'))
        ]
        if per_band_keys:
            new_md = {key: val for key, val in gdal_md.items()
                      if key not in per_band_keys}
            # Keep an existing dataset-level value verbatim (it won on
            # read, so it already equals the applied pair). Identity
            # values are dropped rather than written: absent and 1.0 / 0.0
            # read back the same.
            if 'SCALE' not in new_md and scale != 1.0:
                new_md['SCALE'] = str(float(scale))
            if 'OFFSET' not in new_md and offset != 0.0:
                new_md['OFFSET'] = str(float(offset))
            attrs['gdal_metadata'] = new_md
            # The raw XML still carries the stale per-band items and the
            # writer prefers it over the dict; drop it so GDAL_METADATA is
            # rebuilt from the rewritten dict.
            attrs.pop('gdal_metadata_xml', None)

    # Drop the read-side lifecycle attrs that describe the now-reversed
    # transform. The SCALE / OFFSET GDAL_METADATA stays so the packed file
    # still declares how to unpack. ``masked_nodata`` flips to False: the
    # buffer now carries the literal sentinel, not NaN.
    for key in ('scale_factor', 'add_offset', 'mask_and_scale_dtype'):
        attrs.pop(key, None)
    if 'masked_nodata' in attrs:
        attrs['masked_nodata'] = False
    if nodata_kwarg is not None:
        # The buffer's holes now carry the kwarg sentinel, so the attr
        # follows it. Downstream writers receive the kwarg and never
        # consult the attr, but keeping the intermediate self-consistent
        # means a path that forgot to thread the kwarg would still tag
        # the value the pixels actually hold.
        attrs['nodata'] = nodata_kwarg

    out.attrs = attrs
    out.name = data.name
    return out


def _finalize_eager_read(
    arr,
    *,
    geo_info,
    nodata,
    mask_sentinel,
    mask_nodata,
    dtype,
    window,
    name,
    allow_rotated: bool = False,
    allow_unparseable_crs: bool = False,
    attrs_in: dict | None = None,
    mask_and_scale: bool = False,
    parse_coordinates: bool = True,
    band: int | None = None,
):
    """Validate, populate attrs, mask, cast, and build an eager DataArray.

    Ties together the steps every eager read path runs after the bytes
    land in a host (or cupy) buffer:

    1. :func:`_validate_read_geo_info` -- runs first so a rejected file
       does not leak a partially-populated attrs dict.
    2. :func:`_populate_attrs_from_geo_info` -- writes the canonical attrs
       (transform / crs / georef_status / etc.) onto a fresh dict.
    3. Mask nodata pixels to NaN using ``mask_sentinel`` when
       ``mask_nodata=True`` and the source declared one. Records the
       ``nodata_pixels_present`` bool either way.
    4. Cast to ``dtype`` when explicit; record ``nodata_dtype_cast``.
    5. :func:`_set_nodata_attrs` -- stamps the nodata lifecycle attrs.
    6. Build an :class:`xarray.DataArray` with coords from
       :func:`_coords_from_geo_info`.

    The ``mask_sentinel`` parameter is intentionally separate from
    ``geo_info.nodata`` because the three GPU eager sites derive it three
    different ways (``_mw_mask_nodata`` local on the stripped path, the
    CPU-fallback ``_cpu_fallback_geo._mask_nodata`` on the tiled path,
    raw ``nodata`` on the CPU-decode-then-upload path for URL / fsspec
    sources). Read paths that don't need MinIsWhite inversion can pass
    ``mask_sentinel=nodata``.

    The host-side mask block also runs on CuPy arrays via numpy
    duck-typing, so the GPU eager backends route through
    this helper without a separate GPU mask kernel.

    Returns a :class:`xarray.DataArray` ready for the caller to return
    from the read function. The caller assembles the dask graph
    separately when a lazy backend is in play; this helper is eager-only.

    ``attrs_in`` is shallow-copied via ``dict(attrs_in)``. Nested values
    (e.g. ``extra_tags`` list, ``gdal_metadata`` dict) are shared between
    the caller's seed dict and the returned DataArray's attrs; mutating
    a nested value after the call propagates both ways. Callers that
    care about isolation can ``copy.deepcopy(attrs_in)`` first.
    """
    # Validate first so partial attrs never leak.
    _validate_read_geo_info(
        geo_info, window=window,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
    )

    # Populate attrs from geo_info onto a fresh dict (or onto a
    # caller-supplied seed dict, which lets the GPU/VRT migration carry
    # backend-specific keys through without bypassing the helper).
    attrs: dict = dict(attrs_in) if attrs_in else {}
    _populate_attrs_from_geo_info(attrs, geo_info, window=window)

    # ``mask_and_scale`` implies masking (rioxarray applies scale / offset
    # AND masks the nodata sentinel to NaN), so fold it into the mask gate.
    effective_mask = mask_nodata or mask_and_scale

    # Remember the source dtype before masking / scaling can promote an
    # integer buffer to float so ``to_geotiff(pack=True)`` can restore it
    # (issue #3064). Stamped below only when the promotion actually ran.
    pre_promote_dtype = np.dtype(str(arr.dtype))

    # Apply the nodata-to-NaN mask (or compute pixels_present
    # without rewriting if masking is off). Skipped entirely when
    # the source declared no sentinel.
    nodata_pixels_present: bool | None = None
    if nodata is not None:
        arr, nodata_pixels_present = _apply_eager_nodata_mask(
            arr, mask_sentinel=mask_sentinel, mask_nodata=effective_mask,
        )

    # ``mask_and_scale``: apply ``data * scale + offset`` from the source's
    # GDAL_METADATA. Runs before the caller's ``dtype=`` cast so a
    # ``dtype=<integer>`` request raises the same float-to-int ValueError the
    # mask path raises (scaling promotes to float).
    if mask_and_scale:
        scale, offset = _extract_scale_offset(
            getattr(geo_info, 'gdal_metadata', None), band=band,
            malformed=getattr(geo_info, 'gdal_metadata_malformed', False))
        if scale != 1.0 or offset != 0.0:
            if arr.dtype.kind != 'f':
                arr = arr.astype(np.float64)
            arr = arr * scale + offset
            attrs['scale_factor'] = scale
            attrs['add_offset'] = offset

    # Record the source dtype so ``to_geotiff(pack=True)`` can re-pack the
    # decoded array (issue #3064). Scoped to ``mask_and_scale`` reads (not a
    # plain ``masked`` read) since ``pack`` is the inverse of
    # ``mask_and_scale``; the attr name says as much. Checked before the
    # caller ``dtype=`` cast so it captures the on-disk dtype, not the
    # caller's requested cast. Integer sources record it when the read
    # promoted them to float. Float sources are never promoted, but record
    # it too whenever the read carries unpack state to reverse (a declared
    # sentinel or a non-identity scale / offset): without the attr,
    # ``_pack`` falls back to the nodata scalar's dtype and widens a
    # float32 source to float64 (issue #3080).
    if mask_and_scale and np.dtype(str(arr.dtype)).kind == 'f':
        if (pre_promote_dtype.kind != 'f'
                or nodata is not None or 'scale_factor' in attrs):
            attrs['mask_and_scale_dtype'] = pre_promote_dtype.name

    # Caller-requested dtype cast (post-mask so the integer
    # promotion above runs first). ``_validate_dtype_cast`` lives in
    # ``_validation``; local import keeps ``_attrs`` free of a top-level
    # validation dependency for parity with ``_validate_read_geo_info``.
    dtype_cast_attr: str | None = None
    if dtype is not None:
        from ._validation import _validate_dtype_cast
        target = np.dtype(dtype)
        _validate_dtype_cast(np.dtype(str(arr.dtype)), target)
        arr = arr.astype(target)
        dtype_cast_attr = target.name

    # Stamp the nodata lifecycle attrs. ``masked`` is True iff
    # the caller opted into masking AND the final buffer dtype is float.
    # ``_apply_eager_nodata_mask`` promotes a maskable integer source to
    # float64 whenever masking is on (issue #2990), so an ``int`` buffer +
    # ``mask_nodata=True`` here means the sentinel was unmaskable
    # (out-of-range / non-finite / fractional) and could never match, not
    # that masking was disabled. Either way ``masked`` is correctly False
    # because the literal sentinel still occupies its integer slot.
    _set_nodata_attrs(
        attrs, nodata,
        masked=(effective_mask and np.dtype(str(arr.dtype)).kind == 'f'),
        pixels_present=nodata_pixels_present,
        dtype_cast=dtype_cast_attr,
    )

    # Build the DataArray. ``_coords_from_geo_info`` honours the
    # windowed-read contract (origin shifted to the window's top-left).
    # ``parse_coordinates=False`` skips the x / y coordinate arrays
    # (matching rioxarray); the transform / crs attrs still carry the
    # georeferencing, and the band coord is kept.
    height, width = arr.shape[:2]
    coords = (
        _coords_from_geo_info(geo_info, height, width, window=window)
        if parse_coordinates else {}
    )
    if arr.ndim == 3:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(arr.shape[2])
    else:
        dims = ['y', 'x']

    return xr.DataArray(arr, dims=dims, coords=coords, name=name, attrs=attrs)


def _finalize_lazy_read_attrs(
    *,
    geo_info,
    nodata,
    mask_nodata,
    graph_dtype,
    caller_dtype=None,
    window,
    allow_rotated: bool = False,
    allow_unparseable_crs: bool = False,
    band_nodata: str | None = None,
    band_nodata_values: list | None = None,
    attrs_in: dict | None = None,
    pixels_present: bool | None = None,
):
    """Validate and populate attrs for dask-style lazy reads.

    The lazy counterpart of :func:`_finalize_eager_read`. The dask +
    dask-GPU backends cannot
    fold the nodata mask into a single eager step because masking runs
    per-chunk inside the graph; they only need the attrs side of the
    pipeline. This helper:

    1. :func:`_validate_read_geo_info` -- runs first so partial attrs
       never leak on validation failure.
    2. :func:`_populate_attrs_from_geo_info` -- writes the canonical
       attrs onto a fresh dict.
    3. :func:`_set_nodata_attrs` -- ``masked`` is True iff the caller
       opted into masking AND the graph dtype is float.
       ``dtype_cast`` is recorded when ``caller_dtype`` is not ``None``
       and ``nodata`` is declared. ``pixels_present`` defaults to
       ``None`` per the documented dask contract (a strict per-chunk
       reduction would force eager ``.compute()`` and break the lazy
       contract, so the attr stays absent on lazy outputs). Eager-VRT
       callers that already computed the presence bool via a single
       decode pass can pass it through this kwarg so the attr is stamped
       via the same finalization helper rather than written ad-hoc
       post-call.

    Returns the attrs ``dict`` only; the caller assembles the dask graph
    and builds the :class:`xarray.DataArray` itself, so this helper
    deliberately does not touch arrays or coords.

    Two dtype parameters, split because every lazy/VRT call
    site needs to keep them separate:

    * ``graph_dtype`` -- the resolved graph dtype the backend settled
      on (e.g. ``target_dtype`` after the int->float64 auto-promotion
      that ``mask_nodata=True`` triggers on int sources). Drives
      ``masked`` via ``np.dtype(graph_dtype).kind == 'f'``. ``None``
      means the caller did not resolve a graph dtype, which forces
      ``masked=False``.
    * ``caller_dtype`` -- the caller's ``dtype=`` kwarg verbatim, or
      ``None`` when the caller did not request a cast. Drives
      ``nodata_dtype_cast`` so the attr records caller intent, never
      the masking-induced auto-promotion.

    When ``nodata is None`` both the masking attrs (``masked_nodata``)
    and the cast attr (``nodata_dtype_cast``) stay absent regardless of
    the two dtype arguments -- ``_set_nodata_attrs`` short-circuits
    because the sentinel-lifecycle attrs only make sense when a
    sentinel is declared.

    The eager VRT call sites and the dask call sites both pass these
    asymmetrically (``graph_dtype`` is the pre-cast / pre-promotion
    dtype, ``caller_dtype`` is the user kwarg) so that a caller
    ``dtype=`` cast never flips ``masked_nodata``.

    ``attrs_in`` is shallow-copied via ``dict(attrs_in)``. Nested values
    are shared between the caller's seed dict and the returned attrs;
    callers that care about isolation can ``copy.deepcopy(attrs_in)``
    first.

    ``band_nodata`` and ``band_nodata_values`` forward through to
    :func:`_validate_read_geo_info` so VRT callers can hand the
    mixed-band check the context it needs. Non-VRT callers omit them
    and the mixed-band check short-circuits.
    """
    _validate_read_geo_info(
        geo_info, window=window,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
        band_nodata=band_nodata,
        band_nodata_values=band_nodata_values,
    )

    attrs: dict = dict(attrs_in) if attrs_in else {}
    _populate_attrs_from_geo_info(attrs, geo_info, window=window)

    # ``masked`` mirrors the eager helper's rule and the existing dask
    # call site contract: the graph applies masking per-chunk only when
    # ``mask_nodata=True`` AND the graph dtype is float, so an int graph
    # with ``mask_nodata=True`` still carries literal sentinel values.
    # ``graph_dtype`` is the resolved graph dtype; the dask backend
    # promotes int -> float64 before calling this helper when the
    # caller wants masking on an int source.
    if graph_dtype is None:
        masked = False
    else:
        masked = bool(mask_nodata and np.dtype(graph_dtype).kind == 'f')

    # ``dtype_cast`` records the caller's explicit ``dtype=`` so
    # consumers can tell float-because-masked from float-because-cast.
    # The masking-induced graph-dtype promotion never leaks into the
    # attr -- that is the whole reason ``caller_dtype`` is separate
    # from ``graph_dtype``.
    dtype_cast_attr = (
        np.dtype(caller_dtype).name if caller_dtype is not None else None
    )

    _set_nodata_attrs(
        attrs, nodata,
        masked=masked,
        pixels_present=pixels_present,
        dtype_cast=dtype_cast_attr,
    )

    return attrs
