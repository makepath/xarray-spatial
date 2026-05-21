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
(currently ``3``). Consumers can branch on this integer if the tier
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
  "no declared sentinel" signal. See ``_set_nodata_attrs``.
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

from ._coords import (
    coords_from_geo_info as _coords_from_geo_info,
    transform_tuple_from_pixel_geometry as _transform_tuple_from_pixel_geometry,
)
from ._geotags import (
    RASTER_PIXEL_IS_AREA,
    RASTER_PIXEL_IS_POINT,
    _NO_GEOREF_KEY,
)


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


# Tiered feature inventory for the public geotiff surface (issue #2137).
# Defined in ``_attrs.py`` (not the package ``__init__.py``) so the writers
# can import it at module scope without a circular dependency: the package
# ``__init__`` already imports the writers. The package re-exports
# ``SUPPORTED_FEATURES`` so the public API stays
# ``xrspatial.geotiff.SUPPORTED_FEATURES``.
#
# See ``xrspatial/geotiff/__init__.py`` for the per-tier semantics; the
# inline comments here track the codec/reader/writer split used by the
# user-guide notebook table.
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
    # opt-in (issue #1845). Not covered by ``allow_experimental_codecs``.
    'codec.jpeg': 'internal_only',
    # Read paths.
    'reader.local_file': 'stable',
    'reader.fsspec': 'advanced',
    'reader.http': 'advanced',
    'reader.vrt': 'advanced',
    'reader.sidecar_ovr': 'advanced',
    'reader.allow_rotated': 'advanced',
    'reader.allow_unparseable_crs': 'advanced',
    'reader.gpu': 'experimental',
    # Write paths.
    'writer.local_file': 'stable',
    'writer.cog': 'advanced',
    'writer.overviews': 'advanced',
    'writer.bigtiff': 'advanced',
    'writer.gpu': 'experimental',
    'writer.gdal_metadata_xml': 'experimental',
    'writer.extra_tags': 'experimental',
}


# Tier 3 codec names (lower-cased) gated behind
# ``allow_experimental_codecs`` on the writers. Derived from
# ``SUPPORTED_FEATURES`` so the gate cannot drift from the docs.
_EXPERIMENTAL_CODECS = frozenset(
    name.split('.', 1)[1].lower()
    for name, tier in SUPPORTED_FEATURES.items()
    if name.startswith('codec.') and tier == 'experimental'
)


# TIFF type ids needed when synthesizing extra_tags entries from attrs.
_TIFF_BYTE = 1
_TIFF_ASCII = 2
_TIFF_SHORT = 3


# Contract version emitted on every read; bumped when the attrs contract
# changes. Downstream code reads ``attrs['_xrspatial_geotiff_contract']``
# to learn which attrs-contract revision produced the array. See issue
# #1984 and ``docs/source/user_guide/attrs_contract.rst``.
#
# Version 2 (issue #2016) drops the 13 deprecated GeoKey-derived and
# matplotlib-colormap attrs that v1 still emitted under a
# ``DeprecationWarning``. Downstream code that read those keys via
# ``attrs[key]`` now sees ``KeyError`` rather than the deprecated value.
#
# Version 3 (issue #2136) adds ``attrs['georef_status']`` to the canonical
# tier. Existing keys (``crs``, ``crs_wkt``, ``transform``, the
# ``_xrspatial_no_georef`` marker) keep their pre-v3 shape so downstream
# code that branches on them still works; the new attr is additive and
# disambiguates ``crs_only`` from ``none`` and ``rotated_dropped`` from
# the truly-no-transform case.
#
# Version 4 (issue #2129) adds ``attrs['rotated_affine']`` for the
# ``allow_rotated=True`` opt-in path. The 6-tuple is read-only -- the
# writer drops it on round-trip until ``to_geotiff`` grows a
# ``ModelTransformationTag`` emit path (#2115 follow-up). Existing keys
# keep their pre-v4 shape.
_ATTRS_CONTRACT_VERSION = 4


# Canonical ``attrs['georef_status']`` values (issue #2136). One attr
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
    """Typed internal record for GeoTIFF read/write metadata (issue #2139).

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

    # NoData semantics (issue #1988 / #2092)
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

    # Canonical reader-state classifier (issue #2136). Carried on the
    # record so the eager / dask / GPU / VRT read paths all stamp it via
    # the same :func:`metadata_to_attrs` marshalling step instead of
    # branching on attrs after the dict has been built.
    georef_status: str | None = None

    # Rotated 6-tuple from ``ModelTransformationTag`` on the
    # ``allow_rotated=True`` opt-in path (issue #2129). Carried on the
    # record so the eager / dask / GPU / VRT read paths emit
    # ``attrs['rotated_affine']`` through the same marshalling step.
    # Read-only: :func:`attrs_to_metadata` intentionally does NOT
    # populate this field from incoming attrs so the writer keeps
    # dropping the rotation on round-trip until ``to_geotiff`` learns to
    # emit ``ModelTransformationTag`` (#2115 follow-up).
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

    # ``allow_rotated=True`` opt-in path (#2115): the parser returns a
    # GeoTransform with ``rotated_affine`` set and ``has_georef=False``.
    # The rotated 6-tuple cannot be expressed as an axis-aligned
    # rasterio transform, so the writer cannot round-trip it via
    # ``attrs['transform']``. Per the documented contract on
    # ``open_geotiff(allow_rotated=True)``, CRS attrs are dropped on
    # this path too -- otherwise downstream code that gates on
    # ``"crs" in da.attrs`` treats the array as spatially meaningful
    # while the actual mapping is gone (#2126).
    rotated_optin = (
        src_t is not None
        and getattr(src_t, 'rotated_affine', None) is not None
        and not has_georef
    )
    crs_epsg = None if rotated_optin else geo_info.crs_epsg
    crs_wkt = None if rotated_optin else geo_info.crs_wkt

    # Surface the rotated 6-tuple on the public attrs (issue #2129) so
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
        # ``georef_status`` (#2136) is computed off the unmodified
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

    # ``georef_status`` (#2136) is stamped before the optional CRS /
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
    # would emit ``_NO_GEOREF_KEY`` on a georef'd VRT array. See #2139.
    if md.transform is not None and md.has_georef:
        attrs['transform'] = md.transform
    elif not md.has_georef:
        attrs[_NO_GEOREF_KEY] = True

    # ``rotated_affine`` (issue #2129) rides alongside the
    # ``_xrspatial_no_georef`` marker on the ``allow_rotated=True`` path
    # so callers can recover the rotated mapping. Only set on read; the
    # writer-side :func:`attrs_to_metadata` deliberately does not parse
    # it back, so a read-then-write round-trip drops the rotation until
    # the writer grows ``ModelTransformationTag`` emit support (#2115).
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
      ``test_crs_arg_validation_1971.py``.
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

    * ``masked_nodata`` missing -> default True. Pre-#1988 behaviour:
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

    ``pixels_present`` is the lifecycle signal added in issue #2135. If
    not ``None``, the read path computed whether the read window
    contained at least one pixel matching the declared sentinel before
    masking; the value is forwarded to ``attrs['nodata_pixels_present']``
    so consumers can answer "any nodata in this tile" without scanning
    the buffer. Pass ``None`` when the backend cannot cheaply produce
    the value (e.g. dask, where a strict per-chunk reduction would
    force eager compute).

    ``dtype_cast`` is the second lifecycle signal added in issue #2135.
    If the caller passed an explicit ``dtype=`` kwarg, the backend
    forwards the resolved target dtype string (e.g. ``"float64"``) so
    consumers can distinguish "float because masking promoted it" from
    "float because the caller cast it". ``None`` means no caller cast
    happened; the attr is omitted in that case.

    Contract (splits the two meanings previously fused into
    ``attrs['nodata']`` per issue #1988, extended for #2135):

    * ``attrs['nodata']`` -- declared file sentinel, as a scalar of the
      source dtype. Set whenever the source declared one, regardless of
      whether the array is float-with-NaN or int-with-sentinels.
    * ``attrs['masked_nodata']`` -- the ``masked`` value the caller
      passed, coerced to bool. Only emitted when ``nodata is not
      None``; absence of the flag means there is no declared sentinel.
    * ``attrs['nodata_pixels_present']`` (additive, #2135) -- bool,
      only emitted when ``nodata is not None`` and ``pixels_present``
      is not ``None``. Tracks whether the read window contained any
      sentinel pixel before masking.
    * ``attrs['nodata_dtype_cast']`` (additive, #2135) -- string dtype
      name (e.g. ``"float64"``), only emitted when ``nodata is not
      None`` and ``dtype_cast`` is not ``None``. Records that a
      caller-requested cast happened after masking.

    Pre-#2092 the helper inferred ``masked`` from the final array
    dtype, which lied when ``mask_nodata=False`` left literal sentinel
    values in a float buffer; downstream code that trusted the attr
    treated those literal values as already-NaN. The eager, dask, GPU,
    and VRT paths now compute ``masked`` as
    ``mask_nodata and final_dtype.kind == 'f'``. See issue #2092.
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
    """Run issue #1987 read-side ambiguous-metadata checks against ``geo_info``.

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
    of dispatching the mixed-band check as a no-op (issue #2210).

    Raises whichever ``GeoTIFFAmbiguousMetadataError`` subclass a
    registered check picks. The hook is a no-op when no check is
    registered, so callers can use this helper unconditionally without
    coupling each backend to the current check list.

    Note: the transform tuple built here is always axis-aligned
    (``b == 0`` / ``d == 0``) because ``_transform_tuple_from_pixel_geometry``
    only carries origin + pixel size, and the upstream TIFF reader
    rejects rotated ``ModelTransformationTag`` entries with
    ``NotImplementedError`` in ``_geotags._extract_transform_and_georef``
    before we reach this helper. The rotated-transform check therefore
    fires only on the VRT path, which builds its context from the GDAL
    ``geo_transform`` via ``_gdal_geotransform_to_affine_tuple``.
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
    validate_read_metadata({
        'allow_rotated': allow_rotated,
        'allow_unparseable_crs': allow_unparseable_crs,
        'transform': transform_for_check,
        'crs_wkt': geo_info.crs_wkt,
        'band_nodata': band_nodata,
        'band_nodata_values': band_nodata_values,
    })


def _compute_georef_status(geo_info) -> str:
    """Classify ``geo_info`` into one of the five ``georef_status`` values.

    See the module docstring and issue #2136 for the full rationale. The
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
    """
    transform = getattr(geo_info, 'transform', None)
    rotated_affine = (
        getattr(transform, 'rotated_affine', None)
        if transform is not None else None
    )
    if rotated_affine is not None:
        return GEOREF_STATUS_ROTATED_DROPPED
    has_georef = bool(getattr(geo_info, 'has_georef', False))
    has_crs = (
        getattr(geo_info, 'crs_epsg', None) is not None
        or getattr(geo_info, 'crs_wkt', None) is not None
    )
    if has_georef and has_crs:
        return GEOREF_STATUS_FULL
    if has_georef:
        return GEOREF_STATUS_TRANSFORM_ONLY
    if has_crs:
        return GEOREF_STATUS_CRS_ONLY
    return GEOREF_STATUS_NONE


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
    whether the in-memory array has been NaN-masked (issue #1988).

    ``window`` is a ``(r0, c0, r1, c1)`` tuple for windowed reads; when
    set, the emitted ``attrs['transform']`` shifts the origin to the
    window's top-left. The eager path and the dask path (since #1561,
    which threads ``window=`` through ``read_geotiff_dask``) both pass
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
    # The ``allow_rotated=True`` opt-in CRS-drop (#2126) is handled
    # inside ``geo_info_to_metadata``. ``georef_status`` (#2136) rides
    # on the record so the VRT path can stamp it via the same
    # marshalling step. See issue #2139 / ``metadata_to_attrs``.
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
            return v
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
    non-numeric branches in ``_resolve_nodata_attr`` (#1973)."""
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
    :func:`_write_vrt_tiled`, and :func:`write_geotiff_gpu`:

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
    where ``arr`` may have been promoted from an integer dtype to float64
    when the sentinel matched at least one pixel, and
    ``nodata_pixels_present`` is the bool used to populate
    ``attrs['nodata_pixels_present']``. ``None`` means "no scan was
    appropriate for this dtype / sentinel combination."

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
            # ``open_geotiff`` for #1774 / #1564 / #1616).
            if (np.isfinite(mask_sentinel)
                    and float(mask_sentinel).is_integer()):
                nodata_int = int(mask_sentinel)
                info = np.iinfo(arr.dtype)
                if info.min <= nodata_int <= info.max:
                    mask = arr == arr.dtype.type(nodata_int)
                    nodata_pixels_present = bool(mask.any())
                    if nodata_pixels_present:
                        arr = arr.astype(np.float64)
                        arr[mask] = np.nan
                else:
                    nodata_pixels_present = False
            else:
                nodata_pixels_present = False
    else:
        # ``mask_nodata=False``: do not rewrite pixels, but still surface
        # ``attrs['nodata_pixels_present']`` so callers know whether
        # literal sentinel pixels survive in the buffer (issue #2135).
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
):
    """Validate, populate attrs, mask, cast, and build an eager DataArray.

    Wave 1 of #2162 -- ties together the four steps every eager read path
    runs after the bytes land in a host (or cupy) buffer:

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

    Wave migration plan:

    * Wave 2 (#2178 dask, #2179 eager numpy) migrates the eager numpy
      paths. The mask block inside this helper matches the inline block
      in ``open_geotiff`` field-for-field; the migration is a straight
      swap.
    * Wave 3 (#2180 VRT, GPU) migrates the VRT eager + three GPU eager
      sites. The VRT eager path is host-side and works with the helper
      as-is. The GPU sites apply masking through a CUDA kernel
      (``_apply_nodata_mask_gpu_with_presence``); they can either
      pre-mask and call the helper with ``nodata=None`` to skip the
      helper's host-side mask block, or wave 3 can extend this
      helper's signature with a ``mask_fn`` injection. Either choice
      lives in #2180; the wave 1 contract here is the host-side path.

    Returns a :class:`xarray.DataArray` ready for the caller to return
    from the read function. The caller assembles the dask graph
    separately when a lazy backend is in play; this helper is eager-only.

    ``attrs_in`` is shallow-copied via ``dict(attrs_in)``. Nested values
    (e.g. ``extra_tags`` list, ``gdal_metadata`` dict) are shared between
    the caller's seed dict and the returned DataArray's attrs; mutating
    a nested value after the call propagates both ways. Callers that
    care about isolation can ``copy.deepcopy(attrs_in)`` first.
    """
    # Step 1: validate first so partial attrs never leak.
    _validate_read_geo_info(
        geo_info, window=window,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
    )

    # Step 2: populate attrs from geo_info onto a fresh dict (or onto a
    # caller-supplied seed dict, which lets the GPU/VRT migration carry
    # backend-specific keys through without bypassing the helper).
    attrs: dict = dict(attrs_in) if attrs_in else {}
    _populate_attrs_from_geo_info(attrs, geo_info, window=window)

    # Step 3: apply the nodata-to-NaN mask (or compute pixels_present
    # without rewriting if ``mask_nodata=False``). Skipped entirely when
    # the source declared no sentinel.
    nodata_pixels_present: bool | None = None
    if nodata is not None:
        arr, nodata_pixels_present = _apply_eager_nodata_mask(
            arr, mask_sentinel=mask_sentinel, mask_nodata=mask_nodata,
        )

    # Step 4: caller-requested dtype cast (post-mask so the integer
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

    # Step 5: stamp the nodata lifecycle attrs. ``masked`` is True iff
    # the caller opted into masking AND the final buffer dtype is float,
    # mirroring the existing call sites (the integer promotion above
    # only runs when the sentinel matched at least one pixel, so an
    # ``int`` buffer + ``mask_nodata=True`` here means "no pixels were
    # masked" rather than "masking was disabled").
    _set_nodata_attrs(
        attrs, nodata,
        masked=(mask_nodata and np.dtype(str(arr.dtype)).kind == 'f'),
        pixels_present=nodata_pixels_present,
        dtype_cast=dtype_cast_attr,
    )

    # Step 6: build the DataArray. ``_coords_from_geo_info`` honours the
    # windowed-read contract (origin shifted to the window's top-left).
    height, width = arr.shape[:2]
    coords = _coords_from_geo_info(
        geo_info, height, width, window=window,
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
    dtype,
    window,
    allow_rotated: bool = False,
    allow_unparseable_crs: bool = False,
    band_nodata: str | None = None,
    band_nodata_values: list | None = None,
    attrs_in: dict | None = None,
):
    """Validate and populate attrs for dask-style lazy reads.

    Wave 1 of #2162 -- the lazy counterpart of
    :func:`_finalize_eager_read`. The dask + dask-GPU backends cannot
    fold the nodata mask into a single eager step because masking runs
    per-chunk inside the graph; they only need the attrs side of the
    pipeline. This helper:

    1. :func:`_validate_read_geo_info` -- runs first so partial attrs
       never leak on validation failure.
    2. :func:`_populate_attrs_from_geo_info` -- writes the canonical
       attrs onto a fresh dict.
    3. :func:`_set_nodata_attrs` -- ``masked`` is True iff the caller
       opted into masking AND the graph dtype is float. ``dtype_cast``
       is recorded when the caller passed an explicit ``dtype=`` kwarg.
       ``pixels_present=None`` is the documented dask contract from
       issue #2135: a strict per-chunk reduction would force an eager
       ``.compute()`` and break the lazy contract, so the attr is left
       absent on lazy outputs.

    Returns the attrs ``dict`` only; the caller assembles the dask graph
    and builds the :class:`xarray.DataArray` itself, so this helper
    deliberately does not touch arrays or coords.

    The ``dtype`` parameter accepts a numpy dtype, a string ('float64'),
    or ``None``. It is the **resolved graph dtype** the dask backend
    settled on (e.g. ``target_dtype`` after the int->float64 promotion
    that ``mask_nodata=True`` triggers on int files): the helper uses
    it to derive ``masked`` and writes it as ``nodata_dtype_cast`` when
    non-None.

    Wave 2 migration note: the current pre-helper dask backend
    distinguishes "caller explicitly passed ``dtype=``" from
    "graph dtype was auto-promoted by masking" so that
    ``nodata_dtype_cast`` surfaces only on the explicit-cast case.
    This helper conflates the two -- whatever ``dtype`` value the
    caller passes here becomes the ``nodata_dtype_cast`` attr. The
    migration PR (#2178) can either accept that change, or split the
    helper's ``dtype`` into two parameters at that point. Frozen
    signature here per #2177 means we ship the one-``dtype`` shape
    and leave the split for wave 2 if it turns out to matter.

    ``attrs_in`` is shallow-copied via ``dict(attrs_in)``. Nested values
    are shared between the caller's seed dict and the returned attrs;
    callers that care about isolation can ``copy.deepcopy(attrs_in)``
    first.

    ``band_nodata`` and ``band_nodata_values`` forward through to
    :func:`_validate_read_geo_info` so VRT callers can hand the
    mixed-band check the context it needs. Non-VRT callers omit them
    and the mixed-band check short-circuits. See issue #2210.
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
    # ``dtype`` here is the resolved graph dtype; the dask backend
    # promotes int -> float64 before calling this helper when the
    # caller wants masking on an int source.
    if dtype is None:
        masked = False
    else:
        masked = bool(mask_nodata and np.dtype(dtype).kind == 'f')

    # ``dtype_cast`` records the caller-supplied ``dtype=`` kwarg so
    # consumers can tell float-because-masked from float-because-cast.
    # The dask backend resolves ``dtype`` for the graph internally; the
    # helper exposes it via ``attrs['nodata_dtype_cast']`` when set.
    dtype_cast_attr = (
        np.dtype(dtype).name if dtype is not None else None
    )

    _set_nodata_attrs(
        attrs, nodata,
        masked=masked,
        pixels_present=None,
        dtype_cast=dtype_cast_attr,
    )

    return attrs


def _apply_caller_dtype_cast(
    attrs: dict,
    *,
    caller_dtype,
    has_nodata: bool,
) -> None:
    """Stamp ``attrs['nodata_dtype_cast']`` from a caller-supplied ``dtype=``.

    Companion to :func:`_finalize_lazy_read_attrs` for the two dask
    backends (issue #2178). The helper's ``dtype`` argument doubles as
    the resolved graph dtype (driving ``masked_nodata``) and the
    caller-supplied cast attr (driving ``nodata_dtype_cast``); the
    dask paths must keep those separate because ``mask_nodata=True``
    on an integer source auto-promotes the graph dtype to ``float64``
    without the caller asking, and that auto-promotion must not leak
    out as ``nodata_dtype_cast``.

    Call this immediately after :func:`_finalize_lazy_read_attrs` to
    overwrite the attr with the caller's intent:

    * ``caller_dtype is None`` -- the caller did not ask for a cast;
      drop any value the helper wrote.
    * ``caller_dtype is not None`` AND ``has_nodata`` -- the caller
      asked for a cast on a source with a declared sentinel; write
      ``np.dtype(caller_dtype).name``.
    * ``caller_dtype is not None`` AND not ``has_nodata`` -- no
      sentinel was declared, so the attr is meaningless and should
      stay absent (matches the pre-helper contract where
      ``_set_nodata_attrs(..., nodata=None)`` short-circuited).
    """
    if caller_dtype is None:
        attrs.pop('nodata_dtype_cast', None)
    elif has_nodata:
        attrs['nodata_dtype_cast'] = np.dtype(caller_dtype).name
