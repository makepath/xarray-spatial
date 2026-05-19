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
(currently ``2``). Consumers can branch on this integer if the tier
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
- ``nodata``: declared file sentinel as stored in the GDAL_NODATA tag.
  Set whenever the source declares one, as a scalar of the source
  dtype, regardless of whether the in-memory array is float-with-NaN
  or int-with-sentinels.
- ``masked_nodata``: boolean flag paired with ``nodata``. ``True`` iff
  the in-memory array is float dtype and the reader's sentinel-to-NaN
  step ran; ``False`` iff the array still carries the literal integer
  sentinel. Only emitted when ``nodata`` is set; absence is the
  "no declared sentinel" signal. See ``_set_nodata_attrs``.
- ``raster_type``: ``'area'`` (implicit / RasterPixelIsArea) or ``'point'``
  (explicit / RasterPixelIsPoint).
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

import numpy as np

from ._coords import (
    transform_tuple_from_pixel_geometry as _transform_tuple_from_pixel_geometry,
)
from ._geotags import RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT


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
_ATTRS_CONTRACT_VERSION = 2


# String identifiers (used in xrspatial attrs) -> TIFF ResolutionUnit tag ids.
_RESOLUTION_UNIT_IDS = {'none': 1, 'inch': 2, 'centimeter': 3}


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


def _set_nodata_attrs(attrs: dict, nodata, *, array_dtype) -> None:
    """Set ``attrs['nodata']`` and ``attrs['masked_nodata']`` on a read.

    Splits the two meanings previously fused into ``attrs['nodata']``
    (issue #1988):

    * ``attrs['nodata']`` -- declared file sentinel, as a scalar of the
      source dtype. Set whenever the source declared one, regardless of
      whether the array is float-with-NaN or int-with-sentinels.
    * ``attrs['masked_nodata']`` -- boolean flag. ``True`` iff the in-
      memory array has been NaN-masked (i.e. it is float dtype and the
      reader's sentinel-to-NaN step ran). ``False`` iff the array still
      carries the literal integer sentinel value.

    Callers pass ``array_dtype`` as the final post-mask, post-cast dtype
    of the array that will be wrapped in the returned DataArray. The
    float/non-float split drives the ``masked_nodata`` value: any float
    output is treated as NaN-aware (NaN is the sentinel proxy), any
    integer output still carries the raw sentinel.

    ``masked_nodata`` is only emitted when ``nodata is not None``. With
    no declared sentinel, the flag is meaningless and its absence is the
    signal.
    """
    if nodata is None:
        return
    attrs['nodata'] = nodata
    attrs['masked_nodata'] = bool(np.dtype(array_dtype).kind == 'f')


def _validate_read_geo_info(
    geo_info,
    *,
    window=None,
    allow_rotated: bool = False,
    allow_unparseable_crs: bool = False,
) -> None:
    """Run issue #1987 read-side ambiguous-metadata checks against ``geo_info``.

    Centralised helper so the eager numpy, dask, GPU, and VRT read
    paths run the same checks before constructing the returned
    DataArray. Forwards ``allow_rotated`` / ``allow_unparseable_crs``
    to the registered checks (``_check_read_rotated_transform`` and
    ``_check_read_unparseable_crs`` today; sibling checks attach via
    the registry).

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
    })


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
    # Stamp the contract version first so every read path that funnels
    # through this helper carries the marker. The VRT backends build
    # their attrs dict directly and stamp the version there (see
    # ``_backends/vrt.py``); keep both sites in sync via the constant
    # rather than the bare literal.
    attrs['_xrspatial_geotiff_contract'] = _ATTRS_CONTRACT_VERSION

    src_t = geo_info.transform
    has_georef = getattr(geo_info, 'has_georef', True)
    # Rotated reads under ``allow_rotated=True`` drop the CRS attrs so
    # the in-memory pixel grid is not mistaken for a projected raster.
    # The marker is ``geo_info.transform.rotated_affine``, which the
    # geotag parser sets when it sees a rotated ``ModelTransformationTag``
    # under the opt-in (#2115). General no-georef reads (axis-aligned
    # rasters that simply lack transform tags -- e.g. arrays written
    # with ``to_geotiff(..., crs=NNN)`` and no coords) still surface
    # ``crs`` / ``crs_wkt`` because the CRS is meaningful even without
    # an embedded transform; only the rotated case is misleading.
    # See ``open_geotiff`` docstring + issue #2122.
    is_rotated_no_georef = (
        not has_georef
        and src_t is not None
        and getattr(src_t, 'rotated_affine', None) is not None
    )
    if not is_rotated_no_georef:
        if geo_info.crs_epsg is not None:
            attrs['crs'] = geo_info.crs_epsg
        if geo_info.crs_wkt is not None:
            attrs['crs_wkt'] = geo_info.crs_wkt
    if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
        attrs['raster_type'] = 'point'

    # Skip the transform attr for files where no GeoTIFF transform tags
    # (ModelTransformation, ModelPixelScale, or ModelTiepoint) are
    # present, signalled by ``has_georef=False``. The default unit
    # ``GeoTransform`` is a struct placeholder, not real georef --
    # emitting it leaks an identity transform into attrs and confuses
    # downstream code that expects ``'transform' in attrs`` to mean
    # "this raster has a georef transform" (#1710).
    if src_t is not None and has_georef:
        attrs['transform'] = _transform_tuple_from_pixel_geometry(
            src_t.origin_x, src_t.origin_y,
            src_t.pixel_width, src_t.pixel_height,
            window=window,
        )

    # Contract v2 (issue #2016) removed the 13 secondary GeoKey-derived
    # attrs that v1 emitted under a ``DeprecationWarning`` (``crs_name``,
    # ``geog_citation``, ``datum_code``, ``angular_units``,
    # ``semi_major_axis``, ``inv_flattening``, ``linear_units``,
    # ``projection_code``, ``vertical_crs``, ``vertical_citation``,
    # ``vertical_units``). The underlying ``GeoInfo`` fields are still
    # populated by the GeoKey parser because ``_synthesize_user_defined_wkt``
    # consumes ``geog_citation`` (and siblings) to fill ``crs_wkt`` for
    # user-defined CRSes; the reader no longer surfaces them as
    # separate user-visible attrs.

    if geo_info.gdal_metadata is not None:
        attrs['gdal_metadata'] = geo_info.gdal_metadata
    if geo_info.gdal_metadata_xml is not None:
        attrs['gdal_metadata_xml'] = geo_info.gdal_metadata_xml

    if geo_info.extra_tags is not None:
        attrs['extra_tags'] = geo_info.extra_tags
    if geo_info.image_description is not None:
        attrs['image_description'] = geo_info.image_description
    if geo_info.extra_samples is not None:
        attrs['extra_samples'] = geo_info.extra_samples

    if geo_info.x_resolution is not None:
        attrs['x_resolution'] = geo_info.x_resolution
    if geo_info.y_resolution is not None:
        attrs['y_resolution'] = geo_info.y_resolution
    if geo_info.resolution_unit is not None:
        _unit_names = {1: 'none', 2: 'inch', 3: 'centimeter'}
        attrs['resolution_unit'] = _unit_names.get(
            geo_info.resolution_unit, str(geo_info.resolution_unit))

    # Contract v2 (issue #2016) removed ``attrs['cmap']`` and
    # ``attrs['colormap_rgba']``. The canonical ``attrs['colormap']``
    # (raw uint16 RGB triples from TIFF tag 320) is still emitted below
    # via the ``extra_tags`` scan; callers that need an RGBA palette or
    # a :class:`matplotlib.colors.ListedColormap` should build one from
    # ``attrs['colormap']`` directly.

    if geo_info.extra_tags is not None:
        for _tag_id, _tt, _tc, _tv in geo_info.extra_tags:
            if _tag_id == 320:  # TAG_COLORMAP
                attrs['colormap'] = _tv
                break


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
