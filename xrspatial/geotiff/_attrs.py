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
(currently ``1``). Consumers can branch on this integer if the tier
split changes in a future release.

Canonical (xrspatial owns these; round-trip stable):

- ``crs``: EPSG integer code for the horizontal CRS.
- ``crs_wkt``: WKT string for the horizontal CRS. Present on read whenever
  any CRS information is available.
- ``transform``: rasterio-style 6-tuple
  ``(pixel_width, 0.0, origin_x, 0.0, pixel_height, origin_y)``. Omitted
  for files with no GeoTIFF transform tags (ModelTransformation,
  ModelPixelScale, or ModelTiepoint).
- ``nodata``: declared file sentinel as stored in the GDAL_NODATA tag. The
  declared-vs-masked split is tracked in issue #1988; the canonical
  semantics here describe the intended behaviour once #1988 lands.
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

- ``crs_name``: human-readable CRS name from the GeoKey directory.
- ``geog_citation``: GeographicTypeGeoKey citation string.
- ``datum_code``: GeogGeodeticDatumGeoKey value.
- ``angular_units``: GeogAngularUnitsGeoKey value.
- ``semi_major_axis``: GeogSemiMajorAxisGeoKey value.
- ``inv_flattening``: GeogInvFlatteningGeoKey value.
- ``vertical_crs``: VerticalCSTypeGeoKey value.
- ``vertical_citation``: VerticalCitationGeoKey value.
- ``vertical_units``: VerticalUnitsGeoKey value.
- ``image_description``: TIFF ImageDescription tag.
- ``extra_samples``: TIFF ExtraSamples tag.
- ``colormap``, ``colormap_rgba``, ``cmap``: palette data attached to
  single-band paletted images.

Deprecated (will be removed in a future release; see issue #1984):

- ``linear_units``: ProjLinearUnitsGeoKey value. The writer's
  ``build_geo_tags`` only emits the primary ``GEOKEY_PROJECTED_CS_TYPE``
  and never the secondary projected GeoKeys, so this attr cannot be
  reconstructed on round-trip. Read-side emission triggers a
  ``DeprecationWarning`` for one release cycle before removal.
- ``projection_code``: ProjectionGeoKey value. Same root cause as
  ``linear_units``: the writer never emits the underlying GeoKey, so
  the value cannot survive a round-trip. Read-side emission triggers a
  ``DeprecationWarning`` for one release cycle before removal.
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
_ATTRS_CONTRACT_VERSION = 1


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

    if geo_info.crs_epsg is not None:
        attrs['crs'] = geo_info.crs_epsg
    if geo_info.crs_wkt is not None:
        attrs['crs_wkt'] = geo_info.crs_wkt
    if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
        attrs['raster_type'] = 'point'

    src_t = geo_info.transform
    # Skip the transform attr for files where no GeoTIFF transform tags
    # (ModelTransformation, ModelPixelScale, or ModelTiepoint) are
    # present, signalled by ``has_georef=False``. GeoKeys / CRS metadata
    # can still be present in that case. The default unit
    # ``GeoTransform`` is a struct placeholder, not real georef --
    # emitting it leaks an identity transform into attrs and confuses
    # downstream code that expects ``'transform' in attrs`` to mean
    # "this raster has a georef transform" (#1710).
    has_georef = getattr(geo_info, 'has_georef', True)
    if src_t is not None and has_georef:
        attrs['transform'] = _transform_tuple_from_pixel_geometry(
            src_t.origin_x, src_t.origin_y,
            src_t.pixel_width, src_t.pixel_height,
            window=window,
        )

    if geo_info.crs_name is not None:
        attrs['crs_name'] = geo_info.crs_name
    if geo_info.geog_citation is not None:
        attrs['geog_citation'] = geo_info.geog_citation
    if geo_info.datum_code is not None:
        attrs['datum_code'] = geo_info.datum_code
    if geo_info.angular_units is not None:
        attrs['angular_units'] = geo_info.angular_units
    if geo_info.linear_units is not None:
        warnings.warn(
            "xrspatial.geotiff: attrs['linear_units'] is deprecated; "
            "the writer cannot reconstruct it from the canonical CRS "
            "so it will not round-trip. It will be removed in a future "
            "release. See issue #1984.",
            DeprecationWarning,
            stacklevel=2,
        )
        attrs['linear_units'] = geo_info.linear_units
    if geo_info.semi_major_axis is not None:
        attrs['semi_major_axis'] = geo_info.semi_major_axis
    if geo_info.inv_flattening is not None:
        attrs['inv_flattening'] = geo_info.inv_flattening
    if geo_info.projection_code is not None:
        warnings.warn(
            "xrspatial.geotiff: attrs['projection_code'] is deprecated; "
            "the writer cannot reconstruct it from the canonical CRS "
            "so it will not round-trip. It will be removed in a future "
            "release. See issue #1984.",
            DeprecationWarning,
            stacklevel=2,
        )
        attrs['projection_code'] = geo_info.projection_code
    if geo_info.vertical_epsg is not None:
        attrs['vertical_crs'] = geo_info.vertical_epsg
    if geo_info.vertical_citation is not None:
        attrs['vertical_citation'] = geo_info.vertical_citation
    if geo_info.vertical_units is not None:
        attrs['vertical_units'] = geo_info.vertical_units

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

    if geo_info.colormap is not None:
        try:
            from matplotlib.colors import ListedColormap
            attrs['cmap'] = ListedColormap(
                geo_info.colormap, name='tiff_palette')
            attrs['colormap_rgba'] = geo_info.colormap
        except ImportError:
            attrs['colormap_rgba'] = geo_info.colormap

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
