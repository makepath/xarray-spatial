..  _user_guide.attrs_contract:

***********************
GeoTIFF attrs contract
***********************

When :func:`xrspatial.geotiff.open_geotiff` returns a ``DataArray``, the
``attrs`` mapping carries metadata recovered from the file's GeoTIFF
tags and GeoKeys. xrspatial classifies those keys into three tiers,
each with a different round-trip guarantee when the array is written
back out with :func:`xrspatial.geotiff.to_geotiff`. Canonical keys are
owned by xrspatial and survive a round-trip byte-for-byte.
Compatibility aliases are recognised on read for interoperability with
rioxarray and CF-style pipelines but are never re-emitted on write.
Pass-through keys are folded into ``extra_tags`` by the writer and
rebuilt from the TIFF tag on the next read.

Contract v2 (issue #2016) removed the 13 secondary GeoKey-derived and
matplotlib-colormap attrs that the v1 reader emitted under a
``DeprecationWarning``. See `Removed in contract v2`_ below for the
migration recipe.

.. contents:: On this page
   :local:
   :depth: 1


Canonical keys
==============

xrspatial owns these keys. Every read path emits them when the source
file carries the corresponding information, and every writer consumes
them when serialising back to TIFF. A write followed by a read produces
a byte-equivalent value for every canonical key that was set before the
write.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Type
     - Definition
   * - ``crs``
     - int
     - EPSG code of the horizontal CRS, when one can be resolved.
   * - ``crs_wkt``
     - str
     - WKT string of the horizontal CRS. Present on read when any CRS
       information is available, and treated as the canonical CRS
       representation when both ``crs`` and ``crs_wkt`` are set. The
       dialect depends on the source: paths that synthesise a WKT from
       an EPSG code via pyproj emit WKT2; paths that read a WKT
       verbatim from the file (e.g. a VRT ``SRS`` tag) carry whatever
       dialect was stored.
   * - ``transform``
     - tuple
     - ``(pixel_width, 0.0, origin_x, 0.0, pixel_height, origin_y)``
       affine transform tuple matching the rasterio ``Affine`` ordering.
       Omitted for files with no ``ModelTransformation`` /
       ``ModelPixelScale`` / ``ModelTiepoint`` tags.
   * - ``nodata``
     - scalar
     - Numeric NoData sentinel. Emitted by readers when the file
       carries a ``GDAL_NODATA`` tag, and consumed by writers as the
       primary nodata source. The read-side precedence chain is
       ``nodata``, then ``nodatavals``, then ``_FillValue``; see
       ``_resolve_nodata_attr``.
   * - ``masked_nodata``
     - bool
     - Paired with ``nodata``. ``True`` when the in-memory array is
       float dtype and the reader's sentinel-to-NaN step ran;
       ``False`` when the array still carries the literal integer
       sentinel. Only set when ``nodata`` is set; absence means no
       declared sentinel.
   * - ``raster_type``
     - str
     - ``'point'`` when the file declares ``RasterPixelIsPoint``;
       absent otherwise (treated as ``'area'``).
   * - ``extra_tags``
     - list of tuples
     - Raw TIFF tag entries as
       ``(tag_id, type_id, count, value)`` tuples for tags not
       otherwise covered by the canonical set.
   * - ``gdal_metadata``
     - dict
     - Decoded contents of the ``GDAL_METADATA`` XML tag.
   * - ``gdal_metadata_xml``
     - str
     - Verbatim XML string of the ``GDAL_METADATA`` tag. Preferred
       over ``gdal_metadata`` by writers when both are present.
   * - ``x_resolution``
     - float
     - ``XResolution`` TIFF tag value.
   * - ``y_resolution``
     - float
     - ``YResolution`` TIFF tag value.
   * - ``resolution_unit``
     - str
     - ``'none'``, ``'inch'``, or ``'centimeter'`` (mapped from
       ``ResolutionUnit`` ids 1, 2, 3).
   * - ``_xrspatial_geotiff_contract``
     - int
     - Contract version. Currently ``2``. See `Versioning`_.


Compatibility aliases
=====================

Aliases are recognised on read so attrs produced by other libraries
keep working with xrspatial writers. The writer never emits an alias
when the canonical key is available. After a round-trip through
:func:`xrspatial.geotiff.to_geotiff`, callers should expect the
canonical key only.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Type
     - Definition
   * - ``nodatavals``
     - tuple
     - rioxarray's per-band NoData tuple. Read paths fall back to the
       first numeric, non-NaN entry when ``nodata`` is absent.
   * - ``_FillValue``
     - scalar
     - CF convention fill value. Read paths fall back to it when
       neither ``nodata`` nor ``nodatavals`` carries a usable
       sentinel.


Pass-through keys
=================

These keys are populated on read from the file's GeoKey directory.
The writer attempts to reconstruct each one from ``crs`` or
``crs_wkt``; keys it cannot reconstruct are dropped silently. Callers
must not assume a specific pass-through key survives a round-trip.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Type
     - Definition
   * - ``image_description``
     - str
     - ``ImageDescription`` TIFF tag (tag id 270). The writer folds it
       into ``extra_tags``, so the reader rebuilds the attr from tag
       270 on the next read.
   * - ``extra_samples``
     - tuple
     - ``ExtraSamples`` TIFF tag (tag id 338) describing alpha or
       other auxiliary channels. Same round-trip path as
       ``image_description``.
   * - ``colormap``
     - tuple
     - Raw ``ColorMap`` TIFF tag (tag id 320) values. Round-trips via
       ``_merge_friendly_extra_tags``.

The GeoKey-derived attrs that used to live in this tier
(``crs_name``, ``geog_citation``, ``datum_code``, ``angular_units``,
``linear_units``, ``semi_major_axis``, ``inv_flattening``,
``projection_code``, ``vertical_crs``, ``vertical_citation``,
``vertical_units``) and the matplotlib colormap variants (``cmap``,
``colormap_rgba``) were removed by contract v2 (issue #2016). See
`Removed in contract v2`_ below for the migration recipe.


Removed in contract v2
======================

The following keys were emitted by older xrspatial releases under a
``DeprecationWarning`` and have been removed from the reader as of
contract version ``2`` (issue #2016). Reads no longer surface them on
``DataArray.attrs``; downstream code that accessed them via
``attrs[key]`` will see ``KeyError`` rather than the deprecated value.
Switch to ``attrs.get(key)`` or derive the value from ``crs`` /
``crs_wkt`` with :mod:`pyproj`.

GeoKey-derived attrs
--------------------

Secondary GeoKey directory entries that the reader extracted on the
way in but the writer never emitted on the way out:
``xrspatial.geotiff._geotags.build_geo_tags`` writes only the primary
``GEOKEY_GEOGRAPHIC_TYPE`` / ``GEOKEY_PROJECTED_CS_TYPE`` /
``GEOKEY_VERTICAL_CS_TYPE`` plus the citation for each axis, never the
secondary keys these attrs derived from. A write -> read cycle dropped
them silently in v1; v2 drops them on read too.

* Geographic-CRS GeoKey attrs: ``crs_name``, ``geog_citation``,
  ``datum_code``, ``angular_units``, ``semi_major_axis``,
  ``inv_flattening``.
* Projected-CRS GeoKey attrs: ``linear_units``, ``projection_code``.
* Vertical-CRS GeoKey attrs: ``vertical_crs``, ``vertical_citation``,
  ``vertical_units``.

Canonical replacement: ``crs`` / ``crs_wkt`` plus a one-liner with
:mod:`pyproj` when a derived value is needed::

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

Matplotlib colormap variants
----------------------------

Different root cause: the writer cannot set ``Photometric == 3``, so
the matplotlib-derived attrs never survived a write -> read cycle in
v1. v2 removes the read-side emission too. The plain
``attrs['colormap']`` (raw uint16 RGB triples from TIFF tag 320)
stays in the `Pass-through keys`_ tier and is the canonical
replacement.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Removed key
     - Type
     - Migration
   * - ``colormap_rgba``
     - array
     - Decoded RGBA colormap. Reshape ``attrs['colormap']`` to
       ``(n_colors, 3)`` and append an alpha channel in caller code.
   * - ``cmap``
     - ``matplotlib.colors.ListedColormap``
     - Matplotlib colormap built from the palette. Construct a
       :class:`matplotlib.colors.ListedColormap` from
       ``attrs['colormap']`` in caller code; the
       ``_listed_colormap_from_attrs`` helper in
       :mod:`xrspatial.accessor` is one possible reference
       implementation.


Round-trip invariants
=====================

Canonical tier
--------------

For every canonical key set in ``attrs`` before
:func:`xrspatial.geotiff.to_geotiff` runs, reopening the written file
returns a byte-equivalent value at the same key. The writer is
responsible for serialising each canonical key into the right TIFF
tag, and the reader is responsible for restoring it.

Compatibility alias tier
------------------------

A read path accepts an alias when no canonical key is present. A
write path never emits an alias. Callers that pass an alias to the
writer (because the array came from rioxarray or a CF pipeline) get
the canonical key back after the round-trip. The alias is gone from
the reopened array.

Pass-through tier
-----------------

The pass-through tier now contains only ``image_description``,
``extra_samples``, and ``colormap``. The writer folds each into
``extra_tags`` via ``_merge_friendly_extra_tags`` and the reader
rebuilds the attr from the TIFF tag on the next read, so all three
round-trip. The GeoKey-derived attrs that used to live here were
removed by contract v2 (see `Removed in contract v2`_).


Versioning
==========

The contract is versioned through ``attrs['_xrspatial_geotiff_contract']``.
The current value is ``2``. Future revisions that add canonical keys,
move keys between tiers, or change a key's semantics will bump the
integer. Callers that depend on a specific layout can branch on the
version, and writers will emit the version they were built against.

A read path that encounters an attrs dict with a higher version than
the running xrspatial release should still produce a usable
``DataArray``, but pass-through keys introduced in the newer contract
may surface as ordinary attrs without library-level support.

Contract v2 (issue #2016) removed the 13 deprecated GeoKey-derived
and matplotlib-colormap attrs that v1 emitted on read under a
``DeprecationWarning``. Downstream code that accessed those keys via
``attrs[key]`` will now see ``KeyError``; switch to ``attrs.get(key)``
or migrate to the canonical ``crs`` / ``crs_wkt`` plus :mod:`pyproj`
recipe documented in `Removed in contract v2`_.
