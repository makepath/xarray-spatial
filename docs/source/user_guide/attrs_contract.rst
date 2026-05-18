..  _user_guide.attrs_contract:

***********************
GeoTIFF attrs contract
***********************

When :func:`xrspatial.geotiff.open_geotiff` returns a ``DataArray``, the
``attrs`` mapping carries metadata recovered from the file's GeoTIFF
tags and GeoKeys. xrspatial classifies those keys into four tiers,
each with a different round-trip guarantee when the array is written
back out with :func:`xrspatial.geotiff.to_geotiff`. Canonical keys are
owned by xrspatial and survive a round-trip byte-for-byte.
Compatibility aliases are recognised on read for interoperability with
rioxarray and CF-style pipelines but are never re-emitted on write.
Pass-through keys are folded into ``extra_tags`` by the writer and
rebuilt from the TIFF tag on the next read. Deprecated keys are
emitted on read for one release cycle with a ``DeprecationWarning``;
they do not round-trip and will be removed.

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
     - Contract version. Currently ``1``. See `Versioning`_.


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
``colormap_rgba``) all moved to the `Deprecated keys`_ section below.
They are still emitted on read for one release cycle but fire a
``DeprecationWarning`` and will be removed.


Deprecated keys
===============

These keys are still emitted on read for one release cycle, but each
emission triggers a ``DeprecationWarning``. They do not round-trip
through ``open_geotiff`` -> ``to_geotiff`` -> ``open_geotiff`` and
will be removed at the end of the deprecation window. Callers should
migrate to the canonical alternative listed below. See issue #1984.

GeoKey-derived attrs
--------------------

Secondary GeoKey directory entries that the reader extracts on the
way in but the writer never emits on the way out:
``xrspatial.geotiff._geotags.build_geo_tags`` writes only the primary
``GEOKEY_GEOGRAPHIC_TYPE`` / ``GEOKEY_PROJECTED_CS_TYPE`` /
``GEOKEY_VERTICAL_CS_TYPE`` plus the citation for each axis, never the
secondary keys these attrs derive from. So a write -> read cycle
drops them silently.

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
the matplotlib-derived attrs do not survive a write -> read cycle.
The plain ``attrs['colormap']`` (raw uint16 RGB triples from TIFF
tag 320) stays in the `Pass-through keys`_ tier and is the canonical
replacement.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Key
     - Type
     - Definition and migration
   * - ``colormap_rgba``
     - array
     - Decoded RGBA colormap. Emitted on read when the file's
       photometric interpretation is ``Photometric == 3`` (palette).
       The writer cannot set ``Photometric == 3`` so the attr does
       not round-trip. Reshape ``attrs['colormap']`` to
       ``(n_colors, 3)`` and append an alpha channel in caller code
       if needed.
   * - ``cmap``
     - ``matplotlib.colors.ListedColormap``
     - Matplotlib colormap built from the palette. Same
       ``Photometric == 3`` gate, same round-trip gap. Construct a
       ``ListedColormap`` from ``attrs['colormap']`` in caller code
       if needed.


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
round-trip. The GeoKey-derived attrs that used to live here moved
to the `Deprecated keys`_ tier (see below).

Deprecated tier
---------------

Deprecated keys are still populated on read for one release cycle so
existing consumers keep working, but each emission fires a
``DeprecationWarning``. The write path treats them as advisory only:
none survive a write -> read cycle. They will be removed at the end
of the deprecation window; at that point the contract version stamp
bumps to ``2``.


Versioning
==========

The contract is versioned through ``attrs['_xrspatial_geotiff_contract']``.
The current value is ``1``. Future revisions that add canonical keys,
move keys between tiers, or change a key's semantics will bump the
integer. Callers that depend on a specific layout can branch on the
version, and writers will emit the version they were built against.

A read path that encounters an attrs dict with a higher version than
the running xrspatial release should still produce a usable
``DataArray``, but pass-through keys introduced in the newer contract
may surface as ordinary attrs without library-level support.
