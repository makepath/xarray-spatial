..  _user_guide.attrs_contract:

***********************
GeoTIFF attrs contract
***********************

When :func:`xrspatial.geotiff.open_geotiff` returns a ``DataArray``, the
``attrs`` mapping carries metadata recovered from the file's GeoTIFF
tags and GeoKeys. xrspatial classifies those keys into three tiers.
Each tier offers a different round-trip guarantee when the array is
written back out with :func:`xrspatial.geotiff.to_geotiff`. Canonical
keys are owned by xrspatial and survive a round-trip byte-for-byte.
Compatibility aliases are recognised on read for interoperability with
rioxarray and CF-style pipelines but are never re-emitted on write.
Pass-through keys are surfaced verbatim from the file's GeoKey
directory; whether they survive a round-trip depends on what the
writer can reconstruct from the canonical CRS.

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
   * - ``crs_name``
     - str
     - Human-readable CRS citation from ``GTCitationGeoKey``.
   * - ``geog_citation``
     - str
     - Geographic CRS citation from ``GeogCitationGeoKey``.
   * - ``datum_code``
     - int
     - Geodetic datum EPSG code from ``GeogGeodeticDatumGeoKey``.
   * - ``angular_units``
     - int
     - Angular units code from ``GeogAngularUnitsGeoKey``.
   * - ``linear_units``
     - int
     - Linear units code from ``ProjLinearUnitsGeoKey``.
   * - ``semi_major_axis``
     - float
     - Ellipsoid semi-major axis in metres from
       ``GeogSemiMajorAxisGeoKey``.
   * - ``inv_flattening``
     - float
     - Ellipsoid inverse flattening from
       ``GeogInvFlatteningGeoKey``.
   * - ``projection_code``
     - int
     - Projected CRS code from ``ProjectedCSTypeGeoKey``.
   * - ``vertical_crs``
     - int
     - Vertical CRS EPSG code from ``VerticalCSTypeGeoKey``.
   * - ``vertical_citation``
     - str
     - Vertical CRS citation from ``VerticalCitationGeoKey``.
   * - ``vertical_units``
     - int
     - Vertical units code from ``VerticalUnitsGeoKey``.
   * - ``image_description``
     - str
     - ``ImageDescription`` TIFF tag (tag id 270).
   * - ``extra_samples``
     - tuple
     - ``ExtraSamples`` TIFF tag (tag id 338) describing alpha or
       other auxiliary channels.
   * - ``colormap``
     - tuple
     - Raw ``ColorMap`` TIFF tag (tag id 320) values.
   * - ``colormap_rgba``
     - array
     - Decoded RGBA colormap. Emitted only when the file's photometric
       interpretation is ``Photometric == 3`` (palette).
   * - ``cmap``
     - ``matplotlib.colors.ListedColormap``
     - Matplotlib colormap built from ``colormap_rgba``. Present only
       when matplotlib is importable and the same ``Photometric == 3``
       gate is satisfied.


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

The writer reconstructs as many pass-through keys as it can from
``crs`` or ``crs_wkt``. Keys it cannot reconstruct are dropped
silently rather than failing the write. Callers must not assume any
specific pass-through key survives a round-trip; a key that was
present on the original file may be absent after write→read if the
canonical CRS does not carry enough information to rebuild it.


Deprecated GeoKey attrs (issue #1984)
=====================================

The following attrs are still populated on read for one release
cycle but each emission fires a ``DeprecationWarning``. The writer's
``build_geo_tags`` only emits the primary CRS GeoKey and citation for
each axis (geographic, projected, vertical), so the secondary GeoKeys
these attrs derive from are never written and the values do not
survive a write→read round-trip. Migrate to ``crs`` / ``crs_wkt`` and
derive any needed value with :mod:`pyproj`.

Geographic-CRS GeoKey attrs: ``crs_name``, ``geog_citation``,
``datum_code``, ``angular_units``, ``semi_major_axis``,
``inv_flattening``.

Projected-CRS GeoKey attrs: ``linear_units``, ``projection_code``.

Vertical-CRS GeoKey attrs: ``vertical_crs``, ``vertical_citation``,
``vertical_units``.

Migration recipe::

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
