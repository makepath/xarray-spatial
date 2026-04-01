..  _user_guide.caveats:

***********************
Caveats & Assumptions
***********************

xarray-spatial makes several assumptions about input data that can produce
wrong or confusing results if they aren't met.  This page collects them in
one place.  Each section maps to a Sphinx admonition level so you can gauge
severity at a glance:

.. danger:: marks assumptions that cause **silently wrong output**.

.. warning:: marks gotchas that **change the meaning** of results.

.. caution:: marks **performance or memory** pitfalls.

.. tip:: gives practical workarounds.

.. note:: provides context that clarifies non-obvious behaviour.


.. contents:: On this page
   :local:
   :depth: 1


Geodesic terrain functions assume WGS84
========================================

.. danger::

   ``slope()``, ``aspect()``, ``curvature()``, and ``hillshade()`` with
   ``method='geodesic'`` use the WGS84 ellipsoid (a = 6 378 137 m,
   b = 6 356 752 m).  There is no parameter to select a different
   ellipsoid.

If your elevation data references a different ellipsoid (e.g. Clarke 1866 or
Bessel 1841), the geodesic path will still run, but the ECEF projection and
curvature correction will be slightly off.  For most modern data this
doesn't matter because WGS84 and GRS80 are functionally identical, but
older survey-quality datasets on local datums can differ by tens of metres
in the geoid.


Projected coordinates + geodesic = wrong results
=================================================

.. danger::

   Passing a raster whose coordinates are in metres (UTM, state-plane, etc.)
   to ``method='geodesic'`` will produce garbage.  The geodesic path expects
   latitude/longitude in degrees.

The library does validate that coordinate values fall within geographic
ranges (latitude in [-90, 90], longitude in [-180, 360]) and will raise a
``ValueError`` if they don't.  But this check is not foolproof: coordinates
in small projected grids that happen to look like valid lat/lon values will
slip through.

.. tip::

   Use ``method='planar'`` (the default) for projected CRS data.  Use
   ``method='geodesic'`` only when your coordinates are in degrees on a
   geographic CRS.


All output is float32
=====================

.. warning::

   Every analytical function casts its output to ``float32`` regardless of
   input dtype.  If you pass ``float64`` elevation data, the result is
   ``float32``.

Float32 gives about 7 significant digits, which is more than enough for
slope, NDVI, and similar metrics.  But if you're chaining many operations
or working with data that needs sub-centimetre precision (e.g. LiDAR
heights in a small range), the truncation can matter.

See :ref:`user_guide.data_types` for the full rationale.

.. tip::

   If you need float64 output, cast after the function returns:

   .. code-block:: python

      result = slope(agg).astype('float64')


NaN is the nodata sentinel
==========================

.. warning::

   xarray-spatial uses ``NaN`` as its nodata value everywhere.  There is
   no parameter to choose a different sentinel (e.g. -9999).

Functions generally **propagate** NaN: if any cell in a 3x3 kernel is NaN,
the output cell is NaN.  The exact behaviour varies by function:

- **Slope, aspect, hillshade** -- any NaN neighbour produces a NaN output
  cell.
- **Focal mean/std** -- NaN neighbours are excluded from the average; the
  cell itself is still computed.
- **Zonal stats** -- NaN values are excluded from the aggregation.  An
  all-NaN zone returns NaN, not zero.
- **Hydrology (flow direction)** -- NaN cells are treated as impassable
  barriers.  Flow cannot cross them.
- **Pathfinding (A*)** -- NaN cells are impassable, same as barriers.

.. note::

   Edge cells (within the kernel radius of the raster boundary) default to
   NaN when ``boundary='nan'``.  Use ``boundary='nearest'`` or
   ``boundary='reflect'`` if you need values at the edges.


Proximity distances depend on coordinate units
================================================

.. warning::

   ``proximity()`` with ``distance_metric='EUCLIDEAN'`` (the default)
   computes distances in whatever units the DataArray's coordinates use.
   If coordinates are default integer indices (0, 1, 2, ...), the result
   is in pixel counts.  If coordinates are in metres (UTM), the result is
   in metres.  If coordinates are in degrees, the result is in degrees.

Switch to ``distance_metric='GREAT_CIRCLE'`` for lat/lon data to get
distances in **metres** (the radius used is 6 378 137 m).

.. tip::

   .. code-block:: python

      from xrspatial.proximity import proximity

      # EUCLIDEAN with default coords = pixel-count distances
      dist_px = proximity(raster)

      # GREAT_CIRCLE with lat/lon coords = distances in metres
      dist_m = proximity(raster, distance_metric='GREAT_CIRCLE')


Great-circle distances assume sphere radius = semi-major axis
==============================================================

.. note::

   Great-circle distances in ``proximity()`` and ``surface_distance()``
   model the Earth as a sphere with radius = WGS84 semi-major axis
   (6 378 137 m), not the mean radius (~6 371 km).

The difference is about 0.1 %.  This is negligible for most use cases but
worth knowing if you're comparing against a reference that uses the mean
radius or the IUGG value.


Dask chunk sizes and ``map_overlap``
=====================================

.. caution::

   Focal, slope, aspect, and other kernel-based operations use
   ``dask.array.map_overlap``.  If any chunk dimension is **smaller than
   the kernel depth** (typically 1 cell for a 3x3 kernel), the overlap
   will fail or produce wrong results.

.. caution::

   ``proximity()`` with Dask expands each chunk by ``max_distance`` cells.
   If ``max_distance`` is infinite (the default), the entire array is loaded
   into one chunk.  Set a finite ``max_distance`` to keep memory bounded.

.. tip::

   A good rule of thumb for focal operations: keep chunks at least 64 cells
   on each side.  For proximity, set ``max_distance`` to the largest
   distance you actually care about.

   .. code-block:: python

      import dask.array as da
      import xarray as xr

      # Rechunk to safe sizes before running slope
      agg = xr.DataArray(
          da.from_array(big_array, chunks=(512, 512))
      )


GPU memory is not managed automatically
========================================

.. caution::

   CuPy-backed operations allocate the full output array on the GPU.
   There is no automatic tiling or fallback to CPU if the array exceeds
   VRAM.

If you hit ``cuda.cudadrv.driver.CudaAPIError`` or ``cupy.cuda.memory
.OutOfMemoryError``, the array is too large for your GPU.  Use Dask + CuPy
to process in chunks, or fall back to NumPy.

.. tip::

   Complex geodesic kernels (slope, aspect with ``method='geodesic'``)
   use many float64 registers, which limits thread-block occupancy.
   xarray-spatial already reduces thread blocks to 16x16 for these
   kernels, but very old GPUs (compute capability < 6.0) may still
   hit register spill.


Dimension ordering
==================

.. note::

   xarray-spatial assumes the last two dimensions of a DataArray are
   ``(y, x)`` in that order.  It does not inspect ``attrs['crs']`` or
   CF conventions to verify this.

If your data has ``(x, y)`` ordering (rare, but it happens with some
netCDF conventions), transpose before passing to any function:

.. code-block:: python

   agg = agg.transpose('y', 'x')


Classification and NaN/Inf
===========================

.. warning::

   ``equal_interval()``, ``quantile()``, ``natural_breaks()``, and other
   classification functions set NaN and infinite input values to NaN in the
   output.  They do not raise an error.

This means a raster with a few stray ``inf`` values (e.g. from a division
by zero upstream) will silently produce NaN bins for those cells.

.. tip::

   Clean infinities before classifying:

   .. code-block:: python

      import numpy as np
      agg = agg.where(np.isfinite(agg))
