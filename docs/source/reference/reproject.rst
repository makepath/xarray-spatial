..  _reference.reproject:

*********
Reproject
*********

Reproject a raster to a new coordinate reference system, and merge
multiple rasters into a unified output grid.

Reproject
=========
.. autosummary::
    :toctree: _autosummary

    xrspatial.reproject.reproject
    xrspatial.reproject.merge

Bounds policy
=============

The ``bounds_policy`` parameter on :func:`xrspatial.reproject.reproject`
and :func:`xrspatial.reproject.merge` controls how the output extent is
derived from the input extent when no explicit ``bounds`` is supplied.
The available options are:

- ``"auto"`` (default): clamp geographic source bounds inward by
  0.01 deg from +/-180 longitude and +/-90 latitude, and fall back to
  the 2nd/98th percentile of a dense interior grid when the projected
  extent is more than 50x the source extent. Matches the historical
  behaviour. Emits a ``UserWarning`` when the clamp or percentile
  fallback actually alters the bounds.
- ``"raw"``: use the true projected extent of the source corners and
  edges. No clamp, no percentile. Pass this when downstream tooling
  needs the exact projection of the input footprint and you are willing
  to handle large outputs near projection singularities.
- ``"clamp"``: apply only the geographic-bounds clamp.
- ``"percentile"``: always use the 2/98 percentile bounds, even when
  the projected extent looks well-behaved.
