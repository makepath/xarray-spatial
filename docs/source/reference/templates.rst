..  _reference.templates:

*********
Templates
*********

Empty study-area grids you can start an analysis from. ``from_template`` turns
a region name, a world-city name, or a country code into a NaN-filled
:class:`xarray.DataArray` that follows the xarray-spatial array contract, so it
feeds straight into the rest of the library. Cities (national capitals, major
regional metros, and recognizable US secondary cities) come back as a metro
bounding box in their UTM zone. Curated regions span North America, Europe, and
now Southeast Asia, Central America, the Caribbean, and West Africa, each in an
EPSG-coded continental equal-area projection. Whole-world canvases are available
in a few projections too: ``'web_mercator'`` (EPSG:3857), ``'wgs84'`` /
``'latlon'`` (EPSG:4326), and ``'equal_earth'`` (EPSG:8857).

Call :func:`~xrspatial.templates.list_templates` to discover every name
``from_template`` accepts (curated regions, world cities, and country codes).

From Template
=============
.. autosummary::
    :toctree: _autosummary

    xrspatial.templates.from_template
    xrspatial.templates.list_templates

Putting your data on a template
===============================

A template is an empty canvas; ``coregister`` fills it with your own data. It
reprojects a raster :class:`~xarray.DataArray` onto the template's grid, or
rasterizes a ``GeoDataFrame`` onto it, so every layer lines up cell-for-cell.

.. sourcecode:: python

    from xrspatial import from_template

    grid = from_template("conus", resolution=1000)   # empty Albers grid
    elevation = grid.xrs.coregister(my_dem)           # raster -> grid
    roads = grid.xrs.coregister(my_roads_gdf)         # vectors -> grid
    slope = elevation.xrs.slope()

For an out-of-core run, ask for a dask backend. The template tiles into even,
square blocks tuned for the neighborhood ops that follow, so the downstream
task graph stays parallel and overlap-friendly:

.. sourcecode:: python

    grid = from_template("conus", resolution=250, backend="dask")
    slope = grid.xrs.coregister(my_dem).xrs.slope()   # stays lazy
