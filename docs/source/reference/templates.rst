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
