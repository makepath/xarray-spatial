..  _reference.templates:

*********
Templates
*********

Empty study-area grids you can start an analysis from. ``from_template`` turns
a region name, a world-city name, or a country code into a NaN-filled
:class:`xarray.DataArray` that follows the xarray-spatial array contract, so it
feeds straight into the rest of the library. Cities (national capitals, major
regional metros, and recognizable US secondary cities) come back as a metro
bounding box in their UTM zone.

Call :func:`~xrspatial.templates.list_templates` to discover every name
``from_template`` accepts (curated regions, world cities, and country codes).

From Template
=============
.. autosummary::
    :toctree: _autosummary

    xrspatial.templates.from_template
    xrspatial.templates.list_templates
