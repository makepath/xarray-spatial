..  _reference.templates:

*********
Templates
*********

Empty study-area grids you can start an analysis from. ``from_template`` turns
a region name or country code into a NaN-filled :class:`xarray.DataArray` that
follows the xarray-spatial array contract, so it feeds straight into the rest
of the library.

From Template
=============
.. autosummary::
    :toctree: _autosummary

    xrspatial.templates.from_template
