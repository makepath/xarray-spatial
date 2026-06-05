"""Writer entry points for the geotiff module.

Holds ``to_geotiff`` (and its eager-path helpers), ``_write_geotiff_gpu``,
and ``build_vrt`` in sibling modules. The package ``__init__`` stays
empty so nothing leaks into ``xrspatial.geotiff`` through implicit
re-exports.
"""
