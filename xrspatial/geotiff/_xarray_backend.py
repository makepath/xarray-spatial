"""xarray backend entry point for the native GeoTIFF/COG/VRT reader.

Registers :func:`open_geotiff` under xarray's pluggable backend API so a
GeoTIFF source can be opened through the standard entry point::

    import xarray as xr

    xr.open_dataset("dem.tif", engine="xrspatial_geotiff")
    xr.open_mfdataset("*.tif", engine="xrspatial_geotiff")

The entry point is declared in ``setup.cfg`` under
``[options.entry_points] xarray.backends``. ``open_geotiff`` returns a
:class:`~xarray.DataArray`; xarray backends must return a
:class:`~xarray.Dataset`, so this wrapper promotes the single array to a
one-variable dataset.

GeoTIFF-specific read options (``gpu``, ``masked``, ``band``,
``overview_level``, ``window``, ``bbox``, ``stable_only``, ...) are
forwarded to :func:`open_geotiff` through xarray's ``backend_kwargs``::

    xr.open_dataset(
        "dem.tif", engine="xrspatial_geotiff",
        backend_kwargs={"masked": True, "overview_level": 1},
    )

``chunks`` is the one exception: xarray reserves it as a top-level
argument to ``open_dataset``, so it cannot travel through
``backend_kwargs``. Pass ``chunks=`` directly to ``open_dataset`` to get
a dask-backed dataset (xarray wraps the eager read)::

    xr.open_dataset("dem.tif", engine="xrspatial_geotiff", chunks={})
"""
from __future__ import annotations

import os

from xarray.backends import BackendEntrypoint

# Name for the one data variable when ``open_geotiff`` cannot derive one
# from the source (e.g. an in-memory file-like object with no path).
_DEFAULT_VARIABLE_NAME = "band_data"

# Extensions ``guess_can_open`` claims so ``xr.open_dataset`` /
# ``open_mfdataset`` can auto-select this engine without ``engine=``.
_SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".vrt")


class GeoTIFFBackendEntrypoint(BackendEntrypoint):
    """Open GeoTIFF / COG / VRT files with xrspatial's no-GDAL reader.

    Thin wrapper that calls :func:`xrspatial.geotiff.open_geotiff` and
    promotes its ``DataArray`` to a one-variable ``Dataset``.
    """

    description = (
        "Open GeoTIFF/COG/VRT files using xrspatial's native (no-GDAL) "
        "reader via xrspatial.geotiff.open_geotiff"
    )
    url = "https://github.com/xarray-contrib/xarray-spatial"
    # ``open_geotiff`` takes ~30 keyword options forwarded verbatim via
    # ``**kwargs``, so the parameter list is declared explicitly here:
    # xarray's signature introspection (``detect_parameters``) raises on an
    # ``open_dataset`` that uses ``**kwargs`` without this attribute set. It
    # also stops xarray from injecting its CF decoders -- in particular
    # ``mask_and_scale``, which would collide with open_geotiff's deprecated
    # alias of the same name. GeoTIFF read options come in through
    # ``backend_kwargs`` instead.
    open_dataset_parameters = ("filename_or_obj", "drop_variables")

    def open_dataset(self, filename_or_obj, *, drop_variables=None, **kwargs):
        # Imported here rather than at module scope so importing this
        # backend module stays cheap; the heavy reader package only loads
        # when a source is actually opened.
        from . import open_geotiff

        da = open_geotiff(filename_or_obj, **kwargs)
        name = da.name if da.name is not None else _DEFAULT_VARIABLE_NAME
        ds = da.to_dataset(name=name)
        if drop_variables is not None:
            ds = ds.drop_vars(drop_variables, errors="ignore")
        return ds

    def guess_can_open(self, filename_or_obj):
        if isinstance(filename_or_obj, os.PathLike):
            filename_or_obj = os.fspath(filename_or_obj)
        if not isinstance(filename_or_obj, str):
            return False
        # Strip any query string / fragment so COG URLs such as
        # "https://host/dem.tif?token=..." still match on extension.
        path = filename_or_obj.split("?", 1)[0].split("#", 1)[0]
        return path.lower().endswith(_SUPPORTED_EXTENSIONS)
