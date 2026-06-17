"""xarray backend entry point for the native GeoTIFF/COG/VRT reader.

Registers :func:`open_geotiff` under xarray's pluggable backend API so a
GeoTIFF source can be opened through the standard entry point::

    import xarray as xr

    xr.open_dataset("dem.tif", engine="xrspatial")
    xr.open_mfdataset("*.tif", engine="xrspatial")

The entry point is declared in ``setup.cfg`` under
``[options.entry_points] xarray.backends``. ``open_geotiff`` returns a
:class:`~xarray.DataArray`; xarray backends must return a
:class:`~xarray.Dataset`, so this wrapper promotes the single array to a
one-variable dataset.

GeoTIFF-specific read options (``gpu``, ``masked``, ``band``,
``overview_level``, ``window``, ``bbox``, ``stable_only``, ...) are
forwarded to :func:`open_geotiff` through xarray's ``backend_kwargs``::

    xr.open_dataset(
        "dem.tif", engine="xrspatial",
        backend_kwargs={"masked": True, "overview_level": 1},
    )

``chunks`` is the one exception: xarray reserves it as a top-level
argument to ``open_dataset``, so it cannot travel through
``backend_kwargs``. Pass ``chunks=`` directly to ``open_dataset`` to get
a dask-backed dataset (xarray wraps the eager read)::

    xr.open_dataset("dem.tif", engine="xrspatial", chunks={})

Coregistered reads (``coregister`` / ``auto_reproject`` / ``resampling``)
reproject and resample a source onto an existing array's grid, so they
need a target grid that the plain ``open_dataset`` path does not have.
Pass that target as a ``like=`` backend kwarg (a DataArray or Dataset);
the engine then routes to the ``.xrs.open_geotiff`` accessor on ``like``
instead of the standalone reader::

    xr.open_dataset(
        "scene.tif", engine="xrspatial",
        backend_kwargs={"like": target, "coregister": True,
                        "auto_reproject": True},
    )

``coregister`` / ``auto_reproject`` / ``resampling`` / ``var`` without a
``like=`` raise ``ValueError`` pointing at it, rather than the opaque
``TypeError`` the standalone reader would emit for the unknown kwarg.
"""
from __future__ import annotations

import os

import xarray as xr
from xarray.backends import BackendEntrypoint

# Name for the one data variable when ``open_geotiff`` cannot derive one
# from the source (e.g. an in-memory file-like object with no path).
_DEFAULT_VARIABLE_NAME = "band_data"

# Backend kwargs only the coregistered-read path (``.xrs.open_geotiff`` on
# ``like``) understands. Supplied without ``like=`` they would reach the
# standalone reader and raise an opaque ``TypeError``; the engine raises a
# pointed ``ValueError`` instead.
_COREGISTER_ONLY_KWARGS = ("coregister", "auto_reproject", "resampling", "var")

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

    def open_dataset(self, filename_or_obj, *, drop_variables=None,
                     like=None, **kwargs):
        # Imported here rather than at module scope so importing this
        # backend module stays cheap; the heavy reader package only loads
        # when a source is actually opened.
        from . import open_geotiff

        if like is not None:
            if not isinstance(like, (xr.DataArray, xr.Dataset)):
                raise TypeError(
                    "'like=' must be an xarray DataArray or Dataset whose "
                    "grid the read coregisters onto, got "
                    f"{type(like).__name__}."
                )
            # Importing the accessor module registers the ``.xrs``
            # accessor that carries the coregistered-read path; ``like``
            # may be a DataArray or a Dataset and the accessor dispatches
            # on its type (Datasets also honour the ``var=`` kwarg).
            from .. import accessor  # noqa: F401
            da = like.xrs.open_geotiff(filename_or_obj, **kwargs)
        else:
            offending = [k for k in _COREGISTER_ONLY_KWARGS if k in kwargs]
            if offending:
                raise ValueError(
                    f"{', '.join(offending)} only apply when reading onto a "
                    "target grid, so they need a target. Pass it as a "
                    "'like=' backend kwarg (a DataArray or Dataset), e.g. "
                    "backend_kwargs={'like': target, 'coregister': True}."
                )
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
