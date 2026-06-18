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

Reprojecting reads need a target grid that the plain ``open_dataset``
path does not have, so pass the target as the *value* of the mode you
want -- a DataArray or Dataset. ``coregister=target`` reprojects and
resamples the source onto ``target``'s exact grid; ``auto_reproject=
target`` reprojects onto ``target``'s CRS but keeps the file's native
resolution. The engine routes the read through the target's
``.xrs.open_geotiff`` accessor::

    xr.open_dataset(
        "scene.tif", engine="xrspatial",
        backend_kwargs={"coregister": target},
    )

    xr.open_dataset(
        "scene.tif", engine="xrspatial",
        backend_kwargs={"auto_reproject": target},
    )

``resampling`` / ``var`` are modifiers of those reads, so they need a
target too. A bare ``coregister=True`` / ``auto_reproject=True`` (no
grid), a lone ``resampling`` / ``var``, passing a target on both
``coregister`` and ``auto_reproject``, or the removed ``like=`` kwarg
each raise a pointed ``ValueError``.
"""
from __future__ import annotations

import os

import xarray as xr
from xarray.backends import BackendEntrypoint

# Name for the one data variable when ``open_geotiff`` cannot derive one
# from the source (e.g. an in-memory file-like object with no path).
_DEFAULT_VARIABLE_NAME = "band_data"

# Backend kwargs only the reprojecting-read path (``.xrs.open_geotiff`` on
# the target) understands. Supplied without a target grid they would reach
# the standalone reader and raise an opaque ``TypeError``; the engine
# raises a pointed ``ValueError`` instead.
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

    def open_dataset(self, filename_or_obj, *, drop_variables=None, **kwargs):
        # Imported here rather than at module scope so importing this
        # backend module stays cheap; the heavy reader package only loads
        # when a source is actually opened.
        from . import open_geotiff

        if 'like' in kwargs:
            raise ValueError(
                "the 'like=' backend kwarg was removed; pass the target grid "
                "as the value of 'coregister' or 'auto_reproject', e.g. "
                "backend_kwargs={'coregister': target}."
            )

        # The target grid rides on the mode parameter's value: a DataArray
        # or Dataset on ``coregister`` / ``auto_reproject`` is the grid to
        # read onto. An xarray object is ambiguous in a boolean context, so
        # detect the array form here and hand the accessor a plain ``True``;
        # the accessor's bool-based logic stays untouched.
        cg = kwargs.get('coregister')
        ar = kwargs.get('auto_reproject')
        cg_target = cg if isinstance(cg, (xr.DataArray, xr.Dataset)) else None
        ar_target = ar if isinstance(ar, (xr.DataArray, xr.Dataset)) else None
        if cg_target is not None and ar_target is not None:
            raise ValueError(
                "pass the target grid on exactly one of 'coregister' or "
                "'auto_reproject', not both."
            )
        target = cg_target if cg_target is not None else ar_target

        if target is not None:
            # Importing the accessor module registers the ``.xrs`` accessor
            # that carries the reprojecting-read path; the target may be a
            # DataArray or a Dataset and the accessor dispatches on its type
            # (Datasets also honour the ``var=`` kwarg).
            from .. import accessor  # noqa: F401
            kwargs['coregister' if cg_target is not None
                   else 'auto_reproject'] = True
            da = target.xrs.open_geotiff(filename_or_obj, **kwargs)
        else:
            # No target grid: a truthy ``coregister`` / ``auto_reproject`` or
            # a lone ``resampling`` / ``var`` cannot run.
            offending = [k for k in _COREGISTER_ONLY_KWARGS if kwargs.get(k)]
            if offending:
                raise ValueError(
                    f"{', '.join(offending)} need a target grid. Pass it as "
                    "the value of 'coregister' or 'auto_reproject', e.g. "
                    "backend_kwargs={'coregister': target}."
                )
            # A falsy mode flag (e.g. ``coregister=False``) just means a plain
            # read; the standalone reader does not take these kwargs, so drop
            # them rather than leak an opaque ``TypeError``.
            for k in _COREGISTER_ONLY_KWARGS:
                kwargs.pop(k, None)
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
