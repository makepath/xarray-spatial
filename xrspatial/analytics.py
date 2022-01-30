from xrspatial import slope
from xrspatial import curvature
from xrspatial import aspect

import xarray as xr


def summarize_terrain(raster: xr.DataArray):
    """
    Calculates slope, aspect, and curvature of an elevation raster and return a dataset
    of the computed data.

    Parameters
    ----------
    raster: xarray.DataArray
        2D NumPy, CuPy, or Dask with NumPy-backed xarray DataArray of elevation values.

    Returns
    -------
    summarized_terrain: xarray.Dataset
        Dataset with slope, aspect, curvature variables with a naming convention of
        `raster.name-variable_name`

    Examples
    --------

    """

    if raster.name is None:
        raise NameError('Requires xr.DataArray.name property to be set')

    ds = raster.to_dataset()
    ds[f'{raster.name}-slope'] = slope(raster)
    ds[f'{raster.name}-curvature'] = curvature(raster)
    ds[f'{raster.name}-aspect'] = aspect(raster)
    return ds
