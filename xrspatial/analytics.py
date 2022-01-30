from xrspatial import slope
from xrspatial import curvature
from xrspatial import aspect

import xarray as xr


def summarize_terrain(terrain: xr.DataArray):
    """
    Calculates slope, aspect, and curvature of an elevation terrain and return a dataset
    of the computed data.

    Parameters
    ----------
    terrain: xarray.DataArray
        2D NumPy, CuPy, or Dask with NumPy-backed xarray DataArray of elevation values.

    Returns
    -------
    summarized_terrain: xarray.Dataset
        Dataset with slope, aspect, curvature variables with a naming convention of
        `terrain.name-variable_name`

    Examples
    --------

    """

    if terrain.name is None:
        raise NameError('Requires xr.DataArray.name property to be set')

    ds = terrain.to_dataset()
    ds[f'{terrain.name}-slope'] = slope(terrain)
    ds[f'{terrain.name}-curvature'] = curvature(terrain)
    ds[f'{terrain.name}-aspect'] = aspect(terrain)
    return ds
