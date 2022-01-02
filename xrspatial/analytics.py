from xrspatial import slope
from xrspatial import curvature
from xrspatial import aspect

import xarray as xr


def summarize_terrain(raster: xr.DataArray):

    if raster.name is None:
        raise NameError('Requires xr.DataArray.name property to be set')

    ds = raster.to_dataset()
    ds[f'{raster.name}-slope'] = slope(raster)
    ds[f'{raster.name}-curvature'] = curvature(raster)
    ds[f'{raster.name}-aspect'] = aspect(raster)
    return ds
