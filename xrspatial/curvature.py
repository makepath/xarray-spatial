import numpy as np
import xarray as xr
from numba import stencil, vectorize


@stencil
def kernel_D(arr):
    return (arr[0, -1] + arr[0, 1]) / 2 - arr[0, 0]


@stencil
def kernel_E(arr):
    return (arr[-1, 0] + arr[1, 0]) / 2 - arr[0, 0]


@vectorize(["float64(float64, float64)"], nopython=True, target="parallel")
def _curvature(matrix_D, matrix_E):
    curv = -2 * (matrix_D + matrix_E) * 100
    return curv


def curvature(raster):
    """Compute the curvature (second derivatives) of a raster surface.

    Parameters
    ----------
    raster: xarray.DataArray
        2D input raster image with shape=(height, width)

    Returns
    -------
    curvature: xarray.DataArray
        Curvature image with shape=(height, width)
    """

    if not isinstance(raster, xr.DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    if not (issubclass(raster.values.dtype.type, np.integer) or
            issubclass(raster.values.dtype.type, np.float)):
        raise ValueError(
            "`raster` must be an array of integers or float")

    raster_values = raster.values

    cell_size_x = 1
    cell_size_y = 1

    # calculate cell size from input `raster`
    for dim in raster.dims:
        if (dim.lower().count('x')) > 0 or (dim.lower().count('lon')) > 0:
            # dimension of x-coordinates
            if len(raster[dim]) > 1:
                cell_size_x = raster[dim].values[1] - raster[dim].values[0]
        elif (dim.lower().count('y')) > 0 or (dim.lower().count('lat')) > 0:
            # dimension of y-coordinates
            if len(raster[dim]) > 1:
                cell_size_y = raster[dim].values[1] - raster[dim].values[0]

    cell_size = (cell_size_x + cell_size_y) / 2

    matrix_D = kernel_D(raster_values)
    matrix_E = kernel_E(raster_values)

    curvature_values = _curvature(matrix_D, matrix_E) / (cell_size * cell_size)
    result = xr.DataArray(curvature_values,
                          coords=raster.coords,
                          dims=raster.dims,
                          attrs=raster.attrs)

    return result
