import numpy as np
import xarray as xr

from xrspatial.convolution import convolve_2d
from xrspatial.utils import _validate_raster

# -- Sobel kernels ----------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)

# -- Prewitt kernels ---------------------------------------------------------
PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float64)

PREWITT_Y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float64)

# -- Laplacian kernel --------------------------------------------------------
LAPLACIAN_KERNEL = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float64)


def sobel_x(agg, name='sobel_x', boundary='nan'):
    """Compute the horizontal gradient of a raster using the Sobel operator.

    Detects vertical edges by convolving with the Sobel-X kernel::

        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster. Supports NumPy, CuPy, Dask+NumPy, and Dask+CuPy backends.
    name : str, default='sobel_x'
        Name for the output DataArray.
    boundary : str, default='nan'
        How to handle edges: 'nan', 'nearest', 'reflect', or 'wrap'.

    Returns
    -------
    xarray.DataArray
        Horizontal gradient with the same shape and backend as the input.
    """
    _validate_raster(agg, func_name='sobel_x', name='agg')
    out = convolve_2d(agg.data, SOBEL_X, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)


def sobel_y(agg, name='sobel_y', boundary='nan'):
    """Compute the vertical gradient of a raster using the Sobel operator.

    Detects horizontal edges by convolving with the Sobel-Y kernel::

        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster. Supports NumPy, CuPy, Dask+NumPy, and Dask+CuPy backends.
    name : str, default='sobel_y'
        Name for the output DataArray.
    boundary : str, default='nan'
        How to handle edges: 'nan', 'nearest', 'reflect', or 'wrap'.

    Returns
    -------
    xarray.DataArray
        Vertical gradient with the same shape and backend as the input.
    """
    _validate_raster(agg, func_name='sobel_y', name='agg')
    out = convolve_2d(agg.data, SOBEL_Y, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)


def laplacian(agg, name='laplacian', boundary='nan'):
    """Compute edges using the Laplacian (second-derivative) operator.

    Omnidirectional edge detector using the kernel::

        [[ 0,  1, 0],
         [ 1, -4, 1],
         [ 0,  1, 0]]

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster. Supports NumPy, CuPy, Dask+NumPy, and Dask+CuPy backends.
    name : str, default='laplacian'
        Name for the output DataArray.
    boundary : str, default='nan'
        How to handle edges: 'nan', 'nearest', 'reflect', or 'wrap'.

    Returns
    -------
    xarray.DataArray
        Laplacian response with the same shape and backend as the input.
    """
    _validate_raster(agg, func_name='laplacian', name='agg')
    out = convolve_2d(agg.data, LAPLACIAN_KERNEL, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)


def prewitt_x(agg, name='prewitt_x', boundary='nan'):
    """Compute the horizontal gradient of a raster using the Prewitt operator.

    Detects vertical edges by convolving with the Prewitt-X kernel::

        [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster. Supports NumPy, CuPy, Dask+NumPy, and Dask+CuPy backends.
    name : str, default='prewitt_x'
        Name for the output DataArray.
    boundary : str, default='nan'
        How to handle edges: 'nan', 'nearest', 'reflect', or 'wrap'.

    Returns
    -------
    xarray.DataArray
        Horizontal gradient with the same shape and backend as the input.
    """
    _validate_raster(agg, func_name='prewitt_x', name='agg')
    out = convolve_2d(agg.data, PREWITT_X, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)


def prewitt_y(agg, name='prewitt_y', boundary='nan'):
    """Compute the vertical gradient of a raster using the Prewitt operator.

    Detects horizontal edges by convolving with the Prewitt-Y kernel::

        [[-1, -1, -1],
         [ 0,  0,  0],
         [ 1,  1,  1]]

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster. Supports NumPy, CuPy, Dask+NumPy, and Dask+CuPy backends.
    name : str, default='prewitt_y'
        Name for the output DataArray.
    boundary : str, default='nan'
        How to handle edges: 'nan', 'nearest', 'reflect', or 'wrap'.

    Returns
    -------
    xarray.DataArray
        Vertical gradient with the same shape and backend as the input.
    """
    _validate_raster(agg, func_name='prewitt_y', name='agg')
    out = convolve_2d(agg.data, PREWITT_Y, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)
