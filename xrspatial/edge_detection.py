import numpy as np
import xarray as xr

from xrspatial.convolution import convolve_2d
from xrspatial.utils import _validate_raster

# -- Sobel kernels ----------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=np.float64)

# -- Prewitt kernels ---------------------------------------------------------
PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float64)

PREWITT_Y = np.array([[-1, -1, -1],
                      [0,  0,  0],
                      [1,  1,  1]], dtype=np.float64)

# -- Laplacian kernel --------------------------------------------------------
LAPLACIAN_KERNEL = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float64)


def _promote_wide_int(data):
    """Cast 32/64-bit integer arrays to float64 before convolution.

    ``convolve_2d`` promotes integer inputs to float32, whose 24-bit
    mantissa cannot represent integers above 2**24: unit steps between
    large values vanish in the cast and gradients silently collapse to
    zero (#3680). float64 is exact for every int32/uint32 value and for
    int64/uint64 up to 2**53. 8- and 16-bit integers are exactly
    representable in float32, so they keep ``convolve_2d``'s promotion.
    Works on numpy, cupy, and dask arrays alike (``astype`` is lazy on
    dask).
    """
    if data.dtype.kind in 'iu' and data.dtype.itemsize > 2:
        return data.astype(np.float64)
    return data


def sobel_x(agg, name='sobel_x', boundary='nan'):
    """Compute the horizontal gradient of a raster using the Sobel operator.

    Detects vertical edges by cross-correlating with the Sobel-X kernel::

        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]

    This matches ``scipy.ndimage.correlate``: the response is positive
    where values increase toward higher column index.

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
        Integer inputs are computed in floating point: 8/16-bit
        integers as float32, 32/64-bit integers as float64 so that
        large values keep unit precision (exact up to 2**53 for
        64-bit integers).

    Notes
    -----
    NaN cells in the input propagate: every output cell whose 3x3
    neighborhood (as extended by the boundary mode) contains a NaN
    becomes NaN. With the default ``boundary='nan'``, the outer
    one-cell ring of the output is also NaN.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import sobel_x
        >>> data = np.array([
        ...     [0., 1., 2., 3.],
        ...     [0., 1., 2., 3.],
        ...     [0., 1., 2., 3.],
        ...     [0., 1., 2., 3.]])
        >>> raster = xr.DataArray(data, dims=['y', 'x'])
        >>> sobel_x(raster).data
        array([[nan, nan, nan, nan],
               [nan,  8.,  8., nan],
               [nan,  8.,  8., nan],
               [nan, nan, nan, nan]])
    """
    _validate_raster(agg, func_name='sobel_x', name='agg')
    out = convolve_2d(_promote_wide_int(agg.data), SOBEL_X, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)


def sobel_y(agg, name='sobel_y', boundary='nan'):
    """Compute the vertical gradient of a raster using the Sobel operator.

    Detects horizontal edges by cross-correlating with the Sobel-Y kernel::

        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]

    This matches ``scipy.ndimage.correlate``: the response is positive
    where values increase toward higher row index.

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
        Integer inputs are computed in floating point: 8/16-bit
        integers as float32, 32/64-bit integers as float64 so that
        large values keep unit precision (exact up to 2**53 for
        64-bit integers).

    Notes
    -----
    NaN cells in the input propagate: every output cell whose 3x3
    neighborhood (as extended by the boundary mode) contains a NaN
    becomes NaN. With the default ``boundary='nan'``, the outer
    one-cell ring of the output is also NaN.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import sobel_y
        >>> data = np.array([
        ...     [0., 0., 0., 0.],
        ...     [1., 1., 1., 1.],
        ...     [2., 2., 2., 2.],
        ...     [3., 3., 3., 3.]])
        >>> raster = xr.DataArray(data, dims=['y', 'x'])
        >>> sobel_y(raster).data
        array([[nan, nan, nan, nan],
               [nan,  8.,  8., nan],
               [nan,  8.,  8., nan],
               [nan, nan, nan, nan]])
    """
    _validate_raster(agg, func_name='sobel_y', name='agg')
    out = convolve_2d(_promote_wide_int(agg.data), SOBEL_Y, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)


def laplacian(agg, name='laplacian', boundary='nan'):
    """Compute edges using the Laplacian (second-derivative) operator.

    Omnidirectional edge detector using the kernel::

        [[ 0,  1, 0],
         [ 1, -4, 1],
         [ 0,  1, 0]]

    The kernel is symmetric, so cross-correlation and convolution agree.

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
        Integer inputs are computed in floating point: 8/16-bit
        integers as float32, 32/64-bit integers as float64 so that
        large values keep unit precision (exact up to 2**53 for
        64-bit integers).

    Notes
    -----
    NaN cells in the input propagate: every output cell whose 3x3
    neighborhood (as extended by the boundary mode) contains a NaN
    becomes NaN. With the default ``boundary='nan'``, the outer
    one-cell ring of the output is also NaN.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import laplacian
        >>> data = np.zeros((4, 4))
        >>> data[1, 1] = 1.
        >>> raster = xr.DataArray(data, dims=['y', 'x'])
        >>> laplacian(raster).data
        array([[nan, nan, nan, nan],
               [nan, -4.,  1., nan],
               [nan,  1.,  0., nan],
               [nan, nan, nan, nan]])
    """
    _validate_raster(agg, func_name='laplacian', name='agg')
    out = convolve_2d(_promote_wide_int(agg.data), LAPLACIAN_KERNEL, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)


def prewitt_x(agg, name='prewitt_x', boundary='nan'):
    """Compute the horizontal gradient of a raster using the Prewitt operator.

    Detects vertical edges by cross-correlating with the Prewitt-X kernel::

        [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]

    This matches ``scipy.ndimage.correlate``: the response is positive
    where values increase toward higher column index.

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
        Integer inputs are computed in floating point: 8/16-bit
        integers as float32, 32/64-bit integers as float64 so that
        large values keep unit precision (exact up to 2**53 for
        64-bit integers).

    Notes
    -----
    NaN cells in the input propagate: every output cell whose 3x3
    neighborhood (as extended by the boundary mode) contains a NaN
    becomes NaN. With the default ``boundary='nan'``, the outer
    one-cell ring of the output is also NaN.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import prewitt_x
        >>> data = np.array([
        ...     [0., 1., 2., 3.],
        ...     [0., 1., 2., 3.],
        ...     [0., 1., 2., 3.],
        ...     [0., 1., 2., 3.]])
        >>> raster = xr.DataArray(data, dims=['y', 'x'])
        >>> prewitt_x(raster).data
        array([[nan, nan, nan, nan],
               [nan,  6.,  6., nan],
               [nan,  6.,  6., nan],
               [nan, nan, nan, nan]])
    """
    _validate_raster(agg, func_name='prewitt_x', name='agg')
    out = convolve_2d(_promote_wide_int(agg.data), PREWITT_X, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)


def prewitt_y(agg, name='prewitt_y', boundary='nan'):
    """Compute the vertical gradient of a raster using the Prewitt operator.

    Detects horizontal edges by cross-correlating with the Prewitt-Y kernel::

        [[-1, -1, -1],
         [ 0,  0,  0],
         [ 1,  1,  1]]

    This matches ``scipy.ndimage.correlate``: the response is positive
    where values increase toward higher row index.

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
        Integer inputs are computed in floating point: 8/16-bit
        integers as float32, 32/64-bit integers as float64 so that
        large values keep unit precision (exact up to 2**53 for
        64-bit integers).

    Notes
    -----
    NaN cells in the input propagate: every output cell whose 3x3
    neighborhood (as extended by the boundary mode) contains a NaN
    becomes NaN. With the default ``boundary='nan'``, the outer
    one-cell ring of the output is also NaN.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import prewitt_y
        >>> data = np.array([
        ...     [0., 0., 0., 0.],
        ...     [1., 1., 1., 1.],
        ...     [2., 2., 2., 2.],
        ...     [3., 3., 3., 3.]])
        >>> raster = xr.DataArray(data, dims=['y', 'x'])
        >>> prewitt_y(raster).data
        array([[nan, nan, nan, nan],
               [nan,  6.,  6., nan],
               [nan,  6.,  6., nan],
               [nan, nan, nan, nan]])
    """
    _validate_raster(agg, func_name='prewitt_y', name='agg')
    out = convolve_2d(_promote_wide_int(agg.data), PREWITT_Y, boundary)
    return xr.DataArray(out, name=name, coords=agg.coords,
                        dims=agg.dims, attrs=agg.attrs)
