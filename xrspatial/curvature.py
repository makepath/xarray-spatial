from __future__ import annotations

# std lib
from functools import partial
from typing import Optional, Union

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import xarray as xr
from numba import cuda

# local modules
from xrspatial.utils import ArrayTypeFunctionMapping
from xrspatial.utils import _boundary_to_dask
from xrspatial.utils import _pad_array
from xrspatial.utils import _validate_boundary
from xrspatial.utils import cuda_args
from xrspatial.utils import get_dataarray_resolution
from xrspatial.utils import ngjit
from xrspatial.dataset_support import supports_dataset


@ngjit
def _cpu(data, cellsize):
    out = np.empty(data.shape, np.float32)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            d = (data[y + 1, x] + data[y - 1, x]) / 2 - data[y, x]
            e = (data[y, x + 1] + data[y, x - 1]) / 2 - data[y, x]
            out[y, x] = -2 * (d + e) * 100 / (cellsize * cellsize)
    return out


def _run_numpy(data: np.ndarray,
               cellsize: Union[int, float],
               boundary: str = 'nan') -> np.ndarray:
    data = data.astype(np.float32)
    if boundary == 'nan':
        return _cpu(data, cellsize)
    padded = _pad_array(data, 1, boundary)
    result = _cpu(padded, cellsize)
    return result[1:-1, 1:-1]


def _run_dask_numpy(data: da.Array,
                    cellsize: Union[int, float],
                    boundary: str = 'nan') -> da.Array:
    data = data.astype(np.float32)
    _func = partial(_cpu, cellsize=cellsize)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary),
                           meta=np.array(()))
    return out


@cuda.jit(device=True)
def _gpu(arr, cellsize):
    d = (arr[1, 0] + arr[1, 2]) / 2 - arr[1, 1]
    e = (arr[0, 1] + arr[2, 1]) / 2 - arr[1, 1]
    curv = -2 * (d + e) * 100 / (cellsize[0] * cellsize[0])
    return curv


@cuda.jit
def _run_gpu(arr, cellsize, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i - di >= 0 and i + di <= out.shape[0] - 1 and
            j - dj >= 0 and j + dj <= out.shape[1] - 1):
        out[i, j] = _gpu(arr[i - di:i + di + 1, j - dj:j + dj + 1], cellsize)


def _run_cupy(data: cupy.ndarray,
              cellsize: Union[int, float],
              boundary: str = 'nan') -> cupy.ndarray:
    if boundary != 'nan':
        padded = _pad_array(data, 1, boundary)
        result = _run_cupy(padded, cellsize)
        return result[1:-1, 1:-1]

    data = data.astype(cupy.float32)
    cellsize_arr = cupy.array([float(cellsize)], dtype='f4')

    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan

    _run_gpu[griddim, blockdim](data, cellsize_arr, out)

    return out


def _run_dask_cupy(data: da.Array,
                   cellsize: Union[int, float],
                   boundary: str = 'nan') -> da.Array:
    data = data.astype(cupy.float32)
    cellsize_arr = cupy.array([float(cellsize)], dtype='f4')

    _func = partial(_run_cupy, cellsize=cellsize_arr)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary, is_cupy=True),
                           meta=cupy.array(()))
    return out


@supports_dataset
def curvature(agg: xr.DataArray,
              name: Optional[str] = 'curvature',
              boundary: str = 'nan') -> xr.DataArray:
    """
    Calculates, for all cells in the array, the curvature (second
    derivative) of each cell based on the elevation of its neighbors
    in a 3x3 grid. A positive curvature indicates the surface is
    upwardly convex. A negative value indicates it is upwardly
    concave. A value of 0 indicates a flat surface.

    Units of the curvature output raster are one hundredth (1/100)
    of a z-unit.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask xarray DataArray of elevation values.
        Must contain `res` attribute.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='curvature'
        Name of output DataArray.
    boundary : str, default='nan'
        How to handle edges where the kernel extends beyond the raster.
        ``'nan'``     — fill missing neighbours with NaN (default).
        ``'nearest'`` — repeat edge values.
        ``'reflect'`` — mirror at boundary.
        ``'wrap'``    — periodic / toroidal.

    Returns
    -------
    curvature_agg : xarray.DataArray or xr.Dataset
        If `agg` is a DataArray, returns a DataArray of the same type.
        If `agg` is a Dataset, returns a Dataset with curvature computed
        for each data variable.
        2D aggregate array of curvature values.
        All other input attributes are preserved.

    References
    ----------
        - arcgis: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-curvature-works.htm # noqa

    Examples
    --------
    Curvature works with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import dask.array as da
        >>> import xarray as xr
        >>> from xrspatial import curvature
        >>> flat_data = np.zeros((5, 5), dtype=np.float32)
        >>> flat_raster = xr.DataArray(flat_data, attrs={'res': (1, 1)})
        >>> flat_curv = curvature(flat_raster)
        >>> print(flat_curv)
        <xarray.DataArray 'curvature' (dim_0: 5, dim_1: 5)>
        array([[nan, nan, nan, nan, nan],
               [nan, -0., -0., -0., nan],
               [nan, -0., -0., -0., nan],
               [nan, -0., -0., -0., nan],
               [nan, nan, nan, nan, nan]])
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (1, 1)

    Curvature works with Dask with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> convex_data = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32)
        >>> convex_raster = xr.DataArray(
            da.from_array(convex_data, chunks=(3, 3)),
            attrs={'res': (10, 10)}, name='convex_dask_numpy_raster')
        >>> print(convex_raster)
        <xarray.DataArray 'convex_dask_numpy_raster' (dim_0: 5, dim_1: 5)>
        dask.array<array, shape=(5, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10, 10)
        >>> convex_curv = curvature(convex_raster, name='convex_curvature')
        >>> print(convex_curv)  # return a xarray DataArray with Dask-backed array
        <xarray.DataArray 'convex_curvature' (dim_0: 5, dim_1: 5)>
        dask.array<_trim, shape=(5, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10, 10)
        >>> print(convex_curv.compute())
        <xarray.DataArray 'convex_curvature' (dim_0: 5, dim_1: 5)>
        array([[nan, nan, nan, nan, nan],
               [nan, -0.,  1., -0., nan],
               [nan,  1., -4.,  1., nan],
               [nan, -0.,  1., -0., nan],
               [nan, nan, nan, nan, nan]])
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10, 10)

    Curvature works with CuPy backed xarray DataArray.
    .. sourcecode:: python

        >>> import cupy
        >>> concave_data = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32)
        >>> concave_raster = xr.DataArray(
            cupy.asarray(concave_data),
            attrs={'res': (10, 10)}, name='concave_cupy_raster')
        >>> concave_curv = curvature(concave_raster)
        >>> print(type(concave_curv.data))
        <class 'cupy.core.core.ndarray'>
        >>> print(concave_curv)
        <xarray.DataArray 'curvature' (dim_0: 5, dim_1: 5)>
        array([[nan, nan, nan, nan, nan],
               [nan, -0., -1., -0., nan],
               [nan, -1.,  4., -1., nan],
               [nan, -0., -1., -0., nan],
               [nan, nan, nan, nan, nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10, 10)
    """
    cellsize_x, cellsize_y = get_dataarray_resolution(agg)
    cellsize = (cellsize_x + cellsize_y) / 2

    _validate_boundary(boundary)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=_run_dask_cupy
    )
    out = mapper(agg)(agg.data, cellsize, boundary)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
