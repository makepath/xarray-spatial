import rioxarray  # noqa: F401
from rioxarray.rioxarray import _make_coords
from rasterio.warp import calculate_default_transform
import numpy as np
import xarray as xr
from xrspatial.utils import ArrayTypeFunctionMapping


def _block_reproject(block_da, dst_crs, **kwargs):
    reprojected_block = block_da.rio.reproject(dst_crs, **kwargs)
    return reprojected_block


def _numpy_reproject(arr, dst_crs, **kwargs):
    return _block_reproject(arr, dst_crs, **kwargs)


def _cupy_reproject(arr, dst_crs, **kwargs):
    raise NotImplementedError('no cupy yet')


def _dask_cupy_reproject(arr, dst_crs, **kwargs):
    raise NotImplementedError('no dask cupy yet')


def _dask_reproject(arr, dst_crs, **kwargs):
    src_crs = arr.rio.crs
    arr.data = arr.data.astype(np.uint16)
    src_b, src_height, src_width = arr.data.shape
    src_bounds = arr.rio.bounds()

    dst_transform, dst_width, dst_height =\
        calculate_default_transform(src_crs,
                                    dst_crs,
                                    src_width,
                                    src_height,
                                    *src_bounds)
    dst_shape = (src_b, dst_height, dst_width)

    template = xr.DataArray(np.zeros(dst_shape, dtype=np.uint16),
                            dims=('band', 'y', 'x')).chunk()
    coords = _make_coords(template,
                          dst_transform,
                          dst_width,
                          dst_height)

    template.coords['x'] = coords['x']
    template.coords['y'] = coords['y']

    template.coords['band'] = xr.DataArray(np.arange(1, src_b + 1,
                                           dtype=np.int64),
                                           dims=('band'))

    template.rio.write_crs(dst_crs, inplace=True)
    template.rio.write_transform(inplace=True)

    reprojected = arr.map_blocks(_block_reproject,
                                 template=template,
                                 args=[dst_crs],
                                 kwargs=dict(kwargs))
    return reprojected


def reproject(arr, dst_crs, **kwargs):
    mapper = ArrayTypeFunctionMapping(numpy_func=_numpy_reproject,
                                      cupy_func=_cupy_reproject,
                                      dask_func=_dask_reproject,
                                      dask_cupy_func=_dask_cupy_reproject)
    reprojected = mapper(arr)(arr, dst_crs, **kwargs)
    return reprojected
