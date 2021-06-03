import rioxarray  # noqa: F401
import rasterio
from rasterio.warp import calculate_default_transform, reproject
import dask.array as da
import xarray as xr
import numpy as np
from xrspatial.utils import ArrayTypeFunctionMapping
import math


def gen_coords_list(arr):
    coords_list = [(coord, arr.coords[coord])
                   for coord in arr.coords if coord in ('band', 'y', 'x')]
    return coords_list


def calc_chunks(src_crs,
                dst_crs,
                src_shape,
                src_bounds,
                numblocks):
    bands, height, width = src_shape
    dst_transform, dst_width, dst_height =\
        calculate_default_transform(src_crs,
                                    dst_crs,
                                    width,
                                    height,
                                    *src_bounds)
    dst_shape = (bands, dst_height, dst_width)
    new_chunksizes = []
    for i in range(len(numblocks)):
        new_chunksizes.append([])
        for j in range(numblocks[i]):
            chunk_dim = int(math.ceil(dst_shape[i]/numblocks[i]))
            new_chunksizes[i].append(chunk_dim)
    diffs = [dst_shape[i] - sum(new_chunksizes[i])
             for i in range(len(new_chunksizes))]
    for i in range(len(diffs)):
        if diffs[i] != 0:
            new_chunksizes[i][-1] += diffs[i]
# TODO: decide if needs to check if adjustment goes into next chunk
#             adjusted = new_chunksizes_raw[i][-1] += diffs[i]
#             if adjusted < 0:
    return new_chunksizes


def reproject_nparr(source,
                    src_transform,
                    src_crs,
                    dst_crs,
                    src_bounds,
                    target_shape=None,
                    **kwargs):
    _, height, width = source.shape

    if target_shape is not None:
        _t, target_height, target_width = target_shape
    else:
        target_height = target_width = None

    dst_transform, dst_width, dst_height =\
        calculate_default_transform(src_crs,
                                    dst_crs,
                                    width,
                                    height,
                                    *src_bounds,
                                    dst_width=target_width,
                                    dst_height=target_height)

    destination = np.zeros((1, dst_height, dst_width), dtype=np.uint16)

    destination, transform =\
        reproject(source,
                  destination=destination,
                  src_transform=src_transform,
                  src_crs=src_crs,
                  dst_transform=dst_transform,
                  dst_crs=dst_crs,
                  **kwargs)

    return (destination, transform)


def _send_blocks_reproject(data,
                           src_transform,
                           src_crs,
                           dst_crs,
                           src_res,
                           src_coords,
                           block_info=None,
                           **kwargs):
    if block_info is not None:
        _, block_height, block_width = data.shape
        arr_left, arr_top = block_info[0]['array-location'][2][0],\
            block_info[0]['array-location'][1][1]
        arr_right, arr_bottom = block_info[0]['array-location'][2][1],\
            block_info[0]['array-location'][1][0]
        full_x_coord = src_coords[2][1]
        full_y_coord = src_coords[1][1]
        left, right = float(full_x_coord[arr_left].data),\
            float(full_x_coord[arr_right - 1].data)
        bottom, top = float(full_y_coord[arr_bottom].data),\
            float(full_y_coord[arr_top - 1].data)

        src_res_x, src_res_y = src_res
        x_off = src_res_x / 2
        y_off = src_res_y / 2

        block_bounds = (left - x_off,
                        -bottom + y_off,
                        right + x_off,
                        -top - y_off)

        block_transform = rasterio.transform.from_bounds(*block_bounds,
                                                         block_width,
                                                         block_height)

        dst_array_location = block_info[None]['array-location']
        target_dst_shape = []
        for i in range(len(dst_array_location)):
            target_dst_shape.append(dst_array_location[i][1] -
                                    dst_array_location[i][0])

        reprojected_data, _t = reproject_nparr(data,
                                               block_transform,
                                               src_crs,
                                               dst_crs,
                                               block_bounds,
                                               target_dst_shape,
                                               **kwargs)

        return reprojected_data
    else:
        return None


def _numpy_reproject(arr, dst_crs, **kwargs):
    data = arr.data.astype(np.uint16)
    src_transform = arr.rio.transform()
    src_crs = arr.rio.crs
    _, src_height, src_width = data.shape
    src_bounds = rasterio.transform.array_bounds(src_height,
                                                 src_width,
                                                 src_transform)

    reprojected_data, dst_transform = reproject_nparr(data,
                                                      src_transform,
                                                      src_crs,
                                                      dst_crs,
                                                      src_bounds,
                                                      **kwargs)

    return (reprojected_data, dst_transform)


def _cupy_reproject():
    raise NotImplementedError('cupy not implemented yet')


def _dask_cupy_reproject():
    raise NotImplementedError('dask cupy not implemented yet')


def _dask_reproject(arr, dst_crs, **kwargs):
    data = arr.data.astype(np.uint16)
    _, src_height, src_width = src_shape = data.shape
    src_transform = arr.rio.transform()
    src_crs = arr.rio.crs
    src_bounds = arr.rio.bounds()
    src_res = arr.rio.resolution()
    src_coords = gen_coords_list(arr)

    numblocks = data.numblocks
    chunks = calc_chunks(src_crs, dst_crs, src_shape, src_bounds, numblocks)

    reprojected_data = da.map_blocks(_send_blocks_reproject,
                                     data,
                                     src_transform,
                                     src_crs,
                                     dst_crs,
                                     src_res,
                                     src_coords,
                                     chunks=chunks,
                                     dtype=np.uint16,
                                     **kwargs)

    _b, ret_height, ret_width = reprojected_data.shape
    dst_transform, w, h = calculate_default_transform(src_crs,
                                                      dst_crs,
                                                      src_width,
                                                      src_height,
                                                      *src_bounds,
                                                      dst_width=ret_width,
                                                      dst_height=ret_height)

    return (reprojected_data, dst_transform)


def xrs_reproject(arr, dst_crs, **kwargs):
    mapper = ArrayTypeFunctionMapping(numpy_func=_numpy_reproject,
                                      cupy_func=_cupy_reproject,
                                      dask_cupy_func=_dask_cupy_reproject,
                                      dask_func=_dask_reproject)
    reprojected_data, dst_transform = mapper(arr)(arr, dst_crs, **kwargs)

    reprojected_data = np.rot90(np.rot90(reprojected_data))

    _, dst_height, dst_width = reprojected_data.shape

    left, bottom, right, top =\
        rasterio.transform.array_bounds(dst_height,
                                        dst_width,
                                        dst_transform)

    xres, yres = (right - left)/dst_width, (top - bottom)/dst_height
    xoff, yoff = dst_transform.xoff, dst_transform.yoff

    dst_xs = np.arange(dst_width) * xres + (xoff + xres/2)
    dst_ys = np.arange(dst_height) * yres + (yoff + yres/2)

    xs_da = xr.DataArray(dst_xs, dims=('x'))
    xs_da.coords['x'] = dst_xs
    ys_da = xr.DataArray(dst_ys, dims=('y'))
    ys_da.coords['y'] = dst_ys

    reprojected_da = xr.DataArray(reprojected_data, dims=('band', 'y', 'x'))
    reprojected_da.coords['band'] = arr.coords['band']
    reprojected_da.coords['x'] = xs_da
    reprojected_da.coords['y'] = ys_da
    reprojected_da.rio.write_crs(dst_crs, inplace=True)
    reprojected_da.rio.write_transform(inplace=True)

    return reprojected_da
