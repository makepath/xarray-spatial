import rioxarray  # noqa: F401
from rasterio.warp import calculate_default_transform
import dask.array as da
import xarray as xr
import numpy as np
from xrspatial.utils import ArrayTypeFunctionMapping
import math


def verify_here():
    print('reprojection is here')


def add_coords_crs_on_arr(arr: xr.DataArray, coords_list, crs):
    for coord in coords_list:
        coord_name = coord[0]
        coord_arr = coord[1]
        arr.coords[coord_name] = coord_arr
    arr.rio.set_crs(crs, inplace=True)
    arr.rio.write_crs(crs, inplace=True)
    arr.rio.write_transform(inplace=True)
    return arr


def reproject_coords(src_shape, src_coords, src_crs, dst_crs):
    src_empty = np.empty(src_shape)
    src_empty_da = xr.DataArray(src_empty, dims=('band', 'y', 'x'))
    src_empty_da = add_coords_crs_on_arr(src_empty_da, src_coords, src_crs)
    dst_empty_da = src_empty_da.rio.reproject(dst_crs)
    dst_coords = gen_coords_list(dst_empty_da)
    return dst_coords


def gen_coords_list(src_da):
    src_coords = [[coord, src_da.coords[coord]]
                  for coord in src_da.coords
                  if coord in ('band', 'y', 'x')]
    return src_coords


def gen_chunk_locations(numblocks):
    chunk_locations = []
    for i in range(numblocks[0]):
        for j in range(numblocks[1]):
            for k in range(numblocks[2]):
                chunk_locations.append([i, j, k])
    return chunk_locations


def list_chunk_shapes(chunks, chunk_locations):
    shapes = []
    chunks = [list(chnk) for chnk in chunks]
    for loc in chunk_locations:
        shape = []
        for i in range(len(chunks)):
            shape.append(chunks[i][loc[i]])
        shapes.append(shape)
    return shapes


def gen_array_locations(chunk_locations, shapes):
    array_locations = []
    for i in range(len(shapes)):
        array_locations.append([])
        for j in range(3):
            if i > 0:
                sum_previous = 0
                for k in range(chunk_locations[i][j]):
                    sum_previous +=\
                        (array_locations[k][j][1] - array_locations[k][j][0])
                loc = [sum_previous, sum_previous + shapes[i][j]]
            else:
                loc = [0, shapes[i][j]]
            array_locations[i].append(loc)
    return array_locations


def gen_chunkshapes(chunks):
    chunkshapes = []
    for i in range(len(chunks[0])):
        for j in range(len(chunks[1])):
            for k in range(len(chunks[2])):
                shape = (chunks[0][i], chunks[1][j], chunks[2][k])
                chunkshapes.append(shape)
    return chunkshapes


def gen_new_shapes_chunksizes(src_coords,
                              chunkshapes,
                              array_locations,
                              src_crs,
                              dst_crs):
    new_shapes = []
    for i in range(len(chunkshapes)):
        empty_np = np.empty(chunkshapes[i])
        empty_da = xr.DataArray(empty_np, dims=('band', 'y', 'x'))
        block_coords = []
        for j in range(len(src_coords)):
            coord_name = src_coords[j][0]
            coord_arr = src_coords[j][1]
            if j == 0:
                block_coord = coord_arr
            else:
                array_location = array_locations[i][j]
                start = array_location[0]
                end = array_location[1]
                block_coord = coord_arr[start:end]
            block_coords.append([coord_name, block_coord])
            empty_da.coords[coord_name] = block_coord
        empty_da.rio.set_crs(src_crs, inplace=True)
        empty_da.rio.write_crs(src_crs, inplace=True)
        empty_da.rio.write_transform(inplace=True)
        reprojected = empty_da.rio.reproject(dst_crs)
        new_shape = reprojected.data.shape
        new_shapes.append(new_shape)
    new_chunksizes = list(zip(*new_shapes))
    return (new_shapes, new_chunksizes)


def gen_new_chunks(numblocks, chunksizes):
    new_chunks = []
    for i in range(len(numblocks)):
        num = numblocks[i]
        new_chunks.append(chunksizes[i][-num:])
    return new_chunks


def calc_chunksizes_rio(src_shape,
                        src_coords,
                        chunks,
                        numblocks,
                        src_crs,
                        dst_crs):
    chunk_locations = gen_chunk_locations(numblocks)
    shapes = list_chunk_shapes(chunks, chunk_locations)
    array_locations = gen_array_locations(chunk_locations, shapes)
    chunkshapes = gen_chunkshapes(chunks)
    new_shapes, new_chunksizes = gen_new_shapes_chunksizes(src_coords,
                                                           chunkshapes,
                                                           array_locations,
                                                           src_crs,
                                                           dst_crs)
    new_chunks = gen_new_chunks(numblocks, new_chunksizes)
    return new_chunks


def calc_chunksizes_rasterio(src_crs,
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


def _block_reproject(data,
                     src_coords,
                     src_crs,
                     dst_crs,
                     block_info=None,
                     **kwargs):
    if block_info is not None:
        data_arr = xr.DataArray(data, dims=('band', 'y', 'x'))
        in_arr_loc = block_info[0]['array-location']
        block_coords = []
        for i in range(len(in_arr_loc)):
            dim_start = in_arr_loc[i][0]
            dim_end = in_arr_loc[i][1]
            dim_coord_name = src_coords[i][0]
            dim_coord = src_coords[i][1]
            block_dim_coord = dim_coord[dim_start:dim_end]
            block_coords.append((dim_coord_name, block_dim_coord))
        data_arr = add_coords_crs_on_arr(data_arr,
                                         block_coords,
                                         src_crs)
        dst_arr_loc = block_info[None]['array-location']
        dst_shape = []
        for i in range(len(dst_arr_loc)):
            dim_len = dst_arr_loc[i][1] - dst_arr_loc[i][0]
            dst_shape.append(dim_len)
        _, dst_height, dst_width = dst_shape
        shape = (dst_height, dst_width)
        reprojected_block_da = data_arr.rio.reproject(dst_crs,
                                                      shape=shape,
                                                      **kwargs)
        reprojected_block = reprojected_block_da.data
        return reprojected_block


def _cupy_reproject():
    raise NotImplementedError('cupy is not supported yet; '
                              'please use numpy or dask')


def _dask_cupy_reproject():
    raise NotImplementedError('dask cupy not implemented yet; '
                              'please use numpy or dask')


def _numpy_reproject(arr, dst_crs, **kwargs):
    return arr.rio.reproject(dst_crs, **kwargs)


def _dask_reproject(arr, dst_crs, **kwargs):
    data = arr.data.astype(np.uint16)
    src_coords = gen_coords_list(arr)
    src_crs = arr.rio.crs
    src_shape = data.shape
    src_bounds = arr.rio.bounds()
    src_chunks = data.chunks  # noqa: F841
    numblocks = data.numblocks

#     chunks = calc_chunksizes_rio(src_shape,
#                                  src_coords,
#                                  src_chunks,
#                                  numblocks,
#                                  src_crs,
#                                  dst_crs)
# Using this means dst_shape is necessary in rio.reproject
    chunks = calc_chunksizes_rasterio(src_crs,
                                      dst_crs,
                                      src_shape,
                                      src_bounds,
                                      numblocks)
    reprojected_data = da.map_blocks(_block_reproject,
                                     arr.data,
                                     src_coords,
                                     src_crs,
                                     dst_crs,
                                     chunks=chunks,
                                     dtype=np.uint16,
                                     **kwargs)
    reprojected_data = np.rot90(np.rot90(reprojected_data))
    return reprojected_data


def reproject(arr: xr.DataArray, dst_crs, **kwargs):
    mapper = ArrayTypeFunctionMapping(numpy_func=_numpy_reproject,
                                      cupy_func=_cupy_reproject,
                                      dask_func=_dask_reproject,
                                      dask_cupy_func=_dask_cupy_reproject)
    reprojected = mapper(arr)(arr, dst_crs, **kwargs)
    return reprojected
