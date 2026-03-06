"""Memory-safe raster preview via block-averaged downsampling."""

from xrspatial.dataset_support import supports_dataset
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
)


@supports_dataset
def preview(agg, width=1000, height=None, name='preview'):
    """Downsample a raster to target pixel dimensions.

    For dask-backed arrays, the operation is lazy: each chunk is reduced
    independently, so peak memory is bounded by the largest chunk plus
    the small output array.  A 30 TB raster can be previewed at
    1000x1000 with only a few MB of RAM.

    Parameters
    ----------
    agg : xr.DataArray
        Input raster (2D).
    width : int, default 1000
        Target width in pixels.
    height : int, optional
        Target height in pixels.  If not provided, computed from *width*
        preserving the aspect ratio of *agg*.
    name : str, default 'preview'
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        Downsampled raster with updated coordinates.
    """
    _validate_raster(agg, func_name='preview', ndim=2)

    h = agg.sizes[agg.dims[0]]
    w = agg.sizes[agg.dims[1]]

    if height is None:
        height = max(1, round(width * h / w))

    factor_y = max(1, h // height)
    factor_x = max(1, w // width)

    if factor_y <= 1 and factor_x <= 1:
        return agg

    y_dim = agg.dims[0]
    x_dim = agg.dims[1]

    # CuPy arrays (non-dask): use stride-based subsampling.
    # xarray's coarsen may not support all cupy reduction ops.
    if has_cuda_and_cupy() and is_cupy_array(agg.data):
        result = agg.isel(
            {y_dim: slice(None, None, factor_y),
             x_dim: slice(None, None, factor_x)}
        )
    else:
        # numpy, dask+numpy, dask+cupy: coarsen with block averaging.
        # For dask arrays this builds a lazy graph -- no data is loaded
        # until .compute() is called.
        result = agg.coarsen(
            {y_dim: factor_y, x_dim: factor_x}, boundary='trim'
        ).mean()

    result.name = name
    return result
