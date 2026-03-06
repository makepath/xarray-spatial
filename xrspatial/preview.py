"""Memory-safe raster preview via downsampling."""

import numpy as np
import xarray as xr

from xrspatial.dataset_support import supports_dataset
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
)

_METHODS = ('mean', 'nearest', 'bilinear')


def _bilinear_numpy(data, out_h, out_w):
    """Bilinear interpolation on a 2D numpy array."""
    from scipy.ndimage import zoom

    zoom_y = out_h / data.shape[0]
    zoom_x = out_w / data.shape[1]
    return zoom(data, (zoom_y, zoom_x), order=1)


def _bilinear_cupy(data, out_h, out_w):
    """Bilinear interpolation on a 2D cupy array."""
    import cupy
    from cupyx.scipy.ndimage import zoom

    zoom_y = out_h / data.shape[0]
    zoom_x = out_w / data.shape[1]
    return zoom(data, (zoom_y, zoom_x), order=1)


@supports_dataset
def preview(agg, width=1000, height=None, method='mean', name='preview'):
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
    method : str, default 'mean'
        Downsampling method.  One of:

        - ``'mean'``: block averaging via ``xarray.coarsen``.
        - ``'nearest'``: stride-based subsampling (fastest, no smoothing).
        - ``'bilinear'``: bilinear interpolation via ``scipy.ndimage.zoom``.
    name : str, default 'preview'
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        Downsampled raster with updated coordinates.
    """
    _validate_raster(agg, func_name='preview', ndim=2)

    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {_METHODS!r}, got {method!r}"
        )

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

    out_h = h // factor_y
    out_w = w // factor_x

    if method == 'nearest':
        result = agg.isel(
            {y_dim: slice(None, None, factor_y),
             x_dim: slice(None, None, factor_x)}
        )
    elif method == 'bilinear':
        result = _preview_bilinear(agg, out_h, out_w, y_dim, x_dim)
    else:
        # method == 'mean'
        if has_cuda_and_cupy() and is_cupy_array(agg.data):
            # xarray coarsen has edge cases with cupy; fall back to nearest
            result = agg.isel(
                {y_dim: slice(None, None, factor_y),
                 x_dim: slice(None, None, factor_x)}
            )
        else:
            result = agg.coarsen(
                {y_dim: factor_y, x_dim: factor_x}, boundary='trim'
            ).mean()

    result.name = name
    return result


def _preview_bilinear(agg, out_h, out_w, y_dim, x_dim):
    """Apply bilinear interpolation, handling numpy/cupy/dask backends."""
    import dask.array as da

    if isinstance(agg.data, da.Array):
        # For dask: use map_blocks with a wrapper that resizes each block,
        # then concatenate. Simpler approach: compute target coords and
        # use xarray interp (which handles dask natively).
        y_coords = agg.coords[y_dim]
        x_coords = agg.coords[x_dim]
        new_y = np.linspace(
            float(y_coords[0]), float(y_coords[-1]), out_h
        )
        new_x = np.linspace(
            float(x_coords[0]), float(x_coords[-1]), out_w
        )
        result = agg.interp(
            {y_dim: new_y, x_dim: new_x}, method='linear'
        )
    elif has_cuda_and_cupy() and is_cupy_array(agg.data):
        out_data = _bilinear_cupy(agg.data, out_h, out_w)
        y_coords = agg.coords[y_dim].values
        x_coords = agg.coords[x_dim].values
        new_y = np.linspace(y_coords[0], y_coords[-1], out_h)
        new_x = np.linspace(x_coords[0], x_coords[-1], out_w)
        result = xr.DataArray(
            out_data,
            dims=[y_dim, x_dim],
            coords={y_dim: new_y, x_dim: new_x},
            attrs=agg.attrs,
        )
    else:
        out_data = _bilinear_numpy(agg.data, out_h, out_w)
        y_coords = agg.coords[y_dim].values
        x_coords = agg.coords[x_dim].values
        new_y = np.linspace(y_coords[0], y_coords[-1], out_h)
        new_x = np.linspace(x_coords[0], x_coords[-1], out_w)
        result = xr.DataArray(
            out_data,
            dims=[y_dim, x_dim],
            coords={y_dim: new_y, x_dim: new_x},
            attrs=agg.attrs,
        )
    return result
