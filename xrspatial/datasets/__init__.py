import os

import dask.array as da
import datashader as ds
import noise
import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    'available',
    'get_data',
    'make_terrain',
]

_module_path = os.path.dirname(os.path.abspath(__file__))
_available_datasets = [p for p in next(os.walk(_module_path))[1]
                       if not p.startswith("__")]
available_datasets = _available_datasets


def get_data(dataset):
    """
    Open example multispectral band data.
    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``xrspatial.datasets.available`` for
        all options.
    Examples
    --------
    >>>     xrspatial.datasets.get_data("sentinel-2")
    """
    data = {}
    if dataset in _available_datasets:
        folder_path = os.path.abspath(os.path.join(_module_path, dataset))
        band_files = [p for p in next(os.walk(folder_path))[2]]
        for band_file in band_files:
            array = xr.open_dataarray(os.path.join(folder_path, band_file))
            data[array.Name] = array
    else:
        msg = f'The dataset {dataset} is not available. '
        msg += f'Available folders are {available_datasets}.'
        raise ValueError(msg)
    return data


def make_terrain(
    shape=(1024, 1024),
    scale=100.0,
    octaves=6,
    persistence=0.5,
    lacunarity=2.0,
    chunks=(512, 512)
):
    """
    Generate a pseudo-random terrain data dask array.

    Parameters
    ----------
    shape : int or tuple of int, default=(1024, 1024)
        Output array shape.
    scale : float, default=100.0
        Noise factor scale.
    octaves : int, default=6
        Number of waves when generating the noise.
    persistence : float, default=0.5
        Amplitude of each successive octave relative.
    lacunarity : float, default=2.0
        Frequency of each successive octave relative.
    chunks : int or tuple of int, default=(512, 512)
        Number of samples on each block.

    Returns
    -------
    terrain : xarray.DataArray
        2D array of generated terrain values.
    """
    def _func(arr, block_id=None):
        block_ystart = block_id[0] * arr.shape[0]
        block_xstart = block_id[1] * arr.shape[1]
        out = np.zeros(arr.shape)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i][j] = noise.pnoise2(
                    (block_ystart + i)/scale,
                    (block_xstart + j)/scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    base=42,
                )
        return out

    data = (
        da.zeros(shape=shape, chunks=chunks, dtype=np.float32)
        .map_blocks(_func, dtype=np.float32)
    )

    cvs = ds.Canvas(
        x_range=(0, 500),
        y_range=(0, 500),
        plot_width=shape[1],
        plot_height=shape[0],
    )

    hack_agg = cvs.points(pd.DataFrame({'x': [], 'y': []}), 'x', 'y')

    agg = xr.DataArray(
        data,
        name='terrain',
        coords=hack_agg.coords,
        dims=hack_agg.dims,
        attrs={'res': 1},
    )

    return agg
