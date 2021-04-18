from math import sqrt
import numpy as np
import numba as nb

from numba import cuda

from PIL import Image

import xarray as xr
from xarray import DataArray
import dask.array as da

from xrspatial.utils import cuda_args
from xrspatial.utils import ngjit
from xrspatial.utils import ArrayTypeFunctionMapping
from xrspatial.utils import validate_arrays

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False


@ngjit
def _arvi_cpu(nir_data, red_data, blue_data):
    out = np.zeros(nir_data.shape, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            blue = blue_data[y, x]
            numerator = (nir - (2.0 * red) + blue)
            denominator = (nir + (2.0 * red) + blue)
            if denominator != 0.0:
                out[y, x] = numerator / denominator

    return out


@cuda.jit
def _arvi_gpu(nir_data, red_data, blue_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        red = red_data[y, x]
        blue = blue_data[y, x]
        numerator = (nir - (2.0 * red) + blue)
        denominator = (nir + (2.0 * red) + blue)
        if denominator != 0.0:
            out[y, x] = numerator / denominator


def _arvi_dask(nir_data, red_data, blue_data):
    out = da.map_blocks(_arvi_cpu, nir_data, red_data, blue_data,
                        meta=np.array(()))
    return out


def _arvi_cupy(nir_data, red_data, blue_data):
    griddim, blockdim = cuda_args(nir_data.shape)
    out = cupy.empty(nir_data.shape, dtype='f4')
    out[:] = cupy.nan
    _arvi_gpu[griddim, blockdim](nir_data, red_data, blue_data, out)
    return out

def _arvi_dask_cupy(nir_data, red_data, blue_data):
    out = da.map_blocks(_arvi_cupy, nir_data, red_data, blue_data,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


def arvi(nir_agg: xr.DataArray,
         red_agg: xr.DataArray,
         blue_agg: xr.DataArray,
         name = 'arvi'):
    """
    Computes Atmospherically Resistant Vegetation Index. Allows for
    molecular and ozone correction with no further need for aerosol
    correction, except for dust conditions.

    Parameters
    ----------
    nir_agg : xarray.DataArray
        2D array of near-infrared band data.
    red_agg : xarray.DataArray
        2D array of red band data.
    blue_agg : xarray.DataArray
        2D array of blue band data.
    name : str, default = "arvi"
        Name of output DataArray.
    Returns
    -------
    arvi_agg : xarray.DataArray of the same type as inputs.
        2D array arvi values. All other input attributes are preserved.

    Notes
    -----
    Algorithm References
        - https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=03667

    Example
    -------
    >>>     import datashader as ds
    >>>     import xarray as xr 
    >>>     from xrspatial import generate_terrain
    >>>     from xrspatial.datasets import get_data
    >>>     from xrspatial.multispectral import arvi
    >>>     from datashader.transfer_functions import shade, stack
    >>>     from datashader.colors import Elevation

    >>>     # Open Example Data
    >>>     data = get_data('sentinel-2')
    >>>     # NIR Band
    >>>     nir = data['NIR']
    >>>     print(nir[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1286., 1289., 1285.],
    ...            [1275., 1292., 1312.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     nir_img = shade(agg = nir, cmap = ['black', 'white'])
    >>>     nir_img

            .. image :: ./docs/source/_static/img/docstring/nir_example.png

    >>>     # Red Band
    >>>     red = data['Red']
    >>>     print(red[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1138., 1115., 1112.],
    ...            [1114., 1109., 1133.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
     ...        ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     04
    ...         Name:                     Red
    ...         Bandwidth (µm):           30
    ...         Nominal Wavelength (µm):  0.665
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     red_img = shade(agg = red, cmap = ['black', 'white'])
    >>>     red_img

            .. image :: ./docs/source/_static/img/docstring/red_example.png

    >>>     # Blue Band
    >>>     blue = data['Blue']
    >>>     print(blue[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1197., 1191., 1179.],
    ...            [1178., 1181., 1213.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     02
    ...         Name:                     Blue
    ...         Bandwidth (µm):           65
    ...         Nominal Wavelength (µm):  0.490
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     blue_img = shade(agg = blue, cmap = ['black', 'white'])
    >>>     blue_img

            .. image :: ./docs/source/_static/img/docstring/blue_example.png

    >>>     # Generate ARVI Aggregate Array
    >>>     arvi_agg = arvi(nir_agg = nir,
    >>>                     red_agg = red,
    >>>                     blue_agg = blue)
    >>>     print(arvi_agg[100:102, 200: 203])
    ...     <xarray.DataArray 'arvi' (y: 2, x: 3)>
    ...     array([[0.04349653, 0.05307856, 0.05119454],
    ...            [0.04806665, 0.05435941, 0.0540597 ]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     arvi_img = shade(arvi_agg, cmap = ['black', 'white'])
    >>>     arvi_img

            .. image :: ./docs/source/_static/img/docstring/arvi_example.png

    """

    validate_arrays(red_agg, nir_agg, blue_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_arvi_cpu,
                                      dask_func=_arvi_dask,
                                      cupy_func=_arvi_cupy,
                                      dask_cupy_func=_arvi_dask_cupy)

    out = mapper(red_agg)(nir_agg.data, red_agg.data, blue_agg.data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# EVI ----------
@ngjit
def _evi_cpu(nir_data, red_data, blue_data, c1, c2, soil_factor, gain):
    out = np.zeros(nir_data.shape, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            blue = blue_data[y, x]
            numerator = nir - red
            denominator = nir + c1 * red - c2 * blue + soil_factor
            if denominator != 0.0:
                out[y, x] = gain * (numerator / denominator)
    return out


@cuda.jit
def _evi_gpu(nir_data, red_data, blue_data, c1, c2, soil_factor, gain, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        red = red_data[y, x]
        blue = blue_data[y, x]
        numerator = nir - red
        denominator = nir + c1 * red - c2 * blue + soil_factor
        if denominator != 0.0:
            out[y, x] = gain * (numerator / denominator)


def _evi_dask(nir_data, red_data, blue_data, c1, c2, soil_factor, gain):
    out = da.map_blocks(_evi_cpu, nir_data, red_data, blue_data,
                        c1, c2, soil_factor, gain, meta=np.array(()))
    return out


def _evi_cupy(nir_data, red_data, blue_data, c1, c2, soil_factor, gain):
    griddim, blockdim = cuda_args(nir_data.shape)
    out = cupy.empty(nir_data.shape, dtype='f4')
    out[:] = cupy.nan
    args = (nir_data, red_data, blue_data, c1, c2, soil_factor, gain, out)
    _evi_gpu[griddim, blockdim](*args)
    return out


def _evi_dask_cupy(nir_data, red_data, blue_data, c1, c2, soil_factor, gain):
    out = da.map_blocks(_evi_cupy, nir_data, red_data, blue_data,
                        c1, c2, soil_factor, gain,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


def evi(nir_agg: xr.DataArray,
        red_agg: xr.DataArray,
        blue_agg: xr.DataArray,
        c1 = 6.0,
        c2 = 7.5,
        soil_factor = 1.0,
        gain = 2.5,
        name = 'evi'):
    """
    Computes Enhanced Vegetation Index. Allows for importved sensitivity
    in high biomass regions, de-coupling of the canopy background signal
    and reduction of atmospheric influences.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array of red band data.
    blue_agg : xr.DataArray
        2D array of blue band data.
    c1 : float, default = 6.0
        First coefficient of the aerosol resistance term.
    c2 : float, default = 7.5
        Second coefficients of the aerosol resistance term.
    soil_factor : float, default = 1.0
        Soil adjustment factor between -1.0 and 1.0.
    gain : float, default = 2.5
        Amplitude adjustment factor.
    name : str, default = "evi"
        Name of output DataArray.

    Returns
    -------
    evi_agg : xarray.DataArray of same type as inputs.
        2D array of evi values.
        All other input attributes are preserved.

    Notes
    -----
    Algorithm References
        - https://en.wikipedia.org/wiki/Enhanced_vegetation_index

    Example
    -------
    >>>     import datashader as ds
    >>>     import xarray as xr 
    >>>     from xrspatial import generate_terrain
    >>>     from xrspatial.datasets import get_data
    >>>     from xrspatial.multispectral import evi
    >>>     from datashader.transfer_functions import shade, stack
    >>>     from datashader.colors import Elevation

    >>>     # Open Example Data
    >>>     data = get_data('sentinel-2')
    >>>     # NIR Band
    >>>     nir = data['NIR']
    >>>     print(nir[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1286., 1289., 1285.],
    ...            [1275., 1292., 1312.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     nir_img = shade(agg = nir, cmap = ['black', 'white'])
    >>>     nir_img

            .. image :: ./docs/source/_static/img/docstring/nir_example.png

    >>>     # Red Band
    >>>     red = data['Red']
    >>>     print(red[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1138., 1115., 1112.],
    ...            [1114., 1109., 1133.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
     ...        ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     04
    ...         Name:                     Red
    ...         Bandwidth (µm):           30
    ...         Nominal Wavelength (µm):  0.665
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     red_img = shade(agg = red, cmap = ['black', 'white'])
    >>>     red_img

            .. image :: ./docs/source/_static/img/docstring/red_example.png

    >>>     # Blue Band
    >>>     blue = data['Blue']
    >>>     print(blue[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1197., 1191., 1179.],
    ...            [1178., 1181., 1213.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     02
    ...         Name:                     Blue
    ...         Bandwidth (µm):           65
    ...         Nominal Wavelength (µm):  0.490
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     blue_img = shade(agg = blue, cmap = ['black', 'white'])
    >>>     blue_img

            .. image :: ./docs/source/_static/img/docstring/blue_example.png

    >>>     # Generate EVI Aggregate Array
    >>>     evi_agg = evi(nir_agg = nir,
    >>>                   red_agg = red,
    >>>                   blue_agg = blue)
    >>>     print(evi_agg[100:102, 200: 203])
    ...     <xarray.DataArray 'evi' (y: 2, x: 3)>
    ...     array([[-0.42898551, -0.45669291, -0.48897682],
    ...            [-0.46      , -0.50247117, -0.45362392]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     evi_img = shade(evi_agg, cmap = ['black', 'white'])
    >>>     evi_img

            .. image :: ./docs/source/_static/img/docstring/evi_example.png

    """

    if not red_agg.shape == nir_agg.shape == blue_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    if not isinstance(c1, (float, int)):
        raise ValueError("c1 must be numeric")

    if not isinstance(c2, (float, int)):
        raise ValueError("c2 must be numeric")

    if soil_factor > 1.0 or soil_factor < -1.0:
        raise ValueError("soil factor must be between [-1.0, 1.0]")

    if gain < 0:
        raise ValueError("gain must be greater than 0")

    validate_arrays(nir_agg, red_agg, blue_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_evi_cpu,
                                      dask_func=_evi_dask,
                                      cupy_func=_evi_cupy,
                                      dask_cupy_func=_evi_dask_cupy)

    out = mapper(red_agg)(nir_agg.data, red_agg.data, blue_agg.data, c1, c2,
                          soil_factor, gain)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# GCI ----------
@ngjit
def _gci_cpu(nir_data, green_data):
    out = np.zeros(nir_data.shape, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            green = green_data[y, x]
            if green - 1 != 0:
                out[y, x] = nir / green - 1
    return out


@cuda.jit
def _gci_gpu(nir_data, green_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        green = green_data[y, x]
        if green - 1 != 0:
            out[y, x] = nir / green - 1


def _gci_dask(nir_data, green_data):
    out = da.map_blocks(_gci_cpu, nir_data, green_data, meta=np.array(()))
    return out


def _gci_cupy(nir_data, green_data):
    griddim, blockdim = cuda_args(nir_data.shape)
    out = cupy.empty(nir_data.shape, dtype='f4')
    out[:] = cupy.nan
    _gci_gpu[griddim, blockdim](nir_data, green_data, out)
    return out


def _gci_dask_cupy(nir_data, green_data):
    out = da.map_blocks(_gci_cupy, nir_data, green_data,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


def gci(nir_agg: xr.DataArray,
        green_agg: xr.DataArray,
        name = 'gci'):
    """
    Computes Green Chlorophyll Index. Used to estimate
    the content of leaf chorophyll and predict the
    physiological state of vegetation and plant health.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    green_agg : xr.DataArray
        2D array of green band data.
    name : str, default = "gci"
        Name of output DataArray.
    Returns
    -------
    gci_agg : xarray.DataArray of the same type as inputs.
        2D array of gci values.
        All other input attributes are preserved.

    Notes
    -----
    Algorithm References
        - https://en.wikipedia.org/wiki/Enhanced_vegetation_index

    Example
    -------
    >>>     import datashader as ds
    >>>     import xarray as xr 
    >>>     from xrspatial import generate_terrain
    >>>     from xrspatial.datasets import get_data
    >>>     from xrspatial.multispectral import gci
    >>>     from datashader.transfer_functions import shade, stack
    >>>     from datashader.colors import Elevation

    >>>     # Open Example Data
    >>>     data = get_data('sentinel-2')
    >>>     # NIR Band
    >>>     nir = data['NIR']
    >>>     print(nir[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1286., 1289., 1285.],
    ...            [1275., 1292., 1312.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     nir_img = shade(agg = nir, cmap = ['black', 'white'])
    >>>     nir_img

            .. image :: ./docs/source/_static/img/docstring/nir_example.png

    >>>     # Green Band
    >>>     green = data['Green']
    >>>     print(green[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1012., 1001., 1023.],
    ...            [ 990.,  992., 1009.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     03
    ...         Name:                     Green
    ...         Bandwidth (µm):           35
    ...         Nominal Wavelength (µm):  0.560
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     green_img = shade(agg = green, cmap = ['black', 'white'])
    >>>     green_img

            .. image :: ./docs/source/_static/img/docstring/green_example.png

    >>>     # Generate GCI Aggregate Array
    >>>     gci_agg = gci(nir_agg = nir,
    >>>                   green_agg = green)
    >>>     print(gci_agg[100:102, 200: 203])
    ...     <xarray.DataArray 'gci' (y: 2, x: 3)>
    ...     array([[0.27075099, 0.28771229, 0.25610948],
    ...            [0.28787879, 0.30241935, 0.30029732]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     gci_img = shade(gci_agg, cmap = ['black', 'white'])
    >>>     gci_img

            .. image :: ./docs/source/_static/img/docstring/gci_example.png

    """

    validate_arrays(nir_agg, green_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_gci_cpu,
                                      dask_func=_gci_dask,
                                      cupy_func=_gci_cupy,
                                      dask_cupy_func=_gci_dask_cupy)

    out = mapper(nir_agg)(nir_agg.data, green_agg.data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# NBR ----------
def nbr(nir_agg: xr.DataArray,
        swir2_agg: xr.DataArray,
        name = 'nbr'):
    """
    Computes Normalized Burn Ratio. Used to identify
    burned areas and provide a measure of burn severity.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band.
    swir_agg : xr.DataArray
        2D array of shortwave infrared band.
        (Landsat 4-7: Band 6)
        (Landsat 8: Band 7)
    name : str, default = "nbr"
        Name of output DataArray.
    Returns
    -------
    nbr_agg : xr.DataArray of the same type as inputs.
        2D array of nbr values.
        All other input attributes are preserved.

    Notes
    -----
    Algorithm References
        - https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio

    Example
    -------
    >>>     import datashader as ds
    >>>     import xarray as xr 
    >>>     from xrspatial import generate_terrain
    >>>     from xrspatial.datasets import get_data
    >>>     from xrspatial.multispectral import nbr
    >>>     from datashader.transfer_functions import shade, stack
    >>>     from datashader.colors import Elevation

    >>>     # Open Example Data
    >>>     data = get_data('sentinel-2')
    >>>     # NIR Band
    >>>     nir = data['NIR']
    >>>     print(nir[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1286., 1289., 1285.],
    ...            [1275., 1292., 1312.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 1
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     nir_img = shade(agg = nir, cmap = ['black', 'white'])
    >>>     nir_img

            .. image :: ./docs/source/_static/img/docstring/nir_example.png

    >>>     # SWIR2 Band
    >>>     swir2 = data['SWIR2']
    >>>     print(swir2[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1776., 1765., 1820.],
    ...            [1878., 1859., 1902.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.04e+05 6.04e+05 6.040e+05
    ...       * y        (y) float64 4.698e+06 4.698e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 2.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [20. 20.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     12
    ...         Name:                     SWIR2
    ...         Bandwidth (µm):           180
    ...         Nominal Wavelength (µm):  2.190
    ...         Resolution (m):            20

    >>>     # Shade Image
    >>>     swir2_img = shade(agg = swir2, cmap = ['black', 'white'])
    >>>     swir2_img

            .. image :: ./docs/source/_static/img/docstring/swir2_example.png

    >>>     # Generate NBR Aggregate Array
    >>>     nbr_agg = nbr(nir_agg = nir, 
    >>>                   swir2_agg = swir2)
    >>>     print(nbr_agg[100:102, 200: 203])

    ...     <xarray.DataArray 'nbr' (y: 2, x: 3)>
    ...     array([[-0.16002613, -0.15586117, -0.17230274],
    ...            [-0.19124643, -0.17994288, -0.18357187]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     nbr_img = shade(nbr_agg, cmap = ['black', 'white'])
    >>>     nbr_img

            .. image :: ./docs/source/_static/img/docstring/nbr_example.png

    """

    validate_arrays(nir_agg, swir2_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_normalized_ratio_cpu,
                                      dask_func=_run_normalized_ratio_dask,
                                      cupy_func=_run_normalized_ratio_cupy,
                                      dask_cupy_func=_run_normalized_ratio_dask_cupy)

    out = mapper(nir_agg)(nir_agg.data, swir2_agg.data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


def nbr2(swir1_agg: xr.DataArray,
         swir2_agg: xr.DataArray,
         name = 'nbr2'):
    """
    Computes Normalized Burn Ratio 2
    "NBR2 modifies the Normalized Burn Ratio (NBR)
    to highlight water sensitivity in vegetation and
    may be useful in post-fire recovery studies."
    https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio-2

    Parameters
    ----------
    swir1_agg : xr.DataArray
        2D array of near-infrared band data.
        shortwave infrared band
        (Landsat 4-7: Band 5)
        (Landsat 8: Band 6)
    swir2_agg : xr.DataArray
        2D array of shortwave infrared band data.
        (Landsat 4-7: Band 6)
        (Landsat 8: Band 7)
    name : str default = "nbr2"
        Name of output DataArray.
    Returns
    -------
    nbr2_agg : xr.DataArray of same type as inputs.
        2D array of nbr2 values.
        All other input attributes are preserved.

    Notes
    -----
    Algorithm References
        - https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio-2

    Example
    -------
    >>>     import datashader as ds
    >>>     import xarray as xr 
    >>>     from xrspatial import generate_terrain
    >>>     from xrspatial.multispectral import nbr2
    >>>     from datashader.transfer_functions import shade, stack
    >>>     from datashader.colors import Elevation
    >>>     from datasets import get_data

    >>>     # Open Example Data
    >>>     data = get_data('sentinel-2')
    >>>     # SWIR1 Band
    >>>     swir1 = data['SWIR1']
    >>>     print(swir1[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1963., 1968., 2010.],
    ...            [2054., 2041., 2089.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.04e+05 6.04e+05 6.040e+05
    ...       * y        (y) float64 4.698e+06 4.698e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 2.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [20. 20.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     11
    ...         Name:                     SWIR1
    ...         Bandwidth (µm):           90
    ...         Nominal Wavelength (µm):  1.610
    ...         Resolution (m):            20

    >>>     # Shade Image
    >>>     swir1_img = shade(agg = swir1, cmap = ['black', 'white'])
    >>>     swir1_img

            .. image :: ./docs/source/_static/img/docstring/swir1_example.png

    >>>     # SWIR2 Band
    >>>     swir2 = data['SWIR2']
    >>>     print(swir2[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1776., 1765., 1820.],
       [1878., 1859., 1902.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.04e+05 6.04e+05 6.040e+05
    ...       * y        (y) float64 4.698e+06 4.698e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 2.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [20. 20.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     12
    ...         Name:                     SWIR2
    ...         Bandwidth (µm):           180
    ...         Nominal Wavelength (µm):  2.190
    ...         Resolution (m):            20

    >>>     # Shade Image
    >>>     swir2_img = shade(agg = swir2, cmap = ['black', 'white'])
    >>>     swir2_img

            .. image :: ./docs/source/_static/img/docstring/swir2_example.png

    >>>     # Generate NBR2 Aggregate Array
    >>>     nbr2_agg = nbr2(swir1_agg = swir1, 
    >>>                     swir2_agg = swir2)
    >>>     print(nbr2_agg[100:102, 200: 203])
    ...     <xarray.DataArray 'nbr' (y: 2, x: 3)>
    ...     array([[0.05001337, 0.05437986, 0.04960836],
    ...            [0.04476094, 0.04666667, 0.04685542]])
    ...     Coordinates:
    ...       * x        (x) float64 6.04e+05 6.04e+05 6.040e+05
    ...       * y        (y) float64 4.698e+06 4.698e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 2.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [20. 20.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...        ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     11
    ...         Name:                     SWIR1
    ...         Bandwidth (µm):           90
    ...         Nominal Wavelength (µm):  1.610
    ...         Resolution (m):            20

    >>>     # Shade Image
    >>>     nbr2_img = shade(nbr2_agg, cmap = ['black', 'white'])
    >>>     nbr2_img

            .. image :: ./docs/source/_static/img/docstring/nbr2_example.png

    """

    validate_arrays(swir1_agg, swir2_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_normalized_ratio_cpu,
                                      dask_func=_run_normalized_ratio_dask,
                                      cupy_func=_run_normalized_ratio_cupy,
                                      dask_cupy_func=_run_normalized_ratio_dask_cupy)

    out = mapper(swir1_agg)(swir1_agg.data, swir2_agg.data)

    return DataArray(out,
                     name=name,
                     coords=swir1_agg.coords,
                     dims=swir1_agg.dims,
                     attrs=swir1_agg.attrs)


# NDVI ----------
def ndvi(nir_agg: xr.DataArray,
         red_agg: xr.DataArray,
         name = 'ndvi'):
    """
    Computes Normalized Difference Vegetation Index (NDVI).
    Used to determine if a cell contains live green vegetation.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array red band data.
    name : str default = "ndvi"
        Name of output DataArray.
    Returns
    -------
    ndvi_agg : xarray.DataArray of same type as inputs.
        2D array of ndvi values.
        All other input attributes are preserved.

    Notes
    -----
    Algorithm References
        - http://ceholden.github.io/open-geo-tutorial/python/chapter_2_indices.html

    Example
    -------
    >>>     import datashader as ds
    >>>     import xarray as xr 
    >>>     from xrspatial import generate_terrain
    >>>     from xrspatial.multispectral import ndvi
    >>>     from datashader.transfer_functions import shade, stack
    >>>     from datashader.colors import Elevation
    >>>     from datasets import get_data

    >>>     # Open Example Data
    >>>     data = get_data('sentinel-2')
    >>>     # NIR Band
    >>>     nir = data['NIR']
    >>>     print(nir[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1286., 1289., 1285.],
    ...            [1275., 1292., 1312.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     nir_img = shade(agg = nir, cmap = ['black', 'white'])
    >>>     nir_img

            .. image :: ./docs/source/_static/img/docstring/nir_example.png

    >>>     # Red Band
    >>>     red = data['Red']
    >>>     print(red[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1138., 1115., 1112.],
    ...            [1114., 1109., 1133.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     04
    ...         Name:                     Red
    ...         Bandwidth (µm):           30
    ...         Nominal Wavelength (µm):  0.665
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     red_img = shade(agg = red, cmap = ['black', 'white'])
    >>>     red_img

            .. image :: ./docs/source/_static/img/docstring/red_example.png

    >>>     # Generate NDVI Aggregate Array
    >>>     ndvi_agg = ndvi(nir_agg = nir, 
    >>>                     red_agg = red)
    >>>     print(ndvi_agg[100:102, 200: 203])
    ...     <xarray.DataArray 'ndvi' (y: 2, x: 3)>
    ...     array([[0.06105611, 0.07237937, 0.07217355],
    ...            [0.06739221, 0.07621824, 0.07321063]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     ndvi_img = shade(ndvi_agg, cmap = ['black', 'white'])
    >>>     ndvi_img

            .. image :: ./docs/source/_static/img/docstring/ndvi_example.png

    """

    validate_arrays(nir_agg, red_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_normalized_ratio_cpu,
                                      dask_func=_run_normalized_ratio_dask,
                                      cupy_func=_run_normalized_ratio_cupy,
                                      dask_cupy_func=_run_normalized_ratio_dask_cupy)

    out = mapper(nir_agg)(nir_agg.data, red_agg.data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# NDMI ----------
def ndmi(nir_agg: xr.DataArray,
         swir1_agg: xr.DataArray,
         name = 'ndmi'):
    """
    Computes Normalized Difference Moisture Index.
    Used to determine vegetation water content.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
        (Landsat 4-7: Band 4)
        (Landsat 8: Band 5)
    swir1_agg : xr.DataArray
        2D array of shortwave infrared band.
        (Landsat 4-7: Band 5)
        (Landsat 8: Band 6)
    name: str, default = "ndmi"
        Name of output DataArray.
    Returns
    -------
    ndmi_agg : xr.DataArray of same type as inputs.
        2D array of ndmi values.
        All other input attributes are preserved.

    Notes
    -----
    Algorithm References
        - https://www.usgs.gov/land-resources/nli/landsat/normalized-difference-moisture-index

    Example
    -------
    >>>     import datashader as ds
    >>>     import xarray as xr 
    >>>     from xrspatial import generate_terrain
    >>>     from xrspatial.multispectral import ndmi
    >>>     from datashader.transfer_functions import shade, stack
    >>>     from datashader.colors import Elevation
    >>>     from datasets import get_data

    >>>     # Open Example Data
    >>>     data = get_data('sentinel-2')
    >>>     # NIR Band
    >>>     nir = data['NIR']
    >>>     print(nir[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1286., 1289., 1285.],
    ...            [1275., 1292., 1312.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     nir_img = shade(agg = nir, cmap = ['black', 'white'])
    >>>     nir_img

            .. image :: ./docs/source/_static/img/docstring/nir_example.png

    >>>     # SWIR1 Band
    >>>     swir1 = data['SWIR1']
    >>>     print(swir1[100:102, 200: 203])
    ...     <xarray.DataArray (y: 2, x: 3)>
    ...     array([[1963., 1968., 2010.],
    ...            [2054., 2041., 2089.]])
    ...     Coordinates:
    ...       * x        (x) float64 6.04e+05 6.04e+05 6.040e+05
    ...       * y        (y) float64 4.698e+06 4.698e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 2.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [20. 20.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     11
    ...         Name:                     SWIR1
    ...         Bandwidth (µm):           90
    ...         Nominal Wavelength (µm):  1.610
    ...         Resolution (m):            20

    >>>     # Shade Image
    >>>     swir1_img = shade(agg = swir1, cmap = ['black', 'white'])
    >>>     swir1_img

            .. image :: ./docs/source/_static/img/docstring/swir1_example.png

    >>>     # Generate NDMI Aggregate Array
    >>>     ndmi_agg = ndmi(nir_agg = nir, 
    >>>                     swir1_agg = swir1)
    >>>     print(ndmi_agg[100:102, 200: 203])
    ...     <xarray.DataArray 'ndmi' (y: 2, x: 3)>
    ...     array([[-0.20837181, -0.20847406, -0.22003035],
    ...            [-0.23400421, -0.22472247, -0.22846222]])
    ...     Coordinates:
    ...       * x        (x) float64 6.02e+05 6.02e+05 6.02e+05
    ...       * y        (y) float64 4.699e+06 4.699e+06
    ...         band     int32 ...
    ...     Attributes: (12/13)
    ...         transform:                [ 1.00000e+01  0.00000e+00  6.00000e+05  0.0000...
    ...         crs:                      +init=epsg:32719
    ...         res:                      [10. 10.]
    ...         is_tiled:                 1
    ...         nodatavals:               nan
    ...         scales:                   1.0
    ...         ...                       ...
    ...         instrument:               Sentinel-2
    ...         Band:                     07
    ...         Name:                     NIR
    ...         Bandwidth (µm):           115
    ...         Nominal Wavelength (µm):  0.842
    ...         Resolution (m):            10

    >>>     # Shade Image
    >>>     ndmi_img = shade(ndmi_agg, cmap = ['black', 'white'])
    >>>     ndmi_img

            .. image :: ./docs/source/_static/img/docstring/ndmi_example.png

    """

    validate_arrays(nir_agg, swir1_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_normalized_ratio_cpu,
                                      dask_func=_run_normalized_ratio_dask,
                                      cupy_func=_run_normalized_ratio_cupy,
                                      dask_cupy_func=_run_normalized_ratio_dask_cupy)

    out = mapper(nir_agg)(nir_agg.data, swir1_agg.data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


@ngjit
def _normalized_ratio_cpu(arr1, arr2):
    out = np.zeros(arr1.shape, dtype=np.float32)
    rows, cols = arr1.shape
    for y in range(0, rows):
        for x in range(0, cols):
            val1 = arr1[y, x]
            val2 = arr2[y, x]
            numerator = val1 - val2
            denominator = val1 + val2

            if denominator == 0.0:
                continue
            else:
                out[y, x] = numerator / denominator

    return out


def _run_normalized_ratio_dask(arr1, arr2):
    out = da.map_blocks(_normalized_ratio_cpu, arr1, arr2,
                        meta=np.array(()))
    return out


@cuda.jit
def _normalized_ratio_gpu(arr1, arr2, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        val1 = arr1[y, x]
        val2 = arr2[y, x]
        numerator = val1 - val2
        denominator = val1 + val2
        if denominator != 0.0:
            out[y, x] = numerator / denominator


def _run_normalized_ratio_cupy(arr1, arr2):
    griddim, blockdim = cuda_args(arr1.shape)
    out = cupy.empty(arr1.shape, dtype='f4')
    out[:] = cupy.nan
    _normalized_ratio_gpu[griddim, blockdim](arr1, arr2, out)
    return out


def _run_normalized_ratio_dask_cupy(arr1, arr2):
    out = da.map_blocks(_run_normalized_ratio_cupy, arr1, arr2,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


@ngjit
def _savi_cpu(nir_data, red_data, soil_factor):
    out = np.zeros(nir_data.shape, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            numerator = nir - red
            soma = nir + red + soil_factor
            denominator = soma * (1.0 + soil_factor)
            if denominator != 0.0:
                out[y, x] = numerator / denominator

    return out


@cuda.jit
def _savi_gpu(nir_data, red_data, soil_factor, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        red = red_data[y, x]
        numerator = nir - red
        soma = nir + red + soil_factor
        denominator = soma * (nb.float32(1.0) + soil_factor)
        if denominator != 0.0:
            out[y, x] = numerator / denominator


def _savi_dask(nir_data, red_data, soil_factor):
    out = da.map_blocks(_savi_cpu, nir_data, red_data, soil_factor,
                        meta=np.array(()))
    return out


def _savi_cupy(nir_data, red_data, soil_factor):
    griddim, blockdim = cuda_args(nir_data.shape)
    out = cupy.empty(nir_data.shape, dtype='f4')
    out[:] = cupy.nan
    _savi_gpu[griddim, blockdim](nir_data, red_data, soil_factor, out)
    return out


def _savi_dask_cupy(nir_data, red_data, soil_factor):
    out = da.map_blocks(_savi_cupy, nir_data, red_data, soil_factor,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


# SAVI ----------
def savi(nir_agg: xr.DataArray,
         red_agg: xr.DataArray,
         soil_factor: float = 1.0,
         name: str = 'savi'):
    """
    Computes Soil Adjusted Vegetation Index (SAVI).
    Used to determine if a cell contains living
    vegetation while minimizing soil brightness.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array of red band data.
    soil_factor : float, default = 1.0
        soil adjustment factor between -1.0 and 1.0.
        When set to zero, savi will return the same as ndvi.
    name : str, default = "savi"
        Name of output DataArray.
    Returns
    -------
    savi_agg : xr.DataArray of same type as inputs.
        2D array of  savi values.
        All other input attributes are preserved.

    Notes
    -----
    Algorithm References
        - https://www.sciencedirect.com/science/article/abs/pii/003442578890106X

    Example
    -------
    >>>     # Imports
    >>>     import numpy as np
    >>>     import xarray as xr
    >>>     from xrspatial import savi

    >>>     # Create Sample Band Data
    >>>     np.random.seed(0)
    >>>     ir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>>     height, width = nir_agg.shape
    >>>     _lat = np.linspace(0, height - 1, height)
    >>>     _lon = np.linspace(0, width - 1, width)
    >>>     nir_agg["lat"] = _lat
    >>>     nir_agg["lon"] = _lon

    >>>     np.random.seed(1)
    >>>     red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>>     height, width = red_agg.shape
    >>>     _lat = np.linspace(0, height - 1, height)
    >>>     _lon = np.linspace(0, width - 1, width)
    >>>     red_agg["lat"] = _lat
    >>>     red_agg["lon"] = _lon

    >>>     print(nir_agg, red_agg)
            <xarray.DataArray (lat: 4, lon: 4)>
            array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
                   [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
                   [0.96366276, 0.38344152, 0.79172504, 0.52889492],
                   [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0
            <xarray.DataArray (lat: 4, lon: 4)>
            array([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01],
                   [1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01],
                   [3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01],
                   [2.04452250e-01, 8.78117436e-01, 2.73875932e-02, 6.70467510e-01]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0

    >>>     # Create SAVI DataArray
    >>>     savi_agg = savi(nir_agg, red_agg)
    >>>     print(savi_agg)
            <xarray.DataArray 'savi' (lat: 4, lon: 4)>
            array([[ 0.03352048, -0.00105422,  0.1879897 ,  0.06565303],
                   [ 0.0881613 ,  0.1592294 ,  0.07738627,  0.12206768],
                   [ 0.12008304, -0.04041476,  0.08424787, -0.03530183],
                   [ 0.10256501,  0.0084672 ,  0.01986868, -0.16594768]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0
    """

    validate_arrays(red_agg, nir_agg)

    if not -1.0 <= soil_factor <= 1.0:
        raise ValueError("soil factor must be between [-1.0, 1.0]")

    mapper = ArrayTypeFunctionMapping(numpy_func=_savi_cpu,
                                      dask_func=_savi_dask,
                                      cupy_func=_savi_cupy,
                                      dask_cupy_func=_savi_dask_cupy)

    out = mapper(red_agg)(nir_agg.data, red_agg.data, soil_factor)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# SIPI ----------
@ngjit
def _sipi_cpu(nir_data, red_data, blue_data):
    out = np.zeros(nir_data.shape, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            blue = blue_data[y, x]
            numerator = nir - blue
            denominator = nir - red
            if denominator != 0.0:
                out[y, x] = numerator / denominator
    return out


@cuda.jit
def _sipi_gpu(nir_data, red_data, blue_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        red = red_data[y, x]
        blue = blue_data[y, x]
        numerator = nir - blue
        denominator = nir - red
        if denominator != 0.0:
            out[y, x] = numerator / denominator


def _sipi_dask(nir_data, red_data, blue_data):
    out = da.map_blocks(_sipi_cpu, nir_data, red_data, blue_data,
                        meta=np.array(()))
    return out


def _sipi_cupy(nir_data, red_data, blue_data):
    griddim, blockdim = cuda_args(nir_data.shape)
    out = cupy.empty(nir_data.shape, dtype='f4')
    out[:] = cupy.nan
    _sipi_gpu[griddim, blockdim](nir_data, red_data, blue_data, out)
    return out


def _sipi_dask_cupy(nir_data, red_data, blue_data):
    out = da.map_blocks(_sipi_cupy, nir_data, red_data, blue_data,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


def sipi(nir_agg: xr.DataArray,
         red_agg: xr.DataArray,
         blue_agg: xr.DataArray,
         name = 'sipi'):
    """
    Computes Structure Insensitive Pigment Index which helpful
    in early disease detection in vegetation.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array of red band data.
    blue_agg : xr.DataArray
        2D array of blue band data.
    name: str, default = "sipi"
        Name of output DataArray.
    Returns
    -------
     sipi_agg : xr.DataArray of same type as inputs.
        2D array of sipi values.
        All other input attributes are preserved.

    Notes
    -----
    Algorithm References:
        - https://en.wikipedia.org/wiki/Enhanced_vegetation_index

    Example
    -------
    >>>     # Imports
    >>>     import numpy as np
    >>>     import xarray as xr
    >>>     from xrspatial import sipi

    >>>     # Create Sample Band Data
    >>>     np.random.seed(0)
    >>>     nir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>>     height, width = nir_agg.shape
    >>>     _lat = np.linspace(0, height - 1, height)
    >>>     _lon = np.linspace(0, width - 1, width)
    >>>     nir_agg["lat"] = _lat
    >>>     nir_agg["lon"] = _lon

    >>>     np.random.seed(1)
    >>>     red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>>     height, width = red_agg.shape
    >>>     _lat = np.linspace(0, height - 1, height)
    >>>     _lon = np.linspace(0, width - 1, width)
    >>>     red_agg["lat"] = _lat
    >>>     red_agg["lon"] = _lon

    >>>     np.random.seed(2)
    >>>     blue_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>>     height, width = blue_agg.shape
    >>>     _lat = np.linspace(0, height - 1, height)
    >>>     _lon = np.linspace(0, width - 1, width)
    >>>     blue_agg["lat"] = _lat
    >>>     blue_agg["lon"] = _lon

    >>>     print(nir_agg, red_agg, blue_agg)
            <xarray.DataArray (lat: 4, lon: 4)>
            array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
                   [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
                   [0.96366276, 0.38344152, 0.79172504, 0.52889492],
                   [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0
            <xarray.DataArray (lat: 4, lon: 4)>
            array([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01],
                   [1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01],
                   [3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01],
                   [2.04452250e-01, 8.78117436e-01, 2.73875932e-02, 6.70467510e-01]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0
            <xarray.DataArray (lat: 4, lon: 4)>
            array([[0.4359949 , 0.02592623, 0.54966248, 0.43532239],
                   [0.4203678 , 0.33033482, 0.20464863, 0.61927097],
                   [0.29965467, 0.26682728, 0.62113383, 0.52914209],
                   [0.13457995, 0.51357812, 0.18443987, 0.78533515]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0

    >>>     # Create ARVI DataArray
    >>>     sipi_agg = sipi(nir_agg, red_agg, blue_agg)
    >>>     print(sipi_agg)
            <xarray.DataArray 'sipi' (lat: 4, lon: 4)>
            array([[ 8.56038534e-01, -1.34225137e+02,  8.81124802e-02, 4.51702802e-01],
                   [ 1.18707483e-02,  5.70058976e-01,  9.26834671e-01, 4.98894015e-01],
                   [ 1.17130642e+00, -7.50533112e-01,  4.57925444e-01, 1.58116224e-03],
                   [ 1.19217212e+00,  8.67787369e+00, -2.59811674e+00, 1.19691430e+00]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0

    """

    validate_arrays(red_agg, nir_agg, blue_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_sipi_cpu,
                                      dask_func=_sipi_dask,
                                      cupy_func=_sipi_cupy,
                                      dask_cupy_func=_sipi_dask_cupy)

    out = mapper(red_agg)(nir_agg.data, red_agg.data, blue_agg.data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# EBBI ----------
@ngjit
def _ebbi_cpu(red_data, swir_data, tir_data):
    out = np.zeros(red_data.shape, dtype=np.float32)
    rows, cols = red_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            red = red_data[y, x]
            swir = swir_data[y, x]
            tir = tir_data[y, x]
            numerator = swir - red
            denominator = 10 * np.sqrt(swir + tir)
            if denominator != 0.0:
                out[y, x] = numerator / denominator
    return out


@cuda.jit
def _ebbi_gpu(red_data, swir_data, tir_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        red = red_data[y, x]
        swir = swir_data[y, x]
        tir = tir_data[y, x]
        numerator = swir - red
        denominator = nb.int64(10) * sqrt(swir + tir)
        if denominator != 0.0:
            out[y, x] = numerator / denominator


def _ebbi_dask(red_data, swir_data, tir_data):
    out = da.map_blocks(_ebbi_cpu, red_data, swir_data, tir_data,
                        meta=np.array(()))
    return out


def _ebbi_cupy(red_data, swir_data, tir_data):
    griddim, blockdim = cuda_args(red_data.shape)
    out = cupy.empty(red_data.shape, dtype='f4')
    out[:] = cupy.nan
    _ebbi_gpu[griddim, blockdim](red_data, swir_data, tir_data, out)
    return out


def _ebbi_dask_cupy(red_data, swir_data, tir_data):
    out = da.map_blocks(_ebbi_cupy, red_data, swir_data, tir_data,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


def ebbi(red_agg: xr.DataArray,
         swir_agg: xr.DataArray,
         tir_agg: xr.DataArray,
         name = 'ebbi'):
    """
    Computes Enhanced Built-Up and Bareness Index (EBBI) which
    allows for easily distinguishing between built-up and bare land areas.

    Parameters
    ----------
    red_agg : xr.DataArray
        2D array of red band data.
    swir_agg : xr.DataArray
        2D array of shortwave infrared band data.
    tir_agg: xr.DataArray
        2D array of thermal infrared band data.
    name: str, default = "ebbi"
        Name of output DataArray.

    Returns
    -------
    ebbi_agg = xr.DataArray of same type as inputs.
        2D array of ebbi values.
        All other input attributes are preserved

    Notes
    -----
    Algorithm References
        - https://rdrr.io/cran/LSRS/man/EBBI.html

    Example
    -------
    >>>     # Imports
    >>>     import numpy as np
    >>>     import xarray as xr
    >>>     from xrspatial import ebbi

    >>>     # Create Sample Band Data
    >>>     np.random.seed(1)
    >>>     red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>>     height, width = red_agg.shape
    >>>     _lat = np.linspace(0, height - 1, height)
    >>>     _lon = np.linspace(0, width - 1, width)
    >>>     red_agg["lat"] = _lat
    >>>     red_agg["lon"] = _lon

    >>>     np.random.seed(5)
    >>>     swir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>>     height, width = swir_agg.shape
    >>>     _lat = np.linspace(0, height - 1, height)
    >>>     _lon = np.linspace(0, width - 1, width)
    >>>     swir_agg["lat"] = _lat
    >>>     swir_agg["lon"] = _lon

    >>>     np.random.seed(6)
    >>>     tir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>>     height, width = tir_agg.shape
    >>>     _lat = np.linspace(0, height - 1, height)
    >>>     _lon = np.linspace(0, width - 1, width)
    >>>     tir_agg["lat"] = _lat
    >>>     tir_agg["lon"] = _lon

    >>>     print(red_agg, swir_agg, tir_agg)
            <xarray.DataArray (lat: 4, lon: 4)>
            array([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01],
                   [1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01],
                   [3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01],
                   [2.04452250e-01, 8.78117436e-01, 2.73875932e-02, 6.70467510e-01]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0
            <xarray.DataArray (lat: 4, lon: 4)>
            array([[0.22199317, 0.87073231, 0.20671916, 0.91861091],
                   [0.48841119, 0.61174386, 0.76590786, 0.51841799],
                   [0.2968005 , 0.18772123, 0.08074127, 0.7384403 ],
                   [0.44130922, 0.15830987, 0.87993703, 0.27408646]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0
            <xarray.DataArray (lat: 4, lon: 4)>
            array([[0.89286015, 0.33197981, 0.82122912, 0.04169663],
                   [0.10765668, 0.59505206, 0.52981736, 0.41880743],
                   [0.33540785, 0.62251943, 0.43814143, 0.73588211],
                   [0.51803641, 0.5788586 , 0.6453551 , 0.99022427]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0

    >>>     # Create EBBI DataArray
    >>>     ebbi_agg = ebbi(red_agg, swir_agg, tir_agg)
    >>>     print(ebbi_agg)
            <xarray.DataArray 'ebbi' (lat: 4, lon: 4)>
            array([[-2.43983486, -2.58194492,  3.97432599, -0.42291921],
                   [-0.11444052,  0.96786363,  0.59269999,  0.42374096],
                   [ 0.61379897, -0.23840436, -0.05598088,  0.95193251],
                   [ 1.32393891,  0.41574839,  0.72484653, -0.80669034]])
            Coordinates:
              * lat      (lat) float64 0.0 1.0 2.0 3.0
              * lon      (lon) float64 0.0 1.0 2.0 3.0
    """

    validate_arrays(red_agg, swir_agg, tir_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_ebbi_cpu,
                                      dask_func=_ebbi_dask,
                                      cupy_func=_ebbi_cupy,
                                      dask_cupy_func=_ebbi_dask_cupy)

    out = mapper(red_agg)(red_agg.data, swir_agg.data, tir_agg.data)

    return DataArray(out,
                     name=name,
                     coords=red_agg.coords,
                     dims=red_agg.dims,
                     attrs=red_agg.attrs)


@ngjit
def _normalize_data(agg, pixel_max=255.0):
    out = np.zeros_like(agg)
    min_val = 0
    max_val = 2 ** 16 - 1
    range_val = max_val - min_val
    rows, cols = agg.shape
    c = 40
    th = .125
    # check range_val to avoid dividing by zero
    if range_val != 0:
        for y in range(rows):
            for x in range(cols):
                val = agg[y, x]
                norm = (val - min_val) / range_val

                # sigmoid contrast enhancement
                norm = 1 / (1 + np.exp(c * (th - norm)))
                out[y, x] = norm * pixel_max
    return out


def true_color(r, g, b, nodata=1):
    h, w = r.shape

    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[:, :, 0] = (_normalize_data(r.data)).astype(np.uint8)
    data[:, :, 1] = (_normalize_data(g.data)).astype(np.uint8)
    data[:, :, 2] = (_normalize_data(b.data)).astype(np.uint8)

    a = np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)
    data[:, :, 3] = a.astype(np.uint8)

    return Image.fromarray(data, 'RGBA')
