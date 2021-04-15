from math import sqrt
import numpy as np
import numba as nb

from numba import cuda

from PIL import Image

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


def arvi(nir_agg: DataArray, red_agg: DataArray, blue_agg: DataArray,
         name='arvi'):
    """
    Computes Atmospherically Resistant Vegetation Index.
    Allows for molecular and ozone correction with no further
    need for aerosol correction, except for dust conditions.

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data
        (Sentinel 2: Band 8)
    red_agg : DataArray
        red band data
        (Sentinel 2: Band 4)
    blue_agg : DataArray
        blue band data
        (Sentinel 2: Band 2)
    name: str, optional (default = "arvi")
        Name of output DataArray

    Returns
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated arvi values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
    https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=03667

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

        Create Sample Band Data
    >>> np.random.seed(0)
    >>> nir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    >>> np.random.seed(1)
    >>> red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = red_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> red_agg["lat"] = _lat
    >>> red_agg["lon"] = _lon

    >>> np.random.seed(2)
    >>> blue_agg = xr.DataArray(np.random.rand(4,4),
    >>> dims = ["lat", "lon"])
    >>> height, width = blue_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> blue_agg["lat"] = _lat
    >>> blue_agg["lon"] = _lon

    >>> print(nir_agg, red_agg, blue_agg)
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

    Create ARVI DataArray
    >>> data = xrspatial.multispectral.arvi(nir_agg, red_agg, blue_agg)
    >>> print(data)
    <xarray.DataArray 'arvi' (lat: 4, lon: 4)>
    array([[ 0.08288985, -0.32062735,  0.99960309,  0.23695335],
           [ 0.48395093,  0.68183958,  0.26579331,  0.37232558],
           [ 0.22839874, -0.24733151,  0.2551784 , -0.12864117],
           [ 0.26424862, -0.09922362,  0.64689773, -0.21165207]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
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


def evi(nir_agg: DataArray, red_agg: DataArray, blue_agg: DataArray,
        c1=6.0, c2=7.5, soil_factor=1.0, gain=2.5, name='evi'):
    """
    Computes Enhanced Vegetation Index. Allows for importved
    sensitivity in high biomass regions, de-coupling of the
    canopy background signal and reduction of atmospheric influences.

    Parameters:
    ----------
    nir_agg: xarray.DataArray
        2D array of near-infrared band data.
        (Sentinel 2: Band 8)
    red_agg: xarray.DataArray
        2D array of red band data.
        (Sentinel 2: Band 4)
    blue_agg: xarray.DataArray
        2D array of blue band data.
        (Sentinel 2: Band 2)
    c1: float (default = 6.0)
        First coefficient of the aerosol resistance term.
    c2: float (default = 7.5)
        Second coefficients of the aerosol resistance term.
    soil_factor: float (default = 1.0)
        Soil adjustment factor between -1.0 and 1.0.
    gain: float (default = 2.5)
        Amplitude adjustment factor.
    name: str, optional (default = "evi")
        Name of output DataArray.

    Returns
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated evi values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
    https://en.wikipedia.org/wiki/Enhanced_vegetation_index

        Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Sample Band Data
    >>> np.random.seed(0)
    >>> nir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    >>> np.random.seed(1)
    >>> red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = red_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> red_agg["lat"] = _lat
    >>> red_agg["lon"] = _lon

    >>> np.random.seed(2)
    >>> blue_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = blue_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> blue_agg["lat"] = _lat
    >>> blue_agg["lon"] = _lon

    >>> print(nir_agg, red_agg, blue_agg)
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

    Create EVI DataArray
    >>> data = xrspatial.multispectral.evi(nir_agg, red_agg, blue_agg)
    >>> print(data)
    <xarray.DataArray 'evi' (lat: 4, lon: 4)>
    array([[ 4.21876564e-01, -2.19724452e-03, -5.98098914e-01,
         6.45351400e+00],
       [-8.15782552e-01, -4.98545103e+00,  6.15826250e-01,
        -2.00992194e+00],
       [ 6.75886740e-01, -1.48534469e-01, -2.64873586e+00,
        -2.33788375e-01],
       [ 5.09116426e-01,  3.55121123e-02, -7.37617269e-01,
         1.86948381e+00]])
    Coordinates:
    * lat      (lat) float64 0.0 1.0 2.0 3.0
    * lon      (lon) float64 0.0 1.0 2.0 3.0

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


def gci(nir_agg: DataArray, green_agg: DataArray, name='gci'):
    """
    Computes Green Chlorophyll Index. Used to estimate
    the content of leaf chorophyll and predict the
    physiological state of vegetation and plant health.

    Parameters:
    ----------
    nir_agg: xarray.DataArray
        2D array of near-infrared band data.
        (Sentinel 2: Band 8)
    green_agg: xarray.DataArray
        2D array of green band data.
        (Sentinel 2: Band 3)
    name: str, optional (default = "gci")
        Name of output DataArray

    Returns:
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated gci values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
        https://en.wikipedia.org/wiki/Enhanced_vegetation_index

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Sample Band Data
    >>> np.random.seed(0)
    >>> nir_agg = xr.DataArray(np.random.rand(4,4),
    >>> dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    >>> np.random.seed(3)
    >>> green_agg = xr.DataArray(np.random.rand(4,4),
    >>> dims = ["lat", "lon"])
    >>> height, width = green_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> green_agg["lat"] = _lat
    >>> green_agg["lon"] = _lon

    >>> print(nir_agg, green_agg)
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
           [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
           [0.96366276, 0.38344152, 0.79172504, 0.52889492],
           [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
     <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.5507979 , 0.70814782, 0.29090474, 0.51082761],
           [0.89294695, 0.89629309, 0.12558531, 0.20724288],
           [0.0514672 , 0.44080984, 0.02987621, 0.45683322],
           [0.64914405, 0.27848728, 0.6762549 , 0.59086282]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0

    Create GCI DataArray
    >>> data = xrspatial.multispectral.gci(nir_agg, green_agg)
    >>> print(data)
    <xarray.DataArray 'gci' (lat: 4, lon: 4)>
    array([[-3.60277089e-03,  9.94360715e-03,  1.07203010e+00,
             6.66674578e-02],
           [-5.25554349e-01, -2.79371758e-01,  2.48438213e+00,
             3.30303328e+00],
           [ 1.77238221e+01, -1.30143021e-01,  2.55001824e+01,
             1.57741801e-01],
           [-1.24932959e-01,  2.32365855e+00, -8.94956683e-01,
            -8.52538868e-01]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
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
def nbr(nir_agg: DataArray, swir2_agg: DataArray, name='nbr'):
    """
    Computes Normalized Burn Ratio. Used to identify
    burned areas and provide a measure of burn severity.

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band
        (Sentinel 2: Band 8)
    swir2_agg : DataArray
        shortwave infrared band
        (Sentinel 2: Band 12)
        (Landsat 4-7: Band 6)
        (Landsat 8: Band 7)
    name: str, optional (default = "nbr")
        Name of output DataArray

    Returns
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated nbr values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
    https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial
    Create Sample Band Data
    >>> np.random.seed(0)
    >>> nir_agg = xr.DataArray(np.random.rand(4,4),
    >>> dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    >>> np.random.seed(4)
    >>> swir2_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = swir2_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> swir2_agg["lat"] = _lat
    >>> swir2_agg["lon"] = _lon

    >>> print(nir_agg, swir2_agg)
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
           [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
           [0.96366276, 0.38344152, 0.79172504, 0.52889492],
           [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
     <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.96702984, 0.54723225, 0.97268436, 0.71481599],
           [0.69772882, 0.2160895 , 0.97627445, 0.00623026],
           [0.25298236, 0.43479153, 0.77938292, 0.19768507],
           [0.86299324, 0.98340068, 0.16384224, 0.59733394]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
        Create NBR DataArray
    >>> data = xrspatial.multispectral.nbr(nir_agg, swir2_agg)
    >>> print(data)
    <xarray.DataArray 'nbr' (lat: 4, lon: 4)>
    array([[-0.2758968 ,  0.1330436 , -0.23480372, -0.13489952],
           [-0.24440702,  0.49862273, -0.38100421,  0.9861242 ],
           [ 0.58413122, -0.0627572 ,  0.00785568,  0.45584774],
           [-0.20610824, -0.03027979, -0.39512455, -0.74540839]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
    """

    validate_arrays(nir_agg, swir2_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(nir_agg)(nir_agg.data, swir2_agg.data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


def nbr2(swir1_agg: DataArray, swir2_agg: DataArray, name='nbr2'):
    """
    Computes Normalized Burn Ratio 2
    "NBR2 modifies the Normalized Burn Ratio (NBR)
    to highlight water sensitivity in vegetation and
    may be useful in post-fire recovery studies."
    https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio-2
    Parameters
    ----------
    swir1_agg : DataArray
        near-infrared band
        shortwave infrared band
        (Sentinel 2: Band 11)
        (Landsat 4-7: Band 5)
        (Landsat 8: Band 6)
    swir2_agg : DataArray
        (Sentinel 2: Band 12)
        shortwave infrared band
        (Landsat 4-7: Band 6)
        (Landsat 8: Band 7)
    name: str, optional (default = "nbr2")
        Name of output DataArray

    Returns
    ----------
    data: DataArray

    Notes:
    ----------
    Algorithm References:
    https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio-2

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial
    Create Sample Band Data
    >>> np.random.seed(5)
    >>> swir1_agg = xr.DataArray(np.random.rand(4,4),
    >>>    dims = ["lat", "lon"])
    >>> height, width = swir1_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> swir1_agg["lat"] = _lat
    >>> swir1_agg["lon"] = _lon

    >>> np.random.seed(4)
    >>> swir2_agg = xr.DataArray(np.random.rand(4,4),
    >>>     dims = ["lat", "lon"])
    >>> height, width = swir2_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> swir2_agg["lat"] = _lat
    >>> swir2_agg["lon"] = _lon

    >>> print(swir1_agg, swir2_agg)
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.22199317, 0.87073231, 0.20671916, 0.91861091],
           [0.48841119, 0.61174386, 0.76590786, 0.51841799],
           [0.2968005 , 0.18772123, 0.08074127, 0.7384403 ],
           [0.44130922, 0.15830987, 0.87993703, 0.27408646]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.96702984, 0.54723225, 0.97268436, 0.71481599],
           [0.69772882, 0.2160895 , 0.97627445, 0.00623026],
           [0.25298236, 0.43479153, 0.77938292, 0.19768507],
           [0.86299324, 0.98340068, 0.16384224, 0.59733394]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
        Create NBR DataArray
    >>> data = xrspatial.multispectral.nbr2(swir1_agg, swir2_agg)
    >>> print(data)
    <xarray.DataArray 'nbr' (lat: 4, lon: 4)>
    array([[-0.62659567,  0.22814397, -0.64945135,  0.12476525],
           [-0.17646958,  0.47793963, -0.1207489 ,  0.97624978],
           [ 0.07970081, -0.39689195, -0.81225672,  0.57765256],
           [-0.32330232, -0.7226795 ,  0.6860596 , -0.37094321]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
    """

    validate_arrays(swir1_agg, swir2_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(swir1_agg)(swir1_agg.data, swir2_agg.data)

    return DataArray(out,
                     name=name,
                     coords=swir1_agg.coords,
                     dims=swir1_agg.dims,
                     attrs=swir1_agg.attrs)


# NDVI ----------
def ndvi(nir_agg: DataArray, red_agg: DataArray, name='ndvi'):
    """
    Computes Normalized Difference Vegetation Index (NDVI).
    Used to determine if a cell contains live green vegetation.

    Parameters:
    ----------
    nir_agg: xarray.DataArray
        2D array of near-infrared band data.
        (Sentinel 2: Band 8)
    red_agg: xarray.DataArray
        2D array red band data.
        (Sentinel 2: Band 4)
    name: str, optional (default ="ndvi")
        Name of output DataArray.

    Returns
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated ndvi values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
    http://ceholden.github.io/open-geo-tutorial/python/chapter_2_indices.html

        Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Sample Band Data
    >>> np.random.seed(0)
    >>> nir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    >>> np.random.seed(1)
    >>> red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = red_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> red_agg["lat"] = _lat
    >>> red_agg["lon"] = _lon

    >>> print(nir_agg, red_agg)
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

    Create NDVI DataArray
    >>> data = xrspatial.multispectral.ndvi(nir_agg, red_agg)
    >>> print(data)
    <xarray.DataArray 'ndvi' (lat: 4, lon: 4)>
    array([[ 0.13645336, -0.0035772 ,  0.99962057,  0.28629143],
           [ 0.4854378 ,  0.74983879,  0.40286613,  0.44144297],
           [ 0.41670295, -0.16847257,  0.30764267, -0.12875605],
           [ 0.4706716 ,  0.02632302,  0.44347537, -0.76998504]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
    """

    validate_arrays(nir_agg, red_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(nir_agg)(nir_agg.data, red_agg.data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# NDMI ----------
def ndmi(nir_agg: DataArray, swir1_agg: DataArray, name='ndmi'):
    """
    Computes Normalized Difference Moisture Index.
    Used to determine vegetation water content.

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band
        (Sentinel 2: Band 8)
        (Landsat 4-7: Band 4)
        (Landsat 8: Band 5)
    swir1_agg : DataArray
        shortwave infrared band
        (Sentinel 2: Band 11)
        (Landsat 4-7: Band 5)
        (Landsat 8: Band 6)
    name: str, optional (default ="ndmi")
        Name of output DataArray.

    Returns
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated ndmi values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
    https://www.usgs.gov/land-resources/nli/landsat/normalized-difference-moisture-index

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Sample Band Data
    >>> np.random.seed(0)
    >>> nir_agg = xr.DataArray(np.random.rand(4,4),
    >>>             dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    >>> np.random.seed(5)
    >>> swir1_agg = xr.DataArray(np.random.rand(4,4),
    >>>            dims = ["lat", "lon"])
    >>> height, width = swir1_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> swir1_agg["lat"] = _lat
    >>> swir1_agg["lon"] = _lon

    >>> print(nir_agg, swir1_agg)
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
           [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
           [0.96366276, 0.38344152, 0.79172504, 0.52889492],
           [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
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

    Create NDMI DataArray
    >>> data = xrspatial.multispectral.ndmi(nir_agg, swir1_agg)
    >>> print(data)
    <xarray.DataArray 'ndmi' (lat: 4, lon: 4)>
    array([[ 0.4239978 , -0.09807732,  0.48925604, -0.25536675],
           [-0.07099968,  0.02715428, -0.27280597,  0.26475493],
           [ 0.52906124,  0.34266992,  0.81491258, -0.16534329],
           [ 0.12556087,  0.70789018, -0.85060343, -0.51757753]])
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
    """

    validate_arrays(nir_agg, swir1_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

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
def savi(nir_agg: DataArray, red_agg: DataArray,
         soil_factor: float = 1.0, name: str = 'savi'):
    """
    Computes Soil Adjusted Vegetation Index (SAVI).
    Used to determine if a cell contains living
    vegetation while minimizing soil brightness.

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data
        (Sentinel 2: Band 8)
    red_agg : DataArray
        red band data
        (Sentinel 2: Band 4)
    soil_factor : float
        soil adjustment factor between -1.0 and 1.0.
        when set to zero, savi will return the same as ndvi
    name: str, optional (default ="savi")
        Name of output DataArray.

    Returns
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated savi values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
     - https://www.sciencedirect.com/science/article/abs/pii/003442578890106X

    Examples
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Sample Band Data
    >>> np.random.seed(0)
    >>> nir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    >>> np.random.seed(1)
    >>> red_agg = xr.DataArray(np.random.rand(4,4),
    >>>                 dims = ["lat", "lon"])
    >>> height, width = red_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> red_agg["lat"] = _lat
    >>> red_agg["lon"] = _lon

    >>> print(nir_agg, red_agg)
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

        Create SAVI DataArray
    >>> data = xrspatial.multispectral.savi(nir_agg, red_agg)
    >>> print(data)
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


def sipi(nir_agg: DataArray, red_agg: DataArray, blue_agg: DataArray,
         name='sipi'):
    """
    Computes Structure Insensitive Pigment Index which helpful
    in early disease detection in vegetation.

    Parameters
    ----------
    nir_agg: xarray.DataArray
        2D array of near-infrared band data.
        (Sentinel 2: Band 8)
    red_agg: xarray.DataArray
        2D array of red band data.
        (Sentinel 2: Band 4)
    blue_agg: xarray.DataArray
        2D array of blue band data.
        (Sentinel 2: Band 2)
    name: str, optional (default = "sipi")
        Name of output DataArray.

    Returns
    ----------
     xarray.DataArray
        2D array, of the same type as the input, of calculated sipi values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
    https://en.wikipedia.org/wiki/Enhanced_vegetation_index

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Sample Band Data
    >>> np.random.seed(0)
    >>> nir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    >>> np.random.seed(1)
    >>> red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = red_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> red_agg["lat"] = _lat
    >>> red_agg["lon"] = _lon

    >>> np.random.seed(2)
    >>> blue_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = blue_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> blue_agg["lat"] = _lat
    >>> blue_agg["lon"] = _lon

    >>> print(nir_agg, red_agg, blue_agg)
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

    Create ARVI DataArray
    >>> data = xrspatial.multispectral.sipi(nir_agg, red_agg, blue_agg)
    >>> print(data)
    <xarray.DataArray 'sipi' (lat: 4, lon: 4)>
    array([[ 8.56038534e-01, -1.34225137e+02,  8.81124802e-02,
         4.51702802e-01],
       [ 1.18707483e-02,  5.70058976e-01,  9.26834671e-01,
         4.98894015e-01],
       [ 1.17130642e+00, -7.50533112e-01,  4.57925444e-01,
         1.58116224e-03],
       [ 1.19217212e+00,  8.67787369e+00, -2.59811674e+00,
         1.19691430e+00]])
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


def ebbi(red_agg: DataArray, swir_agg: DataArray, tir_agg: DataArray,
         name='ebbi'):
    """
    Computes Enhanced Built-Up and Bareness Index (EBBI) which
    allows for easily distinguishing between built-up and bare land areas.

    Parameters:
    ----------
    red_agg: xarray.DataArray
        2D array of red band data.
        (Sentinel 2: Band 4)
    swir_agg: xarray.DataArray
        2D array of shortwave infrared band data.
        (Sentinel 2: Band 11)
    tir_agg: xarray.DataArray
        2D array of thermal infrared band data.
    name: str, optional (default = "ebbi")
        Name of output DataArray.

    Returns
    ----------
    xarray.DataArray
        2D array, of the same type as the input of calculated ebbi values.
        All other input attributes are preserved

    Notes:
    ----------
    Algorithm References:
        https://rdrr.io/cran/LSRS/man/EBBI.html

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Sample Band Data
    >>> np.random.seed(1)
    >>> red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = red_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> red_agg["lat"] = _lat
    >>> red_agg["lon"] = _lon

    >>> np.random.seed(5)
    >>> swir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = swir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> swir_agg["lat"] = _lat
    >>> swir_agg["lon"] = _lon

    >>> np.random.seed(6)
    >>> tir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = tir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> tir_agg["lat"] = _lat
    >>> tir_agg["lon"] = _lon

    >>> print(red_agg, swir_agg, tir_agg)
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
        Create EBBI DataArray
    >>> data = xrspatial.multispectral.ebbi(red_agg, swir_agg, tir_agg)
    >>> print(data)
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
def _normalize_data_cpu(data, min_val, max_val, pixel_max):
    out = np.zeros_like(data)
    out[:] = np.nan

    range_val = max_val - min_val
    rows, cols = data.shape

    c = 10
    th = .125

    # check range_val to avoid dividing by zero
    if range_val != 0:
        for y in range(rows):
            for x in range(cols):
                val = data[y, x]
                norm = (val - min_val) / range_val
                # sigmoid contrast enhancement
                norm = 1 / (1 + np.exp(c * (th - norm)))
                out[y, x] = norm * pixel_max
    return out


def _normalize_data_numpy(data, pixel_max):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    out = _normalize_data_cpu(data, min_val, max_val, pixel_max)
    return out


def _normalize_data_dask(data, pixel_max):
    min_val = da.nanmin(data)
    max_val = da.nanmax(data)
    out = da.map_blocks(_normalize_data_cpu, data, min_val, max_val, pixel_max,
                        meta=np.array(()))
    return out


def _normalize_data_cupy(data, pixel_max):
    raise NotImplementedError('Not Supported')


def _normalize_data_dask_cupy(data, pixel_max):
    raise NotImplementedError('Not Supported')


def _normalize_data(agg, pixel_max=255.0):
    mapper = ArrayTypeFunctionMapping(numpy_func=_normalize_data_numpy,
                                      dask_func=_normalize_data_dask,
                                      cupy_func=_normalize_data_cupy,
                                      dask_cupy_func=_normalize_data_dask_cupy)
    out = mapper(agg)(agg.data, pixel_max)
    return out


def true_color(r, g, b, nodata=1):
    h, w = r.shape

    pixel_max = 255

    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[:, :, 0] = (_normalize_data(r, pixel_max)).astype(np.uint8)
    data[:, :, 1] = (_normalize_data(g, pixel_max)).astype(np.uint8)
    data[:, :, 2] = (_normalize_data(b, pixel_max)).astype(np.uint8)

    a = np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)
    data[:, :, 3] = a.astype(np.uint8)

    return Image.fromarray(data, 'RGBA')
