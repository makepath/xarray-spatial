from math import sqrt
import numpy as np
import numba as nb

from numba import cuda

import datashader as ds

from PIL import Image

from xarray import DataArray

from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args
from xrspatial.utils import ngjit


def _check_is_dataarray(val, name='value'):
    if not isinstance(val, DataArray):
        msg = "{} must be instance of DataArray".format(name)
        raise TypeError(msg)


@ngjit
def _arvi(nir_data, red_data, blue_data):
    out = np.zeros_like(nir_data)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):

            nir = nir_data[y, x]
            red = red_data[y, x]
            blue = blue_data[y, x]

            numerator = (nir - (2.0 * red) + blue)
            denominator = (nir + (2.0 * red) + blue)

            if denominator == 0.0:
                continue
            else:
                out[y, x] = numerator / denominator
    return out


def arvi(nir_agg, red_agg, blue_agg, name='arvi', use_cuda=True, use_cupy=True):
    """
Computes Atmospherically Resistant Vegetation Index. Allows for molecular and ozone correction
with no further need for aerosol correction, except for dust conditions.

Parameters:
----------
    nir_agg: xarray.DataArray
        - 2D array of near-infrared band data.
    red_agg: DataArray
        - 2D array of red band data.
    blue_agg: DataArray
        - 2D array of blue band data.
    name: String, optional (default = "arvi")
        - Name of output DataArray.
    use_cuda: Boolean, optional (default = True)
        - 
    use_cupy: Boolean, optional (default = True)
        - 

Returns:
----------
    data: xarray.DataArray
        - 2D array, of the same type as the input, of calculated arvi values.
        - All other input attributes are preserved.
Notes:
----------
    Algorithm References:
        - https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=03667

Examples:
    Imports
>>>     import numpy as np
>>>     import xarray as xr
>>>     import xrspatial

    Create Sample Band Data
>>>     np.random.seed(0)
>>>     nir_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
>>>     height, width = nir_agg.shape
>>>     _lat = np.linspace(0, height - 1, height)
>>>     _lon = np.linspace(0, width - 1, width)
>>>     nir_agg["lat"] = _lat
>>>     nir_agg["lon"] = _lon

>>>     np.random.seed(1)
>>>     red_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
>>>     height, width = red_agg.shape
>>>     _lat = np.linspace(0, height - 1, height)
>>>     _lon = np.linspace(0, width - 1, width)
>>>     red_agg["lat"] = _lat
>>>     red_agg["lon"] = _lon

>>>     np.random.seed(2)
>>>     blue_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
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
  * lon      (lon) float64 0.0 1.0 2.0 3.0 <xarray.DataArray (lat: 4, lon: 4)>
array([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01],
       [1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01],
       [3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01],
       [2.04452250e-01, 8.78117436e-01, 2.73875932e-02, 6.70467510e-01]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0 <xarray.DataArray (lat: 4, lon: 4)>
array([[0.4359949 , 0.02592623, 0.54966248, 0.43532239],
       [0.4203678 , 0.33033482, 0.20464863, 0.61927097],
       [0.29965467, 0.26682728, 0.62113383, 0.52914209],
       [0.13457995, 0.51357812, 0.18443987, 0.78533515]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0

    Create ARVI DataArray
>>>     data = xrspatial.multispectral.arvi(nir_agg, red_agg, blue_agg)
>>>     print(data)
<xarray.DataArray 'arvi' (lat: 4, lon: 4)>
array([[ 0.08288985, -0.32062735,  0.99960309,  0.23695335],
       [ 0.48395093,  0.68183958,  0.26579331,  0.37232558],
       [ 0.22839874, -0.24733151,  0.2551784 , -0.12864117],
       [ 0.26424862, -0.09922362,  0.64689773, -0.21165207]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0
----------
    """
    _check_is_dataarray(nir_agg, 'near-infrared')
    _check_is_dataarray(red_agg, 'red')
    _check_is_dataarray(blue_agg, 'blue')

    if not red_agg.shape == nir_agg.shape == blue_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    return DataArray(_arvi(nir_agg.data, red_agg.data, blue_agg.data),
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


@ngjit
def _evi(nir_data, red_data, blue_data, c1, c2, soil_factor, gain):
    out = np.zeros_like(nir_data)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):

            nir = nir_data[y, x]
            red = red_data[y, x]
            blue = blue_data[y, x]

            numerator = nir - red
            denominator = nir + c1 * red - c2 * blue + soil_factor

            if denominator == 0.0:
                continue
            else:
                out[y, x] = gain * (numerator / denominator)
    return out


def evi(nir_agg, red_agg, blue_agg, c1=6.0, c2=7.5, soil_factor=1.0, gain=2.5,
        name='evi', use_cuda=True, use_cupy=True):
    """
Computes Enhanced Vegetation Index. Allows for importved sensitivity in high biomass regions,
de-coupling of the canopy background signal and reduction of atmospheric influences.

Parameters:
----------
    nir_agg: xarray.DataArray
        - 2D array of near-infrared band data.
    red_agg: xarray.DataArray
        - 2D array of red band data.
    blue_agg: xarray.DataArray
        - 2D array of blue band data.
    c1: Float (default = 6.0)
        - First coefficient of the aerosol resistance term.
    c2: Float (default = 7.5)
        - Second coefficients of the aerosol resistance term.
    soil_factor: Float (default = 1.0)
        - Soil adjustment factor between -1.0 and 1.0.
    gain: float (default = 2.5)
        - Amplitude adjustment factor.
    name: String, optional (default = "evi")
        - Name of output DataArray.
    use_cuda: Boolean, optional (default = True)
        - 
    use_cupy: Boolean, optional (default = True)
        - 

Returns:
----------
    data: DataArray
        - 2D array, of the same type as the input, of calculated evi values.
        - All other input attributes are preserved.

Notes:
----------
    Algorithm References:
        - https://en.wikipedia.org/wiki/Enhanced_vegetation_index

Examples:
----------
    Imports
>>>     import numpy as np
>>>     import xarray as xr
>>>     import xrspatial

    Create Sample Band Data
>>>     np.random.seed(0)
>>>     nir_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
>>>     height, width = nir_agg.shape
>>>     _lat = np.linspace(0, height - 1, height)
>>>     _lon = np.linspace(0, width - 1, width)
>>>     nir_agg["lat"] = _lat
>>>     nir_agg["lon"] = _lon

>>>     np.random.seed(1)
>>>     red_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
>>>     height, width = red_agg.shape
>>>     _lat = np.linspace(0, height - 1, height)
>>>     _lon = np.linspace(0, width - 1, width)
>>>     red_agg["lat"] = _lat
>>>     red_agg["lon"] = _lon

>>>     np.random.seed(2)
>>>     blue_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
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
  * lon      (lon) float64 0.0 1.0 2.0 3.0 <xarray.DataArray (lat: 4, lon: 4)>
array([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01],
       [1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01],
       [3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01],
       [2.04452250e-01, 8.78117436e-01, 2.73875932e-02, 6.70467510e-01]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0 <xarray.DataArray (lat: 4, lon: 4)>
array([[0.4359949 , 0.02592623, 0.54966248, 0.43532239],
       [0.4203678 , 0.33033482, 0.20464863, 0.61927097],
       [0.29965467, 0.26682728, 0.62113383, 0.52914209],
       [0.13457995, 0.51357812, 0.18443987, 0.78533515]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0

    Create EVI DataArray
>>>     data = xrspatial.multispectral.evi(nir_agg, red_agg, blue_agg)
>>>     print(data)
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
    _check_is_dataarray(nir_agg, 'near-infrared')
    _check_is_dataarray(red_agg, 'red')
    _check_is_dataarray(blue_agg, 'blue')

    if not red_agg.shape == nir_agg.shape == blue_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    if not isinstance(c1, (float, int)):
        raise ValueError("c1 must be numeric")

    if not isinstance(c2, (float, int)):
        raise ValueError("c2 must be numeric")

    if soil_factor > 1.0 or soil_factor < -1.0:
        raise ValueError("soil factor must be between (-1.0, 1.0)")

    if gain < 0:
        raise ValueError("gain must be greater than 0")

    arr = _evi(nir_agg.data, red_agg.data, blue_agg.data, c1, c2,
               soil_factor, gain)

    return DataArray(arr,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


@ngjit
def _gci(nir_data, green_data):
    out = np.zeros_like(nir_data)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            green = green_data[y, x]

            if green == 0.0:
                continue
            else:
                out[y, x] = nir / green - 1
    return out


def gci(nir_agg, green_agg, name='gci', use_cuda=True, use_cupy=True):
    """
Computes Green Chlorophyll Index. Used to estimate the content of leaf chorophyll and predict
the physiological state of vegetation and plant health.

Parameters:
----------
    nir_agg: xarray. DataArray
        - 2D array of near-infrared band data.
    green_agg: DataArray
        - 2D array of green band data.
    name: String, optional (default = "gci")
        - Name of output DataArray
    use_cuda: Boolean, optional (default = True)
        - 
    use_cupy: Boolean, optional (default = True)
        - 

Returns:
----------
    data: xarray.DataArray
        - 2D array, of the same type as the input, of calculated gci values.
        - All other input attributes are preserved.

Notes:
----------
    Algorithm References:
        - https://en.wikipedia.org/wiki/Enhanced_vegetation_index
Examples:
----------
    Imports
>>>     import numpy as np
>>>     import xarray as xr
>>>     import xrspatial

    Create Sample Band Data
>>>     np.random.seed(0)
>>>     nir_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
>>>     height, width = nir_agg.shape
>>>     _lat = np.linspace(0, height - 1, height)
>>>     _lon = np.linspace(0, width - 1, width)
>>>     nir_agg["lat"] = _lat
>>>     nir_agg["lon"] = _lon

>>>     np.random.seed(3)
>>>     green_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
>>>     height, width = green_agg.shape
>>>     _lat = np.linspace(0, height - 1, height)
>>>     _lon = np.linspace(0, width - 1, width)
>>>     green_agg["lat"] = _lat
>>>     green_agg["lon"] = _lon



>>>     print(nir_agg, green_agg)
<xarray.DataArray (lat: 4, lon: 4)>
array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
       [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
       [0.96366276, 0.38344152, 0.79172504, 0.52889492],
       [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0 <xarray.DataArray (lat: 4, lon: 4)>
array([[0.5507979 , 0.70814782, 0.29090474, 0.51082761],
       [0.89294695, 0.89629309, 0.12558531, 0.20724288],
       [0.0514672 , 0.44080984, 0.02987621, 0.45683322],
       [0.64914405, 0.27848728, 0.6762549 , 0.59086282]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0

    Create GCI DataArray
>>>     data = xrspatial.multispectral.gci(nir_agg, green_agg)
>>>     print(data)
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
    _check_is_dataarray(nir_agg, 'near-infrared')
    _check_is_dataarray(green_agg, 'green')

    if not nir_agg.shape == green_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    arr = _gci(nir_agg.data, green_agg.data)

    return DataArray(arr,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


@ngjit
def _normalized_ratio(arr1, arr2):
    out = np.zeros_like(arr1)
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


def nbr(nir_agg, swir2_agg, name='nbr', use_cuda=True, use_cupy=True):
    """
Computes Normalized Burn Ratio. Used to identify burned areas and provide a measure of burn
severity.

Parameters:
----------
    nir_agg: xarray.DataArray
        - 2D array of near-infrared band data.
    swir_agg: xarray.DataArray
        - 2D array of shortwave infrared band data.
        - (Landsat 4-7: Band 6)
        - (Landsat 8: Band 7)
    name: String, optional (default = "nbr")
        - Name of output DataArray.
    use_cuda: Boolean (default = "True")
        - 
    use_cupy: Boolean (default = "True")
        -

Returns:
----------
    data: xarray.DataArray
        - 2D array, of the same type as the input, of calculated gci values.
        - All other input attributes are preserved.
Notes:
----------
    Algorithm References:
        - https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio

Examples:
----------
    Imports
>>>     import numpy as np
>>>     import xarray as xr
>>>     import xrspatial
    Create Sample Band Data
>>>     np.random.seed(0)
>>>     nir_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
>>>     height, width = nir_agg.shape
>>>     _lat = np.linspace(0, height - 1, height)
>>>     _lon = np.linspace(0, width - 1, width)
>>>     nir_agg["lat"] = _lat
>>>     nir_agg["lon"] = _lon

>>>     np.random.seed(4)
>>>     swir2_agg = xr.DataArray(np.random.rand(4,4), 
>>>                             dims = ["lat", "lon"])
>>>     height, width = swir2_agg.shape
>>>     _lat = np.linspace(0, height - 1, height)
>>>     _lon = np.linspace(0, width - 1, width)
>>>     swir2_agg["lat"] = _lat
>>>     swir2_agg["lon"] = _lon

>>>     print(nir_agg, swir2_agg)
<xarray.DataArray (lat: 4, lon: 4)>
array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
       [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
       [0.96366276, 0.38344152, 0.79172504, 0.52889492],
       [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0 <xarray.DataArray (lat: 4, lon: 4)>
array([[0.96702984, 0.54723225, 0.97268436, 0.71481599],
       [0.69772882, 0.2160895 , 0.97627445, 0.00623026],
       [0.25298236, 0.43479153, 0.77938292, 0.19768507],
       [0.86299324, 0.98340068, 0.16384224, 0.59733394]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0
    Create NBR DataArray
>>>     data = xrspatial.multispectral.gci(nir_agg, swir2_agg)
>>>     print(data)
<xarray.DataArray 'gci' (lat: 4, lon: 4)>
array([[-4.32475109e-01,  3.06921088e-01, -3.80309378e-01,
        -2.37729447e-01],
       [-3.92808804e-01,  1.98901208e+00, -5.51778489e-01,
         1.42135870e+02],
       [ 2.80920927e+00, -1.18102607e-01,  1.58357541e-02,
         1.67544184e+00],
       [-3.41774028e-01, -5.87797428e-02, -5.66436240e-01,
        -8.54136366e-01]])
Coordinates:
  * lat      (lat) float64 0.0 1.0 2.0 3.0
  * lon      (lon) float64 0.0 1.0 2.0 3.0
    """
    _check_is_dataarray(nir_agg, 'near-infrared')
    _check_is_dataarray(swir2_agg, 'shortwave infrared')

    if not nir_agg.shape == swir2_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    out = _run_normalized_ratio(nir_agg.data, swir2_agg.data, use_cuda=use_cuda, use_cupy=use_cupy)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


def nbr2(swir1_agg, swir2_agg, name='nbr', use_cuda=True, use_cupy=True):
    """Computes Normalized Burn Ratio 2

    "NBR2 modifies the Normalized Burn Ratio (NBR)
    to highlight water sensitivity in vegetation and
    may be useful in post-fire recovery studies."

    https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio-2

    Parameters
    ----------
    swir1_agg : DataArray
        near-infrared band
        shortwave infrared band
        (Landsat 4-7: Band 5)
        (Landsat 8: Band 6)

    swir2_agg : DataArray
        shortwave infrared band
        (Landsat 4-7: Band 6)
        (Landsat 8: Band 7)

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
    https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio-2
    """
    _check_is_dataarray(swir1_agg, 'near-infrared')
    _check_is_dataarray(swir2_agg, 'shortwave infrared')

    if not swir1_agg.shape == swir2_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    out = _run_normalized_ratio(swir1_agg.data, swir2_agg.data, use_cuda=use_cuda, use_cupy=use_cupy)

    return DataArray(out,
                     name=name,
                     coords=swir1_agg.coords,
                     dims=swir1_agg.dims,
                     attrs=swir1_agg.attrs)


def ndvi(nir_agg, red_agg, name='ndvi', use_cuda=True, use_cupy=True):
    """Returns Normalized Difference Vegetation Index (NDVI).

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data
    red_agg : DataArray
        red band data

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
    http://ceholden.github.io/open-geo-tutorial/python/chapter_2_indices.html
    """

    _check_is_dataarray(nir_agg, 'near-infrared')
    _check_is_dataarray(red_agg, 'red')

    if not red_agg.shape == nir_agg.shape:
        raise ValueError("red_agg and nir_agg expected to have equal shapes")

    out = _run_normalized_ratio(nir_agg.data, red_agg.data, use_cuda=use_cuda, use_cupy=use_cupy)

    return DataArray(out,
                     name='ndvi',
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


def _run_normalized_ratio(arr1, arr2, use_cuda=True, use_cupy=True):

    if has_cuda() and use_cuda:
        griddim, blockdim = cuda_args(arr1.shape)
        out = np.empty(arr1.shape, dtype='f4')
        out[:] = np.nan

        if use_cupy:
            import cupy
            out = cupy.asarray(out)

        _normalized_ratio_gpu[griddim, blockdim](arr1, arr2, out)
    else:
        out = _normalized_ratio(arr1, arr2)
    
    return out


def ndmi(nir_agg, swir1_agg, name='ndmi', use_cuda=True, use_cupy=True):
    """Computes Normalized Difference Moisture Index

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band
        (Landsat 4-7: Band 4)
        (Landsat 8: Band 5)

    swir1_agg : DataArray
        shortwave infrared band
        (Landsat 4-7: Band 5)
        (Landsat 8: Band 6)


    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
    https://www.usgs.gov/land-resources/nli/landsat/normalized-difference-moisture-index
    """
    _check_is_dataarray(nir_agg, 'near-infrared')
    _check_is_dataarray(swir1_agg, 'shortwave infrared')

    if not nir_agg.shape == swir1_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    nir_data = nir_agg.data
    swir1_data = swir1_agg.data

    out = _run_normalized_ratio(nir_data, swir1_data, use_cuda=use_cuda, use_cupy=use_cupy)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


@ngjit
def _savi(nir_data, red_data, soil_factor):
    out = np.zeros_like(nir_data)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]

            numerator = nir - red

            soma = nir + red + soil_factor
            denominator = soma * (1.0 + soil_factor)

            if denominator == 0.0:
                continue
            else:
                out[y, x] = numerator / denominator

    return out


@cuda.jit
def _normalized_ratio_gpu(arr1, arr2, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        val1 = arr1[y, x]
        val2 = arr2[y, x]
        numerator = val1 - val2
        denominator = val1 + val2
        out[y, x] = numerator / denominator


@cuda.jit
def _savi_gpu(nir_data, red_data, soil_factor, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        red = red_data[y, x]
        numerator = nir - red
        soma = nir + red + soil_factor[0]
        denominator = soma * (nb.float32(1.0) + soil_factor[0])

        if denominator == 0.0:
            out[y, x] = np.nan
        else:
            out[y, x] = numerator / denominator


def savi(nir_agg, red_agg, soil_factor=1.0, name='savi', use_cuda=True, use_cupy=True):
    """Returns Soil Adjusted Vegetation Index (SAVI).

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data

    red_agg : DataArray
        red band data

    soil_factor : float
      soil adjustment factor between -1.0 and 1.0.
      when set to zero, savi will return the same as ndvi

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
     - https://www.sciencedirect.com/science/article/abs/pii/003442578890106X
    """
    _check_is_dataarray(nir_agg, 'near-infrared')
    _check_is_dataarray(red_agg, 'red')

    if not red_agg.shape == nir_agg.shape:
        raise ValueError("red_agg and nir_agg expected to have equal shapes")

    if soil_factor > 1.0 or soil_factor < -1.0:
        raise ValueError("soil factor must be between (-1.0, 1.0)")

    nir_data = nir_agg.data
    red_data = red_agg.data

    if has_cuda() and use_cuda:
        griddim, blockdim = cuda_args(nir_data.shape)
        soil_factor_arr = np.array([float(soil_factor)], dtype='f4')

        out = np.empty(nir_data.shape, dtype='f4')
        out[:] = np.nan

        if use_cupy:
            import cupy
            out = cupy.asarray(out)

        _savi_gpu[griddim, blockdim](nir_data,
                                     red_data,
                                     soil_factor_arr,
                                     out)
    else:
        out = _savi(nir_agg.data, red_agg.data, soil_factor)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


@ngjit
def _sipi(nir_data, red_data, blue_data):
    out = np.zeros_like(nir_data)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            blue = blue_data[y, x]

            numerator = nir - blue
            denominator = nir - red

            if denominator == 0.0:
                continue
            else:
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

        if denominator == 0.0:
            out[y, x] = np.nan
        else:
            out[y, x] = numerator / denominator


def sipi(nir_agg, red_agg, blue_agg, name='sipi', use_cuda=True, use_cupy=True):
    """Computes Structure Insensitive Pigment Index which helpful
    in early disease detection

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data

    green_agg : DataArray
        green band data

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
    https://en.wikipedia.org/wiki/Enhanced_vegetation_index
    """

    _check_is_dataarray(nir_agg, 'near-infrared')
    _check_is_dataarray(red_agg, 'red')
    _check_is_dataarray(blue_agg, 'blue')

    if not red_agg.shape == nir_agg.shape == blue_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    nir_data = nir_agg.data
    red_data = red_agg.data
    blue_data = blue_agg.data

    if has_cuda() and use_cuda:
        griddim, blockdim = cuda_args(nir_data.shape)
        out = np.empty(nir_data.shape, dtype='f4')
        out[:] = np.nan

        if use_cupy:
            import cupy
            out = cupy.asarray(out)

        _sipi_gpu[griddim, blockdim](nir_data,
                                     red_data,
                                     blue_data,
                                     out)
    else:
        out = _sipi(nir_data, red_data, blue_data)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


@ngjit
def _ebbi(red_data, swir_data, tir_data):
    out = np.zeros_like(red_data)
    rows, cols = red_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            red = red_data[y, x]
            swir = swir_data[y, x]
            tir = tir_data[y, x]

            numerator = swir - red
            denominator = 10 * np.sqrt(swir + tir)

            if denominator == 0.0:
                continue
            else:
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
        out[y, x] = numerator / denominator


def ebbi(red_agg, swir_agg, tir_agg, name='ebbi', use_cuda=True, use_cupy=True):
    """Computes Enhanced Built-Up and Bareness Index
    Parameters
    ----------
    red_agg : DataArray
        red band data
    swir_agg : DataArray
        shortwave infrared band data
    tir_agg : DataArray
        thermal infrared band data
    Returns
    -------
    data: DataArray
    Notes:
    ------
    Algorithm References:
    https://rdrr.io/cran/LSRS/man/EBBI.html
    """

    _check_is_dataarray(red_agg, 'red')
    _check_is_dataarray(swir_agg, 'swir')
    _check_is_dataarray(tir_agg, 'thermal infrared')

    if not red_agg.shape == swir_agg.shape == tir_agg.shape:
        raise ValueError("input layers expected to have equal shapes")

    red_data = red_agg.data
    swir_data = swir_agg.data
    tir_data = tir_agg.data

    if has_cuda() and use_cuda:
        griddim, blockdim = cuda_args(red_data.shape)
        out = np.empty(red_data.shape, dtype='f4')
        out[:] = np.nan

        if use_cupy:
            import cupy
            out = cupy.asarray(out)

        _sipi_gpu[griddim, blockdim](red_data,
                                     swir_data,
                                     tir_data,
                                     out)
    else:
        out = _sipi(red_data, swir_data, tir_data)

    return DataArray(out,
                     name=name,
                     coords=red_agg.coords,
                     dims=red_agg.dims,
                     attrs=red_agg.attrs)

@ngjit
def _normalize_data(agg, pixel_max=255.0):
    out = np.zeros_like(agg)
    min_val = 0
    max_val = 2**16 - 1
    range_val = max_val - min_val
    rows, cols = agg.shape
    c = 40
    th = .125
    for y in range(rows):
        for x in range(cols):
            val = agg[y, x]
            norm = (val - min_val) / range_val

            # sigmoid contrast enhancement
            norm = 1 / (1 + np.exp(c * (th - norm)))
            out[y, x] = norm * pixel_max
    return out


def bands_to_img(r, g, b, nodata=1):
    h, w = r.shape
    r, g, b = [ds.utils.orient_array(img) for img in (r, g, b)]

    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[:, :, 0] = (_normalize_data(r)).astype(np.uint8)
    data[:, :, 1] = (_normalize_data(g)).astype(np.uint8)
    data[:, :, 2] = (_normalize_data(b)).astype(np.uint8)

    a = np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)
    data[:, :, 3] = a.astype(np.uint8)

    return Image.fromarray(data, 'RGBA')
