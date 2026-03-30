from __future__ import annotations

import warnings
from math import sqrt

import numba as nb
import numpy as np
import xarray as xr
from numba import cuda
from xarray import DataArray

from xrspatial.utils import (ArrayTypeFunctionMapping, _validate_raster, cuda_args, ngjit,
                             not_implemented_func, validate_arrays)
from xrspatial.dataset_support import supports_dataset_bands

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


@ngjit
def _arvi_cpu(nir_data, red_data, blue_data):
    out = np.full(nir_data.shape, np.nan, dtype=np.float32)
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


@supports_dataset_bands(nir='nir_agg', red='red_agg', blue='blue_agg')
def arvi(nir_agg: xr.DataArray,
         red_agg: xr.DataArray,
         blue_agg: xr.DataArray,
         name='arvi'):
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
    name : str, default='arvi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        arvi(ds, nir='B8', red='B4', blue='B2')

    Returns
    -------
    arvi_agg : xarray.DataArray of the same type as inputs.
        2D array arvi values. All other input attributes are preserved.

    References
    ----------
        - MODIS: https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=03667 # noqa

    Examples
    --------
    In this example, we'll use data available in xrspatial.datasets

    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> nir = data['NIR']
        >>> red = data['Red']
        >>> blue = data['Blue']
        >>> from xrspatial.multispectral import arvi
        >>> # Generate ARVI Aggregate Array
        >>> arvi_agg = arvi(nir_agg=nir, red_agg=red, blue_agg=blue)
        >>> nir.plot(cmap='Greys', aspect=2, size=4)
        >>> red.plot(aspect=2, size=4)
        >>> blue.plot(aspect=2, size=4)
        >>> arvi_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> y1, x1, y2, x2 = 100, 100, 103, 104
        >>> print(nir[y1:y2, x1:x2].data)
        [[1519. 1504. 1530. 1589.]
         [1491. 1473. 1542. 1609.]
         [1479. 1461. 1592. 1653.]]
        >>> print(red[y1:y2, x1:x2].data)
        [[1327. 1329. 1363. 1392.]
         [1309. 1331. 1423. 1424.]
         [1293. 1337. 1455. 1414.]]
        >>> print(blue[y1:y2, x1:x2].data)
        [[1281. 1270. 1254. 1297.]
         [1241. 1249. 1280. 1309.]
         [1239. 1257. 1322. 1329.]]
        >>> print(arvi_agg[y1:y2, x1:x2].data)
        [[ 0.02676934  0.02135493  0.01052632  0.01798942]
         [ 0.02130841  0.01114413 -0.0042343   0.01214013]
         [ 0.02488688  0.00816024  0.00068681  0.02650602]]
    """

    _validate_raster(nir_agg, func_name='arvi', name='nir_agg')
    _validate_raster(red_agg, func_name='arvi', name='red_agg')
    _validate_raster(blue_agg, func_name='arvi', name='blue_agg')

    validate_arrays(red_agg, nir_agg, blue_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_arvi_cpu,
                                      dask_func=_arvi_dask,
                                      cupy_func=_arvi_cupy,
                                      dask_cupy_func=_arvi_dask_cupy)

    out = mapper(red_agg)(
        nir_agg.data.astype('f4'), red_agg.data.astype('f4'), blue_agg.data.astype('f4')
    )

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# EVI ----------
@ngjit
def _evi_cpu(nir_data, red_data, blue_data, c1, c2, soil_factor, gain):
    out = np.full(nir_data.shape, np.nan, dtype=np.float32)
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


@supports_dataset_bands(nir='nir_agg', red='red_agg', blue='blue_agg')
def evi(nir_agg: xr.DataArray,
        red_agg: xr.DataArray,
        blue_agg: xr.DataArray,
        c1=6.0,
        c2=7.5,
        soil_factor=1.0,
        gain=2.5,
        name='evi'):
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
    c1 : float, default=6.0
        First coefficient of the aerosol resistance term.
    c2 : float, default=7.5
        Second coefficients of the aerosol resistance term.
    soil_factor : float, default=1.0
        Soil adjustment factor between -1.0 and 1.0.
    gain : float, default=2.5
        Amplitude adjustment factor.
    name : str, default='evi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        evi(ds, nir='B8', red='B4', blue='B2')

    Returns
    -------
    evi_agg : xarray.DataArray of same type as inputs
        2D array of evi values.
        All other input attributes are preserved.

    References
    ----------
        - Wikipedia: https://en.wikipedia.org/wiki/Enhanced_vegetation_index

    Examples
    --------
    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> nir = data['NIR']
        >>> red = data['Red']
        >>> blue = data['Blue']
        >>> from xrspatial.multispectral import evi
        >>> # Generate EVI Aggregate Array
        >>> evi_agg = evi(nir_agg=nir, red_agg=red, blue_agg=blue)
        >>> nir.plot(aspect=2, size=4)
        >>> red.plot(aspect=2, size=4)
        >>> blue.plot(aspect=2, size=4)
        >>> evi_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> y, x = 100, 100
        >>> m, n = 3, 4
        >>> print(nir[y1:y2, x1:x2].data)
        [[1519. 1504. 1530. 1589.]
         [1491. 1473. 1542. 1609.]
         [1479. 1461. 1592. 1653.]]
        >>> print(red[y1:y2, x1:x2].data)
        [[1327. 1329. 1363. 1392.]
         [1309. 1331. 1423. 1424.]
         [1293. 1337. 1455. 1414.]]
        >>> print(blue[y1:y2, x1:x2].data)
        [[1281. 1270. 1254. 1297.]
         [1241. 1249. 1280. 1309.]
         [1239. 1257. 1322. 1329.]]
        >>> print(evi_agg[y1:y2, x1:x2].data)
        [[-3.8247013 -9.51087    1.3733553  2.2960372]
         [11.818182   3.837838   0.6185031  1.3744428]
         [-8.53211    5.486726   0.8394608  3.5043988]]
    """

    _validate_raster(nir_agg, func_name='evi', name='nir_agg')
    _validate_raster(red_agg, func_name='evi', name='red_agg')
    _validate_raster(blue_agg, func_name='evi', name='blue_agg')

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

    out = mapper(red_agg)(
        nir_agg.data.astype('f4'), red_agg.data.astype('f4'), blue_agg.data.astype('f4'),
        c1, c2, soil_factor, gain
    )

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# GCI ----------
@ngjit
def _gci_cpu(nir_data, green_data):
    out = np.full(nir_data.shape, np.nan, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            green = green_data[y, x]
            if green != 0:
                out[y, x] = nir / green - 1
    return out


@cuda.jit
def _gci_gpu(nir_data, green_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        green = green_data[y, x]
        if green != 0:
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


@supports_dataset_bands(nir='nir_agg', green='green_agg')
def gci(nir_agg: xr.DataArray,
        green_agg: xr.DataArray,
        name='gci'):
    """
    Computes Green Chlorophyll Index. Used to estimate the content of
    leaf chorophyll and predict the physiological state of vegetation
    and plant health.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    green_agg : xr.DataArray
        2D array of green band data.
    name : str, default='gci'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        gci(ds, nir='B8', green='B3')

    Returns
    -------
    gci_agg : xarray.DataArray of the same type as inputs
        2D array of gci values.
        All other input attributes are preserved.

    References
    ----------
        - Wikipedia: https://en.wikipedia.org/wiki/Enhanced_vegetation_index

    Examples
    --------
    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> nir = data['NIR']
        >>> green = data['Green']
        >>> from xrspatial.multispectral import gci
        >>> # Generate GCI Aggregate Array
        >>> gci_agg = gci(nir_agg=nir, green_agg=green)
        >>> nir.plot(aspect=2, size=4)
        >>> green.plot(aspect=2, size=4)
        >>> gci_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> y1, x1, y2, x2 = 100, 100, 103, 104
        >>> print(nir[y1:y2, x1:x2].data)
        [[1519. 1504. 1530. 1589.]
         [1491. 1473. 1542. 1609.]
         [1479. 1461. 1592. 1653.]]
        >>> print(green[y1:y2, x1:x2].data])
        [[1120. 1130. 1157. 1191.]
         [1111. 1137. 1190. 1221.]
         [1097. 1139. 1228. 1216.]]
        >>> print(gci_agg[y1:y2, x1:x2].data)
        [[0.35625    0.33097345 0.3223855  0.33417296]
         [0.3420342  0.29551452 0.29579833 0.31777233]
         [0.34822243 0.28270411 0.29641694 0.359375  ]]
    """

    _validate_raster(nir_agg, func_name='gci', name='nir_agg')
    _validate_raster(green_agg, func_name='gci', name='green_agg')

    validate_arrays(nir_agg, green_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_gci_cpu,
                                      dask_func=_gci_dask,
                                      cupy_func=_gci_cupy,
                                      dask_cupy_func=_gci_dask_cupy)

    out = mapper(nir_agg)(nir_agg.data.astype('f4'), green_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# NBR ----------
@supports_dataset_bands(nir='nir_agg', swir2='swir2_agg')
def nbr(nir_agg: xr.DataArray,
        swir2_agg: xr.DataArray,
        name='nbr'):
    """
    Computes Normalized Burn Ratio. Used to identify burned areas and
    provide a measure of burn severity.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band.
    swir_agg : xr.DataArray
        2D array of shortwave infrared band.
        (Landsat 4-7: Band 6)
        (Landsat 8: Band 7)
    name : str, default='nbr'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        nbr(ds, nir='B8', swir2='B12')

    Returns
    -------
    nbr_agg : xr.DataArray of the same type as inputs
        2D array of nbr values.
        All other input attributes are preserved.

    References
    ----------
        - USGS: https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio # noqa
    Examples
    --------
    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> nir = data['NIR']
        >>> swir2 = data['SWIR2']
        >>> from xrspatial.multispectral import nbr
        >>> # Generate NBR Aggregate Array
        >>> nbr_agg = nbr(nir_agg=nir, swir2_agg=swir2)
        >>> nir.plot(aspect=2, size=4)
        >>> swir2.plot(aspect=2, size=4)
        >>> nbr_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> y1, x1, y2, x2 = 100, 100, 103, 104
        >>> print(nir[y1:y2, x1:x2].data)
        [[1519. 1504. 1530. 1589.]
         [1491. 1473. 1542. 1609.]
         [1479. 1461. 1592. 1653.]]
        >>> print(swir2[y1:y2, x1:x2].data)
        [[1866. 1962. 2086. 2112.]
         [1811. 1900. 2012. 2041.]
         [1838. 1956. 2067. 2109.]]
        >>> print(nbr_agg[y1:y2, x1:x2].data)
        [[-0.10251108 -0.1321408  -0.15376106 -0.14131317]
         [-0.09691096 -0.12659353 -0.13224536 -0.11835616]
         [-0.10823033 -0.14486392 -0.12981689 -0.12121212]]
    """

    _validate_raster(nir_agg, func_name='nbr', name='nir_agg')
    _validate_raster(swir2_agg, func_name='nbr', name='swir2_agg')

    validate_arrays(nir_agg, swir2_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(nir_agg)(nir_agg.data.astype('f4'), swir2_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


@supports_dataset_bands(swir1='swir1_agg', swir2='swir2_agg')
def nbr2(swir1_agg: xr.DataArray,
         swir2_agg: xr.DataArray,
         name='nbr2'):
    """
    Computes Normalized Burn Ratio 2 "NBR2 modifies the Normalized Burn
    Ratio (NBR) to highlight water sensitivity in vegetation and may be
    useful in post-fire recovery studies." [1]_

    Parameters
    ----------
    swir1_agg : xr.DataArray
        2D array of near-infrared band data.
        shortwave infrared band
        (Sentinel 2: Band 11)
        (Landsat 4-7: Band 5)
        (Landsat 8: Band 6)
    swir2_agg : xr.DataArray
        2D array of shortwave infrared band data.
        (Landsat 4-7: Band 6)
        (Landsat 8: Band 7)
    name : str default='nbr2'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        nbr2(ds, swir1='B11', swir2='B12')

    Returns
    -------
    nbr2_agg : xr.DataArray of same type as inputs.
        2D array of nbr2 values.
        All other input attributes are preserved.

    Notes
    -----
    .. [1] https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio-2 # noqa

    Examples
    --------
    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> swir1 = data['SWIR1']
        >>> swir2 = data['SWIR2']
        >>> from xrspatial.multispectral import nbr2
        >>> # Generate NBR2 Aggregate Array
        >>> nbr2_agg = nbr2(swir1_agg=swir1, swir2_agg=swir2)
        >>> swir1.plot(aspect=2, size=4)
        >>> swir2.plot(aspect=2, size=4)
        >>> nbr2_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> y1, x1, y2, x2 = 100, 100, 103, 104
        >>> print(swir1[y1:y2, x1:x2].data)
        [[2092. 2242. 2333. 2382.]
         [2017. 2150. 2303. 2344.]
         [2124. 2244. 2367. 2452.]]
        >>> print(swir2[y1:y2, x1:x2].data)
        [[1866. 1962. 2086. 2112.]
         [1811. 1900. 2012. 2041.]
         [1838. 1956. 2067. 2109.]]
        >>> print(nbr2_agg[y1:y2, x1:x2].data)
        [[0.05709954 0.06660324 0.055895   0.06008011]
         [0.053814   0.0617284  0.06743917 0.0690992 ]
         [0.07218576 0.06857143 0.067659   0.07520281]]
    """

    _validate_raster(swir1_agg, func_name='nbr2', name='swir1_agg')
    _validate_raster(swir2_agg, func_name='nbr2', name='swir2_agg')

    validate_arrays(swir1_agg, swir2_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(swir1_agg)(swir1_agg.data.astype('f4'), swir2_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=swir1_agg.coords,
                     dims=swir1_agg.dims,
                     attrs=swir1_agg.attrs)


# NDVI ----------
@supports_dataset_bands(nir='nir_agg', red='red_agg')
def ndvi(nir_agg: xr.DataArray,
         red_agg: xr.DataArray,
         name='ndvi'):
    """
    Computes Normalized Difference Vegetation Index (NDVI). Used to
    determine if a cell contains live green vegetation.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array red band data.
    name : str default='ndvi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        ndvi(ds, nir='B8', red='B4')

    Returns
    -------
    ndvi_agg : xarray.DataArray of same type as inputs
        2D array of ndvi values.
        All other input attributes are preserved.

    References
    ----------
        - Chris Holden: http://ceholden.github.io/open-geo-tutorial/python/chapter_2_indices.html # noqa

    Examples
    --------
    .. plot::
       :include-source:

       >>> from xrspatial.datasets import get_data
       >>> data = get_data('sentinel-2')  # Open Example Data
       >>> nir = data['NIR']
       >>> red = data['Red']
       >>> from xrspatial.multispectral import ndvi
       >>> # Generate NDVI Aggregate Array
       >>> ndvi_agg = ndvi(nir_agg=nir, red_agg=red)
       >>> nir.plot(aspect=2, size=4)
       >>> red.plot(aspect=2, size=4)
       >>> ndvi_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> y1, x1, y2, x2 = 100, 100, 103, 104
        >>> print(nir[y1:y2, x1:x2].data)
        [[1519. 1504. 1530. 1589.]
         [1491. 1473. 1542. 1609.]
         [1479. 1461. 1592. 1653.]]
        >>> print(red[y1:y2, x1:x2].data)
        [[1327. 1329. 1363. 1392.]
         [1309. 1331. 1423. 1424.]
         [1293. 1337. 1455. 1414.]]
        >>> print(ndvi_agg[y1:y2, x1:x2].data)
        [[0.06746311 0.06177197 0.05772555 0.0660852 ]
         [0.065      0.05064194 0.04013491 0.06099571]
         [0.06709956 0.04431737 0.04496226 0.07792632]]
    """

    _validate_raster(nir_agg, func_name='ndvi', name='nir_agg')
    _validate_raster(red_agg, func_name='ndvi', name='red_agg')

    validate_arrays(nir_agg, red_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(nir_agg)(nir_agg.data.astype('f4'), red_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# NDMI ----------
@supports_dataset_bands(nir='nir_agg', swir1='swir1_agg')
def ndmi(nir_agg: xr.DataArray,
         swir1_agg: xr.DataArray,
         name='ndmi'):
    """
    Computes Normalized Difference Moisture Index. Used to determine
    vegetation water content.

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
    name: str, default='ndmi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        ndmi(ds, nir='B8', swir1='B11')

    Returns
    -------
    ndmi_agg : xr.DataArray of same type as inputs
        2D array of ndmi values.
        All other input attributes are preserved.

    References
    ----------
        - USGS: https://www.usgs.gov/land-resources/nli/landsat/normalized-difference-moisture-index # noqa

    Examples
    --------
    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> nir = data['NIR']
        >>> swir1 = data['SWIR1']
        >>> from xrspatial.multispectral import ndmi
        >>> # Generate NDMI Aggregate Array
        >>> ndmi_agg = ndmi(nir_agg=nir, swir1_agg=swir1)
        >>> nir.plot(aspect=2, size=4)
        >>> swir1.plot(aspect=2, size=4)
        >>> ndmi_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> y1, x1, y2, x2 = 100, 100, 103, 104
        >>> print(nir[y1:y2, x1:x2].data)
        [[1519. 1504. 1530. 1589.]
         [1491. 1473. 1542. 1609.]
         [1479. 1461. 1592. 1653.]]
        >>> print(swir1[y1:y2, x1:x2].data)
        [[2092. 2242. 2333. 2382.]
         [2017. 2150. 2303. 2344.]
         [2124. 2244. 2367. 2452.]]
        >>> print(ndmi_agg[y1:y2, x1:x2].data)
        [[-0.15868181 -0.19701014 -0.20786953 -0.1996978 ]
         [-0.149943   -0.18686172 -0.19791937 -0.18593474]
         [-0.17901748 -0.21133603 -0.19575651 -0.19464068]]
    """

    _validate_raster(nir_agg, func_name='ndmi', name='nir_agg')
    _validate_raster(swir1_agg, func_name='ndmi', name='swir1_agg')

    validate_arrays(nir_agg, swir1_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(nir_agg)(nir_agg.data.astype('f4'), swir1_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# NDWI ----------
@supports_dataset_bands(green='green_agg', nir='nir_agg')
def ndwi(green_agg: xr.DataArray,
         nir_agg: xr.DataArray,
         name='ndwi'):
    """
    Computes Normalized Difference Water Index (NDWI).

    NDWI highlights open water features while suppressing vegetation
    and soil signal.

    Parameters
    ----------
    green_agg : xr.DataArray
        2D array of green band data.
        (Landsat 8: Band 3)
        (Sentinel-2: Band 3)
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
        (Landsat 8: Band 5)
        (Sentinel-2: Band 8)
    name : str, default='ndwi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        ndwi(ds, green='B3', nir='B8')

    Returns
    -------
    ndwi_agg : xr.DataArray of same type as inputs
        2D array of ndwi values in the range [-1, 1].
        All other input attributes are preserved.

    References
    ----------
        - McFeeters, S.K., 1996. The use of the Normalized Difference
          Water Index (NDWI) in the delineation of open water features.
          International Journal of Remote Sensing, 17(7), pp.1425-1432.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.multispectral import ndwi
        >>> green = xr.DataArray(np.array([[600., 500.], [400., 300.]]))
        >>> nir = xr.DataArray(np.array([[300., 400.], [500., 600.]]))
        >>> ndwi(green, nir).values
        array([[ 0.33333334,  0.11111111],
               [-0.11111111, -0.33333334]], dtype=float32)
    """

    _validate_raster(green_agg, func_name='ndwi', name='green_agg')
    _validate_raster(nir_agg, func_name='ndwi', name='nir_agg')

    validate_arrays(green_agg, nir_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(green_agg)(green_agg.data.astype('f4'), nir_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=green_agg.coords,
                     dims=green_agg.dims,
                     attrs=green_agg.attrs)


# MNDWI ----------
@supports_dataset_bands(green='green_agg', swir='swir_agg')
def mndwi(green_agg: xr.DataArray,
          swir_agg: xr.DataArray,
          name='mndwi'):
    """
    Computes Modified Normalized Difference Water Index (MNDWI).

    MNDWI improves on NDWI for urban areas by substituting SWIR for
    NIR, which reduces false positives from built-up surfaces.

    Parameters
    ----------
    green_agg : xr.DataArray
        2D array of green band data.
        (Landsat 8: Band 3)
        (Sentinel-2: Band 3)
    swir_agg : xr.DataArray
        2D array of shortwave infrared band data.
        (Landsat 8: Band 6)
        (Sentinel-2: Band 11)
    name : str, default='mndwi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        mndwi(ds, green='B3', swir='B11')

    Returns
    -------
    mndwi_agg : xr.DataArray of same type as inputs
        2D array of mndwi values in the range [-1, 1].
        All other input attributes are preserved.

    References
    ----------
        - Xu, H., 2006. Modification of normalised difference water
          index (NDWI) to enhance open water features in remotely
          sensed imagery. International Journal of Remote Sensing,
          27(14), pp.3025-3033.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.multispectral import mndwi
        >>> green = xr.DataArray(np.array([[600., 500.], [400., 300.]]))
        >>> swir = xr.DataArray(np.array([[300., 400.], [500., 600.]]))
        >>> mndwi(green, swir).values
        array([[ 0.33333334,  0.11111111],
               [-0.11111111, -0.33333334]], dtype=float32)
    """

    _validate_raster(green_agg, func_name='mndwi', name='green_agg')
    _validate_raster(swir_agg, func_name='mndwi', name='swir_agg')

    validate_arrays(green_agg, swir_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(green_agg)(green_agg.data.astype('f4'), swir_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=green_agg.coords,
                     dims=green_agg.dims,
                     attrs=green_agg.attrs)


@ngjit
def _normalized_ratio_cpu(arr1, arr2):
    out = np.full(arr1.shape, np.nan, dtype=np.float32)
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
    out = np.full(nir_data.shape, np.nan, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            numerator = nir - red
            soma = nir + red + soil_factor
            if soma != 0.0:
                out[y, x] = (numerator / soma) * (1.0 + soil_factor)

    return out


@cuda.jit
def _savi_gpu(nir_data, red_data, soil_factor, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        red = red_data[y, x]
        numerator = nir - red
        soma = nir + red + soil_factor
        if soma != 0.0:
            out[y, x] = (numerator / soma) * (1.0 + soil_factor)


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
@supports_dataset_bands(nir='nir_agg', red='red_agg')
def savi(nir_agg: xr.DataArray,
         red_agg: xr.DataArray,
         soil_factor: float = 1.0,
         name: str = 'savi'):
    """
    Computes Soil Adjusted Vegetation Index (SAVI). Used to determine
    if a cell contains living vegetation while minimizing soil
    brightness.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array of red band data.
    soil_factor : float, default=1.0
        soil adjustment factor between -1.0 and 1.0.
        When set to zero, savi will return the same as ndvi.
    name : str, default='savi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        savi(ds, nir='B8', red='B4')

    Returns
    -------
    savi_agg : xr.DataArray of same type as inputs
        2D array of  savi values.
        All other input attributes are preserved.

    References
    ----------
        - ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/003442578890106X # noqa

    Examples
    --------
    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> nir = data['NIR']
        >>> red = data['Red']
        >>> from xrspatial.multispectral import savi
        >>> # Generate SAVI Aggregate Array
        >>> savi_agg = savi(nir_agg=nir, red_agg=red)
        >>> nir.plot(aspect=2, size=4)
        >>> red.plot(aspect=2, size=4)
        >>> savi_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> print(nir[y1:y2, x1:x2].data)
        [[1519. 1504. 1530. 1589.]
         [1491. 1473. 1542. 1609.]
         [1479. 1461. 1592. 1653.]]
        >>> print(red[y1:y2, x1:x2].data)
        [[1327. 1329. 1363. 1392.]
         [1309. 1331. 1423. 1424.]
         [1293. 1337. 1455. 1414.]]
        >>> print(savi_agg[y1:y2, x1:x2].data)
        [[0.0337197  0.03087509 0.0288528  0.03303152]
         [0.0324884  0.02531194 0.02006069 0.03048781]
         [0.03353769 0.02215077 0.02247375 0.03895046]]
    """

    _validate_raster(nir_agg, func_name='savi', name='nir_agg')
    _validate_raster(red_agg, func_name='savi', name='red_agg')

    validate_arrays(red_agg, nir_agg)

    if not -1.0 <= soil_factor <= 1.0:
        raise ValueError("soil factor must be between [-1.0, 1.0]")

    mapper = ArrayTypeFunctionMapping(numpy_func=_savi_cpu,
                                      dask_func=_savi_dask,
                                      cupy_func=_savi_cupy,
                                      dask_cupy_func=_savi_dask_cupy)

    out = mapper(red_agg)(nir_agg.data.astype('f4'), red_agg.data.astype('f4'), soil_factor)

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# SIPI ----------
@ngjit
def _sipi_cpu(nir_data, red_data, blue_data):
    out = np.full(nir_data.shape, np.nan, dtype=np.float32)
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


@supports_dataset_bands(nir='nir_agg', red='red_agg', blue='blue_agg')
def sipi(nir_agg: xr.DataArray,
         red_agg: xr.DataArray,
         blue_agg: xr.DataArray,
         name='sipi'):
    """
    Computes Structure Insensitive Pigment Index which helpful in early
    disease detection in vegetation.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array of red band data.
    blue_agg : xr.DataArray
        2D array of blue band data.
    name: str, default='sipi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        sipi(ds, nir='B8', red='B4', blue='B2')

    Returns
    -------
     sipi_agg : xr.DataArray of same type as inputs
        2D array of sipi values.
        All other input attributes are preserved.

    References
    ----------
        - Wikipedia: https://en.wikipedia.org/wiki/Enhanced_vegetation_index

    Examples
    --------
    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> nir = data['NIR']
        >>> red = data['Red']
        >>> blue = data['Blue']
        >>> from xrspatial.multispectral import sipi
        >>> # Generate SIPI Aggregate Array
        >>> sipi_agg = sipi(nir_agg=nir, red_agg=red, blue_agg=blue)
        >>> nir.plot(cmap='Greys', aspect=2, size=4)
        >>> red.plot(aspect=2, size=4)
        >>> blue.plot(aspect=2, size=4)
        >>> sipi_agg.plot(aspect=2, size=4)

    .. sourcecode:: python

        >>> y1, x1, y2, x2 = 100, 100, 103, 104
        >>> print(nir[y1:y2, x1:x2].data)
        [[1519. 1504. 1530. 1589.]
         [1491. 1473. 1542. 1609.]
         [1479. 1461. 1592. 1653.]]
        >>> print(red[y1:y2, x1:x2].data)
        [[1327. 1329. 1363. 1392.]
         [1309. 1331. 1423. 1424.]
         [1293. 1337. 1455. 1414.]]
        >>> print(blue[y1:y2, x1:x2].data)
        [[1281. 1270. 1254. 1297.]
         [1241. 1249. 1280. 1309.]
         [1239. 1257. 1322. 1329.]]
        >>> print(sipi_agg[y1:y2, x1:x2].data)
        [[1.2395834 1.3371428 1.6526946 1.4822335]
         [1.3736264 1.5774648 2.2016807 1.6216216]
         [1.2903225 1.6451613 1.9708029 1.3556485]]
    """

    _validate_raster(nir_agg, func_name='sipi', name='nir_agg')
    _validate_raster(red_agg, func_name='sipi', name='red_agg')
    _validate_raster(blue_agg, func_name='sipi', name='blue_agg')

    validate_arrays(red_agg, nir_agg, blue_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_sipi_cpu,
                                      dask_func=_sipi_dask,
                                      cupy_func=_sipi_cupy,
                                      dask_cupy_func=_sipi_dask_cupy)

    out = mapper(red_agg)(
        nir_agg.data.astype('f4'), red_agg.data.astype('f4'), blue_agg.data.astype('f4')
    )

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# EBBI ----------
@ngjit
def _ebbi_cpu(red_data, swir_data, tir_data):
    out = np.full(red_data.shape, np.nan, dtype=np.float32)
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
        denominator = 10.0 * sqrt(swir + tir)
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


@supports_dataset_bands(red='red_agg', swir='swir_agg', tir='tir_agg')
def ebbi(red_agg: xr.DataArray,
         swir_agg: xr.DataArray,
         tir_agg: xr.DataArray,
         name='ebbi'):
    """
    Computes Enhanced Built-Up and Bareness Index (EBBI) which allows
    for easily distinguishing between built-up and bare land areas.

    Parameters
    ----------
    red_agg : xr.DataArray
        2D array of red band data.
    swir_agg : xr.DataArray
        2D array of shortwave infrared band data.
    tir_agg: xr.DataArray
        2D array of thermal infrared band data.
    name: str, default='ebbi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        ebbi(ds, red='B4', swir='B11', tir='B10')

    Returns
    -------
    ebbi_agg = xr.DataArray of same type as inputs
        2D array of ebbi values.
        All other input attributes are preserved

    References
    ----------
        - rdrr: https://rdrr.io/cran/LSRS/man/EBBI.html

    Examples
    --------
    .. sourcecode:: python

        >>> # Imports
        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.multispectral import ebbi
        >>> # Create Sample Band Data, RED band
        >>> np.random.seed(1)
        >>> red_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
        >>> height, width = red_agg.shape
        >>> _lat = np.linspace(0, height - 1, height)
        >>> _lon = np.linspace(0, width - 1, width)
        >>> red_agg["lat"] = _lat
        >>> red_agg["lon"] = _lon
        >>> # SWIR band
        >>> np.random.seed(5)
        >>> swir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
        >>> height, width = swir_agg.shape
        >>> _lat = np.linspace(0, height - 1, height)
        >>> _lon = np.linspace(0, width - 1, width)
        >>> swir_agg["lat"] = _lat
        >>> swir_agg["lon"] = _lon
        >>> # TIR band
        >>> np.random.seed(6)
        >>> tir_agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
        >>> height, width = tir_agg.shape
        >>> _lat = np.linspace(0, height - 1, height)
        >>> _lon = np.linspace(0, width - 1, width)
        >>> tir_agg["lat"] = _lat
        >>> tir_agg["lon"] = _lon

        >>> print(red_agg, swir_agg, tir_agg)
        <xarray.DataArray (lat: 4, lon: 4)>
        array([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01], # noqa
                [1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01], # noqa
                [3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01], # noqa
                [2.04452250e-01, 8.78117436e-01, 2.73875932e-02, 6.70467510e-01]]) # noqa
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

        >>> # Create EBBI DataArray
        >>> ebbi_agg = ebbi(red_agg, swir_agg, tir_agg)
        >>> print(ebbi_agg)
        <xarray.DataArray 'ebbi' (lat: 4, lon: 4)>
        array([[-2.43983486, -2.58194492,  3.97432599, -0.42291921],
                [-0.11444052,  0.96786363,  0.59269999,  0.42374096],
                [ 0.61379897, -0.23840436, -0.05598088,  0.95193251],
                [ 1.32393891,  0.41574839,  0.72484653, -0.80669034]])
        Coordinates:
            * lat      (lat) float64 0.0 1.0 2.0 3.0
            * lon      (lon) float64 0.0 1.0 2.0 3.0
    """

    _validate_raster(red_agg, func_name='ebbi', name='red_agg')
    _validate_raster(swir_agg, func_name='ebbi', name='swir_agg')
    _validate_raster(tir_agg, func_name='ebbi', name='tir_agg')

    validate_arrays(red_agg, swir_agg, tir_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_ebbi_cpu,
                                      dask_func=_ebbi_dask,
                                      cupy_func=_ebbi_cupy,
                                      dask_cupy_func=_ebbi_dask_cupy)

    out = mapper(red_agg)(
        red_agg.data.astype('f4'), swir_agg.data.astype('f4'), tir_agg.data.astype('f4')
    )

    return DataArray(out,
                     name=name,
                     coords=red_agg.coords,
                     dims=red_agg.dims,
                     attrs=red_agg.attrs)


@ngjit
def _normalize_data_cpu(data, min_val, max_val, pixel_max, c, th):
    out = np.full(data.shape, np.nan, dtype=np.float32)

    range_val = max_val - min_val
    rows, cols = data.shape

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


def _normalize_data_numpy(data, pixel_max, c, th):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    out = _normalize_data_cpu(
        data, min_val, max_val, pixel_max, c, th
    )
    return out


def _normalize_data_dask(data, pixel_max, c, th):
    min_val = da.nanmin(data)
    max_val = da.nanmax(data)
    out = da.map_blocks(
        _normalize_data_cpu, data, min_val, max_val, pixel_max,
        c, th, meta=np.array(())
    )
    return out


def _normalize_data_cupy(data, pixel_max, c, th):
    min_val = cupy.nanmin(data)
    max_val = cupy.nanmax(data)
    range_val = max_val - min_val
    out = cupy.full(data.shape, cupy.nan, dtype=cupy.float32)
    if range_val != 0:
        norm = (data - min_val) / range_val
        norm = 1 / (1 + cupy.exp(c * (th - norm)))
        out = norm * pixel_max
    return out


def _normalize_data_cupy_block(data, min_val, max_val, pixel_max, c, th):
    range_val = max_val - min_val
    out = cupy.full(data.shape, cupy.nan, dtype=cupy.float32)
    if range_val != 0:
        norm = (data - min_val) / range_val
        norm = 1 / (1 + cupy.exp(c * (th - norm)))
        out = norm * pixel_max
    return out


def _normalize_data_dask_cupy(data, pixel_max, c, th):
    min_val = da.nanmin(data)
    max_val = da.nanmax(data)
    out = da.map_blocks(
        _normalize_data_cupy_block, data, min_val, max_val, pixel_max,
        c, th, meta=cupy.array(())
    )
    return out


def _normalize_data(agg, pixel_max, c, th):
    mapper = ArrayTypeFunctionMapping(numpy_func=_normalize_data_numpy,
                                      dask_func=_normalize_data_dask,
                                      cupy_func=_normalize_data_cupy,
                                      dask_cupy_func=_normalize_data_dask_cupy)
    out = mapper(agg)(agg.data.astype('f4'), pixel_max, c, th)
    return out


def _true_color_numpy(r, g, b, nodata, c, th):
    a = np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)

    h, w = r.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)

    pixel_max = 255
    out[:, :, 0] = (_normalize_data(r, pixel_max, c, th)).astype(np.uint8)
    out[:, :, 1] = (_normalize_data(g, pixel_max, c, th)).astype(np.uint8)
    out[:, :, 2] = (_normalize_data(b, pixel_max, c, th)).astype(np.uint8)
    out[:, :, 3] = a.astype(np.uint8)
    return out


def _true_color_dask(r, g, b, nodata, c, th):
    pixel_max = 255

    alpha = da.where(
        da.logical_or(da.isnan(r), r <= nodata), 0, pixel_max
    ).astype(np.uint8)

    red = (_normalize_data(r, pixel_max, c, th)).astype(np.uint8)
    green = (_normalize_data(g, pixel_max, c, th)).astype(np.uint8)
    blue = (_normalize_data(b, pixel_max, c, th)).astype(np.uint8)

    out = da.stack([red, green, blue, alpha], axis=-1)
    return out


def _true_color_cupy(r, g, b, nodata, c, th):
    pixel_max = 255
    r_data = r.data
    a = cupy.where(
        cupy.logical_or(cupy.isnan(r_data), r_data <= nodata), 0, pixel_max
    ).astype(cupy.uint8)
    red = (_normalize_data(r, pixel_max, c, th)).astype(cupy.uint8)
    green = (_normalize_data(g, pixel_max, c, th)).astype(cupy.uint8)
    blue = (_normalize_data(b, pixel_max, c, th)).astype(cupy.uint8)
    out = cupy.stack([red, green, blue, a], axis=-1)
    return out


def _true_color_dask_cupy(r, g, b, nodata, c, th):
    pixel_max = 255
    r_data = r.data
    alpha = da.where(
        da.logical_or(da.isnan(r_data), r_data <= nodata), 0, pixel_max
    ).astype(cupy.uint8)
    red = (_normalize_data(r, pixel_max, c, th)).astype(cupy.uint8)
    green = (_normalize_data(g, pixel_max, c, th)).astype(cupy.uint8)
    blue = (_normalize_data(b, pixel_max, c, th)).astype(cupy.uint8)
    out = da.stack([red, green, blue, alpha], axis=-1)
    return out


def true_color(r, g, b, nodata=1, c=10.0, th=0.125, name='true_color'):
    """
    Create true color composite from a combination of red, green and
    blue bands satellite images.

    A sigmoid function will be used to improve the contrast of output image.
    The function is defined as
    ``normalized_pixel = 1 / (1 + np.exp(c * (th - normalized_pixel)))``
    where ``c`` and ``th`` are contrast and brightness controlling parameters.

    Parameters
    ----------
    r : xarray.DataArray
        2D array of red band data.
    g : xarray.DataArray
        2D array of green band data.
    b : xarray.DataArray
        2D array of blue band data.
    nodata : int, float numeric value
        Nodata value of input DataArrays.
    c : float, default=10
        Contrast and brighness controlling parameter for output image.
    th : float, default=0.125
        Contrast and brighness controlling parameter for output image.
    name : str, default='true_color'
        Name of output DataArray.

    Returns
    -------
    true_color_agg : xarray.DataArray of the same type as inputs
        3D array true color image with dims of [y, x, band].
        All output attributes are copied from red band image.

    Examples
    --------
    .. plot::
       :include-source:

        >>> from xrspatial.datasets import get_data
        >>> data = get_data('sentinel-2')  # Open Example Data
        >>> red = data['Red']
        >>> green = data['Green']
        >>> blue = data['Blue']
        >>> from xrspatial.multispectral import true_color
        >>> # Generate true color image
        >>> true_color_img = true_color(r=red, g=green, b=blue)
        >>> true_color_img.plot.imshow()
    """

    _validate_raster(r, func_name='true_color', name='r')
    _validate_raster(g, func_name='true_color', name='g')
    _validate_raster(b, func_name='true_color', name='b')

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_true_color_numpy,
        dask_func=_true_color_dask,
        cupy_func=_true_color_cupy,
        dask_cupy_func=_true_color_dask_cupy,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = mapper(r)(r, g, b, nodata, c, th)

    # TODO: output metadata: coords, dims, attrs
    _dims = ['y', 'x', 'band']
    _attrs = r.attrs
    _coords = {'y': r['y'],
               'x': r['x'],
               'band': [0, 1, 2, 3]}

    return DataArray(
        out,
        name=name,
        dims=_dims,
        coords=_coords,
        attrs=_attrs,
    )


# NDSI ----------
@supports_dataset_bands(green='green_agg', swir1='swir1_agg')
def ndsi(green_agg: xr.DataArray,
         swir1_agg: xr.DataArray,
         name='ndsi'):
    """
    Computes Normalized Difference Snow Index (NDSI).

    NDSI separates snow and ice from clouds and other bright surfaces
    by exploiting the high reflectance of snow in the green band and
    low reflectance in the shortwave infrared.

    Parameters
    ----------
    green_agg : xr.DataArray
        2D array of green band data.
        (Landsat 8: Band 3)
        (Sentinel-2: Band 3)
    swir1_agg : xr.DataArray
        2D array of shortwave infrared band data.
        (Landsat 8: Band 6)
        (Sentinel-2: Band 11)
    name : str, default='ndsi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        ndsi(ds, green='B3', swir1='B11')

    Returns
    -------
    ndsi_agg : xr.DataArray of same type as inputs
        2D array of ndsi values in the range [-1, 1].
        All other input attributes are preserved.

    References
    ----------
        - Hall, D.K., Riggs, G.A. and Salomonson, V.V., 1995.
          Development of methods for mapping global snow cover using
          moderate resolution imaging spectroradiometer data.
          Remote Sensing of Environment, 54(2), pp.127-140.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.multispectral import ndsi
        >>> green = xr.DataArray(np.array([[600., 500.], [400., 300.]]))
        >>> swir1 = xr.DataArray(np.array([[100., 200.], [300., 400.]]))
        >>> ndsi(green, swir1).values
        array([[ 0.71428573,  0.42857143],
               [ 0.14285715, -0.14285715]], dtype=float32)
    """

    _validate_raster(green_agg, func_name='ndsi', name='green_agg')
    _validate_raster(swir1_agg, func_name='ndsi', name='swir1_agg')

    validate_arrays(green_agg, swir1_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(green_agg)(green_agg.data.astype('f4'), swir1_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=green_agg.coords,
                     dims=green_agg.dims,
                     attrs=green_agg.attrs)


# NDBI ----------
@supports_dataset_bands(swir1='swir1_agg', nir='nir_agg')
def ndbi(swir1_agg: xr.DataArray,
         nir_agg: xr.DataArray,
         name='ndbi'):
    """
    Computes Normalized Difference Built-up Index (NDBI).

    NDBI picks out built-up and urban areas by exploiting the higher
    reflectance of impervious surfaces in SWIR relative to NIR.

    Parameters
    ----------
    swir1_agg : xr.DataArray
        2D array of shortwave infrared band data.
        (Landsat 8: Band 6)
        (Sentinel-2: Band 11)
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
        (Landsat 8: Band 5)
        (Sentinel-2: Band 8)
    name : str, default='ndbi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        ndbi(ds, swir1='B11', nir='B8')

    Returns
    -------
    ndbi_agg : xr.DataArray of same type as inputs
        2D array of ndbi values in the range [-1, 1].
        All other input attributes are preserved.

    References
    ----------
        - Zha, Y., Gao, J. and Ni, S., 2003. Use of normalized
          difference built-up index in automatically mapping urban
          areas from TM imagery. International Journal of Remote
          Sensing, 24(3), pp.583-594.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.multispectral import ndbi
        >>> swir1 = xr.DataArray(np.array([[600., 500.], [400., 300.]]))
        >>> nir = xr.DataArray(np.array([[300., 400.], [500., 600.]]))
        >>> ndbi(swir1, nir).values
        array([[ 0.33333334,  0.11111111],
               [-0.11111111, -0.33333334]], dtype=float32)
    """

    _validate_raster(swir1_agg, func_name='ndbi', name='swir1_agg')
    _validate_raster(nir_agg, func_name='ndbi', name='nir_agg')

    validate_arrays(swir1_agg, nir_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_normalized_ratio_cpu,
        dask_func=_run_normalized_ratio_dask,
        cupy_func=_run_normalized_ratio_cupy,
        dask_cupy_func=_run_normalized_ratio_dask_cupy,
    )

    out = mapper(swir1_agg)(swir1_agg.data.astype('f4'), nir_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=swir1_agg.coords,
                     dims=swir1_agg.dims,
                     attrs=swir1_agg.attrs)


# BAI ----------
@ngjit
def _bai_cpu(red_data, nir_data):
    out = np.full(red_data.shape, np.nan, dtype=np.float32)
    rows, cols = red_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            red = red_data[y, x]
            nir = nir_data[y, x]
            dr = np.float32(0.1) - red
            dn = np.float32(0.06) - nir
            denominator = dr * dr + dn * dn
            if denominator != 0.0:
                out[y, x] = np.float32(1.0) / denominator
    return out


@cuda.jit
def _bai_gpu(red_data, nir_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        red = red_data[y, x]
        nir = nir_data[y, x]
        dr = nb.float32(0.1) - red
        dn = nb.float32(0.06) - nir
        denominator = dr * dr + dn * dn
        if denominator != 0.0:
            out[y, x] = nb.float32(1.0) / denominator


def _bai_dask(red_data, nir_data):
    out = da.map_blocks(_bai_cpu, red_data, nir_data,
                        meta=np.array(()))
    return out


def _bai_cupy(red_data, nir_data):
    griddim, blockdim = cuda_args(red_data.shape)
    out = cupy.empty(red_data.shape, dtype='f4')
    out[:] = cupy.nan
    _bai_gpu[griddim, blockdim](red_data, nir_data, out)
    return out


def _bai_dask_cupy(red_data, nir_data):
    out = da.map_blocks(_bai_cupy, red_data, nir_data,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


@supports_dataset_bands(red='red_agg', nir='nir_agg')
def bai(red_agg: xr.DataArray,
        nir_agg: xr.DataArray,
        name='bai'):
    """
    Computes Burn Area Index (BAI).

    BAI measures the spectral distance of each pixel to a charcoal
    reflectance point (red=0.1, NIR=0.06). Higher values indicate
    recently burned areas. Unlike NBR, BAI works on the raw
    reflectance values rather than a normalized difference.

    Input bands should be in reflectance units (0-1 range). If your
    data is in DN or scaled integers, divide by the appropriate scale
    factor first.

    Parameters
    ----------
    red_agg : xr.DataArray
        2D array of red band reflectance data (0-1 range).
    nir_agg : xr.DataArray
        2D array of near-infrared band reflectance data (0-1 range).
    name : str, default='bai'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        bai(ds, red='B4', nir='B8')

    Returns
    -------
    bai_agg : xr.DataArray of same type as inputs
        2D array of bai values. Higher values indicate burned areas.
        All other input attributes are preserved.

    References
    ----------
        - Chuvieco, E., Martin, M.P. and Palacios, A., 2002.
          Assessment of different spectral conditions for the
          detection of burned areas with Landsat TM data.
          International Journal of Remote Sensing, 23(1), pp.71-85.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.multispectral import bai
        >>> red = xr.DataArray(np.array([[0.1, 0.2], [0.3, 0.05]]))
        >>> nir = xr.DataArray(np.array([[0.06, 0.3], [0.4, 0.02]]))
        >>> bai(red, nir).values  # pixel (0,0) is at charcoal point
        array([[       inf, 0.01686341, 0.00858369, 1111.1111  ]],
              dtype=float32)
    """

    _validate_raster(red_agg, func_name='bai', name='red_agg')
    _validate_raster(nir_agg, func_name='bai', name='nir_agg')

    validate_arrays(red_agg, nir_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_bai_cpu,
                                      dask_func=_bai_dask,
                                      cupy_func=_bai_cupy,
                                      dask_cupy_func=_bai_dask_cupy)

    out = mapper(red_agg)(red_agg.data.astype('f4'), nir_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=red_agg.coords,
                     dims=red_agg.dims,
                     attrs=red_agg.attrs)


# MSAVI2 ----------
@ngjit
def _msavi2_cpu(nir_data, red_data):
    out = np.full(nir_data.shape, np.nan, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            term = (np.float32(2.0) * nir + np.float32(1.0))
            discriminant = term * term - np.float32(8.0) * (nir - red)
            if discriminant >= 0.0:
                out[y, x] = (term - np.sqrt(discriminant)) / np.float32(2.0)
    return out


@cuda.jit
def _msavi2_gpu(nir_data, red_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        red = red_data[y, x]
        term = nb.float32(2.0) * nir + nb.float32(1.0)
        discriminant = term * term - nb.float32(8.0) * (nir - red)
        if discriminant >= nb.float32(0.0):
            out[y, x] = (term - sqrt(discriminant)) / nb.float32(2.0)


def _msavi2_dask(nir_data, red_data):
    out = da.map_blocks(_msavi2_cpu, nir_data, red_data,
                        meta=np.array(()))
    return out


def _msavi2_cupy(nir_data, red_data):
    griddim, blockdim = cuda_args(nir_data.shape)
    out = cupy.empty(nir_data.shape, dtype='f4')
    out[:] = cupy.nan
    _msavi2_gpu[griddim, blockdim](nir_data, red_data, out)
    return out


def _msavi2_dask_cupy(nir_data, red_data):
    out = da.map_blocks(_msavi2_cupy, nir_data, red_data,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


@supports_dataset_bands(nir='nir_agg', red='red_agg')
def msavi2(nir_agg: xr.DataArray,
           red_agg: xr.DataArray,
           name='msavi2'):
    """
    Computes Modified Soil Adjusted Vegetation Index (MSAVI2).

    MSAVI2 is a self-adjusting vegetation index that does not require
    an empirical soil-brightness correction factor (L). It produces
    less noisy results than SAVI in areas with sparse vegetation and
    exposed soil.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array of red band data.
    name : str, default='msavi2'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        msavi2(ds, nir='B8', red='B4')

    Returns
    -------
    msavi2_agg : xr.DataArray of same type as inputs
        2D array of msavi2 values.
        All other input attributes are preserved.

    References
    ----------
        - Qi, J., Chehbouni, A., Huete, A.R., Kerr, Y.H. and
          Sorooshian, S., 1994. A modified soil adjusted vegetation
          index. Remote Sensing of Environment, 48(2), pp.119-126.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.multispectral import msavi2
        >>> nir = xr.DataArray(np.array([[0.5, 0.3], [0.1, 0.4]]))
        >>> red = xr.DataArray(np.array([[0.1, 0.2], [0.05, 0.3]]))
        >>> msavi2(nir, red).values
        array([[0.38729835, 0.08284271, 0.0248457 , 0.08284271]],
              dtype=float32)
    """

    _validate_raster(nir_agg, func_name='msavi2', name='nir_agg')
    _validate_raster(red_agg, func_name='msavi2', name='red_agg')

    validate_arrays(nir_agg, red_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_msavi2_cpu,
                                      dask_func=_msavi2_dask,
                                      cupy_func=_msavi2_cupy,
                                      dask_cupy_func=_msavi2_dask_cupy)

    out = mapper(nir_agg)(nir_agg.data.astype('f4'), red_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)


# OSAVI ----------
@ngjit
def _osavi_cpu(nir_data, red_data):
    out = np.full(nir_data.shape, np.nan, dtype=np.float32)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]
            numerator = nir - red
            denominator = nir + red + np.float32(0.16)
            if denominator != 0.0:
                out[y, x] = numerator / denominator
    return out


@cuda.jit
def _osavi_gpu(nir_data, red_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        nir = nir_data[y, x]
        red = red_data[y, x]
        numerator = nir - red
        denominator = nir + red + nb.float32(0.16)
        if denominator != 0.0:
            out[y, x] = numerator / denominator


def _osavi_dask(nir_data, red_data):
    out = da.map_blocks(_osavi_cpu, nir_data, red_data,
                        meta=np.array(()))
    return out


def _osavi_cupy(nir_data, red_data):
    griddim, blockdim = cuda_args(nir_data.shape)
    out = cupy.empty(nir_data.shape, dtype='f4')
    out[:] = cupy.nan
    _osavi_gpu[griddim, blockdim](nir_data, red_data, out)
    return out


def _osavi_dask_cupy(nir_data, red_data):
    out = da.map_blocks(_osavi_cupy, nir_data, red_data,
                        dtype=cupy.float32, meta=cupy.array(()))
    return out


@supports_dataset_bands(nir='nir_agg', red='red_agg')
def osavi(nir_agg: xr.DataArray,
          red_agg: xr.DataArray,
          name='osavi'):
    """
    Computes Optimized Soil Adjusted Vegetation Index (OSAVI).

    OSAVI uses a fixed soil-brightness correction factor of L=0.16,
    chosen to work well across a range of soil conditions without
    requiring per-scene tuning. It performs best in areas with sparse
    to moderate vegetation cover.

    Parameters
    ----------
    nir_agg : xr.DataArray
        2D array of near-infrared band data.
    red_agg : xr.DataArray
        2D array of red band data.
    name : str, default='osavi'
        Name of output DataArray.

    Alternatively, a single ``xr.Dataset`` may be passed as the first
    argument with keyword arguments mapping band names to Dataset
    variables. For example::

        osavi(ds, nir='B8', red='B4')

    Returns
    -------
    osavi_agg : xr.DataArray of same type as inputs
        2D array of osavi values.
        All other input attributes are preserved.

    References
    ----------
        - Rondeaux, G., Steven, M. and Baret, F., 1996.
          Optimization of soil-adjusted vegetation indices.
          Remote Sensing of Environment, 55(2), pp.95-107.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.multispectral import osavi
        >>> nir = xr.DataArray(np.array([[0.5, 0.3], [0.1, 0.4]]))
        >>> red = xr.DataArray(np.array([[0.1, 0.2], [0.05, 0.3]]))
        >>> osavi(nir, red).values
        array([[0.5263158 , 0.15151516, 0.16129032, 0.11627907]],
              dtype=float32)
    """

    _validate_raster(nir_agg, func_name='osavi', name='nir_agg')
    _validate_raster(red_agg, func_name='osavi', name='red_agg')

    validate_arrays(nir_agg, red_agg)

    mapper = ArrayTypeFunctionMapping(numpy_func=_osavi_cpu,
                                      dask_func=_osavi_dask,
                                      cupy_func=_osavi_cupy,
                                      dask_cupy_func=_osavi_dask_cupy)

    out = mapper(nir_agg)(nir_agg.data.astype('f4'), red_agg.data.astype('f4'))

    return DataArray(out,
                     name=name,
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)
