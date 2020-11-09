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
    """Computes Atmospherically Resistant Vegetation Index

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data

    red_agg : DataArray
        red band data

    blue_agg : DataArray
        blue band data

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
    https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=03667
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
    """Computes Enhanced Vegetation Index

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data

    red_agg : DataArray
        red band data

    blue_agg : DataArray
        blue band data

    c1 : float (default=6.0)
        first coefficient of the aerosol resistance term

    c2 : float (default=7.5)
        second coefficients of the aerosol resistance term

    soil_factor : float (default=1.0)
      soil adjustment factor between -1.0 and 1.0.

    gain : float (default=2.5)
      amplitude adjustment factor

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
    """Computes Green Chlorophyll Index

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
    """Computes Normalized Burn Ratio

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band

    swir_agg : DataArray
        shortwave infrared band
        (Landsat 4-7: Band 6)
        (Landsat 8: Band 7)

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
    https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-burn-ratio
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
