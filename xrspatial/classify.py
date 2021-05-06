from functools import partial
import xarray as xr

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

import datashader.transfer_functions as tf
import numpy as np
from datashader.colors import rgb
from xarray import DataArray

from numba import cuda
import dask.array as da

from numpy.random import RandomState

from xrspatial.utils import cuda_args
from xrspatial.utils import has_cuda
from xrspatial.utils import ngjit
from xrspatial.utils import is_cupy_backed

from typing import List, Optional


import warnings


def color_values(agg, color_key, alpha=255):
    def _convert_color(c):
        r, g, b = rgb(c)
        return np.array([r, g, b, alpha]).astype(np.uint8).view(np.uint32)[0]

    _converted_colors = {k: _convert_color(v) for k, v in color_key.items()}
    f = np.vectorize(lambda v: _converted_colors.get(v, 0))
    return tf.Image(f(agg.data))


@ngjit
def _binary(data, values):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            if np.any(values == data[y, x]):
                out[y, x] = 1
            else:
                out[y, x] = 0
    return out


def binary(agg, values, name='binary'):

    if isinstance(values, (list, tuple)):
        vals = np.array(values)
    else:
        vals = values

    return DataArray(_binary(agg.data, vals),
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


@ngjit
def _cpu_bin(data, bins, new_values):
    out = np.zeros(data.shape, dtype=np.float32)
    out[:, :] = np.nan
    rows, cols = data.shape
    nbins = len(bins)
    for y in range(0, rows):
        for x in range(0, cols):
            val = data[y, x]
            val_bin = -1

            # find bin
            for b in range(0, nbins):

                # first bin
                if b == 0:
                    if val <= bins[b]:
                        val_bin = b
                        break
                else:
                    if val > bins[b - 1] and val <= bins[b]:
                        val_bin = b
                        break

            if val_bin > -1:
                out[y, x] = new_values[val_bin]
            else:
                out[y, x] = np.nan

    return out


def _run_numpy_bin(data, bins, new_values):
    out = _cpu_bin(data, bins, new_values)
    return out


def _run_dask_numpy_bin(data, bins, new_values):
    _func = partial(_run_numpy_bin,
                    bins=bins,
                    new_values=new_values)

    out = data.map_blocks(_func)
    return out


@cuda.jit(device=True)
def _gpu_bin(data, bins, new_values):
    nbins = len(bins)
    val = data[0, 0]
    val_bin = -1

    # find bin
    for b in range(0, nbins):

        # first bin
        if b == 0:
            if val <= bins[b]:
                val_bin = b
                break
        else:
            if val > bins[b - 1] and val <= bins[b]:
                val_bin = b
                break

    if val_bin > -1:
        out = new_values[val_bin]
    else:
        out = np.nan

    return out


@cuda.jit
def _run_gpu_bin(data, bins, new_values, out):
    i, j = cuda.grid(2)
    if (i >= 0 and i < out.shape[0] and j >= 0 and j < out.shape[1]):
        out[i, j] = _gpu_bin(data[i:i+1, j:j+1], bins, new_values)


def _run_cupy_bin(data, bins_cupy, new_values_cupy):
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    _run_gpu_bin[griddim, blockdim](data,
                                    bins_cupy,
                                    new_values_cupy,
                                    out)
    return out


def _run_dask_cupy_bin(data, bins_cupy, new_values_cupy):
    out = data.map_blocks(lambda da:
                          _run_cupy_bin(da, bins_cupy, new_values_cupy),
                          meta=cupy.array(()))
    return out


def _bin(data, bins, new_values):
    # numpy case
    if isinstance(data, np.ndarray):
        out = _run_numpy_bin(data, np.asarray(bins), np.asarray(new_values))

    # cupy case
    elif has_cuda() and isinstance(data, cupy.ndarray):
        bins_cupy = cupy.asarray(bins, dtype='f4')
        new_values_cupy = cupy.asarray(new_values, dtype='f4')
        out = _run_cupy_bin(data, bins_cupy, new_values_cupy)

    # dask + cupy case
    elif has_cuda() and isinstance(data, da.Array) and \
            type(data._meta).__module__.split('.')[0] == 'cupy':
        bins_cupy = cupy.asarray(bins, dtype='f4')
        new_values_cupy = cupy.asarray(new_values, dtype='f4')
        out = _run_dask_cupy_bin(data, bins_cupy, new_values_cupy)

    # dask + numpy case
    elif isinstance(data, da.Array):
        out = _run_dask_numpy_bin(data, np.asarray(bins),
                                  np.asarray(new_values))

    return out


def reclassify(agg: xr.DataArray,
               bins: List[int],
               new_values: List[int],
               name: Optional[str] = 'reclassify') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on user
    defined bins.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    bins : array-like object
        Values or ranges of values to be changed.
    new_values : array-like object
        New values for each bin.
    name : str, default='reclassify'
        Name of output aggregate array.

    Returns
    -------
    reclass_agg : xarray.DataArray, of the same type as `agg`
        2D aggregate array of reclassified allocations.
        All other input attributes are preserved.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.classify import reclassify

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))

        # Generate Example Terrain
        terrain_agg = generate_terrain(canvas = cvs)

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            },
        )

        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Create Reclassified Aggregate Array
        bins = list(range(0, 3000))
        new_vals = list(range(1000, 4000))
        reclass_agg = reclassify(agg = terrain_agg,
                                 bins = bins,
                                 new_values = new_vals,
                                 name = 'Elevation')

        # Edit Attributes
        reclass_agg = reclass_agg.assign_attrs(
            {'Description': 'Example Reclassify'}
        )

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Reclassify
        reclass_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Reclassify")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02249454, 1261.94748873],
                [1285.37061171, 1282.48046696],
                [1306.02305679, 1303.40657515]])
        Coordinates:
            * lon      (lon) float64 -3.96e+06 -3.88e+06
            * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

        >>> print(reclass_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[2265., 2262.],
                [2286., 2283.],
                [2307., 2304.]], dtype=float32)
        Coordinates:
            * lon      (lon) float64 -3.96e+06 -3.88e+06
            * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Reclassify
            units:          km
            Max Elevation:  4000
    """
    if len(bins) != len(new_values):
        raise ValueError('bins and new_values mismatch.'
                         'Should have same length.')
    out = _bin(agg.data, bins, new_values)
    return DataArray(out,
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


def _run_cpu_quantile(data, k):
    w = 100.0 / k
    p = np.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    q = np.percentile(data, p)
    q = np.unique(q)
    return q


def _run_dask_numpy_quantile(data, k):
    w = 100.0 / k
    p = da.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    q = da.percentile(data.flatten(), p)
    q = da.unique(q)
    return q


def _run_cupy_quantile(data, k):
    w = 100.0 / k
    p = cupy.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    q = cupy.percentile(data, p)
    q = cupy.unique(q)
    return q


def _run_dask_cupy_quantile(data, k):
    msg = 'Currently percentile calculation has not' \
          'been supported for Dask array backed by CuPy.' \
          'See issue at https://github.com/dask/dask/issues/6942'
    raise NotImplementedError(msg)


def _quantile(agg, k):
    # numpy case
    if isinstance(agg.data, np.ndarray):
        q = _run_cpu_quantile(agg.data, k)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        q = _run_cupy_quantile(agg.data, k)

    # dask + cupy case
    elif has_cuda() and \
        isinstance(agg.data, cupy.ndarray) and \
            is_cupy_backed(agg):
        q = _run_dask_cupy_quantile(agg.data, k)

    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        q = _run_dask_numpy_quantile(agg.data, k)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return q


def quantile(agg: xr.DataArray,
             k: int = 4,
             name: Optional[str] = 'quantile') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on quantile
    groups of equal size.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    k : int, default=4
        Number of quantiles to be produced.
    name : str, default='quantile'
        Name of the output aggregate array.

    Returns
    -------
    quantile_agg : xarray.DataArray, of the same type as `agg`
        2D aggregate array, of quantile allocations.
        All other input attributes are preserved.

    Notes
    -----
        - Dask's percentile algorithm is approximate, while numpy's is exact.
        - This may cause some differences between results of vanilla numpy
        and dask version of the input agg. (https://github.com/dask/dask/issues/3099) # noqa

    References
    ----------
        -  PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#Quantiles # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.classify import quantile

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))

        # Generate Example Terrain
        terrain_agg = generate_terrain(canvas = cvs)

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            }
        )
        
        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Create Quantiled Aggregate Array
        quantile_agg = quantile(agg = terrain_agg, name = 'Elevation')

        # Edit Attributes
        quantile_agg = quantile_agg.assign_attrs({'Description': 'Example Quantile'})

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Quantile
        quantile_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Quantile")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02249454, 1261.94748873],
                [1285.37061171, 1282.48046696],
                [1306.02305679, 1303.40657515]])
        Coordinates:
            * lon      (lon) float64 -3.96e+06 -3.88e+06
            * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

        >>> print(quantile_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[2., 2.],
                [2., 2.],
                [2., 2.]], dtype=float32)
        Coordinates:
            * lon      (lon) float64 -3.96e+06 -3.88e+06
            * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Quantile
            units:          km
            Max Elevation:  4000
    """
    q = _quantile(agg, k)
    k_q = q.shape[0]
    if k_q < k:
        print("Quantile Warning: Not enough unique values"
              "for k classes (using {} bins)".format(k_q))
        k = k_q

    out = _bin(agg.data, bins=q, new_values=np.arange(k))

    return DataArray(out,
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


@ngjit
def _run_numpy_jenks_matrices(data, n_classes):
    n_data = data.shape[0]
    lower_class_limits = np.zeros((n_data + 1, n_classes + 1),
                                  dtype=np.float64)
    lower_class_limits[1, 1:n_classes + 1] = 1.0

    var_combinations = np.zeros((n_data + 1, n_classes + 1), dtype=np.float64)
    var_combinations[2:n_data + 1, 1:n_classes + 1] = np.inf

    nl = data.shape[0] + 1
    variance = 0.0

    for l in range(2, nl): # noqa
        sum = 0.0
        sum_squares = 0.0
        w = 0.0

        for m in range(1, l + 1):
            # `III` originally
            lower_class_limit = l - m + 1
            i4 = lower_class_limit - 1

            val = data[i4]

            # here we're estimating variance for each potential classing
            # of the data, for each potential number of classes. `w`
            # is the number of data points considered so far.
            w += 1.0

            # increase the current sum and sum-of-squares
            sum += val
            sum_squares += val * val

            # the variance at this point in the sequence is the difference
            # between the sum of squares and the total x 2, over the number
            # of samples.
            variance = sum_squares - (sum * sum) / w

            if i4 != 0:
                for j in range(2, n_classes + 1):
                    jm1 = j - 1
                    if var_combinations[l, j] >= \
                            (variance + var_combinations[i4, jm1]):
                        lower_class_limits[l, j] = lower_class_limit
                        var_combinations[l, j] = variance + \
                            var_combinations[i4, jm1]

        lower_class_limits[l, 1] = 1.
        var_combinations[l, 1] = variance

    return lower_class_limits, var_combinations


@ngjit
def _run_numpy_jenks(data, n_classes):
    # ported from existing cython implementation:
    # https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    data.sort()

    lower_class_limits, _ = _run_numpy_jenks_matrices(data, n_classes)

    k = data.shape[0]
    kclass = [0.] * (n_classes + 1)
    count_num = n_classes

    kclass[n_classes] = data[len(data) - 1]
    kclass[0] = data[0]

    while count_num > 1:
        elt = int(lower_class_limits[k][count_num] - 2)
        kclass[count_num - 1] = data[elt]
        k = int(lower_class_limits[k][count_num] - 1)
        count_num -= 1

    return kclass


def _run_numpy_natural_break(data, num_sample, k):
    num_data = data.size

    if num_sample is not None and num_sample < num_data:
        # randomly select sample from the whole dataset
        # create a pseudo random number generator
        generator = RandomState(1234567890)
        idx = [i for i in range(0, data.size)]
        generator.shuffle(idx)
        sample_idx = idx[:num_sample]
        sample_data = data.flatten()[sample_idx]
    else:
        sample_data = data.flatten()

    # warning if number of total data points to fit the model bigger than 40k
    if sample_data.size >= 40000:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.warn('natural_breaks Warning: Natural break '
                          'classification (Jenks) has a complexity of O(n^2), '
                          'your classification with {} data points may take '
                          'a long time.'.format(sample_data.size),
                          Warning)

    uv = np.unique(sample_data)
    uvk = len(uv)

    if uvk < k:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.warn('natural_breaks Warning: Not enough unique values '
                          'in data array for {} classes. '
                          'n_samples={} should be >= n_clusters={}. '
                          'Using k={} instead.'.format(k, uvk, k, uvk),
                          Warning)
        uv.sort()
        bins = uv
    else:
        centroids = _run_numpy_jenks(sample_data, k)
        bins = np.array(centroids[1:])

    out = _bin(data, bins, np.arange(uvk))
    return out


def _run_cupy_jenks_matrices(data, n_classes):
    n_data = data.shape[0]
    lower_class_limits = cupy.zeros((n_data + 1, n_classes + 1), dtype='f4')
    lower_class_limits[1, 1:n_classes + 1] = 1.0

    var_combinations = cupy.zeros((n_data + 1, n_classes + 1), dtype='f4')
    var_combinations[2:n_data + 1, 1:n_classes + 1] = cupy.inf

    nl = data.shape[0] + 1
    variance = 0.0

    for l in range(2, nl): # noqa
        sum = 0.0
        sum_squares = 0.0
        w = 0.0

        for m in range(1, l + 1):
            # `III` originally
            lower_class_limit = l - m + 1
            i4 = lower_class_limit - 1

            val = data[i4]

            # here we're estimating variance for each potential classing
            # of the data, for each potential number of classes. `w`
            # is the number of data points considered so far.
            w += 1.0

            # increase the current sum and sum-of-squares
            sum += val
            sum_squares += val * val

            # the variance at this point in the sequence is the difference
            # between the sum of squares and the total x 2, over the number
            # of samples.
            variance = sum_squares - (sum * sum) / w

            if i4 != 0:
                for j in range(2, n_classes + 1):
                    jm1 = j - 1
                    if var_combinations[l, j] >= \
                            (variance + var_combinations[i4, jm1]):
                        lower_class_limits[l, j] = lower_class_limit
                        var_combinations[l, j] = variance + \
                            var_combinations[i4, jm1]

        lower_class_limits[l, 1] = 1.
        var_combinations[l, 1] = variance

    return lower_class_limits, var_combinations


def _run_cupy_jenks(data, n_classes):
    data.sort()

    lower_class_limits, _ = _run_cupy_jenks_matrices(data, n_classes)

    k = data.shape[0]
    kclass = [0.] * (n_classes + 1)
    count_num = n_classes

    kclass[n_classes] = data[len(data) - 1]
    kclass[0] = data[0]

    while count_num > 1:
        elt = int(lower_class_limits[k][count_num] - 2)
        kclass[count_num - 1] = data[elt]
        k = int(lower_class_limits[k][count_num] - 1)
        count_num -= 1

    return kclass


def _run_cupy_natural_break(data, num_sample, k):
    num_data = data.size

    if num_sample is not None and num_sample < num_data:
        generator = cupy.random.RandomState(1234567890)
        idx = [i for i in range(0, data.size)]
        generator.shuffle(idx)
        sample_idx = idx[:num_sample]
        sample_data = data.flatten()[sample_idx]
    else:
        sample_data = data.flatten()

    # warning if number of total data points to fit the model bigger than 40k
    if sample_data.size >= 40000:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.warn('natural_breaks Warning: Natural break '
                          'classification (Jenks) has a complexity of O(n^2), '
                          'your classification with {} data points may take '
                          'a long time.'.format(sample_data.size),
                          Warning)

    uv = cupy.unique(sample_data)
    uvk = len(uv)

    if uvk < k:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.warn('natural_breaks Warning: Not enough unique values '
                          'in data array for {} classes. '
                          'n_samples={} should be >= n_clusters={}. '
                          'Using k={} instead.'.format(k, uvk, k, uvk),
                          Warning)
        uv.sort()
        bins = uv
    else:
        centroids = _run_cupy_jenks(sample_data, k)
        bins = cupy.array(centroids[1:])

    out = _bin(data, bins, cupy.arange(uvk))
    return out


def natural_breaks(agg: xr.DataArray,
                   num_sample: Optional[int] = None,
                   name: Optional[str] = 'natural_breaks',
                   k: int = 5) -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on Natural
    Breaks or K-Means clustering method. Values are grouped so that
    similar values are placed in the same group and space between
    groups is maximized.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    num_sample : int, default=None
        Number of sample data points used to fit the model.
        Natural Breaks (Jenks) classification is indeed O(n²) complexity,
        where n is the total number of data points, i.e: `agg.size`
        When n is large, we should fit the model on a small sub-sample
        of the data instead of using the whole dataset.
    k : int, default=5
        Number of classes to be produced.
    name : str, default='natural_breaks'
        Name of output aggregate.

    Returns
    -------
    natural_breaks_agg : xarray.DataArray of the same type as `agg`
        2D aggregate array of natural break allocations.
        All other input attributes are preserved.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#NaturalBreaks # noqa
        - jenks: https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.classify import natural_breaks

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))

        # Generate Example Terrain
        terrain_agg = generate_terrain(canvas = cvs)

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            }
        )
        
        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Create Natural Breaks Aggregate Array
        natural_breaks_agg = natural_breaks(agg = terrain_agg, name = 'Elevation')

        # Edit Attributes
        natural_breaks_agg = natural_breaks_agg.assign_attrs({'Description': 'Example Natural Breaks'})

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Natural Breaks
        natural_breaks_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Natural Breaks")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02249454, 1261.94748873],
               [1285.37061171, 1282.48046696],
               [1306.02305679, 1303.40657515]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

        >>> print(natural_breaks_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1., 1.],
               [1., 1.],
               [1., 1.]], dtype=float32)
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Natural Breaks
            units:          km
            Max Elevation:  4000
    """
    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy_natural_break(agg.data, num_sample, k)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy_natural_break(agg.data, num_sample, k)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return DataArray(out,
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)


def _run_numpy_equal_interval(data, k):
    max_data = np.nanmax(data)
    min_data = np.nanmin(data)
    rg = max_data - min_data
    width = rg * 1.0 / k
    cuts = np.arange(min_data + width, max_data + width, width)
    l_cuts = len(cuts)
    if l_cuts > k:
        # handle overshooting
        cuts = cuts[0:k]
    cuts[-1] = max_data
    out = _run_numpy_bin(data, cuts, np.arange(l_cuts))
    return out


def _run_dask_numpy_equal_interval(data, k):
    max_data = da.nanmax(data)
    min_data = da.nanmin(data)
    width = (max_data - min_data) / k
    cuts = da.arange(min_data + width, max_data + width, width)
    l_cuts = cuts.shape[0]
    if l_cuts > k:
        # handle overshooting
        cuts = cuts[0:k]
    # work around to assign cuts[-1] = max_data
    bins = da.concatenate([cuts[:k-1], [max_data]])
    out = _bin(data, bins, np.arange(l_cuts))
    return out


def _run_cupy_equal_interval(data, k):
    max_data = cupy.nanmax(data)
    min_data = cupy.nanmin(data)
    width = (max_data - min_data) / k
    cuts = cupy.arange(min_data.get() +
                       width.get(), max_data.get() +
                       width.get(), width.get())
    l_cuts = cuts.shape[0]
    if l_cuts > k:
        # handle overshooting
        cuts = cuts[0:k]
    cuts[-1] = max_data
    out = _bin(data, cuts, cupy.arange(l_cuts))
    return out


def _run_dask_cupy_equal_interval(data, k):
    msg = 'Not yet supported.'
    raise NotImplementedError(msg)


def equal_interval(agg: xr.DataArray,
                   k: int = 5,
                   name: Optional[str] = 'equal_interval') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on intervals
    of equal width.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    k : int, default=5
        Number of classes to be produced.
    name : str, default='equal_interval'
        Name of output aggregate.

    Returns
    -------
    equal_interval_agg : xarray.DataArray of the same type as `agg`
        2D aggregate array of equal interval allocations.
        All other input attributes are preserved.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#EqualInterval # noqa
        - scikit-learn: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.classify import equal_interval

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))

        # Generate Example Terrain
        terrain_agg = generate_terrain(canvas = cvs)

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            }
        )
        
        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Create Equal Interval Aggregate Array
        equal_interval_agg = equal_interval(agg = terrain_agg, name = 'Elevation')

        # Edit Attributes
        equal_interval_agg = equal_interval_agg.assign_attrs({'Description': 'Example Equal Interval'})

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Equal Interval
        equal_interval_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Equal Interval")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02249454, 1261.94748873],
            [1285.37061171, 1282.48046696],
            [1306.02305679, 1303.40657515]])
        Coordinates:
        * lon      (lon) float64 -3.96e+06 -3.88e+06
        * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

    .. sourcecode:: python

        >>> print(equal_interval_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1., 1.],
            [1., 1.],
            [1., 1.]], dtype=float32)
        Coordinates:
        * lon      (lon) float64 -3.96e+06 -3.88e+06
        * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Equal Interval
            units:          km
            Max Elevation:  4000
    """
    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy_equal_interval(agg.data, k)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy_equal_interval(agg.data, k)

    # dask + cupy case
    elif has_cuda() and \
            isinstance(agg.data, cupy.ndarray) and \
            is_cupy_backed(agg):
        out = _run_dask_cupy_equal_interval(agg.data, k)

    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        out = _run_dask_numpy_equal_interval(agg.data, k)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return DataArray(out,
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)
