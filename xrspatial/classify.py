from __future__ import annotations

import warnings
from functools import partial
from typing import List, Optional

import cmath

import xarray as xr

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None
    da = None

import numba as nb
import numpy as np

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    _validate_scalar,
    cuda_args,
    ngjit,
    not_implemented_func,
)
from xrspatial.dataset_support import supports_dataset


@ngjit
def _cpu_binary(data, values):
    out = np.zeros_like(data)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            if np.any(values == data[y, x]):
                out[y, x] = 1
            elif np.isfinite(data[y, x]):
                out[y, x] = 0
    return out


def _run_numpy_binary(data, values):
    values = np.asarray(values)
    out = _cpu_binary(data, values)
    return out


def _run_dask_numpy_binary(data, values):
    _func = partial(_run_numpy_binary, values=values)
    out = data.map_blocks(_func)
    return out


@nb.cuda.jit(device=True)
def _gpu_binary(val, values):
    for v in values:
        if val == v:
            return 1
    return 0


@nb.cuda.jit
def _run_gpu_binary(data, values, out):
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < out.shape[0] and j >= 0 and j < out.shape[1]:
        if cmath.isfinite(data[i, j]):
            out[i, j] = _gpu_binary(data[i, j], values)


def _run_cupy_binary(data, values):
    values_cupy = cupy.asarray(values)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    _run_gpu_binary[griddim, blockdim](data, values_cupy, out)
    return out


def _run_dask_cupy_binary(data, values_cupy):
    out = data.map_blocks(lambda da: _run_cupy_binary(da, values_cupy), meta=cupy.array(()))
    return out


@supports_dataset
def binary(agg, values, name='binary'):
    """
    Binarize a data array based on a set of values. Data that equals to a value in the set will be
    set to 1. In contrast, data that does not equal to any value in the set will be set to 0.
    Note that NaNs and infinite values will be set to NaNs.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    values : array-like object
        Values to keep in the binarized array.
    name : str, default='binary'
        Name of output aggregate array.

    Returns
    -------
    binarized_agg : xr.DataArray or xr.Dataset
        2D aggregate array of binarized data array.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    Examples
    --------
    Binary works with NumPy backed xarray DataArray

    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import binary

        >>> data = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., np.inf],
        ], dtype=np.float32)
        >>> agg = xr.DataArray(data)
        >>> values = [1, 2, 3]
        >>> agg_binary = binary(agg, values)
        >>> print(agg_binary)
        <xarray.DataArray 'binary' (dim_0: 4, dim_1: 5)>
        array([[np.nan,  1.,  1.,  1.,  0.],
               [0.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  0.,  np.nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
    """
    _validate_raster(agg, func_name='binary', name='agg', ndim=None)

    mapper = ArrayTypeFunctionMapping(numpy_func=_run_numpy_binary,
                                      dask_func=_run_dask_numpy_binary,
                                      cupy_func=_run_cupy_binary,
                                      dask_cupy_func=_run_dask_cupy_binary)
    out = mapper(agg)(agg.data, values)
    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


@ngjit
def _cpu_bin(data, bins, new_values):
    out = np.zeros(data.shape, dtype=np.float32)
    out[:] = np.nan
    rows, cols = data.shape
    nbins = len(bins)
    for y in range(0, rows):
        for x in range(0, cols):
            val = data[y, x]
            val_bin = -1

            # find bin
            if np.isfinite(val):
                if val <= bins[0]:
                    val_bin = 0
                elif val <= bins[nbins - 1]:
                    start = 0
                    end = nbins - 1
                    mid = (end + start) // 2
                    while start <= end:
                        if bins[mid] < val:
                            start = mid + 1
                        elif val > bins[mid - 1]:
                            break
                        else:
                            end = mid - 1
                        mid = (end + start) // 2

                    val_bin = mid

            if val_bin > -1:
                out[y, x] = new_values[val_bin]
            else:
                out[y, x] = np.nan

    return out


def _run_numpy_bin(data, bins, new_values):
    bins = np.asarray(bins)
    new_values = np.asarray(new_values)
    out = _cpu_bin(data, bins, new_values)
    return out


def _run_dask_numpy_bin(data, bins, new_values):
    _func = partial(_run_numpy_bin,
                    bins=bins,
                    new_values=new_values)

    out = data.map_blocks(_func)
    return out


@nb.cuda.jit(device=True)
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


@nb.cuda.jit
def _run_gpu_bin(data, bins, new_values, out):
    i, j = nb.cuda.grid(2)
    if (i >= 0 and i < out.shape[0] and j >= 0 and j < out.shape[1]):
        out[i, j] = _gpu_bin(data[i:i+1, j:j+1], bins, new_values)


def _run_cupy_bin(data, bins, new_values):
    # replace ±inf with nan in a single pass to avoid classifying outliers
    data = cupy.where(cupy.isinf(data), cupy.nan, data)

    bins_cupy = cupy.asarray(bins)
    new_values_cupy = cupy.asarray(new_values)
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


def _bin(agg, bins, new_values):
    mapper = ArrayTypeFunctionMapping(numpy_func=_run_numpy_bin,
                                      dask_func=_run_dask_numpy_bin,
                                      cupy_func=_run_cupy_bin,
                                      dask_cupy_func=_run_dask_cupy_bin)

    out = mapper(agg)(agg.data, bins, new_values)
    return out


@supports_dataset
def reclassify(agg: xr.DataArray,
               bins: List[int],
               new_values: List[int],
               name: Optional[str] = 'reclassify') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on user
    defined bins.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
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
    reclass_agg : xr.DataArray or xr.Dataset
        2D aggregate array of reclassified allocations.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html # noqa

    Examples
    --------
    Reclassify works with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import reclassify

        >>> data = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., np.inf],
        ])
        >>> agg = xr.DataArray(data)
        >>> print(agg)
        <xarray.DataArray (dim_0: 4, dim_1: 5)>
        array([[nan,  1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.,  9.],
               [10., 11., 12., 13., 14.],
               [15., 16., 17., 18., inf]])
        Dimensions without coordinates: dim_0, dim_1
        >>> bins=[10, 15, np.inf]
        >>> new_values=[1, 2, 3]
        >>> agg_reclassify = reclassify(agg, bins=bins, new_values=new_values)
        >>> print(agg_reclassify)
        <xarray.DataArray 'reclassify' (dim_0: 4, dim_1: 5)>
        array([[nan,  1.,  1.,  1.,  1.],
               [ 1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 2.,  3.,  3.,  3.,  3.]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1

    Reclassify works with Dask with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import dask.array as da
        >>> data_da = da.from_array(data, chunks=(3, 3))
        >>> agg_da = xr.DataArray(data_da, name='agg_da')
        >>> print(agg_da)
        <xarray.DataArray 'agg_da' (dim_0: 4, dim_1: 5)>
        dask.array<array, shape=(4, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
        >>> agg_reclassify_da = reclassify(agg_da, bins=bins, new_values=new_values)  # noqa
        >>> print(agg_reclassify_da)
        <xarray.DataArray 'reclassify' (dim_0: 4, dim_1: 5)>
        dask.array<_run_numpy_bin, shape=(4, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
        >>> print(agg_reclassify_da.compute())  # print the computed the results
        <xarray.DataArray 'reclassify' (dim_0: 4, dim_1: 5)>
        array([[nan,  1.,  1.,  1.,  1.],
               [ 1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 2.,  3.,  3.,  3.,  3.]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1

    Reclassify works with CuPy backed xarray DataArray.
    Make sure you have a GPU and CuPy installed to run this example.
    .. sourcecode:: python

        >>> import cupy
        >>> data_cupy = cupy.asarray(data)
        >>> agg_cupy = xr.DataArray(data_cupy)
        >>> agg_reclassify_cupy = reclassify(agg_cupy, bins, new_values)
        >>> print(type(agg_reclassify_cupy.data))
        <class 'cupy.core.core.ndarray'>
        >>> print(agg_reclassify_cupy)
        <xarray.DataArray 'reclassify' (dim_0: 4, dim_1: 5)>
        array([[nan,  1.,  1.,  1.,  1.],
               [ 1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 2.,  3.,  3.,  3.,  3.]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1

    Reclassify works with Dask with CuPy backed xarray DataArray.
    """
    _validate_raster(agg, func_name='reclassify', name='agg', ndim=None)

    if len(bins) != len(new_values):
        raise ValueError(
            'bins and new_values mismatch. Should have same length.'
        )
    out = _bin(agg, bins, new_values)
    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


def _run_quantile(data, num_sample, k, module):
    # num_sample ignored for in-memory backends
    w = 100.0 / k
    p = module.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    q = module.percentile(data[module.isfinite(data)], p)
    q = module.unique(q)
    return q


def _run_dask_quantile(data, num_sample, k):
    # Avoid boolean fancy indexing (data[da.isfinite(data)]) which creates
    # unknown dask chunk sizes.  Use sampling via indexed access to avoid
    # materialising the full array (#884).
    w = 100.0 / k
    p = np.arange(w, 100 + w, w)
    if p[-1] > 100.0:
        p[-1] = 100.0
    clean = da.where(da.isinf(data), np.nan, data)
    num_data = data.size
    if num_sample is None or num_sample >= num_data:
        num_sample = num_data
    sample_idx = _generate_sample_indices(num_data, num_sample)
    values = np.asarray(clean.ravel()[sample_idx].compute())
    values = values[np.isfinite(values)]
    q = np.percentile(values, p)
    q = np.unique(q)
    return q


def _run_dask_cupy_quantile(data, num_sample, k):
    # Convert dask+cupy chunks to numpy, then same safe path as dask (#884).
    data_cpu = data.map_blocks(cupy.asnumpy, dtype=data.dtype, meta=np.array(()))
    return _run_dask_quantile(data_cpu, num_sample, k)


def _quantile(agg, num_sample, k):
    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_quantile(*args, module=np),
        dask_func=_run_dask_quantile,
        cupy_func=lambda *args: _run_quantile(*args, module=cupy),
        dask_cupy_func=_run_dask_cupy_quantile
    )
    out = mapper(agg)(agg.data, num_sample, k)
    return out


@supports_dataset
def quantile(agg: xr.DataArray,
             k: int = 4,
             num_sample: Optional[int] = 20_000,
             name: Optional[str] = 'quantile') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on quantile
    groups of equal size.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    k : int, default=4
        Number of quantiles to be produced.
    num_sample : int or None, default=20000
        Number of sample data points used to compute percentile
        breakpoints.  For dask-backed arrays the sample is drawn
        lazily to avoid materialising the entire array into RAM.
        ``None`` means use all data (safe for numpy/cupy,
        automatically capped for dask).
    name : str, default='quantile'
        Name of the output aggregate array.

    Returns
    -------
    quantile_agg : xr.DataArray or xr.Dataset
        2D aggregate array, of quantile allocations.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    Notes
    -----
        - Dask's percentile algorithm is approximate, while numpy's is exact.
        - This may cause some differences between results of vanilla numpy
    and dask version of the input agg. (https://github.com/dask/dask/issues/3099) # noqa

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#Quantiles # noqa

    Examples
    --------
    Quantile work with numpy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import quantile

        >>> elevation = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., np.inf]
        ])
        >>> agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
        >>> numpy_quantile = quantile(agg_numpy, k=5)
        >>> print(numpy_quantile)
        <xarray.DataArray 'quantile' (dim_0: 5, dim_1: 5)>
        array([[nan,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  1.,  1.,  1.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 3.,  3.,  3.,  3.,  4.],
               [ 4.,  4.,  4.,  4., nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10.0, 10.0)
    """
    _validate_raster(agg, func_name='quantile', name='agg', ndim=None)
    _validate_scalar(k, func_name='quantile', name='k', dtype=int, min_val=2)

    q = _quantile(agg, num_sample, k)
    k_q = q.shape[0]
    if k_q < k:
        print("Quantile Warning: Not enough unique values"
              "for k classes (using {} bins)".format(k_q))
        k = k_q

    out = _bin(agg, bins=q, new_values=np.arange(k))

    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


@nb.jit(nopython=True)
def _run_numpy_jenks_matrices(data, n_classes):
    n_data = data.shape[0]
    lower_class_limits = np.zeros(
        (n_data + 1, n_classes + 1), dtype=np.float32
    )
    lower_class_limits[1, 1:n_classes + 1] = 1.0

    var_combinations = np.zeros(
        (n_data + 1, n_classes + 1), dtype=np.float32
    )
    var_combinations[2:n_data + 1, 1:n_classes + 1] = np.inf

    variance = 0.0

    for l in range(2, n_data + 1): # noqa
        sum = 0.0
        sum_squares = 0.0
        w = 0.0

        for m in range(l):
            # `III` originally
            lower_class_limit = l - m
            i4 = lower_class_limit - 1

            val = np.float32(data[i4])

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

            if i4 == 0:
                continue
            for j in range(2, n_classes + 1):
                # if adding this element to an existing class
                # will increase its variance beyond the limit, break
                # the class at this point, setting the lower_class_limit
                # at this point.
                new_variance = variance + var_combinations[i4, j-1]
                if var_combinations[l, j] >= new_variance:
                    lower_class_limits[l, j] = lower_class_limit
                    var_combinations[l, j] = new_variance

        lower_class_limits[l, 1] = 1.
        var_combinations[l, 1] = variance

    return lower_class_limits, var_combinations


def _run_jenks(data, n_classes):
    # ported from existing cython implementation:
    # https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    data.sort()
    lower_class_limits, _ = _run_numpy_jenks_matrices(data, n_classes)

    k = data.shape[0]
    kclass = np.zeros(n_classes + 1, dtype=np.float32)
    kclass[0] = data[0]
    kclass[-1] = data[-1]
    count_num = n_classes

    while count_num > 1:
        elt = int(lower_class_limits[k][count_num] - 2)
        kclass[count_num - 1] = data[elt]
        k = int(lower_class_limits[k][count_num] - 1)
        count_num -= 1

    return kclass


def _compute_natural_break_bins(data_flat_np, num_sample, k, max_data):
    """Shared helper: compute natural break bins from a flat numpy array.

    Returns (bins, uvk) where bins is a numpy array of bin edges
    and uvk is the number of unique values.
    """
    num_data = data_flat_np.size

    if num_sample is not None and num_sample < num_data:
        # randomly select sample from the whole dataset
        # create a pseudo random number generator
        # Note: cupy and numpy generate different random numbers
        # use numpy.random to ensure the same result
        generator = np.random.RandomState(1234567890)
        idx = np.linspace(
            0, num_data, num_data, endpoint=False, dtype=np.uint32
        )
        generator.shuffle(idx)
        sample_idx = idx[:num_sample]
        sample_data = data_flat_np[sample_idx]
    else:
        sample_data = data_flat_np

    # warning if number of total data points to fit the model bigger than 40k
    if sample_data.size >= 40000:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.warn('natural_breaks Warning: Natural break '
                          'classification (Jenks) has a complexity of O(n^2), '
                          'your classification with {} data points may take '
                          'a long time.'.format(sample_data.size),
                          Warning)

    sample_data = np.asarray(sample_data)

    # only include finite values
    sample_data = sample_data[np.isfinite(sample_data)]
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
        centroids = _run_jenks(sample_data, k)
        bins = np.array(centroids[1:])
        bins[-1] = max_data

    return bins, uvk


def _run_natural_break(agg, num_sample, k):
    data = agg.data
    max_data = float(np.max(data[np.isfinite(data)]))
    data_flat_np = data.flatten()

    bins, uvk = _compute_natural_break_bins(data_flat_np, num_sample, k, max_data)
    out = _bin(agg, bins, np.arange(uvk))
    return out


def _run_cupy_natural_break(agg, num_sample, k):
    data = agg.data
    max_data = float(cupy.max(data[cupy.isfinite(data)]).get())
    data_flat_np = cupy.asnumpy(data.ravel())

    bins, uvk = _compute_natural_break_bins(data_flat_np, num_sample, k, max_data)
    out = _bin(agg, bins, np.arange(uvk))
    return out


def _generate_sample_indices(num_data, num_sample, seed=1234567890):
    """Generate sorted sample indices for natural breaks sampling.

    For small datasets (<=10M elements), uses the same linspace+shuffle
    approach as the numpy backend for exact reproducibility.
    For large datasets, uses memory-efficient RandomState.choice()
    which is O(num_sample) rather than O(num_data).
    """
    generator = np.random.RandomState(seed)
    if num_data <= 10_000_000:
        idx = np.linspace(
            0, num_data, num_data, endpoint=False, dtype=np.uint32
        )
        generator.shuffle(idx)
        sample_idx = idx[:num_sample]
    else:
        sample_idx = generator.choice(
            num_data, size=num_sample, replace=False
        )
    # sort for efficient dask chunk access
    sample_idx.sort()
    return sample_idx


def _run_dask_natural_break(agg, num_sample, k):
    data = agg.data
    # Avoid boolean fancy indexing which flattens dask arrays and
    # produces chunks of unknown size; use element-wise where instead
    max_data = float(da.nanmax(da.where(da.isinf(data), np.nan, data)).compute())

    num_data = data.size
    if num_sample is None or num_sample >= num_data:
        num_sample = num_data  # cap: still uses indexed access, never .compute() all
    sample_idx = _generate_sample_indices(num_data, num_sample)
    sample_data_np = np.asarray(data.ravel()[sample_idx].compute())
    bins, uvk = _compute_natural_break_bins(sample_data_np, None, k, max_data)

    out = _bin(agg, bins, np.arange(uvk))
    return out


def _run_dask_cupy_natural_break(agg, num_sample, k):
    data = agg.data
    # Avoid boolean fancy indexing which flattens dask arrays and
    # produces chunks of unknown size; use element-wise where instead
    max_data = float(da.nanmax(da.where(da.isinf(data), np.nan, data)).compute().item())

    num_data = data.size
    if num_sample is None or num_sample >= num_data:
        num_sample = num_data  # cap: still uses indexed access, never .compute() all
    sample_idx = _generate_sample_indices(num_data, num_sample)
    sample_data = data.ravel()[sample_idx].compute()
    sample_data_np = cupy.asnumpy(sample_data)
    bins, uvk = _compute_natural_break_bins(sample_data_np, None, k, max_data)

    out = _bin(agg, bins, np.arange(uvk))
    return out


@supports_dataset
def natural_breaks(agg: xr.DataArray,
                   num_sample: Optional[int] = 20000,
                   name: Optional[str] = 'natural_breaks',
                   k: int = 5) -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on Natural
    Breaks or K-Means clustering method. Values are grouped so that
    similar values are placed in the same group and space between
    groups is maximized.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask array
        of values to be reclassified.
    num_sample : int, default=20000
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
    natural_breaks_agg : xr.DataArray or xr.Dataset
        2D aggregate array of natural break allocations.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#NaturalBreaks # noqa
        - jenks: https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    Examples
    -------
    natural_breaks() works with numpy backed xarray DataArray.
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import natural_breaks

        >>> elevation = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., np.inf]
        ])
        >>> agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
        >>> numpy_natural_breaks = natural_breaks(agg_numpy, k=5)
        >>> print(numpy_natural_breaks)
        <xarray.DataArray 'natural_breaks' (dim_0: 5, dim_1: 5)>
        array([[nan,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  1.,  2.],
               [ 2.,  2.,  2.,  2.,  3.],
               [ 3.,  3.,  3.,  3.,  4.],
               [ 4.,  4.,  4.,  4., nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10.0, 10.0)

    natural_breaks() works with cupy backed xarray DataArray.
    .. sourcecode:: python

        >>> import cupy
        >>> agg_cupy = xr.DataArray(cupy.asarray(elevation))
        >>> cupy_natural_breaks = natural_breaks(agg_cupy)
        >>> print(type(cupy_natural_breaks))
        <class 'xarray.core.dataarray.DataArray'>
        >>> print(cupy_natural_breaks)
        <xarray.DataArray 'natural_breaks' (dim_0: 5, dim_1: 5)>
        array([[nan,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  1.,  2.],
               [ 2.,  2.,  2.,  2.,  3.],
               [ 3.,  3.,  3.,  3.,  4.],
               [ 4.,  4.,  4.,  4., nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
    """
    _validate_raster(agg, func_name='natural_breaks', name='agg', ndim=None)
    _validate_scalar(k, func_name='natural_breaks', name='k', dtype=int, min_val=2)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_natural_break,
        dask_func=_run_dask_natural_break,
        cupy_func=_run_cupy_natural_break,
        dask_cupy_func=_run_dask_cupy_natural_break,
    )
    out = mapper(agg)(agg, num_sample, k)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)


def _run_equal_interval(agg, k, module):
    data = agg.data

    # Replace ±inf with nan in a single pass (no ravel needed)
    data_clean = module.where(module.isinf(data), np.nan, data)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice', RuntimeWarning)
        min_lazy = module.nanmin(data_clean)
        max_lazy = module.nanmax(data_clean)

    if module == cupy:
        min_data = float(min_lazy.get())
        max_data = float(max_lazy.get())
    elif module == da:
        # Compute both in a single pass over the data
        min_data, max_data = dask.compute(min_lazy, max_lazy)
        min_data = float(min_data)
        max_data = float(max_data)
    else:
        min_data = float(min_lazy)
        max_data = float(max_lazy)

    width = (max_data - min_data) / k
    # Build cuts as numpy — only k elements, no need for dask/cupy overhead
    cuts = np.arange(min_data + width, max_data + width, width)
    l_cuts = cuts.shape[0]
    if l_cuts > k:
        cuts = cuts[0:k]

    cuts[-1] = max_data
    out = _bin(agg, cuts, np.arange(l_cuts))
    return out


@supports_dataset
def equal_interval(agg: xr.DataArray,
                   k: int = 5,
                   name: Optional[str] = 'equal_interval') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on intervals
    of equal width.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    k : int, default=5
        Number of classes to be produced.
    name : str, default='equal_interval'
        Name of output aggregate.

    Returns
    -------
    equal_interval_agg : xr.DataArray or xr.Dataset
        2D aggregate array of equal interval allocations.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#EqualInterval # noqa
        - scikit-learn: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import equal_interval
        >>> elevation = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., np.inf]
        ])
        >>> agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
        >>> numpy_equal_interval = equal_interval(agg_numpy, k=5)
        >>> print(numpy_equal_interval)
        <xarray.DataArray 'equal_interval' (dim_0: 5, dim_1: 5)>
        array([[nan,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  1.],
               [ 1.,  1.,  1.,  1.,  1.],
               [ 1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2., nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10.0, 10.0)
    """
    _validate_raster(agg, func_name='equal_interval', name='agg', ndim=None)
    _validate_scalar(k, func_name='equal_interval', name='k', dtype=int, min_val=1)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_equal_interval(*args, module=np),
        dask_func=lambda *args: _run_equal_interval(*args, module=da),
        cupy_func=lambda *args: _run_equal_interval(*args, module=cupy),
        dask_cupy_func=lambda *args: _run_equal_interval(*args, module=da)
    )
    out = mapper(agg)(agg, k)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)


def _run_std_mean(agg, module):
    data = agg.data
    data_clean = module.where(module.isinf(data), np.nan, data)
    mean_lazy = module.nanmean(data_clean)
    std_lazy = module.nanstd(data_clean)
    max_lazy = module.nanmax(data_clean)

    if module == cupy:
        mean_v = float(mean_lazy.get())
        std_v = float(std_lazy.get())
        max_v = float(max_lazy.get())
    elif module == da:
        mean_v, std_v, max_v = dask.compute(mean_lazy, std_lazy, max_lazy)
        mean_v, std_v, max_v = float(mean_v), float(std_v), float(max_v)
    else:
        mean_v, std_v, max_v = float(mean_lazy), float(std_lazy), float(max_lazy)

    bins = np.sort(np.unique([
        mean_v - 2 * std_v,
        mean_v - std_v,
        mean_v + std_v,
        mean_v + 2 * std_v,
        max_v,
    ]))
    out = _bin(agg, bins, np.arange(len(bins)))
    return out


@supports_dataset
def std_mean(agg: xr.DataArray,
             name: Optional[str] = 'std_mean') -> xr.DataArray:
    """
    Classify data based on standard deviations from the mean.

    Creates bins at mean +/- 1 and 2 standard deviations.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask array
        of values to be classified.
    name : str, default='std_mean'
        Name of output aggregate array.

    Returns
    -------
    std_mean_agg : xr.DataArray or xr.Dataset
        2D aggregate array of standard deviation classifications.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#StdMean
    """
    _validate_raster(agg, func_name='std_mean', name='agg', ndim=None)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_std_mean(*args, module=np),
        dask_func=lambda *args: _run_std_mean(*args, module=da),
        cupy_func=lambda *args: _run_std_mean(*args, module=cupy),
        dask_cupy_func=lambda *args: _run_std_mean(*args, module=da),
    )
    out = mapper(agg)(agg)
    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


def _compute_head_tail_bins(values_np):
    """Compute head/tail break bins from flat finite numpy values."""
    bins = []
    data = values_np.copy()
    while len(data) > 1:
        mean_v = float(np.nanmean(data))
        bins.append(mean_v)
        head = data[data > mean_v]
        if len(head) == 0 or len(head) / len(data) > 0.40:
            break
        data = head
    if not bins:
        bins = [float(np.nanmean(values_np))]
    bins.append(float(np.nanmax(values_np)))
    return np.array(bins)


def _run_head_tail_breaks(agg, module):
    data = agg.data
    if module == cupy:
        finite = data[cupy.isfinite(data)]
        values_np = cupy.asnumpy(finite)
    else:
        finite = data[np.isfinite(data)]
        values_np = np.asarray(finite)

    bins = _compute_head_tail_bins(values_np)
    out = _bin(agg, bins, np.arange(len(bins)))
    return out


def _run_dask_head_tail_breaks(agg):
    data = agg.data
    data_clean = da.where(da.isinf(data), np.nan, data)
    bins = []
    mask = da.isfinite(data_clean)
    while True:
        current = da.where(mask, data_clean, np.nan)
        mean_v = float(da.nanmean(current).compute())
        bins.append(mean_v)
        new_mask = mask & (data_clean > mean_v)
        head_count = int(new_mask.sum().compute())
        total_count = int(mask.sum().compute())
        if head_count == 0 or head_count / total_count > 0.40:
            break
        mask = new_mask
    max_v = float(da.nanmax(data_clean).compute())
    bins.append(max_v)
    bins = np.array(bins)
    out = _bin(agg, bins, np.arange(len(bins)))
    return out


@supports_dataset
def head_tail_breaks(agg: xr.DataArray,
                     name: Optional[str] = 'head_tail_breaks') -> xr.DataArray:
    """
    Classify data using the Head/Tail Breaks algorithm.

    Iteratively partitions data around the mean. Values below the mean
    form a class, and values above continue to be split until the head
    proportion exceeds 40%.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask array
        of values to be classified.
    name : str, default='head_tail_breaks'
        Name of output aggregate array.

    Returns
    -------
    head_tail_agg : xr.DataArray or xr.Dataset
        2D aggregate array of head/tail break classifications.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#HeadTailBreaks
    """
    _validate_raster(agg, func_name='head_tail_breaks', name='agg', ndim=None)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_head_tail_breaks(*args, module=np),
        dask_func=_run_dask_head_tail_breaks,
        cupy_func=lambda *args: _run_head_tail_breaks(*args, module=cupy),
        dask_cupy_func=_run_dask_head_tail_breaks,
    )
    out = mapper(agg)(agg)
    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


def _run_percentiles(data, num_sample, pct, module):
    # num_sample ignored for in-memory backends
    q = module.percentile(data[module.isfinite(data)], pct)
    q = module.unique(q)
    return q


def _run_dask_percentiles(data, num_sample, pct):
    # Avoid boolean fancy indexing (data[da.isfinite(data)]) which creates
    # unknown dask chunk sizes.  Use sampling via indexed access to avoid
    # materialising the full array (#884).
    clean = da.where(da.isinf(data), np.nan, data)
    num_data = data.size
    if num_sample is None or num_sample >= num_data:
        num_sample = num_data
    sample_idx = _generate_sample_indices(num_data, num_sample)
    values = np.asarray(clean.ravel()[sample_idx].compute())
    values = values[np.isfinite(values)]
    q = np.percentile(values, pct)
    q = np.unique(q)
    return q


def _run_dask_cupy_percentiles(data, num_sample, pct):
    # Convert dask+cupy chunks to numpy, then same safe path as dask (#884).
    data_cpu = data.map_blocks(cupy.asnumpy, dtype=data.dtype, meta=np.array(()))
    return _run_dask_percentiles(data_cpu, num_sample, pct)


@supports_dataset
def percentiles(agg: xr.DataArray,
                pct: Optional[List] = None,
                num_sample: Optional[int] = 20_000,
                name: Optional[str] = 'percentiles') -> xr.DataArray:
    """
    Classify data based on percentile breakpoints.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask array
        of values to be classified.
    pct : list of float, default=[1, 10, 50, 90, 99]
        Percentile values to use as breakpoints.
    num_sample : int or None, default=20000
        Number of sample data points used to compute percentile
        breakpoints.  For dask-backed arrays the sample is drawn
        lazily to avoid materialising the entire array into RAM.
        ``None`` means use all data (safe for numpy/cupy,
        automatically capped for dask).
    name : str, default='percentiles'
        Name of output aggregate array.

    Returns
    -------
    percentiles_agg : xr.DataArray or xr.Dataset
        2D aggregate array of percentile classifications.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#Percentiles
    """
    _validate_raster(agg, func_name='percentiles', name='agg', ndim=None)

    if pct is None:
        pct = [1, 10, 50, 90, 99]

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_percentiles(*args, module=np),
        dask_func=_run_dask_percentiles,
        cupy_func=lambda *args: _run_percentiles(*args, module=cupy),
        dask_cupy_func=_run_dask_cupy_percentiles,
    )
    q = mapper(agg)(agg.data, num_sample, pct)

    # Materialize bin edges to numpy
    if hasattr(q, 'compute'):
        q_np = np.asarray(q.compute())
    elif hasattr(q, 'get'):
        q_np = np.asarray(q.get())
    else:
        q_np = np.asarray(q)

    # Append max of finite data so all values get classified
    data = agg.data
    if hasattr(data, 'compute'):
        max_v = float(da.nanmax(da.where(da.isinf(data), np.nan, data)).compute())
    elif hasattr(data, 'get'):
        clean = cupy.where(cupy.isinf(data), cupy.nan, data)
        max_v = float(cupy.nanmax(clean).get())
    else:
        clean = np.where(np.isinf(data), np.nan, data)
        max_v = float(np.nanmax(clean))

    bins = np.sort(np.unique(np.append(q_np, max_v)))
    k = len(bins)
    out = _bin(agg, bins, np.arange(k))

    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


def _compute_maximum_break_bins(values_np, k):
    """Compute maximum breaks bins from sorted finite numpy values."""
    uv = np.unique(values_np)
    if len(uv) < k:
        return uv
    diffs = np.diff(uv)
    n_gaps = min(k - 1, len(diffs))
    top_indices = np.argsort(diffs, kind='stable')[-n_gaps:]
    top_indices.sort()
    bins = np.array([(uv[i] + uv[i + 1]) / 2.0 for i in top_indices])
    bins = np.append(bins, float(uv[-1]))
    return bins


def _run_maximum_breaks(agg, num_sample, k, module):
    # num_sample ignored for in-memory backends
    data = agg.data
    if module == cupy:
        finite = data[cupy.isfinite(data)]
        values_np = cupy.asnumpy(finite)
    else:
        finite = data[np.isfinite(data)]
        values_np = np.asarray(finite)

    bins = _compute_maximum_break_bins(values_np, k)
    out = _bin(agg, bins, np.arange(len(bins)))
    return out


def _run_dask_maximum_breaks(agg, num_sample, k):
    data = agg.data
    data_clean = da.where(da.isinf(data), np.nan, data)
    num_data = data.size
    if num_sample is None or num_sample >= num_data:
        num_sample = num_data
    sample_idx = _generate_sample_indices(num_data, num_sample)
    values_np = np.asarray(data_clean.ravel()[sample_idx].compute())
    values_np = values_np[np.isfinite(values_np)]
    bins = _compute_maximum_break_bins(values_np, k)
    out = _bin(agg, bins, np.arange(len(bins)))
    return out


def _run_dask_cupy_maximum_breaks(agg, num_sample, k):
    data = agg.data
    data_clean = da.where(da.isinf(data), np.nan, data)
    data_cpu = data_clean.map_blocks(cupy.asnumpy, dtype=data.dtype, meta=np.array(()))
    num_data = data.size
    if num_sample is None or num_sample >= num_data:
        num_sample = num_data
    sample_idx = _generate_sample_indices(num_data, num_sample)
    values_np = np.asarray(data_cpu.ravel()[sample_idx].compute())
    values_np = values_np[np.isfinite(values_np)]
    bins = _compute_maximum_break_bins(values_np, k)
    out = _bin(agg, bins, np.arange(len(bins)))
    return out


@supports_dataset
def maximum_breaks(agg: xr.DataArray,
                   k: int = 5,
                   num_sample: Optional[int] = 20_000,
                   name: Optional[str] = 'maximum_breaks') -> xr.DataArray:
    """
    Classify data using the Maximum Breaks algorithm.

    Finds the k-1 largest gaps between sorted unique values and uses
    midpoints of those gaps as bin edges.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask array
        of values to be classified.
    k : int, default=5
        Number of classes to be produced.
    num_sample : int or None, default=20000
        Number of sample data points used to fit the model.
        For dask-backed arrays the sample is drawn lazily to avoid
        materialising the entire array into RAM.  ``None`` means use
        all data (safe for numpy/cupy, automatically capped for dask).
    name : str, default='maximum_breaks'
        Name of output aggregate array.

    Returns
    -------
    max_breaks_agg : xr.DataArray or xr.Dataset
        2D aggregate array of maximum break classifications.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#MaximumBreaks
    """
    _validate_raster(agg, func_name='maximum_breaks', name='agg', ndim=None)
    _validate_scalar(k, func_name='maximum_breaks', name='k', dtype=int, min_val=2)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_maximum_breaks(*args, module=np),
        dask_func=_run_dask_maximum_breaks,
        cupy_func=lambda *args: _run_maximum_breaks(*args, module=cupy),
        dask_cupy_func=_run_dask_cupy_maximum_breaks,
    )
    out = mapper(agg)(agg, num_sample, k)
    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


def _run_box_plot(agg, hinge, module):
    data = agg.data
    data_clean = module.where(module.isinf(data), np.nan, data)
    finite_data = data_clean[module.isfinite(data_clean)]

    if module == cupy:
        q1 = float(cupy.percentile(finite_data, 25).get())
        q2 = float(cupy.percentile(finite_data, 50).get())
        q3 = float(cupy.percentile(finite_data, 75).get())
        max_v = float(cupy.nanmax(finite_data).get())
    elif module == da:
        q1_l = da.percentile(finite_data, 25)
        q2_l = da.percentile(finite_data, 50)
        q3_l = da.percentile(finite_data, 75)
        max_l = da.nanmax(data_clean)
        q1, q2, q3, max_v = dask.compute(q1_l, q2_l, q3_l, max_l)
        q1, q2, q3, max_v = float(q1), float(q2), float(q3), float(max_v)
    else:
        q1 = float(np.percentile(finite_data, 25))
        q2 = float(np.percentile(finite_data, 50))
        q3 = float(np.percentile(finite_data, 75))
        max_v = float(np.nanmax(finite_data))

    iqr = q3 - q1
    raw_bins = [q1 - hinge * iqr, q1, q2, q3, q3 + hinge * iqr, max_v]
    bins = np.sort(np.unique(raw_bins))
    # Remove bins above max (they'd create empty classes)
    bins = bins[bins <= max_v]
    if bins[-1] < max_v:
        bins = np.append(bins, max_v)
    out = _bin(agg, bins, np.arange(len(bins)))
    return out


def _run_dask_cupy_box_plot(agg, hinge):
    data = agg.data
    data_cpu = data.map_blocks(cupy.asnumpy, dtype=data.dtype, meta=np.array(()))
    data_clean = da.where(da.isinf(data_cpu), np.nan, data_cpu)
    finite_data = data_clean[da.isfinite(data_clean)]

    q1_l = da.percentile(finite_data, 25)
    q2_l = da.percentile(finite_data, 50)
    q3_l = da.percentile(finite_data, 75)
    max_l = da.nanmax(data_clean)
    q1, q2, q3, max_v = dask.compute(q1_l, q2_l, q3_l, max_l)
    q1, q2, q3, max_v = float(q1), float(q2), float(q3), float(max_v)

    iqr = q3 - q1
    raw_bins = [q1 - hinge * iqr, q1, q2, q3, q3 + hinge * iqr, max_v]
    bins = np.sort(np.unique(raw_bins))
    bins = bins[bins <= max_v]
    if bins[-1] < max_v:
        bins = np.append(bins, max_v)
    out = _bin(agg, bins, np.arange(len(bins)))
    return out


@supports_dataset
def box_plot(agg: xr.DataArray,
             hinge: float = 1.5,
             name: Optional[str] = 'box_plot') -> xr.DataArray:
    """
    Classify data using box plot breakpoints.

    Uses Q1, median (Q2), Q3, and the whiskers (Q1 - hinge*IQR,
    Q3 + hinge*IQR) as class boundaries.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask array
        of values to be classified.
    hinge : float, default=1.5
        Multiplier for the IQR to determine whisker extent.
    name : str, default='box_plot'
        Name of output aggregate array.

    Returns
    -------
    box_plot_agg : xr.DataArray or xr.Dataset
        2D aggregate array of box plot classifications.
        All other input attributes are preserved.
        If `agg` is a Dataset, returns a Dataset with each variable
        classified independently.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#BoxPlot
    """
    _validate_raster(agg, func_name='box_plot', name='agg', ndim=None)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_box_plot(*args, module=np),
        dask_func=lambda *args: _run_box_plot(*args, module=da),
        cupy_func=lambda *args: _run_box_plot(*args, module=cupy),
        dask_cupy_func=_run_dask_cupy_box_plot,
    )
    out = mapper(agg)(agg, hinge)
    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)
