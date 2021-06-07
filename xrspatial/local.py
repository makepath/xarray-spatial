from collections import Counter

import numpy as np
import xarray as xr


def combine(raster, dims=None):
    """
    Combine the dimensions of a `xarray.Dataset` so that a unique
    output value is assigned to each unique combination.

    Parameters
    ----------
    raster : xarray.Dataset
        The input raster to combine the dimensions.
    dims : list of string
        The list of dimensions name to be combined.

    Returns
    -------
    combined_arr : xarray.DataArray
        The combined dimensions data.

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/combine.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if dims:
        if (
            not isinstance(dims, list) or
            not all([isinstance(dim, str) for dim in dims])
        ):
            raise TypeError('Expected dims to be a list of string.')

        if not set(dims).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the dimensions of dims. "
                f"The dimensions available are '{list(raster.data_vars)}'."
            )
    else:
        dims = list(raster.data_vars)

    iter_list = []

    for comb in np.nditer([raster[dim].data for dim in dims]):
        iter_list.append(tuple(items.item() for items in comb))

    unique_comb = {}
    unique_values = {}
    all_comb = []
    all_values = []
    value = 1

    for comb in iter_list:
        if np.isnan(comb).any():
            all_values.append(np.nan)
            all_comb.append('NAN')
            continue
        if comb in unique_comb.keys():
            all_comb.append(comb)
            all_values.append(0)
        else:
            unique_comb[comb] = value
            unique_values[value] = comb
            all_comb.append(comb)
            all_values.append(value)
            value += 1

    k = 0
    for value in all_values:
        if value == 0:
            comb = all_comb[k]
            all_values[k] = [unique_comb[comb]][0]
        k += 1

    final_arr = np.array(all_values)
    final_arr = np.reshape(final_arr, (-1, raster[dims[0]].data.shape[1]))

    combined_arr = xr.DataArray(
        data=final_arr,
        attrs=dict(key=unique_values)
    )

    return combined_arr


def equal_frequency(raster, dim_ref, dims=None):
    """
    Evaluates on a cell-by-cell basis the number of times the values
    in a set of rasters are equal to another raster.

    Parameters
    ----------
    raster : xarray.Dataset
        The input raster to be compared.
    dim_ref : string
        The reference dimension name. 
    dims : list of string
        The list of dimensions name to be compared.

    Returns
    -------
    final_arr : xarray.DataArray
        The result.

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/equal-to-frequency.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(dim_ref, str):
        raise TypeError(
            "Expected dim_ref to be a 'str'. "
            f"Received '{type(dim_ref).__name__}' instead."
        )

    if dim_ref not in list(raster.data_vars):
        raise ValueError('raster must contain dim_ref.')

    if dims:
        if (
            not isinstance(dims, list) or
            not all([isinstance(dim, str) for dim in dims])
        ):
            raise TypeError('Expected dims to be a list of string.')

        if not set(dims).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the dimensions of dims. "
                f"The dimensions available are '{list(raster.data_vars)}'."
            )

        if dim_ref in dims:
            raise ValueError('dim_ref must not be an element of dims.')

    else:
        dims = list(raster.data_vars)
        dims.remove(dim_ref)

    iter_list = []

    for comb in np.nditer([raster[dim].data for dim in dims]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[dim_ref].data for item in arr]
    for ref, comb in zip(ref_list, iter_list):
        count = 0
        if np.isnan(comb).any():
            out.append(np.nan)
            continue

        for item in comb:
            if ref == item:
                count += 1

        out.append(count)

    final_arr = np.array(out)
    final_arr = np.reshape(final_arr, (-1, raster[dims[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def greater_frequency(raster, dim_ref, dims=None):
    """
    Evaluates on a cell-by-cell basis the number of times a set of
    rasters is greater than another raster.

    Parameters
    ----------
    raster : xarray.Dataset
        The input raster to be compared.
    dim_ref : string
        The reference dimension name. 
    dims : list of string
        The list of dimensions name to be compared.

    Returns
    -------
    final_arr : xarray.DataArray
        The result.

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/greater-than-frequency.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(dim_ref, str):
        raise TypeError(
            "Expected dim_ref to be a 'str'. "
            f"Received '{type(dim_ref).__name__}' instead."
        )

    if dim_ref not in list(raster.data_vars):
        raise ValueError('raster must contain dim_ref.')

    if dims:
        if (
            not isinstance(dims, list) or
            not all([isinstance(dim, str) for dim in dims])
        ):
            raise TypeError('Expected dims to be a list of string.')

        if not set(dims).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the dimensions of dims. "
                f"The dimensions available are '{list(raster.data_vars)}'."
            )

        if dim_ref in dims:
            raise ValueError('dim_ref must not be an element of dims.')

    else:
        dims = list(raster.data_vars)
        dims.remove(dim_ref)

    iter_list = []

    for comb in np.nditer([raster[dim].data for dim in dims]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[dim_ref].data for item in arr]
    for ref, comb in zip(ref_list, iter_list):
        count = 0
        if np.isnan(comb).any():
            out.append(np.nan)
            continue

        for item in comb:
            if ref < item:
                count += 1

        out.append(count)

    final_arr = np.array(out)
    final_arr = np.reshape(final_arr, (-1, raster[dims[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def highest_position(raster, dims=None):
    """
    Determines on a cell-by-cell basis the position of the raster with
    the maximum value in a `xarray.Dataset` dimensions.

    Parameters
    ----------
    raster : xarray.Dataset
        The input raster.
    dims : list of string
        The list of dimensions name.

    Returns
    -------
    final_arr : xarray.DataArray
        The result.

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/highest-position.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if dims:
        if (
            not isinstance(dims, list) or
            not all([isinstance(dim, str) for dim in dims])
        ):
            raise TypeError('Expected dims to be a list of string.')

        if not set(dims).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the dimensions of dims. "
                f"The dimensions available are '{list(raster.data_vars)}'."
            )
    else:
        dims = list(raster.data_vars)

    iter_list = []

    for comb in np.nditer([raster[dim].data for dim in dims]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []

    for comb in iter_list:
        if np.isnan(comb).any():
            out.append(np.nan)
            continue

        max_value = max(comb)
        max_index = comb.index(max_value) + 1

        out.append(max_index)

    final_arr = np.array(out)
    final_arr = np.reshape(final_arr, (-1, raster[dims[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def lesser_frequency(raster, dim_ref, dims=None):
    """
    Evaluates on a cell-by-cell basis the number of times a set of
    rasters is less than another raster.

    Parameters
    ----------
    raster : xarray.Dataset
        The input raster to be compared.
    dim_ref : string
        The reference dimension name. 
    dims : list of string
        The list of dimensions name to be compared.

    Returns
    -------
    final_arr : xarray.DataArray
        The result.

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/less-than-frequency.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(dim_ref, str):
        raise TypeError(
            "Expected dim_ref to be a 'str'. "
            f"Received '{type(dim_ref).__name__}' instead."
        )

    if dim_ref not in list(raster.data_vars):
        raise ValueError('raster must contain dim_ref.')

    if dims:
        if (
            not isinstance(dims, list) or
            not all([isinstance(dim, str) for dim in dims])
        ):
            raise TypeError('Expected dims to be a list of string.')

        if not set(dims).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the dimensions of dims. "
                f"The dimensions available are '{list(raster.data_vars)}'."
            )

        if dim_ref in dims:
            raise ValueError('dim_ref must not be an element of dims.')

    else:
        dims = list(raster.data_vars)
        dims.remove(dim_ref)

    iter_list = []

    for comb in np.nditer([raster[dim].data for dim in dims]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[dim_ref].data for item in arr]
    for ref, comb in zip(ref_list, iter_list):
        count = 0
        if np.isnan(comb).any():
            out.append(np.nan)
            continue

        for item in comb:
            if ref > item:
                count += 1

        out.append(count)

    final_arr = np.array(out)
    final_arr = np.reshape(final_arr, (-1, raster[dims[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def lowest_position(raster, dims=None):
    """
    Determines on a cell-by-cell basis the position of the raster with
    the minimum value in a `xarray.Dataset` dimensions.

    Parameters
    ----------
    raster : xarray.Dataset
        The input raster.
    dims : list of string
        The list of dimensions name.

    Returns
    -------
    final_arr : xarray.DataArray
        The result.

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/lowest-position.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if dims:
        if (
            not isinstance(dims, list) or
            not all([isinstance(dim, str) for dim in dims])
        ):
            raise TypeError('Expected dims to be a list of string.')

        if not set(dims).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the dimensions of dims. "
                f"The dimensions available are '{list(raster.data_vars)}'."
            )
    else:
        dims = list(raster.data_vars)

    iter_list = []

    for comb in np.nditer([raster[dim].data for dim in dims]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []

    for comb in iter_list:
        if np.isnan(comb).any():
            out.append(np.nan)
            continue

        min_value = min(comb)
        min_index = comb.index(min_value) + 1

        out.append(min_index)

    final_arr = np.array(out)
    final_arr = np.reshape(final_arr, (-1, raster[dims[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def popularity(raster, dim_ref, dims=None):
    """
    Determines the value in an argument list that is at a certain
    level of popularity on a cell-by-cell basis which the number
    of occurrences of each value is specified by the first argument.


    Parameters
    ----------
    raster : xarray.Dataset
        The input raster to be compared.
    dim_ref : string
        The reference dimension name. 
    dims : list of string
        The list of dimensions name to be compared.

    Returns
    -------
    final_arr : xarray.DataArray
        The result.

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/popularity.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(dim_ref, str):
        raise TypeError(
            "Expected dim_ref to be a 'str'. "
            f"Received '{type(dim_ref).__name__}' instead."
        )

    if dim_ref not in list(raster.data_vars):
        raise ValueError('raster must contain dim_ref.')

    if dims:
        if (
            not isinstance(dims, list) or
            not all([isinstance(dim, str) for dim in dims])
        ):
            raise TypeError('Expected dims to be a list of string.')

        if not set(dims).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the dimensions of dims. "
                f"The dimensions available are '{list(raster.data_vars)}'."
            )

        if dim_ref in dims:
            raise ValueError('dim_ref must not be an element of dims.')

    else:
        dims = list(raster.data_vars)
        dims.remove(dim_ref)

    iter_list = []

    for comb in np.nditer([raster[dim].data for dim in dims]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[dim_ref].data for item in arr]

    for ref, comb in zip(ref_list, iter_list):
        comb = np.array(comb)
        comb_ref = ref - 1
        comb_counts = sorted(list(dict(Counter(comb)).keys()))

        if (np.isnan(comb).any() or len(comb_counts) >= len(comb)):
            out.append(np.nan)
            continue
        elif len(comb_counts) == 1:
            out.append(comb_counts[0])
        else:
            if comb_ref >= len(comb_counts):
                out.append(np.nan)
                continue

            out.append(comb_counts[comb_ref])

    final_arr = np.array(out)
    final_arr = np.reshape(final_arr, (-1, raster[dims[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def rank(val_agg, agg_list):
    out = []
    in_aggs = [val_agg]
    for agg in agg_list:
        in_aggs.append(agg)

    # Iterate through each array simultaneously
    for v, a, b, c in np.nditer(in_aggs):
        if np.isnan((a, b, c)).any():   # skip nan
            out.append(np.nan)
            continue

        sort = np.sort((a, b, c))[::-1]
        if v == 1:
            out.append(sort[2])
        elif v == 3:
            out.append(sort[0])
        else:
            out.append(sort[1])

    # create new array
    out = np.array(out)
    out = np.reshape(out, (-1, agg_list[0].shape[1]))
    out = xr.DataArray(out)

    return out
