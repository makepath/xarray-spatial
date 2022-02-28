from collections import Counter

import numpy as np
import xarray as xr

funcs = {
    'max': np.max,
    'mean': np.mean,
    'median': np.median,
    'min': np.min,
    'std': np.std,
    'sum': np.sum,
}


def cell_stats(raster, data_vars=None, func='sum'):
    """
    Calculates statistics of raster dataset on a cell-by-cell basis.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    data_vars : list of string
        Variable name list.
    func : string, default=sum
        Statistic type. The supported types are max, mean, median,
        min, std, and sum.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/cell-statistics.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if func not in funcs:
        raise ValueError(
            f'{func} is not supported. '
            f"The supported types are '{list(funcs.keys())}'."
        )

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )
    else:
        data_vars = list(raster.data_vars)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []

    for comb in iter_list:
        out.append(funcs[func](comb))

    final_arr = np.array(out)
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def combine(raster, data_vars=None):
    """
    Combines raster dataset, a unique output value is assigned to each
    unique combination of raster values.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    data_vars : list of string
        Variable name list.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/combine.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )
    else:
        data_vars = list(raster.data_vars)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
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
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))

    final_arr = xr.DataArray(
        data=final_arr,
        attrs=dict(key=unique_values)
    )

    return final_arr


def lesser_frequency(raster, ref_var, data_vars=None):
    """
    Calculates the number of times the raster dataset has a lesser
    frequency on a cell-by-cell basis.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    ref_var : string
        The reference variable name.
    data_vars : list of string
        Variable name list.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/less-than-frequency.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(ref_var, str):
        raise TypeError(
            "Expected ref_var to be a 'str'. "
            f"Received '{type(ref_var).__name__}' instead."
        )

    if ref_var not in list(raster.data_vars):
        raise ValueError('raster must contain ref_var.')

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )

        if ref_var in data_vars:
            raise ValueError('ref_var must not be an element of data_vars.')

    else:
        data_vars = list(raster.data_vars)
        data_vars.remove(ref_var)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[ref_var].data for item in arr]
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
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def equal_frequency(raster, ref_var, data_vars=None):
    """
    Calculates the number of times the raster dataset has a equal
    frequency on a cell-by-cell basis.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    ref_var : string
        The reference variable name. 
    data_vars : list of string
        Variable name list.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/equal-to-frequency.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(ref_var, str):
        raise TypeError(
            "Expected ref_var to be a 'str'. "
            f"Received '{type(ref_var).__name__}' instead."
        )

    if ref_var not in list(raster.data_vars):
        raise ValueError('raster must contain ref_var.')

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )

        if ref_var in data_vars:
            raise ValueError('ref_var must not be an element of data_vars.')

    else:
        data_vars = list(raster.data_vars)
        data_vars.remove(ref_var)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[ref_var].data for item in arr]
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
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def greater_frequency(raster, ref_var, data_vars=None):
    """
    Calculates the number of times the raster dataset has a greater
    frequency on a cell-by-cell basis.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    ref_var : string
        The reference variable name. 
    data_vars : list of string
        Variable name list.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/greater-than-frequency.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(ref_var, str):
        raise TypeError(
            "Expected ref_var to be a 'str'. "
            f"Received '{type(ref_var).__name__}' instead."
        )

    if ref_var not in list(raster.data_vars):
        raise ValueError('raster must contain ref_var.')

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )

        if ref_var in data_vars:
            raise ValueError('ref_var must not be an element of data_vars.')

    else:
        data_vars = list(raster.data_vars)
        data_vars.remove(ref_var)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[ref_var].data for item in arr]
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
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def lowest_position(raster, data_vars=None):
    """
    Calculates the data variable index of the lowest value on a
    cell-by-cell basis.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    data_vars : list of string
        Variable name list.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/lowest-position.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )
    else:
        data_vars = list(raster.data_vars)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
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
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def highest_position(raster, data_vars=None):
    """
    Calculates the data variable index of the highest value on a
    cell-by-cell basis.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    data_vars : list of string
        Variable name list.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/highest-position.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )
    else:
        data_vars = list(raster.data_vars)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
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
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def popularity(raster, ref_var, data_vars=None):
    """
    Calculates the popularity, the number of occurrences of each value,
    of raster dataset on a cell-by-cell basis. The output value is
    assigned based on the reference data variable nth most popular.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    ref_var : string
        The reference variable name. 
    data_vars : list of string
        Variable name list.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/popularity.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(ref_var, str):
        raise TypeError(
            "Expected ref_var to be a 'str'. "
            f"Received '{type(ref_var).__name__}' instead."
        )

    if ref_var not in list(raster.data_vars):
        raise ValueError('raster must contain ref_var.')

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )

        if ref_var in data_vars:
            raise ValueError('ref_var must not be an element of data_vars.')

    else:
        data_vars = list(raster.data_vars)
        data_vars.remove(ref_var)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
        iter_list.append(tuple(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[ref_var].data for item in arr]

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
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr


def rank(raster, ref_var, data_vars=None):
    """
    Calculates the rank of raster dataset on a cell-by-cell basis.
    The output value is assigned based on the reference data variable
    rank.

    Parameters
    ----------
    raster : xarray.Dataset
        2D or 3D labelled array.
    ref_var : string
        The reference variable name. 
    data_vars : list of string
        Variable name list.

    Returns
    -------
    final_arr : xarray.DataArray

    References
    ----------
        - https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/rank.htm # noqa
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            "Expected raster to be a 'xarray.Dataset'. "
            f"Received '{type(raster).__name__}' instead."
        )

    if not isinstance(ref_var, str):
        raise TypeError(
            "Expected ref_var to be a 'str'. "
            f"Received '{type(ref_var).__name__}' instead."
        )

    if ref_var not in list(raster.data_vars):
        raise ValueError('raster must contain ref_var.')

    if data_vars:
        if (
            not isinstance(data_vars, list) or
            not all([isinstance(var, str) for var in data_vars])
        ):
            raise TypeError('Expected data_vars to be a list of string.')

        if not set(data_vars).issubset(raster.data_vars):
            raise ValueError(
                "raster must contain all the variables of data_vars. "
                f"The variables available are '{list(raster.data_vars)}'."
            )

        if ref_var in data_vars:
            raise ValueError('ref_var must not be an element of data_vars.')

    else:
        data_vars = list(raster.data_vars)
        data_vars.remove(ref_var)

    iter_list = []

    for comb in np.nditer([raster[var].data for var in data_vars]):
        iter_list.append(list(items.item() for items in comb))

    out = []
    ref_list = [item for arr in raster[ref_var].data for item in arr]

    for ref, comb in zip(ref_list, iter_list):
        comb_ref = ref - 1
        comb.sort()

        if np.isnan(comb).any() or comb_ref >= len(comb):
            out.append(np.nan)
            continue

        out.append(comb[comb_ref])

    final_arr = np.array(out)
    final_arr = np.reshape(final_arr, (-1, raster[data_vars[0]].data.shape[1]))
    final_arr = xr.DataArray(final_arr)

    return final_arr
