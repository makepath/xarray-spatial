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


def equal_frequency(val_agg, agg_list):
    out = []
    in_aggs = [val_agg]
    for agg in agg_list:
        in_aggs.append(agg)

    # Iterate through each array simultaneously
    for v, a, b, c in np.nditer(in_aggs):
        count = 0
        if np.isnan((a, b, c)).any():   # skip nan
            out.append(np.nan)
            continue
        if v == a:
            count += 1
        if v == b:
            count += 1
        if v == c:
            count += 1
        out.append(count)

    # create new array
    out = np.array(out)
    out = np.reshape(out, (-1, agg_list[0].shape[1]))
    out = xr.DataArray(out)
    return out


def greater_frequency(val_agg, agg_list):
    out = []
    in_aggs = [val_agg]
    for agg in agg_list:
        in_aggs.append(agg)

    # Iterate through each array simultaneously
    for v, a, b, c in np.nditer(in_aggs):
        count = 0
        if np.isnan((a, b, c)).any():   # skip nan
            out.append(np.nan)
            continue
        if v < a:
            count += 1
        if v < b:
            count += 1
        if v < c:
            count += 1
        out.append(count)

    # create new array
    out = np.array(out)
    out = np.reshape(out, (-1, agg_list[0].shape[1]))
    out = xr.DataArray(out)
    return out


def highest_array(array1, array2, array3):
    out = []

    # Iterate through each array simultaneously
    for a, b, c in np.nditer([array1.data, array2.data, array3.data]):
        combo = (a.item(), b.item(), c.item())
        if np.isnan(combo).any():   # skip nan
            out.append(np.nan)
            continue

        max_value = max(combo)
        max_index = combo.index(max_value) + 1

        out.append(max_index)

    # create new array
    out = np.array(out)
    out = np.reshape(out, (-1, array1.shape[1]))
    out = xr.DataArray(out)

    return out


def lesser_frequency(val_agg, agg_list):
    out = []
    in_aggs = [val_agg]
    for agg in agg_list:
        in_aggs.append(agg)

    # Iterate through each array simultaneously
    for v, a, b, c in np.nditer(in_aggs):
        count = 0
        if np.isnan((a, b, c)).any():   # skip nan
            out.append(np.nan)
            continue
        if v > a:
            count += 1
        if v > b:
            count += 1
        if v > c:
            count += 1
        out.append(count)

    # create new array
    out = np.array(out)
    out = np.reshape(out, (-1, agg_list[0].shape[1]))
    out = xr.DataArray(out)
    return out


def lowest_array(array1, array2, array3):
    out = []

    # Iterate through each array simultaneously
    for a, b, c in np.nditer([array1.data, array2.data, array3.data]):
        combo = (a.item(), b.item(), c.item())
        if np.isnan(combo).any():   # skip nan
            out.append(np.nan)
            continue

        min_value = min(combo)
        min_index = combo.index(min_value) + 1

        out.append(min_index)

    # create new array
    out = np.array(out)
    out = np.reshape(out, (-1, array1.shape[1]))
    out = xr.DataArray(out)

    return out


def popularity(pop_agg, agg_list):
    out = []
    in_aggs = [pop_agg]
    for agg in agg_list:
        in_aggs.append(agg)

    for p, a, b, c in np.nditer(in_aggs):
        if np.isnan((a, b, c)).any():   # skip nan
            out.append(np.nan)
            continue

        inputs = np.array([a, b, c])

        count_a = np.count_nonzero(inputs == a)
        count_b = np.count_nonzero(inputs == b)
        count_c = np.count_nonzero(inputs == c)
        counts = np.array([count_a, count_b, count_c])

        countsI = counts.argsort()
        sorted_inputs = inputs[countsI][::-1]
        sorted_counts = counts[countsI][::-1]

        first = 0
        second = 0
        third = 0

        if sorted_counts[0] == 1:
            out.append(np.nan)
            continue
        elif sorted_counts[0] == 2:
            first = sorted_inputs[0]
            second = sorted_inputs[2]
            third = np.nan
        elif sorted_counts[0] == 3:
            first = sorted_inputs[0]
            second = sorted_inputs[1]
            third = sorted_inputs[2]

        if p == 1:
            out.append(first)
        elif p == 2:
            out.append(second)
        elif p == 3:
            out.append(third)

    # create new array
    out = np.array(out)
    out = np.reshape(out, (-1, agg_list[0].shape[1]))
    out = xr.DataArray(out)

    return out


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
