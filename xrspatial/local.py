import numpy as np
import xarray as xr


def combine_arrays(array1, array2, array3):
    unique_combos = {}
    unique_values = {}
    all_combos = []
    all_values = []
    value = 1

    # Iterate through each array simultaneously
    for a, b, c in np.nditer([array1.data, array2.data, array3.data]):
        combo = (a.item(), b.item(), c.item())
        if np.isnan(combo).any():   # skip nan
            all_values.append(np.nan)
            all_combos.append('NAN')
            continue
        if combo in unique_combos.keys():   # apply 0 combos already found
            all_combos.append(combo)
            all_values.append(0)
        else:                               # apply new value to unique combos
            unique_combos[combo] = value
            unique_values[value] = combo
            all_combos.append(combo)
            all_values.append(value)
            value += 1

    # apply new value to matching combos
    k = 0
    for value in all_values:
        if value == 0:
            combo = all_combos[k]
            all_values[k] = [unique_combos[combo]][0]
        k += 1

    # create new array
    new_array = np.array(all_values)
    new_array = np.reshape(new_array, (-1, array1.shape[1]))

    out = xr.DataArray(
        data=new_array,
        attrs=dict(key=unique_values)
    )

    return out


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
