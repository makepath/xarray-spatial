import numpy as np
import xarray as xr

from xrspatial import equal_interval
from xrspatial import natural_breaks
from xrspatial import quantile


def test_quantile():
    k = 5
    n, m = 5, 5
    agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['x', 'y'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    quantile_agg = quantile(agg, k=5)
    assert quantile_agg is not None

    unique_elements, counts_elements = np.unique(quantile_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k
    assert len(np.unique(counts_elements)) == 1


def test_natural_breaks():
    k = 5
    n, m = 5, 5
    agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['x', 'y'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    natural_breaks_agg = natural_breaks(agg, k=5, init=10)
    assert natural_breaks_agg is not None

    unique_elements, counts_elements = np.unique(natural_breaks_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k


def test_small_natural_breaks():
    k = 5
    n, m = 4, 3
    agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['x', 'y'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    natural_breaks_agg = natural_breaks(agg, k=5, init=10)
    assert natural_breaks_agg is not None

    unique_elements, counts_elements = np.unique(natural_breaks_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k


def test_equal_interval():
    k = 4
    n, m = 4, 4
    agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['x', 'y'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    equal_interval_agg = equal_interval(agg, k=4)
    assert equal_interval_agg is not None

    unique_elements, counts_elements = np.unique(equal_interval_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k
    assert len(np.unique(counts_elements)) == 1


def test_small_equal_interval():
    k = 4
    n, m = 3, 2
    agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['x', 'y'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    equal_interval_agg = equal_interval(agg, k=4)
    assert equal_interval_agg is not None

    unique_elements, counts_elements = np.unique(equal_interval_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k
    assert len(np.unique(counts_elements)) == n-1
