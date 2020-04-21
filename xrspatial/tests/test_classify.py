import xarray as xr
import numpy as np

from xrspatial.classify import quantile


def test_quantile():
    k = 5
    n, m = 5, 5
    agg = xr.DataArray(np.arange(n*m).reshape((n, m)), dims=['x', 'y'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    quantile_agg = quantile(agg, k=5)
    assert quantile_agg is not None

    print(quantile_agg)
    print(quantile_agg.mean())

    unique_elements, counts_elements = np.unique(quantile_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k
    assert len(np.unique(counts_elements)) == 1
