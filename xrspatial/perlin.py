import numpy as np

import xarray as xr
from xarray import DataArray

from xrspatial.utils import ngjit


# TODO: change parameters to take agg instead of height / width
def perlin(width: int,
           height: int,
           freq: tuple = (1, 1),
           seed: int = 5) -> xr.DataArray:
    """
    Generate perlin noise aggregate.

    Parameters:
    ----------
    width: int
        Width of output aggregate array.
    height: int
        Height of output aggregate array.
    freq: tuple (default = (1,1))
        (x, y) frequency multipliers.
    seed: int (default = 5)
        Seed for random number generator.

    Returns:
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated perlin
    noise values.

    Notes:
    ----------
    Algorithm References:
        numba-ized from Paul Panzer example available here:
        https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
        http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import perlin

    Generate Perlin Aggregate
    >>> print(perlin(5, 5))
    <xarray.DataArray (y: 5, x: 5)>
    array([[0.38502038, 0.3235394 , 0.13230299, 0.02275815, 0.13502038],
           [0.69650136, 0.6169794 , 0.34002832, 0.10065245, 0.10962136],
           [1.        , 0.90777853, 0.55206348, 0.16047902, 0.01192   ],
           [1.        , 0.92388163, 0.60956174, 0.21797728, 0.        ],
           [0.69650136, 0.6604194 , 0.50362745, 0.29227467, 0.15306136]])
    Dimensions without coordinates: y, x
    Attributes:
        res:      1
    """

    linx = range(width)
    liny = range(height)
    linx = np.linspace(0, 1, width, endpoint=False)
    liny = np.linspace(0, 1, height, endpoint=False)
    x, y = np.meshgrid(linx, liny)
    data = _perlin(x * freq[0], y * freq[1], seed=seed)
    data = (data - np.min(data))/np.ptp(data)
    return DataArray(data, dims=['y', 'x'], attrs=dict(res=1))


@ngjit
def _lerp(a, b, x):
    return a + x * (b-a)


@ngjit
def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


@ngjit
def _gradient(h, x, y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    dim_ = h.shape
    out = np.zeros(dim_)
    for j in range(dim_[1]):
        for i in range(dim_[0]):
            f = np.mod(h[i, j], 4)
            g = vectors[f]
            out[i, j] = g[0] * x[i, j] + g[1] * y[i, j]
    return out


def _perlin(x, y, seed=0):
    np.random.seed(seed)
    p = np.arange(2**20, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)

    # internal coordinates
    xf = x - xi
    yf = y - yi

    # fade factors
    u = _fade(xf)
    v = _fade(yf)

    # noise components
    n00 = _gradient(p[p[xi]+yi], xf, yf)
    n01 = _gradient(p[p[xi]+yi+1], xf, yf-1)
    n11 = _gradient(p[p[xi+1]+yi+1], xf-1, yf-1)
    n10 = _gradient(p[p[xi+1]+yi], xf-1, yf)

    # combine noises
    x1 = _lerp(n00, n10, u)
    x2 = _lerp(n01, n11, u)
    a = _lerp(x1, x2, v)
    return a
