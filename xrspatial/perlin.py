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

    Parameters
    ----------
    width : int
        Width of output aggregate array.
    height : int
        Height of output aggregate array.
    freq : tuple, default=(1,1)
        (x, y) frequency multipliers.
    seed : int, default=5
        Seed for random number generator.

    Returns
    -------
    perlin_agg : xarray.DataArray
        2D array of perlin noise values.

    References
    ----------
        - Paul Panzer: https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy # noqa
        - ICA: http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        from xrspatial import perlin

        # Generate Perlin Noise Aggregate
        perlin_default = perlin(width = 500, height = 300)

        # With Increased x Frequency
        perlin_high_x_freq = perlin(width = 500, height = 300, freq = (5, 1))

        # With Increased y Frequency
        perlin_high_y_freq = perlin(width = 500, height = 300, freq = (1, 5))

        # With a Different Seed
        perlin_seed_1 = perlin(width = 500, height = 300, seed = 1)

        # Plot Default Perlin
        perlin_default.plot(cmap = 'inferno', aspect = 2, size = 4)
        plt.title("Default")

        # Plot High x Frequency
        perlin_high_x_freq.plot(cmap = 'inferno', aspect = 2, size = 4)
        plt.title("High x Frequency")

        # Plot High y Frequency
        perlin_high_y_freq.plot(cmap = 'inferno', aspect = 2, size = 4)
        plt.title("High y Frequency")

        # Plot Seed = 1
        perlin_seed_1.plot(cmap = 'inferno', aspect = 2, size = 4)
        plt.title("Seed = 1")

    .. sourcecode:: python

        >>> print(perlin_default[200:203, 200: 202])
        <xarray.DataArray (y: 3, x: 2)>
        array([[0.56800979, 0.56477393],
               [0.56651744, 0.56331014],
               [0.56499184, 0.56181344]])
        Dimensions without coordinates: y, x
        Attributes:
            res:      1

        >>> print(perlin_high_x_freq[200:203, 200: 202])
        <xarray.DataArray (y: 3, x: 2)>
        array([[0.5       , 0.48999444],
               [0.5       , 0.48999434],
               [0.5       , 0.48999425]])
        Dimensions without coordinates: y, x
        Attributes:
            res:      1

        >>> print(perlin_high_y_freq[200:203, 200: 202])
        <xarray.DataArray (y: 3, x: 2)>
        array([[0.31872961, 0.31756859],
               [0.2999256 , 0.2988189 ],
               [0.28085118, 0.27979834]])
        Dimensions without coordinates: y, x
        Attributes:
            res:      1

        >>> print(perlin_seed_1[200:203, 200: 202])
        <xarray.DataArray (y: 3, x: 2)>
        array([[0.12991498, 0.12984185],
               [0.13451158, 0.13441514],
               [0.13916956, 0.1390495 ]])
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
