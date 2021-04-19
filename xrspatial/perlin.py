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
    freq : tuple, default = (1,1)
        (x, y) frequency multipliers.
    seed : int, default = 5
        Seed for random number generator.
    Returns
    -------
    perlin_agg : xarray.DataArray
        2D array of perlin noise values.

    Notes
    -----
    Algorithm References
        numba-ized from Paul Panzer example available here:
        - https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
        - http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf

    Example
    -------
    >>>     from datashader.transfer_functions import shade
    >>>     from xrspatial import perlin

    >>>     # Default Noise Aggregate
    >>>     perlin_default = perlin(width = 500, height = 300)
    >>>     print(perlin_default[250:256, 150:154])
    ...     <xarray.DataArray (y: 6, x: 4)>
    ...     array([[0.5398953 , 0.53867152, 0.53743887, 0.53619742],
    ...            [0.53621057, 0.53501437, 0.53380949, 0.53259599],
    ...            [0.53253872, 0.53137001, 0.5301928 , 0.52900716],
    ...            [0.52888066, 0.52773933, 0.52658969, 0.5254318 ],
    ...            [0.52523726, 0.5241232 , 0.52300102, 0.52187079],
    ...            [0.52160935, 0.52052245, 0.51942763, 0.51832494]])
    ...     Dimensions without coordinates: y, x
    ...     Attributes:
    ...         res:      1

    >>>     # Shade Image
    >>>     perlin_default_img = shade(perlin_default)
    >>>     perlin_default_img

            .. image :: ./docs/source/_static/img/docstring/perlin_example.png

    >>>     # Increased x Frequency
    >>>     perlin_high_x_freq = perlin(width = 500,
    >>>                                 height = 300,
    >>>                                 freq = (5, 1),
    >>>                                 seed = 3)
    >>>     print(perlin_high_x_freq[250:256, 150:154])
    ...     <xarray.DataArray (y: 6, x: 4)>
    ...     array([[0.03455066, 0.03698676, 0.0398157 , 0.04302266],
    ...            [0.03599555, 0.03846388, 0.04132664, 0.04456897],
    ...            [0.03746666, 0.03996852, 0.04286637, 0.04614528],
    ...            [0.0389632 , 0.04149989, 0.04443408, 0.04775076],
    ...            [0.04048437, 0.0430572 , 0.04602897, 0.04938461],
    ...            [0.04202938, 0.04463964, 0.04765022, 0.05104599]])
    ...     Dimensions without coordinates: y, x
    ...     Attributes:
    ...         res:      1

    >>>     # Shade Image
    >>>     perlin_high_x_freq_img = shade(perlin_high_x_freq)
    >>>     perlin_high_x_freq_img

            .. image :: ./docs/source/_static/img/docstring/perlin_example_x.png

    >>>     # Increased y Frequency
    >>>     perlin_high_y_freq = perlin(width = 500,
    >>>                                 height = 300,
    >>>                                 freq = (1, 5),
    >>>                                 seed = 3)
    >>>     print(perlin_high_y_freq[250:256, 150:154])
    ...     <xarray.DataArray (y: 6, x: 4)>
    ...     array([[0.16069496, 0.15846582, 0.15624912, 0.15404513],
    ...            [0.16830904, 0.16612992, 0.16396296, 0.16180843],
    ...            [0.17707647, 0.17495495, 0.17284527, 0.17074769],
    ...            [0.18701164, 0.18495538, 0.18291061, 0.18087757],
    ...            [0.19811672, 0.19613343, 0.19416121, 0.19220029],
    ...            [0.21038244, 0.20847973, 0.20658764, 0.2047064 ]])
    ...     Dimensions without coordinates: y, x
    ...     Attributes:
    ...         res:      1

    >>>     # Shade Image
    >>>     perlin_high_y_freq_img = shade(perlin_high_y_freq)
    >>>     perlin_high_y_freq_img

            .. image :: ./docs/source/_static/img/docstring/perlin_example_y.png

    >>>     # Different Seed
    >>>     perlin_seed_1 = perlin(width = 500, height = 300, seed = 1)
    >>>     print(perlin_seed_1[250:256, 150:154])
    ...     <xarray.DataArray (y: 6, x: 4)>
    ...     array([[0.48914834, 0.48723859, 0.48532976, 0.4834222 ],
    ...            [0.49598854, 0.49405708, 0.49212637, 0.49019678],
    ...            [0.50280309, 0.50085   , 0.49889751, 0.49694596],
    ...            [0.50959052, 0.50761589, 0.5056417 , 0.50366828],
    ...            [0.51634941, 0.51435333, 0.51235753, 0.51036234],
    ...            [0.52307839, 0.52106096, 0.51904364, 0.51702677]])
    ...     Dimensions without coordinates: y, x
    ...     Attributes:
    ...         res:      1

    >>>     # Shade Image
    >>>     perlin_seed_1_img = shade(perlin_seed_1)
    >>>     perlin_seed_1_img

            .. image :: ./docs/source/_static/img/docstring/perlin_example_seed.png

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
