import numpy as np
import xarray as xr

from xrspatial.analytics import summarize_terrain


def test_summarize_terrain():

    data = np.array([[10., 10., 10., 10.],
                    [0., 0., 0., 0.],
                    [10., 10., 10., 10.],
                    [20., 20., 20., 20.],
                    [20., 20., 20., 20.],
                    [30., 30., 30., 30.]])
    test_terrain = xr.DataArray(data,
                                dims=['y', 'x'],
                                coords={'y': np.arange(0, 6),
                                        'x': np.arange(0, 4)},
                                name='myterrain')
    summarized_ds = summarize_terrain(test_terrain)
    variables = [v for v in summarized_ds]
    should_have = ['myterrain',
                   'myterrain-slope',
                   'myterrain-curvature',
                   'myterrain-aspect']
    assert variables == should_have
