import pytest
import numpy as np
import xarray as xr

from xrspatial import aspect
from xrspatial import curvature
from xrspatial import slope
from xrspatial.analytics import summarize_terrain


@pytest.fixture
def test_terrain():
    # TODO: use function random_data() from global pytest fixture in xrspatial.tests.conftest
    #  once it gets merged
    data = np.array([
        [10., 10., 10., 10.],
        [0., 0., 0., 0.],
        [10., 10., 10., 10.],
        [20., 20., 20., 20.],
        [20., 20., np.nan, 20.],
        [30., 30., 30., 30.]
    ])
    terrain = xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.arange(0, 6),
                'x': np.arange(0, 4)},
        name='myterrain'
    )
    return terrain


def test_summarize_terrain(test_terrain):
    summarized_ds = summarize_terrain(test_terrain)
    variables = [v for v in summarized_ds]
    should_have = ['myterrain',
                   'myterrain-slope',
                   'myterrain-curvature',
                   'myterrain-aspect']
    assert variables == should_have
    np.testing.assert_allclose(summarized_ds['myterrain-slope'], slope(test_terrain))
    np.testing.assert_allclose(summarized_ds['myterrain-curvature'], curvature(test_terrain))
    np.testing.assert_allclose(summarized_ds['myterrain-aspect'], aspect(test_terrain))
