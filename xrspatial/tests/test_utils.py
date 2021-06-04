from xrspatial.datasets import make_terrain
from xrspatial.utils import resample


def test_resample():
    terrain = make_terrain()
    terrain_res = resample(terrain, height=50, width=50)
    assert terrain_res.shape == (50, 50)
    assert terrain_res.chunks == ((50,), (50,))
